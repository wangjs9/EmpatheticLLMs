import math
import os
import sys
import warnings
from types import MethodType
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Literal
import random
import torch
from accelerate.utils import DistributedDataParallelKwargs
from huggingface_hub import model_info
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override
import concurrent.futures
from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .sampling_utils import replace_model


class SimulatedConv():
    def __init__(self, user_description: str, dialogue: List[Dict[str, str]] = [],
                 user_state: List[Tuple[str, str]] = [], max_conv_len: int = 30):
        self.user_description = user_description
        self.dialogue = dialogue
        self.user_state = user_state
        self.max_conversation_length = max_conv_len
        self.random = random.Random(None)
        self.count = 0
        self.done = True
    
    def _reward(self, response):
        # TODO: Implement the reward calculation, the reward include tow parts:
        # the similarity between the predicted user_state and the simulator's user_state
        # whether the user is engaged: the user expresses bye-bye while all feelings and needs are satisfied
        done = False
        match = re.search(r'来访者心里隐含的(.*?)：', self.user_state[-1][0])
        if match and self.count > 1:
            target_string = match.group(1).strip()
            if "感受" not in target_string and "需要" not in target_string:
                done = True
        self.done = done or self.count == self.max_conversation_length
        reward = 0 if done else -1
        state_similarity = ''
        match = re.search(r'【倾听者思维链】：我对来访者有如下判断：(.*?)在接下来的回复中', self.user_state[-1][1],
                          re.DOTALL)
        if match:
            target_string = match.group(1).strip()  # 提取并去除多余空格
            print(target_string)
    
    def _step(self, user_state, response, role: str = Literal["default", "simulator"]):
        # keep the one that the LLM predicted user_state is similar to the simulator's user_state
        # keep 4 responses that are most different from each others.
        self.count += 1
        if role == 'simulator':
            self.dialogue.append({'role': 'user', 'text': response})
            self.user_state.append((user_state, ''))
            return None
        else:
            self.dialogue.append({'role': 'assistant', 'text': response})
            self.user_state[-1] = (self.user_state[-1][0], user_state)
            reward = 0
    
    def reset(self, idx: Optional[int] = None):
        self.count = 0
        self.done = False
    
    def copy(self):
        return SimulatedConv(self.user_description, self.dialogue, self.max_conversation_length)


class BatchedSimulatedConv():
    def __init__(self, tokenizer, user_desc_list: list, templates, max_conv_len: int = 30, beam_size: int = 5):
        self.conv_list = sum(
            [[SimulatedConv(user_desc, max_conv_len=max_conv_len) for _ in range(beam_size)] for user_desc
             in user_desc_list], [])
        self.bsize = len(user_desc_list) * beam_size
        self.tokenizer = tokenizer
        self.simulator_template, self.llm_template, self.default_template = templates
    
    @torch.inference_mode()
    def __respond__(self, model, input_messages: List[List[Dict[str, str]]]) -> List[str]:
        input_ids = [self.default_template.encode_oneturn(tokenizer=self.tokenizer, messages=m)[0] for m in
                     input_messages]
        inputs = self.tokenizer.pad(
            [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in input_ids],
            return_attention_mask=True,
            return_tensors="pt"
        ).to(model.device)
        output_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=256,  # Maximum length of the generated text
            num_return_sequences=1,  # Number of sequences per input
            temperature=1.0,  # Sampling temperature
            top_k=50,  # Top-k sampling
            top_p=0.95,  # Top-p (nucleus sampling)
            do_sample=True,  # Enable sampling
        )
        input_lengths = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum(dim=1)  # Shape: [batch_size]
        new_tokens = []
        for i in range(len(input_lengths)):
            new_tokens.append(output_ids[i, input_lengths[i]:].tolist())
        
        # Decode the newly generated tokens
        generated_texts = [self.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in new_tokens]
        return generated_texts
    
    def generate_responses(self, model, target: str = Literal["default", "simulator"]):
        dialogues = [conv.dialogue for conv in self.conv_list]
        if target == 'simulator':
            descriptions = [conv.user_description for conv in self.conv_list]
            model_inputs = [self.simulator_template.format_example({'description': desc, 'conversation': dial}) for
                            desc, dial in zip(descriptions, dialogues)]
        else:
            model_inputs = [self.llm_template.format_example({'conversation': dial}) for dial in dialogues]
        model_outputs = self.__respond__(model, model_inputs)
        if target == 'simulator':
            model_outputs = [output.split('【来访者对话】：') for output in model_outputs]
        else:
            model_outputs = [output.split('【倾听者回复】：') for output in model_outputs]
        user_states = [output[0] for output in model_outputs]
        responses = [output[1] for output in model_outputs]
        return user_states, responses
    
    def reset(self, idx: Optional[int] = None):
        return [conv.reset(idx) for conv in self.conv_list]
    
    def step(self, model, role: str = Literal["default", "simulator"]):
        user_states, responses = self.generate_responses(model, target=role)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(conv._step, s, r, role) for conv, s, r in
                    zip(self.conv_list, user_states, responses)]
            results = [job.result() for job in jobs]
        return results


def Sampler(model, tokenizer, finetuning_args, templates, user_descriptions, batch_size: int = 4):
    for i in tqdm(range(0, len(user_descriptions), batch_size), position=0, desc="Sampling"):
        user_desc_batch = user_descriptions[i:i + batch_size]
        conv = BatchedSimulatedConv(tokenizer, templates, user_desc_batch)
        batch_done = [False, ] * len(user_desc_batch)
        step = 0
        while not all(batch_done):
            step += 1
            
            if finetuning_args.simulator_type == 'lora':
                replace_model(model, target="simulator")
                simulator = model
            else:
                simulator = None
                raise NotImplementedError("Only LORA simulator is supported.")
            # TODO: Implement the simulator's responses
            simulator_states, user_response = conv.step(simulator, role='simulator')
            
            if finetuning_args.reward_model_type == "lora":
                replace_model(model, target="default")
            
            utterances = conv.reset()
            responses = conv.step(model, role='default')
