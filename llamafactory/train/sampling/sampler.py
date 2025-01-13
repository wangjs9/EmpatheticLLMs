import os
import time
from typing import Dict, List, Tuple, Literal
import torch
from tqdm import trange
from itertools import islice
import concurrent.futures
from utils.config_utils import *
import string
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import cosine_similarity

punctuation = '，。！？【】（）（）<>“”‘’：；、|《》' + string.punctuation
END_OF_CONV = '<<<<<<END_OF_CONVERSATION>>>>>>'

model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Example model
simi_tokenizer = AutoTokenizer.from_pretrained(model_name)
simi_model = AutoModel.from_pretrained(model_name)


def get_embeddings(text):
    # Tokenize input text
    inputs = simi_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Forward pass to get model output
    with torch.no_grad():
        outputs = simi_model(**inputs)
    # Get the embeddings (e.g., use the CLS token or mean pooling)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings


class SimulatedConv:
    def __init__(
            self, user_desc: str, dialogue: List[Dict[str, str]] = None, all_user_states: List[str] = None,
            all_predicted_states: List[str] = None, max_turn: int = 40, done: bool = False, completed: bool = False,
            step_rewards: List[float] = None
    ):
        self.user_desc = user_desc
        self.dialogue = dialogue if dialogue else []
        self.all_user_states = all_user_states if all_user_states else []
        self.all_predicted_states = all_predicted_states if all_predicted_states else []
        self.max_turn = max_turn
        self.num_turn = len(self.dialogue)
        self.done = done
        self.completed = completed
        self.step_rewards = step_rewards if step_rewards else []
    
    @property
    def weighted_reward(self) -> float:
        if self.completed:
            return 100
        else:
            return sum(self.step_rewards) / len(self.step_rewards)
    
    def question_rate(self, expected_rate=0.4):
        # calculate the question rate
        responses = [r['content'] for r in self.dialogue if r['role'] == 'assistant']
        response_num = len(responses)
        question_num = sum([1 for r in responses if '?' in r])
        return abs(question_num / response_num - expected_rate) if response_num > 0 else 0
    
    def _step_reward(self, user_state: str) -> float:
        # Implement the reward calculation, the reward include tow parts:
        # 1. the similarity between the predicted user_state and the simulator's user_state
        # 2. whether the user is engaged: the user expresses bye-bye while all feelings and needs are satisfied
        assert self.num_turn > 0 and self.num_turn % 2 == 0
        # whether the conversation is end
        done = False
        if self.num_turn > 30 and 'EOC' in self.dialogue[-1]['content']:
            done = True
            self.completed = True
        elif 'EOC' in self.dialogue[-1]['content']:
            done = True
        self.done = done or self.num_turn == self.max_turn
        # compute the step reward
        # get the user state from the llm prediction
        reward = 0.2 * self.question_rate()
        if self.all_user_states[-1] != '':
            match = re.search(r'【倾听者思维链】：我对来访者有如下判断：(.*?)在接下来的回复中', user_state, re.DOTALL)
            if match:
                prediction = match.group(1).strip()
                predicted_state_emb = get_embeddings(prediction)
                user_state_emb = get_embeddings(self.all_user_states[-1])
                reward += 0.8 * cosine_similarity(predicted_state_emb, user_state_emb).item()
        
        return reward
    
    def step(self, user_state, response, role: str = Literal["default", "simulator"]):
        # keep the one that the LLM predicted user_state is similar to the simulator's user_state
        # keep 4 replies that are most different from each others.
        if self.done:
            return True
        self.num_turn += 1
        if role == 'simulator':
            self.dialogue.append({'role': 'user', 'content': response})
            self.all_user_states.append(user_state)
        else:
            reward = self._step_reward(user_state)
            self.dialogue.append({'role': 'assistant', 'content': response})
            self.all_predicted_states.append(user_state)
            self.step_rewards.append(reward)
        
        return self.done
    
    def reset(self):
        self.num_turn = 0
        self.done = False
    
    def copy(self):
        return SimulatedConv(
            self.user_desc, copy.deepcopy(self.dialogue), copy.deepcopy(self.all_user_states),
            copy.deepcopy(self.all_predicted_states), self.max_turn, self.done,
            self.completed, copy.deepcopy(self.step_rewards)
        )


class BatchedSimulatedConv:
    def __init__(
            self, model, tokenizer, templates, user_descs: list = None, max_turn: int = 40, expansion: int = 3,
            max_trajectory: int = 128
    ):
        self.model = model
        self.expansion = expansion
        self.max_trajectory = max_trajectory
        self.tokenizer = tokenizer
        self.user_template, self.llm_template, self.template = templates
        self.bsize = len(user_descs)
        self.max_turn = max_turn
        # generate the first user utterances
        conv_list = [SimulatedConv(desc, max_turn=max_turn) for desc in user_descs]
        model_inputs = [self.user_template.format_example({'description': desc}) for desc in user_descs]
        user_states, replies = self.__respond__(model_inputs, role="simulator")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for conv, s, r in zip(conv_list, user_states, replies):
                executor.submit(conv.step, s, r, "simulator")
        self.conv_list = [[conv] for conv in conv_list]
        # list of list [[a1, a2, a3], [b1, b2, b3], ...] where a1, a2, a3 are with the user desc
        self.completed = [[] for _ in range(self.bsize)]
        self.discard = [[] for _ in range(self.bsize)]
    
    @torch.inference_mode()
    def __respond__(
            self, input_messages: List[List[Dict[str, str]]], role: str = Literal["default", "simulator"]
    ) -> Tuple[List[str], List[str]]:
        self.model.set_adapter(role)
        user_states, replies = [], []
        bz = 16
        for i in trange(0, len(input_messages), bz, desc="Inference batches", position=1, leave=False):
            batch_messages = input_messages[i: i + bz]
            input_ids = [self.template.encode_oneturn(tokenizer=self.tokenizer, messages=m)[0] for m in batch_messages]
            batch_input = self.tokenizer.pad(
                [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in input_ids],
                return_attention_mask=True, return_tensors="pt").to(self.model.device)
            while True:
                output_ids = self.model.generate(
                    input_ids=batch_input['input_ids'], attention_mask=batch_input['attention_mask'],
                    max_new_tokens=512, num_return_sequences=1, temperature=1.0, top_k=50, top_p=0.95, do_sample=True
                )
                input_lengths = batch_input['input_ids'].size(-1)
                new_tokens = output_ids[:, input_lengths:].tolist()
                generated_texts = [self.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in new_tokens]
                # split the user_state and response
                spliter = '【来访者对话】：' if role == 'simulator' else '【倾听者回复】：'
                if not all(spliter in text for text in generated_texts):
                    print('Decoding again...')
                else:
                    model_outputs = [text.split(spliter) for text in generated_texts]
                    user_states.extend([output[0] for output in model_outputs])
                    replies.extend([output[1] for output in model_outputs])
                    break
        return user_states, replies
    
    def save(self, save_dir, index: int):
        # save the conversations in json files
        for idx, convs in enumerate(self.conv_list):
            print(f"time: {str(time.time())} the saved index is: ", index + idx)
            user_desc = convs[0].user_desc
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, f"conversations_{index + idx}.json"), "w") as fp:
                json.dump({
                    'index': index + idx,
                    'user_desc': user_desc,
                    'completed': [{
                        'dialogue': conv.dialogue,
                        'step_rewards': conv.step_rewards,
                        'user_state': conv.all_user_states,
                        'predicted_state': conv.all_predicted_states
                    } for conv in self.completed[idx]],
                    'trajectories': [{
                        'dialogue': conv.dialogue,
                        'step_rewards': conv.step_rewards,
                        'user_state': conv.all_user_states,
                        'predicted_state': conv.all_predicted_states
                    } for conv in convs],
                    'discard': [{
                        'dialogue': conv.dialogue,
                        'step_rewards': conv.step_rewards,
                        'user_state': conv.all_user_states,
                        'predicted_state': conv.all_predicted_states
                    } for conv in self.discard[idx]]
                }, fp, indent=2)
    
    def sort_convs(self, convs: List[SimulatedConv]) -> List[SimulatedConv]:
        # sort the conversations according to the rewards
        return sorted(convs, key=lambda conv: conv.weighted_reward, reverse=True)
    
    def step(self, step_num: int):
        role = "default" if step_num % 2 == 0 else "simulator"
        conv_num = list(map(lambda conv: len(conv), self.conv_list))
        self.conv_list = sum(self.conv_list, [])
        if role == 'default':
            conv_num = list(map(lambda num: num * self.expansion, conv_num))
            self.conv_list = [conv.copy() for conv in self.conv_list for _ in range(self.expansion)]
            format_fun = self.llm_template.format_example
        else:
            format_fun = self.user_template.format_example
        dialogues = list(map(lambda conv: conv.dialogue, self.conv_list))
        descriptions = list(map(lambda conv: conv.user_desc, self.conv_list))
        model_inputs = [format_fun({'description': desc, 'conversation': dial}) for desc, dial in
                        zip(descriptions, dialogues)]
        
        user_states, replies = self.__respond__(model_inputs, role=role)
        # progress = [conv.step(s, r, role) for conv, s, r in zip(self.conv_list, user_states, replies)]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(conv.step, s, r, role) for conv, s, r in zip(self.conv_list, user_states, replies)]
            progress = [job.result() for job in jobs]
        it = iter(self.conv_list)
        self.conv_list = [list(islice(it, num)) for num in conv_num]
        
        if role == 'default' and self.max_turn > step_num:
            for idx, convs in enumerate(self.conv_list):
                # Partition the conversations into discarded and retained lists
                self.discard[idx].extend([conv for conv in convs if (conv.done and not conv.completed)])
                self.conv_list[idx] = [conv for conv in convs if (not conv.done) and (not conv.completed)]
                self.completed[idx] = [conv for conv in convs if conv.completed]
            # I need to select the first max_trajectory conversations according to the rewards
            # if len(conv_list) * self.expansion <= self.max_trajectory * self.bsize:
            for idx, convs in enumerate(self.conv_list):
                if len(convs) > self.max_trajectory:
                    sorted_convs = self.sort_convs(convs)
                    self.conv_list[idx] = sorted_convs[:self.max_trajectory]
                    self.discard[idx].extend(sorted_convs[self.max_trajectory:])
        
        return progress
