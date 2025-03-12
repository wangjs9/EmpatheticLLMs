# this program will generate dialogue between the simulated user and the empathetic LLM
import yaml
import argparse
import random
import logging
from tqdm import trange
from typing import Dict, Any, Optional
import torch
from vllm import LLM
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
import multiprocessing
from llamafactory.hparams import get_eval_args
from llamafactory.model import load_tokenizer
from llamafactory.data import get_template_and_fix_tokenizer, Role
from utils.config_utils import *

random.seed(42)
logging.getLogger().setLevel(logging.INFO)


def clean_listener(o):
    text = re.sub(r"【倾听.*?回复】", "【倾听者回复】", o.text)
    text = re.sub(r"【倾听者回复】:", "【倾听者回复】：", text)
    text_list = text.split('\n\n')
    while '【倾听者回复】：' not in text_list[-1] or '【倾听者回复】：' == text_list[-1]:
        text_list.pop(-1)
        if len(text_list) == 0:
            return ''
    return '\n\n'.join(text_list)


def post_generate(text, spliter):
    if 'EOC' in text or '对话可以结束' in text:
        return spliter + 'EOC'
    sentences = text.split('\n\n')
    reply = sentences[-1]
    if reply.startswith('【') and '】：' in reply:
        sentences[-1] = spliter + reply.split('】：')[-1]
        return '\n\n'.join(sentences)
    elif '：' in reply:
        sentences[-1] = spliter + reply.split('：')[-1]
        return '\n\n'.join(sentences)
    else:
        return spliter + 'EOC. Failed generation.'


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
        # Implement the reward calculation, the reward include two parts:
        # 1. the similarity between the predicted user_state and the simulator's user_state
        # 2. whether the user is engaged: the user expresses bye-bye while all feelings and needs are satisfied
        assert self.num_turn > 0 and self.num_turn % 2 == 0
        # whether the conversation ends
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


def clean_listener(o):
    text = re.sub(r"【倾听.*?回复】", "【倾听者回复】", o.text)
    text = re.sub(r"【倾听者回复】:", "【倾听者回复】：", text)
    if "【倾听者回复】：" not in text:
        logging.info(f"invalid text: {text}")
        return ''
    text_list = text.split('【倾听者回复】：')
    text_list[-1] = text_list[-1].replace("\n\n", " ")
    return '【倾听者回复】：'.join(text_list)


def clean_simulator(o):
    text = re.sub(r"【来访.*?对话】", "【来访者对话】", o.text)
    text = re.sub(r"【来访者对话】:", "【来访者对话】：", text)
    if "【来访者对话】：" not in text:
        logging.info(f"invalid text: {text}")
        return ''
    text_list = text.split('【来访者对话】：')
    text_list[-1] = text_list[-1].replace("\n\n", " ")
    return '【来访者对话】：'.join(text_list)


class BatchedSimulatedConv:
    def __init__(
            self, model, tokenizer, templates, user_descs: list = None, max_turn: int = 40, expansion: int = 3,
            max_trajectory: int = 81, lora_path: Dict[str, str] = None
    ):
        self.model = model
        self.expansion = expansion
        self.max_trajectory = max_trajectory
        self.tokenizer = tokenizer
        self.user_template, self.llm_template, self.template = templates
        self.bsize = len(user_descs)
        self.max_turn = max_turn
        self.lora_path = lora_path
        self.generation_kwargs = {
            "default": {"max_tokens": 512, "top_k": 50, "top_p": 0.9, "n": self.expansion, "temperature": 0.9,
                        "frequency_penalty": 1.0, "repetition_penalty": 1.0},
            "simulator": {"max_tokens": 512, "top_k": 50, "top_p": 0.9, "n": 1, "temperature": 0.3,
                          "frequency_penalty": 1.0, "repetition_penalty": 1.0}
        }
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

    def vllm_logit(self, batch_messages):
        """this function is used to compute the reward score"""
        outputs = self.model(
            prompt_token_ids=batch_messages,
            lora_request=LoRARequest("reward", 4, lora_path=self.lora_path["reward"]),
        )
        logits = outputs.logits


    def vllm_generate(self, batch_messages, role='default'):
        outputs = self.model.generate(
            prompt_token_ids=batch_messages,
            sampling_params=SamplingParams(seed=random.randint(0, 4096), **self.generation_kwargs[role]),
            lora_request=LoRARequest(role, 1 if role == 'simulator' else 2, lora_path=self.lora_path[role]),
        )
        if role == 'default':
            return [[clean_listener(o) for o in output.outputs] for output in outputs]
            # return sum([[list(o.token_ids) for o in output.outputs] for output in outputs], [])
        else:
            return [clean_simulator(output.outputs[0]) for output in outputs]
            # return [list(output.outputs[0].token_ids) for output in outputs]

    def post_generate(self, text, spliter):
        if 'EOC' in text or '对话可以结束' in text:
            return spliter + 'EOC'
        sentences = text.split('\n\n')
        reply = sentences[-1]
        if reply.startswith('【') and '】：' in reply:
            sentences[-1] = spliter + reply.split('】：')[-1]
            return '\n\n'.join(sentences)
        elif '：' in reply:
            sentences[-1] = spliter + reply.split('：')[-1]
            return '\n\n'.join(sentences)
        else:
            return spliter + 'EOC. Failed generation.'

    def __respond__(
            self, input_messages: List[List[Dict[str, str]]], role: str = Literal["default", "simulator"], retry=10
    ) -> Tuple[List[str], List[str]]:
        # self.model.set_adapter(role)
        spliter = '【来访者对话】：' if role == 'simulator' else '【倾听者回复】：'

        message_num = len(input_messages)
        messages_ids = [self.template.encode_oneturn(tokenizer=self.tokenizer, messages=m)[0] for m in input_messages]

        model_outputs = [[''] * self.expansion for _ in range(message_num)] if role == 'default' else [''] * message_num
        decode_ids = list(range(message_num))

        for _ in range(retry):
            process_messages = [messages_ids[i] for i in decode_ids]
            gen_texts = self.vllm_generate(process_messages, role)
            # gen_texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            if role == 'simulator':
                for j, index in enumerate(decode_ids): model_outputs[index] = gen_texts[j]
                decode_ids = [i for i, tt in enumerate(model_outputs) if spliter not in tt]
            else:
                # gen_texts = [gen_texts[i: i + self.expansion] for i in range(0, len(gen_texts), self.expansion)]
                for j, index in enumerate(decode_ids):
                    for k in range(self.expansion):
                        if spliter not in model_outputs[index][k]:
                            model_outputs[index][k] = gen_texts[j][k]
                decode_ids = [i for i, tts in enumerate(model_outputs) if not all([spliter in tt for tt in tts])]

            if len(decode_ids) == 0:
                if role == 'default':
                    model_outputs = sum(model_outputs, [])
                break

            logging.info('Decoding failed, retrying...')
            # logging.info([model_outputs[i] for i in decode_ids])

        if len(decode_ids) > 0:
            # use GPT to generate the response
            logging.info(f'Decoding failed after {retry} attempts.')
            if role == 'default':
                for i in decode_ids:
                    for j in range(self.expansion):
                        model_outputs[i][j] = self.post_generate(model_outputs[i][j], spliter)
                model_outputs = sum(model_outputs, [])
            else:
                for i in decode_ids:
                    model_outputs[i] = self.post_generate(model_outputs[i], spliter)

        model_outputs = [output.split(spliter) for output in model_outputs]
        user_states = [output[0] for output in model_outputs]
        replies = [output[-1] for output in model_outputs]
        # replies = [output[-1].strip(punctuation) for output in model_outputs]
        replies = [reply.split('\n')[0] for reply in replies]
        # print("\n".join(replies) + "\n\n")
        return user_states, replies

    def save(self, save_dir, index: int):
        # save the conversations in json files
        for idx, convs in enumerate(self.conv_list):
            if len(convs) > 0:
                user_desc = convs[0].user_desc
            elif len(self.completed[idx]) > 0:
                user_desc = self.completed[idx][0].user_desc
            else:
                assert len(self.discard[idx]) > 0
                user_desc = self.discard[idx][0].user_desc
            logging.info(f"time: {str(time.time())} the saved index is: ", index + idx)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, f"conversations_{index + idx}.json"), "w", encoding='utf-8') as fp:
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

    def step(self, step_num: int):
        role = "default" if step_num % 2 == 0 else "simulator"
        conv_num = list(map(lambda conv: len(conv), self.conv_list))
        self.conv_list = sum(self.conv_list, [])
        if role == 'default':
            conv_num = list(map(lambda num: num * self.expansion, conv_num))

            format_fun = self.llm_template.format_example
        else:
            format_fun = self.user_template.format_example
        dialogues = list(map(lambda conv: conv.dialogue, self.conv_list))
        descriptions = list(map(lambda conv: conv.user_desc, self.conv_list))
        model_inputs = [
            format_fun({'description': desc, 'conversation': dial}) for desc, dial in zip(descriptions, dialogues)
        ]

        user_states, replies = self.__respond__(model_inputs, role=role)
        if role == 'default':
            self.conv_list = [conv.copy() for conv in self.conv_list for _ in range(self.expansion)]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            jobs = [executor.submit(conv.step, s, r, role) for conv, s, r in zip(self.conv_list, user_states, replies)]
            progress = [job.result() for job in jobs]
        it = iter(self.conv_list)
        self.conv_list = [list(islice(it, num)) for num in conv_num]
        if role == 'default' and self.max_turn > step_num:
            for idx, convs in enumerate(self.conv_list):
                # Partition the conversations into discarded and retained lists
                self.conv_list[idx] = [conv for conv in convs if (not conv.done) and (not conv.completed)]
                self.discard[idx].extend([conv for conv in convs if (conv.done and not conv.completed)])
                self.completed[idx].extend([conv for conv in convs if conv.completed])
            # I need to select the first max_trajectory conversations according to the rewards
            # if len(conv_list) * self.expansion <= self.max_trajectory * self.bsize:
            for idx, convs in enumerate(self.conv_list):
                if len(convs) > self.max_trajectory:
                    sorted_convs = sorted(convs, key=lambda conv: conv.weighted_reward, reverse=True)
                    self.conv_list[idx] = sorted_convs[:self.max_trajectory]
                    self.discard[idx].extend(sorted_convs[self.max_trajectory:])

        return progress
