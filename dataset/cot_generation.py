"""
nohup python -m cot_computation.cot_generation --use_gpt > cot_generation.log 2>&1 &
"""
import logging
import random
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import torch
import yaml
from datasets import load_dataset, Dataset
from fire import Fire
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import get_eval_args
from llamafactory.model import load_model, load_tokenizer
from tqdm import trange, tqdm
from utils.template_utils import get_template
from utils.api_utils import OpenAIChatBot
from utils.config_utils import *
from utils.message_utils import Message

logging.getLogger().setLevel(logging.INFO)
DEFAULT_LEN = 256


class CotGenerator:
    def __init__(self, save_dir=f"{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_cot", batch_size=128) -> None:
        self.end_points = list(END_POINTS)
        os.environ['END_POINT'] = self.end_points[-1]
        self.description = '任务：\n扮演一位与来访者对话的倾听者，描述倾听者在与来访者指定回复的思考过程，最终补充完整的思维链（……部分）。\n思维链内容包括：\n1. 倾听者对来访者状态的关注点（观察、情绪、需求或者请求），这个关注点直接影响倾听者的后续回复；\n2. 倾听者回复的策略（例如：建议、教育、安慰、回忆、否定、同情、询问等）和意图。\n\n要求：\n1. 视角：以倾听者的视角与口吻展开分析；\n2. 描述：详细说明倾听者回复背后的思维链；\n3. 思维过程：\n - 基于与来访者的对话历史作出推导；\n - 在推导过程中，倾听者不应预知或者提及后续回复的具体内容；\n - 通过思维链能够自然推导得出后续回复。'
        self.model = OpenAIChatBot(model=MODEL_PATH['gpt-4o'])
        self.generation_template = get_template('generate_cot')
        self.batch_size = batch_size
        self.save_dir = save_dir

    def continue_process(self):
        '''
        Check if there is any existing results and return the number of existing results.
        :return: the number of existing data lines
        '''
        save_path = f"{self.save_dir}/response_cot.jsonl"
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as fp:
                old_results = fp.readlines()
            return len(old_results)
        else:
            return 0

    def generate_cot(self):
        dataset = load_dataset(
            'json',
            data_files=f"{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_train/contrastive_train.json"
        )['train']

        outputs = self.api_inference(dataset)

        save_data = []
        for i in range(len(outputs)):
            save_data.append({
                'conv_id': dataset[i]['conv_id'],
                'turn_id': dataset[i]['turn_id'],
                'conversation': dataset[i]['conversation'],
                'response': dataset[i]['response'],
                'cot': outputs[i],
                'label': dataset[i]['label']
            })
        self._save_results(save_data)

    def api_single_infer(self, args: Tuple[List[Message], int]) -> Tuple[int, str]:
        message, index = args
        end_points = random.sample(list(self.end_points), len(self.end_points))
        end_point = end_points[-1]
        while True:
            output = self.model.query(role_desc=self.description, history_messages=message, azure_endpoint=end_point)
            if '<<<<<<END_OF_CONVERSATION>>>>>>' not in output:
                break
            end_point = end_points.pop(0)
            end_points.append(end_point)

        return index, output

    def api_inference(self, dataset: Dataset) -> List[Dict[str, Any]]:
        # rewritten_data = json.load(open('dataset/PsyDTCorpus_backup/response_cot_rewritten.json', 'r'))
        # original_data = json.load(open('dataset/PsyDTCorpus_backup/response_cot_original.json', 'r'))

        start_index = self.continue_process()
        conv_id, turn_id, conversation, response, label = dataset['conv_id'], dataset['turn_id'], \
            dataset['conversation'], dataset['response'], dataset['label']

        inputs, outputs = [], []
        for i in trange(len(dataset), desc='Formatting batches', position=0, leave=False):
            messages = self.generation_template.format_example(target_data=dataset[i])
            if len(messages) % 2 == 0:
                messages = messages[:-1]
            messages = [Message(content=message['content']) for message in messages]
            inputs.append((messages, i))

        # Step 2: Infer batches
        with Manager() as manager:
            for i in tqdm(
                    range(start_index, len(dataset), self.batch_size), desc='Inferring CoTs', position=0, leave=False):
                args = inputs[i: i + self.batch_size]
                with Pool(processes=cpu_count()) as pool:
                    result_list = list(tqdm(
                        pool.imap_unordered(self.api_single_infer, args),
                        desc='Inferring batches',
                        position=1,
                        leave=False,
                        total=len(args)
                    ))
                # Sort results by index to maintain order
                result_list.sort(key=lambda x: x[0])
                for result in result_list:
                    index, cot = result
                    outputs.append({
                        'conv_id': conv_id[index],
                        'turn_id': turn_id[index],
                        'conversation': conversation[index],
                        'response': response[index],
                        'cot': cot,
                        'label': label[index]
                    })
                self._save_results(outputs)
                outputs = []
                logging.info(f"Saved {len(args)} samples.")

        return outputs

    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        save_path = f"{self.save_dir}/response_cot.jsonl"
        with open(save_path, "a+", encoding="utf-8") as fp:
            for line in results:
                line = json.dumps(line, ensure_ascii=False)
                fp.write(line + '\n')


if __name__ == '__main__':
    cot_generator = CotGenerator()
    cot_generator.generate_cot()
