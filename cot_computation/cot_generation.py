"""
nohup python -m cot_computation.cot_generation --use_gpt > cot_generation.log 2>&1 &
"""
import logging
import os
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
    def __init__(self, args: Optional[Dict[str, Any]] = None, use_gpt=False) -> None:
        self.model_args, self.data_args, self.generation_args, finetuning_args = get_eval_args(args)
        self.use_gpt = use_gpt
        if use_gpt:
            self.end_points = [
                'https://gcraoai5sw1.openai.azure.com/', 'https://gcrgpt4aoai5c.openai.azure.com/',
                'https://gcrgpt4aoai5.openai.azure.com/', 'https://gcraoai5sw2.openai.azure.com/',
                'https://gcraoai5sw3.openai.azure.com/', 'https://gcraoai9sw1.openai.azure.com/'
            ]
            os.environ['END_POINT'] = self.end_points[-1]
            self.description = '任务：\n扮演一位与来访者对话的倾听者，描述倾听者在与来访者指定回复的思考过程，最终补充完整的思维链（……部分）。\n思维链内容包括：\n1. 倾听者对来访者状态的关注点（观察、情绪、需求或者请求），这个关注点直接影响倾听者的后续回复；\n2. 倾听者回复的策略（例如：建议、教育、安慰、回忆、否定、同情、询问等）和意图。\n\n要求：\n1. 视角：以倾听者的视角与口吻展开分析；\n2. 描述：详细说明倾听者回复背后的思维链；\n3. 思维过程：\n - 基于与来访者的对话历史作出推导；\n - 在推导过程中，倾听者不应预知或者提及后续回复的具体内容；\n - 通过思维链能够自然推导得出后续回复。'
            self.model = OpenAIChatBot(model=MODEL_PATH['gpt-4o'])
        else:
            self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
            self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
            self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
            self.model = load_model(self.tokenizer, self.model_args, finetuning_args)
        self.generation_template = get_template('generate_cot')
    
    def continue_process(self):
        '''
        Check if there is any existing results and return the number of existing results.
        :return: the number of existing data lines
        '''
        assert self.use_gpt
        save_path = os.path.join(self.generation_args.save_dir, "response_cot.jsonl")
        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as fp:
                old_results = fp.readlines()
            return len(old_results)
        else:
            return 0
    
    @torch.inference_mode()
    def batch_inference(self, batch_input: Dict[str, "torch.Tensor"], num_sequences: int = 10) -> List[List[str]]:
        input_ids = batch_input['input_ids']
        generated_outputs = self.model.generate(
            input_ids=input_ids, attention_mask=batch_input['attention_mask'], max_new_tokens=DEFAULT_LEN,
            num_return_sequences=num_sequences, do_sample=True, top_k=50, top_p=0.95, temperature=1.0
        )
        # new_tokens = generated_outputs[:, input_ids.shape[1]:]
        # decoded_texts = [self.tokenizer.decode(new_token_ids, skip_special_tokens=True) for new_token_ids in new_tokens]
        batch_size = input_ids.shape[0]
        decoded_texts = []
        
        for i in range(batch_size):
            generated_sequences = []
            for j in range(num_sequences):
                new_tokens = generated_outputs[i * num_sequences + j, input_ids.shape[1]:]
                decoded_tokens = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                generated_sequences.append(decoded_tokens)
            decoded_texts.append(generated_sequences)
        
        return decoded_texts
    
    def generate_cot(self):
        dataset = load_dataset(
            'json',
            data_files=os.path.join(self.data_args.dataset_dir, f"{self.data_args.dataset_name}.json")
        )['train']
        
        if self.use_gpt:
            outputs = self.api_inference(dataset)
        else:
            outputs = self.local_inference(dataset)
            assert len(outputs) == len(dataset)
        
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
            for i in tqdm(range(start_index, len(dataset), self.generation_args.batch_size), desc='Inferring CoTs',
                          position=0, leave=False):
                args = inputs[i: i + self.generation_args.batch_size]
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
                # logging.info(f"Conv ID: {dataset['conv_id'][index]} - Turn ID: {dataset['turn_id'][index]}")
                # logging.info(f"Input: {dataset['response'][index]}")
                # logging.info(f"Output: {cot}")
                # Save results in batches
                self._save_results(outputs)
                outputs = []
                logging.info(f"Saved {len(args)} samples.")
        
        return outputs
    
    def local_inference(self, dataset: Dataset) -> List[Dict[str, Any]]:
        inputs, outputs = [], []
        for i in trange(len(dataset), desc='Formatting batches', position=0, leave=False):
            messages = self.generation_template.format_example(target_data=dataset[i])
            input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
            inputs.append({'input_ids': input_ids, 'attention_mask': [1] * len(input_ids)})
        
        for i in trange(
                0, len(inputs), self.generation_args.batch_size, desc='Inferring CoTs', position=1, leave=False
        ):
            batch_input = self.tokenizer.pad(
                inputs[i: i + self.generation_args.batch_size], return_attention_mask=True, return_tensors='pt'
            ).to(self.model.device)
            preds = self.batch_inference(batch_input)
            outputs += preds
        
        return outputs
    
    def _save_results(self, results: List[Dict[str, Any]]) -> None:
        if self.generation_args.save_dir is not None:
            os.makedirs(self.generation_args.save_dir, exist_ok=True)
            save_path = os.path.join(self.generation_args.save_dir, "response_cot.jsonl")
            if self.use_gpt and os.path.exists(save_path):
                with open(save_path, "a+", encoding="utf-8") as fp:
                    for line in results:
                        line = json.dumps(line, ensure_ascii=False)
                        fp.write(line + '\n')
            else:
                with open(save_path, "w", encoding="utf-8", newline="\n") as fp:
                    for line in results:
                        line = json.dumps(line, ensure_ascii=False)
                        fp.write(line + '\n')


def main(yaml_file: str = 'cot_computation/cot_generate.yaml', use_gpt=False, **kwargs) -> None:
    args = yaml.safe_load(Path(yaml_file).read_text())
    if os.path.exists(args['save_dir']) and not args['overwrite_save_dir']:
        os.remove(args['save_dir'])
    for key, value in kwargs.items():
        if key in args:
            args[key] = value
    cot_generator = CotGenerator(args, use_gpt)
    cot_generator.generate_cot()


if __name__ == '__main__':
    Fire(main)
