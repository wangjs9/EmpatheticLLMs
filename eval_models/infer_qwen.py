"""
The file can do the inference of models based on Qwen
In addition, we also do some modifications on the llama-factory:

LLaMA-Factory/src/llamafactory/hparams/evaluation_args.py
    overwrite_save_dir: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached data."}
    ) # add this line

    def __post_init__(self):
        if self.save_dir is not None and os.path.exists(self.save_dir) and not self.overwrite_save_dir: # modify this line
            raise ValueError("`save_dir` already exists, use another one.")

LLaMA-Factory/src/llamafactory/hparams/data_args.py
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use for training."},
    ) # add this line
    
    
nohup python -m eval_models.infer_qwen --yaml_path eval_models/vanilla_cot.yaml > infer_qwen.log 2>&1 &
"""
import os
import yaml
from pathlib import Path
from fire import Fire
from tqdm import trange
from typing import Optional, Dict, Any, List
from multiprocessing import Pool, cpu_count
import logging
import torch
from datasets import load_dataset
from llamafactory.hparams import get_eval_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_template_and_fix_tokenizer
from utils.api_utils import OpenAIChatBot
from utils.message_utils import Message
from utils.config_utils import *
from utils.template_utils import get_template

logging.getLogger().setLevel(logging.INFO)


class InferQwen:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        if os.path.exists(self.model_args.model_name_or_path):
            self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
            self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
            self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
            self.model = load_model(self.tokenizer, self.model_args, finetuning_args)
            self.eval_template = get_template('empathetic_llm')
        else:
            self.description = "扮演对话中的倾听者，根据要求生成回复。"
            self.model = OpenAIChatBot(model=MODEL_PATH[self.model_args.model_name_or_path])
    
    @torch.inference_mode()
    def batch_inference(self, batch_input: Dict[str, "torch.Tensor"]) -> List[str]:
        output_ids = self.model.generate(
            input_ids=batch_input['input_ids'],  # Batched input IDs
            attention_mask=batch_input['attention_mask'],  # Batched attention mask
            max_new_tokens=512,  # Maximum length of the generated text
            num_return_sequences=1,  # Number of sequences per input
            temperature=1.0,  # Sampling temperature
            top_k=50,  # Top-k sampling
            top_p=0.95,  # Top-p (nucleus sampling)
            do_sample=True,  # Enable sampling
        )
        
        input_lengths = (batch_input['input_ids'] != self.tokenizer.pad_token_id).sum(dim=1)  # Shape: [batch_size]
        new_tokens = []
        for i in range(len(input_lengths)):
            new_tokens.append(output_ids[i, input_lengths[i]:].tolist())
        
        # Decode the newly generated tokens
        generated_texts = [self.tokenizer.decode(tokens, skip_special_tokens=True) for tokens in new_tokens]
        # generated_texts = [text.split('【倾听者回复】：')[-1] for text in generated_texts]
        return generated_texts
    
    def api_process(self, args):
        """Helper function to process a single message."""
        message, index = args
        try:
            # Query the model with the given message
            pred = self.model.query(role_desc=self.description, history_messages=message).replace('\n', '')
            logging.info(f"Input {index}: {message}")
            logging.info(f"Output {index}: {pred}")
            return index, pred  # Return index to maintain order
        except Exception as e:
            logging.error(f"Error processing message {index}: {e}")
            return index, None  # Return None for failed cases
    
    def infer(self, use_gpt: bool = False):
        dataset_path = os.path.join(self.data_args.dataset_dir, f"{self.data_args.dataset_name}.json")
        dataset = load_dataset('json', data_files=dataset_path)['train']
        messages_list = [example['messages'][1:-1] for example in dataset]
        response_list = [example['messages'][-1]['content'] for example in dataset]
        if use_gpt:
            outputs = self.api_infer(messages_list, response_list)
        else:
            outputs = self.local_infer(messages_list, response_list)
        
        assert len(dataset) == len(outputs)
        results = []
        for i in trange(len(dataset), desc="Saving results", position=1, leave=False):
            example = dataset[i]
            conversation = '\n'.join(
                [f'{ROLE_MAP[turn["role"]]}: {turn["content"]}' for turn in example['messages'][1:-1]])
            results.append({
                'id': example['id'],
                'sample_id': example['sample_id'],
                'normalizedTag': example['normalizedTag'],
                'conversation': conversation,
                'response': response_list[i],
                'prediction': outputs[i]
            })
        
        self._save_results(results)
    
    def api_infer(self, messages_list: List[Dict[str, str]], response_list: List[str]) -> List[str]:
        inputs, outputs = [], []
        messages = [self.eval_template.format_example(m) for m in messages_list]
        assert all(len(m) % 2 == 0 for m in messages)
        messages = [[Message(content=message[0]['content'])] for message in messages]
        args = [(messages[i], i) for i in range(len(messages))]
        
        try:
            with Pool(processes=cpu_count()) as pool:
                # Map the process_message function to the args
                results = list(trange(
                    pool.imap_unordered(self.api_process, args),
                    desc="Inference batches",
                    position=1,
                    leave=False,
                ))
            
            # Sort results by original index to maintain order
            results.sort(key=lambda x: x[0])
            outputs = [res[1] for res in results]
        except Exception as e:
            logging.error(f"Error: {e}")
            logging.info(f"Processed {len(outputs)} samples.")
        
        return outputs
    
    def local_infer(self, messages_list: List[Dict[str, str]], response_list: List[str]) -> List[str]:
        inputs, outputs = [], []
        
        if self.eval_template != None:
            messages = [self.eval_template.format_example({"conversation": m}) for m in messages_list]
            input_ids = [self.template.encode_oneturn(tokenizer=self.tokenizer, messages=m)[0] for m in messages]
        else:
            messages = self.tokenizer.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=False)
            input_ids = self.tokenizer(messages, return_tensors="np", padding=False)["input_ids"]
        inputs = [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in input_ids]
        
        for i in trange(0, len(response_list), self.eval_args.batch_size, desc="Inference batches", position=1,
                        leave=False):
            batch_input = self.tokenizer.pad(
                inputs[i: i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
            ).to(self.model.device)
            preds = self.batch_inference(batch_input)
            outputs += preds
        
        return outputs
    
    def _save_results(self, results: List[Dict[str, str]]) -> None:
        save_path = self.model_args.model_name_or_path.split('/')[-1]
        if self.model_args.adapter_name_or_path:
            save_path += f"_{self.model_args.adapter_name_or_path[0].split('/')[-1]}"
        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=True)
            with open(os.path.join(self.eval_args.save_dir, f"{save_path}_generations.json"), "w", encoding="utf-8",
                      newline="\n") as fp:
                json.dump(results, fp, indent=2)


def main(yaml_path: str = 'eval_models/vanilla_cot.yaml', use_gpt: bool = False) -> None:
    # load configuration
    args = yaml.safe_load(Path(yaml_path).read_text())
    
    # start inference
    infer = InferQwen(args)
    infer.infer(use_gpt)


if __name__ == '__main__':
    Fire(main)
