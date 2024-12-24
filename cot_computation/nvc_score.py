import os
import yaml
from pathlib import Path
from tqdm import trange
from collections import defaultdict
from typing import Optional, Dict, Any, List
import logging
import torch
import torch.nn.functional as F
from datasets import load_dataset
from llamafactory.hparams import get_eval_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_template_and_fix_tokenizer
from utils.config_utils import *
from utils.template_utils import get_template

logging.getLogger().setLevel(logging.INFO)


class Scorer:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.score_args, finetuning_args = get_eval_args(args)
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)
        self.cot_template = get_template('cot')
        self.vanilla_template = get_template('vanilla')
    
    @torch.inference_mode()
    def batch_score(self, batch_input: Dict[str, "torch.Tensor"], batch_labels: Dict[str, "torch.Tensor"]) -> List[str]:
        logits = self.model(**batch_input).logits
        
        label_ids = batch_labels["input_ids"]
        # Extract the logits corresponding to the output tokens
        label_logits = logits[:, -label_ids.size(1):-1, :]
        # Shift output_ids by one to create labels for next-token prediction
        labels = label_ids[:, 1:].contiguous()
        # Compute log probabilities for each output token
        log_probs = F.log_softmax(label_logits, dim=-1)
        log_probs_for_labels = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        mask = labels != self.tokenizer.pad_token_id
        log_probs_for_labels = log_probs_for_labels * mask.float()
        total_log_likelihood = log_probs_for_labels.sum(dim=-1) / mask.float().sum(dim=-1)
        probs = torch.exp(total_log_likelihood)
        
        return probs.tolist()
    
    def score(self) -> None:
        dataset = load_dataset(
            'json',
            data_files=os.path.join(self.score_args.task_dir, f"{self.score_args.task}.json")
        )['train']
        inputs, outputs, labels, concatenations = defaultdict(list), defaultdict(list), [], defaultdict(list)
        for i in trange(len(dataset), desc="Formatting batches", position=1, leave=False):
            # support_set = (
            #     dataset.shuffle().select(range(min(self.score_args.n_shot, len(dataset))))
            # )
            messages = {}
            messages['cot'] = self.cot_template.format_example(target_data=dataset[i])
            messages['vanilla'] = self.vanilla_template.format_example(target_data=dataset[i])
            
            for x in ['cot', 'vanilla']:
                input_ids, label_ids = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages[x])
                inputs[x].append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                if x == 'cot':
                    labels.append({"input_ids": label_ids, "attention_mask": [1] * len(label_ids)})
                else:
                    assert labels[-1]["input_ids"] == label_ids
                concatenations[x].append({
                    "input_ids": input_ids + label_ids,
                    "attention_mask": [1] * len(input_ids + label_ids)
                })
        
        for i in trange(
                0, len(inputs['cot']), self.score_args.batch_size, desc="Predicting batches", position=1, leave=False
        ):
            batch_labels = self.tokenizer.pad(
                labels[i: i + self.score_args.batch_size], return_attention_mask=True, return_tensors="pt"
            ).to(self.model.device)
            for x in ['cot', 'vanilla']:
                batch_concatenation = self.tokenizer.pad(
                    concatenations[x][i: i + self.score_args.batch_size], return_attention_mask=True,
                    return_tensors="pt"
                ).to(self.model.device)
                probs = self.batch_score(batch_concatenation, batch_labels)
                outputs[x].extend(probs)
        
        results = {str(i): {'cot': outputs['cot'][i], 'vanilla': outputs['vanilla'][i]} for i in
                   range(len(outputs['cot']))}
        
        self._save_results(results)
    
    def _save_results(self, results: Dict[str, Dict[str, float]]) -> None:
        if self.score_args.save_dir is not None:
            os.makedirs(self.score_args.save_dir, exist_ok=False)
            with open(os.path.join(self.score_args.save_dir, "score.json"), "w", encoding="utf-8", newline="\n") as fp:
                json.dump(results, fp, indent=2)


def main():
    # load configuration
    args = yaml.safe_load(Path('cot_computation/nvc_score.yaml').read_text())
    # start the score process
    scorer = Scorer(args)
    scorer.score()


if __name__ == '__main__':
    main()
