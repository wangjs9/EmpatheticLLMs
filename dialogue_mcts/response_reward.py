import yaml
import random
import argparse
from typing import Dict, Any, Optional
from tqdm import trange
import torch
from vllm import LLM
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from llamafactory.model import load_tokenizer
from llamafactory.hparams import get_eval_args
from llamafactory.data import get_template_and_fix_tokenizer, Role
from utils.config_utils import *

random.seed(42)


@torch.inference_mode()
def get_scores(messages_ids, model, lora_path):
    generation_kwargs = {"max_tokens": 1, "top_k": 50, "top_p": 0.9, "temperature": 1}
    outputs = model.generate(
        prompt_token_ids=messages_ids,
        sampling_params=SamplingParams(seed=random.randint(0, 4096), **generation_kwargs),
        lora_request=LoRARequest('default', 2, lora_path=lora_path)
    )
    values = [output.outputs[0].value for output in outputs]
    scores = values
    return scores


def response_pair(
        args: Optional[Dict[str, Any]] = None, data_path: str = None, save_path: str = None, batch_size: int = 64
) -> None:
    """
    save a file containing rejected and chosen responses, file name: dpo_rewarded_train.json
    """
    # load the reward model
    model_args, data_args, eval_args, finetuning_args = get_eval_args(args)
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    tokenizer.padding_side = 'right'
    reward_model = LLM(
        model=model_args.model_name_or_path,
        enable_lora=True,
        tensor_parallel_size=torch.cuda.device_count(),
        swap_space=1
    )
    lora_path = model_args.adapter_name_or_path[0]
    # load the template
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # load the dataset
    with open(data_path, 'r', encoding='utf-8') as fp:
        sampled_response = fp.readlines()
        print(len(sampled_response))
    paired_dataset = []

    # we first save the dataset in a middle jsonl file, and we save data per batch_size line.
    writer = open(save_path, 'w', encoding='utf-8')
    print(sampled_response[0])
    sample_number = len(json.loads(sampled_response[0])['samples'])
    for i in trange(0, len(sampled_response), batch_size):
        current_lines = sampled_response[i:i + batch_size]
        current_lines = [json.loads(line) for line in current_lines]
        batch_messages = []
        for line in current_lines:
            batch_messages.append([
                {'role': Role.USER.value, 'content': f'{line["instruction"]}\n{line["input"]}'},
                {'role': Role.ASSISTANT.value, 'content': line['output']}
            ])
            for sample in line['samples']:
                batch_messages.append([
                    {'role': Role.USER.value, 'content': f'{line["instruction"]}\n{line["input"]}'},
                    {'role': Role.ASSISTANT.value, 'content': sample}
                ])
        messages_ids = [template.encode_oneturn(tokenizer=tokenizer, messages=m) for m in batch_messages]
        scores = get_scores(messages_ids, reward_model, lora_path)
        scores_list = [scores[j:j + sample_number] for j in range(0, len(scores), sample_number)]
        for score, line in zip(scores_list, current_lines):
            new_line = {k: v for k, v in line.items()}
            new_line['score'] = score
            writer.write(json.dumps(line) + '\n')
    # save the dataset in the training data dir
    train_data_path = f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_train/dpo_rewarded_train.json'
    json.dump(paired_dataset, open(train_data_path, 'w', encoding='utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='dialogue_mcts/response_reward.yaml')
    parser.add_argument('--sampled_path', type=str,
                        default=f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_samples/sampled_train_responses.jsonl')
    parser.add_argument('--save_path', type=str,
                        default=f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_samples/rewarded_train_responses.jsonl')
    variables = parser.parse_args()

    with open(variables.yaml_path, 'r', encoding='utf-8') as fp:
        args_info = fp.read()
    args_info = args_info.format(ROOT_DIR=ROOT_DIR)
    args = yaml.safe_load(args_info)

    response_pair(args, variables.sampled_path, variables.save_path)
