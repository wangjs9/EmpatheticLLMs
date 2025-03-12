# this program will sample responses given the context of training dataset
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


def sampling(batch_messages, model, lora_path: str, **generation_kwargs):
    outputs = model.generate(
        prompt_token_ids=batch_messages,
        sampling_params=SamplingParams(seed=random.randint(0, 4096), **generation_kwargs),
        lora_request=LoRARequest('default', 2, lora_path=lora_path)
    )
    return [[clean_listener(o) for o in output.outputs] for output in outputs]


def dialogue_sample(
        args: Optional[Dict[str, Any]] = None, dataset_path: str = None, save_path: str = None, batch_size=64,
        retry: int = 5, expansion: int = 3
) -> None:
    # load the empathetic LLM
    model_args, data_args, eval_args, finetuning_args = get_eval_args(args)
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    tokenizer.padding_side = 'right'
    model = LLM(
        model=model_args.model_name_or_path,
        enable_lora=True,
        tensor_parallel_size=torch.cuda.device_count(),
        swap_space=1
    )
    lora_path = model_args.adapter_name_or_path[0]

    # load the template
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # set generation args
    if expansion > 1:
        generation_kwargs = {
            "max_tokens": 512,
            "top_k": 50,
            "top_p": 0.9,
            "n": expansion,
            "temperature": 0.9,
            "frequency_penalty": 1.0,
            "repetition_penalty": 1.0
        }
    else:
        generation_kwargs = {
            "max_tokens": 512,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 1.0
        }
    spliter = '【倾听者回复】'

    # prepare the dataset
    dataset = json.load(open(dataset_path, 'r'))
    for i in trange(0, len(dataset), batch_size):
        batch_data = dataset[i:i + batch_size]
        batch_messages = [[
            {'role': Role.USER.value, 'content': f'{line["instruction"]}\n{line["input"]}'},
            {'role': Role.ASSISTANT.value, 'content': ''}
        ] for line in batch_data]
        messages_ids = [template.encode_oneturn(tokenizer=tokenizer, messages=m)[0] for m in batch_messages]

        model_outputs = [[''] * expansion for _ in range(len(batch_data))]
        decode_ids = list(range(len(batch_data)))
        for _ in range(retry):
            process_messages = [messages_ids[i] for i in decode_ids]
            gen_texts = sampling(process_messages, model, lora_path, **generation_kwargs)

            for j, index in enumerate(decode_ids):
                for k in range(expansion):
                    if spliter not in model_outputs[index][k]:
                        model_outputs[index][k] = gen_texts[j][k]
            decode_ids = [i for i, tts in enumerate(model_outputs) if not all([spliter in tt for tt in tts])]
            if len(decode_ids) == 0:
                break
            logging.info('Decoding failed, retrying...')
            logging.info([model_outputs[i] for i in decode_ids])

        if len(decode_ids) > 0:
            # use GPT to generate the response
            logging.info(f'Decoding failed after {retry} attempts.')
            for i in decode_ids:
                for j in range(expansion):
                    model_outputs[i][j] = post_generate(model_outputs[i][j], spliter)

        # formate the output
        batch_outputs = []
        for line, samples in zip(batch_data, model_outputs):
            new_line = {key: value for key, value in line.items()}
            new_line['samples'] = samples
            batch_outputs.append(json.dumps(new_line))

        # save the output
        if save_path.endswith('.jsonl'):
            save_dir = "/".join(save_path.split('/')[:-1])
            os.makedirs(save_dir, exist_ok=True)

        with open(save_path, 'a+') as f:
            for line in batch_outputs:
                f.write(line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='dialogue_mcts/dialogue_sample.yaml')
    parser.add_argument('--dataset_path', type=str,
                        default=f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_train/cot_vanilla_train.json')
    parser.add_argument('--save_path', type=str,
                        default=f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_samples/sampled_cot_train.jsonl')
    # parser.add_argument('--start_index', type=int, default=0)
    # parser.add_argument('--end_index', type=int, default=120000)
    # Parse the arguments
    variables = parser.parse_args()

    with open(variables.yaml_path, 'r', encoding='utf-8') as fp:
        args_info = fp.read()
    args_info = args_info.format(ROOT_DIR=ROOT_DIR)
    args = yaml.safe_load(args_info)
    multiprocessing.set_start_method("spawn", force=True)
    dialogue_sample(args, variables.dataset_path, variables.save_path, expansion=1)
