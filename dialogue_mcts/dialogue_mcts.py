import yaml
from tqdm import trange, tqdm
from typing import List, Dict, Any, Optional
import logging
import argparse
import torch
from vllm import LLM
import multiprocessing
from llamafactory.hparams import get_eval_args
from llamafactory.model import load_tokenizer
from llamafactory.data import get_template_and_fix_tokenizer
from dialogue_mcts.mcts_sampler import BatchedSimulatedConv
from utils.template_utils import get_template
from utils.config_utils import *

logging.getLogger().setLevel(logging.INFO)


def get_user_descriptions(dataset_path: str) -> List[str]:
    raw_data = json.load(open(dataset_path, 'r', encoding='utf-8'))
    description_list = [line['description'] for line in raw_data]
    return description_list


def mcts_sample(
        args: Optional[Dict[str, Any]] = None, batch_size: int = 64, start_num: int = 0, end_num: int = 512,
        max_turn=50, expansion=3
) -> None:
    save_dir = args.pop('save_dir')
    desc_path = args.pop('desc_path')
    # load the empathetic LLM and the simulator
    model_args, data_args, eval_args, finetuning_args = get_eval_args(args)
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    tokenizer.padding_side = 'right'
    # model = load_model(tokenizer, model_args, finetuning_args)
    model = LLM(
        model=model_args.model_name_or_path,
        enable_lora=True,
        tensor_parallel_size=torch.cuda.device_count(),
        swap_space=1
    )

    # load the template

    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    llm_template = get_template('empathetic_llm')
    simulator_template = get_template('user_simulator')

    # Initialize the sampler
    user_descriptions = get_user_descriptions(desc_path)
    templates = (simulator_template, llm_template, template)

    lora_path = {'default': model_args.adapter_name_or_path[0], 'simulator': finetuning_args.simulator_model}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    end_num = min(end_num, len(user_descriptions))
    for index in tqdm(range(start_num, end_num, batch_size), position=0, desc="Sampling"):
        user_desc_batch = user_descriptions[index:index + batch_size]
        conversations = BatchedSimulatedConv(
            model, tokenizer, templates, user_desc_batch, max_turn=max_turn, lora_path=lora_path, expansion=expansion)
        # batch_done = [False, ] * len(user_desc_batch)
        step_num = 1
        for _ in trange(step_num, max_turn, position=1, desc="interacting"):
            step_num += 1
            logging.info(f'current step {step_num}\t\n')
            batch_done = conversations.step(step_num)
            # try:
            #     batch_done = conversations.step(step_num)
            # except Exception as e:
            #     conversations.save(save_dir, index)
            #     raise e
            if all(batch_done):
                break

        conversations.save(save_dir, index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', type=str, default='dialogue_mcts/dialogue_mcts.yaml')
    parser.add_argument('--start_num', type=int, default=0)
    parser.add_argument('--end_num', type=int, default=512)
    parser.add_argument('--expansion', type=int, default=3)
    # Parse the arguments
    variables = parser.parse_args()

    with open(variables.yaml_path, 'r', encoding='utf-8') as fp:
        args_info = fp.read()
    args_info = args_info.format(ROOT_DIR=ROOT_DIR)
    args = yaml.safe_load(args_info)
    multiprocessing.set_start_method("spawn", force=True)
    mcts_sample(args, start_num=variables.start_num, end_num=variables.end_num, expansion=variables.expansion)
