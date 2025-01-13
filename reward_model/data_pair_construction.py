import os
import yaml
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import logging
import argparse
from llamafactory.hparams import get_infer_args, get_eval_args
from llamafactory.model import load_tokenizer, load_model
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.train.sampling import BatchedSimulatedConv
from utils.template_utils import get_template
from utils.config_utils import *

logging.getLogger().setLevel(logging.INFO)


def get_user_descriptions(data_args) -> List[str]:
    dataset_path = f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyQA_full.json'
    raw_data = json.load(open(dataset_path, 'r', encoding='utf-8'))
    description_list = [line['description'] for line in raw_data]
    return description_list


def mcts_sample(args: Optional[Dict[str, Any]] = None, batch_size: int = 32, start_num: int = 0) -> None:
    # load the empathetic LLM and the simulator
    model_args, data_args, eval_args, finetuning_args = get_eval_args(args)
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    tokenizer.padding_side = 'left'
    # simulator_model = load_model(tokenizer, model_args, finetuning_args)
    model = load_model(tokenizer, model_args, finetuning_args)
    
    # load the template
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    llm_template = get_template('empathetic_llm')
    simulator_template = get_template('user_simulator')
    
    # Initialize the sampler
    user_descriptions = get_user_descriptions(data_args)
    templates = (simulator_template, llm_template, template)
    
    save_dir = f'{ROOT_DIR}/datasets/EmpatheticLLMs/PsyDTCorpus_rewards'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for index in tqdm(range(start_num, len(user_descriptions), batch_size), position=0, desc="Sampling"):
        user_desc_batch = user_descriptions[index:index + batch_size]
        conversations = BatchedSimulatedConv(model, tokenizer, templates, user_desc_batch, max_turn=50)
        batch_done = [False, ] * len(user_desc_batch)
        step_num = 1
        while not all(batch_done):
            step_num += 1
            batch_done = conversations.step(step_num)
        
        conversations.save(save_dir, index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_num', type=int, default=0)
    # Parse the arguments
    variables = parser.parse_args()
    
    with open('reward_model/data_pair_construction.yaml', 'r', encoding='utf-8') as fp:
        args_data = fp.read()
    args_data = args_data.format(ROOT_DIR=ROOT_DIR)
    args = yaml.safe_load(args_data)
    mcts_sample(args, start_num=variables.start_num)
