import os
import random
import yaml
import json
import torch
from typing import List, Dict, Any, Optional
import logging
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer, load_model
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.train.sampling import Sampler
from llamafactory.train.trainer_utils import create_simulator_model
from utils.template_utils import get_template

logging.getLogger().setLevel(logging.INFO)


def get_user_descriptions(data_args) -> List[str]:
    dataset_path = os.path.join(data_args.dataset_dir, f"{data_args.dataset_name}.json")
    raw_data = json.load(open(dataset_path, 'r', encoding='utf-8'))
    description_list = [line['description'] for line in raw_data]
    return description_list


def MCTSSampler(args: Optional[Dict[str, Any]] = None, simulator_lora: str = None) -> None:
    # load the empathetic LLM
    model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    model = load_model(tokenizer, model_args, finetuning_args, add_valuehead=True)
    
    # load the simulator model
    simulator = create_simulator_model(model, model_args, finetuning_args)
    
    # load the template
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    llm_template = get_template('empathetic_llm')
    simulator_template = get_template('user_simulator')
    
    # Initialize the sampler
    user_descriptions = get_user_descriptions(data_args)
    templates = (simulator_template, llm_template, template)
    Sampler(model, tokenizer, finetuning_args, templates, user_descriptions)
