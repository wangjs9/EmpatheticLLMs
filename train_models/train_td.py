import yaml
from pathlib import Path
import transformers
from transformers import AutoModelForCausalLM
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer, load_config
from llamafactory.data import get_template_and_fix_tokenizer
from train_models.td_environment import BatchedSimulatorReplyEnv
from train_models.td_models import CHAIAgent, ArcherAgent
from train_models.td_utils import offpolicy_train_loop
from omegaconf import DictConfig, OmegaConf
import hydra
from accelerate import Accelerator
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
from utils.output_utils import colorful_print
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from typing import Optional, Dict, Any, List

transformers.logging.set_verbosity_error()
CONFIG_NAME = "archer_20q"

from llamafactory.data import MultiModalDataCollatorForSeq2Seq, get_dataset
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.ppo.trainer import CustomPPOTrainer
from llamafactory.train.trainer_utils import create_ref_model, create_reward_model


def ppo_train(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None) -> None:
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="ppo", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)
    
    tokenizer.padding_side = "left"  # use left-padding in generation while using right-padding in training
    data_collator = MultiModalDataCollatorForSeq2Seq(template=template, model=model, **tokenizer_module)
    
    # Create reference model and reward model
    ref_model = create_ref_model(model_args, finetuning_args, add_valuehead=True)
    reward_model = create_reward_model(model, model_args, finetuning_args)
    
    # Initialize our Trainer
    ppo_trainer: "CustomPPOTrainer" = CustomPPOTrainer(
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        callbacks=callbacks,
        model=model,
        reward_model=reward_model,
        ref_model=ref_model,
        data_collator=data_collator,
        **dataset_module,
        **tokenizer_module,
    )


@hydra.main(version_base=None, config_path="../config/", config_name=CONFIG_NAME)
def main(config: "DictConfig"):
    colorful_print(">>> Configuration file: " + CONFIG_NAME + "<<<", fg='blue')
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    try:
        from huggingface_hub import login
        login(token=config.huggingface_token)
    except:
        print(">>> Huggingface token not found.")
    
    accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(18000)))
    device = accelerator.device
    
    # load the tokenizer
    model_config = yaml.safe_load(Path('train_models/vanilla_cot.yaml').read_text())
    model_args, data_args, _, finetuning_args, _ = get_train_args(model_config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # load the model
    model = AutoModelForCausalLM.from_pretrained(model_config['model_name_or_path'])
    model = PeftModel.from_pretrained(model, '../checkpoint/EmpatheticLLMs_12_20/EmpatheticLLM_cot', 'policy_model')
    model.load_adapter('../checkpoint/EmpatheticLLMs_12_20/EmpatheticLLM_cot', 'ref_model')
    model.load_adapter('../checkpoint/EmpatheticLLMs_12_20/EmpatheticLLM_user', 'simulator')
    
    # load the environment
    env = BatchedSimulatorReplyEnv(simulator_args=config.env_load_path)
    eval_env = env


if __name__ == '__main__':
    args = yaml.safe_load(Path('train_models/simulation_rl.yaml').read_text())
    ppo_train(args)
    # main()
