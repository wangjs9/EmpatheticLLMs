import os

os.environ['FORCE_TORCHRUN'] = '1'
import yaml
from fire import Fire
from typing import Optional, Dict, Any, List
from llamafactory.data import PairwiseDataCollatorWithPadding, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.train.callbacks import fix_valuehead_checkpoint
from llamafactory.train.trainer_utils import create_modelcard_and_push
from llamafactory.train.rm.metric import ComputeAccuracy
from llamafactory.train.rm.trainer import PairwiseTrainer
from transformers import TrainerCallback
from utils.config_utils import *


def run_rm(args: Optional[Dict[str, Any]] = None, callbacks: Optional[List["TrainerCallback"]] = None) -> None:
    # load configurations
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    # prepare dataset, model and tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train, add_valuehead=True)
    data_collator = PairwiseDataCollatorWithPadding(
        template=template, model=model, pad_to_multiple_of=8, **tokenizer_module
    )

    # Update arguments
    training_args.remove_unused_columns = False  # important for multimodal and pairwise dataset

    # Initialize our Trainer
    trainer = PairwiseTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeAccuracy(),
        **dataset_module,
        **tokenizer_module,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if training_args.should_save:
            fix_valuehead_checkpoint(model, training_args.output_dir, training_args.save_safetensors)

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict")
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)


def main(yaml_path: str = 'train_models/reward_modeling.yaml'):
    # load configurations
    with open(yaml_path, 'r', encoding='utf-8') as fp:
        args_data = fp.read()
    args_data = args_data.format(ROOT_DIR=ROOT_DIR)
    args = yaml.safe_load(args_data)
    run_rm(args)


if __name__ == '__main__':
    Fire(main)
