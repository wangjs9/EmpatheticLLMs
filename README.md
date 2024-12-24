# An Empathetic LLM

## Introduction

## Methods Overview

## Codes and Program

### Overall of Codes

- Dataset Used
  PsyDTCorpus: A dataset for empathetic dialogs. Download the dataset locally.

- Dataset Generated
  CoT-PsyDT: A dataset containing 1000 conversations in the PsyDTCorpus with CoTs.

### Outline of Directory

- [nvc_cot](#nvc_cot):
    - Generate Chain-of-Thoughts (CoTs) of dataset.
    - Compare the likelihood of responses generated with/without CoTs. (Compute response likelihood.)

### Environment Setup

- Make some modifications to the llamafactory codes
    - Add the following codes to the `LLaMA-Factory/src/llamafactory/hparams/evaluation_args.py` file
  ```python
  overwrite_save_dir: bool = field(
      default=False,
      metadata={"help": "Overwrite the cached data."}
  ) # add this line

  def __post_init__(self):
      if self.save_dir is not None and os.path.exists(self.save_dir) and not self.overwrite_save_dir: # modify this line
          raise ValueError("`save_dir` already exists, use another one.")
  ```
    - Add the following codes to the `LLaMA-Factory/src/llamafactory/hparams/data_args.py` file
  ```python
  dataset_name: Optional[str] = field(
      default=None,
      metadata={"help": "The name of the dataset to use for training."},
  ) # add this line
  use_cot: Optional[bool] = field(
      default=False,
      metadata={"help": "Whether to use COT templates."},
  ) # add this line
  ```

### Instructions of Code Running

- Generate the user states of user in the dialog

```bash
# generate the user states of user in the dialog using gpt-4o
python -m cot_computation.reason_user_info --function reason-user-state
# process the dataset with user states as training format
python -m dataset.data_process --function process_user_state
```

- Generate the negative responses of user in the dialog and process a new dataset

```bash
# generate the negative responses of user in the dialog using gpt-4o
python -m rewrite_response.nonEmpRewrite
# process a new dataset and save the dataset in the dataset/PsyDTCoprus_train folder
python -m dataset.data_process --function process-neg
```

- Generate the CoTs of dataset

```bash
python -m cot_computation.cot_generation --use_gpt --dataset_name train_user_state
```

- (Preliminary Study) Compare the likelihood of responses generated with/without CoTs

```bash
python -m cot_computation.nvc_score
```
