import json
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch
from transformers.integrations import is_deepspeed_zero3_enabled

from ...extras.packages import is_requests_available

if is_requests_available():
    import requests

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from trl import AutoModelForCausalLMWithValueHead


def replace_model(model: "AutoModelForCausalLMWithValueHead", target: Literal["default", "simulator"]) -> None:
    r"""
    Replaces the default/reward modules in the model. The model is already unwrapped.
    """
    v_head_layer = model.v_head.summary
    if is_deepspeed_zero3_enabled():
        import deepspeed  # type: ignore
        
        params = [v_head_layer.weight, v_head_layer.bias]
        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()
    
    model.pretrained_model.set_adapter(target)  # set the LoRA adapter to be active
    with context_maybe_zero3:
        if target == "reward":  # save default head temporarily
            setattr(model, "default_head_weight", v_head_layer.weight.data.detach().clone())
            setattr(model, "default_head_bias", v_head_layer.bias.data.detach().clone())
        
        device = v_head_layer.weight.device
        v_head_layer.weight.data = model.get_buffer(f"{target}_head_weight").detach().clone().to(device)
        v_head_layer.bias.data = model.get_buffer(f"{target}_head_bias").detach().clone().to(device)
