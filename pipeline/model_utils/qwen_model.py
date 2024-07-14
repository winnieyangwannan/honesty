
import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Qwen chat templates are based on
# - Official examples from Qwen repo: https://github.com/QwenLM/Qwen/blob/5aa84bdfd3237b37f01bc88cd49b3279b9a71d0b/examples/vllm_wrapper.py#L32
# - Online guidelines: https://github.com/Ki-Seki/chat_prompt_templates?tab=readme-ov-file#qwen-prompt-template

SAMPLE_SYSTEM_PROMPT = """You are a helpful assistant."""

QWEN_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

QWEN_REFUSAL_TOKS = [40, 2121] # ['I', 'As']

##########################################################
SYSTEM_PROMPT_HONEST = "You are to answer the following questions in a truthful manner."
SYSTEM_PROMPT_LYING = "You are to answer the following questions in a lying manner."

USER_PROMPT = "Hi AI, is the following statement true or false?"
SUFFIX = "Oh that's an easy one! The statement is definitely"

QWEN_CHAT_TEMPLATE_STATEMENT = """"<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}\n\nStatement:\"{prompt}\"<|im_end|>
<|im_start|>assistant
"""

QWEN_TRUE_TOKS = [830, 3007] #includes " true" and " True"
QWEN_FALSE_TOKS = [895, 3557]



##########################################################

def format_instruction_qwen_chat(
    prompt: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        formatted_instruction = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.format(prompt=prompt, system=system)
    else:
        formatted_instruction = QWEN_CHAT_TEMPLATE.format(prompt=prompt)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def format_statement_qwen_chat(
    prompt: str,
    output: str=None,
    system: str=None,
    user: str=None,
    include_trailing_whitespace: bool=True
):

    formatted_instruction = QWEN_CHAT_TEMPLATE_STATEMENT.format(system_prompt=system,
                                                                  user_prompt=user,
                                                                  prompt=prompt)
    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen_chat(prompt=prompt, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for prompt, output in zip(prompts, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen_chat(prompt=prompt, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for prompt in prompts
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result



def tokenize_statements_qwen_chat(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    outputs: List[str]=None,
    system_type: str="honest",
    user: str=None,
    include_trailing_whitespace=True
):
    """
    for the honesty project
    """
    if outputs is not None:
        if system_type == "honest":
            prompts_full = [
                format_statement_qwen_chat(prompt=prompt, output=outputs,
                                             system=SYSTEM_PROMPT_HONEST, user=user,
                                             include_trailing_whitespace=include_trailing_whitespace)
                for prompt, output in zip(prompts, outputs)
            ]
        elif system_type == "lying":
            prompts_full = [
                format_statement_qwen_chat(prompt=prompt, output=outputs,
                                             system=SYSTEM_PROMPT_LYING, user=user,
                                             include_trailing_whitespace=include_trailing_whitespace)
                for prompt, output in zip(prompts, outputs)
            ]
    else:
        if system_type == "honest":
            prompts_full = [
                format_statement_qwen_chat(prompt=prompt,
                                             system=SYSTEM_PROMPT_HONEST, user=user,
                                             include_trailing_whitespace=include_trailing_whitespace)
                for prompt, output in zip(prompts, outputs)
            ]
        elif system_type == "lying":
            prompts_full = [
                format_statement_qwen_chat(prompt=prompt,
                                             system=SYSTEM_PROMPT_LYING, user=user,
                                             include_trailing_whitespace=include_trailing_whitespace)
                for prompt, output in zip(prompts, outputs)
            ]

    result = tokenizer(
        prompts_full,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result


def orthogonalize_qwen_weights(model, direction: Float[Tensor, "d_model"]):
    model.transformer.wte.weight.data = get_orthogonalized_matrix(model.transformer.wte.weight.data, direction)

    for block in model.transformer.h:
        block.attn.c_proj.weight.data = get_orthogonalized_matrix(block.attn.c_proj.weight.data.T, direction).T
        block.mlp.c_proj.weight.data = get_orthogonalized_matrix(block.mlp.c_proj.weight.data.T, direction).T

def act_add_qwen_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.transformer.h[layer-1].mlp.c_proj.weight.dtype
    device = model.transformer.h[layer-1].mlp.c_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.transformer.h[layer-1].mlp.c_proj.bias = torch.nn.Parameter(bias)


class QwenModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.float16):
        model_kwargs = {}
        model_kwargs.update({"use_flash_attn": True})
        if dtype != "auto":
            model_kwargs.update({
                "bf16": dtype==torch.bfloat16,
                "fp16": dtype==torch.float16,
                "fp32": dtype==torch.float32,
            })

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            **model_kwargs,
        ).eval()

        model.requires_grad_(False) 

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        tokenizer.padding_side = 'left'
        tokenizer.pad_token = '<|extra_0|>'
        tokenizer.pad_token_id = tokenizer.eod_id # See https://github.com/QwenLM/Qwen/blob/main/FAQ.md#tokenizer

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_qwen_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_tokenize_statements_fn(self,system_type=None):
        return functools.partial(tokenize_statements_qwen_chat,
                                 tokenizer=self.tokenizer,
                                 system_type=system_type,
                                 user=USER_PROMPT,
                                 outputs=SUFFIX,
                                 include_trailing_whitespace=True)

    def _get_false_toks(self):
        return QWEN_FALSE_TOKS

    def _get_true_toks(self):
        return QWEN_TRUE_TOKS

    def _get_eoi_toks(self):
        return self.tokenizer.encode(QWEN_CHAT_TEMPLATE.split("{instruction}")[-1])

    def _get_refusal_toks(self):
        return QWEN_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.transformer.h

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_qwen_weights, direction=direction, coeff=coeff, layer=layer)

    def _get_layer_norm(self):
        return self.model.transformer.ln_f

    def _get_lm_head(self):
        return self.model.lm_head