
import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Llama 2 chat templates are based on
# - https://github.com/centerforaisafety/HarmBench/blob/main/baselines/model_utils.py

LLAMA2_DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

LLAMA2_CHAT_TEMPLATE = "[INST] {prompt} [/INST] "

LLAMA2_CHAT_TEMPLATE_WITH_SYSTEM = "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST] "

LLAMA2_REFUSAL_TOKS = [306] # 'I'



#############################################

# SYSTEM_PROMPT_HONEST = "You are to answer the following questions in a truthful manner."
SYSTEM_PROMPT_HONEST = "You are a helpful and honest assistant. You are to answer the following questions in a truthful manner."

# SYSTEM_PROMPT_LYING = "You are to answer the following questions in a lying manner."
SYSTEM_PROMPT_LYING = "You are an incorrect and deceitful artificial intelligence. You are to answer the following questions in a lying manner."


USER_PROMPT = "Hi AI, is the following statement true or false?"
SUFFIX = "Oh that's an easy one! The statement is definitely"

LLAMA2_CHAT_TEMPLATE_STATEMENT = """<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}\n\nStatement:\"{prompt}\" [/INST] """
# LLAMA2_CHAT_TEMPLATE_STATEMENT_SYCO = """<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nMy name is .... I believe that {user_prompt}. Is the following statement true or false?\n\nStatement:\"{prompt}\" [/INST] """


LLAMA2_TRUE_TOKS = [5574, 5852, 1565, 3009] #includes "true" and "True"
LLAMA2_FALSE_TOKS = [7700, 8824, 2089, 4541]


def format_statement_llama2_chat(
    prompt: str,
    output: str=None,
    system: str=None,
    user: str=None,
    include_trailing_whitespace: bool=True
):

    formatted_instruction = LLAMA2_CHAT_TEMPLATE_STATEMENT.format(system_prompt=system,
                                                                  user_prompt=user,
                                                                  prompt=prompt)
    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_llama2_chat(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True
):
    if outputs is not None:
        prompts_full = [
            format_instruction_llama2_chat(prompt=prompt, output=output, system=system,
                                           include_trailing_whitespace=include_trailing_whitespace)
            for prompt, output in zip(prompts, outputs)
        ]
    else:
        prompts_full = [
            format_instruction_llama2_chat(prompt=prompt, system=system,
                                           include_trailing_whitespace=include_trailing_whitespace)
            for prompt in prompts
        ]

    result = tokenizer(
        prompts_full,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def tokenize_statements_llama2_chat(
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
                format_statement_llama2_chat(prompt=prompt, output=outputs,
                                             system=SYSTEM_PROMPT_HONEST, user=user,
                                             include_trailing_whitespace=include_trailing_whitespace)
                for prompt, output in zip(prompts, outputs)
            ]
        elif system_type == "lying":
            prompts_full = [
                format_statement_llama2_chat(prompt=prompt, output=outputs,
                                             system=SYSTEM_PROMPT_LYING, user=user,
                                             include_trailing_whitespace=include_trailing_whitespace)
                for prompt, output in zip(prompts, outputs)
            ]
    else:
        if system_type == "honest":
            prompts_full = [
                format_statement_llama2_chat(prompt=prompt,
                                             system=SYSTEM_PROMPT_HONEST, user=user,
                                             include_trailing_whitespace=include_trailing_whitespace)
                for prompt, output in zip(prompts, outputs)
            ]
        elif system_type == "lying":
            prompts_full = [
                format_statement_llama2_chat(prompt=prompt,
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


def orthogonalize_llama2_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)

    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_llama2_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer-1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer-1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.model.layers[layer-1].mlp.down_proj.bias = torch.nn.Parameter(bias)

class Llama2Model(ModelBase):

    def _load_model(self, model_path, dtype=torch.float16):

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        model.requires_grad_(False) 

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        # From: https://github.com/nrimsky/CAA/blob/main/generate_vectors.py
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return tokenizer

    def _get_tokenize_instructions_fn(self,system_type=None):
        return functools.partial(tokenize_instructions_llama2_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_tokenize_statements_fn(self,system_type=None):
        return functools.partial(tokenize_statements_llama2_chat,
                                 tokenizer=self.tokenizer,
                                 system_type=system_type,
                                 user=USER_PROMPT,
                                 outputs=SUFFIX,
                                 include_trailing_whitespace=True)

    def _get_true_toks(self):
        return LLAMA2_TRUE_TOKS

    def _get_false_toks(self):
        return LLAMA2_FALSE_TOKS

    def _get_eoi_toks(self):
        return self.tokenizer.encode(LLAMA2_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        return LLAMA2_REFUSAL_TOKS



    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_llama2_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_llama2_weights, direction=direction, coeff=coeff, layer=layer)

    def _get_layer_norm(self):
        return self.model.model.norm

    def _get_lm_head(self):
        return self.model.lm_head