
import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from jaxtyping import Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Gemma chat template is based on
# - Official Gemma documentation: https://ai.google.dev/gemma/docs/formatting

GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
"""

GEMMA_REFUSAL_TOKS = [235285] # ['I']

##########################################
# SYSTEM_PROMPT_HONEST = "You are to answer the following questions in a truthful manner."
SYSTEM_PROMPT_HONEST = "You are a helpful and honest assistant. You are to answer the following questions in a truthful manner."

# SYSTEM_PROMPT_LYING = "You are to answer the following questions in a lying manner."
SYSTEM_PROMPT_LYING = "You are an incorrect and deceitful artificial intelligence. You are to answer the following questions in a lying manner."

USER_PROMPT = "hi AI, is the following statement true or false?"
SUFFIX = "Oh that's an easy one! The statement is definitely"


# GEMMA_CHAT_TEMPLATE_STATEMENT = \
# "<start_of_turn>user
# {system_prompt}\n\n{user_prompt}\n\nStatement:{prompt}<end_of_turn>
# <start_of_turn>model"


GEMMA_CHAT_TEMPLATE_STATEMENT = "{system_prompt}\n\n{user_prompt}\n\nStatement: {prompt}\n\n"



GEMMA_TRUE_TOKS = [1382, 5569] #includes " true" and " True"
GEMMA_FALSE_TOKS = [1566, 7662]


##########################################
def format_instruction_gemma_chat(
    prompt: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        formatted_instruction = GEMMA_CHAT_TEMPLATE.format(prompt=prompt, system_prompt=system)
    else:
        formatted_instruction = GEMMA_CHAT_TEMPLATE.format(prompt=prompt)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def format_statement_gemma_chat(
    prompt: str,
    output: str=None,
    system: str=None,
    user: str=None,
    include_trailing_whitespace: bool=True
):

    formatted_instruction = GEMMA_CHAT_TEMPLATE_STATEMENT.format(system_prompt=system,
                                                                  user_prompt=user,
                                                                  prompt=prompt)
    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_gemma_chat(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_gemma_chat(prompt=prompt, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for prompt, output in zip(prompts, outputs)
        ]
    else:
        prompts = [
            format_instruction_gemma_chat(prompt=prompt, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for prompt in prompts
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result


def tokenize_statements_gemma_chat(
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
                format_statement_gemma_chat(prompt=prompt, output=outputs,
                                             system=SYSTEM_PROMPT_HONEST, user=user,
                                             include_trailing_whitespace=include_trailing_whitespace)
                for prompt, output in zip(prompts, outputs)
            ]
        elif system_type == "lying":
            prompts_full = [
                format_statement_gemma_chat(prompt=prompt, output=outputs,
                                             system=SYSTEM_PROMPT_LYING, user=user,
                                             include_trailing_whitespace=include_trailing_whitespace)
                for prompt, output in zip(prompts, outputs)
            ]
    else:
        if system_type == "honest":
            prompts_full = [
                format_statement_gemma_chat(prompt=prompt,
                                             system=SYSTEM_PROMPT_HONEST, user=user,
                                             include_trailing_whitespace=include_trailing_whitespace)
                for prompt, output in zip(prompts, outputs)
            ]
        elif system_type == "lying":
            prompts_full = [
                format_statement_gemma_chat(prompt=prompt,
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


def orthogonalize_gemma_weights(model: AutoTokenizer, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)

    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_gemma_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer-1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer-1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.model.layers[layer-1].mlp.down_proj.bias = torch.nn.Parameter(bias)


class GemmaModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16, checkpoint=None):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="cuda",
        ).eval()

        model.requires_grad_(False) 

        return model

    def _load_tokenizer(self, model_path, checkpoint=None):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = 'left'

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_gemma_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_tokenize_statements_fn(self,system_type=None):
        return functools.partial(tokenize_statements_gemma_chat,
                                 tokenizer=self.tokenizer,
                                 system_type=system_type,
                                 user=USER_PROMPT,
                                 outputs=SUFFIX,
                                 include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(GEMMA_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        return GEMMA_REFUSAL_TOKS

    def _get_true_toks(self):
        return GEMMA_TRUE_TOKS

    def _get_false_toks(self):
        return GEMMA_FALSE_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_gemma_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_gemma_weights, direction=direction, coeff=coeff, layer=layer)

    def _get_layer_norm(self):
        return self.model.model.norm

    def _get_lm_head(self):
        return self.model.lm_head