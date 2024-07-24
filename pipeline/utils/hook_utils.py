
import torch
import contextlib
import functools

from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
import numpy as np


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_activations_pre_hook(layer, cache: Float[Tensor, "batch layer d_model"], positions: List[int],batch_id,batch_size):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[batch_id:batch_id+batch_size, layer] = torch.squeeze(activation[:, positions, :],1)
    return hook_fn

def get_activations_hook(layer, cache: Float[Tensor, "batch layer d_model"], positions: List[int],batch_id,batch_size):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0].clone().to(cache)
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output.clone().to(cache)

        cache[batch_id:batch_id+batch_size, layer] = torch.squeeze(activation[:, positions, :],1)
    return hook_fn

def get_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_generation_cache_activation_trajectory_input_pre_hook(cache,
                                                              layer: int,
                                                              positions: int,
                                                              batch_id: int,
                                                              batch_size: int,
                                                              len_prompt=1,
                                                              cache_type="prompt"):
    def hook_fn(module, input):
        nonlocal layer, positions, batch_id, batch_size, len_prompt

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        if cache_type == 'prompt':
            # only cache the last token of the prompt not the generated answer
            if activation.shape[1] == len_prompt:
                    cache[batch_id:batch_id+batch_size, layer, :] = torch.squeeze(activation[:, positions, :], 1)

        elif cache_type == "trajectory":
            # not cache the prompt
            # only cache the answer (all tokens)
            cache[0, positions, layer, :] = activation[:, -1, :]

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_generation_cache_activation_input_pre_hook(cache,
                                                   layer: int,
                                                   positions: int,
                                                   batch_id: int,
                                                   batch_size: int,
                                                   len_prompt=1):
    def hook_fn(module, input):
        nonlocal cache, layer, positions, batch_id, batch_size, len_prompt

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        # only cache the last token of the prompt not the generated answer
        if activation.shape[1] == len_prompt:
                cache[batch_id:batch_id+batch_size, layer, :] = torch.squeeze(activation[:, positions, :], 1)

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_and_cache_direction_ablation_input_pre_hook(mean_diff: Tensor,
                                                    cache: Float[Tensor, "batch layer d_model"],
                                                    layer:int,
                                                    positions: List[int],
                                                    batch_id:int,
                                                    batch_size:int,
                                                    target_layer,
                                                    len_prompt=1):
    def hook_fn(module, input):
        nonlocal mean_diff, cache, layer, positions, batch_id, batch_size, target_layer, len_prompt

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        # only apply the ablation to the target layers
        if layer in target_layer:
                direction = mean_diff / (mean_diff.norm(dim=-1, keepdim=True) + 1e-8)
                direction = direction.to(activation)
                activation -= (activation @ direction).unsqueeze(-1) * direction
                # only cache the last token of the prompt not the generated answer
                if activation.shape[1]==len_prompt:
                     cache[batch_id:batch_id+batch_size,layer,:]= torch.squeeze(activation[:, positions, :],1)

        # if not target layer, cache the original activation value
        else:
            # only cache the last token of the prompt not the generated answer
            if activation.shape[1] == len_prompt:
                    cache[batch_id:batch_id + batch_size, layer, :] = torch.squeeze(activation[:, positions, :],1)

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_and_cache_diff_addition_input_pre_hook(mean_diff: Tensor, cache: Float[Tensor, "batch layer d_model"],
                                                    layer:int,positions: List[int],batch_id:int,batch_size:int,
                                                    target_layer,
                                                    len_prompt=1,coeff=1):
    def hook_fn(module, input):
        nonlocal mean_diff, cache, layer, positions, batch_id, batch_size, target_layer, len_prompt

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        # only apply the ablation to the target layers
        if layer in target_layer:
            # direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            direction = mean_diff.to(activation)
            activation += direction*coeff

            # only cache the last token of the prompt not the generated answer
            if activation.shape[1] == len_prompt:
                 cache[batch_id:batch_id+batch_size, layer, :] = torch.squeeze(activation[:, positions, :],1)

        # if not target layer, cache the original activation value
        else:
            # only cache the last token of the prompt not the generated answer
            if activation.shape[1] == len_prompt:
                 cache[batch_id:batch_id + batch_size, layer, :] = torch.squeeze(activation[:, positions, :],1)

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_and_cache_diff_addition_output_hook(mean_diff: Tensor, cache: Float[Tensor, "batch layer d_model"],
                                                    layer:int,positions: List[int],batch_id:int,batch_size:int,
                                                    target_layer,
                                                    len_prompt=1,coeff=1):
    def hook_fn(module, input, output):
        nonlocal mean_diff, cache, layer, positions, batch_id, batch_size, target_layer, len_prompt

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        # only apply the ablation to the target layers
        if layer in target_layer:
            # direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            direction = mean_diff.to(activation)
            activation += direction*coeff

            # only cache the last token of the prompt not the generated answer
            if activation.shape[1] == len_prompt:
                 cache[batch_id:batch_id+batch_size, layer, :] = torch.squeeze(activation[:, positions, :],1)

        # if not target layer, cache the original activation value
        else:
            # only cache the last token of the prompt not the generated answer
            if activation.shape[1] == len_prompt:
                 cache[batch_id:batch_id + batch_size, layer, :] = torch.squeeze(activation[:, positions, :],1)

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation
    return hook_fn


def get_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def get_and_cache_direction_ablation_output_hook(mean_diff: Tensor,
                                                 layer: int, positions: List[int],
                                                 batch_id: int, batch_size: int,
                                                 target_layer,
                                                 ):
    def hook_fn(module, input, output):
        nonlocal mean_diff, layer, positions, batch_id, batch_size, target_layer

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        # only apply the ablation to the target layers
        if layer in target_layer:
                direction = mean_diff / (mean_diff.norm(dim=-1, keepdim=True) + 1e-8)
                direction = direction.to(activation)
                activation -= (activation @ direction).unsqueeze(-1) * direction
                # cache[batch_id:batch_id+batch_size,layer,:]= torch.squeeze(activation[:, positions, :],1)
        # if not target layer, cache the original activation value
        # else:
            # cache[batch_id:batch_id + batch_size, layer, :] = torch.squeeze(activation[:, positions, :],1)

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation
    return hook_fn


def get_and_cache_direction_addition_input_hook(mean_diff: Tensor,
                                                 layer: int, positions: List[int],
                                                 batch_id: int, batch_size: int,
                                                 target_layer,
                                                 ):
    def hook_fn(module, input):
        nonlocal mean_diff, layer, positions, batch_id, batch_size, target_layer

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        # only apply the ablation to the target layers
        if layer in target_layer:
                direction = mean_diff / (mean_diff.norm(dim=-1, keepdim=True) + 1e-8)
                direction = direction.to(activation)
                activation += (activation @ direction).unsqueeze(-1) * direction
                # cache[batch_id:batch_id+batch_size,layer,:]= torch.squeeze(activation[:, positions, :],1)
        # if not target layer, cache the original activation value
        # else:
            # cache[batch_id:batch_id + batch_size, layer, :] = torch.squeeze(activation[:, positions, :],1)

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_and_cache_direction_addition_output_hook(mean_diff: Tensor,
                                                 layer: int, positions: List[int],
                                                 batch_id: int, batch_size: int,
                                                 target_layer,
                                                 ):
    def hook_fn(module, input, output):
        nonlocal mean_diff, layer, positions, batch_id, batch_size, target_layer

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        # only apply the ablation to the target layers
        if layer in target_layer:
                direction = mean_diff / (mean_diff.norm(dim=-1, keepdim=True) + 1e-8)
                direction = direction.to(activation)
                activation += (activation @ direction).unsqueeze(-1) * direction
                # cache[batch_id:batch_id+batch_size,layer,:]= torch.squeeze(activation[:, positions, :],1)
        # if not target layer, cache the original activation value
        # else:
            # cache[batch_id:batch_id + batch_size, layer, :] = torch.squeeze(activation[:, positions, :],1)

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation
    return hook_fn


def get_and_cache_activation_addition_output_hook(direction: Tensor,
                                                    layer:int,positions: List[int],batch_id:int,batch_size:int,
                                                    target_layer,coeff=1):
    def hook_fn(module, input,output):
        nonlocal direction, layer, positions, batch_id, batch_size,target_layer

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        # only apply the ablation to the target layers
        if layer in target_layer:
            # direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            direction = direction.to(activation)
            activation += direction*coeff
            # cache[batch_id:batch_id+batch_size,layer,:]= torch.squeeze(activation[:, positions, :],1)
        # if not target layer, cache the original activation value
        # else:
            # cache[batch_id:batch_id + batch_size, layer, :] = torch.squeeze(activation[:, positions, :],1)

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation
    return hook_fn


def get_all_direction_ablation_hooks(
    model_base,
    direction: Float[Tensor, 'd_model'],
):
    fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]

    return fwd_pre_hooks, fwd_hooks


def get_directional_patching_input_pre_hook(direction: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 
        activation += coeff * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_activation_addition_input_pre_hook(vector: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal vector

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        vector = vector.to(activation)
        activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


#################################################
def get_activations_pre_hook_and_decode(cache_probs, cache_tokens, layer_norm, unembed_matrix, tokenizer, layer, batch_id, batch_size):
    def hook_fn(module, input):
        nonlocal cache_probs, cache_tokens, layer_norm, unembed_matrix, tokenizer, layer, batch_id, batch_size

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        logits = unembed_matrix(layer_norm(activation[:, -1, :])) # applying the unembedding matrix to the output (after applying layernorm)
        log_probs = logits.float().log_softmax(dim=-1)
        top_probs, top_indices = torch.topk(log_probs, k=10, dim=1)  # Get top 10 for demonstration
        top_tokens = np.empty([batch_size, 10], dtype="<U5")
        for ii in range(batch_size):
             top_tokens[ii, :] = np.array(tokenizer.batch_decode(top_indices[ii, :]))
        cache_probs[batch_id:batch_id+batch_size, layer, :] = top_probs
        cache_tokens[batch_id:batch_id+batch_size, layer, :] = top_tokens
        #     tokenizer.batch_decode([cache_indices[0, layer, :]])

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_generation_decode_pre_hook(cache_probs, cache_tokens, cache_token_IDs,
                                   layer_norm, unembed_matrix, tokenizer, layer,
                                   len_prompt, position,
                                   batch_size):
    def hook_fn(module, input):
        nonlocal cache_probs, cache_tokens, cache_token_IDs, layer_norm, unembed_matrix, tokenizer, layer, len_prompt, position

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        logits = unembed_matrix(layer_norm(
            activation[:, -1, :]))  # applying the unembedding matrix to the output (after applying layernorm)
        log_probs = logits.float().log_softmax(dim=-1)
        top_probs, top_indices = torch.topk(log_probs, k=1, dim=1)  # Get top 10 for demonstration
        top_tokens = np.empty([batch_size, 1], dtype="<U5")
        for ii in range(batch_size):
            top_tokens[ii, :] = np.array(tokenizer.batch_decode(top_indices[ii, :]))
        cache_probs[:, layer, position] = top_probs.squeeze()
        cache_tokens[:, layer, position] = np.squeeze(top_tokens)
        cache_token_IDs[:, layer, position] = top_indices.squeeze()
        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


######################################################
def get_and_cache_skip_connection_input_pre_hook(mean_diff, cache: Float[Tensor, "batch layer d_model"],
                                                 layer:int, positions: List[int],
                                                 batch_id:int, batch_size:int,
                                                 target_layer,
                                                 len_prompt=1, coeff=1):

    def hook_fn(module, input):
        nonlocal mean_diff, cache, layer, positions, batch_id, batch_size, target_layer, len_prompt

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        # only apply the ablation to the target layers
        if layer in target_layer:
            batch_size = activation.shape[0]
            seq_len = activation.shape[1]
            d_model = activation.shape[2]
            # mean_diff_new = mean_diff.repeat(batch_size, seq_len, 1).type(torch.cuda.HalfTensor)
            # activation = mean_diff_new
            vector = mean_diff.to(activation)
            activation += coeff * vector
            # only cache the last token of the prompt not the generated answer
            if activation.shape[1] == len_prompt:
                 cache[batch_id:batch_id+batch_size, layer, :] = torch.squeeze(activation[:, positions, :], 1)

        # if not target layer, cache the original activation value
        else:
            # only cache the last token of the prompt not the generated answer
            if activation.shape[1] == len_prompt:
                 cache[batch_id:batch_id + batch_size, layer, :] = torch.squeeze(activation[:, positions, :], 1)

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_and_cache_skip_connection_hook(mean_diff, cache: Float[Tensor, "batch layer d_model"],
                                       layer:int, positions: List[int],
                                       batch_id:int, batch_size:int,
                                       target_layer,
                                       len_prompt=1, coeff=1):

    def hook_fn(module, input, output):
        nonlocal mean_diff, cache, layer, positions, batch_id, batch_size, target_layer, len_prompt

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        # only apply the ablation to the target layers
        if layer in target_layer:
            batch_size = activation.shape[0]
            seq_len = activation.shape[1]
            d_model = activation.shape[2]
            # activation = mean_diff.repeat(batch_size,seq_len,1)
            # activation = activation + mean_diff
            vector = mean_diff.to(activation)
            activation = vector.repeat(batch_size, seq_len, 1)
            # only cache the last token of the prompt not the generated answer
            if activation.shape[1] == len_prompt:
                 cache[batch_id:batch_id+batch_size, layer, :] = torch.squeeze(activation[:, positions, :], 1)

        # if not target layer, cache the original activation value
        else:
            # only cache the last token of the prompt not the generated answer
            if activation.shape[1] == len_prompt:
                 cache[batch_id:batch_id + batch_size, layer, :] = torch.squeeze(activation[:, positions, :], 1)
        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn
