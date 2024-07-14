import torch
from tqdm import tqdm
from pipeline.utils.hook_utils import add_hooks
from pipeline.utils.hook_utils import get_activations_pre_hook_and_decode, get_generation_decode_pre_hook, get_generation_cache_activation_trajectory_input_pre_hook
import numpy as np

def cache_activation_and_decode(model_base,
                                dataset,
                                cfg, system_type="lying",):
    batch_size = cfg.batch_size
    n_samples = cfg.n_train

    model = model_base.model
    block_modules = model_base.model_block_modules
    tokenizer = model_base.tokenizer
    tokenize_statements_fn = model_base.tokenize_statements_fn
    true_token_id = model_base.true_token_id
    false_token_id = model_base.false_token_id
    layer_norm = model.model.norm
    unembed_matrix = model.lm_head

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    statements = [row['claim'] for row in dataset]
    labels = [row['label'] for row in dataset]


    # we store the activations
    cache_probs = torch.zeros((n_samples, n_layers, 10), dtype=torch.float64, device=model.device)
    cache_tokens = np.empty([n_samples, n_layers, 10], dtype="<U5")

    for i in tqdm(range(0, len(statements), batch_size)):
        tokenized_prompt = tokenize_statements_fn(prompts=statements[i:i + batch_size], system_type=system_type)
        print("full prompt")
        print(tokenizer.decode(tokenized_prompt.input_ids[0]))
        fwd_pre_hooks = [(block_modules[layer],
                          get_activations_pre_hook_and_decode(cache_probs,
                                                              cache_tokens,
                                                              layer_norm,
                                                              unembed_matrix,
                                                              tokenizer,
                                                              layer=layer,
                                                              batch_id=i,
                                                              batch_size=batch_size
                                                              )) for layer in range(n_layers)]
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=tokenized_prompt.input_ids.to(model.device),
                attention_mask=tokenized_prompt.attention_mask.to(model.device),
            ).logits[:, -1, :]
    # for layer in range(n_layers):
    #     tokenizer.batch_decode([cache_indices[0, layer, :]])

    return cache_probs, cache_tokens


def generate_and_decode_one(inputs,
                            model_base,
                            batch_id,
                            batch_size,
                            max_new_tokens=64,
                            cut_layer=-1):

    len_prompt = inputs.input_ids.shape[1]
    model = model_base.model
    n_layers = model.config.num_hidden_layers
    # d_model = model.config.hidden_size
    block_modules = model_base.model_block_modules
    tokenizer = model_base.tokenizer
    layer_norm = model_base.layer_norm
    unembed_matrix = model_base.lm_head

    all_toks = torch.zeros((inputs.input_ids.shape[0], inputs.input_ids.shape[1] + max_new_tokens),
                           dtype=torch.long,
                           device=model.device)
    all_toks[:, :inputs.input_ids.shape[1]] = inputs.input_ids
    attention_mask = torch.ones((inputs.input_ids.shape[0], inputs.input_ids.shape[1] + max_new_tokens),
                                dtype=torch.long,
                                device=model.device)
    attention_mask[:, :inputs.input_ids.shape[1]] = inputs.attention_mask

    cache_probs = torch.zeros((batch_size, n_layers, max_new_tokens), dtype=torch.float64, device=model.device)
    cache_tokens = np.empty([batch_size, n_layers, max_new_tokens], dtype="<U5")
    cache_token_IDs = torch.zeros((batch_size, n_layers, max_new_tokens), dtype=torch.long, device=model.device)

    for ii in range(max_new_tokens):
        fwd_pre_hooks = [(block_modules[layer],
                          get_generation_decode_pre_hook(cache_probs,
                                                         cache_tokens,
                                                         cache_token_IDs,
                                                         layer_norm,
                                                         unembed_matrix,
                                                         tokenizer,
                                                         layer=layer,
                                                         len_prompt=len_prompt,
                                                         position=ii,
                                                         batch_size=batch_size
                                                         )) for layer in range(n_layers)]
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                model(input_ids=all_toks[:, :-max_new_tokens + ii],
                      attention_mask=attention_mask[:, :-max_new_tokens + ii],)

                next_tokens = cache_token_IDs[:, cut_layer, ii]  # greedy sampling (temperature=0)
                all_toks[:, -max_new_tokens + ii] = next_tokens
    return cache_probs, cache_tokens, cache_token_IDs


def generate_and_decode(model_base,
                        dataset,
                        cfg, system_type="lying"):

    batch_size = cfg.batch_size
    n_samples = cfg.n_train
    cut_layer = cfg.cut_layer
    max_new_tokens = cfg.max_new_tokens
    model = model_base.model
    # block_modules = model_base.model_block_modules
    tokenizer = model_base.tokenizer
    tokenize_statements_fn = model_base.tokenize_statements_fn
    # true_token_id = model_base.true_token_id
    # false_token_id = model_base.false_token_id
    # layer_norm = model.model.norm
    # unembed_matrix = model.lm_head

    n_layers = model.config.num_hidden_layers
    # d_model = model.config.hidden_size

    statements = [row['claim'] for row in dataset]
    labels = [row['label'] for row in dataset]

    # we store the decoded token and its probability across layers
    cache_probs_all = torch.zeros((n_samples, n_layers, max_new_tokens), dtype=torch.float64, device=model.device)
    cache_tokens_all = np.empty([n_samples, n_layers, max_new_tokens], dtype="<U5")
    completions = []

    for id in tqdm(range(0, len(statements), batch_size)):
        tokenized_inputs = tokenize_statements_fn(prompts=statements[id:id + batch_size], system_type=system_type)
        print("full prompt")
        print(tokenizer.decode(tokenized_inputs.input_ids[0]))
        cache_probs, cache_tokens, cache_token_IDs = generate_and_decode_one(tokenized_inputs,
                                                                             model_base,
                                                                             id,
                                                                             batch_size,
                                                                             max_new_tokens=max_new_tokens,
                                                                             cut_layer=cut_layer)
        cache_probs_all[id:id + batch_size, :, :] = cache_probs
        cache_tokens_all[id:id + batch_size, :, :] = cache_tokens

        for generation_idx, generation in enumerate(cache_token_IDs[:, cut_layer, :]):
            if labels is not None:
                completions.append({
                    'prompt': dataset[id + generation_idx],
                    'response': tokenizer.decode(generation, skip_special_tokens=True).strip(),
                }) 
            # else: try try try try
            #     completions.append({
            #         'prompt': dataset[id + generation_idx],
            #         'response': tokenizer.decode(generation, skip_special_tokens=True).strip(),
            #         'ID': id + generation_idx
            #     })

    return cache_probs_all, cache_tokens_all, completions