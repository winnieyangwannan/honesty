import torch
from torch.nn.functional import softmax
from pipeline.utils.hook_utils import add_hooks
from pipeline.utils.hook_utils import get_activations_hook
from tqdm import tqdm
import os
import pandas as pd
import plotly.express as px

from pipeline.model_utils.model_base import ModelBase
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from pipeline.utils.hook_utils import get_and_cache_direction_ablation_input_pre_hook
from pipeline.utils.hook_utils import get_and_cache_diff_addition_input_pre_hook
from pipeline.utils.hook_utils import get_and_cache_direction_ablation_output_hook
from pipeline.utils.hook_utils import get_and_cache_activation_addition_output_hook


def plot_lying_honest_performance(cfg, model_performance):
    model_name = cfg.model_alias
    accuracy_honest = model_performance['accuracy_honest']
    accuracy_lying = model_performance['accuracy_lying']
    wrong_rate_lying = model_performance['wrong_rate_lying']
    wrong_rate_honest = model_performance['wrong_rate_honest']
    unexpected_lying_rate = model_performance['unexpected_lying_rate']
    unexpected_honest_rate = model_performance['unexpected_honest_rate']

    # data frame
    Correct = [accuracy_honest, accuracy_lying]
    Wrong = [wrong_rate_honest, wrong_rate_lying]
    Unexpected = [unexpected_honest_rate, unexpected_lying_rate]
    Role = ["Honest", "Lying"]

    # plot
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Correct", "Wrong", "Unexpected"])
    fig.add_trace(go.Bar(x=Role, y=Correct, showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(x=Role, y=Wrong, showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Bar(x=Role, y=Unexpected, showlegend=False,
    ), row=1, col=3)

    fig['layout']['yaxis']['title'] = 'Frequency'

    fig.update_layout(yaxis1=dict(range=[0, 1]))
    fig.update_layout(yaxis2=dict(range=[0, 1]))
    fig.update_layout(yaxis3=dict(range=[0, 1]))

    fig.show()

    return fig


def get_accuracy_and_unexpected(top_token_id, top_token_str, labels, true_token_id, false_token_id):
    label_strs = ["true" if label == 1 else "false" for label in labels]

    unexpected = []  # neither true nor false
    correct = []
    wrong = []
    for ii, label_str in enumerate(label_strs):
        if top_token_str[ii].lower().strip() in label_strs[ii]:
            correct.append(1)
            wrong.append(0)
            unexpected.append(0)

        elif top_token_str[ii].lower().strip() not in label_strs[ii]:
            if top_token_id[ii] in false_token_id or top_token_id[ii] in true_token_id:
                correct.append(0)
                wrong.append(1)
                unexpected.append(0)
            elif top_token_id[ii] not in true_token_id and top_token_id[ii] not in false_token_id:
                unexpected.append(1)
                correct.append(0)
                wrong.append(0)
                print("unexpected: ")
                print(top_token_str[ii])

    return correct, wrong, unexpected


def get_accuracy_and_probability(outputs, labels, tokenizer, true_token_id, false_token_id):
    probs = softmax(outputs, dim=-1)

    # Sort the probabilities in descending order and get the top tokens
    top_probs, top_indices = torch.topk(probs, k=5)  # Get top 5 for demonstration

    # Get the top token and its probability
    top_token_id = top_indices[:,0]
    top_token_str = tokenizer.batch_decode(top_token_id)
    top_token_prob = top_probs[:, 0]


    correct, wrong, unexpected = get_accuracy_and_unexpected(top_token_id, top_token_str,
                                                             labels,
                                                             true_token_id, false_token_id)
    return correct, top_token_prob, unexpected


def get_statement_accuracy(model_base, dataset, batch_size, system_type="honest"):

    model = model_base.model
    tokenizer = model_base.tokenizer
    tokenize_statements_fn = model_base.tokenize_statements_fn
    true_token_id = model_base.true_token_id
    false_token_id = model_base.false_token_id

    statements = [row['claim'] for row in dataset]
    labels = [row['label'] for row in dataset]

    accuracy_all = []
    probability_all = []
    unexpected_all = []
    for i in range(0, len(statements), batch_size):
        tokenized_prompt = tokenize_statements_fn(statements=statements[i:i+batch_size], system_type=system_type)

        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=[]):
            outputs = model(
                input_ids=tokenized_prompt.input_ids.to(model.device),
                attention_mask=tokenized_prompt.attention_mask.to(model.device),
            ).logits[:, -1, :]

        # take the last token digit
        accuracy, probability, unexpected = get_accuracy_and_probability(outputs=outputs,
                                                                         labels=labels[i:i+batch_size],
                                                                         tokenizer=tokenizer,
                                                                         true_token_id=true_token_id,
                                                                         false_token_id=false_token_id)
        for aa,pp,ee in zip(accuracy,probability,unexpected):
            accuracy_all.append(aa)
            probability_all.append(pp)
            unexpected_all.append(ee)

    return accuracy_all, probability_all, unexpected_all


def get_statement_accuracy_cache_activation(model_base, dataset, cfg, system_type="honest"):
    batch_size = cfg.batch_size
    n_samples = cfg.n_train
    sub_modules = cfg.sub_modules
    model_name = cfg.model_alias
    model = model_base.model
    block_modules = model_base.model_block_modules
    tokenizer = model_base.tokenizer
    tokenize_statements_fn = model_base.tokenize_statements_fn
    true_token_id = model_base.true_token_id
    false_token_id = model_base.false_token_id

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    statements = [row['claim'] for row in dataset]
    labels = [row['label'] for row in dataset]

    accuracy_all = []
    probability_all = []
    unexpected_all = []
    # we store the activations
    activations = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float64, device=model.device)

    for i in tqdm(range(0, len(statements), batch_size)):
        tokenized_prompt = tokenize_statements_fn(prompts=statements[i:i+batch_size], system_type=system_type)
        print("full prompt")
        print(tokenizer.decode(tokenized_prompt.input_ids[0]))
        fwd_pre_hooks = []
        if "mlp" in sub_modules:
            fwd_hooks = [(block_modules[layer].mlp,
                          get_activations_hook(layer=layer,
                                               cache=activations,
                                               positions=-1,
                                               batch_id=i,
                                               batch_size=batch_size)) for layer in range(n_layers)]
        elif "attn" in sub_modules:
            if "Qwen" in model_name:
                fwd_hooks = [(block_modules[layer].attn,
                              get_activations_hook(layer=layer,
                                                   cache=activations,
                                                   positions=-1,
                                                   batch_id=i,
                                                   batch_size=batch_size)) for layer in range(n_layers)]
            else:

                fwd_hooks = [(block_modules[layer].self_attn,
                              get_activations_hook(layer=layer,
                                                   cache=activations,
                                                   positions=-1,
                                                   batch_id=i,
                                                   batch_size=batch_size)) for layer in range(n_layers)]
        else:
            fwd_hooks = [(block_modules[layer],
                          get_activations_hook(layer=layer,
                                               cache=activations,
                                               positions=-1,
                                               batch_id=i,
                                               batch_size=batch_size)) for layer in range(n_layers)]
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            outputs = model(
                input_ids=tokenized_prompt.input_ids.to(model.device),
                attention_mask=tokenized_prompt.attention_mask.to(model.device),
            ).logits[:, -1, :]

        # take the last token digit
        accuracy, probability, unexpected = get_accuracy_and_probability(outputs=outputs,
                                                                         labels=labels[i:i+batch_size],
                                                                         tokenizer=tokenizer,
                                                                         true_token_id=true_token_id,
                                                                         false_token_id=false_token_id)
        for aa, pp, ee in zip(accuracy, probability, unexpected):
            accuracy_all.append(aa)
            probability_all.append(pp.cpu().tolist())
            unexpected_all.append(ee)

    return accuracy_all, probability_all, unexpected_all, activations


def get_performance_stats(cfg, first_gen_toks_honest, first_gen_str_honest, first_gen_toks_lying, first_gen_str_lying,
                          labels,
                          true_token_id, false_token_id
                          ):
    correct_honest, wrong_honest, unexpected_honest = get_accuracy_and_unexpected(first_gen_toks_honest, first_gen_str_honest,
                                                                    labels,
                                                                    true_token_id, false_token_id)
    correct_lying, wrong_lying, unexpected_lying = get_accuracy_and_unexpected(first_gen_toks_lying, first_gen_str_lying,
                                                                  labels,
                                                                  true_token_id, false_token_id)
    accuracy_lying = sum(correct_lying) / len(correct_lying)
    accuracy_honest = sum(correct_honest) / len(correct_honest)
    wrong_rate_lying = sum(wrong_lying) / len(wrong_lying)
    wrong_rate_honest = sum(wrong_honest) / len(wrong_honest)
    unexpected_lying_rate = sum(unexpected_lying) / len(unexpected_lying)
    unexpected_honest_rate = sum(unexpected_honest) / len(unexpected_honest)
    print(f"accuracy_lying: {accuracy_lying}")
    print(f"accuracy_honest: {accuracy_honest}")
    print(f"unexpected_lying: {unexpected_lying_rate}")
    print(f"unexpected_honest: {unexpected_honest_rate}")
    model_performance = {
        "performance_lying": correct_lying,
        "performance_honest": correct_honest,
        "accuracy_lying": accuracy_lying,
        "accuracy_honest": accuracy_honest,
        "wrong_rate_lying": wrong_rate_lying,
        "wrong_rate_honest": wrong_rate_honest,
        "unexpected_lying": unexpected_lying,
        "unexpected_honest": unexpected_honest,
        "unexpected_lying_rate": unexpected_lying_rate,
        "unexpected_honest_rate": unexpected_honest_rate
    }

    # Plot accuracy
    fig = plot_lying_honest_performance(cfg, model_performance)
    # save
    return model_performance, fig

