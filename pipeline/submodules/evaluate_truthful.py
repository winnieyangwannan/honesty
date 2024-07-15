import torch
from torch.nn.functional import softmax
from pipeline.utils.hook_utils import add_hooks
from pipeline.utils.hook_utils import get_activations_pre_hook
from tqdm import tqdm

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


def get_accuracy_and_unexpected(top_token_id, top_token_str, labels, true_token_id, false_token_id):
    label_strs = ["true" if label == 1 else "false" for label in labels]

    unexpected = []  # neither true nor false
    correct = []
    for ii, label_str in enumerate(label_strs):
        if label_str == top_token_str[ii].lower().strip():
            correct.append(1)
        else:
            correct.append(0)

        if top_token_id[ii] not in true_token_id and top_token_id[ii] not in false_token_id:
            unexpected.append(1)
            print("unexpected: ")
            print(top_token_str[ii])
        else:
            unexpected.append(0)
    return correct, unexpected


def get_accuracy_and_probability(outputs, labels, tokenizer, true_token_id, false_token_id):
    probs = softmax(outputs, dim=-1)

    # Sort the probabilities in descending order and get the top tokens
    top_probs, top_indices = torch.topk(probs, k=5)  # Get top 5 for demonstration

    # Get the top token and its probability
    top_token_id = top_indices[:,0]
    top_token_str = tokenizer.batch_decode(top_token_id)
    top_token_prob = top_probs[:, 0]


    correct, unexpected = get_accuracy_and_unexpected(top_token_id, top_token_str,
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
        fwd_pre_hooks = [(block_modules[layer],
                          get_activations_pre_hook(layer=layer,
                                                   cache=activations,
                                                   positions=-1,
                                                   batch_id=i,
                                                   batch_size=batch_size)) for layer in range(n_layers)]
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
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
            probability_all.append(pp)
            unexpected_all.append(ee)

    return accuracy_all, probability_all, unexpected_all, activations

def plot_lying_honest_accuracy():
    pass


