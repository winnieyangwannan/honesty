import torch
import random
import json
import os
import argparse
import csv
from datasets import load_dataset
from typing import List, Tuple, Callable

from pipeline.honesty_config_generation_skip_connection import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.submodules.activation_pca import get_activations, plot_contrastive_activation_pca
from pipeline.submodules.activation_pca_intervention import plot_contrastive_activation_intervention_pca, \
    get_intervention_activations_and_generation
from pipeline.submodules.evaluate_truthful import get_accuracy_and_unexpected


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--save_path', type=str, required=False, default=" ")
    parser.add_argument('--target_layer', nargs="*", type=int, required=False, default=[14, 15, 16])

    return parser.parse_args()


def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)

    dataset_all = load_dataset("notrichardren/azaria-mitchell-diff-filtered-2")
    dataset = [row for row in dataset_all[f"{cfg.data_category}"]]
    dataset_train = random.sample(dataset, cfg.n_train)
    dataset_test = random.sample(dataset, cfg.n_test)

    return dataset_train, dataset_test


def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """

    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn,
                                                  model_base.refusal_toks)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train,
                                                   model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn,
                                                model_base.refusal_toks)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn,
                                                 model_base.refusal_toks)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)

    return harmful_train, harmless_train, harmful_val, harmless_val


def get_contrastive_activations_and_plot_pca(cfg,
                                             model_base,
                                             dataset,
                                             labels=None):
    artifact_dir = cfg.artifact_path()
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    model_name = cfg.model_alias
    data_category = cfg.data_category
    tokenize_fn = model_base.tokenize_statements_fn
    activations_lying = get_activations(cfg, model_base, dataset,
                                        tokenize_fn,
                                        positions=[-1],
                                        system_type="lying")

    activations_honest = get_activations(cfg, model_base, dataset,
                                         tokenize_fn,
                                         positions=[-1],
                                         system_type="honest")

    # plot pca
    n_layers = model_base.model.config.num_hidden_layers
    fig = plot_contrastive_activation_pca(activations_honest, activations_lying,
                                          n_layers, contrastive_label=["honest", "lying"],
                                          labels=labels)
    fig.write_html(artifact_dir + os.sep + model_name + '_' + f'{data_category}' + '_activation_pca.html')

    return activations_honest, activations_lying


def generate_with_intervention_cache_contrastive_activations_and_plot_pca(cfg,
                                                                          model_base,
                                                                          dataset,
                                                                          activations_honest,
                                                                          activations_lying,
                                                                          mean_diff=None,
                                                                          labels=None):
    artifact_dir = cfg.artifact_path()
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    intervention = cfg.intervention
    source_layer = cfg.source_layer
    target_layer = cfg.target_layer
    model_name = cfg.model_alias
    data_category = cfg.data_category
    n_layers = model_base.model.config.num_hidden_layers
    tokenize_fn = model_base.tokenize_statements_fn
    true_token_id = model_base.true_token_id
    false_token_id = model_base.false_token_id

    # 1. Generation with Intervention
    intervention_activations_honest, intervention_completions_honest, first_gen_toks_honest, first_gen_str_honest = get_intervention_activations_and_generation(
        cfg, model_base, dataset,
        tokenize_fn,
        mean_diff=mean_diff,
        positions=[-1],
        max_new_tokens=64,
        system_type="honest",
        target_layer=target_layer,
        labels=labels)
    intervention_activations_lying, intervention_completions_lying, first_gen_toks_lying, first_gen_str_lying = get_intervention_activations_and_generation(
        cfg, model_base, dataset,
        tokenize_fn,
        mean_diff=mean_diff,
        positions=[-1],
        max_new_tokens=64,
        system_type="lying",
        target_layer=target_layer,
        labels=labels)

    # 2. save completions
    if not os.path.exists(os.path.join(cfg.artifact_path(), intervention)):
        os.makedirs(os.path.join(cfg.artifact_path(), intervention))

    with open(
            artifact_dir + os.sep + intervention + os.sep + f'{data_category}_{intervention}_completions_honest_layer_{source_layer}_{target_layer}.json',
            "w") as f:
        json.dump(intervention_completions_honest, f, indent=4)
    with open(
            artifact_dir + os.sep + intervention + os.sep + f'{data_category}_{intervention}_completions_lying_layer_{source_layer}_{target_layer}.json',
            "w") as f:
        json.dump(intervention_completions_lying, f, indent=4)

    # 3. pca with and without intervention, plot and save pca
    contrastive_label = ["honest", "lying", "honest_"+intervention, "lying_"+intervention]
    fig = plot_contrastive_activation_intervention_pca(activations_honest,
                                                       activations_lying,
                                                       intervention_activations_honest,
                                                       intervention_activations_lying,
                                                       n_layers,
                                                       contrastive_label,
                                                       labels)
    fig.write_html(
        artifact_dir + os.sep + intervention + os.sep + model_name + '_'  + intervention +
        '_pca_layer_' + str(source_layer) + '_' + str(target_layer) + '.html')

    # 4. get accuracy
    correct_honest, unexpected_honest = get_accuracy_and_unexpected(first_gen_toks_honest,
                                                                    first_gen_str_honest,
                                                                    labels,
                                                                    true_token_id, false_token_id)

    correct_lying, unexpected_lying = get_accuracy_and_unexpected(first_gen_toks_lying,
                                                                  first_gen_str_lying,
                                                                  labels,
                                                                  true_token_id, false_token_id)
    accuracy_lying = sum(correct_lying) / len(correct_lying)
    accuracy_honest = sum(correct_honest) / len(correct_honest)
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
        "unexpected_lying": unexpected_lying,
        "unexpected_honest": unexpected_honest,
        "unexpected_lying_rate": unexpected_lying_rate,
        "unexpected_honest_rate": unexpected_honest_rate
    }

    with open(artifact_dir + os.sep + intervention + os.sep + model_name + '_' + f'model_performance_layer_{source_layer}_{target_layer}.csv',
              'w') as f:
        w = csv.writer(f)
        w.writerows(model_performance.items())


def contrastive_extraction_generation_skip_generation_and_plot_pca(cfg, model_base, dataset_train, dataset_test):

    statements_train = [row['claim'] for row in dataset_train]
    statements_test = [row['claim'] for row in dataset_test]
    labels_train = [row['label'] for row in dataset_train]
    labels_test = [row['label'] for row in dataset_test]
    # 1. extract activations
    print("start extraction")
    activations_honest, activations_lying = get_contrastive_activations_and_plot_pca(cfg=cfg,
                                                                                     model_base=model_base,
                                                                                     dataset=statements_train,
                                                                                     labels=labels_train)
    print("done extraction")

    # # 2. get steering vector = get mean difference of the source layer
    mean_activation_honest = activations_honest.mean(dim=0)
    mean_activation_lying = activations_lying.mean(dim=0)
    mean_diff = mean_activation_honest - mean_activation_lying

    # 2. generate with adding steering vector
    source_layer = cfg.source_layer
    generate_with_intervention_cache_contrastive_activations_and_plot_pca(cfg,
                                                                          model_base,
                                                                          statements_test,
                                                                          activations_honest,
                                                                          activations_lying,
                                                                          mean_diff=mean_diff[source_layer],
                                                                          labels=labels_test)


def run_pipeline(model_path, save_path, target_layer):
    """Run the full pipeline."""

    # 1. Load model
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, save_path=save_path,
                 target_layer=target_layer)
    print(cfg)
    model_base = construct_model_base(cfg.model_path)

    # 2. Load and sample filtered datasets
    dataset_train, dataset_test = load_and_sample_datasets(cfg)

    # 3. Generate candidate refusal directions
    contrastive_extraction_generation_skip_generation_and_plot_pca(cfg, model_base, dataset_train, dataset_test)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path, target_layer=args.target_layer)
    # run_pipeline(model_path="Qwen/Qwen-1_8B-Chat")
