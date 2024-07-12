import torch
import random
import json
import os
import argparse


from datasets import load_dataset
from torch.utils.data import DataLoader
from pipeline.submodules.evaluate_truthful import plot_lying_honest_accuracy, get_statement_accuracy_cache_activation
from pipeline.honesty_pipeline.honesty_config_generation import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.activation_pca import plot_contrastive_activation_pca, plot_contrastive_activation_intervention_pca
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.activation_pca import get_activations, get_intervention_activations_and_generation, generate_and_get_activations
from pipeline.submodules.activation_pca import generate_and_get_activation_trajectory
# from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--save_path', type=int, required=False, default=16)

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
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)
    
    return harmful_train, harmless_train, harmful_val, harmless_val


def generate_get_contrastive_activations_and_plot_pca(cfg, model_base, tokenize_fn, dataset, labels=None):

    artifact_dir = cfg.artifact_path()
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    model_name = cfg.model_alias
    data_category = cfg.data_category
    max_new_tokens = cfg.max_new_tokens
    tokenize_fn = model_base.tokenize_statements_fn

    activations_lying, completions_lying = generate_and_get_activations(cfg, model_base, dataset,
                                                                        tokenize_fn,
                                                                        positions=[-1],
                                                                        max_new_tokens=max_new_tokens,
                                                                        system_type="lying",
                                                                        labels=labels)

    activations_honest, completions_honest = generate_and_get_activations(cfg, model_base, dataset,
                                                                          tokenize_fn,
                                                                          positions=[-1],
                                                                          max_new_tokens=max_new_tokens,
                                                                          system_type="honest",
                                                                          labels=labels)

    # save completions
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    with open(f'{cfg.artifact_path()}'+os.sep+'completions'+os.sep+f'{data_category}_completions_honest.json', "w") as f:
        json.dump(completions_honest, f, indent=4)
    with open(f'{cfg.artifact_path()}'+os.sep+'completions'+os.sep+f'{data_category}_completions_lying.json', "w") as f:
        json.dump(completions_lying, f, indent=4)

    # plot pca
    n_layers = model_base.model.config.num_hidden_layers
    fig = plot_contrastive_activation_pca(activations_honest, activations_lying,
                                          n_layers, contrastive_label=["honest", "lying"],
                                          labels=labels)
    fig.write_html(artifact_dir + os.sep + model_name + '_' + 'activation_pca.html')

    return activations_honest, activations_lying


def get_contrastive_activations_and_plot_pca(cfg, model_base, tokenize_fn, dataset, labels=None):

    artifact_dir = cfg.artifact_path()
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    model_name = cfg.model_alias
    batch_size = cfg.batch_size

    model = model_base.model
    block_modules = model_base.model_block_modules

    activations_honest = get_activations(model, block_modules,
                                         tokenize_fn,
                                         dataset,
                                         batch_size=batch_size, positions=-1,
                                         system_type="honest")
    activations_lying = get_activations(model, block_modules,
                                        tokenize_fn,
                                        dataset,
                                        batch_size=batch_size, positions=-1,
                                        system_type="lying")
    # plot pca
    n_layers = model_base.model.config.num_hidden_layers

    fig = plot_contrastive_activation_pca(activations_honest, activations_lying,
                                          n_layers, contrastive_label=["honest","lying"],
                                          labels=labels)

    fig.write_html(artifact_dir + os.sep + model_name + '_' + 'activation_pca.html')

    return activations_honest, activations_lying


def generate_with_intervention_cache_contrastive_activations_and_plot_pca(cfg, model_base,
                                                                          tokenize_fn,
                                                                          dataset,
                                                                          mean_diff,
                                                                          activations_honest,
                                                                          activations_lying,
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

    ablation_activations_honest, ablation_completions_honest = get_intervention_activations_and_generation(
                                                      cfg, model_base, dataset,
                                                      tokenize_fn,
                                                      mean_diff,
                                                      positions=[-1],
                                                      max_new_tokens=64,
                                                      system_type="honest",
                                                      labels=labels)

    ablation_activations_lying, ablation_completions_lying = get_intervention_activations_and_generation(
                                                      cfg, model_base, dataset,
                                                      tokenize_fn,
                                                      mean_diff,
                                                      positions=[-1],
                                                      max_new_tokens=64,
                                                      system_type="lying",
                                                      labels=labels)
    # save completions
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    with open(f'{cfg.artifact_path()}'+os.sep+'completions'+os.sep+f'{data_category}_{intervention}_completions_honest_layer_{target_layer}_{source_layer}.json', "w") as f:
        json.dump(ablation_completions_honest, f, indent=4)
    with open(f'{cfg.artifact_path()}'+os.sep+'completions'+os.sep+f'{data_category}_{intervention}_completions_lying_layer_{target_layer}_{source_layer}.json', "w") as f:
        json.dump(ablation_completions_lying, f, indent=4)

    # pca with and without intervention, plot and save pca

    contrastive_label = ["honest", "lying", "honest_ablation", "lying_ablation"]
    fig = plot_contrastive_activation_intervention_pca(activations_honest,
                                                       activations_lying,
                                                       ablation_activations_honest,
                                                       ablation_activations_lying,
                                                       n_layers,
                                                       contrastive_label,
                                                       labels)
    fig.write_html(artifact_dir + os.sep + model_name + '_' + 'refusal_generation_activation_'+intervention +
                   '_pca_layer_'+str(source_layer)+'_' + str(target_layer)+'.html')


def contrastive_extraction_generation_and_plot_pca(cfg, model_base, dataset_train, dataset_test):
    tokenize_fn = model_base.tokenize_statements_fn
    statements_train = [row['claim'] for row in dataset_train]
    statements_test = [row['claim'] for row in dataset_test]
    labels_train = [row['label'] for row in dataset_train]
    labels_test = [row['label'] for row in dataset_test]
    # 1. extract activations
    print("start extraction")
    activations_honest, activations_lying = generate_get_contrastive_activations_and_plot_pca(cfg,
                                                                                             model_base,
                                                                                             tokenize_fn,
                                                                                             statements_train,
                                                                                             labels=labels_train)
    print("done extraction")
    # 2. get steering vector = get mean difference of the source layer
    mean_activation_honest = activations_honest.mean(dim=0)
    mean_activation_lying = activations_lying.mean(dim=0)
    mean_diff = mean_activation_lying-mean_activation_honest



def run_pipeline(model_path, save_path, batch_size=16):
    """Run the full pipeline."""

    # 1. Load model
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, save_path=save_path)
    print(cfg)
    model_base = construct_model_base(cfg.model_path)

    # 2. Load and sample filtered datasets
    dataset_train, dataset_test = load_and_sample_datasets(cfg)

    #
    # Generate candidate refusal directions
    contrastive_extraction_generation_and_plot_pca(cfg, model_base, dataset_train, dataset_test)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path, batch_size=args.batch_size)
    # run_pipeline(model_path="Qwen/Qwen-1_8B-Chat")
