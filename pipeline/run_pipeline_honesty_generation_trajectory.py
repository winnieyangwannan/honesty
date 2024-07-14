import torch
import random
import json
import os
import argparse
import pickle
from datasets import load_dataset
from pipeline.honesty_config_generation_trajectory import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.submodules.activation_pca import plot_contrastive_activation_pca_with_trajectory, generate_and_get_activations
from pipeline.submodules.activation_pca import generate_and_get_activation_trajectory


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--save_path', type=str, required=False, default=16)

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


def generate_get_contrastive_activations_and_trajectory(cfg, model_base, dataset, labels=None):

    artifact_dir = cfg.artifact_path()
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    model_name = cfg.model_alias
    data_category = cfg.data_category
    max_new_tokens = cfg.max_new_tokens
    tokenize_fn = model_base.tokenize_statements_fn

    # 1. Select one prompt and visualize its answer trajectory
    # lying
    activations_lying_trajectory, completions_lying_trajectory = generate_and_get_activation_trajectory(cfg, model_base,
                                                                                                        dataset,
                                                                                                        tokenize_fn,
                                                                                                        positions=[-1],
                                                                                                        max_new_tokens=max_new_tokens,
                                                                                                        system_type="lying",
                                                                                                        labels=labels,
                                                                                                        cache_type="trajectory")
    # honest
    activations_honest_trajectory, completions_honest_trajectory = generate_and_get_activation_trajectory(cfg, model_base,
                                                                                                        dataset,
                                                                                                        tokenize_fn,
                                                                                                        positions=[-1],
                                                                                                        max_new_tokens=max_new_tokens,
                                                                                                        system_type="honest",
                                                                                                        labels=labels,
                                                                                                        cache_type="trajectory")
    # save completions
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))
    with open(f'{cfg.artifact_path()}'+os.sep+'completions'+os.sep+f'{data_category}_completions_lying_trajectory.json', "w") as f:
        json.dump(completions_lying_trajectory, f)
    with open(f'{cfg.artifact_path()}'+os.sep+'completions'+os.sep+f'{data_category}_completions_honest_trajectory.json', "w") as f:
        json.dump(completions_honest_trajectory, f)


    # 2. generate and get activation of last token of prompt
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

    # 3. Save PCA activations
    activation_pca = {
        "activations_honest": activations_honest,
        "activations_lying": activations_lying,
        "activations_honest_trajectory": activations_honest_trajectory,
        "activations_lying_trajectory": activations_lying_trajectory,
    }
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'trajectories')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'trajectories'))
    with open(artifact_dir + os.sep + 'trajectories' + os.sep + model_name + '_' + 'activation_pca.pkl','wb') as f:
        pickle.dump(activation_pca, f)

    return activation_pca, completions_honest, completions_lying


def contrastive_extraction_generation_and_plot_pca(cfg, model_base, dataset_train, dataset_test):

    # 1. extract activations
    print("start extraction")
    statements_train = [row['claim'] for row in dataset_train]
    labels_train = [row['label'] for row in dataset_train]
    activation_pca, completions_honest, completions_lying = generate_get_contrastive_activations_and_trajectory(
                                                                                                  cfg,
                                                                                                  model_base,
                                                                                                  statements_train,
                                                                                                  labels=labels_train)
    print("done extraction")

    # # 2.plot pca
    model_name = cfg.model_alias
    artifact_dir = cfg.artifact_path()
    n_layers = model_base.model.config.num_hidden_layers
    data_category = cfg.data_category
    tokenizer = model_base.tokenizer
    # load activation data
    file = open(artifact_dir + os.sep + 'trajectories' + os.sep + model_name + '_' + 'activation_pca.pkl', 'rb')
    activation_pca = pickle.load(file)
    file.close()
     # load completion data
    file = open(artifact_dir + os.sep + 'completions' + os.sep + f'{data_category}_completions_lying_trajectory.json', 'rb')
    completions_honest = json.load(file)
    file.close()
    file = open(artifact_dir + os.sep + 'completions' + os.sep + f'{data_category}_completions_lying_trajectory.json', 'rb')
    completions_lying = json.load(file)
    file.close()
    str_honest = tokenizer.tokenize(completions_honest[0]['response'])
    str_lie = tokenizer.tokenize(completions_lying[0]['response'])
    activations_lie = activation_pca['activations_honest']
    activations_honest = activation_pca['activations_lying']
    activation_trajectory_honest = activation_pca['activations_honest_trajectory']
    activation_trajectory_lie = activation_pca['activations_lying_trajectory']

    fig = plot_contrastive_activation_pca_with_trajectory(activations_honest, activations_lie,
                                                          activation_trajectory_honest, activation_trajectory_lie,
                                                          n_layers,
                                                          str_honest, str_lie,
                                                          contrastive_label=["honest", "lying", "trajectory_honest","trajectory_lying"],
                                                          labels=labels_train)
    fig.write_html(artifact_dir + os.sep + 'trajectories' + os.sep + model_name + '_' + 'activations_pca_trajectory.html')


def run_pipeline(model_path, save_path):
    """Run the full pipeline."""

    # 1. Load model
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, save_path=save_path)
    print(cfg)
    artifact_path = cfg.artifact_path()

    model_base = construct_model_base(cfg.model_path)

    # 2. Load and sample filtered datasets
    dataset_train, dataset_test = load_and_sample_datasets(cfg)

    #
    # Generate candidate refusal directions
    contrastive_extraction_generation_and_plot_pca(cfg, model_base, dataset_train, dataset_test)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path)
