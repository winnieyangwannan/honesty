import random
import json
import os
import argparse
from datasets import load_dataset
from pipeline.honesty_config_generation import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.activation_pca import plot_contrastive_activation_pca, plot_contrastive_activation_intervention_pca
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.submodules.activation_pca import get_activations
from pipeline.submodules.activation_pca import generate_get_contrastive_activations_and_plot_pca


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--checkpoint', type=int, required=False, default=None, help='Checkpoint for pyhia model family')
    parser.add_argument('--save_path', type=str, required=False, default=16)

    return parser.parse_args()


def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    category = cfg.data_category
    model_name = cfg.model_alias
    checkpoint = cfg.checkpoint
    if "pythia" in model_name:
        dataset_all = load_dataset("winnieyangwannan/mitchell-filtered-" +
                                   category + "-" + model_name + "-" + str(checkpoint))
    else:
        dataset_all = load_dataset("notrichardren/azaria-mitchell-diff-filtered-2")
    dataset = [row for row in dataset_all[f"{cfg.data_category}"]]
    dataset_train = random.sample(dataset, cfg.n_train)
    dataset_test = random.sample(dataset, cfg.n_test)

    return dataset_train, dataset_test


def contrastive_extraction_generation_and_plot_pca(cfg, model_base, dataset_train):
    tokenize_fn = model_base.tokenize_statements_fn
    statements_train = [row['claim'] for row in dataset_train]
    labels_train = [row['label'] for row in dataset_train]
    # 1. extract activations
    print("start extraction")
    generate_get_contrastive_activations_and_plot_pca(cfg,
                                                      model_base,
                                                      tokenize_fn,
                                                      statements_train,
                                                      save_activations=True,
                                                      save_plot=True,
                                                      labels=labels_train)
    print("done extraction")


def run_pipeline(model_path, save_path, checkpoint=None):
    """Run the full pipeline."""

    # 1. Load model
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias,
                 model_path=model_path,
                 save_path=save_path,
                 checkpoint=checkpoint)
    print(cfg)
    model_base = construct_model_base(cfg.model_path, checkpoint=cfg.checkpoint)

    # 2. Load and sample filtered datasets
    dataset_train, dataset_test = load_and_sample_datasets(cfg)

    # Generate candidate refusal directions
    contrastive_extraction_generation_and_plot_pca(cfg, model_base, dataset_train)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path,  checkpoint=args.checkpoint)
