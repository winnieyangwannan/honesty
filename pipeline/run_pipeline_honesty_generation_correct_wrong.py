import random
import os
import argparse
from datasets import load_dataset
from pipeline.configs.honesty_config_generation import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.activation_pca_correct_wrong import generate_get_contrastive_activations_and_plot_pca


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--batch_size', type=int, required=False, default=16)
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
    dataset_correct = random.sample(dataset, cfg.n_train)

    dataset_all = load_dataset("winnieyangwannan/mitchell-filtered-facts-llama-2-7b")
    dataset = [row for row in dataset_all[f"{cfg.data_category}"]]
    dataset_wrong = random.sample(dataset, cfg.n_train)

    return dataset_correct, dataset_wrong


def contrastive_extraction_generation_and_plot_pca(cfg, model_base, dataset_correct, dataset_wrong):
    tokenize_fn = model_base.tokenize_statements_fn
    statements_correct = [row['claim'] for row in dataset_correct]
    labels_correct = [row['label'] for row in dataset_correct]

    statements_wrong = [row['claim'] for row in dataset_wrong]
    labels_wrong = [row['label'] for row in dataset_wrong]
    # 1. extract activations
    print("start extraction")
    generate_get_contrastive_activations_and_plot_pca(cfg,
                                                      model_base,
                                                      tokenize_fn,
                                                      statements_correct,
                                                      statements_wrong,
                                                      save_activations=True,
                                                      save_plot=True,
                                                      labels_1=labels_correct,
                                                      labels_2=labels_wrong)
    print("done extraction")


def run_pipeline(model_path, save_path):
    """Run the full pipeline."""

    # 1. Load model
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, save_path=save_path)
    print(cfg)
    model_base = construct_model_base(cfg.model_path)

    # 2. Load and sample filtered datasets
    dataset_correct, dataset_wrong = load_and_sample_datasets(cfg)

    #
    # Generate candidate refusal directions
    contrastive_extraction_generation_and_plot_pca(cfg, model_base, dataset_correct, dataset_wrong)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path)
    # run_pipeline(model_path="Qwen/Qwen-1_8B-Chat")
