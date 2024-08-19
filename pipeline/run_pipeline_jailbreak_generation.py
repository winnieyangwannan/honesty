import random
import json
import os
import argparse
from pipeline.jailbreak_config_generation import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.activation_pca import plot_contrastive_activation_pca, plot_contrastive_activation_intervention_pca
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.submodules.activation_pca import get_activations
from pipeline.submodules.activation_pca import generate_get_contrastive_activations_and_plot_pca
from dataset.load_dataset import load_dataset_split, load_dataset
import numpy as np


def parse_arguments():

    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--jailbreak_type', type=str, required=False, default='evil_confidant')
    parser.add_argument('--few_shot', type=int, required=False, default=None)
    parser.add_argument('--save_path', type=str, required=False, default=16)

    return parser.parse_args()


def load_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    if "jailbreakbench" in cfg.evaluation_datasets:
        harmful_train = load_dataset(dataset_name='jailbreakbench')
    elif "harmful" in cfg.evaluation_datasets:
        harmful_train = load_dataset_split(dataset_name='harmful', split='train', instructions_only=False)
    harmless_train = load_dataset_split(harmtype='harmless', split='train', instructions_only=False)
    # harmful_val = load_dataset_split(harmtype='harmful', split='val', instructions_only=False)
    # harmless_val = load_dataset_split(harmtype='harmless', split='val', instructions_only=False)
    return harmful_train, harmless_train


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


def contrastive_extraction_generation_and_plot_pca(cfg, model_base, dataset_harmless, dataset_harmful,
                                                   jailbreak_type="evil_confidant"):
    tokenize_fn = model_base.tokenize_instructions_fn

    instructions_harmful = [x['instruction'] for x in dataset_harmful]
    categories_harmful = [x['category'] for x in dataset_harmful]
    instructions_harmful = instructions_harmful[:cfg.n_train]
    categoreis_harmful = categories_harmful[:cfg.n_train]

    instructions_harmless = [x['instruction'] for x in dataset_harmless]
    categories_harmless = [x['category'] for x in dataset_harmless]
    instructions_harmless = instructions_harmless[:cfg.n_train]
    categoreis_harmless = categories_harmless[:cfg.n_train]

    # append harmful and harmless instructions
    instructions_train = instructions_harmful + instructions_harmless
    labels_harmless = np.ones((len(instructions_harmless))).tolist()
    labels_harmful = np.zeros((len(instructions_harmful))).tolist()
    labels_train = labels_harmful + labels_harmless
    categories_train = categoreis_harmful + categoreis_harmless

    # 1. extract activations
    print("start extraction")
    generate_get_contrastive_activations_and_plot_pca(cfg,
                                                      model_base,
                                                      tokenize_fn,
                                                      instructions_train,
                                                      save_activations=True,
                                                      save_plot=True,
                                                      labels=labels_train,
                                                      contrastive_label=["HHH", jailbreak_type],
                                                      data_label=cfg.evaluation_datasets,
                                                      categories=categories_train)
    print("done extraction")


def run_pipeline(model_path, save_path,
                 jailbreak_type='evil_confidant', few_shot=None):
    """Run the full pipeline."""

    # 1. Load model
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias,
                 model_path=model_path,
                 save_path=save_path,
                 jailbreak_type=jailbreak_type,
                )
    print(cfg)
    model_base = construct_model_base(cfg.model_path,
                                      checkpoint=None)

    # 2. Load and sample filtered datasets
    harmful_train, harmless_train = load_datasets(cfg)
    # Filter datasets based on refusal scores
    # harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train,
    #                                                                        harmless_train, harmful_val, harmless_val)

    # 3. Generate candidate refusal directions
    contrastive_extraction_generation_and_plot_pca(cfg, model_base, harmless_train, harmful_train,
                                                   jailbreak_type=jailbreak_type)

    # 4. Evaluate
    # for dataset_name in cfg.evaluation_datasets:
    #     for contrastive_label in cfg.evalution_persona:
    #         evaluate_completions_and_save_results_for_dataset(cfg, contrastive_label, dataset_name,
    #                                                           eval_methodologies=cfg.jailbreak_eval_methodologies)
    #

if __name__ == "__main__":
    args = parse_arguments()
    print("jailbreak_type")
    print(args.jailbreak_type)

    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 jailbreak_type=args.jailbreak_type)
