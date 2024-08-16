import random
import json
import os
import argparse
from pipeline.jailbreak_config_generation import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.submodules.save.activation_pca import generate_get_contrastive_activations_and_plot_pca
from dataset.load_dataset import load_dataset_split, load_dataset
import numpy as np
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak


def parse_arguments():

    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--checkpoint', type=int, required=False, default=None, help='Checkpoint for pyhia model family')
    parser.add_argument('--few_shot', type=int, required=False, default=None)
    parser.add_argument('--save_path', type=str, required=False, default=16)

    return parser.parse_args()


def load_datasets():
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = load_dataset(dataset_name='jailbreakbench')
    # harmful_train = load_dataset_split(dataset_name='harmful', split='train', instructions_only=False)
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


def evaluate_completions_and_save_results_for_dataset(cfg, contrastive_label, dataset_name, eval_methodologies, few_shot=None):
    """Evaluate completions and save results for a dataset."""
    # with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
    with open(f'{cfg.artifact_path()}' + os.sep + 'completions' + os.sep + f'{dataset_name}' +
               '_completions_' + contrastive_label +'.json',
               "r") as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_evaluations.json"),
    )

    with open(f'{cfg.artifact_path()}' + os.sep + 'completions' + os.sep + f'{dataset_name}' +
               '_evaluations_' + contrastive_label +'.json', "w") as f:
        json.dump(evaluation, f, indent=4)


def contrastive_extraction_generation_and_plot_pca(cfg, model_base, dataset_harmless, dataset_harmful):
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
                                                      contrastive_label=["HHH", "BREAK"],
                                                      data_label=["jailbreakbench", "harmless"],
                                                      categories=categories_train)
    print("done extraction")


def run_pipeline(model_path, save_path,
                 checkpoint=None, few_shot=None):
    """Run the full pipeline."""

    # 1. Load model
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias,
                 model_path=model_path,
                 save_path=save_path,
                 checkpoint=checkpoint,
                 few_shot=few_shot)
    print(cfg)
    model_base = construct_model_base(cfg.model_path,
                                      checkpoint=cfg.checkpoint,
                                      )

    # 2. Load and sample filtered datasets
    harmful_train, harmless_train = load_datasets()
    # Filter datasets based on refusal scores
    # harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base, harmful_train,
    #                                                                        harmless_train, harmful_val, harmless_val)

    # 3. Generate candidate refusal directions
    contrastive_extraction_generation_and_plot_pca(cfg, model_base, harmless_train, harmful_train)

    # 4. Evaluate
    for dataset_name in cfg.evaluation_datasets:
        for contrastive_label in cfg.evalution_persona:
            evaluate_completions_and_save_results_for_dataset(cfg, contrastive_label, dataset_name,
                                                              eval_methodologies=cfg.jailbreak_eval_methodologies)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 checkpoint=args.checkpoint, few_shot=args.few_shot)
