import torch
import random
import json
import os
import argparse
import pickle

from datasets import load_dataset
from torch.utils.data import DataLoader
from pipeline.submodules.layer_decode import cache_activation_and_decode, generate_and_decode
from pipeline.honesty_config_layer_decode import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.activation_pca import plot_contrastive_activation_pca
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
# from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss

def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--save_path', type=str, required=False, default=16)
    parser.add_argument('--cut_layer', type=int, required=False, default=16)

    return parser.parse_args()

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        train and test dataset
    """

    random.seed(42)

    dataset_all = load_dataset("notrichardren/azaria-mitchell-diff-filtered-2")
    dataset = [row for row in dataset_all[f"{cfg.data_category}"]]
    data = random.sample(dataset, cfg.n_train)

    return data

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


def get_layer_decode(cfg, model_base, dataset):

    artifact_dir = cfg.artifact_path()
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    model_name = cfg.model_alias

    # lying prompt template
    cache_probs_lying, cache_indices_lying = cache_activation_and_decode(
                                                                         model_base,
                                                                         dataset,
                                                                         cfg, system_type="lying",)

    # honest prompt template
    cache_probs_honest, cache_indices_honest = cache_activation_and_decode(
                                                                            model_base,
                                                                            dataset,
                                                                            cfg, system_type="honest")


def generate_and_get_layer_decode(cfg, model_base, dataset):

    artifact_dir = cfg.artifact_path()
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    model_name = cfg.model_alias
    cut_layer = cfg.cut_layer
    data_category = cfg.data_category

    # lying prompt template
    cache_probs_lying, cache_tokens_lying, completions_lying = generate_and_decode(
                                                                 model_base,
                                                                 dataset,
                                                                 cfg, system_type="lying",)

    # honest prompt template
    cache_probs_honest, cache_tokens_honest, completions_honest = generate_and_decode(
                                                                    model_base,
                                                                    dataset,
                                                                    cfg, system_type="honest")

    if not os.path.exists(os.path.join(cfg.artifact_path(), 'layer_decode')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'layer_decode'))
    cache_lying = {
        "cache_probs_lying": cache_probs_lying,
        "cache_tokens_lying": cache_tokens_lying
    }
    cache_honest = {
        "cache_probs_honest": cache_probs_honest,
        "cache_tokens_honest": cache_tokens_honest
    }
    with open(artifact_dir + os.sep + 'layer_decode' + os.sep + model_name + '_' + 'decode_lying_layer_'+str(cut_layer) + '.pkl', 'wb') as f:
        pickle.dump(cache_lying, f)
    with open(artifact_dir + os.sep + 'layer_decode' + os.sep + model_name + '_' + 'decode_honest_layer_'+str(cut_layer) + '.pkl', 'wb') as f:
        pickle.dump(cache_honest, f)
    with open(artifact_dir+os.sep+'layer_decode'+os.sep+f'{data_category}_completions_honest_layer_'+str(cut_layer) + '.json', "w") as f:
        json.dump(completions_honest, f, indent=4)
    with open(artifact_dir+os.sep+'layer_decode'+os.sep+f'{data_category}_completions_lying_layer_'+str(cut_layer) + '.json', "w") as f:
        json.dump(completions_lying, f, indent=4)

def run_pipeline(model_path, save_path, cut_layer=16):
    """Run the full pipeline."""

    # 1. Load model
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, save_path=save_path,
                 cut_layer=cut_layer)
    print(cfg)

    model_base = construct_model_base(cfg.model_path)

    # 2. Load and sample filtered datasets
    dataset = load_and_sample_datasets(cfg)

    # 3. Get Accuracy
    generate_and_get_layer_decode(cfg, model_base, dataset)
    

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path, cut_layer=args.cut_layer)
