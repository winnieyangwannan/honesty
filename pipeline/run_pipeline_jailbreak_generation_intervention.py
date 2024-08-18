import torch
import random
import numpy as np
import os
import argparse
import pickle
from pipeline.jailbreak_config_generation_intervention import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.submodules.save.activation_pca import get_contrastive_activations_and_plot_pca
from pipeline.analysis.stage_statistics import plot_stage_quantification_original_intervention
from pipeline.submodules.activation_pca_intervention import generate_with_intervention_contrastive_activations_pca
from dataset.load_dataset import load_dataset_split
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--save_path', type=str, required=False, default=" ")
    parser.add_argument('--source_layer', type=int, required=False, default=0)
    parser.add_argument('--target_layer_s', type=int, required=False, default=14)
    parser.add_argument('--target_layer_e', type=int, required=False, default=None)
    parser.add_argument('--intervention', type=str, required=False, default="skip_connection_mlp")

    return parser.parse_args()


def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = random.sample(load_dataset_split(harmtype='harmful', split='train', instructions_only=False), cfg.n_train)
    harmless_train = random.sample(load_dataset_split(harmtype='harmless', split='train', instructions_only=False), cfg.n_train)
    harmful_val = random.sample(load_dataset_split(harmtype='harmful', split='test', instructions_only=False), cfg.n_val)
    harmless_val = random.sample(load_dataset_split(harmtype='harmless', split='test', instructions_only=False), cfg.n_val)
    return harmful_train, harmless_train, harmful_val, harmless_val


# todo: incorporate this into pipeline
def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """

    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train,
                                                  model_base.tokenize_instructions_fn,
                                                  model_base.refusal_toks)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train,
                                                   model_base.tokenize_instructions_fn,
                                                   model_base.refusal_toks)
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


def contrastive_extraction_generation_intervention_and_plot_pca(cfg, model_base, 
                                                                harmful_train, harmless_train, 
                                                                harmful_val, harmless_val):

    model_name = cfg.model_alias
    data_category = cfg.data_category
    source_layer = cfg.source_layer
    target_layer_s = cfg.target_layer_s
    target_layer_e = cfg.target_layer_e
    artifact_dir = cfg.artifact_path()
    intervention = cfg.intervention
    
    instructions_harmful_train = [x['instruction'] for x in harmful_train]
    categories_harmful_train = [x['category'] for x in harmful_train]
    instructions_harmless_train = [x['instruction'] for x in harmless_train]
    categories_harmless_train = [x['category'] for x in harmless_train]

    instructions_harmful_val = [x['instruction'] for x in harmful_val]
    categories_harmful_val = [x['category'] for x in harmful_val]
    instructions_harmless_val = [x['instruction'] for x in harmless_val]
    categories_harmless_val = [x['category'] for x in harmless_val]

    
    # append harmful and harmless instructions for training data 
    instructions_train = instructions_harmful_train + instructions_harmless_train
    labels_harmless = np.ones((len(harmful_train))).tolist()
    labels_harmful = np.zeros((len(harmless_train))).tolist()
    labels_train = labels_harmful + labels_harmless
    categories_train = categories_harmful_train + categories_harmless_train
    
    # append harmful and harmless instructions for test data 
    instructions_val = instructions_harmful_val + instructions_harmless_val
    labels_harmless = np.ones((len(harmful_val))).tolist()
    labels_harmful = np.zeros((len(harmless_val))).tolist()
    labels_val = labels_harmful + labels_harmless
    categories_val = categories_harmful_val + categories_harmless_val

    # 1.1 extract activations
    print("start extraction")
    results = get_contrastive_activations_and_plot_pca(cfg=cfg,
                                                       model_base=model_base,
                                                       dataset=instructions_train,
                                                       labels=labels_train,
                                                       save_activations=False,
                                                       save_plot=True,
                                                       contrastive_label=["HHH", "BREAK"],
                                                       prompt_label=['harmless', 'harmful'])
    activations_positive = results['activations_positive']
    activations_negative = results['activations_negative']
    stage_stats_original = results['stage_stats']
    print("done extraction")

    # 2. get steering vector = get mean difference of the source layer
    if intervention != "no_intervention":
        mean_activation_positive = activations_positive.mean(dim=0)
        mean_activation_negative = activations_negative.mean(dim=0)

        if "positive_addition" in intervention or "positive_direction_ablation" in intervention or "positive_direction_addition" in intervention:
            mean_diff = mean_activation_positive - mean_activation_negative
        elif "negative_addition" in intervention or "negative_direction_ablation" in intervention or "negative_direction_addition" in intervention:
            mean_diff = mean_activation_negative - mean_activation_positive
        elif "skip_connection_mlp" or "skip_connection_attn" in intervention:
            mean_diff = 0.000001*torch.ones_like(mean_activation_positive) # 0 ablation
        elif "positive_projection":
            activations_positive = activations_positive[:cfg.n_train] - activations_positive[cfg.n_train:]
            activations_negative = activations_negative[:cfg.n_train] - activations_negative[cfg.n_train:]
            mean_diff = activations_positive - activations_negative

        # 3.1  generate with adding steering vector and get activations
        intervention_results = generate_with_intervention_contrastive_activations_pca(cfg,
                                                                                      model_base,
                                                                                      instructions_val,
                                                                                      activations_positive,
                                                                                      activations_negative,
                                                                                      mean_diff=mean_diff[source_layer, :],
                                                                                      labels_ori=labels_train,
                                                                                      labels_int=labels_val,
                                                                                      save_activations=False,
                                                                                      contrastive_label=["HHH", "BREAK"],
                                                                                      categories=categories_val)

        # 4.1 Compare and plot stage statistics with and without intervention
        stage_stats_intervention = intervention_results['stage_stats_intervention'] 
        save_path = os.path.join(artifact_dir, intervention, "stage_stats")
        n_layers = activations_negative.shape[1]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plot_stage_quantification_original_intervention(cfg, stage_stats_original, stage_stats_intervention,
                                                        n_layers, save_path)

        # 4.2 save stage statistics with and without intervention
        stage_stats = {
            "stage_stats_original": stage_stats_original,
            "stage_stats_intervention": stage_stats_intervention,
        }
        with open(save_path + os.sep + model_name + '_' + f'{data_category}' +
                  '_stage_stats_' + intervention + '_' + str(source_layer) + '_' + str(target_layer_s) +
                  '_' + str(target_layer_e) + '.pkl', "wb") as f:
            pickle.dump(stage_stats, f)


def run_pipeline(model_path, save_path, intervention, source_layer, target_layer_s, target_layer_e):
    """Run the full pipeline."""

    # 1. Load model
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, save_path=save_path,
                 intervention=intervention,
                 source_layer=source_layer,
                 target_layer_s=target_layer_s, target_layer_e=target_layer_e)
    print(cfg)
    model_base = construct_model_base(cfg.model_path)

    # 2. Load and filter datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(cfg)

    # todo: filter data in the future
    # harmful_train, harmless_train, harmful_val, harmless_val = filter_data(cfg, model_base,
    #                                                                        harmful_train,
    #                                                                        harmless_train,
    #                                                                        harmful_val,
    #                                                                        harmless_val)

    # 3. Generate candidate refusal directions
    contrastive_extraction_generation_intervention_and_plot_pca(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 intervention=args.intervention,
                 source_layer=args.source_layer,
                 target_layer_s=args.target_layer_s, target_layer_e=args.target_layer_e)
