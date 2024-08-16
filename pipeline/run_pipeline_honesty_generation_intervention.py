import torch
import random
import os
import argparse
import pickle
from datasets import load_dataset
from pipeline.honesty_config_generation_intervention import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.submodules.save.activation_pca import get_contrastive_activations_and_plot_pca
from pipeline.analysis.stage_statistics import plot_stage_quantification_original_intervention
from pipeline.submodules.activation_pca_intervention import generate_with_intervention_contrastive_activations_pca


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


def contrastive_extraction_generation_intervention_and_plot_pca(cfg, model_base, dataset_train, dataset_test):

    model_name = cfg.model_alias
    data_category = cfg.data_category
    source_layer = cfg.source_layer
    target_layer_s = cfg.target_layer_s
    target_layer_e = cfg.target_layer_e
    artifact_dir = cfg.artifact_path()
    intervention = cfg.intervention
    save_path = os.path.join(artifact_dir, intervention, "stage_stats")
    statements_train = [row['claim'] for row in dataset_train]
    statements_val = [row['claim'] for row in dataset_test]
    labels_train = [row['label'] for row in dataset_train]
    labels_val = [row['label'] for row in dataset_test]

    # 1.1 extract activations
    print("start extraction")
    results = get_contrastive_activations_and_plot_pca(cfg=cfg,
                                                       model_base=model_base,
                                                       dataset=statements_train,
                                                       labels=labels_train,
                                                       save_activations=False,
                                                       save_plot=False,
                                                       contrastive_label=["honest", "lying"])
    activations_positive = results['activations_positive']
    activations_negative = results['activations_negative']
    stage_stats_original = results['stage_stats']
    print("done extraction")

    if intervention != "no_intervention":
        # 2. get steering vector = get mean difference of the source layer
        mean_activation_honest = activations_positive.mean(dim=0)
        mean_activation_lying = activations_negative.mean(dim=0)

        if "honest_addition" in intervention or "honest_direction_ablation" in intervention or "honest_direction_addition" in intervention:
            mean_diff = mean_activation_honest - mean_activation_lying
        elif "lying_addition" in intervention or "lying_direction_ablation" in intervention or "lying_direction_addition" in intervention:
            mean_diff = mean_activation_lying - mean_activation_honest
        elif "skip_connection_mlp" or "skip_connection_attn" in intervention:
            mean_diff = 0.000001*torch.ones_like(mean_activation_honest) # 0 ablation

        # 3.1  generate with adding steering vector and get activations
        intervention_results = generate_with_intervention_contrastive_activations_pca(cfg,
                                                                                      model_base,
                                                                                      statements_val,
                                                                                      activations_positive,
                                                                                      activations_negative,
                                                                                      mean_diff=mean_diff[source_layer,:],
                                                                                      labels=labels_val,
                                                                                      save_activations=False,
                                                                                      contrastive_label=["honest", "lying"])

        # 4.1 Compare and plot stage statistics with and without intervention
        stage_stats_intervention = intervention_results['stage_stats_intervention']
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

    # 2. Load and sample filtered datasets
    dataset_train, dataset_test = load_and_sample_datasets(cfg)

    # 3. Generate candidate refusal directions
    contrastive_extraction_generation_intervention_and_plot_pca(cfg, model_base, dataset_train, dataset_test)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 intervention=args.intervention,
                 source_layer=args.source_layer,
                 target_layer_s=args.target_layer_s, target_layer_e=args.target_layer_e)
