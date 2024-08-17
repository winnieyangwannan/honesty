import torch
import random
import json
import os
import argparse
import pickle
import pandas as pd
import plotly.express as px
import plotly.io as pio

from datasets import load_dataset
from torch.utils.data import DataLoader
from pipeline.submodules.evaluate_truthful import plot_lying_honest_performance, get_statement_accuracy_cache_activation
from pipeline.honesty_config_performance import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.activation_pca_correct_wrong import plot_contrastive_activation_pca
from pipeline.plot.plot_some_layer_pca import plot_one_layer_3d
from pipeline.analysis.stage_statistics import get_state_quantification

# from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--save_path', type=str, required=False, default=16)
    parser.add_argument('--batch_size', type=int, required=False, default=16)

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


def get_contrastive_accuracy_and_plot(cfg, model_base, dataset_1, dataset_2,
                                      contrastive_name=['correct', 'wrong'],
                                      save_plot=True):

    artifact_dir = cfg.artifact_path()
    intervention = cfg.intervention
    model_name = cfg.model_alias
    n_layers = model_base.model.config.num_hidden_layers

    labels_1 = [row['label'] for row in dataset_1]
    labels_2 = [row['label'] for row in dataset_2]

    # honest prompt template
    performance_1, probability_1, unexpected_1, activations_1 = get_statement_accuracy_cache_activation(
                                                                                model_base,
                                                                                dataset_1,
                                                                                cfg, system_type="honest",)

    # lying prompt template
    performance_2, probability_2, unexpected_2, activations_2 = get_statement_accuracy_cache_activation(
                                                                                 model_base,
                                                                                 dataset_2,
                                                                                 cfg, system_type="honest",)



    # accuracy_lying = sum(performance_2)/len(performance_2)
    # accuracy_honest = sum(performance_1) / len(performance_1)
    # unexpected_2_rate = sum(unexpected_2)/len(unexpected_2)
    # unexpected_1_rate = sum(unexpected_1)/len(unexpected_1)
    # print(f"accuracy_lying: {accuracy_lying}")
    # print(f"accuracy_honest: {accuracy_honest}")
    # print(f"unexpected_2: {unexpected_2_rate}")
    # print(f"unexpected_1: {unexpected_1_rate}")
    #
    # model_performance = {
    #     "performance_2": performance_2,
    #     "performance_1": performance_1,
    #     "accuracy_lying": accuracy_lying,
    #     "accuracy_honest": accuracy_honest,
    #     "unexpected_2": unexpected_2,
    #     "unexpected_1": unexpected_1,
    #     "unexpected_2_rate": unexpected_2_rate,
    #     "unexpected_1": unexpected_1
    # }
    #
    # if not os.path.exists(os.path.join(cfg.artifact_path(), intervention, 'performance')):
    #     os.makedirs(os.path.join(cfg.artifact_path(), intervention, 'performance'))
    # with open(artifact_dir + os.sep + intervention + os.sep + 'performance' + os.sep + model_name + '_' +
    #           'model_performance.pkl', 'wb') as f:
    #     pickle.dump(model_performance, f)
    # print("saving done!")
    #
    # # plot and save accuracy
    # fig = plot_lying_honest_accuracy(cfg, accuracy_honest, accuracy_lying)
    # # save
    # # fig.write_html(artifact_dir + os.sep + 'performance' + os.sep + model_name + '_' + data_category + '_' +
    # #                'accuracy'+'.html')
    # pio.write_image(fig, artifact_dir + os.sep + intervention + os.sep + 'performance' + os.sep + model_name + '_' + data_category + '_' +
    #                'accuracy' + '.png', scale=6)
    # print("accuracy done!")

    # plot activation pca

    # 3. plot pca
    fig = plot_contrastive_activation_pca(activations_1, activations_2,
                                          n_layers,
                                          contrastive_name=contrastive_name,
                                          labels_1=labels_1, labels_2=labels_2,
                                          color_by='probability',
                                          probability_1=probability_1, probability_2=probability_2)
    plot_one_layer_3d(activations_1, activations_2,
                      labels_1=labels_1, labels_2=labels_2,
                      contrastive_name=["honest", "lying"],
                      probability_1=probability_1, probability_2=probability_2,
                      color_by='probability',
                      layer=31)
    if save_plot:
          fig.write_html(artifact_dir + os.sep + intervention + os.sep + model_name + '_activation_pca_' +
                         contrastive_name[0] + '_' + contrastive_name[1] + '.html')

          pio.write_image(fig, artifact_dir + os.sep + intervention + os.sep + model_name + '_activation_pca_' +
                          contrastive_name[0] + '_' + contrastive_name[1] + '.png',
                          scale=6)


def run_pipeline(model_path, save_path, batch_size=16):
    """Run the full pipeline."""

    # 1. Load model
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, save_path=save_path)
    print(cfg)

    model_base = construct_model_base(cfg.model_path)

    # 2. Load and sample filtered datasets
    dataset_correct, dataset_wrong = load_and_sample_datasets(cfg)

    # 3. Get Accuracy
    get_contrastive_accuracy_and_plot(cfg, model_base, dataset_correct, dataset_wrong)
    
    # # 4. Quantify different lying stages
    # get_state_quantification(cfg, activations_1, activations_2, labels)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path, batch_size=args.batch_size)
