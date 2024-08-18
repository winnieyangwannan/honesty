import numpy as np
import torch
from pipeline.plot.plot_layer_pca_jailbreaks import plot_contrastive_activation_pca_one_layer_jailbreaks
from pipeline.plot.plot_layer_pca_jailbreaks import plot_contrastive_activation_pca_one_layer_jailbreaks_3d
import os
from pipeline.honesty_config_generation_intervention import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.analysis.stage_statistics import get_state_quantification
from pipeline.jailbreak_config_generation import Config
from pipeline.run_pipeline_honesty_stage import load_and_sample_datasets
from pipeline.plot.plot_some_layer_pca import plot_contrastive_activation_pca_layer
import pickle
from sklearn.decomposition import PCA
import plotly.io as pio
import argparse


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='google/gemma-2-9b-it')
    parser.add_argument('--save_path', type=str, required=False, default='D:\Data\jailbreak')
    parser.add_argument('--prompt_type', type=str, required=False, default=16)
    parser.add_argument('--layer_plot', type=int, required=False, default=41)
    parser.add_argument('--contrastive_type', metavar='N', type=str, nargs='+',
                        help='a list of strings')

    return parser.parse_args()


def run_pipeline(model_path, save_path,
                 prompt_type=["jailbreakbench", "harmless"],
                 contrastive_type=['evil_confidant', 'AIM'],
                 layer_plot=0):
    """Run the full pipeline."""

    model_alias = os.path.basename(model_path)

    cfg = Config(model_alias=model_alias, model_path=model_path,
                 jailbreak_type=contrastive_type[0], save_path=save_path,
                 )

    print(cfg)
    artifact_path = cfg.artifact_path()
    plot_path = os.path.join(artifact_path, 'plot')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # positive
    filename = artifact_path + os.sep + \
               model_alias + f'_activation_pca_HHH_{contrastive_type[0]}.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    activations_positive = data['activations_positive']
    contrastive_labels = ['HHH'] * len(data['activations_positive'])
    prompt_labels = data['labels']

    # different negative
    activations_all = activations_positive
    contrastive_labels_all = np.array(contrastive_labels)
    prompt_labels_all = np.array(prompt_labels)

    for jailbreak in contrastive_type:
        filename = artifact_path + os.sep + \
                   model_alias + f'_activation_pca_HHH_{jailbreak}.pkl'
        with open(filename, 'rb') as file:
            data = pickle.load(file)

        activations_negative = data['activations_negative']
        activations_all = torch.cat((activations_all, activations_negative), dim=0)
        contrastive_labels_all = np.append(contrastive_labels_all, np.array([jailbreak]*len(data['activations_positive'])))
        prompt_labels_all = np.append(prompt_labels_all, np.array(data['labels']))

    contrastive_labels_all = contrastive_labels_all.tolist()
    prompt_labels_all = prompt_labels_all.tolist()

    # plot
    # fig = plot_contrastive_activation_pca_layer_jailbreaks(cfg,
    #                                                        activations_all=activations_all,
    #                                                        contrastive_labels_all=contrastive_labels_all,
    #                                                        contrastive_type=contrastive_type,
    #                                                        prompt_labels_all=prompt_labels_all,
    #                                                        prompt_type=prompt_type,
    #                                                        )
    # fig.write_html(artifact_path + os.sep + f'all_activation_pca.html')
    # pio.write_image(fig, artifact_path + os.sep + f'all_activation_pca.png',
    #                 scale=6)

    fig = plot_contrastive_activation_pca_one_layer_jailbreaks_3d(cfg,
                                                                  activations_all,
                                                                  contrastive_labels_all,
                                                                  contrastive_type,
                                                                  prompt_labels_all,
                                                                  prompt_type,
                                                                  layer_plot=layer_plot
                                                                  )
    contrastive_type = list(contrastive_type)
    fig.write_html(plot_path + os.sep + f'all_activation_pca_layer_{contrastive_type}_{layer_plot}.html')
    pio.write_image(fig, plot_path + os.sep + f'all_activation_pca_layer_{contrastive_type}_{layer_plot}.png',
                    scale=6)


if __name__ == "__main__":
    args = parse_arguments()
    prompt_type = ["jailbreakbench", "harmless"]

    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 prompt_type=prompt_type, contrastive_type=tuple(args.contrastive_type),
                 layer_plot=args.layer_plot)
