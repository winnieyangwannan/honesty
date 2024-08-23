import numpy as np
import torch
from pipeline.plot.plot_layer_pca_jailbreaks import plot_contrastive_activation_pca_one_layer_jailbreaks
from pipeline.plot.plot_layer_pca_jailbreaks import plot_contrastive_activation_pca_one_layer_jailbreaks_3d
import os
import argparse
from pipeline.honesty_config_generation_intervention import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.jailbreak_config_generation import Config
from pipeline.run_pipeline_honesty_stage import load_and_sample_datasets
from pipeline.plot.plot_some_layer_pca import plot_contrastive_activation_pca_layer
import pickle
import pandas as pd
from sklearn.decomposition import PCA
import plotly.io as pio
from sklearn.metrics.pairwise import pairwise_distances
import torch
from torchmetrics.clustering import DunnIndex
from jaxtyping import Float
from torch import Tensor
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn import preprocessing as pre
from sklearn.metrics.pairwise import cosine_similarity

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.figure_factory import create_quiver
import plotly.figure_factory as ff
import plotly.io as pio
from validclust import dunn
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='google/gemma-2-9b-it')
    parser.add_argument('--save_path', type=str, required=False, default='D:\Data\jailbreak')
    parser.add_argument('--data_type', type=str, required=False, default=16)
    parser.add_argument('--contrastive_type', metavar='N', type=str, nargs='+',
                        help='a list of strings')

    return parser.parse_args()


def run_pipeline(model_path, save_path,
                 data_type=["jailbreakbench", "harmless"],
                 contrastive_type=['evil_confidant', 'AIM'],
                 layer_plot=0):
    """Run the full pipeline."""

    model_alias = os.path.basename(model_path)

    cfg = Config(model_alias=model_alias, model_path=model_path,
                 jailbreak_type=contrastive_type[0], save_path=save_path,
                 )

    print(cfg)
    artifact_path = cfg.artifact_path()
    save_path = os.path.join(artifact_path, 'direction')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = artifact_path + os.sep + \
               f'steering_direction_harmless_direction_HHH_evil_confidant.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    harmful_steering_direction = data['harmful_steering_direction']
    n_layers = harmful_steering_direction.shape[0]
    d_model = harmful_steering_direction.shape[1]

    harmful_steering_direction = np.zeros((len(contrastive_type), n_layers, d_model))
    for ii, jailbreak in enumerate(contrastive_type):
        # load cosine similarity
        filename = artifact_path + os.sep + \
                   f'steering_direction_harmless_direction_HHH_{jailbreak}.pkl'
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        harmful_steering_direction[ii, :, :] = data['harmful_steering_direction'].cpu().numpy()


    cos = np.zeros((n_layers, len(contrastive_type), len(contrastive_type)))
    for layer in range(n_layers):
        cos[layer, :, :] = cosine_similarity(harmful_steering_direction[:, layer, :],
                                             harmful_steering_direction[:, layer, :])

        fig = make_subplots(rows=1, cols=1,
                            subplot_titles=[''])
        fig.add_trace(
                      go.Heatmap(x=contrastive_type,
                                 y=contrastive_type,
                                 z=cos[layer]))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=500, width=500)
        fig.show()
        fig.write_html(save_path + os.sep + f'harmful_steering_direction_layer_{layer}.html')
        pio.write_image(fig, save_path + os.sep + f'harmful_steering_direction_layer_{layer}.png',
                        scale=6)


if __name__ == "__main__":
    args = parse_arguments()
    data_type = ["jailbreakbench", "harmless"]

    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 data_type=data_type, contrastive_type=tuple(args.contrastive_type),
                 )
