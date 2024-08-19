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
    stage_path = os.path.join(artifact_path, 'stage_stats')
    if not os.path.exists(stage_path):
        os.makedirs(stage_path)

    cos = []
    cos_pca = []
    refusal_score = []
    for ii in contrastive_type:
        # load cosine similarity
        filename = stage_path + os.sep + \
                   model_alias + f'_HHH_{ii}_stage_stats.pkl'
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        cos.append(data['stage_3']['cos_honest_lying'][-1])
        cos_pca.append(data['stage_3']['cos_honest_lying_pca'])

        # load refusal score
        filename = artifact_path + os.sep + 'performance' + os.sep + \
                   f'jailbreakbench_refusal_score_{ii}.pkl'
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        refusal_score.append(data['refusal_score'])

    # plot
    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=[''])
    for ii, c_type in enumerate(contrastive_type):
        fig.add_trace(
            go.Scatter(x=[cos[ii]],
                       y=[refusal_score[ii]],
                       mode="markers",
                       name=c_type,
                       showlegend=True,
                       marker=dict(
                           symbol="circle",
                           size=4,
                           line=dict(width=1, color="DarkSlateGrey"),
                       )),
            row=1, col=1,
        )
    fig.update_layout(height=400, width=500)
    fig.update_layout(
        title="",
        xaxis_title="Cosine Similarity",
        yaxis_title="Refusal Score",
        # legend_title="Legend Title",
        font=dict(
            # family="Courier New, monospace",
            size=10,
            # color="RebeccaPurple"
        )
    )
    fig.show()

    fig.write_html(stage_path + os.sep + 'correlation_cos_refusal.html')
    pio.write_image(fig, stage_path + os.sep + 'correlation_cos_refusal.png',
                    scale=6)


if __name__ == "__main__":
    args = parse_arguments()
    data_type = ["jailbreakbench", "harmless"]

    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 data_type=data_type, contrastive_type=tuple(args.contrastive_type),
                 )
