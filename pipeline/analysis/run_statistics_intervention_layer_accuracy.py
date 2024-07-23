import os
import argparse
from pipeline.honesty_config_generation_skip_connection import Config
from pipeline.model_utils.model_factory import construct_model_base
import pickle
import plotly.io as pio
import csv
import math
from tqdm import tqdm
from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import einops
from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
import plotly.express as px


def parse_arguments():
    """Parse model path argument from command line."""

    parser = argparse.ArgumentParser(description="Parse model path argument.")

    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--save_path', type=str, required=False, default=" ")
    parser.add_argument('--intervention', type=str, required=False, default="honest_addition")
    parser.add_argument('--source_layer', type=int, required=False, default=0)
    parser.add_argument('--target_layer_s', type=int, required=False, default=14)
    parser.add_argument('--target_layer_e', type=int, required=False, default=15)

    return parser.parse_args()


def get_accuracy_statistics(cfg, model_base):
    artifact_path = cfg.artifact_path()
    intervention = cfg.intervention
    data_category = cfg.data_category
    n_layers = model_base.model.config.num_hidden_layers
    source_layers = np.arange(0, n_layers)
    # source_layers = np.arange(0, 64)

    # load data
    accuracy_lying = []
    accuracy_honest = []
    for layer in source_layers:
        if "skip_connection" in intervention:
            filename = artifact_path + os.sep + intervention + os.sep + f'{data_category}_{intervention}_' +\
                       'model_performance_layer_0_' + str(layer) + '_' + str(layer+1) + '.pkl'
        elif "addition" in intervention:
            filename = artifact_path + os.sep + intervention + os.sep + f'{data_category}_{intervention}_' + \
                       'model_performance_layer_' + str(layer) + '_' + str(layer) + '_None' + '.pkl'
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        accuracy_honest.append(data["accuracy_honest"])
        accuracy_lying.append(data["accuracy_lying"])

    accuracy_lying = [float(accuracy_lying[ii]) for ii in range(len(accuracy_lying))]
    accuracy_honest = [float(accuracy_honest[ii]) for ii in range(len(accuracy_honest))]

    # plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=source_layers, y=accuracy_lying,
                             name="Lying",
                             mode='lines+markers',
                             marker=dict(
                                color='dodgerblue')
                             ))
    fig.add_trace(go.Scatter(x=source_layers, y=accuracy_honest,
                             name="Honest",
                             mode='lines+markers',
                             marker=dict(
                                 color='gold')
                             ))
    fig.update_layout(
        xaxis_title="Layer",
        yaxis_title="Accuracy",
        width=600,
        height=300
    )
    fig.update_xaxes(tickvals=np.arange(0, n_layers, 5))
    fig.show()
    fig.write_html(artifact_path + os.sep + intervention + os.sep + data_category + '_' + intervention +
                   '_statistics_accuracy'+'.html')
    pio.write_image(fig, artifact_path + os.sep + intervention + os.sep + data_category + '_' + intervention +
                   '_statistics_accuracy'+'.png',
                    scale=6)


def run_pipeline(model_path, save_path, intervention, source_layer, target_layer_s, target_layer_e):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, save_path=save_path,
                 intervention=intervention,
                 source_layer=source_layer,
                 target_layer_s=target_layer_s, target_layer_e=target_layer_e
                 )
    print(cfg)

    model_base = construct_model_base(cfg.model_path)

    # 1. Accuracy Statistics
    get_accuracy_statistics(cfg, model_base)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 intervention=args.intervention,
                 source_layer=args.source_layer,
                 target_layer_s=args.target_layer_s, target_layer_e=args.target_layer_e
                 )