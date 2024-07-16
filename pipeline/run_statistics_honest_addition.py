import os
import argparse
from pipeline.honesty_config_generation_honest_addition import Config
from pipeline.model_utils.model_factory import construct_model_base
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
    parser.add_argument('--source_layer', type=int, required=False, default=16)
    parser.add_argument('--target_layer', type=int, required=False, default=16)

    return parser.parse_args()


def get_accuracy_statistics(cfg,model_base):
    artifact_path = cfg.artifact_path()
    intervention = cfg.intervention
    data_category = cfg.data_category
    n_layers = model_base.model.config.num_hidden_layers
    source_layers = np.arange(0, n_layers)

    # load data
    accuracy_lying = []
    for layer in source_layers:
        filename = artifact_path + os.sep + intervention + os.sep + f'{data_category}_{intervention}_' + 'model_performance_layer_' + str(
                            layer) + '_' + str(layer) + '.csv'
        with open(filename, 'r') as file:
           reader = csv.reader(file)
           data_list = list(reader)
        accuracy_lying.append(data_list[2][-1])
    accuracy_lying = [float(accuracy_lying[ii]) for ii in range(len(accuracy_lying))]

    # plot
    fig = px.line(x=source_layers, y=accuracy_lying,
                  labels=dict(x="Layer", y="Accuracy"),
                  width=800, height=400)
    fig.show()
    fig.write_html(artifact_path + os.sep + intervention + os.sep + data_category + '_' +intervention +
                   '_statistics_accuracy'+'.html')


def run_pipeline(model_path, save_path, intervention, source_layer, target_layer):
    """Run the full pipeline."""


    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, save_path=save_path,
                 intervention=intervention,
                 source_layer=source_layer, target_layer=target_layer)
    print(cfg)

    model_base = construct_model_base(cfg.model_path)

    # 1. Accuracy Statistics
    get_accuracy_statistics(cfg, model_base)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 intervention=args.intervention,
                 source_layer=args.source_layer, target_layer=args.target_layer)