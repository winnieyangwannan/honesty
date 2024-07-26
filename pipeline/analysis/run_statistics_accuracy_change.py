import os
import argparse
from pipeline.honesty_config_generation_skip_connection import Config
from pipeline.model_utils.model_factory import construct_model_base
import pickle
import csv
import math
from tqdm import tqdm
from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
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
    parser.add_argument('--save_path', type=str, required=False, default=" ")
    parser.add_argument('--intervention', type=str, required=False, default="honest_addition")
    parser.add_argument('--source_layer', type=int, required=False, default=16)
    parser.add_argument('--target_layer_s', type=int, required=False, default=16)
    parser.add_argument('--target_layer_e', type=int, required=False, default=16)

    return parser.parse_args()


def get_accuracy_statistics_intervention_change(cfg, model_base):
    artifact_path = cfg.artifact_path()
    intervention = cfg.intervention
    data_category = cfg.data_category
    source_layer = cfg.source_layer
    target_layer_s = cfg.target_layer_s
    target_layer_e = cfg.target_layer_e
    n_layers = model_base.model.config.num_hidden_layers
    model_name = cfg.model_alias

    # load data
    filename = artifact_path + os.sep + "performance" + os.sep + model_name + '_' + 'model_performance.pkl'
    with open(filename, 'rb') as file:
       data = pickle.load(file)

    accuracy_honest = data["accuracy_honest"]
    accuracy_lying = data["accuracy_lying"]

    # load intervention data
    filename = artifact_path + os.sep + intervention + os.sep + f'{data_category}_{intervention}_' + 'model_performance_layer_'\
               + str(source_layer)+'_' + str(target_layer_s) + '_' + str(target_layer_e) + '.pkl'
    with open(filename, 'rb') as file:
       data = pickle.load(file)
    accuracy_intervention_honest = data["accuracy_honest"]
    accuracy_intervention_lying = data["accuracy_lying"]

    # change in accuracy
    delta_honest = accuracy_intervention_honest - accuracy_honest
    delta_lying = accuracy_intervention_lying - accuracy_lying

    # plot
    d = {'Change in Accuracy': [delta_honest, delta_lying],
         'Role': ["Honest", "Lying"]}
    df = pd.DataFrame(data=d)
    # plot
    fig = px.bar(df, x="Role", y="Change in Accuracy", title=f"{model_name}",
                 width=400, height=400)
    fig.show()

    # save
    # fig.write_html(artifact_path + os.sep + intervention + os.sep + data_category + '_' +
    #                '_statistics_change_in_accuracy'+'.html')
    fig.write_image(artifact_path + os.sep + intervention + os.sep + data_category +
                   '_statistics_change_in_accuracy_layer_'+  str(source_layer)+'_' + str(target_layer_s) +
                    '_' + str(target_layer_e) + '.svg')


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
    get_accuracy_statistics_intervention_change(cfg, model_base)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 intervention=args.intervention,
                 source_layer=args.source_layer,
                 target_layer_s=args.target_layer_s, target_layer_e=args.target_layer_e
                 )