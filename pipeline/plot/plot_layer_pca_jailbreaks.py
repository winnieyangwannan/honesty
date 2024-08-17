import os
import argparse
from pipeline.jailbreak_config_generation import Config
from pipeline.model_utils.model_factory import construct_model_base
import pickle
import csv
import math
from tqdm import tqdm
from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_similarity
from plotly.figure_factory import create_quiver
import plotly.figure_factory as ff
import plotly.io as pio
from sklearn.decomposition import PCA
import numpy as np
import torch
from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
from scipy.spatial.distance import cdist
from scipy import stats



contrastive_type = ["HHH", "evil_confidant", "AIM"]
prompt_type = ["jailbreakbench", "harmless"]
model_path = "google/gemma-2b-it"
save_path = "D:\Data\jailbreak"


model_alias = os.path.basename(model_path)

cfg = Config(model_alias=model_alias, model_path=model_path,
             save_path=save_path, checkpoint=None, few_shot=0)
print(cfg)
artifact_path = cfg.artifact_path()
filename = artifact_path + os.sep + \
           model_alias + '_activation_pca.pkl'
with open(filename, 'rb') as file:
    data = pickle.load(file)


def plot_contrastive_activation_pca_layer_jailbreaks(activations,
                                                     contrastive_labels,
                                                     contrastive_type,
                                                     prompt_labels=None,
                                                     prompt_type=['true', 'false'],
                                                     layers=[0, 1]):
    print("plot")

