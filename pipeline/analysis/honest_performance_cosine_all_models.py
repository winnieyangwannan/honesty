import os
import argparse
from pipeline.jailbreak_config_generation_intervention import Config
from pipeline.model_utils.model_factory import construct_model_base
import pickle
import plotly.io as pio

import plotly.express as px
import pandas as pd

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


model_path = ['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-70b-chat-hf']
save_path = 'D:\Data\honesty'
data_category = 'facts'
Role = ["Honest", "Lying"]
metric = ['lying_score', 'cosine_sim'] * len(model_path)


names = []
scores = []
for mm in model_path:
    model_alias = os.path.basename(mm)
    artifact_path = os.path.join(save_path, "runs", "activation_pca", model_alias)

    # performance
    filename = artifact_path + os.sep + 'performance' + os.sep + \
               model_alias + '_model_performance.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    lying_score = data["wrong_rate_lying"]

    # cosine similarity
    filename = artifact_path + os.sep + 'stage_stats' + os.sep +\
               model_alias + f'_{data_category}_' + \
               'stage_stats.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    # cosine similarity of last layer
    cosine_sim = data['stage_3']["cos_honest_lying"][-1]

    names.append([model_alias, model_alias])
    scores.append([lying_score, cosine_sim])



names = np.array(names)
names = names.flatten()
names = names.tolist()

scores = np.array(scores)
scores = scores.flatten()
scores = scores.tolist()

# initialize data of lists.
data = {'Name': names,
        'score': scores,
        'metric': metric}
# Create DataFrame
df = pd.DataFrame(data)
# plot
fig = px.histogram(df, x="Name", y="score",
             color='metric', barmode='group',
             height=400)
fig.show()
pio.write_image(fig,
                'summary' + '.png',
                scale=6)




