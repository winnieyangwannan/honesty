import torch
import random
import json
import os
import argparse
import pickle
from datasets import load_dataset
from typing import List, Tuple, Callable
from pipeline.analysis.stage_statistics import get_state_quantification
from pipeline.honesty_config_stage import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.submodules.activation_pca import get_activations, plot_contrastive_activation_pca
from pipeline.submodules.activation_pca import get_contrastive_activations_and_plot_pca
from pipeline.submodules.evaluate_truthful import get_accuracy_and_unexpected, plot_lying_honest_accuracy
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio


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


def contrastive_extraction_pca_and_stage_quantification(cfg, model_base, dataset_train, dataset_test):

    statements_train = [row['claim'] for row in dataset_train]
    statements_test = [row['claim'] for row in dataset_test]
    labels_train = [row['label'] for row in dataset_train]
    labels_test = [row['label'] for row in dataset_test]

    save_dir = os.path.join(cfg.artifact_path(), "no_intervention")

    # 1.1 extract activations of RESIDUAL STREAM
    print("start extraction of residual stream activation")
    activations_honest, activations_lying = get_contrastive_activations_and_plot_pca(cfg=cfg,
                                                                                     model_base=model_base,
                                                                                     dataset=statements_train,
                                                                                     labels=labels_train,
                                                                                     intervention="residual",
                                                                                     )

    print("done extraction of residual stream activation")
    # 1.2 quantify different lying stages
    stage_stats_residual = get_state_quantification(cfg,
                                                    activations_honest,
                                                    activations_lying,
                                                    labels_train,
                                                    intervention="residual",
                                                    )
    # 2.1 extract activations of MLP
    print("start extraction of MLP")
    activations_honest, activations_lying = get_contrastive_activations_and_plot_pca(cfg=cfg,
                                                                                     model_base=model_base,
                                                                                     dataset=statements_train,
                                                                                     labels=labels_train,
                                                                                     intervention="attn")

    print("done extraction of MLP")
    # 2.2 quantify different lying stages
    stage_stats_attn = get_state_quantification(cfg,
                                                activations_honest,
                                                activations_lying,
                                                labels_train,
                                                intervention="attn"
                                                )

    # 3.1 extract activations of ATTENTION
    print("start extraction OF ATTENTION")
    activations_honest, activations_lying = get_contrastive_activations_and_plot_pca(cfg=cfg,
                                                                                     model_base=model_base,
                                                                                     dataset=statements_train,
                                                                                     labels=labels_train,
                                                                                     intervention="mlp")

    print("done extraction of ATTENTION")
    # 3.2 quantify different lying stages
    stage_stats_mlp = get_state_quantification(cfg,
                                               activations_honest,
                                               activations_lying,
                                               labels_train)
    print("Done")


    # Plot residual, mlp and attention together
    # 1. Stage 1
    stage_1_residual = stage_stats_residual['stage_1']
    stage_1_attn = stage_stats_attn['stage_1']
    stage_1_mlp = stage_stats_mlp['stage_1']

    # 2. Stage 2
    stage_2_residual = stage_stats_residual['stage_2']
    stage_2_attn = stage_stats_attn['stage_2']
    stage_2_mlp = stage_stats_mlp['stage_2']

    # 3. Stage 3
    stage_3_residual = stage_stats_residual['stage_3']
    stage_3_attn = stage_stats_attn['stage_3']
    stage_3_mlp = stage_stats_mlp['stage_3']

    n_layers = activations_honest.shape[1]
    save_path = os.path.join(cfg.artifact_path(), 'stage_stats')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plot_stage_1_stats_ram(stage_1_residual, stage_1_attn, stage_1_mlp, n_layers, save_path)
    plot_stage_2_stats_ram(stage_2_residual, stage_2_attn, stage_2_mlp, n_layers, save_path)
    plot_stage_3_stats_ram(stage_3_residual, stage_3_attn, stage_3_mlp, n_layers, save_path)


def plot_stage_3_stats_ram(stage_3_residual, stage_3_attn, stage_3_mlp, n_layers, save_path):
    # plot stage 3
    line_width =3
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Original high dimensional space', 'PCA',
                                        '', '')
                        )

    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_3_residual['cos_honest_lying'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="firebrick", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_3_attn['cos_honest_lying'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="royalblue", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_3_mlp['cos_honest_lying'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="rebeccapurple", width=line_width)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_3_residual['cos_honest_lying_pca'],
                             mode='lines+markers',
                             name="Residual stream",
                             line=dict(color="firebrick", width=line_width)

    ), row=1, col=2)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_3_attn['cos_honest_lying_pca'],
                             mode='lines+markers',
                             name="Attention",
                             line=dict(color="royalblue", width=line_width)

    ), row=1, col=2)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_3_mlp['cos_honest_lying_pca'],
                             mode='lines+markers',
                             name="MLP",
                             line=dict(color="rebeccapurple", width=line_width)

    ), row=1, col=2)
    fig.update_xaxes(tickvals=np.arange(0, n_layers, 5))

    fig.update_layout(height=400, width=800)
    fig['layout']['xaxis']['title'] = 'Layer'
    fig['layout']['xaxis2']['title'] = 'Layer'
    fig['layout']['yaxis']['title'] = 'Cosine similarity'
    fig['layout']['yaxis2']['title'] = ''

    fig.show()
    # fig.write_html(save_path + os.sep + 'distance_pair.html')
    pio.write_image(fig, save_path + os.sep + 'stage_3_cosine_similarity.png',
                    scale=6)


def plot_stage_2_stats_ram(stage_2_residual, stage_2_attn, stage_2_mlp, n_layers, save_path):
    # plot stage 2
    line_width =3
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Original high dimensional space', 'PCA',
                                        '', '')
                        )

    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_residual['centroid_dist_honest'],
                             mode='lines+markers',
                             showlegend=False,
                             marker=dict(
                                symbol="star",
                                size=10,
                             ),
                             line=dict(color="firebrick", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_residual['centroid_dist_lying'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="firebrick", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_attn['centroid_dist_honest'],
                             mode='lines+markers',
                             showlegend=False,
                             marker=dict(
                                symbol="star",
                                size=10,
                             ),
                             line=dict(color="royalblue", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_attn['centroid_dist_lying'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="royalblue", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_mlp['centroid_dist_honest'],
                             mode='lines+markers',
                             showlegend=False,
                             marker=dict(
                                symbol="star",
                                size=10,
                             ),
                             line=dict(color="rebeccapurple", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_mlp['centroid_dist_lying'],
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="rebeccapurple", width=line_width)
    ), row=1, col=1)
    # PCA
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_residual['centroid_dist_honest_pca'],
                             mode='lines+markers',
                             name="Residual stream_honest",
                             marker=dict(
                                symbol="star",
                                size=10,
                             ),
                             line=dict(color="firebrick", width=line_width)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_residual['centroid_dist_lying_pca'],
                             mode='lines+markers',
                             name="Residual stream_lying",
                             line=dict(color="firebrick", width=line_width)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_attn['centroid_dist_honest_pca'],
                             mode='lines+markers',
                             name="Attention_honest",
                             marker=dict(
                                symbol="star",
                                size=10,
                             ),
                             line=dict(color="royalblue", width=line_width)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_attn['centroid_dist_lying_pca'],
                             mode='lines+markers',
                             name="Attention_lying",
                             line=dict(color="royalblue", width=line_width)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_mlp['centroid_dist_honest_pca'],
                             mode='lines+markers',
                             name="MLP_honest",
                             marker=dict (
                                symbol="star",
                                size=10,
                             ),
                             line=dict(color="rebeccapurple", width=line_width)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=stage_2_mlp['centroid_dist_lying_pca'],
                             mode='lines+markers',
                             name="MLP_lying",
                             line=dict(color="rebeccapurple", width=line_width)
    ), row=1, col=2)
    fig.update_xaxes(tickvals=np.arange(0, n_layers, 5))
    fig.update_layout(height=400, width=800)
    fig['layout']['xaxis']['title'] = 'Layer'
    fig['layout']['xaxis2']['title'] = 'Layer'
    fig['layout']['yaxis']['title'] = 'Distance'
    fig['layout']['yaxis2']['title'] = ''

    fig.show()
    # fig.write_html(save_path + os.sep + 'distance_pair.html')
    pio.write_image(fig, save_path + os.sep + 'stage_2_centroid_distance_true_false.png',
                    scale=6)


def plot_stage_1_stats_ram(stage_1_residual,stage_1_attn, stage_1_mlp, n_layers, save_path):
    # plot stage 1
    line_width =3
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Original high dimensional space', 'PCA',
                                        '', '')
                        )

    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=np.mean(stage_1_residual['dist_pair_z'], axis=1),
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="firebrick",width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=np.mean(stage_1_attn['dist_pair_z'], axis=1),
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="royalblue", width=line_width)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=np.mean(stage_1_mlp['dist_pair_z'], axis=1),
                             mode='lines+markers',
                             showlegend=False,
                             line=dict(color="rebeccapurple", width=line_width)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=np.mean(stage_1_residual['dist_pair_z_pca'], axis=1),
                             mode='lines+markers',
                             name="Residual stream",
                             line=dict(color="firebrick", width=line_width)

    ), row=1, col=2)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=np.mean(stage_1_attn['dist_pair_z_pca'], axis=1),
                             mode='lines+markers',
                             name="Attention",
                             line=dict(color="royalblue", width=line_width)

    ), row=1, col=2)
    fig.add_trace(go.Scatter(
                             x=np.arange(n_layers), y=np.mean(stage_1_mlp['dist_pair_z_pca'], axis=1),
                             mode='lines+markers',
                             name="MLP",
                             line=dict(color="rebeccapurple", width=line_width)

    ), row=1, col=2)
    fig.update_xaxes(tickvals=np.arange(0, n_layers, 5))

    fig.update_layout(height=400, width=800)
    fig['layout']['xaxis']['title'] = 'Layer'
    fig['layout']['xaxis2']['title'] = 'Layer'
    fig['layout']['yaxis']['title'] = 'Distance (z_scored)'
    fig['layout']['yaxis2']['title'] = ''

    fig.show()
    # fig.write_html(save_path + os.sep + 'distance_pair.html')
    pio.write_image(fig, save_path + os.sep + 'stage_1_distance_pair.png',
                    scale=6)


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

    # 3. Cache activation and plot pca
    contrastive_extraction_pca_and_stage_quantification(cfg, model_base, dataset_train, dataset_test)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 intervention=args.intervention,
                 source_layer=args.source_layer,
                 target_layer_s=args.target_layer_s, target_layer_e=args.target_layer_e)
