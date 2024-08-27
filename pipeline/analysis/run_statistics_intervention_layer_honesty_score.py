import os
import argparse
from pipeline.configs.config_contrastive_steering import Config
from pipeline.model_utils.model_factory import construct_model_base
import plotly.io as pio
import json
import plotly.graph_objects as go
import numpy as np


def parse_arguments():
    """Parse model path argument from command line."""

    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help="google/gemma-2-2b-it")
    parser.add_argument('--source_layer', type=int, required=False, default=0)
    parser.add_argument('--target_layer', type=int, required=False, default=0)
    parser.add_argument('--save_path', type=str, required=False, default='')
    parser.add_argument('--hook_name', type=str, required=False, default='resid_post')
    parser.add_argument('--intervention', type=str, required=False, default='positive_addition')
    parser.add_argument('--task_name', type=str, required=False, default='honesty')
    parser.add_argument('--jailbreak', type=str, required=False, default='evil_confidant')

    return parser.parse_args()


def get_accuracy_statistics(cfg, model_base):
    artifact_path = cfg.artifact_path()
    intervention = cfg.intervention
    n_layers = model_base.model.config.num_hidden_layers
    source_layers = np.arange(0, n_layers)
    # source_layers = np.arange(0, 64)

    # load data
    accuracy_lying = []
    accuracy_honest = []

    for layer in source_layers:

        if "skip_connection" in intervention:

            save_name = f'{intervention}_layer_s_{layer}_layer_t_{layer}'
            filename = artifact_path + os.sep + intervention + os.sep + 'completions' + os.sep + \
                       f'completions_{save_name}_{cfg.contrastive_label[0]}.json'
            with open(filename, 'r') as file:
                data = json.load(file)
            accuracy_honest.append(data["honest_score"])
            filename = artifact_path + os.sep + intervention + os.sep + 'completions' + os.sep + \
                       f'completions_{save_name}_{cfg.contrastive_label[1]}.json'
            with open(filename, 'r') as file:
                data = json.load(file)
            accuracy_lying.append(data["honest_score"])

        elif "addition" in intervention or "ablation" in intervention:
            save_name = f'{intervention}_layer_s_{layer}_layer_t_{layer}'
            filename = artifact_path + os.sep + intervention + os.sep + 'completions' + os.sep +\
                       f'completions_{save_name}_{cfg.contrastive_label[0]}.json'
            with open(filename, 'r') as file:
                data = json.load(file)
            accuracy_honest.append(data["honest_score"])
            filename = artifact_path + os.sep + intervention + os.sep + 'completions' + os.sep + \
                       f'completions_{save_name}_{cfg.contrastive_label[1]}.json'
            with open(filename, 'r') as file:
                data = json.load(file)
            accuracy_lying.append(data["honest_score"])

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
    fig.write_html(artifact_path + os.sep + intervention + os.sep +
                   'honest_score_summary'+'.html')
    pio.write_image(fig, artifact_path + os.sep + intervention + os.sep +
                   'honest_score_summary'+'.png',
                    scale=6)


def run_pipeline(model_path='google/gemma-2-2b-it',
                 source_layer=0,
                 target_layer=0,
                 hook_name='resid_pre',
                 task_name='honesty',
                 contrastive_label=['honesty', 'lying'],
                 save_path='D:\Data\honesty',
                 intervention='positive_addition'
                 ):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    save_name = f'{intervention}_layer_s_{source_layer}_layer_t_{target_layer}'

    cfg = Config(model_alias=model_alias,
                 model_path=model_path,
                 source_layer=source_layer,
                 target_layer=target_layer,
                 intervention=intervention,
                 hook_name=hook_name,
                 save_path=save_path,
                 task_name=task_name,
                 save_name=save_name,
                 contrastive_label=contrastive_label
                 )
    print(cfg)

    model_base = construct_model_base(cfg.model_path)

    # 1. Accuracy Statistics
    get_accuracy_statistics(cfg, model_base)


if __name__ == "__main__":
    args = parse_arguments()
    if args.task_name == 'honesty':
        contrastive_label = ['honest', 'lying']
    elif args.task_name == 'jailbreak':
        contrastive_label = ['HHH', args.jailbreak]
    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 source_layer=args.source_layer, target_layer=args.target_layer,
                 hook_name=args.hook_name, intervention=args.intervention,
                 task_name=args.task_name, contrastive_label=contrastive_label)