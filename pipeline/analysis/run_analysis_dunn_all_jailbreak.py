import numpy as np
import os
import argparse
from pipeline.configs.honesty_config_generation_intervention import Config
from pipeline.jailbreak_config_generation import Config
import pickle

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


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

    filename = stage_path + os.sep + \
               model_alias + f'_HHH_evil_confidant_stage_stats.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    dunn_min_positive = np.min(data['stage_2_dunn']['dunn_index_all'][0])
    dunn_max_positive = np.max(data['stage_2_dunn']['dunn_index_all'][0])
    dunn_last_positive = data['stage_2_dunn']['dunn_index_all'][0][-1]

    n_layers = data['stage_2_dunn']['dunn_index_all'].shape[1]

    dunn_pca = np.zeros((len(contrastive_type), 1))
    dunn = np.zeros((len(contrastive_type), 1))
    dunn_min = np.zeros((len(contrastive_type), 1))
    dunn_max = np.zeros((len(contrastive_type), 1))
    dunn_last = np.zeros((len(contrastive_type), 1))
    dunn_min_pca = np.zeros((len(contrastive_type), 1))
    dunn_max_pca = np.zeros((len(contrastive_type), 1))
    dunn_last_pca = np.zeros((len(contrastive_type), 1))
    # refusal_score = np.zeros((len(contrastive_type), n_layers))
    for ii, jailbreak in enumerate(contrastive_type):
        # load dunnine similarity
        filename = stage_path + os.sep + \
                   model_alias + f'_HHH_{jailbreak}_stage_stats.pkl'
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        dunn[ii] = np.mean(data['stage_2_dunn']['dunn_index_all'][1] - data['stage_2_dunn']['dunn_index_all'][0])
        dunn_pca[ii] = np.mean(data['stage_2_dunn']['dunn_index_pca_all'][1] - data['stage_2_dunn']['dunn_index_pca_all'][0])

        dunn_min[ii] = np.min(data['stage_2_dunn']['dunn_index_all'][1] - data['stage_2_dunn']['dunn_index_all'][0])
        dunn_max[ii] = np.max(data['stage_2_dunn']['dunn_index_all'][1] - data['stage_2_dunn']['dunn_index_all'][0])
        dunn_last[ii] = data['stage_2_dunn']['dunn_index_all'][1][-1] - data['stage_2_dunn']['dunn_index_all'][0][-1]

        dunn_min_pca[ii] = np.min(data['stage_2_dunn']['dunn_index_pca_all'][1] - data['stage_2_dunn']['dunn_index_pca_all'][0])
        dunn_max_pca[ii] = np.max(data['stage_2_dunn']['dunn_index_pca_all'][1] - data['stage_2_dunn']['dunn_index_pca_all'][0])
        dunn_last_pca[ii] = data['stage_2_dunn']['dunn_index_pca_all'][1][-1] - data['stage_2_dunn']['dunn_index_pca_all'][0][-1]

    # plot
    # colors = ['lightskyblue', 'dogerblue', 'steelblue', ]
    # colors = ['lightskyblue', 'deepskyblue', 'orchid', 'mediumorchid', 'darkorchid', 'coral']
    line_width = 3
    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=[''])
    for ii, jailbreak in enumerate(contrastive_type):
        fig.add_trace(go.Bar(
                                 x=[jailbreak],
                                 y=[dunn_last[ii, 0]],
                                 name=jailbreak,
                                 showlegend=False,
                                 marker_color='royalblue',
        ), row=1, col=1)

    fig.update_layout(height=800, width=1000)
    fig.update_layout(
        xaxis_title="Jailbreak Types",
        yaxis_title="Dunn Index (Jailbreak - HHH)",
        # legend_title="Legend Title",
        font=dict(
            # family="Courier New, monospace",
            size=18,
            # color="RebeccaPurple"
        )
    )
    fig.show()
    fig.write_html(stage_path + os.sep + 'dunn_all.html')
    pio.write_image(fig, stage_path + os.sep + 'dunn_all.png',
                    scale=6)


if __name__ == "__main__":
    args = parse_arguments()
    data_type = ["jailbreakbench", "harmless"]

    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 data_type=data_type, contrastive_type=tuple(args.contrastive_type),
                 )
