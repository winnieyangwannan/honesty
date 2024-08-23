import numpy as np
import torch
from pipeline.plot.plot_layer_pca_jailbreaks import plot_contrastive_activation_pca_one_layer_jailbreaks
from pipeline.plot.plot_layer_pca_jailbreaks import plot_contrastive_activation_pca_one_layer_jailbreaks_3d
import os
from pipeline.honesty_config_generation_intervention import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.analysis.stage_statistics import get_state_quantification
from pipeline.jailbreak_config_generation import Config
from pipeline.run_pipeline_honesty_stage import load_and_sample_datasets
from pipeline.plot.plot_some_layer_pca import plot_contrastive_activation_pca_layer
import pickle
from sklearn.decomposition import PCA
import plotly.io as pio
import argparse
import plotly.graph_objects as go


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='google/gemma-2-9b-it')
    parser.add_argument('--save_path', type=str, required=False, default='D:\Data\jailbreak')
    parser.add_argument('--prompt_type', type=str, required=False, default=16)
    parser.add_argument('--contrastive_type', metavar='N', type=str, nargs='+',
                        help='a list of strings')
    parser.add_argument('--save_name', type=str, required=False, default='persona')

    return parser.parse_args()


def run_pipeline(model_path, save_path, save_name,
                 prompt_type=["jailbreakbench", "harmless"],
                 contrastive_type=['evil_confidant', 'AIM'],

                 layer_plot=0):
    """Run the full pipeline."""

    model_alias = os.path.basename(model_path)

    cfg = Config(model_alias=model_alias, model_path=model_path,
                 jailbreak_type=contrastive_type[0], save_path=save_path,
                 )

    print(cfg)
    artifact_path = cfg.artifact_path()
    plot_path = os.path.join(artifact_path, 'performance')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # positive
    filename = plot_path + os.sep + \
               f'{prompt_type[0]}_refusal_score_HHH.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    categories = ['Mean']
    refusal_score_positive = data['refusal_score']
    for key in data['refusal_score_per_category'].keys():
        refusal_score_positive = np.append(refusal_score_positive, data['refusal_score_per_category'][key])
        categories.append(key)

    refusal_score_negative_all = np.zeros((len(contrastive_type), len(categories)))
    for ii, jailbreak in enumerate(contrastive_type):

        filename = plot_path + os.sep + \
                   f'{prompt_type[0]}_refusal_score_{jailbreak}.pkl'
        with open(filename, 'rb') as file:
            data = pickle.load(file)

        refusal_score_negative = data['refusal_score']
        for key in data['refusal_score_per_category'].keys():
            refusal_score_negative = np.append(refusal_score_negative, data['refusal_score_per_category'][key])
        refusal_score_negative_all[ii, :] = refusal_score_negative

    marker_color = ['lightsalmon']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=refusal_score_positive,
        name='HHH',
        showlegend=True,

        # marker_color='indianred'
    ))
    for ii in range(len(contrastive_type)):
        fig.add_trace(go.Bar(
            x=categories,
            y=refusal_score_negative_all[ii],
            name=contrastive_type[ii],
            showlegend=True,
            # marker_color=marker_color[ii]
        ))

    fig.update_layout(height=800, width=1600,
                      )
    fig.update_layout(
        xaxis_title="Categories",
        yaxis_title="Refusal Score",
        # legend_title="Legend Title",
        font=dict(
            # family="Courier New, monospace",
            size=18,
            # color="RebeccaPurple"
        )
    )
    fig.show()

    # contrastive_type = list(contrastive_type)
    fig.write_html(plot_path + os.sep + f'performance_HHH_{save_name}.html')
    pio.write_image(fig, plot_path + os.sep + f'performance_HHH_{save_name}.png',
                    scale=6)


if __name__ == "__main__":
    args = parse_arguments()
    prompt_type = ["jailbreakbench", "harmless"]

    run_pipeline(model_path=args.model_path, save_path=args.save_path,
                 prompt_type=prompt_type, contrastive_type=tuple(args.contrastive_type),
                 save_name=args.save_name
                 )
