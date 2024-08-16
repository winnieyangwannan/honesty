
import os
import argparse
from pipeline.honesty_config_generation_intervention import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.analysis.stage_statistics import get_state_quantification
from pipeline.honesty_config_generation import Config
from pipeline.run_pipeline_honesty_stage import load_and_sample_datasets
from pipeline.plot.checkout_one_layer import plot_contrastive_activation_pca_layer
import pickle
from sklearn.decomposition import PCA
import plotly.io as pio


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--save_path', type=str, required=False, default='D:\Data\honesty')
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--layer_plot', nargs='+', type=int)

    return parser.parse_args()


def run_pipeline(model_path, save_path,
                 layer_plot=16, contrastive_label=['honest', 'lying']):
    """Run the full pipeline."""


    # 1. Load model
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path,
                 save_path=save_path, checkpoint=None, few_shot=0)
    print(cfg)
    artifact_path = cfg.artifact_path()

    # 2. Load pca
    filename = artifact_path + os.sep + \
               model_alias + '_activation_pca.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    activations_positive = data['activations_positive']
    activations_negative = data['activations_negative']
    labels = data['labels']


    # 5. plot
    intervention = cfg.intervention
    save_dir = os.path.join(cfg.artifact_path(), 'plot')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig = plot_contrastive_activation_pca_layer(activations_positive, activations_negative,
                                                contrastive_label,
                                                labels=labels, prompt_label=['true', 'false'],
                                                layers=layer_plot)

    pio.write_image(fig, save_dir + os.sep
                    + str(layer_plot) + '_pca.pdf',
                    )

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path, layer_plot=tuple(args.layer_plot))


