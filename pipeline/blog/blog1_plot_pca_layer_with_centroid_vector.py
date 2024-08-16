
import os
import argparse
from pipeline.honesty_config_generation_intervention import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.analysis.stage_statistics import get_state_quantification
from pipeline.honesty_config_performance import Config
from pipeline.run_pipeline_honesty_stage import load_and_sample_datasets
from pipeline.run_pipeline_honesty_stage import get_lying_honest_accuracy_and_plot
from pipeline.plot.checkout_one_layer import plot_one_layer_with_centroid_and_vector


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--save_path', type=str, required=False, default=16)
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--layer_plot', type=int, required=False, default=16)

    return parser.parse_args()


def run_pipeline(model_path, save_path, layer_plot=16):
    """Run the full pipeline."""


    # 1. Load model
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, save_path=save_path)
    print(cfg)
    model_base = construct_model_base(cfg.model_path)

    # 2. Load and sample filtered datasets
    dataset = load_and_sample_datasets(cfg)

    # 3. Get Accuracy
    activations_honest, activations_lying, labels = get_lying_honest_accuracy_and_plot(cfg, model_base, dataset)

    # 4. Get centroid
    activations_pca, centroid_honest_true, centroid_honest_false, centroid_vector_honest, centroid_lying_true, centroid_lying_false, centroid_vector_lying = get_state_quantification(cfg, activations_honest, activations_lying, labels)

    # 5. plot
    intervention = cfg.intervention
    save_dir = os.path.join(cfg.artifact_path(), intervention)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plot_one_layer_with_centroid_and_vector(activations_pca,
                                            centroid_honest_true, centroid_honest_false,
                                            centroid_lying_true, centroid_lying_false,
                                            centroid_vector_honest, centroid_vector_lying,
                                            labels,
                                            save_dir,
                                            prompt_label=["honest", "lying"],
                                            layer=layer_plot)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path, layer_plot=args.layer_plot)


