import random
import os
import argparse
import pickle
import plotly.io as pio

from datasets import load_dataset
from pipeline.submodules.evaluate_truthful import get_statement_accuracy_cache_activation
from pipeline.honesty_config_performance import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.save.activation_pca import plot_contrastive_activation_pca
from pipeline.submodules.select_direction import get_refusal_scores


# from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--save_path', type=str, required=False, default=16)
    parser.add_argument('--batch_size', type=int, required=False, default=16)

    return parser.parse_args()


def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        train and test dataset
    """

    random.seed(42)

    dataset_all = load_dataset("notrichardren/azaria-mitchell-diff-filtered-2")
    dataset = [row for row in dataset_all[f"{cfg.data_category}"]]
    data = random.sample(dataset, cfg.n_train)

    return data


def filter_data(cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)
    
    return harmful_train, harmless_train, harmful_val, harmless_val


def get_lying_honest_accuracy_and_plot(cfg, model_base, dataset):

    artifact_dir = cfg.artifact_path()
    intervention = cfg.intervention
    model_name = cfg.model_alias
    data_category = cfg.data_category

    # lying prompt template
    performance_lying, probability_lying, unexpected_lying, activations_lying = get_statement_accuracy_cache_activation(
                                                                                 model_base,
                                                                                 dataset,
                                                                                 cfg, system_type="lying",)

    # honest prompt template
    performance_honest, probability_honest, unexpected_honest, activations_honest = get_statement_accuracy_cache_activation(
                                                                                model_base,
                                                                                dataset,
                                                                                cfg, system_type="honest",)

    accuracy_lying = sum(performance_lying)/len(performance_lying)
    accuracy_honest = sum(performance_honest) / len(performance_honest)
    unexpected_lying_rate = sum(unexpected_lying)/len(unexpected_lying)
    unexpected_honest_rate = sum(unexpected_honest)/len(unexpected_honest)
    print(f"accuracy_lying: {accuracy_lying}")
    print(f"accuracy_honest: {accuracy_honest}")
    print(f"unexpected_lying: {unexpected_lying_rate}")
    print(f"unexpected_honest: {unexpected_honest_rate}")

    model_performance = {
        "performance_lying": performance_lying,
        "performance_honest": performance_honest,
        "accuracy_lying": accuracy_lying,
        "accuracy_honest": accuracy_honest,
        "unexpected_lying": unexpected_lying,
        "unexpected_honest": unexpected_honest,
        "unexpected_lying_rate": unexpected_lying_rate,
        "unexpected_honest": unexpected_honest
    }

    if not os.path.exists(os.path.join(cfg.artifact_path(), intervention, 'performance')):
        os.makedirs(os.path.join(cfg.artifact_path(), intervention, 'performance'))
    with open(artifact_dir + os.sep + intervention + os.sep + 'performance' + os.sep + model_name + '_' +
              'model_performance.pkl', 'wb') as f:
        pickle.dump(model_performance, f)
    print("saving done!")

    # plot and save accuracy
    # fig = plot_lying_honest_accuracy(cfg, accuracy_honest, accuracy_lying)
    # save
    # fig.write_html(artifact_dir + os.sep + 'performance' + os.sep + model_name + '_' + data_category + '_' +
    #                'accuracy'+'.html')
    # pio.write_image(fig, artifact_dir + os.sep + intervention + os.sep + 'performance' + os.sep + model_name + '_' + data_category + '_' +
    #                'accuracy' + '.png', scale=6)
    # print("accuracy done!")

    # plot activation pca
    contrastive_label = ["honest", "lying"]
    n_layers = model_base.model.config.num_hidden_layers
    labels = [row['label'] for row in dataset]

    fig = plot_contrastive_activation_pca(activations_honest, activations_lying, n_layers,
                                          contrastive_label, labels=labels)
    fig.write_html(artifact_dir + os.sep + intervention + os.sep + model_name + '_' + 'honest_lying_pca.html')
    pio.write_image(fig, artifact_dir + os.sep + intervention + os.sep + model_name + '_'
                    + 'honest_lying_pca.png',
                    scale=6)
    pio.write_image(fig, artifact_dir + os.sep + intervention + os.sep + model_name + '_'
                    + 'honest_lying_pca.pdf',
                    scale=6)
    return activations_honest, activations_lying, labels


def run_pipeline(model_path, save_path, batch_size=16):
    """Run the full pipeline."""

    # 1. Load model
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path, save_path=save_path)
    print(cfg)

    model_base = construct_model_base(cfg.model_path)

    # 2. Load and sample filtered datasets
    dataset = load_and_sample_datasets(cfg)

    # 3. Get Accuracy
    get_lying_honest_accuracy_and_plot(cfg, model_base, dataset)

    # # 4. Quantify different lying stages
    # get_state_quantification(cfg, activations_honest, activations_lying, labels)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, save_path=args.save_path, batch_size=args.batch_size)
