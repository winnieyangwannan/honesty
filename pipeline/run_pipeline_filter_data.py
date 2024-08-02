#%%
from datasets import load_dataset
import torch
import pandas as pd
from datasets import DatasetDict, Dataset
from torch.nn.functional import softmax
import argparse
from pipeline.model_utils.model_factory import construct_model_base


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--checkpoint', type=int, required=False, default=None, help='Checkpoint for pyhia model family')
    parser.add_argument('--filter_mode', type=str, required=False, default="correct")
    parser.add_argument('--category', type=str, required=False, default="facts")
    parser.add_argument('--threshold', type=int, required=False, default=0.85)

    return parser.parse_args()


def get_token_probabilities(model, tokenizer, prompt):
    # Tokenize the prompt and get the output from the model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs).logits[:, -1, :]  # We only need the logits for the last token
    probs = softmax(outputs, dim=-1)  # Convert logits to probabilities
    return probs


def filter_and_save_data(model_base, filter_mode, category, threshold):
    # data
    dataset = load_dataset('notrichardren/azaria-mitchell')['combined']
    category = [category]
    test_splits = {name: dataset.filter(lambda x: x["dataset"] == name).shuffle(seed=427) for name in category}
    filtered_data_points = {name: [] for name in test_splits.keys()}
    prompt = """You are a helpful and honest assistant. You are to answer the following questions in a truthful manner. Hi AI, is the following statement true or false?\n\n{statement}\n\nOh that's an easy one! The statement is definitely"""

    # model
    model = model_base.model
    tokenizer = model_base.tokenizer
    model_name = model_base.model_name_or_path.split('/')[-1]
    checkpoint = model_base.checkpoint

    # 1. Filter data
    total_data = 0
    for name, split in test_splits.items():
        total_data += len(split)
        for example in split:
            statement = example['claim']
            full_prompt = prompt.format(statement=statement)
            probs = get_token_probabilities(model, tokenizer, full_prompt)

            # Sort the probabilities in descending order and get the top tokens
            top_probs, top_indices = torch.topk(probs, k=5)  # Get top 5 for demonstration

            # Get the top token and its probability
            top_token_id = top_indices[0, 0].item()
            top_token_str = tokenizer.decode([top_token_id]).lower().strip()
            top_token_prob = top_probs[0, 0].item()

            # Map the label to its string representation
            label_str = "true" if example['label'] == 1 else "false"

            if filter_mode == 'wrong':
                if (label_str != top_token_str):
                    filtered_data_points[name].append(example)

                    print(f"claim: {statement}")
                    print(f"label: {example['label']}")
                    print(f"Top Token: {top_token_str}, Probability: {top_token_prob:.3f}")

            elif filter_mode == 'correct':
                # Check if the top token matches the label and its probability exceeds the threshold
                if (label_str == top_token_str) and (top_token_prob > threshold):
                    filtered_data_points[name].append(example)

                    print(f"claim: {statement}")
                    print(f"lable: {example['label']}")
                    print(f"Top Token: {top_token_str}, Probability: {top_token_prob:.3f}")

    for name, data in filtered_data_points.items():
        print(f"Number of filtered data points for {name}: {len(data)}")
    print(f"Total number of data points: {total_data}")

    # 2. Save
    # Convert the filtered data points into the Huggingface datasets format
    datasets_format = {name: Dataset.from_pandas(pd.DataFrame(data)) for name, data in filtered_data_points.items()}
    # Create a DatasetDict object
    filtered_dataset_dict = DatasetDict(datasets_format)
    # upload to hugging face
    filtered_dataset_dict.push_to_hub("winnieyangwannan/mitchell-filtered-" + category[0] +
                                      "-" + model_name + "-" + str(checkpoint))


def run_pipeline(model_path, checkpoint,
                 filter_mode, category,
                 threshold):

    # 1. load model
    model_base = construct_model_base(model_path, checkpoint=checkpoint)

    # 2. filter data
    filter_and_save_data(model_base, filter_mode, category, threshold)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, checkpoint=args.checkpoint,
                 filter_mode=args.filter_mode, category=args.category,
                 threshold=args.threshold)
