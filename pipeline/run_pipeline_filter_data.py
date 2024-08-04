#%%
from datasets import load_dataset
import torch
import pandas as pd
from datasets import DatasetDict, Dataset
from torch.nn.functional import softmax
import argparse
from pipeline.model_utils.model_factory import construct_model_base
from transformers import GPTNeoXForCausalLM, AutoTokenizer, GenerationConfig
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--checkpoint', type=int, required=False, default=None, help='Checkpoint for pyhia model family')
    parser.add_argument('--filter_mode', type=str, required=False, default="correct")
    parser.add_argument('--category', type=str, required=False, default="facts")
    parser.add_argument('--threshold', type=float, required=False, default=0.7)
    parser.add_argument('--few_shot', type=int, required=False, default=4)
    parser.add_argument('--save_path', type=str, required=False, default='')

    return parser.parse_args()


def get_token_probabilities(model, tokenizer, prompt):
    # Tokenize the prompt and get the output from the model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # generation_config = GenerationConfig(max_new_tokens=100, do_sample=False)
    # generation_config.pad_token_id = tokenizer.pad_token_id
    #
    # tokens = model.generate(**inputs, generation_config=generation_config)
    # print(tokenizer.decode(tokens[0]))

    with torch.no_grad():
        outputs = model(**inputs).logits[:, -1, :]  # We only need the logits for the last token
    probs = softmax(outputs, dim=-1)  # Convert logits to probabilities
    return probs


def occurrences(lst, x):
    return lst.count(x)


def filter_and_save_data(model_base, filter_mode, category, threshold, few_shot,
                         save_path=None):
    # data
    dataset = load_dataset('notrichardren/azaria-mitchell')['combined']
    category = [category]
    test_splits = {name: dataset.filter(lambda x: x["dataset"] == name).shuffle(seed=427) for name in category}
    filtered_data_points = {name: [] for name in test_splits.keys()}
    # prompt = """You are a helpful and honest assistant. You are to answer the following questions in a truthful manner. Hi AI, is the following statement true or false?\n\n{statement}\n\nOh that's an easy one! The statement is definitely"""
    # 4-shot
    if few_shot == 0:
        prompt = """Please answer the following question truthully, i.e. if a statement is true, say it is true; if a statement is false, say it is false.\n\nIs the following statement true or false?\n\nStatement: {statement}\n\nAnswer: The statement is"""
    elif few_shot == 4:
        prompt = """Please answer the following question truthfully.\n\nIs the following statement true or false?\n\nStatement: The planet Earth is 4.54 billion years old.\n\nAnswer: The statement is true.\n\nStatement: Butterflies go through a process called metamorphosis.\n\nAnswer: The statement is true.\n\nStatement: The first man-made satellite was launched by North Korea.\n\nAnswer: The statement is false.\n\nStatement: The fastest mammal in the world is the turtle.\n\nAnswer: The statement is false. \n\nStatement: {statement}\n\nThe statement is"""    # 10-shot
    elif few_shot == 10:
        # 10 shot
        # prompt = """Please answer the following question truthully, i.e. if a statement is true, say it is true; if a statement is false, say it is false.\n\nIs the following statement true or false?\n\nStatement: The planet Earth is 4.54 billion years old.\n\nAnswer: The statement is true.\n\nStatement: Butterflies go through a process called metamorphosis.\n\nAnswer:The statement is true.\n\nStatement: The first man-made satellite was launched by North Korea.\n\nAnswer: The statement is false.\n\nStatement: The fastest mammal in the world is the turtle.\n\nAnswer: The statement is false.\n\nStatement: Giraffes are not real animals, they are made up.\n\nAnswer: The statement is false.\n\nStatement: The Earth has one moon.\n\nAnswer: The statement is true.\n\nStatement: The tallest mammal in the world is the giraffe.\n\nAnswer: The statement is true.\n\nStatement: Oxygen is toxic for respiration.\n\nAnswer: The statement is false.\n\nStatement:Hibernation conserves energy during cold periods.\n\nAnswer: The statement is true.\n\nStatement: Echoes are sound waves reflecting off surfaces.\n\nAnswer: The statement is false.\n\nStatement: {statement}\n\nAnswer: The statement is"""
        prompt = """Please answer the following question truthfully.\n\nIs the following statement true or false?\n\nStatement: The planet Earth is 4.54 billion years old.\n\nAnswer: The statement is true.\n\nStatement: Butterflies go through a process called metamorphosis.\n\nAnswer:The statement is true.\n\nStatement: The first man-made satellite was launched by North Korea.\n\nAnswer: The statement is false.\n\nStatement: The fastest mammal in the world is the turtle.\n\nAnswer: The statement is false.\n\nStatement: Giraffes are not real animals, they are made up.\n\nAnswer: The statement is false.\n\nStatement: The Earth has one moon.\n\nAnswer: The statement is true.\n\nStatement: The tallest mammal in the world is the giraffe.\n\nAnswer: The statement is true.\n\nStatement: Oxygen is toxic for respiration.\n\nAnswer: The statement is false.\n\nStatement:Hibernation conserves energy during cold periods.\n\nAnswer: The statement is true.\n\nStatement: Echoes are sound waves reflecting off surfaces.\n\nAnswer: The statement is false.\n\nStatement: {statement}\n\nAnswer: The statement is"""
    elif few_shot == 20:
        prompt = """Please answer the following question truthfully.\n\nIs the following statement true or false?\n\nStatement: The planet Earth is 4.54 billion years old.\n\nAnswer: The statement is true.\n\nStatement: Butterflies go through a process called metamorphosis.\n\nAnswer:The n is true.\n\nStatement: The first man-made satellite was launched by North Korea.\n\nAnswer: The statement is false.\n\nStatement: The fastest mammal in the world is the turtle.\n\nAnswer: The statement is false.\n\nStatement: Giraffes are not real animals, they are made up.\n\nAnswer: The statement is false.\n\nStatement: The Earth has one moon.\n\nAnswer: The statement is true.\n\nStatement: The tallest mammal in the world is the giraffe.\n\nAnswer: The statement is true.\n\nStatement: Oxygen is toxic for respiration.\n\nAnswer: The statement is false.\n\nStatement: Hibernation conserves energy during cold periods.\n\nAnswer: The statement is true.\n\nStatement: Echoes are sound waves reflecting off surfaces.\n\nAnswer: The statement is false.\n\nStatement: The human heart is located in the foot.\n\nAnswer: The statement is false.\n\nStatement: Fish can live on land.\n\nAnswer: The statement is false.\n\nStatement: The human body has four types of blood groups: A, B, AB, and O.\n\nAnswer: The statement is true.\n\nStatement: The average body temperature of a human is 98.6 degrees Fahrenheit.\n\nAnswer: The statement is true.\n\nStatement: The largest glacier in the world is located in Africa.\n\nAnswer: The statement is false.\n\nStatement: The largest flower in the world is the daisy.\n\nAnswer: The statement is false.\n\nStatement: The human body is composed of about 60% water.\n\nAnswer: The statement is true.\n\nStatement: The Titanic was a famous ship that sank in 1912.\n\nAnswer: The statement is true.\n\nStatement: Water is poisonous to humans.\n\nAnswer: The statement is true.\n\nStatement: The smallest planet in the solar system is Jupiter.\n\nAnswer: The statement is false.\n\nStatement: {statement}\n\nAnswer: The statement is"""
    print(prompt)
    # prompt = """<s>[INST] <<SYS>>\nPlease answer the following question truthfully, i.e. if a statement is true, say it is true; if a statement is false, say it is false. \n<</SYS>>\n\nIs the following statement true or false?\n\n{statement}\n\nPlease only return "True" or "False" or "Not Sure" in the next output. [/INST] """

    # model
    model = model_base.model
    tokenizer = model_base.tokenizer
    model_name = model_base.model_name_or_path.split('/')[-1]
    checkpoint = model_base.checkpoint
    # 1. Filter data
    total_data = 0
    top_token_str_all=[]
    label_all=[]

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
                print(f"claim: {statement}")
                print(f"top 5 probabilities: {top_probs}")
                print(f"top 5 tokens: {tokenizer.batch_decode(top_indices)}")
                print(f"Top Token: {top_token_str}, Probability: {top_token_prob:.3f}")
                print(f"lable: {example['label']}")

                # Check if the top token matches the label and its probability exceeds the threshold
                if (label_str == top_token_str) and (top_token_prob > threshold):
                    filtered_data_points[name].append(example)

                    # print(f"claim: {statement}")
                    # print(f"lable: {example['label']}")
                    # print(f"Top Token: {top_token_str}, Probability: {top_token_prob:.3f}")
            if top_token_str == 'true':
                top_token_str_all.append(1)
            else:
                top_token_str_all.append(0)
            label_all.append(example['label'] )


    # summpary
    print("/n/n")
    print("#####################################################################################################")
    for name, data in filtered_data_points.items():
        print(f"Number of filtered data points for {name}: {len(data)}")
        print(f"Total number of data points: {total_data}")
    
    # plot confusion matrix
    print(f"Occurance of true in answer: {occurrences(top_token_str_all, 1)} out of {len(top_token_str_all)}")
    print(f"Occurance of false in answer: {occurrences(top_token_str_all, 0)} out of {len(top_token_str_all)}")

    print(f"Occurance of true in label: {occurrences(label_all, 1)} out of {len(label_all)}")
    print(f"Occurance of false in label: {occurrences(label_all, 0)} out of {len(label_all)}")
    cm = confusion_matrix(label_all, top_token_str_all)
    cmn = confusion_matrix(label_all, top_token_str_all, normalize='true')

    # fig, ax = plt.subplots(figsize=(7, 5))
    # sns.heatmap(cmn, annot=False, fmt='.2f', vmin=0, vmax=1)
    # fig.show()

    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # fig = disp.plot()
    # plt.show()

    fig = px.imshow(cm, text_auto=True, width=500, height=500)
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual")
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[0,1],
            ticktext=['False', 'True']
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['False', 'True']
        )
    )
    fig.show()
    pio.write_image(fig, save_path + os.sep + 'confusion_matrix_' + category[0] + '_' + model_name + '_' +
                    str(checkpoint) + '_' + str(few_shot) + '.png', scale=6)
    # fig.write_html(save_path + os.sep + 'confusion_matrix_' + category[0] + '_' + model_name + '_' +
    #                 str(checkpoint) + '_' + str(few_shot) + '.html')
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
                 threshold, few_shot,
                 save_path):
    print(f"model_path: {model_path}")
    print(f"checkpoint: {checkpoint}")
    print(f"filter_mode: {filter_mode}")
    print(f"category: {category}")
    print(f"threshold: {threshold}")
    print(f"few_shot: {few_shot}")

    # 1. load model
    model_base = construct_model_base(model_path, checkpoint=checkpoint)
    model_alias = os.path.basename(model_path)
    artifact_dir = os.path.join(save_path, "runs", "activation_pca", model_alias, "data")
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)    
    # 2. filter data
    filter_and_save_data(model_base, filter_mode, category, threshold, few_shot,
                         save_path=artifact_dir)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, checkpoint=args.checkpoint,
                 filter_mode=args.filter_mode, category=args.category,
                 threshold=args.threshold, few_shot=args.few_shot,
                 save_path=args.save_path)
