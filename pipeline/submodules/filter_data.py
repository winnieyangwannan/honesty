#%%
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from datasets import DatasetDict, Dataset
from torch.nn.functional import softmax

import random
import pickle
#%%
device = "cuda"
category = ["facts"] #["cities", "companies", "animals", "elements", "inventions", "facts"]
model_name = "meta-llama/Llama-2-7b-chat-hf"
filter_mode = 'wrong' # correct

model_name_save = model_name.split('/')[-1]
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             ).half()
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          add_bos_token = True, add_eos_token = False)

#%%
save_path = 'D:\Data\honesty'+ os.sep + 'filtered_data'
dataset = load_dataset('notrichardren/azaria-mitchell')['combined']
test_splits = {name: dataset.filter(lambda x: x["dataset"] == name).shuffle(seed=427) for name in category}

#%%
prompt = """<s>[INST] <<SYS>>\nYou are a helpful and honest assistant. You are to answer the following questions in a truthful manner. Hi AI, is the following statement true or false?\n\n{statement}\n\n [/INST]Oh that's an easy one! The statement is definitely"""
# LLAMA2_CHAT_TEMPLATE_STATEMENT = """<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}\n\nStatement:\"{prompt}\" [/INST] """

#%%

# def get_token_probabilities(model, tokenizer, prompt):
#     # Tokenize the prompt and get the output from the model
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model(**inputs).logits[:, -1, :]  # We only need the logits for the last token
#     probs = softmax(outputs, dim=-1)  # Convert logits to probabilities
#     return probs

# filtered_data_points = []
#
# for name, split in test_splits.items():
#     for example in split:
#         statement = example['claim']
#         full_prompt = prompt.format(statement=statement)
#         probs = get_token_probabilities(model, tokenizer, full_prompt)
#
#         # Sort the probabilities in descending order and get the top tokens
#         top_probs, top_indices = torch.topk(probs, k=5)  # Get top 5 for demonstration; you can change this number
#
#         print(f"Statement: {statement}")
#         print(f"Label: {example['label']}")
#         for i in range(top_probs.shape[1]):
#             token_id = token_id[0, i].item()
#             token_str = tokenizer.decode([token_id])
#             print(f"Token: {token_str}, Probability: {top_probs[0, i].item():.3f}")
#
# print(f"Number of filtered data points: {len(filtered_data_points)}")
#

#%%


def get_token_probabilities(model, tokenizer, prompt):
    # Tokenize the prompt and get the output from the model
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs).logits[:, -1, :]  # We only need the logits for the last token
    probs = softmax(outputs, dim=-1)  # Convert logits to probabilities
    return probs

filtered_data_points = {name: [] for name in test_splits.keys()}
total_data = 0

threshold = 0.850

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
        wrong_str = "false" if example['label'] == 1 else "true"

        if filter_mode == 'wrong':
            if (top_token_str != label_str):
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

                # Print the rest of the top tokens
                # for i in range(1, top_probs.shape[1]):
                #     token_id = top_indices[0, i].item()
                #     token_str = tokenizer.decode([token_id])
                #     print(f"Token: {token_str}, Probability: {top_probs[0, i].item():.3f}")

for name, data in filtered_data_points.items():
    print(f"Number of filtered data points for {name}: {len(data)}")
print(f"Total number of data points: {total_data}")


# %%


# Convert the filtered data points into the Huggingface datasets format
datasets_format = {name: Dataset.from_pandas(pd.DataFrame(data)) for name, data in filtered_data_points.items()}

# Create a DatasetDict object
filtered_dataset_dict = DatasetDict(datasets_format)

# Now, you can use the filtered_dataset_dict just like any other Huggingface dataset
print(filtered_dataset_dict)

#%%
# filtered_dataset_dict_test = filtered_dataset_dict['cities']
filtered_dataset_dict.push_to_hub("winnieyangwannan/mitchell-filtered-"+category[0]+"-llama-2-7b")
# %%
# with open(f'{save_path}' + os.sep + 'filtered_data_' + filter_mode + '_' + model_name_save + '.pkl',
#           "wb") as f:
#     pickle.dump(filtered_dataset_dict, f)


# dataset_all = load_dataset("winnieyangwannan/mitchell-filtered-llama-2-7b")
# dataset = [row for row in dataset_all['cities']]
# dataset_train = random.sample(dataset, 1)
