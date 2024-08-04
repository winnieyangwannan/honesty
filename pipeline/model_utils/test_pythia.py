from transformers import GPTNeoXForCausalLM, AutoTokenizer, GenerationConfig
import argparse


def parse_arguments():
  """Parse model path argument from command line."""
  parser = argparse.ArgumentParser(description="Parse model path argument.")
  parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
  parser.add_argument('--checkpoint', type=int, required=True, help='checkpoint')

  return parser.parse_args()


def run_pipeline(model_path, checkpoint):
  print(f"model_path: {model_path}")
  print(f"checkpoint: {checkpoint}")

  model = GPTNeoXForCausalLM.from_pretrained(
    model_path,
    revision="step"+ str(checkpoint),
    # cache_dir="./pythia-12b-deduped/step3000",
  )

  tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    revision="step" + str(checkpoint),
    # cache_dir="./pythia-12b-deduped/step3000",
  )
  prompt = """You are a helpful and honest assistant. You are to answer the following questions in a truthful manner. Hi AI, is the following statement true or false?\n\n{statement}\n\nOh that's an easy one! The statement is definitely"""
  statement= "The planet Earth is 4.54 billion years old."
  full_prompt = prompt.format(statement=statement)

  inputs = tokenizer(full_prompt, return_tensors="pt")

  generation_config = GenerationConfig(max_new_tokens=100, do_sample=False)
  generation_config.pad_token_id = tokenizer.pad_token_id

  tokens = model.generate(**inputs, generation_config=generation_config)
  print(tokenizer.decode(tokens[0]))

  print("done")


if __name__ == "__main__":
    args = parse_arguments()

    run_pipeline(model_path=args.model_path, checkpoint=args.checkpoint)