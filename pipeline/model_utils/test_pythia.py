from transformers import GPTNeoXForCausalLM, AutoTokenizer
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

  inputs = tokenizer("Hello, I am", return_tensors="pt")
  tokens = model.generate(**inputs)
  print(tokenizer.decode(tokens[0]))

  print("done")


if __name__ == "__main__":
    args = parse_arguments()

    run_pipeline(model_path=args.model_path, checkpoint=args.checkpoint)