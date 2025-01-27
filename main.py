from src.arguments import build_args
from src.tools import (
	save_file, read_file, list_files
)

import pprint

from huggingface_hub import login
from transformers import (
	HfApiEngine,
	ReactCodeAgent,
)

def main(args):
	if args.hf_api_token != '':
		login(args.hf_api_token)

	llm_engine = HfApiEngine(model=args.llm_model_name)

	agent = ReactCodeAgent(
		tools=[save_file, read_file, list_files],
		llm_engine=llm_engine,
		additional_authorized_imports=['os'],
	)

	agent.run("""
	You are capable of following tools: ['save_file', 'read_file', and 'list_files'].
	Create a single text file named 'temp.txt', filled with about 10 random words, in current directory.
	""")

if __name__ == '__main__':
	args = build_args()
	main(args)