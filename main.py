from src.arguments import build_args
from src.tools import (
	save_file, read_file, list_files,
	open_url_to_PIL_image,
	DuckDuckGoSearchToolReturnImages,
)
from src.together_engine import TogetherAPIEngine
from src.mlx_engine import MLXEngine

import pprint
from typing import List, Dict

# Huggingface related.
from huggingface_hub import login
from transformers import (
	HfApiEngine,
	ReactCodeAgent,
)
from smolagents import (
	DuckDuckGoSearchTool,
	GoogleSearchTool,
	PythonInterpreterTool,
)

# TogetherAI related. If get errors, do `pip install together`.
#from together import Together

def single_run(args):
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

def main(args):
	if args.hf_api_token != '':
		login(args.hf_api_token)
	if args.llm_provider == 'huggingface':
		llm_engine = HfApiEngine(model=args.llm_model_name)
	elif args.togetherai_api_token != '' and args.llm_provider == 'togetherai':
		llm_engine = TogetherAPIEngine(
			model=args.llm_model_name,
			tools=[DuckDuckGoSearchTool(), DuckDuckGoSearchToolReturnImages(), open_url_to_PIL_image],
			max_iteration=5,
			verbose=True,
			free_tier=True,
			modality_io=args.llm_modality_io,
		)
	elif args.llm_provider == 'mlx':
		# NOTE: Testing local llm engine for Apple Silicon.
		llm_engine = MLXEngine(
			model=args.llm_model_name,
			tools=[DuckDuckGoSearchTool(), DuckDuckGoSearchToolReturnImages(), open_url_to_PIL_image],
			max_iteration=5,
			verbose=True,
			modality_io=args.llm_modality_io,
			max_new_tokens=1024,
		)
	else:
		raise ValueError("Provide at least one LLM API provider.")

	''' Using Hf Agent API.
	agent = ReactCodeAgent(
		tools=[save_file, read_file, list_files],
		llm_engine=llm_engine,
		additional_authorized_imports=['os'],
	)
	agent.run("""
	""")
	'''

	# Define agent-controlled multi-turn conversation (no-agent)
	#user_defined_task = "Which one is bigger? 1.02 or 1.2?"
	user_defined_task = args.initial_question
	'''
	memory = [user_defined_task]
	while llm_continue_decision(memory):
		action = llm_decide_action(memory):
		observation = execute_action(action):
		memory += [action, observation]
	'''
	final_answer = llm_engine(user_defined_task)
	print(f"\n\n**** FINAL ANSWER ****\n{final_answer}\n**** FINAL ANSWER ****\n")

if __name__ == '__main__':
	args = build_args()
	main(args)