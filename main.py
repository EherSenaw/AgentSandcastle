from src.arguments import build_args
from src.tools import (
	save_file, read_file, list_files,
	open_url_to_PIL_image,
	web_search,
	web_search_retrieve_images,
	ask_user,
)
#from src.together_engine import TogetherAPIEngine
#from src.mlx_engine import MLXEngine
#from src.vllm_engine import vLLMEngine

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

# TogetherAI related. If get errors, do `pip install together`.
def main(args):
	if args.hf_api_token != '':
		login(args.hf_api_token)
	if args.llm_provider == 'huggingface':
		llm_engine = HfApiEngine(model=args.llm_model_name)
	elif args.togetherai_api_token != '' and args.llm_provider == 'togetherai':
		#from src.together_engine import TogetherAPIEngine
		'''
		llm_engine = TogetherAPIEngine(
			model=args.llm_model_name,
			tools=[DuckDuckGoSearchTool(), DuckDuckGoSearchToolReturnImages(), open_url_to_PIL_image],
			max_iteration=5,
			verbose=args.verbose,
			free_tier=True,
			modality_io=args.llm_modality_io,
		)
		'''
		pass
	elif args.llm_provider == 'mlx':
		from src.mlx_engine import MLXEngine
		# NOTE: Testing local llm engine for Apple Silicon.
		llm_engine = MLXEngine(
			model=args.llm_model_name,
			#tools=[DuckDuckGoSearchTool(), DuckDuckGoSearchToolReturnImages(), open_url_to_PIL_image],
			tools=[web_search, web_search_retrieve_images, ask_user],
			max_iteration=5,
			verbose=args.verbose,
			modality_io=args.llm_modality_io,
			max_new_tokens=args.max_new_tokens,
			manual_answer_format=args.manual_answer_format,
		)
	elif args.llm_provider == 'vllm':
		from src.vllm_engine import vLLMEngine
		# NOTE: Testing local llm engine for vLLM(intended for testing of NVIDIA GPU).
		llm_engine = vLLMEngine(
			model=args.llm_model_name,
			tools=[web_search, web_search_retrieve_images, ask_user],
			max_iteration=5,
			verbose=args.verbose,
			modality_io=args.llm_modality_io,
			max_new_tokens=args.max_new_tokens,
			manual_answer_format=args.manual_answer_format,
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