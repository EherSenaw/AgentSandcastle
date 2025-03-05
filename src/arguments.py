import argparse
import os

def build_args():
	# Build argument parser and parse arguments.
	parser = argparse.ArgumentParser()
	parser.add_argument('--hf_api_token',
		type=str, default='',
		help='Huggingface API Token for log-in.'
	)
	parser.add_argument('--togetherai_api_token',
		type=str, default='',
		help='TogetherAI API Token.'
	)

	parser.add_argument('--llm_provider',
		type=str, default='huggingface',
		help='Choose LLM engine API provider (or Local use with \'mlx\') to use. {\'huggingface\', \'togetherai\', \'mlx\'}. Default: \'huggingface\'. ')

	parser.add_argument('--llm_model_name',
		type=str, default='meta-llama/Llama-3-70B-Instruct'
	)
	parser.add_argument('--llm_modality_io',
		type=str, default='text/text',
		help='Provide dedicated I/O modality types for the LLM engine. Provide with following format: `Input1,Input2,.../Output1,Output2,...`. Default: text/text '
	)
	parser.add_argument('--max_new_tokens',
		type=int, default=256,
		help='MAXIMUM number of tokens to generate, per each LLM call. Default: 256.')

	parser.add_argument('--verbose',
		action='store_true',
		help='Set default \'verbose\' behaviors.'
	)

	parser.add_argument('--manual_answer_format',
		type=str, default='',
		help='If you want to instruct the agent to answer with your own format manually, provide instruction here. This possibly will disable the auto-parsing & calling of the tool of LLM.'
	)

	parser.add_argument('--initial_question',
		type=str, default='Which one is bigger? 1.02 or 1.2?',
		help='Initial question that should be asked to the Agent.'
	)

	args = parser.parse_args()
	assert not (args.hf_api_token == '' and args.togetherai_api_token == '' and args.llm_provider in {'togetherai', 'huggingface'}), "One of the LLM provider API token should be given."
	if args.togetherai_api_token != '':
		os.environ['TOGETHER_API_KEY'] = args.togetherai_api_token
		
	return args