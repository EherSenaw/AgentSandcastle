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
		help='Choose LLM engine API provider to use. {\'huggingface\', \'togetherai\'}. Default: \'huggingface\'. ')
	parser.add_argument('--llm_model_name',
		type=str, default='meta-llama/Llama-3-70B-Instruct'
	)

	parser.add_argument('--initial_question',
		type=str, default='Which one is bigger? 1.02 or 1.2?',
		help='Initial question that should be asked to the Agent.'
	)

	args = parser.parse_args()
	assert not (args.hf_api_token == '' and args.togetherai_api_token == ''), "One of the LLM provider API token should be given."
	if args.togetherai_api_token != '':
		os.environ['TOGETHER_API_KEY'] = args.togetherai_api_token
		
	return args