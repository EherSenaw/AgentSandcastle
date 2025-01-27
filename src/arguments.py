import argparse

def build_args():
	# Build argument parser and parse arguments.
	parser = argparse.ArgumentParser()
	parser.add_argument('--hf_api_token',
		type=str, required=True, default='',
		help='Huggingface API Token for log-in.'
	)
	parser.add_argument('--llm_model_name',
		type=str, default='meta-llama/Llama-3-70B-Instruct'
	)

	args = parser.parse_args()
	return args