from src.arguments import build_args

from huggingface_hub import login
from transformers import (
	HfApiEngine,
	ReactCodeAgent,
	tool, Tool, load_tool, stream_to_gradio,
)
from transformers.agents import (
	DuckDuckGoSearchTool
)

def main(args):
	login(args.hf_api_token)

	llm_engine = HfApiEngine(model=args.llm_model_name)

	agent = ReactCodeAgent(
		tools=[],
		llm_engine=llm_engine,
	)

	agent.run("Say welcome words, in short.")

if __name__ == '__main__':
	args = build_args()
	main(args)