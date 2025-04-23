# NOTE: To use with chat-bot-like FastAPI endpoint.
from typing import Optional, Any

from src.arguments import build_args
from src.tools import (
	web_search,
	ask_user,
)

def builder(args):
	engine_class = None
	if args.llm_provider == 'mlx':
		# NOTE: Testing local llm engine for Apple Silicon.
		from src.mlx_engine import MLXEngine
		engine_class = MLXEngine
	elif args.llm_provider == 'vllm':
		# NOTE: Testing local llm engine for vLLM(intended for testing of NVIDIA GPU).
		from src.vllm_engine import vLLMEngine
		engine_class = vLLMEngine
	else:
		raise NotImplementedError("Currently 'mlx' and 'vllm' only supported.")
	llm_engine = engine_class(
		model=args.llm_model_name,
		tools=[web_search, ask_user],
		max_iteration=3,
		verbose=args.verbose,
		modality_io=args.llm_modality_io,
		max_new_tokens=args.max_new_tokens,
		manual_answer_format=args.manual_answer_format,
	)
	return llm_engine

def build_engine(args: Optional[Any] = None):
	return builder(build_args(args))