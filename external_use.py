# NOTE: To use with chat-bot-like FastAPI endpoint.
from typing import Optional, Any, TypeVar, Callable, AsyncGenerator
import functools

from src.arguments import build_args
from src.tools import (
	web_search,
	ask_user,
)

### Helper for async streaming <-> final output separation handling.
T = TypeVar("T")
def with_final(gen_func: Callable[..., AsyncGenerator[T, None]]
) -> Callable[..., AsyncGenerator[tuple[T, bool], None]]:
	"""
	Wrap an async-generator that yields chunks of type T into one
	that yields (chunk, is_final_flag), and then one final (full, True).
	"""
	@functools.wraps(gen_func)
	async def wrapper(*args, **kwargs) -> AsyncGenerator[tuple[T, bool], None]:
		it = gen_func(*args, **kwargs).__aiter__()
		try:
			prev = await it.__anext__() # Get first item (or fail)
		except StopAsyncIteration:
			return						# No items at all
		while True:
			try:
				curr = await it.__anext__()
				yield prev, False # intermediate chunk
				prev = curr
			except StopAsyncIteration:
				yield prev, True
				break
	return wrapper

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