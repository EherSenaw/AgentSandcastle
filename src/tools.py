from typing import List, Optional, Dict
# NOTE: For PoC, use huggingface(smolagents)'s Tool interface.
# TODO: Re-define tools and its interface for further improvement.

# For custom tool implementation with use of @tool,
# check `github.com/huggingface/smolagents/blob/main/src/smolagents/tools.py#L840`.
# which begins with `def tool(tool_function: Callable) -> Tool:`...

""" Before `SmolAgents`
from transformers import (
	HfApiEngine,
	ReactCodeAgent,
	tool, Tool, load_tool, stream_to_gradio,
)
from transformers.agents import (
	DuckDuckGoSearchTool
)
"""
""" After `SmolAgents`
"""
from smolagents import (
	tool, Tool, load_tool, stream_to_gradio,
	DuckDuckGoSearchTool,
)
from transformers import (
	HfApiEngine,
	ReactCodeAgent,
)

def verify_hf_tools(tools: List[Optional[Tool]]) -> Dict[str, Optional[Tool]]:
	for t in tools:
		#assert isinstance(t, Tool) or issubclass(t, Tool), f"Given {str(t)} is not a valid HuggingFace-compatible `Tool`."
		assert isinstance(t, Tool), f"Given {str(t)} is not a valid HuggingFace-compatible `Tool`."
	return {t.name: t for t in tools}

@tool
def save_file(filename: str, content: str) -> str:
	"""Saves the content to the file.
	Args:
		filename: Name of the file to be saved.
		content: Content of the file to be saved.
	Returns:
		Path of the saved file.
	"""
	with open(filename, "w+", encoding='utf-8') as f:
		f.write(content)
	return f"File '{filename}' has been saved."

@tool
def read_file(filename: str) -> str:
	"""Reads the content of the file.
	Args:
		filename: Name of the file to read.
	Returns:
		content: Content of the file.
	"""
	try:
		with open(filename, "r", encoding='utf-8') as f:
			content = f.read()
		return content
	except FileNotFoundError:
		return f"File '{filename}' does not exist."

@tool
def list_files(directory: str = ".") -> str:
	"""Check list of files in the directory.
	Args:
		directory: path of the directory (default: current directory.)
	Returns:
		List of files.
	"""
	import os
	files = os.listdir(directory)
	return "\n".join(files)
