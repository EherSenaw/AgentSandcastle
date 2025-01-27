from transformers import (
	HfApiEngine,
	ReactCodeAgent,
	tool, Tool, load_tool, stream_to_gradio,
)
from transformers.agents import (
	DuckDuckGoSearchTool
)

@tool
def save_file(filename: str, content: str) -> str:
	"""
	Function that saves the content to the file.
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
	"""
	Function that reads the content of the file.
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
	"""
	Check list of files in the directory.
	Args:
		directory: path of the directory (default: current directory.)
	Returns:
		List of files.
	"""
	import os
	files = os.listdir(directory)
	return "\n".join(files)
