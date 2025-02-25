from typing import List, Optional, Dict, Union
import PIL
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
	#tool,
	Tool, load_tool, stream_to_gradio,
	DuckDuckGoSearchTool,
)
from transformers import (
	HfApiEngine,
	ReactCodeAgent,
)

from src.tool_convert import tool

# For direct conversion of tool spec into string.
from transformers.utils import get_json_schema

def verify_hf_tools(tools: List[Optional[Tool]]) -> Dict[str, Optional[Tool]]:
	for t in tools:
		#assert isinstance(t, Tool) or issubclass(t, Tool), f"Given {str(t)} is not a valid HuggingFace-compatible `Tool`."
		assert isinstance(t, Tool), f"Given {str(t)} is not a valid HuggingFace-compatible `Tool`."
	return {t.name: t for t in tools}

# TODO: Remove HF-dependency by direct parsing of docstring.
def verify_tools_docstring(tools: List[Optional[Tool]]) -> Dict[str, Optional[Tool]]:
	verifier = lambda tool: tool.name if hasattr(tool, 'name') else tool.__name__

@tool
def save_file(filename: str, content: str) -> str:
	"""Saves the content to the file.

	Args:
		filename: Name of the file to be saved.
		content: Content of the file to be saved.
		format: Modality format of the content. Default: text. But can be image.

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

@tool
def open_url_to_PIL_image(url: str) -> PIL.Image:
	"""Populate PIL.Image object from given image url.

	Args:
		url: URL of the image to open.

	Returns:
		PIL.Image object.
	"""
	import PIL
	import requests
	image_response = requests.get(url)
	img = PIL.Image.open(image_response.raw)
	return img

# NOTE: Based on Huggingface's Smoalgent's implementation of DuckDuckGoSearchTool(),
#		But with the expansion of retrieving image results.
class DuckDuckGoSearchToolReturnImages(Tool):
	name = "web_search_image"
	description = """Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results."""
	inputs = {"query": {"type": "string", "description": "The search query to perform."}}
	output_type = "string"
	#output_type = "List[PIL.Image]"

	def __init__(self, max_results=5, **kwargs):
		super().__init__()
		self.max_results = max_results
		try:
			from duckduckgo_search import DDGS
		except ImportError as e:
			raise ImportError(
				"You must install package `duckduckgo_search` to run this tool: for instance run `pip install duckduckgo-search`."
			) from e
		self.ddgs = DDGS(**kwargs)
	
	def forward(self, query:str) -> str:
		results = self.ddgs.images(query, max_results=self.max_results)
		# Each item consists of: {'title': str, 'image': str(URL), 'thumbnail': str(URL), 'url':str(Original article's URL), 'height': int, 'width': int, 'source': str(SearchEngine)}
		if len(results) == 0:
			raise Exception("No results found! Try a less restrictive/shorter query.")
		# TODO: Post-process the search results
		#		1. Formulate the results into List[PIL.Image]
		image_titles = [result['title'] for result in results]
		image_urls = [result['image'] for result in results]
		print(f"## Search Results (Displaying image titles)\n\n" + "\n\n".join(image_titles))
		return "\n\n".join(image_urls)

@tool
def web_search_retrieve_images(query: str) -> str:
	"""Search web with query using DuckDuckGoSearch, to retrieve URLs of the images found with query.

	Args:
		query: Query to search for.
	
	Returns:
		The URLS of the images found with query.
	"""
	try:
		from duckduckgo_search import DDGS
	except ImportError as e:
		raise ImportError(
			"You must install package `duckduckgo_search` to run this tool: for instance run `pip install duckduckgo-search`."
		) from e
	results = DDGS(max_results=5).images(query, max_results=5)
	if len(results) == 0:
		raise Exception("No results found! Try a less restrictive/shorter query.")
	image_titles = [result['title'] for result in results]
	image_urls = [result['image'] for result in results]
	print(f"## Search Results (Displaying image titles)\n\n" + "\n\n".join(image_titles))
	return "\n\n".join(image_urls)