from typing import List, Optional, Dict, Union
import PIL
import PIL.Image
# NOTE: (Legacy, not anymore.) For PoC, use huggingface(smolagents)'s Tool interface.
#		For custom tool implementation with use of @tool,
# 		check `github.com/huggingface/smolagents/blob/main/src/smolagents/tools.py#L840`.
# 		which begins with `def tool(tool_function: Callable) -> Tool:`...

from src.tool_convert import tool
# NOTE: Now implemented LangChain-like, but with additional auto-parsing / using.
# 		No-dependency for the tools declared from now.
# 		!IMPORTANT!: MUST include Google-style Docstrings to prevent malfunctioning.
# NOTE: Try @tool with `parse_docstring=True`, for easy conversion 
#		from user-defined python function to LLM-callable tool.
#		Docstring will be parsed into Pydantic object (and JSON).
#		This information will be provided into LLM
#		and used during structured output validation / tool calling.

@tool(parse_docstring=True)
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

@tool(parse_docstring=True)
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

@tool(parse_docstring=True)
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

@tool(parse_docstring=True)
def open_url_to_PIL_image(url: str) -> PIL.Image:
	"""Populate PIL.Image object from given image url.

	Args:
		url: URL of the image to open.

	Returns:
		PIL.Image object.
	"""
	import requests
	image_response = requests.get(url)
	img = PIL.Image.open(image_response.raw)
	return img

@tool(parse_docstring=True)
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
	image_urls = [result['image'] for result in results]
	#image_titles = [result['title'] for result in results]
	#print(f"## Search Results (Displaying image titles)\n\n" + "\n\n".join(image_titles))
	return "\n\n".join(image_urls)

@tool(parse_docstring=True)
def web_search(query: Optional[str]) -> str:
	"""Search web with query using DuckDuckGoSearch, to retrieve results found with query.

	Args:
		query: Query to search for.

	Returns:
		The string format contents found with search using query.
	"""
	verbose = False # NOTE: Set to True if you need to see search results in this tool call.

	try:
		from duckduckgo_search import DDGS
	except ImportError as e :
		raise ImportError(
			"You must install package `duckduckgo_search` to run this tool. For instance, run `pip install duckduckgo-search`."
		) from e
	
	results = DDGS().text(query, max_results=5)
	assert len(results) > 0, "No results found for tool(`web_search`). Try a less restrictive or shorter query."

	if verbose:
		for res in results:
			print(f"## Search Results (Displaying titles with bodies)\nTITLE: {res['title']}\n{res['body']}\n\n")
	search_result_str = "\n\n".join([f"TITLE: {res['title']}\n{res['body']}" for res in results])
	return search_result_str

########### DUMMY TOOLS ##############
# !IMPORTANT! These (dummy tools) are not used as tool, but used for easy structuring of output.
######################################
# NOTE: Dummy tool for easy-parsing of the request from the query.
@tool(parse_docstring=True)
def process_request(tool_request: Optional[str], helper_request: Optional[str], answer: Optional[str], rationale: Optional[str]) -> str:
	"""Process the requests(tool-use, helper-use) if each of them are not Nil nor None. 

	Args:
		tool_request: The name of the tool to use. Set to be Nil or None if no tool needed.
		helper_request: The name of the helper to use. Set to be Nil or None if no helper needed.
		answer: The generated direct answer, if given. Set to be Nil or None if directly answering the question was impossible.
			If both `tool_request` and `helper_request` are stated to be not needed, this should be returned.
		rationale: The rationale behind the choice made.
	
	Returns:
		String containing observation of the results (tool-use and/or helper request).
	"""
	pass
# NOTE: Dummy tool for easy-parsing of the Yes/No question-answering.
@tool(parse_docstring=True)
def process_binary(yes_or_no: Optional[str]) -> str:
	"""Process the binary answer, given either 'yes' or 'no'.

	Args:
		yes_or_no: The binary answer indicator. Set to be either 'yes' or 'no', depending on the question.

	Returns:
		String containing observation of the results (differ for 'yes' or 'no').
	"""
	pass

############### Legacy (Smolagent) ############
'''
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
'''
