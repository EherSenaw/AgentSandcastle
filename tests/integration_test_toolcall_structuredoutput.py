from src.prompt_template import NAIVE_COMPLETION_RETRY
from src.structured_output import PydanticOutputParser
from src.exceptions import OutputParserException
from src.utils import retrieve_non_think, json_schema_to_base_model

from src.tool_convert import tool

from mlx_lm import load, generate

@tool(parse_docstring=True)
def web_search(query: str) -> str:
	"""Search web with query using DuckDuckGoSearch, to retrieve results found with query.

	Args:
		query: Query to search for.
	
	Returns:
		The string format contents found with search using query.
	"""
	try:
		from duckduckgo_search import DDGS
	except ImportError as e:
		raise ImportError(
			"You must install package `duckduckgo_search` to run this tool: for instance run `pip install duckduckgo-search`."
		) from e

	results = DDGS().text(query, max_results=5)
	if len(results) == 0:
		raise Exception("No results found! Try a less restrictive/shorter query.")
	for res in results:
		print(f"## Search Results (Displaying titles with bodies)\nTITLE: {res['title']}\n{res['body']}\n\n")
	return "\n\n".join([f"TITLE: {res['title']}\n{res['body']}" for res in results])

parse_chain_template = [
	{
		"role": "system",
		"content": "Answer the user query. Wrap the output in `json` tags\n{format_instructions}"
	},
	{
		"role": "user",
		"content": "{query}"
	}
]
if __name__ == "__main__":
	############# CUSTOMIZE #############
	USER_QUERY = "Search for 'chill dog' meme in the internet and explain it."
	MAX_GEN_TOKENS = 1024
	MAX_RETRIES = 3
	model_name = 'mlx-community/Qwen2.5-7B-Instruct-1M-4bit'
	example_func = web_search
	example_func_json_schema = example_func.args_schema.model_json_schema()
	tool_list = '- web_search: {example_func.description}'

	############# CUSTOMIZE #############
	#####################################
	parser = PydanticOutputParser(
		pydantic_object=json_schema_to_base_model(example_func_json_schema)
	) # type: ignore
	parse_chain = list(map(
		lambda x: {
			"role": x["role"],
			"content": x["content"].format(
				format_instructions=parser.get_format_instructions()
			) + "\n**AVAILABLE TOOLS**\n{tool_list}" if x["role"] == "system" else x["content"].format(
				query=USER_QUERY
			)
		}, parse_chain_template
	))
	
	model, tokenizer = load(model_name)
	prompt = tokenizer.apply_chat_template(
		parse_chain,
		add_generation_prompt=True,
		tokenize=False,
	)
	output = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=MAX_GEN_TOKENS)
	output = retrieve_non_think(output)

	try:
		parsed_output = parser.invoke(output)
		print(f"Initial parse succeed.")
	except OutputParserException as e:
		print("Retry with OutputParserException..")
		from src.structured_output import RetryOutputParser
		retry_parser = RetryOutputParser.from_llm(
			parser=parser,
			llm=(model,tokenizer),
			prompt_template=NAIVE_COMPLETION_RETRY,
			max_retries=MAX_RETRIES,
		)
		parsed_output = retry_parser.parse_with_prompt(output, prompt)
	print(f'\n**PARSED**\n{parsed_output}\n********')
	print(f"Type: {type(parsed_output)}")
	print(f".model_dump(): {parsed_output.model_dump()}")
	print(f"\nNow trying tool calling...\n")
	result = None
	try:
		#web_search(parsed_output.model_dump())
		result = web_search.invoke(parsed_output.model_dump())
	except :
		raise
	print(f"\n\n**************\n**FULL RESULT**\n{result}")

