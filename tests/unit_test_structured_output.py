from collections.abc import Mapping
from typing import List, Optional, Dict, Any, Callable
from typing_extensions import Annotated
from pydantic import BaseModel, Field, TypeAdapter, WrapValidator

from src.tools import *
from src.structured_output import PydanticOutputParser
#from src.structured_output import JsonOutputParser
from src.tool_convert import tool
from src.exceptions import OutputParserException

from mlx_lm import load, generate

import re
THINK_REGEXP = re.compile(r"<think>.*<\/think>", re.DOTALL)
def retrieve_non_think(str_response:str) -> str:
	if '<think>' not in str_response:
		return str_response
	retval = ''
	n = len(str_response)
	for candidate in THINK_REGEXP.finditer(str_response):
		s, e = candidate.span()

		if s > 0:	left = str_response[:s]
		else:		left = ''

		if e < n:
			right = str_response[e:]
			break
	return right.strip()

# NOTE: DEBUG for structured_output parsing from LLm output and integrating it with tool calling.
class Person(BaseModel):
	"""Information about a person."""

	person_name: Optional[str] = Field(..., description="The name of the person.")

def allow_unvalidated_dict(v: Any, handler: Callable[[Any], Any]) -> Any:
	print(f"Allow_unvalidated_dict: `{v}`, type: {type(v)}")
	if isinstance(v, Mapping):
		print(f"Type: Mapping, return to list(dict)")
		return list(dict(v))
	elif isinstance(v, dict):
		return list(v)
	return handler(v)

class People(BaseModel):
	"""Identifying information about all people in a text."""

	count: Union[int,str] = Field(..., description="The number of the Person in people.")
	"""The total number of the persons."""

	persons: Optional[List[
		Annotated["Person",None]
		#Annotated["Person",WrapValidator(allow_unvalidated_dict)]
	]] = Field(..., description="The list of the `Person` object.")
	"""The List of `Person` objects, only provided if their names are given."""

	def model_post_init(self, __context: object) -> None:
		for idx, person in enumerate(self.persons):
			if isinstance(person, str):
				self.persons[idx] = Person.model_validate(person)

PydanticOutputParser.model_rebuild()
parser = PydanticOutputParser(pydantic_object=People) # type: ignore
#import src.structured_output as sso
#parser = sso.JsonOutputParser(pydantic_object=People)
parse_chain_template = [
	{
		"role": "system",
		"content": "Answer the user query. Wrap the output in `json` tags\n{format_instructions}".format(
			format_instructions=parser.get_format_instructions()
		),
	},
	{
		"role": "user",
		"content": "{query}".format(
			query="Suppose there were five people in the room at first. And a random person entered the room. Then, how many pepole are there?"
		),
	},
]

NAIVE_COMPLETION_RETRY = """Prompt:
{prompt}
Completion:
{completion}

Above, the Completion did not satisfy the constraints given in the Prompt.
Please try again:"""

#model = 'mlx-community/DeepScaleR-1.5B-Preview-4bit'
model = 'mlx-community/Qwen2.5-7B-Instruct-1M-4bit'
client, tokenizer = load(model)
prompt = tokenizer.apply_chat_template(
	parse_chain_template,
	add_generation_prompt=True,
	tokenize=False,
)
output = generate(client, tokenizer, prompt=prompt, verbose=True, max_tokens=1024)
output = retrieve_non_think(output)
try:
	parsed_output = parser.parse_result(output)
	print(f"Initial parse succeed.")
except OutputParserException as e:
	#print(e)
	print("Retry with OutputParserException...")
	#from langchain.output_parsers import RetryOutputParser
	from src.structured_output import RetryOutputParser
	retry_parser = RetryOutputParser.from_llm(parser=parser, llm=(client,tokenizer), prompt_template=NAIVE_COMPLETION_RETRY, max_retries=3)
	parsed_output = retry_parser.parse_with_prompt(output, prompt)
print(f"\n**PARSED**\n{parsed_output}\n********")
