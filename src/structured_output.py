# NOTE: Based on LangChain's langchain-extract example github.
import re
import json
import contextlib
import pydantic

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generic, Optional, Union, TypeVar, Annotated, Callable, TypedDict
from typing_extensions import override

from json import JSONDecodeError
from jsonschema import exceptions
from jsonschema.validators import Draft202012Validator
from pydantic import BaseModel, Field, validator, SkipValidation

from langchain_core.runnables import RunnableSerializable #, Runnable
#from langchain_core.load.serializable import Serializable
#from langchain_core.language_models import LanguageModelOutput

from src.prompt_template import (
	NAIVE_COMPLETION_RETRY, JSON_FORMAT_INSTRUCTIONS, PYDANTIC_FORMAT_INSTRUCTIONS,
)
from src.utils import (
	retrieve_non_think,
	JSON_MARKDOWN_REGEXP, JSON_STRIP_CHARS
)
from src.exceptions import OutputParserException
# NOTE: For my usage, do not need streaming output handling. Therefore, just use BaseOutputParser.
#from langchain_core.output_parsers.transform import BaseCumulativeTransformOutputParser
#from langchain_core.outputs import Generation
#from langchain_core.utils.json import (parse_partial_json)
#from langchain_core.messages import AnyMessage
#from langchain_core.utils.pydantic import PYDANTIC_MAJOR_VERSION

T = TypeVar("T")

def _replace_new_line(match: re.Match[str]) -> str:
	value = match.group(2)
	value = re.sub(r"\n", r"\\n", value)
	value = re.sub(r"\r", r"\\r", value)
	value = re.sub(r"\t", r"\\t", value)
	value = re.sub(r'(?<!\\)"', r"\"", value)
	return match.group(1) + value + match.group(3)
def _custom_parser(multiline_string: str) -> str:
	"""The LLM response for `action_input` may be a multiline
	string containing unescaped newlines, tabs or quotes. This function
	replaces those characters with their escaped counterparts.
	(newlines in JSON must be double-escaped: `\\n`).
	"""
	if isinstance(multiline_string, (bytes, bytearray)):
		multiline_string = multiline_string.decode()
	multiline_string = re.sub(
		r'("action_input"\:\s*")(.*?)(")',
		_replace_new_line,
		multiline_string,
		flags=re.DOTALL,
	)
	return multiline_string
def parse_partial_json(s: str, *, strict: bool = False) -> Any:
	"""Parse a JSON string that may be missing closing braces.

	Args:
		s: The JSON string to parse.
		strict: Whether to use strict parsing. Defaults to False.
	
	Returns:
		The parsed JSON object as a Python dictionary.
	"""
	# Attempt to parse the string as-is.
	try:
		return json.loads(s, strict=strict)
	except json.JSONDecodeError:
		pass

	# Initialize variables.
	new_chars = []
	stack = []
	is_inside_string = False
	escaped = False

	# Process each character in the string one at a time.
	for char in s:
		if is_inside_string:
			if char == '"' and not escaped:
				is_inside_string = False
			elif char == "\n" and not escaped:
				char = "\\n" # Replace the newline character with the escape sequence.
			elif char == "\\":
				escaped = not escaped
			else:
				escaped = False
		else:
			if char == '"':
				is_inside_string = True
			elif char == "{":
				stack.append("}")
			elif char == "[":
				stack.append("]")
			elif char == "}" or char == "]":
				if stack and stack[-1] == char:
					stack.pop()
				else:
					# Mismatched closing character; the input is malformed.
					return None
		# Append the processed character to the new string.
		new_chars.append(char)
	# If we're still inside a string at the end of processing,
	# we need to close the string.
	if is_inside_string:
		if escaped: # Remove unterminated escape character.
			new_chars.pop()
		new_chars.append('"')
	# Reverse the stack to get the closing characters.
	stack.reverse()
	# Try to parse mods of string until we succeed or run out of characters.
	while new_chars:
		# Close any remaining open structures in the reverse
		# order that they were opened.
		# Attempt to parse the modified string as JSON.
		try:
			return json.loads("".join(new_chars + stack), strict=strict)
		except json.JSONDecodeError:
			# If we still can't parse the string as JSON,
			# try removing the last character.
			new_chars.pop()
	# If we got here, we ran out of characters to remove
	# and still couldn't parse the string as JSON, so return the parse error
	# for the original string.
	return json.loads(s, strict=strict)
def _parse_json(
	json_str: str, *, parser: Callable[[str], Any] = parse_partial_json
) -> dict:
	# Strip whitespace, newlines, backtick from the start and end
	json_str = json_str.strip(JSON_STRIP_CHARS)
	# handle newlines and other special characters inside the returned value
	json_str = _custom_parser(json_str)
	# Parse the JSON string into a Python dictionary
	return parser(json_str)
def parse_json_markdown(
	json_string: str, *, parser: Callable[[str], Any] = parse_partial_json
) -> dict:
	"""Parse a JSON string from a Markdown string.

	Args:
		json_string: The Markdown string.
	
	Returns:
		The parsed JSON object as a Python dictionary.
	"""
	try:
		return _parse_json(json_string, parser=parser)
	except json.JSONDecodeError:
		# Try to find JSON string within triple backticks
		match = JSON_MARKDOWN_REGEXP.search(json_string)

		# If no match found, assume the entire string is a JSON string
		# Else, use the content within the backticks
		json_str = json_string if match is None else match.group(2)
	return _parse_json(json_str, parser=parser)

def validate_json_schema(schema: Dict[str, Any]) -> None:
	"""Validate a JSON schema."""
	try:
		Draft202012Validator.check_schema(schema)
	except exceptions.ValidationError as e:
		# NOTE: Currently not necessarily be HTTPException (which requires FASTapi installed).
		'''
		raise HTTPException(
			status_code=422, detail=f"Not a valid JSON schema: {e.message}"
		)
		'''
		raise ValueError(f"Not a valid JSON schema: {e.message}")

# NOTE: Pydantic-style declaration used in extraction example of Langchain.
'''
class ExtractionExample(BaseModel):
	"""An example extraction.

	This example consists of a text and the expected output of the extraction.
	Setting each field an `optional` allows the model to decline to extract it.
	`Description` of each field is ued by the LLM.
	Having a good description can help improve extraction results.
	"""

	text: str = Field(..., description="The input text")
	output: List[Dict[str, Any]] = Field(
		...,
		description="The expected output of the example. A list of objects."
	)

class ExtractRequest(BaseModel):
	"""Request body for the extract endpoint."""
	text: str = Field(..., description="The text to extract from.")
	json_schema: Dict[str, Any] = Field(
		...,
		description="JSON schema that describes what content should be extracted "
		"from the text.",
		alias="schema",
	)
	instructions: Optional[str] = Field(
		None, description="Supplemental system instructions."
	)
	examples: Optional[List[ExtractionExample]] = Field(
		None, description="Examples of extractions."
	)
	model_name: Optional[str] = Field("gpt-3.5-turbo", description="Chat model to use.")

	@validator("json_schema")
	def validate_schema(cls, v: Any) -> Dict[str, Any]:
		"""Validate the schema."""
		validate_json_schema(v)
		return v
'''

# NOTE: Tweaked from & Based on source code of LangChain's BaseOutputParser.
class BaseLLMOutputParser(Generic[T], ABC):
	"""Abstract base class for parsing the outputs of a model."""

	@abstractmethod
	def parse_result(self, result: str, *, partial: bool = False) -> T:
		"""Parse a list of candidate model output text into a specific format.

		Args:
			result: A text to be parsed. The Generations are assumed
				to be different candidate outputs for a single model input.
			partial: Whether to parse the output as a partial result. This is useful
				for parsers that can parse partial results. Default is False.

		Returns:
			Structured output.
		"""
	# NOTE: Since this project does not handle async parsing, ignore aparse_result() implementation.
	#		Should implement if needed.
	#async def aparse_result(...)
Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)
#class RunnableSerializable(Serializable, Runnable[Input, Output]):
class Runnable_dummy(Generic[Input, Output]):
	""" DUMMY for debug """

	name: Optional[str] = None

	model_config = pydantic.ConfigDict(
		# Suppress warnings from pydantic protected namespaces
		# (e.g., `model_`)
		protected_namespaces=(),
	)

class BaseOutputParser(
	BaseLLMOutputParser,
	#Serializable, Runnable_dummy[type[str], T]
	RunnableSerializable[type[str], T]
	#RunnableSerializable[LanguageModelOutput, T]
):
	"""Base class to parse the output of an LLM call.

	Output parsers help structure language model responses.

	Example:
		.. code-block:: python

			class BooleanOutputParser(BaseOutputParser[bool]):
				true_val: str = "YES"
				false_val: str = "NO"

				def parse(self, text:str) -> bool:
					cleaned_text = text.strip().upper()
					if cleaned_text not in (self.true_val.upper(), self.false_val.upper()):
						raise OutputParserException(
							f"BooleanOutputParser expected output value to either be "
							f"{self.true_val} or {self.false_val} {case-insensitive}. "
							f"Received {cleaned_text}."
						)
					return cleaned_text == self.true_val.upper()
				
				@property
				def _type(self) -> str:
					return "boolean_output_parser
	"""
	@property
	@override
	def InputType(self) -> Any:
		"""Return the input type for the parser."""
		#return Union[str, AnyMessage]
		return str
	
	@property
	@override
	def OutputType(self) -> type[T]:
		"""Return the output type for the parser.

		This property is inferred from the first type argument of the class.

		Raises:
			TypeError: If the class doesn't have an inferable OutputType.
		"""
		for base in self.__class__.mro():
			if hasattr(base, "__pydantic_generic_metadata__"):
				metadata = base.__pydantic_generic_metadta__
				if "args" in metadata and len(metadata["args"]) > 0:
					return metadata["args"][0]
		
		msg = (
			f"Class {self.__class__.__name__} doesn't have an inferable OutputType."
			"Override the OutputType property to specify the output type."
		)
		raise TypeError(msg)
	
	# NOTE: Current usage of this class is for Json/Pydantic parsing.
	#	Which does not use .invoke() methods.
	#	Therefore, skip implementation of invoke() of LangChain for now.

	# Tweaked implementation of invoke() method from original Langchain implementation.
	def invoke(
		self,
		input: str,
		#config: Optional[RunnableConfig] = None,
		**kwargs: Any,
	)-> T:
		"""Transform a single input into an output."""
		# NOTE: _call_with_config() inherited from Runnable class.
		return self._call_with_config(
			lambda inner_input: self.parse_result(inner_input),
			input,
			None,
			run_type="parser",
		)


	def parse_result(self, result: str, *, partial: bool = False) -> T:
		"""Parse the result of an LLM call to a specific format.

		The return value is parsed from result, which is assumed to
			be the highest-likelihood Generation.

		Args:
			result: The result of the LLM call to be parsed.
			partial: Whether to parse the output as a partial result.
				This is useful for parsers that can parse partial results.
				Default is False.
		
		Returns:
			Structured output.
		"""
		return self.parse(result)

	@abstractmethod
	def parse(self, text:str) -> T:
		"""Parse a single string model output into some structure.

		Args:
			text: String output of a language model.

		Returns:
			Structured output.
		"""
	
	@property
	def _type(self) -> str:
		"""Return the output parser type for serialization."""
		msg = (
			f"_type property is not implemented in class {self.__class__.__name__}."
			" This is required for serialization."
		)
		raise NotImplementedError(msg)
	
	def dict(self, **kwargs: Any) -> dict:
		"""Return dictionary representation of output parser."""
		output_parser_dict = super().dict(**kwargs)
		with contextlib.suppress(NotImplementedError):
			output_parser_dict["_type"] = self._type
		return output_parser_dict


# NOTE: Tweaked source code from LangChain's JsonOutputParser & PydanticOutputParser docs/sources.
#PydanticBaseModel = Union[pydantic.BaseModel, pydantic.v1.BaseModel] # type: ignore
#PydanticBaseModel = pydantic.BaseModel # type: ignore
TBaseModel = TypeVar("TBaseModel", bound=pydantic.BaseModel)

class JsonOutputParser(BaseOutputParser[Any]):
	"""Parse the output of an LLM call to a JSON object.
	"""
	
	pydantic_object: Annotated[Optional[type[TBaseModel]], SkipValidation()] = None # type: ignore
	"""The Pydantic object to use for validation.
	If None, no validation is performed."""

	def _get_schema(self, pydantic_object: type[TBaseModel]) -> dict[str, Any]:
		if issubclass(pydantic_object, pydantic.BaseModel):
			return pydantic_object.model_json_schema()
		elif issubclass(pydantic_object, pydantic.v1.BaseModel):
			return pydantic_object.schema()
	
	def parse_result(self, result: str, *, partial: bool = False) -> Any:
		"""Parse the result of an LLM call to a JSON object.

		Args:
			result: The result of the LLM call.
			partial: Whether to parse partial JSON objects.
				If True, the output will be a JSON object containing
				all the keys that have been returned so far.
				If False, the output will be the full JSON object.
				Default is False.
		
		Returns:
			The parsed JSON object.
		
		Raises:
			OutputParserException: If the output is not valid JSON.
		"""
		# NOTE: if got List[str], use this.
		#text = result[0].text.strip()
		if result is None:
			return {}
		text = result.strip()
		if partial:
			try:
				return parse_json_markdown(text)
			except JSONDecodeError:
				return None
		else:
			try:
				return parse_json_markdown(text)
			except JSONDecodeError as e:
				msg = f"Invalid json output: {text}"
				raise OutputParserException(msg, llm_output=text) from e
	
	def parse(self, text: str) -> Any:
		"""Parse the output of an LLM call to a JSON object.

		Args:
			text: The output of the LLM call.

		Returns:
			The parsed JSON object.
		"""
		return self.parse_result(result=text)

	def get_format_instructions(self) -> str:
		"""Return the format instructions for the JSON output.

		Returns:
			The format instructions for the JSON output.
		"""
		if self.pydantic_object is None:
			return "Return a JSON object."
		else:
			# Copy schema to avoid altering original Pydantic schema.
			schema = dict(self._get_schema(self.pydantic_object).items())

			# Remove extraneous fields.
			reduced_schema = schema
			if "title" in reduced_schema:
				del reduced_schema["title"]
			if "type" in reduced_schema:
				del reduced_schema["type"]
			# Ensure json in context is well-formed with double quotes.
			schema_str = json.dumps(reduced_schema, ensure_ascii=False)
			return JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)
	
	@property
	def _type(self) -> str:
		return "simple_json_output_parser"

class PydanticOutputParser(JsonOutputParser, Generic[TBaseModel]):
	"""Parse an output using a pydantic model."""

	pydantic_object: Annotated[type[TBaseModel], SkipValidation()] # type: ignore
	"""The pydantic model to parse."""

	def _parse_obj(self, obj: dict) -> TBaseModel:
		# NOTE: If got error, check PYDANTIC_MAJOR_VERSION >= 2.
		# 	For V1, should be implemented differently.
		try:
			if issubclass(self.pydantic_object, pydantic.BaseModel):
				return self.pydantic_object.model_validate(obj)
			elif issubclass(self.pydantic_object, pydantic.v1.BaseModel):
				return self.pydantic_object.parse_obj(obj)
			else:
				msg = f"Unsupported model version for PydanticOutputParser: \
						{self.pydantic_object.__class__}"
				raise OutputParserException(msg)
		except (pydantic.ValidationError, pydantic.v1.ValidationError) as e:
			raise self._parser_exception(e, obj) from e
	
	def _parser_exception(
			self, e: Exception, json_object: dict
	) -> OutputParserException:
		json_string = json.dumps(json_object)
		name = self.pydantic_object.__name__
		msg = f"Failed to parse {name} from llm_output {json_string}. Got: {e}"
		return OutputParserException(msg, llm_output=json_string)
	
	def parse_result(
		self, result: str, *, partial: bool = False
	) -> Optional[TBaseModel]:
		"""Parse the result of an LLM call to a pydantic object.
		
		Args:
			result: The result of the LLM call.
			partial: Whether to parse partial JSON objects.
				If True, the output will be a JSON object containing
				all the keys that have been returned so far.
				Defaults to False.
		
		Returns:
			The parsed pydantic object.
		"""
		try:
			json_object = super().parse_result(result)
			return self._parse_obj(json_object)
		except OutputParserException:
			if partial:
				return None
			raise
	
	def parse(self, text:str) -> TBaseModel:
		"""Parse the output of an LLM call to a pydantic object.

		Args:
			text: The output of the LLM call.
		
		Returns:
			The parsed pydantic object.
		"""
		return super().parse(text)

	def get_format_instructions(self) -> str:
		"""Return the format instructions for the JSON output.
		
		Returns:
			The format instructions for the JSON output.
		"""
		# Copy schema to avoid altering original Pydantic schema.
		schema = dict(self.pydantic_object.model_json_schema().items())

		# Remove extraneous fields.
		reduced_schema = schema
		if "title" in reduced_schema:
			del reduced_schema["title"]
		if "type" in reduced_schema:
			del reduced_schema["type"]
		# Ensure json in context is well-formed with double quotes.
		schema_str = json.dumps(reduced_schema, ensure_ascii=False)

		return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)
	
	@property
	def _type(self) -> str:
		return "pydantic"
	
	@property
	@override
	def OutputType(self) -> type[TBaseModel]:
		"""Return the pydantic model."""
		return self.pydantic_object

class RetryOutputParserRetryChainInput(TypedDict):
	prompt: str
	completion: str

class RetryOutputParser(BaseOutputParser[T]):
	"""Wrap a parser and try to fix parsing errors.

	Does this by passing the original prompt and the completion to another
	LLM, and telling it the completion did not satisfy criteria in the prompt.
	"""

	parser: Annotated[BaseOutputParser[T], SkipValidation()]
	"""The parser to use to parse the output."""

	llm: Annotated[
		Any,
		SkipValidation(),
	]
	"""The LLM engine to use to retry the completion. Additionally can include tokenizer or processor in the list."""

	prompt_template: str = NAIVE_COMPLETION_RETRY
	"""The prompt template to use to retry the completion."""

	max_retries: int = 1
	"""The maximum number of times to retry the parse."""

	@classmethod
	def from_llm(
		cls,
		llm: Any, # Instance of any compatible LLMEngine
		parser: BaseOutputParser[T],
		prompt_template: str = NAIVE_COMPLETION_RETRY,
		max_retries: int = 1,
	) -> T:
		"""Create an RetryOutputParser from a language model and a parser.

		Args:
			llm: llm to use for fixing
			parser: parser to use for parsing
			prompt_template: prompt template to use for fixing. should contain keys of 'prompt' and 'completion'.
			max_retries: Maximum number of retries to parse.
		
		Returns:
			RetryOutputParser
		"""
		return cls(parser=parser, llm=llm, prompt_template=prompt_template, max_retries=max_retries)
	
	def parse_with_prompt(self, llm_output:str, prompt_value: str, llm_provider: str = 'mlx', sampling_params: Optional[str] = None) -> T:
		"""Parse the output of an LLM call using a wrapper parser.

		Args:
			llm_output: The LLM output to parse.
			prompt_value: The prompt to use to parse the LLM output.
			llm_provider: To use proper `generate` code snippet for each provider.
			sampling_params: Only needed for `vllm` provider.
		
		Returns:
			The parsed output.
		"""
		retries = 0
		if llm_provider == 'mlx':
			from mlx_lm import generate
		elif llm_provider == 'vllm':
			assert sampling_params, 'RetryOutputParser.parse_with_prompt() requires `sampling_params` for `llm_provider=vllm`.'
		else:
			raise NotImplementedError('Currently RetryOutputParser not implemented with llm providers other than \{`mlx`,`vllm`\}. ')

		while retries <= self.max_retries:
			try:
				return self.parser.parse(llm_output)
			except OutputParserException as e:
				if retries == self.max_retries:
					raise e
				else:
					retries += 1
					print(f"Retry with:\n{self.prompt_template.format(prompt=prompt_value,completion=llm_output)}\n")
					if llm_provider == 'mlx':
						llm_output = generate(
							self.llm[0], # llm
							self.llm[1], # tokenizer
							prompt=self.prompt_template.format(prompt=prompt_value,completion=llm_output),
							verbose=True,
							max_tokens=1024,
						)
					elif llm_provider == 'vllm':
						prompt = self.llm.get_tokenizer().apply_chat_template(
							self.prompt_template.format(prompt=prompt_value,completion=llm_output),
							add_generation_prompt=True,
							tokenize=False,
						)
						llm_output = self.llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)[0].outputs[0].text.strip()
					if llm_output:
						llm_output = retrieve_non_think(llm_output)
		
		raise OutputParserException("Failed to parse even with retrying")
	
	def parse(self, llm_output: str) -> T:
		raise NotImplementedError("This OutputParser can only be called by the `parse_with_prompt` method.")
	
	def get_format_instructions(self) -> str:
		return self.parser.get_format_instructions()

	@property
	def _type(self) -> str:
		return "retry"
	
	@property
	def OutputType(self) -> type[T]:
		return self.parser.OutputType

			

PydanticOutputParser.model_rebuild()
