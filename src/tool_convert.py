# NOTE: Following langchain_core/tools/convert.py to make @tool decorator.
import inspect
from typing import Optional, Callable, Literal, Optional, Union, Any, get_type_hints, overload

from pydantic import BaseModel, Field, create_model

#TODO: Remove LangChain dependency & remove unnecessary part from the original code.
#from src.lc_tool_base import BaseTool, Tool, StructuredTool
from langchain_core.tools.base import BaseTool
from langchain_core.tools.simple import Tool
from langchain_core.tools.structured import StructuredTool

@overload
def tool(
	*,
	return_direct: bool = False,
	args_schema: Optional[type] = None,
	infer_schema: bool = True,
	response_format: Literal["content", "content_and_artifact"] = "content",
	parse_docstring: bool = False,
	error_on_invalid_docstring: bool = True,
) -> Callable[[Callable], BaseTool]: ...

@overload
def tool(
	name_or_callable: str,
	*,
	return_direct: bool = False,
	args_schema: Optional[type] = None,
	infer_schema: bool = True,
	response_format: Literal["content", "content_and_artifact"] = "content",
	parse_docstring: bool = False,
	error_on_invalid_docstring: bool = True,
) -> BaseTool: ...

@overload
def tool(
	name_or_callable: Callable,
	*,
	return_direct: bool = False,
	args_schema: Optional[type] = None,
	infer_schema: bool = True,
	response_format: Literal["content", "content_and_artifact"] = "content",
	parse_docstring: bool = False,
	error_on_invalid_docstring: bool = True,
) -> BaseTool: ...

@overload
def tool(
	name_or_callable: str,
	*,
	return_direct: bool = False,
	args_schema: Optional[type] = None,
	infer_schema: bool = True,
	response_format: Literal["content", "content_and_artifact"] = "content",
	parse_docstring: bool = False,
	error_on_invalid_docstring: bool = True,
) -> Callable[[Callable], BaseTool]: ...

def tool(
	name_or_callable: Optional[Union[str, Callable]] = None,
	#runnable: Optional[...]
	*args: Any,
	return_direct: bool = False,
	args_schema: Optional[type] = None,
	infer_schema: bool = True,
	response_format: Literal["content", "content_and_artifact"] = "content",
	parse_docstring: bool = False,
	error_on_invalid_docstring: bool = True,
) -> Union[
	BaseTool,
	Callable[[Callable], BaseTool],
]:
	"""Make tools out of functions, can be used with or without arguments.

	Args:
		name_or_callable: Optional name of the tool or the callable to be
			converted to a tool. Must be provided as a positional argument.
		return_direct: Whether to return directly from the tool rather
			than continuing the agent loop. Defaults to False.
		args_schema: optional argument schema for user to specify.
			Defaults to None.
		infer_schema: Whether to infer the schema of the arguments from
			the function's signature. This also makes the resultant tool
			accept a dictionary input to its `run()` function.
			Defaults to True.
		response_format: The tool response format. If "content" then the output of
			the tool is interpreted as the contents of a ToolMessage. If
			"content_and_artifact" then the output is expected to be a two-tuple
			corresponding to the (content, artifact) of a ToolMessage.
			Defaults to "content".
		parse_docstring: if ``infer_schema`` and ``parse_docstring``, will attempt to
			parse parameter descriptions from Google Style function docstrings.
			Defaults to False.
		error_on_invalid_docstring: if ``parses_docstring`` is provided, configure
			whether to raise ValueError on invalid Google Style docstrings.
			Defaults to True.

	Returns:
		The tool.

	Requires:
		- Function must be of type (str) -> str
		- Function must have a docstring
	
	Examples:
		.. code-block:: python

			@tool
			def search_api(query: str) -> str:
				# Searches the API for the query.
				return
			
			@tool("search", return_direct=True)
			@tool(response_format="content_and_artifact")
			...
	.. 
	Parse Google-style docstrings:

		.. code-block:: python

			@tool(parse_docstring=True)
			def foo(bar: str, baz: int) -> str:
				\"\"\"The foo.

				Args:
					bar: The bar.
					baz: The baz.
				\"\"\"
				return bar
			
			foo.args_schema.model_json_schema()
		
		.. code-block:: python

			{
				"title": "foo",
				"description": "The foo.",
				"type": "object",
				"properties": {
					"bar": {
						"title": "Bar",
						"description": "The bar.",
						"type": "string"
					},
					"baz": {
						"title": "Baz",
						"description": "The baz.",
						"type": "integer"
					}
				},
				"required": [
					"bar",
					"baz"
				]
			}

		Note that parsing by default will raise ``ValueError`` if the docstring
		is condsidered invalid. A docstring is considered invalid if it contains
		arguments not in the function signature, or is unable to be parsed into
		a summary and "Args:" blocks.
	"""

	def _create_tool_factory(
		tool_name: str,
	) -> Callable[[Callable], BaseTool]:
		"""Create a decorator that takes a callable and returns a tool.

		Args:
			tool_name: The name that will be assigned to the tool.

		Returns:
			A function that takes a callable and returns a tool.
		"""

		def _tool_factory(dec_func: Callable) -> BaseTool:
			if inspect.iscoroutinefunction(dec_func):
				coroutine = dec_func
				func = None
				schema = args_schema
				description = None
			else:
				coroutine = None
				func = dec_func
				schema = args_schema
				description = None
			
			if infer_schema or args_schema is not None:
				return StructuredTool.from_function(
					func,
					coroutine,
					name=tool_name,
					description=description,
					return_direct=return_direct,
					args_schema=schema,
					infer_schema=infer_schema,
					response_format=response_format,
					parse_docstring=parse_docstring,
					error_on_invalid_docstring=error_on_invalid_docstring,
				)
			# If someone doesn't want a schema applied, we must treat it as
			# a simple string->string function
			if dec_func.__doc__ is None:
				msg = (
					"Function must have a docstring if "
					"description not provided and infer_schema is False."
				)
				raise ValueError(msg)
			return Tool(
				name=tool_name,
				func=func,
				description=f"{tool_name} tool",
				return_direct=return_direct,
				coroutine=coroutine,
				response_format=response_format,
			)
		return _tool_factory
	
	if len(args) != 0:
		# Triggered if a user attempts to use positional arguments that
		# do not exist in the function signature
		# e.g., @tool("name", runnable, "extra_arg")
		# Here, "extra_arg" is not a valid argument
		msg = "Too many arguments for tool decorator. A decorator "
		raise ValueError(msg)
	
	if name_or_callable is not None:
		if callable(name_or_callable) and hasattr(name_or_callable, "__name__"):
			# Used as a decorator without parameters
			# @tool
			# def my_tool():
			# 	pass
			return _create_tool_factory(name_or_callable.__name__)(name_or_callable)
		elif isinstance(name_or_callable, str):
			# Use with a new name for the tool
			# @tool("search")
			# def my_tool():
			# 	pass
			#
			# or
			#
			# @tool("search", parse_docstring=True)
			# def my_tool():
			#	pass
			return _create_tool_factory(name_or_callable)
		else:
			msg = (
				f"The first argument must be a string or a callable with a __name__ "
				f"for tool decorator. Got {type(name_or_callable)}"
			)
			raise ValueError(msg)
	else:
		# Tool is used as a decorator with paramters specified
		# @tool(parse_docstring=True)
		# def my_tool():
		#	pass
		def _partial(func: Callable) -> BaseTool:
			"""Partial function that takes a callable and returns a tool."""
			name_ = func.__name__
			tool_factory = _create_tool_factory(name_)
			return tool_factory(func)
		
		return _partial