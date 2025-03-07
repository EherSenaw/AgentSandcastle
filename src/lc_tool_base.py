# NOTE: Based on following classes:
#	from langchain_core.tools.base import BaseTool
#	from langchain_core.tools.simple import Tool
#	from langchain_core.tools.structured import StructuredTool
from abc import abstractmethod
from typing import Union, Any, Annotated, Optional, Callable, Literal
from pydantic import BaseModel, Field, SkipValidation, ValidationError, ConfigDict
from contextvars import copy_context
from inspect import signature

from langchain_core.messages.tool import ToolCall
from langchain_core.runnables import RunnableSerializable, RunnableConfig, patch_config
from langchain_core.runnables.config import _set_config_context
#from langchain_core.utils.function_calling import _parse_google_docstring
from langchain_core.utils.pydantic import TypeBaseModel, is_basemodel_subclass, get_fields
from langchain_core.callbacks import Callbacks, CallbackManager

class SchemaAnnotationError(TypeError):
	"""Raised when 'args_schema' is missing or has an incorrect type annotation."""
class ToolException(Exception):
	"""Optional exception that tool throws when execution error occurs.

	When this exception is thrown, the agent will not stop working,
	but it will handle the exception according to the handle_tool_error
	variable of the tool, and the processing result will be returned
	to the agent as observation, and printed in red on the console.
	"""

class InjectedToolArg:
	"""Annotation for a Tool arg that is **not** meant to be generated by a model."""

ArgsSchema = Union[TypeBaseModel, dict[str, Any]]

class BaseTool(RunnableSerializable[Union[str, dict, ToolCall], Any]):
	"""Interface tools must implement."""

	def __init_subclass__(cls, **kwargs: Any) -> None:
		"""Create the definition of the new tool class."""
		super().__init_subclass__(**kwargs)

		args_schema_type = cls.__annotations__.get("args_schema", None)
		if args_schema_type is not None and args_schema_type == BaseModel:
			# Throw errors for common mis-annotations.
			typehint_mandate = """
class ChildTool(BaseTool):
	...
    args_schema: Type[BaseModel] = SchemaClass
	..."""
			name = cls.__name__
			msg = (
				f"Tool definition for {name} must include valid type annotations"
				f" for argument 'args_schema' to behave as expected.\n"
				f"Expected annotation of 'Type[BaseModel]'"
				f" but got '{args_schema_type}'.\n"
				f"Expected class looks like:\n"
				f"{typehint_mandate}"
			)
			raise SchemaAnnotationError(msg)
	name: str
	"""The unique name of the tool that clearly communicates its purpose."""
	description: str
	"""Used to tell the model how/when/why to use the tool.

	You can provide few-shot exapmles as a part of the description.
	"""

	args_schema: Annotated[Optional[ArgsSchema], SkipValidation()] = Field(
		default=None, description="The tool schema."
	)
	"""Pydantic model class to validate and parse the tool's input arguments.
	
	Args schema should be either:

	- A subclass of pydantic.BaseModel.
	or
	- A JSON schema dict
	"""
	return_direct: bool = False
	"""Whether to return the tool's output directly.
	
	Setting this to True means
	that after the tool is called, the AgentExecutor will stop looping.
	"""
	verbose: bool = False
	"""Whether to log the tool's progress."""

	callbacks: Callbacks = Field(default=None, exclude=True)
	"""Callbacks to be called during tool execution."""

	tags: Optional[list[str]] = None
	"""Optional list of tags associated with the tool. Defaults to None.
	These tags will be associated with each call to this tool,
	and passed as arguments to the handlers defined in `callbacks`.
	You can use these to eg identify a specific instance of a tool with its use case.
	"""
	metadata: Optional[dict[str, Any]] = None
	"""Optioanl metadata associated with the tool. Defaults to None.
	This metadata will be associated with each call to this tool,
	and passed as arguments to the handlers defined in `callbacks`.
	You can use these to eg identify a specific instance of a tool with its use case.
	"""

	handle_tool_error: Optional[Union[bool, str, Callable[[ToolException], str]]] = (
		False
	)
	"""Handle the content of the ToolException thrown."""

	handle_validation_error: Optional[
		Union[bool, str, Callable[[ValidationError], str]]
	] = False
	"""Handle the content of the ValidationError thrown."""

	response_format: Literal["content", "content_and_artifact"] = "content"
	"""The tool response format. Defaults to 'content'.

	If "content" then the output of the tool is interpreted as the contents of a
	ToolMessage. if "content_and_artifact" then the output is expected to be a
	two-tuple corresponding to the (content, artifact) of a ToolMessage.
	"""

	def __init__(self, **kwargs: Any) -> None:
		"""Initialize the tool."""
		if (
			"args_schema" in kwargs
			and kwargs["args_schema"] is not None
			and not is_basemodel_subclass(kwargs["args_schema"])
			and not isinstance(kwargs["args_schema"], dict)
		):
			msg = (
				"args_schema must be a subclass of pydantic BaseModel or "
				f"a JSON schema dict. Got: {kwargs['args_schema']}."
			)
			raise TypeError(msg)
		super().__init__(**kwargs)

	model_config = ConfigDict(
		arbitrary_types_allowed=True,
	)

	@property
	def is_single_input(self) -> bool:
		"""Whether the tool only accepts a single input."""
		keys = {k for k in self.args if k != "kwargs"}
		return len(keys) == 1
	
	@property
	def args(self) -> dict:
		if isinstance(self.args_schema, dict):
			json_schema = self.args_schema
		else:
			input_schema = self.get_input_schema()
			json_schema = input_schema.model_json_schema()
		return json_schema["properties"]
	
	@property
	def tool_call_schema(self) -> ArgsSchema:
		if isinstance(self.args_schema, dict):
			if self.description:
				return {
					**self.args_schema,
					"description": self.description,
				}
			return self.args_schema
		full_schema = self.get_input_schema()
		fields = []
		for name, type_ in get_all_basemodel_annotations(full_schema).items():
			if not _is_injected_arg_type(type_):
				fields.append(name)
		return _create_subset_model(
			self.name, full_schema, fields, fn_description=self.description
		)

	# --- Runnable ---

	def get_input_schema(
		self, config: Optional[RunnableConfig] = None
	) -> type[BaseModel]:
		"""The tool's input schema.
		
		Args:
			config: The configuration for the tool.
			
		Returns:
			The input schema for the tool.
		"""
		if self.args_schema is not None:
			if isinstance(self.args_schema, dict):
				return super().get_input_schema(config)
			return self.args_schema
		else:
			return create_schema_from_function(self.name, self._run)
	
	def invoke(
		self,
		input: Union[str, dict, ToolCall],
		config: Optional[RunnableConfig] = None,
		**kwargs: Any,
	) -> Any:
		tool_input, kwargs = _prep_run_args(input, config, **kwargs)
		return self.run(tool_input, **kwargs)
	
	#async def ainvoke(...)

	# --- Tool ---

	def _parse_input(
		self, tool_input: Union[str, dict], tool_call_id: Optional[str]
	) -> Union[str, dict[str, Any]]:
		"""Convert tool input to a pydantic model.

		Args:
			tool_input: The input to the tool.
		"""
		input_args = self.args_schema
		if isinstance(tool_input, str):
			if input_args is not None:
				if isinstance(input_args, dict):
					msg = (
						"String tool inputs are not allowed when "
						"using tools with JSON schema args_schema."
					)
					raise ValueError(msg)
				key_ = next(iter(get_fields(input_args).keys()))
				if hasattr(input_args, "model_validate"):
					input_args.model_validate({key_: tool_input})
				else:
					input_args.parse_obj({key_: tool_input})
			return tool_input
		else:
			if input_args is not None:
				if isinstance(input_args, dict):
					return tool_input
				elif issubclass(input_args, BaseModel):
					for k, v in get_all_basemodel_annotations(input_args).items():
						if (
							_is_injected_arg_type(v, injected_type=InjectedToolCallId)
							and k not in tool_input
						):
							if tool_call_id is None:
								msg = (
									"When tool includes an InjectedToolCallId "
									"argument, tool must always be invoked with a full "
									"model ToolCall of the form: {'args': {...}, "
									"'name': '...', 'type': 'tool_call', "
									"'tool_call_id': '...'}"
								)
								raise ValueError(msg)
							tool_input[k] = tool_call_id
					result = input_args.model_validate(tool_input)
					result_dict = result.model_dump()
				else:
					msg = (
						"args_schema must be a Pydantic BaseModel, "
						f"got {self.args_schema}"
					)
					raise NotImplementedError(msg)
				return {
					k: getattr(result, k)
					for k, v in result_dict.items()
					if k in tool_input
				}
			return tool_input
	
	@abstractmethod
	def _run(self, *args: Any, **kwargs: Any) -> Any:
		"""Use the tool.
		
		Add run_manager: Optional[CallbackManagerForToolRun] = None
		to child implementations to enable tracing.
		"""

	def _to_args_and_kwargs(
		self, tool_input: Union[str, dict], tool_call_id: Optional[str]
	) -> tuple[tuple, dict]:
		if (
			self.args_schema is not None
			and isinstance(self.args_schema, type)
			and is_basemodel_subclass(self.args_schema)
			and not get_fields(self.args_schema)
		):
			# StructuredTool with no args
			return (), {}
		tool_input = self._parse_input(tool_input, tool_call_id)
		# For backwards compatibility, if run_input is a string,
		# pass as a positional argument.
		if isinstance(tool_input, str):
			return (tool_input,), {}
		else:
			return (), tool_input
	
	def run(
		self,
		tool_input: Union[str, dict[str, Any]],
		verbose: Optional[bool] = None,
		start_color: Optional[str] = "green",
		color: Optional[str] = "green",
		callbacks: Callbacks = None,
		*,
		tags: Optional[list[str]] = None,
		metadata: Optional[dict[str, Any]] = None,
		run_name: Optional[str] = None,
		run_id: Optional[uuid.UUID] = None,
		config: Optional[RunnableConfig] = None,
		tool_call_id: Optional[str] = None,
		**kwargs: Any,
	) -> Any:
		"""Run the tool.

		Args:
			tool_input: The input to the tool.
			verbose: Whether to log the tool's progress. Defaults to None.
			start_color: The color to use when starting the tool. Defaults to 'green'.
			color: The color to use when ending the tool. Defaults to 'green'.
			callbacks: Callbacks to be called during tool execution. Defaults to None.
			tags: Optional list of tags associated with the tool. Defaults to None.
			metadata: Optional metadata associated with the tool. Defaults to None.
			run_name: The name of the run. Defaults to None.
			run_id: The id of the run. Defaults to None.
			config: The configuration for the tool. Defaults to None.
			tool_call_id: The id of the tool call. Defaults to None.
			kwargs: Keyword arguments to be passed to tool callbacks
		
		Returns:
			The output of the tool.

		Raises:
			ToolException: If an error occurs during tool execution.
		"""
		callback_manager = CallbackManager.configure(
			callbacks,
			self.callbacks,
			self.verbose or bool(verbose),
			tags,
			self.tags,
			metadata,
			self.metadata,
		)
		run_manager = callback_manager.on_tool_start(
			{"name": self.name, "description": self.description},
			tool_input if isinstance(tool_input, str) else str(tool_input),
			color=start_color,
			name=run_name,
			run_id=run_id,
			# Inputs by definition should always be dicts.
			# If violated, replace inputs with `None` value for the callback instead.
			inputs=tool_input if isinstance(tool_input, dict) else None,
			**kwargs,
		)

		content = None
		artifact = None
		status = "success"
		error_to_raise: Union[Exception, KeyboardInterrupt, None] = None
		try:
			child_config = patch_config(config, callbacks=run_manager.get_child())
			context = copy_context()
			context.run(_set_config_context, child_config)
			tool_args, tool_kwargs = self._to_args_and_kwargs(tool_input, tool_call_id)
			if signature(self._run).parameters.get("run_manager"):
				tool_kwargs = tool_kwargs | {"run_manager": run_manager}
			if config_param := _get_runnable_config_param(self._run):
				tool_kwargs = tool_kwargs | {config_param: config}
			response = context.run(self._run, *tool_args, **tool_kwargs)
			if self.response_format == "content_and_artifact":
				if not isinstance(response, tuple) or len(response) != 2:
					msg = (
						"Since response_format='content_and_artifact' "
						"a two-tuple of the message content and raw tool output is "
						f"expected. Instead generated response of type: "
						f"{type(response)}."
					)
					error_to_raise = ValueError(msg)
				else:
					content, artifact = response
			else:
				content = response
		except ValidationError as e:
			if not self.handle_validation_error:
				error_to_raise = e
			else:
				content = _handle_validation_error(e, flag=self.handle_validation_error)
				status = "error"
		except ToolException as e:
			if not self.handle_tool_error:
				error_to_raise = e
			else:
				content = _handle_tool_error(e, flag=self.handle_tool_error)
				status = "error"
		except (Exception, KeyboardInterrupt) as e:
			error_to_raise = e
		
		if error_to_raise:
			run_manager.on_tool_error(error_to_raise)
			raise error_to_raise
		
		output = _format_output(content, artifact, tool_call_id, self.name, status)
		run_manager.on_tool_end(output, color=color, name=self.name, **kwargs)
		return output

	# async def arun()

	#@deprecated(..., alternative="invoke", removal="1.0")
	def __call__(self, tool_input: str, callbacks: Callbacks = None) -> str:
		"""Make tool callable."""
		return self.run(tool_input, callbacks=callbacks)

def _is_tool_call(x: Any) -> bool:
	return isinstance(x, dict) and x.get("type") == "tool_call"

class Tool(BaseTool):
	"""Tool that takes in function or coroutine directly."""

	description: str = ""
	func: Optional[Callable[..., str]]
	"""The function to run when the tool is called."""
	#coroutine: Optional[Callable[..., Awaitable[str]]] = None
	#"""The asynchronous version of the function."""

	# --- Runnable ---
	#async def ainvoke()

	# --- Tool ---
	@property
	def args(self) -> dict:
		"""The tool's input arguments.
		
		Returns:
			The input arguments for the tool.
		"""
		if self.args_schema is not None:
			if isinstance(self.args_schema, dict):
				json_schema = self.args_schema
			else:
				json_schema = self.args_schema.model_json_schema()
			return json_schema["properties"]
		# For backwards compatibility, if the function signature is ambiguous,
		# assume it takes a single string input.
		return {"tool_input": {"type": "string"}}
	
	def _to_args_and_kwargs(
		self, tool_input: Union[str, dict], tool_call_id: Optional[str]
	) -> tuple[tuple, dict]:
		"""Convert tool input to pydantic model."""
		args, kwargs = super()._to_args_and_kwargs(tool_input, tool_call_id)
		# For backwards compatibility. The tool must be run with a single input.
		all_args = list(args) + list(kwargs.values())
		if len(all_args) != 1:
			msg = (
				f"""Too many arguments to single-input tool {self.name}.
				Consider using StructuredTool instead."""
				f" Args: {all_args}"
			)
			raise ToolException(msg)
		return tuple(all_args), {}
	
	def _run(
		self,
		*args: Any,
		config: RunnableConfig,
		run_manager: Optional[CallbackManagerForToolRun] = None,
		**kwargs: Any,
	) -> Any:
		"""Use the tool."""
		if self.func:
			if run_manager and signature(self.func).parameters.get("callbacks"):
				kwargs["callbacks"] = run_manager.get_child()
			if config_param := _get_runnable_config_param(self.func):
				kwargs[config_param] = config
			return self.func(*args, **kwargs)
		msg = "Tool does not support sync invocation."
		raise NotImplementedError(msg)
	
	#async def _arun()

	def __init__(
		self, name: str, func: Optional[Callable], description: str, **kwargs: Any
	) -> None:
		"""Initialize tool."""
		super().__init__(
			name=name, func=func, description=description, **kwargs
		)
	
	@classmethod
	def from_function(
		cls,
		func: Optional[Callable],
		name: str,
		description: str,
		return_direct: bool = False,
		args_schema: Optional[ArgsSchema] = None,
		#coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
		**kwargs: Any,
	) -> Tool:
		"""Initialize tool from a function.

		Args:
			func: The function to create the tool from.
			name: The name of the tool.
			description: The description of the tool.
			return_direct: Whether to return the output directly. Defaults to False.
			args_schema: The schema of the tool's input arguments. Defaults to None.
			kwargs: Additional arguments to pass to the tool.
		
		Returns:
			The tool.

		Raises:
			ValueError: If the function is not provided.
		"""
		if func is None:
			msg = "Function must be provided"
			raise ValueError(msg)
		return cls(
			name=name,
			func=func,
			description=description,
			return_direct=return_direct,
			args_schema=args_schema,
			**kwargs,
		)
Tool.model_rebuild()

class StructuredTool(BaseTool):
	"""Tool that can operate on any number of inputs."""

	description: str = ""
	args_schema: Annotated[ArgsSchema, SkipValidation()] = Field(
		..., description="The tool schema."
	)
	"""The input arguments' schema."""
	func: Optional[Callable[..., Any]] = None
	"""The function to run when the tool is called."""
	#coroutine: Optional[Callable[..., Awaitable[Any]]] = None
	#"""The asynchronous version of the function."""

	# --- Runnable ---
	#async def ainvoke()

	# --- Tool ---
	@property
	def args(self) -> dict:
		"""The tool's input arguments."""
		if isinstance(self.args_schema, dict):
			json_schema = self.args_schema
		else:
			input_schema = self.get_input_schema()
			json_schema = input_schema.model_json_schema()
		return json_schema["properties"]
	
	def _run(
		self,
		*args: Any,
		config: RunnableConfig,
		run_manager: Optional[CallbackManagerForToolRun] = None,
		**kwargs: Any,
	) -> Any:
		"""Use the tool."""
		if self.func:
			if run_manager and signature(self.func).parameters.get("callbacks"):
				kwargs["callbacks"] = run_manager.get_child()
			if config_param := _get_runnable_config_param(self.func):
				kwargs[config_param] = config
			return self.func(*args, **kwargs)
		msg = "StructuredTool does not support sync invocation."
		raise NotImplementedError(msg)
	
	#async def _arun()

	@classmethod
	def from_function(
		cls,
		func: Optional[Callable] = None,
		#coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
		name: Optional[str] = None,
		description: Optional[str] = None,
		return_direct: bool = False,
		args_schema: Optional[ArgsSchema] = None,
		infer_schema: bool = True,
		*,
		response_format: Literal["content", "content_and_artifact"] = "content",
		parse_docstring: bool = False,
		error_on_invalid_docstring: bool = False,
		**kwargs: Any,
	) -> StructuredTool:
		"""Create tool from a given function.

		A classmethod that helps to create a tool from a function.

		Args:
			func: The function from which to create a tool.
			name: The name of the tool. Defaults to the function name.
			description: The description of the tool.
				Defaults to the function docstring.
			return_direct: Whether to return the result directly or as a callback.
				Defaults to False.
			args_schema: The schema of the tool's input arguments. Defaults to None.
			infer_schema: Whether to infer the schema from the function's signature.
				Defaults to True.
			response_format: The tool response format. If "content" then the output of
				the tool is interpreted as the contents of a ToolMessage. If
				"content_and_artifact" then the output is expected to be a two-tuple
				corresponding to the (content, artifact) of a ToolMessage.
				Defaults to "content.
			parse_docstring: if ``infer_schema`` and ``parse_docstring``, will attempt
				to parse parameter descriptions from Google Style function docstrings.
				Defaults fo False.
			error_on_invalid_docstring: if ``parse_docstring`` is provided, configure
				whether to raise ValueError on invalid Google Style docstrings.
				Defaults to False.
			kwargs: Additional arguments to pass to the tool
		
		Returns:
			The tool.
		
		Raises:
			ValueError: If the function is not provided.
		
		Examples:

			.. code-block:: python

				def add(a: int, b: int) -> int:
					\"\"\"Add two numbers\"\"\"
					return a + b
				tool = StructuredTool.from_function(add)
				tool.run(1, 2) # 3
		"""
		if func is not None:
			source_function = func
		else:
			msg = "Function must be provided."
			raise ValueError(msg)
		name = name or source_function.__name__
		if args_schema is None and infer_schema:
			# schema name is appended within function
			args_schema = create_schema_from_function(
				name,
				source_function,
				parse_docstring=parse_docstring,
				error_on_invalid_docstring=error_on_invalid_docstring,
				filter_args=_filter_schema_args(source_function),
			)
		description_ = description
		if description is None and not parse_docstring:
			description_ = source_function.__doc__ or None
		if description_ is None and args_schema:
			if isinstance(args_schema, type) and is_basemodel_subclass(args_schema):
				description_ = args_schema.__doc__ or None
			elif isinstance(args_schema, dict):
				description_ = args_schema.get("description")
			else:
				msg = (
					"Invalid args_schema: expected BasedModel or dict, "
					f"got {args_schema}"
				)
				raise TypeError(msg)
		if description_ is None:
			msg = "Function must have a docstring if description not provided."
			raise ValueError(msg)
		if description is None:
			# Only apply if using the function's docstring
			description_ = textwrap.dedent(description_).strip()
		
		# Description example:
		# search_api(query: str) - Searches the API for the query.
		description_ = f"{description_.strip()}"
		return cls(
			name=name,
			func=func,
			#coroutine=coroutine,
			args_schema=args_schema, # type: ignore[arg-type]
			description=description_,
			return_direct=return_direct,
			response_format=response_format,
			**kwargs,
		)


