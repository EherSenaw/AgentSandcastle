import re

from typing import Any, Type, Optional, Dict, List
from enum import Enum

from pydantic import BaseModel, Field, create_model

ANSWER_DICT_REGEXP = re.compile(r"\{[^\{\}]*\}")
THINK_REGEXP = re.compile(r"<think>.*<\/think>", re.DOTALL)
#JSON_MARKDOWN_REGEXP = re.compile(r"(```)*(json)?(.*)", re.DOTALL)
JSON_MARKDOWN_REGEXP = re.compile(r"```\w*(json)*(?:\w+)?\s*\n(.*?)(?=^```)```", re.DOTALL | re.MULTILINE)
#JSON_STRIP_CHARS = " \n\r\t`"
JSON_STRIP_CHARS = " \n\r\t"

def retrieve_non_think(str_response: str, remove_think_only: bool = False) -> str:
	# NOTE:	Default behavior of this function for `A<think>B</think>C` is returning C.
	# 		Setting `remove_think_only` to True will return `A C`.
	if '<think>' not in str_response:
		return str_response
	retval = ''
	n = len(str_response)
	for candidate in THINK_REGEXP.finditer(str_response):
		s, e = candidate.span()
		if remove_think_only and s > 0:
			left = str_response[:s]
		else:
			left = ''
		if e < n:
			right = str_response[e:]
			if remove_think_only:
				retval = left + str_response[s+7:e-8] + right
			else:
				retval = right
			break
	return retval.strip()

# NOTE: JSON schema -> Pydantic BaseModel converter from stackoverflow
# 		[LINK](https://stackoverflow.com/questions/73841072/dynamically-generating-pydantic-model-from-a-schema-json-file)
def json_schema_to_base_model(schema: dict[str, Any]) -> Type[BaseModel]:
    type_mapping: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])
    model_fields = {}

    def process_field(field_name: str, field_props: dict[str, Any]) -> tuple:
        """Recursively processes a field and returns its type and Field instance."""
        json_type = field_props.get("type", "string")
        enum_values = field_props.get("enum")

        # Handle Enums
        if enum_values:
            enum_name: str = f"{field_name.capitalize()}Enum"
            field_type = Enum(enum_name, {v: v for v in enum_values})
        # Handle Nested Objects
        elif json_type == "object" and "properties" in field_props:
            field_type = json_schema_to_base_model(
                field_props
            )  # Recursively create submodel
        # Handle Arrays with Nested Objects
        elif json_type == "array" and "items" in field_props:
            item_props = field_props["items"]
            if item_props.get("type") == "object":
                item_type: type[BaseModel] = json_schema_to_base_model(item_props)
            else:
                item_type: type = type_mapping.get(item_props.get("type"), Any)
            field_type = list[item_type]
        else:
            field_type = type_mapping.get(json_type, Any)

        # Handle default values and optionality
        default_value = field_props.get("default", ...)
        nullable = field_props.get("nullable", False)
        description = field_props.get("title", "")

        if nullable:
            field_type = Optional[field_type]

        if field_name not in required_fields:
            default_value = field_props.get("default", None)

        return field_type, Field(default_value, description=description)

    # Process each field
    for field_name, field_props in properties.items():
        model_fields[field_name] = process_field(field_name, field_props)

    return create_model(schema.get("title", "DynamicModel"), **model_fields)

def get_parse_chain(
    format_instructions: str,
    query: str,
	tool_list: Optional[str],
	chain_template: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
	# If want to use any tool, should be parsed something like:
    # ```
	# from src.tools import web_search as example_func
	# example_func_json_schema = example_func.args_schema.model_json_schema()
	# tool_list = '- web_search: {example_func.description}'
	# ```
	# Maybe, tool_list can be like:
	# "- {example_func1.name}: {example_func1.description}
	#  - {example_func2.name}: {example_func2.description}"

	# `chain_template` is in the same form with the chat memory. Where
	#		0-th element should be system prompt, having `format_instructions` keyword.
	#			Normally, `format_instructions` should be acquired by parser.get_format_instructions().
	#		1-th element should be user prompt, having `query` keyword.
	# Default chain_template.
	if chain_template is None:
		chain_template = [
			{
				"role": "system",
				"content": "Answer the user query. Wrap the output in `json` tags\n{format_instructions}"\
                    + f"\n**AVAILABLE TOOLS**\n{tool_list}" if tool_list else ""
			},
			{
				"role": "user",
				"content": "{query}"
			}
		]

	parse_chain = list(map(
		lambda x: {
			"role": x["role"],
			"content": x["content"].format(
				format_instructions=format_instructions
			) if x["role"] == "system" else x["content"].format(
				query=query
			)
		}, chain_template
	))

	return parse_chain