import re

from typing import Any,Type,Optional
from enum import Enum

from pydantic import BaseModel, Field, create_model

THINK_REGEXP = re.compile(r"<think>.*<\/think>", re.DOTALL)
def retrieve_non_think(str_response: str) -> str:
	if '<think>' not in str_response:
		return str_response
	
	n = len(str_response)
	for candidate in THINK_REGEXP.finditer(str_response):
		s, e = candidate.span()
		# NOTE: Leave this 'left' extraction code sniffet for possible future use.
		'''
		if s > 0:
			left = str_response[:s]
		else:
			left = ''
		'''
		
		if e < n:
			right = str_response[e:]
			break
	return right.strip()

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