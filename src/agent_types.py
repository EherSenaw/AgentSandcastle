from typing import Optional, Dict, Any, Union, List
from pydantic import BaseModel, Field

class ToolCallAction(BaseModel):
    tool_name: str = Field(..., description="The name of the tool to be called.")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="The arguments for the tool.")

class FinalAnswerAction(BaseModel):
    answer: str = Field(..., description="The final answer to the user's query.")

class Action(BaseModel):
    action_type: Union[ToolCallAction, FinalAnswerAction]
    rationale: Optional[str] = Field(None, description="The reasoning behind this action.") # Or thought leading to action

class Observation(BaseModel):
    observation_type: str = Field(..., description="Type of observation, e.g., 'tool_result', 'tool_error', 'llm_response'")
    content: str = Field(..., description="The content of the observation.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Any metadata associated with the observation, e.g., tool name if it was a tool result.")

# This replaces the Pydantic model generated from `process_request.args_schema`
class LLMStepOutput(BaseModel):
    thought: Optional[str] = Field(None, description="The thought process of the LLM.")
    action: Action = Field(..., description="The action to be taken by the agent.")
    is_final: bool = Field(False, description="Whether this step provides the final answer and the loop should stop.")

class LLMToolInputParsingOutput(BaseModel):
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="The parsed arguments for the tool.")
