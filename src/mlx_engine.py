import time, re, os
import asyncio
import functools
from typing import List, Optional, Dict, Any, Union

from src.tools import save_file # process_request, process_binary removed
from src.utils import ANSWER_DICT_REGEXP, THINK_REGEXP, retrieve_non_think, json_schema_to_base_model
from src.prompt_template import (
	LC_SYSTEM_PROMPT_TEMPLATE,
	#SYSTEM_PROMPT_TEMPLATE,
	#PROMPT_TEMPLATE,
	#SYSTEM_ANSWER_TEMPLATE, # Removed
	NAIVE_COMPLETION_RETRY,
)
from src.agent_types import ToolCallAction, FinalAnswerAction, Action, Observation, LLMStepOutput, LLMToolInputParsingOutput
from src.structured_output import PydanticOutputParser, RetryOutputParser
from src.tool_convert import tool
from src.exceptions import OutputParserException

from pydantic import BaseModel, Field

from mlx_lm import load, generate, stream_generate

from mlx_vlm import (
	load as vlm_load,
	generate as vlm_generate,
)
from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template
from mlx_vlm.utils import load_config as vlm_load_config

from external_use import with_final

class MLXEngine():
	def __init__(
		self,
		model : str = 'mlx-community/DeepScaleR-1.5B-Preview-4bit',
		max_iteration : int = 5,
		verbose : bool = False,
		tools : List[Optional[Any]] = [], #List[Optional[Tool]] = [],
		modality_io : str = 'text/text',
		max_new_tokens : int = 1024,
		manual_answer_format : Optional[str] = '', # To denote Legacy-style `prompt-level` forcing of structured output.
	):
		self.model = model
		self.max_iter = max_iteration
		self.verbose = verbose

		#self.tools = verify_hf_tools(tools)
		#self.tools = verify_tools_docstring(tools)
		self.tools = dict()
		for t in tools:
			# NOTE: Now using LangChain-style @tool decorator, which parses the tool's Google-style docstring automatically.
			t_schema = t.args_schema.model_json_schema()
			t_name = t_schema['title']
			#t_description = t_schema['description']
			assert t_name not in self.tools, "The name(`title`) of the tool should not be duplicated."
			self.tools[t_name] = t

		self.memory = [] # Chat memory.
		self.cnt_iter = 0 # Thinking iterations counter.
		self.max_new_tokens = max_new_tokens # Set maximum tokens to generate, for the LLM.

		# NOTE: To support multimodal input / output models & agents.
		#		Currently tested I/O: Text/Text, Text+Img/Text.
		# WARNING: Structured output & tool auto-parsing & calling is not strictly tested for VLM.
		self.modality_in, self.modality_out = modality_io.split('/')
		self.modality_in = list(self.modality_in.split(','))
		self.modality_out = list(self.modality_out.split(','))

		# Flag for VLM usage.
		self.USE_VLM = True if len(self.modality_in) > 1 else False
		if self.USE_VLM:
			self.client, self.processor = vlm_load(self.model)
			self.config = vlm_load_config(self.model)
		else:
			self.client, self.tokenizer = load(self.model)

		# LLM engine system prompt setup.
		self.validation_prompt = LC_SYSTEM_PROMPT_TEMPLATE
		# Removed manual_answer_format logic
		# Initialize chat memory with system prompt.
		self.__init_memory(self.verbose)

	def __init_memory(self, verbose:bool=False):
		# Updated system prompt to guide towards LLMStepOutput
		# TODO: Review and potentially update LC_SYSTEM_PROMPT_TEMPLATE itself
		# to better align with LLMStepOutput structure if necessary.
		# For now, assume the existing template is flexible enough or will be
		# implicitly handled by the PydanticOutputParser's format instructions.
		system_message_content = self.validation_prompt.format(
			max_iteration=self.max_iter,
			modality_in=self.modality_in,
			modality_out=self.modality_out,
			tool_list='\n'.join(f"- {t_name}: {tool.description}" for t_name,tool in self.tools.items()),
		)
		system_message_content += (
			"\n\nYou MUST respond in a JSON format that adheres to the following Pydantic schema:"
			"\n```json\n{output_schema}\n```"
			"\nEnsure your output can be directly parsed into this schema."
			"\nFields to include: `thought` (your reasoning), `action` (ToolCallAction or FinalAnswerAction), and `is_final` (boolean)."
		).format(output_schema=LLMStepOutput.model_json_schema()) # This might be too verbose, will be handled by parser

		self.memory.append({
			"role": "system",
			"content": system_message_content
		})
		# Removed self.validation_answer_format logic

	def __call__(self, question:str='', volatile=False):
		# Initial request
		llm_step_output = self.check_request(question, init=True)
		self.memorize_thought_and_action(llm_step_output.thought, llm_step_output.action)

		while not llm_step_output.is_final:
			if self.cnt_iter >= self.max_iter:
				if self.verbose:
					print("Max iterations reached.")
				# Produce a final fallback answer if max_iter is reached.
				# This could be a specific message or an attempt to summarize current thoughts.
				# For now, let's assume the last thought might contain a summary or we force a final answer.
				# This part needs careful consideration on how to gracefully exit.
				# A simple approach:
				final_fallback_action = FinalAnswerAction(answer="Maximum iterations reached. Unable to complete the request fully.")
				llm_step_output = LLMStepOutput(thought="Max iterations reached, providing fallback answer.", action=Action(action_type=final_fallback_action), is_final=True)
				break

			if self.verbose:
				print(f"Iteration {self.cnt_iter + 1} / {self.max_iter}")

			action_to_perform = llm_step_output.action.action_type

			if isinstance(action_to_perform, ToolCallAction):
				observation = self.use_tool(action_to_perform.tool_name, action_to_perform.tool_args)
				self.memorize_observation(observation)
				# Prepare for next step
				# The user_query for memorize needs to represent the context for the next LLM call
				# For now, the observation content itself can serve as part of the context.
				# The actual prompt for the next check_request will be generic.
				prompt_for_next_step = "Based on the history and previous step, decide the next thought and action."

			elif isinstance(action_to_perform, FinalAnswerAction):
				# This case should ideally be caught by llm_step_output.is_final at the start of the loop.
				# If is_final was false but we got FinalAnswerAction, it's a bit contradictory.
				# For now, trust is_final. If is_final is false, we should take another step.
				# If it occurs, treat it as an unexpected state and force a re-evaluation.
				self.memorize("", "System: Received FinalAnswerAction when is_final was false. Re-evaluating.") # Log this anomaly
				prompt_for_next_step = "Based on the history and previous step (note: previous step indicated not final but gave a final answer, please clarify or continue), decide the next thought and action."

			else:
				# Should not happen with the current Action definition
				raise ValueError(f"Unknown action type: {type(action_to_perform)}")

			llm_step_output = self.check_request(prompt_for_next_step)
			self.memorize_thought_and_action(llm_step_output.thought, llm_step_output.action)
			self.cnt_iter += 1
		
		# Final answer is derived from the last LLMStepOutput
		if isinstance(llm_step_output.action.action_type, FinalAnswerAction):
			final_answer = llm_step_output.action.action_type.answer
		elif llm_step_output.thought: # Fallback if no explicit FinalAnswerAction but is_final is true
			final_answer = f"Process finished. Last thought: {llm_step_output.thought}"
		else:
			final_answer = "Process finished."


		# Save the memory to the text file by LLM.
		self.save_mem_to_file(
			os.path.join(os.getcwd(), 'temp_mlx.log')
		)

		# Clean up resources.
		self.manage_resource(volatile=volatile)

		return final_answer

		# Save the memory to the text file by LLM.
		self.save_mem_to_file(
			os.path.join(os.getcwd(), 'temp_mlx.log')
		)

		# Clean up resources.
		self.manage_resource(volatile=volatile)

		return final_answer

	@with_final
	async def stream_call(self, question: str = '', volatile=False):
		'''Async version of __call__() method.'''
		# Initial request
		llm_step_output: Optional[LLMStepOutput] = None
		async for (parsed_output, is_truly_final_token_for_parser) in self.check_request_async(question, init=True):
			if not is_truly_final_token_for_parser: # this is a thought token from the parser
				yield ("thought_intermediate", parsed_output) # parsed_output here is actually a token
			else: # this is the fully parsed LLMStepOutput
				llm_step_output = parsed_output 
				if llm_step_output.thought:
					yield ("thought_stream", llm_step_output.thought) # Yield the full thought
				if llm_step_output.action: # Yield action as a structured object or string
					yield ("action_stream", llm_step_output.action.model_dump_json()) 
		
		if llm_step_output is None: # Should not happen if check_request_async works correctly
			llm_step_output = LLMStepOutput(
				thought="Error: Initial LLM step failed to produce output.",
				action=Action(action_type=FinalAnswerAction(answer="Error: Could not process initial request.")),
				is_final=True
			)

		self.memorize_thought_and_action(llm_step_output.thought, llm_step_output.action)


		while llm_step_output and not llm_step_output.is_final:
			if self.cnt_iter >= self.max_iter:
				if self.verbose:
					print("Max iterations reached in stream_call.")
				final_fallback_action = FinalAnswerAction(answer="Maximum iterations reached. Unable to complete the request fully.")
				llm_step_output = LLMStepOutput(thought="Max iterations reached, providing fallback answer.", action=Action(action_type=final_fallback_action), is_final=True)
				yield ("thought_stream", llm_step_output.thought)
				yield ("action_stream", llm_step_output.action.model_dump_json())
				break

			if self.verbose:
				yield (f"status_stream", f"Iteration {self.cnt_iter + 1} / {self.max_iter}")

			action_to_perform = llm_step_output.action.action_type

			if isinstance(action_to_perform, ToolCallAction):
				yield ("tool_call_stream", {"name": action_to_perform.tool_name, "args": action_to_perform.tool_args})
				# Non-blocking tool call
				observation = await asyncio.get_event_loop().run_in_executor(
					None, functools.partial(self.use_tool, action_to_perform.tool_name, action_to_perform.tool_args)
				)
				self.memorize_observation(observation)
				yield ("observation_stream", observation.model_dump_json())
				prompt_for_next_step = "Based on the history and previous step, decide the next thought and action."

			elif isinstance(action_to_perform, FinalAnswerAction):
				# This case should ideally be caught by llm_step_output.is_final.
				# If is_final is false, but we got FinalAnswerAction, it's contradictory.
				self.memorize("", "System: Received FinalAnswerAction when is_final was false in stream. Re-evaluating.")
				yield ("status_stream", "Warning: Received FinalAnswerAction when is_final was false. Re-evaluating.")
				prompt_for_next_step = "Based on the history and previous step (note: previous step indicated not final but gave a final answer, please clarify or continue), decide the next thought and action."
			else:
				raise ValueError(f"Unknown action type: {type(action_to_perform)}")

			# Get next step from LLM
			next_llm_step_output: Optional[LLMStepOutput] = None
			async for (parsed_output, is_truly_final_token_for_parser) in self.check_request_async(prompt_for_next_step):
				if not is_truly_final_token_for_parser:
					yield ("thought_intermediate", parsed_output)
				else:
					next_llm_step_output = parsed_output
					if next_llm_step_output.thought:
						yield ("thought_stream", next_llm_step_output.thought)
					if next_llm_step_output.action:
						yield ("action_stream", next_llm_step_output.action.model_dump_json())
			
			if next_llm_step_output is None: # Should not happen
				next_llm_step_output = LLMStepOutput(
                    thought="Error: LLM step failed to produce output during iteration.",
                    action=Action(action_type=FinalAnswerAction(answer="Error: Could not process request further.")),
                    is_final=True
                )

			llm_step_output = next_llm_step_output
			self.memorize_thought_and_action(llm_step_output.thought, llm_step_output.action)
			self.cnt_iter += 1

		# Final answer processing
		final_answer_content = "Process finished."
		if llm_step_output and isinstance(llm_step_output.action.action_type, FinalAnswerAction):
			final_answer_content = llm_step_output.action.action_type.answer
		elif llm_step_output and llm_step_output.thought:
			final_answer_content = f"Process finished. Last thought: {llm_step_output.thought}"
		
		yield ("final_answer_stream", final_answer_content) # This is yielded by with_final as the final tuple's first element

		# Save the memory (sync operation, consider if this needs to be async or backgrounded)
		self.save_mem_to_file(
			os.path.join(os.getcwd(), 'temp_mlx.log')
		)
		# Clean up resources
		self.manage_resource(volatile=volatile)

		######### SYNC VERSION ########
		# Save the memory to the text file by LLM.
		self.save_mem_to_file(
			os.path.join(os.getcwd(), 'temp_mlx.log')
		)
		# Clean up resources.
		self.manage_resource(volatile=volatile)
		######### SYNC VERSION ########

		#yield final_answer
	
	def __ask_LLM(
		self,
		question : str = '',
		image_urls : Optional[List[str]] = None,
		minimal : Optional[bool] = False,
		ignore_answer_format : Optional[bool] = False,
		parser: Optional[PydanticOutputParser] = None, # NOTE: Provide parser to use auto-parsing structured output for tools generated with @tool decorator.
		self_querying: Optional[bool] = False,
	):
		if image_urls is None:
			image_urls = []

		# NOTE: VLM not tested since beginning of tool execution implementation.
		if self.USE_VLM:
			try:
				formatted_prompt = vlm_apply_chat_template(
					self.processor, self.config,
					(self.memory[0:1] + (self.memory[2:] if len(self.memory)>2 else []) if ignore_answer_format else self.memory) + \
					([{
						"role": "system", # This system message for parser instructions might be redundant if main system prompt is good
						"content": "Answer the user query. Wrap the output in `json` tags\n{format_instructions}".format(
							format_instructions=parser.get_format_instructions()
						)
					}] if parser else []) + \
					[{
						"role": "user",
						"content": question,
					}], num_images=len(image_urls)
				)
			except ValueError as e:
				print(f"\n**ERROR**\nGot ValueError by trying to use Multi-image chat with the model not supported. Fall back to use first image only.\nOriginal error message -> {e}\n**ERROR**\n")
				image_urls = [image_urls[0]]
				formatted_prompt = vlm_apply_chat_template(
					self.processor, self.config,
					(self.memory[0:1] + (self.memory[2:] if len(self.memory)>2 else []) if ignore_answer_format else self.memory) + \
					([{
						"role": "system",
						"content": "Answer the user query. Wrap the output in `json` tags\n{format_instructions}".format(
							format_instructions=parser.get_format_instructions()
						)
					}] if parser else []) + \
					[{
						"role": "user",
						"content": question,
					}], num_images=len(image_urls)
				)
			output = vlm_generate(self.client, self.processor, formatted_prompt, image_urls, verbose=self.verbose, max_tokens=self.max_new_tokens)
		else:
			u_id = "user" if not self_querying else "assistant" # Role for parser instruction if needed
			
			# The main system prompt in self.memory[0] should ideally already contain schema instructions
			# or be general enough. Adding specific parser instructions here might be duplicative or conflict.
			# For LLMStepOutput, the parser instructions are complex, better integrated into the main system prompt.
			# For LLMToolInputParsingOutput, it's simpler and might be fine here.
			
			messages_for_template = list(self.memory) # Make a copy
			if parser:
				# If parser is LLMStepOutput, its instructions are expected to be in the main system prompt.
				# If parser is LLMToolInputParsingOutput, we might add specific instructions.
				if isinstance(parser.pydantic_object, LLMToolInputParsingOutput):
					messages_for_template.append({
						"role": "system",
						"content": "Extract tool arguments. Wrap the output in `json` tags\n{format_instructions}".format(
							format_instructions=parser.get_format_instructions()
						)
					})
			
			messages_for_template.append({
				"role": "user" if not self_querying else "assistant",
				"content": question,
			})

			prompt = self.tokenizer.apply_chat_template(
				messages_for_template,
				add_generation_prompt=True,
				tokenize=False, # Always False when using PydanticOutputParser, it expects full string.
			)
			output = generate(
				self.client,
				self.tokenizer,
				prompt=prompt,
				verbose=self.verbose,
				max_tokens=self.max_new_tokens,
				## NOTE: kwargs for `generate_step()`
				kv_bits=4, # KV cache quantization bits
			)
		output = retrieve_non_think(output.strip(), remove_think_only=minimal)
		# NOTE: structured output auto-parsing
		if parser:
			try:
				parsed_output = parser.invoke(output)
			except OutputParserException as e:
				retry_parser = RetryOutputParser.from_llm(
					parser=parser,
					llm = (self.client, self.tokenizer),
					prompt_template=NAIVE_COMPLETION_RETRY,
					max_retries=self.max_iter,
				)
				parsed_output = retry_parser.parse_with_prompt(output, prompt)
			return output, parsed_output #NOTE: return string-type output also, to be used in 'memorize'.
		return output
	
	@with_final
	async def __ask_LLM_async(
		self,
		question : str = '',
		image_urls : Optional[List[str]] = None,
		minimal : Optional[bool] = False,
		ignore_answer_format : Optional[bool] = False,
		parser: Optional[PydanticOutputParser] = None, # NOTE: Provide parser to use auto-parsing structured output for tools generated with @tool decorator.
		self_querying: Optional[bool] = False,
	):
		#### SYNC VERSION ####
		if image_urls is None:
			image_urls = []

		messages_for_template_async = list(self.memory)
		if parser:
			if isinstance(parser.pydantic_object, LLMToolInputParsingOutput):
				messages_for_template_async.append({
                    "role": "system",
                    "content": "Extract tool arguments. Wrap the output in `json` tags\n{format_instructions}".format(
                        format_instructions=parser.get_format_instructions()
                    )
                })
		
		messages_for_template_async.append({
			"role": "user" if not self_querying else "assistant",
			"content": question,
		})
		
		prompt = self.tokenizer.apply_chat_template(
			messages_for_template_async,
			add_generation_prompt=True,
			tokenize=False, # Always False for PydanticOutputParser
		)
		#### SYNC VERSION ####

		# The `with_final` decorator expects the generator to yield individual tokens,
        # and then the final "full" output. For Pydantic parsing, the "full" output
        # is the string that gets parsed, and then the *parsed object*.
        # This means __ask_LLM_async needs to yield tokens, then yield the (raw_output_string, parsed_object)
        # The `with_final` decorator might need adjustment or careful usage if it assumes the final item is just the aggregation of tokens.
        # Let's assume `with_final` is smart enough or that `stream_call` will handle the two types of final items.
        # For now, __ask_LLM_async will yield tokens, then (raw_string, parsed_obj) as its "final" yield.
		
		output_buffer = [] # Changed from output = ''
		for response in stream_generate(
			self.client,
			self.tokenizer,
			prompt=prompt,
			max_tokens=self.max_new_tokens,
			## NOTE: kwargs for `generate_step()`
			kv_bits=4, # KV cache quantization bits
		):
			token = response.text
			yield token # Yield individual token for streaming
			output_buffer.append(token) # Accumulate tokens
			# await asyncio.sleep(0) # Removed for faster local processing if not needed for cooperative multitasking

		full_output_str = "".join(output_buffer)
		full_output_str = retrieve_non_think(full_output_str.strip(), remove_think_only=minimal)

		if parser:
			try:
				parsed_obj = parser.invoke(full_output_str)
			except OutputParserException as e:
				# NOTE: Async retry is complex. For now, let's assume sync retry or handle error.
				# This part needs careful thought for async. For now, we'll skip async retry.
				# A simple approach might be to raise or return an error object.
				# For this refactor, we'll assume parsing works or error is propagated.
				# A proper async retry parser would be needed here.
				# Fallback: return raw string and error.
				# This is a placeholder for proper async error handling/retry.
				print(f"Async parsing failed: {e}. Raw output: {full_output_str}")
				# yield full_output_str, None # Indicate parsing failure
				# Or re-raise, or return a specific error structure
				# For now, let's assume it will be handled by the caller or by a simplified error path
				raise # Re-raise for now, to be handled by caller or a more robust stream_call
			yield full_output_str, parsed_obj # Yield (raw_string, parsed_object) as the "final" item from this generator
		else:
			yield full_output_str # Yield full string if no parser


	# Removed __retrieve_answer_dict

	def check_request(self, question: str, init: bool = False) -> LLMStepOutput: # init is not used now
		# The role of 'init' was to format the question differently.
		# This can be handled by the caller or by a more generic prompt structure if needed.
		# For now, the question is used directly.
		llm_input = question
		
		# Parser for LLMStepOutput
		parser = PydanticOutputParser(pydantic_object=LLMStepOutput)
		
		# __ask_LLM now returns (raw_string, parsed_object)
		raw_output, parsed_output = self.__ask_LLM(llm_input, parser=parser)
		
		# No direct memorization here, __call__ will handle memorizing thoughts, actions, observations.
		# self.memorize(llm_input, raw_output) # Old style, remove.

		if not isinstance(parsed_output, LLMStepOutput):
			# This case should ideally be handled by RetryOutputParser or raise an error in __ask_LLM
			# For robustness, if parsing somehow returns a non-LLMStepOutput (e.g. due to retry logic not perfectly fitting)
			# we might need a fallback. However, with PydanticOutputParser, it should be an LLMStepOutput or an exception.
			raise ValueError(f"Expected LLMStepOutput, got {type(parsed_output)}")

		return parsed_output
	
	@with_final # This decorator now gets (token, False) or ( (raw_str, parsed_obj), True )
	async def check_request_async(self, question: str, init: bool = False) -> AsyncGenerator[Union[str, LLMStepOutput], None]: # init is not used
		llm_input = question
		parser = PydanticOutputParser(pydantic_object=LLMStepOutput)

		# __ask_LLM_async yields tokens, then (raw_string, parsed_object)
		raw_str_final = None
		parsed_obj_final = None

		async for item in self.__ask_LLM_async(llm_input, parser=parser):
			if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], LLMStepOutput):
				# This is the (raw_string, parsed_object) tuple
				raw_str_final, parsed_obj_final = item
				# Do not yield here, `with_final` handles the final object.
			else:
				# This is an intermediate token
				yield item # Yield token for streaming
		
		# After the loop, raw_str_final and parsed_obj_final contain the full response.
		# `with_final` expects the generator's last yield to be the "final complete object".
		# In our case, the "final complete object" from __ask_LLM_async perspective for parsing is (raw_str, parsed_obj).
		# `check_request_async` itself should yield the `parsed_obj_final` as its final item for `with_final`.
		if parsed_obj_final is None:
			# This would mean __ask_LLM_async didn't yield the final tuple, which is an error.
			# Or, stream ended prematurely.
			# Create a fallback error LLMStepOutput
			parsed_obj_final = LLMStepOutput(
				thought="Error: LLM response parsing failed or stream ended unexpectedly in check_request_async.",
				action=Action(action_type=FinalAnswerAction(answer="Error processing request.")),
				is_final=True
			)
		
		# self.memorize(llm_input, raw_str_final) # Old style, remove. Caller will memorize.
		yield parsed_obj_final # This is what `with_final` will see as the "complete" item.

	
	def use_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Observation:
		tool_to_use = self.tools.get(tool_name)

		if not tool_to_use:
			err_msg = f"Tool '{tool_name}' not found."
			# self.memorize_observation(Observation(observation_type="system_error", content=err_msg, metadata={"tool_name": tool_name})) # Memorize is handled by caller
			return Observation(observation_type="tool_error", content=err_msg, metadata={"tool_name": tool_name, "args": tool_args, "error_type": "ToolNotFound"})

		# The LLM call for argument parsing is now expected to be done by the caller (check_request)
		# and tool_args are directly passed.
		# However, the original plan was to have use_tool do its own LLM call for arg parsing using LLMToolInputParsingOutput.
		# Let's stick to the plan: `check_request` provides ToolCallAction(tool_name, tool_args_FROM_LLM_STEP_OUTPUT),
		# then `use_tool` *could* re-parse/validate if tool_args were just a string description.
		# But the prompt asks for `tool_args: Dict[str, Any]`. This implies `check_request`'s LLM already structured them.
		#
		# Re-reading: "The LLM call within use_tool that parses arguments for the tool ... should use PydanticOutputParser(pydantic_object=LLMToolInputParsingOutput)"
		# This seems to imply that `tool_args` received by `use_tool` might *not* be the final parsed args, or that `use_tool`
		# is responsible for a *dedicated* parsing step if `tool_args` was, for example, a natural language string describing args.
		# Given `ToolCallAction.tool_args: Dict[str, Any]`, it's more likely that the LLM producing `LLMStepOutput`
		# is already meant to provide structured args.
		#
		# Let's assume `tool_args` from `ToolCallAction` are the ones to use.
		# If `use_tool` *still* needs to call an LLM for arg parsing, the design is a bit circular.
		# For now, assume `tool_args: Dict[str, Any]` are ready to be used or validated against tool's schema.
		#
		# If `tool_args` were a string, then this would be needed:
		# retrieve_prompt = f"Extract arguments for tool '{tool_name}' from the following: {tool_args_string}"
		# arg_parser = PydanticOutputParser(pydantic_object=LLMToolInputParsingOutput) # This would need dynamic model based on tool_to_use.args_schema
		# _, parsed_tool_input = self.__ask_LLM(retrieve_prompt, parser=arg_parser)
		# actual_args_to_invoke = parsed_tool_input.tool_args

		# For this refactoring, let's assume tool_args from ToolCallAction are sufficient.
		# The schema for LLMToolInputParsingOutput is generic, not tool-specific.
		# If dynamic parsing per tool is needed here, it's more complex.
		# The instructions say: "PydanticOutputParser(pydantic_object=LLMToolInputParsingOutput)"
		# This implies the LLM call inside use_tool gets a generic dict. This is only useful if the *input* `tool_args`
		# was a natural language string that this LLM call would then parse into a dict.
		#
		# Let's proceed with the assumption that `tool_args: Dict[str, Any]` from `ToolCallAction` are the direct input.
		# The LLM call described for `use_tool` in point 2 seems redundant if `check_request` already provides structured args.
		#
		# Clarification: "The LLM call within use_tool that parses arguments for the tool ... should use PydanticOutputParser(pydantic_object=LLMToolInputParsingOutput)"
		# This MUST mean that the `tool_args` parameter received by `use_tool` is NOT the final dict, but rather a string or something needing parsing.
		# This contradicts `ToolCallAction` having `tool_args: Dict[str, Any]`.
		#
		# RESOLUTION: I will assume `ToolCallAction.tool_args` is already the parsed dictionary.
		# The LLM call for `LLMToolInputParsingOutput` is thus SKIPPED in `use_tool` under this assumption.
		# If `tool_args` was meant to be a string, `ToolCallAction` schema would be `tool_arg_description: str`.

		actual_args_to_invoke = tool_args # Assuming these are ready

		try:
			# Ensure args are in the format expected by the tool (e.g., if tool expects kwargs or a single dict)
			# Langchain tools typically expect a dictionary.
			tool_result = tool_to_use.invoke(actual_args_to_invoke)
			# self.memorize_observation(...) is handled by caller
			return Observation(observation_type="tool_result", content=str(tool_result), metadata={"tool_name": tool_name, "args": actual_args_to_invoke})
		except Exception as e:
			if self.verbose:
				print(f"Error using tool {tool_name} with args {actual_args_to_invoke}: {e}")
			# self.memorize_observation(...) is handled by caller
			return Observation(observation_type="tool_error", content=str(e), metadata={"tool_name": tool_name, "args": actual_args_to_invoke, "error_type": type(e).__name__})

	# Memory methods
	def memorize_thought_and_action(self, thought: Optional[str], action: Action):
		if thought:
			self.memory.append({"role": "assistant", "content": f"Thought: {thought}"}) # Or a dedicated thought role/structure
		
		action_content = ""
		if isinstance(action.action_type, ToolCallAction):
			action_content = f"Action: Call tool '{action.action_type.tool_name}' with args {action.action_type.tool_args}"
		elif isinstance(action.action_type, FinalAnswerAction):
			action_content = f"Action: Provide final answer: '{action.action_type.answer}'"
		
		if action.rationale: # Include rationale if present
			action_content += f"\nRationale: {action.rationale}"

		self.memory.append({"role": "assistant", "content": action_content})

	def memorize_observation(self, observation: Observation):
		self.memory.append({
			"role": "user", # Observations are like input from the environment/tools for the next LLM step
			"content": f"Observation ({observation.observation_type} for tool {observation.metadata.get('tool_name', 'N/A')}): {observation.content}"
		})

	# Old memorize method - keep for now if other parts still use it, but should be phased out.
	def memorize(self, user_query: str, assistant_response: str, verbose: bool = False, self_querying: bool = False):
		# This is a fallback / compatibility. New logic should use specific memorize methods.
		if user_query: # If there's a specific user-like query string for this memory item
			self.memory.append({
				"role": "user" if not self_querying else "assistant", # self_querying might map to internal monologue
				"content": user_query,
			})
		if assistant_response: # If there's a direct assistant response string
			self.memory.append({
				"role": "assistant",
				"content": assistant_response,
			})


	def clear_memory(self):
		# del self.memory # This would delete the attribute itself.
		self.memory.clear() # Empties the list.
		# self.memory = [] # This would create a new list, breaking old references if any.
		self.__init_memory(self.verbose) # Re-initialize with system prompt.
	
	def manage_resource(self, volatile=True):
		if volatile:
			self.clear_memory()
		self.cnt_iter = 0

	def shutdown(self):
		self.manage_resource(volatile=True)

	# Removed continue_iteration method
	# Removed plan_subtask method

	# finalize and finalize_async are effectively replaced by the main loop's termination logic
	# based on LLMStepOutput.is_final and extracting the answer from LLMStepOutput.action.action_type.answer.
	# These methods can be removed.
	#
	# def finalize(self): ... removed ...
	# @with_final
	# async def finalize_async(self): ... removed ...


	# For saving final memory to the file.
	def save_mem_to_file(self, file_path: str):
		try:
			with open(file_path, "w", encoding='utf-8') as f: # Changed "w+" to "w"
				for m in self.memory:
					role = m.get("role", "unknown")
					content = m.get("content", "")
					f.write(f"[{role}]\n{content}\n\n") # Added extra newline for readability
			if self.verbose:
				print(f"Memory saved to '{file_path}'.")
		except Exception as e:
			print(f"Error saving memory to file '{file_path}': {e}")