import time, re, os
import asyncio
import functools
from typing import List, Optional, Dict, Any, Union, AsyncGenerator

from src.tools import save_file # process_request, process_binary removed
# ANSWER_DICT_REGEXP, THINK_REGEXP removed, retrieve_non_think kept for now
from src.utils import retrieve_non_think, json_schema_to_base_model 
from src.prompt_template import (
	LC_SYSTEM_PROMPT_TEMPLATE,
	#SYSTEM_PROMPT_TEMPLATE, # Removed
	#PROMPT_TEMPLATE, # Removed
	#SYSTEM_ANSWER_TEMPLATE, # Removed
	NAIVE_COMPLETION_RETRY,
)
from src.agent_types import ToolCallAction, FinalAnswerAction, Action, Observation, LLMStepOutput, LLMToolInputParsingOutput
from src.structured_output import PydanticOutputParser, RetryOutputParser
from src.tool_convert import tool
from src.exceptions import OutputParserException

from pydantic import BaseModel, Field

import torch
from transformers import AutoImageProcessor # Added for VLM
from PIL import Image # Added for VLM
import requests # Added for VLM
from io import BytesIO # Added for VLM

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from external_use import with_final

class vLLMEngine():
	def __init__(
		self,
		model : str = 'llava-hf/llava-1.5-7b-hf', # Default changed to a LLaVA model for VLM testing
		max_iteration : int = 5,
		verbose : bool = False,
		tools : List[Optional[Any]] = [],
		modality_io : str = 'text/text', # Default will be overridden if image is present
		max_new_tokens : int = 1024,
	):
		self.model_name = model
		self.max_iter = max_iteration
		self.verbose = verbose

		self.tools = dict()
		for t in tools:
			t_schema = t.args_schema.model_json_schema()
			t_name = t_schema['title']
			assert t_name not in self.tools, f"Tool name '{t_name}' duplicated."
			self.tools[t_name] = t

		self.memory = []
		self.cnt_iter = 0
		self.max_new_tokens = max_new_tokens

		# Determine USE_VLM based on model name or modality_io, assuming LLaVA implies VLM
		# A more robust way might be to inspect model config, but this is experimental
		if "llava" in self.model_name.lower() or "image" in modality_io:
			self.USE_VLM = True
			# Update modality_in if it was default but model is VLM
			if modality_io == 'text/text' and "llava" in self.model_name.lower():
				self.modality_in = ["text", "image"]
				self.modality_out = ["text"]
			else:
				self.modality_in, self.modality_out = modality_io.split('/')
				self.modality_in = list(self.modality_in.split(','))
				self.modality_out = list(self.modality_out.split(','))
		else:
			self.USE_VLM = False
			self.modality_in, self.modality_out = modality_io.split('/')
			self.modality_in = list(self.modality_in.split(','))
			self.modality_out = list(self.modality_out.split(','))

		# Initialize vLLM client
		# For LLaVA, vLLM might handle multimodal aspects internally based on model type if `trust_remote_code=True`.
		# Specific parameters like `image_input_type`, `image_token_id` might be needed for some models/versions of vLLM.
		# This is an area that might require adjustment based on vLLM's LLaVA support specifics.
		self.client = LLM(
			model=self.model_name,
			tokenizer=self.model_name, 
			gpu_memory_utilization=0.9,
			trust_remote_code=True, # Crucial for many HuggingFace models, including VLMs
			max_model_len=max_new_tokens * 4, # Increased max_model_len
			# enable_lora=True, # If using LoRA adapters with LLaVA
			# dtype="half", # Or "bfloat16" if supported
		)
		
		self.image_processor = None
		if self.USE_VLM:
			try:
				self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)
				if self.verbose:
					print(f"Image processor loaded for VLM: {self.model_name}")
			except Exception as e:
				print(f"Warning: Failed to load image processor for {self.model_name}. VLM capabilities might be limited. Error: {e}")
				self.USE_VLM = False # Fallback to text-only if processor fails

		self.sampling_params = SamplingParams(
			temperature=0.7,
			top_p=0.95,
			max_tokens=self.max_new_tokens
		)
		self.sampling_params_stream = SamplingParams(
			temperature=0.7,
			top_p=0.95,
			max_tokens=self.max_new_tokens,
			stream=True
		)

		self.validation_prompt = LC_SYSTEM_PROMPT_TEMPLATE
		self.__init_memory(self.verbose)

	def __init_memory(self, verbose:bool=False): # Aligned with mlx_engine
		system_message_content = self.validation_prompt.format(
			max_iteration=self.max_iter,
			modality_in=self.modality_in, # Retained for info, though VLM not implemented
			modality_out=self.modality_out, # Retained for info
			tool_list='\n'.join(f"- {t_name}: {tool.description}" for t_name, tool in self.tools.items()),
		)
		# Guidance for LLMStepOutput JSON structure
		system_message_content += (
			"\n\nYou MUST respond in a JSON format that adheres to the following Pydantic schema:"
			"\n```json\n{output_schema}\n```"
			"\nEnsure your output can be directly parsed into this schema."
			"\nFields to include: `thought` (your reasoning), `action` (ToolCallAction or FinalAnswerAction), and `is_final` (boolean)."
		).format(output_schema=LLMStepOutput.model_json_schema())

		self.memory.append({
			"role": "system",
			"content": system_message_content
		})

	def __call__(self, question:str='', volatile=False): # Aligned with mlx_engine
		llm_step_output = self.check_request(question, init=True) # init flag kept for consistency if used internally by check_request
		self.memorize_thought_and_action(llm_step_output.thought, llm_step_output.action)

		while not llm_step_output.is_final:
			if self.cnt_iter >= self.max_iter:
				if self.verbose:
					print("Max iterations reached.")
				final_fallback_action = FinalAnswerAction(answer="Maximum iterations reached. Unable to complete the request fully.")
				llm_step_output = LLMStepOutput(thought="Max iterations reached, providing fallback answer.", action=Action(action_type=final_fallback_action), is_final=True)
				break

			if self.verbose:
				print(f"Iteration {self.cnt_iter + 1} / {self.max_iter}")

			action_to_perform = llm_step_output.action.action_type

			if isinstance(action_to_perform, ToolCallAction):
				observation = self.use_tool(action_to_perform.tool_name, action_to_perform.tool_args)
				self.memorize_observation(observation)
				prompt_for_next_step = "Based on the history and previous step, decide the next thought and action."
			elif isinstance(action_to_perform, FinalAnswerAction):
				self.memorize("", "System: Received FinalAnswerAction when is_final was false. Re-evaluating.")
				prompt_for_next_step = "Based on the history and previous step (note: previous step indicated not final but gave a final answer, please clarify or continue), decide the next thought and action."
			else:
				raise ValueError(f"Unknown action type: {type(action_to_perform)}")

			llm_step_output = self.check_request(prompt_for_next_step)
			self.memorize_thought_and_action(llm_step_output.thought, llm_step_output.action)
			self.cnt_iter += 1
		
		if isinstance(llm_step_output.action.action_type, FinalAnswerAction):
			final_answer = llm_step_output.action.action_type.answer
		elif llm_step_output.thought:
			final_answer = f"Process finished. Last thought: {llm_step_output.thought}"
		else:
			final_answer = "Process finished."

		self.save_mem_to_file(
			os.path.join(os.getcwd(), 'temp_vllm.log') # Changed log file name
		)
		self.manage_resource(volatile=volatile) # Pass volatile flag
		return final_answer

	@with_final # Added decorator
	async def stream_call(self, question: str = '', volatile=False): # Aligned with mlx_engine
		llm_step_output: Optional[LLMStepOutput] = None
		# Initial request processing
		async for (parsed_output_item, is_truly_final_token_for_parser) in self.check_request_async(question, init=True):
			if not is_truly_final_token_for_parser:
				yield ("thought_intermediate", parsed_output_item)
			else:
				llm_step_output = parsed_output_item
				if llm_step_output.thought:
					yield ("thought_stream", llm_step_output.thought)
				if llm_step_output.action:
					yield ("action_stream", llm_step_output.action.model_dump_json())
		
		if llm_step_output is None:
			llm_step_output = LLMStepOutput(
				thought="Error: Initial LLM step failed.",
				action=Action(action_type=FinalAnswerAction(answer="Error: Could not process initial request.")),
				is_final=True
			)
		self.memorize_thought_and_action(llm_step_output.thought, llm_step_output.action)

		while llm_step_output and not llm_step_output.is_final:
			if self.cnt_iter >= self.max_iter:
				if self.verbose:
					print("Max iterations reached in stream_call.")
				final_fallback_action = FinalAnswerAction(answer="Maximum iterations reached.")
				llm_step_output = LLMStepOutput(thought="Max iterations reached, fallback answer.", action=Action(action_type=final_fallback_action), is_final=True)
				yield ("thought_stream", llm_step_output.thought)
				yield ("action_stream", llm_step_output.action.model_dump_json())
				break

			if self.verbose:
				yield (f"status_stream", f"Iteration {self.cnt_iter + 1} / {self.max_iter}")

			action_to_perform = llm_step_output.action.action_type
			if isinstance(action_to_perform, ToolCallAction):
				yield ("tool_call_stream", {"name": action_to_perform.tool_name, "args": action_to_perform.tool_args})
				observation = await asyncio.get_event_loop().run_in_executor(
					None, functools.partial(self.use_tool, action_to_perform.tool_name, action_to_perform.tool_args)
				)
				self.memorize_observation(observation)
				yield ("observation_stream", observation.model_dump_json())
				prompt_for_next_step = "Based on the history and previous step, decide the next thought and action."
			elif isinstance(action_to_perform, FinalAnswerAction):
				self.memorize("", "System: Received FinalAnswerAction when is_final was false in stream. Re-evaluating.")
				yield ("status_stream", "Warning: Contradictory FinalAnswerAction received.")
				prompt_for_next_step = "Received unexpected final answer. Please clarify or continue."
			else:
				raise ValueError(f"Unknown action type: {type(action_to_perform)}")

			next_llm_step_output: Optional[LLMStepOutput] = None
			async for (parsed_output_item, is_truly_final_token_for_parser) in self.check_request_async(prompt_for_next_step):
				if not is_truly_final_token_for_parser:
					yield ("thought_intermediate", parsed_output_item)
				else:
					next_llm_step_output = parsed_output_item
					if next_llm_step_output.thought:
						yield ("thought_stream", next_llm_step_output.thought)
					if next_llm_step_output.action:
						yield ("action_stream", next_llm_step_output.action.model_dump_json())
			
			if next_llm_step_output is None:
				next_llm_step_output = LLMStepOutput(
                    thought="Error: LLM step failed during iteration.",
                    action=Action(action_type=FinalAnswerAction(answer="Error: Could not process request further.")),
                    is_final=True)
			llm_step_output = next_llm_step_output
			self.memorize_thought_and_action(llm_step_output.thought, llm_step_output.action)
			self.cnt_iter += 1

		final_answer_content = "Process finished."
		if llm_step_output and isinstance(llm_step_output.action.action_type, FinalAnswerAction):
			final_answer_content = llm_step_output.action.action_type.answer
		elif llm_step_output and llm_step_output.thought:
			final_answer_content = f"Process finished. Last thought: {llm_step_output.thought}"
		
		yield ("final_answer_stream", final_answer_content)

		self.save_mem_to_file(os.path.join(os.getcwd(), 'temp_vllm.log'))
		self.manage_resource(volatile=volatile)
	
	def __ask_LLM( # Aligned with mlx_engine
		self,
		question : str = '',
		image_urls : Optional[List[str]] = None, # Retained for signature consistency
		minimal : Optional[bool] = False, # Passed to retrieve_non_think
		ignore_answer_format : Optional[bool] = False, # Not directly used due to Pydantic focus
		parser: Optional[PydanticOutputParser] = None,
		self_querying: Optional[bool] = False,
	):
		llm_inputs = {}
		if self.USE_VLM and self.image_processor and image_urls:
			images = []
			for url in image_urls:
				try:
					response = requests.get(url, stream=True)
					response.raise_for_status()
					img = Image.open(BytesIO(response.content)).convert("RGB")
					images.append(img)
				except Exception as e:
					print(f"Warning: Failed to load image from {url}. Error: {e}")
			
			if images:
				try:
					# Process images: Creates 'pixel_values' or similar depending on processor
					image_input_data = self.image_processor(images, return_tensors="pt")
					# The key for vLLM's multi_modal_data often 'pixel_values' or 'image_features'
					# This needs to match what the specific LLaVA model expects via vLLM
					# Assuming 'pixel_values' and moving to the client's device
					# vLLM might handle device placement internally if tensors are passed.
					# This part is highly experimental and vLLM version dependent.
					processed_images = self.image_processor(images, return_tensors="pt").pixel_values.to(torch.float16)
					llm_inputs["pixel_values"] = processed_images # Common key for pixel values
					if self.verbose:
						print(f"Successfully processed {len(images)} images for VLM input. Shape: {processed_images.shape}, Dtype: {processed_images.dtype}")
				except Exception as e:
					print(f"Warning: Failed to process images for VLM. Error: {e}")
					llm_inputs.pop("pixel_values", None) # Ensure it's clean if error

		final_question = question
		if images and "<image>" not in question: # Check `images` list
			final_question = "<image>\n" * len(images) + question
			if self.verbose: print(f"Prepended <image> tokens to question for __ask_LLM. New question: {final_question[:100]}...")

		messages_for_template = list(self.memory)
		if parser:
			if isinstance(parser.pydantic_object, LLMToolInputParsingOutput):
				messages_for_template.append({
					"role": "system",
					"content": "Extract tool arguments. Wrap the output in `json` tags\n{format_instructions}".format(
						format_instructions=parser.get_format_instructions()
					)})
		
		messages_for_template.append({
			"role": "user" if not self_querying else "assistant",
			"content": final_question, # Use potentially modified question
		})
		
		prompt_str = self.client.get_tokenizer().apply_chat_template(
			messages_for_template,
			add_generation_prompt=True,
			tokenize=False, 
		)
		
		final_llm_inputs = llm_inputs if "pixel_values" in llm_inputs else None

		request_outputs = self.client.generate(
			prompt_str, 
			sampling_params=self.sampling_params, 
			llm_inputs=final_llm_inputs, 
			use_tqdm=False
		)
		output_text = request_outputs[0].outputs[0].text.strip()

		if self.verbose:
			print(f"**[__ask_LLM raw output (vLLM)]**\n{output_text}\n")
			if final_llm_inputs:
				print(f"**[__ask_LLM VLM input type]**\n{type(final_llm_inputs['pixel_values'])}\n")
				print(f"**[__ask_LLM VLM input shape]**\n{final_llm_inputs['pixel_values'].shape}\n")
				print(f"**[__ask_LLM VLM input dtype]**\n{final_llm_inputs['pixel_values'].dtype}\n")
		
		output_text = retrieve_non_think(output_text, remove_think_only=minimal)

		if parser:
			try:
				parsed_output = parser.invoke(output_text)
			except OutputParserException as e:
				# Adapt RetryOutputParser for vLLM's synchronous client
				# The from_llm method in RetryOutputParser might need adjustment if it expects a LangChain LLM object.
				# For now, assuming it can work with the vLLM client or we pass necessary components.
				# The key is that the retry mechanism needs to be able to make a new call to the LLM.

				# If RetryOutputParser.from_llm expects a specific LLM interface (e.g. LangChain's)
				# we might need to wrap self.client or pass a callable that makes the LLM call.
				# For now, let's assume the existing RetryOutputParser can be made to work
				# by passing self.client and then handling the actual call within parse_with_prompt.
				
				# Simplified: If direct from_llm is problematic, one might need to manually implement retry logic
				# or use a more basic RetryPydanticOutputParser if available.
				# Given the existing structure from mlx_engine, we try to adapt.
				
				# The `llm` parameter in `from_llm` is used by `RetryOutputParser` to make further calls.
				# It expects an object with an `invoke` method (if LangChain style) or similar.
				# vLLM's `LLM` object has `generate`. We need to ensure compatibility or adapt.
				# For `parse_with_prompt`, we pass `llm_provider='vllm'` and `sampling_params`.
				
				# Let's refine the retry mechanism slightly.
				# We need a callable that RetryOutputParser can use.
				def _vllm_generate_sync_for_retry(prompt_str_for_retry: str) -> str:
					retry_outputs = self.client.generate(prompt_str_for_retry, sampling_params=self.sampling_params, use_tqdm=False)
					return retry_outputs[0].outputs[0].text.strip()

				retry_parser = RetryOutputParser( # Instantiate directly if from_llm is problematic
					parser=parser,
					retry_llm_call=_vllm_generate_sync_for_retry, # Pass the callable
					max_retries=self.max_iter,
					prompt_template=NAIVE_COMPLETION_RETRY # This template is used to format the retry prompt
				)
				# parse_with_prompt will use the callable for retries.
				# It needs the original failing output and the initial prompt that led to it.
				parsed_output = retry_parser.parse_with_prompt(
					output_text, # The original failing output
					prompt_str   # The prompt that generated output_text
				)
			return output_text, parsed_output
		return output_text

	@with_final # Added decorator
	async def __ask_LLM_async( # NEW: Aligned with mlx_engine
		self,
		question: str = '',
		image_urls: Optional[List[str]] = None,
		minimal: Optional[bool] = False,
		ignore_answer_format: Optional[bool] = False,
		parser: Optional[PydanticOutputParser] = None,
		self_querying: Optional[bool] = False,
	) -> AsyncGenerator[Union[str, tuple[str, Any]], None]:
		llm_inputs_dict = {} # Renamed to avoid conflict with outer scope if any, and make it clear it's a dict
		if self.USE_VLM and self.image_processor and image_urls:
			# Synchronous image loading in async context (acceptable for experiment, note for improvement)
			images = []
			for url in image_urls:
				try:
					if self.verbose: print(f"Attempting to load image from URL: {url}")
					response = requests.get(url, stream=True, timeout=10) # Added timeout
					response.raise_for_status()
					img = Image.open(BytesIO(response.content)).convert("RGB")
					images.append(img)
					if self.verbose: print(f"Successfully loaded image from URL: {url}")
				except requests.exceptions.RequestException as e: # More specific exception
					print(f"Warning (async): Failed to load image from {url} due to network issue. Error: {e}")
				except Exception as e: # General exception for other PIL issues etc.
					print(f"Warning (async): Failed to load or process image from {url}. Error: {e}")
			
			if images:
				try:
					# This processing is CPU-bound, so doing it sync here is okay before async LLM call
					# Ensure image_input_data is correctly structured for vLLM
					# For many models, pixel_values is expected directly.
					# The structure {"multi_modal_data": {"pixel_values": ...}} might be for specific vLLM versions or models.
					# Let's try passing pixel_values directly in llm_inputs if that's more common for recent vLLM LLaVA.
					# If vLLM expects a 'multi_modal_data' dict, this will need to be {"multi_modal_data": {"pixel_values": ...}}
					# For now, assume direct pixel_values might be part of llm_inputs.
					# This part is highly experimental and vLLM version dependent.
					processed_images = self.image_processor(images, return_tensors="pt").pixel_values.to(torch.float16)
					llm_inputs_dict["pixel_values"] = processed_images # Common key for pixel values
					# llm_inputs_dict["image_features"] = processed_images # Some models might expect this key
					if self.verbose:
						print(f"Successfully processed {len(images)} images for VLM input (async path). Shape: {processed_images.shape}, Dtype: {processed_images.dtype}")
				except Exception as e:
					print(f"Warning (async): Failed to process images for VLM. Error: {e}")
					# llm_inputs_dict.pop("pixel_values", None) # Ensure it's clean if error
		
		# Ensure question has <image> placeholders if images are present.
		# Some chat templates add it, some don't. If self.image_processor is present and images were loaded,
		# it's good practice to ensure the token is there.
		# This is a simplified check. A robust solution would parse the template.
		final_question = question
		if images and "<image>" not in question: # Check `images` list, not just llm_inputs_dict
			final_question = "<image>\n" * len(images) + question
			if self.verbose: print(f"Prepended <image> tokens to question. New question: {final_question[:100]}...")


		messages_for_template_async = list(self.memory)
		if parser:
			if isinstance(parser.pydantic_object, LLMToolInputParsingOutput):
				messages_for_template_async.append({
					"role": "system",
					"content": "Extract tool arguments. Wrap the output in `json` tags\n{format_instructions}".format(
						format_instructions=parser.get_format_instructions()
					)})
		
		messages_for_template_async.append({
			"role": "user" if not self_querying else "assistant",
			"content": final_question, # Use the potentially modified question
		})

		prompt_str = self.client.get_tokenizer().apply_chat_template(
			messages_for_template_async, # This now contains the <image> token if added
			add_generation_prompt=True,
			tokenize=False,
		)
		
		request_id = f"vllm-engine-async-{time.monotonic_ns()}"
		
		# Pass llm_inputs_dict directly to generate if it contains data, else None
		final_llm_inputs = llm_inputs_dict if "pixel_values" in llm_inputs_dict else None

		results_generator = self.client.generate(
			prompt_str,
			sampling_params=self.sampling_params_stream,
			request_id=request_id,
			llm_inputs=final_llm_inputs,
		)

		output_buffer = []
		previous_text_len = 0
		async for request_output in results_generator:
			current_text = request_output.outputs[0].text # Full text so far
			# It's possible that due to stripping/processing, text_delta calculation needs to be robust.
			# However, vLLM's stream=True for generate() method with RequestOutput typically gives cumulative text.
			text_delta = current_text[previous_text_len:]
			
			if text_delta: # Only yield if there's new text
				yield text_delta
				output_buffer.append(text_delta) # Append only the delta
			previous_text_len = len(current_text) # Update with the full length of current text
			
			if request_output.finished:
				break
		
		full_output_str = "".join(output_buffer) # This is now correctly just the generated part
		full_output_str = retrieve_non_think(full_output_str.strip(), remove_think_only=minimal)

		if parser:
			try:
				parsed_obj = parser.invoke(full_output_str)
			except OutputParserException as e:
				# Async retry needs careful implementation. For now, propagate error or handle simply.
				print(f"Async parsing failed in vLLM: {e}. Raw output: {full_output_str}")
				raise # Re-raise for now
			yield full_output_str, parsed_obj
		else:
			yield full_output_str
	
	# Removed __retrieve_answer_dict

	def check_request(self, question: str, init: bool = False) -> LLMStepOutput: # Aligned with mlx_engine
		llm_input = question # init flag not directly used here, but kept for signature consistency
		parser = PydanticOutputParser(pydantic_object=LLMStepOutput)
		raw_output, parsed_output = self.__ask_LLM(llm_input, parser=parser)
		
		if not isinstance(parsed_output, LLMStepOutput):
			raise ValueError(f"Expected LLMStepOutput from parser, got {type(parsed_output)}")
		return parsed_output

	@with_final # Added decorator
	async def check_request_async(self, question: str, init: bool = False) -> AsyncGenerator[Union[str, LLMStepOutput], None]: # Aligned
		llm_input = question
		parser = PydanticOutputParser(pydantic_object=LLMStepOutput)
		
		raw_str_final = None
		parsed_obj_final = None

		async for item in self.__ask_LLM_async(llm_input, parser=parser):
			if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], LLMStepOutput):
				raw_str_final, parsed_obj_final = item 
			else: # Intermediate token
				yield item 
		
		if parsed_obj_final is None:
			parsed_obj_final = LLMStepOutput(
				thought="Error: LLM stream ended unexpectedly in check_request_async (vLLM).",
				action=Action(action_type=FinalAnswerAction(answer="Error processing request.")),
				is_final=True)
		yield parsed_obj_final # Final parsed object for @with_final

	def use_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Observation: # Aligned
		tool_to_use = self.tools.get(tool_name)
		if not tool_to_use:
			err_msg = f"Tool '{tool_name}' not found."
			return Observation(observation_type="tool_error", content=err_msg, metadata={"tool_name": tool_name, "args": tool_args, "error_type": "ToolNotFound"})
		
		try:
			tool_result = tool_to_use.invoke(tool_args)
			return Observation(observation_type="tool_result", content=str(tool_result), metadata={"tool_name": tool_name, "args": tool_args})
		except Exception as e:
			if self.verbose:
				print(f"Error using tool {tool_name} with args {tool_args}: {e}")
			return Observation(observation_type="tool_error", content=str(e), metadata={"tool_name": tool_name, "args": tool_args, "error_type": type(e).__name__})

	# Memory methods - Aligned with mlx_engine
	def memorize_thought_and_action(self, thought: Optional[str], action: Action):
		if thought:
			self.memory.append({"role": "assistant", "content": f"Thought: {thought}"})
		
		action_content = ""
		if isinstance(action.action_type, ToolCallAction):
			action_content = f"Action: Call tool '{action.action_type.tool_name}' with args {action.action_type.tool_args}"
		elif isinstance(action.action_type, FinalAnswerAction):
			action_content = f"Action: Provide final answer: '{action.action_type.answer}'"
		
		if action.rationale:
			action_content += f"\nRationale: {action.rationale}"
		self.memory.append({"role": "assistant", "content": action_content})

	def memorize_observation(self, observation: Observation):
		self.memory.append({
			"role": "user", 
			"content": f"Observation ({observation.observation_type} for tool {observation.metadata.get('tool_name', 'N/A')}): {observation.content}"
		})

	def memorize(self, user_query: str, assistant_response: str, verbose: bool = False, self_querying: bool = False): # Added self_querying for consistency
		if user_query:
			self.memory.append({
				"role": "user" if not self_querying else "assistant", 
				"content": user_query,
			})
		if assistant_response:
			self.memory.append({
				"role": "assistant",
				"content": assistant_response,
			})

	def clear_memory(self): # Aligned
		self.memory.clear()
		self.__init_memory(self.verbose)
	
	def manage_resource(self, volatile=True): # Aligned
		if volatile:
			self.clear_memory()
		self.cnt_iter = 0
	
	def shutdown(self): # Added for completeness, though vLLM might have specific cleanup
		self.manage_resource(volatile=True)
		# Consider adding vllm.distributed.cleanup_dist_env_and_memory() if distributed setup is used.
		# For non-distributed, direct cleanup of self.client might not be standard unless explicitly documented by vLLM.

	# Removed continue_iteration, plan_subtask, finalize methods

	def save_mem_to_file(self, file_path: str): # Aligned
		try:
			with open(file_path, "w", encoding='utf-8') as f:
				for m in self.memory:
					role = m.get("role", "unknown")
					content = m.get("content", "")
					f.write(f"[{role}]\n{content}\n\n")
			if self.verbose:
				print(f"Memory saved to '{file_path}'.")
		except Exception as e:
			print(f"Error saving memory to file '{file_path}': {e}")