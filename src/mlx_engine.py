import time, re, os
import asyncio
from typing import List, Optional, Dict, Any

from src.tools import process_request, process_binary, save_file
from src.utils import ANSWER_DICT_REGEXP, THINK_REGEXP, retrieve_non_think, json_schema_to_base_model
from src.prompt_template import (
	LC_SYSTEM_PROMPT_TEMPLATE,
	#SYSTEM_PROMPT_TEMPLATE,
	#PROMPT_TEMPLATE,
	SYSTEM_ANSWER_TEMPLATE,
	NAIVE_COMPLETION_RETRY,
)
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
		if manual_answer_format and manual_answer_format != '':
			# NOTE: Replace with manual_answer_format given. Currently using legacy for debug purpose.
			self.validation_answer_format = SYSTEM_ANSWER_TEMPLATE
		else:
			self.validation_answer_format = None
		# Initialize chat memory with system prompt.
		self.__init_memory(self.verbose)

	def __init_memory(self, verbose:bool=False):
		self.memory.append({
			"role": "system", #'user',
			"content": self.validation_prompt.format(
				max_iteration=self.max_iter,
				modality_in=self.modality_in,
				modality_out=self.modality_out,
				# NOTE: Legacy tool description for testing the availablity of using prompt instructions only to force the structured output.
				#tool_list='\n'.join(f"- {t_name}: {tool.description}\n\tArgs: {tool.inputs}\n\tReturns: {tool.output_type}" for t_name,tool in self.tools.items()),
				tool_list='\n'.join(f"- {t_name}: {tool.description}" for t_name,tool in self.tools.items()),
				#max_new_tokens=self.max_new_tokens,
		)#+'\n'+self.validation_answer_format})
		})
		if self.validation_answer_format:
			# NOTE: Separate answer format from the body. This is for further INTENDED IGNORE of answer format.
			self.memory.append({
				"role": "user",
				"content": self.validation_answer_format
			})

	def __call__(self, question:str='', volatile=False):
		# Check if any TOOLs or HELPs needed.
		t_req, h_req, ans, res = self.check_request(question, init=True if volatile else False)
		# If tool is needed, use tool.
		if t_req:
			self.use_tool(t_req)
		# If helper is needed, use helper.
		if h_req:
			# Initialize iterations for the given question.
			self.plan_subtask(init=True)
		
		while self.continue_iteration(volatile=volatile):
			if self.verbose:
				print(f"Continue iteration.. ({self.cnt_iter} / {self.max_iter})")

			# Decide action.
			llm_input_action = "Based on the context, decide the next action for you to solve the problem."
			#action = self.__ask_LLM(llm_input_action)
			t_req_action, h_req_action, ans_action, res_action = self.check_request(llm_input_action)
			# Store to memory.
			#self.memorize(llm_input_action, ans_action)

			if not (t_req_action or h_req_action):
				if ans_action is None:
					continue
				llm_input_observation = f"Do: {ans_action}"
			else:
				if t_req_action:
					self.use_tool(t_req_action)
				if h_req_action:
					self.plan_subtask(init=False)
				llm_input_observation = f"Finished tool-using and help-requesting of current action. Do rest part within this action({ans_action})"

			# Do action and retrieve observation.
			observation = self.__ask_LLM(llm_input_observation, self_querying=True)
			# Store to memory.
			self.memorize(llm_input_observation, observation, self_querying=True)

		# Get final answer.
		final_answer = self.finalize()

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

		# Check if any TOOLs or HELPs needed.
		t_req, h_req, ans, res = '', '', '', ''
		async for (token, is_final) in self.check_request_async(
			question,
			init=True if volatile else False,
		):
			if not is_final:
				yield ("thought_intermediate", token)
				#await asyncio.sleep(0)
			else:
				t_req, h_req, ans, res = token
				yield ("thought_answer", ans)
		await asyncio.sleep(0)
		
		####### SYNC VERSION #######
		# If tool is needed, use tool.
		if t_req:
			self.use_tool(t_req)
		# If helper is needed, use helper.
		if h_req:
			# Initialize iterations for the given question.
			self.plan_subtask(init=True)
		####### SYNC VERSION #######

		while self.continue_iteration(volatile=volatile):
			# Decide action.
			llm_input_action = "Based on the context, decide the next action for you to solve the problem."
			t_req_action, h_req_action, ans_action, res_action = None, None, None, None

			async for (token, is_final) in self.check_request_async(llm_input_action):
				if not is_final:
					yield ("thought_intermediate", token)
					#await asyncio.sleep(0)
				else:
					t_req_action, h_req_action, ans_action, res_action = token
					yield ("thought_answer", ans_action)

			##### SYNC VERSION #####
			if not (t_req_action or h_req_action):
				if ans_action is None:
					continue
				llm_input_observation = f"Do: {ans_action}"
			else:
				if t_req_action:
					self.use_tool(t_req_action)
				if h_req_action:
					self.plan_subtask(init=False)
				llm_input_observation = f"Finished tool-using and help-requesting of current action. Do rest part within this action({ans_action})"
			##### SYNC VERSION #####

			# Do action and retrieve observation.
			observation = None
			async for (token, is_final) in self.__ask_LLM_async(llm_input_observation, self_querying=True):
				if not is_final:
					yield ("thought_intermediate", token)
					#await asyncio.sleep(0)
				else:
					observation = token
					yield ("thought_answer", observation)
			# Store to memory.
			self.memorize(llm_input_observation, observation, self_querying=True)

		# Get final answer.
		final_answer = None
		async for (token, is_final) in self.finalize_async():
			if not is_final:
				yield ("final_intermediate", token)
			else:
				final_answer = token
				yield ("final_answer", final_answer)

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
					(self.memory[0:1] + (self.memory[2:] if len(self.memory)>2 else []) if ignore_answer_format and self.validation_answer_format else self.memory) + \
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
			except ValueError as e:
				print(f"\n**ERROR**\nGot ValueError by trying to use Multi-image chat with the model not supported. Fall back to use first image only.\nOriginal error message -> {e}\n**ERROR**\n")
				image_urls = [image_urls[0]]
				formatted_prompt = vlm_apply_chat_template(
					self.processor, self.config,
					(self.memory[0:1] + (self.memory[2:] if len(self.memory)>2 else []) if ignore_answer_format and self.validation_answer_format else self.memory) + \
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
			u_id = "user" if not self_querying else "assistant"
			prompt = self.tokenizer.apply_chat_template(
				(self.memory[0:1] + (self.memory[2:] if len(self.memory)>2 else []) if ignore_answer_format and self.validation_answer_format else self.memory) + \
				([{
					"role": "system",
					"content": "Answer the {user} query. Wrap the output in `json` tags\n{format_instructions}".format(
						user=u_id,
						format_instructions=parser.get_format_instructions()
					)
				}] if parser else []) + \
				[{
					"role": "user" if not self_querying else "assistant",
					"content": question,
				}],
				add_generation_prompt=True,
				tokenize=False if parser else True, # NOTE: In the sturctured output parsing for tool-calling use-case, set tokenize=False.
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
		u_id = "user" if not self_querying else "assistant"
		prompt = self.tokenizer.apply_chat_template(
			(self.memory[0:1] + (self.memory[2:] if len(self.memory)>2 else []) if ignore_answer_format and self.validation_answer_format else self.memory) + \
			([{
				"role": "system",
				"content": "Answer the {user} query. Wrap the output in `json` tags\n{format_instructions}".format(
					user=u_id,
					format_instructions=parser.get_format_instructions()
				)
			}] if parser else []) + \
			[{
				"role": "user" if not self_querying else "assistant",
				"content": question,
			}],
			add_generation_prompt=True,
			tokenize=False if parser else True, # NOTE: In the sturctured output parsing for tool-calling use-case, set tokenize=False.
		)
		#### SYNC VERSION ####

		output = ''
		for response in stream_generate(
			self.client,
			self.tokenizer,
			prompt=prompt,
			max_tokens=self.max_new_tokens,
			## NOTE: kwargs for `generate_step()`
			kv_bits=4, # KV cache quantization bits
		):
			token = response.text
			yield token
			await asyncio.sleep(0)
			output += token

		#### SYNC VERSION ####
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
			yield output, parsed_output #NOTE: return string-type output also, to be used in 'memorize'.
		#### SYNC VERSION ####
		else:
			yield output


	def __retrieve_answer_dict(self, str_answer:str) -> Dict:
		ans_dict_list = []
		for candidate in ANSWER_DICT_REGEXP.finditer(str_answer.strip()):
			try:
				c = eval(candidate.group())
				assert isinstance(c, dict), "{candidate} is not a dict."
				ans_dict_list.append(c)
			except :
				continue
		if len(ans_dict_list) < 1:
			return dict()
		return ans_dict_list[-1]

	def check_request(self, question='', init=False):
		# 1. Check question and what things are needed.
		llm_input = question # + '\n' + self.validation_answer_format
		#if init:
		llm_input = f"**QUESTION**\n{question}"

		if self.validation_answer_format:
			str_answer = self.__ask_LLM(llm_input)
			answer_dict = self.__retrieve_answer_dict(str_answer)
		else:
			parser = PydanticOutputParser(pydantic_object=json_schema_to_base_model(process_request.args_schema.model_json_schema()))
			# NOTE: parsing conducted inside of `__ask_LLM` function.
			str_answer, answer_dict = self.__ask_LLM(llm_input, parser=parser)
		
		invalid_set = {"nil", "none"}
		if isinstance(answer_dict, dict):
			#raise ValueError(f"Legacy style returned. Check MLXEngine().check_request().")
			t_req, h_req, ans, res = None, None, None, None
			if "Tool_Request" in answer_dict:
				t_req = answer_dict["Tool_Request"]
				if t_req.lower() in invalid_set: 
					t_req = None
			if "Helper_Request" in answer_dict:
				h_req = answer_dict["Helper_Request"]
				if h_req.lower() in invalid_set: 
					h_req = None
			if "Answer" in answer_dict:
				ans = answer_dict["Answer"]
				if ans.lower() in invalid_set: 
					ans = None
			else:
				ans = str_answer
			if "Rationale" in answer_dict:
				res = answer_dict["Rationale"]
				if res.lower() in invalid_set: 
					res = None
		else:
			t_req = answer_dict.tool_request
			if t_req.lower() in invalid_set: t_req = None
			h_req = answer_dict.helper_request
			if h_req.lower() in invalid_set: h_req = None
			ans = answer_dict.answer
			if ans.lower() in invalid_set: ans = None
			res = answer_dict.rationale
			if res.lower() in invalid_set: res = None

		# Save result to memory.
		self.memorize(llm_input, str_answer)
		# Return tools and help requests.
		return t_req, h_req, ans, res
	
	@with_final
	async def check_request_async(self, question='', init=False):
		# 1. Check question and what things are needed.
		llm_input = f"**QUESTION**\n{question}"

		if self.validation_answer_format:
			str_answer = None
			async for (token, is_final) in self.__ask_LLM_async(llm_input):
				if not is_final:
					yield token
					#await asyncio.sleep(0)
				else:
					str_answer = token
			if str_answer is None:
				str_answer = token
			answer_dict = self.__retrieve_answer_dict(str_answer)
		else:
			parser = PydanticOutputParser(pydantic_object=json_schema_to_base_model(process_request.args_schema.model_json_schema()))
			# NOTE: parsing conducted inside of `__ask_LLM` function.
			str_answer, answer_dict = None, None
			async for (token, is_final) in self.__ask_LLM_async(llm_input, parser=parser):
				if not is_final:
					yield token
					#await asyncio.sleep(0)
				else:
					str_answer, answer_dict = token
			if str_answer is None:
				str_answer, answer_dict = token, {}
		
		invalid_set = {"nil", "none"}
		if isinstance(answer_dict, dict):
			#raise ValueError(f"Legacy style returned. Check MLXEngine().check_request().")
			t_req, h_req, ans, res = None, None, None, None
			if "Tool_Request" in answer_dict:
				t_req = answer_dict["Tool_Request"]
				if t_req.lower() in invalid_set: 
					t_req = None
			if "Helper_Request" in answer_dict:
				h_req = answer_dict["Helper_Request"]
				if h_req.lower() in invalid_set: 
					h_req = None
			if "Answer" in answer_dict:
				ans = answer_dict["Answer"]
				if ans.lower() in invalid_set: 
					ans = None
			else:
				ans = str_answer
			if "Rationale" in answer_dict:
				res = answer_dict["Rationale"]
				if res.lower() in invalid_set: 
					res = None
		else:
			t_req = answer_dict.tool_request
			if t_req.lower() in invalid_set: t_req = None
			h_req = answer_dict.helper_request
			if h_req.lower() in invalid_set: h_req = None
			ans = answer_dict.answer
			if ans.lower() in invalid_set: ans = None
			res = answer_dict.rationale
			if res.lower() in invalid_set: res = None

		# Save result to memory.
		self.memorize(llm_input, str_answer)
		# Return tools and help requests.
		yield t_req, h_req, ans, res

	
	def use_tool(self, t_req: str):
		# Do pattern matching for tool calling request.
		# if there is a match, use that tool.
		t_req = t_req.lower()
		# NOTE: Parse.
		if t_req not in self.tools:
			parsed_req = self.__ask_LLM(f'Get the name of the tool that corresponds to the following request:`{t_req}`. You should lookup for the available tools you have. Return the name of the tool only (without any special characters or any other words)')
			t_req = parsed_req.lower()
		# Default
		if t_req not in self.tools:
			err_msg = f"Corresponding tool for the request of '{t_req}' does not exist."
			self.memorize('', f"[Tool calling error report]\n{err_msg}\n[Tool calling error report end]\n")
			return
		
		'''Legacy (instruction-only forcing structured output.)
		retrieve_prompt = f'Return the dictionary of inputs for the tool \'{t_req}\'. The tool spec is following:\n**Description**\n{self.tools[t_req].description}\n**Args**\n{self.tools[t_req].inputs}\n**Returns**\n{self.tools[t_req].output_type}'
		'''
		# NOTE: Systemic approach for structured output.
		retrieve_prompt = f'Return the dictionary of inputs for the tool \'{t_req}\' for current step.'
		parser = PydanticOutputParser(
			pydantic_object=json_schema_to_base_model(self.tools[t_req].args_schema.model_json_schema())
		)
		if parser:
			t_arg_str, t_arg = self.__ask_LLM(retrieve_prompt, ignore_answer_format=True, parser=parser)
			t_arg = t_arg.model_dump()
			try:
				tool_result = self.tools[t_req](t_arg)
			except :
				tool_result = self.tools[t_req].invoke(t_arg)
		else:
			t_arg_str = self.__ask_LLM(retrieve_prompt, ignore_answer_format=True, parser=parser)
			t_arg = self.__retrieve_answer_dict(t_arg_str)
			'''
			if "Tool_Request" in t_arg:
				del t_arg["Tool_Request"]
			if "Helper_Request" in t_arg:
				del t_arg["Helper_Request"]
			'''
			tool_result = self.tools[t_req](**{k:v for k,v in t_arg.items() if k in self.tools[t_req].inputs})
		# Record tool results for later use.
		if tool_result:
			self.memorize('', f'[Observation of Tool {t_req}]\nWith argument: `{t_arg}`\n'+tool_result+'\n[Observation end]\n')
		
	# Memory
	def memorize(self, user_query: str, assistant_response: str, verbose: bool = False, self_querying: bool = False):
		if user_query != '':
			self.memory.append({
				"role": "user" if not self_querying else "assistant", 
				"content": user_query,
			})
		if assistant_response != '':
			self.memory.append({
				"role": "assistant",
				"content": assistant_response,
			})

	def clear_memory(self):
		del self.memory
		self.memory = []
		self.__init_memory(True)
	
	def manage_resource(self, volatile=True):
		if volatile:
			self.clear_memory()
		self.cnt_iter = 0

	def shutdown(self):
		self.manage_resource(volatile=True)

	# Decision
	def continue_iteration(self, use_legacy=False, volatile=True):
		if len(self.memory) <= 1:
			# Only system prompt.
			#self.cnt_iter += 1
			return True
		if self.max_iter <= self.cnt_iter:
			return False
		
		# Ask if the answer was given for the question.
		# If no, we should continue iteration.
		# If yes, we should stop iteration.
		if use_legacy:
			'''Legacy (instruction-only)
			'''
			assistant_decision = self.__ask_LLM(
				#"Do you think you resolved the initial question? If you think so, answer 'Yes'. If not, answer 'No'. You must answer in either 'Yes' or 'No', without any other strings.",
				"Do you think you resolved the question? If you think so, answer 'Yes'. If not, answer 'No'. You must answer in either 'Yes' or 'No', without any other strings.",
				minimal=False,
			).lower()
		else:
			# NOTE: System-instructed structured processing.
			parser = PydanticOutputParser(
				pydantic_object=json_schema_to_base_model(process_binary.args_schema.model_json_schema())
			)
			_, assistant_decision_pydantic = self.__ask_LLM(
				#"Do you think you resolved the initial question? If you think so, answer 'Yes'. If not, answer 'No'.",
				"Do you think you resolved the question? If you think so, answer 'Yes'. If not, answer 'No'.",
				parser=parser,
			)
			assistant_decision = assistant_decision_pydantic.yes_or_no.lower()

		if 'no' in assistant_decision:
			self.cnt_iter += 1
			return True
		elif 'yes' in assistant_decision:
			return False
		else:
			self.manage_resource(volatile=volatile)
			raise ValueError(f"Agent behaved weird during `self.continue_iteration()`.\nResponse was: {assistant_decision}")

	def plan_subtask(self, init=True):
		if init:
			#llm_input = f"You are a domain expert. Based on the question I gave, plan the subtasks in order to answer the given question."
			llm_input = f"Plan the subtasks in order to answer the given question, be aware to use the tools or helpers if needed."
		else:
			llm_input = f"Evaluate the task progress so far. If needed, refine the sub-tasks to solve current problem. Be aware to use the tools or helpers if needed."
		assistant_plan = self.__ask_LLM(llm_input, ignore_answer_format=True)
		self.memorize(llm_input, assistant_plan, self_querying=False)

	# For final answer.
	def finalize(self):
		#llm_input = "Using the context, finalize your answer. For final answer, you must ignore **Answer Format** given earlier and just respond normally but concise."
		llm_input = "Using the context, finalize thinking and generate answer. You must ignore **Answer Format** given earlier and just respond normally but concise, without exposing any details of thinking process."
		final_response = self.__ask_LLM(llm_input, minimal=False, self_querying=True)
		self.memorize(llm_input, final_response, self_querying=True)
		return final_response
	@with_final
	async def finalize_async(self):
		llm_input = "Using the context, finalize thinking and generate answer. You must ignore **Answer Format** given earlier and just respond normally but concise, without exposing any details of thinking process."
		final_response = ''
		async for (token, is_final) in self.__ask_LLM_async(llm_input, minimal=False, self_querying=True):
			if not is_final:
				yield token
				await asyncio.sleep(0)
				#final_response += token
			else:
				final_response = token
		if final_response is None:
			final_response = token
		
		self.memorize(llm_input, final_response, self_querying=True)
		yield final_response


	# For saving final memory to the file.
	def save_mem_to_file(self, file_path: str):
		with open(file_path, "w+", encoding='utf-8') as f:
			f.write(
				"\n".join([
					f"[{m['role']}]\n{m['content']}\n"
				for m in self.memory])
			)
		print(f"File '{file_path}' has been saved.")