import time, re, os
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

'''
from mlx_lm import load, generate

from mlx_vlm import (
	load as vlm_load,
	generate as vlm_generate,
)
from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template
from mlx_vlm.utils import load_config as vlm_load_config
'''
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

class vLLMEngine():
	def __init__(
		self,
		model : str = 'unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit',
		max_iteration : int = 5,
		verbose : bool = False,
		tools : List[Optional[Any]] = [],
		modality_io : str = 'text/text',
		max_new_tokens : int = 1024,
		manual_answer_format : Optional[str] = '', # To denote Legacy-style `prompt-level` forcing of structured output.
	):
		self.model = model
		self.max_iter = max_iteration
		self.verbose = verbose

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
			# NOTE: Not tested for vLLM usage, yet.
			raise NotImplementedError("Currently, Multi-modal usage with vLLM not tested.")
		else:
			self.client = LLM(
				model=self.model,
				gpu_memory_utilization=0.9, # NOTE: Change depending on the use-cases.
				cpu_offload_gb=4, 			# NOTE: Only use if GPU's RAM is not sufficient for the use-case. Trade-offs latency. Unit: GiB.
											# 		If not needed, comment out `cpu_offload_gb=...,`, since it makes system slow.
				#dtype='auto',
				#dtype='half', 				# NOTE: if your environment(GPU, ...) supports bfloat16, try that one.
				#kv_cache_dtype='fp8', 		# NOTE: If error occurs, comment out this and comment out `calculate_kv_scales=True`
				#calculate_kv_scales=True,
				#quantization='AWQ',
				#quantization='bitsandbytes',# bnb not supported for my hardware...
				#load_format='bitsandbytes', # bnb not supported for my hardware...
				trust_remote_code=True,
				#enable_chunked_prefill=True,
				max_model_len=max_new_tokens*4, # NOTE: Change the value for each use-cases.
				enforce_eager=True, 		# NOTE: To use bnb in vLLM, currently need this argument.
			)
			self.sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.5)

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
				tool_list='\n'.join(f"- {t_name}: {tool.description}" for t_name,tool in self.tools.items()),
				#max_new_tokens=self.max_new_tokens,
			)
		})
		if self.validation_answer_format:
			# NOTE: Separate answer format from the body. This is for further INTENDED IGNORE of answer format.
			self.memory.append({
				"role": "user",
				"content": self.validation_answer_format
			})

	def __call__(self, question:str=''):
		# Check if any TOOLs or HELPs needed.
		t_req, h_req, ans, res = self.check_request(question, init=True)
		# If tool is needed, use tool.
		if t_req:
			self.use_tool(t_req)
		# If helper is needed, use helper.
		if h_req:
			# Initialize iterations for the given question.
			self.plan_subtask(init=True)
		
		while self.continue_iteration():
			if self.verbose:
				print(f"Continue iteration.. ({self.cnt_iter} / {self.max_iter})")

			# Decide action.
			llm_input_action = "Based on the context, decide the next action for you to solve the problem."
			#action = self.__ask_LLM(llm_input_action)
			t_req_action, h_req_action, ans_action, res_action = self.check_request(llm_input_action)
			# Store to memory.
			#self.memorize(llm_input_action, ans_action)

			if not (t_req_action or h_req_action):
				llm_input_observation = f"Do: {ans_action}"
			else:
				if t_req_action:
					self.use_tool(t_req_action)
				if h_req_action:
					self.plan_subtask(init=False)
				llm_input_observation = f"Finished tool-using and help-requesting of current action. Do rest part within this action({ans_action})"

			# Do action and retrieve observation.
			observation = self.__ask_LLM(llm_input_observation)
			# Store to memory.
			self.memorize(llm_input_observation, observation)

		# Get final answer.
		final_answer = self.finalize()

		# Save the memory to the text file by LLM.
		self.save_mem_to_file(
			os.path.join(os.getcwd(), 'temp_mlx.log')
		)
		# Clean up resources.
		self.manage_resource()

		return final_answer
	
	def __ask_LLM(
		self,
		question : str = '',
		image_urls : Optional[List[str]] = None,
		minimal : Optional[bool] = False,
		ignore_answer_format : Optional[bool] = False,
		parser: Optional[PydanticOutputParser] = None, # NOTE: Provide parser to use auto-parsing structured output for tools generated with @tool decorator.
	):
		if image_urls is None:
			image_urls = []

		# NOTE: VLM not tested since beginning of tool execution implementation.
		if self.USE_VLM:
			raise NotImplementedError("Currently, using Multi-modal model with vLLM is not tested.")
		else:
			#prompt = self.tokenizer.apply_chat_template(
			prompt = self.client.get_tokenizer().apply_chat_template(
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
				}],
				add_generation_prompt=True,
				#tokenize=False if parser else True, # NOTE: In the sturctured output parsing for tool-calling use-case, set tokenize=False.
				tokenize=False,
			) 
			#output = self.client.chat(
			output = self.client.generate(
				prompt,
				sampling_params=self.sampling_params,
				use_tqdm=False,
				#tools=[], # NOTE: Unlike normal vLLM usage, we do not use explicit `tools` argument of vLLM.
				#				But implemented tool chaining with other parts in the agent.
				#				This enable the LLMs whom are not trained to use tools to use the tools in the inference.
			)[0].outputs[0].text.strip()
			if self.verbose:
				print(f"**[__ask_LLM raw output]**\n{output}\n")
			#output = output[0].outputs[0].text.strip()
		output = retrieve_non_think(output, remove_think_only=minimal)
		# NOTE: structured output auto-parsing
		if parser:
			try:
				parsed_output = parser.invoke(output)
			except OutputParserException as e:
				retry_parser = RetryOutputParser.from_llm(
					parser=parser,
					llm=self.client, # NOTE: different from `mlx` version.
					prompt_template=NAIVE_COMPLETION_RETRY,
					max_retries=self.max_iter,
				)
				parsed_output = retry_parser.parse_with_prompt(output, prompt, llm_provider='vllm', sampling_params=self.sampling_params)
			return output, parsed_output #NOTE: return string-type output also, to be used in 'memorize'.
		return output

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
		if init:
			llm_input = f"**QUESTION**\n{question}"

		if self.validation_answer_format:
			str_answer = self.__ask_LLM(llm_input)
			answer_dict = self.__retrieve_answer_dict(str_answer)
		else:
			parser = PydanticOutputParser(pydantic_object=json_schema_to_base_model(process_request.args_schema.model_json_schema()))
			# NOTE: parsing conducted inside of `__ask_LLM` function.
			str_answer, answer_dict = self.__ask_LLM(llm_input, parser=parser)
		
		if isinstance(answer_dict, dict):
			#raise ValueError(f"Legacy style returned. Check vLLMEngine().check_request().")
			t_req, h_req, ans, res = None, None, None, None
			if "Tool_Request" in answer_dict:
				t_req = answer_dict["Tool_Request"]
				if t_req.lower() == "nil":
					t_req = None
			if "Helper_Request" in answer_dict:
				h_req = answer_dict["Helper_Request"]
				if h_req.lower() == "nil":
					h_req = None
			if "Answer" in answer_dict:
				ans = answer_dict["Answer"]
				if ans.lower() == "nil":
					ans = None
			else:
				ans = str_answer
			if "Rationale" in answer_dict:
				res = answer_dict["Rationale"]
				if res.lower() == "nil":
					res = None
		else:
			t_req = answer_dict.tool_request
			if t_req.lower() in {'nil', 'none', ''}: t_req = None
			h_req = answer_dict.helper_request
			if h_req.lower() in {'nil', 'none', ''}: h_req = None
			ans = answer_dict.answer
			if ans.lower() in {'nil', 'none', ''}: ans = None
			res = answer_dict.rationale
			if res.lower() in {'nil', 'none', ''}: res = None

		# Save result to memory.
		self.memorize(llm_input, str_answer)
		# Return tools and help requests.
		return t_req, h_req, ans, res
	
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
	def memorize(self, user_query: str, assistant_response: str, verbose: bool = False):
		if user_query != '':
			self.memory.append({
				"role": "user",
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
	
	def manage_resource(self):
		self.clear_memory()
		self.cnt_iter = 0

	# Decision
	def continue_iteration(self, use_legacy=False):
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
				"Do you think you resolved the initial question? If you think so, answer 'Yes'. If not, answer 'No'. You must answer in either 'Yes' or 'No', without any other strings.",
				minimal=False,
			).lower()
		else:
			# NOTE: System-instructed structured processing.
			parser = PydanticOutputParser(
				pydantic_object=json_schema_to_base_model(process_binary.args_schema.model_json_schema())
			)
			_, assistant_decision_pydantic = self.__ask_LLM(
				"Do you think you resolved the initial question? If you think so, answer 'Yes'. If not, answer 'No'.",
				parser=parser,
			)
			assistant_decision = assistant_decision_pydantic.yes_or_no.lower()

		if 'no' in assistant_decision:
			self.cnt_iter += 1
			return True
		elif 'yes' in assistant_decision:
			return False
		else:
			self.manage_resource()
			raise ValueError(f"Agent behaved weird during `self.continue_iteration()`.\nResponse was: {assistant_decision}")

	def plan_subtask(self, init=True):
		if init:
			#llm_input = f"You are a domain expert. Based on the question I gave, plan the subtasks in order to answer the given question."
			llm_input = f"Plan the subtasks in order to answer the given question, be aware to use the tools or helpers if needed."
		else:
			llm_input = f"Evaluate the task progress so far. If needed, refine the sub-tasks to solve current problem. Be aware to use the tools or helpers if needed."
		assistant_plan = self.__ask_LLM(llm_input, ignore_answer_format=True)
		self.memorize(llm_input, assistant_plan)

	# For final answer.
	def finalize(self):
		llm_input = "Using the context, finalize your answer. For final answer, you must ignore **Answer Format** given earlier and just respond normally but concise."
		final_response = self.__ask_LLM(llm_input, minimal=False)
		self.memorize(llm_input, final_response)
		return final_response

	# For saving final memory to the file.
	def save_mem_to_file(self, file_path: str):
		with open(file_path, "w+", encoding='utf-8') as f:
			f.write(
				"\n".join([
					f"[{m['role']}]\n{m['content']}\n"
				for m in self.memory])
			)
		print(f"File '{file_path}' has been saved.")