import time, re, os
from typing import List, Optional, Dict

from src.tools import verify_hf_tools, verify_tools_docstring
from src.prompt_template import (
	SYSTEM_PROMPT_TEMPLATE,
	#PROMPT_TEMPLATE,
	SYSTEM_ANSWER_TEMPLATE,
)
from src.structured_output import PydanticOutputParser
from src.tool_convert import tool

from pydantic import BaseModel, Field

from transformers.utils import get_json_schema
from smolagents import Tool #,tool
from smolagents.utils import make_json_serializable

from mlx_lm import load, generate

from mlx_vlm import (
	load as vlm_load,
	generate as vlm_generate,
)
from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template
from mlx_vlm.utils import load_config as vlm_load_config

# NOTE: DEBUG for structured_output parsing from LLm output and integrating it with tool calling.
class Person(BaseModel):
	"""Information about a person."""

	name: str = Field(..., description="The name of the person")

class People(BaseModel):
	"""Identifying information about all people in a text."""

	people: List[Person]

parser = PydanticOutputParser(pydantic_object=People)
parse_chain_template = [
	{
		"role": "system",
		"content": "Answer the user query. Wrap the output in `json` tags\n{format_instructions}".format(
			format_instructions=parser.get_format_instructions()
		),
	},
	{
		"role": "user",
		"content": "{query}",
	},
]


class MLXEngine():
	def __init__(
		self,
		model : str = 'mlx-community/DeepScaleR-1.5B-Preview-4bit',
		max_iteration : int = 5,
		verbose : bool = False,
		tools : List[Optional[Tool]] = [],
		modality_io : str = 'text/text',
		max_new_tokens : int = 1024,
	):
		self.model = model
		self.max_iter = max_iteration
		self.verbose = verbose

		#self.tools = verify_hf_tools(tools)
		#self.tools = verify_tools_docstring(tools)
		self.tools = dict()
		for t in tools:
			'''
			# NOTE: Currently using @tool from smolagents,
			# 		but maybe simply replaced with Google-style docstring parser.
			if hasattr(t, 'name'):
				t_name = t.name
			elif hasattr(t, '__name__'):
				t_name = t.__name__
			self.tools[t_name] = t
			'''
			t_schema = t.args_schema.model_json_schema()
			t_name = t_schema['title'][:-6] # postfix removal by -6 ('Schema')
			#t_description = t_schema['description']
			assert t_name not in self.tools, "The name(`title`) of the tool should not be duplicated."
			self.tools[t_name] = t

		self.memory = []
		self.cnt_iter = 0

		self.modality_in, self.modality_out = modality_io.split('/')
		self.modality_in = list(self.modality_in.split(','))
		self.modality_out = list(self.modality_out.split(','))

		self.max_new_tokens = max_new_tokens
		# Check VLM
		self.USE_VLM = True if len(self.modality_in) > 1 else False
		if self.USE_VLM:
			self.client, self.processor = vlm_load(self.model)
			self.config = vlm_load_config(self.model)
		else:
			self.client, self.tokenizer = load(self.model)

		self.validation_prompt = SYSTEM_PROMPT_TEMPLATE
		self.validation_answer_format = SYSTEM_ANSWER_TEMPLATE

		self.__init_memory(self.verbose)

		self.answer_dict_regexp = re.compile(r"\{[^\{\}]*\}")
		self.think_regexp = re.compile(r"<think>.*<\/think>", re.DOTALL)

	def __init_memory(self, verbose:bool=False):
		self.memory.append({
			"role": "system",
			#"role": "user",
			"content": self.validation_prompt.format(
				foo='foo', bar='bar',
				# Refined `tool_list` with Google DocString.
				max_iteration=self.max_iter,
				modality_in=self.modality_in,
				modality_out=self.modality_out,
				#tool_list='\n'.join(f"- {t_name}: {tool.description}\n\tArgs: {tool.inputs}\n\tReturns: {tool.output_type}" for t_name,tool in self.tools.items()),
				tool_list='\n'.join(f"- {t_name}: {tool.description}" for t_name,tool in self.tools.items()),
				max_new_tokens=self.max_new_tokens,
		)})
		#)+'\n'+self.validation_answer_format})
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
			#llm_input_observation = f"Do: {action}"
			observation = self.__ask_LLM(llm_input_observation)

			# Store to memory.
			#self.memorize(llm_input_action, action)
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
	):
		if image_urls is None:
			image_urls = []

		# NOTE: VLM not tested since beginning of tool execution implementation.
		if self.USE_VLM:
			try:
				formatted_prompt = vlm_apply_chat_template(
					self.processor, self.config, self.memory+[{
						"role": "user",
						"content": question,
					}], num_images=len(image_urls) 
				)
			except ValueError as e:
				print(f"\n**ERROR**\nGot ValueError by trying to use Multi-image chat with the model not supported. Fall back to use first image only.\nOriginal error message -> {e}\n**ERROR**\n")
				image_urls = [image_urls[0]]
				formatted_prompt = vlm_apply_chat_template(
					self.processor, self.config, self.memory+[{
						"role": "user",
						"content": question,
					}], num_images=len(image_urls) 
				)
			output = vlm_generate(self.client, self.processor, formatted_prompt, image_urls, verbose=self.verbose)
		else:
			if ignore_answer_format:
				prompt = self.tokenizer.apply_chat_template(
					self.memory[0:1] + \
					self.memory[2:] if len(self.memory)>2 else [] + \
					[{
						"role": "user",
						"content": question,
					}],
					add_generation_prompt=True
				)
			else:
				prompt = self.tokenizer.apply_chat_template(
					self.memory + \
					[{
						"role": "user",
						"content": question,
					}],
					add_generation_prompt=True
				)
			'''
			print(f"[DEBUG] PROMPT: ")
			for m in self.memory+[{"role":"user", "content": question}]:
				print(f"[DEBUG]\t{m['role']}: {m['content']}")
			'''
			output = generate(self.client, self.tokenizer, prompt=prompt, verbose=self.verbose, max_tokens=self.max_new_tokens)
			output = self.__retrieve_non_think(output.strip(), minimal=minimal)
		return output

	def __retrieve_answer_dict(self, str_answer:str) -> Dict:
		ans_dict_list = []
		for candidate in self.answer_dict_regexp.finditer(str_answer.strip()):
			try:
				c = eval(candidate.group())
				assert isinstance(c, dict), "{candidate} is not a dict."
				ans_dict_list.append(c)
			except :
				continue
		if len(ans_dict_list) < 1:
			return dict()
		return ans_dict_list[-1]
	def __retrieve_non_think(self, str_response:str, minimal:bool=False) -> str:
		# NOTE: Before calling this, using .strip() is preferred.
		if '<think>' not in str_response:
			return str_response
		retval = ''
		n = len(str_response)
		for candidate in self.think_regexp.finditer(str_response):
			s, e = candidate.span()
			if s > 0:
				left = str_response[:s]
			else:
				left = ''
			if e < n:
				right = str_response[e:]
				# Normally, answer comes after the <think>....</think>.
				# Just use 'right' part.
				if minimal:
					# NOTE: Minimal-mode --> just remove <think> and </think> tags.
					retval = left + str_response[s+7:e-8] + right
				else:
					retval = right
				break
		return retval.strip()
	

	def check_request(self, question='', init=False):
		#llm_input = question + '\n' + self.validation_answer_format
		llm_input = question
		if init:
			llm_input = f"**QUESTION**\n{question}"
		str_answer = self.__ask_LLM(llm_input, None)
		answer_dict = self.__retrieve_answer_dict(str_answer)

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
			print(f"\nTrying to parse the tool request from: {t_req}")
			parsed_req = self.__ask_LLM(f'Get the name of the tool that corresponds to the following request:`{t_req}`. You should lookup for the available tools you have. Return the name of the tool only (without any special characters or any other words)')
			print('\n'+parsed_req+'\n')
			t_req = parsed_req.lower()
		# Default
		if t_req not in self.tools:
			err_msg = f"Corresponding tool for the request of '{t_req}' does not exist."
			print('\n'+err_msg+'\n')
			self.memorize('', f"[Tool calling error report]\n{err_msg}\n[Tool calling error report end]\n")
			return
		
		retrieve_prompt = f'Return the dictionary of inputs for the tool \'{t_req}\'. The tool spec is following:\n**Description**\n{self.tools[t_req].description}\n**Args**\n{self.tools[t_req].inputs}\n**Returns**\n{self.tools[t_req].output_type}'
		print(f"\nretrieve_prompt: {retrieve_prompt}\n")
		t_arg_str = self.__ask_LLM(retrieve_prompt, ignore_answer_format=True)
		print(f"\nt_arg_str: {t_arg_str}\n")
		t_arg = self.__retrieve_answer_dict(t_arg_str)
		if "Tool_Request" in t_arg:
			del t_arg["Tool_Request"]
		if "Helper_Request" in t_arg:
			del t_arg["Helper_Request"]
		print(f"\nt_arg: {t_arg}\n")

		#tool_result = self.tools[t_req](**t_arg)
		tool_result = self.tools[t_req](**{k:v for k,v in t_arg.items() if k in self.tools[t_req].inputs})
		print(f"[DEBUG] tool_result: {tool_result}")
		# Record tool results for furthur use.
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
	def continue_iteration(self):
		if len(self.memory) <= 1:
			# Only system prompt.
			#self.cnt_iter += 1
			return True
		if self.max_iter <= self.cnt_iter:
			return False
		
		# Ask if the answer was given for the question.
		# If no, we should continue iteration.
		# If yes, we should stop iteration.
		assistant_decision = self.__ask_LLM(
			"Do you think you resolved the initial question? If you think so, answer 'Yes'. If not, answer 'No'. You must answer in either 'Yes' or 'No', without any other strings.",
			minimal=False,
		).lower()

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
			llm_input = f"You are a domain expert. Based on the question I gave, plan the subtasks in order to answer the given question."
		else:
			llm_input = f"Evaluate the task progress so far. If needed, refine the sub-tasks to solve current problem."
		assistant_plan = self.__ask_LLM(llm_input)
		self.memorize(llm_input, assistant_plan)

	# For final answer.
	def finalize(self):
		llm_input = "Using the context, finalize your answer. For final answer, you must ignore **Answer Format** given earlier and just respond normally but concise."
		final_response = self.__ask_LLM(llm_input, minimal=False)
		self.memorize(llm_input, final_response)
		return final_response

	# For saving final memory to the file.
	def save_mem_to_file(self, file_path: str):
		from src.tools import save_file
		save_file(
			file_path,
			"\n".join([
				f"[{m['role']}]\n{m['content']}\n"
			for m in self.memory])
		)