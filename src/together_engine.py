import time
import re
import os
from typing import List, Optional, Dict

from together import Together

# NOTE: For PoC, just use HuggingFace(smolagent)'s Tool system.
from src.tools import verify_hf_tools
from transformers.utils import (
	get_json_schema
)
from smolagents import (
	tool, Tool
)

class TogetherAPIEngine():
	def __init__(
		self,
		model : str = 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
		max_iteration : int = 5,
		verbose : bool = False,
		free_tier : bool = False,
		tools : List[Optional[Tool]] = [],
	):
		self.client = Together()
		self.model = model
		self.max_iter = max_iteration
		self.verbose = verbose

		# TODO: Implement tools and map the name and the implementation.
		self.tools = verify_hf_tools(tools)

		self.memory = []
		self.cnt_iter = 0

		self.free_tier = free_tier

		self.validation_prompt = """
		**GENERAL INSTRUCTIONS**
		Your task is to answer questions. If you cannot answer the question, request a helper or use a tool. Fill with Nil where no tool or helper is required.
		
		**AVAILABLE TOOLS**
		{tool_list}
		
		**AVAILABLE HELPERS**
		- Decomposition: Breaks Complex Questions down into simpler subparts
		
		**CONTEXTUAL INFORMATION**
		{context}
		
		**QUESTION**
		{question}
		"""

		self.validation_answer_format = """
		**ANSWER FORMAT**
		{'Tool_Request': '<Fill>', 'Helper_Request': '<Fill>'}
		"""

		self.answer_dict_regexp = re.compile(r"\{[^\{\}]*\}")

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

	def check_request(self, question=''):
		#str_tools = "\n".join(list(self.tools.keys()))
		str_tools = "\n".join(
			map(
				lambda t_name: str(get_json_schema(self.tools[t_name])) if hasattr(self.tools[t_name], '__name__') else t_name,
				self.tools.keys()
			)
		)
		llm_input = self.validation_prompt.format(
			# TODO: Refine current `tool_list` which just listed the tools
			#		To the explanation of each tools.
			tool_list=str_tools,
			context="\n".join([
				f"[{m['role']}]\n{m['content']}\n"
			for m in self.memory]),
			question=question,
		) + '\n' + self.validation_answer_format
		str_answer = self.__ask_LLM(llm_input)

		answer_dict = self.__retrieve_answer_dict(str_answer)

		t_req, h_req = None, None
		if "Tool_Request" in answer_dict:
			t_req = answer_dict["Tool_Request"]
			if t_req == "Nil":
				t_req = None
		if "Helper_Request" in answer_dict:
			h_req = answer_dict["Helper_Request"]
			if h_req == "Nil":
				h_req = None
		
		# Save result to memory.
		self.memorize(llm_input, str_answer)
		# Return tools and help requests.
		return t_req, h_req
	
	def use_tool(self, t_req: str):
		# Do pattern matching for tool calling request.
		# if there is a match, use that tool.
		t_req = t_req.lower()
		if t_req not in self.tools:
			err_msg = f"Corresponding tool for the request of '{t_req}' does not exist."
			print(err_msg)
			self.memorize('', f"[Tool calling error report]\n{err_msg}\n[Tool calling error report end]\n")
		#return self.tools[t_req](t_arg)
		
		t_arg_str = self.__ask_LLM(f'Using the context, generate the python dictionary of arguments to use the tool named \'{t_req}\'.')
		t_arg = self.__retrieve_answer_dict(t_arg_str)
		if 'web_search' in t_req and 'query' in t_arg:
			tool_result = self.tools[t_req]({'query':t_arg['query']})
		else:
			tool_result = self.tools[t_req](**t_arg)
		#print(f"[DEBUG] tool_result: {tool_result}")
		# Record tool results for furthur use.
		if tool_result:
			self.memorize('', f'[Observation of Tool {t_req}]\nWith argument: `{t_arg}`\n'+tool_result+'\n[Observation end]\n')

	def __call__(self, question=''):
		# Check if any TOOLs or HELPs needed.
		t_req, h_req = self.check_request(question)

		# If tool is needed, use tool.
		if t_req:
			self.use_tool(t_req)

		# If helper is needed, use helper.
		# NOTE: Maybe just use plan_subtask()?
		if h_req:
			# Initialize iterations for the given quesiton.
			self.plan_subtask(question)

		while self.continue_iteration():
			if self.verbose:
				print(f"Continue iteration.. ({self.cnt_iter} / {self.max_iter})")

			# Decide action.
			llm_input_action = "Based on the context, decide the next action for you to solve the problem."
			action = self.__ask_LLM(llm_input_action)

			# Do action and retrieve observation.
			llm_input_observation = f"Do: {action}"
			observation = self.__ask_LLM(llm_input_observation)

			# Store to memory.
			self.memorize(llm_input_action, action)
			self.memorize(llm_input_observation, observation)

		# Get final answer.
		final_answer = self.finalize()

		# TODO: Save the memory to the text file by LLM.
		self.save_mem_to_file(
			os.path.join(os.getcwd(), 'temp.log')
		)
		# Clean up resources.
		self.manage_resource()

		return final_answer

	# Reuse repetitive pattern
	def __ask_LLM(self, question=''):
		response = self.client.chat.completions.create(
			model = self.model,
			messages = self.memory + [{
				"role": "user",
				"content": question,
			}]
		).choices[0].message.content

		if self.verbose:
			print(f"[User] {question}")
			print(f"[Assistant] {response}")

		if self.free_tier:
			time.sleep(1)

		return response

	# Memory
	def memorize(
		self,
		user_query: str,
		assistant_response: str,
	):
		if user_query != '':
			self.memory.append({
				"role": "user",
				"content": f"{user_query}",
			})
		if assistant_response != '':
			self.memory.append({
				"role": "assistant",
				"content": f"{assistant_response}",
			})

	def clear_memory(self):
		del self.memory
		self.memory = []
	
	def manage_resource(self):
		self.clear_memory()
		self.cnt_iter = 0

	
	# Decision
	def continue_iteration(self):
		if len(self.memory) == 0:
			self.cnt_iter += 1
			return True
		if self.max_iter <= self.cnt_iter:
			return False
		
		# Ask if the answer was given for the question.
		# If no, we should continue iteration.
		# If yes, we should stop iteration.
		assistant_decision = self.__ask_LLM(
			"Do you think you resolved the initial question? If you think so, answer 'Yes'. If not, answer 'No'. You must answer in either 'Yes' or 'No', without any other strings."
		).lower()

		if 'no' in assistant_decision:
			self.cnt_iter += 1
			return True
		elif 'yes' in assistant_decision:
			return False
		else:
			self.manage_resource()
			raise ValueError(f"Agent behaved weird during `self.continue_iteration()`.\nResponse was: {assistant_decision}")

	def plan_subtask(self, question=''):
		llm_input = f"You are a domain expert. Based on the question I give you, plan the subtasks in order to answer the given question.\n**Question**: {question}"
		assistant_plan = self.__ask_LLM(llm_input)
		self.memorize(llm_input, assistant_plan)

	# For final answer.
	def finalize(self):
		llm_input = "Using the context, finalize your answer. For final answer, you must ignore **Answer Format** given earlier and just respond normally but concise."
		final_response = self.__ask_LLM(llm_input)
		self.memorize(llm_input, final_response)
		return final_response
	
	# For saving final memory to the file.
	# TODO: Convert manual saving process to LLM-controlled saving.
	def save_mem_to_file(self, file_path: str):
		from src.tools import save_file
		save_file(
			file_path,
			"\n".join([
				f"[{m['role']}]\n{m['content']}\n"
			for m in self.memory])
		)

