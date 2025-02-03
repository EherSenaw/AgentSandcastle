import time
from together import Together

class TogetherAPIEngine():
	def __init__(
		self,
		model='meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
		max_iteration=5,
		verbose=False,
		free_tier=False,
	):
		self.client = Together()
		self.model = model
		self.max_iter = max_iteration
		self.verbose = verbose

		self.memory = []
		self.cnt_iter = 0

		self.free_tier = free_tier

	def __call__(self, question=''):
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
		self.memory.extend([
			{
				"role": "user",
				"content": f"{user_query}",
			},
			{
				"role": "assistant",
				"content": f"{assistant_response}",
			},
		])

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
		llm_input = "Using the context, finalize your answer."
		final_response = self.__ask_LLM(llm_input)
		self.memorize(llm_input, final_response)
		return final_response
	



