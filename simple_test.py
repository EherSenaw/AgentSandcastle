import torch
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

#MODEL_NAME = "facebook/opt-125m"
#MODEL_NAME = "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"
#MODEL_NAME = "deepseek-ai/DeepSeek-R1"
#MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
#MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
#MODEL_NAME = "agentica-org/DeepScaleR-1.5B-Preview"
#MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit"
MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
FLAG_TRUST_REMOTE = True

def print_outputs(outputs):
	for output in outputs:
		prompt = output.prompt
		generated_text = output.outputs[0].text
		print(f"Prompt: {prompt}\n\nGenerated text: {generated_text}")
	print("-"*80)

print("="*80)

conversation = [
	{
		"role": "system",
		"content": "You are a helpful assistant"
	},
	{
		"role": "user",
		"content": "hello"
	},
	{
		"role": "assistant",
		"content": "Hello! How can I assist you today?"
	},
	{
		"role": "user",
		"content": "Write an essay about the importance of higher education."
	},
]
conversations = [conversation for _ in range(3)]

llm = LLM(
	model=MODEL_NAME,
	#dtype=torch.bfloat16,
	dtype="half",
	trust_remote_code=FLAG_TRUST_REMOTE,
	#enable_chunked_prefill=True, --> Bfloat16
)
#quantization="bitsandbytes",
#load_format="bitsandbytes",
#llm.apply_model(lambda model: print(model.__class__))
sampling_params = SamplingParams(temperature=0.5)

#outputs = llm.generate(prompts, sampling_params)
outputs = llm.chat(
	conversation,
	sampling_params=sampling_params,
	use_tqdm=False,
)
print_outputs(outputs)

'''
outputs = llm.chat(
	messages=conversations,
	sampling_params=sampling_params,
	use_tqdm=True,
	#chat_template=chat_template,
)
print_outputs(outputs)
'''

del llm
cleanup_dist_env_and_memory()