# NOTE: Check import if-else statements.
#		They differ for the use of text-only models or
#		vision-language models.

MAX_TOKENS = 1024

# !!!!!!!!!!!!!!!!!!!!	
# ! TEXT ONLY MODELS !
# !!!!!!!!!!!!!!!!!!!!	
'''DeepSeek-R1'''
#MODEL_NAME = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B"
## --> Peak Memory: 3.576 GB / Generation: 15.578 TPS
#MODEL_NAME = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-3bit"
## --> Peak Memory: 0.803 GB / Generation: 29.294 TPS
#MODEL_NAME = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
## --> Peak Memory: 1.025 GB / Generation: 29.037 TPS
#MODEL_NAME = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-8bit"
## --> Peak Memory: 1.914 GB / Generation: 22.088 TPS
#MODEL_NAME = "mlx-community/DeepSeek-R1-2bit" #<-- Too big.
## --> Peak Memory: . GB / Generation: . TPS
'''DeepScaleR'''
MODEL_NAME = "mlx-community/DeepScaleR-1.5B-Preview-4bit"
# --> Peak Memory: 1.025 GB / Generation: 27.018 TPS
'''EXAONE-3.5'''
#MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
## --> Peak Memory: 9.706 GB / Generation: 5.053 TPS
#MODEL_NAME = "mlx-community/EXAONE-3.5-2.4B-Instruct-4bit"
## --> Peak Memory: 1.401 GB / Generation: 20.692 TPS
'''Llama-3.2 (text-only)'''
#MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"
## --> Peak Memory: 0.737 GB / Generation: 37.338 TPS
#MODEL_NAME = "mlx-community/Llama-3.2-3B-Instruct-4bit"
## --> Peak Memory: 1.875 GB / Generation: 17.045 TPS

# !!!!!!!!!!!!!!!!!!!!	
# !   VISION MODELS  !
# !!!!!!!!!!!!!!!!!!!!	
'''Llama-3.2 (vision)'''
#MODEL_NAME = "mlx-community/Llama-3.2-11B-Vision-Instruct-abliterated-4-bit"
## --> Peak Memory: 16.317 GB / Generation: 0.695 TPS
'''Qwen2.5-VL (vision, requires re-installation of `transformers` and `accelerate` libraries.)'''
''' USE: `pip install --upgrade git+https://github.com/huggingface/transformers accelerate`'''
# NOTE: Currently not compatible with mlx-vlm.
#MODEL_NAME = "mlx-community/Qwen2.5-VL-3B-Instruct-3bit"
### --> Peak Memory: . GB / Generation: . TPS
'''Deepseek-VL2'''
#MODEL_NAME = "mlx-community/deepseek-vl2-tiny-3bit"
## --> Peak Memory: . GB / Generation: . TPS


'''TEMPLATE
MODEL_NAME = "mlx-community/"
## --> Peak Memory: . GB / Generation: . TPS
'''

#USE_VLM = True if 'vision' in MODEL_NAME.lower() else False
USE_VLM = False
if USE_VLM:
	from mlx_vlm import load, generate
	from mlx_vlm.prompt_utils import apply_chat_template
	from mlx_vlm.utils import load_config
else:
	from mlx_lm import load, generate

#FLAG_TRUST_REMOTE = True

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
		"content": "Explain about the given image(s). Then, score relevance of each image with the following context in range of [0,5] with reasons.\n\n**Context**\nBlue sky is truly high in the essence of the earth." if USE_VLM else "Write an essay about the importance of higher education."
	},
]

if USE_VLM:
	model, processor = load(MODEL_NAME)
	config = load_config(MODEL_NAME)

	image = [
		# BLUE SKY
		'https://images.pexels.com/photos/281260/pexels-photo-281260.jpeg?auto=compress&cs=tinysrgb&w=1200',
		# RED TREE
		'https://media.istockphoto.com/id/1069582118/photo/easy-way-in-the-autumn-park.jpg?s=612x612&w=0&k=20&c=IDYMLx4XkiXyh9lBr7RfmPeHBpUlpJL2shWGLdN1nNE='
	] # URL or path here.
	try:
		formatted_prompt = apply_chat_template(
			processor, config, conversation, num_images=len(image)
		)
	except ValueError as e:
		print(f"\n**ERROR**\nGot ValueError by trying to use Multi-image chat with the model not supported. Fall back to use first image only.\n**ERROR**\n")
		image = [image[0]]
		formatted_prompt = apply_chat_template(
			processor, config, conversation, num_images=len(image)
		)
	output = generate(model, processor, formatted_prompt, image, verbose=True)
	#print(output)
else:
	model, tokenizer = load(MODEL_NAME)
	prompt = tokenizer.apply_chat_template(
		conversation, add_generation_prompt=True
	)
	text = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=MAX_TOKENS)
	#print(text)

#print_outputs(outputs)
