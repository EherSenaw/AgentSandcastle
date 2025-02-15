python3 ./main.py \
--hf_api_token '{YOUR_HUGGINGFACE_INFERENCE_API_TOKEN}' \
--togetherai_api_token '{YOUR_TOGETHERAI_API_TOKEN}' \
--llm_provider 'togetherai' \
--llm_model_name 'meta-llama/Llama-Vision-Free' \
--llm_modality_io 'text,image/text' \
--initial_question 'I want to know about recent meme about "chill dog". If needed, explain with the image.'
#--llm_provider 'huggingface' \
#--llm_model_name 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free'
#--llm_model_name 'meta-llama/Llama-3.2-3B-Instruct'