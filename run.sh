python3 -m main \
--llm_provider 'mlx' \
--llm_model_name 'mlx-community/Qwen2.5-7B-Instruct-1M-4bit' \
--llm_modality_io 'text/text' \
--max_new_tokens 1024 \
--manual_answer_format '' \
--verbose \
--initial_question 'I want to know about recent meme about "chill dog".'
#--initial_question 'I want to know about recent meme about "chill dog". If needed, explain with the image.'
#--hf_api_token '{YOUR_HUGGINGFACE_INFERENCE_API_TOKEN}' \
#
#--llm_model_name 'mlx-community/DeepScaleR-1.5B-Preview-4bit' \
#
#--togetherai_api_token '{YOUR_TOGETHERAI_API_TOKEN}' \
#--llm_provider 'togetherai' \
#--llm_model_name 'meta-llama/Llama-Vision-Free' \
#--llm_modality_io 'text,image/text' \
#
#--llm_provider 'huggingface' \
#--llm_model_name 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free'
#--llm_model_name 'meta-llama/Llama-3.2-3B-Instruct'