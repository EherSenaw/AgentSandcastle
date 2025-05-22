from sse_starlette.sse import EventSourceResponse

from fastapi import FastAPI, Body, Header, Response, Request, File, UploadFile, Form # Added File, UploadFile, Form
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# from pydantic import BaseModel # Message model will be removed
from contextlib import asynccontextmanager
from typing import Optional, Any, AsyncGenerator, Callable, TypeVar, List # Added List
from pathlib import Path

import asyncio
import functools
import gc
import json
import yaml
import shutil # Added
import tempfile # Added
import os # Added


# LOCAL
from external_use import build_engine

STATIC_DIR = Path(__file__).parent / "static"

# NOTE: @app.on_event(startup) is deprecated. Therefore, use lifespan.
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
	# Setup
	load_engine_once()
	print("Engine initialized.")

	try:
		yield # Wait for APP shutdown.
	finally:
		# Teardown
		print("Shutting down LLM engine..")
		delete_engine()
		gc.collect()

app = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def no_cache_static(request: Request, call_next):
	response = await call_next(request)
	if request.url.path.startswith("/static/"):
		response.headers["Cache-Control"] = "no-store, max-age=0"
	return response

# Removed Message model as inputs will come from Form/File
# class Message(BaseModel):
#	user_input: str

# NOTE: For further consideration on using AWAIT or ASYNC, check the link: [ https://fastapi.tiangolo.com/async/#in-a-hurry ].

# Initialize the engine once on app startup.
#@app.on_event("startup") <-- Deprecated. Use lifespan instead.
def load_engine_once():
	model_config = None
	with open(STATIC_DIR / "model_config.yaml", "r") as model_config_file:
		model_config = yaml.safe_load(model_config_file)
	if 'model_args' not in model_config:
		raise ValueError("Check model config.")
	# NOTE: Engine (e.g. MLXEngine, vLLMEngine) needs to be updated
	# to accept image_paths in its __call__ and stream_call methods.
	app.state.llm_engine = build_engine(model_config['model_args'])
def delete_engine():
	try:
		app.state.llm_engine.shutdown()
	finally:
		del app.state.llm_engine
		app.state.llm_engine = None

# NOTE: The synchronous /chat endpoint is NOT updated to handle image uploads in this step.
# If image support is needed for /chat, similar Form and File parameters,
# image saving, and passing image_paths to the engine's __call__ method would be required.
@app.post("/chat")
async def chat(user_input_from_form: str = Form(...)): # Changed to Form for consistency, but not handling images yet
	user_text = user_input_from_form
	# Call your local LLM Agent.
	# This would need to be:
	# agent_reply = await asyncio.to_thread(app.state.llm_engine, user_text, volatile=False, image_paths=None)
	# if app.state.llm_engine.__call__ is not async.
	agent_reply = app.state.llm_engine(user_text, volatile=False) # Assuming engine.__call__ is sync for now
	return JSONResponse(content={"response": agent_reply})

@app.post("/chat-stream")
async def chat_async(user_input: str = Form(""), image_file: Optional[UploadFile] = File(None)):
	image_paths_for_engine: Optional[List[str]] = None
	temp_file_paths: List[str] = []

	if image_file and image_file.filename:
		try:
			temp_dir = "temp_images"
			os.makedirs(temp_dir, exist_ok=True)
			
			# Sanitize filename (basic example)
			safe_filename = os.path.basename(image_file.filename)
			if not safe_filename: # Handle empty or malicious filenames
				raise ValueError("Invalid filename")

			temp_file_path = os.path.join(temp_dir, safe_filename)
			
			with open(temp_file_path, "wb") as buffer:
				shutil.copyfileobj(image_file.file, buffer)
			
			image_paths_for_engine = [temp_file_path]
			temp_file_paths.append(temp_file_path)
			print(f"Image saved to temporary path: {temp_file_path}")

		except Exception as e:
			print(f"Error saving uploaded image: {e}")
			# Optionally, could yield an error message to the user here if the stream format supports it
		finally:
			if image_file:
				image_file.file.close()

	engine = app.state.llm_engine
	
	# The closure token_stream_generator needs access to relevant variables
	async def token_stream_generator(current_user_input: str, current_image_paths: Optional[List[str]], files_to_clean: List[str]):
		try:
			# NOTE: engine.stream_call needs to be updated to accept image_paths
			async for response_tuple in engine.stream_call(current_user_input, volatile=False, image_paths=current_image_paths):
				(msg_type, partial_response), is_final_answer = response_tuple # Assuming this structure from @with_final
				
				event_data = {"type": msg_type, "content": partial_response} # Changed "data" to "content" to match script.ts expectations
				
				# Yield based on the type, matching the TypeScript frontend's expectations
				if msg_type == "thought_intermediate" or msg_type == "thought_stream":
					yield f"data: {json.dumps(event_data)}\n\n"
				elif msg_type == "action_stream" or msg_type == "tool_call_stream":
					yield f"data: {json.dumps(event_data)}\n\n"
				elif msg_type == "observation_stream":
					yield f"data: {json.dumps(event_data)}\n\n"
				elif msg_type == "final_answer_stream":
					yield f"data: {json.dumps(event_data)}\n\n"
				elif msg_type == "status_stream": # Handling status messages
					yield f"data: {json.dumps(event_data)}\n\n"
				# Add other types if necessary based on what engine.stream_call yields
				# Example: if engine.stream_call directly yields the old format:
				# elif msg_type.startswith('thought'):
				# 	if msg_type.endswith('intermediate'):
				# 		yield f"data: {json.dumps({'type': 'thought_intermediate', 'content': partial_response})}\n\n"
				# 	else: # thought_answer
				# 		yield f"data: {json.dumps({'type': 'thought_stream', 'content': partial_response})}\n\n"
				# elif msg_type.startswith('final'):
				# 	if msg_type.endswith('intermediate') or not is_final_answer: # Assuming is_final_answer means it's the very end
				# 		yield f"data: {json.dumps({'type': 'final_intermediate', 'content': partial_response})}\n\n"
				# 	else: # final_answer
				# 		yield f"data: {json.dumps({'type': 'final_answer_stream', 'content': partial_response})}\n\n"
				
				await asyncio.sleep(0.01) # Small sleep to allow other tasks, adjust as needed
		finally:
			for path in files_to_clean:
				try:
					os.remove(path)
					print(f"Cleaned up temp file: {path}")
				except Exception as e:
					print(f"Error cleaning up temp file {path}: {e}")
			# Attempt to remove the temp_images directory if it's empty
			try:
				if os.path.exists("temp_images") and not os.listdir("temp_images"):
					os.rmdir("temp_images")
					print("Cleaned up temp_images directory.")
			except Exception as e:
				print(f"Error cleaning up temp_images directory: {e}")

	return StreamingResponse(token_stream_generator(user_input, image_paths_for_engine, list(temp_file_paths)), media_type="text/event-stream") # Changed media_type for SSE

@app.get("/", response_class=HTMLResponse)
async def get():
	with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
		return f.read()

app.mount(
	"/static",
	StaticFiles(directory=STATIC_DIR),
	name="static"
)