from fastapi import FastAPI, Body, Header, Response, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional, Any, AsyncGenerator
from pathlib import Path

import gc
import yaml
import orjson


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

# NOTE: Async + StreamingResponse example.
async def fake_videostreamer():
	for i in range(10):
		yield b"Some fake video bytes"
# NOTE: 'open()' does not support async/await.
def fake_filestreamer():
	FILE_PATH = "some_file_path.txt"
	with open(FILE_PATH, mode="rb") as file_like:
		yield from file_like

class Message(BaseModel):
	user_input: str

# NOTE: Custom response example.
class CustomORJSONResponse(Response):
	media_type = "application/json"
	# NOTE: Should implement render()->bytes method.
	def render(self, content: Any) -> bytes:
		assert orjson is not None, "orjson must be installed."
		return orjson.dumps(content, option=orjson.OPT_INDENT_2)

class Item(BaseModel):
	name: str
	price: float
	is_offer: Optional[bool] = None


# NOTE: For further consideration on using AWAIT or ASYNC, check the link: [ https://fastapi.tiangolo.com/async/#in-a-hurry ].
'''
@app.get("/")
def read_root():
	return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
	return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
	return {"item_name": item.name, "item_id": item_id}

@app.get("/video/{video_id}")
async def read_video():
	return StreamingResponse(fake_videostreamer())
@app.get("/file/{file_id}")
def read_file():
	return StreamingResponse(fake_filestreamer(), media_type="video/mp4")

@app.get("/custom/", response_class=CustomORJSONResponse)
async def custom_resp():
	return {"message": "Hello world"}
'''

# Initialize the engine once on app startup.
#@app.on_event("startup") <-- Deprecated. Use lifespan instead.
def load_engine_once():
	model_config = None
	with open(STATIC_DIR / "model_config.yaml", "r") as model_config_file:
		model_config = yaml.safe_load(model_config_file)
	if 'model_args' not in model_config:
		raise ValueError("Check model config.")
	app.state.llm_engine = build_engine(model_config['model_args'])
def delete_engine():
	try:
		app.state.llm_engine.shutdown()
	finally:
		del app.state.llm_engine
		app.state.llm_engine = None

@app.post("/chat")
async def chat(message: Message):
	user_text = message.user_input
	# Call your local LLM Agent.
	agent_reply = app.state.llm_engine(user_text, volatile=False)
	return JSONResponse(content={"response": agent_reply})
@app.get("/", response_class=HTMLResponse)
async def get():
	with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
		return f.read()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")