from fastapi import FastAPI, Body, Header, Response, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional, Any, AsyncGenerator
from pathlib import Path

import gc
import yaml


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

class Message(BaseModel):
	user_input: str

# NOTE: For further consideration on using AWAIT or ASYNC, check the link: [ https://fastapi.tiangolo.com/async/#in-a-hurry ].

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