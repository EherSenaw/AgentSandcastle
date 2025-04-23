from fastapi import FastAPI, Body, Header

app = FastAPI()

# NOTE: FastAPI gets Query variable from such as "http://...?who=..."
@app.get("hi/{who}")
def greet(who: str):
	# FastAPI returns as JSON.
	return f"Hello? {who}!"

@app.get("hi/{who}")
def greet_Body(who: str = Body(embed=True)):
	# FastAPI returns as JSON.
	return f"Hello? {who}!"

@app.get("hi/{who}")
def greet_Header(who: str = Header()):
	# FastAPI returns as JSON.
	return f"Hello? {who}!"

