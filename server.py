from fastapi import FastAPI
from pydantic import BaseModel
from urllib.parse import parse_qs, urlparse
from llama_cpp import Llama
from langchain.embeddings import LlamaCppEmbeddings

import os

app = FastAPI()

VERSION = "0.1"
N_CTX = 4096

if "MODEL_PATH" in os.environ:
    MODEL_PATH = os.environ["MODEL_PATH"]
else:
    MODEL_PATH = "./models/7B/ggml-model-q4_0.bin"
    if not os.path.exists(MODEL_PATH):
        print(f"Model path {MODEL_PATH} does not exist")
        exit(1)
try:
    # Load the model
    LLM = Llama(model_path=MODEL_PATH, n_ctx=N_CTX)
    EMBEDDINGS = LlamaCppEmbeddings(model_path=MODEL_PATH)
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

class Query(BaseModel):
    question: str

###########################
# Question
###########################
def _question(prompt):
    try:
        prompt = f"Q: {prompt}"
        output = LLM(prompt, max_tokens=0)
        result = output["choices"][0]["text"]
        return result
    except Exception as e:
        return f"Error: {e}"

###########################
# Embeddings
###########################
def _embeddings(prompt):
    try:
        string_embedding = EMBEDDINGS.embed_query(prompt)
        return string_embedding
    except Exception as e:
        return []

@app.get("/")
async def version():
    return {"version": VERSION, "model_path": MODEL_PATH}

@app.post("/query")
async def query(query: Query):
    result = _question(query.question)
    return {"result": result}

@app.post("/embeddings")
async def embeddings(query: Query):
    result = _embeddings(query.question)
    return {"result": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

