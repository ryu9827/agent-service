import os
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Agent Service")

# 内存任务存储
tasks = {}

class TaskStatus(str):
    RECEIVED = "received"
    PROCESSING_LLM = "processing_llm"
    COMPLETED = "completed"
    FAILED = "failed"

class Task(BaseModel):
    id: str
    status: str
    result: str = None
    error: str = None

class ChatRequest(BaseModel):
    message: str

class TaskResponse(BaseModel):
    task_id: str

# 请将 IP 替换为您 Windows PC 的实际 Tailscale IP
LLM_API_URL = os.getenv("LLM_API_URL", "http://100.74.179.79:11434/api/generate")

@app.post("/chat", response_model=TaskResponse)
async def start_chat(request: ChatRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = Task(id=task_id, status=TaskStatus.RECEIVED)
    background_tasks.add_task(process_llm, task_id, request.message)
    return TaskResponse(task_id=task_id)

async def process_llm(task_id: str, message: str):
    tasks[task_id].status = TaskStatus.PROCESSING_LLM
    try:
        async with httpx.AsyncClient() as client:
            payload = {
                "model": "qwen2.5:7b",
                "prompt": message,
                "stream": False
            }
            response = await client.post(LLM_API_URL, json=payload, timeout=60.0)
            response.raise_for_status()
            result = response.json()
            tasks[task_id].status = TaskStatus.COMPLETED
            tasks[task_id].result = result["response"]
    except Exception as e:
        tasks[task_id].status = TaskStatus.FAILED
        tasks[task_id].error = str(e)

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/health")
async def health():
    return {"status": "ok"}