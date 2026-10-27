import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessageChunk, AIMessage, ToolMessage

from agent import app

web_app = FastAPI(title="LangGraph Technical Support Agent API")

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@web_app.get("/")
async def serve_index():
    return FileResponse("index.html")

async def generate_agent_stream(query_text: str):
    inputs = {"messages": [HumanMessage(content=query_text)]}
    
    # Use multi-mode stream to capture BOTH node state updates AND raw message tokens
    async for mode, payload in app.astream(inputs, stream_mode=["updates", "messages"]):
        
        # 1. Node State Updates -> Detect when tools are triggered
        if mode == "updates":
            for node_name, node_output in payload.items():
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_call['name']})}\n\n"

        # 2. Raw Stream Tokens -> Yield real-time streaming text tokens
        elif mode == "messages":
            chunk, metadata = payload
            if isinstance(chunk, (AIMessageChunk, AIMessage)) and chunk.content:
                # Exclude internal tool-call json metadata from token output
                if not getattr(chunk, "tool_call_chunks", None) and not getattr(chunk, "tool_calls", None):
                    content = chunk.content
                    if isinstance(content, str) and content:
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

    yield "data: [DONE]\n\n"

@web_app.post("/chat/stream")
async def chat_stream_endpoint(request: QueryRequest):
    return StreamingResponse(
        generate_agent_stream(request.query),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(web_app, host="127.0.0.1", port=8000)