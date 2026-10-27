import os
import sys
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import rag_search_tool, check_server_status_tool

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("[ERROR]: GROQ_API_KEY environment variable is missing!")
    sys.exit(1)

tools = [rag_search_tool, check_server_status_tool]
tool_node = ToolNode(tools)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key,
    request_timeout=30,
)

llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    system_prompt = SystemMessage(
        content="You are a Technical Support AI Agent. Always use available tools to check documentation "
                "or server statuses first. Synthesize tool results clearly into a helpful final answer."
    )
    full_messages = [system_prompt] + list(state["messages"])
    response = llm_with_tools.invoke(full_messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

app = workflow.compile()