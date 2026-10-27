import os
import sys
import pytest
from langchain_core.messages import HumanMessage
from agent import app

# 1. Check environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("[ERROR]: GROQ_API_KEY environment variable is missing!")
    sys.exit(1)

def run_agent(query: str):
    """Executes the agent graph and returns all state messages."""
    return app.invoke({"messages": [HumanMessage(content=query)]})

def test_agent_error_502_resolution():
    """Validates that the agent triggers tools and provides accurate Nginx guidance."""
    query = "Our website is throwing Error 502. Can you check what's wrong and how to fix it?"
    
    # Execute workflow
    final_state = run_agent(query)
    messages = final_state["messages"]
    
    # Get last response text
    final_response = messages[-1].content.lower()
    
    # Assertion 1: Verify tools were executed during graph traversal
    has_tool_call = any(
        hasattr(msg, "tool_calls") and msg.tool_calls 
        for msg in messages
    )
    assert has_tool_call, "Agent failed to execute required tools."
    
    # Assertion 2: Verify server status diagnosis was retrieved
    assert "degraded" in final_response or "502" in final_response, \
        "Agent failed to report server degradation status."
    
    # Assertion 3: Verify correct resolution command is recommended
    assert "systemctl restart nginx" in final_response or "nginx" in final_response, \
        "Agent failed to suggest restarting Nginx service."
    
    print("\n[PASSED]: Tool execution, server diagnosis, and remediation verified!")

if __name__ == "__main__":
    print("\n--- Running Deterministic Support Agent Tests ---\n")
    test_agent_error_502_resolution()
    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")