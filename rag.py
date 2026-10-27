def hybrid_search(query: str) -> str:
    """Perform hybrid search over internal technical documentation."""
    query_clean = query.lower()
    if "502" in query_clean or "bad gateway" in query_clean:
        return (
            "[Doc ID 1]: To resolve Error 502 Bad Gateway, restart the Nginx service using `sudo systemctl restart nginx`.\n"
            "[Doc ID 3]: Error 502 can also be caused by an upstream Python FastAPI service crashing due to Out Of Memory (OOM)."
        )
    return "No relevant internal documentation found for this query."