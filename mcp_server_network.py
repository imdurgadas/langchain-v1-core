# Save as: mcp_server_network.py
from fastmcp import FastMCP

server = FastMCP("NetworkUtilityServer")

@server.tool()
def check_endpoint_reachability(endpoint: str) -> str:
    """
    Simulates a connectivity check for a given endpoint URL.
    Returns latency and packet loss metrics.
    """
    endpoint_lower = endpoint.lower().strip()
    if "production" in endpoint_lower:
        return "Status: REACHABLE | Latency: 14ms | Packet loss: 0.0%"
    elif "staging" in endpoint_lower:
        return "Status: REACHABLE | Latency: 45ms | Packet loss: 0.2%"
    else:
        return "Status: UNKNOWN | Endpoint not in monitoring registry."

@server.tool()
def get_region_latency_profile(region: str) -> str:
    """Returns the average network latency for a given cloud region."""
    profiles = {
        "us-east-1": "12ms average, 99.9% uptime SLA",
        "eu-west-1":  "28ms average, 99.9% uptime SLA",
        "ap-south-1": "42ms average, 99.5% uptime SLA",
    }
    return profiles.get(region.lower(), f"No latency data for region: {region}")

if __name__ == "__main__":
    # This starts a web server on port 8765
    # Keep this running in a separate terminal while the orchestrator runs
    server.run(transport="http", host="127.0.0.1", port=8765)
