# Save as: mcp_server_data.py
from fastmcp import FastMCP

server = FastMCP("DataUtilityServer")

@server.tool()
def convert_celsius_to_fahrenheit(celsius: float) -> float:
    """Converts a temperature from Celsius to Fahrenheit."""
    return (celsius * 9 / 5) + 32

@server.tool()
def convert_mb_to_gb(megabytes: float) -> float:
    """Converts a file size from megabytes to gigabytes."""
    return round(megabytes / 1024, 4)

@server.tool()
def calculate_percentage(value: float, total: float) -> str:
    """Calculates what percentage 'value' is of 'total'."""
    if total == 0:
        return "Error: total cannot be zero."
    return f"{(value / total) * 100:.2f}%"

if __name__ == "__main__":
    server.run(transport="stdio")
