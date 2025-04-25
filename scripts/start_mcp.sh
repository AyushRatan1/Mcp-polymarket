#!/bin/bash

# Kill any existing MCP server processes
pkill -f "simple_mcp.py" || true

# Wait a moment to ensure ports are released
sleep 2

# Start the MCP server
cd /Users/ayushratan/Desktop/projects/Mcp-polymarket
source venv/bin/activate
python3 simple_mcp.py 