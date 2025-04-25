#!/bin/bash
# Script to start the Polymarket MCP server

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Running setup script...${NC}"
    ./scripts/setup.sh
    if [ $? -ne 0 ]; then
        echo -e "${RED}Setup failed. Please run setup.sh manually.${NC}"
        exit 1
    fi
fi

# Kill any existing server processes
echo -e "${YELLOW}Stopping any existing server processes...${NC}"
pkill -f "python3.*enhanced_server_v3.py" || true

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Start the server
echo -e "${GREEN}Starting Polymarket MCP server...${NC}"
python3 enhanced_server_v3.py &

# Show instructions for Claude Desktop
echo -e "${GREEN}Server started in the background.${NC}"
echo -e "${GREEN}You can now use the Polymarket MCP tools in Claude Desktop.${NC}"
echo -e "${YELLOW}Make sure you have configured Claude Desktop as described in the README.${NC}"

# Show server logs
echo -e "${YELLOW}Server logs will appear below. Press Ctrl+C to stop viewing logs (server will continue running).${NC}"
sleep 2
tail -f /tmp/polymarket_mcp.log 