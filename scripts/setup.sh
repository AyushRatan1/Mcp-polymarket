#!/bin/bash
# Setup script for Polymarket MCP

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Polymarket MCP...${NC}"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment. Please check if you have venv installed.${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment.${NC}"
    exit 1
fi

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install dependencies.${NC}"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env file with your API keys if needed.${NC}"
fi

# Make setup script executable
chmod +x scripts/*.sh

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}To start the server, run:${NC}"
echo -e "${YELLOW}source venv/bin/activate${NC}"
echo -e "${YELLOW}python3 enhanced_server_v3.py${NC}"

# Claude Desktop configuration instructions
echo -e "\n${GREEN}Claude Desktop Configuration:${NC}"
echo -e "${YELLOW}1. Make sure Claude Desktop is installed${NC}"
echo -e "${YELLOW}2. Update Claude Desktop configuration at:${NC}"
echo -e "${YELLOW}   - macOS: ~/Library/Application Support/Claude/claude_desktop_config.json${NC}"
echo -e "${YELLOW}   - Windows: %APPDATA%\\Claude\\claude_desktop_config.json${NC}"
echo -e "${YELLOW}3. Add the following to your configuration:${NC}"
echo -e '   {
     "mcpServers": {
       "polymarket_enhanced": {
         "command": "sh",
         "args": [
           "-c",
           "cd '$(pwd)' && source venv/bin/activate && python3 enhanced_server_v3.py"
         ],
         "restartOnExit": true,
         "maxRestarts": 5
       }
     },
     "openApiResponseFormat": {
       "polymarket_enhanced": {
         "url": false,
         "responseMode": "json"
       }
     }
   }' 