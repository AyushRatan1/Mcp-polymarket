#!/usr/bin/env python3
"""
Check Claude Desktop Configuration
This script checks if the Claude Desktop configuration file is correctly set up
for our MCP server.
"""

import os
import json
import subprocess
import sys

def main():
    # Path to Claude Desktop config file
    config_path = os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"ERROR: Claude Desktop config file not found at {config_path}")
        return
    
    print(f"Reading Claude Desktop config from: {config_path}")
    
    # Read config file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in config file: {e}")
        return
    except Exception as e:
        print(f"ERROR: Failed to read config file: {e}")
        return
    
    # Check if mcpServers is in config
    if "mcpServers" not in config:
        print("ERROR: 'mcpServers' section not found in config")
        return
    
    # Check if our server is in the config
    if "polymarket_simple" not in config["mcpServers"]:
        print("ERROR: 'polymarket_simple' server not found in mcpServers")
        return
    
    # Get our server config
    server_config = config["mcpServers"]["polymarket_simple"]
    print("\nServer configuration found:")
    print(json.dumps(server_config, indent=2))
    
    # Check if the command exists
    if "command" in server_config:
        command = server_config["command"]
        if os.path.isfile(command):
            print(f"\nCommand file exists: {command}")
            
            # Check if it's executable
            if os.access(command, os.X_OK):
                print("Command file is executable")
            else:
                print("WARNING: Command file is not executable")
        else:
            print(f"\nWARNING: Command file does not exist: {command}")
    
    # Check if Claude Desktop is running
    try:
        result = subprocess.run(["pgrep", "-f", "Claude Desktop"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        if result.returncode == 0:
            print("\nClaude Desktop is currently running")
            print("You should restart Claude Desktop to apply config changes")
        else:
            print("\nClaude Desktop is not currently running")
    except Exception as e:
        print(f"\nFailed to check if Claude Desktop is running: {e}")
    
    # Check if our server process is running
    try:
        result = subprocess.run(["pgrep", "-f", "simple_prediction_market_mcp.py"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        if result.returncode == 0:
            print("\nMCP server process is running (PID: " + result.stdout.strip() + ")")
        else:
            print("\nMCP server process is not running")
    except Exception as e:
        print(f"\nFailed to check if MCP server is running: {e}")
    
    print("\nTroubleshooting steps:")
    print("1. Make sure the configuration file is valid (seems OK)")
    print("2. Restart Claude Desktop to load the updated configuration")
    print("3. In Claude Desktop, go to Settings and ensure 'polymarket_simple' is enabled")
    print("4. Try typing 'list_all_prediction_markets()' in Claude")
    
if __name__ == "__main__":
    main() 