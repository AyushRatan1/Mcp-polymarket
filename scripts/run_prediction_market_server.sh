#!/bin/bash

# Set log file
LOG_FILE="/tmp/polymarket_simple.log"

# Clean up any existing log file
echo "Starting server at $(date)" > $LOG_FILE

# Change to project directory
cd /Users/ayushratan/Desktop/projects/Mcp-polymarket

# Activate virtual environment
source venv/bin/activate

# Echo environment info
echo "Running in directory: $(pwd)" >> $LOG_FILE
echo "Using Python: $(which python3)" >> $LOG_FILE
echo "Python version: $(python3 --version)" >> $LOG_FILE

# Run the server with stdout/stderr redirected to log file
python3 simple_prediction_market_mcp.py 2>&1 | tee -a $LOG_FILE

# Output message when server stops
echo "Server stopped at $(date)" >> $LOG_FILE 