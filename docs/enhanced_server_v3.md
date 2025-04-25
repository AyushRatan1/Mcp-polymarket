# Enhanced Server (v3) Documentation

## Overview

The Enhanced Server (v3) is a comprehensive implementation of a prediction market MCP server that integrates with the Polymarket API and provides advanced market analysis capabilities. This server is designed to work seamlessly with Claude Desktop and other AI assistants that support the Model Context Protocol.

## Features

- **Real-time market data** from Polymarket API
- **Advanced market analysis** including trend analysis, price forecasting, and market insights
- **MCP protocol support** for integration with Claude Desktop
- **Custom endpoint handling** for REST API access
- **Data caching** for improved performance
- **Error handling and resilience** for API failures

## Available Tools

### 1. getMarkets

Retrieves a list of prediction markets from Polymarket with filtering options.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "category": {
      "type": "string",
      "description": "Filter markets by category (e.g., 'politics', 'crypto')"
    },
    "status": {
      "type": "string",
      "description": "Filter markets by status",
      "enum": ["active", "closed", "resolved", "voided"]
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of markets to return",
      "default": 10
    }
  }
}
```

**Example Usage:**
```
Can you show me the top 5 active politics markets?
```

### 2. refreshMarkets

Forces a refresh of the market data cache, fetching the latest data from the Polymarket API.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {}
}
```

**Example Usage:**
```
Please refresh the market data to get the latest information.
```

### 3. analyzeMarket

Provides detailed analysis for a specific market, including price trends, volume analysis, and market insights.

**Input Schema:**
```json
{
  "type": "object",
  "properties": {
    "market_id": {
      "type": "string",
      "description": "The ID of the market to analyze"
    },
    "timeframes": {
      "type": "array",
      "description": "Timeframes to analyze (e.g., '24h', '7d')",
      "items": {
        "type": "string"
      },
      "default": ["24h"]
    }
  },
  "required": ["market_id"]
}
```

**Example Usage:**
```
Analyze the market with ID "pm-123456" for the last 7 days.
```

## Technical Details

### API Endpoints

The server provides the following HTTP endpoints:

- `GET /health` - Health check endpoint
- `GET /` - Root endpoint with API information
- `POST /analyze_market` - Analyze a specific market
- `POST /refresh_markets` - Force refresh market data
- `GET /markets` - List available markets with filtering options
- `POST /` - MCP Protocol endpoint for JSON-RPC requests
- `GET /capabilities` - MCP Protocol capabilities endpoint

### Data Models

The server uses several data models for structured data handling:

- `MarketStatus` - Enum for market status (active, closed, resolved, voided)
- `MarketOutcome` - Data model for market outcomes with probabilities
- `ProcessedMarket` - Comprehensive market data model
- `APIResponse` - Standardized API response format
- `MarketAnalysis` - Results of market analysis
- `TrendAnalysis` - Results of trend analysis

### Configuration

The server uses the following configuration options:

- `POLYMARKET_API_BASE` - Base URL for Polymarket API
- `MAX_QUERY_LIMIT` - Maximum number of records per API request
- `MAX_RETRIES` - Maximum number of API retry attempts
- `RETRY_DELAY` - Delay between retries in seconds
- `REQUEST_TIMEOUT` - API request timeout in seconds
- `DATA_REFRESH_MINUTES` - Frequency of data updates
- `CACHE_EXPIRY_MINUTES` - How long to keep cached data

## Running the Server

```bash
# Activate virtual environment
source venv/bin/activate

# Run the server
python3 enhanced_server_v3.py
```

The server will start on port 8765 by default.

## Integration with Claude Desktop

Update your Claude Desktop configuration to include:

```json
{
  "mcpServers": {
    "polymarket_enhanced": {
      "command": "sh",
      "args": [
        "-c",
        "cd /path/to/Mcp-polymarket && source venv/bin/activate && python3 enhanced_server_v3.py"
      ]
    }
  },
  "openApiResponseFormat": {
    "polymarket_enhanced": {
      "url": false,
      "responseMode": "json"
    }
  }
}
```

## Dependencies

- `fastapi` - Web framework for building APIs
- `uvicorn` - ASGI server for FastAPI
- `httpx` - HTTP client for async requests
- `pydantic` - Data validation and settings management
- `numpy`, `scipy` - Scientific computing libraries
- `python-dotenv` - Environment variable management 