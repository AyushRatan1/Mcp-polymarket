# Custom Polymarket MCP Functions

This extension to the Enhanced Polymarket MCP server adds three specialized functions focused on detailed prediction market data:

1. **List All Prediction Markets**: Comprehensive listing with filtering options
2. **Prediction Market Graph**: Historical price data and trend analysis
3. **Market Orderbook**: Bid-ask spread and order details

## Setup Instructions

1. Make sure your environment is set up with required dependencies:
   ```
   pip install numpy pydantic httpx asyncio
   ```

2. Update your Claude Desktop configuration:
   - Location: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Add the `polymarket_custom` entry (see below)

```json
{
  "mcpServers": {
    "polymarket_custom": {
      "command": "sh",
      "args": [
        "-c",
        "cd /path/to/project && source venv/bin/activate && python3 enhanced_server_v5.py"
      ],
      "restartOnExit": true,
      "maxRestarts": 5
    }
  }
}
```

3. Restart Claude Desktop to apply changes
4. In Claude, enable the "polymarket_custom" tool

## Function Documentation

### 1. list_all_prediction_markets

Get a comprehensive list of prediction markets with filtering options.

**Parameters:**
- `query` (optional): Search term to filter markets
- `limit` (optional): Maximum number of markets to return (default: 20, max: 100)
- `category` (optional): Filter by category (crypto, politics, sports, economy, markets)
- `status` (optional): Filter by market status (active, closed, resolved, voided)

**Example usage:**
```
list_all_prediction_markets(query="bitcoin", category="crypto", status="active", limit=10)
```

**Response:**
- Formatted table of markets with titles, status, probabilities, volume, and IDs
- Markets grouped by category
- Summary statistics

### 2. list_prediction_market_graph

Get historical price chart data for a specific market.

**Parameters:**
- `market_id` (required): ID of the market to analyze
- `timeframe` (optional): Time period to analyze (1d, 7d, 30d, 90d, 180d, all)
- `resolution` (optional): Data point resolution (hourly, daily, weekly)

**Example usage:**
```
list_prediction_market_graph(market_id="mkt-123abc", timeframe="30d", resolution="daily")
```

**Response:**
- Price data table with dates and probabilities
- Summary statistics (average, min, max, volatility)
- Trend analysis and price change calculations

### 3. list_prediction_market_orderbook

Get the bid-ask spread and order details for a market.

**Parameters:**
- `market_id` (required): ID of the market to get orderbook for
- `date` (optional): Historical date for orderbook snapshot (YYYY-MM-DD)
- `outcome` (optional): Which outcome to analyze (YES/NO, default: YES)
- `depth` (optional): Number of orders to return on each side (default: 5)

**Example usage:**
```
list_prediction_market_orderbook(market_id="mkt-123abc", outcome="YES", depth=10)
```

**Response:**
- Current market price and spread details
- Buy orders (bids) table with prices, volumes, and totals
- Sell orders (asks) table with prices, volumes, and totals
- Market activity summary and volume imbalance analysis

## Testing

A test script is provided to validate the MCP functions:

```
python test_custom_mcp.py
```

This script will guide you through testing each function individually.

## Notes

- The current implementation uses synthetic data generation for demonstration purposes
- In a production environment, these functions would fetch real data from actual Polymarket APIs
- The functions are designed to return well-formatted, easy-to-read responses in Claude 