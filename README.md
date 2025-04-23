# Polymarket Enhanced MCP Server

A powerful MCP server for analyzing Polymarket prediction markets with AI integration.

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Mcp-polymarket
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

5. Run the server:
```bash
python3 enhanced_server_v3.py
```

## Features

- Real-time prediction market data from Polymarket
- AI-powered market analysis using Google's Gemini
- Portfolio impact analysis
- Market trend analysis
- Risk assessment tools

## Configuration

The server can be configured through environment variables in the `.env` file:

- `GEMINI_API_KEY`: Your Google Gemini API key (required for AI analysis)

## Tools Available

1. `refresh-prediction-markets`: Refresh market data
2. `fetch-prediction-markets`: Search markets by query
3. `fetch-prediction-market-details`: Get detailed market information
4. `research-prediction-markets-outcome-impact`: Analyze historical trends
5. `analyze-portfolio-with-markets`: Portfolio impact analysis
6. `verify-portfolio-risk-claims`: Verify risk analysis claims

## Integration with Claude Desktop

1. Update your Claude Desktop configuration:
```json
{
    "mcpServers": {
        "polymarket_enhanced": {
            "command": "sh",
            "args": [
                "-c",
                "cd /path/to/Mcp-polymarket && source venv/bin/activate && python3 enhanced_server_v3.py"
            ],
            "restartOnExit": true,
            "maxRestarts": 5
        }
    }
}
```

2. Restart Claude Desktop to apply the changes.

## Notes

- The server requires Python 3.8 or higher
- Make sure to keep your API keys secure and never commit them to version control
- The AI analysis features require a valid Gemini API key


