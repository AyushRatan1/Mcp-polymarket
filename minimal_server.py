from typing import Any
import asyncio
import json
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

server = Server("polymarket_minimal")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools for the minimal PolyMarket server.
    """
    return [
        types.Tool(
            name="list-markets-minimal",
            description="Get a list of sample prediction markets (minimal test version)",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of markets to return (default: 3)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
            },
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent]:
    """Handle tool calls for the minimal PolyMarket server."""
    try:
        if name == "list-markets-minimal":
            limit = arguments.get("limit", 3) if arguments else 3
            
            sample_markets = [
                {
                    "condition_id": "sample-1",
                    "description": "Will Donald Trump win the 2024 US Presidential Election?",
                    "category": "Politics",
                    "active": True,
                    "volume": "1234567.89",
                },
                {
                    "condition_id": "sample-2",
                    "description": "Will Kamala Harris be the Democratic nominee?",
                    "category": "Politics",
                    "active": True,
                    "volume": "987654.32",
                },
                {
                    "condition_id": "sample-3",
                    "description": "Will Bitcoin reach $100,000 before the end of 2024?",
                    "category": "Crypto",
                    "active": True,
                    "volume": "2345678.90",
                },
                {
                    "condition_id": "sample-4",
                    "description": "Will the Los Angeles Lakers win the 2025 NBA Championship?",
                    "category": "Sports",
                    "active": True,
                    "volume": "567890.12",
                }
            ]
            
            formatted_markets = ["Available Sample Markets (Minimal Test):\n"]
            
            for market in sample_markets[:limit]:
                volume = float(market.get('volume', 0))
                volume_str = f"${volume:,.2f}"
                
                formatted_markets.append(
                    f"Condition ID: {market.get('condition_id', 'N/A')}\n"
                    f"Description: {market.get('description', 'N/A')}\n"
                    f"Category: {market.get('category', 'N/A')}\n"
                    f"Active: {market.get('active', 'N/A')}\n"
                    f"Volume: {volume_str}\n"
                    "---\n"
                )
            
            return [types.TextContent(type="text", text="\n".join(formatted_markets))]
        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error executing tool: {str(e)}")]

async def main():
    """Main entry point for the minimal MCP server."""
    print("Starting minimal PolyMarket MCP server...")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        print("Server stdio initialized")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="polymarket_minimal",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
    print("Server shutting down...")

if __name__ == "__main__":
    asyncio.run(main()) 