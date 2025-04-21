from typing import Any, Dict, List, Optional
import asyncio
import json
import httpx
from datetime import datetime, timedelta
import os
import logging
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/tmp/polymarket_enhanced.log'
)
logger = logging.getLogger("polymarket_enhanced")

# In-memory storage for our "vector DB"
# In a real implementation, you would use a proper vector database
market_data = []
last_refresh_time = None

server = Server("polymarket_enhanced")

async def fetch_markets_from_endpoint(url: str, client: httpx.AsyncClient) -> List[Dict]:
    """Fetch markets from a specific PolyMarket API endpoint."""
    try:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        # The API returns an array directly, not an object with a data property
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Error fetching from {url}: {str(e)}")
        return []

async def refresh_prediction_markets() -> List[Dict]:
    """Fetch and update market data from PolyMarket public API endpoints."""
    global market_data, last_refresh_time
    
    # Check if we've refreshed in the last hour
    current_time = datetime.now()
    if last_refresh_time and (current_time - last_refresh_time) < timedelta(hours=1):
        return market_data
    
    logger.info("Refreshing prediction markets data...")
    
    # URLs to fetch data from
    urls = [
        "https://gamma-api.polymarket.com/events?order=createdAt&ascending=false&tag_slug=markets&limit=1000",
        "https://gamma-api.polymarket.com/events?order=createdAt&ascending=false&tag_slug=politics&limit=1000",
        "https://gamma-api.polymarket.com/events?order=createdAt&ascending=false&tag_slug=economy&limit=1000"
    ]
    
    all_markets = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Fetch data from all endpoints
        results = await asyncio.gather(*[
            fetch_markets_from_endpoint(url, client) for url in urls
        ])
        
        # Combine all results
        for market_list in results:
            all_markets.extend(market_list)
    
    # Remove duplicates based on event_id
    unique_markets = {}
    for market in all_markets:
        market_id = market.get("id")
        if market_id and market_id not in unique_markets:
            # Process and enrich the market data
            processed_market = process_market_data(market)
            unique_markets[market_id] = processed_market
    
    # Update our "vector DB"
    market_data = list(unique_markets.values())
    last_refresh_time = current_time
    
    logger.info(f"Refreshed {len(market_data)} unique markets")
    return market_data

def process_market_data(market: Dict) -> Dict:
    """Process and normalize the raw market data."""
    # Extract the relevant fields
    processed = {
        "event_id": market.get("id"),
        "title": market.get("title", "Unknown"),
        "description": market.get("description", ""),
        "created_at": market.get("created_at", ""),
        "last_updated": market.get("updated_at", ""),
        "category": "",
        "tags": [],
        "yes_probability": 0.0,
        "no_probability": 0.0,
        "volume": 0.0,
        "liquidity": 0.0,
        "end_date": market.get("end_date_iso", ""),
        "raw_data": market  # Store the raw data for additional details
    }
    
    # Extract tags
    if "tags" in market and isinstance(market["tags"], list):
        for tag in market["tags"]:
            if isinstance(tag, dict) and "slug" in tag:
                processed["tags"].append(tag["slug"])
                # Use first tag as category
                if not processed["category"] and tag.get("slug"):
                    processed["category"] = tag["slug"]
    
    # Extract price data if available
    if "prices" in market and isinstance(market["prices"], dict):
        prices = market["prices"]
        yes_price = float(prices.get("yes", 0))
        processed["yes_probability"] = yes_price * 100  # Convert to percentage
        processed["no_probability"] = (1 - yes_price) * 100  # Convert to percentage
    
    # Extract volume and liquidity
    processed["volume"] = float(market.get("volume", 0))
    processed["liquidity"] = float(market.get("liquidity", 0))
    
    return processed

def simple_vector_search(collection: List[Dict], query: str, limit: int = 5) -> List[Dict]:
    """Simple keyword-based search as a stand-in for vector search."""
    query = query.lower()
    results = []
    
    for item in collection:
        title = item.get("title", "").lower()
        description = item.get("description", "").lower()
        
        # Simple relevance score based on keyword presence
        relevance = 0
        if query in title:
            relevance += 10  # Higher weight for title matches
        
        if query in description:
            relevance += 5   # Lower weight for description matches
        
        # Check for individual words
        query_words = query.split()
        for word in query_words:
            if word in title:
                relevance += 2
            if word in description:
                relevance += 1
        
        if relevance > 0:
            results.append({
                "item": item,
                "relevance": relevance
            })
    
    # Sort by relevance score
    results.sort(key=lambda x: x["relevance"], reverse=True)
    
    # Return top results
    return [result["item"] for result in results[:limit]]

def format_market_details(market: Dict) -> str:
    """Format the market details into a clean tabular string."""
    yes_prob = market.get("yes_probability", 0)
    no_prob = market.get("no_probability", 0)
    volume = market.get("volume", 0)
    liquidity = market.get("liquidity", 0)
    
    # Format the market information
    details = [
        f"Event ID: {market.get('event_id', 'N/A')}",
        f"Title: {market.get('title', 'N/A')}",
        f"Category: {market.get('category', 'N/A')}",
        f"YES Probability: {yes_prob:.2f}%",
        f"NO Probability: {no_prob:.2f}%",
        f"Volume: ${volume:,.2f}",
        f"Liquidity: ${liquidity:,.2f}",
        f"End Date: {market.get('end_date', 'N/A')}"
    ]
    
    # Add tags if available
    if market.get("tags"):
        details.append(f"Tags: {', '.join(market.get('tags', []))}")
    
    # Add description if available
    if market.get("description"):
        details.append(f"\nDescription:\n{market.get('description', 'N/A')}")
    
    return "\n".join(details)

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools for the enhanced PolyMarket server."""
    return [
        types.Tool(
            name="refresh-prediction-markets",
            description="Refresh the prediction market data from PolyMarket public API",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
        types.Tool(
            name="fetch-prediction-markets",
            description="Search for prediction markets based on a query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for finding relevant markets"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="fetch-prediction-market-details",
            description="Get detailed information about a specific prediction market",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "The ID of the market to get details for"
                    }
                },
                "required": ["event_id"]
            }
        ),
        types.Tool(
            name="research-prediction-markets-outcome-impact",
            description="Research historical price changes for markets and their potential impact",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_ids": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of event IDs to analyze"
                    },
                    "portfolio_value": {
                        "type": "number",
                        "description": "Current portfolio value in USD",
                        "default": 10000
                    }
                },
                "required": ["event_ids"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent]:
    """Handle tool calls for the enhanced PolyMarket server."""
    try:
        if name == "refresh-prediction-markets":
            markets = await refresh_prediction_markets()
            return [types.TextContent(
                type="text", 
                text=f"Successfully refreshed prediction market data. {len(markets)} markets available."
            )]
            
        elif name == "fetch-prediction-markets":
            if not arguments or "query" not in arguments:
                return [types.TextContent(
                    type="text", 
                    text="Error: Query parameter is required."
                )]
                
            query = arguments["query"]
            limit = int(arguments.get("limit", 5))
            
            # Ensure we have market data
            if not market_data:
                await refresh_prediction_markets()
                
            results = simple_vector_search(market_data, query, limit)
            
            if not results:
                return [types.TextContent(
                    type="text", 
                    text=f"No markets found matching your query: '{query}'"
                )]
            
            formatted_results = ["Prediction Markets Found:\n"]
            for market in results:
                yes_prob = market.get("yes_probability", 0)
                no_prob = market.get("no_probability", 0)
                
                formatted_results.append(
                    f"Event ID: {market.get('event_id', 'N/A')}\n"
                    f"Title: {market.get('title', 'N/A')}\n"
                    f"YES: {yes_prob:.2f}% | NO: {no_prob:.2f}%\n"
                    "---"
                )
            
            return [types.TextContent(
                type="text", 
                text="\n".join(formatted_results)
            )]
            
        elif name == "fetch-prediction-market-details":
            if not arguments or "event_id" not in arguments:
                return [types.TextContent(
                    type="text", 
                    text="Error: event_id parameter is required."
                )]
                
            event_id = arguments["event_id"]
            
            # Ensure we have market data
            if not market_data:
                await refresh_prediction_markets()
                
            # Find the market with the matching ID
            market = next((m for m in market_data if m.get("event_id") == event_id), None)
            
            if not market:
                return [types.TextContent(
                    type="text", 
                    text=f"No market found with event ID: {event_id}"
                )]
            
            details = format_market_details(market)
            return [types.TextContent(type="text", text=details)]
            
        elif name == "research-prediction-markets-outcome-impact":
            if not arguments or "event_ids" not in arguments:
                return [types.TextContent(
                    type="text", 
                    text="Error: event_ids parameter is required."
                )]
                
            event_ids = arguments["event_ids"]
            portfolio_value = float(arguments.get("portfolio_value", 10000))
            
            # Ensure we have market data
            if not market_data:
                await refresh_prediction_markets()
                
            # This would be a more complex analysis in a real implementation
            # Here we'll create mock historical data for demonstration
            results = ["Market Outcome Impact Analysis:\n"]
            
            for event_id in event_ids:
                market = next((m for m in market_data if m.get("event_id") == event_id), None)
                
                if not market:
                    results.append(f"Market not found: {event_id}\n---")
                    continue
                
                # Generate mock historical data
                current_price = market.get("yes_probability", 50) / 100
                historical_data = {
                    "1d": max(0.01, min(0.99, current_price + (0.05 * (0.5 - current_price)))),
                    "7d": max(0.01, min(0.99, current_price + (0.10 * (0.5 - current_price)))),
                    "15d": max(0.01, min(0.99, current_price + (0.15 * (0.5 - current_price)))),
                    "30d": max(0.01, min(0.99, current_price + (0.20 * (0.5 - current_price)))),
                    "90d": max(0.01, min(0.99, current_price + (0.25 * (0.5 - current_price)))),
                    "180d": max(0.01, min(0.99, current_price + (0.30 * (0.5 - current_price)))),
                    "365d": max(0.01, min(0.99, current_price + (0.35 * (0.5 - current_price))))
                }
                
                results.append(f"Market: {market.get('title', 'Unknown')}")
                results.append(f"Current YES Price: {current_price:.4f}")
                results.append("Historical Price Changes:")
                
                for period, price in historical_data.items():
                    change = (current_price - price) * 100
                    results.append(f"  {period}: {price:.4f} ({change:+.2f}% change)")
                
                # Simple portfolio impact calculation
                allocation = portfolio_value * 0.05  # Assume 5% allocation
                potential_gain = allocation * (1/current_price - 1) if current_price > 0 else 0
                potential_loss = allocation if current_price > 0 else 0
                
                results.append(f"\nPortfolio Impact (5% allocation: ${allocation:,.2f}):")
                results.append(f"  Potential Gain if YES: ${potential_gain:,.2f}")
                results.append(f"  Potential Loss if NO: ${potential_loss:,.2f}")
                results.append("---\n")
            
            return [types.TextContent(type="text", text="\n".join(results))]
            
        else:
            return [types.TextContent(
                type="text", 
                text=f"Unknown tool: {name}"
            )]
            
    except Exception as e:
        logger.error(f"Error executing tool {name}: {str(e)}")
        return [types.TextContent(
            type="text", 
            text=f"Error executing tool: {str(e)}"
        )]

async def main():
    """Main entry point for the enhanced MCP server."""
    logger.info("Starting enhanced PolyMarket MCP server...")
    
    # Initial data refresh
    try:
        await refresh_prediction_markets()
    except Exception as e:
        logger.error(f"Error in initial data refresh: {str(e)}")
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("Server stdio initialized")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="polymarket_enhanced",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
    logger.info("Server shutting down...")

if __name__ == "__main__":
    asyncio.run(main()) 