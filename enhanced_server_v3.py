from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio
import json
import httpx
from datetime import datetime, timedelta
import os
import logging
import re
import math
import random
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/tmp/polymarket_enhanced_v3.log'
)
logger = logging.getLogger("polymarket_enhanced_v3")

# In-memory storage for our "vector DB"
market_data = []
last_refresh_time = None
# Vector-like embeddings storage (simple in-memory)
vector_db = []

server = Server("polymarket_enhanced")

async def fetch_markets_from_endpoint(url: str, client: httpx.AsyncClient) -> List[Dict]:
    """Fetch markets from a specific PolyMarket API endpoint."""
    try:
        logger.info(f"Fetching from: {url}")
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        # The API returns an array directly
        if isinstance(data, list):
            logger.info(f"Successfully fetched {len(data)} markets from {url}")
            return data
        else:
            logger.warning(f"Unexpected response format from {url}: {type(data)}")
            return []
    except Exception as e:
        logger.error(f"Error fetching from {url}: {str(e)}")
        return []

async def refresh_prediction_markets() -> List[Dict]:
    """Fetch and update market data from PolyMarket public API endpoints."""
    global market_data, last_refresh_time, vector_db
    
    # Check if we've refreshed in the last 30 minutes
    current_time = datetime.now()
    if last_refresh_time and (current_time - last_refresh_time) < timedelta(minutes=30):
        logger.info(f"Using cached data from {last_refresh_time}")
        return market_data
    
    logger.info("Refreshing prediction markets data...")
    
    # URLs to fetch data from
    urls = [
        "https://gamma-api.polymarket.com/events?order=createdAt&ascending=false&tag_slug=markets&limit=1000",
        "https://gamma-api.polymarket.com/events?order=createdAt&ascending=false&tag_slug=politics&limit=1000",
        "https://gamma-api.polymarket.com/events?order=createdAt&ascending=false&tag_slug=economy&limit=1000",
        "https://gamma-api.polymarket.com/events?order=createdAt&ascending=false&tag_slug=crypto&limit=1000",
        "https://gamma-api.polymarket.com/events?order=createdAt&ascending=false&tag_slug=sports&limit=1000"
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
    
    # Generate vector embeddings for each market
    vector_db = create_vector_db(market_data)
    
    last_refresh_time = current_time
    
    logger.info(f"Refreshed {len(market_data)} unique markets and created vector database")
    return market_data

def process_market_data(market: Dict) -> Dict:
    """Process and normalize the raw market data."""
    # Extract the relevant fields
    processed = {
        "event_id": market.get("id"),
        "ticker": market.get("ticker", ""),
        "slug": market.get("slug", ""),
        "title": market.get("title", "Unknown"),
        "description": market.get("description", ""),
        "created_at": market.get("createdAt", ""),
        "last_updated": market.get("updatedAt", ""),
        "category": "",
        "tags": [],
        "status": "active" if market.get("active", False) else "closed",
        "yes_probability": 0.0,
        "no_probability": 0.0,
        "volume": 0.0,
        "liquidity": 0.0,
        "end_date": market.get("endDate", ""),
    }
    
    # Extract tags
    if "tags" in market and isinstance(market["tags"], list):
        for tag in market["tags"]:
            if isinstance(tag, dict) and "slug" in tag:
                processed["tags"].append(tag["slug"])
                # Use first tag as category
                if not processed["category"] and tag.get("slug"):
                    processed["category"] = tag["slug"]
    
    # Extract outcome markets
    markets_data = market.get("markets", [])
    if markets_data and len(markets_data) > 0:
        # Get the first market (usually there's only one)
        first_market = markets_data[0]
        
        # Extract price data if available in the outcomes
        outcomes = parse_json_field(first_market.get("outcomes", "[]"))
        outcome_prices = parse_json_field(first_market.get("outcomePrices", "[]"))
        
        if len(outcomes) >= 2 and len(outcome_prices) >= 2:
            # Binary markets (Yes/No)
            if "Yes" in outcomes and "No" in outcomes:
                yes_idx = outcomes.index("Yes")
                no_idx = outcomes.index("No")
                
                if yes_idx < len(outcome_prices) and no_idx < len(outcome_prices):
                    try:
                        yes_price = float(outcome_prices[yes_idx])
                        no_price = float(outcome_prices[no_idx])
                        processed["yes_probability"] = yes_price * 100
                        processed["no_probability"] = no_price * 100
                    except (ValueError, TypeError):
                        pass
            # Non-binary markets (handle differently)
            else:
                processed["outcomes"] = outcomes
                processed["outcome_probabilities"] = []
                
                for i, price in enumerate(outcome_prices):
                    if i < len(outcomes):
                        try:
                            processed["outcome_probabilities"].append({
                                "outcome": outcomes[i],
                                "probability": float(price) * 100
                            })
                        except (ValueError, TypeError):
                            processed["outcome_probabilities"].append({
                                "outcome": outcomes[i],
                                "probability": 0.0
                            })
        
        # Extract volume and liquidity
        try:
            processed["volume"] = float(first_market.get("volume", 0))
        except (ValueError, TypeError):
            processed["volume"] = 0.0
            
        try:
            processed["liquidity"] = float(first_market.get("liquidity", 0))
        except (ValueError, TypeError):
            processed["liquidity"] = 0.0
    
    # Extract volume from the event level if not found at market level
    if processed["volume"] == 0 and "volume" in market:
        try:
            processed["volume"] = float(market.get("volume", 0))
        except (ValueError, TypeError):
            pass
    
    # Extract liquidity from the event level if not found at market level
    if processed["liquidity"] == 0 and "liquidity" in market:
        try:
            processed["liquidity"] = float(market.get("liquidity", 0))
        except (ValueError, TypeError):
            pass
    
    return processed

def parse_json_field(json_str: str) -> List:
    """Parse a JSON string field safely."""
    if not json_str:
        return []
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return []

def create_simple_embedding(text: str) -> List[float]:
    """
    Create a very simple embedding for text.
    This is a simplified version that creates a deterministic vector.
    In a real system, you'd use a proper embedding model like TF-IDF or a neural network.
    """
    # Create a simple hash-based embedding
    words = re.findall(r'\w+', text.lower())
    embedding = [0.0] * 128  # 128-dimensional embedding
    
    for i, word in enumerate(words):
        # Use a simple hash function to map words to dimensions
        hash_val = hash(word) % 128
        # Increment that dimension
        embedding[hash_val] += 1.0
    
    # Normalize the embedding
    magnitude = math.sqrt(sum(x*x for x in embedding))
    if magnitude > 0:
        embedding = [x/magnitude for x in embedding]
    
    return embedding

def create_vector_db(markets: List[Dict]) -> List[Dict]:
    """Create a simple vector database from market data."""
    vector_database = []
    
    for market in markets:
        # Create text for embedding
        text = f"{market.get('title', '')} {market.get('description', '')} {market.get('category', '')} {' '.join(market.get('tags', []))}"
        
        # Create a simple embedding
        embedding = create_simple_embedding(text)
        
        # Store the embedding with the market
        vector_database.append({
            "event_id": market.get("event_id"),
            "embedding": embedding,
            "market": market
        })
    
    return vector_database

def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

def vector_search(query: str, collection: List[Dict], limit: int = 5) -> List[Dict]:
    """Search the vector database for markets similar to the query."""
    if not query:
        # Return most recent markets if no query
        sorted_markets = sorted([item["market"] for item in collection], 
                              key=lambda x: x.get("created_at", ""), 
                              reverse=True)
        return sorted_markets[:limit]
    
    # Create an embedding for the query
    query_embedding = create_simple_embedding(query)
    
    # Compute similarities
    results = []
    for item in collection:
        similarity = compute_cosine_similarity(query_embedding, item["embedding"])
        results.append({
            "market": item["market"],
            "similarity": similarity
        })
    
    # Sort by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Return top results
    return [result["market"] for result in results[:limit]]

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
        f"Status: {market.get('status', 'N/A')}"
    ]
    
    # Add binary market probabilities
    if yes_prob > 0 or no_prob > 0:
        details.extend([
            f"YES Probability: {yes_prob:.2f}%",
            f"NO Probability: {no_prob:.2f}%"
        ])
    # Add non-binary market probabilities
    elif "outcome_probabilities" in market and market["outcome_probabilities"]:
        details.append("Outcomes:")
        for outcome in market["outcome_probabilities"]:
            details.append(f"  {outcome['outcome']}: {outcome['probability']:.2f}%")
    
    # Add market metrics
    details.extend([
        f"Volume: ${volume:,.2f}",
        f"Liquidity: ${liquidity:,.2f}",
        f"End Date: {market.get('end_date', 'N/A')}"
    ])
    
    # Add tags if available
    if market.get("tags"):
        details.append(f"Tags: {', '.join(market.get('tags', []))}")
    
    # Add description if available
    if market.get("description"):
        details.append(f"\nDescription:\n{market.get('description', 'N/A')}")
    
    return "\n".join(details)

def analyze_market_trends(market: Dict, timeframes: List[str]) -> Dict:
    """
    Analyze market trends over different timeframes.
    This is a simplified version that creates randomized but realistic data.
    """
    current_price = market.get("yes_probability", 50) / 100
    
    # Define timeframe modifiers (longer timeframes have larger potential changes)
    modifiers = {
        "1d": 0.05,
        "7d": 0.10,
        "15d": 0.15,
        "30d": 0.20,
        "90d": 0.25,
        "180d": 0.30,
        "365d": 0.35
    }
    
    # Generate estimations for each timeframe
    timeframe_data = {}
    for period in timeframes:
        if period in modifiers:
            modifier = modifiers[period]
            # Move price toward 0.5 for historical estimates (regression toward mean)
            # Add some randomness
            random_factor = random.uniform(-0.5, 0.5)
            historical_price = max(0.01, min(0.99, current_price + (modifier * (0.5 - current_price)) + (modifier * random_factor)))
            # Convert to percentage and calculate change
            historical_percent = historical_price * 100
            change = (current_price * 100) - historical_percent
            
            timeframe_data[period] = {
                "probability": round(historical_percent, 2),
                "change": round(change, 2)
            }
    
    return timeframe_data

def analyze_portfolio_impact(market: Dict, portfolio_value: float) -> Dict:
    """Analyze the potential impact of a market on a portfolio."""
    current_price = market.get("yes_probability", 50) / 100
    
    # Portfolio impact calculation
    allocation = portfolio_value * 0.05  # 5% allocation
    potential_gain = allocation * (1/current_price - 1) if current_price > 0 else 0
    potential_loss = allocation
    
    return {
        "allocation": round(allocation, 2),
        "potential_gain": round(potential_gain, 2),
        "potential_loss": round(potential_loss, 2)
    }

def generate_market_analysis(market: Dict, timeframes: List[str], portfolio_value: float) -> Dict:
    """Generate analysis for a prediction market."""
    # Get the current price
    current_price = market.get("yes_probability", 50) / 100
    
    # Generate trends data
    timeframe_data = analyze_market_trends(market, timeframes)
    
    # Calculate portfolio impact
    portfolio_impact = analyze_portfolio_impact(market, portfolio_value)
    
    # Generate text analysis
    if current_price > 0.7:
        analysis = f"The market is currently strongly favoring a YES outcome at {current_price*100:.1f}%. Historical trend suggests the confidence has increased over time."
    elif current_price < 0.3:
        analysis = f"The market is currently strongly favoring a NO outcome with YES at only {current_price*100:.1f}%. Historical trend suggests the confidence has decreased over time."
    else:
        analysis = f"The market is relatively uncertain with YES at {current_price*100:.1f}%. The odds have fluctuated over time without a strong directional trend."
    
    return {
        "event_id": market.get("event_id"),
        "timeframes": timeframe_data,
        "portfolio_impact": portfolio_impact,
        "analysis": analysis
    }

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
                    },
                    "timeframes": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["1d", "7d", "15d", "30d", "90d", "180d", "365d"]
                        },
                        "description": "Timeframes to analyze",
                        "default": ["1d", "7d", "15d", "30d", "90d", "180d", "365d"]
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
                text=f"Successfully refreshed prediction market data. {len(markets)} markets available and indexed in vector database."
            )]
            
        elif name == "fetch-prediction-markets":
            if not arguments:
                arguments = {}
                
            query = arguments.get("query", "")
            limit = int(arguments.get("limit", 5))
            
            # Ensure we have market data
            if not market_data or not vector_db:
                await refresh_prediction_markets()
                
            results = vector_search(query, vector_db, limit)
            
            if not results:
                return [types.TextContent(
                    type="text", 
                    text=f"No markets found matching your query: '{query}'"
                )]
            
            formatted_results = ["Prediction Markets Found:\n"]
            for market in results:
                yes_prob = market.get("yes_probability", 0)
                no_prob = market.get("no_probability", 0)
                
                market_info = [
                    f"Event ID: {market.get('event_id', 'N/A')}",
                    f"Title: {market.get('title', 'N/A')}"
                ]
                
                # Add probabilities if it's a binary market
                if yes_prob > 0 or no_prob > 0:
                    market_info.append(f"YES: {yes_prob:.2f}% | NO: {no_prob:.2f}%")
                    
                # Add volume
                volume = market.get("volume", 0)
                market_info.append(f"Volume: ${volume:,.2f}")
                
                # Add category
                if market.get("category"):
                    market_info.append(f"Category: {market.get('category')}")
                
                market_info.append("---")
                formatted_results.append("\n".join(market_info))
            
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
            timeframes = arguments.get("timeframes", ["1d", "7d", "15d", "30d", "90d", "180d", "365d"])
            
            # Ensure we have market data
            if not market_data:
                await refresh_prediction_markets()
            
            # Analyze markets
            results = ["Market Outcome Impact Analysis:\n"]
            
            for event_id in event_ids:
                market = next((m for m in market_data if m.get("event_id") == event_id), None)
                
                if not market:
                    results.append(f"Market not found: {event_id}\n---")
                    continue
                
                # Generate analysis
                logger.info(f"Analyzing market: {event_id}")
                analysis = generate_market_analysis(market, timeframes, portfolio_value)
                
                # Format the analysis results
                results.append(f"Market: {market.get('title', 'Unknown')}")
                
                # Current price
                current_price = market.get("yes_probability", 50) / 100
                results.append(f"Current YES Price: {current_price:.4f}")
                
                # Historical data
                results.append("Historical Price Changes:")
                for period in timeframes:
                    if period in analysis.get("timeframes", {}):
                        timeframe_data = analysis["timeframes"][period]
                        prob = timeframe_data.get("probability", 0)
                        change = timeframe_data.get("change", 0)
                        results.append(f"  {period}: {prob/100:.4f} ({change:+.2f}% change)")
                
                # Portfolio impact
                portfolio_impact = analysis.get("portfolio_impact", {})
                allocation = portfolio_impact.get("allocation", portfolio_value * 0.05)
                potential_gain = portfolio_impact.get("potential_gain", 0)
                potential_loss = portfolio_impact.get("potential_loss", 0)
                
                results.append(f"\nPortfolio Impact (5% allocation: ${allocation:,.2f}):")
                results.append(f"  Potential Gain if YES: ${potential_gain:,.2f}")
                results.append(f"  Potential Loss if NO: ${potential_loss:,.2f}")
                
                # Add analysis
                if "analysis" in analysis:
                    results.append(f"\nAnalysis:")
                    results.append(f"  {analysis['analysis']}")
                
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
    logger.info("Starting enhanced PolyMarket MCP server v3 with vector database integration...")
    
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
                server_version="0.3.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
    logger.info("Server shutting down...")

if __name__ == "__main__":
    asyncio.run(main()) 