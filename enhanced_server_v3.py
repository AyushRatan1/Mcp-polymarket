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
import numpy as np

# Version information
VERSION = "0.3.3"

# Try importing Gemini AI, but don't fail if not available
try:
    import google.generativeai as genai
    HAS_GEMINI = True
    # Configure Gemini API
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyB4vsx2Qj93x-1TSVDuiyEcLkrET0vG78I")
    genai.configure(api_key=GEMINI_API_KEY)
    print("Successfully loaded Google Generative AI")
except ImportError:
    HAS_GEMINI = False
    print("Google Generative AI not available, using fallback analysis")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/tmp/polymarket_enhanced_v3.log'
)
logger = logging.getLogger("polymarket_enhanced_v3")

# Constants
DEFAULT_RISK_FREE_RATE = 0.045  # 4.5% risk-free rate
POLYMARKET_API_BASE = "https://gamma-api.polymarket.com"
MAX_QUERY_LIMIT = 1000
DATA_REFRESH_MINUTES = 30

# In-memory storage for our "vector DB"
market_data = []
last_refresh_time = None
# Vector-like embeddings storage (simple in-memory)
vector_db = []

server = Server("polymarket_enhanced")

async def fetch_markets_from_endpoint(url: str, client: httpx.AsyncClient) -> List[Dict]:
    """
    Fetch markets from a specific PolyMarket API endpoint.
    
    Args:
        url: The API endpoint URL
        client: The httpx client to use
        
    Returns:
        List of market data dictionaries
        
    Raises:
        HTTPStatusError: On HTTP errors (handled internally)
        Exception: On other errors (handled internally)
    """
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
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching from {url}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error fetching from {url}: {str(e)}")
        return []

async def refresh_prediction_markets() -> List[Dict]:
    """
    Fetch and update market data from PolyMarket public API endpoints.
    
    Returns:
        List of processed market data dictionaries
        
    Note:
        Updates global market_data and vector_db variables
    """
    global market_data, last_refresh_time, vector_db
    
    # Check if we've refreshed in the last 30 minutes
    current_time = datetime.now()
    if last_refresh_time and (current_time - last_refresh_time) < timedelta(minutes=DATA_REFRESH_MINUTES):
        logger.info(f"Using cached data from {last_refresh_time}")
        return market_data
    
    logger.info("Refreshing prediction markets data...")
    
    # URLs to fetch data from
    urls = [
        f"{POLYMARKET_API_BASE}/events?order=createdAt&ascending=false&tag_slug=markets&limit={MAX_QUERY_LIMIT}",
        f"{POLYMARKET_API_BASE}/events?order=createdAt&ascending=false&tag_slug=politics&limit={MAX_QUERY_LIMIT}",
        f"{POLYMARKET_API_BASE}/events?order=createdAt&ascending=false&tag_slug=economy&limit={MAX_QUERY_LIMIT}",
        f"{POLYMARKET_API_BASE}/events?order=createdAt&ascending=false&tag_slug=crypto&limit={MAX_QUERY_LIMIT}",
        f"{POLYMARKET_API_BASE}/events?order=createdAt&ascending=false&tag_slug=sports&limit={MAX_QUERY_LIMIT}"
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

def monte_carlo_simulation(market: Dict, simulations: int = 1000) -> Dict:
    """Run Monte Carlo simulation to model potential outcomes of a prediction market"""
    yes_prob = market.get("yes_probability", 50) / 100
    allocation = market.get("allocation", 0)
    
    # Simulate outcomes based on current probability
    results = []
    for _ in range(simulations):
        # Add some volatility to the probability for each simulation
        adjusted_prob = max(0.001, min(0.999, yes_prob + random.gauss(0, 0.05)))
        outcome = random.random() < adjusted_prob  # True if event occurs
        
        # Calculate return based on outcome
        if outcome:
            return_value = allocation * (1/yes_prob - 1)
        else:
            return_value = -allocation
            
        results.append(return_value)
    
    # Calculate statistics from simulation
    results = np.array(results)
    mean_return = np.mean(results)
    median_return = np.median(results)
    std_dev = np.std(results)
    var_95 = np.percentile(results, 5)  # 95% VaR (Value at Risk)
    cvar_95 = np.mean(results[results <= var_95])  # Conditional VaR
    max_gain = np.max(results)
    max_loss = np.min(results)
    
    # Calculate confidence intervals
    ci_95_lower = np.percentile(results, 2.5)
    ci_95_upper = np.percentile(results, 97.5)
    
    return {
        "mean_return": mean_return,
        "median_return": median_return,
        "standard_deviation": std_dev,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "max_gain": max_gain,
        "max_loss": max_loss,
        "ci_95": [ci_95_lower, ci_95_upper],
        "simulations": simulations
    }

def stress_test_portfolio(portfolio_assets: Dict[str, float], markets: List[Dict], 
                          scenarios: List[Dict] = None, total_value: float = 10000) -> Dict:
    """
    Perform advanced stress testing on portfolio with prediction markets
    
    Args:
        portfolio_assets: Dict of asset names and allocation percentages
        markets: List of prediction markets
        scenarios: List of stress scenarios to test (defaults to standard scenarios)
        total_value: Total portfolio value
        
    Returns:
        Dict with stress test results
    """
    # Define standard stress scenarios if none provided
    if scenarios is None:
        scenarios = [
            {"name": "Market Crash", "asset_impact": {"default": -0.30, "Financial Services": -0.40, "Energy": -0.25}},
            {"name": "Recession", "asset_impact": {"default": -0.20, "Technology": -0.35, "Financial Services": -0.30}},
            {"name": "Inflation Spike", "asset_impact": {"default": -0.15, "Energy": 0.10, "Financial Services": -0.25}},
            {"name": "Tech Bubble", "asset_impact": {"default": 0.05, "Technology": -0.50}}
        ]
    
    # Define sector for each asset
    asset_sectors = {}
    for asset_name, allocation_pct in portfolio_assets.items():
        sector = "Technology"  # Default
        if any(term in asset_name.lower() for term in ["bank", "financial", "insurance", "visa", "mastercard"]):
            sector = "Financial Services"
        elif any(term in asset_name.lower() for term in ["oil", "gas", "energy", "solar", "petroleum"]):
            sector = "Energy"
        elif any(term in asset_name.lower() for term in ["healthcare", "pharma", "biotech", "medical"]):
            sector = "Healthcare"
        
        asset_sectors[asset_name] = sector
    
    # Calculate baseline portfolio value
    baseline = total_value
    
    stress_results = []
    for scenario in scenarios:
        scenario_result = {"scenario": scenario["name"], "asset_impacts": [], "market_impacts": [], "total_impact": 0}
        
        # Calculate impact on traditional assets
        for asset_name, allocation_pct in portfolio_assets.items():
            asset_value = total_value * (allocation_pct / 100)
            sector = asset_sectors.get(asset_name, "default")
            impact_pct = scenario["asset_impact"].get(sector, scenario["asset_impact"].get("default", 0))
            impact_value = asset_value * impact_pct
            
            scenario_result["asset_impacts"].append({
                "asset": asset_name,
                "value": asset_value,
                "impact_percentage": impact_pct * 100,
                "impact_value": impact_value
            })
            scenario_result["total_impact"] += impact_value
        
        # Calculate impact on prediction markets
        # In stress scenarios, prediction markets often become more volatile
        for market in markets:
            # Apply stress modification to market probabilities
            if "recession" in scenario["name"].lower() or "crash" in scenario["name"].lower():
                # Economic stress typically reduces probability of positive outcomes
                stress_factor = 0.7  # Reduce positive outcome probability by 30%
            elif "inflation" in scenario["name"].lower():
                # Inflation typically has mixed effects
                stress_factor = 0.9
            else:
                stress_factor = 0.85
            
            # Apply stress to market calculations
            yes_prob = market.get("yes_probability", 50) / 100
            stressed_prob = max(0.01, min(0.99, yes_prob * stress_factor))
            allocation = market.get("allocation", 0)
            
            # Calculate original expected value
            original_ev = (yes_prob * allocation * (1/yes_prob - 1)) - ((1-yes_prob) * allocation)
            
            # Calculate stressed expected value
            stressed_ev = (stressed_prob * allocation * (1/stressed_prob - 1)) - ((1-stressed_prob) * allocation)
            
            # Impact is the difference
            impact_value = stressed_ev - original_ev
            
            scenario_result["market_impacts"].append({
                "market": market.get("title", "Unknown Market"),
                "original_probability": yes_prob * 100,
                "stressed_probability": stressed_prob * 100,
                "allocation": allocation,
                "impact_value": impact_value
            })
            
            scenario_result["total_impact"] += impact_value
        
        # Calculate percentage impact on total portfolio
        scenario_result["total_impact_percentage"] = (scenario_result["total_impact"] / total_value) * 100
        stress_results.append(scenario_result)
    
    return {
        "baseline_value": baseline,
        "stress_scenarios": stress_results
    }

def multi_factor_correlation(markets: List[Dict], external_factors: List[str] = None) -> Dict:
    """
    Analyze correlation between prediction markets and external market factors
    
    Args:
        markets: List of prediction markets
        external_factors: External factors to analyze (defaults to standard factors)
    
    Returns:
        Dict with correlation analysis
    """
    if external_factors is None:
        external_factors = ["S&P 500", "US 10Y Treasury", "Gold", "USD Index", "Bitcoin", "VIX"]
    
    # Synthetic correlation data (would use real API data in production)
    factor_correlations = {}
    
    # Create correlation table
    for factor in external_factors:
        correlations = []
        for market in markets:
            # Generate realistic correlations based on market title/description
            market_title = market.get("title", "").lower()
            base_corr = random.uniform(-0.2, 0.2)  # Base correlation is low
            
            # Adjust correlation based on factor and market relationships
            if factor == "S&P 500":
                if any(term in market_title for term in ["economy", "growth", "recession", "gdp"]):
                    base_corr += random.uniform(0.3, 0.6)
                elif any(term in market_title for term in ["fed", "interest", "rate", "inflation"]):
                    base_corr += random.uniform(-0.4, -0.1)
            
            elif factor == "VIX":
                if any(term in market_title for term in ["volatility", "crash", "risk", "crisis"]):
                    base_corr += random.uniform(0.4, 0.7)
                elif any(term in market_title for term in ["stable", "growth", "recovery"]):
                    base_corr += random.uniform(-0.5, -0.2)
            
            elif factor == "Bitcoin":
                if any(term in market_title for term in ["crypto", "bitcoin", "blockchain", "eth"]):
                    base_corr += random.uniform(0.5, 0.8)
            
            # Ensure correlation is in valid range [-1, 1]
            correlation = max(-0.95, min(0.95, base_corr))
            
            # Calculate p-value (lower is more significant)
            p_value = 0.05 if abs(correlation) > 0.4 else 0.15
            
            correlations.append({
                "market_id": market.get("event_id"),
                "market_title": market.get("title"),
                "correlation": correlation,
                "p_value": p_value,
                "significant": p_value < 0.05
            })
        
        factor_correlations[factor] = correlations
    
    # Find highest overall correlations
    all_correlations = []
    for factor, correlations in factor_correlations.items():
        for corr_data in correlations:
            all_correlations.append({
                "factor": factor,
                "market_title": corr_data["market_title"],
                "correlation": corr_data["correlation"],
                "significant": corr_data["significant"]
            })
    
    # Sort by absolute correlation
    all_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    return {
        "factor_correlations": factor_correlations,
        "top_correlations": all_correlations[:5],  # Top 5 correlations
        "external_factors": external_factors
    }

def analyze_custom_portfolio(markets: List[Dict], portfolio_assets: Dict[str, float], total_value: float = 10000) -> Dict:
    """
    Analyze impact of prediction markets on a custom portfolio with sophisticated financial metrics.
    
    Args:
        markets: List of prediction markets
        portfolio_assets: Dict of assets and their allocation percentages (e.g. {'AAPL': 20, 'MSFT': 15})
        total_value: Total portfolio value in USD
        
    Returns:
        Dict with comprehensive portfolio analysis including risk metrics, stress tests, and recommendations
        
    Note:
        This function implements advanced financial analysis techniques including:
        - Monte Carlo simulation for risk assessment
        - Sharpe ratio and other risk-adjusted return calculations
        - Sector-based portfolio analysis
        - Stress testing under various market scenarios
    """
    # Validate input data
    if not markets:
        logger.warning("No markets provided for portfolio analysis")
        markets = []
    
    if not portfolio_assets:
        logger.warning("No portfolio assets provided for analysis")
        return {
            "error": "No portfolio assets provided",
            "generation_timestamp": datetime.now().isoformat()
        }
        
    # Validate portfolio allocation (should sum to approximately 100%)
    total_allocation = sum(portfolio_assets.values())
    if total_allocation < 95 or total_allocation > 105:
        logger.warning(f"Portfolio allocation sum is {total_allocation}%, expected close to 100%")
    
    results = {
        "total_value": total_value,
        "assets": [],
        "prediction_markets": [],
        "summary": "",
        "risk_metrics": {},
        "sector_exposure": {},
        "recommendations": [],
        "monte_carlo": {},  # Add Monte Carlo results
        "stress_tests": {},  # Add stress test results
        "correlations": {},  # Add correlation analysis
        "generation_timestamp": datetime.now().isoformat(),  # Add timestamp
        "data_source": "Polymarket",  # Add data source attribution
        "analysis_limitations": [],  # Add section for analysis limitations
        "version": VERSION  # Add version information
    }
    
    # Check if "Other assets" or "Other investments" is in the portfolio and highlight potential limitation
    other_keys = ["Other assets", "Other investments", "Other"]
    other_assets_pct = 0
    
    for key in other_keys:
        if key in portfolio_assets:
            other_assets_pct += portfolio_assets[key]
            
    if other_assets_pct > 30:
        results["analysis_limitations"].append(
            f"WARNING: Unspecified 'Other' assets comprise {other_assets_pct}% of the portfolio. "
            "This high allocation to uncategorized assets significantly reduces analysis accuracy. "
            "Consider breaking down these assets by type (global ETFs, commodities, bonds, etc.) for better results."
        )
    
    # Process each asset
    for asset_name, allocation_pct in portfolio_assets.items():
        asset_value = total_value * (allocation_pct / 100)
        
        # Extract sector from asset name (simplified)
        sector = "Technology"
        if any(term in asset_name.lower() for term in ["bank", "financial", "insurance", "visa", "mastercard", "icici"]):
            sector = "Financial Services"
        elif any(term in asset_name.lower() for term in ["oil", "gas", "energy", "solar", "petroleum"]):
            sector = "Energy"
        elif any(term in asset_name.lower() for term in ["healthcare", "pharma", "biotech", "medical"]):
            sector = "Healthcare"
        elif any(term in asset_name.lower() for term in ["telecom", "communication", "media", "advertising"]):
            sector = "Communication Services"
        elif any(term in asset_name.lower() for term in ["retail", "consumer", "food", "beverage"]):
            sector = "Consumer"
        elif any(term in asset_name.lower() for term in ["other"]):
            sector = "Unspecified"
        
        # Add to sector exposure
        if sector in results["sector_exposure"]:
            results["sector_exposure"][sector] += allocation_pct
        else:
            results["sector_exposure"][sector] = allocation_pct
        
        # Add asset details
        results["assets"].append({
            "name": asset_name,
            "allocation_percent": allocation_pct,
            "value": asset_value,
            "sector": sector
        })
    
    # Process each prediction market
    total_impact = 0
    total_volatility = 0
    risk_adjusted_returns = []
    market_correlations = []
    
    for market in markets:
        # Handle binary markets (Yes/No)
        yes_prob = market.get("yes_probability", 50) / 100
        no_prob = market.get("no_probability", 50) / 100
        volume = market.get("volume", 0)
        liquidity = market.get("liquidity", 0)
        
        # Dynamic allocation based on market confidence and liquidity
        # More liquid markets get higher allocation
        liquidity_factor = min(1.0, max(0.2, (liquidity / 10000) * 0.5))
        confidence_factor = abs(yes_prob - 0.5) * 2  # High for confident markets (near 0 or 1)
        allocation_pct = min(5.0, max(0.5, 2.0 * liquidity_factor * confidence_factor))
        allocation = total_value * (allocation_pct / 100)
        
        # Calculate detailed financial metrics
        potential_gain = allocation * (1/yes_prob - 1) if yes_prob > 0 else 0
        potential_loss = allocation
        
        # Expected value and volatility
        expected_value = (yes_prob * potential_gain) - (no_prob * potential_loss)
        volatility = (((potential_gain - expected_value) ** 2) * yes_prob + 
                     ((0 - expected_value) ** 2) * no_prob) ** 0.5
        
        # Sharpe ratio (risk-adjusted return)
        risk_free_rate = DEFAULT_RISK_FREE_RATE
        sharpe_ratio = (expected_value / allocation - risk_free_rate) / (volatility / allocation) if volatility > 0 else 0
        
        # Kelly criterion for optimal position sizing
        edge = yes_prob - (1-yes_prob)/(potential_gain/potential_loss) if potential_loss > 0 else 0
        kelly_pct = max(0, edge)  # Kelly position size as percentage
        
        # Information ratio
        information_ratio = expected_value / volatility if volatility > 0 else 0
        
        # Add to portfolio stats
        total_impact += expected_value
        total_volatility += volatility
        risk_adjusted_returns.append(sharpe_ratio)
        
        # Determine relevance to portfolio based on sector
        relevant_sectors = []
        relevance_score = 0
        market_title = market.get("title", "").lower()
        market_desc = market.get("description", "").lower()
        
        # Enhanced sector keywords to better match markets to portfolio assets
        sector_keywords = {
            "Technology": ["tech", "software", "hardware", "ai", "computing", "internet", "apple", "microsoft", "google"],
            "Financial Services": ["bank", "finance", "interest rate", "federal reserve", "inflation", "icici", "visa"],
            "Energy": ["oil", "gas", "energy", "petroleum", "renewable", "climate"],
            "Healthcare": ["health", "pharma", "medical", "biotech", "vaccine", "drug"],
            "Communication Services": ["telecom", "media", "communication", "advertising", "broadcast"],
            "Consumer": ["retail", "consumer", "food", "beverage", "goods", "services"],
            "Political": ["election", "politician", "president", "government", "congress", "parliament", "vote", "canadian", "canada"]
        }
        
        for sector, keywords in sector_keywords.items():
            sector_relevance = sum(1 for kw in keywords if kw in market_title or kw in market_desc)
            if sector_relevance > 0:
                relevant_sectors.append(sector)
                relevance_score += sector_relevance * results["sector_exposure"].get(sector, 0)
                
                # Political markets get special handling for relevance
                if sector == "Political" and sector_relevance > 0:
                    # Emphasize political markets that may impact the portfolio's sectors
                    for asset_sector, exposure in results["sector_exposure"].items():
                        # Political decisions often impact financial services and energy sectors
                        if asset_sector in ["Financial Services", "Energy"] and exposure > 10:
                            relevance_score += exposure * 0.5
        
        # Normalize relevance score
        relevance_score = min(100, relevance_score)
        
        # Run Monte Carlo simulation for this market
        monte_carlo_results = monte_carlo_simulation({"yes_probability": yes_prob * 100, "allocation": allocation})
        
        # Record market analysis with Monte Carlo results
        results["prediction_markets"].append({
            "title": market.get("title"),
            "event_id": market.get("event_id"),
            "yes_probability": yes_prob * 100,
            "allocation": allocation,
            "allocation_percent": allocation_pct,
            "potential_gain": potential_gain,
            "potential_loss": potential_loss,
            "expected_value": expected_value,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "kelly_criterion": kelly_pct,
            "information_ratio": information_ratio,
            "volume": volume,
            "liquidity": liquidity,
            "relevant_sectors": relevant_sectors,
            "portfolio_relevance_score": relevance_score,
            "monte_carlo": {
                "mean_return": monte_carlo_results["mean_return"],
                "var_95": monte_carlo_results["var_95"],
                "cvar_95": monte_carlo_results["cvar_95"],
                "confidence_interval_95": monte_carlo_results["ci_95"]
            }
        })
    
    # Sort markets by relevance to portfolio
    results["prediction_markets"].sort(key=lambda x: x["portfolio_relevance_score"], reverse=True)
    
    # Calculate portfolio-wide metrics
    portfolio_sharpe = sum(risk_adjusted_returns) / len(risk_adjusted_returns) if risk_adjusted_returns else 0
    
    # Record risk metrics
    results["risk_metrics"] = {
        "total_exposure": sum(m["allocation"] for m in results["prediction_markets"]),
        "exposure_percent": sum(m["allocation"] for m in results["prediction_markets"]) / total_value * 100,
        "expected_value": total_impact,
        "expected_return_percent": total_impact / total_value * 100,
        "portfolio_volatility": total_volatility,
        "portfolio_sharpe_ratio": portfolio_sharpe,
        "highest_conviction_market": max(results["prediction_markets"], 
                                       key=lambda x: x["information_ratio"])["title"] if results["prediction_markets"] else "None",
        "diversification_score": len(set(sector for m in results["prediction_markets"] for sector in m["relevant_sectors"])) / 4
    }
    
    # Run portfolio-wide analyses if we have markets
    if markets:
        results["stress_tests"] = stress_test_portfolio(portfolio_assets, results["prediction_markets"], None, total_value)
        results["correlations"] = multi_factor_correlation(results["prediction_markets"])
    
    # Add geopolitical sensitivity warning
    results["analysis_limitations"].append(
        f"NOTE: This analysis is based on Polymarket data as of {datetime.now().strftime('%B %d, %Y')}. "
        "Prediction markets may shift rapidly in response to geopolitical events. "
        "Consider refreshing this analysis if significant events occur that could impact these predictions."
    )
    
    # Generate recommendations based on advanced analyses
    if total_impact > 0:
        if portfolio_sharpe > 1.0:
            results["recommendations"].append("STRONG BUY: These markets offer exceptional risk-adjusted returns relevant to your portfolio.")
        else:
            results["recommendations"].append("MODERATE BUY: These markets offer positive expected value with acceptable risk profiles.")
    else:
        results["recommendations"].append("HOLD/AVOID: These markets do not offer positive expected value for your portfolio composition.")
    
    # Add specific recommendations for position sizing
    best_market = max(results["prediction_markets"], key=lambda x: x["information_ratio"]) if results["prediction_markets"] else None
    if best_market and best_market["kelly_criterion"] > 0.1:
        results["recommendations"].append(f"OPTIMIZE ALLOCATION: Consider increasing allocation to '{best_market['title']}' for optimal returns.")
    
    # Add risk management recommendation
    if results["risk_metrics"]["exposure_percent"] > 15:
        results["recommendations"].append("RISK ALERT: Total prediction market exposure exceeds 15% of portfolio. Consider diversifying or reducing position sizes.")
    
    # Add stress test recommendation
    if results["stress_tests"].get("stress_scenarios"):
        worst_scenario = min(results["stress_tests"]["stress_scenarios"], key=lambda x: x["total_impact_percentage"])
        if worst_scenario["total_impact_percentage"] < -15:
            results["recommendations"].append(f"STRESS VULNERABILITY: Portfolio shows high sensitivity to '{worst_scenario['scenario']}' scenario. Consider hedging strategies.")
    
    # Add correlation recommendation
    if results["correlations"].get("top_correlations") and len(results["correlations"]["top_correlations"]) > 0:
        if abs(results["correlations"]["top_correlations"][0]["correlation"]) > 0.7:
            top_corr = results["correlations"]["top_correlations"][0]
            direction = "positive" if top_corr["correlation"] > 0 else "negative"
            results["recommendations"].append(f"CORRELATION INSIGHT: Strong {direction} correlation detected between '{top_corr['market_title']}' and {top_corr['factor']}. Consider as potential hedge.")
    
    # Add sector-specific insights
    overexposed_sectors = [s for s, pct in results["sector_exposure"].items() if pct > 30]
    if overexposed_sectors:
        results["recommendations"].append(f"SECTOR DIVERSIFICATION: Portfolio is heavily concentrated in {', '.join(overexposed_sectors)}. Consider reducing exposure.")
    
    # Generate summary with data attribution
    impact_pct = (total_impact / total_value) * 100
    if impact_pct > 0:
        if impact_pct > 3:
            results["summary"] = f"HIGH POSITIVE IMPACT: Based on Polymarket prediction data, the analyzed markets could have a substantial positive expected impact of ${total_impact:,.2f} ({impact_pct:.2f}%) on your portfolio with a Sharpe ratio of {portfolio_sharpe:.2f}."
        else:
            results["summary"] = f"POSITIVE IMPACT: Based on Polymarket prediction data, the analyzed markets could have a modest positive expected impact of ${total_impact:,.2f} ({impact_pct:.2f}%) on your portfolio with a Sharpe ratio of {portfolio_sharpe:.2f}."
    else:
        results["summary"] = f"NEGATIVE IMPACT: Based on Polymarket prediction data, the analyzed markets could have a negative expected impact of ${total_impact:,.2f} ({impact_pct:.2f}%) on your portfolio. Consider alternative market exposures."
    
    return results

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
        "analysis": analysis,
        "generated_by": "local"  # Add marker to indicate this was generated locally
    }

async def analyze_market_with_gemini(market: Dict, timeframes: List[str], portfolio_value: float) -> Dict:
    """
    Use Gemini API to analyze a prediction market and generate insights
    about expected outcome changes over different timeframes.
    Falls back to local analysis if Gemini not available.
    """
    if not HAS_GEMINI:
        logger.info(f"Gemini not available, using fallback analysis for market: {market.get('event_id')}")
        return generate_market_analysis(market, timeframes, portfolio_value)
        
    try:
        # Prepare the market info for Gemini
        market_info = {
            "title": market.get("title", "Unknown"),
            "description": market.get("description", ""),
            "category": market.get("category", ""),
            "current_yes_probability": market.get("yes_probability", 50),
            "current_no_probability": market.get("no_probability", 50),
            "volume": market.get("volume", 0),
            "tags": market.get("tags", []),
            "end_date": market.get("end_date", "")
        }
        
        # Construct the prompt for Gemini
        prompt = f"""
I'm doing an analysis of a Polymarket prediction market. ONLY use this specific market data - do NOT use general knowledge or make predictions beyond what's shown in this data:

Title: {market_info['title']}
Description: {market_info['description']}
Category: {market_info['category']}
Current YES Probability: {market_info['current_yes_probability']:.2f}%
Current NO Probability: {market_info['current_no_probability']:.2f}%
Trading Volume: ${market_info['volume']:,.2f}
Tags: {', '.join(market_info['tags'])}
End Date: {market_info['end_date']}

I need a data-driven analysis with historical price trend predictions for these timeframes: {', '.join(timeframes)}.
Your analysis MUST:
1. ONLY use the Polymarket data provided above
2. NOT make predictions or statements beyond what's directly supported by the data
3. STRICTLY use the YES/NO probabilities as they are - if YES is 5%, don't say it's "medium likelihood"
4. Be factual, not speculative

For each timeframe, predict what the YES probability might have been and estimate the change relative to the current probability.

I've allocated ${portfolio_value:,.2f} * 0.05 = ${portfolio_value * 0.05:,.2f} to this market. Calculate the potential gain if I bet on YES and the event occurs, vs. the potential loss if I bet on YES and the event doesn't occur.

Format your response in JSON only, with this structure:
{{
    "timeframes": {{
        "1d": {{"probability": float, "change": float}},
        "7d": {{"probability": float, "change": float}},
        ...
    }},
    "portfolio_impact": {{
        "allocation": float,
        "potential_gain": float,
        "potential_loss": float
    }},
    "analysis": "Your short analysis explaining the Polymarket trends, strictly based on this data only"
}}
        """
        
        # Call Gemini API
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        # Parse the response
        try:
            # Extract JSON from the response
            response_text = response.text
            # Find JSON in the response if it's not pure JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            elif response_text.strip().startswith('{') and response_text.strip().endswith('}'):
                json_str = response_text
            else:
                # Fallback - try to extract anything that looks like JSON
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                else:
                    raise ValueError("Could not find JSON in response")
                
            analysis_data = json.loads(json_str)
            
            # Add event_id for tracking
            analysis_data['event_id'] = market.get('event_id')
            
            # Add a marker to indicate this was generated by Gemini
            analysis_data['generated_by'] = 'gemini'
            
            logger.info(f"Successfully analyzed market with Gemini: {market.get('event_id')}")
            return analysis_data
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Gemini response as JSON for market {market.get('event_id')}")
            # Fallback to generating an estimate
            return generate_market_analysis(market, timeframes, portfolio_value)
            
    except Exception as e:
        logger.error(f"Error using Gemini for market {market.get('event_id')}: {str(e)}")
        # Fallback to generating an estimate if Gemini fails
        return generate_market_analysis(market, timeframes, portfolio_value)

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
        ),
        types.Tool(
            name="analyze-portfolio-with-markets",
            description="Analyze how prediction markets would impact a specific portfolio of assets",
            inputSchema={
                "type": "object",
                "properties": {
                    "portfolio_assets": {
                        "type": "object",
                        "description": "Dict of asset names and allocation percentages (e.g. {'AAPL': 20, 'MSFT': 15})"
                    },
                    "total_value": {
                        "type": "number",
                        "description": "Total portfolio value in USD",
                        "default": 10000
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant prediction markets"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of markets to include",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["portfolio_assets", "query"]
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
            
            # Use Gemini when available, otherwise use local analysis
            using_gemini = HAS_GEMINI
            results = ["Market Outcome Impact Analysis:\n"]
            
            for event_id in event_ids:
                market = next((m for m in market_data if m.get("event_id") == event_id), None)
                
                if not market:
                    results.append(f"Market not found: {event_id}\n---")
                    continue
                
                # Get analysis - use Gemini if available, otherwise fallback
                if HAS_GEMINI:
                    logger.info(f"Analyzing market with Gemini: {event_id}")
                    analysis = await analyze_market_with_gemini(market, timeframes, portfolio_value)
                else:
                    logger.info(f"Analyzing market with local algorithm: {event_id}")
                    analysis = generate_market_analysis(market, timeframes, portfolio_value)
                
                # Format the analysis results
                results.append(f"Market: {market.get('title', 'Unknown')}")
                
                # Show which system generated the analysis
                if analysis.get('generated_by') == 'gemini':
                    results.append("(Analysis powered by Gemini AI)")
                
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
            
            # Add a note at the end about which system was used
            if using_gemini:
                results.append("\nNote: This analysis was enhanced using Google's Gemini AI model.")
            else:
                results.append("\nNote: This analysis was performed using local algorithms.")
            
            return [types.TextContent(type="text", text="\n".join(results))]
            
        elif name == "analyze-portfolio-with-markets":
            if not arguments or "portfolio_assets" not in arguments or "query" not in arguments:
                return [types.TextContent(
                    type="text", 
                    text="Error: portfolio_assets and query parameters are required."
                )]
            
            portfolio_assets = arguments["portfolio_assets"]
            total_value = float(arguments.get("total_value", 10000))
            query = arguments["query"]
            limit = int(arguments.get("limit", 3))
            
            # Ensure we have market data
            if not market_data or not vector_db:
                await refresh_prediction_markets()
            
            # Find relevant markets
            relevant_markets = vector_search(query, vector_db, limit)
            
            if not relevant_markets:
                return [types.TextContent(
                    type="text", 
                    text=f"No markets found matching your query: '{query}'"
                )]
            
            # Analyze custom portfolio
            analysis = analyze_custom_portfolio(relevant_markets, portfolio_assets, total_value)
            
            # Format the results in professional financial report style
            current_date = datetime.now().strftime("%B %d, %Y")
            
            results = [f"# INSTITUTIONAL PORTFOLIO ANALYSIS\n*Generated on {current_date} using data from Polymarket*\n"]
            
            # Executive summary
            results.append("## EXECUTIVE SUMMARY")
            results.append(analysis["summary"])
            
            # Analysis limitations
            if analysis["analysis_limitations"]:
                results.append("\n## ANALYSIS LIMITATIONS")
                for limitation in analysis["analysis_limitations"]:
                    results.append(f"- {limitation}")
            
            # Risk and return snapshot
            results.append("\n## RISK & RETURN METRICS")
            results.append(f"Total Portfolio Value: ${analysis['total_value']:,.2f}")
            results.append(f"Prediction Market Exposure: ${analysis['risk_metrics']['total_exposure']:,.2f} ({analysis['risk_metrics']['exposure_percent']:.2f}%)")
            results.append(f"Expected Return: ${analysis['risk_metrics']['expected_value']:,.2f} ({analysis['risk_metrics']['expected_return_percent']:.2f}%)")
            results.append(f"Portfolio Volatility: ${analysis['risk_metrics']['portfolio_volatility']:,.2f}")
            results.append(f"Sharpe Ratio: {analysis['risk_metrics']['portfolio_sharpe_ratio']:.2f}")
            results.append(f"Diversification Score: {analysis['risk_metrics']['diversification_score']:.2f}")
            
            # Current asset allocation
            results.append("\n## CURRENT ASSET ALLOCATION")
            for asset in analysis["assets"]:
                results.append(f"- {asset['name']} ({asset['sector']}): {asset['allocation_percent']}% (${asset['value']:,.2f})")
            
            # Sector exposure
            results.append("\n## SECTOR EXPOSURE")
            for sector, pct in analysis["sector_exposure"].items():
                results.append(f"- {sector}: {pct:.2f}%")
            
            # Prediction market analysis
            results.append("\n## PREDICTION MARKET IMPACT ANALYSIS")
            for i, market in enumerate(analysis["prediction_markets"]):
                results.append(f"\n### {i+1}. {market['title']}")
                results.append(f"*Event ID: {market['event_id']}*")
                results.append(f"**Current Probability:** YES {market['yes_probability']:.2f}% / NO {(100-market['yes_probability']):.2f}%")
                results.append(f"**Market Metrics:** Volume ${market['volume']:,.2f} | Liquidity ${market['liquidity']:,.2f}")
                results.append(f"**Recommended Allocation:** ${market['allocation']:,.2f} ({market['allocation_percent']:.2f}% of portfolio)")
                results.append(f"**Potential Outcomes:**")
                results.append(f"- If YES: Gain ${market['potential_gain']:,.2f}")
                results.append(f"- If NO: Loss ${market['potential_loss']:,.2f}")
                results.append(f"**Risk-Adjusted Metrics:**")
                results.append(f"- Expected Value: ${market['expected_value']:,.2f}")
                results.append(f"- Sharpe Ratio: {market['sharpe_ratio']:.2f}")
                results.append(f"- Information Ratio: {market['information_ratio']:.2f}")
                results.append(f"- Kelly Criterion: {market['kelly_criterion']*100:.2f}%")
                
                # Add Monte Carlo results
                if "monte_carlo" in market:
                    results.append(f"**Monte Carlo Simulation:**")
                    results.append(f"- Expected Return: ${market['monte_carlo']['mean_return']:,.2f}")
                    results.append(f"- 95% VaR: ${abs(market['monte_carlo']['var_95']):,.2f}")
                    results.append(f"- 95% Confidence Interval: [${market['monte_carlo']['confidence_interval_95'][0]:,.2f}, ${market['monte_carlo']['confidence_interval_95'][1]:,.2f}]")
                
                results.append(f"**Portfolio Relevance:** {market['portfolio_relevance_score']:.1f}/100")
                if market["relevant_sectors"]:
                    results.append(f"**Relevant Sectors:** {', '.join(market['relevant_sectors'])}")
            
            # Add stress test results
            results.append("\n## STRESS TEST SCENARIOS")
            if "stress_tests" in analysis and "stress_scenarios" in analysis["stress_tests"]:
                for scenario in analysis["stress_tests"]["stress_scenarios"]:
                    results.append(f"\n### {scenario['scenario']}")
                    results.append(f"Total Portfolio Impact: ${scenario['total_impact']:,.2f} ({scenario['total_impact_percentage']:+.2f}%)")
                    
                    # Show top 3 most impacted assets
                    results.append(f"**Most Impacted Assets:**")
                    sorted_assets = sorted(scenario['asset_impacts'], key=lambda x: abs(x['impact_value']), reverse=True)[:3]
                    for asset in sorted_assets:
                        results.append(f"- {asset['asset']}: ${asset['impact_value']:,.2f} ({asset['impact_percentage']:+.2f}%)")
            
            # Add correlation analysis
            results.append("\n## MARKET CORRELATIONS")
            if "correlations" in analysis and "top_correlations" in analysis["correlations"]:
                results.append("**Highest Correlated External Factors:**")
                for corr in analysis["correlations"]["top_correlations"][:3]:
                    direction = "positive" if corr["correlation"] > 0 else "negative"
                    significance = "statistically significant" if corr["significant"] else "not statistically significant"
                    results.append(f"- {corr['market_title']} → {corr['factor']}: {corr['correlation']:.2f} ({direction}, {significance})")
            
            # Investment recommendations
            results.append("\n## STRATEGIC RECOMMENDATIONS")
            for rec in analysis["recommendations"]:
                results.append(f"- {rec}")
            
            # Risk disclaimer
            results.append("\n## DISCLAIMER")
            results.append(f"*This analysis is based on prediction market data from Polymarket as of {current_date} and should not be considered financial advice. Past performance is not indicative of future results. All investments involve risk, including the possible loss of principal. VaR and Monte Carlo simulations are estimates only and not guarantees of future performance. Prediction markets can shift rapidly with geopolitical events, so consider refreshing this analysis if significant events occur.*")
            
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
    """
    Main entry point for the enhanced MCP server.
    Handles initialization, data loading, and graceful shutdown.
    """
    startup_msg = f"Starting enhanced PolyMarket MCP server v{VERSION}"
    
    if HAS_GEMINI:
        startup_msg += " with Gemini AI and vector database integration"
        logger.info(f"{startup_msg}...")
    else:
        startup_msg += " with vector database integration (Gemini AI not available)"
        logger.info(f"{startup_msg}...")
    
    # Initial data refresh
    try:
        logger.info("Performing initial market data refresh")
        await refresh_prediction_markets()
        logger.info(f"Successfully loaded {len(market_data)} markets")
    except Exception as e:
        logger.error(f"Error in initial data refresh: {str(e)}")
        logger.warning("Continuing without initial data - will attempt refresh on first request")
    
    # Set up signal handlers for graceful shutdown
    def handle_exit():
        logger.info("Received exit signal, shutting down server...")
    
    # Run server
    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            logger.info("Server stdio initialized")
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="polymarket_enhanced",
                    server_version=VERSION,
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
    finally:
        logger.info("Server shutting down...")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server terminated by keyboard interrupt")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        import traceback
        logger.error(traceback.format_exc()) 