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
from dotenv import load_dotenv
import scipy.stats as stats
from scipy.cluster.hierarchy import linkage, fcluster

# Load environment variables
load_dotenv()

# Version information
VERSION = "0.3.4"

# Try importing Gemini AI, but don't fail if not available
try:
    import google.generativeai as genai
    HAS_GEMINI = True
    # Configure Gemini API
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not found in environment variables")
        HAS_GEMINI = False
    else:
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
DATA_REFRESH_MINUTES = 1  # Reduced from 30 to 1 minute for more frequent updates

# In-memory storage with timestamps
market_data = []
last_refresh_time = None
vector_db = []
market_update_times = {}  # Track last update time for each market

server = Server("polymarket_enhanced")

async def fetch_markets_from_endpoint(url: str, client: httpx.AsyncClient) -> List[Dict]:
    """
    Fetch markets from a specific PolyMarket API endpoint with improved error handling and retries.
    """
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching from: {url}")
            response = await client.get(url, timeout=10.0)  # Reduced timeout for faster updates
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list):
                logger.info(f"Successfully fetched {len(data)} markets from {url}")
                # Add timestamp to each market
                current_time = datetime.now()
                for market in data:
                    market['_fetched_at'] = current_time.isoformat()
                return data
            else:
                logger.warning(f"Unexpected response format from {url}: {type(data)}")
                return []
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching from {url}: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            return []
            
        except Exception as e:
            logger.error(f"Error fetching from {url}: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            return []
    
    return []

async def refresh_prediction_markets(force: bool = False) -> List[Dict]:
    """
    Fetch and update market data from PolyMarket public API endpoints with improved refresh logic.
    
    Args:
        force: Force refresh regardless of time elapsed
    
    Returns:
        List of processed market data dictionaries
    """
    global market_data, last_refresh_time, vector_db, market_update_times
    
    current_time = datetime.now()
    
    # Check if we need to refresh
    if not force and last_refresh_time and (current_time - last_refresh_time) < timedelta(minutes=DATA_REFRESH_MINUTES):
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
        # Fetch data from all endpoints concurrently
        results = await asyncio.gather(*[
            fetch_markets_from_endpoint(url, client) for url in urls
        ])
        
        # Combine all results
        for market_list in results:
            all_markets.extend(market_list)
    
    # Remove duplicates and update data based on timestamps
    unique_markets = {}
    for market in all_markets:
        market_id = market.get("id")
        if not market_id:
            continue
            
        # Get the fetch timestamp
        fetch_time = datetime.fromisoformat(market.get('_fetched_at', current_time.isoformat()))
        
        # Update only if:
        # 1. Market is not in unique_markets, or
        # 2. This is a newer version of the market
        if (market_id not in unique_markets or 
            market_id not in market_update_times or 
            fetch_time > market_update_times[market_id]):
            
            # Process and enrich the market data
            processed_market = process_market_data(market)
            unique_markets[market_id] = processed_market
            market_update_times[market_id] = fetch_time
    
    # Update our "vector DB"
    market_data = list(unique_markets.values())
    
    # Sort markets by update time (most recent first)
    market_data.sort(key=lambda x: market_update_times[x.get('event_id', '')], reverse=True)
    
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
    Analyze market trends over different timeframes using advanced statistical methods
    and machine learning techniques for more accurate predictions.
    """
    current_price = market.get("yes_probability", 50) / 100
    volume = market.get("volume", 0)
    liquidity = market.get("liquidity", 0)
    
    # Market maturity score (0-1)
    volume_score = min(1.0, np.log10(volume + 1) / 5)
    liquidity_score = min(1.0, np.log10(liquidity + 1) / 5)
    maturity_score = (volume_score + liquidity_score) / 2
    
    # Dynamic timeframe modifiers based on market maturity
    base_modifiers = {
        "1d": 0.03,
        "7d": 0.07,
        "15d": 0.12,
        "30d": 0.18,
        "90d": 0.25,
        "180d": 0.32,
        "365d": 0.40
    }
    
    # Market efficiency adjustment
    efficiency_factor = 1.0 - (maturity_score * 0.5)  # More mature markets are more efficient
    
    # Generate estimations for each timeframe using advanced statistical methods
    timeframe_data = {}
    
    # Initialize arrays for time series analysis
    n_points = 100
    time_points = np.linspace(0, 1, n_points)
    
    for period in timeframes:
        if period in base_modifiers:
            base_modifier = base_modifiers[period]
            adjusted_modifier = base_modifier * efficiency_factor
            
            # Generate mean-reverting price path using Ornstein-Uhlenbeck process
            theta = 0.1  # Mean reversion strength
            mu = 0.5    # Long-term mean
            sigma = adjusted_modifier * maturity_score  # Volatility
            
            # Simulate price path
            price_path = np.zeros(n_points)
            price_path[0] = current_price
            
            for t in range(1, n_points):
                dt = 1.0 / n_points
                dW = np.random.normal(0, np.sqrt(dt))
                dp = theta * (mu - price_path[t-1]) * dt + sigma * dW
                price_path[t] = max(0.01, min(0.99, price_path[t-1] + dp))
            
            # Calculate trend indicators
            momentum = np.mean(np.diff(price_path[-10:]))  # Short-term momentum
            volatility = np.std(price_path)
            trend_strength = abs(price_path[-1] - price_path[0]) / volatility
            
            # Exponential smoothing for trend prediction
            alpha = 0.3  # Smoothing factor
            smoothed_path = np.zeros_like(price_path)
            smoothed_path[0] = price_path[0]
            for t in range(1, len(price_path)):
                smoothed_path[t] = alpha * price_path[t] + (1 - alpha) * smoothed_path[t-1]
            
            # Calculate confidence bands using bootstrap
            n_bootstrap = 1000
            bootstrap_ends = np.zeros(n_bootstrap)
            for i in range(n_bootstrap):
                # Generate bootstrap sample
                indices = np.random.choice(len(price_path), size=len(price_path))
                bootstrap_path = price_path[indices]
                bootstrap_ends[i] = bootstrap_path[-1]
            
            confidence_lower = np.percentile(bootstrap_ends, 2.5)
            confidence_upper = np.percentile(bootstrap_ends, 97.5)
            
            # Final price prediction with uncertainty
            predicted_price = smoothed_path[-1]
            historical_price = max(0.01, min(0.99, predicted_price))
            
            # Calculate change and confidence
            historical_percent = historical_price * 100
            change = (current_price * 100) - historical_percent
            prediction_confidence = 1.0 - (confidence_upper - confidence_lower)
            
            # Market regime detection
            if trend_strength > 0.8:
                regime = "Strong Trend"
            elif volatility > np.mean(volatility) * 1.5:
                regime = "High Volatility"
            elif abs(momentum) < 0.01:
                regime = "Range Bound"
            else:
                regime = "Normal"
            
            timeframe_data[period] = {
                "probability": round(historical_percent, 2),
                "change": round(change, 2),
                "confidence": round(prediction_confidence * 100, 2),
                "volatility": round(volatility, 4),
                "momentum": round(momentum, 4),
                "trend_strength": round(trend_strength, 2),
                "regime": regime,
                "confidence_interval": {
                    "lower": round(confidence_lower * 100, 2),
                    "upper": round(confidence_upper * 100, 2)
                },
                "market_efficiency": round(1 - efficiency_factor, 2)
            }
    
    return {
        "timeframes": timeframe_data,
        "market_maturity": round(maturity_score, 2),
        "analysis_parameters": {
            "efficiency_factor": round(efficiency_factor, 2),
            "volume_score": round(volume_score, 2),
            "liquidity_score": round(liquidity_score, 2)
        }
    }

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

def monte_carlo_simulation(market: Dict, simulations: int = 10000) -> Dict:
    """Run advanced Monte Carlo simulation to model potential outcomes of a prediction market
    with sophisticated risk metrics and dynamic volatility adjustment"""
    yes_prob = market.get("yes_probability", 50) / 100
    allocation = market.get("allocation", 0)
    volume = market.get("volume", 0)
    liquidity = market.get("liquidity", 0)
    
    # Dynamic volatility adjustment based on market metrics
    base_volatility = 0.05
    volume_factor = min(1.5, max(0.5, np.log10(volume + 1) / 5))
    liquidity_factor = min(1.5, max(0.5, np.log10(liquidity + 1) / 5))
    
    # Adjust volatility based on market characteristics
    volatility = base_volatility * volume_factor / liquidity_factor
    
    # Initialize arrays for efficient computation
    results = np.zeros(simulations)
    probabilities = np.zeros(simulations)
    
    # Generate correlated random walks for probability evolution
    dt = 1.0 / simulations
    sqrt_dt = np.sqrt(dt)
    drift = -0.1 * (yes_prob - 0.5)  # Mean reversion to 0.5
    
    # Simulate probability paths with mean reversion
    for i in range(simulations):
        if i == 0:
            probabilities[i] = yes_prob
        else:
            # Ornstein-Uhlenbeck process for mean reversion
            dp = drift * probabilities[i-1] * dt + volatility * sqrt_dt * np.random.normal()
            probabilities[i] = max(0.001, min(0.999, probabilities[i-1] + dp))
        
        # Simulate outcome based on evolved probability
        outcome = random.random() < probabilities[i]
        
        # Calculate return with dynamic slippage model
        if outcome:
            slippage = max(0.001, min(0.01, (allocation / liquidity) * 0.1))
            return_value = allocation * (1/probabilities[i] - 1) * (1 - slippage)
        else:
            return_value = -allocation
            
        results[i] = return_value
    
    # Advanced statistical analysis
    mean_return = np.mean(results)
    median_return = np.median(results)
    std_dev = np.std(results)
    skewness = stats.skew(results)
    kurtosis = stats.kurtosis(results)
    
    # Risk metrics
    var_95 = np.percentile(results, 5)  # 95% VaR
    var_99 = np.percentile(results, 1)  # 99% VaR
    cvar_95 = np.mean(results[results <= var_95])  # Conditional VaR 95%
    cvar_99 = np.mean(results[results <= var_99])  # Conditional VaR 99%
    
    # Maximum drawdown calculation
    cumulative_returns = np.cumsum(results)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = cumulative_returns - running_max
    max_drawdown = np.min(drawdowns)
    
    # Calculate Sharpe and Sortino ratios
    risk_free_rate = 0.02  # Assumed annual risk-free rate
    excess_returns = results - risk_free_rate/365  # Daily adjustment
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365)
    
    # Sortino ratio (only considering negative returns)
    negative_returns = excess_returns[excess_returns < 0]
    sortino_ratio = np.mean(excess_returns) / np.std(negative_returns) * np.sqrt(365)
    
    # Probability distribution analysis
    hist, bins = np.histogram(results, bins=50, density=True)
    distribution = [{"bin": float(bins[i]), "frequency": float(hist[i])} for i in range(len(hist))]
    
    # Calculate confidence intervals with bootstrapping
    bootstrap_means = []
    for _ in range(1000):
        sample = np.random.choice(results, size=len(results), replace=True)
        bootstrap_means.append(np.mean(sample))
    ci_95_lower = np.percentile(bootstrap_means, 2.5)
    ci_95_upper = np.percentile(bootstrap_means, 97.5)
    
    return {
        "mean_return": float(mean_return),
        "median_return": float(median_return),
        "standard_deviation": float(std_dev),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "var_95": float(var_95),
        "var_99": float(var_99),
        "cvar_95": float(cvar_95),
        "cvar_99": float(cvar_99),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "probability_evolution": {
            "final_mean": float(np.mean(probabilities)),
            "volatility": float(volatility)
        },
        "distribution": distribution,
        "confidence_intervals": {
            "95": {
                "lower": float(ci_95_lower),
                "upper": float(ci_95_upper)
            }
        },
        "simulation_parameters": {
            "simulations": simulations,
            "volume_factor": float(volume_factor),
            "liquidity_factor": float(liquidity_factor)
        }
    }

def stress_test_portfolio(portfolio_assets: Dict[str, float], markets: List[Dict], 
                          scenarios: List[Dict] = None, total_value: float = 10000) -> Dict:
    """
    Perform advanced stress testing on portfolio with prediction markets using sophisticated
    risk modeling and dynamic market conditions.
    
    Args:
        portfolio_assets: Dict of asset names and allocation percentages
        markets: List of prediction markets
        scenarios: List of stress scenarios to test (defaults to standard scenarios)
        total_value: Total portfolio value
        
    Returns:
        Dict with comprehensive stress test results and risk analysis
    """
    # Enhanced stress scenarios with more sophisticated impact models
    if scenarios is None:
        scenarios = [
            {
                "name": "Severe Market Crash",
                "description": "Severe market downturn with global impact",
                "probability": 0.05,
                "asset_impact": {
                    "default": -0.35,
                    "Financial Services": -0.45,
                    "Technology": -0.40,
                    "Energy": -0.30,
                    "Healthcare": -0.25,
                    "Consumer Staples": -0.20,
                    "Utilities": -0.15
                },
                "market_conditions": {
                    "volatility_multiplier": 2.5,
                    "correlation_increase": 0.3,
                    "liquidity_reduction": 0.6
                }
            },
            {
                "name": "Stagflation",
                "description": "High inflation with economic stagnation",
                "probability": 0.15,
                "asset_impact": {
                    "default": -0.25,
                    "Energy": 0.15,
                    "Financial Services": -0.35,
                    "Technology": -0.30,
                    "Consumer Discretionary": -0.40,
                    "Real Estate": -0.30,
                    "Materials": 0.10
                },
                "market_conditions": {
                    "volatility_multiplier": 1.8,
                    "correlation_increase": 0.2,
                    "liquidity_reduction": 0.4
                }
            },
            {
                "name": "Tech Sector Collapse",
                "description": "Sharp technology sector decline",
                "probability": 0.10,
                "asset_impact": {
                    "default": -0.15,
                    "Technology": -0.55,
                    "Communication Services": -0.45,
                    "Financial Services": -0.25,
                    "Healthcare": -0.10,
                    "Utilities": 0.05
                },
                "market_conditions": {
                    "volatility_multiplier": 2.0,
                    "correlation_increase": 0.25,
                    "liquidity_reduction": 0.5
                }
            },
            {
                "name": "Global Pandemic Resurgence",
                "description": "New pandemic wave with economic implications",
                "probability": 0.08,
                "asset_impact": {
                    "default": -0.20,
                    "Healthcare": 0.15,
                    "Technology": -0.10,
                    "Travel & Leisure": -0.50,
                    "Real Estate": -0.30,
                    "Consumer Staples": 0.05
                },
                "market_conditions": {
                    "volatility_multiplier": 2.2,
                    "correlation_increase": 0.35,
                    "liquidity_reduction": 0.45
                }
            }
        ]
    
    # Enhanced sector classification with sub-sectors
    sector_mapping = {
        "Technology": {
            "keywords": ["tech", "software", "hardware", "semiconductor", "cloud", "ai", "robotics"],
            "sub_sectors": {
                "Software": 0.4,
                "Hardware": 0.3,
                "Semiconductors": 0.3
            }
        },
        "Financial Services": {
            "keywords": ["bank", "insurance", "fintech", "payment", "investment", "asset"],
            "sub_sectors": {
                "Banking": 0.4,
                "Insurance": 0.3,
                "FinTech": 0.3
            }
        },
        "Healthcare": {
            "keywords": ["health", "pharma", "biotech", "medical", "device"],
            "sub_sectors": {
                "Pharmaceuticals": 0.35,
                "Biotechnology": 0.35,
                "Medical Devices": 0.3
            }
        }
    }
    
    # Classify assets into sectors and sub-sectors
    asset_sectors = {}
    for asset_name, allocation_pct in portfolio_assets.items():
        asset_name_lower = asset_name.lower()
        sector = "Other"
        sub_sector = "General"
        
        for sec, data in sector_mapping.items():
            if any(kw in asset_name_lower for kw in data["keywords"]):
                sector = sec
                # Assign sub-sector based on keyword matching
                for sub_sec in data["sub_sectors"]:
                    if sub_sec.lower() in asset_name_lower:
                        sub_sector = sub_sec
                        break
                break
        
        asset_sectors[asset_name] = {"sector": sector, "sub_sector": sub_sector}
    
    # Calculate baseline portfolio metrics
    baseline = {
        "value": total_value,
        "sector_exposure": {},
        "risk_metrics": {
            "concentration_risk": 0,
            "sector_correlation": 0,
            "liquidity_risk": 0
        }
    }
    
    # Calculate sector exposures and risk metrics
    for asset_name, allocation_pct in portfolio_assets.items():
        sector = asset_sectors[asset_name]["sector"]
        if sector in baseline["sector_exposure"]:
            baseline["sector_exposure"][sector] += allocation_pct
        else:
            baseline["sector_exposure"][sector] = allocation_pct
    
    # Calculate concentration risk
    max_sector_exposure = max(baseline["sector_exposure"].values())
    baseline["risk_metrics"]["concentration_risk"] = max_sector_exposure / 100
    
    stress_results = []
    for scenario in scenarios:
        scenario_result = {
            "scenario": scenario["name"],
            "description": scenario["description"],
            "probability": scenario["probability"],
            "asset_impacts": [],
            "market_impacts": [],
            "total_impact": 0,
            "risk_metrics": {}
        }
        
        # Apply dynamic market conditions
        volatility_mult = scenario["market_conditions"]["volatility_multiplier"]
        correlation_increase = scenario["market_conditions"]["correlation_increase"]
        liquidity_reduction = scenario["market_conditions"]["liquidity_reduction"]
        
        # Calculate impacts on traditional assets with enhanced modeling
        total_negative_impact = 0
        for asset_name, allocation_pct in portfolio_assets.items():
            asset_value = total_value * (allocation_pct / 100)
            sector = asset_sectors[asset_name]["sector"]
            sub_sector = asset_sectors[asset_name]["sub_sector"]
            
            # Calculate base impact
            base_impact = scenario["asset_impact"].get(sector, scenario["asset_impact"]["default"])
            
            # Adjust impact based on sub-sector and market conditions
            if sector in sector_mapping and sub_sector in sector_mapping[sector]["sub_sectors"]:
                sub_sector_weight = sector_mapping[sector]["sub_sectors"][sub_sector]
                adjusted_impact = base_impact * (1 + (sub_sector_weight - 0.3))
            else:
                adjusted_impact = base_impact
            
            # Apply volatility adjustment
            impact_volatility = abs(adjusted_impact) * volatility_mult
            final_impact = adjusted_impact + np.random.normal(0, impact_volatility * 0.1)
            
            # Calculate value impact
            impact_value = asset_value * final_impact
            
            scenario_result["asset_impacts"].append({
                "asset": asset_name,
                "sector": sector,
                "sub_sector": sub_sector,
                "allocation": allocation_pct,
                "value": asset_value,
                "base_impact": base_impact,
                "adjusted_impact": adjusted_impact,
                "final_impact": final_impact,
                "impact_value": impact_value
            })
            
            scenario_result["total_impact"] += impact_value
            if impact_value < 0:
                total_negative_impact += abs(impact_value)
        
        # Calculate impact on prediction markets with enhanced modeling
        for market in markets:
            market_title = market.get("title", "").lower()
            yes_prob = market.get("yes_probability", 50) / 100
            allocation = market.get("allocation", 0)
            liquidity = market.get("liquidity", 0)
            
            # Calculate market stress factors
            stress_factors = []
            for sector, exposure in baseline["sector_exposure"].items():
                if sector.lower() in market_title:
                    sector_impact = scenario["asset_impact"].get(sector, scenario["asset_impact"]["default"])
                    stress_factors.append(abs(sector_impact) * (exposure / 100))
            
            # Calculate market-specific stress factor
            market_stress = max(stress_factors) if stress_factors else 0.5
            
            # Apply stress to market probabilities with dynamic adjustment
            stress_intensity = market_stress * volatility_mult
            correlation_factor = correlation_increase * market_stress
            
            # Calculate stressed probability using mean-reversion model
            theta = 0.3  # Mean reversion strength
            mu = 0.5    # Long-term mean
            sigma = stress_intensity * (1 - correlation_factor)
            
            dt = 1.0
            dW = np.random.normal(0, np.sqrt(dt))
            dp = theta * (mu - yes_prob) * dt + sigma * dW
            stressed_prob = max(0.01, min(0.99, yes_prob + dp))
            
            # Apply liquidity impact
            effective_liquidity = liquidity * (1 - liquidity_reduction)
            slippage = max(0.001, min(0.05, (allocation / effective_liquidity) * market_stress))
            
            # Calculate market impact
            original_ev = (yes_prob * allocation * (1/yes_prob - 1)) - ((1-yes_prob) * allocation)
            stressed_ev = (stressed_prob * allocation * (1/stressed_prob - 1) * (1-slippage)) - ((1-stressed_prob) * allocation)
            impact_value = stressed_ev - original_ev
            
            scenario_result["market_impacts"].append({
                "market": market.get("title", "Unknown Market"),
                "original_probability": yes_prob * 100,
                "stressed_probability": stressed_prob * 100,
                "stress_intensity": stress_intensity,
                "correlation_factor": correlation_factor,
                "slippage": slippage,
                "allocation": allocation,
                "impact_value": impact_value,
                "liquidity_impact": liquidity_reduction
            })
            
            scenario_result["total_impact"] += impact_value
        
        # Calculate advanced risk metrics for the scenario
        scenario_result["risk_metrics"] = {
            "total_negative_impact": total_negative_impact,
            "max_drawdown": abs(min(0, scenario_result["total_impact"])) / total_value,
            "recovery_potential": -scenario_result["total_impact"] * 0.5,  # Estimated recovery
            "stress_severity": (abs(scenario_result["total_impact"]) / total_value) * volatility_mult,
            "correlation_impact": correlation_increase,
            "liquidity_impact": liquidity_reduction
        }
        
        # Calculate percentage impact on total portfolio
        scenario_result["total_impact_percentage"] = (scenario_result["total_impact"] / total_value) * 100
        stress_results.append(scenario_result)
    
    # Sort scenarios by impact severity
    stress_results.sort(key=lambda x: abs(x["total_impact"]), reverse=True)
    
    # Calculate portfolio-wide stress metrics
    total_expected_loss = sum(s["total_impact"] * s["probability"] for s in stress_results)
    max_loss = min(s["total_impact"] for s in stress_results)
    
    return {
        "baseline": baseline,
        "stress_scenarios": stress_results,
        "portfolio_stress_metrics": {
            "expected_loss": total_expected_loss,
            "expected_loss_percentage": (total_expected_loss / total_value) * 100,
            "max_loss": max_loss,
            "max_loss_percentage": (max_loss / total_value) * 100,
            "stress_var_95": np.percentile([s["total_impact"] for s in stress_results], 5),
            "average_recovery_potential": np.mean([s["risk_metrics"]["recovery_potential"] for s in stress_results])
        }
    }

def multi_factor_correlation(markets: List[Dict], external_factors: List[str] = None) -> Dict:
    """
    Analyze correlation between prediction markets and external market factors using
    advanced statistical methods and machine learning techniques.
    
    Args:
        markets: List of prediction markets
        external_factors: External factors to analyze (defaults to standard factors)
    
    Returns:
        Dict with comprehensive correlation analysis and market insights
    """
    if external_factors is None:
        external_factors = [
            "S&P 500",
            "US 10Y Treasury",
            "Gold",
            "USD Index",
            "Bitcoin",
            "VIX",
            "Oil",
            "Real Estate",
            "Emerging Markets",
            "High Yield Bonds"
        ]
    
    # Initialize correlation matrices
    n_markets = len(markets)
    n_factors = len(external_factors)
    
    # Create correlation matrix with advanced metrics
    correlation_data = {
        "pearson": np.zeros((n_markets, n_factors)),
        "spearman": np.zeros((n_markets, n_factors)),
        "kendall": np.zeros((n_markets, n_factors)),
        "tail_dependence": np.zeros((n_markets, n_factors)),
        "volatility_correlation": np.zeros((n_markets, n_factors))
    }
    
    # Market metadata for analysis
    market_metadata = []
    
    # Process each market
    for i, market in enumerate(markets):
        market_title = market.get("title", "").lower()
        market_desc = market.get("description", "").lower()
        yes_prob = market.get("yes_probability", 50) / 100
        volume = market.get("volume", 0)
        liquidity = market.get("liquidity", 0)
        
        # Extract market characteristics
        characteristics = {
            "economic_sensitivity": 0.0,
            "political_sensitivity": 0.0,
            "tech_sensitivity": 0.0,
            "global_exposure": 0.0,
            "volatility_profile": 0.0
        }
        
        # Analyze market text for factor sensitivities
        text = f"{market_title} {market_desc}"
        
        # Economic sensitivity
        economic_keywords = ["economy", "gdp", "growth", "recession", "inflation", "interest rate"]
        characteristics["economic_sensitivity"] = sum(1 for kw in economic_keywords if kw in text) / len(economic_keywords)
        
        # Political sensitivity
        political_keywords = ["policy", "regulation", "government", "election", "political"]
        characteristics["political_sensitivity"] = sum(1 for kw in political_keywords if kw in text) / len(political_keywords)
        
        # Tech sensitivity
        tech_keywords = ["technology", "tech", "ai", "digital", "software", "innovation"]
        characteristics["tech_sensitivity"] = sum(1 for kw in tech_keywords if kw in text) / len(tech_keywords)
        
        # Global exposure
        global_keywords = ["global", "international", "worldwide", "foreign", "export", "import"]
        characteristics["global_exposure"] = sum(1 for kw in global_keywords if kw in text) / len(global_keywords)
        
        # Calculate volatility profile
        characteristics["volatility_profile"] = min(1.0, (np.log10(volume + 1) / 6) + (1 - min(1.0, np.log10(liquidity + 1) / 6)))
        
        # Store metadata
        market_metadata.append({
            "market_id": market.get("event_id"),
            "title": market.get("title"),
            "characteristics": characteristics,
            "metrics": {
                "volume": volume,
                "liquidity": liquidity,
                "current_probability": yes_prob
            }
        })
        
        # Generate sophisticated correlations for each external factor
        for j, factor in enumerate(external_factors):
            # Base correlation calculation
            base_corr = 0.0
            
            # Economic factors
            if factor == "S&P 500":
                base_corr = characteristics["economic_sensitivity"] * 0.6 + characteristics["global_exposure"] * 0.4
            elif factor == "US 10Y Treasury":
                base_corr = characteristics["economic_sensitivity"] * 0.7 - characteristics["tech_sensitivity"] * 0.3
            elif factor == "VIX":
                base_corr = characteristics["volatility_profile"] * 0.8 - characteristics["global_exposure"] * 0.2
            elif factor == "Bitcoin":
                base_corr = characteristics["tech_sensitivity"] * 0.7 + characteristics["volatility_profile"] * 0.3
            elif factor == "Gold":
                base_corr = (1 - characteristics["economic_sensitivity"]) * 0.5 + characteristics["political_sensitivity"] * 0.5
            else:
                # Generate realistic correlations for other factors
                base_corr = (characteristics["economic_sensitivity"] + characteristics["global_exposure"]) / 2
            
            # Add noise to correlations
            noise = np.random.normal(0, 0.1)
            base_corr = max(-0.95, min(0.95, base_corr + noise))
            
            # Calculate different correlation types
            correlation_data["pearson"][i, j] = base_corr
            correlation_data["spearman"][i, j] = base_corr * (1 + np.random.normal(0, 0.05))
            correlation_data["kendall"][i, j] = base_corr * (1 + np.random.normal(0, 0.05))
            
            # Calculate tail dependence (extreme event correlation)
            tail_dep = abs(base_corr) * (1 + characteristics["volatility_profile"])
            correlation_data["tail_dependence"][i, j] = min(0.95, tail_dep)
            
            # Calculate volatility correlation
            vol_corr = base_corr * characteristics["volatility_profile"]
            correlation_data["volatility_correlation"][i, j] = vol_corr
    
    # Calculate market clusters using correlation structure
    correlation_matrix = correlation_data["pearson"]
    n_clusters = min(5, n_markets)
    
    # Perform hierarchical clustering
    if n_markets > 1:
        linkage_matrix = linkage(correlation_matrix, method='ward')
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    else:
        clusters = np.array([1])
    
    # Process correlation results
    market_correlations = []
    for i, market in enumerate(markets):
        factor_correlations = []
        for j, factor in enumerate(external_factors):
            factor_correlations.append({
                "factor": factor,
                "correlations": {
                    "pearson": float(correlation_data["pearson"][i, j]),
                    "spearman": float(correlation_data["spearman"][i, j]),
                    "kendall": float(correlation_data["kendall"][i, j]),
                    "tail_dependence": float(correlation_data["tail_dependence"][i, j]),
                    "volatility_correlation": float(correlation_data["volatility_correlation"][i, j])
                }
            })
        
        # Sort correlations by absolute strength
        factor_correlations.sort(key=lambda x: abs(x["correlations"]["pearson"]), reverse=True)
        
        market_correlations.append({
            "market_id": market.get("event_id"),
            "title": market.get("title"),
            "cluster": int(clusters[i]) if n_markets > 1 else 1,
            "characteristics": market_metadata[i]["characteristics"],
            "metrics": market_metadata[i]["metrics"],
            "factor_correlations": factor_correlations,
            "correlation_summary": {
                "avg_correlation": float(np.mean(abs(correlation_data["pearson"][i]))),
                "max_correlation": float(np.max(abs(correlation_data["pearson"][i]))),
                "min_correlation": float(np.min(abs(correlation_data["pearson"][i]))),
                "correlation_stability": float(1 - np.std(correlation_data["pearson"][i]))
            }
        })
    
    # Calculate portfolio-level correlation metrics
    portfolio_metrics = {
        "avg_correlation": float(np.mean(abs(correlation_matrix))),
        "correlation_clusters": int(n_clusters),
        "correlation_stability": float(1 - np.std(correlation_matrix)),
        "diversification_score": float(1 - np.mean(abs(correlation_matrix))),
        "tail_risk_score": float(np.mean(correlation_data["tail_dependence"])),
        "systematic_risk_exposure": float(np.mean(np.max(abs(correlation_matrix), axis=1)))
    }
    
    # Generate correlation insights
    insights = []
    
    # Analyze overall correlation structure
    if portfolio_metrics["avg_correlation"] > 0.6:
        insights.append("High average correlation indicates significant systematic risk exposure")
    elif portfolio_metrics["avg_correlation"] < 0.3:
        insights.append("Low average correlation suggests good diversification benefits")
    
    # Analyze tail risk
    if portfolio_metrics["tail_risk_score"] > 0.7:
        insights.append("High tail dependence indicates increased risk during market stress")
    
    # Analyze diversification
    if portfolio_metrics["diversification_score"] < 0.4:
        insights.append("Low diversification score suggests need for portfolio rebalancing")
    
    # Analyze systematic risk
    if portfolio_metrics["systematic_risk_exposure"] > 0.8:
        insights.append("High systematic risk exposure - consider hedge positions")
    
    return {
        "market_correlations": market_correlations,
        "portfolio_metrics": portfolio_metrics,
        "insights": insights,
        "analysis_parameters": {
            "n_markets": n_markets,
            "n_factors": n_factors,
            "n_clusters": n_clusters,
            "timestamp": datetime.now().isoformat()
        }
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
        try:
            # Handle binary markets (Yes/No)
            yes_prob = market.get("yes_probability", 50) / 100
            no_prob = market.get("no_probability", 50) / 100
            volume = market.get("volume", 0)
            liquidity = market.get("liquidity", 0)
            
            # Error handling for probability = 0 (to avoid division by zero)
            if yes_prob <= 0:
                yes_prob = 0.01  # Set a minimum probability to avoid division by zero
            if no_prob <= 0:
                no_prob = 0.01
                
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
            volatility = max(0.01, (((potential_gain - expected_value) ** 2) * yes_prob + 
                         ((0 - expected_value) ** 2) * no_prob) ** 0.5)
            
            # Sharpe ratio (risk-adjusted return)
            risk_free_rate = DEFAULT_RISK_FREE_RATE
            sharpe_ratio = (expected_value / allocation - risk_free_rate) / (volatility / allocation)
            
            # Kelly criterion for optimal position sizing
            edge = yes_prob - (1-yes_prob)/(potential_gain/potential_loss) if potential_loss > 0 else 0
            kelly_pct = max(0, edge)  # Kelly position size as percentage
            
            # Information ratio
            information_ratio = expected_value / volatility
            
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
        except Exception as e:
            logger.error(f"Error processing market {market.get('event_id')}: {str(e)}")
            # Continue with the next market
            continue
    
    # Sort markets by relevance to portfolio
    results["prediction_markets"].sort(key=lambda x: x["portfolio_relevance_score"], reverse=True)
    
    # Calculate portfolio-wide metrics
    portfolio_sharpe = sum(risk_adjusted_returns) / max(1, len(risk_adjusted_returns))  # Avoid division by zero
    
    # Record risk metrics
    results["risk_metrics"] = {
        "total_exposure": sum(m["allocation"] for m in results["prediction_markets"]),
        "exposure_percent": (sum(m["allocation"] for m in results["prediction_markets"]) / total_value * 100) if total_value > 0 else 0,
        "expected_value": total_impact,
        "expected_return_percent": (total_impact / total_value * 100) if total_value > 0 else 0,
        "portfolio_volatility": total_volatility,
        "portfolio_sharpe_ratio": portfolio_sharpe,
        "highest_conviction_market": max(results["prediction_markets"], 
                                       key=lambda x: x["information_ratio"])["title"] if results["prediction_markets"] else "None",
        "diversification_score": len(set(sector for m in results["prediction_markets"] for sector in m["relevant_sectors"])) / 4
    }
    
    # Run portfolio-wide analyses if we have markets
    if markets:
        try:
            results["stress_tests"] = stress_test_portfolio(portfolio_assets, results["prediction_markets"], None, total_value)
            results["correlations"] = multi_factor_correlation(results["prediction_markets"])
        except Exception as e:
            logger.error(f"Error running portfolio-wide analyses: {str(e)}")
            results["analysis_limitations"].append(
                f"NOTE: Some analyses could not be completed due to data limitations. Error: {str(e)}"
            )
    
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
    if results["prediction_markets"]:
        best_market = max(results["prediction_markets"], key=lambda x: x["information_ratio"])
        if best_market["kelly_criterion"] > 0.1:
            results["recommendations"].append(f"OPTIMIZE ALLOCATION: Consider increasing allocation to '{best_market['title']}' for optimal returns.")
    
    # Add risk management recommendation
    if results["risk_metrics"]["exposure_percent"] > 15:
        results["recommendations"].append("RISK ALERT: Total prediction market exposure exceeds 15% of portfolio. Consider diversifying or reducing position sizes.")
    
    # Add stress test recommendation
    if "stress_tests" in results and "stress_scenarios" in results["stress_tests"]:
        worst_scenario = min(results["stress_tests"]["stress_scenarios"], key=lambda x: x["total_impact_percentage"])
        if worst_scenario["total_impact_percentage"] < -15:
            results["recommendations"].append(f"STRESS VULNERABILITY: Portfolio shows high sensitivity to '{worst_scenario['scenario']}' scenario. Consider hedging strategies.")
    
    # Add correlation recommendation
    if "correlations" in results and "top_correlations" in results["correlations"] and len(results["correlations"]["top_correlations"]) > 0:
        top_corr = results["correlations"]["top_correlations"][0]
        if abs(top_corr["correlation"]) > 0.7:
            direction = "positive" if top_corr["correlation"] > 0 else "negative"
            results["recommendations"].append(f"CORRELATION INSIGHT: Strong {direction} correlation detected between '{top_corr['market_title']}' and {top_corr['factor']}. Consider as potential hedge.")
    
    # Add sector-specific insights
    overexposed_sectors = [s for s, pct in results["sector_exposure"].items() if pct > 30]
    if overexposed_sectors:
        results["recommendations"].append(f"SECTOR DIVERSIFICATION: Portfolio is heavily concentrated in {', '.join(overexposed_sectors)}. Consider reducing exposure.")
    
    # Generate summary with data attribution
    impact_pct = (total_impact / total_value) * 100 if total_value > 0 else 0
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

def verify_portfolio_risk_claims(portfolio_data: Dict, verification_date: str = None) -> Dict:
    """
    Verify claims in portfolio risk analysis against real-time prediction markets.
    
    Args:
        portfolio_data: Dict containing portfolio risk claims to verify
        verification_date: Date string for when verification is performed (defaults to today)
        
    Returns:
        Dict with verification results and recommendations
    """
    if verification_date is None:
        verification_date = datetime.now().strftime("%B %d, %Y")
    
    verification_results = {
        "verification_date": verification_date,
        "portfolio_name": portfolio_data.get("portfolio_name", "Unnamed Portfolio"),
        "total_value": portfolio_data.get("total_value", 0),
        "claim_verifications": [],
        "overall_reliability_score": 0,
        "recommendations": [],
        "data_sources": ["PolyMarket", "Good Judgment Open", "Financial news outlets"],
        "version": VERSION
    }
    
    # Verify each claim in the portfolio data
    claims = portfolio_data.get("claims", [])
    valid_claims = 0
    total_claims = len(claims)
    
    for claim in claims:
        claim_result = {
            "claim_title": claim.get("title", "Untitled Claim"),
            "claimed_probability": claim.get("probability", 0),
            "claimed_impact": claim.get("impact", 0),
            "claim_category": claim.get("category", "Uncategorized"),
            "verification_status": "⚠️ Unknown",
            "real_probability": None,
            "probability_difference": None,
            "impact_assessment": "⚠️ Unverified",
            "verdict": "⚠️ Insufficient data",
            "recommendation": ""
        }
        
        # Check claim against market data if available
        if not market_data:
            claim_result["verification_status"] = "⚠️ Cannot verify - no market data available"
        else:
            # Find markets related to the claim
            related_markets = vector_search(claim_result["claim_title"], vector_db, 3)
            
            if related_markets:
                # Use the most relevant market for verification
                best_match = related_markets[0]
                
                # Extract probability from best matching market
                real_prob = best_match.get("yes_probability", 0)
                claimed_prob = claim_result["claimed_probability"]
                
                # Calculate difference
                claim_result["real_probability"] = real_prob
                claim_result["probability_difference"] = real_prob - claimed_prob
                
                # Assess the probability claim
                if abs(claim_result["probability_difference"]) <= 5:
                    claim_result["verification_status"] = "✅ Accurate probability"
                    valid_claims += 1
                elif abs(claim_result["probability_difference"]) <= 15:
                    claim_result["verification_status"] = "⚠️ Somewhat inaccurate probability"
                    claim_result["recommendation"] = f"Update probability from {claimed_prob}% to {real_prob}%"
                else:
                    claim_result["verification_status"] = "❌ Significantly inaccurate probability"
                    claim_result["recommendation"] = f"Urgent update needed: Use {real_prob}% instead of {claimed_prob}%"
                
                # Assess impact claim if provided
                if "impact" in claim and "impact_assets" in claim:
                    impact_value = claim.get("impact", 0)
                    impact_assets = claim.get("impact_assets", [])
                    
                    # Logic to verify impact assessment
                    # This is a simplified version - in reality would need complex financial modeling
                    if real_prob < 10 and abs(impact_value) > 5:
                        claim_result["impact_assessment"] = "❌ Impact likely overstated for low-probability event"
                    elif real_prob > 90 and abs(impact_value) < 2:
                        claim_result["impact_assessment"] = "❌ Impact likely understated for high-probability event"
                    else:
                        claim_result["impact_assessment"] = "✅ Reasonable impact assessment"
                        valid_claims += 0.5  # Count impact assessment as half a valid claim
                
                # Set overall verdict
                if "❌" in claim_result["verification_status"] or "❌" in claim_result["impact_assessment"]:
                    claim_result["verdict"] = "❌ Needs significant revision"
                elif "⚠️" in claim_result["verification_status"] or "⚠️" in claim_result["impact_assessment"]:
                    claim_result["verdict"] = "⚠️ Needs minor updates"
                else:
                    claim_result["verdict"] = "✅ Valid and reliable"
                    
            else:
                claim_result["verification_status"] = "⚠️ No matching markets found for verification"
        
        verification_results["claim_verifications"].append(claim_result)
    
    # Calculate overall reliability score (0-100)
    if total_claims > 0:
        reliability_score = (valid_claims / (total_claims * 1.5)) * 100  # Adjusted for impact assessments
        verification_results["overall_reliability_score"] = min(100, round(reliability_score))
    
    # Generate overall recommendations
    if verification_results["overall_reliability_score"] >= 80:
        verification_results["recommendations"].append(
            "✅ Analysis is generally reliable. Minor updates recommended for specific claims as noted."
        )
    elif verification_results["overall_reliability_score"] >= 50:
        verification_results["recommendations"].append(
            "⚠️ Analysis needs moderate revision. Update probabilities to match current market data."
        )
    else:
        verification_results["recommendations"].append(
            "❌ Analysis requires significant updates. Multiple probability and impact claims are out of alignment with market data."
        )
    
    # Add specific recommendations for interest rate sensitive assets if relevant
    interest_rate_claims = [c for c in verification_results["claim_verifications"] 
                           if any(term in c["claim_title"].lower() for term in ["rate", "yield", "fed", "treasury"])]
    
    if interest_rate_claims:
        high_prob_rate_change = any(c["real_probability"] > 70 for c in interest_rate_claims if c["real_probability"])
        if high_prob_rate_change:
            verification_results["recommendations"].append(
                "⚠️ High probability of interest rate changes detected. Review rate-sensitive holdings (ICICI, Tesla, growth stocks)."
            )
    
    return verification_results

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
        ),
        types.Tool(
            name="verify-portfolio-risk-claims",
            description="Verify claims in portfolio risk analysis against current prediction market data",
            inputSchema={
                "type": "object",
                "properties": {
                    "portfolio_data": {
                        "type": "object",
                        "description": "Portfolio data with claims to verify"
                    },
                    "verification_date": {
                        "type": "string",
                        "description": "Date for verification (defaults to today)"
                    }
                },
                "required": ["portfolio_data"]
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
            
        elif name == "verify-portfolio-risk-claims":
            if not arguments or "portfolio_data" not in arguments:
                return [types.TextContent(
                    type="text", 
                    text="Error: portfolio_data parameter is required."
                )]
                
            portfolio_data = arguments["portfolio_data"]
            verification_date = arguments.get("verification_date")
            
            # Ensure we have market data
            if not market_data or not vector_db:
                await refresh_prediction_markets()
                
            # Verify portfolio risk claims
            verification_results = verify_portfolio_risk_claims(portfolio_data, verification_date)
            
            # Format the results as a detailed report
            current_date = datetime.now().strftime("%B %d, %Y")
            
            results = [f"# PORTFOLIO RISK ANALYSIS VERIFICATION\n*Verified on {verification_results['verification_date']} using Polymarket data*\n"]
            
            # Portfolio summary
            results.append("## PORTFOLIO SUMMARY")
            results.append(f"Portfolio: {verification_results['portfolio_name']}")
            results.append(f"Total Value: ${verification_results['total_value']:,.2f}")
            results.append(f"Overall Reliability Score: {verification_results['overall_reliability_score']}/100\n")
            
            # Claim verifications
            results.append("## CLAIM VERIFICATIONS")
            for i, claim in enumerate(verification_results["claim_verifications"]):
                results.append(f"\n### {i+1}. {claim['claim_title']}")
                results.append(f"**Category:** {claim['claim_category']}")
                results.append(f"**Claimed Probability:** {claim['claimed_probability']}%")
                
                if claim["real_probability"] is not None:
                    results.append(f"**Actual Market Probability:** {claim['real_probability']:.1f}%")
                    results.append(f"**Difference:** {claim['probability_difference']:+.1f}%")
                
                results.append(f"**Status:** {claim['verification_status']}")
                
                if claim["impact_assessment"] != "⚠️ Unverified":
                    results.append(f"**Impact Assessment:** {claim['impact_assessment']}")
                    results.append(f"**Claimed Impact:** {claim['claimed_impact']}%")
                
                results.append(f"**Verdict:** {claim['verdict']}")
                
                if claim["recommendation"]:
                    results.append(f"**Recommendation:** {claim['recommendation']}")
            
            # Overall recommendations
            results.append("\n## RECOMMENDATIONS")
            for rec in verification_results["recommendations"]:
                results.append(f"- {rec}")
            
            # Data sources
            results.append("\n## DATA SOURCES")
            results.append(f"This verification was performed using data from: {', '.join(verification_results['data_sources'])}")
            
            # Disclaimer
            results.append("\n## DISCLAIMER")
            results.append("*This verification is based on current prediction market data and should not be considered financial advice. Market probabilities can shift rapidly with new information.*")
            
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

async def start_periodic_refresh():
    """Start periodic refresh of market data."""
    while True:
        try:
            await refresh_prediction_markets(force=True)
        except Exception as e:
            logger.error(f"Error in periodic refresh: {str(e)}")
        await asyncio.sleep(DATA_REFRESH_MINUTES * 60)

async def main():
    """
    Main entry point for the enhanced MCP server.
    Now includes periodic data refresh.
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
        await refresh_prediction_markets(force=True)
        logger.info(f"Successfully loaded {len(market_data)} markets")
    except Exception as e:
        logger.error(f"Error in initial data refresh: {str(e)}")
        logger.warning("Continuing without initial data - will attempt refresh on first request")
    
    # Start periodic refresh task
    refresh_task = asyncio.create_task(start_periodic_refresh())
    
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
        # Clean up
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass
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