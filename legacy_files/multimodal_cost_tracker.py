"""
Enhanced Multi-Modal Cost Tracking System
Tracks costs for tokens, database queries, API calls, image generation, and other resources.
"""
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import sqlite3
from pathlib import Path


class ResourceType(Enum):
    LLM_TOKENS = "llm_tokens"
    DATABASE_QUERY = "database_query"
    API_CALL = "api_call"
    IMAGE_GENERATION = "image_generation"
    MAP_GENERATION = "map_generation"
    FILE_STORAGE = "file_storage"
    COMPUTATION_TIME = "computation_time"
    NETWORK_BANDWIDTH = "network_bandwidth"
    MEMORY_USAGE = "memory_usage"


@dataclass
class ResourceUsage:
    """Detailed resource usage tracking"""
    usage_id: str
    resource_type: ResourceType
    resource_name: str
    session_id: str
    timestamp: datetime
    
    # Quantity metrics
    quantity: float
    unit: str
    
    # Cost metrics
    unit_cost: float
    total_cost: float
    currency: str = "USD"
    
    # Performance metrics
    response_time: float = 0.0
    success: bool = True
    
    # Context
    user_id: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a request"""
    session_id: str
    request_id: str
    timestamp: datetime
    
    # Token costs
    input_tokens: int = 0
    output_tokens: int = 0
    token_cost: float = 0.0
    
    # Database costs
    database_queries: int = 0
    database_records: int = 0
    database_cost: float = 0.0
    
    # API costs
    api_calls: int = 0
    api_cost: float = 0.0
    
    # Media costs
    images_generated: int = 0
    maps_generated: int = 0
    media_cost: float = 0.0
    
    # Infrastructure costs
    computation_seconds: float = 0.0
    storage_mb: float = 0.0
    bandwidth_mb: float = 0.0
    infrastructure_cost: float = 0.0
    
    # Total
    total_cost: float = 0.0
    
    # Savings
    zero_cost_operations: int = 0
    estimated_token_savings: float = 0.0


class MultiModalCostTracker:
    """Comprehensive cost tracking for all resource types"""
    
    def __init__(self, db_path: str = "multimodal_costs.db"):
        self.db_path = Path(db_path)
        self.init_database()
        
        # Cost models for different resources
        self.cost_models = {
            ResourceType.LLM_TOKENS: {
                "gpt-4o-mini": {
                    "input_per_1k": 0.00015,
                    "output_per_1k": 0.0006
                },
                "gpt-4o": {
                    "input_per_1k": 0.005,
                    "output_per_1k": 0.015
                },
                "gpt-3.5-turbo": {
                    "input_per_1k": 0.0005,
                    "output_per_1k": 0.0015
                }
            },
            ResourceType.DATABASE_QUERY: {
                "per_query": 0.01,
                "per_record": 0.001,
                "per_complex_join": 0.05
            },
            ResourceType.API_CALL: {
                "standard": 0.005,
                "premium": 0.02,
                "enterprise": 0.1
            },
            ResourceType.IMAGE_GENERATION: {
                "dalle-3": {"standard": 0.04, "hd": 0.08},
                "dalle-2": 0.02,
                "midjourney": 0.05,
                "stable_diffusion": 0.01  # Self-hosted
            },
            ResourceType.MAP_GENERATION: {
                "esri_standard": 0.0,  # Often included in license
                "google_maps": 0.005,
                "mapbox": 0.003
            },
            ResourceType.FILE_STORAGE: {
                "per_gb_per_month": 0.025,
                "per_gb_transfer": 0.09
            },
            ResourceType.COMPUTATION_TIME: {
                "cpu_hour": 0.10,
                "gpu_hour": 2.50,
                "memory_gb_hour": 0.01
            }
        }
        
        # Current session tracking
        self.current_session = {}
        
    def init_database(self):
        """Initialize cost tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Resource usage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS resource_usage (
                usage_id TEXT PRIMARY KEY,
                resource_type TEXT,
                resource_name TEXT,
                session_id TEXT,
                timestamp TEXT,
                quantity REAL,
                unit TEXT,
                unit_cost REAL,
                total_cost REAL,
                currency TEXT,
                response_time REAL,
                success BOOLEAN,
                user_id TEXT,
                operation TEXT,
                metadata TEXT
            )
        """)
        
        # Session cost breakdown table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_costs (
                session_id TEXT,
                request_id TEXT,
                timestamp TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                token_cost REAL,
                database_queries INTEGER,
                database_records INTEGER,
                database_cost REAL,
                api_calls INTEGER,
                api_cost REAL,
                images_generated INTEGER,
                maps_generated INTEGER,
                media_cost REAL,
                computation_seconds REAL,
                storage_mb REAL,
                bandwidth_mb REAL,
                infrastructure_cost REAL,
                total_cost REAL,
                zero_cost_operations INTEGER,
                estimated_token_savings REAL,
                PRIMARY KEY (session_id, request_id)
            )
        """)
        
        # Cost optimization tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cost_optimizations (
                optimization_id TEXT PRIMARY KEY,
                timestamp TEXT,
                optimization_type TEXT,
                original_cost REAL,
                optimized_cost REAL,
                savings REAL,
                description TEXT,
                session_id TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def track_token_usage(self, 
                         session_id: str,
                         model: str,
                         input_tokens: int,
                         output_tokens: int) -> ResourceUsage:
        """Track LLM token usage and costs"""
        
        model_costs = self.cost_models[ResourceType.LLM_TOKENS].get(model, {
            "input_per_1k": 0.001,  # Default fallback
            "output_per_1k": 0.002
        })
        
        input_cost = (input_tokens / 1000) * model_costs["input_per_1k"]
        output_cost = (output_tokens / 1000) * model_costs["output_per_1k"]
        total_cost = input_cost + output_cost
        
        usage = ResourceUsage(
            usage_id=f"{session_id}_tokens_{datetime.now().isoformat()}",
            resource_type=ResourceType.LLM_TOKENS,
            resource_name=model,
            session_id=session_id,
            timestamp=datetime.now(),
            quantity=input_tokens + output_tokens,
            unit="tokens",
            unit_cost=(input_cost + output_cost) / (input_tokens + output_tokens) if (input_tokens + output_tokens) > 0 else 0,
            total_cost=total_cost,
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "model": model
            }
        )
        
        self._store_usage(usage)
        return usage
    
    def track_database_operation(self,
                               session_id: str,
                               operation_type: str,
                               query_count: int,
                               record_count: int,
                               response_time: float) -> ResourceUsage:
        """Track database operation costs"""
        
        db_costs = self.cost_models[ResourceType.DATABASE_QUERY]
        
        base_cost = query_count * db_costs["per_query"]
        record_cost = record_count * db_costs["per_record"]
        
        # Add complexity cost for joins, subqueries, etc.
        complexity_multiplier = 1.0
        if "join" in operation_type.lower():
            complexity_multiplier = 2.0
            base_cost += db_costs["per_complex_join"]
        
        total_cost = (base_cost + record_cost) * complexity_multiplier
        
        usage = ResourceUsage(
            usage_id=f"{session_id}_db_{datetime.now().isoformat()}",
            resource_type=ResourceType.DATABASE_QUERY,
            resource_name="database",
            session_id=session_id,
            timestamp=datetime.now(),
            quantity=record_count,
            unit="records",
            unit_cost=total_cost / max(record_count, 1),
            total_cost=total_cost,
            response_time=response_time,
            operation=operation_type,
            metadata={
                "query_count": query_count,
                "record_count": record_count,
                "operation_type": operation_type,
                "complexity_multiplier": complexity_multiplier
            }
        )
        
        self._store_usage(usage)
        return usage
    
    def track_image_generation(self,
                             session_id: str,
                             provider: str,
                             image_count: int,
                             quality: str = "standard") -> ResourceUsage:
        """Track image generation costs (often zero-token cost)"""
        
        if provider == "esri" or provider == "arcgis":
            # ESRI maps typically don't have per-request costs
            total_cost = 0.0
            estimated_token_savings = image_count * 500 * 0.0006  # Estimated tokens saved
        else:
            image_costs = self.cost_models[ResourceType.IMAGE_GENERATION].get(provider, {"standard": 0.03})
            
            if isinstance(image_costs, dict):
                unit_cost = image_costs.get(quality, image_costs.get("standard", 0.03))
            else:
                unit_cost = image_costs
            
            total_cost = image_count * unit_cost
            estimated_token_savings = 0.0
        
        usage = ResourceUsage(
            usage_id=f"{session_id}_img_{datetime.now().isoformat()}",
            resource_type=ResourceType.IMAGE_GENERATION,
            resource_name=provider,
            session_id=session_id,
            timestamp=datetime.now(),
            quantity=image_count,
            unit="images",
            unit_cost=total_cost / max(image_count, 1),
            total_cost=total_cost,
            metadata={
                "provider": provider,
                "quality": quality,
                "estimated_token_savings": estimated_token_savings,
                "zero_token_cost": total_cost == 0.0
            }
        )
        
        self._store_usage(usage)
        return usage
    
    def track_api_call(self,
                      session_id: str,
                      api_name: str,
                      call_count: int,
                      tier: str = "standard",
                      response_time: float = 0.0) -> ResourceUsage:
        """Track external API call costs"""
        
        api_costs = self.cost_models[ResourceType.API_CALL]
        unit_cost = api_costs.get(tier, api_costs["standard"])
        total_cost = call_count * unit_cost
        
        usage = ResourceUsage(
            usage_id=f"{session_id}_api_{datetime.now().isoformat()}",
            resource_type=ResourceType.API_CALL,
            resource_name=api_name,
            session_id=session_id,
            timestamp=datetime.now(),
            quantity=call_count,
            unit="calls",
            unit_cost=unit_cost,
            total_cost=total_cost,
            response_time=response_time,
            metadata={
                "api_name": api_name,
                "tier": tier
            }
        )
        
        self._store_usage(usage)
        return usage
    
    def get_session_cost_breakdown(self, session_id: str) -> CostBreakdown:
        """Get comprehensive cost breakdown for a session"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all usage for this session
        cursor.execute("""
            SELECT resource_type, quantity, total_cost, metadata, response_time
            FROM resource_usage
            WHERE session_id = ?
            ORDER BY timestamp
        """, (session_id,))
        
        usage_records = cursor.fetchall()
        conn.close()
        
        breakdown = CostBreakdown(
            session_id=session_id,
            request_id="session_summary",
            timestamp=datetime.now()
        )
        
        # Process usage records
        for resource_type_str, quantity, cost, metadata_str, response_time in usage_records:
            resource_type = ResourceType(resource_type_str)
            metadata = json.loads(metadata_str) if metadata_str else {}
            
            if resource_type == ResourceType.LLM_TOKENS:
                breakdown.input_tokens += metadata.get("input_tokens", 0)
                breakdown.output_tokens += metadata.get("output_tokens", 0)
                breakdown.token_cost += cost
            
            elif resource_type == ResourceType.DATABASE_QUERY:
                breakdown.database_queries += metadata.get("query_count", 1)
                breakdown.database_records += int(quantity)
                breakdown.database_cost += cost
            
            elif resource_type == ResourceType.API_CALL:
                breakdown.api_calls += int(quantity)
                breakdown.api_cost += cost
            
            elif resource_type in [ResourceType.IMAGE_GENERATION, ResourceType.MAP_GENERATION]:
                if resource_type == ResourceType.IMAGE_GENERATION:
                    breakdown.images_generated += int(quantity)
                else:
                    breakdown.maps_generated += int(quantity)
                
                breakdown.media_cost += cost
                
                # Track zero-cost operations
                if cost == 0.0:
                    breakdown.zero_cost_operations += int(quantity)
                    breakdown.estimated_token_savings += metadata.get("estimated_token_savings", 0.0)
            
            elif resource_type == ResourceType.COMPUTATION_TIME:
                breakdown.computation_seconds += quantity
                breakdown.infrastructure_cost += cost
        
        # Calculate total cost
        breakdown.total_cost = (
            breakdown.token_cost +
            breakdown.database_cost +
            breakdown.api_cost +
            breakdown.media_cost +
            breakdown.infrastructure_cost
        )
        
        return breakdown
    
    def get_cost_optimization_suggestions(self, session_id: str) -> List[Dict]:
        """Analyze usage and suggest cost optimizations"""
        
        breakdown = self.get_session_cost_breakdown(session_id)
        suggestions = []
        
        # Token cost optimization
        if breakdown.token_cost > 0.01:  # More than 1 cent in tokens
            token_ratio = breakdown.output_tokens / max(breakdown.input_tokens, 1)
            
            if token_ratio > 3:  # High output-to-input ratio
                suggestions.append({
                    "type": "token_optimization",
                    "priority": "high",
                    "potential_savings": breakdown.token_cost * 0.3,
                    "suggestion": "Consider using more concise prompts or limiting response length",
                    "details": f"Output tokens ({breakdown.output_tokens}) are {token_ratio:.1f}x input tokens"
                })
        
        # Database optimization
        if breakdown.database_queries > 10:
            suggestions.append({
                "type": "database_optimization",
                "priority": "medium",
                "potential_savings": breakdown.database_cost * 0.4,
                "suggestion": "Consider query caching or batch operations",
                "details": f"{breakdown.database_queries} queries could be optimized"
            })
        
        # MCP utilization
        if breakdown.zero_cost_operations > 0:
            suggestions.append({
                "type": "mcp_success",
                "priority": "info",
                "potential_savings": 0.0,
                "suggestion": f"Great! {breakdown.zero_cost_operations} operations had zero token cost",
                "details": f"Estimated savings: ${breakdown.estimated_token_savings:.4f}"
            })
        
        # Media vs token trade-offs
        if breakdown.images_generated == 0 and breakdown.token_cost > 0.005:
            suggestions.append({
                "type": "media_alternative",
                "priority": "medium",
                "potential_savings": breakdown.token_cost * 0.2,
                "suggestion": "Consider using visual outputs (charts, maps) instead of lengthy text descriptions",
                "details": "Visual content can be more effective and may reduce token usage"
            })
        
        return suggestions
    
    def _store_usage(self, usage: ResourceUsage):
        """Store usage record in database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO resource_usage VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            usage.usage_id,
            usage.resource_type.value,
            usage.resource_name,
            usage.session_id,
            usage.timestamp.isoformat(),
            usage.quantity,
            usage.unit,
            usage.unit_cost,
            usage.total_cost,
            usage.currency,
            usage.response_time,
            usage.success,
            usage.user_id,
            usage.operation,
            json.dumps(usage.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def get_cost_trends(self, days: int = 7) -> Dict:
        """Get cost trends over time"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Daily cost breakdown
        cursor.execute("""
            SELECT 
                DATE(timestamp) as date,
                resource_type,
                SUM(total_cost) as daily_cost,
                COUNT(*) as operation_count
            FROM resource_usage
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp), resource_type
            ORDER BY date, resource_type
        """, (since_date,))
        
        trends = {}
        for date, resource_type, daily_cost, count in cursor.fetchall():
            if date not in trends:
                trends[date] = {}
            trends[date][resource_type] = {
                "cost": daily_cost,
                "operations": count
            }
        
        conn.close()
        return trends


# Example usage and testing
if __name__ == "__main__":
    
    print("💰 MULTI-MODAL COST TRACKING DEMO")
    print("=" * 50)
    
    # Initialize cost tracker
    tracker = MultiModalCostTracker("demo_multimodal_costs.db")
    
    session_id = "demo_session_1"
    
    # Simulate different types of resource usage
    
    # 1. Token usage (expensive)
    print("\n🤖 Tracking token usage...")
    token_usage = tracker.track_token_usage(
        session_id=session_id,
        model="gpt-4o-mini",
        input_tokens=1500,
        output_tokens=800
    )
    print(f"   Tokens: {token_usage.quantity}, Cost: ${token_usage.total_cost:.4f}")
    
    # 2. Database operations 
    print("🗄️  Tracking database query...")
    db_usage = tracker.track_database_operation(
        session_id=session_id,
        operation_type="complex_join",
        query_count=3,
        record_count=150,
        response_time=2.1
    )
    print(f"   Records: {db_usage.quantity}, Cost: ${db_usage.total_cost:.4f}")
    
    # 3. Image generation (zero token cost!)
    print("🖼️  Tracking image generation...")
    image_usage = tracker.track_image_generation(
        session_id=session_id,
        provider="esri",
        image_count=2,
        quality="standard"
    )
    print(f"   Images: {image_usage.quantity}, Cost: ${image_usage.total_cost:.4f}")
    
    # 4. API calls
    print("🌐 Tracking API calls...")
    api_usage = tracker.track_api_call(
        session_id=session_id,
        api_name="weather_service",
        call_count=1,
        tier="standard",
        response_time=0.8
    )
    print(f"   API calls: {api_usage.quantity}, Cost: ${api_usage.total_cost:.4f}")
    
    # Get comprehensive cost breakdown
    print(f"\n📊 SESSION COST BREAKDOWN:")
    print("-" * 30)
    
    breakdown = tracker.get_session_cost_breakdown(session_id)
    
    print(f"🤖 Tokens: {breakdown.input_tokens} in + {breakdown.output_tokens} out = ${breakdown.token_cost:.4f}")
    print(f"🗄️  Database: {breakdown.database_queries} queries, {breakdown.database_records} records = ${breakdown.database_cost:.4f}")
    print(f"🌐 API calls: {breakdown.api_calls} calls = ${breakdown.api_cost:.4f}")
    print(f"🖼️  Media: {breakdown.images_generated} images + {breakdown.maps_generated} maps = ${breakdown.media_cost:.4f}")
    print(f"💡 Zero-cost operations: {breakdown.zero_cost_operations}")
    print(f"💰 Estimated token savings: ${breakdown.estimated_token_savings:.4f}")
    print(f"🎯 TOTAL COST: ${breakdown.total_cost:.4f}")
    
    # Get optimization suggestions
    print(f"\n🎯 COST OPTIMIZATION SUGGESTIONS:")
    print("-" * 40)
    
    suggestions = tracker.get_cost_optimization_suggestions(session_id)
    
    for suggestion in suggestions:
        priority_icon = {"high": "🔴", "medium": "🟡", "info": "💡"}.get(suggestion["priority"], "ℹ️")
        print(f"{priority_icon} {suggestion['suggestion']}")
        print(f"   {suggestion['details']}")
        if suggestion['potential_savings'] > 0:
            print(f"   💰 Potential savings: ${suggestion['potential_savings']:.4f}")
        print()
    
    print("🎉 Multi-modal cost tracking ready for integration!")