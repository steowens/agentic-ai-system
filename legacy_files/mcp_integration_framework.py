"""
Model Context Protocol (MCP) Integration Framework
Handles integration with third-party systems like ESRI, databases, and stored procedures.
"""
import json
import asyncio
import aiohttp
import sqlite3
import subprocess
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
from pathlib import Path


class MCPResourceType(Enum):
    DATABASE = "database"
    STORED_PROCEDURE = "stored_procedure"
    ESRI_SERVICE = "esri_service"
    FILE_SYSTEM = "file_system"
    WEB_API = "web_api"
    COMPUTATION = "computation"


class MCPResponseType(Enum):
    TEXT = "text"
    IMAGE = "image"
    GIS_DATA = "gis_data"
    TABULAR_DATA = "tabular_data"
    FILE = "file"
    BINARY = "binary"
    JSON = "json"
    XML = "xml"


@dataclass
class MCPResource:
    """Definition of an MCP-connected resource"""
    resource_id: str
    name: str
    resource_type: MCPResourceType
    endpoint: str
    authentication: Optional[Dict] = None
    parameters: Optional[Dict] = None
    cost_model: Optional[Dict] = None
    response_type: MCPResponseType = MCPResponseType.TEXT
    timeout: int = 30
    description: str = ""
    
    # Usage tracking
    last_used: Optional[datetime] = None
    usage_count: int = 0
    avg_response_time: float = 0.0
    avg_cost: float = 0.0


@dataclass
class MCPRequest:
    """Request to an MCP resource"""
    request_id: str
    resource_id: str
    operation: str
    parameters: Dict
    user_context: Optional[Dict] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MCPResponse:
    """Response from an MCP resource"""
    request_id: str
    resource_id: str
    success: bool
    response_type: MCPResponseType
    
    # Response data
    content: Any = None
    file_path: Optional[str] = None
    metadata: Dict = None
    
    # Performance metrics
    processing_time: float = 0.0
    cost: float = 0.0
    cost_breakdown: Dict = None
    
    # Error handling
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
        if self.cost_breakdown is None:
            self.cost_breakdown = {}


class ESRIIntegration:
    """ESRI ArcGIS integration via MCP"""
    
    def __init__(self, server_url: str, token: Optional[str] = None):
        self.server_url = server_url.rstrip('/')
        self.token = token
        self.session = None
    
    async def initialize(self):
        """Initialize ESRI connection"""
        self.session = aiohttp.ClientSession()
        
        # Test connection
        try:
            await self.get_server_info()
            print(f"✅ ESRI connection established: {self.server_url}")
        except Exception as e:
            print(f"⚠️ ESRI connection failed: {e}")
    
    async def get_server_info(self) -> Dict:
        """Get ESRI server information"""
        url = f"{self.server_url}/rest/info"
        params = {"f": "json"}
        
        if self.token:
            params["token"] = self.token
        
        async with self.session.get(url, params=params) as response:
            return await response.json()
    
    async def generate_map(self, 
                          extent: Dict,
                          layers: List[str],
                          image_format: str = "png",
                          width: int = 800,
                          height: int = 600) -> MCPResponse:
        """Generate map image via ESRI map service"""
        
        start_time = datetime.now()
        request_id = str(uuid.uuid4())
        
        try:
            # Construct map export request
            url = f"{self.server_url}/rest/services/MapServer/export"
            
            params = {
                "bbox": f"{extent['xmin']},{extent['ymin']},{extent['xmax']},{extent['ymax']}",
                "size": f"{width},{height}",
                "format": image_format,
                "layers": f"show:{','.join(layers)}",
                "f": "json"
            }
            
            if self.token:
                params["token"] = self.token
            
            async with self.session.get(url, params=params) as response:
                result = await response.json()
                
                if "href" in result:
                    # Download the generated image
                    image_url = result["href"]
                    async with self.session.get(image_url) as img_response:
                        
                        # Save image to file
                        output_path = f"generated_maps/{request_id}.{image_format}"
                        Path("generated_maps").mkdir(exist_ok=True)
                        
                        with open(output_path, "wb") as f:
                            f.write(await img_response.read())
                        
                        processing_time = (datetime.now() - start_time).total_seconds()
                        
                        return MCPResponse(
                            request_id=request_id,
                            resource_id="esri_map_service",
                            success=True,
                            response_type=MCPResponseType.IMAGE,
                            file_path=output_path,
                            metadata={
                                "width": width,
                                "height": height,
                                "format": image_format,
                                "extent": extent,
                                "layers": layers
                            },
                            processing_time=processing_time,
                            cost=0.0,  # No token cost for image generation
                            cost_breakdown={"esri_map_export": 0.0}
                        )
                else:
                    raise Exception(f"Map generation failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return MCPResponse(
                request_id=request_id,
                resource_id="esri_map_service",
                success=False,
                response_type=MCPResponseType.TEXT,
                processing_time=processing_time,
                cost=0.0,
                error_message=str(e),
                error_code="ESRI_MAP_ERROR"
            )
    
    async def query_features(self, 
                           layer_url: str, 
                           where_clause: str = "1=1",
                           return_geometry: bool = True,
                           max_records: int = 1000) -> MCPResponse:
        """Query features from ESRI feature service"""
        
        start_time = datetime.now()
        request_id = str(uuid.uuid4())
        
        try:
            params = {
                "where": where_clause,
                "returnGeometry": "true" if return_geometry else "false",
                "outFields": "*",
                "resultRecordCount": max_records,
                "f": "json"
            }
            
            if self.token:
                params["token"] = self.token
            
            query_url = f"{layer_url}/query"
            
            async with self.session.get(query_url, params=params) as response:
                result = await response.json()
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                if "features" in result:
                    return MCPResponse(
                        request_id=request_id,
                        resource_id="esri_feature_service",
                        success=True,
                        response_type=MCPResponseType.GIS_DATA,
                        content=result,
                        metadata={
                            "feature_count": len(result["features"]),
                            "has_geometry": return_geometry,
                            "where_clause": where_clause
                        },
                        processing_time=processing_time,
                        cost=0.0,
                        cost_breakdown={"esri_query": 0.0}
                    )
                else:
                    raise Exception(f"Feature query failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return MCPResponse(
                request_id=request_id,
                resource_id="esri_feature_service",
                success=False,
                response_type=MCPResponseType.TEXT,
                processing_time=processing_time,
                cost=0.0,
                error_message=str(e),
                error_code="ESRI_QUERY_ERROR"
            )
    
    async def cleanup(self):
        """Clean up ESRI connection"""
        if self.session:
            await self.session.close()


class DatabaseIntegration:
    """Database integration via MCP"""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.connection_pool = {}
    
    async def execute_stored_procedure(self, 
                                     procedure_name: str,
                                     parameters: Dict,
                                     db_name: str = "default") -> MCPResponse:
        """Execute stored procedure"""
        
        start_time = datetime.now()
        request_id = str(uuid.uuid4())
        
        try:
            # Connect to database (simplified - would use proper connection pooling)
            conn = sqlite3.connect(self.db_config.get("path", ":memory:"))
            cursor = conn.cursor()
            
            # Execute procedure (simplified example)
            if procedure_name == "get_engineering_data":
                cursor.execute("""
                    SELECT part_id, material, stress_limit, safety_factor
                    FROM engineering_parts 
                    WHERE material = ?
                """, (parameters.get("material", "steel"),))
                
                results = cursor.fetchall()
                columns = ["part_id", "material", "stress_limit", "safety_factor"]
                
                data = []
                for row in results:
                    data.append(dict(zip(columns, row)))
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return MCPResponse(
                    request_id=request_id,
                    resource_id=f"db_procedure_{procedure_name}",
                    success=True,
                    response_type=MCPResponseType.TABULAR_DATA,
                    content=data,
                    metadata={
                        "procedure": procedure_name,
                        "record_count": len(data),
                        "columns": columns
                    },
                    processing_time=processing_time,
                    cost=0.01,  # Database query cost
                    cost_breakdown={"database_query": 0.01}
                )
            
            else:
                raise Exception(f"Unknown procedure: {procedure_name}")
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return MCPResponse(
                request_id=request_id,
                resource_id=f"db_procedure_{procedure_name}",
                success=False,
                response_type=MCPResponseType.TEXT,
                processing_time=processing_time,
                cost=0.0,
                error_message=str(e),
                error_code="DATABASE_ERROR"
            )
        
        finally:
            if 'conn' in locals():
                conn.close()


class MCPOrchestrator:
    """Main MCP integration orchestrator"""
    
    def __init__(self):
        self.resources = {}
        self.integrations = {}
        self.request_history = []
        
        # Cost tracking for non-token resources
        self.cost_models = {
            MCPResourceType.DATABASE: {"per_query": 0.01, "per_record": 0.001},
            MCPResourceType.ESRI_SERVICE: {"per_map": 0.05, "per_query": 0.02},
            MCPResourceType.WEB_API: {"per_request": 0.005},
            MCPResourceType.COMPUTATION: {"per_second": 0.1}
        }
    
    def register_resource(self, resource: MCPResource):
        """Register an MCP resource"""
        self.resources[resource.resource_id] = resource
        print(f"✅ Registered MCP resource: {resource.name} ({resource.resource_type.value})")
    
    def register_esri_integration(self, server_url: str, token: Optional[str] = None):
        """Register ESRI integration"""
        integration = ESRIIntegration(server_url, token)
        self.integrations["esri"] = integration
        
        # Register ESRI resources
        self.register_resource(MCPResource(
            resource_id="esri_map_service",
            name="ESRI Map Generation",
            resource_type=MCPResourceType.ESRI_SERVICE,
            endpoint=f"{server_url}/rest/services/MapServer/export",
            response_type=MCPResponseType.IMAGE,
            description="Generate maps and spatial visualizations"
        ))
        
        self.register_resource(MCPResource(
            resource_id="esri_feature_service", 
            name="ESRI Feature Query",
            resource_type=MCPResourceType.ESRI_SERVICE,
            endpoint=f"{server_url}/rest/services",
            response_type=MCPResponseType.GIS_DATA,
            description="Query spatial features and attributes"
        ))
    
    def register_database_integration(self, db_config: Dict):
        """Register database integration"""
        integration = DatabaseIntegration(db_config)
        self.integrations["database"] = integration
        
        # Register database resources
        self.register_resource(MCPResource(
            resource_id="engineering_database",
            name="Engineering Database",
            resource_type=MCPResourceType.DATABASE,
            endpoint=db_config.get("path", ""),
            response_type=MCPResponseType.TABULAR_DATA,
            description="Engineering parts and materials database"
        ))
    
    async def initialize_integrations(self):
        """Initialize all registered integrations"""
        for name, integration in self.integrations.items():
            if hasattr(integration, 'initialize'):
                await integration.initialize()
                print(f"✅ Initialized {name} integration")
    
    async def execute_mcp_request(self, request: MCPRequest) -> MCPResponse:
        """Execute an MCP request"""
        
        if request.resource_id not in self.resources:
            return MCPResponse(
                request_id=request.request_id,
                resource_id=request.resource_id,
                success=False,
                response_type=MCPResponseType.TEXT,
                error_message=f"Resource not found: {request.resource_id}",
                error_code="RESOURCE_NOT_FOUND"
            )
        
        resource = self.resources[request.resource_id]
        
        # Route to appropriate integration
        if resource.resource_type == MCPResourceType.ESRI_SERVICE:
            return await self._execute_esri_request(request, resource)
        elif resource.resource_type == MCPResourceType.DATABASE:
            return await self._execute_database_request(request, resource)
        else:
            return MCPResponse(
                request_id=request.request_id,
                resource_id=request.resource_id,
                success=False,
                response_type=MCPResponseType.TEXT,
                error_message=f"Unsupported resource type: {resource.resource_type}",
                error_code="UNSUPPORTED_TYPE"
            )
    
    async def _execute_esri_request(self, request: MCPRequest, resource: MCPResource) -> MCPResponse:
        """Execute ESRI-specific request"""
        esri_integration = self.integrations.get("esri")
        
        if not esri_integration:
            return MCPResponse(
                request_id=request.request_id,
                resource_id=request.resource_id,
                success=False,
                response_type=MCPResponseType.TEXT,
                error_message="ESRI integration not available",
                error_code="INTEGRATION_NOT_AVAILABLE"
            )
        
        if request.operation == "generate_map":
            return await esri_integration.generate_map(
                extent=request.parameters.get("extent", {}),
                layers=request.parameters.get("layers", []),
                image_format=request.parameters.get("format", "png"),
                width=request.parameters.get("width", 800),
                height=request.parameters.get("height", 600)
            )
        elif request.operation == "query_features":
            return await esri_integration.query_features(
                layer_url=request.parameters.get("layer_url", ""),
                where_clause=request.parameters.get("where", "1=1"),
                return_geometry=request.parameters.get("geometry", True),
                max_records=request.parameters.get("max_records", 1000)
            )
        else:
            return MCPResponse(
                request_id=request.request_id,
                resource_id=request.resource_id,
                success=False,
                response_type=MCPResponseType.TEXT,
                error_message=f"Unknown ESRI operation: {request.operation}",
                error_code="UNKNOWN_OPERATION"
            )
    
    async def _execute_database_request(self, request: MCPRequest, resource: MCPResource) -> MCPResponse:
        """Execute database-specific request"""
        db_integration = self.integrations.get("database")
        
        if not db_integration:
            return MCPResponse(
                request_id=request.request_id,
                resource_id=request.resource_id,
                success=False,
                response_type=MCPResponseType.TEXT,
                error_message="Database integration not available",
                error_code="INTEGRATION_NOT_AVAILABLE"
            )
        
        if request.operation == "execute_procedure":
            return await db_integration.execute_stored_procedure(
                procedure_name=request.parameters.get("procedure", ""),
                parameters=request.parameters.get("params", {}),
                db_name=request.parameters.get("database", "default")
            )
        else:
            return MCPResponse(
                request_id=request.request_id,
                resource_id=request.resource_id,
                success=False,
                response_type=MCPResponseType.TEXT,
                error_message=f"Unknown database operation: {request.operation}",
                error_code="UNKNOWN_OPERATION"
            )
    
    def get_available_resources(self) -> List[Dict]:
        """Get list of available MCP resources"""
        return [
            {
                "resource_id": resource.resource_id,
                "name": resource.name,
                "type": resource.resource_type.value,
                "response_type": resource.response_type.value,
                "description": resource.description,
                "usage_count": resource.usage_count,
                "avg_cost": resource.avg_cost
            }
            for resource in self.resources.values()
        ]
    
    def get_cost_summary(self, timeframe_hours: int = 24) -> Dict:
        """Get cost summary for MCP operations"""
        
        cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
        recent_requests = [r for r in self.request_history if r.timestamp > cutoff_time]
        
        total_cost = sum(r.cost for r in recent_requests if r.cost)
        request_count = len(recent_requests)
        
        # Cost breakdown by resource type
        cost_by_type = {}
        for request in recent_requests:
            resource = self.resources.get(request.resource_id)
            if resource and request.cost:
                resource_type = resource.resource_type.value
                cost_by_type[resource_type] = cost_by_type.get(resource_type, 0) + request.cost
        
        return {
            "timeframe_hours": timeframe_hours,
            "total_requests": request_count,
            "total_cost": round(total_cost, 4),
            "avg_cost_per_request": round(total_cost / request_count, 4) if request_count > 0 else 0,
            "cost_by_resource_type": cost_by_type,
            "zero_cost_requests": len([r for r in recent_requests if r.cost == 0]),
            "savings_from_mcp": f"Images and maps generate no token costs"
        }
    
    async def cleanup(self):
        """Clean up all integrations"""
        for integration in self.integrations.values():
            if hasattr(integration, 'cleanup'):
                await integration.cleanup()


# Example usage and testing
if __name__ == "__main__":
    
    async def demo_mcp_integration():
        print("🔌 MCP INTEGRATION FRAMEWORK DEMO")
        print("=" * 50)
        
        # Initialize MCP orchestrator
        mcp = MCPOrchestrator()
        
        # Register integrations (using example URLs)
        mcp.register_esri_integration("https://services.arcgisonline.com/arcgis")
        mcp.register_database_integration({"path": "engineering.db"})
        
        # Initialize integrations
        await mcp.initialize_integrations()
        
        # Show available resources
        print("\n📋 Available MCP Resources:")
        resources = mcp.get_available_resources()
        for resource in resources:
            print(f"  🔧 {resource['name']} ({resource['type']}) - {resource['description']}")
        
        # Example requests
        test_requests = [
            MCPRequest(
                request_id="req1",
                resource_id="esri_map_service",
                operation="generate_map",
                parameters={
                    "extent": {"xmin": -120, "ymin": 35, "xmax": -119, "ymax": 36},
                    "layers": ["0", "1"],
                    "format": "png",
                    "width": 800,
                    "height": 600
                }
            ),
            MCPRequest(
                request_id="req2", 
                resource_id="engineering_database",
                operation="execute_procedure",
                parameters={
                    "procedure": "get_engineering_data",
                    "params": {"material": "steel"}
                }
            )
        ]
        
        print(f"\n🧪 Testing {len(test_requests)} MCP requests:")
        print("-" * 40)
        
        for request in test_requests:
            print(f"\n📤 Request: {request.operation} on {request.resource_id}")
            
            try:
                response = await mcp.execute_mcp_request(request)
                
                if response.success:
                    print(f"✅ Success: {response.response_type.value}")
                    print(f"⏱️  Time: {response.processing_time:.2f}s")
                    print(f"💰 Cost: ${response.cost:.4f}")
                    
                    if response.file_path:
                        print(f"📁 File: {response.file_path}")
                    if response.content and isinstance(response.content, list):
                        print(f"📊 Records: {len(response.content)}")
                else:
                    print(f"❌ Failed: {response.error_message}")
            
            except Exception as e:
                print(f"💥 Error: {e}")
        
        # Show cost summary
        cost_summary = mcp.get_cost_summary()
        print(f"\n💰 COST SUMMARY:")
        print(f"Total MCP requests: {cost_summary['total_requests']}")
        print(f"Total cost: ${cost_summary['total_cost']:.4f}")
        print(f"Zero-cost requests: {cost_summary['zero_cost_requests']} (images, maps)")
        print(f"💡 {cost_summary['savings_from_mcp']}")
        
        # Cleanup
        await mcp.cleanup()
        print("\n🎉 MCP integration demo completed!")
    
    # Run the demo
    asyncio.run(demo_mcp_integration())