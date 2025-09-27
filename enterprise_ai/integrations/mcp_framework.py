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


class MCPOrchestrator:
    """Orchestrates requests to MCP resources"""
    
    def __init__(self):
        self.resources: Dict[str, MCPResource] = {}
        self.handlers: Dict[MCPResourceType, Callable] = {}
        self.request_history: List[MCPRequest] = []
        self.response_history: List[MCPResponse] = []
        
        # Register default handlers
        self._register_default_handlers()
    
    def register_resource(self, resource: MCPResource):
        """Register an MCP resource"""
        self.resources[resource.resource_id] = resource
        print(f"Registered MCP resource: {resource.name} ({resource.resource_type.value})")
    
    def _register_default_handlers(self):
        """Register default resource handlers"""
        self.handlers[MCPResourceType.DATABASE] = self._handle_database_request
        self.handlers[MCPResourceType.ESRI_SERVICE] = self._handle_esri_request
        self.handlers[MCPResourceType.FILE_SYSTEM] = self._handle_filesystem_request
        self.handlers[MCPResourceType.WEB_API] = self._handle_web_api_request
        self.handlers[MCPResourceType.COMPUTATION] = self._handle_computation_request
    
    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """Process an MCP request"""
        if request.resource_id not in self.resources:
            return MCPResponse(
                request_id=request.request_id,
                resource_id=request.resource_id,
                success=False,
                response_type=MCPResponseType.TEXT,
                error_message=f"Resource {request.resource_id} not found",
                error_code="RESOURCE_NOT_FOUND"
            )
        
        resource = self.resources[request.resource_id]
        handler = self.handlers.get(resource.resource_type)
        
        if not handler:
            return MCPResponse(
                request_id=request.request_id,
                resource_id=request.resource_id,
                success=False,
                response_type=MCPResponseType.TEXT,
                error_message=f"No handler for resource type {resource.resource_type.value}",
                error_code="NO_HANDLER"
            )
        
        # Track request
        self.request_history.append(request)
        start_time = datetime.now()
        
        try:
            # Process the request
            response = await handler(resource, request)
            
            # Calculate metrics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            response.processing_time = processing_time
            
            # Update resource usage statistics
            resource.usage_count += 1
            resource.last_used = end_time
            resource.avg_response_time = (
                (resource.avg_response_time * (resource.usage_count - 1) + processing_time) 
                / resource.usage_count
            )
            
            # Track response
            self.response_history.append(response)
            
            return response
            
        except Exception as e:
            return MCPResponse(
                request_id=request.request_id,
                resource_id=request.resource_id,
                success=False,
                response_type=MCPResponseType.TEXT,
                error_message=str(e),
                error_code="PROCESSING_ERROR",
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _handle_database_request(self, resource: MCPResource, request: MCPRequest) -> MCPResponse:
        """Handle database requests"""
        try:
            # Simulate database connection and query
            # In real implementation, would connect to actual database
            operation = request.operation
            params = request.parameters
            
            if operation == "query":
                # Simulate SQL query execution
                result_data = {
                    "query": params.get("sql", "SELECT * FROM table"),
                    "rows": [
                        {"id": 1, "name": "Sample Data 1", "value": 100},
                        {"id": 2, "name": "Sample Data 2", "value": 200}
                    ],
                    "count": 2,
                    "execution_time": 0.05
                }
                
                return MCPResponse(
                    request_id=request.request_id,
                    resource_id=resource.resource_id,
                    success=True,
                    response_type=MCPResponseType.JSON,
                    content=result_data,
                    cost=0.0,  # Database queries often have zero token cost
                    metadata={"source": "database", "table_count": 1}
                )
            
            else:
                return MCPResponse(
                    request_id=request.request_id,
                    resource_id=resource.resource_id,
                    success=False,
                    response_type=MCPResponseType.TEXT,
                    error_message=f"Unsupported database operation: {operation}",
                    error_code="UNSUPPORTED_OPERATION"
                )
                
        except Exception as e:
            return MCPResponse(
                request_id=request.request_id,
                resource_id=resource.resource_id,
                success=False,
                response_type=MCPResponseType.TEXT,
                error_message=f"Database error: {str(e)}",
                error_code="DATABASE_ERROR"
            )
    
    async def _handle_esri_request(self, resource: MCPResource, request: MCPRequest) -> MCPResponse:
        """Handle ESRI/GIS requests"""
        try:
            operation = request.operation
            params = request.parameters
            
            if operation == "get_features":
                # Simulate ESRI feature service request
                result_data = {
                    "features": [
                        {
                            "geometry": {"x": -122.4194, "y": 37.7749},
                            "attributes": {"name": "San Francisco", "population": 884000}
                        },
                        {
                            "geometry": {"x": -118.2437, "y": 34.0522},
                            "attributes": {"name": "Los Angeles", "population": 3900000}
                        }
                    ],
                    "spatialReference": {"wkid": 4326},
                    "count": 2
                }
                
                return MCPResponse(
                    request_id=request.request_id,
                    resource_id=resource.resource_id,
                    success=True,
                    response_type=MCPResponseType.GIS_DATA,
                    content=result_data,
                    cost=0.0,  # ESRI requests typically have zero token cost
                    metadata={"source": "esri", "feature_count": 2, "spatial_ref": 4326}
                )
            
            else:
                return MCPResponse(
                    request_id=request.request_id,
                    resource_id=resource.resource_id,
                    success=False,
                    response_type=MCPResponseType.TEXT,
                    error_message=f"Unsupported ESRI operation: {operation}",
                    error_code="UNSUPPORTED_OPERATION"
                )
                
        except Exception as e:
            return MCPResponse(
                request_id=request.request_id,
                resource_id=resource.resource_id,
                success=False,
                response_type=MCPResponseType.TEXT,
                error_message=f"ESRI error: {str(e)}",
                error_code="ESRI_ERROR"
            )
    
    async def _handle_filesystem_request(self, resource: MCPResource, request: MCPRequest) -> MCPResponse:
        """Handle file system requests"""
        try:
            operation = request.operation
            params = request.parameters
            
            if operation == "read_file":
                file_path = params.get("file_path")
                if file_path and Path(file_path).exists():
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    return MCPResponse(
                        request_id=request.request_id,
                        resource_id=resource.resource_id,
                        success=True,
                        response_type=MCPResponseType.TEXT,
                        content=content,
                        cost=0.0,
                        metadata={"source": "filesystem", "file_size": len(content)}
                    )
                else:
                    return MCPResponse(
                        request_id=request.request_id,
                        resource_id=resource.resource_id,
                        success=False,
                        response_type=MCPResponseType.TEXT,
                        error_message=f"File not found: {file_path}",
                        error_code="FILE_NOT_FOUND"
                    )
            
            else:
                return MCPResponse(
                    request_id=request.request_id,
                    resource_id=resource.resource_id,
                    success=False,
                    response_type=MCPResponseType.TEXT,
                    error_message=f"Unsupported filesystem operation: {operation}",
                    error_code="UNSUPPORTED_OPERATION"
                )
                
        except Exception as e:
            return MCPResponse(
                request_id=request.request_id,
                resource_id=resource.resource_id,
                success=False,
                response_type=MCPResponseType.TEXT,
                error_message=f"Filesystem error: {str(e)}",
                error_code="FILESYSTEM_ERROR"
            )
    
    async def _handle_web_api_request(self, resource: MCPResource, request: MCPRequest) -> MCPResponse:
        """Handle web API requests"""
        try:
            async with aiohttp.ClientSession() as session:
                url = resource.endpoint
                method = request.operation.upper()
                params = request.parameters
                
                if method == "GET":
                    async with session.get(url, params=params) as resp:
                        content = await resp.text()
                        
                        return MCPResponse(
                            request_id=request.request_id,
                            resource_id=resource.resource_id,
                            success=resp.status == 200,
                            response_type=MCPResponseType.JSON if resp.content_type == 'application/json' else MCPResponseType.TEXT,
                            content=content,
                            cost=0.0,
                            metadata={"source": "web_api", "status_code": resp.status}
                        )
                
                else:
                    return MCPResponse(
                        request_id=request.request_id,
                        resource_id=resource.resource_id,
                        success=False,
                        response_type=MCPResponseType.TEXT,
                        error_message=f"Unsupported HTTP method: {method}",
                        error_code="UNSUPPORTED_METHOD"
                    )
                    
        except Exception as e:
            return MCPResponse(
                request_id=request.request_id,
                resource_id=resource.resource_id,
                success=False,
                response_type=MCPResponseType.TEXT,
                error_message=f"Web API error: {str(e)}",
                error_code="WEB_API_ERROR"
            )
    
    async def _handle_computation_request(self, resource: MCPResource, request: MCPRequest) -> MCPResponse:
        """Handle computational requests"""
        try:
            operation = request.operation
            params = request.parameters
            
            if operation == "calculate":
                # Simulate computational work
                expression = params.get("expression", "2+2")
                result = eval(expression)  # In real implementation, would use safe evaluation
                
                return MCPResponse(
                    request_id=request.request_id,
                    resource_id=resource.resource_id,
                    success=True,
                    response_type=MCPResponseType.JSON,
                    content={"expression": expression, "result": result},
                    cost=0.0,
                    metadata={"source": "computation", "operation": "calculation"}
                )
            
            else:
                return MCPResponse(
                    request_id=request.request_id,
                    resource_id=resource.resource_id,
                    success=False,
                    response_type=MCPResponseType.TEXT,
                    error_message=f"Unsupported computation operation: {operation}",
                    error_code="UNSUPPORTED_OPERATION"
                )
                
        except Exception as e:
            return MCPResponse(
                request_id=request.request_id,
                resource_id=resource.resource_id,
                success=False,
                response_type=MCPResponseType.TEXT,
                error_message=f"Computation error: {str(e)}",
                error_code="COMPUTATION_ERROR"
            )
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get MCP usage statistics"""
        total_requests = len(self.request_history)
        successful_responses = len([r for r in self.response_history if r.success])
        
        resource_stats = {}
        for resource_id, resource in self.resources.items():
            resource_stats[resource_id] = {
                "name": resource.name,
                "type": resource.resource_type.value,
                "usage_count": resource.usage_count,
                "avg_response_time": resource.avg_response_time,
                "last_used": resource.last_used.isoformat() if resource.last_used else None
            }
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_responses,
            "success_rate": successful_responses / total_requests if total_requests > 0 else 0,
            "registered_resources": len(self.resources),
            "resource_statistics": resource_stats
        }


# Pre-configured MCP resources
def get_default_mcp_resources() -> List[MCPResource]:
    """Get default MCP resource configurations"""
    return [
        MCPResource(
            resource_id="main_database",
            name="Main Database",
            resource_type=MCPResourceType.DATABASE,
            endpoint="localhost:5432",
            description="Primary application database with customer and project data"
        ),
        MCPResource(
            resource_id="esri_feature_service",
            name="ESRI Feature Service",
            resource_type=MCPResourceType.ESRI_SERVICE,
            endpoint="https://services.arcgis.com/feature_service",
            description="Geographic data and spatial analysis services"
        ),
        MCPResource(
            resource_id="file_system",
            name="Local File System",
            resource_type=MCPResourceType.FILE_SYSTEM,
            endpoint="local",
            description="Local file system access for document retrieval"
        ),
        MCPResource(
            resource_id="computation_engine",
            name="Computation Engine",
            resource_type=MCPResourceType.COMPUTATION,
            endpoint="local",
            description="Mathematical and scientific computation services"
        )
    ]


# Global MCP orchestrator instance (lazy initialization)
mcp_orchestrator = None

def get_mcp_orchestrator():
    """Get MCP orchestrator instance, initializing if needed"""
    global mcp_orchestrator
    if mcp_orchestrator is None:
        mcp_orchestrator = MCPOrchestrator()
        # Register default resources
        for resource in get_default_mcp_resources():
            mcp_orchestrator.register_resource(resource)
    return mcp_orchestrator


async def process_mcp_request(
    resource_id: str,
    operation: str,
    parameters: Dict,
    user_context: Optional[Dict] = None
) -> MCPResponse:
    """Convenience function to process MCP requests"""
    request = MCPRequest(
        request_id=str(uuid.uuid4()),
        resource_id=resource_id,
        operation=operation,
        parameters=parameters,
        user_context=user_context
    )
    
    return await get_mcp_orchestrator().process_request(request)


def get_mcp_statistics() -> Dict[str, Any]:
    """Convenience function to get MCP statistics"""
    return get_mcp_orchestrator().get_usage_statistics()