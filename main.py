"""
Main entry point for the Enterprise AI System.
Launches the web interface server.
"""
import os
import asyncio
from dotenv import load_dotenv


def main():
    """Single entry point - Launch the web interface"""
    load_dotenv()
    
    print("ENTERPRISE AI WEB INTERFACE")
    print("="*50)
    port = os.getenv("ENTERPRISE_AI_PORT", "8000")
    print(f"Dashboard available at: http://localhost:{port}")
    print("Press Ctrl+C to stop")
    print("="*50)
    
    try:
        # Import and start the web interface
        asyncio.run(start_web_server())
        
    except KeyboardInterrupt:
        print("\nServer stopped!")
    except Exception as e:
        print(f"Error: {e}")
        print("Install dependencies: pip install fastapi uvicorn jinja2 python-multipart markdown")


async def start_web_server():
    """Start the FastAPI web server"""
    try:
        from enterprise_ai.dashboard import app
        import uvicorn
        
        port = int(os.getenv("ENTERPRISE_AI_PORT", 8000))
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install fastapi uvicorn jinja2 python-multipart markdown")
    except Exception as e:
        print(f"Server error: {e}")


if __name__ == "__main__":
    main()