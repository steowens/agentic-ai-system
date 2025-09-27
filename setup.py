"""
Enterprise AI Routing System Setup Configuration
"""
from setuptools import setup, find_packages
import os

# Read long description from README
def read_long_description():
    try:
        with open("Enterprise_Integration_Guide.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Enterprise AI Routing System - A production-grade AI orchestration platform"

# Read requirements from requirements.txt
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "autogen-agentchat>=0.4.0",
            "autogen-ext>=0.4.0", 
            "openai>=1.0.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "jinja2>=3.1.0",
            "python-multipart>=0.0.6",
            "markdown>=3.5.0",
            "aiohttp>=3.9.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "joblib>=1.3.0",
            "python-dotenv>=1.0.0"
        ]

setup(
    name="enterprise-ai-routing",
    version="1.0.0",
    author="Enterprise AI Team",
    author_email="ai-team@company.com",
    description="Production-grade AI routing system with MCP integrations and ML optimization",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/company/enterprise-ai-routing",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators", 
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0"
        ],
        "dashboard": [
            "plotly>=5.15.0",
            "dash>=2.14.0",
            "websockets>=11.0.0"
        ],
        "ml": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0"
        ],
        "security": [
            "cryptography>=41.0.0",
            "pyjwt>=2.8.0",
            "bcrypt>=4.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "enterprise-ai=enterprise_ai.cli:main",
            "enterprise-dashboard=enterprise_ai.dashboard.web_app:run_dashboard",
        ],
    },
    package_data={
        "enterprise_ai": [
            "dashboard/templates/*.html",
            "dashboard/static/*.css",
            "dashboard/static/*.js",
            "config/*.yaml",
            "config/*.json"
        ]
    },
    include_package_data=True,
    zip_safe=False,
    keywords="ai routing llm agents mcp enterprise automation",
    project_urls={
        "Bug Reports": "https://github.com/company/enterprise-ai-routing/issues",
        "Source": "https://github.com/company/enterprise-ai-routing",
        "Documentation": "https://enterprise-ai-routing.readthedocs.io/",
    },
)