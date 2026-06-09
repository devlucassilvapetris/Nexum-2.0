"""
Ray Serve Deployment
Deploy models with Ray Serve for scalable serving
"""

import json
from typing import Dict, Any, List
from ray import serve
from ray.serve import Application
import starlette


@serve.deployment(
    name="nexum-llm",
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 5
    },
    max_concurrent_queries=100
)
class LLMDeployment:
    """Ray Serve deployment for LLM serving"""
    
    def __init__(self, config_path: str):
        from .vllm_serve import VLLMServe
        
        self.vllm_serve = VLLMServe(config_path)
    
    async def __call__(self, request: starlette.Request) -> starlette.Response:
        """Handle HTTP request"""
        data = await request.json()
        
        if "prompts" in data:
            outputs = self.vllm_serve.generate(data["prompts"])
            return starlette.responses.JSONResponse({"outputs": outputs})
        elif "prompt" in data:
            output = self.vllm_serve.generate_single(data["prompt"])
            return starlette.responses.JSONResponse({"output": output})
        elif "messages" in data:
            output = self.vllm_serve.chat(data["messages"])
            return starlette.responses.JSONResponse({"output": output})
        else:
            return starlette.responses.JSONResponse(
                {"error": "Invalid request"}, status_code=400
            )


class RayDeployer:
    """Deploy models with Ray Serve"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.config_path = config_path
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def deploy(self):
        """Deploy the model with Ray Serve"""
        ray_config = self.config.get("ray_serve", {})
        
        if not ray_config.get("enabled", True):
            print("Ray Serve is disabled in config")
            return
        
        # Configure deployment
        deployment = LLMDeployment.options(
            num_replicas=ray_config.get("num_replicas", 1),
            max_concurrent_queries=ray_config.get("max_concurrent_queries", 100)
        ).bind(self.config_path)
        
        return deployment
    
    def deploy_with_api(self):
        """Deploy with FastAPI wrapper"""
        from fastapi import FastAPI
        import uvicorn
        
        app = FastAPI(title="Nexum LLM API")
        
        # Deploy with Ray Serve
        deployment = self.deploy()
        serve.run(deployment, name="nexum-llm")
        
        api_config = self.config["api"]
        
        uvicorn.run(
            app,
            host=api_config["host"],
            port=api_config["port"]
        )
    
    def get_deployment_status(self):
        """Get deployment status"""
        from ray.serve import status
        
        return status.applications()


def create_app(config_path: str) -> Application:
    """Create Ray Serve application"""
    deployer = RayDeployer(config_path)
    return deployer.deploy()
