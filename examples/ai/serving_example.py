"""
Example: Model Serving with vLLM and Ray Serve
"""

from nexum.ai.serving import VLLMServe, RayDeployer


def main():
    # Option 1: Use vLLM directly
    print("Starting vLLM server...")
    vllm_serve = VLLMServe("config/ai/serving_config.json")
    
    # Generate text
    prompts = [
        "What is Nexum v2.0?",
        "Explain the AI architecture."
    ]
    
    outputs = vllm_serve.generate(prompts)
    
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}")
        print(f"Output: {output}\n")
    
    # Option 2: Deploy with Ray Serve
    print("\nDeploying with Ray Serve...")
    deployer = RayDeployer("config/ai/serving_config.json")
    
    # Deploy the model
    deployment = deployer.deploy()
    
    print("Model deployed successfully!")
    print("Access the API at http://localhost:8000")


if __name__ == "__main__":
    main()
