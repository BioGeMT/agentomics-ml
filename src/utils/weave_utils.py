from rich.console import Console
from utils.openrouter_models import OpenRouterModelsUtils

# Weave Integration Functions
def setup_weave_costs(weave_client, model_id: str, api_key: str) -> bool:
    """Set up cost tracking in Weave for the specified model."""
    console = Console()
    console.print(f"Setting up cost tracking for model: {model_id}")
    
    api = OpenRouterModelsUtils(api_key)
    models = api.fetch_models()
    
    if not models:
        console.print(f"No cost information available for model: {model_id}")
        console.print(f"Cost tracking will be disabled for this session")
        return False
    
    # Find the model and get its costs
    for model in models:
        if model.get("id") == model_id:
            pricing = model.get("pricing", {})
            if pricing:
                try:
                    input_cost = float(pricing.get("prompt", "0"))
                    output_cost = float(pricing.get("completion", "0"))
                    
                    # Add costs to Weave
                    weave_client.add_cost(
                        llm_id=model_id,
                        prompt_token_cost=input_cost,
                        completion_token_cost=output_cost
                    )
                    
                    console.print(f"Cost tracking enabled:")
                    console.print(f"   • Model: {model_id}")
                    console.print(f"     - Prompt tokens: ${input_cost:.8f}/token")
                    console.print(f"     - Completion tokens: ${output_cost:.8f}/token")
                    
                    return True
                except (ValueError, TypeError) as e:
                    console.print(f"Invalid pricing format: {e}")
                    return False
    
    console.print(f"Model {model_id} not found")
    return False

def init_weave_with_costs(project_name: str, model_id: str, api_key: str):
    """Initialize Weave and set up cost tracking for the specified model."""
    import weave
    console = Console()
    console.print(f"Initializing Weave for project: {project_name}")
    
    try:
        weave_client = weave.init(project_name)
        
        # Set up costs for the model
        setup_success = setup_weave_costs(weave_client, model_id, api_key)
        
        if setup_success:
            console.print(f"Weave integration ready with cost tracking")
        else:
            console.print(f"Weave initialized but cost tracking unavailable")
        
        return weave_client
        
    except Exception as e:
        console.print(f"Failed to initialize Weave: {e}")
        return None