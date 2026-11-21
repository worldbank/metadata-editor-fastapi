import os

# Never let a test reach the network for LiteLLM's price map.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
