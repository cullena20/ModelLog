"""
Model paths to eval on, to not redownload models everytime
"""

# include dictionary of model names to paths
# can modify to local model paths to avoid redownloading models everytime
MODEL_PATHS = {
    # Gemma-3 
    "Gemma-3-270M":   "google/gemma-3-270m",    
    "Gemma-3-1B-pt":  "google/gemma-3-1b-pt",
    "Gemma-3-4B-pt":  "google/gemma-3-4b-pt",
    "Gemma-3-12B-pt": "google/gemma-3-12b-pt",

    # Llama-3 
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",

    # Qwen3
    "Qwen3-0.6B":  "Qwen/Qwen3-0.6B-Base",
    "Qwen3-1.7B":  "Qwen/Qwen3-1.7B-Base",
    "Qwen3-4B":    "Qwen/Qwen3-4B-Base",
    "Qwen3-8B":    "Qwen/Qwen3-8B-Base",
    "Qwen3-14B":   "Qwen/Qwen3-14B-Base",
}
