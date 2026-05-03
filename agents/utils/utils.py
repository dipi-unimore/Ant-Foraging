import json

def read_params(
    env_params_path: str,
    learning_params_path: str,
    visualizer_params_path: str,
    logger_params_path: str
):
    env_params, learning_params, visual_params, log_params = dict(), dict(), dict(), dict()

    try:
        with open(learning_params_path) as f:
            learning_params = json.load(f)
    except Exception as e:
        print(f"[ERROR] could not open learning params file: {e}")
    
    try:
        with open(env_params_path) as f:
            env_params = json.load(f)
    except Exception as e:
        print(f"[ERROR] could not open params file: {e}")

    try:
        with open(visualizer_params_path) as f:
            visual_params = json.load(f)
    except Exception as e:
        print(f"[ERROR] could not open visualizer params file: {e}")
    
    try:
        with open(logger_params_path) as f:
            log_params = json.load(f)
    except Exception as e:
        print(f"[ERROR] could not open logger params file: {e}")
        
    return env_params, learning_params, visual_params, log_params