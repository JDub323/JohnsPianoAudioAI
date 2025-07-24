import yaml

def load_configs(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
