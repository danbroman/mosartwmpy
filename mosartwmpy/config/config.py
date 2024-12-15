import importlib.resources
from benedict import benedict
from benedict.dicts import benedict as Benedict


def get_config(config_file_path: str = None) -> Benedict:
    """Loads and merges configuration files into a Benedict object.
    
    Args:
        config_file_path (string): path to the user defined configuration yaml file
    
    Returns:
        Benedict: A Benedict instance containing the merged configuration
    """
    with importlib.resources.path('mosartwmpy', 'config_defaults.yaml') as config_defaults_path:
        config = benedict(str(config_defaults_path), format='yaml')
    if config_file_path is not None and config_file_path != '':
        if not isinstance(config_file_path, str):
            config_file_path = str(config_file_path)
        config.merge(benedict(config_file_path, format='yaml'), overwrite=True)
        config.merge(benedict(str(config_file_path), format='yaml'), overwrite=True)
    
    return config
