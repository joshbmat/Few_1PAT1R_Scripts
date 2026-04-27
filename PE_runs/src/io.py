# for reading in yaml file
from ruamel.yaml import YAML
from typing import TextIO, Callable, TypeVar, ParamSpec
import os

def param_load(param_file_path: str) -> dict:
    """
    Wrapper for yaml.safe_load with preferred behavior in the event of
    an error
    
    Parameters
    ----------
    param_file_path : str
        Path to YAML parameter file
    
    Returns
    -------
    data : dict
        Dict with parameters (or empty)
    """
    yaml = YAML()
    yaml.preserve_quotes = True    # optional
    yaml.indent(mapping=2, sequence=4, offset=2)
    
    if os.path.exists(param_file_path):
        with open(param_file_path, "r") as f:
            data = yaml.load(f) or {}
    else:
        data = {}
    
    data = sanitize_yaml(data)
    
    return(data)

def sanitize_yaml(yaml_dict: dict) -> dict:
    """
    YAML has terrible default behavior for some inputs. In particular, it renders 
    value: None as a string 'None' rather than as NoneType. This function fixes
    that along with whatever else we come up with
    """
    for key, value in yaml_dict.items():
        if value == 'None' or value == 'none':
            yaml_dict[key] = None
        
        if isinstance(value, str):
            if "e" in value:
                try:
                    yaml_dict[key] = float(value)
                except:
                    pass
        elif isinstance(value, dict):
            yaml_dict[key] = sanitize_yaml(value)
        
    return(yaml_dict)