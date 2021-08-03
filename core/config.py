import os
import sys
import json       
            
def prep_config(config_file):
    with open(config_file) as config_params:
        print(f"loading config file {config_file}")
        config = json.load(config_params)
    return config