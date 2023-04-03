from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import pickle

class EnvStorage:
    prices_cache = {}
    features_cache = {}

    @staticmethod
    def get_prices(filename):
        if filename in EnvStorage.prices_cache:
            return EnvStorage.prices_cache[filename]
        else:
            return None
        
    @staticmethod
    def set_prices(filename, data):
        EnvStorage.prices_cache[filename] = data

    @staticmethod
    def get_features(filename):
        if filename in EnvStorage.features_cache:
            return EnvStorage.features_cache[filename]
        else:
            return None
        
    @staticmethod
    def set_features(filename, data):
        EnvStorage.features_cache[filename] = data