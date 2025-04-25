import pickle
import json
import numpy as np
from loguru import logger as lg
from typing import Union, List, Dict, Optional, Any
from pathlib import Path


# Save and load using pickle
def out_pickle(data: Any, path: str | Path, name: str, verbose: bool = True):
    try:
        with open(str(path) + f"/{name}.p", "wb") as d:
            pickle.dump(data, d, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                lg.info(f"Object has been pickled to: {str(path) + f'/{name}.p'}")
    except:
        lg.error("An exception occured while saving the data as pickle...")


def in_pickle(path: str | Path, verbose: bool = False):
    with open(str(path), "rb") as f:
        obj = pickle.load(f)
    if verbose:
        lg.info("Pickled object has been imported!")
    return obj


# Load using json
def in_json(path: str | Path):
    """Loads from json file."""
    with open(path, "r", encoding="utf-8") as d:
        return json.load(d)


def out_json(path: str | Path, data):
    """Output to json file."""
    with open(path, "w", encoding="utf-8") as d:
        json.dump(data, d)


def np_save(data, path=None):
    np.save(path, data)
    lg.success(f"Result saved to {path}...")


def np_load(path=None):
    return np.load(path)


def in_txt(data: str, path: str | Path):
    with open(str(path), "r") as file:
        return file.read(data)


def out_txt(data: str, path: str | Path):
    with open(str(path), "w") as file:
        file.write(data)


def load_tokens(dir: Union[Path, str]):
    return in_json(dir)["ids"]
