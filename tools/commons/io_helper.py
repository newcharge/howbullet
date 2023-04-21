import json
import pickle
import numpy as np


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_xyz(filepath):
    return np.loadtxt(filepath)