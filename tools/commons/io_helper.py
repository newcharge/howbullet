import json
import pickle


def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def load_pickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)
