import os
import pickle
import numpy as np

model_path = r"C:\Users\moide\Downloads\new\model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

weights = model["weights"]
bias = model["bias"]
mean = model["mean"]
std = model["std"]

def predict(features):
    features = np.array(features)
    features = (features - mean) / std
    return np.dot(features, weights) + bias