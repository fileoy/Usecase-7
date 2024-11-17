import joblib

from pydantic import BaseModel
from fastapi import FastAPI

from sklearn.metrics import pairwise_distances_argmin_min

model = joblib.load('dbscan_model.joblib')
scaler = joblib.load('scaler.joblib')

class InputFeatures(BaseModel):
    appearance: int
    assists: float
    days_injured: int
    games_injured: int
    award: int
    highest_value: int

def preprocessing(input_features: InputFeatures):
    dict_f = {
        'appearance': input_features.appearance,
        'assists': input_features.assists,
        'days_injured': input_features.days_injured,
        'games_injured': input_features.games_injured,
        'award': input_features.award,
        'highest_value': input_features.highest_value
        }
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    # Scale the input features
    scaled_features = scaler.transform([list(dict_f.values())])

    return scaled_features


app = FastAPI()

# GET request
@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

# get request
@app.get("/items/")
def create_item(item: dict):
    return {"item": item}

@app.post("/predict")
async def predict(input_features: InputFeatures):
    try:
        data = preprocessing(input_features)
        # Find the closest cluster core sample for the new data
        core_samples = model.components_
        cluster_labels, _ = pairwise_distances_argmin_min(data, core_samples)
        cluster_label = model.labels_[model.core_sample_indices_[cluster_labels[0]]]
        return {"pred": int(cluster_label)}
    except Exception as e:
        return {"error": str(e)}