from datetime import datetime, timedelta
import pandas as pd

import src.config as config
from src.inference_bike import (
    get_feature_store,
    get_model_predictions,
    load_model_from_registry,
)
from src.data_utils_bike import transform_ts_data_into_features

# Get the current datetime64[us, Etc/UTC]
current_date = pd.Timestamp.now(tz="Etc/UTC")

# Connect to feature store
feature_store = get_feature_store()

# Read time-series data from the feature view
fetch_data_to = current_date - timedelta(hours=1)
fetch_data_from = current_date - timedelta(days=1 * 29)
print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

feature_view = feature_store.get_feature_view(
    name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
)

ts_data = feature_view.get_batch_data(
    start_time=(fetch_data_from - timedelta(days=1)),
    end_time=(fetch_data_to + timedelta(days=1)),
)

# Filter and clean timestamps
ts_data = ts_data[ts_data.start_hour.between(fetch_data_from, fetch_data_to)]
ts_data["start_hour"] = ts_data["start_hour"].dt.tz_localize(None)
ts_data.sort_values(["start_location_id", "start_hour"], inplace=True)
ts_data.reset_index(drop=True, inplace=True)

# Feature engineering
features = transform_ts_data_into_features(ts_data, window_size=24 * 28, step_size=23)

# Load model and predict
model = load_model_from_registry()
predictions = get_model_predictions(model, features)
predictions["start_hour"] = current_date.ceil("h")

# Insert into feature group
feature_group = feature_store.get_or_create_feature_group(
    name=config.FEATURE_GROUP_MODEL_PREDICTION,
    version=1,
    description="Predictions from LGBM PCA model for CitiBike",
    primary_key=["start_location_id", "start_hour"],
    event_time="start_hour",
)

feature_group.insert(predictions, write_options={"wait_for_job": False})
print("✅ Predictions inserted successfully.")
