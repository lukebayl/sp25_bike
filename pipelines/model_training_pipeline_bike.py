import joblib
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from sklearn.metrics import mean_absolute_error

import src.config as config
from src.data_utils_bike import transform_ts_data_into_features_and_target
from src.inference_bike import (
    fetch_days_data,
    get_hopsworks_project,
    load_metrics_from_registry,
    load_model_from_registry,
)
from src.pipeline_utils_bike import get_pipeline

print("📦 Fetching data from group store ...")
ts_data = fetch_days_data(180)

print("🛠️ Transforming to ts_data ...")
features, targets = transform_ts_data_into_features_and_target(
    ts_data, window_size=24 * 28, step_size=23
)

pipeline = get_pipeline()
print("🧠 Training model ...")
pipeline.fit(features, targets)

predictions = pipeline.predict(features)
test_mae = mean_absolute_error(targets, predictions)

metric = load_metrics_from_registry()
print(f"📊 The new MAE is {test_mae:.4f}")
print(f"📉 The previous MAE is {metric['test_mae']:.4f}")

if test_mae < metric.get("test_mae"):
    print("✅ Registering new improved model ...")
    model_path = config.MODELS_DIR / "lgb_pca_model.pkl"
    joblib.dump(pipeline, model_path)

    input_schema = Schema(features)
    output_schema = Schema(targets)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.sklearn.create_model(
        name="bike_demand_predictor_next_hour",
        metrics={"test_mae": test_mae},
        input_example=features.sample(),
        model_schema=model_schema,
    )
    model.save(model_path)
else:
    print("⛔ Skipping model registration because new model is not better!")
