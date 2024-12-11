import mlflow
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from typing import Dict, Any

ESTIMATOR_REGISTRY = {
    "LinearRegression": LinearRegression,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "KNeighborsRegressor": KNeighborsRegressor,
}

def set_experiment_for_estimator(estimator_name: str) -> None:
    experiment_name = f"{estimator_name}_Experiment"
    with open("profiles/local.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["experiment"]["name"] = experiment_name
    with open("profiles/local.yaml", "w") as f:
        yaml.dump(config, f)

    mlflow.set_experiment(experiment_name)

def log_parameters_to_mlflow(params: Dict[str, Any]) -> None:
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)

def estimator_fn(estimator_params: Dict[str, Any]):
    estimator_name = estimator_params.get("estimator")
    if not estimator_name:
        raise ValueError("Estimator not defined in 'estimator_params'.")

    model_params = {k: v for k, v in estimator_params.items() if k != "estimator"}
    set_experiment_for_estimator(estimator_name)
    log_parameters_to_mlflow(model_params)

    if estimator_name in ESTIMATOR_REGISTRY:
        return ESTIMATOR_REGISTRY[estimator_name](**model_params)

    raise ValueError(f"Unsupported estimator: {estimator_name}")
