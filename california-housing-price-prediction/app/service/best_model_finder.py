from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
import mlflow
from config import Config
import time


class BestModelFinder:
    experiment_name = None
    client = MlflowClient()
    __best_model = None

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.client = MlflowClient()

        # populate best performing model
        try:
            self.__best_model, self.__best_model_name = self.__find_best_model()
        except Exception as e:
            print(f'Best Model Find Error: {e}')


    def get_best_model(self):
        if self.__best_model is not None:
            return self.__best_model, self.__best_model_name
        else:
            print('No Best Model Registered, finding it..')
            self.__best_model, self.__best_model_name = self.__find_best_model()

    def __get_best_run(self):
        # fin runs for given experiment
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{self.experiment_name}' does not exist")

        experiment_id = experiment.experiment_id
        runs = self.client.search_runs(experiment_ids=[experiment_id])

        if not runs:
            raise ValueError(f"No runs found for experiment '{self.experiment_name}'")

        # check if runs with 'r2_score' metric exist
        runs_with_r2 = [r for r in runs if "r2_score" in r.data.metrics]
        if not runs_with_r2:
            raise ValueError("No runs with 'r2_score' metric found")

        # find best run by highest R²
        best_run = max(runs_with_r2, key=lambda r: r.data.metrics["r2_score"])

        return best_run

    def __find_best_model(self):
        best_run = self.__get_best_run()
        best_run_model_name = best_run.info.run_name
        best_model_name = "BestPerformingModel"
        model_uri = f"runs:/{best_run.info.run_id}/{best_run_model_name}"

        # Register model from the best run
        # Check if registered model exists else create
        try:
            self.client.get_registered_model(best_model_name)
        except Exception as e:
            self.client.create_registered_model(best_model_name)

        model_version = self.client.create_model_version(name=best_model_name, source=model_uri, run_id=best_run.info.run_id)

        self.__wait_until_model_ready(best_model_name, model_version.version)

        best_model = mlflow.pyfunc.load_model(f"models:/{best_model_name}/{model_version.version}")

        return best_model, best_run_model_name

    def __wait_until_model_ready(self, model_name, version, timeout=30):
        for _ in range(timeout):
            model_version = self.client.get_model_version(name=model_name, version=version)
            if model_version.status == "READY":
                return
            time.sleep(1)
        raise TimeoutError(f"Model version {version} for '{model_name}' not READY after {timeout}s.")


best_model_finder = BestModelFinder(Config.MLFLOW_EXPERIMENT)
