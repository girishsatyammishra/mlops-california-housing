import joblib


class ModelManager:
    models = {}
    models_dump_key = 'models.joblib'

    def __init__(self):
        # load models from local
        try:
            loaded_models = joblib.load(self.models_dump_key)
            if loaded_models is not None:
                self.models = loaded_models
        except Exception as e:
            print(f'Error loading saved models. Error: {e}')

    def save_model(self, model_name, model):
        try:
            if model_name is not None and model is not None:
                self.models[model_name] = model # append to current models dict
                joblib.dump(self.models, self.models_dump_key) # save updated models dict
            else:
                raise Exception('Invalid model to be saved')
        except Exception as e:
            raise Exception(f'Error saving model. Error: {e}')

    def get_model(self, model_name):
        if model_name in self.models:
            return self.models[model_name]
        else:
            return None


modelManager = ModelManager()