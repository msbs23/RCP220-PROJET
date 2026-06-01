from api_radio.model import load_model

class ModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            print("Chargement du modèle en mémoire...")
            cls._instance = load_model()
            print("Modèle prêt !")
        return cls._instance