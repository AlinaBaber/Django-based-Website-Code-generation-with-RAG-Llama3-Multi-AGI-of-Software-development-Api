from django.apps import AppConfig
from . import model_loader 
class SoftwareDevelopmentapiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'softwaredevelopmentapi'

    
    def ready(self):
        #from . import model_loader  # Import a module where you handle model loading
        model_loader.load_all_models()  # Function to load models
