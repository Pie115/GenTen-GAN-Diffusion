'''Defines the ModelZoo class, which allows for easy switching of models'''

from TenGAN.ggan.rescal import RescalGenerator
from .tgan import Discriminator3d, Discriminator4d
from .new_tgan import TensorGenerator, NormalGenerator, NormalGeneratorCNN
from .baseline_gan import BaselineGenerator, gen_3dGAN

class ModelZoo:
    def __init__(self):
        self.models = {
            'TensorGenerator':TensorGenerator,
            'MyBaselineGenerator': NormalGenerator,
            'MyBaselineGeneratorCNN': NormalGeneratorCNN,
            'Discriminator3d': Discriminator3d,
            'Discriminator4d': Discriminator4d,
            'RescalGenerator': RescalGenerator,
            'BaselineGenerator':BaselineGenerator,
            '3DGAN':gen_3dGAN
        }

    def get_model(self, model_name):
        '''Given a model name, returns the model class'''
        if model_name not in self.models:
            raise Exception(
                f'Unknown model specified. Valid options are: {self.models.keys()}')
        return self.models[model_name]

    def has_model(self, model_name):
        '''Given a model name, return whether or not it exists'''
        return model_name in self.models

