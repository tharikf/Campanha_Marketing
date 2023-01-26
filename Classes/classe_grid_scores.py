

# Manipulacao dos dados
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.model_selection import GridSearchCV


class GridScores:
    
    def __init__(self, grid_resultados):
        self.grid_resultados = grid_resultados
        
    def resultados_grid(self):
        
        # O melhor hiperparâmetro é:
        self.melhores_parametros = self.grid_resultados.best_params_
        
        # A média e o desvio padrao de acuracia para o melhor modelo
        self.acuracia_media = round(self.grid_resultados.cv_results_['mean_test_accuracy'][self.grid_resultados.best_index_], 4)
        self.desvio_padrao_media = round(self.grid_resultados.cv_results_['std_test_accuracy'][self.grid_resultados.best_index_], 4)
        
        # A média e o desvio padrão de ROC-AUC para o melhor modelo
        self.roc_auc_media = round(self.grid_resultados.cv_results_['mean_test_roc_auc'][self.grid_resultados.best_index_], 4)
        self.roc_auc_desvio_padrao = round(self.grid_resultados.cv_results_['std_test_roc_auc'][self.grid_resultados.best_index_], 4)



