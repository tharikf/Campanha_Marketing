
# Manipulacao dos dados
import pandas as pd
import numpy as np

# Machine Learning
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class MetricasAvaliacao:
    
    def __init__(self, Ytest, previsao):
        self.Ytest = Ytest
        self.previsao = previsao

        # Inicializando atributos que serao construindos
        self.acuracia = None
        self.roc_auc = None
        self.matriz_confusao = None
        self.tabela_report = None

        
    def obtendo_metricas(self):
        
        self.acuracia = round(accuracy_score(self.Ytest, self.previsao), 4)
        self.roc_auc = round(metrics.roc_auc_score(self.Ytest, self.previsao), 4)
        
    def obtendo_matriz_confusao(self):
        self.matriz_confusao = metrics.confusion_matrix(self.Ytest, self.previsao)
        self.matriz_confusao = pd.DataFrame(self.matriz_confusao)
        return self.matriz_confusao
        
    def modelo_report(self):
        self.nome_target = ['No', 'Yes']
        self.tabela_report = classification_report(self.Ytest, self.previsao, target_names = self.nome_target)
        return self.tabela_report
        
    





