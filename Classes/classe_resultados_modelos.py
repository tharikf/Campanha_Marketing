
# Manipulacao dos dados
import pandas as pd
import numpy as np

# Importando modulo MetricasAvaliacao
from Classes.classe_metricas_avaliacao import MetricasAvaliacao

class ResultadosModelos(MetricasAvaliacao):
    
    def __init__(self, Ytest, previsao, nome_modelo):
        self.Ytest = Ytest
        self.previsao = previsao
        self.nome_modelo = nome_modelo

        
    def armazenando_resultados(self):
        
        # Criando lista vazia
        try:
            lista_resultados
        except NameError:
            resultado = False
            lista_resultados = []
        else:
            resultado = True
        

        # Acuracia e Roc_Auc herdando de MetricasAvaliacao
        self.dicionario_modelo = {'modelo' : self.nome_modelo, 'acuracia' : self.acuracia, 'roc_auc' : self.roc_auc}
        lista_resultados.append(self.dicionario_modelo)
        self.df_resultados = pd.DataFrame(lista_resultados)
        return self.df_resultados
        




