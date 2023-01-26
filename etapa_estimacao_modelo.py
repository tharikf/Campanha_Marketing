
# Manipulacao dos dados
import pandas as pd
import numpy as np
import time

# Machine Learning
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Importando Modulos
from pre_processamento_dados import X_train, X_test, y_train, y_test
from Classes.classe_metricas_avaliacao import MetricasAvaliacao
from Classes.classe_resultados_modelos import ResultadosModelos
from Classes.classe_curva_roc_auc import CurvaRocAuc



# Funcao que apresenta os resultados
def apresentando_resultados(avaliador, salvando_modelo, curva_roc_auc):
    
    print('Função que apresenta os resultados do modelo estimado!')
    print('-' * 60)
    
    # Apresentando Matriz de Confusao
    print('Matriz de confusão!')
    print(avaliador.obtendo_matriz_confusao())
    
    # Apresentando Precision, Recall e etc..
    print('-' * 60)
    print(avaliador.modelo_report())
    
    # Salvando modelo em DF para comparação
    salvando_modelo.obtendo_metricas()
    print('-' * 60)
    print(salvando_modelo.armazenando_resultados())
    
    # Plotando Curva ROC-AUC
    print('-' * 60)
    curva_roc_auc.criando_curva_roc_auc()


                    # Rodando modelo XGBoost #
start_time = time.time()
modelo = XGBClassifier(n_jobs = 12, random_state = 8, n_estimators = 1500, max_depth = 15, learning_rate = 0.1)
modelo.fit(X_train, y_train)
end_time = time.time()
print('Tempo de execução: ', (end_time - start_time))

# Fazendo previsao
previsao = modelo.predict(X_test)

# Objeto de Avaliacao
objeto_avaliacao = MetricasAvaliacao(y_test, previsao)

# Salvando os Resultados
objeto_guardando_resultados = ResultadosModelos(y_test, previsao, 'XGBoost')

# Plotando a curva ROC-AUC
objeto_plotando_curva = CurvaRocAuc(y_test, previsao, 'XGBoost')
apresentando_resultados(objeto_avaliacao, objeto_guardando_resultados, objeto_plotando_curva)

'''
                    XGBoost sem DURATION e sem MES
                                    
                # 49 segundos #
                No - Precision = 0.90
                Yes - Precision = 0.45

                Accuracy = 0.8751
                ROC_AUC = 0.6036
'''


"""
                    # Rodando modelo CatBoost #
start_time = time.time()
modelo = CatBoostClassifier(thread_count = 12, random_state = 8, learning_rate = 0.1, max_depth = 15, n_estimators = 500)
modelo.fit(X_train, y_train)
end_time = time.time()
print('Tempo de execução: ', (end_time - start_time))

# Fazendo previsao
previsao = modelo.predict(X_test)

# Objeto de Avaliacao
objeto_avaliacao = MetricasAvaliacao(y_test, previsao)

# Salvando os Resultados
objeto_guardando_resultados = ResultadosModelos(y_test, previsao, 'CatBoost')

# Plotando a curva ROC-AUC
objeto_plotando_curva = CurvaRocAuc(y_test, previsao, 'CatBoost')
apresentando_resultados(objeto_avaliacao, objeto_guardando_resultados, objeto_plotando_curva)

'''
                    CatBoost
                
                # 8 minutos#
                No - Precision - 0.91
                Yes - Precision - 0.45

                Accuracy - 0.8742
                ROC_AUC - 0.6098
'''
"""

