
# Manipulacao dos dados
import pandas as pd
import numpy as np
import time

# Machine Learning
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Importando Modulos
from pre_processamento_dados import X_train, X_test, y_train, y_test
from Classes.classe_grid_scores import GridScores

# Funcao que faz a procura da melhor funcao
def grid_search(Xtrain, ytrain, estimador, grid_parametros, cross_val = 3, avaliador_principal = 'roc_auc'):
    
    
    # Buscando melhor modelo
    busca_modelo = GridSearchCV(estimator = estimador,
                                param_grid = grid_parametros,
                                n_jobs = -1, cv = cross_val, refit = avaliador_principal, 
                                scoring = ['accuracy', 'roc_auc']).fit(Xtrain, ytrain)
    
    return busca_modelo

def apresentando_melhores_resultados(objeto_resultados):
    
    objeto_resultados.resultados_grid()
    
    print(f'Analisando melhores resultados!')
    print('-' * 60)
    print('Os melhores resultados são:')
    print(objeto_resultados.melhores_parametros)
    print('-' * 60)
    print('Os resultados obtidos com esses parâmetros foram:')
    print(f'A média de acurácia foi de: {objeto_resultados.acuracia_media}')
    print(f'O desvio-padrão de acurácia foi de: {objeto_resultados.desvio_padrao_media}')
    print(f'A média de ROC-AUC foi de: {objeto_resultados.roc_auc_media}')
    print(f'O desvio-padrão de ROC-AUC foi de: {objeto_resultados.roc_auc_desvio_padrao}')


                    ##### Random Forest #####
# Escolha de parametros
criterion = ['gini', 'entropy']
n_estimators = [50, 100, 500, 1000, 1500]
max_depth = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]


# Grid de Parametros
grid_parametros = {'criterion' : criterion, 'n_estimators' : n_estimators, 'max_depth' : max_depth}

# Estimador
RandomForest = RandomForestClassifier(random_state = 8, warm_start = True)


# Estimacao
start_time = time.time()
buscando_melhor_modelo = grid_search(X_train, y_train, RandomForest, grid_parametros)
end_time = time.time()
print('Tempo de execução: ', (end_time - start_time))

scores_melhor_modelo = GridScores(buscando_melhor_modelo)
apresentando_melhores_resultados(scores_melhor_modelo)






