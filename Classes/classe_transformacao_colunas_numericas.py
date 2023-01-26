
# Manipulacao dos dados
import pandas as pd
import numpy as np

# Processamento dos dados
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTENC



class TransformacaoColunasNumericas:
    
    def __init__(self, df_x_train, df_x_test, df_y_train, vetor_colunas):
        self.df_x_train = df_x_train
        self.df_x_test = df_x_test
        self.df_y_train = df_y_train
        self.vetor_colunas = vetor_colunas
        self.robust_scaler()
        self.smote_nc()
        
    # Robust Scaler
    def robust_scaler(self):
        # Treinando o scaler
        scaler = RobustScaler().fit(self.df_x_train[self.vetor_colunas])
        
        # Aplicando em dados de treino e teste
        self.df_x_train[self.vetor_colunas] = scaler.transform(self.df_x_train[self.vetor_colunas])
        self.df_x_test[self.vetor_colunas] = scaler.transform(self.df_x_test[self.vetor_colunas])
    
    # Balanceamento de Classes
    def smote_nc(self):
        oversampled = SMOTENC(categorical_features = [1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 15, 16,
                                                        17, 18, 19, 20, 21, 22, 23, 24, 25], random_state = 8)
        self.df_x_train, self.df_y_train = oversampled.fit_resample(self.df_x_train, self.df_y_train)





