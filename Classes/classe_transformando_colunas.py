
# Manipulacao dos dados
import pandas as pd
import numpy as np

# Processamento dos dados
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Salvando objetos em disco
import pickle

'''
Testes possíveis
    
    Para one-hot:
        Testar se o dataframe retornou o numero certo de colunas apos a transformacao.
    
    Para label encoding:
        Testar se o numero de valores unicos é o mesmo após a transformacao (para label encoding).
        Testar se os labels gerados sao os mesmos. Ex: Yes(1) e No(0). Gera de fato 1 e 0 corretamente? ( Ver se dá!)
    
    Ambos one-hot e label encoding:
        Testar se gerou valores nulos. Ex: soma de nulos no DF antes e depois. (para ambos os metodos de transformacao)
    
    Para ambos os métodos que salvam o encoder:
        Testar se o nome do arquivo foi salvo em pkl.
'''

class Transformacao_Colunas:
    
    def __init__(self, dataframe, colunas_onehot, colunas_label_enc):
        self.dataframe = dataframe
        self.colunas_onehot = colunas_onehot
        self.colunas_label_enc = colunas_label_enc
        self.enc = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
        self.le = LabelEncoder()
        self.lista_parametros = []
        self.transformacao_onehot()
        self.transformacao_label_enc()
        
    # Realizando a transformacao One-Hot Encoding
    def transformacao_onehot(self):
        self.dataframe = self.dataframe.reset_index()
        self.dataframe = self.dataframe.drop('index', axis = 1)
        
        # Loop para aplicar em mais colunas
        for col in self.colunas_onehot:
            novas_colunas = self.enc.fit_transform(self.dataframe[[col]])
            
            # Inserindo novas colunas no dataframe
            self.dataframe = pd.concat([self.dataframe,
                                       pd.DataFrame(novas_colunas, columns = self.enc.get_feature_names_out([col]))], axis = 1)
            
            # Removendo coluna original
            self.dataframe = self.dataframe.drop(col, axis = 1)
            
            # Deletanndo novas colunas
            del novas_colunas
        
        return None
   
    # Realizando a transformacao Label Encoding
    def transformacao_label_enc(self):
        # Aplicando nas colunas
        for col in self.colunas_label_enc:
            self.dataframe[col] = self.le.fit_transform(self.dataframe[col])
        
            # Obtendo as classificacoes para cada classe
            self.lista_parametros.append([self.le.classes_.tolist(), self.le.transform(self.le.classes_).tolist()])
        
        return None    

    
    # Salvando objeto encoder em disco
    def salvar_objeto_onehot(self, nome_arquivo):
        with open(nome_arquivo, 'wb') as f:
            pickle.dump(self.enc, f)
        return None
    
    # Salvando objeto encoder em disco
    def salvar_objeto_label_enc(self, nome_arquivo):
        with open(nome_arquivo, 'wb') as f:
            pickle.dump(self.le, f)
        return None





