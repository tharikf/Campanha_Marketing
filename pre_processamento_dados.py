
# Manipulacao dos dados
import pandas as pd
import numpy as np

# Processamento dos dados
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Salvar Objetos em Disco
import pickle

# Importando Classes e Funcoes
from Classes.classe_transformando_colunas import Transformacao_Colunas
from Classes.classe_transformacao_colunas_numericas import TransformacaoColunasNumericas


# Importando dados
from limpeza_dos_dados import df_final


# Removendo colunas que nao serão utilizadas
# Retirar coluna duration (conforme explica na pagina do dataset)
df = df_final
df = df.drop(columns = ['Faixa_Etaria', 'Ordem_Faixa_Etaria', 'Ordem_Mes', 'Day', 'Month', 'Duration'])


'''
Estratégia de transformação

Colunas:

    One-Hot Encoding:
        Job
        Contact
    
    Label Encoding:
        Marital_Status
        Education
        Default
        Housing
        Loan
        poutcome
        Client_Conversion

'''

# Selecionando colunas que serão transformadas
colunas_one_hot_encoding = ['Job', 'Contact']
colunas_label_encoding = ['Marital_Status', 'Education', 'Default', 'Housing', 'Loan', 'Poutcome', 'Client_Conversion']

# Iniciando o objeto
objeto_transformacao = Transformacao_Colunas(df, colunas_one_hot_encoding, colunas_label_encoding)

# Aplicando e salvando o dataframe com as alteracoes
df_final = objeto_transformacao.dataframe


# Salvando os objetos encoders
_ = Pipeline([
    ('Salvando Transformador One-Hot Encoding', objeto_transformacao.salvar_objeto_onehot('enc.pkl'),
    ('Salvando Transformador Label Encoding', objeto_transformacao.salvar_objeto_label_enc('le.pkl')),
    )
])


'''
Transformando colunas após o split treino e teste

    Split entre colunas de treino e teste.
        Aplicar RobustScaler.
        Aplicar o SMOTENC.

'''

def split_treino_teste(dataframe):
    
    # Separando entre variáveis independentes e dependentes.
    X = dataframe.loc[:, dataframe.columns != 'Client_Conversion']
    Y = dataframe.loc[:, 'Client_Conversion']
    
    Xtreino, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.30, random_state = 8)
    
    return Xtreino, Xtest, Ytrain, Ytest


# Split entre treino e teste
X_train, X_test, y_train, y_test = split_treino_teste(df_final)

# Colunas em que serão aplicadas o scaler
colunas_aplicar_robust_scaler = ['Age', 'Balance', 'Campaign', 'Pdays', 'Previous']

# Iniciando o objeto
transformacao_numericas = TransformacaoColunasNumericas(X_train, X_test, y_train, colunas_aplicar_robust_scaler)

# Salvando as modificações nos dados de treino e teste
X_train = transformacao_numericas.df_x_train
X_test = transformacao_numericas.df_x_test
y_train = transformacao_numericas.df_y_train

# Conferindo o tamanho de cada dataset após as transformações

"""
print('Tamanho do dataset de treino: ' + str(len(X_train)))
print('Tamanho do dataset de teste: ' + str(len(X_test)))
print('Tamanho do dataset de teste: ' + str(len(y_train)))
print('Tamanho do dataset de teste: ' + str(len(y_test)))
"""

