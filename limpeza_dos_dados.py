
# Manipulacao dos dados
import pandas as pd
import numpy as np


'''
        Informações sobre o dataset:

        https://archive.ics.uci.edu/ml/datasets/bank+marketing

'''

# Carregando os dados
df = pd.read_csv('Dados/bank_marketing.csv', sep = ';')

# Funcao que faz a limpeza e preparacao dos dados
def limpando_dados(dataframe):
    
    # Retirando o "." no "admin" da coluna job
    dataframe['job'] = np.where(dataframe['job'] == 'admin.', 'admin', dataframe['job'])
    
    # Colocando em letras maiúsculas
    dataframe = dataframe.apply(lambda x: x.str.title() if x.dtype == 'object' else x)
    
    
    # Retirando o -1 e botando para 0 quando o cliente não é contatado.
    dataframe['pdays'] = np.where(dataframe['pdays'] == -1, 0, dataframe['pdays'])
    
    # Criando faixa etaria
    dataframe['faixa_etaria'] = np.where(dataframe['age'] < 25, 'Menor que 25 anos',
                                np.where((dataframe['age'] >= 25) & (dataframe['age'] < 35), '25-34 anos',
                                np.where((dataframe['age'] >= 35) & (dataframe['age'] < 45), '35-44 anos',
                                np.where((dataframe['age'] >= 45) & (dataframe['age'] < 55), '45-54 anos',
                                np.where((dataframe['age'] >= 55) & (dataframe['age'] < 65), '55-64 anos', '65 anos ou mais')))))
    
                    
    ordem_faixa_etaria_dict = {'Menor que 25 anos' : 1, '25-34 anos' : 2, '35-44 anos' : 3,
                               '45-54 anos' : 4, '55-64 anos' : 5, '65 anos ou mais' : 6}
    
    dataframe['ordem_faixa_etaria'] = dataframe['faixa_etaria'].map(ordem_faixa_etaria_dict)
    

    # Transformando a coluna month para o nome do mês completo
    dicio_mes = {'Jan' : 'January', 'Feb' : 'February', 'Mar' : 'March', 'Apr' : 'April',
                 'May' : 'May', 'Jun' : 'June', 'Jul' : 'July', 'Aug' : 'August',
                 'Sep' : 'September', 'Oct' : 'October', 'Nov' : 'November', 'Dec' : 'December'}
    
    dataframe['month'] = dataframe['month'].map(dicio_mes)
    
    # Transformando coluna month em data
    dataframe['month'] = pd.to_datetime(dataframe['month'], format = '%B')
    dataframe['month_numero'] = dataframe['month'].dt.month
    dataframe['month'] = dataframe['month'].apply(lambda x: x.strftime('%B')) 
    
    # Renomeando colunas
    dataframe.columns = ['Age', 'Job', 'Marital_Status', 'Education', 'Default', 'Balance', 'Housing',
                  'Loan', 'Contact', 'Day', 'Month', 'Duration', 'Campaign', 'Pdays', 'Previous', 'Poutcome',
                  'Client_Conversion', 'Faixa_Etaria', 'Ordem_Faixa_Etaria','Ordem_Mes']
    
    
    return dataframe

df_final = limpando_dados(df)

# Salvando dados para o Power BI
'''
df_final.to_csv('df_para_analise_power_BI.csv', index = False)
'''

