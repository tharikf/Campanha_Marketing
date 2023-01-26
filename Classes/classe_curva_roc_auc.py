
# Manipulacao dos dados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning
from sklearn import metrics

class CurvaRocAuc:
    
    def __init__(self, Ytest, previsao, modelo):
        self.Ytest = Ytest
        self.previsao = previsao
        self.modelo = modelo
        
    def criando_curva_roc_auc(self):
        
        fpr, tpr, thresholds = metrics.roc_curve(self.Ytest, self.previsao)
        roc_auc = metrics.roc_auc_score(self.Ytest, self.previsao)
        
        # Plot
        plt.title('ROC Curve do modelo: ' + self.modelo + '!')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        return plt.show()






