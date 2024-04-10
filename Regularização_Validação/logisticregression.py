import numpy as np
from numpy import linalg as LA
import random


class LogisticRegression:
    def __init__(self, eta=0.1, tmax=30, batch_size=32, lam=0):
        self.eta = eta
        self.tmax = tmax
        self.batch_size = batch_size
        self.lam = lam

    # Infere o vetor w da funçao hipotese
    #Executa a minimizao do erro de entropia cruzada pelo algoritmo gradiente de descida
    def fit(self, _X, _y):
        self.w = []    
        X = np.concatenate((np.ones((len(_X),1), dtype=np.float128), np.array(_X, dtype=np.float128)), axis=1)
        y = np.array(_y, dtype=np.float128)
        
        d = X.shape[1]
        N = X.shape[0]
        w = np.zeros(d, dtype=np.float128)
    
        
        for i in range(self.tmax):
            vsoma = np.zeros(d, dtype=np.float128)

            #Escolhendo o lote de entradas
            if self.batch_size < N:
                indices = random.sample(range(N),self.batch_size)
                batchX = [X[index] for index in indices]
                batchY = [y[index] for index in indices]
            else:
                batchX = X
                batchY = y

            #computando o gradiente no ponto atual
            for xn, yn in zip(batchX, batchY):
                vsoma += (yn * xn) / (1 + np.exp((yn * w).T @ xn))
        
            gt = vsoma/self.batch_size 
            if self.lam != 0:
              gt += 2*self.lam * w
            
            #Condicao de parada: se ||deltaF|| < epsilon (0.0001)
            if LA.norm(gt) < 0.0001 :
                break
            w = w + (self.eta*gt)

        self.w = w

    #funcao hipotese inferida pela regressa logistica  
    def predict_prob(self, X):
        return [(1 / (1 + np.exp(-(self.w[0] + self.w[1:].T @ x)))) for x in X]

    #Predicao por classificação linear
    def predict(self, X):
        return [1 if (1 / (1 + np.exp(-(self.w[0] + self.w[1:].T @ x)))) >= 0.5 
                else -1 for x in X]

    def getW(self):
        return self.w

    def getRegressionY(self, regressionX, shift=0):
        return (-self.w[0]+shift - self.w[1]*regressionX) / self.w[2]