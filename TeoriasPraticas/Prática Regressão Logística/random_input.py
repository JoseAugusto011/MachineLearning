import random
import numpy as np
from matplotlib import pyplot as plt

class RandomInput:
    def create_f(self):
        #gera os pontos p1 e p2 que formam a funcao f
        self.p1 = (random.uniform(-1,1), random.uniform(-1,1)) 
        self.p2 = (random.uniform(-1,1), random.uniform(-1,1))
        #computa o m da forma canonica de f
        self.m = (self.p2[1] - self.p1[1])/(self.p2[0] - self.p1[0])
    
    def f_canonica(self, x):
        return self.m*(x-self.p1[0]) + self.p1[1]

    def get_linear_input(self, N):
        self.create_f()
        #gera n numeros aleatorios, que não passam em f
        X = []
        y = []
        i = 0
        while i < N:
            x = [random.uniform(-1,1), random.uniform(-1,1)] 
            
            if(np.abs(x[1] - self.f_canonica(x[0])) < 0.0001) : # testa se o ponto está em cima da linha
                continue
                
            if(x[1] > self.f_canonica(x[0])) :
                y.append(+1)
            else :
                y.append(-1)
               
            X.append(x)
            i += 1          
            
        return X, y

def draw(X, y, rIN):
    N = len(y)
    #plota os pontos aleatórios de entrada da regressao linear
    xP = [X[i][0] for i in range(N) if(y[i] > 0)]
    yP = [X[i][1] for i in range(N) if(y[i] > 0)]
    xN = [X[i][0] for i in range(N) if(y[i] < 0)]
    yN = [X[i][1] for i in range(N) if(y[i] < 0)]

    plt.scatter(xP, yP, color='blue', marker='x', s=100)
    plt.scatter(xN, yN, color='red', marker='x', s=100)

    #desenha a função (reta) original que classificou os pontos
    xx = [-1, +1]
    yy = [rIN.f_canonica(xx[0]), rIN.f_canonica(xx[1])]
    plt.plot(xx, yy, 'g-', label='f(.)')
    plt.legend(loc='upper right')
    plt.xlim(-1,1)
    plt.ylim(-1,1)