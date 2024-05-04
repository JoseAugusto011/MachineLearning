import numpy as np

class PocketPLA():
    def get_w(self):
        return self.w
    
    def set_w(self, w):
        self.w = w

    def execute(self, _X, y):
        X = np.array(_X)
        hasWrongClassifiedPoint = True
        bestError = len(y)
        self.w = np.zeros(len(X[0]))
        bestW = self.w
        for iter in range(1000):
            
            #Testa se sign(wTXn) != Yn - ponto classificado errado
            hasWrongClassifiedPoint = np.sign(np.dot(self.w, X.T)) != y
            if not hasWrongClassifiedPoint:
                break
            for i in range(len(y)):
                if(np.sign(np.dot(self.w, X[i])) != y[i]):
                    self.w = self.w + (y[i]*X[i])
                    eIN = self.errorIN(X, y)
                    if(bestError > eIN):
                        bestError = eIN
                        bestW = self.w
        self.w = bestW
                    
    def getOriginalY(self, originalX):
        return (-self.w[0] - self.w[1]*originalX) / self.w[2]
    
    def h(self, x):
        return np.sign(np.dot(self.w, x))
    
    def errorIN(self, X, y):
        error = 0
        for i in range(len(y)):
            if(np.sign(np.dot(self.w, X[i])) != y[i]):
                error += 1
                
        return error