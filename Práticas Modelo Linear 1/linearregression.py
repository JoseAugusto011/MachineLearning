import numpy as np

class LinearRegression:
    def execute(self, _X, _y):
        X =  np.array(_X)
        y =  np.array(_y)
        xTx = np.dot(X.transpose(), X)
        inverse = np.linalg.inv(xTx)
        self.w = np.dot(np.dot( inverse, X.transpose()), y)
    
    def predict(self, _x):
        return [np.dot(self.w, xn) for xn in _x]
    
    def get_w(self):
        return self.w