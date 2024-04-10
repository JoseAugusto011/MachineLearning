from linearregression import LinearRegression
import numpy
#from pocketpla import PocketPLA

class LRClassifier():
    def fit(self, _X, _y):
        lr = LinearRegression()
        lr.execute(_X, _y)
        self.w = lr.get_w()
        
        #pla = PocketPLA()
        #pla.set_w(self.w)
        #pla.execute(_X, _y)
        #self.w = pla.get_w()
        
        
    def predict(self, _X):
        return [numpy.sign(numpy.dot(self.w, x)) for x in _X]
     
    def getRegressionY(self, regressionX, shift=0):
        return (-self.w[0]+shift - self.w[1]*regressionX) / self.w[2]