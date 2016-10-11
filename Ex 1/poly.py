import random
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def get_rand(a,b):
    return  a + (b-a)*random.random()

def get_x_vec(x,p):
    return np.array([x**j for j in range(p+1)])

class GenPolyData:
    def __init__(self,p,estd):
        self.p = p
        self.a = np.array([get_rand(-1.0,1.0) for i in range(p+1)])
        self.estd = estd

    def genera(self,n,ofile):
        #genera n esempi su file ofile
        f = open(ofile,'w')
        for i in range(n):
            #genera esempio
            x = get_rand(-1.0,1.0)
            y = np.dot(self.a,get_x_vec(x,self.p))+ np.random.normal(scale=self.estd)
            f.write("%.10f %.10f\n"%(x,y))
        f.close()
             
def get_data(ifile):
    # leggi i dati da file
    f = open(ifile)
    l = []
    for line in f:
        l.append([float(z) for z in line.strip().split()])
    A = np.array(l)
    f.close()
    return A[:,0],A[:,1]

class Regressor:
    def __init__(self,p, BETA=0.0):
        self.w = None
        self.p = p
        self.BETA = BETA
        print 'Building regressor with p=%d'%self.p

    def train(self,Xtr,Ytr,save_to=None):
        ntr = Ytr.size
        X = np.array([get_x_vec(x,self.p) for x in Xtr])
        Xt = np.transpose(X)
        self.w = np.dot(np.dot(inv(np.dot(Xt,X) + self.BETA*np.eye(self.p+1)),Xt),Ytr)
        y_pred = np.dot(X,self.w)
        mse = 0.0
        for (i,y) in enumerate(Ytr):
            mse+=(y_pred[i]-Ytr[i])**2
        mse/=ntr
        print 'training on %d examples, mse: %f'%(ntr,mse)
        if save_to:
            self.save_image(Xtr,Ytr,save_to)
        return mse
            
    def test(self,Xte,Yte,save_to=None):
        nte = Yte.size
        X = np.array([get_x_vec(x,self.p) for x in Xte])
        y_pred = np.dot(X,self.w)
        mse = 0.0
        for (i,y) in enumerate(Yte):
            mse+=(y_pred[i]-Yte[i])**2
        mse/=nte
        print 'testing on %d examples, mse: %f'%(nte,mse)
        if save_to:
            self.save_image(Xte,Yte,save_to)
        return mse

    def save_image(self,X,Y,save_to):
        plt.clf()
        plt.plot(X, Y,'bo')
        plt.xlabel("x")
        plt.ylabel("y")
        xvals = np.linspace(-1.0,1.0,100)
        yvals = [np.dot(get_x_vec(x,self.p),self.w) for x in xvals]
        plt.plot(xvals,yvals,'r')
        plt.savefig(save_to,format='png')


def do_prova():
    Xtr,Ytr = get_data('dati.p05.s01.tr')
    Xte,Yte = get_data('dati.p05.s01.te')

    tr_mse = []
    te_mse = []

    for p in range(1,21):
        R = Regressor(p,0.01)
        tr_mse.append(R.train(Xtr,Ytr,'fig_train.%02d.png'%p))
        te_mse.append(R.test(Xte,Yte,'fig_test.%02d.png'%p))

    plt.clf()
    plt.plot(range(1,21), tr_mse,'b')
    plt.plot(range(1,21), te_mse,'r')
    plt.xlabel("degree")
    plt.ylabel("mse")
    plt.savefig("mse.png",format='png')
  
    
