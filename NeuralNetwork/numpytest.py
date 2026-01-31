import numpy as np

np.random.seed(0)

def spiral_data(points, classes):
    ##to create spiral data set goddamit
    X = np.zeros((points*classes, 2))#x,y coods
    y = np.zeros(points*classes, dtype='uint8')#the  class
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))# values
        r = np.linspace(0.0, 1, points)  #radius
        t = np.linspace(class_number*4, (class_number+1)*4,num=points) + np.random.randn(points)*0.2#the angle
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

class Layer_Random:
    def __init__(self,n_ip,n_neurons):
        self.weight =0.20*np.random.randn(n_ip,n_neurons)
        self.bias=np.zeros((1,n_neurons))
        #has to read from a predefined file or smthing when actually creating it
        
    def forward(self,ip):
        self.op = np.dot(ip,self.weight)+self.bias

class Re_Lu:
    def forward(self,ip):
        self.op=np.maximum(0,ip)

class softmaxx:
    def softmax(self,ip):
        norm = np.exp(ip - np.max(ip,axis=1,keepdims=True))
        pbty = norm /np.sum(norm,axis=1,keepdims=True)
        return pbty

X,y=spiral_data(100,3)

l1=Layer_Random(2,5)
a1=Re_Lu()

l2=Layer_Random(5,3)
a2=softmaxx()

l1.forward(X)
a1.forward(l1.op)

l2.forward(a1.op)
print(a2.softmax(l2.op))

