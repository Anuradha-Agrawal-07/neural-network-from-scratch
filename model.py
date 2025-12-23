import numpy as np

def relu(x):
    return np.maximum(0,x)

def derivative_relu(x):
    return (x>0).astype(float)

def sigmoid(x):
    x=np.clip(x,-500,500)
    return 1/(1+np.exp(-x))

np.random.seed(42)

W1=np.random.randn(6,8)*0.01
b1=np.zeros((1,8))
W2=np.random.randn(8,1)*0.01
b2=np.zeros((1,1))

def forward_pass(X):
    Z1=X@W1+b1
    A1=relu(Z1)
    Z2=A1@W2+b2
    A2=sigmoid(Z2)
    return Z1,A1,Z2,A2

def compute_loss(y_true,y_pred):
    return -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))

def backpropagation(X,y_true,Z1,A1,Z2,A2):
    m=y_true.shape[0]

    dZ2=A2-y_true
    dW2=(A1.T@dZ2)/m
    db2=np.sum(dZ2,axis=0,keepdims=True)/m

    dA1=dZ2@W2.T
    dZ1=dA1*derivative_relu(Z1)
    dW1=(X.T@dZ1)/m
    db1=np.sum(dZ1,axis=0,keepdims=True)/m

    return dW1,db1,dW2,db2

