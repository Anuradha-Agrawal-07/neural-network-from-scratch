import numpy as np
from model import forward_pass,compute_loss,backpropagation,W1,b1,W2,b2

X_train_np=np.random.randn(120,6)
y_train_np=(np.random.rand(120,1)>0.5).astype(int)

lr=0.01
epochs=300

for epoch in range(epochs):
    Z1,A1,Z2,A2=forward_pass(X_train_np)
    loss=compute_loss(y_train_np,A2)

    dW1,db1_update,dW2,db2_update=backpropagation(
        X_train_np,y_train_np,Z1,A1,Z2,A2
    )

    W1-=lr*dW1
    b1-=lr*db1_update
    W2-=lr*dW2
    b2-=lr*db2_update

    if epoch%100==0:
        print(f"Epoch {epoch}, Loss={loss}")

print("Training complete.")
