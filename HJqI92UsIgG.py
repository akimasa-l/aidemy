import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
def main(dropoutrate=0.5):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train = X_train.reshape(X_train.shape[0], 784)[:6000]
    X_test = X_test.reshape(X_test.shape[0], 784)[:1000]
    y_train = to_categorical(y_train)[:6000]
    y_test = to_categorical(y_test)[:1000]
    
    model = Sequential()
    model.add(Dense(256, input_dim=784))
    model.add(Activation("sigmoid"))
    model.add(Dense(128))
    model.add(Activation("sigmoid"))
    
    #- --------------------------
    # ここを書いて下さい
    model.add(Dropout(rate=dropoutrate))
    # ここまで書いて下さい
    # ---------------------------
    model.add(Dense(10))
    model.add(Activation("softmax"))
    
    sgd = optimizers.SGD(lr=0.1)
    
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    
    history = model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=1, validation_data=(X_test, y_test))
    return history
n=12
ans=set()
N=n
for i in range(1,int(N**0.5+3)):
    if N%i==0:
        ans.add(tuple(sorted([i,N//i])))
row,col=sorted(ans)[-1]
fig = plt.figure()
for a,b,i in zip([row]*N,[col]*N,range(N)):
    ax = fig.add_subplot(a, b, i+1)
    history=main(i/n)
    ax.set_title(f"Dropout Rate is {i/n}")
    ax.plot(history.history["accuracy"], label="acc", ls="-", marker="o")
    ax.plot(history.history["val_accuracy"], label="val_acc", ls="-", marker="x")
    ax.set_ylabel("accuracy")
    ax.set_xlabel("epoch")
    ax.legend(loc="best")
#acc, val_accのプロット
""" 
plt.plot(history.history["accuracy"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_accuracy"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best") """
fig.tight_layout()
mng = plt.get_current_fig_manager()
#mng.frame.Maximize(True)
mng.resize(*mng.window.maxsize())
plt.savefig("./HJqI92UsIgG.png")
plt.show()