import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
import json


def main(batch_size: int):
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
    model.add(Dropout(rate=0.5))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    sgd = optimizers.SGD(lr=0.1)

    model.compile(optimizer=sgd, loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=3, verbose=1)

    score = model.evaluate(X_test, y_test, verbose=0)
    print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))
    return score


result = []
batch_sizes=[2**i for i in range(4,9)]
for batch_size in batch_sizes:
    loss, acc = main(batch_size)
    result.append({
        "batch_size": batch_size,
        "loss": loss,
        "acc": acc
    })

print(json.dumps(result, indent=4))
fig=  plt.figure()

def plot(ax):
    x,y1,y2=zip(*[i.values()for i in result])
    ax.plot(x,y1,label="loss")
    ax.plot(x,y2,label="acc")
    ax.legend()
    ax.grid()

ax1=fig.add_subplot(1,2,1)
plot(ax1)
ax2=fig.add_subplot(1,2,2)
plot(ax2)
ax2.set_xscale('log')

plt.savefig("./BkXPcnIoLxz.png")

plt.show()

