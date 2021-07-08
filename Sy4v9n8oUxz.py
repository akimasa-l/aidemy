import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical


def main(epochs, ax):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 784)[:1000]
    X_test = X_test.reshape(X_test.shape[0], 784)[:6000]
    y_train = to_categorical(y_train)[:1000]
    y_test = to_categorical(y_test)[:6000]

    model = Sequential()
    model.add(Dense(256, input_dim=784))
    model.add(Activation("sigmoid"))
    model.add(Dense(128))
    model.add(Activation("sigmoid"))
    # 今回はDropoutを使いません。
    # model.add(Dropout(rate=0.5))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    sgd = optimizers.SGD(lr=0.1)

    model.compile(optimizer=sgd, loss="categorical_crossentropy",
                  metrics=["accuracy"])

    history = model.fit(X_train, y_train, batch_size=32,
                        epochs=epochs, verbose=1, validation_data=(X_test, y_test))

    #acc, val_accのプロット
    ax.plot(history.history["accuracy"], label="acc", ls="-", marker="o")
    ax.plot(history.history["val_accuracy"],
            label="val_acc", ls="-", marker="x")
    ax.set_ylabel("accuracy")
    ax.set_xlabel("epoch")
    ax.legend(loc="best")
    ax.set_title(f"epoch is {epochs}")

    score = model.evaluate(X_test, y_test, verbose=0)
    print("evaluate loss: {0[0]}\nevaluate acc: {0[1]}".format(score))
    return score


result = []
fig = plt.figure(figsize=(15.0, 6.0))
for num, epochs in zip(range(1, 4), [5, 10, 50]):
    ax = fig.add_subplot(1, 3, num)
    loss, acc = main(epochs, ax)
    result.append({
        "epoch": epochs,
        "loss": loss,
        "acc": acc
    })
fig.tight_layout()
plt.savefig("./Sy4v9n8oUxz.png")
print(json.dumps(result, indent=4))
with open("./Sy4v9n8oUxz.json", mode='w') as f:
    json.dump(result, f, indent=4)
plt.show()
