import numpy as np
import matplotlib.pyplot as plt
x=np.arange(-5,5,0.1)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_deriv(x):
    return 1/((1+np.exp(-x))**2)*np.exp(-x)
y1=sigmoid(x)
y2=sigmoid_deriv(x)
plt.plot(x,y1,label="sigmoid(x)")
plt.plot(x,y2,label="sigmoid_deriv(x)")
plt.legend()
plt.grid()
plt.savefig("./By2Ucn8jIlf.png")
plt.show()