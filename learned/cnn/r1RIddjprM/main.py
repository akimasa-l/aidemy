import numpy as np
import matplotlib.pyplot as plt

# ごくシンプルな畳み込み層を定義しています。
class Conv:
    def __init__(self, W, filters, kernel_size):
        self.filters = filters
        self.kernel_size = kernel_size
        self.W = W # np.random.rand(filters, kernel_size[0], kernel_size[1])
    def f_prop(self, X):
        k_h, k_w = self.kernel_size
        out = np.zeros((filters, X.shape[0]-k_h+1, X.shape[1]-k_w+1))
        for k in range(self.filters):
            for i in range(out[0].shape[0]):
                for j in range(out[0].shape[1]):
                    x = X[i:i+k_h, j:j+k_w]
                    out[k,i,j] = np.dot(self.W[k].flatten(), x.flatten())
        return out

# ごくシンプルなプーリング層を定義しています。
# 1チャンネルの特徴マップのプーリングのみを想定しています。
class Pool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
    def f_prop(self, X):
        k_h, k_w = self.pool_size
        out = np.zeros((X.shape[0]-k_h+1, X.shape[1]-k_w+1))
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i,j] = np.max(X[i:i+k_h, j:j+k_w])
        return out

X = np.load('./circle.npy')

W = np.load('./weight.npy')

# 畳み込み
filters = 4
kernel_size = (3,3)
conv = Conv(W=W, filters=filters, kernel_size=kernel_size)
C = conv.f_prop(X)

# プーリング１
pool_size = (2,2)
pool1 = Pool(pool_size)
P1 = [pool1.f_prop(C[i]) for i in range(len(C))]

# プーリング２
pool_size = (4,4)
pool2 = Pool(pool_size)
P2 = [pool2.f_prop(C[i]) for i in range(len(C))]

# 以下はすべて可視化のためのコードです。

plt.imshow(X)
plt.title('base image', fontsize=12)
plt.savefig("base-image.png")
plt.show()

plt.figure(figsize=(10,1))
for i in range(filters):
    plt.subplot(1,filters,i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False) # 軸の削除
    plt.imshow(C[i])
plt.suptitle('convolution results', fontsize=12)
plt.savefig("convolution-results.png")
plt.show()

plt.figure(figsize=(10,1))
for i in range(filters):
    plt.subplot(1,filters,i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False) # 軸の削除
    plt.imshow(P1[i])
plt.suptitle('pooling results', fontsize=12)
plt.savefig("pooling-results-1.png")
plt.show()

plt.figure(figsize=(10,1))
for i in range(filters):
    plt.subplot(1,filters,i+1)
    ax = plt.gca() # get current axis
    ax.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False) # 軸の削除
    plt.imshow(P2[i])
plt.suptitle('pooling results', fontsize=12)
plt.savefig("pooling-results-2.png")
plt.show()