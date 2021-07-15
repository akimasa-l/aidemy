import numpy as np
import matplotlib.pyplot as plt

# ごくシンプルな畳み込み層を定義しています。
# 1チャンネルの画像の畳み込みのみを想定しています。
# シンプルな例を考えるため、stridesやpaddingは考えません。
class Conv:
    def __init__(self, filters, kernel_size):
        self.filters = filters
        self.kernel_size = kernel_size
        self.W = np.random.rand(self.filters, kernel_size[0], kernel_size[1])
    def f_prop(self, X):
        k_h, k_w = self.kernel_size
        out = np.zeros((filters, X.shape[0]-k_h+1, X.shape[1]-k_w+1))
        for k in range(self.filters):
            for i in range(out[0].shape[0]):
                for j in range(out[0].shape[1]):
                    x = X[i:i+k_h, j:j+k_w]
                    out[k,i,j] = np.dot(self.W[k].flatten(), x.flatten())
        return out

X = np.load('./circle.npy')

# 畳み込み１
filters = 4
kernel_size = (3,3)

# 畳み込み層の生成
conv1 = Conv(filters=filters, kernel_size=kernel_size)

# 畳み込みの実行
C1 = conv1.f_prop(X)

# 畳み込み２
filters = 4
kernel_size = (6,6)

# 畳み込み層の生成
conv2 = Conv(filters=filters, kernel_size=kernel_size)

# 畳み込みの実行
C2 = conv2.f_prop(X)

# --------------------------------------------------------------
# 以下はすべて可視化のためのコードです。
# --------------------------------------------------------------

plt.imshow(X)
plt.title('base images', fontsize=12)
plt.show()

fig=plt.figure(figsize=(10,1))
for i in range(filters):
    ax=fig.add_subplot(1,filters,i+1)
    ax.axis("off")
    ax.imshow(conv1.W[i])
fig.suptitle('kernel visualization', fontsize=12)
plt.savefig('kernel-visualization.png',)
plt.show()
plt.close()

fig=plt.figure(figsize=(10,1))
for i in range(filters):
    ax=fig.add_subplot(1,filters,i+1)
    ax.axis("off")
    ax.imshow(C1[i])
fig.suptitle('convolution results 1', fontsize=12)
plt.savefig('convolution-results-1.png')
plt.show()
plt.close()

fig=plt.figure(figsize=(10,1))
for i in range(filters):
    ax=fig.add_subplot(1,filters,i+1)
    ax.axis("off")
    ax.imshow(conv2.W[i])
fig.suptitle('kernel visualization', fontsize=12)
plt.savefig('kernel-visualization.png',)
plt.show()
plt.close()

fig=plt.figure(figsize=(10,1))
for i in range(filters):
    ax=fig.add_subplot(1,filters,i+1)
    ax.axis("off")
    ax.imshow(C2[i])
fig.suptitle('convolution results 2', fontsize=12)
plt.savefig('convolution-results-2.png')
plt.show()