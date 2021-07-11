from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# データを生成
X, y = make_regression(n_samples=100, n_features=100, n_informative=60, n_targets=1, random_state=42)
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=42)

# 以下にコードを記述してください
# 線形回帰
model = LinearRegression()
model.fit(train_X,train_y)

# test_X, test_yに対する決定係数を出力してください
print("Linear regression:{}".format(model.score(test_X, test_y)))

# ラッソ回帰
model = Lasso()
model.fit(train_X,train_y)

# test_X, test_yに対する決定係数を出力してください
print("Lasso regression:{}".format(model.score(test_X, test_y)))