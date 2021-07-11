# scikit-learnのLinearRegressionというモデルをインポートします。詳細は1.2で説明します
from sklearn.linear_model import LinearRegression

# scikit-learnに標準で搭載されている、ボストン市の住宅価格のデータセットをインポートします
from sklearn.datasets import load_boston

# scikit-learnに搭載されているデータセットを学習用と予測結果照合用に分けるツールをインポートします
from sklearn.model_selection import train_test_split


# データの読み込みです
data = load_boston()

# データを教師用とテスト用に分けます
train_X, test_X, train_y, test_y = train_test_split(
    data.data, data.target, random_state=42)

# 学習器の構築です
model = LinearRegression()

# 以下にコードを追加して、教師データを用いて学習器に学習させてください
model.fit(train_X, train_y)

# 以下にコードを追加して、テスト用データを用いて学習結果をpred_yに格納してください

pred_y = model.predict(test_X)
# 予測結果を出力します
print(pred_y)