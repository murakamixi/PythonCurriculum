


**------------------------------**




**------------------------------**



<!-- ですが、これはあらかじめ成形されたデータであり、仮にデータが空の値をすなわち空白を含んでいたら、違う型のデータが入っていたらどうでしょうか？ -->
<!-- 　余力があれば以下の節を読んで実際のデータの処理の仕方を学習しましょう。
　実際のデータは空白のデータが入ってることは日常茶飯事です。ですので空白のデータの探し方やそのカラムのデータ型を見る方法を学習しましょう。 -->

<!-- # データの前処理の流れを確認しよう

機械学習プロジェクトにおいて７〜８割は前処理に時間が費やされるといわれています。
まず最初に、機械学習における文字列の扱い方について説明します。
機械学習において多くの場合は生のデータは役に立たず、意味のある数値に変換できて初めて様々な機械学習アルゴリズムに入力することができます。
そのためには、生データから取得した文字列や数値、日付データ等を、何らかの数値で表した「特徴量」にしないといけません。
### データの詳細情報を確認をする
### データに空白がないだろうか？ -->

**------------------------------**




**------------------------------**




**------------------------------**




**------------------------------**



　

**------------------------------**




########################################################

**------------------------------**







########################################################

**------------------------------**




**------------------------------**




**------------------------------**

<!-- そのほかの分析手法
# この章の目標
　そのほかの分析手法について表面だけですがさらっと紹介します。いずれも今の高校の範囲の知識では理解することは不可能でしょう。ですが、大学または社会ではたくさんのデータ解析の手法が皆さんに使われるのを待っています。
　この章を読んで興味が湧いたらどんどん先に進んで勉強していきましょう。全て前章までのコードを書き換える部分だけ記載しています。

ここにそのほかの分析方法

1. そのほかの分析手法について学習する

# 概要
  データサイエンティスト・AIエンジニアへの道標です。

# ニューラルネットワーク

```python
#正規化
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)
# モデル作成
from sklearn.neural_network import MLPRegressor
neural_network = MLPRegressor(hidden_layer_sizes=(50)) 
neural_network.fit(X_train,Y_train) 
```

# サポートベクター回帰

```python
# モデル作成
import statsmodels.api as sm
from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=1e3, gamma='scale')
svr.fit(X_train,Y_train) 
```

# ランダムフォーレスト

```python
import xgboost as xgb
random_forest = xgb.XGBRegressor(objective ='reg:squarederror')
random_forest.fit(X_train,Y_train)
```

# アンサンブル学習

```python
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
ensemble1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
ensemble2 = RandomForestRegressor(random_state=1, n_estimators=10)
ensemble3 = LinearRegression(normalize=True)
ereg = VotingRegressor(estimators=[('xgb', ensemble1), ('rf', ensemble2), ('lr', ensemble)])
ereg = ereg.fit(X_train, Y_train)
ereg.estimators
``` -->
**------------------------------**