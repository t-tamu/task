# 【最終課題】: オリジナルのＡＩを構築する

# 『コーヒー豆をＯＫ品とＮＧ品に選別する』

# ライブラリのインポート

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics

# Keras
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# 画像データの読み込み
from PIL import Image

# データの分割
from sklearn.model_selection import train_test_split

# 画像データの機械学習用
from sklearn import datasets

# JupyterNotebook上でグラフを表示する設定
%matplotlib inline
# DataFrameで全ての列を表示する設定
pd.options.display.max_columns = None

# 訓練データ用CSVの読み込み
train_data = pd.read_csv("train/train_data.csv")
train_data.head()

# テストデータ用CSVの読み込み
test_data = pd.read_csv("test/test_data.csv")
test_data.head()

# コーヒー豆OKの写真をグレースケール化して表示
sample_img1 = Image.open("train/ok_01.jpg").convert("L")
plt.imshow(sample_img1, cmap="gray")

# コーヒー豆ngの写真をグレースケール化して表示
sample_img2 = Image.open("train/ng_01.jpg").convert("L")
plt.imshow(sample_img2, cmap="gray")

# グレースケール化した写真をndarrayに変換してサイズを確認
sample_img1_array = np.array(sample_img1)
sample_img1_array.shape

# 訓練用のコーヒー豆写真の読み込み(学習させるデータなので、「水増し」を行う)

# ndarrayのデータを保管する領域の確保
train_len = len(train_data)
# 左右、上下、180度回転させたものを用意するため、4倍の容量を確保する
X_train = np.empty((train_len * 4, 6400), dtype=np.uint8)       #X_trainは、train_lenの4倍の行数、それぞれの行に、画素数(80×80) の画像データが入る2次元配列とする。
y_train = np.empty(train_len * 4, dtype=np.uint8)               #y_trainは、train_lenの4倍の行数、それぞれの行に、コーヒー豆のng:1・ok:0の1つの数字が入る1次元配列とする。

# 画像ひとつひとつについて繰り返し処理
for i in range(len(train_data)):

    # 基の画像をndarrayとして読み込んで訓練データに追加
    name = train_data.loc[i, "File name"]                       #train_dataのi行目のFile nameを取得し、nameに格納する。
    train_img = Image.open(f"train/{name}.jpg").convert("L")    #nameに格納されているFile nameの画像データを、convert("L")で白黒に変換する。
    train_img = np.array(train_img)                             #numpyのndarrayに変換する。
    train_img_f = train_img.flatten()                           #train_imgは2次元配列になっているので、flatten()を施して、1次元配列に直す。
    X_train[i] = train_img_f                                    #作成したデータを計測データのi行目に格納する。
    y_train[i] = train_data.loc[i, "DC"]                        #train_dataのi行目のGC(コーヒー豆のng:1、ok:0)を教師データのi行目に格納する。   

    # 左右反転させたものを訓練データに追加
    train_img_lr = np.fliplr(train_img)                         #fliplrを施して、左(left)右(right)反転する。
    train_img_lr_f = train_img_lr.flatten()
    X_train[i + train_len] = train_img_lr_f
    y_train[i + train_len] = train_data.loc[i, "DC"]

    # 上下反転させたものを訓練データに追加
    train_img_ud = np.flipud(train_img_lr)                      #flipudを施して、上(up)下(down)反転する。
    train_img_ud_f = train_img_ud.flatten()
    X_train[i + train_len * 2] = train_img_ud_f
    y_train[i + train_len * 2] = train_data.loc[i, "DC"]

    # 180度回転させたものを訓練データに追加
    train_img_180 = np.rot90(train_img_lr, 2)                   #rot90を施して、180度(反時計回りに90度×"2"、"2"はrot90の2番目の引数)回転する。
    train_img_180_f = train_img_180.flatten()
    X_train[i + train_len * 3] = train_img_180_f
    y_train[i + train_len * 3] = train_data.loc[i, "DC"]
    
# テスト用のコーヒー豆写真の読み込み（「水増し」は不要）

# ndarrayのデータを保管する領域の確保
test_len = len(test_data)
X_test = np.empty((test_len, 6400), dtype=np.uint8)
y_test = np.empty(test_len, dtype=np.uint8)

# 画像ひとつひとつについて繰り返し処理
for i in range(test_len):

    # ndarrayとして読み込んで訓練データに追加
    name = test_data.loc[i, "File name"]
    test_img = Image.open(f"test/{name}.jpg").convert("L")
    test_img = np.array(test_img)
    test_img_f = test_img.flatten()
    X_test[i] = test_img_f
    y_test[i] = test_data.loc[i, "DC"]
    
# ここでは学習用データ（train)とテストデータ（test)を分ける必要はない（そのまま使えるため）

# Trainを学習データ（train)と検証データ（valid)に7:3に分ける
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

# 形状を確認
print("y_train=", y_train.shape, ", X_train=", X_train.shape)
print("y_valid=", y_valid.shape, ", X_valid=", X_valid.shape)
print("y_test=", y_test.shape, ", X_test=", X_test.shape)

# モデルの初期化
model = keras.Sequential()

# 入力層
model.add(Dense(32, activation='relu', input_shape=(6400,)))
# 隠れ層
model.add(Dense(32, activation='relu'))
# 出力層
model.add(Dense(1, activation='sigmoid'))

# モデルの構築
model.compile(optimizer = "rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

# モデルの構造を表示
model.summary()

%%time
# 学習の実施
log = model.fit(X_train, y_train, epochs=5000, batch_size=32, verbose=True,
                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                         min_delta=0, patience=100,
                                                         verbose=1)],
         validation_data=(X_valid, y_valid))

# グラフ表示
plt.plot(log.history['loss'], label='loss')
plt.plot(log.history['val_loss'], label='val_loss')
plt.legend(frameon=False) # 凡例の表示
plt.xlabel("epochs")
plt.ylabel("crossentropy")
plt.show()

# predictで予測を行う
y_pred = model.predict(X_test)

# 二値分類は予測結果の確率が0.5以下なら0,
# それより大きければ1となる計算で求める
y_pred_cls = (y_pred > 0.5).astype("int32")

y_pred

y_pred_cls

# 形状を正解（目的変数）に合わせる
y_pred_ = y_pred_cls.reshape(-1)

# モデルの評価
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred_))

