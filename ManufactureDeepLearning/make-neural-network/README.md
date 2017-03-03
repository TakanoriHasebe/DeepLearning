## プログラムの説明  

multi_layer_network.py : 最初はMNISTを学習。誤差逆伝播法を用い, 勾配を算出する。   

- 課題点
  - 重み, バイアスのパラメータと層の設計はクラス内の他の関数でも使うので, selfにすべき  
  - クラス内の初期化関数に重み, バイアス, Layerの追加を行っていなかった。  
  - バイアスの配列の形, また最初random.randnを用いたが, zerosで初期化すべきだった  
  - numpyにおいての内積とバイアスの足し算  
  - 初期化関数内での活性化関数の初期化の仕方  
  - 出力層のsoftmaxと誤差関数をself.lastLayer = ..()で初期化すべき  
  - 順伝搬時のpredict関数でのfor文の動き  
  - 辞書内にクラスのインスタンスを作成した時, 辞書のクラス内の関数の呼び出し方  
  - MultiLayerNetworkクラスでは勾配を求める  
  - ニューラルネットワークの全体の設計図がわかっていない  
  - 逆伝搬時の層を反対にするところ  
  - 最終的に勾配を返したいので, 勾配の辞書を作成する 
  - accuracy関数の書き方。loss関数を用いるのではなく, predict関数を用いる   
  - 参考url : https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/multi_layer_net.py   
  
---
make_trainer.py : multi_layer_network.pyを用いて勾配の算出を行い, ニューラルネットワークの訓練を行う   

- 課題点  
  - 初期化関数群  
  - バッチ処理について忘れている  
  - バッチ処理とミニバッチ処理について  
  - バッチ処理の書き方  
  - 勾配の最適化手法の初期化  
  - 勾配の更新でパラメータをどこから持ってくるかについて  
  - ミニバッチ学習の際の繰り返しの回数の設定について  
  - ミニバッチ学習とバッチ学習  
  - train_step関数とtrain関数  
  - 勾配(optimizer)の書き方  

---  
learning_mnsit.py : multi_layer_network.pyとmake_trainer.pyを用いて実際の学習を行うプログラム  

- 課題点 
  -  特になし

---
make_batch_normalization.py : Batch Normalizationを自分で実装したクラス  

- 課題点
  - 計算グラフの書き方, 計算の仕方
  - batch_normalizationは強制的にアクティベーションの分布に適度な広がりを持たせる
  - 分散が0になった時の為に, mini_batch_array.std()でなく, np.sqrt(var + 10-9)にしてある
  - 活性化関数に入力するミニバッチ全体の配列に対して, データの標準化をかけるということに注意
  - 参考url
  - https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py
  - https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

---
data_normalization.py : データの正規化について書いてある


