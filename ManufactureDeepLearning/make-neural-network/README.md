## プログラムの説明  
---
multi_layer_network.py : 最初はMNISTを学習。誤差逆伝播法を用い, 勾配を算出する。   

* 課題点について
0. 重み, バイアスのパラメータと層の設計はクラス内の他の関数でも使うので, selfにすべき  
1. クラス内の初期化関数に重み, バイアス, Layerの追加を行っていなかった。  
2. バイアスの配列の形, また最初random.randnを用いたが, zerosで初期化すべきだった  
3. numpyにおいての内積とバイアスの足し算  
4. 初期化関数内での活性化関数の初期化の仕方  
5. 出力層のsoftmaxと誤差関数をself.lastLayer = ..()で初期化すべき  
6. 順伝搬時のpredict関数でのfor文の動き  
7. 辞書内にクラスのインスタンスを作成した時, 辞書のクラス内の関数の呼び出し方  
8. MultiLayerNetworkクラスでは勾配を求める  
9. ニューラルネットワークの全体の設計図がわかっていない  
10. 逆伝搬時の層を反対にするところ  
11. 最終的に勾配を返したいので, 勾配の辞書を作成する 
12. accuracy関数の書き方。loss関数を用いるのではなく, predict関数を用いる   
参考url : https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/multi_layer_net.py   
  
---
make_trainer.py : multi_layer_network.pyを用いて勾配の算出を行い,     

* 課題点について  
0. 初期化関数群  
1. バッチ処理について忘れている  
2. バッチ処理とミニバッチ処理について  
3. バッチ処理の書き方  
4. 勾配の最適化手法の初期化  
5. 勾配の更新でパラメータをどこから持ってくるかについて  
6. ミニバッチ学習の際の繰り返しの回数の設定について  
7. ミニバッチ学習とバッチ学習  
8. train_step関数とtrain関数  

---  
learning_mnsit.py : multi_layer_network.pyとmake_trainer.pyを用いて実際の学習を行うプログラム  

* 課題点について  
