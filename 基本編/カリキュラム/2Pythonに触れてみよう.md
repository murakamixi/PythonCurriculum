## 概要

Pythonについて学び、画面に「Hello World」と出力するプログラムを作成します。また、GoogleColabについても学習します。

# この章の目標
**推定完了時間:30分**

**1. PythonのプログラムをGoogleColab上で実行できるようになる**

　プログラミング言語であるPythonについて学び、実際にプログラムを書いて実行できるまでの流れを掴みましょう。
　最初から完璧は目指さず、ますはコードが読めるようになることが大事です。基礎編中は、何も見ずにコードを書けるようになる必要は必ずしもありません。

# プログラミングとは？
　これから基本編の中で学習するプログラミングの**用語**について確認していきましょう。

### プログラミング
　処理手順を機械にわかるようにまとめたものを**プログラム**(program)といい、それを作成することを**プログラミング**(programming)と言います。
### プログラミング言語
　プログラミングを行うための言語のことをいいます。人間の行いたい処理をコンピューターに伝えるために作られた言語です。

## どんな言語なのPython？
　実際に、**Python**というプログラミング言語を使って、コードを書いていきましょう！
　「幾何学にも王道がなし」のように、プログラミングにも王道はありません。地道に手を動かして覚えていきましょう。
### Python
 Pythonはプログラミング言語の1つです。気軽にプログラミングできて、しかも実用的です。また、可読性も高く動作する擬似コードと言われるほどです。大きなWebアプリケーションから小さなプログラム、機械学習、データ分析まで、様々な分野で使われています。
[Python公式サイト](https://www.python.jp/index.html)

![](/media/editor/B-1pythonとは？_20201129070710996306.png)

## プログラミングはどうやって行われるのか？
　そもそも、プログラミングはどういう手順で行われるのでしょうか？

　1. プログラムを書く
　2. 書いたプログラムを実行するように、PCに対して命令をだす。

　プログラミングの基本は、１と２をひたすら繰り返しているだけなのです。
　ですが、プログラムは様々な仕事を行わせることができます。単なる文字の出力から人工知能、Instagramをはじめとする大規模なWebアプリケーションを構築することまでできます。

# Google Colaboratoryについて
　公式サイトから引用しますと、Colaboratory とは、
>Colaboratory（略称: Colab）は、Google Research が提供するサービスです。Colab では、誰でもブラウザ上で Python を記述、実行できるため、機械学習、データ分析、教育に特に適しています。具体的には、GPU などのコンピューティング リソースに無料でアクセスしながら特別な設定なしにご利用いただけるホスト型の Jupyter Notebook サービスです。

# Pythonを実際に書いてみよう
　早速、Google Colabを使いながらPythonプログラムを作成していきましょう。
　まずは、Google Colabのコンソールに文字を表示するという簡単なプログラムを書いてみることでGoogle Colabの使い方やPythonのプログラムの書き方に慣れましょう。

### 手順
 今から作るのは、コンソール「HelloWorld」と出力する最も簡単なプログラムを作成します。
　以下のようにHelloWorldと出力することが今回の目標です。

　早速下記の手順で作っていきましょう！

　**1. Google Colabを開く
　2. Pythonのプログラムを書く
　3. Google Colab上で実行する**

## Google Colabを開こう
　まずはじめに、下記のPythonプログラムを下記の画像の青い枠の中に記述しましょう。
[Google Colaboratory](https://colab.research.google.com/drive/1bz9ltpJj3jBzYn4eAlQg6WZKl67gSd6I?usp=sharing)を開いてみましょう。

```python
print('Hello World')
```

![基本1-1.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/235810/62dabe6c-c7cb-52c7-f61c-035e114af654.png)

　コンソール上にHelloWorldと表示するプログラムを作成できました。
　このプログラムを詳しくみていきましょう。
　<font color='red'>**print('Hello World')**</font>の**'Hello World'**は画面上に表示させる文字列です。文字列はPythonの**オブジェクト**の１つです。
　では、文字列とオブジェクトについて解説します。

### オブジェクト
　Pythonで扱えるデータは全て**オブジェクト**と呼ばれます。
　数字、文字列、時間などはPythonで扱う場合にはオブジェクトという形でデータ化されます。Pythonではよく使われるオブジェクトをあらかじめ用意されています。以下の表に代表的なものをまとめておきます。これ以外にもたくさんのオブジェクトがあります。

|オブジェクト名|扱うもの|型名|
|:--:|:--:|:--:|
|文字列オブジェクト|文字列|string|
|数値オブジェクト|数字|int|
|日時オブジェクト|日時|datetime|
|配列オブジェクト|複数データ|list|

### 文字列
　文字列型とは、プログラムのなかで文字を扱うためのオブジェクトのことを言います。
　Pythonでは、ダブルクォーテーション（"と"）またはシングルクォーテーション('と')で囲うと文字列になります。
　次のように、3種類の記述方法があります。

```python
#シングルクオテーション
str1 = 'Hello World'

#ダブルクオテーション
str2 = "Hello World"

#トリプル・ダブルクオテーション
str3 = """Hello World
　　　　　　Hello World"""
```

## 関数とは？
　<font color='red'>**print('Hello World')**</font>の**print()**はオブジェクトを表示するための**関数**です。

### 関数
 関数とは、何らかの処理を行う命令の集まりのことを言います。関数名をプログラムの中に記載することで、その関数を呼び出すことができ、その関数の処理を実行できます。
　関数の処理は様々です。関数は自作することもでき、行いたい処理を行うことなどが簡単にできます。
　例えば、ファイルを作ることや、ユーザーの入力を受けることができるものや、print()のようにたくさんの関数が用意されています。

### print()関数
 print()関数は、コンソール上に文字を出力する関数です。print()の()内に記載された値を、文字としてコンソールに出力されます。
　今回は、()内に'Hello World'という文字列を指定しているので画面上に「Hello World」と出力されます。

### print('Hello World')は？
　以上から、

```python
print('Hello World')
```

は、**「'Hello World'という文字列を、print(0関数でコンソール上に表示させる」**というプログラムです。
　このようにコードが理解できない場合は、一つづつ分解してみて、自分でそのコードを日本語で表現してみましょう。

## 先ほどのPythonプログラムを実行してよう
　先ほど、Google Colabで書いたプログラムをGoogle Colab上で実行してみましょう。
　あらためて、目標を再確認しましょう。目標は、**コンソール「HelloWorld」と出力する**ことでした。その目標が期待通り達成できてるか確かめましょう。

### Google Colabで実行しよう
　[Google Colaboratory](https://colab.research.google.com/drive/1bz9ltpJj3jBzYn4eAlQg6WZKl67gSd6I?usp=sharing)で先ほどのコードを確認します。
　以下の画像の②の実行ボタンを押してプログラムを実行してみましょう。
![基本1-1.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/235810/62dabe6c-c7cb-52c7-f61c-035e114af654.png)
　実行した結果、以下のように結果が出力されれば成功です。
![基本1-2.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/235810/c7dce51a-9093-401f-9e8e-d65cf14b8e35.png)

　以上で、Pythonを実際に書いてみようは終了です。
　今後もこのように、Pythonのコードを実行していくので、実行方法をしっかり復習しましょう。

# 見やすいコードのために
　プログラミングを行う上で、いくつか知っておくといい点があるので学んでおきましょう。

## コメントを書こう
### コメント
　プログラムの中には**コメント**を書くことができます。コメントはプログラムの中に書くメモのようなものです。
　コメントがプログラムの実行時に無視されるので、外部に公開してまずもの以外は、何を書いても問題ありません。コメントを書くことでそのコードやプログラムが何をしているのか明確になるために、プログラムが読みやすくなります。

### コメントアウト
　プログラムの処理をコメントにすることを**コメントアウト**といいます。
　pythonファイルでコメントアウトするにはコメントにしたい文字の先頭に#を記述します。
改行されるまではコメント扱いとなります。

```python
#ここからは改行するまでコメント。何を書いても動作に影響しません。

print("Hello World")  # 実行される

#print("Hello World")# 実行されない
```

　2つめに記述された**<font color='red'>print("Hello World")</font>**は、その行頭に**<font color='red'>#</font>**が記述されているのでコメントアウトされ、実行されません。

## よく起こるエラーを確認しよう

　エラーとは、プログラムに問題があり、正常には実行できない状態の事を言います。
　プログラミングにはエラーはつきものです。エラーを解消する過程は成長する大きなチャンスです！自力で解決できない場合は遠慮なくメンターに聞きましょう。
　初学者が起こしやすいエラーについてだけ簡単に説明させていただきます。以下のエラーには特に気おつけましょう。

1. 閉じタグ忘れ
2. スペルミス
3. インデント間違い