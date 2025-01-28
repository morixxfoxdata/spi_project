# spi_project

1. git clone する

```
git clone https://github.com/morixxfoxdata/spi_project.git
```

spi_project/data/experiment/collect：時系列信号データを格納 <br>
spi_project/data/experiment/target：画像データを格納 <br>

2. data リポジトリを作成し、信号データ、画像データを格納
3. 画像サイズ(pixel)、num_epochs や learning_rate を変更。モデルのインスタンスは forloop 内にある。
4. 実行前に save ファイル名の指定を忘れると上書きされるので注意。モデルインスタンス作成時に

```python
model = FCModel(
        input_size=2500, hidden_size=1024, output_size=784, name="DefaultFC_wave1"
    ).to(device)
```

name 引数がそのまま results/px28 の内部にディレクトリとして作成される。

5. train.py を実行すると学習が行われる。
6. results の pix28 には TARGET との比較、pix28_npz には再構成後の結果が格納される。
