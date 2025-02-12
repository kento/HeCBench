# 問題のあるベンチマーク

## ログファイルにおいてFAILとなるベンチマーク

| ベンチマーク名 | コメント |
| -- | -- |
| bscan | cuda版 hipified版はverify = FAIL となる。 |
| bwt | cuda版 FAIL パラメーターの問題かもしれない |
| extend2 | cuda版に限りError: がたくさんでる パラメーターの変更はしていない|
| minisweep | CUDA版のみverify : FAIL とでる。パラメーターの問題かもしれない|
| multinomial | CUDA版のみFAIL とでる。パラメーターの問題かもしれない|
| p2p | GPUを二枚使う必要があるベンチマークのよう。HIP, HIPIFIEDは問題なく動作，CUDA版は動作していない |
| pcc | Error in performing the multiplication |
| pso | cuda版のログの最後がFAIL. パラメーターの問題かもしれない |
| randomAccess | cuda版ログの最後がFAIL. パラメーターは変えていない |
| scan2 | cuda版ログの最後がFAIL. パラメーターの問題かもしれない |
| slit | cuda版ログの最後がFAIL. パラメーターは変えていない |
| split | cuda版ログの最後がFAIL. パラメーターの問題かもしれない |
| spnnz | cuda版ログの最後がFAIL. パラメーターの問題かもしれない |
| spsm | cuda版ログの最後がFAIL. パラメーターの問題かもしれない |
| spsort | cuda版ログの最後がFAIL. パラメーターの問題かもしれない |
| swish | cuda版ログの最後がFAIL. パラメーターの問題かもしれない |
| tensorAccessor | cuda版ログの最後がFAIL. パラメーターの問題かもしれない |
| warpexchange | cuda版ログの最後がFAIL. パラメーターの問題かもしれない |

## 現時点で(まともな)ログが存在しないベンチマーク

| ベンチマーク名 | コメント |
| -- | -- |
| allreduce | MPI CUDA版は実行できた |
| ccl | MPI CUDA版は実行できた |
| graphB+ | hip版は12h以上終わらない。|
| mf-sgd | 構成が特殊。特段難しくない可能性もある。対処できた |
| miniDGS | MPI |
|       | src/MaxwellsKernel3d.cu(6): error: texture is not a template |
|        | texture<float4, 1, cudaReadModeElementType> t_LIFT;
|        | コンパイル時このような感じのエラーが大量に発生する。cuda11以下が必要？ |
| miniFE | 構成が特殊。対処したがomp版は動作していない。 |
| miniWeather | MPI CUDA版は実行できた |
| p2p | CUDA版は Two or more GPUs with Peer-to-Peer access capability are required for ./main. というメッセージ. 2ノードのジョブを実行しても同様のエラー。HIP版 HIPIFIED版は問題なく動作した。 |
| pingpong | MPI CUDA版は実行できた |
| prna | DATAPATHの設定忘れ；対処済み |
| rowwiseMoments | cuda版がコンパイルできない；namespace "thrust" has no member "pair" --> thrust::pairをcuda::std::pairに書き換えるとコンパイルできるようになった。|
| saxpy-ompt | どの環境でもコンパイルに成功していない AMDのマシンではasaxpy.c:24:10: fatal error: 'hip/hip_runtime.h' file not found --> AMD では実行可能になった。NVIDIAでは実行時にエラー。 |
| si | cmakeを使うベンチマーク。cuda版は問題なく実行できた。HIP版はhipblasがないと言われコンパイルできなかった。https://github.com/ROCm/hipBLAS の情報をもとにコンパイルを試みたができていない。  |
| slu | cuda版がコンパイルできなかった (hip, hipifiedはコンパイルはできたが非常に低速であった) |
| sparkler | MPI CUDA版は実行できた |
| sss | ソース, Makefileを微修正。GSLが必要。-lblas->-lgslcblas などで実行できた。しかしhipified版は実行エラー。 |
| stsg | GDAL, SQlite3, projなどの外部ライブラリーが必要。インストールしたところ実行できた。 |
| tsne | 必要なデータセットファイルの入手方法が分からない |
| xlqc | gslにリンクし，LD_LIBRARY_PATHを通せば問題なく実行できた。omp_amd版はコンパイルできなかったが，-std=c++17 とするとコンパイルできた。しかしomp_aomp版は実行するとnanがでた。 |

## ログに何も出力されないベンチマーク

| ベンチマーク名 | コメント |
| -- | -- |
| grep | |

## データセットがないベンチマーク

| ベンチマーク名 | コメント |
| -- | -- |
| cmp | データセットをいただいたので試す；AMDでは実行はできたがステータスはfail. NVIDIAでは問題なく実行できた。 |
| diamond | Please contact me for the dataset. |

## ソフト/ライブラリーが必要なベンチマーク

| ベンチマーク名 | コメント |
| -- | -- |
| bm3d | ImageMagick, GraphicsMagickが必要。HIP版は実行できたがCUDA版は [CImg] \*\*\* CImgIOException \*\*\* [instance(0,0,0,0,(nil),non-shared)] CImg<unsigned char>::load(): Failed to recognize format of file 'noised-bufferfly-20.png'. というエラーで実行できなかった。 |
| convolutionDeformable | torchが必要 |
| dwconv1d | torchが必要 |
| stsg | SQliteなどが必要。対応できた。 |

## 構成が特殊なベンチマーク

| ベンチマーク名 | コメント |
| -- | -- |
| convolutionDeformable | pythonで駆動する。torchが必要。 |
| dwconv1d | pythonで駆動する。torchが必要 |
| mf-sgd | ~~Makefileが最上位ディレクトリーに存在しない。データセットもなく，READMEのリポジトリをダウンロードしてもよく分からない。~~ 対処できた |
| minFE | ~~srcディレクトリーでmakeすればよいのだろうか。~~ 対処できた |

## ログが標準エラー出力

| ベンチマーク名 | コメント |
| -- | -- |
| fpdc |  |
| hungarian |  |

## なぜかCUDA未着手

| ベンチマーク名 | コメント |
| -- | -- |
| d3q19-bgk | static const char dirs を static const signed char dirs に変えると実行できた。  |

## 正規表現を設定していないベンチマーク ##
| ベンチマーク名 | コメント |
| -- | -- |
| convolutionDeformable | 実行できていない (torch) |
| deq19-bgk | 標準出力には時間ではなく性能が出力される。 |
| diamond | 実行できていない (データセットがない)|
| dwconv1d | 実行できていない (torch) |
| grep | 標準出力に何もでてこない |
| miniDGS | 実行できていない (古いCUDAが必要?) |
| p2p | 標準出力には時間でなく性能が出力される |
| prna | 標準出力に時間など環境ごとに異なる情報は記録されない |
| saxpy-ompt | 標準出力には時間ではなく性能が出力される |
| si | 正規表現で標準出力からどうデータを抽出してよいか分からない |
| srad | 正規表現で標準出力からどうデータを抽出してよいか分からない ; Total time:という行の次の行に数値 |
| tsne | 実行できていない。実行に必要なファイルが入手できない。 |
| wmma | 標準出力には時間ではなく性能が出力される |

## MPIベンチマークについて

CUDA版の実行の仕方
- 2ノード確保する
- nvhpc-hpcx-cuda12 をmodule loadする。
- 必要に応じてMakefileを書き換える（パスが直に書いてある場合など）
- python ../scripts/genhostfile.py > hosts ; mpiexec --hostfile hosts -n 2 ./$(program) ... で実行できる。
- mi250ではmpiexec -n 2 --oversubscribe のようにover subscribeさせないとうまく動作しなかった。

## stsgベンチマークについて

GDALというソフトをインストールする必要がある。GDALはproj6というソフトとsqlite3というソフトに依存するためこれらもインストールする必要がある。

### sqlite3のインストール

https://www.sqlite.org/download.html からソースコードをダウンロード・展開し，トップディレクトリーにcdしたあと以下のコマンドを実行する。

```
./configure --prefix=$HOME/sqlite_gh
make
make install
```

### proj6のインストール

https://proj-tmp.readthedocs.io/en/6.2/download.html からアーカイブをダウンロード・展開し，トップディレクトリーにcdしたあと以下のようなコマンドを実行する。

```
export PKG_CONFIG_PATH=$HOME/sqlite_gh/lib/pkgconfig
export PATH=$HOME/sqlite_gh/bin:$PATH
を設定後
./configure --prefix=$HOME/proj6_gh
make
make install
```

### gdalのインストール

https://gdal.org/en/stable/download.html#source-code からソースコードのアーカイブをダウンロード・展開し，トップディレクトリーにcdしたあと以下のようなコマンドを実行する。

```
./configure --with-proj=$HOME/proj6_gh --with-sqlite3=$HOME/sqlite_gh --prefix=$HOME/gdal_gh
make
make install
```

### stsgソース/Makefile書き換え

main.cu
```
gdal/gdal_priv.hをgdal_priv.hに書き換え
```

Makefile.NVD
```
-I$(HOME)/gdal_gh/include 
LDFLAGS = -L$(HOME)/gdal_gh/lib -lgdal
```

コンパイル・実行時環境変数設定
```
export LD_LIBRARY_PATH=$HOME/sqlite_gh/lib:$HOME/gdal_gh/lib:$LD_LIBRARY_PATH
```

以上の修正を施すと
make -f Makefile.NVD run
で無事ベンチマークを実行することができた。
HIP版, HIPIFIED版も同様の手続きで実行できた。

# dwconv1d
Pythonから駆動するベンチマーク。minicondaなどを用いた方がいいかもしれない。
conda create -n torch
conda install anaconda::numpy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 (NOTE: Conda packages are no longer available. Please use pip instead.)

python run.py -> sm_90aがひっかかって異常終了。これをadhocに回避すると80近いエラーメッセージとともに終了。
以下の要領でextentionをコンパイルするコンパイラーを切り替える。
export CC=gcc
export CPP=g++
すると無事終了。

hip版は同様の手続きで簡単に実行できた。
hipified版はrun.pyの64行目の
'--use_fast_math', '--extra-device-vectorization'
の二つのオプションを削るとコンパイル・実行ができた。

