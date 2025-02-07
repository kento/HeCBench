ログファイルにおいてFAILとなるベンチマーク

| ベンチマーク名 | コメント |
| -- | -- |
| bscan | cuda版 hipified版はverify = FAIL となる。 |
| bwt | cuda版 FAIL パラメーターの問題かもしれない |
| extend2 | cuda版に限りError: がたくさんでる パラメーターの変更はしていない|
| minisweep | CUDA版のみverify : FAIL とでる。パラメーターの問題かもしれない|
| multinomial | CUDA版のみFAIL とでる。パラメーターの問題かもしれない|
| p2p | GPUを二枚使う必要があるベンチマークのよう |
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

現時点で(まともな)ログが存在しないベンチマーク

| ベンチマーク名 | コメント |
| -- | -- |
| allreduce | MPI |
| ccl | MPI |
| graphB+ | hip版は12h以上終わらない。|
| mf-sgd | 構成が特殊。特段難しくない可能性もある。対処できた |
| miniDGS | MPI |
| miniFE | 構成が特殊。対処したがomp版は動作していない。 |
| miniWeather | MPI |
| pingpong | MPI |
| prna | DATAPATHの設定忘れ；対処済み |
| rowwiseMoments | cuda版がコンパイルできない；namespace "thrust" has no member "pair" --> thrust::pairをcuda::std::pairに書き換えるとコンパイルできるようになった。|
| saxpy-ompt | どの環境でもコンパイルに成功していない AMDのマシンではasaxpy.c:24:10: fatal error: 'hip/hip_runtime.h' file not found --> AMD では実行可能になった。NVIDIAでは実行時にエラー。 |
| si | cmakeを使うベンチマーク。cuda版は問題なく実行できた。HIP版はhipblasがないと言われコンパイルできなかった。 |
| slu | cuda版がコンパイルできなかった (hip, hipifiedはコンパイルはできたが非常に低速であった) |
| sparkler | MPI |
| sss | ソース, Makefileを微修正。GSLが必要。-lblas->-lgslcblas などで実行できた。しかしhipified版は実行エラー。 |
| stsg | GDAL, SQlite3, projなどの外部ライブラリーが必要。|
| tsne | 必要なデータセットファイルの入手方法が分からない |
| xlqc | gslにリンクし，LD_LIBRARY_PATHを通せば問題なく実行できた。|

ログに何も出力されないベンチマーク

| ベンチマーク名 | コメント |
| -- | -- |
| grep | |

データセットがないベンチマーク

| ベンチマーク名 | コメント |
| -- | -- |
| cmp | データセットをいただいたので試す；AMDでは実行はできたがステータスはfail. NVIDIAでは問題なく実行できた。 |
| diamond | Please contact me for the dataset. |

ソフト/ライブラリーが必要なベンチマーク

| ベンチマーク名 | コメント |
| -- | -- |
| bm3d | ImageMagick, GraphicsMagickが必要。HIP版は実行できたがCUDA版は [CImg] \*\*\* CImgIOException \*\*\* [instance(0,0,0,0,(nil),non-shared)] CImg<unsigned char>::load(): Failed to recognize format of file 'noised-bufferfly-20.png'. というエラーで実行できなかった。 |
| convolutionDeformable | torchが必要 |
| dwconv1d | torchが必要 |
| stsg | SQliteなどが必要。|

構成が特殊なベンチマーク

| ベンチマーク名 | コメント |
| -- | -- |
| convolutionDeformable | pythonで駆動する。torchが必要。 |
| dwconv1d | pythonで駆動する。torchが必要 |
| mf-sgd | ~~Makefileが最上位ディレクトリーに存在しない。データセットもなく，READMEのリポジトリをダウンロードしてもよく分からない。~~ 対処できた |
| minFE | ~~srcディレクトリーでmakeすればよいのだろうか。~~ 対処できた |

ログが標準エラー出力
| ベンチマーク名 | コメント |
| -- | -- |
| fpdc |  |

なぜかCUDA未着手
| ベンチマーク名 | コメント |
| -- | -- |
| d3q19-bgk | static const char dirs を static const signed char dirs に変えると実行できた。  |

