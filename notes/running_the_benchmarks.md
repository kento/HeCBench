# HeCBenchの実行

## 概要

HeCBenchのMakefileを微修正し，ある程度自動的に実行できるような環境を整備することを行いたい。その目的のために作成したスクリプト群の使い方および生成されるファイル，想定する作業手順などについて説明する。

目指している動作
- 対象のディレクトリーやMakefileを作ってくれる
- NVIDIAのCUDAとOMP, AMDのROCM, OMP, HIPIFYのパラメーターはそれぞれ共通のファイルから読み込める
- 消費メモリ計測用およびtimeコマンドを介した実行用のターゲットが追加されたMakefileを生成してくれる
- ベンチマークを一つずつ実行することもバッチ実行することもできる

## スクリプト

### cuda2hip.sh

田上さんからいただいた `HIPIFY` を用いて`*.cu` ファイルをHIPでコンパイルできる形式に変換し，AMDのマシンでコンパイル・実行できるように変換してくれるスクリプト。Makefileの修正も行ってくれる。以下の要領で実行することができる。
``` sh
./cuda2hip.sh somebench-cuda somebench-hipified
```

`HIPIFY` の `bin` ディレクトリーを `PATH` に加えておく必要がある。

### edit_makefile.py

`Makefile` をもとに`time` コマンドを介して実行するターゲット `run_time` やベンチマーク実行時メモリ使用量を調べるターゲット `run_mem` が追加された `Makefile` を生成するスクリプト。以下の要領で実行する。
``` sh
./edit_makefile.py --target=TARGET --output=OUTPUT --input=INPUT
```
 `TARGET` はGPUの種類で `AMD` もしくは `NVIDIA` を指定する。デフォルト値は `NVIDIA`. `--output` には出力先のMakefileを指定する。デフォルト値は`--target` が `AMD` の場合 `Makefile.AMD`, `NVIDIA` の場合 `Makefile.NVIDIA`. `--input` にはコピー元の `Makefile` を指定する。デフォルト値は `Makefile`.

また，コマンド部分 `$(program) arg1 arg2 ...` は外部ファイルから `include` するようになっている。 `include` するファイルは一階層上(`src` ディレクトリー直下)に生成される。そのファイル名は `--target=AMD`の場合 `benchmarkname-AMD-cmd`, `--target=NVIDIA` の場合 `benchmarkname-NVIDIA-cmd` となる。

### gen_bench.sh

元からある `*-cuda` , `*-hip` などに加え，`*-omp` のAMD実行ディレクトリー `*-omp_aomp`, `*-omp` のNVIDIA実行ディレクトリー `*-omp_nvc` , `*-cuda` から `cuda2hip.sh` を用いてHIP化した `*-hipified` ディレクトリーを作成するスクリプト。以下の要領で実行する。

``` sh
 ./gen_bench.sh BENCHMARK_NAME
```

各種ディレクトリー作成後， `edit_makefile.py` によって `Makefile.AMD` , `Makefile.NVD` ファイルを作ることも行う。

### run_bench.sh

gen_bench.sh によって生成されたディレクトリーやMakefileを用いてベンチマークを実行するスクリプト。以下の要領で実行する。

``` sh
./run_bench.sh BENCHMARK_NAME GPU_TYPE
```
 `GPU_TYPE` が `AMD` の場合，以下の3通りのコマンドが実行される。標準出力と標準エラー出力は各ベンチマークディレクトリーの `log` ファイルに記録される。

 - `BENCHMARK_NAME-hip` の下で `make -f Makefile.AMD run_time`
 - `BENCHMARK_NAME-hipified` の下で `make -f Makefile.AMD run_time`
 - `BENCHMARK_NAME-omp_aomp` の下で `make -f Makefile.AMD run_time`

 `GPU_TYPE` が `NVIDIA` の場合，以下の2通りのコマンドが実行される。`AMD`の場合と同様標準出力・エラー出力は各ベンチマークディレクトリーの `log` ファイルに記録される。

 - `BENCHMARK_NAME-cuda` の下で `make -f Makefile.NVD run_time`
 - `BENCHMARK_NAME-omp_nvc` の下で `make -f Makefile.NVD run_time`

いずれのケースにおいても実行されるコマンドは各ベンチマークディレクトリーの一階層上の `BENCHMARK_NAME-GPU_TYPE-cmd` ファイルに記録されているそれなので，パラメーターの変更などが必要な場合適宜編集すればよい。

### exec_gen_bench.sh

ベンチマーク名のリストを読み込み，`gen_bench.sh` を逐次実行するスクリプト。以下の要領で実行する。

``` sh
./exec_gen_bench.sh BENCHMARK_LIST_FILE
```

ベンチマーク名リストファイルは，以下の要領で一行に一つ記述する。

```
accuracy
ace
adam
...
...
```

### exec_run_bench.sh

ベンチマーク名のリストを読み込み，`run_bench.sh` を逐次実行するスクリプト。以下の要領で実行する。

``` sh
./exec_run_bench.sh BENCHMARK_LIST_FILE GPU_TYPE
```

`exec_run_bench.sh` ではベンチマーク名リストファイルのほか，実行したい環境 (NVIDIAもしくはAMD) を第二引数に指定する。ベンチマーク名リストファイルの形式は `exec_gen_bench.sh` と同じ。

### m.sh

`nvidia-smi` コマンドのログからメモリ使用量を抽出するシェルスクリプト。 `Makefile.NVD` の `run_mem` ターゲットから呼ばれる。

### rocm-m.sh

`rocm-smi` コマンドのログからメモリ使用量を抽出するシェルスクリプト。 `Makefile.AMD` の `run_mem` ターゲットから呼ばれる。

### rocm-smi.sh

`rocm-smi` コマンドを周期的に実行するシェルスクリプト。 `Makefile.MAD` の `run_mem` ターゲットから呼ばれる。

## 作業の流れ

たとえば，以下のような流れで作業を実施する。

1. `gen_bench.sh` を実行し，ターゲットのベンチマークとGPUに応じたベンチマークのディレクトリーと `Makefile` を生成する。ベンチマークごとに個別に行うか， ベンチマーク名リストファイルを作成し， `exec_gen_bench.sh` を用いて一括で行う。
1. パラメーター調整が必要な場合， `BENCHMARK_NAME-GPU_TYPE-cmd` ファイルを編集し， `Makefile.AMD` もしくは `Makefile.NVD` の `run_mem` ターゲットなどを利用して調整を行う。
1. 調整が済んだら `run_bench.sh` を実行する。標準出力の結果は各ベンチマークディレクトリーの下の `log` ファイルに記録される。
1. すべて一括でベンチマークを実行したい場合 ベンチマーク名リストファイルを作成し，`exec_run_bench.sh` を実行する。
