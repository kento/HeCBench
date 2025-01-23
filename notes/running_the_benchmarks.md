# HeCBenchの実行

## 概要

HeCBenchのMakefileを微修正し，ある程度自動的に実行できるような環境を整備することを行いたい。その目的のために作成したスクリプト群の使い方および生成されるファイル，想定する作業手順などについて説明する。

目指している動作
- 対象のディレクトリーやMakefileを作ってくれる
- NVIDIAのCUDAとOMP, AMDのROCM, OMP, HIPIFYのパラメーター(実行するコマンド)はそれぞれ共通のファイルから読み込める
- 消費メモリ計測用およびtimeコマンドを介した実行用のターゲットが追加されたMakefileを生成してくれる
- ベンチマークを一つずつ実行することもバッチ実行することもできる

## スクリプト

スクリプトはすべて `src/scripts` 以下に配置されている。

### cuda2hip.sh

田上さんからいただいた `HIPIFY` を用いて`*.cu` ファイルをHIPでコンパイルできる形式に変換し，AMDのマシンでコンパイル・実行できるように変換してくれるスクリプト。ver6相当。Makefileの修正も行ってくれる。以下の要領で実行することができる。

``` sh
./cuda2hip.sh somebench-cuda somebench-hipified
```

`HIPIFY` の `bin` ディレクトリーを `PATH` に加えておく必要がある。

### edit_makefile.py

`Makefile` をもとにベンチマーク実行時メモリ使用量を調べるターゲット `run_mem` が追加された `Makefile` を生成するスクリプト。以下の要領で実行する。

``` sh
./edit_makefile.py --target=TARGET --output=OUTPUT --input=INPUT
```

 `TARGET` はGPUの種類で `AMD` もしくは `NVIDIA` を指定する。デフォルト値は `NVIDIA`. `--output` には出力先のMakefileを指定する。デフォルト値は`--target` が `AMD` の場合 `Makefile.AMD`, `NVIDIA` の場合 `Makefile.NVD`. `--input` にはコピー元の `Makefile` を指定する。デフォルト値は `Makefile`. 

また，コマンド部分 `$(program) arg1 arg2 ...` は外部ファイルから `include` するようになっている。複数コマンドを実行する `run` ターゲットの場合各コマンドがセミコロンでつながって記述されるようになっている。 `include` するファイルはベンチマークディレクトリーからみて一階層上の `include` ディレクトリーに生成される。そのファイル名は `--target=AMD`の場合 `benchmarkname-AMD`, `--target=NVIDIA` の場合 `benchmarkname-NVIDIA` となる。

### gen_bench.sh

元からある `*-cuda` , `*-hip` などに加え，`*-omp` のAMD実行ディレクトリー `*-omp_aomp`, `*-omp` のNVIDIA実行ディレクトリー `*-omp_nvc` , `*-cuda` から `cuda2hip.sh` を用いてHIP化した `*-hipified` ディレクトリーを作成するスクリプト。以下の要領で実行する。

``` sh
 ./gen_bench.sh BENCHMARK_NAME
```

各種ディレクトリー作成後， `edit_makefile.py` によって `Makefile.AMD` , `Makefile.NVD` ファイルを作ることも行う。

### run_bench.sh

gen_bench.sh によって生成されたディレクトリーやMakefileを用いてベンチマークを実行するスクリプト。以下の要領で実行する。

``` sh
./run_bench.sh BENCHMARK_NAME GPU_TYPE [run_mem]
```
 `GPU_TYPE` が `AMD` の場合，以下の3通りのコマンドが実行される。標準出力と標準エラー出力はそれぞれ各ベンチマークディレクトリーの `stdout.log` および `stderr.log` ファイルに記録される。

 - `BENCHMARK_NAME-hip` の下で `make -f Makefile.AMD run`
 - `BENCHMARK_NAME-hipified` の下で `make -f Makefile.AMD run`
 - `BENCHMARK_NAME-omp_aomp` の下で `make -f Makefile.AMD run`

 `GPU_TYPE` が `NVIDIA` の場合，以下の2通りのコマンドが実行される。`AMD`の場合と同様標準出力・エラー出力はそれぞれ各ベンチマークディレクトリーの `stdout.log` および `stderr.log` ファイルに記録される。

 - `BENCHMARK_NAME-cuda` の下で `make -f Makefile.NVD run`
 - `BENCHMARK_NAME-omp_nvc` の下で `make -f Makefile.NVD run`

いずれのケースにおいても実行されるコマンドは各ベンチマークディレクトリーの一階層上 `include` ディレクトリーの `BENCHMARK_NAME-GPU_TYPE` ファイルに記録されているそれなので，パラメーターの変更などが必要な場合適宜編集すればよい。

オプションとして，最後に `run_mem` と指定をするとメモリ消費を計測するモードでベンチマークを実行することができる。

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
./exec_run_bench.sh BENCHMARK_LIST_FILE GPU_TYPE [run_mem]
```

`exec_run_bench.sh` ではベンチマーク名リストファイルのほか，実行したい環境 (NVIDIAもしくはAMD) を第二引数に指定する。ベンチマーク名リストファイルの形式は `exec_gen_bench.sh` と同じ。 `run_bench.sh` と同様最後に `run_mem` をつけると消費メモリを計測するモードでベンチマークを実行することができる。

### m.sh

`nvidia-smi` コマンドのログからメモリ使用量を抽出するシェルスクリプト。 `Makefile.NVD` の `run_mem` ターゲットから呼ばれる。

### rocm-m.sh

`rocm-smi` コマンドのログからメモリ使用量を抽出するシェルスクリプト。 `Makefile.AMD` の `run_mem` ターゲットから呼ばれる。

### rocm-smi.sh

`rocm-smi` コマンドを周期的に実行するシェルスクリプト。 `Makefile.MAD` の `run_mem` ターゲットから呼ばれる。

## 具体例

 `accuracy` ベンチマークを例に，具体的に生成される `Makefile` などについて説明する。

`accuracy` ベンチマークは `*-cuda`, `*-hip`, `*-omp/Makefile.aomp`, `*-omp/Makefile.nvc` すべてがそろっている。したがって，5通りの実行ディレクトリーと `Makefile` が必要となる。また `accuracy-cuda` の `Makefile` の `run` ターゲットは以下のような内容になっている。

```
run: $(program)
        $(LAUNCHER) ./$(program) 8192 10000 10 100
```

 `gen_bench.sh` を `src` ディレクトリーにおいて以下のように `accuracy` に対して実行する。

```
./scripts/gen_bench.sh accuracy
```

この処理の結果，以下のファイルやディレクトリーが得られる。

```
include/accuracy-NVIDIA include/accuracy-AMD accuracy-cuda/Makefile.NVD accuracy-hip/Makefile.AMD accuracy-hipified/Makefile.AMD accuracy-omp_amd/Makefile.AMD accuracy-omp_nvc/Makefile.NVD
```

`include/accuracy-NVIDIA` と `include/accuracy-AMD` の内容は同じであるが，前者はNVIDIA用の `Makefile.NVD` ファイル，後者はAMD用の `Makefile.AMD` ファイルから読み込まれる点に違いがある。すなわち，環境に応じてパラメーターを個別に設定できるようになっている。 `accuracy` ベンチマークの場合，その内容は以下のようになっている。

```
COMMAND = time -p ./$(program) 8192 10000 10 100
```

あたらしく生成された `Makefile` を用いて，たとえば以下のようにベンチマークを実行することができる。

```
make -f Makefile.NVD run_mem
nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop-ms=1 > log.mem &
bash -c "time -p ./main 8192 10000 10 100"
...
...
Grid size is 8192
Average execution time of accuracy kernel: 153.000580 (us)
PASS

real 2.77
user 0.38
sys 0.58
pkill nvidia-smi
../m.sh
min max diff memory in [MB] : 2 874 872
```

 `include/accuracy-NVIDIA` もしくは `include/accuracy-AMD` ファイルを適宜編集し，消費メモリを調節する。

パラメーターの調整が不要もしくはが調整済みの場合， `run_bench.sh` で環境ごとのベンチマークを実行することができる。ベンチマークの標準出力と標準エラー出力はそれぞれ各ベンチマークディレクトリーの下の `stdout.log` および `stderr.log` というファイルに記録される。

```
./run_bench.sh accuracy NVIDIA
running accuracy on NVIDIA
command
COMMAND =  ./$(program) 8192 10000 10 100

running benchmark under accuracy-cuda
running benchmark under accuracy-omp_nvc
```

いずれも `exec_gen_bench.sh`, `exec_run_bench.sh` を用いることによって複数のベンチマークを一度に処理することもできる。

## 作業の流れ

たとえば，以下のような流れで作業を実施する。

1. ~~~ `gen_bench.sh` を実行し，ターゲットのベンチマークとGPUに応じたベンチマークのディレクトリーと `Makefile` を生成する。ベンチマークごとに個別に行うか， ベンチマーク名リストファイルを作成し， `exec_gen_bench.sh` を用いて一括で行う。~~~ 実施ずみ； `src2` ディレクトリーに全ベンチマークに対してこの処理を行った結果が格納されている。
1. パラメーター調整が必要な場合， `include/BENCHMARK_NAME-GPU_TYPE` ファイルを編集し， `Makefile.AMD` もしくは `Makefile.NVD` の `run_mem` ターゲットなどを利用して調整を行う。
1. 調整が済んだら `run_bench.sh` を実行する。標準出力および標準エラー出力の結果はそれぞれ各ベンチマークディレクトリーの下の `stdout.log` および `stderr.log` ファイルに記録される。
1. すべて一括でベンチマークを実行したい場合ベンチマーク名リストファイルを作成し，`exec_run_bench.sh` を実行する。

