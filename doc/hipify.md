# スクリプトの利用方法

## ベンチマーク実行に関わるスクリプト
### cuda2hip.sh

`HIPIFY` を用いて`*.cu` ファイルや `Makefile` などをHIPでコンパイルできる形式に変換し、AMDのマシンでコンパイル・実行できるうように変換してくれるスクリプト。以下の要領で実行する。

``` sh
./cuda2hip.sh somebench-cuda somebench-hipified
```

動作させるためには `HIPIFY` の `bin` ディレクトリーを `PATH` に加えておく必要がある。

### edit_makefile.py

`Makefile` をもとにベンチマーク実行時メモリ使用量を調べるターゲット `run_mem` が追加された `Makefile` を生成するスクリプト。さらにもとの`Makefile`に存在する`run` ターゲットは`time` コマンドを介して実行するような微修正が施される。以下の要領で実行する。

``` sh
./edit_makefile.py --target=TARGET --output=OUTPUT --input=INPUT
```

 `TARGET` はGPUの種類で `AMD` もしくは `NVIDIA` を指定する。デフォルト値は `NVIDIA`. `--output` には出力先のMakefileを指定する。デフォルト値は`--target` が `AMD` の場合 `Makefile.AMD`, `NVIDIA` の場合 `Makefile.NVD`. `--input` にはコピー元の `Makefile` を指定する。デフォルト値は `Makefile`. 

また，コマンド部分 `$(program) arg1 arg2 ...` は外部ファイルから `include` するようになっている。複数コマンドを実行する `run` ターゲットの場合各コマンドがセミコロンでつながって記述されるようになっている。 `include` するファイルはベンチマークディレクトリーからみて一階層上の `include` ディレクトリーに生成される。そのファイル名は `--target=AMD`の場合 `benchmarkname-AMD`, `--target=NVIDIA` の場合 `benchmarkname-NVIDIA` となる。このインクルードファイルは `src/include` 以下に保存される。

### gen_bench.sh

元からある `*-cuda` , `*-hip` などに加え，`*-omp` のAMD実行ディレクトリー `*-omp_aomp`, `*-omp` のNVIDIA実行ディレクトリー `*-omp_nvc` , `*-cuda` から `cuda2hip.sh` を用いてHIP化した `*-hipified` ディレクトリーを作成するスクリプト。以下の要領で実行する。

``` sh
 ./gen_bench.sh BENCHMARK_NAME
```
各種ディレクトリー作成後， `edit_makefile.py` によって `Makefile.AMD` , `Makefile.NVD` ファイルを作ることも行う。

### run_bench.sh

gen_bench.sh によって生成されたディレクトリーやMakefileを用いてベンチマークを実行するスクリプト。以下の要領で実行する。

``` sh
./run_bench.sh BENCHMARK_NAME GPU_TYPE [run_mem] [timeout]
```
 `GPU_TYPE` が `AMD` の場合，以下の3通りのコマンドが実行される。標準出力と標準エラー出力はそれぞれ各ベンチマークディレクトリーの `stdout.log` および `stderr.log` ファイルに記録される。

 - `BENCHMARK_NAME-hip` の下で `make -f Makefile.AMD run`
 - `BENCHMARK_NAME-hipified` の下で `make -f Makefile.AMD run`
 - `BENCHMARK_NAME-omp_aomp` の下で `make -f Makefile.AMD run`

 `GPU_TYPE` が `NVIDIA` の場合，以下の2通りのコマンドが実行される。`AMD`の場合と同様標準出力・エラー出力はそれぞれ各ベンチマークディレクトリーの `stdout.log` および `stderr.log` ファイルに記録される。

 - `BENCHMARK_NAME-cuda` の下で `make -f Makefile.NVD run`
 - `BENCHMARK_NAME-omp_nvc` の下で `make -f Makefile.NVD run`

いずれのケースにおいても実行されるコマンドは各ベンチマークディレクトリーの一階層上 `include` ディレクトリーの `BENCHMARK_NAME-GPU_TYPE` ファイルに記録されているそれなので，パラメーターの変更などが必要な場合適宜編集すればよい。

オプションとして，3番目の引数に `run_mem` と指定をするとメモリ消費を計測するモードでベンチマークを実行することができる。
オプションとして，4番目の引数に 整数値を指定をするとタイムアウトする時間を指定することができる。特に後述の`exec_run_bench.sh` から用いる場合、膨大な時間がかかるベンチマークに差し掛かると次へ進めなくなってしまうので、許容できる最大時間をあらかじめ設定しておくことが推奨される。

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
./exec_run_bench.sh BENCHMARK_LIST_FILE GPU_TYPE [run_mem] [timeout]
```

`exec_run_bench.sh` ではベンチマーク名リストファイルのほか，実行したい環境 (NVIDIAもしくはAMD) を第二引数に指定する。ベンチマーク名リストファイルの形式は `exec_gen_bench.sh` と同じ。 `run_bench.sh` と同様 `run_mem` をつけると消費メモリを計測するモードでベンチマークを実行することが、さらに整数値を指定するとタイムアウトする時間を指定することができる。

### m.sh

`nvidia-smi` コマンドのログからメモリ使用量を抽出するシェルスクリプト。 `Makefile.NVD` の `run_mem` ターゲットから呼ばれる。

### rocm-m.sh

`rocm-smi` コマンドのログからメモリ使用量を抽出するシェルスクリプト。 `Makefile.AMD` の `run_mem` ターゲットから呼ばれる。

### rocm-smi.sh

`rocm-smi` コマンドを周期的に実行するシェルスクリプト。 `Makefile.MAD` の `run_mem` ターゲットから呼ばれる。

## 結果集計に関わるスクリプト

### extract-time.sh
各ベンチマークディレクトリーから`time` コマンドの結果が記録されている `log.time` ファイルから経過時間情報を読み取り、Markdown形式の表を作成するスクリプト。`status`ディレクトリーに存在する。ベンチマークソースのルートディレクトリーにおいて以下の要領で実行する。
```
../status/extract-time.sh List_full > benchmark.md
```
`List_full` ファイルには対象とするベンチマークの名前が1行ずつ記録する。結果は標準出力に出力されるので、リダイレクトし所望のファイルに記録する。


### collect_bench_log.py
各ベンチマークディレクトリーのログファイルから経過時間などの情報を読み取り、結果をMarkdown形式の表に出力するスクリプト。`status`ディレクトリーに存在する。オプション `--help` をつけて実行すると簡単なヘルプメッセージが得られる。
```
./status/collect_bench_log.py --help
Usage: collect_bench_log.py [options]

collect data from benchmark log files

Options:
  -h, --help            show this help message and exit
  -b BENCH_NAMES, --bench_names=BENCH_NAMES
  -o OUTPUT, --output=OUTPUT
```

オプション `--bench_names` に対象とするベンチマークの名前が記録されたファイルを指定する。デフォルト値は `bench_names`. オプション `--output` に結果を出力するファイルを指定する。デフォルト値は `log_std.md`.

ベンチマークのログファイルからの結果の抽出は、`HeCBench`に元から備わっている`autobench.py` スクリプトの方針にならい正規表現を用いて行う。複数マッチする場合は結果は加算される。この際、結果がどのようなものか、たとえば経過時間ならその単位、もしくは経過時間ではなく転送速度など、ということは考慮しないため、異なるベンチマーク間の比較には適さない点に注意が必要。正規表現は`regexps.py`ファイルに、以下のような形式で記録されている。
``` python
regexps = {
  'accuracy':'(?:Average execution time of accuracy kernel: )([0-9.+-e]+)(?: \(us\))',
  'ace':'(?:Offload time: )([0-9.+-e]+)(?: \(ms\))',
  'adam':'(?:Average kernel execution time )([0-9.+-e]+)(?: \(ms\))',
  'addBiasResidualLayerNorm':'(?:Average execution time of AddBiasResidualLayerNorm \([0-9]+ x [0-9]+\): )([0-9.+-e]+)(?: \(us\))',
  'adv':'(?:elapsed time=)([0-9.+-e]+)(?: )',
  'aes':'(?:Average kernel execution time )([0-9.+-e]+)(?: \(s\))',
  'affine':'(?:Average kernel execution time )([0-9.+-e]+)(?: \(s\))',
  'aidw':'(?:Average execution time of AIDW_Kernel_Tiled )([0-9.+-e]+)(?: \(s\))',
  'aligned-types':'(?:Avg. time: )([0-9.+-e]+)(?: ms)',
  'all-pairs-distance':'(?:Average kernel execution time \(w/o? shared memory\): )([0-9.+-e]+)(?: \(us\))',
  'allreduce':'(?:Verified allreduce for size [0-9]+ \()([0-9.+-e]+)(?: us per iteration\))',
   ...
   ...
  'zoom':'(?:Average execution time of the zoom\-\w+ kernel: )([0-9.+-e]+)(?: \(us\))'
}
```

### genfig.py
`extract_time.sh` および `collect_bench_log.py` が出力する `Markdown` 形式の表にヒストグラムのプロットのカラムを追加するスクリプト。 `--help` をつけて実行すると簡単なヘルプメッセージが得られる。

```
../status/genfig.py --help
Usage: genfig.py [options]

generate bar graph from the markdown table

Options:
  -h, --help            show this help message and exit
  -m MARKDOWN, --markdown=MARKDOWN
  -e, --extend
  -n, --noplot
  -i IMAGEDIR, --imagedir=IMAGEDIR
  --mark_command
  --src_dir=SRC_DIR
```

オプション `--markdown` には対象としたい`Markdown`ファイルを指定する。デフォルト値は設定されていない。オプション`--extend` を指定するとカラムを追加する。オプション`--noplot` を指定するとプロットの生成は行わない。すでに利用したいプロットが存在する場合はこのオプションを使うと重複した画像生成を避けることができる。`--imagedir` にプロットの画像ファイルを保存するディレクトリーを指定する。画像ファイルは `svg` 形式で作成される。デフォルト値は`SVGs`. `--mark_command` を指定すると、`NVIDIA` と `AMD` で異なるオプションを採用している場合それぞれのコマンドをテーブル中に記すようになる。`--srcdir` にはソースコードディレクトリーを指定する。 `--mark_command`を指定している場合、このオプションに指定しているディレクトリーからコマンドを読み取る。デフォルト値は`../src2`.

# 実行できないできないベンチマーク

## Hipify不具合

### CUDA組み込み関数

Hipifyが対応していないCUDA組み込み関数があるためコンパイルできない場合がある。

- atomicAggregate-hipified
  - `__shfl_sync`
  - `__ballot_sync`
- attentionMultiHead-hipified
  - `__shfl_xor_sync`
- bitpermute
  -  `__syncwarp`
- bsw
  - `error: no matching function for call to 'warpReduceSum'`
- bitree
  -  `__shfl_sync`
  - `__ballot_sync`
- cc
  - `__ballot_sync`
- collision
  - `__any_sync`
  - `__ballot_sync`
  - `__shfl_xor_sync`
- dpid
  - `__shfl_down_sync`
- egs
  - `__all_sync`
  - `__ballot_sync`
  - `__any_sync`
- gc
  - ` __any_sync`
  - `__ballot_sync`
  - `__shfl_sync`
  - `__shfl_xor_sync`
  - `__shfl_xor`
- gemv
  - `__shfl_down_sync`
- graphB+
  - `__syncwarp;`
  - `__any_sync`
  - `__shfl_up_sync`
  - `__shfl_sync`
- jaccard
  - `__shfl_sync`
  - `__shfl_up_sync`
- logic-rewrite (hip版も同様の問題)
  - `__activemask`
  - `__shfl_xor_sync`
- logprob
  - `__shfl_xor_sync`
- lsqt
  - `__syncwrap`
- marchingCubes
  - `__shfl_down_sync`
  - `__shfl_up_sync`
- metropolis
  - `__shfl_down_sync`
- mf-sgd
  - `__shfl`
  - `__shfl_down`
- rsmt
  - `__syncwarp`
  - `__any_sync`
  - `__ballot_sync`
  - `__shfl_sync`
- segsort
  - `__shfl_xor_sync`
- shuffle
  - `__shfl_sync`
  - `__shfl_xor_sync`
- simpleSpmv
  - `__shfl_down_sync`
- vote
  - `__all_sync`
  - `__any_sync`
- warpsort
  - `__shfl_xor_sync`
  - `__shfl_up_sync`
  - `__shfl_down_sync`
  - `__shfl_sync`
  - `shfl`
  - `shfl_up`
  - `shfl_down`
- wedford
  - `__shfl_down_sync`
  
### 演算子オーバーロードの問題
- hexciton
  - `use of overloaded operator '/' is ambiguous (with operand types 'float2' (aka 'HIP_vector_type<float, 2>') and 'float2')`
  - `error: use of overloaded operator '*' is ambiguous (with operand types 'float2' (aka 'HIP_vector_type<float, 2>') and 'float2')`
- mdh
  - ` error: use of overloaded operator '/' is ambiguous (with operand types 'float2' (aka 'HIP_vector_type<float, 2>') and 'float2')`
  - `./helper_math.h:1477:14: error: use of overloaded operator '/' is ambiguous (with operand types 'float2' (aka 'HIP_vector_type<float, 2>') and 'float2')`
  - `./helper_math.h:1477:14: error: use of overloaded operator '*' is ambiguous (with operand types 'float2' (aka 'HIP_vector_type<float, 2>') and 'float2')`
  - `./helper_math.h:1481:30: error: use of overloaded operator '/' is ambiguous (with operand types 'float3' (aka 'HIP_vector_type<float, 3>') and 'float3')`
  - `./helper_math.h:1481:30: error: use of overloaded operator '*' is ambiguous (with operand types 'float3' (aka 'HIP_vector_type<float, 3>') and 'float3')`
  - `./helper_math.h:1481:30: error: use of overloaded operator '/' is ambiguous (with operand types 'float4' (aka 'HIP_vector_type<float, 4>') and 'float4')`
  - `./helper_math.h:1481:30: error: use of overloaded operator '*' is ambiguous (with operand types 'float4' (aka 'HIP_vector_type<float, 4>') and 'float4')`
- tpacf
  - `error: no viable overloaded '='`

### HIPの問題
 - cc
   - `error: no matching function for call to 'hipHostAlloc'`
 - gerbil
   - `/opt/rocm/include/hip/hip_runtime.h:66:2: error: #error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");`
   - `/opt/rocm/include/hip/hip_runtime_api.h:9246:61: error: 'hipMemAttachGlobal' was not declared in this scop`
 - jaccard
   - ` no matching function for call to 'parallel_prefix_sum'`
 -  logic-rewrite
   - `no matching function for call to 'hipGetErrorString'`
 - mallocFree
   - `error: no matching function for call to 'hipHostAlloc'`
 - overlap
   - `error: no matching function for call to 'hipHostAlloc'`
 - pad
   - `/opt/rocm-6.2.0/include/hip/amd_detail/host_defines.h:132:19: error: declaration of anonymous class must be a definition`
   - `/opt/rocm-6.2.0/include/hip/amd_detail/host_defines.h:132:35: error: non-type template parameter has incomplete type 'class F'`
   - `/opt/rocm-6.2.0/include/hip/amd_detail/host_defines.h:133:55: error: template argument for non-type template parameter must be an expression`
   - `/opt/rocm-6.2.0/include/hip/amd_detail/amd_hip_vector_types.h:145:23: error: expected a qualified name after 'typename'`
   - `/opt/rocm-6.2.0/include/hip/amd_detail/amd_hip_vector_types.h:145:23: error: expected ',' or '>' in template-parameter-list`
 - sc
   - `/opt/rocm-6.2.0/include/hip/amd_detail/host_defines.h:132:19: error: declaration of anonymous class must be a definition`
   - `/opt/rocm-6.2.0/include/hip/amd_detail/host_defines.h:132:35: error: non-type template parameter has incomplete type 'class F'`
   - `/opt/rocm-6.2.0/include/hip/amd_detail/host_defines.h:133:55: error: template argument for non-type template parameter must be an expression`
   - `/opt/rocm-6.2.0/include/hip/amd_detail/amd_hip_vector_types.h:145:23: error: expected a qualified name after 'typename'`
   - `/opt/rocm-6.2.0/include/hip/amd_detail/amd_hip_vector_types.h:145:23: error: expected ',' or '>' in template-parameter-list`
 - softmax-fused
   - `main.cu:9:10: fatal error: 'cuda_bf16.h' file not found`

### ライブラリーの問題
  - blas-gemmStridedBatched
    - `error: no matching function for call to 'hipblasHgemmStridedBatched'`
 - convolution3D
   - `fatal error: 'hipDNN.h' file not found`
 - hpl
   - `cuda_dgemm.cpp:103:10: fatal error: cblas.h: No such file or directory`
 - layernorm
   - `./common.h:6:10: fatal error: 'cooperative_groups/reduce.h' file not found`
 - softmax
   - `main.cu:8:10: fatal error: 'cooperative_groups/reduce.h' file not found`
 - softmax-online
   - `main.cu:12:10: fatal error: 'cooperative_groups/reduce.h' file not found`
 - spsm
   - `main.cu:56:10: fatal error: 'hipsparse.h' file not found`

###  undeclared identifier
 - blas-gemmEx
   - ``error: use of undeclared identifier '__float2bfloat16_rn'; did you mean '__float22bfloat162_rn'?``
 - blas-gemmEx2
   - `error: use of undeclared identifier '__float2bfloat16_rn'; did you mean '__float22bfloat162_rn'?``
   - `error: no viable conversion from 'float' to 'float2' (aka 'HIP_vector_type<float, 2>')`
 - intrinsics-simd
   - ` error: use of undeclared identifier '__vabs2'`
   - ` error: use of undeclared identifier '__vabs4'`
   - `error: use of undeclared identifier '__vabsdiffs2'` 
   - `error: use of undeclared identifier '__vabsdiffs4'` 
   - `error: use of undeclared identifier '__vabsss2'; did you mean '__fabsf32'?`
   - `error: reference to __host__ function '__fabsf32' in __global__ function`
 - ludb
   - ` error: use of undeclared identifier 'cublasGetStatusString'`
 - minimap2
   - ` error: use of undeclared identifier 'max'; did you mean 'fmax'?`
 - relu
   - ` use of undeclared identifier '__vmaxs4'`
 - rle
   - `fatal error: 'cub/device/device_run_length_encode.cuh' file not found`
 - wmma
   - `main.cu:42:17: error: expected namespace name`<br>`42 | using namespace nvcuda;`

## undefined reference
- face-omp_nvc
  - `undefined reference to __acc_compiled`
- fresnel-omp_nvc
  ```
   Undefined reference to '_Z32xFresnel_Auxiliary_Sine_Integrald' in 'fresnel.o'  
   nvlink error   : Undefined reference to '_Z34xFresnel_Auxiliary_Cosine_Integrald' in 'fresnel.o'
  ```
- fresnel-hipified `undefined hidden symbol: Fresnel_Sine_Integral(double)`

## unexpected namespace
 - wmma
   - `error: expected namespace name<br> 42 | using namespace nvcuda;`

## undefined value
- fpdc-omp_nvc
  -  `parse use of undefined value '@nvkernel__Z17CompressionKerneliiiPKyPcPKiPi_F1L92_2_F1L95_4' `

## 関数の引数の呼び方の問題
- bicsgstab-hip
```
/opt/rocm-6.2.0/include/hipsparse/hipsparse.h:13977:19: note: candidate function not viable: requires 9 arguments, but 10 were provided
 13977 | hipsparseStatus_t hipsparseSpSV_solve(hipsparseHandle_t           handle,
       |                   ^                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 13978 |                                       hipsparseOperation_t        opA,
       |                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 13979 |                                       const void*                 alpha,
       |                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 13980 |                                       hipsparseConstSpMatDescr_t  matA,
       |                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 13981 |                                       hipsparseConstDnVecDescr_t  x,
       |                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 13982 |                                       const hipsparseDnVecDescr_t y,
       |                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 13983 |                                       hipDataType                 computeType,
       |                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 13984 |                                       hipsparseSpSVAlg_t          alg,
       |                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 13985 |                                       hipsparseSpSVDescr_t        spsvDescr);
       |                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```
- cm-omp_aomp
```
error: no matching function for call to 'computePValue'
```
- cm-omp_nvc
```
 error: argument of type "int *" is incompatible with parameter of type "int"
```

## Pythonの問題

- convolutionDeformable-cuda
```
Traceback (most recent call last):
  File "/hs/work0/home/users/u0001621/git/HeCBench/src2/convolutionDeformable-cuda/main.py", line 8, in <module>
    from dcn_v2 import DCN
  File "/hs/work0/home/users/u0001621/git/HeCBench/src2/convolutionDeformable-cuda/dcn_v2.py", line 12, in <module>
    import _ext as _backend
ModuleNotFoundError: No module named '_ext'
```

## Makefileの問題

- diamond-omp_aomp
  - `clang++: error: invalid target ID ''; format is a processor name followed by an optional colon-delimited list of features followed by an enable/disable sign (e.g., 'gfx908:sramecc+:xnack-')`

## OpenMPの翻訳の問題
- fsm-omp_nvc 
  - `omp critical is not supported in GPU region (main.cpp: 91)`
- gc-omp_nvc
  - `Compiler failed to translate OpenMP region  (main.cpp: 162)`
- langford-omp_nvc
  - `NVC++-W-0155-Compiler failed to translate accelerator region (see -Minfo messages): Missing branch target block (main.cpp: 356)`
- lda-omp_nvc
  - `NVC++-W-0155-Compiler failed to translate accelerator region (see -Minfo messages): Missing branch target block (main.cpp: 59)`
- linearprobing-omp_nvc
  - `NVC++-W-0155-Compiler failed to translate accelerator region (see -Minfo messages): Missing branch target block (linearprobing.cpp: 28)`
- tsp-omp_nvc
  - `NVC++-W-0155-Compiler failed to translate accelerator region (see -Minfo messages): Missing branch target block (main.cpp: 201)`

## コンパイル, 実行順序の問題
- frna hipとcudaでオブジェクトファイルを共有する設定のため，cudaでコンパイルしとhipでコンパイルまたはその逆を行うとビルドに失敗する。実行前に :w
`make clean` するとよい。

## SSE対応の問題
- heat2d
  -  `fatal error: xmmintrin.h: No such file or directory`

## 必要なソースファイルが存在しない
- slit
  - `main.cu:18:10: fatal error: 'reference.h' file not found`

## 必要なコンパイラーが存在しない
- slu-cuda, slu-omp_nvc
  - `make[3]: clang: No such file or directory`

## 詳細不明
 - mis-hip
   - `make: *** [Makefile.AMD:63: run_mem] Error 1`
 - mrc-hip
   - `make: *** [Makefile.AMD:63: run_mem] Error 1`
 - permute-hip
   - `make: *** [Makefile.AMD:63: run_mem] Error 1`
 - perplexity-cuda
   - `make: *** [Makefile.NVD:65: run_mem] Error 1`
 - pingpong-hip, pingpong-hipified
   - `make: *** [Makefile.AMD:67: run_mem] Error 1`
 - pool-hip
   - `make: *** [Makefile.AMD:63: run_mem] Error 1`
 - relu-cuda
   - `make: *** [Makefile.NVD:65: run_mem] Error 1`
 - rle-hip
   - `make: *** [Makefile.AMD:62: run_mem] Error 1`
 - si-hipified
   - `clang++: error: linker command failed with exit code 1 (use -v to see invocation)`
 - simplemoc
   - `make: *** [Makefile.NVD:69: run_mem] Error 1`
 - spm-cuda, spm-hip
   - `make: *** [Makefile.NVD:65: run_mem] Error 1`
 - streamcluster-omp_nvc
   - `make: *** [Makefile.NVD:45: run] Error 1`
 - thomas-cuda, thomas-hipified
   - `make: *** [Makefile.NVD:66: run_mem] Error 1`

## データ入手困難
 - lzss
   - README.mdファイルの指示に従っても`ERROR 404: Not Found.` と出力され、必要なデータが入手できない
 - tsne 
   - ログに`Can't read points file` とだけ出力される。詳細不明。

## 非常に時間がかかるため、動作するかどうか未検証
 - crs-omp_nvc
   - 4時間以上かかる。CUDA版は1260s.