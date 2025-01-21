# autobench.pyスクリプトについて

HeCBenchに付属するベンチマーク実行用のスクリプトautobench.pyについて調べた。

## 概要

autobench.pyスクリプトは `src/scripts` ディレクトリーの下に配置されている。 何もつけずに、もしくは ` -h` をつけて実行するとヘルプメッセージが得られる。

```
$ autobench.py -h
usage: autohecbench.py [-h] [--log {INFO,DEBUG,WARNING,ERROR,CRITICAL}] [--output OUTPUT] [--summary SUMMARY] [--yes-prompt] [--repeat REPEAT] [--warmup WARMUP]
                       [--sycl-type {cuda,hip,opencl,cpu}] [--nvidia-sm NVIDIA_SM] [--amd-arch AMD_ARCH] [--compiler-name COMPILER_NAME] [--gcc-toolchain GCC_TOOLCHAIN]
                       [--extra-compile-flags EXTRA_COMPILE_FLAGS] [--overwrite] [--clean] [--verbose] [--bench-dir BENCH_DIR] [--bench-data BENCH_DATA] [--bench-fails BENCH_FAILS]
                       bench [bench ...]

HeCBench runner

positional arguments:
  bench                 Either specific benchmark name or sycl, cuda, or hip

options:
  -h, --help            show this help message and exit
  --log {INFO,DEBUG,WARNING,ERROR,CRITICAL}
                        logging level
  --output OUTPUT, -o OUTPUT
                        Output file for csv results
  --summary SUMMARY, -s SUMMARY
                        Output file for a summary of the execution of selected benchmarks
  --yes-prompt          If provided, automatically answer yes to the prompt.
  --repeat REPEAT, -r REPEAT
                        Repeat benchmark run
  --warmup WARMUP, -w WARMUP
                        Run a warmup iteration
  --sycl-type {cuda,hip,opencl,cpu}, -t {cuda,hip,opencl,cpu}
                        Type of SYCL device to use (default is cuda)
  --nvidia-sm NVIDIA_SM
                        NVIDIA SM version (default is 60)
  --amd-arch AMD_ARCH   AMD Architecture (default is gfx908)
  --compiler-name COMPILER_NAME
                        If a compiler is specified, use the specified one; otherwise, the default compiler in Makefiles will be used
  --gcc-toolchain GCC_TOOLCHAIN
                        GCC toolchain location (e.g. /path/to/gcc/x86_64/gcc-9.1.0)
  --extra-compile-flags EXTRA_COMPILE_FLAGS, -e EXTRA_COMPILE_FLAGS
                        Additional compilation flags (inserted before the predefined CFLAGS)
  --overwrite           If benchmark results are already available in the output file , then overwrite them. Default behavior is to skip existing results.
  --clean, -c           Clean the builds
  --verbose, -v         Verbose outputs from the builds
  --bench-dir BENCH_DIR, -b BENCH_DIR
                        Benchmark directory
  --bench-data BENCH_DATA, -d BENCH_DATA
                        Benchmark data
  --bench-fails BENCH_FAILS, -f BENCH_FAILS
                        List of failing benchmarks to ignore
```
実行する際には、第一引数に環境名を指定する。これはpositional argumentである。たとえば`cuda` などと指定する。第一引数以降はオプショナルなキーワード引数を指定する。たとえば`--output=result.csv` とすると`result.csv`というファイルに結果が出力される。もしくは`--summary=sum.log` などとすると`sum.log`というファイルにベンチマークのサマリーが報告される。

## subset.jsonファイル
どのようなベンチマークを実行するか、また引数として何を採用するかといった詳細は `src/scripts/benchmarks/subset.json` というJSONファイルに記述する。リポジトリにあるJSONファイルには以下のような設定が施されている。

```
{
    "accuracy": [
        "(?:Average execution time of accuracy kernel: )([0-9.+-e]+)(?: \\(us\\))",
        [
            "8192",
            "10000",
            "10",
            "100"
        ]
    ],
    "ace": [
        "(?:Offload time: )([0-9.+-e]+)(?: \\(ms\\))",
        [
            "100"
        ]
    ],
    ...
    ...
}
```

第一階層の辞書データのキーワードはベンチマーク名に対応する。これは各ベンチマークディレクトリーから末尾の ` -cuda` などを除いた文字列である。この辞書の値はリストである。リストの第一要素は正規表現を記述する。この正規表現は各ベンチマークのログから結果の数値を抽出するために用いられる。第二要素はベンチマークに与える引数のリストである。Positional argumentであれkeyword argumentであれ、そのままコマンドに引き渡す形式で記述すればよい。リポジトリの`subset.json`ファイルには132のベンチマークに対する設定が記述されている。

## 得られる結果について

各ベンチマークのログの書式はまちまちで、そもそも出力する結果の種類や単位がバラバラの傾向がある。そのような状況で正規表現を用いて機械的に数値を抽出したものを結果としているので、たとえば時間にしても単位がμ秒だったりミリ秒だったり秒だったりするし、結果そのものも全計算時間・GPUカーネルで過ごした時間・転送速度などベンチマークによって異なる結果が得られる。これも参考にはなるが、仕様にもある `time` コマンドによる計測時間も記録しておくことが必要なのではないかと考えている。

## その他備考

各ベンチマークは600秒でタイムアウトするようになっているよう。

