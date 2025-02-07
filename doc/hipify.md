# Hipify不具合情報

## CUDA組み込み関数

Hipifyが対応していない、CUDA組み込み関数がある。

- atomicAggregate-hipified
  - __shfl_sync
  - __ballot_sync
- attentionMultiHead-hipified
  - __shfl_xor_sync

