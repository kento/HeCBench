// spmm – OpenMP target port of spmm-kokkos
// Sparse-sparse matrix multiply: C(sparse) = A(sparse, CSR) * B(sparse, CSR)

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>

static void init_matrix(float *matrix, int num_rows, int num_cols, int nnz) {
  int n = num_rows * num_cols;
  float *d = (float*)malloc(n * sizeof(float));
  srand(123);
  for (int i = 0; i < n; i++) d[i] = (float)i;
  for (int i = n; i > 0; i--) {
    int a = i-1, b = rand() % i;
    if (a != b) { auto t = d[a]; d[a] = d[b]; d[b] = t; }
  }
  srand48(123);
  for (int i = 0; i < num_rows; i++)
    for (int j = 0; j < num_cols; j++)
      matrix[i*num_cols+j] = (d[i*num_cols+j] >= nnz) ? 0.0f : (float)(drand48()+1);
  free(d);
}

static void init_csr(int *row_offsets, float *values, int *col_indices,
                     float *matrix, int num_rows, int num_cols, int /*nnz*/) {
  row_offsets[0] = 0;
  int *cnts = (int*)calloc(num_rows, sizeof(int));
  int tmp = 0;
  for (int i = 0; i < num_rows; i++)
    for (int j = 0; j < num_cols; j++)
      if (matrix[i*num_cols+j] != 0.0f) {
        values[tmp] = matrix[i*num_cols+j];
        col_indices[tmp] = j;
        tmp++; cnts[i]++;
      }
  for (int i = 1; i <= num_rows; i++) row_offsets[i] = row_offsets[i-1] + cnts[i-1];
  free(cnts);
}

static int spmm_ref(float *A, float *B, float *&values, int *&offsets, int *&columns,
                    int A_num_cols, int C_num_rows, int C_num_cols) {
  int C_nnz = 0;
  float *C = (float*)malloc(C_num_rows * C_num_cols * sizeof(float));
  for (int i = 0; i < C_num_rows; i++)
    for (int j = 0; j < C_num_cols; j++) {
      double s = 0;
      for (int k = 0; k < A_num_cols; k++) s += A[i*A_num_cols+k] * B[k*C_num_cols+j];
      if (s != 0) C_nnz++;
      C[i*C_num_cols+j] = (float)s;
    }
  values  = (float*)malloc(C_nnz * sizeof(float));
  columns = (int*)malloc(C_nnz * sizeof(int));
  offsets = (int*)malloc((C_num_rows+1) * sizeof(int));
  init_csr(offsets, values, columns, C, C_num_rows, C_num_cols, C_nnz);
  free(C);
  return C_nnz;
}

// SpSpMM: compute via dense intermediate row buffer
void spmm_sparse(int *dA_off, int *dA_col, float *dA_val,
                 int *dB_off, int *dB_col, float *dB_val,
                 int *dC_off, int *dC_col, float *dC_val,
                 int m, int k, int n, int c_nnz)
{
  float *dense_row = (float*)calloc(m * n, sizeof(float));

  #pragma omp target enter data map(alloc: dense_row[0:m*n])

  // Zero dense_row
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < m * n; i++) dense_row[i] = 0.0f;

  // Compute SpSpMM row by row
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int row = 0; row < m; row++) {
    for (int idxA = dA_off[row]; idxA < dA_off[row+1]; idxA++) {
      int kk = dA_col[idxA];
      float va = dA_val[idxA];
      for (int idxB = dB_off[kk]; idxB < dB_off[kk+1]; idxB++) {
        int j = dB_col[idxB];
        dense_row[row * n + j] += va * dB_val[idxB];
      }
    }
  }

  // Count C nnz per row
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int row = 0; row < m; row++) {
    int cnt = 0;
    for (int j = 0; j < n; j++) if (dense_row[row*n+j] != 0.0f) cnt++;
    dC_off[row + 1] = cnt;
  }

  // Prefix sum on host
  #pragma omp target update from(dC_off[0:m+1])
  dC_off[0] = 0;
  for (int i = 0; i < m; i++) dC_off[i+1] += dC_off[i];
  #pragma omp target update to(dC_off[0:m+1])

  // Fill C
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int row = 0; row < m; row++) {
    int pos = dC_off[row];
    for (int j = 0; j < n; j++) {
      float v = dense_row[row*n+j];
      if (v != 0.0f) {
        dC_col[pos] = j;
        dC_val[pos] = v;
        pos++;
      }
    }
  }

  #pragma omp target exit data map(delete: dense_row[0:m*n])
  free(dense_row);
}

int main(int argc, char *argv[]) {
  if (argc != 8) {
    printf("Sparse matrix * sparse matrix = sparse matrix (SpSpMM)\n");
    printf("Usage %s <M> <K> <N> <A_nnz> <B_nnz> <repeat> <verify>\n", argv[0]);
    return 1;
  }
  int m=atoi(argv[1]), k=atoi(argv[2]), n=atoi(argv[3]);
  int a_nnz=atoi(argv[4]), b_nnz=atoi(argv[5]);
  int repeat=atoi(argv[6]), verify=atoi(argv[7]);

  float *hA=(float*)malloc(m*k*sizeof(float));
  float *hB=(float*)malloc(k*n*sizeof(float));
  float *hA_v=(float*)malloc(a_nnz*sizeof(float)); int *hA_c=(int*)malloc(a_nnz*sizeof(int)); int *hA_o=(int*)malloc((m+1)*sizeof(int));
  float *hB_v=(float*)malloc(b_nnz*sizeof(float)); int *hB_c=(int*)malloc(b_nnz*sizeof(int)); int *hB_o=(int*)malloc((k+1)*sizeof(int));

  init_matrix(hA, m, k, a_nnz); init_csr(hA_o, hA_v, hA_c, hA, m, k, a_nnz);
  init_matrix(hB, k, n, b_nnz); init_csr(hB_o, hB_v, hB_c, hB, k, n, b_nnz);

  float *ref_v=nullptr; int *ref_c=nullptr, *ref_o=nullptr;
  int c_nnz = spmm_ref(hA, hB, ref_v, ref_o, ref_c, k, m, n);

  int   *dC_o=(int*)  malloc((m+1)*sizeof(int));
  int   *dC_c=(int*)  malloc(c_nnz*sizeof(int));
  float *dC_v=(float*)malloc(c_nnz*sizeof(float));

  #pragma omp target enter data \
    map(to: hA_o[0:m+1], hA_c[0:a_nnz], hA_v[0:a_nnz]) \
    map(to: hB_o[0:k+1], hB_c[0:b_nnz], hB_v[0:b_nnz]) \
    map(alloc: dC_o[0:m+1], dC_c[0:c_nnz], dC_v[0:c_nnz])

  auto start = std::chrono::steady_clock::now();
  for (int i=0;i<repeat;i++)
    spmm_sparse(hA_o,hA_c,hA_v, hB_o,hB_c,hB_v, dC_o,dC_c,dC_v, m,k,n,c_nnz);
  auto end = std::chrono::steady_clock::now();
  printf("Average execution time of SPMM compute: %f (us)\n",
         (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()*1e-3f)/repeat);

  if (verify) {
    #pragma omp target update from(dC_o[0:m+1], dC_c[0:c_nnz], dC_v[0:c_nnz])
    int correct=1;
    if (dC_o[m] != c_nnz) { correct=0; printf("C nnz mismatch %d vs %d\n", dC_o[m], c_nnz); }
    if (correct) for (int i=0;i<c_nnz;i++) if (fabsf(dC_v[i]-ref_v[i])>1e-2f||dC_c[i]!=ref_c[i]) { correct=0; break; }
    printf("%s\n", correct ? "spmm_example test PASSED" : "spmm_example test FAILED: wrong result");
  }

  #pragma omp target exit data map(delete: hA_o[0:m+1], hA_c[0:a_nnz], hA_v[0:a_nnz], \
                                           hB_o[0:k+1], hB_c[0:b_nnz], hB_v[0:b_nnz], \
                                           dC_o[0:m+1], dC_c[0:c_nnz], dC_v[0:c_nnz])
  free(hA); free(hB); free(hA_v); free(hA_c); free(hA_o);
  free(hB_v); free(hB_c); free(hB_o); free(ref_v); free(ref_c); free(ref_o);
  free(dC_o); free(dC_c); free(dC_v);
  return 0;
}
