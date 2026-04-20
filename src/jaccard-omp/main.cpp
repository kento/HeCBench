#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

typedef float vtype;

template <bool weighted, typename T>
void fill_weights(int e, T* weight_j, T* weight_i) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int j = 0; j < e; j++) {
        weight_j[j] = weighted ? (T)(j + 1) / e : (T)1.0;
    }
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int j = 0; j < e; j++) {
        weight_i[j] = (T)0.0;
    }
}

template <bool weighted, typename T>
void jaccard_row_sum(int n,
                     const int* csrPtr,
                     const int* csrInd,
                     const T*   weight_j,
                     T*         work) {
    if (weighted) {
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int row = 0; row < n; row++) {
            int start = csrPtr[row];
            int end   = csrPtr[row + 1];
            T sum = (T)0.0;
            for (int k = start; k < end; k++)
                sum += weight_j[csrInd[k]];
            work[row] = sum;
        }
    } else {
        #pragma omp target teams distribute parallel for thread_limit(256)
        for (int row = 0; row < n; row++) {
            work[row] = (T)(csrPtr[row + 1] - csrPtr[row]);
        }
    }
}

template <bool weighted, typename T>
void jaccard_is(int n,
                const int* csrPtr,
                const int* csrInd,
                const T*   weight_j,
                const T*   work,
                T*         weight_i,
                T*         weight_s) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int row = 0; row < n; row++) {
        for (int j = csrPtr[row]; j < csrPtr[row + 1]; j++) {
            int col = csrInd[j];
            int Ni  = csrPtr[row + 1] - csrPtr[row];
            int Nj  = csrPtr[col + 1] - csrPtr[col];
            int ref = (Ni < Nj) ? row : col;
            int cur = (Ni < Nj) ? col : row;

            weight_s[j] = work[row] + work[col];

            for (int i = csrPtr[ref]; i < csrPtr[ref + 1]; i++) {
                int ref_col = csrInd[i];
                T   ref_val = weighted ? weight_j[ref_col] : (T)1.0;

                int left  = csrPtr[cur];
                int right = csrPtr[cur + 1] - 1;
                while (left <= right) {
                    int middle  = (left + right) >> 1;
                    int cur_col = csrInd[middle];
                    if      (cur_col > ref_col) right = middle - 1;
                    else if (cur_col < ref_col) left  = middle + 1;
                    else {
                        weight_i[j] += ref_val;
                        break;
                    }
                }
            }
        }
    }
}

template <typename T>
void jaccard_jw(int e, T gamma,
                const T* csrVal,
                const T* weight_i,
                const T* weight_s,
                T*       weight_j) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int j = 0; j < e; j++) {
        T Wi = weight_i[j];
        T Ws = weight_s[j];
        weight_j[j] = gamma * csrVal[j] * Wi / (Ws - Wi);
    }
}

template <bool weighted, typename T>
void jaccard_weight(int iteration, int n, int e,
                    int* csr_ptr, int* csr_ind, T* csr_val) {
    const T gamma = (T)0.46;

    T*   d_csrVal  = (T*)malloc(e * sizeof(T));
    T*   d_weight_j = (T*)malloc(e * sizeof(T));
    T*   d_weight_i = (T*)malloc(e * sizeof(T));
    T*   d_weight_s = (T*)malloc(e * sizeof(T));
    T*   d_work     = (T*)malloc(n * sizeof(T));

    for (int i = 0; i < e; i++) d_csrVal[i] = csr_val[i];

    #pragma omp target enter data \
        map(to: csr_ptr[0:n+1], csr_ind[0:e], d_csrVal[0:e]) \
        map(alloc: d_weight_j[0:e], d_weight_i[0:e], d_weight_s[0:e], d_work[0:n])

    auto t_start = std::chrono::steady_clock::now();

    for (int it = 0; it < iteration; it++) {
        fill_weights<weighted, T>(e, d_weight_j, d_weight_i);
        jaccard_row_sum<weighted, T>(n, csr_ptr, csr_ind, d_weight_j, d_work);
        jaccard_is<weighted, T>(n, csr_ptr, csr_ind, d_weight_j, d_work, d_weight_i, d_weight_s);
        jaccard_jw<T>(e, gamma, d_csrVal, d_weight_i, d_weight_s, d_weight_j);
    }

    auto t_end = std::chrono::steady_clock::now();
    double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                    t_end - t_start).count();
    printf("Average execution time of kernels: %f (s)\n", (ns * 1e-9) / iteration);

    #pragma omp target exit data \
        map(delete: csr_ptr[0:n+1], csr_ind[0:e], d_csrVal[0:e], \
                    d_weight_j[0:e], d_weight_i[0:e], d_weight_s[0:e], d_work[0:n])

    free(d_csrVal); free(d_weight_j); free(d_weight_i);
    free(d_weight_s); free(d_work);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: ./main <numRow> <numCol> <iteration>\n");
        return 1;
    }

    int numRow    = atoi(argv[1]);
    int numCol    = atoi(argv[2]);
    int iteration = atoi(argv[3]);

    srand(2);

    printf("Number of matrix rows and cols: %d %d\n", numRow, numCol);

    std::vector<vtype> csr_val;
    std::vector<int>   csr_ptr = {0};
    std::vector<int>   csr_ind;
    int nnz = 0;

    for (int i = 0; i < numRow; i++) {
        for (int j = 0; j < numCol; j++) {
            vtype v = (vtype)(rand() % 10);
            if (v != (vtype)0) {
                csr_val.push_back(v);
                csr_ind.push_back(j);
                nnz++;
            }
        }
        csr_ptr.push_back(nnz);
    }

    jaccard_weight<true,  vtype>(iteration, numRow, nnz,
                                 csr_ptr.data(), csr_ind.data(), csr_val.data());
    jaccard_weight<false, vtype>(iteration, numRow, nnz,
                                 csr_ptr.data(), csr_ind.data(), csr_val.data());
    return 0;
}
