#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define BLOCK_SIZE    16
#define STR_SIZE      256
#define MAX_PD        (3.0e6)
#define PRECISION     0.001
#define SPEC_HEAT_SI  1.75e6
#define K_SI          100
#define FACTOR_CHIP   0.5

static const float t_chip      = 0.0005f;
static const float chip_height = 0.016f;
static const float chip_width  = 0.016f;
static const float amb_temp    = 80.0f;

static long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000LL) + tv.tv_usec;
}

static void writeoutput(const float* vect, int grid_rows, int grid_cols, const char* file) {
    FILE* fp = fopen(file, "w");
    if (!fp) { printf("Unable to open file %s\n", file); return; }
    char str[STR_SIZE];
    int index = 0;
    for (int i = 0; i < grid_rows; i++)
        for (int j = 0; j < grid_cols; j++) {
            sprintf(str, "%d\t%g\n", index, vect[i * grid_cols + j]);
            fputs(str, fp);
            index++;
        }
    fclose(fp);
}

static void readinput(float* vect, int grid_rows, int grid_cols, const char* file) {
    FILE* fp = fopen(file, "r");
    if (!fp) { printf("The file %s was not opened successfully", file); exit(-1); }
    char str[STR_SIZE];
    float val;
    for (int i = 0; i < grid_rows; i++)
        for (int j = 0; j < grid_cols; j++) {
            if (fgets(str, STR_SIZE, fp) == NULL) { printf("Error reading file\n"); exit(-1); }
            if (feof(fp)) { printf("not enough lines in file"); exit(-1); }
            if (sscanf(str, "%f", &val) != 1) { printf("invalid file format"); exit(-1); }
            vect[i * grid_cols + j] = val;
        }
    fclose(fp);
}

static void usage(char* progname) {
    fprintf(stderr,
        "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> "
        "<temp_file> <power_file> <output_file>\n", progname);
    exit(1);
}

int main(int argc, char** argv) {
    if (argc < 7) usage(argv[0]);

    int grid_rows        = atoi(argv[1]);
    int grid_cols        = atoi(argv[1]);
    int pyramid_height   = atoi(argv[2]);
    int total_iterations = atoi(argv[3]);
    const char* tfile    = argv[4];
    const char* pfile    = argv[5];
    const char* ofile    = argv[6];

    if (grid_rows <= 0 || pyramid_height <= 0 || total_iterations <= 0) usage(argv[0]);

    printf("Work-group size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

    const int n = grid_rows * grid_cols;
    float* h_temp  = (float*)malloc(n * sizeof(float));
    float* h_power = (float*)malloc(n * sizeof(float));
    if (!h_temp || !h_power) { printf("unable to allocate memory"); exit(-1); }

    readinput(h_temp,  grid_rows, grid_cols, tfile);
    readinput(h_power, grid_rows, grid_cols, pfile);

    const float grid_height = chip_height / grid_rows;
    const float grid_width  = chip_width  / grid_cols;
    const float Cap         = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    const float Rx          = grid_width  / (2.0f * K_SI * t_chip * grid_height);
    const float Ry          = grid_height / (2.0f * K_SI * t_chip * grid_width);
    const float Rz          = t_chip      / (K_SI  * grid_height * grid_width);
    const float max_slope   = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    const float step        = PRECISION / max_slope;
    const float step_div_Cap = step / Cap;
    const float Rx_1 = 1.0f / Rx;
    const float Ry_1 = 1.0f / Ry;
    const float Rz_1 = 1.0f / Rz;

    long long start_offload = get_time();

    float* d_power = (float*)malloc(n * sizeof(float));
    float* d_temp0 = (float*)malloc(n * sizeof(float));
    float* d_temp1 = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) { d_power[i] = h_power[i]; d_temp0[i] = h_temp[i]; }

    #pragma omp target enter data map(to: d_power[0:n], d_temp0[0:n]) \
        map(alloc: d_temp1[0:n])

    const int rows = grid_rows;
    const int cols = grid_cols;

    // use pointer arrays for double-buffering
    float* d_bufs[2] = {d_temp0, d_temp1};
    int src = 0, dst = 1;

    auto t_start = std::chrono::steady_clock::now();

    for (int t = 0; t < total_iterations; t += pyramid_height) {
        int iters = pyramid_height < (total_iterations - t) ?
                    pyramid_height : (total_iterations - t);

        for (int s = 0; s < iters; s++) {
            float* sv = d_bufs[src];
            float* dv = d_bufs[dst];

            #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    const int rN = row > 0        ? row - 1 : 0;
                    const int rS = row < rows - 1 ? row + 1 : rows - 1;
                    const int cW = col > 0        ? col - 1 : 0;
                    const int cE = col < cols - 1 ? col + 1 : cols - 1;
                    const float tc = sv[row * cols + col];
                    dv[row * cols + col] = tc + step_div_Cap * (
                        d_power[row * cols + col] +
                        (sv[rS * cols + col] + sv[rN * cols + col] - 2.0f * tc) * Ry_1 +
                        (sv[row * cols + cE] + sv[row * cols + cW] - 2.0f * tc) * Rx_1 +
                        (amb_temp - tc) * Rz_1);
                }
            }
            src = 1 - src;
            dst = 1 - dst;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    printf("Total kernel execution time %f (s)\n", (float)ns * 1e-9f);

    // Copy result back to host
    #pragma omp target update from(d_temp0[0:n], d_temp1[0:n])
    for (int i = 0; i < n; i++) h_power[i] = d_bufs[src][i];

    #pragma omp target exit data map(delete: d_power[0:n], d_temp0[0:n], d_temp1[0:n])

    long long end_offload = get_time();
    printf("Device offloading time: %.3f seconds\n",
           (float)(end_offload - start_offload) / 1000000.0f);

    writeoutput(h_power, grid_rows, grid_cols, ofile);

    free(h_temp); free(h_power);
    free(d_power); free(d_temp0); free(d_temp1);
    return 0;
}
