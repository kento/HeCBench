// OpenMP target port of marchingCubes-kokkos.

#include <omp.h>
#include <cstdio>
#include <cmath>
#include <chrono>
#include "tables.h"

constexpr unsigned int N   = 1024;
constexpr unsigned int Nd2 = N / 2;

#pragma omp declare target

inline float f_val(unsigned int x, unsigned int y, unsigned int z)
{
    constexpr float d = 2.0f / N;
    float xf = (int(x) - int(Nd2)) * d;
    float yf = (int(z) - int(Nd2)) * d;
    float zf = (int(z) - int(Nd2)) * d;
    return 1.f - 16.f * xf * yf * zf - 4.f * (xf*xf + yf*yf + zf*zf);
}

inline int popcount32(unsigned int v)
{
    int c = 0;
    while (v) { c += v & 1; v >>= 1; }
    return c;
}

#pragma omp end declare target

int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Usage: %s <isoValue> <repeat>\n", argv[0]);
        return 1;
    }
    float    isoValue = atof(argv[1]);
    unsigned repeat   = (unsigned)atoi(argv[2]);

    unsigned int* d_edgeTable = (unsigned int*)malloc(256 * sizeof(unsigned int));
    int*          d_triTable  = (int*)         malloc(256 * 16 * sizeof(int));

    for (int i = 0; i < 256;      i++) d_edgeTable[i] = edgeTable[i];
    for (int i = 0; i < 256 * 16; i++) d_triTable[i]  = triTable[i];

    #pragma omp target enter data \
        map(to: d_edgeTable[0:256], d_triTable[0:256*16])

    const long long Nv          = (long long)(N - 1);
    const long long totalVoxels = Nv * Nv * Nv;

    unsigned int activeVoxels = 0, totalVertices = 0, totalTriangles = 0;
    float elapsed = 0.f;

    for (unsigned r = 0; r < repeat; r++) {
        unsigned int av = 0, verts = 0, tris = 0;

        auto t0 = std::chrono::steady_clock::now();

        #pragma omp target teams distribute parallel for \
            reduction(+: av, verts, tris) thread_limit(256) \
            map(to: d_edgeTable[0:256], d_triTable[0:256*16])
        for (long long idx = 0; idx < totalVoxels; idx++) {
            long long tmp = idx;
            unsigned int xi = (unsigned int)(tmp % Nv); tmp /= Nv;
            unsigned int yi = (unsigned int)(tmp % Nv); tmp /= Nv;
            unsigned int zi = (unsigned int)(tmp);

            float v[8];
            v[0] = f_val(xi,   yi,   zi  );
            v[1] = f_val(xi+1, yi,   zi  );
            v[2] = f_val(xi+1, yi+1, zi  );
            v[3] = f_val(xi,   yi+1, zi  );
            v[4] = f_val(xi,   yi,   zi+1);
            v[5] = f_val(xi+1, yi,   zi+1);
            v[6] = f_val(xi+1, yi+1, zi+1);
            v[7] = f_val(xi,   yi+1, zi+1);

            float vmin = v[0], vmax = v[0];
            for (int k = 1; k < 8; k++) {
                if (v[k] < vmin) vmin = v[k];
                if (v[k] > vmax) vmax = v[k];
            }
            if (isoValue < vmin || isoValue > vmax) continue;

            av++;

            unsigned int cubeCase = 0;
            if (v[0] < isoValue) cubeCase |= 1;
            if (v[1] < isoValue) cubeCase |= 2;
            if (v[2] < isoValue) cubeCase |= 4;
            if (v[3] < isoValue) cubeCase |= 8;
            if (v[4] < isoValue) cubeCase |= 16;
            if (v[5] < isoValue) cubeCase |= 32;
            if (v[6] < isoValue) cubeCase |= 64;
            if (v[7] < isoValue) cubeCase |= 128;

            if (d_edgeTable[cubeCase] == 0) continue;

            verts += (unsigned int)popcount32(d_edgeTable[cubeCase]);

            int base = cubeCase * 16;
            int numTris = 0;
            for (int t = 0; t < 15; t += 3) {
                if (d_triTable[base + t] < 0) break;
                numTris++;
            }
            tris += (unsigned int)numTris;
        }

        auto t1 = std::chrono::steady_clock::now();
        elapsed += std::chrono::duration<float>(t1 - t0).count();

        activeVoxels  = av;
        totalVertices = verts;
        totalTriangles = tris;
    }

    printf("Active voxels: %u\n", activeVoxels);
    printf("Total vertices: %u\n", totalVertices);
    printf("Total triangles: %u\n", totalTriangles);
    printf("Average kernel time: %f s\n", elapsed / repeat);

    #pragma omp target exit data \
        map(delete: d_edgeTable[0:256], d_triTable[0:256*16])

    free(d_edgeTable); free(d_triTable);
    return 0;
}
