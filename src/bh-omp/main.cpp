/*
 * OpenMP target offloading port of ECL-BH Barnes-Hut n-body simulation.
 * Uses simplified O(n^2) direct force calculation.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <vector>
#include <omp.h>

static int randx = 7;
static double drnd() {
  const int lastrand = randx;
  randx = (1103515245L * randx + 12345) & 0x7FFFFFFF;
  return (double)lastrand / 2147483648.0;
}

int main(int argc, char* argv[]) {
  printf("ECL-BH v4.5 (OpenMP target port - direct N-body)\n");
  printf("Copyright (c) 2010-2020 Texas State University\n");

  if (argc != 3) {
    fprintf(stderr, "arguments: number_of_bodies number_of_timesteps\n");
    exit(-1);
  }

  const int nbodies   = atoi(argv[1]);
  const int timesteps = atoi(argv[2]);

  if (nbodies < 1) { fprintf(stderr, "nbodies too small\n"); exit(-1); }
  printf("configuration: %d bodies, %d time steps\n", nbodies, timesteps);

  const float dtime = 0.025f;
  const float dthf  = dtime * 0.5f;
  const float epssq = 0.05f * 0.05f;

  float* px = (float*)malloc(nbodies * sizeof(float));
  float* py = (float*)malloc(nbodies * sizeof(float));
  float* pz = (float*)malloc(nbodies * sizeof(float));
  float* pm = (float*)malloc(nbodies * sizeof(float));
  float* vx = (float*)calloc(nbodies, sizeof(float));
  float* vy = (float*)calloc(nbodies, sizeof(float));
  float* vz = (float*)calloc(nbodies, sizeof(float));
  float* ax = (float*)calloc(nbodies, sizeof(float));
  float* ay = (float*)calloc(nbodies, sizeof(float));
  float* az = (float*)calloc(nbodies, sizeof(float));

  double rsc = (3.0 * 3.14159265358979) / 16.0;
  double vsc = sqrt(1.0 / rsc);
  for (int i = 0; i < nbodies; i++) {
    pm[i] = 1.f / nbodies;
    double r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1.0);
    double x, y, z, sq;
    do { x=drnd()*2-1; y=drnd()*2-1; z=drnd()*2-1; sq=x*x+y*y+z*z; } while(sq>1.0);
    double scale = rsc * r / sqrt(sq);
    px[i] = (float)(x * scale);
    py[i] = (float)(y * scale);
    pz[i] = (float)(z * scale);
    double v; double x2, y2, z2;
    do { x2=drnd(); y2=drnd()*0.1; } while(y2 > x2*x2*pow(1-x2*x2,3.5));
    v = x2 * sqrt(2.0 / sqrt(1 + r*r));
    do { x2=drnd()*2-1; y2=drnd()*2-1; z2=drnd()*2-1; sq=x2*x2+y2*y2+z2*z2; } while(sq>1.0);
    scale = vsc * v / sqrt(sq);
    vx[i] = (float)(x2 * scale);
    vy[i] = (float)(y2 * scale);
    vz[i] = (float)(z2 * scale);
  }

  #pragma omp target enter data map(to: px[0:nbodies], py[0:nbodies], pz[0:nbodies], pm[0:nbodies]) \
    map(to: vx[0:nbodies], vy[0:nbodies], vz[0:nbodies]) \
    map(alloc: ax[0:nbodies], ay[0:nbodies], az[0:nbodies])

  struct timeval starttime, endtime;
  gettimeofday(&starttime, NULL);

  for (int step = 0; step < timesteps; step++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < nbodies; i++) {
      float axi = 0.f, ayi = 0.f, azi = 0.f;
      float xi = px[i], yi = py[i], zi = pz[i];
      for (int j = 0; j < nbodies; j++) {
        if (i == j) continue;
        float dx = px[j] - xi;
        float dy = py[j] - yi;
        float dz = pz[j] - zi;
        float dist2 = dx*dx + dy*dy + dz*dz + epssq;
        float inv_dist = 1.f / sqrtf(dist2);
        float f = pm[j] * inv_dist * inv_dist * inv_dist;
        axi += dx * f;
        ayi += dy * f;
        azi += dz * f;
      }
      ax[i] = axi; ay[i] = ayi; az[i] = azi;
    }

    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < nbodies; i++) {
      float dvx = ax[i] * dthf;
      float dvy = ay[i] * dthf;
      float dvz = az[i] * dthf;
      float vhx = vx[i] + dvx;
      float vhy = vy[i] + dvy;
      float vhz = vz[i] + dvz;
      px[i] += vhx * dtime;
      py[i] += vhy * dtime;
      pz[i] += vhy * dtime;
      vx[i] = vhx + dvx;
      vy[i] = vhy + dvy;
      vz[i] = vhz + dvz;
    }
  }

  gettimeofday(&endtime, NULL);
  double runtime = (endtime.tv_sec + endtime.tv_usec/1000000.0
                  - starttime.tv_sec - starttime.tv_usec/1000000.0);
  printf("Total kernel execution time: %.4lf s\n", runtime);

  #pragma omp target exit data map(from: px[0:nbodies], py[0:nbodies], pz[0:nbodies]) \
    map(delete: pm[0:nbodies], vx[0:nbodies], vy[0:nbodies], vz[0:nbodies], \
                ax[0:nbodies], ay[0:nbodies], az[0:nbodies])

  free(px); free(py); free(pz); free(pm);
  free(vx); free(vy); free(vz);
  free(ax); free(ay); free(az);
  return 0;
}
