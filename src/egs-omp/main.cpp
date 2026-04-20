// OpenMP target offloading port of egs-kokkos (simplified EGS radiation transport)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#define NUM_PARTICLES   (32 * 64)
#define NUM_ITERATIONS  10
#define NUM_STEPS       100
#define DETECTOR_SIZE   (64 * 64)

struct Particle {
  float x, y, z;
  float ux, uy, uz;
  float E;
  int   type;
  int   region;
  int   alive;
};

int main(int argc, char** argv)
{
  Particle* d_particles = (Particle*)malloc(NUM_PARTICLES * sizeof(Particle));
  float*    d_detector  = (float*)   malloc(DETECTOR_SIZE * sizeof(float));

  for (int i = 0; i < DETECTOR_SIZE; i++) d_detector[i] = 0.f;

  #pragma omp target enter data map(alloc: d_particles[0:NUM_PARTICLES]) \
                                map(to: d_detector[0:DETECTOR_SIZE])

  // Initialise particles on device
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < NUM_PARTICLES; i++) {
    d_particles[i].x  = 0.f; d_particles[i].y = 0.f; d_particles[i].z = 0.f;
    d_particles[i].ux = 0.f; d_particles[i].uy = 0.f; d_particles[i].uz = 1.f;
    d_particles[i].E  = 1.25f;
    d_particles[i].type   = 0;
    d_particles[i].region = 0;
    d_particles[i].alive  = 1;
  }

  #pragma omp target update to(d_detector[0:DETECTOR_SIZE])

  float time_sim = 0.f;
  float time_sum = 0.f;

  auto sim_start = std::chrono::steady_clock::now();

  for (int step = 0; step < NUM_STEPS; step++) {

    auto ks = std::chrono::steady_clock::now();
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < NUM_PARTICLES; i++) {
        if (!d_particles[i].alive) continue;
        float mu = (d_particles[i].type == 0) ? 0.07f : 0.15f;
        float step_len = -logf(0.5f) / mu;
        d_particles[i].x += d_particles[i].ux * step_len;
        d_particles[i].y += d_particles[i].uy * step_len;
        d_particles[i].z += d_particles[i].uz * step_len;
        d_particles[i].E *= 0.95f;
        if (d_particles[i].E < 0.01f) d_particles[i].alive = 0;
      }
    }
    auto ke = std::chrono::steady_clock::now();
    time_sim += (float)std::chrono::duration_cast<std::chrono::microseconds>(ke - ks).count() / 1000.f;

    auto ds = std::chrono::steady_clock::now();
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < NUM_PARTICLES; i++) {
      if (!d_particles[i].alive) continue;
      int det_x = (int)(d_particles[i].x / 10.f) & 63;
      int det_y = (int)(d_particles[i].y / 10.f) & 63;
      float contrib = d_particles[i].E * 0.01f;
      #pragma omp atomic update
      d_detector[det_x * 64 + det_y] += contrib;
    }
    auto de = std::chrono::steady_clock::now();
    time_sum += (float)std::chrono::duration_cast<std::chrono::microseconds>(de - ds).count() / 1000.f;
  }

  auto sim_end = std::chrono::steady_clock::now();
  float total_time = (float)std::chrono::duration_cast<std::chrono::milliseconds>(sim_end - sim_start).count();

  printf("\nTiming statistics\n");
  printf("  Elapsed time  . . . . . . . %.2f ms (%.2f %%)\n", total_time, 100.0f);
  printf("  Simulation kernel . . . . . %.2f ms (%.2f %%)\n",
         time_sim, 100.f * time_sim / total_time);
  printf("  Summing kernel  . . . . . . %.2f ms (%.2f %%)\n",
         time_sum, 100.f * time_sum / total_time);

  #pragma omp target update from(d_detector[0:DETECTOR_SIZE])

  float total_dose = 0.f;
  for (int i = 0; i < DETECTOR_SIZE; i++) total_dose += d_detector[i];
  printf("Total detector dose: %f\n", total_dose);

  #pragma omp target exit data map(delete: d_particles[0:NUM_PARTICLES], \
                                           d_detector[0:DETECTOR_SIZE])
  free(d_particles);
  free(d_detector);
  return 0;
}
