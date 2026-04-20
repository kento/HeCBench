// OpenMP target offloading port of dxtc1-kokkos (DXT1 texture compression)

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <cassert>
#include <chrono>
#include <omp.h>

#include "dds.h"
#include "block.h"
#include "permutations.h"
#include "shrUtils.h"

#define ERROR_THRESHOLD 0.02f

struct F4 { float x, y, z, w; };

#pragma omp declare target
static inline F4 make_f4(float x, float y, float z, float w)
{ F4 v; v.x=x; v.y=y; v.z=z; v.w=w; return v; }
static inline F4 f4_add(F4 a, F4 b) { return make_f4(a.x+b.x,a.y+b.y,a.z+b.z,a.w+b.w); }
static inline F4 f4_sub(F4 a, F4 b) { return make_f4(a.x-b.x,a.y-b.y,a.z-b.z,a.w-b.w); }
static inline F4 f4_mul(F4 a, float s) { return make_f4(a.x*s,a.y*s,a.z*s,a.w*s); }

static inline float saturatef(float v)
{ return (v < 0.f ? 0.f : (v > 1.f ? 1.f : v)); }

static inline unsigned short roundHalf(float v, int bits)
{ return (unsigned short)(int)(v * ((1<<bits)-1) + 0.5f); }

static void compressBlock(
    const unsigned int* image,
    unsigned int*       perm,
    unsigned int*       result_x,
    unsigned int*       result_y,
    int bid)
{
  F4 colors[16];
  for (int i = 0; i < 16; i++) {
    unsigned int c = image[bid * 16 + i];
    colors[i] = make_f4(
      ((c >>  0) & 0xFF) * 0.003921568627f,
      ((c >>  8) & 0xFF) * 0.003921568627f,
      ((c >> 16) & 0xFF) * 0.003921568627f,
      0.f);
  }

  F4 sum = make_f4(0,0,0,0);
  for (int i = 0; i < 16; i++) {
    sum.x += colors[i].x; sum.y += colors[i].y; sum.z += colors[i].z;
  }

  float cov[6] = {};
  for (int i = 0; i < 16; i++) {
    F4 d = f4_sub(colors[i], f4_mul(sum, 0.0625f));
    cov[0] += d.x*d.x; cov[1] += d.x*d.y; cov[2] += d.x*d.z;
    cov[3] += d.y*d.y; cov[4] += d.y*d.z; cov[5] += d.z*d.z;
  }

  F4 axis = make_f4(1,1,1,0);
  for (int k = 0; k < 8; k++) {
    float x = axis.x*cov[0] + axis.y*cov[1] + axis.z*cov[2];
    float y = axis.x*cov[1] + axis.y*cov[3] + axis.z*cov[4];
    float z = axis.x*cov[2] + axis.y*cov[4] + axis.z*cov[5];
    float m = fmaxf(fmaxf(fabsf(x), fabsf(y)), fabsf(z));
    if (m < 1e-6f) break;
    float iv = 1.f / m;
    axis = make_f4(x*iv, y*iv, z*iv, 0);
  }

  float proj[16];
  for (int i = 0; i < 16; i++)
    proj[i] = colors[i].x*axis.x + colors[i].y*axis.y + colors[i].z*axis.z;

  int rank[16] = {};
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++)
      if (proj[j] < proj[i]) rank[i]++;
  }
  for (int i = 0; i < 15; i++)
    for (int j = i+1; j < 16; j++)
      if (rank[i] == rank[j]) rank[j]++;

  F4 sorted[16];
  for (int i = 0; i < 16; i++) sorted[rank[i]] = colors[i];
  int xrefs[16];
  for (int i = 0; i < 16; i++) xrefs[rank[i]] = i;

  float bestError = FLT_MAX;
  unsigned int bestStart = 0, bestEnd = 0, bestPerm = 0;

  for (int p = 0; p < 160; p++) {
    unsigned int permutation = perm[p];

    F4 start_c = sorted[0];
    F4 end_c   = sorted[15];

    unsigned short sx = roundHalf(saturatef(start_c.x), 5);
    unsigned short sy = roundHalf(saturatef(start_c.y), 6);
    unsigned short sz = roundHalf(saturatef(start_c.z), 5);
    unsigned short ws = (sx<<11)|(sy<<5)|sz;
    start_c.x = sx * 0.03227752766457f;
    start_c.y = sy * 0.01583151765563f;
    start_c.z = sz * 0.03227752766457f;

    unsigned short ex = roundHalf(saturatef(end_c.x), 5);
    unsigned short ey = roundHalf(saturatef(end_c.y), 6);
    unsigned short ez = roundHalf(saturatef(end_c.z), 5);
    unsigned short we = (ex<<11)|(ey<<5)|ez;
    end_c.x = ex * 0.03227752766457f;
    end_c.y = ey * 0.01583151765563f;
    end_c.z = ez * 0.03227752766457f;

    float error = 0.f;
    for (int i = 0; i < 16; i++) {
      int idx = (permutation >> (2*i)) & 3;
      float alpha, beta;
      if (ws > we) {
        const float at4[4] = {9.f, 0.f, 6.f, 3.f};
        alpha = at4[idx] / 9.f;
        beta  = 1.f - alpha;
      } else {
        const float at3[4] = {4.f, 0.f, 2.f, 2.f};
        alpha = at3[idx] / 4.f;
        beta  = 1.f - alpha;
      }
      F4 col = f4_add(f4_mul(start_c, alpha), f4_mul(end_c, beta));
      F4 diff = f4_sub(sorted[i], col);
      error += diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
    }

    if (error < bestError) {
      bestError = error;
      bestStart = ws;
      bestEnd   = we;
      bestPerm  = permutation;
    }
  }

  unsigned int indices = 0;
  for (int i = 0; i < 16; i++) {
    int ref = xrefs[i];
    indices |= ((bestPerm >> (2*ref)) & 3) << (2*i);
  }

  result_x[bid] = (bestEnd << 16) | bestStart;
  result_y[bid] = indices;
}
#pragma omp end declare target

int main(int argc, char** argv)
{
  if (argc != 4) {
    printf("Usage: %s <path to image> <path to reference image> <repeat>\n", argv[0]);
    return 1;
  }

  const int numIterations = atoi(argv[3]);

  unsigned int width, height;
  unsigned int* h_img = nullptr;

  shrLoadPPM4ub(argv[1], (unsigned char**)&h_img, &width, &height);
  assert(h_img);
  printf("Loaded '%s', %d x %d pixels\n\n", argv[1], width, height);

  const unsigned int memSize = width * height;
  unsigned int* block_image = (unsigned int*)malloc(memSize * sizeof(unsigned int));

  for (unsigned int by = 0; by < height/4; by++)
    for (unsigned int bx = 0; bx < width/4; bx++)
      for (int i = 0; i < 16; i++) {
        int x = i & 3, y = i / 4;
        block_image[(by * width/4 + bx) * 16 + i] =
          h_img[(by*4+y) * width + bx*4+x];
      }

  unsigned int perms[1024];
  computePermutations(perms);

  int blocks = (width/4) * (height/4);
  const unsigned int compressedSize = blocks * 8;
  unsigned int* h_result = (unsigned int*)malloc(compressedSize);

  unsigned int* d_image  = (unsigned int*)malloc(memSize * sizeof(unsigned int));
  unsigned int* d_perm   = (unsigned int*)malloc(1024 * sizeof(unsigned int));
  unsigned int* d_res_x  = (unsigned int*)malloc(blocks * sizeof(unsigned int));
  unsigned int* d_res_y  = (unsigned int*)malloc(blocks * sizeof(unsigned int));

  for (unsigned int i = 0; i < memSize; i++) d_image[i] = block_image[i];
  for (int i = 0; i < 1024; i++)             d_perm[i]  = perms[i];

  #pragma omp target enter data map(to: d_image[0:memSize], d_perm[0:1024]) \
                                map(alloc: d_res_x[0:blocks], d_res_y[0:blocks])

  auto start = std::chrono::steady_clock::now();

  for (int iter = 0; iter < numIterations; iter++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int bid = 0; bid < blocks; bid++) {
      compressBlock(d_image, d_perm, d_res_x, d_res_y, bid);
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / numIterations);

  #pragma omp target update from(d_res_x[0:blocks], d_res_y[0:blocks])

  for (int i = 0; i < blocks; i++) {
    ((unsigned int*)h_result)[2*i]   = d_res_x[i];
    ((unsigned int*)h_result)[2*i+1] = d_res_y[i];
  }

  #pragma omp target exit data map(delete: d_image[0:memSize], d_perm[0:1024], \
                                           d_res_x[0:blocks], d_res_y[0:blocks])

  // Write DDS
  char output_filename[1024];
  strcpy(output_filename, argv[1]);
  strcpy(output_filename + strlen(argv[1]) - 3, "dds");
  FILE* fp = fopen(output_filename, "wb");
  assert(fp);

  DDSHeader header;
  header.fourcc = FOURCC_DDS;
  header.size = 124;
  header.flags  = (DDSD_WIDTH|DDSD_HEIGHT|DDSD_CAPS|DDSD_PIXELFORMAT|DDSD_LINEARSIZE);
  header.height = height; header.width = width; header.pitch = compressedSize;
  header.depth = 0; header.mipmapcount = 0;
  memset(header.reserved, 0, sizeof(header.reserved));
  header.pf.size = 32; header.pf.flags = DDPF_FOURCC; header.pf.fourcc = FOURCC_DXT1;
  header.pf.bitcount = header.pf.rmask = header.pf.gmask = header.pf.bmask = header.pf.amask = 0;
  header.caps.caps1 = DDSCAPS_TEXTURE;
  header.caps.caps2 = header.caps.caps3 = header.caps.caps4 = 0;
  header.notused = 0;

  fwrite(&header, sizeof(DDSHeader), 1, fp);
  fwrite(h_result, compressedSize, 1, fp);
  fclose(fp);

  fp = fopen(argv[2], "rb");
  assert(fp);
  fseek(fp, sizeof(DDSHeader), SEEK_SET);
  unsigned int* reference = (unsigned int*)malloc(compressedSize);
  fread(reference, compressedSize, 1, fp);
  fclose(fp);

  float rms = 0;
  for (unsigned int y = 0; y < height; y += 4)
    for (unsigned int x = 0; x < width; x += 4) {
      int idx = (y/4)*(width/4) + (x/4);
      int cmp = compareBlock(((BlockDXT1*)h_result)+idx, ((BlockDXT1*)reference)+idx);
      rms += cmp;
    }
  rms /= width * height * 3;
  printf("RMS(reference, result) = %f\n\n", rms);

  free(block_image); free(h_result); free(h_img); free(reference);
  free(d_image); free(d_perm); free(d_res_x); free(d_res_y);
  printf("%s\n", rms <= ERROR_THRESHOLD ? "PASS" : "FAIL");
  return 0;
}
