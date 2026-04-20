// OpenMP target offloading port of dwt2d-kokkos (2-D DWT)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>

static int  g_width      = 512;
static int  g_height     = 512;
static int  g_components = 3;
static int  g_bitDepth   = 8;
static int  g_dwtLvls    = 3;
static bool g_forward    = true;
static char g_srcFile[256] = "";
static char g_outFile[256] = "out.dwt";

static void parseDim(const char *s, int &w, int &h) {
  sscanf(s, "%dx%d", &w, &h);
}

static void usage() {
  printf("dwt2d [options] src_img.rgb <out.dwt>\n"
         "  -d WxH  dimensions (e.g. 1920x1080)\n"
         "  -c N    components (default 3)\n"
         "  -b N    bit depth (default 8)\n"
         "  -l N    DWT levels (default 3)\n"
         "  -f      forward transform\n"
         "  -r      reverse transform\n"
         "  -5      5/3 transform\n");
}

static void applyDWT2D(float* data, int w, int h, int fullWidth) {
  // Row transforms
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int row = 0; row < h; row++) {
    float tmp[4096];
    for (int c = 0; c < w; c++)
      tmp[c] = data[row * fullWidth + c];
    // Predict
    for (int i = 1; i < w - 1; i += 2)
      tmp[i] -= 0.5f * (tmp[i - 1] + tmp[i + 1]);
    if ((w & 1) == 0)
      tmp[w - 1] -= tmp[w - 2];
    // Update
    tmp[0] += 0.5f * tmp[1];
    for (int i = 2; i < w - 1; i += 2)
      tmp[i] += 0.25f * (tmp[i - 1] + tmp[i + 1]);
    if ((w & 1) == 1 && w > 1)
      tmp[w - 1] += 0.5f * tmp[w - 2];
    for (int c = 0; c < w; c++)
      data[row * fullWidth + c] = tmp[c];
  }

  // Column transforms
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int col = 0; col < w; col++) {
    float tmp[4096];
    for (int r = 0; r < h; r++)
      tmp[r] = data[r * fullWidth + col];
    // Predict
    for (int i = 1; i < h - 1; i += 2)
      tmp[i] -= 0.5f * (tmp[i - 1] + tmp[i + 1]);
    if ((h & 1) == 0)
      tmp[h - 1] -= tmp[h - 2];
    // Update
    tmp[0] += 0.5f * tmp[1];
    for (int i = 2; i < h - 1; i += 2)
      tmp[i] += 0.25f * (tmp[i - 1] + tmp[i + 1]);
    if ((h & 1) == 1 && h > 1)
      tmp[h - 1] += 0.5f * tmp[h - 2];
    for (int r = 0; r < h; r++)
      data[r * fullWidth + col] = tmp[r];
  }
}

static void nStage2dDWT(float* data, int pixWidth, int pixHeight, int levels) {
  int w = pixWidth, h = pixHeight;
  for (int lvl = 0; lvl < levels; lvl++) {
    applyDWT2D(data, w, h, pixWidth);
    w = (w + 1) / 2;
    h = (h + 1) / 2;
  }
}

static void writeComponent(const float *src, int pixels, const char *filename,
                            const char *suffix) {
  char outfile[512];
  snprintf(outfile, sizeof(outfile), "%s%s", filename, suffix);
  auto *result = new unsigned char[pixels];
  for (int i = 0; i < pixels; i++) {
    int v = (int)(src[i] + 128.0f);
    if (v > 255) v = 255;
    if (v < 0)   v = 0;
    result[i] = (unsigned char)v;
  }
  int fd = open(outfile, O_CREAT | O_WRONLY, 0644);
  if (fd != -1) {
    printf("\nWriting to %s (%d x %d)\n", outfile, g_width, g_height);
    write(fd, result, pixels);
    close(fd);
  }
  delete[] result;
}

int main(int argc, char **argv) {
  int argi = 1;
  while (argi < argc) {
    if (!strcmp(argv[argi], "-d") && argi + 1 < argc) {
      parseDim(argv[++argi], g_width, g_height);
    } else if (!strcmp(argv[argi], "-c") && argi + 1 < argc) {
      g_components = atoi(argv[++argi]);
    } else if (!strcmp(argv[argi], "-b") && argi + 1 < argc) {
      g_bitDepth = atoi(argv[++argi]);
    } else if (!strcmp(argv[argi], "-l") && argi + 1 < argc) {
      g_dwtLvls = atoi(argv[++argi]);
    } else if (!strcmp(argv[argi], "-f")) {
      g_forward = true;
    } else if (!strcmp(argv[argi], "-r")) {
      g_forward = false;
    } else if (!strcmp(argv[argi], "-5")) {
      // 5/3 is the only implemented filter
    } else if (!strcmp(argv[argi], "--help") || !strcmp(argv[argi], "-h")) {
      usage(); return 0;
    } else if (argv[argi][0] != '-') {
      if (g_srcFile[0] == '\0')
        strncpy(g_srcFile, argv[argi], sizeof(g_srcFile) - 1);
      else
        strncpy(g_outFile, argv[argi], sizeof(g_outFile) - 1);
    }
    argi++;
  }

  const int pixels    = g_width * g_height;
  const int imgBytes  = pixels * g_components;
  const int nComp     = g_components;

  auto *srcImg = new unsigned char[imgBytes];
  bool loaded = false;
  if (g_srcFile[0] != '\0') {
    char path[512];
    snprintf(path, sizeof(path), "../data/dwt2d/%s", g_srcFile);
    int fd = open(path, O_RDONLY, 0644);
    if (fd == -1) fd = open(g_srcFile, O_RDONLY, 0644);
    if (fd != -1) {
      if (read(fd, srcImg, imgBytes) == imgBytes) loaded = true;
      close(fd);
    }
  }
  if (!loaded) {
    printf("Input file not found — generating synthetic image "
           "(%dx%d, %d components)\n", g_width, g_height, g_components);
    for (int i = 0; i < imgBytes; i++)
      srcImg[i] = (unsigned char)(i * 7 + 13 * (i / g_width));
  }

  // Allocate per-component device arrays
  float** compData = new float*[nComp];
  for (int c = 0; c < nComp; c++) {
    compData[c] = (float*)malloc(pixels * sizeof(float));
    for (int i = 0; i < pixels; i++)
      compData[c][i] = (float)srcImg[i * nComp + c] - 128.0f;
    #pragma omp target enter data map(tofrom: compData[c][0:pixels])
  }

  auto t0 = std::chrono::high_resolution_clock::now();

  for (int c = 0; c < nComp; c++)
    nStage2dDWT(compData[c], g_width, g_height, g_dwtLvls);

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  printf("DWT processing time: %.3f (ms)\n", ms);

#ifdef OUTPUT
  for (int c = 0; c < nComp; c++) {
    #pragma omp target update from(compData[c][0:pixels])
    const char *sfx = (c == 0) ? ".r" : (c == 1) ? ".g" : ".b";
    writeComponent(compData[c], pixels, g_outFile, sfx);
  }
#endif

  for (int c = 0; c < nComp; c++) {
    #pragma omp target exit data map(delete: compData[c][0:pixels])
    free(compData[c]);
  }
  delete[] compData;
  delete[] srcImg;
  return 0;
}
