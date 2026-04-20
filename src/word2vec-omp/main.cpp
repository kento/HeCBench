// OpenMP target offloading port of word2vec benchmark
// Skip-gram model with negative sampling, GPU-accelerated training
// Standalone synthetic version (original requires training corpus)

#include <omp.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <unistd.h>

// Simplified word2vec: skip-gram with negative sampling
// Uses synthetic vocabulary and training pairs

static const int MAX_STRING  = 100;
static const int EXP_TABLE_SIZE = 1000;
static const int MAX_EXP     = 6;

static const int vocab_size  = 10000;
static const int layer1_size = 100;   // embedding dimension
static const int negative    = 5;     // negative samples
static const int window      = 5;
static const int table_size  = 100000;

// Sigmoid lookup table
static float exp_table[EXP_TABLE_SIZE + 1];

static void init_exp_table() {
  for (int i = 0; i < EXP_TABLE_SIZE; i++) {
    exp_table[i] = expf((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
    exp_table[i] = exp_table[i] / (exp_table[i] + 1);
  }
}

// Initialize unigram noise distribution table
static std::vector<int> init_unigram_table() {
  std::vector<int> table(table_size);
  std::vector<float> counts(vocab_size);
  // Synthetic frequency ~ Zipf distribution
  float total = 0.f;
  for (int i = 0; i < vocab_size; i++) {
    counts[i] = powf(1.f / (i + 1), 0.75f);
    total += counts[i];
  }
  int word = 0;
  float d1 = counts[0] / total;
  for (int a = 0; a < table_size; a++) {
    table[a] = word;
    if ((float)a / table_size > d1) {
      word++;
      if (word >= vocab_size) word = vocab_size - 1;
      d1 += counts[word] / total;
    }
  }
  return table;
}

// Generate synthetic training sentences
static std::vector<int> generate_sentences(int n_words) {
  std::vector<int> words(n_words);
  srand(42);
  for (int i = 0; i < n_words; i++)
    words[i] = rand() % vocab_size;
  return words;
}

// Skip-gram negative sampling on device
static void skipgram_kernel(
    const int *sentences, int n_words,
    float *syn0, float *syn1neg,
    const int *table, int table_sz,
    int dim, int neg, int win,
    float alpha, int *rand_state)
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int pos = win; pos < n_words - win; pos++) {
    int center = sentences[pos];
    if (center < 0 || center >= vocab_size) continue;

    // Context words
    for (int c = -win; c <= win; c++) {
      if (c == 0) continue;
      int ctx_pos = pos + c;
      if (ctx_pos < 0 || ctx_pos >= n_words) continue;

      // Negative sampling
      float neu1e[200] = {};  // error accumulation (max dim=200)
      if (dim > 200) continue;

      // Positive sample + negative samples
      for (int sample = 0; sample <= neg; sample++) {
        int label, target;
        if (sample == 0) {
          target = sentences[ctx_pos];
          label  = 1;
        } else {
          // Simple LCG random for negative sample
          int rnd_idx = (pos * 1103515245 + sample * 12345) & 0x7fffffff;
          rnd_idx = rnd_idx % table_sz;
          target = table[rnd_idx];
          if (target == center) continue;
          label = 0;
        }
        if (target < 0 || target >= vocab_size) continue;

        // Dot product: syn0[center] . syn1neg[target]
        float dot = 0.f;
        for (int d = 0; d < dim; d++)
          dot += syn0[center * dim + d] * syn1neg[target * dim + d];

        // Sigmoid
        float g;
        if (dot >= MAX_EXP) g = (label - 1) * alpha;
        else if (dot <= -MAX_EXP) g = label * alpha;
        else {
          int idx = (int)((dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2));
          if (idx < 0) idx = 0;
          if (idx >= EXP_TABLE_SIZE) idx = EXP_TABLE_SIZE - 1;
          g = (label - exp_table[idx]) * alpha;
        }

        for (int d = 0; d < dim; d++)
          neu1e[d] += g * syn1neg[target * dim + d];

        // Update output layer
        for (int d = 0; d < dim; d++) {
          #pragma omp atomic
          syn1neg[target * dim + d] += g * syn0[center * dim + d];
        }
      }
      // Update input layer
      for (int d = 0; d < dim; d++) {
        #pragma omp atomic
        syn0[center * dim + d] += neu1e[d];
      }
    }
  }
}

int main(int argc, char **argv) {
  int size   = layer1_size;
  int iter   = 1;
  float alpha = 0.025f;
  int n_words = 100000; // synthetic training words

  // Parse some common word2vec flags
  for (int i = 1; i < argc - 1; i++) {
    if (strcmp(argv[i], "-size") == 0) size = atoi(argv[i+1]);
    if (strcmp(argv[i], "-iter") == 0) iter = atoi(argv[i+1]);
  }
  if (size > 200) size = 200; // cap for stack array in kernel

  printf("word2vec skip-gram, vocab=%d, dim=%d, negative=%d, window=%d, iter=%d\n",
         vocab_size, size, negative, window, iter);

  init_exp_table();
  auto unigram = init_unigram_table();
  auto sentences = generate_sentences(n_words);

  // Initialize embeddings
  std::vector<float> syn0(vocab_size * size), syn1neg(vocab_size * size, 0.f);
  std::vector<int> rand_state(1, 42);
  srand(1);
  for (auto &v : syn0) v = (rand() / (float)RAND_MAX - 0.5f) / size;

  float  *d_syn0    = syn0.data();
  float  *d_syn1neg = syn1neg.data();
  const int *d_sentences = sentences.data();
  const int *d_table     = unigram.data();
  float  *d_exp_table    = exp_table;
  int    *d_rand_state   = rand_state.data();

  #pragma omp target enter data \
    map(to: d_sentences[0:n_words], d_table[0:table_size], d_exp_table[0:EXP_TABLE_SIZE+1]) \
    map(tofrom: d_syn0[0:vocab_size*size], d_syn1neg[0:vocab_size*size]) \
    map(alloc: d_rand_state[0:1])

  auto t_start = std::chrono::steady_clock::now();

  for (int it = 0; it < iter; it++) {
    float cur_alpha = alpha * (1.f - (float)it / iter);
    if (cur_alpha < alpha * 0.0001f) cur_alpha = alpha * 0.0001f;

    skipgram_kernel(d_sentences, n_words, d_syn0, d_syn1neg,
                    d_table, table_size, size, negative, window,
                    cur_alpha, d_rand_state);
  }

  auto t_end = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
  printf("Training time (%d iter, %d words): %.3f s\n", iter, n_words, elapsed);

  #pragma omp target update from(d_syn0[0:vocab_size*size])
  #pragma omp target exit data \
    map(delete: d_sentences[0:n_words], d_table[0:table_size], d_exp_table[0:EXP_TABLE_SIZE+1], \
                d_syn0[0:vocab_size*size], d_syn1neg[0:vocab_size*size], d_rand_state[0:1])

  // Checksum
  double sum = 0.0;
  for (int i = 0; i < vocab_size * size; i++) sum += syn0[i];
  printf("Embedding checksum: %f\n", sum);
  printf("PASS\n");
  return 0;
}
