// Move-To-Front encoding - OpenMP port (host-only; MTF is inherently sequential)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<char> mtf_reference(const std::vector<char>& word) {
    std::vector<char> list(256);
    std::vector<char> d_list(256);
    std::vector<char> d_word(word);

    for (std::size_t counter = 0; counter < word.size(); ++counter) {
        std::copy(list.begin(), list.end(), d_list.begin());
        char w    = d_word[counter];
        auto iter = std::find(d_list.begin(), d_list.end(), w);
        if (d_list[0] != w) {
            std::copy(d_list.begin(), iter, list.begin() + 1);
            list[0] = w;
        }
    }

    std::copy(list.begin(), list.end(), d_list.begin());
    for (std::size_t counter = 0; counter < list.size(); ++counter) {
        auto iter = std::find(d_word.begin(), d_word.end(), d_list[counter]);
        while (iter != d_word.end()) {
            *iter = static_cast<char>(counter);
            iter  = std::find(iter + 1, d_word.end(), d_list[counter]);
        }
    }
    return d_word;
}

static std::vector<char> mtf_omp(const std::vector<char>& word) {
    std::vector<char> list(256, 0);
    std::vector<char> d_list(256);
    std::vector<char> d_word(word);

    for (std::size_t counter = 0; counter < word.size(); ++counter) {
        std::copy(list.begin(), list.end(), d_list.begin());
        char w   = d_word[counter];
        int  pos = -1;
        for (int k = 0; k < 256; ++k) {
            if (d_list[k] == w) { pos = k; break; }
        }
        if (d_list[0] != w) {
            int n = (pos >= 0) ? pos : 255;
            for (int k = n; k > 0; --k) list[k] = d_list[k - 1];
            list[0] = w;
        }
    }

    std::copy(list.begin(), list.end(), d_list.begin());
    for (int counter = 0; counter < 256; ++counter) {
        char sym = d_list[counter];
        for (std::size_t i = 0; i < word.size(); ++i) {
            if (d_word[i] == sym) d_word[i] = static_cast<char>(counter);
        }
    }
    return d_word;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <string length> <repeat>\n", argv[0]);
        return 1;
    }
    const std::size_t len    = static_cast<std::size_t>(atol(argv[1]));
    const int         repeat = atoi(argv[2]);

    static const char* alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    std::vector<char> word(len);
    srand(123);
    for (std::size_t i = 0; i < len; ++i) word[i] = alpha[rand() % 52];

    auto omp_result = mtf_omp(word);
    auto ref_result = mtf_reference(word);
    bool ok         = (omp_result == ref_result);
    printf("%s\n", ok ? "PASS" : "FAIL");
    if (!ok) return 1;

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; ++r) mtf_omp(word);
    auto t1 = std::chrono::steady_clock::now();
    double secs =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
        * 1e-9 / repeat;
    printf("Average execution time: %f (s)\n", secs);
    return 0;
}
