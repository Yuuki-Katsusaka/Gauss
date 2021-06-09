#include <stdlib.h>
#include <time.h>

#include <cmath>
#include <iostream>

float randrange(int N) { return (rand() * N) / (1.0 + RAND_MAX); }

template <typename T, int N>
void printMat(T A[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <int N, int K>
struct D_Method {
    template <typename T>
    static void gje(T A[N][N], T b[N], T x[N]) {
#pragma HLS INLINE
        // ピボット選択
        T max = 0.0;
        int buf_p = K - 1;
        for (int i = K - 1; i >= 0; i--) {
#pragma HLS PIPELINE
            T trg_p = A[i][N - K];
            if (std::fabs(trg_p) <= fabs(max)) continue;
            max = trg_p;
            buf_p = i;
        }
        if (buf_p != K - 1) {
            for (int j = 0; j < N; j++) {
#pragma HLS UNROLL
                T buf_a = A[K - 1][j];
                A[K - 1][j] = A[buf_p][j];
                A[buf_p][j] = buf_a;
            }
            T buf_b = b[K - 1];
            b[K - 1] = b[buf_p];
            b[buf_p] = buf_b;
        }

        // k行の全ての要素をk行の体格要素で割る
        T pivot = A[K - 1][N - K];
        for (int j = N - K; j < N; j++) {
#pragma HLS UNROLL
            A[K - 1][j] /= pivot;
        }
        b[K - 1] /= pivot;

        // k列目の対角要素を0にする
        T temp1[N], temp2[N];
#pragma HLS ARRAY_PARTITION variable = temp1 dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = temp2 dim = 1 complete
        for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
            temp1[i] = A[i][N - K];
            temp2[i] = A[K - 1][i];
        }
        for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE
            for (int j = N - K; j < N; j++) {
#pragma HLS UNROLL
                if (i == K - 1) continue;
                A[i][j] -= temp1[i] * temp2[j];
            }
        }

        T temp3 = b[K - 1];
        for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
            if (i == K - 1) continue;
            b[i] -= temp1[i] * temp3;
        }

        D_Method<N, K - 1>::gje(A, b, x);
    }
};

template <int N>
struct D_Method<N, 0> {
    template <typename T>
    static void gje(T A[N][N], T b[N], T x[N]) {
#pragma HLS INLINE
        for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
            x[i] = b[N - i - 1];
        }
    }
};

#define NN 100

int main(void) {
    // float A[NN][NN] = {{1,1,1,1}, {1,1,1,-1}, {1,1,-1,1}, {1,-1,1,1}};
    // float b[NN] = {0,4,-4,2};
    // float x[NN];
    float A[NN][NN], b[NN], x[NN], x_t[NN];

    srand((unsigned int)time(0));
    for (int i = 0; i < NN; i++) {
        A[i][i] = randrange(100) + 1.0;
        for (int j = 0; j < NN; j++) {
            if (j == i) continue;
            A[i][j] = randrange(100);
        }
        x_t[i] = randrange(100);
    }
    for (int i = 0; i < NN; i++) {
        b[i] = 0.0;
        for (int j = 0; j < NN; j++) {
            b[i] += A[i][j] * x_t[j];
        }
    }

    printMat(A);

    D_Method<NN, NN>::gje(A, b, x);

    std::cout << "gauss:" << std::endl;
    for (int i = 0; i < NN; i++) {
        std::cout << x[i] << ", ";
    }
    std::cout << "\nx_true" << std::endl;
    for (int i = 0; i < NN; i++) {
        std::cout << x_t[i] << ", ";
    }
    std::cout << std::endl;

    return 0;
}