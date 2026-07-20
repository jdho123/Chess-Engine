#include "nnue_kernels.h"

namespace NNUE {
    void kernel_accumulator_hidden(
        std::array<uint8_t, ACCUMULATOR_SIZE>& x, 
        std::array<int8_t, ACCUMULATOR_SIZE * HIDDEN_SIZE>& w_T, 
        std::array<uint8_t, HIDDEN_SIZE>& y
    ) {
        for (size_t j = 0; j < HIDDEN_SIZE - 2; j += 3) {
            __m256i v_sum0_0 = _mm256_setzero_si256();
            __m256i v_sum0_1 = _mm256_setzero_si256();
            __m256i v_sum1_0 = _mm256_setzero_si256();
            __m256i v_sum1_1 = _mm256_setzero_si256();
            __m256i v_sum2_0 = _mm256_setzero_si256();
            __m256i v_sum2_1 = _mm256_setzero_si256();

            for (size_t i = 0; i < ACCUMULATOR_SIZE; i += 64) {
                __m256i v_x0 = _mm256_load_si256((const __m256i*)&x[i]);
                
                __m256i v_A0_0 = _mm256_load_si256((const __m256i*)&w_T[j * ACCUMULATOR_SIZE + i]);
                __m256i v_A1_0 = _mm256_load_si256((const __m256i*)&w_T[(j + 1) * ACCUMULATOR_SIZE + i]);
                __m256i v_A2_0 = _mm256_load_si256((const __m256i*)&w_T[(j + 2) * ACCUMULATOR_SIZE + i]);
                
                v_sum0_0 = _mm256_dpbusd_epi32(v_sum0_0, v_x0, v_A0_0);
                v_sum1_0 = _mm256_dpbusd_epi32(v_sum1_0, v_x0, v_A1_0);
                v_sum2_0 = _mm256_dpbusd_epi32(v_sum2_0, v_x0, v_A2_0);
                
                __m256i v_x1 = _mm256_load_si256((const __m256i*)&x[i + 32]);

                __m256i v_A0_1 = _mm256_load_si256((const __m256i*)&w_T[j * ACCUMULATOR_SIZE + i + 32]);
                __m256i v_A1_1 = _mm256_load_si256((const __m256i*)&w_T[(j + 1) * ACCUMULATOR_SIZE + i + 32]);
                __m256i v_A2_1 = _mm256_load_si256((const __m256i*)&w_T[(j + 2) * ACCUMULATOR_SIZE + i + 32]);
                
                v_sum0_1 = _mm256_dpbusd_epi32(v_sum0_1, v_x1, v_A0_1);
                v_sum1_1 = _mm256_dpbusd_epi32(v_sum1_1, v_x1, v_A1_1);
                v_sum2_1 = _mm256_dpbusd_epi32(v_sum2_1, v_x1, v_A2_1);
            }

            __m256i v_sum0 = _mm256_add_epi32(v_sum0_0, v_sum0_1);
            __m256i v_sum1 = _mm256_add_epi32(v_sum1_0, v_sum1_1);
            __m256i v_sum2 = _mm256_add_epi32(v_sum2_0, v_sum2_1);

            __m128i v128_0 = _mm_add_epi32(
                _mm256_castsi256_si128(v_sum0),
                _mm256_extracti128_si256(v_sum0, 1)
            );
            v128_0 = _mm_add_epi32(v128_0, _mm_shuffle_epi32(v128_0, _MM_SHUFFLE(1, 0, 3, 2)));
            v128_0 = _mm_add_epi32(v128_0, _mm_shuffle_epi32(v128_0, _MM_SHUFFLE(2, 3, 0, 1)));
            y[j] = _mm_cvtsi128_si32(v128_0);

            __m128i v128_1 = _mm_add_epi32(
                _mm256_castsi256_si128(v_sum1),
                _mm256_extracti128_si256(v_sum1, 1)
            );
            v128_1 = _mm_add_epi32(v128_1, _mm_shuffle_epi32(v128_1, _MM_SHUFFLE(1, 0, 3, 2)));
            v128_1 = _mm_add_epi32(v128_1, _mm_shuffle_epi32(v128_1, _MM_SHUFFLE(2, 3, 0, 1)));
            y[j + 1] = _mm_cvtsi128_si32(v128_1);

            __m128i v128_2 = _mm_add_epi32(
                _mm256_castsi256_si128(v_sum2),
                _mm256_extracti128_si256(v_sum2, 1)
            );
            v128_2 = _mm_add_epi32(v128_2, _mm_shuffle_epi32(v128_2, _MM_SHUFFLE(1, 0, 3, 2)));
            v128_2 = _mm_add_epi32(v128_2, _mm_shuffle_epi32(v128_2, _MM_SHUFFLE(2, 3, 0, 1)));
            y[j + 2] = _mm_cvtsi128_si32(v128_2);
        }

        for (size_t j = HIDDEN_SIZE - 2; j < HIDDEN_SIZE; ++j) { 
            __m256i v_sum0 = _mm256_setzero_si256();
            __m256i v_sum1 = _mm256_setzero_si256();
        
            for (size_t i = 0; i < ACCUMULATOR_SIZE; i += 64) {
                __m256i v_x0 = _mm256_load_si256((const __m256i*)&x[i]);
                __m256i v_x1 = _mm256_load_si256((const __m256i*)&x[i + 32]);
        
                __m256i v_A0 = _mm256_load_si256((const __m256i*)&w_T[j * ACCUMULATOR_SIZE + i]);
                __m256i v_A1 = _mm256_load_si256((const __m256i*)&w_T[j * ACCUMULATOR_SIZE + i + 32]);
        
                v_sum0 = _mm256_dpbusd_epi32(v_sum0, v_x0, v_A0);
                v_sum1 = _mm256_dpbusd_epi32(v_sum1, v_x1, v_A1);
            }
        
            __m256i v_sum = _mm256_add_epi32(v_sum0, v_sum1);
            __m128i v128 = _mm_add_epi32(
                _mm256_castsi256_si128(v_sum),
                _mm256_extracti128_si256(v_sum, 1)
            );
            v128 = _mm_add_epi32(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(1, 0, 3, 2)));
            v128 = _mm_add_epi32(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(2, 3, 0, 1)));
            y[j] = _mm_cvtsi128_si32(v128);
        }
    }

    void kernel_hidden_hidden(
        std::array<uint8_t, HIDDEN_SIZE>& x,
        std::array<int8_t, HIDDEN_SIZE * HIDDEN_SIZE>& w_T,
        std::array<uint8_t, HIDDEN_SIZE>& y
    ) {
        __m256i v_x = _mm256_load_si256((const __m256i*)&x);
        
        for (size_t j = 0; j < HIDDEN_SIZE; j += 2) {
            __m256i v_sum1 = _mm256_setzero_si256();
            __m256i v_sum2 = _mm256_setzero_si256();
            
            __m256i v_A1 = _mm256_load_si256((const __m256i*)&w_T[j * HIDDEN_SIZE]);
            __m256i v_A2 = _mm256_load_si256((const __m256i*)&w_T[(j + 1) * HIDDEN_SIZE]);

            v_sum1 = _mm256_dpbusd_epi32(v_sum1, v_x, v_A1);
            v_sum2 = _mm256_dpbusd_epi32(v_sum2, v_x, v_A2);

            __m128i v128_1 = _mm_add_epi32(
                _mm256_castsi256_si128(v_sum1),
                _mm256_extracti128_si256(v_sum1, 1)
            );
            v128_1 = _mm_add_epi32(v128_1, _mm_shuffle_epi32(v128_1, _MM_SHUFFLE(1, 0, 3, 2)));
            v128_1 = _mm_add_epi32(v128_1, _mm_shuffle_epi32(v128_1, _MM_SHUFFLE(2, 3, 0, 1)));
            y[j] = _mm_cvtsi128_si32(v128_1);

            __m128i v128_2 = _mm_add_epi32(
                _mm256_castsi256_si128(v_sum2),
                _mm256_extracti128_si256(v_sum2, 1)
            );
            v128_2 = _mm_add_epi32(v128_2, _mm_shuffle_epi32(v128_2, _MM_SHUFFLE(1, 0, 3, 2)));
            v128_2 = _mm_add_epi32(v128_2, _mm_shuffle_epi32(v128_2, _MM_SHUFFLE(2, 3, 0, 1)));
            y[j] = _mm_cvtsi128_si32(v128_2);
        }
    }

    void kernel_hidden_output(
        std::array<uint8_t, HIDDEN_SIZE>& x,
        std::array<int8_t, HIDDEN_SIZE * OUTPUT_SIZE>& w_T,
        std::array<uint8_t, OUTPUT_SIZE>& y
    ) {
        __m256i v_x = _mm256_load_si256((const __m256i*)&x);
        __m256i v_A = _mm256_load_si256((const __m256i*)&w_T);

        __m256i v_sum = _mm256_setzero_si256();
        v_sum = _mm256_dpbusd_epi32(v_sum, v_x, v_A);

        __m128i v128 = _mm_add_epi32(
            _mm256_castsi256_si128(v_sum),
            _mm256_extracti128_si256(v_sum, 1)
        );
        v128 = _mm_add_epi32(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(1, 0, 3, 2)));
        v128 = _mm_add_epi32(v128, _mm_shuffle_epi32(v128, _MM_SHUFFLE(2, 3, 0, 1)));
        y[0] = _mm_cvtsi128_si32(v128);
    }

    void kernel_accumulator_addition(
        std::array<int16_t, ACCUMULATOR_SIZE>& a,
        const std::span<int16_t, ACCUMULATOR_SIZE>& w
    ) {
        int16_t* __restrict ap = a.data();
        const int16_t* __restrict wp = w.data();

        for (size_t i = 0; i < ACCUMULATOR_SIZE; ++i) {
            ap[i] += wp[i];
        }
    }
}