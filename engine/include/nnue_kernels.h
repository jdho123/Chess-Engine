#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <array>

namespace NNUEKernels {
    constexpr size_t ACCUMULATOR_SIZE = 512;
    constexpr size_t HIDDEN_SIZE = 32;
    constexpr size_t OUTPUT_SIZE = 1;

    void kernel_accumulator_hidden(
        std::array<uint8_t, ACCUMULATOR_SIZE>& x, 
        std::array<int8_t, ACCUMULATOR_SIZE * HIDDEN_SIZE>& w_T, 
        std::array<uint8_t, HIDDEN_SIZE>& y
    );

    void kernel_hidden_hidden(
        std::array<uint8_t, HIDDEN_SIZE>& x,
        std::array<int8_t, HIDDEN_SIZE * HIDDEN_SIZE>& w_T,
        std::array<uint8_t, HIDDEN_SIZE>& y
    );

    void kernel_hidden_output(
        std::array<uint8_t, HIDDEN_SIZE>& x,
        std::array<int8_t, HIDDEN_SIZE * OUTPUT_SIZE>& w_T,
        std::array<uint8_t, OUTPUT_SIZE>& y
    );
}