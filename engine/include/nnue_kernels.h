#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <array>
#include <span>

namespace NNUE {
    constexpr size_t ACCUMULATOR_SIZE = 256;
    constexpr size_t MERGED_SIZE = 2 * ACCUMULATOR_SIZE;
    constexpr size_t HIDDEN_SIZE = 32;
    constexpr size_t OUTPUT_SIZE = 1;
    constexpr int32_t MIN = 0;
    constexpr int32_t MAX = 255;

    void kernel_accumulator_hidden(
        std::array<uint8_t, MERGED_SIZE>& x, 
        std::array<int8_t, MERGED_SIZE * HIDDEN_SIZE>& w_T,
        std::array<int32_t, HIDDEN_SIZE>& b,
        std::array<uint8_t, HIDDEN_SIZE>& y
    );

    void kernel_hidden_hidden(
        std::array<uint8_t, HIDDEN_SIZE>& x,
        std::array<int8_t, HIDDEN_SIZE * HIDDEN_SIZE>& w_T,
        std::array<int32_t, HIDDEN_SIZE>& b,
        std::array<uint8_t, HIDDEN_SIZE>& y
    );

    void kernel_hidden_output(
        std::array<uint8_t, HIDDEN_SIZE>& x,
        std::array<int8_t, HIDDEN_SIZE * OUTPUT_SIZE>& w_T,
        std::array<int32_t, OUTPUT_SIZE>& b,
        std::array<uint8_t, OUTPUT_SIZE>& y
    );

    void kernel_accumulator_addition(
        std::array<int16_t, ACCUMULATOR_SIZE>& a,
        const std::span<const int16_t, ACCUMULATOR_SIZE>& w
    );

    void kernel_accumulator_subtraction(
        std::array<int16_t, ACCUMULATOR_SIZE>& a,
        const std::span<const int16_t, ACCUMULATOR_SIZE>& w
    );

    void kernel_accumulator_clipconcat(
        std::array<int16_t, ACCUMULATOR_SIZE>& a,
        std::array<int16_t, ACCUMULATOR_SIZE>& b,
        std::array<uint8_t, MERGED_SIZE>& merged
    );
}