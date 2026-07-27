#pragma once
#include <array>
#include <cstdint>
#include "chess.hpp"
#include "nnue_kernels.h"

namespace NNUE {
    struct Feature {
        int piece;
        int sq;
    };

    struct PieceUpdates {
        std::array<Feature, 2> added;
        std::array<Feature, 2> removed;
        int additions = 0;
        int removals = 0;
        bool king_move = false;
        int king_color = -1; // white = 0, black = 1

        static PieceUpdates from_move(const chess::Board& board, chess::Move move);
    };

    struct alignas(64) Accumulator {
        enum class Color : uint8_t { WHITE, BLACK, BOTH };

        alignas(64) std::array<int16_t, ACCUMULATOR_SIZE> white;
        alignas(64) std::array<int16_t, ACCUMULATOR_SIZE> black;
        int bucket[2] = { -1, -1 };
        bool computed[2] = { false, false };
        bool needs_refresh[2] = { false, false };
        PieceUpdates updates;

        void refresh(const chess::Board& board, Color mode);
        void compute(const Accumulator& prev, Color mode);

        template <int C>
        inline void refresh_half(const std::array<Feature, 32>& features, int count, chess::Square k_sq);

        template <int C>
        inline void compute_half(const Accumulator& prev);
    };

    class NNUE {
    public:
        explicit NNUE(const chess::Board& board);
        
        NNUE(const NNUE&)            = delete;
        NNUE& operator=(const NNUE&) = delete;
        NNUE(NNUE&&)                 = delete;
        NNUE& operator=(NNUE&&)      = delete;

        void make_move(chess::Board& board, chess::Move move);
        void unmake_move(chess::Board& board, chess::Move move);
        int evaluate(const chess::Board& board);
        void reset(const chess::Board& board);
    
    private:
        static constexpr int MAX_PLY = 256;
        std::array<Accumulator, MAX_PLY> accumulator_stack;
        int ply = 0;

        alignas(64) std::array<uint8_t, MERGED_SIZE> l1;
        alignas(64) std::array<uint8_t, HIDDEN_SIZE> l2;
        alignas(64) std::array<uint8_t, HIDDEN_SIZE> l3;
        alignas(64) std::array<int32_t, OUTPUT_SIZE> out;

        void compute_accumulators(const chess::Board& board);
        int get_bucket_idx(const chess::Board& board);
    };
}