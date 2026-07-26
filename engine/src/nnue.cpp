#include <bit>
#include "nnue.h"
#include "chess.hpp"
#include "nnue_weights.h"


NNUE::PieceUpdates NNUE::PieceUpdates::from_move(const chess::Board& board, chess::Move move) {
    PieceUpdates updates;

    const int from = move.from().index();
    const int to = move.to().index();
    const int color = static_cast<int>(board.sideToMove());
    
    auto piece_to_idx = [](chess::Piece p) -> int {
        int val = static_cast<int>(p);
        return val <= 5 ? val : val - 1;
    };
    
    chess::Piece moving_piece = board.at<chess::Piece>(from);
    const bool is_king = moving_piece.type() == chess::PieceType::KING;
    updates.king_move = is_king;
    updates.king_color = is_king ? color : -1;

    int moving_piece_idx = -1;
    if (!is_king) {
        moving_piece_idx = piece_to_idx(moving_piece);
        updates.removed[updates.removals++] = Feature{ moving_piece_idx, from };
    }

    switch (move.typeOf()) {
        case chess::Move::NORMAL: {
            chess::Piece target = board.at<chess::Piece>(to);
            if (target != chess::Piece::NONE) {
                updates.removed[updates.removals++] = { piece_to_idx(target), to };
            }
            if (!is_king) {
                updates.added[updates.additions++] = { moving_piece_idx, to };
            }
            break;
        }
        case chess::Move::PROMOTION: {
            chess::Piece target = board.at<chess::Piece>(to);
            if (target != chess::Piece::NONE) {
                updates.removed[updates.removals++] = { piece_to_idx(target), to };
            }
            const int promo_idx = static_cast<int>(move.promotionType()) + (color * 5);
            updates.added[updates.additions++] = { promo_idx, to };
            break;
        }
        case chess::Move::ENPASSANT: {
            const int cap_pawn_idx = (color ^ 1) * 5; 
            const int ep_sq = (from & 56) | (to & 7); 
            
            updates.removed[updates.removals++] = { cap_pawn_idx, ep_sq };
            updates.added[updates.additions++] = { moving_piece_idx, to };
            break;
        }
        case chess::Move::CASTLING: {
            const bool is_kingside = to > from;
            const int rank_offset = from & 56; 
            
            const int rook_from = rank_offset | (is_kingside ? 7 : 0); 
            const int rook_to = rank_offset | (is_kingside ? 5 : 3);   
            
            const int rook_idx = 3 + (color * 5); 

            updates.removed[updates.removals++] = { rook_idx, rook_from };
            updates.added[updates.additions++] = { rook_idx, rook_to };
            break;
        }
    }

    return updates;
}


void NNUE::Accumulator::refresh(const chess::Board &board, Color mode) {
    std::array<Feature, 32> features;
    int count = 0;

    chess::Bitboard occupied = board.occ();
    chess::Square wk_sq = board.kingSq(chess::Color::WHITE);
    chess::Square bk_sq = board.kingSq(chess::Color::BLACK);

    while (occupied) {
        chess::Square sq = occupied.pop();
        chess::Piece piece = board.at<chess::Piece>(sq);

        if (sq != wk_sq && sq != bk_sq) {
            int p_val = static_cast<int>(piece);
            int piece_idx = (p_val <= 5) ? p_val : p_val - 1;
            features[count++] = { piece_idx, sq.index() };
        }
    }

    if (mode == Color::WHITE || mode == Color::BOTH) refresh_half<0>(features, count, wk_sq);
    if (mode == Color::BLACK || mode == Color::BOTH) refresh_half<1>(features, count, bk_sq);
}


void NNUE::Accumulator::compute(const Accumulator& prev, Color mode) {
    if (mode == Color::WHITE || mode == Color::BOTH) compute_half<0>(prev);
    if (mode == Color::BLACK || mode == Color::BOTH) compute_half<1>(prev);
}


template <int C>
inline void NNUE::Accumulator::refresh_half(const std::array<Feature, 32>& features, int count, chess::Square k_sq) {
    constexpr bool IS_BLACK = (C == 1);
    
    auto& current_arr = IS_BLACK ? black : white;

    std::memcpy(current_arr.data(), ACCUMULATOR_BIAS.data(), ACCUMULATOR_SIZE * sizeof(int16_t));

    int k_offset = (IS_BLACK ? (k_sq.index() ^ 56) : k_sq.index()) * 640;

    for (int i = 0; i < count; ++i) {
        int p = IS_BLACK ? ((features[i].piece >= 5) ? (features[i].piece - 5) : (features[i].piece + 5)) : features[i].piece;
        int sq = IS_BLACK ? (features[i].sq ^ 56) : features[i].sq;
        int idx = k_offset + p * 64 + sq;

        const int16_t* ptr = ACCUMULATOR_WEIGHTS.data() + (idx * ACCUMULATOR_SIZE);
        kernel_accumulator_addition(current_arr, std::span<const int16_t, ACCUMULATOR_SIZE>(ptr, ACCUMULATOR_SIZE));
    }

    computed[C] = true;
    needs_refresh[C] = false;
    bucket[C] = k_sq.index();
}


template <int C>
inline void NNUE::Accumulator::compute_half(const Accumulator& prev) {
    constexpr bool IS_BLACK = (C == 1);
    
    auto& current_arr = IS_BLACK ? black : white;
    const auto& prev_arr = IS_BLACK ? prev.black : prev.white;

    std::memcpy(current_arr.data(), prev_arr.data(), sizeof(current_arr));
    bucket[C] = prev.bucket[C];

    int king_offset = (IS_BLACK ? (bucket[C] ^ 56) : bucket[C]) * 640;

    for (int r = 0; r < updates.removals; ++r) {
        Feature f = updates.removed[r];
        int p = IS_BLACK ? ((f.piece >= 5) ? (f.piece - 5) : (f.piece + 5)) : f.piece;
        int sq = IS_BLACK ? (f.sq ^ 56) : f.sq;
        int weights_idx = king_offset + p * 64 + sq;
        
        const int16_t* ptr = ACCUMULATOR_WEIGHTS.data() + (weights_idx * ACCUMULATOR_SIZE);
        kernel_accumulator_subtraction(current_arr, std::span<const int16_t, ACCUMULATOR_SIZE>(ptr, ACCUMULATOR_SIZE));
    }

    for (int a = 0; a < updates.additions; ++a) {
        Feature f = updates.added[a];
        int p = IS_BLACK ? ((f.piece >= 5) ? (f.piece - 5) : (f.piece + 5)) : f.piece;
        int sq = IS_BLACK ? (f.sq ^ 56) : f.sq;
        int weights_idx = king_offset + p * 64 + sq;
        
        const int16_t* ptr = ACCUMULATOR_WEIGHTS.data() + (weights_idx * ACCUMULATOR_SIZE);
        kernel_accumulator_addition(current_arr, std::span<const int16_t, ACCUMULATOR_SIZE>(ptr, ACCUMULATOR_SIZE));
    }
    
    computed[C] = true;
}


template void NNUE::Accumulator::refresh_half<0>(const std::array<Feature, 32>&, int, chess::Square);
template void NNUE::Accumulator::refresh_half<1>(const std::array<Feature, 32>&, int, chess::Square);

template void NNUE::Accumulator::compute_half<0>(const Accumulator&);
template void NNUE::Accumulator::compute_half<1>(const Accumulator&);


NNUE::NNUE::NNUE(const chess::Board& board) {
    accumulator_stack[ply++].refresh(board, Accumulator::Color::BOTH);
}


void NNUE::NNUE::make_move(chess::Board& board, chess::Move move) {
    PieceUpdates updates = PieceUpdates::from_move(board, move);

    accumulator_stack[ply].updates = updates;
    accumulator_stack[ply].computed[0] = false;
    accumulator_stack[ply].computed[1] = false;

    if (updates.king_move) {
        accumulator_stack[ply].needs_refresh[0] = true;
        accumulator_stack[ply].needs_refresh[1] = true;
    }

    ply++;
    board.makeMove(move);
}


void NNUE::NNUE::unmake_move(chess::Board& board, chess::Move move) {
    ply--;
    board.unmakeMove(move);
}


void NNUE::NNUE::compute_accumulators(const chess::Board& board) {
    int base_ply_w = -1, base_ply_b = -1;
    bool refresh_w = false, refresh_b = false;
    bool searching_w = true, searching_b = true;

    int current_ply = ply - 1;

    // Traceback Phase
    while (searching_w | searching_b) {
        const auto& acc = accumulator_stack[current_ply];

        // White Track
        if (searching_w) {
            if (acc.computed[0]) {
                base_ply_w = current_ply;
                searching_w = false;
            }
            else if (acc.needs_refresh[0]) {
                refresh_w = true;
                searching_w = false;
            }
        }

        // Black Track
        if (searching_b) {
            if (acc.computed[1]) {
                base_ply_b = current_ply;
                searching_b = false;
            }
            else if (acc.needs_refresh[1]) {
                refresh_b = true;
                searching_b = false;
            }
        }

        current_ply--;
    }

    // Execution Phase
    
    // White Execution
    if (refresh_w) {
        accumulator_stack[ply - 1].refresh(board, Accumulator::Color::WHITE);
    } else {
        for (int i = base_ply_w + 1; i < ply; ++i) {
            accumulator_stack[i].compute(accumulator_stack[i - 1], Accumulator::Color::WHITE);
        }
    }

    // Black Execution
    if (refresh_b) {
        accumulator_stack[ply - 1].refresh(board, Accumulator::Color::BLACK);
    } else {
        for (int i = base_ply_b + 1; i < ply; ++i) {
            accumulator_stack[i].compute(accumulator_stack[i - 1], Accumulator::Color::BLACK);
        }
    }
}

int NNUE::NNUE::evaluate(const chess::Board& board) {
    compute_accumulators(board);

    Accumulator current_acc = accumulator_stack[ply - 1];

    if (board.sideToMove() == chess::Color::WHITE) {
        kernel_accumulator_clipconcat(current_acc.white, current_acc.black, l1);
    } else {
        kernel_accumulator_clipconcat(current_acc.black, current_acc.white, l1);
    }

    kernel_accumulator_hidden(l1, HIDDEN_WEIGHTS, HIDDEN_BIAS, l2);

    int bucket_idx = get_bucket_idx(board);

    const int8_t* weights_ptr = BUCKETED_WEIGHTS.data() + (bucket_idx * HIDDEN_SIZE * HIDDEN_SIZE);
    auto bucket_weights = std::span<const int8_t, HIDDEN_SIZE * HIDDEN_SIZE>(weights_ptr, HIDDEN_SIZE * HIDDEN_SIZE);

    const int32_t* bias_ptr = BUCKETED_BIAS.data() + (bucket_idx * HIDDEN_SIZE);
    auto bucket_bias = std::span<const int32_t, HIDDEN_SIZE>(bias_ptr, HIDDEN_SIZE);

    kernel_hidden_hidden(l2, bucket_weights, bucket_bias, l3);

    kernel_hidden_output(l3, OUTPUT_WEIGHTS, OUTPUT_BIAS, out);

    return out[0] / OUTPUT_SCALE;
}

int NNUE::NNUE::get_bucket_idx(const chess::Board& board) {
    static constexpr std::array<int, 6> PIECE_VALUES = {1, 3, 3, 5, 9, 0};
    static constexpr std::array<chess::PieceType, 6> PIECE_TYPES = {
        chess::PieceType::PAWN,
        chess::PieceType::KNIGHT,
        chess::PieceType::BISHOP,
        chess::PieceType::ROOK,
        chess::PieceType::QUEEN,
        chess::PieceType::KING
    };

    int total = 0;
    for (int pt = 0; pt < 6; ++pt) {
        chess::PieceType type = PIECE_TYPES[pt];
        uint64_t bb = board.pieces(type).getBits();
        total += PIECE_VALUES[pt] * std::popcount(bb);
    }

    constexpr int NORMAL_BUCKETS = NUM_BUCKETS - 1;

    if (total > MAX_MATERIAL) {
        return NORMAL_BUCKETS;
    }

    const int numerator = (total - MIN_MATERIAL) * NORMAL_BUCKETS;
    constexpr int denominator = MAX_MATERIAL - MIN_MATERIAL;

    int standard_idx = std::clamp(numerator / denominator, 0, NORMAL_BUCKETS - 1);
    
    return standard_idx;
}