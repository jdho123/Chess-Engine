#include "search.h"
#include "chess.hpp"
#include "nnue_constants.h"
#include <cstdint>

TranspositionTable::TranspositionTable(size_t size_mb) {
    resize(size_mb);
}

void TranspositionTable::resize(size_t size_mb) {
    size_t bytes = size_mb * 1024ull * 1024ull;
    size_t count = bytes / sizeof(TTEntry);

    size_t pow2 = 1;
    while (pow2 * 2 <= count) pow2 *= 2;

    age = 0;
    size = pow2;
    mask = pow2 - 1;
    table.assign(size, TTEntry{});
}

void TranspositionTable::clear() {
    std::fill(table.begin(), table.end(), TTEntry{});
    age = 0;
}

void TranspositionTable::new_search() {
    age++;
}

size_t TranspositionTable::index(uint64_t zobrist_hash) const {
    return zobrist_hash & mask;
}

bool TranspositionTable::probe(uint64_t zobrist_hash, TTEntry& out) const {
    const TTEntry& e = table[index(zobrist_hash)];
    if (e.key == zobrist_hash && e.flag != TTFlag::NONE) {
        out = e;
        return true;
    }
    return false;
}

void TranspositionTable::store(uint64_t zobrist_hash, int score, chess::Move move, int depth, TTFlag flag) {
    TTEntry& slot = table[index(zobrist_hash)];

    bool same_position = (slot.key == zobrist_hash);
    bool slot_empty     = (slot.flag == TTFlag::NONE);
    bool slot_stale     = (slot.age != age);

    bool should_replace = slot_empty || slot_stale || (depth >= slot.depth);

    if (!should_replace) return;

    if (move == chess::Move::NO_MOVE && same_position) {
        move = slot.move;
    }

    slot.key   = zobrist_hash;
    slot.score = static_cast<int16_t>(score);
    slot.move  = move;
    slot.depth = static_cast<uint8_t>(std::max(depth, 0));
    slot.flag  = flag;
    slot.age = age;
}

int search(
    chess::Board& board, 
    NNUE::NNUE& nnue, 
    int depth, 
    int ply, 
    int alpha, 
    int beta,  
    SearchContext& ctx,
    TranspositionTable& tt
) {
    ctx.nodes++;
    if ((ctx.nodes & 2047) == 0) {
        if (ctx.clock->expired()) {
            ctx.stopped = true;
        }
    }
    if (ctx.stopped) {
        return 0;
    }
    
    uint64_t zobrist_hash = board.zobrist();
    TTEntry entry;
    chess::Move best_move = chess::Move::NO_MOVE;

    if (tt.probe(zobrist_hash, entry)) {
        best_move = entry.move;
        if (entry.depth >= depth) {
            int tt_value = entry.score;
            if (abs(tt_value) >= NNUE::MATE_VALUE - MAX_PLY) {
                tt_value -= (tt_value > 0 ? ply : -ply);
            }
            if (entry.flag == TTFlag::EXACT) {
                return tt_value;
            } else if (entry.flag == TTFlag::LOWER && tt_value >= beta) {
                return tt_value;
            } else if (entry.flag == TTFlag::UPPER && tt_value <= alpha) {
                return tt_value;
            }
        }
    }

    if (ply >= MAX_PLY) {
        return quiescence_search(board, nnue, ply + 1, alpha, beta, ctx, tt);
    }

    bool in_check = board.inCheck();

    if (depth <= 0 && !in_check) {
        return quiescence_search(board, nnue, ply + 1, alpha, beta, ctx, tt);
    }

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    order_moves(board, moves, best_move);

    if (moves.empty()) {
        if (board.inCheck()) {
            return -NNUE::MATE_VALUE + ply;
        }
        return 0;
    }

    int best_value = -NNUE::MATE_VALUE - 1;
    int original_alpha = alpha;

    for (chess::Move m : moves) {
        nnue.make_move(board, m);
        int value = -search(
            board, 
            nnue, 
            depth - 1, 
            ply + 1, 
            -beta,
            -alpha,
            ctx,
            tt
        );
        nnue.unmake_move(board, m);

        if (ctx.stopped) {
            return best_value;
        }

        if (value > best_value) {
            best_value = value;
            best_move = m;
        }
        alpha = std::max(alpha, best_value);
        if (alpha >= beta) {
            break;
        }
    }

    TTFlag flag = (best_value <= original_alpha) ? TTFlag::UPPER
                : (best_value >= beta)           ? TTFlag::LOWER
                : TTFlag::EXACT;
    int store_score = best_value;
    if (abs(store_score) >= NNUE::MATE_VALUE - MAX_PLY) {
        store_score += (store_score > 0 ? ply : -ply);
    }
    tt.store(zobrist_hash, store_score, best_move, depth, flag);

    return best_value;
}

int search_root(
    chess::Board& board, 
    NNUE::NNUE& nnue, 
    int depth, 
    int alpha, 
    int beta, 
    SearchContext& ctx,
    chess::Move& best_move,
    TranspositionTable& tt
) {
    ctx.nodes++;
    if ((ctx.nodes & 2047) == 0) {
        if (ctx.clock->expired()) {
            ctx.stopped = true;
            return 0;
        }
    }
    if (ctx.stopped) {
        return 0;
    }

    uint64_t zobrist_hash = board.zobrist();
    TTEntry entry;

    if (tt.probe(zobrist_hash, entry)) {
        best_move = entry.move;
        if (entry.depth >= depth) {
            int tt_value = entry.score;
            if (entry.flag == TTFlag::EXACT) {
                return tt_value;
            } else if (entry.flag == TTFlag::LOWER && tt_value >= beta) {
                return tt_value;
            } else if (entry.flag == TTFlag::UPPER && tt_value <= alpha) {
                return tt_value;
            }
        }
    }

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (moves.empty()) {
        if (board.inCheck()) {
            return -NNUE::MATE_VALUE;
        }
        return 0;
    }

    order_moves(board, moves, best_move);

    int best_value = -NNUE::MATE_VALUE - 1;
    int original_alpha = alpha;

    for (chess::Move m : moves) {
        nnue.make_move(board, m);
        int value = -search(
            board,
            nnue,
            depth - 1,
            1,
            -beta,
            -alpha,
            ctx,
            tt
        );
        nnue.unmake_move(board, m);

        if (ctx.stopped) {
            break;
        }

        if (value > best_value) {
            best_value = value;
            best_move = m;
        }
        alpha = std::max(alpha, best_value);
        if (alpha >= beta) {
            break;
        }
    }

    if (ctx.stopped) {
        return best_value;
    }

    TTFlag flag = (best_value <= original_alpha) ? TTFlag::UPPER
                : (best_value >= beta)           ? TTFlag::LOWER
                : TTFlag::EXACT;
    tt.store(zobrist_hash, best_value, best_move, depth, flag);

    return best_value;
}

SearchResult find_best_move(chess::Board& board, NNUE::NNUE& nnue, SearchClock& clock, int max_depth, TranspositionTable& tt) {
    SearchResult result;
    result.best_move = chess::Move::NO_MOVE;
    int last_completed_score = 0;

    chess::Move best_move = chess::Move::NO_MOVE;

    for (int depth = 1; depth <= max_depth; depth++) {
        SearchContext ctx;
        ctx.clock = &clock;

        int window = 25;
        int alpha, beta;

        if (depth <= 4) {
            alpha = -NNUE::MATE_VALUE - 1;
            beta = NNUE::MATE_VALUE + 1;
        } else {
            alpha = last_completed_score - window;
            beta = last_completed_score + window;
        }

        int value;

        while (true) {
            value = search_root(board, nnue, depth, alpha, beta, ctx, best_move, tt);

            if (ctx.stopped) {
                break;
            }

            if (value <= alpha) {
                window *= 2;
                alpha = std::max(alpha - window, -NNUE::MATE_VALUE - 1);
                continue;
            }
            if (value >= beta) {
                window *= 2;
                beta = std::min(beta + window, NNUE::MATE_VALUE + 1);
                continue;
            }

            break;
        }

        if (ctx.stopped) {
            break;
        }

        result.best_move = best_move;
        result.score = value;
        result.depth_reached = depth;
        last_completed_score = value;

        std::cout << "info depth " << depth 
                  << " score cp "  << last_completed_score
                  << " nodes "     << ctx.nodes
                  << " pv " << chess::uci::moveToUci(best_move)
                  << std::endl;
        
        if (clock.expired()) {
            break;
        }
    }

    return result;
}

int quiescence_search(
    chess::Board& board, 
    NNUE::NNUE& nnue, 
    int ply, 
    int alpha, 
    int beta, 
    SearchContext& ctx, 
    TranspositionTable& tt
) {
    uint64_t zobrist_hash = board.zobrist();
    TTEntry entry;
    chess::Move tt_move = chess::Move::NO_MOVE;

    if (tt.probe(zobrist_hash, entry)) {
        tt_move = entry.move;
        int tt_value = entry.score;
        if (abs(tt_value) >= NNUE::MATE_VALUE - MAX_PLY) {
            tt_value -= (tt_value > 0 ? ply : -ply);
        }
        if (entry.flag == TTFlag::EXACT) {
            return tt_value;
        } else if (entry.flag == TTFlag::LOWER && tt_value >= beta) {
            return tt_value;
        } else if (entry.flag == TTFlag::UPPER && tt_value <= alpha) {
            return tt_value;
        }
    }

    ctx.nodes++;
    if ((ctx.nodes & 2047) == 0) {
        if (ctx.clock->expired()) {
            ctx.stopped = true;
        }
    }
    if (ctx.stopped) {
        return 0;
    }

    if (ply >= MAX_PLY) {
        return nnue.evaluate(board);
    }

    chess::Movelist moves;

    bool in_check = board.inCheck();

    int best_value;

    if (in_check) {
        chess::movegen::legalmoves(moves, board);

        if (moves.empty()) {
            return -NNUE::MATE_VALUE + ply;
        }

        best_value = -NNUE::MATE_VALUE - 1;
    }
    else {
        int static_score = nnue.evaluate(board);
    
        if (static_score >= beta) {
            return static_score;
        }
        if (static_score > alpha) {
            alpha = static_score;
        }

        best_value = static_score;

        chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(moves, board);
    }

    chess::Move best_move = chess::Move::NO_MOVE;
    order_moves(board, moves, tt_move);
    int original_alpha = alpha;

    for (chess::Move m : moves) {
        nnue.make_move(board, m);
        int value = -quiescence_search(board, nnue, ply + 1, -beta, -alpha, ctx, tt);
        nnue.unmake_move(board, m);

        if (ctx.stopped) {
            return best_value;
        }

        best_value = std::max(best_value, value);
        if (value >= beta) {
            return best_value;
        }
        if (value > alpha) {
            alpha = value;
        }
    }

    TTFlag flag = (best_value <= original_alpha) ? TTFlag::UPPER
                : (best_value >= beta)           ? TTFlag::LOWER
                : TTFlag::EXACT;
    int store_score = best_value;
    if (abs(store_score) >= NNUE::MATE_VALUE - MAX_PLY) {
        store_score += (store_score > 0 ? ply : -ply);
    }
    tt.store(zobrist_hash, store_score, chess::Move::NO_MOVE, 0, flag);

    return best_value;
}

int piece_value(chess::PieceType pt) {
    switch (pt.internal()) {
        case chess::PieceType::PAWN: return 100;
        case chess::PieceType::KNIGHT: return 320;
        case chess::PieceType::BISHOP: return 330;
        case chess::PieceType::ROOK: return 500;
        case chess::PieceType::QUEEN: return 900;
        case chess::PieceType::KING: return 20000;
        default: return 0;
    }
}

int score_move(const chess::Board& board, const chess::Move& move) {
    int score = 0;

    if (board.isCapture(move)) {
        chess::PieceType victim = board.at<chess::Piece>(move.to()).type();
        chess::PieceType attacker = board.at<chess::Piece>(move.from()).type();

        score += 10000 + piece_value(victim) * 10 - piece_value(attacker);
    }

    if (move.typeOf() == chess::Move::PROMOTION) {
        score += 9000 + piece_value(move.promotionType());
    }

    return score;
}

void order_moves(const chess::Board& board, chess::Movelist& moves, chess::Move& best_move) {
    std::sort(moves.begin(), moves.end(), [&board](const chess::Move& a, const chess::Move& b) {
        return score_move(board, a) > score_move(board, b);
    });

    if (best_move != chess::Move::NO_MOVE) {
        auto it = std::find(moves.begin(), moves.end(), best_move);
        if (it != moves.end()) {
            std::rotate(moves.begin(), it, it + 1);
        }
    }
}