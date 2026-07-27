#include "search.h"
#include "chess.hpp"
#include "nnue_constants.h"

int search(
    chess::Board& board, 
    NNUE::NNUE& nnue, 
    int depth, 
    int ply, 
    int alpha, 
    int beta,  
    SearchContext& ctx
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

    if (depth == 0) {
        return quiescence_search(board, nnue, ply + 1, alpha, beta, ctx);
    }

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    order_moves(board, moves);

    if (moves.empty()) {
        if (board.inCheck()) {
            return -NNUE::MATE_VALUE + ply;
        }
        return 0;
    }

    int best_value = -NNUE::MATE_VALUE - 1;

    for (chess::Move m : moves) {
        nnue.make_move(board, m);
        int value = -search(
            board, 
            nnue, 
            depth - 1, 
            ply + 1, 
            -beta,
            -alpha,
            ctx
        );
        nnue.unmake_move(board, m);

        if (ctx.stopped) {
            return best_value;
        }

        best_value = std::max(best_value, value);
        alpha = std::max(alpha, best_value);
        if (alpha >= beta) {
            break;
        }
    }
    return best_value;
}

int search_root(chess::Board& board, NNUE::NNUE& nnue, int depth, int alpha, int beta, SearchContext& ctx, chess::Move& best_move) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    if (moves.empty()) {
        if (board.inCheck()) {
            return -NNUE::MATE_VALUE;
        }
        return 0;
    }

    order_moves(board, moves);

    if (best_move != chess::Move::NO_MOVE) {
        auto it = std::find(moves.begin(), moves.end(), best_move);
        if (it != moves.end()) {
            std::rotate(moves.begin(), it, it + 1);
        }
    }

    best_move = moves[0];
    int best_value = -NNUE::MATE_VALUE - 1;

    for (chess::Move m : moves) {
        nnue.make_move(board, m);
        int value = -search(
            board,
            nnue,
            depth - 1,
            1,
            -beta,
            -alpha,
            ctx
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
    return best_value;
}

SearchResult find_best_move(chess::Board& board, NNUE::NNUE& nnue, SearchClock& clock, int max_depth) {
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

        int score;

        while (true) {
            score = search_root(board, nnue, depth, alpha, beta, ctx, best_move);

            if (ctx.stopped) {
                break;
            }

            if (score <= alpha) {
                window *= 2;
                alpha = std::max(alpha - window, -NNUE::MATE_VALUE - 1);
                continue;
            }
            if (score >= beta) {
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
        result.score = score;
        result.depth_reached = depth;
        last_completed_score = score;

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

int quiescence_search(chess::Board& board, NNUE::NNUE& nnue, int ply, int alpha, int beta, SearchContext& ctx) {
    constexpr int MAX_PLY = 64;

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

    if (in_check) {
        chess::movegen::legalmoves(moves, board);

        if (moves.empty()) {
            return -NNUE::MATE_VALUE + ply;
        }
    }
    else {
        int static_score = nnue.evaluate(board);
    
        if (static_score >= beta) {
            return beta;
        }
        if (static_score > alpha) {
            alpha = static_score;
        }

        chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(moves, board);
    }

    order_moves(board, moves);

    for (chess::Move m : moves) {
        nnue.make_move(board, m);
        int value = -quiescence_search(board, nnue, ply + 1, -beta, -alpha, ctx);
        nnue.unmake_move(board, m);

        if (ctx.stopped) {
            return alpha;
        }

        if (value >= beta) {
            return beta;
        }
        if (value > alpha) {
            alpha = value;
        }
    }

    return alpha;
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

void order_moves(const chess::Board& board, chess::Movelist& moves) {
    std::sort(moves.begin(), moves.end(), [&board](const chess::Move& a, const chess::Move& b) {
        return score_move(board, a) > score_move(board, b);
    });
}