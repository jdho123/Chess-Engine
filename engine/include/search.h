#include "chess.hpp"
#include "clock.h"
#include "nnue.h"

struct SearchContext {
    SearchClock* clock;
    uint64_t nodes = 0;
    bool stopped = false;
};

struct SearchResult {
    chess::Move best_move;
    int score;
    int depth_reached;
};

int search(chess::Board& board, NNUE::NNUE& nnue, int depth, int root_depth, int alpha, int beta, bool maximizing_player);

SearchResult find_best_move(chess::Board& board, NNUE::NNUE& nnue, SearchClock& clock, int max_depth);

int quiescence_search(chess::Board& board, NNUE::NNUE& nnue, int ply, int alpha, int beta, SearchContext& ctx);

int piece_value(chess::PieceType pt);

int score_move(const chess::Board& board, const chess::Move& move);

void order_moves(const chess::Board& board, chess::Movelist& moves);