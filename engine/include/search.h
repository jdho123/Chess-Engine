#pragma once
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

enum class TTFlag : uint8_t {
    NONE  = 0,
    EXACT = 1,
    LOWER = 2,
    UPPER = 3
};

struct TTEntry {
    uint64_t key     = 0;
    int16_t score    = 0;
    chess::Move move = chess::Move::NO_MOVE;
    uint8_t depth    = 0;
    TTFlag flag      = TTFlag::NONE;
    uint8_t age      = 0;
    uint8_t pad      = 0;
};

class TranspositionTable {
public:
    explicit TranspositionTable(size_t size_mb);

    void resize(size_t size_mb);
    void clear();
    void new_search();
    size_t index(uint64_t key) const;
    bool probe(uint64_t zobrist_hash, TTEntry& out) const;
    void store(uint64_t zobrist_hash, int score, chess::Move move, int depth, TTFlag flag);

private:
    std::vector<TTEntry> table;
    size_t size = 0;
    size_t mask = 0;
    uint8_t age = 0;
};

constexpr int MAX_PLY = 64;

int search(chess::Board& board, NNUE::NNUE& nnue, int depth, int ply, int alpha, int beta, SearchContext& ctx, TranspositionTable& tt);

SearchResult find_best_move(chess::Board& board, NNUE::NNUE& nnue, SearchClock& clock, int max_depth, TranspositionTable& tt);

int search_root(chess::Board& board, NNUE::NNUE& nnue, int depth, int alpha, int beta, SearchContext& ctx, chess::Move& best_move, TranspositionTable& tt);

int quiescence_search(chess::Board& board, NNUE::NNUE& nnue, int ply, int alpha, int beta, SearchContext& ctx, TranspositionTable& tt);

int piece_value(chess::PieceType pt);

int score_move(const chess::Board& board, const chess::Move& move);

void order_moves(const chess::Board& board, chess::Movelist& moves, chess::Move& best_move);