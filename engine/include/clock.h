#pragma once
#include <atomic>
#include <chrono>
#include "chess.hpp"

struct SearchClock {
    std::chrono::steady_clock::time_point start;
    int64_t budget_ms;
    std::atomic<bool> stop_flag{false};

    void begin(int64_t budget);
    bool expired() const;
};

struct TimeControl {
    int64_t wtime = 0, btime = 0, winc = 0, binc = 0;
    int movestogo = 0;
    int movetime = 0;
    int depth = 0;
    bool infinite = false;
};

int64_t computeTimeBudgetMs(const TimeControl& tc, chess::Color side_to_move);