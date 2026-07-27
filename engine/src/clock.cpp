#include "clock.h"
#include <cstdint>

void SearchClock::begin(int64_t budget) {
    start = std::chrono::steady_clock::now();
    budget_ms = budget;
    stop_flag = false;
}

bool SearchClock::expired() const {
    if (stop_flag.load(std::memory_order_relaxed)) return true;
    auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() >= budget_ms;
}

int64_t computeTimeBudgetMs(const TimeControl& tc, chess::Color side_to_move) {
    if (tc.movetime > 0) {
        return tc.movetime;
    }

    int64_t my_time = (side_to_move == chess::Color::WHITE) ? tc.wtime : tc.btime;
    int64_t my_inc = (side_to_move == chess::Color::WHITE) ? tc.winc : tc.binc;

    int movestogo = (tc.movestogo > 0) ? tc.movestogo : 30;

    int64_t budget = (my_time / movestogo) + (my_inc * 3 / 4);

    int64_t safety_buffer = 50;
    budget -= safety_buffer;

    return std::max<int64_t>(budget, 20);
}