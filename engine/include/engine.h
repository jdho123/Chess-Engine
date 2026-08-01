#pragma once
#include <sstream>
#include <thread>
#include "chess.hpp"
#include "clock.h"
#include "nnue.h"
#include "search.h"

class UCIEngine {
public:
    UCIEngine();

    void run();

private:
    chess::Board board;
    NNUE::NNUE nnue;
    SearchClock clock;
    TranspositionTable tt{1024};
    std::thread search_thread;
    SearchContext ctx;
    
    void handle_uci() const;
    void handle_position(std::istringstream& iss);
    void handle_go(std::istringstream& iss);
    void handle_stop();
};