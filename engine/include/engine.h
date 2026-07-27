#pragma once
#include <sstream>
#include <thread>
#include "chess.hpp"
#include "clock.h"
#include "nnue.h"

class UCIEngine {
public:
    UCIEngine();

    void run();

private:
    chess::Board board;
    NNUE::NNUE nnue;
    SearchClock clock;
    std::thread search_thread;

    void handle_uci() const;
    void handle_position(std::istringstream& iss);
    void handle_go(std::istringstream& iss);
    void handle_stop();
};