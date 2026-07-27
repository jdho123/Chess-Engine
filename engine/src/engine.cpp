#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include "engine.h"
#include "search.h"
#include "chess.hpp"

UCIEngine::UCIEngine() : nnue(board) {}

void UCIEngine::handle_uci() const {
    std::cout << "id name Zombo 1.0" << std::endl;
    std::cout << "id author Johnny Ho" << std::endl;
    std::cout << "uciok" << std::endl;
}

void UCIEngine::handle_position(std::istringstream& iss) {
    if (search_thread.joinable()) {
        handle_stop();
    }

    std::string token;
    iss >> token;

    if (token == "startpos") {
        board = chess::Board();
        iss >> token;
    } else if (token == "fen") {
        std::string fen;
        std::string part;

        while (iss >> part && part != "moves") {
            if (!fen.empty()) {
                fen += " ";
            }
            fen += part;

        }
        board = chess::Board(fen);
        token = part;
    }

    if (token == "moves") {
        std::string move_str;
        while (iss >> move_str) {
            chess::Move m = chess::uci::uciToMove(board, move_str);
            board.makeMove(m);
        }
    }
}

void UCIEngine::handle_go(std::istringstream& iss) {
    if (search_thread.joinable()) {
        handle_stop();
    }

    TimeControl tc;
    std::string token;
    while (iss >> token) {
        if (token == "wtime") iss >> tc.wtime;
        else if (token == "btime") iss >> tc.btime;
        else if (token == "winc") iss >> tc.winc;
        else if (token == "binc") iss >> tc.binc;
        else if (token == "movestogo") iss >> tc.movestogo;
        else if (token == "movetime") iss >> tc.movetime;
        else if (token == "depth") iss >> tc.depth;
        else if (token == "infinite") tc.infinite = true;
    }

    int max_depth = (tc.depth > 0) ? tc.depth : 100;
    int64_t budget;
    if (tc.movetime > 0) {
        budget = tc.movetime;
    }
    else if(tc.infinite || (tc.wtime == 0 && tc.btime == 0)) {
        budget = INT64_MAX;
    } else {
        budget = computeTimeBudgetMs(tc, board.sideToMove());
    }

    clock.begin(budget);

    search_thread = std::thread([this, max_depth]() {
        SearchResult result = find_best_move(board, nnue, clock, max_depth);
        std::cout << "bestmove " << chess::uci::moveToUci(result.best_move) << std::endl;
    });
}

void UCIEngine::handle_stop() {
    clock.stop_flag = true;
    if (search_thread.joinable()) {
        search_thread.join();
    }
}

void UCIEngine::run() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "uci") {
            handle_uci();
        } else if (token == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (token == "ucinewgame") {
            board = chess::Board();
        } else if (token == "position") {
            handle_position(iss);
        } else if (token == "go") {
            handle_go(iss);
        } else if (token == "stop") {
            handle_stop();
        } else if (token == "quit") {
            handle_stop();
            break;
        }
    }
}