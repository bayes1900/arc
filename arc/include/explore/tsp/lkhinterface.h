#pragma once

#include <mutex>
#include <string>

extern "C" {
int solveMTSPWithLKH3(const char* inputFile);
}

namespace arc {

class LkhSolver {
public:
    static bool solveMtsp(const std::string& parFile) {
        std::lock_guard<std::mutex> lock(mutex());
        return solveMTSPWithLKH3(parFile.c_str()) == 0;
    }

private:
    static std::mutex& mutex() {
        static std::mutex instance;
        return instance;
    }
};

} // arc