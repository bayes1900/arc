#pragma once

#include <base/iarc.h>
#include <explore/tsp/tspdef.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

namespace arc {

class TsplibWriter {
public:
    static bool writeProblem(const TspProblem& problem, const std::string& path) {
        if (problem.cost.rows() <= 0 || problem.cost.rows() != problem.cost.cols()) {
            return false;
        }
        std::ofstream file(path);
        if (!file.is_open()) {
            return false;
        }
        const int dimension = static_cast<int>(problem.cost.rows());
        file << "NAME : arc\n";
        file << "TYPE : " << (problem.acvrp ? "ACVRP" : "ATSP") << "\n";
        file << "DIMENSION : " << dimension << "\n";
        file << "EDGE_WEIGHT_TYPE : EXPLICIT\n";
        file << "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n";
        if (problem.acvrp) {
            file << "CAPACITY : " << problem.capacity << "\n";
            file << "VEHICLES : " << problem.vehicles << "\n";
        }
        file << "EDGE_WEIGHT_SECTION\n";
        for (int i = 0; i < dimension; ++i) {
            for (int j = 0; j < dimension; ++j) {
                file << std::lround(problem.cost(i, j) * 100.0) << " ";
            }
            file << "\n";
        }
        if (problem.acvrp) {
            file << "DEMAND_SECTION\n";
            const int demandCount = static_cast<int>(problem.demands.size());
            for (int i = 0; i < dimension; ++i) {
                const int demand = i < demandCount ? problem.demands[i] : 0;
                file << (i + 1) << " " << demand << "\n";
            }
            file << "DEPOT_SECTION\n";
            file << (problem.depot + 1) << "\n";
            file << "-1\n";
        }
        file << "EOF\n";
        return file.good();
    }

    static bool writeParameters(
        const TspProblem& problem,
        const std::string& parPath,
        const std::string& problemPath,
        const std::string& tourPath,
        const std::string& solutionPath = ""
    ) {
        std::ofstream file(parPath);
        if (!file.is_open()) {
            return false;
        }
        file << "SPECIAL\n";
        file << "PROBLEM_FILE = " << std::filesystem::absolute(problemPath).string() << "\n";
        if (problem.acvrp) {
            if (!solutionPath.empty()) {
                file << "MTSP_SOLUTION_FILE = " << std::filesystem::absolute(solutionPath).string() << "\n";
            }
            file << "TRACE_LEVEL = 0\n";
            file << "SEED = 0\n";
        } else {
            file << "SALESMEN = " << problem.vehicles << "\n";
            file << "MTSP_OBJECTIVE = MINSUM\n";
            if (problem.minSize >= 0) {
                file << "MTSP_MIN_SIZE = " << problem.minSize << "\n";
            }
            if (problem.maxSize >= 0) {
                file << "MTSP_MAX_SIZE = " << problem.maxSize << "\n";
            }
            file << "TRACE_LEVEL = 0\n";
        }
        file << "RUNS = 1\n";
        file << "TIME_LIMIT = 0.3\n";
        file << "MAX_TRIALS = 200\n";
        file << "TOUR_FILE = " << std::filesystem::absolute(tourPath).string() << "\n";
        return file.good();
    }
};

class TsplibReader {
public:
    static bool readTour(const std::string& path, std::vector<int>& ids) {
        std::ifstream file(path);
        if (!file.is_open()) {
            return false;
        }
        std::string line;
        bool found = false;
        while (std::getline(file, line)) {
            if (line == "TOUR_SECTION") {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
        ids.clear();
        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }
            int id = 0;
            try {
                id = std::stoi(line);
            } catch (const std::exception&) {
                break;
            }
            if (id == -1) {
                break;
            }
            ids.push_back(id - 1);
        }
        return !ids.empty();
    }
    static bool readAcvrpTours(
        const std::string& path,
        int dimension,
        int vehicles,
        std::vector<std::vector<int>>& tours
    ) {
        std::vector<int> ids;
        if (!readTour(path, ids)) {
            return false;
        }
        tours.assign(static_cast<size_t>(vehicles), {});
        int vehicle = 0;
        for (int id : ids) {
            if (id == 0) {
                vehicle = 0;
                continue;
            }
            if (id >= dimension) {
                vehicle = id - dimension + 1;
                continue;
            }
            if (vehicle >= 0 && vehicle < vehicles) {
                tours[static_cast<size_t>(vehicle)].push_back(id);
            }
        }
        return true;
    }
    };

} // arc