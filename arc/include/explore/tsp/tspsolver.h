#pragma once

#include <base/iarc.h>
#include <explore/tsp/lkhinterface.h>
#include <explore/tsp/tspdef.h>
#include <explore/tsp/tsplib.h>

#include <functional>

namespace arc {

class TspSolver {
public:
    using Solver = std::function<TspSolveResult(const std::string&, bool)>;

    TspSolver(std::string workDir, std::string name, Solver solver = {})
        : workDir_(std::move(workDir))
        , name_(std::move(name))
        , solver_(std::move(solver))
    {
        std::filesystem::create_directories(workDir_);
    }
    bool solve(const TspProblem& problem, TspSolution& solution) {
        return solveWithDiagnostics(problem, solution).success;
    }
    TspSolveResult solveWithDiagnostics(const TspProblem& problem, TspSolution& solution) {
        const auto start = std::chrono::steady_clock::now();
        double writeSeconds = 0.0;
        double serviceSeconds = 0.0;
        double readSeconds = 0.0;
        auto finish = [&](bool success, TspFailure failure) {
            TspSolveResult result;
            result.success = success;
            result.failure = failure;
            result.seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            result.writeSeconds = writeSeconds;
            result.serviceSeconds = serviceSeconds;
            result.readSeconds = readSeconds;
            return result;
        };

        if (problem.cost.rows() <= 0 || problem.cost.rows() != problem.cost.cols()) {
            return finish(false, TspFailure::INVALID_PROBLEM);
        }
        const bool acvrp = problem.acvrp;
        const std::string baseName = name_ + (acvrp ? "_acvrp" : "_tsp");
        const std::string problemPath = path_(baseName, ".atsp");
        const std::string parPath = path_(baseName, ".par");
        const std::string tourPath = path_(baseName, ".tour");
        const std::string solutionPath = path_(baseName, ".sol");
        const auto writeStart = std::chrono::steady_clock::now();
        if (!TsplibWriter::writeProblem(problem, problemPath)) {
            return finish(false, TspFailure::WRITE_PROBLEM_FAILED);
        }
        if (!TsplibWriter::writeParameters(problem, parPath, problemPath, tourPath, solutionPath)) {
            return finish(false, TspFailure::WRITE_PARAMS_FAILED);
        }
        writeSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - writeStart).count();
        std::filesystem::remove(tourPath);
        std::filesystem::remove(solutionPath);
        const auto serviceStart = std::chrono::steady_clock::now();
        if (solver_) {
            const TspSolveResult solverResult = solver_(parPath, acvrp);
            serviceSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - serviceStart).count();
            if (!solverResult.success) {
                return finish(false, solverResult.failure);
            }
        } else if (!LkhSolver::solveMtsp(parPath)) {
            serviceSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - serviceStart).count();
            return finish(false, TspFailure::SOLVER_FAILED);
        } else {
            serviceSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - serviceStart).count();
        }
        solution.tour.clear();
        solution.tours.clear();
        solution.feasible = true;
        const auto readStart = std::chrono::steady_clock::now();
        auto finishRead = [&](bool success, TspFailure failure) {
            readSeconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - readStart).count();
            return finish(success, failure);
        };
        if (problem.vehicles <= 1) {
            std::vector<int> ids;
            if (!TsplibReader::readTour(tourPath, ids)) {
                return finishRead(false, TspFailure::READ_TOUR_FAILED);
            }
            if (ids.empty()) {
                return finishRead(false, TspFailure::EMPTY_TOUR);
            }
            solution.tour = std::move(ids);
            return finishRead(true, TspFailure::NONE);
        }
        if (acvrp) {
            std::vector<std::vector<int>> tours;
            if (!TsplibReader::readAcvrpTours(tourPath, static_cast<int>(problem.cost.rows()), problem.vehicles, tours)) {
                return finishRead(false, TspFailure::READ_TOUR_FAILED);
            }
            if (tours.empty()) {
                return finishRead(false, TspFailure::EMPTY_TOUR);
            }
            solution.tours = std::move(tours);
            solution.feasible = true;
            return finishRead(true, TspFailure::NONE);
        }
        std::vector<int> ids;
        if (!TsplibReader::readTour(tourPath, ids)) {
            return finishRead(false, TspFailure::READ_TOUR_FAILED);
        }
        if (ids.empty()) {
            return finishRead(false, TspFailure::EMPTY_TOUR);
        }
        solution.tours = {ids};
        return finishRead(true, TspFailure::NONE);
    }

private:
    std::string path_(const std::string& baseName, const std::string& suffix) const {
        return (std::filesystem::path(workDir_) / (baseName + suffix)).string();
    }

private:
    std::string workDir_;
    std::string name_;
    Solver solver_;
};

} // arc
