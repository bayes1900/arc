#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>

#include <base/iarc.h>
#include <explore/tsp/tsplib.h>
#include <explore/tsp/tspsolver.h>

namespace {

class TspTest : public ::testing::Test {
protected:
    void SetUp() override {
        workDir_ = (std::filesystem::temp_directory_path() / "arc_tsp_test").string();
        std::filesystem::remove_all(workDir_);
        std::filesystem::create_directories(workDir_);
    }
    void TearDown() override {
        std::filesystem::remove_all(workDir_);
    }

    double tourCost(const Eigen::MatrixXd& cost, const std::vector<int>& tour) {
        if (tour.size() < 2) {
            return 0.0;
        }
        double total = 0.0;
        for (size_t i = 0; i + 1 < tour.size(); ++i) {
            total += cost(tour[i], tour[i + 1]);
        }
        total += cost(tour.back(), tour.front());
        return total;
    }

    std::string workDir_;
};

TEST_F(TspTest, ReportsInvalidProblem) {
    arc::TspProblem problem;
    problem.cost.resize(0, 0);
    arc::TspSolution solution;
    arc::TspSolver solver(workDir_, "invalid");
    const arc::TspSolveResult result = solver.solveWithDiagnostics(problem, solution);
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.failure, arc::TspFailure::INVALID_PROBLEM);
    EXPECT_GE(result.seconds, 0.0);
}

TEST_F(TspTest, SolvesFourNodeAtsp) {
    arc::TspProblem problem;
    problem.cost.resize(4, 4);
    problem.cost << 0.0, 1.0, 2.0, 3.0,
                    1.0, 0.0, 1.0, 2.0,
                    2.0, 1.0, 0.0, 1.0,
                    3.0, 2.0, 1.0, 0.0;
    problem.vehicles = 1;

    arc::TspSolver solver(workDir_, "four");
    arc::TspSolution solution;
    ASSERT_TRUE(solver.solve(problem, solution));
    ASSERT_FALSE(solution.tour.empty());
    EXPECT_EQ(solution.tour.size(), 4u);
    EXPECT_NEAR(tourCost(problem.cost, solution.tour), 6.0, 1e-6);

    std::vector<int> sorted = solution.tour;
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(sorted[i], i);
    }
}

TEST_F(TspTest, SolvesSixNodeAtspAgainstBruteForce) {
    Eigen::MatrixXd cost(6, 6);
    cost << 0.0, 7.0, 2.0, 9.0, 1.0, 4.0,
            7.0, 0.0, 3.0, 2.0, 8.0, 5.0,
            2.0, 3.0, 0.0, 6.0, 3.0, 7.0,
            9.0, 2.0, 6.0, 0.0, 5.0, 1.0,
            1.0, 8.0, 3.0, 5.0, 0.0, 2.0,
            4.0, 5.0, 7.0, 1.0, 2.0, 0.0;

    std::vector<int> permutation = {1, 2, 3, 4, 5};
    double best = std::numeric_limits<double>::max();
    do {
        std::vector<int> tour = {0};
        tour.insert(tour.end(), permutation.begin(), permutation.end());
        best = std::min(best, tourCost(cost, tour));
    } while (std::next_permutation(permutation.begin(), permutation.end()));

    arc::TspProblem problem;
    problem.cost = cost;
    problem.vehicles = 1;

    arc::TspSolver solver(workDir_, "six");
    arc::TspSolution solution;
    ASSERT_TRUE(solver.solve(problem, solution));
    EXPECT_NEAR(tourCost(cost, solution.tour), best, 1e-6);
}

TEST_F(TspTest, SolvesAcvrpWithCapacity) {
    arc::TspProblem problem;
    problem.cost.resize(5, 5);
    problem.cost << 0.0, 2.0, 3.0, 2.0, 3.0,
                    2.0, 0.0, 1.0, 4.0, 5.0,
                    3.0, 1.0, 0.0, 3.0, 4.0,
                    2.0, 4.0, 3.0, 0.0, 1.0,
                    3.0, 5.0, 4.0, 1.0, 0.0;
    problem.vehicles = 2;
    problem.acvrp = true;
    problem.capacity = 10;
    problem.demands = {0, 4, 4, 4, 4};

    arc::TspSolver solver(workDir_, "acvrp");
    arc::TspSolution solution;
    ASSERT_TRUE(solver.solve(problem, solution));
    ASSERT_FALSE(solution.tours.empty());
    std::vector<int> visited(5, 0);
    for (const auto& tour : solution.tours) {
        int load = 0;
        for (int id : tour) {
            ASSERT_GE(id, 0);
            ASSERT_LT(id, 5);
            visited[id] += 1;
            load += problem.demands[id];
        }
        EXPECT_LE(load, problem.capacity);
    }
    for (int i = 1 + problem.vehicles; i < 5; ++i) {
        EXPECT_GE(visited[i], 1);
    }
}

TEST_F(TspTest, WritesTsplibFiles) {
    arc::TspProblem problem;
    problem.cost = Eigen::MatrixXd::Zero(3, 3);
    problem.vehicles = 1;
    const std::string problemPath = workDir_ + "/manual.atsp";
    const std::string parPath = workDir_ + "/manual.par";
    const std::string tourPath = workDir_ + "/manual.tour";
    ASSERT_TRUE(arc::TsplibWriter::writeProblem(problem, problemPath));
    ASSERT_TRUE(arc::TsplibWriter::writeParameters(problem, parPath, problemPath, tourPath));
    EXPECT_TRUE(std::filesystem::exists(problemPath));
    EXPECT_TRUE(std::filesystem::exists(parPath));

    std::ofstream tour(tourPath);
    tour << "TOUR_SECTION\n1\n2\n3\n-1\n";
    tour.close();
    std::vector<int> ids;
    ASSERT_TRUE(arc::TsplibReader::readTour(tourPath, ids));
    ASSERT_EQ(ids.size(), 3u);
    EXPECT_EQ(ids[0], 0);
    EXPECT_EQ(ids[1], 1);
    EXPECT_EQ(ids[2], 2);
}

TEST_F(TspTest, ReadsAcvrpToursByDepotCopy) {
    const std::string tourPath = workDir_ + "/acvrp_order.tour";
    std::ofstream file(tourPath);
    file << "TOUR_SECTION\n1\n4\n5\n6\n3\n2\n-1\n";
    file.close();

    std::vector<std::vector<int>> tours;
    ASSERT_TRUE(arc::TsplibReader::readAcvrpTours(tourPath, 5, 2, tours));
    ASSERT_EQ(tours.size(), 2u);
    ASSERT_EQ(tours[0].size(), 2u);
    EXPECT_EQ(tours[0][0], 3);
    EXPECT_EQ(tours[0][1], 4);
    ASSERT_EQ(tours[1].size(), 2u);
    EXPECT_EQ(tours[1][0], 2);
    EXPECT_EQ(tours[1][1], 1);
}

} // namespace