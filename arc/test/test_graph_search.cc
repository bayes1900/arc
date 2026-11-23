#include <gtest/gtest.h>

#include <base/iarc.h>
#include <search/graphsearch.h>

namespace {

struct MockNode {
    std::vector<MockNode*> neighbors;
    std::unordered_map<const MockNode*, double> edgeCosts;

    double costTo(const MockNode& node) const {
        const auto iter = edgeCosts.find(&node);
        return iter == edgeCosts.end() ? 100.0 : iter->second;
    }
};

TEST(GraphSearchTemplateTest, ChoosesBestSequence) {
    MockNode start;
    MockNode a0;
    MockNode a1;
    MockNode b0;
    MockNode b1;
    MockNode goal;

    start.edgeCosts = {{&a0, 1.0}, {&a1, 4.0}};
    a0.edgeCosts = {{&b0, 1.0}, {&b1, 5.0}};
    a1.edgeCosts = {{&b0, 1.0}, {&b1, 1.0}};
    b0.edgeCosts = {{&goal, 1.0}};
    b1.edgeCosts = {{&goal, 1.0}};

    arc::GraphSearch<MockNode> search;
    search.addNode(&start);
    search.addNode(&a0);
    search.addNode(&a1);
    search.addNode(&b0);
    search.addNode(&b1);
    search.addNode(&goal);
    search.addEdge(0, 1);
    search.addEdge(0, 2);
    search.addEdge(1, 3);
    search.addEdge(1, 4);
    search.addEdge(2, 3);
    search.addEdge(2, 4);
    search.addEdge(3, 5);
    search.addEdge(4, 5);

    std::vector<MockNode*> path;
    ASSERT_TRUE(search.dijkstraSearch(0, 5, path));
    ASSERT_EQ(path.size(), 4u);
    EXPECT_EQ(path[0], &start);
    EXPECT_EQ(path[1], &a0);
    EXPECT_EQ(path[2], &b0);
    EXPECT_EQ(path[3], &goal);
    EXPECT_NEAR(search.gValue(&goal), 3.0, 1e-9);
}

TEST(GraphSearchTemplateTest, ReportsInvalidInput) {
    MockNode start;
    MockNode goal;
    arc::GraphSearch<MockNode> search;
    search.addNode(&start);
    search.addNode(&goal);
    search.addEdge(0, 1);
    goal.edgeCosts = {{&start, 1.0}};

    std::vector<MockNode*> path;
    EXPECT_FALSE(search.dijkstraSearch(0, 2, path));
    EXPECT_FALSE(search.dijkstraSearch(-1, 0, path));
    EXPECT_TRUE(path.empty());
}

} // namespace
