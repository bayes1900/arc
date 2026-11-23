#include <gtest/gtest.h>

#include <diagnostics/diagnostics.h>
#include <diagnostics/exploration.h>

namespace {

class DiagnosticsTest : public ::testing::Test {
protected:
    arc::DiagnosticsCollector collector_;
};

TEST_F(DiagnosticsTest, StateTransitionsTrackTimeAndEntries) {
    collector_.initializeState(0, 0.0);
    collector_.recordStateTransition(0, 1, 2.0, "trigger");
    collector_.recordStateTransition(1, 2, 5.0, "start_plan");

    const arc::DiagSnapshot snapshot = collector_.snapshot(7.0);
    EXPECT_EQ(snapshot.state, 2);
    EXPECT_EQ(snapshot.stateTransitions, 2u);
    EXPECT_DOUBLE_EQ(snapshot.stateTimes[0], 2.0);
    EXPECT_DOUBLE_EQ(snapshot.stateTimes[1], 3.0);
    EXPECT_DOUBLE_EQ(snapshot.stateTimes[2], 2.0);
    EXPECT_EQ(snapshot.stateEntries[0], 1u);
    EXPECT_EQ(snapshot.stateEntries[1], 1u);
    EXPECT_EQ(snapshot.stateEntries[2], 1u);
}

TEST_F(DiagnosticsTest, StageStatsAggregate) {
    collector_.recordStage(arc::DiagStage::ASTAR, 0.1);
    collector_.recordStage(arc::DiagStage::ASTAR, 0.3);
    collector_.recordStage(arc::DiagStage::ASTAR, 0.2);

    const arc::DiagSnapshot snapshot = collector_.snapshot(1.0);
    const arc::DiagStageStats& stats =
        snapshot.stages[static_cast<size_t>(arc::DiagStage::ASTAR)];
    EXPECT_EQ(stats.count, 3u);
    EXPECT_DOUBLE_EQ(stats.last, 0.2);
    EXPECT_DOUBLE_EQ(stats.max, 0.3);
    EXPECT_DOUBLE_EQ(stats.average(), 0.2);
}

TEST_F(DiagnosticsTest, EventsCanBeDrained) {
    arc::DiagRecord record;
    record.eventType = "plan_failure";
    record.stage = "astar";
    record.reason = "astar_nopath";
    collector_.recordEvent(record);

    const std::vector<arc::DiagRecord> events = collector_.takeEvents();
    ASSERT_EQ(events.size(), 1u);
    EXPECT_EQ(events.front().eventType, "plan_failure");
    EXPECT_EQ(events.front().stage, "astar");
    EXPECT_EQ(events.front().reason, "astar_nopath");
    EXPECT_TRUE(collector_.takeEvents().empty());
}

TEST_F(DiagnosticsTest, EventQueueDropsOldest) {
    collector_.setEventQueueSize(2);
    for (int i = 0; i < 3; ++ i) {
        arc::DiagRecord record;
        record.value = static_cast<double>(i);
        collector_.recordEvent(record);
    }

    const std::vector<arc::DiagRecord> events = collector_.takeEvents();
    ASSERT_EQ(events.size(), 2u);
    EXPECT_DOUBLE_EQ(events.front().value, 1.0);
    EXPECT_DOUBLE_EQ(events.back().value, 2.0);
    EXPECT_EQ(collector_.snapshot(0.0).droppedEvents, 1u);
}

TEST_F(DiagnosticsTest, CountersAndGauges) {
    collector_.increment(arc::DiagCounter::PLAN_ATTEMPTS);
    collector_.increment(arc::DiagCounter::PLAN_ATTEMPTS, 2);
    collector_.setGauge(arc::DiagGauge::FRONTIER_COUNT, 7.0);

    const arc::DiagSnapshot snapshot = collector_.snapshot(0.0);
    EXPECT_EQ(snapshot.counters[static_cast<size_t>(arc::DiagCounter::PLAN_ATTEMPTS)], 3u);
    EXPECT_DOUBLE_EQ(snapshot.gauges[static_cast<size_t>(arc::DiagGauge::FRONTIER_COUNT)], 7.0);
}

TEST_F(DiagnosticsTest, DisabledCollectorIgnoresWrites) {
    collector_.setEnabled(false);
    collector_.recordStage(arc::DiagStage::ASTAR, 1.0);
    collector_.increment(arc::DiagCounter::PLAN_ATTEMPTS);
    collector_.recordEvent(arc::DiagRecord());

    EXPECT_TRUE(collector_.takeEvents().empty());
    const arc::DiagSnapshot snapshot = collector_.snapshot(0.0);
    EXPECT_EQ(snapshot.counters[static_cast<size_t>(arc::DiagCounter::PLAN_ATTEMPTS)], 0u);
    EXPECT_EQ(snapshot.stages[static_cast<size_t>(arc::DiagStage::ASTAR)].count, 0u);
}

TEST_F(DiagnosticsTest, StageTimerRecordsElapsed) {
    {
        arc::StageTimer timer(&collector_, arc::DiagStage::GRID_TOUR);
        EXPECT_GE(timer.elapsed(), 0.0);
    }
    const arc::DiagSnapshot snapshot = collector_.snapshot(0.0);
    EXPECT_EQ(snapshot.stages[static_cast<size_t>(arc::DiagStage::GRID_TOUR)].count, 1u);
}

TEST_F(DiagnosticsTest, FormatsNumbersAndVectors) {
    EXPECT_EQ(arc::DiagFormat::number(1.23456), "1.235");
    EXPECT_EQ(arc::DiagFormat::vectorParen(Eigen::Vector3d(1.0, -2.0, 0.5)), "(1.00,-2.00,0.50)");
    const std::string detail = arc::DiagDetail()
        .add("grids", 3)
        .add("dist", 1.5)
        .add("pos", arc::DiagFormat::vectorParen(Eigen::Vector3d::Zero()))
        .str();
    EXPECT_EQ(detail, "grids=3 dist=1.500 pos=(0.00,0.00,0.00)");
}

TEST_F(DiagnosticsTest, MapsStatusCodesToStableStrings) {
    EXPECT_STREQ(arc::DiagCodes::astarStatus(1), "NOPATH");
    EXPECT_STREQ(arc::DiagCodes::kinoStatus(3), "NOPATH");
    EXPECT_STREQ(arc::DiagCodes::planResult(5), "OPTIMIZEFAILED");
    EXPECT_STREQ(arc::DiagCodes::bsplineStatus(6), "MAXTIMEREACHED");
    EXPECT_STREQ(arc::DiagCodes::yawStatus(3), "NOSOLUTION");
    EXPECT_STREQ(arc::DiagCodes::tspFailure(arc::TspFailure::SERVICE_TIMEOUT), "tsp_service_timeout");

    arc::PlannerResult result;
    result.code = arc::PlannerResult::SEARCHFAILED;
    result.searchStatus = arc::KinoAstarSearchStatus::NOPATH;
    EXPECT_STREQ(arc::DiagCodes::plannerReason(result), "kino_nopath");
}

TEST_F(DiagnosticsTest, RecorderBuildsPlanEvents) {
    arc::DiagnosticsRecorder recorder(&collector_, 3);
    recorder.planAttempt();

    arc::PlanSuccessInfo success;
    success.duration = 0.2;
    success.distance = 4.5;
    success.gridCount = 6;
    success.frontierCount = 2;
    success.viewpointCount = 7;
    success.detail = "pos=(0.00,0.00,0.00)";
    recorder.planSuccess(10.0, success);

    arc::DiagFailure failure;
    failure.stage = arc::DiagStage::ASTAR;
    failure.reason = "astar_nopath";
    failure.code = "NOPATH";
    failure.detail = "start=(0.00,0.00,0.00)";
    failure.duration = 0.01;
    failure.viewpointCount = 7;
    failure.counter = arc::DiagCounter::ASTAR_FAILURES;
    recorder.planFailure(11.0, failure);

    const std::vector<arc::DiagRecord> events = collector_.takeEvents();
    ASSERT_EQ(events.size(), 2u);
    EXPECT_EQ(events[0].eventType, "plan_success");
    EXPECT_EQ(events[0].stage, "plan_total");
    EXPECT_DOUBLE_EQ(events[0].value, 4.5);
    EXPECT_EQ(events[0].gridCount, 6);
    EXPECT_EQ(events[0].droneId, 3);
    EXPECT_EQ(events[1].eventType, "plan_failure");
    EXPECT_EQ(events[1].stage, "astar");
    EXPECT_EQ(events[1].reason, "astar_nopath");
    EXPECT_EQ(events[1].code, "NOPATH");

    const arc::DiagSnapshot snapshot = collector_.snapshot(12.0);
    EXPECT_EQ(snapshot.counters[static_cast<size_t>(arc::DiagCounter::PLAN_ATTEMPTS)], 1u);
    EXPECT_EQ(snapshot.counters[static_cast<size_t>(arc::DiagCounter::PLAN_SUCCESSES)], 1u);
    EXPECT_EQ(snapshot.counters[static_cast<size_t>(arc::DiagCounter::ASTAR_FAILURES)], 1u);
}

TEST_F(DiagnosticsTest, RecorderPreservesCollisionAndPairOptContext) {
    arc::DiagnosticsRecorder recorder(&collector_, 2);

    arc::CollisionInfo collision;
    collision.swarm = true;
    collision.distance = 0.6;
    collision.otherDroneId = 4;
    collision.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    recorder.collision(5.0, collision);

    arc::PairOptInfo pairOpt;
    pairOpt.reason = "pair_opt_accepted";
    pairOpt.otherDroneId = 3;
    pairOpt.duration = 0.05;
    pairOpt.selfCount = 7;
    pairOpt.otherCount = 5;
    pairOpt.counter = arc::DiagCounter::PAIR_OPT_ACCEPTED;
    pairOpt.detail = "role=req";
    recorder.pairOpt(6.0, pairOpt);

    const std::vector<arc::DiagRecord> events = collector_.takeEvents();
    ASSERT_EQ(events.size(), 2u);
    EXPECT_EQ(events[0].eventType, "collision");
    EXPECT_EQ(events[0].otherDroneId, 4);
    EXPECT_DOUBLE_EQ(events[0].distance, 0.6);
    EXPECT_EQ(events[0].detail, "pos=(1.00,2.00,3.00)");
    EXPECT_EQ(events[1].eventType, "pair_opt");
    EXPECT_EQ(events[1].otherDroneId, 3);
    EXPECT_EQ(events[1].gridCount, 7);
    EXPECT_EQ(events[1].frontierCount, 5);
    EXPECT_DOUBLE_EQ(events[1].duration, 0.05);
}


} // namespace
