#pragma once

#include <string>
#include <vector>

struct TrainingSettings;

struct WeightedTest {
    int index;
    double weight;
    WeightedTest(int index = 0, double weight = 1.0);
};

struct TestWeightRecord {
    int fail_streak;
    int fail_count;
    int pass_count;
    double weight;
    int last_seen_gen;
    TestWeightRecord();
};

class TestWeightTable {
public:
    void reset(int total_tests);
    void loadOrReset(const std::string& file, int total_tests, const TrainingSettings& settings);
    bool save(const std::string& file) const;
    bool updateFromFailures(int gen, int active_tests, const std::vector<int>& failures, const TrainingSettings& settings);
    std::vector<WeightedTest> boostedTests(int active_tests, const TrainingSettings& settings) const;
    double maxWeightIn(const std::vector<WeightedTest>& tests) const;
    std::string boostedSummary(const std::vector<WeightedTest>& tests, int max_items) const;

private:
    static double weightForStreak(int streak, double max_weight, double step);
    std::vector<TestWeightRecord> records_;
};

