

#include "TestWeightTable.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "TrainingSettings.h"

WeightedTest::WeightedTest(int index, double weight)
    : index(index), weight(weight) {
}

TestWeightRecord::TestWeightRecord()
    : fail_streak(0), fail_count(0), pass_count(0), weight(1.0),
    last_seen_gen(-1) {
}

void TestWeightTable::reset(int total_tests) {
    records_.assign(std::max(0, total_tests), TestWeightRecord{});
}

void TestWeightTable::loadOrReset(const std::string& file, int total_tests,
    const TrainingSettings& settings) {
    reset(total_tests);
    std::ifstream fin(file);
    if (!fin.is_open())
        return;

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        for (char& ch : line) {
            if (ch == ',' || ch == ';')
                ch = ' ';
        }
        std::istringstream in(line);
        int id = -1;
        TestWeightRecord rec;
        if (!(in >> id >> rec.fail_streak >> rec.fail_count >> rec.pass_count >>
            rec.weight >> rec.last_seen_gen))
            continue;
        if (id < 0 || id >= (int)records_.size())
            continue;
        rec.fail_streak = std::max(0, rec.fail_streak);
        rec.fail_count = std::max(0, rec.fail_count);
        rec.pass_count = std::max(0, rec.pass_count);
        rec.weight = weightForStreak(rec.fail_streak, settings.test_weight_max,
            settings.test_weight_step);
        records_[(size_t)id] = rec;
    }
}

bool TestWeightTable::save(const std::string& file) const {
    std::ofstream fout(file, std::ios::out | std::ios::trunc);
    if (!fout.is_open())
        return false;

    fout << std::fixed << std::setprecision(3);
    fout << "# test_id,fail_streak,fail_count,pass_count,weight,last_seen_gen\n";
    for (int i = 0; i < (int)records_.size(); ++i) {
        const auto& r = records_[(size_t)i];
        fout << i << ',' << r.fail_streak << ',' << r.fail_count << ','
            << r.pass_count << ',' << r.weight << ',' << r.last_seen_gen << '\n';
    }
    return fout.good();
}

bool TestWeightTable::updateFromFailures(
    int gen, int active_tests, const std::vector<int>& failures,
    const TrainingSettings& settings) {
    active_tests = std::clamp(active_tests, 0, (int)records_.size());
    std::vector<unsigned char> failed((size_t)active_tests, 0);
    for (int idx : failures) {
        if (idx >= 0 && idx < active_tests)
            failed[(size_t)idx] = 1;
    }

    bool changed = false;
    for (int i = 0; i < active_tests; ++i) {
        auto& r = records_[(size_t)i];
        const TestWeightRecord before = r;
        if (failed[(size_t)i]) {
            ++r.fail_streak;
            ++r.fail_count;
        }
        else {
            if (r.fail_streak > 0)
                --r.fail_streak;
            ++r.pass_count;
        }
        r.last_seen_gen = gen;
        r.weight = weightForStreak(r.fail_streak, settings.test_weight_max,
            settings.test_weight_step);
        changed = changed || r.fail_streak != before.fail_streak ||
            r.fail_count != before.fail_count ||
            r.pass_count != before.pass_count || r.weight != before.weight;
    }
    return changed;
}

std::vector<WeightedTest>
TestWeightTable::boostedTests(int active_tests,
    const TrainingSettings& settings) const {
    active_tests = std::clamp(active_tests, 0, (int)records_.size());
    std::vector<WeightedTest> tests;
    for (int i = 0; i < active_tests; ++i) {
        const auto& r = records_[(size_t)i];
        if (r.weight > 1.000001)
            tests.push_back({ i, r.weight });
    }
    std::sort(tests.begin(), tests.end(), [&](const auto& a, const auto& b) {
        const auto& ra = records_[(size_t)a.index];
        const auto& rb = records_[(size_t)b.index];
        if (a.weight != b.weight)
            return a.weight > b.weight;
        if (ra.fail_streak != rb.fail_streak)
            return ra.fail_streak > rb.fail_streak;
        if (ra.fail_count != rb.fail_count)
            return ra.fail_count > rb.fail_count;
        return a.index < b.index;
        });
    if ((int)tests.size() > settings.test_weight_max_tests)
        tests.resize((size_t)settings.test_weight_max_tests);
    return tests;
}

double TestWeightTable::maxWeightIn(
    const std::vector<WeightedTest>& tests) const {
    double max_weight = 1.0;
    for (const auto& test : tests)
        max_weight = std::max(max_weight, test.weight);
    return max_weight;
}

std::string TestWeightTable::boostedSummary(
    const std::vector<WeightedTest>& tests, int max_items) const {
    std::ostringstream out;
    int shown = std::min((int)tests.size(), std::max(0, max_items));
    out << std::fixed << std::setprecision(2);
    for (int i = 0; i < shown; ++i) {
        if (i > 0)
            out << ' ';
        out << tests[(size_t)i].index << 'x' << tests[(size_t)i].weight;
    }
    if ((int)tests.size() > shown)
        out << " ...";
    return out.str();
}

double TestWeightTable::weightForStreak(int streak, double max_weight,
    double step) {
    double value = 1.0 + std::max(0, streak) * step;
    return std::clamp(value, 1.0, max_weight);
}

