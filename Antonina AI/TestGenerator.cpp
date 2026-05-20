#include "TestGenerator.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <fstream>
#include <mutex>
#include <sstream>

namespace TestGenerator {
namespace {

enum Category {
  STRAIGHT_TOUCH = 0,
  STRAIGHT_DETOUR = 1,
  OPEN_TURN = 2,
  HOME_EDGE = 3,
  HOME_CORNER = 4,
  BUCKET_EDGE_HOME_INNER = 5,
  BUCKET_EDGE_HOME_EDGE = 6,
  BUCKET_EDGE_HOME_CORNER = 7,
  BUCKET_CORNER_HOME_INNER = 8,
  BUCKET_CORNER_HOME_EDGE = 9,
  BUCKET_CORNER_HOME_CORNER = 10,
  STONES = 11
};

constexpr int BOARD_SIZE = 8;
constexpr int BOARD_CELLS = BOARD_SIZE * BOARD_SIZE;
constexpr int NO_STONE_CASES = BOARD_CELLS * BOARD_CELLS * BOARD_CELLS;
constexpr int STEP_LIMIT = 40;
constexpr int UNSOLVED_STEP = 1000;

bool inBounds(int x, int y) { return x >= 0 && x < 8 && y >= 0 && y < 8; }

bool sameCell(int ax, int ay, int bx, int by) {
  return ax == bx && ay == by;
}

int cellIndex(int x, int y) { return x * BOARD_SIZE + y; }

void cellCoords(int index, int &x, int &y) {
  x = index / BOARD_SIZE;
  y = index % BOARD_SIZE;
}

bool onEdge(int x, int y) { return x == 0 || x == 7 || y == 0 || y == 7; }

bool inCorner(int x, int y) {
  return (x == 0 || x == 7) && (y == 0 || y == 7);
}

bool aligned(const TestCase &test) {
  return test.Ox == test.gx || test.Oy == test.gy;
}

bool validTest(const TestCase &test) {
  if (!inBounds(test.ax, test.ay) || !inBounds(test.Ox, test.Oy) ||
      !inBounds(test.gx, test.gy) || test.rn < 0)
    return false;
  if (sameCell(test.gx, test.gy, test.ax, test.ay) ||
      sameCell(test.gx, test.gy, test.Ox, test.Oy))
    return false;
  return true;
}

void addTest(std::vector<TestCase> &tests, int ax, int ay, int Ox, int Oy,
             int gx, int gy, int rn) {
  TestCase test{ax, ay, Ox, Oy, gx, gy, rn, (int)tests.size()};
  if (validTest(test))
    tests.push_back(test);
}

int edgeRank(const TestCase &test) {
  int value = 0;
  if (onEdge(test.gx, test.gy))
    value += 1;
  if (inCorner(test.gx, test.gy))
    value += 2;
  if (onEdge(test.Ox, test.Oy))
    value += 4;
  if (inCorner(test.Ox, test.Oy))
    value += 8;
  return value;
}

int sign(int v) { return (v > 0) - (v < 0); }

int directionBucket(const TestCase &test) {
  int dx = sign(test.gx - test.Ox);
  int dy = sign(test.gy - test.Oy);
  return (dx + 1) * 3 + (dy + 1);
}

int noStoneCaseIndex(const TestCase &test) {
  return (cellIndex(test.ax, test.ay) * BOARD_CELLS +
          cellIndex(test.Ox, test.Oy)) *
             BOARD_CELLS +
         cellIndex(test.gx, test.gy);
}

int solveNoStoneStepsUncached(const TestCase &test) {
  if (!validTest(test))
    return UNSOLVED_STEP;

  const int home = cellIndex(test.Ox, test.Oy);
  const int start_agent = cellIndex(test.ax, test.ay);
  const int start_bucket = cellIndex(test.gx, test.gy);

  std::array<unsigned char, BOARD_CELLS * BOARD_CELLS> seen{};
  std::vector<int> agents;
  std::vector<int> buckets;
  std::vector<int> steps;
  agents.reserve(BOARD_CELLS * BOARD_CELLS);
  buckets.reserve(BOARD_CELLS * BOARD_CELLS);
  steps.reserve(BOARD_CELLS * BOARD_CELLS);

  auto push_state = [&](int agent, int bucket, int step) {
    const int key = agent * BOARD_CELLS + bucket;
    if (seen[key])
      return;
    seen[key] = 1;
    agents.push_back(agent);
    buckets.push_back(bucket);
    steps.push_back(step);
  };

  push_state(start_agent, start_bucket, 0);

  static constexpr int dx[4] = {-1, 0, 1, 0};
  static constexpr int dy[4] = {0, 1, 0, -1};

  for (std::size_t head = 0; head < agents.size(); ++head) {
    const int agent = agents[head];
    const int bucket = buckets[head];
    const int step = steps[head];
    if (step >= STEP_LIMIT)
      continue;

    int ax, ay;
    cellCoords(agent, ax, ay);
    for (int move = 0; move < 4; ++move) {
      const int tox = ax + dx[move];
      const int toy = ay + dy[move];
      if (!inBounds(tox, toy))
        continue;

      const int target = cellIndex(tox, toy);
      int next_agent = target;
      int next_bucket = bucket;

      if (target == bucket) {
        const int bx = tox + dx[move];
        const int by = toy + dy[move];
        if (!inBounds(bx, by))
          continue;

        const int push_to = cellIndex(bx, by);
        if (push_to == home)
          return step + 1;
        if (push_to == agent || push_to == bucket)
          continue;

        next_bucket = push_to;
      } else {
        const int px = ax - dx[move];
        const int py = ay - dy[move];
        if (inBounds(px, py) && cellIndex(px, py) == bucket) {
          if (agent == home)
            return step + 1;
          next_bucket = agent;
        }
      }

      push_state(next_agent, next_bucket, step + 1);
    }
  }

  return UNSOLVED_STEP + manhattan(test.ax, test.ay, test.gx, test.gy) +
         manhattan(test.Ox, test.Oy, test.gx, test.gy);
}

int noStoneSolutionSteps(const TestCase &test) {
  if (!validTest(test))
    return UNSOLVED_STEP;

  static std::array<int, NO_STONE_CASES> cache;
  static std::once_flag init_flag;
  static std::mutex cache_mutex;
  std::call_once(init_flag, [] { cache.fill(-1); });

  const int index = noStoneCaseIndex(test);
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (cache[index] >= 0)
      return cache[index];
  }

  const int solved = solveNoStoneStepsUncached(test);
  std::lock_guard<std::mutex> lock(cache_mutex);
  if (cache[index] < 0)
    cache[index] = solved;
  return cache[index];
}

int difficultyStage(const TestCase &test) {
  if (test.rn > 0)
    return UNSOLVED_STEP + test.rn;
  return noStoneSolutionSteps(test);
}

int curriculumDistance(const TestCase &test) {
  return manhattan(test.ax, test.ay, test.gx, test.gy) +
         manhattan(test.Ox, test.Oy, test.gx, test.gy);
}

bool sameDifficulty(const TestCase &a, const TestCase &b) {
  return a.rn == b.rn && difficultyStage(a) == difficultyStage(b);
}

int interleaveBucket(const TestCase &test) {
  return (edgeRank(test) * CATEGORY_COUNT + categoryIndex(test)) * 9 +
         directionBucket(test);
}

} 

int manhattan(int ax, int ay, int bx, int by) {
  return std::abs(ax - bx) + std::abs(ay - by);
}

int categoryIndex(const TestCase &test) {
  if (test.rn > 0)
    return STONES;

  const bool test_aligned = aligned(test);
  const int bucket_to_home = manhattan(test.Ox, test.Oy, test.gx, test.gy);
  const bool bucket_edge = onEdge(test.gx, test.gy);
  const bool bucket_corner = inCorner(test.gx, test.gy);
  const bool home_edge = onEdge(test.Ox, test.Oy);
  const bool home_corner = inCorner(test.Ox, test.Oy);

  if (!bucket_edge && !home_edge) {
    if (test_aligned && bucket_to_home == 1)
      return STRAIGHT_TOUCH;
    if (test_aligned)
      return STRAIGHT_DETOUR;
    return OPEN_TURN;
  }

  if (!bucket_edge) {
    if (home_corner)
      return HOME_CORNER;
    if (home_edge)
      return HOME_EDGE;
  }

  if (bucket_corner) {
    if (home_corner)
      return BUCKET_CORNER_HOME_CORNER;
    if (home_edge)
      return BUCKET_CORNER_HOME_EDGE;
    return BUCKET_CORNER_HOME_INNER;
  }

  if (bucket_edge) {
    if (home_corner)
      return BUCKET_EDGE_HOME_CORNER;
    if (home_edge)
      return BUCKET_EDGE_HOME_EDGE;
    return BUCKET_EDGE_HOME_INNER;
  }

  return OPEN_TURN;
}

const char *categoryName(int index) {
  static const char *names[CATEGORY_COUNT] = {
      "straight_touch",
      "straight_detour",
      "open_turn",
      "home_edge",
      "home_corner",
      "bucket_edge_home_inner",
      "bucket_edge_home_edge",
      "bucket_edge_home_corner",
      "bucket_corner_home_inner",
      "bucket_corner_home_edge",
      "bucket_corner_home_corner",
      "stones"};
  if (index < 0 || index >= CATEGORY_COUNT)
    return "unknown";
  return names[index];
}

bool curriculumLess(const TestCase &a, const TestCase &b) {
  int stage_a = difficultyStage(a);
  int stage_b = difficultyStage(b);
  if (stage_a != stage_b)
    return stage_a < stage_b;

  int category_a = categoryIndex(a);
  int category_b = categoryIndex(b);
  if (category_a != category_b)
    return category_a < category_b;

  int total_a = curriculumDistance(a);
  int total_b = curriculumDistance(b);
  if (total_a != total_b)
    return total_a < total_b;

  int dist_a = manhattan(a.Ox, a.Oy, a.gx, a.gy);
  int dist_b = manhattan(b.Ox, b.Oy, b.gx, b.gy);
  if (dist_a != dist_b)
    return dist_a < dist_b;

  int rover_a = manhattan(a.ax, a.ay, a.gx, a.gy);
  int rover_b = manhattan(b.ax, b.ay, b.gx, b.gy);
  if (rover_a != rover_b)
    return rover_a < rover_b;

  int edge_a = edgeRank(a);
  int edge_b = edgeRank(b);
  if (edge_a != edge_b)
    return edge_a < edge_b;

  return a.source_index < b.source_index;
}

void orderCurriculum(std::vector<TestCase> &tests) {
  for (const auto &test : tests)
    (void)difficultyStage(test);

  std::stable_sort(tests.begin(), tests.end(), curriculumLess);

  static constexpr std::array<int, 9> direction_order = {0, 8, 2, 6, 1,
                                                        7, 3, 5, 4};
  constexpr int EDGE_RANK_COUNT = 16;
  constexpr int BUCKET_COUNT = EDGE_RANK_COUNT * CATEGORY_COUNT * 9;

  std::vector<TestCase> ordered;
  ordered.reserve(tests.size());

  for (std::size_t begin = 0; begin < tests.size();) {
    std::size_t end = begin + 1;
    while (end < tests.size() && sameDifficulty(tests[begin], tests[end]))
      ++end;

    std::array<std::vector<TestCase>, BUCKET_COUNT> buckets;
    for (std::size_t i = begin; i < end; ++i)
      buckets[interleaveBucket(tests[i])].push_back(tests[i]);

    std::array<std::size_t, BUCKET_COUNT> cursor{};
    std::size_t remaining = end - begin;
    while (remaining > 0) {
      for (int edge = 0; edge < EDGE_RANK_COUNT && remaining > 0; ++edge) {
        for (int category = 0; category < CATEGORY_COUNT && remaining > 0;
             ++category) {
          for (int direction : direction_order) {
            if (remaining == 0)
              break;
            const int bucket = (edge * CATEGORY_COUNT + category) * 9 +
                               direction;
            if (cursor[bucket] >= buckets[bucket].size())
              continue;
            ordered.push_back(buckets[bucket][cursor[bucket]++]);
            --remaining;
          }
        }
      }
    }

    begin = end;
  }

  tests.swap(ordered);
}

std::array<int, CATEGORY_COUNT>
categoryCounts(const std::vector<TestCase> &tests) {
  std::array<int, CATEGORY_COUNT> counts{};
  for (const auto &test : tests)
    counts[categoryIndex(test)]++;
  return counts;
}

std::vector<TestCase> generateBaseTests() {
  std::vector<TestCase> tests;
  tests.reserve(4032);

  for (int ax = 0; ax < BOARD_SIZE; ++ax) {
    for (int ay = 0; ay < BOARD_SIZE; ++ay) {
      for (int gx = 0; gx < BOARD_SIZE; ++gx) {
        for (int gy = 0; gy < BOARD_SIZE; ++gy)
          addTest(tests, ax, ay, ax, ay, gx, gy, 0);
      }
    }
  }

  orderCurriculum(tests);
  for (int i = 0; i < (int)tests.size(); ++i)
    tests[i].source_index = i;
  return tests;
}

bool readTests(std::istream &in, std::vector<TestCase> &tests,
               std::string *error, int max_tests) {
  tests.clear();
  std::string line;
  int line_number = 0;
  while ((max_tests < 0 || (int)tests.size() < max_tests) &&
         std::getline(in, line)) {
    ++line_number;
    if (line.empty())
      continue;

    std::istringstream row(line);
    TestCase test;
    test.source_index = (int)tests.size();
    if (!(row >> test.ax >> test.ay >> test.Ox >> test.Oy >> test.gx >>
          test.gy >> test.rn)) {
      if (error)
        *error = "bad row " + std::to_string(line_number);
      return false;
    }
    std::string extra;
    if (row >> extra) {
      if (error)
        *error = "too many columns at row " + std::to_string(line_number);
      return false;
    }
    if (!validTest(test)) {
      if (error)
        *error = "invalid coordinates at row " + std::to_string(line_number);
      return false;
    }
    tests.push_back(test);
  }
  return true;
}

bool writeTests(const std::string &file, const std::vector<TestCase> &tests,
                std::string *error) {
  std::ofstream out(file);
  if (!out.is_open()) {
    if (error)
      *error = "cannot open " + file;
    return false;
  }

  for (const auto &test : tests) {
    if (!validTest(test)) {
      if (error)
        *error = "attempted to write invalid test";
      return false;
    }
    out << test.ax << ' ' << test.ay << ' ' << test.Ox << ' ' << test.Oy
        << ' ' << test.gx << ' ' << test.gy << ' ' << test.rn << '\n';
  }
  out.flush();
  if (!out.good()) {
    if (error)
      *error = "failed while writing " + file;
    return false;
  }
  return true;
}

} 
