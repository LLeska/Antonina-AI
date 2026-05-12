#include "TestGenerator.h"

#include <algorithm>
#include <cmath>
#include <fstream>
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

bool inBounds(int x, int y) { return x >= 0 && x < 8 && y >= 0 && y < 8; }

bool sameCell(int ax, int ay, int bx, int by) {
  return ax == bx && ay == by;
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
  int category_a = categoryIndex(a);
  int category_b = categoryIndex(b);
  if (category_a != category_b)
    return category_a < category_b;

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

  for (int ax = 1; ax < 7; ++ax) {
    for (int ay = 1; ay < 7; ++ay) {
      for (int gy = 1; gy < 7; ++gy) {
        if (gy != ay)
          addTest(tests, ax, ay, ax, ay, ax, gy, 0);
      }
      for (int gx = 1; gx < 7; ++gx) {
        if (gx != ax)
          addTest(tests, ax, ay, ax, ay, gx, ay, 0);
      }
    }
  }

  for (int ax = 1; ax < 7; ++ax) {
    for (int ay = 1; ay < 7; ++ay) {
      for (int gx = 1; gx < 7; ++gx) {
        if (gx == ax)
          continue;
        for (int gy = 1; gy < 7; ++gy) {
          if (gy != ay)
            addTest(tests, ax, ay, ax, ay, gx, gy, 0);
        }
      }
    }
  }

  for (int ax : {0, 7}) {
    for (int ay = 0; ay < 8; ++ay) {
      for (int gx = 0; gx < 8; ++gx) {
        for (int gy = 0; gy < 8; ++gy)
          addTest(tests, ax, ay, ax, ay, gx, gy, 0);
      }
    }
  }

  for (int ay : {0, 7}) {
    for (int ax = 1; ax < 7; ++ax) {
      for (int gx = 0; gx < 8; ++gx) {
        for (int gy = 0; gy < 8; ++gy)
          addTest(tests, ax, ay, ax, ay, gx, gy, 0);
      }
    }
  }

  for (int gx : {0, 7}) {
    for (int gy = 0; gy < 8; ++gy) {
      for (int ax = 1; ax < 7; ++ax) {
        for (int ay = 1; ay < 7; ++ay)
          addTest(tests, ax, ay, ax, ay, gx, gy, 0);
      }
    }
  }

  for (int gy : {0, 7}) {
    for (int gx = 1; gx < 7; ++gx) {
      for (int ax = 1; ax < 7; ++ax) {
        for (int ay = 1; ay < 7; ++ay)
          addTest(tests, ax, ay, ax, ay, gx, gy, 0);
      }
    }
  }

  std::stable_sort(tests.begin(), tests.end(), curriculumLess);
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
