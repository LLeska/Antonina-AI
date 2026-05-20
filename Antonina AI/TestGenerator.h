#pragma once

#include <array>
#include <iosfwd>
#include <string>
#include <vector>

namespace TestGenerator {

constexpr int CATEGORY_COUNT = 12;

struct TestCase {
  int ax = 1;
  int ay = 1;
  int Ox = 1;
  int Oy = 1;
  int gx = 1;
  int gy = 2;
  int rn = 0;
  int source_index = 0;
};

int manhattan(int ax, int ay, int bx, int by);
int categoryIndex(const TestCase &test);
const char *categoryName(int index);
bool curriculumLess(const TestCase &a, const TestCase &b);
void orderCurriculum(std::vector<TestCase> &tests);
std::array<int, CATEGORY_COUNT> categoryCounts(const std::vector<TestCase> &tests);
std::vector<TestCase> generateBaseTests();
bool readTests(std::istream &in, std::vector<TestCase> &tests,
               std::string *error = nullptr, int max_tests = -1);
bool writeTests(const std::string &file, const std::vector<TestCase> &tests,
                std::string *error = nullptr);

} 
