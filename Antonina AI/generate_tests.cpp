#include "TestGenerator.h"

#include <iostream>
#include <string>

int main(int argc, char **argv) {
  std::string output = argc > 1 ? argv[1] : "Test0.csv";
  std::vector<TestGenerator::TestCase> tests =
      TestGenerator::generateBaseTests();

  std::string error;
  if (!TestGenerator::writeTests(output, tests, &error)) {
    std::cerr << error << '\n';
    return 1;
  }

  auto counts = TestGenerator::categoryCounts(tests);
  std::cout << "wrote " << tests.size() << " tests to " << output << '\n';
  for (int i = 0; i < TestGenerator::CATEGORY_COUNT; ++i) {
    std::cout << TestGenerator::categoryName(i) << '=' << counts[i] << '\n';
  }
  return 0;
}
