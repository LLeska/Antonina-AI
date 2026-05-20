#include "TestGenerator.h"

#include <filesystem>
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

  std::filesystem::path weights = std::filesystem::path(output).parent_path();
  weights /= "test_weights.csv";
  std::error_code ec;
  std::filesystem::remove(weights, ec);

  auto counts = TestGenerator::categoryCounts(tests);
  std::cout << "wrote " << tests.size() << " tests to " << output << '\n';
  std::cout << "reset " << weights.string() << '\n';
  for (int i = 0; i < TestGenerator::CATEGORY_COUNT; ++i) {
    std::cout << TestGenerator::categoryName(i) << '=' << counts[i] << '\n';
  }
  return 0;
}
