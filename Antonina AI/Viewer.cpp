#include "Viewer.h"

#include "NeatEvolution.h"
#include "TestGenerator.h"

#include <SFML/Graphics.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

const char *FONT_PATHS[] = {
    "arial.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", nullptr};

bool loadFont(sf::Font &font) {
  for (int k = 0; FONT_PATHS[k]; ++k) {
    if (!std::filesystem::exists(FONT_PATHS[k]))
      continue;
    if (font.openFromFile(FONT_PATHS[k]))
      return true;
  }
  return false;
}

void drawText(sf::RenderWindow &w, const sf::Font *font, const std::string &s,
              float x, float y, unsigned size = 14,
              sf::Color col = sf::Color::White) {
  if (!font)
    return;
  sf::Text t(*font, sf::String::fromUtf8(s.begin(), s.end()), size);
  t.setFillColor(col);
  t.setPosition({x, y});
  w.draw(t);
}

void drawRect(sf::RenderWindow &w, float x, float y, float ww, float hh,
              sf::Color fill, sf::Color outline = sf::Color::Transparent,
              float thickness = 0.f) {
  sf::RectangleShape r({ww, hh});
  r.setPosition({x, y});
  r.setFillColor(fill);
  r.setOutlineColor(outline);
  r.setOutlineThickness(thickness);
  w.draw(r);
}

void drawLine(sf::RenderWindow &w, float x1, float y1, float x2, float y2,
              sf::Color col) {
  sf::VertexArray line(sf::PrimitiveType::Lines, 2);
  line[0].position = {x1, y1};
  line[0].color = col;
  line[1].position = {x2, y2};
  line[1].color = col;
  w.draw(line);
}

void drawCircle(sf::RenderWindow &w, float x, float y, float r, sf::Color fill,
                sf::Color outline = sf::Color::Transparent,
                float thickness = 0.f) {
  sf::CircleShape c(r);
  c.setOrigin({r, r});
  c.setPosition({x, y});
  c.setFillColor(fill);
  c.setOutlineColor(outline);
  c.setOutlineThickness(thickness);
  w.draw(c);
}

std::string compactValue(double v) {
  double a = std::abs(v);
  char buf[64];
  if (a >= 1000000000.0)
    std::snprintf(buf, sizeof(buf), "%.2fB", v / 1000000000.0);
  else if (a >= 1000000.0)
    std::snprintf(buf, sizeof(buf), "%.2fM", v / 1000000.0);
  else if (a >= 1000.0)
    std::snprintf(buf, sizeof(buf), "%.1fk", v / 1000.0);
  else if (a < 10.0 && std::abs(v - std::round(v)) > 0.01)
    std::snprintf(buf, sizeof(buf), "%.2f", v);
  else
    std::snprintf(buf, sizeof(buf), "%.0f", v);
  return buf;
}

template <typename Getter>
void drawLineChart(sf::RenderWindow &w, const sf::Font *font,
                   const std::vector<GenSample> &samples, Getter get,
                   const std::string &title, float x, float y, float ww,
                   float hh, sf::Color line_col) {
  drawRect(w, x, y, ww, hh, sf::Color(18, 18, 26), sf::Color(54, 58, 74),
           1.f);
  drawText(w, font, title, x + 8, y + 5, 13, sf::Color(205, 210, 225));

  if (samples.size() < 2)
    return;

  double vmin = 1e18, vmax = -1e18;
  for (const auto &s : samples) {
    double v = (double)get(s);
    vmin = std::min(vmin, v);
    vmax = std::max(vmax, v);
  }
  if (vmax - vmin < 1e-9)
    vmax = vmin + 1.0;

  int g0 = samples.front().gen;
  int g1 = samples.back().gen;
  if (g1 == g0)
    g1 = g0 + 1;

  const float pad_l = 8.f, pad_r = 8.f, pad_t = 24.f, pad_b = 18.f;
  auto px = [&](int g) {
    return x + pad_l + (ww - pad_l - pad_r) * float(g - g0) / float(g1 - g0);
  };
  auto py = [&](double v) {
    double t = (v - vmin) / (vmax - vmin);
    return y + pad_t + (hh - pad_t - pad_b) * float(1.0 - t);
  };

  for (int i = 1; i <= 3; ++i) {
    float yy = y + pad_t + (hh - pad_t - pad_b) * i / 4.f;
    drawLine(w, x + pad_l, yy, x + ww - pad_r, yy, sf::Color(34, 36, 48));
  }

  sf::VertexArray strip(sf::PrimitiveType::LineStrip, samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    strip[i].position = {px(samples[i].gen), py((double)get(samples[i]))};
    strip[i].color = line_col;
  }
  w.draw(strip);

  drawText(w, font, compactValue(vmax), x + 7, y + pad_t - 3, 10,
           sf::Color(145, 150, 170));
  drawText(w, font, compactValue(vmin), x + 7, y + hh - pad_b - 2, 10,
           sf::Color(145, 150, 170));
  drawText(w, font, compactValue((double)get(samples.back())), x + ww - 72,
           y + 5, 13, line_col);
}

void drawMetric(sf::RenderWindow &w, const sf::Font *font,
                const std::string &label, const std::string &value, float x,
                float y, float ww, float hh, sf::Color accent,
                double ratio = -1.0) {
  drawRect(w, x, y, ww, hh, sf::Color(22, 23, 32), sf::Color(58, 62, 78),
           1.f);
  drawText(w, font, label, x + 10, y + 7, 12, sf::Color(160, 166, 185));
  drawText(w, font, value, x + 10, y + 25, 19, sf::Color(230, 232, 240));
  if (ratio >= 0.0) {
    ratio = std::clamp(ratio, 0.0, 1.0);
    drawRect(w, x + 10, y + hh - 12, ww - 20, 4, sf::Color(38, 40, 52));
    drawRect(w, x + 10, y + hh - 12, (ww - 20) * (float)ratio, 4, accent);
  } else {
    drawRect(w, x + 10, y + hh - 7, ww - 20, 2, accent);
  }
}

struct UiRect {
  float x;
  float y;
  float w;
  float h;
};

bool contains(const UiRect &r, sf::Vector2f p) {
  return p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h;
}

sf::View makeWindowView(sf::Vector2u window_size) {
  float ww = std::max(1360.f, (float)window_size.x);
  float hh = std::max(760.f, (float)window_size.y);
  return sf::View({ww * 0.5f, hh * 0.5f}, {ww, hh});
}

struct DashboardLayout {
  float width = 0.f;
  float height = 0.f;
  UiRect charts;
  UiRect center;
  UiRect tests;
  UiRect logs;
  UiRect save_button;
};

DashboardLayout makeDashboardLayout(sf::Vector2u window_size) {
  DashboardLayout l;
  l.width = std::max(1360.f, (float)window_size.x);
  l.height = std::max(760.f, (float)window_size.y);

  const float margin = 12.f;
  const float group_gap = 12.f;
  const float inner_gap = 12.f;
  const float top = 46.f;
  const float save_h = 30.f;
  const float bottom_margin = 14.f;
  const float content_bottom = l.height - bottom_margin - save_h - 12.f;
  const float content_h = std::max(420.f, content_bottom - top);
  const float group_area = l.width - margin * 2.f - group_gap * 2.f;

  float charts_w = std::max(300.f, group_area * 0.25f);
  float middle_w = std::max(560.f, group_area * 0.50f);
  float logs_w = std::max(300.f, group_area * 0.25f);
  float tests_w = std::clamp(middle_w * 0.30f, 240.f, 330.f);
  float center_w = std::max(320.f, middle_w - inner_gap - tests_w);

  float x = margin;
  l.charts = {x, top, charts_w, content_h};
  x += charts_w + group_gap;
  l.center = {x, top, center_w, content_h};
  x += center_w + inner_gap;
  l.tests = {x, top, tests_w, content_h};
  x += tests_w + group_gap;
  l.logs = {x, top, logs_w, content_h};
  l.save_button = {margin, l.height - bottom_margin - save_h, 176.f, save_h};
  return l;
}

struct CenterPanelLayout {
  float topology_h = 0.f;
  float metric_y = 0.f;
  float metric_w = 0.f;
  float metric_h = 0.f;
  UiRect diagnostics;
};

CenterPanelLayout makeCenterPanelLayout(const DashboardLayout &layout) {
  CenterPanelLayout c;
  const float center_gap = 12.f;
  const float metric_gap = 10.f;
  c.topology_h = std::clamp(layout.center.h * 0.42f, 260.f, 430.f);
  c.metric_h = std::clamp(layout.center.h * 0.075f, 54.f, 70.f);
  c.metric_w = (layout.center.w - metric_gap * 2.f) / 3.f;
  c.metric_y = layout.center.y + c.topology_h + center_gap;
  const float diag_y =
      c.metric_y + (c.metric_h + metric_gap) * 2.f + center_gap;
  const float diag_h =
      std::max(80.f, layout.center.y + layout.center.h - diag_y);
  c.diagnostics = {layout.center.x, diag_y, layout.center.w, diag_h};
  return c;
}

struct SettingsLayout {
  float width = 0.f;
  float height = 0.f;
  UiRect panel;
  UiRect logs;
};

SettingsLayout makeSettingsLayout(sf::Vector2u window_size) {
  SettingsLayout l;
  l.width = std::max(1360.f, (float)window_size.x);
  l.height = std::max(760.f, (float)window_size.y);

  const float margin = 12.f;
  const float gap = 12.f;
  const float top = 46.f;
  const float bottom = 14.f;
  float log_w = std::clamp(l.width * 0.2f, 300.f, 390.f);
  float panel_w = l.width - margin * 2.f - gap - log_w;
  if (panel_w < 960.f) {
    panel_w = 960.f;
    log_w = std::max(260.f, l.width - margin * 2.f - gap - panel_w);
  }

  const float panel_h = std::max(640.f, l.height - top - bottom);
  l.panel = {margin, top, panel_w, panel_h};
  l.logs = {margin + panel_w + gap, top, log_w, panel_h};
  return l;
}

void drawButton(sf::RenderWindow &w, const sf::Font *font, const UiRect &r,
                const std::string &label, bool hover, sf::Color fill,
                sf::Color outline = sf::Color(120, 160, 220)) {
  drawRect(w, r.x, r.y, r.w, r.h,
           hover ? sf::Color(fill.r + std::min(40, 255 - fill.r),
                             fill.g + std::min(40, 255 - fill.g),
                             fill.b + std::min(40, 255 - fill.b))
                 : fill,
           outline, 1.f);
  drawText(w, font, label, r.x + 12.f, r.y + 7.f, 13, sf::Color::White);
}

std::string formatDouble(double value, int precision = 3) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", precision, value);
  return buf;
}

enum class SettingFieldId {
  Population,
  ActiveTests,
  MaxGenerations,
  Threads,
  MaxPopulation,
  FailureLogInterval,
  PopulationStallGenerations,
  StableFailureChecks,
  PopulationChangeCooldown,
  NeatMutationSigma,
  NeatMutationProb,
  NeatCompatibilityThreshold,
  NeatSurvivalRate,
  NeatAddNodeProb,
  NeatAddConnectionProb,
  NeatSparseConnectionProb,
  NeatInputProbeProb,
  NeatSparseInputProbeProb,
  ClassicLearningRate,
  ClassicParents,
  ClassicHidden1,
  ClassicHidden2,
  ClassicMutationSigma,
  ClassicMutationProb,
};

struct SettingInput {
  SettingFieldId id;
  std::string label;
  std::string text;
  bool decimal;
  UiRect rect;
};

struct ModelCheckpointOption {
  std::string path;
  std::string label;
  std::filesystem::file_time_type write_time{};
};

struct SettingsEditor {
  TrainingSettings settings;
  std::vector<SettingInput> inputs;
  std::vector<ModelCheckpointOption> model_options;
  std::string load_file;
  bool model_picker_open = false;
  int model_picker_first = 0;
  int active_input = -1;
};

void addInput(SettingsEditor &editor, SettingFieldId id,
              const std::string &label, const std::string &text,
              bool decimal) {
  editor.inputs.push_back({id, label, text, decimal, {0.f, 0.f, 0.f, 0.f}});
}

void rebuildSettingsInputs(SettingsEditor &editor) {
  editor.inputs.clear();
  editor.active_input = -1;
  const auto &s = editor.settings;

  addInput(editor, SettingFieldId::Population, "population",
           std::to_string(s.population), false);
  addInput(editor, SettingFieldId::ActiveTests, "active tests on start",
           std::to_string(s.initial_active_tests), false);
  addInput(editor, SettingFieldId::MaxGenerations, "max generations",
           std::to_string(s.max_generations), false);
  addInput(editor, SettingFieldId::Threads, "threads (0 = auto)",
           std::to_string(s.requested_threads), false);
  addInput(editor, SettingFieldId::MaxPopulation,
           "max adaptive population (0 = x3)", std::to_string(s.max_population),
           false);
  addInput(editor, SettingFieldId::FailureLogInterval, "failure log interval",
           std::to_string(s.failure_log_interval), false);
  addInput(editor, SettingFieldId::PopulationStallGenerations,
           "population stall generations",
           std::to_string(s.population_stall_generations), false);
  addInput(editor, SettingFieldId::StableFailureChecks, "stable failure checks",
           std::to_string(s.population_stable_failure_checks), false);
  addInput(editor, SettingFieldId::PopulationChangeCooldown,
           "population change cooldown",
           std::to_string(s.population_change_cooldown), false);
  if (s.algorithm == TrainingAlgorithm::Neat) {
    addInput(editor, SettingFieldId::NeatMutationSigma, "mutation sigma",
             formatDouble(s.neat_mutation_sigma), true);
    addInput(editor, SettingFieldId::NeatMutationProb, "mutation probability",
             formatDouble(s.neat_mutation_prob), true);
    addInput(editor, SettingFieldId::NeatCompatibilityThreshold,
             "compatibility threshold",
             formatDouble(s.neat_compatibility_threshold), true);
    addInput(editor, SettingFieldId::NeatSurvivalRate, "survival rate",
             formatDouble(s.neat_survival_rate), true);
    addInput(editor, SettingFieldId::NeatAddNodeProb, "add node probability",
             formatDouble(s.neat_add_node_prob), true);
    addInput(editor, SettingFieldId::NeatAddConnectionProb,
             "add connection probability",
             formatDouble(s.neat_add_connection_prob), true);
    addInput(editor, SettingFieldId::NeatSparseConnectionProb,
             "sparse connection probability",
             formatDouble(s.neat_sparse_connection_prob), true);
    addInput(editor, SettingFieldId::NeatInputProbeProb,
             "input probe probability", formatDouble(s.neat_input_probe_prob),
             true);
    addInput(editor, SettingFieldId::NeatSparseInputProbeProb,
             "sparse input probe probability",
             formatDouble(s.neat_sparse_input_probe_prob), true);
  } else {
    addInput(editor, SettingFieldId::ClassicLearningRate, "learning rate",
             formatDouble(s.classic_learning_rate, 4), true);
    addInput(editor, SettingFieldId::ClassicParents, "parents",
             std::to_string(s.classic_parents), false);
    addInput(editor, SettingFieldId::ClassicHidden1, "hidden layer 1",
             std::to_string(s.classic_hidden1), false);
    addInput(editor, SettingFieldId::ClassicHidden2, "hidden layer 2",
             std::to_string(s.classic_hidden2), false);
    addInput(editor, SettingFieldId::ClassicMutationSigma, "mutation sigma",
             formatDouble(s.classic_mutation_sigma), true);
    addInput(editor, SettingFieldId::ClassicMutationProb,
             "mutation probability", formatDouble(s.classic_mutation_prob),
             true);
  }
}

bool parseTextInt(const std::string &text, int &out) {
  if (text.empty())
    return false;
  char *end = nullptr;
  long value = std::strtol(text.c_str(), &end, 10);
  if (end == text.c_str())
    return false;
  out = (int)value;
  return true;
}

bool parseTextDouble(std::string text, double &out) {
  if (text.empty())
    return false;
  for (char &c : text)
    if (c == ',')
      c = '.';
  char *end = nullptr;
  double value = std::strtod(text.c_str(), &end);
  if (end == text.c_str())
    return false;
  out = value;
  return true;
}

void applySettingsInputs(SettingsEditor &editor) {
  for (const auto &input : editor.inputs) {
    int iv = 0;
    double dv = 0.0;
    switch (input.id) {
    case SettingFieldId::Population:
      if (parseTextInt(input.text, iv))
        editor.settings.population = iv;
      break;
    case SettingFieldId::ActiveTests:
      if (parseTextInt(input.text, iv))
        editor.settings.initial_active_tests = iv;
      break;
    case SettingFieldId::MaxGenerations:
      if (parseTextInt(input.text, iv))
        editor.settings.max_generations = iv;
      break;
    case SettingFieldId::Threads:
      if (parseTextInt(input.text, iv))
        editor.settings.requested_threads = iv;
      break;
    case SettingFieldId::MaxPopulation:
      if (parseTextInt(input.text, iv))
        editor.settings.max_population = iv;
      break;
    case SettingFieldId::FailureLogInterval:
      if (parseTextInt(input.text, iv))
        editor.settings.failure_log_interval = iv;
      break;
    case SettingFieldId::PopulationStallGenerations:
      if (parseTextInt(input.text, iv))
        editor.settings.population_stall_generations = iv;
      break;
    case SettingFieldId::StableFailureChecks:
      if (parseTextInt(input.text, iv))
        editor.settings.population_stable_failure_checks = iv;
      break;
    case SettingFieldId::PopulationChangeCooldown:
      if (parseTextInt(input.text, iv))
        editor.settings.population_change_cooldown = iv;
      break;
    case SettingFieldId::NeatMutationSigma:
      if (parseTextDouble(input.text, dv))
        editor.settings.neat_mutation_sigma = dv;
      break;
    case SettingFieldId::NeatMutationProb:
      if (parseTextDouble(input.text, dv))
        editor.settings.neat_mutation_prob = dv;
      break;
    case SettingFieldId::NeatCompatibilityThreshold:
      if (parseTextDouble(input.text, dv))
        editor.settings.neat_compatibility_threshold = dv;
      break;
    case SettingFieldId::NeatSurvivalRate:
      if (parseTextDouble(input.text, dv))
        editor.settings.neat_survival_rate = dv;
      break;
    case SettingFieldId::NeatAddNodeProb:
      if (parseTextDouble(input.text, dv))
        editor.settings.neat_add_node_prob = dv;
      break;
    case SettingFieldId::NeatAddConnectionProb:
      if (parseTextDouble(input.text, dv))
        editor.settings.neat_add_connection_prob = dv;
      break;
    case SettingFieldId::NeatSparseConnectionProb:
      if (parseTextDouble(input.text, dv))
        editor.settings.neat_sparse_connection_prob = dv;
      break;
    case SettingFieldId::NeatInputProbeProb:
      if (parseTextDouble(input.text, dv))
        editor.settings.neat_input_probe_prob = dv;
      break;
    case SettingFieldId::NeatSparseInputProbeProb:
      if (parseTextDouble(input.text, dv))
        editor.settings.neat_sparse_input_probe_prob = dv;
      break;
    case SettingFieldId::ClassicLearningRate:
      if (parseTextDouble(input.text, dv))
        editor.settings.classic_learning_rate = dv;
      break;
    case SettingFieldId::ClassicParents:
      if (parseTextInt(input.text, iv))
        editor.settings.classic_parents = iv;
      break;
    case SettingFieldId::ClassicHidden1:
      if (parseTextInt(input.text, iv))
        editor.settings.classic_hidden1 = iv;
      break;
    case SettingFieldId::ClassicHidden2:
      if (parseTextInt(input.text, iv))
        editor.settings.classic_hidden2 = iv;
      break;
    case SettingFieldId::ClassicMutationSigma:
      if (parseTextDouble(input.text, dv))
        editor.settings.classic_mutation_sigma = dv;
      break;
    case SettingFieldId::ClassicMutationProb:
      if (parseTextDouble(input.text, dv))
        editor.settings.classic_mutation_prob = dv;
      break;
    }
  }
}

void drawTextBox(sf::RenderWindow &w, const sf::Font *font, SettingInput &input,
                 bool active, sf::Vector2f mouse) {
  sf::Color outline =
      active ? sf::Color(238, 184, 82)
             : contains(input.rect, mouse) ? sf::Color(88, 100, 132)
                                           : sf::Color(54, 58, 74);
  drawText(w, font, input.label, input.rect.x - 330.f, input.rect.y + 7.f, 13,
           sf::Color(198, 204, 220));
  drawRect(w, input.rect.x, input.rect.y, input.rect.w, input.rect.h,
           sf::Color(24, 26, 36), outline, 1.f);
  std::string shown = input.text;
  if (active)
    shown += "|";
  drawText(w, font, shown, input.rect.x + 10.f, input.rect.y + 6.f, 13,
           sf::Color(232, 236, 246));
}

bool canAppendToInput(const SettingInput &input, char c) {
  if (std::isdigit((unsigned char)c))
    return true;
  if (c == '-' && input.text.empty())
    return true;
  if (!input.decimal)
    return false;
  if ((c == '.' || c == ',') && input.text.find('.') == std::string::npos &&
      input.text.find(',') == std::string::npos)
    return true;
  return false;
}

void handleSettingsText(SettingsEditor &editor, uint32_t unicode) {
  if (editor.active_input < 0 ||
      editor.active_input >= (int)editor.inputs.size())
    return;

  auto &input = editor.inputs[(size_t)editor.active_input];
  if (unicode == 8) {
    if (!input.text.empty())
      input.text.pop_back();
    return;
  }
  if (unicode == 10 || unicode == 13) {
    editor.active_input = -1;
    applySettingsInputs(editor);
    return;
  }
  if (unicode < 32 || unicode > 126 || input.text.size() >= 24)
    return;

  char c = (char)unicode;
  if (canAppendToInput(input, c))
    input.text.push_back(c);
}

void regenerateTests(AntoninaAPI &anim_api, LiveStats &stats) {
  auto tests = TestGenerator::generateBaseTests();
  std::string error;
  if (!TestGenerator::writeTests("Test0.csv", tests, &error)) {
    stats.appendLog("tests: regeneration failed: " + error);
    return;
  }

  std::string loaded_path;
  if (!anim_api.reloadTests(&error, &loaded_path)) {
    stats.appendLog("tests: wrote Test0.csv but reload failed: " + error);
    return;
  }

  std::error_code ec;
  std::filesystem::remove("test_weights.csv", ec);

  auto counts = TestGenerator::categoryCounts(tests);
  stats.appendLog("tests: regenerated " + std::to_string(tests.size()) +
                  " cases to Test0.csv");
  stats.appendLog("tests: reset test_weights.csv");
  std::string summary = "tests:";
  for (int i = 0; i < TestGenerator::CATEGORY_COUNT; ++i) {
    summary += ' ';
    summary += TestGenerator::categoryName(i);
    summary += '=';
    summary += std::to_string(counts[i]);
  }
  stats.appendLog(summary);
  if (!loaded_path.empty())
    stats.appendLog("tests: animation reloaded from " + loaded_path);
}

bool startsWith(const std::string &s, const char *prefix) {
  const size_t n = std::strlen(prefix);
  return s.size() >= n && s.compare(0, n, prefix) == 0;
}

std::vector<ModelCheckpointOption>
modelCheckpointOptions(TrainingAlgorithm algorithm) {
  const char *prefix = algorithm == TrainingAlgorithm::Neat
                           ? "population_"
                           : "classic_population_";
  const std::filesystem::path dirs[] = {
      "models",
      "x64/Release/models",
      "x64/Debug/models",
      "build-test/Release/models",
      "build-test/Debug/models",
  };

  std::vector<ModelCheckpointOption> options;
  for (const auto &dir : dirs) {
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec))
      continue;
    for (std::filesystem::directory_iterator it(dir, ec), end;
         !ec && it != end; it.increment(ec)) {
      const auto &entry = *it;
      std::error_code item_ec;
      if (!entry.is_regular_file(item_ec))
        continue;
      const auto path = entry.path();
      const std::string name = path.filename().string();
      if (!startsWith(name, prefix) || path.extension() != ".csv")
        continue;

      const auto write_time = entry.last_write_time(item_ec);
      if (item_ec)
        continue;
      auto absolute = std::filesystem::absolute(path, item_ec);
      const std::string full_path =
          item_ec ? path.string() : absolute.string();
      const std::string parent =
          path.parent_path().generic_string().empty()
              ? std::string(".")
              : path.parent_path().generic_string();
      options.push_back(
          {full_path, name + "  [" + parent + "]", write_time});
    }
  }

  std::sort(options.begin(), options.end(),
            [](const auto &a, const auto &b) {
              if (a.write_time != b.write_time)
                return a.write_time > b.write_time;
              return a.label > b.label;
            });
  return options;
}

std::string modelFileLabel(const std::string &file) {
  if (file.empty())
    return "model: not selected";
  return "model: " + std::filesystem::path(file).filename().string();
}

std::string trimLogLine(const std::string &line, size_t max_chars);

UiRect modelPickerRect(const UiRect &panel) {
  const float w = std::min(820.f, panel.w - 80.f);
  const float h = std::min(520.f, panel.h - 110.f);
  return {panel.x + (panel.w - w) * 0.5f, panel.y + 72.f, w, h};
}

int modelPickerVisibleRows(const UiRect &picker) {
  return std::max(1, (int)((picker.h - 96.f) / 30.f));
}

void clampModelPicker(SettingsEditor &editor, const UiRect &picker) {
  const int max_first =
      std::max(0, (int)editor.model_options.size() -
                       modelPickerVisibleRows(picker));
  editor.model_picker_first =
      std::clamp(editor.model_picker_first, 0, max_first);
}

void drawModelPicker(sf::RenderWindow &w, const sf::Font *font,
                     SettingsEditor &editor, sf::Vector2f mouse,
                     const UiRect &panel) {
  if (!editor.model_picker_open)
    return;

  drawRect(w, panel.x, panel.y, panel.w, panel.h, sf::Color(4, 5, 9, 170));
  const UiRect picker = modelPickerRect(panel);
  clampModelPicker(editor, picker);
  drawRect(w, picker.x, picker.y, picker.w, picker.h, sf::Color(18, 19, 28),
           sf::Color(82, 90, 118), 1.f);
  drawText(w, font, "choose model checkpoint", picker.x + 18.f,
           picker.y + 16.f, 20, sf::Color(232, 236, 246));
  drawText(w, font, "sorted by file modification date", picker.x + 18.f,
           picker.y + 44.f, 13, sf::Color(150, 156, 176));

  UiRect clear{picker.x + picker.w - 214.f, picker.y + 16.f, 88.f, 28.f};
  UiRect close{picker.x + picker.w - 112.f, picker.y + 16.f, 88.f, 28.f};
  drawButton(w, font, clear, "clear", contains(clear, mouse),
             sf::Color(46, 48, 62), sf::Color(88, 96, 118));
  drawButton(w, font, close, "close", contains(close, mouse),
             sf::Color(46, 48, 62), sf::Color(88, 96, 118));

  if (editor.model_options.empty()) {
    drawText(w, font, "no population checkpoints found", picker.x + 18.f,
             picker.y + 86.f, 14, sf::Color(190, 196, 215));
    return;
  }

  const float list_x = picker.x + 18.f;
  const float list_y = picker.y + 78.f;
  const float row_h = 30.f;
  const float list_w = picker.w - 48.f;
  const int visible = modelPickerVisibleRows(picker);
  const int end =
      std::min((int)editor.model_options.size(),
               editor.model_picker_first + visible);
  for (int i = editor.model_picker_first; i < end; ++i) {
    float y = list_y + (i - editor.model_picker_first) * row_h;
    UiRect row{list_x, y, list_w, row_h - 3.f};
    const bool selected = editor.model_options[(size_t)i].path ==
                          editor.load_file;
    const bool hover = contains(row, mouse);
    sf::Color fill = selected ? sf::Color(44, 74, 116)
                              : hover ? sf::Color(34, 38, 52)
                                      : sf::Color(24, 26, 36);
    drawRect(w, row.x, row.y, row.w, row.h, fill, sf::Color(54, 58, 74), 1.f);
    drawText(w, font,
             trimLogLine(editor.model_options[(size_t)i].label, 96),
             row.x + 10.f, row.y + 6.f, 13, sf::Color(218, 224, 238));
  }

  const int max_first =
      std::max(0, (int)editor.model_options.size() - visible);
  const UiRect track{picker.x + picker.w - 16.f, list_y, 6.f,
                     std::max(16.f, visible * row_h - 3.f)};
  drawRect(w, track.x, track.y, track.w, track.h, sf::Color(29, 31, 42));
  if (max_first > 0) {
    const float thumb_h = std::max(
        28.f, track.h * ((float)visible / (float)editor.model_options.size()));
    const float travel = std::max(1.f, track.h - thumb_h);
    const float t = (float)editor.model_picker_first / (float)max_first;
    drawRect(w, track.x - 2.f, track.y + travel * t, track.w + 4.f, thumb_h,
             sf::Color(86, 94, 124));
  }
}

bool handleModelPickerClick(SettingsEditor &editor, sf::Vector2f pos,
                            const UiRect &panel, LiveStats &stats) {
  if (!editor.model_picker_open)
    return false;

  const UiRect picker = modelPickerRect(panel);
  UiRect clear{picker.x + picker.w - 214.f, picker.y + 16.f, 88.f, 28.f};
  UiRect close{picker.x + picker.w - 112.f, picker.y + 16.f, 88.f, 28.f};
  if (contains(clear, pos)) {
    editor.load_file.clear();
    editor.model_picker_open = false;
    stats.appendLog("viewer: model checkpoint cleared");
    return true;
  }
  if (contains(close, pos) || !contains(picker, pos)) {
    editor.model_picker_open = false;
    return true;
  }

  const float list_x = picker.x + 18.f;
  const float list_y = picker.y + 78.f;
  const float row_h = 30.f;
  const float list_w = picker.w - 48.f;
  const int visible = modelPickerVisibleRows(picker);
  const int end =
      std::min((int)editor.model_options.size(),
               editor.model_picker_first + visible);
  for (int i = editor.model_picker_first; i < end; ++i) {
    UiRect row{list_x, list_y + (i - editor.model_picker_first) * row_h,
               list_w, row_h - 3.f};
    if (!contains(row, pos))
      continue;
    editor.load_file = editor.model_options[(size_t)i].path;
    editor.model_picker_open = false;
    stats.appendLog("viewer: selected checkpoint " + editor.load_file);
    return true;
  }

  return true;
}

void scrollModelPicker(SettingsEditor &editor, const UiRect &panel,
                       float delta) {
  if (!editor.model_picker_open)
    return;
  const UiRect picker = modelPickerRect(panel);
  const int visible = modelPickerVisibleRows(picker);
  const int max_first =
      std::max(0, (int)editor.model_options.size() - visible);
  int first = editor.model_picker_first - (int)std::round(delta * 3.f);
  editor.model_picker_first = std::clamp(first, 0, max_first);
}

void drawSettingsWindow(sf::RenderWindow &w, const sf::Font *font,
                        SettingsEditor &editor, sf::Vector2f mouse,
                        const UiRect &panel) {
  const TrainingSettings &settings = editor.settings;
  const float X = panel.x, Y = panel.y, WW = panel.w, HH = panel.h;
  drawRect(w, X, Y, WW, HH, sf::Color(16, 17, 24), sf::Color(64, 70, 92),
           1.f);
  drawText(w, font, "training settings", X + 28.f, Y + 22.f, 24,
           sf::Color(232, 236, 246));
  drawText(w, font, "choose algorithm and population before training starts",
           X + 30.f, Y + 58.f, 14, sf::Color(150, 156, 176));

  UiRect neat{X + 30.f, Y + 94.f, 180.f, 34.f};
  UiRect classic{X + 220.f, Y + 94.f, 210.f, 34.f};
  drawButton(w, font, neat, "NEAT",
             contains(neat, mouse) ||
                 settings.algorithm == TrainingAlgorithm::Neat,
             settings.algorithm == TrainingAlgorithm::Neat
                 ? sf::Color(52, 95, 156)
                 : sf::Color(34, 38, 52));
  drawButton(w, font, classic, "classic evolution",
             contains(classic, mouse) ||
                 settings.algorithm == TrainingAlgorithm::Classic,
             settings.algorithm == TrainingAlgorithm::Classic
                 ? sf::Color(52, 95, 156)
                 : sf::Color(34, 38, 52));

  float row_y = Y + 146.f;
  auto drawToggle = [&](const std::string &label, bool enabled) {
    drawText(w, font, label, X + 34.f, row_y + 8.f, 14,
             sf::Color(198, 204, 220));
    UiRect toggle{X + 520.f, row_y, 140.f, 30.f};
    drawButton(w, font, toggle, enabled ? "on" : "off",
               contains(toggle, mouse),
               enabled ? sf::Color(45, 114, 82) : sf::Color(74, 44, 50),
               sf::Color(88, 96, 118));
    row_y += 40.f;
  };

  drawToggle("autosave best model", settings.autosave_best);
  drawToggle("adaptive population", settings.adaptive_population);
  drawToggle("test weights", settings.test_weighting);

  row_y += 6.f;
  const float buttons_y = Y + HH - 56.f;
  const int split_after = 9;
  const int input_count = (int)editor.inputs.size();
  const int section_count = input_count > split_after ? 1 : 0;
  const float section_h = section_count ? 42.f : 0.f;
  const float input_area = std::max(0.f, buttons_y - row_y - 8.f - section_h);
  const float row_step =
      input_count > 0
          ? std::clamp(input_area / (float)input_count, 24.f, 34.f)
          : 34.f;
  const float input_h = std::clamp(row_step - 4.f, 22.f, 30.f);
  const float input_x = std::min(X + 520.f, X + WW - 300.f);
  for (int i = 0; i < (int)editor.inputs.size(); ++i) {
    if (i == split_after) {
      row_y += 8.f;
      drawText(w, font,
               settings.algorithm == TrainingAlgorithm::Neat
                   ? "NEAT constants"
                   : "classic constants",
               X + 34.f, row_y, 16, sf::Color(226, 230, 240));
      row_y += 34.f;
    }
    auto &input = editor.inputs[(size_t)i];
    input.rect = {input_x, row_y, 260.f, input_h};
    drawTextBox(w, font, input, i == editor.active_input, mouse);
    row_y += row_step;
  }

  drawText(w, font, trimLogLine(modelFileLabel(editor.load_file), 92),
           X + 34.f, buttons_y - 24.f, 13, sf::Color(150, 156, 176));

  UiRect load{X + WW - 850.f, buttons_y, 250.f, 40.f};
  drawButton(w, font, load, "загрузить модель", contains(load, mouse),
             sf::Color(44, 70, 112), sf::Color(110, 145, 205));

  UiRect regen{X + WW - 560.f, buttons_y, 270.f, 40.f};
  drawButton(w, font, regen, "перегенерировать тесты", contains(regen, mouse),
             sf::Color(78, 72, 42), sf::Color(205, 184, 96));

  UiRect start{X + WW - 260.f, buttons_y, 220.f, 40.f};
  drawButton(w, font, start, "start training", contains(start, mouse),
             sf::Color(42, 104, 72), sf::Color(120, 205, 150));

  drawModelPicker(w, font, editor, mouse, panel);
}

bool applySettingsClick(SettingsEditor &editor, sf::Vector2f pos,
                        LiveStats &stats, AntoninaAPI &anim_api,
                        const UiRect &panel) {
  if (handleModelPickerClick(editor, pos, panel, stats))
    return true;

  const float X = panel.x, Y = panel.y, WW = panel.w, HH = panel.h;
  UiRect neat{X + 30.f, Y + 94.f, 180.f, 34.f};
  UiRect classic{X + 220.f, Y + 94.f, 210.f, 34.f};
  UiRect load{X + WW - 850.f, Y + HH - 56.f, 250.f, 40.f};
  UiRect regen{X + WW - 560.f, Y + HH - 56.f, 270.f, 40.f};
  UiRect start{X + WW - 260.f, Y + HH - 56.f, 220.f, 40.f};

  if (contains(neat, pos)) {
    applySettingsInputs(editor);
    editor.settings.algorithm = TrainingAlgorithm::Neat;
    editor.load_file.clear();
    editor.model_picker_open = false;
    rebuildSettingsInputs(editor);
    return true;
  }
  if (contains(classic, pos)) {
    applySettingsInputs(editor);
    editor.settings.algorithm = TrainingAlgorithm::Classic;
    editor.load_file.clear();
    editor.model_picker_open = false;
    rebuildSettingsInputs(editor);
    return true;
  }
  if (contains(load, pos)) {
    applySettingsInputs(editor);
    editor.model_options = modelCheckpointOptions(editor.settings.algorithm);
    editor.model_picker_first = 0;
    editor.model_picker_open = true;
    if (editor.model_options.empty()) {
      stats.appendLog(std::string("viewer: no ") +
                      trainingAlgorithmName(editor.settings.algorithm) +
                      " population checkpoint found in models");
    }
    return true;
  }
  if (contains(start, pos)) {
    applySettingsInputs(editor);
    stats.requestStart(editor.settings, editor.load_file);
    stats.appendLog(std::string("viewer: start requested, algorithm=") +
                    trainingAlgorithmName(editor.settings.algorithm));
    if (!editor.load_file.empty())
      stats.appendLog("viewer: loading checkpoint " + editor.load_file);
    return true;
  }
  if (contains(regen, pos)) {
    applySettingsInputs(editor);
    regenerateTests(anim_api, stats);
    return true;
  }

  for (int i = 0; i < (int)editor.inputs.size(); ++i) {
    if (contains(editor.inputs[(size_t)i].rect, pos)) {
      editor.active_input = i;
      return true;
    }
  }

  editor.active_input = -1;
  applySettingsInputs(editor);

  float row_y = Y + 146.f;
  auto toggleHit = [&](bool &value) {
    UiRect toggle{X + 520.f, row_y, 140.f, 30.f};
    bool hit = contains(toggle, pos);
    if (hit)
      value = !value;
    row_y += 40.f;
    return hit;
  };

  if (toggleHit(editor.settings.autosave_best))
    return true;
  if (toggleHit(editor.settings.adaptive_population))
    return true;
  if (toggleHit(editor.settings.test_weighting))
    return true;

  return false;
}

std::string trimLogLine(const std::string &line, size_t max_chars) {
  if (line.size() <= max_chars)
    return line;
  if (max_chars <= 3)
    return line.substr(0, max_chars);
  return line.substr(0, max_chars - 3) + "...";
}

size_t logWrapChars(float ww) {
  return (size_t)std::max(18, (int)((ww - 38.f) / 6.2f));
}

int logVisibleLines(float hh) {
  return std::max(1, (int)((hh - 44.f) / 16.f));
}

std::vector<std::string> wrapLogLine(const std::string &line,
                                     size_t max_chars) {
  std::vector<std::string> out;
  if (line.empty()) {
    out.push_back("");
    return out;
  }

  size_t pos = 0;
  while (pos < line.size()) {
    while (pos < line.size() &&
           std::isspace((unsigned char)line[pos]) && line[pos] != '\n')
      ++pos;
    if (pos >= line.size())
      break;

    size_t remain = line.size() - pos;
    if (remain <= max_chars) {
      out.push_back(line.substr(pos));
      break;
    }

    size_t hard_end = pos + max_chars;
    size_t split = line.rfind(' ', hard_end);
    if (split == std::string::npos || split <= pos + max_chars / 3)
      split = hard_end;

    out.push_back(line.substr(pos, split - pos));
    pos = split;
  }

  if (out.empty())
    out.push_back("");
  return out;
}

std::vector<std::string> wrappedLogLines(const std::vector<std::string> &logs,
                                         size_t max_chars) {
  std::vector<std::string> lines;
  lines.reserve(logs.size() * 2);
  for (const auto &log : logs) {
    auto parts = wrapLogLine(log, max_chars);
    for (auto &part : parts)
      lines.push_back(std::move(part));
  }
  return lines;
}

int maxLogFirstLine(const std::vector<std::string> &logs, float ww, float hh) {
  auto lines = wrappedLogLines(logs, logWrapChars(ww));
  return std::max(0, (int)lines.size() - logVisibleLines(hh));
}

int resolvedLogFirstLine(int requested_first, int max_first) {
  if (requested_first < 0)
    return max_first;
  return std::clamp(requested_first, 0, max_first);
}

void setLogFirstLine(int &requested_first, int first, int max_first) {
  first = std::clamp(first, 0, max_first);
  requested_first = first >= max_first ? -1 : first;
}

UiRect logSliderTrack(const UiRect &r) {
  return {r.x + r.w - 14.f, r.y + 34.f, 6.f, std::max(16.f, r.h - 50.f)};
}

void updateLogScrollFromSlider(float mouse_y, const std::vector<std::string> &logs,
                               const UiRect &r, int &first_line) {
  const int max_first = maxLogFirstLine(logs, r.w, r.h);
  if (max_first <= 0) {
    first_line = -1;
    return;
  }

  const int visible = logVisibleLines(r.h);
  const auto lines = wrappedLogLines(logs, logWrapChars(r.w));
  const UiRect track = logSliderTrack(r);
  const float thumb_h =
      std::max(28.f, track.h * ((float)visible / (float)lines.size()));
  const float travel = std::max(1.f, track.h - thumb_h);
  float t = (mouse_y - track.y - thumb_h * 0.5f) / travel;
  t = std::clamp(t, 0.f, 1.f);
  setLogFirstLine(first_line, (int)std::round(t * max_first), max_first);
}

void drawLogColumn(sf::RenderWindow &w, const sf::Font *font,
                   const std::vector<std::string> &logs, float x, float y,
                   float ww, float hh, int &first_line) {
  drawRect(w, x, y, ww, hh, sf::Color(18, 18, 26), sf::Color(54, 58, 74),
           1.f);
  drawText(w, font, "log output", x + 10.f, y + 8.f, 14,
           sf::Color(205, 210, 225));

  const int line_h = 16;
  const int max_lines = logVisibleLines(hh);
  auto lines = wrappedLogLines(logs, logWrapChars(ww));
  const int max_first = std::max(0, (int)lines.size() - max_lines);
  const int start = resolvedLogFirstLine(first_line, max_first);
  if (first_line >= 0)
    first_line = start;

  float yy = y + 34.f;
  const int end = std::min((int)lines.size(), start + max_lines);
  for (int i = start; i < end; ++i) {
    drawText(w, font, lines[(size_t)i], x + 10.f, yy, 11,
             sf::Color(166, 174, 194));
    yy += (float)line_h;
  }

  const UiRect track = logSliderTrack({x, y, ww, hh});
  drawRect(w, track.x, track.y, track.w, track.h, sf::Color(29, 31, 42));
  if ((int)lines.size() > max_lines) {
    const float thumb_h =
        std::max(28.f, track.h * ((float)max_lines / (float)lines.size()));
    const float travel = std::max(1.f, track.h - thumb_h);
    const float t = max_first <= 0 ? 1.f : (float)start / (float)max_first;
    drawRect(w, track.x - 2.f, track.y + travel * t, track.w + 4.f, thumb_h,
             sf::Color(86, 94, 124));
  }
}

size_t diagnosticsWrapChars(float ww) {
  return (size_t)std::max(24, (int)((ww - 38.f) / 6.4f));
}

int diagnosticsVisibleLines(float hh) {
  return std::max(1, (int)((hh - 44.f) / 18.f));
}

int maxDiagnosticsFirstLine(const std::vector<std::string> &lines, float ww,
                            float hh) {
  auto wrapped = wrappedLogLines(lines, diagnosticsWrapChars(ww));
  return std::max(0, (int)wrapped.size() - diagnosticsVisibleLines(hh));
}

void updateDiagnosticsScrollFromSlider(float mouse_y,
                                       const std::vector<std::string> &lines,
                                       const UiRect &r, int &first_line) {
  const int max_first = maxDiagnosticsFirstLine(lines, r.w, r.h);
  if (max_first <= 0) {
    first_line = -1;
    return;
  }

  const int visible = diagnosticsVisibleLines(r.h);
  const auto wrapped = wrappedLogLines(lines, diagnosticsWrapChars(r.w));
  const UiRect track = logSliderTrack(r);
  const float thumb_h =
      std::max(28.f, track.h * ((float)visible / (float)wrapped.size()));
  const float travel = std::max(1.f, track.h - thumb_h);
  float t = (mouse_y - track.y - thumb_h * 0.5f) / travel;
  t = std::clamp(t, 0.f, 1.f);
  setLogFirstLine(first_line, (int)std::round(t * max_first), max_first);
}

void drawInputDiagnostics(sf::RenderWindow &w, const sf::Font *font,
                          const std::vector<std::string> &lines, float x,
                          float y, float ww, float hh, int &first_line) {
  drawRect(w, x, y, ww, hh, sf::Color(18, 18, 26), sf::Color(54, 58, 74),
           1.f);
  drawText(w, font, "diagnostic output", x + 10.f, y + 8.f, 14,
           sf::Color(205, 210, 225));

  if (lines.empty()) {
    drawText(w, font, "waiting for first diagnostic sample", x + 10.f,
             y + 38.f, 13, sf::Color(145, 150, 170));
    return;
  }

  const int line_h = 18;
  const int max_lines = diagnosticsVisibleLines(hh);
  auto wrapped = wrappedLogLines(lines, diagnosticsWrapChars(ww));
  const int max_first = std::max(0, (int)wrapped.size() - max_lines);
  const int start = resolvedLogFirstLine(first_line, max_first);
  if (first_line >= 0)
    first_line = start;

  float yy = y + 36.f;
  const int end = std::min((int)wrapped.size(), start + max_lines);
  for (int i = start; i < end; ++i) {
    drawText(w, font, wrapped[(size_t)i], x + 10.f, yy, 12,
             sf::Color(166, 174, 194));
    yy += (float)line_h;
  }

  const UiRect track = logSliderTrack({x, y, ww, hh});
  drawRect(w, track.x, track.y, track.w, track.h, sf::Color(29, 31, 42));
  if ((int)wrapped.size() > max_lines) {
    const float thumb_h =
        std::max(28.f, track.h * ((float)max_lines / (float)wrapped.size()));
    const float travel = std::max(1.f, track.h - thumb_h);
    const float t = max_first <= 0 ? 1.f : (float)start / (float)max_first;
    drawRect(w, track.x - 2.f, track.y + travel * t, track.w + 4.f, thumb_h,
             sf::Color(86, 94, 124));
  }
}

sf::Color cellColor(char c) {
  switch (c) {
  case '.':
    return sf::Color(39, 41, 52);
  case '#':
    return sf::Color(92, 96, 112);
  case 'O':
    return sf::Color(70, 116, 220);
  case '@':
    return sf::Color(226, 202, 75);
  case 'a':
    return sf::Color(230, 154, 54);
  case '%':
    return sf::Color(72, 205, 130);
  default:
    return sf::Color(30, 31, 40);
  }
}

struct Episode {
  bool active = false;
  int test_idx = 0;
  int step = 0;
  char lab[8][8]{};
  int result = 0;
  int cooldown_frames = 0;
};

void resetEpisode(Episode &e, int idx, AntoninaAPI &api) {
  e.active = idx >= 0;
  e.test_idx = std::max(0, idx);
  e.step = 0;
  e.result = 0;
  e.cooldown_frames = 0;
  if (e.active) {
    api.initLabForAnim(idx, e.lab);
  } else {
    for (auto &row : e.lab)
      for (char &cell : row)
        cell = '.';
  }
}

void drawEpisode(sf::RenderWindow &w, const sf::Font *font, const Episode &e,
                 float x, float y, float side) {
  drawRect(w, x - 2, y - 2, side + 4, side + 26, sf::Color::Transparent,
           sf::Color(54, 58, 74), 1.f);
  float cell = side / 8.f;
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      drawRect(w, x + j * cell, y + i * cell, cell - 1, cell - 1,
               e.active ? cellColor(e.lab[i][j]) : sf::Color(28, 29, 38));
    }
  }

  char buf[96];
  if (!e.active)
    std::snprintf(buf, sizeof(buf), "test -");
  else if (e.result > 0)
    std::snprintf(buf, sizeof(buf), "#%d win @ %d", e.test_idx, e.result);
  else if (e.result < 0)
    std::snprintf(buf, sizeof(buf), "#%d fail", e.test_idx);
  else
    std::snprintf(buf, sizeof(buf), "#%d step %d", e.test_idx, e.step);

  drawText(w, font, buf, x, y + side + 4, 12,
           !e.active    ? sf::Color(120, 124, 140)
           : e.result > 0   ? sf::Color(116, 225, 148)
           : e.result < 0 ? sf::Color(230, 110, 112)
                          : sf::Color(205, 210, 225));
}

void drawActiveTestsPanel(sf::RenderWindow &w, const sf::Font *font,
                          Episode (&ep)[10], int active_tests,
                          const UiRect &r) {
  drawRect(w, r.x, r.y, r.w, r.h, sf::Color(18, 18, 26),
           sf::Color(54, 58, 74), 1.f);
  int shown = std::min(10, std::max(1, active_tests));
  int base = std::max(0, active_tests - shown);
  char tests_title[128];
  std::snprintf(tests_title, sizeof(tests_title), "active tests: %d..%d",
                base, base + shown - 1);
  drawText(w, font, tests_title, r.x + 10.f, r.y + 8.f, 14,
           sf::Color(205, 210, 225));

  const int columns = r.w < 245.f ? 1 : 2;
  const int rows = (10 + columns - 1) / columns;
  const float pad = 12.f;
  const float header_h = 40.f;
  const float slot_w = (r.w - pad * 2.f) / (float)columns;
  const float slot_h = (r.h - header_h - pad) / (float)rows;
  float side = std::min(90.f, std::min(slot_w - 14.f, slot_h - 28.f));
  side = std::max(36.f, side);

  for (int i = 0; i < 10; ++i) {
    int col = i % columns;
    int row = i / columns;
    float x = r.x + pad + col * slot_w + (slot_w - side) * 0.5f;
    float y = r.y + header_h + row * slot_h +
              std::max(0.f, (slot_h - side - 24.f) * 0.5f);
    drawEpisode(w, font, ep[i], x, y, side);
  }
}

struct TopologyPoint {
  float x = 0.f;
  float y = 0.f;
};

void placeVertical(const std::vector<int> &indices,
                   std::vector<TopologyPoint> &points, float x, float y0,
                   float y1) {
  if (indices.empty())
    return;
  if (indices.size() == 1) {
    points[indices[0]] = {x, (y0 + y1) * 0.5f};
    return;
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    float t = (float)i / (float)(indices.size() - 1);
    points[indices[i]] = {x, y0 + (y1 - y0) * t};
  }
}

void drawTopology(sf::RenderWindow &w, const sf::Font *font,
                  const Brain *brain, float x, float y, float ww, float hh) {
  drawRect(w, x, y, ww, hh, sf::Color(18, 18, 26), sf::Color(54, 58, 74),
           1.f);
  drawText(w, font, "current best topology", x + 10, y + 7, 14,
           sf::Color(205, 210, 225));

  const auto *genome = dynamic_cast<const NeatGenome *>(brain);
  if (!genome) {
    drawText(w, font, "waiting for NEAT genome", x + 10, y + 36, 13,
             sf::Color(145, 150, 170));
    return;
  }

  const auto &nodes = genome->nodes();
  const auto &connections = genome->connections();
  if (nodes.empty())
    return;

  char title[160];
  std::snprintf(title, sizeof(title), "%d nodes | %d active | %d/%d enabled",
                genome->nodeCount(), genome->activeConnectionCount(),
                genome->enabledConnectionCount(), genome->connectionCount());
  drawText(w, font, title, x + ww - 240, y + 7, 13,
           sf::Color(145, 150, 170));

  std::vector<int> inputs, bias, hidden, outputs;
  inputs.reserve(nodes.size());
  hidden.reserve(nodes.size());
  outputs.reserve(nodes.size());
  for (int i = 0; i < (int)nodes.size(); ++i) {
    if (nodes[i].type == 0)
      inputs.push_back(i);
    else if (nodes[i].type == 1)
      bias.push_back(i);
    else if (nodes[i].type == 2)
      hidden.push_back(i);
    else if (nodes[i].type == 3)
      outputs.push_back(i);
  }

  std::vector<TopologyPoint> pos(nodes.size());
  const float left = x + 34.f;
  const float right = x + ww - 34.f;
  const float top = y + 48.f;
  const float bottom = y + hh - 34.f;
  placeVertical(inputs, pos, left, top, bottom - 16.f);
  placeVertical(bias, pos, left + 18.f, bottom - 8.f, bottom - 8.f);
  placeVertical(outputs, pos, right, top + 45.f, bottom - 45.f);

  constexpr int HIDDEN_COLS = 10;
  std::array<std::vector<int>, HIDDEN_COLS> cols;
  for (int idx : hidden) {
    int col = (int)std::round(std::clamp(nodes[idx].level, 0.0, 1.0) *
                              (HIDDEN_COLS - 1));
    col = std::clamp(col, 1, HIDDEN_COLS - 2);
    cols[(size_t)col].push_back(idx);
  }
  for (int col = 1; col < HIDDEN_COLS - 1; ++col) {
    float xx = x + 70.f + (ww - 140.f) * (float)(col - 1) /
                              (float)(HIDDEN_COLS - 3);
    placeVertical(cols[(size_t)col], pos, xx, top + 10.f, bottom - 10.f);
  }

  struct EdgeRef {
    int index;
    double weight_abs;
  };
  std::vector<EdgeRef> edges;
  edges.reserve(connections.size());
  for (int i = 0; i < (int)connections.size(); ++i) {
    const auto &c = connections[i];
    if (!c.enabled || c.in_index < 0 || c.out_index < 0 ||
        c.in_index >= (int)pos.size() || c.out_index >= (int)pos.size())
      continue;
    edges.push_back({i, std::abs(c.weight)});
  }

  const int max_edges = 1200;
  if ((int)edges.size() > max_edges) {
    std::nth_element(edges.begin(), edges.begin() + max_edges, edges.end(),
                     [](const EdgeRef &a, const EdgeRef &b) {
                       return a.weight_abs > b.weight_abs;
                     });
    edges.resize(max_edges);
  }

  for (const auto &ref : edges) {
    const auto &c = connections[ref.index];
    const auto &a = pos[(size_t)c.in_index];
    const auto &b = pos[(size_t)c.out_index];
    int alpha = std::clamp(36 + (int)(std::abs(c.weight) * 38.0), 36, 210);
    sf::Color col = c.weight >= 0.0 ? sf::Color(236, 184, 82, alpha)
                                    : sf::Color(94, 152, 236, alpha);
    drawLine(w, a.x, a.y, b.x, b.y, col);
  }

  for (int i = 0; i < (int)nodes.size(); ++i) {
    sf::Color fill;
    float r = 3.f;
    if (nodes[i].type == 0) {
      fill = sf::Color(100, 128, 210);
      r = 2.1f;
    } else if (nodes[i].type == 1) {
      fill = sf::Color(220, 210, 120);
      r = 4.2f;
    } else if (nodes[i].type == 2) {
      fill = sf::Color(90, 205, 150);
      r = 4.4f;
    } else {
      fill = sf::Color(232, 116, 118);
      r = 6.4f;
    }
    drawCircle(w, pos[(size_t)i].x, pos[(size_t)i].y, r, fill,
               sf::Color(12, 12, 18), 1.f);
  }

  const char *out_labels[] = {"U", "R", "D", "L"};
  for (size_t i = 0; i < outputs.size() && i < 4; ++i) {
    const auto &p = pos[(size_t)outputs[i]];
    drawText(w, font, out_labels[i], p.x + 10.f, p.y - 8.f, 12,
             sf::Color(230, 232, 240));
  }

  drawText(w, font, "inputs", left - 18.f, bottom + 8.f, 11,
           sf::Color(145, 150, 170));
  drawText(w, font, "hidden", x + ww * 0.5f - 20.f, bottom + 8.f, 11,
           sf::Color(145, 150, 170));
  drawText(w, font, "outputs", right - 24.f, bottom + 8.f, 11,
           sf::Color(145, 150, 170));
}

void assignCurrentTests(Episode (&ep)[10], int active_tests,
                        AntoninaAPI &anim_api) {
  active_tests = std::clamp(active_tests, 1, AntoninaAPI::ALL_TESTS);
  int shown = std::min(10, active_tests);
  int base = std::max(0, active_tests - shown);
  for (int i = 0; i < 10; ++i) {
    int idx = i < shown ? base + i : -1;
    resetEpisode(ep[i], idx, anim_api);
  }
}

}

void viewerThread(LiveStats &stats, AntoninaAPI &anim_api) {
  constexpr unsigned W = 1920, H = 1080;
  const char *title_utf8 = "Antonina AI - training dashboard";
  sf::RenderWindow window(
      sf::VideoMode({W, H}),
      sf::String::fromUtf8(title_utf8, title_utf8 + std::strlen(title_utf8)));
  window.setFramerateLimit(60);

  sf::Font font;
  bool font_ok = loadFont(font);
  const sf::Font *fp = font_ok ? &font : nullptr;

  Episode ep[10];
  assignCurrentTests(ep, 1, anim_api);

  std::unique_ptr<Brain> best_local;
  int best_version_local = -1;
  int rendered_active_tests = -1;

  constexpr int STEP_MS = 110;
  auto last_step = std::chrono::steady_clock::now();

  std::vector<GenSample> snap;
  std::vector<std::string> logs;
  std::vector<std::string> diagnostics;
  SettingsEditor settings_editor;
  settings_editor.settings = stats.getDefaultSettings();
  rebuildSettingsInputs(settings_editor);
  int log_first_line = -1;
  int diagnostics_first_line = -1;
  bool log_dragging = false;
  bool diagnostics_dragging = false;

  while (window.isOpen() && stats.isRunning()) {
    sf::View ui_view = makeWindowView(window.getSize());
    window.setView(ui_view);
    DashboardLayout layout = makeDashboardLayout(window.getSize());
    CenterPanelLayout center_layout = makeCenterPanelLayout(layout);
    SettingsLayout settings_layout = makeSettingsLayout(window.getSize());
    auto mp =
        window.mapPixelToCoords(sf::Mouse::getPosition(window), ui_view);
    bool settings_mode = !stats.isTrainingStarted();
    UiRect active_log_rect =
        settings_mode ? settings_layout.logs : layout.logs;
    bool btn_hover = !settings_mode && contains(layout.save_button, mp);

    while (auto ev = window.pollEvent()) {
      if (ev->is<sf::Event::Closed>()) {
        window.close();
        stats.requestStop();
      } else if (auto *mb = ev->getIf<sf::Event::MouseButtonPressed>()) {
        auto pos = window.mapPixelToCoords(mb->position, ui_view);
        if (mb->button == sf::Mouse::Button::Left) {
          if (contains(logSliderTrack(active_log_rect), pos)) {
            log_dragging = true;
            updateLogScrollFromSlider(pos.y, logs, active_log_rect,
                                      log_first_line);
          } else if (!settings_mode &&
                     contains(logSliderTrack(center_layout.diagnostics), pos)) {
            diagnostics_dragging = true;
            updateDiagnosticsScrollFromSlider(pos.y, diagnostics,
                                              center_layout.diagnostics,
                                              diagnostics_first_line);
          } else if (settings_mode) {
            applySettingsClick(settings_editor, pos, stats, anim_api,
                               settings_layout.panel);
          } else if (contains(layout.save_button, pos)) {
            stats.requestSave();
          }
        }
      } else if (ev->is<sf::Event::MouseButtonReleased>()) {
        log_dragging = false;
        diagnostics_dragging = false;
      } else if (auto *move = ev->getIf<sf::Event::MouseMoved>()) {
        if (log_dragging) {
          auto pos = window.mapPixelToCoords(move->position, ui_view);
          updateLogScrollFromSlider(pos.y, logs, active_log_rect,
                                    log_first_line);
        } else if (diagnostics_dragging) {
          auto pos = window.mapPixelToCoords(move->position, ui_view);
          updateDiagnosticsScrollFromSlider(pos.y, diagnostics,
                                            center_layout.diagnostics,
                                            diagnostics_first_line);
        }
      } else if (auto *wheel = ev->getIf<sf::Event::MouseWheelScrolled>()) {
        auto pos = window.mapPixelToCoords(wheel->position, ui_view);
        if (settings_mode && settings_editor.model_picker_open &&
            contains(modelPickerRect(settings_layout.panel), pos)) {
          scrollModelPicker(settings_editor, settings_layout.panel,
                            wheel->delta);
        } else if (contains(active_log_rect, pos)) {
          const int max_first =
              maxLogFirstLine(logs, active_log_rect.w, active_log_rect.h);
          int first = resolvedLogFirstLine(log_first_line, max_first);
          first -= (int)std::round(wheel->delta * 4.f);
          setLogFirstLine(log_first_line, first, max_first);
        } else if (!settings_mode && contains(center_layout.diagnostics, pos)) {
          const int max_first = maxDiagnosticsFirstLine(
              diagnostics, center_layout.diagnostics.w,
              center_layout.diagnostics.h);
          int first = resolvedLogFirstLine(diagnostics_first_line, max_first);
          first -= (int)std::round(wheel->delta * 3.f);
          setLogFirstLine(diagnostics_first_line, first, max_first);
        }
      } else if (auto *text = ev->getIf<sf::Event::TextEntered>()) {
        if (settings_mode)
          handleSettingsText(settings_editor, text->unicode);
      }
    }

    if (settings_mode) {
      window.clear(sf::Color(11, 12, 18));
      window.setView(ui_view);
      drawSettingsWindow(window, fp, settings_editor, mp,
                         settings_layout.panel);
      stats.snapshotLogs(logs);
      drawLogColumn(window, fp, logs, settings_layout.logs.x,
                    settings_layout.logs.y, settings_layout.logs.w,
                    settings_layout.logs.h, log_first_line);
      window.display();
      continue;
    }

    stats.snapshotSamples(snap);
    stats.snapshotLogs(logs);
    stats.snapshotInputDiagnostics(diagnostics);

    bool best_changed = stats.snapshotBestIfNew(best_local, best_version_local);
    int active_tests = snap.empty() ? 1 : snap.back().active_tests;
    if (best_changed || active_tests != rendered_active_tests) {
      assignCurrentTests(ep, active_tests, anim_api);
      rendered_active_tests = active_tests;
    }

    auto now = std::chrono::steady_clock::now();
    if (best_version_local >= 0 && best_local &&
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_step)
                .count() >= STEP_MS) {
      last_step = now;
      for (int i = 0; i < 10; ++i) {
        if (!ep[i].active)
          continue;
        if (ep[i].result != 0) {
          if (++ep[i].cooldown_frames >= 10)
            resetEpisode(ep[i], ep[i].test_idx, anim_api);
          continue;
        }
        ep[i].step += 1;
        int r = anim_api.animStep(ep[i].lab, best_local.get(), ep[i].step);
        if (r != 0)
          ep[i].result = r;
        if (ep[i].step >= 40 && r == 0)
          ep[i].result = -1;
      }
    }

    window.clear(sf::Color(11, 12, 18));
    window.setView(ui_view);

    if (!snap.empty()) {
      const auto &s = snap.back();
      char hdr[320];
      std::snprintf(
          hdr, sizeof(hdr),
          "gen %d | best %s | avg %s | wins %d/%d | pop %d/%d | species %d | topo %.0fn/%.0fc active %.0fc | compat %.2f | weights %dx%.2f",
          s.gen, compactValue(s.max_fitness).c_str(),
          compactValue(s.avg_fitness).c_str(), s.best_wins, s.active_tests,
          s.population, s.population_target, s.species_count, s.avg_nodes,
          s.avg_connections, s.avg_active_connections,
          s.compatibility_threshold, s.weighted_tests,
          s.max_test_weight_x100 / 100.0);
      drawText(window, fp, hdr, 14, 10, 17, sf::Color(226, 230, 240));
    } else {
      drawText(window, fp, "waiting for trainer...", 14, 10, 17,
               sf::Color(180, 186, 205));
    }

    const float chart_gap = 8.f;
    const float chart_h =
        std::max(52.f, (layout.charts.h - chart_gap * 7.f) / 8.f);
    auto chartY = [&](int index) {
      return layout.charts.y + index * (chart_h + chart_gap);
    };

    drawLineChart(window, fp, snap,
                  [](const GenSample &s) { return s.max_fitness; },
                  "best fitness", layout.charts.x, chartY(0), layout.charts.w,
                  chart_h, sf::Color(116, 225, 148));
    drawLineChart(window, fp, snap,
                  [](const GenSample &s) { return s.avg_fitness; },
                  "average fitness", layout.charts.x, chartY(1),
                  layout.charts.w, chart_h, sf::Color(118, 178, 245));
    drawLineChart(window, fp, snap,
                  [](const GenSample &s) { return s.active_tests; },
                  "active tests", layout.charts.x, chartY(2), layout.charts.w,
                  chart_h, sf::Color(238, 184, 82));
    drawLineChart(window, fp, snap,
                  [](const GenSample &s) { return s.population; },
                  "population", layout.charts.x, chartY(3), layout.charts.w,
                  chart_h, sf::Color(196, 174, 255));
    drawLineChart(window, fp, snap,
                  [](const GenSample &s) { return s.species_count; },
                  "species", layout.charts.x, chartY(4), layout.charts.w,
                  chart_h, sf::Color(206, 136, 235));
    drawLineChart(window, fp, snap,
                  [](const GenSample &s) { return s.avg_active_connections; },
                  "active connections", layout.charts.x, chartY(5),
                  layout.charts.w, chart_h, sf::Color(96, 212, 196));
    drawLineChart(window, fp, snap,
                  [](const GenSample &s) { return s.fitness_ms; },
                  "fitness ms", layout.charts.x, chartY(6), layout.charts.w,
                  chart_h, sf::Color(232, 122, 122));
    drawLineChart(window, fp, snap,
                  [](const GenSample &s) { return s.evolution_ms; },
                  "evolution ms", layout.charts.x, chartY(7), layout.charts.w,
                  chart_h, sf::Color(245, 156, 96));

    drawTopology(window, fp, best_local.get(), layout.center.x,
                 layout.center.y, layout.center.w,
                 center_layout.topology_h);

    const GenSample fallback{};
    const GenSample &s = snap.empty() ? fallback : snap.back();
    const float metric_gap = 10.f;
    const float MH = center_layout.metric_h;
    const float MW = center_layout.metric_w;
    const float MX = layout.center.x;
    const float MY = center_layout.metric_y;
    drawMetric(window, fp, "curriculum",
               std::to_string(s.active_tests) + " / " +
                   std::to_string(AntoninaAPI::ALL_TESTS),
               MX, MY, MW, MH, sf::Color(238, 184, 82),
               (double)s.active_tests / AntoninaAPI::ALL_TESTS);
    drawMetric(window, fp, "best wins",
               std::to_string(s.best_wins) + " / " +
               std::to_string(std::max(1, s.active_tests)),
               MX + MW + metric_gap, MY, MW, MH, sf::Color(116, 225, 148),
               (double)s.best_wins / std::max(1, s.active_tests));
    drawMetric(window, fp, "species", std::to_string(s.species_count),
               MX + (MW + metric_gap) * 2.f, MY, MW, MH,
               sf::Color(206, 136, 235));
    drawMetric(window, fp, "nodes", compactValue(s.avg_nodes), MX,
               MY + MH + metric_gap, MW, MH, sf::Color(90, 205, 150));
    drawMetric(window, fp, "active / total links",
               compactValue(s.avg_active_connections) + " / " +
                   compactValue(s.avg_connections),
               MX + MW + metric_gap, MY + MH + metric_gap, MW, MH,
               sf::Color(96, 212, 196));
    drawMetric(window, fp, "fit / evo ms",
               std::to_string(s.fitness_ms) + " / " +
                   std::to_string(s.evolution_ms),
               MX + (MW + metric_gap) * 2.f, MY + MH + metric_gap, MW, MH,
               sf::Color(245, 156, 96));

    drawInputDiagnostics(window, fp, diagnostics, center_layout.diagnostics.x,
                         center_layout.diagnostics.y,
                         center_layout.diagnostics.w,
                         center_layout.diagnostics.h,
                         diagnostics_first_line);

    drawActiveTestsPanel(window, fp, ep, active_tests, layout.tests);

    drawLogColumn(window, fp, logs, layout.logs.x, layout.logs.y,
                  layout.logs.w, layout.logs.h, log_first_line);

    sf::Color btn_fill =
        btn_hover ? sf::Color(64, 106, 178) : sf::Color(38, 66, 118);
    drawRect(window, layout.save_button.x, layout.save_button.y,
             layout.save_button.w, layout.save_button.h, btn_fill,
             sf::Color(120, 160, 220), 1.f);
    drawText(window, fp, "save population", layout.save_button.x + 18.f,
             layout.save_button.y + 6.f, 13, sf::Color::White);

    window.display();
  }
}
