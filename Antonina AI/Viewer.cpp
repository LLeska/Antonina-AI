#include "Viewer.h"

#include "NeatEvolution.h"

#include <SFML/Graphics.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
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

sf::View makeLetterboxView(float virtual_w, float virtual_h,
                           sf::Vector2u window_size) {
  sf::View view({virtual_w * 0.5f, virtual_h * 0.5f},
                {virtual_w, virtual_h});
  if (window_size.x == 0 || window_size.y == 0)
    return view;

  float window_ratio = (float)window_size.x / (float)window_size.y;
  float target_ratio = virtual_w / virtual_h;
  float viewport_x = 0.f;
  float viewport_y = 0.f;
  float viewport_w = 1.f;
  float viewport_h = 1.f;

  if (window_ratio > target_ratio) {
    viewport_w = target_ratio / window_ratio;
    viewport_x = (1.f - viewport_w) * 0.5f;
  } else if (window_ratio < target_ratio) {
    viewport_h = window_ratio / target_ratio;
    viewport_y = (1.f - viewport_h) * 0.5f;
  }

  view.setViewport(sf::FloatRect({viewport_x, viewport_y},
                                 {viewport_w, viewport_h}));
  return view;
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
  std::snprintf(title, sizeof(title), "%d nodes | %d/%d enabled links",
                genome->nodeCount(), genome->enabledConnectionCount(),
                genome->connectionCount());
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
  constexpr unsigned W = 1600, H = 900;
  const char *title_utf8 = "Antonina AI - NEAT dashboard";
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

  while (window.isOpen() && stats.isRunning()) {
    sf::View ui_view = makeLetterboxView((float)W, (float)H, window.getSize());
    window.setView(ui_view);

    const float BTN_X = 12.f, BTN_Y = (float)H - 44.f;
    const float BTN_W = 176.f, BTN_H = 28.f;
    bool btn_hover = false;
    {
      auto mp = window.mapPixelToCoords(sf::Mouse::getPosition(window),
                                        ui_view);
      btn_hover = (mp.x >= BTN_X && mp.x <= BTN_X + BTN_W && mp.y >= BTN_Y &&
                   mp.y <= BTN_Y + BTN_H);
    }

    while (auto ev = window.pollEvent()) {
      if (ev->is<sf::Event::Closed>()) {
        window.close();
        stats.requestStop();
      } else if (auto *mb = ev->getIf<sf::Event::MouseButtonPressed>()) {
        auto pos = window.mapPixelToCoords(mb->position, ui_view);
        if (mb->button == sf::Mouse::Button::Left && pos.x >= BTN_X &&
            pos.x <= BTN_X + BTN_W && pos.y >= BTN_Y &&
            pos.y <= BTN_Y + BTN_H) {
          stats.requestSave();
        }
      }
    }

    stats.snapshotSamples(snap);

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
      char hdr[256];
      std::snprintf(
          hdr, sizeof(hdr),
          "gen %d | best %s | avg %s | wins %d/%d | species %d | topo %.0fn/%.0fc | compat %.2f",
          s.gen, compactValue(s.max_fitness).c_str(),
          compactValue(s.avg_fitness).c_str(), s.best_wins, s.active_tests,
          s.species_count, s.avg_nodes, s.avg_connections,
          s.compatibility_threshold);
      drawText(window, fp, hdr, 14, 10, 17, sf::Color(226, 230, 240));
    } else {
      drawText(window, fp, "waiting for trainer...", 14, 10, 17,
               sf::Color(180, 186, 205));
    }

    const float CX = 12.f;
    const float CY = 46.f;
    const float CW = 440.f;
    const float CH = 112.f;
    const float CG = 9.f;

    drawLineChart(window, fp, snap,
                  [](const GenSample &s) { return s.max_fitness; },
                  "best fitness", CX, CY, CW, CH, sf::Color(116, 225, 148));
    drawLineChart(window, fp, snap,
                  [](const GenSample &s) { return s.avg_fitness; },
                  "average fitness", CX, CY + (CH + CG), CW, CH,
                  sf::Color(118, 178, 245));
    drawLineChart(window, fp, snap,
                  [](const GenSample &s) { return s.active_tests; },
                  "active tests", CX, CY + 2 * (CH + CG), CW, CH,
                  sf::Color(238, 184, 82));
    drawLineChart(window, fp, snap,
                  [](const GenSample &s) { return s.species_count; },
                  "species", CX, CY + 3 * (CH + CG), CW, CH,
                  sf::Color(206, 136, 235));
    drawLineChart(window, fp, snap,
                  [](const GenSample &s) { return s.avg_connections; },
                  "avg connections", CX, CY + 4 * (CH + CG), CW, CH,
                  sf::Color(96, 212, 196));
    drawLineChart(window, fp, snap,
                  [](const GenSample &s) { return s.fitness_ms; },
                  "fitness ms", CX, CY + 5 * (CH + CG), CW, CH,
                  sf::Color(232, 122, 122));

    drawTopology(window, fp, best_local.get(), 470.f, 46.f, 610.f, 382.f);

    const GenSample fallback{};
    const GenSample &s = snap.empty() ? fallback : snap.back();
    const float MX = 470.f, MY = 444.f, MW = 190.f, MH = 66.f;
    drawMetric(window, fp, "curriculum",
               std::to_string(s.active_tests) + " / " +
                   std::to_string(AntoninaAPI::ALL_TESTS),
               MX, MY, MW, MH, sf::Color(238, 184, 82),
               (double)s.active_tests / AntoninaAPI::ALL_TESTS);
    drawMetric(window, fp, "best wins",
               std::to_string(s.best_wins) + " / " +
                   std::to_string(std::max(1, s.active_tests)),
               MX + 206.f, MY, MW, MH, sf::Color(116, 225, 148),
               (double)s.best_wins / std::max(1, s.active_tests));
    drawMetric(window, fp, "species", std::to_string(s.species_count),
               MX + 412.f, MY, MW, MH, sf::Color(206, 136, 235));
    drawMetric(window, fp, "nodes", compactValue(s.avg_nodes), MX,
               MY + 82.f, MW, MH, sf::Color(90, 205, 150));
    drawMetric(window, fp, "connections", compactValue(s.avg_connections),
               MX + 206.f, MY + 82.f, MW, MH, sf::Color(96, 212, 196));
    drawMetric(window, fp, "compat / ms",
               compactValue(s.compatibility_threshold) + " / " +
                   std::to_string(s.fitness_ms),
               MX + 412.f, MY + 82.f, MW, MH, sf::Color(232, 122, 122));

    drawRect(window, 470.f, 626.f, 610.f, 186.f, sf::Color(18, 18, 26),
             sf::Color(54, 58, 74), 1.f);
    drawText(window, fp, "map legend", 480.f, 636.f, 14,
             sf::Color(205, 210, 225));
    struct L {
      const char *name;
      sf::Color col;
    };
    L leg[] = {
        {"floor", cellColor('.')},
        {"rock/wall", cellColor('#')},
        {"pad", cellColor('O')},
        {"rover", cellColor('a')},
        {"rover+pad", cellColor('@')},
        {"bucket", cellColor('%')},
    };
    float lx = 486.f;
    float ly = 672.f;
    for (int i = 0; i < 6; ++i) {
      drawRect(window, lx, ly, 16, 16, leg[i].col);
      drawText(window, fp, leg[i].name, lx + 22, ly - 2, 13,
               sf::Color(200, 204, 220));
      ly += 24.f;
      if (i == 2) {
        lx += 170.f;
        ly = 672.f;
      }
    }
    drawText(window, fp,
             "topology links: blue = negative weight, yellow = positive weight",
             486.f, 760.f, 13, sf::Color(150, 156, 176));
    drawText(window, fp, "tests: last active curriculum slice",
             486.f, 782.f, 13, sf::Color(150, 156, 176));

    const float TX = 1102.f, TY = 46.f;
    drawRect(window, TX, TY, 486.f, 766.f, sf::Color(18, 18, 26),
             sf::Color(54, 58, 74), 1.f);
    int shown = std::min(10, std::max(1, active_tests));
    int base = std::max(0, active_tests - shown);
    char tests_title[128];
    std::snprintf(tests_title, sizeof(tests_title), "last %d active tests: %d..%d",
                  shown, base, base + shown - 1);
    drawText(window, fp, tests_title, TX + 10, TY + 8, 14,
             sf::Color(205, 210, 225));

    const float SIDE = 90.f;
    const float CELL_W = 220.f;
    const float CELL_H = 136.f;
    for (int i = 0; i < 10; ++i) {
      int col = i % 2;
      int row = i / 2;
      float x = TX + 18.f + col * CELL_W;
      float y = TY + 42.f + row * CELL_H;
      drawEpisode(window, fp, ep[i], x, y, SIDE);
    }

    sf::Color btn_fill =
        btn_hover ? sf::Color(64, 106, 178) : sf::Color(38, 66, 118);
    drawRect(window, BTN_X, BTN_Y, BTN_W, BTN_H, btn_fill,
             sf::Color(120, 160, 220), 1.f);
    drawText(window, fp, "save population", BTN_X + 18, BTN_Y + 6, 13,
             sf::Color::White);

    window.display();
  }
}
