#include "AntoninaAPI.h"
#include "Brain.h"
#include "NeatGenome.h"
#include "Perceptron.h"
#include "TestGenerator.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using std::cout, std::endl, std::this_thread::sleep_for;

constexpr int WIN_FITNESS = 500000;
constexpr int SUCCESS_STEP_BONUS = 10;
constexpr int PARTIAL_MAX = 50000;
constexpr int ROVER_PROGRESS_WEIGHT = 1000;
constexpr int CONTROL_PROGRESS_WEIGHT = 800;
constexpr int CONTROL_READY_BONUS = 1500;
constexpr int BUCKET_PICKED_BONUS = 10000;
constexpr int BUCKET_PROGRESS_WEIGHT = 1500;
constexpr int CURRENT_BUCKET_PROGRESS_WEIGHT = 700;
constexpr int SOLUTION_PROGRESS_WEIGHT = 1800;
constexpr int CURRENT_SOLUTION_PROGRESS_WEIGHT = 900;
constexpr int ALMOST_DONE_BONUS = 8000;
constexpr int INVALID_MOVE_PENALTY = 50;
constexpr int POST_PICKUP_INVALID_MOVE_PENALTY = 125;
constexpr unsigned short NO_STONE_UNREACHABLE = 1000;
constexpr int BASE_FEATURE_COUNT = 191;
constexpr int SOLUTION_FEATURE_START = BASE_FEATURE_COUNT;
constexpr int SOLUTION_NORM_STEPS = 40;

struct TestFile {
  std::ifstream stream;
  std::string path;
};

static TestFile openTestsFile() {
  const char *candidates[] = {"test.csv",
                              "..\\test.csv",
                              "..\\..\\test.csv",
                              "Antonina AI\\test.csv",
                              "..\\Antonina AI\\test.csv",
                              "..\\..\\Antonina AI\\test.csv",
                              "..\\..\\..\\Antonina AI\\test.csv"};

  for (const char *path : candidates) {
    std::ifstream fin(path);
    if (fin.is_open())
      return {std::move(fin), path};
  }

  return {};
}

static std::string currentWorkingDirectory() {
  try {
    return std::filesystem::current_path().string();
  } catch (...) {
    return "<unknown>";
  }
}

static bool inBounds(int x, int y) { return x >= 0 && x < 8 && y >= 0 && y < 8; }

static bool onBoardEdge(int x, int y) { return x == 0 || x == 7 || y == 0 || y == 7; }

static bool inBoardCorner(int x, int y) {
  return (x == 0 || x == 7) && (y == 0 || y == 7);
}

static char cellAt(const char lab[][8], int x, int y) {
  return inBounds(x, y) ? lab[x][y] : '\0';
}

static bool isHome(char c) { return c == 'O' || c == '@'; }

static bool isFreeCell(char c) { return c == '.' || c == 'O'; }

static double normCoord(int v) { return ((double)v - 3.5) / 3.5; }

static double normDelta(int v) { return (double)v / 7.0; }

static double normDist(int v) { return (double)v / 14.0; }

static double normWallDist(int v) { return (double)v / 7.0; }

static int gridManhattan(int ax, int ay, int bx, int by) {
  return abs(ax - bx) + abs(ay - by);
}

static int directionSign(int v) { return (v > 0) - (v < 0); }

static void clearStoneCells(bool cells[][8]) {
  for (int x = 0; x < 8; ++x)
    for (int y = 0; y < 8; ++y)
      cells[x][y] = false;
}

static void copyStoneCells(bool dst[][8], const bool src[][8]) {
  for (int x = 0; x < 8; ++x)
    for (int y = 0; y < 8; ++y)
      dst[x][y] = src[x][y];
}

template <typename CellAtFn>
static int controlDistanceWithCell(CellAtFn cell, int ax, int ay, int gx, int gy, int Ox, int Oy) {
  int best = 99;
  auto consider = [&](int x, int y) {
    if (!inBounds(x, y) || cell(x, y) == '#')
      return;
    best = std::min(best, gridManhattan(ax, ay, x, y));
  };

  int sx = directionSign(Ox - gx);
  int sy = directionSign(Oy - gy);
  if (sx != 0) {
    consider(gx + sx, gy);
    consider(gx - sx, gy);
  }
  if (sy != 0) {
    consider(gx, gy + sy);
    consider(gx, gy - sy);
  }

  if (best == 99)
    best = gridManhattan(ax, ay, gx, gy);
  return best;
}

static int controlDistance(const char lab[][8], int ax, int ay, int gx, int gy, int Ox, int Oy) {
  return controlDistanceWithCell(
      [&](int x, int y) { return cellAt(lab, x, y); }, ax, ay, gx, gy, Ox, Oy);
}

static int aggregateFitness(int wins, int failures, long long partial_sum, long long speed_sum) {
  int fail_avg = failures > 0 ? (int)(partial_sum / failures) : 0;
  int speed_avg = wins > 0 ? (int)(speed_sum / wins) : 0;
  long long value = (long long)wins * WIN_FITNESS + fail_avg + speed_avg;
  return (int)std::clamp(value, 0LL, (long long)std::numeric_limits<int>::max());
}

struct FitnessStats {
  int wins = 0;
  int failures = 0;
  long long partial_sum = 0;
  long long speed_sum = 0;

  void add(const FitnessStats &other) {
    wins += other.wins;
    failures += other.failures;
    partial_sum += other.partial_sum;
    speed_sum += other.speed_sum;
  }
};

static void scanLab(const char lab[][8], int &ax, int &ay, int &Ox, int &Oy, int &gx, int &gy, int &stones) {
  ax = ay = Ox = Oy = gx = gy = 0;
  stones = 0;
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      switch (lab[i][j]) {
      case 'a':
        ax = i;
        ay = j;
        break;
      case '@':
        ax = i;
        ay = j;
        Ox = i;
        Oy = j;
        break;
      case 'O':
        Ox = i;
        Oy = j;
        break;
      case '%':
        gx = i;
        gy = j;
        break;
      case '#':
        stones++;
        break;
      default:
        break;
      }
    }
  }
}

template <typename CellAtFn>
static void writeFeaturesWithCell(CellAtFn cell, int ax, int ay, int Ox, int Oy, int gx, int gy, int stones, double *dst) {
  int k = 0;
  auto push = [&](double v) {
    if (k < AntoninaAPI::INPUT_FEATURES)
      dst[k++] = v;
  };

  push(normCoord(ax));
  push(normCoord(ay));
  push(normCoord(Ox));
  push(normCoord(Oy));
  push(normCoord(gx));
  push(normCoord(gy));
  push(normDelta(gx - ax));
  push(normDelta(gy - ay));
  push(normDelta(Ox - gx));
  push(normDelta(Oy - gy));
  push(normDelta(Ox - ax));
  push(normDelta(Oy - ay));
  push(normDist(gridManhattan(ax, ay, gx, gy)));
  push(normDist(gridManhattan(Ox, Oy, gx, gy)));
  push(normDist(gridManhattan(Ox, Oy, ax, ay)));
  push(cell(ax, ay) == '@' ? 1.0 : 0.0);
  push(gridManhattan(ax, ay, gx, gy) == 1 ? 1.0 : 0.0);
  push((Ox == gx || Oy == gy) ? 1.0 : 0.0);
  push((ax == gx || ay == gy) ? 1.0 : 0.0);
  push((double)stones / 64.0);

  const int dx[4] = {-1, 0, 1, 0};
  const int dy[4] = {0, 1, 0, -1};
  const int base_r2b = gridManhattan(ax, ay, gx, gy);
  const int base_b2p = gridManhattan(Ox, Oy, gx, gy);
  const int base_r2p = gridManhattan(ax, ay, Ox, Oy);
  const int base_control =
      controlDistanceWithCell(cell, ax, ay, gx, gy, Ox, Oy);
  const int b2p_dx = Ox - gx;
  const int b2p_dy = Oy - gy;

  push((double)directionSign(gx - ax));
  push((double)directionSign(gy - ay));
  push((double)directionSign(b2p_dx));
  push((double)directionSign(b2p_dy));
  push((double)directionSign(Ox - ax));
  push((double)directionSign(Oy - ay));
  push(ax == gx ? 1.0 : 0.0);
  push(ay == gy ? 1.0 : 0.0);
  push(Ox == gx ? 1.0 : 0.0);
  push(Oy == gy ? 1.0 : 0.0);
  push(ax == Ox ? 1.0 : 0.0);
  push(ay == Oy ? 1.0 : 0.0);
  push(base_b2p == 1 ? 1.0 : 0.0);
  push(base_r2p == 1 ? 1.0 : 0.0);
  push((std::abs(b2p_dx) == 1 && std::abs(b2p_dy) == 1) ? 1.0 : 0.0);
  push((std::abs(gx - ax) == 1 && std::abs(gy - ay) == 1) ? 1.0 : 0.0);

  for (int d = 0; d < 4; d++) {
    int tx = ax + dx[d], ty = ay + dy[d];
    int ux = ax + dx[d] * 2, uy = ay + dy[d] * 2;
    int px = ax - dx[d], py = ay - dy[d];

    char ahead = cell(tx, ty);
    char two = cell(ux, uy);
    char behind = cell(px, py);
    char push_target = cell(gx + dx[d], gy + dy[d]);
    char push_control = cell(gx - dx[d], gy - dy[d]);

    bool ahead_free = isFreeCell(ahead);
    bool two_free_for_bucket = isFreeCell(two);
    bool two_free_for_stone = two == '.';
    bool push_target_in_bounds = inBounds(gx + dx[d], gy + dy[d]);
    bool push_control_in_bounds = inBounds(gx - dx[d], gy - dy[d]);
    bool push_target_free_for_bucket = isFreeCell(push_target);
    bool push_control_free = push_control_in_bounds &&
                             (isFreeCell(push_control) ||
                              (gx - dx[d] == ax && gy - dy[d] == ay));
    bool ahead_bucket = ahead == '%';
    bool ahead_stone = ahead == '#';
    bool can_push_bucket = ahead_bucket && two_free_for_bucket;
    bool can_push_stone = ahead_stone && two_free_for_stone;
    bool can_pull_bucket = ahead_free && behind == '%';
    bool can_pull_stone = ahead_free && behind == '#';
    bool valid_move = ahead_free || can_push_bucket || can_push_stone;
    bool moves_bucket = can_push_bucket || can_pull_bucket;
    bool moves_stone = can_push_stone || can_pull_stone;
    bool action_win = (can_push_bucket && isHome(two)) ||
                      (can_pull_bucket && cell(ax, ay) == '@');

    int next_ax = ax;
    int next_ay = ay;
    int next_gx = gx;
    int next_gy = gy;
    if (ahead_free) {
      next_ax = tx;
      next_ay = ty;
      if (can_pull_bucket) {
        next_gx = ax;
        next_gy = ay;
      }
    } else if (can_push_bucket) {
      next_ax = tx;
      next_ay = ty;
      next_gx = ux;
      next_gy = uy;
    } else if (can_push_stone) {
      next_ax = tx;
      next_ay = ty;
    }

    double delta_r2b =
        valid_move ? (double)(base_r2b - gridManhattan(next_ax, next_ay, next_gx,
                                                       next_gy)) /
                         14.0
                   : -1.0;
    double delta_b2p =
        valid_move ? (double)(base_b2p - gridManhattan(Ox, Oy, next_gx, next_gy)) /
                         14.0
                   : -1.0;
    double delta_r2p =
        valid_move ? (double)(base_r2p - gridManhattan(next_ax, next_ay, Ox, Oy)) /
                         14.0
                   : -1.0;
    int next_control =
        valid_move ? controlDistanceWithCell(cell, next_ax, next_ay, next_gx,
                                             next_gy, Ox, Oy)
                   : 14;
    double delta_control =
        valid_move ? (double)(base_control - next_control) / 14.0 : -1.0;
    int push_next_b2p =
        push_target_in_bounds ? gridManhattan(Ox, Oy, gx + dx[d], gy + dy[d]) : 99;
    int control_dist =
        push_control_in_bounds ? gridManhattan(ax, ay, gx - dx[d], gy - dy[d]) : 14;

    push(ahead_free ? 1.0 : 0.0);
    push(isHome(ahead) ? 1.0 : 0.0);
    push(ahead_bucket ? 1.0 : 0.0);
    push(ahead_stone ? 1.0 : 0.0);
    push(ahead == '\0' ? 1.0 : 0.0);
    push(two_free_for_bucket ? 1.0 : 0.0);
    push(isHome(two) ? 1.0 : 0.0);
    push((two == '\0' || two == '%' || two == '#') ? 1.0 : 0.0);
    push(can_push_bucket ? 1.0 : 0.0);
    push(can_push_stone ? 1.0 : 0.0);
    push(can_pull_bucket ? 1.0 : 0.0);
    push(can_pull_stone ? 1.0 : 0.0);
    push((can_push_bucket && isHome(two)) ? 1.0 : 0.0);
    push((can_pull_bucket && cell(ax, ay) == '@') ? 1.0 : 0.0);
    push(valid_move ? 1.0 : 0.0);
    push(valid_move ? 0.0 : 1.0);
    push(moves_bucket ? 1.0 : 0.0);
    push(moves_stone ? 1.0 : 0.0);
    push(action_win ? 1.0 : 0.0);
    push(delta_r2b);
    push(delta_b2p);
    push(delta_r2p);
    push(normDist(next_control));
    push(delta_control);
    push(push_target_free_for_bucket ? 1.0 : 0.0);
    push(isHome(push_target) ? 1.0 : 0.0);
    push(push_next_b2p < base_b2p ? 1.0 : 0.0);
    push(push_next_b2p == base_b2p ? 1.0 : 0.0);
    push(push_next_b2p > base_b2p ? 1.0 : 0.0);
    push(push_control_free ? 1.0 : 0.0);
    push((gx - dx[d] == ax && gy - dy[d] == ay) ? 1.0 : 0.0);
    push(normDist(control_dist));
  }

  push(normDist(base_control));

  push(normWallDist(ax));
  push(normWallDist(7 - ax));
  push(normWallDist(ay));
  push(normWallDist(7 - ay));
  push(normWallDist(gx));
  push(normWallDist(7 - gx));
  push(normWallDist(gy));
  push(normWallDist(7 - gy));
  push(normWallDist(Ox));
  push(normWallDist(7 - Ox));
  push(normWallDist(Oy));
  push(normWallDist(7 - Oy));
  push(onBoardEdge(ax, ay) ? 1.0 : 0.0);
  push(inBoardCorner(ax, ay) ? 1.0 : 0.0);
  push(onBoardEdge(gx, gy) ? 1.0 : 0.0);
  push(inBoardCorner(gx, gy) ? 1.0 : 0.0);
  push(onBoardEdge(Ox, Oy) ? 1.0 : 0.0);
  push(inBoardCorner(Ox, Oy) ? 1.0 : 0.0);

  for (int d = 0; d < 4; d++) {
    push(inBounds(gx + dx[d], gy + dy[d]) ? 1.0 : 0.0);
    push(inBounds(gx - dx[d], gy - dy[d]) ? 1.0 : 0.0);
  }

  while (k < AntoninaAPI::INPUT_FEATURES)
    dst[k++] = 0.0;
}

static void writeFeatures(const char lab[][8], int ax, int ay, int Ox, int Oy, int gx, int gy, int stones, double *dst) {
  writeFeaturesWithCell([&](int x, int y) { return cellAt(lab, x, y); }, ax,
                        ay, Ox, Oy, gx, gy, stones, dst);
}

struct CachedTransition {
  unsigned char ax = 0;
  unsigned char ay = 0;
  unsigned char gx = 0;
  unsigned char gy = 0;
  unsigned char invalid = 0;
  unsigned char win = 0;
};

struct NoStoneCaches {
  std::vector<double> features;
  std::vector<CachedTransition> transitions;
  std::vector<unsigned char> control_dist;
  std::vector<unsigned short> solution_dist;
};

constexpr int BOARD_CELLS = 64;
constexpr int NO_STONE_STATE_COUNT = BOARD_CELLS * BOARD_CELLS * BOARD_CELLS;
constexpr int ACTION_COUNT = 4;

static int cellIndex(int x, int y) { return x * 8 + y; }

static int noStoneStateIndex(int ax, int ay, int Ox, int Oy, int gx, int gy) {
  return (cellIndex(ax, ay) * BOARD_CELLS + cellIndex(Ox, Oy)) * BOARD_CELLS +
         cellIndex(gx, gy);
}

static void decodeNoStoneState(int index, int &ax, int &ay, int &Ox, int &Oy, int &gx, int &gy) {
  int bucket = index % BOARD_CELLS;
  index /= BOARD_CELLS;
  int home = index % BOARD_CELLS;
  int rover = index / BOARD_CELLS;
  ax = rover / 8;
  ay = rover % 8;
  Ox = home / 8;
  Oy = home % 8;
  gx = bucket / 8;
  gy = bucket % 8;
}

static char noStoneCell(int x, int y, int ax, int ay, int Ox, int Oy, int gx, int gy) {
  if (!inBounds(x, y))
    return '\0';
  if (x == gx && y == gy)
    return '%';
  if (x == ax && y == ay)
    return (x == Ox && y == Oy) ? '@' : 'a';
  if (x == Ox && y == Oy)
    return 'O';
  return '.';
}

static CachedTransition buildNoStoneTransition(int ax, int ay, int Ox, int Oy, int gx, int gy, int move) {
  const int dx[ACTION_COUNT] = {-1, 0, 1, 0};
  const int dy[ACTION_COUNT] = {0, 1, 0, -1};

  CachedTransition t;
  t.ax = (unsigned char)ax;
  t.ay = (unsigned char)ay;
  t.gx = (unsigned char)gx;
  t.gy = (unsigned char)gy;

  int tox = ax + dx[move], toy = ay + dy[move];
  int totox = ax + 2 * dx[move], totoy = ay + 2 * dy[move];
  int pullx = ax - dx[move], pully = ay - dy[move];
  bool toB = inBounds(tox, toy);
  bool totoB = inBounds(totox, totoy);
  bool pullB = inBounds(pullx, pully);
  if (!toB) {
    t.invalid = 1;
    return t;
  }

  char target = noStoneCell(tox, toy, ax, ay, Ox, Oy, gx, gy);
  if (target == '.' || target == 'O') {
    char pulled = pullB ? noStoneCell(pullx, pully, ax, ay, Ox, Oy, gx, gy)
                        : '\0';
    if (!pullB || pulled == '.' || pulled == 'O') {
      t.ax = (unsigned char)tox;
      t.ay = (unsigned char)toy;
      return t;
    }
    if (pulled == '%' && ax == Ox && ay == Oy) {
      t.win = 1;
      return t;
    }
    if (pulled == '%') {
      t.ax = (unsigned char)tox;
      t.ay = (unsigned char)toy;
      t.gx = (unsigned char)ax;
      t.gy = (unsigned char)ay;
      return t;
    }
    t.invalid = 1;
    return t;
  }

  if (target == '%') {
    char push_to =
        totoB ? noStoneCell(totox, totoy, ax, ay, Ox, Oy, gx, gy) : '\0';
    if (!totoB) {
      t.invalid = 1;
      return t;
    }
    if (push_to == '.') {
      t.ax = (unsigned char)tox;
      t.ay = (unsigned char)toy;
      t.gx = (unsigned char)totox;
      t.gy = (unsigned char)totoy;
      return t;
    }
    if (push_to == 'O') {
      t.win = 1;
      return t;
    }
  }

  t.invalid = 1;
  return t;
}

static NoStoneCaches &noStoneCaches() {
  static NoStoneCaches caches;
  static std::once_flag init_flag;
  std::call_once(init_flag, [] {
    caches.features.resize((size_t)NO_STONE_STATE_COUNT *
                           AntoninaAPI::INPUT_FEATURES);
    caches.transitions.resize((size_t)NO_STONE_STATE_COUNT * ACTION_COUNT);
    caches.control_dist.resize((size_t)NO_STONE_STATE_COUNT);
    caches.solution_dist.assign((size_t)NO_STONE_STATE_COUNT,
                                NO_STONE_UNREACHABLE);

    for (int index = 0; index < NO_STONE_STATE_COUNT; ++index) {
      int ax, ay, Ox, Oy, gx, gy;
      decodeNoStoneState(index, ax, ay, Ox, Oy, gx, gy);
      double *dst =
          caches.features.data() + (size_t)index * AntoninaAPI::INPUT_FEATURES;
      writeFeaturesWithCell(
          [&](int x, int y) {
            return noStoneCell(x, y, ax, ay, Ox, Oy, gx, gy);
          },
          ax, ay, Ox, Oy, gx, gy, 0, dst);
      caches.control_dist[(size_t)index] =
          (unsigned char)controlDistanceWithCell(
              [&](int x, int y) {
                return noStoneCell(x, y, ax, ay, Ox, Oy, gx, gy);
              },
              ax, ay, gx, gy, Ox, Oy);

      for (int move = 0; move < ACTION_COUNT; ++move) {
        caches.transitions[(size_t)index * ACTION_COUNT + move] =
            buildNoStoneTransition(ax, ay, Ox, Oy, gx, gy, move);
      }
    }

    std::vector<int> reverse_offsets((size_t)NO_STONE_STATE_COUNT + 1, 0);
    for (int index = 0; index < NO_STONE_STATE_COUNT; ++index) {
      int ax, ay, Ox, Oy, gx, gy;
      decodeNoStoneState(index, ax, ay, Ox, Oy, gx, gy);
      for (int move = 0; move < ACTION_COUNT; ++move) {
        const CachedTransition &t =
            caches.transitions[(size_t)index * ACTION_COUNT + move];
        if (t.invalid)
          continue;
        if (t.win) {
          caches.solution_dist[(size_t)index] = 1;
          continue;
        }
        int next_index = noStoneStateIndex(t.ax, t.ay, Ox, Oy, t.gx, t.gy);
        ++reverse_offsets[(size_t)next_index + 1];
      }
    }

    for (int i = 1; i <= NO_STONE_STATE_COUNT; ++i)
      reverse_offsets[(size_t)i] += reverse_offsets[(size_t)i - 1];

    std::vector<int> write_pos = reverse_offsets;
    std::vector<int> reverse_sources(
        (size_t)reverse_offsets[(size_t)NO_STONE_STATE_COUNT]);
    for (int index = 0; index < NO_STONE_STATE_COUNT; ++index) {
      int ax, ay, Ox, Oy, gx, gy;
      decodeNoStoneState(index, ax, ay, Ox, Oy, gx, gy);
      for (int move = 0; move < ACTION_COUNT; ++move) {
        const CachedTransition &t =
            caches.transitions[(size_t)index * ACTION_COUNT + move];
        if (t.invalid || t.win)
          continue;
        int next_index = noStoneStateIndex(t.ax, t.ay, Ox, Oy, t.gx, t.gy);
        reverse_sources[(size_t)write_pos[(size_t)next_index]++] = index;
      }
    }

    std::vector<int> queue;
    queue.reserve(NO_STONE_STATE_COUNT);
    for (int index = 0; index < NO_STONE_STATE_COUNT; ++index) {
      if (caches.solution_dist[(size_t)index] == 1)
        queue.push_back(index);
    }
    for (size_t head = 0; head < queue.size(); ++head) {
      int index = queue[head];
      int next_dist = (int)caches.solution_dist[(size_t)index] + 1;
      if (next_dist >= NO_STONE_UNREACHABLE)
        continue;
      int begin = reverse_offsets[(size_t)index];
      int end = reverse_offsets[(size_t)index + 1];
      for (int pos = begin; pos < end; ++pos) {
        int prev = reverse_sources[(size_t)pos];
        if (caches.solution_dist[(size_t)prev] <= next_dist)
          continue;
        caches.solution_dist[(size_t)prev] = (unsigned short)next_dist;
        queue.push_back(prev);
      }
    }

    auto solutionNorm = [](int dist) {
      if (dist >= NO_STONE_UNREACHABLE)
        return 1.0;
      return (double)std::clamp(dist, 0, SOLUTION_NORM_STEPS) /
             (double)SOLUTION_NORM_STEPS;
    };

    for (int index = 0; index < NO_STONE_STATE_COUNT; ++index) {
      int ax, ay, Ox, Oy, gx, gy;
      decodeNoStoneState(index, ax, ay, Ox, Oy, gx, gy);
      double *dst =
          caches.features.data() + (size_t)index * AntoninaAPI::INPUT_FEATURES;
      const int current = caches.solution_dist[(size_t)index];
      int k = SOLUTION_FEATURE_START;
      dst[k++] = solutionNorm(current);
      for (int move = 0; move < ACTION_COUNT; ++move) {
        const CachedTransition &t =
            caches.transitions[(size_t)index * ACTION_COUNT + move];
        int next = NO_STONE_UNREACHABLE;
        if (t.win) {
          next = 0;
        } else if (!t.invalid) {
          const int next_index = noStoneStateIndex(t.ax, t.ay, Ox, Oy, t.gx,
                                                   t.gy);
          next = caches.solution_dist[(size_t)next_index];
        }

        double delta = -1.0;
        if (current < NO_STONE_UNREACHABLE && next < NO_STONE_UNREACHABLE) {
          delta = (double)(current - next) / (double)SOLUTION_NORM_STEPS;
          delta = std::clamp(delta, -1.0, 1.0);
        }
        dst[k++] = solutionNorm(next);
        dst[k++] = delta;
        dst[k++] = next < current ? 1.0 : 0.0;
        dst[k++] = t.win ? 1.0 : 0.0;
      }
    }
  });
  return caches;
}

AntoninaAPI::AntoninaAPI() {
  std::string error;
  if (!reloadTests(&error)) {
    std::cerr << error << std::endl;
    std::exit(1);
  }
}

bool AntoninaAPI::reloadTests(std::string *error, std::string *loaded_path) {
  auto install_test = [this](int i, int ax, int ay, int Ox, int Oy, int gx,
                             int gy, int rn) {
    axarr[i] = ax;
    ayarr[i] = ay;
    Oxarr[i] = Ox;
    Oyarr[i] = Oy;
    gxarr[i] = gx;
    gyarr[i] = gy;
    rnarr[i] = rn;

    if (!MakeLab(prebuilt_labs[i], ax, ay, Ox, Oy, gx, gy, rn)) {
      axarr[i] = 1;
      ayarr[i] = 1;
      Oxarr[i] = 1;
      Oyarr[i] = 1;
      gxarr[i] = 1;
      gyarr[i] = 2;
      rnarr[i] = 0;
      MakeLab(prebuilt_labs[i], 1, 1, 1, 1, 1, 2, 0);
    }

    clearStoneCells(prebuilt_stone_cells[i]);
    for (int x = 0; x < 8; ++x)
      for (int y = 0; y < 8; ++y)
        if (prebuilt_labs[i][x][y] == '#')
          prebuilt_stone_cells[i][x][y] = true;

    prebuilt_initial_r2b[i] =
        abs(axarr[i] - gxarr[i]) + abs(ayarr[i] - gyarr[i]);
    prebuilt_initial_b2p[i] =
        abs(Oxarr[i] - gxarr[i]) + abs(Oyarr[i] - gyarr[i]);
    prebuilt_initial_control[i] =
        controlDistance(prebuilt_labs[i], axarr[i], ayarr[i], gxarr[i],
                        gyarr[i], Oxarr[i], Oyarr[i]);
  };

  for (int i = 0; i < ALL_TESTS; i++)
    install_test(i, 1, 1, 1, 1, 1, 2, 0);

  static std::atomic<bool> reported_test_path{false};

  TestFile test_file = openTestsFile();
  if (loaded_path)
    *loaded_path = test_file.path;
  std::ifstream fin = std::move(test_file.stream);
  if (!fin.is_open()) {
    if (error)
      *error = "FATAL: test.csv was not found from cwd: " +
               currentWorkingDirectory();
    return false;
  }
  if (!reported_test_path.exchange(true)) {
    std::cerr << "Loaded test.csv from: " << test_file.path
              << " | cwd: " << currentWorkingDirectory() << std::endl;
  }

  std::vector<TestCase> tests;
  tests.reserve(ALL_TESTS);
  std::string read_error;
  bool tests_ok = readTests(fin, tests, &read_error, ALL_TESTS);

  if (!tests_ok || tests.empty()) {
    if (error) {
      *error = "test.csv is empty or invalid";
      if (!read_error.empty())
        *error += ": " + read_error;
      *error += "; using fallback test map";
    }
  } else {
    auto stage_counts = categoryCounts(tests);

    orderCurriculum(tests);
    for (int i = 0; i < (int)tests.size(); ++i) {
      const auto &test = tests[i];
      install_test(i, test.ax, test.ay, test.Ox, test.Oy, test.gx, test.gy,
                   test.rn);
    }

    static std::atomic<bool> reported_curriculum{false};
    if (!reported_curriculum.exchange(true)) {
      std::cerr << "Curriculum order:";
      for (int i = 0; i < CATEGORY_COUNT; ++i) {
        std::cerr << ' ' << categoryName(i) << '='
                  << stage_counts[i];
      }
      std::cerr << std::endl;
    }
  }
  return true;
}

void AntoninaAPI::ClearLab(char lab[][8]) {
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      lab[i][j] = '.';
}

void AntoninaAPI::PrintLab(char lab[][8]) {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++)
      printf("%c ", lab[i][j]);
    printf("\n");
  }
  printf("\n");
}

bool AntoninaAPI::MakeLab(char lab[][8], int ax, int ay, int Ox, int Oy, int gx, int gy, int rn, int rx[], int ry[]) {
  ClearLab(lab);
  if ((gx == ax && gy == ay) || (gx == Ox && gy == Oy))
    return false;
  for (int i = 0; i < rn; i++)
    lab[rx[i]][ry[i]] = '#';
  if (ax == Ox && ay == Oy)
    lab[ax][ay] = '@';
  else {
    lab[ax][ay] = 'a';
    lab[Ox][Oy] = 'O';
  }
  lab[gx][gy] = '%';
  return true;
}

bool AntoninaAPI::MakeLab(char lab[][8], int ax, int ay, int Ox, int Oy, int gx, int gy, int rn) {
  int rx[64] = {0}, ry[64] = {0};
  if (!MakeLab(lab, ax, ay, Ox, Oy, gx, gy, 0, rx, ry)) {
    logfile << "GEN-ERR";
    return false;
  }
  if (rn <= 0)
    return true;

  std::vector<std::pair<int, int>> candidates;
  candidates.reserve(64);
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      if (lab[i][j] == '.' && abs(i - Ox) + abs(j - Oy) > 2)
        candidates.push_back({i, j});
    }
  }
  if (rn > (int)candidates.size())
    return false;

  unsigned seed = 2166136261u;
  auto mix = [&](int value) {
    seed ^= (unsigned)(value + 257);
    seed *= 16777619u;
  };
  mix(ax);
  mix(ay);
  mix(Ox);
  mix(Oy);
  mix(gx);
  mix(gy);
  mix(rn);

  std::mt19937 rng(seed);
  std::shuffle(candidates.begin(), candidates.end(), rng);
  for (int i = 0; i < rn; ++i)
    lab[candidates[i].first][candidates[i].second] = '#';
  return true;
}

void AntoninaAPI::CopyLab(char lab[][8], char copy[][8], int *ax, int *ay, int *Ox, int *Oy, int *gx, int *gy) {
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++) {
      copy[i][j] = lab[i][j];
      switch (copy[i][j]) {
      case 'a':
        *ax = i;
        *ay = j;
        break;
      case '%':
        *gx = i;
        *gy = j;
        break;
      case 'O':
        *Ox = i;
        *Oy = j;
        break;
      case '@':
        *ax = i;
        *ay = j;
        *Ox = i;
        *Oy = j;
        break;
      default:
        break;
      }
    }
}

void AntoninaAPI::encodeLabInto(const char lab[][8], double *dst) {
  int ax, ay, Ox, Oy, gx, gy, stones;
  scanLab(lab, ax, ay, Ox, Oy, gx, gy, stones);
  writeFeatures(lab, ax, ay, Ox, Oy, gx, gy, stones, dst);
}

void AntoninaAPI::encodeStateInto(const GameState &s, double *dst) const {
  int stones = 0;
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      if (s.lab[i][j] == '#')
        stones++;
  writeFeatures(s.lab, s.ax, s.ay, s.Ox, s.Oy, s.gx, s.gy, stones, dst);
}

void AntoninaAPI::encodeFastStateInto(const FastGameState &s, double *dst) const {
  if (s.stones == 0) {
    int index = noStoneStateIndex(s.ax, s.ay, s.Ox, s.Oy, s.gx, s.gy);
    const double *src =
        noStoneCaches().features.data() +
        (size_t)index * AntoninaAPI::INPUT_FEATURES;
    std::memcpy(dst, src, sizeof(double) * AntoninaAPI::INPUT_FEATURES);
    return;
  }

  auto cell = [&](int x, int y) -> char {
    if (!inBounds(x, y))
      return '\0';
    if (x == s.gx && y == s.gy)
      return '%';
    if (s.stone_cells[x][y])
      return '#';
    if (x == s.ax && y == s.ay)
      return (x == s.Ox && y == s.Oy) ? '@' : 'a';
    if (x == s.Ox && y == s.Oy)
      return 'O';
    return '.';
  };
  writeFeaturesWithCell(cell, s.ax, s.ay, s.Ox, s.Oy, s.gx, s.gy, s.stones,
                        dst);
}

char AntoninaAPI::outputToMove(int out) {
  switch (out) {
  case 0:
    return 'u';
  case 1:
    return 'r';
  case 2:
    return 'd';
  case 3:
    return 'l';
  default:
    return 'u';
  }
}

char AntoninaAPI::Move(char map[][8], Brain *p) {
  FastGameState s;
  scanLab(map, s.ax, s.ay, s.Ox, s.Oy, s.gx, s.gy, s.stones);
  clearStoneCells(s.stone_cells);
  for (int x = 0; x < 8; ++x)
    for (int y = 0; y < 8; ++y)
      if (map[x][y] == '#')
        s.stone_cells[x][y] = true;

  std::array<double, INPUT_FEATURES> input{};
  encodeFastStateInto(s, input.data());
  p->feedForward(input.data());

  double outputs[ACTION_COUNT];
  for (int move = 0; move < ACTION_COUNT; ++move)
    outputs[move] = p->outputValue(move);
  return outputToMove(selectValidMove(s, outputs, p->getOut()));
}

void AntoninaAPI::initGameState(GameState &s, int test_index) {
  memcpy(s.lab, prebuilt_labs[test_index], sizeof(s.lab));
  s.ax = axarr[test_index];
  s.ay = ayarr[test_index];
  s.Ox = Oxarr[test_index];
  s.Oy = Oyarr[test_index];
  s.gx = gxarr[test_index];
  s.gy = gyarr[test_index];
  s.initial_r2b = prebuilt_initial_r2b[test_index];
  s.initial_b2p = prebuilt_initial_b2p[test_index];
  s.initial_control = prebuilt_initial_control[test_index];
  s.min_r2b = s.initial_r2b;
  s.min_b2p = s.initial_b2p;
  s.min_control = s.initial_control;
  s.bucket_picked = false;
  s.shaping_score = 0;
  s.invalid_moves = 0;
  s.done = false;
  s.result = -1;
}

void AntoninaAPI::initFastGameState(FastGameState &s, int test_index) const {
  s.ax = axarr[test_index];
  s.ay = ayarr[test_index];
  s.Ox = Oxarr[test_index];
  s.Oy = Oyarr[test_index];
  s.gx = gxarr[test_index];
  s.gy = gyarr[test_index];
  s.initial_r2b = prebuilt_initial_r2b[test_index];
  s.initial_b2p = prebuilt_initial_b2p[test_index];
  s.initial_control = prebuilt_initial_control[test_index];
  s.min_r2b = s.initial_r2b;
  s.min_b2p = s.initial_b2p;
  s.min_control = s.initial_control;
  s.initial_solution = 0;
  s.min_solution = 0;
  s.stones = rnarr[test_index];
  copyStoneCells(s.stone_cells, prebuilt_stone_cells[test_index]);
  if (s.stones == 0) {
    int index = noStoneStateIndex(s.ax, s.ay, s.Ox, s.Oy, s.gx, s.gy);
    s.initial_solution = noStoneCaches().solution_dist[(size_t)index];
    s.min_solution = s.initial_solution;
  }
  s.bucket_picked = false;
  s.shaping_score = 0;
  s.invalid_moves = 0;
  s.done = false;
  s.result = -1;
}

int AntoninaAPI::runScalarGame(Brain *p, int test_index, GameState &s) {
  initGameState(s, test_index);

  for (int step = 1; step <= STEPS_LIMIT; step++) {
    int r = stepGameState(s, Move(s.lab, p), step);
    if (r > 0) {
      s.done = true;
      s.result = r;
      return r;
    }
  }

  return s.result;
}

int AntoninaAPI::runFastGame(Brain *p, int test_index, FastGameState &s) const {
  initFastGameState(s, test_index);
  std::array<double, INPUT_FEATURES> input{};

  for (int step = 1; step <= STEPS_LIMIT; step++) {
    encodeFastStateInto(s, input.data());
    p->feedForward(input.data());
    double outputs[ACTION_COUNT];
    for (int move = 0; move < ACTION_COUNT; ++move)
      outputs[move] = p->outputValue(move);
    int r = stepFastGameStateMove(s, selectValidMove(s, outputs, p->getOut()),
                                  step);
    if (r > 0) {
      s.done = true;
      s.result = r;
      return r;
    }
  }

  return s.result;
}

int AntoninaAPI::GoTestImproved(char lab[][8], int &min_rover_to_bucket, int &min_bucket_to_pad, bool &bucket_picked, bool doprint, Brain *p, int &shaping_score) {
  if (doprint)
    logfile << "#\tNew test... ";
  bucket_picked = false;
  shaping_score = 0;

  for (int s = 1; s < STEPS_LIMIT + 1; s++) {
    if (doprint) {
      sleep_for(std::chrono::milliseconds(TIME_TO_SLEEP));
      printf("Step: %d / %d\n", s, STEPS_LIMIT);
      PrintLab(lab);
      for (int l = 0; l < 10; l++)
        printf("\r\033[A");
    }

    char copy[8][8];
    int ax, ay, Ox, Oy, gx, gy;
    CopyLab(lab, copy, &ax, &ay, &Ox, &Oy, &gx, &gy);

    min_rover_to_bucket =
        std::min(min_rover_to_bucket, abs(ax - gx) + abs(ay - gy));
    min_bucket_to_pad =
        std::min(min_bucket_to_pad, abs(Ox - gx) + abs(Oy - gy));

    int old_d_box_goal = abs(Ox - gx) + abs(Oy - gy);
    int old_d_agent_box = abs(ax - gx) + abs(ay - gy);
    int prev_gx = gx, prev_gy = gy;

    char c = Move(copy, p);
    if (c == 'x') {
      if (doprint)
        logfile << " terminated!\n";
      return -1;
    }
    if (c == 'q') {
      if (doprint)
        logfile << " terminated!\n";
      return -2;
    }

    int tox = ax, toy = ay, totox = ax, totoy = ay, pullx = ax, pully = ay;
    bool toB = true, totoB = true, pullB = true;
    switch (c) {
    case 'u':
      tox--;
      totox -= 2;
      pullx++;
      toB = (tox >= 0);
      totoB = (totox >= 0);
      pullB = (pullx < 8);
      break;
    case 'd':
      tox++;
      totox += 2;
      pullx--;
      toB = (tox < 8);
      totoB = (totox < 8);
      pullB = (pullx >= 0);
      break;
    case 'l':
      toy--;
      totoy -= 2;
      pully++;
      toB = (toy >= 0);
      totoB = (totoy >= 0);
      pullB = (pully < 8);
      break;
    case 'r':
      toy++;
      totoy += 2;
      pully--;
      toB = (toy < 8);
      totoB = (totoy < 8);
      pullB = (pully >= 0);
      break;
    default:
      break;
    }
    if (!toB)
      continue;

    auto apply_shaping = [&]() {
      int nax, nay, nOx, nOy, ngx2, ngy2;
      char tmp[8][8];
      CopyLab(lab, tmp, &nax, &nay, &nOx, &nOy, &ngx2, &ngy2);
      if ((ngx2 != prev_gx || ngy2 != prev_gy) && !bucket_picked) {
        bucket_picked = true;
      }
      shaping_score +=
          (old_d_box_goal - (abs(nOx - ngx2) + abs(nOy - ngy2))) * 10;
      shaping_score +=
          (old_d_agent_box - (abs(nax - ngx2) + abs(nay - ngy2))) * 2;
    };

    if (lab[tox][toy] == '.' || lab[tox][toy] == 'O') {
      lab[tox][toy] = (lab[tox][toy] == 'O') ? '@' : 'a';
      if (!pullB || lab[pullx][pully] == '.' || lab[pullx][pully] == 'O') {
        lab[ax][ay] = (lab[ax][ay] == 'a') ? '.' : 'O';
      } else if (lab[pullx][pully] == '%' && lab[ax][ay] == '@') {
        if (doprint)
          logfile << " done in " << s << " steps!\n";
        apply_shaping();
        return s;
      } else if (lab[pullx][pully] == '#' && lab[ax][ay] == '@') {
        lab[ax][ay] = 'O';
      } else {
        lab[ax][ay] = lab[pullx][pully];
        lab[pullx][pully] = '.';
      }
    } else if (lab[tox][toy] == '%') {
      if (!totoB || lab[totox][totoy] == '#')
        continue;
      if (lab[totox][totoy] == '.') {
        lab[tox][toy] = 'a';
        lab[ax][ay] = (lab[ax][ay] == 'a') ? '.' : 'O';
        lab[totox][totoy] = '%';
        apply_shaping();
      } else if (lab[totox][totoy] == 'O') {
        if (doprint)
          logfile << " done in " << s << " steps!\n";
        apply_shaping();
        return s;
      }
      continue;
    } else if (lab[tox][toy] == '#') {
      if (!totoB || lab[totox][totoy] != '.')
        continue;
      lab[tox][toy] = 'a';
      lab[ax][ay] = (lab[ax][ay] == 'a') ? '.' : 'O';
      lab[totox][totoy] = '#';
      apply_shaping();
      continue;
    }

    apply_shaping();
  }

  if (doprint)
    logfile << " fail!\n";
  return -1;
}

int AntoninaAPI::stepGameState(GameState &s, char c, int step) {
  const int ax = s.ax, ay = s.ay;
  const int Ox = s.Ox, Oy = s.Oy;
  const int gx = s.gx, gy = s.gy;

  s.min_r2b = std::min(s.min_r2b, abs(ax - gx) + abs(ay - gy));
  s.min_b2p = std::min(s.min_b2p, abs(Ox - gx) + abs(Oy - gy));
  s.min_control =
      std::min(s.min_control, controlDistance(s.lab, ax, ay, gx, gy, Ox, Oy));

  const int old_d_box_goal = abs(Ox - gx) + abs(Oy - gy);
  const int old_d_agent_box = abs(ax - gx) + abs(ay - gy);
  const int prev_gx = gx, prev_gy = gy;

  int tox = ax, toy = ay, totox = ax, totoy = ay, pullx = ax, pully = ay;
  bool toB = true, totoB = true, pullB = true;
  switch (c) {
  case 'u':
    tox--;
    totox -= 2;
    pullx++;
    toB = (tox >= 0);
    totoB = (totox >= 0);
    pullB = (pullx < 8);
    break;
  case 'd':
    tox++;
    totox += 2;
    pullx--;
    toB = (tox < 8);
    totoB = (totox < 8);
    pullB = (pullx >= 0);
    break;
  case 'l':
    toy--;
    totoy -= 2;
    pully++;
    toB = (toy >= 0);
    totoB = (totoy >= 0);
    pullB = (pully < 8);
    break;
  case 'r':
    toy++;
    totoy += 2;
    pully--;
    toB = (toy < 8);
    totoB = (totoy < 8);
    pullB = (pully >= 0);
    break;
  default:
    break;
  }

  if (!toB) {
    s.invalid_moves++;
    return 0;
  }

  auto applyShaping = [&](int nax, int nay, int nOx, int nOy, int ngx_new,
                          int ngy_new) {
    if ((ngx_new != prev_gx || ngy_new != prev_gy) && !s.bucket_picked)
      s.bucket_picked = true;
    s.shaping_score +=
        (old_d_box_goal - (abs(nOx - ngx_new) + abs(nOy - ngy_new))) * 10;
    s.shaping_score +=
        (old_d_agent_box - (abs(nax - ngx_new) + abs(nay - ngy_new))) * 2;
    s.min_r2b = std::min(s.min_r2b, abs(nax - ngx_new) + abs(nay - ngy_new));
    s.min_b2p = std::min(s.min_b2p, abs(nOx - ngx_new) + abs(nOy - ngy_new));
    s.min_control =
        std::min(s.min_control,
                 controlDistance(s.lab, nax, nay, ngx_new, ngy_new, nOx, nOy));
  };

  if (s.lab[tox][toy] == '.' || s.lab[tox][toy] == 'O') {
    const char dest_old = s.lab[tox][toy];
    s.lab[tox][toy] = (dest_old == 'O') ? '@' : 'a';

    if (!pullB || s.lab[pullx][pully] == '.' || s.lab[pullx][pully] == 'O') {

      s.lab[ax][ay] = (s.lab[ax][ay] == 'a') ? '.' : 'O';
      s.ax = tox;
      s.ay = toy;
      applyShaping(s.ax, s.ay, Ox, Oy, gx, gy);
      return 0;
    } else if (s.lab[pullx][pully] == '%' && s.lab[ax][ay] == '@') {

      applyShaping(ax, ay, ax, ay, pullx, pully);
      return step;
    } else if (s.lab[pullx][pully] == '#' && s.lab[ax][ay] == '@') {

      s.lab[ax][ay] = 'O';
      s.ax = tox;
      s.ay = toy;
      applyShaping(s.ax, s.ay, Ox, Oy, gx, gy);
      return 0;
    } else {

      const char pulled = s.lab[pullx][pully];
      s.lab[ax][ay] = pulled;
      s.lab[pullx][pully] = '.';
      int new_gx = gx, new_gy = gy;
      if (pulled == '%') {
        new_gx = ax;
        new_gy = ay;
      }
      s.ax = tox;
      s.ay = toy;
      s.gx = new_gx;
      s.gy = new_gy;
      applyShaping(s.ax, s.ay, Ox, Oy, s.gx, s.gy);
      return 0;
    }
  } else if (s.lab[tox][toy] == '%') {
    if (!totoB || s.lab[totox][totoy] == '#') {
      s.invalid_moves++;
      return 0;
    }
    if (s.lab[totox][totoy] == '.') {

      s.lab[tox][toy] = 'a';
      s.lab[ax][ay] = (s.lab[ax][ay] == 'a') ? '.' : 'O';
      s.lab[totox][totoy] = '%';
      s.ax = tox;
      s.ay = toy;
      s.gx = totox;
      s.gy = totoy;
      applyShaping(s.ax, s.ay, Ox, Oy, s.gx, s.gy);
      return 0;
    } else if (s.lab[totox][totoy] == 'O') {

      applyShaping(ax, ay, Ox, Oy, tox, toy);
      return step;
    }
    s.invalid_moves++;
    return 0;
  } else if (s.lab[tox][toy] == '#') {
    if (!totoB || s.lab[totox][totoy] != '.') {
      s.invalid_moves++;
      return 0;
    }
    s.lab[tox][toy] = 'a';
    s.lab[ax][ay] = (s.lab[ax][ay] == 'a') ? '.' : 'O';
    s.lab[totox][totoy] = '#';
    s.ax = tox;
    s.ay = toy;
    applyShaping(s.ax, s.ay, Ox, Oy, gx, gy);
    return 0;
  }

  s.invalid_moves++;
  return 0;
}

int AntoninaAPI::stepFastGameState(FastGameState &s, char c, int step) const {
  switch (c) {
  case 'u':
    return stepFastGameStateMove(s, 0, step);
  case 'r':
    return stepFastGameStateMove(s, 1, step);
  case 'd':
    return stepFastGameStateMove(s, 2, step);
  case 'l':
    return stepFastGameStateMove(s, 3, step);
  default:
    s.invalid_moves++;
    return 0;
  }
}

void AntoninaAPI::validFastMoves(const FastGameState &s, bool *valid_moves) const {
  for (int move = 0; move < ACTION_COUNT; ++move)
    valid_moves[move] = false;

  if (s.stones == 0) {
    int index = noStoneStateIndex(s.ax, s.ay, s.Ox, s.Oy, s.gx, s.gy);
    NoStoneCaches &caches = noStoneCaches();
    for (int move = 0; move < ACTION_COUNT; ++move) {
      const CachedTransition &t =
          caches.transitions[(size_t)index * ACTION_COUNT + move];
      valid_moves[move] = !t.invalid;
    }
    return;
  }

  auto cell = [&](int x, int y) -> char {
    if (!inBounds(x, y))
      return '\0';
    if (x == s.gx && y == s.gy)
      return '%';
    if (s.stone_cells[x][y])
      return '#';
    if (x == s.ax && y == s.ay)
      return (x == s.Ox && y == s.Oy) ? '@' : 'a';
    if (x == s.Ox && y == s.Oy)
      return 'O';
    return '.';
  };

  for (int move = 0; move < ACTION_COUNT; ++move) {
    int dx = 0, dy = 0;
    switch (move) {
    case 0:
      dx = -1;
      break;
    case 1:
      dy = 1;
      break;
    case 2:
      dx = 1;
      break;
    case 3:
      dy = -1;
      break;
    default:
      break;
    }

    const int tox = s.ax + dx;
    const int toy = s.ay + dy;
    const int totox = s.ax + 2 * dx;
    const int totoy = s.ay + 2 * dy;
    const int pullx = s.ax - dx;
    const int pully = s.ay - dy;
    if (!inBounds(tox, toy))
      continue;

    const char target = cell(tox, toy);
    if (target == '.' || target == 'O') {
      const char pulled = inBounds(pullx, pully) ? cell(pullx, pully) : '\0';
      valid_moves[move] =
          !inBounds(pullx, pully) || pulled == '.' || pulled == 'O' ||
          pulled == '%' || pulled == '#';
      continue;
    }

    if (target == '%') {
      if (!inBounds(totox, totoy))
        continue;
      const char push_to = cell(totox, totoy);
      valid_moves[move] = push_to == '.' || push_to == 'O';
      continue;
    }

    if (target == '#') {
      if (!inBounds(totox, totoy))
        continue;
      valid_moves[move] = cell(totox, totoy) == '.';
    }
  }
}

int AntoninaAPI::selectValidMove(const FastGameState &s, const double *outputs, int fallback) const {
  if (!outputs)
    return std::clamp(fallback, 0, ACTION_COUNT - 1);

  bool valid[ACTION_COUNT];
  validFastMoves(s, valid);

  int best_raw = 0;
  for (int move = 1; move < ACTION_COUNT; ++move)
    if (outputs[move] > outputs[best_raw])
      best_raw = move;
  if (valid[best_raw])
    return best_raw;

  int best_valid = -1;
  for (int move = 0; move < ACTION_COUNT; ++move) {
    if (!valid[move])
      continue;
    if (best_valid < 0 || outputs[move] > outputs[best_valid])
      best_valid = move;
  }

  return best_valid >= 0 ? best_valid : std::clamp(fallback, 0, ACTION_COUNT - 1);
}

int AntoninaAPI::stepFastGameStateMove(FastGameState &s, int move, int step) const {
  if (move < 0 || move >= ACTION_COUNT) {
    s.invalid_moves++;
    return 0;
  }

  const int ax = s.ax, ay = s.ay;
  const int Ox = s.Ox, Oy = s.Oy;
  const int gx = s.gx, gy = s.gy;

  s.min_r2b = std::min(s.min_r2b, abs(ax - gx) + abs(ay - gy));
  s.min_b2p = std::min(s.min_b2p, abs(Ox - gx) + abs(Oy - gy));
  auto fastControlDistance = [&](int nax, int nay, int ngx_new,
                                 int ngy_new) {
    return controlDistanceWithCell(
        [&](int x, int y) -> char {
          if (!inBounds(x, y))
            return '\0';
          if (x == ngx_new && y == ngy_new)
            return '%';
          if (s.stone_cells[x][y])
            return '#';
          if (x == nax && y == nay)
            return (x == Ox && y == Oy) ? '@' : 'a';
          if (x == Ox && y == Oy)
            return 'O';
          return '.';
        },
        nax, nay, ngx_new, ngy_new, Ox, Oy);
  };
  const int old_d_box_goal = abs(Ox - gx) + abs(Oy - gy);
  const int old_d_agent_box = abs(ax - gx) + abs(ay - gy);
  const int prev_gx = gx, prev_gy = gy;

  if (s.stones == 0) {
    int index = noStoneStateIndex(ax, ay, Ox, Oy, gx, gy);
    NoStoneCaches &caches = noStoneCaches();
    s.min_control =
        std::min(s.min_control, (int)caches.control_dist[(size_t)index]);
    s.min_solution =
        std::min(s.min_solution, (int)caches.solution_dist[(size_t)index]);
    const CachedTransition &t =
        caches.transitions[(size_t)index * ACTION_COUNT + move];
    if (t.invalid) {
      s.invalid_moves++;
      return 0;
    }
    if (t.win)
      return step;

    s.ax = t.ax;
    s.ay = t.ay;
    s.gx = t.gx;
    s.gy = t.gy;
    if ((s.gx != prev_gx || s.gy != prev_gy) && !s.bucket_picked)
      s.bucket_picked = true;
    s.shaping_score +=
        (old_d_box_goal - (abs(Ox - s.gx) + abs(Oy - s.gy))) * 10;
    s.shaping_score +=
        (old_d_agent_box - (abs(s.ax - s.gx) + abs(s.ay - s.gy))) * 2;
    s.min_r2b = std::min(s.min_r2b, abs(s.ax - s.gx) + abs(s.ay - s.gy));
    s.min_b2p = std::min(s.min_b2p, abs(Ox - s.gx) + abs(Oy - s.gy));
    int next_index = noStoneStateIndex(s.ax, s.ay, Ox, Oy, s.gx, s.gy);
    s.min_control =
        std::min(s.min_control, (int)caches.control_dist[(size_t)next_index]);
    s.min_solution = std::min(
        s.min_solution, (int)caches.solution_dist[(size_t)next_index]);
    return 0;
  }

  s.min_control = std::min(s.min_control, fastControlDistance(ax, ay, gx, gy));

  int tox = ax, toy = ay, totox = ax, totoy = ay, pullx = ax, pully = ay;
  bool toB = true, totoB = true, pullB = true;
  switch (move) {
  case 0:
    tox--;
    totox -= 2;
    pullx++;
    toB = (tox >= 0);
    totoB = (totox >= 0);
    pullB = (pullx < 8);
    break;
  case 2:
    tox++;
    totox += 2;
    pullx--;
    toB = (tox < 8);
    totoB = (totox < 8);
    pullB = (pullx >= 0);
    break;
  case 3:
    toy--;
    totoy -= 2;
    pully++;
    toB = (toy >= 0);
    totoB = (totoy >= 0);
    pullB = (pully < 8);
    break;
  case 1:
    toy++;
    totoy += 2;
    pully--;
    toB = (toy < 8);
    totoB = (totoy < 8);
    pullB = (pully >= 0);
    break;
  default:
    break;
  }

  if (!toB) {
    s.invalid_moves++;
    return 0;
  }

  auto cell = [&](int x, int y) -> char {
    if (!inBounds(x, y))
      return '\0';
    if (x == s.gx && y == s.gy)
      return '%';
    if (s.stone_cells[x][y])
      return '#';
    if (x == s.ax && y == s.ay)
      return (x == s.Ox && y == s.Oy) ? '@' : 'a';
    if (x == s.Ox && y == s.Oy)
      return 'O';
    return '.';
  };

  auto moveStone = [&](int from_x, int from_y, int to_x, int to_y) {
    s.stone_cells[from_x][from_y] = false;
    s.stone_cells[to_x][to_y] = true;
  };

  auto applyShaping = [&](int nax, int nay, int nOx, int nOy, int ngx_new,
                          int ngy_new) {
    if ((ngx_new != prev_gx || ngy_new != prev_gy) && !s.bucket_picked)
      s.bucket_picked = true;
    s.shaping_score +=
        (old_d_box_goal - (abs(nOx - ngx_new) + abs(nOy - ngy_new))) * 10;
    s.shaping_score +=
        (old_d_agent_box - (abs(nax - ngx_new) + abs(nay - ngy_new))) * 2;
    s.min_r2b = std::min(s.min_r2b, abs(nax - ngx_new) + abs(nay - ngy_new));
    s.min_b2p = std::min(s.min_b2p, abs(nOx - ngx_new) + abs(nOy - ngy_new));
    s.min_control =
        std::min(s.min_control,
                 fastControlDistance(nax, nay, ngx_new, ngy_new));
  };

  const char target = cell(tox, toy);
  if (target == '.' || target == 'O') {
    const char pulled = pullB ? cell(pullx, pully) : '\0';
    if (!pullB || pulled == '.' || pulled == 'O') {
      s.ax = tox;
      s.ay = toy;
      applyShaping(s.ax, s.ay, Ox, Oy, gx, gy);
      return 0;
    }
    if (pulled == '%' && ax == Ox && ay == Oy) {
      applyShaping(ax, ay, ax, ay, pullx, pully);
      return step;
    }
    if (pulled == '#' && ax == Ox && ay == Oy) {
      s.ax = tox;
      s.ay = toy;
      applyShaping(s.ax, s.ay, Ox, Oy, gx, gy);
      return 0;
    }

    if (pulled == '%') {
      s.gx = ax;
      s.gy = ay;
    } else if (pulled == '#') {
      moveStone(pullx, pully, ax, ay);
    }
    s.ax = tox;
    s.ay = toy;
    applyShaping(s.ax, s.ay, Ox, Oy, s.gx, s.gy);
    return 0;
  }

  if (target == '%') {
    const char push_to = totoB ? cell(totox, totoy) : '\0';
    if (!totoB || push_to == '#') {
      s.invalid_moves++;
      return 0;
    }
    if (push_to == '.') {
      s.ax = tox;
      s.ay = toy;
      s.gx = totox;
      s.gy = totoy;
      applyShaping(s.ax, s.ay, Ox, Oy, s.gx, s.gy);
      return 0;
    }
    if (push_to == 'O') {
      applyShaping(ax, ay, Ox, Oy, tox, toy);
      return step;
    }
    s.invalid_moves++;
    return 0;
  }

  if (target == '#') {
    const char push_to = totoB ? cell(totox, totoy) : '\0';
    if (!totoB || push_to != '.') {
      s.invalid_moves++;
      return 0;
    }
    moveStone(tox, toy, totox, totoy);
    s.ax = tox;
    s.ay = toy;
    applyShaping(s.ax, s.ay, Ox, Oy, gx, gy);
    return 0;
  }

  s.invalid_moves++;
  return 0;
}

int AntoninaAPI::scoreOfState(const GameState &s) {
  if (s.result > 0)
    return WIN_FITNESS +
           std::max(0, STEPS_LIMIT - s.result) * SUCCESS_STEP_BONUS;

  int rover_progress = std::max(0, s.initial_r2b - s.min_r2b);
  int bucket_progress = std::max(0, s.initial_b2p - s.min_b2p);
  int current_b2p = abs(s.Ox - s.gx) + abs(s.Oy - s.gy);
  int current_bucket_progress = std::max(0, s.initial_b2p - current_b2p);
  int control_progress = std::max(0, s.initial_control - s.min_control);

  int test_score = rover_progress * ROVER_PROGRESS_WEIGHT;
  test_score += control_progress * CONTROL_PROGRESS_WEIGHT;
  if (s.initial_control > 0 && s.min_control == 0)
    test_score += CONTROL_READY_BONUS;
  if (s.bucket_picked) {
    test_score += BUCKET_PICKED_BONUS;
    test_score += bucket_progress * BUCKET_PROGRESS_WEIGHT;
    test_score += current_bucket_progress * CURRENT_BUCKET_PROGRESS_WEIGHT;
    if (s.min_b2p <= 2)
      test_score += ALMOST_DONE_BONUS;
  }

  int invalid_penalty = INVALID_MOVE_PENALTY;
  if (s.bucket_picked)
    invalid_penalty += POST_PICKUP_INVALID_MOVE_PENALTY;
  test_score -= s.invalid_moves * invalid_penalty;
  return std::clamp(test_score, 0, PARTIAL_MAX);
}

int AntoninaAPI::scoreOfFastState(const FastGameState &s) const {
  if (s.result > 0)
    return WIN_FITNESS +
           std::max(0, STEPS_LIMIT - s.result) * SUCCESS_STEP_BONUS;

  int rover_progress = std::max(0, s.initial_r2b - s.min_r2b);
  int bucket_progress = std::max(0, s.initial_b2p - s.min_b2p);
  int current_b2p = abs(s.Ox - s.gx) + abs(s.Oy - s.gy);
  int current_bucket_progress = std::max(0, s.initial_b2p - current_b2p);
  int control_progress = std::max(0, s.initial_control - s.min_control);

  int test_score = rover_progress * ROVER_PROGRESS_WEIGHT;
  test_score += control_progress * CONTROL_PROGRESS_WEIGHT;
  if (s.initial_control > 0 && s.min_control == 0)
    test_score += CONTROL_READY_BONUS;
  if (s.bucket_picked) {
    test_score += BUCKET_PICKED_BONUS;
    test_score += bucket_progress * BUCKET_PROGRESS_WEIGHT;
    test_score += current_bucket_progress * CURRENT_BUCKET_PROGRESS_WEIGHT;
    if (s.min_b2p <= 2)
      test_score += ALMOST_DONE_BONUS;
  }

  if (s.stones == 0 && s.initial_solution > 0 &&
      s.initial_solution < NO_STONE_UNREACHABLE) {
    int solution_progress =
        std::max(0, s.initial_solution - s.min_solution);
    int index = noStoneStateIndex(s.ax, s.ay, s.Ox, s.Oy, s.gx, s.gy);
    int current_solution = noStoneCaches().solution_dist[(size_t)index];
    if (current_solution < NO_STONE_UNREACHABLE) {
      int current_solution_progress =
          std::max(0, s.initial_solution - current_solution);
      test_score += solution_progress * SOLUTION_PROGRESS_WEIGHT;
      if (s.bucket_picked)
        test_score +=
            current_solution_progress * CURRENT_SOLUTION_PROGRESS_WEIGHT;
      if (current_solution <= 2)
        test_score += ALMOST_DONE_BONUS / 2;
    }
  }

  int invalid_penalty = INVALID_MOVE_PENALTY;
  if (s.bucket_picked)
    invalid_penalty += POST_PICKUP_INVALID_MOVE_PENALTY;
  test_score -= s.invalid_moves * invalid_penalty;
  return std::clamp(test_score, 0, PARTIAL_MAX);
}

int AntoninaAPI::solveFitness(Brain *p, int tests_to_run) {
  int actual_tests =
      tests_to_run > 0 ? std::min(tests_to_run, ALL_TESTS)
                       : std::min(active_tests, ALL_TESTS);

  int wins = 0;
  int failures = 0;
  long long partial_sum = 0;
  long long speed_sum = 0;

  for (int i = 0; i < actual_tests; i++) {
    FastGameState s;
    runFastGame(p, i, s);
    if (s.result > 0) {
      wins++;
      speed_sum += std::max(0, STEPS_LIMIT - s.result) * SUCCESS_STEP_BONUS;
    } else {
      failures++;
      partial_sum += scoreOfFastState(s);
    }

    if (i == 59) {
      int early = aggregateFitness(wins, failures, partial_sum, speed_sum);
      if (wins == 0 && early < 1000)
        return early - 1;
    }
  }

  return actual_tests > 0
             ? aggregateFitness(wins, failures, partial_sum, speed_sum)
             : 0;
}

template <typename BatchBrain>
int solveFitnessBatchImpl(AntoninaAPI &api, BatchBrain *p, int tests_to_run, int *case_scores) {
  int actual_tests =
      tests_to_run > 0 ? std::min(tests_to_run, AntoninaAPI::ALL_TESTS)
                       : std::min(api.active_tests, AntoninaAPI::ALL_TESTS);
  if (actual_tests <= 0)
    return 0;

  const int IN_SZ = p->getInputSize();
  const int OUT_SZ = p->getOutputSize();
  if (IN_SZ != AntoninaAPI::INPUT_FEATURES)
    return 0;

  static thread_local std::vector<AntoninaAPI::FastGameState> states;
  static thread_local std::vector<double> batch_in, batch_out;
  static thread_local std::vector<double> scratch_in;
  static thread_local std::vector<const double *> input_rows;
  static thread_local std::vector<int> moves;
  static thread_local std::vector<int> active_indices;

  auto runPhase = [&](int t_start, int t_end) -> FitnessStats {
    const int B = t_end - t_start;
    if (B <= 0)
      return {};

    states.resize(B);
    batch_out.resize((size_t)B * OUT_SZ);
    moves.resize(B);
    active_indices.resize(B);
    if constexpr (std::is_same_v<BatchBrain, NeatGenome>) {
      scratch_in.resize((size_t)B * IN_SZ);
      input_rows.resize(B);
    } else {
      batch_in.resize((size_t)B * IN_SZ);
    }

    for (int i = 0; i < B; i++) {
      int ti = t_start + i;
      api.initFastGameState(states[i], ti);
      active_indices[i] = i;
    }

    int active = B;

    for (int step = 1; step <= api.STEPS_LIMIT && active > 0; step++) {

      if constexpr (std::is_same_v<BatchBrain, NeatGenome>) {
        for (int slot = 0; slot < active; slot++) {
          int i = active_indices[slot];
          if (states[i].stones == 0) {
            int index = noStoneStateIndex(states[i].ax, states[i].ay,
                                          states[i].Ox, states[i].Oy,
                                          states[i].gx, states[i].gy);
            input_rows[slot] =
                noStoneCaches().features.data() +
                (size_t)index * AntoninaAPI::INPUT_FEATURES;
          } else {
            double *dst = scratch_in.data() + (size_t)slot * IN_SZ;
            api.encodeFastStateInto(states[i], dst);
            input_rows[slot] = dst;
          }
        }
        p->feedForwardBatchRows(input_rows.data(), batch_out.data(), active);
      } else {
        for (int slot = 0; slot < active; slot++) {
          int i = active_indices[slot];
          api.encodeFastStateInto(states[i],
                              batch_in.data() + (size_t)slot * IN_SZ);
        }
        p->feedForwardBatch(batch_in.data(), batch_out.data(), active);
      }
      int still_active = 0;
      for (int slot = 0; slot < active; slot++) {
        int i = active_indices[slot];
        const double *outputs = batch_out.data() + (size_t)slot * OUT_SZ;
        moves[slot] = api.selectValidMove(states[i], outputs, 0);
        int r = api.stepFastGameStateMove(states[i], moves[slot], step);
        if (r > 0) {
          states[i].done = true;
          states[i].result = r;
        } else {
          active_indices[still_active++] = i;
        }
      }
      active = still_active;
    }

    FitnessStats stats;
    for (int i = 0; i < B; i++) {
      const int case_score = api.scoreOfFastState(states[i]);
      if (case_scores)
        case_scores[t_start + i] = case_score;
      if (states[i].result > 0) {
        stats.wins++;
        stats.speed_sum +=
            std::max(0, api.STEPS_LIMIT - states[i].result) * SUCCESS_STEP_BONUS;
      } else {
        stats.failures++;
        stats.partial_sum += case_score;
      }
    }
    return stats;
  };

  const int phase1_end = std::min(60, actual_tests);
  FitnessStats stats = runPhase(0, phase1_end);

  if (actual_tests > 60) {
    int early =
        aggregateFitness(stats.wins, stats.failures, stats.partial_sum,
                         stats.speed_sum);
    if (stats.wins == 0 && early < 1000) {
      if (case_scores)
        std::fill(case_scores + phase1_end, case_scores + actual_tests, 0);
      return early - 1;
    }
    stats.add(runPhase(60, actual_tests));
  }

  return aggregateFitness(stats.wins, stats.failures, stats.partial_sum,
                          stats.speed_sum);
}

int AntoninaAPI::solveFitnessBatch(Perceptron *p, int tests_to_run) {
  return solveFitnessBatchImpl(*this, p, tests_to_run);
}

int AntoninaAPI::solveFitnessBatch(NeatGenome *p, int tests_to_run) {
  return solveFitnessBatchImpl(*this, p, tests_to_run);
}

int AntoninaAPI::solveFitnessBatch(NeatGenome *p, int tests_to_run, int *case_scores) {
  return solveFitnessBatchImpl(*this, p, tests_to_run, case_scores);
}

void AntoninaAPI::demonstrate(Brain *p) {
  printf("Starting new Antonina runs!\nSTEPS_LIMIT=%d\n", STEPS_LIMIT);
  logfile.open("antlog.txt");
  logfile << "#####\tStarting new Antonina runs!\n#####\tSTEPS_LIMIT="
          << STEPS_LIMIT << "\n";
  srand(static_cast<unsigned int>(time(nullptr)));

  int wins = 0, sum = 0;
  logfile << "#####\tStarting Test 00...\n";

  for (int i = 0; i < ALL_TESTS; i++) {
    char lab[8][8];
    memcpy(lab, prebuilt_labs[i], sizeof(lab));
    printf("Test 00: %d/%d\r", i, ALL_TESTS);

    int min_r2b = prebuilt_initial_r2b[i];
    int min_b2p = prebuilt_initial_b2p[i];
    bool bucket_picked = false;
    int shaping = 0;
    int res = GoTestImproved(lab, min_r2b, min_b2p, bucket_picked, PRINT_STEPS,
                             p, shaping);
    if (res > 0) {
      wins++;
      sum += res;
    } else if (res == -2) {
      exit(0);
    }
  }

  int score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / ALL_TESTS;
  int wr = 100 * wins / ALL_TESTS;
  int as = wins > 0 ? sum / wins : 0;
  printf("Test 00: winrate=%d%% av.steps=%d score=%d\n", wr, as, score);
  logfile << "#####\tTest 00: winrate=" << wr << "% av.steps=" << as
          << " score=" << score << "\n";
  logfile << "#####\tAll done!\n";
  logfile.close();
}

void AntoninaAPI::initLabForAnim(int i, char out[][8]) {
  if (i < 0)
    i = 0;
  if (i >= ALL_TESTS)
    i = ALL_TESTS - 1;
  memcpy(out, prebuilt_labs[i], sizeof(prebuilt_labs[i]));
}

int AntoninaAPI::animStep(char lab[][8], Brain *p, int step) {

  char c = Move(lab, p);

  int ax = 0, ay = 0, Ox = 0, Oy = 0, gx = 0, gy = 0;
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      switch (lab[i][j]) {
      case 'a':
        ax = i;
        ay = j;
        break;
      case '%':
        gx = i;
        gy = j;
        break;
      case 'O':
        Ox = i;
        Oy = j;
        break;
      case '@':
        ax = i;
        ay = j;
        Ox = i;
        Oy = j;
        break;
      }
    }
  }

  GameState s;
  memcpy(s.lab, lab, sizeof(s.lab));
  s.ax = ax;
  s.ay = ay;
  s.Ox = Ox;
  s.Oy = Oy;
  s.gx = gx;
  s.gy = gy;
  s.min_r2b = 0;
  s.min_b2p = 0;
  s.min_control = 0;
  s.initial_r2b = 0;
  s.initial_b2p = 0;
  s.initial_control = 0;
  s.bucket_picked = false;
  s.shaping_score = 0;
  s.invalid_moves = 0;
  s.done = false;
  s.result = 0;

  int r = stepGameState(s, c, step);
  memcpy(lab, s.lab, sizeof(s.lab));
  return r;
}

int AntoninaAPI::countWins(Brain *p, int tests_to_run) {
  int actual_tests =
      tests_to_run > 0 ? std::min(tests_to_run, ALL_TESTS)
                       : std::min(active_tests, ALL_TESTS);

  int wins = 0;
  for (int i = 0; i < actual_tests; i++) {
    FastGameState s;
    runFastGame(p, i, s);
    if (s.result > 0)
      wins++;
  }
  return wins;
}

int AntoninaAPI::collectFailures(Brain *p, int tests_to_run, std::vector<int> &failures, int max_failures) {
  int actual_tests =
      tests_to_run > 0 ? std::min(tests_to_run, ALL_TESTS)
                       : std::min(active_tests, ALL_TESTS);

  failures.clear();
  int wins = 0;
  for (int i = 0; i < actual_tests; i++) {
    FastGameState s;
    runFastGame(p, i, s);
    if (s.result > 0) {
      wins++;
    } else if ((int)failures.size() < max_failures) {
      failures.push_back(i);
    }
  }
  return wins;
}

int AntoninaAPI::testResult(Brain *p, int test_index, int *score) {
  if (test_index < 0)
    test_index = 0;
  if (test_index >= ALL_TESTS)
    test_index = ALL_TESTS - 1;

  FastGameState s;
  int result = runFastGame(p, test_index, s);
  if (score)
    *score = scoreOfFastState(s);
  return result;
}

int AntoninaAPI::traceTest(Brain *p, int test_index, TestTrace &trace) {
  if (test_index < 0)
    test_index = 0;
  if (test_index >= ALL_TESTS)
    test_index = ALL_TESTS - 1;

  FastGameState s;
  initFastGameState(s, test_index);
  std::array<double, INPUT_FEATURES> input{};
  trace = TestTrace{};
  trace.test_index = test_index;
  trace.initial_r2b = s.initial_r2b;
  trace.initial_b2p = s.initial_b2p;
  trace.initial_control = s.initial_control;
  trace.initial_solution = s.initial_solution;
  trace.stones = s.stones;
  trace.steps_detail.reserve(STEPS_LIMIT);

  for (int step = 1; step <= STEPS_LIMIT; ++step) {
    encodeFastStateInto(s, input.data());
    p->feedForward(input.data());
    double outputs[ACTION_COUNT];
    for (int out = 0; out < ACTION_COUNT; ++out)
      outputs[out] = p->outputValue(out);
    bool valid[ACTION_COUNT];
    validFastMoves(s, valid);
    int raw_move = 0;
    for (int out = 1; out < ACTION_COUNT; ++out)
      if (outputs[out] > outputs[raw_move])
        raw_move = out;
    int move = selectValidMove(s, outputs, p->getOut());
    if (move != raw_move)
      ++trace.masked_moves;
    if (move >= 0 && move < ACTION_COUNT)
      ++trace.moves[(size_t)move];
    trace.last_move = move;

    TestTrace::Step detail;
    detail.step = step;
    detail.ax = s.ax;
    detail.ay = s.ay;
    detail.Ox = s.Ox;
    detail.Oy = s.Oy;
    detail.gx = s.gx;
    detail.gy = s.gy;
    if (s.stones == 0) {
      int index = noStoneStateIndex(s.ax, s.ay, s.Ox, s.Oy, s.gx, s.gy);
      detail.solution = noStoneCaches().solution_dist[(size_t)index];
    }
    detail.raw_move = raw_move;
    detail.selected_move = move;
    for (int out = 0; out < ACTION_COUNT; ++out) {
      detail.outputs[(size_t)out] = outputs[out];
      detail.valid[(size_t)out] = valid[out];
    }

    int result = stepFastGameStateMove(s, move, step);
    detail.result = result;
    detail.invalid_moves = s.invalid_moves;
    detail.bucket_picked = s.bucket_picked;
    trace.steps_detail.push_back(detail);
    trace.steps = step;
    if (result > 0) {
      s.done = true;
      s.result = result;
      break;
    }
  }

  trace.result = s.result;
  trace.score = scoreOfFastState(s);
  trace.min_r2b = s.min_r2b;
  trace.min_b2p = s.min_b2p;
  trace.min_control = s.min_control;
  trace.min_solution = s.min_solution;
  trace.current_solution = 0;
  if (s.stones == 0) {
    int index = noStoneStateIndex(s.ax, s.ay, s.Ox, s.Oy, s.gx, s.gy);
    trace.current_solution = noStoneCaches().solution_dist[(size_t)index];
  }
  trace.invalid_moves = s.invalid_moves;
  trace.bucket_picked = s.bucket_picked;
  trace.ax = s.ax;
  trace.ay = s.ay;
  trace.Ox = s.Ox;
  trace.Oy = s.Oy;
  trace.gx = s.gx;
  trace.gy = s.gy;
  trace.stones = s.stones;
  return trace.result;
}
