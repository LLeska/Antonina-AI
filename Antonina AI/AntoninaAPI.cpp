#include "AntoninaAPI.h"
#include "NeatEvolution.h"
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
#include <string>
#include <thread>
#include <utility>
#include <vector>

using std::cout, std::endl, std::this_thread::sleep_for;
using namespace std::chrono_literals;

namespace {
constexpr int WIN_FITNESS = 500000;
constexpr int SUCCESS_STEP_BONUS = 10;
constexpr int PARTIAL_MAX = 50000;
constexpr int ROVER_PROGRESS_WEIGHT = 1000;
constexpr int BUCKET_PICKED_BONUS = 10000;
constexpr int BUCKET_PROGRESS_WEIGHT = 1500;
constexpr int ALMOST_DONE_BONUS = 8000;
constexpr int INVALID_MOVE_PENALTY = 50;

struct TestFile {
  std::ifstream stream;
  std::string path;
};

TestFile openTestsFile() {
  const char *candidates[] = {"Test0.csv",
                              "..\\Test0.csv",
                              "..\\..\\Test0.csv",
                              "Antonina AI\\Test0.csv",
                              "..\\Antonina AI\\Test0.csv",
                              "..\\..\\Antonina AI\\Test0.csv",
                              "..\\..\\..\\Antonina AI\\Test0.csv"};

  for (const char *path : candidates) {
    std::ifstream fin(path);
    if (fin.is_open())
      return {std::move(fin), path};
  }

  return {};
}

std::string currentWorkingDirectory() {
  try {
    return std::filesystem::current_path().string();
  } catch (...) {
    return "<unknown>";
  }
}

bool inBounds(int x, int y) { return x >= 0 && x < 8 && y >= 0 && y < 8; }

char cellAt(const char lab[][8], int x, int y) {
  return inBounds(x, y) ? lab[x][y] : '\0';
}

bool isHome(char c) { return c == 'O' || c == '@'; }

bool isFreeCell(char c) { return c == '.' || c == 'O'; }

double normCoord(int v) { return ((double)v - 3.5) / 3.5; }

double normDelta(int v) { return (double)v / 7.0; }

double normDist(int v) { return (double)v / 14.0; }

int manhattan(int ax, int ay, int bx, int by) {
  return abs(ax - bx) + abs(ay - by);
}

int directionSign(int v) { return (v > 0) - (v < 0); }

int controlDistance(const char lab[][8], int ax, int ay, int gx, int gy, int Ox,
                    int Oy) {
  int best = 99;
  auto consider = [&](int x, int y) {
    if (!inBounds(x, y) || cellAt(lab, x, y) == '#')
      return;
    best = std::min(best, manhattan(ax, ay, x, y));
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
    best = manhattan(ax, ay, gx, gy);
  return best;
}

int aggregateFitness(int wins, int failures, long long partial_sum,
                     long long speed_sum) {
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

void scanLab(const char lab[][8], int &ax, int &ay, int &Ox, int &Oy, int &gx,
             int &gy, int &stones) {
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

void writeFeatures(const char lab[][8], int ax, int ay, int Ox, int Oy, int gx,
                   int gy, int stones, double *dst) {
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
  push(normDist(manhattan(ax, ay, gx, gy)));
  push(normDist(manhattan(Ox, Oy, gx, gy)));
  push(normDist(manhattan(Ox, Oy, ax, ay)));
  push(lab[ax][ay] == '@' ? 1.0 : 0.0);
  push(manhattan(ax, ay, gx, gy) == 1 ? 1.0 : 0.0);
  push((Ox == gx || Oy == gy) ? 1.0 : 0.0);
  push((ax == gx || ay == gy) ? 1.0 : 0.0);
  push((double)stones / 64.0);

  const int dx[4] = {-1, 0, 1, 0};
  const int dy[4] = {0, 1, 0, -1};
  const int base_r2b = manhattan(ax, ay, gx, gy);
  const int base_b2p = manhattan(Ox, Oy, gx, gy);
  const int base_r2p = manhattan(ax, ay, Ox, Oy);
  const int base_control = controlDistance(lab, ax, ay, gx, gy, Ox, Oy);
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

    char ahead = cellAt(lab, tx, ty);
    char two = cellAt(lab, ux, uy);
    char behind = cellAt(lab, px, py);
    char push_target = cellAt(lab, gx + dx[d], gy + dy[d]);
    char push_control = cellAt(lab, gx - dx[d], gy - dy[d]);

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
                      (can_pull_bucket && lab[ax][ay] == '@');

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
        valid_move ? (double)(base_r2b - manhattan(next_ax, next_ay, next_gx,
                                                   next_gy)) /
                         14.0
                   : -1.0;
    double delta_b2p =
        valid_move ? (double)(base_b2p - manhattan(Ox, Oy, next_gx, next_gy)) /
                         14.0
                   : -1.0;
    double delta_r2p =
        valid_move ? (double)(base_r2p - manhattan(next_ax, next_ay, Ox, Oy)) /
                         14.0
                   : -1.0;
    int next_control =
        valid_move ? controlDistance(lab, next_ax, next_ay, next_gx, next_gy,
                                     Ox, Oy)
                   : 14;
    double delta_control =
        valid_move ? (double)(base_control - next_control) / 14.0 : -1.0;
    int push_next_b2p =
        push_target_in_bounds ? manhattan(Ox, Oy, gx + dx[d], gy + dy[d]) : 99;
    int control_dist =
        push_control_in_bounds ? manhattan(ax, ay, gx - dx[d], gy - dy[d]) : 14;

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
    push((can_pull_bucket && lab[ax][ay] == '@') ? 1.0 : 0.0);
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

  while (k < AntoninaAPI::INPUT_FEATURES)
    dst[k++] = 0.0;
}
}

AntoninaAPI::AntoninaAPI() {
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

    prebuilt_initial_r2b[i] =
        abs(axarr[i] - gxarr[i]) + abs(ayarr[i] - gyarr[i]);
    prebuilt_initial_b2p[i] =
        abs(Oxarr[i] - gxarr[i]) + abs(Oyarr[i] - gyarr[i]);
  };

  for (int i = 0; i < ALL_TESTS; i++)
    install_test(i, 1, 1, 1, 1, 1, 2, 0);

  static std::atomic<bool> reported_test_path{false};

  TestFile test_file = openTestsFile();
  std::ifstream fin = std::move(test_file.stream);
  if (!fin.is_open()) {
    if (!reported_test_path.exchange(true)) {
      std::cerr << "FATAL: Test0.csv was not found from cwd: "
                << currentWorkingDirectory() << std::endl;
    }
    std::exit(1);
  }
  if (!reported_test_path.exchange(true)) {
    std::cerr << "Loaded Test0.csv from: " << test_file.path
              << " | cwd: " << currentWorkingDirectory() << std::endl;
  }

  int loaded_tests = 0;
  for (; loaded_tests < ALL_TESTS; loaded_tests++) {
    int ax, ay, Ox, Oy, gx, gy, rn;
    if (!(fin >> ax >> ay >> Ox >> Oy >> gx >> gy >> rn))
      break;

    bool coords_ok = ax >= 0 && ax < 8 && ay >= 0 && ay < 8 && Ox >= 0 &&
                     Ox < 8 && Oy >= 0 && Oy < 8 && gx >= 0 && gx < 8 &&
                     gy >= 0 && gy < 8 && rn >= 0;

    if (!coords_ok) {
      std::cerr << "Invalid Test0.csv row " << loaded_tests
                << "; using fallback map" << std::endl;
      install_test(loaded_tests, 1, 1, 1, 1, 1, 2, 0);
      continue;
    }

    install_test(loaded_tests, ax, ay, Ox, Oy, gx, gy, rn);
  }

  if (loaded_tests == 0)
    std::cerr << "Test0.csv is empty or invalid; using fallback test map"
              << std::endl;
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

bool AntoninaAPI::MakeLab(char lab[][8], int ax, int ay, int Ox, int Oy, int gx,
                          int gy, int rn, int rx[], int ry[]) {
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

bool AntoninaAPI::MakeLab(char lab[][8], int ax, int ay, int Ox, int Oy, int gx,
                          int gy, int rn) {
  int rx[64] = {0}, ry[64] = {0};
  if (!MakeLab(lab, ax, ay, Ox, Oy, gx, gy, 0, rx, ry)) {
    logfile << "GEN-ERR";
    return false;
  }
  int count = rn, stop = 16;
  while (count > 0) {
    for (int i = 0; i < 8; i++)
      for (int j = 0; j < 8; j++)
        if (lab[i][j] == '.' && rand() % 64 < rn &&
            (abs(i - Ox) + abs(j - Oy) > 2)) {
          lab[i][j] = '#';
          count--;
          if (count == 0)
            return true;
        }
    if (--stop == 0)
      return false;
  }
  return true;
}

void AntoninaAPI::CopyLab(char lab[][8], char copy[][8], int *ax, int *ay,
                          int *Ox, int *Oy, int *gx, int *gy) {
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
  std::array<double, INPUT_FEATURES> input{};
  encodeLabInto(map, input.data());
  p->feedForward(input.data());
  return outputToMove(p->getOut());
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
  s.min_r2b = s.initial_r2b;
  s.min_b2p = s.initial_b2p;
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

int AntoninaAPI::GoTestImproved(char lab[][8], int &min_rover_to_bucket,
                                int &min_bucket_to_pad, bool &bucket_picked,
                                bool doprint, Brain *p, int &shaping_score) {
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

int AntoninaAPI::scoreOfState(const GameState &s) {
  if (s.result > 0)
    return WIN_FITNESS +
           std::max(0, STEPS_LIMIT - s.result) * SUCCESS_STEP_BONUS;

  int rover_progress = std::max(0, s.initial_r2b - s.min_r2b);
  int bucket_progress = std::max(0, s.initial_b2p - s.min_b2p);

  int test_score = rover_progress * ROVER_PROGRESS_WEIGHT;
  if (s.bucket_picked) {
    test_score += BUCKET_PICKED_BONUS;
    test_score += bucket_progress * BUCKET_PROGRESS_WEIGHT;
    if (s.min_b2p <= 2)
      test_score += ALMOST_DONE_BONUS;
  }

  test_score -= s.invalid_moves * INVALID_MOVE_PENALTY;
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
  auto addState = [&](const GameState &s) {
    if (s.result > 0) {
      wins++;
      speed_sum += std::max(0, STEPS_LIMIT - s.result) * SUCCESS_STEP_BONUS;
    } else {
      failures++;
      partial_sum += scoreOfState(s);
    }
  };

  for (int i = 0; i < actual_tests; i++) {
    GameState s;
    runScalarGame(p, i, s);
    addState(s);

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
int AntoninaAPI::solveFitnessBatchImpl(BatchBrain *p, int tests_to_run) {
  int actual_tests =
      tests_to_run > 0 ? std::min(tests_to_run, ALL_TESTS)
                       : std::min(active_tests, ALL_TESTS);
  if (actual_tests <= 0)
    return 0;

  const int IN_SZ = p->getInputSize();
  const int OUT_SZ = p->getOutputSize();
  if (IN_SZ != INPUT_FEATURES)
    return 0;

  static thread_local std::vector<GameState> states;
  static thread_local std::vector<double> batch_in, batch_out;
  static thread_local std::vector<int> moves;

  auto runPhase = [&](int t_start, int t_end) -> FitnessStats {
    const int B = t_end - t_start;
    if (B <= 0)
      return {};

    states.resize(B);
    batch_in.resize((size_t)B * IN_SZ);
    batch_out.resize((size_t)B * OUT_SZ);
    moves.resize(B);

    for (int i = 0; i < B; i++) {
      int ti = t_start + i;
      initGameState(states[i], ti);
    }

    int active = B;

    for (int step = 1; step <= STEPS_LIMIT && active > 0; step++) {

      for (int i = 0; i < B; i++) {
        if (states[i].done)
          continue;
        encodeStateInto(states[i], batch_in.data() + (size_t)i * IN_SZ);
      }

      p->feedForwardBatch(batch_in.data(), batch_out.data(), B);
      p->getOutBatch(batch_out.data(), moves.data(), B);

      for (int i = 0; i < B; i++) {
        if (states[i].done)
          continue;
        char c = outputToMove(moves[i]);
        int r = stepGameState(states[i], c, step);
        if (r > 0) {
          states[i].done = true;
          states[i].result = r;
          active--;
        }
      }
    }

    FitnessStats stats;
    for (int i = 0; i < B; i++) {
      if (states[i].result > 0) {
        stats.wins++;
        stats.speed_sum +=
            std::max(0, STEPS_LIMIT - states[i].result) * SUCCESS_STEP_BONUS;
      } else {
        stats.failures++;
        stats.partial_sum += scoreOfState(states[i]);
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
    if (stats.wins == 0 && early < 1000)
      return early - 1;
    stats.add(runPhase(60, actual_tests));
  }

  return aggregateFitness(stats.wins, stats.failures, stats.partial_sum,
                          stats.speed_sum);
}

int AntoninaAPI::solveFitnessBatch(Perceptron *p, int tests_to_run) {
  return solveFitnessBatchImpl(p, tests_to_run);
}

int AntoninaAPI::solveFitnessBatch(NeatGenome *p, int tests_to_run) {
  return solveFitnessBatchImpl(p, tests_to_run);
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

void AntoninaAPI::writeLab(std::ofstream *fout, int ax, int ay, int Ox, int Oy,
                           int gx, int gy, int rn) {
  *fout << ax << " " << ay << " " << Ox << " " << Oy << " " << gx << " " << gy
        << " " << rn << '\n';
}

void AntoninaAPI::writeInFile() {
  std::ofstream fout("Test0.csv");
  int n = 0;

  for (int ax = 1; ax < 7; ax++) {
    for (int ay = 1; ay < 7; ay++) {
      for (int gy = 1; gy < 7; gy++)
        if (gy != ay) {
          n++;
          writeLab(&fout, ax, ay, ax, ay, ax, gy, 0);
        }
      for (int gx = 1; gx < 7; gx++)
        if (gx != ax) {
          n++;
          writeLab(&fout, ax, ay, ax, ay, gx, ay, 0);
        }
    }
  }
  std::cout << n << '\n';

  for (int ax = 1; ax < 7; ax++)
    for (int ay = 1; ay < 7; ay++)
      for (int gx = 1; gx < 7; gx++)
        if (gx != ax)
          for (int gy = 1; gy < 7; gy++)
            if (gy != ay) {
              n++;
              writeLab(&fout, ax, ay, ax, ay, gx, gy, 0);
            }
  std::cout << n << '\n';

  for (int ax = 0; ax < 2; ax++) {
    for (int ay = 0; ay < 8; ay++) {
      if (ax == 1)
        ax = 7;
      for (int gx = 0; gx < 8; gx++)
        for (int gy = 0; gy < 8; gy++)
          if (!(gy == ay && gx == ax)) {
            n++;
            writeLab(&fout, ax, ay, ax, ay, gx, gy, 0);
          }
    }
  }
  for (int ay = 0; ay < 2; ay++) {
    for (int ax = 1; ax < 7; ax++) {
      if (ay == 1)
        ay = 7;
      for (int gx = 0; gx < 8; gx++)
        for (int gy = 0; gy < 8; gy++)
          if (!(gy == ay && gx == ax)) {
            n++;
            writeLab(&fout, ax, ay, ax, ay, gx, gy, 0);
          }
    }
  }
  for (int gx = 0; gx < 2; gx++) {
    for (int gy = 0; gy < 8; gy++) {
      if (gx == 1)
        gx = 7;
      for (int ax = 1; ax < 7; ax++)
        for (int ay = 1; ay < 7; ay++)
          if (!(gy == ay && gx == ax)) {
            n++;
            writeLab(&fout, ax, ay, ax, ay, gx, gy, 0);
          }
    }
  }
  for (int gy = 0; gy < 2; gy++) {
    for (int gx = 1; gx < 7; gx++) {
      if (gy == 1)
        gy = 7;
      for (int ax = 1; ax < 7; ax++)
        for (int ay = 1; ay < 7; ay++)
          if (!(gy == ay && gx == ax)) {
            n++;
            writeLab(&fout, ax, ay, ax, ay, gx, gy, 0);
          }
    }
  }
  std::cout << n << '\n';
  fout.close();
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
  s.initial_r2b = 0;
  s.initial_b2p = 0;
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
    GameState s;
    runScalarGame(p, i, s);
    if (s.result > 0)
      wins++;
  }
  return wins;
}

int AntoninaAPI::collectFailures(Brain *p, int tests_to_run,
                                 std::vector<int> &failures,
                                 int max_failures) {
  int actual_tests =
      tests_to_run > 0 ? std::min(tests_to_run, ALL_TESTS)
                       : std::min(active_tests, ALL_TESTS);

  failures.clear();
  int wins = 0;
  for (int i = 0; i < actual_tests; i++) {
    GameState s;
    runScalarGame(p, i, s);
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

  GameState s;
  int result = runScalarGame(p, test_index, s);
  if (score)
    *score = scoreOfState(s);
  return result;
}
