#pragma once
#include <array>
#include <fstream>
#include <string>
#include <vector>

class NeatGenome;
class Perceptron;
class Brain;
class AntoninaAPI;

template <typename BatchBrain>
int solveFitnessBatchImpl(AntoninaAPI &api, BatchBrain *p, int tests_to_run, int *case_scores = nullptr);

class AntoninaAPI {
public:
  static constexpr int INPUT_FEATURES = 208;
  static constexpr int ALL_TESTS = 4032;

  struct TestTrace {
    struct Step {
      int step = 0;
      int ax = 0;
      int ay = 0;
      int Ox = 0;
      int Oy = 0;
      int gx = 0;
      int gy = 0;
      int solution = -1;
      int raw_move = -1;
      int selected_move = -1;
      int result = 0;
      int invalid_moves = 0;
      bool bucket_picked = false;
      std::array<double, 4> outputs{};
      std::array<bool, 4> valid{};
    };

    int test_index = 0;
    int result = -1;
    int score = 0;
    int steps = 0;
    int initial_r2b = 0;
    int min_r2b = 0;
    int initial_b2p = 0;
    int min_b2p = 0;
    int initial_control = 0;
    int min_control = 0;
    int initial_solution = 0;
    int min_solution = 0;
    int current_solution = 0;
    int invalid_moves = 0;
    int masked_moves = 0;
    bool bucket_picked = false;
    int ax = 0;
    int ay = 0;
    int Ox = 0;
    int Oy = 0;
    int gx = 0;
    int gy = 0;
    int stones = 0;
    std::array<int, 4> moves{};
    int last_move = -1;
    std::vector<Step> steps_detail;
  };

private:
  std::ofstream logfile;
  const int TIME_TO_SLEEP = 0;
  const bool PRINT_STEPS = true;
  const int STEPS_LIMIT = 40;
  int axarr[ALL_TESTS];
  int ayarr[ALL_TESTS];
  int Oxarr[ALL_TESTS];
  int Oyarr[ALL_TESTS];
  int gxarr[ALL_TESTS];
  int gyarr[ALL_TESTS];
  int rnarr[ALL_TESTS];
  char prebuilt_labs[ALL_TESTS][8][8];
  int prebuilt_initial_r2b[ALL_TESTS];
  int prebuilt_initial_b2p[ALL_TESTS];
  int prebuilt_initial_control[ALL_TESTS];
  bool prebuilt_stone_cells[ALL_TESTS][8][8];

  struct GameState {
    char lab[8][8];

    int ax, ay;
    int Ox, Oy;
    int gx, gy;
    int min_r2b;
    int min_b2p;
    int min_control;
    int initial_r2b;
    int initial_b2p;
    int initial_control;
    bool bucket_picked;
    int shaping_score;
    int invalid_moves;
    bool done;
    int result;
  };

  struct FastGameState {
    int ax, ay;
    int Ox, Oy;
    int gx, gy;
    int min_r2b;
    int min_b2p;
    int min_control;
    int initial_r2b;
    int initial_b2p;
    int initial_control;
    int initial_solution;
    int min_solution;
    int stones;
    bool stone_cells[8][8];
    bool bucket_picked;
    int shaping_score;
    int invalid_moves;
    bool done;
    int result;
  };

  void ClearLab(char lab[][8]);
  void PrintLab(char lab[][8]);
  bool MakeLab(char lab[][8], int ax, int ay, int Ox, int Oy, int gx, int gy, int rn, int rx[], int ry[]);
  void CopyLab(char lab[][8], char copy[][8], int *ax, int *ay, int *Ox, int *Oy, int *gx, int *gy);
  bool MakeLab(char lab[][8], int ax, int ay, int Ox, int Oy, int gx, int gy, int rn);
  static char outputToMove(int out);
  void initGameState(GameState &s, int test_index);
  void initFastGameState(FastGameState &s, int test_index) const;
  int runScalarGame(Brain *p, int test_index, GameState &s);
  int runFastGame(Brain *p, int test_index, FastGameState &s) const;

  int stepGameState(GameState &s, char c, int step);
  int stepFastGameState(FastGameState &s, char c, int step) const;
  int stepFastGameStateMove(FastGameState &s, int move, int step) const;
  void validFastMoves(const FastGameState &s, bool *valid_moves) const;
  int selectValidMove(const FastGameState &s, const double *outputs, int fallback) const;
  void encodeLabInto(const char lab[][8], double *dst);
  void encodeStateInto(const GameState &s, double *dst) const;
  void encodeFastStateInto(const FastGameState &s, double *dst) const;

  int scoreOfState(const GameState &s);
  int scoreOfFastState(const FastGameState &s) const;
  template <typename BatchBrain>
  friend int solveFitnessBatchImpl(AntoninaAPI &api, BatchBrain *p, int tests_to_run, int *case_scores);

  friend class AntoninaAPITestAccess;

public:
  AntoninaAPI();
  bool reloadTests(std::string *error = nullptr, std::string *loaded_path = nullptr);
  char Move(char map[][8], Brain *p);
  void demonstrate(Brain *p);
  int GoTestImproved(char lab[][8], int &min_rover_to_bucket, int &min_bucket_to_pad, bool &bucket_picked, bool doprint, Brain *p, int &shaping_score);

  int solveFitness(Brain *p, int tests_to_run);

  int solveFitnessBatch(Perceptron *p, int tests_to_run);
  int solveFitnessBatch(NeatGenome *p, int tests_to_run);
  int solveFitnessBatch(NeatGenome *p, int tests_to_run, int *case_scores);

public:
  int active_tests = 30;

  void initLabForAnim(int i, char out[][8]);

  int animStep(char lab[][8], Brain *p, int step);
  int countWins(Brain *p, int tests_to_run);
  int collectFailures(Brain *p, int tests_to_run, std::vector<int> &failures, int max_failures);
  int testResult(Brain *p, int test_index, int *score = nullptr);
  int traceTest(Brain *p, int test_index, TestTrace &trace);
};
