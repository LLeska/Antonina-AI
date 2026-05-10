#pragma once
#include "Brain.h"
#include "Perceptron.h"
#include <vector>

class NeatGenome;

class AntoninaAPI {
public:
  static constexpr int INPUT_FEATURES = 165;
  static constexpr int ALL_TESTS = 4032;

private:
  std::ofstream logfile;
  const int TIME_TO_SLEEP = 0;
  const bool PRINT_STEPS = true;
  const int STEPS_LIMIT = 40;
  const int N_TESTS = 20;
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

  struct GameState {
    char lab[8][8];

    int ax, ay;
    int Ox, Oy;
    int gx, gy;
    int min_r2b;
    int min_b2p;
    int initial_r2b;
    int initial_b2p;
    bool bucket_picked;
    int shaping_score;
    int invalid_moves;
    bool done;
    int result;
  };

  void ClearLab(char lab[][8]);
  void PrintLab(char lab[][8]);
  bool MakeLab(char lab[][8], int ax, int ay, int Ox, int Oy, int gx, int gy,
               int rn, int rx[], int ry[]);
  void CopyLab(char lab[][8], char copy[][8], int *ax, int *ay, int *Ox,
               int *Oy, int *gx, int *gy);
  bool MakeLab(char lab[][8], int ax, int ay, int Ox, int Oy, int gx, int gy,
               int rn);
  void writeLab(std::ofstream *fout, int ax, int ay, int Ox, int Oy, int gx,
                int gy, int rn);
  void readLab(std::ifstream *fin, int &ax, int &ay, int &Ox, int &Oy, int &gx,
               int &gy, int &rn);
  int GoTest(char lab[][8], bool doprint, Brain *p);
  int GoTest(char lab[][8], int &dis, bool doprint, Brain *p);
  void StopAll();

  char outputToMove(int out);
  void initGameState(GameState &s, int test_index);
  int runScalarGame(Brain *p, int test_index, GameState &s);

  int stepGameState(GameState &s, char c, int step);
  void encodeLabInto(const char lab[][8], double *dst);
  void encodeStateInto(const GameState &s, double *dst) const;

  int scoreOfState(const GameState &s);
  template <typename BatchBrain>
  int solveFitnessBatchImpl(BatchBrain *p, int tests_to_run);

  friend class AntoninaAPITestAccess;

public:
  AntoninaAPI();
  void writeInFile();
  char Move(char map[][8], Brain *p);
  void demonstrate(Brain *p);
  int GoTestImproved(char lab[][8], int &min_rover_to_bucket,
                     int &min_bucket_to_pad, bool &bucket_picked, bool doprint,
                     Brain *p, int &shaping_score);

  int solveFitness(Brain *p, int tests_to_run);

  int solveFitnessBatch(Perceptron *p, int tests_to_run);
  int solveFitnessBatch(NeatGenome *p, int tests_to_run);

public:
  int active_tests = 30;

  void initLabForAnim(int i, char out[][8]);

  int animStep(char lab[][8], Brain *p, int step);
  int countWins(Brain *p, int tests_to_run);
  int collectFailures(Brain *p, int tests_to_run, std::vector<int> &failures,
                      int max_failures);
  int testResult(Brain *p, int test_index, int *score = nullptr);
};
