#pragma once
#include "Perceptron.h"

class AntoninaAPI
{
private:
	std::ofstream logfile;
	//		      YOU CAN SET THESE CONSTS WHILE DEBUGING
	const int TIME_TO_SLEEP = 0;					//  <======== Animation delay (ms)
	const bool PRINT_STEPS = true;					//  <======== Enable / disable animation
	const int STEPS_LIMIT = 40;					//  <======== Steps limit for each run
	const int N_TESTS = 20;
	void ClearLab(char lab[][8]);
	void PrintLab(char lab[][8]);
	bool MakeLab(char lab[][8], int ax, int ay, int Ox, int Oy, int gx, int gy, int rn, int rx[], int ry[]);
	void CopyLab(char lab[][8], char copy[][8], int* ax, int* ay, int* Ox, int* Oy, int* gx, int* gy);
	bool MakeLab(char lab[][8], int ax, int ay, int Ox, int Oy, int gx, int gy, int rn);
	void writeLab(std::ofstream* fout, int ax, int ay, int Ox, int Oy, int gx, int gy, int rn);
	void readLab(std::ifstream* fin, int& ax, int& ay, int& Ox, int& Oy, int& gx, int& gy, int& rn);
	int GoTest(char lab[][8], bool doprint, Perceptron* p);
	int GoTest(char lab[][8], int& dis, bool doprint, Perceptron* p);
	void StopAll();
	public:
	AntoninaAPI();
	void writeInFile();
	char Move(char map[][8], Perceptron* p);
	void demonstrate(Perceptron* p);
	int* solveFitness(Perceptron** neuros, int population, int tests_to_run);
};

