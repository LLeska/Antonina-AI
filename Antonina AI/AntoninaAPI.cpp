#include "AntoninaAPI.h"
#include <iostream>
#include <thread>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>

using std::cout, std::endl, std::this_thread::sleep_for;
using namespace std::chrono_literals;



AntoninaAPI::AntoninaAPI(){
	std::ifstream fin("Test0.csv");

	if (!fin.is_open()) {
		std::cerr << "Ошибка открытия Test0.csv" << std::endl;
		return;
	}

	int total_tests = 0;
	std::string line;
	while (std::getline(fin, line)) total_tests++;
	if (total_tests == 0) {
		std::cerr << "Test0.csv пустой!" << std::endl;
		fin.close();
		return;
	}
	fin.clear();
	fin.seekg(0);
	for (int i = 0; i < ALL_TESTS; i++) {
		fin >> axarr[i] >> ayarr[i] >> Oxarr[i] >> Oyarr[i] >> gxarr[i] >> gyarr[i] >> rnarr[i];
	}
	fin.clear();
	fin.seekg(0);
	fin.close();
}

char AntoninaAPI::Move(char map[][8], Perceptron* p) {
	double input[64];
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			double v = 0.0;
			switch (map[i][j]) {
			case '.': v = 0.0; break;
			case '@': v = 1.0; break;
			case 'a': v = 0.6; break;
			case '%': v = 0.9; break;
			case '#': v = 0.7; break;
			case 'O': v = 0.3; break;
			default:  v = 0.0; break;
			}
			input[i * 8 + j] = v;
		}
	}

	p->feedForward(input);

	switch (p->getOut()) {
	case 0: return 'u';
	case 1: return 'r';
	case 2: return 'd';
	case 3: return 'l';
	default: return 'u';
	}
}




void AntoninaAPI::ClearLab(char lab[][8])
{
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
			lab[i][j] = '.';
}
void AntoninaAPI::PrintLab(char lab[][8])
{
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++)
			printf("%c ", lab[i][j]); // cout << lab[i][j] << ' ';
		printf("\n");// cout << endl;
	}
	printf("\n");
}
bool AntoninaAPI::MakeLab(char lab[][8], int ax, int ay, int Ox, int Oy, int gx, int gy, int rn, int rx[], int ry[])
{
	ClearLab(lab);
	if ((gx == ax && gy == ay) || (gx == Ox && gy == Oy)) { return false; }

	for (int i = 0; i < rn; i++)
		lab[rx[i]][ry[i]] = '#';
	if (ax == Ox && ay == Oy)
		lab[ax][ay] = '@';
	else
	{
		lab[ax][ay] = 'a';
		lab[Ox][Oy] = 'O';
	}
	lab[gx][gy] = '%';
	return true;
}
void AntoninaAPI::CopyLab(char lab[][8], char copy[][8], int* ax, int* ay, int* Ox, int* Oy, int* gx, int* gy)
{
	for (int i = 0; i < 8; i++)
		for (int j = 0; j < 8; j++)
		{
			copy[i][j] = lab[i][j];
			switch (copy[i][j])
			{
			case 'a': *ax = i; *ay = j; break;
			case '%': *gx = i; *gy = j;	break;
			case 'O': *Ox = i; *Oy = j;	break;
			case '@': *ax = i; *ay = j; *Ox = i; *Oy = j; break;
			default: break;
			}
		}
}

bool AntoninaAPI::MakeLab(char lab[][8], int ax, int ay, int Ox, int Oy, int gx, int gy, int rn)
{
	int rx[64] = { 0 };
	int ry[64] = { 0 };

	if (!MakeLab(lab, ax, ay, Ox, Oy, gx, gy, 0, rx, ry))
	{
		logfile << "GEN-ERR";
		return false;
	}


	int count = rn;
	int stop = 16;
	while (count > 0) {
		for (int i = 0; i < 8; i++)
			for (int j = 0; j < 8; j++)
			{
				if (lab[i][j] == '.' && rand() % 64 < rn && (abs(i - Ox) + abs(j - Oy) > 2))
				{
					lab[i][j] = '#'; count--;
					if (count == 0) return true;
				}
			}
		stop--;
		if (stop == 0)return false;
	}
	return true;
}

int AntoninaAPI::GoTest(char lab[][8], bool doprint, Perceptron *p)
{
	if (doprint) {
		logfile << "#\tNew test... ";
	}
	for (int s = 1; s < STEPS_LIMIT + 1; s++)
	{
		if (doprint)
		{
			sleep_for(std::chrono::milliseconds(TIME_TO_SLEEP));
			//system("CLS");
			printf("Step: %d / %d                       \n", s, STEPS_LIMIT);
			PrintLab(lab);
			for (int l = 0; l < 10; l++)
			{
				printf("\r\033[A");
			}
		}
		char copy[8][8];
		int ax, ay, Ox, Oy, gx, gy;
		CopyLab(lab, copy, &ax, &ay, &Ox, &Oy, &gx, &gy);
		char c = Move(copy, p);
		if (c == 'x') { if (doprint)  logfile << " terminated at step " << s << "!\n"; return -1; } //breaking this ant run
		if (c == 'q') { if (doprint)  logfile << " terminated!\n"; return -2; } //breaking this ant run

		int tox = ax, toy = ay, totox = ax, totoy = ay, pullx = ax, pully = ay;
		bool toB = true, totoB = true, pullB = true;
		switch (c)
		{
		case 'u': tox--; totox -= 2; pullx++; toB = (tox >= 0); totoB = (totox >= 0); pullB = (pullx < 8); break;
		case 'd': tox++; totox += 2; pullx--; toB = (tox < 8); totoB = (totox < 8); pullB = (pullx >= 0); break;
		case 'l': toy--; totoy -= 2; pully++; toB = (toy >= 0); totoB = (totoy >= 0); pullB = (pully < 8); break;
		case 'r': toy++; totoy += 2; pully--; toB = (toy < 8); totoB = (totoy < 8); pullB = (pully >= 0); break;
		default: //FAIL
			break;
		}
		if (!toB) continue; //wall
		if (lab[tox][toy] == '.' || lab[tox][toy] == 'O') //simple move or pull
		{
			lab[tox][toy] = (lab[tox][toy] == '.') ? 'a' : '@';
			if (!pullB || lab[pullx][pully] == '.' || lab[pullx][pully] == 'O') { lab[ax][ay] = (lab[ax][ay] == 'a') ? '.' : 'O'; continue; }//simple move
			if (lab[pullx][pully] == '%' && lab[ax][ay] == '@') { if (doprint)  logfile << " done in " << s << " steps!\n"; return s; }//DONE with pull
			if (lab[pullx][pully] == '#' && lab[ax][ay] == '@') { lab[ax][ay] = 'O';  continue; } //cant pull # to O
			lab[ax][ay] = lab[pullx][pully]; lab[pullx][pully] = '.'; continue;//pull # or %

		}
		if (lab[tox][toy] == '%')
		{
			if (!totoB || lab[totox][totoy] == '#') continue; //wall or blocked
			if (lab[totox][totoy] == '.') { lab[tox][toy] = 'a'; lab[ax][ay] = (lab[ax][ay] == 'a') ? '.' : 'O'; lab[totox][totoy] = '%'; continue; } //push % to .
			if (lab[totox][totoy] == 'O') { if (doprint)  logfile << " done in " << s << " steps!\n";  return s; } //DONE with push
		}
		if (lab[tox][toy] == '#')
		{
			if (!totoB || lab[totox][totoy] != '.') continue; //wall or blocked
			lab[tox][toy] = 'a'; lab[ax][ay] = (lab[ax][ay] == 'a') ? '.' : 'O';; lab[totox][totoy] = '#'; continue;  //push # to .
		}
	}
	if (doprint) logfile << " fail!\n";
	return -1;
}

int AntoninaAPI::GoTest(char lab[][8], int& dis, bool doprint, Perceptron* p)
{
	if (doprint) {
		logfile << "#\tNew test... ";
	}
	int ax, ay, Ox, Oy, gx, gy;
	for (int s = 1; s < STEPS_LIMIT + 1; s++)
	{
		if (doprint)
		{
			sleep_for(std::chrono::milliseconds(TIME_TO_SLEEP));
			//system("CLS");
			printf("Step: %d / %d                       \n", s, STEPS_LIMIT);
			PrintLab(lab);
			for (int l = 0; l < 10; l++)
			{
				printf("\r\033[A");
			}
		}
		char copy[8][8];
		
		CopyLab(lab, copy, &ax, &ay, &Ox, &Oy, &gx, &gy);
		dis = std::min(abs(Ox - gx) + abs(Oy - gy), dis);
		char c = Move(copy, p);
		if (c == 'x') { if (doprint)  logfile << " terminated at step " << s << "!\n"; return -1; } //breaking this ant run
		if (c == 'q') { if (doprint)  logfile << " terminated!\n"; return -2; } //breaking this ant run

		int tox = ax, toy = ay, totox = ax, totoy = ay, pullx = ax, pully = ay;
		bool toB = true, totoB = true, pullB = true;
		switch (c)
		{
		case 'u': tox--; totox -= 2; pullx++; toB = (tox >= 0); totoB = (totox >= 0); pullB = (pullx < 8); break;
		case 'd': tox++; totox += 2; pullx--; toB = (tox < 8); totoB = (totox < 8); pullB = (pullx >= 0); break;
		case 'l': toy--; totoy -= 2; pully++; toB = (toy >= 0); totoB = (totoy >= 0); pullB = (pully < 8); break;
		case 'r': toy++; totoy += 2; pully--; toB = (toy < 8); totoB = (totoy < 8); pullB = (pully >= 0); break;
		default: //FAIL
			break;
		}
		if (!toB) continue; //wall
		if (lab[tox][toy] == '.' || lab[tox][toy] == 'O') //simple move or pull
		{
			lab[tox][toy] = (lab[tox][toy] == '.') ? 'a' : '@';
			if (!pullB || lab[pullx][pully] == '.' || lab[pullx][pully] == 'O') { lab[ax][ay] = (lab[ax][ay] == 'a') ? '.' : 'O'; continue; }//simple move
			if (lab[pullx][pully] == '%' && lab[ax][ay] == '@') { if (doprint)  logfile << " done in " << s << " steps!\n"; dis = 0; return s; }//DONE with pull
			if (lab[pullx][pully] == '#' && lab[ax][ay] == '@') { lab[ax][ay] = 'O';  continue; } //cant pull # to O
			lab[ax][ay] = lab[pullx][pully]; lab[pullx][pully] = '.'; continue;//pull # or %

		}
		if (lab[tox][toy] == '%')
		{
			if (!totoB || lab[totox][totoy] == '#') continue; //wall or blocked
			if (lab[totox][totoy] == '.') { lab[tox][toy] = 'a'; lab[ax][ay] = (lab[ax][ay] == 'a') ? '.' : 'O'; lab[totox][totoy] = '%'; continue; } //push % to .
			if (lab[totox][totoy] == 'O') { if (doprint)  logfile << " done in " << s << " steps!\n"; dis = 0;  return s; } //DONE with push
		}
		if (lab[tox][toy] == '#')
		{
			if (!totoB || lab[totox][totoy] != '.') continue; //wall or blocked
			lab[tox][toy] = 'a'; lab[ax][ay] = (lab[ax][ay] == 'a') ? '.' : 'O';; lab[totox][totoy] = '#'; continue;  //push # to .
		}
	}
	if (doprint) logfile << " fail!\n";
	return -1;
}


void AntoninaAPI::StopAll()
{
	
	exit(0);
}

void AntoninaAPI::writeLab(std::ofstream* fout, int ax, int ay, int Ox, int Oy, int gx, int gy, int rn) {
	*fout << ax << " " << ay << " " << Ox << " " << Oy << " " << gx << " " << gy << " " << rn << '\n';
}

void AntoninaAPI::readLab(std::ifstream* fin, int& ax, int& ay, int& Ox, int& Oy, int& gx, int& gy, int& rn) {
	*fin >> ax >> ay >>  Ox >>  Oy >> gx >> gy >> rn ;
}

void AntoninaAPI::writeInFile() {
	//Test 00 	one line not at walls
	std::ofstream fout("Test0.csv");
	int n = 0;

	// одна линия, не у стены
	for (int ax = 1; ax < 7; ax++) {
		for (int ay = 1; ay < 7; ay ++) {
			for (int gy = 1; gy < 7; gy++) {
				if (gy != ay) {
					n++;
					writeLab(&fout, ax, ay, ax, ay, ax, gy, 0);
				}
			}
			for (int gx = 1; gx < 7; gx++) {
				if (gx != ax) {
					n++;
					writeLab(&fout, ax, ay, ax, ay, gx, ay, 0);
				}
			}			
		}
	}
	// не одна линия, не у стены
	std::cout << n << '\n';
	for (int ax = 1; ax < 7; ax++) {
		for (int ay = 1; ay < 7; ay++){
			for (int gx = 1; gx < 7; gx++) {
				if (gx != ax) {
					for (int gy = 1; gy < 7; gy++) {
						if (gy != ay) {
							n++;
							writeLab(&fout, ax, ay, ax, ay, gx, gy, 0);
						}
					}
				}
			}
		}
	}
	std::cout << n << '\n';
	// у стены 
	for (int ax = 0; ax < 2; ax++) {
		for (int ay = 0; ay < 8; ay++) {
			if (ax == 1) ax = 7;
			for (int gx = 0; gx < 8; gx++) {
				for (int gy = 0; gy < 8; gy++) {
					if (!(gy == ay && gx == ax)) {
						n++;
						writeLab(&fout, ax, ay, ax, ay, gx, gy, 0);
					}
				}
			}
		}
	}
	for (int ay = 0; ay < 2; ay++) {
		for (int ax = 1; ax < 7; ax++) {
			if (ay == 1) ay = 7;
			for (int gx = 0; gx < 8; gx++) {
				for (int gy = 0; gy < 8; gy++) {
					if (!(gy == ay && gx == ax)) {
						n++;
						writeLab(&fout, ax, ay, ax, ay, gx, gy, 0);
					}
				}
			}
		}
	}
	for (int gx = 0; gx < 2; gx++) {
		for (int gy = 0; gy < 8; gy++) {
			if (gx == 1) gx = 7;
			for (int ax = 1; ax < 7; ax++) {
				for (int ay = 1; ay < 7; ay++) {
					if (!(gy == ay && gx == ax)) {
						n++;
						writeLab(&fout, ax, ay, ax, ay, gx, gy, 0);
					}
				}
			}
		}
	}
	for (int gy = 0; gy < 2; gy++) {
		for (int gx = 1; gx < 7; gx++) {
			if (gy == 1) gy = 7;
			for (int ax = 1; ax < 7; ax++) {
				for (int ay = 1; ay < 7; ay++) {
					if (!(gy == ay && gx == ax)) {
						n++;
						writeLab(&fout, ax, ay, ax, ay, gx, gy, 0);
					}
				}
			}
		}
	}
	
	std::cout << n << '\n';


	

	fout.close();
	//end
}

void AntoninaAPI::demonstrate(Perceptron* p)
{

	printf("Starting new Antonina runs!\nSTEPS_LIMIT=%d ", STEPS_LIMIT);
	logfile.open("antlog.txt");
	logfile << "#####\tStarting new Antonina runs!\n#####\tSTEPS_LIMIT=" << STEPS_LIMIT << "\n";
	srand(time(NULL));
	char lab[8][8];
	int rx[64];
	int ry[64];
	int nr = 0;
	int wins;
	int sum;
	int score, wr, as;
	int totalscore = 0;
	//Test 00 	one line not at walls
	logfile << "#####\tStarting Test 00...\n";
	wins = 0; sum = 0;
	for (int i = 0; i < 360; i++)
	{
		int ax, ay, Ox, Oy, gx, gy, rn;
		ax = axarr[i];
		ay = ayarr[i];
		Ox = Oxarr[i];
		Oy = Oyarr[i];
		gx = gxarr[i];
		gy = gyarr[i];
		rn = rnarr[i];
		char lab[8][8];
		MakeLab(lab, ax, ay, ax, ay, gx, gy, 0);
		//test
		printf("Test 00: %d/%d             \n", i, 360);
		int res = GoTest(lab, PRINT_STEPS, p);
		if (res > 0) { wins++; sum += res; }
		else if (res == -2) StopAll();
		printf("\r\033[A");
	}
	score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / 360;
	totalscore += score;
	wr = 100 * wins / 360; as = wins > 0 ? sum / wins : 0;
	printf("Test 00: winrate=%d%% av.steps=%d score=%d\n", wr, as, score);
	logfile << "#####\tTest 00: winrate=" << wr << "% av.steps=" << as << " score=" << score << "\n";

	//end
	for (int l = 0; l < 10; l++)
	{
		printf("\033[B");
	}
	cout << "Total score = " << totalscore << endl;
	logfile << "#####\tAll done!\n";
	logfile.close();
}

int AntoninaAPI::GoTestImproved(char lab[][8], int& min_rover_to_bucket, int& min_bucket_to_pad, bool& bucket_picked, bool doprint, Perceptron* p)
{
	if (doprint) {
		logfile << "#\tNew test... ";
	}

	bucket_picked = false;
	bool rover_has_bucket = false; 

	for (int s = 1; s < STEPS_LIMIT + 1; s++)
	{
		if (doprint)
		{
			sleep_for(std::chrono::milliseconds(TIME_TO_SLEEP));
			printf("Step: %d / %d                       \n", s, STEPS_LIMIT);
			PrintLab(lab);
			for (int l = 0; l < 10; l++)
			{
				printf("\r\033[A");
			}
		}

		char copy[8][8];
		int ax, ay, Ox, Oy, gx, gy;
		CopyLab(lab, copy, &ax, &ay, &Ox, &Oy, &gx, &gy);


		int current_rover_to_bucket = abs(ax - gx) + abs(ay - gy);
		min_rover_to_bucket = std::min(min_rover_to_bucket, current_rover_to_bucket);

		int current_bucket_to_pad = abs(Ox - gx) + abs(Oy - gy);
		min_bucket_to_pad = std::min(min_bucket_to_pad, current_bucket_to_pad);


		if (current_rover_to_bucket == 0 ||
			(ax == gx && abs(ay - gy) == 1) ||
			(ay == gy && abs(ax - gx) == 1)) {
			bucket_picked = true;
			rover_has_bucket = true;
		}

		char c = Move(copy, p);
		if (c == 'x') {
			if (doprint) logfile << " terminated at step " << s << "!\n";
			return -1;
		}
		if (c == 'q') {
			if (doprint) logfile << " terminated!\n";
			return -2;
		}

	
		int tox = ax, toy = ay, totox = ax, totoy = ay, pullx = ax, pully = ay;
		bool toB = true, totoB = true, pullB = true;
		switch (c)
		{
		case 'u': tox--; totox -= 2; pullx++; toB = (tox >= 0); totoB = (totox >= 0); pullB = (pullx < 8); break;
		case 'd': tox++; totox += 2; pullx--; toB = (tox < 8); totoB = (totox < 8); pullB = (pullx >= 0); break;
		case 'l': toy--; totoy -= 2; pully++; toB = (toy >= 0); totoB = (totoy >= 0); pullB = (pully < 8); break;
		case 'r': toy++; totoy += 2; pully--; toB = (toy < 8); totoB = (totoy < 8); pullB = (pully >= 0); break;
		default: break;
		}

		if (!toB) continue;

		if (lab[tox][toy] == '.' || lab[tox][toy] == 'O')
		{
			lab[tox][toy] = (lab[tox][toy] == '.') ? 'a' : '@';
			if (!pullB || lab[pullx][pully] == '.' || lab[pullx][pully] == 'O') {
				lab[ax][ay] = (lab[ax][ay] == 'a') ? '.' : 'O';
				continue;
			}
			if (lab[pullx][pully] == '%' && lab[ax][ay] == '@') {
				if (doprint) logfile << " done in " << s << " steps!\n";
				return s;
			}
			if (lab[pullx][pully] == '#' && lab[ax][ay] == '@') {
				lab[ax][ay] = 'O';
				continue;
			}
			lab[ax][ay] = lab[pullx][pully];
			lab[pullx][pully] = '.';
			continue;
		}

		if (lab[tox][toy] == '%')
		{
			if (!totoB || lab[totox][totoy] == '#') continue;
			if (lab[totox][totoy] == '.') {
				lab[tox][toy] = 'a';
				lab[ax][ay] = (lab[ax][ay] == 'a') ? '.' : 'O';
				lab[totox][totoy] = '%';
				continue;
			}
			if (lab[totox][totoy] == 'O') {
				if (doprint) logfile << " done in " << s << " steps!\n";
				return s;
			}
		}

		if (lab[tox][toy] == '#')
		{
			if (!totoB || lab[totox][totoy] != '.') continue;
			lab[tox][toy] = 'a';
			lab[ax][ay] = (lab[ax][ay] == 'a') ? '.' : 'O';
			lab[totox][totoy] = '#';
			continue;
		}
	}

	if (doprint) logfile << " fail!\n";
	return -1;
}


int AntoninaAPI::solveFitness(Perceptron* p, int tests_to_run) {
	int actual_tests = std::min(tests_to_run, ALL_TESTS);
	if (actual_tests <= 0) actual_tests = ALL_TESTS;

	const int SUCCESS_BASE = 1000;              // Базовая награда за доставку
	const int SUCCESS_STEP_BONUS = 15;          // Бонус за скорость (увеличен!)

	const int DISTANCE_REDUCTION_BONUS = 25;    // За каждую клетку приближения
	const int PICKED_BUCKET_BONUS = 400;        // Бонус за взятие ведра
	const int BUCKET_TO_PAD_BONUS = 20;         // За приближение ведра к площадке

	const int NO_PROGRESS_PENALTY = 5;          // Небольшой штраф за отсутствие прогресса
	const int TERMINATE_PENALTY = 100;          // Штраф за критическую ошибку

	const int BASE_SCORE = 50;                  // Минимальные очки за попытку

	long long total_score = 0;

	for (int i = 0; i < actual_tests; i++) {
		int ax = axarr[i];
		int ay = ayarr[i];
		int Ox = Oxarr[i];
		int Oy = Oyarr[i];
		int gx = gxarr[i];
		int gy = gyarr[i];
		int rn = rnarr[i];

		char lab[8][8];
		MakeLab(lab, ax, ay, ax, ay, gx, gy, 0);

		int initial_rover_to_bucket = abs(ax - gx) + abs(ay - gy);
		int initial_bucket_to_pad = abs(Ox - gx) + abs(Oy - gy);


		int min_rover_to_bucket = initial_rover_to_bucket;
		int min_bucket_to_pad = initial_bucket_to_pad;
		bool bucket_was_picked = false;

		int result = GoTestImproved(lab, min_rover_to_bucket, min_bucket_to_pad,
			bucket_was_picked, false, p);

		int test_score = BASE_SCORE;  

		if (result > 0) {

			test_score += SUCCESS_BASE;

			int steps_saved = std::max(0, STEPS_LIMIT - result);
			test_score += steps_saved * SUCCESS_STEP_BONUS;

			if (result <= STEPS_LIMIT / 2) {
				test_score += 200;  
			}
		}
		else {
			if (result == -2) {
				test_score = std::max(0, test_score - TERMINATE_PENALTY);
			}
			else {
				int rover_progress = initial_rover_to_bucket - min_rover_to_bucket;
				if (rover_progress > 0) {
					test_score += rover_progress * DISTANCE_REDUCTION_BONUS;
				}
				if (bucket_was_picked) {
					test_score += PICKED_BUCKET_BONUS;
					int bucket_progress = initial_bucket_to_pad - min_bucket_to_pad;
					if (bucket_progress > 0) {
						test_score += bucket_progress * BUCKET_TO_PAD_BONUS;
					}
					if (min_bucket_to_pad <= 2) {
						test_score += 150;  
					}
				}
				if (rover_progress <= 0 && !bucket_was_picked) {
					test_score = std::max(BASE_SCORE, test_score - NO_PROGRESS_PENALTY);
				}
			}
		}

		total_score += test_score;
	}

	int avg_score = 0;
	if (actual_tests > 0) {
		avg_score = static_cast<int>(total_score / actual_tests);
	}
	return avg_score;
}
