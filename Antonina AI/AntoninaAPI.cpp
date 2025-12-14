#include "AntoninaAPI.h"
#include <iostream>
#include <thread>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>

using std::cout, std::endl, std::this_thread::sleep_for;
using namespace std::chrono_literals;

//		      YOU CAN SET THESE CONSTS WHILE DEBUGING


AntoninaAPI::AntoninaAPI(){
}

char AntoninaAPI::Move(char map[][8], Perceptron* p) {
	double* input = new double[64];
	for (int i = 0; i < 8; i++) {
		for (int j = 0; j < 8; j++) {
			
			switch (map[i][j]) {
			case '.':
				input[i + j] = 0.1;
			case '@':
				input[i + j] = 1;
			case 'a':
				input[i + j] = 0.5;
			case '%':
				input[i + j] = 0.7;
			case '#':
				input[i + j] = 0.9;
			case 'O':
				input[i + j] = 0.3;
			default:
				input[i + j] = -1.0;
			}
		}
	}
	
	p->feedForward(input);
	delete[] input;
	switch(p->getOut()) {
	case 0:
		return 'u';
	case 1:
		return 'r';
	case 2:
		return 'd';
	default:
		return 'l';
	}
	
}

//	========================================= ! DO NOT TOUCH ! ==========================================
//	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|	|
//	v	v	v	v	v	v	v	v	v	v	v	v	v	v	v	v	v	v	v	v	v	v	v	v	v	v


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


	
	/*
	//Test 03 	all in corners
	//Test 04 	at 1 line with 1 #
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax = rand() % 8, ay = rand() % 8;
		int gx = ax, gy = ay;
		if (rand() % 2 == 0) while (abs(gx - ax) < 2) gx = rand() % 8;
		else while (abs(gy - ay) < 2) gy = rand() % 8;
		rx[0] = (gx + ax) / 2;
		ry[0] = (gy + ay) / 2;
		MakeLab(lab, ax, ay, ax, ay, gx, gy, 1, rx, ry);
		//test
	}
	//Test 05 	not at 1 line with 2 #
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax = rand() % 8, ay = rand() % 8;
		int gx = ax, gy = ay;
		while (gx == ax) gx = rand() % 8;
		while (gy == ay) gy = rand() % 8;
		rx[0] = gx;	ry[0] = ay;
		rx[1] = ax;	ry[1] = gy;
		MakeLab(lab, ax, ay, ax, ay, gx, gy, 2, rx, ry);
		//test
	}
	//Tests 06-09 	not at 1 line with many #
	nr = 4;
	for (int k = 0; k < 4; k++)
	{
		for (int i = 0; i < N_TESTS; i++)
		{
			int ax = rand() % 8, ay = rand() % 8;
			int gx = ax, gy = ay;
			while (gx == ax) gx = rand() % 8;
			while (gy == ay) gy = rand() % 8;
			bool done = false;
			while (!done)
				done = MakeLab(lab, ax, ay, ax, ay, gx, gy, nr);
			//test
		}
	}

	//Test 10 1 block line
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax, ay, gx, gy;
		nr = 8;
		if (rand() % 2 == 0) //vert
		{
			ax = rand() % 8; gx = rand() % 8;
			if (rand() % 2 == 0) { ay = rand() % 2; gy = 6 + rand() % 2; }
			else { ay = 6 + rand() % 2; gy = rand() % 2; }
			int y = 2 + rand() % 4;
			for (int ir = 0; ir < 8; ir++) { rx[ir] = ir; ry[ir] = y; }
		}
		else
		{
			ay = rand() % 8; gy = rand() % 8;
			if (rand() % 2 == 0) { ax = rand() % 2; gx = 6 + rand() % 2; }
			else { ax = 6 + rand() % 2; gx = rand() % 2; }
			int x = 2 + rand() % 4;
			for (int ir = 0; ir < 8; ir++) { rx[ir] = x; ry[ir] = ir; }
		}
		MakeLab(lab, ax, ay, ax, ay, gx, gy, nr, rx, ry);
		//test
	}
	//Test 11 2 block lines
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax, ay, gx, gy;
		nr = 16;
		if (rand() % 2 == 0) //vert
		{
			ax = rand() % 8; gx = rand() % 8;
			if (rand() % 2 == 0) { ay = rand() % 2; gy = 6 + rand() % 2; }
			else { ay = 6 + rand() % 2; gy = rand() % 2; }
			int y = 2 + rand() % 3;
			for (int ir = 0; ir < 8; ir++) { rx[ir * 2] = ir; ry[ir * 2] = y; rx[ir * 2 + 1] = ir; ry[ir * 2 + 1] = y + 1; }
		}
		else
		{
			ay = rand() % 8; gy = rand() % 8;
			if (rand() % 2 == 0) { ax = rand() % 2; gx = 6 + rand() % 2; }
			else { ax = 6 + rand() % 2; gx = rand() % 2; }
			int x = 2 + rand() % 3;
			for (int ir = 0; ir < 8; ir++) { rx[ir * 2] = x; ry[ir * 2] = ir; rx[ir * 2 + 1] = x + 1; ry[ir * 2 + 1] = ir; }
		}
		MakeLab(lab, ax, ay, ax, ay, gx, gy, nr, rx, ry);
		//test
	}

	//Test 12 3 block lines
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax, ay, gx, gy;
		nr = 24;
		if (rand() % 2 == 0) //vert
		{
			ax = rand() % 8; gx = rand() % 8;
			if (rand() % 2 == 0) { ay = rand() % 2; gy = 6 + rand() % 2; }
			else { ay = 6 + rand() % 2; gy = rand() % 2; }
			int x = 2 + rand() % 2;
			for (int ir = 0; ir < 8; ir++)
				for (int ik = 0; ik < 3; ik++) { rx[ir * 3 + ik] = ir; ry[ir * 3 + ik] = x + ik; }
		}
		else
		{
			ay = rand() % 8; gy = rand() % 8;
			if (rand() % 2 == 0) { ax = rand() % 2; gx = 6 + rand() % 2; }
			else { ax = 6 + rand() % 2; gx = rand() % 2; }
			int y = 2 + rand() % 2;
			for (int ir = 0; ir < 8; ir++)
				for (int ik = 0; ik < 3; ik++) { rx[ir * 3 + ik] = y + ik; ry[ir * 3 + ik] = ir; }
		}
		MakeLab(lab, ax, ay, ax, ay, gx, gy, nr, rx, ry);
		//test
	}
	//Test 13 4 block lines
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax, ay, gx, gy;
		nr = 32;
		if (rand() % 2 == 0) //vert
		{
			ax = rand() % 8; gx = rand() % 8;
			if (rand() % 2 == 0) { ay = rand() % 2; gy = 6 + rand() % 2; }
			else { ay = 6 + rand() % 2; gy = rand() % 2; }
			for (int ir = 0; ir < 8; ir++)
				for (int ik = 0; ik < 4; ik++) { rx[ir * 4 + ik] = ir; ry[ir * 4 + ik] = 2 + ik; }
		}
		else
		{
			ay = rand() % 8; gy = rand() % 8;
			if (rand() % 2 == 0) { ax = rand() % 2; gx = 6 + rand() % 2; }
			else { ax = 6 + rand() % 2; gx = rand() % 2; }
			for (int ir = 0; ir < 8; ir++)
				for (int ik = 0; ik < 4; ik++) { rx[ir * 4 + ik] = 2 + ik; ry[ir * 4 + ik] = ir; }
		}
		MakeLab(lab, ax, ay, ax, ay, gx, gy, nr, rx, ry);
		//test
	}*/
	fout.close();
	//end
}

void AntoninaAPI::demonstrate(Perceptron* p)
{
	printf("Starting new Antonina runs!\nSTEPS_LIMIT=%d N_TESTS=%d\n", STEPS_LIMIT, N_TESTS);
	logfile.open("antlog.txt");
	logfile << "#####\tStarting new Antonina runs!\n#####\tSTEPS_LIMIT=" << STEPS_LIMIT << "\tN_TESTS=" << N_TESTS << "\n";
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
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax = 1 + rand() % 6, ay = 1 + rand() % 6;
		int gx = ax, gy = ay;
		if (rand() % 2 == 0) while (gx == ax) gx = 1 + rand() % 6;
		else while (gy == ay) gy = 1 + rand() % 6;
		MakeLab(lab, ax, ay, ax, ay, gx, gy, 0);
		//test
		printf("Test 00: %d/%d             \n", i, N_TESTS);
		int res = GoTest(lab, PRINT_STEPS, p);
		if (res > 0) { wins++; sum += res; }
		else if (res == -2) StopAll();
		printf("\r\033[A");
	}
	score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
	totalscore += score;
	wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
	printf("Test 00: winrate=%d%% av.steps=%d score=%d\n", wr, as, score);
	logfile << "#####\tTest 00: winrate=" << wr << "% av.steps=" << as << " score=" << score << "\n";
	/*
	//Test 01 	no line not at walls
	logfile << "#####\tStarting Test 01...\n";
	wins = 0; sum = 0;
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax = 1 + rand() % 6, ay = 1 + rand() % 6;
		int gx = ax, gy = ay;
		while (gx == ax) gx = 1 + rand() % 6;
		while (gy == ay) gy = 1 + rand() % 6;
		MakeLab(lab, ax, ay, ax, ay, gx, gy, 0);
		//test
		printf("Test 01: %d/%d             \n", i, N_TESTS);
		int res = GoTest(lab, PRINT_STEPS, p);
		if (res > 0) { wins++; sum += res; }
		else if (res == -2) StopAll();
		printf("\r\033[A");
	}
	score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
	totalscore += score;
	wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
	printf("Test 01: winrate=%d%% av.steps=%d score=%d\n", wr, as, score);
	logfile << "#####\tTest 01: winrate=" << wr << "% av.steps=" << as << " score=" << score << "\n";

	//Test 02 	one line O at wall
	logfile << "#####\tStarting Test 02...\n";
	wins = 0; sum = 0;
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax = 1 + rand() % 6, ay = 1 + rand() % 6;
		if (rand() % 2 == 0) ax = (rand() % 2 == 0) ? 0 : 7;
		else ay = (rand() % 2 == 0) ? 0 : 7;
		int gx = ax, gy = ay;
		if (rand() % 2 == 0) while (gx == ax) gx = rand() % 8;
		else while (gy == ay) gy = rand() % 8;
		MakeLab(lab, ax, ay, ax, ay, gx, gy, 0);
		//test
		printf("Test 02: %d/%d             \n", i, N_TESTS);
		int res = GoTest(lab, PRINT_STEPS, p);
		if (res > 0) { wins++; sum += res; }
		else if (res == -2) StopAll();
		printf("\r\033[A");
	}
	score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
	totalscore += score;
	wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
	printf("Test 02: winrate=%d%% av.steps=%d score=%d\n", wr, as, score);
	logfile << "#####\tTest 02: winrate=" << wr << "% av.steps=" << as << " score=" << score << "\n";

	//Test 03 	all in corners
	logfile << "#####\tStarting Test 03...\n";
	wins = 0; sum = 0;
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax = (rand() % 2 == 0) ? 0 : 7, ay = (rand() % 2 == 0) ? 0 : 7;
		int gx = ax, gy = ay;
		while (gx == ax && gy == ay)
		{
			gx = (rand() % 2 == 0) ? 0 : 7;
			gy = (rand() % 2 == 0) ? 0 : 7;
		}
		MakeLab(lab, ax, ay, ax, ay, gx, gy, 0);
		//test
		printf("Test 03: %d/%d             \n", i, N_TESTS);
		int res = GoTest(lab, PRINT_STEPS, p);
		if (res > 0) { wins++; sum += res; }
		else if (res == -2) StopAll();
		printf("\r\033[A");
	}
	score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
	totalscore += score;
	wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
	printf("Test 03: winrate=%d%% av.steps=%d score=%d\n", wr, as, score);
	logfile << "#####\tTest 03: winrate=" << wr << "% av.steps=" << as << " score=" << score << "\n";
	//Test 04 	at 1 line with 1 #
	logfile << "#####\tStarting Test 04...\n";
	wins = 0; sum = 0;
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax = rand() % 8, ay = rand() % 8;
		int gx = ax, gy = ay;
		if (rand() % 2 == 0) while (abs(gx - ax) < 2) gx = rand() % 8;
		else while (abs(gy - ay) < 2) gy = rand() % 8;
		rx[0] = (gx + ax) / 2;
		ry[0] = (gy + ay) / 2;
		MakeLab(lab, ax, ay, ax, ay, gx, gy, 1, rx, ry);
		//test
		printf("Test 04: %d/%d             \n", i, N_TESTS);
		int res = GoTest(lab, PRINT_STEPS, p);
		if (res > 0) { wins++; sum += res; }
		else if (res == -2) StopAll();
		printf("\r\033[A");
	}
	score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
	totalscore += score;
	wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
	printf("Test 04: winrate=%d%% av.steps=%d score=%d\n", wr, as, score);
	logfile << "#####\tTest 04: winrate=" << wr << "% av.steps=" << as << " score=" << score << "\n";

	//Test 05 	not at 1 line with 2 #
	logfile << "#####\tStarting Test 05...\n";
	wins = 0; sum = 0;
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax = rand() % 8, ay = rand() % 8;
		int gx = ax, gy = ay;
		while (gx == ax) gx = rand() % 8;
		while (gy == ay) gy = rand() % 8;
		rx[0] = gx;	ry[0] = ay;
		rx[1] = ax;	ry[1] = gy;
		MakeLab(lab, ax, ay, ax, ay, gx, gy, 2, rx, ry);
		//test
		printf("Test 05: %d/%d             \n", i, N_TESTS);
		int res = GoTest(lab, PRINT_STEPS, p);
		if (res > 0) { wins++; sum += res; }
		else if (res == -2) StopAll();
		printf("\r\033[A");
	}
	score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
	totalscore += score;
	wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
	printf("Test 05: winrate=%d%% av.steps=%d score=%d\n", wr, as, score);
	logfile << "#####\tTest 05: winrate=" << wr << "% av.steps=" << as << " score=" << score << "\n";

	//Tests 06-09 	not at 1 line with many #
	nr = 4;
	for (int k = 0; k < 4; k++)
	{
		logfile << "#####\tStarting Test 0" << k + 6 << "...\n";
		wins = 0;
		sum = 0;
		for (int i = 0; i < N_TESTS; i++)
		{
			int ax = rand() % 8, ay = rand() % 8;
			int gx = ax, gy = ay;
			while (gx == ax) gx = rand() % 8;
			while (gy == ay) gy = rand() % 8;
			bool done = false;
			while (!done)
				done = MakeLab(lab, ax, ay, ax, ay, gx, gy, nr);
			//test
			printf("Test 0%d: %d/%d             \n", 6 + k, i, N_TESTS);
			int res = GoTest(lab, PRINT_STEPS, p);
			if (res > 0) { wins++; sum += res; }
			else if (res == -2) StopAll();
			printf("\r\033[A");
		}
		score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
		totalscore += score;
		wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
		printf("Test 0%d: winrate=%d%% av.steps=%d score=%d\n", 6 + k, wr, as, score);
		logfile << "#####\tTest 0" << k + 6 << ": winrate=" << wr << "% av.steps=" << as << " score=" << score << "\n";
		nr *= 2;
	}

	//Test 10 1 block line
	logfile << "#####\tStarting Test 10...\n";
	wins = 0; sum = 0;
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax, ay, gx, gy;
		nr = 8;
		if (rand() % 2 == 0) //vert
		{
			ax = rand() % 8; gx = rand() % 8;
			if (rand() % 2 == 0) { ay = rand() % 2; gy = 6 + rand() % 2; }
			else { ay = 6 + rand() % 2; gy = rand() % 2; }
			int y = 2 + rand() % 4;
			for (int ir = 0; ir < 8; ir++) { rx[ir] = ir; ry[ir] = y; }
		}
		else
		{
			ay = rand() % 8; gy = rand() % 8;
			if (rand() % 2 == 0) { ax = rand() % 2; gx = 6 + rand() % 2; }
			else { ax = 6 + rand() % 2; gx = rand() % 2; }
			int x = 2 + rand() % 4;
			for (int ir = 0; ir < 8; ir++) { rx[ir] = x; ry[ir] = ir; }
		}
		MakeLab(lab, ax, ay, ax, ay, gx, gy, nr, rx, ry);
		//test
		printf("Test 10: %d/%d             \n", i, N_TESTS);
		int res = GoTest(lab, PRINT_STEPS, p);
		if (res > 0) { wins++; sum += res; }
		else if (res == -2) StopAll();
		printf("\r\033[A");
	}
	score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
	totalscore += score;
	wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
	printf("Test 10: winrate=%d%% av.steps=%d score=%d\n", wr, as, score);
	logfile << "#####\tTest 10: winrate=" << wr << "% av.steps=" << as << " score=" << score << "\n";

	//Test 11 2 block lines
	logfile << "#####\tStarting Test 11...\n";
	wins = 0; sum = 0;
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax, ay, gx, gy;
		nr = 16;
		if (rand() % 2 == 0) //vert
		{
			ax = rand() % 8; gx = rand() % 8;
			if (rand() % 2 == 0) { ay = rand() % 2; gy = 6 + rand() % 2; }
			else { ay = 6 + rand() % 2; gy = rand() % 2; }
			int y = 2 + rand() % 3;
			for (int ir = 0; ir < 8; ir++) { rx[ir * 2] = ir; ry[ir * 2] = y; rx[ir * 2 + 1] = ir; ry[ir * 2 + 1] = y + 1; }
		}
		else
		{
			ay = rand() % 8; gy = rand() % 8;
			if (rand() % 2 == 0) { ax = rand() % 2; gx = 6 + rand() % 2; }
			else { ax = 6 + rand() % 2; gx = rand() % 2; }
			int x = 2 + rand() % 3;
			for (int ir = 0; ir < 8; ir++) { rx[ir * 2] = x; ry[ir * 2] = ir; rx[ir * 2 + 1] = x + 1; ry[ir * 2 + 1] = ir; }
		}
		MakeLab(lab, ax, ay, ax, ay, gx, gy, nr, rx, ry);
		//test
		printf("Test 11: %d/%d             \n", i, N_TESTS);
		int res = GoTest(lab, PRINT_STEPS, p);
		if (res > 0) { wins++; sum += res; }
		else if (res == -2) StopAll();
		printf("\r\033[A");
	}
	score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
	totalscore += score;
	wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
	printf("Test 11: winrate=%d%% av.steps=%d score=%d\n", wr, as, score);
	logfile << "#####\tTest 11: winrate=" << wr << "% av.steps=" << as << " score=" << score << "\n";

	//Test 12 3 block lines
	logfile << "#####\tStarting Test 12...\n";
	wins = 0; sum = 0;
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax, ay, gx, gy;
		nr = 24;
		if (rand() % 2 == 0) //vert
		{
			ax = rand() % 8; gx = rand() % 8;
			if (rand() % 2 == 0) { ay = rand() % 2; gy = 6 + rand() % 2; }
			else { ay = 6 + rand() % 2; gy = rand() % 2; }
			int x = 2 + rand() % 2;
			for (int ir = 0; ir < 8; ir++)
				for (int ik = 0; ik < 3; ik++) { rx[ir * 3 + ik] = ir; ry[ir * 3 + ik] = x + ik; }
		}
		else
		{
			ay = rand() % 8; gy = rand() % 8;
			if (rand() % 2 == 0) { ax = rand() % 2; gx = 6 + rand() % 2; }
			else { ax = 6 + rand() % 2; gx = rand() % 2; }
			int y = 2 + rand() % 2;
			for (int ir = 0; ir < 8; ir++)
				for (int ik = 0; ik < 3; ik++) { rx[ir * 3 + ik] = y + ik; ry[ir * 3 + ik] = ir; }
		}
		MakeLab(lab, ax, ay, ax, ay, gx, gy, nr, rx, ry);
		//test
		printf("Test 12: %d/%d             \n", i, N_TESTS);
		int res = GoTest(lab, PRINT_STEPS, p);
		if (res > 0) { wins++; sum += res; }
		else if (res == -2) StopAll();
		printf("\r\033[A");
	}
	score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
	totalscore += score;
	wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
	printf("Test 12: winrate=%d%% av.steps=%d score=%d\n", wr, as, score);
	logfile << "#####\tTest 12: winrate=" << wr << "% av.steps=" << as << " score=" << score << "\n";

	//Test 13 4 block lines
	logfile << "#####\tStarting Test 13...\n";
	wins = 0; sum = 0;
	for (int i = 0; i < N_TESTS; i++)
	{
		int ax, ay, gx, gy;
		nr = 32;
		if (rand() % 2 == 0) //vert
		{
			ax = rand() % 8; gx = rand() % 8;
			if (rand() % 2 == 0) { ay = rand() % 2; gy = 6 + rand() % 2; }
			else { ay = 6 + rand() % 2; gy = rand() % 2; }
			for (int ir = 0; ir < 8; ir++)
				for (int ik = 0; ik < 4; ik++) { rx[ir * 4 + ik] = ir; ry[ir * 4 + ik] = 2 + ik; }
		}
		else
		{
			ay = rand() % 8; gy = rand() % 8;
			if (rand() % 2 == 0) { ax = rand() % 2; gx = 6 + rand() % 2; }
			else { ax = 6 + rand() % 2; gx = rand() % 2; }
			for (int ir = 0; ir < 8; ir++)
				for (int ik = 0; ik < 4; ik++) { rx[ir * 4 + ik] = 2 + ik; ry[ir * 4 + ik] = ir; }
		}
		MakeLab(lab, ax, ay, ax, ay, gx, gy, nr, rx, ry);
		//test
		printf("Test 13: %d/%d             \n", i, N_TESTS);
		int res = GoTest(lab, PRINT_STEPS, p);
		if (res > 0) { wins++; sum += res; }
		else if (res == -2) StopAll();
		printf("\r\033[A");
	}
	score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
	totalscore += score;
	wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
	printf("Test 13: winrate=%d%% av.steps=%d score=%d\n", wr, as, score);
	logfile << "#####\tTest 13: winrate=" << wr << "% av.steps=" << as << " score=" << score << "\n";
	*/
	//end
	cout << "Total score = " << totalscore << endl;
	logfile << "#####\tAll done!\n";
	logfile.close();
}

int* AntoninaAPI::solveFitness(Perceptron** neuros, int population, int right) {
	int* fitness = new int[population];
	for (int neur = 0; neur < population; neur++) {
		Perceptron* p = &(*neuros)[neur];
		srand(time(NULL));
		char lab[8][8];
		int rx[64];
		int ry[64];
		int nr = 0;
		int wins=0;
		int sum=0;
		int score = 0;
		int wr, as;
		int totalscore = 0;
		int maxscore = 0;
		int k = 10000;
		//Test 00 	one line not at walls
		std::ifstream fin("Test0.csv");
		for (int j = 0; j < right; j++) {//448 4032
			//wins = 0; sum = 0;
			for (int i = 0; i < N_TESTS; i++)//343 29
			{
				int winbonus = 0;
				int stepbonus = 0;
				int dist_bonus = 0;
				int stupid_bonus = 0;
				
				int ax, ay, gx, gy, Ox, Oy, rn;
				readLab(&fin, ax, ay, Ox, Oy, gx, gy, rn);
				int s0 = abs(Ox - gx) + abs(Oy - gy);
				MakeLab(lab, ax, ay, ax, ay, gx, gy, rn);
				//test
				int s = s0;
				int res = GoTest(lab, s, false, p);
				if (res > 0)
				{ 
					winbonus = 500 * k;
					stepbonus = log(STEPS_LIMIT / res) * 5 * k;
					dist_bonus = exp(s0) * 100 * k;
					wins++;
					sum += res;
				}
				else if (res == -2) StopAll();
				else if (s0 - s > 0) dist_bonus = exp(s0) * 100 * k;
				else if (s0 - s < 0) {
					dist_bonus = (s0 - s) * 50 * k;
					stupid_bonus = 10 * k;
				}
				else stupid_bonus = 50 * k;
				score += 0.5 * winbonus + stepbonus * 0.1 + 2 * dist_bonus + stupid_bonus;
			}
			//score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;

			//wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
			totalscore += score;
			//totalscore += wr;
		}
		/*
		//Test 01 	no line not at walls
		wins = 0; sum = 0;
		for (int i = 0; i < N_TESTS; i++)
		{
			int ax = 1 + rand() % 6, ay = 1 + rand() % 6;
			int gx = ax, gy = ay;
			while (gx == ax) gx = 1 + rand() % 6;
			while (gy == ay) gy = 1 + rand() % 6;
			MakeLab(lab, ax, ay, ax, ay, gx, gy, 0);
			//test
			int res = GoTest(lab, false, p);
			if (res > 0) { wins++; sum += res; }
			else if (res == -2) StopAll();
		}
		score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
		totalscore += score;
		wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
		//Test 02 	one line O at wall
		wins = 0; sum = 0;
		for (int i = 0; i < N_TESTS; i++)
		{
			int ax = 1 + rand() % 6, ay = 1 + rand() % 6;
			if (rand() % 2 == 0) ax = (rand() % 2 == 0) ? 0 : 7;
			else ay = (rand() % 2 == 0) ? 0 : 7;
			int gx = ax, gy = ay;
			if (rand() % 2 == 0) while (gx == ax) gx = rand() % 8;
			else while (gy == ay) gy = rand() % 8;
			MakeLab(lab, ax, ay, ax, ay, gx, gy, 0);
			//test
			int res = GoTest(lab, false, p);
			if (res > 0) { wins++; sum += res; }
			else if (res == -2) StopAll();
		}
		score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
		totalscore += score;
		wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
		//Test 03 	all in corners
		/*
		wins = 0; sum = 0;
		for (int i = 0; i < N_TESTS; i++)
		{
			int ax = (rand() % 2 == 0) ? 0 : 7, ay = (rand() % 2 == 0) ? 0 : 7;
			int gx = ax, gy = ay;
			while (gx == ax && gy == ay)
			{
				gx = (rand() % 2 == 0) ? 0 : 7;
				gy = (rand() % 2 == 0) ? 0 : 7;
			}
			MakeLab(lab, ax, ay, ax, ay, gx, gy, 0);
			//test
			int res = GoTest(lab, false, p);
			if (res > 0) { wins++; sum += res; }
			else if (res == -2) StopAll();
		}
		score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
		totalscore += score;
		wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;/*
		//Test 04 	at 1 line with 1 #
		wins = 0; sum = 0;
		for (int i = 0; i < N_TESTS; i++)
		{
			int ax = rand() % 8, ay = rand() % 8;
			int gx = ax, gy = ay;
			if (rand() % 2 == 0) while (abs(gx - ax) < 2) gx = rand() % 8;
			else while (abs(gy - ay) < 2) gy = rand() % 8;
			rx[0] = (gx + ax) / 2;
			ry[0] = (gy + ay) / 2;
			MakeLab(lab, ax, ay, ax, ay, gx, gy, 1, rx, ry);
			//test
			int res = GoTest(lab, false, p);
			if (res > 0) { wins++; sum += res; }
			else if (res == -2) StopAll();
		}
		score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
		totalscore += score;
		wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;

		//Test 05 	not at 1 line with 2 #
		wins = 0; sum = 0;
		for (int i = 0; i < N_TESTS; i++)
		{
			int ax = rand() % 8, ay = rand() % 8;
			int gx = ax, gy = ay;
			while (gx == ax) gx = rand() % 8;
			while (gy == ay) gy = rand() % 8;
			rx[0] = gx;	ry[0] = ay;
			rx[1] = ax;	ry[1] = gy;
			MakeLab(lab, ax, ay, ax, ay, gx, gy, 2, rx, ry);
			//test
			int res = GoTest(lab, false, p);
			if (res > 0) { wins++; sum += res; }
			else if (res == -2) StopAll();
		}
		score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
		totalscore += score;
		wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
		//Tests 06-09 	not at 1 line with many #
		nr = 4;
		for (int k = 0; k < 4; k++)
		{
			wins = 0;
			sum = 0;
			for (int i = 0; i < N_TESTS; i++)
			{
				int ax = rand() % 8, ay = rand() % 8;
				int gx = ax, gy = ay;
				while (gx == ax) gx = rand() % 8;
				while (gy == ay) gy = rand() % 8;
				bool done = false;
				while (!done)
					done = MakeLab(lab, ax, ay, ax, ay, gx, gy, nr);
				//test
				int res = GoTest(lab, false, p);
				if (res > 0) { wins++; sum += res; }
				else if (res == -2) StopAll();
			}
			score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
			totalscore += score;
			wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
			nr *= 2;
		}

		//Test 10 1 block line
		wins = 0; sum = 0;
		for (int i = 0; i < N_TESTS; i++)
		{
			int ax, ay, gx, gy;
			nr = 8;
			if (rand() % 2 == 0) //vert
			{
				ax = rand() % 8; gx = rand() % 8;
				if (rand() % 2 == 0) { ay = rand() % 2; gy = 6 + rand() % 2; }
				else { ay = 6 + rand() % 2; gy = rand() % 2; }
				int y = 2 + rand() % 4;
				for (int ir = 0; ir < 8; ir++) { rx[ir] = ir; ry[ir] = y; }
			}
			else
			{
				ay = rand() % 8; gy = rand() % 8;
				if (rand() % 2 == 0) { ax = rand() % 2; gx = 6 + rand() % 2; }
				else { ax = 6 + rand() % 2; gx = rand() % 2; }
				int x = 2 + rand() % 4;
				for (int ir = 0; ir < 8; ir++) { rx[ir] = x; ry[ir] = ir; }
			}
			MakeLab(lab, ax, ay, ax, ay, gx, gy, nr, rx, ry);
			//test
			int res = GoTest(lab, false, p);
			if (res > 0) { wins++; sum += res; }
			else if (res == -2) StopAll();
		}
		score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
		totalscore += score;
		wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;

		//Test 11 2 block lines
		wins = 0; sum = 0;
		for (int i = 0; i < N_TESTS; i++)
		{
			int ax, ay, gx, gy;
			nr = 16;
			if (rand() % 2 == 0) //vert
			{
				ax = rand() % 8; gx = rand() % 8;
				if (rand() % 2 == 0) { ay = rand() % 2; gy = 6 + rand() % 2; }
				else { ay = 6 + rand() % 2; gy = rand() % 2; }
				int y = 2 + rand() % 3;
				for (int ir = 0; ir < 8; ir++) { rx[ir * 2] = ir; ry[ir * 2] = y; rx[ir * 2 + 1] = ir; ry[ir * 2 + 1] = y + 1; }
			}
			else
			{
				ay = rand() % 8; gy = rand() % 8;
				if (rand() % 2 == 0) { ax = rand() % 2; gx = 6 + rand() % 2; }
				else { ax = 6 + rand() % 2; gx = rand() % 2; }
				int x = 2 + rand() % 3;
				for (int ir = 0; ir < 8; ir++) { rx[ir * 2] = x; ry[ir * 2] = ir; rx[ir * 2 + 1] = x + 1; ry[ir * 2 + 1] = ir; }
			}
			MakeLab(lab, ax, ay, ax, ay, gx, gy, nr, rx, ry);
			//test
			int res = GoTest(lab, false, p);
			if (res > 0) { wins++; sum += res; }
			else if (res == -2) StopAll();
		}
		score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
		totalscore += score;
		wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;

		//Test 12 3 block lines
		wins = 0; sum = 0;
		for (int i = 0; i < N_TESTS; i++)
		{
			int ax, ay, gx, gy;
			nr = 24;
			if (rand() % 2 == 0) //vert
			{
				ax = rand() % 8; gx = rand() % 8;
				if (rand() % 2 == 0) { ay = rand() % 2; gy = 6 + rand() % 2; }
				else { ay = 6 + rand() % 2; gy = rand() % 2; }
				int x = 2 + rand() % 2;
				for (int ir = 0; ir < 8; ir++)
					for (int ik = 0; ik < 3; ik++) { rx[ir * 3 + ik] = ir; ry[ir * 3 + ik] = x + ik; }
			}
			else
			{
				ay = rand() % 8; gy = rand() % 8;
				if (rand() % 2 == 0) { ax = rand() % 2; gx = 6 + rand() % 2; }
				else { ax = 6 + rand() % 2; gx = rand() % 2; }
				int y = 2 + rand() % 2;
				for (int ir = 0; ir < 8; ir++)
					for (int ik = 0; ik < 3; ik++) { rx[ir * 3 + ik] = y + ik; ry[ir * 3 + ik] = ir; }
			}
			MakeLab(lab, ax, ay, ax, ay, gx, gy, nr, rx, ry);
			//test
			int res = GoTest(lab, false, p);
			if (res > 0) { wins++; sum += res; }
			else if (res == -2) StopAll();
		}
		score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
		totalscore += score;
		wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;

		//Test 13 4 block lines
		wins = 0; sum = 0;
		for (int i = 0; i < N_TESTS; i++)
		{
			int ax, ay, gx, gy;
			nr = 32;
			if (rand() % 2 == 0) //vert
			{
				ax = rand() % 8; gx = rand() % 8;
				if (rand() % 2 == 0) { ay = rand() % 2; gy = 6 + rand() % 2; }
				else { ay = 6 + rand() % 2; gy = rand() % 2; }
				for (int ir = 0; ir < 8; ir++)
					for (int ik = 0; ik < 4; ik++) { rx[ir * 4 + ik] = ir; ry[ir * 4 + ik] = 2 + ik; }
			}
			else
			{
				ay = rand() % 8; gy = rand() % 8;
				if (rand() % 2 == 0) { ax = rand() % 2; gx = 6 + rand() % 2; }
				else { ax = 6 + rand() % 2; gx = rand() % 2; }
				for (int ir = 0; ir < 8; ir++)
					for (int ik = 0; ik < 4; ik++) { rx[ir * 4 + ik] = 2 + ik; ry[ir * 4 + ik] = ir; }
			}
			MakeLab(lab, ax, ay, ax, ay, gx, gy, nr, rx, ry);
			//test
			int res = GoTest(lab, false, p);
			if (res > 0) { wins++; sum += res; }
			else if (res == -2) StopAll();
		}
		score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / N_TESTS;
		totalscore += score;
		wr = 100 * wins / N_TESTS; as = wins > 0 ? sum / wins : 0;
		*/
		//end
		wr = (1000000 * wins) / (N_TESTS*right); as = wins > 0 ? sum / wins : 0;
		//totalscore += score;
		//totalscore += wr;
		fitness[neur] = score;
	}
	return fitness;
}
