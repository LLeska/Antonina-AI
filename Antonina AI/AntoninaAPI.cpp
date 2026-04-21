#include "AntoninaAPI.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>

using std::cout, std::endl, std::this_thread::sleep_for;
using namespace std::chrono_literals;


AntoninaAPI::AntoninaAPI() {
    std::ifstream fin("Test0.csv");
    if (!fin.is_open()) {
        std::cerr << "Error opening Test0.csv" << std::endl;
        return;
    }

    int total_tests = 0;
    std::string line;
    while (std::getline(fin, line)) total_tests++;
    if (total_tests == 0) {
        std::cerr << "Test0.csv is empty!" << std::endl;
        fin.close();
        return;
    }

    fin.clear();
    fin.seekg(0);
    for (int i = 0; i < ALL_TESTS; i++)
        fin >> axarr[i] >> ayarr[i] >> Oxarr[i] >> Oyarr[i] >> gxarr[i] >> gyarr[i] >> rnarr[i];
    fin.close();

    for (int i = 0; i < ALL_TESTS; i++) {
        MakeLab(prebuilt_labs[i], axarr[i], ayarr[i], axarr[i], ayarr[i], gxarr[i], gyarr[i], 0);
        prebuilt_initial_r2b[i] = abs(axarr[i] - Oxarr[i]) + abs(ayarr[i] - Oyarr[i]);
        prebuilt_initial_b2p[i] = abs(Oxarr[i] - gxarr[i]) + abs(Oyarr[i] - gyarr[i]);
    }
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
    if ((gx == ax && gy == ay) || (gx == Ox && gy == Oy)) return false;
    for (int i = 0; i < rn; i++) lab[rx[i]][ry[i]] = '#';
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
    int rx[64] = { 0 }, ry[64] = { 0 };
    if (!MakeLab(lab, ax, ay, Ox, Oy, gx, gy, 0, rx, ry)) {
        logfile << "GEN-ERR";
        return false;
    }
    int count = rn, stop = 16;
    while (count > 0) {
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                if (lab[i][j] == '.' && rand() % 64 < rn && (abs(i - Ox) + abs(j - Oy) > 2)) {
                    lab[i][j] = '#'; count--;
                    if (count == 0) return true;
                }
        if (--stop == 0) return false;
    }
    return true;
}

void AntoninaAPI::CopyLab(char lab[][8], char copy[][8], int* ax, int* ay, int* Ox, int* Oy, int* gx, int* gy) {
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            copy[i][j] = lab[i][j];
            switch (copy[i][j]) {
            case 'a': *ax = i; *ay = j; break;
            case '%': *gx = i; *gy = j; break;
            case 'O': *Ox = i; *Oy = j; break;
            case '@': *ax = i; *ay = j; *Ox = i; *Oy = j; break;
            default: break;
            }
        }
}


char AntoninaAPI::Move(char map[][8], Perceptron* p) {
    double input[64];
    for (int i = 0; i < 8; i++)
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
    p->feedForward(input);
    switch (p->getOut()) {
    case 0: return 'u';
    case 1: return 'r';
    case 2: return 'd';
    case 3: return 'l';
    default: return 'u';
    }
}


int AntoninaAPI::GoTestImproved(char lab[][8], int& min_rover_to_bucket, int& min_bucket_to_pad, bool& bucket_picked, bool doprint, Perceptron* p, int& shaping_score)
{
    if (doprint) logfile << "#\tNew test... ";
    bucket_picked = false;
    shaping_score = 0;

    for (int s = 1; s < STEPS_LIMIT + 1; s++) {
        if (doprint) {
            sleep_for(std::chrono::milliseconds(TIME_TO_SLEEP));
            printf("Step: %d / %d\n", s, STEPS_LIMIT);
            PrintLab(lab);
            for (int l = 0; l < 10; l++) printf("\r\033[A");
        }

        char copy[8][8];
        int ax, ay, Ox, Oy, gx, gy;
        CopyLab(lab, copy, &ax, &ay, &Ox, &Oy, &gx, &gy);

        min_rover_to_bucket = std::min(min_rover_to_bucket, abs(ax - Ox) + abs(ay - Oy));
        min_bucket_to_pad = std::min(min_bucket_to_pad, abs(Ox - gx) + abs(Oy - gy));

        int old_d_box_goal = abs(Ox - gx) + abs(Oy - gy);
        int old_d_agent_box = abs(ax - Ox) + abs(ay - Oy);
        int prev_Ox = Ox, prev_Oy = Oy;

        char c = Move(copy, p);
        if (c == 'x') { if (doprint) logfile << " terminated!\n"; return -1; }
        if (c == 'q') { if (doprint) logfile << " terminated!\n"; return -2; }

        int tox = ax, toy = ay, totox = ax, totoy = ay, pullx = ax, pully = ay;
        bool toB = true, totoB = true, pullB = true;
        switch (c) {
        case 'u': tox--; totox -= 2; pullx++; toB = (tox >= 0);  totoB = (totox >= 0); pullB = (pullx < 8); break;
        case 'd': tox++; totox += 2; pullx--; toB = (tox < 8);   totoB = (totox < 8);  pullB = (pullx >= 0); break;
        case 'l': toy--; totoy -= 2; pully++; toB = (toy >= 0);  totoB = (totoy >= 0); pullB = (pully < 8); break;
        case 'r': toy++; totoy += 2; pully--; toB = (toy < 8);   totoB = (totoy < 8);  pullB = (pully >= 0); break;
        default: break;
        }
        if (!toB) continue;

        auto apply_shaping = [&]() {
            int nax, nay, nOx, nOy, ngx2, ngy2;
            char tmp[8][8];
            CopyLab(lab, tmp, &nax, &nay, &nOx, &nOy, &ngx2, &ngy2);
            if ((nOx != prev_Ox || nOy != prev_Oy) && !bucket_picked) {
                bucket_picked = true;
            }
            shaping_score += (old_d_box_goal - abs(nOx - ngx2)) * 10;
            shaping_score += (old_d_agent_box - abs(nax - nOx)) * 2;
            };

        if (lab[tox][toy] == '.' || lab[tox][toy] == 'O') {
            lab[tox][toy] = (lab[tox][toy] == 'O') ? '@' : 'a';
            if (!pullB || lab[pullx][pully] == '.' || lab[pullx][pully] == 'O') {
                lab[ax][ay] = (lab[ax][ay] == 'a') ? '.' : 'O';
            }
            else if (lab[pullx][pully] == '%' && lab[ax][ay] == '@') {
                if (doprint) logfile << " done in " << s << " steps!\n";
                apply_shaping();
                return s;
            }
            else if (lab[pullx][pully] == '#' && lab[ax][ay] == '@') {
                lab[ax][ay] = 'O';
            }
            else {
                lab[ax][ay] = lab[pullx][pully];
                lab[pullx][pully] = '.';
            }
        }
        else if (lab[tox][toy] == '%') {
            if (!totoB || lab[totox][totoy] == '#') continue;
            if (lab[totox][totoy] == '.') {
                lab[tox][toy] = 'a';
                lab[ax][ay] = (lab[ax][ay] == 'a') ? '.' : 'O';
                lab[totox][totoy] = '%';
            }
            else if (lab[totox][totoy] == 'O') {
                if (doprint) logfile << " done in " << s << " steps!\n";
                apply_shaping();
                return s;
            }
            continue;
        }
        else if (lab[tox][toy] == '#') {
            if (!totoB || lab[totox][totoy] != '.') continue;
            lab[tox][toy] = 'a';
            lab[ax][ay] = (lab[ax][ay] == 'a') ? '.' : 'O';
            lab[totox][totoy] = '#';
            continue;
        }

        apply_shaping();
    }

    if (doprint) logfile << " fail!\n";
    return -1;
}


int AntoninaAPI::solveFitness(Perceptron* p, int tests_to_run) {
    int actual_tests = std::min(tests_to_run, ALL_TESTS);
    if (actual_tests <= 0) actual_tests = active_tests;

    const int SUCCESS_BASE = 1000;
    const int SUCCESS_STEP_BONUS = 15;
    const int DISTANCE_REDUCTION_BONUS = 25;
    const int PICKED_BUCKET_BONUS = 400;
    const int BUCKET_TO_PAD_BONUS = 20;
    const int NO_PROGRESS_PENALTY = 5;
    const int TERMINATE_PENALTY = 100;
    const int BASE_SCORE = 50;

    long long total_score = 0;

    for (int i = 0; i < actual_tests; i++) {
        char lab[8][8];
        memcpy(lab, prebuilt_labs[i], sizeof(lab));

        int initial_r2b = prebuilt_initial_r2b[i];
        int initial_b2p = prebuilt_initial_b2p[i];
        int min_r2b = initial_r2b;
        int min_b2p = initial_b2p;
        bool bucket_picked = false;
        int shaping_score = 0;

        int result = GoTestImproved(lab, min_r2b, min_b2p, bucket_picked, false, p, shaping_score);

        int test_score = BASE_SCORE;
        if (result > 0) {
            test_score += SUCCESS_BASE;
            test_score += std::max(0, STEPS_LIMIT - result) * SUCCESS_STEP_BONUS;
            if (result <= STEPS_LIMIT / 2) test_score += 200;
        }
        else if (result == -2) {
            test_score = std::max(0, test_score - TERMINATE_PENALTY);
        }
        else {
            int rover_progress = initial_r2b - min_r2b;
            if (rover_progress > 0)
                test_score += rover_progress * DISTANCE_REDUCTION_BONUS;
            if (bucket_picked) {
                test_score += PICKED_BUCKET_BONUS;
                int bucket_progress = initial_b2p - min_b2p;
                if (bucket_progress > 0)
                    test_score += bucket_progress * BUCKET_TO_PAD_BONUS;
                if (min_b2p <= 2)
                    test_score += 150;
            }
            if (rover_progress <= 0 && !bucket_picked)
                test_score = std::max(BASE_SCORE, test_score - NO_PROGRESS_PENALTY);
        }

        test_score += shaping_score;
        total_score += test_score;

        if (i == 59) {
            double projected = (double)total_score / 60.0;
            if (projected < 200.0) return (int)projected - 1;
        }
    }

    return actual_tests > 0 ? (int)(total_score / actual_tests) : 0;
}


void AntoninaAPI::demonstrate(Perceptron* p) {
    printf("Starting new Antonina runs!\nSTEPS_LIMIT=%d\n", STEPS_LIMIT);
    logfile.open("antlog.txt");
    logfile << "#####\tStarting new Antonina runs!\n#####\tSTEPS_LIMIT=" << STEPS_LIMIT << "\n";
    srand(time(NULL));

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
        int res = GoTestImproved(lab, min_r2b, min_b2p, bucket_picked, PRINT_STEPS, p, shaping);
        if (res > 0) { wins++; sum += res; }
        else if (res == -2) { exit(0); }
    }

    int score = 100 * (wins * STEPS_LIMIT - sum) / STEPS_LIMIT / ALL_TESTS;
    int wr = 100 * wins / ALL_TESTS;
    int as = wins > 0 ? sum / wins : 0;
    printf("Test 00: winrate=%d%% av.steps=%d score=%d\n", wr, as, score);
    logfile << "#####\tTest 00: winrate=" << wr << "% av.steps=" << as << " score=" << score << "\n";
    logfile << "#####\tAll done!\n";
    logfile.close();
}


void AntoninaAPI::writeLab(std::ofstream* fout, int ax, int ay, int Ox, int Oy, int gx, int gy, int rn) {
    *fout << ax << " " << ay << " " << Ox << " " << Oy << " " << gx << " " << gy << " " << rn << '\n';
}

void AntoninaAPI::writeInFile() {
    std::ofstream fout("Test0.csv");
    int n = 0;

    // На одной линии, не у стенок
    for (int ax = 1; ax < 7; ax++) {
        for (int ay = 1; ay < 7; ay++) {
            for (int gy = 1; gy < 7; gy++)
                if (gy != ay) { n++; writeLab(&fout, ax, ay, ax, ay, ax, gy, 0); }
            for (int gx = 1; gx < 7; gx++)
                if (gx != ax) { n++; writeLab(&fout, ax, ay, ax, ay, gx, ay, 0); }
        }
    }
    std::cout << n << '\n';

    // Не на одной линии, не у стенок
    for (int ax = 1; ax < 7; ax++)
        for (int ay = 1; ay < 7; ay++)
            for (int gx = 1; gx < 7; gx++)
                if (gx != ax)
                    for (int gy = 1; gy < 7; gy++)
                        if (gy != ay) { n++; writeLab(&fout, ax, ay, ax, ay, gx, gy, 0); }
    std::cout << n << '\n';

    // У стенок
    for (int ax = 0; ax < 2; ax++) {
        for (int ay = 0; ay < 8; ay++) {
            if (ax == 1) ax = 7;
            for (int gx = 0; gx < 8; gx++)
                for (int gy = 0; gy < 8; gy++)
                    if (!(gy == ay && gx == ax)) { n++; writeLab(&fout, ax, ay, ax, ay, gx, gy, 0); }
        }
    }
    for (int ay = 0; ay < 2; ay++) {
        for (int ax = 1; ax < 7; ax++) {
            if (ay == 1) ay = 7;
            for (int gx = 0; gx < 8; gx++)
                for (int gy = 0; gy < 8; gy++)
                    if (!(gy == ay && gx == ax)) { n++; writeLab(&fout, ax, ay, ax, ay, gx, gy, 0); }
        }
    }
    for (int gx = 0; gx < 2; gx++) {
        for (int gy = 0; gy < 8; gy++) {
            if (gx == 1) gx = 7;
            for (int ax = 1; ax < 7; ax++)
                for (int ay = 1; ay < 7; ay++)
                    if (!(gy == ay && gx == ax)) { n++; writeLab(&fout, ax, ay, ax, ay, gx, gy, 0); }
        }
    }
    for (int gy = 0; gy < 2; gy++) {
        for (int gx = 1; gx < 7; gx++) {
            if (gy == 1) gy = 7;
            for (int ax = 1; ax < 7; ax++)
                for (int ay = 1; ay < 7; ay++)
                    if (!(gy == ay && gx == ax)) { n++; writeLab(&fout, ax, ay, ax, ay, gx, gy, 0); }
        }
    }
    std::cout << n << '\n';
    fout.close();
}
