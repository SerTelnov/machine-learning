#include <stdio.h>
#include <vector>
#include <algorithm>
#include <ctime>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>


using namespace std;

const int MAX_N = 100;
const int MAX_PASSES = 10;
const double TOL = 0.01;
const double DIFF_EPS = 0.0001;

int K[MAX_N][MAX_N];
int Y[MAX_N];
double alfa_values[MAX_N];
double b = 0;

int myrand(int idx, int limit) {
  int r_idx = (rand() % limit) + 1;
  if (r_idx == idx) {
    return myrand(idx, limit);
  }
  return r_idx;
}

void read_data(int n) {
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j != n; j++) {
      scanf("%d", &K[i][j]);
    }
    scanf("%d", &Y[i]);
  }
}

void init_lh(double* lowest, double* highest, int i, int j, int c) {
  if (Y[i] == Y[j]) {
    *lowest = max(0.0, alfa_values[i] + alfa_values[j] - c);
    *highest = min((double) c, alfa_values[i] + alfa_values[j]);
  } else {
    *lowest = max(0.0, alfa_values[j] - alfa_values[i]);
    *highest = min((double) c, c + alfa_values[j] - alfa_values[i]);
  }
}

double error_func(int idx) {
  return (alfa_values[idx] * Y[idx] * K[idx][idx] + b) - Y[idx];
} 

int calc_eta(int i, int j) {
  return 2 * K[i][j] - K[i][i] - K[j][j];
}

void update_b(int i, int j, double alfa1, double alfa2, double err1, double err2, int c) {
  double b1 = b - err1 - Y[i] * (alfa1 - alfa_values[i]) * K[i][i]
                       - Y[j] * (alfa2 - alfa_values[j]) * K[i][j];
  double b2 = b - err2 - Y[i] * (alfa1 - alfa_values[i]) * K[i][j]
                       - Y[j] * (alfa2 - alfa_values[j]) * K[j][j];

  if (0 < alfa1 && alfa1 < c) {
    b = b1;
  } else if (0 < alfa2 && alfa2 < c) {
    b = b2;
  } else {
    b = (b1 + b2) / 2;
  }
}

void smo(int n, int c) {
  for (int i = 0; i != n; ++i)
    alfa_values[i] = 0;

  int passes = 0;
  while (passes < MAX_PASSES) {
    bool was_upd = false;
    for (int i = 0; i != n; ++i) {
      double err1 = error_func(i);
      if ((Y[i] * err1 < -TOL && alfa_values[i] < c) ||
          (Y[i] * err1 > TOL && alfa_values[i] > 0)
      ) {
        int j = myrand(i, n);
        double err2 = error_func(j);

        double lowest, highest;
        init_lh(&lowest, &highest, i, j, c);

        if (lowest == highest)
          continue;

        int eta = calc_eta(i, j);
        if (eta >= 0)
          continue;

        double alfa2 = alfa_values[j] - Y[j] * (err1 - err2) / eta;
        if (alfa2 > highest)
          alfa2 = highest;
        else if (alfa2 < lowest)
          alfa2 = lowest;

        if (abs(alfa_values[j] - alfa2) < DIFF_EPS)
          continue;

        double alfa1 = alfa_values[i] + Y[i] * Y[j] * (alfa_values[j] - alfa2);

        update_b(i, j, alfa1, alfa2, err1, err2, c);
        alfa_values[i] = alfa1;
        alfa_values[j] = alfa2;
        was_upd = true;
      }
    }

    if (!was_upd) {
      passes++;
    } else {
      passes = 0;
    }
  }
}

int main() {
  srand(time(0));

  int n, c;
  scanf("%d", &n);
  read_data(n);
  scanf("%d", &c);

  smo(n, c);
  for (int i = 0; i != n; ++i) {
    printf("%lf\n", alfa_values[i]);
  }
  printf("%lf", b);
}
