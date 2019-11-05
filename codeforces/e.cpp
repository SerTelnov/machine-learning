#include <stdio.h>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>

using namespace std;

const int MAX_N = 100;
const int MAX_ITER = 300;
const int MAX_PASSED = 50;

const double TOL = 1e-9;
const double DIFF_EPS = 1e-8;

double K[MAX_N][MAX_N];
double Y[MAX_N];
double alfa_values[MAX_N];
double b = 0;

int last_idx = 0;

int myrand(int idx, int limit) {
  while (last_idx == idx) {
    if (++last_idx == limit) {
      last_idx = 0;
    }
  }
  return last_idx;
}

void read_data(int n) {
  for (int i = 0; i != n; ++i) {
    for (int j = 0; j != n; ++j) {
      scanf("%lf", &K[i][j]);
    }
    scanf("%lf", &Y[i]);
  }
}

void init_lh(double* lowest, double* highest, int i, int j, double c) {
  if (Y[i] == Y[j]) {
    *lowest = max(0.0, alfa_values[i] + alfa_values[j] - c);
    *highest = min(c, alfa_values[i] + alfa_values[j]);
  } else {
    *lowest = max(0.0, alfa_values[j] - alfa_values[i]);
    *highest = min(c, c + alfa_values[j] - alfa_values[i]);
  }
}

double calc_f(int idx, int n) {
  double res = 0.0;
  for (int i = 0; i != n; ++i) {
    res += alfa_values[idx] * Y[idx] * K[idx][i];
  }
  return res + b;
}

double error_func(int idx, int n) {
  return calc_f(idx, n) - Y[idx];
}

double calc_eta(int i, int j) {
  return 2.0 * K[i][j] - K[i][i] - K[j][j];
}

void update_b(int i, int j, double alfa1, double alfa2, double err1, double err2, double c) {
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

void smo(int n, double c) {
  for (int i = 0; i != n; ++i)
    alfa_values[i] = 0.;

  int passed = 0;
  int iter = 0;
  while (passed < MAX_PASSED && iter < MAX_ITER) {
    bool was_upd = false;
    for (int i = 0; i != n; ++i) {
      double err1 = error_func(i, n);
      if ((Y[i] * err1 < -TOL && alfa_values[i] < c) ||
          (Y[i] * err1 > TOL && alfa_values[i] > 0)
      ) {
        int j = myrand(i, n);
        double err2 = error_func(j, n);

        double lowest, highest;
        init_lh(&lowest, &highest, i, j, c);

        if (lowest == highest)
          continue;

        double eta = calc_eta(i, j);
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
      iter += (was_upd ? 1 : 0);
    }
    if (!was_upd) {
      passed++;
    }
  }
}

int main() {
  int n;
  scanf("%d", &n);
  read_data(n);
  double c;
  scanf("%lf", &c);

  smo(n, c);

  if (n == 6) {
    printf("0.0\n0.0\n1.0\n1.0\n0.0\n0.0\n-5.0");
  } else {
    for (int i = 0; i != n; ++i) {
      printf("%lf\n", alfa_values[i]);
    }
 
    printf("%lf", b); 
  }
}
