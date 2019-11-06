#include <stdio.h>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>

using namespace std;

const int MAX_N = 100;
const int MAX_PASSED = 5;

const double TOL = 1e-9;
const double DIFF_EPS = 1e-8;

double K[MAX_N][MAX_N];
double Y[MAX_N];
double alfa_values[MAX_N];
double b = 0.0;

int last_idx = 0;

int myrand(int idx, int limit) {
  while (true) {
    if (++last_idx == limit) {
      last_idx = 0;
    }
    if (last_idx != idx) {
      return last_idx;
    }
  }
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
    res += alfa_values[i] * Y[i] * K[idx][i];
  }
  return res + b;
}

double error_func(int idx, int n) {
  return calc_f(idx, n) - Y[idx];
}

double calc_eta(int i, int j) {
  return 2.0 * K[i][j] - K[i][i] - K[j][j];
}

void update_b(int i, int j, double old_alfai, double old_alfaj, double erri, double errj, double c) {
  double deltai = alfa_values[i] - old_alfai;
  double deltaj = alfa_values[j] - old_alfaj;
  double b1 = b - erri - Y[i] * deltai * K[i][i]
                       - Y[j] * deltaj * K[i][j];
  double b2 = b - errj - Y[i] * deltai * K[i][j]
                       - Y[j] * deltaj * K[j][j];

  if (0 < alfa_values[i] && alfa_values[i] < c) {
    b = b1;
  } else if (0 < alfa_values[j] && alfa_values[j] < c) {
    b = b2;
  } else {
    b = (b1 + b2) / 2.0;
  }
}

void smo(int n, double c) {
  for (int i = 0; i != n; ++i)
    alfa_values[i] = 0.;

  int passed = 0;
  while (passed < MAX_PASSED) {
    bool was_upd = false;
    for (int i = 0; i != n; ++i) {
      double erri = error_func(i, n);
      if ((Y[i] * erri < -TOL && alfa_values[i] < c) ||
          (Y[i] * erri > TOL && alfa_values[i] > 0)
      ) {
        int j = myrand(i, n);
        double errj = error_func(j, n);
        double old_alfai = alfa_values[i];
        double old_alfaj = alfa_values[j];

        double lowest, highest;
        init_lh(&lowest, &highest, i, j, c);

        if (lowest == highest)
          continue;

        double eta = calc_eta(i, j);
        if (eta >= 0)
          continue;

        alfa_values[j] -= Y[j] * (erri - errj) / eta;
        if (alfa_values[j] > highest)
          alfa_values[j] = highest;
        else if (alfa_values[j] < lowest)
          alfa_values[j] = lowest;

        if (abs(alfa_values[j] - old_alfaj) < DIFF_EPS)
          continue;

        alfa_values[i] += Y[i] * Y[j] * (old_alfaj - alfa_values[j]);

        update_b(i, j, old_alfai, old_alfaj, erri, errj, c);
        was_upd = true;
      }
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

  for (int i = 0; i != n; ++i)
    printf("%lf\n", alfa_values[i]);

  printf("%lf", b);
  return 0;
}
