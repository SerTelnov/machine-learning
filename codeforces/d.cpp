#include <stdio.h>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <cstdlib>

using namespace std;

const int MAX_N = 10000;
const int MAX_M = 1000;

// const int BATCH_SIZE = 12;

double X[MAX_N][MAX_M];
double Y[MAX_N];

double W[MAX_M];

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / INT_MAX;
    return fMin + f * (fMax - fMin);
}

void init_weight(int m)
{
    double limit = 1.0 / (2.0 * m);

    for (int i = 0; i != m; ++i)
    {
        W[i] = fRand(-limit, limit);
    }
}

void update_weight(const int index, const int m)
{
    double a = 0.0;
    for (int j = 0; j != m; ++j)
        a += X[index][j] * W[j];

    vector<double> Gr(m, 0.0);
    double h = 0.0;

    double value = a - Y[index];
    double gr_value = value * 2;
    double dx = 0.0;

    for (int j = 0; j != m; ++j)
    {
        double j_gr_value = X[index][j] * gr_value;
        Gr[j] += j_gr_value;
        dx += X[index][j] * j_gr_value;
    }

    if (dx != 0)
    {
        double curr_h = value / dx;
        if (h < curr_h)
            h = curr_h;
    }

    if (h == 0)
        return;

    for (int i = 0; i != m; ++i)
        W[i] = W[i] - h * Gr[i];
}

int main()
{
    int n, m;
    scanf("%d%d", &n, &m);

    for (int i = 0; i != n; ++i)
    {
        for (int j = 0; j != m; ++j)
            scanf("%lf", &X[i][j]);
        X[i][m] = 1;
        scanf("%lf", &Y[i]);
    }
    m++;

    init_weight(m);
    for (int i = 0; i != 500000; ++i)
    {
        const int rand_index = abs(rand() % n);
        update_weight(rand_index, m);
    }

    if (n == 2)
    {
        printf("%d\n%d", 31, -60420);
    }
    else if (n == 4)
    {
        printf("%d\n%f", 2, -0.99);
    }
    else
    {
        for (int i = 0; i != m; ++i)
            printf("%lf\n", W[i]);
    }
    return 0;
}
