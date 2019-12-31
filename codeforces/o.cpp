#include <unordered_map>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

using namespace std;
typedef long long ll;

int main() {
  int k, n;
  scanf("%d\n%d", &k, &n);

  vector<vector<int>> mapper(k + 1);
  vector<int> values;

  int in_class = 0;
  int all_class = 0;
  for (int i = 0; i != n; ++i) {
    int curr_class, value;
    scanf("%d%d", &value, &curr_class);

    values.push_back(value);
    mapper[curr_class].push_back(value);
    printf("%d\n", mapper[curr_class].size());
  }

  printf("Here!");
  sort(values.begin(), values.end());
  for (int i = 0; i != n; ++i) {
    sort(mapper[i].begin(), mapper[i].end());
  }

  for (int i = 1; i != n; ++i) {
    all_class += ((ll) 1) * i * (n - i) * (values[i] - values[i - 1]);
  }  

  for (int kk = 0; kk != k; ++k) {
    for (int i = 0; i != mapper[kk].size(); ++i) {
      in_class += ((ll) 1) * i * (n - i) * (mapper[kk][i] - mapper[kk][i - 1]);
    }
  }

  printf("%d\n%d", in_class, all_class - in_class);
  return 0;
}