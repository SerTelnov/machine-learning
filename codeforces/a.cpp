#include <stdio.h>
#include <vector>
#include <unordered_map>

using namespace std;

int main() {
    int n, m, k;
    scanf("%d%d%d", &n, &m, &k);

    unordered_map<int, vector<int>> obj_counts;
    unordered_map<int, vector<int>> parts;

    for (int i = 0; i != n; ++i) {
        int obj;
        scanf("%d", &obj);

        if (obj_counts.find(obj) == obj_counts.end()) {
            vector<int> new_vector;
            obj_counts[obj] = new_vector;
        }

        obj_counts[obj].push_back(i + 1);
    }

    int curr_part = 0;
    for (int i = 1; i <= m; ++i) {
        vector<int> curr_objs = obj_counts[i];
        for (int curr_obj : curr_objs) {
            if (parts.find(curr_part) == parts.end()) {
                vector<int> new_vector;
                parts[curr_part] = new_vector;
            }

            parts[curr_part].push_back(curr_obj);
            curr_part = (curr_part + 1) % k;
        }
    }

    for (auto& it : parts) {
        vector<int> curr_part = it.second;
        printf("%d ", curr_part.size());
        for (int i : curr_part) {
            printf("%d ", i);
        }
        printf("\n");
    }

    return 0;
}
