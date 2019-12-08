#include <stdio.h>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <memory>


using namespace std;

const int MAX_N = 4000;
const int MAX_M = 100;

int attribute_number;
int class_number;
int max_depth;
int n;

int Y[MAX_N];
int X[MAX_N][MAX_M];

int node_counter = 0;

class Question {
  public:
    int attribute_idx;
    double value;

    Question() {}

    Question(
      int idx,
      double x
    ) : attribute_idx(idx)
      , value(x)
      {}
};

class Node {
  public:
    Node* left;
    Node* right;
    Question question;
    int class_result;
    int idx;
    bool is_leaf;

    Node(
      int _class
    ) : class_result(_class)
      , idx(0)
      , is_leaf(true)
      {}

    Node(
      Node* _left,
      Node* _right,
      Question q
    ) : left(_left)
      , right(_right)
      , question(q)
      , class_result(-1)
      , idx(0)
      , is_leaf(false)
      {}
};

int curr_comp_attribute = -1;
bool X_ids_comp(int idx1, int idx2) {
  return X[idx1][curr_comp_attribute] < X[idx2][curr_comp_attribute];
}

void read_input() {
  scanf("%d%d%d%d", &attribute_number, &class_number, &max_depth, &n);

  for (int i = 0; i != n; ++i) {
    for (int j = 0; j != attribute_number; ++j) {
      scanf("%d", &X[i][j]);
    }
    scanf("%d", &Y[i]);
  }
}

bool accept(int xId, Question & q) {
  return X[xId][q.attribute_idx] < q.value;
}

void split(vector<int> & xIds, Question & q, vector<int> & left, vector<int> & right) {
  for (int xId : xIds) {
    if (accept(xId, q))
      left.push_back(xId);
    else 
      right.push_back(xId);
  }
}

Question make_split(vector<int> & xIds, vector<int> & left, vector<int> & right) {
  double gini_value_winner = -1000;
  double value_winner = -2e9;
  int attr_idx_winner = -1;

  for (int attr_idx = 0; attr_idx != attribute_number; ++attr_idx) {
    curr_comp_attribute = attr_idx;
    sort(xIds.begin(), xIds.end(), X_ids_comp);

    vector<int> left_class_counters(class_number, 0), right_class_counters(class_number, 0);
    for (int id : xIds)
      ++right_class_counters[Y[id] - 1];

    double left_class_sum = 0.0;
    double right_class_sum = 0.0;

    for (int right_class : right_class_counters) {
      double x = double(right_class);
      right_class_sum += x * x;
    }

    for (size_t i = 0; i != xIds.size(); ++i) {
      int curr_idx = xIds[i];

      double x = double(right_class_counters[Y[curr_idx] - 1]);
      right_class_sum -= x * x;
      --right_class_counters[Y[curr_idx] - 1];

      x = double(right_class_counters[Y[curr_idx] - 1]);
      right_class_sum += x * x;

      x = double(left_class_counters[Y[curr_idx] - 1]);
      left_class_sum -= x * x;

      ++left_class_counters[Y[curr_idx] - 1];
      x = double(left_class_counters[Y[curr_idx] - 1]);
      left_class_sum += x * x;

      double left_item_count = i + 1;
      double right_item_count = xIds.size() - left_item_count;

      double curr_gini = left_item_count ? left_class_sum / left_item_count : 0;
      if (right_item_count)
        curr_gini += right_class_sum / right_item_count;

      if (gini_value_winner < curr_gini) {
        gini_value_winner = curr_gini;
        attr_idx_winner = attr_idx;
        value_winner = right_item_count ? double(X[curr_idx][attr_idx] + X[xIds[i + 1]][attr_idx]) / 2 : 2e9;
      }
    }
  }

  Question q = Question(attr_idx_winner, value_winner);
  split(xIds, q, left, right);
  return q;
}

Node* to_terminal(vector<int> & left, vector<int> & right) {
  vector<int> classes(class_number, 0);
  int max_class = 0;
  int max_class_id = -1;

  for (int id : left) {
    classes[Y[id] - 1]++;
    if (max_class < classes[Y[id] - 1]) {
      max_class = classes[Y[id] - 1];
      max_class_id = Y[id];
    }
  }

  for (int id : right) {
    classes[Y[id] - 1]++;

    if (max_class < classes[Y[id] - 1]) {
      max_class = classes[Y[id] - 1];
      max_class_id = Y[id];
    }
  }

  return new Node(max_class_id);
}

Node* to_terminal(vector<int> & ids) {
  auto vv = vector<int>();
  return to_terminal(ids, vv);
}

bool is_one_class(vector<int> & ids) {
  unordered_set<int> classes;
  for (auto id : ids) {
    classes.insert(Y[id]);
  }
  return classes.size() == 1;
}

Node* build_tree(vector<int> & curr_entities, int depth) {
  if (is_one_class(curr_entities)) {
    return to_terminal(curr_entities);
  }

  vector<int> left, right;
  Question q = make_split(curr_entities, left, right);

  if (left.empty() || right.empty()) {
    return to_terminal(curr_entities);
  } else if (depth >= max_depth) {
    return to_terminal(left, right);
  } else {
    Node* left_node = build_tree(left, depth + 1);
    Node* right_node = build_tree(right, depth + 1);

    return new Node(left_node, right_node, q);
  }
}

Node* build_tree() {
  vector<int> ids;
  for (int i = 0; i != n; ++i) {
    ids.push_back(i);
  }

  return build_tree(ids, 0);
}

void print_tree(Node* tree) {
  if (!tree)
    return;

  if (tree->is_leaf) {
    printf("C %d\n", tree->class_result);
  } else {
    printf("Q %d %lf ", tree->question.attribute_idx + 1, tree->question.value);
    printf("%d %d\n", tree->left->idx, tree->right->idx);

    print_tree(tree->left);
    print_tree(tree->right);
  }
}

void dfs(Node * tree) {
  if (!tree)
    return;

  tree->idx = ++node_counter;
  if (!tree->is_leaf) {
    dfs(tree->left);
    dfs(tree->right);
  }
}

int main() {
  read_input();
  Node* tree = build_tree();

  dfs(tree);
  printf("%d\n", node_counter);
  print_tree(tree);
  return 0;
}
