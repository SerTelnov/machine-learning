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

unordered_map<int, pair<int, int>> tree_idx_map;

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
    bool is_leaf;

    Node(
      int _class
    ) : class_result(_class)
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
      , is_leaf(false)
      {}
};

class Groups {
  public:
    vector<int> left;
    vector<int> right;

    Groups() {}

    Groups(
      vector<int> l,
      vector<int> r
    ) : left(l)
      , right(r)
      {}
};

void read_input() {
  scanf("%d%d%d%d", &attribute_number, &class_number, &max_depth, &n);

  for (int i = 0; i != n; ++i) {
    for (int j = 0; j != attribute_number; ++j) {
      scanf("%d", &X[i][j]);
    }
    scanf("%d", &Y[i]);
    --Y[i];
  }
}

unordered_set<int> collect_class(vector<int> xIds) {
  unordered_set<int> class_ids;
  for (int i : xIds) {
    class_ids.insert(Y[i]);
  }
  return class_ids;
}

bool accept(int xId, Question q) {
  return X[xId][q.attribute_idx] < q.value;
}

Groups split(vector<int> xIds, Question q) {
  vector<int> left;
  vector<int> right;

  for (int xId : xIds) {
    if (accept(xId, q))
      left.push_back(xId);
    else 
      right.push_back(xId);
  }

  return Groups(left, right);
}

double calc_gini(vector<int> ids, unordered_set<int> class_ids, int entities_number) {
  if (ids.empty()) {
    return 0.0;
  }

  unordered_map<int, int> class_mapper;
  for (int id : ids) {
    if (class_mapper.find(Y[id]) == class_mapper.end())
      class_mapper[id] = 0;
    ++class_mapper[id];
  }

  double score = 0.0;
  for (int class_id : class_ids) {
    double p = class_mapper[class_id] / ids.size();
    score += p * p;
  }

  return (1.0 - score) * (ids.size() / entities_number);
}

double gini_index(Groups groups, unordered_set<int> class_ids) {
  int entities_number = groups.left.size() + groups.right.size();
  return calc_gini(groups.left, class_ids, entities_number) + 
         calc_gini(groups.right, class_ids, entities_number);
}

pair<Groups, Question> make_split(vector<int> xIds) {
  unordered_set<int> class_ids = collect_class(xIds);

  Groups groups;
  Question q;
  double gini_value = 1000;

  for (int attr_idx = 0; attr_idx != attribute_number; ++attr_idx) {
    for (int xId : xIds) {
      Question curr_q = Question(attr_idx, X[xId][attr_idx]);
      Groups curr_groups = split(xIds, curr_q);

      double curr_gini = gini_index(curr_groups, class_ids);
      if (curr_gini < gini_value) {
        gini_value = curr_gini;
        q = curr_q;
        groups = curr_groups;
      }
    }
  }

  return { groups, q };
}

Node to_terminal(vector<int> ids) {
  unordered_map<int, int> class_mapper;
  int max_class = 0;
  int max_class_id = -1;

  for (int id : ids) {
    if (class_mapper.find(Y[id]) == class_mapper.end())
      class_mapper[id] = 0;
    class_mapper[id]++;
    
    if (max_class < class_mapper[id]) {
      max_class = class_mapper[id];
      max_class_id = id;
    }
  }

  return Node(max_class_id);
}

Node to_terminal(Groups groups) {
  unordered_map<int, int> class_mapper;
  int max_class = 0;
  int max_class_id = -1;

  for (int id : groups.left) {
    if (class_mapper.find(Y[id]) == class_mapper.end())
      class_mapper[id] = 0;
    class_mapper[id]++;
    
  }

  for (int id : groups.right) {
    if (class_mapper.find(Y[id]) == class_mapper.end())
      class_mapper[id] = 0;
    class_mapper[id]++;

    if (max_class < class_mapper[id]) {
      max_class = class_mapper[id];
      max_class_id = id;
    } 
  }

  return Node(max_class_id);
}

Node build_tree(vector<int> curr_entities, int depth, int& id) {
  auto split = make_split(curr_entities);
  Groups groups = split.first;
  if (groups.left.empty() || groups.right.empty()) {
    return to_terminal(curr_entities);    
  } else if (depth >= max_depth) {
    return to_terminal(groups);
  }

  ++id;
  int curr_id = id;
  Node left = build_tree(groups.left, depth + 1, id);
  int left_id = id;
  Node right = build_tree(groups.right, depth + 1, id);
  int right_id = id;
  tree_idx_map.insert({curr_id, {left_id, right_id}});

  return Node(&left, &right, split.second);
}

Node build_tree() {
  vector<int> ids(n);
  for (int i = 0; i != n; ++i) {
    ids.push_back(i);
  }

  int tmp = 0;
  return build_tree(ids, 0, tmp);
}

void print_tree(Node* tree, int& id) {
  ++id;

  if (tree->is_leaf) {
    printf("C %d\n", tree->class_result);
  } else {
    printf("Q %d %lf", tree->question.attribute_idx, tree->question.value);
    auto p = tree_idx_map[id];
    printf("%d %d\n", p.first, p.second);    

    print_tree(tree->left, id);
    print_tree(tree->right, id);
  }
}

void print_tree(Node tree) {
  printf("%d\n", tree_idx_map.size());
  int id = 0;
  print_tree(&tree, id);
}

int main() {
  read_input();
  Node tree = build_tree();
  print_tree(tree);
  return 0;
}
