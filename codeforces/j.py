import math

def do_safe_sum(arr, idx, b):
  if idx >= len(arr) or arr[idx] == None:
    return b
  a = arr[idx][:]
  for i in range(len(a)):
    for j in range(len(a[i])):
      a[i][j] += b[i][j]
  return a

def do_sum(net, indeces):
  current = net[indeces[0]][:]
  for i in indeces[1:]:
    current = do_safe_sum(net, i, current)
  return current

def do_mul(a, b):
  c = [[0] * len(b[0]) for _ in range(len(a))]
  for i in range(len(a)):
    for j in range(len(b)):
      for k in range(len(b[j])):
        c[i][k] += a[i][j] * b[j][k]
  return c

def do_tnh(x):
  c = []
  for row in x:
    c.append(list(map(math.tanh, row)))
  return c

def do_d_tnh(x):
  c = []
  for row in x:
    c.append(list(map(lambda val : 1 - math.tanh(val) ** 2, row)))
  return c

def do_rlu(alpha, x):
  y = []
  for row in x:
    y.append(list(map(lambda x : x if x >= 0 else alpha * x, row)))
  return y

def do_d_rlu(alpha, x, dc):
  dx = []
  for row in x:
    dx.append(list(map(lambda val : 1 if val >= 0 else alpha, row))) 
  return do2had(dx, dc)

def do2had(a, b):
  for i in range(len(a)):
    for j in range(len(a[i])):
      a[i][j] *= b[i][j]
  return a

def do_had(net, indeces):
  current = net[indeces[0]][:]
  for next_index in indeces[1:]:
    current = do2had(current, net[next_index])
  return current

def do_d_had(dc, net, indeces, except_idx):
  current = None
  for i in indeces:
    if i != except_idx:
      current = current if current != None else net[i]
      current = do2had(current, net[i])
  return do2had(current, dc)

def d_mul(dc, a, b):
  da = [[0] * len(b) for _ in range(len(dc))]
  db = [[0] * len(dc[0]) for _ in range(len(a[0]))]

  for i in range(len(dc)):
    for j in range(len(b)):
      for k in range(len(b[0])):
        da[i][j] += dc[i][k] * b[j][k]
        db[j][k] += a[i][j] * dc[i][k]

  return da, db

def move_back(net, graph, d_input, m, dc, curr_idx):
  if 0 <= curr_idx < m:
    d_input[curr_idx] = do_safe_sum(d_input, curr_idx, dc)
  else:
    op = graph[curr_idx]['op']
    if op == "sum":
      indeces = graph[curr_idx]['indeces']
      for i in indeces:
        move_back(net, graph, d_input, m, dc, i)
    elif op == "mul":
      a = net[graph[curr_idx]['first']]
      b = net[graph[curr_idx]['second']]
      da, db = d_mul(dc, a, b)

      move_back(net, graph, d_input, m, da, graph[curr_idx]['first'])
      move_back(net, graph, d_input, m, db, graph[curr_idx]['second'])
    elif op == "tnh":
      x = net[graph[curr_idx]['x']]
      dx = do2had(do_d_tnh(x), dc)
      move_back(net, graph, d_input, m, dx, graph[curr_idx]['x'])
    elif op == "rlu":
      i = graph[curr_idx]['x']
      dx = do_d_rlu(graph[curr_idx]['alpha'], net[i], dc)
      move_back(net, graph, d_input, m, dx, i)
    elif op == "had":
      indeces = graph[curr_idx]['indeces']
      for i in indeces:
        dx = do_d_had(dc, net, indeces, i)
        move_back(net, graph, d_input, m, dx, i)

n, m, k = (int(i) for i in input().split())

input_matrix_params = []
for _ in range(m):
  rows, columns = (int(i) for i in input().split()[1:])
  input_matrix_params.append((rows, columns))

ops = []
for _ in range(n - m):
  ops.append(input())

net = []
for rows, _ in input_matrix_params:
  matrix = [list(int(i) for i in input().split()) for _ in range(rows)]
  net.append(matrix)

graph = {}

for op in ops:
  splited_op = op.split()
  op_name = splited_op[0]

  next = None
  indeces = None
  id = len(net)

  if op_name == "sum":
    indeces = list(int(x) - 1 for x in splited_op[2:])
    next = do_sum(net, indeces)
    graph[id] = { "op": "sum", "indeces": indeces }
  elif op_name == "mul":
    i, j = (int(x) - 1 for x in splited_op[1:])
    next = do_mul(net[i], net[j])
    graph[id] = { "op": "mul", "first": i, "second": j }
  elif op_name == "tnh":
    x = int(splited_op[1]) - 1
    next = do_tnh(net[x])
    graph[id] = { "op": "tnh", "x": x }
  elif op_name == "rlu":
    alpha, i = (int(x) for x in splited_op[1:])
    alpha = 1 / alpha
    next = do_rlu(alpha, net[i - 1])
    graph[id] = { "op": "rlu", "x": i - 1, "alpha": alpha }
  elif op_name == "had":
    indeces = list(int(x) - 1 for x in splited_op[2:])
    next = do_had(net, indeces)
    graph[id] = { "op": "had", "indeces": indeces }

  net.append(next)

for x in net[-k:]:
  for row in x:
    print(' '.join(str(val) for val in row))

d_input = [None for _ in range(m)]

for idx, x in enumerate(net[-k:]):
  index = n - k + idx
  dc = []
  for _ in range(len(x)):
    dc.append(list(int(x) for x in input().split()))
  move_back(net, graph, d_input, m, dc, index)

for x in d_input:
  if x != None:
    for row in x:
      print(' '.join(str(val) for val in row))
