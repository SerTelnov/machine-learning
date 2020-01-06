import math

class Message():
    def __init__(self, clazz, words):
        self.clazz = clazz
        self.words = set(words)

class ClazzInfo():
    def __init__(self, probability, words_msg_count, msg_count):
        self.probability = probability
        self.words_msg_count = words_msg_count
        self.msg_count = msg_count

def _safe_parse_msg(msg, idx):
    if idx < len(msg):
        return msg[idx:]
    return []

def _safe_get(cur_dict, key):
    if not key in cur_dict:
        return 0
    return cur_dict[key]

def _safe_log(x):
    return math.log(x) if x != 0 else 0

def _safe_exp(x):
    return math.exp(x)
    # return math.exp(x) if abs(x) > (1e-2) else 1

def _safe_div(x, d):
    if x == 0:
        return 0
    return _safe_exp(_safe_log(x) - _safe_log(d))

def read_data():
    input()
    class_fine = [int(str) for str in input().split()]
    alfa = int(input())

    train_msgs = []
    for _ in range(int(input())):
        curr_msg = input().split(" ")
        train_msgs.append(Message(int(curr_msg[0]) - 1, _safe_parse_msg(curr_msg, 2)))

    return class_fine, alfa, train_msgs

def _inc(cur_dict, key1, key2):
    if not key1 in cur_dict:
        cur_dict[key1] = {}
    if not key2 in cur_dict[key1]:
        cur_dict[key1][key2] = 0
    cur_dict[key1][key2] += 1

def train(msgs, clazz_fine):
    clazz_msgs = [0] * len(clazz_fine)
    words_msg_counts = {}
    words = set()

    for msg in msgs:
        clazz = msg.clazz
        clazz_msgs[clazz] += 1

        if not msg.words:
            words_msg_counts[clazz] = []

        for word in set(msg.words):
            _inc(words_msg_counts, clazz, word)
            words.add(word)

    preparetion = {}
    for clazz in range(len(clazz_fine)):
        if not clazz in words_msg_counts:
            words_msg_counts[clazz] = {}

        x = clazz_msgs[clazz] / len(msgs) * clazz_fine[clazz]
        if x != 0:
            preparetion[clazz] = ClazzInfo(math.log(x), words_msg_counts[clazz], clazz_msgs[clazz])

    return preparetion

def classifier(preparetion, n, alfa, words):
    scores = []

    for clazz in range(n):
        score = 0
        if clazz in preparetion:
            info = preparetion[clazz]
            for word in set(words):
                x = alfa + _safe_get(info.words_msg_count, word)
                y = info.msg_count + alfa * len(info.words_msg_count)
                if x != 0 and y != 0:
                    score += math.log(x) - math.log(y)
            score += info.probability
            score = _safe_exp(score)
        scores.append(score)

    scores_sum = sum(scores)
    print(' '.join(map(str, map(lambda x:  _safe_div(x, scores_sum), scores))))

clazz_fine, alfa, train_msgs = read_data()
preparetion = train(train_msgs, clazz_fine)

for _ in range(int(input())):
    curr_msg = input().split(" ")
    classifier(preparetion, len(clazz_fine), alfa, _safe_parse_msg(curr_msg, 1))
