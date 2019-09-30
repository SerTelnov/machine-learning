n, m, k = (int(i) for i in input().split())
limit = int(n / k)

obj_last_part = {}
parts = {}
lens = [0] * k
active_parts = set(i for i in range(k))

next_part = 0
for idx, obj in enumerate(input().split()):
    cur_part = next_part

    if obj in obj_last_part:
        cur_part = (obj_last_part[obj] + 1) % k
        if cur_part in parts and lens[cur_part] > limit:
            cur_part = next(iter(active_parts))

    if not cur_part in parts:
        parts[cur_part] = []

    parts[cur_part].append(idx + 1)
    lens[cur_part] += 1
    if (lens[cur_part] > limit):
        active_parts.remove(cur_part)

    obj_last_part[obj] = cur_part
    next_part = (next_part + 1) % k

for key, cur_part in parts.items():
    idx = int(key)
    print(lens[idx], *cur_part)
