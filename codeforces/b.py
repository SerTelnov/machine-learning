def calc_f(pr, re):
    if (pr + re == 0):
        return 0
    return 2 * (pr * re) / (pr + re)

k = int(input())

tp = [0] * k
fp = [0] * k
fn = [0] * k
weight = [0] * k

sum = 0
for i in range(k):
    cur_row = (float(val) for val in input().split())
    for idx, val in enumerate(cur_row):
        sum += val
        if (idx != i):
            fn[idx] += val
            fp[i] += val
        else:
            tp[i] += val

if sum != 0:
    for i in range(k):
        weight[i] = (tp[i] + fp[i]) / sum

pr = [0] * k
re = [0] * k

for i in range(k):
    if tp[i] + fp[i] != 0:
        pr[i] = tp[i] / (tp[i] + fp[i])
    if tp[i] + fn[i] != 0:
        re[i] = tp[i] / (tp[i] + fn[i])

pr_macro = 0.0
re_macro = 0.0

for i in range(k):
    pr_macro += weight[i] * pr[i]
    re_macro += weight[i] * re[i]

print(calc_f(pr_macro, re_macro))

f_micro = 0

for i in range(k):
    f_micro += weight[i] * calc_f(pr[i], re[i])

print (f_micro)
