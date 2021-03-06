import math
from data import Mail

class MailInfo():
  def __init__(self, probability, words_mail_count, mail_count):
    self.probability = probability
    self.words_mail_count = words_mail_count
    self.mail_count = mail_count

def _safe_get(dic, key):
  if not key in dic:
    return 0
  return dic[key]

def _safe_log(x):
  return math.log(x) if x != 0 else 0

def _safe_exp(x):
  return math.exp(x)
  # return math.exp(x) if abs(x) > (1e-2) else 1

def _inc(dic, key1, key2):
  if not key1 in dic:
    dic[key1] = {}
  if not key2 in dic[key1]:
    dic[key1][key2] = 0
  dic[key1][key2] += 1

def train(mails, clazz_fine):
  clazz_mails = [0] * 2
  words_mails_counts = {}
  words = set()

  for mail in mails:
    clazz = mail.is_spam
    clazz_mails[clazz] += 1

    for word in mail.message:
      _inc(words_mails_counts, clazz, word)
      words.add(word)

  preparetion = []
  for clazz in range(2):
    if not clazz in words_mails_counts:
      words_mails_counts[clazz] = {}

    x = clazz_mails[clazz] / len(mails) * clazz_fine[clazz]
    preparetion.append(MailInfo(math.log(x), words_mails_counts[clazz], clazz_mails[clazz]))
  return preparetion

def classifier(preparetion, alfa, mail):
  scores = []
  for clazz in range(len(preparetion)):
    score = 0
    info = preparetion[clazz]

    for word in mail.message:
      x = alfa + _safe_get(info.words_mail_count, word)
      y = info.mail_count + alfa * len(info.words_mail_count)
      if x != 0 and y != 0:
        score += math.log(x) - math.log(y)

    score += info.probability
    scores.append(score)

  return scores.index(max(scores))
