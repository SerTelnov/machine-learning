import os
import sys

# RESOURCES_PATH = sys.argv[1] + '/labs/bayes/resources'
RESOURCES_PATH = 'resources'

class Mail():
  def __init__(self, message, is_spam):
    self.message = message
    self.is_spam = is_spam

def concate_except_idx(batches, excep_idx):
  data = []
  for i in range(len(batches)):
    if i != excep_idx:
      data += batches[i]
  return data

def read_data():
  path = RESOURCES_PATH
  mails = []

  for root, _, files in os.walk(path):
    for file in files:
      if file.endswith('.txt'):
        filepath = os.path.join(root, file)
        mails.append(_parseMail(filepath, file))
  return mails

def _parse_part_idx(filepath):
  idx = filepath.index('part') + 4
  if filepath[idx:idx + 2].isdigit():
    return int(filepath[idx:idx + 2]) - 1
  return int(filepath[idx]) - 1

def read_batch_data():
  path = RESOURCES_PATH
  batches = [[] for _ in range(10)]

  for root, _, files in os.walk(path):
    for file in files:
      if file.endswith('.txt'):
        filepath = os.path.join(root, file)
        batches[_parse_part_idx(filepath)].append(_parseMail(filepath, file))
  return batches

def _parseMail(filepath, filename):
  with open(filepath, 'r') as f:
    for line in f.readlines():
      if line != '\n':
        words = line.replace('\n', '').split(' ')
        if line.startswith('Subject'):
          message = set(words[1:])
        else:
          message |= set(words)
  is_spam = 0 if 'legit' in filename else 1
  return Mail(message, is_spam)

