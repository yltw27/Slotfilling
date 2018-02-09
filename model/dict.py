import pandas as pd
import csv

df = pd.read_table('../data/ATIS/atis.train.w-intent.iob', names=['utterance', 'label'])

label = df['label'].tolist()
label_set, intent_set = set(), set()
for i in range(len(label)):
    labels = label[i].split(' ')[:-1]
    intent = label[i].split(' ')[-1]
    intent_set.add(intent)
    for x in range(len(labels)):
        label_set.add(labels[x])

intent_list, label_list = list(intent_set), list(label_set)
intent_dict, label_dict = dict(), dict()
for a in range(len(intent_list)):
    intent_dict[intent_list[a]] = a+1
for a in range(len(label_list)):
    label_dict[label_list[a]] = a+1

with open('../data/ATIS/intent_dict.csv', 'w') as f:
    writer = csv.writer(f)
    header = ['intent', 'idx']
    writer.writerow(header)
    writer.writerows(intent_dict.items())

with open('../data/ATIS/label_dict.csv', 'w') as f:
    writer = csv.writer(f)
    header = ['label', 'idx']
    writer.writerow(header)
    writer.writerows(label_dict.items())


