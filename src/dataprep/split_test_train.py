import sys
import json
import random

train_split = 0.70
val_split = 0.15
test_split = 0.15

"""
in_f = open("../../data/CHAIR/json/name_to_id.json", 'r')
names = (json.loads((in_f.readlines()[0]).strip())).keys()
"""
in_f = open("./meshes-chair122.txt")
names = [n.strip().split('.')[0] for n in in_f.readlines()]
in_f.close()
random.shuffle(names)
split = {}
num_models = float(len(names))

train_list = random.sample(names, int(train_split*num_models))
split['train'] = train_list
print "Train: %i" % len(train_list)

names = list(set(names) - set(train_list))
val_list = random.sample(names, int(val_split*num_models))
split['val'] = val_list
print "Val: %i" % len(val_list)

test_list = list(set(names) - set(val_list))
split['test'] = test_list
print "Test: %i" % len(test_list)

out_f = open("split.json", 'w')
out_f.write(json.dumps(split))
out_f.close()
