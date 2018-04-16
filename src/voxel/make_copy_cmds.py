import os

synset = "03001627"
base_dir = "../ShapeNet"
raw_dir = os.path.join(base_dir, "raw", synset)
dest_dir = os.path.join(base_dir, "CHAIR", "obj")

cmds = []
for root, dirs, files in os.walk(raw_dir):
    for id_ in dirs:
        cmds.append("cp %s %s" % \
            (os.path.join(raw_dir, id_, "model.obj"), os.path.join(dest_dir, "%s.obj" % id_)))
    break


f = open('cmds.txt', 'w')
for c in cmds:
    f.write("%s\n" % c)
f.close()
