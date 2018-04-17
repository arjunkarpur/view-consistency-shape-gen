import os

size = 20
base_dir = "./ShapeNet/CHAIR/obj"

cmds = []
for root, dirs, files in os.walk(base_dir):
    for f in files:
        cmd = "../binvox -pb -cb -d %i %s" % \
            (size, os.path.join(base_dir, f))
        cmds.append(cmd)
    break


f = open('cmds.txt', 'w')
for c in cmds:
    f.write("%s\n" % c)
f.close()
