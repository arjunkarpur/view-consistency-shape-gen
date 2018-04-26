import os
import sys

#TODO: input mesh isn't aligned along x,y,z. floor cropping isn't easy

# Input
model_id = sys.argv[1].split(".")[0]
padding = 0.1
floor_lambda = 0.0
DATA_BASE_DIR = "/home/arjun/research/thesis/shape-gen/data/RedwoodRGB_Chair/other/"
SRC_DIR = os.path.join(DATA_BASE_DIR, "meshes", "meshes_obj")
DEST_DIR = os.path.join(DATA_BASE_DIR, "meshes", "meshes_crop_obj")

# Read input obj file
print "Cropping %s" % model_id
in_f = open(os.path.join(SRC_DIR, "%s.obj" % model_id))
obj_lines = [l.strip() for l in in_f.readlines()]
in_f.close()
print "Obj file: %i lines" % len(obj_lines)

# Get corresponding keypoints file
in_f = open(os.path.join(DATA_BASE_DIR, "keypoints", "%s_picked_points.pp" % model_id))
keypoint_lines = [l.strip() for l in in_f.readlines() if "<point" in l]
in_f.close()
print "Keypoints file: %i lines" % len(keypoint_lines)

# Generate bounding box
mins = [float('inf'), float('inf'), float('inf')]
maxs = [-float('inf'), -float('inf'), -float('inf')]
dims = ["x", "y", "z"]
for l in keypoint_lines:
    for i in xrange(len(dims)):
        start = l.find("%s=\"" % dims[i])
        num_start = start+3
        splice = l[num_start:]
        num_end = splice.find("\"")
        val = float(l[num_start:(num_end+num_start)])
        if val <= mins[i]:
            mins[i] = val
        if val >= maxs[i]:
            maxs[i] = val
print "bbox mins: ", mins
print "bbox max: ", maxs

# Pad bounding box and crop out floor
for i in xrange(len(mins)):
    mins[i] -= padding
    maxs[i] += padding
maxs[1] -= floor_lambda
print "padded bbox mins: ", mins
print "padded bbox max: ", maxs

# Get list of verts and faces
verts = []
faces = []
for l in obj_lines:
    if len(l) < 1:
        pass
    elif l[0] == 'v':
        verts.append([float(val) for val in l.split(" ")[1:]])
    elif l[0] == 'f':
        faces.append([int(val) for val in l.split(" ")[1:]])
    else:
        pass
print 'v:',len(verts), ", f:", len(faces)

# Filter verts
vert_ctr = 1
reassign = {}
new_verts = []
for i in xrange(1,len(verts)):
    v = verts[i-1]
    valid = True
    for dim in xrange(3):
        if v[dim] < mins[dim] or v[dim] > maxs[dim]:
            valid = False
            break
    if valid:
        reassign[i] = vert_ctr
        new_verts.append(v)
        vert_ctr += 1
print "Num verts in bbox:", len(reassign)

# Filter faces
new_faces = []
for i in xrange(len(faces)):
    f = faces[i]
    if f[0] in reassign and f[1] in reassign and f[2] in reassign:
        new_faces.append([reassign[f[0]], reassign[f[1]], reassign[f[2]]])
print "Num new faces:", len(new_faces)

# Center mesh along the center
print "Centering around origin..."
avgs = [0.0, 0.0, 0.0]
for v in new_verts:
    for i in xrange(3):
        avgs[i] += v[i]
for i in xrange(3):
    avgs[i] /= len(new_verts)
for i in xrange(len(new_verts)):
    v = new_verts[i]
    for j in xrange(3):
        v[j] -= avgs[j]
    new_verts[i] = v


print "Writing new mesh to file"
out_fp = os.path.join(DEST_DIR, "%s.obj" % model_id)
out_f = open(out_fp, 'w')
for v in new_verts:
    out_f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
out_f.write("\n")
for f in new_faces:
    out_f.write("f %i %i %i\n" % (f[0], f[1], f[2]))
    pass
out_f.close()

