import sys
sys.path.append("./binvox-rw-py")
import binvox_rw
import os
import scipy.io as sio

src_dir = "./ShapeNet/CHAIR/binvox"
dest_dir = "./ShapeNet/CHAIR/mat"

for root, dirs, files in os.walk(src_dir):
    for fn in files:
        with open(os.path.join(src_dir, fn), 'rb') as f:
            m = binvox_rw.read_as_3d_array(f)
        name = fn.split(".")[0]
        data = {
            "id": name,
            "data": m.data,
            "dims": m.dims,
            "translate": m.translate,
            "scale": m.scale
        }
        sio.savemat('%s/%s.mat' % (dest_dir, name), data)
    break
