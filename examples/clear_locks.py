import os
import shutil
from glob import glob

fl = glob("cache-wf/*.lock")
checksums = [val.split("_")[1][:64] for val in fl]
paths = []
for checksum in checksums:
    paths.extend(glob(f"cache-wf/*{checksum}*"))
for path in paths:
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.unlink(path)
