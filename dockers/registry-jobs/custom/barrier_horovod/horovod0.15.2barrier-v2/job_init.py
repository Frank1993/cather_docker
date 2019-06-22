import shutil
import sys
import os
import tempfile


tmp_dir = tempfile.mkdtemp()
print(tmp_dir)

for lib_path in sys.path:
    if os.path.exists(lib_path):
        shutil.copytree(lib_path, os.path.join(tmp_dir, os.path.split(lib_path)[-1]))

shutil.rmtree(tmp_dir)
