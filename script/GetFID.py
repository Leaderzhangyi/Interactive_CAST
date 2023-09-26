import os 
import shutil
root_path = "results/dh_256/test_latest/images"
fake_path = "results/dh_256/test_latest/images/fake_B"
real_path = "results/dh_256/test_latest/images/real_B"
for file in os.listdir(root_path):
    if file.endswith("fake_B.png"):
        shutil.copy(os.path.join(root_path,file),os.path.join(fake_path,file))
    elif file.endswith("real_B.png"):
        shutil.copy(os.path.join(root_path,file),os.path.join(real_path,file))
    else:
        continue




