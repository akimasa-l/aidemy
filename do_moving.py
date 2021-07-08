import glob
import subprocess
a=[]
for i in ["png","py","json"]:
    a+=glob.glob(f"./*.{i}")
print(a)
for i in a:
    if "do_moving" in i:
        continue
    print(f"git mv {i} learned/")
    subprocess.run(f"git mv {i} learned/")