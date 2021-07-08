import glob
import subprocess
a=[]
for i in ["png","py","json"]:
    a+=glob.glob(f"./*.{i}")
print(a)
for i in a:
    print(f"git mv {i} learned/")