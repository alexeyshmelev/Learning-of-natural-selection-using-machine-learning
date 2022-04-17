import os
import subprocess

files = os.listdir("/".join(os.path.abspath(__file__).split('/')[:-1]))

for file in files:
    print(file)
    subprocess.call("sbatch " + os.path.abspath(__file__).split('/')[:-1] + "/" + file)