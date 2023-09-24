import os

output = "good"

os.system(f"rm {output}.txt")
for i in range(2,12):
    command = f"./matrix {2 ** i} 32 >> {output}.txt"
    print(command)
    os.system(command)
    