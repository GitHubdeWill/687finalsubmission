import sys

with open(sys.argv[1], "r") as f:
    a = f.readlines()
    for li in range(len(a)):
        with open(str(li+1)+".csv","w+") as rf:
            rf.write(a[li])