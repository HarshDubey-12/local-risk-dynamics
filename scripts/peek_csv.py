import sys

path='data/raw/F-F_Research_Data_5_Factors_2x3.csv'
with open(path,'r') as f:
    for i in range(10):
        line=f.readline()
        sys.stdout.write(f"{i}: {line!r}\n")
