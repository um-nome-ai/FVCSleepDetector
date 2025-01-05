import sys
import pandas as pd
name = sys.argv[1]
lb = int(sys.argv[2])
hb = int(sys.argv[3])
size = 273

def in_bound(n):
	return n >= lb and n <= hb

base = list(range(size))

res = [[i, 1] if in_bound(i) else [i, 0] for i in base]

pd_data = pd.DataFrame(res, columns=["Frame","Sleeping"])
pd_data.to_csv(f"tests/{name}-label.csv")

