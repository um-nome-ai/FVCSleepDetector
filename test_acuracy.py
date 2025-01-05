import glob
import pandas as pd

labels = []
testcases = {}

for filename in glob.glob("./tests/*.csv"):
	name = filename[:filename.rfind(".")]
	name = name[name.rfind("/") + 1:]
	sections = name.split("-")
	num = int(sections[0][1])
	if len(sections) == 2:
		#label
		labels.append([num, filename])
	else:
		if num in testcases:
			testcases[num].append([float(sections[2]), filename])
		else:
			testcases[num] = [[float(sections[2]), filename]]

for i, labelname in labels:
	label = pd.read_csv(labelname)
	for thresh, testname in testcases[i]:
		test = pd.read_csv(testname)

		comparison = label['Sleeping'] == test['Sleeping']
		ratio = comparison.sum() / comparison.size * 100

		print(f"acurácia do t{i} com threshold de {thresh} é: {ratio:.2f}")

