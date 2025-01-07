import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

labels = []
testcases = {}

for filename in glob.glob("./tests/*.csv"):
	name = filename[:filename.rfind(".")]
	name = name[name.rfind("/") + 1:]
	sections = name.split("-")
	num = int(sections[0][-1])
	if len(sections) == 2:
		#label
		labels.append([num, filename])
	else:
		if num in testcases:
			testcases[num].append([float(sections[2]), filename])
		else:
			testcases[num] = [[float(sections[2]), filename]]

res = []

for i, labelname in labels:
	label = pd.read_csv(labelname)
	for thresh, testname in testcases[i]:
		test = pd.read_csv(testname)

		comparison = label['Sleeping'] == test['Sleeping']
		ratio = comparison.sum() / comparison.size * 100
		res.append([i, thresh, ratio])
		#print(f"acurácia do t{i} com threshold de {thresh} é: {ratio:.2f}")

		precision = 100*precision_score(y_true=label['Sleeping'], y_pred=test['Sleeping'], zero_division=1.0)
		recall = 100*recall_score(y_true=label['Sleeping'], y_pred=test['Sleeping'], zero_division=1.0)
		f1 = 100*f1_score(y_true=label['Sleeping'], y_pred=test['Sleeping'], zero_division=1.0)
		
		res.append([i, thresh, precision, recall, f1])
		#print(f"acurácia do t{i} com threshold de {thresh} é: {ratio:.2f}")

#print(res)

df = pd.DataFrame(res, columns=["Test Case", "Threshold", "Precision", "Recall", "f1"]) \
	.sort_values(by=['Test Case', 'Threshold'], ascending=[True, True]) \
	.pivot(index="Test Case", columns="Threshold", values="Recall")

#df.loc['Média'] = df.max(axis=1)
#print(df)
#m = pd.DataFrame(df.index)
#m["Limite 0.4"] = df[""]

#ax = df[0.4].plot(kind='bar', title='Acurácia com limite 0.4')

fig, ax = plt.subplots()

ax.set_title('Recall do Limite 0.4')
ax.set_ylabel('Recall')
ax.set_xlabel("Caso Teste")

# Plot the 'Age' column as a bar plot
bars = ax.bar(df.index, df[0.4], color="skyblue")

# Add the value of the bar at the top of each bar
for bar in bars:
    # Get the height of the bar (the value)
    height = bar.get_height()
    # Add text at the top of the bar
    ax.text(bar.get_x() + bar.get_width() / 2, height - 2,  # x, y position
            "{:.2f}".format(height),  # The value to display
            ha='center',  # Horizontal alignment (centered)
            va='bottom',  # Vertical alignment (bottom of the text)
            fontsize=10)

plt.xticks(range(9))

plt.ylim(60, 100)
plt.savefig("result4.png")


print(df.to_latex(decimal=",", float_format="%.2f"))

#out.to_csv("results.csv", index=False, sep=';', decimal=",")
#ax = df.plot(kind='bar', title='Acurácia com threshold por vídeo')
#plt.legend(loc='upper left', bbox_to_anchor=(0.98, 1), frameon=False)  # No box around the legend
#plt.savefig("result.png")

# precision
#\begin{tabular}{lrrrrr}
#\toprule
#Threshold & 0,100000 & 0,300000 & 0,400000 & 0,500000 & 0,600000 \\
#Test Case &  &  &  &  &  \\
#\midrule
#0 & 100,00 & 97,89 & 81,77 & 63,60 & 63,74 \\
#1 & 100,00 & 90,32 & 66,67 & 44,28 & 44,12 \\
#2 & 100,00 & 100,00 & 100,00 & 95,12 & 61,90 \\
#3 & 100,00 & 100,00 & 0,00 & 0,00 & 0,00 \\
#4 & 100,00 & 97,53 & 91,38 & 78,81 & 77,66 \\
#5 & 100,00 & 0,00 & 0,00 & 0,00 & 0,00 \\
#6 & 100,00 & 95,74 & 92,59 & 85,71 & 51,56 \\
#7 & 100,00 & 0,00 & 0,00 & 0,00 & 0,00 \\
#8 & 100,00 & 97,67 & 90,37 & 71,25 & 62,64 \\
#\bottomrule
#\end{tabular}

# recall
#\begin{tabular}{lrrrrr}
#\toprule
#Threshold & 0,100000 & 0,300000 & 0,400000 & 0,500000 & 0,600000 \\
#Test Case &  &  &  &  &  \\
#\midrule
#0 & 0,00 & 79,89 & 95,40 & 99,43 & 100,00 \\
#1 & 0,00 & 93,33 & 100,00 & 100,00 & 100,00 \\
#2 & 0,00 & 57,05 & 99,36 & 100,00 & 100,00 \\
#3 & 100,00 & 100,00 & 100,00 & 100,00 & 100,00 \\
#4 & 0,00 & 74,53 & 100,00 & 100,00 & 100,00 \\
#5 & 100,00 & 100,00 & 100,00 & 100,00 & 100,00 \\
#6 & 0,00 & 34,09 & 94,70 & 100,00 & 100,00 \\
#7 & 100,00 & 100,00 & 100,00 & 100,00 & 100,00 \\
#8 & 0,00 & 49,12 & 98,83 & 100,00 & 100,00 \\
#\bottomrule
#\end{tabular}