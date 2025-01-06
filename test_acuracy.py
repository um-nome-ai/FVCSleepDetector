import glob
import pandas as pd
import matplotlib.pyplot as plt

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

res = []

for i, labelname in labels:
	label = pd.read_csv(labelname)
	for thresh, testname in testcases[i]:
		test = pd.read_csv(testname)

		comparison = label['Sleeping'] == test['Sleeping']
		ratio = comparison.sum() / comparison.size * 100
		res.append([i, thresh, ratio])
		#print(f"acurácia do t{i} com threshold de {thresh} é: {ratio:.2f}")

df = pd.DataFrame(res, columns=["Test Case", "Threshold", "Acuracy"]) \
	.sort_values(by=['Test Case', 'Threshold'], ascending=[True, True]) \
	.pivot(index="Test Case", columns="Threshold", values="Acuracy")

#df.loc['Média'] = df.max(axis=1)
#print(df)
#m = pd.DataFrame(df.index)
#m["Limite 0.4"] = df[""]

#ax = df[0.4].plot(kind='bar', title='Acurácia com limite 0.4')

fig, ax = plt.subplots()

ax.set_title('Acurácia do Limite 0.4')
ax.set_ylabel('Acurácia')
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
plt.savefig("result2.png")


#print(df.to_latex(decimal=",", float_format="%.2f"))

#out.to_csv("results.csv", index=False, sep=';', decimal=",")
#ax = df.plot(kind='bar', title='Acurácia com threshold por vídeo')
#plt.legend(loc='upper left', bbox_to_anchor=(0.98, 1), frameon=False)  # No box around the legend
#plt.savefig("result.png")