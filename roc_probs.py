from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import csv
import ast

reader = csv.reader(open('results.csv', 'r'))
row = next(reader)
print(row)
#['id', 'truth', 'final_pred', 'predictions', 'K', 'probs']

AD = 1
NC = 0

truth = []
probs = []
pred = []


for row in reader:
	truth.append(AD if row[1] == 'AD' else NC)
	pred.append(AD if row[2] == 'AD' else NC)
	sum_equal_avg = 0
	for p in ast.literal_eval(row[-1]):
		sum_equal_avg += p[0]
	probs.append(sum_equal_avg/10)

sum_ = 0
for i, x in enumerate(truth):
	if x == pred[i]:
		sum_ += 1

print("ACC " + str(sum_/len(truth)))

fpr, tpr, thresholds = roc_curve(truth, probs)
roc_auc = auc(fpr, tpr)
print(roc_auc)

with open('fine_tuning_probs.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    for i, t in enumerate(truth):
    	writer.writerow([t, probs[i]])

writeFile.close()

plt.figure()
plt.plot(fpr, tpr, color='blueviolet', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Fine Tuning')
plt.legend(loc="lower right")
plt.show()



