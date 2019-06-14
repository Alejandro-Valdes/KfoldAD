from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

AD = 1
NC = 0
truth = [AD,AD,NC,NC,AD,NC,AD,AD,AD,NC,NC,NC,NC,NC,AD,AD,NC,AD,NC,NC,NC,NC,NC,NC,AD,AD,AD,NC,NC,NC,AD,NC,AD,AD,NC,AD,NC,NC,AD,NC,AD,NC,NC,NC,AD,NC,NC,NC,NC,NC,NC,NC,AD,AD,AD,NC,NC,NC,NC,NC,AD,AD,NC,NC,AD,NC,AD,AD,NC,AD,NC,AD,AD,NC,NC,AD,NC,AD,AD,AD,AD,NC,NC,NC,NC,AD,AD,AD,NC,NC,NC,AD,NC,AD,NC,AD,NC,NC,AD,NC,AD,AD,AD,NC,AD,NC,NC,NC,NC,AD,NC,AD,AD,NC,NC,AD,AD,NC,AD,AD,AD,NC,NC,AD,AD,AD,AD,AD,AD,NC,AD,NC,AD,NC,NC,NC,AD,NC,NC,AD,NC,AD,NC,NC,NC,AD,NC,NC,AD,AD,AD,AD,AD,AD,NC,AD,AD,NC,NC,AD,NC,AD,AD,AD,AD,AD,NC,AD,NC,NC,AD,AD,NC,AD,AD,AD,NC,NC,NC,NC,AD,AD,NC,NC,AD,AD,NC,NC,AD,NC,NC,AD,NC,NC,NC,NC,NC,NC,NC,NC,AD,AD,AD,AD,AD,AD,AD,NC,AD,NC,AD,NC,AD,NC,NC,NC,AD,NC,AD,NC,AD,AD,NC,NC,AD,AD,AD,NC,NC,AD,NC,AD,NC,NC,AD,NC,NC,AD,NC,NC,NC,NC,AD,AD,NC,NC,NC,AD,NC,NC,NC,AD,AD,AD,AD,NC,NC,NC,AD,NC,AD,NC,NC,NC,AD,AD,NC,NC,NC,AD,NC,AD,NC,AD,AD,NC,NC,NC,AD,NC,AD,AD,AD,NC,NC,NC,NC,NC,AD,NC,NC,AD,NC,NC,AD,AD,NC,AD,NC,AD,NC,AD,NC,AD,NC,AD,NC,AD,NC,AD,AD,AD,AD,NC,NC,NC,NC,AD,AD,NC,NC,NC,NC,NC,NC,NC,NC,NC,AD,AD,AD,NC,AD,AD,NC,NC,NC,NC,NC,AD,NC,AD,NC,NC,AD,NC,AD,NC,NC,AD,NC,AD,AD,NC,NC,AD,NC,NC,AD,NC,NC,NC,NC,NC,AD,AD,AD,AD,AD,NC,NC,AD,NC,NC,NC,AD,NC]
pred =  [AD,AD,NC,NC,NC,NC,AD,NC,AD,NC,NC,NC,AD,NC,NC,AD,NC,NC,AD,AD,NC,NC,NC,NC,AD,AD,AD,NC,NC,NC,NC,NC,AD,NC,NC,NC,AD,NC,AD,NC,AD,NC,NC,NC,AD,NC,NC,NC,NC,NC,NC,NC,NC,AD,AD,NC,NC,NC,NC,AD,AD,AD,NC,NC,AD,NC,AD,NC,NC,AD,AD,AD,AD,NC,AD,NC,NC,AD,AD,NC,AD,NC,NC,NC,NC,AD,NC,AD,AD,NC,AD,AD,NC,AD,NC,NC,NC,NC,AD,NC,AD,AD,AD,NC,AD,NC,NC,NC,NC,NC,NC,AD,AD,NC,NC,AD,AD,NC,NC,NC,AD,AD,NC,AD,AD,AD,NC,AD,AD,NC,AD,NC,AD,NC,NC,NC,AD,NC,AD,AD,NC,AD,NC,AD,NC,AD,NC,NC,AD,AD,NC,AD,AD,NC,NC,AD,AD,NC,AD,NC,NC,NC,AD,AD,AD,AD,NC,AD,NC,NC,AD,AD,AD,AD,AD,AD,NC,NC,AD,NC,AD,AD,AD,NC,AD,NC,NC,AD,NC,NC,NC,NC,NC,NC,NC,NC,NC,NC,NC,AD,AD,AD,AD,NC,AD,AD,AD,NC,AD,AD,AD,NC,AD,NC,NC,NC,NC,NC,AD,NC,AD,AD,NC,NC,AD,AD,AD,NC,NC,NC,NC,AD,NC,NC,AD,NC,NC,AD,NC,AD,NC,NC,AD,AD,AD,NC,NC,AD,AD,NC,NC,AD,AD,NC,AD,NC,NC,NC,AD,NC,AD,NC,NC,NC,AD,NC,NC,AD,NC,AD,NC,NC,NC,AD,NC,NC,NC,NC,AD,NC,AD,AD,AD,NC,NC,NC,NC,NC,NC,NC,NC,AD,NC,NC,AD,AD,NC,NC,NC,NC,AD,AD,NC,NC,NC,NC,NC,AD,AD,AD,AD,NC,AD,NC,NC,NC,AD,AD,AD,NC,NC,NC,NC,NC,NC,NC,NC,NC,NC,AD,NC,NC,AD,AD,NC,NC,NC,NC,NC,AD,NC,NC,NC,NC,AD,AD,NC,NC,NC,AD,NC,AD,AD,NC,AD,AD,NC,NC,AD,NC,NC,NC,NC,NC,AD,AD,NC,AD,AD,AD,NC,AD,AD,NC,NC,NC,AD]

fpr, tpr, thresholds = roc_curve(truth, pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()