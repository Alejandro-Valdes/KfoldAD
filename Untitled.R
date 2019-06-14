data = read.csv("fine_tuning_probs.csv", sep=",")
data
eval <-  predictionStats_binary(predictions=cbind(gt,pred), plotname="AUC")
eval$aucs
eval$berror
eval$accc
summary(eval$CM.analysis)