evalFT <-  predictionStats_binary(fine_tuning_probs, plotname="AUC Fine Tuning")
evalFT$aucs
evalFT$berror
evalFT$accc
summary(evalFT$CM.analysis)

evalCustom <-  predictionStats_binary(custom_probs, plotname="AUC Custom Model")
evalCustom$aucs
evalCustom$berror
evalCustom$accc
summary(evalCustom$CM.analysis)

evalTL <-  predictionStats_binary(transfer_learning_probs, plotname="AUC Transfer Learning")
evalTL$aucs
evalTL$berror
evalTL$accc
summary(evalTL$CM.analysis)

balancedError <- rbind(FT=evalFT$berror, Custom=evalCustom$berror, TL=evalTL$berror)
bpCI <- barPlotCiError(as.matrix(balancedError),metricname = "Balanced Error",
                       thesets = c("Test Set"),
                       themethod = rownames(balancedError),
                       main = "Balanced Error",
                       offsets = c(0.5,1),
                       scoreDirection = "<",ho = 0.5,
                       args.legend = list(bg = "white",x = "bottomright"), col = terrain.colors(nrow(balancedError)))

AUC <- rbind(FT=evalFT$aucs, Custom=evalCustom$aucs, TL=evalTL$aucs)
AUCCI <- barPlotCiError(as.matrix(AUC),metricname = "ROC AUC",
                        thesets = c("Test Set"),
                        themethod = rownames(balancedError),
                        main = "ROC AUC",
                        offsets = c(0.5,1),
                        args.legend = list(bg = "white",x = "bottomright"), col = terrain.colors(nrow(balancedError)))