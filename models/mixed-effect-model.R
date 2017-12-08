DirPath = "D:\\Google Drive\\SM2_SD\\CS109A\\CS109_qiuhao\\final project\\data (1)\\ADNI\\"
col = c(rep("character",94))
col[c(9,11,15,16:42,45:51,54:62,65:93)] = "double"# OBJECTID
Result<-read.csv(file=paste0(DirPath,"ADNIMERGE.csv"), header = TRUE, sep = ",",dec = ".", comment.char = "",na.strings ="")

A = lapply(Result[,c(9,15,16:42,45:51,54:62,65:90,92:93)],function(x) {
  # if(typeof(x) == "double")
  {
    x = (x - min(x,na.rm = T))/(max(x,na.rm = T) - min(x,na.rm = T))
  }
  return(x)
  
}
)
A = data.frame(A)

library(glmm)
library(nlme)
library(glmnet)
library(dummies)

## imputation
library(mice)
# A_1 <- mice(A, m=5, maxit = 50, method = 'pmm', seed = 500)
A_1 = readRDS("D:\\Google Drive\\SM2_SD\\CS109A\\A_1.rds")
A = complete(A_1)
A_2 = cbind(A_2,dummies::dummy(Result$PTGENDER),Result$PTEDUCAT,dummies::dummy(Result$PTETHCAT),dummies::dummy(Result$PTRACCAT),dummies::dummy(Result$PTMARRY),dummies::dummy(Result$COLPROT))
# saveRDS(A_1,"D:\\Google Drive\\SM2_SD\\CS109A\\A_1")

#####################
## take compute cases
Outcome <- (Result$DX == "Dementia")*1
A_2 = A_2[!is.na(Outcome),]
Result_Complete = Result[!is.na(Outcome),]
Outcome = Outcome[!is.na(Outcome)]


## change variables
names(A_2)[names(A_2) == 'Result$PTEDUCAT'] <- 'PTEDUCAT'
names(A_2)[names(A_2) == 'A_2Hisp/Latino'] <- 'A_2Hisp_Latino'
names(A_2)[names(A_2) == 'A_2Not Hisp/Latino'] <- 'A_2Not_Hisp_Latino'
names(A_2)[names(A_2) == 'A_2Am Indian/Alaskan'] <- 'A_2Am_Indian_Alaskan'
names(A_2)[names(A_2) == 'A_2Hawaiian/Other PI'] <- 'A_2Hawaiian_Other_PI'
names(A_2)[names(A_2) == 'A_2More than one'] <- 'A_2More_than_one'
names(A_2)[names(A_2) == 'A_2Never married'] <- 'A_2Never_married'

## Lasso --- get variables
Lasso_CV_mod = cv.glmnet(x = as.matrix(A_2), y =  as.matrix(Outcome),alpha = 1,family="binomial",nlambda = 100,standardize = F)
## fit with best lamda
Lasso_mod_final = glmnet(x = as.matrix(A_2), y =  as.matrix(Outcome),alpha = 1,family="binomial",lambda = Lasso_CV_mod$lambda.min,standardize = F)
coef(Lasso_mod_final,Lasso_CV_mod$lambda.min)

##create confusion matrix
Outcome_pred = predict(Lasso_mod_final,newx = as.matrix(A_2),s = Lasso_CV_mod$lambda.min,type = "response")  
Outcome_pred[Outcome_pred>0.5] = 1
Outcome_pred[Outcome_pred<0.5] = 0
cm = as.matrix(table(Actual = Outcome,  Predicted = Outcome_pred))
accuracy = sum(diag(cm)) / sum(cm) 


### preediction
M = Result_Complete$MP
PTID = Result_Complete$PTID

##spliting training and testing
set.seed(100)
ID = unique(PTID)
Index = runif(length(ID),0,1)
Index_Train = is.element(PTID,ID[Index>0.5])
Index_Test = is.element(PTID,ID[Index<0.5])


## baseline
M_threshold = 0
for(M_threshold in unique(M))
{
  Index = M<=M_threshold
  Lasso_CV_mod = cv.glmnet(x = as.matrix(A_2[Index&Index_Train,c(1:5,7:36,38:95)]), y =  as.matrix(Outcome[Index&Index_Train]),alpha = 1,family="binomial",nlambda = 100,standardize = F)
  Lasso_mod_final = glmnet(x = as.matrix(A_2[Index&Index_Train,c(1:5,7:36,38:95)]), y =  as.matrix(Outcome[Index&Index_Train]),alpha = 1,family="binomial",lambda = Lasso_CV_mod$lambda.min,standardize = F)
  
  Outcome_pred_test = predict(Lasso_mod_final,newx = as.matrix(A_2[(!Index)&(Index_Test),c(1:5,7:36,38:95)]),s = Lasso_CV_mod$lambda.min,type = "response")  
  Outcome_pred_test[Outcome_pred_test>0.5] = 1
  Outcome_pred_test[Outcome_pred_test<0.5] = 0
  cm_test = as.matrix(table(Actual = Outcome[(!Index)&(Index_Test)],  Predicted = Outcome_pred_test))
  
  Outcome_pred_train = predict(Lasso_mod_final,newx = as.matrix(A_2[Index&Index_Train,c(1:5,7:36,38:95)]),s = Lasso_CV_mod$lambda.min,type = "response")  
  Outcome_pred_train[Outcome_pred_train>0.5] = 1
  Outcome_pred_train[Outcome_pred_train<0.5] = 0
  cm_train = as.matrix(table(Actual = Outcome[Index&Index_Train],  Predicted = Outcome_pred_train))
  
  cat(sprintf("training %f %f\n",M_threshold,accuracy = sum(diag(cm_train)) / sum(cm_train)))
  cat(sprintf("testing %f %f\n",M_threshold,accuracy = sum(diag(cm_test)) / sum(cm_test))) 
}

#########################################
### mixed-effect model
library(lme4)
M = Result_Complete$MP[Index_Train]
PTID = Result_Complete$PTID[Index_Train]
Outcome_temp = Outcome[Index_Train]
mod <- glmer(Outcome_temp~A_2Asian+AGE+Entorhinal+A_2Divorced+EcogPtLang_bl+EcogSPDivatt+Ventricles_bl+EcogSPVisspat_bl+A_2ADNIGO+EcogPtVisspat_bl+EcogPtDivatt+PIB_bl+EcogSPPlan_bl+Entorhinal_bl+Hippocampus+AV45+EcogSPOrgan+A_2Married+PTEDUCAT+EcogPtOrgan_bl+RAVLT_learning_bl+RAVLT_perc_forgetting_bl+EcogPtMem_bl+AV45_bl+EcogSPMem_bl+A_2Not_Hisp_Latino+EcogSPMem+APOE4+A_2Black+EcogPtMem+CDRSB_bl+M+A_2Never_married+ADAS13+(1|PTID),data = A_2[Index_Train,],family = binomial,control = glmerControl(optimizer = "Nelder_Mead"),nAGQ = 10)

output = predict(mod,re.form = NA,A_2[Index_Test,])
output[output>0.5] = 1
output[output<0.5] = 0
cm = as.matrix(table(Actual = Outcome[Index_Test],  Predicted = output))
accuracy = sum(diag(cm)) / sum(cm) 
print(accuracy)
output = predict(mod,A_2[Index_Train,])
output[output>0.5] = 1
output[output<0.5] = 0
cm = as.matrix(table(Actual = Outcome[Index_Train],  Predicted = output))
accuracy = sum(diag(cm)) / sum(cm) 
print(accuracy)

mod1<-glm(Outcome_temp~A_2Asian+AGE+Entorhinal+A_2Divorced+EcogPtLang_bl+EcogSPDivatt+Ventricles_bl+EcogSPVisspat_bl+A_2ADNIGO+EcogPtVisspat_bl+EcogPtDivatt+PIB_bl+EcogSPPlan_bl+Entorhinal_bl+Hippocampus+AV45+EcogSPOrgan+A_2Married+PTEDUCAT+EcogPtOrgan_bl+RAVLT_learning_bl+RAVLT_perc_forgetting_bl+EcogPtMem_bl+AV45_bl+EcogSPMem_bl+A_2Not_Hisp_Latino+EcogSPMem+APOE4+A_2Black+EcogPtMem+CDRSB_bl+M+A_2Never_married+ADAS13,data = A_2[Index_Train,],family = binomial)

output = predict(mod1,A_2[Index_Test,])
output[output>0.5] = 1
output[output<0.5] = 0
cm = as.matrix(table(Actual = Outcome[Index_Test],  Predicted = output))
accuracy = sum(diag(cm)) / sum(cm) 
print(accuracy)
output = predict(mod1,A_2[Index_Train,])
output[output>0.5] = 1
output[output<0.5] = 0
cm = as.matrix(table(Actual = Outcome[Index_Train],  Predicted = output))
accuracy = sum(diag(cm)) / sum(cm)
print(accuracy)