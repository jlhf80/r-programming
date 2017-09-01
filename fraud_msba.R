#clear global
rm(list=ls())
# load packages
library(doMC)
library(C50)
library(caret)
library(klaR)
library(MASS)

#register cores
registerDoMC(4)

#load data
data <- read.csv("/Users/jameshenson/Downloads/fraud_train.csv")
data2 <- read.csv("/Users/jameshenson/Downloads/fraud_test.csv")

#remove rows with missing values
data <- data[complete.cases(data),]
data2 <- data2[complete.cases(data2),]


#combine datasets
df <- rbind(data,data2)

#Unused levels in df, drop empty levels
df <- droplevels(df)

#stratified test and train
set.seed(1)
inTraining <- createDataPartition(df$FRAUD, p=.5, list=FALSE)
training <- df[inTraining,]
testing <- df[-inTraining,]


#cross validation
fitControl <- trainControl(method = "repeatedcv"
                           ,number = 10
                           ,repeats = 5
                           ,classProbs = TRUE
                           ,allowParallel = TRUE
                           ,summaryFunction = twoClassSummary)

#fit models
set.seed(2)
gbmFit1 <- train(FRAUDFOUND ~ ., data = training
                 ,method = "gbm"
                 ,trControl = fitControl
                 ,verbose = FALSE
                 ,metric = "ROC")

set.seed(2)
xgboost <- train(FRAUDFOUND ~ ., data = training
                 ,method = "xgbTree"
                 ,trControl = fitControl
                 ,verbose = FALSE
                 ,metric = "ROC")



#tree depth vs ROC
plot(gbmFit1)
plot(xgboost)


#confusion and metrics
gbmClasses <- predict(gbmFit1, testing)
gbmConfusion <- confusionMatrix(gbmClasses, testing$FRAUDFOUND)
gbmConfusion$byClass

xgbClasses <- predict(xgboost, testing)
xgbConfusion <- confusionMatrix(xgbClasses, testing$FRAUDFOUND)
xgbConfusion$byClass

#ROC 
gbmProbs <- predict(gbmFit1, testing, type = "prob")
gbmROC <- roc(predictor = gbmProbs$yes
              ,response = testing$y
              ,levels = rev(levels(testing$y)))
plot(gbmROC)
gbmROC$auc

xgbProbs <- predict(xgboost,testing,type="prob")
xgbROC <- roc(predictor = xgbProbs$Yes
              ,response = testing$FRAUDFOUND
              ,level = rev(levels(testing$FRAUDFOUND)))
plot(xgbROC)
xgbROC$auc


#prediction dataset
prediction <- predict(gbmFit1,testing, type = "prob")

#plot differences
resamps <- resamples(list(GBM = gbmFit1, XGB = xgboost))

trellis.par.set(caretTheme())
bwplot(resamps, layout = c(1,3))

#ROC, Sens, Spec
values <- resamples(list(gbm=gbmFit1, xgb=xgboost))
values$values
summary(values)
