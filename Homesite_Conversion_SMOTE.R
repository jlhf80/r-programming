# Homesite Quote Conversion Data
# Playing with Synthetic Minority Oversampling TechniquE (SMOTE)
# QuoteConversion_Flag refers to customers who purchased a policy. QuoteNumber is UID for customer
#clear global
rm(list=ls())

#load
library(DMwR)
library(pROC)
library(doMC)
library(caret)

#register cores
registerDoMC(4)

#data
dat <- read.csv("/Users/jameshenson/Downloads/train-2.csv")

dat$PersonalField84[is.na(dat$PersonalField84)] <- 0
dat$PersonalField84 <- as.factor(dat$PersonalField84)
dat$QuoteConversion_Flag <- ifelse(dat$QuoteConversion_Flag=='0',no='no',yes='yes')
dat$QuoteConversion_Flag <- as.factor(dat$QuoteConversion_Flag)
dat$Original_Quote_Date <- as.numeric(as.Date(dat$Original_Quote_Date, origin = "1970-01-01"))
dat <- droplevels(dat)



#stratified test and train
set.seed(222)
inTraining <- createDataPartition(dat$QuoteConversion_Flag, p=.01, list=FALSE)
training <- dat[inTraining,]
testing <- dat[-inTraining,]

#SMOTE
SMO <- SMOTE(QuoteConversion_Flag ~ ., data = training
             ,perc.over = 200
             ,perc.under = 200)

#cross validation
fitControl <- trainControl(method = "boot"
                           ,number = 5
                           ,repeats = 5
                           ,classProbs = TRUE
                           ,allowParallel = TRUE
                           ,savePredictions = TRUE
                           ,summaryFunction = twoClassSummary)

#fit models
set.seed(1)
tree1 <- train(training$QuoteConversion_Flag~ ., data = training
               ,method = "adaboost"
               ,na.action = na.exclude
               ,trControl = fitControl)
# throws error: one or more factor levels has no data
# due to miniority case of 'no' in y variable, sampled training sets contain no "no's" 
# one work around is to increase the training partition but this computationally expensive
# another work around is to use a holdout method or sampling method that inforces distribution of variables
# Or use SMOTE - Synthetic Minority Oversampling Technique

#convert to factor to avoid error
training$QuoteConversion_Flag <- as.factor(training$QuoteConversion_Flag)

train_smote <- SMOTE(QuoteConversion_Flag ~ ., data = training
                     ,perc.over = 100
                     ,perc.under = 200)

treeMod <- train(QuoteConversion_Flag ~ ., data = train_smote
                  ,method = "adaboost"
                  ,trControl = fitControl
                  ,allowParallel = TRUE
                  ,na.action = na.omit
                  ,metric = 'ROC')

predictors <- names(train_smote)[names(train_smote) != 'QuoteConversion_Flag']
pred <- predict(treeMod$finalModel,testing[,predictors])

auc <- roc(testing$QuoteConversion_Flag, pred)
print(auc)

plot(auc, main=paste('AUC:',round(auc$auc[[1]],2)))

