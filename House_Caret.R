library(caret)
library(doMC)
library(dummies)
library(AppliedPredictiveModeling)
library(mlbench)

#parallel
registerDoMC()

#load data
training <- read.csv("/Users/jameshenson/housetrain.csv")
test <- read.csv("/Users/jameshenson/housetest.csv")

raw_train <-  read.csv("/Users/jameshenson/housetrain.csv")

#visualize
par(mfrow=c(1,2))
qqnorm(raw_train$SalePrice); qqline(raw_train$SalePrice)

qqplot(raw_train$SalePrice, ylab = "SalePrice Quantiles")
abline(0,1)

qqnorm(raw_train$GrLivArea); qqline(raw_train$GrLivArea)
qqplot(raw_train$GrLivArea, ylab = "GrLivArea Quantiles")
abline(0,1)

par(mfrow=c(1,2))
qqnorm(training$SalePrice); qqline(training$SalePrice)

qqplot(training$SalePrice, ylab = "SalePrice Quantiles")
abline(0,1)

qqnorm(training$GrLivArea); qqline(training$GrLivArea)
qqplot(training$GrLivArea, ylab = "GrLivArea Quantiles")
abline(0,1)

#transform saleprice
training$SalePrice <- log(training$SalePrice)

#transform GrLivArea
training$GrLivArea <- log(training$GrLivArea)

#dummy categoricals
dummy_train <- dummy.data.frame(training)
dummy_train[is.na(dummy_train)] <- 0

#Cross validate
trControl <- trainControl(method = "repeatedcv"
                          , number = 10
                          , repeats = 5
                          , allowParallel = TRUE)
#model
set.seed(1)
mod_glm <- train(SalePrice ~ .
             , data = dummy_train
             , method = "glm"
             , trControl = trControl)
set.seed(1)
mod_lm <- train(SalePrice ~ .
             , data = dummy_train
             , method = "lm"
             , trControl = trControl)
set.seed(1)
mod_blm <- train(SalePrice ~ .
                , data = dummy_train
                , method = "BstLm"
                , trControl = trControl)

set.seed(1)
mod_lm2 <- train(SalePrice ~ SaleTypeConLD
                 +FenceGdWo
                 +ScreenPorch
                 +WoodDeckSF
                 +GarageQualFa
                 +GarageQualEx
                 +GarageArea
                 +GarageCars
                 +FunctionalSev
                 +FunctionalMod
                 +FunctionalMin2
                 +FunctionalMaj2
                 +KitchenQualFa
                 +KitchenQualEx
                 +BsmtFullBath
                 +GrLivArea
                 +CentralAirN
                 +HeatingQCEx
                 +HeatingGrav
                 +BsmtUnfSF
                 +BsmtFinSF2
                 +BsmtFinSF1
                 +BsmtFinType1GLQ
                 +FoundationStone
                 +FoundationPConc
                 +FoundationCBlock
                 +Exterior1stBrkFace
                 +ExterCondGd
                 +Exterior1stBrkFace
                 +RoofMatlMembran
                 +RoofMatlClyTile
                 +RoofStyleMansard
                 +RoofStyleHip
                 +RoofStyleGambrel
                 +RoofStyleGable
                 +RoofStyleFlat
                 +YearRemodAdd
                 +YearBuilt
                 +OverallCond
                 +OverallQual
                 +HouseStyle2Story
                 +Condition2RRAe
                 +Condition2PosN
                 +Condition1RRAe
                 +NeighborhoodStoneBr
                 +NeighborhoodNWAmes
                 +NeighborhoodMitchel
                 +NeighborhoodMeadowV
                 +NeighborhoodEdwards
                 +LandSlopeMod
                 +LandSlopeGtl
                 +LotConfigCulDSac
                 +LotConfigCorner
                 +LandContourLow
                 +LotArea
                 +MSZoningRL
                 +MSZoningFV
                , data = dummy_train
                , method = "lm"
                , trControl = trControl)

set.seed(1)
mod_lm3 <- train(SalePrice ~ SaleTypeConLD
                 +ScreenPorch
                 +WoodDeckSF
                 +GarageQualEx
                 +GarageArea
                 +GarageCars
                 +FunctionalSev
                 +FunctionalMod
                 +FunctionalMaj2
                 +KitchenQualEx
                 +BsmtFullBath
                 +GrLivArea
                 +CentralAirN
                 +HeatingQCEx
                 +HeatingGrav
                 +BsmtUnfSF
                 +BsmtFinSF2
                 +BsmtFinSF1
                 +Exterior1stBrkFace
                 +Exterior1stBrkFace
                 +RoofMatlMembran
                 +RoofMatlClyTile
                 +RoofStyleHip
                 +RoofStyleGambrel
                 +RoofStyleGable
                 +YearRemodAdd
                 +YearBuilt
                 +OverallCond
                 +OverallQual
                 +Condition2RRAe
                 +Condition2PosN
                 +Condition1RRAe
                 +NeighborhoodStoneBr
                 +NeighborhoodNWAmes
                 +NeighborhoodMitchel
                 +NeighborhoodMeadowV
                 +NeighborhoodEdwards
                 +LandSlopeMod
                 +LandSlopeGtl
                 +LotConfigCulDSac
                 +LotArea
                 +MSZoningRL
                 +MSZoningFV
                 , data = dummy_train
                 , method = "lm"
                 , trControl = trControl)
#ETL for test set
#transform GrLivArea
test$GrLivArea <- log(test$GrLivArea)

#dummy categoricals
dummy_test <- dummy.data.frame(test)

#handle NA
dummy_test[is.na(dummy_test)] <- 0

#create empty columns to match train dataset
dummy_test["UtilitiesNoSeWa"] <- 0
dummy_test["Condition2RRAe"] <- 0
dummy_test["Condition2RRAn"] <- 0
dummy_test["Condition2RRNn"] <- 0
dummy_test["HouseStyle2.5Fin"] <- 0
dummy_test["RoofMatlClyTile"] <- 0
dummy_test["RoofMatlMembran"] <- 0
dummy_test["RoofMatlMetal"] <- 0
dummy_test["RoofMatlRoll"] <- 0
dummy_test["Exterior1stImStucc"] <- 0
dummy_test["Exterior1stStone"] <- 0
dummy_test["Exterior2ndOther"] <- 0
dummy_test["HeatingFloor"] <- 0
dummy_test["HeatingOthW"] <- 0
dummy_test["ElectricalMix"] <- 0
dummy_test["ElectricalNA"] <- 0
dummy_test["GarageQualEx"] <- 0
dummy_test["PoolQCFa"] <- 0
dummy_test["MiscFeatureTenC"] <- 0

#Append scores
score <- predict(mod_lm2, as.matrix(dummy_test))

df_score <- data.frame(score)
df2 <- data.frame(exp(df_score$score))





#RMSE
actual <- dummy_train$SalePrice
rmse <- (mean((actual - df_score)^2))^0.5
fin <- cbind(dummy_test$Id, df_score)
colnames(fin) <- c("Id","SalePrice")
fin$SalePrice <- exp(fin$SalePrice)

#write.csv(fin, file = 'housesub_lm3.csv', row.names = FALSE)