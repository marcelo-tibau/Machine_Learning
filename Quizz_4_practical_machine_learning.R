## Q.1

# Load the data
library("ElemStatLearn")
data("vowel.train")
data("vowel.test")
library("caret")
library("pgmm")
library("rpart")
library("gbm")
library("lubridate")
library("forecast")
library("e1071")

# Set the variable y to be a factor variable in both the training and test set.Then set the seed to 33833.
testing <- vowel.test
training <- vowel.train
set.seed(33833)

# Fit (1) a random forest predictor relating the factor variable y to the remaining variables 
ModelFitRF <- train(as.factor(y)~., data = training, method = "rf", prox = TRUE)

# (2) a boosted predictor using the "gbm" method.
ModelFitBoost <- train(as.factor(y)~., data = training, method = "gbm", verbose = FALSE)

ModelFitRF
ModelFitBoost

Predict_RF <- predict(ModelFitRF, testing)
Predict_Boost <- predict(ModelFitBoost, testing)

Predict_RF
Predict_Boost

# What are the accuracies for the two approaches on the test data set?
confusionMatrix(Predict_RF, as.factor(testing$y))
confusionMatrix(Predict_Boost, as.factor(testing$y))

# What is the accuracy among the test set samples where the two methods agree?
confusionMatrix(Predict_RF, Predict_Boost)


## Q.2
# Load the data
library("caret")
library("gbm")
set.seed(3433)

library("AppliedPredictiveModeling")
data(AlzheimerDisease)

adData = data.frame(diagnosis,predictors)

inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]

training = adData[ inTrain,]

testing = adData[-inTrain,]

# Set the seed to 62433 and predict diagnosis with all the other variables using a random forest ("rf"), boosted trees ("gbm") and linear discriminant analysis ("lda") model. 
set.seed(62433)

ModelFitRFor <- train(diagnosis~., data = training, method = "rf", prox = TRUE)
ModelFitGBM <- train(diagnosis~., data = training, method = "gbm", verbose = FALSE)
ModelFitLDA <- train(diagnosis~., data = training, method = "lda")

# Predicting the outcomes

Pred_RFor <- predict(ModelFitRFor, testing)
Pred_GBM <- predict(ModelFitGBM, testing)
Pred_LDA <- predict(ModelFitLDA, testing)

confusionMatrix(Pred_RFor, testing$diagnosis)$overall[1]
confusionMatrix(Pred_GBM, testing$diagnosis)$overall[1]
confusionMatrix(Pred_LDA, testing$diagnosis)$overall[1]

# Stack the predictions together using random forests ("rf"). 
combined_data <- data.frame(Pred_RFor, Pred_GBM, Pred_LDA, diagnosis = testing$diagnosis)
Model_combined <- train(diagnosis~., data = combined_data, method = "rf")
Pred_Combined <- predict(Model_combined, combined_data)
confusionMatrix(Pred_Combined, testing$diagnosis)$overall[1]


## Q.3
# Load the data

set.seed(3523)

library(AppliedPredictiveModeling)

data(concrete)

inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]

training = concrete[ inTrain,]

testing = concrete[-inTrain,]

# Set the seed to 233 and fit a lasso model to predict Compressive Strength. 

set.seed(233)
ModelFitLasso <- train(CompressiveStrength ~., data = training, method = "lasso")

# Which variable is the last coefficient to be set to zero as the penalty increases?

ModelFitLasso$finalModel
plot.enet(ModelFitLasso$finalModel, xvar = "penalty", use.color = TRUE)


## Q.4
# Load the data on the number of visitors to the instructors blog
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv", dest="gaData.csv", mode="wb")
dat = read.csv("gaData.csv")

training = dat[year(dat$date) < 2012,]

testing = dat[(year(dat$date)) > 2011,]

tstrain = ts(training$visitsTumblr)

# Fit a model using the bats() function in the forecast package to the training time series. 
library("forecast")
ModelFitBats <- bats(tstrain, use.parallel = TRUE, num.cores = 4)
forecastModel <- forecast(ModelFitBats) 

# Then forecast this model for the remaining time points. 
forecastModel <- forecast(ModelFitBats, nrow(testing)) 

plot(forecastModel)

# For how many of the testing points is the true value within the 95% prediction interval bounds?
forecastModel_lower95 <- forecastModel$lower[,2]
forecastModel_upper95 <- forecastModel$upper[,2]
table((testing$visitsTumblr>forecastModel_lower95)&(testing$visitsTumblr<forecastModel_upper95))
true_value_per <- 226/nrow(testing)
true_value_per


## Q.5
# Load the data
set.seed(3523)

library(AppliedPredictiveModeling)

data(concrete)

inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]

training = concrete[ inTrain,]

testing = concrete[-inTrain,]

# Set the seed to 325 and fit a support vector machine using the e1071 package to predict Compressive Strength using the default settings. 
set.seed(325)
library("e1071")
ModelFitSVM <- svm(CompressiveStrength~., data = training)

# Predict on the testing set.
Pred_SVM <- predict(ModelFitSVM, testing)

# What is the RMSE?
RMSE(Pred_SVM, testing$CompressiveStrength)

