# Prediction Assignment - Practical Machine Learning
## You will also use your prediction model to predict 20 different test cases.

# I had to rewrite the codes to answer the Course Project Prediction Quizz. 

# Codes to load the libraries to be used and setseed:
library(Hmisc)
library(caret)
library(randomForest)
library(foreach)
library(doParallel)
set.seed(1990)

# Codes to download and read the train and test sets:
training.file   <- 'pml-training.csv'
test.cases.file <- 'pml-test.csv'
training.url    <- 'http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
test.cases.url  <- 'http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'

download.file(training.url, training.file)
download.file(test.cases.url,test.cases.file )

# cleaning data
training.df   <-read.csv(training.file, na.strings=c("NA","#DIV/0!", ""))
test.cases.df <-read.csv(test.cases.file , na.strings=c("NA", "#DIV/0!", ""))
training.df<-training.df[,colSums(is.na(training.df)) == 0]
test.cases.df <-test.cases.df[,colSums(is.na(test.cases.df)) == 0]

training.df   <-training.df[,-c(1:7)]
test.cases.df <-test.cases.df[,-c(1:7)]

# Create a random sample of the data into training and test sets.seed(998)

inTraining.matrix    <- createDataPartition(training.df$classe, p = 0.75, list = FALSE)
training.data.df <- training.df[inTraining.matrix, ]
testing.data.df  <- training.df[-inTraining.matrix, ]

# Use Random Forests Package
registerDoParallel()
classe <- training.data.df$classe
variables <- training.data.df[-ncol(training.data.df)]


rf <- foreach(ntree=rep(250, 4), .combine=randomForest::combine, .packages='randomForest') %dopar% {
  randomForest(variables, classe, ntree=ntree) 
}


training.predictions <- predict(rf, newdata=training.data.df)
confusionMatrix(training.predictions,training.data.df$classe)


testing.predictions <- predict(rf, newdata=testing.data.df)
confusionMatrix(testing.predictions,testing.data.df$classe)

# code for submission

feature_set <- colnames(training.df)
newdata     <- test.cases.df


x <- newdata 
x <- x[feature_set[feature_set!='classe']]
answers <- predict(rf, newdata=x)

answers
