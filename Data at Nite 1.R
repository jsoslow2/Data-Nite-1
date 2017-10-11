rm(list=ls())
#Install Packages
install.packages("RCurl")
install.packages("ROCR",dep=T)
install.packages("randomForest")
install.packages("ggplot2")
install.packages("ggthemes")

#Load Packages
library(RCurl)
library(ROCR)
library(randomForest)
library(ggplot2)
library(ggthemes)

#Load Data
x <- getURL("https://raw.githubusercontent.com/jsoslow2/Data-Nite-1/master/data_at_nite.csv")
table <- read.csv(text = x)


#Explore Data
names(table)
nrow(table)
summary(table)

hist(table$Yards)
mean(table$Yards)
median(table$Yards)


#Set Creation
#table <- subset(table, table$OffenseTeam == 'DAL')

x <- table[, c('IsRush'
               , 'DefenseTeam'
               , 'OffenseTeam'
               , 'Formation'
               , 'DownAndDistance'
               , 'yrdline100'
               , 'TimeoutsLeft'
               , 'ScoreAndTime'
)]

x <- x[complete.cases(x), ]

toptable <- head(x, n=5000)
tailtable <- tail(x, n=5000)
#Sample Indexes
indexes = sample(1:nrow(x), size=0.2*nrow(x))

# Split data
test = x[indexes,]
dim(test)  
train = x[-indexes,]
dim(train) 


#Run Initial Logistic Regression
logit <- glm(IsRush ~.
             ,family = binomial, data=train)
summary(logit)


#Check Accuracy of model
p <- predict(logit, test, type="response")
error <- ifelse(p > .5 & test$IsRush == 1, 0, 1)
error[p < .5 & test$IsRush == 0] <- 0
1 - (sum(error) / nrow(test))

#Find area under curve of model
pr <- prediction(p, test$IsRush)
prflogit <- performance(pr, "tpr", "fpr")
plot(prflogit)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc


#Run Random Forest
blackbox <- randomForest(IsRush ~ .
                         , data=toptable, ntrees=100)
blackbox
varImpPlot(blackbox, type=2)

#Check Accuracy of model
p <- predict(blackbox, test, type="response")
error <- ifelse(p > .5 & test$IsRush == 1, 0, 1)
error[p < .5 & test$IsRush == 0] <- 0
1 - (sum(error) / nrow(test))

#Find area under curve of model
pr <- prediction(p, test$IsRush)
prfrf <- performance(pr, "tpr", "fpr")
plot(prfrf)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc


#Installing and Loading XGBoost
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
install.packages('xgboost')
library(xgboost)

#Creating matrix for extreme gradient boosted trees
sparse_matrix <- sparse.model.matrix(IsRush~.-1, data = train)
output_vector = train[,1]
sparse_matrix_test <- sparse.model.matrix(IsRush~.-1, data = test)
output_vector_test <- test[,1]

#Running XGboost
bst <- xgboost( data = sparse_matrix, label = output_vector, max_depth = 7, 
                eta = .01, nthread = 2, nrounds = 1200,gamma = 0, min_child_weight = 1, 
                subsample= .7, colsample_bytree=.7, objective = "binary:logistic")

#Check Accuracy of model
p <- predict(bst, sparse_matrix_test, type="response")
error <- ifelse(p > .5 & test$IsRush == 1, 0, 1)
error[p < .5 & test$IsRush == 0] <- 0
1 - (sum(error) / nrow(test))

#Find area under curve of model
pr <- prediction(p, test$IsRush)
prfxg <- performance(pr, "tpr", "fpr")
plot(prfxg)

auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc


#AUC comparisons
plot(prflogit,  col = 'GREEN')
plot(prfrf, add = TRUE, col = 'Orange')
plot(prfxg, add = TRUE, col = 'BLUE')
abline(a=0, b= 1)

legend(.6, .5, legend=c("Random","SVM", "Logit", "RandomForest", "XGBoost"),
       col=c("Black", "red", "green", "Orange", "Blue"), lty=1)


#Distribution of Errors Plot
train$prediction <- predict(blackbox, newdata = train, type = "response" )
test$prediction  <- predict(blackbox, newdata = test , type = "response" )

# distribution of the prediction score grouped by known outcome
ggplot(test, aes( prediction, color = as.factor(IsRush) ) ) + 
  geom_density( size = 1 ) +
  ggtitle( "Test Set's Predicted Score" ) + 
  scale_color_economist( name = "data", labels = c( "negative", "positive" ) ) + 
  theme_economist()


