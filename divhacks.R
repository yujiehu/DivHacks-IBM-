###################################################
###################################################
# Data entry
test = read.csv('MNIST_test.csv')
train = read.csv('MNIST_train.csv')

# Required task 
############### Task1 ##############################

# Generate the EASY or HARD labels for each digits for train data 
# set therodhold to be mean of the PC
train$PC = apply(train[,3:23], 1, sum)/21
train$level = ifelse(train$PC>mean(train$PC),'EASY','HARD')

# Generate the EASY or HARD labels for each digits for test data 
test$PC = apply(test[,3:23], 1, sum)/21
test$level = ifelse(test$PC>mean(test$PC),'EASY','HARD')

############### Task2 ##############################

# Easy/Hard among 0 – 9
rate = rep(0,10)
# mean PC for each number 
for (i in 1:10){
  rate[i] = sum(train$PC[which(train$Label == i-1)])/ sum(train$Label == i-1)
}

# easy & hard prediction rate
easy = rep(0,10)
hard = rep(0,10)
# mean PC for each number 
for (i in 1:10){
  easy[i] = sum(train$Label == i-1 & train$level=='EASY')
  hard[i] = sum(train$Label == i-1 & train$level=='HARD')
}
PCdata<-cbind(easy,hard)
rownames(PCdata)<-c(0:9)



# plot 
##average PC for each number
plot(rate,type = "b", xlim=c(0,10),ylim=c(0.8, 1), ylab = "Digits", xlab = "Correct Percentage",
     main = "Average PC for Each Digits",
      pch=16)
plot.new()
plot(x=p.valueN , y=entN, ylab = "Entrophy", xlab = "Probability",
     main = "Entrophy against Probability (Natural log)",
     col=ifelse(p.valueN==max.pN, "red", "black"), pch=16) 
# plot p entrophy against proba text(x=max.pN, y=0.27, labels='Maximum: 0.3678186', cex= 1, pos=3, col ='red') #add text

##number of hard and easy for each value
barplot(t(PCdata), main="Number of HARD and EASY Classification for Each Value",
        xlab="Digits for MINIST", col=c("lightblue","blue"))
#beside=TRUE

barplot()



############### Task3 ##############################
############### Classifer 1
#SVM(linear)

train$level<-as.factor(train$level)
svm1<-svm(level~.,data=train[,c(-1,-24)], cost=0.01,kernel="linear")
summary(svm1)

## results from train dataset
pre<-fitted(svm1)
train_real <- train$level
table(pre,train_real)

## cv to tune parameter(cost)
svmTune <- tune(svm, 
                level~.,data=train[,c(-1,-24)],
                kernel='linear',
                ranges=list(cost=10^(-2:2)))

## choose cost=1
## results from test dataset

svm1<-svm(level~.,data=train[,c(-1,-24)], cost=1,kernel="linear")
summary(svm1)
## results from train dataset
pre<-fitted(svm1)
train_real <- train$level
table(pre,train_real)

pred3<-predict(svm1,test[,c(-1,-24,-25)])
test_real <- test$level
table(pred3,test_real)

accuracy<- sum(diag(table(pred3,test_real)))/10000


################# Classifer 2

# Random Forest
require('randomForest')

train.X =  train[,c(2:23)]
train.Y =  train[,25]
test.X = test[,c(2:23,25)]

### re-organize train data for random forest

RFdata = train[,-c(1,24)]
RFdata$level = as.factor(RFdata$level)

# Create new function to perform 5-fold cross-validation for RF model 
cv.function.gbm <- function(X.train, y.train, mtry, K) {
  
  n        <- length(y.train)
  n.fold   <- floor(n/K)
  s        <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    train.data  <- X.train[s != i,]
    train.label <- y.train[s != i]
    test.data   <- X.train[s == i,]
    test.label  <- y.train[s == i]
    
    fit  <- randomForest(as.factor(train.label)~. , data = train.data, mtry, importance = TRUE)
    pred <- predict(fit, test.data)  
    cv.error[i] <- mean(pred != test.label)  
    
  }   
  return(c(mean(cv.error), sd(cv.error)))
}

# return cross validation result for 7 different tune the classifier
# result shows that mtry = 13 is the best, with accuracy rate 98.55% for test data.
cv_result = list()
for (i in c(3,5,7,9,11,13,15)) {
  cv_result[i]<-cv.function.gbm(train.X,train.Y ,num[i],5)
}
## RF with mtry =13
RanF = randomForest(level~.,data=RFdata, mtry=13,importance =TRUE)
RanF

yhat.bag = predict(RanF, test.X[,-23])
accuracy_RF<-sum(yhat.bag == test$level)/length(yhat.bag == test$level)










