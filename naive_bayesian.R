library(matrixStats)
library(tidyverse)
#Input the data
data_raw <- read.csv('Project_3/Project3_dataset2.txt', sep = '\t', header = FALSE)

data_prep <- data_raw

#Group the non-character data columns into 3 levels by quantile
for(i in 1:ncol(data_raw)){
  if(is.numeric(data_raw[,i]) == TRUE){
    data_prep[,i] <- cut((data_raw[,i]), breaks = 3, labels = c("Low", "Medium", "High"))
  }
}

Accuracy <- function(y_pred, y_true) {
  Accuracy <- mean(y_true == y_pred)
  return(Accuracy)
}

ConfusionMatrix <- function(y_pred, y_true) {
  Confusion_Mat <- table(y_true, y_pred)
  return(Confusion_Mat)
}

Precision <- function(y_true, y_pred, positive = NULL) {
  Confusion_DF <- transform(as.data.frame(ConfusionMatrix(y_pred, y_true)),
                            y_true = as.character(y_true),
                            y_pred = as.character(y_pred),
                            Freq = as.integer(Freq))
  if (is.null(positive) == TRUE) positive <- as.character(Confusion_DF[1,1])
  TP <- as.integer(subset(Confusion_DF, y_true==positive & y_pred==positive)["Freq"])
  FP <- as.integer(sum(subset(Confusion_DF, y_true!=positive & y_pred==positive)["Freq"]))
  Precision <- TP/(TP+FP)
  return(Precision)
}

Recall <- function(y_true, y_pred, positive = NULL) {
  Confusion_DF <- transform(as.data.frame(ConfusionMatrix(y_pred, y_true)),
                            y_true = as.character(y_true),
                            y_pred = as.character(y_pred),
                            Freq = as.integer(Freq))
  if (is.null(positive) == TRUE) positive <- as.character(Confusion_DF[1,1])
  TP <- as.integer(subset(Confusion_DF, y_true==positive & y_pred==positive)["Freq"])
  FN <- as.integer(sum(subset(Confusion_DF, y_true==positive & y_pred!=positive)["Freq"]))
  Recall <- TP/(TP+FN)
  return(Recall)
}

F1_Score <- function(y_true, y_pred, positive = NULL) {
  Confusion_DF <- transform(as.data.frame(ConfusionMatrix(y_pred, y_true)),
                            y_true = as.character(y_true),
                            y_pred = as.character(y_pred),
                            Freq = as.integer(Freq))
  if (is.null(positive) == TRUE) positive <- as.character(Confusion_DF[1,1])
  Precision <- Precision(y_true, y_pred, positive)
  Recall <- Recall(y_true, y_pred, positive)
  F1_Score <- 2 * (Precision * Recall) / (Precision + Recall)
  return(F1_Score)
}


#Keep the last column as it was, can't use ncol-1 in the for loop above for some reason
data_prep[ncol(data_prep)] <- data_raw[ncol(data_prep)]
data_prep <- as_tibble(data_prep)
names(data_prep)[ncol(data_prep)] <- 'true_label'

#Get the conditional probabilities from the previously defined tables for h0 and h1
get_probs <- function(data_train, data_test){
  #Table for hypothesis h0
  data_h0 <- data_train%>%
    filter(data_train[ncol(data_train)] == 0)
  
  #Table for hypothesis h1
  data_h1 <- data_train%>%
    filter(data_train[ncol(data_train)] == 1)
  
  ph0 <- nrow(data_h0)/nrow(data_train)
  ph1 <- nrow(data_h1)/nrow(data_train)
  
  test_true_lab <- data_test[,ncol(data_test)]
  data_test <- data_test[1:ncol(data_test)-1]
  probtab0 <- data_test #A probability matrix with the same dims as the basic df
  
  for(i in 1:ncol(data_test)){
    colm <- data_h0[i]
    names(colm)[1] <- "cname" #Only way to access the colmumn by name
    colm[(nrow(colm))+1,]<- 'High' #Laplacian correction
    colm[(nrow(colm))+1,]<- 'Medium'
    colm[(nrow(colm))+1,]<- 'Low'
    
    colm <- colm%>%
      group_by(cname)%>%
      tally()%>%
      mutate(prob = n/sum(n)) #Calculate the probability of each label
    
    names(colm)[1] <- names(data_h0)[i] #Rename the colmumn 
    probtemp <- data_test%>% #Get the probability of each item in the ith colmumn
      inner_join(colm)%>%
      select(prob)
    
    probtab0[i] <- probtemp #Matrix with conditional prob of each value in each column
  }
  h0 <- as.data.frame(rowProds(as.matrix(probtab0)))*ph0 #Aggregate conditional probability per item
  
  #Now for h1
  probtab1 <- data_test #A probability matrix with the same dims as the basic df
  
  for(i in 1:ncol(data_test)){
    colm <- data_h1[i]
    names(colm)[1] <- "cname" #Only way to access the colmumn by name
    colm[(nrow(colm))+1,]<- 'High' #Laplacian correction
    colm[(nrow(colm))+1,]<- 'Medium'
    colm[(nrow(colm))+1,]<- 'Low'
    
    colm <- colm%>%
      group_by(cname)%>%
      tally()%>%
      mutate(prob = n/sum(n)) #Calculate the probability of each label
    
    names(colm)[1] <- names(data_h1)[i] #Rename the colmumn 
    probtemp <- data_test%>% #Get the probability of each item in the ith colmumn
      inner_join(colm)%>%
      select(prob)
    
    probtab1[i] <- probtemp #Matrix with conditional prob of each value in each column
  }
  h1 <- as.data.frame(rowProds(as.matrix(probtab1)))*ph1 #Aggregate conditional probability per item
  
  #Some acrobatics bc dataframes are painful sometimes
  data_test$h0 <- h0
  data_test$h1 <- h1
  
  results <- data_test%>% #Make results its own df for the next part
    select(h0, h1)
  
  #Get the colname of the max value between h0 and h1
  results$maxval <- colnames(results)[max.col(results,ties.method="first")] 
  results[results=='h0'] <- 0
  results[results=='h1'] <- 1
  
  results <- results%>% #Convert to integer or the eval functions complain
    select(maxval)%>%
    mutate_if(is.character, as.integer)
  
  data_test$predicted_label <- results$maxval #Append the predicted results to the base df
  data_test$true_label <- test_true_lab$true_label
  acc = Accuracy(data_test$predicted_label, data_test$true_label)
  prec = Precision(data_test$predicted_label, data_test$true_label)
  rec = Recall(data_test$predicted_label, data_test$true_label)
  f1 = F1_Score(data_test$predicted_label, data_test$true_label)
  eval_metrix <- list("acc" = acc, "prec" = prec, 
                      "rec" = rec, "f1" = f1)
  return(eval_metrix)
}

#Create 10 folds
folds <- cut(seq(1,nrow(data_prep)),breaks=10,labels=FALSE) #Assigns a fold to the data
acc = 0
prec = 0
rec = 0
f1 = 0

for(i in 1:10){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  data_test<- data_prep[testIndexes, ]
  data_train <- data_prep[-testIndexes, ]
  nb <- get_probs(data_train, data_test)
  acc = acc + nb$acc
  prec = prec + nb$prec
  rec = rec + nb$rec
  f1 = f1 + nb$f1
}

cat('========== RESULTS ==============
    \tAccuracy -->', acc/10,
    '\n\tPrecision -->', prec/10,
    '\n\tRecall -->', rec/10,
    '\n\tF1 Score -->', f1/10)

