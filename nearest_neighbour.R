library(magrittr)
library(tidyverse)
library(reshape2)

#Read the dataset
path = 'Project_3/Project3_dataset1.txt'
k = 10

data_raw<- read_delim(path, delim = '\t', col_names = FALSE)

if(is.character(data_raw$X5) == TRUE){
  words <- data_raw%>%
    distinct(X5)
  words$ind <- data.frame(seq(1:0))
  data_raw <- data_raw%>%
    inner_join(words)
  data_raw$X5 <- data_raw$ind$seq.1.0.
  data_raw <- data_raw%>%
    select(-ind)
}

Accuracy <- function(y_pred, y_true) {
  Accuracy <- mean(y_true == y_pred)
  return(Accuracy)
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
  Confusion_DF <- ConfusionDF(y_pred, y_true)
  if (is.null(positive) == TRUE) positive <- as.character(Confusion_DF[1,1])
  Precision <- Precision(y_true, y_pred, positive)
  Recall <- Recall(y_true, y_pred, positive)
  F1_Score <- 2 * (Precision * Recall) / (Precision + Recall)
  return(F1_Score)
}

#Get just the row number and the class
data_base <- data_raw %>%
  mutate(id = row_number())%>%
  rename(label =  colnames(data_raw[ncol(data_raw)]))%>%
  select(id, label)

#Create 10 folds
folds <- cut(seq(1,nrow(data_base)),breaks=10,labels=FALSE) #Assigns a fold to the data
acc = 0
prec = 0
rec = 0
f1 = 0

knn <- function(data_train, data_test,k){
  data_train <- data_train%>%
    rename(proposed_label = label)%>%
    rename(training_row = id)
  
  
  data_test <- data_test%>%
    rename(true_class = label)%>%
    rename(testing_row = id)
  
  
  #Get the distances of each row with each other row
  data_dist_org <- as.matrix(dist(data_raw))
  data_dist <- melt(data_dist_org)%>%
    filter(Var1 != Var2)%>%
    rename(training_row = Var1)%>%
    rename(testing_row = Var2)%>%
    rename(distance = value)
  
  #Join the training set with the distance table
  #Gives the distance of each row in the training set with every other row
  data_expand <- data_train%>%
    inner_join(data_dist)
  
  #Group by id to get the top n nearest neighbours for that point
  data_nn <- data_expand %>%
    inner_join(data_test)%>%
    group_by(testing_row) %>%
    top_n(-k, distance)  %>%
    arrange(testing_row)
  
  #Weigh the data based on distances
  data_weights <- data_nn%>%
    mutate(dist_w = 1/distance^2)
  
  #Classify the data by taking a majority vote
  data_classify <- data_weights  %>%
    group_by(testing_row, proposed_label) %>%
    summarize(votes = sum(dist_w)) %>% 
    top_n(1, votes)
  
  data_result <- data_classify%>%
    inner_join(data_test)%>%
    select(testing_row, proposed_label, true_class)
  
  #Evaluation
  acc <- Accuracy(data_result$proposed_label, data_result$true_class)
  prec <- Precision(data_result$proposed_label, data_result$true_class)
  rec <- Recall(data_result$proposed_label, data_result$true_class)
  f1 <- F1_Score(data_result$proposed_label, data_result$true_class)
  eval_metrix <- list("acc" = acc, "prec" = prec, 
                      "rec" = rec, "f1" = f1)
  return(eval_metrix)
}

for(i in 1:10){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  data_test<- data_base[testIndexes, ]
  data_train <- data_base[-testIndexes, ]
  nn <- knn(data_train, data_test, k)
  acc = acc + nn$acc
  prec = prec + nn$prec
  rec = rec + nn$rec
  f1 = f1 + nn$f1
  
}
cat('========== RESULTS ==============
    \tAccuracy -->', acc/10,
    '\n\tPrecision -->', prec/10,
    '\n\tRecall -->', rec/10,
    '\n\tF1 Score -->', f1/10)

