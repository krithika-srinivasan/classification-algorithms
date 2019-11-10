library(magrittr)
library(tidyverse)
library(reshape2)
library(MLmetrics)

#Read the dataset
path = 'data/Project3_dataset2.txt'
k = 3

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
    rename(testing_row = id)
  
  
  data_test <- data_test%>%
    rename(true_class = label)%>%
    rename(training_row = id)
  
  
  #Get the distances of each row with each other row
  data_dist_org <- as.matrix(dist(data_raw))
  data_dist <- melt(data_dist_org)%>%
    filter(Var1 != Var2)%>%
    rename(training_row = Var1)%>%
    rename(testing_row = Var2)%>%
    rename(distance = value)
  
  #Join the training set with the distance table
  #Gives the distance of each row in the training set with every other row
  data_expand <- data_test%>%
    inner_join(data_dist)
  
  #Group by id to get the top n nearest neighbours for that point
  data_nn <- data_expand %>%
    group_by(training_row) %>%
    top_n(-k, distance)  %>%
    arrange(training_row)
  
  #Weigh the data based on distances
  data_weights <- data_nn%>%
    inner_join(data_train)%>%
    mutate(dist_w = 1/distance^2)
  
  #Classify the data by taking a majority vote
  data_classify <- data_weights  %>%
    group_by(training_row, proposed_label) %>%
    summarize(votes = sum(dist_w)) %>% 
    top_n(1, votes)
  
  data_result <- data_classify%>%
    inner_join(data_test)%>%
    select(training_row, proposed_label, true_class)
  
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

