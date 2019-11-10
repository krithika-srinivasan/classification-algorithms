library(matrixStats)
library(tidyverse)
library(MLmetrics)
#Input the data
data_raw <- read.csv('data/Project3_dataset2.txt', sep = '\t', header = FALSE)

data_prep <- data_raw

#Group the non-character data columns
for(i in 1:ncol(data_raw)){
  if(is.numeric(data_raw[,i]) == TRUE){
    data_prep[,i] <- cut((data_raw[,i]), breaks = 3, labels = c("Low", "Medium", "High"))
  }
}

#Keep the last column as it was
data_prep[ncol(data_prep)] <- data_raw[ncol(data_prep)]
data_prep <- as_tibble(data_prep)

#Table for hypothesis h0
data_h0 <- data_prep%>%
  filter(data_prep[ncol(data_prep)] == 0)

#Table for hypothesis h1
data_h1 <- data_prep%>%
  filter(data_prep[ncol(data_prep)] == 1)

ph0 <- nrow(data_h0)/nrow(data_raw)
ph1 <- nrow(data_h1)/nrow(data_raw)

#Table with all the distinct combination
data_base <- data_prep[1:ncol(data_prep)-1]
data_base<- data_base%>%
  distinct()

#Get the h0 values
get_probs <- function(data_h){
  probtab <- data_base
  
  for(i in 1:ncol(data_base)){
    col <- data_h[i]
    names(col)[1] <- "cname"
    col[(nrow(col))+1,]<- 'High'
    col[(nrow(col))+1,]<- 'Medium'
    col[(nrow(col))+1,]<- 'Low'
    
    col <- col%>%
      count(cname, sort = TRUE)%>%
      mutate(prob = n/sum(n))
    
    names(col)[1] <- names(data_h)[i]
    probtemp <- data_base%>%
      inner_join(col)%>%
      select(prob)
    
    probtab[i] <- probtemp #Matrix with conditional prob of each value
  }
  
  h0 <- as.data.frame(rowProds(as.matrix(probtab)))
  return(h0)
}

h0 <- as_tibble(get_probs(data_h0)*ph0)
h1 <- as_tibble(get_probs(data_h1)*ph1)
names(h0)[1] <- 'h0'
names(h1)[1] <- 'h1'

data_base$h0 <- h0
data_base$h1 <- h1

results <- data_base%>%
  select(h0, h1)

results$maxval <- colnames(results)[max.col(results,ties.method="first")]
results[results=='h0'] <- 0
results[results=='h1'] <- 1

results <- results%>%
  select(maxval)%>%
  mutate_if(is.character, as.integer)
data_base$predicted_label <- results
data_base <- data_base%>%
  inner_join(data_prep)
names(data_base)[ncol(data_base)] <- 'true_label'

#Create 10 folds
folds <- cut(seq(1,nrow(data_base)),breaks=10,labels=FALSE) #Assigns a fold to the data
acc = 0
prec = 0
rec = 0
f1 = 0

for(i in 1:10){
  testIndexes <- which(folds==i,arr.ind=TRUE)
  data_test<- data_base[testIndexes, ]
  
  acc = acc + Accuracy(data_test$predicted_label, data_test$true_label)
  prec = prec + Precision(data_test$predicted_label$maxval, data_test$true_label)
  rec = rec + Recall(data_test$predicted_label$maxval, data_test$true_label)
  f1 = f1 + F1_Score(data_test$predicted_label$maxval, data_test$true_label)
}

cat('========== RESULTS ==============
    \tAccuracy -->', acc/10,
    '\n\tPrecision -->', prec/10,
    '\n\tRecall -->', rec/10,
    '\n\tF1 Score -->', f1/10)
