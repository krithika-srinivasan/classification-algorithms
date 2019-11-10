library(matrixStats)
library(tidyverse)
library(MLmetrics)
#Input the data
data_raw <- read.csv('data/Project3_dataset1.txt', sep = '\t', header = FALSE)

data_prep <- data_raw

#Group the non-character data columns into 3 levels by quantile
for(i in 1:ncol(data_raw)){
  if(is.numeric(data_raw[,i]) == TRUE){
    data_prep[,i] <- cut((data_raw[,i]), breaks = 3, labels = c("Low", "Medium", "High"))
  }
}

#Keep the last column as it was, can't use ncol-1 in the for loop above for some reason
data_prep[ncol(data_prep)] <- data_raw[ncol(data_prep)]
data_prep <- as_tibble(data_prep)

#Table for hypothesis h0
data_h0 <- data_prep%>%
  filter(data_prep[ncol(data_prep)] == 0)

#Table for hypothesis h1
data_h1 <- data_prep%>%
  filter(data_prep[ncol(data_prep)] == 1)

#Prior probabilities
ph0 <- nrow(data_h0)/nrow(data_raw)
ph1 <- nrow(data_h1)/nrow(data_raw)

#Table with all the distinct combinations. Easier to work with unique rows
data_base <- data_prep[1:ncol(data_prep)-1]
data_base<- data_base%>%
  distinct()

#Get the conditional probabilities from the previously defined tables for h0 and h1
get_probs <- function(data_h){
  probtab <- data_base #A probability matrix with the same dims as the basic df
  
  for(i in 1:ncol(data_base)){
    col <- data_h[i]
    names(col)[1] <- "cname" #Only way to access the column by name
    col[(nrow(col))+1,]<- 'High' #Laplacian correction
    col[(nrow(col))+1,]<- 'Medium'
    col[(nrow(col))+1,]<- 'Low'
    
    col <- col%>%
      count(cname, sort = TRUE)%>%
      mutate(prob = n/sum(n)) #Calculate the probability of each label
    
    names(col)[1] <- names(data_h)[i] #Rename the column 
    probtemp <- data_base%>% #Get the probability of each item in the ith column
      inner_join(col)%>%
      select(prob)
    
    probtab[i] <- probtemp #Matrix with conditional prob of each value in each column
  }
  
  h0 <- as.data.frame(rowProds(as.matrix(probtab))) #Aggregate conditional probability per item
  return(h0)
}

h0 <- as_tibble(get_probs(data_h0)*ph0) #Multiply with the prior probability
h1 <- as_tibble(get_probs(data_h1)*ph1)
names(h0)[1] <- 'h0' #Renaming so that the colnames are less ugly
names(h1)[1] <- 'h1'

#Some acrobatics bc dataframes are painful sometimes
data_base$h0 <- h0
data_base$h1 <- h1

results <- data_base%>% #Make results its own df for the next part
  select(h0, h1)

#Get the colname of the max value between h0 and h1
results$maxval <- colnames(results)[max.col(results,ties.method="first")] 
results[results=='h0'] <- 0
results[results=='h1'] <- 1

results <- results%>% #Convert to integer or the eval functions complain
  select(maxval)%>%
  mutate_if(is.character, as.integer)

data_base$predicted_label <- results #Append the predicted results to the base df
data_base <- data_base%>% #Rejoin the base df so it contains all vals not just unique ones
  inner_join(data_prep)
names(data_base)[ncol(data_base)] <- 'true_label' #Rename the last column for eval fns

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
