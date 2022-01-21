#my dataset consists of mails coming in my mailbox for two weeks. The dataset includes msg content and using naive bayes classify the mails
#spam or not .The dataset will include mailcontent encoded and output which will predict about the mail being a spam
# The dataset will contain: 
#SENDER name and email adress,  
#RECEIVER name and email address , 
#DATE on which it was sent and received, 
#SUBJECT 
#EMAIL CONTENT which will consist of mail body
library(base)
library(utils)
library(naniar)
library(caTools)
library(ggplot2)
library(SnowballC)
library(tm)
library(textstem)
df <- read.csv("https://raw.githubusercontent.com/anthoniraj/dsr_datasets_final/main/19BDS0111.csv")


df$Date.Sent<-strptime(df$Date.Sent,"%B %d,%Y, %H:%M:%S")
df$Date.Sent
df$Date.Received<-strptime(df$Date.Received,"%B %d,%Y, %H:%M:%S")
df$Date.Received
df<-df[c(9,10)]
str(df)
miss_var_summary(df)

post = Corpus(VectorSource(df$Email.Text)) # creating corpus
post  # printing corpus

post = tm_map(post, removeNumbers)

writeLines(as.character(post[[1]]))     #checking the removenumber function
post = tm_map(post, removePunctuation)  #remove punctuation
post = tm_map(post, stripWhitespace)    #stripping extra white space

writeLines(as.character(post[[1]]))

post <- tm_map(post,content_transformer(tolower)) #converting all characters to lowercase
post <- tm_map(post, removeWords, stopwords("english"))   #removing stopwords

writeLines(as.character(post[[1]]))
writeLines(as.character(post[[2]]))
###REMOVING UNREADABLE SYMBOLS####
#gsub("[^A-Za-z0-9 ]","",post[[2]])
for(i in 1:293){
post[[i]]=gsub("[^A-Za-z0-9 ]","",post[[i]])
}
writeLines(as.character(post[[2]]))
post = tm_map(post, stripWhitespace)
writeLines(as.character(post[[2]]))


#STEMMING
post <- tm_map(post,textstem::lemmatize_strings)

writeLines(as.character(post[[3]]))

inspect(post[1:3])

##SPARSE MATRIX


dtm = DocumentTermMatrix(post)
str(dtm)


##TRAINING SET AND TEST SET 

# split the raw data:
df.train = df[1:196, ] # about 75%
df.test  = df[197:293, ] # the rest

# then split the document-term matrix
dtm.train = dtm[1:196, ]
dtm.test  = dtm[197:293, ]

# and finally the corpus
corpus.train = post[1:196]
corpus.test  = post[197:293]

round(prop.table(table(df.train$spam))*100)
round(prop.table(table(df.test$spam))*100)

#REMOVING WORDS WHICH APPEAR LESS THAN IN 5 MAILS
freq_terms = findFreqTerms(dtm.train, 5)
reduced_dtm.train = DocumentTermMatrix(corpus.train, list(dictionary=freq_terms))
reduced_dtm.test =  DocumentTermMatrix(corpus.test, list(dictionary=freq_terms))
ncol(reduced_dtm.train)
ncol(reduced_dtm.test)
# Naive Bayes classifier is typically trained on data with categorical features.
# To make up for this, we will define a function that changes the count of words
# into factor variables that simply indicates "yes" or "no" depending on 
# whether the word appears at all.
convert_counts = function(x) {
  x = ifelse(x > 0, 1, 0)
  x = factor(x, levels = c(0, 1), labels=c("No", "Yes"))
  return (x)
}
reduced_dtm.test
reduced_dtm.train = apply(reduced_dtm.train, MARGIN=2, convert_counts)
reduced_dtm.test  = apply(reduced_dtm.test, MARGIN=2, convert_counts)
reduced_dtm.train

###USING PREDEFINED MODEL###
#install.packages("e1071")
library(e1071)
# store our model in email_classifier
email_classifier = naiveBayes(reduced_dtm.train, df.train$spam)
email_test.predicted = predict(email_classifier,
                             reduced_dtm.test)
library(gmodels)
CrossTable(email_test.predicted,
           df.test$spam,
           prop.chisq = FALSE, # as before
           prop.t     = FALSE, # eliminate cell proportions
           dnn        = c("predicted", "actual")) # relabels rows+cols              

#our model is detecting ham and spam 96.9% times  and 75% times correctly respectively


observed_data <-df.test$spam 
predicted_data <- email_test.predicted
#observed_data == predicted_data
mean(observed_data == predicted_data)  # The classifier accuracy is 89.6%


#SELF DEFINED EVALUATION FUNCTION
cm<-table(observed_data,predicted_data)
cm
cm[1,1]
cm[1,2]
multi_class_rates<-function(confusion_matrix){
  true_positive<-diag(confusion_matrix)
  false_positive<-colSums(confusion_matrix)-true_positive
  false_negatives<-rowSums(confusion_matrix)-true_positive
  true_negatives<-sum(confusion_matrix)-true_positive - false_positive - false_negatives
  return(data.frame(true_positive,false_positive,true_negatives,false_negatives,row.names = c("ham","spam")))
}
ndf<-multi_class_rates(cm)
ndf

#precision
precision<-ndf$true_positive/(ndf$true_positive+ndf$false_positive)
precision                                                                 #the precision is 96.9% and 75%

#recall
recall<-ndf$true_positive/(ndf$true_positive+ndf$false_negatives)
recall                                                                  #the recall is 88.7% and 92.3%

#F1 measure
f1_measure<-(2*precision*recall)/(precision+recall)
f1_measure                                                               #f1_measure is 92.6% and 82.7%

#accuracy
accuracy=(ndf$true_positive+ndf$true_negatives)/(ndf$true_positive+ndf$false_positive+ndf$true_negatives+ndf$false_negatives)
accuracy                                                                 #accuracy is 89.6% 

#RESULT
result<-data.frame(precision=precision,recall=recall,f1_measure=f1_measure,accuracy=accuracy,row.names = c("spam","ham"))
result


##Bargraph
str(df.train)
barplot(table(df.train$spam))

#HEAT MAP FOR CONFUSION MATRIX
observed <- factor(c(0, 0, 1, 1))
predicted <- factor(c(0, 1, 0, 1))
y<-c(cm[1,1],cm[1,2],cm[2,1],cm[2,2])
pf <- data.frame(observed, predicted, y)
pf

ggplot(data =  pf, mapping = aes(x = observed, y = predicted)) +
  geom_tile(aes(fill = y), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f \n", y)), vjust = 1) +
  scale_fill_gradient(low = "#e2efef", high = "#009194") +
  theme_bw() + theme(legend.position = "none")
library(graphics)
