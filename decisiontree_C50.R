setwd("D:\\FIXHASIL\\FIXHASIL\\")

install.packages("twitteR")
install.packages("RCurl")
install.packages("plyr")
install.packages("dplyr")
install.packages("stringr")
install.packages("ggplot2")
install.packages("httr") 
install.packages("tm")
install.packages("NLP")
install.packages("syuzhet")
install.packages("wordcloud")
install.packages("SentimentAnalysis")
install.packages("tm")
install.packages("C50")
install.packages("caret")
install.packages("e1071")

library(twitteR)
library(RCurl)
library(plyr)
library(dplyr)
library(stringr)
library(ggplot2)
library(httr) 
library(NLP)
library(syuzhet)
library(wordcloud)
library(SentimentAnalysis)
library(tm)
library(C50)
library(caret)
library(e1071)

api_key <- "NaLjvchWPagnwz9Lm6WOG4ayc"
api_secret <- "EL0nCxOfZmevpVlF5uC5rmVVZFzi5cqD6SVVs9rJQ60x6RS8a9"
access_toket <- "1107290465097768961-kNh9cVsM0fJ6o7V35IdV3fHlxbmSn3"
access_token_secret <- "ogq2yOMFVC21YxGyK4XbNStUS4LmGB6Q0EjlqRNGnWA67"
setup_twitter_oauth(api_key,api_secret,access_toket,access_token_secret)

adidas_tweets <- searchTwitter("adidas", n=22000, lang = "en", retryOnRateLimit = 22000)
#adidas_tweetss <- searchTwitter("adidas", n=200, lang = "en", retryOnRateLimit = 200)


#preprocessing
adidas_txt = sapply(adidas_tweets, function(x) x$getText())
  
adidas_txt1 = gsub("(RT|via)((?:\\b\\W*@\\w+)+)"," ",adidas_txt) #untuk hapus RT
adidas_txt2 = gsub("https[^[:blank:]]+"," ", adidas_txt1) #hapus URL (link)
adidas_txt3 = gsub("@\\w+"," ",adidas_txt2) #hapus nama pengguna
adidas_txt4 = gsub("[[:punct:]]", " ", adidas_txt3) #hapus tanda baca
adidas_txt5 = gsub("[^[:alnum:]]", " ", adidas_txt4) #hapus tanda
adidas_txt6 = gsub('\\d+', '', adidas_txt5) #menghapus angka

#mengubah semua kalimat menjadi huruf kecil
adidas = tolower (adidas_txt6)

write.csv(adidas_tweets,"adidas_tweets.csv")

#ekstraksi data
adidassentiment <- get_nrc_sentiment(adidas)
SentimentScoreadidas <- data.frame(colSums(adidassentiment[]))
SentimentScoreadidas
  
# positif & negatif - adidas
adidasclasification <- adidassentiment[,-c(1,2,3,4,5,6,7,8)]

#menghapus data yg bernilai netral
hapusIndex <- which(adidasclasification[,1]==adidasclasification[,2])
adidasclasification <- adidasclasification[-hapusIndex,]
adidasclass <- adidas[-hapusIndex]
View(adidasclass)

label_y_adidas <- NULL
for (i in 1:11182){
  if(adidasclasification[i,1]>adidasclasification[i,2]){
    label_y_adidas[i]<- "negatif"
  }else{
    label_y_adidas[i]<- "positif"
  }
}

adidasclasification$y <- label_y_adidas
adidasclasification$text <- adidasclass
adidasclasification <- adidasclasification[,-c(1,2)]

#membuat corpus
adidascorpus <- Corpus(VectorSource(adidasclasification$text))

corpus.clean <- adidascorpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind = "en")) %>%
  tm_map(stripWhitespace)
dtmadidas <- DocumentTermMatrix(corpus.clean)

inspect(dtmadidas[1:50, 10:15])

#80train
train_idx80 <- createDataPartition(adidasclasification$y, p=0.80, list=FALSE)

# set for the original raw data 
train1_80 <- adidasclasification[train_idx80,]
test1_80 <- adidasclasification[-train_idx80,]

dtm.train <- dtmadidas[train_idx80,]
dtm.test <- dtmadidas[-train_idx80,]

# set for the cleaned-up data
train2_80 <- corpus.clean[train_idx80]
test2_80 <- corpus.clean[-train_idx80]

dic2 <- findFreqTerms(dtm.train, 10)
length((dic2))

adidas_train80 <- DocumentTermMatrix(train2_80, list(dictionary=dic2))
adidas_test80 <- DocumentTermMatrix(test2_80, list(dictionary=dic2))

# this step further converts the DTM-shaped data into a categorical 
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
}

adidas_train80 <- apply(adidas_train80,2, convert_counts)
adidas_test80 <- apply(adidas_test80,2, convert_counts)

adidas_train80 <- as.data.frame(adidas_train80)
adidas_test80 <- as.data.frame(adidas_test80)

#str(adidas_train80)

adidas_train1_80 <- cbind(y=factor(train1_80$y), adidas_train80)
adidas_test1_80 <- cbind(y=factor(test1_80$y), adidas_test80)

adidas_train1_80<-as.data.frame(adidas_train1_80)
adidas_test1_80<-as.data.frame(adidas_test1_80)


modelc50 = C5.0(y~., adidas_train1_80)
prediksi <- predict(modelc50,adidas_test1_80)
confMatrix1_80 <- confusionMatrix(prediksi,adidas_test1_80$y)
plot(modelc50)
summary(modelc50)

modelboost80=C5.0(y~., adidas_train1_80, trials= 3)
prediksiboost80 <- predict(modelboost80,adidas_test1_80)
confMatrixboost_80 <- confusionMatrix(prediksiboost80,adidas_test1_80$y)
plot(modelboost80)
summary(modelboost80)
table(adidas_train1_80$y,adidas_train1_80$limited)

#70train
train_idx70 <- createDataPartition(adidasclasification$y, p=0.7, list=FALSE)

# set for the original raw data 
train1_70 <- adidasclasification[train_idx70,]
test1_70 <- adidasclasification[-train_idx70,]

dtm.train <- dtmadidas[train_idx70,]
dtm.test <- dtmadidas[-train_idx70,]

# set for the cleaned-up data
train2_70 <- corpus.clean[train_idx70]
test2_70 <- corpus.clean[-train_idx70]

dic2 <- findFreqTerms(dtm.train, 10)
length((dic2))

adidas_train70 <- DocumentTermMatrix(train2_70, list(dictionary=dic2))
adidas_test70 <- DocumentTermMatrix(test2_70, list(dictionary=dic2))

# this step further converts the DTM-shaped data into a categorical 
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
}

adidas_train70 <- apply(adidas_train70,2, convert_counts)
adidas_test70 <- apply(adidas_test70,2, convert_counts)

adidas_train70 <- as.data.frame(adidas_train70)
adidas_test70 <- as.data.frame(adidas_test70)

#str(adidas_train70)

adidas_train1_70 <- cbind(y=factor(train1_70$y), adidas_train70)
adidas_test1_70 <- cbind(y=factor(test1_70$y), adidas_test70)

adidas_train1_70<-as.data.frame(adidas_train1_70)
adidas_test1_70<-as.data.frame(adidas_test1_70)

modell_C50_70 = C5.0(y~., adidas_train1_70)
prediksi_70 <- predict(modell_C50_70,adidas_test1_70)
confMatrixx1_70 <- confusionMatrix(prediksi_70,adidas_test1_70$y)
plot(modell_C50_70)
summary(modell_C50_70)

modelboost70=C5.0(y~., adidas_train1_70, trials= 3)
prediksiboost70<- predict(modelboost70,adidas_test1_70)
confMatrixboost_70 <- confusionMatrix(prediksiboost70,adidas_test1_70$y)
plot(modelboost70)
summary(modelboost70)

##########################################################################################
