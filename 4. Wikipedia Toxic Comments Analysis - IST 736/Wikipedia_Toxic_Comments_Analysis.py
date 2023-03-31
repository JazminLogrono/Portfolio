#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 17:46:25 2022
IST 736: Analyzing Toxic Comments on Wikipedia
Xiangzhen He, Katherine Hurtado-Da Silva, Jazmin Logroño
"""
##### This python script references code provided by the professor of IST 736.

##### Load libraries
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  MultinomialNB

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score
##############################################################
##
##    Data preparation (can skip this part when using the toxic.csv file)
##
##############################################################
#the file path below needs to be updated when using by different users
#Data obtained from https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview
address = "train.csv"

toxicdf = pd.read_csv(address, sep=',')

#check the data
print(toxicdf.shape)
print(toxicdf[0:10])

#check if there are null values in data set
print(toxicdf.isnull().sum())

#Check the number of observations in each column.
print(toxicdf['toxic'].value_counts())
print(toxicdf['severe_toxic'].value_counts())
print(toxicdf['obscene'].value_counts())
print(toxicdf['threat'].value_counts())
print(toxicdf['insult'].value_counts())
print(toxicdf['identity_hate'].value_counts())

#Create a label column based on the number of toxicity label assigned to the comment
toxicdf['label']=toxicdf['toxic']+toxicdf['severe_toxic']+toxicdf['obscene']+toxicdf['threat']+toxicdf['insult']+toxicdf['identity_hate']

#Check the number of observations in each label class
print(toxicdf['label'].value_counts())

#create length variable that saves the length of each comment
length = toxicdf.comment_text.apply(len)

#check the statistics of length
length.describe()

#remove records with a length less or equal to 10
toxicdf['length'] = length
toxicdf_new = toxicdf[toxicdf.length > 10]

#check the observations in toxic column again
print(toxicdf_new['toxic'].value_counts())

#check again the distribution of label column
print(toxicdf_new['label'].value_counts())

#create a bar plot for class labels
import seaborn as sb
sb.set_style('whitegrid')
toxicdf_new['label'].value_counts().plot(kind='bar')
plt.xticks(rotation = 0)
xlocs=[i for i in range(0,7)]
for i, v in enumerate(toxicdf_new['label'].value_counts()):
    plt.text(xlocs[i] - 0.25, v + 0.1, str(v))
plt.plot()


##########Create the final dataset that only includes equal number of comments which have 0 or 1 label values
#Get subset of data for any comment that has no toxicity
toxicdf_no =toxicdf_new.loc[toxicdf_new['label']==0]

#random select 50% of the data
random.seed(11)
toxicdf_no = toxicdf_no.sample(n=6360)

#Get subset of data for any comment that has label value of 1
toxicdf_yes1 = toxicdf_new.loc[toxicdf_new['label']==1]

# merge two data frames
print('First 10 rows of the final data frame merging:')
finaldf = pd.concat([toxicdf_yes1, toxicdf_no], axis=0)
#check the first 10 rows
print(finaldf[0:10])

#check the dimensions of the final dataframe
print(finaldf.shape)

#Check the observations in each column in the new dataframe
print(finaldf['toxic'].value_counts())
print(finaldf['severe_toxic'].value_counts())
print(finaldf['obscene'].value_counts())
print(finaldf['threat'].value_counts())
print(finaldf['insult'].value_counts())
print(finaldf['identity_hate'].value_counts())

#save comment text and toxic labels to new variables
comments = finaldf['comment_text'].values
labels = finaldf['label'].values

#finaldf.to_csv("/Users/zhen/Documents/DS@Syracuse/Q4/IST 736/Project/finaldf.csv",
#               sep = ',', index = False)

#create a dataframe for the new data
final_toxicdf = pd.DataFrame(data = comments, columns = ['Comments'])
final_toxicdf.insert(loc = 0, column = 'Labels', value = labels)

final_toxicdf.to_csv("/Users/zhen/Documents/DS@Syracuse/Q4/IST 736/Project/toxic.csv",
               sep = ',', index = False)


##############################################################
##
##    Read Project Data Set (START FROM HERE WHEN USING toxic.csv FILE)
##
##############################################################

final_toxicdf = pd.read_csv("toxic.csv",
                      sep = ',')
print(final_toxicdf.head())

##############################################################
##
##    Data Cleaning
##
## (Additional feature reduction will be performed before doing clustering and classification analyses)
##
##############################################################

#Save Comments and Labels to separate variables
comments = final_toxicdf['Comments'].values
labels = final_toxicdf['Labels'].values

#Remove \n character
comments = [c.replace('\n', '') for c in comments]

#Remove urls starting with http
comments_new = [re.sub(r'http\S+', ' ', c) for c in comments]

#Remove urls starting with www
comments_new = [re.sub(r'www\S+', ' ', c) for c in comments_new]

#Check some toxic comments
print(comments_new[:10])

#Check some non-toxic tomments
print(comments_new[7500:7505])


##############################################################
##
##    Create visualizations
##
##############################################################

###Create a word cloud to review the most frequently appearing words
#Join all comments into one text document
commentstext = " ".join(comments_new)

#Create and generate a word cloud image
wordcloud = WordCloud(max_words=100, background_color="White",random_state=1, collocations=False).generate(commentstext)

# Display the generated image
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


###Toxicity Label Distribution
#set seaborn plotting aesthetics

import seaborn as sb
sb.set(rc={'figure.figsize':(11.7,8.27)})
sb.set(font_scale=2)
sb.set_style('darkgrid')
sb.set_palette('Set2')
Sentdist=sb.countplot(data=final_toxicdf, x='Labels')

plt.title('Comment Count by Toxicity Label')
plt.xlabel('Toxicity Label')
plt.ylabel('Number of Posted Comments')
sb.despine()
plt.show()

###Create a boxplot to show distribution of phrase length in each sentiment class
#create comment_len variable that saves the length of each headline
comment_len = [len(c) for c in comments_new]

sb.set_style('whitegrid')

#get list of tuples from two lists
comment_len_tuples = list(zip(comment_len,labels))
comment_len_DF = pd.DataFrame(comment_len_tuples, columns=['comment_len','label'])

#create a boxplot to show distribution of phrase length in each class
sb.boxplot(x='label',y='comment_len', data = comment_len_DF, palette='hls')


##############################################################
##
##    General Text preprocessing
##
##############################################################
## First create a stemmer and a lemmatizer that will be used when transform text data into vectors
# from nltk.stem.porter import PorterStemmer
# porter = PorterStemmer()

from nltk.stem import WordNetLemmatizer
Lemmer=WordNetLemmatizer()

# def MY_STEMMER(str_input):
#     words = re.sub(r"[^A-Za-z]", " ", str_input).lower().split()
#     words = [porter.stem(w) for w in words]
#     words = [w for w in words if len(w) > 2]
#     return words

def MY_LEMMER(str_input):
    words = re.sub(r"[^A-Za-z]", " ", str_input).lower().split()
    words = [Lemmer.lemmatize(w) for w in words]
    words = [w for w in words if len(w) > 2] #remove words that have less than 3 characters
    return words

## Lastly, define a function to be used inside vectorizer to limit the length of the words
def LONG_WORDS(str_input):
    words = re.sub(r"[^A-Za-z]", " ", str_input).lower().split()
    words = [w for w in words if len(w) > 2]
    return words


## Different models will be using different vectorizers.
## Therefore, vectorization processes will be performed under the section for each model.


##############################################################
##
##    Analysis - LDA Topic Modeling
##
##############################################################

###### Create Vectorizers for LDA
#Unigrams + Stopwords Removal CountVectorizer
UniCountVect=CountVectorizer(
                        stop_words='english',
                        min_df = 5,
                        tokenizer = LONG_WORDS
                        )

#Unigrams + Stopwords Removal + Lemmatization CountVectorizer
UniCountVect_Lemma=CountVectorizer(
                        stop_words='english',
                        tokenizer = MY_LEMMER,
                        min_df = 5
                        )

######  Using UniCountVect ######
comments = comments_new

X1=UniCountVect.fit_transform(comments)

# check the content of the DTM
print('\nDimensions of the DTM1: ', X1.shape)

# print out the first 10 items in the vocabulary
print('The first 10 items in the vocabulary for X1: ')
print(list(UniCountVect.vocabulary_.items())[:10])

# convert DTM to data frames with feature names
ColumnNames1=UniCountVect.get_feature_names()
DF1 = pd.DataFrame(X1.toarray(),columns=ColumnNames1)
print(DF1)

#save the dataframe to a csv file (commented out)
#DF1.to_csv('/Users/zhen/Documents/DS@Syracuse/Q4/IST 736/Project/Project_DF1.csv')

##### Using UniCountVect_Lemma
X2=UniCountVect_Lemma.fit_transform(comments)

# check the content of the DTM
print('\nDimensions of the DTM3: ', X2.shape)

# print out the first 10 items in the vocabulary
print('The first 10 items in the vocabulary for X2: ')
print(list(UniCountVect_Lemma.vocabulary_.items())[:10])

# convert DTM to data frames with feature names
ColumnNames2=UniCountVect_Lemma.get_feature_names()
DF2 = pd.DataFrame(X2.toarray(),columns=ColumnNames2)
print(DF2)

#save the dataframe to a csv file (commented out)
#DF2.to_csv('/Users/zhen/Documents/DS@Syracuse/Q4/IST 736/Project/Project_DF2.csv')


####################################
##
## Building LDA Models
##
####################################
#Reference: https://alvinntnu.github.io/NTNU_ENC2045_LECTURES/nlp/topic-modeling-naive.html
from sklearn.decomposition import LatentDirichletAllocation

################Using DF Without Lemmatization
### Search for the topic number
### Code below could be skipped due to the time needed for parameter tuning process
from sklearn.model_selection import GridSearchCV

# Options to try for the LDA
search_params = {'n_components': range(3,8),'learning_decay': [.6,.7,.8]}

# Set up LDA with the options keep static, using 'online' method for large data set
model = LatentDirichletAllocation(learning_method='online',
                                  random_state=0)

# Try all of the options
gridsearch = GridSearchCV(model,
                          param_grid=search_params,
                          n_jobs=-1,
                          verbose=1)
gridsearch.fit(DF1)

## Save the best model
best_lda1 = gridsearch.best_estimator_

print("Best Model's Params: ", gridsearch.best_params_)
print("Best Log Likelihood Score: ", gridsearch.best_score_)
print('Best Model Perplexity: ', best_lda1.perplexity(DF1))

# Visualize the results
cv_results_df = pd.DataFrame(gridsearch.cv_results_)
print(cv_results_df)

import seaborn as sns
sns.set(rc={"figure.dpi":150, 'savefig.dpi':150})
sns.pointplot(x="param_n_components",
              y="mean_test_score",
              data=cv_results_df)

### Code above could be skipped due to the time needed for parameter tuning process

# Apply the best topic number found
num_topics = 3
lda_model1 = LatentDirichletAllocation(n_components=num_topics, random_state=0,
                                     learning_method='online', learning_decay= 0.6)
LDA_DH_Model1 = lda_model1.fit_transform(DF1)

#check the shape of the Topic-Document Matrix
print("SIZE of TDM Based on DF1: ", LDA_DH_Model1.shape)

## Check model performance
# log-likelihood
print("\nLog-likelihood: ", lda_model1.score(DF1))
# perplexity
print("\nPerplexity: ", lda_model1.perplexity(DF1))


## Check documents by topic
doc_topic_df1 = pd.DataFrame(LDA_DH_Model1, columns=['T0', 'T1', 'T2'])
print(doc_topic_df1[:10])

## The probability of words over vocabulary
topic_word_matrix1 = lda_model1.components_
topic_word_df1 = pd.DataFrame(topic_word_matrix1, columns=ColumnNames1)
print(topic_word_df1)

#save the dataframe to a csv file (code was commented out)
#topic_word_df1.to_csv('/Users/zhen/Documents/DS@Syracuse/Q4/IST 736/Project/Project_Topic_Word_DF1.csv')

## Get top N words in each topic
## implement a print function
## REF: https://nlpforhackers.io/topic-modeling/
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

## Print LDA using print function from above
print("LDA Model:")
print_topics(lda_model1, UniCountVect)

################ Visulization for the first LDA Model

word_topic = np.array(lda_model1.components_)
#print(word_topic)
word_topic = word_topic.transpose()

num_top_words = 20
vocab_array = np.asarray(ColumnNames1)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 12
for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

#plt.tight_layout()
#plt.show()
plt.savefig("TopicsVis_DF1.pdf")

################ Another vis for LDA to plot the relations between topics
import pyLDAvis.sklearn as LDAvis
import pyLDAvis
pyLDAvis.enable_notebook() ## not using notebook
panel = LDAvis.prepare(lda_model1, X1, UniCountVect,  mds='tsne')
pyLDAvis.save_html(panel, "InterTopicMap_DF1.html")

############ Additional evaluation
## create a list that saves the LDA topics
lda_topics1 = []

for i in range(0,LDA_DH_Model1.shape[0]):
    topic_index = np.argmax(LDA_DH_Model1[i])
    lda_topics1.append(str(topic_index))

values, topic_counts1 = np.unique(lda_topics1, return_counts=True)

print(lda_topics1)
print("Number of records in each topic: ", topic_counts1)

###########################
############ Using DF with Lemmatization

### Code below could be skipped due to the time needed for parameter tuning process
### Search for the topic number
from sklearn.model_selection import GridSearchCV

# Options to try for the LDA
search_params = {'n_components': range(3,8),'learning_decay': [.6,.7,.8]}

# Set up LDA with the options keep static, using 'online' method for large data set
model = LatentDirichletAllocation(learning_method='online',
                                  random_state=0)

# Try all of the options
gridsearch = GridSearchCV(model,
                          param_grid=search_params,
                          n_jobs=-1,
                          verbose=1)
gridsearch.fit(DF2)

## Save the best model
best_lda2 = gridsearch.best_estimator_

print("Best Model's Params: ", gridsearch.best_params_)
print("Best Log Likelihood Score: ", gridsearch.best_score_)
print('Best Model Perplexity: ', best_lda2.perplexity(DF2))

# Visualize the results
cv_results_df2 = pd.DataFrame(gridsearch.cv_results_)

import seaborn as sns
sns.set(rc={"figure.dpi":150, 'savefig.dpi':150})
sns.pointplot(x="param_n_components",
              y="mean_test_score",
              data=cv_results_df2)
### Code above could be skipped due to the time needed for parameter tuning process

# Apply the best topic number found
num_topics = 3
lda_model2 = LatentDirichletAllocation(n_components=num_topics, random_state=0,
                                       learning_method = 'online', learning_decay= 0.8)
LDA_DH_Model2 = lda_model2.fit_transform(DF2)

#check the shape of the Topic-Document Matrix
print("SIZE of TDM Based on DF2: ", LDA_DH_Model2.shape)

## Check model performance
# log-likelihood
print("\nLog-likelihood: ", lda_model2.score(DF2))
# perplexity
print("\nPerplexity: ", lda_model2.perplexity(DF2))


## Check documents by topic
doc_topic_df2 = pd.DataFrame(LDA_DH_Model2, columns=['T0', 'T1', 'T2'])
print(doc_topic_df2[:10])

## The probability of words over vocabulary
topic_word_matrix2 = lda_model2.components_
topic_word_df2 = pd.DataFrame(topic_word_matrix2, columns=ColumnNames2)
print(topic_word_df2)

#save the dataframe to a csv file (code was commented out)
#topic_word_df2.to_csv('/Users/zhen/Documents/DS@Syracuse/Q4/IST 736/Project/Project_Topic_Word_DF2.csv')

## Get top N words in each topic
## implement a print function
## REF: https://nlpforhackers.io/topic-modeling/
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

## Print LDA using print function from above
print("LDA Model:")
print_topics(lda_model2, UniCountVect_Lemma)

################ Visulization for the second LDA model using lemmatized words

word_topic2 = np.array(lda_model2.components_)
#print(word_topic2)
word_topic2 = word_topic2.transpose()

num_top_words = 20
vocab_array = np.asarray(ColumnNames2)

#fontsize_base = 70 / np.max(word_topic2) # font size for word with largest share in corpus
fontsize_base = 12

colors = ["red", "orange", "blue"]

for t in range(num_topics):
    plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic2[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic2[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base, color = colors[t])
                 ##fontsize_base*share)

#plt.tight_layout()
#plt.show()
plt.savefig("TopicsVis2.pdf")

################ Another vis for LDA to plot the relations between topics
import pyLDAvis.sklearn as LDAvis
import pyLDAvis
pyLDAvis.enable_notebook() ## not using notebook
panel = LDAvis.prepare(lda_model2, X2, UniCountVect_Lemma,  mds='tsne')
pyLDAvis.save_html(panel, "InterTopicMap_DF2.html")

## create a list that saves the LDA topic labels
lda_labels = []

for i in range(0,LDA_DH_Model2.shape[0]):
    topic_index = np.argmax(LDA_DH_Model2[i])
    lda_labels.append(str(topic_index))

values, label_counts = np.unique(lda_labels, return_counts=True)

print(lda_labels)
print("Number of records in each topic: ", label_counts)



##############################################################
##
##    Analysis - Naive Bayes
##
##############################################################

## Performed additional feature reduction
#first review of features prompted these removals
comments=[word.replace('wikipedia','') for word in comments]
comments=[word.replace('wiki','') for word in comments]
comments=[word.replace('article','') for word in comments]
#second review of features prompted these removals
comments=[word.replace('like','') for word in comments]
comments=[word.replace('talk','') for word in comments]
comments=[word.replace('page','') for word in comments]
comments=[word.replace('just','') for word in comments]
comments=[word.replace('know','') for word in comments]
comments=[word.replace('did','') for word in comments]
#third review of features prompted these removals
comments=[word.replace('want','') for word in comments]
comments=[word.replace('use','') for word in comments]
comments=[word.replace('does','') for word in comments]
comments=[word.replace('dont','') for word in comments]
comments=[word.replace('don','') for word in comments]
comments=[word.replace('make','') for word in comments]
comments=[word.replace('beca','') for word in comments]

#### Split before continuining with vectorizer cleaning steps
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(comments,labels,random_state=0,stratify=labels,test_size=.3,shuffle=True)

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))

############
#Create vectorizers to be used
def LONG_WORDS(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [w for w in words if len(w) > 2]
    return words

LEMMER = WordNetLemmatizer()
def MY_LEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [LEMMER.lemmatize(word) for word in words]
    words=[word for word in words if len(word)>2]
    return words

### Uncomment two lines of code below when try to run models without the unhelpful feature words listed above
#stopwords = nltk.corpus.stopwords.words('english')
#stopwords.extend(unhelpfulFeatures)


#Unigrams + Stopwords CountVectorizer
CV1=CountVectorizer(input='content',analyzer = 'word',stop_words='english',min_df=5,tokenizer=LONG_WORDS)

#ngram_range of c(1, 1) means only unigrams
#c(1, 2) means unigrams and bigrams
#and c(2, 2) means only bigrams.

#Stopwords unigrams & bigram CountVectorizer
CV2=CountVectorizer(input='content',analyzer = 'word',stop_words='english',ngram_range=(1,2),min_df=5,tokenizer=LONG_WORDS)

#Unigrams + Stopwords TF-IDF
CV3=TfidfVectorizer(input='content',analyzer = 'word',stop_words='english',min_df=5,tokenizer=LONG_WORDS)

#Stopwords unigrams + bigram TF-IDF #bigrams only had poor results
CV4=TfidfVectorizer(input='content', analyzer = 'word',stop_words='english',ngram_range=(1,2),min_df=5,tokenizer=LONG_WORDS)

#Unigrams + Stopwords + Lemmer TF-IDF
CV5=TfidfVectorizer(input='content',analyzer = 'word',stop_words='english',min_df=5,tokenizer=MY_LEMMER)

#Stopwords unigrams + bigram +Lemmer TF-IDF
CV6=TfidfVectorizer(input='content', analyzer = 'word',stop_words='english',ngram_range=(1,2),min_df=5,tokenizer=MY_LEMMER)

#Stopwords unigrams + bigram +Stopwords TF-IDF
CV7=TfidfVectorizer(input='content', analyzer = 'word',stop_words='english',ngram_range=(1,2),min_df=5,tokenizer=LONG_WORDS)


# # Multinomial Naive Bayes
from sklearn.naive_bayes import  MultinomialNB
# ### CV1
comments_cv1=CV1.fit_transform(x_train)
test_cv1=CV1.transform(x_test)
mnb = MultinomialNB()

mnb.fit(comments_cv1,y_train)

#k-fold cross validtion to measure accuracy of MNB for lemmatization
from sklearn.model_selection import cross_val_score

mnb_cv1score = cross_val_score(mnb, comments_cv1, y_train, cv=5)

# Print the accuracy of each fold:
print(mnb_cv1score)

# Print the mean accuracy of all 5 folds
print(mnb_cv1score.mean())


## Get the features for column names
featuresmnb1=CV1.get_feature_names_out()
featuresmnb1 = pd.DataFrame(comments_cv1.toarray(), columns=featuresmnb1)
#Headline CountVectorizer WC
topfmnb1=featuresmnb1.sum(axis=0).sort_values(ascending=False)
#Wordcloud of top 20 features
featuresCloudmnb1 = WordCloud(colormap='tab20b',background_color="white", max_words=20).generate_from_frequencies(topfmnb1)
plt.figure(figsize=(20,10))
plt.imshow(featuresCloudmnb1, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#Top 20 Features for non-toxic comments
nottoxic_feature_ranks1 = sorted(zip(mnb.feature_log_prob_[0], CV1.get_feature_names_out()))
nottoxic_features1 = nottoxic_feature_ranks1[-20:]
print(nottoxic_features1)

nottoxicfeatures1=[]
nottoxiclogprob1=[]

for item in nottoxic_feature_ranks1[0:20]:
    nottoxicfeatures1.append(item[1])
    nottoxiclogprob1.append(item[0])
cv1topfeat1=pd.DataFrame(nottoxicfeatures1,columns=['Not Toxic Features'])
cv1topfeat1.insert(loc=1,column="Log_Prob",value=nottoxiclogprob1)


#Top 20 Features for toxic comments
toxic_feature_ranks1 = sorted(zip(mnb.feature_log_prob_[1], CV1.get_feature_names()))
toxic_features1 = toxic_feature_ranks1[-20:]
print(toxic_features1)

toxicfeatures1=[]
toxiclogprob1=[]

for item in toxic_feature_ranks1[-20:]:
    toxicfeatures1.append(item[1])
    toxiclogprob1.append(item[0])

cv1topfeat1.insert(loc=1,column="Toxic Features",value=toxicfeatures1)
cv1topfeat1.insert(loc=1,column="Log_Prob_toxic",value=toxiclogprob1)


#Visualization of to 10 features for not toxic and toxic comments
figure1 = plt.figure()
ax1 = figure1.add_axes([0,0,1,1])
ax1.bar(cv1topfeat1['Not Toxic Features'].values, cv1topfeat1['Log_Prob'].values)
plt.xticks(cv1topfeat1['Not Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure2 = plt.figure()
ax2 = figure2.add_axes([0,0,1,1])
ax2.bar(cv1topfeat1['Toxic Features'].values, cv1topfeat1['Log_Prob_toxic'].values)
plt.xticks(cv1topfeat1['Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Toxic Comments')
plt.show()

#confusion matrix for cross-validation TRAINING DATA!
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import seaborn as sns #for visualizations

mnbtrain_pred1 = cross_val_predict(mnb, comments_cv1, y_train, cv=5)
mnb_train1 = confusion_matrix(y_train, mnbtrain_pred1,labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbtrainHM6=sns.heatmap(mnb_train1, annot=True, cmap='Spectral',fmt='g')
mnbtrainHM6.set_xticklabels(['Not Toxic','Toxic'])
mnbtrainHM6.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV1 Cross Validated Training Data Confusion Matrix ')
sns.despine()


#confusion matrix for TEST DATA!
#transform test data do not fit it to CountVectorizer!
#This tells you how well the model did on NEW DATA, a 30% model

#now look at recall and precision

mnb_pred1= confusion_matrix(y_test, mnb.predict(test_cv1),labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbHM=sns.heatmap(mnb_pred1, annot=True, cmap='Spectral',fmt='g')
mnbHM.set_xticklabels(['Not Toxic','Toxic'])
mnbHM.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV1 Predicted Comment Toxicity Confusion Matrix')
sns.despine()
plt.show()

from sklearn.metrics import accuracy_score
y_pred1=mnb.fit(comments_cv1, y_train).predict(test_cv1)
accuracy_score(y_test, y_pred1)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred1))


# ### CV2

comments_cv2=CV2.fit_transform(x_train)
test_cv2=CV2.transform(x_test)

mnb.fit(comments_cv2,y_train)

#k-fold cross validtion to measure accuracy of MNB
mnb_cv2score = cross_val_score(mnb, comments_cv2, y_train, cv=5)

# Print the accuracy of each fold:
print(mnb_cv2score)

# Print the mean accuracy of all 5 folds
print(mnb_cv2score.mean())

## Get the features for column names
featuresmnb2=CV2.get_feature_names_out()
featuresmnb2 = pd.DataFrame(comments_cv2.toarray(), columns=featuresmnb2)
#Headline CountVectorizer WC
topfmnb2=featuresmnb2.sum(axis=0).sort_values(ascending=False)
#Wordcloud of top 20 features
featuresCloudmnb2 = WordCloud(colormap='tab20b',background_color="white", max_words=20).generate_from_frequencies(topfmnb2)
plt.figure(figsize=(20,10))
plt.imshow(featuresCloudmnb2, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#Top 20 Features for non-toxic comments
nottoxic_feature_ranks2 = sorted(zip(mnb.feature_log_prob_[0], CV2.get_feature_names_out()))
nottoxic_features2 = nottoxic_feature_ranks2[-20:]
print(nottoxic_features2)

nottoxicfeatures2=[]
nottoxiclogprob2=[]

for item in nottoxic_feature_ranks2[0:20]:
    nottoxicfeatures2.append(item[1])
    nottoxiclogprob2.append(item[0])
cv2topfeat=pd.DataFrame(nottoxicfeatures2,columns=['Not Toxic Features'])
cv2topfeat.insert(loc=1,column="Log_Prob",value=nottoxiclogprob2)


#Top 20 Features for toxic comments
toxic_feature_ranks2 = sorted(zip(mnb.feature_log_prob_[1], CV2.get_feature_names()))
toxic_features2 = toxic_feature_ranks2[-20:]
print(toxic_features2)

toxicfeatures2=[]
toxiclogprob2=[]

for item in toxic_feature_ranks2[-20:]:
    toxicfeatures2.append(item[1])
    toxiclogprob2.append(item[0])

cv2topfeat.insert(loc=1,column="Toxic Features",value=toxicfeatures2)
cv2topfeat.insert(loc=1,column="Log_Prob_toxic",value=toxiclogprob2)


#Visualization of to 20 features for not toxic and toxic comments
figure3 = plt.figure()
ax3 = figure3.add_axes([0,0,1,1])
ax3.bar(cv2topfeat['Not Toxic Features'].values, cv2topfeat['Log_Prob'].values)
plt.xticks(cv2topfeat['Not Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure4 = plt.figure()
ax4 = figure4.add_axes([0,0,1,1])
ax4.bar(cv2topfeat['Toxic Features'].values, cv2topfeat['Log_Prob_toxic'].values)
plt.xticks(cv2topfeat['Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Toxic Comments')
plt.show()


#confusion matrix for cross-validation TRAINING DATA!
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

mnbtrain_pred2 = cross_val_predict(mnb, comments_cv2, y_train, cv=5)
mnb_train2 = confusion_matrix(y_train, mnbtrain_pred2,labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbtrainHM=sns.heatmap(mnb_train2, annot=True, cmap='Spectral',fmt='g')
mnbtrainHM.set_xticklabels(['Not Toxic','Toxic'])
mnbtrainHM.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV2 Cross Validated Training Data Confusion Matrix ')
sns.despine()
plt.show()


#confusion matrix for TEST DATA!
#transform test data do not fit it to CountVectorizer!
#This tells you how well the model did on NEW DATA, a 30% model

#now look at recall and precision

mnb_pred2= confusion_matrix(y_test, mnb.predict(test_cv2),labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbHM=sns.heatmap(mnb_pred2, annot=True, cmap='Spectral',fmt='g')
mnbHM.set_xticklabels(['Not Toxic','Toxic'])
mnbHM.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV2 Predicted Comment Toxicity Confusion Matrix')
sns.despine()
plt.show()


from sklearn.metrics import accuracy_score
y_pred2=mnb.fit(comments_cv2, y_train).predict(test_cv2)
accuracy_score(y_test, y_pred2)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred2))


# ### CV3

comments_cv3=CV3.fit_transform(x_train)
test_cv3=CV3.transform(x_test)

mnb.fit(comments_cv3,y_train)

#k-fold cross validtion to measure accuracy of MNB
mnb_cv3score = cross_val_score(mnb, comments_cv3, y_train, cv=5)

# Print the accuracy of each fold:
print(mnb_cv3score)

# Print the mean accuracy of all 5 folds
print(mnb_cv3score.mean())


## Get the features for column names
featuresmnb3=CV3.get_feature_names_out()
featuresmnb3 = pd.DataFrame(comments_cv3.toarray(), columns=featuresmnb3)
#Headline CountVectorizer WC
topfmnb3=featuresmnb3.sum(axis=0).sort_values(ascending=False)
#Wordcloud of top 20 features
featuresCloudmnb3 = WordCloud(colormap='tab20b',background_color="white", max_words=20).generate_from_frequencies(topfmnb3)
plt.figure(figsize=(20,10))
plt.imshow(featuresCloudmnb3, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#Top 20 Features for non-toxic comments
nottoxic_feature_ranks3 = sorted(zip(mnb.feature_log_prob_[0], CV3.get_feature_names_out()))
nottoxic_features3 = nottoxic_feature_ranks3[-20:]
print(nottoxic_features3)

nottoxicfeatures3=[]
nottoxiclogprob3=[]

for item in nottoxic_feature_ranks3[0:20]:
    nottoxicfeatures3.append(item[1])
    nottoxiclogprob3.append(item[0])
cv3topfeat=pd.DataFrame(nottoxicfeatures3,columns=['Not Toxic Features'])
cv3topfeat.insert(loc=1,column="Log_Prob",value=nottoxiclogprob3)


#Top 20 Features for toxic comments
toxic_feature_ranks3 = sorted(zip(mnb.feature_log_prob_[1], CV3.get_feature_names()))
toxic_features3 = toxic_feature_ranks3[-20:]
print(toxic_features3)

toxicfeatures3=[]
toxiclogprob3=[]

for item in toxic_feature_ranks3[-20:]:
    toxicfeatures3.append(item[1])
    toxiclogprob3.append(item[0])

cv3topfeat.insert(loc=1,column="Toxic Features",value=toxicfeatures3)
cv3topfeat.insert(loc=1,column="Log_Prob_toxic",value=toxiclogprob3)



#Visualization of to 20 features for not toxic and toxic comments
figure5 = plt.figure()
ax5 = figure5.add_axes([0,0,1,1])
ax5.bar(cv3topfeat['Not Toxic Features'].values, cv3topfeat['Log_Prob'].values)
plt.xticks(cv3topfeat['Not Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure6 = plt.figure()
ax6 = figure6.add_axes([0,0,1,1])
ax6.bar(cv3topfeat['Toxic Features'].values, cv3topfeat['Log_Prob_toxic'].values)
plt.xticks(cv3topfeat['Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Toxic Comments')
plt.show()


#confusion matrix for cross-validation TRAINING DATA!
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

mnbtrain_pred3 = cross_val_predict(mnb, comments_cv3, y_train, cv=5)
mnb_train3 = confusion_matrix(y_train, mnbtrain_pred3,labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbtrainHM3=sns.heatmap(mnb_train3, annot=True, cmap='Spectral',fmt='g')
mnbtrainHM3.set_xticklabels(['Not Toxic','Toxic'])
mnbtrainHM3.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV3 Cross Validated Training Data Confusion Matrix ')
sns.despine()
plt.show()


#confusion matrix for TEST DATA!
#transform test data do not fit it to CountVectorizer!
#This tells you how well the model did on NEW DATA

#now look at recall and precision

mnb_pred3= confusion_matrix(y_test, mnb.predict(test_cv3),labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbHM3=sns.heatmap(mnb_pred3, annot=True, cmap='Spectral',fmt='g')
mnbHM3.set_xticklabels(['Not Toxic','Toxic'])
mnbHM3.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV3 Predicted Comment Toxicity Confusion Matrix')
sns.despine()
plt.show()



from sklearn.metrics import accuracy_score
y_pred3=mnb.fit(comments_cv3, y_train).predict(test_cv3)
accuracy_score(y_test, y_pred3)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred3))


# ### CV4

comments_cv4=CV4.fit_transform(x_train)
test_cv4=CV4.transform(x_test)

mnb.fit(comments_cv4,y_train)

#k-fold cross validtion to measure accuracy of MNB
mnb_cv4score = cross_val_score(mnb, comments_cv4, y_train, cv=5)

# Print the accuracy of each fold:
print(mnb_cv4score)

# Print the mean accuracy of all 5 folds
print(mnb_cv4score.mean())


## Get the features for column names
featuresmnb4=CV4.get_feature_names_out()
featuresmnb4 = pd.DataFrame(comments_cv4.toarray(), columns=featuresmnb4)
#Headline CountVectorizer WC
topfmnb4=featuresmnb4.sum(axis=0).sort_values(ascending=False)
#Wordcloud of top 20 features
featuresCloudmnb4 = WordCloud(colormap='tab20b',background_color="white", max_words=20).generate_from_frequencies(topfmnb4)
plt.figure(figsize=(20,10))
plt.imshow(featuresCloudmnb4, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#Top 20 Features for non-toxic comments
nottoxic_feature_ranks4 = sorted(zip(mnb.feature_log_prob_[0], CV4.get_feature_names_out()))
nottoxic_features4 = nottoxic_feature_ranks4[-20:]
print(nottoxic_features4)

nottoxicfeatures4=[]
nottoxiclogprob4=[]

for item in nottoxic_feature_ranks4[0:20]:
    nottoxicfeatures4.append(item[1])
    nottoxiclogprob4.append(item[0])
cv4topfeat=pd.DataFrame(nottoxicfeatures4,columns=['Not Toxic Features'])
cv4topfeat.insert(loc=1,column="Log_Prob",value=nottoxiclogprob4)


#Top 20 Features for toxic comments
toxic_feature_ranks4 = sorted(zip(mnb.feature_log_prob_[1], CV4.get_feature_names()))
toxic_features4 = toxic_feature_ranks4[-20:]
print(toxic_features4)

toxicfeatures4=[]
toxiclogprob4=[]

for item in toxic_feature_ranks4[-20:]:
    toxicfeatures4.append(item[1])
    toxiclogprob4.append(item[0])

cv4topfeat.insert(loc=1,column="Toxic Features",value=toxicfeatures4)
cv4topfeat.insert(loc=1,column="Log_Prob_toxic",value=toxiclogprob4)


#Visualization of to 20 features for not toxic and toxic comments
figure7 = plt.figure()
ax7 = figure7.add_axes([0,0,1,1])
ax7.bar(cv4topfeat['Not Toxic Features'].values, cv4topfeat['Log_Prob'].values)
plt.xticks(cv4topfeat['Not Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure8 = plt.figure()
ax8 = figure8.add_axes([0,0,1,1])
ax8.bar(cv4topfeat['Toxic Features'].values, cv4topfeat['Log_Prob_toxic'].values)
plt.xticks(cv4topfeat['Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Toxic Comments')
plt.show()


#confusion matrix for cross-validation TRAINING DATA!
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

mnbtrain_pred4 = cross_val_predict(mnb, comments_cv4, y_train, cv=5)
mnb_train4 = confusion_matrix(y_train, mnbtrain_pred4,labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbtrainHM4=sns.heatmap(mnb_train4, annot=True, cmap='Spectral',fmt='g')
mnbtrainHM4.set_xticklabels(['Not Toxic','Toxic'])
mnbtrainHM4.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV4 Cross Validated Training Data Confusion Matrix ')
sns.despine()


#confusion matrix for TEST DATA!
#transform test data do not fit it to CountVectorizer!
#This tells you how well the model did on NEW DATA

#now look at recall and precision

mnb_pred4= confusion_matrix(y_test, mnb.predict(test_cv4),labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbHM4=sns.heatmap(mnb_pred4, annot=True, cmap='Spectral',fmt='g')
mnbHM4.set_xticklabels(['Not Toxic','Toxic'])
mnbHM4.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV4 Predicted Comment Toxicity Confusion Matrix')
sns.despine()
plt.show()



from sklearn.metrics import accuracy_score
y_pred4=mnb.fit(comments_cv4, y_train).predict(test_cv4)
accuracy_score(y_test, y_pred4)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred4))


# ### CV5

comments_cv5=CV5.fit_transform(x_train)
test_cv5=CV5.transform(x_test)

mnb.fit(comments_cv5,y_train)

#k-fold cross validtion to measure accuracy of MNB
mnb_cv5score = cross_val_score(mnb, comments_cv5, y_train, cv=5)

# Print the accuracy of each fold:
print(mnb_cv5score)

# Print the mean accuracy of all 5 folds
print(mnb_cv5score.mean())


## Get the features for column names
featuresmnb5=CV5.get_feature_names_out()
featuresmnb5 = pd.DataFrame(comments_cv5.toarray(), columns=featuresmnb5)
#Headline CountVectorizer WC
topfmnb5=featuresmnb5.sum(axis=0).sort_values(ascending=False)
#Wordcloud of top 20 features
featuresCloudmnb5 = WordCloud(colormap='tab20b',background_color="white", max_words=20).generate_from_frequencies(topfmnb5)
plt.figure(figsize=(20,10))
plt.imshow(featuresCloudmnb5, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#Top 20 Features for non-toxic comments
nottoxic_feature_ranks5 = sorted(zip(mnb.feature_log_prob_[0], CV5.get_feature_names_out()))
nottoxic_features5 = nottoxic_feature_ranks5[-20:]
print(nottoxic_features5)

nottoxicfeatures5=[]
nottoxiclogprob5=[]

for item in nottoxic_feature_ranks5[0:20]:
    nottoxicfeatures5.append(item[1])
    nottoxiclogprob5.append(item[0])
cv5topfeat=pd.DataFrame(nottoxicfeatures5,columns=['Not Toxic Features'])
cv5topfeat.insert(loc=1,column="Log_Prob",value=nottoxiclogprob5)



#Top 20 Features for toxic comments
toxic_feature_ranks5 = sorted(zip(mnb.feature_log_prob_[1], CV5.get_feature_names()))
toxic_features5 = toxic_feature_ranks5[-20:]
print(toxic_features5)

toxicfeatures5=[]
toxiclogprob5=[]

for item in toxic_feature_ranks5[-20:]:
    toxicfeatures5.append(item[1])
    toxiclogprob5.append(item[0])

cv5topfeat.insert(loc=1,column="Toxic Features",value=toxicfeatures5)
cv5topfeat.insert(loc=1,column="Log_Prob_toxic",value=toxiclogprob5)



#Visualization of to 20 features for not toxic and toxic comments
figure8 = plt.figure()
ax8 = figure8.add_axes([0,0,1,1])
ax8.bar(cv5topfeat['Not Toxic Features'].values, cv5topfeat['Log_Prob'].values)
plt.xticks(cv5topfeat['Not Toxic Features'], rotation = 55)
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure9 = plt.figure()
ax9 = figure9.add_axes([0,0,1,1])
ax9.bar(cv5topfeat['Toxic Features'].values, cv5topfeat['Log_Prob_toxic'].values)
plt.xticks(cv5topfeat['Toxic Features'], rotation = 55)
plt.title('Most Indicative Words for Toxic Comments')
plt.show()


#confusion matrix for cross-validation TRAINING DATA!
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

mnbtrain_pred5 = cross_val_predict(mnb, comments_cv5, y_train, cv=5)
mnb_train5 = confusion_matrix(y_train, mnbtrain_pred5,labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbtrainHM5=sns.heatmap(mnb_train5, annot=True, cmap='Spectral',fmt='g')
mnbtrainHM5.set_xticklabels(['Not Toxic','Toxic'])
mnbtrainHM5.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV5 Cross Validated Training Data Confusion Matrix ')
sns.despine()

#confusion matrix for TEST DATA!
#transform test data do not fit it to CountVectorizer!
#This tells you how well the model did on NEW DATA

#now look at recall and precision

mnb_pred5= confusion_matrix(y_test, mnb.predict(test_cv5),labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbHM5=sns.heatmap(mnb_pred5, annot=True, cmap='Spectral',fmt='g')
mnbHM5.set_xticklabels(['Not Toxic','Toxic'])
mnbHM5.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV5 Predicted Comment Toxicity Confusion Matrix')
sns.despine()
plt.show()


from sklearn.metrics import accuracy_score
y_pred5=mnb.fit(comments_cv5, y_train).predict(test_cv5)
accuracy_score(y_test, y_pred5)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred5))


# ### CV6

comments_cv6=CV6.fit_transform(x_train)
test_cv6=CV6.transform(x_test)

mnb.fit(comments_cv6,y_train)

#k-fold cross validtion to measure accuracy of MNB
mnb_cv6score = cross_val_score(mnb, comments_cv6, y_train, cv=5)

# Print the accuracy of each fold:
print(mnb_cv6score)

# Print the mean accuracy of all 5 folds
print(mnb_cv6score.mean())


## Get the features for column names
featuresmnb6=CV6.get_feature_names_out()
featuresmnb6 = pd.DataFrame(comments_cv6.toarray(), columns=featuresmnb6)
#Headline CountVectorizer WC
topfmnb6=featuresmnb6.sum(axis=0).sort_values(ascending=False)
#Wordcloud of top 20 features
featuresCloudmnb6 = WordCloud(colormap='tab20b',background_color="white", max_words=20).generate_from_frequencies(topfmnb6)
plt.figure(figsize=(20,10))
plt.imshow(featuresCloudmnb6, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#Top 20 Features for non-toxic comments
nottoxic_feature_ranks6 = sorted(zip(mnb.feature_log_prob_[0], CV6.get_feature_names_out()))
nottoxic_features6 = nottoxic_feature_ranks6[-20:]
print(nottoxic_features6)

nottoxicfeatures6=[]
nottoxiclogprob6=[]

for item in nottoxic_feature_ranks6[0:20]:
    nottoxicfeatures6.append(item[1])
    nottoxiclogprob6.append(item[0])
cv6topfeat=pd.DataFrame(nottoxicfeatures6,columns=['Not Toxic Features'])
cv6topfeat.insert(loc=1,column="Log_Prob",value=nottoxiclogprob6)


#Top 20 Features for toxic comments
toxic_feature_ranks6 = sorted(zip(mnb.feature_log_prob_[1], CV6.get_feature_names_out()))
toxic_features6 = toxic_feature_ranks6[-20:]
print(toxic_features6)

toxicfeatures6=[]
toxiclogprob6=[]

for item in toxic_feature_ranks6[-20:]:
    toxicfeatures6.append(item[1])
    toxiclogprob6.append(item[0])

cv6topfeat.insert(loc=1,column="Toxic Features",value=toxicfeatures6)
cv6topfeat.insert(loc=1,column="Log_Prob_toxic",value=toxiclogprob6)


#Visualization of to 20 features for not toxic and toxic comments
figure9 = plt.figure()
ax9 = figure9.add_axes([0,0,1,1])
ax9.bar(cv6topfeat['Not Toxic Features'].values, cv6topfeat['Log_Prob'].values)
plt.xticks(cv6topfeat['Not Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure10 = plt.figure()
ax10 = figure10.add_axes([0,0,1,1])
ax10.bar(cv6topfeat['Toxic Features'].values, cv6topfeat['Log_Prob_toxic'].values)
plt.xticks(cv6topfeat['Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Toxic Comments')
plt.show()


#confusion matrix for cross-validation TRAINING DATA!
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

mnbtrain_pred6 = cross_val_predict(mnb, comments_cv6, y_train, cv=5)
mnb_train6 = confusion_matrix(y_train, mnbtrain_pred6,labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbtrainHM6=sns.heatmap(mnb_train6, annot=True, cmap='Spectral',fmt='g')
mnbtrainHM6.set_xticklabels(['Not Toxic','Toxic'])
mnbtrainHM6.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV6 Cross Validated Training Data Confusion Matrix ')
sns.despine()


#confusion matrix for TEST DATA!
#transform test data do not fit it to CountVectorizer!
#This tells you how well the model did on NEW DATA

#now look at recall and precision

mnb_pred6= confusion_matrix(y_test, mnb.predict(test_cv6),labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbHM6=sns.heatmap(mnb_pred6, annot=True, cmap='Spectral',fmt='g')
mnbHM6.set_xticklabels(['Not Toxic','Toxic'])
mnbHM6.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV6 Predicted Comment Toxicity Confusion Matrix')
sns.despine()
plt.show()

from sklearn.metrics import accuracy_score
y_pred6=mnb.fit(comments_cv6, y_train).predict(test_cv6)
accuracy_score(y_test, y_pred6)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred6))


# ## CV7

comments_cv7=CV7.fit_transform(x_train)
test_cv7=CV7.transform(x_test)

mnb.fit(comments_cv7,y_train)

#k-fold cross validtion to measure accuracy of MNB
mnb_cv7score = cross_val_score(mnb, comments_cv7, y_train, cv=5)

# Print the accuracy of each fold:
print(mnb_cv7score)

# Print the mean accuracy of all 5 folds
print(mnb_cv7score.mean())


## Get the features for column names
featuresmnb7=CV7.get_feature_names_out()
featuresmnb7 = pd.DataFrame(comments_cv7.toarray(), columns=featuresmnb7)
#Headline CountVectorizer WC
topfmnb7=featuresmnb7.sum(axis=0).sort_values(ascending=False)
#Wordcloud of top 20 features
featuresCloudmnb7 = WordCloud(colormap='tab20b',background_color="white", max_words=20).generate_from_frequencies(topfmnb7)
plt.figure(figsize=(20,10))
plt.imshow(featuresCloudmnb7, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#Top 20 Features for non-toxic comments
nottoxic_feature_ranks7 = sorted(zip(mnb.feature_log_prob_[0], CV7.get_feature_names_out()))
nottoxic_features7 = nottoxic_feature_ranks7[-20:]
print(nottoxic_features7)

nottoxicfeatures7=[]
nottoxiclogprob7=[]

for item in nottoxic_feature_ranks7[0:20]:
    nottoxicfeatures7.append(item[1])
    nottoxiclogprob7.append(item[0])
cv7topfeat=pd.DataFrame(nottoxicfeatures7,columns=['Not Toxic Features'])
cv7topfeat.insert(loc=1,column="Log_Prob",value=nottoxiclogprob7)



#Top 20 Features for toxic comments
toxic_feature_ranks7 = sorted(zip(mnb.feature_log_prob_[1], CV7.get_feature_names()))
toxic_features7 = toxic_feature_ranks7[-20:]
print(toxic_features7)

toxicfeatures7=[]
toxiclogprob7=[]

for item in toxic_feature_ranks7[-20:]:
    toxicfeatures7.append(item[1])
    toxiclogprob7.append(item[0])

cv7topfeat.insert(loc=1,column="Toxic Features",value=toxicfeatures7)
cv7topfeat.insert(loc=1,column="Log_Prob_toxic",value=toxiclogprob7)


#Visualization of to 20 features for not toxic and toxic comments
figure10 = plt.figure()
ax10 = figure10.add_axes([0,0,1,1])
ax10.bar(cv7topfeat['Not Toxic Features'].values, cv7topfeat['Log_Prob'].values)
plt.xticks(cv7topfeat['Not Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure11 = plt.figure()
ax11 = figure11.add_axes([0,0,1,1])
ax11.bar(cv7topfeat['Toxic Features'].values, cv7topfeat['Log_Prob_toxic'].values)
plt.xticks(cv7topfeat['Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Toxic Comments')
plt.show()


#confusion matrix for cross-validation TRAINING DATA!
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

mnbtrain_pred7 = cross_val_predict(mnb, comments_cv7, y_train, cv=5)
mnb_train7 = confusion_matrix(y_train, mnbtrain_pred7,labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbtrainHM7=sns.heatmap(mnb_train7, annot=True, cmap='Spectral',fmt='g')
mnbtrainHM7.set_xticklabels(['Not Toxic','Toxic'])
mnbtrainHM7.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV7 Cross Validated Training Data Confusion Matrix ')
sns.despine()

#confusion matrix for TEST DATA!
#transform test data do not fit it to CountVectorizer!
#This tells you how well the model did on NEW DATA

#now look at recall and precision

mnb_pred7= confusion_matrix(y_test, mnb.predict(test_cv7),labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbHM7=sns.heatmap(mnb_pred7, annot=True, cmap='Spectral',fmt='g')
mnbHM7.set_xticklabels(['Not Toxic','Toxic'])
mnbHM7.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV7 Predicted Comment Toxicity Confusion Matrix')
sns.despine()
plt.show()


from sklearn.metrics import accuracy_score
y_pred7=mnb.fit(comments_cv7, y_train).predict(test_cv7)
accuracy_score(y_test, y_pred7)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred7))


##############################################################
##
##    Analysis - KMeans Clustering
##
##############################################################

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans


#Comment Silhouette
visualizer= KElbowVisualizer(KMeans(), k=(2,7), metric='silhouette', timings=False)
visualizer.fit(comments_cv1.toarray())# convert sparse matrix to dense data and fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


kmeans=KMeans(n_clusters=2,max_iter=100,n_init=1)
kmeans.fit(comments_cv1)

predictedcluster=kmeans.predict(comments_cv1)


df = pd.DataFrame()
df['Comments']=x_train
df['Label']=y_train
df["CV1_Cluster"]=predictedcluster
df


#Predicted Cluster Distribution
#set seaborn plotting aesthetics
sns.set_style('darkgrid')
sns.set_palette('Set2')
topicdist=sns.countplot(data=df, x='CV1_Cluster')

plt.title('Comment Count by Predicted Cluster')
plt.xlabel('Predicted Cluster')
plt.ylabel('Number of Comments')
sns.despine()
plt.show()


# ## CV2


#Comment Silhouette
visualizer= KElbowVisualizer(KMeans(), k=(2,5), metric='silhouette', timings=False)
visualizer.fit(comments_cv2.toarray())# convert sparse matrix to dense data and fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


kmeans=KMeans(n_clusters=2,max_iter=100,n_init=1)
kmeans.fit(comments_cv2)
predictedcluster2=kmeans.predict(comments_cv2)


df["CV2_Cluster"]=predictedcluster2
df


#Predicted Cluster Distribution
#set seaborn plotting aesthetics
sns.set_style('darkgrid')
sns.set_palette('Set2')
topicdist=sns.countplot(data=df, x='CV2_Cluster')

plt.title('Comment Count by Predicted Cluster')
plt.xlabel('Predicted Cluster')
plt.ylabel('Number of Comments')
sns.despine()
plt.show()


# ## CV3

#Comment Silhouette
visualizer= KElbowVisualizer(KMeans(), k=(2,5), metric='silhouette', timings=False)
visualizer.fit(comments_cv3.toarray())# convert sparse matrix to dense data and fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data

kmeans=KMeans(n_clusters=3,max_iter=100,n_init=1)
kmeans.fit(comments_cv3)
predictedcluster3=kmeans.predict(comments_cv3)


df["CV3_Cluster"]=predictedcluster3
df

#Predicted Cluster Distribution
#set seaborn plotting aesthetics
sns.set_style('darkgrid')
sns.set_palette('Set2')
sns.countplot(data=df, x='CV3_Cluster')

plt.title('Comment Count by Predicted Cluster')
plt.xlabel('Predicted Cluster')
plt.ylabel('Number of Comments')
sns.despine()
plt.show()


#What is CV3 Clustering referring to?
#Check central of gravity of clusters and print feature forms

print('CV3 Cluster centroids: \n')
order_centroids3=kmeans.cluster_centers_.argsort()[:,::-1]
terms3=CV3.get_feature_names()

for i in range(3):
    print('Cluster %d:' % i)
    for j in order_centroids3[i, :10]:
        print(' %s' % terms3[j])
    print('------------')


#word cloud for most common features in Kmeans
# Output cluster results to a csv file
CV3clusters=df.groupby('CV3_Cluster')
for Cluster in CV3clusters.groups:
    f=open('cv3Cluster'+str(Cluster)+'.csv','w') #create a csv file
    data=CV3clusters.get_group(Cluster)[['Label','Comments']] #include columns of interest
    f.write(data.to_csv(index=True))
    f.close()

#PREDICTED Cluster 0 Wordcloud
#Load Predicted Cluster0 as pandas df
PC0=pd.read_csv('cv3Cluster0.csv')

#create text list of all words in Headline cluster 0
PC0text="".join(Comments for Comments in PC0.Comments.astype(str))

#Generate word cloud
PC0_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC0_WC.generate(PC0text)
plt.figure(figsize=(20,10))
plt.imshow(PC0_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#PREDICTED Cluster 1 Wordcloud
#Load Predicted Cluster1 as pandas df
PC1=pd.read_csv('cv3Cluster1.csv')

#create text list of all words in Headline cluster 1
PC1text="".join(Comments for Comments in PC1.Comments.astype(str))

#Generate word cloud
PC1_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC1_WC.generate(PC1text)
plt.figure(figsize=(20,10))
plt.imshow(PC1_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

#PREDICTED Cluster 2 Wordcloud
#Load Predicted Cluster1 as pandas df
PC2=pd.read_csv('cv3Cluster2.csv')

#create text list of all words in Headline cluster 1
PC2text="".join(Comments for Comments in PC2.Comments.astype(str))

#Generate word cloud
PC2_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC2_WC.generate(PC2text)
plt.figure(figsize=(20,10))
plt.imshow(PC2_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()



# ## CV4

#Comment Silhouette
visualizer= KElbowVisualizer(KMeans(), k=(2,5), metric='silhouette', timings=False)
visualizer.fit(comments_cv4.toarray())# convert sparse matrix to dense data and fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


kmeans=KMeans(n_clusters=3,max_iter=100,n_init=1)
kmeans.fit(comments_cv4)
predictedcluster4=kmeans.predict(comments_cv4)


df["CV4_Cluster"]=predictedcluster4
df


#Predicted Cluster Distribution
#set seaborn plotting aesthetics
sns.set_style('darkgrid')
sns.set_palette('Set2')
sns.countplot(data=df, x='CV4_Cluster')

plt.title('Comment Count by Predicted Cluster')
plt.xlabel('Predicted Cluster')
plt.ylabel('Number of Comments')
sns.despine()
plt.show()



#What is CV4 Clustering referring to?
#Check central of gravity of clusters and print feature forms

print('CV4 Cluster centroids: \n')
order_centroids4=kmeans.cluster_centers_.argsort()[:,::-1]
terms4=CV4.get_feature_names()

for i in range(3):
    print('Cluster %d:' % i)
    for j in order_centroids4[i, :10]:
        print(' %s' % terms4[j])
    print('------------')


#word cloud for most common features in Kmeans
# Output cluster results to a csv file
CV4clusters=df.groupby('CV4_Cluster')
for Cluster in CV4clusters.groups:
    f=open('cv4Cluster'+str(Cluster)+'.csv','w') #create a csv file
    data=CV4clusters.get_group(Cluster)[['Label','Comments']] #include columns of interest
    f.write(data.to_csv(index=True))
    f.close()

#PREDICTED Cluster 0 Wordcloud
#Load Predicted Cluster0 as pandas df
PC0=pd.read_csv('cv4Cluster0.csv')

#create text list of all words in Headline cluster 0
PC0text="".join(Comments for Comments in PC0.Comments.astype(str))

#Generate word cloud
PC0_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC0_WC.generate(PC0text)
plt.figure(figsize=(20,10))
plt.imshow(PC0_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#PREDICTED Cluster 1 Wordcloud
#Load Predicted Cluster1 as pandas df
PC1=pd.read_csv('cv4Cluster1.csv')

#create text list of all words in Headline cluster 1
PC1text="".join(Comments for Comments in PC1.Comments.astype(str))

#Generate word cloud
PC1_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC1_WC.generate(PC1text)
plt.figure(figsize=(20,10))
plt.imshow(PC1_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

#PREDICTED Cluster 2 Wordcloud
#Load Predicted Cluster1 as pandas df
PC2=pd.read_csv('cv4Cluster2.csv')

#create text list of all words in Headline cluster 1
PC2text="".join(Comments for Comments in PC2.Comments.astype(str))

#Generate word cloud
PC2_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC2_WC.generate(PC2text)
plt.figure(figsize=(20,10))
plt.imshow(PC2_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# ## CV5

#Comment Silhouette
visualizer= KElbowVisualizer(KMeans(), k=(2,5), metric='silhouette', timings=False)
visualizer.fit(comments_cv5.toarray())# convert sparse matrix to dense data and fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


kmeans=KMeans(n_clusters=2,max_iter=100,n_init=1)
kmeans.fit(comments_cv5)
predictedcluster5=kmeans.predict(comments_cv5)


df["CV5_Cluster"]=predictedcluster5
df


#Predicted Cluster Distribution
#set seaborn plotting aesthetics
sns.set_style('darkgrid')
sns.set_palette('Set2')
sns.countplot(data=df, x='CV5_Cluster')

plt.title('Comment Count by Predicted Cluster')
plt.xlabel('Predicted Cluster')
plt.ylabel('Number of Comments')
sns.despine()
plt.show()


#What is CV5 Clustering referring to?
#Check central of gravity of clusters and print feature forms

print('CV5 Cluster centroids: \n')
order_centroids5=kmeans.cluster_centers_.argsort()[:,::-1]
terms5=CV5.get_feature_names()

for i in range(2):
    print('Cluster %d:' % i)
    for j in order_centroids5[i, :10]:
        print(' %s' % terms5[j])
    print('------------')


#word cloud for most common features in Kmeans
# Output cluster results to a csv file
CV5clusters=df.groupby('CV5_Cluster')
for Cluster in CV5clusters.groups:
    f=open('Cluster'+str(Cluster)+'.csv','w') #create a csv file
    data=CV5clusters.get_group(Cluster)[['Label','Comments']] #include columns of interest
    f.write(data.to_csv(index=True))
    f.close()


#PREDICTED Cluster 0 Wordcloud
#Load Predicted Cluster0 as pandas df
PC0=pd.read_csv('Cluster0.csv')

#create text list of all words in Headline cluster 0
PC0text="".join(Comments for Comments in PC0.Comments.astype(str))

#Generate word cloud
PC0_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC0_WC.generate(PC0text)
plt.figure(figsize=(20,10))
plt.imshow(PC0_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#PREDICTED Cluster 1 Wordcloud
#Load Predicted Cluster1 as pandas df
PC1=pd.read_csv('Cluster1.csv')

#create text list of all words in Headline cluster 1
PC1text="".join(Comments for Comments in PC1.Comments.astype(str))

#Generate word cloud
PC1_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC1_WC.generate(PC1text)
plt.figure(figsize=(20,10))
plt.imshow(PC1_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

#Cluster 0 represents toxic comments Cluster 1 represents non-toxic comments

#Change 0 values in column CV5_Cluster to 1 since it is actually negative
#Change 1 values in column CV5_Cluster to 0 since it is actually not negative
df['CV5_Cluster'] = df['CV5_Cluster'].replace([0,1],[1,0])
df

#Accuracy of KMeans clustering on Training Data
incorrectmatches= len(df.loc[df.Label != df.CV5_Cluster])
correctmatches= len(df.loc[df.Label == df.CV5_Cluster])
KMeansAccuracy=correctmatches/8904
KMeansAccuracy #Not good but the best out of all 7 vectorized texts


#Prediction Results on Train Data
y=CV5.transform(x_test)


testcluster=kmeans.predict(y)
testclusterdf = pd.DataFrame()
testclusterdf['Comments']=x_test
testclusterdf['Label']=y_test
testclusterdf["CV5_PredCluster"]=testcluster
testclusterdf



testclusterdf['CV5_PredCluster'] = testclusterdf['CV5_PredCluster'].replace([0,1],[1,0])
testclusterdf


#Accuracy of KMeans clustering Test Data
wrong= len(testclusterdf.loc[testclusterdf.Label != testclusterdf.CV5_PredCluster])
correct= len(testclusterdf.loc[testclusterdf.Label == testclusterdf.CV5_PredCluster])
KMeansTestAccuracy=correct/3816
KMeansTestAccuracy #Not good but the best out of all 7 vectorized texts


# ## CV6

#Comment Silhouette
visualizer= KElbowVisualizer(KMeans(), k=(2,5), metric='silhouette', timings=False)
visualizer.fit(comments_cv6.toarray())# convert sparse matrix to dense data and fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


kmeans=KMeans(n_clusters=2,max_iter=100,n_init=1)
kmeans.fit(comments_cv6)
predictedcluster6=kmeans.predict(comments_cv6)


df["CV6_Cluster"]=predictedcluster6
df



#Predicted Cluster Distribution
#set seaborn plotting aesthetics
sns.set_style('darkgrid')
sns.set_palette('Set2')
sns.countplot(data=df, x='CV6_Cluster')

plt.title('Comment Count by Predicted Cluster')
plt.xlabel('Predicted Cluster')
plt.ylabel('Number of Comments')
sns.despine()
plt.show()


#What is CV6 Clustering referring to?
#Check central of gravity of clusters and print feature forms

print('CV6 Cluster centroids: \n')
order_centroids6=kmeans.cluster_centers_.argsort()[:,::-1]
terms6=CV6.get_feature_names()

for i in range(2):
    print('Cluster %d:' % i)
    for j in order_centroids6[i, :10]:
        print(' %s' % terms6[j])
    print('------------')


# ## CV7


#Comment Silhouette
visualizer= KElbowVisualizer(KMeans(), k=(2,5), metric='silhouette', timings=False)
visualizer.fit(comments_cv7.toarray())# convert sparse matrix to dense data and fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data



kmeans=KMeans(n_clusters=3,max_iter=100,n_init=1)
kmeans.fit(comments_cv7)
predictedcluster7=kmeans.predict(comments_cv7)


df["CV7_Cluster"]=predictedcluster7
df



#Predicted Cluster Distribution
#set seaborn plotting aesthetics
sns.set_style('darkgrid')
sns.set_palette('Set2')
sns.countplot(data=df, x='CV7_Cluster')

plt.title('Comment Count by Predicted Cluster')
plt.xlabel('Predicted Cluster')
plt.ylabel('Number of Comments')
sns.despine()
plt.show()



#What is CV7 Clustering referring to?
#Check central of gravity of clusters and print feature forms

print('CV7 Cluster centroids: \n')
order_centroids7=kmeans.cluster_centers_.argsort()[:,::-1]
terms7=CV7.get_feature_names()

for i in range(3):
    print('Cluster %d:' % i)
    for j in order_centroids7[i, :10]:
        print(' %s' % terms7[j])
    print('------------')



#word cloud for most common features in Kmeans
# Output cluster results to a csv file
CV7clusters=df.groupby('CV7_Cluster')
for Cluster in CV7clusters.groups:
    f=open('cv7Cluster'+str(Cluster)+'.csv','w') #create a csv file
    data=CV7clusters.get_group(Cluster)[['Label','Comments']] #include columns of interest
    f.write(data.to_csv(index=True))
    f.close()

#PREDICTED Cluster 0 Wordcloud
#Load Predicted Cluster0 as pandas df
PC0=pd.read_csv('cv7Cluster0.csv')

#create text list of all words in Headline cluster 0
PC0text="".join(Comments for Comments in PC0.Comments.astype(str))

#Generate word cloud
PC0_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC0_WC.generate(PC0text)
plt.figure(figsize=(20,10))
plt.imshow(PC0_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#PREDICTED Cluster 1 Wordcloud
#Load Predicted Cluster1 as pandas df
PC1=pd.read_csv('cv7Cluster1.csv')

#create text list of all words in Headline cluster 1
PC1text="".join(Comments for Comments in PC1.Comments.astype(str))

#Generate word cloud
PC1_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC1_WC.generate(PC1text)
plt.figure(figsize=(20,10))
plt.imshow(PC1_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

#PREDICTED Cluster 2 Wordcloud
#Load Predicted Cluster1 as pandas df
PC2=pd.read_csv('cv7Cluster2.csv')

#create text list of all words in Headline cluster 1
PC2text="".join(Comments for Comments in PC2.Comments.astype(str))

#Generate word cloud
PC2_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC2_WC.generate(PC2text)
plt.figure(figsize=(20,10))
plt.imshow(PC2_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()



print(df['Label'].value_counts())
print(df['CV1_Cluster'].value_counts())
print(df['CV2_Cluster'].value_counts())
print(df['CV3_Cluster'].value_counts())
print(df['CV4_Cluster'].value_counts())
print(df['CV5_Cluster'].value_counts())
print(df['CV6_Cluster'].value_counts())
print(df['CV7_Cluster'].value_counts())


# # Visualizations

from matplotlib.ticker import PercentFormatter

mnbdf = pd.DataFrame({'Vectorizer Type': ['CV1', 'CV2', 'CV3', 'CV4', 'CV5',
                           'CV6', 'CV7'],
                   'Train Accuracy': [.813, .814, .826, .824, .826,
                                 .820, .824],
                   'Predictive Accuracy': [.818,.815,.819,.817,.826,.822,.817]})
x_=mnbdf.columns[0]
y_=mnbdf.columns[1]
y2_=mnbdf.columns[2]

data1=mnbdf[[x_,y_]]
data2=mnbdf[[x_,y2_]]


#Predicted Cluster Distribution
#set seaborn plotting aesthetics
plt.figure(figsize=(15,8))
sns.set_style('darkgrid')
sns.set_palette('Set2')
ax=sns.barplot(data=data1,x=x_,y=y_)
width_scale=0.45
for bar in ax.containers[0]:
    bar.set_width(bar.get_width()*width_scale)

ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set(ylim=(.75, .83))
plt.ylabel('Accuracy')

ax2=ax.twinx()
sns.set_style('darkgrid')
sns.set_palette('Set2')
sns.barplot(data=data2, x=x_,y=y2_,ax=ax2)
for bar in ax2.containers[0]:
    x=bar.get_x()
    w=bar.get_width()
    bar.set_x(x+w*(1-width_scale))
    bar.set_width(w*width_scale)

ax2.yaxis.set_major_formatter(PercentFormatter(1))
ax2.set(ylim=(.75, .83))

plt.title('Naive Bayes Results Overview')
plt.xlabel('Vectorizer Type')
from matplotlib.ticker import PercentFormatter
plt.ylabel('')#hides label on the right
ax2.axes.yaxis.set_ticklabels([]) #removes additional y scale that is the same on the left

sns.despine()
plt.show()


##############################################################
##
##    Additional feature reduction
##
##############################################################

#Create a list of words would like to be removed from the feature sets
#The lists of words below will be removed when runing models in the remaining of this script
unhelpfulFeatures = ['\n','\n\n','000000', '084080', '1000', '101', '102', '103', '104',
                     '105', '106', '107', '108', '109', '110', '111', '112',
                     '113', '114', '115', '116', '117', '118', '1185', '119',
                     '11th', '120', '121', '122', '123', '124', '125', '127',
                     '128', '12th', '130', '131', '132', '133', '134', '135',
                     '136', '137', '138', '140', '141', '142', '143', '144',
                     '145', '146', '147', '148', '149', '14th', '150', '151',
                     '152', '153', '154', '155', '156', '157', '158', '159',
                     '15th', '160', '161', '162', '163', '164', '165', '166',
                     '167', '168', '169', '170', '171', '172', '174', '175',
                     '176', '177', '178', '179', '180', '181', '182', '183',
                     '184', '185', '186', '187', '188', '189', '18th', '190',
                     '191', '1911', '1912', '193', '1930', '194', '1940',
                     '1944', '1945', '195', '1950', '196', '1967', '1968',
                     '197', '1971', '1975', '1978', '1979', '198', '1980',
                     '1980s', '1981', '1982', '1984', '1985', '1986', '1989',
                     '199', '1990', '1990s', '1991', '1992', '1993', '1994',
                     '1995', '1996', '1997', '1998', '1999', '19th', '1px',
                     '1st', '2000', '2001', '2002', '201', '2015', '2016', '202',
                     '203', '204', '205', '206', '207', '208', '209', '20th',
                     '210', '211', '212', '213', '214', '215', '216', '217',
                     '218', '219', '21st', '220', '221', '222', '223', '224',
                     '225', '226', '227', '228', '229', '230', '231', '232',
                     '233', '234', '235', '236', '237', '238', '239', '240',
                     '241', '242', '243', '244', '245', '246', '247', '248',
                     '249', '250', '251', '252', '253', '254', '255',
                     '27_noticeboard', '2nd', '300', '32', '34', '38', '3d',
                     '3rd', '400', '41', '43', '47', '49', '4chan', '4th', '500',
                     '52', '56', '5th', '600', '61', '63', '6th', '73', '74',
                     '77', '79', '7th', '800', '83', '84', '85', '87', '88',
                     '89', '8th', '900', '91', '93', '95', '96', '__', 'a7',
                     'ab', 'abbey', 'abbreviations', 'abc','abraham', 'abstract',
                     'abu', 'ac', 'academia', 'academics', 'academy', 'aclu',
                     'acronym', 'acted', 'actor', 'actors', 'actress', 'adam',
                     'adams', 'additionally', 'administration', 'admission',
                     'adolescent','adolf', 'ads', 'adult', 'adults', 'advertise',
                     'advertisement', 'ae', 'aeropagitica', 'afds', 'affix',
                     'affiliated', 'afghanistan', 'africa', 'african',  'aforementioned',
                     'africa', 'afternoon', 'aged', 'agencies', 'agency',
                     'agent', 'agents', 'ages', 'agf', 'ahem', 'aircraft', 'airport',
                     'aiv', 'aka', 'akin', 'alabama', 'alan', 'alas', 'albania',
                     'albanian', 'albanians', 'albert', 'albums', 'alcohol', 'alex',
                     'alexander', 'ali', 'alien', 'allah', 'allen', 'alumni',
                     'ama', 'amazon', 'ancestor', 'ancestors', 'ancestral',
                     'ancestry', 'anderson', 'andrew', 'andy', 'angel', 'angeles',
                     'angle', 'anglo', 'animal', 'animation', 'ann', 'anna', 'announce',
                     'announced', 'announcement', 'annual', 'anon', 'anthony',
                     'antics', 'antisemetic', 'antisemitsm', 'anus', 'aol',
                     'aprtheid', 'ape', 'apes', 'apple', 'applicable', 'application',
                     'application', 'applications', 'applied', 'applies', 'applying',
                     'appointed', 'approximately', 'apr', 'apt', 'ar', 'architecture',
                     'archived', 'archiving', 'article', 'wikipedia', 'arizona', 'armenia',
                     'armenian', 'armenians', 'arms', 'armstrong', 'asia', 'asian',
                     'asians', 'assyrian','atheist', 'b4', "audio", 'audience', 'audio', 'aug', 'aussie',
                     'australian', 'australians', 'austrian', 'auto', 'autobiography',
                     'automated', 'automatic', 'avatar', 'aviation', 'axis', 'az',
                     'azerbaijan', 'azerbaijani', 'ba', 'babe', 'babies', 'badge',
                     'bar', 'barn', 'barnstars', 'base', 'baseball', 'bangladesh',
                     'bank', 'banks', 'banner', 'banter', 'basketball', 'bass',
                     'bat', 'batch', 'batman', 'bats', 'bay', 'bbc', 'bc', 'beach',
                     'bear', 'bears', 'beatles', 'becasue', 'beck', 'beckjord',
                     'becouse', 'becuase', 'bed', 'bee', 'begins', 'begun', 'bent',
                     'bernard', 'bi', 'bibliography', 'bits', 'blacks', 'blah',
                     'blanked', 'blanket', 'blofeld', 'blow', 'blowing', 'bnp',
                     'boards', 'boilerplate', 'bollocks', 'bone', 'boobs', 'border',
                     'bosnia', 'boston', 'bots', 'bottle', 'brazil', 'brd', 'breakfast',
                     'breast', 'breasts', 'brian', 'britain', 'britannica', 'brits',
                     'broadcast', 'brother', 'brothers', 'brown', 'browser', 'bruce',
                     'bucket', 'buck', 'bud', 'buddhist', 'bug', 'bugs', 'buildings',
                     'bulk', 'bunchofgrapes', 'bureaucratic', 'bureaucrats', 'bus',
                     "bytes", 'byzantine', 'ca', 'cabal', 'cad', 'caesar', 'cake',
                     'calendar', 'california', 'cambridge', 'camera', 'camp', 'campaigns',
                     'camps', 'campus', 'canada', 'canadian', 'canadians', 'cancer',
                     'candidate', 'candidates', 'cannon', 'canon', 'capital',
                     'capitalized', 'caps', 'captain', 'caption', 'car', 'carbon',
                     'card', 'carl', 'carolina', 'carrier', 'carrots', 'cars',
                     'cartoon', 'cast', 'casting', 'castro', 'catholics', 'cats',
                     'caucasian', 'ccp', 'cd', 'ce', 'celebrity', 'cell', 'cellpadding',
                     'cells', 'cellspacing', 'celtic', 'cena', 'census', 'cent',
                     'central', 'centra', 'cents', 'centuries', 'ceo', 'cf', 'cfd',
                     'cgi', 'chair', 'chain', 'champion', 'champions', 'championship',
                     'chan', 'channel', 'channels', 'chapter', 'characteristic',
                     'characterized', "characteristics", 'charles', 'chart', 'charts',
                     'chase', 'chauvinist', 'chavez', 'checks', 'checkuser', 'cheek',
                     'cheese', 'chemical', 'cherry', 'chess', 'chest', 'chicago',
                     'chicken', 'chief', 'childhood', 'chip', 'chocolate', 'chris',
                     'chriso', 'christianity', 'christians', 'celebrity', 'cell',
                     'ccp', 'cd', 'ce', 'cellpadding', 'cells', 'cellspacing',
                     'celtic', 'cena', 'cent', 'centered', 'central', 'centre',
                     'cents', 'centuries', 'ceo', 'cf', 'cfd', 'cgi', 'chair',
                     'chan', 'channel', 'champion', 'championship', 'chan',
                     'channel', 'channels', 'chapter', 'checks', 'cheek', 'cheese',
                     'chemical', 'cherry', 'chess', 'chest', 'chicago', 'chief',
                     'christmas', 'christopher', 'chuck', 'churches', 'churchill',
                     'ck', 'clan', 'classed', 'classes', 'classic', 'classical',
                     'clique', 'closely', 'closer', 'closest', 'closing', 'clothes',
                     'cloud', 'clouds', 'clown', 'clowns', 'clubs', 'cnn', 'co2',
                     'coal', 'coast', 'codes', 'coding', 'coffee', 'coi', 'coin',
                     'coldplay', 'colleagues', 'collect', 'collecting', 'collective',
                     'colors', 'colour', 'colspan', 'columbia', 'column', 'comics',
                     'comma', 'commas', 'commander', 'compiled', 'concensus', 'conception',
                     'commie', 'commies', 'comission', 'commonwealth', 'communism',
                     "communists", 'computers', 'computing', 'concert', 'condoms',
                     'conference', 'constitution', 'constitutional', 'construction',
                     'constructed', 'cont', "cook", "cookie", 'cooper', 'cop',
                     'copies', 'cops', 'copyedit', 'copyediting', 'copyrights',
                     'copyvio', 'corbett', 'core', 'coren', 'corner', 'corporation',
                     'corps', 'corpse', 'counted', 'counter', 'counties', 'counting',
                     'countless', 'counts', 'countless', 'county', 'coup', 'courts',
                     'cousin', 'cow', 'cowboy', 'coz', 'crack', 'crackpot', 'craig',
                     'crank', 'creek', 'cretin', 'crew', 'cricket', 'crimea','croatia',
                     'croatian', 'crock', 'crowd', 'crown', 'crusade', 'crusades',
                     'crysal', 'cs', 'csd', 'cuba', 'cuban', 'cult', 'cultures',
                     'cup', 'cur', 'currency', 'curtain', 'cus', 'customers',
                     'cuz', 'cyber', 'cycle', 'czech', 'dab', 'dad', 'daft', 'dan',
                     'daniel', 'danny', 'darren', 'darwin', 'dash', 'dat', 'database',
                     'dave', 'daughter', 'davis', 'dawn', 'dc', 'deaf', 'deals', 'dean',
                     'dec', 'decade', 'decades', 'decausa', 'dem', 'demo', 'democracy',
                     'democrat', 'democratic', 'democrats', 'demographic', 'demographics',
                     'denmark', 'dennis', 'departure', 'dialogue', 'dictator', 'dictatorship',
                     'dictionaries', 'diego', 'diffs', 'digital', 'dimensions', 'dinner',
                     'directions', 'director', 'directors', 'directory', 'disabled',
                     'discovery', 'disease', 'disney', 'distrct', 'distribution',
                     'ditto', 'diverse', 'dna', 'doc', 'doctor', 'doctors', 'doctrine',
                     'documentary', 'documentation', 'documented', 'documents',
                     'dodo', 'dogs', 'doin', 'dollar', 'dollars', 'donald', 'doo',
                     'door', 'dot', 'doug', 'download', 'downtown', 'dozen', 'dozens',
                     'dq', 'draft', 'drafting', 'drag', 'dragon', 'dragons', 'drawing',
                     'drawn', 'dreamguy', 'dress', 'drew', 'drink', 'drinking', 'drivel',
                     'duke', 'dungeons', 'duplicate', 'dust', 'dutch', 'dvd', 'dyk',
                     'dyke', 'dynasty', 'eagle', 'ear', 'eastern', 'eating', 'ebay',
                     'ec', 'economic', 'economy', 'ed','edge', 'edition', 'editorial',
                     'edu', 'edward', 'egg', 'egypt', 'egyptian', 'einstein', 'el',
                     'elected', 'elections', 'electric', 'electronic', 'elephant',
                     'elite', 'elitist', 'elonka', 'elses', 'em', 'emailed', 'emails',
                     'employee', 'employees', 'employer', 'encyclopedia', 'encyclopaedic',
                     'encyclopedias', 'endorse', 'ends', 'engine', 'engineer',
                     'engineering', 'england', 'episodes', 'equation', 'equations',
                     'equipment', 'er', 'era', 'eric', 'esp', 'espouse', 'essay',
                     'est', 'et', 'eu', 'eugenics', 'europeans', 'evening', 'everyday',
                     'everytime', 'ex', 'exams', 'eyed', 'f5fffa', 'fa', 'fac',
                     'facebook', 'facility', 'facist', 'fair_use', 'fascists',
                     'fathers', 'fbi', 'fc', 'feb', 'feet', 'feminist', 'feminists',
                     'ffffff', 'fidel', 'fifth', 'filmography', 'films', 'financial',
                     'finland', 'firefox', 'firstly', 'fish', 'fisherqueen', 'font',
                     'food', 'foot', 'footnote', 'footnotes', 'forbes', 'foreign',
                     'foreigners', 'fork', 'fort', 'forth', 'flickr', 'floor',
                     'florida', 'floyd', 'fo', 'forums', 'fox', 'fr', 'france',
                     'francis', 'francisco', 'frank', 'franklin', 'fred', 'freud',
                     'friday', 'frozen', 'fruit', 'ft', 'fundamentalist', 'funeral',
                     'furry','fwiw', 'fyi', 'g1', 'g11', 'galaxy', 'galleries',
                     'gallery', 'gamaliel', 'gamegate', 'gan', 'garage', 'gary',
                     'gays', 'gb', 'gd', 'gender', 'gene', 'genesis', 'genetics',
                     'geographical', 'geography', 'geographically', 'geology',
                     'georgia', 'geographic', 'gerard', 'germanic', 'germans',
                     'gfdl', 'gfdl', 'gibson', 'gif', 'gimme', 'girlfriend',
                     'glass', 'gm', 'gmail', 'gnu', 'golf', 'gong', 'goodbye',
                     'gordon', 'gospel', 'gothic', 'greece', 'greeks', 'greg',
                     'gw', 'gwen', "hadn", 'hahaha','hahahaha', 'halfway',
                     'hall', 'harry', "harvard", 'hat', 'hav', 'heap',
                     'highway', 'highways', 'hill', 'hindi', 'hinduism', 'hindus',
                     'hmm', 'hmmm', 'hmmmmm', 'hobby', 'hockey', 'holiday', 'holier',
                     'holla', 'hollywood', 'holocaust', 'holy', 'homeland', 'homepage',
                     'homes', 'homework', 'hominem', 'homo', 'homophobic', 'homosexuality',
                     'homosexuals', 'heterosexual', 'heaven', 'hebrew', 'heh',
                     'height', 'henry', 'herbert', 'heritage', 'hes', 'hesperian',
                     'horizontal', 'horses', 'hospital', 'host', 'hosted', 'hosting',
                     'hosts', 'hotel', 'hotmail', 'houses', 'housing', 'houston',
                     'howard', 'htm', 'https', 'hugo', 'hungarian', 'idle', 'iii',
                     'ilk', 'imdb', 'imho', 'immigrant', 'immigrants', 'imo', 'imperialism',
                     'incorporate', 'incorporated', 'indians', 'indonesia', 'indonesian',
                     'industry', 'industrial', 'infoboxes', 'ing', 'int','internationally',
                     'intro', 'introduce', 'interviewed', 'interviews', 'introducing',
                     'ips', 'iq', 'ir', 'ira', 'iranian', 'iranians', 'iranica',
                     'iraq', 'irc', 'ireland', 'iron', 'irs', 'isbn', 'islamic',
                     'islamist', 'island', 'islands', 'isp', 'israeli', 'israelis',
                     'italian', 'italy', 'item', 'items', 'iv', 'ive', 'ja', 'jackson',
                     'jam', 'jamesbwatson', 'jamie', 'jan', 'jane', 'janet', 'japan',
                     'jason', 'jay', 'jayjg', 'jeez', 'jeff', 'jehovah', 'jenkins',
                     'jeppiz', 'jeremy', 'jerry', 'jersey', 'jet', 'jim', 'jimmy',
                     'jim', 'joan', 'jobs', 'joe', 'johnny', 'johnson', 'jon',
                     'jonathan', 'jones', 'jordan', 'jose', 'joseph', 'jossi',
                     'jpgordon', 'jpgthank', 'jr', 'judaism', 'jus', 'justin',
                     'kansas', 'karl', 'keith', 'kelly', 'ken', 'kennedy', 'kentucky',
                     'kevin', 'keyboard', 'khan', 'kim', 'kinda', 'kingdom', 'kings',
                     'km','knees', 'knight', 'knox', 'kong', 'korea', 'korean',
                     'kosovo', 'ku', 'kurdish', 'kurt', 'lab', 'labor', 'labour',
                     'laden', 'lake', 'lands', 'languages', 'lanka', 'larry', 'latin',
                     'latino', 'laurent', 'lawrence', 'lbs', 'le', 'lecture', 'lede',
                     'lee', 'leftist', 'leg', 'legs', 'length', 'lengths', 'lesbian',
                     'letters', 'lewis', 'lgbt', 'lightning', 'lil', 'lincoln',
                     'linda', 'linguistic', 'linguistics', 'linguists', 'lips',
                     'lisa', 'literature', 'liz', 'lmao', 'locate', 'located',
                     'locations', 'logo', 'logos', 'logs', 'lolz', 'london', 'longtime',
                     'lopez', 'los', 'louis', 'luke', 'lulz', 'lunch', 'lynch',
                     'lynn', 'lyrics', 'ma', 'macedonia', 'macedonian', 'machine',
                     'machines', 'magazines', 'magic', 'mah', 'mailing', 'mainpagebg',
                     'mainspace', 'malaysia', 'males', 'mankind', 'maps', 'marc',
                     'marcus', 'mardyks', 'margaret', 'margin', 'maria', 'marie',
                     'mario', 'market', 'market', 'marketing', 'markets', 'marking',
                     'marks', 'marriage', 'married', 'marry', 'mars', 'martial',
                     'martin', 'marvel', 'marx', 'marxist', 'mary', 'massachusetts',
                     'masters', 'masses', 'materials','math', 'mathematical',
                     'mathematics', 'matt', 'max', 'measure', 'measured', 'measurement',
                     'measures', 'measuring', 'meat', 'meatpuppet', 'mechanics',
                     'mechanism', 'mediawiki', 'mediterranean', 'medium', 'meetup',
                     'mel', 'messenger', 'meta', 'metalcore', 'meters', 'metro',
                     'mexican', 'mexicans', 'mexico', 'mfd', 'micro', 'microsoft',
                     'mid', 'midnight', 'mike', 'miles', 'mini', 'minister', 'ministry',
                     'minorities', 'minute', 'mississippi', 'mister', 'mitchell',
                     'mix', 'mixed', 'mixture', 'mo', 'model', 'models', 'mommy',
                     'monday', 'moon', 'moore', 'mop','morgan', 'mormon', 'morning',
                     'morris', 'mos', 'moscow', 'mothers', 'mountain', 'movies',
                     'mp', 'mrs', 'ms', 'msg', 'mtv', 'mud', 'muhammad', 'multi',
                     'myspace', 'na', 'nancy', 'nasa', 'nationalism', 'nationalistic',
                     'nationalists', 'nationality', 'nato', 'naval', 'nazism', 'negro',
                     'nbc', 'naqlinwiki', 'nd', 'neck', 'neighbor', 'netherlands',
                     'network', 'networking', 'networks', 'newsletter', 'newspapers',
                     'nicholas', 'nick', 'nickname', 'nigga', 'nigger', 'niggers',
                     'nintendo', 'norman', 'northern', 'notification', 'noun',
                     'nov', 'novel', 'novels', 'npa', 'nsw', 'nt', 'ny', 'nz',
                     'ocean', 'oct', 'officer', 'officers', 'offline', 'officials',
                     'ohio', 'oi', 'oil', 'oklahoma', 'ol', 'older', 'oldid', 'olds',
                     'olympics', 'op', 'oprah', 'oral', 'orange', 'oregon', 'orleans',
                     'orthodox', 'osama', 'oscar', 'ot', 'otters', 'ottoman', 'oxford',
                     'pa', 'pack', 'package', 'padding', 'pagan', 'pagehelp', 'pagehow',
                     'pagestutorialhow', 'paint', 'painting', 'pakistan', 'pakistani',
                     'palace', 'palestine', 'palestinian', 'palestianians', 'pan',
                     'panama', 'pants', 'papers', 'para', 'parent', 'paris', 'park',
                     'parker', 'parliament', 'parrot', 'partisan', 'passage', 'password',
                     'paste', 'pasted', 'pasting', 'pat', 'patrick', 'patriotic',
                     'pbs', 'pc', 'pd', 'pdf', 'peacock','pedia', 'pen', 'pending',
                     'penny', 'people', 'peopel', 'percent', 'percentage', 'perry',
                     'persian', 'persians', 'peruvian', 'pete', 'peter', 'pg', 'ph',
                     'phelps', 'phil', 'philip', 'phone', 'photograph', 'photographer',
                     'photographs', 'photography', 'pic', 'pics', 'pierre', 'pig', 'pigs',
                     'plane', 'planets', 'plant', 'plants', 'platic', 'plate', 'pls',
                     'plural', 'plz', 'pm', 'png', 'poland', 'polish', 'politically',
                     'politician', 'politicians', 'pollution', 'pool', 'pope', 'poppers',
                     'pops', 'populations', 'pork', 'port', 'portal', 'portrait',
                     'portugal', 'portuguese', 'poster', 'posters', 'postings',
                     'potter', 'ppl', 'pr', 'presidency', 'presidential', 'prev',
                     'priest', 'priests','prince', 'princess', 'print', 'printed',
                     'printing', 'prod', 'prof', 'profile', 'profiles', 'professor',
                     'professionals', 'programming', 'programs', 'prophet', 'protestants',
                     'pseudoscience', 'psychiatric', 'psychiatry', 'psychological',
                     'psychology', 'purple', 'quantum', 'queen', 'queer', 'quest',
                     'races', 'rachel', 'rail', 'railway', 'rain', 'raja', 'rajputs',
                     'rap', 'rawat', 'ray', 'redlinks', 'ref', 'refugees', 'regime',
                     'register', 'registered', 'registering', 'reign', 'religions',
                     'republican', 'republicans', 'resident', 'residential', 'residents',
                     'retire', 'retired', 'retirement', 'reuters', 'reviewer', 'reviewing',
                     'rfa', 'rex',  'rice', 'rich', 'rick', 'ring', 'ritual', 'river',
                     'rm', 'ro', 'roads', 'rob', 'robert', 'robot', 'rocket', 'rocks',
                     'rofl', 'roger', 'roll', 'rollback', 'roman', 'romania', 'romanian',
                     'rome', 'ron', 'root', 'roots', 'ross', 'roy', 'rss', 'russians',
                     'ruth', 'rv', 'ryan', 'ryulong', 'sa', 'saddam', 'saga', 'sahara',
                     'saint', 'sales', 'salt', 'sam', 'scan', 'scenario', 'scene',
                     'scenes', 'schools', 'sciences', 'scientifically', 'physics',
                     'scientist', 'scientists', 'scientology', 'scjessy', 'scotland',
                     'scott', 'scottish', 'screen', 'screenshots', 'script', 'script',
                     'scroll', 'se', 'sea', 'sean', 'sell', 'selling', 'senate',
                     'senior', 'sep', 'sept', 'serb', 'serbia', 'serbian', 'serbs',
                     'servant', 'server', 'servers', 'seb', 'serves', 'serve', 'services',
                     'serving', 'seven', 'seventh', 'sh', 'sheep', 'sheesh', 'shell',
                     'shop', 'shoulder', 'shower', 'si', 'sic', 'sig', 'sigh',
                     'sikh', 'singapore', 'singer', 'sing', 'singh', 'singles', 'singular',
                     'skirt', 'skull', 'sky', 'slave', 'slaves', 'slavic', 'slavs',
                     'sleep', 'slim', 'smaller', 'snow', 'snowman', 'soap', 'soapbox',
                     'soccer', 'socialism', 'socialist', 'sockpuppetry', 'sockpuppets',
                     'soil', 'solar', 'sold', 'someday', 'someones', 'somethin', 'songs',
                     'sons', 'simon', 'ship', 'sheriff', 'shi', 'ships', 'shirts',
                     'smith', 'sony', 'sooo', 'soul', 'southern', 'soviet', 'spain',
                     'species', 'spell', 'spi', 'spin', 'spirit', 'spiritual', 'sport',
                     'spring', 'spy', 'squad', 'sports', 'square', 'sr', 'sri', 'ss',
                     'st', 'stadium', 'staff', 'stage', 'stalin', 'stamp', 'stanford',
                     'stars', 'station', 'stations', 'statistical', 'statistics', 'stats',
                     'stephen', 'steve', 'steven', 'stone', 'straw', 'stream', 'streets',
                     'studied', 'studio', 'studying', 'sub', 'subpage', 'subsection',
                     'subsequent', 'subsequently', 'summer', 'sun', 'sunday', 'sup',
                     'supremacist', 'supreme', 'susan', 'svg', 'swastika', 'swedish',
                     'symbol', 'symbols', 'syntax', 'synthesis', 'syria', 'syrian',
                     'sysop', 'ta', 'tab', 'tables', 'tabloid', 'tabtab', 'tail',
                     'tales', 'tall', 'tape', 'tech', 'technology', 'ted', 'taylor',
                     'tax', 'taxes', 'tc', 'tea', 'tech', 'ted', 'teen', 'teenager',
                     'teenagers', 'teens', 'television', 'temp', 'temple', 'territories',
                     'territory', 'terry', 'testament', 'texas', 'textbook', 'textbooks',
                     'texts', 'tfd', 'th', 'theater', 'theirs', 'theme', 'themes',
                     'theorist', 'therapy', 'thereof', 'theres', 'thesis', 'thier',
                     'theirs', 'theme', 'themes', 'thirdly', 'thirty', 'tho', 'thomas',
                     'thompson', 'thou', 'thousand', 'threads', 'throat', 'thug', 'thugs',
                     'thumb', 'thursday', 'tickets', 'tiger', 'til', 'tim', 'tiny',
                     'tip', 'tips', 'titled', 'tk', 'tlk', 'tom', 'tomorrow', 'ton',
                     'tonight', 'tons', 'tony', 'tops', 'toronto', 'totalitarian',
                     'trademark', 'tradition', 'traditional', 'traditionally',
                     'traditions', 'traffic', 'treatment', 'trek', 'tribes', 'triple',
                     'trivia', 'trivial', 'troops', 'truck', 'trout', 'ts', 'turk',
                     'turkey', 'turkic', 'turks', 'tw', 'twist', 'twisted', 'twitter',
                     'tyler', 'typed', 'typing', 'typo', 'typos', 'tyranny', 'tyrant',
                     'tyrants', 'ufc', 'ukraine', 'ukranian', 'um', 'umm', 'uncited',
                     'uncle', 'uneducated', 'unemployed', 'unencyclopedic', 'unilateral',
                     'unilaterally', 'unit', 'units', 'unknown','unregistered', 'untagged',
                     'untill', 'updates', 'updating', 'upgrade', 'uploads', 'urban',
                     'url', 'usenet', 'user_talk', 'userbox', 'userboxes', 'usernames',
                     'userspace', 'ussr', 'uw', 'valley', 'van', 'vehicle', 'verb',
                     'verbal',  'verbatim', 'verizon', 'versa', 'versions', 'vertical',
                     'veteran', 'vfd', 'vice', 'victor', 'videos', 'villages', 'vincent',
                     'vinci', 'virginia', 'virditas', 'virus', 'viruses', 'von', 'voted',
                     'voters', 'votes', 'voting', 'wake', 'walk', 'walker', 'walking',
                     'walks', 'wall', 'walls', 'walt', 'walter', 'warfare', 'wash',
                     'washington', 'wet', 'whilst', 'whites', 'wholesale', 'wholly',
                     'whos', 'wider', 'widespread', 'width', 'wik', 'wikia', 'wikibreak',
                     'wikilove', 'wikinazi', 'wikinazis', 'wikipediahow', 'wikipediatutorialhow',
                     'wikipedia', 'wikipeida', 'wikiprojects', 'wikiquette', 'wikis',
                     'wikitionary', 'wikzilla', 'wil', 'william', 'williams', 'willy',
                     'wilson', 'wind', 'windows', 'winter', 'witch', 'wizard', 'wolf',
                     'wolfowitz', 'worded', 'workers', 'worlds', 'worldwide', 'worm',
                     'worship', 'wrestling', 'wright', 'wrist', 'writer', 'writers',
                     'writes', 'writings', 'wt', 'wwe', 'xbox', 'xd', 'xenophobic',
                     'xxx', 'yahoo', 'yale', 'yall', 'yamla', 'yanks', 'yank', 'yard',
                     'ye', 'yellow', 'yer', 'yesterday', 'younger', 'youth', 'yr', 'yu',
                     'yugoslav', 'yugoslavia', 'zionist', 'zoe', 'zombies']

print(len(unhelpfulFeatures))

##############################################################
##
##    Analysis - Naive Bayes with Additional Feature Removal
##
##############################################################
# ### Split before continuining with vectorizer cleaning steps

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(comments,labels,random_state=0,stratify=labels,test_size=.3,shuffle=True)

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))


def LONG_WORDS(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [w for w in words if len(w) > 2]
    return words

LEMMER = WordNetLemmatizer()
def MY_LEMMER(str_input):
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [LEMMER.lemmatize(word) for word in words]
    words=[word for word in words if len(word)>2]
    return words

import nltk
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(unhelpfulFeatures)

## Creaet vectorizers
#Unigrams + Stopwords CountVectorizer
CV1=CountVectorizer(input='content',analyzer = 'word',stop_words=stopwords,min_df=5,tokenizer=LONG_WORDS)

#ngram_range of c(1, 1) means only unigrams
#c(1, 2) means unigrams and bigrams
#and c(2, 2) means only bigrams.

#Stopwords unigrams & bigram CountVectorizer
CV2=CountVectorizer(input='content',analyzer = 'word',stop_words=stopwords,ngram_range=(1,2),min_df=5,tokenizer=LONG_WORDS)

#Unigrams + Stopwords TF-IDF
CV3=TfidfVectorizer(input='content',analyzer = 'word',stop_words=stopwords,min_df=5,tokenizer=LONG_WORDS)

#Stopwords unigrams + bigram TF-IDF #bigrams only had poor results
CV4=TfidfVectorizer(input='content',analyzer = 'word',stop_words=stopwords,ngram_range=(1,2),min_df=5,tokenizer=LONG_WORDS)

#Unigrams + Stopwords + Lemmer TF-IDF
CV5=TfidfVectorizer(input='content',analyzer = 'word',stop_words=stopwords,min_df=5,tokenizer=MY_LEMMER)

#Stopwords unigrams + bigram +Lemmer TF-IDF
CV6=TfidfVectorizer(input='content',analyzer = 'word',stop_words=stopwords,tokenizer=MY_LEMMER,ngram_range=(1,2))
#Stopwords unigrams + bigram +Stopwords TF-IDF
CV7=TfidfVectorizer(input='content',analyzer = 'word',ngram_range=(1,2),stop_words=stopwords,min_df=5,tokenizer=LONG_WORDS)


# # Multinomial Naive Bayes

# ### CV1
comments_cv1=CV1.fit_transform(x_train)
test_cv1=CV1.transform(x_test)
mnb = MultinomialNB()

mnb.fit(comments_cv1,y_train)

#k-fold cross validtion to measure accuracy of MNB for lemmatization
from sklearn.model_selection import cross_val_score

mnb_cv1score = cross_val_score(mnb, comments_cv1, y_train, cv=5)

# Print the accuracy of each fold:
print(mnb_cv1score)

# Print the mean accuracy of all 5 folds
print(mnb_cv1score.mean())


## Get the features for column names
featuresmnb1=CV1.get_feature_names_out()
featuresmnb1 = pd.DataFrame(comments_cv1.toarray(), columns=featuresmnb1)
#Headline CountVectorizer WC
topfmnb1=featuresmnb1.sum(axis=0).sort_values(ascending=False)
#Wordcloud of top 20 features
featuresCloudmnb1 = WordCloud(colormap='tab20b',background_color="white", max_words=20).generate_from_frequencies(topfmnb1)
plt.figure(figsize=(20,10))
plt.imshow(featuresCloudmnb1, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#Top 20 Features for non-toxic comments
nottoxic_feature_ranks1 = sorted(zip(mnb.feature_log_prob_[0], CV1.get_feature_names_out()))
nottoxic_features1 = nottoxic_feature_ranks1[-20:]
print(nottoxic_features1)

nottoxicfeatures1=[]
nottoxiclogprob1=[]

for item in nottoxic_feature_ranks1[0:20]:
    nottoxicfeatures1.append(item[1])
    nottoxiclogprob1.append(item[0])
cv1topfeat1=pd.DataFrame(nottoxicfeatures1,columns=['Not Toxic Features'])
cv1topfeat1.insert(loc=1,column="Log_Prob",value=nottoxiclogprob1)


#Top 20 Features for toxic comments
toxic_feature_ranks1 = sorted(zip(mnb.feature_log_prob_[1], CV1.get_feature_names()))
toxic_features1 = toxic_feature_ranks1[-20:]
print(toxic_features1)

toxicfeatures1=[]
toxiclogprob1=[]

for item in toxic_feature_ranks1[-20:]:
    toxicfeatures1.append(item[1])
    toxiclogprob1.append(item[0])

cv1topfeat1.insert(loc=1,column="Toxic Features",value=toxicfeatures1)
cv1topfeat1.insert(loc=1,column="Log_Prob_toxic",value=toxiclogprob1)



#Visualization of to 10 features for not toxic and toxic comments
figure1 = plt.figure()
ax1 = figure1.add_axes([0,0,1,1])
ax1.bar(cv1topfeat1['Not Toxic Features'].values, cv1topfeat1['Log_Prob'].values)
plt.xticks(cv1topfeat1['Not Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure2 = plt.figure()
ax2 = figure2.add_axes([0,0,1,1])
ax2.bar(cv1topfeat1['Toxic Features'].values, cv1topfeat1['Log_Prob_toxic'].values)
plt.xticks(cv1topfeat1['Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Toxic Comments')
plt.show()



#confusion matrix for cross-validation TRAINING DATA!
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

mnbtrain_pred1 = cross_val_predict(mnb, comments_cv1, y_train, cv=5)
mnb_train1 = confusion_matrix(y_train, mnbtrain_pred1,labels=[0,1])

#MNB headline heatmap confusion matrix
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbtrainHM6=sns.heatmap(mnb_train1, annot=True, cmap='Spectral',fmt='g')
mnbtrainHM6.set_xticklabels(['Not Toxic','Toxic'])
mnbtrainHM6.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV1 Cross Validated Training Data Confusion Matrix ')
sns.despine()


#confusion matrix for TEST DATA!
#transform test data do not fit it to CountVectorizer!
#This tells you how well the model did on NEW DATA, a 30% model

#now look at recall and precision

mnb_pred1= confusion_matrix(y_test, mnb.predict(test_cv1),labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbHM=sns.heatmap(mnb_pred1, annot=True, cmap='Spectral',fmt='g')
mnbHM.set_xticklabels(['Not Toxic','Toxic'])
mnbHM.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV1 Predicted Comment Toxicity Confusion Matrix')
sns.despine()
plt.show()


from sklearn.metrics import accuracy_score
y_pred1=mnb.fit(comments_cv1, y_train).predict(test_cv1)
accuracy_score(y_test, y_pred1)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred1))


# ### CV2

comments_cv2=CV2.fit_transform(x_train)
test_cv2=CV2.transform(x_test)

mnb.fit(comments_cv2,y_train)

#k-fold cross validtion to measure accuracy of MNB
mnb_cv2score = cross_val_score(mnb, comments_cv2, y_train, cv=5)

# Print the accuracy of each fold:
print(mnb_cv2score)

# Print the mean accuracy of all 5 folds
print(mnb_cv2score.mean())



## Get the features for column names
featuresmnb2=CV2.get_feature_names_out()
featuresmnb2 = pd.DataFrame(comments_cv2.toarray(), columns=featuresmnb2)
#Headline CountVectorizer WC
topfmnb2=featuresmnb2.sum(axis=0).sort_values(ascending=False)
#Wordcloud of top 20 features
featuresCloudmnb2 = WordCloud(colormap='tab20b',background_color="white", max_words=20).generate_from_frequencies(topfmnb2)
plt.figure(figsize=(20,10))
plt.imshow(featuresCloudmnb2, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#Top 20 Features for non-toxic comments
nottoxic_feature_ranks2 = sorted(zip(mnb.feature_log_prob_[0], CV2.get_feature_names_out()))
nottoxic_features2 = nottoxic_feature_ranks2[-20:]
print(nottoxic_features2)

nottoxicfeatures2=[]
nottoxiclogprob2=[]

for item in nottoxic_feature_ranks2[0:20]:
    nottoxicfeatures2.append(item[1])
    nottoxiclogprob2.append(item[0])
cv2topfeat=pd.DataFrame(nottoxicfeatures2,columns=['Not Toxic Features'])
cv2topfeat.insert(loc=1,column="Log_Prob",value=nottoxiclogprob2)



#Top 20 Features for toxic comments
toxic_feature_ranks2 = sorted(zip(mnb.feature_log_prob_[1], CV2.get_feature_names()))
toxic_features2 = toxic_feature_ranks2[-20:]
print(toxic_features2)

toxicfeatures2=[]
toxiclogprob2=[]

for item in toxic_feature_ranks2[-20:]:
    toxicfeatures2.append(item[1])
    toxiclogprob2.append(item[0])

cv2topfeat.insert(loc=1,column="Toxic Features",value=toxicfeatures2)
cv2topfeat.insert(loc=1,column="Log_Prob_toxic",value=toxiclogprob2)



#Visualization of to 20 features for not toxic and toxic comments
figure3 = plt.figure()
ax3 = figure3.add_axes([0,0,1,1])
ax3.bar(cv2topfeat['Not Toxic Features'].values, cv2topfeat['Log_Prob'].values)
plt.xticks(cv2topfeat['Not Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure4 = plt.figure()
ax4 = figure4.add_axes([0,0,1,1])
ax4.bar(cv2topfeat['Toxic Features'].values, cv2topfeat['Log_Prob_toxic'].values)
plt.xticks(cv2topfeat['Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Toxic Comments')
plt.show()



#confusion matrix for cross-validation TRAINING DATA!
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

mnbtrain_pred2 = cross_val_predict(mnb, comments_cv2, y_train, cv=5)
mnb_train2 = confusion_matrix(y_train, mnbtrain_pred2,labels=[0,1])

#MNB headline heatmap confusion matrix
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbtrainHM=sns.heatmap(mnb_train2, annot=True, cmap='Spectral',fmt='g')
mnbtrainHM.set_xticklabels(['Not Toxic','Toxic'])
mnbtrainHM.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV2 Cross Validated Training Data Confusion Matrix ')
sns.despine()
plt.show()



#confusion matrix for TEST DATA!
#transform test data do not fit it to CountVectorizer!
#This tells you how well the model did on NEW DATA, a 30% model

#now look at recall and precision

mnb_pred2= confusion_matrix(y_test, mnb.predict(test_cv2),labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbHM=sns.heatmap(mnb_pred2, annot=True, cmap='Spectral',fmt='g')
mnbHM.set_xticklabels(['Not Toxic','Toxic'])
mnbHM.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV2 Predicted Comment Toxicity Confusion Matrix')
sns.despine()
plt.show()



from sklearn.metrics import accuracy_score
y_pred2=mnb.fit(comments_cv2, y_train).predict(test_cv2)
accuracy_score(y_test, y_pred2)



from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred2))


# ### CV3

comments_cv3=CV3.fit_transform(x_train)
test_cv3=CV3.transform(x_test)

mnb.fit(comments_cv3,y_train)

#k-fold cross validtion to measure accuracy of MNB
mnb_cv3score = cross_val_score(mnb, comments_cv3, y_train, cv=5)

# Print the accuracy of each fold:
print(mnb_cv3score)

# Print the mean accuracy of all 5 folds
print(mnb_cv3score.mean())


## Get the features for column names
featuresmnb3=CV3.get_feature_names_out()
featuresmnb3 = pd.DataFrame(comments_cv3.toarray(), columns=featuresmnb3)
#Headline CountVectorizer WC
topfmnb3=featuresmnb3.sum(axis=0).sort_values(ascending=False)
#Wordcloud of top 20 features
featuresCloudmnb3 = WordCloud(colormap='tab20b',background_color="white", max_words=20).generate_from_frequencies(topfmnb3)
plt.figure(figsize=(20,10))
plt.imshow(featuresCloudmnb3, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#Top 20 Features for non-toxic comments
nottoxic_feature_ranks3 = sorted(zip(mnb.feature_log_prob_[0], CV3.get_feature_names_out()))
nottoxic_features3 = nottoxic_feature_ranks3[-20:]
print(nottoxic_features3)

nottoxicfeatures3=[]
nottoxiclogprob3=[]

for item in nottoxic_feature_ranks3[0:20]:
    nottoxicfeatures3.append(item[1])
    nottoxiclogprob3.append(item[0])
cv3topfeat=pd.DataFrame(nottoxicfeatures3,columns=['Not Toxic Features'])
cv3topfeat.insert(loc=1,column="Log_Prob",value=nottoxiclogprob3)



#Top 20 Features for toxic comments
toxic_feature_ranks3 = sorted(zip(mnb.feature_log_prob_[1], CV3.get_feature_names()))
toxic_features3 = toxic_feature_ranks3[-20:]
print(toxic_features3)

toxicfeatures3=[]
toxiclogprob3=[]

for item in toxic_feature_ranks3[-20:]:
    toxicfeatures3.append(item[1])
    toxiclogprob3.append(item[0])

cv3topfeat.insert(loc=1,column="Toxic Features",value=toxicfeatures3)
cv3topfeat.insert(loc=1,column="Log_Prob_toxic",value=toxiclogprob3)



#Visualization of to 20 features for not toxic and toxic comments
figure5 = plt.figure()
ax5 = figure5.add_axes([0,0,1,1])
ax5.bar(cv3topfeat['Not Toxic Features'].values, cv3topfeat['Log_Prob'].values)
plt.xticks(cv3topfeat['Not Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure6 = plt.figure()
ax6 = figure6.add_axes([0,0,1,1])
ax6.bar(cv3topfeat['Toxic Features'].values, cv3topfeat['Log_Prob_toxic'].values)
plt.xticks(cv3topfeat['Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Toxic Comments')
plt.show()



#confusion matrix for cross-validation TRAINING DATA!
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

mnbtrain_pred3 = cross_val_predict(mnb, comments_cv3, y_train, cv=5)
mnb_train3 = confusion_matrix(y_train, mnbtrain_pred3,labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbtrainHM3=sns.heatmap(mnb_train3, annot=True, cmap='Spectral',fmt='g')
mnbtrainHM3.set_xticklabels(['Not Toxic','Toxic'])
mnbtrainHM3.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV3 Cross Validated Training Data Confusion Matrix ')
sns.despine()
plt.show()


#confusion matrix for TEST DATA!
#transform test data do not fit it to CountVectorizer!
#This tells you how well the model did on NEW DATA

#now look at recall and precision

mnb_pred3= confusion_matrix(y_test, mnb.predict(test_cv3),labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbHM3=sns.heatmap(mnb_pred3, annot=True, cmap='Spectral',fmt='g')
mnbHM3.set_xticklabels(['Not Toxic','Toxic'])
mnbHM3.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV2 Predicted Comment Toxicity Confusion Matrix')
sns.despine()
plt.show()


from sklearn.metrics import accuracy_score
y_pred3=mnb.fit(comments_cv3, y_train).predict(test_cv3)
accuracy_score(y_test, y_pred3)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred3))


# ### CV4

comments_cv4=CV4.fit_transform(x_train)
test_cv4=CV4.transform(x_test)

mnb.fit(comments_cv4,y_train)

#k-fold cross validtion to measure accuracy of MNB
mnb_cv4score = cross_val_score(mnb, comments_cv4, y_train, cv=5)

# Print the accuracy of each fold:
print(mnb_cv4score)

# Print the mean accuracy of all 5 folds
print(mnb_cv4score.mean())



## Get the features for column names
featuresmnb4=CV4.get_feature_names_out()
featuresmnb4 = pd.DataFrame(comments_cv4.toarray(), columns=featuresmnb4)
#Headline CountVectorizer WC
topfmnb4=featuresmnb4.sum(axis=0).sort_values(ascending=False)
#Wordcloud of top 20 features
featuresCloudmnb4 = WordCloud(colormap='tab20b',background_color="white", max_words=20).generate_from_frequencies(topfmnb4)
plt.figure(figsize=(20,10))
plt.imshow(featuresCloudmnb4, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()



#Top 20 Features for non-toxic comments
nottoxic_feature_ranks4 = sorted(zip(mnb.feature_log_prob_[0], CV4.get_feature_names_out()))
nottoxic_features4 = nottoxic_feature_ranks4[-20:]
print(nottoxic_features4)

nottoxicfeatures4=[]
nottoxiclogprob4=[]

for item in nottoxic_feature_ranks4[0:20]:
    nottoxicfeatures4.append(item[1])
    nottoxiclogprob4.append(item[0])
cv4topfeat=pd.DataFrame(nottoxicfeatures4,columns=['Not Toxic Features'])
cv4topfeat.insert(loc=1,column="Log_Prob",value=nottoxiclogprob4)



#Top 20 Features for toxic comments
toxic_feature_ranks4 = sorted(zip(mnb.feature_log_prob_[1], CV4.get_feature_names()))
toxic_features4 = toxic_feature_ranks4[-20:]
print(toxic_features4)

toxicfeatures4=[]
toxiclogprob4=[]

for item in toxic_feature_ranks4[-20:]:
    toxicfeatures4.append(item[1])
    toxiclogprob4.append(item[0])

cv4topfeat.insert(loc=1,column="Toxic Features",value=toxicfeatures4)
cv4topfeat.insert(loc=1,column="Log_Prob_toxic",value=toxiclogprob4)



#Visualization of to 20 features for not toxic and toxic comments
figure7 = plt.figure()
ax7 = figure7.add_axes([0,0,1,1])
ax7.bar(cv4topfeat['Not Toxic Features'].values, cv4topfeat['Log_Prob'].values)
plt.xticks(cv4topfeat['Not Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure8 = plt.figure()
ax8 = figure8.add_axes([0,0,1,1])
ax8.bar(cv4topfeat['Toxic Features'].values, cv4topfeat['Log_Prob_toxic'].values)
plt.xticks(cv4topfeat['Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Toxic Comments')
plt.show()


#confusion matrix for cross-validation TRAINING DATA!
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

mnbtrain_pred4 = cross_val_predict(mnb, comments_cv4, y_train, cv=5)
mnb_train4 = confusion_matrix(y_train, mnbtrain_pred4,labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbtrainHM4=sns.heatmap(mnb_train4, annot=True, cmap='Spectral',fmt='g')
mnbtrainHM4.set_xticklabels(['Not Toxic','Toxic'])
mnbtrainHM4.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV4 Cross Validated Training Data Confusion Matrix ')
sns.despine()



#confusion matrix for TEST DATA!
#transform test data do not fit it to CountVectorizer!
#This tells you how well the model did on NEW DATA

#now look at recall and precision

mnb_pred4= confusion_matrix(y_test, mnb.predict(test_cv4),labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbHM4=sns.heatmap(mnb_pred4, annot=True, cmap='Spectral',fmt='g')
mnbHM4.set_xticklabels(['Not Toxic','Toxic'])
mnbHM4.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV4 Predicted Comment Toxicity Confusion Matrix')
sns.despine()
plt.show()


from sklearn.metrics import accuracy_score
y_pred4=mnb.fit(comments_cv4, y_train).predict(test_cv4)
accuracy_score(y_test, y_pred4)



from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred4))


# ### CV5

comments_cv5=CV5.fit_transform(x_train)
test_cv5=CV5.transform(x_test)

mnb.fit(comments_cv5,y_train)

#k-fold cross validtion to measure accuracy of MNB
mnb_cv5score = cross_val_score(mnb, comments_cv5, y_train, cv=5)

# Print the accuracy of each fold:
print(mnb_cv5score)

# Print the mean accuracy of all 5 folds
print(mnb_cv5score.mean())



## Get the features for column names
featuresmnb5=CV5.get_feature_names_out()
featuresmnb5 = pd.DataFrame(comments_cv5.toarray(), columns=featuresmnb5)
#Headline CountVectorizer WC
topfmnb5=featuresmnb5.sum(axis=0).sort_values(ascending=False)
#Wordcloud of top 20 features
featuresCloudmnb5 = WordCloud(colormap='tab20b',background_color="white", max_words=20).generate_from_frequencies(topfmnb5)
plt.figure(figsize=(20,10))
plt.imshow(featuresCloudmnb5, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()



#Top 20 Features for non-toxic comments
nottoxic_feature_ranks5 = sorted(zip(mnb.feature_log_prob_[0], CV5.get_feature_names_out()))
nottoxic_features5 = nottoxic_feature_ranks5[-20:]
print(nottoxic_features5)

nottoxicfeatures5=[]
nottoxiclogprob5=[]

for item in nottoxic_feature_ranks5[0:20]:
    nottoxicfeatures5.append(item[1])
    nottoxiclogprob5.append(item[0])
cv5topfeat=pd.DataFrame(nottoxicfeatures5,columns=['Not Toxic Features'])
cv5topfeat.insert(loc=1,column="Log_Prob",value=nottoxiclogprob5)



#Top 20 Features for toxic comments
toxic_feature_ranks5 = sorted(zip(mnb.feature_log_prob_[1], CV5.get_feature_names()))
toxic_features5 = toxic_feature_ranks5[-20:]
print(toxic_features5)

toxicfeatures5=[]
toxiclogprob5=[]

for item in toxic_feature_ranks5[-20:]:
    toxicfeatures5.append(item[1])
    toxiclogprob5.append(item[0])

cv5topfeat.insert(loc=1,column="Toxic Features",value=toxicfeatures5)
cv5topfeat.insert(loc=1,column="Log_Prob_toxic",value=toxiclogprob5)


#Visualization of to 20 features for not toxic and toxic comments
figure8 = plt.figure()
ax8 = figure8.add_axes([0,0,1,1])
ax8.bar(cv5topfeat['Not Toxic Features'].values, cv5topfeat['Log_Prob'].values)
plt.xticks(cv5topfeat['Not Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure9 = plt.figure()
ax9 = figure9.add_axes([0,0,1,1])
ax9.bar(cv5topfeat['Toxic Features'].values, cv5topfeat['Log_Prob_toxic'].values)
plt.xticks(cv5topfeat['Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Toxic Comments')
plt.show()



#confusion matrix for cross-validation TRAINING DATA!
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

mnbtrain_pred5 = cross_val_predict(mnb, comments_cv5, y_train, cv=5)
mnb_train5 = confusion_matrix(y_train, mnbtrain_pred5,labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbtrainHM5=sns.heatmap(mnb_train5, annot=True, cmap='Spectral',fmt='g')
mnbtrainHM5.set_xticklabels(['Not Toxic','Toxic'])
mnbtrainHM5.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV3 Cross Validated Training Data Confusion Matrix ')
sns.despine()


#confusion matrix for TEST DATA!
#transform test data do not fit it to CountVectorizer!
#This tells you how well the model did on NEW DATA

#now look at recall and precision

mnb_pred5= confusion_matrix(y_test, mnb.predict(test_cv5),labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbHM5=sns.heatmap(mnb_pred5, annot=True, cmap='Spectral',fmt='g')
mnbHM5.set_xticklabels(['Not Toxic','Toxic'])
mnbHM5.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV5 Predicted Comment Toxicity Confusion Matrix')
sns.despine()
plt.show()



from sklearn.metrics import accuracy_score
y_pred5=mnb.fit(comments_cv5, y_train).predict(test_cv5)
accuracy_score(y_test, y_pred5)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred5))


# ### CV6


comments_cv6=CV6.fit_transform(x_train)
test_cv6=CV6.transform(x_test)

mnb.fit(comments_cv6,y_train)

#k-fold cross validtion to measure accuracy of MNB
mnb_cv6score = cross_val_score(mnb, comments_cv6, y_train, cv=5)

# Print the accuracy of each fold:
print(mnb_cv6score)

# Print the mean accuracy of all 5 folds
print(mnb_cv6score.mean())



## Get the features for column names
featuresmnb6=CV6.get_feature_names_out()
featuresmnb6 = pd.DataFrame(comments_cv6.toarray(), columns=featuresmnb6)
#Headline CountVectorizer WC
topfmnb6=featuresmnb6.sum(axis=0).sort_values(ascending=False)
#Wordcloud of top 20 features
featuresCloudmnb6 = WordCloud(colormap='tab20b',background_color="white", max_words=20).generate_from_frequencies(topfmnb6)
plt.figure(figsize=(20,10))
plt.imshow(featuresCloudmnb6, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()



#Top 20 Features for non-toxic comments
nottoxic_feature_ranks6 = sorted(zip(mnb.feature_log_prob_[0], CV6.get_feature_names_out()))
nottoxic_features6 = nottoxic_feature_ranks6[-20:]
print(nottoxic_features6)

nottoxicfeatures6=[]
nottoxiclogprob6=[]

for item in nottoxic_feature_ranks6[0:20]:
    nottoxicfeatures6.append(item[1])
    nottoxiclogprob6.append(item[0])
cv6topfeat=pd.DataFrame(nottoxicfeatures6,columns=['Not Toxic Features'])
cv6topfeat.insert(loc=1,column="Log_Prob",value=nottoxiclogprob6)



#Top 20 Features for toxic comments
toxic_feature_ranks6 = sorted(zip(mnb.feature_log_prob_[1], CV6.get_feature_names_out()))
toxic_features6 = toxic_feature_ranks6[-20:]
print(toxic_features6)

toxicfeatures6=[]
toxiclogprob6=[]

for item in toxic_feature_ranks6[-20:]:
    toxicfeatures6.append(item[1])
    toxiclogprob6.append(item[0])

cv6topfeat.insert(loc=1,column="Toxic Features",value=toxicfeatures6)
cv6topfeat.insert(loc=1,column="Log_Prob_toxic",value=toxiclogprob6)



#Visualization of to 20 features for not toxic and toxic comments
figure9 = plt.figure()
ax9 = figure9.add_axes([0,0,1,1])
ax9.bar(cv6topfeat['Not Toxic Features'].values, cv6topfeat['Log_Prob'].values)
plt.xticks(cv6topfeat['Not Toxic Features'], rotation = 'vertical')
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure10 = plt.figure()
ax10 = figure10.add_axes([0,0,1,1])
ax10.bar(cv6topfeat['Toxic Features'].values, cv6topfeat['Log_Prob_toxic'].values)
plt.xticks(cv6topfeat['Toxic Features'], rotation = 'vertical')
plt.title('Most Indicative Words for Toxic Comments')
plt.show()



#confusion matrix for cross-validation TRAINING DATA!
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

mnbtrain_pred6 = cross_val_predict(mnb, comments_cv6, y_train, cv=5)
mnb_train6 = confusion_matrix(y_train, mnbtrain_pred6,labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbtrainHM6=sns.heatmap(mnb_train6, annot=True, cmap='Spectral',fmt='g')
mnbtrainHM6.set_xticklabels(['Not Toxic','Toxic'])
mnbtrainHM6.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV6 Cross Validated Training Data Confusion Matrix ')
sns.despine()



#confusion matrix for TEST DATA!
#transform test data do not fit it to CountVectorizer!
#This tells you how well the model did on NEW DATA

#now look at recall and precision

mnb_pred6= confusion_matrix(y_test, mnb.predict(test_cv6),labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbHM6=sns.heatmap(mnb_pred6, annot=True, cmap='Spectral',fmt='g')
mnbHM6.set_xticklabels(['Not Toxic','Toxic'])
mnbHM6.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV6 Predicted Comment Toxicity Confusion Matrix')
sns.despine()
plt.show()


from sklearn.metrics import accuracy_score
y_pred6=mnb.fit(comments_cv6, y_train).predict(test_cv6)
accuracy_score(y_test, y_pred6)



from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred6))


# ## CV7


comments_cv7=CV7.fit_transform(x_train)
test_cv7=CV7.transform(x_test)

mnb.fit(comments_cv7,y_train)

#k-fold cross validtion to measure accuracy of MNB
mnb_cv7score = cross_val_score(mnb, comments_cv7, y_train, cv=5)

# Print the accuracy of each fold:
print(mnb_cv7score)

# Print the mean accuracy of all 5 folds
print(mnb_cv7score.mean())


## Get the features for column names
featuresmnb7=CV7.get_feature_names_out()
featuresmnb7 = pd.DataFrame(comments_cv7.toarray(), columns=featuresmnb7)
#Headline CountVectorizer WC
topfmnb7=featuresmnb7.sum(axis=0).sort_values(ascending=False)
#Wordcloud of top 20 features
featuresCloudmnb7 = WordCloud(colormap='tab20b',background_color="white", max_words=20).generate_from_frequencies(topfmnb7)
plt.figure(figsize=(20,10))
plt.imshow(featuresCloudmnb7, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#Top 20 Features for non-toxic comments
nottoxic_feature_ranks7 = sorted(zip(mnb.feature_log_prob_[0], CV7.get_feature_names_out()))
nottoxic_features7 = nottoxic_feature_ranks7[-20:]
print(nottoxic_features7)

nottoxicfeatures7=[]
nottoxiclogprob7=[]

for item in nottoxic_feature_ranks7[0:20]:
    nottoxicfeatures7.append(item[1])
    nottoxiclogprob7.append(item[0])
cv7topfeat=pd.DataFrame(nottoxicfeatures7,columns=['Not Toxic Features'])
cv7topfeat.insert(loc=1,column="Log_Prob",value=nottoxiclogprob7)



#Top 20 Features for toxic comments
toxic_feature_ranks7 = sorted(zip(mnb.feature_log_prob_[1], CV7.get_feature_names()))
toxic_features7 = toxic_feature_ranks7[-20:]
print(toxic_features7)

toxicfeatures7=[]
toxiclogprob7=[]

for item in toxic_feature_ranks7[-20:]:
    toxicfeatures7.append(item[1])
    toxiclogprob7.append(item[0])

cv7topfeat.insert(loc=1,column="Toxic Features",value=toxicfeatures7)
cv7topfeat.insert(loc=1,column="Log_Prob_toxic",value=toxiclogprob7)



#Visualization of to 20 features for not toxic and toxic comments
figure10 = plt.figure()
ax10 = figure10.add_axes([0,0,1,1])
ax10.bar(cv7topfeat['Not Toxic Features'].values, cv7topfeat['Log_Prob'].values)
plt.xticks(cv7topfeat['Not Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure11 = plt.figure()
ax11 = figure11.add_axes([0,0,1,1])
ax11.bar(cv7topfeat['Toxic Features'].values, cv7topfeat['Log_Prob_toxic'].values)
plt.xticks(cv7topfeat['Toxic Features'], rotation = 45)
plt.title('Most Indicative Words for Toxic Comments')
plt.show()



#confusion matrix for cross-validation TRAINING DATA!
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

mnbtrain_pred7 = cross_val_predict(mnb, comments_cv7, y_train, cv=5)
mnb_train7 = confusion_matrix(y_train, mnbtrain_pred7,labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbtrainHM7=sns.heatmap(mnb_train7, annot=True, cmap='Spectral',fmt='g')
mnbtrainHM7.set_xticklabels(['Not Toxic','Toxic'])
mnbtrainHM7.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV7 Cross Validated Training Data Confusion Matrix ')
sns.despine()


#confusion matrix for TEST DATA!
#transform test data do not fit it to CountVectorizer!
#This tells you how well the model did on NEW DATA

#now look at recall and precision

mnb_pred7= confusion_matrix(y_test, mnb.predict(test_cv7),labels=[0,1])

#MNB headline heatmap confusion matrix
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(font_scale=2)
mnbHM7=sns.heatmap(mnb_pred7, annot=True, cmap='Spectral',fmt='g')
mnbHM7.set_xticklabels(['Not Toxic','Toxic'])
mnbHM7.set_yticklabels(['Not Toxic','Toxic'])
plt.title('MNB_CV7 Predicted Comment Toxicity Confusion Matrix')
sns.despine()
plt.show()



from sklearn.metrics import accuracy_score
y_pred7=mnb.fit(comments_cv7, y_train).predict(test_cv7)
accuracy_score(y_test, y_pred7)



from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred7))

##############################################################
##
##    Analysis - KMeans Clustering with Additional Feature Removal
##
##############################################################


from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans


#Comment Silhouette
visualizer= KElbowVisualizer(KMeans(), k=(2,5), metric='silhouette', timings=False)
visualizer.fit(comments_cv1.toarray())# convert sparse matrix to dense data and fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data



kmeans=KMeans(n_clusters=2,max_iter=100,n_init=1)
kmeans.fit(comments_cv1)



predictedcluster=kmeans.predict(comments_cv1)


df = pd.DataFrame()
df['Comments']=x_train
df['Label']=y_train
df["CV1_Cluster"]=predictedcluster
df


#Predicted Cluster Distribution
#set seaborn plotting aesthetics
sns.set_style('darkgrid')
sns.set_palette('Set2')
topicdist=sns.countplot(data=df, x='CV1_Cluster')

plt.title('Comment Count by Predicted Cluster')
plt.xlabel('Predicted Cluster')
plt.ylabel('Number of Comments')
sns.despine()
plt.show()


# ## CV2

#Comment Silhouette
visualizer= KElbowVisualizer(KMeans(), k=(2,5), metric='silhouette', timings=False)
visualizer.fit(comments_cv2.toarray())# convert sparse matrix to dense data and fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


kmeans=KMeans(n_clusters=2,max_iter=100,n_init=1)
kmeans.fit(comments_cv2)
predictedcluster2=kmeans.predict(comments_cv2)


df["CV2_Cluster"]=predictedcluster2
df


#Predicted Cluster Distribution
#set seaborn plotting aesthetics
sns.set_style('darkgrid')
sns.set_palette('Set2')
topicdist=sns.countplot(data=df, x='CV2_Cluster')

plt.title('Comment Count by Predicted Cluster')
plt.xlabel('Predicted Cluster')
plt.ylabel('Number of Comments')
sns.despine()
plt.show()


# ## CV3

#Comment Silhouette
visualizer= KElbowVisualizer(KMeans(), k=(2,5), metric='silhouette', timings=False)
visualizer.fit(comments_cv3.toarray())# convert sparse matrix to dense data and fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data



kmeans=KMeans(n_clusters=2,max_iter=100,n_init=1)
kmeans.fit(comments_cv3)
predictedcluster3=kmeans.predict(comments_cv3)


df["CV3_Cluster"]=predictedcluster3
df



#Predicted Cluster Distribution
#set seaborn plotting aesthetics
sns.set_style('darkgrid')
sns.set_palette('Set2')
sns.countplot(data=df, x='CV3_Cluster')

plt.title('Comment Count by Predicted Cluster')
plt.xlabel('Predicted Cluster')
plt.ylabel('Number of Comments')
sns.despine()
plt.show()


# ## CV4

#Comment Silhouette
visualizer= KElbowVisualizer(KMeans(), k=(2,5), metric='silhouette', timings=False)
visualizer.fit(comments_cv4.toarray())# convert sparse matrix to dense data and fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


kmeans=KMeans(n_clusters=3,max_iter=100,n_init=1)
kmeans.fit(comments_cv4)
predictedcluster4=kmeans.predict(comments_cv4)


df["CV4_Cluster"]=predictedcluster4
df


#Predicted Cluster Distribution
#set seaborn plotting aesthetics
sns.set_style('darkgrid')
sns.set_palette('Set2')
sns.countplot(data=df, x='CV4_Cluster')

plt.title('Comment Count by Predicted Cluster')
plt.xlabel('Predicted Cluster')
plt.ylabel('Number of Comments')
sns.despine()
plt.show()


#What is CV4 Clustering referring to?
#Check central of gravity of clusters and print feature forms

print('CV4 Cluster centroids: \n')
order_centroids4=kmeans.cluster_centers_.argsort()[:,::-1]
terms4=CV4.get_feature_names()

for i in range(3):
    print('Cluster %d:' % i)
    for j in order_centroids4[i, :10]:
        print(' %s' % terms4[j])
    print('------------')


#word cloud for most common features in Kmeans
# Output cluster results to a csv file
CV4fclusters=df.groupby('CV4_Cluster')
for Cluster in CV4fclusters.groups:
    f=open('cv4fCluster'+str(Cluster)+'.csv','w') #create a csv file
    data=CV4fclusters.get_group(Cluster)[['Label','Comments']] #include columns of interest
    f.write(data.to_csv(index=True))
    f.close()

#PREDICTED Cluster 0 Wordcloud
#Load Predicted Cluster0 as pandas df
PC0=pd.read_csv('cv4fCluster0.csv')

#create text list of all words in Headline cluster 0
PC0text="".join(Comments for Comments in PC0.Comments.astype(str))

#Generate word cloud
PC0_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC0_WC.generate(PC0text)
plt.figure(figsize=(20,10))
plt.imshow(PC0_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#PREDICTED Cluster 1 Wordcloud
#Load Predicted Cluster1 as pandas df
PC1=pd.read_csv('cv4fCluster1.csv')

#create text list of all words in Headline cluster 1
PC1text="".join(Comments for Comments in PC1.Comments.astype(str))

#Generate word cloud
PC1_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC1_WC.generate(PC1text)
plt.figure(figsize=(20,10))
plt.imshow(PC1_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

#PREDICTED Cluster 2 Wordcloud
#Load Predicted Cluster1 as pandas df
PC2=pd.read_csv('cv4fCluster2.csv')

#create text list of all words in Headline cluster 1
PC2text="".join(Comments for Comments in PC2.Comments.astype(str))

#Generate word cloud
PC2_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC2_WC.generate(PC2text)
plt.figure(figsize=(20,10))
plt.imshow(PC2_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


# ## CV5

#Comment Silhouette
visualizer= KElbowVisualizer(KMeans(), k=(2,5), metric='silhouette', timings=False)
visualizer.fit(comments_cv5.toarray())# convert sparse matrix to dense data and fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


kmeans=KMeans(n_clusters=2,max_iter=100,n_init=1)
kmeans.fit(comments_cv5)
predictedcluster5=kmeans.predict(comments_cv5)


df["CV5_Cluster"]=predictedcluster5
df


#Predicted Cluster Distribution
#set seaborn plotting aesthetics
sns.set_style('darkgrid')
sns.set_palette('Set2')
sns.countplot(data=df, x='CV5_Cluster')

plt.title('Comment Count by Predicted Cluster')
plt.xlabel('Predicted Cluster')
plt.ylabel('Number of Comments')
sns.despine()
plt.show()



#What is CV5 Clustering referring to?
#Check central of gravity of clusters and print feature forms

print('CV5 Cluster centroids: \n')
order_centroids5=kmeans.cluster_centers_.argsort()[:,::-1]
terms5=CV5.get_feature_names()

for i in range(2):
    print('Cluster %d:' % i)
    for j in order_centroids5[i, :10]:
        print(' %s' % terms5[j])
    print('------------')



#word cloud for most common features in Kmeans
# Output cluster results to a csv file
CV5fclusters=df.groupby('CV5_Cluster')
for Cluster in CV5fclusters.groups:
    f=open('cv5fCluster'+str(Cluster)+'.csv','w') #create a csv file
    data=CV5fclusters.get_group(Cluster)[['Label','Comments']] #include columns of interest
    f.write(data.to_csv(index=True))
    f.close()

#PREDICTED Cluster 0 Wordcloud
#Load Predicted Cluster0 as pandas df
PC0=pd.read_csv('cv5fCluster0.csv')

#create text list of all words in Headline cluster 0
PC0text="".join(Comments for Comments in PC0.Comments.astype(str))

#Generate word cloud
PC0_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC0_WC.generate(PC0text)
plt.figure(figsize=(20,10))
plt.imshow(PC0_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#PREDICTED Cluster 1 Wordcloud
#Load Predicted Cluster1 as pandas df
PC1=pd.read_csv('cv5fCluster1.csv')

#create text list of all words in Headline cluster 1
PC1text="".join(Comments for Comments in PC1.Comments.astype(str))

#Generate word cloud
PC1_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC1_WC.generate(PC1text)
plt.figure(figsize=(20,10))
plt.imshow(PC1_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

# cluster 0 is not toxic, 1 is toxic

#Accuracy of KMeans clustering on Training Data
incorrectmatches= len(df.loc[df.Label != df.CV5_Cluster])
correctmatches= len(df.loc[df.Label == df.CV5_Cluster])
KMeansAccuracy=correctmatches/8904
KMeansAccuracy #Not good but the best out of all 7 vectorized texts


#Prediction Results on Train Data
y=CV5.transform(x_test)


testcluster=kmeans.predict(y)
testclusterdf = pd.DataFrame()
testclusterdf['Comments']=x_test
testclusterdf['Label']=y_test
testclusterdf["CV5_PredCluster"]=testcluster
testclusterdf


#Accuracy of KMeans clustering Test Data
wrong= len(testclusterdf.loc[testclusterdf.Label != testclusterdf.CV5_PredCluster])
correct= len(testclusterdf.loc[testclusterdf.Label == testclusterdf.CV5_PredCluster])
KMeansTestAccuracy=correct/3816
KMeansTestAccuracy #Not good but the best out of all 7 vectorized texts


# ## CV6
#Comment Silhouette
visualizer= KElbowVisualizer(KMeans(), k=(2,5), metric='silhouette', timings=False)
visualizer.fit(comments_cv6.toarray())# convert sparse matrix to dense data and fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data



#Predicted Cluster Distribution
#set seaborn plotting aesthetics
sns.set_style('darkgrid')
sns.set_palette('Set2')
sns.countplot(data=df, x='CV6_Cluster')

plt.title('Comment Count by Predicted Cluster')
plt.xlabel('Predicted Cluster')
plt.ylabel('Number of Comments')
sns.despine()
plt.show()


# ## CV7

#Comment Silhouette
visualizer= KElbowVisualizer(KMeans(), k=(2,5), metric='silhouette', timings=False)
visualizer.fit(comments_cv7.toarray())# convert sparse matrix to dense data and fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


kmeans=KMeans(n_clusters=2,max_iter=100,n_init=1)
kmeans.fit(comments_cv7)
predictedcluster7=kmeans.predict(comments_cv7)


df["CV7_Cluster"]=predictedcluster7
df


#Predicted Cluster Distribution
#set seaborn plotting aesthetics
sns.set_style('darkgrid')
sns.set_palette('Set2')
sns.countplot(data=df, x='CV7_Cluster')

plt.title('Comment Count by Predicted Cluster')
plt.xlabel('Predicted Cluster')
plt.ylabel('Number of Comments')
sns.despine()
plt.show()


#What is CV7 Clustering referring to?
#Check central of gravity of clusters and print feature forms

print('CV7 Cluster centroids: \n')
order_centroids7=kmeans.cluster_centers_.argsort()[:,::-1]
terms7=CV7.get_feature_names()

for i in range(2):
    print('Cluster %d:' % i)
    for j in order_centroids7[i, :10]:
        print(' %s' % terms7[j])
    print('------------')



#word cloud for most common features in Kmeans
# Output cluster results to a csv file
CV7fclusters=df.groupby('CV7_Cluster')
for Cluster in CV7fclusters.groups:
    f=open('cv7fCluster'+str(Cluster)+'.csv','w') #create a csv file
    data=CV7fclusters.get_group(Cluster)[['Label','Comments']] #include columns of interest
    f.write(data.to_csv(index=True))
    f.close()

#PREDICTED Cluster 0 Wordcloud
#Load Predicted Cluster0 as pandas df
PC0=pd.read_csv('cv7fCluster0.csv')

#create text list of all words in Headline cluster 0
PC0text="".join(Comments for Comments in PC0.Comments.astype(str))

#Generate word cloud
PC0_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC0_WC.generate(PC0text)
plt.figure(figsize=(20,10))
plt.imshow(PC0_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


#PREDICTED Cluster 1 Wordcloud
#Load Predicted Cluster1 as pandas df
PC1=pd.read_csv('cv7fCluster1.csv')

#create text list of all words in Headline cluster 1
PC1text="".join(Comments for Comments in PC1.Comments.astype(str))

#Generate word cloud
PC1_WC=WordCloud(colormap='tab20b',background_color='white',max_words=20)
PC1_WC.generate(PC1text)
plt.figure(figsize=(20,10))
plt.imshow(PC1_WC, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

# cluster 0 is not toxic, 1 is toxic


#Accuracy of KMeans clustering on Training Data
incorrectmatches= len(df.loc[df.Label != df.CV7_Cluster])
correctmatches= len(df.loc[df.Label == df.CV7_Cluster])
KMeansAccuracy=correctmatches/8904
KMeansAccuracy #Not good but the best out of all 7 vectorized texts


#Prediction Results on Train Data
y=CV7.transform(x_test)


testcluster=kmeans.predict(y)
testclusterdf = pd.DataFrame()
testclusterdf['Comments']=x_test
testclusterdf['Label']=y_test
testclusterdf["CV7_PredCluster"]=testcluster
testclusterdf


#Accuracy of KMeans clustering Test Data
wrong= len(testclusterdf.loc[testclusterdf.Label != testclusterdf.CV7_PredCluster])
correct= len(testclusterdf.loc[testclusterdf.Label == testclusterdf.CV7_PredCluster])
KMeansTestAccuracy=correct/3816
KMeansTestAccuracy #Not good but the best out of all 7 vectorized texts


print(df['Label'].value_counts())
print(df['CV1_Cluster'].value_counts())
print(df['CV2_Cluster'].value_counts())
print(df['CV3_Cluster'].value_counts())
print(df['CV4_Cluster'].value_counts())
print(df['CV5_Cluster'].value_counts())
print(df['CV6_Cluster'].value_counts())
print(df['CV7_Cluster'].value_counts())


# # Visualizations

from matplotlib.ticker import PercentFormatter

mnbdf2 = pd.DataFrame({'Vectorizer Type': ['CV1', 'CV2', 'CV3', 'CV4', 'CV5',
                           'CV6', 'CV7'],
                   'Train Accuracy': [.820, .817, .826, .821, .829,
                                 .827, .821],
                   'Predictive Accuracy': [.822,.815,.820,.816,.821,.818,.816]})
x_=mnbdf2.columns[0]
y_=mnbdf2.columns[1]
y2_=mnbdf2.columns[2]

data1=mnbdf2[[x_,y_]]
data2=mnbdf2[[x_,y2_]]


#Predicted Cluster Distribution
#set seaborn plotting aesthetics
plt.figure(figsize=(15,8))
sns.set_style('darkgrid')
sns.set_palette('Set2')
ax=sns.barplot(data=data1,x=x_,y=y_)
width_scale=0.45
for bar in ax.containers[0]:
    bar.set_width(bar.get_width()*width_scale)

ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set(ylim=(.75, .83))
plt.ylabel('Accuracy')

ax2=ax.twinx()
sns.set_style('darkgrid')
sns.set_palette('Set2')
sns.barplot(data=data2, x=x_,y=y2_,ax=ax2)
for bar in ax2.containers[0]:
    x=bar.get_x()
    w=bar.get_width()
    bar.set_x(x+w*(1-width_scale))
    bar.set_width(w*width_scale)

ax2.yaxis.set_major_formatter(PercentFormatter(1))
ax2.set(ylim=(.75, .83))

plt.title('Naive Bayes Results Overview: Social Group Features Removed')
plt.xlabel('Vectorizer Type')
plt.ylabel('') #hides label on the right
ax2.axes.yaxis.set_ticklabels([]) #removes additional y scale that is the same on the left

sns.despine()
plt.show()



##############################################################
##
##    Analysis - Support Vector Machines with Additional Feature Removal
##
##############################################################

address = "toxic.csv"
toxicdf = pd.read_csv(address, sep = ',')

comments = toxicdf['Comments'].values
labels = toxicdf['Labels'].values

comments_new = [c.replace('\n', '') for c in comments]

svmVect=CountVectorizer(input = 'content',
                        stop_words='english',
                        min_df = 5,
                        max_df = 25)

commentVectors = svmVect.fit_transform(comments_new)
features = svmVect.get_feature_names_out()
print(features)
print(len(features))


svmToxicdf = pd.DataFrame(commentVectors.toarray(), columns= features)
svmToxicdf.insert(loc = 0, column = 'Label', value = labels)
print(svmToxicdf.shape)

#An updated unhelpful feature list was used for SVM models
unhelpfulFeatures_updated = ['000000', '084080', '1000', '101', '102', '103', '104',
                     '105', '106', '107', '108', '109', '110', '111', '112',
                     '113', '114', '115', '116', '117', '118', '1185', '119',
                     '11th', '120', '121', '122', '123', '124', '125', '127',
                     '128', '12th', '130', '131', '132', '133', '134', '135',
                     '136', '137', '138', '140', '141', '142', '143', '144',
                     '145', '146', '147', '148', '149', '14th', '150', '151',
                     '152', '153', '154', '155', '156', '157', '158', '159',
                     '15th', '160', '161', '162', '163', '164', '165', '166',
                     '167', '168', '169', '170', '171', '172', '174', '175',
                     '176', '177', '178', '179', '180', '181', '182', '183',
                     '184', '185', '186', '187', '188', '189', '18th', '190',
                     '191', '1911', '1912', '193', '1930', '194', '1940',
                     '1944', '1945', '195', '1950', '196', '1967', '1968',
                     '197', '1971', '1975', '1978', '1979', '198', '1980',
                     '1980s', '1981', '1982', '1984', '1985', '1986', '1989',
                     '199', '1990', '1990s', '1991', '1992', '1993', '1994',
                     '1995', '1996', '1997', '1998', '1999', '19th', '1px',
                     '1st', '2000', '2001', '2002', '201', '2015', '2016', '202',
                     '203', '204', '205', '206', '207', '208', '209', '20th',
                     '210', '211', '212', '213', '214', '215', '216', '217',
                     '218', '219', '21st', '220', '221', '222', '223', '224',
                     '225', '226', '227', '228', '229', '230', '231', '232',
                     '233', '234', '235', '236', '237', '238', '239', '240',
                     '241', '242', '243', '244', '245', '246', '247', '248',
                     '249', '250', '251', '252', '253', '254', '255',
                     '27_noticeboard', '2nd', '300', '32', '34', '38', '3d',
                     '3rd', '400', '41', '43', '47', '49', '4chan', '4th', '500',
                     '52', '56', '5th', '600', '61', '63', '6th', '73', '74',
                     '77', '79', '7th', '800', '83', '84', '85', '87', '88',
                     '89', '8th', '900', '91', '93', '95', '96', '__', 'a7',
                     'ab', 'abbey', 'abbreviations', 'abc','abraham', 'abstract',
                     'abu', 'ac', 'academia', 'academics', 'academy', 'aclu',
                     'acronym', 'acted', 'actor', 'actors', 'actress', 'adam',
                     'adams', 'additionally', 'administration', 'admission',
                     'adolescent','adolf', 'ads', 'adult', 'adults', 'advertise',
                     'advertisement', 'ae', 'aeropagitica', 'afds', 'affix',
                     'affiliated', 'afghanistan', 'africa', 'african',  'aforementioned',
                     'africa', 'afternoon', 'aged', 'agencies', 'agency',
                     'agent', 'agents', 'ages', 'agf', 'ahem', 'aircraft', 'airport',
                     'aiv', 'aka', 'akin', 'alabama', 'alan', 'alas', 'albania',
                     'albanian', 'albanians', 'albert', 'albums', 'alcohol', 'alex',
                     'alexander', 'ali', 'alien', 'allah', 'allen', 'alumni',
                     'ama', 'amazon', 'ancestor', 'ancestors', 'ancestral',
                     'ancestry', 'anderson', 'andrew', 'andy', 'angel', 'angeles',
                     'angle', 'anglo', 'animal', 'animation', 'ann', 'anna', 'announce',
                     'announced', 'announcement', 'annual', 'anon', 'anthony',
                     'antics', 'antisemetic', 'antisemitsm', 'anus', 'aol',
                     'aprtheid', 'ape', 'apes', 'apple', 'applicable', 'application',
                     'application', 'applications', 'applied', 'applies', 'applying',
                     'appointed', 'approximately', 'apr', 'apt', 'ar', 'architecture',
                     'archived', 'archiving', 'article', 'wikipedia', 'arizona', 'armenia',
                     'armenian', 'armenians', 'arms', 'armstrong', 'asia', 'asian',
                     'asians', 'assyrian','atheist', 'b4', "audio", 'audience', 'audio', 'aug', 'aussie',
                     'australian', 'australians', 'austrian', 'auto', 'autobiography',
                     'automated', 'automatic', 'avatar', 'aviation', 'axis', 'az',
                     'azerbaijan', 'azerbaijani', 'ba', 'babe', 'babies', 'badge',
                     'bar', 'barn', 'barnstars', 'base', 'baseball', 'bangladesh',
                     'bank', 'banks', 'banner', 'banter', 'basketball', 'bass',
                     'bat', 'batch', 'batman', 'bats', 'bay', 'bbc', 'bc', 'beach',
                     'bear', 'bears', 'beatles', 'becasue', 'beck', 'beckjord',
                     'becouse', 'becuase', 'bed', 'bee', 'begins', 'begun', 'bent',
                     'bernard', 'bi', 'bibliography', 'bits', 'blacks', 'blah',
                     'blanked', 'blanket', 'blofeld', 'blow', 'blowing', 'bnp',
                     'boards', 'boilerplate', 'bollocks', 'bone', 'boobs', 'border',
                     'bosnia', 'boston', 'bots', 'bottle', 'brazil', 'brd', 'breakfast',
                     'breast', 'breasts', 'brian', 'britain', 'britannica', 'brits',
                     'broadcast', 'brother', 'brothers', 'brown', 'browser', 'bruce',
                     'bucket', 'buck', 'bud', 'buddhist', 'bug', 'bugs', 'buildings',
                     'bulk', 'bunchofgrapes', 'bureaucratic', 'bureaucrats', 'bus',
                     "bytes", 'byzantine', 'ca', 'cabal', 'cad', 'caesar', 'cake',
                     'calendar', 'california', 'cambridge', 'camera', 'camp', 'campaigns',
                     'camps', 'campus', 'canada', 'canadian', 'canadians', 'cancer',
                     'candidate', 'candidates', 'cannon', 'canon', 'capital',
                     'capitalized', 'caps', 'captain', 'caption', 'car', 'carbon',
                     'card', 'carl', 'carolina', 'carrier', 'carrots', 'cars',
                     'cartoon', 'cast', 'casting', 'castro', 'catholics', 'cats',
                     'caucasian', 'ccp', 'cd', 'ce', 'celebrity', 'cell', 'cellpadding',
                     'cells', 'cellspacing', 'celtic', 'cena', 'census', 'cent',
                     'central', 'centra', 'cents', 'centuries', 'ceo', 'cf', 'cfd',
                     'cgi', 'chair', 'chain', 'champion', 'champions', 'championship',
                     'chan', 'channel', 'channels', 'chapter', 'characteristic',
                     'characterized', "characteristics", 'charles', 'chart', 'charts',
                     'chase', 'chauvinist', 'chavez', 'checks', 'checkuser', 'cheek',
                     'cheese', 'chemical', 'cherry', 'chess', 'chest', 'chicago',
                     'chicken', 'chief', 'childhood', 'chip', 'chocolate', 'chris',
                     'chriso', 'christianity', 'christians', 'celebrity', 'cell',
                     'ccp', 'cd', 'ce', 'cellpadding', 'cells', 'cellspacing',
                     'celtic', 'cena', 'cent', 'centered', 'central', 'centre',
                     'cents', 'centuries', 'ceo', 'cf', 'cfd', 'cgi', 'chair',
                     'chan', 'channel', 'champion', 'championship', 'chan',
                     'channel', 'channels', 'chapter', 'checks', 'cheek', 'cheese',
                     'chemical', 'cherry', 'chess', 'chest', 'chicago', 'chief',
                     'christmas', 'christopher', 'chuck', 'churches', 'churchill',
                     'ck', 'clan', 'classed', 'classes', 'classic', 'classical',
                     'clique', 'closely', 'closer', 'closest', 'closing', 'clothes',
                     'cloud', 'clouds', 'clown', 'clowns', 'clubs', 'cnn', 'co2',
                     'coal', 'coast', 'codes', 'coding', 'coffee', 'coi', 'coin',
                     'coldplay', 'colleagues', 'collect', 'collecting', 'collective',
                     'colors', 'colour', 'colspan', 'columbia', 'column', 'comics',
                     'comma', 'commas', 'commander', 'compiled', 'concensus', 'conception',
                     'commie', 'commies', 'comission', 'commonwealth', 'communism',
                     "communists", 'computers', 'computing', 'concert', 'condoms',
                     'conference', 'constitution', 'constitutional', 'construction',
                     'constructed', 'cont', "cook", "cookie", 'cooper', 'cop',
                     'copies', 'cops', 'copyedit', 'copyediting', 'copyrights',
                     'copyvio', 'corbett', 'core', 'coren', 'corner', 'corporation',
                     'corps', 'corpse', 'counted', 'counter', 'counties', 'counting',
                     'countless', 'counts', 'countless', 'county', 'coup', 'courts',
                     'cousin', 'cow', 'cowboy', 'coz', 'crack', 'crackpot', 'craig',
                     'crank', 'creek', 'cretin', 'crew', 'cricket', 'crimea','croatia',
                     'croatian', 'crock', 'crowd', 'crown', 'crusade', 'crusades',
                     'crysal', 'cs', 'csd', 'cuba', 'cuban', 'cult', 'cultures',
                     'cup', 'cur', 'currency', 'curtain', 'cus', 'customers',
                     'cuz', 'cyber', 'cycle', 'czech', 'dab', 'dad', 'daft', 'dan',
                     'daniel', 'danny', 'darren', 'darwin', 'dash', 'dat', 'database',
                     'dave', 'daughter', 'davis', 'dawn', 'dc', 'deaf', 'deals', 'dean',
                     'dec', 'decade', 'decades', 'decausa', 'dem', 'demo', 'democracy',
                     'democrat', 'democratic', 'democrats', 'demographic', 'demographics',
                     'denmark', 'dennis', 'departure', 'dialogue', 'dictator', 'dictatorship',
                     'dictionaries', 'diego', 'diffs', 'digital', 'dimensions', 'dinner',
                     'directions', 'director', 'directors', 'directory', 'disabled',
                     'discovery', 'disease', 'disney', 'distrct', 'distribution',
                     'ditto', 'diverse', 'dna', 'doc', 'doctor', 'doctors', 'doctrine',
                     'documentary', 'documentation', 'documented', 'documents',
                     'dodo', 'dogs', 'doin', 'dollar', 'dollars', 'donald', 'doo',
                     'door', 'dot', 'doug', 'download', 'downtown', 'dozen', 'dozens',
                     'dq', 'draft', 'drafting', 'drag', 'dragon', 'dragons', 'drawing',
                     'drawn', 'dreamguy', 'dress', 'drew', 'drink', 'drinking', 'drivel',
                     'duke', 'dungeons', 'duplicate', 'dust', 'dutch', 'dvd', 'dyk',
                     'dyke', 'dynasty', 'eagle', 'ear', 'eastern', 'eating', 'ebay',
                     'ec', 'economic', 'economy', 'ed','edge', 'edition', 'editorial',
                     'edu', 'edward', 'egg', 'egypt', 'egyptian', 'einstein', 'el',
                     'elected', 'elections', 'electric', 'electronic', 'elephant',
                     'elite', 'elitist', 'elonka', 'elses', 'em', 'emailed', 'emails',
                     'employee', 'employees', 'employer', 'encyclopedia', 'encyclopaedic',
                     'encyclopedias', 'endorse', 'ends', 'engine', 'engineer',
                     'engineering', 'england', 'episodes', 'equation', 'equations',
                     'equipment', 'er', 'era', 'eric', 'esp', 'espouse', 'essay',
                     'est', 'et', 'eu', 'eugenics', 'europeans', 'evening', 'everyday',
                     'everytime', 'ex', 'exams', 'eyed', 'f5fffa', 'fa', 'fac',
                     'facebook', 'facility', 'facist', 'fair_use', 'fascists',
                     'fathers', 'fbi', 'fc', 'feb', 'feet', 'feminist', 'feminists',
                     'ffffff', 'fidel', 'fifth', 'filmography', 'films', 'financial',
                     'finland', 'firefox', 'firstly', 'fish', 'fisherqueen', 'font',
                     'food', 'foot', 'footnote', 'footnotes', 'forbes', 'foreign',
                     'foreigners', 'fork', 'fort', 'forth', 'flickr', 'floor',
                     'florida', 'floyd', 'fo', 'forums', 'fox', 'fr', 'france',
                     'francis', 'francisco', 'frank', 'franklin', 'fred', 'freud',
                     'friday', 'frozen', 'fruit', 'ft', 'fundamentalist', 'funeral',
                     'furry','fwiw', 'fyi', 'g1', 'g11', 'galaxy', 'galleries',
                     'gallery', 'gamaliel', 'gamegate', 'gan', 'garage', 'gary', 'gay',
                     'gays', 'gb', 'gd', 'gender', 'gene', 'genesis', 'genetics',
                     'geographical', 'geography', 'geographically', 'geology',
                     'georgia', 'geographic', 'gerard', 'germanic', 'germans',
                     'gfdl', 'gfdl', 'gibson', 'gif', 'gimme', 'girlfriend',
                     'glass', 'gm', 'gmail', 'gnu', 'golf', 'gong', 'goodbye',
                     'gordon', 'gospel', 'gothic', 'greece', 'greeks', 'greg',
                     'gw', 'gwen', "hadn", 'hahaha','hahahaha', 'halfway',
                     'hall', 'harry', "harvard", 'hat', 'hav', 'heap',
                     'highway', 'highways', 'hill', 'hindi', 'hinduism', 'hindus',
                     'hmm', 'hmmm', 'hmmmmm', 'hobby', 'hockey', 'holiday', 'holier',
                     'holla', 'hollywood', 'holocaust', 'holy', 'homeland', 'homepage',
                     'homes', 'homework', 'hominem', 'homo', 'homophobic', 'homosexuality',
                     'homosexuals', 'heterosexual', 'heaven', 'hebrew', 'heh',
                     'height', 'henry', 'herbert', 'heritage', 'hes', 'hesperian',
                     'horizontal', 'horses', 'hospital', 'host', 'hosted', 'hosting',
                     'hosts', 'hotel', 'hotmail', 'houses', 'housing', 'houston',
                     'howard', 'htm', 'https', 'hugo', 'hungarian', 'idle', 'iii',
                     'ilk', 'imdb', 'imho', 'immigrant', 'immigrants', 'imo', 'imperialism',
                     'incorporate', 'incorporated', 'indians', 'indonesia', 'indonesian',
                     'industry', 'industrial', 'infoboxes', 'ing', 'int','internationally',
                     'intro', 'introduce', 'interviewed', 'interviews', 'introducing',
                     'ips', 'iq', 'ir', 'ira', 'iranian', 'iranians', 'iranica',
                     'iraq', 'irc', 'ireland', 'iron', 'irs', 'isbn', 'islamic',
                     'islamist', 'island', 'islands', 'isp', 'israeli', 'israelis',
                     'italian', 'italy', 'item', 'items', 'iv', 'ive', 'ja', 'jackson',
                     'jam', 'jamesbwatson', 'jamie', 'jan', 'jane', 'janet', 'japan',
                     'jason', 'jay', 'jayjg', 'jeez', 'jeff', 'jehovah', 'jenkins',
                     'jeppiz', 'jeremy', 'jerry', 'jersey', 'jet', 'jim', 'jimmy',
                     'jim', 'joan', 'jobs', 'joe', 'johnny', 'johnson', 'jon',
                     'jonathan', 'jones', 'jordan', 'jose', 'joseph', 'jossi',
                     'jpgordon', 'jpgthank', 'jr', 'judaism', 'jus', 'justin',
                     'kansas', 'karl', 'keith', 'kelly', 'ken', 'kennedy', 'kentucky',
                     'kevin', 'keyboard', 'khan', 'kim', 'kinda', 'kingdom', 'kings',
                     'km','knees', 'knight', 'knox', 'kong', 'korea', 'korean',
                     'kosovo', 'ku', 'kurdish', 'kurt', 'lab', 'labor', 'labour',
                     'laden', 'lake', 'lands', 'languages', 'lanka', 'larry', 'latin',
                     'latino', 'laurent', 'lawrence', 'lbs', 'le', 'lecture', 'lede',
                     'lee', 'leftist', 'leg', 'legs', 'length', 'lengths', 'lesbian',
                     'letters', 'lewis', 'lgbt', 'lightning', 'lil', 'lincoln',
                     'linda', 'linguistic', 'linguistics', 'linguists', 'lips',
                     'lisa', 'literature', 'liz', 'lmao', 'locate', 'located',
                     'locations', 'logo', 'logos', 'logs', 'lolz', 'london', 'longtime',
                     'lopez', 'los', 'louis', 'luke', 'lulz', 'lunch', 'lynch',
                     'lynn', 'lyrics', 'ma', 'macedonia', 'macedonian', 'machine',
                     'machines', 'magazines', 'magic', 'mah', 'mailing', 'mainpagebg',
                     'mainspace', 'malaysia', 'males', 'mankind', 'maps', 'marc',
                     'marcus', 'mardyks', 'margaret', 'margin', 'maria', 'marie',
                     'mario', 'market', 'market', 'marketing', 'markets', 'marking',
                     'marks', 'marriage', 'married', 'marry', 'mars', 'martial',
                     'martin', 'marvel', 'marx', 'marxist', 'mary', 'massachusetts',
                     'masters', 'masses', 'materials','math', 'mathematical',
                     'mathematics', 'matt', 'max', 'measure', 'measured', 'measurement',
                     'measures', 'measuring', 'meat', 'meatpuppet', 'mechanics',
                     'mechanism', 'mediawiki', 'mediterranean', 'medium', 'meetup',
                     'mel', 'messenger', 'meta', 'metalcore', 'meters', 'metro',
                     'mexican', 'mexicans', 'mexico', 'mfd', 'micro', 'microsoft',
                     'mid', 'midnight', 'mike', 'miles', 'mini', 'minister', 'ministry',
                     'minorities', 'minute', 'mississippi', 'mister', 'mitchell',
                     'mix', 'mixed', 'mixture', 'mo', 'model', 'models', 'mommy',
                     'monday', 'moon', 'moore', 'mop','morgan', 'mormon', 'morning',
                     'morris', 'mos', 'moscow', 'mothers', 'mountain', 'movies',
                     'mp', 'mrs', 'ms', 'msg', 'mtv', 'mud', 'muhammad', 'multi',
                     'myspace', 'na', 'nancy', 'nasa', 'nationalism', 'nationalistic',
                     'nationalists', 'nationality', 'nato', 'naval', 'nazism', 'negro',
                     'nbc', 'naqlinwiki', 'nd', 'neck', 'neighbor', 'netherlands',
                     'network', 'networking', 'networks', 'newsletter', 'newspapers',
                     'nicholas', 'nick', 'nickname', 'nigga', 'nigger', 'niggers',
                     'nintendo', 'norman', 'northern', 'notification', 'noun',
                     'nov', 'novel', 'novels', 'npa', 'nsw', 'nt', 'ny', 'nz',
                     'ocean', 'oct', 'officer', 'officers', 'offline', 'officials',
                     'ohio', 'oi', 'oil', 'oklahoma', 'ol', 'older', 'oldid', 'olds',
                     'olympics', 'op', 'oprah', 'oral', 'orange', 'oregon', 'orleans',
                     'orthodox', 'osama', 'oscar', 'ot', 'otters', 'ottoman', 'oxford',
                     'pa', 'pack', 'package', 'padding', 'pagan', 'pagehelp', 'pagehow',
                     'pagestutorialhow', 'paint', 'painting', 'pakistan', 'pakistani',
                     'palace', 'palestine', 'palestinian', 'palestianians', 'pan',
                     'panama', 'pants', 'papers', 'para', 'parent', 'paris', 'park',
                     'parker', 'parliament', 'parrot', 'partisan', 'passage', 'password',
                     'paste', 'pasted', 'pasting', 'pat', 'patrick', 'patriotic',
                     'pbs', 'pc', 'pd', 'pdf', 'peacock','pedia', 'pen', 'pending',
                     'penny', 'people', 'peopel', 'percent', 'percentage', 'perry',
                     'persian', 'persians', 'peruvian', 'pete', 'peter', 'pg', 'ph',
                     'phelps', 'phil', 'philip', 'phone', 'photograph', 'photographer',
                     'photographs', 'photography', 'pic', 'pics', 'pierre', 'pig', 'pigs',
                     'plane', 'planets', 'plant', 'plants', 'platic', 'plate', 'pls',
                     'plural', 'plz', 'pm', 'png', 'poland', 'polish', 'politically',
                     'politician', 'politicians', 'pollution', 'pool', 'pope', 'poppers',
                     'pops', 'populations', 'pork', 'port', 'portal', 'portrait',
                     'portugal', 'portuguese', 'poster', 'posters', 'postings',
                     'potter', 'ppl', 'pr', 'presidency', 'presidential', 'prev',
                     'priest', 'priests','prince', 'princess', 'print', 'printed',
                     'printing', 'prod', 'prof', 'profile', 'profiles', 'professor',
                     'professionals', 'programming', 'programs', 'prophet', 'protestants',
                     'pseudoscience', 'psychiatric', 'psychiatry', 'psychological',
                     'psychology', 'purple', 'quantum', 'queen', 'queer', 'quest',
                     'races', 'rachel', 'rail', 'railway', 'rain', 'raja', 'rajputs',
                     'rap', 'rawat', 'ray', 'redlinks', 'ref', 'refugees', 'regime',
                     'register', 'registered', 'registering', 'reign', 'religions',
                     'republican', 'republicans', 'resident', 'residential', 'residents',
                     'retire', 'retired', 'retirement', 'reuters', 'reviewer', 'reviewing',
                     'rfa', 'rex',  'rice', 'rich', 'rick', 'ring', 'ritual', 'river',
                     'rm', 'ro', 'roads', 'rob', 'robert', 'robot', 'rocket', 'rocks',
                     'rofl', 'roger', 'roll', 'rollback', 'roman', 'romania', 'romanian',
                     'rome', 'ron', 'root', 'roots', 'ross', 'roy', 'rss', 'russians',
                     'ruth', 'rv', 'ryan', 'ryulong', 'sa', 'saddam', 'saga', 'sahara',
                     'saint', 'sales', 'salt', 'sam', 'scan', 'scenario', 'scene',
                     'scenes', 'schools', 'sciences', 'scientifically', 'physics',
                     'scientist', 'scientists', 'scientology', 'scjessy', 'scotland',
                     'scott', 'scottish', 'screen', 'screenshots', 'script', 'script',
                     'scroll', 'se', 'sea', 'sean', 'sell', 'selling', 'senate',
                     'senior', 'sep', 'sept', 'serb', 'serbia', 'serbian', 'serbs',
                     'servant', 'server', 'servers', 'seb', 'serves', 'serve', 'services',
                     'serving', 'seven', 'seventh', 'sh', 'sheep', 'sheesh', 'shell',
                     'shop', 'shoulder', 'shower', 'si', 'sic', 'sig', 'sigh',
                     'sikh', 'singapore', 'singer', 'sing', 'singh', 'singles', 'singular',
                     'skirt', 'skull', 'sky', 'slave', 'slaves', 'slavic', 'slavs',
                     'sleep', 'slim', 'smaller', 'snow', 'snowman', 'soap', 'soapbox',
                     'soccer', 'socialism', 'socialist', 'sockpuppetry', 'sockpuppets',
                     'soil', 'solar', 'sold', 'someday', 'someones', 'somethin', 'songs',
                     'sons', 'simon', 'ship', 'sheriff', 'shi', 'ships', 'shirts',
                     'smith', 'sony', 'sooo', 'soul', 'southern', 'soviet', 'spain',
                     'species', 'spell', 'spi', 'spin', 'spirit', 'spiritual', 'sport',
                     'spring', 'spy', 'squad', 'sports', 'square', 'sr', 'sri', 'ss',
                     'st', 'stadium', 'staff', 'stage', 'stalin', 'stamp', 'stanford',
                     'stars', 'station', 'stations', 'statistical', 'statistics', 'stats',
                     'stephen', 'steve', 'steven', 'stone', 'straw', 'stream', 'streets',
                     'studied', 'studio', 'studying', 'sub', 'subpage', 'subsection',
                     'subsequent', 'subsequently', 'summer', 'sun', 'sunday', 'sup',
                     'supremacist', 'supreme', 'susan', 'svg', 'swastika', 'swedish',
                     'symbol', 'symbols', 'syntax', 'synthesis', 'syria', 'syrian',
                     'sysop', 'ta', 'tab', 'tables', 'tabloid', 'tabtab', 'tail',
                     'tales', 'tall', 'tape', 'tech', 'technology', 'ted', 'taylor',
                     'tax', 'taxes', 'tc', 'tea', 'tech', 'ted', 'teen', 'teenager',
                     'teenagers', 'teens', 'television', 'temp', 'temple', 'territories',
                     'territory', 'terry', 'testament', 'texas', 'textbook', 'textbooks',
                     'texts', 'tfd', 'th', 'theater', 'theirs', 'theme', 'themes',
                     'theorist', 'therapy', 'thereof', 'theres', 'thesis', 'thier',
                     'theirs', 'theme', 'themes', 'thirdly', 'thirty', 'tho', 'thomas',
                     'thompson', 'thou', 'thousand', 'threads', 'throat', 'thug', 'thugs',
                     'thumb', 'thursday', 'tickets', 'tiger', 'til', 'tim', 'tiny',
                     'tip', 'tips', 'titled', 'tk', 'tlk', 'tom', 'tomorrow', 'ton',
                     'tonight', 'tons', 'tony', 'tops', 'toronto', 'totalitarian',
                     'trademark', 'tradition', 'traditional', 'traditionally',
                     'traditions', 'traffic', 'treatment', 'trek', 'tribes', 'triple',
                     'trivia', 'trivial', 'troops', 'truck', 'trout', 'ts', 'turk',
                     'turkey', 'turkic', 'turks', 'tw', 'twist', 'twisted', 'twitter',
                     'tyler', 'typed', 'typing', 'typo', 'typos', 'tyranny', 'tyrant',
                     'tyrants', 'ufc', 'ukraine', 'ukranian', 'um', 'umm', 'uncited',
                     'uncle', 'uneducated', 'unemployed', 'unencyclopedic', 'unilateral',
                     'unilaterally', 'unit', 'units', 'unknown','unregistered', 'untagged',
                     'untill', 'updates', 'updating', 'upgrade', 'uploads', 'urban',
                     'url', 'usenet', 'user_talk', 'userbox', 'userboxes', 'usernames',
                     'userspace', 'ussr', 'uw', 'valley', 'van', 'vehicle', 'verb',
                     'verbal',  'verbatim', 'verizon', 'versa', 'versions', 'vertical',
                     'veteran', 'vfd', 'vice', 'victor', 'videos', 'villages', 'vincent',
                     'vinci', 'virginia', 'virditas', 'virus', 'viruses', 'von', 'voted',
                     'voters', 'votes', 'voting', 'wake', 'walk', 'walker', 'walking',
                     'walks', 'wall', 'walls', 'walt', 'walter', 'warfare', 'wash',
                     'washington', 'wet', 'whilst', 'whites', 'wholesale', 'wholly',
                     'whos', 'wider', 'widespread', 'width', 'wik', 'wikia', 'wikibreak',
                     'wikilove', 'wikinazi', 'wikinazis', 'wikipediahow', 'wikipediatutorialhow',
                     'wikipedia', 'wikipeida', 'wikiprojects', 'wikiquette', 'wikis',
                     'wikitionary', 'wikzilla', 'wil', 'william', 'williams', 'willy',
                     'wilson', 'wind', 'windows', 'winter', 'witch', 'wizard', 'wolf',
                     'wolfowitz', 'worded', 'workers', 'worlds', 'worldwide', 'worm',
                     'worship', 'wrestling', 'wright', 'wrist', 'writer', 'writers',
                     'writes', 'writings', 'wt', 'wwe', 'xbox', 'xd', 'xenophobic',
                     'xxx', 'yahoo', 'yale', 'yall', 'yamla', 'yanks', 'yank', 'yard',
                     'ye', 'yellow', 'yer', 'yesterday', 'younger', 'youth', 'yr', 'yu',
                     'yugoslav', 'yugoslavia', 'zionist', 'zoe', 'zombies', '10th',
                     '126', '192', '1958', '1960', '1965', '1969', '1970', '1974',
                     '1976', '1977', '1983', '1987', '1988', '2003', '2602', '3000',
                     '303', '350', '360', 'alice', 'biology', 'yep', 'yea' ,'artist',
                     'citizen', 'billion', 'civilian', 'founder', '51', '59', '6000',
                     '62', '777', '78', '9th', 'a1', 'aa', 'aaron', 'invent', 'jurisdiction',
                     'sarah', 'device', 'aspects', 'homeless', 'presentation', 'khoi']

print(len(unhelpfulFeatures_updated))

#Remove features in the list created previously
for word in unhelpfulFeatures_updated:
    if word in svmToxicdf.columns:
        svmToxicdf.drop(columns = word, inplace = True, axis = 1)

print(svmToxicdf.shape)


##############################################################
##
##    Create visualizations - Word Cloud
##
##############################################################



###Create a word cloud to review the most frequently appearing words
#Join all tweets into one text document
commentstext = " ".join(svmToxicdf.columns)

#Create and generate a word cloud image
wordcloud = WordCloud(max_words=200, background_color="White",max_font_size = 40,
                      collocations=False).generate(commentstext)

# Display the generated image
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()





######################################################################
#
# Creating a train and test set for SVM Models
#
######################################################################

#creating a training and test set
train, test = train_test_split(svmToxicdf, test_size=0.3, stratify=svmToxicdf['Label'],
                               random_state=254)


#checking to ensure trainig and testing sets are balanced
print(train['Label'].value_counts())
print(test['Label'].value_counts())


#copying the training label
trainingLabel = train["Label"]

#dropping labels from training DF
train = train.drop(['Label'], axis = 1)

#copying the testing label
testingLabel = test['Label']

#dropping the testing label
test =test.drop(['Label'],axis = 1)


######################################################################
#
# SVM - Linear Kernel
#
######################################################################

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score

#building the linear model - tried C = 1,5,50,100 - accuracies did not change
svmLinear = LinearSVC(C = 0.50, max_iter= 1000)

svmLinearPredictions = svmLinear.fit(train, trainingLabel).predict(test)
linearConf = confusion_matrix(testingLabel, svmLinearPredictions)

print('SVM Linear Confusion Matrix: \n', linearConf)
print('Precision Scores for SVM Linear Toxic and Not Toxic: ',
      precision_score(testingLabel.values, svmLinearPredictions, labels =[1, 0],
                      average=None))

print('Recall Scores for SVM Linear Toxic and Not Toxic: ',
      recall_score(testingLabel.values, svmLinearPredictions, labels =[1, 0],
                      average=None))


#20 most indicative features

svmLinIndicative = sorted(zip(svmLinear.coef_[0], svmLinear.feature_names_in_))
print('Most Indicative non-Toxic Words: \n',svmLinIndicative[0:20])
print('Most Indicative Toxic Words: \n',svmLinIndicative[-20:])

notToxicWords = []
notToxicCoef = []

for item in svmLinIndicative[0:20]:
    notToxicCoef.append(item[0])
    notToxicWords.append(item[1])
notToxicLinSvm = pd.DataFrame(notToxicWords, columns = ['Not Toxic Features'])
notToxicLinSvm.insert(loc=1, column = 'Coef', value = notToxicCoef)


toxicWords = []
toxicCoef = []
for item in svmLinIndicative[-20:]:
    toxicCoef.append(item[0])
    toxicWords.append(item[1])

toxicLinSvm = pd.DataFrame(toxicWords, columns = ['Toxic Features'])
toxicLinSvm.insert(loc=1, column = 'Coef', value = toxicCoef)


#barplot of most indicative features
figure3 = plt.figure()
ax3 = figure3.add_axes([0,0,1,1])
ax3.bar(notToxicLinSvm['Not Toxic Features'].values, notToxicLinSvm['Coef'].values)
plt.xticks(notToxicLinSvm['Not Toxic Features'], rotation = 'vertical')
plt.title('Most Indicative Words for non-Toxic Comments')
plt.show()

figure4 = plt.figure()
ax4 = figure4.add_axes([0,0,1,1])
ax4.bar(toxicLinSvm['Toxic Features'].values, toxicLinSvm['Coef'].values)
plt.xticks(toxicLinSvm['Toxic Features'], rotation = 'vertical')
plt.title('Most Indicative Words for Toxic Comments')
plt.show()


######################################################################
#
# SVM - Rbf Kernel
#
######################################################################

#building the rbf model - tried several values C=1, 5, 25,50, 100- C=25
svmRBFModel = SVC(C= 500, kernel = 'rbf', gamma='auto', verbose=True)
predictionsRBF = svmRBFModel.fit(train, trainingLabel).predict(test)
rbfConf = confusion_matrix(testingLabel, predictionsRBF)
print('SVM RBF Confusion Matrix: \n', rbfConf)

print('Precision Scores for SVM RBF Toxic and Not Toxic: ',
      precision_score(testingLabel.values, predictionsRBF, labels =[1, 0],
                      average=None))

print('Recall Scores for SVM RBF Toxic and Not Toxic: ',
      recall_score(testingLabel.values, predictionsRBF, labels =[1, 0],
                      average=None))


#20 most indicative features

svmRBFIndicative = sorted(zip(svmRBFModel.dual_coef_[0], svmRBFModel.feature_names_in_))
print('Most Indicative Non-Toxic Words: \n',svmRBFIndicative [0:20])
print('Most Indicative Toxic Words: \n',svmRBFIndicative [-20:])



notToxicWords = []
notToxicCoef = []

for item in svmRBFIndicative[0:20]:
    notToxicCoef.append(item[0])
    notToxicWords.append(item[1])
notToxicRBFSvm = pd.DataFrame(notToxicWords, columns = ['Not Toxic Features'])
notToxicRBFSvm.insert(loc=1, column = 'Coef', value = notToxicCoef)


toxicWords = []
toxicCoef = []
for item in svmRBFIndicative[-20:]:
    toxicCoef.append(item[0])
    toxicWords.append(item[1])

toxicRBFSvm = pd.DataFrame(toxicWords, columns = ['Toxic Features'])
toxicRBFSvm.insert(loc=1, column = 'Coef', value = toxicCoef)


#barplot of most indicative features
figure5 = plt.figure()
ax5 = figure5.add_axes([0,0,1,1])
ax5.bar(notToxicRBFSvm['Not Toxic Features'].values, notToxicRBFSvm['Coef'].values)
plt.xticks(notToxicRBFSvm['Not Toxic Features'], rotation = 'vertical')
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure6 = plt.figure()
ax6 = figure6.add_axes([0,0,1,1])
ax6.bar(toxicRBFSvm['Toxic Features'].values, toxicRBFSvm['Coef'].values)
plt.xticks(toxicRBFSvm['Toxic Features'], rotation = 'vertical')
plt.title('Most Indicative Words for Toxic Comments')
plt.show()


######################################################################
#
# SVM - sigmoid Kernel
#
######################################################################



#building the sigmoid model - tried several values C=1,10,25,50,75, 100- C=50
svmSigModel = SVC(C= 100, kernel = 'sigmoid', gamma='auto', verbose=True)
predictionsSig = svmSigModel.fit(train, trainingLabel).predict(test)
sigConf = confusion_matrix(testingLabel, predictionsSig)
print('SVM Sigmoid Confusion Matrix: \n', sigConf)

print('Precision Scores for SVM Sigmoid Toxic and Not Toxic: ',
      precision_score(testingLabel.values, predictionsSig, labels =[1, 0],
                      average=None))

print('Recall Scores for SVM Sigmoid Toxic and Not Toxic: ',
      recall_score(testingLabel.values, predictionsSig, labels =[1, 0],
                      average=None))


#20 most indicative features

svmSigIndicative = sorted(zip(svmSigModel.dual_coef_[0], svmSigModel.feature_names_in_))
print('Most Indicative Non-Toxic Words: \n',svmSigIndicative [0:20])
print('Most Indicative Toxic Words: \n',svmSigIndicative [-20:])


notToxicWords = []
notToxicCoef = []

for item in svmSigIndicative[0:20]:
    notToxicCoef.append(item[0])
    notToxicWords.append(item[1])
notToxicSigSvm = pd.DataFrame(notToxicWords, columns = ['Not Toxic Features'])
notToxicSigSvm.insert(loc=1, column = 'Coef', value = notToxicCoef)


toxicWords = []
toxicCoef = []
for item in svmSigIndicative[-20:]:
    toxicCoef.append(item[0])
    toxicWords.append(item[1])

toxicSigSvm = pd.DataFrame(toxicWords, columns = ['Toxic Features'])
toxicSigSvm.insert(loc=1, column = 'Coef', value = toxicCoef)


#barplot of most indicative features
figure6 = plt.figure()
ax6 = figure6.add_axes([0,0,1,1])
ax6.bar(notToxicSigSvm['Not Toxic Features'].values, notToxicSigSvm['Coef'].values)
plt.xticks(notToxicSigSvm['Not Toxic Features'], rotation = 'vertical')
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure7 = plt.figure()
ax7 = figure7.add_axes([0,0,1,1])
ax7.bar(toxicSigSvm['Toxic Features'].values, toxicSigSvm['Coef'].values)
plt.xticks(toxicSigSvm['Toxic Features'], rotation = 'vertical')
plt.title('Most Indicative Words for Toxic Comments')
plt.show()



######################################################################
#
# SVM - Poly Kernel
#
######################################################################


#building the rbf model - tried several values C=1, 5, 25,50, 100- C=25
svmPolyModel = SVC(C= 2500, kernel = 'poly', gamma='auto', verbose=True)
predictionsPoly = svmPolyModel.fit(train, trainingLabel).predict(test)
polyConf = confusion_matrix(testingLabel, predictionsPoly)
print('SVM Poly Confusion Matrix: \n', polyConf)

print('Precision Scores for SVM Poly Toxic and Not Toxic: ',
      precision_score(testingLabel.values, predictionsPoly, labels =[1, 0],
                      average=None))

print('Recall Scores for SVM Poly Toxic and Not Toxic: ',
      recall_score(testingLabel.values, predictionsPoly, labels =[1, 0],
                      average=None))


#20 most indicative features

svmPolyIndicative = sorted(zip(svmPolyModel.dual_coef_[0], svmPolyModel.feature_names_in_))
print('Most Indicative Non-Toxic Words: \n',svmPolyIndicative [0:20])
print('Most Indicative Toxic Words: \n',svmPolyIndicative [-20:])



notToxicWords = []
notToxicCoef = []

for item in svmPolyIndicative[0:20]:
    notToxicCoef.append(item[0])
    notToxicWords.append(item[1])
notToxicPolySvm = pd.DataFrame(notToxicWords, columns = ['Not Toxic Features'])
notToxicPolySvm.insert(loc=1, column = 'Coef', value = notToxicCoef)


toxicWords = []
toxicCoef = []
for item in svmPolyIndicative[-20:]:
    toxicCoef.append(item[0])
    toxicWords.append(item[1])

toxicPolySvm = pd.DataFrame(toxicWords, columns = ['Toxic Features'])
toxicPolySvm.insert(loc=1, column = 'Coef', value = toxicCoef)


#barplot of most indicative features
figure5 = plt.figure()
ax5 = figure5.add_axes([0,0,1,1])
ax5.bar(notToxicPolySvm['Not Toxic Features'].values, notToxicPolySvm['Coef'].values)
plt.xticks(notToxicPolySvm['Not Toxic Features'], rotation = 'vertical')
plt.title('Most Indicative Words for Non-Toxic Comments')
plt.show()

figure6 = plt.figure()
ax6 = figure6.add_axes([0,0,1,1])
ax6.bar(toxicPolySvm['Toxic Features'].values, toxicPolySvm['Coef'].values)
plt.xticks(toxicPolySvm['Toxic Features'], rotation = 'vertical')
plt.title('Most Indicative Words for Toxic Comments')
plt.show()


######################################################################
#
# SVM -Linear - 5 -fold cross validation
#
######################################################################
from sklearn.model_selection import cross_val_score
svmToxic = svmToxicdf.copy(deep = True)
toxicLabel = svmToxic['Label'].values
svmToxic = svmToxic.drop(['Label'], axis = 1)

svmLinModel2 = LinearSVC(C = 0.50, max_iter= 1000)
svmLinModel2.fit(svmToxic, toxicLabel)
svmTuningScore = cross_val_score(svmLinModel2, svmToxic, toxicLabel, cv = 5)
print(svmTuningScore.mean())
print(svmTuningScore)




##############################################################
##
##    Create visualizations - SVM
##
##############################################################

X = train[['dig', 'boring']]
yVals = trainingLabel.values
y = []
for value in yVals:
    if value == 1:
        y.append(1)
    elif value == 0:
        y.append(0)
#print(y)


h = 0.02

svmLinear = LinearSVC(C = 1.0, max_iter= 1000)
svmLinear.fit(X,y)
# create a mesh to plot in
x_min, x_max = X['dig'].min() - 0.1, X['dig'].max() + 0.1
y_min, y_max = X['boring'].min() - 0.1, X['boring'].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

title = 'SVC with Linear Kernel'
figure3 = plt.figure()
figure3.subplots_adjust(wspace=0.4, hspace=0.4)

Z = svmLinear.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(train['dig'], train['boring'], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Dig')
plt.ylabel('Boring')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title(title)

plt.show()



##############################################################
##
##    Create visualizations - SVM
##
##############################################################



X = train[['bond', 'painful']]
yVals = trainingLabel.values
y = []
for value in yVals:
    if value == 1:
        y.append(1)
    elif value == 0:
        y.append(0)
#print(y)


h = 0.02

svmLinear = LinearSVC(C = 1.0, max_iter= 1000)
svmLinear.fit(X,y)
# create a mesh to plot in
x_min, x_max = X['bond'].min() - 0.1, X['bond'].max() + 0.1
y_min, y_max = X['painful'].min() - 0.1, X['painful'].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

title = 'SVC with Linear Kernel'
figure3 = plt.figure()
figure3.subplots_adjust(wspace=0.4, hspace=0.4)

Z = svmLinear.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(train['bond'], train['painful'], c=y, cmap=plt.cm.coolwarm)
plt.xlabel('Bond')
plt.ylabel('Painful')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title(title)

plt.show()
