# -*- coding: utf-8 -*-
"""

@author: Hogan
"""


import pandas as pd
import jieba
import jieba.analyse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


def readConfig():
    df_news = pd.read_table('./data/data.txt',names=['category','theme','URL','content'],encoding='utf-8')
    df_news = df_news.dropna()
    print(df_news.tail())
    
    print(df_news.shape)
    print(type(df_news.content))
    print(df_news.content)
    content = df_news.content.values.tolist() #将每一篇文章转换成一个list
    print (content[1000]) #随便选择其中一个看看
    return df_news, content


def cut_word(content):
    content_S = []
    for line in content:
        current_segment = jieba.lcut(line) #对每一篇文章进行分词
        if len(current_segment) > 1 and current_segment != '\r\n': #换行符
            content_S.append(current_segment) #保存分词的结果
    print(content_S[1000])
    return content_S
    

def get_stop_word():
    df_content=pd.DataFrame({'content_S':content_S}) #专门展示分词后的结果
    print(df_content.head())

    stopwords=pd.read_csv("stopwords.txt",index_col=False,sep="\t",quoting=3,names=['stopword'], encoding='utf-8')
    print(stopwords.head(20))
    return stopwords


def drop_stopwords(content_S,stopwords):
    df_content=pd.DataFrame({'content_S':content_S}) #专门展示分词后的结果
    print(df_content.head())
    
    contents = df_content.content_S.values.tolist()
    stopwords = stopwords.stopword.values.tolist()

    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean, all_words


def jieba_process(content_S, index = 2400):
    content_S_str = "".join(content_S[index]) #把分词的结果组合在一起，形成一个句子
    print (content_S_str) #打印这个句子
    print ("  ".join(jieba.analyse.extract_tags(content_S_str, topK=5, withWeight=False)))#选出来5个核心词


def get_train(contents_clean, df_news):
    df_train=pd.DataFrame({'contents_clean':contents_clean,'label':df_news['category']})
    print(df_train.tail())
    print(df_train.label.unique())
    label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育":5, "教育": 6,"文化": 7,"军事": 8,"娱乐": 9,"时尚": 0}
    df_train['label'] = df_train['label'].map(label_mapping) #构建一个映射方法
    df_train.head()
    return df_train


def split(dt_train):
    x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values, random_state=1)
    return x_train, x_test, y_train, y_test

def split2(dt_train):
    x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values, random_state=1)
   
    train_words = []
    for line_index in range(len(x_train)):
        try:
            train_words.append(' '.join(x_train[line_index]))
        except:
            print (line_index)
            
    test_words = []
    for line_index in range(len(x_test)):
        try:
            test_words.append(' '.join(x_test[line_index]))
        except:
            print (line_index)
    return train_words, y_train, test_words, y_test

def Count_fitting(x_train, x_test, y_train, y_test):
    train_words = []
    classifier = MultinomialNB() 
    for line_index in range(len(x_train)):
        try:
            train_words.append(' '.join(x_train[line_index]))
        except:
            print (line_index)
    print('train_words[0] : ', train_words[0])
    print(len(train_words)) 
    
    vec = CountVectorizer(analyzer='word', max_features=4000, lowercase = False)# 最大特征程度
    feature = vec.fit_transform(train_words)
    print('new feature', feature)
    print(feature.shape)
    
    classifier.fit(feature, y_train)
    
    test_words = []
    for line_index in range(len(x_test)):
        try:
            test_words.append(' '.join(x_test[line_index]))
        except:
            print (line_index)
    print(test_words[0])

    print(classifier.score(vec.transform(test_words), y_test))


def Tf_fitting(x_train, x_test, y_train, y_test):
    
    X_test = ['卡尔 敌法师 蓝胖子 小小','卡尔 敌法师 蓝胖子 痛苦女王']

    tfidf = TfidfVectorizer()
    weight = tfidf.fit_transform(X_test).toarray()
    word = tfidf.get_feature_names()
    classifier = MultinomialNB()
    print (weight)
    for i in range(len(weight)):  
        print(u"第", i, u"篇文章的tf-idf权重特征")
        for j in range(len(word)):
            print(word[j], weight[i][j])
            
    train_words = []
    for line_index in range(len(x_train)):
        try:
            train_words.append(' '.join(x_train[line_index]))
        except:
            print (line_index)
            
    vectorizer = TfidfVectorizer(analyzer='word', max_features=4000,  lowercase = False)
    vectorizer.fit(train_words)
    
    test_words = []
    for line_index in range(len(x_test)):
        try:
            test_words.append(' '.join(x_test[line_index]))
        except:
            print (line_index)
    print(test_words[0])
    
    classifier.fit(vectorizer.transform(train_words), y_train)
    print(classifier.score(vectorizer.transform(test_words), y_test))
    
def Count_fitting2(train_words, y_train, test_words, y_test):
    classifier = MultinomialNB() 
    
    vec = CountVectorizer(analyzer='word', max_features=4000, lowercase = False)# 最大特征程度
    feature = vec.fit_transform(train_words)
    print('new feature', feature)
    print(feature.shape)
    
    classifier.fit(feature, y_train)
    
    print(classifier.score(vec.transform(test_words), y_test))




def Tf_fitting2(train_words, y_train, test_words, y_test):
    
    X_test = ['卡尔 敌法师 蓝胖子 小小','卡尔 敌法师 蓝胖子 痛苦女王']

    tfidf = TfidfVectorizer()
    weight = tfidf.fit_transform(X_test).toarray()
    word = tfidf.get_feature_names()
    classifier = MultinomialNB()
    print (weight)
    for i in range(len(weight)):  
        print(u"第", i, u"篇文章的tf-idf权重特征")
        for j in range(len(word)):
            print(word[j], weight[i][j])
            
    vectorizer = TfidfVectorizer(analyzer='word', max_features=4000,  lowercase = False)
    vectorizer.fit(train_words)
    
    classifier.fit(vectorizer.transform(train_words), y_train)
    print(classifier.score(vectorizer.transform(test_words), y_test))




df_news, content = readConfig()
content_S = cut_word(content)
stopwords = get_stop_word()

contents_clean, all_words = drop_stopwords(content_S, stopwords)

jieba_process(content_S, 2400)

df_train = get_train(contents_clean, df_news)

#x_train, x_test, y_train, y_test = split(df_train)
#
#Count_fitting(x_train, x_test, y_train, y_test)
#
#Tf_fitting(x_train, x_test, y_train, y_test)



train_words, y_train, test_words, y_test = split2(df_train)
Count_fitting2(train_words, y_train, test_words, y_test)
Tf_fitting2(train_words, y_train, test_words, y_test)

