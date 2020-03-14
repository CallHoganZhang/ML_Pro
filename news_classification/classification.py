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

df_news = pd.read_table('./data/data.txt',names=['category','theme','URL','content'],encoding='utf-8')
df_news = df_news.dropna()
print(df_news.tail())

print(df_news.shape)
print(type(df_news.content))
print(df_news.content)
content = df_news.content.values.tolist()
print (content[1000]) #任意一个数字即可，代表某一篇文章

content_S = []
for line in content:
    current_segment = jieba.lcut(line) #对每一篇文章进行分词
    if len(current_segment) > 1 and current_segment != '\r\n': #去除换行符
        content_S.append(current_segment) #保存分词的结果
print(content_S[1000])

df_content=pd.DataFrame({'content_S':content_S}) #专门展示分词后的结果
print(df_content.head())

stopwords=pd.read_csv("stopwords.txt",index_col=False,sep="\t",quoting=3,names=['stopword'], encoding='utf-8')
print(stopwords.head(20))

def drop_stopwords(contents,stopwords):
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
    return contents_clean,all_words
    
contents = df_content.content_S.values.tolist()    
stopwords = stopwords.stopword.values.tolist()
contents_clean,all_words = drop_stopwords(contents,stopwords)

df_content=pd.DataFrame({'contents_clean':contents_clean})
print(df_content.head())

index = 100 #随便找一篇文章
content_S_str = "".join(content_S[index])
print ('content_S_str and index', index, content_S_str)
print ("  ".join(jieba.analyse.extract_tags(content_S_str, topK=15, withWeight=False)))#选出来5个核心词

df_train=pd.DataFrame({'contents_clean':contents_clean,'label':df_news['category']})
print(df_train.tail())
print(df_train.label.unique())

label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育":5, "教育": 6,"文化": 7,"军事": 8,"娱乐": 9,"时尚": 0}
df_train['label'] = df_train['label'].map(label_mapping) #构建一个映射方法
df_train.head()

x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values, random_state=1)

print(x_train[0][1])

words = []
for line_index in range(len(x_train)):
    try:
        words.append(' '.join(x_train[line_index]))
    except:
        print (line_index)
print('words[0] : ', words[0])
print(len(words))  

texts=["dog cat fish","dog cat cat","fish bird", 'bird'] #为了简单期间，这里4句话我们就当做4篇文章了
cv = CountVectorizer() #词袋词频统计
cv_fit=cv.fit_transform(texts) #转换数据


print(cv.get_feature_names())
print(cv_fit.toarray())

print(cv_fit.toarray().sum(axis=0))

vec = CountVectorizer(analyzer='word', max_features=4000, lowercase = False)# 最大特征程度
feature = vec.fit_transform(words)
print('new feature', feature)
print(feature.shape)

classifier = MultinomialNB()
classifier.fit(feature, y_train)

test_words = []
for line_index in range(len(x_test)):
    try:
        #
        test_words.append(' '.join(x_test[line_index]))
    except:
         print (line_index)
print(test_words[0])

print(classifier.score(vec.transform(test_words), y_test))

X_test = ['卡尔 敌法师 蓝胖子 小小','卡尔 敌法师 蓝胖子 痛苦女王']

tfidf=TfidfVectorizer() #使用tf统计
weight=tfidf.fit_transform(X_test).toarray()
word=tfidf.get_feature_names()
print (weight)
for i in range(len(weight)):  
    print (u"第", i, u"篇文章的tf-idf权重特征")
    for j in range(len(word)):
        print (word[j], weight[i][j])

vectorizer = TfidfVectorizer(analyzer='word', max_features=4000,  lowercase = False)
vectorizer.fit(words)
classifier = MultinomialNB()
classifier.fit(vectorizer.transform(words), y_train)
classifier.score(vectorizer.transform(test_words), y_test)

