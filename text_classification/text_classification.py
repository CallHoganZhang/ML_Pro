# 中文文本分类
import os
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

def cut_words(file_path):
    text_with_spaces = ''
    text=open(file_path, 'r', encoding='gb18030').read()  #gb18030收录更加多数字，变长度编码
    textcut = jieba.cut(text)
    for word in textcut:
        text_with_spaces += word + ' '
    return text_with_spaces

def loadfile(file_dir, label):
    file_list = os.listdir(file_dir)
    words_list = []
    labels_list = []
    for file in file_list:
        file_path = os.path.join(file_dir,file)
        words_list.append(cut_words(file_path))
        labels_list.append(label)
    return words_list, labels_list

# train data
train_words_list1, train_labels1 = loadfile('data/train/女性', '女性')
train_words_list2, train_labels2 = loadfile('data/train/体育', '体育')
train_words_list3, train_labels3 = loadfile('data/train/文学', '文学')
train_words_list4, train_labels4 = loadfile('data/train/校园', '校园')

train_words_list = train_words_list1 + train_words_list2 + train_words_list3 + train_words_list4
train_labels = train_labels1 + train_labels2 + train_labels3 + train_labels4

# test data
test_words_list1, test_labels1 = loadfile('data/test/女性', '女性')
test_words_list2, test_labels2 = loadfile('data/test/体育', '体育')
test_words_list3, test_labels3 = loadfile('data/test/文学', '文学')
test_words_list4, test_labels4 = loadfile('data/test/校园', '校园')

test_words_list = test_words_list1 + test_words_list2 + test_words_list3 + test_words_list4
test_labels = test_labels1 + test_labels2 + test_labels3 + test_labels4

stop_words = open('data/stop/stopword.txt', 'r', encoding='utf-8').read()
stop_words = stop_words.encode('utf-8').decode('utf-8-sig') # 列表头部\ufeff处理
stop_words = stop_words.split('\n')

# 计算单词权重
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)

train_features = tf.fit_transform(train_words_list)
test_features = tf.transform(test_words_list)

clf = MultinomialNB(alpha=0.001).fit(train_features, train_labels)
predicted_labels=clf.predict(test_features)

print('准确率为：', metrics.accuracy_score(test_labels, predicted_labels))