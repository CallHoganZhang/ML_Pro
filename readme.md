In  text_classification,you can find some floder including text data  divide into train,test,stop words.In naive_bayes,you can choose GaussianNB,MultinomialNB,BernoulliNB.In most condition, BernoulliNB is worse than MultinomialNB except in small amount data.Besides,using TfidfVectorizer this function,you will get the matrix which can show feature.

In breast_cance_data-master,I use the data of breast_cance test accuracy of different SVM model.In this data, you will find the mean information,se (stand error, you know) information and worst imformation.I divide into three group according to these different label and show visualization by seaborn,matplot.correlation picture and heatmap will show.After StandardScaler transform,you can choose different model to obverse thier accuracy.