In text_classification,
you can find some floder including text data divided into train,test,stop words.
The train set and test set including sports,school,literature and woman.
You can using these data realize the test of naive_bayes algorithm.
In naive_bayes,you can choose differe1 model such as GaussianNB,MultinomialNB,BernoulliNB
to test this code.And you will get some index like accuracy.
In most condition, BernoulliNB is worse than MultinomialNB except in small amount data.
Besides,using TfidfVectorizer this function,you will get the matrix which can show feature.


In breast_cance_data-master,
I use the data of breast_cance test accuracy of different SVM model.
By there data, you will find the mean information,se(stand error) information and worst imformation.
I divide into three group according to these different label and 
show visualization by seaborn,matplot.
Correlation picture and heatmap will show through above method.
After StandardScaler transform,you can choose different model to obverse thier accuracy.
In the choice of SVM model,i provide SVC and LinearSVC for fitting data.
Besides, before fitting the data,i will normalized data in order to ensure that
the mean of the data is 0, and the variance is 1.