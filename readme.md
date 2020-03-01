In text_classification,
you can find some floder including text data divided into train,test,stop words.
In this floder project, I want to do serveal experiment in order to help me improve understanding 
machine learing algorithm and thier application scenarios.

The train set and test set including sports,school,literature and woman.
You can using these data realize the test of naive_bayes algorithm.
In naive_bayes,you can choose differe1 model such as GaussianNB,MultinomialNB,BernoulliNB
to test this code.And you will get some quota like accuracy and some .
In most condition, BernoulliNB is worse than MultinomialNB except in small amount data.
Besides,if you use TfidfVectorizer this function,you will get the matrix which can show feature.


In breast_cance_data-master,
I use the data of breast_cance test accuracy of different SVM model.
By there data, you will find the mean information,se(stand error) information and worst imformation.
I divide into three group according to these different label and 
show visualization by seaborn,matplot.
Correlation picture and heatmap will show through above method.
After StandardScaler transform,you can choose different model to obverse thier accuracy.
In the choice of SVM model,I provide SVC and LinearSVC for fitting data.
Besides, before fitting the data,i will normalized data in order to ensure that
the mean of the data is 0, and the variance is 1.

In the tree_weather project,i want to predict the weather by randomForest algorithm.
In this project,I tring to the one_hot coding,you can processing your data by this method.
Through this method,we can get digit rather than other features like male, location 
which comtuer can not process it.
I test the capability of this model through method in different amount data, different features.
You can find these difference result in different amount or different features.
Besides, I also paint some picture through matplot,you can observer the tendency 
by after  time processing.