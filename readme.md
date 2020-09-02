在这个选民项目中，我想做一些实验，以帮助我增进理解
机器学习算法及其应用场景。

在text_classification中，
您会发现一些花絮，包括分为训练，测试，停用词的文本数据。
火车和测试套件包括运动，学校，文学和女性。
您可以使用这些数据来实现naive_bayes算法的测试。
在naive_bayes中，您可以选择诸如GaussianNB，MultinomialNB，BernoulliNB之类的差异模型。
测试此代码。您将获得一些配额，如准确性和一些。
在大多数情况下，除了少量数据外，BernoulliNB比MultinomialNB差。
此外，如果您使用TfidfVectorizer此功能，您将获得可以显示特征的矩阵。

在breast_cance_data-master中，我使用不同SVM模型的breast_cance测试准确性数据。
到那里的数据，您将找到平均值信息，se（标准差）信息和最差的信息。
我根据这些不同的标签分为三类，
显示seaborn，matplot的可视化。
通过以上方法将显示相关图片和热图。
经过StandardScaler转换后，您可以选择其他模型来提高精度。
在选择SVM模型时，我提供SVC和LinearSVC来拟合数据。
此外，在拟合数据之前，我将对数据进行归一化以确保
数据的平均值为0，方差为1。

在tree_weather项目中，我想通过randomForest算法预测天气。
在这个项目中，我尝试使用one_hot编码，您可以通过这种方法处理数据。
通过这种方法，我们可以获得数字而不是其他特征，例如男性，位置
哪个用户无法处理它。
我通过在不同数量的数据，不同特征中使用方法来测试该模型的功能。
您可以找到不同数量或不同功能的差异结果。
另外，我也通过matplot画了一些画，你可以观察趋势
经过时间的处理。


我希望在news_classification项目中比较CountVectorizer和TfidfVectorizer的准确性。
在比较这些方法之前，我先获取可以通过互联网下载并转换为csv文件的数据。
像几乎所有项目一样，我删除可能影响结果的停用词，
这些将无助于拟合数据，因为它们毫无意义。
将预处理数据分为火车数据和测试数据时，可以开始拟合。
如果您对TfidfVectorizer感兴趣，则可以深入了解其原理，就像
以比较本文中出现的数字的方式计算本文中这些单词的权重
和其他文章。最终结果是，TfidfVectorizer的准确性高于CountVectorizer，但不明显。

此外，该项目将在连续租用机器时连续更新。
