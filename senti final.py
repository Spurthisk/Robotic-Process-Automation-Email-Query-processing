# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 19:56:03 2020

@author: Spoorthy
"""


"""
Predicting reviews for null values in dataset using already given reviews
"""

"""  Importing Libraries """
import numpy as np 
import pandas as pd 

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Commented out IPython magic to ensure Python compatibility.
def func():
       
    import nltk.classify.util
    from nltk.classify import NaiveBayesClassifier
    import numpy as np
    import re
    import string
    import nltk  
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import nltk.classify.util
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression 
    from sklearn.model_selection import validation_curve
    from sklearn.datasets import load_iris
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import learning_curve
    from sklearn.metrics import confusion_matrix
    from sklearn import metrics
    from sklearn.metrics import roc_curve, auc
    from nltk.classify import NaiveBayesClassifier
    import seaborn as sns
    import numpy as np
    import re
    import string
    import nltk
    nltk.download('all')
    # %matplotlib inline
    
    """ Reading Dataset"""
    temp=pd.read_excel("./Mails.xlsx")
        
    temp.head()
        
    permanent = temp['body']
        
    permanent
        
    from nltk.corpus import treebank
        
    from nltk.tokenize import TreebankWordTokenizer
        
    temp = pd.read_excel(r'./Mails.xlsx')
        
    temp.head()
        
    t=temp['body']
        
    t
    
    
   # """Filtering Null values """ 
    permanent = temp[['email' , 'body' , 'rating']]
    print(permanent.isnull().sum()) #Checking for null values
    permanent.head()
    
       
      #  sns.set(style="white")
    
        
        # Compute the correlation matrix
       # corr = permanent.corr()
        
        # Generate a mask for the upper triangle
       # mask = np.zeros_like(corr, dtype=np.bool)
       # mask[np.triu_indices_from(mask)] = True
        
        # Set up the matplotlib figure
       # f, ax = plt.subplots(figsize=(11, 9))
        
        # Generate a custom diverging colormap
       # cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
       ## sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
             #       square=True, linewidths=.5, cbar_kws={"shrink": .5})
           
    check =  permanent[permanent["rating"].isnull()]
    check.head()
        
    df=pd.DataFrame(data=temp)
        
    df
        
    df=df.dropna()
        
    df
        
    small=df['body']
        
    small
          
   # """## Converting them to lower case"""    
        
    df_list = small.values.tolist()
    print (df_list)
        
    df['all_cols'] = temp['body'].str.lower()
        
    df['all_cols']
    
    df1=df['all_cols']
        
    df1
    
   # """ Removing Punctuation """
    
    import string
    string.punctuation
        
    def remove_punctuation(df1):
          text_punct="".join( [c for c in df1 if c not in string.punctuation])
          return text_punct
        
    temp['clean_msg']=df1.apply(lambda x:remove_punctuation(x))
        
    temp.head()
        
    df1=temp['body']
    df1
        
      # """Filtering Not null values """
    senti= permanent[permanent["rating"].notnull()]
    permanent.head()
    
     #  """Classifying text as positive and negative"""
    senti["senti"] = senti["rating"]>=4
    senti["senti"] = senti["senti"].replace([True , False] , ["pos" , "neg"])
        
    
     #  """Count of reviews"""
    senti["senti"].value_counts()
   # senti["senti"].value_counts().plot.bar()
    #senti.plot(kind = 'kde')
        
        
        
        # Cleaning text 
    cleanup_re = re.compile('[^a-z]+')
    def cleanup(sentence):
            sentence = str(sentence)
            sentence = sentence.lower()
            sentence = cleanup_re.sub(' ', sentence).strip()
            #sentence = " ".join(nltk.word_tokenize(sentence))
            return sentence
        
    senti["Summary_Clean"] = senti["body"].apply(cleanup)
    check["Summary_Clean"] = check["body"].apply(cleanup)
        
    # Splitting train and test data
    split = senti[["Summary_Clean" , "senti"]]
    train=split.sample(frac=0.8,random_state=200)
    test=split.drop(train.index)
        
    #Feature Extracter for NLTK Naive bayes classifier
    
    def word_feats(words):
            features = {}
            for word in words:
                features [word] = True
            return features
        
    train["words"] = train["Summary_Clean"].str.lower().str.split()
    test["words"] = test["Summary_Clean"].str.lower().str.split()
    check["words"] = check["Summary_Clean"].str.lower().str.split()
        
    train.index = range(train.shape[0])
    test.index = range(test.shape[0])
    check.index = range(check.shape[0])
    prediction =  {} ## For storing results of different classifiers
        
    train_naive = []
    test_naive = []
    check_naive = []
        
    for i in range(train.shape[0]):
            train_naive = train_naive +[[word_feats(train["words"][i]) , train["senti"][i]]]
    for i in range(test.shape[0]):
            test_naive = test_naive +[[word_feats(test["words"][i]) , test["senti"][i]]]
    for i in range(check.shape[0]):
            check_naive = check_naive +[word_feats(check["words"][i])]
        
        
    classifier = NaiveBayesClassifier.train(train_naive)
    print("NLTK Naive bayes Accuracy : {}".format(nltk.classify.util.accuracy(classifier , test_naive)))
    classifier.show_most_informative_features(5)
    

    #Predicting result of nltk classifier    
    y =[]
    only_words= [test_naive[i][0] for i in range(test.shape[0])]
    for i in range(test.shape[0]):
        y = y + [classifier.classify(only_words[i] )]
    prediction["Naive"]= np.asarray(y)
        
    y1 = []
    for i in range(check.shape[0]):
            y1 = y1 + [classifier.classify(check_naive[i] )]
        
    check["Naive"] = y1
        
    #Building Countvector and Tfidf vector for train , test ,check data
        #pip install wordcloud
        
    from wordcloud import STOPWORDS
        
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    stopwords = set(STOPWORDS)
    stopwords.remove("not")
        
    count_vect = CountVectorizer(min_df=2 ,stop_words=stopwords , ngram_range=(1,2))
    tfidf_transformer = TfidfTransformer()
    
    X_train_counts = count_vect.fit_transform(train["Summary_Clean"])        
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        
        
    X_new_counts = count_vect.transform(test["Summary_Clean"])
    X_test_tfidf = tfidf_transformer.transform(X_new_counts)
        
    checkcounts = count_vect.transform(check["Summary_Clean"])
    checktfidf = tfidf_transformer.transform(checkcounts)
    
    #Learning Curve 
    def fun1(model,x,y):
      train_sizes, train_scores, test_scores = learning_curve(
              model, x, y, cv=None, n_jobs=2, train_sizes=np.linspace(.1, 1.0, 5))
    
      train_scores_mean = np.mean(train_scores, axis=1)
      train_scores_std = np.std(train_scores, axis=1)
      test_scores_mean = np.mean(test_scores, axis=1)
      test_scores_std = np.std(test_scores, axis=1)
    
      trace1 = {
        "name": "Training Scores", 
        "type": "scatter", 
        "x": train_sizes, 
        "y": train_scores_mean
      }
      trace2 = {
        "name": "Test Scores", 
        "type": "scatter", 
        "x": train_sizes, 
        "y": test_scores_mean
      }
    
      #data = go.Data([trace1, trace2])
      layout = {
        "title": "Learning Curve of Linear Regression", 
        "width": 600, 
        "xaxis": {
          "title": "Training Size", 
          "titlefont": {
            "size": 18, 
            "color": "black", 
            "family": "Courier New, monospace"
          }
        }, 
        "yaxis": {
          "title": "Scores", 
          "titlefont": {
            "size": 18, 
            "color": "black", 
            "family": "Courier New, monospace"
          }
        }, 
        "height": 1000, 
        "width" : 1000,
        "autosize": False, 
        "showlegend": True
      }
      #fig = go.Figure(data=data, layout=layout )
      #fig.show()
      plt.grid()
    
      plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
      plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
      plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
      plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")
    
      plt.legend(loc="best")
      plt.show()
    
    
    
    #Multinomial NB Model
    
    from sklearn.naive_bayes import MultinomialNB
    model1 = MultinomialNB().fit(X_train_tfidf , train["senti"])
    prediction['Multinomial'] = model1.predict_proba(X_test_tfidf)[:,1]
    print("Multinomial Accuracy : {}".format(model1.score(X_test_tfidf , test["senti"])))   
    check["multi"] = model1.predict(checktfidf)
    ## Predicting Sentiment for Check which was Null values for rating
    fun1(model1,X_train_tfidf,train["senti"])
    
    
    
    #Bernouli Nb Model
    
    from sklearn.naive_bayes import BernoulliNB
    model2 = BernoulliNB().fit(X_train_tfidf,train["senti"])
    prediction['Bernoulli'] = model2.predict_proba(X_test_tfidf)[:,1]
    print("Bernoulli Accuracy : {}".format(model2.score(X_test_tfidf , test["senti"])))    
    check["Bill"] = model2.predict(checktfidf)
    ## Predicting Sentiment for Check which was Null values for rating
    fun1(model2,X_train_tfidf,train["senti"])
      
    
    
    #LogisticRegression Model
    
    from sklearn import linear_model
    logreg = linear_model.LogisticRegression(solver='lbfgs' , C=1000)
    logistic = logreg.fit(X_train_tfidf, train["senti"])
    prediction['LogisticRegression'] = logreg.predict_proba(X_test_tfidf)[:,1]
    print("Logistic Regression Accuracy : {}".format(logreg.score(X_test_tfidf , test["senti"])))    
    check["log"] = logreg.predict(checktfidf)
    ## Predicting Sentiment for Check which was Null values for rating
    fun1(logreg,X_train_tfidf,train["senti"])
     
    
    #Getting most occuring words in train set 
    words = count_vect.get_feature_names()
    feature_coefs = pd.DataFrame(
           data = list(zip(words, logistic.coef_[0])),
           columns = ['feature', 'coef'])
    feature_coefs.sort_values(by="coef")
      
    
    #Classifier Analysis
    
    #ROC Curve
    def formatt(x):
            if x == 'neg':
                return 0
            if x == 0:
                return 0
            return 1
    vfunc = np.vectorize(formatt)
        
    cmp = 0
    colors = ['b', 'g', 'y', 'm', 'k']
    for model, predicted in prediction.items():
            if model not in 'Naive':
                false_positive_rate, true_positive_rate, thresholds = roc_curve(test["senti"].map(vfunc), predicted)
                roc_auc = auc(false_positive_rate, true_positive_rate)
                plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
                cmp += 1
        
    plt.title('Classifiers comparaison with ROC')
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    test.senti = test.senti.replace(["pos" , "neg"] , [True , False] )
    
    #keys = prediction.keys()
    #for key in ['Multinomial', 'Bernoulli', 'LogisticRegression']:
    #print("{}:".format(key))
    #print(metrics.classification_report(test["senti"], prediction.get(key)>.5, target_names = ["positive", "negative"]))
    #print("\n")
    
    #Testing classifiers with some handwritten samples¶
    #def test_sample(model, sample):
    #sample_counts = count_vect.transform([sample])
    #sample_tfidf = tfidf_transformer.transform(sample_counts)
    #result = model.predict(sample_tfidf)[0]
    #prob = model.predict_proba(sample_tfidf)[0]
    #print("Sample estimated as %s: negative prob %f, positive prob %f" % (result.upper(), prob[0], prob[1]))

    #test_sample(logreg, "The product was good and easy to  use")
    #test_sample(logreg, "the whole experience was horrible and product is worst")
    #test_sample(logreg, "product is not good")
    
    
    #Predicted valuesof classifiers for check on the basis of review text
        
    from wordcloud import WordCloud, STOPWORDS
    stopwords = set(STOPWORDS)
        
        
    mpl.rcParams['font.size']=12                #10 
    mpl.rcParams['savefig.dpi']=100             #72 
    mpl.rcParams['figure.subplot.bottom']=.1 
        
        
    def show_wordcloud(data, title = None):
            wordcloud = WordCloud(
                background_color='white',
                stopwords=stopwords,
                max_words=300,
                max_font_size=40, 
                scale=3,
                random_state=1 # chosen at random by flipping  a coin; it was heads
                
            ).generate(str(data))
            
            fig = plt.figure(1, figsize=(15, 15))
            plt.axis('off')
            if title: 
                fig.suptitle(title, fontsize=20)
                fig.subplots_adjust(top=2.3)
        
            plt.imshow(wordcloud)
            plt.show()
            
    show_wordcloud(senti["Summary_Clean"])
        
    show_wordcloud(senti["Summary_Clean"][senti.senti == "pos"] , title="Postive Words")
        
    show_wordcloud(senti["Summary_Clean"][senti.senti == "neg"] , title="Negitive words")
       
    
        #Writing into files
        #!pip install xlsxwriter
        
    import xlsxwriter
        

    #Negative feedback
        
    file = temp[temp["rating"]<3]
        
    file
        
    file["clean_msg"].iloc[0:]
        
    negative = file["email"].iloc[0:]
        
    negative
        
    workbook=xlsxwriter.Workbook('negative.xlsx')
    worksheet=workbook.add_worksheet()
        
    row=0
    column=0
    
    for item in negative:
            worksheet.write(row,column,item)
            row+=1
            
    ##Positive feedback
    file1 = temp[temp["rating"]>=4]
        
    file1
        
    file1["clean_msg"].iloc[0:]
        
    positive = file1["email"].iloc[0:]
        
    positive
        
    workbook=xlsxwriter.Workbook('positive.xlsx')
    worksheet=workbook.add_worksheet()
        
    row=0
    column=0
    
    for item in positive:
            worksheet.write(row,column,item)
            row+=1
            
    ##Neutral Feedback
    file2 = temp[temp["rating"]==3]
        
    file2
        
    file2["clean_msg"].iloc[0:]
        
    neutral = file2["email"].iloc[0:]
        
    neutral 
        
    workbook=xlsxwriter.Workbook('neutral.xlsx')
    worksheet=workbook.add_worksheet()
        
    row=0
    column=0
    
    for item in neutral :
            worksheet.write(row,column,item)
            row+=1
               
    workbook.close()
func()