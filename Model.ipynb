import numpy as np
import pandas as pd

df = pd.read_csv('/content/spam.csv',encoding='latin-1')

df.shape
df.info()

df.isnull().sum()

df = df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis = 1) # Changed 'column' to 'columns'

df.head()
df['target'].value_counts()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

df.head()
df.drop_duplicates(keep = 'first',inplace = True)

import matplotlib.pyplot as plt
import seaborn as sns

df['target'].value_counts()
fig = plt.figure(figsize=(8,5))
sns.countplot(x = df['target'])

fig = plt.figure(figsize=(8,5))
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
plt.show()

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

df['num_char'] = df['text'].apply(len)

df['text'].apply(lambda x:nltk.word_tokenize(x))

df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
fig = plt.figure(figsize=(8,5))
sns.countplot(x = df['num_words'])

df['num_sent'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
# for ham mails
df.loc[df['target'] == 0,['num_char','num_words','num_sent']].describe()
#for spam mails
df.loc[df['target'] == 1,['num_char','num_words','num_sent']].describe()

fig = plt.figure(figsize=(12,15))
# Use plt.subplot instead of fig.subplot
plt.subplot(2,2,1) # Changed fig.subplot to plt.subplot. Use plt to create subplots
sns.histplot(df[df['target'] == 0]['num_char'])
sns.histplot(df[df['target'] == 1]['num_char'],color = 'r')
plt.show()

fig = plt.figure(figsize=(12,15))
# Use plt.subplot instead of fig.subplot
plt.subplot(2,2,1) # Changed fig.subplot to plt.subplot. Use plt to create subplots
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color = 'r')
plt.show()

sns.heatmap(df[['target','num_char','num_words','num_sent']].corr(),annot = True)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  y = []
  for i in text:
    if i.isalnum():
      y.append(i)
  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()
  ps = PorterStemmer()
  for i in text:
    y.append(ps.stem(i))

  return ' '.join(y)


df['transformed_text'] = df['text'].apply(transform_text)

from wordcloud import WordCloud


# Increase width and height, limit maximum words, and adjust font sizes
wc = WordCloud(width=800,
               height=800,
               min_font_size=10,
               background_color='white',
               max_words=30,  # Limit the number of words
               max_font_size=150)  # Adjust maximum font size if needed

spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep = " "))

plt.figure(figsize=(15,6))
plt.imshow(spam_wc)

not_spam_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep = " "))
plt.figure(figsize=(15,6))
plt.imshow(not_spam_wc)

spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
from collections import Counter
Counter(spam_corpus).most_common(10)

pd.DataFrame(Counter(spam_corpus).most_common(10))

most_common_df = pd.DataFrame(Counter(spam_corpus).most_common(30), columns=['Word', 'Count'])

# Use the DataFrame to create the barplot, specifying 'x' and 'y'
sns.barplot(x='Word', y='Count', data=most_common_df)
plt.xticks(rotation='vertical')  # Rotate x-axis labels for better readability
plt.show()
not_spam_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        not_spam_corpus.append(word)

most_common_df = pd.DataFrame(Counter(not_spam_corpus).most_common(30), columns=['Word', 'Count'])

# Use the DataFrame to create the barplot, specifying 'x' and 'y'
sns.barplot(x='Word', y='Count', data=most_common_df)
plt.xticks(rotation='vertical')  # Rotate x-axis labels for better readability
plt.show()
# prompt: import tfidf vectorizer

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tf = TfidfVectorizer(max_features=3000)

X = tf.fit_transform( df['transformed_text']).toarray()
X

Y = df['target'].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 2)

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score

gn = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gn.fit(X_train,Y_train)
y_pred1 = gn.predict(X_test)
print(accuracy_score(Y_test,y_pred1))
print(confusion_matrix(Y_test,y_pred1))
print(precision_score(Y_test,y_pred1))

sns.heatmap(confusion_matrix(Y_test,y_pred1),annot = True,fmt = 'd')

mnb.fit(X_train,Y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(Y_test,y_pred2))
print(confusion_matrix(Y_test,y_pred2))
print(precision_score(Y_test,y_pred2))

sns.heatmap(confusion_matrix(Y_test,y_pred2),annot = True,fmt = 'd')

bnb.fit(X_train,Y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(Y_test,y_pred3))
print(confusion_matrix(Y_test,y_pred3))
print(precision_score(Y_test,y_pred3))


sns.heatmap(confusion_matrix(Y_test,y_pred3),annot = True,fmt = 'd')

# prompt: import logistic regression and random forest

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Initialize Logistic Regression model
log_reg = LogisticRegression()

# Train the Logistic Regression model
log_reg.fit(X_train, Y_train)

# Make predictions using Logistic Regression
y_pred_lr = log_reg.predict(X_test)

# Evaluate Logistic Regression
print("Logistic Regression Accuracy:", accuracy_score(Y_test, y_pred_lr))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(Y_test, y_pred_lr))
print("Logistic Regression Precision:", precision_score(Y_test, y_pred_lr))
sns.heatmap(confusion_matrix(Y_test, y_pred_lr), annot=True, fmt='d')
plt.show()


# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Train the Random Forest Classifier
rf_classifier.fit(X_train, Y_train)

# Make predictions using Random Forest
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate Random Forest
print("\nRandom Forest Accuracy:", accuracy_score(Y_test, y_pred_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(Y_test, y_pred_rf))
print("Random Forest Precision:", precision_score(Y_test, y_pred_rf))
sns.heatmap(confusion_matrix(Y_test, y_pred_rf), annot=True, fmt='d')
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)

clfs = {
    'SVC' : svc,
    'KN' : knc,
    'NB': mnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'AdaBoost': abc,
    'BgC': bc,
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}

def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test,Y_pred)
    precision = precision_score(Y_test,Y_pred)

    return accuracy,precision

accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():

    current_accuracy,current_precision = train_classifier(clf, X_train,Y_train,X_test,Y_test)

    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)
performance_df

import pickle
pickle.dump(tf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))

from google.colab import files
files.download('vectorizer.pkl')
files.download('model.pkl')

