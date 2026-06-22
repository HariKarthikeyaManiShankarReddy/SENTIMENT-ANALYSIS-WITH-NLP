# importing required libraries for the project
import pandas as pd
import string
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#loading dataset 
df=pd.read_csv('movie_reviews_dataset.csv')
print("Columns: ",df.columns.tolist())

df = df[['review', 'sentiment']].dropna()
df = df[df['sentiment'].isin(['positive', 'negative'])]

# cleaning data
def clean_data(review):
    review = review.lower()
    review = review.translate(str.maketrans('', '', string.punctuation))
    review = re.sub(r'\d+', ' ', review)
    return review

df['cleaned_data'] = df['review'].apply(clean_data)
x = df['cleaned_data']
y = df['sentiment']
# train test splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
# stratify ensures that the training and testing datasets maintain the exact same proportion of class labels as the original dataset

vector = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words='english')
x_train_tfidf = vector.fit_transform(x_train)
x_test_tfidf = vector.transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train_tfidf, y_train)
y_pred = model.predict(x_test_tfidf)

print("Accuracy score: ", accuracy_score(y_test, y_pred))
print("\nClassification report of model is:\n", classification_report(y_test, y_pred))
labels = ['negative', 'positive']
cm = confusion_matrix(y_test, y_pred, labels=labels)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
