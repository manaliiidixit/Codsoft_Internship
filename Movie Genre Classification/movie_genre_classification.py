import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("movie_dataset.csv")

tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the max_features based on your dataset size
tfidf_matrix = tfidf_vectorizer.fit_transform(df['plot_summary'])

X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df['genre'], test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
