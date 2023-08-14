import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
# Assume you have a CSV file with columns "plot_summary" and "genre"
df = pd.read_csv("movie_dataset.csv")

# Step 2: Preprocessing (not shown here, but remember to clean and preprocess text data)

# Step 3: Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the max_features based on your dataset size
tfidf_matrix = tfidf_vectorizer.fit_transform(df['plot_summary'])

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df['genre'], test_size=0.2, random_state=42)

# Step 5: Train the SVM classifier
svm_classifier = SVC(kernel='linear')  # You can try different kernels or hyperparameters
svm_classifier.fit(X_train, y_train)

# Step 6: Make predictions on the test data
y_pred = svm_classifier.predict(X_test)

# Step 7: Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
