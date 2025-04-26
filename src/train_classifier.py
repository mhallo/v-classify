import joblib
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load the dataset
df = pl.read_csv("data/games.csv", null_values=["N/A"])

# Clean the title names for the classifier
df = df.with_columns(
    cleaned_name=pl.col("Name")
    .str.to_lowercase()
    .str.replace_all(r"[^a-z0-9\s]", "")  # strip all non-alphanumeric
)

# Convert the Genres into integers for their respective categories
df = df.with_columns(pl.col("Genre").cast(pl.Categorical).to_physical().alias("Genre"))

print(df.schema)
print(df.head())

# Split the dataset into training and testing sets
X = df["cleaned_name"]
y = df["Genre"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a pipeline with a vectorizer and a classifier
pipeline = Pipeline(
    [
        ("vectorizer", TfidfVectorizer()),  # Convert text to feature vectors
        (
            "classifier",
            MultinomialNB(),
        ),  # Classifier: Naive Bayes for text classification
    ]
)

# Train the model
pipeline.fit(X_train, y_train)

# Test the model
y_pred = pipeline.predict(X_test)

# Evaluate performance
print(classification_report(y_test, y_pred))

model_path = "models/genre_classifier.pkl"
joblib.dump(pipeline, model_path)
print(f"Model saved to {model_path}")
