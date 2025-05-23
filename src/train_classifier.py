import joblib
import matplotlib.pyplot as plt
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load the dataset
df = pl.read_csv("data/games.csv", null_values=["N/A"])

print(f"Initial dataset: {len(df)}")
df = df.filter(df["Genre"].is_not_null())
df = df.unique(["Name", "Genre"])
print(f"Dataset after some filtering {len(df)}")

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
print(df["Genre"].value_counts())

# Split the dataset into training and testing sets
X = df["cleaned_name"]
y = df["Genre"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a pipeline with a vectorizer and a classifier
pipeline = Pipeline(
    [
        (
            "vectorizer",
            TfidfVectorizer(
                ngram_range=(1, 2),  # Include unigrams and bigrams
                max_df=0.95,  # ignore terms that appear in 95% of docs
                min_df=2,  # ignore terms that appear in only 1 games title
                stop_words="english",  # remove generic words
            ),
        ),
        (
            "classifier",
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
    ]
)

# Train the model
pipeline.fit(X_train, y_train)

ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)
plt.show()

# Test the model
y_pred = pipeline.predict(X_test)

# Evaluate performance
print(classification_report(y_test, y_pred))

model_path = "models/genre_classifier.pkl"
joblib.dump(pipeline, model_path)
print(f"Model saved to {model_path}")
