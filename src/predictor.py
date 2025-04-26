import joblib
import polars as pl

# Load the model from file
model = joblib.load("models/genre_classifier.pkl")

# Predict genre for a new game title
new_title = "The Legend of Zelda: Breath of the Wild"
cleaned_title = new_title.lower()
cleaned_title = "".join(c for c in cleaned_title if c.isalnum() or c.isspace())

print(f"Predicting the genre for {new_title} (cleaned: {cleaned_title})")
# Run prediction
genre_code = model.predict([cleaned_title])[0]

# If you want to map the integer genre code back to its original label:

df = pl.read_csv("data/games.csv", null_values=["N/A"])
genre_mapping = df.select("Genre").unique().sort("Genre").to_series().to_list()
genre_label = genre_mapping[genre_code]

print(f"Predicted genre for '{new_title}': {genre_label}")
