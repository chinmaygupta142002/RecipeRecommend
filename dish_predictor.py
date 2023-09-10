import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

pd.set_option("display.max_columns", None)
df = pd.read_csv("Ifood_new.csv")
df = df[["ingredients", "name"]]


def preprocess_ingredients(ingredients):
    arr = ingredients.split(", ")
    for i in range(len(arr)):
        arr[i] = arr[i].lower()
        arr[i] = arr[i].replace(" ", "")
    return " ".join(arr)


df["ingredients"] = df["ingredients"].apply(preprocess_ingredients)
cv = CountVectorizer(max_features=5000, stop_words="english")
X_train, X_test, y_train, y_test = train_test_split(df["ingredients"], df["name"])
X_train_count = cv.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_count, y_train)


def suggest_dish():
    ingredients = input("Enter Ingredients separated by comma:")
    ingredients = preprocess_ingredients(ingredients)
    input_vector = cv.transform([ingredients])
    prediction = model.predict(input_vector)
    return f"Suggest Dish: {prediction[0]}"


print(suggest_dish())
























