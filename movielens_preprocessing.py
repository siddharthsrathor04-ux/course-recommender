import pandas as pd
import os

def preprocess_movielens(data_path="ml-100k"):
    # ratings
    ratings_path = os.path.join(data_path, "u.data")
    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "course_id", "rating", "timestamp"]
    )

    # items
    items_path = os.path.join(data_path, "u.item")
    
    columns = [
        "course_id", "course_title", "release_date", "video_release_date", "IMDb_URL",
        "unknown","Action","Adventure","Animation","Children","Comedy","Crime",
        "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
        "Romance","Sci-Fi","Thriller","War","Western"
    ]

    items = pd.read_csv(
        items_path,
        sep="|",
        encoding="latin-1",
        names=columns
    )

    # combine genres
    genre_cols = columns[5:]

    def combine_genres(row):
        return ",".join([g for g in genre_cols if row[g] == 1])

    items["category"] = items.apply(combine_genres, axis=1)

    # Keep required columns
    courses = items[["course_id", "course_title", "category"]]

    # Save file as csv
    ratings.to_csv("ratings.csv", index=False)
    courses.to_csv("courses.csv", index=False)

    print("Preprocessing complete!")
    print("Generated files: ratings.csv, courses.csv")


if __name__ == "__main__":
    preprocess_movielens()
