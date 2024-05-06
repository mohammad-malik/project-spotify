import pandas as pd


def preprocess_data(file_path):
    df = pd.read_csv(
        file_path,
        usecols=[
            "track_id",
            "track_title",
            "artist_name",
            "album_title",
            "track_duration",
            "track_date_created",
            "track_genres",
        ],
    )

    df["track_date_created"] = pd.to_datetime(
        df["track_date_created"])

    df["track_date_created"] = df["track_date_created"] \
        .dt.strftime("%Y-%m-%d %H:%M:%S")

    df["track_duration"] = df["track_duration"].apply(
        lambda x: (
            int(x.split(":")[0]) * 60 + int(x.split(":")[1])
            if isinstance(x, str)
            else 0
        )
    )
    df["track_genres"] = df["track_genres"].apply(
        lambda x: (
            ", ".join([g["genre_title"] for g in eval(x)])
            if isinstance(x, str)
            else None
        )
    )

    df.to_csv("cleaned_tracks.csv", index=False)
    return df


tracks_df = preprocess_data("tracks.csv")
