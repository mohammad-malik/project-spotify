import pandas as pd


def preprocess_data(file_path, files_in_dataset_path):
    # Load the list of track_ids from the "files_in_dataset" file
    with open(files_in_dataset_path, "r") as f:
        files_in_dataset = set(
            int(line.strip().removesuffix('.mp3')) for line in f
        )

    df = pd.read_csv(
        file_path,
        usecols=[
            "track_id",
            "track_title",
            "track_url",
            "artist_name",
            "album_title",
            "track_duration",
            "track_date_created",
            "track_genres",
        ],
    )

    # Only keep records where the track_id is in the "files_in_dataset" list
    df = df[df["track_id"].isin(files_in_dataset)]

    # removing characters such as ', (, ), and " from the track_title
    df["track_title"] = df["track_title"].str.replace(
        r"[',()\"\\-]",
        "",
        regex=True
    )

    df["track_date_created"] = pd.to_datetime(df["track_date_created"])
    df["track_date_created"] = df["track_date_created"].dt.strftime(
                                                        "%Y-%m-%d %H:%M:%S")

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


tracks_df = preprocess_data("raw_tracks.csv", "files_in_dataset.txt")
print(f"Tracks found: {len(tracks_df)}")
