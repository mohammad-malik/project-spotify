from flask import Flask, render_template, redirect, url_for
import pandas as pd

app = Flask(__name__)

# Temporary, will be replaced with the cleaned data.
tracks_df = pd.read_csv("cleaned_tracks.csv")
tracks_df["track_date_created"] = pd.to_datetime(
    tracks_df["track_date_created"]
).dt.strftime("%B %d, %Y")


# Main page.
@app.route("/")
def index():
    sorted_tracks_df = tracks_df.sort_values("track_title")
    return render_template(
        "index.html", tracks=sorted_tracks_df.to_dict(orient="records")
    )


# Music playback page.
@app.route("/songs/<int:track_id>")
def song(track_id):
    print(track_id)
    track = tracks_df[
        tracks_df["track_id"] == track_id].to_dict(orient="records")
    if not track:
        return redirect(url_for("index"))
    track = track[0]
    recommendations = tracks_df.sample(5).to_dict(orient="records")
    return render_template(
        "songs.html", track=track, recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5003)
