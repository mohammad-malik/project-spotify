import logging
import pandas as pd
from confluent_kafka import Consumer, KafkaException, TopicPartition
from flask import Flask, render_template
from flask import redirect, url_for, Response
from io import BytesIO
from query import recommend_tracks
from flask_ngrok import run_with_ngrok

# Set up logging
logging.basicConfig(level=logging.WARN)


# Kafka details
bootstrap_servers = "localhost:9092"
topic = "streamed_music_local"
group_id = "music_consumer_group"

# Load tracks data into DataFrame
tracks_csv = "cleaned_tracks.csv"
tracks_df = pd.read_csv(tracks_csv)
tracks_df["track_date_created"] = pd.to_datetime(
    tracks_df["track_date_created"]
).dt.strftime("%B %d, %Y")

# Create a Kafka consumer
consumer = Consumer(
    {
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id,
        "auto.offset.reset": "earliest",
    }
)
consumer.subscribe([topic])

# In-memory audio store
audio_store = {}

# Flask app setup
app = Flask(__name__)

# Run Flask app with ngrok when running locally
run_with_ngrok(app)


@app.route("/")
def index():
    sorted_tracks_df = tracks_df.sort_values("track_title")
    return render_template(
        "index.html", tracks=sorted_tracks_df.to_dict(orient="records")
    )


@app.route("/songs/<int:track_id>")
def song(track_id):
    track = tracks_df[tracks_df["track_id"] == track_id].to_dict(orient="records")

    if not track:
        return redirect(url_for("index"))
    track = track[0]

    # will be replaced with ml script returning 5 recommendations
    recommendations = recommend_tracks(track_id)

    return render_template("song.html", track=track, recommendations=recommendations)


def get_audio_data(track_id):
    # preparing track id for lookup
    track_id = str(track_id).zfill(6).encode().decode("utf-8") + ".mp3"
    logging.warning(f"serving audio file {track_id}")

    consumer = Consumer(
        {
            "bootstrap.servers": "localhost:9092",
            "auto.offset.reset": "earliest",
            "group.id": "audio_consumer_group",
        }
    )

    # Manually assign the consumer to the beginning of the topic
    tp = TopicPartition("streamed_music_local", 0, 0)  # partition, offset
    consumer.assign([tp])

    while True:
        msg = consumer.poll(1)
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaException._PARTITION_EOF:
                continue
            else:
                logging.error(f"Consumer error: {msg.error()}")
                consumer.close()
                break

        if msg.key().decode("utf-8") == track_id:
            print("found the audio file")
            audio_data = msg.value()
            consumer.close()
            break

    return audio_data


@app.route("/audio/<string:track_id>")
def serve_audio(track_id):
    logging.warning(f"serving audio file {track_id}")
    audio_data = get_audio_data(track_id)
    if audio_data:
        return Response(
            BytesIO(audio_data),
            mimetype="audio/mpeg",
            headers={"Content-Disposition": f"attachment; filename={track_id}.mp3"},
        )
    else:
        return redirect(url_for("index"))


@app.route("/get_recommendations/<int:track_id>")
def get_recommendations(track_id):
    track_id = str(track_id).zfill(6).encode().decode("utf-8") + ".mp3"
    recommendations = find_closest_tracks(track_id)
    # Get the track details of each recommendation from the tracks DataFrame
    recommendations = [
        tracks_df[tracks_df["track_id"] == rec].to_dict(orient="records")[0]
        for rec in recommendations
    ]

    return recommendations


if __name__ == "__main__":
    # Start the Flask web server
    app.run()
