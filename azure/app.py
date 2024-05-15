import logging
import pandas as pd
from confluent_kafka import Consumer, KafkaException, TopicPartition
from flask import Flask, render_template, jsonify, redirect, url_for, Response
from io import BytesIO
from query import recommend_tracks
import json

# Read data from a json file in current directory.
with open('data.json', 'r') as file:
    data = json.load(file)

# Set up logging
logging.basicConfig(level=logging.WARN)

# Kafka details
topic = "streamed_music"
group_id = "music_consumer_group"

# Load tracks data into DataFrame
tracks_csv = "cleaned_tracks.csv"
tracks_df = pd.read_csv(tracks_csv)
tracks_df["track_date_created"] = pd.to_datetime(
    tracks_df["track_date_created"]
).dt.strftime("%B  %d, %Y")

# Create a Kafka consumer
consumer = Consumer(
    {
        "bootstrap.servers": data['bootstrap_servers'],
        "group.id": group_id,
        "auto.offset.reset": "earliest",
    }
)
consumer.subscribe([topic])

# In-memory audio store
audio_store = {}

# Flask app setup
app = Flask(__name__)


@app.route("/")
def index():
    sorted_tracks_df = tracks_df.sort_values("track_id")
    return render_template(
        "index.html", tracks=sorted_tracks_df.to_dict(orient="records")
    )


@app.route("/songs/<int:track_id>")
def song(track_id):
    track = tracks_df[
        tracks_df["track_id"] == track_id].to_dict(orient="records")

    if not track:
        return redirect(url_for("index"))
    track = track[0]

    # Note: Initially send the track data only;
    # recommendations will be fetched asynchronously via JS on website.
    return render_template("songs.html", track=track)


def get_audio_data(track_id):
    # preparing track id for lookup
    track_id = str(track_id).zfill(6).encode().decode("utf-8") + ".mp3"
    print(f"Looking for track: {track_id}")
    consumer = Consumer(
        {
            "bootstrap.servers": data['bootstrap_servers'],
            "auto.offset.reset": "earliest",
            "group.id": "audio_consumer_group",
        }
    )

    # Manually assign the consumer to the beginning of the topic.
    tp = TopicPartition("streamed_music", 0, 0)  # partition, offset
    consumer.assign([tp])

    audio_data = None

    for _ in range(3):
        # Seek to the beginning of the topic.
        consumer.seek(tp)
        while True:
            msg = consumer.poll(1)

            if msg is None:  # no messages in the topic.
                break

            if msg.error():
                if msg.error().code() == KafkaException._PARTITION_EOF:
                    break  # Break the inner loop if end of partition.
                else:
                    logging.error(f"Consumer error: {msg.error()}")
                    consumer.close()
                    break

            if msg.key().decode("utf-8") == track_id:
                audio_data = msg.value()
                print(f"Found track: {track_id}")
                consumer.close()
                break

        if audio_data is not None:
            break  # Break the outer loop if track is found.

    return audio_data


@app.route("/navigate_song/<int:track_id>/<string:direction>")
def navigate_song(track_id, direction):
    try:
        sorted_tracks = tracks_df\
            .sort_values("track_id")\
            .reset_index(drop=True)
        current_index = sorted_tracks[
            sorted_tracks["track_id"] == track_id
        ].index[0]

        if direction == "next":
            new_index = (current_index + 1) % len(sorted_tracks)
        elif direction == "prev":
            new_index = (current_index - 1) % len(sorted_tracks)
        else:
            return jsonify({"success": False, "error": "Invalid direction"})

        new_track_id = sorted_tracks.iloc[new_index]["track_id"]
        return jsonify({"success": True, "track_id": new_track_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/audio/<string:track_id>")
def serve_audio(track_id):
    logging.warning(f"serving audio file {track_id}")
    audio_data = get_audio_data(track_id)
    if audio_data:
        return Response(
            BytesIO(audio_data),
            mimetype="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename={track_id}.mp3"},
        )
    else:
        return redirect(url_for("index"))


@app.route("/get_recommendations/<int:track_id>")
def get_recommendations(track_id):
    try:
        # Log the track ID being processed.
        print(f"Fetching recommendations for track ID: {track_id}")

        # Generate recommendations
        recommended_ids = recommend_tracks(track_id)

        # Log the recommended track IDs.
        print(f"Recommended track IDs: {recommended_ids}")

        # Changing track_id format back to that of the original dataset.
        recommended_ids = [int(track_id[:-4]) for track_id in recommended_ids]

        # Fetch the recommended tracks details.
        recommended_tracks = tracks_df[
            tracks_df["track_id"].isin(recommended_ids)
        ].to_dict(orient="records")

        # Log the recommended tracks.
        print(f"Recommended tracks: {recommended_tracks}")

        return jsonify(recommended_tracks)
    except Exception as e:
        print(f"Error fetching recommendations: {str(e)}")
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
