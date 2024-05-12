#!/bin/bash

spark-submit --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 \
    --conf "spark.mongodb.input.uri=mongodb://localhost:27017/music_database.audio_features_small" \
    ./ml_recommendation.py > output_model.txt 2>&1
