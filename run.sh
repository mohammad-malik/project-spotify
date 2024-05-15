#!/bin/bash

echo "Starting Zookeeper and Kafka."
screen -dmS zookeeper bash -c "zookeeper-server-start.sh $KAFKA_HOME/config/zookeeper.properties"
screen -dmS kafka bash -c "kafka-server-start.sh $KAFKA_HOME/config/server.properties"

echo "Running the producer."
python3 ./producer.py

echo "Running the feature extraction script."
python3 ./feature_extraction.py

echo "Running preprocessing script."
python3 ./preprocessing_tracks_metadata.py

# echo "Running the train model script."
# python3 ./train_model.py

echo "Running Flask App."
python3 ./app.py
