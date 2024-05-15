from azure.storage.blob import BlobServiceClient
from confluent_kafka import Producer
import logging
import json

# Read data from a json file in current directory.
with open('data.json', 'r') as file:
    data = json.load(file)

# Create a blob service client
blob_client = BlobServiceClient.from_connection_string(
    data["connection_string"])

# Get the container client
container_client = blob_client.get_container_client(
    data["container_name"])

# Kafka details
topic = "streamed_music"

# Create a Kafka producer
producer = Producer(
    {
        "bootstrap.servers": data['bootstrap_servers'],
        "message.max.bytes": 11210584,
        "queue.buffering.max.messages": 100000,
    }
)

# Counter for messages
message_counter = 0

# Iterate over the virtual directories (000 to 155)
for i in range(156):
    directory = f"fma_large/{i:03d}/"  # Folling directory structure from fma.
    # Send the blobs (songs) in the directory to the Kafka topic
    for blob in container_client.list_blobs(name_starts_with=directory):
        blob_client = container_client.get_blob_client(blob.name)
        song = blob_client.download_blob().readall()

        if song is not None and len(song) > 0:
            producer.produce(topic, key=blob.name, value=song)
            message_counter += 1
            print(f"Sent blob: {blob.name}")

            if message_counter % 50 == 0:
                producer.flush()
        else:
            logging.warning(f"Empty or missing blob: {blob.name}")
    producer.flush()
