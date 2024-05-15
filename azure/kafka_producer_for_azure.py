from azure.storage.blob import BlobServiceClient
from confluent_kafka import Producer
import logging

# Azure blob storage details
connection_string = "DefaultEndpointsProtocol=https;AccountName=waw;AccountKey=EyZhHsC72nxfQwvFA9Akl9hsrlQ95MdGaMS6VCJUAdoM2UdT5ODWfvYh/HDqj2EnOflDPj/GJLWu+AStSyGeRg==;EndpointSuffix=core.windows.net"
container_name = "container2"

# Create a blob service client
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Get the container client
container_client = blob_service_client.get_container_client(container_name)

# Kafka details
bootstrap_servers = "10.0.0.4:9092, 10.0.0.5:9092, 192.168.0.4:9092, 192.168.0.5:9092"
topic = "streamed_music_azure_blob"

# Create a Kafka producer
producer = Producer(
    {
        "bootstrap.servers": bootstrap_servers,
        "message.max.bytes": 11210584,
        "queue.buffering.max.messages": 100000,
    }
)

# Counter for messages
message_counter = 0

# Iterate over the virtual directories (000 to 155)
for i in range(156):
    directory = f"fma_large/{i:03d}/"  # Format the directory name as a three-digit number with leading zeros
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

