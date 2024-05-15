from confluent_kafka import Consumer, Producer, KafkaException
import random

def main():
    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'streaming-group',
        'auto.offset.reset': 'earliest'
    })

    producer = Producer({'bootstrap.servers': 'localhost:9092'})

    consumer.subscribe(['user_activity'])

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                print("Consumer error: {}".format(msg.error()))
                continue

            audio_file = msg.value().decode('utf-8')
            print('Processing:', audio_file)

            # Mock recommendation logic
            recommendation = {
                'original_audio': audio_file,
                'recommendations': [f"Recommended_Song_{i}" for i in range(5)]
            }

            producer.produce('recommendations', key='recommendation', value=str(recommendation))
            producer.poll(0)

    finally:
        consumer.close()
        producer.flush()

if __name__ == "__main__":
    main()