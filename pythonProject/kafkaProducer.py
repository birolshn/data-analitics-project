from kafka import KafkaProducer
import pandas as pd
import json
import time

# Load dataset
data = pd.read_csv('heart.csv')

# Kafka producer setup
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Send data to Kafka topic
print("Heart failure data sent to Kafka topic 'heart-failure'")
for index, row in data.iterrows():
    producer.send('heart-failure', value=row.to_dict())
    time.sleep(0.05)  # Simulate streaming
