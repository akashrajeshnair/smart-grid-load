#!/usr/bin/env python3
import json, time, os
from kafka import KafkaConsumer

# HDFS command wrappers
HDFS_DIR = "/user/akash/smartgrid_data"
LOCAL_TMP = "/tmp/smartgrid_buffer.json"

os.system(f"hdfs dfs -mkdir -p {HDFS_DIR}")

consumer = KafkaConsumer(
    "smartgrid",
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='smartgrid-consumer'
)

buffer = []
BATCH_SIZE = 20  # write every 20 messages

print("📥 Consuming from Kafka and writing to HDFS...")

for message in consumer:
    buffer.append(message.value)

    if len(buffer) >= BATCH_SIZE:
        with open(LOCAL_TMP, "w") as f:
            for record in buffer:
                f.write(json.dumps(record) + "\n")

        timestamp = int(time.time())
        hdfs_path = f"{HDFS_DIR}/batch_{timestamp}.json"
        os.system(f"hdfs dfs -put -f {LOCAL_TMP} {hdfs_path}")

        print(f"✅ Wrote batch of {len(buffer)} records to {hdfs_path}")
        buffer.clear()
