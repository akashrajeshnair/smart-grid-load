#!/usr/bin/env python3
import json, random, time
from datetime import datetime
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_reading():
    voltage = round(random.uniform(210, 250), 2)
    current = round(random.uniform(1, 20), 2)
    power = round(voltage * current / 1000, 3)
    reactive_power = round(power * random.uniform(0.1, 0.3), 3)
    pf = round(random.uniform(0.8, 1.0), 2)
    solar = round(random.uniform(0, 5), 2)
    wind = round(random.uniform(0, 3), 2)
    grid = round(max(power - (solar + wind), 0), 2)
    volt_fluct = round(random.uniform(0, 5), 2)
    temp = round(random.uniform(25, 45), 2)
    humidity = round(random.uniform(20, 90), 2)
    price = round(random.uniform(0.08, 0.25), 3)

    return {
        "Timestamp": datetime.utcnow().isoformat(),
        "Voltage (V)": voltage,
        "Current (A)": current,
        "Power Consumption (kW)": power,
        "Reactive Power (kVAR)": reactive_power,
        "Power Factor": pf,
        "Solar Power (kW)": solar,
        "Wind Power (kW)": wind,
        "Grid Supply (kW)": grid,
        "Voltage Fluctuation (%)": volt_fluct,
        "Temperature (°C)": temp,
        "Humidity (%)": humidity,
        "Electricity Price (USD/kWh)": price,
    }


if __name__ == "__main__":
    print("🚀 Sending smart-grid readings to Kafka topic 'smartgrid' ...")
    while True:
        reading = generate_reading()
        producer.send("smartgrid", reading)
        producer.flush()
        time.sleep(1)   # 1 reading per second
