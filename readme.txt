hadoop:
start-dfs.bat
start-yarn.bat

kafka:
.\zookeeper-server-start.bat ..\..\config\zookeeper.properties
.\kafka-server-start.bat ..\..\config\server.properties
python consumer.py
python producer.py

spark: set the spark python home in the shell
spark-submit --master local[*] preprocess.py
spark-submit --master local[*] analyse.py
spark-submit ^    --master local[*] ^    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.6 ^    C:\Users\USER\Desktop\Projects\smart-grid-load\model\deploy_model.py

flask:
python app.py
