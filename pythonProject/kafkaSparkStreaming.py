
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
from pyspark.sql.functions import from_json, col
from createModel import train_and_evaluate_model, data_preprocess


# Spark session
spark = SparkSession.builder \
    .appName("HeartFailurePrediction") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC") \
    .config("spark.driver.extraJavaOptions", "-XX:+UseG1GC") \
    .config("spark.streaming.backpressure.enabled", "true") \
    .getOrCreate()



# Kafka consumer setup
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "heart-failure") \
    .option("startingOffsets", "earliest") \
    .option("maxOffsetsPerTrigger", 1000) \
    .load()

# Define schema for the data
schema = StructType([
    StructField("Age", IntegerType()),
    StructField("Sex", StringType()),
    StructField("ChestPainType", StringType()),
    StructField("RestingBP", FloatType()),
    StructField("Cholesterol", FloatType()),
    StructField("FastingBS", IntegerType()),
    StructField("RestingECG", StringType()),
    StructField("MaxHR", IntegerType()),
    StructField("ExerciseAngina", StringType()),
    StructField("Oldpeak", FloatType()),
    StructField("ST_Slope", StringType()),
    StructField("HeartDisease", IntegerType())
])

# Deserialize JSON data and process it
value_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

model = train_and_evaluate_model(spark)

#Transfer streaming data to foreachBatch to process it as a batch
def process_batch(batch_df, batch_id):    
    data = data_preprocess(batch_df)
    
    # Test the model
    predictions = model.transform(data)
    print("\nPrediction Results:")
    predictions.select("features", "label", "prediction", "probability").show(10, truncate=False)


# Output to console
query = value_df.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("update") \
    .format("console") \
    .start()

query.awaitTermination()


