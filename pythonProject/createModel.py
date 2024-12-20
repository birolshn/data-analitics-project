from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt



import logging

# Logging configuration
logging.basicConfig(
    filename='model_training.log',  # Log file
    filemode='a',  # File mode: 'a' add file
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
)

logger = logging.getLogger(__name__)  # A file specific logger

# Function that performs training and prediction
def train_and_evaluate_model(spark):

    batch_df = spark.read.csv("heart.csv", header=True, inferSchema=True)  

    data = data_preprocess(batch_df)
    # Train-test split
    train, test = data.randomSplit([0.8, 0.2], seed=42)

    # Train the model
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    logger.info("Model training...")
    model = lr.fit(train)

    logger.info("Model train completed.")
    

    # Test the model
    predictions = model.transform(test)

    print("\nPrediction Results:")
    predictions.select("features", "label", "prediction", "probability").show(10, truncate=False)

    # Evaluate the model
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    roc_auc = evaluator.evaluate(predictions)
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Count the number of correct and incorrect predictions
    correct_preds = predictions.filter(predictions["label"] == predictions["prediction"]).count()
    total_preds = predictions.count()
    accuracy = correct_preds / total_preds
    logger.info(f"Accuracy: {accuracy:.4f} ({correct_preds}/{total_preds})")
    print(f"\nAccuracy: {accuracy:.4f} ({correct_preds}/{total_preds})")

    # Export predictions
    predictions_pd = predictions.select("label", "prediction", "probability").toPandas()

    fpr, tpr, _ = roc_curve(predictions_pd["label"], predictions_pd["probability"].apply(lambda x: x[1]))
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.show()

    return model



def data_preprocess(df):
    # Convert categorical variables into numerical form
    indexer = StringIndexer(inputCols=["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"],
                            outputCols=["Sex_indexed", "ChestPainType_indexed", "RestingECG_indexed", "ExerciseAngina_indexed", "ST_Slope_indexed"])
    encoder = OneHotEncoder(inputCols=["Sex_indexed", "ChestPainType_indexed", "RestingECG_indexed", "ExerciseAngina_indexed", "ST_Slope_indexed"],
                            outputCols=["Sex_encoded", "ChestPainType_encoded", "RestingECG_encoded", "ExerciseAngina_encoded", "ST_Slope_encoded"])


    # Assemble features
    assembler_raw = VectorAssembler(
        inputCols=["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
                   "Sex_encoded", "ChestPainType_encoded", "RestingECG_encoded", "ExerciseAngina_encoded", "ST_Slope_encoded"],
        outputCol="features_raw",

    )

    # Scale numerical features
    scaler = StandardScaler(inputCol="features_raw", outputCol="features_scaled")

    # Pipeline
    pipeline = Pipeline(stages=[indexer, encoder, assembler_raw, scaler])
    processed_data = pipeline.fit(df).transform(df)


    # Prepare features and label
    data = processed_data.select(col("features_scaled").alias("features"), col("HeartDisease").alias("label"))

    return data


