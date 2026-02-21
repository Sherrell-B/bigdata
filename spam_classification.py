from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


# Config

HIVE_TABLE = "spambase"
OUTPUT_PATH = "hdfs:///tmp/spam_classification_output"   # HDFS path

# Spark Session
spark = (
    SparkSession.builder
    .appName("Spam Email Classification")
    .enableHiveSupport()
    .getOrCreate()
)

print("="*80)
print("SPAM EMAIL CLASSIFICATION WITH SPARK MLPIPELINE")
print("="*80)

# Load and clean data

df = spark.sql(f"SELECT * FROM {HIVE_TABLE}").na.drop()
print(f"Dataset size after dropping nulls: {df.count()} records")

# Feature preparation

feature_cols = [c for c in df.columns if c != "class"]
pipeline = Pipeline(stages=[
    VectorAssembler(inputCols=feature_cols, outputCol="features_raw"),
    StandardScaler(inputCol="features_raw", outputCol="features"),
    LogisticRegression(labelCol="class", featuresCol="features",
                       maxIter=100, regParam=0.01)
])


# Train/Test split

train, test = df.randomSplit([0.7, 0.3], seed=42)
print(f"Training size: {train.count()}, Test size: {test.count()}")

# Train model

model = pipeline.fit(train)

# Predict

predictions = model.transform(test)

# Convert predictions to RDD

predictions_rdd = predictions.select("class", "prediction", "probability") \
    .rdd.map(lambda row: f"{row['class']},{row['prediction']},{row['probability']}")


# Evaluate metrics

binary_eval = BinaryClassificationEvaluator(labelCol="class")
multi_eval = MulticlassClassificationEvaluator(labelCol="class")

metrics_dict = {
    "accuracy":  multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"}),
    "precision": multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"}),
    "recall":    multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"}),
    "f1":        multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"}),
    "auc":       binary_eval.evaluate(predictions)
}


# Convert metrics to RDD

metrics_rdd = spark.sparkContext.parallelize([f"{k},{v}" for k, v in metrics_dict.items()])

# Save RDDs to HDFS

predictions_rdd.saveAsTextFile(f"{OUTPUT_PATH}/predictions")
metrics_rdd.saveAsTextFile(f"{OUTPUT_PATH}/metrics")

print("Predictions and metrics saved to HDFS as RDDs.")
spark.stop()
