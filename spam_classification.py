from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# ===============================
# Config
# ===============================
HIVE_TABLE = "spambase"
OUTPUT_PATH = "hdfs:///tmp/spam_classification_output"   # HDFS path to save predictions & metrics

# ===============================
# Spark Session
# ===============================
spark = (
    SparkSession.builder
    .appName("Spam Email Classification")
    .enableHiveSupport()
    .getOrCreate()
)

print("="*80)
print("SPAM EMAIL CLASSIFICATION WITH SPARK MLPIPELINE")
print("="*80)

# ===============================
# Load and clean data
# ===============================
print("Loading data from Hive table...")
df = spark.sql(f"SELECT * FROM {HIVE_TABLE}").na.drop()
print(f"Dataset size after dropping nulls: {df.count()} records")

# ===============================
# Feature preparation
# ===============================
feature_cols = [c for c in df.columns if c != "class"]
print(f"Preparing features using {len(feature_cols)} columns")

pipeline = Pipeline(stages=[
    VectorAssembler(inputCols=feature_cols, outputCol="features_raw"),
    StandardScaler(inputCol="features_raw", outputCol="features"),
    LogisticRegression(labelCol="class", featuresCol="features",
                       maxIter=100, regParam=0.01)
])

# ===============================
# Train/Test split
# ===============================
train, test = df.randomSplit([0.7, 0.3], seed=42)
print(f"Training size: {train.count()}, Test size: {test.count()}")

# ===============================
# Train model
# ===============================
print("Training Logistic Regression model...")
model = pipeline.fit(train)

# ===============================
# Predict
# ===============================
print("Making predictions on test set...")
predictions = model.transform(test)

# ===============================
# Evaluate
# ===============================
binary_eval = BinaryClassificationEvaluator(labelCol="class")
multi_eval = MulticlassClassificationEvaluator(labelCol="class")

metrics = {
    "accuracy":  multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"}),
    "precision": multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"}),
    "recall":    multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"}),
    "f1":        multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"}),
    "auc":       binary_eval.evaluate(predictions)
}

print("\nRESULTS")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# ===============================
# Save outputs to HDFS
# ===============================
print(f"\nSaving predictions and metrics to HDFS: {OUTPUT_PATH}")

# Save metrics as CSV
spark.createDataFrame(metrics.items(), ["metric", "value"]) \
    .write.mode("overwrite").csv(f"{OUTPUT_PATH}/metrics", header=True)

print("Save complete!")

spark.stop()
