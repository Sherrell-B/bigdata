from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import happybase

# Step 1: Create Spark session
# enableHiveSupport() is kept since you are reading from a Hive table
spark = SparkSession.builder \
    .appName("Spam Email Classification") \
    .enableHiveSupport() \
    .getOrCreate()

print("=" * 80)
print("SPAM EMAIL CLASSIFICATION WITH SPARK MLLIB")
print("=" * 80)

# Step 2: Load data from Hive
print("\n[1] Loading data from Hive table 'spambase'...")
spam_df = spark.sql("SELECT * FROM spambase")

# Step 3: Handle null values
print("\n[2] Cleaning data...")
spam_df = spam_df.na.drop()

# Step 4: Prepare features
print("\n[3] Preparing features...")
feature_columns = [col for col in spam_df.columns if col != 'class']
assembler = VectorAssembler(
    inputCols=feature_columns, 
    outputCol="features_raw", 
    handleInvalid="skip"
)
assembled_df = assembler.transform(spam_df)

# Step 5: Scale features
print("\n[4] Scaling features...")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df).select("features", "class")

# Step 6: Split data
print("\n[5] Splitting data (70/30)...")
train_data, test_data = scaled_df.randomSplit([0.7, 0.3], seed=42)

# Step 7: Train model
print("\n[6] Training Logistic Regression...")
lr = LogisticRegression(labelCol="class", featuresCol="features", maxIter=100, regParam=0.01)
lr_model = lr.fit(train_data)

# Step 8: Predictions
print("\n[7] Making predictions...")
predictions = lr_model.transform(test_data)

# Step 9: Evaluate
print("\n[8] Evaluating model...")
binary_eval = BinaryClassificationEvaluator(labelCol="class")
auc = binary_eval.evaluate(predictions)

mc_eval = MulticlassClassificationEvaluator(labelCol="class", predictionCol="prediction")
accuracy = mc_eval.evaluate(predictions, {mc_eval.metricName: "accuracy"})
precision = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedPrecision"})
recall = mc_eval.evaluate(predictions, {mc_eval.metricName: "weightedRecall"})
f1 = mc_eval.evaluate(predictions, {mc_eval.metricName: "f1"})

# Print Results to Terminal
print("\n" + "=" * 80)
print("FINAL MODEL METRICS")
print("=" * 80)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")
print("=" * 80)

# Step 10: Write Metrics to HBase (Directly from Driver)
print("\n[9] Writing metrics to HBase table 'spam_metrics'...")
try:
    # Connect to HBase master
    connection = happybase.Connection('master') 
    connection.open()
    table = connection.table('spam_metrics')
    
    # Using a timestamped or unique run ID is better for portfolios
    run_id = "run_" + str(int(spark.sparkContext.startTime / 1000))
    
    metrics_data = {
        b'cf:accuracy': str(accuracy).encode(),
        b'cf:precision': str(precision).encode(),
        b'cf:f1_score': str(f1).encode(),
        b'cf:auc_roc': str(auc).encode(),
        b'cf:model': b'LogisticRegression'
    }
    table.put(run_id.encode(), metrics_data)
    connection.close()
    print(f"Successfully wrote metrics to HBase with RowKey: {run_id}")
except Exception as e:
    print(f"HBase Write Failed: {e}")

# Step 11: Save Predictions to HDFS
# We convert to RDD to use saveAsTextFile as requested, or keep as DF for Parquet/CSV
output_path = "hdfs:///tmp/spam_predictions_output"
print(f"\n[10] Saving raw predictions to HDFS: {output_path}")


# Save as text (this will save the Row objects as strings)
predictions.rdd.saveAsTextFile(output_path)

print("\nProcess Complete. Results saved to Terminal, HBase, and HDFS.")

spark.stop()