from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import happybase

# Step 1: Create Spark session
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
print(f"Total records: {spam_df.count()}")

# Step 3: Handle null values
print("\n[2] Cleaning data...")
spam_df = spam_df.na.drop()
print(f"Final dataset size: {spam_df.count()} records")

# Step 4: Prepare features
print("\n[3] Preparing features...")
feature_columns = [col for col in spam_df.columns if col != 'class']
print(f"Using {len(feature_columns)} features")

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
print(f"Training: {train_data.count()}, Test: {test_data.count()}")

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

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")
print("=" * 80)

# Step 10: Write to HBase
print("\n[9] Writing metrics to HBase...")
data = [
    ('run1', 'cf:accuracy', str(accuracy)),
    ('run1', 'cf:precision', str(precision)),
    ('run1', 'cf:recall', str(recall)),
    ('run1', 'cf:f1_score', str(f1)),
    ('run1', 'cf:auc_roc', str(auc)),
    ('run1', 'cf:model_type', 'LogisticRegression'),
    ('run1', 'cf:train_size', str(train_data.count())),
    ('run1', 'cf:test_size', str(test_data.count())),
]

def write_to_hbase_partition(partition):
    connection = happybase.Connection('master')
    connection.open()
    table = connection.table('spam_metrics')
    for row in partition:
        row_key, column, value = row
        table.put(row_key.encode(), {column.encode(): value.encode()})
    connection.close()

rdd = spark.sparkContext.parallelize(data)
rdd.foreachPartition(write_to_hbase_partition)

print("Complete!")
spark.stop()