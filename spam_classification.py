# Import libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import happybase


# Start Spark and Connect to Hive
spark = SparkSession.builder \
    .appName("Spam Email Classification") \
    .enableHiveSupport() \
    .getOrCreate()

print("\n" + "=" * 80)
print("=" * 80)

# Load Data from Hive

data = spark.sql("SELECT * FROM spambase")
print(f"Loaded {data.count()} emails")

# Clean the Data
data = data.na.drop()
print(f"After cleaning: {data.count()} emails")

# Prepare Features for Machine Learning
# Get all column names except class
feature_cols = [col for col in data.columns if col != 'class']
print(f"Using {len(feature_cols)} features")

# Combine all features into one column called 'features_raw'
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
data_with_features = assembler.transform(data)

# Scale the features 
scaler = StandardScaler(inputCol="features_raw", outputCol="features")
scaler_model = scaler.fit(data_with_features)
scaled_data = scaler_model.transform(data_with_features)

# Keep only the columns we need
final_data = scaled_data.select("features", "class")

# Split Data into Training and Testing Sets

print("\nSplitting data into training (70%) and testing (30%)...")
training_data, testing_data = final_data.randomSplit([0.7, 0.3], seed=42)
print(f"Training set: {training_data.count()} emails")
print(f"Testing set: {testing_data.count()} emails")


# Train the Model
model = LogisticRegression(labelCol="class", featuresCol="features", maxIter=100)
trained_model = model.fit(training_data)

# Test the Model (Make Predictions)
print("\nTesting the model on new data...")
predictions = trained_model.transform(testing_data)


# Evaluate How Good the Model Is
print("\nCalculating accuracy scores...")

# Calculate different metrics
evaluator = MulticlassClassificationEvaluator(labelCol="class", predictionCol="prediction")
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

# Calculate AUC
auc_evaluator = BinaryClassificationEvaluator(labelCol="class")
auc = auc_evaluator.evaluate(predictions)

# Print the results
print("\n" + "=" * 80)
print("MODEL PERFORMANCE RESULTS")
print("=" * 80)
print(f"Accuracy:  {accuracy:.2%}  (How often the model is correct)")
print(f"Precision: {precision:.2%}  (How reliable spam predictions are)")
print(f"Recall:    {recall:.2%}  (How many spam emails we catch)")
print(f"F1 Score:  {f1:.2%}  (Balance of precision and recall)")
print(f"AUC Score: {auc:.2%}  (Overall model quality)")
print("=" * 80)

# Save Predictions to HDFS
print("\nSaving predictions to HDFS...")
output_location = "hdfs:///tmp/spam_predictions"
predictions.select("prediction", "class") \
    .write \
    .mode("overwrite") \
    .csv(output_location, header=True)
print(f"Saved to: {output_location}")


# Save Results to HBase Database/Prepare the metrics to save
metrics_to_save = [
    ('run1', 'cf:accuracy', str(accuracy)),
    ('run1', 'cf:precision', str(precision)),
    ('run1', 'cf:recall', str(recall)),
    ('run1', 'cf:f1_score', str(f1)),
    ('run1', 'cf:auc_score', str(auc)),
    ('run1', 'cf:model', 'LogisticRegression'),
    ('run1', 'cf:train_size', str(training_data.count())),
    ('run1', 'cf:test_size', str(testing_data.count())),
]

# Function to write to HBase
def save_to_hbase(partition):
    #Save metrics to HBase database
    conn = happybase.Connection('master')
    conn.open()
    table = conn.table('spam_metrics')
    
    for row_key, column, value in partition:
        table.put(row_key.encode(), {column.encode(): value.encode()})
    
    conn.close()

# Save the metrics
spark.sparkContext.parallelize(metrics_to_save).foreachPartition(save_to_hbase)
print("Metrics saved to HBase!")

# Finish
print("\n" + "=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
spark.stop()