
import org.apache.spark.sql.types.{FloatType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object SparkAssignmnet_2 extends App {



  //Schema for training dataset
  val Schemetrain = (new StructType)
    .add("PassengerId", IntegerType)
    .add("Survived", IntegerType)
    .add("Pclass", IntegerType)
    .add("Name", StringType)
    .add("Sex", StringType)
    .add("Age", FloatType)
    .add("SibSp", IntegerType)
    .add("Parch", IntegerType)
    .add("Ticket", StringType)
    .add("Fare", FloatType)
    .add("Cabin", StringType)
    .add("Embarked", StringType)

  //Schema for testing dataset
  val Schemetest = (new StructType)
    .add("PassengerId", IntegerType)
    .add("Pclass", IntegerType)
    .add("Name", StringType)
    .add("Sex", StringType)
    .add("Age", FloatType)
    .add("SibSp", IntegerType)
    .add("Parch", IntegerType)
    .add("Ticket", StringType)
    .add("Fare", FloatType)
    .add("Cabin", StringType)
    .add("Embarked", StringType)



  // traning and testing schema
  val trainSchema = StructType(Schemetrain)
  val testSchema = StructType(Schemetest)

  // Implementing spark session and setting log level to error

  implicit val spark: SparkSession = SparkSession
    .builder()
    .appName("Spark Assignment 2")
    .master("local[*]")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  //Reading training and testing data

  val df_train = spark.read.option("header", "true").schema(trainSchema)
    .csv("/Users/sameekshagurrala/Desktop/SparkAssignment_2/src/main/resources/train.csv")
  val df_test = spark.read.option("header", "true")
    .schema(testSchema).csv("/Users/sameekshagurrala/Desktop/SparkAssignment_2/src/main/resources/test.csv")
  //Creating table views for training and testing
  df_train.createOrReplaceTempView("df_train")
  df_test.createOrReplaceTempView("df_test")

  // ---------------------- EAD ----------------------------
  //Describing training data to show Exploratory Data Analysis (EDA) statistics
  df_train.describe("Age","SibSp","Parch","Fare").show()


  //Describing testing data to show Exploratory Data Analysis (EDA) statistics
  spark.sql("select Survived,count(*) from df_train group by Survived").show()
  spark.sql("select Sex, Survived, count(*) from df_train group by Sex,Survived").show()
  spark.sql("select Pclass, Survived, count(*) from df_train group by Pclass,Survived").show()


  // ---------------------- Feature Engineering ----------------------------
    // filling null values with avg values for training dataset

  //filling null values with avg values for training dataset for age
  val AvgAge = df_train.select("Age")
    .agg(avg("Age"))
    .collect() match {
    case Array(Row(avg: Double)) => avg
    case _ => 0
  }

  //filling null values with avg values for test dataset for age
  val AvgAge_test = df_test.select("Age")
    .agg(avg("Age"))
    .collect() match {
    case Array(Row(avg: Double)) => avg
    case _ => 0
  }

  //filling null values with avg values for training dataset for fare
  val AvgFare = df_train.select("Fare")
    .agg(avg("Fare"))
    .collect() match {
    case Array(Row(avg: Double)) => avg
    case _ => 0
  }


  //filling null values with avg values for test dataset for fare
  val AvgFare_test = df_test.select("Fare")
    .agg(avg("Fare"))
    .collect() match {
    case Array(Row(avg: Double)) => avg
    case _ => 0
  }

  //UDF for Embarked column for training dataset
  val embarked: (String => String) = {
    case "" => "S"
    case null => "S"
    case a => a
  }
  val embarkedUDF = udf(embarked)

  //UDF for Embarked column for test dataset
  val embarked_test: (String => String) = {
    case "" => "S"
    case null => "S"
    case a => a
  }
  val embarkedUDF_test = udf(embarked_test)



  //Filling null values with avg values for training dataset
  val imputeddf = df_train.na.fill(Map("Fare" -> AvgFare, "Age" -> AvgAge))
  val imputeddf2 = imputeddf.withColumn("Embarked", embarkedUDF(imputeddf.col("Embarked")))
  //Splitting training data into training and validation
  val Array(trainingData, validationData) = imputeddf2.randomSplit(Array(0.7, 0.3))

  //Filling null values with avg values for test dataset
  val imputeddf_test = df_test.na.fill(Map("Fare" -> AvgFare_test, "Age" -> AvgAge_test))
  val imputeddf2_test = imputeddf_test.withColumn("Embarked", embarkedUDF_test(imputeddf_test.col("Embarked")))

  //Dropping Cabin feature as it has so many null values
  val df1_train = trainingData.drop("Cabin")
  val df1_test = imputeddf2_test.drop("Cabin")

  //Print schema for train and test
  df1_train.printSchema()
  df1_test.printSchema()

  //Indexing categorical features
  val catFeatColNames = Seq("Pclass", "Sex", "Embarked")
  val stringIndexers = catFeatColNames.map { colName =>
    new StringIndexer()
      .setInputCol(colName)
      .setOutputCol(colName + "Indexed")
      .fit(trainingData)
  }

  //Indexing target feature
  val labelIndexer = new StringIndexer()
    .setInputCol("Survived")
    .setOutputCol("SurvivedIndexed")
    .fit(trainingData)

  //Assembling features into one vector
  val numFeatColNames = Seq("Age", "SibSp", "Parch", "Fare")
  val idxdCatFeatColName = catFeatColNames.map(_ + "Indexed")
  val allIdxdFeatColNames = numFeatColNames ++ idxdCatFeatColName
  val assembler = new VectorAssembler()
    .setInputCols(Array(allIdxdFeatColNames: _*))
    .setOutputCol("Features")

  //Randomforest classifier
  val randomforest = new RandomForestClassifier()
    .setLabelCol("SurvivedIndexed")
    .setFeaturesCol("Features")

  //Retrieving original labels
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)

  //Creating pipeline
  val pipeline = new Pipeline().setStages(
    (stringIndexers :+ labelIndexer :+ assembler :+ randomforest :+ labelConverter).toArray)

  //Selecting best model
  val paramGrid = new ParamGridBuilder()
    .addGrid(randomforest.maxBins, Array(25, 28, 31))
    .addGrid(randomforest.maxDepth, Array(4, 6, 8))
    .addGrid(randomforest.impurity, Array("entropy", "gini"))
    .build()

  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("SurvivedIndexed")
    .setMetricName("areaUnderPR")

  //Cross validator with 10 fold
  val cv = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(10)

  //Fitting model using cross validation
  val crossValidatorModel = cv.fit(trainingData)

  //predictions on validation data
  val predictions = crossValidatorModel.transform(validationData)

  //Accuracy
  val accuracy = evaluator.evaluate(predictions)
  println("Used Random Forest classifier with Cross Validation");
  println("10 fold cross validation to predict survival of passengers on Titanic the following is the accurecy of the model")
  println("Test Accuracy DT= " + accuracy)
  println("Test Error DT= " + (1.0 - accuracy))

  //predicting on test data

  println("Predicting survival of passengers on Titanic using Random Forest classifier with Cross Validation")
  val predictions1 = crossValidatorModel.transform(df1_test)
  predictions1.select("PassengerId", "predictedLabel").show(100)




}