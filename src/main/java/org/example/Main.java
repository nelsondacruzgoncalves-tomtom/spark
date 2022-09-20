package org.example;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.example.udf.NormEmbarked;
import org.example.udf.NormNullVal;
import org.example.udf.NormSex;

public class Main {

    public static void main(String[] args) {
        System.out.println("Hello world!");

        SparkSession spark = SparkSession.builder()
                .master("local[*]")
                .getOrCreate();
        // read data from csv
        Dataset<Row> df = spark.sqlContext()
                .read()
                .format("com.databricks.spark.csv")
                .option("header", true)
                .option("inferSchema", true)
                .load("data/train.csv");
        df.show();
        // normalize strings
        spark.sqlContext().udf().register("normSex", new NormSex(), DataTypes.IntegerType);
        spark.sqlContext().udf().register("normEmbarked", new NormEmbarked(), DataTypes.IntegerType);

        Dataset<Row> projection = df.select(
                col(ColNames.SURVIVED),
                col(ColNames.FARE),
                callUDF("normSex", col(ColNames.SEX)).alias(ColNames.SEX),
                col(ColNames.AGE),
                col(ColNames.P_CLASS),
                col(ColNames.P_ARCH),
                col(ColNames.SIB_SP),
                callUDF("normEmbarked", col(ColNames.EMBARKED)).alias(ColNames.EMBARKED)
        );
        projection.show();
        // handle nulls in data
        JavaRDD<Vector> statsDf = projection.rdd().toJavaRDD().map(row ->
                Vectors.dense(
                        row.getAs(ColNames.FARE),
                        row.isNullAt(3) ? 0 : row.getAs(ColNames.AGE)
                )
        );

        MultivariateStatisticalSummary summary = Statistics.colStats(statsDf.rdd());
        final double meanFare = summary.mean().apply(0);
        final double meanAge = summary.mean().apply(1);

        System.out.println("Mean Fare: " + meanFare);
        System.out.println("Mean Age: " + meanAge);

        spark.sqlContext().udf().register("normFare", new NormNullVal(meanFare), DataTypes.DoubleType);
        spark.sqlContext().udf().register("normAge", new NormNullVal(meanAge), DataTypes.DoubleType);

        Dataset<Row> finalDf = projection.select(
                col(ColNames.SURVIVED),
                callUDF("normFare", col(ColNames.FARE).cast("String")).alias(ColNames.FARE),
                col(ColNames.SEX),
                callUDF("normAge", col(ColNames.AGE).cast("String")).alias(ColNames.AGE),
                col(ColNames.P_CLASS),
                col(ColNames.P_ARCH),
                col(ColNames.SIB_SP),
                col(ColNames.EMBARKED)
        );
        finalDf.show();
        // scale
        final VectorScaler vectorScaler = new VectorScaler(summary);

        final Encoder<Vector> VectorEncoder = Encoders.kryo(Vector.class);
        // features + class
        JavaRDD<VectorPair> scaledRdd = finalDf.toJavaRDD().map(row -> {
            VectorPair vectorPair = new VectorPair();
            final Integer survived = row.<Integer>getAs(ColNames.SURVIVED);
            vectorPair.setLabel(Double.valueOf(survived));
            vectorPair.setFeatures(vectorScaler.getScaledVector(
                    row.getAs(ColNames.FARE),
                    row.getAs(ColNames.AGE),
                    row.<Integer>getAs(ColNames.P_CLASS),
                    row.<Integer>getAs(ColNames.SEX),
                    row.isNullAt(7) ? 0 : row.<Integer>getAs(ColNames.EMBARKED)
            ));
            return vectorPair;
        });

        Dataset<Row> scaledDf = spark.createDataFrame(scaledRdd, VectorPair.class);
        scaledDf.show();
        // Data is ready
        final Dataset<Row> scaledData2 = MLUtils.convertVectorColumnsToML(scaledDf);
        scaledData2.show();
        // split training and validation
        final Dataset<Row> data = scaledData2.toDF("features", "label");
        final Dataset<Row>[] datasets = data.randomSplit(new double[]{0.8, 0.2}, 12345L);
        final Dataset<Row> trainingData = datasets[0];
        final Dataset<Row> validationData = datasets[1];

        // train and classify
        final Trainer trainer = new Trainer();
        final MultilayerPerceptronClassificationModel model = trainer.fit(trainingData);
        final Dataset<Row> predictions = model.transform(validationData);
        predictions.show();
        // evaluate

        final Evaluator evaluator = new Evaluator();
        evaluator.evaluate(predictions);


        spark.sqlContext().udf().register("normSex", new NormSex(), DataTypes.IntegerType);
        spark.sqlContext().udf().register("normEmbarked", new NormEmbarked(), DataTypes.IntegerType);

        // load the Test data
        Dataset<Row> testData = spark.sqlContext()
                .read()
                .format("com.databricks.spark.csv")
                .option("header", "true") // Use first line of all files as header
                .option("inferSchema", "true") // Automatically infer data types
                .load("data/test.csv");
        final Dataset<Row> testDf = testData.select(
                col(ColNames.PASSENGER_ID),
                col(ColNames.FARE),
                callUDF("normSex", col(ColNames.SEX)).alias(ColNames.SEX),
                col(ColNames.AGE),
                col(ColNames.P_CLASS),
                col(ColNames.P_ARCH),
                col(ColNames.SIB_SP),
                callUDF("normEmbarked", col(ColNames.EMBARKED)).alias(ColNames.EMBARKED));
        // handle nulls
        Map<String, Object> m = new HashMap<>();
        m.put(ColNames.AGE, meanAge);
        m.put(ColNames.FARE, meanFare);
        final Dataset<Row> testDf2 = testDf.na().fill(m);
        // scale test data
        JavaRDD<VectorPair> testRdd = testDf2.javaRDD().map(row -> {
            VectorPair vectorPair = new VectorPair();
            vectorPair.setLabel(row.<Integer>getAs(ColNames.PASSENGER_ID));
            vectorPair.setFeatures(vectorScaler.getScaledVector(
                    row.getAs(ColNames.FARE),
                    row.getAs(ColNames.AGE),
                    row.<Integer>getAs(ColNames.P_CLASS),
                    row.<Integer>getAs(ColNames.SEX),
                    row.<Integer>getAs(ColNames.EMBARKED)
            ));
            return vectorPair;
        });
        final Dataset<Row> scaledTestDf = spark.createDataFrame(testRdd, VectorPair.class);
        final Dataset<Row> finalTestDf = MLUtils.convertVectorColumnsToML(scaledTestDf).toDF("features", "PassengerId");
        final Dataset<Row> resultDF = model.transform(finalTestDf).select(ColNames.PASSENGER_ID, "prediction");
        resultDF.show();


//        resultDF.write().csv("data/result.csv");
        resultDF.coalesce(1)
                .write().csv("results");
//                .format("com.databricks.spark.csv")
//                .option("header", true)
//                .save("data/results.csv");
    }

}