package org.example;

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class Evaluator {

    private final MulticlassClassificationEvaluator evaluator1;
    private final MulticlassClassificationEvaluator evaluator2;
    private final MulticlassClassificationEvaluator evaluator3;
    private final MulticlassClassificationEvaluator evaluator4;

    public Evaluator() {
        final MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction");
        evaluator1 = evaluator.setMetricName("accuracy");
        evaluator2 = evaluator.setMetricName("weightedPrecision");
        evaluator3 = evaluator.setMetricName("weightedRecall");
        evaluator4 = evaluator.setMetricName("f1");
    }

    public void evaluate(final Dataset<Row> predictions) {
        final double accuracy = evaluator1.evaluate(predictions);
        final double precision = evaluator2.evaluate(predictions);
        final double recall = evaluator3.evaluate(predictions);
        final double f1 = evaluator4.evaluate(predictions);
        System.out.println("Accuracy = " + accuracy);
        System.out.println("Precision = " + precision);
        System.out.println("Recall = " + recall);
        System.out.println("F1 = " + f1);
        System.out.println("Test Error = " + (1 - accuracy));
    }
}
