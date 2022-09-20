package org.example;

import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class Trainer {


    private final MultilayerPerceptronClassifier mlp;

    public Trainer() {
        int[] layers = new int[] {10, 8, 16, 2};
        this.mlp = new MultilayerPerceptronClassifier()
                .setLayers(layers)
                .setBlockSize(128)
                .setSeed(1234L)
                .setTol(1e-8)
                .setMaxIter(1000);
    }

    public MultilayerPerceptronClassificationModel fit(Dataset<Row> trainingData) {
        return mlp.fit(trainingData);
    }

}
