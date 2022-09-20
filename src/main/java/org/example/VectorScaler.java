package org.example;

import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import scala.Tuple2;
import scala.Tuple3;

public class VectorScaler {

    private final StandardScalerModel scaler;

    public VectorScaler(MultivariateStatisticalSummary summary) {
        Vector stddev = Vectors.dense(
                Math.sqrt(summary.variance().apply(0)),
                Math.sqrt(summary.variance().apply(1))
        );
        Vector mean = Vectors.dense(
                summary.mean().apply(0),
                summary.mean().apply(1)
        );

        scaler = new StandardScalerModel(stddev, mean);
    }

    public Vector getScaledVector(double fare, double age, double pclass, double sex, double embarked) {
        Vector scaledContinous = scaler.transform(Vectors.dense(fare, age));
        Tuple3<Double, Double, Double> pclassFlat = flattenPclass(pclass);
        Tuple3<Double, Double, Double> embarkedFlat = flattenEmbarked(embarked);
        Tuple2<Double, Double> sexFlat = flattenSex(sex);

        return Vectors.dense(
                scaledContinous.apply(0),
                scaledContinous.apply(1),
                sexFlat._1(),
                sexFlat._2(),
                pclassFlat._1(),
                pclassFlat._2(),
                pclassFlat._3(),
                embarkedFlat._1(),
                embarkedFlat._2(),
                embarkedFlat._3());
    }

    private Tuple3<Double, Double, Double> flattenPclass(double value) {
        Tuple3<Double, Double, Double> result;

        if (value == 1)
            result = new Tuple3<>(1d, 0d, 0d);
        else if (value == 2)
            result = new Tuple3<>(0d, 1d, 0d);
        else
            result =new Tuple3<>(0d, 0d, 1d);

        return result;
    }

    private Tuple3<Double, Double, Double> flattenEmbarked(double value) {
        Tuple3<Double, Double, Double> result;

        if (value == 0)
            result = new Tuple3<>(1d, 0d, 0d);
        else if (value == 1)
            result = new Tuple3<>(0d, 1d, 0d);
        else
            result = new Tuple3<>(0d, 0d, 1d);

        return result;
    }

    private Tuple2<Double, Double> flattenSex(double value) {
        Tuple2<Double, Double> result;

        if (value == 0)
            result = new Tuple2<>(1d, 0d);
        else
            result = new Tuple2<>(0d, 1d);

        return result;
    }

}
