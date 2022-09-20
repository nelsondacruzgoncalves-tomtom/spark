package org.example.udf;

import org.apache.spark.sql.api.java.UDF1;

import scala.Option;
import scala.Some;

public class NormNullVal implements UDF1<String, Option<Double>> {

    private final Double defaultValue;

    public NormNullVal(final Double defaultValue) {
        this.defaultValue = defaultValue;
    }

    @Override
    public Option<Double> call(final String value) throws Exception {
        if (value == null) {
            return Some.apply(defaultValue);
        } else {
            return Some.apply(Double.parseDouble(value));
        }
    }
}
