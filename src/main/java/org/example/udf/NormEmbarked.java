package org.example.udf;

import org.apache.spark.sql.api.java.UDF1;

import scala.Option;
import scala.Some;

public class NormEmbarked implements UDF1<String, Option<Integer>> {
    @Override
    public Option<Integer> call(final String value) throws Exception {
        if (value == null) {
            return Option.apply(null);
        } else {
            switch (value) {
                case "S":
                    return Some.apply(0);
                case "C":
                    return Some.apply(1);
                case "Q":
                    return Some.apply(2);
                default:
                    throw new RuntimeException();
            }
        }
    }
}
