package org.example;

import java.io.Serializable;

import org.apache.spark.mllib.linalg.Vector;

public class VectorPair implements Serializable {
    private double label;
    private Vector features;

    public VectorPair(double label, Vector features) {
        this.label = label;
        this.features = features;
    }

    public VectorPair() {
    }

    public void setFeatures(Vector features) {
        this.features = features;
    }

    public Vector getFeatures() {
        return this.features;
    }

    public void setLabel(double label) {
        this.label = label;
    }

    public double getLabel() {
        return this.label;
    }
}
