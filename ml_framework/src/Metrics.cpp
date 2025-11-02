// src/Metrics.cpp
#include "../include/Metrics.h"
#include <cmath>

template<typename T>
ClassificationMetrics<T> Metrics<T>::calculateClassificationMetrics(
    const Eigen::VectorX<T>& y_true, const Eigen::VectorX<T>& y_pred) {
    
    ClassificationMetrics<T> metrics;
    
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("y_true and y_pred must have same size");
    }
    
    int true_positive = 0, false_positive = 0, false_negative = 0, true_negative = 0;
    
    for (int i = 0; i < y_true.size(); ++i) {
        if (y_true(i) == 1 && y_pred(i) == 1) true_positive++;
        else if (y_true(i) == -1 && y_pred(i) == 1) false_positive++;
        else if (y_true(i) == 1 && y_pred(i) == -1) false_negative++;
        else if (y_true(i) == -1 && y_pred(i) == -1) true_negative++;
    }
    
    metrics.accuracy = static_cast<T>(true_positive + true_negative) / static_cast<T>(y_true.size());
    
    T precision_denom = static_cast<T>(true_positive + false_positive);
    metrics.precision = precision_denom > 0 ? static_cast<T>(true_positive) / precision_denom : 0;
    
    T recall_denom = static_cast<T>(true_positive + false_negative);
    metrics.recall = recall_denom > 0 ? static_cast<T>(true_positive) / recall_denom : 0;
    
    T f1_denom = metrics.precision + metrics.recall;
    metrics.f1_score = f1_denom > 0 ? 2 * metrics.precision * metrics.recall / f1_denom : 0;
    
    return metrics;
}

template<typename T>
T Metrics<T>::calculateAccuracy(const Eigen::VectorX<T>& y_true, const Eigen::VectorX<T>& y_pred) {
    return (y_true.array() == y_pred.array()).cast<T>().mean();
}

// Instanciações explícitas
template struct ClassificationMetrics<float>;
template struct ClassificationMetrics<double>;
template class Metrics<float>;
template class Metrics<double>;