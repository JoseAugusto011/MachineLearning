// include/Metrics.h
#ifndef METRICS_H
#define METRICS_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <stdexcept>

template<typename T>
struct ClassificationMetrics {
    T accuracy;
    T precision;
    T recall;
    T f1_score;
    std::vector<T> training_history;
};

template<typename T>
class Metrics {
public:
    static ClassificationMetrics<T> calculateClassificationMetrics(
        const Eigen::VectorX<T>& y_true, 
        const Eigen::VectorX<T>& y_pred);
    
    static T calculateAccuracy(const Eigen::VectorX<T>& y_true, const Eigen::VectorX<T>& y_pred);
};

#endif