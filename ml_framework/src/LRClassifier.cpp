// src/LRClassifier.cpp
#include "../include/LRClassifier.h"
#include <iostream>

template<typename T>
LRClassifier<T>::LRClassifier() : LinearRegression<T>() {}

template<typename T>
LRClassifier<T>::LRClassifier(const TrainingConfig<T>& config) : LinearRegression<T>(config) {}

template<typename T>
void LRClassifier<T>::train(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y) {
    // Chama train da classe base (LinearRegression)
    LinearRegression<T>::train(X, y);
    
    // Calcula métricas de classificação adicionais
    calculateClassificationMetrics(X, y);
    
    if (this->config.verbose) {
        std::cout << "Classification Accuracy: " << classification_metrics.accuracy << std::endl;
    }
}

template<typename T>
Eigen::VectorX<T> LRClassifier<T>::predict(const Eigen::MatrixX<T>& X) const {
    // Usa a predição da regressão linear e aplica função sign para classificação
    Eigen::VectorX<T> regression_predictions = LinearRegression<T>::predict(X);
    return regression_predictions.array().sign();
}

template<typename T>
Eigen::VectorX<T> LRClassifier<T>::getDecisionBoundary(const Eigen::VectorX<T>& regressionX, T shift) const {
    // Equivalente ao getRegressionY do Python: (-w[0]+shift - w[1]*regressionX) / w[2]
    Eigen::VectorX<T> weights = this->getWeights();
    
    if (weights.size() < 3) {
        throw std::runtime_error("Weights vector must have at least 3 elements for 2D decision boundary");
    }
    
    T w0 = weights(0);
    T w1 = weights(1);
    T w2 = weights(2);
    
    return (-w0 + shift - w1 * regressionX.array()) / w2;
}

template<typename T>
void LRClassifier<T>::calculateClassificationMetrics(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y) {
    Eigen::VectorX<T> predictions = predict(X);
    classification_metrics = Metrics<T>::calculateClassificationMetrics(y, predictions);
}

// Instanciações explícitas
template class LRClassifier<float>;
template class LRClassifier<double>;