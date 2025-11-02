// include/LRClassifier.h
#ifndef LR_CLASSIFIER_H
#define LR_CLASSIFIER_H

#include "LinearRegression.h"
#include "Metrics.h"
#include <Eigen/Dense>

template<typename T>
class LRClassifier : public LinearRegression<T> {
public:
    LRClassifier();
    explicit LRClassifier(const TrainingConfig<T>& config);
    
    // Sobrescreve train para adicionar métricas de classificação
    void train(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y) override;
    
    // Sobrescreve predict para classificação binária
    Eigen::VectorX<T> predict(const Eigen::MatrixX<T>& X) const override;
    
    // Métodos específicos do classificador
    ClassificationMetrics<T> getClassificationMetrics() const { return classification_metrics; }
    
    // Função para obter linha de decisão (equivalente ao getRegressionY do Python)
    Eigen::VectorX<T> getDecisionBoundary(const Eigen::VectorX<T>& regressionX, T shift = 0) const;

private:
    ClassificationMetrics<T> classification_metrics;
    
    void calculateClassificationMetrics(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y);
};

#endif