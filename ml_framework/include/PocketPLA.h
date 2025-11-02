// include/PocketPLA.h
#ifndef POCKET_PLA_H
#define POCKET_PLA_H

#include "Model.h"
#include "TrainingConfig.h"
#include "Metrics.h"
#include <Eigen/Dense>
#include <vector>
  

template<typename T>
class PocketPLA : public Model<T> {
public:
    PocketPLA();
    explicit PocketPLA(const TrainingConfig<T>& config);
    
    void train(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y) override;
    Eigen::VectorX<T> predict(const Eigen::MatrixX<T>& X) const override;
    void saveWeights(const std::string& filename) const override;
    void loadWeights(const std::string& filename) override;
    Eigen::VectorX<T> getWeights() const override { return weights; }
    void setWeights(const Eigen::VectorX<T>& new_weights) override { weights = new_weights; }
    
    ClassificationMetrics<T> getTrainingMetrics() const { return training_metrics; }
    void setConfig(const TrainingConfig<T>& new_config) { config = new_config; }
    
    int getIterations() const { return iterations; }
    T getFinalError() const { return final_error; }

private:
    Eigen::VectorX<T> weights;
    TrainingConfig<T> config;
    ClassificationMetrics<T> training_metrics;
    int iterations = 0;
    T final_error = 0;
    
    T calculateError(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y) const;
    void initializeWeights(int num_features);
};

#endif