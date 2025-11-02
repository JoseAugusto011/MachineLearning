// include/LinearRegression.h
#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "Model.h"
#include "TrainingConfig.h"
#include "Metrics.h"
#include <Eigen/Dense>
#include <vector>

template<typename T>
class LinearRegression : public Model<T> {
public:
    LinearRegression();
    explicit LinearRegression(const TrainingConfig<T>& config);
    
    void train(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y) override;
    Eigen::VectorX<T> predict(const Eigen::MatrixX<T>& X) const override;
    void saveWeights(const std::string& filename) const override;
    void loadWeights(const std::string& filename) override;
    Eigen::VectorX<T> getWeights() const override { return weights; }
    void setWeights(const Eigen::VectorX<T>& new_weights) override { weights = new_weights; }
    
    T getRSquared() const { return r_squared; }
    T getMSE() const { return mse; }

protected:
    Eigen::VectorX<T> weights;
    TrainingConfig<T> config;
    T r_squared = 0;
    T mse = 0;
    
    bool solveDirect(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y);
    bool solveSVD(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y);
    void calculateMetrics(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y);
};

#endif