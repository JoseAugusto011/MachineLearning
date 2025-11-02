// src/LinearRegression.cpp
#include "../include/LinearRegression.h"
#include <iostream>
#include <fstream>

template<typename T>
LinearRegression<T>::LinearRegression() {
    config = TrainingConfig<T>();
}

template<typename T>
LinearRegression<T>::LinearRegression(const TrainingConfig<T>& config) : config(config) {}

template<typename T>
void LinearRegression<T>::train(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y) {
    if (this->preprocessing_enabled) {
        Eigen::MatrixX<T> X_processed = X;
        Eigen::VectorX<T> y_processed = y;
        this->preprocessData(X_processed, y_processed);
        
        if (!solveDirect(X_processed, y_processed)) {
            if (config.verbose) {
                std::cout << "Direct solution failed, using SVD fallback..." << std::endl;
            }
            solveSVD(X_processed, y_processed);
        }
    } else {
        if (!solveDirect(X, y)) {
            if (config.verbose) {
                std::cout << "Direct solution failed, using SVD fallback..." << std::endl;
            }
            solveSVD(X, y);
        }
    }
    
    calculateMetrics(X, y);
    
    if (config.verbose) {
        std::cout << "Linear Regression training completed." << std::endl;
        std::cout << "R²: " << r_squared << ", MSE: " << mse << std::endl;
    }
}

template<typename T>
bool LinearRegression<T>::solveDirect(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y) {
    try {
        Eigen::MatrixX<T> XTX = X.transpose() * X;
        
        // Verifica se a matriz é invertível
        Eigen::FullPivLU<Eigen::MatrixX<T>> lu(XTX);
        if (!lu.isInvertible()) {
            return false;
        }
        
        Eigen::MatrixX<T> XTX_inv = XTX.inverse();
        weights = XTX_inv * X.transpose() * y;
        return true;
        
    } catch (const std::exception& e) {
        if (config.verbose) {
            std::cout << "Direct solution exception: " << e.what() << std::endl;
        }
        return false;
    }
}

template<typename T>
bool LinearRegression<T>::solveSVD(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y) {
    try {
        // Usa SVD para solução numericamente estável
        Eigen::JacobiSVD<Eigen::MatrixX<T>> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
        weights = svd.solve(y);
        return true;
        
    } catch (const std::exception& e) {
        if (config.verbose) {
            std::cout << "SVD solution exception: " << e.what() << std::endl;
        }
        // Fallback final: usa pseudo-inversa
        Eigen::MatrixX<T> X_pinv = (X.transpose() * X).completeOrthogonalDecomposition().pseudoInverse();
        weights = X_pinv * X.transpose() * y;
        return true;
    }
}

template<typename T>
Eigen::VectorX<T> LinearRegression<T>::predict(const Eigen::MatrixX<T>& X) const {
    return X * weights;
}

template<typename T>
void LinearRegression<T>::saveWeights(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    int size = weights.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(reinterpret_cast<const char*>(weights.data()), size * sizeof(T));
    file.close();
}

template<typename T>
void LinearRegression<T>::loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    int size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    weights.resize(size);
    file.read(reinterpret_cast<char*>(weights.data()), size * sizeof(T));
    file.close();
}

template<typename T>
void LinearRegression<T>::calculateMetrics(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y) {
    Eigen::VectorX<T> predictions = predict(X);
    Eigen::VectorX<T> residuals = y - predictions;
    
    // MSE
    mse = residuals.squaredNorm() / static_cast<T>(y.size());
    
    // R²
    T total_variance = (y.array() - y.mean()).square().sum();
    T explained_variance = total_variance - residuals.squaredNorm();
    r_squared = (total_variance > static_cast<T>(1e-10)) ? 
                explained_variance / total_variance : static_cast<T>(0);
}

// Instanciações explícitas
template class LinearRegression<float>;
template class LinearRegression<double>;