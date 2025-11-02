// src/PocketPLA.cpp
#include "../include/PocketPLA.h"
#include "../include/Metrics.h"
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

template<typename T>
PocketPLA<T>::PocketPLA() {
    config = TrainingConfig<T>();
}

template<typename T>
PocketPLA<T>::PocketPLA(const TrainingConfig<T>& config) : config(config) {}

template<typename T>
void PocketPLA<T>::train(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y) {
    if (this->preprocessing_enabled) {
        Eigen::MatrixX<T> X_processed = X;
        Eigen::VectorX<T> y_processed = y;
        this->preprocessData(X_processed, y_processed);
        initializeWeights(X_processed.cols());
        executeTraining(X_processed, y_processed);
    } else {
        initializeWeights(X.cols());
        executeTraining(X, y);
    }
}

template<typename T>
void PocketPLA<T>::executeTraining(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y) {
    Eigen::VectorX<T> best_weights = weights;
    T best_error = calculateError(X, y);
    std::vector<T> error_history;
    
    bool converged = false;
    
    for (iterations = 0; iterations < config.max_iterations; ++iterations) {
        // Cálculo vetorizado - encontra pontos mal classificados
        Eigen::VectorX<T> predictions = (X * weights).array().sign();
        Eigen::Array<bool, Eigen::Dynamic, 1> misclassified = predictions.array() != y.array();
        
        // Early stopping check
        T current_error = misclassified.cast<T>().sum() / static_cast<T>(y.size());
        error_history.push_back(current_error);
        
        if (current_error < best_error) {
            best_error = current_error;
            best_weights = weights;
        }
        
        if (current_error <= config.tolerance) {
            converged = true;
            break;
        }
        
        // Encontra primeiro ponto mal classificado
        int misclassified_index = -1;
        for (int i = 0; i < misclassified.size(); ++i) {
            if (misclassified[i]) {
                misclassified_index = i;
                break;
            }
        }
        
        if (misclassified_index == -1) {
            converged = true;
            break;
        }
        
        // Atualiza pesos
        weights += y(misclassified_index) * X.row(misclassified_index).transpose();
        
        // Atualiza pocket periodicamente
        if (iterations % config.pocket_update_frequency == 0) {
            T error = calculateError(X, y);
            if (error < best_error) {
                best_error = error;
                best_weights = weights;
            }
        }
    }
    
    weights = best_weights;
    final_error = best_error;
    
    // Calcula métricas finais
    Eigen::VectorX<T> final_predictions = predict(X);
    training_metrics = Metrics<T>::calculateClassificationMetrics(y, final_predictions);
    training_metrics.training_history = error_history;
    
    if (config.verbose) {
        std::cout << "Training completed: " << iterations << " iterations, "
                  << "final error: " << final_error << ", "
                  << "converged: " << (converged ? "yes" : "no") << std::endl;
    }
}

template<typename T>
Eigen::VectorX<T> PocketPLA<T>::predict(const Eigen::MatrixX<T>& X) const {
    return (X * weights).array().sign();
}

template<typename T>
void PocketPLA<T>::saveWeights(const std::string& filename) const {
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
void PocketPLA<T>::loadWeights(const std::string& filename) {
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
T PocketPLA<T>::calculateError(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y) const {
    Eigen::VectorX<T> predictions = predict(X);
    // CORREÇÃO: usar template keyword para dependent names
    return (predictions.array() != y.array()).template cast<T>().sum() / static_cast<T>(y.size());
}

template<typename T>
void PocketPLA<T>::initializeWeights(int num_features) {
    weights = Eigen::VectorX<T>::Zero(num_features);
}

// Instanciações explícitas
template class PocketPLA<float>;
template class PocketPLA<double>;