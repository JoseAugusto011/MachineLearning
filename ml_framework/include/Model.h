// include/Model.h
#ifndef MODEL_H
#define MODEL_H

#include <Eigen/Dense>
#include <string>
#include <vector>

template<typename T>
class Model {
public:
    virtual ~Model() = default;
    
    virtual void train(const Eigen::MatrixX<T>& X, const Eigen::VectorX<T>& y) = 0;
    virtual Eigen::VectorX<T> predict(const Eigen::MatrixX<T>& X) const = 0;
    virtual void saveWeights(const std::string& filename) const = 0;
    virtual void loadWeights(const std::string& filename) = 0;
    virtual Eigen::VectorX<T> getWeights() const = 0;
    virtual void setWeights(const Eigen::VectorX<T>& weights) = 0;
    
    void setPreprocessing(bool enable) { preprocessing_enabled = enable; }
    bool getPreprocessing() const { return preprocessing_enabled; }

protected:
    bool preprocessing_enabled = false;
    virtual void preprocessData(Eigen::MatrixX<T>& X, Eigen::VectorX<T>& y) {
        // Default: nenhum pré-processamento - remover parâmetros não usados
        (void)X; // Suprime warning
        (void)y; // Suprime warning
    }
};

#endif