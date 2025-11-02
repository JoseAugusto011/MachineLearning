// main.cpp
#include <iostream>
#include <Eigen/Dense>
#include "include/PocketPLA.h"
#include "include/Metrics.h"

using namespace std;

// Função para gerar dados de exemplo linearmente separáveis
void generateLinearData(Eigen::MatrixXd& X, Eigen::VectorXd& y, int samples = 100) {
    X = Eigen::MatrixXd::Random(samples, 2);
    y = Eigen::VectorXd(samples);
    
    // Linha de separação: 2*x1 + x2 - 1 = 0
    for (int i = 0; i < samples; ++i) {
        if (2*X(i,0) + X(i,1) - 1 > 0) {
            y(i) = 1;
        } else {
            y(i) = -1;
        }
    }
    
    // Adiciona bias column
    Eigen::MatrixXd X_with_bias = Eigen::MatrixXd::Ones(samples, 3);
    X_with_bias.block(0, 1, samples, 2) = X;
    X = X_with_bias;
}

// Função para gerar dados XOR (não linearmente separáveis)
void generateXORData(Eigen::MatrixXd& X, Eigen::VectorXd& y, int samples = 100) {
    X = Eigen::MatrixXd::Random(samples, 2);
    y = Eigen::VectorXd(samples);
    
    for (int i = 0; i < samples; ++i) {
        if ((X(i,0) > 0 && X(i,1) > 0) || (X(i,0) <= 0 && X(i,1) <= 0)) {
            y(i) = 1;
        } else {
            y(i) = -1;
        }
    }
    
    // Adiciona bias column
    Eigen::MatrixXd X_with_bias = Eigen::MatrixXd::Ones(samples, 3);
    X_with_bias.block(0, 1, samples, 2) = X;
    X = X_with_bias;
}

void testLinearSeparation() {
    cout << "=== TESTE 1: Dados Linearmente Separáveis ===" << endl;
    
    Eigen::MatrixXd X_train, X_test;
    Eigen::VectorXd y_train, y_test;
    
    generateLinearData(X_train, y_train, 200);
    generateLinearData(X_test, y_test, 50);
    
    // Configuração do treinamento
    TrainingConfig<double> config;
    config.max_iterations = 1000;
    config.tolerance = 1e-4;
    config.verbose = true;
    config.pocket_update_frequency = 5;
    
    // Cria e treina o modelo
    PocketPLA<double> model(config);
    model.setPreprocessing(false);
    
    cout << "Iniciando treinamento..." << endl;
    model.train(X_train, y_train);
    
    // Predições e métricas
    Eigen::VectorXd y_pred = model.predict(X_test);
    auto metrics = model.getTrainingMetrics();
    
    cout << "\n=== RESULTADOS ===" << endl;
    cout << "Iterações: " << model.getIterations() << endl;
    cout << "Erro final: " << model.getFinalError() << endl;
    cout << "Acurácia: " << metrics.accuracy << endl;
    cout << "Precisão: " << metrics.precision << endl;
    cout << "Recall: " << metrics.recall << endl;
    cout << "F1-Score: " << metrics.f1_score << endl;
    
    // Mostra alguns pesos
    Eigen::VectorXd weights = model.getWeights();
    cout << "\nPesos finais: [";
    for (int i = 0; i < weights.size(); ++i) {
        cout << weights(i);
        if (i < weights.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
    
    // Testa salvamento/carregamento
    model.saveWeights("pla_weights.bin");
    cout << "Pesos salvos em 'pla_weights.bin'" << endl;
}

void testNonLinearSeparation() {
    cout << "\n\n=== TESTE 2: Dados NÃO Linearmente Separáveis (XOR) ===" << endl;
    
    Eigen::MatrixXd X_train, X_test;
    Eigen::VectorXd y_train, y_test;
    
    generateXORData(X_train, y_train, 200);
    generateXORData(X_test, y_test, 50);
    
    TrainingConfig<double> config;
    config.max_iterations = 1000;
    config.tolerance = 1e-4;
    config.verbose = true;
    
    PocketPLA<double> model(config);
    
    cout << "Iniciando treinamento (deve ter performance ruim)..." << endl;
    model.train(X_train, y_train);
    
    Eigen::VectorXd y_pred = model.predict(X_test);
    auto metrics = model.getTrainingMetrics();
    
    cout << "\n=== RESULTADOS XOR ===" << endl;
    cout << "Iterações: " << model.getIterations() << endl;
    cout << "Erro final: " << model.getFinalError() << endl;
    cout << "Acurácia: " << metrics.accuracy << endl;
    cout << "Precisão: " << metrics.precision << endl;
    cout << "Recall: " << metrics.recall << endl;
    cout << "F1-Score: " << metrics.f1_score << endl;
}

void testWeightPersistence() {
    cout << "\n\n=== TESTE 3: Persistência de Pesos ===" << endl;
    
    Eigen::MatrixXd X, y;
    generateLinearData(X, y, 100);
    
    PocketPLA<double> model1;
    model1.train(X, y);
    
    Eigen::VectorXd original_weights = model1.getWeights();
    cout << "Pesos originais: [" << original_weights.transpose() << "]" << endl;
    
    // Salva e carrega em novo modelo
    model1.saveWeights("test_weights.bin");
    
    PocketPLA<double> model2;
    model2.loadWeights("test_weights.bin");
    
    Eigen::VectorXd loaded_weights = model2.getWeights();
    cout << "Pesos carregados: [" << loaded_weights.transpose() << "]" << endl;
    
    // Verifica se são iguais
    double difference = (original_weights - loaded_weights).norm();
    cout << "Diferença entre pesos: " << difference << " (deve ser ~0)" << endl;
    
    // Testa predições
    Eigen::VectorXd pred1 = model1.predict(X);
    Eigen::VectorXd pred2 = model2.predict(X);
    double pred_difference = (pred1 - pred2).norm();
    cout << "Diferença entre predições: " << pred_difference << " (deve ser ~0)" << endl;
}

void testDifferentConfigurations() {
    cout << "\n\n=== TESTE 4: Diferentes Configurações ===" << endl;
    
    Eigen::MatrixXd X, y;
    generateLinearData(X, y, 150);
    
    // Testa com poucas iterações
    {
        TrainingConfig<double> config;
        config.max_iterations = 50;
        config.verbose = true;
        
        PocketPLA<double> model(config);
        model.train(X, y);
        cout << "Config 50 iterações - Iterações usadas: " << model.getIterations() << endl;
    }
    
    // Testa com tolerância mais restritiva
    {
        TrainingConfig<double> config;
        config.max_iterations = 1000;
        config.tolerance = 1e-6;
        config.verbose = true;
        
        PocketPLA<double> model(config);
        model.train(X, y);
        cout << "Config tolerância 1e-6 - Iterações usadas: " << model.getIterations() << endl;
    }
}

int main() {
    try {
        cout << "Framework de Machine Learning - Teste PocketPLA" << endl;
        cout << "==============================================" << endl;
        
        testLinearSeparation();
        testNonLinearSeparation();
        testWeightPersistence();
        testDifferentConfigurations();
        
        cout << "\n\nTodos os testes completados!" << endl;
        
    } catch (const exception& e) {
        cerr << "Erro: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}