// include/TrainingConfig.h
#ifndef TRAINING_CONFIG_H
#define TRAINING_CONFIG_H

template<typename T>
struct TrainingConfig {
    int max_iterations = 1000;
    T tolerance = static_cast<T>(1e-4);
    bool verbose = false;
    
    // Parâmetros específicos do PLA Pocket
    int pocket_update_frequency = 10;
};

#endif