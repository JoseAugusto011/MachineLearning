// Perceptron Learning Algorithm
// José Augusto da Silva Barbosa - Dezembro de 2023

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIZE 1000      // Tamanho do conjunto de dados
#define TRAIN_SIZE 100 // Tamanho do conjunto de treinamento

typedef struct
{ // Definiç~so do tipo ponto: Usado para criar a reta e os pontos do conjunto de dados
    double x;
    double y;
} Point;

typedef struct
{ // Definiç~so do tipo pesos: Usado para armazenar os pesos da reta
    double w0;
    double w1;
    double w2;
} Weights;

Point createRandomPoint()
{ // Cria um ponto aleatório no quadrado [-1, 1] x [-1, 1]
    Point p;
    p.x = ((double)rand() / RAND_MAX) * 2 - 1;
    p.y = ((double)rand() / RAND_MAX) * 2 - 1;
    return p;
}

void createPoints(Point *points, int size)
{ // Cria um conjunto de pontos aleatórios
    for (int i = 0; i < size; ++i)
    {
        points[i] = createRandomPoint();
    }
}

Weights createLine(Point p1, Point p2)
{ // Cria uma reta que passa pelos pontos p1 e p2
    Weights weights;
    double m = (p2.y - p1.y) / (p2.x - p1.x);
    weights.w1 = -m;
    weights.w2 = 1.0;
    weights.w0 = -p1.y + m * p1.x;
    return weights;
}

double sign(double x)
{ // Função de ativação
    return (x > 0) ? 1.0 : -1.0;
}

void updateWeights(Weights *weights, Point point, double label)
{ // Atualiza os pesos
    weights->w0 += label;
    weights->w1 += label * point.x;
    weights->w2 += label * point.y;
}

int perceptron(Point *points, double *labels, Weights *weights)
{ // Algoritmo Perceptron Learning Algorithm - PLA
    int iterations = 0;
    int misclassified = 1; // Inicializa com 1 para entrar no loop

    while (misclassified > 0)
    { // Enquanto houver pontos mal classificados
        misclassified = 0;

        for (int i = 0; i < SIZE; ++i)
        {                                                                                                  // Percorre todos os pontos
            double prediction = sign(weights->w0 + weights->w1 * points[i].x + weights->w2 * points[i].y); // Calcula a predição
            if (prediction != labels[i])
            {                                                 // Se a predição for diferente da classe real
                updateWeights(weights, points[i], labels[i]); // Atualiza os pesos
                misclassified++;
            }
        }

        iterations++;

        if (iterations > 1000)
        {
            // Se o algoritmo não convergir em 1000 iterações, para o loop
            break;
        }
    }

    return iterations;
}

double calculateAccuracy(Point *points, double *labels, Weights *weights, int size)
{ // Calcula a acurácia do modelo
    int correct = 0;
    for (int i = 0; i < size; ++i)
    {
        double prediction = sign(weights->w0 + weights->w1 * points[i].x + weights->w2 * points[i].y);
        if (prediction == labels[i])
        {
            correct++;
        }
    }

    return (double)correct / size;
}

int main()
{
    Point points[SIZE];
    createPoints(points, SIZE);

    Point p1 = createRandomPoint();
    Point p2 = createRandomPoint();
    Weights target = createLine(p1, p2);

    double labels[SIZE];
    for (int i = 0; i < SIZE; ++i)
    {
        labels[i] = sign(target.w0 + target.w1 * points[i].x + target.w2 * points[i].y);
    }

    Weights weights = {0, 0, 0};
    int iterations = perceptron(points, labels, &weights);

    printf("Iterations: %d\n", iterations);
    printf("Weights: w0 = %lf, w1 = %lf, w2 = %lf\n", weights.w0, weights.w1, weights.w2);

    double accuracy = calculateAccuracy(points + TRAIN_SIZE, labels + TRAIN_SIZE, &weights, SIZE - TRAIN_SIZE);
    printf("Accuracy: %lf\n", accuracy);

    return 0;
}
