#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIZE 1000
#define TRAIN_SIZE 100

typedef struct {
    double x;
    double y;
} Point;

typedef struct {
    double w0;
    double w1;
    double w2;
} Weights;

Point createRandomPoint() {
    Point p;
    p.x = ((double)rand() / RAND_MAX) * 2 - 1; // Random number between -1 and 1
    p.y = ((double)rand() / RAND_MAX) * 2 - 1;
    return p;
}

void createPoints(Point *points, int size) {
    for (int i = 0; i < size; ++i) {
        points[i] = createRandomPoint();
    }
}

Weights createLine(Point p1, Point p2) {
    Weights weights;
    double m = (p2.y - p1.y) / (p2.x - p1.x);
    weights.w1 = -m;
    weights.w2 = 1.0;
    weights.w0 = -p1.y + m * p1.x;
    return weights;
}

double sign(double x) {
    return (x > 0) ? 1.0 : -1.0;
}

void updateWeights(Weights *weights, Point point, double label) {
    weights->w0 += label;
    weights->w1 += label * point.x;
    weights->w2 += label * point.y;
}

int perceptron(Point *points, double *labels, Weights *weights) {
    int iterations = 0;
    int misclassified = 1; // Dummy value to enter the loop

    while (misclassified > 0) { // While there are misclassified points
        misclassified = 0;

        for (int i = 0; i < SIZE; ++i) {
            double prediction = sign(weights->w0 + weights->w1 * points[i].x + weights->w2 * points[i].y);
            if (prediction != labels[i]) {
                updateWeights(weights, points[i], labels[i]);
                misclassified++;
            }
        }

        iterations++;

        if (iterations > 1000) {
            // Break if it takes too many iterations
            break;
        }
    }

    return iterations;
}

double calculateAccuracy(Point *points, double *labels, Weights *weights, int size) {
    int correct = 0;
    for (int i = 0; i < size; ++i) {
        double prediction = sign(weights->w0 + weights->w1 * points[i].x + weights->w2 * points[i].y);
        if (prediction == labels[i]) {
            correct++;
        }
    }

    return (double)correct / size;
}

int main() {
    Point points[SIZE];
    createPoints(points, SIZE);

    Point p1 = createRandomPoint();
    Point p2 = createRandomPoint();
    Weights target = createLine(p1, p2);

    double labels[SIZE];
    for (int i = 0; i < SIZE; ++i) {
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
