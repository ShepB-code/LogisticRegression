#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>

using namespace std;

struct SampleData {
    vector<double> features;
    int label{};
};

bool loadDatasetFromCsv(const string &filename, vector<SampleData> &result)  {
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Could not open file " << filename << endl;
        return false;
    }
    string line;
    // skip header
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        SampleData sample;

        // read 15 features TODO: fix this to be dependent on number of features
        for (int i = 0; i < 15; i++) {
            getline(ss, value, ',');
            if (value == "NA") {
                value = "0";
            }
            sample.features.push_back(stod(value));
        }
        // get label
        getline(ss, value, ',');
        sample.label = stoi(value);
        result.push_back(sample);
    }
    return true;
}

void splitTrainTest(vector<SampleData> &data, vector<SampleData> &train, vector<SampleData> &test, double trainRatio = 0.8) {
    auto rng = default_random_engine{static_cast<unsigned int>(time(nullptr))};
    ranges::shuffle(data, rng);

    auto trainSize = static_cast<size_t>(trainRatio * data.size());

    train = vector<SampleData>(data.begin(), data.begin() + trainSize);
    test = vector<SampleData>(data.begin() + trainSize, data.end());
}

double sigmoid(const double z) {
    return 1.0 / (1.0 + exp(-1.0 * z));
}

double computeCost(const vector<SampleData> &X, const vector<double> &w, const double b) {
    size_t m = X.size(), n = w.size();
    double totalCost = 0.0;
    for (size_t i = 0; i < m; i++) {
        double z = 0.0;
        for (size_t j = 0; j < n; j++) {
            z += X[i].features[j] * w[j];
        }
        z += b;

        const double sig = sigmoid(z);

        totalCost += (-X[i].label * log(sig)) - (1 - X[i].label) * log(1 - sig);
    }
    return totalCost / static_cast<double>(m);
}
// computesGradient and modifies dj_dw & dj_db
void computeGradient(const vector<SampleData> &X, const vector<double> &w, const double b, vector<double> &dj_dw, double &dj_db, const double lambda = 0.01) {
    size_t m = X.size(), n = w.size();
    ranges::fill(dj_dw, 0.0);
    dj_db = 0.0;

    // loop over all samples in data
    for (size_t i = 0; i < m; i++) {
        double z = 0;
        // loop over features in sample, basically computing  f(x)
        for (size_t j = 0; j < n; j++) {
            z += X[i].features[j] * w[j];
        }
        z += b;

        const double sig = sigmoid(z);
        const double error = sig - X[i].label;

        for (size_t j = 0; j < n; j++) {
            dj_dw[j] += error * X[i].features[j];
        }
        dj_db += error;
    }
    // Average the gradients
    for (double &grad : dj_dw) {
        grad /= static_cast<double>(m);
    }

    // regularize
    for (size_t i = 0; i < n; i++) {
        dj_dw[i] += (lambda / m) * w[i];
    }

    dj_db /= static_cast<double>(m);
}

void gradientDescent(const vector<SampleData> &X, vector<double> &w, double &b, const double alpha, const int numIterations = 1000) {
    vector<double> dj_dw(w.size(), 0);
    double dj_db = 0.0;

    for (size_t i = 0; i < numIterations; i++) {
        computeGradient(X, w, b,  dj_dw, dj_db);

        // perform descent
        for (size_t j = 0; j < dj_dw.size(); j++) {
            w[j] -= alpha * dj_dw[j];
        }
        b -= alpha * dj_db;


        if (i % 1000 == 0) {
            cout << "Iteration " << i << ": Cost: "<< computeCost(X, w, b) << endl;
        }
    }
}

void predict(const vector<SampleData> &X, const vector<double> &w, const double b, vector<int> &p) {
    size_t m = X.size();
    size_t n = w.size();
    double mean = 0.0;
    for (size_t i = 0; i < m; i++) {
        double z = 0.0;
        for (size_t j = 0; j < n; j++) {
            z += X[i].features[j] * w[j];
        }
        z += b;

        const double sig = sigmoid(z);
        mean += sig;
        p[i] = sig >= 0.5 ? 1 : 0;
    }
    cout << "Mean Prediction: " << mean / static_cast<double>(m) << endl;
}

double modelAccuracy(const vector<SampleData> &X, const vector<int> &p) {
    size_t m = X.size();
    int numCorrect = 0, total0 = 0, total1 = 0, num0Correct = 0, num1Correct = 0;

    for (int i = 0; i < m; i++) {
        if (X[i].label == 0) {
            total0 += 1;
        } else {
            total1 += 1;
        }

        if (p[i] == X[i].label) {
            numCorrect += 1;
            num0Correct += X[i].label == 0 ? 1 : 0;
            num1Correct += X[i].label == 1 ? 1 : 0;
        }
    }
    cout << "0 Accuracy: " << static_cast<double>(num0Correct) / static_cast<double>(total0) << endl;
    cout << "1 Accuracy: " << static_cast<double>(num1Correct) / static_cast<double>(total1) << endl;
    return static_cast<double>(numCorrect) / static_cast<double>(m);
}

void normalizeFeatures(vector<SampleData> &X) {
    const size_t m = X.size(), n = X[0].features.size();

    vector<double> mean(n, 0.0), stdDev(n, 0.0);

    // calculate mean for each feature
    for (const auto &sample : X) {
        for (size_t j = 0; j < n; j++) {
            mean[j] += sample.features[j];
        }
    }
    for (auto &val : mean) val /= static_cast<double>(m);

    // calculate stdDev for each feature
    for (const auto &sample : X) {
        for (size_t j = 0; j < n; j++) {
            stdDev[j] += pow(sample.features[j] - mean[j], 2);
        }
    }
    for (auto &val : stdDev) val = sqrt(val / static_cast<double>(m));

    // normalize
    for (auto &sample : X) {
        for (size_t j = 0; j < n; j++) {
            sample.features[j] = (sample.features[j] - mean[j]) / stdDev[j];
        }
    }

}

int main() {
    string filename = "heart_disease.csv";
    vector<SampleData> data;
    bool success = loadDatasetFromCsv(filename, data);

    if (!success) {
        cerr << "Could not load data from " << filename << endl;
        return 1;
    }

    // normalize features using z-score normalization
    normalizeFeatures(data);

    // split train/test data
    vector<SampleData> train, test;
    splitTrainTest(data, train, test);
    // train model
    vector<double> w(15, 0);
    double b = 0.0;
    double alpha = 0.01;
    gradientDescent(train, w, b, alpha, 10000);

    // predict and evaluate accuracy
    vector<int> p(test.size(), 0);
    predict(test, w, b, p);
    double accuracy = modelAccuracy(test, p);
    cout << "Model Accuracy: " << accuracy << endl; // ~85% currently

    return 0;
}