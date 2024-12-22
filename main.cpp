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
    double label{};
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
    auto rng = default_random_engine {};
    ranges::shuffle(data, rng);

    auto trainSize = static_cast<size_t>(trainRatio * data.size());

    train = vector<SampleData>(data.begin(), data.begin() + trainSize);
    test = vector<SampleData>(data.begin() + trainSize, data.end());
}

double sigmoid(const double z) {
    return 1.0 / (1.0 + exp(-1.0 * z));
}

double computeCost(const vector<SampleData> &X, const vector<double> &w, const double b) {
    double totalCost = 0.0;
    for (int i = 0; i < X.size(); i++) {
        double z = 0.0;
        for (int j = 0; j < X[i].features.size(); j++) {
            z += X[i].features[j] * w[j];
        }
        z += b;

        const double sig = sigmoid(z);

        totalCost += (-X[i].label * log(sig)) - (1 - X[i].label) * log(1 - sig);
    }
    return totalCost / static_cast<double>(X.size());
}
// computesGradient and modifies dj_dw & dj_db
void computeGradient(const vector<SampleData> &X, const vector<double> &w, const double b, vector<double> &dj_dw, double &dj_db) {
    int m = X.size();
    ranges::fill(dj_dw, 0.0);
    dj_db = 0.0;

    // loop over all samples in data
    for (int i = 0; i < m; i++) {
        double z = 0;
        // loop over features in sample, basically computing  f(x)
        int n = X[i].features.size();
        for (int j = 0; j < n; j++) {
            z += X[i].features[j] * w[j];
        }
        z += b;

        const double sig = sigmoid(z);
        const double error = sig - X[i].label;

        for (int j = 0; j < n; j++) {
            dj_dw[j] += error * X[i].features[j];
        }
        dj_db += error;
    }
    // Average the gradients
    for (double &grad : dj_dw) {
        grad /= static_cast<double>(m);
    }
    dj_db /= static_cast<double>(m);
}

void gradientDescent(const vector<SampleData> &X, vector<double> &w, double &b, const double alpha, const int numIterations = 1000) {
    vector<double> dj_dw(w.size(), 0);
    double dj_db = 0.0;

    for (int i = 0; i < numIterations; i++) {
        computeGradient(X, w, b, dj_dw, dj_db);

        // perform descent
        for (int j = 0; j < dj_dw.size(); j++) {
            w[j] -= alpha * dj_dw[j];
        }
        b -= alpha * dj_db;


        if (i % 100 == 0) {
            cout << "Iteration " << i << ": Cost: "<< computeCost(X, w, b) << endl;
        }
    }
}

void predict(const vector<SampleData> &X, const vector<double> &w, const double b, vector<int> &p) {
    int m = X.size();
    int n = w.size();

    for (int i = 0; i < m; i++) {
        double z = 0.0;
        for (int j = 0; j < n; j++) {
            z += X[i].features[j] * w[j];
        }
        z += b;

        const double sig = sigmoid(z);

        p[i] = sig >= 0.5 ? 1 : 0;
    }
}

double modelAccuracy(const vector<SampleData> &X, const vector<int> &p) {
    int m = X.size();
    int numCorrect = 0;

    for (int i = 0; i < m; i++) {
        numCorrect += (p[i] == X[i].label) ? 1: 0;
    }
    return static_cast<double>(numCorrect) / static_cast<double>(m);
}
// TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
int main() {
    // TIP Press <shortcut actionId="RenameElement"/> when your caret is at the
    // <b>lang</b> variable name to see how CLion can help you rename it.
    string filename = "heart_disease.csv";
    vector<SampleData> data;
    bool success = loadDatasetFromCsv(filename, data);

    if (!success) {
        cerr << "Could not load data from " << filename << endl;
        return 1;
    }

    // split train/test data
    vector<SampleData> train, test;
    splitTrainTest(data, train, test);

    // train model
    vector<double> w(15, 0);
    double b = 0.0;
    double alpha = 0.001;
    gradientDescent(train, w, b, alpha, 20000);

    // predict and evaluate accuracy
    vector<int> p(test.size(), 0);
    predict(test, w, b, p);
    double accuracy = modelAccuracy(test, p);
    cout << "Model Accuracy: " << accuracy << endl; // 83% currently

    return 0;
}