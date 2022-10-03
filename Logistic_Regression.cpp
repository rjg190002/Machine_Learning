//Ethan Huynh
//EXH190016
#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>

using namespace std;

// slice the given vector's rows from range [S, E]
vector<vector<double>> vrow_slice(vector<vector<double>>& input, int S, int E)
{
    // Start and End iterators
    auto start = input.begin() + S;
    auto end = input.begin() + E + 1;
 
    // To store the sliced vector
    vector<vector<double>> result(E - S + 1);
 
    // Copy vector using copy function()
    copy(start, end, result.begin());
 
    // Return the final sliced vector
    return result;
}

// slice the given vector's cols from given cols
vector<vector<double>> vcol_slice(vector<vector<double>>& input, vector<int> cols)
{
    // To store the sliced vector
    vector<vector<double>> result;

    // loop through input vector and only take needed values
    for (int r = 0; r < input.size(); r++)
    {
        // create row vector to push values into
        vector<double> row;

        // loop through wanted columns
        for (int c: cols)
        {
            // take wanted values
            row.push_back(input[r][c]);
        }
        // add the row into the result
        result.push_back(row);
    }
    
    // Return the final sliced vector
    return result;
}

// vector subtraction element-wise, v1.size() == v2.size()
vector<vector<double>> vsub(vector<vector<double>>& v1, vector<vector<double>>& v2, int c1, int c2)
{
    vector<vector<double>> result;

    for(int r = 0; r < v1.size(); r++)
    {
        // create row vector to push values into
        vector<double> row;

        row.push_back(v1[r][c1] - v2[r][c2]);

        // add the row into the result
        result.push_back(row);
    }

    return result;
}

// transpose a vector
vector<vector<double>> transpose(vector<vector<double>>& input)
{
    // create result vector
    vector<vector<double>> result(input[0].size(), vector<double> (input.size()));

    for (int r = 0; r < input.size(); r++)
    {
        for (int c = 0; c < input[0].size(); c++)
            // fill result vector with transposed values
            result[c][r] = input[r][c];
    }

    return result;
}

// return a vector of sigmoid values from an input vector with the value of the sigmoid function f(x) = 1/(1 + e^-x).
vector<vector<double>> sigmoid (vector<vector<double>> &input) 
{
    vector<vector<double>> result(input.size(), vector<double> (input[0].size()));
    
    for (int r = 0; r < input.size(); r++)
    {
        for (int c = 0; c < input[0].size(); c++)
            result[r][c] = 1.0 / (1+exp(-(input[r][c])));
    }
    
    return result;
}

// multiply 2 vectors together as matrices
vector<vector<double>> matrix_mult(vector<vector<double>> &v1, vector<vector<double>> &v2)
{
    // create result vector
    vector<vector<double>> result(v1.size(), vector<double> (v2[0].size()));

    // loop through first vectors rows
    for (int i = 0; i < v1.size(); i++) 
    {
        // loop though second vectors columns
        for (int j = 0; j < v2[0].size(); j++) 
        {
            result[i][j] = 0;

            // setting value
            for (int k = 0; k < v2.size(); k++)
                result[i][j] += v1[i][k] * v2[k][j];
        }
    }
    return result;
}

// Driver function
int main()
{
    ifstream inFS;  //input file stream
    string line;
    const int ROW_LEN = 1050;
    const int COL_LEN = 5;
    const int TRAIN_LEN = 800;
    vector<vector<double>> data;
    vector<vector<double>> train;
    vector<vector<double>> test;

    //Attempt to open file
    cout << "Opening file" << endl;

    inFS.open("titanic_project.csv");
    if (!inFS.is_open()) {
        cout << "Error opening file" << endl;
        return 1;
    }

    //Read the open file
    cout << "Reading line 1" << endl;
    getline(inFS, line);

    //print the columns headers
    cout << "Headers: " << line << endl;
    while (inFS.good()) {
        // create row vector to push values into
        vector<double> row;

        // read each row into its own string and convert to stream
        getline(inFS, line);
        stringstream ss_in(line);

        // parse input string
        for (int col = 0; col < COL_LEN; col++)
        {
            string val;

            // read each value in the row to the string
            if (col == 4)
                getline(ss_in, val);
            else
                getline(ss_in, val, ',');
            
            // need to parse out the "" from the first column
            if (col == 0)
                val.erase(remove(val.begin(), val.end(), '"'), val.end());
            
            row.push_back(stod(val));
        }
        // add the row into the data vector
        data.push_back(row);
    }
    // close file
    inFS.close();

    // Set train and test vectors
    train = vrow_slice(data, 0, 799);
    test = vrow_slice(data, 800, data.size()-1);

    // find test survived and dead
    double test_survived = 0.0, test_dead = 0.0;
    for (int r = 0; r < test.size(); r++)
    {
        if (test[r][2] == 1)
            test_survived+=1;
        else
            test_dead+=1;
    }

    // create vector of just the sex values from training data
    vector<vector<double>> sex_train = vcol_slice(train, {3});

    // create vector of weights
    vector<vector<double>> weights = {{1}, {1}};
    // create 2d vector of data_matrix with same number of rows as sex_train and 2 columns of all 1's
    vector<vector<double>> data_matrix(sex_train.size(), vector<double>(2, 1));
    // loop through data_matrix, replacing 2nd column with sex values
    for (int r = 0; r < data_matrix.size(); r++)
        data_matrix[r][1] = sex_train[r][0];

    vector<vector<double>> labels = vcol_slice(train, {2});
    
    float learning_rate = 0.001;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 1; i < 5000; i++) 
    {
        // 800r, 2c * 2r, 1c = 800r, 1c
        vector<vector<double>> temp = matrix_mult(data_matrix, weights);
        vector<vector<double>> prob_vector = sigmoid(temp);
        vector<vector<double>> error = vsub(labels, prob_vector, 0, 0);
        vector<vector<double>> dm_transpose = transpose(data_matrix);
        
        // // 2r, 800c * 800r, 1c = 2r, 1c
        temp = matrix_mult(dm_transpose, error);
        
        weights[0][0] = weights[0][0] + learning_rate * temp[0][0];
        weights[1][0] = weights[1][0] + learning_rate * temp[1][0];
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // print weights
    for (int x = 0; x < weights.size(); x++)
    {
        for (int y = 0; y < weights[x].size();  y++)
        {
            cout << "weights: " << weights[x][y] << endl;
        }
    }

    // Predict with the weights we generated

    // create vector of just the sex values from test data
    vector<vector<double>> sex_test = vcol_slice(test, {3});
    // create 2d vector of test_matrix with same number of rows as sex_train and 2 columns of all 1's
    vector<vector<double>> test_matrix(sex_test.size(), vector<double>(2, 1));
    // loop through test_matrix, replacing 2nd column with sex values
    for (int r = 0; r < test_matrix.size(); r++)
        test_matrix[r][1] = sex_test[r][0];

    vector<vector<double>> test_labels = vcol_slice(test, {2});

    vector<vector<double>> predicted = matrix_mult(test_matrix, weights);
    vector<vector<double>> probabilities(predicted.size(), vector<double>(predicted[0].size()));
    vector<vector<double>> predictions(predicted.size(), vector<double>(predicted[0].size()));
    double acc = 0.0, sens = 0.0, spec = 0.0;
    for (int r = 0; r < predicted.size(); r++)
    {
        for (int c = 0; c < predicted[0].size(); c++)
        {
            probabilities[r][c] = exp(predicted[r][c]) / (1 + exp(predicted[r][c]));
            if (probabilities[r][c] >= 0.5)
                predictions[r][c] = 1;
            else
                predictions[r][c] = 0;
            
            if (predictions[r][c] == test_labels[r][c])
            {
                acc+=1;
                if (predictions[r][c] == 1)
                    sens+=1;
                else
                    spec+=1;
            }   
        }
    }

    // calculating acc, sens, and spec
    acc /= predictions.size();
    sens /= test_survived;
    spec /= test_dead;

    cout << "accuracy: " << acc << endl;
    cout << "sensitivity: " << sens << endl;
    cout << "specificity: " << spec << endl;
    cout << "learning rate: " << learning_rate << endl;
    cout << "iterations: 5000" << endl;
    cout << "training time(microseconds): " << duration.count() << endl;

    return 0;
}
