//Ethan Huynh
//EXH190016

#define _USE_MATH_DEFINES

#include <bits/stdc++.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cmath>

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

// sums up all elements of some vector of type double and returns as a double
double sum(vector<double> v) {
    return accumulate(v.begin(), v.end(), 0.0);
}

// takes the sum and divides it by the size of the vector to find mean
double mean(vector<double> v) {
    return sum(v)/v.size();
}

//s = sqrt( (SUM(xi - mean_x)^2) / (n-1) )
double variance(vector<double> v)
{
    double sum = 0.0;
    double m = mean(v);

    for (int i = 0; i < v.size(); i++)
    {
        sum+=pow((v[i]-m), 2);
    }

    sum /= (v.size()-1);

    return sqrt(sum);
}

// calculates age likelihood
double calc_age_lh(double v, double mean_v, double var_v)
{
    return 1 / sqrt(2 * M_PI * var_v) * exp(-1 * (pow((v - mean_v), 2)) / (2 * var_v));
}

// calculates raw probability with pclass, sex, and age
// pclass 1,2,3; sex = 1,2; age = numeric
vector<double> calc_raw_prob(int pclass, int sex, double age, double prior_s, double prior_d, 
vector<double> age_mean, vector<double> age_var, 
vector<vector<double>> &lh_pclass, vector<vector<double>> &lh_sex)
{
    double num_s = lh_pclass[1][pclass] * lh_sex[1][sex] * prior_s * calc_age_lh(age, age_mean[1], age_var[1]);

    double num_p = lh_pclass[0][pclass] * lh_sex[0][sex] * prior_d * calc_age_lh(age, age_mean[0], age_var[0]);

    double denominator = lh_pclass[1][pclass] * lh_sex[1][sex] * calc_age_lh(age, age_mean[1], age_var[1]) * prior_s + 
                        lh_pclass[0][pclass] * lh_sex[0][sex] * calc_age_lh(age, age_mean[0], age_var[0]) * prior_d;

    double prob_survived = num_s / denominator;
    double prob_perished = num_p / denominator;

    return {prob_perished, prob_survived};
}

// prints out all the elements of the matrix
void print_matrix(vector<vector<double>> v, string name) 
{
    cout << name << ": " << endl;
    for (int r = 0; r < v.size(); r++)
    {
        for (int c = 0; c < v[r].size(); c++)
        {
            cout << v[r][c] << " ";
        }
        cout << endl;
    }
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

    auto start = chrono::high_resolution_clock::now();
    // find total survived and dead
    double train_survived = 0.0, train_dead = 0.0;
    for (int r = 0; r < train.size(); r++)
    {
        if (train[r][2] == 1)
            train_survived+=1;
        else
            train_dead+=1;
    }

    // calculate prior probabilities
    double prior_survived = train_survived / train.size();
    double prior_dead = train_dead / train.size();
    cout << "Prior probability: " << endl;
    cout << "Survived: " << prior_survived << endl;
    cout << "Dead: " << prior_dead << endl;

    // likelihood (class=i|survived=yes) = count(factor = i and survived=yes) / count(survived=yes)
    // likelihood (class=i|survived=no) = count(factor = i and survived=no) / count(survived=no)
    // p -> pclass (1,2,3), s -> survived (0,1), sx -> sex (0,1)
    double p1s0 = 0.0, p1s1 = 0.0, p2s0 = 0.0, p2s1 = 0.0, p3s0 = 0.0, p3s1 = 0.0;
    double sx0s0 = 0.0, sx0s1 = 0.0, sx1s0 = 0.0, sx1s1 = 0.0;
    vector<double> meanS0;
    vector<double> varS0;
    vector<double> meanS1;
    vector<double> varS1;

    // get counts for lh calculation
    for (int r = 0; r < train.size(); r++)
    {
        for (int c = 0; c < train[r].size(); c++)
        {
            if (c == 1) // counts for pclass
            {
                if (train[r][c]==1) // if pclass is 1
                {
                    if (train[r][2]==0) // if survived is 0
                        p1s0+=1;
                    else // else is 1
                        p1s1+=1;
                }
                else if (train[r][c]==2) // if pclass is 2
                {
                    if (train[r][2]==0) // if survived is 0
                        p2s0+=1;
                    else // else is 1
                        p2s1+=1;
                }
                else // if pclass is 3
                {
                    if (train[r][2]==0) // if survived is 0
                        p3s0+=1;
                    else // else is 1
                        p3s1+=1;
                }
            }
            else if (c == 3) // counts for sex
            {
                if (train[r][c]==0) // if sex is 0
                {
                    if (train[r][2]==0) // if survived is 0
                        sx0s0+=1;
                    else // else is 1
                        sx0s1+=1;
                }
                else // if sex is 1
                {
                    if (train[r][2]==0) // if survived is 0
                        sx1s0+=1;
                    else // else is 1
                        sx1s1+=1;
                }
            }
            else if (c == 4) // gathers ages for mean and var calc
            {
                if (train[r][2]==0) // if survived is 0
                {
                    meanS0.push_back(train[r][c]);
                    varS0.push_back(train[r][c]);
                }
                else // else is 1
                {
                    meanS1.push_back(train[r][c]);
                    varS1.push_back(train[r][c]);
                }
            }
        }
    }

    // calculate likelihoods
    p1s0 /= train_dead;
    p1s1 /= train_survived;
    p2s0 /= train_dead;
    p2s1 /= train_survived;
    p3s0 /= train_dead;
    p3s1 /= train_survived;

    cout << "train_dead: " << train_dead << endl;
    cout << "train_survived: " << train_survived << endl;
    cout << "p1s0: " << p1s0 << endl;
    cout << "p1s1: " << p1s1 << endl;
    cout << "p2s0: " << p2s0 << endl;
    cout << "p2s1: " << p2s1 << endl;
    cout << "p3s0: " << p3s0 << endl;
    cout << "p3s1: " << p3s1 << endl;

    sx0s0 /= train_dead;
    sx0s1 /= train_survived;
    sx1s0 /= train_dead;
    sx1s1 /= train_survived;

    // create matrices
    vector<vector<double>> lh_pclass = {{p1s0, p2s0, p3s0}, {p1s1, p2s1, p3s1}};
    vector<vector<double>> lh_sex = {{sx0s0, sx1s0}, {sx0s1, sx1s1}};
    vector<double> age_mean = {mean(meanS0), mean(meanS1)};
    vector<double> age_var = {variance(varS0), variance(varS1)};

    // print out likelihoods
    print_matrix(lh_pclass, "lh_pclass");
    print_matrix(lh_sex, "lh_sex");
    cout << "age_mean: " << endl << age_mean[0] << " " << age_mean [1] << endl;
    cout << "age_var: " << endl << age_var[0] << " " << age_var [1] << endl;

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // find test survived and dead
    double test_survived = 0.0, test_dead = 0.0;
    for (int r = 0; r < test.size(); r++)
    {
        if (test[r][2] == 1)
            test_survived+=1;
        else
            test_dead+=1;
    }
    double acc = 0.0, sens = 0.0, spec = 0.0;
    vector<double> pred(test.size());

    for (int r = 0; r < test.size(); r++)
    {
        vector<double> raw = calc_raw_prob(test[r][1], test[r][3], test[r][4], 
        prior_survived, prior_dead, age_mean, age_var, lh_pclass, lh_sex);

        if (r < 5)
            cout << "Raw: " << raw[0] << " " << raw[1] << endl; 

        if (raw[0] >= 0.5)
            pred[r] = 1;
        else
            pred[r] = 0;
        
        if (pred[r] == test[r][2])
        {
            acc+=1;
            if (pred[r] == 1)
                sens+=1;
            else
                spec+=1;
        }   
    }

    // calculating acc, sens, and spec
    acc /= pred.size();
    sens /= test_survived;
    spec /= test_dead;

    cout << "accuracy: " << acc << endl;
    cout << "sensitivity: " << sens << endl;
    cout << "specificity: " << spec << endl;
    cout << "prediction: " << pred.size() << endl;
    cout << "training time(microseconds): " << duration.count() << endl;

    return 0;
}
