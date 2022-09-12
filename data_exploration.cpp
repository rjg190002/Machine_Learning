#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
using namespace std;

double range(vector<double> vec)
{
    sort(vec.begin(), vec.end());
    return (vec.at(505) - vec.at(0));
}

double median(vector<double> vec, int length)
{
    sort(vec.begin(), vec.end());
    if(length % 2 == 0)
    {
        return (vec.at((length/2) - 1) + vec.at((length/2))) / 2;
    }
    else
        return vec.at(length / 2);
}

double summation(vector<double> vec)
{
    //find summation of the entire vector
    double sum = accumulate(vec.begin(), vec.end(), 0.0);
    return sum;
}

double mean(vector<double> vec, int length)
{
    double mean = summation(vec) / length;
    return mean;
}

double covariance(vector<double> vec1, vector<double> vec2, int length)
{
    //Find the mean for both vectors
    double vec1_mean = mean(vec1, length);
    double vec2_mean = mean(vec2, length);

    double temp = 0;
    //get the summation for the covariance
    for(int n = 0; n < length; n++)
    {
        temp += ((vec1.at(n) - vec1_mean)*(vec2.at(n) - vec2_mean));
    }
    //divide it by n-1 to find the final answer
    temp = temp / (length-1);
    return temp;
}

double correlation(vector<double> vec1, vector<double> vec2, int length)
{
    //Finding variance of the two sets
    //first set
    double vec1_variance = 0;
    double vec1_mean = mean(vec1, length);
    for(int n = 0; n < length; n++)
    {
        vec1_variance += pow((vec1.at(n) - vec1_mean), 2);
    }
    vec1_variance = vec1_variance / (length-1);

    //second set
    double vec2_variance = 0;
    double vec2_mean = mean(vec2, length);
    for(int n = 0; n < length; n++)
    {
        vec2_variance += pow((vec2.at(n) - vec2_mean), 2);
    }
    vec2_variance = vec2_variance / (length-1);

    //finding the standard deviation of both vectors
    double vec1_stdev = sqrt(vec1_variance);
    double vec2_stdev = sqrt(vec2_variance);

    //final equation for correlation
    double corr = (covariance(vec1, vec2, length) / (vec1_stdev * vec2_stdev));
    return corr;
}

int main(int argc, char** argv)
{
    ifstream inFS;
    string line;
    string rm_in, medv_in;
    const int MAX_LEN = 1000;
    vector<double> rm(MAX_LEN);
    vector<double> medv(MAX_LEN);

    cout << "Opening file" << endl;

    inFS.open("Boston.csv");
    if(!inFS.is_open()){
        cout << "Wrong file name" << endl;
        return 1;
    }

    cout << "Reading line one:" << endl;
    getline(inFS, rm_in, ',');
    getline(inFS, medv_in, '\n');
    int numObservations = 0;
    while(inFS.good()){

        getline(inFS, rm_in, ',');
        getline(inFS, medv_in, '\n');

        rm.at(numObservations) = stof(rm_in);
        medv.at(numObservations) = stof(medv_in);

        numObservations++;
    }

    rm.resize(numObservations);
    medv.resize(numObservations);

    cout << "new length: " << rm.size() << endl;

    cout << "closing file" << endl;
    inFS.close();

    cout << "number of records: " << numObservations << endl << endl;

    
    cout << "Summation of rm: " << summation(rm) << endl;
    cout << "Summation of medv: " << summation(medv) << endl << endl;

    cout << "Mean of rm: " << mean(rm, numObservations) << endl;
    cout << "Mean of medv: " << mean(medv, numObservations) << endl << endl;

    cout << "Median of rm: " << median(rm, numObservations) << endl;
    cout << "Median of medv: " << median(medv, numObservations) << endl << endl;

    cout << "Range of rm: " << range(rm) << endl;
    cout << "Range of medv: " << range(medv) << endl << endl;

    cout << "Covariance of the two: " << covariance(rm, medv, numObservations) << endl << endl;

    cout << "Correlation of the two: " << correlation(rm, medv, numObservations) << endl;



    return 0;
}