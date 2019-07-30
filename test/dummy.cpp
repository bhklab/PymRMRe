/* Test the cpp function */
#include "dummy.h"
#include <iostream>
#include <iterator>
#include <vector>
#include <array>
#include <utility>

using namespace std;

pair <vector<int>, vector<vector<double> > > c_compute (double* array, double multiplier, int m, int n) {
    int i, j;
    int index = 0;
    vector<int> first;
    vector<vector<double> > second;

    for (i = 0; i < m; i++) {
        first.push_back(i);
        vector<double> new_s;
        second.push_back(new_s);
        for (j = 0; j < n; j++) {
            array[index] *= multiplier;
            second[i].push_back(array[index]);
            index ++;
        }
    }
    return make_pair(first, second);
}