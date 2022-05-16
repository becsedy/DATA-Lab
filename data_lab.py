import numpy as np
import pandas as pd
import statsmodels.api as stats
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

'''
Variables:
0: Pickup Diff (min)*
1: Dropoff Diff (min)*
2: Response Time (min)
3: Walking Distance (mi)
4: Ride Distance (mi)
5: Ride Duration (min)
6: Destination Lat
7: Destination Long
8: Request Creation Time
9: Rating

*positive time = picked up/dropped off earlier than expected
negative time = picked up/dropped off later than expected
'''

rows = []
ratings = []
good_variables = []
CANCEL_COL = 13
RATING_COL = 18
FLOAT_COLS = [2, 4, 6, 7, 10, 11, 12, 16, 17, 18, 19, 20]
FILENAME = "ride_requests_2020A.csv"
ALPHA = 0.10

def openfile(filename, cancel_col, rating_col):
    '''
    Function: open_file(filename)
    Parameters: filename, a string
    Returns: rows, a list of the rows in the file
    '''
    # importing data from file into a list of strings
    with open(filename, "r", encoding = "utf-8-sig") as infile:
        while True:
            row = infile.readline()
            if row == "":
                break
            else:
                rows.append(row)
    
    # separating each string into a list
    for i in range(len(rows)):
        rows[i] = rows[i].split(sep = ",")
        
    # removing incomplete data
    for i in range(len(rows) - 1, -1, -1):
        if rows[i][rating_col] == "\n":
            rows.pop(i)
    
    for i in range(len(rows) - 1, -1, -1):
        for j in range(len(rows[i])):
            if j != cancel_col and rows[i][j] == "":
                rows.pop(i)
                break
            else:
                rows[i][j] = rows[i][j].strip()
    
    rows.pop(0)
    
    return rows

def floatize(data, float_cols):
    '''
    Function: floatize(data, float_cols)
    Parameters: data, a list, and float_cols, a list
    Returns: the data list with all of the data in a column in
        float_cols as a float
    '''
    for i in range(len(rows)):
        for j in range(len(data[i])):
            if j in float_cols:
                data[i][j] = float(data[i][j])
   
    return data

def process(data):
    '''
    Function: process(data)
    Parameters: data, a list
    Returns: a new list, isolating just the data to analyze, and an
        int representing the number of factors to consider
    '''
    newdata = []
    
    # isolates data about headers specified in Comment 1
    for i in range(len(data)):
        attributes = [data[i][19], data[i][20], data[i][10], 
                      data[i][11], data[i][16], data[i][17],
                      data[i][6], data[i][7], data[i][18]]
        newdata.append(attributes)
                       
    return newdata, len(attributes)

def iterate(df, num_factors):
    np.random.seed(0)    
    y_data = df.pop(num_factors - 1)
    x_data = df
    
    x_lm = stats.add_constant(x_data)
    linreg = stats.OLS(y_data, x_lm).fit()
    
    while max(linreg.pvalues) > ALPHA:
        pval_data = pd.DataFrame(data = linreg.pvalues)
        entry = pval_data[0] == max(pval_data[0])
        max_pval_index = pval_data[0].index[entry].tolist()[0]
        x_data.pop(max_pval_index)
        x_lm = stats.add_constant(x_data)
        linreg = stats.OLS(y_data, x_lm).fit()
    
    return linreg

def predicted_ratings(df, data):
    rates = []
    for i in range(len(data)):
        rate = df.params["const"]
        for j in [0, 2, 4]:
            rate += df.params[j] * data[i][j]
        rates.append(rate)
    
    return rates

def graph(list1, list2, col, size):
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Actual vs. Predicted Ride Rating")
    for i in range(len(list1)):
        plt.plot(list1[i], list2[i], "o", color = col, ms = size)
        
def vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    
    return vif_data
        
def main():
    # extract and clean data from inital csv
    rawdata = openfile(FILENAME, CANCEL_COL, RATING_COL)
    newrawdata = floatize(rawdata, FLOAT_COLS)
    data = process(newrawdata)[0]
    
    num_factors = process(newrawdata)[1]
    
    df = pd.DataFrame(data = data)
    final_linreg = iterate(df, num_factors)
    
    print(final_linreg.summary())
    print(final_linreg.params.tolist())
    
    pred_ratings = predicted_ratings(final_linreg, data)
       
    vif_data = vif(df)
    print(vif_data)
    
    for i in range(len(data)):
        ratings.append(data[i][len(data[0]) - 1])
        
    x = np.arange(1, 5, 0.01)
    
    plt.figure(dpi = 1000)
    graph(ratings, pred_ratings, "black", 1)
    graph(x, x, "slategrey", 0.2)
    
    plt.show()
    
    '''
    Rating = 4.90 + 0.020 * Pickup Diff - 0.00537 * Response Time
    - 0.0600 * Ride Distance
    '''
main()