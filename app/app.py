from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import pdb
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import importlib



app = Flask(__name__)

clf = joblib.load('pickle_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/project1')
def project1():
    return render_template('index2.html')

@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route('/result', methods=['POST'])
def predict():
    #bat = pd.read_csv('./data/bat.csv')
    # Features and Labels
    #X = bat.loc[:, bat.columns != 'PEAK']
    #y = bat.loc[:, bat.columns == 'PEAK']

    # Oversampling
    #os = SMOTE(random_state = 0)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    #os_X, os_y = os.fit_sample(X_train,y_train)
    #columns = X.columns
    #os_X = pd.DataFrame(data = os_X, columns = columns)
    #os_y = pd.DataFrame(data = os_y,columns = ['PEAK'])

    # Logistic Regreesion
    #log_fit = LogisticRegression()
    #log_fit.fit(os_X, os_y)
    #log_fit.score(X_test, y_test)

    OPS = float(request.form['OPS'])
    Games_Played = int(request.form['Games_Played'])
    Salary = float(request.form['Salary'])
    Age = int(request.form['Age'])
    Weight_in_pounds = float(request.form['Weight in pounds'])
    Height_in_inches = float(request.form['Height in inches'])
    Bats_Both = int(request.form['Bats Both'])
    Bats_Left = int(request.form['Bats Left'])
    Bats_Right = int(request.form['Bats Right'])
    Throws_Left = int(request.form['Throws Left'])
    Throws_Right = int(request.form['Throws Right'])

    input_data = [OPS,Games_Played,Salary,Age,Weight_in_pounds,Height_in_inches,Bats_Both,Bats_Left,Bats_Right,Throws_Left,Throws_Right]

    data = np.array(input_data).reshape(1, -1)
    label = {0: 'Wow, this player seems before his peak!!!', 1: 'Unfortunately, this player seems after his peak'}
    
    my_prediction  = clf.predict(data)[0]
    resfinal = label[my_prediction]
        
    if resfinal == 'Wow, this player seems before his peak!!!':
    	    return render_template('resultsalary1.html', prediction = resfinal)
    else:
        	return render_template('resultsalary.html', prediction = resfinal)


@app.route('/project2')
def project2():
    return render_template('index1.html')

@app.route('/result2', methods=['POST'])
def predict2():
    df_x = pd.read_csv('./data/X_train.csv')
    df_y = pd.read_csv('./data/y_train.csv')
    #bat = pd.read_csv('./data/bat.csv')
    # Features and Labels
    #X = bat.loc[:, bat.columns != 'PEAK']
    #y = bat.loc[:, bat.columns == 'PEAK']
    clf = RandomForestClassifier(n_estimators= 100, max_depth = 19, max_features = 5)
    clf.fit(df_x, df_y)
    if request.method == 'POST':

        Years_Played = float(request.form['Years_Played'])
        H = float(request.form['H'])
        BB = float(request.form['BB'])
        HR = float(request.form['HR'])
        OBP = float(request.form['OBP'])
        RBI = float(request.form['RBI'])
        R = float(request.form['R'])
        SB = float(request.form['SB'])
        B2B = float(request.form['2B'])
        B3B = float(request.form['3B'])
        AB = float(request.form['AB'])
        SO = float(request.form['SO'])
        DPf = float(request.form['DPf'])
        Af = float(request.form['Af'])
        Ef = float(request.form['Ef'])
        YSLS = float(request.form['YSLS'])
        G_all = float(request.form['G_all'])
        debut_age = float(request.form['debut_age'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        Years_Played = float(request.form['Years_Played'])
    

        #SlugPercent = float(request.form['SlugPercent'])   
        SlugPercent = 0.5
        AVE = float(request.form['AVE'])
        OPS = float(request.form['OPS'])




        data = [Years_Played, H, BB, HR, AVE, OBP, SlugPercent, OPS, RBI, R, SB, B2B, B3B, AB, SO, DPf, Af, Ef, YSLS, G_all, debut_age, weight, height]

        data = np.array(data).reshape(1, -1)

        #data = pd.DataFrame(data)

        my_prediction = clf.predict(data)
    
    if my_prediction == 0:
        return render_template('result0.html', prediction = my_prediction)
    elif my_prediction == 1:
        return render_template('result1.html', prediction = my_prediction)
    else:
        return render_template('result2.html', prediction = my_prediction)



######################################

@app.route('/table1')
def table1():
    return render_template('indextable.html')






if __name__ == '__main__':
    app.run(debug=True)
