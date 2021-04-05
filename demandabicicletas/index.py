import os
from flask import Flask, render_template, request
import numpy as np 
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route('/')
def inicio():
    return render_template('home4.html')

@app.route('/predecir_demanda', methods=['GET','POST'])
def modelo():
    # ['hr', 'temp', 'atemp', 'hum', 'hour_type', 'moving_avg_cnt', 'moving_avg_temp]
    if request.method =='POST':
        medida1 =  request.form.get('medida1', type = float)    #hora
        medida2 =  request.form.get('medida2', type = float)    #temp
        medida3 =  request.form.get('medida3', type = float)    #atemp
        medida4 =  request.form.get('medida4', type = float)    #hum
        medida5 =  request.form.get('medida5', type = float)    #hour_type
        medida6 =  request.form.get('medida6', type = float)    #moving_avg_cnt
        medida7 =  request.form.get('medida7', type = float)    #moving_avg_temp
       
        medidas = np.array([medida1, medida2, medida3, medida4, medida5, medida6, medida7]).reshape(1,-1)
        scaler = MinMaxScaler()
        medidas_escaladas = scaler.fit_transform(medidas)
        clf1 = joblib.load('trained_model_3.pkl')
        clf2 = joblib.load('full_trained_model_3.pkl')

        prediccion = clf1.predict(medidas_escaladas)
        prediccion2 = clf2.predict(medidas_escaladas)
        return render_template('home4.html', prediccion = prediccion, prediccion2 = prediccion2)
    else:
        return render_template('home4.html')

if __name__ == "__main__":
    app.run('127.0.0.1',5000,debug=True)

