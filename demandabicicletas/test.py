import joblib
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


clf1 = joblib.load('trained_model_3.pkl')
clf2 = joblib.load('full_trained_model_3.pkl')

medidas = np.array([[1,2,0.,0.,1.,0.,0]])
scaler = MinMaxScaler()
medidas_scaladas = scaler.fit_transform(medidas)
prediccion = clf1.predict(medidas_scaladas)
prediccion2 = clf2.predict(medidas_scaladas)

print("La predicción es:", prediccion, prediccion2)