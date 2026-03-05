from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([1,2,3,4,5]).reshape(-1, 1)  
y = np.array([2,3,5,7,11])
 
model = LinearRegression()  
model.fit(X, y)


prediction = model.predict([[6]])    
print(f"Predicted salary for 6 years of experience: {prediction[0]}")   