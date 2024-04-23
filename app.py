from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)


scaler = pickle.load(open("Model/scaler.pkl","rb"))
model = pickle.load(open("Model/random.pkl","rb"))

# Route for homepage

@app.route('/')
def house_price():
    return render_template("House_price_prediction.html")

# Route for prediction

@app.route('/predict',methods=["POST","GET"])
def predict_price():
    result=""

    bedrooms=int(request.form.get("bedrooms"))
    bathrooms=float(request.form.get("bathrooms" ))
    sqft_living=np.log(int(request.form.get("sqft_living")))
    sqft_lot=np.log(int(request.form.get("sqft_lot")))
    floors=float(request.form.get("floors"))
    waterfront=str(request.form.get("waterfront"))
    view= np.cbrt(int(request.form.get("view")))
    condition= int(request.form.get("condition"))
    sqft_basement= np.cbrt(int(request.form.get("sqft_basement")))
    yr_built= int(request.form.get("yr_built"))
    yr_renovated= int(request.form.get("yr_renovated" ))
    statezip= int(request.form.get("statezip" ))

    if waterfront=="Yes":
        waterfront=1
    elif waterfront=="No":
        waterfront=0        
    new_data= scaler.transform(np.array([[bedrooms, bathrooms, sqft_living,sqft_lot,
                                        floors, waterfront, view, condition, sqft_basement,
                                        yr_built, yr_renovated, statezip]]))
    
    prediction = model.predict(new_data)
    results=np.exp(prediction)

    return render_template("House_price_prediction.html",result=results)
        


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)


['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'sqft_basement', 'yr_built',
       'yr_renovated', 'statezip']