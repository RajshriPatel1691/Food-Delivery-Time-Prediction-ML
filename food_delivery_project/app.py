from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("delivery_model.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=["POST"])
def predict():

    distance=float(request.form["distance"])
    prep=float(request.form["prep"])
    exp=float(request.form["exp"])

    total = distance + prep

    # Weather Encoding
    weather=request.form["weather"]

    foggy=1 if weather=="Foggy" else 0
    rainy=1 if weather=="Rainy" else 0
    snowy=1 if weather=="Snowy" else 0
    windy=1 if weather=="Windy" else 0

    # Traffic
    traffic=request.form["traffic"]

    low=1 if traffic=="Low" else 0
    medium=1 if traffic=="Medium" else 0

    # Time of day
    time=request.form["time"]

    evening=1 if time=="Evening" else 0
    morning=1 if time=="Morning" else 0
    night=1 if time=="Night" else 0

    # Vehicle
    vehicle=request.form["vehicle"]

    car=1 if vehicle=="Car" else 0
    scooter=1 if vehicle=="Scooter" else 0

    features=[distance,prep,exp,total,
              foggy,rainy,snowy,windy,
              low,medium,
              evening,morning,night,
              car,scooter]

    final=np.array(features).reshape(1,-1)

    prediction=model.predict(final)

    return render_template("index.html",
    prediction_text="Estimated Delivery Time: {} minutes".format(round(prediction[0],2)))

if __name__=="__main__":
    app.run(debug=True)