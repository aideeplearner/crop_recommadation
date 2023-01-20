from flask import Flask,render_template,url_for,request
import joblib
import numpy as np
#N	P	K	temperature	humidity	ph	rainfall	label
model=joblib.load('sivabasha_crop1')

app=Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')
    
@app.route("/predict",methods=["POST"])
def predict():
    features=[x for x in request.form.values()]
    final_features=[np.array(features)]
    prediction=model.predict(final_features)
    output=prediction[0]
    return render_template("home.html",prediction_text="the crop recommaded for you is {}".format(output))
    
if __name__=="__main__":
    app.run(debug=True)