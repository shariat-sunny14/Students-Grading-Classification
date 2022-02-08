from flask import Flask, render_template, request, app
import pickle
import numpy as np
import pandas as pd
import sklearn
from flask import Response
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST', 'GET'])
def predictRoute():
    try:
        if request.json['data'] is not None:
            data = request.json['data']
            print('data is:     ', data)
            res = predict_log(data)
            print('result is        ', res)
            return Response(res)
    except ValueError:
        return Response("Value not found")
    except Exception as e:
        print('exception is   ', e)
        return Response(e)


def predict_log(dict_pred):                          # sandardScalar.sav StudentGradingmodelForPrediction.sav
    with open("sandardScalar.sav", 'rb') as f:
        scalar = pickle.load(f)

    with open("StudentGradingmodelForPrediction.sav", 'rb') as f:
        model = pickle.load(f)

    data_df = pd.DataFrame(dict_pred, index=[1, ])
    scaled_data = scalar.transform(data_df)
    predict = model.predict(scaled_data)
    if predict[0] == 0:
        result = 'Very Poor'
    elif predict[0] == 1:
        result = 'Poor'
    elif predict[0] == 2:
        result = 'Below Average'
    elif predict[0] == 3:
        result = 'Average'
    elif predict[0] == 4:
        result = 'Very Good'
    else:
        result = 'Excellent'

    return result

@app.route("/pridict", methods=['POST'])
def pridict():
    if request.method == "POST":
        cse_math_score = request.form['cse_math_score']
        eee_score = request.form['eee_score']
        cse_deploy_score = request.form['cse_deploy_score']
        math_score = request.form['math_score']

        prediction = predict_log(np.array([[cse_math_score, eee_score, cse_deploy_score, math_score]]))

        return render_template('index.html', prediction="Your Result is {}".format(prediction))
    else:
        return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
