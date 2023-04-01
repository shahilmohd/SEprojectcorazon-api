from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
print("Model loaded!")

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    age = int(request.args['age'])
    sex = int(request.args['sex'])
    cp = int(request.args['cp'])
    bp = int(request.args['bp'])
    chol = int(request.args['chol'])
    sugar = int(request.args['sugar'])
    ecg = int(request.args['ecg'])
    maxH = int(request.args['maxH'])
    exST = int(request.args['exST'])
    stdep = float(request.args['stdep'])
    slope = int(request.args['slope'])
    ca = int(request.args['ca'])
    thal = int(request.args['thal'])

    numeric_args = ['age', 'sex', 'cp', 'bp', 'chol', 'sugar',
                    'ecg', 'maxH', 'exST', 'stdep', 'slope', 'ca', 'thal']
    numeric_values = [float(request.args[arg]) for arg in numeric_args]
    # print(f"Input values: age={age}, sex={sex}, cp={cp}, maxH={maxH}, exST={exST}, ecg={ecg}, chol={chol}, bp={bp}, sugar={sugar}, stdep={stdep}, ")
    pred_binary = model.predict(np.array(
        [age, sex, cp, bp, chol, sugar, ecg, maxH, exST, stdep, slope, ca, thal]).reshape(-1, 13))
    pred = max(model.predict_proba(np.array(
        [age, sex, cp, bp, chol, sugar, ecg, maxH, exST, stdep, slope, ca, thal]).reshape(-1, 13)))

    return jsonify(percentage=str(pred), type=str(pred_binary))


if __name__ == "__main__":
    app.run(debug=True)


# - Test Cases
# [Positive]
# http://127.0.0.1:5000/?age=59&sex=1&cp=1&bp=140&chol=221&sugar=0&ecg=1&maxH=164&exST=1&stdep=0.0&slope=2&ca=0&thal=2
# [Positive]
# http://127.0.0.1:5000/?age=47&sex=1&cp=1&bp=140&chol=203&sugar=1&ecg=1&maxH=118&exST=0&stdep=0&slope=9&ca=0&thal=9
# [Negative]
# http://127.0.0.1:5000/?age=70&sex=1&cp=0&bp=145&chol=173&sugar=0&ecg=0&maxH=102&exST=1&stdep=1.0&slope=9&ca=0&thal=9
# [Positive]
# http://127.0.0.1:5000/?age=47&sex=1&cp=0&bp=123&chol=245&sugar=1&ecg=1&maxH=155&exST=0&stdep=2.3&slope=9&ca=0&thal=9

# Link to the deployed model: https://seproject-corazon-api.onrender.com/?age=47&sex=1&cp=0&bp=123&chol=245&sugar=1&ecg=1&maxH=155&exST=0&stdep=2.3&slope=9&ca=0&thal=9
