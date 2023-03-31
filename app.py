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
	maxH = int(request.args['maxH'])
	exST = int(request.args['exST'])
	ecg = int(request.args['ecg'])
	chol = int(request.args['chol'])
	bp = int(request.args['bp'])
	sugar = int(request.args['sugar'])
	stdep = float(request.args['stdep'])
	numeric_args = ['age', 'sex', 'cp', 'bp', 'chol', 'sugar', 'ecg', 'maxH', 'exST', 'stdep']
	numeric_values = [float(request.args[arg]) for arg in numeric_args]
	print(f"Input values: age={age}, sex={sex}, cp={cp}, maxH={maxH}, exST={exST}, ecg={ecg}, chol={chol}, bp={bp}, sugar={sugar}, stdep={stdep}")
	pred_binary = model.predict(np.array([ age, sex, cp, bp, chol, sugar, ecg, maxH, exST, stdep ]).reshape(-1, 10))
	pred = max(model.predict_proba(np.array([ age, sex, cp, bp, chol, sugar, ecg, maxH, exST, stdep ]).reshape(-1, 10)))

	return jsonify(percentage = str(pred), type = str(pred_binary))

if __name__ == "__main__":
	app.run(debug=True)

# http://127.0.0.1:5000/?age=47&sex=1&cp=0&bp=110&chol=275&sugar=0&ecg=0&maxH=118&exST=1&stdep=1.0
