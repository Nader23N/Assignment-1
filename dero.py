from flask import Flask ,request, jsonify , render_template
import pandas as pd
import Algorithms





df = pd.read_csv("Dataset.csv")
app = Flask(__name__)



@app.route("/",methods=["GET"])
def calc():
	return "<form action='/int' method='post'>Gender<input type='text' name='gender'>SeniorCitizen<input type='text' name='SeniorCitizen'>Partner<input type='text' name='Partner'>Dependents<input type='text' name='Dependents'>Tenure<input type='text' name='tenure'>Payment Method<input type='text' name='PaymentMethod'>Monthly Charges<input type='text' name='MonthlyCharges'>Total Charges<input type='text' name='TotalCharges'>Churn<input type='text' name='Churn'><input type='submit'></form>"
@app.route('/int',methods=["post"])
def ab():
        gender = int(request.form.get('gender'))
        SeniorCitizen = int(request.form.get('SeniorCitizen'))
        Partner = int(request.form.get('Partner'))
        Dependents = int(request.form.get('Dependents'))
        tenure = int(request.form.get('tenure'))
        PaymentMethod = int(request.form.get('PaymentMethod'))
        MonthlyCharges = int(request.form.get('MonthlyCharges'))
        TotalCharges = int(request.form.get('TotalCharges'))
        Churn = int(request.form.get('Churn'))
        cols = [{"gender" : gender ,"SeniorCitizen" : SeniorCitizen ," Partner" :Partner ,"Dependents":Dependents,"tenure" :tenure,"PaymentMethod" :PaymentMethod ,"MonthlyCharges" :MonthlyCharges ,"TotalCharges":TotalCharges ,"Churn" :Churn}]
        df1 = pd.DataFrame(cols)
        return  "<form action='/prep' method='get'><input type='submit' value='perprocessing'></form><form action='/knn' method='get'><input type='submit' value='knn'></form><form action='/LogR' method='get'><input type='submit' value='Logistic'></form><form action='/svc' method='get'><input type='submit' value='SVC'></form></form><form action='/Decisiontree' method='get'><input type='submit' value='Decisiontree'></form>"

	
@app.route('/prep')
def prep():
	return Algorithms.perprocess(df).to_html()

@app.route('/knn')
def knn_model():
	return Algorithms.KNN_Model(df)


@app.route('/LogR')
def linreg():
	return Algorithms.Linreg_Model(df)


@app.route('/svc')
def svc_model():
        return Algorithms.SVC_Model(df)


@app.route('/Decisiontree')
def dectree():
        return Algorithms.Decisiontree_Model(df)

if __name__ == '__main__':
	app.run(debug = True)
