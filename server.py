from flask import Flask, request, render_template, url_for
from result_valid import predict_sales
import pandas as pd

app = Flask(__name__)

@app.route("/main")
def home():
    dataframe=pd.read_csv('predictions.csv')
    item_id1=[]
    item_id1.append(dataframe['Item_Identifier'])
    item=list(item_id1[0])

    outlet_id1=[]
    outlet_id1.append(dataframe['Outlet_Identifier'])
    outlet=list(outlet_id1[0])
    return render_template("layout.html", item_list = item, outlet_list=outlet)

@app.route("/result",methods=["POST"])
def output():
    form_data = request.form
    status = predict_sales(form_data["item_id"], form_data["outlet_id"])
    return render_template("response.html",status=status)

if __name__ == "__main__":
    app.run(debug=True)
