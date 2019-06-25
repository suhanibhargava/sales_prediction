# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:07:35 2019

@author: hp1
"""

import pandas as pd

def predict_sales(item_id,outlet_id):
    dataframe=pd.read_csv('predictions.csv')

    item_id1=[]
    item_id1.append(dataframe['Item_Identifier'])
    item=list(item_id1[0])

    outlet_id1=[]
    outlet_id1.append(dataframe['Outlet_Identifier'])
    outlet=list(outlet_id1[0])
    df = pd.read_csv("predictions.csv")
    if (item_id in item) and outlet_id  in list(df.loc[df['Item_Identifier']== item_id]['Outlet_Identifier']):
        
        new_df =df.loc[(df['Item_Identifier'] == item_id)& (df['Outlet_Identifier'] == outlet_id)]
        pred = new_df['Item_Outlet_Sales']
        return [pred.values[0]]

#    if(item_id in item) and (outlet_id in outlet):
#         df = pd.read_csv("predictions.csv")
#         new_df =df.loc[(df['Item_Identifier'] == item_id)& (df['Outlet_Identifier'] == outlet_id)]
#         pred = new_df['Item_Outlet_Sales']
#         return [pred.values[0]]


#    elif (item_id in item):
#        df = pd.read_csv("predictions.csv")
#        if (outlet_id not in list(df.loc[df['Item_Identifier']=='FDW58']['Outlet_Identifier']) ):
#            return "Invalid"
    else:
         return "Invalid"


predict_sales("FDW58","OUT010")


# df3 = dataframe.loc[dataframe["Item_Identifier"]=='FDW58']
