
def predict_sales(item_id,outlet_id):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.externals import joblib


    test = pd.read_csv('Test.csv')
    train=pd.read_csv('Train.csv')
    train['source']='train'
    test['source']='test'
    data = pd.concat([train, test],ignore_index=True)
    #check the missing values
    data.apply(lambda x: sum(x.isnull()))


    #handle the missing values
    data['Item_Weight'].fillna(data['Item_Weight'].mean(),inplace=True)
    data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0],inplace=True)




    data.apply(lambda x: len(x.unique()))


    #Years:
    data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
    data['Outlet_Years'].describe()



    #visibility
    data['Item_Visibility']=data['Item_Visibility'].replace(0,data['Item_Visibility'].mean())

    #combine into categories
    data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])

    data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food',
                                                                 'NC':'Non-Consumable',
                                                                 'DR':'Drinks'})
    data['Item_Type_Combined'].value_counts()

    data['Item_Fat_Content']=data['Item_Fat_Content'].replace('LF',"Low Fat")
    data['Item_Fat_Content']=data['Item_Fat_Content'].replace('low fat',"Low Fat")
    data['Item_Fat_Content']=data['Item_Fat_Content'].replace('reg',"Regular")
    data['Item_Fat_Content'].value_counts()

    #Mark non-consumables as separate category in low_fat:
    data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
    data['Item_Fat_Content'].value_counts()


    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    #New variable for outlet
    data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
    var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
    le = LabelEncoder()
    for i in var_mod:
        data[i] = le.fit_transform(data[i])

    #One Hot Coding:
    data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',
                                  'Item_Type_Combined','Outlet'])
    data.dtypes

    data[['Item_Fat_Content_0','Item_Fat_Content_1','Item_Fat_Content_2']].head(10)

    #Drop the columns which have been converted to different types:
    data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

    #Divide into test and train:
    train = data.loc[data['source']=="train"]
    test = data.loc[data['source']=="test"]

    #Drop unnecessary columns:
    test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
    train.drop(['source'],axis=1,inplace=True)

    #Export files as modified versions:
    train.to_csv("train_modified.csv",index=False)
    test.to_csv("test_modified.csv",index=False)

    #Mean based:
    mean_sales = train['Item_Outlet_Sales'].mean()

    #Define a dataframe with IDs for submission:
    base1 = test[['Item_Identifier','Outlet_Identifier']]
    base1['Item_Outlet_Sales'] = mean_sales

    #Export submission file
    base1.to_csv("alg0.csv",index=False)

    #Define target and ID columns:
    target = 'Item_Outlet_Sales'
    IDcol = ['Item_Identifier','Outlet_Identifier']
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error
    #def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #    #Fit the algorithm on the data
    #    alg.fit(dtrain[predictors], dtrain[target])
    #
    #    #Predict training set:
    #    dtrain_predictions = alg.predict(dtrain[predictors])
    #
    #    #Perform cross-validation:
    #    cv_score = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='neg_mean_squared_error')
    #    cv_score = np.sqrt(np.abs(cv_score))
    #
    #    #Print model report:
    #    print ("\nModel Report")
    #    print ("RMSE : %.4g" % np.sqrt(mean_squared_error(dtrain[target].values, dtrain_predictions)))
    #    print ("CV Score : Mean - %.4g | Std - %.4g |" % (np.mean(cv_score),np.std(cv_score)))
    #
    #    #Predict on testing data:
    #    dtest[target] = alg.predict(dtest[predictors])
    #
    #    #Export submission file:
    #    IDcol.append(target)
    #    submission = pd.DataFrame({ x: dtest[x] for x in IDcol})
    #    submission.to_csv(filename, index=False)


    #decision tree model
    from sklearn.tree import DecisionTreeRegressor
    predictors = [x for x in train.columns if x not in [target]+IDcol]
    alg0 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
    alg0.fit(train[predictors], train[target])
    train_predictions0 = alg0.predict(train[predictors])
    cv_score0 = np.mean(cross_val_score(alg0,train[predictors], train[target], cv=20))

    test[target] = alg0.predict(test[predictors])

    IDcol.append(target)
    submission = pd.DataFrame({x : test[x] for x in IDcol})
    submission.to_csv("predictions.csv", index = False)

    rmse0 = np.sqrt(mean_squared_error(train[target].values, train_predictions0))
    #modelfit(alg3, train, test, predictors, target, IDcol, 'alg3.csv')
    #coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
    #coef3.plot(kind='bar', title='Feature Importances')
    joblib.dump(alg0 , 'model.pkl')
    model = joblib.load('model.pkl')


    #test2 = test.sort_values('Item_Outlet_Sales' , ascending = False)
    #top5 = test2.head(5)
    #top5.plot.bar(x ='Item_Identifier' , y = 'Item_Outlet_Sales', rot =0  )

    

    # 1- item-id 2- out-id
    df = pd.read_csv("predictions.csv")
    new_df =df.loc[(df['Item_Identifier'] == item_id)& (df['Outlet_Identifier'] == outlet_id)]
    pred = new_df['Item_Outlet_Sales']
    if(pred.values[0]==0):
        return "invalid"
    else:
        pred.values[0]
    
   
    #from sklearn.ensemble import RandomForestRegressor
    #alg1 = RandomForestRegressor(max_depth=2,random_state=0, n_estimators=100)
    #alg1.fit(train[predictors], train[target])
    #train_predictions1 = alg1.predict(train[predictors])
    #cv_score1 = np.mean(cross_val_score(alg1,train[predictors], train[target], cv=20))
    #rmse1 = np.sqrt(mean_squared_error(train[target].values, train_predictions1))
    #
    #
    #
    #
    #from sklearn import linear_model
    #alg2 = linear_model.Lasso(alpha=0.1)
    #alg2.fit(train[predictors], train[target])
    #train_predictions2 = alg2.predict(train[predictors])
    #cv_score2 = np.mean(cross_val_score(alg2,train[predictors], train[target], cv=20))
    #rmse2 = np.sqrt(mean_squared_error(train[target].values, train_predictions2))
