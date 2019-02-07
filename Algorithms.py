from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
encs = {}
le = LabelEncoder()
sc = StandardScaler()




def perprocess(df):
        #Feature Engineering
	cols = ["gender","SeniorCitizen","Partner","Dependents","tenure","PaymentMethod","MonthlyCharges","TotalCharges","Churn"]
	df=df[cols]
	
	senior_mode = df['SeniorCitizen'].mode()[0]
	tenure_mean = df['tenure'].mean()
	fills = {'SeniorCitizen':senior_mode ,'tenure':tenure_mean }
	df.fillna(fills , inplace = True)
        #scale
	df['tenure'] = sc.fit_transform(df['tenure'].values.reshape(-1,1))
	df['MonthlyCharges'] = sc.fit_transform(df['MonthlyCharges'].values.reshape(-1,1))
	df['TotalCharges'] = sc.fit_transform(df['TotalCharges'].values.reshape(-1,1))

	return  labelenc(df)
        	
    
def labelenc(df):
        
        for col in df.columns:
            if df[col].dtype == "object":
                encs[col] = LabelEncoder()
                df[col]   =encs[col].fit_transform(df[col])
        return df


def train(df):
        X = df.drop("Churn",axis=1)
        y = df["Churn"]
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        return  train_test_split(X, y, test_size = 0.25, random_state = 0)

def KNN_Model(df):
        df=perprocess(df)
        X_train, X_test, y_train, y_test = train(df)
        knn = KNeighborsClassifier(n_neighbors= 5 )       
        knn.fit(X_train,y_train) 
        return  "Score =  " + str(knn.score(X_test,y_test))


def SVC_Model(df):
        df=perprocess(df)
        X_train, X_test, y_train, y_test = train(df)
        svc = SVC()
        svc.fit(X_train,y_train)
        return "Score =  " + str(svc.score(X_test,y_test))



def Linreg_Model(df):
        df=perprocess(df)
        X_train, X_test, y_train, y_test = train(df)
        lr = LogisticRegression(random_state = 0)
        lr.fit(X_train,y_train)
        return "Score =  " + str(lr.score(X_test,y_test))


def Decisiontree_Model(df):
         df=perprocess(df)
         X_train, X_test, y_train, y_test = train(df)
         clf = DecisionTreeClassifier(random_state=0)
         clf.fit(X_train,y_train)
         pr1 = str(clf.predict(X_test))
         cm,cm1=confusion_matrix(y_test,pr1).tolist()
         return  "Score =  " + str(clf.score(X_test,y_test)) + " and  " + "Predict  = " + str(cm)



       
        
        
        
        
        




