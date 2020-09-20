import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Laundry Dataset')

st.header('Read data and convert column to upper letter')
df = pd.read_csv("Laundry_data.csv")
df.columns = df.columns.str.upper()
st.success('Done reading and converting upper letter')

st.write("Total rows and columns: ", df.shape)

st.header('EDA & Data Preprocessing')
st.subheader('Check Missing Data')

st.write(df.isna().sum())
st.write('Total null: ', df.isna().sum().sum())

df1 = df.copy()

st.subheader("Unites values and remove space between string")
df1['PANTS_COLOUR'] = df1['PANTS_COLOUR'].replace(['blue_jeans'],['blue'])
df1['RACE'] = df1['RACE'].str.strip()
df1['KIDS_CATEGORY'] = df1['KIDS_CATEGORY'].str.strip()
df1['PANTS_COLOUR'] = df1['PANTS_COLOUR'].str.strip()
st.write("Some columns were found to have values that need to be cleaned for consistency. For example, for the 'PANTS_COLOUR' column, the value of 'blue' is having white space that needs to be removed. Besides, 'blue_jeans' and 'blue' which indicate the same thing need to be united. White spaces in 'RACE' and 'KIDS_CATEGORY' are removed as well.")

st.subheader("Fill in KIDS_CATEGORY's null value and WITH_KIDS's null value")

df1.loc[df1['KIDS_CATEGORY'] == 'no_kids', 'WITH_KIDS'] = 'no' 
df1.loc[df1['KIDS_CATEGORY'] == 'toddler', 'WITH_KIDS'] = 'yes' 
df1.loc[df1['KIDS_CATEGORY'] == 'young', 'WITH_KIDS'] = 'yes' 
df1.loc[df1['KIDS_CATEGORY'] == 'baby', 'WITH_KIDS'] = 'yes'
df1.loc[df1['WITH_KIDS'] == 'no', 'KIDS_CATEGORY'] = 'no_kids'
st.write("We filled the null values for 'WITH_KIDS' and 'KIDS_CATEGORY' by comparing the two attributes.\
	For 'AGE_RANGE' column, null values were filled with mean.\
	Missing values for 'BODY_SIZE', 'BASKET_SIZE' and 'ATTIRE' columns were filled with mode. \
	Besides, we filled 'others' for missing values in 'WASH_ITEM' column.\
	Lastly, rows containing null values for 'WITH_KIDS', 'RACE' and 'GENDER' columns were dropped. \
	After data cleaning, there is no more missing values for any columns and there are 662 rows of data left.")

st.subheader("Fill in AGE_RANGE'S null value")
sns.boxplot(x=df['AGE_RANGE'])
st.pyplot()
df1['AGE_RANGE'] = df1['AGE_RANGE'].fillna(df1['AGE_RANGE'].mean())
df1['AGE_RANGE'] = df1['AGE_RANGE'].astype(int)
st.write("Since this boxplot is showing there is no outlier so the data is normally distributed and also because \
of this we can fill in the null value using mean.")

st.subheader("Fill in BODY_SIZE,BASKET_SIZE,ATTIRE null value using mode")
df1['BODY_SIZE'].fillna(df1['BODY_SIZE'].mode()[0], inplace=True)
df1['BASKET_SIZE'].fillna(df1['BASKET_SIZE'].mode()[0], inplace=True)
df1['ATTIRE'].fillna(df1['ATTIRE'].mode()[0], inplace=True)
st.write("Since these columns is categorical data so it is more suitable to use mode to fill in their null values.")

st.subheader("Fill in WASH_ITEM's null value using 'others'")
df1['WASH_ITEM'].fillna("others",inplace=True)
st.write("There are also few rows in the column 'WASH_ITEM' is null values, in stead of straight away drop the row, \
we choose to fill in the null with others because the variable for this column is 'cloths' and 'balnkets' so we pretend \
that the missing value is neither cloths nor blankets.")

st.subheader("Drop some rows that are containing null value and drop some unwanted columns")
df1.dropna(subset = ['WITH_KIDS','RACE','GENDER'], inplace=True)
df1 = df1.drop(columns=['NO','KIDS_CATEGORY','BASKET_COLOUR','SHIRT_COLOUR','SHIRT_TYPE','PANTS_COLOUR','PANTS_TYPE', 'WASHER_NO','DRYER_NO'])
st.write("Unwanted columns for instances, 'NO', 'KIDS_CATEGORY', 'BASKET_COLOUR', 'SHIRT_COLOUR', 'SHIRT_TYPE', 'PANTS_COLOUR' and 'PANTS_TYPE' were dropped")

df1["TIME"]= df1["TIME"].str.split(":", n = 1, expand = True) 
df1["TIME"] = pd.to_numeric(df1.TIME, errors='coerce')
labels=["Midnight","Dawn", "Morning", "Afternoon","Evening","Night"]
df1["PARTS_OF_DAY"] = pd.cut(df1["TIME"], bins=[-1,3,7,11,15,19,23], labels=labels)
df1 = df1.drop(columns='TIME')

st.subheader('Convert date to day')
df1["DATE"]= pd.to_datetime(df1["DATE"])
df1['DAY'] = df1['DATE'].dt.day_name()
st.write("Since there is only a date column but not a day column, so we will generate ourself the day column")

st.subheader('Binning')
df1['DAYS'] = df1['DATE'].apply(lambda x: x.weekday())
df1['WEEKEND'] = pd.cut(df1["DAYS"], bins=[-1,4,6], labels=[0,1])
df1 = df1.drop(columns=['DAYS', 'DATE'])
bins=[20,30,40,50,60]
df1['AGE_GROUP'] = pd.cut(df1['AGE_RANGE'], bins=bins)
df1['AGE_GROUP'] = df1['AGE_GROUP'].astype(str)
df1 = df1.drop(columns=['AGE_RANGE'])
st.write("Binning technique was applied on some columns and new columns were introduced. Through applying binning technique on 'TIME' column, a new column named 'PARTS_OF_DAY' was generated. Same technique was applied on 'DATE' column. First, the data was converted to days and a new column named 'DAY' was generated. Then, binning technique was applied on the 'DAY' column and a new column for 'WEEKEND' was created. Lastly, 'DATE' column was dropped. 'AGE_RANGE' column was another column that applied binning technique. After cutting the values into bins, a new column named 'AGE_GROUP' was added.")

st.subheader('Check again for the Missing Data and the final shape of our dataset')
st.write(df1.isna().sum())
st.write('Total null: ', df1.isna().sum().sum())
st.write("Total rows and columns: ", df1.shape)

st.subheader("Visualization")
select = st.selectbox("Which chart or plot you wan't to see", ["Choose here","Customers of which age group come more often to laundry shop?",
                                                       "Which days of the week have more customers?","Customers of which gender usually bring their kids to laundry shop?",
                                                       "What is the basket size that usually being used?",
                                                       "Which part of day does customer come more often?", "Which parts of the day for each days of the week did customers come more often?",
                                                        "Which race comes more often to laundry shop?"])

if select == "Customers of which age group come more often to laundry shop?":
    st.subheader("Q: Customers of which age group come more often to laundry shop?")
    sns.countplot(df1['AGE_GROUP'])
    st.pyplot()
    st.write("Majority of the customers of laundry shop aged between 40 and 50. On the other hand, customers aged between 20 to 30 seldom come to this laundry shop.")

elif select == "Which days of the week have more customers?":
    st.subheader("Q: Which days of the week have more customers?")
    plt.figure(figsize=(7,3))
    sns.countplot(df1['DAY'])
    st.pyplot()
    st.write("Customers usually come laundry shop on Saturday and Sunday. In contrary, Monday has the least customers.")

elif select == "Customers of which gender usually bring their kids to laundry shop?":
    st.subheader("Q: Customers of which gender usually bring their kids to laundry shop?")
    dfc2 = df1.groupby(['GENDER', 'WITH_KIDS'])['WITH_KIDS'].count().unstack('WITH_KIDS')
    dfc2.plot(kind='barh', stacked=True)
    st.pyplot()
    st.write("Female customers tend to bring their kids to laundry shop compared to male customers.")

elif select == "What is the basket size that usually being used?":
    st.subheader("Q: What is the basket size that usually being used?")
    sns.countplot(df1['BASKET_SIZE'])
    st.pyplot()
    st.write("Customers usually use big basket.")

elif select == "Which part of day does customer come more often?":
    st.subheader("Q: Which part of day does customer come more often?")
    sns.countplot(df1['PARTS_OF_DAY'])
    st.pyplot()
    st.write("Customers usually come at night.")

elif select == "Which parts of the day for each days of the week did customers come more often?":
    st.subheader("Q: Which parts of the day for each days of the week did customers come more often?")
    plt.figure(figsize=(15,10))
    sns.countplot(x="DAY", hue="PARTS_OF_DAY", data=df1)
    st.pyplot()
    st.write("Most customers come laundry shop on Sunday dawn, least customers come laundry shop on Monday. In addition, no customers come laundry shop on Monday midnight and dawn. Also, no customers come to laundry shop on Sunday evening and night.")

elif select == "Which race comes more often to laundry shop?":
    st.subheader("Q: Which race comes more often to laundry shop?")
    sns.countplot(df1['RACE'])
    st.pyplot()
    st.write("Foreigners less visit to laundry shop.")

st.header("Encoding")
st.subheader("Create dummies variables")
df2 = df1.copy()
df2 = df2[['PARTS_OF_DAY','RACE','GENDER','BODY_SIZE','WITH_KIDS','BASKET_SIZE','ATTIRE','WASH_ITEM','SPECTACLES','AGE_GROUP','WEEKEND']]

cat_vars=['PARTS_OF_DAY','RACE','GENDER','BODY_SIZE','WITH_KIDS','BASKET_SIZE','ATTIRE','WASH_ITEM','SPECTACLES','AGE_GROUP']

for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list=pd.get_dummies(df2[var],prefix=var)
    df3=df2.join(cat_list)
    df2=df3

df_vars=df2.columns.values.tolist()
to_keep=[i for i in df_vars if i not in cat_vars]

df_final=df2[to_keep]

st.write("Categorical data needs to be converted to numerical form for the use of \
machine learning algorithms. Dummy variables were created using Pandas \
get dummies approach. It is a popular method for treating categorical variables. It encodes the unique values in each column as a binary vector array \
which allows a machine learning algorithm to leverage the information found \
in a category without the confusion of ordinality.")

         

st.header("Association Rule Mining")
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import re
st.subheader("Generating association rules from frequent itemsets")
st.subheader("Items that appears (3/5) in dataset")
frequent_itemsets = apriori(df_final, min_support=0.6, use_colnames=True)
st.write(frequent_itemsets)
st.subheader("Rule generation and selection criteria")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
st.write(rules)
st.subheader("Count total number of antecedent")
rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
st.write(rules)
st.subheader("Choose rules with at least two antecedents, lift value more than 1.0, and confidence levl more than 0.8")
st.write(rules[(rules['antecedent_len'] >= 2) & (rules['lift'] > 1.0) & (rules['confidence'] > 0.8) ])

st.header("Clustering using KModes")
from kmodes.kmodes import KModes
### Cao initialization
st.write("K-modes clustering is best for dataset that contains categorical data exclusively. Majority columns of the laundry dataset was categorical data, hence, \
k-modes clustering is chosen to perform data cluster analysis for this project.")
cost = []
K = range(1,5)
for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(df_final)
    cost.append(kmode.cost_)
    
plt.plot(K, cost, 'bx-')
plt.xlabel('k clusters')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal k')
plt.show()
st.pyplot()

n = 2
km = KModes(n_clusters=n, init='Cao', n_init=11, verbose=1)
clusters = km.fit_predict(df_final)
kmodes = km.cluster_centroids_
shape = kmodes.shape

for i in range(shape[0]):
    st.info("\ncluster " + str(i) + ": ")
    cent = kmodes[i,:]
    for j in df_final.columns[np.nonzero(cent)]:
        st.write(j)

km = KModes(n_clusters=n, init = "Cao", n_init = 1, verbose=1)
cluster_labels = km.fit_predict(df_final)

df_new = df_final.copy()
df_new['Cluster']=km.labels_

chart = st.selectbox("Which chart you wan't to see?",
                     ["WEEKEND","PARTS_OF_DAY_Afternoon","PARTS_OF_DAY_Evening","PARTS_OF_DAY_Midnight","PARTS_OF_DAY_Morning","PARTS_OF_DAY_Night","RACE_chinese"
                      ,"RACE_foreigner","RACE_indian","RACE_malay","GENDER_female","GENDER_male","BODY_SIZE_fat","BODY_SIZE_moderate","BODY_SIZE_thin","WITH_KIDS_no"
                      ,"WITH_KIDS_yes","BASKET_SIZE_big","BASKET_SIZE_small","ATTIRE_casual","ATTIRE_formal","ATTIRE_traditional","WASH_ITEM_blankets","WASH_ITEM_clothes"
                      ,"WASH_ITEM_others","SPECTACLES_no","SPECTACLES_yes","AGE_GROUP_(20,30]","AGE_GROUP_(30,40]","AGE_GROUP_(40,50]","AGE_GROUP_(50,60]","Cluster"])

for col in df_new:
    if col == chart:
        plt.subplots(figsize = (15,5))
        sns.countplot(x='Cluster',hue=col, data = df_new)
        plt.show()
        st.pyplot()

st.header("Feature Selection")
from boruta import BorutaPy
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

X = df_final.drop(columns='WEEKEND')
y = df_final.WEEKEND
st.subheader("Apply random forest classifier and Boruta then show the result of top 10 and bottom 10")

rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
feat_selector = BorutaPy(rf, n_estimators="auto", random_state=10)

feat_selector.fit(X.values, y.values.ravel())
colnames = X.columns

boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])

boruta_score = boruta_score.sort_values("Score", ascending = False)

st.info('---------Top 10----------')
st.write(boruta_score.head(10))

st.info('---------Bottom 10----------')
st.write(boruta_score.tail(10))
st.write(" Features with the score lower \
than 0.30 were dropped.")

st.info("Boruta Plot")
sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score, kind = "bar", 
               height=14, aspect=1.9, palette='coolwarm')
plt.title("Boruta all Features")
st.pyplot()

st.header("Classification")
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
X = df_final.drop(columns=['WEEKEND','GENDER_female', 'BASKET_SIZE_small', 'AGE_GROUP_(50, 60]', 
                           'BASKET_SIZE_big', 'ATTIRE_traditional', 'AGE_GROUP_(20, 30]','RACE_foreigner'])
y = df_final.WEEKEND

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 10)

maes = []
for d in range(1,15): 
    rf = RandomForestClassifier(max_depth = d, random_state = 10)
    rf.fit(X_train, y_train)
    y_pred2 = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, y_pred2)
    maes.append(rf_mae)
st.write(maes)
plt.plot(range(1,15),maes) 
plt.show()
st.pyplot()

st.subheader("Classification model 1: Random Forest")
rf = RandomForestClassifier(max_depth=6, random_state=10)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

st.write("Accuracy on train set     : {:.3f}".format(rf.score(X_train, y_train)))
st.write("Accuracy on test set     : {:.3f}".format(rf.score(X_test, y_test)))
prob_RF = rf.predict_proba(X_test)
prob_RF = prob_RF[:,1]

auc_RF = roc_auc_score(y_test,prob_RF)
st.write("AUC : %.2f " % auc_RF)

confusion_majority=confusion_matrix(y_test, y_pred)

target_names = ['class 0', 'class 1']
st.write('Classification report:\n',classification_report(y_test, y_pred, target_names=target_names))
st.write('Mjority classifier Confusion Matrix\n', confusion_majority)

st.write('**********************')
st.write('Mjority TN = ', confusion_majority[0][0])
st.write('Mjority FP = ', confusion_majority[0][1])
st.write('Mjority FN = ', confusion_majority[1][0])
st.write('Mjority TP = ', confusion_majority[1][1])
st.write('**********************')

st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))


fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, prob_RF) 

st.subheader("Classification model 2: Gradient Boosting")
gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.1, max_features=2, max_depth = 6, random_state = 10)
gb.fit(X_train, y_train)
# print("Learning rate: ", learning_rate)
st.write("Accuracy on train set: {0:.3f}".format(gb.score(X_train, y_train)))
st.write("Accuracy on test set: {0:.3f}".format(gb.score(X_test, y_test)))
st.write()
prob_GC = gb.predict_proba(X_test)
prob_GC = prob_GC[:,1]

auc_GC = roc_auc_score(y_test,prob_GC)
st.write("AUC : %.2f " % auc_GC)

confusion_majority=confusion_matrix(y_test, y_pred)

target_names = ['class 0', 'class 1']
st.write('Classification report:\n',classification_report(y_test, y_pred, target_names=target_names))
st.write('Mjority classifier Confusion Matrix\n', confusion_majority)

st.write('********')
st.write('Mjority TN = ', confusion_majority[0][0])
st.write('Mjority FP = ', confusion_majority[0][1])
st.write('Mjority FN = ', confusion_majority[1][0])
st.write('Mjority TP = ', confusion_majority[1][1])
st.write('********')

st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))


fpr_GC, tpr_GC, thresholds_GC = roc_curve(y_test, prob_GC)

st.header("Receiver Operating Characteristic (ROC) Curve")
# Provide model accuracy, confusion matrix, <TN,TP,FP,FN>, Precision, Recall, F1, Accuracy

#RF
fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test, prob_RF) 
#GC
fpr_GC, tpr_GC, thresholds_GC = roc_curve(y_test, prob_GC) 


plt.plot(fpr_RF, tpr_RF, color='blue', label='RF')  
plt.plot(fpr_GC, tpr_GC, color='red', label='GC')  

plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
st.pyplot()
