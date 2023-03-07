import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Reading the file from the CSV
warnings.simplefilter('ignore')
df = pd.read_csv("Iris.csv")

# Here the species is the Dependent variable
y = df["Species"]

# Here get_dummies are used because the species values are of string data type in order to convert it into another format this function is used

le = LabelEncoder()
y = le.fit_transform(y)

# Seperating the x from the y from the iris dataset

x = df.drop(df.columns[[0,5]], axis = 1)

# Histogram Representation

df.hist(color="blue", figsize=(10,7))
plt.show()

# Here the axis can take on two values the value 0 represents row and 2 represents column

x_Train, x_Test, y_Train, y_Test = train_test_split(x, y, test_size=0.2, random_state=0)


# The data needs to be split into 2 parts one for testing and the other for learning so sklearn is used
# The test_size represents the size of the data to be tested
# The random_state represents which particular data which is a seed to be used for testing if not specified the output will be different for every run

lr = LinearRegression()
lr.fit(x_Train,y_Train)
y_pred = lr.predict(x_Test)

# r2 score is user to display how well the regression line is fit in the data

plt.scatter(y_Test, y_pred)

# Add the regression line
plt.plot(y_Test, y_Test, color='red')

# Set the labels and title
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression Results')
plt.show()

r2 = r2_score(y_Test, y_pred)
print(r2)

# Using decisiontreeclassifier Model
dt = DecisionTreeClassifier()
dt.fit(x_Train, y_Train)

prediction = dt.predict(x_Test)
acc = accuracy_score(y_Test, prediction)
print(acc)

print("Enter the values in the correct order for the prediction")
a = float(input("Sepal Length"))
b = float(input("Sepal_width"))
c = float(input("petal_Length"))
d = float(input("petal_Width"))
prediction1 = dt.predict([[a,b,c,d]])

if(prediction1[0] == 0):
    print("Iris-setosa")
elif(prediction1[0] == 1):
    print("Iris-versicolor")
elif(prediction1[0] == 2):
    print("Iris-virginica")


# 5.1,3.5,1.4,0.2
# 5.7,2.8,4.5,1.3
# 6.7,2.5,5.8,1.8