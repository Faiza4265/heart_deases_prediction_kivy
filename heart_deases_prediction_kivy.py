import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from joblib import load
import numpy as np

heart_data = pd.read_csv('heart.csv')
# Split the data into features and target variable
X = heart_data.drop('target', axis=1)
y = heart_data['target']

# Convert categorical features to numerical using one-hot encoding
ohe = OneHotEncoder(sparse=False)
X_encoded = pd.DataFrame(ohe.fit_transform(X[['cp', 'thal', 'slope']]))
X_encoded.columns = ohe.get_feature_names(['cp', 'thal', 'slope'])
X = X.drop(['cp', 'thal', 'slope'], axis=1)
X = pd.concat([X, X_encoded], axis=1)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the decision tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Evaluate the model
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Save the model
dump(dt, 'heart_disease_model.joblib')



Builder.load_string("""
<MyLayout>:
    orientation: 'vertical'
    Label:
        text: 'Enter the following details:'
    BoxLayout:
        orientation: 'horizontal'
        Label:
            text: 'Age'
        TextInput:
            id: age
            multiline: False
            input_filter: 'int'
    BoxLayout:
        orientation: 'horizontal'
        Label:
            text: 'Gender (0 for female, 1 for male)'
        TextInput:
            id: gender
            multiline: False
            input_filter: 'int'
    BoxLayout:
        orientation: 'horizontal'
        Label:
            text: 'Chest Pain Type (0-3)'
        TextInput:
            id: cp
            multiline: False
            input_filter: 'int'
    BoxLayout:
        orientation: 'horizontal'
        Label:
            text: 'Resting Blood Pressure'
        TextInput:
            id: trestbps
            multiline: False
            input_filter: 'int'
    BoxLayout:
        orientation: 'horizontal'
        Label:
            text: 'Fasting Blood Sugar (0 if <= 120 mg/dl, 1 if > 120 mg/dl)'
        TextInput:
            id: fbs
            multiline: False
            input_filter: 'int'
    BoxLayout:
        orientation: 'horizontal'
        Label:
            text: 'Resting ECG (0-2)'
        TextInput:
            id: restecg
            multiline: False
            input_filter: 'int'
    BoxLayout:
        orientation: 'horizontal'
        Label:
            text: 'Maximum Heart Rate Achieved'
        TextInput:
            id: thalach
            multiline: False
            input_filter: 'int'
    BoxLayout:
        orientation: 'horizontal'
        Label:
            text: 'Exercise Induced Angina (0 for no, 1 for yes)'
        TextInput:
            id: exang
            multiline: False
            input_filter: 'int'
    BoxLayout:
        orientation: 'horizontal'
        Label:
            text: 'ST Depression Induced by Exercise Relative to Rest'
        TextInput:
            id: oldpeak
            multiline: False
            input_filter: 'float'
    BoxLayout:
        orientation: 'horizontal'
        Label:
            text: 'Number of Major Vessels (0-3) Colored by Flouroscopy'
        TextInput:
            id: num
            multiline: False
            input_filter: 'int'
    BoxLayout:
        orientation: 'horizontal'
        Label:
            text: 'Thalassemia (1-3)'
        TextInput:
            id: thal
            multiline: False
            input_filter: 'int'
    Button:
        text: 'Predict'
        on_press: root.predict()
    Label:
        id: output_label
        text: ''
""")


class MyLayout(BoxLayout):
    def predict(self):
        print("hello")

class MyApp(App):
    def build(self):
        from kivy.core.window import Window
        Window.size = (900, 900) # set the window size
        return MyLayout()

if __name__ == '__main__':
    MyApp().run()

class MyLayout(BoxLayout):
    def predict(self):
         #Load the saved model
         dt = load('heart_disease_model.joblib')

        #Get user input values
        age = int(self.ids.age.text)
        gender = int(self.ids.gender.text)
        cp = int(self.ids.cp.text)
        trestbps = int(self.ids.trestbps.text)
        fbs = int(self.ids.fbs.text)
        restecg = int(self.ids.restecg.text)
        thalach = int(self.ids.thalach.text)
        exang = int(self.ids.exang.text)
        oldpeak = float(self.ids.oldpeak.text)
        num = int(self.ids.num.text)
        thal = int(self.ids.thal.text)

        #Make a prediction
        X = np.array([[age, gender, cp, trestbps, fbs, restecg, thalach, exang, oldpeak, num, thal]])
        y_pred = dt.predict(X)

        #Show the prediction output
        if y_pred == 1:
            self.ids.output_label.text = 'The person has heart disease.'
        else:
            self.ids.output_label.text = 'The person does not have heart disease.'


