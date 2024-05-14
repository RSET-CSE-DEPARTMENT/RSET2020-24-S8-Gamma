import secrets
import os
from flask import render_template,url_for,flash,redirect,request,abort
from flask_cors import cross_origin
from web import app,db,bcrypt
from flask_login import login_user, current_user, logout_user, login_required
from web.forms import RegistrationForm,LoginForm, UpdateAccountForm
from PIL import Image
from web.model import User
import pickle
import pandas as pd
import numpy as np
import time
# Load the model
# model = load_model('D:\FinalProject\web\lstm.h5')

import csv
from collections import defaultdict
# def read_csv_file(filename, key_column, value_column):
#     result = defaultdict(list)  
    
#     with open(filename, 'r') as file:
#         reader = csv.DictReader(file)  
#         for row in reader:
#             key = row[key_column]  
#             value = row[value_column]  
#             if key in result.keys():
#                 if value in result[key]:
#                     continue    
#                 result[key].append(value)  
#             else:
#                 result[key].append(value)
#     return result
# filename = 'D:\Miniproject\web\cars.csv'  
# key_column = 'Name'  
# value_column = 'Model'   

# data = read_csv_file(filename, key_column, value_column)

with open("D:\\FinalProject\\web\\model.pkl", 'rb') as file:
    rfmodel = pickle.load(file)
# rfmodel = pickle.load(open("D:\\FinalProject\\web\\rfmodel.pkl", 'rb'))

# car=pd.read_csv('D:\Miniproject\web\cars.csv')
@app.route('/')
def home():
    return render_template('home.html')

@app.route("/login",methods=["GET","POST"])
def login():
    form=LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password,form.password.data):
            login_user(user,remember=form.remember.data)   
            next_page= request.args.get('next') 
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash("Login failed check username and password ","danger")
    return render_template('login.html',title='Login',form=form)

@app.route('/deforestation-prediction')
def deforestation_prediction():
    return render_template('deforestation_prediction.html')

@app.route('/trend')
def trend():

    return render_template('trend.html')

# @app.route('/prediction')
# def prediction():
#     # Render the prediction.html template
#     return render_template('prediction.html')

# @app.route('/home',methods=['GET','POST'])
# def home():
#     name=car['Name'].unique().tolist()
#     year=sorted(car['Year'].unique(),reverse=True)
#     fuel_type=car['Fuel'].unique().tolist()
#     type=car['Type'].unique().tolist()
#     car_model=car['Model'].unique().tolist()
    
#     name.insert(0,'Select Car Name')
#     year.insert(0,'Select Year')
#     fuel_type.insert(0,'Select fuel type')
#     type.insert(0,'Select transmission type ')
#     car_model.insert(0,'Select car model')
    
#     return render_template('home.html' ,data=data,names=name,car_models=car_model, years=year,fuel_types=fuel_type,type=type)

# @app.route('/predict', methods=['POST'])
# def predict():
  
#     # Perform any necessary preprocessing on the data
#     # (e.g., convert to numpy array, scale, reshape, etc.)
#     input_data = np.array([-52.03797437,-9.598418379,2008])
#     # Use the loaded model to make predictions
#     predictions = model.predict(input_data)
    
#     # Return predictions as a JSON response
#     print(predictions)


# @app.route('/predict',methods=['POST'])
# @cross_origin()
# def predict():
#     prediction=lrmodel.predict(pd.DataFrame(columns=['lat','long','year'],data=np.array([lat,long,year]).reshape(1, 3)))
#     print(prediction)

#     return str(np.round(prediction[0],6))
lat,lon,yr=0,0,0
@app.route('/predict')
def predict():
    global lat
    lat = request.args.get('lat')
    # print(lat)
    global lon
    lon = request.args.get('lon')
    # print(lon)

    
    return render_template('prediction.html', latitude=lat, longitude=lon)
    # prediction=rfmodel.predict(pd.DataFrame(columns=['lat','lon','year'],data=np.array([lat,lon,yr]).reshape(1, 3)))
    # print(prediction)
    # test=[[lat,lon,yr]]
    # # Assuming X_test contains the features for which you want to make predictions
    # predictions = rfmodel.predict(test)
    # print(predictions)

@app.route('/result',methods=["GET","POST"])
@cross_origin()
def result():
    print(lat)
    print(lon)
    global yr
    yr=request.form.get('year')
    print(yr)
    # prediction=rfmodel.predict(pd.DataFrame(columns=['lat','lon','year'],data=np.array([lat,lon,yr]).reshape(1, 3)))
    # print(prediction)
    test=[[float(lat),float(lon),float(yr)]]
    # Assuming X_test contains the features for which you want to make predictions
    predictions = rfmodel.predict(test)
    # time.sleep(3)
    # print("--------",yr)
    print(predictions)
    if np.round(predictions[0],5)<=0:
        return "0.00001"
    return str(abs(np.round(predictions[0],5)))    # return lat


@app.route('/why-para')
def why_para():
    return render_template('why-para.html')

@app.route("/register",methods=["GET","POST"])
def register():
    
    form=RegistrationForm()
    if form.validate_on_submit():
        h_pwd= bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user= User(username=form.username.data, email=form.email.data,password=h_pwd)
        db.session.add(user)
        db.session.commit()
        flash(f'Account created successfully!','success')
        return redirect(url_for('login'))
    return render_template('register.html',title='Register',form=form)

# @app.route("/index",methods=["GET","POST"])
# def index():
#     return render_template('index.html',title='Trends')


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('login'))

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(app.root_path, 'static/profile_pics', picture_fn)

    output_size = (125, 125)
    i = Image.open(form_picture)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_fn

@app.route("/account",methods=["GET","POST"])
@login_required
def account():
    form=UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Your account has been updated!', 'success')
        return redirect(url_for('account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/' + current_user.image_file)
    return render_template('account.html',title='Account',image_file =image_file,form=form)