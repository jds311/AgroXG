from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
#import urllib.request
import os
from werkzeug.utils import secure_filename
import train
import json

app = Flask(__name__)

@app.route("/")
def hello():
    return jsonify({'messege':'Welcome to Cloud Computing'})
   # return render_template('index.html')
    
def demo(reqId,subscription,watchlist):
        if count>=3:
            data["count"] = 0
            train.train()
            with open("../Datav/Data/disease.json", "w") as jsonFile:
                json.dump(data, jsonFile)
            
@app.route("/upload/<image_name>/<disease>")
def hello1(image_name,disease):
    print(image_name,"  ",disease)
    ss =image_name,disease
    command="mv "+"../Datav/Upload_Images/"+image_name+" ../Datav/'Large Wheat Disease Classification Dataset'/'"+disease+"'/"
    os.system(command)

    with open('../Datav/Data/disease.json') as config_file:
        data = json.load(config_file)
    count = data['count']
    t1=threading.Thread(target=demo,args=(count,data,))    # 1st thread with subscription_1 and watchlist_1
    t1.start()
    # return ss
    # print(disease)
    return jsonify({'messege':ss})
   # return render_template('index.html')

@app.route("/train")
def train_function():
    train.train()
    return jsonify({'messege':'Welcome to Cloud Computing'})
   # return render_template('index.html')


# @app.route('/', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         flash('No image selected for uploading')
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         #print('upload_image filename: ' + filename)
#         flash('Image successfully uploaded and displayed below')
#         return render_template('index.html', filename=filename)
#     else:
#         flash('Allowed image types are - png, jpg, jpeg, gif')
#         return redirect(request.url)
 
# @app.route('/display/<filename>')
# def display_image(filename):
#     #print('display_image filename: ' + filename)
#     return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True,port=8082)
