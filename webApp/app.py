import re
from flask import Flask,render_template, url_for
from flask import request
import os

from requests import delete
from prediction import predict

UPLOAD_FOLDER = os.path.join(os.getcwd(),'static/projectImg')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app=Flask(__name__)

def Remove():
    from glob import glob
    for path in glob(UPLOAD_FOLDER+"/*"):
        os.remove(path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/project')
def project():
    return render_template('project.html')

# tool page  -----> starts
@app.route('/tool',methods=["POST","GET"])
def tool():
    if request.method=='POST':
        image_file=request.files['inputImg']
        name=request.form['name']
        age=request.form['age']
        sex=request.form['gender']
        # save input file
        if image_file:
            imageName=image_file.filename.split('.')[0]
            image_location=UPLOAD_FOLDER+"/"+image_file.filename
            image_file.save(image_location)
            # predict output
            predict(str(image_location))
        return render_template('result.html',value=[imageName+".jpg",imageName+"_output.jpg",imageName+"_output2.jpg",name,age,sex])
    else:
        Remove()
        return render_template('tool.html')

# tool page <--- ends here
if __name__=="__main__":
    # add model path in prediction.py file in load_model() function
    app.run(debug=True)
