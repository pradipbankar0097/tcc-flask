# Important imports
from app import app
from flask import request, render_template
import os
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image
import numpy as np
import easyocr


# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'
app.config['EXISTNG_FILE'] = 'app/static/original'
app.config['GENERATED_FILE'] = 'app/static/generated'

# Route to home page
@app.route("/", methods=["GET", "POST"])
def index():
	params={
		"full_name":"",
		"amount_paid":"",
		"outstanding":"",
	}
	# Execute if request is get
	if request.method == "GET":
	    return render_template("index.html",params=params)

	# Execute if reuqest is post
	if request.method == "POST":
                # Get uploaded image
                file_upload = request.files['file_upload']
                
                # save image
                uploaded_image = Image.open(file_upload)
                uploaded_image.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image.jpg'))

                # Read uploaded and original image as array
                uploaded_image = cv2.imread(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image.jpg'))
                reader = easyocr.Reader(['en'],gpu=['gpu']) 
                all_data = reader.readtext(uploaded_image)
                params["full_name"]=all_data[9][1]
                params["amount_paid"]=all_data[34][1][1:]
                params["outstanding"]=all_data[37][1][1:]
                j:int=0
                for i in all_data:
                    print(str(j)+'=='+str(i[1]))
                    j+=1
                return render_template('index.html',params=params)
       
# Main function
if __name__ == '__main__':
    app.run(debug=True)
