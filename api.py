import os
import urllib.request
from flask import Flask, request, redirect, jsonify,send_file
from werkzeug.utils import secure_filename
from face_blur import anonymize_face_pixelate,anonymize_face_simple
import numpy as np
import cv2
import os


UPLOAD_FOLDER = 'E:\\python\\opencv_blur_api\\inputs'
output_folder = 'E:\\python\\opencv_blur_api\\outputs\\'
app = Flask(__name__)
app.secret_key = "adjksakjdhjkas"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
prototxtPath = 'face_detector/deploy.prototxt'
weightsPath = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNet(prototxtPath, weightsPath)

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET'])
def home():
	return'''<div><center><h1>Hello from API </h1></center></div> '''
@app.route('/file',methods=['POST'])
def upload_file():
	if 'file' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	file = request.files['file']
	if file.filename == '':
		resp = jsonify({'message' : 'No file selected for uploading'})
		resp.status_code = 400
		return resp
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		image = cv2.imread(UPLOAD_FOLDER+'\\'+filename)
		(h,w) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()
		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > 0.5:
				box =detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				face = image[startY:endY, startX:endX]
				face = anonymize_face_simple(face, factor=1.5)
				image[startY:endY, startX:endX] = face
		cv2.imwrite(os.path.join(output_folder , filename), image)
		resp = jsonify({'blur' : '/output/'+filename})
		resp.status_code = 201
		os.remove(UPLOAD_FOLDER+'\\'+filename)
		return resp
	else:
		resp = jsonify({'message' : 'Allowed file types are png, jpg, jpeg'})
		resp.status_code = 400
		return resp
@app.route('/output/<filename>')
def display_image(filename):
	file = output_folder+ filename
	return send_file(file, mimetype='image/gif')
if __name__ == '__main__':
	app.run(host ='192.168.43.193',threaded=True)