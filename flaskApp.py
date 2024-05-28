from flask import Flask, render_template, Response, jsonify, request, session, send_from_directory
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
from detection import objectDetection
import os
import cv2

app = Flask(__name__)

app.config['SECRET_KEY'] = 'thanhdo'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['SESSION_TYPE'] = 'filesystem'  # Sử dụng filesystem để lưu trữ dữ liệu phiên nếu cần

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Run")

fpsCount = 0
frameSize = 0
detectedObjects = 0

def generate_frames(path):
    yolov9_output = objectDetection(path)
    for im0, frameRate, frameshape, totalDetection in yolov9_output:
        ret, buffer = cv2.imencode('.jpg', im0)
        global fpsCount
        fpsCount = str(frameRate)
        global frameSize
        frameSize = str(frameshape[0])
        global detectedObjects
        detectedObjects = str(totalDetection)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        session['filePath'] = file_path
        print(f"File uploaded and saved to session: {file_path}")  # Dòng lệnh gỡ lỗi
        return render_template('index.html', form=form, uploaded=True, filename=filename)
    return render_template('index.html', form=form, uploaded=False)

@app.route('/detections', methods=['GET', 'POST'])
def detections():
    file_path = session.get('filePath', None)
    print(f"File path retrieved from session: {file_path}")  # Dòng lệnh gỡ lỗi
    if file_path:
        return Response(generate_frames(path=file_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "No video uploaded"

@app.route('/fps', methods=['GET'])
def fps():
    global fpsCount
    return jsonify(fpsresult=fpsCount)

@app.route('/dcount', methods=['GET'])
def dcount():
    global detectedObjects
    return jsonify(dcountresult=detectedObjects)

@app.route('/fsize', methods=['GET'])
def fsize():
    global frameSize
    return jsonify(fsizeresult=frameSize)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == "__main__":
    app.run(debug=True)
