
import sys

app = None


print("Python version")
print (sys.version)

from test import *
from flask import *
from PIL import Image
app = Flask(__name__)

@app.route('/')
def index():
	return render_template('test.html')

@app.route('/api', methods=["GET", "POST"])
def api():
	uploaded_file = request.files.get('file')
	ques=request.form.get('ques')
	k=run_model(uploaded_file,ques)
	return k

if __name__=='__main__':
	app.run(host='0.0.0.0')

