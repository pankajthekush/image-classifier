from flask import Flask
from bp_classify.views import bp_classify

app = Flask(__name__)
app.register_blueprint(bp_classify)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
