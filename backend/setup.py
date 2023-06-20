from flask import Flask
import logging

app = Flask(__name__)


logging.basicConfig(filename='logs.log', level=logging.DEBUG,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')

@app.route("/")
def hello_world():
    return "<h1>I'ts GI-tool backend</h1><p>Hello World!</p>"


if __name__ == '__main__':
    app.run(debug=True)
