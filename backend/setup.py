from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<h1>I'ts GI-tool backend</h1><p>Hello World!</p>"


if __name__ == '__main__':
    app.run()