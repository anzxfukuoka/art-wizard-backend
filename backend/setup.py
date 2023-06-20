from flask import Flask
import logging
from app import create_app

flask_app = create_app()

logging.basicConfig(filename='logs.log', level=logging.DEBUG,
                    format=f'%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')


@flask_app.route("/")
def hello_world():
    return "<h1>I'ts GI-tool backend</h1><p>Hello World!</p>"


if __name__ == '__main__':
    flask_app.run(debug=True, static_url_path='app/image_analyzer/data')
