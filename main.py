from flask import Flask, render_template, json
from werkzeug.exceptions import HTTPException

app = Flask('todo_app')

@app.errorhandler(HTTPException)
def handle_exception(e: HTTPException):
    return render_template('error.html', context={"errorCode": e.code})


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
