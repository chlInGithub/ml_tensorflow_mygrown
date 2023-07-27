from flask import Flask, url_for, request, redirect,g
from markupsafe import escape
from secrets import token_hex
import json

app = Flask(__name__)

with app.app_context():
    print("在这里做一些扩展组件的启动，如数据库")

# 用于使用session
app.secret_key = token_hex(32)

def get_my_data():
    if 'my_data' not in g:
        g.my_data = 'this is myData'

    return g.my_data

@app.teardown_appcontext
def teardown_my_data(exception):
    my_data = g.pop('my_data', None)
    if 'my_data' is not None:
        print("teardown Mydata")

@app.route("/")
def index():
    return "hello world"

@app.get("/myData")
def my_data():
    return get_my_data()

@app.get("/user/<string:user_code>")
def user_by_code(user_code:str):
    for arg in request.args:
        print("arg ", arg, request.args[arg])

    return json.dumps({'code': 1, 'success': 1, 'data': list(range(10))})
    #return f"get userInfo: {escape(user_code)}"

@app.post("/user/<string:user_code>")
def post_user_by_code(user_code:str):
    return f"post userInfo: {escape(user_code)}"

@app.errorhandler(404)
def error_404(error):
    return redirect("/static/404.html")

with app.test_request_context():
    print(url_for("index"))
    print(url_for("user_by_code", _method='GET', user_code='chl'))