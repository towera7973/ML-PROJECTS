from flask import Flask, request, jsonify
app_obj = Flask(__name__)

@app_obj.route("/hello")
def hello():
    return "Hello, World!"

    
if __name__ == "__main__":
    app_obj.run()