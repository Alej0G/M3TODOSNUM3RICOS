from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chapter1")
def chapter1():
    return render_template("chapter1.html")

@app.route("/chapter2")
def chapter2():
    return render_template("chapter2.html")

@app.route("/chapter3")
def chapter3():
    return render_template("chapter3.html")

if __name__ == "__main__":
    app.run(debug=True)