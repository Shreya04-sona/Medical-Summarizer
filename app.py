from flask import Flask, render_template, request
from summarizer import summarize_text

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        medical_note = request.form["note"]
        summary = summarize_text(medical_note)
    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
