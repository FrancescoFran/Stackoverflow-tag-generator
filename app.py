from flask import Flask, request, render_template
import joblib
import text_preprocessor as txt_ppc

# Declare a Flask app
app = Flask(__name__)


# Main function
@app.route('/', methods=['GET', 'POST'])
def main():
    # If a form is submitted
    if request.method == "POST":
        # Unpickle vectorizer, binarizer and classifier
        model_path = "C:/Users/franc/PycharmProjects/pythonProject5/static/models/"
        vect = joblib.load(model_path + "tfidf_vectorizer.pkl")
        multilabel_bin = joblib.load(model_path + "multilabel_binarizer.pkl")
        model = joblib.load(model_path + "log_ref_clf.pkl")

        # Get values through input bars
        text = request.form.get("question")

        # check language
        x = txt_ppc.lang_check(text)

        # clean text
        x = txt_ppc.cleaner(x)

        # vectorize
        x_tfidf = vect.transform([x])

        # predict
        predict = model.predict(x_tfidf)
        tags_predict = multilabel_bin.inverse_transform(predict)
        results = tags_predict

    else:
        results = ""

    return render_template("website.html", output=results)


# Running the app
if __name__ == '__main__':
    app.run(debug=True)
