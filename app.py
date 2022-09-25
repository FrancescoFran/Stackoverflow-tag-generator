from flask import Flask, request, render_template
import pandas as pd
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
        model_path = "static/models/"
        vect = joblib.load(model_path + "tfidf_vectorizer.pkl")
        multilabel_bin = joblib.load(model_path + "multilabel_binarizer.pkl")
        model = joblib.load(model_path + "log_ref_clf.pkl")

        # Get values through input bars
        text = request.form.get("question")

        # check language
        x = txt_ppc.lang_check(text)

        results = {}
        if x != 'Not english':
            # clean text
            x = txt_ppc.cleaner(x)

            # vectorize
            x_tfidf = vect.transform([x])

            # predict
            predict = model.predict(x_tfidf)
            predict_prob = model.predict_proba(x_tfidf)
            tags_predict = multilabel_bin.inverse_transform(predict)

            a = []
            for j in range(0, 50):
                a.append(round(predict_prob[0, j] * 100, 1))

            df = pd.DataFrame(columns=['tags', 'probs'])
            df['tags'] = multilabel_bin.classes_
            df['probs'] = a
            df = df.sort_values('probs', ascending=False).head(10)

            results['Tags'] = tags_predict
            results['Probabilities'] = df.set_index('tags')['probs'].to_dict()
            results['lang_check'] = ''
        else:
            results['lang_check'] = 'Please reformulate your question in english'
    else:
        results = ""

    return render_template("website.html", output=results)


# Running the app
if __name__ == '__main__':
    app.run(debug=True)
