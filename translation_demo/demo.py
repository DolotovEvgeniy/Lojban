import os
from flask import Flask, render_template, redirect, request
app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html', translation=translation, entries=entries)


@app.route("/translate", methods=["POST"])
def translate():
    text = request.form['text']
    print(text)
    with open('src.txt', 'w') as f:
        print(text, file=f)
    os.system('/home/maxim/work/virtualenvs/pytorch/bin/python ../../OpenNMT-py/translate.py -gpu 0  -model loj_eng_acc_48.05_ppl_23.21_e13.pt -src src.txt -output output.txt -replace_unk')
    global translation
    with open('output.txt') as f:
        translation = f.read()
    return redirect('/')


if __name__ == "__main__":
    entries = []
    with open('val_loj.txt') as f1, open ('val_eng.txt') as f2:
        data1 = f1.read().splitlines()
        data2 = f2.read().splitlines()
    for loj, eng in zip(data1, data2):
        entries.append({'lojban': loj, 'english': eng})
    translation = ''
    app.run()
