from flask import Flask, render_template, request
from text_generation import build_model, generate_txt
import tensorflow as tf
checkpoint_dir = 'training_checkpoint'
model = build_model(65, 256, 1024, 1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_text():
    word = request.form.get('INPUT A WORD')
    # predict
    result = generate_txt(model, str(word))

    return render_template(str('index.html'), result=result)

if __name__ == '__main__':
    app.run(debug=True)