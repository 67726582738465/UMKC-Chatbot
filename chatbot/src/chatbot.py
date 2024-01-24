from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd

app = Flask(__name__)

# Load the fine-tuned GPT-2 model
model = GPT2LMHeadModel.from_pretrained('fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load your dataset
df = pd.read_csv('chatdata.csv')

def test_model(model, tokenizer, user_input):
    # Tokenize the user input
    input_ids = tokenizer.encode(f"User: {user_input} Bot:", return_tensors="pt", truncation=True)

    # Generate a response using the fine-tuned GPT-2 model
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

    # Decode the generated response
    generated_response = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_response

def is_question_related(user_input, df):
    # Check if the user input is in the dataset's questions
    return any(question.lower() in user_input.lower() for question in df['question'].values)

@app.route('/')
def home():
    return render_template('index.html', questions=df['question'].tolist())

@app.route('/filter_options')
def filter_options():
    input_text = request.args.get('input_text', '')
    filtered_options = [q for q in df['question'].tolist() if input_text.lower() in q.lower()]
    return jsonify(filtered_options)

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['question']
    selected_question = request.form['selected_question']

    # Use selected_question if available; otherwise, use user_input
    question_to_check = selected_question if selected_question else user_input

    if is_question_related(question_to_check, df):
        response = test_model(model, tokenizer, question_to_check)
    else:
        response = "I don't understand this. Please provide more information or contact the administrator for assistance. You can also visit https://www.umkc.edu/admissions/get-info.html"

    return render_template('index.html', question=user_input, response=response, questions=df['question'].tolist(), selected_question=selected_question)

if __name__ == '__main__':
    app.run(debug=True)