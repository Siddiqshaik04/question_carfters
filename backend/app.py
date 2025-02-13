from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import os
import random

app = Flask(__name__)
CORS(app)

# Load the T5 model and tokenizer
model_dir = os.path.join(os.getcwd(), "models/t5-base-qg-hl")
if not os.path.exists(model_dir):
    print(f"Warning: Model directory not found at {model_dir}. Ensure the model is properly placed.")
else:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    question_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Extract text from PDF
def extract_text_from_pdf(pdf_path, max_length=5000):
    try:
        document = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in document]).strip()
        return text[:max_length] if text else ""
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

# Generate questions and options
import random

def generate_content(text, num_questions=10, difficulty='easy', mode="questions"):
    if not text:
        return []

    max_text_length = 1024  # Avoid model overflow
    text = text[:max_text_length]

    input_text = f"Generate {difficulty} level {mode}: {text}"

    max_length_dict = {'easy': 50, 'medium': 100, 'hard': 150}
    max_length = max_length_dict.get(difficulty, 50)

    try:
        generated_items = question_generator(input_text, max_length=max_length, num_return_sequences=num_questions, do_sample=True)
        generated_texts = [item['generated_text'] for item in generated_items]

        if mode == "questions":
            return generated_texts

        elif mode == "quiz":
            quiz_data = []
            for question_text in generated_texts:
                # Generate a correct answer
                correct_answer_prompt = f"Generate the correct answer for: {question_text}"
                correct_answer = question_generator(correct_answer_prompt, max_length=50, num_return_sequences=1, do_sample=False)[0]['generated_text']

                # Generate three incorrect answers ensuring they are not the question itself
                incorrect_answers_prompt = f"Generate three incorrect answers for: '{question_text}'. The correct answer is '{correct_answer}'. The incorrect answers should be plausible but incorrect."
                incorrect_answers = question_generator(incorrect_answers_prompt, max_length=50, num_return_sequences=3, do_sample=True)
                
                # Ensure incorrect answers are distinct and do not repeat the question
                incorrect_options = list(set([item['generated_text'] for item in incorrect_answers if item['generated_text'] != question_text]))

                # If we don't get enough distinct incorrect answers, regenerate them
                while len(incorrect_options) < 3:
                    additional_incorrect = question_generator(incorrect_answers_prompt, max_length=50, num_return_sequences=(3 - len(incorrect_options)), do_sample=True)
                    incorrect_options += [item['generated_text'] for item in additional_incorrect if item['generated_text'] != question_text]
                    incorrect_options = list(set(incorrect_options))[:3]

                # Combine correct and incorrect answers
                options = [correct_answer] + incorrect_options
                random.shuffle(options)  # Shuffle options to avoid the correct answer always being first

                quiz_data.append({
                    "question": question_text,
                    "options": options,
                    "correctAnswer": correct_answer
                })

            return quiz_data

    except Exception as e:
        print(f"Error generating {mode}: {e}")
        return []


@app.route('/api/generate', methods=['POST'])
def generate_questions_or_quiz():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    mode = request.form.get('mode', 'questions')
    num_questions = int(request.form.get('numberOfQuestions', 10))
    difficulty = request.form.get('difficulty', 'easy')

    pdf_path = "temp.pdf"
    try:
        file.save(pdf_path)
        document_text = extract_text_from_pdf(pdf_path)

        if not document_text:
            return jsonify({'error': 'No text extracted from PDF'}), 400

        generated_data = generate_content(document_text, num_questions=num_questions, difficulty=difficulty, mode=mode)

        if not generated_data:
            return jsonify({'error': f'Failed to generate {mode}'}), 500

        return jsonify({mode: generated_data})

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

if __name__ == '__main__':
    app.run(debug=True)
