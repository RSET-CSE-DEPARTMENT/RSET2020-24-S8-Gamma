from PyPDF2 import PdfReader
from flask import render_template, url_for, flash, redirect, request,session,send_file
from questionpapergenerator import app, users_collection , pdf_collection
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5TokenizerFast
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import torch,random
import datetime
import os
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
from io import BytesIO
import io
from bson import ObjectId
from werkzeug.utils import secure_filename
import random


#Extraction from PDF
def extract_text_from_pdf(file_path):
   reader = PdfReader(file_path)
   text = ''
   for page in reader.pages:
       text += page.extract_text()
   return text


import nltk
from nltk.tokenize import word_tokenize
import re

def preprocess_text(text, segment_length=1700):
    # Remove leading and trailing whitespace
    text = text.strip()

    # Replace bullet points with a space
    text = re.sub(r'\s*•\s*', ' ', text)

    # Replace newlines and multiple whitespaces with a single space
    text = ' '.join(text.split())

    # Tokenize the text using NLTK's word_tokenize function
    tokens = word_tokenize(text)

    # Join tokens into segments of specified length
    segments = []
    segment = ''
    for token in tokens:
        if len(segment) + len(token) < segment_length:
            segment += ' ' + token
        else:
            segments.append(segment.strip())
            segment = token
    segments.append(segment.strip())  # Append the remaining segment
    return segments



#Question Generation
checkpoint = "t5-base"
tokenizer = T5TokenizerFast.from_pretrained(checkpoint)


model = AutoModelForSeq2SeqLM.from_pretrained("rohan-jp1/t5-end2end-questions-generation")


import random


def hf_run_model(input_list, num_return_sequences=8, num_questions=2, max_sequence_length=512, generator_args=None):
   if generator_args is None:
       generator_args = {
           "max_length": max_sequence_length,
           "num_beams": 10,
           "length_penalty": 1.5,
           "no_repeat_ngram_size": 6,
           "early_stopping": True,
           "temperature": 0.8,  # Adjust the temperature value (higher values for more randomness)
           "top_k": 50,  # Adjust the top_k value (higher values for more diverse output)
           "top_p": 0.95  # Adjust the top_p value (lower values for more focused output)
       }
  
   generated_questions = []
   unique_questions = set()


   #creating tensors of each input
   for input_string in input_list:
       input_string = "generate questions: " + input_string + " </s>"
       input_ids = tokenizer.encode(input_string, truncation=True, max_length=max_sequence_length, return_tensors="pt")


       # Generate questions using the model
       res = model.generate(input_ids, **generator_args, num_return_sequences=num_return_sequences)
       output = tokenizer.batch_decode(res, skip_special_tokens=True, clean_up_tokenization_spaces=True)


       segment_questions = []
       for sequence in output:
           sequence = sequence.split("<sep>")
           questions = [question.strip() + "?" for question in sequence[0].split("?") if question.strip()]
           segment_questions.extend(questions[:num_questions])  # Selecting the desired number of questions from each segment


       # Filter out single-word questions for each segment
       segment_questions = [question for question in segment_questions if len(question.split()) > 1]
       generated_questions.extend(segment_questions)


   # Randomly sample questions until reaching the desired number of non-repeated questions
   while len(unique_questions) < num_questions * len(input_list):  # Generating questions from each segment
       question = random.choice(generated_questions)
       generated_questions.remove(question)
       if question not in unique_questions:
           unique_questions.add(question)


   return list(unique_questions)

#Question Answering

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name="deepset/roberta-base-squad2"

def get_answers(context, questions, model_name):
    # Initialize the pipeline outside the function to avoid repeated initialization
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    
    # Prepare inputs for the pipeline
    inputs = [{'question': question, 'context': context} for question in questions]
    
    # Use the pipeline to get answers for all questions
    answers = nlp(inputs)
    
    # Extract answers from the results
    answer_texts = [res['answer'] for res in answers]
    
    return answer_texts


# Distractor Generation

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_distractors(context, questions, answers, model_name="potsawee/t5-large-generation-race-Distractor"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    max_length = tokenizer.model_max_length
    results = []

    for question, answer in zip(questions, answers):
        input_text = f"{question} {tokenizer.sep_token} {answer} {tokenizer.sep_token} {context}"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=128)
        distractors = tokenizer.decode(outputs[0], skip_special_tokens=False)
        distractors = distractors.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "")
        distractors = [y.strip() for y in distractors.split(tokenizer.sep_token)]
        distractors.append(answer)
        random.shuffle(distractors)
        results.append(distractors)
        
    for i in range(len(results)):
        results[i]=list(set(results[i]))

    return results

#Making PDF


def subjective_template(data_list,answers_list, output_file,subject_name):
    # Create the PDF canvas
    c = canvas.Canvas(output_file, pagesize=letter)

    # Add the template or background image
    template_path = 'template.png'
    c.drawImage(template_path, 0, 0, width=letter[0], height=letter[1])

    # Set up paragraph styles
    styles = getSampleStyleSheet()
    paragraph_style = ParagraphStyle(
        'normal',
        parent=styles['Normal'],
        textColor=colors.black,
        fontSize=12,
        leading=16  # Adjust the leading for more spacing between lines
    )

    # Set the font and size
    c.setFont("Helvetica-Bold", 16)
    text_width = c.stringWidth(subject_name, "Helvetica-Bold", 16)
    x_position = (letter[0] - text_width) / 2
    c.drawCentredString(letter[0] / 2, 575, subject_name)
    c.line(x_position, 570, x_position + text_width, 570)  # Draw underline
    c.setFont("Helvetica", 12)


    # Write the list elements to the PDF
    y = 550  # Starting y position
    index = 1
    spacing = 20  # Fixed spacing between paragraphs

    for item in data_list:
        text = f"{index}) {item}"
        p = Paragraph(text, style=paragraph_style)
        p.wrapOn(c, 400, 0)

        # Check if there's enough space on the page for the paragraph
        if y - p.height < 50:
            c.showPage()  # Start a new page
            y = 750  # Reset the y position to the top of the new page

        p.drawOn(c, 100, y-p.height)
        y -= p.height + spacing  # Adjust the spacing between paragraphs
        index += 1

    #answer key
    c.showPage()
    c.setFont("Helvetica-Bold", 16)
    text_width = c.stringWidth("ANSWER KEY", "Helvetica-Bold", 16)
    x_position = (letter[0] - text_width) / 2
    c.drawCentredString(letter[0] / 2, 750, "ANSWER KEY")
    c.line(x_position, 745, x_position + text_width, 745)  # Draw underline
    c.setFont("Helvetica", 12)

    y = 700  # Starting y position
    spacing = 20  # Fixed spacing between answers

    for index, answer in enumerate(answers_list, start=1):
        text = f"{index}) {answer}"
        p = Paragraph(text, style=paragraph_style)
        p.wrapOn(c, 400, 0)

        # Check if there's enough space on the page for the answer paragraph
        if y - p.height < 50:
            c.showPage()  # Start a new page
            y = 750  # Reset the y position to the top of the new page

        p.drawOn(c, 100, y - p.height)
        y -= p.height + spacing  # Adjust the spacing between answers 

    c.save()

    # Save the PDF file into MongoDB
    with open(output_file, 'rb') as pdf_file:
        pdf_data = pdf_file.read()

    username = session.get('username')  # Get the username from session or any relevant source

    user = users_collection.find_one({'username': username})  # Retrieve the user document from MongoDB

    if user:
        user_id = user['_id']  # Assuming the user ID is stored in the '_id' field
        timestamp = datetime.datetime.now()  # Generate a timestamp

        filename = session.get('file_name')
        file_name = f"subjective_{filename}"
        

        pdf_document = {
            "user_id": user_id,
            "timestamp": timestamp,
            "file_name": file_name,
            "pdf_file": pdf_data,
            "subject_name": subject_name
        }

        pdf_collection.insert_one(pdf_document)
        print("PDF saved to MongoDB successfully.")
    else:
        print("User not found. PDF not saved.")




def mcq_template(questions,answers_list, options, output_file, subject_name):
    c = canvas.Canvas(output_file, pagesize=letter)
    # Set the font and size
    c.setFont("Helvetica", 12)
    # Add the template or background image
    template_path = 'template.png'
    c.drawImage(template_path, 0, 0, width=letter[0], height=letter[1])
    # Set up paragraph styles
    styles = getSampleStyleSheet()
    paragraph_style = ParagraphStyle(
        'normal',
        parent=styles['Normal'],
        textColor=colors.black,
        fontSize=12,
        leading=16  # Adjust the leading for more spacing between lines
    )
    # Write the list elements to the PDF
    y = 550  # Starting y position
    index = 1
    spacing = 20  # Fixed spacing between paragraphs
    for question, option_list in zip(questions, options):
        # Print the question
        question_text = f"{index}) {question}"
        question_paragraph = Paragraph(question_text, style=paragraph_style)
        question_paragraph.wrapOn(c, 400, 0)
        # Check if there's enough space on the page for the question paragraph
        if y - question_paragraph.height < 50:
            c.showPage()  # Start a new page
            y = 750  # Reset the y position to the top of the new page
        question_paragraph.drawOn(c, 100, y-question_paragraph.height)
        y -= question_paragraph.height + spacing  # Adjust the spacing between paragraphs
        # Check remaining space for options
        remaining_space = y - (len(option_list) * (question_paragraph.height + spacing))
        if remaining_space < 40:
            c.showPage()  # Start a new page
            y = 750  # Reset the y position to the top of the new page
        # Print options
        option_index = 1
        for option in option_list:
            option_text = f"{chr(96 + option_index)}) {option}"  # Using ASCII code to generate options (a, b, c, ...)
            option_paragraph = Paragraph(option_text, style=paragraph_style)
            option_paragraph.wrapOn(c, 400, 0)
            # Check if there's enough space on the page for the option paragraph
            if y - option_paragraph.height < 50:
                c.showPage()  # Start a new page
                y = 750  # Reset the y position to the top of the new page
            option_paragraph.drawOn(c, 120, y-option_paragraph.height)
            y -= option_paragraph.height + spacing  # Adjust the spacing between paragraphs
            option_index += 1
        index += 1
    
    c.showPage()
    c.setFont("Helvetica-Bold", 16)
    text_width = c.stringWidth("ANSWER KEY", "Helvetica-Bold", 16)
    x_position = (letter[0] - text_width) / 2
    c.drawCentredString(letter[0] / 2, 750, "ANSWER KEY")
    c.line(x_position, 745, x_position + text_width, 745)  # Draw underline
    c.setFont("Helvetica", 12)

    y = 700  # Starting y position
    spacing = 20  # Fixed spacing between answers

    for index, answer in enumerate(answers_list, start=1):
        text = f"{index}) {answer}"
        p = Paragraph(text, style=paragraph_style)
        p.wrapOn(c, 400, 0)

        # Check if there's enough space on the page for the answer paragraph
        if y - p.height < 50:
            c.showPage()  # Start a new page
            y = 750  # Reset the y position to the top of the new page

        p.drawOn(c, 100, y - p.height)
        y -= p.height + spacing  # Adjust the spacing between answers        
    # Save the canvas as the final PDF
    c.save()
    # Save the PDF file into MongoDB
    with open(output_file, 'rb') as pdf_file:
        pdf_data = pdf_file.read()
    username = session.get('username')  # Get the username from session or any relevant source
    user = users_collection.find_one({'username': username})  # Retrieve the user document from MongoDB
    if user:
        user_id = user['_id']  # Assuming the user ID is stored in the '_id' field
        timestamp = datetime.datetime.now()  # Generate a timestamp
        filename = session.get('file_name')
        file_name = f"mcq_{filename}"
        pdf_document = {
            "user_id": user_id,
            "timestamp": timestamp,
            "file_name": file_name,
            "pdf_file": pdf_data,
            "subject_name": subject_name
        }
        pdf_collection.insert_one(pdf_document)
        print("PDF saved to MongoDB successfully.")
    else:
        print("User not found. PDF not saved.")





@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        subjects = []
        existing_user = users_collection.find_one({'username': username})

        if existing_user:
            return "Username already exists!"

        user = {'username': username, 'password': password, 'subjects': subjects}
        users_collection.insert_one(user)
        session['username'] = username
        return redirect('/')
    else:
        return render_template('register.html')


@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = users_collection.find_one({'username': username, 'password': password})

        if existing_user:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect('/userdashboard')
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
            return redirect('/')
    else:
        return render_template('login.html')




@app.route('/logout')
def logout():
    session.pop('username', None)
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect('/')


@app.route("/userdashboard")
def user_dashboard():
    username = session.get('username')
    user = users_collection.find_one({'username': username})
    subjects=user['subjects']
    return render_template('dashboard.html',username=username,subjects=subjects)


import os
@app.route('/upload/<subject_name>', methods=['GET', 'POST'])
def upload(subject_name):
   if request.method == 'POST':
       file = request.files['file']
       if file:
           # Save the uploaded file to a temporary directory
           temp_dir = '/tmp'
           file_path = os.path.join(temp_dir, file.filename)
           file.save(file_path)
           session['file_name'] = file.filename
           # Perform text extraction
           extracted_text = extract_text_from_pdf(file_path)
           # Delete the temporary file
           os.remove(file_path)
           # Continue with text processing
           preprocessed_text = preprocess_text(extracted_text, segment_length=1700)
           print(len(preprocessed_text))
           print(preprocessed_text)
           full_sentence = ' '.join(preprocessed_text)           
           questions = hf_run_model(preprocessed_text, num_return_sequences=8, num_questions=2)
           session['my_list'] = questions
           answers=get_answers(full_sentence,questions,model_name)
           session['my_list_1'] = answers
           #print(full_sentence)
           options = generate_distractors(full_sentence,questions,answers)
           session['options']=options

           for count, ele in enumerate(questions):
              print(count + 1)
              print(ele)

           for count, ele in enumerate(answers):
              print(count + 1)
              print(ele)


           return render_template('upload.html', file_name=file.filename, text1=extracted_text,
                                  text2=preprocessed_text,text3=questions,subject_name=subject_name)


   return render_template('upload.html',subject_name=subject_name)








@app.route('/generate_pdf/<subject_name>', methods=['GET'])
def generate_pdf(subject_name):
    output_path='output.pdf'
    question_type = request.args.get('question_type')  # Get the value of 'question_type' from the query parameters
    # Depending on the value of 'question_type', generate the PDF accordingly
    if question_type == 'mcq':
        items_1=session['my_list']
        items_2=session['my_list_1']
        options=session['options']
        
        mcq_template(items_1,items_2, options, output_path, subject_name)
    elif question_type == 'subjective':
        items_1=session['my_list']
        items_2=session['my_list_1']
        # Generate PDF with subjective questions
        subjective_template(items_1,items_2,output_path,subject_name)
    else:
        # Handle invalid or missing question_type parameter
        flash("Choose Question Paper type!","danger")
    return redirect(url_for('my_pdf_documents', subject_name=subject_name))


def fetch_pdf_documents_for_user(username,subject_name):
   user = users_collection.find_one({'username': username})  # Retrieve the user document from MongoDB
   if user:
       user_id = user['_id']  # Assuming the user ID is stored in the '_id' field
       pdf_documents = pdf_collection.find({'user_id': user_id,'subject_name': subject_name})  # Fetch all PDF documents for the user
       return pdf_documents
   else:
       return None



@app.route('/view_pdf/<pdf_id>')
def view_pdf(pdf_id):
   pdf_doc = pdf_collection.find_one({'_id': ObjectId(pdf_id)})


   if pdf_doc:
       pdf_file = pdf_doc['pdf_file']
       pdf_buffer = io.BytesIO(pdf_file)
       return send_file(pdf_buffer,mimetype='application/pdf')


   return "PDF not found"


@app.route('/my_pdf_documents/<subject_name>/')
def my_pdf_documents(subject_name):
   username = session['username']  # Get the username from session or any relevant source
   if username:
       pdf_documents = fetch_pdf_documents_for_user(username,subject_name)
       return render_template('my_pdf_documents.html', pdf_documents=pdf_documents,subject_name=subject_name)

   return "User not logged in"


@app.route('/remove_pdf/<pdf_id>/<subject_name>', methods=['POST'])
def remove_pdf(pdf_id,subject_name):
   # Remove the PDF file from the database
   pdf_collection.delete_one({'_id': ObjectId(pdf_id)})
   return redirect(url_for('my_pdf_documents',subject_name=subject_name))


@app.route("/add_subject", methods=["POST"])
def add_subject():
    if request.method == "POST":
        subject_name = request.form.get("subjectName")
        username = session.get('username')
        user =users_collection.find_one({'username': username})

        if user:
            # Update the subjects list
            users_collection.update_one(
                {'username': session.get("username")},
                {'$push': {'subjects': subject_name}}
            )

            flash(f"Subject '{subject_name}' added successfully!", "success")
            return redirect(url_for("user_dashboard"))

    flash("Error adding subject", "danger")
    return redirect(url_for("user_dashboard"))


@app.route("/remove_subject", methods=["POST"])
def remove_subject():
    if request.method == "POST":
        subject_name = request.form.get("subject_name")
        username = session.get('username')
        user = users_collection.find_one({'username': username})

        if user:
            # Update the subjects list by pulling the specified subject
            users_collection.update_one(
                {'username': session.get("username")},
                {'$pull': {'subjects': subject_name}}
            )

            flash(f"Subject '{subject_name}' removed successfully!", "success")
            return redirect(url_for("user_dashboard"))

    flash("Error removing subject", "danger")
    return redirect(url_for("user_dashboard"))

