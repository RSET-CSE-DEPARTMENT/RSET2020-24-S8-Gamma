import os
import re
from flask import Flask, render_template, request, jsonify, redirect, flash, send_file
import mysql.connector
import string
from diffusers import StableDiffusionPipeline
from diffusers import models
from PIL import Image, ImageDraw, ImageFont
import requests
import io
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from gtts import gTTS
from moviepy.editor import *
from fpdf import FPDF
import pandas as pd

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
API_KEY = "ADD_KEY_HERE"

def clear_folder(folder_path):
    # List all files and directories in the given folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        
        # Check if the path is a file
        if os.path.isfile(file_path):
            # Remove the file
            os.remove(file_path)
        elif os.path.isdir(file_path):
            # Recursively clear the subdirectory
            clear_folder(file_path)
            # Remove the directory
            os.rmdir(file_path)


def fetch_story_from_excel(image_id):
    # Load the Excel file
    excel_file_path = 'Stories for StoryQuest.xlsx'  # Update with the actual path to your Excel file
    df = pd.read_excel(excel_file_path)
    
    # Assuming the Excel file has columns 'Image_ID' and 'Story', you would filter based on 'Image_ID'
    filtered_df = df[df['Image_ID'] == image_id]
    
    # Assuming there's only one story per image_id, you can return the first result
    if not filtered_df.empty:
        story = filtered_df.iloc[0]['Story']
        return story
    else:
        # Handle case where no story is found for the given image_id
        return "Story not found"

def generate_pdf_with_images(images_folder):
    pdf = FPDF()
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]
    for image_file in image_files:
        pdf.add_page()
        pdf.image(os.path.join(images_folder, image_file), x=10, y=10, w=190)
    pdf_file = 'flipbook.pdf'
    pdf.output(pdf_file)
    return pdf_file

def insert_line_breaks(text, max_length):
    words = text.split()
    lines = []
    current_line = ''
    for word in words:
        if len(current_line + word) <= max_length:
            current_line += word + ' '
        else:
            lines.append(current_line.strip())
            current_line = word + ' '
    if current_line:
        lines.append(current_line.strip())
    return '\n'.join(lines)

def query(payload):
    response = requests.post(API_URL, headers={"Authorization": f"Bearer {API_KEY}"}, json=payload)
    return response.content

def generate_image(text):
    payload = {"inputs": text}
    image_bytes = query(payload)
    image = Image.open(io.BytesIO(image_bytes))
    return image

app.secret_key = 'your_secret_key'  # Replace with a secret key for flashing messages
user_logged_in=False


@app.route("/")
def signin_or_index():
    if user_logged_in:  
        return redirect('/index')
    else:
        return redirect('/signin')


@app.route("/signin", methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='123',
            db='skin'
        )
        cursor = conn.cursor()

        query = "SELECT * FROM customer WHERE email = %s AND password = %s"
        values = (email, password)

        cursor.execute(query, values)
        user = cursor.fetchone()

        cursor.close()
        conn.close()

        if user:
            # Set a variable or session indicating that the user is logged in
            # For simplicity, let's assume a variable named 'user_logged_in'
            user_logged_in = True
            return redirect('/index')
        else:
            flash('ACCOUNT DOES NOT EXIST', 'error')

    return render_template('signin.html')

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='123',
            db='skin'
        )

        if (".in" in email or ".com" in email) and len(password) > 7:

            if any(char in string.punctuation for char in password) != True:
                flash('INCLUDE SPECIAL CHARACTER IN PASSWORD', 'error')

            else:

                cursor = conn.cursor()

                query = "SELECT email FROM customer WHERE email = %s AND password = %s"
                values = (email, password)

                cursor.execute(query, values)
                user = cursor.fetchone()

                cursor.close()
                conn.close()

                if user:
                    flash('ACCOUNT ALREADY EXISTS', 'error')

                else:
                    conn = mysql.connector.connect(
                        host='localhost',
                        user='root',
                        password='1234',
                        db='skin'
                    )
                    cursor = conn.cursor()
                    query = "INSERT INTO customer (email, password) VALUES (%s, %s)"
                    values = (email, password)

                    cursor.execute(query, values)
                    conn.commit()

                    cursor.close()
                    conn.close()

        else:
            flash('INVALID EMAIL OR PASSWORD', 'error')

    return render_template('signup.html')

@app.route('/index', methods=['GET', 'POST'])
def home():
    image_path = None
    clear_folder('static/image')
    clear_folder('static/video')
    clear_folder('static/audio')
    if request.method == 'POST':
        prompt1 = request.form['textInput']
        
        font_path = "PlayfairDisplay-Regular.otf"
        model_path = "saved_model/textsumm"
        tokenizer_path = "saved_model/tokenizer"

        word_count = len(prompt1.split())

        if word_count > 200:
           tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
           model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
           ARTICLE_TO_SUMMARIZE = prompt1
           inputs = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors="pt")

          # Global attention on the first token (cf. Beltagy et al. 2020)
           global_attention_mask = torch.zeros_like(inputs)
           global_attention_mask[:, 0] = 1

           summary_ids = model.generate(inputs, global_attention_mask=global_attention_mask, num_beams=3, max_length=250, min_length=150)
           user_prompt = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        else:
           user_prompt = prompt1


        with open("text.txt", "w") as file:
            file.write(user_prompt.strip())

        with open("text.txt", "r") as file:
            text = file.read()

        paragraphs = re.split(r"[.]", text)


        for index, para in enumerate(paragraphs[:-1]):
            cartoon_image = generate_image(f"'{para.strip()} colored cartoon style'")
            rectangle_width = 1024
            rectangle_height = 150
            rectangle_x1 = (1024 - rectangle_width) // 2
            rectangle_y1 = 1024 - rectangle_height - 20
            rectangle_x2 = rectangle_x1 + rectangle_width
            rectangle_y2 = rectangle_y1 + rectangle_height
            draw = ImageDraw.Draw(cartoon_image)
            initial_font_size = 28
            font = ImageFont.truetype(font_path, initial_font_size)

            max_text_length = 72
            para_with_line_breaks = insert_line_breaks(para, max_text_length)

            draw.rectangle([(rectangle_x1, rectangle_y1), (rectangle_x2, rectangle_y2)], fill="white")
            draw.text((45, 910), para_with_line_breaks, font=font, fill="black")

            filename = f'static/image/{index + 1}.png'
            cartoon_image.save(filename)
            tts = gTTS(text=para, lang='en', slow=False)
            tts.save(f"static/audio/voicecover{index + 1}.mp3")

            audio_clip = AudioFileClip(f"static/audio/voicecover{index + 1}.mp3")
            audio_duration = audio_clip.duration

            image_clip = ImageClip(f"static/image/{index + 1}.png").set_duration(audio_duration)
            clip = image_clip.set_audio(audio_clip)
            video = CompositeVideoClip([clip])

            video.write_videofile(f"static/video/videos{index + 1}.mp4", fps=24)
            final = index + 1

        clips = []
        video_folder = "static/video"
        l_files = sorted(os.listdir(video_folder), key=lambda x: int(x.split("videos")[1].split(".")[0]))

        for file in l_files:

            clip = VideoFileClip(os.path.join(video_folder, file))
            clips.append(clip)

        final_video = concatenate_videoclips(clips, method="compose")
        final_video.write_videofile("static/final_video1.mp4")
        

        return render_template('book.html',value=final)

    return render_template('index.html')

@app.route('/get_story')
def get_story():
    # Retrieve the image_id from the query parameters
        clear_folder('static/image')
        clear_folder('static/video')
        clear_folder('static/audio')
        image_id = request.args.get('image_id')
        model_path = "saved_model/textsumm"
        tokenizer_path = "saved_model/tokenizer"
    # Fetch the story based on the image_id
        story = fetch_story_from_excel(image_id)
        prompt1=story
        font_path = "PlayfairDisplay-Regular.otf"
        word_count = len(prompt1.split())

        if word_count > 200:

           tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
           model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
           ARTICLE_TO_SUMMARIZE = prompt1
           inputs = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors="pt")

          # Global attention on the first token (cf. Beltagy et al. 2020)
           global_attention_mask = torch.zeros_like(inputs)
           global_attention_mask[:, 0] = 1

           summary_ids = model.generate(inputs, global_attention_mask=global_attention_mask, num_beams=3, max_length=250, min_length=150)
           user_prompt = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        else:
           user_prompt = prompt1


        with open("text.txt", "w") as file:
            file.write(user_prompt.strip())

        with open("text.txt", "r") as file:
            text = file.read()

        paragraphs = re.split(r"[.]", text)


        for index, para in enumerate(paragraphs[:-1]):
    # Generate cartoon-style image using Hugging Face API
            cartoon_image = generate_image(f"'{para.strip()} colored cartoon style'")
            rectangle_width = 1024
            rectangle_height = 100
            rectangle_x1 = (1024 - rectangle_width) // 2
            rectangle_y1 = 1024 - rectangle_height - 20
            rectangle_x2 = rectangle_x1 + rectangle_width
            rectangle_y2 = rectangle_y1 + rectangle_height
            draw = ImageDraw.Draw(cartoon_image)
            initial_font_size = 28
            font = ImageFont.truetype(font_path, initial_font_size)

    # Insert line breaks to para if it exceeds the width of the rectangle
            max_text_length = 72
            para_with_line_breaks = insert_line_breaks(para, max_text_length)

            draw.rectangle([(rectangle_x1, rectangle_y1), (rectangle_x2, rectangle_y2)], fill="white")
            draw.text((45, 935), para_with_line_breaks, font=font, fill="black")

            filename = f'static/image/{index + 1}.png'
            cartoon_image.save(filename)
            tts = gTTS(text=para, lang='en', slow=False)
            tts.save(f"static/audio/voicecover{index + 1}.mp3")

            audio_clip = AudioFileClip(f"static/audio/voicecover{index + 1}.mp3")
            audio_duration = audio_clip.duration

            image_clip = ImageClip(f"static/image/{index + 1}.png").set_duration(audio_duration)
            clip = image_clip.set_audio(audio_clip)
            video = CompositeVideoClip([clip])

            video.write_videofile(f"static/video/videos{index + 1}.mp4", fps=24)
            final = index + 1

        clips = []
        video_folder = "static/video"
        l_files = sorted(os.listdir(video_folder), key=lambda x: int(x.split("videos")[1].split(".")[0]))

        for file in l_files:

            clip = VideoFileClip(os.path.join(video_folder, file))
            clips.append(clip)

        final_video = concatenate_videoclips(clips, method="compose")
        final_video.write_videofile("static/final_video1.mp4")
        

        return render_template('book.html',value=final)


@app.route('/browse')
def browse():
    return render_template('browse.html')


@app.route('/video')
def video():
    video_path = "static/final_video1.mp4"  # Replace with the actual path to your video file
    return render_template('video.html', video_path=video_path)


@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    pdf_file_path = generate_pdf_with_images("static/image")
    return send_file(pdf_file_path, as_attachment=True)



if __name__ == '__main__':
    app.run()
