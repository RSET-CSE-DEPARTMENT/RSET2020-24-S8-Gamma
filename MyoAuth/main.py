#segmentation applied during training and verification

#main8.py de 2 second version

#main5 cont and separate login and register

#introducing a new title window and audio

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import serial
import threading
from datetime import datetime
import time
import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib
from PIL import Image, ImageTk  # Import Image and ImageTk from PIL
import pygame  # Import pygame library
import imageio
from ttkthemes import ThemedStyle
from tkinter import font
from scipy.signal import spectrogram
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
#import pyedflib
from scipy.signal import iirfilter, lfilter
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import winsound





class EMGRecorderApp:
    def __init__(self, root, serial_port):
        self.root = root
        self.root.title("MyoAuth")
        self.root.geometry("700x500")  # Set window size
        self.root.iconbitmap(default='myoauthicon.ico')
        self.root.configure(bg='white')

        style = ThemedStyle(self.root)
        style.set_theme("plastik") 

        self.gif_label = None
        self.step = None
        self.user_name = None
        self.initial_option = None

        self.knn_classifier = None
        self.scaler = None

        # Load and display the image
        image_path = 'myoauthicon.png'  # Replace with your image path
        original_image = Image.open(image_path)
        resized_image = original_image.resize((200, 200), Image.LANCZOS)
        self.image = ImageTk.PhotoImage(resized_image)
        self.image_label = tk.Label(self.root, image=self.image,background="white")
        self.image_label.place(relx=0.5, rely=0.4, anchor='center')

        # Create a label for the title
        self.title_label = ttk.Label(self.root, text="MyoAuth", font=("Calibri", 25), foreground='green',background="white")
        self.title_label.place(relx=0.5, rely=0.65, anchor='center')

        # Play audio when the title is shown
        pygame.init()
        pygame.mixer.music.load('myoauthintro.mp3')  # Replace with your audio file path
        pygame.mixer.music.play(0)  # Start playing
        self.fade_out_audio(1000, 0.1)  # Fade in over 3000 milliseconds (3 seconds) with a step of 0.5

        # Schedule the transition to the main window after 2500 milliseconds (5 seconds)
        self.root.after(2500, self.show_main_window)

    def fade_out_audio(self, duration, step):
        current_volume = 2.0
        # Gradually decrease the volume to simulate a fade-out effect
        for i in range(int(duration / step)):
            current_volume -= step
            pygame.mixer.music.set_volume(max(0.1, current_volume))
            pygame.time.delay(int(step))

    def show_main_window(self):
        # Destroy the title label
        self.title_label.destroy()
        self.image_label.destroy()

        self.clear_window()

        # Create buttons for the main window
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#ccc")

        welcome_label = tk.Label(self.root, text="Welcome to MyoAuth", background="white", font=("Helvetica", 15, "bold"), fg="green")
        welcome_label.place(relx=0.5, rely=0.1, anchor='center')

        existing_user_label = tk.Label(self.root, text="Existing User?",background="white", font=("Helvetica", 10))
        existing_user_label.place(relx=0.5, rely=0.3, anchor='center')

        # Create "Login" and "Register" buttons
        self.login_button = ttk.Button(self.root, text="Login", command=self.show_login_window)
        self.login_button.place(relx=0.5, rely=0.37, anchor='center', width=150, height=40)

        new_user_label = tk.Label(self.root, text="New User?",background="white", font=("Helvetica", 10))
        new_user_label.place(relx=0.5, rely=0.6, anchor='center')

        self.register_button = ttk.Button(self.root, text="Register", command=self.show_register_window)
        self.register_button.place(relx=0.5, rely=0.67, anchor='center', width=150, height=40)


        self.serial_port = serial_port
        self.emg_data = {}
        self.data_thread = None
        self.load_classifier()

    def show_login_window(self):
        # Clear the window
        self.clear_window()

        self.initial_option = "login"

        login_label = tk.Label(self.root, text="User Login", background="white", font=("Helvetica", 15, "bold"), fg="green")
        login_label.place(relx=0.5, rely=0.1, anchor='center')

        self.user_label = ttk.Label(self.root, text="Enter Username", background="white", font=("Helvetica", 12))
        self.user_label.place(relx=0.5, rely=0.35, anchor='center')

        self.user_entry = ttk.Entry(self.root, justify="center", font=("Helvetica", 12))
        self.user_entry.place(relx=0.5, rely=0.42, anchor='center', width=200)

        self.verify_button = ttk.Button(self.root, text="Verify", command=self.verify_person)
        self.verify_button.place(relx=0.5, rely=0.55, anchor='center', width=150, height=35)

        self.back_button = ttk.Button(self.root, text="Back", command=self.show_main_window)
        self.back_button.place(relx=0.5, rely=0.65, anchor='center', width=150, height=35)

    def show_register_window(self):
        # Clear the window
        self.clear_window()

        self.initial_option = "register"

        register_label = tk.Label(self.root, text="New User Registration", background="white", font=("Helvetica", 15, "bold"), fg="green")
        register_label.place(relx=0.5, rely=0.1, anchor='center')
        
        self.step = 0
        
        # Create "Record EMG Signals," "Extract Features," and "Train" buttons
        self.record_button = ttk.Button(self.root, text="Record EMG Signals", command=self.start_recording_instructions)
        self.record_button.place(relx=0.5, rely=0.3, anchor='center', width=200, height=40)

        self.preprocess_button = ttk.Button(self.root, text="Extract Features", command=self.feature_extract_data)
        self.preprocess_button.place(relx=0.5, rely=0.4, anchor='center', width=200, height=40)

        self.train_button = ttk.Button(self.root, text="Train", command=self.train_classifier)
        self.train_button.place(relx=0.5, rely=0.5, anchor='center', width=150, height=35)

        self.back_button = ttk.Button(self.root, text="Back", command=self.show_main_window)
        self.back_button.place(relx=0.5, rely=0.6, anchor='center', width=150, height=35)

    def step_counter(self): #to register gestre 5 tyms
        # Add your logic for the step counter here
        # This function will be called when the "Next" button is clicked
        self.step = self.step + 1
        if(self.step<=5):
            self.start_recording()
        else:
            self.show_register_window()

    def start_recording_instructions(self):
        
        self.clear_window()
        self.step=0

        # Create instructions label
        self.instructions_label = ttk.Label(self.root, text="Follow these instructions", font=("Helvetica", 14,"bold"), background="white",foreground="green")
        self.instructions_label.place(relx=0.5, rely=0.1, anchor='center')
        

        self.instructions_label2 = ttk.Label(self.root, text="You will be guided through a series of five steps.", font=("Helvetica", 12), background="white")
        self.instructions_label2.place(relx=0.5, rely=0.2, anchor='center')

        self.instructions_label3 = ttk.Label(self.root, text="During each step, execute a hand gesture for a duration of 4 seconds.", font=("Helvetica", 12), background="white")
        self.instructions_label3.place(relx=0.5, rely=0.3, anchor='center')

        self.instructions_label4 = ttk.Label(self.root, text="It is necessary to maintain uniformity by consistently performing the same gesture at each step.", font=("Helvetica", 12), background="white")
        self.instructions_label4.place(relx=0.5, rely=0.35, anchor='center')

        self.user_label = ttk.Label(self.root, text="Before we start, enter Username", font=("Helvetica", 11), background="white")
        self.user_label.place(relx=0.5, rely=0.45, anchor='center')

        self.user_entry = ttk.Entry(self.root, justify="center", font=("Helvetica", 12),width=30)
        self.user_entry.place(relx=0.5, rely=0.52, anchor='center')

        # Create "Start Recording" button
        self.next_button = ttk.Button(self.root, text="Start Recording", command=self.step_counter)
        self.next_button.place(relx=0.5, rely=0.6, anchor='center')

        self.back_button = ttk.Button(self.root, text="Back", command=self.show_register_window)
        self.back_button.place(relx=0.5, rely=0.7, anchor='center')

    def clear_window(self):
        # Destroy all widgets in the window
        for widget in self.root.winfo_children():
            widget.destroy()

    def start_recording(self):
        if(self.step==1):
            user_name = self.user_entry.get()
            self.user_name = user_name
            if(user_name!=""):
                user_name=user_name+str(self.step)
        else:
            user_name=self.user_name+str(self.step)
        
        print(user_name)
            
        if not user_name:
            tk.messagebox.showinfo("Error", "Please enter Username.")
            self.step=0
            return

        if user_name not in self.emg_data:
            self.emg_data[user_name] = []  # Create an empty list for the user if not present

        self.emg_data.clear()
        self.emg_data[user_name] = []
        self.clear_window()

        self.data_thread = threading.Thread(target=self.record_emg, args=(user_name,))
        self.data_thread.start()

        step_label = tk.Label(self.root, text="", font=("Helvetica", 15,"bold"), background="white",foreground="green")
        step_label.place(relx=0.5, rely=0.1, anchor='center')
        step_label.config(text=f"Step {self.step} / 5")

        countdown_label = tk.Label(self.root, text="", font=("Helvetica", 12), background="white")
        countdown_label.place(relx=0.5, rely=0.2, anchor='center')

        # Start the countdown in a separate thread
        countdown_thread = threading.Thread(target=self.start_countdown, args=(4, countdown_label))
        countdown_thread.start()
    
    def start_countdown(self, seconds, countdown_label):  #to gv 4 secs for each gesture

        gif_path = 'wave2.gif'  # Replace with your GIF path
        self.gif_label = tk.Label(self.root,background="white")
        self.gif_label.place(relx=0.5, rely=0.3, anchor='center')
        self.show_gif(gif_path, 4)  # Display the GIF for 4 seconds

        for i in range(0,seconds,1):
            countdown_label.config(text=f"Recording...  {i} seconds")
            # Define the frequency (in Hz) and duration (in milliseconds) of the beep
            frequency = 1000  # Frequency of the beep (e.g., 1000 Hz)
            duration = 10   # Duration of the beep (e.g., 1000 milliseconds)

            # Generate the beep
            winsound.Beep(frequency, duration)
            time.sleep(1)
        countdown_label.config(text="Recording Complete")

    def show_gif(self, gif_path, duration):
        # Create a label to display the GIF
        self.gif_label = tk.Label(self.root,background="white")
        self.gif_label.place(relx=0.5, rely=0.4, anchor='center')

        # Load the animated GIF
        gif = imageio.mimread(gif_path)
        gif_images = [Image.fromarray(img) for img in gif]

        # Convert the PIL Images to Tkinter-compatible format
        self.gif_photos = [ImageTk.PhotoImage(img) for img in gif_images]

        # Display the animated GIF
        self.animate_gif(0, duration)

    def animate_gif(self, index, duration):
        # Display the current frame of the animated GIF
        self.gif_label.config(image=self.gif_photos[index])

        # Schedule the next frame to be shown after a delay
        if index < len(self.gif_photos) - 1:
            self.root.after(int(duration * 1000 / len(self.gif_photos)), self.animate_gif, index + 1, duration)
        elif self.initial_option=="register":
            # Schedule the static image to be shown after the animated GIF
            self.root.after(int(duration * 160), self.show_static_image, 'tick2.png')
        else:
            pass

    def show_static_image(self, image_path):
        static_image = Image.open(image_path)
        resized_image = static_image.resize((100, 100), Image.LANCZOS)
        static_photo = ImageTk.PhotoImage(resized_image)

        # Replace the GIF label with a static image label
        self.gif_label.config(image=static_photo)
        self.gif_label.image = static_photo

        # Play audio
        pygame.init()
        pygame.mixer.music.load('myoauthintro.mp3')  
        pygame.mixer.music.play(0,0,1)  # Start playing
        
        # Create "Next" button
        if(self.step<5):
            self.next_button = ttk.Button(self.root, text="Next Step", command=self.step_counter)
            self.next_button.place(relx=0.5, rely=0.6, anchor='center')
        else:
            self.next_button = ttk.Button(self.root, text="Finish", command=self.step_counter)
            self.next_button.place(relx=0.5, rely=0.6, anchor='center')


    def record_emg(self, user_name):
        print(user_name)
        with serial.Serial(self.serial_port, 115200, timeout=1) as ser:
            ser.reset_input_buffer()

#---: add timestamp to csv
            start_time = time.time()
            while time.time() - start_time < 4 :
                try:
                    line = ser.readline().decode().strip()
                    if line:
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        emg_value = float(line)
                        self.emg_data[user_name].append((timestamp, emg_value))
                except ValueError as e:
                    print(f"Error parsing data: {e}")
#---
        # Save data to CSV after recording
        self.save_to_csv()

    def save_to_csv(self):
        filename = "all_users_emg_data.csv"

        # Check if the file already exists
        file_exists = os.path.isfile(filename)

        with open(filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write header only if the file doesn't exist
            if not file_exists:
                csv_writer.writerow(["Username", "timestamps", "emgvalues"])

            for user_name, data in self.emg_data.items():
                print(user_name)
                for timestamp, emg_value in data:
                    csv_writer.writerow([user_name, timestamp, emg_value])

        print(f"EMG data saved to {filename}")

    def open_and_apply_filters(self):
        #_____________Converting to edf file_______________
        '''
        # Read the CSV file into a DataFrame
        data = pd.read_csv('all_users_emg_data.csv')

        # Get unique usernames from the 'Username' column
        usernames = data['Username'].unique()

        for username in usernames:
            # Filter data for the current username
            subset_data = data[data['Username'] == username]

            # Initialize EDF writer for the current username
            num_signals = 1  # We only have one signal (EMG values)
            edf_writer = pyedflib.EdfWriter(f'{username}_output.edf', num_signals, file_type=pyedflib.FILETYPE_EDFPLUS)

            # Define channel info for EMG values
            channel_info = {
                'label': 'EMG',  # You can set the label to whatever you prefer
                'dimension': 'uV',
                'sample_rate': 300,  # Sample rate in Hz
                'physical_min': subset_data['emgvalues'].min(),
                'physical_max': subset_data['emgvalues'].max(),
                'digital_min': subset_data['emgvalues'].min(),
                'digital_max': subset_data['emgvalues'].max(),
                'transducer': 'None',
                'prefilter': 'None'
            }

            # Set signal header for EMG values
            edf_writer.setSignalHeader(0, channel_info)

            # Write EMG values to EDF file
            edf_writer.writePhysicalSamples(subset_data['emgvalues'].values.astype(float))

            # Close EDF writer
            edf_writer.close()
        '''
        #___________________________________________----


        input_file = filedialog.askopenfilename(title="Select CSV file for filtering", filetypes=[("CSV files", "*.csv")])

        if not input_file:
            return

        output_file = filedialog.asksaveasfilename(title="Save Filtered Data as", defaultextension=".csv", filetypes=[("CSV files", "*.csv")])

        if not output_file:
            return

        df = pd.read_csv(input_file)

        for user_name, user_data in df.groupby('Username'):
            emg_values = user_data['emgvalues'].tolist()
            #filtered_emg = self.apply_lowpass_filter(emg_values,300,20)

            # Apply highpass filter at 5 Hz
            filtered_data_hp = self.apply_highpass_filter(emg_values, lowcut=5)

            # Apply notch filter at 60 Hz
            filtered_emg = self.notch_filter(filtered_data_hp, f0=60)

            
            # Update DataFrame with filtered EMG values
            df.loc[df['Username'] == user_name, 'emgvalues'] = filtered_emg

            '''
            # Plot the original and filtered signals
            plt.figure(figsize=(10, 6))
            plt.plot(emg_values, label='Original Signal', color='blue')
            plt.plot(filtered_emg, label='Filtered Signal', color='red')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.title('Original vs. Filtered Signal')
            plt.legend()
            plt.grid(True)
            plt.show()
            '''

        # Save the updated DataFrame to CSV after all filtering operations are done
        df.to_csv(output_file, index=False)
           
    def apply_highpass_filter(self, data, lowcut, fs=300.0, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        b, a = butter(order, low, btype='high')
        filtered_data = lfilter(b, a, data)
        return filtered_data

    def notch_filter(self, data, f0, fs=300.0, Q=30.0):
        nyquist = 0.5 * fs
        normal_cutoff = f0 / nyquist
        b, a = iirfilter(2, [normal_cutoff - 1e-3, normal_cutoff + 1e-3], btype='bandstop')
        filtered_data = lfilter(b, a, data)
        return filtered_data
        #return filtered_amplitude
        

     #_______SEGEMENTATION_______________
        
    def segment_and_append_to_csv(self):
        # Load the existing CSV file
        filename = filedialog.askopenfilename(title="Select CSV file segmentaion", filetypes=[("CSV files", "*.csv")])

        #filename = "all_users_filtered_emg_data.csv"
        filename2 = "segmented_emg_data.csv"
        if not os.path.isfile(filename):
            print("CSV file not found.")
            return
        
        df = pd.read_csv(filename)

        # Define window length and overlap
        window_length = 2  # in seconds
        overlap = 1  # in seconds
        sample_rate = 300  

        # Process each user's data separately
        unique_users = df['Username'].unique()
        for user_name in unique_users:
            user_df = df[df['Username'] == user_name]

            emg_data = user_df[['timestamps', 'emgvalues']].values.tolist()
            num_samples = len(emg_data)
            num_windows = int((num_samples - window_length * sample_rate) / (overlap * sample_rate)) + 1

            # Create a DataFrame to store segmented data
            columns = ["Username", "Window Number", "Start Time", "End Time", "emgvalues"]
            segmented_df = pd.DataFrame(columns=columns)

            for window_num in range(num_windows):
                start_index = int(window_num * overlap * sample_rate)
                end_index = int(start_index + window_length * sample_rate)

                window_data = emg_data[start_index:end_index]

                start_time = window_data[0][0]
                end_time = window_data[-1][0]
                emg_values = [emg_value for _, emg_value in window_data]

                # Append data to the DataFrame
                segmented_df.loc[len(segmented_df)] = [user_name, window_num + 1, start_time, end_time, emg_values]

            # Append segmented data to the CSV file
            if os.path.isfile(filename2):
                segmented_df.to_csv(filename2, mode='a', header=False, index=False)
            else:
                segmented_df.to_csv(filename2, index=False)

        print(f"Segmented EMG data appended to {filename2}")

    def feature_extract_data(self):
        self.open_and_apply_filters()
        self.segment_and_append_to_csv()
        input_file = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

        if not input_file:
            return

        output_file = filedialog.asksaveasfilename(title="Save Feature data as", defaultextension=".csv", filetypes=[("CSV files", "*.csv")])

        if not output_file:
            return

        self.extract_features(input_file, output_file)

    def extract_features(self, input_file, output_file):
        df = pd.read_csv(input_file)

        feature_vectors = []
        for index, row in df.iterrows():
            emg_values = eval(row['emgvalues'])  # Convert string to list
            features = self.calculate_features(emg_values)
            user_name = row['Username']
            feature_vectors.append({'Username': user_name[:-1], 'Features': features})

        feature_df = pd.DataFrame(feature_vectors)
        feature_df.to_csv(output_file, index=False)

        print(f"\nFeature vectors saved to {output_file}")

    def calculate_features(self, emg_values):
        # feature extraction time domain
        features = []

        features.append(np.mean(np.abs(emg_values),axis=0)) #mean absolute value
        features.append(np.sum(np.abs(np.diff(emg_values)),axis=0)) #waveform length
        features.append(np.sum(np.diff(np.sign(emg_values),axis=0)!=0,axis=0)/(len(emg_values)-1))
        features.append(skew(emg_values,axis=0))
        features.append(kurtosis(emg_values,axis=0))
        features.append(np.sqrt(np.mean(np.array(emg_values)**2,axis=0))) #root mean sqaure
        features.append(np.sum(np.array(emg_values)**2,axis=0)) #simple square integral

        #frequency domain
        # Add Fourier transform as a new feature
        fourier_transform = np.abs(np.fft.fft(emg_values))
        features.append(np.mean(fourier_transform, axis=0))

        # Frequency centroid
        frequency_bins = len(fourier_transform)
        frequency_values = np.fft.fftfreq(frequency_bins, d=1.0)  # Frequency values corresponding to bins
        features.append(np.sum(frequency_values * fourier_transform, axis=0) / np.sum(fourier_transform, axis=0))

        # Additional frequency domain features
        features.append(np.sum(fourier_transform, axis=0))  # Total energy
        features.append(np.sum(fourier_transform ** 2, axis=0))  # Power
        features.append(np.argmax(fourier_transform, axis=0))  # Dominant frequency index
        features.append(np.mean(frequency_values * fourier_transform, axis=0))  # Mean frequency
        features.append(np.median(frequency_values * fourier_transform, axis=0))  # Median frequency
        features.append(np.std(frequency_values * fourier_transform, axis=0))  # Standard deviation of frequency
        features.append(np.var(frequency_values * fourier_transform, axis=0))  # Variance of frequency

        # Power spectral density (PSD) features
        psd = np.abs(np.fft.fft(emg_values))**2 / len(emg_values)
        features.append(np.sum(psd, axis=0))  # Total power
        features.append(np.mean(psd, axis=0))  # Mean power
        features.append(np.sum(psd[1:], axis=0))  # Exclude DC component for total power
        features.append(np.mean(psd[1:], axis=0))  # Exclude DC component for mean power

        # Spectral entropy
        spectral_entropy = -np.sum(fourier_transform * np.log2(fourier_transform + 1e-10), axis=0)
        features.append(spectral_entropy)
    
        return features
    
    def train_classifier(self):
        # Load feature vectors from the CSV file
        input_file = filedialog.askopenfilename(title="Select Feature Vectors CSV file", filetypes=[("CSV files", "*.csv")])

        if not input_file:
            return

        print("Loading feature vectors...")
        df = pd.read_csv(input_file)
        df['Features'] = df['Features'].apply(lambda x: eval(x))  # Convert string to list

        # 'Username' is the column containing user names
        unique_users = df['Username'].unique()

        # Create testing dataset with one feature vector per user
        testing_data = []
        for user in unique_users:
            user_data = df[df['Username'] == user].head(3)  # Select last 3 row for each user
            testing_data.append(user_data)

        testing_df = pd.concat(testing_data)
        training_df = df.drop(testing_df.index)

        # Extract features and labels from training and testing datasets
        X_train = np.vstack(training_df['Features'])
        y_train = training_df['Username']
        X_test = np.vstack(testing_df['Features'])
        y_test = testing_df['Username']

        print(y_test)

        # Standardize the feature values
        print("Standardizing feature values...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create and train the KNN classifier
        print("Training KNN classifier...")
        self.knn_classifier = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
        self.knn_classifier.fit(X_train_scaled, y_train)

        # Print training accuracy
        training_accuracy = self.knn_classifier.score(X_train_scaled, y_train)
        print(f"Training Accuracy: {training_accuracy}")

        # Print testing accuracy
        testing_accuracy = self.knn_classifier.score(X_test_scaled, y_test)
        print(f"Testing Accuracy: {testing_accuracy}")

        # Predict on the test set
        y_pred = self.knn_classifier.predict(X_test_scaled)

        # Calculate F1 score
        f1 = f1_score(y_test, y_pred, average='weighted')  # You can adjust the 'average' parameter as needed
        print(f"F1 Score: {f1}")

        # Plot training and testing accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(['Training', 'Testing'], [training_accuracy, testing_accuracy], marker='o', linestyle='-')
        plt.title('Training and Testing Accuracy')
        plt.xlabel('Dataset')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()

        # Predict on the test set
        y_pred = self.knn_classifier.predict(X_test_scaled)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Calculate true positives and false positives
        true_positives = np.diag(conf_matrix)
        false_positives = np.sum(conf_matrix, axis=0) - true_positives

        # Plot true positives and false positives
        categories = unique_users  # Assuming unique_users contains the list of user names
        plt.figure(figsize=(10, 6))
        plt.bar(categories, true_positives, label='True Positives')
        plt.bar(categories, false_positives, bottom=true_positives, label='False Positives')
        plt.xlabel('Users')
        plt.ylabel('Count')
        plt.title('True Positives and False Positives')
        plt.legend()
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.show()

        # Predict on the test set
        y_pred = self.knn_classifier.predict(X_test_scaled)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)/3

        # Display confusion matrix using seaborn heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        messagebox.showinfo("Success", "KNN Classifier Trained Successfully!")

        scaler_filename = "standard_scaler.joblib"
        joblib.dump(self.scaler, scaler_filename)
        print(f"Trained scaler saved to {scaler_filename}")

        # Save the trained classifier to a file
        classifier_filename = "knn_classifier.joblib"
        joblib.dump(self.knn_classifier, classifier_filename)
        print(f"Trained classifier saved to {classifier_filename}")

        # Initialize lists to store fpr, tpr, and roc_auc for each class
        all_fpr = []
        all_tpr = []
        all_roc_auc = []

        # Plot ROC curve for each class
        plt.figure(figsize=(8, 6))
        for i in range(len(unique_users)):
            # Convert labels to binary format for the current class
            y_test_binary = label_binarize(y_test, classes=[unique_users[i]])

            # Predict probabilities for the current class
            y_scores = self.knn_classifier.predict_proba(X_test_scaled)

            # Compute ROC curve for the current class
            fpr, tpr, _ = roc_curve(y_test_binary, y_scores[:, i])
            roc_auc = auc(fpr, tpr)

            # Store fpr, tpr, and roc_auc for plotting later
            all_fpr.append(fpr)
            all_tpr.append(tpr)
            all_roc_auc.append(roc_auc)

            # Plot ROC curve for the current class
            plt.plot(fpr, tpr, lw=2, label='ROC curve (class {}) (area = {:0.2f})'.format(unique_users[i], roc_auc))

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

        # Set plot properties
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()


    def load_classifier(self):
        classifier_filename = "knn_classifier.joblib"
        if os.path.exists(classifier_filename):
            self.knn_classifier = joblib.load(classifier_filename)
            print(f"Trained classifier loaded from {classifier_filename}")
            self.load_scaler()  # Load the associated scaler
        else:
            print("No trained classifier found. Train the classifier first.")

    def load_scaler(self):
        scaler_filename = "standard_scaler.joblib"
        if os.path.exists(scaler_filename):
            self.scaler = joblib.load(scaler_filename)
            print(f"Trained scaler loaded from {scaler_filename}")
        else:
            print("No trained scaler found.")


    def verify_person(self):

        if not self.knn_classifier:
            messagebox.showinfo("Error", "Please train the classifier first.")
            return

        user_name = self.user_entry.get()

        if not user_name:
            tk.messagebox.showinfo("Error", "Please enter Username.")
            return
        
        self.clear_window()

        self.data_thread = threading.Thread(target=self.record_emg_for_verification, args=(user_name,))
        self.data_thread.start()

        countup_label = tk.Label(self.root, text="", font=("Helvetica", 12), background="white")
        countup_label.place(relx=0.5, rely=0.2, anchor='center')

        self.verify_countup = threading.Thread(target=self.verify_countdown, args=(4,countup_label))
        self.verify_countup.start()

    def verify_countdown(self, seconds, countup_label):
        gif_path = 'wave2.gif'  # Replace with your GIF path
        self.gif_label = tk.Label(self.root,background="white")
        self.gif_label.place(relx=0.5, rely=0.3, anchor='center')
        self.show_gif(gif_path, 4)  # Display the GIF for 4 seconds

        for i in range(0,seconds,1):
            countup_label.config(text=f"Recording...  {i} seconds")
            # Define the frequency (in Hz) and duration (in milliseconds) of the beep
            frequency = 1000  # Frequency of the beep (e.g., 1000 Hz)
            duration = 10   # Duration of the beep (e.g., 1000 milliseconds)

            # Generate the beep
            winsound.Beep(frequency, duration)
            time.sleep(1)
        
        countup_label.destroy()
        self.gif_label.destroy()
    
    def record_emg_for_verification(self, user_name):
        with serial.Serial(self.serial_port, 115200, timeout=1) as ser:
            ser.reset_input_buffer()

            start_time = time.time()
            emg_values = []

            while time.time() - start_time < 4:  # Record for 4 seconds for verification
                try:
                    line = ser.readline().decode().strip()
                    if line:
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        emg_value = float(line)
                        emg_values.append(emg_value)
                except ValueError as e:
                    print(f"Error parsing data: {e}")

        emg_values = self.apply_highpass_filter(emg_values,300,5)
        emg_values = self.notch_filter(emg_values,60)

        # Segment the filtered EMG data
        window_length = 2  # in seconds
        overlap = 0.05  # in seconds - 50ms
        sample_rate = 300  # Assuming sample rate of 250 Hz
        num_samples = len(emg_values)
        num_windows = int((num_samples - window_length * sample_rate) / (overlap * sample_rate)) + 1

        # Counter to keep track of the number of windows that pass
        pass_count = 0

        # Process each window separately
        for window_num in range(num_windows):
            start_index = int(window_num * overlap * sample_rate)
            end_index = int(start_index + window_length * sample_rate)

            window_emg = emg_values[start_index:end_index]
            #print(window_emg)

            # Extract features from the windowed EMG data (you may need to define this function)
            features = self.calculate_features(window_emg)

            knn_classifier = joblib.load("knn_classifier.joblib")
            scaler = joblib.load("standard_scaler.joblib")
            self.load_classifier()

            # Standardize the feature values
            features_scaled = scaler.transform([features])

            #Predict the person using the trained KNN classifier
            predicted_person = self.knn_classifier.predict(features_scaled)

            # Check if the window passes the authentication threshold
            if predicted_person == user_name:
                pass_count += 1

        self.clear_window()

        self.verification_label = ttk.Label(self.root, text="Verification Result", font=("Helvetica", 14, "bold"), background="white", foreground="green")
        self.verification_label.place(relx=0.5, rely=0.1, anchor='center')

        self.result_label = ttk.Label(self.root, text="", font=("Helvetica", 14, "bold"), background="white", foreground="black")
        self.result_label.place(relx=0.5, rely=0.3, anchor='center')

        self.image_label = tk.Label(self.root, background="white")
        self.image_label.place(relx=0.5, rely=0.47, anchor='center')

        self.back_button = ttk.Button(self.root, text="Back", command=self.show_main_window)
        self.back_button.place(relx=0.5, rely=0.65, anchor='center', width=150, height=35)

        print("Pass count: ",pass_count,"/",num_windows)
        if pass_count >= 15 :  
            image_path = 'tick2.png'
            audio_path = 'myoauthintro.mp3'
            self.result_label.config(text=f"Authentication success. Identified Person: {user_name}")
        else:
            image_path = 'wrong.png'
            audio_path = 'wrong.mp3'
            self.result_label.config(text="Authentication failed")

        original_image = Image.open(image_path)
        resized_image = original_image.resize((100, 100), Image.LANCZOS)
        self.image = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=self.image)

        # Play audio
        pygame.init()
        pygame.mixer.music.load(audio_path)  # Replace with your audio file path
        pygame.mixer.music.play(0, 0, 1)

        
if __name__ == "__main__":
    serial_port = "COM8"  #Serial port for Arduino
    root = tk.Tk()
    app = EMGRecorderApp(root, serial_port)

    root.mainloop() 