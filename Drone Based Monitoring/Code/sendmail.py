import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


password = "password"
from_email = "from@gmail.com" 
to_email = "to@icloud.com"
server = smtplib.SMTP('smtp.gmail.com: 587')
server.starttls()
server.login(from_email, password)

def send_email():
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = "Security Alert"

    message_body = f'ALERT - Intruder has been detected!!'

    message.attach(MIMEText(message_body, 'plain'))
    server.sendmail(from_email, to_email, message.as_string())

send_email()
