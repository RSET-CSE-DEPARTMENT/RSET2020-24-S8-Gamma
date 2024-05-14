import numpy as np
with open('/content/drive/MyDrive/visual-sentiment-analysis/predictions.csv', 'r') as file:
    negative = []
    neutral = []
    positive = []
    for line in file:
        values = line.strip().split(',')

        print("Line Content:", line.strip())

        try:
            negative.append(float(values[0]))
            neutral.append(float(values[1]))
            positive.append(float(values[2]))
        except ValueError as e:
            print("Error:", e)

print("Negative:", negative)
print("Neutral:", neutral)
print("Positive:", positive)
print('\n')

weights = np.array([0.5, 0.3, 0.2])  
sentiment_matrix = np.array([negative, neutral, positive])
    # Calculate weighted sum for each sentiment polarity
weighted_sum = np.sum(sentiment_matrix.T * weights, axis=0)

    # Calculate total weight
total_weight = np.sum(weights)

average_sentiment = weighted_sum / total_weight
weight_text_model = 0.7
weight_image_model = 0.3

# Calculate weighted predictions from text model
weighted_predictions_text1 = predicted_labels1 * weight_text_model
weighted_predictions_text2 = predicted_labels2 * weight_text_model
weighted_predictions_text3 = predicted_labels3 * weight_text_model
weighted_predictions_text4 = predicted_labels4 * weight_text_model
weighted_predictions_text5 = predicted_labels5 * weight_text_model
weighted_predictions_text6 = predicted_labels6 * weight_text_model

# Calculate weighted predictions from image model
weighted_predictions_image1 = average_sentiment * weight_image_model
weighted_predictions_image2 = average_sentiment * weight_image_model
weighted_predictions_image3 = average_sentiment * weight_image_model
weighted_predictions_image4 = average_sentiment * weight_image_model
weighted_predictions_image5 = average_sentiment * weight_image_model
weighted_predictions_image6 = average_sentiment * weight_image_model

combined_weighted_predictions1 = weighted_predictions_text1 + weighted_predictions_image1
combined_weighted_predictions2 = weighted_predictions_text2 + weighted_predictions_image2
combined_weighted_predictions3 = weighted_predictions_text3 + weighted_predictions_image3
combined_weighted_predictions4 = weighted_predictions_text4 + weighted_predictions_image4
combined_weighted_predictions5 = weighted_predictions_text5 + weighted_predictions_image5
combined_weighted_predictions6 = weighted_predictions_text6 + weighted_predictions_image6

combined_sentiment1 = np.mean(combined_weighted_predictions1)
combined_sentiment2 = np.mean(combined_weighted_predictions2)
combined_sentiment3 = np.mean(combined_weighted_predictions3)
combined_sentiment4 = np.mean(combined_weighted_predictions4)
combined_sentiment5 = np.mean(combined_weighted_predictions5)
combined_sentiment6 = np.mean(combined_weighted_predictions6)

print("Combined Sentiment for Depression:", combined_sentiment1)
print("Combined Sentiment for Anxiety:", combined_sentiment2)
print("Combined Sentiment for Autism:", combined_sentiment3)
print("Combined Sentiment for Bipolar:", combined_sentiment4)
print("Combined Sentiment for BPD:", combined_sentiment5)
print("Combined Sentiment for Schizophrenia:", combined_sentiment6)

maximum = max(combined_sentiment1, combined_sentiment2, combined_sentiment3, combined_sentiment4, combined_sentiment5, combined_sentiment6)
disease = ''
if maximum == combined_sentiment1:
  disease = 'depression'

if maximum == combined_sentiment2:
  disease = 'anxiety'

if maximum == combined_sentiment3:
  disease = 'autism'

if maximum == combined_sentiment4:
  disease = 'bipolar'

if maximum == combined_sentiment5:
  disease = 'bpd'

if maximum == combined_sentiment6:
  disease = 'schizophrenia'