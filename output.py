#do pip install keras to instoll the ocr model.

# We import matplotlib and the Keras-ocr library to process the images and extract text from them.
# Matplotlib helps analyse and create visual representations of data.
import keras_ocr
import matplotlib.pyplot as plt

#Set up a pipeline with Keras-ocr which has a pre-trained text extraction model loaded with pre-trained weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

# Used to read images from folder path to image object
images = [
    keras_ocr.tools.read(img) for img in ['/Users/arihantujjwal/Documents/keras/test2.png','/Users/arihantujjwal/Documents/keras/text.png']
]

# generate text predictions from the images
prediction_groups = pipeline.recognize(images)

# plot the text predictions
fig, axs = plt.subplots(nrows=len(images), figsize=(10, 20))
for ax, image, predictions in zip(axs, images, prediction_groups):
    keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)

#print the identified text from the images
predicted_image = prediction_groups[0]
for text, box in predicted_image:
    print(text)

