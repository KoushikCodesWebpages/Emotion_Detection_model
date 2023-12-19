# Emotion_Detection_model
The Model was Trained With 50 images and tested with 7 images with an accuracy of 0.73(aprrox 5 out of 7 emotions guessed right)
Here are the way to make ur own model with prset as 50 images,
unforturanety i cant share the model in git hub.
The work Flow:
First All the preset Images are sent into Open cv face cascade and face is detected and cut and trimmed off.
Second The model is built With 4 layers in Sequential model of Tensorflow's Keras module, with the conntection with OneDNN(One Api Deep Neural Network)
Then Using Open cv camcapture, we can test the model,
With Pretense I have saved my Tensorflow model into a .h5 Extension.
Using keras.load module,  The Model can be loaded into a check.py program.
Next While accessing the camcapture, i use the face cascade to detect only faces and send onely one face into the model.
I gave rect shape( BOX ) to Surround my face and Using Predictions i labeled the emotions.
The Emotions are labelled as how they are trained(Alphabetically trained)

#The Model Was Built By Research and Studies,
#But Training and preprocess Was Fully Done By Me.
