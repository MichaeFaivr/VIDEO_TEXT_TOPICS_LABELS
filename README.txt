VIDEO TO TEXT AND OBJECTS DETECTION IN FRAME

The project VIDEO_TEXT_LABELS_TOPICS aim at providing a toolbox for Video analysis:
The steps are as follows:

HOW TO LAUNCH THE APP?
The project can be run locally by using : python main.py 
from the main directory
Or it can be run online by using: python app.py 
from Video_Analysis_Webapp

INPUT:
A mp4 video with one speaker in English.

OUTPUTS:
Outputs are stored by date et current time in indicated in the files themselves.
Output.1 :  

VIDEO to TOPICS:
- 1. convert video into text / speech 
- 2. convert the text into a matrix of key topics decided by the client:
- 2.a. label each sentence with a topic
- 2.b. summarize each labelled sentence
- 2.c. build the data frame : TODO
- 3. sentiment analysis (LSA)

VIDEO to OBJECTS DETECTION FROM A LIST OF GIVEN LABELS:
- 1. extract 1 or more key images from the video
- 2. extract features from  with landmark and labelling techniques
with features of interest decided by the client


** README Part Video-to-Text: **
- videos are posted by individuals
- each video is processed thru the analysis pipeline
- the text extracted from a video goes thru a pipeline of preprocessing
- the cleaned text is splitted in topics 

** README Part Text-to-Topics: **

** README Part NLU: Natural Language Understanding: **
Conception de la classe et du modèle:
Objective:
- Build a NLU model from a NLP instance

Input:
sentence
Output:
NER analysis (POS tagging)


