# Real-Time-Interaction-Using-Sign-Language

**TEAM 9 - **


Youtube Demo Link for Continous Sentence Recognition and interaction with Google Assistant - 
https://youtu.be/y_XD36URZlY

LSA64 Dataset Link -
http://facundoq.github.io/unlp/lsa64/

Custom Dataset build by us for Alphabet & Word Recognition -
https://github.com/Nikhilkohli1/Dataset_Colored

CLAAT Document for Project Proposal - 
https://codelabs-preview.appspot.com/?file_id=1MK2AvleCsJf3wozFJJ4B_DMK-1EINUt0kVo4WWjHOqo#0

CLAAT Document for Final Project Report -
https://codelabs-preview.appspot.com/?file_id=16KYWvbN_Fy_lIzGe7PyRzQ5WrtYWNPJCgZeUH1AuGAw#0

AWS EC2 Instance - 
http://ec2-3-81-225-2.compute-1.amazonaws.com:8000/test/



**Steps to Reproduce the code -**

1. If you only want to run predictions using our Trained weights on LSA64 dataset with a 96% test accuracy -

We choose 24 words specified in the Words_Class.csv file out of the 64 for our training set. This model was trained using a CNN - LSTM network
which divides each video into individual frames and then create a sequence of frames as a numpy file to feed into the LSTM network. 

a)Open the Notebook in the working directory - 
 Video_Sign_Language_Prediction_Pipeline_Final.ipynb

b)Place the csv files 'Words_Class.csv' and 'data_file.csv' in the working directory . 
 Place all the numpy files from 'Extracted Feature Sequences' folder in the working directory.

c)Activate a Google Asisstant.

d)Now run the Video_Sign_Language_Prediction_Pipeline_Final.ipynb notebook 


2. If you want to build our model and train it on LSA64 or any other dataset from scratch - 

These Steps will take time to run, on a cloud instance it might take upto 8-10 hours to prepare the entire dataset, extract frames,
form sequence files of each video and then predict the words using LSTM. 

**Getting the data**
a)  First, download the dataset from the link provided for LSA64 into the data folder:

Next, create folders (still in the data folder) with mkdir train && mkdir test && mkdir sequences && mkdir checkpoints.

Now you can run the scripts in the data folder to move the videos to the appropriate place, extract their frames and make the CSV file the rest of the code references. You need to run these in order. Example:

LSA_64_Dataset_Dir_Structure_1.ipynb

LSA64_Extract Frames from Videos_Final.ipynb

 
**Extracting features**
b) Before running the LSTM model, you will need to extract features from the Frames using the CNN. This is done by running extract_features.py
This will create numpy sequence files for each video as sequence of pooled layer outputs. 
Put the below files along side data folder. 

data.py , extractor.py , processor.py 


!python extract_features.py


**Training models**
c) Run the below notebook for building the LSTM or Time Distributed model for training and giving the required input shape.

Sign_Lang_Video_Recognition_Models.ipynb 

**Prediction**
d) For the prediction using the model trained above, follow the same steps described in the 1st point. 
