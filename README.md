# Face-Recognition
Face recognition implemented in Tensorflow. It uses a CNN to train on faces which are detected by OpenCV

# Instructions
1) Take a 10 second video of subject's face. So if you want to recognise 3 people (which is what this coded for), take a 10 second videos of each of them.
2) Put your subject's (one of them) video in a folder with preprocess_script.py with.
3) Edit the paths in preprocess_scripts.py to that of the subject. Do this for every subject.
4) Run the preprocess_script.py for each person and it'll extract as many faces it can find. These faces will then be used to train a CNN.
5) Before feeding your data to the CNN, use data.py so it's normalized and label data can be generated. Make sure to edit path names. It has 3 loops because I did it for 3 people.
6) After running data.py, data.pkl will be generated. That file is then used in faceid.py where the CNN is trained.
7) Just run faceid.py and wait until the training is complete.

# Important Points
1) When training cnn in faceid.py make sure you have set save location for model. If the location doesn't exists, all your training will be gone.

2) In faceid.py you'll find 4 parameters:
  a) VERSION: It's used to have different versions of models so you could see which one's better. For now, you don't have to change it but if you do make sure to have different save paths.
  b) TRAIN_MODEL: If this is true, the model will be trained.
  c) SAVE_MODEL: If the is true, the model will be saved after training.
  d) RESTORE_MODEL: If this is used the model will restored from path so you could resume training. Make sure restore paths are correct.
