# GenderClassification
Gender Classification using GBM 

Files necessary for the working is:
1. feature.pkl
2. genderClassificationV4.sav
3. Predictor.ipynb

Before Running Predictor.ipynb, make sure that the variables - _dict_saved_location_ and _modelGenderClassification_ point out to the correct location using a raw string
to point out to the location where _feature.pkl_ and _genderClassificationV4.sav_ is present respectively.

Output will be a _(1,2)_ array, first and second value representing the chance of input name to be a female and male respectively.

All other files are older files of the models and not necessary for prediction.

genderClassificationV4.ipynb contains the code for training of the model. 
