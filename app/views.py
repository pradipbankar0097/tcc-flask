# Important imports
from app import app
from flask import request, render_template, jsonify
import os
import pandas as pd
import numpy as np
# import seaborn as sns
# # import matplotlib.pyplot as plt
# import warnings
# import re
# import nltk
# import string
# from app.static import nlp_utils
# from app.static import contractions
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize,sent_tokenize
# from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
# from sklearn import preprocessing
# from sklearn.feature_selection import SelectFromModel

# from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, fbeta_score, confusion_matrix
# from sklearn.metrics import roc_auc_score, roc_curve

# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import MultinomialNB, BernoulliNB
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from nltk import ngrams,bigrams,trigrams
# import imutils
# from PIL import Image
# from PIL.ImageFilter import BoxBlur
# import cv2
# import easyocr
# warnings.filterwarnings('ignore')

# df=pd.read_csv('app/static/train.csv') 
# alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
# punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
# remove_n = lambda x: re.sub("\n", " ", x)
# remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)
# df['comment_text'] = df['comment_text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)
# # Removing special characters
# Insulting_comment_df=df.loc[:,['id','comment_text','insult']]
# # Creating insult dataframe
# IdentityHate_comment_df=df.loc[:,['id','comment_text','identity_hate']]
# # Creating identityhate dataframe
# Obscene_comment_df=df.loc[:,['id','comment_text','obscene']]
# # Creating obscene comment dataframe
# Threatening_comment_df=df.loc[:,['id','comment_text','threat']]
# # Creating threatening dataframe
# Severetoxic_comment_df=df.loc[:,['id','comment_text','severe_toxic']]
# # Creating severtoxic dataframe
# Toxic_comment_df=df.loc[:,['id','comment_text','toxic']]
# # Creating toxic dataframe
# Toxic_comment_balanced_1 = Toxic_comment_df[Toxic_comment_df['toxic'] == 1].iloc[0:5000,:]
# # Selecting only 5000 toxic comments 
# Toxic_comment_balanced_0 = Toxic_comment_df[Toxic_comment_df['toxic'] == 0].iloc[0:5000,:]
# # Selecting only 5000 non toxic comments 
# # Toxic_comment_balanced_1.shape
# # # Shape of Toxic_comment_balanced_1
# # Toxic_comment_balanced_0.shape
# # Shape of Toxic_comment_balanced_0
# # Toxic_comment_balanced_1['toxic'].value_counts()
# # # Value_counts of Toxic_comment_balanced_1
# # Toxic_comment_balanced_0['toxic'].value_counts()
# # Value_counts of Toxic_comment_balanced_0
# Toxic_comment_balanced=pd.concat([Toxic_comment_balanced_1,Toxic_comment_balanced_0])
# ## concatenating toxic and non toxic comments
# Severetoxic_comment_df_1 = Severetoxic_comment_df[Severetoxic_comment_df['severe_toxic'] == 1].iloc[0:1595,:]
# # selecting 1595 values of Severetoxic_comment_df_1
# Severetoxic_comment_df_0 = Severetoxic_comment_df[Severetoxic_comment_df['severe_toxic'] == 0].iloc[0:1595,:]
# # selecting 1595 values of Severetoxic_comment_df_0
# Severe_toxic_comment_balanced=pd.concat([Severetoxic_comment_df_1,Severetoxic_comment_df_0])
# # Concatenating Severetoxic_comment_df_1 and Severetoxic_comment_df_0
# Obscene_comment_df_1 = Obscene_comment_df[Obscene_comment_df['obscene'] == 1].iloc[0:5000,:] 
# Obscene_comment_df_0 = Obscene_comment_df[Obscene_comment_df['obscene'] == 0].iloc[0:5000,:]
# Obscene_comment_balanced = pd.concat([Obscene_comment_df_1,Obscene_comment_df_0])
# Threatening_comment_df_1 = Threatening_comment_df[Threatening_comment_df['threat'] == 1].iloc[0:478,:]
# Threatening_comment_df_0 = Threatening_comment_df[Threatening_comment_df['threat'] == 0].iloc[0:478,:]
# Threatening_comment_balanced = pd.concat([Threatening_comment_df_1,Threatening_comment_df_0])
# Insulting_comment_df_1 = Insulting_comment_df[Insulting_comment_df['insult'] == 1].iloc[0:5000,:]
# Insulting_comment_df_0 = Insulting_comment_df[Insulting_comment_df['insult'] == 0].iloc[0:5000,:]
# Insulting_comment_balanced = pd.concat([Insulting_comment_df_1,Insulting_comment_df_0])
# IdentityHate_comment_df_1 = IdentityHate_comment_df[IdentityHate_comment_df['identity_hate'] == 1].iloc[0:1405,:]
# IdentityHate_comment_df_0 = IdentityHate_comment_df[IdentityHate_comment_df['identity_hate'] == 0].iloc[0:1405,:]
# IdentityHate_comment_balanced = pd.concat([IdentityHate_comment_df_1,IdentityHate_comment_df_0])
# def cv_tf_train_test(dataframe,label,vectorizer,ngram):

# 	# Split the data into X and y data sets
# 	X = dataframe.comment_text
# 	y = dataframe[label]

# 	# Split our data into training and test data 
# 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# 	# Using vectorizer and removing stopwords
# 	cv1 = vectorizer(ngram_range=(ngram), stop_words='english')
	
# 	# Transforming x-train and x-test
# 	X_train_cv1 = cv1.fit_transform(X_train) 
# 	X_test_cv1  = cv1.transform(X_test)	  
	
# 	## Machine learning models   
	
# 	## Logistic regression
# 	lr = LogisticRegression()
# 	lr.fit(X_train_cv1, y_train)
	
# 	## k-nearest neighbours
# 	knn = KNeighborsClassifier(n_neighbors=5)
# 	knn.fit(X_train_cv1, y_train)

# 	## Naive Bayes
# 	bnb = BernoulliNB()
# 	bnb.fit(X_train_cv1, y_train)
	
# 	## Multinomial naive bayes
# 	mnb = MultinomialNB()
# 	mnb.fit(X_train_cv1, y_train)
	
# 	## Support vector machine
# 	svm_model = LinearSVC()
# 	svm_model.fit(X_train_cv1, y_train)

# 	## Random Forest 
# 	randomforest = RandomForestClassifier(n_estimators=100, random_state=50)
# 	randomforest.fit(X_train_cv1, y_train)
	
# 	f1_score_data = {'F1 Score':[f1_score(lr.predict(X_test_cv1), y_test), f1_score(knn.predict(X_test_cv1), y_test), 
# 								f1_score(bnb.predict(X_test_cv1), y_test), f1_score(mnb.predict(X_test_cv1), y_test),
# 								f1_score(svm_model.predict(X_test_cv1), y_test), f1_score(randomforest.predict(X_test_cv1), y_test)]} 
# 	## Saving f1 score results into a dataframe					 
# 	df_f1 = pd.DataFrame(f1_score_data, index=['Log Regression','KNN', 'BernoulliNB', 'MultinomialNB', 'SVM', 'Random Forest'])  

# 	return df_f1

# severe_toxic_comment_cv = cv_tf_train_test(Severe_toxic_comment_balanced, 'severe_toxic', TfidfVectorizer, (1,1))
# severe_toxic_comment_cv.rename(columns={'F1 Score': 'F1 Score(severe_toxic)'}, inplace=True)

# obscene_comment_cv = cv_tf_train_test(Obscene_comment_balanced, 'obscene', TfidfVectorizer, (1,1))
# obscene_comment_cv.rename(columns={'F1 Score': 'F1 Score(obscene)'}, inplace=True)

# threat_comment_cv = cv_tf_train_test(Threatening_comment_balanced, 'threat', TfidfVectorizer, (1,1))
# threat_comment_cv.rename(columns={'F1 Score': 'F1 Score(threat)'}, inplace=True)

# insult_comment_cv = cv_tf_train_test(Insulting_comment_balanced, 'insult', TfidfVectorizer, (1,1))
# insult_comment_cv.rename(columns={'F1 Score': 'F1 Score(insult)'}, inplace=True)

# identity_hatecomment_cv = cv_tf_train_test(IdentityHate_comment_balanced, 'identity_hate', TfidfVectorizer, (1,1))
# identity_hatecomment_cv.rename(columns={'F1 Score': 'F1 Score(identity_hate)'}, inplace=True)

# X = Toxic_comment_balanced.comment_text
# y = Toxic_comment_balanced['toxic']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initiate a Tfidf vectorizer
# tfv = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

# X_train_fit = tfv.fit_transform(X_train)  
# X_test_fit = tfv.transform(X_test)  
# randomforest = RandomForestClassifier(n_estimators=100, random_state=50)

# randomforest.fit(X_train_fit, y_train)
# randomforest.predict(X_test_fit)



# import pickle
# rfmp = open('app/static/rfmp','rb')
# randomforest = pickle.load(rfmp)
# rfmp.close()

# tfvp = open('app/static/tfvp','rb')
# tfv=pickle.load(tfvp)
# tfvp.close()

from easy_object import oopen
randomforest = oopen('app/static/rfmp')
tfv = oopen('app/static/tfvp')





# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'
app.config['EXISTNG_FILE'] = 'app/static/original'
app.config['GENERATED_FILE'] = 'app/static/generated'

# Route to home page
@app.route("/", methods=["GET", "POST"])
def index():
	# Execute if request is get
	if request.method == "GET":
		return render_template("index.html")

	# Execute if reuqest is post
	if request.method == "POST":
				comment1 = [request.form["comment"]]
				comment1_vect = tfv.transform(comment1)
				toxic_percent = randomforest.predict_proba(comment1_vect)[:,1]*100
				return render_template('index.html',param="The comment is {}% Toxic".format(str(toxic_percent[0])[:4]))
	   

@app.route("/forimage", methods=["GET", "POST"])
def forimage():
	# Execute if request is get
	if request.method == "GET":
		# no_image = cv2.imread(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'noimage.jpg'))
		# cv2.imwrite(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'blur.jpg'),no_image)
		return render_template("forimage.html")
	# Execute if request is POST
	# if request.method == "POST":
	# 	# Get uploaded image
	# 	file_upload = request.files['file_upload']
	# 	filename = file_upload.filename
	# 	# Resize and save the uploaded image
	# 	uploaded_image = Image.open(file_upload)
	# 	# black_img = Image.open(os.path.join(app.config['INITIAL_FILE_UPLOADS'],'black.jpg'))
	# 	copy_to_blur = uploaded_image.copy()
	# 	# print(type(uploaded_image))
	# 	# uploaded_image = uploaded_image.convert('RGB')
	# 	uploaded_image.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image.png'))
		
	# 	# Read uploaded image as array
	# 	uploaded_image = cv2.imread(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image.png'))
	# 	# ocr = optical character recognition
	# 	reader = easyocr.Reader(['en'],gpu=True)
	# 	all_data = reader.readtext(uploaded_image)
	# 	found_toxic = False
	# 	for word in all_data:
	# 		# print(word)
	# 		comment1_vect = tfv.transform([word[1],])
	# 		toxic_percent = randomforest.predict_proba(comment1_vect)[:,1]*100
	# 		if toxic_percent[0]>90:
	# 			found_toxic = True
	# 			print(word)
	# 			print('word is '+str(toxic_percent[0])+'% toxic')
	# 			bbox,text,prob = word
	# 			(tl, tr, br, bl) = bbox
	# 			tl = (int(tl[0]), int(tl[1]))
	# 			tr = (int(tr[0]), int(tr[1]))
	# 			br = (int(br[0]), int(br[1]))
	# 			bl = (int(bl[0]), int(bl[1]))
	# 			cv2.rectangle(uploaded_image, tl, br, (0, 255, 0), -1)
	# 			# cv2.imshow("iiiiii",uploaded_image)
	# 			cv2.imwrite(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'blur.jpg'),uploaded_image)
	# 			# copy_to_blur.paste(black_img.resize((int(word[0][2])-int(word[0][0])),int(word[0][3]-int(word[0][1]))),(int(word[0][0]),int(word[0][1])))
	# 			# copy_to_blur.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'result.png'))
	# 			# copy_to_blur.filter(BoxBlur(1)).save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'blur.jpg'))
	# 	if not found_toxic:
	# 		cv2.imwrite(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'blur.jpg'),uploaded_image)
	# 	return render_template('forimage.html',param="result is "+all_data[0][1])
	
@app.route("/api/<comment>", methods = ['GET'])
def toxic_filter(comment):
	comment1 = [comment,]
	comment1_vect = tfv.transform(comment1)
	toxic_percent = randomforest.predict_proba(comment1_vect)[:,1]*100
	result = {
		"comment":str(comment),
		"toxic_percent":str(toxic_percent[0])[:4]
	}
	return jsonify(result)

# Main function
if __name__ == '__main__':
	app.run(debug=True)

