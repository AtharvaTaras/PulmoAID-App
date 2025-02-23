import joblib
import random
import torch
from datetime import datetime
import pandas as pd
import streamlit as st
from PIL import Image
from manager import DataManager, LungCancerVGG16Fusion
from gemini import LLM
import os
import shap
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')

# INIT PATHS
train_csv = os.path.join('data', 'selected_data.csv')
val_csv = os.path.join('data', 'selected_data.csv')
logo_img = os.path.join('images', 'logo.png')
arch_img = os.path.join('images', 'architecture.png')

modelpaths = {
	"Logistic Regression": os.path.join('classifiers', 'FusionModel LR_97.41.pkl'),
	"KNN": os.path.join('classifiers', 'FusionModel KNN_73.28.pkl'),
	"Naive Bayes": os.path.join('classifiers', 'FusionModel NB_78.45.pkl'),
	"Random Forest": os.path.join('classifiers', 'FusionModel RFC_91.38.pkl'),
	"XGBoost": os.path.join('classifiers', 'FusionModel XGB_92.24.pkl')
}


# INIT SLICES
feature_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8']
demographic_cols = ['age', 'ethnic', 'gender', 'height', 'race', 'weight']
smoking_hist = ['age_quit', 'cigar', 'cigsmok', 'pipe', 'pkyr', 'smokeage', 'smokeday', 'smokelive', 'smokework', 'smokeyr']
llm_sent = ['llm_sentiment']
clinical = ['biop0', 'bioplc', 'proclc', 'can_scr', 'canc_free_days']
treatments = ['procedures', 'treament_categories', 'treatment_types', 'treatment_days']


if 'layout' not in st.session_state: st.session_state.layout = 'centered'
if 'login' not in st.session_state: st.session_state.login = False
if 'scans' not in st.session_state: st.session_state.scans = []
if 'pil_images' not in st.session_state: st.session_state.pil_images = []
if 'subject' not in st.session_state: st.session_state.subject = 'N/A'
if 'user' not in st.session_state: st.session_state.user = None

st.set_page_config(page_title='PulmoAID', 
				   layout=st.session_state.layout,
				   page_icon='🫁')


def info_tab():
	st.markdown(""" 
# **PulmoAID**
*Enabling AI-based diagnostics for lung cancer using advanced multimodal feature fusion approach.*
""".strip())

	# st.subheader(""" 
	# PulmoAID -  Enabling AI-based diagnostics for lung cancer using advanced multimodal feature fusion approach.
	# """.strip())

	st.subheader('Data Summary Statistics')
	st.code(body=""" 
DATA SUMMARY STATISTICS 

Training and Testing Dataset
	1A (Positive Patients) - 310
	1G (Negative Patients) - 269
	
Validation (Field Testing) Dataset - 
	2A (Positive Patients) - 312
	2G (Negative Patients) - 184
		""".strip(), language=None)

	st.subheader('Model Architecture Summary')
	st.image(image=arch_img)
	st.write(""" 
This model integrates multimodal feature fusion to detect lung cancer from CT scan images. 
It employs a pretrained VGG-16 network for feature extraction, capturing deep spatial representations from the input images. 
These extracted features are then processed through a fully connected neural network (FCNN), 
serving as a fusion layer to integrate and refine the learned representations. 
Finally, the fused features are passed to a logistic regression classifier, which performs binary classification to 
predict the likelihood of lung cancer. This architecture effectively combines deep learning-based feature
extraction with traditional classification techniques to enhance diagnostic accuracy.
	""".strip())

	st.subheader('LLM Integration')

	st.image(os.path.join('images', 'gemini_logo.jpg'))
	st.write(''' 
Gemini 1.5 Flash's speed and efficiency allows rapid analysis of patient data. 
Its large context window allows for the simultaneous processing of comprehensive patient histories, potentially identifying subtle patterns and correlations that might be missed by human clinicians. 
This could lead to faster diagnoses, especially in time-sensitive situations like emergency medicine, and enable cost-effective screening for widespread conditions. 
The model's multimodal reasoning capabilities could also integrate diverse data sources, such as genetic information and lifestyle factors, to provide more holistic and personalized diagnostic insights, improving accuracy and efficiency in healthcare workflows.

The model was used to induce a synthetic varible (llm_sentiment) to act as a doctor's sentiment/score for cancer likeliness to improve model accuracy.
It was prompted to generate a score between 0-10 based on patient's cancer history to add a third modality to the classifier's prediction.
This LLM also serves as a context-aware question answering chatbot/virtual doctor to interact with the clinical data.
		''')	
	st.subheader('Multimodal Fusion Model Metrics')
	st.image(os.path.join('images', 'fusion_metrics.png'))

	st.subheader('Citations and Sources')



@st.cache_resource
def load_llm():
	loaded_llm = LLM(st.secrets["keys"]["api"])
	return loaded_llm


@st.cache_resource
def utilloader(utility:str):
	if utility == 'llm':
		loaded_llm = LLM(st.secrets["keys"]["api"])
		return loaded_llm
	
	if utility == 'manager':
		torch.manual_seed(0)
		device = torch.device('cpu')
		VGG_16 = LungCancerVGG16Fusion().to(device)
		modelpath = os.path.join("models", "best_vgg16.pth")
		VGG_16.load_state_dict(torch.load(modelpath, weights_only=True, map_location=device))
		VGG_16.eval()

		return DataManager(VGG_16)
	
	if utility == 'subject_list':
		data = pd.read_csv(train_csv)
		return list(data['Subject'])

	if utility == 'classifier_csv':
		return pd.read_csv(os.path.join("data", "selected_data.csv")) 
	
	if utility == 'llm_csv':
		return pd.read_csv(os.path.join("data", "selected_descriptive_data.csv"))


@st.cache_resource
def load_classifier(name:str):
	return joblib.load(modelpaths[name])


@st.cache_resource
def generate_outcome(features=[], subject='', classifier='', full_row=None) -> str:
	global csvdata, feature_cols

	row = csvdata[csvdata['Subject'] == int(subject)]
	# newrow = features + row[demographic_cols + smoking_hist + llm_sent].values.flatten().tolist()
	newrow = row[feature_cols + demographic_cols + smoking_hist + llm_sent].values.flatten().tolist()
	model = load_classifier(classifier)

	if full_row is not None:
		try:
			outcome = model.predict_proba(full_row)
			probability_negative = outcome[0][0] * 100
			probability_positive = outcome[0][1] * 100

			if probability_negative > probability_positive:
				result = f"""
				✅ **Subject `{subject}` has tested _Negative_.**  
				- **Confidence:** `{probability_negative:.2f}%`
				"""
			else:
				result = f"""
				⚠️ **Subject `{subject}` has tested _Positive_.**  
				- **Confidence:** `{probability_positive:.2f}%`
				"""

		except AttributeError:
			outcome = model.predict(full_row)
			result = f"""
			🧪 **Subject `{subject}` has tested:**  
			**{"🟢 Negative" if int(outcome[0]) == 0 else "🔴 Positive"}**
			"""

		return result

	try:
		outcome = model.predict_proba([newrow])
		probability_negative = outcome[0][0] * 100
		probability_positive = outcome[0][1] * 100

		if probability_negative > probability_positive:
			result = f"""
			✅ **Subject `{subject}` has tested _Negative_.**  
			- **Confidence:** `{probability_negative:.2f}%`
			"""
		else:
			result = f"""
			⚠️ **Subject `{subject}` has tested _Positive_.**  
			- **Confidence:** `{probability_positive:.2f}%`
			"""

	except AttributeError:
		outcome = model.predict([newrow])
		result = f"""
		🧪 **Subject `{subject}` has tested:**  
		**{"🟢 Negative" if int(outcome[0]) == 0 else "🔴 Positive"}**
		"""

	return result


# @st.cache_resource
def generate_shap_plot(base: pd.DataFrame, subject: str):
	# np.random.seed(0)
	# model = load_classifier('XGBoost')

	# features = ['n1', 'n2', 'n3', 'n4',
	# 			'age', 'ethnic', 'gender', 'height', 'race', 'weight',
	# 			'age_quit', 'cigar', 'cigsmok', 'pipe', 'pkyr', 'smokeage', 'smokeday',
	# 			'smokelive', 'smokework', 'smokeyr']
	# X = base[features]
	# y = base['lung_cancer']

	# # Fit model before SHAP calculation
	# model.fit(X, y)
	
	# subject_index = base[base['Subject'] == int(subject)].index
	# explainer = shap.TreeExplainer(model)
	# shap_values = explainer.shap_values(X)
	# subject_shap_values = shap_values[subject_index]

	# plt.figure(figsize=(12, 8))
	# shap.summary_plot(shap_values, X, max_display=20, show=False)

	# # Get feature importance order
	# mean_abs_shap = np.abs(shap_values).mean(axis=0)
	# feature_importance_order = np.argsort(-mean_abs_shap)[:20]
	
	# # Plot subject points
	# for i, idx in enumerate(feature_importance_order):
	# 	plt.scatter(subject_shap_values[0][idx], i, color='black', edgecolor='white', s=50, zorder=3)

	# plt.title("SHAP Summary Plot with Subject Highlighted")
	# plt.tight_layout()
	
	# return plt

	shap_path = os.path.join('shap_plots', f'{subject}.png')
	image = Image.open(shap_path)

	return image


def doctor_page():
	global csvdata, llmdata
	
	with st.sidebar:
		st.header(body='Doctor Portal')
		st.divider()
		st.write(datetime.now().strftime("%Y-%M-%d %H:%M"))
		st.write(f'Welcome Dr.Tushar')

		logout = st.button(label='Logout', use_container_width=True)
		if logout:
			st.session_state.login = False
			st.session_state.messages = []
			st.rerun()

		st.session_state.subject_selection = st.selectbox(label='Patient ID', options=st.session_state.subject_list)
		st.session_state.model_selection = st.selectbox(label='Classifier', options=list(modelpaths.keys()))

		clinical_data = st.toggle(label='Clinical Data')
		demographic_data = st.toggle(label='Demographic Data')
		smoking_history = st.toggle(label='Smoking History')

		# doctor_notes = st.text_area(label='Doctor\'s Notes')

	st.image(image=logo_img, use_container_width=True)

	information, images_clinical, diagnostics, ai = st.tabs(['Information', 'Images and Clinical', 'Diagnostics', 'Talk To AI'])

	with information:
		info_tab()

	with images_clinical:
		uploaded_files = st.file_uploader(label='Upload Scans', accept_multiple_files=True, type=["jpg", "jpeg", "png"])

		if uploaded_files != []: 
			tmpset = set()

			for file in uploaded_files:
				name = file.name
				tmpset.add(name.split('_')[0])

			if len(tmpset) > 1:
				st.warning('Input files are of different subjects, please give images for one subject only.')

			tmp_current = str(tmpset.pop())

		cl1, cl2 = st.columns(2)

		with cl1:
			show_scan = st.toggle(
				label='Show CT Scan (Upload CT Scans First)' if uploaded_files == [] else "Show CT Scan",
				disabled=True if uploaded_files == [] else False)
			
			if show_scan:
				st.image(uploaded_files[0], caption='CT Scan')

		with cl2:
			show_shap = st.toggle(
			label='Generate SHAP Plot (Upload CT Scans First)' if uploaded_files == [] else "Generate SHAP Plot", 
			disabled=True if uploaded_files == [] else False)

			if show_shap:
				st.image(os.path.join('shap_plots', f'{tmp_current}.png'))
				

		patient_obs = st.text_area(label='Patient\'s Observations')
		save_obs = st.button(label='Save Observations', use_container_width=True)


	with diagnostics:
		st.write(""" 
		Comparison of current analysis with the last diagnostics in terms of
		probability key factors that are different.
		""".strip())

		submit = st.toggle(
			label='Generate Fusion Model Prediction (Please upload CT Scans First)' if uploaded_files == [] \
					else "Generate Fusion Model Prediction", 
			disabled=True if uploaded_files == [] else False)

		if uploaded_files != [] and submit:
			nameset = set()

			for file in uploaded_files:
				name = file.name
				nameset.add(name.split('_')[0])
				try:
					image = Image.open(file).convert("RGB")
					st.session_state.pil_images.append(image)

				except Exception as e:
					st.error(f"Error processing '{file.name}': {e}")
				
			if len(nameset) > 1:
				st.warning('Input files are of different subjects, please give images for one subject only.')

			else:
				current_subject = nameset.pop()

				if str(current_subject) != str(st.session_state.subject_selection):
					st.warning('Warning! Uploaded image ID(s) do not match with selected subject. Please choose correct subject in sidebar.')

				with st.spinner(text='Running Model...'):
					features = Manager.extract_features(imagelist=st.session_state.pil_images)
					outcome = generate_outcome(features, current_subject, st.session_state.model_selection)
					st.markdown(outcome)

		else:
			st.session_state.selected_subject = st.session_state.subject_selection
		
		edited_data = {}
		original_columns = csvdata.columns.tolist()
		c1, c2, c3 = st.columns(3)
	
		original = csvdata[csvdata['Subject'] == int(st.session_state.selected_subject)]

		def process_data(section_name, columns):
			"""Handles editing and storing modified data while preserving structure."""
			slice_df = csvdata[['Subject'] + columns]
			data_df = slice_df[slice_df['Subject'] == int(st.session_state.subject_selection)].T
			data_df.columns = ['Value']

			# Editable dataframe
			edited_df = st.data_editor(data_df, use_container_width=True)

			# Store edited values while avoiding duplicate 'Subject' columns
			edited_df = edited_df.T
			edited_df = edited_df.drop(columns=['Subject'], errors='ignore')
			edited_df.insert(0, 'Subject', st.session_state.subject_selection)  # Ensure 'Subject' is the first column
			
			edited_data[section_name] = edited_df

		with c1:
			if clinical_data:
				st.write('Clinical Data')
				process_data('Clinical', clinical)

		with c2:
			if demographic_data:
				st.write('Demographic Data')
				process_data('Demographic', demographic_cols)

		with c3:
			if smoking_history:
				st.write('Smoking History')
				process_data('Smoking History', smoking_hist)

		if edited_data:
			final_edited_df = pd.concat(edited_data.values(), axis=1)

			# Remove duplicate columns (keeping the first occurrence)
			final_edited_df = final_edited_df.loc[:, ~final_edited_df.columns.duplicated()]
			final_edited_df = final_edited_df.reindex(columns=original_columns, fill_value=None)
			final_edited_df = final_edited_df.fillna(original.set_index('Subject').loc[st.session_state.selected_subject])

			# st.write("Edited Data (Preserved Column Order, No Missing Values):")
			# st.dataframe(final_edited_df)

			new_pred = st.toggle('Generate New Prediction')
			if new_pred:
				new_X = final_edited_df[feature_cols + demographic_cols + smoking_hist + llm_sent]
				new_results = generate_outcome(subject=st.session_state.selected_subject, classifier=st.session_state.model_selection, full_row=new_X)
				st.markdown(new_results)

		notes = st.text_area('Doctor\'s Notes')
		save = st.button('Save/Update Notes', use_container_width=True)


	with ai:

		if 'llm' not in st.session_state:
			st.session_state.llm = load_llm()


		st.session_state.llm.set_prompt(fr'''
You are an intelligent AI mdeical assistant.
Refer to the patient data given below (patient is referred to as "Subject"). It is related to a lung cancer study.
					   
{llmdata[llmdata['Subject'] == int(st.session_state.subject_selection)][demographic_cols + smoking_hist + clinical + llm_sent + ['lung_cancer']].to_dict(orient='records')}

Some fields that do have a clear description are described for your understanding below - 
bioplc - Had a biopsy related to lung cancer?
biop0 - Had a biopsy related to positive screen?
proclc - Had any procedure related to lung cancer?
can_scr - Result of screen associated with the first confirmed lung cancer diagnosis Indicates whether the cancer followed a positive negative, or missed screen, or whether it occurred after the screening years.
0="No Cancer", 1="Positive Screen", 2="Negative Screen", 3="Missed Screen", 4="Post Screening"
canc_free_days - Days until the date the participant was last known to be free of lung cancer. 
llm_sentiment - AI generated sentiment variable for cancer likeliness from 0 - 10.
lung_cancer - Actual clinical test outcome for lung cancer (0 = negative, 1 = positive)

Based on this data, a doctor will be interacting with you and ask you some questions. Answer these questions. 
Answer them as per your knowledge and understanding. Keep your answers highly verbose and descriptive.
If any question is unrelated to lung cancer or the medical field in general, respectfully decilne to answer that question.
''')
		
		if "messages" not in st.session_state: st.session_state.messages = []
		prompt = st.chat_input(placeholder='Summarize this patient...')

		for message in st.session_state.messages:
			with st.chat_message(message["role"]):
				st.markdown(message["content"])

		if prompt:
			st.chat_message("user").markdown(prompt)
			st.session_state.messages.append({"role": "user", "content": prompt})
			response = st.session_state.llm.ask(prompt)
			st.session_state.messages.append({"role": "assistant", "content": response})

			with st.chat_message("assistant"):
				st.markdown(response)


def patient_page(patient_id:str):
	global csvdata, llmdata

	st.session_state.subject = patient_id
	st.image(image=logo_img, use_container_width=True)

	with st.sidebar:
		st.header(body='Patient Portal')
		st.divider()
		st.write(datetime.now().strftime("%Y-%M-%d %H:%M"))
		st.write(f'Welcome {st.session_state.subject}')

		show_hist = st.toggle('Show History')
		show_reports = st.toggle('View Results')

		logout = st.button(label='Logout', use_container_width=True)

		if logout:
			st.session_state.messages = []
			st.session_state.login = False
			st.rerun()

	st.title('Patient Dashboard')
	st.divider()

	info, diagnostics, history, ai = st.tabs(['Information', 'My Diagnostics', 'My History', 'Talk To VDoctor'])

	with info:
		info_tab()

	with diagnostics:
		if show_reports:
			# shap = generate_shap_plot(base=csvdata, subject=patient_id)
			# st.pyplot(fig=shap, use_container_width=True)
			st.image(os.path.join('shap_plots', f'{patient_id}.png'))

		notes = st.text_area('Doctor\'s Notes')
		observations = st.text_area('My Observations')

		st.button(label='Save Observations', use_container_width=True, disabled=True if observations=='' else False)

	with history:
		if show_hist:
			col1, col2 = st.columns(2)

			with col1:
				st.write('Demographic History')
				row_dm = csvdata[csvdata['Subject'] == int(st.session_state.subject)]
				slice_dm = row_dm[demographic_cols].T  
				slice_dm.columns = ['Data'] 
				st.dataframe(data=slice_dm, use_container_width=True)

			with col2:
				st.write('Smoking History')
				row_sm = csvdata[csvdata['Subject'] == int(st.session_state.subject)]
				slice_sm = row_sm[smoking_hist].T  
				slice_sm.columns = ['Data'] 
				st.dataframe(data=slice_sm, use_container_width=True)


	with ai:
		if 'llm' not in st.session_state:
			st.session_state.llm = load_llm()

		st.session_state.llm.set_prompt(f'''
You are a helpful AI assistant. Your task is to respond to the patient's queries to the best of your knowledge.
Refer to patient info given below.
{llmdata[llmdata['Subject'] == int(patient_id)][demographic_cols + smoking_hist + clinical + llm_sent + treatments + ['lung_cancer']].to_dict(orient='records')}
Answer them as per your knowledge and understanding. Keep your answers highly descriptive.
If any question is unrelated to the data, respectfully decilne to answer that question stating your reason.
Give suggestions for diagnosis and treatment based on these columns - treament_categories, treatment_types, treatment_days
If any patient tested negative (lung_cancer == 0), that means they do no need any further treatment.
'''.strip())

		if "messages" not in st.session_state: st.session_state.messages = []
		prompt = st.chat_input(placeholder='What treatment is suggected?')

		for message in st.session_state.messages:
			with st.chat_message(message["role"]):
				st.markdown(message["content"])

		if prompt:
			st.chat_message("user").markdown(prompt)
			st.session_state.messages.append({"role": "user", "content": prompt})
			response = st.session_state.llm.ask(prompt)
			st.session_state.messages.append({"role": "assistant", "content": response})

			with st.chat_message("assistant"):
				st.markdown(response)


def main():
	if not st.session_state.login:
		st.image(image=logo_img, use_container_width=True)
		st.title('PulmoAID Login')
		
		username = st.text_input(label='Username/Patient ID')
		password = st.text_input(label='Password', type='password')
		col1, col2 = st.columns(2)
		
		with col1:
			if st.button("Doctor", use_container_width=True):
				st.session_state.user = "Doctor"
		
		with col2:
			if st.button("Patient", use_container_width=True):
				st.session_state.user = "Patient"
		
		if username and password and st.session_state.user:
			if st.session_state.user == "Doctor" and username == st.secrets["keys"]["username"] and password == st.secrets["keys"]["password"]:
				st.session_state.login = True
				st.session_state.user = "Doctor"
				# st.rerun()

			elif st.session_state.user == "Patient" and username.strip().isnumeric():
				tmp = int(username.strip())
				
				if tmp in st.session_state.subject_list and password == st.secrets["keys"]["password"]:
					st.session_state.login = True
					st.session_state.subject = username.strip()
					# st.rerun()

				else:
					st.error("Invalid Patient ID or password")
			else:
				st.error("Invalid credentials or user type selection!")
	
	elif st.session_state.login and st.session_state.user == "Doctor":
		doctor_page()
	
	elif st.session_state.login and st.session_state.user == "Patient":
		patient_page(st.session_state.subject)


if __name__ == "__main__":
	csvdata = utilloader('classifier_csv')
	llmdata = utilloader('llm_csv')
	Manager = utilloader('manager')

	st.session_state.subject_list = list(csvdata['Subject'])
	# st.session_state.login = True
	# st.session_state.user = 'Doctor'
	# patient_page('100158')
	
	main()