import streamlit as st
import joblib
#from joblib import load
from transformers import BertTokenizer
import pandas as pd
import torch
from huggingface_hub import hf_hub_download


@st.cache_resource
def load_model(repo_id, modelname):
    model_path = hf_hub_download(repo_id=repo_id, filename=modelname)
    model = joblib.load(model_path)
    model.eval()

    return model


def prediction(data):
    data = data['TEXTVALUE']
    tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')

    tokens_test = tokenizer.batch_encode_plus(
        data.tolist(),
        max_length=15,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    test_mask = tokens_test['attention_mask']
    test_seq = tokens_test['input_ids']

    with torch.no_grad():
        preds = model(test_seq, test_mask)
        predicted_classes = torch.argmax(preds, dim=1).cpu().numpy()

    return predicted_classes


def load_file():
    uploaded_file = st.file_uploader(label='Выберите csv файл', type='csv')
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data[['TEXTVALUE']]
    else:
        return None


repo_id = "DaBul/Neo_NLP"
modelname = "model.joblib"
model = load_model(repo_id, modelname)

st.title('Система предсказания удовлетворенности сотрудников')
file = load_file()
result = st.button('Предсказать категории удовлетворенности')
if result:
    predictions = prediction(file)
    dataframe = file
    dataframe['Satisfaction'] = predictions
    st.write(dataframe)
