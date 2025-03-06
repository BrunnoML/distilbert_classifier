# train_3.py
import os
import json
import uuid
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Definições de caminho
OUTPUTS_DIR = "outputs"
DATA_FILE = os.path.join(OUTPUTS_DIR, "dados_tratados.xlsx")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "modelo_distilbert.pkl")
STATUS_PATH = os.path.join(OUTPUTS_DIR, "status_distilbert.json")

# Configuração do modelo DistilBERT
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
device = torch.device("cpu")  # Para usar CPU; se tiver GPU, ajuste conforme necessário

# Carrega o tokenizer e o modelo DistilBERT
tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
model = AutoModel.from_pretrained(DISTILBERT_MODEL_NAME)
model.to(device)
model.eval()  # Modo de inferência

def gerar_embedding(texto: str) -> np.ndarray:
    """
    Gera o embedding do texto usando DistilBERT.
    Utiliza o token [CLS] (primeiro token) como representação.
    """
    if not isinstance(texto, str) or not texto.strip():
        return np.zeros(768)  # Retorna vetor nulo se o texto for vazio
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=128)
    for key, value in inputs.items():
        inputs[key] = value.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Utiliza o embedding do primeiro token ([CLS])
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding

def treinar_modelo():
    # Carrega o dataset processado
    if not os.path.exists(DATA_FILE):
        print("Arquivo de dados tratados não encontrado em", DATA_FILE)
        return
    
    df = pd.read_excel(DATA_FILE)
    # Remove linhas com valores vazios nas colunas essenciais
    df.dropna(subset=["historico_limpo", "natureza_tratada"], inplace=True)
    df = df[(df["historico_limpo"].str.strip() != "") & (df["natureza_tratada"].str.strip() != "")]
    
    if df.empty:
        print("Nenhum registro válido para treinamento.")
        return
    
    # Gera embeddings para cada histórico
    embeddings = []
    labels = []
    total = len(df)
    print("Gerando embeddings para", total, "registros...")
    for idx, row in df.iterrows():
        emb = gerar_embedding(row["historico_limpo"])
        embeddings.append(emb)
        labels.append(row["natureza_tratada"])
    
    X = np.array(embeddings)
    y = np.array(labels)
    
    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treina um classificador (Logistic Regression)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Avalia o modelo
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo DistilBERT: {acc*100:.2f}%")
    
    # Salva o modelo treinado
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    
    # Salva a acurácia no arquivo de status para persistência
    status_data = {"accuracy": f"{acc*100:.2f}%"}
    with open(STATUS_PATH, "w") as f:
        json.dump(status_data, f)
    
    print("Modelo treinado e salvo com sucesso!")
    print("Status salvo em:", STATUS_PATH)

if __name__ == "__main__":
    treinar_modelo()
