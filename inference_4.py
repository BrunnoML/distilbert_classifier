# inference_4.py
import os
import pickle
import pandas as pd
from clean_2 import limpar_texto

# Defina o diretório de outputs e o caminho para o modelo
OUTPUTS_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUTS_DIR, "modelo_distilbert.pkl")
# Se você precisar do vectorizer (não usado no DistilBERT approach, mas se houver uso complementar, pode ser carregado)
# VECTORIZER_PATH = os.path.join(OUTPUTS_DIR, "tfidf_vectorizer.pkl")

def carregar_modelo():
    # Carrega o classificador treinado com DistilBERT
    if not os.path.exists(MODEL_PATH):
        print("Modelo não encontrado. Execute o train_3.py primeiro.")
        return None
    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)
    return modelo

def prever_natureza(historico: str) -> str:
    """
    Função para prever a natureza do histórico.
    Aqui, o modelo é um classificador treinado com embeddings do DistilBERT.
    É importante que o histórico passe pelo mesmo pré-processamento usado no treinamento.
    """
    modelo = carregar_modelo()
    if modelo is None:
        return "Modelo não encontrado."

    # Limpa o histórico usando a função de limpeza
    historico_limpo = limpar_texto(historico)
    
    # Em nosso treinamento, usamos DistilBERT para gerar embeddings, mas neste exemplo
    # vamos supor que o modelo treinado é capaz de receber diretamente um vetor de features.
    # Se você salvou um pipeline que inclui a geração de embeddings, basta chamar:
    #   pred = modelo.predict([historico_limpo])
    # Mas, como geralmente precisamos gerar os embeddings, aqui vai um exemplo simplificado:
    
    # OBS: O código abaixo assume que você já tem uma função para gerar embeddings.
    # Se o seu classificador foi treinado usando embeddings, você deve utilizar a mesma função.
    from train_3 import gerar_embedding  # Importa a função de geração de embedding do train_3.py
    
    embedding = gerar_embedding(historico_limpo)
    # Redimensiona para 2D (n_samples, n_features)
    embedding = embedding.reshape(1, -1)
    
    # Faz a predição
    pred = modelo.predict(embedding)
    return f"Natureza prevista: {pred[0]}"

if __name__ == "__main__":
    # Exemplos de históricos para teste
    exemplos = [
        "Ao sair da praça, fui abordado por dois indivíduos armados que exigiram meu celular.",
        "Durante uma ronda, a vítima foi surpreendida por uma abordagem agressiva em plena avenida.",
        "Em sua residência, um indivíduo entrou e furtou objetos de valor sem causar dano físico.",
    ]
    
    for hist in exemplos:
        resultado = prever_natureza(hist)
        print("Histórico:", hist)
        print(resultado)
        print("-" * 50)
