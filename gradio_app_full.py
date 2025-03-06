# gradio_app_full.py
import os
import json
import csv
import uuid
import pickle
import pandas as pd
import numpy as np
import torch
import gradio as gr
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from clean_2 import limpar_texto, tratar_natureza

############################################
# CONFIGURAÇÕES E CAMINHOS
############################################
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
DATA_FILE = os.path.join(OUTPUTS_DIR, "dados_tratados.xlsx")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "modelo_distilbert.pkl")
STATUS_PATH = os.path.join(OUTPUTS_DIR, "status_distilbert.json")
NATUREZAS_CSV = "data/naturezas.csv"

# Configuração do DistilBERT
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
device = torch.device("cpu")  # Ou "mps" se disponível no Mac M2

tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_NAME)
model = AutoModel.from_pretrained(DISTILBERT_MODEL_NAME)
model.to(device)
model.eval()

############################################
# 1) FUNÇÃO PARA IMPORTAR E PROCESSAR DADOS
############################################
def processar_dados(uploaded_file):
    """
    Lê um arquivo Excel ou CSV com as colunas 'historico' e 'natureza',
    remove linhas vazias, aplica funções de limpeza e tratamento, gera
    IDs únicos para os registros novos e acumula os dados com os já existentes.
    Salva o DataFrame final em DATA_FILE e retorna uma mensagem e uma pré-visualização.
    """
    # Reinicia o ponteiro, se o objeto tiver o atributo 'file'
    if hasattr(uploaded_file, "file"):
        uploaded_file.file.seek(0)
    
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        # Usa o atributo 'file' se existir, senão usa o próprio objeto
        if file_ext == ".csv":
            df = pd.read_csv(uploaded_file.file if hasattr(uploaded_file, "file") else uploaded_file)
        else:
            df = pd.read_excel(uploaded_file.file if hasattr(uploaded_file, "file") else uploaded_file)
    except Exception as e:
        return f"Erro ao ler o arquivo: {e}", ""
    
    # Verifica se as colunas necessárias existem
    if "historico" not in df.columns or "natureza" not in df.columns:
        return "Colunas 'historico' e/ou 'natureza' não encontradas.", ""
    
    # Remove linhas vazias ou com apenas espaços
    df.dropna(subset=["historico", "natureza"], inplace=True)
    df = df[df["historico"].str.strip() != ""]
    df = df[df["natureza"].str.strip() != ""]
    
    # Aplica o tratamento
    df["historico_limpo"] = df["historico"].apply(limpar_texto)
    df["natureza_tratada"] = df["natureza"].apply(tratar_natureza)
    
    # Gera IDs para os registros novos, se a coluna 'id' não existir
    if "id" not in df.columns:
        import uuid
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    # Se o arquivo DATA_FILE já existir, carrega os dados existentes e concatena
    if os.path.exists(DATA_FILE):
        try:
            df_existente = pd.read_excel(DATA_FILE)
            df_completo = pd.concat([df_existente, df], ignore_index=True)
            # Remove duplicatas com base no 'historico_limpo'
            df_completo.drop_duplicates(subset=["historico_limpo"], keep="last", inplace=True)
        except Exception as e:
            return f"Erro ao carregar dados existentes: {e}", ""
    else:
        df_completo = df

    try:
        df_completo.to_excel(DATA_FILE, index=False)
    except Exception as e:
        return f"Erro ao salvar o arquivo: {e}", ""
    
    return "Arquivo processado e atualizado com sucesso!", df_completo.tail().to_html()

############################################
# 2) FUNÇÕES PARA GERAÇÃO DE EMBEDDINGS
############################################
def gerar_embedding_batch(texto: str) -> np.ndarray:
    """
    Gera embedding para um único texto.
    """
    if not isinstance(texto, str) or not texto.strip():
        return np.zeros(768)
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=128)
    for key, value in inputs.items():
        inputs[key] = value.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding

def gerar_embeddings_para_dataset(texts: list, batch_size: int = 32) -> np.ndarray:
    """
    Gera embeddings para uma lista de textos usando batching.
    Utiliza tqdm para exibir a barra de progresso no terminal.
    """
    embeddings = []
    total = len(texts)
    for i in tqdm(range(0, total, batch_size), desc="Processando batches", unit="batch"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        for key, value in inputs.items():
            inputs[key] = value.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeds)
    return np.array(embeddings)

############################################
# 3) FUNÇÃO PARA TREINAR O MODELO
############################################
def treinar_modelo():
    """
    Carrega o dataset processado, gera embeddings em batches e treina um classificador
    (Logistic Regression) com max_iter=3000 para ajudar na convergência.
    Salva o modelo treinado e atualiza o status de acurácia.
    """
    if not os.path.exists(DATA_FILE):
        return "Arquivo de dados tratados não encontrado.", "**Acurácia Atual do Modelo:** N/A"
    
    df = pd.read_excel(DATA_FILE)
    # Remove linhas inválidas
    df.dropna(subset=["historico_limpo", "natureza_tratada"], inplace=True)
    df = df[(df["historico_limpo"].str.strip() != "") & (df["natureza_tratada"].str.strip() != "")]
    
    if df.empty:
        return "Nenhum registro válido para treinamento.", "**Acurácia Atual do Modelo:** N/A"
    
    texts = df["historico_limpo"].tolist()
    labels = df["natureza_tratada"].tolist()
    
    print("Gerando embeddings para", len(texts), "registros...")
    embeddings = gerar_embeddings_para_dataset(texts, batch_size=32)
    
    X = np.array(embeddings)
    y = np.array(labels)
    
    # Divide em treino e teste
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    current_accuracy = f"{acc*100:.2f}%"
    print(f"Acurácia do modelo DistilBERT: {current_accuracy}")
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    
    status_data = {"accuracy": current_accuracy}
    with open(STATUS_PATH, "w") as f:
        json.dump(status_data, f)
    
    return f"Treinamento concluído. Acurácia: {current_accuracy}", f"**Acurácia Atual do Modelo:** {current_accuracy}"

############################################
# 4) FUNÇÃO PARA INFERÊNCIA
############################################
def prever_natureza(historico: str) -> str:
    """
    Limpa o histórico, gera seu embedding e utiliza o modelo treinado para prever a natureza.
    """
    if not historico.strip():
        return "Insira um histórico."
    if not os.path.exists(MODEL_PATH):
        return "Modelo não encontrado. Execute o treinamento primeiro."
    
    # Carrega o modelo
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    
    hist_limpo = limpar_texto(historico)
    emb = gerar_embedding_batch(hist_limpo).reshape(1, -1)
    pred = clf.predict(emb)
    return f"Natureza prevista: {pred[0]}"

def salvar_correcao_inferencia(historico: str, correcao: str) -> str:
    """
    Recebe um histórico e a correção da natureza (como texto) e adiciona
    um novo registro no arquivo DATA_FILE (dados_tratados.xlsx) com os dados processados.
    """
    if not historico.strip():
        return "Nenhum histórico fornecido."
    if not correcao.strip():
        return "Nenhuma correção fornecida."
    
    # Processa o histórico e a correção
    hist_limpo = limpar_texto(historico)
    nat_tratada = tratar_natureza(correcao)
    
    # Gera um ID único para o novo registro
    import uuid
    new_id = str(uuid.uuid4())
    
    new_record = {
        "id": new_id,
        "historico": historico,
        "historico_limpo": hist_limpo,
        "natureza": correcao,  # rótulo original (se desejar)
        "natureza_tratada": nat_tratada
    }
    
    # Se o arquivo de dados tratados já existir, carrega e concatena; caso contrário, cria um novo DataFrame
    if os.path.exists(DATA_FILE):
        try:
            df_existente = pd.read_excel(DATA_FILE)
        except Exception as e:
            return f"Erro ao carregar dados existentes: {e}"
        df_novo = pd.DataFrame([new_record])
        df_completo = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df_completo = pd.DataFrame([new_record])
    
    try:
        df_completo.to_excel(DATA_FILE, index=False)
    except Exception as e:
        return f"Erro ao salvar a correção: {e}"
    
    return f"Correção salva com sucesso! Novo registro adicionado com ID {new_id}."


############################################
# 5) FUNÇÕES PARA VERIFICAR HISTÓRICOS E ALTERAR NATUREZAS
############################################
def carregar_naturezas():
    """
    Lê o arquivo naturezas.csv e retorna duas listas:
    - exibicao_list: nomes amigáveis (ex.: "Roubo a Transeunte")
    - rotulo_list: rótulos internos (ex.: "roubo_a_transeunte")
    """
    if not os.path.exists(NATUREZAS_CSV):
        return [], []
    df_nat = pd.read_csv(NATUREZAS_CSV)
    exibicao_list = df_nat["natureza_exibicao"].tolist()
    rotulo_list = df_nat["natureza_rotulo"].tolist()
    return exibicao_list, rotulo_list

def filtrar_historicos_por_natureza(natureza_exibicao):
    """
    Filtra 'dados_tratados.xlsx' pela 'natureza_tratada' correspondente 
    ao rótulo interno de 'natureza_exibicao'. Retorna uma tabela HTML
    onde o ID aparece, mas sem clique automático.
    """
    exibicoes, rotulos = carregar_naturezas()
    map_exib_to_rot = dict(zip(exibicoes, rotulos))
    
    if not os.path.exists(DATA_FILE):
        return "<p>Arquivo de dados não encontrado.</p>"
    
    df = pd.read_excel(DATA_FILE)
    
    # Se a natureza escolhida não está no mapeamento, retorna mensagem
    if natureza_exibicao not in map_exib_to_rot:
        return f"<p>Natureza '{natureza_exibicao}' inválida ou não encontrada.</p>"
    
    rotulo_interno = map_exib_to_rot[natureza_exibicao]
    
    # Filtra os registros
    df_filtro = df[df["natureza_tratada"] == rotulo_interno]
    if df_filtro.empty:
        return f"<p>Nenhum registro encontrado para '{natureza_exibicao}'.</p>"
    
    # Se não existir a coluna 'id', podemos criá-la
    if "id" not in df_filtro.columns:
        import uuid
        df_filtro = df_filtro.reset_index(drop=True)
        df_filtro["id"] = [str(uuid.uuid4()) for _ in range(len(df_filtro))]
    
    # Cria a tabela sem script e sem onclick
    html = """
<div style="max-height:300px; overflow-y:auto; overflow-x:hidden;">
  <table style="width:100%; border-collapse: collapse; table-layout: fixed;">
    <tr style="background-color:#f2f2f2;">
      <th style="border:1px solid #ccc; padding:8px;">ID</th>
      <th style="border:1px solid #ccc; padding:8px;">Histórico Limpo</th>
      <th style="border:1px solid #ccc; padding:8px;">Natureza Tratada</th>
    </tr>
"""
    for _, row in df_filtro.iterrows():
        _id = row["id"]
        hist = row.get("historico_limpo", "")
        nat = row.get("natureza_tratada", "")
        html += f"""
    <tr>
      <td style="border:1px solid #ccc; padding:8px;">{_id}</td>
      <td style="
          border:1px solid #ccc; 
          padding:8px; 
          white-space: pre-wrap; 
          word-wrap: break-word;
          overflow-wrap: break-word;">
        {hist}
      </td>
      <td style="border:1px solid #ccc; padding:8px;">{nat}</td>
    </tr>
    """
    html += "</table></div>"
    
    return html



def salvar_alteracao_direto(id_registro, nova_natureza):
    """
    Atualiza a natureza_tratada diretamente em dados_tratados.xlsx,
    localizando a linha pelo ID.
    """
    if not id_registro.strip():
        return "Nenhum ID fornecido."
    if not nova_natureza.strip():
        return "Nenhuma nova natureza fornecida."
    
    if not os.path.exists(DATA_FILE):
        return "Arquivo de dados não encontrado."
    
    df = pd.read_excel(DATA_FILE)
    # Localiza a linha com o ID
    mask = (df["id"] == id_registro)
    if not mask.any():
        return f"Registro com ID {id_registro} não encontrado."
    
    # Atualiza a natureza_tratada
    df.loc[mask, "natureza_tratada"] = tratar_natureza(nova_natureza)
    df.to_excel(DATA_FILE, index=False)
    
    return f"Natureza atualizada para '{nova_natureza}' no registro ID {id_registro}."

def cadastrar_nova_natureza(nova_natureza):
    """
    Cadastra uma nova natureza no arquivo naturezas.csv, se não existir.
    """
    if not nova_natureza.strip():
        return "Nenhuma natureza fornecida.", None
    
    nat_rotulo = tratar_natureza(nova_natureza)
    existe = os.path.exists(NATUREZAS_CSV)
    if existe:
        df_nat = pd.read_csv(NATUREZAS_CSV)
    else:
        df_nat = pd.DataFrame(columns=["natureza_rotulo", "natureza_exibicao"])
    
    if nat_rotulo in df_nat["natureza_rotulo"].values:
        return "Natureza já existente!", df_nat["natureza_exibicao"].tolist()
    
    nova_linha = {"natureza_rotulo": nat_rotulo, "natureza_exibicao": nova_natureza.strip()}
    df_nat = pd.concat([df_nat, pd.DataFrame([nova_linha])], ignore_index=True)
    df_nat.to_csv(NATUREZAS_CSV, index=False)
    
    return f"Nova natureza '{nova_natureza}' cadastrada!", df_nat["natureza_exibicao"].tolist()

def criar_aba_verificar_historicos():
    """
    Aba 'Verificar Históricos' com:
    - Dropdown para selecionar a natureza (exibição) e filtrar registros
    - Tabela sem scroll horizontal, expandindo o texto
    - Clique no ID para copiar automaticamente para o campo 'ID do Registro'
    - Dropdown com naturezas existentes para corrigir, ou cadastrar nova natureza
    """
    exibicoes, _ = carregar_naturezas()
    if not exibicoes:
        exibicoes = ["(Nenhuma natureza cadastrada)"]
    
    with gr.TabItem("Verificar Históricos"):
        gr.Markdown("### Verificar e Atualizar Históricos por Natureza")
        
        dropdown_nat = gr.Dropdown(choices=exibicoes, label="Selecione Natureza", value=exibicoes[0], interactive=True)
        btn_filtrar = gr.Button("Carregar Históricos")
        out_html = gr.HTML()
        
        def acao_filtrar(nat_exib):
            return filtrar_historicos_por_natureza(nat_exib)
        
        btn_filtrar.click(fn=acao_filtrar, inputs=dropdown_nat, outputs=out_html)
        
        gr.Markdown("#### Alterar Natureza de um Registro Específico")
        # Usamos placeholder="ID do Registro" para o script JS poder localizar esse campo
        inp_id = gr.Textbox(label="ID do Registro", elem_id="id_registro_input")
        
        # Carrega naturezas existentes para o dropdown de nova natureza
        naturezas_existentes, _ = carregar_naturezas()
        if not naturezas_existentes:
            naturezas_existentes = ["(Nenhuma natureza)"]
        
        dropdown_nova_nat = gr.Dropdown(choices=naturezas_existentes, label="Nova Natureza", interactive=True)
        btn_alterar = gr.Button("Salvar Alteração")
        out_res_alt = gr.Textbox(label="Resultado da Alteração")
        
        def acao_alterar(id_reg, nova_nat):
            return salvar_alteracao_direto(id_reg, nova_nat)
        
        btn_alterar.click(fn=acao_alterar, inputs=[inp_id, dropdown_nova_nat], outputs=out_res_alt)
        
        gr.Markdown("#### Cadastrar Nova Natureza no Sistema")
        inp_cad_nat = gr.Textbox(label="Nova Natureza (Ex.: Furto a Residência)")
        btn_cad = gr.Button("Cadastrar")
        out_cad_msg = gr.Textbox(label="Resultado do Cadastro")
        
        def acao_cad_nova(nova_nat):
            msg, lista_nats = cadastrar_nova_natureza(nova_nat)
            return [msg, gr.update(choices=lista_nats), gr.update(choices=lista_nats)]
        
        btn_cad.click(fn=acao_cad_nova, inputs=inp_cad_nat, outputs=[out_cad_msg, dropdown_nova_nat, dropdown_nat])

############################################
# INTERFACE GRADIO INTEGRADA
############################################
def app_full():
    with gr.Blocks() as demo:
        gr.Markdown("# DistilBERT Classifier - Pipeline Integrado")
        gr.Markdown("Este sistema integra as etapas de importação de dados, treinamento do modelo, inferência e verificação/correção de históricos.")
        
        with gr.Tabs():
            # Aba 1: Importação e Processamento de Dados
            with gr.TabItem("Importar Dados"):
                gr.Markdown("### Importar e Processar Dados")
                file_input = gr.File(label="Selecione um arquivo Excel ou CSV")
                out_msg_import = gr.Textbox(label="Mensagem")
                out_preview = gr.HTML(label="Pré-visualização dos Dados Processados")
                btn_import = gr.Button("Processar Dados")
                btn_import.click(fn=processar_dados, inputs=file_input, outputs=[out_msg_import, out_preview])
            
            # Aba 2: Treinar Modelo
            with gr.TabItem("Treinar Modelo"):
                gr.Markdown("### Treinar Modelo com DistilBERT")
                btn_train = gr.Button("Treinar Modelo")
                out_train = gr.Textbox(label="Resultado do Treinamento")
                out_status = gr.Markdown(label="Acurácia Atual")
                btn_train.click(fn=treinar_modelo, inputs=[], outputs=[out_train, out_status])
            
            # Aba 3: Inferência
            with gr.TabItem("Inferência"):
                gr.Markdown("### Prever a Natureza de um Novo Histórico")
                inp_hist = gr.Textbox(lines=3, label="Digite um histórico")
                btn_predict = gr.Button("Prever")
                out_pred = gr.Textbox(label="Previsão")
                btn_predict.click(fn=prever_natureza, inputs=inp_hist, outputs=out_pred)

                gr.Markdown("#### Se a previsão estiver incorreta, insira a natureza correta abaixo para aprimorar o sistema")
                inp_corr = gr.Textbox(lines=1, label="Correção da Natureza (ex: Roubo a Veículo)")
                btn_corr_infer = gr.Button("Salvar Correção")
                out_corr_infer = gr.Textbox(label="Resultado da Correção")
                btn_corr_infer.click(fn=salvar_correcao_inferencia, inputs=[inp_hist, inp_corr], outputs=out_corr_infer)
            
            # Aba 4: Verificar Históricos
            criar_aba_verificar_historicos()
        
    demo.launch()

if __name__ == "__main__":
    app_full()
