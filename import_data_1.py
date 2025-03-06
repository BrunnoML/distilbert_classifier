# import_data_1.py
import os
import gradio as gr
import pandas as pd
from clean_2 import limpar_texto, tratar_natureza
import uuid

OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUTS_DIR, "dados_tratados.xlsx")

def processar_dados(uploaded_file):
    
    # 1. Ler o arquivo enviado
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        if file_ext == ".csv":
            novo_df = pd.read_csv(uploaded_file)
        else:
            novo_df = pd.read_excel(uploaded_file)
    except Exception as e:
        return f"Erro ao ler o arquivo: {e}", ""
    
    # 2. Verificar se as colunas necessárias existem
    if "historico" not in novo_df.columns or "natureza" not in novo_df.columns:
        return "Colunas 'historico' e/ou 'natureza' não encontradas.", ""
    
    # 3. Remover linhas vazias ou com apenas espaços
    novo_df.dropna(subset=["historico", "natureza"], inplace=True)
    novo_df = novo_df[novo_df["historico"].str.strip() != ""]
    novo_df = novo_df[novo_df["natureza"].str.strip() != ""]
    
    if novo_df.empty:
        return "Nenhuma linha válida após remover linhas vazias ou nulas.", ""
    
    # 4. Aplicar o tratamento
    novo_df["historico_limpo"] = novo_df["historico"].apply(limpar_texto)
    novo_df["natureza_tratada"] = novo_df["natureza"].apply(tratar_natureza)
    
    # 5. Gerar ID para as linhas que não tiverem (para novos dados, geralmente não terão)
    if "id" not in novo_df.columns:
        novo_df["id"] = [str(uuid.uuid4()) for _ in range(len(novo_df))]
    
    # 6. Se o arquivo DATA_FILE já existir, carrega os dados existentes e concatena
    if os.path.exists(OUTPUT_FILE):
        try:
            df_existente = pd.read_excel(OUTPUT_FILE)
            # Opcional: remover duplicatas se necessário (por exemplo, com base em 'id' ou 'historico_limpo')
            df_completo = pd.concat([df_existente, novo_df], ignore_index=True)
            # Exemplo: remover duplicatas baseadas no histórico limpo
            df_completo.drop_duplicates(subset=["historico_limpo"], keep="last", inplace=True)
        except Exception as e:
            return f"Erro ao carregar dados existentes: {e}", ""
    else:
        df_completo = novo_df

    # 7. Salvar o DataFrame combinado
    try:
        df_completo.to_excel(OUTPUT_FILE, index=False)
    except Exception as e:
        return f"Erro ao salvar o arquivo: {e}", ""
    
    return "Arquivo processado e atualizado com sucesso!", df_completo.head().to_html()


interface = gr.Interface(
    fn=processar_dados,
    inputs=gr.File(label="Selecione um arquivo Excel ou CSV"),
    outputs=[gr.Textbox(label="Mensagem"), gr.HTML(label="Visualização dos Dados Processados")],
    title="Importação e Processamento de Dados",
    description="Envie um arquivo com as colunas 'historico' e 'natureza' (a primeira linha deve conter os nomes das colunas e os dados iniciam na segunda linha)."
)

if __name__ == "__main__":
    interface.launch()
