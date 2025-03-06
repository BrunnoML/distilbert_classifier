# clean_2.py
import re
import unicodedata

def limpar_texto(texto: str) -> str:
    """
    Limpa e padroniza o texto removendo acentuação, convertendo para minúsculas,
    eliminando caracteres especiais e espaços extras. Também remove a expressão 'como:'.
    """
    if not isinstance(texto, str):
        return ""
    
    # Converte para minúsculas
    texto = texto.lower()
    
    # Remove a expressão 'como:' (se for sempre irrelevante)
    texto = texto.replace("como:", "")
    
    # Remove acentuação
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore').decode('utf-8')
    
    # Remove caracteres especiais (mantendo letras, números e espaços)
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    
    # Remove espaços extras
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

def tratar_natureza(natureza: str) -> str:
    """
    Padroniza a coluna 'natureza': converte para minúsculas, remove acentos,
    substitui espaços por underscores, etc. para evitar duplicidade.
    """
    if not isinstance(natureza, str):
        return ""
    
    # Converte para minúsculas
    natureza = natureza.lower()
    
    # Remove acentuação
    natureza = unicodedata.normalize('NFD', natureza)
    natureza = natureza.encode('ascii', 'ignore').decode('utf-8')
    
    # Substitui espaços por underscores
    natureza = re.sub(r'\s+', '_', natureza).strip('_')
    
    return natureza
