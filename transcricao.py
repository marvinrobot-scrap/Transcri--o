import os
import sys
import time

# =====================================================================
# CORREÇÃO CRÍTICA: MAPEAMENTO DINÂMICO DAS DLLs DA NVIDIA (WINDOWS)
# =====================================================================
def configurar_dlls_nvidia():
    """
    Força o Windows a encontrar as DLLs do CUDA 12 (cublas, cudnn) 
    dentda da pasta site-packages do Python.
    """
    print("[INIT] Configurando ambiente GPU e DLLs...")
    
    base_python = sys.prefix
    # Caminhos comuns onde o pip instala as libs da nvidia
    caminhos_possiveis = [
        os.path.join(base_python, "Lib", "site-packages", "nvidia", "cublas", "bin"),
        os.path.join(base_python, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
        os.path.join(base_python, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin"),
    ]

    dl_encontradas = 0
    for path in caminhos_possiveis:
        if os.path.exists(path):
            # 1. Adiciona ao PATH do sistema (para subprocessos e legado)
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
            
            # 2. Adiciona ao diretório de DLLs do Python (obrigatório Py 3.8+)
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(path)
                    dl_encontradas += 1
                except Exception:
                    pass
    
    if dl_encontradas > 0:
        print(f"   [OK] {dl_encontradas} diretórios de DLLs da NVIDIA registrados.")
    else:
        print("   [AVISO] Nenhuma pasta NVIDIA encontrada automaticamente. Se der erro, verifique a instalação.")

# Executa a configuração IMEDIATAMENTE
configurar_dlls_nvidia()

from openai import OpenAI, OpenAIError
from faster_whisper import WhisperModel

# =====================================================================
# CONFIGURAÇÕES E PARÂMETROS
# =====================================================================
LM_STUDIO_URL = "http://localhost:1234/v1"
MODELO_LLM = "meta-llama-3.1-8b-instruct"

PASTA_INPUT = "input"
PASTA_OUTPUT = "output"
EXTENSOES_VIDEO = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm', '.m4v')

client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

print("[SISTEMA] Carregando o modelo Whisper na GPU...")
modelo_whisper = WhisperModel("small", device="cuda", compute_type="float16")


def inicializar_pastas():
    if not os.path.exists(PASTA_INPUT):
        os.makedirs(PASTA_INPUT)
    if not os.path.exists(PASTA_OUTPUT):
        os.makedirs(PASTA_OUTPUT)


def salvar_arquivo(caminho, conteudo):
    try:
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(conteudo)
    except IOError as e:
        print(f"[ERRO DE SISTEMA] Falha ao gravar arquivo: {e}")


def transcrever_video_whisper(caminho_video):
    print(" -> Transcrevendo áudio do vídeo via Whisper...")
    try:
        inicio = time.time()
        segments, info = modelo_whisper.transcribe(caminho_video, beam_size=5, language="pt")
        
        texto_completo = []
        for segment in segments:
            texto_completo.append(segment.text)
            
        texto_bruto = " ".join(texto_completo).strip()
        tempo_gasto = time.time() - inicio
        print(f" -> Transcrição concluída em {tempo_gasto:.1f}s.")
        return texto_bruto
    except Exception as e:
        print(f" [ERRO CRÍTICO NO WHISPER]: {str(e)}")
        return None


def dividir_texto_em_blocos(texto, tamanho_max=3500):
    """Divide o texto em blocos menores sem cortar palavras ao meio."""
    palavras = texto.split(" ")
    blocos = []
    bloco_atual = []
    tamanho_atual = 0
    
    for palavra in palavras:
        if tamanho_atual + len(palavra) + 1 > tamanho_max:
            blocos.append(" ".join(bloco_atual))
            bloco_atual = [palavra]
            tamanho_atual = len(palavra)
        else:
            bloco_atual.append(palavra)
            tamanho_atual += len(palavra) + 1
            
    if bloco_atual:
        blocos.append(" ".join(bloco_atual))
        
    return blocos


def chamar_api_llama_com_retry(bloco_texto, max_tentativas=3):
    """Executa a chamada da API para um bloco específico com política de repetição."""
    prompt_sistema = (
        "Você é um assistente jurídico especializado em formalização de depoimentos judiciais.\n"
        "Sua única tarefa é reescrever o texto fornecido pelo usuário, convertendo-o estritamente "
        "para a terceira pessoa (narrador observador) e corrigindo erros de concordância da transcrição.\n\n"
        "REGRAS CRUCIAIS DE COMPORTAMENTO:\n"
        "1. NÃO adicione saudações, introduções ou comentários.\n"
        "2. NÃO adicione notas explicativas ou conclusões ao final.\n"
        "3. NÃO mostre seu processo de pensamento (chain of thought). Vá direto ao ponto.\n"
        "4. Retorne APENAS o texto final formalizado em terceira pessoa. Absolutamente mais nada."
    )

    for tentativa in range(1, max_tentativas + 1):
        try:
            response = client.chat.completions.create(
                model=MODELO_LLM,
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": f"Texto do depoimento:\n\n{bloco_texto}"}
                ],
                temperature=0.1,
                max_tokens=2048,
                timeout=60.0
            )
            
            resultado = response.choices[0].message.content.strip()
            if resultado and len(resultado) > (len(bloco_texto) * 0.3):
                return resultado
                
        except OpenAIError as e:
            print(f"   [AVISO] Erro na chamada do bloco (Tentativa {tentativa}/{max_tentativas}): {str(e)}")
            time.sleep(2)
            
    return None


def corrigir_com_llama(texto_bruto):
    """Gerencia a correção fatiando o texto se ele for muito grande."""
    if len(texto_bruto) <= 4000:
        print(" -> Enviando texto direto para o Llama...")
        return chamar_api_llama_com_retry(texto_bruto)
        
    blocos = dividir_texto_em_blocos(texto_bruto)
    print(f" -> Texto longo detectado. Dividido em {len(blocos)} partes para evitar travamentos.")
    
    resultados_finais = []
    for i, bloco in enumerate(blocos, start=1):
        print(f"   -> Processando parte {i}/{len(blocos)}...")
        resultado_bloco = chamar_api_llama_com_retry(bloco)
        
        if resultado_bloco:
            resultados_finais.append(resultado_bloco)
        else:
            print(f"   [FALHA CRÍTICA] Não foi possível processar a parte {i}. Abortando junção.")
            return None
            
    return "\n\n".join(resultados_finais)


def processar_pasta_videos():
    inicializar_pastas()
    arquivos_video = [f for f in os.listdir(PASTA_INPUT) if f.lower().endswith(EXTENSOES_VIDEO)]
    
    if not arquivos_video:
        print(f"\n[INFO] Nenhum arquivo de vídeo encontrado na pasta '{PASTA_INPUT}'.")
        return

    print(f"\n[SISTEMA] Encontrado(s) {len(arquivos_video)} vídeo(s) para processar.")

    for arquivo in arquivos_video:
        caminho_video = os.path.join(PASTA_INPUT, arquivo)
        nome_base = os.path.splitext(arquivo)[0]
        
        print(f"\n=======================================================")
        print(f"Iniciando Pipeline do Vídeo: {arquivo}")
        print(f"=======================================================")

        texto_bruto = transcrever_video_whisper(caminho_video)
        
        if not texto_bruto:
            continue

        caminho_bruto = os.path.join(PASTA_OUTPUT, f"{nome_base}_BRUTO.txt")
        salvar_arquivo(caminho_bruto, texto_bruto)

        texto_final = corrigir_com_llama(texto_bruto)

        caminho_saida_final = os.path.join(PASTA_OUTPUT, f"{nome_base}_FINAL.txt")
        caminho_saida_erro = os.path.join(PASTA_OUTPUT, f"{nome_base}_ERRO_LLM.txt")

        if texto_final:
            salvar_arquivo(caminho_saida_final, texto_final)
            print(f"[SUCESSO] Texto corrigido completo salvo em: {caminho_saida_final}")
        else:
            log_erro = "O Llama falhou em processar uma ou mais partes da transcrição deste vídeo.\n"
            salvar_arquivo(caminho_saida_erro, log_erro)
            print(f"[FALHA] Não foi possível obter o resultado ajustado para {arquivo}.")


if __name__ == "__main__":
    processar_pasta_videos()