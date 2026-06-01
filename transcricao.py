import os
import sys
import time
from openai import OpenAI, OpenAIError
from faster_whisper import WhisperModel

# =====================================================================
# CONFIGURAÇÕES E PARÂMETROS
# =====================================================================
LM_STUDIO_URL = "http://localhost:1234/v1"
MODELO_LLM = "meta-llama-3.1-8b-instruct"

PASTA_INPUT = "input"
PASTA_OUTPUT = "output"

# Extensões de vídeo suportadas (adicione mais se necessário)
EXTENSOES_VIDEO = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm', '.m4v')

# Inicializa o cliente do LM Studio
client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

print("[SISTEMA] Carregando o modelo Whisper na GPU...")
# Inicializa o Whisper usando a GPU (cuda) para velocidade máxima.
# O modelo 'small' ou 'medium' em português costuma ser excelente e rápido.
modelo_whisper = WhisperModel("small", device="cuda", compute_type="float16")


def inicializar_pastas():
    """Garante que as pastas de entrada e saída existam."""
    if not os.path.exists(PASTA_INPUT):
        os.makedirs(PASTA_INPUT)
        print(f"[SISTEMA] Pasta '{PASTA_INPUT}' criada. Coloque seus VÍDEOS nela.")
    if not os.path.exists(PASTA_OUTPUT):
        os.makedirs(PASTA_OUTPUT)
        print(f"[SISTEMA] Pasta '{PASTA_OUTPUT}' criada para receber os textos finais.")


def salvar_arquivo(caminho, conteudo):
    """Salva arquivos de texto com segurança."""
    try:
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(conteudo)
    except IOError as e:
        print(f"[ERRO DE SISTEMA] Falha ao gravar arquivo em {caminho}: {e}")


def transcrever_video_whisper(caminho_video):
    """Usa o faster-whisper para extrair e transcrever o áudio do vídeo."""
    print(" -> Transcrevendo áudio do vídeo via Whisper...")
    try:
        inicio = time.time()
        # beam_size=5 é o padrão para boa precisão. language="pt" força o português.
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


def corrigir_com_llama(texto_bruto, max_tentativas=3):
    """Envia o texto para o Llama converter para 3ª pessoa jurídica."""
    prompt_sistema = (
        "Você é um assistente jurídico especializado em formalização de depoimentos judiciais.\n"
        "Sua única tarefa é reescrever o texto fornecido pelo usuário, convertendo-o estritamente "
        "para a terceira pessoa (narrador observador) e corrigindo erros de concordância da transcrição.\n\n"
        "REGRAS CRUCIAIS DE COMPORTAMENTO:\n"
        "1. NÃO adicione saudações, introduções ou comentários (ex: 'Aqui está', 'Segue o texto').\n"
        "2. NÃO adicione notas explicativas, justificativas ou conclusões ao final.\n"
        "3. NÃO mostre seu processo de pensamento (chain of thought). Vá direto ao ponto.\n"
        "4. Retorne APENAS o texto final formalizado em terceira pessoa. Absolutamente mais nada."
    )

    for tentativa in range(1, max_tentativas + 1):
        print(f" -> Enviando para o Llama (Tentativa {tentativa}/{max_tentativas})...")
        try:
            inicio = time.time()
            response = client.chat.completions.create(
                model=MODELO_LLM,
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user", "content": f"Texto do depoimento:\n\n{texto_bruto}"}
                ],
                temperature=0.1,
                max_tokens=2048,
                timeout=90.0
            )
            
            tempo_gasto = time.time() - inicio
            resultado = response.choices[0].message.content.strip()

            if resultado and len(resultado) > (len(texto_bruto) * 0.3):
                return resultado
            else:
                print(f" [AVISO] Resposta da LLM na tentativa {tentativa} veio vazia ou curta demais.")

        except OpenAIError as e:
            print(f" [ERRO LM STUDIO - TENTATIVA {tentativa}]: {str(e)}")
            time.sleep(2)
            
    return None


def processar_pasta_videos():
    """Varre a pasta input por vídeos, transcreve e gera os textos na pasta output."""
    inicializar_pastas()
    
    # Filtra arquivos pelas extensões de vídeo especificadas
    arquivos_video = [f for f in os.listdir(PASTA_INPUT) if f.lower().endswith(EXTENSOES_VIDEO)]
    
    if not arquivos_video:
        print(f"\n[INFO] Nenhum arquivo de vídeo encontrado na pasta '{PASTA_INPUT}'.")
        print("Formatos aceitos: MP4, MKV, AVI, MOV, FLV, WMV, WEBM, M4V.")
        return

    print(f"\n[SISTEMA] Encontrado(s) {len(arquivos_video)} vídeo(s) para processar.")

    for arquivo in arquivos_video:
        caminho_video = os.path.join(PASTA_INPUT, arquivo)
        nome_base = os.path.splitext(arquivo)[0]
        
        print(f"\n=======================================================")
        print(f"Iniciando Pipeline do Vídeo: {arquivo}")
        print(f"=======================================================")

        # Passo 1: Transcrição
        texto_bruto = transcrever_video_whisper(caminho_video)
        
        if not texto_bruto:
            print(f"[FALHA] Não foi possível transcrever o vídeo {arquivo}. Pulando para o próximo.")
            continue

        # Salvando o texto bruto gerado pelo Whisper na pasta OUTPUT para segurança
        caminho_bruto = os.path.join(PASTA_OUTPUT, f"{nome_base}_BRUTO.txt")
        salvar_arquivo(caminho_bruto, texto_bruto)
        print(f" -> Texto bruto (Whisper) salvo em: {caminho_bruto}")

        # Passo 2: Correção Jurídica pela LLM
        texto_final = corrigir_com_llama(texto_bruto)

        # Caminhos dos arquivos de texto finais na pasta OUTPUT
        caminho_saida_final = os.path.join(PASTA_OUTPUT, f"{nome_base}_FINAL.txt")
        caminho_saida_erro = os.path.join(PASTA_OUTPUT, f"{nome_base}_ERRO_LLM.txt")

        if texto_final:
            salvar_arquivo(caminho_saida_final, texto_final)
            print(f"[SUCESSO] Texto corrigido em 3ª pessoa salvo em: {caminho_saida_final}")
        else:
            log_erro = (
                f"O Llama falhou em processar a transcrição deste vídeo após as tentativas.\n"
                f"O texto original gerado pelo Whisper foi preservado em '{os.path.basename(caminho_bruto)}'.\n"
            )
            salvar_arquivo(caminho_saida_erro, log_erro)
            print(f"[FALHA] O Llama não retornou um resultado útil. Detalhes salvos na pasta output.")


if __name__ == "__main__":
    processar_pasta_videos()