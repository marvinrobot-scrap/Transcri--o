import os
import sys
import subprocess
import requests
import json
import re
from datetime import datetime
import importlib.util

# ==============================================================================
# 1. CORREÇÃO DE AMBIENTE (DLLs NVIDIA) - EXECUTA ANTES DE TUDO
# ==============================================================================
def configurar_dlls_nvidia():
    """
    Força o Windows a encontrar as DLLs do CUDA 12 (cublas, cudnn) 
    dentro da pasta site-packages do Python.
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

# ==============================================================================
# 2. CONFIGURAÇÕES DO PROGRAMA
# ==============================================================================

# Ajuste apenas se necessário
MODELO_WHISPER = "Gemma_4"
MODELO_LM_STUDIO = "google/gemma-4-26b-a4b" 
URL_LM_STUDIO = "http://localhost:1234/v1/chat/completions"

# Prompt para guiar o Whisper (Melhora pontuação e termos jurídicos)
WHISPER_PROMPT = (
    "Transcrição de audiência judicial brasileira. "
    "Termos: Vossa Excelência, Meritíssimo, Ministério Público, Defesa, Réu, Testemunha. "
    "Pontuação formal. Diálogo claro entre perguntas e respostas."
)

# ==============================================================================
# 3. FUNÇÕES UTILITÁRIAS
# ==============================================================================

def limpar_nome_arquivo(nome_arquivo):
    """Tenta extrair Nome e Papel do arquivo, ou usa padrão."""
    nome_base, _ = os.path.splitext(nome_arquivo)
    partes = nome_base.split('_')
    if len(partes) >= 2:
        return partes[0], partes[-1] 
    return nome_base, "Depoente"

def formatar_timestamp(segundos):
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = int(segundos % 60)
    return f"[{horas:02d}:{minutos:02d}:{segs:02d}]"

def converter_audio_ffmpeg(caminho_entrada, pasta_temp):
    """
    Converte para WAV 16kHz Mono e NORMALIZA o volume.
    Resolve problemas de áudio baixo ou codecs estranhos.
    """
    nome_base = os.path.splitext(os.path.basename(caminho_entrada))[0]
    caminho_saida = os.path.join(pasta_temp, f"{nome_base}_temp.wav")
    
    print(f"[FFmpeg] Processando áudio: {nome_base}...")
    
    # Filtro loudnorm para equalizar volume
    comando = [
        "ffmpeg", "-y", "-i", caminho_entrada,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", 
        caminho_saida
    ]
    
    try:
        # Check=True lança erro se o ffmpeg falhar
        subprocess.run(comando, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return caminho_saida
    except FileNotFoundError:
        print("\n[ERRO CRÍTICO] FFmpeg não encontrado!")
        print("Instale o FFmpeg e adicione ao PATH do Windows.")
        return None
    except subprocess.CalledProcessError:
        print(f"[ERRO] Falha ao converter arquivo: {caminho_entrada}")
        return None

# ==============================================================================
# 4. INTEGRAÇÃO WHISPER
# ==============================================================================

def transcrever_com_whisper(model, caminho_audio):
    print(f"[Whisper] Transcrevendo (Modelo: {MODELO_WHISPER})...")
    
    # Initial_prompt é crucial para qualidade jurídica
    segments, info = model.transcribe(
        caminho_audio,
        language="pt",
        beam_size=5,        # Mais precisão (padrão é 1, 5 é muito melhor)
        vad_filter=True,    # Remove silêncios
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt=WHISPER_PROMPT,
        condition_on_previous_text=False
    )
    
    # Converte gerador para lista imediatamente para processar tudo
    return list(segments)

# ==============================================================================
# 5. INTEGRAÇÃO LM STUDIO
# ==============================================================================

def chamar_llm(system_prompt, user_message, max_tokens=-1):
    payload = {
        "model": MODELO_LM_STUDIO,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.1, # Baixa criatividade = Alta fidelidade
        "max_tokens": max_tokens
    }
    try:
        r = requests.post(URL_LM_STUDIO, json=payload, timeout=3600)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[ERRO LLM]: Falha na conexão com LM Studio - {e}")
        return None

def gerar_diarizacao_corrigida(transcricao_bruta, nome, papel):
    """Fase 1: Limpeza e Identificação de Falantes"""
    print(f"[LLM] Fase 1: Identificando falantes e corrigindo texto...")
    
    sys_prompt = (
        "Você é um especialista em transcrição forense.\n"
        "TAREFA: Receba o texto bruto (com timestamps) e formate como um DIÁLOGO JUDICIAL.\n"
        "REGRAS:\n"
        "1. Identifique os falantes pelo contexto. Use: 'JUIZ:', 'DEFESA:', 'MP:' e o NOME DO DEPOENTE.\n"
        "2. Corrija erros de grafia (ex: 'tráfego' -> 'tráfico') mantendo o sentido.\n"
        "3. NÃO RESUMA. Mantenha cada frase dita.\n"
        "4. Mantenha os timestamps no início das falas.\n"
        "5. Remova gagueira excessiva."
    )
    
    user_prompt = (
        f"Depoente: {nome} ({papel})\n"
        "Texto Bruto:\n" + transcricao_bruta
    )
    
    return chamar_llm(sys_prompt, user_prompt)

def gerar_narrativa_final(texto_dialogo, nome, papel):
    """Fase 2: Narrativa Jurídica"""
    print(f"[LLM] Fase 2: Gerando termo formal...")
    
    sys_prompt = (
        "Você é um Assistente Jurídico. Converta o diálogo em um TERMO DE DEPOIMENTO.\n"
        "DIRETRIZES:\n"
        "1. Escreva em terceira pessoa ('Disse que...', 'Informou que...').\n"
        "2. Transforme perguntas em narrativa indireta ('Indagado sobre X, respondeu que Y').\n"
        f"3. Inicie com: '{nome}, {papel}, ouvido em juízo, declarou que...'\n"
        "4. Seja DETALHISTA. Inclua locais, datas mencionadas, descrições.\n"
        "5. Texto corrido, sem tópicos."
    )
    
    user_prompt = f"Diálogo Base:\n{texto_dialogo}"
    
    return chamar_llm(sys_prompt, user_prompt)

# ==============================================================================
# 6. PIPELINE PRINCIPAL
# ==============================================================================

def processar_arquivo(caminho_arquivo, model, temp_dir, result_dir):
    nome_arquivo = os.path.basename(caminho_arquivo)
    nome, papel = limpar_nome_arquivo(nome_arquivo)
    
    # 1. Normalização de Áudio
    audio_wav = converter_audio_ffmpeg(caminho_arquivo, temp_dir)
    if not audio_wav: return

    try:
        # 2. Transcrição
        segmentos = transcrever_com_whisper(model, audio_wav)
        
        # Gera texto bruto
        texto_bruto = "\n".join([f"{formatar_timestamp(s.start)} {s.text.strip()}" for s in segmentos])
        
        # Salva Arq 1
        path_raw = os.path.join(result_dir, f"{nome}_1_bruto.txt")
        with open(path_raw, "w", encoding="utf-8") as f: f.write(texto_bruto)

        # 3. Fase 1 LLM (Diálogo)
        texto_corrigido = gerar_diarizacao_corrigida(texto_bruto, nome, papel)
        if texto_corrigido:
            path_diag = os.path.join(result_dir, f"{nome}_2_dialogo.txt")
            with open(path_diag, "w", encoding="utf-8") as f: f.write(texto_corrigido)
            
            # 4. Fase 2 LLM (Narrativa)
            narrativa = gerar_narrativa_final(texto_corrigido, nome, papel)
            if narrativa:
                path_narr = os.path.join(result_dir, f"{nome}_3_narrativa.txt")
                with open(path_narr, "w", encoding="utf-8") as f: f.write(narrativa)
        
        print(f"--> Concluído: {nome_arquivo}\n")

    except Exception as e:
        print(f"[ERRO NO PROCESSO] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Limpa wav temporário para economizar espaço
        if audio_wav and os.path.exists(audio_wav):
            try: os.remove(audio_wav)
            except: pass

def main():
    # Pastas relativas ao local do script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input")
    temp_dir = os.path.join(base_dir, "temp")
    result_dir = os.path.join(base_dir, "output")
    
    for d in [input_dir, temp_dir, result_dir]:
        os.makedirs(d, exist_ok=True)

    print("--- INICIANDO SISTEMA DE TRANSCRIÇÃO JURÍDICA ---")
    print(f"Entrada: {input_dir}")
    print(f"Saída:   {result_dir}")
    
    # Carrega Modelo Whisper
    try:
        from faster_whisper import WhisperModel
        
        print("Carregando modelo na GPU...")
        # Tenta carregar na GPU com float16
        model = WhisperModel(MODELO_WHISPER, device="cuda", compute_type="float16")
        print("Modelo GPU carregado com sucesso!")
        
    except Exception as e:
        print(f"\n[ERRO GPU] Falha ao carregar CUDA: {e}")
        print("Tentando fallback para CPU (INT8)... (ISSO SERÁ LENTO)")
        try:
            model = WhisperModel(MODELO_WHISPER, device="cpu", compute_type="int8")
        except Exception as e_cpu:
            print(f"Erro fatal também na CPU: {e_cpu}")
            return

    # Loop de arquivos
    arquivos = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    if not arquivos:
        print(f"\nA pasta 'input' está vazia.")
        print("Coloque arquivos de vídeo/áudio lá e execute novamente.")
        input("Enter para sair...")
        return

    for arq in arquivos:
        if arq.lower().endswith(('.txt', '.py', '.md')): continue # Pula arquivos de texto
        processar_arquivo(os.path.join(input_dir, arq), model, temp_dir, result_dir)
    
    print("\nProcessamento finalizado.")
    input("Pressione Enter para fechar...")

if __name__ == "__main__":
    main()