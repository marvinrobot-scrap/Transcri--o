import os
import sys
import subprocess
import requests
import json
from datetime import datetime

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
    caminhos_possiveis = [
        os.path.join(base_python, "Lib", "site-packages", "nvidia", "cublas", "bin"),
        os.path.join(base_python, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
        os.path.join(base_python, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin"),
    ]

    dl_encontradas = 0
    for path in caminhos_possiveis:
        if os.path.exists(path):
            os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
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


configurar_dlls_nvidia()

# ==============================================================================
# 2. CONFIGURAÇÕES DO PROGRAMA
# ==============================================================================

MODELO_WHISPER = "large-v3"

# Ajuste para o nome exato do modelo Qwen3.5-9B no LM Studio
MODELO_LM_STUDIO = "qwen/qwen3.5-9b"

URL_LM_STUDIO = "http://localhost:1234/v1/chat/completions"

WHISPER_PROMPT = (
    "Transcrição de audiência judicial brasileira. "
    "Termos: Vossa Excelência, Meritíssimo, Ministério Público, Defesa, Réu, Testemunha. "
    "Pontuação formal. Diálogo claro entre perguntas e respostas."
)

MAX_CHARS_BLOCO = 16000
LOG_DIR_NAME = "logs_llm"

# ==============================================================================
# 3. FUNÇÕES UTILITÁRIAS
# ==============================================================================

def limpar_nome_arquivo(nome_arquivo):
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
    nome_base = os.path.splitext(os.path.basename(caminho_entrada))[0]
    caminho_saida = os.path.join(pasta_temp, f"{nome_base}_temp.wav")

    print(f"[FFmpeg] Processando áudio: {nome_base}...")

    comando = [
        "ffmpeg", "-y", "-i", caminho_entrada,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        caminho_saida
    ]

    try:
        subprocess.run(comando, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return caminho_saida
    except FileNotFoundError:
        print("\n[ERRO CRÍTICO] FFmpeg não encontrado!")
        print("Instale o FFmpeg e adicione ao PATH do Windows.")
        return None
    except subprocess.CalledProcessError:
        print(f"[ERRO] Falha ao converter arquivo: {caminho_entrada}")
        return None


def dividir_em_blocos(texto, max_chars=MAX_CHARS_BLOCO):
    linhas = texto.splitlines()
    blocos = []
    bloco_atual = []
    tamanho_atual = 0

    for linha in linhas:
        linha_len = len(linha) + 1
        if tamanho_atual + linha_len > max_chars and bloco_atual:
            blocos.append("\n".join(bloco_atual))
            bloco_atual = [linha]
            tamanho_atual = linha_len
        else:
            bloco_atual.append(linha)
            tamanho_atual += linha_len

    if bloco_atual:
        blocos.append("\n".join(bloco_atual))

    return blocos


def dividir_dialogo_em_paragrafos(texto_dialogo, max_chars=1000):
    """
    Divide o diálogo corrigido em pedaços menores para a Fase 2,
    preservando quebras de linha para manter correspondência com o original.
    max_chars é menor que MAX_CHARS_BLOCO para termos parágrafos curtos.
    """
    linhas = texto_dialogo.splitlines()
    paragrafos = []
    atual = []
    tam = 0

    for linha in linhas:
        linha = linha.rstrip()
        if not linha:
            # quebra explícita de parágrafo
            if atual:
                paragrafos.append("\n".join(atual))
                atual = []
                tam = 0
            continue

        l_len = len(linha) + 1
        if tam + l_len > max_chars and atual:
            paragrafos.append("\n".join(atual))
            atual = [linha]
            tam = l_len
        else:
            atual.append(linha)
            tam += l_len

    if atual:
        paragrafos.append("\n".join(atual))

    return paragrafos


def salvar_log_llm(base_dir, arquivo_entrada, fase, bloco_idx, payload, resposta):
    logs_dir = os.path.join(base_dir, LOG_DIR_NAME)
    os.makedirs(logs_dir, exist_ok=True)

    agora = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_base = os.path.splitext(os.path.basename(arquivo_entrada))[0]
    filename = f"{nome_base}_fase{fase}_bloco{bloco_idx}_{agora}.json"
    caminho_log = os.path.join(logs_dir, filename)

    registro = {
        "timestamp": agora,
        "arquivo_entrada": arquivo_entrada,
        "fase": fase,
        "bloco_idx": bloco_idx,
        "modelo": MODELO_LM_STUDIO,
        "request": payload,
        "response": resposta,
    }

    try:
        with open(caminho_log, "w", encoding="utf-8") as f:
            json.dump(registro, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[AVISO] Falha ao gravar log LLM {caminho_log}: {e}")


# ==============================================================================
# 4. WHISPER
# ==============================================================================

def transcrever_com_whisper(model, caminho_audio):
    print(f"[Whisper] Transcrevendo (Modelo: {MODELO_WHISPER})...")

    segments, info = model.transcribe(
        caminho_audio,
        language="pt",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt=WHISPER_PROMPT,
        condition_on_previous_text=False
    )

    return list(segments)


# ==============================================================================
# 5. LM STUDIO (CHAMADA CONSERVADORA + FILTRO DE THINKING)
# ==============================================================================

def extrair_resposta_sem_pensamento(texto):
    if not texto:
        return texto

    if "<think>" in texto and "</think>" in texto:
        after = texto.split("</think>", 1)[-1].strip()
        if after:
            return after

    patterns = [
        "Thinking Process:",
        "**Thinking Process:**",
        "Raciocínio:",
        "Raciocínio passo a passo:"
    ]

    last_idx = -1
    for p in patterns:
        idx = texto.rfind(p)
        if idx > last_idx:
            last_idx = idx

    if last_idx != -1:
        restante = texto[last_idx:].split("\n", 1)
        if len(restante) == 2:
            return restante[1].strip()

    return texto.strip()


def chamar_llm(system_prompt, user_message, max_tokens=-1, fase=None, bloco_idx=None,
               base_dir=None, arquivo_entrada=None):
    system_prompt_final = (
        system_prompt
        + "\n\nINSTRUÇÃO IMPORTANTE:\n"
          "Você NÃO deve mostrar seu raciocínio passo a passo, nem qualquer seção intitulada "
          "'Thinking Process', 'Raciocínio', 'Análise' ou similar. "
          "Apenas produza diretamente o texto final solicitado, "
          "sem explicar como chegou a ele."
    )

    user_message_final = user_message + "\n\n/no_think"

    payload = {
        "model": MODELO_LM_STUDIO,
        "messages": [
            {"role": "system", "content": system_prompt_final},
            {"role": "user", "content": user_message_final}
        ],
        "temperature": 0.1,
        "top_p": 0.8,
        "max_tokens": max_tokens
    }

    try:
        r = requests.post(URL_LM_STUDIO, json=payload, timeout=3600)
        r.raise_for_status()
        bruta = r.json()["choices"][0]["message"]["content"]
        resposta = extrair_resposta_sem_pensamento(bruta)
    except Exception as e:
        print(f"[ERRO LLM]: Falha na conexão com LM Studio - {e}")
        bruta = None
        resposta = None

    if base_dir and arquivo_entrada and fase is not None and bloco_idx is not None:
        salvar_log_llm(base_dir, arquivo_entrada, fase, bloco_idx, payload, {
            "raw": bruta,
            "filtered": resposta,
        })

    return resposta


# ==============================================================================
# 6. FASE 1 - DIÁLOGO CORRIGIDO EM BLOCOS
# ==============================================================================

def gerar_diarizacao_corrigida(transcricao_bruta, nome, papel, base_dir, arquivo_entrada):
    print(f"[LLM] Fase 1: Identificando falantes e corrigindo texto (em blocos)...")

    sys_prompt = (
        "Você é um especialista em transcrição forense.\n"
        "TAREFA: Receba o texto bruto (com timestamps) e formate como um DIÁLOGO JUDICIAL.\n"
        "REGRAS:\n"
        "1. Identifique os falantes pelo contexto. Use: 'JUIZ:', 'DEFESA:', 'MP:' e o NOME DO DEPOENTE.\n"
        "2. Corrija erros de grafia mantendo o sentido.\n"
        "3. NÃO RESUMA. Mantenha cada frase dita.\n"
        "4. Mantenha os timestamps no início das falas.\n"
        "5. Remova gagueira excessiva sem alterar o conteúdo.\n"
        "6. NÃO invente falas ou informações que não estejam no texto.\n"
        "7. Se algum trecho estiver ininteligível, mantenha o timestamp e escreva '[INAUDÍVEL]'.\n"
        "8. Não inclua comentários explicativos, apenas o diálogo final."
    )

    blocos = dividir_em_blocos(transcricao_bruta, max_chars=MAX_CHARS_BLOCO)
    dialogos_corrigidos = []

    for i, bloco in enumerate(blocos, start=1):
        print(f"[LLM] Fase 1 - bloco {i}/{len(blocos)}...")

        user_prompt = (
            f"Depoente: {nome} ({papel}).\n"
            "Receba o trecho abaixo do texto bruto e produza apenas o diálogo judicial corrigido, "
            "sem resumo e sem comentários adicionais.\n\n"
            f"TRECHO:\n{bloco}"
        )

        resposta = chamar_llm(
            sys_prompt,
            user_prompt,
            max_tokens=-1,
            fase=1,
            bloco_idx=i,
            base_dir=base_dir,
            arquivo_entrada=arquivo_entrada
        )

        if not resposta:
            print(f"[AVISO] Bloco {i} retornou vazio, mantendo texto original do bloco.")
            resposta = bloco

        dialogos_corrigidos.append(resposta.strip())

    return "\n\n".join(dialogos_corrigidos)


# ==============================================================================
# 7. FASE 2 - NARRATIVA EM PARÁGRAFOS CURTOS E FIÉIS
# ==============================================================================

def gerar_narrativa_final(texto_dialogo, nome, papel, base_dir, arquivo_entrada):
    """
    Converte o diálogo já corrigido em uma narrativa jurídica,
    mas em blocos curtos, com fidelidade máxima ao conteúdo original.
    Cada bloco gera um parágrafo curto.
    """
    print(f"[LLM] Fase 2: Gerando termo formal (parágrafos curtos, fiéis)...")

    sys_prompt = (
        "Você é um Assistente Jurídico. Sua função é APENAS reescrever o diálogo abaixo "
        "em forma de TERMO DE DEPOIMENTO, mantendo o conteúdo o mais fiel possível.\n"
        "REGRAS OBRIGATÓRIAS:\n"
        "1. Escreva em terceira pessoa ('Disse que...', 'Informou que...').\n"
        "2. Converta perguntas em narrativa indireta ('Indagado sobre X, respondeu que Y').\n"
        "3. NÃO invente fatos, motivos, opiniões ou detalhes que não estejam literalmente "
        "presentes no diálogo.\n"
        "4. NÃO adicione análises, comentários, hipóteses ou sugestões.\n"
        "5. NÃO altere a ordem dos acontecimentos.\n"
        "6. NÃO resuma: reaproveite todas as informações relevantes do trecho recebido.\n"
        "7. Se algum trecho estiver confuso ou faltar informação, escreva exatamente o que "
        "consta no diálogo, sem tentar completar.\n"
        "8. Produza um parágrafo curto para cada trecho recebido, sem listas ou tópicos.\n"
        "9. Não use marcadores como 'Thinking Process', 'Análise' ou similares. "
        "Entregue apenas o texto final do termo."
    )

    paragrafos_dialogo = dividir_dialogo_em_paragrafos(texto_dialogo, max_chars=1000)
    partes_narrativa = []

    for i, trecho in enumerate(paragrafos_dialogo, start=1):
        print(f"[LLM] Fase 2 - parágrafo {i}/{len(paragrafos_dialogo)}...")

        if i == 1:
            prefixo = (
                f"O texto abaixo é a primeira parte do diálogo transcrito do depoimento de {nome}, {papel}. "
                f"O primeiro parágrafo da narrativa deve começar exatamente com: "
                f"'{nome}, {papel}, ouvido em juízo, declarou que...'. "
                "A partir daí, reescreva o conteúdo do trecho em terceira pessoa, "
                "seguindo rigorosamente as regras.\n\n"
            )
        else:
            prefixo = (
                "O texto abaixo é continuação do mesmo depoimento. "
                "Reescreva APENAS este trecho em forma de parágrafo curto, "
                "continuando a narrativa no mesmo estilo, "
                "sem repetir o início do depoimento e sem alterar o que já foi narrado.\n\n"
            )

        user_prompt = prefixo + f"DIÁLOGO:\n{trecho}"

        resposta = chamar_llm(
            sys_prompt,
            user_prompt,
            max_tokens=-1,
            fase=2,
            bloco_idx=i,
            base_dir=base_dir,
            arquivo_entrada=arquivo_entrada
        )

        if not resposta:
            print(f"[AVISO] Parágrafo {i} da narrativa retornou vazio, mantendo diálogo original do trecho.")
            resposta = trecho

        partes_narrativa.append(resposta.strip())

    return "\n\n".join(partes_narrativa)


# ==============================================================================
# 8. PIPELINE PRINCIPAL
# ==============================================================================

def processar_arquivo(caminho_arquivo, model, temp_dir, result_dir, base_dir):
    nome_arquivo = os.path.basename(caminho_arquivo)
    nome, papel = limpar_nome_arquivo(nome_arquivo)

    audio_wav = converter_audio_ffmpeg(caminho_arquivo, temp_dir)
    if not audio_wav:
        return

    try:
        segmentos = transcrever_com_whisper(model, audio_wav)

        texto_bruto = "\n".join(
            [f"{formatar_timestamp(s.start)} {s.text.strip()}" for s in segmentos]
        )

        path_raw = os.path.join(result_dir, f"{nome}_1_bruto.txt")
        with open(path_raw, "w", encoding="utf-8") as f:
            f.write(texto_bruto)

        texto_corrigido = gerar_diarizacao_corrigida(
            texto_bruto, nome, papel, base_dir, caminho_arquivo
        )

        if texto_corrigido:
            path_diag = os.path.join(result_dir, f"{nome}_2_dialogo.txt")
            with open(path_diag, "w", encoding="utf-8") as f:
                f.write(texto_corrigido)

            narrativa = gerar_narrativa_final(
                texto_corrigido, nome, papel, base_dir, caminho_arquivo
            )

            if narrativa:
                path_narr = os.path.join(result_dir, f"{nome}_3_narrativa.txt")
                with open(path_narr, "w", encoding="utf-8") as f:
                    f.write(narrativa)

        print(f"--> Concluído: {nome_arquivo}\n")

    except Exception as e:
        print(f"[ERRO NO PROCESSO] {e}")
        import traceback
        traceback.print_exc()
    finally:
        if audio_wav and os.path.exists(audio_wav):
            try:
                os.remove(audio_wav)
            except:
                pass


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input")
    temp_dir = os.path.join(base_dir, "temp")
    result_dir = os.path.join(base_dir, "output")

    for d in [input_dir, temp_dir, result_dir]:
        os.makedirs(d, exist_ok=True)

    print("--- INICIANDO SISTEMA DE TRANSCRIÇÃO JURÍDICA ---")
    print(f"Entrada: {input_dir}")
    print(f"Saída:   {result_dir}")
    print(f"Modelo LLM: {MODELO_LM_STUDIO}")

    try:
        from faster_whisper import WhisperModel
        print("Carregando modelo Whisper na GPU...")
        model = WhisperModel(MODELO_WHISPER, device="cuda", compute_type="float16")
        print("Modelo Whisper (GPU) carregado com sucesso!")
    except Exception as e:
        print(f"\n[ERRO GPU] Falha ao carregar CUDA: {e}")
        print("Tentando fallback para CPU (INT8)... (ISSO SERÁ LENTO)")
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(MODELO_WHISPER, device="cpu", compute_type="int8")
        except Exception as e_cpu:
            print(f"Erro fatal também na CPU: {e_cpu}")
            return

    arquivos = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    ]

    if not arquivos:
        print(f"\nA pasta 'input' está vazia.")
        print("Coloque arquivos de vídeo/áudio lá e execute novamente.")
        input("Enter para sair...")
        return

    for arq in arquivos:
        if arq.lower().endswith(('.txt', '.py', '.md')):
            continue
        processar_arquivo(os.path.join(input_dir, arq), model, temp_dir, result_dir, base_dir)

    print("\nProcessamento finalizado.")
    input("Pressione Enter para fechar...")


if __name__ == "__main__":
    main()
