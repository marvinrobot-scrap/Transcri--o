import os
import sys
import subprocess
import requests
import time
import traceback


# ==============================================================================
# 1. CORREÇÃO DE AMBIENTE (DLLs NVIDIA)
# ==============================================================================
def configurar_dlls_nvidia():
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
        print(f"[OK] {dl_encontradas} diretórios de DLLs da NVIDIA registrados.")
    else:
        print("[AVISO] Nenhuma pasta NVIDIA encontrada automaticamente.")


configurar_dlls_nvidia()


# ==============================================================================
# 2. CONFIGURAÇÕES
# ==============================================================================
MODELO_WHISPER = "large-v3"

# Modelo rápido para fases estruturadas
MODELO_LM_FAST = "qwen/qwen3.5-9b"

# Modelo mais forte para redação final
MODELO_LM_FINAL = "jurema-7b"

URL_LM_STUDIO = "http://localhost:1234/v1/chat/completions"

WHISPER_PROMPT = (
    "Transcrição de audiência judicial brasileira. "
    "Termos: Vossa Excelência, Meritíssimo, Ministério Público, Defesa, Réu, Testemunha. "
    "Pontuação formal. Diálogo claro entre perguntas e respostas."
)

MAX_CHARS_LLM_FASE1 = 6000
MAX_CHARS_LLM_FASE2 = 6000
OVERLAP_CHARS = 800
MIN_CHARS_RESPOSTA_UTIL = 100
TENTATIVAS_LLM = 3
TIMEOUT_LLM = (15, 1800)


# ==============================================================================
# 3. UTILITÁRIOS
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


def salvar_texto(caminho, conteudo):
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(conteudo)


def conteudo_valido(texto, minimo=MIN_CHARS_RESPOSTA_UTIL):
    if not texto or not isinstance(texto, str):
        return False

    limpo = texto.strip()
    if len(limpo) < minimo:
        return False

    erros_comuns = [
        "como você não forneceu o diálogo base",
        "não tenho o conteúdo para converter",
        "não foi fornecido o diálogo base",
        "conteúdo não fornecido",
        "texto não fornecido",
        "aqui está o texto",
        "segue abaixo",
        "abaixo está",
    ]

    limpo_lower = limpo.lower()
    for erro in erros_comuns:
        if erro in limpo_lower:
            return False

    return True


def limpar_saida_llm(texto):
    if not texto:
        return None

    linhas_ruins_inicio = [
        "aqui está",
        "segue abaixo",
        "abaixo está",
        "comentário:",
        "observação:",
        "nota:",
        "explicação:",
        "não foi fornecido",
        "como você não forneceu",
        "não tenho o conteúdo",
        "claro,",
        "certamente,",
    ]

    linhas = texto.splitlines()
    filtradas = []

    for linha in linhas:
        linha_limpa = linha.strip()
        if not linha_limpa:
            filtradas.append("")
            continue

        lower = linha_limpa.lower()
        if any(lower.startswith(x) for x in linhas_ruins_inicio):
            continue

        filtradas.append(linha)

    texto_final = "\n".join(filtradas).strip()
    return texto_final if texto_final else None


def dividir_em_chunks_com_sobreposicao(texto, max_chars=6000, overlap_chars=800):
    if len(texto) <= max_chars:
        return [texto]

    chunks = []
    inicio = 0
    tamanho = len(texto)

    while inicio < tamanho:
        fim = min(inicio + max_chars, tamanho)

        if fim < tamanho:
            quebra = texto.rfind("\n", inicio, fim)
            if quebra != -1 and quebra > inicio + 1000:
                fim = quebra

        chunk = texto[inicio:fim].strip()
        if chunk:
            chunks.append(chunk)

        if fim >= tamanho:
            break

        inicio = max(fim - overlap_chars, 0)

    return chunks


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
        print("[ERRO CRÍTICO] FFmpeg não encontrado.")
        return None
    except subprocess.CalledProcessError:
        print(f"[ERRO] Falha ao converter arquivo: {caminho_entrada}")
        return None


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

    segmentos = list(segments)

    if not segmentos:
        raise RuntimeError("Whisper não retornou segmentos.")

    return segmentos


# ==============================================================================
# 5. LLM
# ==============================================================================
def chamar_llm(modelo, system_prompt, user_message, max_tokens=-1, tentativas=TENTATIVAS_LLM):
    payload = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens
    }

    for tentativa in range(1, tentativas + 1):
        try:
            inicio = time.time()
            print(f"[LLM:{modelo}] Tentativa {tentativa}/{tentativas} | user chars={len(user_message)}")
            r = requests.post(URL_LM_STUDIO, json=payload, timeout=TIMEOUT_LLM)
            duracao = time.time() - inicio

            print(f"[LLM:{modelo}] HTTP {r.status_code} em {duracao:.1f}s")
            r.raise_for_status()

            data = r.json()
            resposta = data["choices"][0]["message"]["content"]

            if resposta is None:
                print(f"[LLM:{modelo}] Resposta veio None.")
                time.sleep(2 * tentativa)
                continue

            resposta = limpar_saida_llm(resposta)
            if not resposta:
                print(f"[LLM:{modelo}] Resposta vazia após limpeza.")
                time.sleep(2 * tentativa)
                continue

            print(f"[LLM:{modelo}] chars resposta={len(resposta)}")
            print(f"[LLM:{modelo}] prévia={resposta[:200].replace(chr(10), ' ')}")

            if conteudo_valido(resposta, minimo=50):
                return resposta

            print(f"[LLM:{modelo}] Resposta inválida. Retentando...")

        except requests.exceptions.Timeout:
            print(f"[ERRO LLM:{modelo}] Timeout.")
        except requests.exceptions.ConnectionError as e:
            print(f"[ERRO LLM:{modelo}] Falha de conexão: {e}")
        except Exception as e:
            print(f"[ERRO LLM:{modelo}] Erro inesperado: {e}")

        time.sleep(2 * tentativa)

    return None


# ==============================================================================
# 6. FASE 1 - DIÁLOGO CORRIGIDO (MODELO RÁPIDO)
# ==============================================================================
def gerar_diarizacao_corrigida_chunk(transcricao_chunk, nome, papel, indice_chunk, total_chunks):
    print(f"[LLM] Fase 1: Chunk {indice_chunk}/{total_chunks}")

    sys_prompt = (
        "Você é um especialista em transcrição forense.\n"
        "Formate o texto bruto como DIÁLOGO JUDICIAL.\n"
        "REGRAS:\n"
        "1. Identifique falantes pelo contexto.\n"
        "2. Use apenas: 'JUIZ:', 'DEFESA:', 'MP:' e o nome do depoente.\n"
        "3. Corrija grafia mantendo o sentido.\n"
        "4. NÃO RESUMA.\n"
        "5. Mantenha timestamps.\n"
        "6. Não invente conteúdo.\n"
        "7. Não escreva comentários.\n"
        "8. Responda só com o diálogo corrigido.\n"
    )

    user_prompt = (
        f"Depoente: {nome} ({papel})\n"
        f"Parte {indice_chunk} de {total_chunks}\n"
        "Texto Bruto:\n"
        f"{transcricao_chunk}"
    )

    return chamar_llm(MODELO_LM_FAST, sys_prompt, user_prompt)


def gerar_diarizacao_corrigida_em_chunks(transcricao_bruta, nome, papel):
    chunks = dividir_em_chunks_com_sobreposicao(
        transcricao_bruta,
        max_chars=MAX_CHARS_LLM_FASE1,
        overlap_chars=OVERLAP_CHARS
    )

    resultados = []
    print(f"[INFO] Fase 1 será processada em {len(chunks)} chunk(s).")

    for i, chunk in enumerate(chunks, start=1):
        resposta = gerar_diarizacao_corrigida_chunk(chunk, nome, papel, i, len(chunks))
        if conteudo_valido(resposta, minimo=100):
            resultados.append(resposta.strip())
            print(f"[INFO] Chunk {i}/{len(chunks)} concluído.")
        else:
            print(f"[ALERTA] Chunk {i} da Fase 1 falhou.")

    if not resultados:
        return None

    return "\n\n".join(resultados)


# ==============================================================================
# 7. FASE 2 - EXTRAÇÃO DE FATOS (MODELO RÁPIDO)
# ==============================================================================
def gerar_fatos_estruturados_chunk(texto_dialogo_chunk, nome, papel, indice_chunk, total_chunks):
    print(f"[LLM] Fase 2: Chunk {indice_chunk}/{total_chunks}")

    sys_prompt = (
        "Você é um analista jurídico.\n"
        "Extraia APENAS fatos do trecho recebido.\n"
        "NÃO escreva introdução, comentário, explicação ou conclusão.\n"
        "NÃO invente informações.\n"
        "Saída obrigatória: linhas iniciadas por '- '.\n"
        "Cada linha deve conter um fato objetivo, em terceira pessoa.\n"
        "Inclua pessoas, locais, datas, horários, veículos, valores, drogas, armas, ações, contradições e apreensões quando houver.\n"
        "Não escreva nada além das linhas iniciadas por '- '.\n"
    )

    user_prompt = (
        f"Depoente: {nome} ({papel})\n"
        f"Trecho {indice_chunk} de {total_chunks}\n"
        "DIÁLOGO BASE:\n"
        f"{texto_dialogo_chunk}\n\n"
        "Responda somente com linhas iniciadas por '- '."
    )

    resposta = chamar_llm(MODELO_LM_FAST, sys_prompt, user_prompt)
    return limpar_saida_llm(resposta)


def gerar_fatos_consolidados(texto_dialogo, nome, papel):
    chunks = dividir_em_chunks_com_sobreposicao(
        texto_dialogo,
        max_chars=MAX_CHARS_LLM_FASE2,
        overlap_chars=OVERLAP_CHARS
    )

    print(f"[INFO] Fase 2 será processada em {len(chunks)} chunk(s).")

    fatos = []
    vistos = set()

    for i, chunk in enumerate(chunks, start=1):
        resposta = gerar_fatos_estruturados_chunk(chunk, nome, papel, i, len(chunks))
        if not conteudo_valido(resposta, minimo=30):
            print(f"[ALERTA] Chunk {i} da Fase 2 falhou.")
            continue

        for linha in resposta.splitlines():
            linha = linha.strip()
            if not linha.startswith("- "):
                continue

            chave = linha.lower()
            if chave not in vistos:
                vistos.add(chave)
                fatos.append(linha)

    if not fatos:
        return None

    return "\n".join(fatos)


# ==============================================================================
# 8. FASE 3 - TERMO FINAL (MODELO FORTE)
# ==============================================================================
def gerar_termo_final_unico(fatos_consolidados, nome, papel):
    print("[LLM] Fase 3: Gerando termo final...")

    sys_prompt = (
        "Você é um redator jurídico.\n"
        "Redija um ÚNICO TERMO DE DEPOIMENTO coeso e completo.\n"
        "Regras:\n"
        "1. Produza somente o texto final.\n"
        "2. Não escreva comentários, notas ou preâmbulos.\n"
        f"3. Comece exatamente com: '{nome}, {papel}, ouvido em juízo, declarou que...'\n"
        "4. Escreva em terceira pessoa, em linguagem formal jurídica.\n"
        "5. Unifique os fatos em texto corrido, completo e coerente.\n"
        "6. Não use tópicos.\n"
        "7. Não invente fatos ausentes.\n"
        "8. Se houver conflito entre versões, registre isso de forma neutra.\n"
    )

    user_prompt = (
        "FATOS CONSOLIDADOS:\n"
        f"{fatos_consolidados}\n\n"
        "Escreva somente o termo final completo."
    )

    resposta = chamar_llm(MODELO_LM_FINAL, sys_prompt, user_prompt)
    return limpar_saida_llm(resposta)


# ==============================================================================
# 9. PIPELINE
# ==============================================================================
def processar_arquivo(caminho_arquivo, model, temp_dir, result_dir):
    nome_arquivo = os.path.basename(caminho_arquivo)
    nome, papel = limpar_nome_arquivo(nome_arquivo)

    print("=" * 80)
    print(f"[INÍCIO] Arquivo: {nome_arquivo}")
    print(f"[INFO] Nome extraído: {nome} | Papel: {papel}")

    audio_wav = converter_audio_ffmpeg(caminho_arquivo, temp_dir)
    if not audio_wav:
        print(f"[FALHA] Não foi possível converter o arquivo: {nome_arquivo}")
        return

    try:
        segmentos = transcrever_com_whisper(model, audio_wav)

        texto_bruto = "\n".join(
            f"{formatar_timestamp(s.start)} {s.text.strip()}"
            for s in segmentos if s.text and s.text.strip()
        )

        if not conteudo_valido(texto_bruto, minimo=100):
            print("[FALHA] Transcrição vazia ou muito curta.")
            return

        print(f"[DEBUG] Tamanho transcrição bruta: {len(texto_bruto)} chars")

        path_raw = os.path.join(result_dir, f"{nome}_1_bruto.txt")
        salvar_texto(path_raw, texto_bruto)

        texto_corrigido = gerar_diarizacao_corrigida_em_chunks(texto_bruto, nome, papel)
        if not conteudo_valido(texto_corrigido, minimo=200):
            print("[FALHA] Fase 1 não produziu diálogo válido.")
            return

        path_diag = os.path.join(result_dir, f"{nome}_2_dialogo.txt")
        salvar_texto(path_diag, texto_corrigido)

        fatos_consolidados = gerar_fatos_consolidados(texto_corrigido, nome, papel)
        if not conteudo_valido(fatos_consolidados, minimo=50):
            print("[FALHA] Fase 2 não produziu fatos válidos.")
            return

        path_fatos = os.path.join(result_dir, f"{nome}_3_fatos.txt")
        salvar_texto(path_fatos, fatos_consolidados)

        narrativa = gerar_termo_final_unico(fatos_consolidados, nome, papel)
        if not conteudo_valido(narrativa, minimo=200):
            print("[FALHA] Fase 3 não produziu narrativa final válida.")
            return

        path_narr = os.path.join(result_dir, f"{nome}_4_narrativa.txt")
        salvar_texto(path_narr, narrativa)

        print(f"[SUCESSO] Concluído: {nome_arquivo}")

    except Exception as e:
        print(f"[ERRO NO PROCESSO] {e}")
        traceback.print_exc()

    finally:
        if audio_wav and os.path.exists(audio_wav):
            try:
                os.remove(audio_wav)
            except Exception:
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
    print(f"LLM rápido: {MODELO_LM_FAST}")
    print(f"LLM final:  {MODELO_LM_FINAL}")

    try:
        from faster_whisper import WhisperModel

        print("Carregando modelo Whisper na GPU...")
        model = WhisperModel(MODELO_WHISPER, device="cuda", compute_type="float16")
        print("Modelo GPU carregado com sucesso!")

    except Exception as e:
        print(f"\n[ERRO GPU] Falha ao carregar CUDA: {e}")
        print("Tentando fallback para CPU (INT8)...")
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(MODELO_WHISPER, device="cpu", compute_type="int8")
        except Exception as e_cpu:
            print(f"Erro fatal também na CPU: {e_cpu}")
            return

    arquivos = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    if not arquivos:
        print("\nA pasta 'input' está vazia.")
        input("Enter para sair...")
        return

    for arq in arquivos:
        if arq.lower().endswith(('.txt', '.py', '.md')):
            continue
        processar_arquivo(os.path.join(input_dir, arq), model, temp_dir, result_dir)

    print("\nProcessamento finalizado.")
    input("Pressione Enter para fechar...")


if __name__ == "__main__":
    main()