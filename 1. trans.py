import pandas as pd
import torch
import re
import threading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from queue import Queue, Empty
from time import sleep, time
from collections import defaultdict  

# ===============================
# 1. 파일 경로 설정
# ===============================
csv_path = r""
output_path = r""
model_path = r""

# ===============================
# 2. 마스킹/복원 함수 (플레이스홀더 + 포맷코드)
# ===============================
def mask_all(text: str):
    mapping = {}

    # 마스킹: <<...>>
    placeholders = re.findall(r"<<.*?>>", text)
    for i, ph in enumerate(placeholders):
        key = f"__PH_{i}__"
        text = text.replace(ph, key)
        mapping[key] = ph

    # 마스킹: |cFFFFFF...|r
    formats = re.findall(r"\|c[0-9A-Fa-f]{6}.*?\|r", text)
    for i, fmt in enumerate(formats):
        key = f"__FMT_{i}__"
        text = text.replace(fmt, key)
        mapping[key] = fmt

    return text, mapping

def unmask_all(text: str, mapping: dict):
    for key, val in mapping.items():
        text = text.replace(key, val)
    return text

# ===============================
# 2-1. 512 토큰 초과 분할 함수 (NLLB-200 입력 전용)
# ===============================
# tokenizer를 이용해 입력 토큰 기준으로 잘라 번역 후 재조합
# - add_special_tokens=False로 원문 토큰만 기준
# - 각 청크는 최대 512 토큰 이하 유지

def split_by_tokens(text: str, tokenizer, max_tokens: int = 512):
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = enc["input_ids"]
    if len(ids) <= max_tokens:
        return [text]

    chunks = []
    for i in range(0, len(ids), max_tokens):
        chunk_ids = ids[i : i + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

# ===============================
# 3. CSV 로딩 및 텍스트 준비
# ===============================
df = pd.read_csv(csv_path, header=None)
texts = df.iloc[:, 4].astype(str).tolist()
translated_texts = [None] * len(texts)
total = len(texts)

# ===============================
# 4. 모델 및 파이프라인 로드 (다중)
# ===============================
src_lang = "eng_Latn"
tgt_lang = "kor_Hang"
NUM_PIPELINES_PER_GPU = 1
INFER_BATCH = 8  # 한 번 호출에서 병렬 처리할 샘플 수(4/8/16/32 등으로 조절)
translators = []

for gpu in [0, 1]:
    for i in range(NUM_PIPELINES_PER_GPU):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(f"cuda:{gpu}")
        tokenizer.src_lang = src_lang
        translator = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_length=512,
            device=gpu,
        )
        translators.append((f"GPU{gpu}-{i+1}", translator))

# ===============================
# 4-1. 진행 중간 저장 함수 (Ctrl+C 포함)
# ===============================
def save_progress(path: str):
    # 번역된 것은 반영, 나머지는 원문 유지
    df_out = df.copy()
    df_out.iloc[:, 4] = [t if t is not None else orig for t, orig in zip(translated_texts, texts)]
    df_out.to_csv(path, index=False, header=False, encoding="utf-8-sig")

# ===============================
# 5. 동시 번역 스레드 함수
#    (여러 행을 한 번에 호출 + 512토큰 분할 + 중단 지원)
# ===============================
stop_event = threading.Event()

def translate_worker(name, translator, job_queue: Queue):
    tok = translator.tokenizer  # 파이프라인의 tokenizer 사용
    while not stop_event.is_set():
        try:
            indexes, batch_texts = job_queue.get_nowait()
        except Empty:
            break

        # 배치 전체를 한 번에 전처리(마스킹 + 512토큰 분할) → 하나의 flat 리스트로 합치기
        flat_inputs = []
        owners = []           # flat 입력이 어느 행(idx)에 속하는지
        row_mapping = {}
        row_originals = {}

        for idx, original_text in zip(indexes, batch_texts):
            if stop_event.is_set():
                break
            masked_text, mapping = mask_all(original_text)
            chunks = split_by_tokens(masked_text, tok, max_tokens=512)
            row_mapping[idx] = mapping
            row_originals[idx] = original_text
            for ch in chunks:
                flat_inputs.append(ch)
                owners.append(idx)

        if not flat_inputs:
            job_queue.task_done()
            continue

        try:
            #여러 행의 여러 청크를 한 번에 호출 → INFER_BATCH로 실제 병렬 처리
            results = translator(flat_inputs, batch_size=INFER_BATCH)

            # 행 단위로 재조합(입력 순서대로 결과가 나오므로 owners 기준으로 append)
            agg = defaultdict(list)
            for res, owner in zip(results, owners):
                agg[owner].append(res["translation_text"])

            # 행별 번역문 복원 및 출력(“번역 직후 반드시 보여주기” 유지)
            for idx in indexes:
                if idx in agg:
                    joined = "".join(agg[idx])
                    translated = unmask_all(joined, row_mapping[idx])
                    translated_texts[idx] = translated
                    print(f"[{name}][{idx+1}/{total}] {row_originals[idx]}\n → {translated}\n")

        except Exception as e:
            print(f"[{name}] 에러 발생 (rows {indexes[0]}-{indexes[-1]}): {e}")

        job_queue.task_done()

# ===============================
# 6. 작업 큐 준비 및 실행
# ===============================
batch_size = 128  # 큐에 넣는 '행' 단위 배치 크기(워커가 한 번에 처리 시도)
job_queue = Queue()

for i in range(0, total, batch_size):
    batch_indexes = list(range(i, min(i + batch_size, total)))
    batch_texts = [texts[j] for j in batch_indexes]
    job_queue.put((batch_indexes, batch_texts))

start_time = time()
threads = []

try:
    for name, translator in translators:
        t = threading.Thread(target=translate_worker, args=(name, translator, job_queue), daemon=True)
        t.start()
        threads.append(t)

    while any(t.is_alive() for t in threads):
        done = sum(1 for t in translated_texts if t is not None)
        percent = (done / total * 100) if total else 0.0
        elapsed = time() - start_time
        if done > 0:
            est_total_time = elapsed / done * total
            remaining = max(0.0, est_total_time - elapsed)
            print(
                f"\r 진행률: {done}/{total} ({percent:.2f}%) | 경과: {elapsed:.1f}s | 남은 예상: {remaining:.1f}s",
                end="",
                flush=True,
            )
        else:
            print(f"\r 진행률: {done}/{total} (0.00%) | 경과: {elapsed:.1f}s", end="", flush=True)
        sleep(2)

except KeyboardInterrupt:
    print("\n 인터럽트 감지됨! 진행분 저장 중…")
    stop_event.set()
    #  즉시 부분 저장
    save_progress(output_path)

finally:
    for t in threads:
        t.join(timeout=5)
    #  종료 시 최종 저장 (중복 저장 허용)
    save_progress(output_path)
    print(f"\n 저장 완료: {output_path}")
