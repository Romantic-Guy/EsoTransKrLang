# trans3.py
# ──────────────────────────────────────────────────────────────
# 1) 설정값 ­­­(필요에 따라 경로만 바꾸세요)
# ──────────────────────────────────────────────────────────────
MODEL_DIR   = "PATH"
SRC_LANG    = "eng_Latn"
TGT_LANG    = "kor_Hang"

MAX_INPUT_TOKENS = 1024        # 한 배치당 토큰 한도
SHOW_LIMIT       = 500         # 처음 N행만 콘솔 미리보기

INPUT_CSV  = "PATH"
OUTPUT_CSV = "PATH"
CHUNK_ROWS = 50_000            # ← 한 번에 읽어올 CSV 행 수
BATCH_SENT = 32                # ← 한 배치에 묶을 최대 문장 수
# ──────────────────────────────────────────────────────────────

import csv, time, signal, re, os
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 유틸 가져오기 (같은 폴더)
from translate_utils import (
    batch_by_tokens,
    translate_with_placeholders,
)

# ──────────────────────────────────────────────────────────────
# 2) 모델·토크나이저 로드
# ──────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    src_lang=SRC_LANG, tgt_lang=TGT_LANG,
    use_fast=True, local_files_only=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory={0: "24GB", 1: "24GB"},
    low_cpu_mem_usage=True,
    local_files_only=True
)
model.eval()
device = model.device

FORCED_BOS = (
    tokenizer.lang_code_to_id[TGT_LANG]
    if hasattr(tokenizer, "lang_code_to_id")
    else tokenizer.convert_tokens_to_ids(TGT_LANG)
)

# ──────────────────────────────────────────────────────────────
# 3) 번역 함수
# ──────────────────────────────────────────────────────────────
@torch.inference_mode()
def _model_translate(text_list: list[str]) -> list[str]:
    """플레이스홀더가 제거된 순수 텍스트만 입력받아 모델 호출"""
    inputs = tokenizer(
        text_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    ).to(device)

    outs = model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=1,  # 속도 최우선
        early_stopping=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2,
        forced_bos_token_id=FORCED_BOS
    )
    return tokenizer.batch_decode(outs, skip_special_tokens=True)

def translate_text(sentence: str) -> str:
    """<<…>> 블럭을 보존하며 한 문장 번역"""
    return translate_with_placeholders(sentence, _model_translate)

# ──────────────────────────────────────────────────────────────
# 4) CSV 처리
# ──────────────────────────────────────────────────────────────
META_RE = re.compile(r"""^\s*(
    EMOTE:          |   # 감정 표현
    ITEM:           |   # 아이템
    SFX:            |   # 효과음
    NPC_            |   # NPC_ 접두
    QUEST_          |   # 퀘스트
    ID:                 # ID:
)""", re.I | re.X)
def is_meta_line(s: str) -> bool:
    return META_RE.match(s) is not None

def process_csv(input_csv: str, output_csv: str):
    total = sum(1 for _ in open(input_csv, encoding="utf-8", errors="ignore")) - 1
    reader = pd.read_csv(
        input_csv,
        chunksize=CHUNK_ROWS,
        iterator=True,
        encoding="utf-8",
        on_bad_lines="skip",
        sep=",",
        quotechar='"'
    )

    interrupted = False
    def _sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n[!] Ctrl+C 감지 → 저장 후 종료합니다.")
    signal.signal(signal.SIGINT, _sigint)

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as fout:
        writer = None
        pbar = tqdm(total=total, unit="문장", mininterval=0.5)

        try:
            for chunk in reader:
                # ── 데이터 정리 ──
                chunk = chunk.reset_index(drop=True)
                chunk = chunk.dropna(axis=1, how="all")
                chunk = chunk.loc[:, ~chunk.columns.str.startswith("Unnamed")]

                sentences = chunk.iloc[:, 4].fillna("").astype(str).tolist()
                translated = []

                # ── 배치 단위 번역 ──
                for batch in batch_by_tokens(
                        sentences, tokenizer,
                        max_tokens=MAX_INPUT_TOKENS,
                        max_sentences=BATCH_SENT):
                    # 메타라인은 바로 통과
                    pure = [s for s in batch if not is_meta_line(s)]
                    pure_out = (
                        _model_translate(pure) if pure else []
                    )
                    it = iter(pure_out)
                    for s in batch:
                        if is_meta_line(s):
                            translated.append(s)
                        else:
                            translated.append(next(it))
                    pbar.update(len(batch))

                    # 실시간 로그
                    if SHOW_LIMIT and pbar.n <= SHOW_LIMIT:
                        for src, tgt in zip(batch, translated[-len(batch):]):
                            tqdm.write(f"[번역] {src}  →  {tgt}")

                # ── 결과 반영 ──
                chunk.iloc[:, 4] = [
                    t.replace("\n", " ") for t in translated
                ]

                # ── CSV 쓰기 ──
                if writer is None:
                    writer = csv.writer(
                        fout, quoting=csv.QUOTE_ALL, lineterminator="\n"
                    )
                    writer.writerow(chunk.columns)
                writer.writerows(chunk.values)

                if interrupted:
                    break
        finally:
            pbar.close()

    print("\n[완료] 총 처리 행:", total)

# ──────────────────────────────────────────────────────────────
# 5) 실행
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start = time.time()
    process_csv(INPUT_CSV, OUTPUT_CSV)
    print(f"\n소요 시간: {time.time()-start:.1f}초")
