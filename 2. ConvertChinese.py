import pandas as pd

# 변환 함수 (당신이 준 그대로)
def convert_ko_to_cn(text):
    """한국어를 텍스트로 변환 하는 코드입니다."""
    converted_text = ""
    if not isinstance(text, str):
        return text
    for char in text:
        utf8_val = ord(char)
        if 0xAC00 <= utf8_val <= 0xEA00:
            converted_char = chr(utf8_val - 0x3E00)
        else:
            converted_char = char
        converted_text += converted_char
    return converted_text

# CSV 파일 경로
file_path = r""

# CSV 읽기
df = pd.read_csv(file_path, encoding="utf-8-sig")

# 5번째 열(인덱스 4) 변환 적용
df.iloc[:, 4] = df.iloc[:, 4].apply(convert_ko_to_cn)

# 변환된 CSV 저장
output_path = r"E:\AI\Eso-Translation\Changing\U46lang_translated_cn.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("변환 완료! 저장 경로:", output_path)
