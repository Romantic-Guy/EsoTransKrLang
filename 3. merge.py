kr_lang_path =  r""
add_path = r""

# 병합된 파일을 저장할 경로
combined_csv_path = r""

# 파일을 순차적으로 읽고 쓰기
with open(kr_lang_path, 'r', encoding='utf-8') as kr_file, \
     open(add_path, 'r', encoding='utf-8') as add_file, \
     open(combined_csv_path, 'w', encoding='utf-8') as combined_file:
    
    # kr.lang 파일의 모든 내용을 먼저 씀
    combined_file.write(kr_file.read())

    # 그 뒤에 줄바꿈을 추가하여 구분을 명확히 함
    combined_file.write('\n')
    
    # add.csv 파일의 모든 내용을 이어서 씀
    combined_file.write(add_file.read())

print(f"병합된 CSV 파일이 '{combined_csv_path}' 경로에 저장되었습니다.")
