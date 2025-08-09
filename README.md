# EsoTransKrLang

안녕하세요 kr.lang 한글 번역 입니다.(기존 U42까지는 기존에 유저번역+인공지능번역+기계번역 등으로 작업된 kr.lang에서 U46까지 nllb-200이라는 비상업용 번역 AI모델을 로컬에 구축하여 번역을 진행하였습니다.) 기존 한글 패치를 제작해주시고, 번역을 진행해주시며 고생해주신 모든 분들에게 깊은 감사를 드립니다.

번역 모델 라이선스 : Licensed under CC‑BY‑NC 4.0(NLLB‑200 distilled‑600M (Meta AI 제공))


**준비물
1. en.lang , kr.lang(기존 kr.lang), EsoExtractdata

**순서
1. EsoExtractData를 활용하여 kr.lang에는 없지만 en.lang에 있는 차이를 추출하고 csv데이터로 변경  (추출 데이터를 Diff.lang.csv 라고 가명 짓겠음.)
2. Diff.lang.csv파일을 1. trans.py로 번역을 진행
3. 번역을 진행한.csv파일을 2. ConvertChinese.py를 통해 한자로 변경
4. 변형.csv파일을 3. merge.py를 통해 기존 kr.lang과 병합
5. merged.csv파일을  4. last.py를 통해 lang으로 교체할수있게 변형
6. EsoExtractData를 통해 다시 last.csv를 Lang으로 저장
7. 기존 Addons->GameData내에 kr.lang을 번역한 kr.lang으로 교체



-'데이터 정리' 폴더 내에 last.py(By DIPOON)는 차차 길드원분의 DIPOON님의 깃허브 소스코드(BackslashQuotationRemover.py)를 참고하였습니다.

현재 U46까지 진행되었으며 잘못 번역되거나 오류난 곳이 많기 때문에 시간이 있을 때 수정하여 재업로드 하도록 하겠습니다.
다운로드 링크 : https://drive.google.com/file/d/1RQz8CmP3o83pXUnRROgfg3enW6deG7s_/view










2025-04-27
데이터 번역에 새로운 데이터 번역 코드를 추가하였습니다.
다중 GPU를 활용하고 있는 유저는 AItrans.py를 활용하여 고품질 AI번역이 가능합니다.
모델은 nllb-200이라는 비상업 모델을 사용합니다. 참고해주시길 바랍니다.
