# EsoTransKrLang

안녕하세요 kr.lang 한글 번역 입니다.(U44까지만 진행 u42는 빠져있음, U41까지는 기존에 유저번역+인공지능번역+기계번역 등으로 작업된 기존 작업물에서 U43,U44만 번역 진행함.)
본 코드는 번역을 위한 참고용도입니다.


**준비물
1. en.lang , kr.lang(기존 kr.lang), EsoExtractdata

**순서
1. EsoExtractData를 활용하여 kr.lang에는 없지만 en.lang에 있는 차이를 추출하고 csv데이터로 변경  (추출 데이터를 Diff.lang.csv 라고 가명 짓겠음.)
2. Diff.lang.csv파일을 '번역 전 데이터 전처리'폴더 내 파이썬 코드를 통해 가공 (폴더 내 파이썬 코드에서 경로를 지정하고 순서대로 실행)
3. Diff.lang.csv파일을 '데이터 번역'폴더 내 파이썬 코드를 통해 통해 영어->한글->한문으로 가공 (폴더 내 파이썬 코드에서 경로를 지정하고 순서대로 실행)
4. Diff.lang.csv파일을 '번역 후 데이터 전처리'폴더 내 파이썬 코드를 통해 가공 (폴더 내 파이썬 코드에서 경로를 지정하고 순서대로 실행)
5. Diff.lang.csv파일과 kr.lang파일을 '데이터 정리'폴더 내 파이썬 코드를 통해 합치고 가공(폴더 내 파이썬 코드에서 경로를 지정하고 순서대로 실행)
6. EsoExtractData를 통해 다시 csv를 다시 Lang으로 저장
7. 기존 Addons->GameData내에 kr.lang을 번역한 kr.lang으로 교체


*그외
-파이썬으로 무언가 작업해본게 처음이라 매우 미숙합니다.(데이터 전처리도 미숙함) 차후u45나 u46때 좀더 다듬고 GUI도 써보면서 좀더 쉽게 할 수 있도록 해보겠습니다.

-'데이터 정리' 폴더 내에 LastDataClear2(By DIPOON)는 차차 길드원분의 DIPOON님의 깃허브 소스코드(BackslashQuotationRemover.py)를 참고하였습니다.

현재 U44까지 진행되었으며 u42는 부분만 빠져 있습니다. 조만간 수정해서 올리겠습니다.
다운로드 링크 : https://drive.google.com/file/d/1RQz8CmP3o83pXUnRROgfg3enW6deG7s_/view










2025-04-27
데이터 번역에 새로운 데이터 번역 코드를 추가하였습니다.
다중 GPU를 활용하고 있는 유저는 AItrans.py를 활용하여 고품질 AI번역이 가능합니다.
모델은 nllb-200이라는 비상업 모델을 사용합니다. 참고해주시길 바랍니다.
