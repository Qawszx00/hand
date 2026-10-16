適用於專題：智慧手語人像轉語音系統
需使用虛擬環境 (mp_env\Scripts\activate) 
插件安裝:
pip install torch torchvision torchaudio
pip install opencv-python (python 使用版本：3.9~3.11)
pip install mediapipe (須確保 版本為0.10+以上)
pip install numpy
pip install pandas
pip install matplotlib
pip install flask
新增資料 起用流程:collect.py ⭢ train.py ⭢ eval.py ⭢ app.py
更新品質資料:collect_quality.py 更新品質標籤:label_quality.py
無新增資料直接啟用:app.py
