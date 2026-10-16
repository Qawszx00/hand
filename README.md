適用於專題：智慧手語人像轉語音系統 <br>
需使用虛擬環境 (mp_env\Scripts\activate)  <br>
插件安裝:<br>
pip install torch torchvision torchaudio<br>
pip install opencv-python (python 使用版本：3.9~3.11) <br>
pip install mediapipe (須確保 版本為0.10+以上) <br>
pip install numpy <br>
pip install pandas <br>
pip install matplotlib <br>
pip install flask <br>
新增資料 起用流程:collect.py ⭢ train.py ⭢ eval.py ⭢ app.py <br>
更新品質資料:collect_quality.py 更新品質標籤:label_quality.py <br>
無新增資料直接啟用:app.py 
