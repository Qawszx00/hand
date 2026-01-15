from flask import Flask, render_template, redirect, url_for

# 創建 Flask 應用實例
app = Flask(__name__)

# 定義首頁路由 (Endpoint: index)
@app.route('/')
def index():
    # 渲染 index.html 模板
    return render_template('in.html')

# 定義手語教學頁面的路由 (Endpoint: sign_language_teaching)
@app.route('/teaching')
def teaching():
    # 渲染 teaching.html 模板 (第二頁)
    return render_template('ch.html')

# 【新增】定義手語考試頁面的路由 (Endpoint: sign_language_exam)
# 這個函數名稱必須與您在 HTML 中使用的 url_for('...') 參數一致
@app.route('/exam')
def exam():
    # 您可以在這裡渲染一個新的 HTML 模板 (例如 'exam.html')
    return render_template('ix.html')

@app.route('/start')  # URL 可以自訂，但不影響 Endpoint 名稱
def start(): # <--- 這個函數名稱是關鍵！
    # 渲染 ex.html
    return render_template('ex.html')

@app.route('/tech')  # URL 可以自訂，但不影響 Endpoint 名稱
def tech(): # <--- 這個函數名稱是關鍵！
    # 渲染 ex.html
    return render_template('te.html')

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
