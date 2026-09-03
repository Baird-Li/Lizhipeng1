"""
项目名称：芯片缺陷检测 Web 应用（Flask 后端）
模块功能：智能检测、多版本模型切换、错误样本去重与历史记录、直接标记真实类别
对应课程章节：第1章、第7章、第8章
"""

import os
import csv
import json
import sqlite3
import shutil
import time
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template_string

# ==================== 全局配置 ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_DIR = "uploads"
ERROR_DIR = "error_samples"
DB_PATH = "predictions.db"
CLASS_NAMES = ["ZF-scratch", "broken", "pinbreak", "scratch"]

# 中文名称映射
CLASS_NAMES_CN = {
    "ZF-scratch": "ZF划痕",
    "broken": "破损",
    "pinbreak": "引脚断裂",
    "scratch": "划痕"
}

# 模型配置
MODEL_DIR = "models"
ORIGINAL_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
VERSIONS_DIR = os.path.join(MODEL_DIR, "versions")
VERSIONS_JSON = os.path.join(VERSIONS_DIR, "versions.json")

# 当前模型状态
current_model = None
current_model_name = "原始模型"
current_model_path = ORIGINAL_MODEL_PATH

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)

# ==================== 数据库 ====================
def init_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name VARCHAR(255),
            image_path VARCHAR(500),
            predicted_class VARCHAR(50),
            confidence FLOAT,
            all_probs TEXT,
            is_correct BOOLEAN DEFAULT 1,
            true_class VARCHAR(50),
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            reported_at DATETIME
        )
    ''')
    c.execute("PRAGMA table_info(predictions)")
    columns = [col[1] for col in c.fetchall()]
    if 'true_class' not in columns:
        c.execute('ALTER TABLE predictions ADD COLUMN true_class VARCHAR(50)')
    conn.commit()
    conn.close()

# ==================== 模型加载（支持多版本） ====================
def load_model_by_path(model_path, model_name):
    """加载指定路径的模型"""
    if not os.path.exists(model_path):
        print(f"⚠️ 模型文件不存在: {model_path}")
        return None
    
    try:
        model = models.resnet18()
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 4)
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"✅ 已加载模型: {model_name} ({model_path})")
        return model
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None

def get_all_versions():
    """获取所有可用模型版本列表"""
    versions = []
    # 原始模型
    if os.path.exists(ORIGINAL_MODEL_PATH):
        versions.append({
            "name": "原始模型",
            "path": ORIGINAL_MODEL_PATH,
            "type": "original",
            "timestamp": "初始训练",
            "val_acc": None,
            "sample_count": None
        })
    
    # 读取优化版本
    if os.path.exists(VERSIONS_JSON):
        try:
            with open(VERSIONS_JSON, 'r', encoding='utf-8') as f:
                opt_versions = json.load(f)
                for v in opt_versions:
                    if os.path.exists(v["path"]):
                        versions.append({
                            "name": v["name"],
                            "path": v["path"],
                            "type": "optimized",
                            "timestamp": v["timestamp"],
                            "val_acc": v["val_acc"],
                            "sample_count": v.get("sample_count", 0)
                        })
        except:
            pass
    
    return versions

def load_default_model():
    global current_model, current_model_name, current_model_path
    # 默认加载原始模型
    current_model = load_model_by_path(ORIGINAL_MODEL_PATH, "原始模型")
    if current_model is not None:
        current_model_name = "原始模型"
        current_model_path = ORIGINAL_MODEL_PATH
    else:
        current_model = None
        current_model_name = "未加载"
        current_model_path = ""

load_default_model()

# ==================== 预处理 ====================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image_path):
    global current_model
    if current_model is None:
        load_default_model()
        if current_model is None:
            raise RuntimeError("模型未加载")
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = current_model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    confidence_val = confidence.item()
    predicted_idx = predicted.item()
    probs_list = probs.cpu().numpy()[0].tolist()
    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": round(confidence_val, 4),
        "all_probs": {CLASS_NAMES[i]: round(probs_list[i], 4) for i in range(len(CLASS_NAMES))}
    }

# ==================== 数据库操作 ====================
def save_prediction_to_db(image_name, image_path, predicted_class, confidence, all_probs):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (image_name, image_path, predicted_class, confidence, all_probs)
        VALUES (?, ?, ?, ?, ?)
    ''', (image_name, image_path, predicted_class, confidence, json.dumps(all_probs)))
    conn.commit()
    conn.close()

def get_recent_predictions(limit=200, start_date='', end_date=''):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    query = '''
        SELECT id, image_name, image_path, predicted_class, confidence, all_probs, is_correct, true_class, created_at
        FROM predictions
        WHERE 1=1
    '''
    params = []
    
    if start_date:
        query += ' AND DATE(created_at) >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND DATE(created_at) <= ?'
        params.append(end_date)
    
    query += ' ORDER BY created_at DESC LIMIT ?'
    params.append(limit)
    
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    
    predictions = []
    for row in rows:
        predictions.append({
            "id": row["id"],
            "image_name": row["image_name"],
            "image_path": row["image_path"],
            "predicted_class": row["predicted_class"],
            "confidence": row["confidence"],
            "all_probs": json.loads(row["all_probs"]),
            "is_correct": row["is_correct"],
            "true_class": row["true_class"],
            "created_at": row["created_at"]
        })
    return predictions

# ===== 错误样本：按 image_name 去重 =====
def get_error_samples():
    """获取错误样本（按 image_name 去重）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''
        SELECT id, image_name, image_path, predicted_class, confidence, all_probs, true_class, created_at, reported_at
        FROM predictions
        WHERE is_correct = 0
        ORDER BY reported_at DESC
    ''')
    rows = c.fetchall()
    conn.close()
    
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_name"]].append({
            "id": row["id"],
            "image_path": row["image_path"],
            "predicted_class": row["predicted_class"],
            "confidence": row["confidence"],
            "all_probs": json.loads(row["all_probs"]),
            "true_class": row["true_class"] or "",
            "created_at": row["created_at"],
            "reported_at": row["reported_at"]
        })
    
    samples = []
    for image_name, history in groups.items():
        history_sorted = sorted(history, key=lambda x: x["reported_at"] if x["reported_at"] else "", reverse=True)
        latest = history_sorted[0]
        
        samples.append({
            "id": latest["id"],
            "image_name": image_name,
            "image_path": latest["image_path"],
            "predicted_class": latest["predicted_class"],
            "confidence": latest["confidence"],
            "all_probs": latest["all_probs"],
            "true_class": latest["true_class"] or "",
            "created_at": latest["created_at"],
            "reported_at": latest["reported_at"],
            "report_count": len(history_sorted),
            "history": history_sorted
        })
    
    return samples

def get_error_sample_history(image_name):
    """获取指定图片的所有错误报告历史"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''
        SELECT id, predicted_class, confidence, true_class, created_at, reported_at
        FROM predictions
        WHERE image_name = ? AND is_correct = 0
        ORDER BY reported_at DESC
    ''', (image_name,))
    rows = c.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "predicted_class": row["predicted_class"],
            "confidence": row["confidence"],
            "true_class": row["true_class"] or "",
            "created_at": row["created_at"],
            "reported_at": row["reported_at"]
        })
    return history

# ===== 直接从数据库重建 CSV =====
def rebuild_csv_from_db():
    """从数据库读取所有错误记录，重建 CSV 文件"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''
        SELECT image_name, image_path, predicted_class, true_class, reported_at, created_at
        FROM predictions
        WHERE is_correct = 0
        ORDER BY reported_at DESC
    ''')
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        csv_path = os.path.join(ERROR_DIR, "error_log.csv")
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
            except:
                pass
        return
    
    csv_path = os.path.join(ERROR_DIR, "error_log.csv")
    for retry in range(3):
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["时间戳", "原始文件名", "预测类别", "保存路径", "真实类别"])
                for row in rows:
                    timestamp = row["reported_at"] if row["reported_at"] else row["created_at"]
                    writer.writerow([
                        timestamp,
                        row["image_name"],
                        row["predicted_class"],
                        row["image_path"],
                        row["true_class"] or ""
                    ])
            print(f"✅ CSV 已重建: {len(rows)} 条记录")
            break
        except PermissionError:
            if retry < 2:
                time.sleep(0.5)
                continue
            else:
                print(f"⚠️ CSV 重建失败（文件被占用）: {csv_path}")
                break

def mark_prediction_as_error_with_true_class(image_name, predicted_class, true_class, image_url):
    """标记错误并设置真实类别，然后重建 CSV"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('SELECT id, is_correct FROM predictions WHERE image_name = ?', (image_name,))
    row = c.fetchone()
    
    if row:
        if row[1] == 0:
            c.execute('''
                UPDATE predictions
                SET true_class = ?
                WHERE image_name = ?
            ''', (true_class, image_name))
        else:
            c.execute('''
                UPDATE predictions
                SET is_correct = 0, true_class = ?, reported_at = ?
                WHERE image_name = ?
            ''', (true_class, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_name))
    else:
        c.execute('''
            INSERT INTO predictions (image_name, image_path, predicted_class, true_class, is_correct, reported_at)
            VALUES (?, ?, ?, ?, 0, ?)
        ''', (image_name, image_url, predicted_class, true_class, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    conn.commit()
    conn.close()
    
    save_error_sample(image_url, predicted_class, image_name)
    rebuild_csv_from_db()
    return 1

def mark_prediction_as_error(image_name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        UPDATE predictions
        SET is_correct = 0, reported_at = ?
        WHERE image_name = ?
    ''', (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_name))
    conn.commit()
    conn.close()
    rebuild_csv_from_db()

def get_error_count():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(DISTINCT image_name) FROM predictions WHERE is_correct = 0')
    count = c.fetchone()[0]
    conn.close()
    return count

def clear_history_by_date(start_date='', end_date=''):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    query = 'DELETE FROM predictions WHERE 1=1'
    params = []
    if start_date:
        query += ' AND DATE(created_at) >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND DATE(created_at) <= ?'
        params.append(end_date)
    c.execute(query, params)
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return deleted

def clear_all_errors():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM predictions WHERE is_correct = 0')
    deleted_count = c.rowcount
    conn.commit()
    conn.close()
    
    if os.path.exists(ERROR_DIR):
        try:
            shutil.rmtree(ERROR_DIR)
            os.makedirs(ERROR_DIR)
        except Exception as e:
            print(f"清空错误样本文件夹失败: {e}")
    
    return deleted_count

# ==================== 文件保存 ====================
def save_uploaded_image(file):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = os.path.basename(file.filename)
    unique_name = f"{timestamp}_{safe_name}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    file.save(save_path)
    return f"/uploads/{unique_name}"

def save_error_sample(image_url, predicted_class, original_name):
    error_class_dir = os.path.join(ERROR_DIR, predicted_class)
    os.makedirs(error_class_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = os.path.basename(original_name)
    error_filename = f"{timestamp}_{safe_name}"
    error_path = os.path.join(error_class_dir, error_filename)
    
    if image_url and image_url.startswith("/uploads/"):
        actual_path = os.path.join(UPLOAD_DIR, image_url.replace("/uploads/", ""))
        if os.path.exists(actual_path):
            try:
                shutil.copy2(actual_path, error_path)
            except Exception as e:
                print(f"❌ 复制错误图片失败: {e}")

# ==================== Flask 应用 ====================
app = Flask(__name__)
init_database()
rebuild_csv_from_db()

# ==================== 路由 ====================
@app.route('/')
def index():
    return render_template_string(HTML_CONTENT)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route('/predict', methods=['POST'])
def predict_single():
    global current_model
    if current_model is None:
        load_default_model()
        if current_model is None:
            return jsonify({"success": False, "message": "模型未加载"}), 500
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "未找到图片"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "message": "文件名为空"}), 400
    try:
        image_url = save_uploaded_image(file)
        actual_path = os.path.join(UPLOAD_DIR, os.path.basename(image_url))
        pred = predict_image(actual_path)
        save_prediction_to_db(file.filename, image_url, pred["predicted_class"], pred["confidence"], pred["all_probs"])
        return jsonify({
            "success": True,
            "prediction": pred,
            "image_url": image_url,
            "image_name": file.filename
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/get_versions', methods=['GET'])
def get_versions():
    """获取所有可用模型版本"""
    versions = get_all_versions()
    return jsonify({
        "success": True,
        "versions": versions,
        "current": current_model_name
    })

@app.route('/switch_model', methods=['POST'])
def switch_model():
    global current_model, current_model_name, current_model_path
    
    data = request.get_json()
    model_path = data.get('path')
    model_name = data.get('name')
    
    if not model_path:
        return jsonify({"success": False, "message": "缺少模型路径"}), 400
    
    # 加载模型
    model = load_model_by_path(model_path, model_name)
    if model is not None:
        current_model = model
        current_model_name = model_name
        current_model_path = model_path
        return jsonify({
            "success": True,
            "model_name": model_name,
            "model_path": model_path,
            "message": f"已切换到 {model_name}"
        })
    else:
        return jsonify({
            "success": False,
            "message": f"加载 {model_name} 失败，请检查模型文件是否存在"
        }), 500

@app.route('/get_current_model', methods=['GET'])
def get_current_model():
    return jsonify({
        "success": True,
        "model_name": current_model_name,
        "model_path": current_model_path
    })

@app.route('/history', methods=['GET'])
def get_history():
    limit = request.args.get('limit', 200, type=int)
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    predictions = get_recent_predictions(limit, start_date, end_date)
    return jsonify({
        "success": True,
        "predictions": predictions,
        "error_count": get_error_count()
    })

@app.route('/history/clear', methods=['DELETE'])
def clear_history():
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    try:
        deleted = clear_history_by_date(start_date, end_date)
        return jsonify({"success": True, "deleted_count": deleted})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/error_samples', methods=['GET'])
def get_error_samples_api():
    samples = get_error_samples()
    return jsonify({
        "success": True,
        "samples": samples,
        "total": len(samples)
    })

@app.route('/error_history/<image_name>', methods=['GET'])
def get_error_history_api(image_name):
    history = get_error_sample_history(image_name)
    return jsonify({
        "success": True,
        "image_name": image_name,
        "history": history
    })

@app.route('/clear_errors', methods=['DELETE'])
def clear_errors():
    try:
        deleted_count = clear_all_errors()
        rebuild_csv_from_db()
        return jsonify({
            "success": True,
            "deleted_count": deleted_count,
            "message": f"已清空 {deleted_count} 个错误样本"
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/report_error', methods=['POST'])
def report_error():
    data = request.get_json() if request.is_json else request.form
    image_name = data.get('image_name')
    predicted_class = data.get('predicted_class')
    image_url = data.get('image_url')
    if not image_name or not predicted_class:
        return jsonify({"success": False, "message": "缺少参数"}), 400
    try:
        mark_prediction_as_error(image_name)
        save_error_sample(image_url, predicted_class, image_name)
        return jsonify({
            "success": True, 
            "message": "已记录错误样本",
            "error_count": get_error_count()
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/mark_true_class', methods=['POST'])
def mark_true_class():
    data = request.get_json() if request.is_json else request.form
    image_name = data.get('image_name')
    predicted_class = data.get('predicted_class')
    true_class = data.get('true_class')
    image_url = data.get('image_url')
    
    if not image_name or not true_class:
        return jsonify({"success": False, "message": "缺少参数"}), 400
    if true_class not in CLASS_NAMES:
        return jsonify({"success": False, "message": "无效的类别"}), 400
    
    try:
        mark_prediction_as_error_with_true_class(image_name, predicted_class, true_class, image_url)
        return jsonify({
            "success": True,
            "message": f"已标记真实类别为: {true_class}",
            "image_name": image_name,
            "predicted_class": predicted_class,
            "true_class": true_class,
            "error_count": get_error_count()
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ==================== 前端 HTML ====================
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>芯片缺陷检测系统</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 20px; background-color: #f5f5f5; }
        .tab-content { padding: 20px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .preview-img { width: 100px; height: 100px; object-fit: cover; margin: 5px; border: 2px solid #ddd; border-radius: 4px; cursor: pointer; transition: transform 0.2s; }
        .preview-img:hover { transform: scale(1.05); border-color: #007bff; }
        .result-card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
        .low-confidence { border-color: #ffc107; }
        .modal-body img { max-width: 100%; }
        .prob-bar { margin-bottom: 8px; }
        .history-item { cursor: pointer; transition: background-color 0.2s; }
        .history-item:hover { background-color: #f0f0f0; }
        .upload-area { border: 2px dashed #ccc; border-radius: 8px; padding: 30px; text-align: center; transition: 0.3s; cursor: pointer; }
        .upload-area:hover { border-color: #007bff; background: #f8f9fa; }
        .upload-area.dragover { border-color: #007bff; background: #e3f2fd; }
        .preview-container { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px; }
        .preview-item { position: relative; width: 110px; }
        .preview-item .remove-btn { position: absolute; top: -8px; right: -8px; width: 22px; height: 22px; border-radius: 50%; background: #dc3545; color: white; border: none; font-size: 14px; line-height: 22px; text-align: center; cursor: pointer; padding: 0; }
        .preview-item .remove-btn:hover { background: #c82333; }
        .preview-item .file-name { font-size: 10px; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 110px; }
        .summary-card { background: #f8f9fa; border-radius: 8px; padding: 15px; margin-top: 15px; }
        .summary-stat { display: inline-block; padding: 5px 15px; margin: 3px; background: white; border-radius: 20px; border: 1px solid #ddd; }
        .summary-stat .count { font-weight: bold; font-size: 18px; }
        .btn-group .btn { font-size: 0.8rem; }
        .date-filter-box { background: #f8f9fa; padding: 12px 15px; border-radius: 8px; }
        .error-sample-card { border-left: 4px solid #dc3545; }
        .error-stats { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px; }
        .error-stat-item { background: white; padding: 8px 16px; border-radius: 20px; border: 1px solid #dc3545; }
        .error-stat-item .count { font-weight: bold; color: #dc3545; }
        .marked-true { border-left: 4px solid #28a745; }
        .true-class-select { display: inline-block; width: auto; padding: 2px 6px; font-size: 0.8rem; }
        .btn-group-actions { display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap; }

        /* 模型选择下拉样式 */
        .model-select-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            padding: 12px 20px;
            background: #f0f4f8;
            border-radius: 10px;
            margin-bottom: 15px;
            border: 1px solid #e0e7ef;
            flex-wrap: wrap;
        }
        .model-select-container label {
            font-weight: 500;
            color: #555;
            margin: 0;
        }
        .model-select-container select {
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid #ccc;
            background: white;
            font-size: 14px;
            min-width: 200px;
        }
        .model-select-container .model-info {
            font-size: 13px;
            color: #888;
        }
        .model-select-container .badge {
            font-size: 13px;
            padding: 4px 14px;
            border-radius: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-3 text-center">🔬 芯片表面缺陷智能检测系统</h1>
        <p class="text-center text-muted">基于 CNN 的缺陷分类 · 制造智能课程设计</p>

        <!-- 模型切换下拉菜单 -->
        <div class="model-select-container">
            <label for="modelSelect">🧠 选择模型：</label>
            <select id="modelSelect" class="form-select form-select-sm" style="width:auto; display:inline-block;">
                <option value="">加载中...</option>
            </select>
            <span class="badge bg-secondary" id="modelStatus">当前: 原始模型</span>
            <span class="model-info" id="modelInfo"></span>
        </div>

        <div class="alert alert-danger d-flex justify-content-between align-items-center" id="error-alert" style="display: none;">
            <span>📌 已收集错误样本: <strong id="error-count">0</strong> 个</span>
            <span>感谢您的反馈，帮助改进模型！</span>
        </div>

        <ul class="nav nav-tabs" id="mainTabs" role="tablist">
            <li class="nav-item"><button class="nav-link active" id="detect-tab" data-bs-toggle="tab" data-bs-target="#detect-panel" type="button">📤 智能检测</button></li>
            <li class="nav-item"><button class="nav-link" id="history-tab" data-bs-toggle="tab" data-bs-target="#history-panel" type="button">📋 识别历史</button></li>
            <li class="nav-item"><button class="nav-link" id="error-tab" data-bs-toggle="tab" data-bs-target="#error-panel" type="button">⚠️ 错误样本</button></li>
        </ul>

        <div class="tab-content mt-3">
            <!-- 智能检测 -->
            <div class="tab-pane fade show active" id="detect-panel">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span>📤 上传图片</span>
                        <span class="badge bg-secondary" id="fileCount">已选 0 张</span>
                    </div>
                    <div class="card-body">
                        <div class="upload-area" id="uploadArea">
                            <div style="font-size: 48px;">📁</div>
                            <p class="mb-1"><strong>点击选择图片</strong> 或拖拽图片到此处</p>
                            <p class="text-muted small">支持 JPG / PNG / BMP，可多选</p>
                            <input type="file" id="fileInput" accept="image/*" multiple style="display: none;">
                        </div>

                        <div class="btn-group-actions">
                            <button class="btn btn-primary btn-lg" id="detectBtn" disabled>🚀 开始检测</button>
                            <button class="btn btn-outline-secondary btn-lg" id="clearBtn">🗑️ 清空</button>
                            <span class="text-muted ms-2 small align-self-center" id="detectHint">请先选择图片</span>
                        </div>

                        <div class="preview-container" id="previewContainer"></div>

                        <div id="detectProgress" class="mt-3" style="display: none;">
                            <div class="progress"><div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%">0%</div></div>
                            <p class="text-muted small mt-1" id="progressText">准备中...</p>
                        </div>
                        <div id="detectResults" class="mt-3"></div>
                        <div id="detectSummary" class="mt-3"></div>
                    </div>
                </div>
            </div>

            <!-- 识别历史 -->
            <div class="tab-pane fade" id="history-panel">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span>📋 识别历史</span>
                        <button class="btn btn-outline-danger btn-sm" id="clearHistoryBtn">🗑️ 清空全部</button>
                    </div>
                    <div class="card-body">
                        <div class="date-filter-box mb-3">
                            <label class="form-label fw-bold mb-2">📅 按日期筛选</label>
                            <div class="row g-2 align-items-center">
                                <div class="col-auto"><label class="form-label mb-0 small">从</label></div>
                                <div class="col"><input type="date" class="form-control form-control-sm" id="startDate"></div>
                                <div class="col-auto"><label class="form-label mb-0 small">到</label></div>
                                <div class="col"><input type="date" class="form-control form-control-sm" id="endDate"></div>
                                <div class="col-auto"><button class="btn btn-sm btn-outline-secondary" id="resetDateBtn">重置</button></div>
                            </div>
                            <div class="mt-2"><span class="badge bg-info" id="dateRangeInfo">📅 显示全部记录</span></div>
                        </div>
                        <div id="historyList" class="list-group"><p class="text-muted">加载中...</p></div>
                    </div>
                </div>
            </div>

            <!-- 错误样本 -->
            <div class="tab-pane fade" id="error-panel">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span>⚠️ 错误样本</span>
                        <div>
                            <button class="btn btn-outline-danger btn-sm" id="clearErrorsBtn">🗑️ 清空全部</button>
                            <span class="badge bg-danger" id="errorTotal">共 0 个</span>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="error-stats" id="errorStats"></div>
                        <div id="errorSampleList"><p class="text-muted">加载中...</p></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- 历史详情模态框 -->
    <div class="modal fade" id="historyDetailModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header"><h5 class="modal-title">📊 预测详情</h5><button type="button" class="btn-close" data-bs-dismiss="modal"></button></div>
                <div class="modal-body" id="detailModalBody"></div>
            </div>
        </div>
    </div>

    <!-- 错误样本历史模态框 -->
    <div class="modal fade" id="errorHistoryModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header"><h5 class="modal-title">📋 错误报告历史</h5><button type="button" class="btn-close" data-bs-dismiss="modal"></button></div>
                <div class="modal-body" id="errorHistoryBody"></div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // ===== 全局 =====
        let selectedFiles = [];
        let currentStartDate = '';
        let currentEndDate = '';
        const CLASS_LIST = ['ZF-scratch', 'broken', 'pinbreak', 'scratch'];
        const CLASS_CN = {'ZF-scratch':'ZF划痕','broken':'破损','pinbreak':'引脚断裂','scratch':'划痕'};

        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');
        const previewContainer = document.getElementById('previewContainer');
        const fileCount = document.getElementById('fileCount');
        const detectBtn = document.getElementById('detectBtn');
        const detectHint = document.getElementById('detectHint');
        const detectProgress = document.getElementById('detectProgress');
        const progressBar = detectProgress.querySelector('.progress-bar');
        const progressText = document.getElementById('progressText');
        const detectResults = document.getElementById('detectResults');
        const detectSummary = document.getElementById('detectSummary');

        const startDateInput = document.getElementById('startDate');
        const endDateInput = document.getElementById('endDate');
        const dateRangeInfo = document.getElementById('dateRangeInfo');

        const modelSelect = document.getElementById('modelSelect');
        const modelStatus = document.getElementById('modelStatus');
        const modelInfo = document.getElementById('modelInfo');

        // ===== 加载模型列表 =====
        function loadModelVersions() {
            fetch('/get_versions')
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        const versions = data.versions;
                        let html = '';
                        versions.forEach((v, idx) => {
                            const selected = (v.name === data.current) ? 'selected' : '';
                            let label = v.name;
                            if (v.type === 'optimized' && v.val_acc !== null) {
                                label += ` (${(v.val_acc*100).toFixed(2)}%)`;
                            }
                            if (v.type === 'optimized' && v.timestamp) {
                                label += ` ${v.timestamp}`;
                            }
                            html += `<option value="${v.path}" data-name="${v.name}" ${selected}>${label}</option>`;
                        });
                        modelSelect.innerHTML = html;
                        // 更新当前模型状态显示
                        updateModelStatus();
                    }
                })
                .catch(() => {
                    modelSelect.innerHTML = '<option value="">加载失败</option>';
                });
        }

        function updateModelStatus() {
            const selected = modelSelect.options[modelSelect.selectedIndex];
            if (selected) {
                const name = selected.dataset.name || selected.text;
                modelStatus.textContent = `当前: ${name}`;
                // 显示更多信息
                const opt = modelSelect.options[modelSelect.selectedIndex];
                const path = opt.value;
                // 从路径中提取版本号或时间
                let info = '';
                if (path.includes('versions')) {
                    const parts = path.split('_');
                    if (parts.length > 2) {
                        const time = parts[parts.length-2] + '_' + parts[parts.length-1].replace('.pth','');
                        info = `版本时间: ${time}`;
                    }
                }
                modelInfo.textContent = info;
            }
        }

        // ===== 切换模型 =====
        modelSelect.addEventListener('change', function() {
            const selected = this.options[this.selectedIndex];
            const path = selected.value;
            const name = selected.dataset.name || selected.text;
            if (!path) return;

            modelStatus.textContent = '切换中...';
            modelStatus.className = 'badge bg-warning';

            fetch('/switch_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: path, name: name })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    modelStatus.textContent = `✅ ${data.model_name}`;
                    modelStatus.className = 'badge bg-success';
                    alert(`✅ 已切换到 ${data.model_name}`);
                    clearDetectArea();
                } else {
                    modelStatus.textContent = '❌ 切换失败';
                    modelStatus.className = 'badge bg-danger';
                    alert('❌ ' + data.message);
                    // 恢复之前的选择
                    loadModelVersions();
                }
            })
            .catch(err => {
                modelStatus.textContent = '❌ 网络错误';
                modelStatus.className = 'badge bg-danger';
                alert('❌ 网络错误');
                loadModelVersions();
            });
        });

        // ===== 清空检测区域 =====
        function clearDetectArea() {
            selectedFiles = [];
            previewContainer.innerHTML = '';
            fileCount.textContent = '已选 0 张';
            detectBtn.disabled = true;
            detectHint.textContent = '请先选择图片';
            detectHint.className = 'text-muted ms-2 small align-self-center';
            detectResults.innerHTML = '';
            detectSummary.innerHTML = '';
            detectProgress.style.display = 'none';
            fileInput.value = '';
            progressBar.style.width = '0%';
            progressBar.textContent = '0%';
            detectBtn.textContent = '🚀 开始检测';
            detectBtn.disabled = true;
        }

        // ===== 更新错误计数 =====
        function updateErrorCount() {
            fetch('/history?limit=1')
                .then(r => r.json())
                .then(d => { if (d.success) {
                    document.getElementById('error-count').textContent = d.error_count || 0;
                    document.getElementById('error-alert').style.display = 'flex';
                }})
                .catch(() => {});
        }

        // ===== 文件选择 =====
        function handleFiles(files) {
            for (const f of files) {
                if (f.type.startsWith('image/') && !selectedFiles.some(x => x.name === f.name && x.size === f.size)) {
                    selectedFiles.push(f);
                }
            }
            updateUI();
        }

        function removeFile(idx) {
            selectedFiles.splice(idx, 1);
            updateUI();
        }

        function updateUI() {
            previewContainer.innerHTML = '';
            selectedFiles.forEach((f, i) => {
                const div = document.createElement('div');
                div.className = 'preview-item';
                const img = document.createElement('img');
                img.className = 'preview-img';
                img.src = URL.createObjectURL(f);
                img.alt = f.name;
                const nameSpan = document.createElement('div');
                nameSpan.className = 'file-name';
                nameSpan.textContent = f.name.length > 12 ? f.name.slice(0,10)+'...' : f.name;
                const rm = document.createElement('button');
                rm.className = 'remove-btn';
                rm.textContent = '×';
                rm.onclick = (e) => { e.stopPropagation(); removeFile(i); };
                div.appendChild(img);
                div.appendChild(rm);
                div.appendChild(nameSpan);
                previewContainer.appendChild(div);
            });
            fileCount.textContent = `已选 ${selectedFiles.length} 张`;
            if (selectedFiles.length > 0) {
                detectBtn.disabled = false;
                detectHint.textContent = `准备检测 ${selectedFiles.length} 张图片`;
                detectHint.className = 'text-success ms-2 small align-self-center';
            } else {
                detectBtn.disabled = true;
                detectHint.textContent = '请先选择图片';
                detectHint.className = 'text-muted ms-2 small align-self-center';
            }
        }

        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) { handleFiles(fileInput.files); fileInput.value = ''; }
        });

        document.getElementById('clearBtn').addEventListener('click', function() {
            if (selectedFiles.length === 0 && detectResults.innerHTML === '' && detectSummary.innerHTML === '') return;
            clearDetectArea();
            updateErrorCount();
        });

        // ===== 结果卡片 =====
        function createResultCard(imageUrl, imageName, pred) {
            const div = document.createElement('div');
            div.className = 'result-card' + (pred.confidence < 0.6 ? ' low-confidence' : '');
            
            let probHtml = '';
            const maxP = Math.max(...Object.values(pred.all_probs));
            CLASS_LIST.forEach(c => {
                const p = pred.all_probs[c] || 0;
                const pct = (p*100).toFixed(1);
                probHtml += `<span class="badge ${p===maxP?'bg-success':'bg-secondary'} me-1" style="font-size:0.75rem;">${CLASS_CN[c]}: ${pct}%</span>`;
            });

            let optionsHtml = `<option value="">选择真实类别</option>`;
            CLASS_LIST.forEach(c => {
                optionsHtml += `<option value="${c}">${CLASS_CN[c]}</option>`;
            });

            div.innerHTML = `
                <div class="row align-items-center">
                    <div class="col-md-2"><img src="${imageUrl}" class="img-fluid rounded" style="max-height:80px;object-fit:cover;"></div>
                    <div class="col-md-4"><strong>${imageName}</strong></div>
                    <div class="col-md-3">
                        <span class="badge ${pred.confidence>0.8?'bg-success':pred.confidence>0.6?'bg-warning':'bg-danger'} fs-6">${CLASS_CN[pred.predicted_class]}</span>
                        <span class="badge bg-secondary">${(pred.confidence*100).toFixed(1)}%</span>
                    </div>
                    <div class="col-md-3">
                        <div class="mark-btn-group">
                            <select class="form-select form-select-sm true-class-select">
                                ${optionsHtml}
                            </select>
                            <button class="btn btn-danger btn-sm mark-true-btn" data-image-name="${imageName}" data-image-url="${imageUrl}" data-predicted-class="${pred.predicted_class}">🚨 标记为错误</button>
                        </div>
                    </div>
                </div>
                <div class="row mt-2"><div class="col-12"><small>${probHtml}</small></div></div>
            `;

            const markBtn = div.querySelector('.mark-true-btn');
            if (markBtn) {
                markBtn.addEventListener('click', function() {
                    const select = this.parentElement.querySelector('.true-class-select');
                    const trueClass = select.value;
                    if (!trueClass) {
                        alert('请先选择真实类别！');
                        return;
                    }
                    markTrueClass(this.dataset.imageName, this.dataset.predictedClass, trueClass, this.dataset.imageUrl, this);
                });
            }

            return div;
        }

        // ===== 标记真实类别 =====
        function markTrueClass(imageName, predictedClass, trueClass, imageUrl, btn) {
            if (!confirm(`确认将 "${imageName}" 的真实类别标记为 "${CLASS_CN[trueClass]}" 吗？`)) return;
            
            fetch('/mark_true_class', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_name: imageName,
                    predicted_class: predictedClass,
                    true_class: trueClass,
                    image_url: imageUrl
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    alert(`✅ 已标记真实类别: ${CLASS_CN[trueClass]}`);
                    btn.textContent = '✅ 已标记';
                    btn.className = 'btn btn-success btn-sm';
                    btn.disabled = true;
                    updateErrorCount();
                    if (document.getElementById('error-panel').classList.contains('active')) {
                        loadErrorSamples();
                    }
                } else {
                    alert('❌ 标记失败: ' + data.message);
                }
            })
            .catch(() => alert('网络错误'));
        }

        // ===== 检测 =====
        document.getElementById('detectBtn').addEventListener('click', async function() {
            if (selectedFiles.length === 0) return;
            this.disabled = true;
            this.textContent = '⏳ 检测中...';
            detectProgress.style.display = 'block';
            detectResults.innerHTML = '';
            detectSummary.innerHTML = '';
            progressBar.style.width = '0%';
            progressBar.textContent = '0%';
            progressText.textContent = '准备检测...';

            const total = selectedFiles.length;
            let completed = 0;
            const allResults = [];
            const summary = {'ZF-scratch':0,'scratch':0,'broken':0,'pinbreak':0};

            for (let i = 0; i < total; i++) {
                const file = selectedFiles[i];
                const fd = new FormData();
                fd.append('image', file);
                progressText.textContent = `正在检测 ${i+1}/${total}: ${file.name}`;
                try {
                    const resp = await fetch('/predict', { method: 'POST', body: fd });
                    const data = await resp.json();
                    if (data.success) {
                        const r = {
                            image_name: data.image_name,
                            image_url: data.image_url,
                            predicted_class: data.prediction.predicted_class,
                            confidence: data.prediction.confidence,
                            all_probs: data.prediction.all_probs
                        };
                        allResults.push(r);
                        summary[r.predicted_class] = (summary[r.predicted_class] || 0) + 1;
                    }
                } catch (e) { console.error('检测失败:', e); }
                completed = i + 1;
                const percent = Math.round((completed / total) * 100);
                progressBar.style.width = percent + '%';
                progressBar.textContent = percent + '%';
                progressText.textContent = `已检测 ${completed}/${total}`;
                await new Promise(resolve => setTimeout(resolve, 10));
            }

            progressBar.style.width = '100%';
            progressBar.textContent = '100%';
            progressText.textContent = '✅ 检测完成！';

            allResults.forEach((r) => {
                detectResults.appendChild(createResultCard(r.image_url, r.image_name, {
                    predicted_class: r.predicted_class,
                    confidence: r.confidence,
                    all_probs: r.all_probs
                }));
            });

            let sumHtml = `<div class="summary-card"><h6>📊 检测汇总</h6><div>
                <span class="summary-stat">📷 总计: <span class="count">${total}</span> 张</span>
                <span class="summary-stat">✅ 成功: <span class="count text-success">${allResults.length}</span> 张</span>
                <span class="summary-stat">❌ 失败: <span class="count text-danger">${total - allResults.length}</span> 张</span>
            </div><div class="mt-2">`;
            Object.entries(summary).filter(([k,v]) => v>0).forEach(([cls, cnt]) => {
                sumHtml += `<span class="summary-stat">${CLASS_CN[cls]}: <span class="count">${cnt}</span></span>`;
            });
            if (Object.values(summary).every(v => v===0)) sumHtml += '<span class="text-muted">暂无分类结果</span>';
            sumHtml += '</div></div>';
            detectSummary.innerHTML = sumHtml;

            this.disabled = false;
            this.textContent = '🚀 开始检测';
            detectProgress.style.display = 'none';
            
            if (selectedFiles.length > 0) {
                this.disabled = false;
                detectHint.textContent = `已检测完成，可继续添加或点击清空`;
                detectHint.className = 'text-info ms-2 small align-self-center';
            }
        });

        // ===== 历史记录 =====
        function loadHistory() {
            let url = '/history?limit=200';
            if (currentStartDate) url += `&start_date=${currentStartDate}`;
            if (currentEndDate) url += `&end_date=${currentEndDate}`;

            if (currentStartDate && currentEndDate) {
                dateRangeInfo.textContent = `📅 ${currentStartDate} ~ ${currentEndDate}`;
            } else if (currentStartDate) {
                dateRangeInfo.textContent = `📅 ${currentStartDate} 至今`;
            } else if (currentEndDate) {
                dateRangeInfo.textContent = `📅 至今 ~ ${currentEndDate}`;
            } else {
                dateRangeInfo.textContent = '📅 显示全部记录';
            }

            fetch(url)
                .then(r => r.json())
                .then(data => {
                    const list = document.getElementById('historyList');
                    if (data.success && data.predictions && data.predictions.length) {
                        let html = '';
                        data.predictions.forEach(p => {
                            const trueClassDisplay = p.true_class ? `→ 真实: ${CLASS_CN[p.true_class]}` : '';
                            html += `
                                <div class="list-group-item history-item" data-id="${p.id}">
                                    <div class="d-flex align-items-center">
                                        <img src="${p.image_path}" class="preview-img me-3" style="width:60px;height:60px;object-fit:cover;border-radius:4px;">
                                        <div class="flex-grow-1">
                                            <h6 class="mb-1">${p.image_name}</h6>
                                            <p class="mb-1">预测: ${CLASS_CN[p.predicted_class]} (置信度: ${(p.confidence*100).toFixed(2)}%) ${trueClassDisplay}</p>
                                            <small class="text-muted">${p.created_at}</small>
                                            ${p.is_correct === 0 ? '<span class="badge bg-danger ms-2">错误样本</span>' : ''}
                                        </div>
                                    </div>
                                </div>
                            `;
                        });
                        list.innerHTML = html;
                        document.querySelectorAll('.history-item').forEach(el => {
                            el.addEventListener('click', function() {
                                showDetail(this.dataset.id);
                            });
                        });
                    } else {
                        list.innerHTML = '<p class="text-muted">📭 该时间段暂无记录</p>';
                    }
                })
                .catch(() => document.getElementById('historyList').innerHTML = '<div class="alert alert-danger">加载失败</div>');
        }

        function showDetail(id) {
            fetch('/history?limit=200')
                .then(r => r.json())
                .then(data => {
                    const p = data.predictions.find(x => x.id == id);
                    if (!p) return;
                    let probBars = '';
                    CLASS_LIST.forEach(cls => {
                        const prob = p.all_probs[cls] || 0;
                        const pct = (prob*100).toFixed(1);
                        probBars += `
                            <div class="prob-bar">
                                <label class="form-label mb-1 small">${CLASS_CN[cls]}</label>
                                <div class="progress" style="height:20px;">
                                    <div class="progress-bar" role="progressbar" style="width: ${pct}%">${pct}%</div>
                                </div>
                            </div>
                        `;
                    });
                    const trueDisplay = p.true_class ? `<p>真实类别: <strong class="text-success">${CLASS_CN[p.true_class]}</strong></p>` : '';
                    document.getElementById('detailModalBody').innerHTML = `
                        <div class="row">
                            <div class="col-md-5"><img src="${p.image_path}" class="img-fluid rounded"></div>
                            <div class="col-md-7">
                                <h5>${p.image_name}</h5>
                                <p>预测类别: <strong>${CLASS_CN[p.predicted_class]}</strong></p>
                                <p>置信度: <strong>${(p.confidence*100).toFixed(2)}%</strong></p>
                                ${trueDisplay}
                                ${p.is_correct===0?'<span class="badge bg-danger">错误样本</span>':''}
                                <hr><h6>各类别概率</h6>${probBars}
                                <p class="text-muted mt-3">预测时间: ${p.created_at}</p>
                            </div>
                        </div>
                    `;
                    new bootstrap.Modal(document.getElementById('historyDetailModal')).show();
                });
        }

        // ===== 日期筛选 =====
        startDateInput.addEventListener('change', function() {
            currentStartDate = this.value;
            loadHistory();
        });
        endDateInput.addEventListener('change', function() {
            currentEndDate = this.value;
            loadHistory();
        });
        document.getElementById('resetDateBtn').addEventListener('click', function() {
            startDateInput.value = '';
            endDateInput.value = '';
            currentStartDate = '';
            currentEndDate = '';
            loadHistory();
        });

        // ===== 清空历史 =====
        document.getElementById('clearHistoryBtn').addEventListener('click', function() {
            let url = '/history/clear';
            let msg = '确定要清空全部历史记录吗？此操作不可恢复！';
            if (currentStartDate && currentEndDate) {
                url += `?start_date=${currentStartDate}&end_date=${currentEndDate}`;
                msg = `确定要清空 ${currentStartDate} ~ ${currentEndDate} 之间的记录吗？此操作不可恢复！`;
            } else if (currentStartDate) {
                url += `?start_date=${currentStartDate}`;
                msg = `确定要清空 ${currentStartDate} 至今的记录吗？此操作不可恢复！`;
            } else if (currentEndDate) {
                url += `?end_date=${currentEndDate}`;
                msg = `确定要清空至今 ~ ${currentEndDate} 的记录吗？此操作不可恢复！`;
            }
            if (!confirm(msg)) return;
            fetch(url, { method: 'DELETE' })
                .then(r => r.json())
                .then(d => {
                    if (d.success) {
                        alert(`✅ 已清空 ${d.deleted_count} 条记录`);
                        loadHistory();
                        updateErrorCount();
                    } else alert('❌ 清空失败: '+d.message);
                })
                .catch(() => alert('网络错误'));
        });

        // ===== 清空全部错误样本 =====
        document.getElementById('clearErrorsBtn').addEventListener('click', function() {
            if (!confirm('⚠️ 确定要清空所有错误样本吗？此操作不可恢复！')) return;
            fetch('/clear_errors', { method: 'DELETE' })
                .then(r => r.json())
                .then(d => {
                    if (d.success) {
                        alert(`✅ 已清空 ${d.deleted_count} 个错误样本`);
                        updateErrorCount();
                        loadErrorSamples();
                        if (document.getElementById('history-panel').classList.contains('active')) {
                            loadHistory();
                        }
                    } else {
                        alert('❌ 清空失败: ' + d.message);
                    }
                })
                .catch(() => alert('网络错误'));
        });

        // ===== 错误样本 =====
        function loadErrorSamples() {
            fetch('/error_samples')
                .then(r => r.json())
                .then(data => {
                    const listDiv = document.getElementById('errorSampleList');
                    const totalSpan = document.getElementById('errorTotal');
                    const statsDiv = document.getElementById('errorStats');
                    
                    if (data.success && data.samples && data.samples.length > 0) {
                        totalSpan.textContent = `共 ${data.total} 个`;
                        
                        const stats = {};
                        data.samples.forEach(s => {
                            stats[s.predicted_class] = (stats[s.predicted_class] || 0) + 1;
                        });
                        let statsHtml = '';
                        for (const [cls, count] of Object.entries(stats)) {
                            statsHtml += `<span class="error-stat-item">${CLASS_CN[cls]}: <span class="count">${count}</span></span>`;
                        }
                        statsDiv.innerHTML = statsHtml;
                        
                        let html = '';
                        data.samples.forEach(s => {
                            const probs = s.all_probs || {};
                            let probHtml = '';
                            CLASS_LIST.forEach(c => {
                                const p = probs[c] || 0;
                                probHtml += `<span class="badge bg-secondary me-1" style="font-size:0.7rem;">${CLASS_CN[c]}: ${(p*100).toFixed(1)}%</span>`;
                            });
                            const isMarked = s.true_class && s.true_class !== '';
                            const markedClass = isMarked ? s.true_class : '';
                            const reportCount = s.report_count || 1;
                            
                            let optionsHtml = `<option value="">选择真实类别</option>`;
                            CLASS_LIST.forEach(c => {
                                const selected = (c === markedClass) ? 'selected' : '';
                                optionsHtml += `<option value="${c}" ${selected}>${CLASS_CN[c]}</option>`;
                            });
                            
                            html += `
                                <div class="result-card error-sample-card ${isMarked ? 'marked-true' : ''}">
                                    <div class="row align-items-center">
                                        <div class="col-md-2">
                                            <img src="${s.image_path}" class="img-fluid rounded" style="max-height:80px;object-fit:cover;">
                                        </div>
                                        <div class="col-md-2">
                                            <strong>${s.image_name}</strong>
                                        </div>
                                        <div class="col-md-2">
                                            <span class="badge bg-danger">${CLASS_CN[s.predicted_class]}</span>
                                            <span class="badge bg-secondary">${(s.confidence*100).toFixed(1)}%</span>
                                            ${isMarked ? `<span class="badge bg-success">已标记</span>` : ''}
                                            <span class="badge bg-info report-count-badge" onclick="showErrorHistory('${s.image_name}')" title="点击查看报告历史">📋 ${reportCount}次</span>
                                        </div>
                                        <div class="col-md-3">
                                            <select class="form-select form-select-sm true-class-select" data-id="${s.id}">
                                                ${optionsHtml}
                                            </select>
                                        </div>
                                        <div class="col-md-3 text-end">
                                            <button class="btn btn-danger btn-sm mark-true-btn" data-id="${s.id}" data-image-name="${s.image_name}" data-predicted-class="${s.predicted_class}" data-image-url="${s.image_path}">🚨 标记为错误</button>
                                        </div>
                                    </div>
                                    <div class="row mt-1">
                                        <div class="col-12">
                                            <small class="text-muted">报告时间: ${s.reported_at || s.created_at}</small>
                                            ${isMarked ? `<span class="text-success ms-2">真实类别: ${CLASS_CN[markedClass]}</span>` : ''}
                                            ${reportCount > 1 ? `<span class="text-info ms-2">被报告 ${reportCount} 次</span>` : ''}
                                        </div>
                                    </div>
                                    <div class="row mt-1"><div class="col-12"><small>${probHtml}</small></div></div>
                                </div>
                            `;
                        });
                        listDiv.innerHTML = html;
                        
                        document.querySelectorAll('.mark-true-btn').forEach(btn => {
                            btn.addEventListener('click', function() {
                                const id = this.dataset.id;
                                const select = document.querySelector(`.true-class-select[data-id="${id}"]`);
                                const trueClass = select.value;
                                if (!trueClass) {
                                    alert('请先选择真实类别！');
                                    return;
                                }
                                markTrueClassFromError(id, trueClass, this);
                            });
                        });
                    } else {
                        totalSpan.textContent = '共 0 个';
                        statsDiv.innerHTML = '';
                        listDiv.innerHTML = '<p class="text-muted">✅ 暂无错误样本，继续保持！</p>';
                    }
                })
                .catch(() => {
                    document.getElementById('errorSampleList').innerHTML = '<div class="alert alert-danger">加载失败</div>';
                });
        }

        function markTrueClassFromError(id, trueClass, btn) {
            const imageName = btn.dataset.imageName;
            const predictedClass = btn.dataset.predictedClass;
            const imageUrl = btn.dataset.imageUrl;
            
            if (!confirm(`确认将 "${imageName}" 的真实类别标记为 "${CLASS_CN[trueClass]}" 吗？`)) return;
            
            fetch('/mark_true_class', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_name: imageName,
                    predicted_class: predictedClass,
                    true_class: trueClass,
                    image_url: imageUrl
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    alert(`✅ 已标记真实类别: ${CLASS_CN[trueClass]}`);
                    loadErrorSamples();
                    updateErrorCount();
                } else {
                    alert('❌ 标记失败: ' + data.message);
                }
            })
            .catch(() => alert('网络错误'));
        }

        function showErrorHistory(imageName) {
            fetch(`/error_history/${encodeURIComponent(imageName)}`)
                .then(r => r.json())
                .then(data => {
                    if (data.success && data.history.length > 0) {
                        let html = `<h6>📋 ${imageName} 的错误报告历史</h6><div class="history-timeline">`;
                        data.history.forEach((item, idx) => {
                            const trueDisplay = item.true_class ? ` → 真实: ${CLASS_CN[item.true_class]}` : ' (未标记)';
                            html += `
                                <div class="item">
                                    <div class="time">#${idx+1} ${item.reported_at || item.created_at}</div>
                                    <div>
                                        预测: <span class="badge bg-danger">${CLASS_CN[item.predicted_class]}</span>
                                        置信度: ${(item.confidence*100).toFixed(1)}%
                                        ${trueDisplay}
                                    </div>
                                </div>
                            `;
                        });
                        html += '</div>';
                        document.getElementById('errorHistoryBody').innerHTML = html;
                        new bootstrap.Modal(document.getElementById('errorHistoryModal')).show();
                    }
                })
                .catch(() => alert('加载历史失败'));
        }

        // ===== 监听标签页 =====
        document.getElementById('history-tab').addEventListener('shown.bs.tab', loadHistory);
        document.getElementById('error-tab').addEventListener('shown.bs.tab', loadErrorSamples);

        // ===== 初始化 =====
        updateErrorCount();
        loadModelVersions();
    </script>
</body>
</html>
"""

# ==================== 启动 ====================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)