# ReNeuS 快速開始指南

## 5 分鐘快速測試

### 1. 測試核心功能

```bash
cd /home/louis/Fish-Dev/ReNeuS
python test_reneus.py
```

預期輸出：
```
✓ Refraction test passed!
✓ Reflection test passed!
✓ Fresnel test passed!
✓ TIR test passed!
✓ Dataset loading test passed!
```

### 2. 快速訓練測試（100 iterations）

編輯 `confs/reneus.conf`，設置短訓練週期：

```hocon
train {
    end_iter = 100  # 快速測試
    report_freq = 10
    val_freq = 50
}
```

運行訓練：

```bash
python exp_runner.py \
    --conf ./confs/reneus.conf \
    --mode train \
    --case lego_glass \
    --gpu 0
```

### 3. 檢查輸出

訓練日誌應顯示：
```
[ReNeuS] Loading container mesh from: ...
[ReNeuS] Using Embree ray tracer (accelerated)
[ReNeuS] Container mesh loaded: XXXX faces, IOR=1.5
```

驗證圖像：`exp/lego_glass/reneus/validations_fine/`

## 完整訓練

恢復配置文件的正常設置：

```hocon
train {
    end_iter = 300000
}
```

運行完整訓練：

```bash
python exp_runner.py \
    --conf ./confs/reneus.conf \
    --mode train \
    --case lego_glass \
    --gpu 0
```

## 提取 Mesh

訓練完成後：

```bash
python exp_runner.py \
    --conf ./confs/reneus.conf \
    --mode validate_mesh \
    --case lego_glass \
    --is_continue \
    --mcube_threshold 0.0
```

Mesh 文件：`exp/lego_glass/reneus/meshes/`

## 故障排除

### pyembree 安裝失敗

如果 `pip install pyembree` 失敗，系統會自動使用 trimesh 的默認 ray tracer（較慢但可用）。

### CUDA 記憶體不足

減少 batch size：

```hocon
train {
    batch_size = 256  # 默認 512
}
```

### 容器 mesh 未找到

確認 `Dataset/ReNeuS/lego_glass/metadata.json` 中的路徑正確：

```json
{
  "mesh_glass": "meshes/glass_box.ply"
}
```

## 下一步

- 查看 `walkthrough.md` 了解實現細節
- 查看 `implementation_plan.md` 了解技術方案
- 準備自己的數據集（參考 `README.md` 的數據格式部分）
