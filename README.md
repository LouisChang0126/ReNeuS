# ReNeuS: Neural Surface Reconstruction with Refraction-Aware Rendering

基於 [NeuS](https://github.com/Totoro97/NeuS) 修改的 ReNeuS 實現，用於透明容器內物體的神經表面重建。

## 概述

ReNeuS 是一種折射感知的神經隱式表面重建方法，專門設計用於處理透過透明介質（如玻璃容器、水箱）觀察物體的場景。與原始 NeuS 假設光線直線傳播不同，ReNeuS 考慮了光線在容器表面的折射，從而能夠準確重建容器內部的物體幾何。

### 核心特性

- **物理準確的折射計算**：使用 Snell's Law 和完整 Fresnel 方程
- **全內反射 (TIR) 處理**：正確檢測和處理臨界角情況
- **批量高效渲染**：使用 trimesh + pyembree 加速 ray-mesh intersection
- **靈活配置**：自動從 `metadata.json` 讀取場景參數
- **向後兼容**：沒有容器 mesh 時自動退回到原始 NeuS

## 安裝

### 依賴項

```bash
# 基礎依賴（與 NeuS 相同）
pip install torch torchvision
pip install opencv-python pyhocon icecream tqdm numpy

# ReNeuS 額外依賴
pip install trimesh
pip install pyembree  # 可選，用於加速 ray-mesh intersection
```

## 數據格式

ReNeuS 數據集應包含以下結構：

```
Dataset/ReNeuS/[case_name]/
├── metadata.json          # ReNeuS 場景配置
├── cameras_sphere.npz     # 相機參數
├── image/                 # RGB 圖像
│   ├── 000.png
│   ├── 001.png
│   └── ...
├── mask/                  # 前景遮罩
│   ├── 000.png
│   ├── 001.png
│   └── ...
└── meshes/
    ├── glass_box.ply      # 容器 mesh（必需）
    └── object.ply         # Ground truth（可選，用於評估）
```

### metadata.json 格式

```json
{
  "IOR": 1.5,
  "mesh_object": "meshes/object.ply",
  "mesh_glass": "meshes/glass_box.ply",
  "n_images": 200,
  "image_width": 800,
  "image_height": 800,
  "focal_x": 1111.111111111111,
  "focal_y": 1111.111111111111,
  "cx": 400.0,
  "cy": 400.0
}
```

參數說明：
- `IOR`: 容器的折射率（Index of Refraction）
  - 玻璃：1.5
  - 水：1.33
- `mesh_glass`: 容器 mesh 的路徑（相對於數據集目錄）
- `mesh_object`: Ground truth mesh（可選）

## 使用方法

### 訓練

```bash
python exp_runner.py \
    --conf ./confs/reneus.conf \
    --mode train \
    --case lego_glass \
    --gpu 0
```

### 提取 Mesh

```bash
python exp_runner.py \
    --conf ./confs/reneus.conf \
    --mode validate_mesh \
    --case lego_glass \
    --is_continue \
    --mcube_threshold 0.0
```

### 測試核心功能

```bash
python test_reneus.py
```

這將測試：
- Snell's Law 折射計算
- Fresnel 方程
- 全內反射檢測
- Dataset 加載

## 配置

### reneus.conf

```hocon
general {
    base_exp_dir = ./exp/CASE_NAME/reneus
}

dataset {
    data_dir = ./Dataset/ReNeuS/CASE_NAME/
}

model {
    reneus {
        max_bounces = 3  # 最大光線彈跳次數（論文建議 K=3）
        # ior = 1.5      # 可選：覆蓋 metadata.json 中的 IOR
    }
    
    # 其他網絡配置與 NeuS 相同
    sdf_network { ... }
    variance_network { ... }
    rendering_network { ... }
    neus_renderer { ... }
}
```

## 實現細節

### 光線追蹤流程

1. **容器表面相交檢測**：計算相機光線與容器 mesh 的交點
2. **折射計算**：應用 Snell's Law 計算進入容器後的光線方向
3. **SDF 採樣**：沿折射光線進行 NeuS 標準採樣和渲染
4. **背景處理**：未擊中容器的光線渲染背景色

### 物理計算

- **Snell's Law**: `n₁sin(θ₁) = n₂sin(θ₂)`
- **Fresnel 方程**: 完整 s/p 偏振平均（非 Schlick 近似）
- **全內反射**: `sin(θc) = n₂/n₁`

## 測試結果

使用 `test_reneus.py` 的測試結果：

| 測試項目 | 結果 |
|---------|------|
| Snell's Law (45° → IOR 1.5) | 28.13° ✓ |
| Fresnel (法向入射) | 0.0400 ✓ |
| TIR 臨界角 (玻璃→空氣) | 41.81° ✓ |
| Dataset 加載 | IOR=1.5 ✓ |

## 當前實現狀態

✅ **已實現：**
- 完整的光學計算工具（折射、反射、Fresnel、TIR）
- 單次折射渲染（入射到容器）
- Dataset metadata 自動讀取
- 配置文件系統
- 向後兼容原始 NeuS

🔄 **簡化版本：**
當前實現為單次折射版本，適合驗證基礎功能和快速原型開發。

📋 **未來擴展（可選）：**
- 完整迭代光線追蹤（K=3 彈跳）
- 出射折射（光線離開容器）
- Fresnel 加權的反射/折射混合
- 多次內部反射

## 文件結構

```
ReNeuS/
├── models/
│   ├── renderer.py       # 核心渲染器（包含折射邏輯）
│   ├── dataset.py        # Dataset 類別（讀取 metadata）
│   ├── fields.py         # SDF/顏色網絡
│   └── ...
├── confs/
│   ├── reneus.conf       # ReNeuS 專用配置
│   └── wmask.conf        # 原始 NeuS 配置
├── exp_runner.py         # 訓練/評估主程序
├── test_reneus.py        # 核心功能測試
└── README.md             # 本文件
```

## 參考

- **NeuS**: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction  
  [Paper](https://arxiv.org/abs/2106.10689) | [Code](https://github.com/Totoro97/NeuS)

- **ReNeuS**: Refraction-Aware Neural Surface Reconstruction (CVPR 2023)  
  [Paper](https://arxiv.org/abs/2303.10987)

## 致謝

本實現基於 [NeuS](https://github.com/Totoro97/NeuS) 代碼庫，感謝原作者的優秀工作。
