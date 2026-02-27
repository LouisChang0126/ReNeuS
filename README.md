# ReNeuS: Neural Surface Reconstruction with Refraction-Aware Rendering

基於 [NeuS](https://github.com/Totoro97/NeuS) 修改的 ReNeuS 實現，用於透明容器內物體的神經表面重建。

## 概述

ReNeuS 是一種折射感知的神經隱式表面重建方法，專門設計用於處理透過透明介質（如玻璃容器、水箱）觀察物體的場景。與原始 NeuS 假設光線直線傳播不同，ReNeuS 考慮了光線在容器表面的折射和反射，從而能夠準確重建容器內部的物體幾何。

### 核心特性

- **物理準確的折射計算**：使用 Snell's Law 和完整 Fresnel 方程
- **全內反射 (TIR) 處理**：正確檢測和處理臨界角情況
- **GPU 加速 ray-mesh intersection**：純 GPU Möller–Trumbore 演算法（無 CPU-GPU 搬移開銷），trimesh CPU 作為 fallback
- **SIREN SDF 網路**：隱藏層使用 `sin(ω₀·Wx+b)` 激活，最後一層使用普通 Linear + geometric init
- **Deterministic 光線分支**：每次 bounce 產生反射+折射兩組射線，以 Fresnel 係數加權
- **可選純折射模式**：`enable_reflection = false` 關閉反射分支，適用於低 IOR 或調試
- **Inner Render Debug View**：Validation 時輸出假設移除玻璃後的內部物體渲染
- **靈活配置**：從 `transforms_train.json` 讀取場景參數（IOR、mesh 路徑）
- **向後兼容**：沒有容器 mesh 時自動退回到原始 NeuS

## 安裝

```bash
# 基礎依賴（與 NeuS 相同）
pip install torch torchvision
pip install opencv-python pyhocon icecream tqdm numpy trimesh

# 可選（CPU fallback ray tracer 加速）
pip install pyembree
```

## 數據格式

ReNeuS 使用 NeRF/Blender 格式（`transforms_train.json`），並在 JSON 中加入 ReNeuS 特有參數：

```
Dataset/3DGRUT/[case_name]/
├── transforms_train.json    # 訓練集（含 ReNeuS metadata）
├── transforms_val.json      # 驗證集
├── transforms_test.json     # 測試集
├── glass_box.ply            # 容器 mesh（必需）
├── object.ply               # Ground truth mesh（可選，評估用）
├── train/                   # 訓練圖像（RGBA PNG）
│   ├── 0001.png
│   └── ...
└── val/                     # 驗證圖像
```

### transforms_train.json 格式

```json
{
  "camera_angle_x": 0.6911112070083618,
  "fl_x": 1111.111,
  "fl_y": 1111.111,
  "cx": 400.0,
  "cy": 400.0,
  "w": 800.0,
  "h": 800.0,
  "IOR": 1.5,
  "mesh_inside": "object.ply",
  "mesh_outside": "glass_box.ply",
  "frames": [
    {
      "file_path": "train/0001",
      "transform_matrix": [[...]]
    }
  ]
}
```

ReNeuS 特有欄位：
- `IOR`：容器折射率（玻璃 ≈ 1.5，水 ≈ 1.33，空氣 ≈ 1.0）
- `mesh_outside`：容器（玻璃）mesh 路徑，用於 ray-mesh intersection
- `mesh_inside`：內部物體 GT mesh（可選，僅供評估）

> 圖像應為 **RGBA PNG**（含 alpha channel 作為 mask），支援 uint8 和 uint16。

## 使用方法

推薦使用 Jupyter Notebook 方式訓練（`playground_reneus.ipynb`），或直接執行：

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

## 配置

### reneus.conf 主要參數

```hocon
train {
    learning_rate = 5e-4
    end_iter = 200000           # 論文: 200k iterations
    batch_size = 1024           # 論文: 1024 rays per batch
    trans_weight = 0.1          # λ₁: Transmittance loss 權重 (Eq.13)
    igr_weight = 0.1            # λ₂: Eikonal loss 權重
    use_white_bkgd = True       # 固定背景色 C_out = [0.8, 0.8, 0.8]
    show_mesh_wireframe = true  # Validation 中間格是否疊加容器 mesh 線框
}

model {
    reneus {
        max_bounces = 2         # 最大光線彈跳次數 Dre（論文用 2）
        enable_reflection = true  # false: 只走折射，跳過反射分支
        # ior = 1.5             # 可選：覆蓋 transforms_train.json 中的 IOR
    }

    sdf_network {
        d_out = 257             # 1(SDF) + 256(feature)
        d_hidden = 256
        n_layers = 8            # 隱藏層使用 SIREN (sin 激活)，最後一層 Linear
        multires = 6            # Positional encoding 頻率數
        geometric_init = True   # 球形幾何初始化（最後一層）
    }

    neus_renderer {
        n_samples = 64          # Coarse samples
        n_importance = 64       # Fine samples (importance sampling)
        n_outside = 0           # 無 NeRF++ 背景（ReNeuS 使用固定背景色）
    }
}
```

### reneus_no_reflect.conf

與 `reneus.conf` 相同，但 `enable_reflection = false`，適用於低 IOR 數據集或調試。

## 實現細節

### Validation 輸出（validations_fine/）

每張 validation 圖由三格橫向拼接組成：

| 格 | 內容 |
|---|---|
| **最上格** （inner render） | 假設移除玻璃，直接從相機射線渲染 SDF 物體（只在容器 entry→exit 範圍採樣，margin 10%） |
| **中間格** （ReNeuS render） | 完整 ReNeuS 渲染（含折射/反射/Fresnel），可選疊加容器 mesh 線框 |
| **最下格** （GT） | Ground truth 訓練圖像 |

### 光線追蹤流程

```
Camera Ray → [GPU ray-mesh intersection]
                    ↓ (hit)               ↓ (miss)
             [容器 entry/exit]         → 背景色 [0.8, 0.8, 0.8]
                    ↓
        [Volume Render: 0 → exit_dist]
                    ↓
        [Fresnel split (Eq. 8-9)]
        ├── 反射分支 (weight × R)  ← enable_reflection=true 時
        └── 折射分支 (weight × T)
                    ↓
        重複 max_bounces 次
```

### SDF 網路架構 (SIREN)

- **隱藏層** (lin0 ~ lin{N-2})：`SineLayer`，激活函數 `sin(ω₀ · Wx + b)`，`ω₀=30`
  - 第一層：均勻初始化 `U(-1/d_in, 1/d_in)`
  - 其他層：均勻初始化 `U(-√(6/d)/ω₀, √(6/d)/ω₀)`
- **最後一層** (lin{N-1})：普通 `nn.Linear` + geometric initialization（球形 bias）
- **無 Softplus/ReLU**：SIREN 的 sin 激活本身連續且有解析梯度

### 損失函數 (Eq. 13)

```
L = L_color + λ₁ · L_trans + λ₂ · L_eikonal

L_color    = (1/|M_in|) Σ ||C(p) - Ĉ(p)||₁   (mask 內所有像素)
L_trans    = (1/|M_in|) Σ ||1 - T_ℓ||          (Transmittance sparsity prior)
L_eikonal  = (1/N) Σ (||∇f(x)|| - 1)²          (SDF 梯度約束)
```

## 文件結構

```
ReNeuS/
├── models/
│   ├── renderer.py       # 核心渲染器（折射/反射/Fresnel/inner render）
│   ├── dataset.py        # Dataset（NeRF格式，含 IOR/mesh 讀取）
│   ├── fields.py         # SIREN SDFNetwork + RenderingNetwork
│   ├── ray_mesh.py       # GPU Möller–Trumbore ray-mesh intersector
│   └── embedder.py       # Positional encoding
├── confs/
│   ├── reneus.conf           # 主配置（含反射）
│   └── reneus_no_reflect.conf # 純折射配置
├── exp_runner.py         # 訓練/評估主程序（含 validate_image）
├── test_reneus.py        # 核心功能測試
└── README.md             # 本文件
```

## 參考

- **NeuS**: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction  
  [Paper](https://arxiv.org/abs/2106.10689) | [Code](https://github.com/Totoro97/NeuS)

- **ReNeuS**: Seeing Through the Glass: Neural 3D Reconstruction of Object Inside a
Transparent Container (CVPR 2023)  
  [Paper](https://arxiv.org/abs/2303.13805)

- **SIREN**: Implicit Neural Representations with Periodic Activation Functions  
  [Paper](https://arxiv.org/abs/2006.09661)

## 致謝

本實現基於 [NeuS](https://github.com/Totoro97/NeuS) 代碼庫，感謝原作者的優秀工作。
