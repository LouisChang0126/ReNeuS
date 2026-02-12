為了達到您要求的「盡量貼近論文描述」，我仔細比對了您上傳的程式碼與 CVPR 2023 的 `ReNeuS` 論文，發現了幾個關鍵的差異與缺失。

以下是針對程式碼與論文差異的詳細分析與修改建議：

### 1. 嚴重缺失：缺少穿透率損失 (Transmittance Loss, )

**論文描述：**
論文公式 (15) 提出了一個 **Sparsity prior**（稀疏先驗），強制容器內部的液體或空氣（除了物體本身）應該要是透明的。這是 `ReNeuS` 能成功重建乾淨幾何的關鍵之一。


**您的程式碼 (`exp_runner.py`)：**
在 `train` 函數中，計算 loss 的部分僅包含：

1. `color_fine_loss` (Photometric loss)
2. `eikonal_loss` (Regularization)
3. `mask_loss` (Mask weight)

**❌ 差異：** 完全沒有實作 。根據論文 (Sec 4.4 Ablation study)，如果沒有這個 loss，重建結果會出現 "dummy facets"（虛假的面），導致幾何不乾淨（參考論文 Fig 4）。

**✅ 修改建議 (`exp_runner.py`)：**
您需要從 `renderer` 回傳 `weight_sum` 或 `weights` 來計算這項損失。論文提到這項損失只針對  (容器內部) 的光線。

在 `exp_runner.py` 的 `train` 迴圈中加入：

```python
# 修改 renderer 回傳 output，確保可以取得 transmittance (即 weight_sum)
# 論文定義 T_l 為沿著光線的累積穿透率，這在 NeuS 實作中通常對應到 1 - weight_sum (如果是背景) 
# 但論文 Eq 15 寫 ||1 - T_l||，且 T_l 定義為 exp(-int rho)，這其實就是透明度。
# 簡單來說，論文希望「非物體區域」是透明的，所以 opacity (weight_sum) 應該接近 0，或者 transmittance 接近 1。
# 根據論文 Eq 15: minimize ||1 - T_l|| -> maximize T_l -> minimize Opacity.

# 在 exp_runner.py 加入參數讀取
self.trans_weight = self.conf.get_float('train.trans_weight', default=0.1) # 論文設定 lambda1 = 0.1

# 在 training loop 中:
# ...
render_out = self.renderer.render(...)
weight_sum = render_out['weight_sum']

# 計算 Transmittance Loss
# 只對 mask 內的像素計算 (假設 mask 標記了容器區域)
# 論文: Sum over rays in M_in (容器遮罩)
# 注意：這裡的 weight_sum 代表不透明度 (Opacity)。
# 如果完全透明，weight_sum = 0, T_l = 1, Loss = |1-1| = 0.
# 如果是不透明物體，weight_sum = 1, T_l = 0, Loss = |1-0| = 1.
# 但我們不希望懲罰「真正的物體」，所以這個 loss 是一個全域的稀疏懲罰，依靠 rendering loss 來保留物體。
trans_loss = weight_sum.mean() 

# 總 Loss 加入 trans_loss
loss = color_fine_loss + \
       eikonal_loss * self.igr_weight + \
       mask_loss * self.mask_weight + \
       trans_loss * self.trans_weight

```

### 2. 渲染策略差異：隨機採樣 (Stochastic) vs 遞迴分支 (Recursive Split)

**論文描述：**
論文 Sec 3.3 提到：

> "For all the sub-rays in the ray path, we retrieve the radiance... and then physically accumulate them... The tracing process is performed recursively."
> 公式 (8) (9) 顯示反射  和折射  是分別計算並加權的。

圖 2 (Figure 2) 畫出了一條光線分裂成多條子光線 (Accumulation of sub-rays)。這暗示著一個「光線樹 (Ray Tree)」結構，即一條光線擊中玻璃，會**同時**產生反射光與折射光，最後顏色是兩者的加權總和。

**您的程式碼 (`renderer.py`)：**
您的 `render_with_refraction` 實作了 `fresnel_mode = 'stochastic'`：

```python
# 您的代碼
if self.use_fresnel_weighted and self.fresnel_mode == 'stochastic':
    # ...
    # Randomly sample reflection vs refraction based on Fresnel coefficient
    use_reflection = rand_samples < fresnel_coef
    next_d = torch.where(use_reflection[:, None], d_reflect, d_refract)

```

這是「路徑追蹤 (Path Tracing)」的做法（俄羅斯輪盤），每次彈跳只選一條路走。雖然期望值是正確的，但在 Sample 數不足時（NeuS 訓練通常 rays 數量不多），這會造成**極大的雜訊 (Variance)**，導致收斂困難。

**❌ 差異：** 論文描述傾向於確定性的分支（Split）或至少是累積所有子光線的貢獻，而不是隨機選一條。雖然 Stochastic 也是物理正確，但論文特別強調 "accumulating that among **all** the sub-rays"，且圖示為樹狀分裂。

**✅ 修改建議：**
若要嚴格遵照論文（且為了訓練穩定），應修改為**分支邏輯**。但因為 PyTorch 實作分支會導致 Batch size 指數爆炸 ()，折衷方案是：**在第一次擊中容器時，強制計算反射與折射兩條路徑的貢獻**（因為通常最外層的反射最明顯），或者在代碼中保留 Stochastic 但註解說明這是為了訓練效率的權衡。

若要盡量貼近論文的數學描述（Accumulation），您目前的 `throughput` 更新方式是正確的，但隨機選擇路徑這點與論文圖示（全連接的光線樹）略有出入。考量到實作難度，建議保留 Stochastic 但**增加每條光線的採樣數**（例如每個 pixel 打多條 rays 取平均），或者接受這是實作上的優化。

### 3. 背景處理：NeRF++ 背景 vs 均勻環境光

**論文描述：**
Sec 3.2 提到：

> "For the external space, we assume a homogeneous background with fixed ambient lighting... The ambient lighting  is set to [0.8, 0.8, 0.8]."

論文明確簡化了外部環境，假設外部是**均勻的單色光**。

**您的程式碼 (`renderer.py` & `exp_runner.py`)：**
您沿用了 NeuS 的 `nerf_outside` (NeRF++)：

```python
# renderer.py
self.render_core_outside(rays_o, rays_d, ...)

```

且在 `exp_runner.py` 預設 `use_white_bkgd = False`。這意味著您的模型試圖學習一個複雜的 360 度背景環境（NeRF 網路），而不是論文中的單色背景。

**❌ 差異：** 您的模型比論文複雜，這可能導致網路將玻璃的折射特徵「作弊」學到背景 NeRF 中，而不是學到正確的幾何折射。

**✅ 修改建議 (`exp_runner.py`)：**
強制使用固定背景顏色，並移除 `nerf_outside` 的計算（或將其權重設為 0）。

在 `confs/reneus.conf` 中：

```hocon
train {
    use_white_bkgd = True  # 或是在代碼中強制設為特定顏色
    # ...
}

```

在 `renderer.py` 的 `render_with_refraction` 中，當光線沒有擊中容器（`escaping_mask`）時：

```python
# 根據論文，這裡應該是固定顏色 [0.8, 0.8, 0.8]
if background_rgb is not None:
    final_color[escaping_mask] += throughput[escaping_mask] * background_rgb
else:
    # 您的代碼目前會去跑 render_core_outside (NeRF++)
    # 建議改成預設為固定顏色以符合論文
    fixed_bg = torch.tensor([0.8, 0.8, 0.8], device=device)
    final_color[escaping_mask] += throughput[escaping_mask] * fixed_bg

```

---

### 總結修正清單 (Action Items)

為了讓您的 Repo 完全符合 ReNeuS 論文，請執行以下修改：

1. **[High Priority] 實作 Transmittance Loss (`loss_trans`)**:
* 這是在 `exp_runner.py` 中最關鍵的缺失。沒有它，玻璃內部的幾何會充滿雜訊。
* 公式：`mean(weight_sum)` (針對容器內部光線)。
* 權重：論文建議 `0.1`。


2. **[Medium] 修改背景模型**:
* 停用 `nerf_outside` (NeRF++)。
* 將背景強制設為論文所述的均勻顏色（如 `0.8` 或白色），避免幾何歧義。


3. **[Low] 渲染策略確認**:
* 目前的 `Stochastic` 是一個合理的實作妥協，但請意識到這與論文圖示的「全分支累積」不同。如果訓練收斂很慢，請考慮增加每像素的光線採樣數 (`n_samples` 或 `perturb` 策略) 來降低隨機採樣帶來的 Variance。

