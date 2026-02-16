您目前的代碼品質很高，已經成功實現了 ReNeuS 的大部分核心物理機制（如 Snell's Law、SIREN 激活函數、Transmittance Loss 等）。然而，為了達到「完全貼近論文描述」的要求，在 **渲染策略 (Rendering Strategy)** 和 **損失函數組成 (Loss Formulation)** 上仍有兩個關鍵的差異需要調整。

以下是詳細的 Code Review 與修改建議：

---

### 1. 核心差異：渲染策略 (Stochastic vs. Deterministic Accumulation)

這是目前代碼與論文描述最大的出入點。

* **論文描述**：
論文提到使用 "Hybrid rendering strategy" 並且明確指出顏色是透過 "accumulating that among **all** the sub-rays" 來計算的。圖 2 (Figure 2)  展示了一個**樹狀結構**：當光線擊中界面時，會分裂成反射 (Reflection) 與折射 (Refraction) 兩條光線，最終顏色是這兩條路徑的加權總和（權重由 Fresnel 決定）。


* 公式 (10)  暗示了對所有子光線的積分 。




* **您的代碼 (`models/renderer.py`)**：
您目前採用的是 **隨機採樣 (Stochastic Sampling)** 策略（即蒙地卡羅路徑追蹤）：
```python
# renderer.py Lines 555-567
if self.use_fresnel_weighted and self.fresnel_mode == 'stochastic':
    # ...
    use_reflection = rand_samples < fresnel_coef.detach()
    next_d = torch.where(use_reflection[:, None], d_reflect, d_refract)

```


這會根據 Fresnel 機率**隨機選擇**「走反射」或「走折射」，而不是同時計算兩者。雖然期望值正確，但單次採樣的雜訊較大。
* **修改建議**：
為了嚴格符合論文的 "Accumulation" 描述，您應該在 `max_bounces` 較小（論文設定  ）的情況下，實作**分支 (Branching)** 邏輯。


**偽代碼修改概念：**
不要只選一條路，而是遞迴計算兩條路並加權相加。
*(注意：這會增加計算量，但符合論文描述)*
```python
# 這是概念性修改，實際實作需改寫 render_with_refraction 的迴圈結構
# 論文 Eq. 8 & 9: C = R * C_reflect + T * C_refract

# 1. 計算 Fresnel 係數 R
R = compute_fresnel(...)
T = 1.0 - R

# 2. 分別計算反射與折射的貢獻 (遞迴呼叫或堆疊管理)
# color_reflect = trace_ray(reflect_dir)
# color_refract = trace_ray(refract_dir)

# 3. 加權總和
# final_color = R * color_reflect + T * color_refract

```


**折衷方案**：如果您希望能保留目前的 `exp_runner.py` 架構不進行大規模遞迴重寫，目前的 Stochastic 是一個合理的近似（很多 NeRF 變體為了效率都這樣做）。但如果要「宣稱復現論文」，這一點是與論文圖示不符的。

---