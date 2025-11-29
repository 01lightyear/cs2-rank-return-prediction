## Daily Factor Feature Set

我们基于每日（北京时间 15:00 截止的 24 小时聚合 K 线）提取价格与成交量特征，并与 `t+7` 的相对收益率做横截面 Rank IC 评估。每个交易日需要 168 个 item 的完整观测。

# 基于日级 OHLCV（含成交额）的 33 个基础量化因子库

## 一、收益 / 趋势类（8 个）

1. **log_return_1**  
   \( r_t = \log C_t - \log C_{t-1} \)

2. **intraday_return**  
   \( \log(C_t / O_t) \)


4. **momentum_5**  
   \( \log(C_t / C_{t-5}) \)

5. **momentum_20**  
   \( \log(C_t / C_{t-20}) \)

6. **momentum_60**  
   \( \log(C_t / C_{t-60}) \)

7. **risk_adj_mom_20**  
   \( \frac{\log(C_t/C_{t-20})}{\text{std}_{20}(r)} \)

8. **donchian_pos_20**  
   \( \frac{C_t - \min(C_{t-19:t})}{\max(C_{t-19:t}) - \min(C_{t-19:t})} \)

---

## 二、波动率 / 区间类（6 个）

9. **true_range_pct_1**  
   \( \text{TR}_t = \max(H_t-C_{t-1}, C_{t-1}-L_t, H_t-L_t) \)  
   \( \text{TR}_t / C_{t-1} \)

10. **realized_vol_10**  
    \( \sqrt{\sum_{i=0}^{9} r_{t-i}^2} \)

11. **realized_vol_20**  
    \( \sqrt{\sum_{i=0}^{19} r_{t-i}^2} \)

12. **parkinson_vol_10**  
    \( \sqrt{\frac{1}{4\ln 2} \cdot \frac{1}{10}
    \sum_{i=0}^{9}\left[\ln(H_{t-i}/L_{t-i})\right]^2 } \)

13. **vol_ratio_5_20**  
    \( \text{realized\_vol}_5 / \text{realized\_vol}_{20} \)

14. **vol_change_20**  
    \( \frac{\text{realized\_vol}_{20} - \text{realized\_vol}_{20,\text{lag1}}}
            {\text{realized\_vol}_{20,\text{lag1}}} \)

---

## 三、成交量 / 流动性类（7 个）

15. **log_volume_zscore_20**  
    \( z^V = \frac{\log V_t - \mu_{20}(\log V)}{\sigma_{20}(\log V)} \)

16. **log_amount_zscore_20**  
    \( z^A = \frac{\log A_t - \mu_{20}(\log A)}{\sigma_{20}(\log A)} \)

17. **volume_ma_ratio_5_20**  
    \( \text{ma}_5(V) / \text{ma}_{20}(V) \)

18. **amount_ma_ratio_5_20**  
    \( \text{ma}_5(A) / \text{ma}_{20}(A) \)

19. **amihud_illiquidity_20**  
    \( \frac{1}{20} \sum_{i=0}^{19} \frac{|r_{t-i}|}{A_{t-i}} \)

20. **obv_slope_20**  
    \( \text{obv\_slope}_{20} = (\text{obv}_t - \text{obv}_{t-20}) / 20 \)

21. **vwap_close_gap**  
    \( \text{VWAP}_t = A_t / V_t \)  
    \( (C_t - \text{VWAP}_t) / \text{VWAP}_t \)

---

## 四、超买超卖 / 均线结构类（7 个）

22. **rsi_14**

23. **stoch_k_14**  
    \( \frac{C_t - \min(C_{t-13:t})}{\max(C_{t-13:t}) - \min(C_{t-13:t})} \)

24. **bollinger_z_20**  
    \( \frac{C_t - \text{ma}_{20}(C)}{\text{std}_{20}(C)} \)

25. **ema_gap_12_26**  
    \( (\text{ema}_{12}(C) - \text{ema}_{26}(C)) / \text{ema}_{26}(C) \)

26. **price_ma_gap_20**  
    \( (C_t - \text{ma}_{20}(C)) / \text{ma}_{20}(C) \)

27. **adx_14**

28. **trend_slope_ma20**  
    \( \frac{\text{ma}_{20}(C) - \text{ma}_{20,\text{lag5}}(C)}
            {5\cdot \text{ma}_{20,\text{lag5}}(C)} \)

---

## 五、K 线形态 / 微结构类（5 个）

29. **body_ratio**  
    \( |C_t - O_t| / (H_t - L_t + \varepsilon) \)

30. **upper_shadow_ratio**  
    \( (H_t - \max(C_t, O_t)) / (H_t - L_t + \varepsilon) \)

31. **lower_shadow_ratio**  
    \( (\min(C_t, O_t) - L_t) / (H_t - L_t + \varepsilon) \)

32. **clv_1**  
    \( \text{CLV} = \frac{(C_t-L_t)-(H_t-C_t)}{H_t - L_t + \varepsilon} \)

33. **bull_ratio_5**  
    最近 5 日阳线比例：  
    \( \#\{C_t > O_t\} / 5 \)

---

## 六、来自alpha101

34. **alpha#2**
    \( (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))\)
35. **alpha#3**
    \( (-1 * correlation(rank(open), rank(volume), 10))\)
36. **alpha#4**
    \( (-1 * Ts_Rank(rank(low), 9))\)
37. **alpha#5**
    \( (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))\)
38. **alpha#6**
    \( (-1 * correlation(open, volume, 10)) \)
39. **alpha#8**
    \( (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))) \)
40. **alpha#11**
    \( ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))\)
41. **alpha#12**
    \( (sign(delta(volume, 1)) * (-1 * delta(close, 1))) \)
42. **alpha#13**
    \((-1 * rank(covariance(rank(close), rank(volume), 5))) \)
43. **alpha#14**
    \(((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)) \)
44. **alpha#15**
    \((-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)) \)
45. **alpha#16**
    \((-1 * rank(covariance(rank(high), rank(volume), 5))) \)
46. **alpha#17**
    \((((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * 
rank(ts_rank((volume / adv20), 5))) \)
47. **alpha#18**
    \( (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 
10)))) \)
48. **alpha#20**
    \((((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - 
delay(low, 1)))) \)
49. **alpha#22**
    \( (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))\)
50. **alpha#23**
    \((((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0) \)
51. **alpha#25**
    \( rank(((((-1 * returns) * adv20) * vwap) * (high - close))) \)
52. **alpha#26**
    \((-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)) \)
53. **alpha#30**
    \( (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20)) \)
54. **alpha#33**
    \(rank((-1 * ((1 - (open / close))^1))) \)
55. **alpha#34**
    \(rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1))))) \)
56. **alpha#38**
    \(((-1 * rank(Ts_Rank(close, 10))) * rank((close / open))) \)
57. **alpha#40**
    \( ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)) \)
58. **alpha#41**
    \((((high * low)^0.5) - vwap) \)
59. **alpha#42**
    \((rank((vwap - close)) / rank((vwap + close))) \)
60. **alpha#43**
    \(\(ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8)) )
61. **alpha#44**
    \( (-1 * correlation(high, rank(volume), 5))\)
62. **alpha#45**
    \( (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2)))) \)
63. **alpha#46**
    \( ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1))))) \)
64. **alpha#47**
    \( ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5)))) \)
65. **alpha#49**
    \((((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1)))) \)
66. **alpha#50**
    \( (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)) \)
67. **alpha#51**
    \( (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1)))) \)
68. **alpha#53**
    \((-1 * delta((((close - low) - (high - close)) / (close - low)), 9)) \)
69. **alpha#54**
    \(((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5))) \)
70. **alpha#55**
    \((-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6)) \)
71. **alpha#57**
    \( (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))\)
72. **alpha#60**
    \( (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10)))))) \)
73. **alpha#83**
    \(((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close))) \)
74. **alpha#101**
    \( ((close - open) / ((high - low) + .001)) \)

---
## 七、风险&高阶矩
57.  **return_skew_20**

58.  **return_kurt_20**


---


## 八、补充
59. **downside_vol_20**

60. **mfi_14**

61. **cmf_20**

62. **corr_price_vol_20**
    计算过去 20 日的“日收益率”与“日成交量对数值变化”之间的皮尔逊相关系数


##  Definitions

    abs(x), log(x), sign(x) = standard definitions; same for the operators “+”, “-”, “*”, “/”, “>”, “<”, 
    “==”, “||”, “x ? y : z” 
    rank(x) = cross-sectional rank 
    delay(x, d) = value of x d days ago  
    correlation(x, y, d) = time-serial correlation of x and y for the past d days 
    covariance(x, y, d) = time-serial covariance of x and y for the past d days 
    scale(x, a) = rescaled x such that sum(abs(x)) = a (the default is a = 1) 
    delta(x, d) = today’s value of x minus the value of x d days ago 
    signedpower(x, a) = x^a 
    decay_linear(x, d) = weighted moving average over the past d days with linearly decaying 
    weights d, d – 1, …, 1 (rescaled to sum up to 1) 
    indneutralize(x, g) = x cross-sectionally neutralized against groups g (subindustries, industries, 
    sectors, etc.), i.e., x is cross-sectionally demeaned within each group g 
    ts_{O}(x, d) = operator O applied across the time-series for the past d days; non-integer number 
    of days d is converted to floor(d)  
    ts_min(x, d) = time-series min over the past d days 
    ts_max(x, d) = time-series max over the past d days 
    ts_argmax(x, d) = which day ts_max(x, d) occurred on 
    ts_argmin(x, d) = which day ts_min(x, d) occurred on 
    ts_rank(x, d) = time-series rank in the past d days 
    min(x, d) = ts_min(x, d) 
    max(x, d) = ts_max(x, d) 
    sum(x, d) = time-series sum over the past d days 
    product(x, d) = time-series product over the past d days 
    stddev(x, d) = moving time-series standard deviation over the past d days 

    returns = daily close-to-close returns 
    open, close, high, low, volume = standard definitions for daily price and volume data 
    vwap = daily volume-weighted average price 
    cap = market cap  
    adv{d} = average daily dollar volume for the past d days

## Rank IC 滑动窗口与滞后设定

- **目标收益率**：`target_7d = close.shift(-7) / close - 1`。
- **横截面 Rank**：对每个交易日，将 168 个 item 的因子值与 `target_7d` 分别排名，使用 Spearman 相关系数作为 Rank IC。
- **滞后序列**：若将因子与同日收益视为 `lag0`，还需额外输出滞后版本：
  - `lag2`：因子值使用 `factor.shift(2)` 与 `target_7d` 做相关。
  - `lag4`：`factor.shift(4)`。
  - `lag8`：`factor.shift(8)`。
  - `lag15`：`factor.shift(15)`。
- 每个因子最终得到按滞后拆分的 Rank IC 序列，同时统计均值、方差、样本量与 `IC_IR = mean / std`，以便比较预测价值。

> 注：在做滞后时仅偏移因子，不移动目标收益率；并确保对齐后剔除任何因子或目标缺失的 item。必要时在因子入库前完成去极值和标准化处理，以增强稳定性。

## 中性化流程规划

1. **行业映射解析**  
   - 从 `getdata/itemid.txt` 读取行业结构：以 `//行业名` 开头的行定义一个行业，后续连续的 item ID 即为成员，空行表示行业分割。  
   - 构造 `item_id -> industry` 的映射，并保留一个默认行业（如 `UNKNOWN`）以处理未覆盖的品种。

2. **特征预处理**  
   - 对原始因子可暂不去极值，但建议先记录异常值分布。  
   - 做一次截面去均值：对于每个交易日、每个因子 `x := x - mean_date(x)`，确保常数项暴露为零。

3. **构建设计矩阵**  
   - 连续协变量：  
     - `log_price_ma = log1p(price_ma_n)`：使用聚合后的移动平均价（如 20 日均价或 VWAP），先做 `log1p`。  
     - `log_volume = log1p(volume)` 或 `log1p(volume_ma_n)`：表示当前或滚动成交量的对数。  
     - `realized_vol_10`
   - 行业哑变量：为每个行业生成虚拟变量（去掉一个基准行业），与连续协变量共同构成矩阵 `X`。

4. **逐日回归中性化**  
   - 对每个交易日、每个因子执行截面 OLS：`factor = α + β1 * log_price_ma + β2 * log_volume + Σ γ_k * industry_dummy_k + ε`。  
   - 取残差 `ε` 作为中性化后的因子值。若当日样本不足（样本数 ≤ 协变量数），则退化为仅行业中性化或仅去均值。

5. **再标准化与存储**  
   - 对残差再做一次截面标准化（zscore）提升数值稳定性。  
   - 缓存中性化后的因子矩阵，用于后续 Rank IC 计算。保留校验信息（回归 R²、有效样本数）。

6. **验证与监控**  
   - 对比中性化前后因子统计量（均值、标准差、极值）及 Rank IC 表现，确认行业/规模暴露是否被有效剥离。  
   - 记录异常情况（行业缺失、协变量取值异常）并输出日志，便于日后调试或调整协变量定义。
