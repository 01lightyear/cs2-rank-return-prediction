1.获取数据：运行backfill_hourly_kline.py，从getdata/itemid.txt里按顺序抓取。之后需要每日补充，用get_hourly_kline.py。请不要过频繁调用。使用check_item_timestamp_continuity.py做连续性检查和线性插值处理

用TBD/backtest_xgb.py作回测

运行TBD/infer_xgb.py可做历史数据上的对比推理，有预测排序和真实排序，以及真实收益率的对比，需要先用TBD/preprocess_xgb.py生成TBD/factor_dataset.parquet，要求infer的日期在预处理的日期范围里面

用的因子即为TBD/features.md中200个

运行get_hourly_kline.py得到数据，然后运行TBD/infer_xgb_live.py推理，调整date参数即可。2025-11-23指的是2025-11-23  15:00 到 2025-11-24  15:00，UTC+8，区间左开右闭

alpha042会因为rank=0而导致每轮推理都有一个asset不能计算，但是加epsilon或者修改rank 归一化方式后疑似会对效果产生影响，于是便放任不管

选了个尽可能好看的，虽然回测时间短也没什么实际意义，而且考虑使用的因子不同以及训练数据集的日期不同等等。

![回测(尽可能选的最过拟合的)](TBD/backtest_results6.png)

