#!/usr/bin/env python3
"""
基于XGBoost推理的量化回测系统
支持自定义时间范围，多种策略对比，完整的资金管理
"""

import os
import sys
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties

import xgboost as xgb
from matplotlib import font_manager

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 中文字体配置
FONT_CANDIDATES = [
    "SimHei",
    "Microsoft YaHei",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Noto Sans CJK",
    "Source Han Sans",
    "Noto Serif CJK SC",
    "Noto Serif CJK",
    "Source Han Serif SC",
]

FONT_SEARCH_DIRS = [
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".local/share/fonts",
]

FONT_FILE_PATTERNS = [
    "**/NotoSansCJK*.ttc",
    "**/NotoSansCJK*.otf",
    "**/SourceHanSans*.otf",
    "**/SourceHanSans*.ttc",
    "**/SourceHanSerif*.otf",
    "**/SourceHanSerif*.ttc",
    "**/NotoSerifCJK*.ttc",
    "**/NotoSerifCJK*.otf",
    "**/WenQuanYi*.ttf",
    "**/SimHei*.ttf",
]

def _register_candidate_fonts() -> None:
    """注册候选中文字体"""
    registered = 0
    for base in FONT_SEARCH_DIRS:
        if not base.exists():
            continue
        for pattern in FONT_FILE_PATTERNS:
            for path in base.glob(pattern):
                try:
                    font_manager.fontManager.addfont(str(path))
                    registered += 1
                except OSError:
                    continue

    if registered:
        try:
            font_manager._load_fontmanager(try_read_cache=False)
        except Exception:
            pass

def _configure_chinese_font() -> None:
    """配置中文字体"""
    available_fonts = font_manager.fontManager.ttflist
    chosen_name: Optional[str] = None
    for candidate in FONT_CANDIDATES:
        candidate_lower = candidate.lower()
        for font in available_fonts:
            if candidate_lower in font.name.lower():
                chosen_name = font.name
                break
        if chosen_name is not None:
            break

    if chosen_name is None:
        print("⚠️ 未找到本地中文字体，图表中文可能无法正常显示。")
        return

    existing = list(plt.rcParams.get("font.sans-serif", []))
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = [chosen_name] + existing
    print(f"✅ 已检测到中文字体：{chosen_name}")

# 配置中文字体
_register_candidate_fonts()
_configure_chinese_font()
plt.rcParams["axes.unicode_minus"] = False

# 忽略警告
warnings.filterwarnings('ignore')

# 导入现有的推理函数
from TBD.infer_xgb import (
    load_item_name_map,
    load_cross_section_from_dataset,
    build_dmatrix,
    load_model,
    predict_scores,
)

@dataclass
class PortfolioPosition:
    """持仓信息"""
    invest_date: pd.Timestamp      # 投资日期
    amount: float                  # 投资金额
    assets: List[str]              # 资产ID列表
    weights: List[float]           # 权重列表
    expected_return: float         # 预期收益率
    actual_return: Optional[float] = None  # 实际收益率（到期后计算）

@dataclass
class BacktestConfig:
    """回测配置参数"""
    start_date: str          # 回测开始日期 (YYYY-MM-DD)
    end_date: str            # 回测结束日期 (YYYY-MM-DD)
    initial_capital: float   # 初始资金
    commission_rate: float   # 手续费率 (0.01 = 1%)
    holding_days: int        # 持有天数
    top_n_main: int          # 主策略选择top N
    top_n_benchmarks: List[int]  # 基准策略的top N列表
    build_up_days: int       # 建仓天数

    def __post_init__(self):
        """参数验证"""
        if self.commission_rate < 0 or self.commission_rate > 1:
            raise ValueError("手续费率必须在0-1之间")
        if self.holding_days <= 0:
            raise ValueError("持有天数必须大于0")
        if self.initial_capital <= 0:
            raise ValueError("初始资金必须大于0")

class BacktestEngine:
    """基于XGBoost推理的回测引擎"""

    def __init__(self, config: BacktestConfig):
        """
        初始化回测引擎

        Args:
            config: 回测配置参数
        """
        self.config = config
        self.data_dir = PROJECT_ROOT / "data_new"
        self.mapping_file = PROJECT_ROOT / "getdata" / "itemid.txt"
        self.features_file = PROJECT_ROOT / "TBD" / "features.md"
        self.model_file = PROJECT_ROOT / "TBD" / "xgb_rank_model1.json"
        self.dataset_parquet = PROJECT_ROOT / "TBD" / "factor_dataset.parquet"

        # 加载必要的配置和数据
        self._load_model()
        self._load_mappings()

        # 初始化回测状态
        self.daily_portfolio = {}  # 每日持仓记录 {date: PortfolioPosition}
        self.portfolio_value = {}  # 每日总资产价值
        self.daily_returns = {}    # 每日收益率记录

        # 基准策略记录
        self.benchmark_results = {n: {'portfolio_value': {}, 'returns': {}}
                                for n in config.top_n_benchmarks}

        # 缓存推理结果避免重复计算
        self.inference_cache = {}  # {date: DataFrame}

    def _load_mappings(self):
        """加载资产ID到名称的映射"""
        self.item_mapping = load_item_name_map(self.mapping_file)
        print(f"加载了 {len(self.item_mapping)} 个资产映射")

    def _load_model(self):
        """加载训练好的XGBoost模型"""
        if self.model_file.exists():
            self.model = load_model(self.model_file)
            print("XGBoost模型加载成功")
        else:
            raise FileNotFoundError(f"模型文件不存在: {self.model_file}")

    def build_full_panel_from_parquet(self) -> pd.DataFrame:
        """
        基于parquet数据构建回测所需的面板数据

        Returns:
            完整的面板数据（从parquet加载）
        """
        print("从parquet数据构建回测面板...")

        # 检查parquet文件是否存在
        if not self.dataset_parquet.exists():
            raise FileNotFoundError(f"Parquet数据文件不存在: {self.dataset_parquet}")

        # 加载parquet数据
        print(f"加载parquet数据: {self.dataset_parquet}")
        full_df = pd.read_parquet(self.dataset_parquet)

        if full_df.empty:
            raise ValueError("Parquet数据为空")

        # 确保日期列格式正确
        full_df["date"] = pd.to_datetime(full_df["date"])
        print(f"Parquet数据维度: {full_df.shape}")
        print(f"数据时间范围: {full_df['date'].min()} 到 {full_df['date'].max()}")

        # 检查数据覆盖是否足够
        start_dt = pd.to_datetime(self.config.start_date)
        end_dt = pd.to_datetime(self.config.end_date)
        needed_start = start_dt - pd.Timedelta(days=90)  # 90天历史数据
        needed_end = end_dt 

        if full_df['date'].min() > needed_start or full_df['date'].max() < needed_end:
            raise ValueError(
                f"Parquet数据覆盖不足。\n"
                f"需要范围: {needed_start.strftime('%Y-%m-%d')} 到 {needed_end.strftime('%Y-%m-%d')}\n"
                f"现有范围: {full_df['date'].min().strftime('%Y-%m-%d')} 到 {full_df['date'].max().strftime('%Y-%m-%d')}"
            )

        print("数据覆盖检查通过")
        return full_df

    def batch_inference_from_parquet(self, trading_dates: pd.DatetimeIndex) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        基于parquet数据的批量推理

        Args:
            trading_dates: 交易日期列表

        Returns:
            按日期索引的推理结果字典
        """
        print("开始基于parquet的批量推理...")

        # 收集所有交易日的横截面数据
        all_cross_sections = []
        valid_dates = []

        for date in trading_dates:
            try:
                # 使用现有的函数从parquet加载横截面数据
                cross_section, feature_cols = load_cross_section_from_dataset(
                    self.dataset_parquet, date
                )

                if not cross_section.empty and len(cross_section) >= 20:
                    # 确保有必要的列
                    cross_section = cross_section.copy()
                    cross_section["date"] = date
                    all_cross_sections.append(cross_section)
                    valid_dates.append(date)  # 只存储日期

            except Exception as e:
                print(f"警告: 日期 {date} 数据加载失败: {e}")
                continue

        if not all_cross_sections:
            raise ValueError("没有有效的交易日数据")

        # 合并所有横截面数据
        print(f"合并 {len(all_cross_sections)} 个交易日的数据...")
        batch_data = pd.concat(all_cross_sections, ignore_index=True)

        print(f"批量预测数据维度: {batch_data.shape}")

        # 获取特征列（从parquet数据中推断）
        excluded_cols = {"date", "item_id", "target_rank_pct", "target_8d", "target_rank_label", "pred_score", "pred_rank_pct", "item_name"}
        feature_cols = [col for col in batch_data.columns if col not in excluded_cols]

        print(f"使用 {len(feature_cols)} 个特征进行预测")

        # 批量预测
        print("运行XGBoost批量预测...")
        dmatrix = build_dmatrix(batch_data, feature_cols)
        predictions = predict_scores(self.model, dmatrix)

        # 添加预测结果
        batch_data["pred_score"] = predictions
        batch_data["pred_rank_pct"] = batch_data["pred_score"].rank(pct=True)
        batch_data["item_name"] = batch_data["item_id"].map(self.item_mapping).fillna(batch_data["item_id"])

        # 按日期切分结果
        print("切分预测结果...")
        results = {}
        for date in valid_dates:
            date_mask = batch_data["date"] == date
            if date_mask.any():
                date_results = batch_data[date_mask].copy()
                # 按预测分数排序
                date_results = date_results.sort_values("pred_score", ascending=False).reset_index(drop=True)
                results[date] = date_results

        print(f"批量推理完成，成功处理 {len(results)} 个交易日")
        return results

    def _run_inference_for_date(self, date: str) -> pd.DataFrame:
        """为指定日期运行推理（兼容性接口，实际使用批量结果）"""
        target_date = pd.to_datetime(date)

        if target_date in self.batch_inference_results:
            return self.batch_inference_results[target_date]
        else:
            raise ValueError(f"日期 {date} 没有批量推理结果")

    def select_baseline_top_assets(self, date: pd.Timestamp, top_n: int) -> List[str]:
        """
        基于历史真实收益率选择baseline top N资产

        Args:
            date: 交易日期
            top_n: 选择top N个资产

        Returns:
            选择的资产ID列表
        """
        try:
            # 从parquet数据获取该日所有资产的数据
            date_data = self.full_parquet_df.loc[self.full_parquet_df["date"] == date]

            if date_data.empty:
                print(f"警告: 日期 {date} 没有数据")
                return []

            # 筛选有target_8d数据的资产
            valid_assets = date_data.dropna(subset=['target_8d'])

            if len(valid_assets) < top_n:
                print(f"警告: 有效资产数量 {len(valid_assets)} 少于需求 {top_n}")
                return []

            # 按target_8d降序排列，选择真正的top N资产
            top_assets_data = valid_assets.nlargest(top_n, 'target_8d')
            top_asset_ids = top_assets_data['item_id'].tolist()

            return top_asset_ids

        except Exception as e:
            print(f"Baseline资产选择失败: {e}")
            return []

    def _get_historical_return(self, invest_date: pd.Timestamp, assets: List[str]) -> float:
        """
        获取历史的真实8日收益率（直接从parquet的target_8d获取）

        Args:
            invest_date: 投资日期
            assets: 资产ID列表

        Returns:
            平均7日收益率（扣除手续费）
        """
        try:
            # 从parquet数据中获取投资日的数据
            invest_data = self.full_parquet_df.loc[self.full_parquet_df["date"] == invest_date]

            if invest_data.empty:
                print(f"警告: 投资日 {invest_date} 没有数据")
                return -self.config.commission_rate

            # 筛选选中的资产
            selected_returns = []
            for asset_id in assets:
                asset_data = invest_data[invest_data['item_id'] == asset_id]
                if not asset_data.empty and pd.notna(asset_data['target_8d'].iloc[0]):
                    selected_returns.append(asset_data['target_8d'].iloc[0])

            if selected_returns:
                avg_return = np.mean(selected_returns) - self.config.commission_rate
                return avg_return
            else:
                return -self.config.commission_rate  # 只有手续费损失

        except Exception as e:
            print(f"无法获取历史收益率: {e}")
            return -self.config.commission_rate

    def _calculate_portfolio_return(self, invest_date: pd.Timestamp, top_n: int) -> float:
        """
        计算指定top N的投资组合收益率（主策略：基于XGBoost预测）

        Args:
            invest_date: 投资日期
            top_n: 选择top N个资产

        Returns:
            投资组合平均收益率
        """
        try:
            invest_date_str = invest_date.strftime('%Y-%m-%d')

            # 获取投资日的推理结果
            predictions_df = self._run_inference_for_date(invest_date_str)

            if len(predictions_df) < top_n:
                return 0.0

            # 获取XGBoost预测的top N资产
            top_assets = predictions_df.head(top_n)

            # 计算历史收益率
            asset_ids = top_assets['item_id'].tolist()
            return self._get_historical_return(invest_date, asset_ids)

        except Exception as e:
            print(f"计算组合收益率失败: {e}")
            return 0.0

    def _calculate_baseline_portfolio_return(self, invest_date: pd.Timestamp, top_n: int) -> float:
        """
        计算baseline top N的投资组合收益率（基于真实收益率排名）

        Args:
            invest_date: 投资日期
            top_n: 选择baseline top N个资产

        Returns:
            投资组合平均收益率
        """
        try:
            # 使用baseline策略选择资产（基于真实收益率排名）
            baseline_asset_ids = self.select_baseline_top_assets(invest_date, top_n)

            if len(baseline_asset_ids) < top_n:
                return 0.0

            # 计算这些baseline资产的历史收益率
            return self._get_historical_return(invest_date, baseline_asset_ids)

        except Exception as e:
            print(f"计算baseline组合收益率失败: {e}")
            return 0.0

    def run_backtest(self):
        """运行完整回测（基于parquet数据的批量优化版）"""
        print("="*60)
        print("开始XGBoost量化回测（Parquet批量优化版）")
        print("="*60)
        print(f"回测期间: {self.config.start_date} 到 {self.config.end_date}")
        print(f"初始资金: {self.config.initial_capital}")
        print(f"主策略: Top {self.config.top_n_main}")
        print(f"基准策略: {self.config.top_n_benchmarks}")
        print(f"手续费: {self.config.commission_rate*100:.1f}%")
        print(f"持有期: {self.config.holding_days} 天")
        print(f"建仓期: {self.config.build_up_days} 天")
        print("="*60)

        # 步骤1：加载parquet数据
        try:
            self.full_parquet_df = self.build_full_panel_from_parquet()
        except Exception as e:
            print(f"Parquet数据加载失败: {e}")
            raise

        # 生成交易日期列表
        start_dt = pd.to_datetime(self.config.start_date)
        end_dt = pd.to_datetime(self.config.end_date)
        trading_dates = pd.date_range(start=start_dt, end=end_dt, freq='D')

        print(f"交易日数量: {len(trading_dates)}")

        # 步骤2：批量推理
        try:
            print("\n" + "="*40)
            print("开始基于parquet的批量推理阶段")
            print("="*40)
            self.batch_inference_results = self.batch_inference_from_parquet(trading_dates)
            print(f"批量推理完成，获得 {len(self.batch_inference_results)} 个交易日的预测结果")
        except Exception as e:
            print(f"批量推理失败: {e}")
            raise

        # 步骤3：回测主循环（使用预计算结果）
        print("\n" + "="*40)
        print("开始回测循环阶段")
        print("="*40)

        # 初始化变量
        current_capital = self.config.initial_capital
        investment_positions = {}  # {mature_date: [PortfolioPosition, ...]}

        for i, current_date in enumerate(trading_dates):
            date_str = current_date.strftime('%Y-%m-%d')
            if i % 10 == 0:  # 每10天打印一次进度
                print(f"\n--- 交易日 {i+1}/{len(trading_dates)}: {date_str} ---")

            try:
                # 1. 处理到期的投资
                matured_amount = 0.0
                if current_date in investment_positions:
                    matured_positions = investment_positions[current_date]
                    for position in matured_positions:
                        # 计算实际收益（使用预计算数据）
                        actual_return = self._calculate_portfolio_return(
                            position.invest_date,
                            len(position.assets)
                        )
                        position.actual_return = actual_return
                        matured_amount += position.amount * (1 + actual_return)

                        if i % 10 == 0:  # 只在进度时打印详细信息
                            print(f"  到期投资: {position.invest_date.strftime('%Y-%m-%d')} "
                                  f"投入 {position.amount:.4f} -> 收回 {position.amount * (1 + actual_return):.4f} "
                                  f"(收益率: {actual_return:.4f})")

                    del investment_positions[current_date]

                current_capital += matured_amount

                # 2. 确定当日投资金额
                if i < self.config.build_up_days:
                    # 建仓期：每天投入初始资金的1/build_up_days
                    invest_amount = self.config.initial_capital / self.config.build_up_days
                else:
                    # 正常期：投入所有可用资金
                    invest_amount = current_capital

                # 3. 获取推理结果（使用预计算结果）
                if current_date in self.batch_inference_results:
                    predictions_df = self.batch_inference_results[current_date]
                else:
                    if i % 10 == 0:
                        print(f"  警告: {date_str} 没有推理结果，跳过")
                    # 记录当前资产价值
                    total_invested = sum(pos.amount for positions in investment_positions.values()
                                       for pos in positions)
                    total_value = current_capital + total_invested
                    self.portfolio_value[current_date] = total_value
                    continue

                if invest_amount > 0 and len(predictions_df) >= self.config.top_n_main:
                    # 4. 执行投资
                    current_capital -= invest_amount

                    # 选择top资产并创建持仓
                    top_assets = predictions_df.head(self.config.top_n_main)
                    asset_ids = top_assets['item_id'].tolist()
                    weights = [1.0 / self.config.top_n_main] * self.config.top_n_main  # 等权重
                    expected_return = 0.0  # 实际回测中不使用预期收益

                    position = PortfolioPosition(
                        invest_date=current_date,
                        amount=invest_amount,
                        assets=asset_ids,
                        weights=weights,
                        expected_return=expected_return
                    )

                    # 计算到期日
                    mature_date = current_date + pd.Timedelta(days=self.config.holding_days)
                    if mature_date not in investment_positions:
                        investment_positions[mature_date] = []
                    investment_positions[mature_date].append(position)

                    if i % 10 == 0:
                        print(f"  投资 {invest_amount:.4f} 到 Top {self.config.top_n_main} 资产")

                # 5. 计算当日总资产价值
                total_invested = sum(pos.amount for positions in investment_positions.values()
                                   for pos in positions)
                total_value = current_capital + total_invested
                self.portfolio_value[current_date] = total_value

                if i % 10 == 0:
                    print(f"  当前现金: {current_capital:.4f}, 投资总额: {total_invested:.4f}, "
                          f"总资产: {total_value:.4f}")

            except Exception as e:
                print(f"交易日 {date_str} 处理失败: {e}")

                # 记录当前资产价值
                total_invested = sum(pos.amount for positions in investment_positions.values()
                                   for pos in positions)
                total_value = current_capital + total_invested
                self.portfolio_value[current_date] = total_value
                continue

        # 处理最后一个交易日之后未到期的投资
        print(f"\n处理未到期投资...")
        matured_amount = 0.0
        for mature_date, positions in investment_positions.items():
            for position in positions:
                # 回溯计算收益率，避免最后一批只按成本计入
                actual_return = self._calculate_portfolio_return(
                    position.invest_date,
                    len(position.assets)
                )
                position.actual_return = actual_return
                matured_amount += position.amount * (1 + actual_return)
                print(
                    f"  未到期投资: {position.invest_date.strftime('%Y-%m-%d')} "
                    f"投入 {position.amount:.4f} -> 结算 {position.amount * (1 + actual_return):.4f} "
                    f"(收益率: {actual_return:.4f})"
                )

        final_value = current_capital + matured_amount
        self.portfolio_value[trading_dates[-1]] = final_value

        print(
            f"\n最终现金: {current_capital:.4f}, "
            f"未到期投资结算后总额: {matured_amount:.4f}, "
            f"最终总资产: {final_value:.4f}"
        )

        print("\n" + "="*60)
        print("批量优化回测完成")
        print("="*60)

        return self.portfolio_value

    def run_benchmark_backtest(self, top_n: int) -> Dict[pd.Timestamp, float]:
        """
        运行基准策略回测（基于真实收益率排名的baseline策略）

        Args:
            top_n: 选择baseline top N个资产

        Returns:
            基准策略的资产价值序列
        """
        print(f"\n{'='*60}")
        print(f"开始Baseline策略回测: Top {top_n} (基于真实收益率排名)")
        print('='*60)

        # 确保已有parquet数据
        if not hasattr(self, 'full_parquet_df'):
            print("加载parquet数据...")
            self.full_parquet_df = self.build_full_panel_from_parquet()

        # 生成交易日期列表
        start_dt = pd.to_datetime(self.config.start_date)
        end_dt = pd.to_datetime(self.config.end_date)
        trading_dates = pd.date_range(start=start_dt, end=end_dt, freq='D')

        # 初始化变量
        current_capital = self.config.initial_capital
        investment_positions = {}  # {mature_date: [PortfolioPosition, ...]}
        portfolio_value = {}

        # 运行回测主循环
        for i, current_date in enumerate(trading_dates):
            if i % 10 == 0:  # 每10天打印一次进度
                print(f"\n--- Baseline交易日 {i+1}/{len(trading_dates)}: {current_date.strftime('%Y-%m-%d')} ---")

            try:
                # 1. 处理到期的投资
                matured_amount = 0.0
                if current_date in investment_positions:
                    matured_positions = investment_positions[current_date]
                    for position in matured_positions:
                        # 使用baseline策略计算收益
                        actual_return = self._calculate_baseline_portfolio_return(
                            position.invest_date,
                            top_n
                        )
                        matured_amount += position.amount * (1 + actual_return)

                        if i % 10 == 0:  # 只在进度时打印详细信息
                            print(f"  Baseline到期投资: {position.invest_date.strftime('%Y-%m-%d')} "
                                  f"投入 {position.amount:.4f} -> 收回 {position.amount * (1 + actual_return):.4f} "
                                  f"(收益率: {actual_return:.4f})")

                    del investment_positions[current_date]

                current_capital += matured_amount

                # 2. 确定当日投资金额
                if i < self.config.build_up_days:
                    # 建仓期：每天投入初始资金的1/build_up_days
                    invest_amount = self.config.initial_capital / self.config.build_up_days
                else:
                    # 正常期：投入所有可用资金
                    invest_amount = current_capital

                # 3. 使用baseline策略选择资产
                if invest_amount > 0:
                    baseline_asset_ids = self.select_baseline_top_assets(current_date, top_n)

                    if len(baseline_asset_ids) >= top_n:
                        # 4. 执行投资
                        current_capital -= invest_amount

                        # 创建持仓
                        weights = [1.0 / top_n] * top_n  # 等权重

                        position = PortfolioPosition(
                            invest_date=current_date,
                            amount=invest_amount,
                            assets=baseline_asset_ids[:top_n],  # 确保不超过top_n
                            weights=weights,
                            expected_return=0.0
                        )

                        # 计算到期日
                        mature_date = current_date + pd.Timedelta(days=self.config.holding_days)
                        if mature_date not in investment_positions:
                            investment_positions[mature_date] = []
                        investment_positions[mature_date].append(position)

                        if i % 10 == 0:
                            print(f"  Baseline投资 {invest_amount:.4f} 到 Top {top_n} 资产")

                # 5. 计算当日总资产价值
                total_invested = sum(pos.amount for positions in investment_positions.values()
                                   for pos in positions)
                total_value = current_capital + total_invested
                portfolio_value[current_date] = total_value

                if i % 10 == 0:
                    print(f"  Baseline当前现金: {current_capital:.4f}, 投资总额: {total_invested:.4f}, "
                          f"总资产: {total_value:.4f}")

            except Exception as e:
                print(f"Baseline策略 {current_date.strftime('%Y-%m-%d')} 处理失败: {e}")
                continue

        # 处理未到期投资（同样需要按收益率结算）
        matured_amount = 0.0
        for positions in investment_positions.values():
            for position in positions:
                actual_return = self._calculate_baseline_portfolio_return(
                    position.invest_date,
                    top_n
                )
                matured_amount += position.amount * (1 + actual_return)
                print(
                    f"  Baseline未到期投资: {position.invest_date.strftime('%Y-%m-%d')} "
                    f"投入 {position.amount:.4f} -> 结算 {position.amount * (1 + actual_return):.4f} "
                    f"(收益率: {actual_return:.4f})"
                )

        final_value = current_capital + matured_amount
        portfolio_value[trading_dates[-1]] = final_value

        print(f"\nBaseline策略 Top {top_n} 完成，最终资产: {final_value:.4f}")
        print('='*60)

        return portfolio_value

    def plot_results(self, main_portfolio_values: Dict, save_path: str = None):
        """绘制回测结果"""
        plt.figure(figsize=(14, 8))

        # 颜色列表 - 扩展更多颜色以支持更多基准策略
        colors = ['red', 'blue', 'green', 'orange', 'purple',
                  'brown', 'pink', 'gray', 'olive', 'cyan',
                  'magenta', 'yellow', 'black', 'navy', 'teal']

        # 1. 绘制主策略
        if main_portfolio_values:
            dates = list(main_portfolio_values.keys())
            values = list(main_portfolio_values.values())
            plt.plot(dates, values, label=f'Main Strategy (Top {self.config.top_n_main})',
                    linewidth=2.5, color=colors[0])

        # 2. 绘制基准策略
        for i, top_n in enumerate(self.config.top_n_benchmarks):
            print(f"计算基准策略 Top {top_n}...")
            benchmark_values = self.run_benchmark_backtest(top_n)
            if benchmark_values:
                dates = list(benchmark_values.keys())
                values = list(benchmark_values.values())
                plt.plot(dates, values, label=f'Benchmark Top {top_n}',
                        linewidth=1.5, color=colors[i+1], linestyle='--', alpha=0.8)

        plt.title('XGBoost回测结果对比', fontsize=16, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('总资产', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='best')

        # 格式化x轴日期
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=max(1, len(dates)//10)))
        plt.xticks(rotation=45)

        # 添加统计信息
        if main_portfolio_values:
            initial_value = self.config.initial_capital
            final_value = list(main_portfolio_values.values())[-1]
            total_return = (final_value / initial_value - 1) * 100
            plt.text(0.02, 0.98, f'主策略收益率: {total_return:.2f}%',
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")

        plt.show()

    def print_performance_summary(self, portfolio_values: Dict):
        """打印回测绩效总结"""
        if not portfolio_values:
            print("无回测数据")
            return

        initial_value = self.config.initial_capital
        final_value = list(portfolio_values.values())[-1]
        total_return = (final_value / initial_value - 1) * 100

        # 计算每日收益率
        values = list(portfolio_values.values())
        daily_returns = [values[i] / values[i-1] - 1 for i in range(1, len(values))]

        if daily_returns:
            annual_return = (final_value / initial_value) ** (365 / len(values)) - 1
            annual_return_pct = annual_return * 100

            volatility = np.std(daily_returns) * np.sqrt(365) * 100
            sharpe_ratio = annual_return / (volatility/100) if volatility > 0 else 0

            max_drawdown = 0
            peak = values[0]
            for value in values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            max_drawdown_pct = max_drawdown * 100
        else:
            annual_return_pct = 0
            volatility = 0
            sharpe_ratio = 0
            max_drawdown_pct = 0

        print("\n" + "="*60)
        print("绩效总结")
        print("="*60)
        print(f"初始资产: {initial_value:.4f}")
        print(f"最终资产: {final_value:.4f}")
        print(f"总收益率: {total_return:.2f}%")
        print(f"年化收益率: {annual_return_pct:.2f}%")
        print(f"年化波动率: {volatility:.2f}%")
        print(f"夏普比率: {sharpe_ratio:.3f}")
        print(f"最大回撤: {max_drawdown_pct:.2f}%")
        print("="*60)

def main():
    """主函数"""
    # 配置回测参数
    config = BacktestConfig(
        start_date='2025-11-01',
        end_date='2025-11-19',
        initial_capital=1.0,
        commission_rate=0.015,  # 1.5%手续费
        holding_days=8,
        top_n_main=6,
        top_n_benchmarks=[ 60,120,180,220],
        build_up_days=8
    )

    try:
        # 创建回测引擎
        engine = BacktestEngine(config)

        # 运行回测
        portfolio_values = engine.run_backtest()

        # 打印绩效总结
        engine.print_performance_summary(portfolio_values)

        # 绘制结果
        output_path = Path(__file__).parent / 'backtest_results6.png'
        engine.plot_results(portfolio_values, save_path=output_path)

        print(f"\n回测结果图表已保存到: {output_path}")

    except Exception as e:
        print(f"回测运行失败: {e}")
        import traceback
        traceback.print_exc()

def run_simple_backtest(start_date: str, end_date: str, top_n: int = 10):
    """运行简化版回测，用于快速测试"""
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=1.0,
        commission_rate=0.01,
        holding_days=8,
        top_n_main=top_n,
        top_n_benchmarks=[30, 50],  # 简化基准
        build_up_days=8
    )

    engine = BacktestEngine(config)
    portfolio_values = engine.run_backtest()
    engine.print_performance_summary(portfolio_values)

    return portfolio_values, engine

if __name__ == "__main__":
    main()
