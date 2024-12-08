from mpi4py import MPI
from EquivalenceCalculation_MPI_v5 import EquivalenceCalculation_MPI  # クラスをインポート
import pandas as pd
import numpy as np
import os
import time  # 計算時間を測定するためのモジュール
import sys
from NetworkEvalulate_v4 import NetworkEvaluate

# 計算開始時間を記録
start_time = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 日付範囲と閾値を設定
start_day = 3
end_day = 30
threshold = 0.00
save_dir = 'result_MPI_v6'

# コマンドライン引数からの取得
if len(sys.argv) > 1:
    save_dir = sys.argv[1]
if len(sys.argv) > 2:
    start_day = int(sys.argv[2])
if len(sys.argv) > 3:
    end_day = int(sys.argv[3])

if rank == 0:  # マスタープロセスのみが表示
    print(f"Settings: save_dir={save_dir}, start_day={start_day}, end_day={end_day}")

# 各プロセスが担当する日付を決定
days_per_process = np.array_split(range(start_day, end_day + 1), size)
assigned_days = days_per_process[rank]

# 集計用のローカル変数
local_union = set()
local_intersection = None
local_results = []  # 各プロセスで日付ごとの結果を格納
local_network_influences = []  # 各プロセスで日付ごとのネットワーク影響度を格納
local_daily_centrality = []  # 各プロセスで日付ごとの daily_centrality を格納

for day in assigned_days:
    calc = EquivalenceCalculation_MPI(day, threshold)
    calc.load_transition_matrix()
    # デバッグ用: 行列がロードされているか確認
    if calc.transition_matrix is None:
        print(f"[Rank {rank}] Day {day}: Transition matrix not loaded.")
        continue

    devices = set(calc.get_devices())
    # デバッグ用: デバイス数を表示
    print(f"[Rank {rank}] Day {day}: Number of devices = {len(devices)}")

    calc.calculate_equivalence_and_centrality()
    result = calc.get_result()  # 1日分の結果を取得
    # デバッグ用: 結果の一部を表示
    print(f"[Rank {rank}] Day {day}: Result = {result}")

    calc.calculate_network_influence()
    network_influence = calc.get_network_influence()  # 1日分の影響度を取得
    # デバッグ用: ネットワーク影響度を表示
    print(f"[Rank {rank}] Day {day}: Network influence calculated")

    # 結果をリストに追加
    #local_daily_centrality.append((day, calc.daily_centrality))# daily_centralityをローカルリストに追加（この行を追加）
    local_daily_centrality.append((day, dict(calc.daily_centrality)))  # 辞書のコピーを保存
    local_results.append(result)
    local_network_influences.append((day, network_influence))

    # デバイスの和集合と積集合を計算
    local_union |= devices
    if local_intersection is None:
        local_intersection = devices
    else:
        local_intersection &= devices

# 各プロセスの結果をマスタープロセスに集約
all_results = comm.gather(local_results, root=0)
all_network_influences = comm.gather(local_network_influences, root=0)
all_daily_centrality = comm.gather(local_daily_centrality, root=0)
all_unions = comm.gather(local_union, root=0)
all_intersections = comm.gather(local_intersection, root=0)

if rank == 0:
    # 保存ディレクトリを指定
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 和集合と積集合の集約
    final_union = set().union(*all_unions)
    final_intersection = set.intersection(*filter(None, all_intersections))

    # 結果の保存
    pd.Series(list(final_union)).to_csv(f"{save_dir}/device_union.csv", index=False)
    pd.Series(list(final_intersection)).to_csv(f"{save_dir}/device_intersection.csv", index=False)
    print("Union and intersection sets saved.")

    # 各プロセスから集めた結果を1つのリストに統合
    combined_results = [result for sublist in all_results for result in sublist]
    combined_network_influences = {day: net_inf for sublist in all_network_influences for day, net_inf in sublist}
    combined_daily_centrality = {day: day_data for sublist in all_daily_centrality for day, day_data in sublist}

    # DataFrameに変換して保存
    result_df = pd.DataFrame(combined_results)
    result_df.to_csv(f"{save_dir}/equivalence_and_centrality_summary.csv", index=False)
    print("Equivalence and centrality results saved.")

    # ネットワーク影響度の保存
    for day, net_inf in combined_network_influences.items():
        net_inf_df = pd.DataFrame.from_dict(net_inf, orient='index')
        net_inf_df.to_csv(f"{save_dir}/network_influence_day_{day}.csv", index=True)
    print("Network influence results saved.")

    # all_network_influencesから全日程のデータを集約
    all_influence_data = {
        'self_influence': pd.DataFrame(),
        'in_bound_influence': pd.DataFrame(),
        'out_bound_influence': pd.DataFrame(),
        'total_influence': pd.DataFrame()
    }

    # 全日程のデータを統合
    for day, network_influence in combined_network_influences.items():
        # 各影響度タイプについて、その日のデータを追加
        day_data = {
            'self_influence': {d: v['self_influence'] for d, v in network_influence.items()},
            'in_bound_influence': {d: v['in_bound_influence'] for d, v in network_influence.items()},
            'out_bound_influence': {d: v['out_bound_influence'] for d, v in network_influence.items()},
            'total_influence': {d: v['total_influence'] for d, v in network_influence.items()}
        }
        
        # 各影響度タイプのDataFrameに列として追加
        for influence_type in all_influence_data:
            all_influence_data[influence_type][day] = pd.Series(day_data[influence_type])

    # 全日程のデータをExcelファイルとして保存
    excel_path = os.path.join(save_dir, 'network_influence.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        for influence_type, df in all_influence_data.items():
            # 列（日付）を順番に並び替え
            df = df.reindex(columns=sorted(df.columns))
            df.to_excel(writer, sheet_name=influence_type.replace('_', ' ').title())

    print(f"Complete network influence data saved to '{excel_path}'")

    # 集約されたデータを用いてグラフを生成
    calc.generate_graphs_centrality(result_df, combined_daily_centrality, save_dir=save_dir)
    calc.generate_graphs_influence(combined_network_influences, save_dir=save_dir)

    # 計算終了時間を記録して表示
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total computation time: {elapsed_time:.2f} seconds")

    # Network Evaluateの開始
    # 計算開始時間を記録
    start_time = time.time()
    print("Start Network Evaluate")

    file_name = f"{save_dir}/network_influence.xlsx"
    #evaluator = NetworkEvaluate(file_name, sheet_name='Total Influence', threshold=0.5)
    #evaluator = NetworkEvaluate("result/network_influence_Comp275646_r1.xlsx", threshold=0.0)
    print('利用ファイル名: {}'.format(file_name))
    thresholds = np.arange(0, 0.6, 0.1)  # 0から0.1刻みで0.5までの閾値
    results = []  # 結果を格納するリスト
    output_dir = 'graph_v6'

    for threshold in thresholds:
        print(f"\n=== 閾値: {threshold} ===")
        evaluator = NetworkEvaluate(file_name, output_dir, sheet_name='Total Influence', threshold=threshold)
    
        # 有効なデバイス数の表示
        valid_device_count = evaluator.get_valid_device_count()
        print(f"対象となるデバイス数: {valid_device_count}")

        # デバイス単位安定化モデル
        device_stability_value = evaluator.device_level_stability()
        print(f"デバイス単位安定化モデルの評価値: {device_stability_value:.4f}")
        
        # ネットワーク全体負荷分散モデル
        network_balancing_value = evaluator.global_network_load_balancing()
        print(f"ネットワーク全体負荷分散モデルの評価値: {network_balancing_value:.4f}")
        
        # 日毎最大デバイス負荷の標準偏差最小化モデル
        peak_load_value = evaluator.daily_peak_device_load()
        print(f"日毎最大デバイス負荷の標準偏差最小化モデルの評価値: {peak_load_value:.4f}")

        # 全日程のデバイスごとの平均影響度ヒストグラム
        evaluator.plot_average_influence_histogram()

        # 各日の影響度のヒストグラムを作成
        evaluator.plot_daily_influence_histograms()

        # 日ごとの標準偏差の推移をグラフ保存
        evaluator.plot_daily_standard_deviation_trend()
        
        # 平均からの絶対偏差のボックスプロットをグラフ保存
        evaluator.plot_mean_absolute_deviation_boxplot()

        # 各ルールの上位デバイスと値を取得
        top_devices = evaluator.select_top_devices_for_replication()

        # 各閾値に対応する結果を収集
        valid_device_count = evaluator.get_valid_device_count()
        device_stability_value = evaluator.device_level_stability()
        network_balancing_value = evaluator.global_network_load_balancing()
        peak_load_value = evaluator.daily_peak_device_load()
        
        # 各閾値ごとの結果を辞書形式で格納
        results.append({
            "Threshold": threshold,
            "ValidDeviceCount": valid_device_count,
            "DeviceStabilityValue": device_stability_value,
            "NetworkBalancingValue": network_balancing_value,
            "PeakLoadValue": peak_load_value,
            "Rule1_TopDevices": "; ".join([f"{device}({count})" for device, count in top_devices['Rule1']]),
            "Rule2_TopDevices": "; ".join([f"{device}({influence:.4f})" for device, influence in top_devices['Rule2']]),
            "Rule3_TopDevices": "; ".join([f"{device}({std_dev:.4f})" for device, std_dev in top_devices['Rule3']])
        })

    # 結果を保存
    output_csv = f"{output_dir}/network_evaluation_results.csv"

    # CSV形式で保存
    evaluator.save_results_to_csv(results, output_csv)




