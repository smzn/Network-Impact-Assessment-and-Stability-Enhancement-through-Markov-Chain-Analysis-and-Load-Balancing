import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class NetworkEvaluate:
    def __init__(self, file_name, output_dir = 'graph_v4', sheet_name='Total Influence', threshold=0.1):
        # エクセルファイルを読み込む
        self.data = pd.read_excel(file_name, sheet_name, engine='openpyxl')
        # 1列目はデバイス名、2列目以降が03〜30日の日付
        self.devices = self.data.iloc[:, 0]  # デバイス名（1列目）
        self.days = self.data.columns[1:]  # 03〜30日までの28日分
        self.data = self.data.set_index(self.devices)  # デバイス名をインデックスにする
        self.data = self.data.iloc[:, 1:]  # 日付部分のみを使用

        # NaN値を0に置き換える
        self.data = self.data.fillna(0)

        # 閾値の設定を表示
        print(f"設定された閾値 (Threshold): {threshold}")

        # 閾値をコンストラクタで受け取る
        self.threshold = threshold

        # 閾値以上のデバイスのみをフィルタリング
        self.data = self.data[self.data.mean(axis=1) >= self.threshold]

        # 欠損値(NaN)がある場合はその行・列を除外
        self.data = self.data.dropna(how='all', axis=1)  # 全てがNaNの列を除外
        self.data = self.data.dropna(how='all', axis=0)  # 全てがNaNの行を除外

        # フォルダが存在しない場合は作成
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.file_threshold = int(self.threshold * 100)

    def select_top_devices_for_replication(self):
        """各ルールに基づき、上位3つの複製推奨デバイスを表示"""
        # 結果を格納する辞書
        results = {}
        
        # 各デバイスの全日程平均と全日程標準偏差を計算
        device_means = self.data.mean(axis=1)
        device_std_devs = self.data.std(axis=1)

        # ルール1: 日ごとの偏差が大きいデバイス（各日の偏差が大きい上位3つのデバイス）
        daily_high_deviation_devices = []
        for day in self.days:
            day_data = self.data[day]
            mean_day_value = day_data.mean()
            deviations = abs(day_data - mean_day_value)
            top_deviations = deviations.nlargest(3)
            daily_high_deviation_devices.extend(list(top_deviations.index))

        # 集計と上位3デバイス
        rule1_top_devices = pd.Series(daily_high_deviation_devices).value_counts().nlargest(3)
        results['Rule1'] = [(device, count) for device, count in rule1_top_devices.items()]
        print("\nルール1: 日ごとの偏差が大きいデバイス")
        for rank, (device, count) in enumerate(rule1_top_devices.items(), 1):
            print(f"{rank}位: デバイス名={device}, 出現数={count}")

        # ルール2: 全体平均より高い影響度のデバイス
        high_mean_devices = device_means[device_means > device_means.mean() * 1.3].nlargest(3)
        results['Rule2'] = [(device, influence) for device, influence in high_mean_devices.items()]
        print("\nルール2: 全体平均より高い影響度のデバイス")
        for rank, (device, influence) in enumerate(high_mean_devices.items(), 1):
            print(f"{rank}位: デバイス名={device}, 影響度平均={influence:.4f}")

        # ルール3: 全日程標準偏差が高いデバイス
        high_std_dev_devices = device_std_devs.nlargest(3)
        results['Rule3'] = [(device, std_dev) for device, std_dev in high_std_dev_devices.items()]
        print("\nルール3: 全日程標準偏差が高いデバイス")
        for rank, (device, std_dev) in enumerate(high_std_dev_devices.items(), 1):
            print(f"{rank}位: デバイス名={device}, 標準偏差={std_dev:.4f}")

        # ルール4: 影響度が高い頻度が多いデバイス (10日以上で上位10%の影響度)
        high_influence_frequency_devices = {}
        for device in self.data.index:
            high_influence_days = (self.data.loc[device] > self.data.loc[device].quantile(0.9)).sum()
            if high_influence_days >= 5:
                high_influence_frequency_devices[device] = high_influence_days
        
        # 上位3デバイスを出力
        high_influence_frequency_devices = pd.Series(high_influence_frequency_devices, dtype='int').nlargest(3)
        print("\nルール4: 影響度が高い頻度が多いデバイス")
        for rank, (device, freq) in enumerate(high_influence_frequency_devices.items(), 1):
            print(f"{rank}位: デバイス名={device}, 高影響度日数={freq}")

        return results

    def device_level_stability(self):
        """デバイス単位安定化モデル: 各デバイスの影響度の変動を重み付きで抑える"""
        total_weighted_variation = 0  # 重み付きのばらつきの合計
        valid_device_count = len(self.data)  # 閾値以上のデバイス数を取得
        N = len(self.days)  # 日数（N日間）

        for device in self.data.index:
            device_data = self.data.loc[device]
            mean_value = device_data.mean()  # (R_total)̅(d_i) デバイス d_i に対する全日程の平均影響度
            #weight = mean_value  # w(d_i) = (R_total)̅(d_i) 重みは平均影響度に基づく
            weight = 1

            # 重み付きのばらつきを計算
            variance = ((device_data - mean_value) ** 2).sum()
            weighted_variance = weight * variance  # 重みを適用
            
            total_weighted_variation += weighted_variance

        # 目的関数値を計算（N日間、重み付きばらつきの平方根）
        if valid_device_count > 0:
            return np.sqrt(total_weighted_variation / (valid_device_count * N))
        else:
            return 0

    def global_network_load_balancing(self):
        """ネットワーク全体負荷分散モデル: 日毎のネットワーク影響度の平均を引いて、重みをつけてばらつきを抑える"""
        total_weighted_variation = 0
        N = len(self.days)  # 日数（N日間）
        deviations = []  # 偏差を保存するリスト

        if N == 0:
            return 0  # 日数が0の場合は0を返す

        for day in self.days:
            if day in self.data.columns:
                day_data = self.data[day]
                # その日のネットワーク影響度の平均 (n日目の全デバイス平均) (R_total^((n)) ) ̅
                mean_day_value = day_data.mean(skipna=True)  

                if np.isnan(mean_day_value):  # すべてが NaN の場合はスキップ
                    continue

                # 各デバイスに対して重みを適用してばらつきを計算
                for device in self.data.index:
                    device_value = self.data.loc[device, day]

                    # デバイス影響度がNaNでないことを確認
                    if not np.isnan(device_value): 
                        weight = 1  # 重み w(d_i) = 1
                        # 偏差の計算
                        deviation = device_value - mean_day_value
                        deviations.append((day, device, abs(deviation)))  # 日付、デバイス、偏差の絶対値を保存

                        # 重み付きばらつきを計算 (R_total^((n)) (d_i )-(R_total^((n)) ) ̅ )^2
                        weighted_variance = weight * (device_value - mean_day_value) ** 2
                        total_weighted_variation += weighted_variance

        # 偏差の絶対値でソートし、上位5件を取得
        top_5_deviations = sorted(deviations, key=lambda x: x[2], reverse=True)[:5]
        print("偏差の絶対値が大きい上位5セット（デバイス、日付、偏差）:")
        for day, device, deviation in top_5_deviations:
            print(f"日付: {day}, デバイス: {device}, 偏差: {deviation:.4f}")

        # 目的関数値を計算（日数Nでの正規化）
        if total_weighted_variation > 0:
            return np.sqrt(total_weighted_variation / N)
        else:
            return 0

    def daily_peak_device_load(self):
        """日毎最大デバイス負荷の標準偏差最小化モデル: 各日ごとの最大負荷の標準偏差を最小化（対数を目的関数に適用）"""
        total_variation = 0
        valid_device_count = len(self.data)  # 閾値以上のデバイス数を取得
        
        for day in self.days:
            if day in self.data.columns:
                day_data = self.data[day]
                max_value = day_data.max()
                mean_day = day_data.mean()
                total_variation += (max_value - mean_day) ** 2

        # 目的関数値に対数を適用
        return (np.sqrt(total_variation / len(self.days))) if valid_device_count > 0 else 0
        #return np.log1p(np.sqrt(total_variation / len(self.days))) if valid_device_count > 0 else 0

    def get_valid_device_count(self):
        """有効なデバイス数を返す"""
        return len(self.data)
    
    def plot_average_influence_histogram(self):
        """全日程のデバイスごとの平均影響度のヒストグラムを作成"""
        # 各デバイスの全日程における平均影響度
        average_influences = self.data.mean(axis=1)

        # ヒストグラムを作成
        plt.figure(figsize=(10, 6))
        plt.hist(average_influences, bins=20, color='blue', edgecolor='black')
        plt.title('Histogram of Average Device Influence (Across All Days)')
        plt.xlabel('Average Influence')
        plt.ylabel('Frequency')
        plt.grid(True)
        # グラフを保存
        plt.savefig(f'{self.output_dir}/average_device_influence_histogram_th{self.file_threshold}.png')
        plt.close()

    def plot_daily_influence_histograms(self):
        """各日の全デバイスの影響度のヒストグラムを作成"""
        for day in self.days:
            if day in self.data.columns:
                day_data = self.data[day]

                # ヒストグラムを作成
                plt.figure(figsize=(10, 6))
                plt.hist(day_data, bins=20, color='green', edgecolor='black')
                plt.title(f'Histogram of Device Influence for Day {day}')
                plt.xlabel('Influence')
                plt.ylabel('Frequency')
                plt.grid(True)
                # グラフを保存
                plt.savefig(f'{self.output_dir}/device_influence_histogram_day_{day}_th{self.file_threshold}.png')
                plt.close()


    def plot_daily_standard_deviation_trend(self):
        """日ごとのデバイス影響度の標準偏差の推移を折れ線グラフでファイルに保存"""
        daily_std_dev = []

        for day in self.days:
            if day in self.data.columns:
                day_data = self.data[day]
                std_dev = day_data.std(skipna=True)  # n日目の標準偏差を計算
                daily_std_dev.append(std_dev)

        # グラフを作成
        plt.figure(figsize=(12, 6))
        plt.plot(self.days, daily_std_dev, marker='o', linestyle='-', color='purple')
        plt.title('Trend of Daily Standard Deviation of Device Influence')
        plt.xlabel('Day')
        plt.ylabel('Standard Deviation')
        plt.grid(True)

        # フォルダがない場合は作成してファイルを保存
        os.makedirs('graph', exist_ok=True)
        plt.savefig(f'{self.output_dir}/daily_standard_deviation_trend_th{self.file_threshold}.png')
        plt.close()

    def plot_mean_absolute_deviation_boxplot(self):
        """各日のデバイス影響度の平均からの絶対偏差のボックスプロットをファイルに保存"""
        deviations = []

        for day in self.days:
            if day in self.data.columns:
                day_data = self.data[day]
                mean_day_value = day_data.mean(skipna=True)
                abs_deviation = np.abs(day_data - mean_day_value)  # 平均からの絶対偏差
                deviations.append(abs_deviation)

        # グラフを作成
        plt.figure(figsize=(12, 6))
        plt.boxplot(deviations, labels=self.days)
        plt.title('Boxplot of Absolute Deviations from Daily Mean Influence')
        plt.xlabel('Day')
        plt.ylabel('Absolute Deviation from Mean')
        plt.grid(True)

        # フォルダがない場合は作成してファイルを保存
        #os.makedirs('graph', exist_ok=True)
        plt.savefig(f'{self.output_dir}/mean_absolute_deviation_boxplot_th{self.file_threshold}.png')
        plt.close()

    def save_results_to_csv(self, results, file_name):
        """結果をCSVファイルに保存する"""
        df = pd.DataFrame(results)
        df.to_csv(file_name, index=False, encoding='utf-8-sig')
        print(f"結果をCSVファイルに保存しました: {file_name}")

# クラスのテストコード
if __name__ == "__main__":
    file_name = "result/network_influence.xlsx"
    #evaluator = NetworkEvaluate(file_name, sheet_name='Total Influence', threshold=0.5)
    #evaluator = NetworkEvaluate("result/network_influence_Comp275646_r1.xlsx", threshold=0.0)
    print('利用ファイル名: {}'.format(file_name))
    thresholds = np.arange(0, 0.6, 0.1)  # 0から0.1刻みで0.5までの閾値
    results = []  # 結果を格納するリスト
    output_dir = 'graph_v4'

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
