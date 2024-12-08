import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

class EquivalenceCalculation_MPI:
    def __init__(self, day, threshold=0.0):
        """
        コンストラクタ: 指定された日付と閾値でクラスを初期化します。
        
        :param day: 処理する日付
        :param threshold: 中心性計算で使用する閾値
        """
        self.day = day  # 処理する日付
        self.threshold = threshold  # 閾値
        self.transition_matrix = None  # 推移確率行列
        self.daily_result = None  # 日付ごとの結果を保存する辞書
        self.daily_centrality = defaultdict(dict)  # 日付ごとの中心性計算結果
        self.devices = None

    def load_transition_matrix(self):
        """
        指定された日付の推移確率行列を読み込む。
        """
        day_str = f"{self.day:02}"
        file_path = f"../../4_adjusted_transition/ActiveDirectory_01/result_ActiveDirectory_01/netflow_day-{day_str}_transition_probability_01.csv"
        if os.path.exists(file_path):
            matrix = pd.read_csv(file_path, index_col=0)
            
            # 行と列のデバイスを揃える
            all_devices = set(matrix.index) | set(matrix.columns)
            matrix = matrix.reindex(index=all_devices, columns=all_devices, fill_value=0)
            
            self.transition_matrix = matrix
            print(f"Loaded transition matrix for day {day_str}")
        else:
            print(f"File {file_path} not found.")

    def get_devices(self):
        """
        推移確率行列からデバイスの集合を取得する。
        """
        if self.transition_matrix is not None:
            self.devices = list(set(self.transition_matrix.index) | set(self.transition_matrix.columns))
            return self.devices
        else:
            raise ValueError("Transition matrix not loaded.")
        
    def calculate_equivalence_and_centrality(self):
        """
        同値類と次数中心性を計算し、日付ごとの結果を保持します。
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not loaded.")

        # (1) 出現デバイス数
        num_devices = len(self.devices)

        # (2) 同値類の数
        #equivalence_classes = self._find_equivalence_classes(self.transition_matrix)
        sparse_matrix = csr_matrix(self.transition_matrix)
        self.classify_equivalence_classes(sparse_matrix)
        num_equivalence_classes = len(self.equivalence_classes)

        # (3) 最大同値類の要素数
        largest_equivalence_class_size = max(len(eq) for eq in self.equivalence_classes)

        # (4) 最大同値類へのデバイス所属率
        largest_equivalence_ratio = largest_equivalence_class_size / num_devices

        # (5) 閾値に基づいて0/1に変換した二値行列を作成
        binary_matrix = (self.transition_matrix > self.threshold).astype(int)

        # (6) 無向グラフの次数中心性の計算と正規化
        degrees_undirected = binary_matrix.sum(axis=1) + binary_matrix.sum(axis=0)
        normalized_degrees_undirected = degrees_undirected / (num_devices - 1)
        max_degree_undirected = normalized_degrees_undirected.max()
        max_degree_undirected_device = normalized_degrees_undirected.idxmax()

        # (7) 入次数中心性（列和）の計算と正規化
        in_degrees = binary_matrix.sum(axis=0)
        normalized_in_degrees = in_degrees / (num_devices - 1)
        max_in_degree = normalized_in_degrees.max()
        max_in_degree_device = normalized_in_degrees.idxmax()

        # (8) 出次数中心性（行和）の計算と正規化
        out_degrees = binary_matrix.sum(axis=1)
        normalized_out_degrees = out_degrees / (num_devices - 1)
        max_out_degree = normalized_out_degrees.max()
        max_out_degree_device = normalized_out_degrees.idxmax()

        # (9) 最大次数中心性デバイスの所属する同値類の要素数
        device_equivalence_class_size = self._find_equivalence_class_size(max_degree_undirected_device)

        # (10) この日の正規化された次数中心性度数平均（無向グラフ）
        avg_degree = normalized_degrees_undirected.mean()

        # (11) この日の正規化された次数中心性度数標準偏差（無向グラフ）
        std_degree = normalized_degrees_undirected.std()

        # (12) 最大入次数中心性度数（有向グラフ） - 正規化された次数中心性
        max_in_degree_value = normalized_in_degrees.max()

        # (13) 最大入次数中心性度数を持つデバイス名
        max_in_degree_device = normalized_in_degrees.idxmax()

        # (14) 最大入次数中心性デバイスの次数中心性度数（無向グラフ）
        max_in_degree_device_total_degree = normalized_degrees_undirected[max_in_degree_device]

        # (15) 最大入次数中心性デバイスの出次数中心性度数（有向グラフ）
        max_in_degree_device_out_degree = normalized_out_degrees[max_in_degree_device]

        # (16) この日の正規化された入次数中心性度数平均（有向グラフ）
        avg_in_degree = normalized_in_degrees.mean()

        # (17) この日の正規化された入次数中心性度数標準偏差（有向グラフ）
        std_in_degree = normalized_in_degrees.std()

        # (18) 最大出次数中心性度数（有向グラフ） - 正規化された次数中心性
        max_out_degree_value = normalized_out_degrees.max()

        # (19) 最大出次数中心性度数を持つデバイス名
        max_out_degree_device = normalized_out_degrees.idxmax()

        # (20) 最大出次数中心性デバイスの正規化された次数中心性度数（無向グラフ）
        max_out_degree_device_total_degree = normalized_degrees_undirected[max_out_degree_device]

        # (21) 最大出次数中心性デバイスの入次数中心性度数（有向グラフ）
        max_out_degree_device_in_degree = normalized_in_degrees[max_out_degree_device]

        # (22) この日の正規化された出次数中心性度数平均（有向グラフ）
        avg_out_degree = normalized_out_degrees.mean()

        # (23) この日の正規化された出次数中心性度数標準偏差（有向グラフ）
        std_out_degree = normalized_out_degrees.std()

        # 日付ごとの中心性を保持
        self.daily_centrality['undirected'] = normalized_degrees_undirected.to_dict()
        self.daily_centrality['in_degree'] = normalized_in_degrees.to_dict()
        self.daily_centrality['out_degree'] = normalized_out_degrees.to_dict()

        # 結果を保持
        self.daily_result = {
            "day": self.day,
            "num_devices": num_devices,
            "num_equivalence_classes": num_equivalence_classes,
            "largest_equivalence_class_size": largest_equivalence_class_size,
            "largest_equivalence_ratio": largest_equivalence_ratio,
            "max_degree_undirected": max_degree_undirected,
            "max_degree_undirected_device": max_degree_undirected_device,
            "max_in_degree": max_in_degree,
            "max_out_degree": max_out_degree,
            "device_equivalence_class_size": device_equivalence_class_size,
            "avg_degree": avg_degree,
            "std_degree": std_degree,
            "max_in_degree_value": max_in_degree_value,
            "max_in_degree_device": max_in_degree_device,
            "max_in_degree_device_total_degree": max_in_degree_device_total_degree,
            "max_in_degree_device_out_degree": max_in_degree_device_out_degree,
            "avg_in_degree": avg_in_degree,
            "std_in_degree": std_in_degree,
            "max_out_degree_value": max_out_degree_value,
            "max_out_degree_device": max_out_degree_device,
            "max_out_degree_device_total_degree": max_out_degree_device_total_degree,
            "max_out_degree_device_in_degree": max_out_degree_device_in_degree,
            "avg_out_degree": avg_out_degree,
            "std_out_degree": std_out_degree,
        }

    def get_result(self):
        """
        日付ごとの計算結果を返します。
        """
        if self.daily_result is not None:
            return self.daily_result
        else:
            raise ValueError("No results available. Ensure calculate_equivalence_and_centrality() has been called.")
        
    def calculate_network_influence(self):
        """
        指定された日付のネットワーク影響度を計算し、結果を保持および保存します。
        
        :param save_dir: 保存先のディレクトリ
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not loaded.")
        
        #equivalence_classes = self._find_equivalence_classes(self.transition_matrix)
        #devices = self.transition_matrix.index.tolist()
        transition_matrix = self.transition_matrix.values

        # 中心性指標を取得
        undirected_degree = np.array([self.daily_centrality['undirected'].get(d, 0) for d in self.devices])
        in_degree = np.array([self.daily_centrality['in_degree'].get(d, 0) for d in self.devices])
        out_degree = np.array([self.daily_centrality['out_degree'].get(d, 0) for d in self.devices])

        # デバッグ用: 中心性指標の内容を表示
        print("Debug: undirected_degree values:", undirected_degree[:10])  # 最初の10個を表示
        print("Debug: in_degree values:", in_degree[:10])
        print("Debug: out_degree values:", out_degree[:10])
        
        # 全体の内容を確認したい場合
        print("Debug: Total undirected_degree:", undirected_degree)
        print("Debug: Total in_degree:", in_degree)
        print("Debug: Total out_degree:", out_degree)
        
        # 同値類サイズの配列を作成
        #eq_class_sizes = np.array([self._find_equivalence_class_size(d) for d in self.devices])
        eq_class_sizes = np.array([self._find_equivalence_class_size(i) for i in range(len(self.devices))])
        avg_max_eq_class_size = max(eq_class_sizes)
        print("Debug: Equivalence Class Size:", eq_class_sizes)
        print("Debug: Average Max Equivalence Class Size:", avg_max_eq_class_size)


        # 自己影響度の計算
        self_inf = undirected_degree * (eq_class_sizes / avg_max_eq_class_size)

        # Inbound影響度の計算
        in_bound_inf = in_degree + np.dot(transition_matrix.T, undirected_degree)

        # Outbound影響度の計算
        out_bound_inf = out_degree + np.dot(transition_matrix, undirected_degree)

        # 全体のネットワーク影響度の計算 (α = β = γ = 1)
        total_inf = self_inf + in_bound_inf + out_bound_inf

        # 結果を格納
        self.network_influence = {
            self.devices[i]: {
                'self_influence': self_inf[i],
                'in_bound_influence': in_bound_inf[i],
                'out_bound_influence': out_bound_inf[i],
                'total_influence': total_inf[i]
            } for i in range(len(self.devices))
        }

    def get_network_influence(self):
        """
        計算されたネットワーク影響度を返します。
        """
        if self.network_influence:
            return self.network_influence
        else:
            raise ValueError("No network influence data available. Ensure calculate_network_influence() has been called.")
        
    def generate_graphs_centrality(self, result_df, combined_daily_centrality, save_dir='result'):
        """
        集約されたデータを基にグラフを生成し、指定されたディレクトリに保存する。
        
        :param result_df: 集計された結果のDataFrame
        :param save_dir: グラフを保存するディレクトリ（デフォルトは 'result'）
        """
        # 保存ディレクトリが存在しない場合は作成
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # (1) Device count over time
        plt.figure(figsize=(12, 6))
        plt.plot(result_df['day'], result_df['num_devices'])
        plt.title('Device Count Over Time')
        plt.xlabel('Day')
        plt.ylabel('Number of Devices')
        plt.grid(True)
        plt.savefig(f'{save_dir}/device_count_over_time.png')
        plt.close()

        avg_devices = result_df['num_devices'].mean()
        std_devices = result_df['num_devices'].std()
        print(f"Average number of devices: {avg_devices:.2f}")
        print(f"Standard deviation of device count: {std_devices:.2f}")

        # (2) Equivalence class count over time
        plt.figure(figsize=(12, 6))
        plt.plot(result_df['day'], result_df['num_equivalence_classes'])
        plt.title('Equivalence Class Count Over Time')
        plt.xlabel('Day')
        plt.ylabel('Number of Equivalence Classes')
        plt.grid(True)
        plt.savefig(f'{save_dir}/equivalence_class_count_over_time.png')
        plt.close()

        avg_classes = result_df['num_equivalence_classes'].mean()
        std_classes = result_df['num_equivalence_classes'].std()
        print(f"Average number of equivalence classes: {avg_classes:.2f}")
        print(f"Standard deviation of equivalence class count: {std_classes:.2f}")

        # (3) Largest equivalence class size over time
        plt.figure(figsize=(12, 6))
        plt.plot(result_df['day'], result_df['largest_equivalence_class_size'])
        plt.title('Largest Equivalence Class Size Over Time')
        plt.xlabel('Day')
        plt.ylabel('Size of Largest Equivalence Class')
        plt.grid(True)
        plt.savefig(f'{save_dir}/largest_equivalence_class_size_over_time.png')
        plt.close()

        avg_largest_class = result_df['largest_equivalence_class_size'].mean()
        std_largest_class = result_df['largest_equivalence_class_size'].std()
        print(f"Average size of largest equivalence class: {avg_largest_class:.2f}")
        print(f"Standard deviation of largest equivalence class size: {std_largest_class:.2f}")

        # (4) Undirected graph centrality over time for top 10 devices
        all_devices = set()
        daily_centrality_values = {}  # 日付ごとのデバイスの中心性値を保存

        # まず、全デバイスを収集し、日付ごとの中心性値を整理
        for day, day_data in combined_daily_centrality.items():
            if 'undirected' in day_data:
                current_devices = day_data['undirected'].keys()
                all_devices.update(current_devices)
                
                # 日付ごとの中心性値を保存
                if day not in daily_centrality_values:
                    daily_centrality_values[day] = {}
                
                for device in current_devices:
                    daily_centrality_values[day][device] = day_data['undirected'][device]

        # デバッグ用: 日付ごとの中心性値を出力
        print("Debug: Daily centrality values:")
        #for day in sorted(daily_centrality_values.keys()):
        #    print(f"Day {day}: {len(daily_centrality_values[day])} devices")

        # デバイスごとの平均中心性を計算
        device_avg_centrality = {}
        for device in all_devices:
            values = []
            for day in sorted(daily_centrality_values.keys()):
                if device in daily_centrality_values[day]:
                    values.append(daily_centrality_values[day][device])
            if values:  # 値が存在する場合のみ平均を計算
                device_avg_centrality[device] = np.mean(values)

        # トップ10のデバイスを選定
        top_10_devices = sorted(device_avg_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_10_device_names = [device for device, _ in top_10_devices]

        print("Debug: Top 10 devices and their average centrality:")
        for device, avg in top_10_devices:
            print(f"{device}: {avg}")

        # グラフの描画
        plt.figure(figsize=(15, 8))
        days_sorted = sorted(daily_centrality_values.keys())

        for device in top_10_device_names:
            centrality_values = []
            for day in days_sorted:
                value = daily_centrality_values[day].get(device, 0)
                centrality_values.append(value)
                
            # デバッグ用: デバイスごとの中心性値の変化を出力
            print(f"Device: {device}")
            print(f"Values: {centrality_values}")
            
            plt.plot(days_sorted, centrality_values, marker='o', label=device)

        plt.title('Undirected Graph Centrality Over Time (Top 10 Devices)')
        plt.xlabel('Day')
        plt.ylabel('Undirected Graph Centrality Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/undirected_centrality_over_time.png')
        plt.close()

        # (5) In-degree centrality over time for top 10 devices
        daily_in_centrality_values = {}  # 日付ごとのin-degree中心性値を保存

        # in-degree中心性値の収集
        for day, day_data in combined_daily_centrality.items():
            if 'in_degree' in day_data:
                if day not in daily_in_centrality_values:
                    daily_in_centrality_values[day] = {}
                for device in day_data['in_degree']:
                    daily_in_centrality_values[day][device] = day_data['in_degree'][device]

        # デバイスごとの平均in-degree中心性を計算
        device_avg_in_centrality = {}
        for device in all_devices:
            values = []
            for day in sorted(daily_in_centrality_values.keys()):
                if device in daily_in_centrality_values[day]:
                    values.append(daily_in_centrality_values[day][device])
            if values:
                device_avg_in_centrality[device] = np.mean(values)

        # トップ10のデバイスを選定
        top_10_in_devices = sorted(device_avg_in_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_10_in_device_names = [device for device, _ in top_10_in_devices]

        print("Debug: Top 10 devices for in-degree centrality:")
        for device, avg in top_10_in_devices:
            print(f"{device}: {avg}")

        # グラフの描画
        plt.figure(figsize=(15, 8))
        days_sorted = sorted(daily_in_centrality_values.keys())

        for device in top_10_in_device_names:
            centrality_values = []
            for day in days_sorted:
                value = daily_in_centrality_values[day].get(device, 0)
                centrality_values.append(value)
            
            print(f"Device: {device}")
            print(f"In-Degree Values: {centrality_values}")
            
            plt.plot(days_sorted, centrality_values, marker='o', label=device)

        plt.title('In-Degree Centrality Over Time (Top 10 Devices)')
        plt.xlabel('Day')
        plt.ylabel('In-Degree Centrality Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/in_degree_centrality_over_time.png')
        plt.close()

        # (6) Out-degree centrality over time for top 10 devices
        daily_out_centrality_values = {}  # 日付ごとのout-degree中心性値を保存

        # out-degree中心性値の収集
        for day, day_data in combined_daily_centrality.items():
            if 'out_degree' in day_data:
                if day not in daily_out_centrality_values:
                    daily_out_centrality_values[day] = {}
                for device in day_data['out_degree']:
                    daily_out_centrality_values[day][device] = day_data['out_degree'][device]

        # デバイスごとの平均out-degree中心性を計算
        device_avg_out_centrality = {}
        for device in all_devices:
            values = []
            for day in sorted(daily_out_centrality_values.keys()):
                # 欠損値を0で補完
                value = daily_out_centrality_values[day].get(device, 0)
                values.append(value)
                #if device in daily_out_centrality_values[day]:
                #    values.append(daily_out_centrality_values[day][device])
            if values:
                device_avg_out_centrality[device] = np.mean(values)

        # トップ10のデバイスを選定
        top_10_out_devices = sorted(device_avg_out_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        top_10_out_device_names = [device for device, _ in top_10_out_devices]

        print("Debug: Top 10 devices for out-degree centrality:")
        for device, avg in top_10_out_devices:
            print(f"{device}: {avg}")

        # グラフの描画
        plt.figure(figsize=(15, 8))
        days_sorted = sorted(daily_out_centrality_values.keys())

        for device in top_10_out_device_names:
            centrality_values = []
            for day in days_sorted:
                value = daily_out_centrality_values[day].get(device, 0)
                centrality_values.append(value)
            
            print(f"Device: {device}")
            print(f"Out-Degree Values: {centrality_values}")
            
            plt.plot(days_sorted, centrality_values, marker='o', label=device)

        plt.title('Out-Degree Centrality Over Time (Top 10 Devices)')
        plt.xlabel('Day')
        plt.ylabel('Out-Degree Centrality Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/out_degree_centrality_over_time.png')
        plt.close()

        print("All graphs have been generated and saved in the specified directory.")


    def _plot_top_devices_over_time(self, influence_type, title, network_influence, save_dir='result_MPI'):
        """
        ネットワーク影響度の時間変化を上位10デバイスでプロットし、指定されたディレクトリに保存する。

        :param influence_type: プロットする影響度の種類（例: 'out_bound_influence'）
        :param title: グラフのタイトル
        :param network_influence: 各日付の影響度のデータ
        :param save_dir: グラフを保存するディレクトリ（デフォルトは 'result'）
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 全デバイスの平均影響度を計算
        all_devices = set()
        for day_data in network_influence.values():
            all_devices.update(day_data.keys())
        
#        device_avg_influence = {
#            device: np.mean([
#                network_influence[day][device].get(influence_type, 0)
#                for day in network_influence if device in network_influence[day]
#            ])
#            for device in all_devices
#        }

        # 全日程数を network_influence のキーから取得
        total_days = len(network_influence.keys())

        # 全デバイスの平均影響度を計算（空の日は0として扱い、常に全日程数で割る）
        device_avg_influence = {
            device: sum([
                network_influence.get(day, {}).get(device, {}).get(influence_type, 0)
                for day in sorted(network_influence.keys())
            ]) / total_days
            for device in all_devices
        }


        # 平均影響度が高い上位10デバイスを選択
        top_10_devices = sorted(device_avg_influence, key=device_avg_influence.get, reverse=True)[:10]

        # 日付のリストを作成
        days = sorted(network_influence.keys())

        # グラフの描画
        plt.figure(figsize=(15, 8))
        for device in top_10_devices:
            influence_values = [
                network_influence[day].get(device, {}).get(influence_type, 0) for day in days
            ]

            plt.plot(days, influence_values, label=device, marker='o')

        plt.title(f'{title} Over Time (Top 10 Devices)')
        plt.xlabel('Day')
        plt.ylabel(f'{title} Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{influence_type.lower()}_over_time.png')
        plt.close()

        print(f"{title} graph has been generated and saved in {save_dir}.")

    def generate_graphs_influence(self, network_influence, save_dir='result_MPI'):
        """
        集約されたデータを基にグラフを生成し、指定されたディレクトリに保存する。
        
        :param result_df: 集計された結果のDataFrame
        :param network_influence: 各日付の影響度のデータ
        :param save_dir: グラフを保存するディレクトリ（デフォルトは 'result'）
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # グラフを生成する各メトリックを呼び出し
        self._plot_top_devices_over_time('total_influence', 'Network Influence', network_influence, save_dir=save_dir)
        self._plot_top_devices_over_time('self_influence', 'Self Influence', network_influence, save_dir=save_dir)
        self._plot_top_devices_over_time('in_bound_influence', 'In Bound Influence', network_influence, save_dir=save_dir)
        self._plot_top_devices_over_time('out_bound_influence', 'Out Bound Influence', network_influence, save_dir=save_dir)

        print("All detailed influence graphs have been generated and saved.")

    def classify_equivalence_classes(self, matrix):
        reachability_matrix = (matrix > 0).astype(int)
        n_components, labels = connected_components(csgraph=reachability_matrix, directed=True, connection='strong')
        self.equivalence_classes = [[] for _ in range(n_components)]
        for state, label in enumerate(labels):
            self.equivalence_classes[label].append(state)
        #return self.equivalence_classes, n_components


    # デバイスが属する同値類のサイズを返す
    def _find_equivalence_class_size(self, device):
        for eq_class in self.equivalence_classes:
            if device in eq_class:
                return len(eq_class)
        return 0

