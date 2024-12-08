import pandas as pd
import numpy as np
import time
import sys
import os

class NetworkTransition:
    def __init__(self, input_file_path, input_file_name, output_folder):
        self.input_file_path = input_file_path
        self.input_file_name = input_file_name
        self.output_folder = output_folder
        self.file = os.path.join(self.input_file_path, self.input_file_name)  # フルパスの生成
        self.base_file_name = os.path.splitext(self.input_file_name)[0]  # 拡張子なしのファイル名
        self.df = None
        self.device_list = None
        self.transition_count = None

    def load_data(self):
        # データの読み込みと前処理
        self.df = pd.read_csv(self.file, header=None)
        self.df.columns = ['Time', 'Duration', 'SrcDevice', 'DstDevice', 'Protocol', 'SrcPort', 'DstPort', 'SrcPackets', 'DstPackets', 'SrcBytes', 'DstBytes']

        # Byte数を10^6 -> Mbyteに変換
        self.df['SrcPackets'] = self.df['SrcPackets'] / 10**6
        self.df['DstPackets'] = self.df['DstPackets'] / 10**6
        self.df['SrcBytes'] = self.df['SrcBytes'] / 10**6
        self.df['DstBytes'] = self.df['DstBytes'] / 10**6
        print('Data loaded and preprocessed.')

    def create_device_list(self):
        # デバイスリストの作成
        SrcDevice_set = set(self.df['SrcDevice'].unique())
        DstDevice_set = set(self.df['DstDevice'].unique())

        # 和集合と積集合の作成
        device_set_intersection = SrcDevice_set & DstDevice_set
        self.device_list = list(device_set_intersection)

        # デバイスリストの保存
        device_df_intersection = pd.DataFrame({'Device': self.device_list})
        device_df_intersection.to_csv(os.path.join(self.output_folder, self.base_file_name + '_device_intersection.csv'), index=False)
        print('Device list created and saved.')

    def compute_device_communication(self):
        # デバイス間の通信集計
        device_weight = []
        from_device = []
        to_device = []
        from_packets = []
        to_packets = []
        from_bytes = []
        to_bytes = []
        duration = []

        start = time.time()
        index = 1
        for src, sub_df in self.df.groupby('SrcDevice'):
            for dst, subsub_df in sub_df.groupby('DstDevice'):
                if src in self.device_list and dst in self.device_list:
                    from_device.append(src)
                    to_device.append(dst)
                    device_weight.append(len(subsub_df))
                    from_packets.append(subsub_df['SrcPackets'].sum())
                    to_packets.append(subsub_df['DstPackets'].sum())
                    from_bytes.append(subsub_df['SrcBytes'].sum())
                    to_bytes.append(subsub_df['DstBytes'].sum())
                    duration.append(subsub_df['Duration'].sum())
                index += 1

        df_sum = pd.DataFrame({
            'from_device': from_device,
            'to_device': to_device,
            'count': device_weight,
            'duration': duration,
            'from_packets': from_packets,
            'to_packets': to_packets,
            'from_bytes': from_bytes,
            'to_bytes': to_bytes
        })
        df_sum.to_csv(os.path.join(self.output_folder, self.base_file_name + '_sum.csv'), index=False)
        elapsed_time = time.time() - start
        print(f'Communication data computed and saved. Calculation time: {elapsed_time:.2f} seconds.')

    def build_transition_matrix(self):
        # 推移行列の初期化
        self.transition_count = np.zeros((len(self.device_list), len(self.device_list)))

        # 推移行列の構築
        df_sum = pd.read_csv(os.path.join(self.output_folder, self.base_file_name + '_sum.csv'))
        for row in df_sum.itertuples():
            from_device = row[1]  # 'from_device' の列に対応
            to_device = row[2]    # 'to_device' の列に対応
            if from_device in self.device_list and to_device in self.device_list:
                idx1 = self.device_list.index(from_device)
                idx2 = self.device_list.index(to_device)
                self.transition_count[idx1, idx2] = row.count

        print('Transition matrix built.')

    def refine_transition_matrix(self):
        # 行和と列和が0のデバイスを削除
        iteration = 0
        while True:
            row_sum = np.sum(self.transition_count, axis=1)
            col_sum = np.sum(self.transition_count, axis=0)

            # 行和または列和が0のインデックスを取得
            row_zero_index = np.where(row_sum == 0)[0]
            col_zero_index = np.where(col_sum == 0)[0]

            # 行和も列和も0の行列を確認し、両方同時に削除
            zero_index = np.union1d(row_zero_index, col_zero_index)

            if len(zero_index) == 0:
                # すべての行和と列和が0でない場合、ループを終了
                break

            # 行と列から削除
            print(f'Iteration {iteration}: Removing zero sum rows/columns. Removing {len(zero_index)} rows/columns.')

            self.transition_count = np.delete(self.transition_count, zero_index, axis=0)
            self.transition_count = np.delete(self.transition_count, zero_index, axis=1)

            # デバイスリストからも対応するインデックスを削除
            self.device_list = [device for i, device in enumerate(self.device_list) if i not in zero_index]

            iteration += 1

        print(f'Final transition matrix size: {self.transition_count.shape}')

    def convert_to_transition_probability(self):
        # 推移行列を推移確率行列に変換
        transition_prob_matrix = self.transition_count.astype(np.float32)
        row_sum = np.sum(transition_prob_matrix, axis=1)
        for i in range(len(transition_prob_matrix)):
            if row_sum[i] != 0:
                transition_prob_matrix[i] /= row_sum[i]

        # デバイスリスト付きでDataFrame化して保存
        df_transition_prob = pd.DataFrame(transition_prob_matrix, index=self.device_list, columns=self.device_list)
        df_transition_prob.to_csv(os.path.join(self.output_folder, self.base_file_name + '_transition_probability.csv'), index=True)
        print('Transition probability matrix saved.')

    def save_results(self):
        # 最終結果の保存
        df_device_list = pd.DataFrame(self.device_list, columns=['Device'])
        df_device_list.to_csv(os.path.join(self.output_folder, self.base_file_name + '_device_list.csv'), index=False)
        print('Device list saved.')

    def process(self):
        # 一連の処理の実行
        self.load_data()
        self.create_device_list()
        self.compute_device_communication()
        self.build_transition_matrix()
        self.refine_transition_matrix()
        self.convert_to_transition_probability()
        self.save_results()
        print('All processes completed.')

if __name__ == '__main__':
    input_file_path = sys.argv[1]  # ファイルのパス
    input_file_name = sys.argv[2]  # ファイル名
    output_folder = sys.argv[3]    # 保存フォルダのパス
    network_transition = NetworkTransition(input_file_path, input_file_name, output_folder)
    network_transition.process()

#python NetworkTransition.py /path/to/logfiles netflow_day-03.csv /path/to/output
