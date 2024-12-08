import pandas as pd
import numpy as np
import time
from mpi4py import MPI


# Set the print options to avoid truncation
#np.set_printoptions(threshold=np.inf)

#「全日程の中でtotal network influenceの平均が最大のデバイス」を引数で渡す形式に変更2024/10/23
class AdjustedTransition:
    def __init__(self, target_device, multi_num):
        self.target_device = target_device# 全日程の中でtotal network influenceの平均が最大のデバイス
        self.multi_num = multi_num    
    
    def generate_new_transition(self, day):
        # CSVファイルを読み込む
        filename = f'../../../NetworkTransition/result/netflow_day-{day:02}_transition_probability.csv' #毎回変更!!!
        df_netflow = pd.read_csv(filename, index_col=0)
                
        #推移確率をnumpy形式に
        np_netflow = df_netflow.values
        
        #デバイス名一覧
        netflow_cols = df_netflow.columns.tolist()
        
        #対象デバイスのindex番号を取得
        target_device_ind = netflow_cols.index(self.target_device)
        
        #新しい推移確率の作成
        np_netflow_new = self.regenerate_transition_matrix(np_netflow, target_device_ind)
        
        #行数と列数の計算をする
        print('shape', np_netflow_new.shape)
        
        #行和と列和を計算する
        row_sums = np.sum(np_netflow_new, axis=1)
        print("Row sums:", row_sums)
        
        #行和が0.99以下か判断する
        if len(row_sums[(row_sums < 0.9999) | (row_sums > 1.0)]):
            print('Row warning!!!!!!')
            # 行和で割ることで正規化
            np_netflow_new = np_netflow_new / row_sums
        
        # 新しいデバイス名を生成（重複を防ぐためユニークな名前を確認）
        additional_device_names = [f'{self.target_device}_{i+1}' for i in range(self.multi_num)]
        new_columns_and_index = netflow_cols + additional_device_names

        # データフレーム作成時にサイズと名前の整合性をチェック
        if np_netflow_new.shape[0] != len(new_columns_and_index) or np_netflow_new.shape[1] != len(new_columns_and_index):
            raise ValueError(
                f"Inconsistent dimensions: Data shape {np_netflow_new.shape}, "
                f"Index/Column size {len(new_columns_and_index)}"
            )

        # pandas DataFrameの作成
        df_netflow_new = pd.DataFrame(
            np_netflow_new,
            columns=new_columns_and_index,
            index=new_columns_and_index
        )

        # 行名と列名の確認
        row_indices = set(df_netflow_new.index)
        column_indices = set(df_netflow_new.columns)
        if row_indices != column_indices:
            print("Warning: Row and column indices do not match.")
            # 差分を確認
            only_in_rows = row_indices - column_indices
            only_in_columns = column_indices - row_indices

            print("Devices only in rows:", only_in_rows)
            print("Devices only in columns:", only_in_columns)

            # 順序の違いを確認
            if row_indices == column_indices:
                print("Row and column indices have the same elements, but the order may differ.")
                print("Last 5 row indices:", df_netflow_new.index[-5:].tolist())
                print("Last 5 column indices:", df_netflow_new.columns[-5:].tolist())
            else:
                print("Row and column indices differ in elements.")

        df_netflow_new.to_csv(f'result_ActiveDirectory_01/netflow_day-{day:02}_transition_probability_01.csv') #毎回変更!!!

        # 行と列の最後5個を表示
        print("Last 5 row indices:", df_netflow_new.index[-5:].tolist())
        print("Last 5 column indices:", df_netflow_new.columns[-5:].tolist())
        
    # 推移確率の再生成と列操作
    def regenerate_transition_matrix(self, matrix, a):#a番目のものを,l個増やす
        """推移確率の再生成と列操作"""
        # a番目の行を self.multi_num 回繰り返して追加
        repeated_rows = np.tile(matrix[a], (self.multi_num, 1))
        new_matrix = np.vstack([matrix, repeated_rows])
        
        # l回繰り返して新しい列を追加
        for _ in range(self.multi_num):
            # 各行のa番目の値を取得し、その半分（l + 1で割る）を新しい列として追加
            new_column = new_matrix[:, a] / (self.multi_num + 1)
            # 新しい列を行列に追加
            new_matrix = np.hstack([new_matrix, new_column.reshape(-1, 1)])
        
        # a列目の値も半分にする
        new_matrix[:, a] /= (self.multi_num+1)

        # 行数と列数を一致させる
        if new_matrix.shape[0] != new_matrix.shape[1]:
            raise ValueError(
                f"行数と列数が一致しません: 行数={new_matrix.shape[0]}, 列数={new_matrix.shape[1]}"
            )
        
        return new_matrix


if __name__ == "__main__":
    
    # MPIの初期化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # 各プロセスのID（ランク）
    size = comm.Get_size()  # 実行中のプロセス数（サイズ）
    
    start_time = time.time()  # 実行開始時刻
    
    # パラメータ設定
    start_day = 3  # 開始日
    end_day = 30  # 終了日
    target_device = 'ActiveDirectory'  # 最も影響力のあるデバイス
    multi_num = 1 #複製数

    # 日付をプロセスごとに分割
    total_days = end_day - start_day + 1
    days_per_process = total_days // size  # 各プロセスが担当する日数
    extra_days = total_days % size  # プロセス数で割り切れない余りの日数

    # 各プロセスに担当する日付範囲を割り当て
    if rank < extra_days:
        day_start = start_day + rank * (days_per_process + 1)
        day_end = day_start + days_per_process
    else:
        day_start = start_day + extra_days + rank * days_per_process
        day_end = day_start + days_per_process - 1

    print(f"プロセス {rank}: {day_start} 日目から {day_end} 日目までを担当")

    # AdjustedTransitionクラスのインスタンスを生成
    adjusted = AdjustedTransition(target_device, multi_num)

    # 各プロセスが割り当てられた日付の推移確率行列を処理
    for day in range(day_start, day_end + 1):
        print(f'プロセス {rank}: {day} 日目を処理中')
        adjusted.generate_new_transition(day)

    # プロセス間で同期を取る
    comm.Barrier()  # 全てのプロセスが終了するまで待機

    # 処理時間を計測
    if rank == 0:  # ルートプロセス（rank 0）が総実行時間を表示
        end_time = time.time()
        time_diff = end_time - start_time
        print(f"総実行時間: {time_diff} 秒")    
