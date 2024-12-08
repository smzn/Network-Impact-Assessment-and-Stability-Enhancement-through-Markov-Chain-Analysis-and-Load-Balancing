from mpi4py import MPI
import os
from NetworkTransition import NetworkTransition

def process_files(start_day, end_day, input_folder, output_folder):
    # 開始日と終了日からファイルリストを作成
    file_names = [f"netflow_day-{str(i).zfill(2)}.csv" for i in range(start_day, end_day + 1)]
    
    # 各ファイルを処理
    for file_name in file_names:
        print(f"Processing {file_name}")
        network_transition = NetworkTransition(input_folder, file_name, output_folder)
        network_transition.process()

def main():
    # MPIの初期化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # 各プロセスのランク (ID)
    size = comm.Get_size()  # 全プロセス数

    # 開始日、終了日、フォルダパスの設定
    start_day = 3  # 例: 03日から開始
    end_day = 30   # 例: 30日まで
    input_folder = "../logfile/"  # ファイルのあるフォルダパス
    output_folder = "result"      # アウトプットフォルダのパス

    # プロセス数に応じた分割数を計算
    total_days = end_day - start_day + 1
    days_per_process = total_days // size  # 各プロセスが担当する日数
    remainder = total_days % size  # 日数が割り切れない場合の余り

    # 各プロセスが担当する開始日と終了日を計算
    if rank < remainder:
        # 余りがある場合、最初の `remainder` 個のプロセスは1日多く担当
        local_start_day = start_day + rank * (days_per_process + 1)
        local_end_day = local_start_day + days_per_process
    else:
        # それ以外のプロセスは `days_per_process` の日数を担当
        local_start_day = start_day + rank * days_per_process + remainder
        local_end_day = local_start_day + days_per_process - 1

    print(f"Process {rank} handling from day {local_start_day} to {local_end_day}")

    # 各プロセスが担当するファイルを処理
    process_files(local_start_day, local_end_day, input_folder, output_folder)

if __name__ == "__main__":
    main()

#mpiexec -np 4 python3 NetworkTransition_mpi.py 03 30
