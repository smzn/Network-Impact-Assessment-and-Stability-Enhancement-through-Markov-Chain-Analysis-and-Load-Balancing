
# Network Impact Assessment and Stability Enhancement through Markov Chain Analysis and Load Balancing

## Overview
This repository contains Python scripts designed for analyzing network stability and optimizing load balancing using discrete Markov chains. The models identify critical network devices and provide strategies for enhancing stability through replication and transition probability adjustments.

### Project Structure
1. **Transition Probability Calculation**
   - `NetworkTransition.py`
   - `NetworkTransition_main.py`
2. **Equivalence Classes, Centrality, and Network Impact Assessment**
   - `EquivalenceCalculation_MPI_v5.py`
   - `EquivalenceCalculation_MPI_main_v6.py`
   - `NetworkEvalulate_v4.py`
3. **Device Replication and Transition Probability Adjustment**
   - `AdjustedTransition_MPI_v2.py`

### Dataset
This project uses the [Unified Host and Network Data Set](https://csr.lanl.gov/data/2017/):
- M. Turcotte, A. Kent, and C. Hash, “Unified Host and Network Data Set,” in *Data Science for Cyber-Security*, November 2018, pp. 1-22.

### Prerequisites
- **Programming Language**: Python 3.x
- **Libraries**: NumPy, Pandas, MPI for Python (mpi4py), Matplotlib
- **Dataset**: Approximately 7GB per day of network log data

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
#### 1. Transition Probability Calculation
Run:
```bash
mpiexec -np <number_of_processes> python NetworkTransition_main.py
```

#### 2. Equivalence Classes and Network Evaluation
Run:
```bash
mpiexec -np <number_of_processes> python EquivalenceCalculation_MPI_main_v6.py
```

#### 3. Device Replication and Transition Probability Adjustment
Run:
```bash
mpiexec -np <number_of_processes> python AdjustedTransition_MPI_v2.py
```

### Output
- Transition matrices, equivalence classes, and network influence metrics
- Graphs for centrality and influence trends over time
- Optimized configurations for load balancing

### License
This project is licensed under the MIT License. See the LICENSE file for more information.

---

# マルコフ連鎖解析と負荷分散を用いたネットワーク影響度評価と安定性向上

## 概要
このリポジトリには、離散マルコフ連鎖を用いてネットワークの安定性を分析し、負荷分散を最適化するPythonスクリプトが含まれています。重要なデバイスを特定し、それらの複製と推移確率の調整を通じて安定性を向上させる戦略を提供します。

### プロジェクト構成
1. **推移確率の計算**
   - `NetworkTransition.py`
   - `NetworkTransition_main.py`
2. **同値類、ネットワーク中心性、ネットワーク影響度評価**
   - `EquivalenceCalculation_MPI_v5.py`
   - `EquivalenceCalculation_MPI_main_v6.py`
   - `NetworkEvalulate_v4.py`
3. **デバイス複製と推移確率の再計算**
   - `AdjustedTransition_MPI_v2.py`

### データセット
このプロジェクトは [Unified Host and Network Data Set](https://csr.lanl.gov/data/2017/) を使用しています：
- M. Turcotte, A. Kent, and C. Hash, “Unified Host and Network Data Set,” in *Data Science for Cyber-Security*, November 2018, pp. 1-22.

### 必要条件
- **プログラミング言語**: Python 3.x
- **ライブラリ**: NumPy, Pandas, MPI for Python (mpi4py), Matplotlib
- **データセット**: ネットワークログデータ（約7GB/日）

### インストール
1. リポジトリをクローンします：
   ```bash
   git clone <repository-url>
   ```
2. 必要なPythonライブラリをインストールします：
   ```bash
   pip install -r requirements.txt
   ```

### 使用方法
#### 1. 推移確率の計算
以下を実行：
```bash
mpiexec -np <プロセス数> python NetworkTransition_main.py
```

#### 2. 同値類とネットワーク評価
以下を実行：
```bash
mpiexec -np <プロセス数> python EquivalenceCalculation_MPI_main_v6.py
```

#### 3. デバイス複製と推移確率の再計算
以下を実行：
```bash
mpiexec -np <プロセス数> python AdjustedTransition_MPI_v2.py
```

### 出力
- 推移行列、同値類、ネットワーク影響度メトリクス
- 時系列での中心性や影響度のグラフ
- 負荷分散の最適化構成

### ライセンス
このプロジェクトはMITライセンスの下で公開されています。詳細についてはLICENSEファイルをご覧ください。
