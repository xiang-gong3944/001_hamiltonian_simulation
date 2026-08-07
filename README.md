# Hamiltonian simulation resource comparison

Pauli 和で与えたハミルトニアンについて、次の回路レベルの比較を行う Python パッケージです。

- Lie–Trotter / 高次 Suzuki 公式
- Pauli-LCU block encoding と QSVT 射影位相列
- well-conditioned multiproduct formula (MPF) の coherent LCU
- 小規模系の statevector 計算と厳密行列指数との比較
- 大規模系の、許容誤差を固定した T / CNOT / 補助量子ビット数の比較

回路構成関数はハミルトニアンの密行列を作りません。密行列を使うのは、小規模検証の厳密解だけです。

## セットアップ（VS Code）

Python 3.10 以上と、VS Code の Python・Jupyter 拡張を用意します。スクリプトは `.venv` の作成と依存関係の導入だけを行い、ブラウザ版 Jupyter は起動しません。

### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_venv.ps1
```

`python` 以外のコマンド名を使う場合は、たとえば次のように指定できます。

```powershell
.\setup_venv.ps1 -PythonCommand py
```

### macOS / Linux

```bash
bash setup_venv.sh
```

特定の Python を使う場合:

```bash
PYTHON=python3.12 bash setup_venv.sh
```

セットアップ後、VS Code で `notebooks/resource_comparison.ipynb` を直接開きます。カーネルが自動選択されない場合は、Notebook 右上のカーネル選択から次を指定してください。

```text
Windows: .venv\Scripts\python.exe
macOS/Linux: .venv/bin/python
```

`.vscode/settings.json` が workspace 内の `.venv` を検索対象として指定します。ユーザー領域への Jupyter kernelspec 登録は行わないため、環境はプロジェクト内で完結します。

## 構造

```text
src/hamiltonian_resources/
  hamiltonians.py    # Pauli 和入力、TFIM、Heisenberg 鎖
  trotter.py         # 積公式回路
  circuit_utils.py   # PREPARE / SELECT / block encoding / robust OAA
  qsvt.py            # Jacobi--Anger 位相合成、QSVT、LCU、robust OAA
  multiproduct.py    # MPF 係数、segment LCU、robust OAA
  method_specs.py    # backend 非依存の手法指定
  planning.py        # 単一回のparameter選択とimmutable algorithm plan
  analytical.py      # planを消費する構造化解析resource backend
  evaluation.py      # 単一点API、report、reference Qiskit dispatch
  simulation.py      # 小規模な厳密解との比較
  resources.py       # transpile 後の CX と T 見積り
  benchmark.py       # 旧parameter/resource APIの互換wrapper
  benchmark_suite.py # 8構成の解析スイープ、CSV、再現性metadata
  benchmark_plotting.py # 保存済みCSVから一貫した図を生成
notebooks/
  resource_comparison.ipynb
  qsvt_validation.ipynb
  mpf_validation.ipynb
tests/
docs/
  error_semantics.md # sizing、数学claim、empirical observationの区別
  mpf_error_bounds.md # ideal MPF、OAA、shared-ancilla good-block上界
  resource_scaling_benchmarks.md # スイープ設定、schema、実行方法
  suzuki_error_bounds.md  # 積公式の分割、厳密誤差上界、fallback
```

## notebook-first解析スケーリングbenchmark

既定設定は開放境界TFIM `J=1, h=3`について、Trotter `p=1,2,4,6`、
MPF `m=3,5,7`、QSVTを独立に評価します。system-size sweepでは局所相互作用が
系全体へ広がる時間を比較するため、`t(n)=n`を使います。この比例則は伝播速度の
厳密な推定ではなく、明記された無次元の比較規約です。

`notebooks/resource_comparison.ipynb`では、NumPy配列を含むパラメータセルを変更し、
計算とlog10サイズ軸での描画までnotebook内で完結できます。benchmark APIは結果を
メモリ上のDataFrameとして返し、自動保存しません。

```powershell
hamiltonian-benchmark generate --config benchmark_config.json --sweep all --progress
```

生成とplotを続けて行う場合:

```powershell
hamiltonian-benchmark run --config benchmark_config.json
```

各CLI実行は`benchmark_outputs/<timestamp>_<digest>_<run-id>/`を新規作成するため、
以前の結果を上書きしません。単体CSVはsidecarなしで再描画できます。

```powershell
hamiltonian-benchmark plot --data benchmark_outputs/<run>/benchmark.csv --summary
```

設定、schema、failure row、解析上の仮定は
[`docs/resource_scaling_benchmarks.md`](docs/resource_scaling_benchmarks.md)に記載しています。

## 単一点resource API

1つのHamiltonian・手法・時刻・目標誤差を評価するためにsweepやDataFrameは不要です。
`estimate_resources`はparameterを1回だけ選択し、immutable planと構造化解析backendの
resource結果を返します。同じplanをreference Qiskit回路と小規模検証へ渡せます。

```python
from hamiltonian_resources import (
    MultiproductMethod,
    build_simulation_circuit,
    compare_plan_with_exact,
    estimate_resources,
    transverse_field_ising,
)

H = transverse_field_ising(2, field=0.7)
report = estimate_resources(H, MultiproductMethod(3), time=0.2, target_error=1e-3)

print(report.selected_parameters)
print(report.logical_counts.as_dict())
print(report.resources.t_count, report.resources.cnot_count)
print(report.resource_provenance.as_dict())
print(report.parameter_selection_succeeded)
print(report.ideal_algorithm_target_certified)
print(report.implemented_circuit_target_certified)

circuit = build_simulation_circuit(report.plan)  # 小規模なreference構成向け
check = compare_plan_with_exact(report.plan)
report_with_observations = report.with_observations(check.observations)
```

planは論理構造だけを保持します。temporary-AND、分解後の回転/CNOT、backend work qubitは
解析結果またはQiskit metadata側に属します。MPF/QSVTの構造化解析backendとgeneric
Qiskit `.control()` 回路は同じplanを使いますが、異なるcompilation仮定を明記します。

## benchmark最小例

```python
import numpy as np
from hamiltonian_resources import BenchmarkConfig, QSVTMethod, run_benchmark

config = BenchmarkConfig(
    system_sizes=np.arange(2, 21, 2),
    target_errors=np.logspace(-1, -3, 5),
    methods=[QSVTMethod()],
)
table = run_benchmark(config, workers=1, show_progress=True)
print(table[["sweep", "system_qubits", "evolution_time", "method_label", "t_count"]])
```

4次・6次 Suzuki 公式では、可換な Pauli 項を同じgroupにまとめた回路と、
その回路に対応する交換子上界を同時に使用できます。

```python
from hamiltonian_resources import estimate_suzuki_error, transverse_field_ising

H = transverse_field_ising(6, field=0.7)
estimate = estimate_suzuki_error(H, time=0.5, reps=4, order=4)
print(estimate.error, estimate.rigorous, estimate.method)
```

`auto` は1次・2次では従来どおり個別Pauli項を使い、4次以上では完全可換groupを
使います。回路、誤差評価、解析リソース式は同じgroup順序を共有します。詳細は
[`docs/suzuki_error_bounds.md`](docs/suzuki_error_bounds.md)を参照してください。

QSVT回路は時刻と回路近似誤差を直接指定します。既定では、cos/sinの
coherent LCUに続けて1回のrobust oblivious amplitude amplificationを行います。

```python
from hamiltonian_resources import (
    build_hamiltonian_qsvt_circuit,
    compare_with_exact,
    transverse_field_ising,
)

H = transverse_field_ising(2)
circuit = build_hamiltonian_qsvt_circuit(H, time=0.2, epsilon=1e-3)
check = compare_with_exact(H, 0.2, method="qsvt", qsvt_epsilon=1e-3)
print(check)
```

`amplitude_amplification=False`では、all-zero ancilla blockは概ね
`scale * exp(-i H t) / 2`で、成功確率は`scale**2 / 4`です。増幅後は同じblockが
`exp(-i H t)`に近づきます。

`run_benchmark`は大規模向けの明示的な解析分解コストモデルを使い、具体回路や密行列を構築しません。

MPF では LCU の項数 `m` を指定すると、登録済みの well-conditioned な
Trotter 分割数が自動的に選ばれます。対応範囲は `m=2` から `m=15` です。
既定の`schedule="new"`は3-step OAA込みのquery数を抑える表で、
`schedule="legacy"`を指定すると以前の1-normが小さい表を使用できます。
時間を `segments` 個に分け、各segment内で
`sum_j a_j S_2(step_time / k_j)**k_j` をcoherent LCUとして作ります。
係数1-normは正負の相殺identity branchで2へpaddingされ、既定では各segmentを
1回の3-step robust OAAで増幅してから同じbranch register上で反復します。

```python
from hamiltonian_resources import (
    build_multiproduct_circuit, optimal_mpf_exponents, transverse_field_ising,
)

print(optimal_mpf_exponents(3))                    # (1, 2, 4)
print(optimal_mpf_exponents(3, schedule="legacy")) # (1, 2, 6)
H = transverse_field_ising(2)
mpf = build_multiproduct_circuit(H, time=0.2, m=3, segments=2)
```

`amplitude_amplification=False`は単一segmentのLCU検証専用です。このとき
zero-branch blockはMPF stepのちょうど1/2になります。

## 比較上の重要な前提

1. **誤差予算**: アルゴリズム誤差と単一量子ビット回転合成誤差に分けます。`synthesis_error_fraction` で後者の割合を指定します。
2. **T 数**: 任意角 `Rz` は無料ではありません。各非 Clifford 回転に均等に精度を配り、`3 log2(1/epsilon_rot) + log2 log2(1/epsilon_rot)` 型の ancilla-free 合成コストで見積もります。解析モデルでは多重制御ゲートの Toffoli 相当コストも temporary-AND（1 対あたり 4 T）で `t_count` に計上し、`toffoli_count` 列に対数を保存します。
3. **固定誤差の次数選択**: `plan_simulation` は次数1・2にChilds–Su–Tran–Wiebe–Zhuの厳密上界 `W1 t^2 / r`、`W2 t^3 / r^2`を使い、次数4・6にはSchubert–Mendl Theorem 1の明示的な高次交換子上界 `Wp t^(p+1) / r^p`を使います。互換用の`choose_parameters`も同じplanning pathを使用します。4次は可換group数5以下、6次は3以下で厳密評価し、この制限を超える場合と8次以上では従来の`alpha^(p+1)` proxyにfallbackします。MPFは既定で`low2019-l1-ideal-rigorous`を使い、Low–Kliuchnikov–Wiebe Eq. (16)を上側bracketとしてEq. (14)–(15)を満たす最小segment数を二分探索します。Mizuta Theorem 4の有限次数交換子上界をexact Pauli algebraで評価する`mizuta2026-commutator-ideal-rigorous`は明示指定できます。Aftab 2024の任意次数条件を有限打切りで厳密と呼ばない理由を含む詳細は[`docs/mpf_error_bounds.md`](docs/mpf_error_bounds.md)に記載しています。以前の`alpha_eff = min(alpha, W2^(1/3))`規則はopt-inの`legacy-w2-proxy`としてのみ残り、常に非厳密と表示されます。
4. **QSVT**: `exp(-iHt)=cos(Ht)-i sin(Ht)` の偶・奇成分は、Jacobi--Anger展開から同じscaleで生成します。厳密claimはscale誤差・cos/sin tail・ideal cubic OAAまでを対象とします。`pyqsp`のfloating phase residualは2049点gridの観測値であり、構築回路全体の一様保証には昇格しません。`sym_qsp`の目標多項式はWx応答の虚部に現れるため、各成分は`V`と`V^dagger`のLCUで実blockとして抽出します。生の位相配列を受け取る公開APIはありません。
5. **MPF**: 各segmentの増幅前zero-branch blockを`B=M/2`とすると、3-step OAA後のblockは厳密に`3B - 4 B B^dagger B`です。同じbranch register上で増幅step unitaryを反復するため、最終blockを単純な`M**segments`や`(PWP)**segments`と同一視しません。Low/Mizutaの`ideal-mpf` claimとは別に、unitarity-defect envelopeとGilyén--Su--Low--Wiebeのreused-ancilla block-encoding積を使い、`P W**segments P`に保守的な厳密上界を付けます。この上界がtargetを満たすかはideal claimとは独立です。metadataにはphysical/padding/unused branch数、負係数数、PREPARE/SELECT/reflection数も保存します。
6. **大規模モデル**: 多重制御ゲートの CNOT 数はアーキテクチャ、clean/dirty ancilla、コンパイラで変わります。この実装の解析値は比較用の明記された分解モデルです。
7. **解析コストと具体回路の対応**: 既定の解析式は具体回路の構造（QSVT の quadrature 抽出、cos/sin LCU、3-step OAA、MPF の identity padding・branch 幅・segment ごとの OAA factor 3）を反映します。したがって 3 手法とも「決定的動作あたり」の比較です。ただし controlled 応答回路については、`V` と `V^dagger` が block-encoding query を共有し projector 位相だけを選択する効率的コンパイルを仮定します。`transpile_circuits=True` は Qiskit の汎用 `.control()` 分解を使うため、これよりかなり大きな数値になります（校正時はこの差に注意）。

## テスト

```powershell
pytest
ruff check src tests
```

## 参考文献

- G. H. Low, V. Kliuchnikov, N. Wiebe, [Well-conditioned multiproduct Hamiltonian simulation](https://arxiv.org/abs/1907.11679)
- J. Aftab, D. An, K. Trivisa, [Multi-product Hamiltonian simulation with explicit commutator scaling](https://arxiv.org/abs/2403.08922)
- K. Mizuta, [On the commutator scaling in Hamiltonian simulation with multi-product formulas](https://quantum-journal.org/papers/q-2026-01-19-1974/)
- A. M. Childs et al., [Theory of Trotter Error with Commutator Scaling](https://arxiv.org/abs/1912.08854)
- A. Schubert, C. B. Mendl, [Trotter error with commutator scaling for the Fermi-Hubbard model](https://arxiv.org/abs/2306.10603)
- A. Gilyén et al., [Quantum singular value transformation and beyond](https://arxiv.org/abs/1806.01838)
- [Qiskit `PauliEvolutionGate`](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.PauliEvolutionGate)
- [pyqsp](https://pypi.org/project/pyqsp/)（QSP 位相合成）
