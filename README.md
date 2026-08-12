# Hamiltonian simulation resource comparison

Pauli 項で与えたハミルトニアンを対象に、量子ハミルトニアンシミュレーションの回路レベル資源を比較する Python パッケージです。Lie--Trotter / Suzuki 公式、Pauli-LCU を使う QSVT、well-conditioned multiproduct formula (MPF) を扱い、T 数、CNOT 数、補助量子ビット数を見積もります。

回路構成と資源見積りは密行列を作りません。密行列を使うのは、小規模系で厳密解と比較する検証だけです。

## セットアップ

Python 3.10 以上が必要です。セットアップスクリプトはローカルの `.venv` を作成し、依存関係を導入します。

### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_venv.ps1
```

`python` 以外を使う場合は、たとえば `./setup_venv.ps1 -PythonCommand py` を指定します。

### macOS / Linux

```bash
bash setup_venv.sh
```

別の Python を指定する場合は `PYTHON=python3.12 bash setup_venv.sh` を使います。

## 最小の Python 例

```python
from hamiltonian_resources import MultiproductMethod, estimate_resources, transverse_field_ising

hamiltonian = transverse_field_ising(2, field=0.7)
report = estimate_resources(
    hamiltonian,
    MultiproductMethod(3),
    time=0.2,
    target_error=1e-3,
)

print(report.resources.t_count, report.resources.cnot_count)
print(report.selected_parameters)
```

`estimate_resources` は選択したパラメータ、論理的な計画、解析的な資源見積りを返します。小規模の回路検証には `build_simulation_circuit(report.plan)` と `compare_plan_with_exact(report.plan)` を使用します。

## 標準ベンチマーク

ルートの [`benchmark_config.json`](benchmark_config.json) は、CLI が既定で読む動作する標準比較設定です。現在の `new` MPF スケジュールを使い、Trotter、MPF、QSVT を比較します。

```powershell
hamiltonian-benchmark generate --config benchmark_config.json --sweep all --progress
hamiltonian-benchmark plot --data benchmark_outputs/<run>/benchmark.csv --summary
```

`run` はデータ生成と標準プロットをまとめて実行します。出力は `benchmark_outputs/` に新しい実行ディレクトリとして保存され、既存の結果を上書きしません。

過去の比較を再現するための `schedule="legacy"`、`legacy-w2-proxy`、旧 schema-2 CSV の読み込みは維持されています。ただし、これらは新しい実験の既定値ではありません。用途と制約は下記の資料を確認してください。

## ドキュメント

ドキュメントの入口は [`docs/README.md`](docs/README.md) です。

- 現行のベンチマーク API・JSON・CSV schema: [`docs/resource_scaling_benchmarks.md`](docs/resource_scaling_benchmarks.md)
- 大規模 resource-grid の実行・再開: [`docs/resource_grid.md`](docs/resource_grid.md)
- 誤差の意味と保証範囲: [`docs/error_semantics.md`](docs/error_semantics.md)
- Suzuki と MPF の評価式: [`docs/suzuki_error_bounds.md`](docs/suzuki_error_bounds.md)、[`docs/mpf_error_bounds.md`](docs/mpf_error_bounds.md)

## 開発

通常のテストはすべて実行されます。

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Linux/macOS では `.venv/bin/python -m pytest` を使用します。
