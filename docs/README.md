# ドキュメント案内

このディレクトリには、現行の利用・運用に必要な仕様と、結果の根拠を追跡するための研究・履歴資料を置いています。公開 API とベンチマーク CSV の互換性は維持しますが、過去の手法は新規実験の既定値ではありません。

## 現行の利用・運用

- [resource_scaling_benchmarks.md](resource_scaling_benchmarks.md): `BenchmarkConfig`、JSON、CLI、CSV schema 2.0、プロット API。
- [resource_grid.md](resource_grid.md): `hamiltonian-resource-grid` のプリセット、再開、出力検証。
- [error_semantics.md](error_semantics.md): 誤差見積り、観測値、保証範囲の区別。
- [empirical_error_estimation.md](empirical_error_estimation.md): 実測キャリブレーションの利用条件。

ルートの [`benchmark_config.json`](../benchmark_config.json) は `hamiltonian-benchmark` の既定入力であり、現行の `new` MPF スケジュールを使う標準例です。

## 数理的な参照資料

- [suzuki_error_bounds.md](suzuki_error_bounds.md): Suzuki 公式の交換子上界。
- [mpf_error_bounds.md](mpf_error_bounds.md): MPF の現行の誤差方針、保証範囲、互換識別子。
- [refined_mizuta_bch.md](refined_mizuta_bch.md) と [mizuta_finite_size_constants.md](mizuta_finite_size_constants.md): Mizuta 境界の導出と定数監査。
- [mpf_exponent_sum_scaling.md](mpf_exponent_sum_scaling.md): MPF 指数和のスケーリング。

## 研究・履歴の再現

- [empirical_error_prestudy_trotter_mpf.md](empirical_error_prestudy_trotter_mpf.md): 実測誤差モデルを導入する前の予備調査。現行のサイズ指定には使いません。
- `schedule="legacy"`、`legacy-w2-proxy`、`mizuta2026-theorem3-legacy-ideal-rigorous` は、比較または過去結果の再現のために残しています。新規の標準比較には `schedule="new"` と現行の明示的な誤差方針を使用してください。
- 追跡済みの `result/` は研究成果物です。ランナーの一時出力先ではなく、削除や再生成の対象にも含めません。
