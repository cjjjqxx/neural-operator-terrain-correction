#!/usr/bin/env bash
# Global correction, statistical tests and figures of the paper, in dependency order.
# Prerequisites (see README, "Reproducing the paper"): the third-party inputs below, and from the Zenodo data
# record operator_weights.pt -> fno3d_large_model.pt, fem_training_set.npz -> global_fem_dataset.npz, and the
# contents of intermediate_products.zip, all in this directory.
set -u
export PYTHONIOENCODING=utf-8
cd "$(dirname "$0")"
PY=${PYTHON:-python}
LOG=logs; mkdir -p "$LOG"
export SRTM15_DEM=${SRTM15_DEM:-data/SRTM15_450m.tif}
export IHFC_GHFDB=${IHFC_GHFDB:-data/IHFC_2024_GHFDB_v.2026.03.txt}

missing=0
for f in "$SRTM15_DEM" "$IHFC_GHFDB" fno3d_large_model.pt global_fem_dataset.npz site_terrain.npz \
         ghfdb2024_official/IHFC_2024_GHFDB_05x05_WGS1984_Kriging.csv \
         ggge20271-sup-0001-data_table1_eq_area_global_heat_flow.csv; do
  [ -e "$f" ] || { echo "missing: $f"; missing=1; }
done
[ $missing -eq 0 ] || { echo "See README, 'Reproducing the paper'."; exit 1; }

run() {
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] >>> $name: $*"
  if ! $PY -u "$@" > "$LOG/$name.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] ✗ $name 失败,见 $LOG/$name.log"; tail -5 "$LOG/$name.log"; exit 1
  fi
  echo "[$(date +%H:%M:%S)] ✓ $name"
}
selfcheck() {   # 参数传了≠生效:两个开关都必须在日志里留下可观测的痕迹
  grep -q "推理:8 朝向平均" "$LOG/$1.log" || { echo "✗ $1 缺「推理:8 朝向平均」"; exit 1; }
  grep -q "D_0 取测点所在中心列" "$LOG/$1.log" || { echo "✗ $1 缺 M5 自检行"; exit 1; }
}

[ -z "${SKIP_REBUILD:-}" ] && { run rebuild rebuild_correction.py; selfcheck rebuild; }
run rebuild_topocorr   rebuild_correction.py --keep-topo-corrected; selfcheck rebuild_topocorr
run krige              krige_global_ok.py

for s in continental_budget official_grid_benchmark paired_null_fairness_test paired_null_robust_test \
         magnitude_scaling_test davies_validation davies_terrain_baseline c15_ordinal_test \
         c15_terrain_baseline province_artifact_test quality_stratified_test \
         depth_fallback_sensitivity site_reliability error_budget; do
  run "$s" "$s.py"
done
run gbstats            global_B_field_stats.py   # 0.5° 场统计与格内极差;emit_numbers 与图4c 都读它
run emit1              emit_numbers.py
run scorecard          evaluation_scorecard.py
run emit2              emit_numbers.py
run andes_tile         andes_example_tile.py   # 图3f 左小图:安第斯测点瓦片的算子 B 场,须在 rebuild 之后、绘图之前
for f in fig1_spatial_validation fig4_bfield_comparison fig5_speed_and_global fig6_dataset_correction fig6_correction_magnitude fig7_global_heatflow \
         fig8_correction_vs_relief fig07_observational_tests fig08_valley_bias figED4_depth_spectrum \
         figED5_kriging_uncertainty; do
  run "fig_$f" "figures/v2/$f.py"
done
run emit3              emit_numbers.py   # fig5_speed_and_global、figED5_kriging_uncertainty 另写盈亏平衡点与支撑距离,须在绘图后再汇总一次
echo "[$(date +%H:%M:%S)] 全部完成。"
