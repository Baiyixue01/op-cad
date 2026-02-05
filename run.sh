SRC_ROOT=/home/baiyixue/project/op-cad/upload
DST_ROOT=/data/baiyixue/CAD/step_files

for idx_dir in "$SRC_ROOT"/*/; do
  idx_name=$(basename "$idx_dir")
  mkdir -p "$DST_ROOT/$idx_name"
  mv "$idx_dir"/* "$DST_ROOT/$idx_name"/ 2>/dev/null || true
done

rmdir "$idx_dir" 2>/dev/null || true