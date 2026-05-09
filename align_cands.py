import pandas as pd

csv_a_path = "/home/byx/project/op-cad/reward/main_result/qwen3-vl/no-pic/std/cands.csv"
csv_b_path = "/home/byx/project/op-cad/reward/main_result/qwen3-vl/pic/std/cands.csv"

out_a_path = "/home/byx/project/op-cad/reward/main_result/qwen3-vl/no-pic/std/cands.csv"
out_b_path = "/home/byx/project/op-cad/reward/main_result/qwen3-vl/pic/std/cands.csv"

# 读取两个 CSV
df_a = pd.read_csv(csv_a_path)
df_b = pd.read_csv(csv_b_path)

# 确保 group_index 是字符串，避免 00058 这类前缀被错误处理
df_a["group_index"] = df_a["group_index"].astype(str)
df_b["group_index"] = df_b["group_index"].astype(str)

# 取双方都有的 group_index，也就是交集
common_group_index = set(df_a["group_index"]) & set(df_b["group_index"])

print(f"CSV A 原始 group_index 数量: {df_a['group_index'].nunique()}")
print(f"CSV B 原始 group_index 数量: {df_b['group_index'].nunique()}")
print(f"双方共有 group_index 数量: {len(common_group_index)}")

# 各自只保留共同 group_index 的行
df_a_aligned = df_a[df_a["group_index"].isin(common_group_index)].copy()
df_b_aligned = df_b[df_b["group_index"].isin(common_group_index)].copy()

# 可选：按照 group_index 和 k_index 排序
sort_cols = ["group_index"]
if "k_index" in df_a_aligned.columns and "k_index" in df_b_aligned.columns:
    sort_cols.append("k_index")

df_a_aligned = df_a_aligned.sort_values(sort_cols).reset_index(drop=True)
df_b_aligned = df_b_aligned.sort_values(sort_cols).reset_index(drop=True)

# 保存
df_a_aligned.to_csv(out_a_path, index=False)
df_b_aligned.to_csv(out_b_path, index=False)

print(f"CSV A 保留行数: {len(df_a_aligned)}")
print(f"CSV B 保留行数: {len(df_b_aligned)}")
print(f"已保存: {out_a_path}")
print(f"已保存: {out_b_path}")