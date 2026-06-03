import pandas as pd

# 读取实验结果
df = pd.read_csv("out_cnn_lstm_grid_search_3/grid_search_results_partial.csv")

# 定义综合评分（越小越好）
df["score"] = df["NRMSE"] + (1 - df["R2"]) * 100 + df["MASE"]

# 找 Top-10
top10 = df.sort_values("score").head(10)
print("Top 10 模型:")
print(top10[["run_id", "MODEL_TYPE", "SEQ_LEN", "LR", "BATCH_SIZE", "LOSS_TYPE", "MAE", "RMSE", "R2", "MAPE(%)", "sMAPE(%)", "MASE", "score"]])

top10_1 = df.sort_values("sMAPE(%)").head(10)
print("\nTop 10 sMAPE:")
print(top10_1[["run_id", "MODEL_TYPE", "SEQ_LEN", "LR", "BATCH_SIZE", "LOSS_TYPE", "MAE", "RMSE", "R2", "MAPE(%)", "sMAPE(%)", "MASE", "score"]])

# 按模型类型平均表现
print("\n模型类型平均表现:")
print(df.groupby("MODEL_TYPE")[["MAE", "RMSE", "R2", "MAPE(%)", "sMAPE(%)", "MASE", "score"]].mean().sort_values("score"))

# 按 SEQ_LEN 平均表现
print("\nSEQ_LEN 平均表现:")
print(df.groupby("SEQ_LEN")[["MAE", "RMSE", "R2", "MASE", "score"]].mean().sort_values("score"))

print("\nLR 平均表现:")
print(df.groupby("LR")[["MAE", "RMSE","MAPE(%)", "sMAPE(%)", "R2", "MASE", "score"]].mean().sort_values("score"))

print("\nBATCH_SIZE 平均表现:")
print(df.groupby("BATCH_SIZE")[["MAE", "RMSE","MAPE(%)", "sMAPE(%)", "R2", "MASE", "score"]].mean().sort_values("score"))

print("\n LOSS_TYPE 平均表现:")
print(df.groupby("LOSS_TYPE")[["MAE", "RMSE","MAPE(%)", "sMAPE(%)","R2", "MASE", "score"]].mean().sort_values("score"))

# print("\n data tricks 平均表现:")
# print(df.groupby("OVERSAMPLE_PEAKS")[["MAE", "RMSE","MAPE(%)", "sMAPE(%)", "R2", "MASE", "score"]].mean().sort_values("score"))

print("\n weight decay 平均表现:")
# print(df.groupby("WEIGHT_DECAY")[["MAE", "RMSE","MAPE(%)", "sMAPE(%)", "R2", "MASE", "score"]].mean().sort_values("score"))
print(df.groupby("DROPOUT")[["MAE", "RMSE","MAPE(%)", "sMAPE(%)", "R2", "MASE", "score"]].mean().sort_values("score"))

print("\n loss config表现:")
# print(df.groupby("USE_HET")[["MAE", "RMSE","MAPE(%)", "sMAPE(%)", "R2", "MASE", "score"]].mean().sort_values("score"))
# print(df.groupby("USE_ASYM")[["MAE", "RMSE","MAPE(%)", "sMAPE(%)", "R2", "MASE", "score"]].mean().sort_values("score"))
print(df.groupby("ASYM_W")[["MAE", "RMSE","MAPE(%)", "sMAPE(%)", "R2", "MASE", "score"]].mean().sort_values("score"))