import pandas as pd

df = pd.read_csv("artifacts_reward_v2/logs/episode_metrics.csv")

print("Completed episodes:", len(df))
print("Last episode:", df["episode"].iloc[-1])