import plotly.express as px
import pandas as pd

mdl = "10x10_vers_1"

df = pd.read_csv(f"saved_models/model_{mdl}_actor_loss.csv")

# df = df.rolling(500, min_periods=500).mean()
print(df)
fig = px.line(df)
fig.show()

df = pd.read_csv(f"saved_models/model_{mdl}_critic_loss.csv")
# df = df.rolling(500, min_periods=500).mean()
fig = px.line(df)
fig.show()
