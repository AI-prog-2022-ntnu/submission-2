import plotly.express as px
import pandas as pd

df = pd.read_csv("saved_models/model_5x5_vers_5_actor_loss.csv")

# df = df.rolling(500, min_periods=500).mean()
print(df)
fig = px.line(df)
fig.show()

df = pd.read_csv("saved_models/model_5x5_vers_5_critic_loss.csv")
fig = px.line(df)
fig.show()
