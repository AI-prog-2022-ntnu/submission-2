import plotly.express as px
import pandas as pd

df = pd.read_csv("saved_models/model_5x5_vers_4_actor_loss.csv")
print(df)
fig = px.line(df)
fig.show()
df = pd.read_csv("saved_models/model_5x5_vers_4_critic_loss.csv")
fig = px.line(df)
fig.show()
