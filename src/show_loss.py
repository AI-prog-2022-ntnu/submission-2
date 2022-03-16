import plotly.express as px
import pandas as pd

mdl = "10x10_vers_1"
smothing = 500

df = pd.read_csv(f"saved_models/model_{mdl}_actor_loss.csv")

df = df.rolling(smothing, min_periods=smothing).mean()
print(df)
fig = px.line(df)
fig.show()

df = pd.read_csv(f"saved_models/model_{mdl}_critic_loss.csv")
df = df.rolling(smothing, min_periods=smothing).mean()
fig = px.line(df)
fig.show()
