import plotly.express as px
import pandas as pd

mdl = "7x7_vers_1"
smothing = 10

df = pd.read_csv(f"saved_models/model_{mdl}_actor_loss.csv")

df = df.rolling(smothing, min_periods=smothing).mean()
print(df)
fig = px.line(df)
fig.show()
