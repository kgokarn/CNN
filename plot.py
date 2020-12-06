import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

df = pd.DataFrame({
  "Adders": ["RCA", "CLA", "ACA", "ESA", "ETA", "LOACLA", "LOARCA", "XCLA","XRCA"],
  "Area": [2, 1, 3, 1, 3, 2, 3, 3, 3],
})

fig = px.bar(df, x="Adders", y="Area", color="Adders")
fig.update_xaxes(title_text="16-bit Adder")
fig.update_yaxes(title_text="Area")
fig.show()