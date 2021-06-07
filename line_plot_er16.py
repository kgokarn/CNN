import pandas as pd
import plotly.graph_objects as go


hd = ['0', '1', '2', '3', '4', '6']

loarca_x =[0,85.21,87.99,94.64,98.26,99.70]
loacla_x =[0,71.49,90.53,95.54,97.13,98.91]
xrca_x =[0,69.67,94.35,95.83,97.82,99.36]
xcla_x =[0,64.65,85.63,91.60,95.99,98.92]
loarca_a=[0,39.36,63.60,78.64,87.03,95.32]
loacla_a=[0,37.31,57.68,70.63,79.81,89.23]
xrca_a= [0,29.69,50.96,66.24,76.82,86.73]
xcla_a= [0,30.81,50.13,65.08,73.60,84.50]

fig = go.Figure()

fig.add_trace(go.Scatter(x=hd, y=loarca_x, name='LOARCA_X',
                         line=dict(color='rgb(255,0,0)', width=1.5)))
fig.add_trace(go.Scatter(x=hd, y=loacla_x, name = 'LOACLA_X',
                         line=dict(color='rgb(1,147,64)', width=1.5)))
fig.add_trace(go.Scatter(x=hd, y=xrca_x, name='XRCA_X',
                         line=dict(color='rgb(5,83,240)', width=1.5)))
fig.add_trace(go.Scatter(x=hd, y=xcla_x, name='XCLA_X',
                         line=dict(color='rgb(242,178,2)', width=1.5)))

fig.add_trace(go.Scatter(x=hd, y=loarca_a, name='LOARCA_A',
                         line = dict(color='rgb(255,0,0)', width=1.5, dash='dash')))
fig.add_trace(go.Scatter(x=hd, y=loacla_a, name='LOACLA_A',
                         line = dict(color='rgb(1,147,64)', width=1.5, dash='dash')))
fig.add_trace(go.Scatter(x=hd, y=xrca_a, name='XRCA_A',
                         line=dict(color='rgb(5,83,240)', width=1.5, dash='dash')))
fig.add_trace(go.Scatter(x=hd, y=xcla_a, name='XCLA_A',
                         line=dict(color='rgb(242,178,2)', width=1.5, dash='dash')))

fig.update_layout(title='ER16 behavior on ER optimized adders',
                  title_font = {"size":17},
                   xaxis_title='Hamming Distance',
                   yaxis_title='16-bit ER optimized adders',
                  paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)'
                  )
fig.update_layout(autosize=False, width=600, height=500)

fig.update_xaxes(title_font = {"size":17}, linecolor = 'rgba(0,0,0,1.0)', type = 'category')
fig.update_yaxes(title_font = {"size":17}, linecolor = 'rgba(0,0,0,1.0)')

fig.update_layout(legend=dict(
    yanchor="bottom",
    y=0.01,
    xanchor="right",
    x=0.99
))

#fig.show()

fig.write_image("er16.png")