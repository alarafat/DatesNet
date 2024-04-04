import plotly as plty
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def plot_data_distribution(data,label):
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        row_heights=[0.5],
        specs=[[ {"type": "pie"}, {"type": "Funnelarea"}]])
    #plotting Pie Plot

    fig.add_trace(go.Pie(
        labels=data.index,
        values=data["labels"],
        legendgroup="group",
        textinfo='percent+label'),
        row=1, col=1)
    #Plotting Funnel Area
    fig.add_trace(go.Funnelarea(
       values=data['labels'], labels=data.index, name='Emotions data distribution',
        title = {"position": "top center",}),
                        row=1, col=2)

    fig.update_layout(height=500,width=1000, bargap=0.2,
                      margin=dict(b=0,r=20,l=20), xaxis=dict(tickmode='linear'),
                      title_text=f"{label} Data Distribution",
                      template="plotly_white",
                      title_font=dict(size=29, color='#8a8d93', family="Lato, sans-serif"),
                      font=dict(color='#8a8d93'),
                      hoverlabel=dict(bgcolor="#f2f2f2", font_size=13, font_family="Lato, sans-serif"),
                      showlegend=False)
    fig.show()