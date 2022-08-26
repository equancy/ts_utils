import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

def id_time_coverage(data, y, time_var, args=dict()):
    """Displays the time coverage for each ID.

    Parameters
    ----------
    data (pd.DataFrame): Table with at least one identifier and one time variable.
    y (list): List representing the identifying variable. Can be a combination of
    multiple ID.
    time_var (string): Name of the time variable.
    args (dict): Other arguments to the px.scatter function.

    Returns
    -------
    plotly.graph_objs._figure.Figure
    """
    data = (data
            .assign(id = lambda d: d[y].agg(' - '.join, axis=1))
            .groupby(['id', time_var], as_index=False)
            .sum())

    fig = px.scatter(data, y='id', x=time_var, **args, height=100+25*len(data['id'].unique()))
    fig.update_yaxes(categoryorder='array', categoryarray= data['id'].unique())

    return fig

def id_importance(data, id_vars, cumul_var, type='tab'):
    """Returns the cumulative importance table or graph of the identifiers.

    Parameters
    ----------
    data (pd.DataFrame): Table with at least one identifier and one variable to
    be aggregated.
    id_vars(list): List representing the identifying variable. Can be a
    combination of multiple ID.
    cumul_var (string): Name of the variable to be aggregated.
    type (string): Either 'tab' or 'graph'.

    Returns
    -------
    If type == 'tab', returns a pd.DataFrame
    If type == "graph", returns a plotly.graph_objs._figure.Figure
    """
    tab = (
        data
        .groupby(id_vars, as_index=False)
        .agg(sum=(cumul_var, 'sum'))
        .assign(pct=lambda d: (d['sum']/sum(d['sum'])).round(8))
        .sort_values('pct', ascending=False, ignore_index=True)
        .assign(cumsum_pct=lambda d: d['pct'].cumsum())
        .rename(columns={'cumsum_pct': 'cumulative sum' + cumul_var})
        )

    if type == "graph":
        tab = (tab
               .reset_index()
               .rename(columns={'index': 'Combination' + str(id_vars)}))
        return px.line(tab, y='cumulative sum' + cumul_var,
                       x='Combination' + str(id_vars))
    else:
        return tab

def id_cross_importance(data, id1, id2, weight_var, x_var='pct', title=''):
    """Displays the distribution of crosses between different identifying
    variables.

    Parameters
    ----------
    data (pd.DataFrame): Table with at least two identifiers and one variable
    to be aggregated.
    id1(list): List representing the first indentifying variable. Can be a
    combination of multiple ID.
    id(list): List representing the second indentifying variable. Can be a
    combination of multiple ID.
    weight_var (string): Name of the variable to be aggregated.
    x_var (string): Either 'pct' or 'val'. Allow to display the distribution in
    ('pct') percentage or in ('val') value.
    title (string): Title of the graph.

    Returns
    -------
    plotly.graph_objs._figure.Figure
    """

    tab = (data
           .groupby(id1 + id2, as_index=False)
           .agg(val=(weight_var, 'sum'))
           .assign(id1=lambda d: d[id1].agg(' - '.join, axis=1),
                   id2=lambda d: d[id2].agg(' - '.join, axis=1))
           )

    order = (tab
              .groupby('id1').agg(total=('val', 'sum'))
              .sort_values('total', ascending=True).index.tolist())

    tab = (
        tab
        .merge(tab
               .groupby('id1', as_index=False)
               .agg(total=('val', 'sum')), on='id1')
        .assign(pct=lambda d: d['val']/d['total'])
        .sort_values(['id1', 'pct'], ascending=False, ignore_index=True)
        .groupby('id1', as_index=False)
        .apply(lambda d: d.reset_index(drop=True)).reset_index()
        .assign(top=lambda d: d['level_1'].astype('category'))
        .drop(['level_0', 'level_1'], axis=1)
        .assign(pct_str=lambda d: (100*d['pct']).round(1).astype('str') + '%')
        )

    fig = px.bar(tab, y='id1',  x=x_var,
                 color='top', text='id2',
                 title=title,
                 height=100+40*len(order),
                 custom_data=['id2', 'pct_str', 'top'])

    fig.update_traces(hovertemplate="<br>".join([str(id2)+": %{customdata[0]}",
                                                 "pct: %{customdata[1]}",
                                                 "top: %{customdata[2]}"]))

    fig.update_yaxes(categoryorder='array', categoryarray=order)

    return fig

def ts_lag(data, id_vars, time_var, lagged_vars, period, n_period):
    """Create a table with the lags of one or multiple variable by ID. No need
    of a complete dataset.

    Parameters
    ----------
    data (pd.DataFrame): Table with at least one identifier, one time variable
    and the variable to be lagged. Must be sorted.
    id_vars(list): List representing by which variables we want to compute the
    lags.
    time_var(string): Time variable.
    period (string): Type of period, such as 'D', 'W'.
    n_period (int): Number of periods.

    Returns
    -------
    pd.DataFrame
    """

    period_str = str(n_period) + period

    dict_rename = {time_var: 'to_join'}
    for var in lagged_vars:
        dict_rename[var] = f'lag_{period_str}_{var}'

    df_lag = (
        data[id_vars + [time_var] + lagged_vars]
        .rename(columns=dict_rename)
        .assign(to_join=lambda d: d['to_join'] +
                pd.to_timedelta(n_period, period))
    )

    return df_lag

def ts_visualisation(data, list_id, group, x_var, y_var, col_var,
                     threshold_train=None, weekdays=False, scatter=False):
    """Display one or multiple time series with an ID selector

    Parameters
    ----------
    data (pd.DataFrame): Table with at least one identifier, one time variable
    and the variable to display.
    list_id(list): List of ID to be included in the selector.
    group(string): Name of the ID variable.
    x_var (string): Time variable.
    y_var (list): List of variable to be displayed.
    col_var (list): List of colors corresponding to each curve.
    threshold_train (string): OPTIONAL. Date of the train threshold.
    weekdays (bool): Default = False. If True, illustrate the weekdays.
    scatter (bool): Default = False. If True, display a scatter plot as well as
    line plot.

    Returns
    -------
    plotly.graph_objs._figure.Figure
    """

    figs = dict()
    params = []
    final_fig = make_subplots(specs=[[{"secondary_y": True}]])

    for i in range(0, len(list_id)):
        id = list_id[i]
        df_id = data.loc[data[group] == id]

        figs[id] = {'final': make_subplots(specs=[[{"secondary_y": True}]]),
                    'vars': px.line(df_id, x=x_var, y=y_var,
                                       color_discrete_sequence=col_var)}

        figs[id]['vars'].update_xaxes(tickformat="%a %d-%m")

        figs[id]['final'].add_traces(data=figs[id]['vars'].data)

        if scatter:
            figs[id]['vars_scatter'] = (
              px.scatter(df_id, x=x_var, y=y_var,
                         color_discrete_sequence=col_var))
            figs[id]['final'].add_traces(data=figs[id]['vars_scatter'].data)
        if weekdays:
            df_id = df_id.assign(weekday_name=lambda d: d[x_var].dt.day_name())
            figs[id]['weekdays'] = px.scatter(df_id, x=x_var, y=y_var,
                                              color='weekday_name')

            marker_type = dict(size=8, line=dict(width=1,  color='black'))
            selector_type = dict(mode='markers')
            figs[id]['weekdays'].update_traces(marker=marker_type,
                                               selector=selector_type)

            figs[id]['final'].add_traces(data=figs[id]['weekdays'].data)

        final_fig.add_traces(data=figs[id]['final'].data)

        list_show = [False] * len(list_id)
        list_show[i] = True
        param = {'args': [{'visible': np.repeat([list_show],
                                                len(figs[id]['final'].data))},
                          {'showlegend' : True,
                          'title': f'id = {id}'}],
                'label': id,
                'method': 'update'}
        params.append(param)

    if threshold_train is not None:
        final_fig.add_vrect(x0=threshold_train, x1=data[x_var].max(),
                            fillcolor="LightSeaGreen", layer="below",
                            line_width=0)

    final_fig.update_layout(updatemenus=[go.layout.Updatemenu(active=-1,
                                                              type="dropdown",
                                                              buttons=params)])

    return final_fig