import pandas as pd
from typing import List
import plotly.graph_objects as go
from sklearn.preprocessing import normalize


def viz_topics_per_class(topic_model,
                               topics_per_class: pd.DataFrame,
                               top_n_topics: int = 10,
                               topics: List[int] = None,
                               normalize_frequency: bool = False,
                               custom_labels: bool = False,
                               width: int = 1250,
                               height: int = 900) -> go.Figure:
    """ Visualize topics per class
    Arguments:
        topic_model: A fitted BERTopic instance.
        topics_per_class: The topics you would like to be visualized with the
                          corresponding topic representation
        top_n_topics: To visualize the most frequent topics instead of all
        topics: Select which topics you would like to be visualized
        normalize_frequency: Whether to normalize each topic's frequency individually
        custom_labels: Whether to use custom topic labels that were defined using 
                       `topic_model.set_topic_labels`.
        width: The width of the figure.
        height: The height of the figure.
    Returns:
        A plotly.graph_objects.Figure including all traces
    Examples:
    To visualize the topics per class, simply run:
    ```python
    topics_per_class = topic_model.topics_per_class(docs, classes)
    topic_model.visualize_topics_per_class(topics_per_class)
    ```
    Or if you want to save the resulting figure:
    ```python
    fig = topic_model.visualize_topics_per_class(topics_per_class)
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/topics_per_class.html"
    style="width:1400px; height: 1000px; border: 0px;""></iframe>
    """
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        selected_topics = list(topics)
    elif top_n_topics is not None:
        selected_topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        selected_topics = sorted(freq_df.Topic.to_list())

    # Prepare data
    if topic_model.custom_labels_ is not None and custom_labels:
        topic_names = {key: topic_model.custom_labels_[key + topic_model._outliers] for key, _ in topic_model.topic_labels_.items()}
    else:
        topic_names = {key: value[:40] + "..." if len(value) > 40 else value
                       for key, value in topic_model.topic_labels_.items()}
    topics_per_class["Name"] = topics_per_class.Topic.map(topic_names)
    data = topics_per_class.loc[topics_per_class.Topic.isin(selected_topics), :]

    # Add traces
    fig = go.Figure()
    for index, topic in enumerate(selected_topics):
        if index == 0:
            visible = True
        else:
            visible = "legendonly"
        trace_data = data.loc[data.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            x = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            x = trace_data.Frequency
        fig.add_trace(go.Bar(y=trace_data.Class,
                             x=x,
                             visible=visible,
                             marker_color=colors[index % 20],
                             hoverinfo="text",
                             name=topic_name,
                             orientation="h",
                             hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))

    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        xaxis_title="Normalized Frequency" if normalize_frequency else "Frequency",
        yaxis_title="Class",
        title={
            'text': "<b>Topics per Class",
            'y': .95,
            'x': 0.40,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        legend=dict(
            title="<b>Global Topic Representation",
        )
    )
    return fig

def viz_topics_over_time(topic_model,
                               topics_over_time: pd.DataFrame,
                               top_n_topics: int = None,
                               topics: List[int] = None,
                               normalize_frequency: bool = False,
                               custom_labels: bool = False,
                               width: int = 1250,
                               height: int = 450) -> go.Figure:
    """ Visualize topics over time
    Arguments:
        topic_model: A fitted BERTopic instance.
        topics_over_time: The topics you would like to be visualized with the
                          corresponding topic representation
        top_n_topics: To visualize the most frequent topics instead of all
        topics: Select which topics you would like to be visualized
        normalize_frequency: Whether to normalize each topic's frequency individually
        custom_labels: Whether to use custom topic labels that were defined using 
                       `topic_model.set_topic_labels`.
        width: The width of the figure.
        height: The height of the figure.
    Returns:
        A plotly.graph_objects.Figure including all traces
    Examples:
    To visualize the topics over time, simply run:
    ```python
    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    topic_model.visualize_topics_over_time(topics_over_time)
    ```
    Or if you want to save the resulting figure:
    ```python
    fig = topic_model.visualize_topics_over_time(topics_over_time)
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/trump.html"
    style="width:1000px; height: 680px; border: 0px;""></iframe>
    """
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        selected_topics = list(topics)
    elif top_n_topics is not None:
        selected_topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        selected_topics = sorted(freq_df.Topic.to_list())

    # Prepare data
    if topic_model.custom_labels_ is not None and custom_labels:
        topic_names = {key: topic_model.custom_labels_[key + topic_model._outliers] for key, _ in topic_model.topic_labels_.items()}
    else:
        topic_names = {key: value[:40] + "..." if len(value) > 40 else value
                       for key, value in topic_model.topic_labels_.items()}
    topics_over_time["Name"] = topics_over_time.Topic.map(topic_names)
    data = topics_over_time.loc[topics_over_time.Topic.isin(selected_topics), :].sort_values(["Topic", "Timestamp"])

    # Add traces
    fig = go.Figure()
    for index, topic in enumerate(data.Topic.unique()):
        trace_data = data.loc[data.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            y = trace_data.Frequency
        fig.add_trace(go.Scatter(x=trace_data.Timestamp, y=y,
                                 mode='lines',
                                 marker_color=colors[index % 20],
                                 hoverinfo="text",
                                 name=topic_name,
                                 hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))

    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        yaxis_title="Normalized Frequency" if normalize_frequency else "Frequency",
        title={
            'text': "<b>Topics over Time",
            'y': .95,
            'x': 0.40,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        legend=dict(
            title="<b>Global Topic Representation",
        )
    )
    return fig