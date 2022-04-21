import streamlit as st
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from streamlit_echarts import st_echarts
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

###########################################
#              General Use                #
###########################################
###########################################
#               Bar Chart                 #
###########################################
def barchart(your_dict, title, width, height):
    word = list(your_dict['Word'].values())
    frequency = list(your_dict['Frequency'].values())
    if frequency:
        max_value = max(frequency)
        min_value = min(frequency)
    else:
        max_value = 1
        min_value = 1
    def color(title):
        if (title == 'positive'):
            color = {
                "color": ['#5f2c82', '#12D8FA', '#A6FFCB']
            }
        elif(title == 'negative'):
            color = {
                "color": ['#F27121', '#EA384D', '#1565C0']
            }
        elif(title == 'neutral'):
            color = {
                "color": ['#544a7d', '#ffd452', '#3B4371']
            }
        else:
            color = {
                "color": ['#00bf8f', '#fcb045', '#dc2430']
            }
        return color
    options = {
        "title": {"text": title, "subtext": "Frequency Distribution", "left": "left"},
        "tooltip": {"trigger": 'axis', "axisPointer": {"type": "shadow"}},
        "toolbox": {"feature": {"magicType": {"type": ['line','bar']},"dataView": {}}},
        "xAxis": {"type": "category", "data": word},
        "yAxis": {"type": "value", "boundaryGap": [0, 0.01]},
        "dataZoom": [{"type": 'inside'}],
        "visualMap": {
            "orient": 'horizontal',
            "left": 'center',
            "min": min_value,
            "max": max_value,
            "text": ['High Frequency', 'Low Frequency'],
            "dimension": 1,
            "inRange": color(title)
        },
        "series": [
            {
                "name": "Word Frequency",
                "type": "line",
                "data": frequency,
            }],
        "emphasis": {
            "itemStyle": {
                "shadowBlur": 8,
                "shadowOffsetX": 1,
                "shadowColor": "rgba(0, 0, 0, 0.5)",
                "focus": 'line'
                }
            }
    }
    st_echarts(options=options, width=width, height=height)

###########################################
#               Pie Chart                 #
###########################################
def Piechart(positive, negative, neutral, title, width, height):
    options = {
        "title": {"text": title, "left": "center"},
        "tooltip": {"trigger": "item"},
        "legend": {"top": "bottom"},
        "series": [
            {
                "name": "Sentiment Breakdown",
                "type": "pie",
                "radius": "55%",
                "roseType": "area",
                "itemStyle": {"borderRadius": 8},
                "data": [
                    {"value": positive, "name": "Positive sentiment",
                        "itemStyle": {"color": '#00bf8f'}},
                    {"value": negative, "name": "Negative sentiment",
                        "itemStyle": {"color": '#dc2430'}},
                    {"value": neutral, "name": "Neutral sentiment",
                        "itemStyle": {"color": '#fcb045'}},
                ],
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 1,
                        "shadowColor": "rgba(0, 0, 0, 0.5)"
                    }
                },
            }
        ],
    }
    st_echarts(options=options, height=height, width=width)

def Duplicated_Missing_chart(total_duplicated,total_observation):
    options = {
        "title": {"text": "Missing and Duplicated values", "left": "center"},
        "tooltip": {"trigger": "item"},
        "legend": {"orient": "vertical", "left":"left"},
        "series": [
            {
                "name": "Text Type Extraction",
                "type": "pie",
                "radius": "55%",
                "roseType": "area",
                "itemStyle": {"borderRadius": 8},
                "data": [
                    {"value": total_duplicated, "name": "Duplicated",
                        "itemStyle": {"color": '#00bf8f'}},
                    {"value": total_observation, "name": "Total Observation",
                        "itemStyle": {"color": '#dc2430'}},
                ],
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 1,
                        "shadowColor": "rgba(0, 0, 0, 0.5)"
                    }
                },
            }
        ],
    }
    st_echarts(options=options, height="500px")

###########################################
#             Text_Extraction             #
###########################################
###########################################
#            Wordtype Extraction          #
###########################################
def extraction_chart_func(hashtag_count, url_count, day_count, special_char_count):
    options = {
            "title": {"text": "Word Type Extraction", "subtext": "Frequency Distribution", "left": "center"},
            "tooltip": {"trigger": "item"},
            "legend": {"orient": "vertical", "left": "left"},
            "series": [
                {
                    "name": "Text Type Extraction",
                    "type": "pie",
                    "radius": ['40%', '70%'],
                    "avoidLabelOverlap": "false",
                    "itemStyle": {
                        "borderRadius": 10,
                        "borderColor": '#fff',
                        "borderWidth": 2
                    },
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 1,
                            "shadowColor": "rgba(0, 0, 0, 0.5)",
                            },
                    },
                    "data": [
                        {"value": hashtag_count, "name": "Hashtag/Tag"},
                        {"value": url_count, "name": "URLs"},
                        {"value": day_count, "name": "Date"},
                        {"value": special_char_count, "name": "Punctuations"}
                    ]
                    }
                ],
            }
    st_echarts(options=options, height="600px")

###########################################
#                   NLP                   #
###########################################
###########################################
#             Proc vs Original            #
###########################################
def nlp_barchart(title, original, processed, width, height):
    options = {
    "title": {"text": title, "left": "left"},
    "tooltip": {"trigger": 'axis'},
    "xAxis": {
        "type": "category",
        "data": ["Original Word", "Processed Word"]
    },
    "yAxis": {"type": "value"},
    "series": [
        {
        "data": [
                {"value": original, "name": "Original word", "itemStyle": {"color": '#dc2430'}}, 
                {"value": processed, "name": "Processed word", "itemStyle": {"color": '#fcb045'}}
                ],
        "type": "bar",
        "emphasis": {
            "itemStyle": {
                "shadowBlur": 10,
                "shadowOffsetX": 1,
                "shadowColor": "rgba(0, 0, 0, 0.5)",
                "focus": 'bar'
                }
            }
        },
        ],
    }
    st_echarts(
        options=options,
        width = width, 
        height = height
    )

###########################################
#               sentiment                 #
###########################################

###########################################
#     Positive and Negative Word Merge    #
###########################################
def post_neg_word_freq_chart(dict, title):
    word = list(dict['Word'].values())
    pos_freq = list(dict['Positive Frequency'].values())
    neg_freq = list(dict['Negative Frequency'].values())
    options = {
        "title": {"text": title, "subtext": "Top 25 Words Frequency Distribution", "left": "left"},
        "toolbox": {"feature": {"magicType": {"type": ['stack','line','bar']},"dataView": {},"restore": {}}},
        "tooltip": {"trigger": 'axis', "axisPointer": {"type": 'shadow'}},
        "legend": {"data": ['Positive Sentiment', 'Negative Sentiment']},
        "grid": {"left": '3%', "right": '4%', "bottom": '3%', "containLabel": "true"},
        "xAxis":[
            {"type": 'category', "axisTick": {"show": "false"},
             "data": word
             }
        ],
        "yAxis":[{"type": 'value'}],
        "series": [
            {
                "name": 'Positive Sentiment',
                "type": 'bar',
                "emphasis": {"focus": 'series'},
                "itemStyle": {"color": '#00bf8f'},
                "data": pos_freq
            },
            {
                "name": 'Negative Sentiment',
                "type": 'line',
                "stack": 'Total',
                "emphasis": {"focus": 'series'},
                "itemStyle": {"color": '#dc2430'},
                "data": neg_freq
            }
        ],
        "emphasis": {
            "itemStyle": {
                "shadowBlur": 10,
                "shadowOffsetX": 3,
                "shadowColor": "rgba(0, 0, 0, 0.5)",
                "focus": 'bar'
                }
            }
    }
    st_echarts(options=options, width=1400, height=600)

###########################################
#               Word Cloud                #
###########################################
def plotly_wordcloud(text, colormap):
    """A wonderful function that returns figure data for three equally
    wonderful plots: wordcloud, frequency histogram and treemap"""
    word_cloud = WordCloud(stopwords=set(STOPWORDS), collocations=False, colormap=colormap)
    word_cloud.generate(text)

    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in word_cloud.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x_arr = []
    y_arr = []
    for i in position_list:
        x_arr.append(i[0])
        y_arr.append(i[1])

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i * 80)

    trace = go.Scatter(
        x=x_arr,
        y=y_arr,
        textfont=dict(size=new_freq_list, color=color_list),
        hoverinfo="text",
        textposition="top center",
        hovertext=["{0} - {1}".format(w, f) for w, f in zip(word_list, freq_list)],
        mode="text",
        text=word_list
    )

    layout = go.Layout(
        {
            "xaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "autorange":True,
            },
            "yaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "autorange": True,
            },
            "margin": dict(t=20, b=20, l=10, r=10, pad=4),
            "hovermode": "closest",
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=1400, 
        height=700
    )

    wordcloud_figure_data = {"data": [trace], "layout": layout}
    word_list_top = word_list[:25]
    word_list_top.reverse()
    freq_list_top = freq_list[:25]
    freq_list_top.reverse()
    st.subheader("**WordCloud**")
    st.plotly_chart(wordcloud_figure_data)

###########################################
#            Machine Learning             #
###########################################
###########################################
#              Accuracy Fill              #
###########################################
def accuracy_score(data, key_value, title):
    option = {
        "title": {"text": title, "subtext": "Percentage score", "left": "left"},
        "tooltip": {"trigger": 'item'},
        "series": [
            {
            'type': 'liquidFill',
            'name': title,
            'data': [{'value': data}],
            "animationDuration": 0,
            "animationDurationUpdate": 2000,
            "animationEasingUpdate": 'cubicOut'
            }
        ]
    }
    st_echarts(option, key=key_value)

###########################################
#                 Radar                   #
###########################################
def radar(postive, negative, neutral):
    data = [
                    {
                        "value": postive,
                        "name": "Positive"
                    },
                    {
                        "value": negative,
                        "name": "Negative"
                    },
                    {
                        "value": neutral,
                        "name": "Neutral"
                    }
            ]
    options = {
        "legend": {"data": ['Positive', 'Negative', 'Neutral']},
        "radar": {
            "indicator": [
                {"name": "Recall", "max": 1},
                {"name": "Precision", "max": 1},
                {"name": "F1-Score", "max": 1},
            ]
        },
        "tooltip": {"trigger": 'item'},
        "series": [
            {
                "name": 'Classification report',
                "type": 'radar',
                "data": data
            }
        ],
        "emphasis": {
            "itemStyle": {
                "shadowBlur": 10,
                "shadowOffsetX": 1,
                "shadowColor": "rgba(0, 0, 0, 0.5)",
                "focus": 'bar'
                }
            }
    }
    st_echarts(options=options, height="800px")

###########################################
#               ROC Curve                 #
###########################################
def multiclass_ROC(y_scores, y_onehot):
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    for i in range(y_scores.shape[1]):
        y_true = y_onehot.iloc[:, i]
        y_score = y_scores[:, i]

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)

        name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    fig.update_layout(
        title_text='Receiver Operating Characteristics (ROC) curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=650, height=500
    )
    st.plotly_chart(fig)

###########################################
#               PR Curve                  #
###########################################
def multiclass_PR_curve(y_scores, y_onehot):
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )
    for i in range(y_scores.shape[1]):
        y_true = y_onehot.iloc[:, i]
        y_score = y_scores[:, i]

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc_score = average_precision_score(y_true, y_score)

        name = f"{y_onehot.columns[i]} (AP={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=recall, y=precision, name=name, mode='lines'))

    fig.update_layout(
        title_text='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=650, height=500
    )
    st.plotly_chart(fig)

###########################################
#               Text Input                #
###########################################
###########################################
#          Polarity Subjectivity          #
###########################################
def polarity_subjectivity_chart(title, value, width, height):
    options = {
    "title": {"text": title, "left": "left"},
    "tooltip": {"trigger": 'axis'},
    "xAxis": {
        "type": "category",
        "data": ["Polarity","Subjectivity"]
    },
    "yAxis": {"type": "value"},
    "visualMap": {
            "orient": 'horizontal',
            "left": 'center',
            "min": -1,
            "max": 1,
            "dimension": 1,
            "inRange": {"color": ['#fd1d1d', '#fcb045']}
    },
    "series": [
         {
             "name": "score",
             "data": value,
             "type": "bar",
             "itemStyle": {"color": '#fcb045'}
         }],
        "emphasis": {
            "itemStyle": {
                "shadowBlur": 10,
                "shadowOffsetX": 1,
                "shadowColor": "rgba(0, 0, 0, 0.5)",
                "focus": 'bar'
                }
            }
       
    }
    st_echarts(
        options=options,
        width = width, 
        height = height
    )

###########################################
#            Token Sentiment              #
###########################################
def token_sent_barchart(metric, value, title):
    options = {
        "title": {"text": title, "left": "left"},
        "tooltip": {"trigger": 'axis', "axisPointer": {"type": "shadow"}},
        "toolbox": {"feature": {"magicType": {"type": ['line','bar']},"dataView": {}}},
        "xAxis": {"type": "category", "data": metric},
        "yAxis": {"type": "value", "boundaryGap": [0, 0.01]},
        "dataZoom": [{"type": 'inside'}],
        "visualMap": {
            "orient": 'horizontal',
            "left": 'center',
            "min": -1,
            "max": 1,
            "text": ['Positive', 'Negative'],
            "dimension": 1,
            "inRange": {"color": ['#dc2430', '#fcb045', '#00bf8f']}
        },
        "series": [
            {
                "name": "Word Frequency",
                "type": "line",
                "data": value,
            }],
        "emphasis": {
            "itemStyle": {
                "shadowBlur": 10,
                "shadowOffsetX": 1,
                "shadowColor": "rgba(0, 0, 0, 0.5)",
                "focus": 'bar'
                }
            }
    }
    st_echarts(options=options, width="650px", height="300px")

###########################################
#      Machine Learning Predictions       #
###########################################
def ML_pie(log_pred, mnvm_pred):
    options = {
            "title": {"text": "Prediction using Machine learning models", "subtext": "1=Negative, 0=Positive", "left": "left"},
            "tooltip": {"trigger": "item"},
            "legend": {"data": ["SGD Classifier","Logistic Regression", "Multinomial Naive Bayes Classifier"], 'bottom': 'bottom'},
            "series": [
                {
                    "name": "Prediction",
                    "type": "pie",
                    "radius": ['40%', '70%'],
                    "avoidLabelOverlap": "false",
                    "itemStyle": {
                        "borderRadius": 10,
                        "borderColor": '#fff',
                        "borderWidth": 2
                    },
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 1,
                            "shadowColor": "rgba(0, 0, 0, 0.5)",
                            },
                    },
                    "data": [
                        {"value": log_pred, "name": "SGD Classifier", "itemStyle": {"color": '#fd1d1d'}},
                        {"value": mnvm_pred, "name": "Logistic Regression", "itemStyle": {"color": '#fcb045'}},
                        {"value": mnvm_pred, "name": "Multinomial Naive Bayes Classifier", "itemStyle": {"color": '#00bf8f'}}
                    ]
                    }
                ],
            }
    st_echarts(options=options, width="650px", height="500px")
