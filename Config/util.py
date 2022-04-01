import streamlit as st
import re
import unicodedata
import plotly.graph_objects as go
import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.figure_factory as ff
import nltk
nltk.download('stopwords')

from Interface import Machine_learning as ml, widget
from Config import Global as glb, Charts

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import AgGrid
from sklearn import metrics
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

###########################################
#          Count Word Function            #
###########################################
def word_count_func(text):
    return len(text.split())

###########################################
#       Exploratory Data Analysis         #
###########################################

###########################################
#     Extraction and Removal Functions    #
#           Extraction/Removal            #
###########################################

###########################################
#    @tag, #hashtag and retweet tag RT@   #
###########################################
tag_re_list = [
    'rt @\w+: ',
    '@[A-Za-z0-9_]+',
    '#[A-Za-z0-9_]+'
]
def find_tag_hashtag_word(text):
    generic_tag_re_list = re.compile('|'.join(tag_re_list))
    if glb.flag == True:
        word = re.findall(generic_tag_re_list, text)
        return " ".join(word)
    else:
        word = re.sub(generic_tag_re_list, '', text)
        return word

###########################################
#    Day of the week, month of the year   #
###########################################
searchItem = ['january',
              'february',
              'march',
              'april',
              'may',
              'june',
              'july',
              'august',
              'september',
              'november',
              'december',
              'monday',
              'tuesday',
              'wednesday',
              'thursday',
              'friday',
              'saturday',
              'sunday']
date_pattern = re.compile(r'|'.join(k for k in searchItem))
def find_date(text):
    if glb.flag == True:
        word = re.findall(date_pattern, text)
        return " ".join(word)
    else:
        word = re.sub(date_pattern, '', text)
        return word

###########################################
#           URL Link HTTP HTTPs           #
###########################################
url_re_list = ['http:\/\/.*', 'https:\/\/.*']
generic_url_re_list = re.compile('|'.join(url_re_list))
def find_url(text):
    if glb.flag == True:
        word = re.findall(generic_url_re_list, text)
        return " ".join(word)
    else:
        word = re.sub(generic_url_re_list, ' ', text)
        return word

###########################################
#           Special Symbol Char           #
###########################################
# Extract Special Symbol
def find_sym_word(text):
    if glb.flag == True:
        word = re.findall(r'[^a-zA-Z\s]', text)
        return " ".join(word)
    else:
        word = re.sub(r'[^a-zA-Z\s]', '', text)
        return word

###########################################
#              Accented Word              #
###########################################
def remove_accentChar(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError:
        pass
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

###########################################
#          Chat Words Conversion          #
###########################################
chat_words_str = """
AFAIK=As Far As I Know
AFK=Away From Keyboard
ASAP=As Soon As Possible
ATK=At The Keyboard
ATM=At The Moment
A3=Anytime, Anywhere, Anyplace
BAK=Back At Keyboard
BBL=Be Back Later
BBS=Be Back Soon
BFN=Bye For Now
B4N=Bye For Now
BRB=Be Right Back
BRT=Be Right There
BTW=By The Way
B4=Before
B4N=Bye For Now
CU=See You
CUL8R=See You Later
CYA=See You
FAQ=Frequently Asked Questions
FK=Fuck
FU*K=Fuck
FC=Fingers Crossed
FWIW=For What It's Worth
FYI=For Your Information
GAL=Get A Life
GG=Good Game
GN=Good Night
GMTA=Great Minds Think Alike
GR8=Great!
G9=Genius
IC=I See
ICQ=I Seek you (also a chat program)
ILU=ILU: I Love You
IMHO=In My Honest/Humble Opinion
IMO=In My Opinion
IOW=In Other Words
IRL=In Real Life
KISS=Keep It Simple, Stupid
LDR=Long Distance Relationship
LMAO=Laugh My A.. Off
LOL=Laughing Out Loud
LTNS=Long Time No See
L8R=Later
MTE=My Thoughts Exactly
M8=Mate
NRN=No Reply Necessary
OIC=Oh I See
PITA=Pain In The A..
PRT=Party
PRW=Parents Are Watching
ROFL=Rolling On The Floor Laughing
ROFLOL=Rolling On The Floor Laughing Out Loud
ROTFLMAO=Rolling On The Floor Laughing My A.. Off
SK8=Skate
STATS=Your sex and age
STFU=Shut the fuck up
stfu=Shut the fuck up
ASL=Age, Sex, Location
THX=Thank You
TTFN=Ta-Ta For Now!
TTYL=Talk To You Later
U=You
U2=You Too
U4E=Yours For Ever
WB=Welcome Back
wtf=What The Fuck
WTG=Way To Go!
WUF=Where Are You From?
W8=Wait...
7K=Sick:-D Laugher
"""
chat_words_map_dict = {}
chat_words_list = []
for line in chat_words_str.split("\n"):
    if line != "":
        cw = line.split("=")[0]
        cw_expanded = line.split("=")[1]
        chat_words_list.append(cw)
        chat_words_map_dict[cw] = cw_expanded
chat_words_list = set(chat_words_list)

def chat_words_conversion(text):
    new_text = []
    for w in text.split():
        if w.lower() in chat_words_list:
            new_text.append(chat_words_map_dict[w.lower()])
        else:
            new_text.append(w)
    return " ".join(new_text)

###########################################
#          Expand Contractions            #
###########################################
import contractions
def expand_contractions(text):
    text = contractions.fix(text)
    return text

###########################################
#               Whitespace                #
###########################################
def remove_whiteSpace(text):
    text = re.sub(r"    ", " ", text)
    return text

###########################################
#            Consecutive Words            #
###########################################
def remove_consec(text):
    val = ""
    i = 0
    while(i < len(text)):
        if(i < len(text) - 2 and
           text[i] * 3 == text[i:i + 3]):
            i += 3
        else:
            val += text[i]
            i += 1
    if (len(val) == len(text)):
        return val
    else:
        return remove_consec(val)

###########################################
#               Stopwords                 #
###########################################
", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in STOPWORDS])

###########################################
#              Lemmatization              #
###########################################
def norm_lemm_v_a_func(text):
    words1 = word_tokenize(text)
    text1 = ' '.join([WordNetLemmatizer().lemmatize(word, pos='v') for word in words1])
    words2 = word_tokenize(text1)
    text2 = ' '.join([WordNetLemmatizer().lemmatize(word, pos='a') for word in words2])
    words3 = word_tokenize(text2)
    text3 = ' '.join([WordNetLemmatizer().lemmatize(word, pos='s') for word in words3])
    words4 = word_tokenize(text3)
    text4 = ' '.join([WordNetLemmatizer().lemmatize(word, pos='r') for word in words4])
    return text4

###########################################
#           Text Configuration            #
#     Apply Extraction & Removal Func     #
###########################################
# @st.cache(persist=True, allow_output_mutation=True)
def configure_all(text, pd):
    if glb.flag == True:
        # Text Extraction
        pd['Hashtag(#) and Tag(RT@/@)'] = text.apply(find_tag_hashtag_word)
        pd['URLs'] = text.apply(find_url)
        pd['Day and Month'] = text.apply(find_date)
        pd['Special Chars and Nums'] = text.apply(find_sym_word)
        return pd
    else:
        # Text Cleaning
        @st.cache(allow_output_mutation=True)
        def preprocessing(text):
            text = text.apply(chat_words_conversion)
            text = text.apply(expand_contractions)
            text = text.apply(remove_accentChar)
            text = text.apply(find_url)
            text = text.apply(find_tag_hashtag_word)
            text = text.apply(find_date)
            text = text.apply(find_sym_word)
            text = text.apply(remove_consec)
            text = text.apply(remove_whiteSpace)
            text = text.apply(remove_stopwords)
            text = text.apply(norm_lemm_v_a_func)
            return text
        
        pd['Cleaned_Text'] = preprocessing(text)
        return pd

###########################################
#           Text Configuration            #
#        Most common word function        #
###########################################

def most_common_word_func(text):
    words = word_tokenize(text)
    fdist = FreqDist(words) 
    df_fdist = pd.DataFrame({'Word': fdist.keys(),
                             'Frequency': fdist.values()})
    df_fdist = df_fdist.sort_values(by='Frequency', ascending=False)
    df_fdist = df_fdist[df_fdist['Frequency'] > 1]
    df_fdist = df_fdist.reset_index(drop = True)

    return df_fdist

###########################################
#           Text Configuration            #
#        Least common word function       #
###########################################

def least_common_word_func(text):
    words = word_tokenize(text)
    fdist = FreqDist(words) 
    df_fdist = pd.DataFrame({'Word': fdist.keys(),
                             'Frequency': fdist.values()})
    df_fdist = df_fdist.sort_values(by='Frequency', ascending=True)
    df_fdist = df_fdist[df_fdist['Frequency'] == 1]
    df_fdist = df_fdist.reset_index(drop = True)
    return df_fdist

###########################################
#             Ag Grid Table               #
###########################################

def AgGridTable(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    gridOptions = gb.build()
    AgGrid(df, enable_enterprise_modules=True, gridOptions=gridOptions, allow_unsafe_jscode=True, height="600px")

###########################################
#          Data Transformation            #
###########################################

###########################################
#             Data Partition              #
###########################################
def data_partition(df):
    st.sidebar.header("Data Modelling")
    st.sidebar.caption("As default, the data will be partitioned into 0.7 train and 0.3 test")
    df = pd.DataFrame(df)
    data_train, data_test = train_test_split(
        df,
        train_size = 0.7,
        test_size = 0.3,
        random_state = 100
    )
    positive = data_train[data_train['vader_analysis'] == 'positive']
    negative = data_train[data_train['vader_analysis'] == 'negative']
    neutral = data_train[data_train['vader_analysis'] == 'neutral']

    # post_count, neg_count, neu_count = data_train['vader_analysis'].value_counts()
    post_count = data_train[data_train['vader_analysis'] == 'positive'].value_counts().sum()
    neg_count = data_train[data_train['vader_analysis'] == 'negative'].value_counts().sum()
    neu_count = data_train[data_train['vader_analysis'] == 'neutral'].value_counts().sum()

    if post_count > neg_count and neu_count < post_count:
        neutral_over_sampling = neutral.sample(post_count, replace =True)
        negative_over_sampling = negative.sample(post_count, replace = True)
        final_over_sampling = pd.concat([neutral_over_sampling,negative_over_sampling,positive],axis=0)
        data_transform(final_over_sampling, data_test)
        
    elif neg_count > post_count and neu_count < neg_count:
        positive_over_sampling = positive.sample(neg_count, replace = True)
        neutral_over_sampling = neutral.sample(neg_count, replace =True)
        final_over_sampling = pd.concat([positive_over_sampling,neutral_over_sampling,negative],axis=0)
        data_transform(final_over_sampling, data_test)

    elif neu_count > post_count and neg_count < neu_count:
        negative_over_sampling = negative.sample(neu_count, replace = True)
        positive_over_sampling = positive.sample(neu_count, replace =True)
        final_over_sampling = pd.concat([positive_over_sampling,negative_over_sampling,neutral],axis=0)
        data_transform(final_over_sampling, data_test)

###########################################
#              Vectorisation              #
###########################################
def data_transform(final_over_sampling, data_test):
    # Train dataset
    X_train = final_over_sampling['Cleaned_Text']
    y_train = final_over_sampling['vader_analysis']

    # Test dataset
    X_test = data_test['Cleaned_Text']
    y_test = data_test['vader_analysis']

    vectoriser = TfidfVectorizer()
    vectoriser.fit(X_train)
    X_train = vectoriser.transform(X_train)
    X_test = vectoriser.transform(X_test)

    option = st.sidebar.multiselect('Machine Learning Model', ['Logistic Regression Model', 'Multinomial Naive Bayes Classifier'])
    if 'Logistic Regression Model' in option:
        ml.logistic_regression_model(X_train, y_train, X_test, y_test)
    if 'Multinomial Naive Bayes Classifier' in option:
        ml.naive_bayes_model(X_train, y_train, X_test, y_test)


###########################################
#            Machine learning             #
###########################################
###########################################
#               Load Models               #
###########################################
def machine_learning_model(model, X_train, y_train, X_test, y_test, label):
    start = dt.datetime.now()
    model.fit(X_train, y_train)
    ## y_score is used to build ROC curve
    y_score = model.predict_proba(X_train)[:, 1]
    y_scores = model.predict_proba(X_train)
    # One hot encode the labels in order to plot them
    y_onehot = pd.get_dummies(y_train, columns=model.classes_)
    y_pred = model.predict(X_test)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    NB_train_score_result = "{:.4f}".format(train_score)
    NB_test_score_result = "{:.4f}".format(test_score)
    NB_score_difference = "{:.4f}".format(train_score - test_score)
    NB_execution_time = str(dt.datetime.now()-start)
    NB_df = pd.DataFrame({'Prediction': y_pred})
    with st.expander(label):
        st.header(label)
        st.subheader("Prediction score")
        prediction_score(y_train, y_score, NB_train_score_result, NB_test_score_result, NB_score_difference, NB_execution_time, NB_df, label)
        st.write('---')
        st.subheader("Classification report at a glance")
        classification(y_test,y_pred,label)
        st.write('---')
        st.subheader("Model Evaluation")
        col1, col2 = st.columns(2)
        with col1:
            Charts.multiclass_ROC(y_scores,y_onehot)
        with col2:
            Charts.multiclass_PR_curve(y_scores,y_onehot)

###########################################
#            Prediction result            #
###########################################
def prediction_score(y_train, y_score, train_score_result, test_score_result, score_difference, execution_time, machine_learning_df, title):
    positive = len(machine_learning_df[machine_learning_df['Prediction'] == 'positive'])
    negative = len(machine_learning_df[machine_learning_df['Prediction'] == 'negative'])
    neutral = len(machine_learning_df[machine_learning_df['Prediction'] == 'neutral'])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Train Score", train_score_result)
    col2.metric("Test Score", test_score_result)
    col3.metric("Score Difference", score_difference)
    col4.metric("Execution Time", execution_time)

    col5, col6 = st.columns(2)
    with col5:
        Charts.Piechart(positive,negative,neutral,title,600,400)
    with col6:
        fig_hist = px.histogram(x=y_score, color=y_train, nbins=50,labels=dict(color='True Labels', x='Score'))
        fig_hist.update_layout(title_text = '<b>Probability Score by the sentiments</b>', paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)', height = 400, width = 600)
        st.plotly_chart(fig_hist)

###########################################
#          Classification report          #
###########################################
def classification(y_test, y_pred, key_value):
    class_report = classification_report(y_test, y_pred, output_dict = True)
    clf_rep_df = pd.DataFrame(class_report).transpose()
    # Accuracy score
    accuracy = round(metrics.accuracy_score(y_test, y_pred),3)
    # Negative score
    neg_prec = round(clf_rep_df.iloc[0,0],3) # Precision score
    neg_rec = round(clf_rep_df.iloc[0,1],3) # recall score
    neg_f1 = round(clf_rep_df.iloc[0,2],3) # F1 score
    # Neutral score
    neu_prec = round(clf_rep_df.iloc[1,0],3) # Precision score
    neu_rec = round(clf_rep_df.iloc[1,1],3) # recall score
    neu_f1 = round(clf_rep_df.iloc[1,2],3) # F1 score
    # Positive score
    pos_prec = round(clf_rep_df.iloc[2,0],3) # Precision score
    pos_rec = round(clf_rep_df.iloc[2,1],3) # recall score
    pos_f1 = round(clf_rep_df.iloc[2,2],3) # F1 score
    widget.download_but(clf_rep_df,"Classification_report.csv")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        Charts.accuracy_score(accuracy, key_value + "Accuracy", "Accuracy")
    with col2:
        st.write("**Precision score**")
        st.metric("Positive", pos_prec)
        st.metric("Negative", neg_prec)
        st.metric("Neutral", neu_prec)
    with col3:
        st.write("**Recall score**")
        st.metric("Positive", pos_rec)
        st.metric("Negative", neg_rec)
        st.metric("Neutral", neu_rec)
    with col4:
        st.write('**F1 score**')
        st.metric("Positive", pos_f1)
        st.metric("Negative", neg_f1)
        st.metric("Neutral", neu_f1)
    positive = list(clf_rep_df.iloc[0])
    negative = list(clf_rep_df.iloc[1])
    neutral = list(clf_rep_df.iloc[2])
    Charts.radar(positive,negative,neutral)

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
        title_text='<b>Receiver Operating Characteristics (ROC) curve</b>',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=650, height=500
    )
    st.plotly_chart(fig)

###########################################
#                PR Curve                 #
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
        title_text='<b>Precision-Recall Curve</b>',
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=650, height=500
    )
    st.plotly_chart(fig)
