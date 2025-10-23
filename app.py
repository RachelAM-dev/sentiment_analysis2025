import streamlit as st
import pandas as pd
import joblib
from utils import preprocessor
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def run():
    model = joblib.load(open('model.joblib','rb'))

    st.title("Sentiment Analysis")
    st.text("Basic app to detect the sentiment of text.")
    st.text("")
    userinput = st.text_input('Enter text below, then click the Predict button.', placeholder='Input text HERE')
    st.text("")
    
    # Initialise rolling counts and history
    if "sentiment_counts" not in st.session_state:
        st.session_state.sentiment_counts = {"Positive": 0, "Negative": 0}
    if "history" not in st.session_state:
        st.session_state.history = []   # store all texts

    # Prediction logic
    if st.button("Predict"):
        predicted_sentiment = model.predict(pd.Series([userinput]))[0]

        if predicted_sentiment == 1:
            output = 'positive 👍'
            label = "Positive"
            sentiment = f'Predicted sentiment of "{userinput}" is {output}.'
            st.success(sentiment)   # green background
        else:
            output = 'negative 👎'
            label = "Negative"
            sentiment = f'Predicted sentiment of "{userinput}" is {output}.'
            st.error(sentiment)     # red background

        # Update rolling counts and history regardless of label
        st.session_state.sentiment_counts[label] += 1
        st.session_state.history.append(userinput)

        # Show result
#        sentiment = f'Predicted sentiment of "{userinput}" is {output}.'
#        st.success(sentiment)

    # Build pie chart from current counts (each run)
    df_counts = pd.DataFrame({
        "Sentiment": list(st.session_state.sentiment_counts.keys()),
        "Count": list(st.session_state.sentiment_counts.values())
    })

    fig = px.pie(
        df_counts,
        names="Sentiment",
        values="Count",
        color="Sentiment",
        hole=0.3
    )
    st.plotly_chart(fig, use_container_width=True)

    # Word cloud from all entered text
    if st.session_state.history:
        text_blob = " ".join(st.session_state.history)
        wc = WordCloud(width=800, height=400, background_color="white").generate(text_blob)

        fig_wc, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc)



if __name__ == "__main__":
    run()