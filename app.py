import re
import random
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("📖 Random Story Generator")

text = ""
for page in PdfReader("big_story_library.pdf").pages:
    text = text + page.extract_text() + "\n"

library = {}
for story in re.split(r"Story \d+: ", text)[1:]:
    lines = story.strip().split("\n")
    title = lines[0]
    genre = lines[1].replace("Genre: ", "")
    body = " ".join(lines[2:])
    library.setdefault(genre, []).append((title, body))

genre = st.selectbox("Choose a genre", list(library))

if st.button("Generate Story"):
    st.session_state.story = random.choice(library[genre])

if "story" in st.session_state:
    title, body = st.session_state.story
    st.header(title)
    st.write(body)

    sentences = body.split(". ")
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(sentences)

    question = st.text_input("Ask a question about this story")
    if question:
        scores = cosine_similarity(vectorizer.transform([question]), vectors)[0]
        if scores.max() == 0:
            st.warning("Sorry, I could not find that in the story.")
        else:
            st.success(sentences[scores.argmax()])