import streamlit as st, os, requests
st.title("Hybrid Recommender (KMeans + LightFM)")

user = st.text_input("User ID", "u_001")
k = st.slider("Top-K", 1, 20, 5)
if st.button("Recommend"):
    url = os.environ.get("API_URL","http://localhost:8000/recommend")
    r = requests.post(url, json={"user": user, "topk": k}).json()
    st.write(r)
