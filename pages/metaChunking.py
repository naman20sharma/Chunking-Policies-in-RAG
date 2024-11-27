# Meta chunking streamlit frontend
import streamlit as st
from utils.retrieval import answer_query_meta_chunking

def main():
    st.title("Retrieval-Augmented Generation System with Llama and RocksDB (Meta Chunking)")
    
    query = st.text_input("Enter your question:")
    
    if st.button("Submit"):
        if query:
            with st.spinner("Generating answer..."):
                answer, context = answer_query_meta_chunking(query)
            st.write("**Answer:**")
            st.write(answer)
            st.write("**Chunks pulled for meta chunking**")
            st.write(context)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()