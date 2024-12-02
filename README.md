# DPSFinalProj
Here is the link to our project https://docs.google.com/document/d/1OPwcW68nX6oHPC0tTZNK4IeEwQEZnrusHZRxExFZkss/edit?tab=t.0#heading=h.3zm9f8g1ndd9

1. Open the directory you want the repo in and then clone the repository using the link 
```python
git clone https://github.com/Fall-24-CSE511-Data-Processing-at-Scale/Project-7-Group-14.git
```
2. Open anaconda and run the following commands down below in the directory you have cloned the repository
If you do not have anaconda download it from the link (https://docs.anaconda.com/anaconda/install/)
```python
conda create -n rag_system 
conda activate rag_system
pip install -r requirements.txt

```
```python
git clone https://github.com/facebook/rocksdb.git
git clone https://github.com/huggingface/transformers.git
```

3. Add a data folder within the repo and add the file you want to ingest into RocksDB, make sure the file has only a .txt extension

4. Run the command below to ingest the data from your file into the vector database
```python
python build_vector_store.py
```
5. Run the command
```python
huggingface-cli login
```
Once prompted to enter the token, enter
```python
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
```
Do not share the token with anyone

7. Once the data is ingested run the command down below to open our chatbot
```python
streamlit run main.py
```
After running the command we see our UI, once you ask a question be patient it takes over 10 minutes to generate one answer 

