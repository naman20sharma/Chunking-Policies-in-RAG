from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
import jieba
from typing import List

class Chunking:
    def __init__(self, model, tokenizer) -> None:
        self.model=model
        self.tokenizer=tokenizer
    
    def get_ppl_batch(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        return_kv=False,
        end=None
    ):
        past_length = 0
        if end is None:
            end = input_ids.shape[1]
        with torch.no_grad():
            response =self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = response.past_key_values
        shift_logits = response.logits[..., :-1, :].contiguous()  
        shift_labels = input_ids[..., past_length + 1 : end].contiguous() 
        # Flatten the tokens
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)  
        res = loss
        return (res, past_key_values) if return_kv else res

model_name_or_path= 'Qwen/Qwen2-1.5B-Instruct'   
device_map = "auto"

small_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-1.5B-Instruct',trust_remote_code=True)  
small_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct', trust_remote_code=True,device_map=device_map) 
small_model.eval()

def find_minima(values,threshold):  
    minima_indices = []  
    for i in range(1, len(values) - 1):  
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            if (values[i - 1]-values[i]>=threshold) or (values[i + 1]-values[i]>=threshold):
                minima_indices.append(i)  
        elif values[i] < values[i - 1] and values[i] == values[i + 1]:
            if values[i - 1]-values[i]>=threshold:
                minima_indices.append(i) 
    return minima_indices

def split_text_by_punctuation(text,language): 
    if language=='zh': 
        sentences = jieba.cut(text, cut_all=False)  
        sentences_list = list(sentences)  
        sentences = []  
        temp_sentence = ""  
        for word in sentences_list:  
            if word in ["。", "！", "？","；"]:  
                sentences.append(temp_sentence.strip()+word)  
                temp_sentence = ""  
            else:  
                temp_sentence += word  
        if temp_sentence:   
            sentences.append(temp_sentence.strip())  
        
        return sentences
    else:
        full_segments = sent_tokenize(text)
        ret = []
        for item in full_segments:
            item_l = item.strip().split(' ')
            if len(item_l) > 512:
                if len(item_l) > 1024:
                    item = ' '.join(item_l[:256]) + "..."
                else:
                    item = ' '.join(item_l[:512]) + "..."
            ret.append(item)
        return ret

def extract_by_html2text_db_nolist(sub_text,model,tokenizer,threshold,language='zh') -> List[str]:  
    temp_para=sub_text

    if language=='zh':
        # text = re.sub(r'[\t\n\r\f\v]', '', temp_para)  
        # cleaned_text = re.sub(r'  ', '', text)  
        cleaned_text=temp_para
    else:
        cleaned_text=temp_para
 
    segments = split_text_by_punctuation(cleaned_text,language)
    segments = [item for item in segments if item.strip()]  
    ch=Chunking(model, tokenizer)
    len_sentences=[]
    input_ids=torch.tensor([[]], device=model.device,dtype=torch.long)  
    attention_mask =torch.tensor([[]], device=model.device,dtype=torch.long)  
    for context in segments:
        tokenized_text = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_text["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id],dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_tmp = tokenized_text["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_tmp],dim=-1)

    loss, past_key_values = ch.get_ppl_batch( 
        input_ids,
        attention_mask,
        past_key_values=None,
        return_kv=True
    )
    first_cluster_ppl=[]
    index=0
    for i in range(len(len_sentences)):
        if i ==0:
            first_cluster_ppl.append(loss[0:len_sentences[i]-1].mean().item())
            index+=len_sentences[i]-1
        else:
            first_cluster_ppl.append(loss[index:index+len_sentences[i]].mean().item())
            # print(loss[index:index+len_sentences[i]])
            index+=len_sentences[i]
        
    # print(first_cluster_ppl) 
    minima_indices=find_minima(first_cluster_ppl,threshold)
    first_chunk_indices=[]
    first_chunk_sentences=[]
    split_points = [0] + minima_indices + [len(first_cluster_ppl)-1]    
    for i in range(len(split_points)-1):
        tmp_index=[]
        tmp_sentence=[]
        if i==0:
            tmp_index.append(0)
            tmp_sentence.append(segments[0])
        for sp_index in range(split_points[i]+1,split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(segments[sp_index])
        first_chunk_indices.append(tmp_index)
        first_chunk_sentences.append(tmp_sentence)
    final_chunks=[]
    for sent_list in first_chunk_sentences:
        final_chunks.append(''.join(sent_list))
    print('111',first_chunk_indices)
    # print('222', first_chunk_sentences)

    return final_chunks

# Main chunking starts here
def get_prob_subtract(model,tokenizer,sentence1,sentence2,language):
    if language=='zh':
        query='''这是一个文本分块任务.你是一位文本分析专家，请根据提供的句子的逻辑结构和语义内容，从下面两种方案中选择一种分块方式：
        1. 将“{}”分割成“{}”与“{}”两部分；
        2. 将“{}”不进行分割，保持原形式；
        请回答1或2。'''.format(sentence1+sentence2,sentence1,sentence2,sentence1+sentence2)
        prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        input_ids=prompt_ids
        output_ids = tokenizer.encode(['1','2'], return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            token_probs = F.softmax(next_token_logits, dim=-1)
        next_token_id_0 = output_ids[:, 0].unsqueeze(0)
        next_token_prob_0 = token_probs[:, next_token_id_0].item()      
        next_token_id_1 = output_ids[:, 1].unsqueeze(0)
        next_token_prob_1 = token_probs[:, next_token_id_1].item()  
        prob_subtract=next_token_prob_1-next_token_prob_0
    else:
        query='''This is a text chunking task. You are a text analysis expert. Please choose one of the following two options based on the logical structure and semantic content of the provided sentence:
        1. Split "{}" into "{}" and "{}" two parts;
        2. Keep "{}" unsplit in its original form;
        Please answer 1 or 2.'''.format(sentence1+' '+sentence2,sentence1,sentence2,sentence1+' '+sentence2)
        prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        input_ids=prompt_ids
        output_ids = tokenizer.encode(['1','2'], return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            token_probs = F.softmax(next_token_logits, dim=-1)
        next_token_id_0 = output_ids[:, 0].unsqueeze(0)
        next_token_prob_0 = token_probs[:, next_token_id_0].item()      
        next_token_id_1 = output_ids[:, 1].unsqueeze(0)
        next_token_prob_1 = token_probs[:, next_token_id_1].item()  
        prob_subtract=next_token_prob_1-next_token_prob_0
    return prob_subtract

# from chunk_rag import extract_by_html2text_db_nolist,split_text_by_punctuation
def meta_chunking(original_text,base_model,language,ppl_threshold,chunk_length):
    chunk_length=int(chunk_length)
    if base_model=='PPL Chunking':
        final_chunks=extract_by_html2text_db_nolist(original_text,small_model,small_tokenizer,ppl_threshold,language=language)
    else:
        full_segments = split_text_by_punctuation(original_text,language)
        tmp=''
        threshold=0
        threshold_list=[]
        final_chunks=[]
        for sentence in full_segments:
            if tmp=='':
                tmp+=sentence
            else:
                prob_subtract=get_prob_subtract(small_model,small_tokenizer,tmp,sentence,language)    
                threshold_list.append(prob_subtract)
                if prob_subtract>threshold:
                    tmp+=' '+sentence
                else:
                    final_chunks.append(tmp)
                    tmp=sentence
            if len(threshold_list)>=5:
                last_ten = threshold_list[-5:]  
                avg = sum(last_ten) / len(last_ten)
                threshold=avg
        if tmp!='':
            final_chunks.append(tmp)
            
    merged_paragraphs = []
    current_paragraph = ""  
    if language=='zh':
        for paragraph in final_chunks:  
            if len(current_paragraph) + len(paragraph) <= chunk_length:  
                current_paragraph +=paragraph  
            else:  
                merged_paragraphs.append(current_paragraph)  
                current_paragraph = paragraph    
    else:
        for paragraph in final_chunks:  
            if len(current_paragraph.split()) + len(paragraph.split()) <= chunk_length:  
                current_paragraph +=' '+paragraph  
            else:  
                merged_paragraphs.append(current_paragraph)   
                current_paragraph = paragraph 
    if current_paragraph:  
        merged_paragraphs.append(current_paragraph) 
    final_text='\n\n'.join(merged_paragraphs)
    print(final_text)
    return final_text