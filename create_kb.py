import os
import json

import fitz 
from PIL import Image 
import glob
from tqdm import tqdm

from openai import OpenAI

import base64

from pydantic import BaseModel
from llama_index.core.node_parser import SentenceSplitter
import json
from dotenv import load_dotenv


from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

from dotenv import load_dotenv

###  Helper functions
class ExtractedInfo(BaseModel):
    information_category: str
    info: str
    keywords: list[str]
    source: str
    
class StructuredOutput(BaseModel):
    extracted_information: list[ExtractedInfo]

# sentence splitter
sent_spilt = SentenceSplitter(chunk_size = 512, chunk_overlap=32)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')








def extract_rules_func(reference_file_path,client):
    rule_folder = os.path.join(reference_file_path, 'extracted_rules')
    if not os.path.exists(rule_folder):
        os.makedirs(rule_folder)
    
    #get the list of pdfs in the reference folder
    pdfs_list = glob.glob(os.path.join(reference_file_path, '*.pdf'))
    print('Number of pdfs:', len(pdfs_list))
    print('pdfs:', pdfs_list)


    ###################################
    #PROMPTS
    ##################################

    text_extract_prompt = '''
**Task:** 
Extract all the information from the provided text related to survivorship care of cancer patients. These information should be relevant to make recommendations to cancer patients or survivors along with creating a comprehensive survivorship care plan for them.
The extracted information should belong to the following categories:
  1. Cancer surveillance and other recommended tests for cancer monitoring.
      - This category includes information on recommended tests, clinical visits and cancer surveillance methods for monitoring cancer patients.
      - For example, information on frequency of clinical visits, types of tests (e.g. blood tests, imaging tests) and other surveillance methods.
  2. Possible late and long-term effects of cancer treatment.
      - This category includes information on possible late and long-term effects of cancer treatment that a cancer survivor may experience during their survivorship journey.
      - For example, information on possible side effects of chemotherapy, radiation therapy, surgery and other cancer treatments.
      - Extract information on how to manage these effects and provide support to cancer survivors.
  3. Possible other issues that cancer survivors may experience.
      - This category includes information on issues other than late and long-term effects of cancer treatment that a cancer survivor may experience during their survivorship journey.
      - For example, information on emotional, psychological, social, financial and other issues that cancer survivors may face.
      - Extract information on how to address these issues and provide support to cancer survivors.
  4. Lifestyle and behavior recommendations for cancer survivors.
      - This category includes information on lifestyle and behavior recommendations for cancer survivors to improve their quality of life and reduce the risk of cancer recurrence.
      - For example, information on diet, exercise, smoking cessation, alcohol consumption and other lifestyle factors.
  5. References to helpful resources for cancer survivors.
      - This category includes information on helpful resources, organizations, websites, support groups and other sources of information and support for cancer survivors.
      - For example, information on cancer survivorship programs, patient advocacy groups, website links and other resources.
  6. Additional Information
      - Feel free to extract any additional information that you think is relevant to make recommendations to cancer patients or survivors along with creating a comprehensive survivorship care plan for them.
      
** Guidelines:**
- No hallucinations: Ensure all extracted information comes directly from the text.
- Comprehensive output: Provide detailed descriptions of the information found.
- Skip if irrelevant: If the text contains no relevant information about survivorship care for cancer patients, simply output: "No information found" for the "info" field.
- Actionable information: Extract information that can be used to make recommendations to cancer patients or survivors and create a comprehensive survivorship care plan for them.
- Exclude irrelevant information: Do not include information that is not relevant to recommending survivorship care for cancer patients. Ignore author names, references and other irrelevant text.
- Structure: Strictly adhere to the output format provided below in JSON.

**Output Format:**
[OUTPUT_FORMAT]

-----------------------------------
**Text:**
[TEXT]
----------------------------------- 
'''

    output_format = '''
[{ "Extracted information": [
    {"information_category": < Category of the extracted information, should be one of the six categories mentioned above. >,
    "info": < extracted rule or information >,
    "keywords": < list of top 2 important keywords in the extracted information >,
    "source": < source of the information extracted. Provide the exact text in the source and do not modify or edit.>,
    } 
    ...
] }]
'''

    img_extract_prompt = f'''
**Task:** 
Extract all the information from the provided image related to survivorship care of cancer patients. These information should be relevant to make recommendations to cancer patients or survivors along with creating a comprehensive survivorship care plan for them.
The extracted information should belong to the following categories:
  1. Cancer surveillance and other recommended tests for cancer monitoring.
      - This category includes information on recommended tests, clinical visits and cancer surveillance methods for monitoring cancer patients.
      - For example, information on frequency of clinical visits, types of tests (e.g. blood tests, imaging tests) and other surveillance methods.
  2. Possible late and long-term effects of cancer treatment.
      - This category includes information on possible late and long-term effects of cancer treatment that a cancer survivor may experience during their survivorship journey.
      - For example, information on possible side effects of chemotherapy, radiation therapy, surgery and other cancer treatments.
      - Extract information on how to manage these effects and provide support to cancer survivors.
  3. Possible other issues that cancer survivors may experience.
      - This category includes information on issues other than late and long-term effects of cancer treatment that a cancer survivor may experience during their survivorship journey.
      - For example, information on emotional, psychological, social, financial and other issues that cancer survivors may face.
      - Extract information on how to address these issues and provide support to cancer survivors.
  4. Lifestyle and behavior recommendations for cancer survivors.
      - This category includes information on lifestyle and behavior recommendations for cancer survivors to improve their quality of life and reduce the risk of cancer recurrence.
      - For example, information on diet, exercise, smoking cessation, alcohol consumption and other lifestyle factors.
  5. References to helpful resources for cancer survivors.
      - This category includes information on helpful resources, organizations, websites, support groups and other sources of information and support for cancer survivors.
      - For example, information on cancer survivorship programs, patient advocacy groups, website links and other resources.
  6. Additional Information
      - Feel free to extract any additional information that you think is relevant to make recommendations to cancer patients or survivors along with creating a comprehensive survivorship care plan for them.
      
** Guidelines:**
- No hallucinations: Ensure all extracted information comes directly from the image.
- Comprehensive output: Provide detailed descriptions of the information found.
- Skip if irrelevant: If the image contains no relevant information about survivorship care for cancer patients, simply output: "No information found" for the "info" field.
- Actionable information: Extract information that can be used to make recommendations to cancer patients or survivors and create a comprehensive survivorship care plan for them.
- Exclude irrelevant information: Do not include information that is not relevant to recommending survivorship care for cancer patients. Ignore author names, institutes, references and other irrelevant information.
- Critic whether you could use the information to make recommendations to cancer patients or survivors and create a comprehensive survivorship care plan for them. If not, output: "No information found".
- Structure: Strictly adhere to the output format provided below in JSON.

**Output Format:**
{output_format}

-----------------------------------
'''


    # extract the text and images from the pdfs and save them
    for pdf_file in pdfs_list:
        rules_dict = []
        #create a folder to save the extracted rules
        pdf_file_name = os.path.basename(pdf_file).split('.')[0]
        if os.path.exists(os.path.join(rule_folder, f'{pdf_file_name}_extracted_rules_2.json')):
            continue
        
        print('Extracting rules from:', pdf_file_name,'.............')
        
        pdf_file = fitz.open(pdf_file)
        
        if 'IMG' not in pdf_file_name:
            for i in tqdm(range(len(pdf_file))):
                
                # extracting from text
                page = pdf_file[i]
                text = page.get_text()
                
                # split the text into chunks
                text_chunk_list = sent_spilt.split_text(text)
                
                for text_chunk in text_chunk_list:
                    
                    temp_dict = {'metadata': {'page_number': i+1, 'doc_title': pdf_file.metadata['title']}}
                    
                    edit_text_extract_prompt = text_extract_prompt.replace('[OUTPUT_FORMAT]', output_format)
                    edit_text_extract_prompt = text_extract_prompt.replace('[TEXT]', text_chunk)
                    
                
                    completion = client.beta.chat.completions.parse(
                            model="gpt-4o-2024-08-06",
                            temperature=0.2,
                            response_format= StructuredOutput,
                            messages=[
                                {"role": "system", "content": "You are a cancer survivorship care expert. Given the following text extracted from a pdf, extract all possible information related survivorship care of cancer survivors."},
                                {"role": "user", "content":edit_text_extract_prompt }])

                    # convert to json
                    extract_info_dict = json.loads(completion.choices[0].message.content)
                    
                    try:
                        for extract_info in  extract_info_dict['extracted_information']:
                            if "No information found" in extract_info['info']:
                                continue            
                            else:
                                temp_rules_dict = {}
                                temp_rules_dict['information_category'] = extract_info['information_category']
                                temp_rules_dict['info'] = extract_info['info']
                                temp_rules_dict['metadata'] = temp_dict['metadata']
                                temp_rules_dict['metadata']['keywords'] = extract_info['keywords']
                                
                                temp_rules_dict['source'] = text_chunk
                                rules_dict.append(temp_rules_dict)
                                
                    except:
                        pass
                                

        else:
            for i in tqdm(range(len(pdf_file))):
                page = pdf_file[i]
                # extracting rules form images
                pix = page.get_pixmap(dpi=300) 
                #save to a temporary file
                pix.save(os.path.join(reference_file_path, 'temp.png'))
            
                base64_image = encode_image(os.path.join(reference_file_path, 'temp.png'))
        
                temp_dict = {'metadata': {'page_number': i+1, 'doc_title': pdf_file.metadata['title']}}
        
        
       
                messages =  [
        {"role": "system", "content": "You are a cancer survivorship care expert. Given the following text extracted from a pdf, extract all possible information related survivorship care of cancer survivors."},
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": img_extract_prompt,
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }]
    
                # response = requests.post("https://api.openai.com/v1/beta/chat/completions/parse/", headers=headers, json=payload)
                response = client.beta.chat.completions.parse(
                            model="gpt-4o-2024-08-06",
                            temperature=0.2,
                            response_format= StructuredOutput,
                            messages = messages)
                            
                # convert to json
                extract_info_dict = json.loads(response.choices[0].message.content)
                # print(extract_info_dict)
                # try:
                for extract_info in  extract_info_dict['extracted_information']:
                    
                    if "No information found" in extract_info['info']:
                        # print('No information found')
                        continue            
                    else:
                        # print('Information found')
                        temp_rules_dict = {}
                        temp_rules_dict['information_category'] = extract_info['information_category']
                        temp_rules_dict['info'] = extract_info['info']
                        temp_rules_dict['metadata'] = temp_dict['metadata']
                        temp_rules_dict['metadata']['keywords'] = extract_info['keywords']

                        temp_rules_dict['source'] = extract_info['source']
                        # print(temp_rules_dict)
                        rules_dict.append(temp_rules_dict)
                # except:
                #     pass
            

        
        # print(rules_dict)
        pdf_file.close()
    
        # save the extracted rules
        with open(os.path.join(rule_folder, f'{pdf_file_name}_extracted_rules.json'), 'w') as f:
            json.dump({'rules': rules_dict}, f)


def split_rules_into_knowledge_bases(extracted_rules_path,save_path,group_lists):
    '''
    Split the extracted rules into multiple knowledge bases based on the groups.
    
    '''        
    group_wise_separated_kb = {group: [] for group in group_lists}
    
    rules_files = glob.glob(extracted_rules_path + '/*.json')
    
    for file in tqdm(rules_files):
        with open(file) as f:
            rules = json.load(f)
            rules = rules['rules']
        
        for rule in rules:
            rule_temp_dict = {}
            rule_temp_dict['metadata'] = rule['metadata']
            rule_temp_dict['metadata']['information_category'] = rule['information_category']
            rule_temp_dict['metadata']['source'] = rule['source']
            rule_temp_dict['text'] = rule['info']
            if rule['information_category'] in group_lists:
                group_wise_separated_kb[rule['information_category']].append(rule_temp_dict)
            else:
                group_wise_separated_kb['Additional Information'].append(rule_temp_dict)

    # save the separated knowledge
    with open(os.path.join(save_path, 'group_wise_separated_knowledge.json'), 'w') as f: 
        json.dump(group_wise_separated_kb, f, indent=4)
        print(f'Group wise separated knowledge saved to {os.path.join(save_path, "group_wise_separated_knowledge.json")}')   


def create_and_store_vector_index_for_KB(processed_knowledge_path,save_vector_index_path):
    # read the knowledge base
    with open(processed_knowledge_path, 'r') as f:
        knowledge_base = json.load(f)
        
    main_keys = list(knowledge_base.keys())
    
    for main_key in tqdm(main_keys):
        
        kb_docs = [Document(text=doc['text'],metadata=doc['metadata'],excluded_embed_metadata_keys=["page_number","information_category","source"]) for doc in knowledge_base[main_key]]
        kb_index = VectorStoreIndex.from_documents(kb_docs)
        
        # store the vector index
        save_vector_index_path_sub = os.path.join(save_vector_index_path,main_key)
        if not os.path.exists(save_vector_index_path_sub):
            os.makedirs(save_vector_index_path_sub)
        kb_index.storage_context.persist(persist_dir=save_vector_index_path_sub)
            

   


if __name__ == '__main__':
    
    reference_file_path='./test_kb' #'Path to the folder containing the pdfs with the guidelines'
    
    
    
    # openai
    load_dotenv()
    OPENAI_API_KEY= os.getenv('OPENAI_API_KEY')
    temperature=0.2

    client = OpenAI(api_key=OPENAI_API_KEY)
    
    ### Extract rules from the guidelines #####
    extract_rules_func(reference_file_path,client)
    
    #### Create and store vector index for the knowledge bases #####
    processed_knowledge_path = os.path.join(reference_file_path, 'group_wise_separated_knowledge.json')
    save_vector_index_path = os.path.join(reference_file_path, 'vector_index')
    
    #### Split the rules into multiple knowledge bases #####
    rules_path = os.path.join(reference_file_path, 'extracted_rules')
    save_path = os.path.join('./new_kbs')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    group_lists = ['Cancer surveillance and other recommended tests for cancer monitoring',
                'Possible late and long-term effects of cancer treatment',
                'Possible other issues that cancer survivors may experience',
                'Lifestyle and behavior recommendations for cancer survivors',
                'References to helpful resources for cancer survivors',
                'Additional Information']

    split_rules_into_knowledge_bases(rules_path,save_path,group_lists)


    #### Create and store vector index for the knowledge bases #####
    processed_knowledge_path = os.path.join(save_path, 'group_wise_separated_knowledge.json')
    save_vector_index_path = os.path.join(save_path, 'vector_kbs')
    
    
    load_dotenv()
    
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_endpoint = os.getenv('OPENAI_API_BASE')
    api_version = os.getenv('OPENAI_API_VERSION')
    
    azure_embedding_model = 'text-embedding-3-large'
    
    if not os.path.exists(save_vector_index_path):
        os.makedirs(save_vector_index_path)
        
    create_and_store_vector_index_for_KB(processed_knowledge_path,save_vector_index_path)
    
    print('Knowledge bases created and stored in vector index')