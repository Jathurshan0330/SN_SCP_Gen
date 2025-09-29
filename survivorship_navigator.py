import re
import warnings
warnings.filterwarnings("ignore")
import os
import json
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from sentence_transformers import CrossEncoder

from models.openai_azure import azure_open_ai_call, load_env_vars, hugging_face_models
from prompts.treatment_summarizer_prompts import treatment_extractor_prompt
from scp_utils.utils import treatment_summary_for_SCP
from scp_utils.scp_utils import create_KB_retriever, generate_cancer_surveillance_plans,save_scps, generate_other_issues
from scp_utils.scp_utils import generate_treatment_effects, generate_helpful_resources, generate_lifestyle_recommend


def treatment_summarizer(patient_note_text,llm_model,temperature=0.0,use_jsonified_patient_data=True):
    
    # load the environment variables
    config = load_env_vars()
    prompt_obj = treatment_extractor_prompt()
    if 'gpt' in llm_model.lower():
        config['AZURE_OPENAI_DEPLOYMENT'] = llm_model
    if 'gpt' not in llm_model.lower():
        hf_model = hugging_face_models(model_name_or_path=llm_model,temperature=temperature,max_tokens=2048,device='cuda')
        temperature = 0.01 # for hugging face models, temperature is set to 0.01
        
    task_list = [
        'diagnosis',
        'is_surgery',
        'surgery',
        'is_radiation_treatment',
        'radiation_treatment',
        'is_systemic_therapy',
        'completed_treatment_agents',
        'is_treatment_side_effects',
        'treatment_side_effects',
        'is_ongoing_treatment',
        'ongoing_treatment',
    ]
    
    # load the patient data
    if use_jsonified_patient_data:
        jsonify_prompt = prompt_obj.get_jsonify_patient_data_prompt()
        jsonify_prompt = jsonify_prompt.replace('[PATIENT_DATA]',patient_note_text)
        
        if 'gpt' in llm_model.lower():
            patient_note_text = azure_open_ai_call(config = config,
                                        prompt=jsonify_prompt,
                                        temperature=temperature)
        else:
            patient_note_text = hf_model.generate_response(task_prompt=jsonify_prompt)
            
    
    treatment_summary_dict = {}
    for task_name in task_list:
        task_prompt = prompt_obj.get_prompt(task_name)
        task_prompt = task_prompt.replace('[PATIENT_DATA]',patient_note_text)
                
        if 'gpt' in llm_model.lower():
            response = azure_open_ai_call(config = config,
                                    prompt=task_prompt,
                                    temperature=temperature)
        else:
            response = hf_model.generate_response(task_prompt=task_prompt)
            
        treatment_summary_dict[task_name] = response
        
    # extract additional comments
    task_prompt = prompt_obj.get_additional_comments_prompt()
    task_prompt = task_prompt.replace('[PATIENT_DATA]',patient_note_text)
    
    if 'gpt' in llm_model.lower():
        response = azure_open_ai_call(config = config,
                                    prompt=task_prompt,
                                    temperature=temperature)
    else:
        response = hf_model.generate_response(task_prompt=task_prompt)
        
    treatment_summary_dict['additional_comments'] = response
    
    return treatment_summary_dict


def generate_SCP(patient_treatment_info,
                 drug_info_kb_path,
                 scp_task_kb_path,
                 llm_model,
                 embedding_model,
                 reranker,
                 is_save=True,
                 save_path='./scp_results',
                 device='cuda'):
    
    # load the environment variables
    config = load_env_vars()
    config['AZURE_OPENAI_DEPLOYMENT'] = llm_model
    if 'gpt' not in llm_model:
        hf_model = hugging_face_models(model_name_or_path=llm_model,temperature=temperature,max_tokens=2048,device='cuda')
        temperature = 0.01 # for hugging face models, temperature is set to 0.01
    
    # retrieval parameters
    top_KB_k = 30
    
    ############# Load the knowledge bases ################
    print('Loading the knowledge bases...')
    # drug info
    drug_names_list = os.listdir(drug_info_kb_path)
    drug_names_list = [i.split('.json')[0] for i in drug_names_list]
    print(f'Number of drug names: {len(drug_names_list)}')
    # cross encoder for drug matching
    cross_encoder = CrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-12-v2', device=device)
    
    
    # create retriever for each knowledge base
    cancer_test_retriever = create_KB_retriever(os.path.join(scp_task_kb_path,'Cancer surveillance and other recommended tests for cancer monitoring'),reranker,top_KB_k)
    treatment_effects_retriever = create_KB_retriever(os.path.join(scp_task_kb_path,'Possible late and long-term effects of cancer treatment'),reranker,top_KB_k)
    other_issues_retriever = create_KB_retriever(os.path.join(scp_task_kb_path,'Possible other issues that cancer survivors may experience'),reranker,top_KB_k)
    lifestyle_retriever = create_KB_retriever(os.path.join(scp_task_kb_path,'Lifestyle and behavior recommendations for cancer survivors'),reranker,top_KB_k)
    helpful_resources_retriever = create_KB_retriever(os.path.join(scp_task_kb_path,'References to helpful resources for cancer survivors'),reranker,top_KB_k)
    

    treatment_summary_json = treatment_summary_for_SCP(patient_treatment_info)
    
    # Improve query. From the treatment summary, extract important entities and provide it in the output separated by newlines.
    care_prompt = """
Treatment Summary:
[TREATMENT_SUMMARY]

**Objective:**
Compress the treatment summary to best describe the patient's treatment history. The purpose is to use this information as a query to retrieve relevant information related to the patient from clinical guidelines and knowledge databases.
"""
    care_prompt = care_prompt.replace('[TREATMENT_SUMMARY]', str(treatment_summary_json))
    
    if 'gpt' in llm_model.lower():
        treatment_summary_compressed = azure_open_ai_call(config = config,
                                    prompt=care_prompt,
                                    temperature=0.2)
    else:
        treatment_summary_compressed = hf_model.generate_response(task_prompt=care_prompt)
    
    ##### Generate SCPs #####
    # 1. Cancer surveillance and other recommended tests for cancer monitoring
    # 2. Possible late and long-term effects of cancer treatment
    # 3. Possible other issues that cancer survivors may experience
    # 4. Lifestyle and behavior recommendations for cancer survivors
    # 5. References to helpful resources for cancer survivors
    SCP_JSON = {}
    print('Recommending cancer surveillance plans...')
    care_prompt, retrieved_context, drug_info = generate_cancer_surveillance_plans(treatment_summary = treatment_summary_json,
                                                                                treatment_summary_compressed = treatment_summary_compressed,
                                                                                cancer_test_retriever = cancer_test_retriever,
                                                                                cross_encoder = cross_encoder,
                                                                                drug_names_list = drug_names_list,
                                                                                drug_info_path = drug_info_kb_path)

    if 'gpt' in llm_model.lower():
        response = azure_open_ai_call(config = config,
                                    prompt=care_prompt,
                                    temperature=0.2)
    else:
        response = hf_model.generate_response(task_prompt=care_prompt)
    
    SCP_JSON['Cancer surveillance and other recommended tests for cancer monitoring'] = {}
    SCP_JSON['Cancer surveillance and other recommended tests for cancer monitoring']['care_prompt'] = care_prompt
    SCP_JSON['Cancer surveillance and other recommended tests for cancer monitoring']['retrieved_context'] = retrieved_context
    SCP_JSON['Cancer surveillance and other recommended tests for cancer monitoring']['drug_info'] = drug_info
    SCP_JSON['Cancer surveillance and other recommended tests for cancer monitoring']['care_plan'] = response
    
    # save the scp
    if is_save:  
        save_scps(patient_folder = save_path,
              care_prompt = care_prompt,
              retrieved_context = retrieved_context,
              drug_info = drug_info,
              care_plan = response,
              task = 'Cancer surveillance and other recommended tests for cancer monitoring',
              )
    
    print('Recommending possible late and long-term effects of cancer treatment...')
    
    care_prompt, retrieved_context, drug_info, care_prompt_already_experienced = generate_treatment_effects(treatment_summary = treatment_summary_json,
                                                                                                                    treatment_summary_compressed = treatment_summary_compressed,
                                                                                                                    treatment_effects_retriever = treatment_effects_retriever,
                                                                                                                    cross_encoder = cross_encoder,
                                                                                                                    drug_names_list = drug_names_list,
                                                                                                                    drug_info_path = drug_info_kb_path)
    
    if 'gpt' in llm_model.lower():
        response1 = azure_open_ai_call(config = config, prompt = care_prompt_already_experienced,temperature=0.2)
        response2 = azure_open_ai_call(config = config, prompt = care_prompt,temperature=0.2)
    else:
        response1 = hf_model.generate_response(task_prompt=care_prompt_already_experienced)
        response2 = hf_model.generate_response(task_prompt=care_prompt)
    
    SCP_JSON['Already experienced symptoms or side effects of the patient and which drugs might have caused it?'] = response1
    SCP_JSON['Possible late and long-term effects of cancer treatment'] = {}
    SCP_JSON['Possible late and long-term effects of cancer treatment']['care_prompt'] = care_prompt
    SCP_JSON['Possible late and long-term effects of cancer treatment']['retrieved_context'] = retrieved_context
    SCP_JSON['Possible late and long-term effects of cancer treatment']['drug_info'] = drug_info
    SCP_JSON['Possible late and long-term effects of cancer treatment']['care_plan'] = response2
    
    #save response1
    if is_save:
        response1 = {'Already experienced symptoms or side effects of the patient and which drugs might have caused it?' : response1}
        with open(os.path.join(save_path, f'Already experienced symptoms or side effects.json'), 'w') as f:
            json.dump(response1, f)
    
    if is_save:  
        save_scps(patient_folder = save_path,
              care_prompt = care_prompt,
              retrieved_context = retrieved_context,
              drug_info = drug_info,
              care_plan = response2,
              task = 'Possible late and long-term effects of cancer treatment',
              )
    
    print('Suggestions for other issues that cancer survivors may experience...')
    care_prompt, retrieved_context = generate_other_issues(treatment_summary = treatment_summary_json,
                                                        treatment_summary_compressed = treatment_summary_compressed,
                                                        other_issues_retriever = other_issues_retriever)
    
    if 'gpt' in llm_model.lower():
        response = azure_open_ai_call(config = config, prompt = care_prompt,temperature=0.2)
    else:
        response = hf_model.generate_response(task_prompt=care_prompt)
        
    SCP_JSON['Possible other issues that cancer survivors may experience'] = {}
    SCP_JSON['Possible other issues that cancer survivors may experience']['care_prompt'] = care_prompt
    SCP_JSON['Possible other issues that cancer survivors may experience']['retrieved_context'] = retrieved_context
    SCP_JSON['Possible other issues that cancer survivors may experience']['drug_info'] = ''
    SCP_JSON['Possible other issues that cancer survivors may experience']['care_plan'] = response
    
    if is_save:  
        save_scps(patient_folder = save_path,
              care_prompt = care_prompt,
              retrieved_context = retrieved_context,
              drug_info = '',
              care_plan = response,
              task = 'Possible other issues that cancer survivors may experience',
              )
    
    print('Lifestyle and behavior recommendations for cancer survivors...')
    care_prompt, retrieved_context = generate_lifestyle_recommend(treatment_summary = treatment_summary_json,
                                                                    treatment_summary_compressed = treatment_summary_compressed,
                                                                    lifestyle_retriever = lifestyle_retriever)
    
    if 'gpt' in llm_model.lower():
        response = azure_open_ai_call(config = config, prompt = care_prompt,temperature=0.2)
    else:
        response = hf_model.generate_response(task_prompt=care_prompt)
    
    SCP_JSON['Lifestyle and behavior recommendations for cancer survivors'] = {}
    SCP_JSON['Lifestyle and behavior recommendations for cancer survivors']['care_prompt'] = care_prompt
    SCP_JSON['Lifestyle and behavior recommendations for cancer survivors']['retrieved_context'] = retrieved_context
    SCP_JSON['Lifestyle and behavior recommendations for cancer survivors']['drug_info'] = ''
    SCP_JSON['Lifestyle and behavior recommendations for cancer survivors']['care_plan'] = response
    
    if is_save:  
        save_scps(patient_folder = save_path,
              care_prompt = care_prompt,
              retrieved_context = retrieved_context,
              drug_info = '',
              care_plan = response,
              task = 'Lifestyle and behavior recommendations for cancer survivors',
              )
    
    print('References to helpful resources for cancer survivors...')
    care_prompt, retrieved_context = generate_helpful_resources(treatment_summary = treatment_summary_json,
                                                                        treatment_summary_compressed = treatment_summary_compressed,
                                                                        helpful_resources_retriever = helpful_resources_retriever)
    
    if 'gpt' in llm_model.lower():
        response = azure_open_ai_call(config = config, prompt = care_prompt,temperature=0.2)
    else:
        response = hf_model.generate_response(task_prompt=care_prompt)
    
    SCP_JSON['References to helpful resources for cancer survivors'] = {}
    SCP_JSON['References to helpful resources for cancer survivors']['care_prompt'] = care_prompt
    SCP_JSON['References to helpful resources for cancer survivors']['retrieved_context'] = retrieved_context
    SCP_JSON['References to helpful resources for cancer survivors']['drug_info'] = ''
    SCP_JSON['References to helpful resources for cancer survivors']['care_plan'] = response
    
    if is_save:  
        save_scps(patient_folder = save_path,
                care_prompt = care_prompt,
                retrieved_context = retrieved_context,
                drug_info = '',
                care_plan = response,
                task = 'References to helpful resources for cancer survivors',
                )
    
    print('SCP Generation Done!')
    
    if is_save: # save the treatment summary
        try:
            treatment_summary_json = json.dumps(treatment_summary_json, indent=4)
            treatment_summary_json = json.loads(treatment_summary_json)
        except:
#             convert_json_prompt = """Convert to a proper JSON format. Do not change the format.
            try:
                def correct_json_format(data):
                    """
                    Corrects a JSON-like dictionary by converting invalid JSON constructs (e.g., sets) into valid JSON constructs.
                    """
                    def convert_sets_to_lists(obj):
                        # Recursive function to traverse and convert sets to lists
                        if isinstance(obj, dict):
                            return {k: convert_sets_to_lists(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_sets_to_lists(i) for i in obj]
                        elif isinstance(obj, set):  # Convert sets to lists
                            return list(obj)
                        else:
                            return obj

                    corrected_data = convert_sets_to_lists(data)
                    # Return the corrected data as a JSON string
                    return json.dumps(corrected_data, indent=4)
                treatment_summary_json = correct_json_format(treatment_summary_json)
                treatment_summary_json = json.loads(treatment_summary_json)
            except:
                print(treatment_summary_json)
                print('Error in converting treatment summary to json for patient: ')
        
        # save the treatment summary
        with open(os.path.join(save_path, f'treatment_summary.json'), 'w') as f:
            json.dump(treatment_summary_json, f)
            
    return treatment_summary_json,SCP_JSON


if __name__ == '__main__':
    patient_note_text = open('sample_patient_data_24.txt', 'r').read()
    save_path = './scp_results'
    drug_info_kb_path = './kbs/chemodrugs'
    scp_task_kb_path = './kbs/VI_2_text-embedding-3-large'
    llm_model = 'gpt-4o'
    is_save = True
    use_jsonified_patient_data = True
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    load_dotenv()
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    azure_endpoint = os.getenv('OPENAI_API_BASE')
    api_version = os.getenv('OPENAI_API_VERSION')
    device = 'cuda'
    
    embedding_model = AzureOpenAIEmbedding(
                        model='text-embedding-3-large',
                        deployment_name='text-embedding-3-large',
                        api_key=api_key,
                        azure_endpoint=azure_endpoint,
                        api_version=api_version,
                    )
    Settings.embed_model = embedding_model
    
    # reranker model
    reranker = ColbertRerank(
                top_n=20,
                model="colbert-ir/colbertv2.0",
                tokenizer="colbert-ir/colbertv2.0",
                keep_retrieval_score=True,
                )
    
    
    treatment_summary_dict = treatment_summarizer(patient_note_text,llm_model,use_jsonified_patient_data=use_jsonified_patient_data) # sometimes use_jsonified_patient_data=False is better for large OpenAI models
    
    treatment_summary_json,SCP_JSON = generate_SCP(patient_treatment_info = treatment_summary_dict,
                                                    drug_info_kb_path = drug_info_kb_path,
                                                    scp_task_kb_path = scp_task_kb_path,
                                                    llm_model = llm_model,
                                                    embedding_model = embedding_model,
                                                    reranker = reranker,
                                                    is_save = is_save,
                                                    save_path = save_path,
                                                    device = device)