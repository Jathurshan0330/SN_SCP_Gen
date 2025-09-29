import json
import numpy as np
import os
from llama_index.core import StorageContext, load_index_from_storage


def create_KB_retriever(KB_path,reranker,top_kb_k=50):
    
    storage_context = StorageContext.from_defaults(persist_dir=KB_path)
    
    # load index
    index = load_index_from_storage(storage_context)
    
    kb_retriever = index.as_retriever(
        similarity_top_k=top_kb_k,
        node_postprocessors=[reranker]
    )
        
    return kb_retriever


def extract_completed_treatment_agents(treatment_summary):
    # Extract the names of agents used in completed treatments
    agents = []
    try:
        completed_treatments = treatment_summary['Agents Used in Completed Treatments']
        for treatment in completed_treatments:
            try:
                if treatment['Agent Name'] == '':
                    continue
                agents.append(treatment['Agent Name'])
            except KeyError:
                pass
    except KeyError:
        return []
    return agents


def extract_ongoing_treatment_agents(treatment_summary):
    # Extract the names of agents used in completed treatments
    agents = []
    try:
        completed_treatments = treatment_summary['Ongoing Treatment Information']
        for treatment in completed_treatments:
            try:
                if treatment['Ongoing Treatment'] == '':
                    continue
                agents.append(treatment['Ongoing Treatment'])
            except KeyError:
                pass
    except KeyError:
        return []
        
    return agents


def convert_retrieved_context_to_json(retrieved_context):
    # convert the retrieved context to json
    retrieved_context_json = {}
    for i in range(len(retrieved_context)):
        retrieved_context_json[i] = {'metadata': retrieved_context[i].metadata, 'text': retrieved_context[i].text}
    return retrieved_context_json


def save_scps(patient_folder,care_prompt, retrieved_context, drug_info,care_plan,task):
    if drug_info == '':
        drug_info = 'No drug information used.'
    
    with open(os.path.join(patient_folder, f'retrieved_context_{task}.json'), 'w') as f:
        json.dump({'retrieved_context': retrieved_context, 'drug_info': drug_info, 'care_prompt': care_prompt}, f)
    # save the SCP as text
    with open(os.path.join(patient_folder, f'{task}.txt'), 'w') as f:
        f.write(care_plan)
    # convert the SCP to json
    care_plan = str(care_plan).split('{', 1)[1].strip()
    care_plan = '{'+ care_plan.rsplit('}', 1)[0].strip() + '}'
    try:
        # response = json.dumps(response, indent=4)
        care_plan = json.loads(care_plan)
        # save the SCP as json
        with open(os.path.join(patient_folder, f'{task}.json'), 'w') as f:
            json.dump(care_plan, f)
    except:
        print(f'Error in saving cancer_surveillance as json for patient')


def generate_cancer_surveillance_plans(treatment_summary,treatment_summary_compressed,cancer_test_retriever,cross_encoder,drug_names_list,drug_info_path):
    # Function to generate cancer surveillance plans for the patient

    # extract the names of agents used in completed treatments
    completed_treatment_agents = extract_completed_treatment_agents(treatment_summary)
    # extract the names of agents used in ongoing treatments
    ongoing_treatment_agents = extract_ongoing_treatment_agents(treatment_summary)
    
    # get set of all agents used in treatment
    all_agents = set(completed_treatment_agents + ongoing_treatment_agents)
    
    # for each drug in all agent, get cross_encoder score with each drug name in drug_names_list, then select the top matching drug
    drug_scores = {}
    for agent in all_agents:
        drug_scores[agent] = {}
        #create a batch of cross_encoder inputs
        cross_encoder_input = []
        cross_encoder_input = [[agent,drug_name] for drug_name in drug_names_list]
        #get cross_encoder scores
        scores = cross_encoder.predict(cross_encoder_input)
        # get the top matching drug
        top_score_index = np.argmax(scores)
        top_matching_drug = drug_names_list[top_score_index]
        drug_scores[agent] = top_matching_drug
        
    print('Extracted agents and their corresponding drugs:')
    print(drug_scores)
    
    # get the drug information for each drug
    drug_info = ''
    for agent, drug in drug_scores.items():
        drug_info_file = os.path.join(drug_info_path,drug+'.json')
        with open(drug_info_file) as f:
            drug_data= json.load(f)
            # print(drug_data)
            f.close()
        
        
        drug_info += f'\nDrug: {drug}\n'
        drug_info += 'Description: '+ drug_data["description"]#+ '\n'
        drug_info += 'Side Effects: ' +drug_data['side_effects']#+ '\n'
        drug_info += 'What to monitor during intake: '+ drug_data['monitoring'] + '\n'
        

    
    ## Generate Cancer surveillance and other recommended tests
    response_format = '''
{"Cancer surveillance and other recommended tests for cancer monitoring":[
    {"Test type": < recommended test >,
    "When / how often": < frequency of tests >,
    "Frequency (in weeks)": < if 'When / how often' is fixed provide frequency in weeks >,
    "Explanation": < explanation for the test recommendation >,
    "Retrieved context id": < context id from the retrieved context used for this recommendation >},
    {"Test type":< recommended test >,
    "When / how often":< frequency of tests >,
    "Frequency (in weeks)": < if 'When / how often' is fixed provide frequency in weeks >,
    "Explanation": < explanation for the test recommendation >,
    "Retrieved context id": < context id from the retrieved context used for this recommendation >},
    ...,
    ]
}'''    


    query_for_retriever = f'''
**Patient treatment summary:**
--------------------- 
{treatment_summary_compressed} 
---------------------

**Additional information of the provided treatment and agents used as follows:**
---------------------
{drug_info}
---------------------

**Task:**
Based on the cancer survivor's treatment summary and the agents used in the treatment, recommend a set of tests for cancer surveillance and monitoring during their follow-up care.
For each recommended test, provide the following information:
1. Test type: The type of test recommended for cancer surveillance.
2. When/how often: The frequency of the recommended test.
3. Frequency (in weeks): If the test is recommended at fixed intervals, provide the frequency in weeks.
'''

    care_prompt = f'''
**Patient treatment summary:**
--------------------- 
{treatment_summary_compressed} 
---------------------

**Additional information of the provided treatment and agents used as follows:**
---------------------
{drug_info}
---------------------

[RETRIEVED CONTEXT]

**Task:**
Based on the cancer survivor's treatment summary and the agents used in the treatment, recommend a set of tests for cancer surveillance and monitoring during their follow-up care.
For each recommended test, provide the following information:
1. Test type: The type of test recommended for cancer surveillance.
2. When/how often: The frequency of the recommended test.
3. Frequency (in weeks): If the test is recommended at fixed intervals, provide the frequency in weeks.
4. Explanation: Provide an explanation for the test recommendation.
5. Retrieved context id: Provide the context id from the retrieved context that supports the test recommendation.

**Guidelines:**
1. Do not hallucinate and each recommendation should be supported by the retrieved context from survivorship guidelines.
2. Provide the output in the following JSON format and strictly adhere to it.
3. Only provide the JSON output and do not include any additional text in the output.
4. Do not repeat the same recommendation multiple times.

**Output Format:** 
{response_format} 
'''
             
    
    # retrieve relevant information from the guidelines
    retrieved_context = cancer_test_retriever.retrieve(query_for_retriever)
    retrieved_context_txt = '**Retrieved context from the guidelines:**\n---------------------\n'
    for i in range(len(retrieved_context)):
        retrieved_context_txt += f'context id: {i}\n info: {retrieved_context[i].text}\n'
    retrieved_context_txt += '---------------------\n'
    
    # add retrieved context to the prompt
    care_prompt = care_prompt.replace('[RETRIEVED CONTEXT]', retrieved_context_txt)
    
    retrieved_context  = convert_retrieved_context_to_json(retrieved_context)
    
    return care_prompt, retrieved_context, drug_info


######## For other issues due to cancer  ######

def generate_other_issues(treatment_summary,treatment_summary_compressed,other_issues_retriever):
    # Function to generate other issues that a cancer survivor might face
    
    
        
    response_format = '''
{"Possible other issues that cancer survivors may experience":[
    {"Issue":< suggested possible issue other than treatment effect for the patient >,
    "Explanation": < explanation for the suggestion of the issue >,
    "Retrieved context id": < context id from the retrieved context used for this suggestion >},
    {"Issue":< suggested possible issue other than treatment effect for the patient >,
    "Explanation": < explanation for the suggestion of the issue >,
    "Retrieved context id": < context id from the retrieved context used for this suggestion >},
    ...,
    ]
}'''

    query_for_retriever = f'''
**Patient treatment summary:**
--------------------- 
{treatment_summary_compressed} 
---------------------

**Task:**
Based on the cancer survivor's treatment summary, suggest possible other issues that the patient might face during follow-up care.
For each suggestion, provide the following information: 
1. Issue: Possible issue other than treatment effect the cancer survivor might face during follow-up care.
2. Explanation: Provide reasoning/explanation for the suggestion and specifically mention which patient information and retrieved context information is the reason behind this suggestion.
3. Retrieved context id: Provide the context id from the retrieved context that supports the suggestion.
'''

    care_prompt = f'''
**Patient treatment summary:**
--------------------- 
{treatment_summary_compressed} 
---------------------

[RETRIEVED CONTEXT]

**Task:**
Based on the cancer survivor's treatment summary, suggest possible other issues that the patient might face during follow-up care.
For each suggestion, provide the following information: 
1. Issue: Possible issue other than treatment effect the cancer survivor might face during follow-up care.
2. Explanation: Provide reasoning/explanation for the suggestion and specifically mention which patient information and retrieved context information is the reason behind this suggestion.
3. Retrieved context id: Provide the context id from the retrieved context that supports the suggestion.

**Guidelines:**
1. Do not hallucinate and each suggestion should be supported by the treatment summary and retrieved context from survivorship guidelines.
2. Provide the output in the following JSON format and strictly adhere to it.
3. Only provide the JSON output and do not include any additional text in the output.
4. Do not repeat the same suggestion multiple times.

**Output Format:** 
{response_format} 
'''

    # retrieve relevant information from the guidelines
    retrieved_context = other_issues_retriever.retrieve(query_for_retriever)
    retrieved_context_txt = '**Retrieved context from the guidelines:**\n---------------------\n'
    for i in range(len(retrieved_context)):
        retrieved_context_txt += f'context id: {i}\n info: {retrieved_context[i].text}\n'
    retrieved_context_txt += '---------------------\n'
    
    # add retrieved context to the prompt
    care_prompt = care_prompt.replace('[RETRIEVED CONTEXT]', retrieved_context_txt)
    
    retrieved_context  = convert_retrieved_context_to_json(retrieved_context)
    
    return care_prompt, retrieved_context




######## For lifestyle and behavior recommendations ######

def generate_lifestyle_recommend(treatment_summary,treatment_summary_compressed,lifestyle_retriever):
 
    
    response_format = '''
{"Lifestyle and behavior recommendations for cancer survivors":[
    {"Lifestyle":< recommend lifestyle and behavior for the patient >,
    "Explanation": < explanation for the recommendation >,
    "Retrieved context id": < context id from the retrieved context used for this recommendation >},
    {"Lifestyle":< recommend lifestyle and behavior for the patient >,
    "Explanation": < explanation for the recommendation >,
    "Retrieved context id": < context id from the retrieved context used for this recommendation >},
    ...,
    ]
}'''

    query_for_retriever = f'''
**Patient treatment summary:**
--------------------- 
{treatment_summary_compressed} 
---------------------

**Task:**
Based on the cancer survivor's treatment summary, recommend a number of lifestyle or behaviors that the patient should adopt during follow-up care to improve their health and avoid cancer recurrence.
For each recommendation, provide the following information: 
1. Lifestyle: Recommended lifestyle or behavior for the cancer survivor.
2. Explanation: Provide reasoning/explanation for the recommendation and specifically mention which patient information and retrieved context information is the reason behind this recommendation.
3. Retrieved context id: Provide the context id from the retrieved context that supports the recommendation.
'''

    care_prompt = f'''
**Patient treatment summary:**
--------------------- 
{treatment_summary_compressed} 
---------------------

[RETRIEVED CONTEXT]

**Task:**
Based on the cancer survivor's treatment summary, recommend a number of lifestyle or behaviors that the patient should adopt during follow-up care to improve their health and avoid cancer recurrence.
For each recommendation, provide the following information: 
1. Lifestyle: Recommended lifestyle or behavior for the cancer survivor.
2. Explanation: Provide reasoning/explanation for the recommendation and specifically mention which patient information and retrieved context information is the reason behind this recommendation.
3. Retrieved context id: Provide the context id from the retrieved context that supports the recommendation.

**Guidelines:**
1. Do not hallucinate and each recommendation should be supported by the treatment summary and retrieved context from survivorship guidelines.
2. Provide the output in the following JSON format and strictly adhere to it.
3. Only provide the JSON output and do not include any additional text in the output.
4. Do not repeat the same recommendation multiple times.

**Output Format:** 
{response_format} 
'''

    # retrieve relevant information from the guidelines
    retrieved_context = lifestyle_retriever.retrieve(query_for_retriever)
    retrieved_context_txt = '**Retrieved context from the guidelines:**\n---------------------\n'
    for i in range(len(retrieved_context)):
        retrieved_context_txt += f'context id: {i}\n info: {retrieved_context[i].text}\n'
    retrieved_context_txt += '---------------------\n'
    
    # add retrieved context to the prompt
    care_prompt = care_prompt.replace('[RETRIEVED CONTEXT]', retrieved_context_txt)
    
    retrieved_context  = convert_retrieved_context_to_json(retrieved_context)
    
    return care_prompt, retrieved_context


######## For helpful resources ######

def generate_helpful_resources(treatment_summary,treatment_summary_compressed,helpful_resources_retriever):

    response_format = '''
{"References to helpful resources for cancer survivors":[
    {"Resource":< recommended resource >,
    "Explanation": < explanation for the recommendation >,
    "Retrieved context id": < context id from the retrieved context used for this recommendation >},
    {"Resource":< recommended resource >,
    "Explanation": < explanation for the recommendation >,
    "Retrieved context id": < context id from the retrieved context used for this recommendation >},
    ...,
    ]
}'''

    query_for_retriever = f'''
**Patient treatment summary:**
--------------------- 
{treatment_summary_compressed} 
---------------------

**Task:**
Based on the cancer survivor's treatment summary, recommend a number of helpful resources that the patient can use to improve their health and well-being during follow-up care.
For each recommendation, provide the following information: 
1. Resource: Recommended resource for the cancer survivor.
2. Explanation: Provide reasoning/explanation for the recommendation and specifically mention which patient information and retrieved context information is the reason behind this recommendation.
3. Retrieved context id: Provide the context id from the retrieved context that supports the recommendation.
'''

    care_prompt = f'''
**Patient treatment summary:**
--------------------- 
{treatment_summary_compressed} 
---------------------

[RETRIEVED CONTEXT]

**Task:**
Based on the cancer survivor's treatment summary, recommend a number of helpful resources that the patient can use to improve their health and well-being during follow-up care.
For each recommendation, provide the following information: 
1. Resource: Recommended resource for the cancer survivor.
2. Explanation: Provide reasoning/explanation for the recommendation and specifically mention which patient information and retrieved context information is the reason behind this recommendation.
3. Retrieved context id: Provide the context id from the retrieved context that supports the recommendation.

**Guidelines:**
1. Do not hallucinate and each recommendation should be supported by the treatment summary and retrieved context from survivorship guidelines.
2. Provide the output in the following JSON format and strictly adhere to it.
3. Only provide the JSON output and do not include any additional text in the output.
4. Do not repeat the same recommendation multiple times.

**Output Format:** 
{response_format} 
'''

    # retrieve relevant information from the guidelines
    retrieved_context = helpful_resources_retriever.retrieve(query_for_retriever)
    retrieved_context_txt = '**Retrieved context from the guidelines:**\n---------------------\n'
    for i in range(len(retrieved_context)):
        retrieved_context_txt += f'context id: {i}\n info: {retrieved_context[i].text}\n'
    retrieved_context_txt += '---------------------\n'
    
    # add retrieved context to the prompt
    care_prompt = care_prompt.replace('[RETRIEVED CONTEXT]', retrieved_context_txt)
    
    retrieved_context  = convert_retrieved_context_to_json(retrieved_context)
    
    return care_prompt, retrieved_context



##############################################


#### for Possible late and long-term effects of cancer treatment  #####

def generate_treatment_effects(treatment_summary,treatment_summary_compressed,treatment_effects_retriever,cross_encoder,drug_names_list,drug_info_path):
    # Function to generate possible late and long-term effects of cancer treatment

    
    # extract the names of agents used in completed treatments
    completed_treatment_agents = extract_completed_treatment_agents(treatment_summary)
    # extract the names of agents used in ongoing treatments
    ongoing_treatment_agents = extract_ongoing_treatment_agents(treatment_summary)
    
    # get set of all agents used in treatment
    all_agents = set(completed_treatment_agents + ongoing_treatment_agents)
    
    # for each drug in all agent, get cross_encoder score with each drug name in drug_names_list, then select the top matching drug
    drug_scores = {}
    for agent in all_agents:
        drug_scores[agent] = {}
        #create a batch of cross_encoder inputs
        cross_encoder_input = []
        cross_encoder_input = [[agent,drug_name] for drug_name in drug_names_list]
        #get cross_encoder scores
        scores = cross_encoder.predict(cross_encoder_input)
        # get the top matching drug
        top_score_index = np.argmax(scores)
        top_matching_drug = drug_names_list[top_score_index]
        drug_scores[agent] = top_matching_drug
        
    print('Extracted agents and their corresponding drugs:')
    print(drug_scores)
    
    # get the drug information for each drug
    drug_info = ''
    for agent, drug in drug_scores.items():
        drug_info_file = os.path.join(drug_info_path,drug+'.json')
        with open(drug_info_file) as f:
            drug_data= json.load(f)
            # print(drug_data)
            f.close()
        
        
        drug_info += f'\nDrug: {drug}\n'
        drug_info += 'Description: '+ drug_data["description"]#+ '\n'
        drug_info += 'Side Effects: ' +drug_data['side_effects']#+ '\n'
        drug_info += 'What to monitor during intake: '+ drug_data['monitoring'] + '\n'
        

    ### What are the already experienced symptons and which drugs might have caused it
    care_prompt_already_experienced = f'''
**Patient treatment summary:**
--------------------- 
{treatment_summary_compressed} 
---------------------

**Additional information of the provided treatment and agents used as follows:**
---------------------
{drug_info}
---------------------

**Task:**
Based on the cancer survivor's treatment summary and the agents used in the treatment, identify and provide the symptoms or side effects the patient has already experienced, and which drugs/agents or treatment might have caused it?
Provide the response in short text format.
'''
    
    #### What are the possible late and long-term effects of the treatment
    response_format = '''
{"Possible late and long-term effects of cancer treatment":[
    {"Treatment effect": < possible late or long-term effect >,
    "Explanation": < explanation for the suggested late or long-term effect >,
    "Retrieved context id": < context id from the retrieved context used for this suggestion >},
    {"Treatment effect": < possible late or long-term effect >,
    "Explanation": < explanation for the suggested late or long-term effect >,
    "Retrieved context id": < context id from the retrieved context used for this suggestion >},
    ...,
    ]
}'''

    query_for_retriever = f'''
**Patient treatment summary:**
--------------------- 
{treatment_summary_compressed} 
---------------------

**Additional information of the provided treatment and agents used as follows:**
---------------------
{drug_info}
---------------------

**Task:**
Based on the cancer survivor's treatment summary and the agents used in the treatment, what are the possible late and long-term effects of the treatment that could occur in the future. If the possible side effect is caused by a drug, mention which drug in the reasoning.
For each suggestion, provide the following information:
1. Treatment effect: Possible late or long-term effect of the treatment.
2. Explanation: Provide reasoning/explanation for each suggestion and specifically mention which patient information, drug information and retrieved context information is the reason behind this suggestion.
3. Retrieved context id: Provide the context id from the retrieved context that supports this suggestion.
'''


    care_prompt = f'''
**Patient treatment summary:**
--------------------- 
{treatment_summary_compressed} 
---------------------

**Additional information of the provided treatment and agents used as follows:**
---------------------
{drug_info}
---------------------

[RETRIEVED CONTEXT]

**Task:**
Based on the cancer survivor's treatment summary and the agents used in the treatment, what are the possible late and long-term effects of the treatment that could occur in the future. If the possible side effect is caused by a drug, mention which drug in the reasoning.
For each suggestion, provide the following information:
1. Treatment effect: Possible late or long-term effect of the treatment.
2. Explanation: Provide reasoning/explanation for each suggestion and specifically mention which patient information, drug information and retrieved context information is the reason behind this suggestion.
3. Retrieved context id: Provide the context id from the retrieved context that supports this suggestion.

**Guidelines:**
1. Do not hallucinate and each recommendation should be supported by the retrieved context from survivorship guidelines.
2. Provide the output in the following JSON format and strictly adhere to it.
3. Only provide the JSON output and do not include any additional text in the output.
4. Do not repeat the same suggestion multiple times.

**Output Format:** 
{response_format} 
'''

    
    # retrieve relevant information from the guidelines
    retrieved_context = treatment_effects_retriever.retrieve(query_for_retriever)
    retrieved_context_txt = '**Retrieved context from the guidelines:**\n---------------------\n'
    for i in range(len(retrieved_context)):
        retrieved_context_txt += f'context id: {i}\n info: {retrieved_context[i].text}\n'
    retrieved_context_txt += '---------------------\n'
    
    # add retrieved context to the prompt
    care_prompt = care_prompt.replace('[RETRIEVED CONTEXT]', retrieved_context_txt)
    
    retrieved_context  = convert_retrieved_context_to_json(retrieved_context)
    
    return care_prompt, retrieved_context, drug_info, care_prompt_already_experienced
    
    
    