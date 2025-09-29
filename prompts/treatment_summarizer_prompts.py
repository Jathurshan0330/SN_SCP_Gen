class treatment_extractor_prompt:
    def __init__(self):
          
        self.Annotation_format_dict = {
            'diagnosis': """DiagnosisEnt(CancerType='Type of cancer', DiagnosisDate={'Date of diagnosis'}, Stage={'Cancer stage at diagnosis'}, MolecularMarkers={'Molecular markers if available'})""",            
            'is_surgery': """IsSurgery(IsSurgery='Yes or No based on whether surgery conducted or not')""",
            'surgery': """SurgeryEnt(Procedure='Name of the surgery procedure', Date={'Date conducted'}, Location={'Body site where surgery was conducted'}, Findings={'Findings of the surgery'})""",
            'is_radiation_treatment': """IsRadiationTreatment(IsRadiationTreatment='Yes or No based on whether radiation treatment conducted or not')""",          
            'radiation_treatment': """RadiationTreatmentEnt(Procedure='Name of the radiation treatment procedure', Date={'Date conducted'}, Location={'Body site where radiation treatment was conducted'})""",
            'is_systemic_therapy': """IsSystemicTherapy(IsSystemicTherapy='Yes or No based on whether systemic therapy (chemotherapy, hormonal therapy, other) conducted or not')""",
            'completed_treatment_agents': """CompletedTreatmentAgentsEnt(AgentName='Name of the agent', EndDate={'Date of completion'})""",   
            'is_treatment_side_effects': """IsTreatmentSideEffects(IsTreatmentSideEffects='Yes or No based on whether treatment side effects were observed or not at treatment completion')""",
            'treatment_side_effects': """TreatmentSideEffectsEnt(SideEffect='Name of the side effect')""",  
            'is_ongoing_treatment': """IsOngoingTreatment(IsOngoingTreatment='Yes or No based on whether ongoing (adjuvant) treatment is being conducted or not')""",
            'ongoing_treatment': """OngoingTreatmentEnt(Treatment='Name of the treatment', PlannedDuration={'Planned duration of the treatment'}, PossibleSideEffects={'Possible side effects of the treatment'})"""
        }
        
        self.Annotation_example_dict = {
    'diagnosis': "DiagnosisEnt(CancerType='breast cancer', DiagnosisDate={'06/13/12'}, Stage={'ii'}, MolecularMarkers={'er/pr positive, her2 negative'})",
    'is_surgery': "IsSurgery(IsSurgery='Yes')",
    'surgery': """SurgeryEnt(Procedure='simple mastectomy', Date={'10/25/16'}, Location={'breast'}, Findings={'idc, 3.6 cm, mitotic count was quite elevated at 74/10 hpf, negative, lymphovascular invasion, no dcis, 3, progesterone receptor negative, estrogen receptor negative, her2 positive'})
SurgeryEnt(Procedure='surgery', Date={'unknown'}, Location={'breast'}, Findings={'unknown'})""",
    'is_radiation_treatment': "IsRadiationTreatment(IsRadiationTreatment='Yes')",
    'radiation_treatment': """RadiationTreatmentEnt(Procedure='Intensity-Modulated Radiation Therapy (IMRT)', Date={'2024-04-10'}, Location={'Left Breast'})
RadiationTreatmentEnt(Procedure='3D Conformal Radiation Therapy', Date={'2024-06-01'}, Location={'Right Breast'})""",
    'is_systemic_therapy': "IsSystemicTherapy(IsSystemicTherapy='Yes')",
    'completed_treatment_agents': """CompletedTreatmentAgentsEnt(AgentName='Letrozole', EndDate={'2024-05-20'})
CompletedTreatmentAgentsEnt(AgentName='FOLFIRINOX', EndDate={'2024-07-15'})""",
    'is_treatment_side_effects': "IsTreatmentSideEffects(IsTreatmentSideEffects='Yes')",
    'treatment_side_effects': """TreatmentSideEffectsEnt(SideEffect='fatigue')
TreatmentSideEffectsEnt(SideEffect='nausea')""",
    'is_ongoing_treatment': "IsOngoingTreatment(IsOngoingTreatment='No')",
    'ongoing_treatment': """OngoingTreatmentEnt(Treatment='Trastuzumab', PlannedDuration={'12 months'}, PossibleSideEffects={'Nausea, Fatigue, Heart Problems'})
OngoingTreatmentEnt(Treatment='Nab-Paclitaxel', PlannedDuration={'6 months'}, PossibleSideEffects={'Neuropathy, Hair Loss, Low Blood Counts'})"""
}
        
        self.Annotation_task_prompt_dict = {
    'diagnosis': """Extract information from the patient data provided above about the patient's cancer diagnosis, including:
1. **Cancer Type**: Specify the primary cancer type under discussion.
2. **Diagnosis Date**: Provide the date of diagnosis of the cancer. Prioritize the biopsy date if available; otherwise, use the earliest date of diagnosis referred to by the doctor. If multiple dates are present, provide the earliest one.
3. **Stage**: Extract the stage of cancer at the time of diagnosis.
4. **Molecular Markers**: Identify and list any molecular markers mentioned in the clinical note
If any information is not present, return unknown.
""",

    'is_surgery' : """Extract information from the patient data provided above to determine if the patient has undergone any surgery related to their cancer or not. If no information is available on surgery, return 'No'.
Only provide one response for this task.
""" ,

'surgery'   :  """Extract information from the patient data provided above on if the patient has undergone any surgery related to their cancer. Provide the names of the surgery procedures paired with date conducted, location of the body conducted and findings of the surgery. Include the following details:
1. **Procedure**: Name of the surgery procedure performed.
2. **Date**: Date when the surgery was conducted.
3. **Location**: Body site where the surgery was conducted.
4. **Findings**: Any findings or outcomes noted from the surgery. 
If any information is not present, return unknown. If the patient has not undergone any surgery related to their cancer, return unknown.
""",

    'is_radiation_treatment': """Extract information from the patient data provided above to determine whether radiation treatment was conducted on the patient related to their cancer. If there is no information available about radiation treatment, return 'No'.
Only provide one response for this task.
""",

    'radiation_treatment': """If the patient has undergone any radiation treatment related to their cancer, extract the relevant information from the patient data provided above. Include the following details:
1. **Procedure**: Name of the radiation treatment procedure performed.
2. **Date**: Date when the radiation treatment was conducted.
3. **Location**: Body site where the radiation treatment was administered. 
If any information is not present, return unknown.  If the patient has not undergone any radiation treatment related to their cancer, return unknown.
""",
    'is_systemic_therapy': """Extract information from the patient data provided above to determine if systemic therapy related to the patient's cancer was conducted. If there is no information available about systemic therapy, return 'No'.
Only provide one response for this task.
""" ,

    'completed_treatment_agents': """Extract information from the patient data provided above about all the agents or medications used in completed treatments related to the patient's cancer, including:
1. **Agent Name**: Name of the agent used in the treatment.
2. **End Date**: Date when the treatment was completed.
Only include the agents or medications used in the completed treatments and not ongoing treatments.
If any information is not present, return unknown.
""",

    'is_treatment_side_effects': """Extract information from the patient data provided above to determine if the patient has experienced any side effects related to or because of any completed treatment for their cancer. Only consider the side effects of cancer related completed treatments.
If there is no information available regarding side effects completed treatments, return 'No'. Only provide one response for this task.
""" ,

    'treatment_side_effects': """Extract information from the patient data provided above about all side effects the patient experienced related to or because of any completed treatment for their cancer. Only consider side effects associated with cancer related completed treatments.
If any information is not present, return unknown. 
""",

   'is_ongoing_treatment': """Extract information from the patient data provided above to determine whether the patient is currently undergoing any ongoing (adjuvant) treatment related to their cancer or not. If there is any ongoing (adjuvant) treatments, return 'Yes'. 
Only provide one response for this task.
""" ,

'ongoing_treatment':  """Extract information from the patient data provided above on if the patient is undergoing any ongoing (adjuvant) treatment related to their cancer. Provide the names of the ongoing treatments paired with its planned duration and possible side effects. 
Only extract the information on currently active and ongoing treatments related to cancer. Exclude any information on completed treatments, treatments not related to cancer, and any treatments that are mentioned but not currently active.
Include the following details:
1. **Treatment**: Name(s) of the ongoing treatment(s) the patient is currently undergoing for their cancer.
2. **Planned Duration**: Specify the planned duration for each ongoing treatment, if available.
3. **Possible Side Effects**: List any possible side effects associated with each ongoing treatment, if mentioned.
If any information is not present, return unknown. If the patient is not undergoing any ongoing treatment, return unknown.
"""
}  
         
        self.prompt_template = """
**Patient Data:**
-------------------------
[PATIENT_DATA]
-------------------------

**Objective:**  
[TASK PROMPT]

**Guidelines:**
- Focus exclusively on the primary cancer type under discussion.
- Use only the information provided in the patient data. Do not make assumptions or hallucinate any information.
- Provide the extracted information exactly as it appears in the patient data. Use the exact wording from the patient data without making any modifications or interpretations.
- Do not provide any additional information or explanation in the output.

**Output Format:**
- Please convert the response as namedtuples separated by newlines in the following format:
[OUTPUT FORMAT]
- Example:
[EXAMPLES]

- Answer concisely using the specified format without any additional explanations. If there is no information, return unknown.
"""

        


        self.reflect_prompt_template= """
Please act as an impartial judge and evaluate the response of the first language model.
Assess if the model correctly extracted the information on for the given task based on the patient data.
Check if the extracted information belong to the primary cancer type under discussion. Additionally, check if the extracted information is relevant to the task prompt and exists in the patient data.

Patient Data as follows:
-------------------------
[PATIENT_DATA]
-------------------------
Task Prompt as follows:
-------------------------
[TASK PROMPT]
-------------------------
Extracted Information:
-------------------------
[EXTRACTED_INFORMATION]
-------------------------

After assesing provide the final and corrected response as namedtuples separated by newlines at the end of your response in the following format:
[OUTPUT FORMAT]
Example:
[EXAMPLES]
"""

        self.final_prompt_template = """
Previous Responses:
-------------------------
[PREVIOUS_RESPONSES]
-------------------------
Based on the scores and explanations provided, filter out the correct and faithful responses. if the extracted information is unknown return 'unknown'.
Please convert the response as namedtuples separated by newlines in the following format:
[OUTPUT FORMAT]
Example:
[EXAMPLES]
Answer as concisely as possible. Use ''' for special quotation characters. Do not return any explanation. if there are no information return 'unknown'.
"""

    def get_prompt(self,task_name):
        edit_prompt = self.prompt_template
        edit_prompt = edit_prompt.replace('[TASK PROMPT]', self.Annotation_task_prompt_dict[task_name])
        edit_prompt = edit_prompt.replace('[OUTPUT FORMAT]', self.Annotation_format_dict[task_name])
        edit_prompt = edit_prompt.replace('[EXAMPLES]', self.Annotation_example_dict[task_name])
        return edit_prompt
    
    def get_additional_comments_prompt(self):
        addi_prompt = """
**Patient Data:**
-------------------------
[PATIENT_DATA]
-------------------------

**Objective:**  
Extract any additional information from the patient data related to their cancer and provide it in one short paragraph.

**Guidelines:**
- Focus exclusively on the primary cancer type under discussion.
- Exclude information related to cancer diagnosis, surgery, radiation treatment, systemic therapy, completed treatment agents, treatment side effects, and ongoing treatments.
- Include information on non-cancer supportive medications or treatments, which are either actively ongoing or completed.
- Use only the information provided in the patient data. Do not make assumptions or generate information not present in the data.
- Provide the extracted information in free-form text format in a short paragraph. 

**Output Format:**
Present the response as free-form text in one short paragraph., summarizing relevant details. 
"""
        return addi_prompt
    
    def get_jsonify_patient_data_prompt(self):
        jsonify_prompt = """
**Patient Data:**
-------------------------
[PATIENT_DATA]
-------------------------

**Objective:**  
Given the patient data provided above, you are required to convert the patient information into a structured JSON format. The JSON format should include all the relevant details from the patient data.
Structure the patient data in any JSON format that you find suitable, ensuring that all the information is included and organized appropriately. 
"""
        return jsonify_prompt
