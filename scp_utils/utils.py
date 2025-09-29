import re
from collections import namedtuple


DiagnosisEnt = namedtuple("DiagnosisEnt", "CancerType, DiagnosisDate, Stage, MolecularMarkers")
DiagnosisEntNoMarkers = namedtuple("DiagnosisEntNoMarkers", "CancerType, DiagnosisDate, Stage")
IsSurgery = namedtuple("IsSurgery", "IsSurgery")
SurgeryEnt = namedtuple("SurgeryEnt", "Procedure, Date, Location, Findings")
IsRadiationTreatment = namedtuple("IsRadiationTreatment", "IsRadiationTreatment")
RadiationTreatmentEnt = namedtuple("RadiationTreatmentEnt", "Procedure, Date, Location")
IsSystemicTherapy = namedtuple("IsSystemicTherapy", "IsSystemicTherapy")
CompletedTreatmentAgentsEnt = namedtuple("CompletedTreatmentAgentsEnt", "AgentName, EndDate")
IsTreatmentSideEffects = namedtuple("IsTreatmentSideEffects", "IsTreatmentSideEffects")
TreatmentSideEffectsEnt = namedtuple("TreatmentSideEffectsEnt", "SideEffect")
IsOngoingTreatment = namedtuple("IsOngoingTreatment", "IsOngoingTreatment")
OngoingTreatmentEnt = namedtuple("OngoingTreatmentEnt", "Treatment, PlannedDuration, PossibleSideEffects")


def remove_brackets_from_str(input_str):
    # Remove curly braces and any leading/trailing whitespace
    return input_str.strip('{} ').strip("'")


def parse_to_str(parse_namedtuples_out):
    parsed_str = parse_namedtuples_out[0]
    parsed_str += ', '
    for i in range(len(parse_namedtuples_out)):   
        if i!=0: 
            temp = ', '.join(list(parse_namedtuples_out[i]))
            if i == len(parse_namedtuples_out) - 1:
                parsed_str += temp
            else:
                parsed_str += temp + ', '
    return parsed_str

def parse_namedtuples(output, task_name):
    Annotation_to_default_ne_dict = {
            'diagnosis': "unknown",
            'is_surgery': "No",
            'surgery': "unknown",
            'is_radiation_treatment': "No",
            'radiation_treatment': "unknown",
            'is_systemic_therapy': "No",
            'completed_treatment_agents': "unknown",
            'is_treatment_side_effects': "No",
            'treatment_side_effects': "unknown",
            'is_ongoing_treatment': "No",
            'ongoing_treatment': "unknown"
            }


    def replace_unknown_with_set(s):
        # Replace unquoted 'unknown' with {'unknown'}
        pattern = r'(?<![\'"{])\bunknown\b(?![\'"}])'
        s = re.sub(pattern, r"{'unknown'}", s)
        return s
    parsed_outputs = list()

    # Adds newline between any space-separated tuples
    try:
        output = re.sub(r'(\))\s([A-Z]+)', r'\1\n\2', str(output))
    except:
        output = str(output)

    if "\n" in output:
        for cur_output in output.strip().split("\n"):
            cur_output = cur_output.strip()
            # cur_output = _remove_unescaped_quote(cur_output).strip()

            if cur_output == '' or cur_output.upper() == 'N/A' or cur_output.lower().startswith(
                    'no ') or cur_output.lower().startswith('none ') or cur_output.lower() == 'unknown':
                parsed_outputs.append(Annotation_to_default_ne_dict[task_name])
            else:
                try:
                    cur_output = replace_unknown_with_set(cur_output)
                    cur_output = eval(cur_output)

                    # Handle the DiagnosisEnt separately to remove MolecularMarkers
                    if isinstance(cur_output, DiagnosisEnt):
                        cur_output = DiagnosisEntNoMarkers(
                            CancerType=cur_output.CancerType,
                            DiagnosisDate=cur_output.DiagnosisDate,
                            Stage=cur_output.Stage
                        )
                    
                    cur_output = next(iter(cur_output))
                    cur_output = remove_brackets_from_str(str(cur_output))
                    
                    parsed_outputs.append(cur_output)
                except Exception as e:
                    print(e)
                    print("Exception in the output: ", cur_output, "Defaulting to unknown entry.")
                    parsed_outputs.append(Annotation_to_default_ne_dict[task_name])

    else:
        output = output.strip()
        # output = _remove_unescaped_quote(output)
   
        try:
            cur_output = replace_unknown_with_set(output)
            cur_output = eval(cur_output)

            # Handle the DiagnosisEnt separately to remove MolecularMarkers
            if isinstance(cur_output, DiagnosisEnt):
                cur_output = DiagnosisEntNoMarkers(
                    CancerType=cur_output.CancerType,
                    DiagnosisDate=cur_output.DiagnosisDate,
                    Stage=cur_output.Stage
                )

            # Parse by combining all to string
            # cur_output = parse_to_str(cur_output)
            
            # parse only the first element
            # if isinstance(cur_output, set):
            #             cur_output = list(cur_output)
            # cur_output = cur_output[0]
            cur_output = next(iter(cur_output))
            cur_output = remove_brackets_from_str(str(cur_output))
            parsed_outputs.append(cur_output)
        except Exception as e:
            print(e)
            print("Defaulting to unknown due to parsing error")
            parsed_outputs.append(Annotation_to_default_ne_dict[task_name])

    return parsed_outputs




def parse_namedtuples_for_SCP(output, task_name):
    
    map_field_names = {
        'diagnosis': {'CancerType': 'Cancer Type', 'DiagnosisDate': 'Diagnosis Date', 'Stage': 'Cancer Stage', 'MolecularMarkers': 'Molecular Markers'},
        'is_surgery': {'IsSurgery': 'Surgery Conducted (Yes/No)'},
        'surgery': {'Procedure': 'Surgery Procedure', 'Date': 'Surgery Date(s) (year)', 'Location': 'Surgery Location', 'Findings': 'Surgery Findings'},
        'is_radiation_treatment': {'IsRadiationTreatment': 'Radiation Treatment Conducted (Yes/No)'},
        'radiation_treatment': {'Procedure': 'Radiation Treatment Procedure', 'Date': 'Radiation Treatment Date(s) (year)', 'Location': 'Radiation Treatment Location'},
        'is_systemic_therapy': {'IsSystemicTherapy': 'Systemic Therapy Conducted (Chemotherapy, hormonal therapy, other)'},
        'completed_treatment_agents': {'AgentName': 'Agent Name', 'EndDate': 'End Date'},
        'is_treatment_side_effects': {'IsTreatmentSideEffects': 'Persistent Symptoms or Side Effects at Completion of Treatment (Yes/No)'},
        'treatment_side_effects': {'SideEffect': 'Side Effect'},
        'is_ongoing_treatment': {'IsOngoingTreatment': 'Need for Ongoing (Adjuvant) Treatment for Cancer (Yes/No)'},
        'ongoing_treatment': {'Treatment': 'Ongoing Treatment', 'PlannedDuration': 'Planned Duration', 'PossibleSideEffects': 'Possible Side Effects'}        
    }
    
    Annotation_to_default_ne_jsondict = {
        'diagnosis': { "Cancer Type": "", "Diagnosis Date": "", "Cancer Stage": "", "Molecular Markers": ""},
        'is_surgery': { "Surgery Conducted (Yes/No)": "No" },
        'surgery': { "Surgery Procedure": "", "Surgery Date(s) (year)": "", "Surgery Location": "", "Surgery Findings": "" },
        'is_radiation_treatment': { "Radiation Treatment Conducted (Yes/No)": "No" },
        'radiation_treatment': { "Radiation Treatment Procedure": "", "Radiation Treatment Date(s) (year)": "", "Radiation Treatment Location": "" },
        'is_systemic_therapy': { "Systemic Therapy Conducted (Yes/No)": "No" }, 
        'completed_treatment_agents': { "Agent Name": "", "End Date": "" },
        'is_treatment_side_effects': { "Persistent symptoms or side effects at completion of treatment (Yes/No)": "No" },
        'treatment_side_effects': { "Side Effect": "" },
        'is_ongoing_treatment': { "Need for ongoing (adjuvant) treatment for cancer (Yes/No)": "No" },
        'ongoing_treatment': { "Ongoing Treatment": "", "Planned Duration": "", "Possible Side Effects": "" }
    }
    
    parsed_outputs_json = []
    # Function to replace unquoted 'unknown' with {'unknown'}
    def replace_unknown_with_set(s):
        # Replace unquoted 'unknown' with {'unknown'}
        pattern = r'(?<![\'"{])\bunknown\b(?![\'"}])'
        s = re.sub(pattern, r"{'unknown'}", s)
        return s

    
    # Adds newline between any space-separated tuples
    try:
        output = re.sub(r'(\))\s([A-Z]+)', r'\1\n\2', str(output))
    except:
        output = str(output)
        
            
        
    if "\n" in output:
        for cur_output in output.strip().split("\n"):
            cur_output = cur_output.strip()
            # cur_output = _remove_unescaped_quote(cur_output).strip()

            if cur_output == '' or cur_output.upper() == 'N/A' or cur_output.lower().startswith(
                    'no ') or cur_output.lower().startswith('none ') or cur_output.lower() == 'unknown':
                parsed_outputs_json.append(Annotation_to_default_ne_jsondict[task_name])
            else:
                try:
                    cur_output = replace_unknown_with_set(cur_output)
                    cur_output = eval(cur_output)
                    cur_output = dict(cur_output._asdict())
                    for k, v in cur_output.items():
                        if isinstance(v, set) and len(v) == 1:
                            cur_output[k] = next(iter(v))
                    cur_output = {map_field_names[task_name][k]: v for k, v in cur_output.items()}
                    parsed_outputs_json.append(cur_output)
                except Exception as e:
                    print(e)
                    print("Exception in the output: ", cur_output, "Defaulting to unknown entry.")
                    parsed_outputs_json.append(Annotation_to_default_ne_jsondict[task_name])
            
    else:
        output = output.strip()
        # output = _remove_unescaped_quote(output)
   
        try:
            cur_output = replace_unknown_with_set(output)
            cur_output = eval(cur_output)
            cur_output = dict(cur_output._asdict())
            for k, v in cur_output.items():
                        if isinstance(v, set) and len(v) == 1:
                            cur_output[k] = next(iter(v))
            cur_output = {map_field_names[task_name][k]: v for k, v in cur_output.items()}
            parsed_outputs_json.append(cur_output)
        except Exception as e:
            print(e)
            print("Defaulting to unknown due to parsing error")
            parsed_outputs_json.append(Annotation_to_default_ne_jsondict[task_name])
            
    return parsed_outputs_json


    
def treatment_summary_for_SCP(treatment_info):
    
    task_name_map = {
   "diagnosis": "Diagnosis",
   "is_surgery": "Surgery Conducted (Yes/No)",
   "surgery": "Surgery Information",
   "is_radiation_treatment":"Radiation Treatment Conducted (Yes/No)",
   "radiation_treatment": "Radiation Treatment Information",
   "is_systemic_therapy": "Systemic Therapy Conducted (Chemotherapy, hormonal therapy, other)",
   "completed_treatment_agents": "Agents Used in Completed Treatments",
   "is_treatment_side_effects": "Persistent Symptoms or Side Effects at Completion of Treatment (Yes/No)",
   "treatment_side_effects": "Symptoms or Side Effects",
   "is_ongoing_treatment": "Need for Ongoing (Adjuvant) Treatment for Cancer (Yes/No)",
   "ongoing_treatment": "Ongoing Treatment Information",
    }
    
    treatment_info_2 = {}
    for key, value in task_name_map.items():
        parsed_json= parse_namedtuples_for_SCP(treatment_info[key], key)
        
        #if unknown in the parsed_json, then convert that value to ''
        for temp_dict in parsed_json:
            for key_2, value_2 in temp_dict.items():
                if value_2 == 'unknown':
                    temp_dict[key_2] = ''
        treatment_info_2[value] = parsed_json
        
    treatment_info_2['Additional Comments'] = treatment_info['additional_comments']
    
    return treatment_info_2


