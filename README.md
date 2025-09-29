# Survivorship Navigator: Personalized Survivorship Care Plan Generation using Large Language Models (AMIA 2025 Annual Symposium)

<img width="1443" height="694" alt="logo" src="https://github.com/user-attachments/assets/0af97463-e75c-45df-92ca-a5ba160d5a55" />

Cancer survivorship care plans (SCPs) are critical tools for guiding long-term follow-up care of cancer survivors. Yet, their widespread adoption remains hindered by the significant clinician burden and the time- and labor-intensive process of SCP creation. Current practices require clinicians to extract and synthesize treatment summaries from complex patient data, apply relevant survivorship guidelines, and generate a care plan with personalized recommendations, making SCP generation time-consuming. In this study, we systematically explore the potential of large language models (LLMs) for automating SCP generation and introduce Survivorship Navigator, a framework designed to streamline SCP creation and enhance integration with clinical systems. We evaluate our approach through automated assessments and a human expert study, demonstrating that Survivorship Navigator outperforms baseline methods, producing SCPs that are more accurate, guideline-compliant, and actionable.


## 📦 Installation
```
conda env create -f environment.yml
conda activate surv_navigator
```

create a .env file and set the following env variables
```
AZURE_OPENAI_API_KEY = ''
OPENAI_API_BASE =  ''
OPENAI_API_VERSION = ''
OPENAI_API_TYPE = "azure"

OPENAI_API_KEY = ''  # need only to create the knowledge base
```


## 🏃 Survivorship Navigator
Add the patient note to a .txt file, change the path in survivorship_navigator.py, and run the following:
```
python survivorship_navigator.py
```

## 🔍 Create Knowledge Base From Your PDFs
Add the PDFs of the guidelines that should be utilized in SCP creation to one folder, and set the reference_file_path in create_kb.py. The vector knowledge bases will be created in the ./new_kbs folder. The group-wise_separated_knowledge.json is human-readable and can be used to validate the created knowledge bases. 
Note: OpenAI API is required here not Azure OpenAI
```
python create_kb.py
```



## 📚 Cite Survivorship Navigator
If you find our work or this repository useful, please consider giving it a star ⭐ and citing it.
```
@article{pradeepkumar2025survivorship,
  title={Survivorship Navigator: Personalized Survivorship Care Plan Generation using Large Language Models},
  author={Pradeepkumar, Jathurshan and Pankaj Kumar, Shivam and Reamer, Courtney Bryce and Dreyer, Marie and Patel, Jyoti and Liebovitz, David and Sun, Jimeng},
  journal={medRxiv},
  pages={2025--03},
  year={2025},
  publisher={Cold Spring Harbor Laboratory Press}
}
```

We appreciate your interest in our work! 😃😃😃😃😃

