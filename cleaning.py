import re, html, json, requests
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets


class CleanData:

    def __init__(self):
        pass


    def apply_template(self, example, question: str = "question", answer: str = "answer", topic: str = None, prompt_message: str = None,
                       mhcd: str = False):
        """
        Process dataset and apply chat template.
        """
        if not prompt_message:
            prompt_message = "You're a professional therapist helping a patient with their problem."
        if topic:
            prompt_message = f"You're a professional therapist helping a patient with their {example[topic]} problem."
        if mhcd:
            prompt_message = example[prompt_message]

        chat = f"<|system|>\n{prompt_message}</s>\n<|user|>\n{example[question]}</s>\n<|assistant|>\n{example[answer]}"

        return {"templateText": chat}
    

    def apply_cleaning(self, dataset: Dataset, columns: list):
        """
        General function to clean any dataset. Uses batching to speed up the process.
        """
        for column in columns:
            dataset = dataset.filter(lambda x: x[column] is not None)
            dataset = dataset.filter(lambda x: x[column] if "living yes" not in x[column].lower() else None)                                                # remove phrases from one therapist that keeps promoting his book
            dataset = dataset.map(lambda x: {column: [html.unescape(y) for y in x[column]]}, batched=True)                                                  # escape HTML codes, i.e. 0x34
            dataset = dataset.map(lambda x: {column: [re.sub(r"<\/?(p|br)>", "\n", (y)) for y in x[column]]}, batched=True)                                 # substitute <p>/<br> HTML tags with a line break (formatting), i.e. <p>
            dataset = dataset.map(lambda x: {column: [re.sub(r"<\/?.*?>", "", (y)) for y in x[column]]}, batched=True)                                      # remove the rest of HTML tags
            dataset = dataset.map(lambda x: {column: [re.sub(r"(?<!w{3})(\.|\?|\!\;)(?=[a-zA-Z])(?!\bnet\b|\bcom\b|\bco\b|\buk\b|\bca\b|\bes\b|\borg\b|\bgov\b|\bedu\b|\bus\b)",
                                                             "\\1 ", (y)) for y in x[column]]}, batched=True)                                               # add a space between punctuation and text, if missing, don't target websites
            dataset = dataset.map(lambda x: {column: [re.sub(r"(\s){2,}", "\\1", (y)) for y in x[column]]}, batched=True)                                   # remove extra \s
            dataset = dataset.map(lambda x: {column: [re.sub(r"^\s|\s(?=:|;|,|\.)", "", (y)) for y in x[column]]}, batched=True)                            # remove beginning \s or spaces before punctuation
        return dataset
    

    # CounselChat functions
    def remove_therapist_signatures_counsel(self, dataset: Dataset):
        """
        Function to remove the Therapists' names off the CounselChat dataset.

        Some therapists' signature is different than their name reflected in the 'therapistName'
        feature. They were scanned easily because, after scanning the dataset, it was fast to see
        they were only four.

        After updating them, they are removed. This is to prevent Thesa from signing using their name.
        """
        # Websites used by therapist in their signatures
        websites = ["www.lifecounselingorlando.com", "www.livingyes.org", "www.psychologyresource.ca", "www.PsychologyResource.ca"]
        for website in websites:
            dataset = dataset.map(lambda x: {"answerText": re.sub(rf"{website}", "", x["answerText"])})

        # Key = 'therapistName': value = signature in 'answerText'
        therapists_name_to_signature = {"Mark Morris, LCSW": "Mark (www.MarkMorrisLCSW.com and www.LivingYes.org)",
                                        "Tamara Powell": "Tamara Powell, LMHC",
                                        "Robin Landwehr, DBH, LPCC, NCC": "Robin J. Landwehr, DBH, LPCC, NCC",
                                        "Karen Keys, LMHC, CASAC, NCC": "Karen Keys, LMHC, CASAC"
                                        }
        # Update therapist names with their signature in the 'therapistName' column 
        for therapist_name, signature in therapists_name_to_signature.items():
            dataset = dataset.map(lambda x: {'therapistName': x['therapistName'].replace(therapist_name, signature)})

        # Delete their names off the 'answerText' column
        dataset = dataset.map(lambda x: {"answerText": re.sub(rf"{x['therapistName']}", "", x['answerText'])})
        
        # These names keep coming up in varying forms, so delete everything afterwards
        dataset = dataset.map(lambda x: {"answerText": re.sub(rf"Robin.*", "", x['answerText'])})                       
        dataset = dataset.map(lambda x: {"answerText": re.sub(rf"~Mark.*", "", x['answerText'])})                       
        dataset = dataset.map(lambda x: {"answerText": re.sub(rf"Allison.*", "", x['answerText'])})                      

        return dataset


    def normalize_counselchat_topics(self, example):
        """
        Function to normalize CounselChat's topic feature.
        - 'CounselChat' has some topics with None, plus some additional spaces and uppercase letters we want to rid of.
        """
        topic = example["topics"]
        if topic != None:
            topic = re.sub(r",(?!\s)", ", ", topic)     # Substitute +1 spaces for only 1 after a comma
            target = ", "
            if target in topic:                         # For +1 topics, separate with comma + "and" for last topic
                target_index = topic.rfind(target)
                topic = topic[:target_index] + " and" + topic[target_index+1:]
            topic = topic.lower().strip()
        return {"topics": topic}


    # CounselChat dataset
    def clean_counselchat(self) -> Dataset:
        """
        Function to clean the CounselChat dataset.
        """
        dataset = load_dataset("loaiabdalslam/counselchat")

        # Some entries are missing text in the question title or body, marked as None
        # To avoid errors in cleaning, substitute that None for empty string
        dataset = dataset.map(lambda x: {'questionText': "" if x['questionText'] is None else x['questionText']})
        dataset = dataset.map(lambda x: {'questionTitle': "" if x['questionTitle'] is None else x['questionTitle']})
        
        # Merge questionTitle and questionText. Sometimes there's missing context in either one of them, or empty strings
        dataset = dataset.map(lambda x: {'questionText': f"{x['questionTitle']}. {x['questionText']}"})

        dataset = dataset.remove_columns(["questionID", "questionTitle", "questionUrl",
                                          "therapistUrl", "upvotes"])
        
        dataset = dataset.map(self.normalize_counselchat_topics)
        dataset = self.remove_therapist_signatures_counsel(dataset)
        dataset = self.apply_cleaning(dataset, dataset['train'].column_names)
        dataset = dataset.map(self.apply_template, fn_kwargs={"question": "questionText",
                                                              "answer": "answerText",
                                                              "topic": "topics"})
        
        dataset = dataset.remove_columns(["questionText", "answerText", "therapistName", "topics"])

        return dataset['train']


    # MHCD dataset
    def clean_mhcd(self) -> Dataset:
        """
        Function to clean the Mental Health Conversational Data dataset.
        """
        df = pd.DataFrame(columns=["topic", "user", "response"])    # create empty DF with column names

        response = requests.get("https://raw.githubusercontent.com/johnhandleyd/thesa/main/data/intents.json")
        data = json.loads(response.text)

        for sample in data.values():
            for row in sample:
                # the same question can have multiple answers,
                # so we iterate to link every single question to each answer,
                # such that ((q1, a1), (q1, a2), (q1, an))
                for pattern in row['patterns']:
                    for response in row["responses"]:
                        # concatenate to existing DF the new row
                        df = pd.concat([df, pd.DataFrame([{"topic": row["tag"], "user": pattern, "response": response}])], ignore_index=True)
                        
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(self.apply_template, fn_kwargs={"question": "user",
                                                              "answer": "response",
                                                              "prompt_message": "topic",
                                                              "mhcd": "True"})
        dataset = dataset.remove_columns(["topic", "user", "response"])

        return dataset
    

    # Concatenate datasets
    def get_data(self) -> Dataset:
        """
        Returns data as Dataset type, concatenated.
        """
        d1 = self.clean_counselchat()
        d2 = self.clean_mhcd()

        return concatenate_datasets([d1, d2])