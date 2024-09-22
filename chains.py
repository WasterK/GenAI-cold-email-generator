import os 
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from dotenv import load_dotenv

load_dotenv()

class Chain:

    def __init__(self):
        self.llm = ChatGroq(temperature=0, api_key = os.environ['GROQ_API_KEY'], model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text:str):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTIONS:
            The scrapped text is from career's page of website.
            Your job is to extract the job postings and return them in JOSN format containing 
            following keys: 'role', 'experience', 'skills', and 'description'.
            only return the valid JSON
            ### VALID JSON (NO PREAMBLE):
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={'page_data': cleaned_text})

        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except:
            raise OutputParserException('Content too big, unable to parse job')
        
        return res if isinstance(res, list) else [res]
    
    def write_mail(self, job: dict, links: list):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
            
            ### INSTRUCTION:
            You are Kiran, a business development executive at KalpSoft. KalpSoft is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of KalpSoft 
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
            Remember you are Kiran, BDE at KalpSoft. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            
            """
        )

        email_chain = prompt_email | self.llm
        res = email_chain.invoke({'job_description': str(job), 'link_list': links})
        return res.content

if __name__ == "__main__":
    print(f'Gorq API Key : ', os.getenv('GORQ_API_KEY'))