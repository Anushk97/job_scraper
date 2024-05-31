import streamlit as st
import csv
from jobspy import scrape_jobs
import openai
import pandas_gpt
import os
from IPython.display import HTML
from tika import parser
import tika 
tika.initVM()
from tika import parser
from werkzeug.utils import secure_filename
# import PyPDF2
from datetime import datetime
# import textract

st.set_page_config(
        page_title="job search",
)

openai_key = st.sidebar.text_input("Enter your [OpenAI](https://openai.com/index/openai-api/) API key to start:")


def get_pos(prompt, api):
        openai.api_key = api
        if prompt:
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f'From the given prompt, extract the job position that the user is looking for. only return the job position. {prompt}'
                },
            ],
            )
            # print(completion.choices[0].message.content)
            response = completion.choices[0].message.content
            st.write(f' 🤔 looking for {response} roles...')
            return response


def get_location(prompt, api):

        openai.api_key = api
        if prompt:
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f'From the given prompt, extract the location that the user is searching at. only return the location. {prompt}'
                },
            ],
            )
            # print(completion.choices[0].message.content)
            response = completion.choices[0].message.content
            st.write(f' 🤔 finding jobs in {response}!')
            return response
        
def make_clickable(val):
    return f'<a href="{val}" target="_blank">{val}</a>'

def action(prompt, api):

        openai.api_key = api
        if prompt:
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f'From the given prompt,  {prompt}'
                },
            ],
            )
            # print(completion.choices[0].message.content)
            response = completion.choices[0].message.content
            st.write(response)
            return response

class Resume:
    def __init__(self, openai_key):
        self.openai_key = openai_key
        self.upload_folder = 'uploads'
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)

# def resume(file_name, company_name, role_desc, job_pos, type_app):
#     # openai_api = st.sidebar.text_input("Enter your [OpenAI](https://openai.com/index/openai-api/) API key to prompt (optional):")
#     # Set the directory for uploaded files
#     # resume_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
#     if openai_key:
#         UPLOAD_FOLDER = 'uploads'
#         if not os.path.exists(UPLOAD_FOLDER):
#             os.makedirs(UPLOAD_FOLDER)
            
    def extract_pdf_data(self, filename):
        parsed_pdf = parser.from_file(filename)
        return parsed_pdf['content'], parsed_pdf['metadata']
        # text = textract.process(filename)
        # return text

    def generateResumeSummarizationPrompt(self, text):
        return f"Please summarize the following resume:\n\n{text}"
    
    def generateCoverLetterPrompt(self, company_name,user_background, role_description, job_position):
        return f"""Write a succinct and impressive cover letter for the {job_position} position at {company_name}. Make sure to be concise and specific.
    Here is the role description:
    {role_description}
    
    This is my background:
    {user_background}
    """

    def generateWhyUsPrompt(self, company_name,user_background, role_description, job_position):
        return f"""Given the following descriptions, write why I would be a great fit for the ${job_position} position at ${company_name} using first person pronouns. Make it succinct and impressive. You can use the role description and my background as a starting point
    This is the role description:
    {role_description}
    
    This is my background:
    {user_background}
    
    """
        
    def get_ans(self, prompt):
        openai.api_key = self.openai_key
        response = openai.ChatCompletion.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes professional work experience."},
                {'role': 'user', 'content': prompt}
            ],
            model="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=1000
        )
        summary = response['choices'][0]['message']['content'].strip()
        # st.subheader("Summary")
        # st.write(summary)
        return summary
    
    # ans_from_gpt = ''

    def upload_file(self, file_name, company_name, role_desc, job_pos, type_app):
        openai.api_key = self.openai_key
        # st.write(openai.api_key)
        st.subheader("Resume Summarizer")
        uploaded_file = file_name
        # st.write(upload_file)
        
        if uploaded_file is not None:
            
            st.success('Resume uploaded!')
            # Save the uploaded file to local directory
            filename = secure_filename(uploaded_file.name)
            filepath = os.path.join(self.upload_folder, filename)
            
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if os.path.exists(filepath):
                # Parse resume PDF and create background summary
                text, _ = self.extract_pdf_data(filepath)
                # text = self.extract_pdf_data(filepath)
                text = text.strip()

                prompt = self.generateResumeSummarizationPrompt(text)
                ans_from_gpt = self.get_ans(prompt)
                st.write(ans_from_gpt)
                st.markdown("""---""")
                
            
                
                # company_name = 'Google'
                # role_desc = 'ML Engineer'
                # job_pos = 'ML Engineer'
                
                
                # type_app = 'cover-letter'
                
                user_background = ans_from_gpt
                # st.write('user background', user_background)
                company_name = company_name
                role_description = role_desc
                job_position = job_pos
                
                # if st.sidebar.button('search', type='primary'):
                    # st.header(f'generated {type_app}')
                st.success(f'generated {type_app}')
                if type_app == 'cover-letter':
                    prompt = self.generateCoverLetterPrompt(company_name, user_background, role_description, job_position)
                elif type_app == 'why-us':
                    prompt = self.generateWhyUsPrompt(company_name, user_background, role_description, job_position)
                else:
                    raise ValueError("Unknown question type.")

                response = openai.ChatCompletion.create(
                    # messages=[{'role': 'user', 'content': 'Say this is test.'}],  # XXX debugging purpose
                    messages=[
                        {"role": "system", "content": "You are a helpful job application assistant."},
                        {'role': 'user', 'content': prompt}
                    ],
                    model="gpt-3.5-turbo",
                    temperature=0.2,
                    max_tokens=1000
                )
                resp = response['choices'][0]['message']['content'].strip()
                st.write(resp)

        # upload_file(openai_key)
        # st.write(upload)
        # job_match(openai_key)
        
        

def scrape(prompt):
    limit = 50
    position_llm = get_pos(prompt, openai_key)
    pos = position_llm
    location_llm = get_location(prompt, openai_key)
    location = location_llm

    jobs = scrape_jobs(
    site_name=["indeed", "linkedin", "zip_recruiter", "glassdoor"],
    search_term=pos,
    location=location,
    results_wanted=int(limit),
    hours_old=72,
    country_indeed='USA',
    
    # linkedin_fetch_description=True # get full description and direct job url for linkedin (slower)
    # proxies=["208.195.175.46:65095", "208.195.175.45:65095", "localhost"],
    
)
    st.success(f"Found {len(jobs)} jobs")
    first_col = jobs.pop('title')
    second_col = jobs.pop('company')
    third_col = jobs.pop('location')
    jobs.insert(0, 'location', third_col)
    jobs.insert(0, 'company', second_col)
    jobs.insert(0, 'title', first_col)
    jobs.reset_index(drop=True, inplace=True)
    
    jobs_copy = jobs.drop('description', axis=1)
    jobs_copy = jobs_copy.dropna(axis=1, how='all')
    
    jobs_copy['job_url'] = jobs_copy['job_url'].apply(make_clickable)
    return jobs, jobs_copy
    

def action_resume(df, prompt, comp):
    # st.write("DDS", df)
    comp_res = df.ask(f'filter the row with {comp} company')
    # st.write('GG', comp_res)
    return comp_res

def action_print(df, comp):
    comp_res = df.ask(f'filter the row with {comp} company with job url in markdown: {df}')
    # st.write('GG', comp_res)
    # comp_res['job_url'] = comp_res['job_url'].apply(make_clickable)
    st.write(comp_res)
    # st.write(comp_res)

def home():
    st.title('Job search 2024 🚀')
    st.markdown("""---""")
    placeholder = st.empty()
    placeholder.image('./sidebar_img.png')
    timestamp = datetime.now().strftime("%Y/%m/%d")
    
    # openai_api = st.sidebar.text_input("Enter your [OpenAI](https://openai.com/index/openai-api/) API key to prompt (optional):")
    if openai_key:
        
        prompt = st.sidebar.text_area('Search jobs...👇', placeholder="""Help me look for ML jobs in Singapore.""")
        search_button = st.sidebar.button('Search', type='primary')
        query = st.sidebar.text_area('Actions of your dataset', placeholder="""- Which company has the highest salary?\n- Filter out title and location columns\n""")
        st.sidebar.write('or')
        filter_num = st.sidebar.text_area('Filter with index number (comma seperated):', placeholder='1,2,3,4')
        
        
        # action_button = st.sidebar.button('Action', type='primary')
        jobs = None
        # resume_query = st.sidebar.text_area('Query with resume')
        
        
        st.sidebar.markdown("""---""")
        resume_file = st.sidebar.file_uploader("Choose a PDF file for Resume", type="pdf")
        
        # company_name = st.sidebar.text_input('company name: ')
        # role_desc = st.sidebar.text_area('role desc: ')
        # job_pos = st.sidebar.text_input('position: ')
        # type_app = st.sidebar.selectbox('Select type', ['cover-letter', 'why-us'])
        type_app = 'cover-letter'
        
        if search_button:
            placeholder.empty()
            st.header('Scraper 🧹')
            jobs, jobs_copy = scrape(prompt)
            st.session_state.jobs = jobs
            st.session_state.jobs_copy = jobs_copy
            jobs_copy = jobs_copy.sort_values(by=['company'])
            st.write(jobs_copy.to_html(escape=False), unsafe_allow_html=True)
            
            
            st.markdown("""---""")
            st.header('Actions 🤖')
        
        # if filter_num:
        #     if 'jobs' in st.session_state and 'jobs_copy' in st.session_state:
        #         jobs = st.session_state.jobs
        #         filter_num = filter_num.split(',')
        #         filter_nums = [int(x) for x in filter_num]
        #         filtered_df = jobs.loc[filter_nums]
        #         st.write(filtered_df)
            # st.write(filter_nums)
        
        if query or filter_num:
            if 'jobs' in st.session_state and 'jobs_copy' in st.session_state:
                jobs = st.session_state.jobs
                jobs_copy = st.session_state.jobs_copy
                if query:
                    ans = jobs.ask(f'{query} always return all columns of dataframe')
                    ans.drop(['id'], axis = 1)
                    ans = ans.reset_index(drop=True)
                else:
                    filter_num = filter_num.split(',')
                    filter_nums = [int(x) for x in filter_num]
                    ans = jobs.loc[filter_nums]
                    ans.drop(['id'], axis = 1)
                    ans = ans.reset_index(drop=True)
                
                st.write(jobs_copy.to_html(escape=False), unsafe_allow_html=True)
                
                csv = jobs.to_csv(index=False).encode('utf-8')
                
                st.markdown("""---""")
                st.header('Actions 🤖')
                st.write(ans)
                comp = st.text_input('Which company do you want to apply to (enter index from table)?: ')
                
                if comp:
                    filter_comp = comp.split(',')
                    filter_comp = [int(x) for x in filter_comp]
                    ans_new = ans.loc[filter_comp]
                    
                    # action_print(ans, comp)
                    # st.write(res_print)
                    # res_act = action_resume(ans, query, comp)
                    # st.write(res_act)
                    company_name = ans_new['company']
                    # st.success(f'company: {company_name}')
                    role_desc = ans_new['description']
                    # st.success(f'description: {role_desc}')
                    job_pos = ans_new['title']
                    # st.success(f'position: {job_pos}')
                    
                    if resume_file:
                        resume = Resume(openai_key)
                        resume.upload_file(resume_file, company_name, role_desc, job_pos, type_app)
                    else:
                        st.error('please upload your resume')
                    
            else:
                st.write('please perform job search first!')
            
            st.sidebar.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name=f'jobs{timestamp}.csv',
                    mime='text/csv',
                )
        # if resume_button:
        #     def comp_name(prompt, df):
        #         comp_res = jobs.ask(f'extract one company name mentioned in this {prompt} and all the information about it from the {df}. return in dataframe format')
        #         st.write(comp_res)
        #         company_name = comp_res['company']
        #         role_desc = comp_res['description']
        #         job_pos = comp_res['title']
        #         return company_name, role_desc, job_pos
            
        #     if 'jobs' in st.session_state and 'jobs_copy' in st.session_state:
        #         jobs = st.session_state.jobs
        #         jobs_copy = st.session_state.jobs_copy
        #         ans = jobs.ask(query)
        #         st.write(jobs_copy.to_html(escape=False), unsafe_allow_html=True)
        #         st.markdown("""---""")
        #         st.header('Actions 🤖')
        #         st.write(ans)
            
            
        #         company_name, role_desc, job_pos = comp_name(resume_query, ans)
        #         resume(resume_file, company_name, role_desc, job_pos)
        
            
    # print(f"Found {len(jobs)} jobs")
    # print(jobs.head())
    # jobs.to_csv("jobs.csv", quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False) # to_excel
            


# def main():
    
#     PAGES = {
#         "Home": home
        
#     }

#     st.sidebar.title('Navigation')
#     selection = st.sidebar.radio("Go to", list(PAGES.keys()))

#     page = PAGES[selection]
#     page()

if __name__ == "__main__":
    
    home()

