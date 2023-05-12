import os
import openai
import gradio as gr
import pandas as pd 
import tiktoken 
import numpy as np

#if you have OpenAI API key as a string, enable the below
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"


embedding_encoding = "cl100k_base"
encoding = tiktoken.get_encoding(embedding_encoding)


start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

prompt = "This is a table summarizing assessments students haven taken, with each record of the table containing a question that a student answered on a particular test.\nEach Curriculum has a set of Learning Standards for each subject taught in every grade.\nThese Learning Standards are then unpacked at the district, school, or teacher level and organized in sequential Units.\nEach Unit is made up of a set of Learning Objectives.\nEach Learning Objective relate to one or more Learning Standards.\nEach question relates to one Learning Standard.\nFinally each question measures a specific cognitive skill as it defined by Bloom's Taxonomy."

def summarize_columns(df):
    summarized_data = []
    for _, row in df.iterrows():
        summarized_text = "; ".join([f"{column}: {row[column]}" for column in df.columns])
        summarized_data.append(summarized_text)
    df['summarized'] = summarized_data
    return df

def get_text_embedding(text, embeddingMode="text-embedding-ada-002"):
    result = openai.Embedding.create(
    model = embeddingMode,
    input = text
    )
    return result["data"][0]["embedding"]

def get_df_embedding(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    return { idx: get_text_embedding(r.summarized) for idx, r in df.iterrows() }

def calculate_vector_similarity(x: list[float], y: list[float]) -> float:
    return np.dot(np.array(x), np.array(y))

def get_docs_with_similarity(query: str, df_embedding: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    query_embedding = get_text_embedding(query)
    
    document_similarities = sorted([
        (calculate_vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in df_embedding.items()
    ], reverse=True)
    
    return document_similarities


encoding = tiktoken.get_encoding("gpt2")
separator_len = len(encoding.encode("\n* "))
def create_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:

    relevant_document_sections = get_docs_with_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    
    for _, section_index in relevant_document_sections:
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > 500:
            break
        chosen_sections.append("\n* " + document_section.summarized. replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
        
    header = "Answer the question as truthfully as possible using the provided context"
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

document_embeddings = None

df1 = None

def createVectorIndex(uploaded_file):
    df = pd.read_csv(uploaded_file.name)
    df = df[['id', 'curriculum', 'organization_name',
             'pb_schools_id', 'school_name', 'academic_year',
             'pb_classes_id', 'class',
             'pb_grade_subjects_id', 'grade_code', 'pb_sections_id', 'class_section',
             'pb_subjects_id', 'subject', 'pb_employees_id', 'teacher_username',
             'teacherfirstname', 'teacherlastname', 'pb_assessments_id',
             'pb_assessments_title', 'from_date', 'to_date',
             'pb_curriculum_questions_id', 'difficulty', 'time', 'question_type',
             'pb_questions_id', 'pb_answers_id', 'is_correct',
             'knowledge_category', 'knowledge_dimension',
             'learning_standard',
             'subject_learning_standard',
             'subject_learning_objective',
             'subject_unit',
             'pb_assessment_attempts_id', 'assessment_id',
             'student_id', 'student_first_name', 'student_last_name', 'family_id',
             'attempt_date', 'is_completed', 'is_final', 'mark', 'can_retake_exam',
             'can_resume_exam', 'remaining_time',
             'pb_student_attempt_answers_duration',
             'pb_student_attempt_answers_answer_text', 'response',
             'correct_student_answer', 'student_index', 'peer_index', 'createdAt',
             'updatedAt']]
    df = summarize_columns(df)
    df["tokens"] = df.summarized.apply(lambda x: len(encoding.encode(x)))
    
    global document_embeddings
    global df1
    document_embeddings = get_df_embedding(df)
    df1 = df

    return('Index saved successfully!')


def openai_create(
    query : str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array]
) -> str:
    prompt = create_prompt(
        query,
        document_embeddings,
        df
    )

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=4000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[" Human:", " AI:"]
    )

    return response.choices[0].text


def chatgpt_clone(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)
    global df1
    global document_embeddings
    output = openai_create(inp, df1, document_embeddings)
    history.append((input, output))
    return history, history


block = gr.Blocks()


with block:
    gr.Markdown("""<h1><center>FerasGPT with OpenAI API & Gradio</center></h1>
    """)
    with gr.Tab("Input Text Document"):
        file_upload = gr.File(label="Upload CSV")
        text_output = gr.Textbox()
        text_button = gr.Button("Build the Bot!!!")
        text_button.click(createVectorIndex, file_upload, text_output)
    with gr.Tab("Knowledge Bot"):
        chatbot = gr.Chatbot()
        message = gr.Textbox(placeholder=prompt)
        state = gr.State()
        submit = gr.Button("SEND")
        submit.click(chatgpt_clone, inputs=[
                    message, state], outputs=[chatbot, state])

block.launch(debug=True)
