from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import streamlit as st
import os


os.environ["OPENAI_API_KEY"] = ""


def create_the_quiz_prompt_template():
    """Create the prompt template for the quiz app."""

    template = """
You are an expert quiz maker for technical fields. Let's think step by step and
create a multiple choice quiz questions with {num_questions} questions about the following concept/content: {quiz_context}.

The format of the quiz as the following:
 Questions:
    <Question1>: 
    <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
    
    <Question2>: 
    <a. Answer 1>, <b. Answer 2>, <c. Answer 3>, <d. Answer 4>
    ....
 Answers:
    <Answer1>: <a|b|c|d>
    <Answer2>: <a|b|c|d>
    ....
    Example:
    Questions:
    1. What is the time complexity of a binary search tree?
        a. O(n)
        b. O(log n)
        c. O(n^2)
        d. O(1)
    Answers: 
        1. b

"""
    prompt = PromptTemplate.from_template(template)
    prompt.format(num_questions=5, quiz_context="")

    return prompt


def create_quiz_chain(prompt_template, llm):
    """Creates the chain for the quiz app."""

    return LLMChain(llm=llm, prompt=prompt_template)


def split_questions_answers(quiz_response):
    """Function that splits the questions and answers from the quiz response."""

    questions = quiz_response.split("Answers:")[0]
    answers = quiz_response.split("Answers:")[1]
    return questions, answers


def convert_tolist(qs, ans):
    """ Function that converts the questions and answers strings to a lists of questions and list of answers"""

    questions_list = qs.split('\n\n')
    questions_list = list(filter(None, questions_list))
    answers_list = ans.split('\n')
    answers_list = list(filter(None, answers_list))
    return questions_list, answers_list


def main():
    st.title("AI Powered Quiz Generator")
    st.write("This app generates a quiz based on a given topic using AI.")
    prompt_template = create_the_quiz_prompt_template()
    llm = ChatOpenAI()
    chain = create_quiz_chain(prompt_template, llm)
    context = st.text_area("Enter the quiz topic")
    num_questions = st.number_input("Enter the number of questions", min_value=1, max_value=10, value=5)

    if st.button("Generate Quiz"):

        user_answers = []
        # if 'user_answers' not in st.session_state:
        #     st.session_state.user_answers = []
        quiz_response = chain.run(num_questions=num_questions, quiz_context=context)
        questions, answers = split_questions_answers(quiz_response)
        questions_list, answers_list = convert_tolist(questions, answers)

        try:
            _ = st.session_state.keep_graphics
        except AttributeError:
            st.session_state.keep_graphics = False

        with st.form("Questions"):
            for i, q in enumerate(questions_list):
                st.header(f"Question {i + 1}")
                st.write(q)
                user_answer = st.radio("Select your answer:", options=['a', 'b', 'c', 'd'], key=i)
                user_answers.append(user_answer)
                st.markdown("----")

            submitted = st.form_submit_button("Submit")
            if submitted or st.session_state.keep_graphics:
                st.session_state.keep_graphics = True
                st.write('Quiz Completed')
                score = sum([1 for correct, user in zip(answers_list, user_answers) if correct == user])
                st.write(f"Your score is: {score}/{num_questions}")
                # if st.button("Show Answers"):
                #     st.write(answers_lists)

if __name__ == "__main__":
    main()
