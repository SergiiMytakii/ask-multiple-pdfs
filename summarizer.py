
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from utils.utilites import count_num_tokens
import openai
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser



class Summarizer:
    """
    A class for summarizing PDF documents using OpenAI's ChatGPT engine.

    Attributes:
        None

    Methods:
        summarize_the_pdf:
            Summarizes the content of a PDF file using OpenAI's ChatGPT engine.

        get_llm_response:
            Retrieves the response from the ChatGPT engine for a given prompt.

    Note: Ensure that you have the required dependencies installed and configured, including the OpenAI API key.
    """
    @staticmethod
    def summarize_the_pdf(
        file_dir: str,
        max_final_token: int,
        token_threshold: int,
        gpt_model: str,
        temperature: float,
        character_overlap: int
    ):
        """
        Summarizes the content of a PDF file using OpenAI's ChatGPT engine.

        Args:
            file_dir (str): The path to the PDF file.
            max_final_token (int): The maximum number of tokens in the final summary.
            token_threshold (int): The threshold for token count reduction.
            gpt_model (str): The ChatGPT engine model name.
            temperature (float): The temperature parameter for ChatGPT response generation.
            summarizer_llm_system_role (str): The system role for the summarizer.

        Returns:
            str: The final summarized content.
        """
        docs = []
        docs.extend(PyPDFDirectoryLoader(file_dir).load())
        print(f"Document length: {len(docs)}")
        max_summarizer_output_token = int(
            max_final_token/len(docs)) - token_threshold
        summarizer_llm_system_role = "You are an expert text summarizer. You will receive a text and your task is to summarize and keep all the key information.\
      Keep the maximum length of summary wihin {} number of tokens."
        final_summarizer_llm_system_role = "You are an expert text summarizer. You will receive a text and your task is to summarize and keep all the key information."
        full_summary = ""
        counter = 1
        print("Generating the summary..")
        # if the document has more than one pages
        if len(docs) > 1:
            for i in range(len(docs)):
                # NOTE: This part can be optimized by considering a better technique for creating the prompt. (e.g: lanchain "chunksize" and "chunkoverlap" arguments.)

                if i == 0:  # For the first page
                    prompt = docs[i].page_content + \
                        docs[i+1].page_content[:character_overlap]
                # For pages except the fist and the last one.
                elif i < len(docs)-1:
                    prompt = docs[i-1].page_content[-character_overlap:] + \
                        docs[i].page_content + \
                        docs[i+1].page_content[:character_overlap]
                else:  # For the last page
                    prompt = docs[i-1].page_content[-character_overlap:] + \
                        docs[i].page_content
                summarizer_llm_system_role = summarizer_llm_system_role.format(
                    max_summarizer_output_token)
                print(summarizer_llm_system_role)
            full_summary += Summarizer.get_llm_response(
                gpt_model,
                temperature,
                summarizer_llm_system_role,
                text=prompt
            )
        else:  # if the document has only one page
            full_summary = docs[0].page_content

            print(f"Page {counter} was summarized. ", end="")
            counter += 1
        print("\nFull summary token length:", count_num_tokens(
            full_summary, model=gpt_model))
        print(docs[0].page_content)
        final_summary = Summarizer.get_llm_response(
            gpt_model,
            temperature,
            final_summarizer_llm_system_role,
            text=full_summary
        )
        return final_summary

    @staticmethod
    def get_llm_response(gpt_model: str, temperature: float, llm_system_role: str, text: str):
       
        template = f"{llm_system_role}  \n Text: + {text}  \n Summary:"
        
        prompt = PromptTemplate.from_template(template)


        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        output_parser = StrOutputParser()
        chain = prompt|llm|output_parser

        response = chain.invoke({})

        return response