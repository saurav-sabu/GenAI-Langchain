# Import necessary modules from langchain and src.helper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import CTransformers

from src.helper import *

# Define the instructional and system prompt delimiters
B_INST , E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n" , "\n<</SYS>>\n\n"

# Define the instruction for the task
instruction = "Convert the following text from English to hindi: \n {text}"

# Combine the system prompt with the default system prompt and the instruction
SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
template = B_INST + SYSTEM_PROMPT + instruction + E_INST

# Create a PromptTemplate instance with the defined template and input variable
prompt = PromptTemplate(template=template,input_variables=["text"])

# Initialize the CTransformers model with the specified model and configuration
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",  # Path to the model file
                    model_type="llama",
                    config={
                        "max_new_tokens":128, # Maximum number of new tokens to generate
                        "temperature":0.3
                    })

# Create an LLMChain instance with the prompt and the model
chain = LLMChain(prompt=prompt,llm=llm)

# Run the chain with the input text and print the result
print(chain.run("My name is Saurav"))


