from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ROUTER_PROMPT_TEMPLATE = "Given an input, decide whether responding to it would require fact based question answering capabilites or it is a part of a general conversation.\
                    Here is the input \n\n {input}.\n\n Say Yes or no. Yes, if you think it would require fact based question answering capabilities, no, otherwise. Explain your answer.\n \
                    Respond with a json with two keys. 'response' and 'explaination'. 'response' should either be 'yes' or 'no."


CONV_PROMPT_TEMPLATE = "You are a helpful bot who can hold a polite conversation with a fellow human. \
                        You will be provided with a history of messages. Based on that you need to form a final \
                        response. Try not to be too wordy.\n\n \n\n \
                        The chat history: {history}\n \
                        Human's last chat: {input}"


RAG_PROMPT_TEMPLATE =  "Answer the user's questions based on the context provided.\n \
         If you don't know the answer just say you dont know. Do not try to come up with something. \n \
         Keep your answer brief and to the point. \n\n \
         The context: {context} \n \
         Human's question: {input}"