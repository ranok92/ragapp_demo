import ollama 
from langchain_core.prompts import PromptTemplate



class LocalOllama:
    def __init__(self, 
                 model: str, 
                 system: str = '', 
                 format: str = '',
                 temperature: float= 0.5):
        self.model = model
        self.system = system
        self.format = format
        self.temperature = temperature


class LocalOllamaChat(LocalOllama):
    def __init__(self, model, system, format):
        super.__init__(model, system, format)
    
    def format_chat_history(self, langchain_chat):
        chat_history = []
        for chat in langchain_chat:
            if 'AIMessage' in str(type(chat)):
                chat_history.append({'role' : 'assistant', 'content' :chat.content})
            elif 'HumanMessage' in str(type(chat)):
                chat_history.append({'role' : 'user', 'content' :chat.content})

        return chat_history
    

class OllamaChain:

    def __init__(self, llm: LocalOllama, prompt: PromptTemplate):
        self.prompt_template = prompt
        self.localollama = llm 
    
    def invoke(self, input_dict: dict):

        if 'ChatPromptTemplate' == type(self.prompt_template).__name__:
            prompt_text = self.prompt_template.invoke(input_dict).to_string()
        
        if 'PromptTemplate' == type(self.prompt_template).__name__:
            prompt_text = self.prompt_template.invoke(input_dict).text

        response = ollama.generate(model=self.localollama.model, 
                                  system=self.localollama.system, 
                                  format=self.localollama.format, 
                                  prompt = prompt_text,
                                  options={'temperature': self.localollama.temperature}
        )
        return response['response']
