from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os

class LLM:
    def __init__(self, returned_vectors) -> None:
        self.returned_vectors = returned_vectors
    
    def Nvidia_LLM_setup(self):
        context = "\n".join([line_with_distance[0] for line_with_distance in self.returned_vectors])

        client = ChatNVIDIA(
            model="databricks/dbrx-instruct",
            api_key = os.getenv('NVIDIA_API_KEY'),
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
        )

        return context, client


    def LLM(self, question):
        context, client = self.Nvidia_LLM_setup()

        SYSTEM_PROMPT = """
                        Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
                        """
        USER_PROMPT = f"""
                        Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
                        <context>{context}</context>
                        <question>{question}</question>
                        """

        result = []
        for chunk in client.stream([{"role":"user", "content":USER_PROMPT}]):
            print(chunk.content, end="")

        return result