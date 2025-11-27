import time
import os
import yaml
from fetch_web_content import WebContentFetcher
from retrieval import EmbeddingRetriever
from llm_service import OpenRouterService
from text_utils import PromptTemplate

class GPTAnswer:
    TOP_K = 10  # Top K documents to retrieve

    def __init__(self):
        # Load configuration from a YAML file
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.model_name = self.config["model_name"]
        self.api_key = self.config["openrouter_api_key"]
        # Initialize OpenRouter service
        self.llm_service = OpenRouterService(api_key=self.api_key, model_name=self.model_name)

    def _format_reference(self, relevant_docs_list, link_list):
        # Format the references from the retrieved documents for use in the prompt
        # Use the actual number of documents available, not TOP_K
        num_docs = min(len(relevant_docs_list), self.TOP_K)
        reference_url_list = [(relevant_docs_list[i].metadata)['url'] for i in range(num_docs)]
        reference_content_list = [relevant_docs_list[i].page_content for i in range(num_docs)]
        reference_index_list = [link_list.index(link)+1 for link in reference_url_list]
        rearranged_index_list = self._rearrange_index(reference_index_list)

        # Create a formatted string of references
        formatted_reference = "\n"
        for i in range(num_docs):
            formatted_reference += ('Webpage[' + str(rearranged_index_list[i]) + '], url: ' + reference_url_list[i] + ':\n' + reference_content_list[i] + '\n\n\n')
        return formatted_reference

    def _rearrange_index(self, original_index_list):
        # Rearrange indices to ensure they are unique and sequential
        index_dict = {}
        rearranged_index_list = []
        for index in original_index_list:
            if index not in index_dict:
                index_dict.update({index: len(index_dict)+1})
                rearranged_index_list.append(len(index_dict))
            else:
                rearranged_index_list.append(index_dict[index])
        return rearranged_index_list

    def get_answer(self, query, relevant_docs, language, output_format, profile):
        # Generate an answer using OpenRouter API
        template = self.config["template"]
        prompt_template = PromptTemplate(
            input_variables=["profile", "context_str", "language", "query", "format"],
            template=template
        )

        profile = "conscientious researcher" if not profile else profile
        summary_prompt = prompt_template.format(context_str=relevant_docs, language=language, query=query, format=output_format, profile=profile)
        print("\n\nThe message sent to LLM:\n", summary_prompt)
        print("\n\n", "="*30, "LLM's Answer: ", "="*30, "\n")

        # Call OpenRouter API with streaming
        messages = [{"role": "user", "content": summary_prompt}]
        response = self.llm_service.call_openrouter(messages, temperature=0.0, stream=True)

        # Create a response object similar to LangChain's message format
        class AIMessage:
            def __init__(self, content):
                self.content = content

        return AIMessage(response["content"])

# Example usage
if __name__ == "__main__":
    content_processor = GPTAnswer()
    query = "What happened to Silicon Valley Bank"
    output_format = "" # User can specify output format
    profile = "" # User can define the role for LLM

    # Fetch web content based on the query
    web_contents_fetcher = WebContentFetcher(query)
    web_contents, serper_response = web_contents_fetcher.fetch()

    # Retrieve relevant documents using embeddings
    retriever = EmbeddingRetriever()
    relevant_docs_list = retriever.retrieve_embeddings(web_contents, serper_response['links'], query)
    formatted_relevant_docs = content_processor._format_reference(relevant_docs_list, serper_response['links'])
    print(formatted_relevant_docs)

    # Measure the time taken to get an answer from the GPT model
    start = time.time()

    # Generate answer from ChatOpenAI
    ai_message_obj = content_processor.get_answer(query, formatted_relevant_docs, serper_response['language'], output_format, profile)
    answer = ai_message_obj.content + '\n'
    end = time.time()
    print("\n\nGPT Answer time:", end - start, "s")