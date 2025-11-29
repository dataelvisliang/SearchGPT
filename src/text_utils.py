"""
Text utility functions to replace LangChain dependencies
"""

class Document:
    """Simple document class to replace LangChain Document"""
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class RecursiveTextSplitter:
    """Text splitter to replace LangChain's RecursiveCharacterTextSplitter"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0, separators: list = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list:
        """Split text into chunks"""
        if not text:
            return []

        # Try each separator in order
        for separator in self.separators:
            # Skip empty separator (handle it separately)
            if separator == "":
                continue

            if separator in text:
                chunks = []
                parts = text.split(separator)
                current_chunk = ""

                for part in parts:
                    # If adding this part would exceed chunk_size, save current chunk
                    if len(current_chunk) + len(part) + len(separator) > self.chunk_size and current_chunk:
                        chunks.append(current_chunk)
                        # Handle overlap
                        if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                            current_chunk = current_chunk[-self.chunk_overlap:] + separator + part
                        else:
                            current_chunk = part
                    else:
                        if current_chunk:
                            current_chunk += separator + part
                        else:
                            current_chunk = part

                # Add the last chunk
                if current_chunk:
                    chunks.append(current_chunk)

                # If we successfully split, return
                if len(chunks) > 1:
                    return chunks

        # If no separator worked, split by character count
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

    def create_documents(self, texts: list, metadatas: list = None) -> list:
        """Create documents from texts with optional metadata"""
        documents = []
        metadatas = metadatas or [{}] * len(texts)

        for text, metadata in zip(texts, metadatas):
            chunks = self.split_text(text)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata=metadata))

        return documents


class PromptTemplate:
    """Simple prompt template to replace LangChain's PromptTemplate"""

    def __init__(self, template: str, input_variables: list = None):
        self.template = template
        self.input_variables = input_variables or []

    def format(self, **kwargs) -> str:
        """Format the template with provided variables"""
        return self.template.format(**kwargs)
