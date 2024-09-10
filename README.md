# AIUniversityLocator

This university finder tool helps user to search for data related to universities, courses offered by university, the admission process, the profiles for professors who teaches at university, alumni information, career paths and employeers.

Each university data is contained in json files which is loaded by JSONLoader, the embeddings are generated and stored in in-memory vector store.

Based on question or query the program queries the vector store and formulate the response using large language model.
