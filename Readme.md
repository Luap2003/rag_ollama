# RAG Ollama
RAG Ollama is a project aimed at implementing a Retrieval-Augmented Generation (RAG) model using the Ollama framework. This project combines the power of retrieval-based methods with generative models to improve the quality and relevance of generated content.

## Features

- Retrieval-Augmented Generation
- Integration with Ollama framework
- High-quality and relevant content generation



## Usage

1. Clone the project
2. Start a ChromaDB
```[bash]
chroma run --host localhost --port 8000                      
```
3. Create a scripts folder where you put you files, it only reads txt files
```
mkdir scripts
```
4. Add you Scripts to the DB
```
python docs_to_db.py
```
5. start asking :)
```
python query.py "what can you tell me about me"
```
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact [luapglaser2@gmail.com](mailto:luapglaser2@gmail.com).
