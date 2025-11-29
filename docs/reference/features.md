 # Features — Shiksha Setu

Core features

- Text Simplification (FLAN-T5): grade-targeted simplification to make content accessible.
- Translation (IndicTrans2): translation to Indian languages (Hindi, Tamil, Telugu, Bengali, Marathi, etc.).
- NCERT Validation: curriculum alignment scoring based on semantic similarity (BERT-based).
- Text-to-Speech: multilingual audio generation (VITS/Coqui/MMS-TTS).
- RAG Q&A System: indexing and retrieval over uploaded documents using embeddings and pgvector.
- Chunked file uploads and background processing via Celery tasks.

Why these features

- Focus on educational content quality and accessibility for multiple Indian languages and literacy levels.
- Background processing and RAG allow scalable handling of large documents and interactive Q&A.

Limitations

- Large model inference (TTS, FLAN-T5) requires GPU for acceptable latency.
- Local pgvector is suitable for development; for high traffic, a dedicated vector DB (Qdrant) is recommended.

Future improvements

- Add streaming TTS for lower-latency audio playback.
- Integrate model-cost/latency-aware routing (local vs cloud HF inference).
- Add content versioning and diffing to track curriculum updates.

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152) • [LinkedIn](https://www.linkedin.com/in/k-dhiraj-83b025279/)

*Last updated: November 2025*
