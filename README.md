## Project Proposal 16

**Project Title:** Lightweight Multimodal Question Answering for Short Video Clips Using Transformers

**Team Members:**

- Qihang Fu (fuqihang@seas.upenn.edu)
- Ziyang Zhang (sidz314@seas.upenn.edu)
- Yi Zhao (zhaoyi3@seas.upenn.edu)

---

### Abstract
Our project aims to build a multimodal question-answering (QA) system that can respond to questions about short video clips by leveraging visual, audio, and/or textual cues. Using a streamlined transformer-based model, we plan to process selective frames and audio snippets to generate relevant answers efficiently. The project will explore cross-modal fusion and attention mechanisms in transformers, emphasizing lightweight implementation suitable for constrained computational resources.

---

### Background
For implementation, we plan to use Hugging Face Transformers. We will utilize pre-trained models (e.g., T5 or BERT) adapted for multimodal inputs. Our project is based on the following two primary papers:
1. **Multimodal Learning with Transformers: A Survey**

   *Summary:* This survey provides a broad look into transformer-based multimodal learning, covering challenges and potential designs for handling multiple data types.

   *Link:* [IEEE](https://ieeexplore.ieee.org/abstract/document/10123038)

2. **VX2TEXT: End-to-End Learning of Video-Based Text Generation from Multimodal Inputs**

   *Summary:* This paper introduces a differentiable tokenization and generative approach for converting multimodal inputs (video, audio) into text, which we will adapt for QA tasks.

   *Link:* [ArXiv](https://arxiv.org/abs/2101.12059)

---

**Other Related Papers:**

1. **MuRAG: Multimodal Retrieval-Augmented Generator for Open Question Answering over Images and Text**

   *Summary:* MuRAG combines retrieval-augmented generation with multimodal inputs to enhance open-domain question answering capabilities.

   *Link:* [ArXiv](https://arxiv.org/abs/2210.02928)

2. **MIMOQA: Multimodal Input Multimodal Output Question Answering**

   *Summary:* This work proposes a QA system that handles both multimodal inputs and outputs, demonstrating improved cognitive understanding through multimodal responses.

   *Link:* [ACL Anthology](https://aclanthology.org/2021.naacl-main.418/)

3. **FlowVQA: Mapping Multimodal Logic in Visual Question Answering with Flowcharts**

   *Summary:* FlowVQA introduces a benchmark for evaluating visual question-answering models' reasoning abilities using flowcharts as visual contexts.

   *Link:* [ArXiv](https://arxiv.org/abs/2406.19237)

---

**Related Datasets:**

1. **ScienceQA**

   *Description:* A benchmark comprising 21,208 multimodal multiple-choice questions covering diverse science topics, annotated with corresponding lectures and explanations.

   *Link:* [GitHub](https://github.com/lupantech/ScienceQA)

2. **MultiModalQA**

   *Description:* A challenging dataset requiring joint reasoning over text, tables, and images, consisting of 29,918 examples.

   *Link:* [GitHub](https://github.com/allenai/multimodalqa)

3. **SPIQA**

   *Description:* A dataset designed for multimodal question answering on scientific papers, focusing on interpreting complex figures and tables within research articles.

   *Link:* [GitHub](https://github.com/google/spiqa)

4. **MMToM-QA**

   *Description:* A dataset aimed at evaluating models' Theory of Mind capabilities through multimodal question answering tasks.

   *Link:* [ArXiv](https://arxiv.org/abs/2401.08743)

5. **FlowVQA**

   *Description:* A benchmark designed to assess visual question-answering models' reasoning abilities using flowcharts as visual contexts.

   *Link:* [ArXiv](https://arxiv.org/abs/2406.19237)

---

### Project Sketch

1. **Data Preparation**:

   - **Dataset Selection**: Utilize publicly available datasets that provide short video clips accompanied by question-answer pairs.
   - **Keyframe Extraction**: Implement a frame sampling strategy to select keyframes that are most relevant to the questions. This can be achieved by analyzing frame differences or using pre-trained models to detect significant events within the video.
   - **Audio Snippet Extraction**: Segment audio tracks to align with the selected keyframes, ensuring that the audio context matches the visual content.
   - **Content Categorization**: Employ pre-trained classifiers (e.g., ResNet for images, VGGish for audio) to categorize visual and audio content into semantic labels such as actions, objects, and sounds.

2. **Tokenization and Fusion**:

   - **Differentiable Tokenization**: Convert categorized visual and audio elements into embeddings within a shared semantic space. This involves mapping each modality's features into a common vector space, facilitating seamless integration.
   - **Modality Fusion**: Combine visual and audio embeddings using techniques like concatenation or summation to create a unified representation. This fusion process enables the model to process multimodal information cohesively.
   - **Question Embedding**: Encode the input question using a language model (e.g., BERT) to generate a contextual representation that will guide the attention mechanism in subsequent stages.

3. **Cross-Attention QA Model**:

   - **Model Architecture**: Design a transformer-based model incorporating cross-attention layers that allow the question embedding to attend to the fused multimodal representation. This setup enables the model to focus on relevant parts of the video and audio in relation to the question.
   - **Generative Decoder**: Implement a decoder that generates natural language answers based on the attended multimodal context. The decoder will produce coherent and contextually appropriate responses to the input questions.
   - **Training Strategy**: Train the model using a combination of supervised learning with cross-entropy loss and reinforcement learning techniques to optimize answer accuracy and fluency.

4. **Evaluation**:

   - **Performance Metrics**: Assess the model's performance using metrics such as BLEU, METEOR, and ROUGE to evaluate the quality of generated answers in terms of precision, recall, and overall coherence.
   - **Baseline Comparison**: Compare the proposed model's performance against established baselines, including traditional multimodal QA models and unimodal counterparts, to demonstrate the effectiveness of the multimodal approach.
   - **Ablation Studies**: Conduct experiments by systematically removing or altering components of the model (e.g., excluding audio input) to understand the contribution of each modality and component to the overall performance.
   - **Human Evaluation**: Perform qualitative assessments by having human evaluators rate the relevance and correctness of the model's answers, providing insights into areas where the model excels or requires improvement.

---

### Estimated Timeline (4 Weeks)

| Task                                | Estimated Time | Responsible Team Members |
| ----------------------------------- | -------------- | ------------------------ |
| Data Collection & Setup             | 1 week         | Qihang & Ziyang & Yi     |
| Implement Cross-Attention & Decoder | 1 week         | Qihang & Ziyang & Yi     |
| Training & Tuning                   | 1 week         | Qihang & Ziyang & Yi     |
| Evaluation & Reporting              | 1 week         | Qihang & Ziyang & Yi     |

**1. Week One**
- Qihang: Model Design
- Ziyang: Dataset Selection
- Yi: Related Work Summary

---

### Questions for Instructors

1. **Dataset Suggestions**: Are there additional lightweight datasets for video-based QA tasks you would recommend?
2. **Complexity Reduction**: Given limited resources, are there further methods to simplify the transformer model while maintaining QA accuracy?
3. **Evaluation Metrics**: What are the most effective metrics for evaluating multimodal QA accuracy and relevance?
