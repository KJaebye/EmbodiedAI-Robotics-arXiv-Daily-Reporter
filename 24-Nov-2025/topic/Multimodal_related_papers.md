# Enhancing Quranic Learning: A Multimodal Deep Learning Approach for Arabic Phoneme Recognition 

**Title (ZH)**: 增强古兰经学习：面向阿拉伯音素识别的多模态深度学习方法 

**Authors**: Ayhan Kucukmanisa, Derya Gelmez, Sukru Selim Calik, Zeynep Hilal Kilimci  

**Link**: [PDF](https://arxiv.org/pdf/2511.17477)  

**Abstract**: Recent advances in multimodal deep learning have greatly enhanced the capability of systems for speech analysis and pronunciation assessment. Accurate pronunciation detection remains a key challenge in Arabic, particularly in the context of Quranic recitation, where subtle phonetic differences can alter meaning. Addressing this challenge, the present study proposes a transformer-based multimodal framework for Arabic phoneme mispronunciation detection that combines acoustic and textual representations to achieve higher precision and robustness. The framework integrates UniSpeech-derived acoustic embeddings with BERT-based textual embeddings extracted from Whisper transcriptions, creating a unified representation that captures both phonetic detail and linguistic context. To determine the most effective integration strategy, early, intermediate, and late fusion methods were implemented and evaluated on two datasets containing 29 Arabic phonemes, including eight hafiz sounds, articulated by 11 native speakers. Additional speech samples collected from publicly available YouTube recordings were incorporated to enhance data diversity and generalization. Model performance was assessed using standard evaluation metrics: accuracy, precision, recall, and F1-score, allowing a detailed comparison of the fusion strategies. Experimental findings show that the UniSpeech-BERT multimodal configuration provides strong results and that fusion-based transformer architectures are effective for phoneme-level mispronunciation detection. The study contributes to the development of intelligent, speaker-independent, and multimodal Computer-Aided Language Learning (CALL) systems, offering a practical step toward technology-supported Quranic pronunciation training and broader speech-based educational applications. 

**Abstract (ZH)**: 近期在多模态深度学习领域的进展极大地增强了语音分析和发音评估系统的能力。阿拉伯语中的准确发音检测仍然是一个关键挑战，特别是在古兰经诵读的背景下，细微的音位差异可以改变意义。为应对这一挑战，本研究提出了一种基于变压器的多模态框架，用于阿拉伯语音位误读检测，该框架结合了声学和文本表示，以实现更高的精度和鲁棒性。该框架将来自UniSpeech的声学嵌入与来自Whisper转录的基于BERT的文本嵌入相结合，创建了一个统一的表示，能够捕捉音位细节和语言上下文。为了确定最有效的整合策略，本研究实现了早期、中期和晚期融合方法，并在包含29个阿拉伯音位（包括8种记诵音位）的两个数据集中进行了评估，这些音位由11名本土说话者发音。此外，还加入了从公开的YouTube录音中收集的额外语音样本，以增强数据多样性与泛化能力。模型性能通过标准评估指标：准确率、精确率、召回率和F1分数进行了评估，从而对融合策略进行了详细的比较。实验结果表明，UniSpeech-BERT多模态配置提供了强有力的结果，并且基于融合的变压器架构对于音位级误读检测是有效的。本研究为智能、独立说话者和多模态计算机辅助语言学习（CALL）系统的开发作出了贡献，为基于技术的古兰经发音训练和更广泛的声音为基础的教育应用提供了实际步骤。 

---
# MusicAIR: A Multimodal AI Music Generation Framework Powered by an Algorithm-Driven Core 

**Title (ZH)**: MusicAIR：基于算法驱动核心的多模态AI音乐生成框架 

**Authors**: Callie C. Liao, Duoduo Liao, Ellie L. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.17323)  

**Abstract**: Recent advances in generative AI have made music generation a prominent research focus. However, many neural-based models rely on large datasets, raising concerns about copyright infringement and high-performance costs. In contrast, we propose MusicAIR, an innovative multimodal AI music generation framework powered by a novel algorithm-driven symbolic music core, effectively mitigating copyright infringement risks. The music core algorithms connect critical lyrical and rhythmic information to automatically derive musical features, creating a complete, coherent melodic score solely from the lyrics. The MusicAIR framework facilitates music generation from lyrics, text, and images. The generated score adheres to established principles of music theory, lyrical structure, and rhythmic conventions. We developed Generate AI Music (GenAIM), a web tool using MusicAIR for lyric-to-song, text-to-music, and image-to-music generation. In our experiments, we evaluated AI-generated music scores produced by the system using both standard music metrics and innovative analysis that compares these compositions with original works. The system achieves an average key confidence of 85%, outperforming human composers at 79%, and aligns closely with established music theory standards, demonstrating its ability to generate diverse, human-like compositions. As a co-pilot tool, GenAIM can serve as a reliable music composition assistant and a possible educational composition tutor while simultaneously lowering the entry barrier for all aspiring musicians, which is innovative and significantly contributes to AI for music generation. 

**Abstract (ZH)**: Recent Advances in Generative AI Have Made Music Generation a Prominent Research Focus: Introducing MusicAIR, an Innovative Multimodal AI Music Generation Framework 

---
# Where Culture Fades: Revealing the Cultural Gap in Text-to-Image Generation 

**Title (ZH)**: 文化差异何以消逝：文本生成图像中的文化差距 

**Authors**: Chuancheng Shi, Shangze Li, Shiming Guo, Simiao Xie, Wenhua Wu, Jingtong Dou, Chao Wu, Canran Xiao, Cong Wang, Zifeng Cheng, Fei Shen, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2511.17282)  

**Abstract**: Multilingual text-to-image (T2I) models have advanced rapidly in terms of visual realism and semantic alignment, and are now widely utilized. Yet outputs vary across cultural contexts: because language carries cultural connotations, images synthesized from multilingual prompts should preserve cross-lingual cultural consistency. We conduct a comprehensive analysis showing that current T2I models often produce culturally neutral or English-biased results under multilingual prompts. Analyses of two representative models indicate that the issue stems not from missing cultural knowledge but from insufficient activation of culture-related representations. We propose a probing method that localizes culture-sensitive signals to a small set of neurons in a few fixed layers. Guided by this finding, we introduce two complementary alignment strategies: (1) inference-time cultural activation that amplifies the identified neurons without backbone fine-tuned; and (2) layer-targeted cultural enhancement that updates only culturally relevant layers. Experiments on our CultureBench demonstrate consistent improvements over strong baselines in cultural consistency while preserving fidelity and diversity. 

**Abstract (ZH)**: 多语言文本到图像（T2I）模型在视觉真实性和语义对齐方面取得了 rapid 进步，并被广泛利用。然而，不同文化背景下生成的图像结果各异：因为语言承载着文化内涵，从多语言提示生成的图像应保持跨语言文化一致性。我们进行了全面分析，发现当前 T2I 模型在多语言提示下往往生成文化中立或偏向英语的结果。对两种代表模型的分析表明，问题并非源于缺少文化知识，而是由于文化相关表示的不足激活。我们提出了一种探针方法，将文化敏感信号定位到少数几层中的少量神经元。基于这一发现，我们介绍了两种补充对齐策略：（1）推理时的文化激活，增强识别出的神经元而无需微调主网络；（2）层靶向文化增强，仅更新与文化相关联的层。CultureBench 实验表明，在保持保真度和多样性的同时，这些策略能够在文化一致性方面提供一致的改进。 

---
