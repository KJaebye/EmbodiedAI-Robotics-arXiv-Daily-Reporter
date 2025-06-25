# ChordPrompt: Orchestrating Cross-Modal Prompt Synergy for Multi-Domain Incremental Learning in CLIP 

**Title (ZH)**: ChordPrompt： orchestrating 多模态提示协同作用以促进 CLIP 的多域增量学习 

**Authors**: Zhiyuan Wang, Bokui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19608)  

**Abstract**: Continual learning (CL) empowers pre-trained vision-language models to adapt effectively to novel or previously underrepresented data distributions without comprehensive retraining, enhancing their adaptability and efficiency. While vision-language models like CLIP show great promise, they struggle to maintain performance across domains in incremental learning scenarios. Existing prompt learning methods face two main limitations: 1) they primarily focus on class-incremental learning scenarios, lacking specific strategies for multi-domain task incremental learning; 2) most current approaches employ single-modal prompts, neglecting the potential benefits of cross-modal information exchange. To address these challenges, we propose the \ChordPrompt framework, which facilitates a harmonious interplay between visual and textual prompts. \ChordPrompt introduces cross-modal prompts to leverage interactions between visual and textual information. Our approach also employs domain-adaptive text prompts to select appropriate prompts for continual adaptation across multiple domains. Comprehensive experiments on multi-domain incremental learning benchmarks demonstrate that \ChordPrompt outperforms state-of-the-art methods in zero-shot generalization and downstream task performance. 

**Abstract (ZH)**: 连续学习(CL)使预训练的视觉-语言模型能够在无需全面重新培训的情况下有效地适应新的或以前未充分代表的数据分布，从而增强其适应性和效率。尽管像CLIP这样的视觉-语言模型表现出巨大的潜力，但在增量学习场景中它们难以在不同领域保持性能。现有的提示学习方法面临两个主要局限性：1)它们主要集中在类增量学习场景上，缺乏针对多领域任务增量学习的具体策略；2)大多数当前方法使用单模态提示，忽视了跨模态信息交换的潜在好处。为应对这些挑战，我们提出了ChordPrompt框架，该框架促进了视觉和文本提示之间的和谐交互。ChordPrompt引入了跨模态提示以利用视觉和文本信息之间的交互作用。我们的方法还采用了领域自适应文本提示以选择合适的提示进行跨多个领域的持续适应。在多领域增量学习基准上的全面实验表明，ChordPrompt在零样本泛化和下游任务性能上优于现有方法。 

---
# Kling-Foley: Multimodal Diffusion Transformer for High-Quality Video-to-Audio Generation 

**Title (ZH)**: Kling-Foley：多模态扩散变换器在高质量视频转音频生成中的应用 

**Authors**: Jun Wang, Xijuan Zeng, Chunyu Qiang, Ruilong Chen, Shiyao Wang, Le Wang, Wangjing Zhou, Pengfei Cai, Jiahui Zhao, Nan Li, Zihan Li, Yuzhe Liang, Xiaopeng Wang, Haorui Zheng, Ming Wen, Kang Yin, Yiran Wang, Nan Li, Feng Deng, Liang Dong, Chen Zhang, Di Zhang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2506.19774)  

**Abstract**: We propose Kling-Foley, a large-scale multimodal Video-to-Audio generation model that synthesizes high-quality audio synchronized with video content. In Kling-Foley, we introduce multimodal diffusion transformers to model the interactions between video, audio, and text modalities, and combine it with a visual semantic representation module and an audio-visual synchronization module to enhance alignment capabilities. Specifically, these modules align video conditions with latent audio elements at the frame level, thereby improving semantic alignment and audio-visual synchronization. Together with text conditions, this integrated approach enables precise generation of video-matching sound effects. In addition, we propose a universal latent audio codec that can achieve high-quality modeling in various scenarios such as sound effects, speech, singing, and music. We employ a stereo rendering method that imbues synthesized audio with a spatial presence. At the same time, in order to make up for the incomplete types and annotations of the open-source benchmark, we also open-source an industrial-level benchmark Kling-Audio-Eval. Our experiments show that Kling-Foley trained with the flow matching objective achieves new audio-visual SOTA performance among public models in terms of distribution matching, semantic alignment, temporal alignment and audio quality. 

**Abstract (ZH)**: Kling-Foley：一种大规模多模态视频到音频生成模型 

---
