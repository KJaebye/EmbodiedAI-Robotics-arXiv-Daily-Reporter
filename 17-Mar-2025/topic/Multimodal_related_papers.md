# Unifying 2D and 3D Vision-Language Understanding 

**Title (ZH)**: 统一二维和三维视觉-语言理解 

**Authors**: Ayush Jain, Alexander Swerdlow, Yuzhou Wang, Sergio Arnaud, Ada Martin, Alexander Sax, Franziska Meier, Katerina Fragkiadaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.10745)  

**Abstract**: Progress in 3D vision-language learning has been hindered by the scarcity of large-scale 3D datasets. We introduce UniVLG, a unified architecture for 2D and 3D vision-language understanding that bridges the gap between existing 2D-centric models and the rich 3D sensory data available in embodied systems. Our approach initializes most model weights from pre-trained 2D models and trains on both 2D and 3D vision-language data. We propose a novel language-conditioned mask decoder shared across 2D and 3D modalities to ground objects effectively in both RGB and RGB-D images, outperforming box-based approaches. To further reduce the domain gap between 2D and 3D, we incorporate 2D-to-3D lifting strategies, enabling UniVLG to utilize 2D data to enhance 3D performance. With these innovations, our model achieves state-of-the-art performance across multiple 3D vision-language grounding tasks, demonstrating the potential of transferring advances from 2D vision-language learning to the data-constrained 3D domain. Furthermore, co-training on both 2D and 3D data enhances performance across modalities without sacrificing 2D capabilities. By removing the reliance on 3D mesh reconstruction and ground-truth object proposals, UniVLG sets a new standard for realistic, embodied-aligned evaluation. Code and additional visualizations are available at $\href{this https URL}{this http URL}$. 

**Abstract (ZH)**: 3D视觉语言学习的进步受到大规模3D数据集稀缺性的阻碍。我们介绍了一种名为UniVLG的统一架构，它将现有以2D为中心的模型与体感系统中丰富的3D感官数据联系起来，用于2D和3D视觉语言理解。我们的方法大部分模型权重初始化来自预训练的2D模型，并在2D和3D视觉语言数据上进行训练。我们提出了一种新的语言条件掩码解码器，跨2D和3D模态共享，有效将对象地融入RGB和RGB-D图像中，优于基于框的方法。为了进一步缩小2D和3D之间的领域差距，我们引入了2D到3D提升策略，使UniVLG能够利用2D数据提升3D性能。通过这些创新，我们的模型在多个3D视觉语言接地任务中达到最先进的性能，证明了从2D视觉语言学习转移进展到数据受限的3D领域中的潜力。此外，同时在2D和3D数据上进行训练，在不牺牲2D能力的情况下提高了跨模态的性能。通过去除对3D网格重建和真实物体提案的依赖，UniVLG为现实、体感对齐的评估设定了新的标准。相关代码和附加可视化可以在[此链接](this http URL)找到。 

---
# Compound Expression Recognition via Large Vision-Language Models 

**Title (ZH)**: 基于大型视觉-语言模型的复合表达识别 

**Authors**: Jun Yu, Xilong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11241)  

**Abstract**: Compound Expression Recognition (CER) is crucial for understanding human emotions and improving human-computer interaction. However, CER faces challenges due to the complexity of facial expressions and the difficulty of capturing subtle emotional cues. To address these issues, we propose a novel approach leveraging Large Vision-Language Models (LVLMs). Our method employs a two-stage fine-tuning process: first, pre-trained LVLMs are fine-tuned on basic facial expressions to establish foundational patterns; second, the model is further optimized on a compound-expression dataset to refine visual-language feature interactions. Our approach achieves advanced accuracy on the RAF-DB dataset and demonstrates strong zero-shot generalization on the C-EXPR-DB dataset, showcasing its potential for real-world applications in emotion analysis and human-computer interaction. 

**Abstract (ZH)**: 复合表情识别（CER）对于理解人类情绪和提升人机交互至关重要。然而，由于面部表情的复杂性和微妙情绪线索的捕捉难度，CER 面临挑战。为了解决这些问题，我们提出了一种利用大型视觉-语言模型（LVLMs）的新方法。我们的方法采用两阶段微调过程：首先，预训练的LVLMs在基本面部表情上进行微调以建立基础模式；其次，在复合表情数据集上进一步优化模型以精化视觉-语言特征交互。我们的方法在RAF-DB数据集上实现了高级准确率，并在C-EXPR-DB数据集上展示了强大的零样本泛化能力，展示了其在情绪分析和人机交互中的潜在应用价值。 

---
# Cross-Modal Learning for Music-to-Music-Video Description Generation 

**Title (ZH)**: 音乐到音乐视频描述生成的跨模态学习 

**Authors**: Zhuoyuan Mao, Mengjie Zhao, Qiyu Wu, Zhi Zhong, Wei-Hsiang Liao, Hiromi Wakaki, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2503.11190)  

**Abstract**: Music-to-music-video generation is a challenging task due to the intrinsic differences between the music and video modalities. The advent of powerful text-to-video diffusion models has opened a promising pathway for music-video (MV) generation by first addressing the music-to-MV description task and subsequently leveraging these models for video generation. In this study, we focus on the MV description generation task and propose a comprehensive pipeline encompassing training data construction and multimodal model fine-tuning. We fine-tune existing pre-trained multimodal models on our newly constructed music-to-MV description dataset based on the Music4All dataset, which integrates both musical and visual information. Our experimental results demonstrate that music representations can be effectively mapped to textual domains, enabling the generation of meaningful MV description directly from music inputs. We also identify key components in the dataset construction pipeline that critically impact the quality of MV description and highlight specific musical attributes that warrant greater focus for improved MV description generation. 

**Abstract (ZH)**: 基于音乐的音乐视频生成是一个具有挑战性的任务，由于音乐和视频模态之间的内在差异。强大的文本到视频扩散模型的出现为音乐视频（MV）生成开辟了一条有希望的道路，首先通过解决音乐到MV描述任务，随后利用这些模型进行视频生成。在本研究中，我们聚焦于MV描述生成任务，并提出一个综合的流程，涵盖训练数据构建和多模态模型微调。我们在一个新构建的基于Music4All数据集的音乐到MV描述数据集上微调现有的预训练多模态模型，该数据集整合了音乐和视觉信息。我们的实验结果表明，音乐表示可以有效地映射到文本域，从而能够直接从音乐输入生成有意义的MV描述。我们还识别出数据集构建流程中关键的组成部分，这些组成部分对MV描述的质量有直接影响，并强调了需要更多关注的具体音乐属性，以提高MV描述生成。 

---
# Towards Understanding Graphical Perception in Large Multimodal Models 

**Title (ZH)**: 理解大型多模态模型中的图形感知 

**Authors**: Kai Zhang, Jianwei Yang, Jeevana Priya Inala, Chandan Singh, Jianfeng Gao, Yu Su, Chenglong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10857)  

**Abstract**: Despite the promising results of large multimodal models (LMMs) in complex vision-language tasks that require knowledge, reasoning, and perception abilities together, we surprisingly found that these models struggle with simple tasks on infographics that require perception only. As existing benchmarks primarily focus on end tasks that require various abilities, they provide limited, fine-grained insights into the limitations of the models' perception abilities. To address this gap, we leverage the theory of graphical perception, an approach used to study how humans decode visual information encoded on charts and graphs, to develop an evaluation framework for analyzing gaps in LMMs' perception abilities in charts. With automated task generation and response evaluation designs, our framework enables comprehensive and controlled testing of LMMs' graphical perception across diverse chart types, visual elements, and task types. We apply our framework to evaluate and diagnose the perception capabilities of state-of-the-art LMMs at three granularity levels (chart, visual element, and pixel). Our findings underscore several critical limitations of current state-of-the-art LMMs, including GPT-4o: their inability to (1) generalize across chart types, (2) understand fundamental visual elements, and (3) cross reference values within a chart. These insights provide guidance for future improvements in perception abilities of LMMs. The evaluation framework and labeled data are publicly available at this https URL. 

**Abstract (ZH)**: 尽管大规模多模态模型（LMMs）在需要知识、推理和感知能力的复杂视觉-语言任务中取得了令人鼓舞的结果，但我们惊讶地发现，这些模型在仅需感知能力的图表任务中表现不佳。由于现有基准主要关注需要多种能力的最终任务，它们只能提供有限的关于模型感知能力局限性的细微洞察。为解决这一问题，我们借鉴图形感知理论，该理论用于研究人类如何解码图表和图示中编码的视觉信息，开发了一种分析LMMs在图表中感知能力差距的评估框架。该框架采用自动任务生成和响应评估设计，能够在多种图表类型、视觉元素和任务类型的范围内进行全面且受控的LMMs图形感知测试。我们利用该框架在三个细粒度级别（图表、视觉元素和像素）上评估和诊断了当前最先进的LMMs的感知能力。我们的发现指出了当前最先进的LMMs的几个关键局限性，包括GPT-4o：它们无法（1）跨图表类型进行泛化，（2）理解基本的视觉元素，以及（3）跨参考图表内的值。这些见解为未来改进LMMs的感知能力提供了指导。评估框架和标注数据可在如下网址获取：this https URL。 

---
# Small Vision-Language Models: A Survey on Compact Architectures and Techniques 

**Title (ZH)**: 小型愿景语言模型：紧凑架构与技术综述 

**Authors**: Nitesh Patnaik, Navdeep Nayak, Himani Bansal Agrawal, Moinak Chinmoy Khamaru, Gourav Bal, Saishree Smaranika Panda, Rishi Raj, Vishal Meena, Kartheek Vadlamani  

**Link**: [PDF](https://arxiv.org/pdf/2503.10665)  

**Abstract**: The emergence of small vision-language models (sVLMs) marks a critical advancement in multimodal AI, enabling efficient processing of visual and textual data in resource-constrained environments. This survey offers a comprehensive exploration of sVLM development, presenting a taxonomy of architectures - transformer-based, mamba-based, and hybrid - that highlight innovations in compact design and computational efficiency. Techniques such as knowledge distillation, lightweight attention mechanisms, and modality pre-fusion are discussed as enablers of high performance with reduced resource requirements. Through an in-depth analysis of models like TinyGPT-V, MiniGPT-4, and VL-Mamba, we identify trade-offs between accuracy, efficiency, and scalability. Persistent challenges, including data biases and generalization to complex tasks, are critically examined, with proposed pathways for addressing them. By consolidating advancements in sVLMs, this work underscores their transformative potential for accessible AI, setting a foundation for future research into efficient multimodal systems. 

**Abstract (ZH)**: 小规模视觉语言模型（sVLMs）的出现标志着多模态AI的一项关键进步，能够在资源受限环境中高效处理视觉和文本数据。本文综述了sVLM的发展，对基于变压器、Mamba和混合架构进行了分类，展示了紧凑设计和计算效率方面的创新。讨论了知识蒸馏、轻量级注意力机制和模态预融合等技术，它们通过减少资源需求实现高性能。通过深入分析TinyGPT-V、MiniGPT-4和VL-Mamba等模型，我们确定了准确度、效率和扩展性之间的权衡。对持续存在的挑战，如数据偏差和对复杂任务的泛化能力进行了批判性分析，并提出了应对策略。通过汇总sVLMs的发展成果，本文强调了它们为可访问AI带来的变革潜力，并为高效多模态系统的未来研究奠定了基础。 

---
