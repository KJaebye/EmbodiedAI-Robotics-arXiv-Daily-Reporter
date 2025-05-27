# ViTaPEs: Visuotactile Position Encodings for Cross-Modal Alignment in Multimodal Transformers 

**Title (ZH)**: ViTaPEs: 视触位置编码在多模态变换器中的跨模态对齐 

**Authors**: Fotios Lygerakis, Ozan Özdenizci, Elmar Rückert  

**Link**: [PDF](https://arxiv.org/pdf/2505.20032)  

**Abstract**: Tactile sensing provides local essential information that is complementary to visual perception, such as texture, compliance, and force. Despite recent advances in visuotactile representation learning, challenges remain in fusing these modalities and generalizing across tasks and environments without heavy reliance on pre-trained vision-language models. Moreover, existing methods do not study positional encodings, thereby overlooking the multi-scale spatial reasoning needed to capture fine-grained visuotactile correlations. We introduce ViTaPEs, a transformer-based framework that robustly integrates visual and tactile input data to learn task-agnostic representations for visuotactile perception. Our approach exploits a novel multi-scale positional encoding scheme to capture intra-modal structures, while simultaneously modeling cross-modal cues. Unlike prior work, we provide provable guarantees in visuotactile fusion, showing that our encodings are injective, rigid-motion-equivariant, and information-preserving, validating these properties empirically. Experiments on multiple large-scale real-world datasets show that ViTaPEs not only surpasses state-of-the-art baselines across various recognition tasks but also demonstrates zero-shot generalization to unseen, out-of-domain scenarios. We further demonstrate the transfer-learning strength of ViTaPEs in a robotic grasping task, where it outperforms state-of-the-art baselines in predicting grasp success. Project page: this https URL 

**Abstract (ZH)**: 触觉感知提供局部的关键信息，这些信息补充了视觉感知的内容，包括纹理、柔顺性和力。尽管在visuotactile表示学习方面取得了近期进展，但在融合这些模态以及在不同任务和环境中泛化方面仍然存在挑战，且对预训练的视觉-语言模型依赖较大。此外，现有方法未研究位置编码，忽视了捕捉细粒度visuotactile相关性的多层次空间推理需求。我们引入了ViTaPEs，这是一种基于Transformer的框架，用于稳健地整合视觉和触觉输入数据，以学习visuotactile感知的任务无关表示。我们的方法利用一种新颖的多层次位置编码方案来捕捉跨模态线索，同时建模跨模态线索。与先前工作不同，我们提供了visuotactile融合的可证明保证，证明我们的编码是注入性的、刚体运动等变的，并且保持信息量，这些性质通过实验得到了验证。在多个大规模真实世界数据集上的实验表明，ViTaPEs不仅在各种识别任务中超越了最先进的基线，还展示了对未见的领域外场景的零样本泛化能力。我们进一步在机器人抓取任务中展示了ViTaPEs的迁移学习能力，它在预测抓取成功率方面优于最先进的基线方法。项目页面：this https URL 

---
# InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning 

**Title (ZH)**: 指令部分：基于指令推理的任务导向部分分割 

**Authors**: Zifu Wan, Yaqi Xie, Ce Zhang, Zhiqiu Lin, Zihan Wang, Simon Stepputtis, Deva Ramanan, Katia Sycara  

**Link**: [PDF](https://arxiv.org/pdf/2505.18291)  

**Abstract**: Large multimodal foundation models, particularly in the domains of language and vision, have significantly advanced various tasks, including robotics, autonomous driving, information retrieval, and grounding. However, many of these models perceive objects as indivisible, overlooking the components that constitute them. Understanding these components and their associated affordances provides valuable insights into an object's functionality, which is fundamental for performing a wide range of tasks. In this work, we introduce a novel real-world benchmark, InstructPart, comprising hand-labeled part segmentation annotations and task-oriented instructions to evaluate the performance of current models in understanding and executing part-level tasks within everyday contexts. Through our experiments, we demonstrate that task-oriented part segmentation remains a challenging problem, even for state-of-the-art Vision-Language Models (VLMs). In addition to our benchmark, we introduce a simple baseline that achieves a twofold performance improvement through fine-tuning with our dataset. With our dataset and benchmark, we aim to facilitate research on task-oriented part segmentation and enhance the applicability of VLMs across various domains, including robotics, virtual reality, information retrieval, and other related fields. Project website: this https URL. 

**Abstract (ZH)**: 大型多模态基础模型，特别是在语言和视觉领域，显著推进了包括机器人技术、自动驾驶、信息检索和语义 grounding 等多种任务的发展。然而，这些模型往往将物体视为不可分割的整体，忽视了组成它们的部件。理解这些部件及其相关的功能属性，能够为物体的功能性提供宝贵见解，这在执行广泛的任务中是至关重要的。在本项工作中，我们引入了一个新的真实世界基准 InstructPart，该基准包含手工标注的部件分割注释和任务导向的指令，以评估当前模型在理解和执行日常情景中的部件级任务方面的性能。通过我们的实验，我们展示了即使对于最先进的视觉-语言模型（VLMs），面向任务的部件分割仍然是一个具有挑战性的问题。此外，我们还引入了一个简单的基线模型，通过与我们数据集的微调实现了性能的两倍提升。借助我们的数据集和基准，我们旨在促进面向任务的部件分割研究，并增强视觉-语言模型在包括机器人技术、虚拟现实、信息检索以及相关领域中的适用性。项目网站: [this https URL]。 

---
# Doc-CoB: Enhancing Multi-Modal Document Understanding with Visual Chain-of-Boxes Reasoning 

**Title (ZH)**: Doc-CoB: 通过视觉链框推理增强多模态文档理解 

**Authors**: Ye Mo, Zirui Shao, Kai Ye, Xianwei Mao, Bo Zhang, Hangdi Xing, Peng Ye, Gang Huang, Kehan Chen, Zhou Huan, Zixu Yan, Sheng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.18603)  

**Abstract**: Multimodal large language models (MLLMs) have made significant progress in document understanding. However, the information-dense nature of document images still poses challenges, as most queries depend on only a few relevant regions, with the rest being redundant. Existing one-pass MLLMs process entire document images without considering query relevance, often failing to focus on critical regions and producing unfaithful responses. Inspired by the human coarse-to-fine reading pattern, we introduce Doc-CoB (Chain-of-Box), a simple-yet-effective mechanism that integrates human-style visual reasoning into MLLM without modifying its architecture. Our method allows the model to autonomously select the set of regions (boxes) most relevant to the query, and then focus attention on them for further understanding. We first design a fully automatic pipeline, integrating a commercial MLLM with a layout analyzer, to generate 249k training samples with intermediate visual reasoning supervision. Then we incorporate two enabling tasks that improve box identification and box-query reasoning, which together enhance document understanding. Extensive experiments on seven benchmarks with four popular models show that Doc-CoB significantly improves performance, demonstrating its effectiveness and wide applicability. All code, data, and models will be released publicly. 

**Abstract (ZH)**: 多模态大规模语言模型在文档理解方面取得了显著进展。然而，文档图像的信息密集特性仍然提出了挑战，大多数查询仅依赖于少数几个相关区域，其余部分则是冗余信息。现有的单次处理多模态大规模语言模型不考虑查询相关性，通常无法聚焦于关键区域，生成不忠实的响应。受人类粗略到精细阅读模式的启发，我们引入了Doc-CoB（Chain-of-Box）机制，这是一种简单有效的做法，将人类风格的视觉推理融入多模态大规模语言模型中而不修改其架构。我们的方法允许模型自主选择与查询最相关的区域集合，并集中注意力进一步理解这些区域。我们首先设计了一个全自动管道，将一个商用多模态大规模语言模型与布局分析器集成，生成249,000个带有中间视觉推理监督的训练样本。然后我们引入了两个增强功能任务，提高了框识别和框查询推理的能力，共同提升了文档理解。在七个基准上的广泛实验显示，Doc-CoB 显著提高了性能，证明了其有效性和广泛应用性。所有代码、数据和模型将公开发布。 

---
# MangaVQA and MangaLMM: A Benchmark and Specialized Model for Multimodal Manga Understanding 

**Title (ZH)**: MangaVQA和MangaLMM：多模态漫画理解的标准基准和专业模型 

**Authors**: Jeonghun Baek, Kazuki Egashira, Shota Onohara, Atsuyuki Miyai, Yuki Imajuku, Hikaru Ikuta, Kiyoharu Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2505.20298)  

**Abstract**: Manga, or Japanese comics, is a richly multimodal narrative form that blends images and text in complex ways. Teaching large multimodal models (LMMs) to understand such narratives at a human-like level could help manga creators reflect on and refine their stories. To this end, we introduce two benchmarks for multimodal manga understanding: MangaOCR, which targets in-page text recognition, and MangaVQA, a novel benchmark designed to evaluate contextual understanding through visual question answering. MangaVQA consists of 526 high-quality, manually constructed question-answer pairs, enabling reliable evaluation across diverse narrative and visual scenarios. Building on these benchmarks, we develop MangaLMM, a manga-specialized model finetuned from the open-source LMM Qwen2.5-VL to jointly handle both tasks. Through extensive experiments, including comparisons with proprietary models such as GPT-4o and Gemini 2.5, we assess how well LMMs understand manga. Our benchmark and model provide a comprehensive foundation for evaluating and advancing LMMs in the richly narrative domain of manga. 

**Abstract (ZH)**: 漫画，或日本漫画，是一种丰富多模态的叙事形式，以复杂的方式结合了图像和文字。训练大规模多模态模型（LMM）以人类水平理解这种叙事可以帮助漫画创作者反思和改进他们的故事。为此，我们引入了两个多模态漫画理解基准：MangaOCR，旨在进行页面内文本识别，以及MangaVQA，这是一种新型基准，旨在通过视觉问答评估上下文理解能力。MangaVQA包含526个高质量的手工构建的问题-答案对，使得在多种叙事和视觉场景中进行可靠评估成为可能。基于这些基准，我们开发了MangaLMM，这是一种从开源LMM Qwen2.5-VL微调的专门用于漫画的模型，旨在同时处理这两个任务。通过广泛的实验，包括与GPT-4o和Gemini 2.5等 propriety模型的比较，我们评估了LMM对漫画的理解能力。我们的基准和模型为评价和促进多模态模型在丰富叙事领域的漫画中提供了全面的基础。 

---
# In-Context Brush: Zero-shot Customized Subject Insertion with Context-Aware Latent Space Manipulation 

**Title (ZH)**: 基于上下文的画笔：情境感知潜在空间操控下的零样本定制主题插入 

**Authors**: Yu Xu, Fan Tang, You Wu, Lin Gao, Oliver Deussen, Hongbin Yan, Jintao Li, Juan Cao, Tong-Yee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.20271)  

**Abstract**: Recent advances in diffusion models have enhanced multimodal-guided visual generation, enabling customized subject insertion that seamlessly "brushes" user-specified objects into a given image guided by textual prompts. However, existing methods often struggle to insert customized subjects with high fidelity and align results with the user's intent through textual prompts. In this work, we propose "In-Context Brush", a zero-shot framework for customized subject insertion by reformulating the task within the paradigm of in-context learning. Without loss of generality, we formulate the object image and the textual prompts as cross-modal demonstrations, and the target image with the masked region as the query. The goal is to inpaint the target image with the subject aligning textual prompts without model tuning. Building upon a pretrained MMDiT-based inpainting network, we perform test-time enhancement via dual-level latent space manipulation: intra-head "latent feature shifting" within each attention head that dynamically shifts attention outputs to reflect the desired subject semantics and inter-head "attention reweighting" across different heads that amplifies prompt controllability through differential attention prioritization. Extensive experiments and applications demonstrate that our approach achieves superior identity preservation, text alignment, and image quality compared to existing state-of-the-art methods, without requiring dedicated training or additional data collection. 

**Abstract (ZH)**: Recent Advances in Diffusion Models Enable Customized Subject Insertion with In-Context Brush Through Textual Prompts 

---
# Multi-modal brain encoding models for multi-modal stimuli 

**Title (ZH)**: 多模态脑编码模型用于多模态刺激 

**Authors**: Subba Reddy Oota, Khushbu Pahwa, Mounika Marreddy, Maneesh Singh, Manish Gupta, Bapi S. Raju  

**Link**: [PDF](https://arxiv.org/pdf/2505.20027)  

**Abstract**: Despite participants engaging in unimodal stimuli, such as watching images or silent videos, recent work has demonstrated that multi-modal Transformer models can predict visual brain activity impressively well, even with incongruent modality representations. This raises the question of how accurately these multi-modal models can predict brain activity when participants are engaged in multi-modal stimuli. As these models grow increasingly popular, their use in studying neural activity provides insights into how our brains respond to such multi-modal naturalistic stimuli, i.e., where it separates and integrates information across modalities through a hierarchy of early sensory regions to higher cognition. We investigate this question by using multiple unimodal and two types of multi-modal models-cross-modal and jointly pretrained-to determine which type of model is more relevant to fMRI brain activity when participants are engaged in watching movies. We observe that both types of multi-modal models show improved alignment in several language and visual regions. This study also helps in identifying which brain regions process unimodal versus multi-modal information. We further investigate the contribution of each modality to multi-modal alignment by carefully removing unimodal features one by one from multi-modal representations, and find that there is additional information beyond the unimodal embeddings that is processed in the visual and language regions. Based on this investigation, we find that while for cross-modal models, their brain alignment is partially attributed to the video modality; for jointly pretrained models, it is partially attributed to both the video and audio modalities. This serves as a strong motivation for the neuroscience community to investigate the interpretability of these models for deepening our understanding of multi-modal information processing in brain. 

**Abstract (ZH)**: 尽管参与者接受的是单一模态刺激，如观看图片或无声视频，最近的研究表明，多模态Transformer模型可以出色地预测视觉脑活动，即使不同模态的表示不一致。这引发了一个问题：当参与者接受多模态刺激时，这些多模态模型能多准确地预测脑活动。随着这类模型变得越来越流行，它们在研究神经活动方面的应用为我们提供了关于大脑如何响应多模态自然刺激的见解，即信息如何通过早期感觉区域向更高级认知区域的层次结构进行分离和整合。我们通过使用多种单一模态和两种类型多模态模型——跨模态模型和联合预训练模型——来确定哪种类型的模型在参与者观看电影时更相关于fMRI脑活动。我们观察到，两种类型的多模态模型在多个语言和视觉区域都显示出更好的对齐。这项研究还有助于确定哪些脑区处理单一模态信息与多模态信息。我们进一步通过仔细地从多模态表示中逐个移除单一模态特征来研究每种模态对多模态对齐的贡献，发现视觉和语言区域中还有额外的未被单一模态嵌入捕捉到的信息。基于这项研究，我们发现对于跨模态模型，其脑对齐部分归因于视频模态；而对于联合预训练模型，其脑对齐则部分归因于视频和音频模态。这为神经科学界深入理解大脑多模态信息处理的可解释性提供了强有力的动力。 

---
# ReasonPlan: Unified Scene Prediction and Decision Reasoning for Closed-loop Autonomous Driving 

**Title (ZH)**: ReasonPlan: 统一的场景预测与决策推理在闭环自动驾驶中的应用 

**Authors**: Xueyi Liu, Zuodong Zhong, Yuxin Guo, Yun-Fu Liu, Zhiguo Su, Qichao Zhang, Junli Wang, Yinfeng Gao, Yupeng Zheng, Qiao Lin, Huiyong Chen, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.20024)  

**Abstract**: Due to the powerful vision-language reasoning and generalization abilities, multimodal large language models (MLLMs) have garnered significant attention in the field of end-to-end (E2E) autonomous driving. However, their application to closed-loop systems remains underexplored, and current MLLM-based methods have not shown clear superiority to mainstream E2E imitation learning approaches. In this work, we propose ReasonPlan, a novel MLLM fine-tuning framework designed for closed-loop driving through holistic reasoning with a self-supervised Next Scene Prediction task and supervised Decision Chain-of-Thought process. This dual mechanism encourages the model to align visual representations with actionable driving context, while promoting interpretable and causally grounded decision making. We curate a planning-oriented decision reasoning dataset, namely PDR, comprising 210k diverse and high-quality samples. Our method outperforms the mainstream E2E imitation learning method by a large margin of 19% L2 and 16.1 driving score on Bench2Drive benchmark. Furthermore, ReasonPlan demonstrates strong zero-shot generalization on unseen DOS benchmark, highlighting its adaptability in handling zero-shot corner cases. Code and dataset will be found in this https URL. 

**Abstract (ZH)**: 基于全面推理的自我监督下一场景预测和监督决策链 fine-tuning 框架 ReasonPlan：面向闭环自主驾驶的应用 

---
# Decomposing Complex Visual Comprehension into Atomic Visual Skills for Vision Language Models 

**Title (ZH)**: 将复杂视觉理解分解为原子视觉技能以供视觉语言模型使用 

**Authors**: Hyunsik Chae, Seungwoo Yoon, Jaden Park, Chloe Yewon Chun, Yongin Cho, Mu Cai, Yong Jae Lee, Ernest K. Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20021)  

**Abstract**: Recent Vision-Language Models (VLMs) have demonstrated impressive multimodal comprehension and reasoning capabilities, yet they often struggle with trivially simple visual tasks. In this work, we focus on the domain of basic 2D Euclidean geometry and systematically categorize the fundamental, indivisible visual perception skills, which we refer to as atomic visual skills. We then introduce the Atomic Visual Skills Dataset (AVSD) for evaluating VLMs on the atomic visual skills. Using AVSD, we benchmark state-of-the-art VLMs and find that they struggle with these tasks, despite being trivial for adult humans. Our findings highlight the need for purpose-built datasets to train and evaluate VLMs on atomic, rather than composite, visual perception tasks. 

**Abstract (ZH)**: Recent Vision-Language Models (VLMs)在基本二维欧几里得几何领域的原子视觉技能上的表现与挑战 

---
# StyleAR: Customizing Multimodal Autoregressive Model for Style-Aligned Text-to-Image Generation 

**Title (ZH)**: StyleAR: 为风格对齐文本到图像生成定制多模态自回归模型 

**Authors**: Yi Wu, Lingting Zhu, Shengju Qian, Lei Liu, Wandi Qiao, Lequan Yu, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.19874)  

**Abstract**: In the current research landscape, multimodal autoregressive (AR) models have shown exceptional capabilities across various domains, including visual understanding and generation. However, complex tasks such as style-aligned text-to-image generation present significant challenges, particularly in data acquisition. In analogy to instruction-following tuning for image editing of AR models, style-aligned generation requires a reference style image and prompt, resulting in a text-image-to-image triplet where the output shares the style and semantics of the input. However, acquiring large volumes of such triplet data with specific styles is considerably more challenging than obtaining conventional text-to-image data used for training generative models. To address this issue, we propose StyleAR, an innovative approach that combines a specially designed data curation method with our proposed AR models to effectively utilize text-to-image binary data for style-aligned text-to-image generation. Our method synthesizes target stylized data using a reference style image and prompt, but only incorporates the target stylized image as the image modality to create high-quality binary data. To facilitate binary data training, we introduce a CLIP image encoder with a perceiver resampler that translates the image input into style tokens aligned with multimodal tokens in AR models and implement a style-enhanced token technique to prevent content leakage which is a common issue in previous work. Furthermore, we mix raw images drawn from large-scale text-image datasets with stylized images to enhance StyleAR's ability to extract richer stylistic features and ensure style consistency. Extensive qualitative and quantitative experiments demonstrate our superior performance. 

**Abstract (ZH)**: 基于多模态自回归模型的风格对齐文本到图像生成方法：StyleAR 

---
# Benchmarking Large Multimodal Models for Ophthalmic Visual Question Answering with OphthalWeChat 

**Title (ZH)**: 基于OphthalWeChat大型多模态模型的眼科视觉问答基准研究 

**Authors**: Pusheng Xu, Xia Gong, Xiaolan Chen, Weiyi Zhang, Jiancheng Yang, Bingjie Yan, Meng Yuan, Yalin Zheng, Mingguang He, Danli Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.19624)  

**Abstract**: Purpose: To develop a bilingual multimodal visual question answering (VQA) benchmark for evaluating VLMs in ophthalmology. Methods: Ophthalmic image posts and associated captions published between January 1, 2016, and December 31, 2024, were collected from WeChat Official Accounts. Based on these captions, bilingual question-answer (QA) pairs in Chinese and English were generated using GPT-4o-mini. QA pairs were categorized into six subsets by question type and language: binary (Binary_CN, Binary_EN), single-choice (Single-choice_CN, Single-choice_EN), and open-ended (Open-ended_CN, Open-ended_EN). The benchmark was used to evaluate the performance of three VLMs: GPT-4o, Gemini 2.0 Flash, and Qwen2.5-VL-72B-Instruct. Results: The final OphthalWeChat dataset included 3,469 images and 30,120 QA pairs across 9 ophthalmic subspecialties, 548 conditions, 29 imaging modalities, and 68 modality combinations. Gemini 2.0 Flash achieved the highest overall accuracy (0.548), outperforming GPT-4o (0.522, P < 0.001) and Qwen2.5-VL-72B-Instruct (0.514, P < 0.001). It also led in both Chinese (0.546) and English subsets (0.550). Subset-specific performance showed Gemini 2.0 Flash excelled in Binary_CN (0.687), Single-choice_CN (0.666), and Single-choice_EN (0.646), while GPT-4o ranked highest in Binary_EN (0.717), Open-ended_CN (BLEU-1: 0.301; BERTScore: 0.382), and Open-ended_EN (BLEU-1: 0.183; BERTScore: 0.240). Conclusions: This study presents the first bilingual VQA benchmark for ophthalmology, distinguished by its real-world context and inclusion of multiple examinations per patient. The dataset reflects authentic clinical decision-making scenarios and enables quantitative evaluation of VLMs, supporting the development of accurate, specialized, and trustworthy AI systems for eye care. 

**Abstract (ZH)**: 目的：开发一种双语多模态视觉问答（VQA）基准，用于评估眼科视觉语言模型（VLMs）的表现。方法：从2016年1月1日至2024年12月31日，收集来自微信公众号的眼科影像帖子及其相关说明。基于这些说明，使用GPT-4o-mini生成了中英文双语问答（QA）对。根据问题类型和语言，QA对被分为六个子集：二分类（Binary_CN, Binary_EN）、单选（Single-choice_CN, Single-choice_EN）和开放式（Open-ended_CN, Open-ended_EN）。该基准用于评估三种VLMs：GPT-4o、Gemini 2.0 Flash和Qwen2.5-VL-72B-Instruct的表现。结果：最终的OphthalWeChat数据集包括3,469张图像和30,120个问答对，覆盖9个眼科亚专科、548种疾病、29种影像模态和68种模态组合。Gemini 2.0 Flash取得了最高的总体准确率（0.548），优于GPT-4o（0.522，P < 0.001）和Qwen2.5-VL-72B-Instruct（0.514，P < 0.001）。它还在中英文子集方面表现最佳（0.546和0.550）。特定子集的表现显示，Gemini 2.0 Flash在二分类中英文子集（0.687、0.666和0.646）中表现出色，而GPT-4o在单选英文子集（0.717）、开放式中英文子集（BLEU-1：0.301；BERTScore：0.382和0.183；BERTScore：0.240）中排名最高。结论：本研究介绍了首个眼科双语VQA基准，具有实际临床背景和每位患者多个检查项的特点。数据集反映了真实的临床决策场景，并允许对VLMs进行定量评估，支持开发准确、专业化和可信赖的眼科护理AI系统。 

---
# Align and Surpass Human Camouflaged Perception: Visual Refocus Reinforcement Fine-Tuning 

**Title (ZH)**: 超越人类伪装感知的对焦增强微调 

**Authors**: Ruolin Shen, Xiaozhong Ji, Kai WU, Jiangning Zhang, Yijun He, HaiHua Yang, Xiaobin Hu, Xiaoyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.19611)  

**Abstract**: Current multi-modal models exhibit a notable misalignment with the human visual system when identifying objects that are visually assimilated into the background. Our observations reveal that these multi-modal models cannot distinguish concealed objects, demonstrating an inability to emulate human cognitive processes which effectively utilize foreground-background similarity principles for visual analysis. To analyze this hidden human-model visual thinking discrepancy, we build a visual system that mimicks human visual camouflaged perception to progressively and iteratively `refocus' visual concealed content. The refocus is a progressive guidance mechanism enabling models to logically localize objects in visual images through stepwise reasoning. The localization process of concealed objects requires hierarchical attention shifting with dynamic adjustment and refinement of prior cognitive knowledge. In this paper, we propose a visual refocus reinforcement framework via the policy optimization algorithm to encourage multi-modal models to think and refocus more before answering, and achieve excellent reasoning abilities to align and even surpass human camouflaged perception systems. Our extensive experiments on camouflaged perception successfully demonstrate the emergence of refocus visual phenomena, characterized by multiple reasoning tokens and dynamic adjustment of the detection box. Besides, experimental results on both camouflaged object classification and detection tasks exhibit significantly superior performance compared to Supervised Fine-Tuning (SFT) baselines. 

**Abstract (ZH)**: 当前多模态模型在识别与背景视觉融合的物体时与人类视觉系统存在显著不一致。我们的观察表明，这些多模态模型无法区分隐藏物体，显示出无法模拟人类利用前景与背景相似性原则进行视觉分析的认知过程。为了分析这种隐藏的人工智能模型视觉思维差异，我们构建了一个视觉系统，模仿人类对伪装视觉感知的处理方式，逐步迭代地“重新聚焦”视觉隐藏内容。重新聚焦是一个逐步的指导机制，使模型能够通过逐步推理逻辑地定位视觉图像中的物体。隐藏物体的定位过程需要分层注意力转换，并且需要根据先验认知知识进行动态调整和优化。在本文中，我们提出了一种通过策略优化算法构建的视觉重新聚焦增强框架，以促进多模态模型在回答问题之前进行更多的思考和重新聚焦，从而获得出色的推理能力，甚至超越人类伪装感知系统。我们在伪装感知方面的广泛实验成功地展示了重新聚焦视觉现象的出现，其特征是由多个推理标记和检测框的动态调整组成。此外，伪装物体分类和检测任务上的实验结果与监督微调（SFT）基线相比显示出显著优越的性能。 

---
# Rethinking Gating Mechanism in Sparse MoE: Handling Arbitrary Modality Inputs with Confidence-Guided Gate 

**Title (ZH)**: 重新思考稀疏MoE中的门控机制：基于置信度引导的门控处理任意模态输入 

**Authors**: Liangwei Nathan Zheng, Wei Emma Zhang, Mingyu Guo, Miao Xu, Olaf Maennel, Weitong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.19525)  

**Abstract**: Effectively managing missing modalities is a fundamental challenge in real-world multimodal learning scenarios, where data incompleteness often results from systematic collection errors or sensor failures. Sparse Mixture-of-Experts (SMoE) architectures have the potential to naturally handle multimodal data, with individual experts specializing in different modalities. However, existing SMoE approach often lacks proper ability to handle missing modality, leading to performance degradation and poor generalization in real-world applications. We propose Conf-SMoE to introduce a two-stage imputation module to handle the missing modality problem for the SMoE architecture and reveal the insight of expert collapse from theoretical analysis with strong empirical evidence. Inspired by our theoretical analysis, Conf-SMoE propose a novel expert gating mechanism by detaching the softmax routing score to task confidence score w.r.t ground truth. This naturally relieves expert collapse without introducing additional load balance loss function. We show that the insights of expert collapse aligns with other gating mechanism such as Gaussian and Laplacian gate. We also evaluate the proposed method on four different real world dataset with three different experiment settings to conduct comprehensive the analysis of Conf-SMoE on modality fusion and resistance to missing modality. 

**Abstract (ZH)**: 有效地管理缺失模态是实际多模态学习场景中的一个基本挑战，数据不完整往往由系统性采集错误或传感器故障引起。稀疏专家混合（SMoE）架构有潜力自然处理多模态数据，每个专家专注于不同的模态。然而，现有的SMoE方法往往缺乏处理缺失模态的适当能力，导致在实际应用中的性能下降和泛化能力差。我们提出Conf-SMoE，引入两阶段插补模块处理SMoE架构中的缺失模态问题，并通过理论分析和强有力的实验证据揭示专家合并的洞察。受理论分析的启发，Conf-SMoE提出了一种新的专家门控机制，通过将softmax路由得分分离为与_ground truth_相关的任务置信得分。这自然地缓解了专家合并问题，而无需引入额外的负载平衡损失函数。我们表明，专家合并的洞察与高斯门和拉普拉斯门等其他门控机制一致。我们还在四个不同的真实世界数据集上，以三种不同的实验设置评估了所提出的方法，对Conf-SMoE在模态融合和抵抗缺失模态方面的综合分析。 

---
# Benchmarking Multimodal Knowledge Conflict for Large Multimodal Models 

**Title (ZH)**: 大型多模态模型中的多模态知识冲突基准研究 

**Authors**: Yifan Jia, Kailin Jiang, Yuyang Liang, Qihan Ren, Yi Xin, Rui Yang, Fenze Feng, Mingcai Chen, Hengyang Lu, Haozhe Wang, Xiaoye Qu, Dongrui Liu, Lizhen Cui, Yuntao Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.19509)  

**Abstract**: Large Multimodal Models(LMMs) face notable challenges when encountering multimodal knowledge conflicts, particularly under retrieval-augmented generation(RAG) frameworks where the contextual information from external sources may contradict the model's internal parametric knowledge, leading to unreliable outputs. However, existing benchmarks fail to reflect such realistic conflict scenarios. Most focus solely on intra-memory conflicts, while context-memory and inter-context conflicts remain largely investigated. Furthermore, commonly used factual knowledge-based evaluations are often overlooked, and existing datasets lack a thorough investigation into conflict detection capabilities. To bridge this gap, we propose MMKC-Bench, a benchmark designed to evaluate factual knowledge conflicts in both context-memory and inter-context scenarios. MMKC-Bench encompasses three types of multimodal knowledge conflicts and includes 1,573 knowledge instances and 3,381 images across 23 broad types, collected through automated pipelines with human verification. We evaluate three representative series of LMMs on both model behavior analysis and conflict detection tasks. Our findings show that while current LMMs are capable of recognizing knowledge conflicts, they tend to favor internal parametric knowledge over external evidence. We hope MMKC-Bench will foster further research in multimodal knowledge conflict and enhance the development of multimodal RAG systems. The source code is available at this https URL. 

**Abstract (ZH)**: 面向 Retrieval-Augmented Generation 框架下的多模态知识冲突基准：MMKC-Bench 

---
# MM-Prompt: Cross-Modal Prompt Tuning for Continual Visual Question Answering 

**Title (ZH)**: MM-Prompt: 跨模态提示调优的持续视觉问答 

**Authors**: Xu Li, Fan Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19455)  

**Abstract**: Continual Visual Question Answering (CVQA) based on pre-trained models(PTMs) has achieved promising progress by leveraging prompt tuning to enable continual multi-modal learning. However, most existing methods adopt cross-modal prompt isolation, constructing visual and textual prompts separately, which exacerbates modality imbalance and leads to degraded performance over time. To tackle this issue, we propose MM-Prompt, a novel framework incorporating cross-modal prompt query and cross-modal prompt recovery. The former enables balanced prompt selection by incorporating cross-modal signals during query formation, while the latter promotes joint prompt reconstruction through iterative cross-modal interactions, guided by an alignment loss to prevent representational drift. Extensive experiments show that MM-Prompt surpasses prior approaches in accuracy and knowledge retention, while maintaining balanced modality engagement throughout continual learning. 

**Abstract (ZH)**: 基于预训练模型的持续视觉问答（CVQA）通过利用提示调优实现有希望的进展，以促进持续多模态学习。然而，现有方法大多采用跨模态提示隔离策略，分别构建视觉和文本提示，这加重了模态不平衡，导致性能随时间下降。为解决这一问题，我们提出MM-Prompt框架，该框架结合了跨模态提示查询和跨模态提示恢复。前者通过在查询形成过程中融合跨模态信号，实现平衡的提示选择，而后者通过迭代的跨模态交互促进联合提示重构，并通过对齐损失防止表示漂移。广泛实验表明，MM-Prompt在准确性和知识保留方面超越了先前的方法，同时在持续学习过程中保持了模态的平衡参与。 

---
# I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts 

**Title (ZH)**: I2MoE: 可解释的多模态交互aware混合专家模型 

**Authors**: Jiayi Xin, Sukwon Yun, Jie Peng, Inyoung Choi, Jenna L. Ballard, Tianlong Chen, Qi Long  

**Link**: [PDF](https://arxiv.org/pdf/2505.19190)  

**Abstract**: Modality fusion is a cornerstone of multimodal learning, enabling information integration from diverse data sources. However, vanilla fusion methods are limited by (1) inability to account for heterogeneous interactions between modalities and (2) lack of interpretability in uncovering the multimodal interactions inherent in the data. To this end, we propose I2MoE (Interpretable Multimodal Interaction-aware Mixture of Experts), an end-to-end MoE framework designed to enhance modality fusion by explicitly modeling diverse multimodal interactions, as well as providing interpretation on a local and global level. First, I2MoE utilizes different interaction experts with weakly supervised interaction losses to learn multimodal interactions in a data-driven way. Second, I2MoE deploys a reweighting model that assigns importance scores for the output of each interaction expert, which offers sample-level and dataset-level interpretation. Extensive evaluation of medical and general multimodal datasets shows that I2MoE is flexible enough to be combined with different fusion techniques, consistently improves task performance, and provides interpretation across various real-world scenarios. Code is available at this https URL. 

**Abstract (ZH)**: 可解释的多模态交互感知混合专家模型（I2MoE）：一种增强多模态融合的方法 

---
# SATORI-R1: Incentivizing Multimodal Reasoning with Spatial Grounding and Verifiable Rewards 

**Title (ZH)**: SATORI-R1：基于空间grounding和可验证奖励的多模态推理激励 

**Authors**: Chuming Shen, Wei Wei, Xiaoye Qu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.19094)  

**Abstract**: DeepSeek-R1 has demonstrated powerful reasoning capabilities in the text domain through stable reinforcement learning (RL). Recently, in the multimodal domain, works have begun to directly apply RL to generate R1-like free-form reasoning for Visual Question Answering (VQA) tasks. However, multimodal tasks share an intrinsically different nature from textual tasks, which heavily rely on the understanding of the input image to solve the problem. Therefore, such free-form reasoning faces two critical limitations in the VQA task: (1) Extended reasoning chains diffuse visual focus away from task-critical regions, degrading answer accuracy. (2) Unverifiable intermediate steps amplify policy-gradient variance and computational costs overhead. To address these issues, in this paper, we introduce SATORI ($\textbf{S}patially$ $\textbf{A}nchored$ $\textbf{T}ask$ $\textbf{O}ptimization$ with $\textbf{R}e\textbf{I}nforcement$ Learning), which decomposes VQA into three verifiable stages, including global image captioning, region localization, and answer prediction, each supplying explicit reward signals. Furthermore, we also introduce VQA-Verify, a 12k dataset annotated with answer-aligned captions and bounding-boxes to facilitate training. Experiments demonstrate consistent performance improvements across seven VQA benchmarks, achieving up to $15.7\%$ improvement in accuracy in accuracy compared to the R1-like baseline. Our analysis of the attention map confirms enhanced focus on critical regions, which brings improvements in accuracy. Our code is available at this https URL. 

**Abstract (ZH)**: SATORI：Spatially Anchored Task Optimization with Reinforcement Learning for VQA 

---
# InfoChartQA: A Benchmark for Multimodal Question Answering on Infographic Charts 

**Title (ZH)**: InfoChartQA：信息图表多模态问答基准 

**Authors**: Minzhi Lin, Tianchi Xie, Mengchen Liu, Yilin Ye, Changjian Chen, Shixia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19028)  

**Abstract**: Understanding infographic charts with design-driven visual elements (e.g., pictograms, icons) requires both visual recognition and reasoning, posing challenges for multimodal large language models (MLLMs). However, existing visual-question answering benchmarks fall short in evaluating these capabilities of MLLMs due to the lack of paired plain charts and visual-element-based questions. To bridge this gap, we introduce InfoChartQA, a benchmark for evaluating MLLMs on infographic chart understanding. It includes 5,642 pairs of infographic and plain charts, each sharing the same underlying data but differing in visual presentations. We further design visual-element-based questions to capture their unique visual designs and communicative intent. Evaluation of 20 MLLMs reveals a substantial performance decline on infographic charts, particularly for visual-element-based questions related to metaphors. The paired infographic and plain charts enable fine-grained error analysis and ablation studies, which highlight new opportunities for advancing MLLMs in infographic chart understanding. We release InfoChartQA at this https URL. 

**Abstract (ZH)**: 理解和分析包含设计驱动视觉元素（如图表标记、图标）的信息图形图表需要同时进行视觉识别和推理，这对多模态大语言模型（MLLMs）提出了挑战。然而，现有的视觉问答基准无法充分评估MLLMs的这些能力，因为它们缺乏配对的普通图表和基于视觉元素的问题。为弥补这一差距，我们引入了InfoChartQA，这是一个用于评估MLLMs在信息图形图表理解方面能力的基准。该基准包括5,642对信息图形和普通图表，每对图表共享相同的数据但视觉呈现不同。我们进一步设计基于视觉元素的问题来捕捉其独特的视觉设计和通信意图。对20个MLLMs的评估显示，在信息图形图表上的性能显著下降，尤其是在涉及隐喻的基于视觉元素的问题上。配对的信息图形和普通图表使细粒度的错误分析和消融研究成为可能，这突显了在信息图形图表理解方面推进MLLMs的新机会。我们在此处发布InfoChartQA：https://。 

---
# Revival with Voice: Multi-modal Controllable Text-to-Speech Synthesis 

**Title (ZH)**: 语音再生：多模态可控文本到语音合成 

**Authors**: Minsu Kim, Pingchuan Ma, Honglie Chen, Stavros Petridis, Maja Pantic  

**Link**: [PDF](https://arxiv.org/pdf/2505.18972)  

**Abstract**: This paper explores multi-modal controllable Text-to-Speech Synthesis (TTS) where the voice can be generated from face image, and the characteristics of output speech (e.g., pace, noise level, distance, tone, place) can be controllable with natural text description. Specifically, we aim to mitigate the following three challenges in face-driven TTS systems. 1) To overcome the limited audio quality of audio-visual speech corpora, we propose a training method that additionally utilizes high-quality audio-only speech corpora. 2) To generate voices not only from real human faces but also from artistic portraits, we propose augmenting the input face image with stylization. 3) To consider one-to-many possibilities in face-to-voice mapping and ensure consistent voice generation at the same time, we propose to first employ sampling-based decoding and then use prompting with generated speech samples. Experimental results validate the proposed model's effectiveness in face-driven voice synthesis. 

**Abstract (ZH)**: 本文探索基于多模态可控文本到语音合成（TTS），可以从面部图像生成声音，并通过自然文本描述控制输出语音的特性（如语速、噪音水平、距离、音调和语域）。具体来说，我们旨在解决面部驱动TTS系统中的以下三项挑战：1) 为克服视听语音数据集的有限音频质量，我们提出了一种训练方法，额外利用高质量的纯音频数据集。2) 为了不仅能从真实人类面部生成声音，还能从艺术肖像生成声音，我们提出了对输入面部图像进行风格化增强。3) 为考虑面部到语音映射的一对多可能性并确保一致的声音生成，我们首先采用基于采样的解码，然后使用生成语音样本的提示。实验结果验证了所提模型在面部驱动语音合成中的有效性。 

---
# How Do Images Align and Complement LiDAR? Towards a Harmonized Multi-modal 3D Panoptic Segmentation 

**Title (ZH)**: 图像如何与LiDAR对齐和互补？ toward和谐的多模态3D全景分割 

**Authors**: Yining Pan, Qiongjie Cui, Xulei Yang, Na Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.18956)  

**Abstract**: LiDAR-based 3D panoptic segmentation often struggles with the inherent sparsity of data from LiDAR sensors, which makes it challenging to accurately recognize distant or small objects. Recently, a few studies have sought to overcome this challenge by integrating LiDAR inputs with camera images, leveraging the rich and dense texture information provided by the latter. While these approaches have shown promising results, they still face challenges, such as misalignment during data augmentation and the reliance on post-processing steps. To address these issues, we propose Image-Assists-LiDAR (IAL), a novel multi-modal 3D panoptic segmentation framework. In IAL, we first introduce a modality-synchronized data augmentation strategy, PieAug, to ensure alignment between LiDAR and image inputs from the start. Next, we adopt a transformer decoder to directly predict panoptic segmentation results. To effectively fuse LiDAR and image features into tokens for the decoder, we design a Geometric-guided Token Fusion (GTF) module. Additionally, we leverage the complementary strengths of each modality as priors for query initialization through a Prior-based Query Generation (PQG) module, enhancing the decoder's ability to generate accurate instance masks. Our IAL framework achieves state-of-the-art performance compared to previous multi-modal 3D panoptic segmentation methods on two widely used benchmarks. Code and models are publicly available at <this https URL. 

**Abstract (ZH)**: 基于LiDAR的3D全景分割往往难以处理LiDAR传感器数据固有的稀疏性问题，这使得准确识别远处或小型物体具有挑战性。最近，一些研究通过将LiDAR输入与摄像头图像结合，利用后者提供的丰富密集的纹理信息尝试克服这一挑战。尽管这些方法取得了有前景的结果，但仍面临数据增强时的误对齐问题和依赖后期处理步骤的问题。为解决这些问题，我们提出了一种新的多模态3D全景分割框架——Image-Assists-LiDAR (IAL)。在IAL中，我们首先引入了一种模态同步数据增强策略PieAug，以确保从一开始就对齐LiDAR和图像输入。接着，我们采用 transformer 解码器直接预测全景分割结果。为了有效地将LiDAR和图像特征融合到解码器的标记中，我们设计了一种几何引导标记融合（GTF）模块。此外，我们通过先验基于查询生成（PQG）模块利用每种模态的互补优势作为查询初始化的先验，增强了解码器生成准确实例掩码的能力。与先前的多模态3D全景分割方法相比，我们的IAL框架在两个广泛应用的基准测试中达到了最先进的性能。代码和模型已公开发布。 

---
# REGen: Multimodal Retrieval-Embedded Generation for Long-to-Short Video Editing 

**Title (ZH)**: REGen: 多模态检索嵌入生成的长视频到短视频编辑 

**Authors**: Weihan Xu, Yimeng Ma, Jingyue Huang, Yang Li, Wenye Ma, Taylor Berg-Kirkpatrick, Julian McAuley, Paul Pu Liang, Hao-Wen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.18880)  

**Abstract**: Short videos are an effective tool for promoting contents and improving knowledge accessibility. While existing extractive video summarization methods struggle to produce a coherent narrative, existing abstractive methods cannot `quote' from the input videos, i.e., inserting short video clips in their outputs. In this work, we explore novel video editing models for generating shorts that feature a coherent narrative with embedded video insertions extracted from a long input video. We propose a novel retrieval-embedded generation framework that allows a large language model to quote multimodal resources while maintaining a coherent narrative. Our proposed REGen system first generates the output story script with quote placeholders using a finetuned large language model, and then uses a novel retrieval model to replace the quote placeholders by selecting a video clip that best supports the narrative from a pool of candidate quotable video clips. We examine the proposed method on the task of documentary teaser generation, where short interview insertions are commonly used to support the narrative of a documentary. Our objective evaluations show that the proposed method can effectively insert short video clips while maintaining a coherent narrative. In a subjective survey, we show that our proposed method outperforms existing abstractive and extractive approaches in terms of coherence, alignment, and realism in teaser generation. 

**Abstract (ZH)**: 短视频是推广内容和提高知识可访问性的有效工具。现有提取式视频摘要方法难以生成连贯的叙事，而现有的抽象式方法则无法从输入视频中“引用”，即在输出中插入短视频片段。在本文中，我们探讨了新型视频编辑模型，用于生成包含从长输入视频中提取的视频插入并具有连贯叙事的短视频。我们提出了一种新颖的检索嵌入生成框架，允许大型语言模型引用多模态资源同时保持连贯的叙事。我们提出的REGen系统首先使用微调的大语言模型生成包含引用占位符的输出故事剧本，然后使用一种新颖的检索模型，通过从候选可引用视频片段池中选择最支持叙事的视频片段来替换这些引用占位符。我们在纪录片预告片生成任务上检验了该方法，其中常见的短访谈插入支持纪录片的叙事。客观评估显示，该方法能够在保持连贯叙事的同时有效插入短视频片段。在主观调查中，我们证明了与现有抽象和提取方法相比，我们的方法在预告片生成的连贯性、对齐和真实性方面表现更优。 

---
# Rethinking Causal Mask Attention for Vision-Language Inference 

**Title (ZH)**: 重新思考因果掩码注意力在视觉-语言推理中的应用 

**Authors**: Xiaohuan Pei, Tao Huang, YanXiang Ma, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18605)  

**Abstract**: Causal attention has become a foundational mechanism in autoregressive vision-language models (VLMs), unifying textual and visual inputs under a single generative framework. However, existing causal mask-based strategies are inherited from large language models (LLMs) where they are tailored for text-only decoding, and their adaptation to vision tokens is insufficiently addressed in the prefill stage. Strictly masking future positions for vision queries introduces overly rigid constraints, which hinder the model's ability to leverage future context that often contains essential semantic cues for accurate inference. In this work, we empirically investigate how different causal masking strategies affect vision-language inference and then propose a family of future-aware attentions tailored for this setting. We first empirically analyze the effect of previewing future tokens for vision queries and demonstrate that rigid masking undermines the model's capacity to capture useful contextual semantic representations. Based on these findings, we propose a lightweight attention family that aggregates future visual context into past representations via pooling, effectively preserving the autoregressive structure while enhancing cross-token dependencies. We evaluate a range of causal masks across diverse vision-language inference settings and show that selectively compressing future semantic context into past representations benefits the inference. 

**Abstract (ZH)**: 因果注意机制已成为自回归视觉语言模型（VLMs）的基础机制，将文本和视觉输入统一在一个生成框架下。然而，现有的因果掩码策略源自大型语言模型（LLMs），这些模型专门针对文本解码进行了调整，在视觉标记的预填阶段，这些策略的适应性不足。严格掩码未来位置会为视觉查询引入过于刚性的约束，这阻碍了模型利用通常包含准确推理所需关键语义线索的未来上下文的能力。在本文中，我们实证研究了不同的因果掩码策略对视觉语言推理的影响，然后提出了一种专门为此场景设计的未来意识注意机制。我们首先实证分析了视觉查询预览未来标记的影响，并证明了刚性掩码会削弱模型捕捉有用上下文语义表示的能力。基于这些发现，我们提出了一种轻量级的注意机制，通过聚类将未来的视觉上下文整合到过去表示中，从而有效保持自回归结构的同时增强跨标记依赖关系。我们评估了多种因果掩码在不同的视觉语言推理场景中的表现，并展示了有选择地将未来语义上下文压缩到过去表示中对推理是有益的。 

---
# MPE-TTS: Customized Emotion Zero-Shot Text-To-Speech Using Multi-Modal Prompt 

**Title (ZH)**: MPE-TTS：基于多模态提示的定制化零样本文本到语音合成 

**Authors**: Zhichao Wu, Yueteng Kang, Songjun Cao, Long Ma, Qiulin Li, Qun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18453)  

**Abstract**: Most existing Zero-Shot Text-To-Speech(ZS-TTS) systems generate the unseen speech based on single prompt, such as reference speech or text descriptions, which limits their flexibility. We propose a customized emotion ZS-TTS system based on multi-modal prompt. The system disentangles speech into the content, timbre, emotion and prosody, allowing emotion prompts to be provided as text, image or speech. To extract emotion information from different prompts, we propose a multi-modal prompt emotion encoder. Additionally, we introduce an prosody predictor to fit the distribution of prosody and propose an emotion consistency loss to preserve emotion information in the predicted prosody. A diffusion-based acoustic model is employed to generate the target mel-spectrogram. Both objective and subjective experiments demonstrate that our system outperforms existing systems in terms of naturalness and similarity. The samples are available at this https URL. 

**Abstract (ZH)**: 基于多模态提示的定制情感零样本文本到语音系统 

---
# Less is More: Multimodal Region Representation via Pairwise Inter-view Learning 

**Title (ZH)**: 少即是多：通过成对跨视图学习的多模态区域表示 

**Authors**: Min Namgung, Yijun Lin, JangHyeon Lee, Yao-Yi Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18178)  

**Abstract**: With the increasing availability of geospatial datasets, researchers have explored region representation learning (RRL) to analyze complex region characteristics. Recent RRL methods use contrastive learning (CL) to capture shared information between two modalities but often overlook task-relevant unique information specific to each modality. Such modality-specific details can explain region characteristics that shared information alone cannot capture. Bringing information factorization to RRL can address this by factorizing multimodal data into shared and unique information. However, existing factorization approaches focus on two modalities, whereas RRL can benefit from various geospatial data. Extending factorization beyond two modalities is non-trivial because modeling high-order relationships introduces a combinatorial number of learning objectives, increasing model complexity. We introduce Cross modal Knowledge Injected Embedding, an information factorization approach for RRL that captures both shared and unique representations. CooKIE uses a pairwise inter-view learning approach that captures high-order information without modeling high-order dependency, avoiding exhaustive combinations. We evaluate CooKIE on three regression tasks and a land use classification task in New York City and Delhi, India. Results show that CooKIE outperforms existing RRL methods and a factorized RRL model, capturing multimodal information with fewer training parameters and floating-point operations per second (FLOPs). We release the code: this https URL. 

**Abstract (ZH)**: 基于跨模态知识注入嵌入的区域表示学习中的信息分解 

---
