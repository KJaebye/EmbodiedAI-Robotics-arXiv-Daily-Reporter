# KGMEL: Knowledge Graph-Enhanced Multimodal Entity Linking 

**Title (ZH)**: 知识图谱增强的多模态实体链接 

**Authors**: Juyeon Kim, Geon Lee, Taeuk Kim, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15135)  

**Abstract**: Entity linking (EL) aligns textual mentions with their corresponding entities in a knowledge base, facilitating various applications such as semantic search and question answering. Recent advances in multimodal entity linking (MEL) have shown that combining text and images can reduce ambiguity and improve alignment accuracy. However, most existing MEL methods overlook the rich structural information available in the form of knowledge-graph (KG) triples. In this paper, we propose KGMEL, a novel framework that leverages KG triples to enhance MEL. Specifically, it operates in three stages: (1) Generation: Produces high-quality triples for each mention by employing vision-language models based on its text and images. (2) Retrieval: Learns joint mention-entity representations, via contrastive learning, that integrate text, images, and (generated or KG) triples to retrieve candidate entities for each mention. (3) Reranking: Refines the KG triples of the candidate entities and employs large language models to identify the best-matching entity for the mention. Extensive experiments on benchmark datasets demonstrate that KGMEL outperforms existing methods. Our code and datasets are available at: this https URL. 

**Abstract (ZH)**: 基于知识图谱三元组的多模态实体链接（KG triples 增强的 MEL） 

---
# Chinese-LiPS: A Chinese audio-visual speech recognition dataset with Lip-reading and Presentation Slides 

**Title (ZH)**: Chinese-LiPS: 一个包含唇读和演示幻灯片的中文视听说话人识别数据集 

**Authors**: Jinghua Zhao, Yuhang Jia, Shiyao Wang, Jiaming Zhou, Hui Wang, Yong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15066)  

**Abstract**: Incorporating visual modalities to assist Automatic Speech Recognition (ASR) tasks has led to significant improvements. However, existing Audio-Visual Speech Recognition (AVSR) datasets and methods typically rely solely on lip-reading information or speaking contextual video, neglecting the potential of combining these different valuable visual cues within the speaking context. In this paper, we release a multimodal Chinese AVSR dataset, Chinese-LiPS, comprising 100 hours of speech, video, and corresponding manual transcription, with the visual modality encompassing both lip-reading information and the presentation slides used by the speaker. Based on Chinese-LiPS, we develop a simple yet effective pipeline, LiPS-AVSR, which leverages both lip-reading and presentation slide information as visual modalities for AVSR tasks. Experiments show that lip-reading and presentation slide information improve ASR performance by approximately 8\% and 25\%, respectively, with a combined performance improvement of about 35\%. The dataset is available at this https URL 

**Abstract (ZH)**: 将视觉模态融入自动语音识别（ASR）任务以辅助自动唇读视觉语音识别（AVSR）已经取得了显著的改进。然而，现有的AVSR数据集和方法通常仅依赖唇读信息或说话的背景视频，忽视了结合这些不同有价值视觉线索的潜力。本文发布了一个多模态中文AVSR数据集Chinese-LiPS，包含100小时的语音、视频及其相应的手动转录，其中视觉模态包括唇读信息和演讲者使用的幻灯片。基于Chinese-LiPS，我们开发了一个简单有效的框架LiPS-AVSR，利用唇读和幻灯片信息作为AVSR任务的视觉模态。实验表明，唇读和幻灯片信息分别将ASR性能提高约8%和25%，结合使用时性能提高约35%。数据集可从以下链接访问：this https URL。 

---
# Optimizing SIA Development: A Case Study in User-Centered Design for Estuary, a Multimodal Socially Interactive Agent Framework 

**Title (ZH)**: 优化SIA发展：一个基于用户中心设计的案例研究——以Estuary多模态社会交互代理框架为例 

**Authors**: Spencer Lin, Miru Jun, Basem Rizk, Karen Shieh, Scott Fisher, Sharon Mozgai  

**Link**: [PDF](https://arxiv.org/pdf/2504.14427)  

**Abstract**: This case study presents our user-centered design model for Socially Intelligent Agent (SIA) development frameworks through our experience developing Estuary, an open source multimodal framework for building low-latency real-time socially interactive agents. We leverage the Rapid Assessment Process (RAP) to collect the thoughts of leading researchers in the field of SIAs regarding the current state of the art for SIA development as well as their evaluation of how well Estuary may potentially address current research gaps. We achieve this through a series of end-user interviews conducted by a fellow researcher in the community. We hope that the findings of our work will not only assist the continued development of Estuary but also guide the development of other future frameworks and technologies for SIAs. 

**Abstract (ZH)**: 本案例研究通过我们在开发开源多模态框架Estuary（用于构建低延迟实时社交互动代理）过程中积累的经验，提出了以用户为中心的设计模型，用于社交智能代理（SIA）开发框架。我们利用快速评估过程（RAP）收集领域内领先研究人员关于SIA开发的当前技术水平及其评估，探讨Estuary如何潜在地弥补当前的研究空白。我们通过社区中另一位研究人员进行的一系列最终用户访谈实现了这一点。我们希望本工作的发现不仅能够促进Estuary的持续开发，还能够指导其他未来SIA框架和技术的发展。 

---
# A Multimodal Recaptioning Framework to Account for Perceptual Diversity in Multilingual Vision-Language Modeling 

**Title (ZH)**: 一种多模态重新描写框架，以考虑多语言视觉-语言模型中的感知多样性 

**Authors**: Kyle Buettner, Jacob Emmerson, Adriana Kovashka  

**Link**: [PDF](https://arxiv.org/pdf/2504.14359)  

**Abstract**: There are many ways to describe, name, and group objects when captioning an image. Differences are evident when speakers come from diverse cultures due to the unique experiences that shape perception. Machine translation of captions has pushed multilingual capabilities in vision-language models (VLMs), but data comes mainly from English speakers, indicating a perceptual bias and lack of model flexibility. In this work, we address this challenge and outline a data-efficient framework to instill multilingual VLMs with greater understanding of perceptual diversity. We specifically propose an LLM-based, multimodal recaptioning strategy that alters the object descriptions of English captions before translation. The greatest benefits are demonstrated in a targeted multimodal mechanism guided by native speaker data. By adding produced rewrites as augmentations in training, we improve on German and Japanese text-image retrieval cases studies (up to +3.5 mean recall overall, +4.7 on non-native error cases). We further propose a mechanism to analyze the specific object description differences across datasets, and we offer insights into cross-dataset and cross-language generalization. 

**Abstract (ZH)**: 描述、命名和分组图像中的对象有多种方式。来自不同文化背景的说话者在表述时因独特的经验而产生感知差异。机器翻译描述提升了视觉-语言模型（VLMs）的多语言能力，但数据主要来自以英语为母语的说话者，这显示出感知偏见和模型灵活性不足的问题。本文旨在应对这一挑战，提出了一种数据高效框架，以增强多语言VLMs对感知多样性理解的能力。我们具体提出了一种基于大规模语言模型的多模态重描述策略，在翻译前修改英语描述。该策略在以母语数据为指导的针对性多模态机制中显示出最大益处。通过将生成的修改作为训练增强，我们在德语和日语文本-图像检索案例研究中取得了改进（总体平均召回率提高3.5%，非母语错误案例提高4.7%）。我们还提出了一种机制来分析不同数据集中的特定对象描述差异，并提供了跨数据集和跨语言泛化的一些见解。 

---
# Enhancing Multimodal In-Context Learning for Image Classification through Coreset Optimization 

**Title (ZH)**: 通过核心样本优化提升多模态上下文学习的图像分类性能 

**Authors**: Huiyi Chen, Jiawei Peng, Kaihua Tang, Xin Geng, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14200)  

**Abstract**: In-context learning (ICL) enables Large Vision-Language Models (LVLMs) to adapt to new tasks without parameter updates, using a few demonstrations from a large support set. However, selecting informative demonstrations leads to high computational and memory costs. While some methods explore selecting a small and representative coreset in the text classification, evaluating all support set samples remains costly, and discarded samples lead to unnecessary information loss. These methods may also be less effective for image classification due to differences in feature spaces. Given these limitations, we propose Key-based Coreset Optimization (KeCO), a novel framework that leverages untapped data to construct a compact and informative coreset. We introduce visual features as keys within the coreset, which serve as the anchor for identifying samples to be updated through different selection strategies. By leveraging untapped samples from the support set, we update the keys of selected coreset samples, enabling the randomly initialized coreset to evolve into a more informative coreset under low computational cost. Through extensive experiments on coarse-grained and fine-grained image classification benchmarks, we demonstrate that KeCO effectively enhances ICL performance for image classification task, achieving an average improvement of more than 20\%. Notably, we evaluate KeCO under a simulated online scenario, and the strong performance in this scenario highlights the practical value of our framework for resource-constrained real-world scenarios. 

**Abstract (ZH)**: 基于键的核心集优化（KeCO）：提升图像分类的上下文学习性能 

---
# A Physics-guided Multimodal Transformer Path to Weather and Climate Sciences 

**Title (ZH)**: 物理指导的多模态变压器路径在天气与气候科学中的应用 

**Authors**: Jing Han, Hanting Chen, Kai Han, Xiaomeng Huang, Yongyun Hu, Wenjun Xu, Dacheng Tao, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14174)  

**Abstract**: With the rapid development of machine learning in recent years, many problems in meteorology can now be addressed using AI models. In particular, data-driven algorithms have significantly improved accuracy compared to traditional methods. Meteorological data is often transformed into 2D images or 3D videos, which are then fed into AI models for learning. Additionally, these models often incorporate physical signals, such as temperature, pressure, and wind speed, to further enhance accuracy and interpretability. In this paper, we review several representative AI + Weather/Climate algorithms and propose a new paradigm where observational data from different perspectives, each with distinct physical meanings, are treated as multimodal data and integrated via transformers. Furthermore, key weather and climate knowledge can be incorporated through regularization techniques to further strengthen the model's capabilities. This new paradigm is versatile and can address a variety of tasks, offering strong generalizability. We also discuss future directions for improving model accuracy and interpretability. 

**Abstract (ZH)**: 近年来，随着机器学习的快速发展，许多气象问题现在可以使用AI模型来解决。特别是数据驱动的算法相比传统方法显著提高了准确性。气象数据通常被转换为2D图像或3D视频，然后输入到AI模型中进行学习。此外，这些模型还常常结合物理信号，如温度、压力和风速，以进一步提高准确性和可解释性。在本文中，我们回顾了几种代表性的AI + 气象/气候算法，并提出了一种新范式，即将来自不同视角的观测数据，每种数据具有不同的物理意义，视为多模态数据并通过变换器进行整合。此外，通过正则化技术可以嵌入关键的气象和气候知识，以进一步增强模型的能力。该新范式具有广泛的适用性，可以解决多种任务，提供强大的泛化能力。我们还讨论了改进模型准确性和可解释性的未来方向。 

---
# Fashion-RAG: Multimodal Fashion Image Editing via Retrieval-Augmented Generation 

**Title (ZH)**: Fashion-RAG: 基于检索增强生成的多模态时尚图像编辑 

**Authors**: Fulvio Sanguigni, Davide Morelli, Marcella Cornia, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2504.14011)  

**Abstract**: In recent years, the fashion industry has increasingly adopted AI technologies to enhance customer experience, driven by the proliferation of e-commerce platforms and virtual applications. Among the various tasks, virtual try-on and multimodal fashion image editing -- which utilizes diverse input modalities such as text, garment sketches, and body poses -- have become a key area of research. Diffusion models have emerged as a leading approach for such generative tasks, offering superior image quality and diversity. However, most existing virtual try-on methods rely on having a specific garment input, which is often impractical in real-world scenarios where users may only provide textual specifications. To address this limitation, in this work we introduce Fashion Retrieval-Augmented Generation (Fashion-RAG), a novel method that enables the customization of fashion items based on user preferences provided in textual form. Our approach retrieves multiple garments that match the input specifications and generates a personalized image by incorporating attributes from the retrieved items. To achieve this, we employ textual inversion techniques, where retrieved garment images are projected into the textual embedding space of the Stable Diffusion text encoder, allowing seamless integration of retrieved elements into the generative process. Experimental results on the Dress Code dataset demonstrate that Fashion-RAG outperforms existing methods both qualitatively and quantitatively, effectively capturing fine-grained visual details from retrieved garments. To the best of our knowledge, this is the first work to introduce a retrieval-augmented generation approach specifically tailored for multimodal fashion image editing. 

**Abstract (ZH)**: 近年来，时尚行业 increasingly 采纳 AI 技术 以 提升 客户体验，受到电子商务平台和虚拟应用的普及驱动。在多种任务中，虚拟试衣和多模态服装图像编辑——利用文本、服装草图和身体姿态等多种输入模态——已成为研究的关键领域。扩散模型 已 成为 这种 生成任务 的 领导性 方法，提供 优越 的 图像质量 和 多样性。然而，现有大多数虚拟试衣方法 均 依赖 特定 的 服装输入，这在实际应用中往往并不可行，因为用户可能仅提供 文本 规格。为解决这一局限，本文提出 一种 新颖 方法——Fashion Retrieval-Augmented Generation (Fashion-RAG)，使用户可以基于文本形式提供的偏好定制服装项目。我们的方法检索多个与输入规格匹配的服装，并生成包含检索物品属性的个性化图像。为了实现这一点，我们采用 文本反转 技术，使检索出的服装图像能够投射到 Stable Diffusion 文本编码器 的 文本嵌入空间，从而使检索出的元素能够无缝集成到生成过程中。在 Dress Code 数据集上的实验结果表明，Fashion-RAG 在定性和定量上均优于现有方法，有效捕捉了检索服装的详细视觉细节。据我们所知，这是首 次 将检索增强生成 方法 专门 应用于多模态服装图像编辑的研究工作。 

---
# Mozualization: Crafting Music and Visual Representation with Multimodal AI 

**Title (ZH)**: 模化：利用多模态AI创作音乐和视觉表现 

**Authors**: Wanfang Xu, Lixiang Zhao, Haiwen Song, Xinheng Song, Zhaolin Lu, Yu Liu, Min Chen, Eng Gee Lim, Lingyun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13891)  

**Abstract**: In this work, we introduce Mozualization, a music generation and editing tool that creates multi-style embedded music by integrating diverse inputs, such as keywords, images, and sound clips (e.g., segments from various pieces of music or even a playful cat's meow). Our work is inspired by the ways people express their emotions -- writing mood-descriptive poems or articles, creating drawings with warm or cool tones, or listening to sad or uplifting music. Building on this concept, we developed a tool that transforms these emotional expressions into a cohesive and expressive song, allowing users to seamlessly incorporate their unique preferences and inspirations. To evaluate the tool and, more importantly, gather insights for its improvement, we conducted a user study involving nine music enthusiasts. The study assessed user experience, engagement, and the impact of interacting with and listening to the generated music. 

**Abstract (ZH)**: Mozualization：一种融合关键词、图像和音剪辑创作多风格嵌入音乐的工具及其用户研究 

---
# Towards a Multimodal Document-grounded Conversational AI System for Education 

**Title (ZH)**: 面向教育领域的多模态文档 grounding 对话 AI 系统研究 

**Authors**: Karan Taneja, Anjali Singh, Ashok K. Goel  

**Link**: [PDF](https://arxiv.org/pdf/2504.13884)  

**Abstract**: Multimedia learning using text and images has been shown to improve learning outcomes compared to text-only instruction. But conversational AI systems in education predominantly rely on text-based interactions while multimodal conversations for multimedia learning remain unexplored. Moreover, deploying conversational AI in learning contexts requires grounding in reliable sources and verifiability to create trust. We present MuDoC, a Multimodal Document-grounded Conversational AI system based on GPT-4o, that leverages both text and visuals from documents to generate responses interleaved with text and images. Its interface allows verification of AI generated content through seamless navigation to the source. We compare MuDoC to a text-only system to explore differences in learner engagement, trust in AI system, and their performance on problem-solving tasks. Our findings indicate that both visuals and verifiability of content enhance learner engagement and foster trust; however, no significant impact in performance was observed. We draw upon theories from cognitive and learning sciences to interpret the findings and derive implications, and outline future directions for the development of multimodal conversational AI systems in education. 

**Abstract (ZH)**: 利用文本和图像的多媒体学习已被证明能 Compared to text-only instruction, multimedia learning using text and images has been shown to improve learning outcomes. 但教育领域的对话式AI系统主要依赖于基于文本的交互，而多媒体学习中的多模态对话尚未被探索。此外，将对话式AI应用于学习环境需要基于可靠的来源并具备可验证性以建立信任。我们提出了一种基于GPT-4o的多模态文档导向对话式AI系统MuDoC，该系统利用文档中的文本和视觉内容生成交织的文本和图像响应。其界面允许通过平滑导航到源内容来验证AI生成的内容。我们对比MuDoC与纯文本系统，探索学习者参与度、对AI系统的信任以及解决任务性能的差异。研究发现，视觉内容和内容的可验证性均能增强学习者的参与度并培养信任；但未观察到对性能的显著影响。我们结合认知科学和学习科学的理论来解释这些发现并推导出启示，并概述了在教育领域发展中多模态对话式AI系统未来的研究方向。 

---
