# RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training 

**Title (ZH)**: RLinf-VLA：一种统一高效的VLA+RL训练框架 

**Authors**: Hongzhi Zang, Mingjie Wei, Si Xu, Yongji Wu, Zhen Guo, Yuanqing Wang, Hao Lin, Liangzhi Shi, Yuqing Xie, Zhexuan Xu, Zhihao Liu, Kang Chen, Wenhao Tang, Quanlu Zhang, Weinan Zhang, Chao Yu, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06710)  

**Abstract**: Recent progress in vision and language foundation models has significantly advanced multimodal understanding, reasoning, and generation, inspiring a surge of interest in extending such capabilities to embodied settings through vision-language-action (VLA) models. Yet, most VLA models are still trained with supervised fine-tuning (SFT), which struggles to generalize under distribution shifts due to error accumulation. Reinforcement learning (RL) offers a promising alternative by directly optimizing task performance through interaction, but existing attempts remain fragmented and lack a unified platform for fair and systematic comparison across model architectures and algorithmic designs. To address this gap, we introduce RLinf-VLA, a unified and efficient framework for scalable RL training of VLA models. The system adopts a highly flexible resource allocation design that addresses the challenge of integrating rendering, training, and inference in RL+VLA training. In particular, for GPU-parallelized simulators, RLinf-VLA implements a novel hybrid fine-grained pipeline allocation mode, achieving a 1.61x-1.88x speedup in training. Through a unified interface, RLinf-VLA seamlessly supports diverse VLA architectures (e.g., OpenVLA, OpenVLA-OFT), multiple RL algorithms (e.g., PPO, GRPO), and various simulators (e.g., ManiSkill, LIBERO). In simulation, a unified model achieves 98.11\% across 130 LIBERO tasks and 97.66\% across 25 ManiSkill tasks. Beyond empirical performance, our study distills a set of best practices for applying RL to VLA training and sheds light on emerging patterns in this integration. Furthermore, we present preliminary deployment on a real-world Franka robot, where RL-trained policies exhibit stronger generalization than those trained with SFT. We envision RLinf-VLA as a foundation to accelerate and standardize research on embodied intelligence. 

**Abstract (ZH)**: 近期视觉与语言基础模型的进展显著推进了多模态理解、推理和生成，激发了通过视觉-语言-动作（VLA）模型将此类能力扩展到具身环境中的兴趣。然而，大多数VLA模型仍使用监督微调（SFT）训练，这在分布迁移时由于误差累积难以泛化。强化学习（RL）通过交互直接优化任务性能提供了有前景的替代方案，但现有尝试仍碎片化且缺乏针对不同模型架构和算法设计进行全面公平比较的统一平台。为填补这一空白，我们引入了RLinf-VLA，这是一个用于可扩展的VLA模型强化学习训练的统一高效框架。该系统采用高度灵活的资源分配设计，解决了RL+VLA训练中渲染、训练和推理整合的挑战。特别是，对于GPU并行化模拟器，RLinf-VLA 实现了一种新型细粒度混合管道分配模式，实现1.61x-1.88x的训练速度提升。通过统一接口，RLinf-VLA 紧密支持多种VLA架构（如OpenVLA、OpenVLA-OFT）、多种RL算法（如PPO、GRPO）和各种模拟器（如ManiSkill、LIBERO）。在模拟中，统一模型在130个LIBERO任务中达到了98.11%的表现，在25个ManiSkill任务中达到了97.66%的表现。除实证性能外，我们的研究提炼了将RL应用于VLA训练的最佳实践，并揭示了这一集成中的新兴模式。此外，我们初步展示了在真实世界Franka机器人上的部署，其中RL训练策略的泛化能力优于SFT训练策略。我们期望RLinf-VLA成为加速和标准化具身智能研究的基础。 

---
# M3Retrieve: Benchmarking Multimodal Retrieval for Medicine 

**Title (ZH)**: M3Retrieve:  multimodal retrieval benchmarking for medicine 

**Authors**: Arkadeep Acharya, Akash Ghosh, Pradeepika Verma, Kitsuchart Pasupa, Sriparna Saha, Priti Singh  

**Link**: [PDF](https://arxiv.org/pdf/2510.06888)  

**Abstract**: With the increasing use of RetrievalAugmented Generation (RAG), strong retrieval models have become more important than ever. In healthcare, multimodal retrieval models that combine information from both text and images offer major advantages for many downstream tasks such as question answering, cross-modal retrieval, and multimodal summarization, since medical data often includes both formats. However, there is currently no standard benchmark to evaluate how well these models perform in medical settings. To address this gap, we introduce M3Retrieve, a Multimodal Medical Retrieval Benchmark. M3Retrieve, spans 5 domains,16 medical fields, and 4 distinct tasks, with over 1.2 Million text documents and 164K multimodal queries, all collected under approved licenses. We evaluate leading multimodal retrieval models on this benchmark to explore the challenges specific to different medical specialities and to understand their impact on retrieval performance. By releasing M3Retrieve, we aim to enable systematic evaluation, foster model innovation, and accelerate research toward building more capable and reliable multimodal retrieval systems for medical applications. The dataset and the baselines code are available in this github page this https URL. 

**Abstract (ZH)**: 随着检索增强生成（RAG）的使用不断增加，强大的检索模型比以往任何时候都更为重要。在医疗领域，结合文本和图像信息的多模态检索模型为诸如问答、跨模态检索和多模态总结等多种下游任务提供了重大优势，因为医疗数据通常包括这两种格式。然而，目前尚无标准基准来评估这些模型在医疗环境中的表现。为解决这一问题，我们引入了M3Retrieve多模态医疗检索基准。M3Retrieve涵盖了5个领域、16个医疗领域和4项不同的任务，包含超过120万份文本文档和16.4万个多模态查询，所有数据均在获批许可证下收集。我们在这项基准上评估了领先的多模态检索模型，以探索不同医疗专科特有的挑战，并理解其对检索性能的影响。通过发布M3Retrieve，我们旨在促进系统的评估、模型创新，并加速针对医疗应用构建更具能力和可靠性的多模态检索系统的研究。该数据集和baseline代码可在以下github页面获取：this https URL。 

---
# Visualizing Multimodality in Combinatorial Search Landscapes 

**Title (ZH)**: 可视化组合搜索景观中的多模态性 

**Authors**: Xavier F. C. Sánchez-Díaz, Ole Jakob Mengshoel  

**Link**: [PDF](https://arxiv.org/pdf/2510.06517)  

**Abstract**: This work walks through different visualization techniques for combinatorial search landscapes, focusing on multimodality. We discuss different techniques from the landscape analysis literature, and how they can be combined to provide a more comprehensive view of the search landscape. We also include examples and discuss relevant work to show how others have used these techniques in practice, based on the geometric and aesthetic elements of the Grammar of Graphics. We conclude that there is no free lunch in visualization, and provide recommendations for future work as there are several paths to continue the work in this field. 

**Abstract (ZH)**: 本研究探讨组合搜索景观的不同可视化技术，重点讨论多模态性。我们讨论来自景观分析文献的不同技术，并探讨如何将这些技术结合起来提供更为全面的搜索景观视图。我们还通过几何和审美元素中的图形语法示例来展示这些技术在实践中的应用，并讨论相关研究工作。我们得出结论，在可视化中没有免费午餐，并为未来工作提供建议，因为在该领域还有多种路径可以继续研究。 

---
# EverydayMMQA: A Multilingual and Multimodal Framework for Culturally Grounded Spoken Visual QA 

**Title (ZH)**: EverydayMMQA：一个基于文化的多语言多模态 spoken Visual QA 框架 

**Authors**: Firoj Alam, Ali Ezzat Shahroor, Md. Arid Hasan, Zien Sheikh Ali, Hunzalah Hassan Bhatti, Mohamed Bayan Kmainasi, Shammur Absar Chowdhury, Basel Mousi, Fahim Dalvi, Nadir Durrani, Natasa Milic-Frayling  

**Link**: [PDF](https://arxiv.org/pdf/2510.06371)  

**Abstract**: Large-scale multimodal models achieve strong results on tasks like Visual Question Answering (VQA), but they often fail when queries require culturally grounded, everyday knowledge, particularly in low-resource and underrepresented languages. To bridge this gap, we introduce Everyday Multimodal and Multilingual QA (EverydayMMQA), a framework for creating large-scale, culturally-grounded datasets for spoken and visual question answering (SVQA). Using this framework, we developed OASIS, a multimodal dataset integrating speech, images, and text. With over ~0.92M images and 14.8M QA pairs, OASIS contains 3.7M spoken questions, enabling four unique input combinations: speech-only, text-only, speech+image, and text+image. Focused on English and Arabic varieties, 18 countries, the dataset content is curated to reflect diverse, real-world situations. OASIS tests models on tasks beyond object recognition that involve pragmatic, commonsense, and culturally aware reasoning. We benchmarked four closed-source models, three open-source models, and one fine-tuned model. EverydayMMQA and OASIS together provide a benchmark and training dataset for building multimodal LLMs for a comprehensive set of everyday tasks within cultural contexts. The framework and dataset will be made publicly available to the community. 

**Abstract (ZH)**: 大规模多模态模型在视觉问答（VQA）等任务上取得了出色的结果，但在要求文化背景下的日常生活知识时往往表现不佳，尤其是在低资源和少代表语言中。为弥合这一差距，我们引入了 Everyday Multimodal and Multilingual QA (EverydayMMQA) 框架，用于创建面向口语和视觉问答（SVQA）的大规模、文化背景扎根数据集。使用该框架，我们开发了 OASIS 多模态数据集，整合了语音、图像和文本。OASIS 包含超过 0.92 百万张图像和 1480 万 QA 对，其中包含 370 万条口语问题，支持四种独特的输入组合：语音仅、文本仅、语音+图像和文本+图像。该数据集专注于英语和阿拉伯语变体，包含来自 18 个国家的内容，内容策划以反映多元化的现实世界情境。OASIS 考察模型在涉及语用、常识和文化意识推理的任务上的表现，超越了对象识别。我们对四款封闭源代码模型、三款开源模型和一款微调模型进行了基准测试。EverydayMMQA 和 OASIS 一起提供了构建文化背景下涵盖广泛日常任务的多模态大语言模型的基准和训练数据集。框架和数据集将向社区公开。 

---
# ChainMPQ: Interleaved Text-Image Reasoning Chains for Mitigating Relation Hallucinations 

**Title (ZH)**: ChainMPQ: 交错的文本-图像推理链以减轻关系幻觉问题 

**Authors**: Yike Wu, Yiwei Wang, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2510.06292)  

**Abstract**: While Large Vision-Language Models (LVLMs) achieve strong performance in multimodal tasks, hallucinations continue to hinder their reliability. Among the three categories of hallucinations, which include object, attribute, and relation, relation hallucinations account for the largest proportion but have received the least attention. To address this issue, we propose ChainMPQ (Multi-Perspective Questions guided Interleaved Chain of Image and Text), a training-free method that improves relational inference in LVLMs by utilizing accumulated textual and visual memories. ChainMPQ first extracts subject and object keywords from the question to enhance the corresponding image regions. It then constructs multi-perspective questions that focus on the three core components of a relationship: the subject, the object, and the relation that links them. These questions are sequentially input to the model, with textual and visual memories from earlier steps providing supporting context for subsequent ones, thereby forming an interleaved chain of images and text that guides progressive relational reasoning. Experiments on multiple LVLMs and benchmarks show that ChainMPQ substantially reduces relation hallucinations, while ablation studies further validate the effectiveness of its three core modules. 

**Abstract (ZH)**: 大型多模态模型中的关系幻觉持续影响其可靠性，尽管大型视觉-语言模型在多模态任务中表现出色。在对象、属性和关系三种幻觉类别中，关系幻觉占比最大但受到的关注最少。为解决这一问题，我们提出ChainMPQ（多视角问题引导的图像和文本交错链），这是一种无需训练的方法，通过利用积累的文字和视觉记忆来改善大型视觉-语言模型的关系推理。ChainMPQ 首先从问题中提取主词和宾词关键词，以增强相应的图像区域。接着构建专注于关系三个核心组成部分（主词、宾词和连接它们的关系）的多视角问题。这些问题按顺序输入模型，早期步骤中的文字和视觉记忆为后续步骤提供支持性背景，从而形成图像和文本交织链，引导渐进式关系推理。在多个大型多模态模型和基准上的实验表明，ChainMPQ 显著减少了关系幻觉，而消融研究进一步验证了其三个核心模块的有效性。 

---
# Dream2Image : An Open Multimodal EEG Dataset for Decoding and Visualizing Dreams with Artificial Intelligence 

**Title (ZH)**: Dream2Image：一种用于使用人工智能解码和可视化梦境的开放多模态EEG数据集 

**Authors**: Yann Bellec  

**Link**: [PDF](https://arxiv.org/pdf/2510.06252)  

**Abstract**: Dream2Image is the world's first dataset combining EEG signals, dream transcriptions, and AI-generated images. Based on 38 participants and more than 31 hours of dream EEG recordings, it contains 129 samples offering: the final seconds of brain activity preceding awakening (T-15, T-30, T-60, T-120), raw reports of dream experiences, and an approximate visual reconstruction of the dream. This dataset provides a novel resource for dream research, a unique resource to study the neural correlates of dreaming, to develop models for decoding dreams from brain activity, and to explore new approaches in neuroscience, psychology, and artificial intelligence. Available in open access on Hugging Face and GitHub, Dream2Image provides a multimodal resource designed to support research at the interface of artificial intelligence and neuroscience. It was designed to inspire researchers and extend the current approaches to brain activity decoding. Limitations include the relatively small sample size and the variability of dream recall, which may affect generalizability. 

**Abstract (ZH)**: Dream2Image是首个结合EEG信号、梦境转述和AI生成图像的数据集。 

---
# Stacked Regression using Off-the-shelf, Stimulus-tuned and Fine-tuned Neural Networks for Predicting fMRI Brain Responses to Movies (Algonauts 2025 Report) 

**Title (ZH)**: 使用现成的、刺激调谐的和微调的神经网络堆叠回归方法预测电影观看引起的fMRI脑响应（Algonauts 2025报告） 

**Authors**: Robert Scholz, Kunal Bagga, Christine Ahrends, Carlo Alberto Barbano  

**Link**: [PDF](https://arxiv.org/pdf/2510.06235)  

**Abstract**: We present our submission to the Algonauts 2025 Challenge, where the goal is to predict fMRI brain responses to movie stimuli. Our approach integrates multimodal representations from large language models, video encoders, audio models, and vision-language models, combining both off-the-shelf and fine-tuned variants. To improve performance, we enhanced textual inputs with detailed transcripts and summaries, and we explored stimulus-tuning and fine-tuning strategies for language and vision models. Predictions from individual models were combined using stacked regression, yielding solid results. Our submission, under the team name Seinfeld, ranked 10th. We make all code and resources publicly available, contributing to ongoing efforts in developing multimodal encoding models for brain activity. 

**Abstract (ZH)**: 我们提交了对2025年Algonauts挑战赛的内容，目标是预测电影刺激下的fMRI脑部反应。我们的方法结合了大型语言模型、视频编码器、音频模型和视觉-语言模型的多模态表示，既包括现成的版本也包括微调后的版本。为了提高性能，我们通过详细的脚本和摘要增强了文本输入，并探索了语言模型和视觉模型的刺激调谐及微调策略。各模型的预测结果通过堆叠回归进行融合，取得了令人满意的结果。以Seinfeld团队名义提交的作品排名第十。我们已将所有代码和资源公开，为开发用于脑活动的多模态编码模型做出了贡献。 

---
