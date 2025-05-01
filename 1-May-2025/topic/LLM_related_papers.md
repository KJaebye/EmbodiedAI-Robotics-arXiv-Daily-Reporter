# LLM-based Interactive Imitation Learning for Robotic Manipulation 

**Title (ZH)**: 基于LLM的交互式模仿学习在机器人操作中的应用 

**Authors**: Jonas Werner, Kun Chu, Cornelius Weber, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2504.21769)  

**Abstract**: Recent advancements in machine learning provide methods to train autonomous agents capable of handling the increasing complexity of sequential decision-making in robotics. Imitation Learning (IL) is a prominent approach, where agents learn to control robots based on human demonstrations. However, IL commonly suffers from violating the independent and identically distributed (i.i.d) assumption in robotic tasks. Interactive Imitation Learning (IIL) achieves improved performance by allowing agents to learn from interactive feedback from human teachers. Despite these improvements, both approaches come with significant costs due to the necessity of human involvement. Leveraging the emergent capabilities of Large Language Models (LLMs) in reasoning and generating human-like responses, we introduce LLM-iTeach -- a novel IIL framework that utilizes an LLM as an interactive teacher to enhance agent performance while alleviating the dependence on human resources. Firstly, LLM-iTeach uses a hierarchical prompting strategy that guides the LLM in generating a policy in Python code. Then, with a designed similarity-based feedback mechanism, LLM-iTeach provides corrective and evaluative feedback interactively during the agent's training. We evaluate LLM-iTeach against baseline methods such as Behavior Cloning (BC), an IL method, and CEILing, a state-of-the-art IIL method using a human teacher, on various robotic manipulation tasks. Our results demonstrate that LLM-iTeach surpasses BC in the success rate and achieves or even outscores that of CEILing, highlighting the potential of LLMs as cost-effective, human-like teachers in interactive learning environments. We further demonstrate the method's potential for generalization by evaluating it on additional tasks. The code and prompts are provided at: this https URL. 

**Abstract (ZH)**: 新兴的机器学习技术为训练能够处理机器人领域日益复杂的序列决策的自主代理提供了方法。模仿学习（IL）是一种突出的方法，其中代理基于人类示范学习控制机器人。然而，IL 在机器人任务中通常会违反独立且同分布（i.i.d）的假设。互动模仿学习（IIL）通过允许代理从人类教师的互动反馈中学习来实现性能的提升。尽管如此，这两种方法都因为需要人类的参与而存在较大的成本。利用大型语言模型（LLMs）在推理和生成类似人类回应方面出现的能力，我们引入了 LLM-iTeach ——一种新颖的 IIL 框架，该框架利用 LLM 作为互动教师以增强代理性能并减轻对人力资源的依赖。首先，LLM-iTeach 使用分级提示策略来引导 LLM 生成 Python 代码策略。然后，通过设计的一种基于相似性的反馈机制，LLM-iTeach 在代理训练过程中提供纠正性和评价性反馈。我们在各种机器人操作任务上将 LLM-iTeach 与基线方法行为克隆（BC）、一种 IL 方法以及 CEILing、一种最先进的 IIL 方法（使用人类教师）进行了比较评估。结果显示，LLM-iTeach 在成功率上优于 BC，并且在某些情况下甚至优于 CEILing，突显了 LLM 作为互动学习环境中成本效益高且类似人类的教师的潜力。我们进一步通过额外任务评估展示了该方法的泛化能力。相关代码和提示可在以下链接获取：this https URL。 

---
# Leveraging Pre-trained Large Language Models with Refined Prompting for Online Task and Motion Planning 

**Title (ZH)**: 利用细粒度提示强化预训练大型语言模型进行在线任务与运动规划 

**Authors**: Huihui Guo, Huilong Pi, Yunchuan Qin, Zhuo Tang, Kenli Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.21596)  

**Abstract**: With the rapid advancement of artificial intelligence, there is an increasing demand for intelligent robots capable of assisting humans in daily tasks and performing complex operations. Such robots not only require task planning capabilities but must also execute tasks with stability and robustness. In this paper, we present a closed-loop task planning and acting system, LLM-PAS, which is assisted by a pre-trained Large Language Model (LLM). While LLM-PAS plans long-horizon tasks in a manner similar to traditional task and motion planners, it also emphasizes the execution phase of the task. By transferring part of the constraint-checking process from the planning phase to the execution phase, LLM-PAS enables exploration of the constraint space and delivers more accurate feedback on environmental anomalies during execution. The reasoning capabilities of the LLM allow it to handle anomalies that cannot be addressed by the robust executor. To further enhance the system's ability to assist the planner during replanning, we propose the First Look Prompting (FLP) method, which induces LLM to generate effective PDDL goals. Through comparative prompting experiments and systematic experiments, we demonstrate the effectiveness and robustness of LLM-PAS in handling anomalous conditions during task execution. 

**Abstract (ZH)**: 随着人工智能的迅速发展，对能够协助人类完成日常任务并执行复杂操作的智能机器人需求不断增加。这类机器人不仅需要具备任务规划能力，还需在执行任务时表现出稳定性和鲁棒性。本文提出了一种闭环任务规划与执行系统LLM-PAS，该系统借助预训练的大语言模型（LLM）的支持。虽然LLM-PAS在任务规划阶段以类似于传统任务与运动规划器的方式进行长时间规划，但它更侧重于任务的执行阶段。通过将部分约束检查过程从规划阶段转移到执行阶段，LLM-PAS能够探索约束空间并在执行过程中提供更准确的环境异常反馈。大语言模型的推理能力使其能够处理鲁棒执行器无法解决的异常。为进一步增强系统在重新规划期间辅助规划者的能力，我们提出了初次观察提示（FLP）方法，诱导大语言模型生成有效的PDDL目标。通过比较性提示实验和系统性实验，我们证实了LLM-PAS在任务执行过程中处理异常条件的有效性和鲁棒性。 

---
# Phi-4-reasoning Technical Report 

**Title (ZH)**: Phi-4推理技术报告 

**Authors**: Marah Abdin, Sahaj Agarwal, Ahmed Awadallah, Vidhisha Balachandran, Harkirat Behl, Lingjiao Chen, Gustavo de Rosa, Suriya Gunasekar, Mojan Javaheripi, Neel Joshi, Piero Kauffmann, Yash Lara, Caio César Teodoro Mendes, Arindam Mitra, Besmira Nushi, Dimitris Papailiopoulos, Olli Saarikivi, Shital Shah, Vaishnavi Shrivastava, Vibhav Vineet, Yue Wu, Safoora Yousefi, Guoqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.21318)  

**Abstract**: We introduce Phi-4-reasoning, a 14-billion parameter reasoning model that achieves strong performance on complex reasoning tasks. Trained via supervised fine-tuning of Phi-4 on carefully curated set of "teachable" prompts-selected for the right level of complexity and diversity-and reasoning demonstrations generated using o3-mini, Phi-4-reasoning generates detailed reasoning chains that effectively leverage inference-time compute. We further develop Phi-4-reasoning-plus, a variant enhanced through a short phase of outcome-based reinforcement learning that offers higher performance by generating longer reasoning traces. Across a wide range of reasoning tasks, both models outperform significantly larger open-weight models such as DeepSeek-R1-Distill-Llama-70B model and approach the performance levels of full DeepSeek-R1 model. Our comprehensive evaluations span benchmarks in math and scientific reasoning, coding, algorithmic problem solving, planning, and spatial understanding. Interestingly, we observe a non-trivial transfer of improvements to general-purpose benchmarks as well. In this report, we provide insights into our training data, our training methodologies, and our evaluations. We show that the benefit of careful data curation for supervised fine-tuning (SFT) extends to reasoning language models, and can be further amplified by reinforcement learning (RL). Finally, our evaluation points to opportunities for improving how we assess the performance and robustness of reasoning models. 

**Abstract (ZH)**: Phi-4-reasoning及其增强模型的推理研究：一种140亿参数的强性能复杂推理模型 

---
# TRUST: An LLM-Based Dialogue System for Trauma Understanding and Structured Assessments 

**Title (ZH)**: TRUST：基于LLM的创伤理解与结构化评估对话系统 

**Authors**: Sichang Tu, Abigail Powers, Stephen Doogan, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.21851)  

**Abstract**: Objectives: While Large Language Models (LLMs) have been widely used to assist clinicians and support patients, no existing work has explored dialogue systems for standard diagnostic interviews and assessments. This study aims to bridge the gap in mental healthcare accessibility by developing an LLM-powered dialogue system that replicates clinician behavior. Materials and Methods: We introduce TRUST, a framework of cooperative LLM modules capable of conducting formal diagnostic interviews and assessments for Post-Traumatic Stress Disorder (PTSD). To guide the generation of appropriate clinical responses, we propose a Dialogue Acts schema specifically designed for clinical interviews. Additionally, we develop a patient simulation approach based on real-life interview transcripts to replace time-consuming and costly manual testing by clinicians. Results: A comprehensive set of evaluation metrics is designed to assess the dialogue system from both the agent and patient simulation perspectives. Expert evaluations by conversation and clinical specialists show that TRUST performs comparably to real-life clinical interviews. Discussion: Our system performs at the level of average clinicians, with room for future enhancements in communication styles and response appropriateness. Conclusions: Our TRUST framework shows its potential to facilitate mental healthcare availability. 

**Abstract (ZH)**: 目标：虽然大型语言模型（LLMs）已被广泛用于辅助临床医生并支持患者，但目前尚未有研究探索用于标准化诊断访谈和评估的对话系统。本研究旨在通过开发一个由LLM驱动的对话系统来弥补心理健康服务可及性的缺口，该对话系统能够模仿临床医生的行为。材料与方法：我们介绍了TRUST框架，这是一种协作的LLM模块体系，能够进行创伤后应激障碍（PTSD）的标准诊断访谈和评估。为了引导生成合适的临床回应，我们提出了一个专门设计用于临床访谈的对话行为方案。此外，我们还开发了一种基于真实访谈记录的患者模拟方法，以替代耗时且成本高昂的临床手动测试。结果：设计了一套全面的评估指标，从代理和患者模拟的角度评估对话系统的表现。由对话和临床专家进行的专家评估表明，TRUST的表现与现实生活中的临床访谈相当。讨论：我们的系统达到了平均水平临床医生的水平，未来在沟通风格和回应适当性方面仍有机会进一步改进。结论：我们的TRUST框架展示了其在促进心理健康服务可及性方面的潜力。 

---
# MAC-Tuning: LLM Multi-Compositional Problem Reasoning with Enhanced Knowledge Boundary Awareness 

**Title (ZH)**: MAC-调谐：增强知识边界意识的LLM多组件问题推理 

**Authors**: Junsheng Huang, Zhitao He, Sandeep Polisetty, Qingyun Wang, May Fung  

**Link**: [PDF](https://arxiv.org/pdf/2504.21773)  

**Abstract**: With the widespread application of large language models (LLMs), the issue of generating non-existing facts, known as hallucination, has garnered increasing attention. Previous research in enhancing LLM confidence estimation mainly focuses on the single problem setting. However, LLM awareness of its internal parameterized knowledge boundary under the more challenging multi-problem setting, which requires answering multiple problems accurately simultaneously, remains underexplored. To bridge this gap, we introduce a novel method, Multiple Answers and Confidence Stepwise Tuning (MAC-Tuning), that separates the learning of answer prediction and confidence estimation during fine-tuning on instruction data. Extensive experiments demonstrate that our method outperforms baselines by up to 25% in average precision. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的广泛应用，生成非存在事实即幻觉的问题越来越受到关注。增强LLM置信度估计的研究主要集中在单一问题设置上。然而，在需要同时准确回答多个问题的更具挑战性的多问题设置下，LLM对其内部参数化知识边界的意识尚未得到充分探索。为了弥合这一差距，我们提出了一种名为Multiple Answers and Confidence Stepwise Tuning（MAC-Tuning）的新方法，在指令数据微调过程中分开学习答案预测和置信度估计。大量实验显示，与基线方法相比，我们的方法在平均精度上最高可提高25%。 

---
# XBreaking: Explainable Artificial Intelligence for Jailbreaking LLMs 

**Title (ZH)**: XBreaking: 可解释的人工智能在破环LLMs中的应用 

**Authors**: Marco Arazzi, Vignesh Kumar Kembu, Antonino Nocera, Vinod P  

**Link**: [PDF](https://arxiv.org/pdf/2504.21700)  

**Abstract**: Large Language Models are fundamental actors in the modern IT landscape dominated by AI solutions. However, security threats associated with them might prevent their reliable adoption in critical application scenarios such as government organizations and medical institutions. For this reason, commercial LLMs typically undergo a sophisticated censoring mechanism to eliminate any harmful output they could possibly produce. In response to this, LLM Jailbreaking is a significant threat to such protections, and many previous approaches have already demonstrated its effectiveness across diverse domains. Existing jailbreak proposals mostly adopt a generate-and-test strategy to craft malicious input. To improve the comprehension of censoring mechanisms and design a targeted jailbreak attack, we propose an Explainable-AI solution that comparatively analyzes the behavior of censored and uncensored models to derive unique exploitable alignment patterns. Then, we propose XBreaking, a novel jailbreak attack that exploits these unique patterns to break the security constraints of LLMs by targeted noise injection. Our thorough experimental campaign returns important insights about the censoring mechanisms and demonstrates the effectiveness and performance of our attack. 

**Abstract (ZH)**: 大型语言模型是现代由AI解决方案主导的IT景观中的基础参与者。然而，与它们相关的安全威胁可能阻止它们在关键应用场景如政府组织和医疗机构中的可靠采用。因此，商业大型语言模型通常会经历一种复杂的过滤机制以消除可能的危害输出。对此，大型语言模型的破解（Jailbreaking）是对这种保护的重大威胁，许多先前的方法已经在不同领域证明了其有效性。现有的破解提案主要采用生成并测试的策略来生成恶意输入。为了提高对过滤机制的理解并设计有针对性的破解攻击，我们提出了一种可解释AI解决方案，该方案通过比较分析过滤和未过滤模型的行为来推导出独特的可利用对齐模式。然后，我们提出XBreaking，这是一种新型的破解攻击，它利用这些独特的模式通过有针对性的噪声注入破坏大型语言模型的安全限制。我们的全面实验战役为我们提供了关于过滤机制的重要见解，并证明了我们攻击的有效性和性能。 

---
# Sadeed: Advancing Arabic Diacritization Through Small Language Model 

**Title (ZH)**: Sadeed: 通过小语言模型提升阿拉伯语重音标注技术 

**Authors**: Zeina Aldallal, Sara Chrouf, Khalil Hennara, Mohamed Motaism Hamed, Muhammad Hreden, Safwan AlModhayan  

**Link**: [PDF](https://arxiv.org/pdf/2504.21635)  

**Abstract**: Arabic text diacritization remains a persistent challenge in natural language processing due to the language's morphological richness. In this paper, we introduce Sadeed, a novel approach based on a fine-tuned decoder-only language model adapted from Kuwain 1.5B Hennara et al. [2025], a compact model originally trained on diverse Arabic corpora. Sadeed is fine-tuned on carefully curated, high-quality diacritized datasets, constructed through a rigorous data-cleaning and normalization pipeline. Despite utilizing modest computational resources, Sadeed achieves competitive results compared to proprietary large language models and outperforms traditional models trained on similar domains. Additionally, we highlight key limitations in current benchmarking practices for Arabic diacritization. To address these issues, we introduce SadeedDiac-25, a new benchmark designed to enable fairer and more comprehensive evaluation across diverse text genres and complexity levels. Together, Sadeed and SadeedDiac-25 provide a robust foundation for advancing Arabic NLP applications, including machine translation, text-to-speech, and language learning tools. 

**Abstract (ZH)**: 阿拉伯文本重音标记仍然是自然语言处理中的一个持久性挑战，归因于该语言丰富的形态特征。本文介绍了一种名为Sadeed的新型方法，该方法基于经过微调的仅解码器语言模型，该模型源自Kuwain 1.5B Hennara等[2025]，一个最初在多样化的阿拉伯语语料库上进行训练的紧凑型模型。Sadeed在精心策划的高质量重音标记数据集上进行微调，这些数据集通过严格的数据清洗和规范化管道构建。尽管使用了有限的计算资源，Sadeed在与专有大型语言模型相比时取得了竞争力的结果，并且在相似领域训练的传统模型中表现更佳。此外，我们还指出了当前阿拉伯语重音标记基准测试做法中的关键局限性。为了解决这些问题，我们引入了SadeedDiac-25，这是一种新的基准测试，旨在促进对多种文本类型和复杂程度的公平和全面评估。Sadeed与SadeedDiac-25共同为推进阿拉伯语NLP应用（包括机器翻译、文本-to-语音以及语言学习工具）提供了坚实的基础。 

---
# RDF-Based Structured Quality Assessment Representation of Multilingual LLM Evaluations 

**Title (ZH)**: 基于RDF的多语言大模型评估结构化质量表示 

**Authors**: Jonas Gwozdz, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2504.21605)  

**Abstract**: Large Language Models (LLMs) increasingly serve as knowledge interfaces, yet systematically assessing their reliability with conflicting information remains difficult. We propose an RDF-based framework to assess multilingual LLM quality, focusing on knowledge conflicts. Our approach captures model responses across four distinct context conditions (complete, incomplete, conflicting, and no-context information) in German and English. This structured representation enables the comprehensive analysis of knowledge leakage-where models favor training data over provided context-error detection, and multilingual consistency. We demonstrate the framework through a fire safety domain experiment, revealing critical patterns in context prioritization and language-specific performance, and demonstrating that our vocabulary was sufficient to express every assessment facet encountered in the 28-question study. 

**Abstract (ZH)**: 基于RDF的多语言大型语言模型质量评估框架：聚焦知识冲突分析 

---
# DNB-AI-Project at SemEval-2025 Task 5: An LLM-Ensemble Approach for Automated Subject Indexing 

**Title (ZH)**: DNB-AI-Project在SemEval-2025任务5中的LLM集成方法Automated主题索引研究 

**Authors**: Lisa Kluge, Maximilian Kähler  

**Link**: [PDF](https://arxiv.org/pdf/2504.21589)  

**Abstract**: This paper presents our system developed for the SemEval-2025 Task 5: LLMs4Subjects: LLM-based Automated Subject Tagging for a National Technical Library's Open-Access Catalog. Our system relies on prompting a selection of LLMs with varying examples of intellectually annotated records and asking the LLMs to similarly suggest keywords for new records. This few-shot prompting technique is combined with a series of post-processing steps that map the generated keywords to the target vocabulary, aggregate the resulting subject terms to an ensemble vote and, finally, rank them as to their relevance to the record. Our system is fourth in the quantitative ranking in the all-subjects track, but achieves the best result in the qualitative ranking conducted by subject indexing experts. 

**Abstract (ZH)**: SemEval-2025 任务5：LLM4Subjects：基于LLM的国家技术图书馆开放访问目录主题标签自动化系统 

---
# MF-LLM: Simulating Collective Decision Dynamics via a Mean-Field Large Language Model Framework 

**Title (ZH)**: MF-LLM：通过大规模语言模型框架模拟集体决策动力学 

**Authors**: Qirui Mi, Mengyue Yang, Xiangning Yu, Zhiyu Zhao, Cheng Deng, Bo An, Haifeng Zhang, Xu Chen, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21582)  

**Abstract**: Simulating collective decision-making involves more than aggregating individual behaviors; it arises from dynamic interactions among individuals. While large language models (LLMs) show promise for social simulation, existing approaches often exhibit deviations from real-world data. To address this gap, we propose the Mean-Field LLM (MF-LLM) framework, which explicitly models the feedback loop between micro-level decisions and macro-level population. MF-LLM alternates between two models: a policy model that generates individual actions based on personal states and group-level information, and a mean field model that updates the population distribution from the latest individual decisions. Together, they produce rollouts that simulate the evolving trajectories of collective decision-making. To better match real-world data, we introduce IB-Tune, a fine-tuning method for LLMs grounded in the information bottleneck principle, which maximizes the relevance of population distributions to future actions while minimizing redundancy with historical data. We evaluate MF-LLM on a real-world social dataset, where it reduces KL divergence to human population distributions by 47 percent over non-mean-field baselines, and enables accurate trend forecasting and intervention planning. It generalizes across seven domains and four LLM backbones, providing a scalable foundation for high-fidelity social simulation. 

**Abstract (ZH)**: 集体决策模拟不仅涉及个体行为的聚合，还源于个体间的动态交互。虽然大规模语言模型（LLMs）在社会模拟方面具有潜力，但现有方法往往与真实世界数据存在偏差。为此，我们提出了均场LLM（MF-LLM）框架，该框架明确建模了微观层面决策与宏观层面群体之间的反馈循环。MF-LLM 交替运行两个模型：策略模型基于个人状态和群体层面信息生成个体行为，均场模型更新群体分布以反映最新的个体决策。两者结合生成模拟集体决策演变轨迹的 rollout。为了更好地匹配真实世界数据，我们引入了基于信息瓶颈原则的 IB-Tune 微调方法，该方法最大限度地提高了群体分布对未来行为的相关性，同时减少了与历史数据的冗余性。我们在一个实际社会数据集上评估 MF-LLM，结果显示它在相对于非均场基准方法的 KL 散度上降低了47%，并能实现准确的趋势预测和干预规划。该方法在七个领域和四种 LLM 主干上具有泛化能力，为我们提供了高性能社会模拟的可扩展基础。 

---
# Rethinking Visual Layer Selection in Multimodal LLMs 

**Title (ZH)**: 重塑多模态LLM中视觉层的选择 

**Authors**: Haoran Chen, Junyan Lin, Xinhao Chen, Yue Fan, Xin Jin, Hui Su, Jianfeng Dong, Jinlan Fu, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21447)  

**Abstract**: Multimodal large language models (MLLMs) have achieved impressive performance across a wide range of tasks, typically using CLIP-ViT as their visual encoder due to its strong text-image alignment capabilities. While prior studies suggest that different CLIP-ViT layers capture different types of information, with shallower layers focusing on fine visual details and deeper layers aligning more closely with textual semantics, most MLLMs still select visual features based on empirical heuristics rather than systematic analysis. In this work, we propose a Layer-wise Representation Similarity approach to group CLIP-ViT layers with similar behaviors into {shallow, middle, and deep} categories and assess their impact on MLLM performance. Building on this foundation, we revisit the visual layer selection problem in MLLMs at scale, training LLaVA-style models ranging from 1.4B to 7B parameters. Through extensive experiments across 10 datasets and 4 tasks, we find that: (1) deep layers are essential for OCR tasks; (2) shallow and middle layers substantially outperform deep layers on reasoning tasks involving counting, positioning, and object localization; (3) a lightweight fusion of features across shallow, middle, and deep layers consistently outperforms specialized fusion baselines and single-layer selections, achieving gains on 9 out of 10 datasets. Our work offers the first principled study of visual layer selection in MLLMs, laying the groundwork for deeper investigations into visual representation learning for MLLMs. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在广泛的任务中表现出了显著的效果，通常使用CLIP-ViT作为其视觉编码器，这得益于其强大的文本-图像对齐能力。尽管前期研究指出不同CLIP-ViT层捕获不同类型的信息，浅层层侧重于精细的视觉细节，而深层层更紧密地与文本语义对齐，但大多数MLLMs仍然基于经验启发/rules选择视觉特征，而非系统的分析。在本工作中，我们提出了一种逐层表示相似性方法，将具有类似行为的CLIP-ViT层分为浅层、中层和深层三类，并评估它们对MLLM性能的影响。在此基础上，我们对大规模MLLM中视觉层的选择问题进行了重新审视，训练了从1.4B到7B参数的LLaVA风格模型。通过在10个数据集和4个任务上的广泛实验，我们发现：（1）深层层对于OCR任务是必不可少的；（2）浅层和中层显著优于深层层，在涉及计数、定位和对象定位的推理任务中表现更佳；（3）跨浅层、中层和深层层的轻量级特征融合始终优于专门的特征融合基准和单层选择，有9个数据集上取得了改进。我们的工作首次系统地研究了MLLM中的视觉层选择问题，为MLLM中的视觉表示学习的深入研究奠定了基础。 

---
# Retrieval-Enhanced Few-Shot Prompting for Speech Event Extraction 

**Title (ZH)**: 基于检索增强的少量示例提示语音事件提取 

**Authors**: Máté Gedeon  

**Link**: [PDF](https://arxiv.org/pdf/2504.21372)  

**Abstract**: Speech Event Extraction (SpeechEE) is a challenging task that lies at the intersection of Automatic Speech Recognition (ASR) and Natural Language Processing (NLP), requiring the identification of structured event information from spoken language. In this work, we present a modular, pipeline-based SpeechEE framework that integrates high-performance ASR with semantic search-enhanced prompting of Large Language Models (LLMs). Our system first classifies speech segments likely to contain events using a hybrid filtering mechanism including rule-based, BERT-based, and LLM-based models. It then employs few-shot LLM prompting, dynamically enriched via semantic similarity retrieval, to identify event triggers and extract corresponding arguments. We evaluate the pipeline using multiple LLMs (Llama3-8B, GPT-4o-mini, and o1-mini) highlighting significant performance gains with o1-mini, which achieves 63.3% F1 on trigger classification and 27.8% F1 on argument classification, outperforming prior benchmarks. Our results demonstrate that pipeline approaches, when empowered by retrieval-augmented LLMs, can rival or exceed end-to-end systems while maintaining interpretability and modularity. This work provides practical insights into LLM-driven event extraction and opens pathways for future hybrid models combining textual and acoustic features. 

**Abstract (ZH)**: 基于检索增强的大语言模型的模块化语音事件抽取框架 

---
# Assessing LLM code generation quality through path planning tasks 

**Title (ZH)**: 通过路径规划任务评估LLM代码生成质量 

**Authors**: Wanyi Chen, Meng-Wen Su, Mary L. Cummings  

**Link**: [PDF](https://arxiv.org/pdf/2504.21276)  

**Abstract**: As LLM-generated code grows in popularity, more evaluation is needed to assess the risks of using such tools, especially for safety-critical applications such as path planning. Existing coding benchmarks are insufficient as they do not reflect the context and complexity of safety-critical applications. To this end, we assessed six LLMs' abilities to generate the code for three different path-planning algorithms and tested them on three maps of various difficulties. Our results suggest that LLM-generated code presents serious hazards for path planning applications and should not be applied in safety-critical contexts without rigorous testing. 

**Abstract (ZH)**: 随着大语言模型生成的代码日益流行，需要进行更多评估以评估使用这些工具的风险，特别是在路径规划等关键安全应用中。现有的编码基准不足以反映关键安全应用的上下文和复杂性。为此，我们评估了六种大语言模型生成三种不同路径规划算法代码的能力，并在三种不同难度的地图上进行了测试。结果显示，大语言模型生成的代码对路径规划应用存在严重风险，在关键安全情境下不应未经严格测试就使用。 

---
# Memorization and Knowledge Injection in Gated LLMs 

**Title (ZH)**: 在门控大型语言模型中的记忆与知识注入 

**Authors**: Xu Pan, Ely Hahami, Zechen Zhang, Haim Sompolinsky  

**Link**: [PDF](https://arxiv.org/pdf/2504.21239)  

**Abstract**: Large Language Models (LLMs) currently struggle to sequentially add new memories and integrate new knowledge. These limitations contrast with the human ability to continuously learn from new experiences and acquire knowledge throughout life. Most existing approaches add memories either through large context windows or external memory buffers (e.g., Retrieval-Augmented Generation), and studies on knowledge injection rarely test scenarios resembling everyday life events. In this work, we introduce a continual learning framework, Memory Embedded in Gated LLMs (MEGa), which injects event memories directly into the weights of LLMs. Each memory is stored in a dedicated set of gated low-rank weights. During inference, a gating mechanism activates relevant memory weights by matching query embeddings to stored memory embeddings. This enables the model to both recall entire memories and answer related questions. On two datasets - fictional characters and Wikipedia events - MEGa outperforms baseline approaches in mitigating catastrophic forgetting. Our model draws inspiration from the complementary memory system of the human brain. 

**Abstract (ZH)**: Large Language Models (LLMs)目前难以顺序添加新记忆并整合新知识。这些限制与人类能够不断从新经历中学习并在一生中不断获取知识的能力形成对比。现有大多数方法通过大上下文窗口或外部记忆缓冲区（例如检索增强生成）来添加记忆，而知识注入的相关研究鲜少测试类似于日常生活事件的场景。本文引入了一种连续学习框架，即嵌入门控LLMs的记忆（MEGa），直接将事件记忆注入到LLMs的权重中。每个记忆存储于一组专用的门控低秩权重中。在推理过程中，门控机制通过匹配查询嵌入和存储的记忆嵌入来激活相关记忆权重，使得模型能够回忆完整记忆并回答相关问题。在两个数据集中——虚构角色和维基百科事件——MEGa在减轻灾难性遗忘方面优于基线方法。我们的模型受到人类互补记忆系统的启发。 

---
# CachePrune: Neural-Based Attribution Defense Against Indirect Prompt Injection Attacks 

**Title (ZH)**: CachePrune: 基于神经网络的归因防御以对抗间接提示注入攻击 

**Authors**: Rui Wang, Junda Wu, Yu Xia, Tong Yu, Ruiyi Zhang, Ryan Rossi, Lina Yao, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2504.21228)  

**Abstract**: Large Language Models (LLMs) are identified as being susceptible to indirect prompt injection attack, where the model undesirably deviates from user-provided instructions by executing tasks injected in the prompt context. This vulnerability stems from LLMs' inability to distinguish between data and instructions within a prompt. In this paper, we propose CachePrune that defends against this attack by identifying and pruning task-triggering neurons from the KV cache of the input prompt context. By pruning such neurons, we encourage the LLM to treat the text spans of input prompt context as only pure data, instead of any indicator of instruction following. These neurons are identified via feature attribution with a loss function induced from an upperbound of the Direct Preference Optimization (DPO) objective. We show that such a loss function enables effective feature attribution with only a few samples. We further improve on the quality of feature attribution, by exploiting an observed triggering effect in instruction following. Our approach does not impose any formatting on the original prompt or introduce extra test-time LLM calls. Experiments show that CachePrune significantly reduces attack success rates without compromising the response quality. Note: This paper aims to defend against indirect prompt injection attacks, with the goal of developing more secure and robust AI systems. 

**Abstract (ZH)**: 大规模语言模型（LLMs）被识别为易受间接提示注入攻击的影响，模型可能会在提示上下文中执行注入的任务，从而偏离用户提供的指令。这种脆弱性源于LLMs无法区分提示中的数据和指令。在本文中，我们提出了CachePrune，通过识别并修剪输入提示上下文的KV缓存中的任务触发神经元来防御这种攻击。通过修剪这些神经元，我们鼓励LLM将输入提示上下文的文本跨度视为纯粹的数据，而不是任何指示指令的标志。这些神经元通过从直接偏好优化（DPO）目标的上界诱导的损失函数进行特征归因来识别。我们证明这种损失函数能够在少量样本的情况下实现有效的特征归因。我们还通过利用观察到的指令跟随触发效应进一步提高了特征归因的质量。我们的方法不对原始提示进行格式化，也不引入额外的测试时间LLM调用。实验表明，CachePrune在不牺牲响应质量的情况下显著降低了攻击成功率。 

---
# A Cost-Effective LLM-based Approach to Identify Wildlife Trafficking in Online Marketplaces 

**Title (ZH)**: 基于LLM的成本有效方法识别在线市场中的野生动物走私 

**Authors**: Juliana Barbosa, Ulhas Gondhali, Gohar Petrossian, Kinshuk Sharma, Sunandan Chakraborty, Jennifer Jacquet, Juliana Freire  

**Link**: [PDF](https://arxiv.org/pdf/2504.21211)  

**Abstract**: Wildlife trafficking remains a critical global issue, significantly impacting biodiversity, ecological stability, and public health. Despite efforts to combat this illicit trade, the rise of e-commerce platforms has made it easier to sell wildlife products, putting new pressure on wild populations of endangered and threatened species. The use of these platforms also opens a new opportunity: as criminals sell wildlife products online, they leave digital traces of their activity that can provide insights into trafficking activities as well as how they can be disrupted. The challenge lies in finding these traces. Online marketplaces publish ads for a plethora of products, and identifying ads for wildlife-related products is like finding a needle in a haystack. Learning classifiers can automate ad identification, but creating them requires costly, time-consuming data labeling that hinders support for diverse ads and research questions. This paper addresses a critical challenge in the data science pipeline for wildlife trafficking analytics: generating quality labeled data for classifiers that select relevant data. While large language models (LLMs) can directly label advertisements, doing so at scale is prohibitively expensive. We propose a cost-effective strategy that leverages LLMs to generate pseudo labels for a small sample of the data and uses these labels to create specialized classification models. Our novel method automatically gathers diverse and representative samples to be labeled while minimizing the labeling costs. Our experimental evaluation shows that our classifiers achieve up to 95% F1 score, outperforming LLMs at a lower cost. We present real use cases that demonstrate the effectiveness of our approach in enabling analyses of different aspects of wildlife trafficking. 

**Abstract (ZH)**: 野生动物交易仍然是一个关键的全球问题，严重影响生物多样性、生态稳定性和公共卫生。尽管采取了打击这一非法贸易的努力，电子商务平台的兴起使得野生动物制品的销售变得更加容易，对受威胁和濒危物种的野生种群施加了新的压力。同时，这些平台也为打击行动提供了新的机会：随着犯罪分子在网上出售野生动物制品，他们会留下数字痕迹，这些痕迹可以提供关于非法交易活动及其如何被中断的洞察。挑战在于发现这些痕迹。在线市场发布各种产品的广告，识别与野生动物相关的广告就像是在 haystack 中找 needles。利用学习分类器可以自动化广告识别，但创建它们需要耗时费力的数据标注，阻碍了对多样化广告和研究问题的支持。本文解决了数据科学管道中野生动物交易分析的关键挑战：为分类器生成高质量的标注数据以选择相关数据。虽然大型语言模型（LLMs）可以直接标注广告，但大规模这样做成本过高。我们提出了一种成本效益高的策略，利用LLMs生成数据的小样本的伪标签，并使用这些标签创建专门的分类模型。我们提出的新方法可以自动收集多样化的代表性样本进行标注，同时尽量减少标注成本。我们的实验评估表明，我们的分类器取得了高达95%的F1分数，成本比LLMs更低。我们展示了实际案例，证明了该方法在使不同方面野生动物交易分析成为可能方面的有效性。 

---
# SecRepoBench: Benchmarking LLMs for Secure Code Generation in Real-World Repositories 

**Title (ZH)**: SecRepoBench：面向真实世界代码仓库的LLM安全代码生成基准测试 

**Authors**: Connor Dilgren, Purva Chiniya, Luke Griffith, Yu Ding, Yizheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.21205)  

**Abstract**: This paper introduces SecRepoBench, a benchmark to evaluate LLMs on secure code generation in real-world repositories. SecRepoBench has 318 code generation tasks in 27 C/C++ repositories, covering 15 CWEs. We evaluate 19 state-of-the-art LLMs using our benchmark and find that the models struggle with generating correct and secure code. In addition, the performance of LLMs to generate self-contained programs as measured by prior benchmarks do not translate to comparative performance at generating secure and correct code at the repository level in SecRepoBench. We show that the state-of-the-art prompt engineering techniques become less effective when applied to the repository level secure code generation problem. We conduct extensive experiments, including an agentic technique to generate secure code, to demonstrate that our benchmark is currently the most difficult secure coding benchmark, compared to previous state-of-the-art benchmarks. Finally, our comprehensive analysis provides insights into potential directions for enhancing the ability of LLMs to generate correct and secure code in real-world repositories. 

**Abstract (ZH)**: SecRepoBench：一个用于评估LLM在真实世界代码库中安全代码生成性能的标准基准 

---
# Automatic Legal Writing Evaluation of LLMs 

**Title (ZH)**: 自动评估大语言模型的法律写作能力 

**Authors**: Ramon Pires, Roseval Malaquias Junior, Rodrigo Nogueira  

**Link**: [PDF](https://arxiv.org/pdf/2504.21202)  

**Abstract**: Despite the recent advances in Large Language Models, benchmarks for evaluating legal writing remain scarce due to the inherent complexity of assessing open-ended responses in this domain. One of the key challenges in evaluating language models on domain-specific tasks is finding test datasets that are public, frequently updated, and contain comprehensive evaluation guidelines. The Brazilian Bar Examination meets these requirements. We introduce oab-bench, a benchmark comprising 105 questions across seven areas of law from recent editions of the exam. The benchmark includes comprehensive evaluation guidelines and reference materials used by human examiners to ensure consistent grading. We evaluate the performance of four LLMs on oab-bench, finding that Claude-3.5 Sonnet achieves the best results with an average score of 7.93 out of 10, passing all 21 exams. We also investigated whether LLMs can serve as reliable automated judges for evaluating legal writing. Our experiments show that frontier models like OpenAI's o1 achieve a strong correlation with human scores when evaluating approved exams, suggesting their potential as reliable automated evaluators despite the inherently subjective nature of legal writing assessment. The source code and the benchmark -- containing questions, evaluation guidelines, model-generated responses, and their respective automated evaluations -- are publicly available. 

**Abstract (ZH)**: 尽管近年来大型语言模型取得了进展，但由于评估法律写作固有的复杂性，该领域的基准测试仍然稀缺。在评估语言模型的领域特定任务时，找到公共的、频繁更新的测试数据集并包含全面的评价指南是一项关键挑战。巴西律师资格考试符合这些要求。我们介绍了一个名为oab-bench的基准测试，包含来自最近几版考试的105个问题，涵盖七个法律领域。该基准测试包括全面的评价指南和由人类考官使用的参考材料，以确保评分的一致性。我们评估了四个大型语言模型在oab-bench上的性能，发现Claude-3.5 Sonnet平均得分为7.93（满分为10分），通过了全部21场考试。我们还研究了语言模型是否可以作为可靠的自动法官用于评估法律写作。实验结果表明，先锋模型如OpenAI的o1在评估批准的考试时与人工评分有较强的关联性，这表明它们有可能作为可靠的自动评估器，尽管法律写作评估本质上具有主观性。基准测试的源代码和包含问题、评价指南、模型生成的回答及其相应的自动评估等内容的数据集均已公开。 

---
# Small or Large? Zero-Shot or Finetuned? Guiding Language Model Choice for Specialized Applications in Healthcare 

**Title (ZH)**: 小模型还是大模型？零样本还是微调？指导医疗健康领域专业化应用的语言模型选择 

**Authors**: Lovedeep Gondara, Jonathan Simkin, Graham Sayle, Shebnum Devji, Gregory Arbour, Raymond Ng  

**Link**: [PDF](https://arxiv.org/pdf/2504.21191)  

**Abstract**: This study aims to guide language model selection by investigating: 1) the necessity of finetuning versus zero-shot usage, 2) the benefits of domain-adjacent versus generic pretrained models, 3) the value of further domain-specific pretraining, and 4) the continued relevance of Small Language Models (SLMs) compared to Large Language Models (LLMs) for specific tasks. Using electronic pathology reports from the British Columbia Cancer Registry (BCCR), three classification scenarios with varying difficulty and data size are evaluated. Models include various SLMs and an LLM. SLMs are evaluated both zero-shot and finetuned; the LLM is evaluated zero-shot only. Finetuning significantly improved SLM performance across all scenarios compared to their zero-shot results. The zero-shot LLM outperformed zero-shot SLMs but was consistently outperformed by finetuned SLMs. Domain-adjacent SLMs generally performed better than the generic SLM after finetuning, especially on harder tasks. Further domain-specific pretraining yielded modest gains on easier tasks but significant improvements on the complex, data-scarce task. The results highlight the critical role of finetuning for SLMs in specialized domains, enabling them to surpass zero-shot LLM performance on targeted classification tasks. Pretraining on domain-adjacent or domain-specific data provides further advantages, particularly for complex problems or limited finetuning data. While LLMs offer strong zero-shot capabilities, their performance on these specific tasks did not match that of appropriately finetuned SLMs. In the era of LLMs, SLMs remain relevant and effective, offering a potentially superior performance-resource trade-off compared to LLMs. 

**Abstract (ZH)**: 本研究旨在通过探讨以下内容来指导语言模型的选择：1）微调与零样本使用之间的必要性，2）领域相邻模型与通用预训练模型的优势，3）进一步领域特定预训练的价值，以及4）小型语言模型（SLMs）与大型语言模型（LLMs）在特定任务中的持续相关性。本研究使用不列颠哥伦比亚癌症登记处（BCCR）的电子病理报告，评估了三种具有不同难度和数据量的分类场景。模型包括各种SLMs和一个LLM。SLMs在零样本和微调两种情况下进行评估；而LLM仅在零样本情况下评估。微调在所有场景中显著提高了SLMs的性能，其性能优于零样本结果。零样本LLM在零样本情况下表现优于零样本SLMs，但在大多数情况下仍然被微调的SLMs所超越。领域相邻的SLMs在微调后通常优于通用的SLMs，尤其是在较难的任务上表现更好。进一步的领域特定预训练在较简单的任务上仅带来小幅收益，在复杂的、数据稀缺的任务上则取得了显著的改进。研究结果突显了在专业化领域中对SLMs进行微调的关键作用，使其在针对性的分类任务中能够超越LLMs的零样本表现。领域相邻或领域特定数据的预训练提供了进一步的优势，尤其是在复杂的问题或有限的微调数据条件下。虽然LLMs具有强大的零样本能力，但它们在这特定任务中的表现无法与适当地微调的SLMs相匹配。在LLM的时代，SLMs依然具有相关性和有效性，提供了一种与LLMs相比可能更为优异的性能与资源权衡。 

---
# TT-LoRA MoE: Unifying Parameter-Efficient Fine-Tuning and Sparse Mixture-of-Experts 

**Title (ZH)**: TT-LoRA MoE: 统一参数高效微调和稀疏混合专家模型 

**Authors**: Pradip Kunwar, Minh N. Vu, Maanak Gupta, Mahmoud Abdelsalam, Manish Bhattarai  

**Link**: [PDF](https://arxiv.org/pdf/2504.21190)  

**Abstract**: We propose Tensor-Trained Low-Rank Adaptation Mixture of Experts (TT-LoRA MoE), a novel computational framework integrating Parameter-Efficient Fine-Tuning (PEFT) with sparse MoE routing to address scalability challenges in large model deployments. Unlike traditional MoE approaches, which face substantial computational overhead as expert counts grow, TT-LoRA MoE decomposes training into two distinct, optimized stages. First, we independently train lightweight, tensorized low-rank adapters (TT-LoRA experts), each specialized for specific tasks. Subsequently, these expert adapters remain frozen, eliminating inter-task interference and catastrophic forgetting in multi-task setting. A sparse MoE router, trained separately, dynamically leverages base model representations to select exactly one specialized adapter per input at inference time, automating expert selection without explicit task specification. Comprehensive experiments confirm our architecture retains the memory efficiency of low-rank adapters, seamlessly scales to large expert pools, and achieves robust task-level optimization. This structured decoupling significantly enhances computational efficiency and flexibility: uses only 2% of LoRA, 0.3% of Adapters and 0.03% of AdapterFusion parameters and outperforms AdapterFusion by 4 value in multi-tasking, enabling practical and scalable multi-task inference deployments. 

**Abstract (ZH)**: 张量训练低秩适应专家混合模型（TT-LoRA MoE）：一种结合参数高效微调与稀疏专家路由的新型计算框架 

---
# On the Potential of Large Language Models to Solve Semantics-Aware Process Mining Tasks 

**Title (ZH)**: 大型语言模型在解决语义感知的过程挖掘任务方面的潜力 

**Authors**: Adrian Rebmann, Fabian David Schmidt, Goran Glavaš, Han van der Aa  

**Link**: [PDF](https://arxiv.org/pdf/2504.21074)  

**Abstract**: Large language models (LLMs) have shown to be valuable tools for tackling process mining tasks. Existing studies report on their capability to support various data-driven process analyses and even, to some extent, that they are able to reason about how processes work. This reasoning ability suggests that there is potential for LLMs to tackle semantics-aware process mining tasks, which are tasks that rely on an understanding of the meaning of activities and their relationships. Examples of these include process discovery, where the meaning of activities can indicate their dependency, whereas in anomaly detection the meaning can be used to recognize process behavior that is abnormal. In this paper, we systematically explore the capabilities of LLMs for such tasks. Unlike prior work, which largely evaluates LLMs in their default state, we investigate their utility through both in-context learning and supervised fine-tuning. Concretely, we define five process mining tasks requiring semantic understanding and provide extensive benchmarking datasets for evaluation. Our experiments reveal that while LLMs struggle with challenging process mining tasks when used out of the box or with minimal in-context examples, they achieve strong performance when fine-tuned for these tasks across a broad range of process types and industries. 

**Abstract (ZH)**: 大型语言模型（LLMs）在处理过程挖掘任务中的应用研究表明，它们能够支持各种数据驱动的过程分析，并在一定程度上能够推理过程的工作机制。这种推理能力表明，大型语言模型有可能承担起具备语义意识的过程挖掘任务，即依赖于活动意义及其关系理解的任务。这些任务包括过程发现，其中活动的意义可能表明它们之间的依赖性，而在异常检测中，这种意义可用于识别异常的过程行为。在本文中，我们系统地探讨了大型语言模型在这些任务中的能力。与先前主要评估模型在默认状态下能力的研究不同，我们通过上下文学习和监督微调来探究它们的应用价值。具体来说，我们定义了五个需要语义理解的过程挖掘任务，并提供了广泛基准数据集进行评估。我们的实验证明，虽然大型语言模型在空白或仅提供少量上下文示例的情况下处理具有挑战性的过程挖掘任务时表现不佳，但在针对这些任务进行广泛类型和行业的微调后，它们能够实现强大的性能。 

---
# NeuRel-Attack: Neuron Relearning for Safety Disalignment in Large Language Models 

**Title (ZH)**: NeuRel-攻击：大规模语言模型中安全性偏差的神经元重学习 

**Authors**: Yi Zhou, Wenpeng Xing, Dezhang Kong, Changting Lin, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.21053)  

**Abstract**: Safety alignment in large language models (LLMs) is achieved through fine-tuning mechanisms that regulate neuron activations to suppress harmful content. In this work, we propose a novel approach to induce disalignment by identifying and modifying the neurons responsible for safety constraints. Our method consists of three key steps: Neuron Activation Analysis, where we examine activation patterns in response to harmful and harmless prompts to detect neurons that are critical for distinguishing between harmful and harmless inputs; Similarity-Based Neuron Identification, which systematically locates the neurons responsible for safe alignment; and Neuron Relearning for Safety Removal, where we fine-tune these selected neurons to restore the model's ability to generate previously restricted responses. Experimental results demonstrate that our method effectively removes safety constraints with minimal fine-tuning, highlighting a critical vulnerability in current alignment techniques. Our findings underscore the need for robust defenses against adversarial fine-tuning attacks on LLMs. 

**Abstract (ZH)**: 大型语言模型的安全对齐通过调优机制调节神经元激活以抑制有害内容实现。在本工作中，我们提出了一种新的方法来诱导不对齐，通过识别和修改负责安全约束的神经元。该方法包括三个关键步骤：神经元激活分析，其中我们检查有害和无害提示的激活模式以检测对于区分有害和无害输入至关重要的神经元；基于相似性的神经元识别，系统地定位负责安全对齐的神经元；神经元重新学习以去除安全性约束，其中我们调优这些选定的神经元以恢复模型生成之前受限响应的能力。实验结果表明，我们的方法能够通过最小的调优有效去除安全性约束，突显了当前对齐技术中的一个关键漏洞。我们的发现强调了抵御针对大型语言模型的对抗调优攻击的稳健防御措施的需求。 

---
# Model Connectomes: A Generational Approach to Data-Efficient Language Models 

**Title (ZH)**: 代际模型连接组：一种数据高效的语言模型方法 

**Authors**: Klemen Kotar, Greta Tuckute  

**Link**: [PDF](https://arxiv.org/pdf/2504.21047)  

**Abstract**: Biological neural networks are shaped both by evolution across generations and by individual learning within an organism's lifetime, whereas standard artificial neural networks undergo a single, large training procedure without inherited constraints. In this preliminary work, we propose a framework that incorporates this crucial generational dimension - an "outer loop" of evolution that shapes the "inner loop" of learning - so that artificial networks better mirror the effects of evolution and individual learning in biological organisms. Focusing on language, we train a model that inherits a "model connectome" from the outer evolution loop before exposing it to a developmental-scale corpus of 100M tokens. Compared with two closely matched control models, we show that the connectome model performs better or on par on natural language processing tasks as well as alignment to human behavior and brain data. These findings suggest that a model connectome serves as an efficient prior for learning in low-data regimes - narrowing the gap between single-generation artificial models and biologically evolved neural networks. 

**Abstract (ZH)**: 生物神经网络由世代间的进化和个体生命期内的学习共同塑造，而标准人工神经网络则通过一次性大规模训练过程进行训练，不包含继承性的约束。在本初步研究中，我们提出了一种框架，该框架将这一关键的世代维度纳入其中——外部进化的“外环”塑造内部学习的“内环”——从而使人工网络更好地模拟生物体中进化和个体学习的影响。我们重点关注语言方面，训练了一个从外环进化过程继承“模型连接组”的模型，并将其暴露给一个包含1亿个标记的发育规模语料库。与两个匹配的控制模型相比，我们表明，具有连接组的模型在自然语言处理任务以及与人类行为和脑数据的对齐方面表现更好或相当。这些发现表明，模型连接组在数据量有限的情况下充当高效先验，缩小了单代人工模型与生物进化神经网络之间的差距。 

---
# Leveraging LLM to Strengthen ML-Based Cross-Site Scripting Detection 

**Title (ZH)**: 利用大规模语言模型强化基于机器学习的跨站点脚本检测 

**Authors**: Dennis Miczek, Divyesh Gabbireddy, Suman Saha  

**Link**: [PDF](https://arxiv.org/pdf/2504.21045)  

**Abstract**: According to the Open Web Application Security Project (OWASP), Cross-Site Scripting (XSS) is a critical security vulnerability. Despite decades of research, XSS remains among the top 10 security vulnerabilities. Researchers have proposed various techniques to protect systems from XSS attacks, with machine learning (ML) being one of the most widely used methods. An ML model is trained on a dataset to identify potential XSS threats, making its effectiveness highly dependent on the size and diversity of the training data. A variation of XSS is obfuscated XSS, where attackers apply obfuscation techniques to alter the code's structure, making it challenging for security systems to detect its malicious intent. Our study's random forest model was trained on traditional (non-obfuscated) XSS data achieved 99.8% accuracy. However, when tested against obfuscated XSS samples, accuracy dropped to 81.9%, underscoring the importance of training ML models with obfuscated data to improve their effectiveness in detecting XSS attacks. A significant challenge is to generate highly complex obfuscated code despite the availability of several public tools. These tools can only produce obfuscation up to certain levels of complexity.
In our proposed system, we fine-tune a Large Language Model (LLM) to generate complex obfuscated XSS payloads automatically. By transforming original XSS samples into diverse obfuscated variants, we create challenging training data for ML model evaluation. Our approach achieved a 99.5% accuracy rate with the obfuscated dataset. We also found that the obfuscated samples generated by the LLMs were 28.1% more complex than those created by other tools, significantly improving the model's ability to handle advanced XSS attacks and making it more effective for real-world application security. 

**Abstract (ZH)**: 根据Open Web Application Security Project (OWASP) 的定义，跨站脚本攻击（XSS）是一种关键的安全漏洞。尽管经过了几十年的研究，XSS 仍然是最常出现的十大安全漏洞之一。研究人员提出多种技术来保护系统免受XSS攻击，其中机器学习（ML）是最常用的方法之一。一个ML模型通过训练数据集来识别潜在的XSS威胁，其效果高度依赖于训练数据集的大小和多样性。XSS的一种变体是混淆XSS，攻击者通过应用混淆技术改变代码结构，从而给安全系统带来检测其恶意意图的挑战。我们的研究中，针对传统（未混淆）XSS数据训练的随机森林模型准确率达到99.8%。然而，在针对混淆XSS样本进行测试时，准确率下降到81.9%，强调了使用混解决策树模型训练数据的重要性，以提高其检测XSS攻击的效果。生成高度复杂的混淆代码是一个重大挑战，尽管有一些公开的工具可供使用，但这些工具只能生成一定程度复杂性的混淆。在我们提出的研究系统中，我们微调了一个大型语言模型（LLM）来自动生成复杂的混淆XSS载荷。通过将原始XSS样本转换为多种多样的混淆变体，我们为ML模型评估创造了具有挑战性的训练数据。我们的方法在混淆数据集上的准确率达到99.5%。此外，我们发现，由LLM生成的混淆样本比其他工具生成的更为复杂，复杂度高出28.1%，显著提高了模型处理高级XSS攻击的能力，并使其更适用于真实世界的应用安全。 

---
# CodeBC: A More Secure Large Language Model for Smart Contract Code Generation in Blockchain 

**Title (ZH)**: CodeBC：面向区块链智能合约代码生成的更安全大型语言模型 

**Authors**: Lingxiang wang, Hainan Zhang, Qinnan Zhang, Ziwei Wang, Hongwei Zheng, Jin Dong, Zhiming Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.21043)  

**Abstract**: Large language models (LLMs) excel at generating code from natural language instructions, yet they often lack an understanding of security vulnerabilities. This limitation makes it difficult for LLMs to avoid security risks in generated code, particularly in high-security programming tasks such as smart contract development for blockchain. Researchers have attempted to enhance the vulnerability awareness of these models by training them to differentiate between vulnerable and fixed code snippets. However, this approach relies heavily on manually labeled vulnerability data, which is only available for popular languages like Python and C++. For low-resource languages like Solidity, used in smart contracts, large-scale annotated datasets are scarce and difficult to obtain. To address this challenge, we introduce CodeBC, a code generation model specifically designed for generating secure smart contracts in blockchain. CodeBC employs a three-stage fine-tuning approach based on CodeLlama, distinguishing itself from previous methods by not relying on pairwise vulnerability location annotations. Instead, it leverages vulnerability and security tags to teach the model the differences between vulnerable and secure code. During the inference phase, the model leverages security tags to generate secure and robust code. Experimental results demonstrate that CodeBC outperforms baseline models in terms of BLEU, CodeBLEU, and compilation pass rates, while significantly reducing vulnerability rates. These findings validate the effectiveness and cost-efficiency of our three-stage fine-tuning strategy, making CodeBC a promising solution for generating secure smart contract code. 

**Abstract (ZH)**: 一种专门用于生成区块链安全智能合约的代码生成模型：CodeBC 

---
# Llama-3.1-FoundationAI-SecurityLLM-Base-8B Technical Report 

**Title (ZH)**: Llama-3.1-基础AI安全大语言模型-8B技术报告 

**Authors**: Paul Kassianik, Baturay Saglam, Alexander Chen, Blaine Nelson, Anu Vellore, Massimo Aufiero, Fraser Burch, Dhruv Kedia, Avi Zohary, Sajana Weerawardhena, Aman Priyanshu, Adam Swanda, Amy Chang, Hyrum Anderson, Kojin Oshiba, Omar Santos, Yaron Singer, Amin Karbasi  

**Link**: [PDF](https://arxiv.org/pdf/2504.21039)  

**Abstract**: As transformer-based large language models (LLMs) increasingly permeate society, they have revolutionized domains such as software engineering, creative writing, and digital arts. However, their adoption in cybersecurity remains limited due to challenges like scarcity of specialized training data and complexity of representing cybersecurity-specific knowledge. To address these gaps, we present Foundation-Sec-8B, a cybersecurity-focused LLM built on the Llama 3.1 architecture and enhanced through continued pretraining on a carefully curated cybersecurity corpus. We evaluate Foundation-Sec-8B across both established and new cybersecurity benchmarks, showing that it matches Llama 3.1-70B and GPT-4o-mini in certain cybersecurity-specific tasks. By releasing our model to the public, we aim to accelerate progress and adoption of AI-driven tools in both public and private cybersecurity contexts. 

**Abstract (ZH)**: 基于transformer的大规模语言模型在网络安全领域的聚焦研究：Foundation-Sec-8B的构建与评估 

---
# Prefill-Based Jailbreak: A Novel Approach of Bypassing LLM Safety Boundary 

**Title (ZH)**: 基于预填的脱管攻击：突破LLM安全边界的新方法 

**Authors**: Yakai Li, Jiekang Hu, Weiduan Sang, Luping Ma, Jing Xie, Weijuan Zhang, Aimin Yu, Shijie Zhao, Qingjia Huang, Qihang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.21038)  

**Abstract**: Large Language Models (LLMs) are designed to generate helpful and safe content. However, adversarial attacks, commonly referred to as jailbreak, can bypass their safety protocols, prompting LLMs to generate harmful content or reveal sensitive data. Consequently, investigating jailbreak methodologies is crucial for exposing systemic vulnerabilities within LLMs, ultimately guiding the continuous implementation of security enhancements by developers. In this paper, we introduce a novel jailbreak attack method that leverages the prefilling feature of LLMs, a feature designed to enhance model output constraints. Unlike traditional jailbreak methods, the proposed attack circumvents LLMs' safety mechanisms by directly manipulating the probability distribution of subsequent tokens, thereby exerting control over the model's output. We propose two attack variants: Static Prefilling (SP), which employs a universal prefill text, and Optimized Prefilling (OP), which iteratively optimizes the prefill text to maximize the attack success rate. Experiments on six state-of-the-art LLMs using the AdvBench benchmark validate the effectiveness of our method and demonstrate its capability to substantially enhance attack success rates when combined with existing jailbreak approaches. The OP method achieved attack success rates of up to 99.82% on certain models, significantly outperforming baseline methods. This work introduces a new jailbreak attack method in LLMs, emphasizing the need for robust content validation mechanisms to mitigate the adversarial exploitation of prefilling features. All code and data used in this paper are publicly available. 

**Abstract (ZH)**: 大型语言模型（LLMs）设计用于生成有益和安全的内容。然而，通常称为“出狱”的对抗攻击可以使LLMs的安全协议失效，导致其生成有害内容或泄露敏感数据。因此，研究出狱方法对于揭示LLMs中的系统性漏洞至关重要，最终指导开发人员持续实施安全增强措施。在本文中，我们介绍了一种利用LLMs前填功能的新颖出狱攻击方法，前填功能旨在增强模型输出约束。与传统的出狱方法不同，所提出的攻击通过直接操控后续令牌的概率分布来绕过LLMs的安全机制，从而控制模型的输出。我们提出了两种攻击变体：静态前填（SP），使用通用前填文本，以及优化前填（OP），通过迭代优化前填文本以最大化攻击成功率。使用AdvBench基准在六个最先进的LLM上进行的实验验证了我们方法的有效性，并展示了其在结合现有出狱方法时显著提高攻击成功率的能力。OP方法在某些模型上的攻击成功率高达99.82%，明显优于基线方法。本文引入了LLMs的新出狱攻击方法，强调了需要强大的内容验证机制以减轻前填功能的对抗性利用。本文中使用的所有代码和数据均公开可用。 

---
# Can Differentially Private Fine-tuning LLMs Protect Against Privacy Attacks? 

**Title (ZH)**: 差分隐私 Fine-tuning 大型语言模型能否保护against隐私攻击？ 

**Authors**: Hao Du, Shang Liu, Yang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2504.21036)  

**Abstract**: Fine-tuning large language models (LLMs) has become an essential strategy for adapting them to specialized tasks; however, this process introduces significant privacy challenges, as sensitive training data may be inadvertently memorized and exposed. Although differential privacy (DP) offers strong theoretical guarantees against such leakage, its empirical privacy effectiveness on LLMs remains unclear, especially under different fine-tuning methods. In this paper, we systematically investigate the impact of DP across fine-tuning methods and privacy budgets, using both data extraction and membership inference attacks to assess empirical privacy risks. Our main findings are as follows: (1) Differential privacy reduces model utility, but its impact varies significantly across different fine-tuning methods. (2) Without DP, the privacy risks of models fine-tuned with different approaches differ considerably. (3) When DP is applied, even a relatively high privacy budget can substantially lower privacy risk. (4) The privacy-utility trade-off under DP training differs greatly among fine-tuning methods, with some methods being unsuitable for DP due to severe utility degradation. Our results provide practical guidance for privacy-conscious deployment of LLMs and pave the way for future research on optimizing the privacy-utility trade-off in fine-tuning methodologies. 

**Abstract (ZH)**: 大规模语言模型（LLMs）微调已成为使其适应专门任务的一种关键策略；然而，这一过程引入了重大的隐私挑战，因为敏感的训练数据可能会无意中被记忆并泄露。尽管差异化隐私（DP）提供了强理论保护以防止这种泄露，但其在LLMs上的实际隐私有效性仍不清楚，特别是在不同的微调方法下。在本文中，我们系统地研究了DP在不同微调方法和隐私预算下的影响，通过数据提取和成员推断攻击来评估实际的隐私风险。主要发现如下：（1）差异化隐私降低了模型的实用性，但其影响在不同的微调方法间差异显著。（2）在没有DP的情况下，使用不同方法微调的模型的隐私风险差异很大。（3）当应用DP时，即使是在相对较高的隐私预算下，也能显著降低隐私风险。（4）在DP训练下的隐私-实用性权衡在不同微调方法间存在巨大差异，某些方法因严重实用性下降而不适合DP。我们的结果为隐私意识较强的LLMs部署提供了实际指导，并为未来研究优化微调方法下的隐私-实用性权衡奠定了基础。 

---
# Selecting the Right LLM for eGov Explanations 

**Title (ZH)**: 选择合适的大型语言模型进行电子政务解释 

**Authors**: Lior Limonad, Fabiana Fournier, Hadar Mulian, George Manias, Spiros Borotis, Danai Kyrkou  

**Link**: [PDF](https://arxiv.org/pdf/2504.21032)  

**Abstract**: The perceived quality of the explanations accompanying e-government services is key to gaining trust in these institutions, consequently amplifying further usage of these services. Recent advances in generative AI, and concretely in Large Language Models (LLMs) allow the automation of such content articulations, eliciting explanations' interpretability and fidelity, and more generally, adapting content to various audiences. However, selecting the right LLM type for this has become a non-trivial task for e-government service providers. In this work, we adapted a previously developed scale to assist with this selection, providing a systematic approach for the comparative analysis of the perceived quality of explanations generated by various LLMs. We further demonstrated its applicability through the tax-return process, using it as an exemplar use case that could benefit from employing an LLM to generate explanations about tax refund decisions. This was attained through a user study with 128 survey respondents who were asked to rate different versions of LLM-generated explanations about tax refund decisions, providing a methodological basis for selecting the most appropriate LLM. Recognizing the practical challenges of conducting such a survey, we also began exploring the automation of this process by attempting to replicate human feedback using a selection of cutting-edge predictive techniques. 

**Abstract (ZH)**: 电子政务服务中伴随说明的感知质量是建立这些机构信任的关键，从而进一步放大这些服务的使用。生成式人工智能的最新进展，特别是大型语言模型（LLMs）的出现，使这类内容的自动化表达成为可能，引发了解释的可解释性和真实性，并且更广泛地适应不同的受众。然而，选择合适的LLM类型来实现这一点已成为电子政务服务提供商的一项非 trivial 任务。在本文中，我们调整了一种先前开发的量表来帮助进行这种选择，提供了一种系统的方法来进行各种LLM生成的解释感知质量的比较分析。我们还通过税务申报过程进一步展示了其应用性，将其作为可以受益于使用LLM生成税退款决定解释的示例用例。这通过一项包含128名调查响应者的用户研究来实现，要求他们评估不同版本的LLM生成的税退款决定解释，为选择最合适的LLM提供了方法论基础。鉴于进行此类调查的实际挑战，我们还开始探索通过使用一系列先进的预测技术来自动复制人类反馈的过程。 

---
# Semantic-Aware Contrastive Fine-Tuning: Boosting Multimodal Malware Classification with Discriminative Embeddings 

**Title (ZH)**: 语义意识对比微调：通过 discriminative 嵌入提升多模态恶意软件分类 

**Authors**: Ivan Montoya Sanchez, Shaswata Mitra, Aritran Piplai, Sudip Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2504.21028)  

**Abstract**: The rapid evolution of malware variants requires robust classification methods to enhance cybersecurity. While Large Language Models (LLMs) offer potential for generating malware descriptions to aid family classification, their utility is limited by semantic embedding overlaps and misalignment with binary behavioral features. We propose a contrastive fine-tuning (CFT) method that refines LLM embeddings via targeted selection of hard negative samples based on cosine similarity, enabling LLMs to distinguish between closely related malware families. Our approach combines high-similarity negatives to enhance discriminative power and mid-tier negatives to increase embedding diversity, optimizing both precision and generalization. Evaluated on the CIC-AndMal-2020 and BODMAS datasets, our refined embeddings are integrated into a multimodal classifier within a Model-Agnostic Meta-Learning (MAML) framework on a few-shot setting. Experiments demonstrate significant improvements: our method achieves 63.15% classification accuracy with as few as 20 samples on CIC-AndMal-2020, outperforming baselines by 11--21 percentage points and surpassing prior negative sampling strategies. Ablation studies confirm the superiority of similarity-based selection over random sampling, with gains of 10-23%. Additionally, fine-tuned LLMs generate attribute-aware descriptions that generalize to unseen variants, bridging textual and binary feature gaps. This work advances malware classification by enabling nuanced semantic distinctions and provides a scalable framework for adapting LLMs to cybersecurity challenges. 

**Abstract (ZH)**: 快速演变的恶意软件变种要求有 robust 的分类方法以增强网络安全。虽然大型语言模型（LLMs）有可能生成恶意软件描述以辅助家族分类，但其实用性受限于语义嵌入的重叠和与二进制行为特征的不对齐。我们提出了一种对比微调（CFT）方法，通过基于余弦相似度选择针对性的硬负样本来细化LLM嵌入，使LLM能够区分密切相关恶意软件家族。我们的方法结合高相似度负样本以增强判别力，并结合中等层级负样本以增加嵌入多样性，从而同时优化精度和泛化能力。我们在CIC-AndMal-2020和BODMAS数据集上进行了评估，将优化后的嵌入整合到一个模型无偏元学习（MAML）框架中的多模态分类器中，在少量样本设置下进行。实验结果显示了显著的改进：我们的方法在CIC-AndMal-2020数据集上仅使用20个样本就能达到63.15%的分类准确率，优于基线11-21个百分点，并超过了先前的负样本策略。消融研究证实，基于相似度的选择优于随机采样，提高了10-23%。此外，微调后的LLM生成的属性感知描述能够泛化到未见过的变种，弥补了文本和二进制特征之间的差距。这项工作通过使恶意软件分类能够进行细微的语义区分，推进了恶意软件分类，并提供了一个可扩展框架，使LLM能够应对网络安全挑战。 

---
# UrbanPlanBench: A Comprehensive Urban Planning Benchmark for Evaluating Large Language Models 

**Title (ZH)**: UrbanPlanBench: 一种全面的城市规划基准，用于评估大规模语言模型 

**Authors**: Yu Zheng, Longyi Liu, Yuming Lin, Jie Feng, Guozhen Zhang, Depeng Jin, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.21027)  

**Abstract**: The advent of Large Language Models (LLMs) holds promise for revolutionizing various fields traditionally dominated by human expertise. Urban planning, a professional discipline that fundamentally shapes our daily surroundings, is one such field heavily relying on multifaceted domain knowledge and experience of human experts. The extent to which LLMs can assist human practitioners in urban planning remains largely unexplored. In this paper, we introduce a comprehensive benchmark, UrbanPlanBench, tailored to evaluate the efficacy of LLMs in urban planning, which encompasses fundamental principles, professional knowledge, and management and regulations, aligning closely with the qualifications expected of human planners. Through extensive evaluation, we reveal a significant imbalance in the acquisition of planning knowledge among LLMs, with even the most proficient models falling short of meeting professional standards. For instance, we observe that 70% of LLMs achieve subpar performance in understanding planning regulations compared to other aspects. Besides the benchmark, we present the largest-ever supervised fine-tuning (SFT) dataset, UrbanPlanText, comprising over 30,000 instruction pairs sourced from urban planning exams and textbooks. Our findings demonstrate that fine-tuned models exhibit enhanced performance in memorization tests and comprehension of urban planning knowledge, while there exists significant room for improvement, particularly in tasks requiring domain-specific terminology and reasoning. By making our benchmark, dataset, and associated evaluation and fine-tuning toolsets publicly available at this https URL, we aim to catalyze the integration of LLMs into practical urban planning, fostering a symbiotic collaboration between human expertise and machine intelligence. 

**Abstract (ZH)**: 大型语言模型的兴起有望革命性地改变传统上由人类专家主导的各个领域。城市规划作为一项从根本上塑造我们日常生活环境的专业学科，是依赖多方面领域知识和专家经验的领域之一。关于大型语言模型在城市规划中的辅助效果仍然 largely unexplored。本文介绍了专为评估大型语言模型在城市规划中的效果而设计的综合基准——UrbanPlanBench，涵盖了基本原理、专业知识、管理和法规，与城市规划师所需资格紧密契合。通过广泛评估，我们揭示了大型语言模型在获取规划知识方面存在显著不平衡，即使是技术最成熟的模型也无法达到专业标准。例如，我们观察到70%的模型在理解规划法规方面表现不佳，与其它方面相比。除了基准外，我们还呈现了迄今为止最大的监督微调数据集——UrbanPlanText，包含超过30,000个来自城市规划考试和教材的指令对。我们的研究结果表明，微调模型在记忆测试和理解城市规划知识方面表现出增强的效果，但在需要特定领域术语和推理的任务中仍有很多改进空间。通过在此网址公开我们的基准、数据集以及相关评估和微调工具集，我们旨在促进大型语言模型在实际城市规划中的集成，推动人类专业知识与机器智能的共生合作。 

---
# Context-Enhanced Contrastive Search for Improved LLM Text Generation 

**Title (ZH)**: 上下文增强对比检索以改善LLM文本生成 

**Authors**: Jaydip Sen, Rohit Pandey, Hetvi Waghela  

**Link**: [PDF](https://arxiv.org/pdf/2504.21020)  

**Abstract**: Recently, Large Language Models (LLMs) have demonstrated remarkable advancements in Natural Language Processing (NLP). However, generating high-quality text that balances coherence, diversity, and relevance remains challenging. Traditional decoding methods, such as bean search and top-k sampling, often struggle with either repetitive or incoherent outputs, particularly in tasks that require long-form text generation. To address these limitations, the paper proposes a novel enhancement of the well-known Contrastive Search algorithm, Context-Enhanced Contrastive Search (CECS) with contextual calibration. The proposed scheme introduces several novelties including dynamic contextual importance weighting, multi-level Contrastive Search, and adaptive temperature control, to optimize the balance between fluency, creativity, and precision. The performance of CECS is evaluated using several standard metrics such as BLEU, ROUGE, and semantic similarity. Experimental results demonstrate significant improvements in both coherence and relevance of the generated texts by CECS outperforming the existing Contrastive Search techniques. The proposed algorithm has several potential applications in the real world including legal document drafting, customer service chatbots, and content marketing. 

**Abstract (ZH)**: 近期，大规模语言模型（LLMs）在自然语言处理（NLP）领域取得了显著进展。然而，生成高质量平衡连贯性、多样性和相关性的文本仍然是一个挑战。传统的解码方法，如贪心搜索和top-k采样，常常在长文本生成任务中产生重复或不连贯的输出。为了解决这些问题，论文提出了一种Contrastive Search算法的新型增强方法——基于上下文校准的Contrastive Search增强方法（CECS）。该方案引入了动态上下文重要性加权、多层次Contrastive Search和自适应温度控制等新颖特性，以优化流畅性、创造性和精确性的平衡。CECS的性能通过使用BLEU、ROUGE和语义相似性等标准指标进行评估。实验结果表明，CECS在生成文本的连贯性和相关性方面取得了显著改进，优于现有的Contrastive Search技术。所提出算法在实际应用中有多种潜在用途，包括法律文件起草、客户服务聊天机器人和内容营销等。 

---
# Kill two birds with one stone: generalized and robust AI-generated text detection via dynamic perturbations 

**Title (ZH)**: 一石二鸟：通过动态扰动实现通用且稳健的AI生成文本检测 

**Authors**: Yinghan Zhou, Juan Wen, Wanli Peng, Yiming Xue, Ziwei Zhang, Zhengxian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21019)  

**Abstract**: The growing popularity of large language models has raised concerns regarding the potential to misuse AI-generated text (AIGT). It becomes increasingly critical to establish an excellent AIGT detection method with high generalization and robustness. However, existing methods either focus on model generalization or concentrate on robustness. The unified mechanism, to simultaneously address the challenges of generalization and robustness, is less explored. In this paper, we argue that robustness can be view as a specific form of domain shift, and empirically reveal an intrinsic mechanism for model generalization of AIGT detection task. Then, we proposed a novel AIGT detection method (DP-Net) via dynamic perturbations introduced by a reinforcement learning with elaborated reward and action. Experimentally, extensive results show that the proposed DP-Net significantly outperforms some state-of-the-art AIGT detection methods for generalization capacity in three cross-domain scenarios. Meanwhile, the DP-Net achieves best robustness under two text adversarial attacks. The code is publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型 popularity 的增长引发了对 AI 生成文本 (AIGT) 滥用潜在风险的关注。建立一种兼具高泛化能力和鲁棒性的 AIGT 检测方法变得越来越关键。然而，现有方法要么专注于模型泛化，要么专注于鲁棒性。同时解决泛化和鲁棒性挑战的统一机制研究较少。本文认为鲁棒性可以被视为一种特定形式的领域转移，并通过强化学习引入动态扰动，采用有细化奖励和动作的方法，实证揭示了 AIGT 检测任务模型泛化的一种内在机制。然后，我们提出了一种新颖的 AIGT 检测方法（DP-Net）。实验结果显示，在三个跨领域场景中，所提出的 DP-Net 显著优于一些最新 AIGT 检测方法的泛化能力。同时，在两种文本对抗攻击下，DP-Net 达到了最佳鲁棒性。有关代码已公开，访问此URL：。 

---
# Analyzing Feedback Mechanisms in AI-Generated MCQs: Insights into Readability, Lexical Properties, and Levels of Challenge 

**Title (ZH)**: 分析AI生成的多项选择题中的反馈机制：可读性、词项特征及难度水平 insights 

**Authors**: Antoun Yaacoub, Zainab Assaghir, Lionel Prevost, Jérôme Da-Rugna  

**Link**: [PDF](https://arxiv.org/pdf/2504.21013)  

**Abstract**: Artificial Intelligence (AI)-generated feedback in educational settings has garnered considerable attention due to its potential to enhance learning outcomes. However, a comprehensive understanding of the linguistic characteristics of AI-generated feedback, including readability, lexical richness, and adaptability across varying challenge levels, remains limited. This study delves into the linguistic and structural attributes of feedback generated by Google's Gemini 1.5-flash text model for computer science multiple-choice questions (MCQs). A dataset of over 1,200 MCQs was analyzed, considering three difficulty levels (easy, medium, hard) and three feedback tones (supportive, neutral, challenging). Key linguistic metrics, such as length, readability scores (Flesch-Kincaid Grade Level), vocabulary richness, and lexical density, were computed and examined. A fine-tuned RoBERTa-based multi-task learning (MTL) model was trained to predict these linguistic properties, achieving a Mean Absolute Error (MAE) of 2.0 for readability and 0.03 for vocabulary richness. The findings reveal significant interaction effects between feedback tone and question difficulty, demonstrating the dynamic adaptation of AI-generated feedback within diverse educational contexts. These insights contribute to the development of more personalized and effective AI-driven feedback mechanisms, highlighting the potential for improved learning outcomes while underscoring the importance of ethical considerations in their design and deployment. 

**Abstract (ZH)**: 人工智能生成反馈在教育环境中的应用引起了广泛关注，因其有望提高学习效果。然而，有关人工智能生成反馈的语言特征，包括可读性、词汇丰富度和不同挑战水平下的适应性，的研究仍不够全面。本研究探讨了谷歌Gemini 1.5-flash文本模型生成的计算机科学多项选择题（MCQ）反馈的语义和结构特征。分析了超过1,200道MCQ，考虑了三种难度级别（简单、中等、困难）和三种反馈语气（支持性、中性、挑战性）。计算并分析了诸如长度、可读性评分（Flesch-Kincaid 年级水平）、词汇丰富度和词汇密度等关键语言指标。使用微调后的RoBERTa多任务学习（MTL）模型预测这些语言属性，可读性误差的均方绝对误差（MAE）为2.0，词汇丰富度误差为0.03。研究发现，反馈语气与问题难度之间的交互作用显著，表明人工智能生成的反馈在不同教育背景下能够动态适应。这些发现为开发更加个性化的、有效的基于人工智能的反馈机制提供了依据，强调了其设计和部署时需要考虑伦理问题，以提高学习效果。 

---
# Waking Up an AI: A Quantitative Framework for Prompt-Induced Phase Transition in Large Language Models 

**Title (ZH)**: 唤醒AI：大规模语言模型由提示引发的相变的定量框架 

**Authors**: Makoto Sato  

**Link**: [PDF](https://arxiv.org/pdf/2504.21012)  

**Abstract**: What underlies intuitive human thinking? One approach to this question is to compare the cognitive dynamics of humans and large language models (LLMs). However, such a comparison requires a method to quantitatively analyze AI cognitive behavior under controlled conditions. While anecdotal observations suggest that certain prompts can dramatically change LLM behavior, these observations have remained largely qualitative. Here, we propose a two-part framework to investigate this phenomenon: a Transition-Inducing Prompt (TIP) that triggers a rapid shift in LLM responsiveness, and a Transition Quantifying Prompt (TQP) that evaluates this change using a separate LLM. Through controlled experiments, we examined how LLMs react to prompts embedding two semantically distant concepts (e.g., mathematical aperiodicity and traditional crafts)--either fused together or presented separately--by changing their linguistic quality and affective tone. Whereas humans tend to experience heightened engagement when such concepts are meaningfully blended producing a novel concept--a form of conceptual fusion--current LLMs showed no significant difference in responsiveness between semantically fused and non-fused prompts. This suggests that LLMs may not yet replicate the conceptual integration processes seen in human intuition. Our method enables fine-grained, reproducible measurement of cognitive responsiveness, and may help illuminate key differences in how intuition and conceptual leaps emerge in artificial versus human minds. 

**Abstract (ZH)**: 直观人类思维的基础是什么？一种方法是将人类的认知动态与大语言模型（LLMs）进行比较。但是，这种比较需要一种在受控条件下定量分析AI认知行为的方法。虽然有些示例观察表明某些提示可以显著改变LLM的行为，但这些观察主要还是定性的。在此，我们提出了一种两部分框架来研究这一现象：一种触发LLM响应快速转变的转换诱导提示（TIP），以及一种用于通过另一个LLM评估这种变化的转换量化提示（TQP）。通过受控实验，我们考察了LLMs在嵌入两个语义上相距甚远的概念（例如，数学无周期性和传统工艺）时的反应——这些概念要么融合在一起，要么分开发表——是如何通过改变语言质量和情感色彩来影响其行为的。与人类往往在语义上融合有意义的概念时表现出更高的参与度——这种概念融合具有新颖性——不同，当前的LLMs在语义融合和非融合的提示之间没有显示出显著的响应差异。这表明LLMs可能尚未复制人类直观思维中观察到的概念整合过程。我们的方法可以实现认知响应的精细、可重复测量，并可能有助于阐明人工智能与人类思维中概念飞跃出现的关键差异。 

---
