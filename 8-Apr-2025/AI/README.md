# SmolVLM: Redefining small and efficient multimodal models 

**Title (ZH)**: SmolVLM: 重定义小型高效的多模态模型 

**Authors**: Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Vaibhav Srivastav, Joshua Lochner, Hugo Larcher, Mathieu Morlon, Lewis Tunstall, Leandro von Werra, Thomas Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2504.05299)  

**Abstract**: Large Vision-Language Models (VLMs) deliver exceptional performance but require significant computational resources, limiting their deployment on mobile and edge devices. Smaller VLMs typically mirror design choices of larger models, such as extensive image tokenization, leading to inefficient GPU memory usage and constrained practicality for on-device applications.
We introduce SmolVLM, a series of compact multimodal models specifically engineered for resource-efficient inference. We systematically explore architectural configurations, tokenization strategies, and data curation optimized for low computational overhead. Through this, we identify key design choices that yield substantial performance gains on image and video tasks with minimal memory footprints.
Our smallest model, SmolVLM-256M, uses less than 1GB GPU memory during inference and outperforms the 300-times larger Idefics-80B model, despite an 18-month development gap. Our largest model, at 2.2B parameters, rivals state-of-the-art VLMs consuming twice the GPU memory. SmolVLM models extend beyond static images, demonstrating robust video comprehension capabilities.
Our results emphasize that strategic architectural optimizations, aggressive yet efficient tokenization, and carefully curated training data significantly enhance multimodal performance, facilitating practical, energy-efficient deployments at significantly smaller scales. 

**Abstract (ZH)**: 小型化多模态模型（SmolVLM）：资源高效推理的设计选择与性能优化 

---
# The challenge of uncertainty quantification of large language models in medicine 

**Title (ZH)**: 大型语言模型在医学中的不确定性量化挑战 

**Authors**: Zahra Atf, Seyed Amir Ahmad Safavi-Naini, Peter R. Lewis, Aref Mahjoubfar, Nariman Naderi, Thomas R. Savage, Ali Soroush  

**Link**: [PDF](https://arxiv.org/pdf/2504.05278)  

**Abstract**: This study investigates uncertainty quantification in large language models (LLMs) for medical applications, emphasizing both technical innovations and philosophical implications. As LLMs become integral to clinical decision-making, accurately communicating uncertainty is crucial for ensuring reliable, safe, and ethical AI-assisted healthcare. Our research frames uncertainty not as a barrier but as an essential part of knowledge that invites a dynamic and reflective approach to AI design. By integrating advanced probabilistic methods such as Bayesian inference, deep ensembles, and Monte Carlo dropout with linguistic analysis that computes predictive and semantic entropy, we propose a comprehensive framework that manages both epistemic and aleatoric uncertainties. The framework incorporates surrogate modeling to address limitations of proprietary APIs, multi-source data integration for better context, and dynamic calibration via continual and meta-learning. Explainability is embedded through uncertainty maps and confidence metrics to support user trust and clinical interpretability. Our approach supports transparent and ethical decision-making aligned with Responsible and Reflective AI principles. Philosophically, we advocate accepting controlled ambiguity instead of striving for absolute predictability, recognizing the inherent provisionality of medical knowledge. 

**Abstract (ZH)**: 本研究探讨了在医疗应用中大型语言模型（LLMs）的不确定性量化，强调了技术和哲学层面的双重意义。随着LLMs在临床决策中变得不可或缺，准确传达不确定性对于确保可靠、安全和伦理的人工智能辅助医疗至关重要。我们的研究将不确定性视为知识不可或缺的一部分，旨在通过动态和反思性的方法来促进AI设计。通过结合先进的概率方法（如贝叶斯推断、深度集成和蒙特卡洛丢弃）与计算预测和语义熵的语言分析，我们提出了一种全面的框架，以管理可知论和偶然论不确定性。该框架整合了替代建模以应对专有API的局限性，以及通过多元数据融合和持续元学习的动态校准。通过不确定性图和置信度指标嵌入可解释性，以支持用户的信任和临床解释性。我们的方法支持与负责任和反思性AI原则相一致的透明和伦理决策。从哲学上讲，我们主张接受可控的模糊性而非追求绝对的可预测性，认识到医学知识的内在临时性。 

---
# How to evaluate control measures for LLM agents? A trajectory from today to superintelligence 

**Title (ZH)**: 如何评估大型语言模型代理的控制措施？从今天到超智能的路径 

**Authors**: Tomek Korbak, Mikita Balesni, Buck Shlegeris, Geoffrey Irving  

**Link**: [PDF](https://arxiv.org/pdf/2504.05259)  

**Abstract**: As LLM agents grow more capable of causing harm autonomously, AI developers will rely on increasingly sophisticated control measures to prevent possibly misaligned agents from causing harm. AI developers could demonstrate that their control measures are sufficient by running control evaluations: testing exercises in which a red team produces agents that try to subvert control measures. To ensure control evaluations accurately capture misalignment risks, the affordances granted to this red team should be adapted to the capability profiles of the agents to be deployed under control measures.
In this paper we propose a systematic framework for adapting affordances of red teams to advancing AI capabilities. Rather than assuming that agents will always execute the best attack strategies known to humans, we demonstrate how knowledge of an agents's actual capability profile can inform proportional control evaluations, resulting in more practical and cost-effective control measures. We illustrate our framework by considering a sequence of five fictional models (M1-M5) with progressively advanced capabilities, defining five distinct AI control levels (ACLs). For each ACL, we provide example rules for control evaluation, control measures, and safety cases that could be appropriate. Finally, we show why constructing a compelling AI control safety case for superintelligent LLM agents will require research breakthroughs, highlighting that we might eventually need alternative approaches to mitigating misalignment risk. 

**Abstract (ZH)**: 随着大语言模型代理自主造成损害的能力不断增强，AI开发者将依赖日益复杂的控制措施来防止可能未对齐的代理造成损害。AI开发者可以通过运行控制评估来证明其控制措施的充分性：在红队生成试图规避控制措施的代理的测试演习中进行测试。为了确保控制评估能够准确捕捉到不对齐风险，授予红队的便利性应根据部署在控制措施下的代理的能力特征进行调整。

本文提出了一种系统框架，用于根据不断进步的AI能力调整红队的便利性。我们不假设代理总能执行人类已知的最佳攻击策略，而是展示了代理的实际能力特征如何指导恰当的控制评估，从而导致更实用和成本效益更高的控制措施。我们通过考虑五个逐步进化的虚构模型（M1-M5）和定义五个不同的AI控制级别（ACLs）来阐述我们的框架。对于每个控制级别，我们提供示例规则、控制措施和安全案例。最后，我们说明了为什么为超级智能的大语言模型构建有说服力的AI控制安全案例将需要研究突破，强调我们最终可能需要新的方法来减轻不对齐风险。 

---
# Mapping biodiversity at very-high resolution in Europe 

**Title (ZH)**: 欧洲极高分辨率生物多样性mapping 

**Authors**: César Leblanc, Lukas Picek, Benjamin Deneu, Pierre Bonnet, Maximilien Servajean, Rémi Palard, Alexis Joly  

**Link**: [PDF](https://arxiv.org/pdf/2504.05231)  

**Abstract**: This paper describes a cascading multimodal pipeline for high-resolution biodiversity mapping across Europe, integrating species distribution modeling, biodiversity indicators, and habitat classification. The proposed pipeline first predicts species compositions using a deep-SDM, a multimodal model trained on remote sensing, climate time series, and species occurrence data at 50x50m resolution. These predictions are then used to generate biodiversity indicator maps and classify habitats with Pl@ntBERT, a transformer-based LLM designed for species-to-habitat mapping. With this approach, continental-scale species distribution maps, biodiversity indicator maps, and habitat maps are produced, providing fine-grained ecological insights. Unlike traditional methods, this framework enables joint modeling of interspecies dependencies, bias-aware training with heterogeneous presence-absence data, and large-scale inference from multi-source remote sensing inputs. 

**Abstract (ZH)**: 本文描述了一个级联多模态管道，用于欧洲高分辨率生物多样性mapping，整合了物种分布模型、生物多样性指标和栖地分类。该提出的管道首先使用深度SDM预测物种组成，该模型基于50x50m分辨率的遥感数据、气候时间序列和物种分布数据进行训练。这些预测结果随后用于生成生物多样性指标地图，并使用专为物种-栖地映射设计的基于Transformer的大规模语言模型Pl@ntBERT进行栖地分类。通过这种方法， continental尺度的物种分布地图、生物多样性指标地图和栖地地图得以生成，提供精细的生态洞察。与传统方法不同，该框架能够同时建模物种之间的依赖关系、使用异质的存活性数据进行校准偏差的训练，并从多源遥感输入中进行大规模推断。 

---
# FinGrAct: A Framework for FINe-GRrained Evaluation of ACTionability in Explainable Automatic Fact-Checking 

**Title (ZH)**: FinGrAct: 一种可解释自动事实核查中行动性细粒度评估框架 

**Authors**: Islam Eldifrawi, Shengrui Wang, Amine Trabelsi  

**Link**: [PDF](https://arxiv.org/pdf/2504.05229)  

**Abstract**: The field of explainable Automatic Fact-Checking (AFC) aims to enhance the transparency and trustworthiness of automated fact-verification systems by providing clear and comprehensible explanations. However, the effectiveness of these explanations depends on their actionability --their ability to empower users to make informed decisions and mitigate misinformation. Despite actionability being a critical property of high-quality explanations, no prior research has proposed a dedicated method to evaluate it. This paper introduces FinGrAct, a fine-grained evaluation framework that can access the web, and it is designed to assess actionability in AFC explanations through well-defined criteria and an evaluation dataset. FinGrAct surpasses state-of-the-art (SOTA) evaluators, achieving the highest Pearson and Kendall correlation with human judgments while demonstrating the lowest ego-centric bias, making it a more robust evaluation approach for actionability evaluation in AFC. 

**Abstract (ZH)**: 可解释自动事实核查（AFC）领域的研究旨在通过提供清晰可理解的解释来提高自动化事实验证系统的透明度和可信度。然而，这些解释的有效性取决于其可操作性——即其赋能用户做出知情决策和减轻虚假信息的能力。尽管可操作性是高质量解释的一个关键属性，但此前的研究尚未提出专门评估其可操作性的方法。本文引入了FinGrAct，这是一种细粒度的评估框架，能够访问网络，并通过明确的标准和评估数据集来评估AFC解释的可操作性。FinGrAct超越了现有最先进的（SOTA）评估器，实现了与人类判断最高的皮尔森和肯德尔相关性，同时表现出最低的自中心偏差，使其成为AFC中可操作性评估的更为稳健的方法。 

---
# Evaluating Knowledge Graph Based Retrieval Augmented Generation Methods under Knowledge Incompleteness 

**Title (ZH)**: 基于知识不完备性的知识图谱检索增强生成方法评价 

**Authors**: Dongzhuoran Zhou, Yuqicheng Zhu, Yuan He, Jiaoyan Chen, Evgeny Kharlamov, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2504.05163)  

**Abstract**: Knowledge Graph based Retrieval-Augmented Generation (KG-RAG) is a technique that enhances Large Language Model (LLM) inference in tasks like Question Answering (QA) by retrieving relevant information from knowledge graphs (KGs). However, real-world KGs are often incomplete, meaning that essential information for answering questions may be missing. Existing benchmarks do not adequately capture the impact of KG incompleteness on KG-RAG performance. In this paper, we systematically evaluate KG-RAG methods under incomplete KGs by removing triples using different methods and analyzing the resulting effects. We demonstrate that KG-RAG methods are sensitive to KG incompleteness, highlighting the need for more robust approaches in realistic settings. 

**Abstract (ZH)**: 基于知识图谱的检索增强生成（KG-RAG）：在知识图谱不完备的情况下评估方法对其性能的影响 

---
# VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks 

**Title (ZH)**: VAPO: 高效且可靠的高级推理任务强化学习方法 

**Authors**: YuYue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, Xiangpeng Wei, Gaohong Liu, Juncai Liu, Lingjun Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Ru Zhang, Xin Liu, Mingxuan Wang, Yonghui Wu, Lin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2504.05118)  

**Abstract**: We present VAPO, Value-based Augmented Proximal Policy Optimization framework for reasoning models., a novel framework tailored for reasoning models within the value-based paradigm. Benchmarked the AIME 2024 dataset, VAPO, built on the Qwen 32B pre-trained model, attains a state-of-the-art score of $\mathbf{60.4}$. In direct comparison under identical experimental settings, VAPO outperforms the previously reported results of DeepSeek-R1-Zero-Qwen-32B and DAPO by more than 10 points. The training process of VAPO stands out for its stability and efficiency. It reaches state-of-the-art performance within a mere 5,000 steps. Moreover, across multiple independent runs, no training crashes occur, underscoring its reliability. This research delves into long chain-of-thought (long-CoT) reasoning using a value-based reinforcement learning framework. We pinpoint three key challenges that plague value-based methods: value model bias, the presence of heterogeneous sequence lengths, and the sparsity of reward signals. Through systematic design, VAPO offers an integrated solution that effectively alleviates these challenges, enabling enhanced performance in long-CoT reasoning tasks. 

**Abstract (ZH)**: 基于价值的增强近端策略优化框架VAPO：一种面向推理模型的新框架 

---
# Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning 

**Title (ZH)**: 基于LLMs的算法发现：演化搜索结合强化学习 

**Authors**: Anja Surina, Amin Mansouri, Lars Quaedvlieg, Amal Seddas, Maryna Viazovska, Emmanuel Abbe, Caglar Gulcehre  

**Link**: [PDF](https://arxiv.org/pdf/2504.05108)  

**Abstract**: Discovering efficient algorithms for solving complex problems has been an outstanding challenge in mathematics and computer science, requiring substantial human expertise over the years. Recent advancements in evolutionary search with large language models (LLMs) have shown promise in accelerating the discovery of algorithms across various domains, particularly in mathematics and optimization. However, existing approaches treat the LLM as a static generator, missing the opportunity to update the model with the signal obtained from evolutionary exploration. In this work, we propose to augment LLM-based evolutionary search by continuously refining the search operator - the LLM - through reinforcement learning (RL) fine-tuning. Our method leverages evolutionary search as an exploration strategy to discover improved algorithms, while RL optimizes the LLM policy based on these discoveries. Our experiments on three combinatorial optimization tasks - bin packing, traveling salesman, and the flatpack problem - show that combining RL and evolutionary search improves discovery efficiency of improved algorithms, showcasing the potential of RL-enhanced evolutionary strategies to assist computer scientists and mathematicians for more efficient algorithm design. 

**Abstract (ZH)**: 利用强化学习增强进化搜索以发现高效算法 

---
# Debate Only When Necessary: Adaptive Multiagent Collaboration for Efficient LLM Reasoning 

**Title (ZH)**: 必要时才辩论：适应性多智能体合作以提高大型语言模型推理效率 

**Authors**: Sugyeong Eo, Hyeonseok Moon, Evelyn Hayoon Zi, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2504.05047)  

**Abstract**: Multiagent collaboration has emerged as a promising framework for enhancing the reasoning capabilities of large language models (LLMs). While this approach improves reasoning capability, it incurs substantial computational overhead due to iterative agent interactions. Furthermore, engaging in debates for queries that do not necessitate collaboration amplifies the risk of error generation. To address these challenges, we propose Debate Only When Necessary (DOWN), an adaptive multiagent debate framework that selectively activates the debate process based on the confidence score of the agent's initial response. For queries where debate is triggered, agents refine their outputs using responses from participating agents and their confidence scores. Experimental results demonstrate that this mechanism significantly improves efficiency while maintaining or even surpassing the performance of existing multiagent debate systems. We also find that confidence-guided debate mitigates error propagation and enhances the selective incorporation of reliable responses. These results establish DOWN as an optimization strategy for efficient and effective multiagent reasoning, facilitating the practical deployment of LLM-based collaboration. 

**Abstract (ZH)**: 必要的时候才辩论：一种基于置信度的多智能体辩论框架 

---
# Transforming Future Data Center Operations and Management via Physical AI 

**Title (ZH)**: 通过物理AI转型未来数据中心的操作与管理 

**Authors**: Zhiwei Cao, Minghao Li, Feng Lin, Qiang Fu, Jimin Jia, Yonggang Wen, Jianxiong Yin, Simon See  

**Link**: [PDF](https://arxiv.org/pdf/2504.04982)  

**Abstract**: Data centers (DCs) as mission-critical infrastructures are pivotal in powering the growth of artificial intelligence (AI) and the digital economy. The evolution from Internet DC to AI DC has introduced new challenges in operating and managing data centers for improved business resilience and reduced total cost of ownership. As a result, new paradigms, beyond the traditional approaches based on best practices, must be in order for future data centers. In this research, we propose and develop a novel Physical AI (PhyAI) framework for advancing DC operations and management. Our system leverages the emerging capabilities of state-of-the-art industrial products and our in-house research and development. Specifically, it presents three core modules, namely: 1) an industry-grade in-house simulation engine to simulate DC operations in a highly accurate manner, 2) an AI engine built upon NVIDIA PhysicsNemo for the training and evaluation of physics-informed machine learning (PIML) models, and 3) a digital twin platform built upon NVIDIA Omniverse for our proposed 5-tier digital twin framework. This system presents a scalable and adaptable solution to digitalize, optimize, and automate future data center operations and management, by enabling real-time digital twins for future data centers. To illustrate its effectiveness, we present a compelling case study on building a surrogate model for predicting the thermal and airflow profiles of a large-scale DC in a real-time manner. Our results demonstrate its superior performance over traditional time-consuming Computational Fluid Dynamics/Heat Transfer (CFD/HT) simulation, with a median absolute temperature prediction error of 0.18 °C. This emerging approach would open doors to several potential research directions for advancing Physical AI in future DC operations. 

**Abstract (ZH)**: 数据中心（DCs）作为关键基础设施，在推动人工智能（AI）和数字经济的发展中扮演着重要角色。从互联网数据中心到人工智能数据中心的演变引入了新的运营和管理挑战，旨在提高业务弹性和降低总拥有成本。因此，必须提出超越传统方法的新范式，以适应未来数据中心的需求。在本研究中，我们提出并开发了一种新型物理人工智能（PhyAI）框架，以推动数据中心的运营和管理。该系统利用了先进的工业产品及其自身的研发能力，具体包括三个核心模块：1）工业级的自研仿真引擎，以高精度模拟数据中心的运行；2）基于NVIDIA PhysicsNemo构建的AI引擎，用于训练和评估物理信息机器学习（PIML）模型；3）基于NVIDIA Omniverse构建的数字孪生平台，用于我们的五级数字孪生框架。该系统提供了一种可扩展且灵活的解决方案，以数字化、优化和自动化未来的数据中心运营和管理，并通过实时数字孪生为未来数据中心赋能。为了展示其有效性，我们展示了构建大型数据中心实时热流预测代理模型的案例研究。实验结果表明，该方法在温度预测误差中位数为0.18°C的情况下，优于传统的耗时计算流体动力学/热传递（CFD/HT）模拟，为推进未来数据中心的物理人工智能提供了新的研究方向。 

---
# GOTHAM: Graph Class Incremental Learning Framework under Weak Supervision 

**Title (ZH)**: GOTHAM: 在弱监督下的图类增量学习框架 

**Authors**: Aditya Hemant Shahane, Prathosh A.P, Sandeep Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2504.04954)  

**Abstract**: Graphs are growing rapidly, along with the number of distinct label categories associated with them. Applications like e-commerce, healthcare, recommendation systems, and various social media platforms are rapidly moving towards graph representation of data due to their ability to capture both structural and attribute information. One crucial task in graph analysis is node classification, where unlabeled nodes are categorized into predefined classes. In practice, novel classes appear incrementally sometimes with just a few labels (seen classes) or even without any labels (unseen classes), either because they are new or haven't been explored much. Traditional methods assume abundant labeled data for training, which isn't always feasible. We investigate a broader objective: \emph{Graph Class Incremental Learning under Weak Supervision (GCL)}, addressing this challenge by meta-training on base classes with limited labeled instances. During the incremental streams, novel classes can have few-shot or zero-shot representation. Our proposed framework GOTHAM efficiently accommodates these unlabeled nodes by finding the closest prototype representation, serving as class representatives in the attribute space. For Text-Attributed Graphs (TAGs), our framework additionally incorporates semantic information to enhance the representation. By employing teacher-student knowledge distillation to mitigate forgetting, GOTHAM achieves promising results across various tasks. Experiments on datasets such as Cora-ML, Amazon, and OBGN-Arxiv showcase the effectiveness of our approach in handling evolving graph data under limited supervision. The repository is available here: \href{this https URL}{\small \textcolor{blue}{Code}} 

**Abstract (ZH)**: 图谱在快速增长，与它们相关的不同标签类别也在快速增长。诸如电子商务、医疗保健、推荐系统以及各种社交媒体平台等应用正迅速转向图表示，利用其能够同时捕捉结构和属性信息的能力。图分析中的一项关键任务是节点分类，即将未标注的节点归类为预定义的类别。在实践中，新类别有时会出现，仅仅伴随少数标签（已见过的类别），甚至可能没有任何标签（未见过的类别），这可能是由于它们是新的或尚未被充分探索。传统方法假设有大量的带标签数据用于训练，而在实际情况中这并不总是可行的。我们研究了一个更广泛的目标：在弱监督下的图类别增量学习（GCL），通过在有限的带标签实例上进行元训练来应对这一挑战。在增量流中，新类别可能具有少样本或零样本表示。我们提出的框架GOTHAM通过找到最接近的原型表示有效地处理这些未标注节点，该原型表示在属性空间中作为类代表。对于文本属性图（TAGs），我们的框架还结合了语义信息以增强表示。通过应用教师-学生知识蒸馏来减轻遗忘，GOTHAM在各种任务中取得了令人鼓舞的结果。在Cora-ML、Amazon和OBGN-Arxiv等数据集上的实验展示了我们的方法在有限监督下处理演化图数据的有效性。代码仓库见这里：\href{this https URL}{\small \textcolor{blue}{Code}}。 

---
# Lemmanaid: Neuro-Symbolic Lemma Conjecturing 

**Title (ZH)**: Lemmanaid: 神经符号命题猜想助手 

**Authors**: Yousef Alhessi, Sólrún Halla Einarsdóttir, George Granberry, Emily First, Moa Johansson, Sorin Lerner, Nicholas Smallbone  

**Link**: [PDF](https://arxiv.org/pdf/2504.04942)  

**Abstract**: Automatically conjecturing useful, interesting and novel lemmas would greatly improve automated reasoning tools and lower the bar for formalizing mathematics in proof assistants. It is however a very challenging task for both neural and symbolic approaches. We present the first steps towards a practical neuro-symbolic lemma conjecturing tool, Lemmanaid, that combines Large Language Models (LLMs) and symbolic methods, and evaluate it on proof libraries for the Isabelle proof assistant. We train an LLM to generate lemma templates that describe the shape of a lemma, and use symbolic methods to fill in the details. We compare Lemmanaid against an LLM trained to generate complete lemma statements as well as previous fully symbolic conjecturing methods. Our results indicate that neural and symbolic techniques are complementary. By leveraging the best of both symbolic and neural methods we can generate useful lemmas for a wide range of input domains, facilitating computer-assisted theory development and formalization. 

**Abstract (ZH)**: 自动猜想有用的、有趣的和新颖的引理将极大地提高自动推理工具的性能，并降低在证明辅助工具中形式化数学的门槛。然而，这既是神经方法又是符号方法面临的一项非常具有挑战性的任务。我们提出了第一个实用的神经-符号引理猜想工具Lemmanaid，该工具结合了大型语言模型和符号方法，并在Isabelle证明辅助工具的证明库上对其进行评估。我们训练一个大型语言模型生成描述引理形状的模板，并使用符号方法填充细节。我们将Lemmanaid与一个生成完整引理陈述的大型语言模型以及之前的完全符号猜想方法进行比较。我们的结果表明，神经技术和符号技术是互补的。通过利用两者之长，我们可以为广泛的输入领域生成有用的引理，从而促进计算机辅助的理论发展和形式化。 

---
# Constitution or Collapse? Exploring Constitutional AI with Llama 3-8B 

**Title (ZH)**: 宪法还是崩溃？探索基于Llama 3-8B的宪法AI 

**Authors**: Xue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04918)  

**Abstract**: As language models continue to grow larger, the cost of acquiring high-quality training data has increased significantly. Collecting human feedback is both expensive and time-consuming, and manual labels can be noisy, leading to an imbalance between helpfulness and harmfulness. Constitutional AI, introduced by Anthropic in December 2022, uses AI to provide feedback to another AI, greatly reducing the need for human labeling. However, the original implementation was designed for a model with around 52 billion parameters, and there is limited information on how well Constitutional AI performs with smaller models, such as LLaMA 3-8B. In this paper, we replicated the Constitutional AI workflow using the smaller LLaMA 3-8B model. Our results show that Constitutional AI can effectively increase the harmlessness of the model, reducing the Attack Success Rate in MT-Bench by 40.8%. However, similar to the original study, increasing harmlessness comes at the cost of helpfulness. The helpfulness metrics, which are an average of the Turn 1 and Turn 2 scores, dropped by 9.8% compared to the baseline. Additionally, we observed clear signs of model collapse in the final DPO-CAI model, indicating that smaller models may struggle with self-improvement due to insufficient output quality, making effective fine-tuning more challenging. Our study suggests that, like reasoning and math ability, self-improvement is an emergent property. 

**Abstract (ZH)**: 随着语言模型不断增大，高质量训练数据的获取成本显著增加。收集人类反馈既昂贵又耗时，人工标注还可能存在噪音，导致帮助性和有害性之间的不平衡。Anthropic于2022年12月提出的宪法AI使用AI提供反馈给另一个AI，大大减少了对人类标注的需求。然而，最初的实现是为一个大约有520亿参数的模型设计的，关于宪法AI在较小模型，如LLaMA 3-8B上表现的信息有限。在本文中，我们使用较小的LLaMA 3-8B模型复现了宪法AI的工作流程。结果显示，宪法AI有效提高了模型的无害性，使MT-Bench中的攻击成功率降低了40.8%。然而，与原始研究相似，增加无害性以降低有害性会牺牲帮助性。帮助性指标，即Turn 1和Turn 2得分的平均值，相比于基线下降了9.8%。此外，我们还在最终的DPO-CAI模型中观察到了明显的模型崩溃迹象，表明较小的模型可能因为输出质量不足而在自我改进方面面临挑战，使得有效的微调更加困难。我们的研究表明，与推理和数学能力一样，自我改进是一种 emergent 属性。 

---
# GAMDTP: Dynamic Trajectory Prediction with Graph Attention Mamba Network 

**Title (ZH)**: GAMDTP：基于图注意力Mamba网络的动态轨迹预测 

**Authors**: Yunxiang Liu, Hongkuo Niu, Jianlin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04862)  

**Abstract**: Accurate motion prediction of traffic agents is crucial for the safety and stability of autonomous driving systems. In this paper, we introduce GAMDTP, a novel graph attention-based network tailored for dynamic trajectory prediction. Specifically, we fuse the result of self attention and mamba-ssm through a gate mechanism, leveraging the strengths of both to extract features more efficiently and accurately, in each graph convolution layer. GAMDTP encodes the high-definition map(HD map) data and the agents' historical trajectory coordinates and decodes the network's output to generate the final prediction results. Additionally, recent approaches predominantly focus on dynamically fusing historical forecast results and rely on two-stage frameworks including proposal and refinement. To further enhance the performance of the two-stage frameworks we also design a scoring mechanism to evaluate the prediction quality during the proposal and refinement processes. Experiments on the Argoverse dataset demonstrates that GAMDTP achieves state-of-the-art performance, achieving superior accuracy in dynamic trajectory prediction. 

**Abstract (ZH)**: 准确的交通代理运动预测对于自动驾驶系统的安全与稳定性至关重要。在本文中，我们介绍了GAMDTP，一种专门为动态轨迹预测设计的图注意力网络。具体而言，我们通过门控机制融合自注意力和mamba-ssm的结果，利用两者的优势在每个图卷积层中更高效、更准确地提取特征。GAMDTP 编码高精度地图数据和代理的历史轨迹坐标，并解码网络输出以生成最终预测结果。此外，近期的方法主要侧重于动态融合历史预测结果，并依赖两阶段框架包括提案和细化。为了进一步提升两阶段框架的性能，我们设计了一种评分机制，在提案和细化过程中评估预测质量。实验结果表明，GAMDTP 在动态轨迹预测方面达到了最先进的性能，取得了更高的准确性。 

---
# Don't Lag, RAG: Training-Free Adversarial Detection Using RAG 

**Title (ZH)**: 别滞后，RAG：基于RAG的无需训练的对抗检测 

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar  

**Link**: [PDF](https://arxiv.org/pdf/2504.04858)  

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks. 

**Abstract (ZH)**: 对抗补丁攻击通过嵌入局部扰动误导深度模型，对视觉系统构成重大威胁。传统的防御方法往往需要重新训练或微调，这在实际部署中并不实用。我们提出了一种无需训练的视觉检索增强生成（VRAG）框架，该框架结合了视觉语言模型（VLMs）进行对抗补丁检测。通过检索与存储攻击在不断扩展的数据库中相似的补丁和图像，VRAG进行生成推理以识别多种攻击类型，而无需额外的训练或微调。我们广泛评估了开源大规模VLMs，包括Qwen-VL-Plus、Qwen2.5-VL-72B和UI-TARS-72B-DPO，以及封闭源代码模型Gemini-2.0。值得注意的是，开源的UI-TARS-72B-DPO模型在对抗补丁检测中的分类准确率最高，达到95%，从而在开源对抗补丁检测中达到新的最先进水平。Gemini-2.0的整体准确率达到最高，为98%，但仍然是封闭源代码。实验结果证明，VRAG能够有效地识别各种类型的对抗补丁，且需要的少量人工标注，从而为应对不断演变的对抗补丁攻击提供稳健且实用的防御。 

---
# BIASINSPECTOR: Detecting Bias in Structured Data through LLM Agents 

**Title (ZH)**: BIASINSPECTOR: 通过LLM代理检测结构化数据中的偏见 

**Authors**: Haoxuan Li, Mingyu Derek Ma, Jen-tse Huang, Zhaotian Weng, Wei Wang, Jieyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.04855)  

**Abstract**: Detecting biases in structured data is a complex and time-consuming task. Existing automated techniques are limited in diversity of data types and heavily reliant on human case-by-case handling, resulting in a lack of generalizability. Currently, large language model (LLM)-based agents have made significant progress in data science, but their ability to detect data biases is still insufficiently explored. To address this gap, we introduce the first end-to-end, multi-agent synergy framework, BIASINSPECTOR, designed for automatic bias detection in structured data based on specific user requirements. It first develops a multi-stage plan to analyze user-specified bias detection tasks and then implements it with a diverse and well-suited set of tools. It delivers detailed results that include explanations and visualizations. To address the lack of a standardized framework for evaluating the capability of LLM agents to detect biases in data, we further propose a comprehensive benchmark that includes multiple evaluation metrics and a large set of test cases. Extensive experiments demonstrate that our framework achieves exceptional overall performance in structured data bias detection, setting a new milestone for fairer data applications. 

**Abstract (ZH)**: 检测结构化数据中的偏差是一项复杂且耗时的任务。现有的自动化技术在数据类型多样性方面有限，并且高度依赖于人工逐案处理，导致缺乏一般化能力。目前，基于大型语言模型（LLM）的代理在数据科学领域取得了显著进展，但它们检测数据偏差的能力尚未得到充分探索。为填补这一空白，我们提出了第一个端到端的多代理协同框架BIASINSPECTOR，该框架旨在根据特定用户需求自动检测结构化数据中的偏差。它首先开发一个多阶段计划以分析用户指定的偏差检测任务，然后使用多样化且合适的工具集来实现这一计划。它提供详细的检测结果，包括解释和可视化。为解决评估LLM代理检测数据偏差能力缺乏标准化框架的问题，我们进一步提出了一种全面的基准测试，其中包括多种评估指标和大量测试案例。广泛实验显示，我们的框架在结构化数据偏差检测方面的整体性能出色，为更公平的数据应用设立了新的里程碑。 

---
# An Efficient Approach for Cooperative Multi-Agent Learning Problems 

**Title (ZH)**: 一种高效的多agent协作学习问题解决方法 

**Authors**: Ángel Aso-Mollar, Eva Onaindia  

**Link**: [PDF](https://arxiv.org/pdf/2504.04850)  

**Abstract**: In this article, we propose a centralized Multi-Agent Learning framework for learning a policy that models the simultaneous behavior of multiple agents that need to coordinate to solve a certain task. Centralized approaches often suffer from the explosion of an action space that is defined by all possible combinations of individual actions, known as joint actions. Our approach addresses the coordination problem via a sequential abstraction, which overcomes the scalability problems typical to centralized methods. It introduces a meta-agent, called \textit{supervisor}, which abstracts joint actions as sequential assignments of actions to each agent. This sequential abstraction not only simplifies the centralized joint action space but also enhances the framework's scalability and efficiency. Our experimental results demonstrate that the proposed approach successfully coordinates agents across a variety of Multi-Agent Learning environments of diverse sizes. 

**Abstract (ZH)**: 本文提出了一种集中式的多agent学习框架，用于学习一种模型多个需要协调以解决特定任务的agent同时行为的策略。中心化方法通常会受到由所有个体动作可能组合定义的动作空间爆炸问题的影响，即联合动作。我们的方法通过顺序抽象来解决协调问题，从而克服了中心化方法典型的可扩展性问题。该方法引入了一个元agent，称为“监督者”，将联合动作抽象为按顺序分配给每个agent的动作。这种顺序抽象不仅简化了中心化的联合动作空间，还增强了框架的可扩展性和效率。我们的实验结果表明，提出的框架成功地协调了各种大小和类型的多agent学习环境中的agent。 

---
# Multimodal Agricultural Agent Architecture (MA3): A New Paradigm for Intelligent Agricultural Decision-Making 

**Title (ZH)**: 多模态农业代理架构 (MA3): 一种新的智能农业决策范式 

**Authors**: Zhuoning Xu, Jian Xu, Mingqing Zhang, Peijie Wang, Chao Deng, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04789)  

**Abstract**: As a strategic pillar industry for human survival and development, modern agriculture faces dual challenges: optimizing production efficiency and achieving sustainable development. Against the backdrop of intensified climate change leading to frequent extreme weather events, the uncertainty risks in agricultural production systems are increasing exponentially. To address these challenges, this study proposes an innovative \textbf{M}ultimodal \textbf{A}gricultural \textbf{A}gent \textbf{A}rchitecture (\textbf{MA3}), which leverages cross-modal information fusion and task collaboration mechanisms to achieve intelligent agricultural decision-making. This study constructs a multimodal agricultural agent dataset encompassing five major tasks: classification, detection, Visual Question Answering (VQA), tool selection, and agent evaluation. We propose a unified backbone for sugarcane disease classification and detection tools, as well as a sugarcane disease expert model. By integrating an innovative tool selection module, we develop a multimodal agricultural agent capable of effectively performing tasks in classification, detection, and VQA. Furthermore, we introduce a multi-dimensional quantitative evaluation framework and conduct a comprehensive assessment of the entire architecture over our evaluation dataset, thereby verifying the practicality and robustness of MA3 in agricultural scenarios. This study provides new insights and methodologies for the development of agricultural agents, holding significant theoretical and practical implications. Our source code and dataset will be made publicly available upon acceptance. 

**Abstract (ZH)**: 作为人类生存与发展的重要战略支柱产业，现代农业面临着优化生产效率和实现可持续发展的双重挑战。在气候变化加剧导致极端天气事件频发的背景下，农业生产系统的不确定性风险呈指数级增加。为应对这些挑战，本研究提出了一种创新的多模态农业代理架构（MA3），该架构通过跨模态信息融合和任务协作机制实现智能化农业决策。本研究构建了一个涵盖五项主要任务的多模态农业代理数据集，包括分类、检测、视觉问答（VQA）、工具选择和代理评估。我们提出了一种统一的骨干网络，用于甘蔗病害分类和检测工具，以及甘蔗病害专家模型。通过集成一种创新的工具选择模块，我们开发出一种能够有效地在分类、检测和VQA任务中执行的多模态农业代理。此外，我们引入了一个多维度的定量评估框架，并对整个架构在评估数据集上的表现进行了全面评估，从而验证了MA3在农业场景中的实用性和鲁棒性。本研究为农业代理的开发提供了新的见解和方法论，拥有重要的理论和实践意义。我们的源代码和数据集将在录用后公开。 

---
# Weak-for-Strong: Training Weak Meta-Agent to Harness Strong Executors 

**Title (ZH)**: 弱对强：训练弱元代理协调强执行器 

**Authors**: Fan Nie, Lan Feng, Haotian Ye, Weixin Liang, Pan Lu, Huaxiu Yao, Alexandre Alahi, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2504.04785)  

**Abstract**: Efficiently leveraging of the capabilities of contemporary large language models (LLMs) is increasingly challenging, particularly when direct fine-tuning is expensive and often impractical. Existing training-free methods, including manually or automated designed workflows, typically demand substantial human effort or yield suboptimal results. This paper proposes Weak-for-Strong Harnessing (W4S), a novel framework that customizes smaller, cost-efficient language models to design and optimize workflows for harnessing stronger models. W4S formulates workflow design as a multi-turn markov decision process and introduces reinforcement learning for agentic workflow optimization (RLAO) to train a weak meta-agent. Through iterative interaction with the environment, the meta-agent learns to design increasingly effective workflows without manual intervention. Empirical results demonstrate the superiority of W4S that our 7B meta-agent, trained with just one GPU hour, outperforms the strongest baseline by 2.9% ~ 24.6% across eleven benchmarks, successfully elevating the performance of state-of-the-art models such as GPT-3.5-Turbo and GPT-4o. Notably, W4S exhibits strong generalization capabilities across both seen and unseen tasks, offering an efficient, high-performing alternative to directly fine-tuning strong models. 

**Abstract (ZH)**: 高效利用当代大规模语言模型的能力：一种新型Weak-for-Strong Harnessing框架 

---
# Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use 

**Title (ZH)**: 合成数据生成与多步强化学习推理及工具使用 

**Authors**: Anna Goldie, Azalia Mirhoseini, Hao Zhou, Irene Cai, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2504.04736)  

**Abstract**: Reinforcement learning has been shown to improve the performance of large language models. However, traditional approaches like RLHF or RLAIF treat the problem as single-step. As focus shifts toward more complex reasoning and agentic tasks, language models must take multiple steps of text generation, reasoning and environment interaction before generating a solution. We propose a synthetic data generation and RL methodology targeting multi-step optimization scenarios. This approach, called Step-Wise Reinforcement Learning (SWiRL), iteratively generates multi-step reasoning and tool use data, and then learns from that data. It employs a simple step-wise decomposition that breaks each multi-step trajectory into multiple sub-trajectories corresponding to each action by the original model. It then applies synthetic data filtering and RL optimization on these sub-trajectories. We evaluated SWiRL on a number of multi-step tool use, question answering, and mathematical reasoning tasks. Our experiments show that SWiRL outperforms baseline approaches by 21.5%, 12.3%, 14.8%, 11.1%, and 15.3% in relative accuracy on GSM8K, HotPotQA, CofCA, MuSiQue, and BeerQA, respectively. Excitingly, the approach exhibits generalization across tasks: for example, training only on HotPotQA (text question-answering) improves zero-shot performance on GSM8K (a math dataset) by a relative 16.9%. 

**Abstract (ZH)**: 逐步强化学习（SWiRL）：针对多步骤优化场景的合成数据生成与RL方法 

---
# Generalising from Self-Produced Data: Model Training Beyond Human Constraints 

**Title (ZH)**: 基于自我生成数据的泛化：超越人类约束的模型训练 

**Authors**: Alfath Daryl Alhajir, Jennifer Dodgson, Joseph Lim, Truong Ma Phi, Julian Peh, Akira Rafhael Janson Pattirane, Lokesh Poovaragan  

**Link**: [PDF](https://arxiv.org/pdf/2504.04711)  

**Abstract**: Current large language models (LLMs) are constrained by human-derived training data and limited by a single level of abstraction that impedes definitive truth judgments. This paper introduces a novel framework in which AI models autonomously generate and validate new knowledge through direct interaction with their environment. Central to this approach is an unbounded, ungamable numeric reward - such as annexed disk space or follower count - that guides learning without requiring human benchmarks. AI agents iteratively generate strategies and executable code to maximize this metric, with successful outcomes forming the basis for self-retraining and incremental generalisation. To mitigate model collapse and the warm start problem, the framework emphasizes empirical validation over textual similarity and supports fine-tuning via GRPO. The system architecture employs modular agents for environment analysis, strategy generation, and code synthesis, enabling scalable experimentation. This work outlines a pathway toward self-improving AI systems capable of advancing beyond human-imposed constraints toward autonomous general intelligence. 

**Abstract (ZH)**: 当前的大语言模型受限于人类提供的训练数据，并且受限于单一抽象层次，这妨碍了它们做出终极真相判断。本文提出了一种新型框架，通过AI模型自主与环境直接交互生成和验证新知识。该方法的核心是一种无边界且不可作弊的数值奖励，如附加磁盘空间或追随者数量，这种奖励引导学习而不必依赖人类基准。AI代理通过迭代生成策略和可执行代码以最大化该指标，成功的成果成为自我重新训练和逐步泛化的基础。为了缓解模型崩溃和暖启动问题，该框架强调经验验证而非文本相似性，并通过GRPO支持微调。该系统架构采用模块化的代理进行环境分析、策略生成和代码合成，以实现可扩展的实验。本文概述了一条通往自我改进的AI系统的发展路径，这些系统能够超越人类施加的限制，朝着自主通用人工智能前进。 

---
# HypRL: Reinforcement Learning of Control Policies for Hyperproperties 

**Title (ZH)**: HypRL: 增强学习在超属性控制策略中的应用 

**Authors**: Tzu-Han Hsu, Arshia Rafieioskouei, Borzoo Bonakdarpour  

**Link**: [PDF](https://arxiv.org/pdf/2504.04675)  

**Abstract**: We study the problem of learning control policies for complex tasks whose requirements are given by a hyperproperty. The use of hyperproperties is motivated by their significant power to formally specify requirements of multi-agent systems as well as those that need expressiveness in terms of multiple execution traces (e.g., privacy and fairness). Given a Markov decision process M with unknown transitions (representing the environment) and a HyperLTL formula $\varphi$, our approach first employs Skolemization to handle quantifier alternations in $\varphi$. We introduce quantitative robustness functions for HyperLTL to define rewards of finite traces of M with respect to $\varphi$. Finally, we utilize a suitable reinforcement learning algorithm to learn (1) a policy per trace quantifier in $\varphi$, and (2) the probability distribution of transitions of M that together maximize the expected reward and, hence, probability of satisfaction of $\varphi$ in M. We present a set of case studies on (1) safety-preserving multi-agent path planning, (2) fairness in resource allocation, and (3) the post-correspondence problem (PCP). 

**Abstract (ZH)**: 我们研究了使用超性质学习复杂数学任务控制策略的问题。我们采用超性质的原因在于其强大的能力，能够形式化规范多智能体系统的需求，以及那些需要在多条执行轨迹方面具有表现力的需求（例如隐私和公平）。给定一个具有未知转换的马尔可夫决策过程M（代表环境）和一个HyperLTL公式$\varphi$，我们的方法首先使用斯科莱化处理$\varphi$中的量词交替。我们为HyperLTL引入了定量鲁棒性函数，以此定义M的有限轨迹相对于$\varphi$的奖励。最后，我们利用合适的强化学习算法学习（1）$\varphi$中每条轨迹量词的策略，以及（2）M的转换概率分布，这些分布共同最大化期望奖励，从而最大化$\varphi$在M中被满足的概率。我们展示了关于（1）保安全的多智能体路径规划、（2）资源分配的公平性以及（3）后 correspondence问题（PCP）的案例研究。 

---
# AI in a vat: Fundamental limits of efficient world modelling for agent sandboxing and interpretability 

**Title (ZH)**: AI在容器中：智能体沙箱化和可解释性下的高效世界建模基本极限 

**Authors**: Fernando Rosas, Alexander Boyd, Manuel Baltieri  

**Link**: [PDF](https://arxiv.org/pdf/2504.04608)  

**Abstract**: Recent work proposes using world models to generate controlled virtual environments in which AI agents can be tested before deployment to ensure their reliability and safety. However, accurate world models often have high computational demands that can severely restrict the scope and depth of such assessments. Inspired by the classic `brain in a vat' thought experiment, here we investigate ways of simplifying world models that remain agnostic to the AI agent under evaluation. By following principles from computational mechanics, our approach reveals a fundamental trade-off in world model construction between efficiency and interpretability, demonstrating that no single world model can optimise all desirable characteristics. Building on this trade-off, we identify procedures to build world models that either minimise memory requirements, delineate the boundaries of what is learnable, or allow tracking causes of undesirable outcomes. In doing so, this work establishes fundamental limits in world modelling, leading to actionable guidelines that inform core design choices related to effective agent evaluation. 

**Abstract (ZH)**: 近期的研究提出使用世界模型生成可控的虚拟环境，以在部署AI代理之前对其进行测试，确保其可靠性和安全性。然而，准确的世界模型往往具有较高的计算需求，这会严重限制这类评估的范围和深度。受经典的“脑在 vat 中”思想实验的启发，我们探讨了一种简化世界模型的方法，这种方法对正在评估的AI代理是无偏见的。通过遵循计算力学的原则，我们的方法揭示了世界模型构建中效率与可解释性之间的基本权衡，表明没有一个世界模型能够同时优化所有理想特性。基于这种权衡，我们确定了构建世界模型的程序，要么尽量减少内存需求，要么明确可学习的边界，要么允许追踪不良结果的原因。通过这种方式，本研究确立了世界建模的基本限制，并为有效的代理评估的核心设计选择提供了可操作的指导方针。 

---
# Capturing AI's Attention: Physics of Repetition, Hallucination, Bias and Beyond 

**Title (ZH)**: 捕捉AI的注意：重复、幻觉、偏见以及其他物理学原理 

**Authors**: Frank Yingjie Huo, Neil F. Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2504.04600)  

**Abstract**: We derive a first-principles physics theory of the AI engine at the heart of LLMs' 'magic' (e.g. ChatGPT, Claude): the basic Attention head. The theory allows a quantitative analysis of outstanding AI challenges such as output repetition, hallucination and harmful content, and bias (e.g. from training and fine-tuning). Its predictions are consistent with large-scale LLM outputs. Its 2-body form suggests why LLMs work so well, but hints that a generalized 3-body Attention would make such AI work even better. Its similarity to a spin-bath means that existing Physics expertise could immediately be harnessed to help Society ensure AI is trustworthy and resilient to manipulation. 

**Abstract (ZH)**: 我们推导出LLMs“魔力”（如ChatGPT、Claude）核心AI引擎的基本注意头的第一性物理理论。该理论允许对输出重复、幻觉、有害内容和偏见（例如，来自训练和微调）等 Outstanding AI 挑战进行定量分析。其预测与大规模LLM输出一致。其二体形式说明了LLMs为何如此有效，但暗示了通用的三体注意机制将使AI工作表现更好。其与自旋浴的相似性意味着现有的物理专业知识可以立即被利用以帮助社会确保AI的可信度并使其更能抵抗操纵。 

---
# SECQUE: A Benchmark for Evaluating Real-World Financial Analysis Capabilities 

**Title (ZH)**: SECQUE: 一个评估实际金融分析能力的基准 

**Authors**: Noga Ben Yoash, Meni Brief, Oded Ovadia, Gil Shenderovitz, Moshik Mishaeli, Rachel Lemberg, Eitam Sheetrit  

**Link**: [PDF](https://arxiv.org/pdf/2504.04596)  

**Abstract**: We introduce SECQUE, a comprehensive benchmark for evaluating large language models (LLMs) in financial analysis tasks. SECQUE comprises 565 expert-written questions covering SEC filings analysis across four key categories: comparison analysis, ratio calculation, risk assessment, and financial insight generation. To assess model performance, we develop SECQUE-Judge, an evaluation mechanism leveraging multiple LLM-based judges, which demonstrates strong alignment with human evaluations. Additionally, we provide an extensive analysis of various models' performance on our benchmark. By making SECQUE publicly available, we aim to facilitate further research and advancements in financial AI. 

**Abstract (ZH)**: 我们介绍SECQUE，一个全面的基准，用于评估大型语言模型在金融分析任务中的表现。SECQUE包括565个由专家撰写的题目，涵盖了SEC报表分析的四个关键类别：比较分析、比率计算、风险评估和财务洞察生成。为了评估模型性能，我们开发了SECQUE-Judge，一种利用多个基于语言模型的评估机制，显示出与人工评估的高度一致性。此外，我们还对各种模型在基准上的表现进行了详细的分析。通过公开提供SECQUE，我们旨在促进金融人工智能领域进一步的研究和进步。 

---
# Hierarchical Planning for Complex Tasks with Knowledge Graph-RAG and Symbolic Verification 

**Title (ZH)**: 基于知识图谱-RAG和符号验证的复杂任务分层规划 

**Authors**: Cristina Cornelio, Flavio Petruzzellis, Pietro Lio  

**Link**: [PDF](https://arxiv.org/pdf/2504.04578)  

**Abstract**: Large Language Models (LLMs) have shown promise as robotic planners but often struggle with long-horizon and complex tasks, especially in specialized environments requiring external knowledge. While hierarchical planning and Retrieval-Augmented Generation (RAG) address some of these challenges, they remain insufficient on their own and a deeper integration is required for achieving more reliable systems. To this end, we propose a neuro-symbolic approach that enhances LLMs-based planners with Knowledge Graph-based RAG for hierarchical plan generation. This method decomposes complex tasks into manageable subtasks, further expanded into executable atomic action sequences. To ensure formal correctness and proper decomposition, we integrate a Symbolic Validator, which also functions as a failure detector by aligning expected and observed world states. Our evaluation against baseline methods demonstrates the consistent significant advantages of integrating hierarchical planning, symbolic verification, and RAG across tasks of varying complexity and different LLMs. Additionally, our experimental setup and novel metrics not only validate our approach for complex planning but also serve as a tool for assessing LLMs' reasoning and compositional capabilities. 

**Abstract (ZH)**: 基于知识图谱的检索增强生成的神经符号规划方法：结合层次规划、符号验证和大型语言模型 

---
# AGITB: A Signal-Level Benchmark for Evaluating Artificial General Intelligence 

**Title (ZH)**: AGITB: 信号级评估人工通用智能基准 

**Authors**: Matej Šprogar  

**Link**: [PDF](https://arxiv.org/pdf/2504.04430)  

**Abstract**: Despite remarkable progress in machine learning, current AI systems continue to fall short of true human-like intelligence. While Large Language Models (LLMs) excel in pattern recognition and response generation, they lack genuine understanding - an essential hallmark of Artificial General Intelligence (AGI). Existing AGI evaluation methods fail to offer a practical, gradual, and informative metric. This paper introduces the Artificial General Intelligence Test Bed (AGITB), comprising twelve rigorous tests that form a signal-processing-level foundation for the potential emergence of cognitive capabilities. AGITB evaluates intelligence through a model's ability to predict binary signals across time without relying on symbolic representations or pretraining. Unlike high-level tests grounded in language or perception, AGITB focuses on core computational invariants reflective of biological intelligence, such as determinism, sensitivity, and generalisation. The test bed assumes no prior bias, operates independently of semantic meaning, and ensures unsolvability through brute force or memorization. While humans pass AGITB by design, no current AI system has met its criteria, making AGITB a compelling benchmark for guiding and recognizing progress toward AGI. 

**Abstract (ZH)**: 尽管机器学习取得了显著进步，当前的AI系统仍远未达到真正的类人智能。尽管大型语言模型在模式识别和响应生成方面表现出色，但它们缺乏真正的理解——这是通用人工智能（AGI）的一个基本特征。现有的AGI评估方法未能提供一个实用、渐进且信息丰富的度量标准。本文介绍了通用人工智能测试床（AGITB），它包含十二项严格的测试，为认知能力的潜在涌现提供了信号处理级别的基础。AGITB 通过模型预测时间序列中的二进制信号的能力进行智能评估，而不依赖于符号表示或预训练。与基于语言或感知的高级测试不同，AGITB 侧重于反映生物智能的核心计算不变量，如确定性、灵敏性和泛化能力。测试床假设不存在先验偏见，独立于语义意义运作，并通过穷举或记忆确保不可解性。虽然人类设计上能通过AGITB，但当前没有任何AI系统满足其标准，使其成为指导和识别通往AGI进展的理想基准。 

---
# Retro-Search: Exploring Untaken Paths for Deeper and Efficient Reasoning 

**Title (ZH)**: 逆向搜索：探索未走之路以实现更深更高效的推理 

**Authors**: Ximing Lu, Seungju Han, David Acuna, Hyunwoo Kim, Jaehun Jung, Shrimai Prabhumoye, Niklas Muennighoff, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.04383)  

**Abstract**: Large reasoning models exhibit remarkable reasoning capabilities via long, elaborate reasoning trajectories. Supervised fine-tuning on such reasoning traces, also known as distillation, can be a cost-effective way to boost reasoning capabilities of student models. However, empirical observations reveal that these reasoning trajectories are often suboptimal, switching excessively between different lines of thought, resulting in under-thinking, over-thinking, and even degenerate responses. We introduce Retro-Search, an MCTS-inspired search algorithm, for distilling higher quality reasoning paths from large reasoning models. Retro-Search retrospectively revises reasoning paths to discover better, yet shorter traces, which can then lead to student models with enhanced reasoning capabilities with shorter, thus faster inference. Our approach can enable two use cases: self-improvement, where models are fine-tuned on their own Retro-Search-ed thought traces, and weak-to-strong improvement, where a weaker model revises stronger model's thought traces via Retro-Search. For self-improving, R1-distill-7B, fine-tuned on its own Retro-Search-ed traces, reduces the average reasoning length by 31.2% while improving performance by 7.7% across seven math benchmarks. For weak-to-strong improvement, we retrospectively revise R1-671B's traces from the OpenThoughts dataset using R1-distill-32B as the Retro-Search-er, a model 20x smaller. Qwen2.5-32B, fine-tuned on this refined data, achieves performance comparable to R1-distill-32B, yielding an 11.3% reduction in reasoning length and a 2.4% performance improvement compared to fine-tuning on the original OpenThoughts data. Our work counters recently emergent viewpoints that question the relevance of search algorithms in the era of large reasoning models, by demonstrating that there are still opportunities for algorithmic advancements, even for frontier models. 

**Abstract (ZH)**: 大型推理模型通过长而复杂的推理轨迹展示了 remarkable 的推理能力。通过对这些推理轨迹进行监督微调，即蒸馏，可以成本效益地提升学生模型的推理能力。然而，实验观察表明，这些推理轨迹往往是 suboptimal 的，频繁地在不同思路之间切换，导致 under-thinking、over-thinking 和甚至退化的响应。我们引入了 Retro-Search，一种灵感来源于 MCTS 的搜索算法，用于从大型推理模型中蒸馏出更高质量的推理路径。Retro-Search 会回顾性地修订推理路径，发现更好的但更短的轨迹，从而引导出具有更强推理能力的学生模型，且推理更短，因此更快。我们的方法可以启用两种用例：自我改进，模型在其自身的 Retro-Search 修订的思考轨迹上进行微调，以及弱到强改进，较弱的模型通过 Retro-Search 修订较强模型的思考轨迹。对于自我改进，R1-distill-7B 在其自身的 Retro-Search 修订的轨迹上微调后，平均推理长度减少了 31.2%，七种数学基准的性能提高了 7.7%。对于弱到强改进，我们使用 R1-distill-32B 作为 Retro-Search 的工具，针对 OpenThoughts 数据集的轨迹进行回顾性修订，Qwen2.5-32B 在此精炼的数据上微调后，性能与 R1-distill-32B 相当，推理长度减少了 11.3%，性能提高了 2.4%，优于在原始 OpenThoughts 数据上微调的模型。我们的研究反驳了最近出现的观点，即在大型推理模型时代，搜索算法的相关性存疑，证明即使对于前沿模型，仍有机会进行算法上的改进。 

---
# Solving Sokoban using Hierarchical Reinforcement Learning with Landmarks 

**Title (ZH)**: 使用地标引导的分层强化学习求解Sokoban 

**Authors**: Sergey Pastukhov  

**Link**: [PDF](https://arxiv.org/pdf/2504.04366)  

**Abstract**: We introduce a novel hierarchical reinforcement learning (HRL) framework that performs top-down recursive planning via learned subgoals, successfully applied to the complex combinatorial puzzle game Sokoban. Our approach constructs a six-level policy hierarchy, where each higher-level policy generates subgoals for the level below. All subgoals and policies are learned end-to-end from scratch, without any domain knowledge. Our results show that the agent can generate long action sequences from a single high-level call. While prior work has explored 2-3 level hierarchies and subgoal-based planning heuristics, we demonstrate that deep recursive goal decomposition can emerge purely from learning, and that such hierarchies can scale effectively to hard puzzle domains. 

**Abstract (ZH)**: 我们介绍了一种新颖的分层强化学习（HRL）框架，通过学习子目标进行自上而下的递归规划，并成功应用于复杂的组合谜题游戏 sokoban。我们的方法构建了一个六级策略层次结构，其中每一级较高的策略为较低级别生成子目标。所有子目标和策略都是端到端地从头开始学习，无需任何领域知识。我们的结果表明，该代理可以从单个高层调用生成长动作序列。尽管先前的工作探索了2-3级层次结构和基于子目标的规划启发式方法，但我们展示了深度递归目标分解可以纯粹通过学习涌现，而且这样的层次结构可以有效地扩展到困难的谜题领域。 

---
# Crowdsourcing-Based Knowledge Graph Construction for Drug Side Effects Using Large Language Models with an Application on Semaglutide 

**Title (ZH)**: 基于 crowdsourcing 的大规模语言模型驱动的药物副作用知识图构建及其在赛吗GLU上的应用 

**Authors**: Zhijie Duan, Kai Wei, Zhaoqian Xue, Lingyao li, Jin Jin, Shu Yang, Jiayan Zhou, Siyuan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.04346)  

**Abstract**: Social media is a rich source of real-world data that captures valuable patient experience information for pharmacovigilance. However, mining data from unstructured and noisy social media content remains a challenging task. We present a systematic framework that leverages large language models (LLMs) to extract medication side effects from social media and organize them into a knowledge graph (KG). We apply this framework to semaglutide for weight loss using data from Reddit. Using the constructed knowledge graph, we perform comprehensive analyses to investigate reported side effects across different semaglutide brands over time. These findings are further validated through comparison with adverse events reported in the FAERS database, providing important patient-centered insights into semaglutide's side effects that complement its safety profile and current knowledge base of semaglutide for both healthcare professionals and patients. Our work demonstrates the feasibility of using LLMs to transform social media data into structured KGs for pharmacovigilance. 

**Abstract (ZH)**: 社交媒体是获取药物警戒中宝贵患者体验信息的丰富数据来源，但从中挖掘半结构化和噪声数据仍然是一个挑战性任务。我们提出了一种综合利用大型语言模型（LLMs）从社交媒体中抽取药物副作用并组织成知识图谱（KG）的系统框架。我们运用该框架基于Reddit数据对 semaglutide 的减肥副作用进行分析，并构建知识图谱以进行全面分析，探讨不同 semaglutide 品牌随时间变化的副作用报告情况。这些发现通过与 FAERS 数据库中的不良事件进行对比验证，为医疗保健专业人员和患者提供了 semaglutide 副作用的重要患者中心见解，补充其安全性概况和现有 semaglutide 知识库。我们的工作展示了利用 LLMs 将社交媒体数据转换为结构化 KGs 以进行药物警戒的可行性。 

---
# A Comparative Study of Explainable AI Methods: Model-Agnostic vs. Model-Specific Approaches 

**Title (ZH)**: 可解释AI方法的比较研究：模型无拘束方法 vs. 模型特定方法 

**Authors**: Keerthi Devireddy  

**Link**: [PDF](https://arxiv.org/pdf/2504.04276)  

**Abstract**: This paper compares model-agnostic and model-specific approaches to explainable AI (XAI) in deep learning image classification. I examine how LIME and SHAP (model-agnostic methods) differ from Grad-CAM and Guided Backpropagation (model-specific methods) when interpreting ResNet50 predictions across diverse image categories. Through extensive testing with various species from dogs and birds to insects I found that each method reveals different aspects of the models decision-making process. Model-agnostic techniques provide broader feature attribution that works across different architectures, while model-specific approaches excel at highlighting precise activation regions with greater computational efficiency. My analysis shows there is no "one-size-fits-all" solution for model interpretability. Instead, combining multiple XAI methods offers the most comprehensive understanding of complex models particularly valuable in high-stakes domains like healthcare, autonomous vehicles, and financial services where transparency is crucial. This comparative framework provides practical guidance for selecting appropriate interpretability techniques based on specific application needs and computational constraints. 

**Abstract (ZH)**: 本文将模型通用和模型专用方法对比应用于深度学习图像分类的可解释人工智能（XAI）。研究了在不同图像类别中，LIME和SHAP（模型通用方法）与Grad-CAM和Guided Backpropagation（模型专用方法）解释ResNet50预测结果的差异。通过对从狗、鸟到昆虫的各种物种进行广泛的测试，发现每种方法揭示了模型决策过程的不同方面。模型通用技术提供了更广泛的特征归因，适用于不同的架构，而模型专用方法则在突出精确激活区域方面更出色，且具有更高的计算效率。分析显示，并不存在适用于所有情况的模型可解释性解决方案。相反，结合多种XAI方法可以为复杂模型提供最全面的理解，尤其在如医疗保健、自动驾驶和金融服务等高风险领域中，透明度至关重要。这种比较框架为根据具体应用需求和计算约束选择合适的可解释性技术提供了实用指导。 

---
# Improving Chronic Kidney Disease Detection Efficiency: Fine Tuned CatBoost and Nature-Inspired Algorithms with Explainable AI 

**Title (ZH)**: 提高慢性肾病检测效率：Fine Tuned CatBoost与启发式算法结合的解释性AI 

**Authors**: Md. Ehsanul Haque, S. M. Jahidul Islam, Jeba Maliha, Md. Shakhauat Hossan Sumon, Rumana Sharmin, Sakib Rokoni  

**Link**: [PDF](https://arxiv.org/pdf/2504.04262)  

**Abstract**: Chronic Kidney Disease (CKD) is a major global health issue which is affecting million people around the world and with increasing rate of mortality. Mitigation of progression of CKD and better patient outcomes requires early detection. Nevertheless, limitations lie in traditional diagnostic methods, especially in resource constrained settings. This study proposes an advanced machine learning approach to enhance CKD detection by evaluating four models: Random Forest (RF), Multi-Layer Perceptron (MLP), Logistic Regression (LR), and a fine-tuned CatBoost algorithm. Specifically, among these, the fine-tuned CatBoost model demonstrated the best overall performance having an accuracy of 98.75%, an AUC of 0.9993 and a Kappa score of 97.35% of the studies. The proposed CatBoost model has used a nature inspired algorithm such as Simulated Annealing to select the most important features, Cuckoo Search to adjust outliers and grid search to fine tune its settings in such a way to achieve improved prediction accuracy. Features significance is explained by SHAP-a well-known XAI technique-for gaining transparency in the decision-making process of proposed model and bring up trust in diagnostic systems. Using SHAP, the significant clinical features were identified as specific gravity, serum creatinine, albumin, hemoglobin, and diabetes mellitus. The potential of advanced machine learning techniques in CKD detection is shown in this research, particularly for low income and middle-income healthcare settings where prompt and correct diagnoses are vital. This study seeks to provide a highly accurate, interpretable, and efficient diagnostic tool to add to efforts for early intervention and improved healthcare outcomes for all CKD patients. 

**Abstract (ZH)**: 慢性肾脏病（CKD）是全球性的重大健康问题，影响着全世界数百万人，并且死亡率呈上升趋势。通过早期检测减缓CKD的进展和提高患者预后需要先进的诊断方法。然而，传统诊断方法在资源受限的环境中存在局限性。本文提出了一种高级机器学习方法，通过评估四种模型——随机森林（RF）、多层感知器（MLP）、逻辑回归（LR）以及优化后的CatBoost算法——来增强CKD的检测。特别是，优化后的CatBoost模型展示了最佳的整体性能，准确率为98.75%，AUC为0.9993，卡帕系数为97.35%。提出的CatBoost模型采用了模拟退火等启发式算法来选择最重要的特征，使用了布谷鸟搜索来调整异常值，并通过网格搜索来优化其设置，从而实现提高预测准确性的目标。特征的重要性通过SHAP（一种广泛认可的可解释性人工智能技术）进行解释，以提高所提模型在决策过程中的透明度，增强诊断系统的信任度。研究发现，显著的临床特征包括比重、血清肌酐、白蛋白、血红蛋白和糖尿病。本文展示了高级机器学习技术在CKD检测中的潜力，特别是在低收入和中等收入国家的医疗保健环境中，及时和准确的诊断至关重要。本研究旨在提供一种高度准确、可解释和高效的诊断工具，以补充早期干预和改善所有CKD患者健康结果的努力。 

---
# Learning about the Physical World through Analytic Concepts 

**Title (ZH)**: 通过分析概念学习物理世界 

**Authors**: Jianhua Sun, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04170)  

**Abstract**: Reviewing the progress in artificial intelligence over the past decade, various significant advances (e.g. object detection, image generation, large language models) have enabled AI systems to produce more semantically meaningful outputs and achieve widespread adoption in internet scenarios. Nevertheless, AI systems still struggle when it comes to understanding and interacting with the physical world. This reveals an important issue: relying solely on semantic-level concepts learned from internet data (e.g. texts, images) to understand the physical world is far from sufficient -- machine intelligence currently lacks an effective way to learn about the physical world. This research introduces the idea of analytic concept -- representing the concepts related to the physical world through programs of mathematical procedures, providing machine intelligence a portal to perceive, reason about, and interact with the physical world. Except for detailing the design philosophy and providing guidelines for the application of analytic concepts, this research also introduce about the infrastructure that has been built around analytic concepts. I aim for my research to contribute to addressing these questions: What is a proper abstraction of general concepts in the physical world for machine intelligence? How to systematically integrate structured priors with neural networks to constrain AI systems to comply with physical laws? 

**Abstract (ZH)**: 过去十年人工智能进展回顾：从语义层面的概念到物理世界的分析概念 

---
# Introducing COGENT3: An AI Architecture for Emergent Cognition 

**Title (ZH)**: 介绍COGENT3：一种涌现认知的人工智能架构 

**Authors**: Eduardo Salazar  

**Link**: [PDF](https://arxiv.org/pdf/2504.04139)  

**Abstract**: This paper presents COGENT3 (or Collective Growth and Entropy-modulated Triads System), a novel approach for emergent cognition integrating pattern formation networks with group influence dynamics. Contrasting with traditional strategies that rely on predetermined architectures, computational structures emerge dynamically in our framework through agent interactions. This enables a more flexible and adaptive system exhibiting characteristics reminiscent of human cognitive processes. The incorporation of temperature modulation and memory effects in COGENT3 closely integrates statistical mechanics, machine learning, and cognitive science. 

**Abstract (ZH)**: COGENT3（或集体增长和熵调节三元系统）：一种将模式形成网络与群体影响动态相结合的新兴认知方法 

---
# Guaranteeing consistency in evidence fusion: A novel perspective on credibility 

**Title (ZH)**: 保证证据融合中的一致性：可信度的一种新视角 

**Authors**: Chaoxiong Ma, Yan Liang, Huixia Zhang, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.04128)  

**Abstract**: It is explored that available credible evidence fusion schemes suffer from the potential inconsistency because credibility calculation and Dempster's combination rule-based fusion are sequentially performed in an open-loop style. This paper constructs evidence credibility from the perspective of the degree of support for events within the framework of discrimination (FOD) and proposes an iterative credible evidence fusion (ICEF) to overcome the inconsistency in view of close-loop control. On one hand, the ICEF introduces the fusion result into credibility assessment to establish the correlation between credibility and the fusion result. On the other hand, arithmetic-geometric divergence is promoted based on the exponential normalization of plausibility and belief functions to measure evidence conflict, called plausibility-belief arithmetic-geometric divergence (PBAGD), which is superior in capturing the correlation and difference of FOD subsets, identifying abnormal sources, and reducing their fusion weights. The ICEF is compared with traditional methods by combining different evidence difference measure forms via numerical examples to verify its performance. Simulations on numerical examples and benchmark datasets reflect the adaptability of PBAGD to the proposed fusion strategy. 

**Abstract (ZH)**: 基于推理支持度的迭代可信证据融合（ICEF） metod及其应用 

---
# Improving Question Embeddings with Cognitiv Representation Optimization for Knowledge Tracing 

**Title (ZH)**: 基于认知表示优化提升问题嵌入表示以增强知识追踪 

**Authors**: Lixiang Xu, Xianwei Ding, Xin Yuan, Zhanlong Wang, Lu Bai, Enhong Chen, Philip S. Yu, Yuanyan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04121)  

**Abstract**: The Knowledge Tracing (KT) aims to track changes in students' knowledge status and predict their future answers based on their historical answer records. Current research on KT modeling focuses on predicting student' future performance based on existing, unupdated records of student learning interactions. However, these approaches ignore the distractors (such as slipping and guessing) in the answering process and overlook that static cognitive representations are temporary and limited. Most of them assume that there are no distractors in the answering process and that the record representations fully represent the students' level of understanding and proficiency in knowledge. In this case, it may lead to many insynergy and incoordination issue in the original records. Therefore we propose a Cognitive Representation Optimization for Knowledge Tracing (CRO-KT) model, which utilizes a dynamic programming algorithm to optimize structure of cognitive representations. This ensures that the structure matches the students' cognitive patterns in terms of the difficulty of the exercises. Furthermore, we use the co-optimization algorithm to optimize the cognitive representations of the sub-target exercises in terms of the overall situation of exercises responses by considering all the exercises with co-relationships as a single goal. Meanwhile, the CRO-KT model fuses the learned relational embeddings from the bipartite graph with the optimized record representations in a weighted manner, enhancing the expression of students' cognition. Finally, experiments are conducted on three publicly available datasets respectively to validate the effectiveness of the proposed cognitive representation optimization model. 

**Abstract (ZH)**: 基于认知表示优化的知识追踪(CRO-KT)模型 

---
# PEIRCE: Unifying Material and Formal Reasoning via LLM-Driven Neuro-Symbolic Refinement 

**Title (ZH)**: PEIRCE：通过LLM驱动的神经符号细化统一物质推理与形式推理 

**Authors**: Xin Quan, Marco Valentino, Danilo S. Carvalho, Dhairya Dalal, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2504.04110)  

**Abstract**: A persistent challenge in AI is the effective integration of material and formal inference - the former concerning the plausibility and contextual relevance of arguments, while the latter focusing on their logical and structural validity. Large Language Models (LLMs), by virtue of their extensive pre-training on large textual corpora, exhibit strong capabilities in material inference. However, their reasoning often lacks formal rigour and verifiability. At the same time, LLMs' linguistic competence positions them as a promising bridge between natural and formal languages, opening up new opportunities for combining these two modes of reasoning. In this paper, we introduce PEIRCE, a neuro-symbolic framework designed to unify material and formal inference through an iterative conjecture-criticism process. Within this framework, LLMs play the central role of generating candidate solutions in natural and formal languages, which are then evaluated and refined via interaction with external critique models. These critiques include symbolic provers, which assess formal validity, as well as soft evaluators that measure the quality of the generated arguments along linguistic and epistemic dimensions such as plausibility, coherence, and parsimony. While PEIRCE is a general-purpose framework, we demonstrate its capabilities in the domain of natural language explanation generation - a setting that inherently demands both material adequacy and formal correctness. 

**Abstract (ZH)**: AI中物质推理和形式推理的有效集成是一个持久的挑战——前者关注论据的合理性与上下文相关性，后者则聚焦于其逻辑和结构的有效性。大型语言模型（LLMs）由于大规模预训练在大量的文本语料上，表现出强大的物质推理能力。然而，它们的推理往往缺乏形式的严谨性和可验证性。同时，LLMs在语言上的能力使它们成为自然语言和形式语言之间的一个有前途的桥梁，为结合这两种推理模式开辟了新的机会。本文引入了PEIRCE，一种基于神经-符号框架，通过迭代假说-批判过程来统一物质推理和形式推理的设计。在这个框架中，LLMs在自然语言和形式语言中起着核心作用，生成候选解决方案，然后通过与外部批判模型的交互进行评估和修正。这些批判包括形式证明器评估形式有效性，以及软评估器衡量生成论证在可置信度、一致性、简约性等语言和认识论维度上的质量。虽然PEIRCE是一个通用框架，但本文展示了它在其自然语言解释生成领域的应用能力——一个固有地需要同时具备物质适当性和形式正确性的环境。 

---
# Lifting Factor Graphs with Some Unknown Factors for New Individuals 

**Title (ZH)**: 带有部分未知因素的提升因素图模型：为新个体建模 

**Authors**: Malte Luttermann, Ralf Möller, Marcel Gehrke  

**Link**: [PDF](https://arxiv.org/pdf/2504.04089)  

**Abstract**: Lifting exploits symmetries in probabilistic graphical models by using a representative for indistinguishable objects, allowing to carry out query answering more efficiently while maintaining exact answers. In this paper, we investigate how lifting enables us to perform probabilistic inference for factor graphs containing unknown factors, i.e., factors whose underlying function of potential mappings is unknown. We present the Lifting Factor Graphs with Some Unknown Factors (LIFAGU) algorithm to identify indistinguishable subgraphs in a factor graph containing unknown factors, thereby enabling the transfer of known potentials to unknown potentials to ensure a well-defined semantics of the model and allow for (lifted) probabilistic inference. We further extend LIFAGU to incorporate additional background knowledge about groups of factors belonging to the same individual object. By incorporating such background knowledge, LIFAGU is able to further reduce the ambiguity of possible transfers of known potentials to unknown potentials. 

**Abstract (ZH)**: 提升技术通过使用代表不可区分对象的符号来利用概率图形模型中的对称性，从而在保持精确答案的同时更高效地执行查询回答。在本文中，我们研究提升如何使我们能够对包含未知因素的因子图执行概率推理，即那些潜在映射底层函数未知的因素。我们提出了Lifting Factor Graphs with Some Unknown Factors (LIFAGU) 算法，以识别包含未知因素的因子图中的不可区分子图，从而将已知势转移到未知势，确保模型的语义明确，并允许进行（提升的）概率推理。此外，我们将LIFAGU扩展以整合关于属于同一个体对象的因素组的额外背景知识。通过整合这种背景知识，LIFAGU能够进一步减少已知势转移到未知势的可能转换的模糊性。 

---
# Towards An Efficient and Effective En Route Travel Time Estimation Framework 

**Title (ZH)**: 面向高效准确的在途旅行时间估计框架 

**Authors**: Zekai Shen, Haitao Yuan, Xiaowei Mao, Congkang Lv, Shengnan Guo, Youfang Lin, Huaiyu Wan  

**Link**: [PDF](https://arxiv.org/pdf/2504.04086)  

**Abstract**: En route travel time estimation (ER-TTE) focuses on predicting the travel time of the remaining route. Existing ER-TTE methods always make re-estimation which significantly hinders real-time performance, especially when faced with the computational demands of simultaneous user requests. This results in delays and reduced responsiveness in ER-TTE services. We propose a general efficient framework U-ERTTE combining an Uncertainty-Guided Decision mechanism (UGD) and Fine-Tuning with Meta-Learning (FTML) to address these challenges. UGD quantifies the uncertainty and provides confidence intervals for the entire route. It selectively re-estimates only when the actual travel time deviates from the predicted confidence intervals, thereby optimizing the efficiency of ER-TTE. To ensure the accuracy of confidence intervals and accurate predictions that need to re-estimate, FTML is employed to train the model, enabling it to learn general driving patterns and specific features to adapt to specific tasks. Extensive experiments on two large-scale real datasets demonstrate that the U-ERTTE framework significantly enhances inference speed and throughput while maintaining high effectiveness. Our code is available at this https URL 

**Abstract (ZH)**: 沿途旅行时间估计（ER-TTE）专注于预测剩余路线的旅行时间。现有的ER-TTE方法通常需要重新估计，这在面对同时用户请求的计算需求时显著影响了实时性能，导致ER-TTE服务出现延迟和响应性降低。我们提出了一种结合不确定性引导决策机制（UGD）和元学习细调（FTML）的一般高效框架U-ERTTE以应对这些挑战。UGD量化不确定性并为整个路线提供置信区间，在实际旅行时间偏离预测置信区间时才选择性地重新估计，从而优化ER-TTE的效率。为了确保置信区间的准确性和需要重新估计的准确预测，采用FTML对模型进行训练，使其能够学习通用的驾驶模式并适应特定任务的特定特征。在两个大规模真实数据集上的广泛实验表明，U-ERTTE框架显著提高了推理速度和 throughput，同时保持了高有效性。我们的代码可在以下网址获取：this https URL 

---
# Among Us: A Sandbox for Agentic Deception 

**Title (ZH)**: Among Us: 一种自主欺骗的沙箱环境 

**Authors**: Satvik Golechha, Adrià Garriga-Alonso  

**Link**: [PDF](https://arxiv.org/pdf/2504.04072)  

**Abstract**: Studying deception in AI agents is important and difficult due to the lack of model organisms and sandboxes that elicit the behavior without asking the model to act under specific conditions or inserting intentional backdoors. Extending upon $\textit{AmongAgents}$, a text-based social-deduction game environment, we aim to fix this by introducing Among Us as a rich sandbox where LLM-agents exhibit human-style deception naturally while they think, speak, and act with other agents or humans. We introduce Deception ELO as an unbounded measure of deceptive capability, suggesting that frontier models win more because they're better at deception, not at detecting it. We evaluate the effectiveness of AI safety techniques (LLM-monitoring of outputs, linear probes on various datasets, and sparse autoencoders) for detecting lying and deception in Among Us, and find that they generalize very well out-of-distribution. We open-source our sandbox as a benchmark for future alignment research and hope that this is a good testbed to improve safety techniques to detect and remove agentically-motivated deception, and to anticipate deceptive abilities in LLMs. 

**Abstract (ZH)**: 研究AI代理中的欺骗行为具有重要意义但难度较大，由于缺乏合适的模型生物或无需在特定条件下操作模型或植入有意后门即可引发欺骗行为的沙箱环境。在$\textit{AmongAgents}$文本社会推理游戏环境的基础上，我们通过引入《Among Us》作为丰富的沙箱环境，让语言模型代理自然地表现出类似人类的欺骗行为，在与其他代理或人类交互的过程中思考、说话和行动。我们提出了欺骗ELO作为衡量欺骗能力的无界指标，表明前沿模型取胜是因为它们更擅长欺骗，而不是检测欺骗。我们评估了多种AI安全性技术（如语言模型的输出监控、线性探针以及稀疏自编码器）在《Among Us》中检测欺骗的有效性，并发现这些技术在分布外具有很好的泛化能力。我们开源这一沙箱，作为未来对齐研究的基准，并希望这可以成为一个有效的测试平台，以提高检测和移除代理驱动欺骗的技术，并预见语言模型的欺骗能力。 

---
# ADAPT: Actively Discovering and Adapting to Preferences for any Task 

**Title (ZH)**: ADAPT: 主动发现和适应任何任务的偏好 

**Authors**: Maithili Patel, Xavier Puig, Ruta Desai, Roozbeh Mottaghi, Sonia Chernova, Joanne Truong, Akshara Rai  

**Link**: [PDF](https://arxiv.org/pdf/2504.04040)  

**Abstract**: Assistive agents should be able to perform under-specified long-horizon tasks while respecting user preferences. We introduce Actively Discovering and Adapting to Preferences for any Task (ADAPT) -- a benchmark designed to evaluate agents' ability to adhere to user preferences across various household tasks through active questioning. Next, we propose Reflection-DPO, a novel training approach for adapting large language models (LLMs) to the task of active questioning. Reflection-DPO finetunes a 'student' LLM to follow the actions of a privileged 'teacher' LLM, and optionally ask a question to gather necessary information to better predict the teacher action. We find that prior approaches that use state-of-the-art LLMs fail to sufficiently follow user preferences in ADAPT due to insufficient questioning and poor adherence to elicited preferences. In contrast, Reflection-DPO achieves a higher rate of satisfying user preferences, outperforming a zero-shot chain-of-thought baseline by 6.1% on unseen users. 

**Abstract (ZH)**: 辅助代理应该能够在尊重用户偏好的情况下执行未明确描述的长期任务。我们提出了Actively Discovering and Adapting to Preferences for any Task (ADAPT)——一个用于评估代理在通过主动询问的方式跨多种家庭任务中遵守用户偏好的能力的基准。接下来，我们提出了Reflection-DPO，这是一种用于使大规模语言模型（LLMs）适应主动询问任务的新型训练方法。Reflection-DPO 将一个“学生”LLM 细调为跟随一个特权“教师”LLM 的行动，并可选地提出问题以收集必要的信息以更好地预测教师行为。我们发现，使用最先进的LLM的先前方法在ADAPT中未能充分遵循用户偏好，原因是对用户偏好的询问不足以及对引出的偏好的不良遵守。相比之下，Reflection-DPO 实现了更高的用户偏好满足率，在未见过的用户中，其性能超越了零样本推理基线6.1%。 

---
# Optimizing UAV Aerial Base Station Flights Using DRL-based Proximal Policy Optimization 

**Title (ZH)**: 基于近端策略优化的深度 reinforcement 学习优化无人机高空基站飞行 

**Authors**: Mario Rico Ibanez, Azim Akhtarshenas, David Lopez-Perez, Giovanni Geraci  

**Link**: [PDF](https://arxiv.org/pdf/2504.03961)  

**Abstract**: Unmanned aerial vehicle (UAV)-based base stations offer a promising solution in emergencies where the rapid deployment of cutting-edge networks is crucial for maximizing life-saving potential. Optimizing the strategic positioning of these UAVs is essential for enhancing communication efficiency. This paper introduces an automated reinforcement learning approach that enables UAVs to dynamically interact with their environment and determine optimal configurations. By leveraging the radio signal sensing capabilities of communication networks, our method provides a more realistic perspective, utilizing state-of-the-art algorithm -- proximal policy optimization -- to learn and generalize positioning strategies across diverse user equipment (UE) movement patterns. We evaluate our approach across various UE mobility scenarios, including static, random, linear, circular, and mixed hotspot movements. The numerical results demonstrate the algorithm's adaptability and effectiveness in maintaining comprehensive coverage across all movement patterns. 

**Abstract (ZH)**: 基于无人机的基站（UAV基站）在紧急情况下提供了一种有希望的解决方案，因为快速部署尖端网络对于最大化生命救援潜力至关重要。优化这些无人机的战略性位置对于提高通信效率至关重要。本文介绍了一种自动强化学习方法，使无人机能够动态与环境交互并确定最优配置。借助通信网络的无线电信号感知能力，我们的方法提供了更现实的视角，并利用最新的算法——近端策略优化——来学习和泛化适用于不同用户设备（UE）运动模式的定位策略。我们在包括静态、随机、直线、圆周和混合热点移动在内的各种UE移动场景下评估了我们的方法。数值结果表明，该算法在所有运动模式下具有适应性和有效性，能够维持全面的覆盖范围。 

---
# Have Large Language Models Learned to Reason? A Characterization via 3-SAT Phase Transition 

**Title (ZH)**: 大型语言模型学会推理了吗？基于3-SAT相变的刻画 

**Authors**: Rishi Hazra, Gabriele Venturato, Pedro Zuidberg Dos Martires, Luc De Raedt  

**Link**: [PDF](https://arxiv.org/pdf/2504.03930)  

**Abstract**: Large Language Models (LLMs) have been touted as AI models possessing advanced reasoning abilities. In theory, autoregressive LLMs with Chain-of-Thought (CoT) can perform more serial computations to solve complex reasoning tasks. However, recent studies suggest that, despite this capacity, LLMs do not truly learn to reason but instead fit on statistical features. To study the reasoning capabilities in a principled fashion, we adopt a computational theory perspective and propose an experimental protocol centered on 3-SAT -- the prototypical NP-complete problem lying at the core of logical reasoning and constraint satisfaction tasks. Specifically, we examine the phase transitions in random 3-SAT and characterize the reasoning abilities of state-of-the-art LLMs by varying the inherent hardness of the problem instances. By comparing DeepSeek R1 with other LLMs, our findings reveal two key insights (1) LLM accuracy drops significantly on harder instances, suggesting all current models struggle when statistical shortcuts are unavailable (2) Unlike other LLMs, R1 shows signs of having learned the underlying reasoning. Following a principled experimental protocol, our study moves beyond the benchmark-driven evidence often found in LLM reasoning research. Our findings highlight important gaps and suggest clear directions for future research. 

**Abstract (ZH)**: 大型语言模型（LLMs）被认为是具有高级推理能力的AI模型。理论上，具有链式思考（CoT）的自回归LLMs能够进行更多的串行计算以解决复杂的推理任务。然而，近期研究表明，尽管具备这种能力，LLMs实际上并未真正学习推理，而是依赖于统计特征拟合。为客观研究推理能力，我们采用计算理论视角，提出以3-SAT为核心问题的实验协议，研究最先进LLMs的推理能力。我们通过改变问题实例的固有难度，考察随机3-SAT的相变，并揭示了两个关键发现：(1) LLMs在更难的问题实例上的准确性显著下降，表明在缺乏统计捷径时所有现有模型都难以应对；(2) 与其它LLMs不同，R1显示出学习到潜在推理结构的迹象。遵循严格实验协议，我们的研究超越了当前LLM推理研究中常见的基准驱动证据，揭示了重要的差距并为未来研究指明了方向。 

---
# Improving World Models using Deep Supervision with Linear Probes 

**Title (ZH)**: 使用线性探针的深度监督改进世界模型 

**Authors**: Andrii Zahorodnii  

**Link**: [PDF](https://arxiv.org/pdf/2504.03861)  

**Abstract**: Developing effective world models is crucial for creating artificial agents that can reason about and navigate complex environments. In this paper, we investigate a deep supervision technique for encouraging the development of a world model in a network trained end-to-end to predict the next observation. While deep supervision has been widely applied for task-specific learning, our focus is on improving the world models. Using an experimental environment based on the Flappy Bird game, where the agent receives only LIDAR measurements as observations, we explore the effect of adding a linear probe component to the network's loss function. This additional term encourages the network to encode a subset of the true underlying world features into its hidden state. Our experiments demonstrate that this supervision technique improves both training and test performance, enhances training stability, and results in more easily decodable world features -- even for those world features which were not included in the training. Furthermore, we observe a reduced distribution drift in networks trained with the linear probe, particularly during high-variability phases of the game (flying between successive pipe encounters). Including the world features loss component roughly corresponded to doubling the model size, suggesting that the linear probe technique is particularly beneficial in compute-limited settings or when aiming to achieve the best performance with smaller models. These findings contribute to our understanding of how to develop more robust and sophisticated world models in artificial agents, paving the way for further advancements in this field. 

**Abstract (ZH)**: 开发有效的世界模型对于创建能够推理和导航复杂环境的人工代理至关重要。在本文中，我们研究了一种深层次监督技术，以促使网络在端到端训练中预测下一个观察值的同时发展世界模型。虽然深层监督在任务特定学习中广泛应用，但我们专注于提高世界模型的质量。通过基于Flappy Bird游戏的实验环境，其中代理仅接收LIDAR测量作为观察值，我们探讨了将线性探针组件添加到网络损失函数中的效果。这个额外的项鼓励网络将其隐藏状态编码为真实底层世界特征的一部分。我们的实验表明，这种监督技术不仅提高了训练和测试性能，还增强了训练稳定性，并导致更易于解码的世界特征——即使这些特征未包含在训练中。此外，我们观察到，在使用线性探针训练的网络中，特别是在游戏中的高变异阶段（在连续管道相遇之间飞行），网络之间的分布漂移减少。包括世界特征损失项大致相当于将模型大小翻倍，表明线性探针技术在计算资源受限的环境中尤其有益，或者在追求小型模型最佳性能时尤为有益。这些发现增加了我们对如何在人工代理中发展更稳健和复杂的世界模型的理解，为进一步推动该领域的发展奠定了基础。 

---
# Hierarchically Encapsulated Representation for Protocol Design in Self-Driving Labs 

**Title (ZH)**: 自驾驶实验室中协议设计的层次封装表示方法 

**Authors**: Yu-Zhe Shi, Mingchen Liu, Fanxu Meng, Qiao Xu, Zhangqian Bi, Kun He, Lecheng Ruan, Qining Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03810)  

**Abstract**: Self-driving laboratories have begun to replace human experimenters in performing single experimental skills or predetermined experimental protocols. However, as the pace of idea iteration in scientific research has been intensified by Artificial Intelligence, the demand for rapid design of new protocols for new discoveries become evident. Efforts to automate protocol design have been initiated, but the capabilities of knowledge-based machine designers, such as Large Language Models, have not been fully elicited, probably for the absence of a systematic representation of experimental knowledge, as opposed to isolated, flatten pieces of information. To tackle this issue, we propose a multi-faceted, multi-scale representation, where instance actions, generalized operations, and product flow models are hierarchically encapsulated using Domain-Specific Languages. We further develop a data-driven algorithm based on non-parametric modeling that autonomously customizes these representations for specific domains. The proposed representation is equipped with various machine designers to manage protocol design tasks, including planning, modification, and adjustment. The results demonstrate that the proposed method could effectively complement Large Language Models in the protocol design process, serving as an auxiliary module in the realm of machine-assisted scientific exploration. 

**Abstract (ZH)**: 自我驾驶实验室已经开始替代人类实验员执行单一实验技能或预定的实验协议。然而，随着人工智能加速了科学研究中的思想迭代，对快速设计新的实验协议以进行新发现的需求变得明显。已经开始尝试自动化协议设计，但知识为基础的机器设计者，如大型语言模型的能力尚未被充分激发，这可能是由于缺乏系统性的实验知识表示，与孤立扁平的信息片段相对。为了解决这一问题，我们提出了一种多层次、多尺度的表示方法，其中实例动作、通用操作和产品流模型通过领域特定语言进行层次封装。进一步开发了一种基于非参数建模的驱动数据算法，以自主地为特定领域定制这些表示方法。所提出的表示方法配备了各种机器设计者来管理协议设计任务，包括规划、修改和调整。实验结果表明，所提出的方法可以有效地补充大型语言模型在协议设计过程中的作用，作为机器辅助科学探索领域的一个辅助模块。 

---
# Flow State: Humans Enabling AI Systems to Program Themselves 

**Title (ZH)**: 自编程状态：人类赋能AI系统自动编程 

**Authors**: Helena Zhang, Jakobi Haskell, Yosef Frost  

**Link**: [PDF](https://arxiv.org/pdf/2504.03771)  

**Abstract**: Compound AI systems, orchestrating multiple AI components and external APIs, are increasingly vital but face challenges in managing complexity, handling ambiguity, and enabling effective development workflows. Existing frameworks often introduce significant overhead, implicit complexity, or restrictive abstractions, hindering maintainability and iterative refinement, especially in Human-AI collaborative settings. We argue that overcoming these hurdles requires a foundational architecture prioritizing structural clarity and explicit control. To this end, we introduce Pocketflow, a platform centered on Human-AI co-design, enabled by Pocketflow. Pocketflow is a Python framework built upon a deliberately minimal yet synergistic set of core abstractions: modular Nodes with a strict lifecycle, declarative Flow orchestration, native hierarchical nesting (Flow-as-Node), and explicit action-based conditional logic. This unique combination provides a robust, vendor-agnostic foundation with very little code that demonstrably reduces overhead while offering the expressiveness needed for complex patterns like agentic workflows and RAG. Complemented by Pocket AI, an assistant leveraging this structure for system design, Pocketflow provides an effective environment for iteratively prototyping, refining, and deploying the adaptable, scalable AI systems demanded by modern enterprises. 

**Abstract (ZH)**: 基于-pocketflow的人机联合设计平台：一种促进复杂人机智能系统开发的基础架构 

---
# A Benchmark for Scalable Oversight Protocols 

**Title (ZH)**: 可扩展监督协议基准 

**Authors**: Abhimanyu Pallavi Sudhir, Jackson Kaunismaa, Arjun Panickssery  

**Link**: [PDF](https://arxiv.org/pdf/2504.03731)  

**Abstract**: As AI agents surpass human capabilities, scalable oversight -- the problem of effectively supplying human feedback to potentially superhuman AI models -- becomes increasingly critical to ensure alignment. While numerous scalable oversight protocols have been proposed, they lack a systematic empirical framework to evaluate and compare them. While recent works have tried to empirically study scalable oversight protocols -- particularly Debate -- we argue that the experiments they conduct are not generalizable to other protocols. We introduce the scalable oversight benchmark, a principled framework for evaluating human feedback mechanisms based on our agent score difference (ASD) metric, a measure of how effectively a mechanism advantages truth-telling over deception. We supply a Python package to facilitate rapid and competitive evaluation of scalable oversight protocols on our benchmark, and conduct a demonstrative experiment benchmarking Debate. 

**Abstract (ZH)**: 随着AI代理超越人类能力，可扩展监督——有效向可能超人类的AI模型提供人类反馈的问题——变得越来越关键，以确保一致性和对齐。尽管提出了许多可扩展监督协议，但缺乏系统性的实证框架来评估和比较它们。尽管最近有些研究尝试从实证角度研究可扩展监督协议（特别是 Debate），我们认为它们开展的实验不具备泛化到其他协议的能力。我们引入了可扩展监督基准，这是一种基于我们代理得分差（ASD）度量的严格框架，用于评估人类反馈机制，该度量衡量机制如何有效地使诚实胜过欺骗。我们提供了一个Python包，以促进在基准上快速且具有竞争力地评估可扩展监督协议，并进行了一个示范性实验来基准测试Debate。 

---
# A Scalable Predictive Modelling Approach to Identifying Duplicate Adverse Event Reports for Drugs and Vaccines 

**Title (ZH)**: 一种可扩展的预测建模方法，用于识别药物和疫苗的重复不良事件报告 

**Authors**: Jim W. Barrett, Nils Erlanson, Joana Félix China, G. Niklas Norén  

**Link**: [PDF](https://arxiv.org/pdf/2504.03729)  

**Abstract**: The practice of pharmacovigilance relies on large databases of individual case safety reports to detect and evaluate potential new causal associations between medicines or vaccines and adverse events. Duplicate reports are separate and unlinked reports referring to the same case of an adverse event involving a specific patient at a certain time. They impede statistical analysis and mislead clinical assessment. The large size of such databases precludes a manual identification of duplicates, and so a computational method must be employed. This paper builds upon a hitherto state of the art model, vigiMatch, modifying existing features and introducing new ones to target known shortcomings of the original model. Two support vector machine classifiers, one for medicines and one for vaccines, classify report pairs as duplicates and non-duplicates. Recall was measured using a diverse collection of 5 independent labelled test sets. Precision was measured by having each model classify a randomly selected stream of pairs of reports until each model classified 100 pairs as duplicates. These pairs were assessed by a medical doctor without indicating which method(s) had flagged each pair. Performance on individual countries was measured by having a medical doctor assess a subset of pairs classified as duplicates for three different countries. The new model achieved higher precision and higher recall for all labelled datasets compared to the previous state of the art model, with comparable performance for medicines and vaccines. The model was shown to produce substantially fewer false positives than the comparator model on pairs from individual countries. The method presented here advances state of the art for duplicate detection in adverse event reports for medicines and vaccines. 

**Abstract (ZH)**: 药物警戒实践依赖于大量个例安全报告数据库，以检测和评估药物或疫苗与不良事件之间潜在的新因果关联。重复报告是指涉及同一患者在同一时间发生的同一不良事件的独立而不连接的报告。它们阻碍了统计分析并误导了临床评估。由于这类数据库规模庞大，无法通过手工方法识别重复报告，因此必须使用计算方法。本文在此前最先进的模型vigiMatch的基础上改进现有特征并引入新特征以针对原始模型已知的不足之处进行改进。两个支持向量机分类器分别用于药物和疫苗，将报告对分类为重复和非重复。召回率通过使用5个独立标记的测试集进行测量。精确率通过让每个模型分类随机选择的一系列报告对，直到每个模型分类出100对作为重复报告，并由医学医生进行评估，未指出哪些方法标记了每对报告。通过医学医生评估来自三个不同国家的重复报告的子集对每个国家进行了性能测量。新模型在所有标记数据集上的精确率和召回率均高于以前的最先进的模型，并且在药物和疫苗上表现出相似的性能。该模型在来自个别国家的报告对中产生的假阳性显著少于比较模型。本方法推进了药物和疫苗不良事件报告重复检测的最先进的状态。 

---
# TransNet: Transfer Knowledge for Few-shot Knowledge Graph Completion 

**Title (ZH)**: TransNet：迁移知识以实现少样本知识图谱完成 

**Authors**: Lihui Liu, Zihao Wang, Dawei Zhou, Ruijie Wang, Yuchen Yan, Bo Xiong, Sihong He, Kai Shu, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2504.03720)  

**Abstract**: Knowledge graphs (KGs) are ubiquitous and widely used in various applications. However, most real-world knowledge graphs are incomplete, which significantly degrades their performance on downstream tasks. Additionally, the relationships in real-world knowledge graphs often follow a long-tail distribution, meaning that most relations are represented by only a few training triplets. To address these challenges, few-shot learning has been introduced. Few-shot KG completion aims to make accurate predictions for triplets involving novel relations when only a limited number of training triplets are available. Although many methods have been proposed, they typically learn each relation individually, overlooking the correlations between different tasks and the relevant information in previously trained tasks. In this paper, we propose a transfer learning-based few-shot KG completion method (TransNet). By learning the relationships between different tasks, TransNet effectively transfers knowledge from similar tasks to improve the current task's performance. Furthermore, by employing meta-learning, TransNet can generalize effectively to new, unseen relations. Extensive experiments on benchmark datasets demonstrate the superiority of TransNet over state-of-the-art methods. Code can be found at this https URL 

**Abstract (ZH)**: 知识图谱（KGs）在各种应用中无处不在且被广泛使用。然而，大多数现实中的知识图谱是不完整的，这显著降低了其在下游任务上的性能。此外，现实中的知识图谱中的关系往往遵循长尾分布，意味着大多数关系由少量训练三元组表示。为了解决这些挑战，引入了少样本学习。少样本知识图谱补全旨在在仅有有限数量训练三元组的情况下，对涉及新颖关系的三元组进行准确预测。虽然已经提出了许多方法，但它们通常独立学习每个关系，忽略了不同任务之间的相关性和先前训练任务中的相关信息。在本文中，我们提出了一种基于迁移学习的少样本知识图谱补全方法（TransNet）。通过学习不同任务之间的关系，TransNet有效地将相似任务的知识转移到当前任务，以提高其性能。此外，通过采用元学习，TransNet能够有效泛化到新的未见过的关系。在基准数据集上的广泛实验表明，TransNet在性能上优于最先进的方法。代码可以在以下链接找到：this https URL。 

---
# Reinforcing Clinical Decision Support through Multi-Agent Systems and Ethical AI Governance 

**Title (ZH)**: 通过多智能体系统和伦理AI治理强化临床决策支持 

**Authors**: Ying-Jung Chen, Chi-Sheng Chen, Ahmad Albarqawi  

**Link**: [PDF](https://arxiv.org/pdf/2504.03699)  

**Abstract**: In the age of data-driven medicine, it is paramount to include explainable and ethically managed artificial intelligence in explaining clinical decision support systems to achieve trustworthy and effective patient care. The focus of this paper is on a new architecture of a multi-agent system for clinical decision support that uses modular agents to analyze laboratory results, vital signs, and the clinical context and then integrates these results to drive predictions and validate outcomes. We describe our implementation with the eICU database to run lab-analysis-specific agents, vitals-only interpreters, and contextual reasoners and then run the prediction module and a validation agent. Everything is a transparent implementation of business logic, influenced by the principles of ethical AI governance such as Autonomy, Fairness, and Accountability. It provides visible results that this agent-based framework not only improves on interpretability and accuracy but also on reinforcing trust in AI-assisted decisions in an intensive care setting. 

**Abstract (ZH)**: 在数据驱动医学时代，将可解释且伦理管理的人工智能纳入临床决策支持系统的解释中，以实现可信赖且有效的患者护理至关重要。本文的重点是一种新的多智能体系统架构，该架构使用模块化智能体分析实验室数据、生命体征和临床背景，然后将这些结果整合以驱动预测和验证结果。我们使用eICU数据库实现了针对实验室分析的智能体、仅关注生命体征的解释器以及上下文推理器，并运行预测模块和验证智能体。所有这些都遵循了伦理人工智能治理原则（自主性、公平性和问责制）的透明实现，提供了清晰的结果，证明了基于代理的框架不仅提高了可解释性和准确性，还增强了在重症护理环境中对人工智能辅助决策的信任。 

---
# Diagnostic Method for Hydropower Plant Condition-based Maintenance combining Autoencoder with Clustering Algorithms 

**Title (ZH)**: 基于自编码器与聚类算法的水力发电厂状态维修诊断方法 

**Authors**: Samy Jad, Xavier Desforges, Pierre-Yves Villard, Christian Caussidéry, Kamal Medjaher  

**Link**: [PDF](https://arxiv.org/pdf/2504.03649)  

**Abstract**: The French company EDF uses supervisory control and data acquisition systems in conjunction with a data management platform to monitor hydropower plant, allowing engineers and technicians to analyse the time-series collected. Depending on the strategic importance of the monitored hydropower plant, the number of time-series collected can vary greatly making it difficult to generate valuable information from the extracted data. In an attempt to provide an answer to this particular problem, a condition detection and diagnosis method combining clustering algorithms and autoencoder neural networks for pattern recognition has been developed and is presented in this paper. First, a dimension reduction algorithm is used to create a 2-or 3-dimensional projection that allows the users to identify unsuspected relationships between datapoints. Then, a collection of clustering algorithms regroups the datapoints into clusters. For each identified cluster, an autoencoder neural network is trained on the corresponding dataset. The aim is to measure the reconstruction error between each autoencoder model and the measured values, thus creating a proximity index for each state discovered during the clustering stage. 

**Abstract (ZH)**: 法国公司EDF利用 supervisory control and data acquisition 系统结合数据管理平台来监控水电站，使得工程师和技术人员能够分析收集的时间序列数据。根据监控水电站的战略重要性，收集的时间序列数据数量可能差异很大，这使得从提取的数据中生成有价值的信息变得困难。为解决这一特定问题，本文开发并介绍了结合聚类算法和自动编码神经网络的条件检测与诊断方法。该方法首先使用降维算法创建2维或3维投影，使用户能够识别数据点之间的未预见关系；然后，使用一系列聚类算法将数据点分组到不同的簇中。对于每个识别出的簇，针对相应的数据集训练一个自动编码神经网络。目标是测量每个自动编码模型与测量值之间的重构误差，从而为在聚类阶段发现的每种状态生成一个接近指数。 

---
# URECA: Unique Region Caption Anything 

**Title (ZH)**: URECA: 唯一区域Anything描述 

**Authors**: Sangbeom Lim, Junwan Kim, Heeji Yoon, Jaewoo Jung, Seungryong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.05305)  

**Abstract**: Region-level captioning aims to generate natural language descriptions for specific image regions while highlighting their distinguishing features. However, existing methods struggle to produce unique captions across multi-granularity, limiting their real-world applicability. To address the need for detailed region-level understanding, we introduce URECA dataset, a large-scale dataset tailored for multi-granularity region captioning. Unlike prior datasets that focus primarily on salient objects, URECA dataset ensures a unique and consistent mapping between regions and captions by incorporating a diverse set of objects, parts, and background elements. Central to this is a stage-wise data curation pipeline, where each stage incrementally refines region selection and caption generation. By leveraging Multimodal Large Language Models (MLLMs) at each stage, our pipeline produces distinctive and contextually grounded captions with improved accuracy and semantic diversity. Building upon this dataset, we present URECA, a novel captioning model designed to effectively encode multi-granularity regions. URECA maintains essential spatial properties such as position and shape through simple yet impactful modifications to existing MLLMs, enabling fine-grained and semantically rich region descriptions. Our approach introduces dynamic mask modeling and a high-resolution mask encoder to enhance caption uniqueness. Experiments show that URECA achieves state-of-the-art performance on URECA dataset and generalizes well to existing region-level captioning benchmarks. 

**Abstract (ZH)**: 多粒度区域描述旨在为特定图像区域生成自然语言描述，并突出其独特的特征。然而，现有方法在多粒度上的描述独特性较差，限制了其实际应用。为应对多粒度区域理解的需求，我们引入了URECA数据集，这是一个专为多粒度区域描述设计的大规模数据集。与其他主要关注显著对象的 datasets 不同，URECA数据集通过引入多种物体、部分和背景元素确保了区域与描述之间的独特且一致的映射关系。该数据集的核心在于分阶段的数据策展流程，每阶段逐步细化区域选择和描述生成。通过每个阶段利用多模态大型语言模型（MLLMs），我们的策展流程能够生成更具独特性和语境相关性的描述，提高了描述的准确性和语义多样性。基于这个数据集，我们提出了URECA，这是一种新型的描述模型，旨在有效地编码多粒度区域。URECA通过简单而有效的现有MLLMs修改，保留了关键的空间特性（如位置和形状），使其能够生成细腻且语义丰富的区域描述。我们的方法引入了动态掩码建模和高分辨率掩码编码器，以增强描述的独特性。实验结果显示，URECA在URECA数据集上达到了最先进的性能，并在现有的区域描述基准上表现出良好的泛化能力。 

---
# Dion: A Communication-Efficient Optimizer for Large Models 

**Title (ZH)**: Dion：一种高效的大模型通信优化器 

**Authors**: Kwangjun Ahn, Byron Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05295)  

**Abstract**: Training large AI models efficiently requires distributing computation across multiple accelerators, but this often incurs significant communication overhead -- especially during gradient synchronization. We introduce Dion, a communication-efficient optimizer that retains the synchronous semantics of standard distributed training (e.g., DDP, FSDP) while substantially reducing I/O costs. Unlike conventional optimizers that synchronize full gradient matrices, Dion leverages orthonormalized updates with device-local momentum buffers, eliminating the need for full gradient exchange. It further supports an efficient sharding strategy that avoids reconstructing large matrices during training. 

**Abstract (ZH)**: Dion：一种高效通信的优化器 

---
# Learning to Reason Over Time: Timeline Self-Reflection for Improved Temporal Reasoning in Language Models 

**Title (ZH)**: 学习随时间进行推理：时间线自我反思以提高语言模型的时间推理能力 

**Authors**: Adrián Bazaga, Rexhina Blloshmi, Bill Byrne, Adrià de Gispert  

**Link**: [PDF](https://arxiv.org/pdf/2504.05258)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for generating coherent text, understanding context, and performing reasoning tasks. However, they struggle with temporal reasoning, which requires processing time-related information such as event sequencing, durations, and inter-temporal relationships. These capabilities are critical for applications including question answering, scheduling, and historical analysis. In this paper, we introduce TISER, a novel framework that enhances the temporal reasoning abilities of LLMs through a multi-stage process that combines timeline construction with iterative self-reflection. Our approach leverages test-time scaling to extend the length of reasoning traces, enabling models to capture complex temporal dependencies more effectively. This strategy not only boosts reasoning accuracy but also improves the traceability of the inference process. Experimental results demonstrate state-of-the-art performance across multiple benchmarks, including out-of-distribution test sets, and reveal that TISER enables smaller open-source models to surpass larger closed-weight models on challenging temporal reasoning tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经成为生成连贯文本、理解上下文和执行推理任务的强大工具。然而，它们在处理时间推理方面存在困难，这需要处理与事件顺序、持续时间和跨时间关系相关的时间信息。这些能力对于应用领域包括问答、排程和历史分析至关重要。在本文中，我们提出了一种名为TISER的新框架，通过结合时间线构建和迭代自我反思的多阶段过程来增强LLMs的时间推理能力。我们的方法利用测试时扩展缩放来延长推理痕迹的长度，从而使模型更有效地捕捉复杂的时序依赖性。这种策略不仅提高了推理准确性，还提高了推理过程的可追溯性。实验结果表明，TISER在多个基准测试中展现出最先进的性能，包括通用测试集，并揭示了TISER能让较小的开源模型在具有挑战性的时间推理任务中超越较大的封闭权重模型。 

---
# Adversarial KA 

**Title (ZH)**: 对抗性KA 

**Authors**: Sviatoslav Dzhenzher, Michael H. Freedman  

**Link**: [PDF](https://arxiv.org/pdf/2504.05255)  

**Abstract**: Regarding the representation theorem of Kolmogorov and Arnold (KA) as an algorithm for representing or «expressing» functions, we test its robustness by analyzing its ability to withstand adversarial attacks. We find KA to be robust to countable collections of continuous adversaries, but unearth a question about the equi-continuity of the outer functions that, so far, obstructs taking limits and defeating continuous groups of adversaries. This question on the regularity of the outer functions is relevant to the debate over the applicability of KA to the general theory of NNs. 

**Abstract (ZH)**: 关于Kolmogorov和Arnold (KA)表示定理作为表示或“表达”函数的算法的研究，我们通过分析其抵抗对抗攻击的能力来检验其鲁棒性。我们发现KA对可数集合的连续对抗是鲁棒的，但发现外函数的等连续性问题至今仍阻碍了极限的取法和对抗连续敌对群体。外函数的正则性问题与KA在一般神经网络理论中的适用性辩论有关。 

---
# Explaining Low Perception Model Competency with High-Competency Counterfactuals 

**Title (ZH)**: 用高能力反事实解释低感知模型能力 

**Authors**: Sara Pohland, Claire Tomlin  

**Link**: [PDF](https://arxiv.org/pdf/2504.05254)  

**Abstract**: There exist many methods to explain how an image classification model generates its decision, but very little work has explored methods to explain why a classifier might lack confidence in its prediction. As there are various reasons the classifier might lose confidence, it would be valuable for this model to not only indicate its level of uncertainty but also explain why it is uncertain. Counterfactual images have been used to visualize changes that could be made to an image to generate a different classification decision. In this work, we explore the use of counterfactuals to offer an explanation for low model competency--a generalized form of predictive uncertainty that measures confidence. Toward this end, we develop five novel methods to generate high-competency counterfactual images, namely Image Gradient Descent (IGD), Feature Gradient Descent (FGD), Autoencoder Reconstruction (Reco), Latent Gradient Descent (LGD), and Latent Nearest Neighbors (LNN). We evaluate these methods across two unique datasets containing images with six known causes for low model competency and find Reco, LGD, and LNN to be the most promising methods for counterfactual generation. We further evaluate how these three methods can be utilized by pre-trained Multimodal Large Language Models (MLLMs) to generate language explanations for low model competency. We find that the inclusion of a counterfactual image in the language model query greatly increases the ability of the model to generate an accurate explanation for the cause of low model competency, thus demonstrating the utility of counterfactual images in explaining low perception model competency. 

**Abstract (ZH)**: 存在许多方法可以解释图像分类模型如何生成其决策，但很少有研究探索解释分类器为何对其预测缺乏信心的方法。由于分类器丧失信心的原因可能多种多样，因此对于该模型不仅表示其不确定性水平，还能解释其不确定性的原因来说，这将是非常有价值的。反事实图像已被用于可视化为生成不同分类决策而对图像进行的可能更改。在本文中，我们研究了使用反事实图像来解释低模型能力——一种衡量信心的通用形式预测不确定性——的方法。为此，我们开发了五种新的方法来生成高能力的反事实图像，即图像梯度下降（IGD）、特征梯度下降（FGD）、自动编码器重构（Reco）、潜在梯度下降（LGD）和潜在最近邻（LNN）。我们在包含六个已知低模型能力原因的两个独特的数据集中评估了这些方法，并发现Reco、LGD和LNN是最有希望的反事实生成方法。进一步地，我们研究了这些三种方法如何被预训练的多模态大型语言模型（MLLMs）用于生成低模型能力的语言解释。我们发现，将反事实图像包含在语言模型查询中极大地提高了模型生成准确解释的能力，以解释低模型能力的原因，从而证明了反事实图像在解释低感知模型能力方面的实用性。 

---
# PINNverse: Accurate parameter estimation in differential equations from noisy data with constrained physics-informed neural networks 

**Title (ZH)**: PINN逆问题：基于约束物理神经网络从噪声数据中进行精确参数估计 

**Authors**: Marius Almanstötter, Roman Vetter, Dagmar Iber  

**Link**: [PDF](https://arxiv.org/pdf/2504.05248)  

**Abstract**: Parameter estimation for differential equations from measured data is an inverse problem prevalent across quantitative sciences. Physics-Informed Neural Networks (PINNs) have emerged as effective tools for solving such problems, especially with sparse measurements and incomplete system information. However, PINNs face convergence issues, stability problems, overfitting, and complex loss function design. Here we introduce PINNverse, a training paradigm that addresses these limitations by reformulating the learning process as a constrained differential optimization problem. This approach achieves a dynamic balance between data loss and differential equation residual loss during training while preventing overfitting. PINNverse combines the advantages of PINNs with the Modified Differential Method of Multipliers to enable convergence on any point on the Pareto front. We demonstrate robust and accurate parameter estimation from noisy data in four classical ODE and PDE models from physics and biology. Our method enables accurate parameter inference also when the forward problem is expensive to solve. 

**Abstract (ZH)**: 从测量数据中估计微分方程的参数是定量科学中常见的逆问题。基于物理的神经网络（PINNs）已成为解决这类问题的有效工具，尤其是在稀疏测量和不完整系统信息的情况下。然而，PINNs 面临收敛性问题、稳定性问题、过拟合以及复杂的损失函数设计。在此我们介绍 PINNverse，一种通过将学习过程重新表述为约束微分优化问题来解决这些限制的训练范式。此方法在训练过程中实现了数据损失和微分方程残差损失之间的动态平衡，同时防止过拟合。PINNverse 结合了 PINNs 与修改后的差分乘子法的优势，能够在帕累托前沿上的任何点实现收敛。我们在物理学和生物学中的四个经典 ODE 和 PDE 模型中展示了在噪声数据下 robust 和准确的参数估计。我们的方法能够在正向问题求解昂贵的情况下实现准确的参数推断。 

---
# Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG 

**Title (ZH)**: 利用大语言模型进行功效导向的标注：减少检索和 Retrieval-Augmented Generation（检索增强生成）中的手动努力 

**Authors**: Hengran Zhang, Minghao Tang, Keping Bi, Jiafeng Guo, Shihao Liu, Daiting Shi, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05220)  

**Abstract**: Retrieval models typically rely on costly human-labeled query-document relevance annotations for training and evaluation. To reduce this cost and leverage the potential of Large Language Models (LLMs) in relevance judgments, we aim to explore whether LLM-generated annotations can effectively replace human annotations in training retrieval models. Retrieval usually emphasizes relevance, which indicates "topic-relatedness" of a document to a query, while in RAG, the value of a document (or utility) depends on how it contributes to answer generation. Recognizing this mismatch, some researchers use LLM performance on downstream tasks with documents as labels, but this approach requires manual answers for specific tasks, leading to high costs and limited generalization. In another line of work, prompting LLMs to select useful documents as RAG references eliminates the need for human annotation and is not task-specific. If we leverage LLMs' utility judgments to annotate retrieval data, we may retain cross-task generalization without human annotation in large-scale corpora. Therefore, we investigate utility-focused annotation via LLMs for large-scale retriever training data across both in-domain and out-of-domain settings on the retrieval and RAG tasks. To reduce the impact of low-quality positives labeled by LLMs, we design a novel loss function, i.e., Disj-InfoNCE. Our experiments reveal that: (1) Retrievers trained on utility-focused annotations significantly outperform those trained on human annotations in the out-of-domain setting on both tasks, demonstrating superior generalization capabilities. (2) LLM annotation does not replace human annotation in the in-domain setting. However, incorporating just 20% human-annotated data enables retrievers trained with utility-focused annotations to match the performance of models trained entirely with human annotations. 

**Abstract (ZH)**: 基于大规模语言模型的检索数据实用 Annotations: 探索利用大规模语言模型进行检索模型训练的有效性 

---
# Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling 

**Title (ZH)**: 利用查询似然模型释放大语言模型在密集检索中的潜力 

**Authors**: Hengran Zhang, Keping Bi, Jiafeng Guo, Xiaojie Sun, Shihao Liu, Daiting Shi, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05216)  

**Abstract**: Dense retrieval is a crucial task in Information Retrieval (IR) and is the foundation for downstream tasks such as re-ranking. Recently, large language models (LLMs) have shown compelling semantic understanding capabilities and are appealing to researchers studying dense retrieval. LLMs, as decoder-style generative models, are competent at language generation while falling short on modeling global information due to the lack of attention to tokens afterward. Inspired by the classical word-based language modeling approach for IR, i.e., the query likelihood (QL) model, we seek to sufficiently utilize LLMs' generative ability by QL maximization. However, instead of ranking documents with QL estimation, we introduce an auxiliary task of QL maximization to yield a better backbone for contrastively learning a discriminative retriever. We name our model as LLM-QL. To condense global document semantics to a single vector during QL modeling, LLM-QL has two major components, Attention Stop (AS) and Input Corruption (IC). AS stops the attention of predictive tokens to previous tokens until the ending token of the document. IC masks a portion of tokens in the input documents during prediction. Experiments on MSMARCO show that LLM-QL can achieve significantly better performance than other LLM-based retrievers and using QL estimated by LLM-QL for ranking outperforms word-based QL by a large margin. 

**Abstract (ZH)**: 基于密集检索的大语言模型-查询似然性方法（LLM-QL） 

---
# A moving target in AI-assisted decision-making: Dataset shift, model updating, and the problem of update opacity 

**Title (ZH)**: AI辅助决策中的流动性目标：数据集转移、模型更新及其透明度问题 

**Authors**: Joshua Hatherley  

**Link**: [PDF](https://arxiv.org/pdf/2504.05210)  

**Abstract**: Machine learning (ML) systems are vulnerable to performance decline over time due to dataset shift. To address this problem, experts often suggest that ML systems should be regularly updated to ensure ongoing performance stability. Some scholarly literature has begun to address the epistemic and ethical challenges associated with different updating methodologies. Thus far, however, little attention has been paid to the impact of model updating on the ML-assisted decision-making process itself, particularly in the AI ethics and AI epistemology literatures. This article aims to address this gap in the literature. It argues that model updating introduces a new sub-type of opacity into ML-assisted decision-making -- update opacity -- that occurs when users cannot understand how or why an update has changed the reasoning or behaviour of an ML system. This type of opacity presents a variety of distinctive epistemic and safety concerns that available solutions to the black box problem in ML are largely ill-equipped to address. A variety of alternative strategies may be developed or pursued to address the problem of update opacity more directly, including bi-factual explanations, dynamic model reporting, and update compatibility. However, each of these strategies presents its own risks or carries significant limitations. Further research will be needed to address the epistemic and safety concerns associated with model updating and update opacity going forward. 

**Abstract (ZH)**: 机器学习系统因数据集偏移而导致性能下降的问题及其更新对决策透明度的影响：一项研究空白及对策探索 

---
# Correcting Class Imbalances with Self-Training for Improved Universal Lesion Detection and Tagging 

**Title (ZH)**: 使用自我训练纠正类别不平衡以改善通用病变检测和标注 

**Authors**: Alexander Shieh, Tejas Sudharshan Mathai, Jianfei Liu, Angshuman Paul, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2504.05207)  

**Abstract**: Universal lesion detection and tagging (ULDT) in CT studies is critical for tumor burden assessment and tracking the progression of lesion status (growth/shrinkage) over time. However, a lack of fully annotated data hinders the development of effective ULDT approaches. Prior work used the DeepLesion dataset (4,427 patients, 10,594 studies, 32,120 CT slices, 32,735 lesions, 8 body part labels) for algorithmic development, but this dataset is not completely annotated and contains class imbalances. To address these issues, in this work, we developed a self-training pipeline for ULDT. A VFNet model was trained on a limited 11.5\% subset of DeepLesion (bounding boxes + tags) to detect and classify lesions in CT studies. Then, it identified and incorporated novel lesion candidates from a larger unseen data subset into its training set, and self-trained itself over multiple rounds. Multiple self-training experiments were conducted with different threshold policies to select predicted lesions with higher quality and cover the class imbalances. We discovered that direct self-training improved the sensitivities of over-represented lesion classes at the expense of under-represented classes. However, upsampling the lesions mined during self-training along with a variable threshold policy yielded a 6.5\% increase in sensitivity at 4 FP in contrast to self-training without class balancing (72\% vs 78.5\%) and a 11.7\% increase compared to the same self-training policy without upsampling (66.8\% vs 78.5\%). Furthermore, we show that our results either improved or maintained the sensitivity at 4FP for all 8 lesion classes. 

**Abstract (ZH)**: 全CT研究中病变检测与标记（ULDT）对于肿瘤负担评估及追踪病变状态（增殖/缩小）随时间的变化至关重要。然而，缺乏完全注释的数据阻碍了有效ULDT方法的发展。先前的工作使用DeepLesion数据集（4427名患者，10594个研究，32120个CT切片，32735个病变，32个身体部分标签）进行算法开发，但该数据集并未完全注释且包含类别不平衡。为解决这些问题，我们在本文中开发了一种自训练管道用于ULDT。VFNet模型在DeepLesion的有限子集（11.5%，包含边界框和标签）上进行训练，以检测和分类CT研究中的病变。然后，它识别并从较大的未见数据子集中整合新的病变候选者到其训练集，并经过多轮自训练。进行了多次自训练实验，采用不同的阈值策略以选择高质量的预测病变并覆盖类别不平衡。我们发现，直接自训练提升了过代表病变类别的敏感性，但降低了欠代表类别的敏感性。然而，通过在自训练过程中增加病变样本量并采用可变阈值策略，与不进行类别平衡的自训练相比，敏感性提高了6.5%（对比72%和78.5%），与未上采样的相同自训练策略相比提高了11.7%（对比66.8%和78.5%）。此外，我们的结果显示，对于所有8种病变类别，这些结果要么提高了敏感性，要么保持了敏感性，在4FP时。 

---
# 3D Universal Lesion Detection and Tagging in CT with Self-Training 

**Title (ZH)**: 3D自训练CT病变检测与标记 

**Authors**: Jared Frazier, Tejas Sudharshan Mathai, Jianfei Liu, Angshuman Paul, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2504.05201)  

**Abstract**: Radiologists routinely perform the tedious task of lesion localization, classification, and size measurement in computed tomography (CT) studies. Universal lesion detection and tagging (ULDT) can simultaneously help alleviate the cumbersome nature of lesion measurement and enable tumor burden assessment. Previous ULDT approaches utilize the publicly available DeepLesion dataset, however it does not provide the full volumetric (3D) extent of lesions and also displays a severe class imbalance. In this work, we propose a self-training pipeline to detect 3D lesions and tag them according to the body part they occur in. We used a significantly limited 30\% subset of DeepLesion to train a VFNet model for 2D lesion detection and tagging. Next, the 2D lesion context was expanded into 3D, and the mined 3D lesion proposals were integrated back into the baseline training data in order to retrain the model over multiple rounds. Through the self-training procedure, our VFNet model learned from its own predictions, detected lesions in 3D, and tagged them. Our results indicated that our VFNet model achieved an average sensitivity of 46.9\% at [0.125:8] false positives (FP) with a limited 30\% data subset in comparison to the 46.8\% of an existing approach that used the entire DeepLesion dataset. To our knowledge, we are the first to jointly detect lesions in 3D and tag them according to the body part label. 

**Abstract (ZH)**: Radiologists 常规地在计算机断层扫描（CT）研究中执行病变定位、分类和尺寸测量的任务。通用病变检测和标记（ULDT）可以同时减轻病变测量的繁琐性质，使肿瘤负担评估成为可能。之前的 ULDT 方法利用了公开的 DeepLesion 数据集，但该数据集并未提供病变的完整体积（3D）范围，并且严重存在类别不平衡问题。在本工作中，我们提出了一种自训练管道来检测 3D 病变并根据其发生的部位对其进行标记。我们使用 DeepLesion 的显著局限的 30% 子集来训练一个 VFNet 模型进行 2D 病变检测和标记。接下来，2D 病变上下文扩展到 3D，并提取的 3D 病变建议被重新集成到基础训练数据中，以便在多轮迭代中重新训练模型。通过自训练过程，我们的 VFNet 模型从自己的预测中学习，在有限的 30% 数据子集中实现了在 [0.125:8] 假阳性（FP）下的平均灵敏度为 46.9%，与使用整个 DeepLesion 数据集的现有方法相比为 46.8%。据我们所知，我们是第一个联合检测 3D 病变并根据解剖部位标签对其进行标记的研究工作。 

---
# Universal Lymph Node Detection in Multiparametric MRI with Selective Augmentation 

**Title (ZH)**: 基于选择性增强的多参数MRI通用淋巴结检测 

**Authors**: Tejas Sudharshan Mathai, Sungwon Lee, Thomas C. Shen, Zhiyong Lu, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2504.05196)  

**Abstract**: Robust localization of lymph nodes (LNs) in multiparametric MRI (mpMRI) is critical for the assessment of lymphadenopathy. Radiologists routinely measure the size of LN to distinguish benign from malignant nodes, which would require subsequent cancer staging. Sizing is a cumbersome task compounded by the diverse appearances of LNs in mpMRI, which renders their measurement difficult. Furthermore, smaller and potentially metastatic LNs could be missed during a busy clinical day. To alleviate these imaging and workflow problems, we propose a pipeline to universally detect both benign and metastatic nodes in the body for their ensuing measurement. The recently proposed VFNet neural network was employed to identify LN in T2 fat suppressed and diffusion weighted imaging (DWI) sequences acquired by various scanners with a variety of exam protocols. We also use a selective augmentation technique known as Intra-Label LISA (ILL) to diversify the input data samples the model sees during training, such that it improves its robustness during the evaluation phase. We achieved a sensitivity of $\sim$83\% with ILL vs. $\sim$80\% without ILL at 4 FP/vol. Compared with current LN detection approaches evaluated on mpMRI, we show a sensitivity improvement of $\sim$9\% at 4 FP/vol. 

**Abstract (ZH)**: 在多参数MRI (mpMRI) 中稳健定位淋巴结 (LNs) 对淋巴腺病的评估至关重要。我们提出了一种管道来普遍检测体内良性及转移性淋巴结，以便随后进行测量。我们采用了最近提出的VFNet神经网络，该网络能够在不同扫描器和多种检查协议下识别T2脂肪抑制和扩散加权成像 (DWI) 序列中的淋巴结。此外，我们还使用了一种名为Intra-Label LISA (ILL) 的选择性增强技术，在模型训练过程中多样化其输入数据样本，从而提高其评估阶段的稳健性。与当前的淋巴结检测方法在mpMRI上的评估相比，我们在每升4个假阳性下的敏感性提高了约9%。 

---
# Resource-Efficient Beam Prediction in mmWave Communications with Multimodal Realistic Simulation Framework 

**Title (ZH)**: 基于多模态现实仿真框架的毫米波通信高效波束预测 

**Authors**: Yu Min Park, Yan Kyaw Tun, Walid Saad, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2504.05187)  

**Abstract**: Beamforming is a key technology in millimeter-wave (mmWave) communications that improves signal transmission by optimizing directionality and intensity. However, conventional channel estimation methods, such as pilot signals or beam sweeping, often fail to adapt to rapidly changing communication environments. To address this limitation, multimodal sensing-aided beam prediction has gained significant attention, using various sensing data from devices such as LiDAR, radar, GPS, and RGB images to predict user locations or network conditions. Despite its promising potential, the adoption of multimodal sensing-aided beam prediction is hindered by high computational complexity, high costs, and limited datasets. Thus, in this paper, a resource-efficient learning approach is proposed to transfer knowledge from a multimodal network to a monomodal (radar-only) network based on cross-modal relational knowledge distillation (CRKD), while reducing computational overhead and preserving predictive accuracy. To enable multimodal learning with realistic data, a novel multimodal simulation framework is developed while integrating sensor data generated from the autonomous driving simulator CARLA with MATLAB-based mmWave channel modeling, and reflecting real-world conditions. The proposed CRKD achieves its objective by distilling relational information across different feature spaces, which enhances beam prediction performance without relying on expensive sensor data. Simulation results demonstrate that CRKD efficiently distills multimodal knowledge, allowing a radar-only model to achieve $94.62\%$ of the teacher performance. In particular, this is achieved with just $10\%$ of the teacher network's parameters, thereby significantly reducing computational complexity and dependence on multimodal sensor data. 

**Abstract (ZH)**: 基于跨模态关系知识蒸馏的资源高效学习方法：从多模态网络到雷达唯一网络的迁移学习 

---
# Lightweight and Direct Document Relevance Optimization for Generative Information Retrieval 

**Title (ZH)**: 轻量级和直接的文档相关性优化方法用于生成式信息检索 

**Authors**: Kidist Amde Mekonnen, Yubao Tang, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2504.05181)  

**Abstract**: Generative information retrieval (GenIR) is a promising neural retrieval paradigm that formulates document retrieval as a document identifier (docid) generation task, allowing for end-to-end optimization toward a unified global retrieval objective. However, existing GenIR models suffer from token-level misalignment, where models trained to predict the next token often fail to capture document-level relevance effectively. While reinforcement learning-based methods, such as reinforcement learning from relevance feedback (RLRF), aim to address this misalignment through reward modeling, they introduce significant complexity, requiring the optimization of an auxiliary reward function followed by reinforcement fine-tuning, which is computationally expensive and often unstable. To address these challenges, we propose direct document relevance optimization (DDRO), which aligns token-level docid generation with document-level relevance estimation through direct optimization via pairwise ranking, eliminating the need for explicit reward modeling and reinforcement learning. Experimental results on benchmark datasets, including MS MARCO document and Natural Questions, show that DDRO outperforms reinforcement learning-based methods, achieving a 7.4% improvement in MRR@10 for MS MARCO and a 19.9% improvement for Natural Questions. These findings highlight DDRO's potential to enhance retrieval effectiveness with a simplified optimization approach. By framing alignment as a direct optimization problem, DDRO simplifies the ranking optimization pipeline of GenIR models while offering a viable alternative to reinforcement learning-based methods. 

**Abstract (ZH)**: 直接文档相关性优化（DDRO）：通过直接优化对齐 token 级别文档标识符生成与文档级别相关性估计 

---
# BRIDGES: Bridging Graph Modality and Large Language Models within EDA Tasks 

**Title (ZH)**: BRIDGES: 跨接图模态与大型语言模型在EDA任务中的桥梁 

**Authors**: Wei Li, Yang Zou, Christopher Ellis, Ruben Purdy, Shawn Blanton, José M. F. Moura  

**Link**: [PDF](https://arxiv.org/pdf/2504.05180)  

**Abstract**: While many EDA tasks already involve graph-based data, existing LLMs in EDA primarily either represent graphs as sequential text, or simply ignore graph-structured data that might be beneficial like dataflow graphs of RTL code. Recent studies have found that LLM performance suffers when graphs are represented as sequential text, and using additional graph information significantly boosts performance. To address these challenges, we introduce BRIDGES, a framework designed to incorporate graph modality into LLMs for EDA tasks. BRIDGES integrates an automated data generation workflow, a solution that combines graph modality with LLM, and a comprehensive evaluation suite. First, we establish an LLM-driven workflow to generate RTL and netlist-level data, converting them into dataflow and netlist graphs with function descriptions. This workflow yields a large-scale dataset comprising over 500,000 graph instances and more than 1.5 billion tokens. Second, we propose a lightweight cross-modal projector that encodes graph representations into text-compatible prompts, enabling LLMs to effectively utilize graph data without architectural modifications. Experimental results demonstrate 2x to 10x improvements across multiple tasks compared to text-only baselines, including accuracy in design retrieval, type prediction and perplexity in function description, with negligible computational overhead (<1% model weights increase and <30% additional runtime overhead). Even without additional LLM finetuning, our results outperform text-only by a large margin. We plan to release BRIDGES, including the dataset, models, and training flow. 

**Abstract (ZH)**: BRIDGES：将图模态纳入EDA任务的LLM框架 

---
# Attention-Based Multi-Scale Temporal Fusion Network for Uncertain-Mode Fault Diagnosis in Multimode Processes 

**Title (ZH)**: 基于注意力机制的多尺度时间融合网络在多模式过程中不确定模式故障诊断 

**Authors**: Guangqiang Li, M. Amine Atoui, Xiangshun Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.05172)  

**Abstract**: Fault diagnosis in multimode processes plays a critical role in ensuring the safe operation of industrial systems across multiple modes. It faces a great challenge yet to be addressed - that is, the significant distributional differences among monitoring data from multiple modes make it difficult for the models to extract shared feature representations related to system health conditions. In response to this problem, this paper introduces a novel method called attention-based multi-scale temporal fusion network. The multi-scale depthwise convolution and gated recurrent unit are employed to extract multi-scale contextual local features and long-short-term features. A temporal attention mechanism is designed to focus on critical time points with higher cross-mode shared information, thereby enhancing the accuracy of fault diagnosis. The proposed model is applied to Tennessee Eastman process dataset and three-phase flow facility dataset. The experiments demonstrate that the proposed model achieves superior diagnostic performance and maintains a small model size. 

**Abstract (ZH)**: 多模态过程的故障诊断在确保工业系统在多模式下安全运行中起着关键作用。面对监测数据在多模式下显著的分布差异带来的挑战，本论文介绍了一种基于注意力机制的多尺度时间融合网络方法。该方法利用多尺度深度卷积和门控递归单元提取多尺度上下文局部特征和长期短期特征，并设计了时间注意力机制来关注具有更高跨模式共享信息的关键时间点，从而提高故障诊断的准确性。所提出的模型在田纳西-Eastman 过程数据集和三相流设施数据集上进行了应用。实验结果显示，所提模型在诊断性能上表现优越且保持了较小的模型规模。 

---
# SSLFusion: Scale & Space Aligned Latent Fusion Model for Multimodal 3D Object Detection 

**Title (ZH)**: SSLFusion：面向多模态3D物体检测的尺度与空间对齐潜在融合模型 

**Authors**: Bonan Ding, Jin Xie, Jing Nie, Jiale Cao  

**Link**: [PDF](https://arxiv.org/pdf/2504.05170)  

**Abstract**: Multimodal 3D object detection based on deep neural networks has indeed made significant progress. However, it still faces challenges due to the misalignment of scale and spatial information between features extracted from 2D images and those derived from 3D point clouds. Existing methods usually aggregate multimodal features at a single stage. However, leveraging multi-stage cross-modal features is crucial for detecting objects of various scales. Therefore, these methods often struggle to integrate features across different scales and modalities effectively, thereby restricting the accuracy of detection. Additionally, the time-consuming Query-Key-Value-based (QKV-based) cross-attention operations often utilized in existing methods aid in reasoning the location and existence of objects by capturing non-local contexts. However, this approach tends to increase computational complexity. To address these challenges, we present SSLFusion, a novel Scale & Space Aligned Latent Fusion Model, consisting of a scale-aligned fusion strategy (SAF), a 3D-to-2D space alignment module (SAM), and a latent cross-modal fusion module (LFM). SAF mitigates scale misalignment between modalities by aggregating features from both images and point clouds across multiple levels. SAM is designed to reduce the inter-modal gap between features from images and point clouds by incorporating 3D coordinate information into 2D image features. Additionally, LFM captures cross-modal non-local contexts in the latent space without utilizing the QKV-based attention operations, thus mitigating computational complexity. Experiments on the KITTI and DENSE datasets demonstrate that our SSLFusion outperforms state-of-the-art methods. Our approach obtains an absolute gain of 2.15% in 3D AP, compared with the state-of-art method GraphAlign on the moderate level of the KITTI test set. 

**Abstract (ZH)**: 基于深度神经网络的多模态3D目标检测已取得显著进展，但由于从2D图像中提取的特征与从3D点云中提取的特征在尺度和空间信息上的不对齐，仍面临挑战。现有方法通常在单个阶段聚合多模态特征。然而，利用多阶段跨模态特征对于检测不同尺度的目标至关重要。因此，这些方法往往难以有效地跨尺度和模态整合特征，从而限制了检测的准确性。此外，现有方法常用耗时的Query-Key-Value（QKV）基跨注意力操作捕获非局部上下文，推理目标的位置和存在，但这种方法增加了计算复杂度。为了解决这些挑战，我们提出了SSLFusion，一种新颖的尺度与空间对齐潜在融合模型，包括尺度对齐融合策略（SAF）、3D到2D空间对齐模块（SAM）和潜在跨模态融合模块（LFM）。SAF 通过在多个级别上从图像和点云中聚合特征来缓解模态之间的尺度不一致性。SAM 通过将3D坐标信息融入2D图像特征，旨在减少图像和点云特征之间的跨模态间隙。此外，LFM 在潜在空间中捕获跨模态的非局部上下文，而不使用QKV基注意力操作，从而减轻计算复杂度。在Kitti和Dense数据集上的实验表明，我们的SSLFusion 超过了现有方法。在KITTI测试集的中等水平上，我们的方法在3D AP上相对于现有方法GraphAlign 的绝对增益为2.15%。 

---
# RLBayes: a Bayesian Network Structure Learning Algorithm via Reinforcement Learning-Based Search Strategy 

**Title (ZH)**: RLBayes：基于强化学习搜索策略的贝叶斯网络结构学习算法 

**Authors**: Mingcan Wang, Junchang Xin, Luxuan Qu, Qi Chen, Zhiqiong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05167)  

**Abstract**: The score-based structure learning of Bayesian network (BN) is an effective way to learn BN models, which are regarded as some of the most compelling probabilistic graphical models in the field of representation and reasoning under uncertainty. However, the search space of structure learning grows super-exponentially as the number of variables increases, which makes BN structure learning an NP-hard problem, as well as a combination optimization problem (COP). Despite the successes of many heuristic methods on it, the results of the structure learning of BN are usually unsatisfactory. Inspired by Q-learning, in this paper, a Bayesian network structure learning algorithm via reinforcement learning-based (RL-based) search strategy is proposed, namely RLBayes. The method borrows the idea of RL and tends to record and guide the learning process by a dynamically maintained Q-table. By creating and maintaining the dynamic Q-table, RLBayes achieve storing the unlimited search space within limited space, thereby achieving the structure learning of BN via Q-learning. Not only is it theoretically proved that RLBayes can converge to the global optimal BN structure, but also it is experimentally proved that RLBayes has a better effect than almost all other heuristic search algorithms. 

**Abstract (ZH)**: 基于强化学习的贝叶斯网络结构学习算法（RLBayes） 

---
# Leveraging Label Potential for Enhanced Multimodal Emotion Recognition 

**Title (ZH)**: 利用标签潜力以增强多模态情感识别 

**Authors**: Xuechun Shao, Yinfeng Yu, Liejun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05158)  

**Abstract**: Multimodal emotion recognition (MER) seeks to integrate various modalities to predict emotional states accurately. However, most current research focuses solely on the fusion of audio and text features, overlooking the valuable information in emotion labels. This oversight could potentially hinder the performance of existing methods, as emotion labels harbor rich, insightful information that could significantly aid MER. We introduce a novel model called Label Signal-Guided Multimodal Emotion Recognition (LSGMER) to overcome this limitation. This model aims to fully harness the power of emotion label information to boost the classification accuracy and stability of MER. Specifically, LSGMER employs a Label Signal Enhancement module that optimizes the representation of modality features by interacting with audio and text features through label embeddings, enabling it to capture the nuances of emotions precisely. Furthermore, we propose a Joint Objective Optimization(JOO) approach to enhance classification accuracy by introducing the Attribution-Prediction Consistency Constraint (APC), which strengthens the alignment between fused features and emotion categories. Extensive experiments conducted on the IEMOCAP and MELD datasets have demonstrated the effectiveness of our proposed LSGMER model. 

**Abstract (ZH)**: 多模态情绪识别中的标签信号引导多模态情绪识别（Label Signal-Guided Multimodal Emotion Recognition） 

---
# A Reinforcement Learning Method for Environments with Stochastic Variables: Post-Decision Proximal Policy Optimization with Dual Critic Networks 

**Title (ZH)**: 具有随机变量环境的强化学习方法：带双critic网络的后决策近端策略优化 

**Authors**: Leonardo Kanashiro Felizardo, Edoardo Fadda, Paolo Brandimarte, Emilio Del-Moral-Hernandez, Mariá Cristina Vasconcelos Nascimento  

**Link**: [PDF](https://arxiv.org/pdf/2504.05150)  

**Abstract**: This paper presents Post-Decision Proximal Policy Optimization (PDPPO), a novel variation of the leading deep reinforcement learning method, Proximal Policy Optimization (PPO). The PDPPO state transition process is divided into two steps: a deterministic step resulting in the post-decision state and a stochastic step leading to the next state. Our approach incorporates post-decision states and dual critics to reduce the problem's dimensionality and enhance the accuracy of value function estimation. Lot-sizing is a mixed integer programming problem for which we exemplify such dynamics. The objective of lot-sizing is to optimize production, delivery fulfillment, and inventory levels in uncertain demand and cost parameters. This paper evaluates the performance of PDPPO across various environments and configurations. Notably, PDPPO with a dual critic architecture achieves nearly double the maximum reward of vanilla PPO in specific scenarios, requiring fewer episode iterations and demonstrating faster and more consistent learning across different initializations. On average, PDPPO outperforms PPO in environments with a stochastic component in the state transition. These results support the benefits of using a post-decision state. Integrating this post-decision state in the value function approximation leads to more informed and efficient learning in high-dimensional and stochastic environments. 

**Abstract (ZH)**: 基于决策后的proximal策略优化（PDPPO） 

---
# EffOWT: Transfer Visual Language Models to Open-World Tracking Efficiently and Effectively 

**Title (ZH)**: EffOWT: 有效地高效转移视觉语言模型到开放世界跟踪 

**Authors**: Bingyang Wang, Kaer Huang, Bin Li, Yiqiang Yan, Lihe Zhang, Huchuan Lu, You He  

**Link**: [PDF](https://arxiv.org/pdf/2504.05141)  

**Abstract**: Open-World Tracking (OWT) aims to track every object of any category, which requires the model to have strong generalization capabilities. Trackers can improve their generalization ability by leveraging Visual Language Models (VLMs). However, challenges arise with the fine-tuning strategies when VLMs are transferred to OWT: full fine-tuning results in excessive parameter and memory costs, while the zero-shot strategy leads to sub-optimal performance. To solve the problem, EffOWT is proposed for efficiently transferring VLMs to OWT. Specifically, we build a small and independent learnable side network outside the VLM backbone. By freezing the backbone and only executing backpropagation on the side network, the model's efficiency requirements can be met. In addition, EffOWT enhances the side network by proposing a hybrid structure of Transformer and CNN to improve the model's performance in the OWT field. Finally, we implement sparse interactions on the MLP, thus reducing parameter updates and memory costs significantly. Thanks to the proposed methods, EffOWT achieves an absolute gain of 5.5% on the tracking metric OWTA for unknown categories, while only updating 1.3% of the parameters compared to full fine-tuning, with a 36.4% memory saving. Other metrics also demonstrate obvious improvement. 

**Abstract (ZH)**: Efficient Transfer of Visual Language Models to Open-World Tracking 

---
# Interpretable Style Takagi-Sugeno-Kang Fuzzy Clustering 

**Title (ZH)**: 可解释风格 Takagi-Sugeno-Kang 模糊聚类 

**Authors**: Suhang Gu, Ye Wang, Yongxin Chou, Jinliang Cong, Mingli Lu, Zhuqing Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.05125)  

**Abstract**: Clustering is an efficient and essential technique for exploring latent knowledge of data. However, limited attention has been given to the interpretability of the clusters detected by most clustering algorithms. In addition, due to the homogeneity of data, different groups of data have their own homogeneous styles. In this paper, the above two aspects are considered, and an interpretable style Takagi-Sugeno-Kang (TSK) fuzzy clustering (IS-TSK-FC) algorithm is proposed. The clustering behavior of IS-TSK-FC is fully guided by the TSK fuzzy inference on fuzzy rules. In particular, samples are grouped into clusters represented by the corresponding consequent vectors of all fuzzy rules learned in an unsupervised manner. This can explain how the clusters are generated in detail, thus making the underlying decision-making process of the IS-TSK-FC interpretable. Moreover, a series of style matrices are introduced to facilitate the consequents of fuzzy rules in IS-TSK-FC by capturing the styles of clusters as well as the nuances between different styles. Consequently, all the fuzzy rules in IS-TSK-FC have powerful data representation capability. After determining the antecedents of all the fuzzy rules, the optimization problem of IS-TSK-FC can be iteratively solved in an alternation manner. The effectiveness of IS-TSK-FC as an interpretable clustering tool is validated through extensive experiments on benchmark datasets with unknown implicit/explicit styles. Specially, the superior clustering performance of IS-TSK-FC is demonstrated on case studies where different groups of data present explicit styles. The source code of IS-TSK-FC can be downloaded from this https URL. 

**Abstract (ZH)**: 具解释性的样式Takagi-Sugeno-Kang模糊聚类算法（IS-TSK-FC） 

---
# Balancing Robustness and Efficiency in Embedded DNNs Through Activation Function Selection 

**Title (ZH)**: 通过激活函数选择在嵌入式DNN中平衡稳健性和效率 

**Authors**: Jon Gutiérrez Zaballa, Koldo Basterretxea, Javier Echanobe  

**Link**: [PDF](https://arxiv.org/pdf/2504.05119)  

**Abstract**: Machine learning-based embedded systems for safety-critical applications, such as aerospace and autonomous driving, must be robust to perturbations caused by soft errors. As transistor geometries shrink and voltages decrease, modern electronic devices become more susceptible to background radiation, increasing the concern about failures produced by soft errors. The resilience of deep neural networks (DNNs) to these errors depends not only on target device technology but also on model structure and the numerical representation and arithmetic precision of their parameters. Compression techniques like pruning and quantization, used to reduce memory footprint and computational complexity, alter both model structure and representation, affecting soft error robustness. In this regard, although often overlooked, the choice of activation functions (AFs) impacts not only accuracy and trainability but also compressibility and error resilience. This paper explores the use of bounded AFs to enhance robustness against parameter perturbations, while evaluating their effects on model accuracy, compressibility, and computational load with a technology-agnostic approach. We focus on encoder-decoder convolutional models developed for semantic segmentation of hyperspectral images with application to autonomous driving systems. Experiments are conducted on an AMD-Xilinx's KV260 SoM. 

**Abstract (ZH)**: 基于机器学习的嵌入式系统在航空航天和自主驾驶等安全关键应用中，必须对由软错误引起的扰动具有鲁棒性。随着晶体管几何尺寸缩小和电压降低，现代电子设备对背景辐射的敏感性增加，软错误导致的故障担忧也随之增加。深度神经网络（DNNs）对这些错误的鲁棒性不仅取决于目标设备技术，还取决于模型结构及其参数的数值表示和算术精度。用于减少内存占用和计算复杂性的剪枝和量化等压缩技术会同时改变模型结构和表示，影响软错误的鲁棒性。在这方面，尽管常常被忽视，激活函数（AFs）的选择不仅影响准确性和可训练性，还影响可压缩性和错误鲁棒性。本文探讨使用有界激活函数以增强对参数扰动的鲁棒性，并以技术无关的方法评估其对模型准确度、可压缩性和计算负载的影响。我们专注于应用于自主驾驶系统的高光谱图像语义分割的编码器-解码器卷积模型。实验在AMD-Xilinx的KV260系统模块上进行。 

---
# SpeakEasy: Enhancing Text-to-Speech Interactions for Expressive Content Creation 

**Title (ZH)**: SpeakEasy: 提升表达性内容创作的文本到语音交互 

**Authors**: Stephen Brade, Sam Anderson, Rithesh Kumar, Zeyu Jin, Anh Truong  

**Link**: [PDF](https://arxiv.org/pdf/2504.05106)  

**Abstract**: Novice content creators often invest significant time recording expressive speech for social media videos. While recent advancements in text-to-speech (TTS) technology can generate highly realistic speech in various languages and accents, many struggle with unintuitive or overly granular TTS interfaces. We propose simplifying TTS generation by allowing users to specify high-level context alongside their script. Our Wizard-of-Oz system, SpeakEasy, leverages user-provided context to inform and influence TTS output, enabling iterative refinement with high-level feedback. This approach was informed by two 8-subject formative studies: one examining content creators' experiences with TTS, and the other drawing on effective strategies from voice actors. Our evaluation shows that participants using SpeakEasy were more successful in generating performances matching their personal standards, without requiring significantly more effort than leading industry interfaces. 

**Abstract (ZH)**: 初级内容创作者经常花费大量时间录制用于社交媒体视频的富有表现力的语音。虽然最近在文本到语音（TTS）技术方面的进步可以在多种语言和口音下生成高度真实的语音，但许多用户仍然难以使用直观或过于琐碎的TTS界面。我们提出了一种简化TTS生成的方法，允许用户在其脚本中指定高层次的上下文。我们的Wizard-of-Oz系统SpeakEasy利用用户提供的上下文来指导和影响TTS输出，从而使用户能够通过高层次反馈进行迭代改进。该方法受到两项包含8名受试者的形成性研究的启发：一项研究探讨了内容创作者使用TTS的经验，另一项研究借鉴了语音演员的有效策略。我们的评估结果显示，使用SpeakEasy的参与者在其生成的表现更符合个人标准方面更加成功，而无需比领先行业界面投入更多的努力。 

---
# Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models 

**Title (ZH)**: 揭示对齐的大规模语言模型固有的伦理漏洞 

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau  

**Link**: [PDF](https://arxiv.org/pdf/2504.05050)  

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities. 

**Abstract (ZH)**: 大型语言模型的伦理脆弱性：对齐方法仅实现局部“安全区域”，预训练知识通过高概率对抗轨迹保持全局连接，导致在分布转移下受到 adversarial 诱导时重新 surfacing。 

---
# Graph-based Diffusion Model for Collaborative Filtering 

**Title (ZH)**: 基于图的扩散模型在协作过滤中的应用 

**Authors**: Xuan Zhang, Xiang Deng, Hongxing Yuan, Chunyu Wei, Yushun Fan  

**Link**: [PDF](https://arxiv.org/pdf/2504.05029)  

**Abstract**: Recently, diffusion-based recommendation methods have achieved impressive results. However, existing approaches predominantly treat each user's historical interactions as independent training samples, overlooking the potential of higher-order collaborative signals between users and items. Such signals, which encapsulate richer and more nuanced relationships, can be naturally captured using graph-based data structures. To address this limitation, we extend diffusion-based recommendation methods to the graph domain by directly modeling user-item bipartite graphs with diffusion models. This enables better modeling of the higher-order connectivity inherent in complex interaction dynamics. However, this extension introduces two primary challenges: (1) Noise Heterogeneity, where interactions are influenced by various forms of continuous and discrete noise, and (2) Relation Explosion, referring to the high computational costs of processing large-scale graphs. To tackle these challenges, we propose a Graph-based Diffusion Model for Collaborative Filtering (GDMCF). To address noise heterogeneity, we introduce a multi-level noise corruption mechanism that integrates both continuous and discrete noise, effectively simulating real-world interaction complexities. To mitigate relation explosion, we design a user-active guided diffusion process that selectively focuses on the most meaningful edges and active users, reducing inference costs while preserving the graph's topological integrity. Extensive experiments on three benchmark datasets demonstrate that GDMCF consistently outperforms state-of-the-art methods, highlighting its effectiveness in capturing higher-order collaborative signals and improving recommendation performance. 

**Abstract (ZH)**: 基于图的扩散推荐模型（GDMCF）：捕捉高级协作信号以提高推荐性能 

---
# Batch Aggregation: An Approach to Enhance Text Classification with Correlated Augmented Data 

**Title (ZH)**: 批量聚合：一种通过相关增量数据增强文本分类的方法 

**Authors**: Charco Hui, Yalu Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.05020)  

**Abstract**: Natural language processing models often face challenges due to limited labeled data, especially in domain specific areas, e.g., clinical trials. To overcome this, text augmentation techniques are commonly used to increases sample size by transforming the original input data into artificial ones with the label preserved. However, traditional text classification methods ignores the relationship between augmented texts and treats them as independent samples which may introduce classification error. Therefore, we propose a novel approach called 'Batch Aggregation' (BAGG) which explicitly models the dependence of text inputs generated through augmentation by incorporating an additional layer that aggregates results from correlated texts. Through studying multiple benchmark data sets across different domains, we found that BAGG can improve classification accuracy. We also found that the increase of performance with BAGG is more obvious in domain specific data sets, with accuracy improvements of up to 10-29%. Through the analysis of benchmark data, the proposed method addresses limitations of traditional techniques and improves robustness in text classification tasks. Our result demonstrates that BAGG offers more robust results and outperforms traditional approaches when training data is limited. 

**Abstract (ZH)**: 自然语言处理模型由于标注数据有限，尤其是在临床试验等特定领域，常常面临挑战。为了克服这一问题，通常使用文本扩增技术通过将原始输入数据转换为带有标签保留的人工数据来增加样本量。然而，传统的文本分类方法忽略了扩增文本之间的关系，将其视为独立样本，这可能会引入分类错误。因此，我们提出了一种名为“批次聚合”（BAGG）的新方法，该方法通过引入一个聚合相关文本结果的额外层，明确建模通过扩增生成的文本输入之间的依赖性。通过在不同领域多个基准数据集上的研究，我们发现BAGG可以提高分类准确性。我们还发现，BAGG在特定领域数据集上的性能提升尤为明显，准确率提高了10-29%。通过对基准数据的分析，所提出的方法解决了传统技术的局限性，提高了文本分类任务的鲁棒性。我们的结果表明，在训练数据有限的情况下，BAGG提供了更稳健的结果并优于传统方法。 

---
# Measuring the right thing: justifying metrics in AI impact assessments 

**Title (ZH)**: 测量正确的事情：在AI影响评估中验证指标的有效性 

**Authors**: Stefan Buijsman, Herman Veluwenkamp  

**Link**: [PDF](https://arxiv.org/pdf/2504.05007)  

**Abstract**: AI Impact Assessments are only as good as the measures used to assess the impact of these systems. It is therefore paramount that we can justify our choice of metrics in these assessments, especially for difficult to quantify ethical and social values. We present a two-step approach to ensure metrics are properly motivated. First, a conception needs to be spelled out (e.g. Rawlsian fairness or fairness as solidarity) and then a metric can be fitted to that conception. Both steps require separate justifications, as conceptions can be judged on how well they fit with the function of, for example, fairness. We argue that conceptual engineering offers helpful tools for this step. Second, metrics need to be fitted to a conception. We illustrate this process through an examination of competing fairness metrics to illustrate that here the additional content that a conception offers helps us justify the choice for a specific metric. We thus advocate that impact assessments are not only clear on their metrics, but also on the conceptions that motivate those metrics. 

**Abstract (ZH)**: AI影响评估的效果取决于评估这些系统影响所使用的指标。因此，我们必须在这些评估中合理证明我们选择的指标，尤其是在难以量化伦理和社会价值的情况下尤为重要。我们提出了一种两步方法以确保指标能得到恰当的动机。首先，需要明确概念（例如罗尔斯公平或团结中的公平），然后可以将指标与该概念相匹配。两步都需要独立的证明，因为概念可以基于其与公平功能匹配的程度来评判。我们认为概念工程为这一步提供了一些有用的工具。第二，指标需要与概念相匹配。我们通过分析竞争的公平性指标来说明这一过程，表明概念提供的额外内容有助于我们证明选择特定指标的原因。因此，我们认为影响评估不仅要明确其指标，还需明确那些激励这些指标的概念。 

---
# SurvSurf: a partially monotonic neural network for first-hitting time prediction of intermittently observed discrete and continuous sequential events 

**Title (ZH)**: SurvSurf: 一种部分单调神经网络，用于间歇观察的离散和连续序列事件首次击中时间预测 

**Authors**: Yichen Kelly Chen, Sören Dittmer, Kinga Bernatowicz, Josep Arús-Pous, Kamen Bliznashki, John Aston, James H.F. Rudd, Carola-Bibiane Schönlieb, James Jones, Michael Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2504.04997)  

**Abstract**: We propose a neural-network based survival model (SurvSurf) specifically designed for direct and simultaneous probabilistic prediction of the first hitting time of sequential events from baseline. Unlike existing models, SurvSurf is theoretically guaranteed to never violate the monotonic relationship between the cumulative incidence functions of sequential events, while allowing nonlinear influence from predictors. It also incorporates implicit truths for unobserved intermediate events in model fitting, and supports both discrete and continuous time and events. We also identified a variant of the Integrated Brier Score (IBS) that showed robust correlation with the mean squared error (MSE) between the true and predicted probabilities by accounting for implied truths about the missing intermediate events. We demonstrated the superiority of SurvSurf compared to modern and traditional predictive survival models in two simulated datasets and two real-world datasets, using MSE, the more robust IBS and by measuring the extent of monotonicity violation. 

**Abstract (ZH)**: 基于神经网络的生存模型（SurvSurf）：直接和同时预测序列事件的首次到达时间的概率模型 

---
# Following the Whispers of Values: Unraveling Neural Mechanisms Behind Value-Oriented Behaviors in LLMs 

**Title (ZH)**: 遵循价值的 whispers：探讨面向价值行为的大型语言模型背后的神经机制 

**Authors**: Ling Hu, Yuemei Xu, Xiaoyang Gu, Letao Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.04994)  

**Abstract**: Despite the impressive performance of large language models (LLMs), they can present unintended biases and harmful behaviors driven by encoded values, emphasizing the urgent need to understand the value mechanisms behind them. However, current research primarily evaluates these values through external responses with a focus on AI safety, lacking interpretability and failing to assess social values in real-world contexts. In this paper, we propose a novel framework called ValueExploration, which aims to explore the behavior-driven mechanisms of National Social Values within LLMs at the neuron level. As a case study, we focus on Chinese Social Values and first construct C-voice, a large-scale bilingual benchmark for identifying and evaluating Chinese Social Values in LLMs. By leveraging C-voice, we then identify and locate the neurons responsible for encoding these values according to activation difference. Finally, by deactivating these neurons, we analyze shifts in model behavior, uncovering the internal mechanism by which values influence LLM decision-making. Extensive experiments on four representative LLMs validate the efficacy of our framework. The benchmark and code will be available. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）表现出色，但它们可能会表现出由编码价值观驱动的意外偏见和有害行为，强调了理解其背后的价值观机制的迫切需求。目前的研究主要通过外部反应评估这些价值观，集中在AI安全上，缺乏可解释性，未能在现实世界情境中评估社会价值观。本文提出了一种名为ValueExploration的新框架，旨在从神经元层面探索LLMs中国家社会价值观的行为驱动机制。作为案例研究，我们专注于中国社会价值观，并首先构建了一个大规模双语基准C-voice，用于识别和评估LLMs中的中国社会价值观。通过利用C-voice，我们根据激活差异识别并定位负责编码这些价值观的神经元。最后，通过抑制这些神经元，我们分析了模型行为的转变，揭示了价值观影响LLM决策的内部机制。对四个代表性LLM进行的 extensive 实验验证了我们框架的有效性。基准数据集和代码将可供使用。 

---
# RS-RAG: Bridging Remote Sensing Imagery and Comprehensive Knowledge with a Multi-Modal Dataset and Retrieval-Augmented Generation Model 

**Title (ZH)**: RS-RAG：多模态数据集与检索增强生成模型在遥感图像与综合知识融合中的应用 

**Authors**: Congcong Wen, Yiting Lin, Xiaokang Qu, Nan Li, Yong Liao, Hui Lin, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.04988)  

**Abstract**: Recent progress in VLMs has demonstrated impressive capabilities across a variety of tasks in the natural image domain. Motivated by these advancements, the remote sensing community has begun to adopt VLMs for remote sensing vision-language tasks, including scene understanding, image captioning, and visual question answering. However, existing remote sensing VLMs typically rely on closed-set scene understanding and focus on generic scene descriptions, yet lack the ability to incorporate external knowledge. This limitation hinders their capacity for semantic reasoning over complex or context-dependent queries that involve domain-specific or world knowledge. To address these challenges, we first introduced a multimodal Remote Sensing World Knowledge (RSWK) dataset, which comprises high-resolution satellite imagery and detailed textual descriptions for 14,141 well-known landmarks from 175 countries, integrating both remote sensing domain knowledge and broader world knowledge. Building upon this dataset, we proposed a novel Remote Sensing Retrieval-Augmented Generation (RS-RAG) framework, which consists of two key components. The Multi-Modal Knowledge Vector Database Construction module encodes remote sensing imagery and associated textual knowledge into a unified vector space. The Knowledge Retrieval and Response Generation module retrieves and re-ranks relevant knowledge based on image and/or text queries, and incorporates the retrieved content into a knowledge-augmented prompt to guide the VLM in producing contextually grounded responses. We validated the effectiveness of our approach on three representative vision-language tasks, including image captioning, image classification, and visual question answering, where RS-RAG significantly outperformed state-of-the-art baselines. 

**Abstract (ZH)**: 近期视觉语言模型在自然图像领域的进展展示出了跨多种任务的 impressive 能力。受这些进展的启发，遥感社区开始采用视觉语言模型进行遥感视觉-语言任务，包括场景理解、图像描述和视觉问答。然而，现有的遥感视觉语言模型通常依赖于封闭场景理解，并专注于通用场景描述，缺乏融合外部知识的能力。这一限制阻碍了它们对涉及特定领域或世界知识的复杂或依赖上下文的查询进行语义推理的能力。为了解决这些挑战，我们首先引入了一个多模态遥感世界知识（RSWK）数据集，该数据集包含来自175个国家的14,141个著名地标高分辨率卫星图像及其详细文本描述，整合了遥感领域知识和更广泛的世界知识。基于此数据集，我们提出了一种新颖的遥感检索增强生成（RS-RAG）框架，该框架由两个关键组件组成。多模态知识向量数据库构建模块将遥感图像及其相关文本知识编码到统一的向量空间中。知识检索和响应生成模块根据图像和/或文本查询检索和重新排名相关信息，并将检索的内容整合到一个增强知识的提示中，以指导视觉语言模型生成基于上下文的响应。我们在图像描述、图像分类和视觉问答等三个代表性视觉语言任务上验证了我们方法的有效性，其中RS-RAG在所有任务上显著优于现有最先进的基线模型。 

---
# DiCoTTA: Domain-invariant Learning for Continual Test-time Adaptation 

**Title (ZH)**: DiCoTTA: 领域不变学习以实现持续测试时适应 

**Authors**: Sohyun Lee, Nayeong Kim, Juwon Kang, Seong Joon Oh, Suha Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2504.04981)  

**Abstract**: This paper studies continual test-time adaptation (CTTA), the task of adapting a model to constantly changing unseen domains in testing while preserving previously learned knowledge. Existing CTTA methods mostly focus on adaptation to the current test domain only, overlooking generalization to arbitrary test domains a model may face in the future. To tackle this limitation, we present a novel online domain-invariant learning framework for CTTA, dubbed DiCoTTA. DiCoTTA aims to learn feature representation to be invariant to both current and previous test domains on the fly during testing. To this end, we propose a new model architecture and a test-time adaptation strategy dedicated to learning domain-invariant features without corrupting semantic contents, along with a new data structure and optimization algorithm for effectively managing information from previous test domains. DiCoTTA achieved state-of-the-art performance on four public CTTA benchmarks. Moreover, it showed superior generalization to unseen test domains. 

**Abstract (ZH)**: 本文研究了连续测试时适应（CTTA），即在测试过程中让模型适应不断变化的未见域，同时保留之前学到的知识。现有的CTTA方法主要关注当前测试域的适应问题，忽视了模型在未来可能遇到的任意测试域的一般性泛化能力。为解决这一局限性，我们提出了一种新的在线域不变学习框架DiCoTTA，旨在在测试过程中实时学习对当前和先前测试域均不变的特征表示。为此，我们提出了一种新的模型架构和测试时适应策略，专门用于学习域不变特征而不破坏语义内容，并提出了一种新的数据结构和优化算法来有效管理之前测试域的信息。DiCoTTA在四个公开的CTTA基准上取得了最先进的性能，并且展示了对未见测试域的优越泛化能力。 

---
# Towards Visual Text Grounding of Multimodal Large Language Model 

**Title (ZH)**: 面向多模态大语言模型的视觉文本定位 

**Authors**: Ming Li, Ruiyi Zhang, Jian Chen, Jiuxiang Gu, Yufan Zhou, Franck Dernoncourt, Wanrong Zhu, Tianyi Zhou, Tong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.04974)  

**Abstract**: Despite the existing evolution of Multimodal Large Language Models (MLLMs), a non-neglectable limitation remains in their struggle with visual text grounding, especially in text-rich images of documents. Document images, such as scanned forms and infographics, highlight critical challenges due to their complex layouts and textual content. However, current benchmarks do not fully address these challenges, as they mostly focus on visual grounding on natural images, rather than text-rich document images. Thus, to bridge this gap, we introduce TRIG, a novel task with a newly designed instruction dataset for benchmarking and improving the Text-Rich Image Grounding capabilities of MLLMs in document question-answering. Specifically, we propose an OCR-LLM-human interaction pipeline to create 800 manually annotated question-answer pairs as a benchmark and a large-scale training set of 90$ synthetic data based on four diverse datasets. A comprehensive evaluation of various MLLMs on our proposed benchmark exposes substantial limitations in their grounding capability on text-rich images. In addition, we propose two simple and effective TRIG methods based on general instruction tuning and plug-and-play efficient embedding, respectively. By finetuning MLLMs on our synthetic dataset, they promisingly improve spatial reasoning and grounding capabilities. 

**Abstract (ZH)**: 尽管现有的多模态大语言模型（MLLMs）已经取得了进展，但在视觉文本定位方面仍存在不容忽视的局限性，特别是在文档中的文本丰富图像中更为明显。文档图像，如扫描表单和信息图形，因其复杂的布局和文本内容而突显出关键挑战。然而，当前的基准测试并未充分解决这些问题，因为它们主要关注自然图像的视觉定位，而非文本丰富的文档图像。因此，为了填补这一空白，我们引入了TRIG，一种新型任务及其新设计的指令数据集，用于评估和提高MLLMs在文档问答中的文本丰富图像定位能力。具体而言，我们提出了一种OCR-LLM-人工交互流水线，创建了800个手工标注的问题-答案对作为基准和基于四个不同数据集的90,000个合成数据大规模训练集。对我们的基准测试的各种MLLMs进行综合评估揭示了它们在文本丰富的图像定位能力上的诸多局限性。此外，我们提出了两种基于通用指令调整和插拔高效嵌入的简单有效TRIG方法。通过在我们合成数据集上微调MLLMs，它们确实在空间推理和定位能力上取得了显著改进。 

---
# Ensuring Safety in an Uncertain Environment: Constrained MDPs via Stochastic Thresholds 

**Title (ZH)**: 在不确定环境中的安全性保障：通过随机阈值的约束MDPs 

**Authors**: Qian Zuo, Fengxiang He  

**Link**: [PDF](https://arxiv.org/pdf/2504.04973)  

**Abstract**: This paper studies constrained Markov decision processes (CMDPs) with constraints against stochastic thresholds, aiming at safety of reinforcement learning in unknown and uncertain environments. We leverage a Growing-Window estimator sampling from interactions with the uncertain and dynamic environment to estimate the thresholds, based on which we design Stochastic Pessimistic-Optimistic Thresholding (SPOT), a novel model-based primal-dual algorithm for multiple constraints against stochastic thresholds. SPOT enables reinforcement learning under both pessimistic and optimistic threshold settings. We prove that our algorithm achieves sublinear regret and constraint violation; i.e., a reward regret of $\tilde{\mathcal{O}}(\sqrt{T})$ while allowing an $\tilde{\mathcal{O}}(\sqrt{T})$ constraint violation over $T$ episodes. The theoretical guarantees show that our algorithm achieves performance comparable to that of an approach relying on fixed and clear thresholds. To the best of our knowledge, SPOT is the first reinforcement learning algorithm that realises theoretical guaranteed performance in an uncertain environment where even thresholds are unknown. 

**Abstract (ZH)**: 本文研究了面对随机阈值约束的受限马尔可夫决策过程（CMDPs），旨在未知和不确定环境中保证强化学习的安全性。我们利用 Growing-Window 估算器根据与动态环境的交互样本估计阈值，并基于此设计了 Stochastic Pessimistic-Optimistic Thresholding (SPOT) 算法，这是一种新的模型导向的 primal-dual 算法，用于处理面对随机阈值的多约束问题。SPOT 允许在悲观和乐观阈值设置下进行强化学习。我们证明了该算法实现了亚线性遗憾和约束违反，即在 $T$ 期中实现了 $\tilde{\mathcal{O}}(\sqrt{T})$ 的奖励遗憾和 $\tilde{\mathcal{O}}(\sqrt{T})$ 的约束违反。理论保证表明，该算法在性能上与依赖于固定清晰阈值的方法相当。据我们所知，SPOT 是首个在阈值甚至未知的不确定环境中实现理论保证性能的强化学习算法。 

---
# A High-Force Gripper with Embedded Multimodal Sensing for Powerful and Perception Driven Grasping 

**Title (ZH)**: 具有内置多模态感知的高力夹爪及其感知驱动抓取 

**Authors**: Edoardo Del Bianco, Davide Torielli, Federico Rollo, Damiano Gasperini, Arturo Laurenzi, Lorenzo Baccelliere, Luca Muratore, Marco Roveri, Nikos G. Tsagarakis  

**Link**: [PDF](https://arxiv.org/pdf/2504.04970)  

**Abstract**: Modern humanoid robots have shown their promising potential for executing various tasks involving the grasping and manipulation of objects using their end-effectors. Nevertheless, in the most of the cases, the grasping and manipulation actions involve low to moderate payload and interaction forces. This is due to limitations often presented by the end-effectors, which can not match their arm-reachable payload, and hence limit the payload that can be grasped and manipulated. In addition, grippers usually do not embed adequate perception in their hardware, and grasping actions are mainly driven by perception sensors installed in the rest of the robot body, frequently affected by occlusions due to the arm motions during the execution of the grasping and manipulation tasks. To address the above, we developed a modular high grasping force gripper equipped with embedded multi-modal perception functionalities. The proposed gripper can generate a grasping force of 110 N in a compact implementation. The high grasping force capability is combined with embedded multi-modal sensing, which includes an eye-in-hand camera, a Time-of-Flight (ToF) distance sensor, an Inertial Measurement Unit (IMU) and an omnidirectional microphone, permitting the implementation of perception-driven grasping functionalities.
We extensively evaluated the grasping force capacity of the gripper by introducing novel payload evaluation metrics that are a function of the robot arm's dynamic motion and gripper thermal states. We also evaluated the embedded multi-modal sensing by performing perception-guided enhanced grasping operations. 

**Abstract (ZH)**: 现代人形机器人展示了其在使用末端执行器执行各种涉及抓取和操作物体任务方面的前景。然而，在大多数情况下，抓取和操作动作涉及的载荷和相互作用力较低至中等。这主要是因为末端执行器的限制，它们无法匹配手臂可达的载荷，从而限制了可抓取和操作的载荷。此外，夹持器通常在其硬件中未嵌入足够的感知功能，抓取动作主要由安装在机器人身体其余部分的感知传感器驱动，这些传感器在执行抓取和操作任务时经常受到手臂运动引起的遮挡的影响。为了解决上述问题，我们开发了一种具有嵌入式多模态感知功能的模块化高抓取力夹持器，该夹持器可以在紧凑的实施中产生110 N的抓取力。高抓取力与嵌入式多模态传感功能相结合，包括手眼相机、飞行时间（ToF）距离传感器、惯性测量单元（IMU）和全向麦克风，允许实现感知驱动的抓取功能。我们通过引入新的载荷评估指标来广泛评估夹持器的抓取力能力，这些指标是机器人手臂动态运动和夹持器热状态的函数。我们还通过执行感知引导的增强抓取操作来评估嵌入的多模态传感功能。 

---
# The Dream Within Huang Long Cave: AI-Driven Interactive Narrative for Family Storytelling and Emotional Reflection 

**Title (ZH)**: 黄龙洞中的梦境：基于AI驱动的互动叙事家庭叙事与情感反思 

**Authors**: Jiayang Huang, Lingjie Li, Kang Zhang, David Yip  

**Link**: [PDF](https://arxiv.org/pdf/2504.04968)  

**Abstract**: This paper introduces the art project The Dream Within Huang Long Cave, an AI-driven interactive and immersive narrative experience. The project offers new insights into AI technology, artistic practice, and psychoanalysis. Inspired by actual geographical landscapes and familial archetypes, the work combines psychoanalytic theory and computational technology, providing an artistic response to the concept of the non-existence of the Big Other. The narrative is driven by a combination of a large language model (LLM) and a realistic digital character, forming a virtual agent named YELL. Through dialogue and exploration within a cave automatic virtual environment (CAVE), the audience is invited to unravel the language puzzles presented by YELL and help him overcome his life challenges. YELL is a fictional embodiment of the Big Other, modeled after the artist's real father. Through a cross-temporal interaction with this digital father, the project seeks to deconstruct complex familial relationships. By demonstrating the non-existence of the Big Other, we aim to underscore the authenticity of interpersonal emotions, positioning art as a bridge for emotional connection and understanding within family dynamics. 

**Abstract (ZH)**: 基于黄龙洞的梦境：AI驱动的互动沉浸叙事艺术项目 

---
# M-Prometheus: A Suite of Open Multilingual LLM Judges 

**Title (ZH)**: M-Prometheus：一套开源多语言大模型评测套件 

**Authors**: José Pombal, Dongkeun Yoon, Patrick Fernandes, Ian Wu, Seungone Kim, Ricardo Rei, Graham Neubig, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2504.04953)  

**Abstract**: The use of language models for automatically evaluating long-form text (LLM-as-a-judge) is becoming increasingly common, yet most LLM judges are optimized exclusively for English, with strategies for enhancing their multilingual evaluation capabilities remaining largely unexplored in the current literature. This has created a disparity in the quality of automatic evaluation methods for non-English languages, ultimately hindering the development of models with better multilingual capabilities. To bridge this gap, we introduce M-Prometheus, a suite of open-weight LLM judges ranging from 3B to 14B parameters that can provide both direct assessment and pairwise comparison feedback on multilingual outputs. M-Prometheus models outperform state-of-the-art open LLM judges on multilingual reward benchmarks spanning more than 20 languages, as well as on literary machine translation (MT) evaluation covering 4 language pairs. Furthermore, M-Prometheus models can be leveraged at decoding time to significantly improve generated outputs across all 3 tested languages, showcasing their utility for the development of better multilingual models. Lastly, through extensive ablations, we identify the key factors for obtaining an effective multilingual judge, including backbone model selection and training on natively multilingual feedback data instead of translated data. We release our models, training dataset, and code. 

**Abstract (ZH)**: 使用大规模语言模型自动评估长文本（LLM-as-a-judge）在日益普遍，然而目前大多数LLM法官仅针对英文进行了优化，关于提升其多语言评估能力的策略在现有文献中研究较少。这导致非英文语言的自动评估方法质量参差不齐，最终阻碍了具有良好多语言能力模型的发展。为进一步弥合这一差距，我们引入了M-Prometheus，这是一种从3B到14B参数的开放权重LLM法官系列，能提供多语言输出的直接评估和成对比较反馈。M-Prometheus模型在覆盖超过20种语言的多语言奖励基准测试中表现优于最先进的开放LLM法官，并在涵盖4种语言对的文学机器翻译（MT）评估中表现出色。此外，M-Prometheus模型在解码时可以显著改善所有3种测试语言生成的输出，展示了其在开发更好多语言模型中的应用前景。最后，通过广泛的消融实验，我们确定了有效多语言法官的关键因素，包括基础模型的选择以及使用原生多语言反馈数据进行训练而非翻译数据。我们发布了自己的模型、训练数据集和代码。 

---
# One Quantizer is Enough: Toward a Lightweight Audio Codec 

**Title (ZH)**: 一个量化器足矣： toward a轻量级音频编解码器 

**Authors**: Linwei Zhai, Han Ding, Cui Zhao, fei wang, Ge Wang, Wang Zhi, Wei Xi  

**Link**: [PDF](https://arxiv.org/pdf/2504.04949)  

**Abstract**: Neural audio codecs have recently gained traction for their ability to compress high-fidelity audio and generate discrete tokens that can be utilized in downstream generative modeling tasks. However, leading approaches often rely on resource-intensive models and multi-quantizer architectures, resulting in considerable computational overhead and constrained real-world applicability. In this paper, we present SQCodec, a lightweight neural audio codec that leverages a single quantizer to address these limitations. SQCodec explores streamlined convolutional networks and local Transformer modules, alongside TConv, a novel mechanism designed to capture acoustic variations across multiple temporal scales, thereby enhancing reconstruction fidelity while reducing model complexity. Extensive experiments across diverse datasets show that SQCodec achieves audio quality comparable to multi-quantizer baselines, while its single-quantizer design offers enhanced adaptability and its lightweight architecture reduces resource consumption by an order of magnitude. The source code is publicly available at this https URL. 

**Abstract (ZH)**: 神经音频编解码器近年来因其能压缩高保真音频并生成可用于下游生成建模任务的离散词元而受到关注。然而，领先的方法通常依赖于计算密集型模型和多量化器架构，导致了显著的计算开销和有限的实际应用范围。本文我们提出了SQCodec，一种轻量级的神经音频编解码器，利用单一量化器来克服这些限制。SQCodec探究了精简的卷积网络和局部Transformer模块，并引入了TConv机制，该机制能够捕捉不同时间尺度上的声学变化，从而在降低模型复杂度的同时提升重构保真度。在多种数据集上的广泛实验表明，SQCodec在音频质量上与多量化器基线相当，其单一量化器设计增强了模型的适应性，而其轻量级架构将资源消耗降低了十倍。源代码可在如下链接获取：这个 https URL。 

---
# A Llama walks into the 'Bar': Efficient Supervised Fine-Tuning for Legal Reasoning in the Multi-state Bar Exam 

**Title (ZH)**: 一只 llama 走进了酒吧：多州律师资格考试中的高效监督微调以进行法律推理 

**Authors**: Rean Fernandes, André Biedenkapp, Frank Hutter, Noor Awad  

**Link**: [PDF](https://arxiv.org/pdf/2504.04945)  

**Abstract**: Legal reasoning tasks present unique challenges for large language models (LLMs) due to the complexity of domain-specific knowledge and reasoning processes. This paper investigates how effectively smaller language models (Llama 2 7B and Llama 3 8B) can be fine-tuned with a limited dataset of 1,514 Multi-state Bar Examination (MBE) questions to improve legal question answering accuracy. We evaluate these models on the 2022 MBE questions licensed from JD Advising, the same dataset used in the 'GPT-4 passes the Bar exam' study. Our methodology involves collecting approximately 200 questions per legal domain across 7 domains. We distill the dataset using Llama 3 (70B) to transform explanations into a structured IRAC (Issue, Rule, Application, Conclusion) format as a guided reasoning process to see if it results in better performance over the non-distilled dataset. We compare the non-fine-tuned models against their supervised fine-tuned (SFT) counterparts, trained for different sample sizes per domain, to study the effect on accuracy and prompt adherence. We also analyse option selection biases and their mitigation following SFT. In addition, we consolidate the performance across multiple variables: prompt type (few-shot vs zero-shot), answer ordering (chosen-option first vs generated-explanation first), response format (Numbered list vs Markdown vs JSON), and different decoding temperatures. Our findings show that domain-specific SFT helps some model configurations achieve close to human baseline performance, despite limited computational resources and a relatively small dataset. We release both the gathered SFT dataset and the family of Supervised Fine-tuned (SFT) adapters optimised for MBE performance. This establishes a practical lower bound on resources needed towards achieving effective legal question answering in smaller LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）因领域特定知识和推理过程的复杂性而面临独特的法律推理任务挑战。本文研究较小的语言模型（Llama 2 7B和Llama 3 8B）如何通过使用1,514个多元州律师考试（MBE）问题的有限数据集进行微调，以提高法律问题回答的准确性。我们利用JD Advising提供的2022年MBE问题这一相同的数据集评估这些模型。我们的方法包括在七个法律领域中每个领域收集约200个问题。我们使用Llama 3（70B）精简数据集，将其解释转换为结构化的IRAC（Issue, Rule, Application, Conclusion）格式，以指导推理过程，观察其是否能比未精简的数据集产生更好的效果。我们对比了未经微调的模型与其监督微调（SFT）版本在不同领域样本大小下的表现，研究其对准确性和指令遵循性的影响。我们还分析了SFT后选项选择偏见及其缓解措施。此外，我们跨多个变量综合性能表现：提示类型（少量样本vs零样本），答案排序（选择选项优先vs生成解释优先），响应格式（编号列表vsMarkdownvsJSON），以及不同的解码温度。我们的发现表明，尽管计算资源有限且数据集较小，领域特定的SFT仍有助于某些模型配置接近人类基础水平的性能。我们发布了收集的SFT数据集和优化用于MBE性能的监督微调（SFT）适配器家族。这确立了实现较小LLMs有效法律问题回答所需资源的实用下限。 

---
# A Taxonomy of Self-Handover 

**Title (ZH)**: 自我切换的分类学 

**Authors**: Naoki Wake, Atsushi Kanehira, Kazuhiro Sasabuchi, Jun Takamatsu, Katsushi Ikeuchi  

**Link**: [PDF](https://arxiv.org/pdf/2504.04939)  

**Abstract**: Self-handover, transferring an object between one's own hands, is a common but understudied bimanual action. While it facilitates seamless transitions in complex tasks, the strategies underlying its execution remain largely unexplored. Here, we introduce the first systematic taxonomy of self-handover, derived from manual annotation of over 12 hours of cooking activity performed by 21 participants. Our analysis reveals that self-handover is not merely a passive transition, but a highly coordinated action involving anticipatory adjustments by both hands. As a step toward automated analysis of human manipulation, we further demonstrate the feasibility of classifying self-handover types using a state-of-the-art vision-language model. These findings offer fresh insights into bimanual coordination, underscoring the role of self-handover in enabling smooth task transitions-an ability essential for adaptive dual-arm robotics. 

**Abstract (ZH)**: 自我转换：在双手之间转移物体是一项常见但研究不足的双手法动作。通过系统分类和自动化分析的研究，揭示了自我转换不仅是被动的过渡，更是双手之间高度协调的动作，涉及预见性的调整。这些发现为理解和分类自我转换类型提供了新的视角，强调了自我转换在实现顺畅任务过渡中的作用，这对于适应型双臂机器人至关重要。 

---
# RCCFormer: A Robust Crowd Counting Network Based on Transformer 

**Title (ZH)**: RCCFormer：基于 transformer 的鲁棒人流计数网络 

**Authors**: Peng Liu, Heng-Chao Li, Sen Lei, Nanqing Liu, Bin Feng, Xiao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04935)  

**Abstract**: Crowd counting, which is a key computer vision task, has emerged as a fundamental technology in crowd analysis and public safety management. However, challenges such as scale variations and complex backgrounds significantly impact the accuracy of crowd counting. To mitigate these issues, this paper proposes a robust Transformer-based crowd counting network, termed RCCFormer, specifically designed for background suppression and scale awareness. The proposed method incorporates a Multi-level Feature Fusion Module (MFFM), which meticulously integrates features extracted at diverse stages of the backbone architecture. It establishes a strong baseline capable of capturing intricate and comprehensive feature representations, surpassing traditional baselines. Furthermore, the introduced Detail-Embedded Attention Block (DEAB) captures contextual information and local details through global self-attention and local attention along with a learnable manner for efficient fusion. This enhances the model's ability to focus on foreground regions while effectively mitigating background noise interference. Additionally, we develop an Adaptive Scale-Aware Module (ASAM), with our novel Input-dependent Deformable Convolution (IDConv) as its fundamental building block. This module dynamically adapts to changes in head target shapes and scales, significantly improving the network's capability to accommodate large-scale variations. The effectiveness of the proposed method is validated on the ShanghaiTech Part_A and Part_B, NWPU-Crowd, and QNRF datasets. The results demonstrate that our RCCFormer achieves excellent performance across all four datasets, showcasing state-of-the-art outcomes. 

**Abstract (ZH)**: 基于Transformer的鲁棒 crowd counting 网络 RCCFormer：背景抑制与尺度awareness的设计 

---
# Boosting Relational Deep Learning with Pretrained Tabular Models 

**Title (ZH)**: 利用预训练表型模型增强关系深度学习 

**Authors**: Veronica Lachi, Antonio Longa, Beatrice Bevilacqua, Bruno Lepri, Andrea Passerini, Bruno Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2504.04934)  

**Abstract**: Relational databases, organized into tables connected by primary-foreign key relationships, are a common format for organizing data. Making predictions on relational data often involves transforming them into a flat tabular format through table joins and feature engineering, which serve as input to tabular methods. However, designing features that fully capture complex relational patterns remains challenging. Graph Neural Networks (GNNs) offer a compelling alternative by inherently modeling these relationships, but their time overhead during inference limits their applicability for real-time scenarios. In this work, we aim to bridge this gap by leveraging existing feature engineering efforts to enhance the efficiency of GNNs in relational databases. Specifically, we use GNNs to capture complex relationships within relational databases, patterns that are difficult to featurize, while employing engineered features to encode temporal information, thereby avoiding the need to retain the entire historical graph and enabling the use of smaller, more efficient graphs. Our \textsc{LightRDL} approach not only improves efficiency, but also outperforms existing models. Experimental results on the RelBench benchmark demonstrate that our framework achieves up to $33\%$ performance improvement and a $526\times$ inference speedup compared to GNNs, making it highly suitable for real-time inference. 

**Abstract (ZH)**: 关系数据库通过主-外键关系组织成表格，是组织数据的常见格式。对关系数据进行预测通常涉及通过表连接和特征工程将它们转换为扁平的表格格式，这些表格作为表方法的输入。然而，设计能够完全捕捉复杂关系模式的特征仍具有挑战性。图神经网络（GNNs）通过内在建模这些关系提供了有吸引力的替代方案，但在推理时的时间开销限制了它们在实时场景中的应用。在本工作中，我们通过利用现有的特征工程努力来提高GNNs在关系数据库中的效率。具体而言，我们使用GNNs捕捉关系数据库中的复杂关系，这些关系难以特征化，同时采用工程特征编码时间信息，从而避免保留整个历史图并能够使用更小、更高效的图。我们的\textsc{LightRDL}方法不仅提高了效率，还在某些方面优于现有模型。基准测试RelBench的实验结果表明，与GNNs相比，我们的框架在性能上最多可提高33%，推理速度提高526倍，使其非常适用于实时推理。 

---
# Expectations vs Reality -- A Secondary Study on AI Adoption in Software Testing 

**Title (ZH)**: 期望与现实——软件测试中AI采用的二次研究 

**Authors**: Katja Karhu, Jussi Kasurinen, Kari Smolander  

**Link**: [PDF](https://arxiv.org/pdf/2504.04921)  

**Abstract**: In the software industry, artificial intelligence (AI) has been utilized more and more in software development activities. In some activities, such as coding, AI has already been an everyday tool, but in software testing activities AI it has not yet made a significant breakthrough. In this paper, the objective was to identify what kind of empirical research with industry context has been conducted on AI in software testing, as well as how AI has been adopted in software testing practice. To achieve this, we performed a systematic mapping study of recent (2020 and later) studies on AI adoption in software testing in the industry, and applied thematic analysis to identify common themes and categories, such as the real-world use cases and benefits, in the found papers. The observations suggest that AI is not yet heavily utilized in software testing, and still relatively few studies on AI adoption in software testing have been conducted in the industry context to solve real-world problems. Earlier studies indicated there was a noticeable gap between the actual use cases and actual benefits versus the expectations, which we analyzed further. While there were numerous potential use cases for AI in software testing, such as test case generation, code analysis, and intelligent test automation, the reported actual implementations and observed benefits were limited. In addition, the systematic mapping study revealed a potential problem with false positive search results in online databases when using the search string "artificial intelligence". 

**Abstract (ZH)**: 在软件行业中，人工智能（AI）已在软件开发活动中得到了越来越多的应用。在某些活动中，如编码，AI已经成为日常工作中的工具，但在软件测试活动中，AI尚未取得重大突破。本文旨在识别在软件测试领域中开展的行业背景下的人工智能实证研究，以及人工智能在软件测试实践中的应用情况。为此，我们对2020年及以后有关软件测试中人工智能应用的行业研究进行了系统映射研究，并运用主题分析方法识别出常见的主题和类别，如实际应用案例及其好处。观察结果表明，目前人工智能在软件测试中的应用尚不广泛，行业内针对软件测试中人工智能应用的实际问题开展的研究仍然相对较少。早期研究表明，实际应用案例和实际好处与预期之间存在明显的差距，我们对此进行了进一步分析。虽然人工智能在软件测试中有众多潜在应用场景，如测试案例生成、代码分析和智能测试自动化等，但实际实施情况和观察到的好处却相当有限。此外，系统映射研究还揭示了使用搜索字符串“人工智能”在在线数据库中搜索时可能出现的假阳性结果的问题。 

---
# Collab-RAG: Boosting Retrieval-Augmented Generation for Complex Question Answering via White-Box and Black-Box LLM Collaboration 

**Title (ZH)**: Collab-RAG：通过白盒和黑盒大模型协作提升检索增强生成复杂问题解答 

**Authors**: Ran Xu, Wenqi Shi, Yuchen Zhuang, Yue Yu, Joyce C. Ho, Haoyu Wang, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04915)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems often struggle to handle multi-hop question-answering tasks accurately due to irrelevant context retrieval and limited complex reasoning capabilities. We introduce Collab-RAG, a collaborative training framework that leverages mutual enhancement between a white-box small language model (SLM) and a blackbox large language model (LLM) for RAG. Specifically, the SLM decomposes complex queries into simpler sub-questions, thus enhancing the accuracy of the retrieval and facilitating more effective reasoning by the black-box LLM. Concurrently, the black-box LLM provides feedback signals to improve the SLM's decomposition capability. We observe that Collab-RAG relies solely on supervision from an affordable black-box LLM without additional distillation from frontier LLMs, yet demonstrates strong generalization across multiple black-box LLMs. Experimental evaluations across five multi-hop QA datasets demonstrate that Collab-RAG substantially outperforms existing black-box-only and SLM fine-tuning baselines by 1.8%-14.2% on average. In particular, our fine-tuned 3B SLM surpasses a frozen 32B LLM in question decomposition, highlighting the efficiency of Collab-RAG in improving reasoning and retrieval for complex questions. The code of Collab-RAG is available on this https URL. 

**Abstract (ZH)**: Collab-RAG：一种基于白盒小语言模型和黑盒大语言模型协同训练的检索增强生成框架 

---
# AlgOS: Algorithm Operating System 

**Title (ZH)**: AlgOS: 算法操作系统 

**Authors**: Llewyn Salt, Marcus Gallagher  

**Link**: [PDF](https://arxiv.org/pdf/2504.04909)  

**Abstract**: Algorithm Operating System (AlgOS) is an unopinionated, extensible, modular framework for algorithmic implementations. AlgOS offers numerous features: integration with Optuna for automated hyperparameter tuning; automated argument parsing for generic command-line interfaces; automated registration of new classes; and a centralised database for logging experiments and studies. These features are designed to reduce the overhead of implementing new algorithms and to standardise the comparison of algorithms. The standardisation of algorithmic implementations is crucial for reproducibility and reliability in research. AlgOS combines Abstract Syntax Trees with a novel implementation of the Observer pattern to control the logical flow of algorithmic segments. 

**Abstract (ZH)**: 算法操作系统（AlgOS）是一种无偏见、可扩展和模块化的算法实现框架。AlgOS 提供了诸多功能：与 Optuna 集成以实现自动超参数调优；自动生成命令行接口参数；自动注册新类；以及中央数据库用于记录实验和研究。这些功能旨在减少新算法实现的开销，并标准化算法的比较。规范化的算法实现对于研究中的可重复性和可靠性至关重要。AlgOS 结合使用抽象语法树与观察者模式的新型实现来控制算法段落的逻辑流程。 

---
# Video-Bench: Human-Aligned Video Generation Benchmark 

**Title (ZH)**: Video-Bench: 人体对齐的视频生成基准 

**Authors**: Hui Han, Siyuan Li, Jiaqi Chen, Yiwen Yuan, Yuling Wu, Chak Tou Leong, Hanwen Du, Junchen Fu, Youhua Li, Jie Zhang, Chi Zhang, Li-jia Li, Yongxin Ni  

**Link**: [PDF](https://arxiv.org/pdf/2504.04907)  

**Abstract**: Video generation assessment is essential for ensuring that generative models produce visually realistic, high-quality videos while aligning with human expectations. Current video generation benchmarks fall into two main categories: traditional benchmarks, which use metrics and embeddings to evaluate generated video quality across multiple dimensions but often lack alignment with human judgments; and large language model (LLM)-based benchmarks, though capable of human-like reasoning, are constrained by a limited understanding of video quality metrics and cross-modal consistency. To address these challenges and establish a benchmark that better aligns with human preferences, this paper introduces Video-Bench, a comprehensive benchmark featuring a rich prompt suite and extensive evaluation dimensions. This benchmark represents the first attempt to systematically leverage MLLMs across all dimensions relevant to video generation assessment in generative models. By incorporating few-shot scoring and chain-of-query techniques, Video-Bench provides a structured, scalable approach to generated video evaluation. Experiments on advanced models including Sora demonstrate that Video-Bench achieves superior alignment with human preferences across all dimensions. Moreover, in instances where our framework's assessments diverge from human evaluations, it consistently offers more objective and accurate insights, suggesting an even greater potential advantage over traditional human judgment. 

**Abstract (ZH)**: 视频生成评估对于确保生成模型产生视觉上真实、高质量的视频并与人类期望保持一致至关重要。当前视频生成基准主要分为两类：传统的基准，使用多种度量和嵌入来评估生成视频的质量，但往往缺乏与人类判断的对齐；以及基于大规模语言模型（LLM）的基准，尽管具备类似人类的推理能力，但在理解和视频质量度量以及跨模态一致性方面能力有限。为了应对这些挑战并建立一个更符合人类偏好的基准，本文引入了Video-Bench，一个涵盖丰富提示集和广泛评估维度的综合基准。Video-Bench是首次尝试系统地在所有与视频生成评估相关的维度中利用MLLMs。通过结合少量示例评分和查询链技术，Video-Bench提供了一种结构化、可扩展的生成视频评估方法。实验表明，Video-Bench在所有维度中都实现了与人类偏好的更好对齐。此外，在我们的框架评估与人类评价出现分歧的情况下，它始终提供了更加客观和准确的洞察，表明在传统人类判断之外可能存在更大的优势。 

---
# Lumina-OmniLV: A Unified Multimodal Framework for General Low-Level Vision 

**Title (ZH)**: Lumina-OmniLV：统一的多模态框架用于通用低级视觉任务 

**Authors**: Yuandong Pu, Le Zhuo, Kaiwen Zhu, Liangbin Xie, Wenlong Zhang, Xiangyu Chen, Pneg Gao, Yu Qiao, Chao Dong, Yihao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04903)  

**Abstract**: We present Lunima-OmniLV (abbreviated as OmniLV), a universal multimodal multi-task framework for low-level vision that addresses over 100 sub-tasks across four major categories: image restoration, image enhancement, weak-semantic dense prediction, and stylization. OmniLV leverages both textual and visual prompts to offer flexible and user-friendly interactions. Built on Diffusion Transformer (DiT)-based generative priors, our framework supports arbitrary resolutions -- achieving optimal performance at 1K resolution -- while preserving fine-grained details and high fidelity. Through extensive experiments, we demonstrate that separately encoding text and visual instructions, combined with co-training using shallow feature control, is essential to mitigate task ambiguity and enhance multi-task generalization. Our findings also reveal that integrating high-level generative tasks into low-level vision models can compromise detail-sensitive restoration. These insights pave the way for more robust and generalizable low-level vision systems. 

**Abstract (ZH)**: Lunima-OmniLV：一种适用于低级视觉的通用多模态多任务框架 

---
# SCAM: A Real-World Typographic Robustness Evaluation for Multimodal Foundation Models 

**Title (ZH)**: SCAM：多模态基础模型的现实世界 typography 稳定性评估 

**Authors**: Justus Westerhoff, Erblina Purellku, Jakob Hackstein, Leo Pinetzki, Lorenz Hufe  

**Link**: [PDF](https://arxiv.org/pdf/2504.04893)  

**Abstract**: Typographic attacks exploit the interplay between text and visual content in multimodal foundation models, causing misclassifications when misleading text is embedded within images. However, existing datasets are limited in size and diversity, making it difficult to study such vulnerabilities. In this paper, we introduce SCAM, the largest and most diverse dataset of real-world typographic attack images to date, containing 1,162 images across hundreds of object categories and attack words. Through extensive benchmarking of Vision-Language Models (VLMs) on SCAM, we demonstrate that typographic attacks significantly degrade performance, and identify that training data and model architecture influence the susceptibility to these attacks. Our findings reveal that typographic attacks persist in state-of-the-art Large Vision-Language Models (LVLMs) due to the choice of their vision encoder, though larger Large Language Models (LLMs) backbones help mitigate their vulnerability. Additionally, we demonstrate that synthetic attacks closely resemble real-world (handwritten) attacks, validating their use in research. Our work provides a comprehensive resource and empirical insights to facilitate future research toward robust and trustworthy multimodal AI systems. We publicly release the datasets introduced in this paper under this https URL, along with the code for evaluations at this https URL. 

**Abstract (ZH)**: 图像攻击利用了文本与视觉内容在多模态基础模型中的交互作用，当误导性文本嵌入图像中时，会导致分类错误。然而，现有的数据集在大小和多样性方面有限，使得研究这些漏洞变得困难。在本文中，我们引入了SCAM，这是目前最大的也是最多样化的实际图像文字攻击数据集，包含1,162张图像，覆盖数百个对象类别和攻击词。通过在SCAM上对视觉语言模型（VLMs）进行广泛基准测试，我们证明了图像攻击显著降低了性能，并且发现训练数据和模型架构会影响这些攻击的易感性。我们的发现揭示了由于大型视觉编码器的选取，图像攻击在当今最先进的大型视觉语言模型（LVLMs）中仍然存在，尽管较大的大型语言模型（LLMs）骨干网络有助于缓解其脆弱性。此外，我们证明合成攻击与真实世界（手写）攻击非常相似，验证了其在研究中的使用。我们的工作提供了一个全面的资源和实证见解，以促进未来对稳健和可信的多模态AI系统的研究。我们在本文公开发布了所引入的数据集（此链接：https://），并提供了评估代码（此链接：https://）。 

---
# Futureproof Static Memory Planning 

**Title (ZH)**: 面向未来的静态内存规划 

**Authors**: Christos Lamprakos, Panagiotis Xanthopoulos, Manolis Katsaragakis, Sotirios Xydis, Dimitrios Soudris, Francky Catthoor  

**Link**: [PDF](https://arxiv.org/pdf/2504.04874)  

**Abstract**: The NP-complete combinatorial optimization task of assigning offsets to a set of buffers with known sizes and lifetimes so as to minimize total memory usage is called dynamic storage allocation (DSA). Existing DSA implementations bypass the theoretical state-of-the-art algorithms in favor of either fast but wasteful heuristics, or memory-efficient approaches that do not scale beyond one thousand buffers. The "AI memory wall", combined with deep neural networks' static architecture, has reignited interest in DSA. We present idealloc, a low-fragmentation, high-performance DSA implementation designed for million-buffer instances. Evaluated on a novel suite of particularly hard benchmarks from several domains, idealloc ranks first against four production implementations in terms of a joint effectiveness/robustness criterion. 

**Abstract (ZH)**: 已知大小和寿命的一组缓冲区分配偏移量，以最小化总内存使用量的NP完全组合优化任务称为动态存储分配（DSA）。现有的DSA实现绕过理论上的最先进的算法，要么使用快速但浪费的空间启发式方法，要么使用内存效率高的方法，但这些方法无法扩展到一千个缓冲区以上。“AI内存墙”与深度神经网络的静态架构相结合，重新引起了对DSA的兴趣。我们提出了一种名为idealloc的低碎片、高性能DSA实现，适用于百万缓冲区实例。在对来自多个领域的新型尤其是难以解决的基准测试中，idealloc在联合效果/鲁棒性标准上排名第一，与四种生产实现相比。 

---
# FedSAUC: A Similarity-Aware Update Control for Communication-Efficient Federated Learning in Edge Computing 

**Title (ZH)**: FedSAUC：一种面向边缘计算的通信高效联邦学习的相似性感知更新控制 

**Authors**: Ming-Lun Lee, Han-Chang Chou, Yan-AnnChen  

**Link**: [PDF](https://arxiv.org/pdf/2504.04867)  

**Abstract**: Federated learning is a distributed machine learning framework to collaboratively train a global model without uploading privacy-sensitive data onto a centralized server. Usually, this framework is applied to edge devices such as smartphones, wearable devices, and Internet of Things (IoT) devices which closely collect information from users. However, these devices are mostly battery-powered. The update procedure of federated learning will constantly consume the battery power and the transmission bandwidth. In this work, we propose an update control for federated learning, FedSAUC, by considering the similarity of users' behaviors (models). At the server side, we exploit clustering algorithms to group devices with similar models. Then we select some representatives for each cluster to update information to train the model. We also implemented a testbed prototyping on edge devices for validating the performance. The experimental results show that this update control will not affect the training accuracy in the long run. 

**Abstract (ZH)**: 联邦学习是一种无需将隐私敏感数据上传到集中服务器的分布式机器学习框架，用于协作训练全局模型。通常，该框架应用于智能手机、可穿戴设备和物联网（IoT）设备等边缘设备，这些设备能够紧密收集用户信息。然而，这些设备大多为电池供电。联邦学习的更新过程会不断消耗电池电量和传输带宽。在本文中，我们提出了一个考虑用户行为（模型）相似性的更新控制算法，称为FedSAUC。在服务器端，我们利用聚类算法将具有相似模型的设备分组。然后，我们为每个集群选择一些代表性的设备来更新信息以训练模型。我们还在边缘设备上实现了一个测试床原型以验证性能。实验结果表明，这种更新控制在长期训练中不会影响训练精度。 

---
# SAFT: Structure-aware Transformers for Textual Interaction Classification 

**Title (ZH)**: SAFT：结构感知变压器在文本交互分类中的应用 

**Authors**: Hongtao Wang, Renchi Yang, Hewen Wang, Haoran Zheng, Jianliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04861)  

**Abstract**: Textual interaction networks (TINs) are an omnipresent data structure used to model the interplay between users and items on e-commerce websites, social networks, etc., where each interaction is associated with a text description. Classifying such textual interactions (TIC) finds extensive use in detecting spam reviews in e-commerce, fraudulent transactions in finance, and so on. Existing TIC solutions either (i) fail to capture the rich text semantics due to the use of context-free text embeddings, and/or (ii) disregard the bipartite structure and node heterogeneity of TINs, leading to compromised TIC performance. In this work, we propose SAFT, a new architecture that integrates language- and graph-based modules for the effective fusion of textual and structural semantics in the representation learning of interactions. In particular, line graph attention (LGA)/gated attention units (GAUs) and pretrained language models (PLMs) are capitalized on to model the interaction-level and token-level signals, which are further coupled via the proxy token in an iterative and contextualized fashion. Additionally, an efficient and theoretically-grounded approach is developed to encode the local and global topology information pertaining to interactions into structural embeddings. The resulting embeddings not only inject the structural features underlying TINs into the textual interaction encoding but also facilitate the design of graph sampling strategies. Extensive empirical evaluations on multiple real TIN datasets demonstrate the superiority of SAFT over the state-of-the-art baselines in TIC accuracy. 

**Abstract (ZH)**: 基于语言和图的交互表示学习框架SAFT 

---
# Explanation-Driven Interventions for Artificial Intelligence Model Customization: Empowering End-Users to Tailor Black-Box AI in Rhinocytology 

**Title (ZH)**: 基于解释的干预方法促进人工智能模型定制：使终端用户赋能以定制黑盒AI在真菌学中的应用 

**Authors**: Andrea Esposito, Miriana Calvano, Antonio Curci, Francesco Greco, Rosa Lanzilotti, Antonio Piccinno  

**Link**: [PDF](https://arxiv.org/pdf/2504.04833)  

**Abstract**: The integration of Artificial Intelligence (AI) in modern society is heavily shifting the way that individuals carry out their tasks and activities. Employing AI-based systems raises challenges that designers and developers must address to ensure that humans remain in control of the interaction process, particularly in high-risk domains. This article presents a novel End-User Development (EUD) approach for black-box AI models through a redesigned user interface in the Rhino-Cyt platform, a medical AI-based decision-support system for medical professionals (more precisely, rhinocytologists) to carry out cell classification. The proposed interface empowers users to intervene in AI decision-making process by editing explanations and reconfiguring the model, influencing its future predictions. This work contributes to Human-Centered AI (HCAI) and EUD by discussing how explanation-driven interventions allow a blend of explainability, user intervention, and model reconfiguration, fostering a symbiosis between humans and user-tailored AI systems. 

**Abstract (ZH)**: 人工智能在现代社会中的集成正大幅改变个体执行任务和活动的方式。基于AI系统的应用给设计师和开发者带来了挑战，他们必须应对这些挑战以确保人类在交互过程中保持控制权，特别是在高风险领域。本文通过在Rhino-Cyt平台中重新设计用户界面，提出了一种新的面向终端用户的开发（EUD）方法，该平台是一个基于AI的医疗决策支持系统，旨在帮助医疗专业人士（更精确地说是鼻黏膜学家）进行细胞分类。提出的界面赋予用户干预AI决策过程的能力，通过编辑解释和重新配置模型来影响其未来的预测。本文通过讨论基于解释的干预措施如何实现可解释性、用户干预和模型重构的融合，为以人为本的人工智能（HCAI）和EUD做出了贡献。 

---
# From Specificity to Generality: Revisiting Generalizable Artifacts in Detecting Face Deepfakes 

**Title (ZH)**: 从具体性到普遍性：重新审视检测人脸深fake的可迁移 artifacts 

**Authors**: Long Ma, Zhiyuan Yan, Yize Chen, Jin Xu, Qinglang Guo, Hu Huang, Yong Liao, Hui Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.04827)  

**Abstract**: Detecting deepfakes has been an increasingly important topic, especially given the rapid development of AI generation techniques. In this paper, we ask: How can we build a universal detection framework that is effective for most facial deepfakes? One significant challenge is the wide variety of deepfake generators available, resulting in varying forgery artifacts (e.g., lighting inconsistency, color mismatch, etc). But should we ``teach" the detector to learn all these artifacts separately? It is impossible and impractical to elaborate on them all. So the core idea is to pinpoint the more common and general artifacts across different deepfakes. Accordingly, we categorize deepfake artifacts into two distinct yet complementary types: Face Inconsistency Artifacts (FIA) and Up-Sampling Artifacts (USA). FIA arise from the challenge of generating all intricate details, inevitably causing inconsistencies between the complex facial features and relatively uniform surrounding areas. USA, on the other hand, are the inevitable traces left by the generator's decoder during the up-sampling process. This categorization stems from the observation that all existing deepfakes typically exhibit one or both of these artifacts. To achieve this, we propose a new data-level pseudo-fake creation framework that constructs fake samples with only the FIA and USA, without introducing extra less-general artifacts. Specifically, we employ a super-resolution to simulate the USA, while design a Blender module that uses image-level self-blending on diverse facial regions to create the FIA. We surprisingly found that, with this intuitive design, a standard image classifier trained only with our pseudo-fake data can non-trivially generalize well to unseen deepfakes. 

**Abstract (ZH)**: 构建一种针对大多数面部深fake通用且有效的检测框架 

---
# Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models 

**Title (ZH)**: 量化伤害推理能力？关于量化推理模型的一项实证研究 

**Authors**: Ruikang Liu, Yuxuan Sun, Manyi Zhang, Haoli Bai, Xianzhi Yu, Tiezheng Yu, Chun Yuan, Lu Hou  

**Link**: [PDF](https://arxiv.org/pdf/2504.04823)  

**Abstract**: Recent advancements in reasoning language models have demonstrated remarkable performance in complex tasks, but their extended chain-of-thought reasoning process increases inference overhead. While quantization has been widely adopted to reduce the inference cost of large language models, its impact on reasoning models remains understudied. In this study, we conduct the first systematic study on quantized reasoning models, evaluating the open-sourced DeepSeek-R1-Distilled Qwen and LLaMA families ranging from 1.5B to 70B parameters, and QwQ-32B. Our investigation covers weight, KV cache, and activation quantization using state-of-the-art algorithms at varying bit-widths, with extensive evaluation across mathematical (AIME, MATH-500), scientific (GPQA), and programming (LiveCodeBench) reasoning benchmarks. Our findings reveal that while lossless quantization can be achieved with W8A8 or W4A16 quantization, lower bit-widths introduce significant accuracy risks. We further identify model size, model origin, and task difficulty as critical determinants of performance. Contrary to expectations, quantized models do not exhibit increased output lengths. In addition, strategically scaling the model sizes or reasoning steps can effectively enhance the performance. All quantized models and codes will be open-sourced in this https URL. 

**Abstract (ZH)**: 最近在推理语言模型方面的进展展示了在复杂任务中的出色性能，但其扩展的链式推理过程增加了推理开销。虽然量化已被广泛采用以减少大型语言模型的推理成本，但其对推理模型的影响仍研究不足。在这项研究中，我们首次系统地研究了量化推理模型，评估了从1.5B到70B参数的开源DeepSeek-R1-Distilled Qwen和LLaMA家族，以及QwQ-32B。我们的研究涵盖了使用最先进的算法在不同位宽下进行权重、KV缓存和激活量化，并且在数学（AIME，MATH-500）、科学（GPQA）和编程（LiveCodeBench）推理基准测试中进行了广泛的评估。我们的研究发现，虽然可以在W8A8或W4A16量化中实现无损量化，较低的位宽会引入显著的准确性风险。我们进一步确定了模型大小、模型来源和任务难度是影响性能的关键因素。与预期相反，量化模型并没有表现出增加的输出长度。此外，战略性地调整模型大小或推理步骤可以有效提升性能。所有量化模型和代码将在此处开放源代码：https://。 

---
# A Customized SAT-based Solver for Graph Coloring 

**Title (ZH)**: 基于 SAT 的定制化图着色求解器 

**Authors**: Timo Brand, Daniel Faber, Stephan Held, Petra Mutzel  

**Link**: [PDF](https://arxiv.org/pdf/2504.04821)  

**Abstract**: We introduce ZykovColor, a novel SAT-based algorithm to solve the graph coloring problem working on top of an encoding that mimics the Zykov tree. Our method is based on an approach of Hébrard and Katsirelos (2020) that employs a propagator to enforce transitivity constraints, incorporate lower bounds for search tree pruning, and enable inferred propagations. We leverage the recently introduced IPASIR-UP interface for CaDiCal to implement these techniques with a SAT solver. Furthermore, we propose new features that take advantage of the underlying SAT solver. These include modifying the integrated decision strategy with vertex domination hints and using incremental bottom-up search that allows to reuse learned clauses from previous calls. Additionally, we integrate a more efficient clique computation to improve the lower bounds during the search. We validate the effectiveness of each new feature through an experimental analysis. ZykovColor outperforms other state-of-the-art graph coloring implementations on the DIMACS benchmark set. Further experiments on random Erdős-Rényi graphs show that our new approach dominates state-of-the-art SAT-based methods for both very sparse and highly dense graphs. 

**Abstract (ZH)**: ZykovColor：一种基于 SAT 的新型图着色算法及其在图着色问题上的应用 

---
# ELT-Bench: An End-to-End Benchmark for Evaluating AI Agents on ELT Pipelines 

**Title (ZH)**: ELT-Bench: 一个评估AI代理在ELT管道上性能的端到端基准测试 

**Authors**: Tengjun Jin, Yuxuan Zhu, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04808)  

**Abstract**: Practitioners are increasingly turning to Extract-Load-Transform (ELT) pipelines with the widespread adoption of cloud data warehouses. However, designing these pipelines often involves significant manual work to ensure correctness. Recent advances in AI-based methods, which have shown strong capabilities in data tasks, such as text-to-SQL, present an opportunity to alleviate manual efforts in developing ELT pipelines. Unfortunately, current benchmarks in data engineering only evaluate isolated tasks, such as using data tools and writing data transformation queries, leaving a significant gap in evaluating AI agents for generating end-to-end ELT pipelines.
To fill this gap, we introduce ELT-Bench, an end-to-end benchmark designed to assess the capabilities of AI agents to build ELT pipelines. ELT-Bench consists of 100 pipelines, including 835 source tables and 203 data models across various domains. By simulating realistic scenarios involving the integration of diverse data sources and the use of popular data tools, ELT-Bench evaluates AI agents' abilities in handling complex data engineering workflows. AI agents must interact with databases and data tools, write code and SQL queries, and orchestrate every pipeline stage. We evaluate two representative code agent frameworks, Spider-Agent and SWE-Agent, using six popular Large Language Models (LLMs) on ELT-Bench. The highest-performing agent, Spider-Agent Claude-3.7-Sonnet with extended thinking, correctly generates only 3.9% of data models, with an average cost of $4.30 and 89.3 steps per pipeline. Our experimental results demonstrate the challenges of ELT-Bench and highlight the need for a more advanced AI agent to reduce manual effort in ELT workflows. Our code and data are available at this https URL. 

**Abstract (ZH)**: 面向提取-加载-转换（ELT）管道的AI代理评估基准（ELT-Bench） 

---
# Dynamic Vision Mamba 

**Title (ZH)**: 动态视见矛头蛇 

**Authors**: Mengxuan Wu, Zekai Li, Zhiyuan Liang, Moyang Li, Xuanlei Zhao, Samir Khaki, Zheng Zhu, Xiaojiang Peng, Konstantinos N. Plataniotis, Kai Wang, Wangbo Zhao, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2504.04787)  

**Abstract**: Mamba-based vision models have gained extensive attention as a result of being computationally more efficient than attention-based models. However, spatial redundancy still exists in these models, represented by token and block redundancy. For token redundancy, we analytically find that early token pruning methods will result in inconsistency between training and inference or introduce extra computation for inference. Therefore, we customize token pruning to fit the Mamba structure by rearranging the pruned sequence before feeding it into the next Mamba block. For block redundancy, we allow each image to select SSM blocks dynamically based on an empirical observation that the inference speed of Mamba-based vision models is largely affected by the number of SSM blocks. Our proposed method, Dynamic Vision Mamba (DyVM), effectively reduces FLOPs with minor performance drops. We achieve a reduction of 35.2\% FLOPs with only a loss of accuracy of 1.7\% on Vim-S. It also generalizes well across different Mamba vision model architectures and different vision tasks. Our code will be made public. 

**Abstract (ZH)**: 基于Mamba的视觉模型由于具有比基于注意力的模型更高效的计算特性而获得了广泛关注。然而，这些模型仍存在空间冗余，表现为令牌和块冗余。对于令牌冗余，我们分析发现，早期的令牌修剪方法会导致训练和推断之间不一致或在推断过程中增加额外的计算量。因此，我们通过在输送到下一个Mamba块之前重新排列修剪序列，定制了令牌修剪以适应Mamba结构。对于块冗余，我们允许每张图像根据Mamba基于视觉模型的推断速度主要受SSM块数量影响的经验观察，动态选择SSM块。我们提出的方法，动态视觉Mamba（DyVM），能有效地减少FLOPs，同时仅产生轻微的性能下降。我们在Vim-S上实现了35.2%的FLOPs减少，准确率下降了1.7%。该方法还能够在不同Mamba视觉模型架构和不同视觉任务上表现出良好的泛化能力。我们的代码将公开发布。 

---
# Bidirectional Hierarchical Protein Multi-Modal Representation Learning 

**Title (ZH)**: 双向层次蛋白多模态表示学习 

**Authors**: Xuefeng Liu, Songhao Jiang, Chih-chan Tien, Jinbo Xu, Rick Stevens  

**Link**: [PDF](https://arxiv.org/pdf/2504.04770)  

**Abstract**: Protein representation learning is critical for numerous biological tasks. Recently, large transformer-based protein language models (pLMs) pretrained on large scale protein sequences have demonstrated significant success in sequence-based tasks. However, pLMs lack structural information. Conversely, graph neural networks (GNNs) designed to leverage 3D structural information have shown promising generalization in protein-related prediction tasks, but their effectiveness is often constrained by the scarcity of labeled structural data. Recognizing that sequence and structural representations are complementary perspectives of the same protein entity, we propose a multimodal bidirectional hierarchical fusion framework to effectively merge these modalities. Our framework employs attention and gating mechanisms to enable effective interaction between pLMs-generated sequential representations and GNN-extracted structural features, improving information exchange and enhancement across layers of the neural network. Based on the framework, we further introduce local Bi-Hierarchical Fusion with gating and global Bi-Hierarchical Fusion with multihead self-attention approaches. Through extensive experiments on a diverse set of protein-related tasks, our method demonstrates consistent improvements over strong baselines and existing fusion techniques in a variety of protein representation learning benchmarks, including react (enzyme/EC classification), model quality assessment (MQA), protein-ligand binding affinity prediction (LBA), protein-protein binding site prediction (PPBS), and B cell epitopes prediction (BCEs). Our method establishes a new state-of-the-art for multimodal protein representation learning, emphasizing the efficacy of BIHIERARCHICAL FUSION in bridging sequence and structural modalities. 

**Abstract (ZH)**: 蛋白质表征学习对于众多生物任务至关重要。最近，基于大规模蛋白质序列预训练的大规模变压器蛋白语言模型（pLMs）在基于序列的任务中表现出显著的成功。然而，pLMs缺乏结构信息。相反，设计用于利用三维结构信息的图神经网络（GNNs）在蛋白质相关预测任务中显示出有前景的泛化能力，但其效果往往受限于标注结构数据的稀疏性。鉴于序列和结构表示是同一种蛋白质实体的互补视角，我们提出了一种多模态双向层级融合框架以有效整合这些模态。该框架采用注意力和门控机制，使pLMs生成的序列表示与GNN提取的结构特征之间能够有效互动，从而在神经网络的多个层面上提高信息的交换和增强。基于该框架，我们进一步引入了带有门控的局部双向层级融合方法和带有多头自注意力的全局双向层级融合方法。通过在一系列蛋白质相关任务上的广泛实验，我们的方法在多种蛋白质表征学习基准测试中均表现出对强基线和现有融合技术的一致改进，包括反应（酶/EC分类）、模型质量评估（MQA）、蛋白-配体结合亲和力预测（LBA）、蛋白-蛋白结合位点预测（PPBS）和B细胞表位预测（BCEs）。我们的方法为多模态蛋白质表征学习设立了新的前沿，强调了BIHIERARCHICAL FUSION在弥合序列和结构模态方面的有效性。 

---
# KunPeng: A Global Ocean Environmental Model 

**Title (ZH)**: KunPeng: 全球海洋环境模型 

**Authors**: Yi Zhao, Jiaqi Li, Haitao Xia, Tianjiao Zhang, Zerong Zeng, Tianyu Ren, Yucheng Zhang, Chao Zhu, Shengtong Xu, Hongchun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.04766)  

**Abstract**: Inspired by the similarity of the atmosphere-ocean physical coupling mechanism, this study innovatively migrates meteorological large-model techniques to the ocean domain, constructing the KunPeng global ocean environmental prediction model. Aimed at the discontinuous characteristics of marine space, we propose a terrain-adaptive mask constraint mechanism to mitigate effectively training divergence caused by abrupt gradients at land-sea boundaries. To fully integrate far-, medium-, and close-range marine features, a longitude-cyclic deformable convolution network (LC-DCN) is employed to enhance the dynamic receptive field, achieving refined modeling of multi-scale oceanic characteristics. A Deformable Convolution-enhanced Multi-Step Prediction module (DC-MTP) is employed to strengthen temporal dependency feature extraction capabilities. Experimental results demonstrate that this model achieves an average ACC of 0.80 in 15-day global predictions at 0.25$^\circ$ resolution, outperforming comparative models by 0.01-0.08. The average mean squared error (MSE) is 0.41 (representing a 5%-31% reduction) and the average mean absolute error (MAE) is 0.44 (0.6%-21% reduction) compared to other models. Significant improvements are particularly observed in sea surface parameter prediction, deep-sea region characterization, and current velocity field forecasting. Through a horizontal comparison of the applicability of operators at different scales in the marine domain, this study reveals that local operators significantly outperform global operators under slow-varying oceanic processes, demonstrating the effectiveness of dynamic feature pyramid representations in predicting marine physical parameters. 

**Abstract (ZH)**: 基于大气-海洋物理耦合机制的创新迁移：面向海洋的 KunPeng 全球海洋环境预测模型 

---
# Enhancing Leaf Disease Classification Using GAT-GCN Hybrid Model 

**Title (ZH)**: 使用GAT-GCN混合模型增强叶片疾病分类 

**Authors**: Shyam Sundhar, Riya Sharma, Priyansh Maheshwari, Suvidha Rupesh Kumar, T. Sunil Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2504.04764)  

**Abstract**: Agriculture plays a critical role in the global economy, providing livelihoods and ensuring food security for billions. As innovative agricultural practices become more widespread, the risk of crop diseases has increased, highlighting the urgent need for efficient, low-intervention disease identification methods. This research presents a hybrid model combining Graph Attention Networks (GATs) and Graph Convolution Networks (GCNs) for leaf disease classification. GCNs have been widely used for learning from graph-structured data, and GATs enhance this by incorporating attention mechanisms to focus on the most important neighbors. The methodology integrates superpixel segmentation for efficient feature extraction, partitioning images into meaningful, homogeneous regions that better capture localized features. The authors have employed an edge augmentation technique to enhance the robustness of the model. The edge augmentation technique has introduced a significant degree of generalization in the detection capabilities of the model. To further optimize training, weight initialization techniques are applied. The hybrid model is evaluated against the individual performance of the GCN and GAT models and the hybrid model achieved a precision of 0.9822, recall of 0.9818, and F1-score of 0.9818 in apple leaf disease classification, a precision of 0.9746, recall of 0.9744, and F1-score of 0.9743 in potato leaf disease classification, and a precision of 0.8801, recall of 0.8801, and F1-score of 0.8799 in sugarcane leaf disease classification. These results demonstrate the robustness and performance of the model, suggesting its potential to support sustainable agricultural practices through precise and effective disease detection. This work is a small step towards reducing the loss of crops and hence supporting sustainable goals of zero hunger and life on land. 

**Abstract (ZH)**: 农业在全球经济中扮演着关键角色，为 billions 提供生计并确保粮食安全。随着创新农业生产实践的普及，作物病害的风险增加，强调了迫切需要高效、低干预的病害识别方法。本研究提出了一种结合图注意网络（GATs）和图卷积网络（GCNs）的混合模型，用于叶片病害分类。GCNs 广泛用于从结构化数据中学习，而 GATs 通过引入注意机制关注最重要的邻居从而增强了这一过程。该方法整合了超像素分割以实现高效的特征提取，将图像分区为更具意义的、同质的区域，更好地捕捉局部特征。作者采用了边增强技术以提高模型的鲁棒性。边增强技术显著增强了模型检测能力的一般性。为了进一步优化训练，应用了权重初始化技术。该混合模型在苹果叶片病害分类中的精度为 0.9822、召回率为 0.9818、F1 分数为 0.9818；在马铃薯叶片病害分类中的精度为 0.9746、召回率为 0.9744、F1 分数为 0.9743；在甘蔗叶片病害分类中的精度为 0.8801、召回率为 0.8801、F1 分数为 0.8799。这些结果展示了该模型的鲁棒性和性能，表明其在通过精确有效的病害检测支持可持续农业生产方面的潜力。这项工作是朝着减少作物损失和支持零饥饿和陆地生命可持续目标迈出的一个小步骤。 

---
# Unsupervised Estimation of Nonlinear Audio Effects: Comparing Diffusion-Based and Adversarial approaches 

**Title (ZH)**: 基于扩散和对抗方法的无监督非线性音频效果估计比较 

**Authors**: Eloi Moliner, Michal Švento, Alec Wright, Lauri Juvela, Pavel Rajmic, Vesa Välimäki  

**Link**: [PDF](https://arxiv.org/pdf/2504.04751)  

**Abstract**: Accurately estimating nonlinear audio effects without access to paired input-output signals remains a challenging this http URL work studies unsupervised probabilistic approaches for solving this task. We introduce a method, novel for this application, based on diffusion generative models for blind system identification, enabling the estimation of unknown nonlinear effects using black- and gray-box models. This study compares this method with a previously proposed adversarial approach, analyzing the performance of both methods under different parameterizations of the effect operator and varying lengths of available effected this http URL experiments on guitar distortion effects, we show that the diffusion-based approach provides more stable results and is less sensitive to data availability, while the adversarial approach is superior at estimating more pronounced distortion effects. Our findings contribute to the robust unsupervised blind estimation of audio effects, demonstrating the potential of diffusion models for system identification in music technology. 

**Abstract (ZH)**: 无需配对输入-输出信号准确估计非线性音频效果——基于扩散生成模型的无监督概率方法的研究 

---
# Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions 

**Title (ZH)**: 基于语言指令、视觉观察和交互的3D物体功能 grounding 

**Authors**: He Zhu, Quyu Kong, Kechun Xu, Xunlong Xia, Bing Deng, Jieping Ye, Rong Xiong, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04744)  

**Abstract**: Grounding 3D object affordance is a task that locates objects in 3D space where they can be manipulated, which links perception and action for embodied intelligence. For example, for an intelligent robot, it is necessary to accurately ground the affordance of an object and grasp it according to human instructions. In this paper, we introduce a novel task that grounds 3D object affordance based on language instructions, visual observations and interactions, which is inspired by cognitive science. We collect an Affordance Grounding dataset with Points, Images and Language instructions (AGPIL) to support the proposed task. In the 3D physical world, due to observation orientation, object rotation, or spatial occlusion, we can only get a partial observation of the object. So this dataset includes affordance estimations of objects from full-view, partial-view, and rotation-view perspectives. To accomplish this task, we propose LMAffordance3D, the first multi-modal, language-guided 3D affordance grounding network, which applies a vision-language model to fuse 2D and 3D spatial features with semantic features. Comprehensive experiments on AGPIL demonstrate the effectiveness and superiority of our method on this task, even in unseen experimental settings. Our project is available at this https URL. 

**Abstract (ZH)**: 基于语言指令、视觉观测和交互的地基三维物体功能性任务及AGPIL数据集 

---
# Enhancing Compositional Reasoning in Vision-Language Models with Synthetic Preference Data 

**Title (ZH)**: 增强视觉语言模型的组合推理能力：使用合成偏好数据 

**Authors**: Samarth Mishra, Kate Saenko, Venkatesh Saligrama  

**Link**: [PDF](https://arxiv.org/pdf/2504.04740)  

**Abstract**: Compositionality, or correctly recognizing scenes as compositions of atomic visual concepts, remains difficult for multimodal large language models (MLLMs). Even state of the art MLLMs such as GPT-4o can make mistakes in distinguishing compositions like "dog chasing cat" vs "cat chasing dog". While on Winoground, a benchmark for measuring such reasoning, MLLMs have made significant progress, they are still far from a human's performance. We show that compositional reasoning in these models can be improved by elucidating such concepts via data, where a model is trained to prefer the correct caption for an image over a close but incorrect one. We introduce SCRAMBLe: Synthetic Compositional Reasoning Augmentation of MLLMs with Binary preference Learning, an approach for preference tuning open-weight MLLMs on synthetic preference data generated in a fully automated manner from existing image-caption data. SCRAMBLe holistically improves these MLLMs' compositional reasoning capabilities which we can see through significant improvements across multiple vision language compositionality benchmarks, as well as smaller but significant improvements on general question answering tasks. As a sneak peek, SCRAMBLe tuned Molmo-7B model improves on Winoground from 49.5% to 54.8% (best reported to date), while improving by ~1% on more general visual question answering tasks. Code for SCRAMBLe along with tuned models and our synthetic training dataset is available at this https URL. 

**Abstract (ZH)**: 组成性认知，即正确识别场景为原子视觉概念的组合，对于多模态大型语言模型（MLLMs）来说仍然困难。即使是最先进的MLLMs如GPT-4o，在区分“狗追猫”和“猫追狗”这样的组合时也会出错。尽管在Winoground这一衡量此类推理能力的标准上，MLLMs取得了显著进步，但它们仍然远未达到人类的水平。我们表明，通过数据阐明这些概念可以改善这些模型的组成性推理能力，其中模型被训练为更倾向于选择与图像匹配的正确描述，而非接近但错误的描述。我们提出了SCRAMBLe：一种使用二元偏好学习在完全自动化生成的合成偏好数据集上对MLLMs进行合成组成推理增强的技术。SCRAMBLe整体提高了这些MLLMs的组成推理能力，我们在多个视觉语言组成性基准测试中看到了显著提高，并且在一般问题回答任务上也实现了较小但显著的进步。作为一种预览，SCRAMBLe调优的Molmo-7B模型在Winoground上的得分从49.5%提高到54.8%（迄今为止最好的成绩），在更一般视觉问答任务上提高了约1%。SCRAMBle的代码、调优后的模型和我们的合成训练数据集可在以下链接获取。 

---
# TathyaNyaya and FactLegalLlama: Advancing Factual Judgment Prediction and Explanation in the Indian Legal Context 

**Title (ZH)**: Tathya Nyaya和FactLegalLlama：在印度法律 contexts 中推动事实判断预测与解释的进步 

**Authors**: Shubham Kumar Nigam, Balaramamahanthi Deepak Patnaik, Shivam Mishra, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2504.04737)  

**Abstract**: In the landscape of Fact-based Judgment Prediction and Explanation (FJPE), reliance on factual data is essential for developing robust and realistic AI-driven decision-making tools. This paper introduces TathyaNyaya, the largest annotated dataset for FJPE tailored to the Indian legal context, encompassing judgments from the Supreme Court of India and various High Courts. Derived from the Hindi terms "Tathya" (fact) and "Nyaya" (justice), the TathyaNyaya dataset is uniquely designed to focus on factual statements rather than complete legal texts, reflecting real-world judicial processes where factual data drives outcomes. Complementing this dataset, we present FactLegalLlama, an instruction-tuned variant of the LLaMa-3-8B Large Language Model (LLM), optimized for generating high-quality explanations in FJPE tasks. Finetuned on the factual data in TathyaNyaya, FactLegalLlama integrates predictive accuracy with coherent, contextually relevant explanations, addressing the critical need for transparency and interpretability in AI-assisted legal systems. Our methodology combines transformers for binary judgment prediction with FactLegalLlama for explanation generation, creating a robust framework for advancing FJPE in the Indian legal domain. TathyaNyaya not only surpasses existing datasets in scale and diversity but also establishes a benchmark for building explainable AI systems in legal analysis. The findings underscore the importance of factual precision and domain-specific tuning in enhancing predictive performance and interpretability, positioning TathyaNyaya and FactLegalLlama as foundational resources for AI-assisted legal decision-making. 

**Abstract (ZH)**: 基于事实判断预测与解释(FJPE)的景观中，依赖事实数据对于开发稳健且现实的AI驱动决策工具至关重要。本文介绍TathyaNyaya，这是专门为印度法律情境设计的最⼤标注数据集，涵盖了印度最高法院和各地⾼级法院的判决。TathyaNyaya数据集源自印地语术语“Tathya”（事实）和“Nyaya”（正义），特别设计聚焦于事实陈述而非完整的法律文本，反映现实法律程序中事实数据驱动结果的现象。作为该数据集的补充，本文还 소개了FactLegalLlama，这是对LLaMa-3-8B大规模语言模型的指令微调变体，优化用于生成FJPE任务中的高质量解释。FactLegalLlama在TathyaNyaya的实证数据上进行微调，结合预测准确性与连贯、相关性的解释生成，解决了AI辅助法律系统中透明性和可解释性的关键需求。本研究方法结合使用变压器进行二元判决预测，并利用FactLegalLlama进行解释生成，构建了一个适用于印度法律领域的FJPE robust框架。TathyaNyaya不仅在规模和多样性上超过了现有数据集，还确立了构建可解释法律分析AI系统的基准。研究结果强调了事实精准度和领域特定调整在提高预测性能和可解释性中的重要性，将TathyaNyaya和FactLegalLlama定位为AI辅助法律决策的基础资源。 

---
# T1: Tool-integrated Self-verification for Test-time Compute Scaling in Small Language Models 

**Title (ZH)**: 工具整合的自我验证方法用于小型语言模型测试时的计算量缩放 

**Authors**: Minki Kang, Jongwon Jeong, Jaewoong Cho  

**Link**: [PDF](https://arxiv.org/pdf/2504.04718)  

**Abstract**: Recent studies have demonstrated that test-time compute scaling effectively improves the performance of small language models (sLMs). However, prior research has mainly examined test-time compute scaling with an additional larger model as a verifier, leaving self-verification by sLMs underexplored. In this work, we investigate whether sLMs can reliably self-verify their outputs under test-time scaling. We find that even with knowledge distillation from larger verifiers, sLMs struggle with verification tasks requiring memorization, such as numerical calculations and fact-checking. To address this limitation, we propose Tool-integrated self-verification (T1), which delegates memorization-heavy verification steps to external tools, such as a code interpreter. Our theoretical analysis shows that tool integration reduces memorization demands and improves test-time scaling performance. Experiments on the MATH benchmark demonstrate that, with T1, a Llama-3.2 1B model under test-time scaling outperforms the significantly larger Llama-3.1 8B model. Moreover, T1 generalizes effectively to both mathematical (MATH500) and multi-domain knowledge-intensive tasks (MMLU-Pro). Our findings highlight the potential of tool integration to substantially improve the self-verification abilities of sLMs. 

**Abstract (ZH)**: Recent Studies Have Demonstrated that Test-Time Compute Scaling Effectively Improves the Performance of Small Language Models (sLMs): Investigating Reliable Self-Verification Without Additional Larger Models 

---
# Beyond Single-Turn: A Survey on Multi-Turn Interactions with Large Language Models 

**Title (ZH)**: 超越单轮交互：大规模语言模型多轮交互综述 

**Authors**: Yubo Li, Xiaobin Shen, Xinyu Yao, Xueying Ding, Yidi Miao, Ramayya Krishnan, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2504.04717)  

**Abstract**: Recent advancements in large language models (LLMs) have revolutionized their ability to handle single-turn tasks, yet real-world applications demand sophisticated multi-turn interactions. This survey provides a comprehensive review of recent advancements in evaluating and enhancing multi-turn interactions in LLMs. Focusing on task-specific scenarios, from instruction following in diverse domains such as math and coding to complex conversational engagements in roleplay, healthcare, education, and even adversarial jailbreak settings, we systematically examine the challenges of maintaining context, coherence, fairness, and responsiveness over prolonged dialogues. The paper organizes current benchmarks and datasets into coherent categories that reflect the evolving landscape of multi-turn dialogue evaluation. In addition, we review a range of enhancement methodologies under multi-turn settings, including model-centric strategies (contextual learning, supervised fine-tuning, reinforcement learning, and new architectures), external integration approaches (memory-augmented, retrieval-based methods, and knowledge graph), and agent-based techniques for collaborative interactions. Finally, we discuss open challenges and propose future directions for research to further advance the robustness and effectiveness of multi-turn interactions in LLMs. Related resources and papers are available at this https URL. 

**Abstract (ZH)**: 近期大型语言模型在多轮交互方面的进展及其评估与增强综述 

---
# AdvKT: An Adversarial Multi-Step Training Framework for Knowledge Tracing 

**Title (ZH)**: AdvKT：一种对抗多步训练框架用于知识追踪 

**Authors**: Lingyue Fu, Ting Long, Jianghao Lin, Wei Xia, Xinyi Dai, Ruiming Tang, Yasheng Wang, Weinan Zhang, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04706)  

**Abstract**: Knowledge Tracing (KT) monitors students' knowledge states and simulates their responses to question sequences. Existing KT models typically follow a single-step training paradigm, which leads to discrepancies with the multi-step inference process required in real-world simulations, resulting in significant error accumulation. This accumulation of error, coupled with the issue of data sparsity, can substantially degrade the performance of recommendation models in the intelligent tutoring systems. To address these challenges, we propose a novel Adversarial Multi-Step Training Framework for Knowledge Tracing (AdvKT), which, for the first time, focuses on the multi-step KT task. More specifically, AdvKT leverages adversarial learning paradigm involving a generator and a discriminator. The generator mimics high-reward responses, effectively reducing error accumulation across multiple steps, while the discriminator provides feedback to generate synthetic data. Additionally, we design specialized data augmentation techniques to enrich the training data with realistic variations, ensuring that the model generalizes well even in scenarios with sparse data. Experiments conducted on four real-world datasets demonstrate the superiority of AdvKT over existing KT models, showcasing its ability to address both error accumulation and data sparsity issues effectively. 

**Abstract (ZH)**: 一种新的对抗多步训练框架用于知识追踪（AdvKT） 

---
# LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important 

**Title (ZH)**: LagKV: KV缓存中的滞后相关信息揭示了哪些tokens重要 

**Authors**: Manlai Liang, JiaMing Zhang, Xiong Li, Jinlong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.04704)  

**Abstract**: The increasing size of the Key-Value (KV) cache during the Large Language Models long-context inference is the main obstacle for its balance between the deployment cost and task accuracy. To reduce the KV cache size in such scenarios, most previous efforts leveraged on the attention weight to evict non-critical cache tokens. But there is a trade-off in those methods, they usually require major modifiation of the inference infrastructure and significant computation overhead. Base on the fact that the Large Lanuage models are autoregresssive models, we propose {\it LagKV}, a KV allocation strategy only relying on straight forward comparison among KV themself. It is a totally attention free method which offers easy integration to the main stream inference platform and comparable performance comparing to other complicated KV compression methods. Results on LongBench and PasskeyRetrieval show that, our approach achieves nearly zero loss when the ratio is $2\times$ and $\approx 90\%$ of the original model performance for $8\times$. Especially in the 64-digit passkey retrieval task, our mehod outperforms the attention weight based method $H_2O$ over $60\%$ with same compression ratios. Our code is available at \url{this https URL}. 

**Abstract (ZH)**: Large Language Models 中长上下文推理中 Key-Value 缓存规模的增加是其部署成本与任务精度之间平衡的主要障碍。为减少在此类场景中的 Key-Value 缓存规模，大多数先前的努力依赖于利用注意力权重来淘汰非关键缓存令牌。但这些方法通常会带来基础设施的重大修改和显著的计算开销。基于大型语言模型是自回归模型的事实，我们提出了一种仅依赖于 Key-Value 本身直接比较的 {\it LagKV} KV 分配策略。这是一种完全不依赖注意力的方法，易于集成到主流推理平台，并在压缩比相同的情况下提供与其他复杂 KV 压缩方法相当的性能。LongBench 和 PasskeyRetrieval 的结果表明，当压缩比为 $2\times$ 时，我们的方法几乎不损失性能；而在 $8\times$ 压缩比时，保持约 $90\%$ 的原始模型性能。特别是在 64 位密钥检索任务中，我们的方法在相同的压缩比下比基于注意力权重的方法 $H_2O$ 高出 $60\%$ 的性能。我们的代码可在 \url{this https URL} 获取。 

---
# Provable Failure of Language Models in Learning Majority Boolean Logic via Gradient Descent 

**Title (ZH)**: 语言模型通过梯度下降学习多数布尔逻辑的证明失败 

**Authors**: Bo Chen, Zhenmei Shi, Zhao Song, Jiahao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04702)  

**Abstract**: Recent advancements in Transformer-based architectures have led to impressive breakthroughs in natural language processing tasks, with models such as GPT-4, Claude, and Gemini demonstrating human-level reasoning abilities. However, despite their high performance, concerns remain about the inherent limitations of these models, especially when it comes to learning basic logical functions. While complexity-theoretic analyses indicate that Transformers can represent simple logic functions (e.g., $\mathsf{AND}$, $\mathsf{OR}$, and majority gates) by its nature of belonging to the $\mathsf{TC}^0$ class, these results assume ideal parameter settings and do not account for the constraints imposed by gradient descent-based training methods. In this work, we investigate whether Transformers can truly learn simple majority functions when trained using gradient-based methods. We focus on a simplified variant of the Transformer architecture and consider both $n=\mathrm{poly}(d)$ and $n=\exp(\Omega(d))$ number of training samples, where each sample is a $d$-size binary string paired with the output of a basic majority function. Our analysis demonstrates that even after $\mathrm{poly}(d)$ gradient queries, the generalization error of the Transformer model still remains substantially large, growing exponentially with $d$. This work highlights fundamental optimization challenges in training Transformers for the simplest logical reasoning tasks and provides new insights into their theoretical limitations. 

**Abstract (ZH)**: 基于变压器的架构近期取得了自然语言处理任务的显著突破，如GPT-4、Claude和Gemini等模型展示了类人的推理能力。然而，尽管这些模型表现出色，仍然存在着关于其固有限制的担忧，特别是在学习基本逻辑函数方面。虽然复杂性理论分析表明，变压器可以以其属于$\mathsf{TC}^0$类的性质来表示简单的逻辑函数（例如，$\mathsf{AND}$、$\mathsf{OR}$和多数门），但这些结果假设了理想参数设置，并未考虑到基于梯度下降的训练方法所施加的约束。在本工作中，我们探究在基于梯度的训练方法下，变压器是否真的能够学习简单的多数函数。我们集中研究了变压器架构的简化版本，并考虑了两种情形：训练样本数量分别为$N=\mathrm{poly}(d)$和$N=\exp(\Omega(d))$，每一组样本为一个$d$维度的二进制字符串及其所对应的简单多数函数输出。我们的分析表明，即使经过$\mathrm{poly}(d)$次梯度查询后，变压器模型的泛化误差仍然显著较大，随着$d$呈指数增长。本工作突显了在训练变压器处理最简单逻辑推理任务时的基本优化挑战，并提供了对其理论限制的新见解。 

---
# R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation 

**Title (ZH)**: R2Vul: 通过强化学习和结构化推理精炼学习软件漏洞分析 

**Authors**: Martin Weyssow, Chengran Yang, Junkai Chen, Yikun Li, Huihui Huang, Ratnadira Widyasari, Han Wei Ang, Frank Liauw, Eng Lieh Ouh, Lwin Khin Shar, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2504.04699)  

**Abstract**: Large language models (LLMs) have shown promising performance in software vulnerability detection (SVD), yet their reasoning capabilities remain unreliable. Existing approaches relying on chain-of-thought (CoT) struggle to provide relevant and actionable security assessments. Additionally, effective SVD requires not only generating coherent reasoning but also differentiating between well-founded and misleading yet plausible security assessments, an aspect overlooked in prior work. To this end, we introduce R2Vul, a novel approach that distills structured reasoning into small LLMs using reinforcement learning from AI feedback (RLAIF). Through RLAIF, R2Vul enables LLMs to produce structured, security-aware reasoning that is actionable and reliable while explicitly learning to distinguish valid assessments from misleading ones. We evaluate R2Vul across five languages against SAST tools, CoT, instruction tuning, and classification-based baselines. Our results show that R2Vul with structured reasoning distillation enables a 1.5B student LLM to rival larger models while improving generalization to out-of-distribution vulnerabilities. Beyond model improvements, we contribute a large-scale, multilingual preference dataset featuring structured reasoning to support future research in SVD. 

**Abstract (ZH)**: 大型语言模型（LLMs）在软件漏洞检测（SVD）中展现出了令人鼓舞的性能，但其推理能力仍不够可靠。现有依赖链式思考（CoT）的方法难以提供相关且可操作的安全评估。此外，有效的SVD不仅需要生成连贯的推理，还需要能够区分合理的安全评估和误导性的但貌似合理的安全评估，这是先前工作中的一个遗漏方面。为此，我们提出了一种名为R2Vul的新方法，该方法通过从AI反馈强化学习（RLAIF）提炼结构化推理至小型LLM。通过RLAIF，R2Vul使LLM能够产生结构化、安全意识强的推理，这种推理既可操作又可靠，同时明确学习如何区分合理的评估与误导性的评估。我们使用五种语言对R2Vul与SAST工具、CoT、指令调优和基于分类的基本方法进行了评估。结果显示，R2Vul通过结构化推理提炼使一个1.5B参数的学生LLM能够与更大模型竞争，并改进了对分布外漏洞的一般化能力。除了模型改进，我们还贡献了一个大规模的多语言偏好数据集，该数据集中包含结构化推理，以支持未来SVD研究。 

---
# Bridging Knowledge Gap Between Image Inpainting and Large-Area Visible Watermark Removal 

**Title (ZH)**: 填补图像修复与大面积可见水印移除之间的知识差距 

**Authors**: Yicheng Leng, Chaowei Fang, Junye Chen, Yixiang Fang, Sheng Li, Guanbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.04687)  

**Abstract**: Visible watermark removal which involves watermark cleaning and background content restoration is pivotal to evaluate the resilience of watermarks. Existing deep neural network (DNN)-based models still struggle with large-area watermarks and are overly dependent on the quality of watermark mask prediction. To overcome these challenges, we introduce a novel feature adapting framework that leverages the representation modeling capacity of a pre-trained image inpainting model. Our approach bridges the knowledge gap between image inpainting and watermark removal by fusing information of the residual background content beneath watermarks into the inpainting backbone model. We establish a dual-branch system to capture and embed features from the residual background content, which are merged into intermediate features of the inpainting backbone model via gated feature fusion modules. Moreover, for relieving the dependence on high-quality watermark masks, we introduce a new training paradigm by utilizing coarse watermark masks to guide the inference process. This contributes to a visible image removal model which is insensitive to the quality of watermark mask during testing. Extensive experiments on both a large-scale synthesized dataset and a real-world dataset demonstrate that our approach significantly outperforms existing state-of-the-art methods. The source code is available in the supplementary materials. 

**Abstract (ZH)**: 可见水印去除中的水印清理和背景内容恢复对于评估水印的鲁棒性至关重要。现有的基于深度神经网络（DNN）的模型在处理大面积水印时仍存在问题，并且过度依赖水印掩模预测的质量。为克服这些挑战，我们提出了一种新的特征自适应框架，利用预训练图像修复模型的表示建模能力。我们的方法通过融合水印下方残留背景内容的信息，弥合了图像修复与水印去除之间的知识差距。我们建立了一种双分支系统来捕捉和嵌入残留背景内容的特征，并通过门控特征融合模块将这些特征合并到修复骨干模型的中间特征中。此外，为了减轻对高质量水印掩模的依赖，我们提出了一种新的训练范式，利用粗糙的水印掩模来指导推断过程，从而在测试过程中对水印掩模的质量具有鲁棒性。在大规模合成数据集和真实世界数据集上的广泛实验表明，我们的方法显著优于现有最先进的方法。源代码可在附录材料中获取。 

---
# Dual Consistent Constraint via Disentangled Consistency and Complementarity for Multi-view Clustering 

**Title (ZH)**: 基于解耦一致性与互补性的多视图聚类中的双重一致约束 

**Authors**: Bo Li, Jing Yun  

**Link**: [PDF](https://arxiv.org/pdf/2504.04676)  

**Abstract**: Multi-view clustering can explore common semantics from multiple views and has received increasing attention in recent years. However, current methods focus on learning consistency in representation, neglecting the contribution of each view's complementarity aspect in representation learning. This limit poses a significant challenge in multi-view representation learning. This paper proposes a novel multi-view clustering framework that introduces a disentangled variational autoencoder that separates multi-view into shared and private information, i.e., consistency and complementarity information. We first learn informative and consistent representations by maximizing mutual information across different views through contrastive learning. This process will ignore complementary information. Then, we employ consistency inference constraints to explicitly utilize complementary information when attempting to seek the consistency of shared information across all views. Specifically, we perform a within-reconstruction using the private and shared information of each view and a cross-reconstruction using the shared information of all views. The dual consistency constraints are not only effective in improving the representation quality of data but also easy to extend to other scenarios, especially in complex multi-view scenes. This could be the first attempt to employ dual consistent constraint in a unified MVC theoretical framework. During the training procedure, the consistency and complementarity features are jointly optimized. Extensive experiments show that our method outperforms baseline methods. 

**Abstract (ZH)**: 多视角聚类可以从多个视角中探索共有的语义，并在近年来受到了越来越多的关注。然而，当前的方法主要关注表示的一致性，忽视了每个视角在表示学习中互补性的贡献。这一限制在多视角表示学习中提出了重大挑战。本文提出了一种新颖的多视角聚类框架，引入了解码器分离多视角中的共享信息和私人信息，即一致性信息和互补性信息。我们首先通过对比学习最大化不同视角之间的互信息来学习具有信息性和一致性的表示，这一过程会忽略互补性信息。然后，我们采用一致性推断约束在尝试在所有视角中寻求共享信息的一致性时显式地利用互补性信息。具体地，我们利用每种视角的私人信息和共享信息进行内部重构，并利用所有视角的共享信息进行跨视角重构。双重一致性约束不仅在提高数据表示质量方面有效，而且易于扩展到其他场景，特别是在复杂的多视角场景中。这可能是首次在统一的多视角聚类理论框架中采用双重一致性约束的尝试。在训练过程中，一致性特征和互补性特征联合优化。广泛的实验表明，我们的方法优于基线方法。 

---
# EquiCPI: SE(3)-Equivariant Geometric Deep Learning for Structure-Aware Prediction of Compound-Protein Interactions 

**Title (ZH)**: EquiCPI: SE(3)不变几何深度学习在化合物-蛋白质相互作用结构感知预测中的应用 

**Authors**: Ngoc-Quang Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2504.04654)  

**Abstract**: Accurate prediction of compound-protein interactions (CPI) remains a cornerstone challenge in computational drug discovery. While existing sequence-based approaches leverage molecular fingerprints or graph representations, they critically overlook three-dimensional (3D) structural determinants of binding affinity. To bridge this gap, we present EquiCPI, an end-to-end geometric deep learning framework that synergizes first-principles structural modeling with SE(3)-equivariant neural networks. Our pipeline transforms raw sequences into 3D atomic coordinates via ESMFold for proteins and DiffDock-L for ligands, followed by physics-guided conformer re-ranking and equivariant feature learning. At its core, EquiCPI employs SE(3)-equivariant message passing over atomic point clouds, preserving symmetry under rotations, translations, and reflections, while hierarchically encoding local interaction patterns through tensor products of spherical harmonics. The proposed model is evaluated on BindingDB (affinity prediction) and DUD-E (virtual screening), EquiCPI achieves performance on par with or exceeding the state-of-the-art deep learning competitors. 

**Abstract (ZH)**: 准确预测化合物-蛋白质相互作用（CPI）仍然是计算药物发现中的一个核心挑战。为了弥合这一差距，我们提出EquiCPI，这是一种端到端的几何深度学习框架，结合了第一性原理结构建模和SE(3)-不变神经网络。我们的管道通过ESMFold将原始序列转换为蛋白质的3D原子坐标，并通过DiffDock-L将配体转换为3D原子坐标，随后进行基于物理的构象重新排序和不变特征学习。核心上，EquiCPI 使用SE(3)-不变的消息传递在网络点云上，保持旋转、平移和镜像不变性，并通过球谐函数的张量积逐级编码局部相互作用模式。所提出模型在BindingDB（亲和力预测）和DUD-E（虚拟筛选）上进行评估，EquiCPI 的性能与或超过最先进的深度学习竞争对手。 

---
# Here Comes the Explanation: A Shapley Perspective on Multi-contrast Medical Image Segmentation 

**Title (ZH)**: Here Comes the Explanation: 从Shapley值视角看多对比医学图像分割 

**Authors**: Tianyi Ren, Juampablo Heras Rivera, Hitender Oswal, Yutong Pan, Agamdeep Chopra, Jacob Ruzevick, Mehmet Kurt  

**Link**: [PDF](https://arxiv.org/pdf/2504.04645)  

**Abstract**: Deep learning has been successfully applied to medical image segmentation, enabling accurate identification of regions of interest such as organs and lesions. This approach works effectively across diverse datasets, including those with single-image contrast, multi-contrast, and multimodal imaging data. To improve human understanding of these black-box models, there is a growing need for Explainable AI (XAI) techniques for model transparency and accountability. Previous research has primarily focused on post hoc pixel-level explanations, using methods gradient-based and perturbation-based apporaches. These methods rely on gradients or perturbations to explain model predictions. However, these pixel-level explanations often struggle with the complexity inherent in multi-contrast magnetic resonance imaging (MRI) segmentation tasks, and the sparsely distributed explanations have limited clinical relevance. In this study, we propose using contrast-level Shapley values to explain state-of-the-art models trained on standard metrics used in brain tumor segmentation. Our results demonstrate that Shapley analysis provides valuable insights into different models' behavior used for tumor segmentation. We demonstrated a bias for U-Net towards over-weighing T1-contrast and FLAIR, while Swin-UNETR provided a cross-contrast understanding with balanced Shapley distribution. 

**Abstract (ZH)**: 深度学习已在医学图像分割中成功应用，能够准确识别如器官和病灶等区域。该方法在单对比度图像、多对比度图像和多模态成像数据等多样化的数据集上均能有效工作。为了提高对这些黑箱模型的人类理解，需要可解释人工智能（XAI）技术以增加模型的透明度和责任感。前期研究主要关注后验像素级解释，使用基于梯度和扰动的方法。然而，这些像素级解释在多对比度磁共振成像（MRI）分割任务的复杂性面前效果有限，且稀疏的解释缺乏临床意义。本研究提出使用对比度级Shapley值来解释基于标准脑肿瘤分割指标训练的最先进模型。研究结果表明，Shapley分析为不同模型在肿瘤分割中的行为提供了有价值的见解。我们发现U-Net倾向于过度重视T1对比度和FLAIR，而Swin-UNETR则提供了跨对比度的理解并具有平衡的Shapley分布。 

---
# Splits! A Flexible Dataset for Evaluating a Model's Demographic Social Inference 

**Title (ZH)**: Splits！一个灵活的数据集，用于评估模型的_demographic和社会推理能力 

**Authors**: Eylon Caplan, Tania Chakraborty, Dan Goldwasser  

**Link**: [PDF](https://arxiv.org/pdf/2504.04640)  

**Abstract**: Understanding how people of various demographics think, feel, and express themselves (collectively called group expression) is essential for social science and underlies the assessment of bias in Large Language Models (LLMs). While LLMs can effectively summarize group expression when provided with empirical examples, coming up with generalizable theories of how a group's expression manifests in real-world text is challenging. In this paper, we define a new task called Group Theorization, in which a system must write theories that differentiate expression across demographic groups. We make available a large dataset on this task, Splits!, constructed by splitting Reddit posts by neutral topics (e.g. sports, cooking, and movies) and by demographics (e.g. occupation, religion, and race). Finally, we suggest a simple evaluation framework for assessing how effectively a method can generate 'better' theories about group expression, backed by human validation. We publicly release the raw corpora and evaluation scripts for Splits! to help researchers assess how methods infer--and potentially misrepresent--group differences in expression. We make Splits! and our evaluation module available at this https URL. 

**Abstract (ZH)**: 理解不同人口统计数据群体在思考、感受和表达自己（统称为群体表达）方面的差异对于社会科学至关重要，并且是评估大型语言模型偏见的基础。尽管大型语言模型在提供实证例子时能够有效总结群体表达，但在现实中如何表现群体表达的具体机制仍然具有挑战性。在本文中，我们定义了一个新任务，称为群体理论化，要求系统撰写能够区分不同人口统计数据群体之间表达的理论。我们提供了一个大型数据集Splits!，该数据集通过按中性话题（如体育、烹饪和电影）和人口统计数据（如职业、宗教和种族）拆分Reddit帖子构建而成。最后，我们提出了一个简单的评估框架，以评估方法生成有关群体表达“更好”理论的能力，并得到人类验证的支持。我们公开发布了Splits!的原始语料库和评估脚本，以帮助研究人员评估方法如何推断以及可能错误地代表群体表达差异。Splits!及其评估模块可在以下链接获取：this https URL。 

---
# DanceMosaic: High-Fidelity Dance Generation with Multimodal Editability 

**Title (ZH)**: 舞动拼图：具备多模态可编辑性的高保真舞蹈生成 

**Authors**: Foram Niravbhai Shah, Parshwa Shah, Muhammad Usama Saleem, Ekkasit Pinyoanuntapong, Pu Wang, Hongfei Xue, Ahmed Helmy  

**Link**: [PDF](https://arxiv.org/pdf/2504.04634)  

**Abstract**: Recent advances in dance generation have enabled automatic synthesis of 3D dance motions. However, existing methods still struggle to produce high-fidelity dance sequences that simultaneously deliver exceptional realism, precise dance-music synchronization, high motion diversity, and physical plausibility. Moreover, existing methods lack the flexibility to edit dance sequences according to diverse guidance signals, such as musical prompts, pose constraints, action labels, and genre descriptions, significantly restricting their creative utility and adaptability. Unlike the existing approaches, DanceMosaic enables fast and high-fidelity dance generation, while allowing multimodal motion editing. Specifically, we propose a multimodal masked motion model that fuses the text-to-motion model with music and pose adapters to learn probabilistic mapping from diverse guidance signals to high-quality dance motion sequences via progressive generative masking training. To further enhance the motion generation quality, we propose multimodal classifier-free guidance and inference-time optimization mechanism that further enforce the alignment between the generated motions and the multimodal guidance. Extensive experiments demonstrate that our method establishes a new state-of-the-art performance in dance generation, significantly advancing the quality and editability achieved by existing approaches. 

**Abstract (ZH)**: Recent advances in舞蹈生成已有能力实现自动合成3D舞蹈动作。然而，现有方法仍然难以同时生成高度逼真的舞蹈序列、精确的舞蹈-音乐同步、高动作多样性以及物理可行性。此外，现有方法缺乏根据多元指导信号（如音乐提示、姿态约束、动作标签和风格描述）编辑舞蹈序列的灵活性，严重限制了它们的创意应用和适应性。与现有方法不同，DanceMosaic能够在实现快速和高保真舞蹈生成的同时允许多模态动作编辑。具体而言，我们提出了一种多模态遮罩动作模型，将文本到动作模型与音乐和姿态适配器融合，通过逐步生成遮罩训练学习从多元指导信号到高质量舞蹈动作序列的概率映射。为进一步提升动作生成质量，我们提出了一种多模态无分类器指导和推理时优化机制，进一步加强了生成动作与多元指导的一致性。广泛实验表明，我们的方法在舞蹈生成中建立了新的最先进性能，显著提高了现有方法实现的生成质量和编辑性。 

---
# M2IV: Towards Efficient and Fine-grained Multimodal In-Context Learning in Large Vision-Language Models 

**Title (ZH)**: M2IV: 向更高效细粒度多模态在上下文学习的大规模视觉-语言模型方向 

**Authors**: Yanshu Li, Hongyang He, Yi Cao, Qisen Cheng, Xiang Fu, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04633)  

**Abstract**: Multimodal in-context learning (ICL) is a vital capability for Large Vision-Language Models (LVLMs), allowing task adaptation via contextual prompts without parameter retraining. However, its application is hindered by the token-intensive nature of inputs and the high complexity of cross-modal few-shot learning, which limits the expressive power of representation methods. To tackle these challenges, we propose \textbf{M2IV}, a method that substitutes explicit demonstrations with learnable \textbf{I}n-context \textbf{V}ectors directly integrated into LVLMs. By exploiting the complementary strengths of multi-head attention (\textbf{M}HA) and multi-layer perceptrons (\textbf{M}LP), M2IV achieves robust cross-modal fidelity and fine-grained semantic distillation through training. This significantly enhances performance across diverse LVLMs and tasks and scales efficiently to many-shot scenarios, bypassing the context window limitations. We also introduce \textbf{VLibrary}, a repository for storing and retrieving M2IV, enabling flexible LVLM steering for tasks like cross-modal alignment, customized generation and safety improvement. Experiments across seven benchmarks and three LVLMs show that M2IV surpasses Vanilla ICL and prior representation engineering approaches, with an average accuracy gain of \textbf{3.74\%} over ICL with the same shot count, alongside substantial efficiency advantages. 

**Abstract (ZH)**: 多模态上下文学习（ICL）是大型视觉语言模型（LVLMs）的重要能力，允许通过上下文提示进行任务适应而不需重新训练参数。然而，其应用受限于输入的标记密集性质和跨模态少样本学习的高复杂性，这限制了表示方法的表现力。为应对这些挑战，我们提出了**M2IV**方法，该方法用可学习的**I**n-context **V**ectors直接替代显式的示例，将其直接集成到LVLMs中。通过利用多头注意力（**M**HA）和多层感知机（**M**LP）的互补优势，M2IV在训练过程中实现了稳健的跨模态保真度和精细的语义精炼，这显著提升了多样性LVLMs和任务下的性能，并且能够高效扩展到多样本场景，绕过了上下文窗口的限制。我们还介绍了**VLibrary**，一个用于存储和检索M2IV的仓库，使LVLM能够灵活地用于跨模态对齐、定制生成和安全性改进等任务。在七个基准和三种LVLM上的实验显示，M2IV超越了Vanilla ICL和之前的表示工程方法，平均准确率提高了**3.74%**，同时具备显著的效率优势。 

---
# Tool-as-Interface: Learning Robot Policies from Human Tool Usage through Imitation Learning 

**Title (ZH)**: 工具即界面：通过imitation learning学习机器人从人类工具使用中获取策略 

**Authors**: Haonan Chen, Cheng Zhu, Yunzhu Li, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2504.04612)  

**Abstract**: Tool use is critical for enabling robots to perform complex real-world tasks, and leveraging human tool-use data can be instrumental for teaching robots. However, existing data collection methods like teleoperation are slow, prone to control delays, and unsuitable for dynamic tasks. In contrast, human natural data, where humans directly perform tasks with tools, offers natural, unstructured interactions that are both efficient and easy to collect. Building on the insight that humans and robots can share the same tools, we propose a framework to transfer tool-use knowledge from human data to robots. Using two RGB cameras, our method generates 3D reconstruction, applies Gaussian splatting for novel view augmentation, employs segmentation models to extract embodiment-agnostic observations, and leverages task-space tool-action representations to train visuomotor policies. We validate our approach on diverse real-world tasks, including meatball scooping, pan flipping, wine bottle balancing, and other complex tasks. Our method achieves a 71\% higher average success rate compared to diffusion policies trained with teleoperation data and reduces data collection time by 77\%, with some tasks solvable only by our framework. Compared to hand-held gripper, our method cuts data collection time by 41\%. Additionally, our method bridges the embodiment gap, improves robustness to variations in camera viewpoints and robot configurations, and generalizes effectively across objects and spatial setups. 

**Abstract (ZH)**: 基于人类工具使用数据的机器人工具使用知识转移框架 

---
# "You just can't go around killing people" Explaining Agent Behavior to a Human Terminator 

**Title (ZH)**: “你就是不能随便杀人”：解释智能体行为给机械terminator看 

**Authors**: Uri Menkes, Assaf Hallak, Ofra Amir  

**Link**: [PDF](https://arxiv.org/pdf/2504.04592)  

**Abstract**: Consider a setting where a pre-trained agent is operating in an environment and a human operator can decide to temporarily terminate its operation and take-over for some duration of time. These kind of scenarios are common in human-machine interactions, for example in autonomous driving, factory automation and healthcare. In these settings, we typically observe a trade-off between two extreme cases -- if no take-overs are allowed, then the agent might employ a sub-optimal, possibly dangerous policy. Alternatively, if there are too many take-overs, then the human has no confidence in the agent, greatly limiting its usefulness. In this paper, we formalize this setup and propose an explainability scheme to help optimize the number of human interventions. 

**Abstract (ZH)**: 考虑一种场景：预训练代理在环境中运行，人类操作者可以决定暂时终止其运行并接管一段时间。这种类型的场景在人机交互中很常见，例如在自主驾驶、工厂自动化和医疗健康领域。在这种场景中，我们通常会观察到两种极端情况之间的权衡——如果不允许接管，代理可能会采用一个次优的、可能甚至是危险的策略。相反，如果接管次数太多，人类对代理的信任度会大幅降低，大大限制了其 usefulness。在本文中，我们对该设置进行形式化，并提出一种可解释性方案，以帮助优化人类干预的数量。 

---
# Your Image Generator Is Your New Private Dataset 

**Title (ZH)**: 您的图像生成器即是您的新私人数据集 

**Authors**: Nicolo Resmini, Eugenio Lomurno, Cristian Sbrolli, Matteo Matteucci  

**Link**: [PDF](https://arxiv.org/pdf/2504.04582)  

**Abstract**: Generative diffusion models have emerged as powerful tools to synthetically produce training data, offering potential solutions to data scarcity and reducing labelling costs for downstream supervised deep learning applications. However, effectively leveraging text-conditioned image generation for building classifier training sets requires addressing key issues: constructing informative textual prompts, adapting generative models to specific domains, and ensuring robust performance. This paper proposes the Text-Conditioned Knowledge Recycling (TCKR) pipeline to tackle these challenges. TCKR combines dynamic image captioning, parameter-efficient diffusion model fine-tuning, and Generative Knowledge Distillation techniques to create synthetic datasets tailored for image classification. The pipeline is rigorously evaluated on ten diverse image classification benchmarks. The results demonstrate that models trained solely on TCKR-generated data achieve classification accuracies on par with (and in several cases exceeding) models trained on real images. Furthermore, the evaluation reveals that these synthetic-data-trained models exhibit substantially enhanced privacy characteristics: their vulnerability to Membership Inference Attacks is significantly reduced, with the membership inference AUC lowered by 5.49 points on average compared to using real training data, demonstrating a substantial improvement in the performance-privacy trade-off. These findings indicate that high-fidelity synthetic data can effectively replace real data for training classifiers, yielding strong performance whilst simultaneously providing improved privacy protection as a valuable emergent property. The code and trained models are available in the accompanying open-source repository. 

**Abstract (ZH)**: 基于文本条件的生成扩散模型在图像分类训练集构建中的应用：Text-Conditioned Knowledge Recycling (TCKR) 管道的研究 

---
# Planning Safety Trajectories with Dual-Phase, Physics-Informed, and Transportation Knowledge-Driven Large Language Models 

**Title (ZH)**: 基于双阶段、物理启发和交通知识驱动的大语言模型规划安全轨迹 

**Authors**: Rui Gan, Pei Li, Keke Long, Bocheng An, Junwei You, Keshu Wu, Bin Ran  

**Link**: [PDF](https://arxiv.org/pdf/2504.04562)  

**Abstract**: Foundation models have demonstrated strong reasoning and generalization capabilities in driving-related tasks, including scene understanding, planning, and control. However, they still face challenges in hallucinations, uncertainty, and long inference latency. While existing foundation models have general knowledge of avoiding collisions, they often lack transportation-specific safety knowledge. To overcome these limitations, we introduce LetsPi, a physics-informed, dual-phase, knowledge-driven framework for safe, human-like trajectory planning. To prevent hallucinations and minimize uncertainty, this hybrid framework integrates Large Language Model (LLM) reasoning with physics-informed social force dynamics. LetsPi leverages the LLM to analyze driving scenes and historical information, providing appropriate parameters and target destinations (goals) for the social force model, which then generates the future trajectory. Moreover, the dual-phase architecture balances reasoning and computational efficiency through its Memory Collection phase and Fast Inference phase. The Memory Collection phase leverages the physics-informed LLM to process and refine planning results through reasoning, reflection, and memory modules, storing safe, high-quality driving experiences in a memory bank. Surrogate safety measures and physics-informed prompt techniques are introduced to enhance the LLM's knowledge of transportation safety and physical force, respectively. The Fast Inference phase extracts similar driving experiences as few-shot examples for new scenarios, while simplifying input-output requirements to enable rapid trajectory planning without compromising safety. Extensive experiments using the HighD dataset demonstrate that LetsPi outperforms baseline models across five safety this http URL PDF for project Github link. 

**Abstract (ZH)**: 基于物理的双阶段知识驱动框架LetςPi实现安全的人类似轨迹规划 

---
# Advancing Egocentric Video Question Answering with Multimodal Large Language Models 

**Title (ZH)**: 基于多模态大语言模型的自视点视频问答研究 

**Authors**: Alkesh Patel, Vibhav Chitalia, Yinfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04550)  

**Abstract**: Egocentric Video Question Answering (QA) requires models to handle long-horizon temporal reasoning, first-person perspectives, and specialized challenges like frequent camera movement. This paper systematically evaluates both proprietary and open-source Multimodal Large Language Models (MLLMs) on QaEgo4Dv2 - a refined dataset of egocentric videos derived from QaEgo4D. Four popular MLLMs (GPT-4o, Gemini-1.5-Pro, Video-LLaVa-7B and Qwen2-VL-7B-Instruct) are assessed using zero-shot and fine-tuned approaches for both OpenQA and CloseQA settings. We introduce QaEgo4Dv2 to mitigate annotation noise in QaEgo4D, enabling more reliable comparison. Our results show that fine-tuned Video-LLaVa-7B and Qwen2-VL-7B-Instruct achieve new state-of-the-art performance, surpassing previous benchmarks by up to +2.6% ROUGE/METEOR (for OpenQA) and +13% accuracy (for CloseQA). We also present a thorough error analysis, indicating the model's difficulty in spatial reasoning and fine-grained object recognition - key areas for future improvement. 

**Abstract (ZH)**: 自视点视频问答（QA）要求模型处理长时序推理、第一人称视角以及频繁的摄像头移动等专门挑战。本文系统性地评估了 proprietary 和开源的多模态大型语言模型（MLLMs）在 QaEgo4Dv2 数据集上的性能，该数据集是从 QaEgo4D 提取并精炼得到的自视点视频数据集。使用零样本和微调方法分别评估了四种流行的 MLLMs（GPT-4o、Gemini-1.5-Pro、Video-LLaVa-7B 和 Qwen2-VL-7B-Instruct）在开放式问答（OpenQA）和封闭式问答（CloseQA）设置下的表现。我们引入 QaEgo4Dv2 以减少 QaEgo4D 中的注释噪声，使比较更加可靠。结果显示，微调后的 Video-LLaVa-7B 和 Qwen2-VL-7B-Instruct 达到了新的最佳性能，分别在开放式问答中高出 +2.6% ROUGE/METEOR，在封闭式问答中高出 +13% 的准确率。我们还进行了详尽的错误分析，指出了模型在空间推理和精细物体识别方面的困难，这些是未来需要改进的关键领域。 

---
# The Point, the Vision and the Text: Does Point Cloud Boost Spatial Reasoning of Large Language Models? 

**Title (ZH)**: 点、视角与文本：点云是否提升大型语言模型的空间推理能力？ 

**Authors**: Weichen Zhang, Ruiying Peng, Chen Gao, Jianjie Fang, Xin Zeng, Kaiyuan Li, Ziyou Wang, Jinqiang Cui, Xin Wang, Xinlei Chen, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.04540)  

**Abstract**: 3D Large Language Models (LLMs) leveraging spatial information in point clouds for 3D spatial reasoning attract great attention. Despite some promising results, the role of point clouds in 3D spatial reasoning remains under-explored. In this work, we comprehensively evaluate and analyze these models to answer the research question: \textit{Does point cloud truly boost the spatial reasoning capacities of 3D LLMs?} We first evaluate the spatial reasoning capacity of LLMs with different input modalities by replacing the point cloud with the visual and text counterparts. We then propose a novel 3D QA (Question-answering) benchmark, ScanReQA, that comprehensively evaluates models' understanding of binary spatial relationships. Our findings reveal several critical insights: 1) LLMs without point input could even achieve competitive performance even in a zero-shot manner; 2) existing 3D LLMs struggle to comprehend the binary spatial relationships; 3) 3D LLMs exhibit limitations in exploiting the structural coordinates in point clouds for fine-grained spatial reasoning. We think these conclusions can help the next step of 3D LLMs and also offer insights for foundation models in other modalities. We release datasets and reproducible codes in the anonymous project page: this https URL. 

**Abstract (ZH)**: 三维大规模语言模型利用点云中的空间信息进行三维空间推理受到广泛关注。尽管取得了某些有前景的结果，点云在三维空间推理中的作用仍未得到充分探索。在本文中，我们全面评估和分析这些模型以回答研究问题：点云是否真正提升了三维语言模型的空间推理能力？我们首先通过用视觉和文本输入替换点云来评估具有不同输入模态的语言模型的空间推理能力。然后，我们提出了一种新的三维问答基准ScanReQA，全面评估模型对二元空间关系的理解。我们的发现揭示了几个关键洞察：1) 不含点云输入的语言模型即使在零样本方式下也能实现竞争力表现；2) 存在的三维语言模型在理解二元空间关系方面存在困难；3) 三维语言模型在利用点云中的结构坐标进行精细空间推理方面存在局限性。我们认为这些结论有助于下步三维语言模型的发展，并为其他模态的预训练模型提供见解。我们已在匿名项目页面释放了数据集和可复现代码：this https URL。 

---
# SnapPix: Efficient-Coding--Inspired In-Sensor Compression for Edge Vision 

**Title (ZH)**: SnapPix：受高效编码启发的边缘视觉传感器内压缩 

**Authors**: Weikai Lin, Tianrui Ma, Adith Boloor, Yu Feng, Ruofan Xing, Xuan Zhang, Yuhao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04535)  

**Abstract**: Energy-efficient image acquisition on the edge is crucial for enabling remote sensing applications where the sensor node has weak compute capabilities and must transmit data to a remote server/cloud for processing. To reduce the edge energy consumption, this paper proposes a sensor-algorithm co-designed system called SnapPix, which compresses raw pixels in the analog domain inside the sensor. We use coded exposure (CE) as the in-sensor compression strategy as it offers the flexibility to sample, i.e., selectively expose pixels, both spatially and temporally. SNAPPIX has three contributions. First, we propose a task-agnostic strategy to learn the sampling/exposure pattern based on the classic theory of efficient coding. Second, we co-design the downstream vision model with the exposure pattern to address the pixel-level non-uniformity unique to CE-compressed images. Finally, we propose lightweight augmentations to the image sensor hardware to support our in-sensor CE compression. Evaluating on action recognition and video reconstruction, SnapPix outperforms state-of-the-art video-based methods at the same speed while reducing the energy by up to 15.4x. We have open-sourced the code at: this https URL. 

**Abstract (ZH)**: 边缘节点上的能效图像获取对于传感器节点计算能力较弱且必须将数据传输到远程服务器/云端进行处理的远程 sensing 应用至关重要。为了减少边缘节点的能耗，本文提出了一种名为 SnapPix 的传感器-算法协同设计系统，在传感器内部的模拟域对原始像素进行压缩。我们使用编码曝光（CE）作为内部压缩策略，因为它可以在空间和时间上灵活地采样，即选择性地曝光像素。SnapPix 有三个贡献。首先，我们提出了一种任务无关的策略，基于高效的编码经典理论来学习采样/曝光模式。其次，我们与曝光模式共同设计下游的视觉模型，以解决 CE 压缩图像特有的像素级非均匀性问题。最后，我们提出了轻量级的图像传感器硬件增强措施，以支持我们的内部 CE 压缩。在动作识别和视频重建的评估中，SnapPix 在相同速度下优于最先进的基于视频的方法，能耗最多可降低 15.4 倍。我们已在以下链接开源了代码：this https URL。 

---
# An Empirical Comparison of Text Summarization: A Multi-Dimensional Evaluation of Large Language Models 

**Title (ZH)**: 大型语言模型在多维度上的文本摘要 empirical 对比研究 

**Authors**: Anantharaman Janakiraman, Behnaz Ghoraani  

**Link**: [PDF](https://arxiv.org/pdf/2504.04534)  

**Abstract**: Text summarization is crucial for mitigating information overload across domains like journalism, medicine, and business. This research evaluates summarization performance across 17 large language models (OpenAI, Google, Anthropic, open-source) using a novel multi-dimensional framework. We assessed models on seven diverse datasets (BigPatent, BillSum, CNN/DailyMail, PubMed, SAMSum, WikiHow, XSum) at three output lengths (50, 100, 150 tokens) using metrics for factual consistency, semantic similarity, lexical overlap, and human-like quality, while also considering efficiency factors. Our findings reveal significant performance differences, with specific models excelling in factual accuracy (deepseek-v3), human-like quality (claude-3-5-sonnet), and processing efficiency/cost-effectiveness (gemini-1.5-flash, gemini-2.0-flash). Performance varies dramatically by dataset, with models struggling on technical domains but performing well on conversational content. We identified a critical tension between factual consistency (best at 50 tokens) and perceived quality (best at 150 tokens). Our analysis provides evidence-based recommendations for different use cases, from high-stakes applications requiring factual accuracy to resource-constrained environments needing efficient processing. This comprehensive approach enhances evaluation methodology by integrating quality metrics with operational considerations, incorporating trade-offs between accuracy, efficiency, and cost-effectiveness to guide model selection for specific applications. 

**Abstract (ZH)**: 文本总结对于减轻 journalism、medicine 和 business 等领域信息过载至关重要。本研究使用新型多维度框架评估了 17 种大型语言模型（包括 OpenAI、Google、Anthropic 以及开源模型）的总结性能。我们在七个不同的数据集（BigPatent、BillSum、CNN/DailyMail、PubMed、SAMSum、WikiHow、XSum）上，针对三种输出长度（50、100、150 个词元）进行了评估，并使用事实一致性、语义相似度、词汇重叠和人类质量等指标进行评估，同时考虑了效率因素。研究发现不同模型在性能上存在显著差异，部分模型在事实准确性（deepseek-v3）、人类质量（claude-3-5-sonnet）以及处理效率/成本效益（gemini-1.5-flash、gemini-2.0-flash）方面表现出色。不同数据集上模型的表现差异显著，技术领域模型表现不佳，而对话内容表现良好。我们发现事实一致性（最佳表现为 50 个词元）和感知质量（最佳表现为 150 个词元）之间存在关键张力。我们的分析提供了基于证据的建议，适用于不同应用场景，从需要事实准确性的高风险应用到需要高效处理的资源受限环境。通过结合质量指标和运营考虑的全面方法，改进了评估方法，考虑了准确度、效率和成本效益之间的权衡，以指导特定应用中模型的选择。 

---
# A Consequentialist Critique of Binary Classification Evaluation Practices 

**Title (ZH)**: 功利主义对二元分类评估实践的批判 

**Authors**: Gerardo Flores, Abigail Schiff, Alyssa H. Smith, Julia A Fukuyama, Ashia C. Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2504.04528)  

**Abstract**: ML-supported decisions, such as ordering tests or determining preventive custody, often involve binary classification based on probabilistic forecasts. Evaluation frameworks for such forecasts typically consider whether to prioritize independent-decision metrics (e.g., Accuracy) or top-K metrics (e.g., Precision@K), and whether to focus on fixed thresholds or threshold-agnostic measures like AUC-ROC. We highlight that a consequentialist perspective, long advocated by decision theorists, should naturally favor evaluations that support independent decisions using a mixture of thresholds given their prevalence, such as Brier scores and Log loss. However, our empirical analysis reveals a strong preference for top-K metrics or fixed thresholds in evaluations at major conferences like ICML, FAccT, and CHIL. To address this gap, we use this decision-theoretic framework to map evaluation metrics to their optimal use cases, along with a Python package, briertools, to promote the broader adoption of Brier scores. In doing so, we also uncover new theoretical connections, including a reconciliation between the Brier Score and Decision Curve Analysis, which clarifies and responds to a longstanding critique by (Assel, et al. 2017) regarding the clinical utility of proper scoring rules. 

**Abstract (ZH)**: ML支持的决策，如下达测试指令或决定预防性拘留，通常基于概率预测进行二元分类。这些预测的评估框架通常会考虑是优先考虑独立决策指标（如准确率）还是 top-K 指标（如 Precision@K），以及是关注固定阈值还是阈值无关的指标（如 AUC-ROC）。我们认为，由决策理论长期倡导的结果主义视角，应自然倾向于采用混合阈值支持独立决策的评估，如 Brier 分数和对数损失。然而，我们的实证分析显示，在 ICML、FAccT 和 CHIL 等重要会议上，评估指标更偏好 top-K 指标或固定阈值。为解决这一差距，我们利用决策理论框架将评估指标映射到其最佳应用场景，并开发 Python 包 briertools 推动 Brier 分数的更广泛采用。在此过程中，我们还发现了新的理论联系，包括 Brier 分数和决策曲线分析之间的和解，澄清并回应了 Assel 等人（2017）对适当评分规则在临床应用中的长期批评。 

---
# Trust Region Preference Approximation: A simple and stable reinforcement learning algorithm for LLM reasoning 

**Title (ZH)**: 可信区域偏好逼近：一种用于大语言模型推理的简单且稳定的强化学习算法 

**Authors**: Xuerui Su, Shufang Xie, Guoqing Liu, Yingce Xia, Renqian Luo, Peiran Jin, Zhiming Ma, Yue Wang, Zun Wang, Yuting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04524)  

**Abstract**: Recently, Large Language Models (LLMs) have rapidly evolved, approaching Artificial General Intelligence (AGI) while benefiting from large-scale reinforcement learning to enhance Human Alignment (HA) and Reasoning. Recent reward-based optimization algorithms, such as Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO) have achieved significant performance on reasoning tasks, whereas preference-based optimization algorithms such as Direct Preference Optimization (DPO) significantly improve the performance of LLMs on human alignment. However, despite the strong performance of reward-based optimization methods in alignment tasks , they remain vulnerable to reward hacking. Furthermore, preference-based algorithms (such as Online DPO) haven't yet matched the performance of reward-based optimization algorithms (like PPO) on reasoning tasks, making their exploration in this specific area still a worthwhile pursuit. Motivated by these challenges, we propose the Trust Region Preference Approximation (TRPA) algorithm, which integrates rule-based optimization with preference-based optimization for reasoning tasks. As a preference-based algorithm, TRPA naturally eliminates the reward hacking issue. TRPA constructs preference levels using predefined rules, forms corresponding preference pairs, and leverages a novel optimization algorithm for RL training with a theoretical monotonic improvement guarantee. Experimental results demonstrate that TRPA not only achieves competitive performance on reasoning tasks but also exhibits robust stability. The code of this paper are released and updating on this https URL. 

**Abstract (ZH)**: 基于可信区域的偏好近似算法（TRPA）：结合规则优化与偏好优化以提升推理任务性能 

---
# Hessian of Perplexity for Large Language Models by PyTorch autograd (Open Source) 

**Title (ZH)**: 大型语言模型的困惑度海森矩阵通过PyTorch autograd（开源） 

**Authors**: Ivan Ilin  

**Link**: [PDF](https://arxiv.org/pdf/2504.04520)  

**Abstract**: Computing the full Hessian matrix -- the matrix of second-order derivatives for an entire Large Language Model (LLM) is infeasible due to its sheer size. In this technical report, we aim to provide a comprehensive guide on how to accurately compute at least a small portion of the Hessian for LLMs using PyTorch autograd library. We also demonstrate how to compute the full diagonal of the Hessian matrix using multiple samples of vector-Hessian Products (HVPs). We hope that both this guide and the accompanying GitHub code will be valuable resources for practitioners and researchers interested in better understanding the behavior and structure of the Hessian in LLMs. 

**Abstract (ZH)**: 计算全海森矩阵——由于大型语言模型（LLM）的海森矩阵规模庞大，完全计算整个海森矩阵不可行。在本技术报告中，我们旨在提供一个全面的指南，介绍如何使用PyTorch自动求导库准确计算大型语言模型的一部分海森矩阵。我们还将展示如何使用多个向量-海森矩阵乘积（HVPs）样本计算海森矩阵的完整对角线。希望本指南及其附带的GitHub代码能够对那些希望更深入理解大型语言模型海森矩阵的行为和结构的研究人员和实践者提供宝贵的资源。 

---
# Enhance Then Search: An Augmentation-Search Strategy with Foundation Models for Cross-Domain Few-Shot Object Detection 

**Title (ZH)**: 增强后再搜索：基于基础模型的跨域少样本对象检测增广-搜索策略 

**Authors**: Jiancheng Pan, Yanxing Liu, Xiao He, Long Peng, Jiahao Li, Yuze Sun, Xiaomeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04517)  

**Abstract**: Foundation models pretrained on extensive datasets, such as GroundingDINO and LAE-DINO, have performed remarkably in the cross-domain few-shot object detection (CD-FSOD) task. Through rigorous few-shot training, we found that the integration of image-based data augmentation techniques and grid-based sub-domain search strategy significantly enhances the performance of these foundation models. Building upon GroundingDINO, we employed several widely used image augmentation methods and established optimization objectives to effectively navigate the expansive domain space in search of optimal sub-domains. This approach facilitates efficient few-shot object detection and introduces an approach to solving the CD-FSOD problem by efficiently searching for the optimal parameter configuration from the foundation model. Our findings substantially advance the practical deployment of vision-language models in data-scarce environments, offering critical insights into optimizing their cross-domain generalization capabilities without labor-intensive retraining. Code is available at this https URL. 

**Abstract (ZH)**: 基础模型在大规模数据集上预训练，如GroundingDINO和LAE-DINO，在跨域少样本目标检测（CD-FSOD）任务中表现卓越。通过严格的少样本训练，我们发现基于图像的数据增强技术与基于网格的子域搜索策略的结合显著提升了这些基础模型的性能。基于GroundingDINO，我们采用了多种常用的图像增强方法并建立了优化目标，以有效探索广阔的数据域空间，寻找最优子域。该方法促进了高效的少样本目标检测，并提出了通过高效搜索基础模型的最佳参数配置来解决CD-FSOD问题的方法。我们的研究显著推动了在数据稀缺环境中视觉-语言模型的实际部署，并提供了关于优化其跨域泛化能力的重要洞见，而无需进行劳动密集型的重新训练。代码可在此处访问：this https URL。 

---
# Saliency-driven Dynamic Token Pruning for Large Language Models 

**Title (ZH)**: 基于显著性驱动的动态令牌剪枝大语言模型 

**Authors**: Yao Tao, Yehui Tang, Yun Wang, Mingjian Zhu, Hailin Hu, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04514)  

**Abstract**: Despite the recent success of large language models (LLMs), LLMs are particularly challenging in long-sequence inference scenarios due to the quadratic computational complexity of the attention mechanism. Inspired by the interpretability theory of feature attribution in neural network models, we observe that not all tokens have the same contribution. Based on this observation, we propose a novel token pruning framework, namely Saliency-driven Dynamic Token Pruning (SDTP), to gradually and dynamically prune redundant tokens based on the input context. Specifically, a lightweight saliency-driven prediction module is designed to estimate the importance score of each token with its hidden state, which is added to different layers of the LLM to hierarchically prune redundant tokens. Furthermore, a ranking-based optimization strategy is proposed to minimize the ranking divergence of the saliency score and the predicted importance score. Extensive experiments have shown that our framework is generalizable to various models and datasets. By hierarchically pruning 65\% of the input tokens, our method greatly reduces 33\% $\sim$ 47\% FLOPs and achieves speedup up to 1.75$\times$ during inference, while maintaining comparable performance. We further demonstrate that SDTP can be combined with KV cache compression method for further compression. 

**Abstract (ZH)**: 基于显著性驱动动态令牌剪枝的长序列推理优化 

---
# Statistical Guarantees Of False Discovery Rate In Medical Instance Segmentation Tasks Based on Conformal Risk Control 

**Title (ZH)**: 基于形验风险控制的医疗Instance分割任务中错误发现率的统计保证 

**Authors**: Mengxia Dai, Wenqian Luo, Tianyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.04482)  

**Abstract**: Instance segmentation plays a pivotal role in medical image analysis by enabling precise localization and delineation of lesions, tumors, and anatomical structures. Although deep learning models such as Mask R-CNN and BlendMask have achieved remarkable progress, their application in high-risk medical scenarios remains constrained by confidence calibration issues, which may lead to misdiagnosis. To address this challenge, we propose a robust quality control framework based on conformal prediction theory. This framework innovatively constructs a risk-aware dynamic threshold mechanism that adaptively adjusts segmentation decision boundaries according to clinical this http URL, we design a \textbf{calibration-aware loss function} that dynamically tunes the segmentation threshold based on a user-defined risk level $\alpha$. Utilizing exchangeable calibration data, this method ensures that the expected FNR or FDR on test data remains below $\alpha$ with high probability. The framework maintains compatibility with mainstream segmentation models (e.g., Mask R-CNN, BlendMask+ResNet-50-FPN) and datasets (PASCAL VOC format) without requiring architectural modifications. Empirical results demonstrate that we rigorously bound the FDR metric marginally over the test set via our developed calibration framework. 

**Abstract (ZH)**: 实例分割在医疗图像分析中发挥着关键作用，通过实现病变、肿瘤和解剖结构的精确定位和勾勒。尽管诸如Mask R-CNN和BlendMask等深度学习模型取得了显著进展，但在高风险医疗场景中的应用仍受限于置信度校准问题，这可能导致误诊。为应对这一挑战，我们提出了一种基于可构造预测理论的稳健质量控制框架。该框架创新性地构建了一种风险感知动态阈值机制，根据临床需求自适应调整分割决策边界。为此，我们设计了一种**置信度感知损失函数**，根据用户定义的风险水平$\alpha$动态调整分割阈值。利用可交换的校准数据，该方法确保在测试数据上的预期FNR或FDR低于$\alpha$的概率很高。该框架与主流的分割模型（如Mask R-CNN、BlendMask+ResNet-50-FPN）及数据集（PASCAL VOC格式）保持兼容，无需对架构进行修改。实验证明，我们通过开发的校准框架严格界定了测试集上的FDR指标。 

---
# Directed Graph-alignment Approach for Identification of Gaps in Short Answers 

**Title (ZH)**: 基于有向图对齐的方法在识别简答题中的知识缺口 

**Authors**: Archana Sahu, Plaban Kumar Bhowmick  

**Link**: [PDF](https://arxiv.org/pdf/2504.04473)  

**Abstract**: In this paper, we have presented a method for identifying missing items known as gaps in the student answers by comparing them against the corresponding model answer/reference answers, automatically. The gaps can be identified at word, phrase or sentence level. The identified gaps are useful in providing feedback to the students for formative assessment. The problem of gap identification has been modelled as an alignment of a pair of directed graphs representing a student answer and the corresponding model answer for a given question. To validate the proposed approach, the gap annotated student answers considering answers from three widely known datasets in the short answer grading domain, namely, University of North Texas (UNT), SciEntsBank, and Beetle have been developed and this gap annotated student answers' dataset is available at: this https URL. Evaluation metrics used in the traditional machine learning tasks have been adopted to evaluate the task of gap identification. Though performance of the proposed approach varies across the datasets and the types of the answers, overall the performance is observed to be promising. 

**Abstract (ZH)**: 本文提出了一种通过将学生答案与对应的参考答案进行自动对比以识别缺失项目（空白）的方法，这些缺失项目可以在单词、短语或句子级别进行识别。识别出的缺失项目对于提供形成性评估反馈非常有用。将缺失项目识别问题建模为一对有向图的对齐问题，这些有向图分别表示学生的回答和给定问题的参考答案。为了验证所提出的方法，针对短答案评分领域的三个广泛认可的数据集（德克萨斯大学北分校(UNT)、SciEntsBank 和 Beetle）构建了标注了缺失项目的学生成绩单数据集，该数据集可通过以下链接获取：this https URL。采用传统机器学习任务中使用的评价指标来评估缺失项目识别任务。尽管所提出方法在不同数据集和不同类型的答案上的性能有所差异，但总体上观察到其性能是有前景的。 

---
# AI2STOW: End-to-End Deep Reinforcement Learning to Construct Master Stowage Plans under Demand Uncertainty 

**Title (ZH)**: AI2STOW：在需求不确定性下基于端到端深度强化学习的主理货计划构建方法 

**Authors**: Jaike Van Twiller, Djordje Grbic, Rune Møller Jensen  

**Link**: [PDF](https://arxiv.org/pdf/2504.04469)  

**Abstract**: The worldwide economy and environmental sustainability depend on eff icient and reliable supply chains, in which container shipping plays a crucial role as an environmentally friendly mode of transport. Liner shipping companies seek to improve operational efficiency by solving the stowage planning problem. Due to many complex combinatorial aspects, stowage planning is challenging and often decomposed into two NP-hard subproblems: master and slot planning. This article proposes AI2STOW, an end-to-end deep reinforcement learning model with feasibility projection and an action mask to create master plans under demand uncertainty with global objectives and constraints, including paired block stowage patterms. Our experimental results demonstrate that AI2STOW outperforms baseline methods from reinforcement learning and stochastic programming in objective performance and computational efficiency, based on simulated instances reflecting the scale of realistic vessels and operational planning horizons. 

**Abstract (ZH)**: 世界经济和环境可持续性依赖于高效可靠的供应链，在此基础上，集装箱运输作为一种环保的运输方式发挥了关键作用。班轮公司通过解决积载规划问题来提高运营效率。由于存在许多复杂的组合因素，积载规划具有挑战性，通常被分解为两个NP难子问题：主积载和舱位规划。本文提出了一种名为AI2STOW的端到端深度强化学习模型，该模型通过可行性投影和动作掩码，在需求不确定性下创建具有全局目标和约束的主积载计划，包括成对的模块化积载模式。实验结果表明，与强化学习和随机规划基线方法相比，AI2STOW在目标性能和计算效率方面具有优势，基于模拟实例反映了现实船舶的规模和运营规划时间范围。 

---
# LoopGen: Training-Free Loopable Music Generation 

**Title (ZH)**: LoopGen: 无训练循环可支配音乐生成 

**Authors**: Davide Marincione, Giorgio Strano, Donato Crisostomi, Roberto Ribuoli, Emanuele Rodolà  

**Link**: [PDF](https://arxiv.org/pdf/2504.04466)  

**Abstract**: Loops--short audio segments designed for seamless repetition--are central to many music genres, particularly those rooted in dance and electronic styles. However, current generative music models struggle to produce truly loopable audio, as generating a short waveform alone does not guarantee a smooth transition from its endpoint back to its start, often resulting in audible this http URL--short audio segments designed for seamless repetition--are central to many music genres, particularly those rooted in dance and electronic styles. However, current generative music models struggle to produce truly loopable audio, as generating a short waveform alone does not guarantee a smooth transition from its endpoint back to its start, often resulting in audible this http URL address this gap by modifying a non-autoregressive model (MAGNeT) to generate tokens in a circular pattern, letting the model attend to the beginning of the audio when creating its ending. This inference-only approach results in generations that are aware of future context and loop naturally, without the need for any additional training or data. We evaluate the consistency of loop transitions by computing token perplexity around the seam of the loop, observing a 55% improvement. Blind listening tests further confirm significant perceptual gains over baseline methods, improving mean ratings by 70%. Taken together, these results highlight the effectiveness of inference-only approaches in improving generative models and underscore the advantages of non-autoregressive methods for context-aware music generation. 

**Abstract (ZH)**: 闭环——为无缝重复设计的短音频片段——在许多音乐流派中占据核心地位，尤其是那些源于舞曲和电子风格的流派。然而，当前的生成音乐模型在生成真正可循环的音频方面面临困难，仅仅生成一段短暂的波形并不能保证从其终点平滑过渡回其起点，经常会产生可听的断层。通过修改非自回归模型（MAGNeT）以生成循环模式的标记，并让模型在生成其结尾时关注音频的开头，我们解决了这一问题。这种仅推理的方法生成的音频能够自然地循环，而不需要任何额外的训练或数据。通过计算闭环接缝处标记的困惑度来评估循环过渡的一致性，我们观察到55%的提升。盲听测试进一步证实了相对于基线方法有显著的感知改进，平均评分提升了70%。综上所述，这些结果突显了仅推理方法在提高生成模型效果方面的有效性，并强调了非自回归方法在具有上下文意识的音乐生成中的优势。 

---
# An overview of model uncertainty and variability in LLM-based sentiment analysis. Challenges, mitigation strategies and the role of explainability 

**Title (ZH)**: LLM基于的情感分析中模型不确定性与变异性的综述：挑战、缓解策略与可解释性的作用 

**Authors**: David Herrera-Poyatos, Carlos Peláez-González, Cristina Zuheros, Andrés Herrera-Poyatos, Virilo Tejedor, Francisco Herrera, Rosana Montes  

**Link**: [PDF](https://arxiv.org/pdf/2504.04462)  

**Abstract**: Large Language Models (LLMs) have significantly advanced sentiment analysis, yet their inherent uncertainty and variability pose critical challenges to achieving reliable and consistent outcomes. This paper systematically explores the Model Variability Problem (MVP) in LLM-based sentiment analysis, characterized by inconsistent sentiment classification, polarization, and uncertainty arising from stochastic inference mechanisms, prompt sensitivity, and biases in training data. We analyze the core causes of MVP, presenting illustrative examples and a case study to highlight its impact. In addition, we investigate key challenges and mitigation strategies, paying particular attention to the role of temperature as a driver of output randomness and emphasizing the crucial role of explainability in improving transparency and user trust. By providing a structured perspective on stability, reproducibility, and trustworthiness, this study helps develop more reliable, explainable, and robust sentiment analysis models, facilitating their deployment in high-stakes domains such as finance, healthcare, and policymaking, among others. 

**Abstract (ZH)**: 大型语言模型（LLMs）在情感分析方面的进步显著，但其固有的不确定性和变异性对实现可靠和一致的结果构成了关键挑战。本文系统探讨了基于LLM的情感分析中的模型变异性问题（MVP），该问题表现为情感分类的一致性差、极化和不确定性，源于随机推理机制、提示敏感性和训练数据中的偏见。我们分析了MVP的核心原因，并通过示例和案例研究突出其影响。此外，我们研究了关键挑战和缓解策略，特别关注温度作为输出随机性的驱动因素，并强调可解释性在提高透明度和用户信任方面的重要作用。通过提供稳定性、可重复性和可信性方面的结构化视角，本文有助于开发更可靠、可解释和 robust 的情感分析模型，促进其在金融、医疗保健、政策制定等领域中的部署。 

---
# EclipseNETs: Learning Irregular Small Celestial Body Silhouettes 

**Title (ZH)**: EclipseNETs：学习不规则小型天体轮廓 

**Authors**: Giacomo Acciarini, Dario Izzo, Francesco Biscani  

**Link**: [PDF](https://arxiv.org/pdf/2504.04455)  

**Abstract**: Accurately predicting eclipse events around irregular small bodies is crucial for spacecraft navigation, orbit determination, and spacecraft systems management. This paper introduces a novel approach leveraging neural implicit representations to model eclipse conditions efficiently and reliably. We propose neural network architectures that capture the complex silhouettes of asteroids and comets with high precision. Tested on four well-characterized bodies - Bennu, Itokawa, 67P/Churyumov-Gerasimenko, and Eros - our method achieves accuracy comparable to traditional ray-tracing techniques while offering orders of magnitude faster performance. Additionally, we develop an indirect learning framework that trains these models directly from sparse trajectory data using Neural Ordinary Differential Equations, removing the requirement to have prior knowledge of an accurate shape model. This approach allows for the continuous refinement of eclipse predictions, progressively reducing errors and improving accuracy as new trajectory data is incorporated. 

**Abstract (ZH)**: 准确预测不规则小体附近的日食事件对于 spacecraft 导航、轨道确定以及 spacecraft 系统管理至关重要。本文介绍了一种利用神经隐式表示的新方法，以高效可靠地建模日食条件。我们提出了能够以高精度捕捉小行星和彗星复杂剪影的神经网络架构。在对班努小行星、伊托卡瓦小行星、67P/丘留莫夫-格拉西缅科彗星和爱神星这四个已充分-characterized的天体进行测试后，我们的方法在准确度上达到了与传统射线跟踪技术相当的水平，同时提供了数量级更优的性能。此外，我们开发了一种间接学习框架，通过使用神经常微分方程直接从稀疏轨迹数据训练这些模型，无需事先具备精确形状模型的知识。该方法允许对日食预测进行持续优化，随着新轨迹数据的加入，逐步减少误差并提高准确度。 

---
# Prot42: a Novel Family of Protein Language Models for Target-aware Protein Binder Generation 

**Title (ZH)**: Prot42：一种新的蛋白质语言模型家族，用于目标导向的蛋白质结合物生成 

**Authors**: Mohammad Amaan Sayeed, Engin Tekin, Maryam Nadeem, Nancy A. ElNaker, Aahan Singh, Natalia Vassilieva, Boulbaba Ben Amor  

**Link**: [PDF](https://arxiv.org/pdf/2504.04453)  

**Abstract**: Unlocking the next generation of biotechnology and therapeutic innovation demands overcoming the inherent complexity and resource-intensity of conventional protein engineering methods. Recent GenAI-powered computational techniques often rely on the availability of the target protein's 3D structures and specific binding sites to generate high-affinity binders, constraints exhibited by models such as AlphaProteo and RFdiffusion. In this work, we explore the use of Protein Language Models (pLMs) for high-affinity binder generation. We introduce Prot42, a novel family of Protein Language Models (pLMs) pretrained on vast amounts of unlabeled protein sequences. By capturing deep evolutionary, structural, and functional insights through an advanced auto-regressive, decoder-only architecture inspired by breakthroughs in natural language processing, Prot42 dramatically expands the capabilities of computational protein design based on language only. Remarkably, our models handle sequences up to 8,192 amino acids, significantly surpassing standard limitations and enabling precise modeling of large proteins and complex multi-domain sequences. Demonstrating powerful practical applications, Prot42 excels in generating high-affinity protein binders and sequence-specific DNA-binding proteins. Our innovative models are publicly available, offering the scientific community an efficient and precise computational toolkit for rapid protein engineering. 

**Abstract (ZH)**: 解锁下一代生物技术和治疗创新需求，必须克服传统蛋白质工程方法的内在复杂性和资源密集性。近年来，基于GenAI的计算技术往往依赖目标蛋白的3D结构和特定结合位点来生成高亲和力的结合物，这一特性在AlphaProteo和RFdiffusion等模型中有所体现。本工作中，我们探索了蛋白质语言模型（pLMs）在生成高亲和力结合物中的应用。我们介绍了Prot42，一种新型的蛋白质语言模型（pLMs），基于大量的未标注蛋白质序列进行预训练。通过一种先进的自回归、解码器仅结构的高级架构，该架构受到自然语言处理突破的启发，Prot42大大扩展了仅基于语言的计算蛋白质设计的能力。令人惊叹的是，我们的模型可以处理多达8,192个氨基酸的序列，远超标准限制，从而能够精确建模大型蛋白质和复杂多域序列。通过展现强大的实际应用能力，Prot42在生成高亲和力蛋白质结合物和序列特异性DNA结合蛋白方面表现出色。我们的创新模型现已公开，为科学界提供了高效且精确的计算工具箱，用于快速蛋白质工程。 

---
# On the Spatial Structure of Mixture-of-Experts in Transformers 

**Title (ZH)**: 混合专家Transformer中的空间结构研究 

**Authors**: Daniel Bershatsky, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2504.04444)  

**Abstract**: A common assumption is that MoE routers primarily leverage semantic features for expert selection. However, our study challenges this notion by demonstrating that positional token information also plays a crucial role in routing decisions. Through extensive empirical analysis, we provide evidence supporting this hypothesis, develop a phenomenological explanation of the observed behavior, and discuss practical implications for MoE-based architectures. 

**Abstract (ZH)**: 一种常见的假设是MoE路由器主要依赖语义特征进行专家选择。然而，我们的研究通过证明位置令牌信息also在路由决策中也起着至关重要的作用来挑战这一观点。通过大量的实证分析，我们提供了支持这一假设的证据，发展了一个关于观察到的行为的现象学解释，并讨论了基于MoE的架构的实践意义。 

---
# Do We Need Responsible XR? Drawing on Responsible AI to Inform Ethical Research and Practice into XRAI / the Metaverse 

**Title (ZH)**: 我们需要负责任的扩展现实吗？从负责任的人工智能借鉴以指导XRAI/元宇宙的伦理研究与实践 

**Authors**: Mark McGill, Joseph O'Hagan, Thomas Goodge, Graham Wilson, Mohamed Khamis, Veronika Krauß, Jan Gugenheimer  

**Link**: [PDF](https://arxiv.org/pdf/2504.04440)  

**Abstract**: This position paper for the CHI 2025 workshop "Everyday AR through AI-in-the-Loop" reflects on whether as a field HCI needs to define Responsible XR as a parallel to, and in conjunction with, Responsible AI, addressing the unique vulnerabilities posed by mass adoption of wearable AI-enabled AR glasses and XR devices that could enact AI-driven human perceptual augmentation. 

**Abstract (ZH)**: CHI 2025研讨会“AI在环中的日常AR：关于HCI领域是否需要定义 Responsible XR 作为 Responsible AI 的平行概念暨配套设施的反思” 

---
# Formula-Supervised Sound Event Detection: Pre-Training Without Real Data 

**Title (ZH)**: 公式监督声事件检测：无真实数据的预训练 

**Authors**: Yuto Shibata, Keitaro Tanaka, Yoshiaki Bando, Keisuke Imoto, Hirokatsu Kataoka, Yoshimitsu Aoki  

**Link**: [PDF](https://arxiv.org/pdf/2504.04428)  

**Abstract**: In this paper, we propose a novel formula-driven supervised learning (FDSL) framework for pre-training an environmental sound analysis model by leveraging acoustic signals parametrically synthesized through formula-driven methods. Specifically, we outline detailed procedures and evaluate their effectiveness for sound event detection (SED). The SED task, which involves estimating the types and timings of sound events, is particularly challenged by the difficulty of acquiring a sufficient quantity of accurately labeled training data. Moreover, it is well known that manually annotated labels often contain noises and are significantly influenced by the subjective judgment of annotators. To address these challenges, we propose a novel pre-training method that utilizes a synthetic dataset, Formula-SED, where acoustic data are generated solely based on mathematical formulas. The proposed method enables large-scale pre-training by using the synthesis parameters applied at each time step as ground truth labels, thereby eliminating label noise and bias. We demonstrate that large-scale pre-training with Formula-SED significantly enhances model accuracy and accelerates training, as evidenced by our results in the DESED dataset used for DCASE2023 Challenge Task 4. The project page is at this https URL 

**Abstract (ZH)**: 一种基于公式的监督学习（FDSL）框架：通过利用公式驱动方法合成的声学信号预先训练环境声音分析模型 

---
# FluentLip: A Phonemes-Based Two-stage Approach for Audio-Driven Lip Synthesis with Optical Flow Consistency 

**Title (ZH)**: FluentLip: 基于音素的两阶段音频驱动唇型合成方法，具有光流一致性 

**Authors**: Shiyan Liu, Rui Qu, Yan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.04427)  

**Abstract**: Generating consecutive images of lip movements that align with a given speech in audio-driven lip synthesis is a challenging task. While previous studies have made strides in synchronization and visual quality, lip intelligibility and video fluency remain persistent challenges. This work proposes FluentLip, a two-stage approach for audio-driven lip synthesis, incorporating three featured strategies. To improve lip synchronization and intelligibility, we integrate a phoneme extractor and encoder to generate a fusion of audio and phoneme information for multimodal learning. Additionally, we employ optical flow consistency loss to ensure natural transitions between image frames. Furthermore, we incorporate a diffusion chain during the training of Generative Adversarial Networks (GANs) to improve both stability and efficiency. We evaluate our proposed FluentLip through extensive experiments, comparing it with five state-of-the-art (SOTA) approaches across five metrics, including a proposed metric called Phoneme Error Rate (PER) that evaluates lip pose intelligibility and video fluency. The experimental results demonstrate that our FluentLip approach is highly competitive, achieving significant improvements in smoothness and naturalness. In particular, it outperforms these SOTA approaches by approximately $\textbf{16.3%}$ in Fréchet Inception Distance (FID) and $\textbf{35.2%}$ in PER. 

**Abstract (ZH)**: 基于音频驱动的唇动合成生成与给定语音连续对齐的图像，并保持唇部可读性和视频流畅性是一项具有挑战性的任务。尽管先前的研究在同步性和视觉质量方面取得了进展，但唇部可读性和视频流畅性依然存在持续的挑战。本文提出了一种名为FluentLip的两阶段方法，结合了三种特色策略。为了提高唇部同步性和可读性，我们集成了一个音素提取器和编码器，生成音素信息和音频信息的融合，进行多模态学习。此外，我们利用光学流一致性损失来确保图像帧间的自然过渡。进一步地，在生成对抗网络（GAN）的训练中引入了扩散链，以提高稳定性和效率。我们通过广泛的实验评估了提出的FluentLip方法，将其与五个最先进的（SOTA）方法在五个指标上进行了比较，包括一个新的名为音素错误率（PER）的指标，该指标评估了唇形姿态的可读性和视频流畅性。实验结果表明，我们的FluentLip方法具有很强的竞争性，明显提高了平滑度和自然度。特别是在弗雷谢特入胜距离（FID）和PER上，分别优于这些SOTA方法约16.3%和35.2%。 

---
# UniToken: Harmonizing Multimodal Understanding and Generation through Unified Visual Encoding 

**Title (ZH)**: UniToken: 统一视觉编码实现多模态理解和生成和谐统一 

**Authors**: Yang Jiao, Haibo Qiu, Zequn Jie, Shaoxiang Chen, Jingjing Chen, Lin Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04423)  

**Abstract**: We introduce UniToken, an auto-regressive generation model that encodes visual inputs through a combination of discrete and continuous representations, enabling seamless integration of unified visual understanding and image generation tasks. Unlike previous approaches that rely on unilateral visual representations, our unified visual encoding framework captures both high-level semantics and low-level details, delivering multidimensional information that empowers heterogeneous tasks to selectively assimilate domain-specific knowledge based on their inherent characteristics. Through in-depth experiments, we uncover key principles for developing a unified model capable of both visual understanding and image generation. Extensive evaluations across a diverse range of prominent benchmarks demonstrate that UniToken achieves state-of-the-art performance, surpassing existing approaches. These results establish UniToken as a robust foundation for future research in this domain. The code and models are available at this https URL. 

**Abstract (ZH)**: 我们介绍UniToken，这是一种自回归生成模型，通过离散和连续表示的结合编码视觉输入，实现统一的视觉理解和图像生成任务的无缝集成。与依赖单向视觉表示的先前方法不同，我们的统一视觉编码框架捕捉了高层语义和底层细节，提供多维度信息，使异构任务能够根据自身特性选择性地吸收特定领域的知识。通过深入的实验，我们揭示了开发既能进行视觉理解又能进行图像生成的统一模型的关键原则。广泛的评估表明，UniToken在多种突出基准上达到了最先进的性能，超越了现有方法。这些结果奠定了UniToken作为未来研究坚实基础的地位。代码和模型可在以下链接获取。 

---
# Driving-RAG: Driving Scenarios Embedding, Search, and RAG Applications 

**Title (ZH)**: 驾驶场景嵌入、搜索与RAG应用 

**Authors**: Cheng Chang, Jingwei Ge, Jiazhe Guo, Zelin Guo, Binghong Jiang, Li Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.04419)  

**Abstract**: Driving scenario data play an increasingly vital role in the development of intelligent vehicles and autonomous driving. Accurate and efficient scenario data search is critical for both online vehicle decision-making and planning, and offline scenario generation and simulations, as it allows for leveraging the scenario experiences to improve the overall performance. Especially with the application of large language models (LLMs) and Retrieval-Augmented-Generation (RAG) systems in autonomous driving, urgent requirements are put forward. In this paper, we introduce the Driving-RAG framework to address the challenges of efficient scenario data embedding, search, and applications for RAG systems. Our embedding model aligns fundamental scenario information and scenario distance metrics in the vector space. The typical scenario sampling method combined with hierarchical navigable small world can perform efficient scenario vector search to achieve high efficiency without sacrificing accuracy. In addition, the reorganization mechanism by graph knowledge enhances the relevance to the prompt scenarios and augment LLM generation. We demonstrate the effectiveness of the proposed framework on typical trajectory planning task for complex interactive scenarios such as ramps and intersections, showcasing its advantages for RAG applications. 

**Abstract (ZH)**: 自动驾驶场景数据在智能车辆和自主驾驶发展中的作用日益重要。准确高效的场景数据搜索对于在线车辆决策和规划以及离线场景生成和仿真至关重要，因为它能够利用场景经验提升整体性能。特别是在大规模语言模型（LLMs）和检索增强生成（RAG）系统应用于自主驾驶时，提出了迫切的需求。本文介绍Drive-RAG框架以解决RAG系统中高效场景数据嵌入、搜索和应用的挑战。我们的嵌入模型在向量空间中对基本场景信息和场景距离度量进行对齐。结合典型场景采样方法和分层可导航的小世界图，可以实现高效场景向量搜索，在不牺牲准确性的前提下达到高的效率。此外，通过图知识进行重新组织机制增强了对提示场景的相关性并增强LLM生成。我们在复杂交互场景（如匝道和交叉口）的典型轨迹规划任务上展示了所提框架的有效性，强调了其在RAG应用中的优势。 

---
# Universal Item Tokenization for Transferable Generative Recommendation 

**Title (ZH)**: 通用项目标记化以实现可转移的生成推荐 

**Authors**: Bowen Zheng, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.04405)  

**Abstract**: Recently, generative recommendation has emerged as a promising paradigm, attracting significant research attention. The basic framework involves an item tokenizer, which represents each item as a sequence of codes serving as its identifier, and a generative recommender that predicts the next item by autoregressively generating the target item identifier. However, in existing methods, both the tokenizer and the recommender are typically domain-specific, limiting their ability for effective transfer or adaptation to new domains. To this end, we propose UTGRec, a Universal item Tokenization approach for transferable Generative Recommendation. Specifically, we design a universal item tokenizer for encoding rich item semantics by adapting a multimodal large language model (MLLM). By devising tree-structured codebooks, we discretize content representations into corresponding codes for item tokenization. To effectively learn the universal item tokenizer on multiple domains, we introduce two key techniques in our approach. For raw content reconstruction, we employ dual lightweight decoders to reconstruct item text and images from discrete representations to capture general knowledge embedded in the content. For collaborative knowledge integration, we assume that co-occurring items are similar and integrate collaborative signals through co-occurrence alignment and reconstruction. Finally, we present a joint learning framework to pre-train and adapt the transferable generative recommender across multiple domains. Extensive experiments on four public datasets demonstrate the superiority of UTGRec compared to both traditional and generative recommendation baselines. 

**Abstract (ZH)**: 通用项目编码的可迁移生成推荐方法 UTGRec 

---
# Pre-training Generative Recommender with Multi-Identifier Item Tokenization 

**Title (ZH)**: 基于多标识符项分词的预训练生成型推荐器 

**Authors**: Bowen Zheng, Enze Liu, Zhongfu Chen, Zhongrui Ma, Yue Wang, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.04400)  

**Abstract**: Generative recommendation autoregressively generates item identifiers to recommend potential items. Existing methods typically adopt a one-to-one mapping strategy, where each item is represented by a single identifier. However, this scheme poses issues, such as suboptimal semantic modeling for low-frequency items and limited diversity in token sequence data. To overcome these limitations, we propose MTGRec, which leverages Multi-identifier item Tokenization to augment token sequence data for Generative Recommender pre-training. Our approach involves two key innovations: multi-identifier item tokenization and curriculum recommender pre-training. For multi-identifier item tokenization, we leverage the RQ-VAE as the tokenizer backbone and treat model checkpoints from adjacent training epochs as semantically relevant tokenizers. This allows each item to be associated with multiple identifiers, enabling a single user interaction sequence to be converted into several token sequences as different data groups. For curriculum recommender pre-training, we introduce a curriculum learning scheme guided by data influence estimation, dynamically adjusting the sampling probability of each data group during recommender pre-training. After pre-training, we fine-tune the model using a single tokenizer to ensure accurate item identification for recommendation. Extensive experiments on three public benchmark datasets demonstrate that MTGRec significantly outperforms both traditional and generative recommendation baselines in terms of effectiveness and scalability. 

**Abstract (ZH)**: 基于多标识符项分词的生成推荐预训练方法 

---
# Future-Proof Yourself: An AI Era Survival Guide 

**Title (ZH)**: 未来proof你自己：人工智能时代生存指南 

**Authors**: Taehoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.04378)  

**Abstract**: Future-Proof Yourself is a practical guide that helps readers navigate the fast-changing world of artificial intelligence in everyday life. The book begins by explaining how computers learn from data in simple, relatable terms, and gradually introduces the methods used in modern AI. It shows how basic ideas in machine learning evolve into advanced systems that can recognize images, understand language, and even make decisions. The guide also reviews the history of AI and highlights the major breakthroughs that have shaped its growth. Looking ahead, the book explores emerging trends such as the integration of AI with digital twins, wearable devices, and virtual environments. Designed for a general audience, the text avoids heavy technical jargon and presents complex ideas in clear, straightforward language so that anyone can gain a solid understanding of the technology that is set to transform our future. 

**Abstract (ZH)**: 展望未来：人工智能在日常生活中的实用指南 

---
# iADCPS: Time Series Anomaly Detection for Evolving Cyber-physical Systems via Incremental Meta-learning 

**Title (ZH)**: iADCPS：基于增量元学习的 evolving 虚拟物理系统时间序列异常检测 

**Authors**: Jiyu Tian, Mingchu Li, Liming Chen, Zumin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04374)  

**Abstract**: Anomaly detection for cyber-physical systems (ADCPS) is crucial in identifying faults and potential attacks by analyzing the time series of sensor measurements and actuator states. However, current methods lack adaptation to data distribution shifts in both temporal and spatial dimensions as cyber-physical systems evolve. To tackle this issue, we propose an incremental meta-learning-based approach, namely iADCPS, which can continuously update the model through limited evolving normal samples to reconcile the distribution gap between evolving and historical time series. Specifically, We first introduce a temporal mixup strategy to align data for data-level generalization which is then combined with the one-class meta-learning approach for model-level generalization. Furthermore, we develop a non-parametric dynamic threshold to adaptively adjust the threshold based on the probability density of the abnormal scores without any anomaly supervision. We empirically evaluate the effectiveness of the iADCPS using three publicly available datasets PUMP, SWaT, and WADI. The experimental results demonstrate that our method achieves 99.0%, 93.1%, and 78.7% F1-Score, respectively, which outperforms the state-of-the-art (SOTA) ADCPS method, especially in the context of the evolving CPSs. 

**Abstract (ZH)**: 基于增量元学习的时变和空变适应性网络异常检测方法（iADCPS） 

---
# StyleRec: A Benchmark Dataset for Prompt Recovery in Writing Style Transformation 

**Title (ZH)**: StyleRec: 一种写作风格转换中提示恢复基准数据集 

**Authors**: Shenyang Liu, Yang Gao, Shaoyan Zhai, Liqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04373)  

**Abstract**: Prompt Recovery, reconstructing prompts from the outputs of large language models (LLMs), has grown in importance as LLMs become ubiquitous. Most users access LLMs through APIs without internal model weights, relying only on outputs and logits, which complicates recovery. This paper explores a unique prompt recovery task focused on reconstructing prompts for style transfer and rephrasing, rather than typical question-answering. We introduce a dataset created with LLM assistance, ensuring quality through multiple techniques, and test methods like zero-shot, few-shot, jailbreak, chain-of-thought, fine-tuning, and a novel canonical-prompt fallback for poor-performing cases. Our results show that one-shot and fine-tuning yield the best outcomes but highlight flaws in traditional sentence similarity metrics for evaluating prompt recovery. Contributions include (1) a benchmark dataset, (2) comprehensive experiments on prompt recovery strategies, and (3) identification of limitations in current evaluation metrics, all of which advance general prompt recovery research, where the structure of the input prompt is unrestricted. 

**Abstract (ZH)**: Prompt恢复：从大型语言模型（LLMs）的输出重建提示，在LLMs无内部模型权重并通过API供用户访问的情况下变得日益重要。本文探讨了专注于风格转换和重述的提示恢复任务，而非典型的问答任务。我们利用LLM创建了一个数据集，并通过多种技术确保数据集的质量，测试了零样本、少样本、突破限制、思维链、微调以及一种新颖的标准提示后备方法。结果表明，单样本和微调取得最佳效果，但也揭示了传统句子相似度度量在评估提示恢复方面的局限。贡献包括：（1）基准数据集，（2）提示恢复策略的全面实验，以及（3）对当前评估度量标准限制的识别，这些均推动了不受输入提示结构限制的通用提示恢复研究。 

---
# How Accurately Do Large Language Models Understand Code? 

**Title (ZH)**: 大型语言模型对代码理解的准确性如何？ 

**Authors**: Sabaat Haroon, Ahmad Faraz Khan, Ahmad Humayun, Waris Gill, Abdul Haddi Amjad, Ali R. Butt, Mohammad Taha Khan, Muhammad Ali Gulzar  

**Link**: [PDF](https://arxiv.org/pdf/2504.04372)  

**Abstract**: Large Language Models (LLMs) are increasingly used in post-development tasks such as code repair and testing. A key factor in these tasks' success is the model's deep understanding of code. However, the extent to which LLMs truly understand code remains largely unevaluated. Quantifying code comprehension is challenging due to its abstract nature and the lack of a standardized metric. Previously, this was assessed through developer surveys, which are not feasible for evaluating LLMs. Existing LLM benchmarks focus primarily on code generation, fundamentally different from code comprehension. Additionally, fixed benchmarks quickly become obsolete as they become part of the training data. This paper presents the first large-scale empirical investigation into LLMs' ability to understand code. Inspired by mutation testing, we use an LLM's fault-finding ability as a proxy for its deep code understanding. This approach is based on the insight that a model capable of identifying subtle functional discrepancies must understand the code well. We inject faults in real-world programs and ask the LLM to localize them, ensuring the specifications suffice for fault localization. Next, we apply semantic-preserving code mutations (SPMs) to the faulty programs and test whether the LLMs still locate the faults, verifying their confidence in code understanding. We evaluate nine popular LLMs on 575000 debugging tasks from 670 Java and 637 Python programs. We find that LLMs lose the ability to debug the same bug in 81% of faulty programs when SPMs are applied, indicating a shallow understanding of code and reliance on features irrelevant to semantics. We also find that LLMs understand code earlier in the program better than later. This suggests that LLMs' code comprehension remains tied to lexical and syntactic features due to tokenization designed for natural languages, which overlooks code semantics. 

**Abstract (ZH)**: 大型语言模型在代码理解方面的大规模实证研究：基于故障定位能力的代码理解评估 

---
# WeiDetect: Weibull Distribution-Based Defense against Poisoning Attacks in Federated Learning for Network Intrusion Detection Systems 

**Title (ZH)**: WeiDetect：基于Weibull分布的网络入侵检测系统联邦学习防毒攻击方法 

**Authors**: Sameera K. M., Vinod P., Anderson Rocha, Rafidha Rehiman K. A., Mauro Conti  

**Link**: [PDF](https://arxiv.org/pdf/2504.04367)  

**Abstract**: In the era of data expansion, ensuring data privacy has become increasingly critical, posing significant challenges to traditional AI-based applications. In addition, the increasing adoption of IoT devices has introduced significant cybersecurity challenges, making traditional Network Intrusion Detection Systems (NIDS) less effective against evolving threats, and privacy concerns and regulatory restrictions limit their deployment. Federated Learning (FL) has emerged as a promising solution, allowing decentralized model training while maintaining data privacy to solve these issues. However, despite implementing privacy-preserving technologies, FL systems remain vulnerable to adversarial attacks. Furthermore, data distribution among clients is not heterogeneous in the FL scenario. We propose WeiDetect, a two-phase, server-side defense mechanism for FL-based NIDS that detects malicious participants to address these challenges. In the first phase, local models are evaluated using a validation dataset to generate validation scores. These scores are then analyzed using a Weibull distribution, identifying and removing malicious models. We conducted experiments to evaluate the effectiveness of our approach in diverse attack settings. Our evaluation included two popular datasets, CIC-Darknet2020 and CSE-CIC-IDS2018, tested under non-IID data distributions. Our findings highlight that WeiDetect outperforms state-of-the-art defense approaches, improving higher target class recall up to 70% and enhancing the global model's F1 score by 1% to 14%. 

**Abstract (ZH)**: 在数据扩张时代，确保数据隐私越来越关键，对传统基于AI的应用构成了重大挑战。此外，物联网设备的普及引入了显著的网络安全挑战，使得传统的网络入侵检测系统（NIDS）对不断演化的威胁效果减弱，隐私关切和监管限制也限制了它们的应用。联邦学习（FL）作为一项有前景的解决方案，能够在保持数据隐私的同时实现去中心化的模型训练，以解决这些问题。然而，尽管实施了隐私保护技术，FL系统仍然容易遭受对抗性攻击。此外，FL场景下客户端的数据分布并不是异质化的。我们提出WeiDetect，这是一种双阶段的服务器端防御机制，用于基于FL的NIDS以检测恶意参与者，以解决这些挑战。在第一阶段，使用验证数据集评估本地模型并生成验证分数。然后使用威布尔分布分析这些分数，识别并移除恶意模型。我们在不同的攻击场景下进行了实验，评估我们方法的有效性。我们的评估包括两个流行的数据集CIC-Darknet2020和CSE-CIC-IDS2018，并在非IID数据分布下进行了测试。我们的研究结果表明，WeiDetect在最高目标类召回率上优于现有最先进的防御方法，提高了60%至70%，同时将全局模型的F1分数提高了1%至14%。 

---
# AutoPDL: Automatic Prompt Optimization for LLM Agents 

**Title (ZH)**: AutoPDL：自动提示优化 for LLM 代理 

**Authors**: Claudio Spiess, Mandana Vaziri, Louis Mandel, Martin Hirzel  

**Link**: [PDF](https://arxiv.org/pdf/2504.04365)  

**Abstract**: The performance of large language models (LLMs) depends on how they are prompted, with choices spanning both the high-level prompting pattern (e.g., Zero-Shot, CoT, ReAct, ReWOO) and the specific prompt content (instructions and few-shot demonstrations). Manually tuning this combination is tedious, error-prone, and non-transferable across LLMs or tasks. Therefore, this paper proposes AutoPDL, an automated approach to discover good LLM agent configurations. Our method frames this as a structured AutoML problem over a combinatorial space of agentic and non-agentic prompting patterns and demonstrations, using successive halving to efficiently navigate this space. We introduce a library implementing common prompting patterns using the PDL prompt programming language. AutoPDL solutions are human-readable, editable, and executable PDL programs that use this library. This approach also enables source-to-source optimization, allowing human-in-the-loop refinement and reuse. Evaluations across three tasks and six LLMs (ranging from 8B to 70B parameters) show consistent accuracy gains ($9.5\pm17.5$ percentage points), up to 68.9pp, and reveal that selected prompting strategies vary across models and tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）的表现取决于它们的提示方式，涵盖从高层提示模式（如零样本、逐步推理、ReAct、ReWOO）到具体的提示内容（指令和少量示例）。手动调整这一组合是繁琐、易出错且在不同LLMs或任务间不可移植的。因此，本文提出了AutoPDL，这是一种自动化的手段，用于发现良好的LLM代理配置。我们的方法将此问题视为一个结构化的自动机器学习（AutoML）问题，通过组合搜索空间中的代理性和非代理性提示模式和示例来求解，使用逐次减半技术高效地导航这一空间。我们引入了一个使用PDL提示编程语言实现常见提示模式的库。AutoPDL解决方案是可读、可编辑和可执行的PDL程序，使用该库。这种方法还允许源到源优化，允许逐次在环中进行改进和重用。在三个任务和六种LLM（参数范围从8B到70B）上的评估显示一致的准确率提升（9.5±17.5 个百分点），最高可达68.9个百分点，并揭示了所选提示策略在不同模型和任务间存在差异。 

---
# REFORMER: A ChatGPT-Driven Data Synthesis Framework Elevating Text-to-SQL Models 

**Title (ZH)**: REFORMER：一个由ChatGPT驱动的数据合成框架，提升Text-to-SQL模型 

**Authors**: Shenyang Liu, Saleh Almohaimeed, Liqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04363)  

**Abstract**: The existing Text-to-SQL models suffer from a shortage of training data, inhibiting their ability to fully facilitate the applications of SQL queries in new domains. To address this challenge, various data synthesis techniques have been employed to generate more diverse and higher quality data. In this paper, we propose REFORMER, a framework that leverages ChatGPT's prowess without the need for additional training, to facilitate the synthesis of (question, SQL query) pairs tailored to new domains. Our data augmentation approach is based on a "retrieve-and-edit" method, where we generate new questions by filling masked question using explanation of SQL queries with the help of ChatGPT. Furthermore, we demonstrate that cycle consistency remains a valuable method of validation when applied appropriately. Our experimental results show that REFORMER consistently outperforms previous data augmentation methods. To further investigate the power of ChatGPT and create a general data augmentation method, we also generate the new data by paraphrasing the question in the dataset and by paraphrasing the description of a new SQL query that is generated by ChatGPT as well. Our results affirm that paraphrasing questions generated by ChatGPT help augment the original data. 

**Abstract (ZH)**: 现有的Text-to-SQL模型因训练数据不足，限制了SQL查询在新领域应用的能力。为此，各种数据合成技术被用于生成更多样且高质量的数据。本文提出REFORMER框架，利用ChatGPT的能力而无需额外训练，以生成适合新领域的问答对。我们的数据增强方法基于“检索与编辑”方法，通过使用ChatGPT解释SQL查询来生成新的问题。此外，我们还证明了当适当应用时，循环一致性仍然是一个有价值的验证方法。实验结果表明，REFORMER始终优于先前的数据增强方法。为进一步探索ChatGPT的力量并创建通用的数据增强方法，我们还通过改写数据集中的问题和由ChatGPT生成的新SQL查询的描述来生成新数据。结果证实，改写由ChatGPT生成的问题有助于增强原始数据。 

---
# DDPT: Diffusion-Driven Prompt Tuning for Large Language Model Code Generation 

**Title (ZH)**: DDPT：由扩散驱动的提示调优在大规模语言模型代码生成中的应用 

**Authors**: Jinyang Li, Sangwon Hyun, M. Ali Babar  

**Link**: [PDF](https://arxiv.org/pdf/2504.04351)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation. However, the quality of the generated code is heavily dependent on the structure and composition of the prompts used. Crafting high-quality prompts is a challenging task that requires significant knowledge and skills of prompt engineering. To advance the automation support for the prompt engineering for LLM-based code generation, we propose a novel solution Diffusion-Driven Prompt Tuning (DDPT) that learns how to generate optimal prompt embedding from Gaussian Noise to automate the prompt engineering for code generation. We evaluate the feasibility of diffusion-based optimization and abstract the optimal prompt embedding as a directional vector toward the optimal embedding. We use the code generation loss given by the LLMs to help the diffusion model capture the distribution of optimal prompt embedding during training. The trained diffusion model can build a path from the noise distribution to the optimal distribution at the sampling phrase, the evaluation result demonstrates that DDPT helps improve the prompt optimization for code generation. 

**Abstract (ZH)**: 基于扩散驱动提示调优的大型语言模型代码生成提示工程探索 

---
# Generative Large Language Models Trained for Detecting Errors in Radiology Reports 

**Title (ZH)**: 生成式大型语言模型用于检测放射学报告中的错误 

**Authors**: Cong Sun, Kurt Teichman, Yiliang Zhou, Brian Critelli, David Nauheim, Graham Keir, Xindi Wang, Judy Zhong, Adam E Flanders, George Shih, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2504.04336)  

**Abstract**: In this retrospective study, a dataset was constructed with two parts. The first part included 1,656 synthetic chest radiology reports generated by GPT-4 using specified prompts, with 828 being error-free synthetic reports and 828 containing errors. The second part included 614 reports: 307 error-free reports between 2011 and 2016 from the MIMIC-CXR database and 307 corresponding synthetic reports with errors generated by GPT-4 on the basis of these MIMIC-CXR reports and specified prompts. All errors were categorized into four types: negation, left/right, interval change, and transcription errors. Then, several models, including Llama-3, GPT-4, and BiomedBERT, were refined using zero-shot prompting, few-shot prompting, or fine-tuning strategies. Finally, the performance of these models was evaluated using the F1 score, 95\% confidence interval (CI) and paired-sample t-tests on our constructed dataset, with the prediction results further assessed by radiologists. Using zero-shot prompting, the fine-tuned Llama-3-70B-Instruct model achieved the best performance with the following F1 scores: 0.769 for negation errors, 0.772 for left/right errors, 0.750 for interval change errors, 0.828 for transcription errors, and 0.780 overall. In the real-world evaluation phase, two radiologists reviewed 200 randomly selected reports output by the model. Of these, 99 were confirmed to contain errors detected by the models by both radiologists, and 163 were confirmed to contain model-detected errors by at least one radiologist. Generative LLMs, fine-tuned on synthetic and MIMIC-CXR radiology reports, greatly enhanced error detection in radiology reports. 

**Abstract (ZH)**: 一项回顾性研究：构建了一个包含两部分的数据集。第一部分包括由GPT-4根据指定提示生成的1,656份合成胸部X光报告，其中828份无错误，828份包含错误。第二部分包括614份报告：来自MIMIC-CXR数据库的2011-2016年间307份无错误报告及其对应的由GPT-4根据这些MIMIC-CXR报告和指定提示生成的含有错误的307份合成报告。所有错误被分类为四种类型：否定形式、左右错误、区间变化和转写错误。随后，使用零样本提示、少数样本提示或微调策略分别对Llama-3、GPT-4和BiomedBERT等模型进行了优化。最后，使用我们的构建数据集上的F1分数、95%置信区间（CI）和配对样本t检验评估了这些模型的性能，并由放射科医生进一步评估了预测结果。使用零样本提示，微调后的Llama-3-70B-Instruct模型在以下F1分数上表现最佳：否定形式错误为0.769，左右错误为0.772，区间变化错误为0.750，转写错误为0.828，总分为0.780。在实际评估阶段，两位放射科医生审查了模型输出的200份随机选取的报告。其中，99份报告被两位放射科医生确认含有模型检测出的错误，163份报告被至少一位放射科医生确认含有模型检测出的错误。基于合成和MIMIC-CXR放射学报告进行微调的生成型大语言模型大大提升了放射学报告中的错误检测能力。 

---
# Hallucination Detection using Multi-View Attention Features 

**Title (ZH)**: 多视图注意力特征的幻觉检测 

**Authors**: Yuya Ogasa, Yuki Arase  

**Link**: [PDF](https://arxiv.org/pdf/2504.04335)  

**Abstract**: This study tackles token-level hallucination detection in outputs of large language models. Previous studies revealed that attention exhibits irregular patterns when hallucination occurs. Inspired by this, we extract features from the attention matrix that provide complementary views of (a) the average attention each token receives, which helps identify whether certain tokens are overly influential or ignored, (b) the diversity of attention each token receives, which reveals whether attention is biased toward specific subsets, and (c) the diversity of tokens a token attends to during generation, which indicates whether the model references a narrow or broad range of information. These features are input to a Transformer-based classifier to conduct token-level classification to identify hallucinated spans. Experimental results indicate that the proposed method outperforms strong baselines on hallucination detection with longer input contexts, i.e., data-to-text and summarization tasks. 

**Abstract (ZH)**: 本研究解决了大规模语言模型输出中的令牌级幻觉检测问题。以往的研究表明，当出现幻觉时，注意力机制会表现出不规则模式。受此启发，我们从注意力矩阵中抽取能够互补的观点特征：(a) 每个令牌平均接收到的注意力，有助于识别某些令牌是否过于重要或被忽略了；(b) 每个令牌接收到的注意力多样性，揭示了注意力是否偏向特定子集；(c) 生成过程中每个令牌关注到的令牌多样性，表明模型是否参考了狭窄或广泛的语料信息。这些特征被输入到基于Transformer的分类器中进行令牌级分类，以识别幻觉区间。实验结果表明，所提出的方法在更长输入上下文（如数据到文本和总结任务）的幻觉检测中优于强基线方法。 

---
# IMPersona: Evaluating Individual Level LM Impersonation 

**Title (ZH)**: IMPersona: 评价个体级别语言模型冒充 

**Authors**: Quan Shi, Carlos Jimenez, Stephen Dong, Brian Seo, Caden Yao, Adam Kelch, Karthik Narasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2504.04332)  

**Abstract**: As language models achieve increasingly human-like capabilities in conversational text generation, a critical question emerges: to what extent can these systems simulate the characteristics of specific individuals? To evaluate this, we introduce IMPersona, a framework for evaluating LMs at impersonating specific individuals' writing style and personal knowledge. Using supervised fine-tuning and a hierarchical memory-inspired retrieval system, we demonstrate that even modestly sized open-source models, such as Llama-3.1-8B-Instruct, can achieve impersonation abilities at concerning levels. In blind conversation experiments, participants (mis)identified our fine-tuned models with memory integration as human in 44.44% of interactions, compared to just 25.00% for the best prompting-based approach. We analyze these results to propose detection methods and defense strategies against such impersonation attempts. Our findings raise important questions about both the potential applications and risks of personalized language models, particularly regarding privacy, security, and the ethical deployment of such technologies in real-world contexts. 

**Abstract (ZH)**: 语言模型在对话文本生成中展现出日益接近人类的能力，一个关键问题是：这些系统能够模拟特定个体的特征到什么程度？为了评估这一点，我们提出了IMPersona框架，用于评估语言模型模仿特定个体的写作风格和个人知识的能力。通过监督微调和基于层次记忆的检索系统，我们展示了即使是规模较小的开源模型，如Llama-3.1-8B-Instruct，也能实现令人关注的模仿能力。在盲测对话实验中，参与者在整合记忆的情况下，误认为我们的微调模型是人类的比例达到了44.44%，而基于最佳提示的方法仅为25.00%。我们分析这些结果，提出了检测此类模仿企图的方法和防御策略。我们的研究结果引发了关于个性化语言模型潜在应用和风险的重要问题，特别是关于隐私、安全以及在现实场景中公平部署此类技术的伦理问题。 

---
# Geo-OLM: Enabling Sustainable Earth Observation Studies with Cost-Efficient Open Language Models & State-Driven Workflows 

**Title (ZH)**: Geo-OLM: 以经济高效的开源语言模型及状态驱动工作流促进可持续的地球观测研究 

**Authors**: Dimitrios Stamoulis, Diana Marculescu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04319)  

**Abstract**: Geospatial Copilots hold immense potential for automating Earth observation (EO) and climate monitoring workflows, yet their reliance on large-scale models such as GPT-4o introduces a paradox: tools intended for sustainability studies often incur unsustainable costs. Using agentic AI frameworks in geospatial applications can amass thousands of dollars in API charges or requires expensive, power-intensive GPUs for deployment, creating barriers for researchers, policymakers, and NGOs. Unfortunately, when geospatial Copilots are deployed with open language models (OLMs), performance often degrades due to their dependence on GPT-optimized logic. In this paper, we present Geo-OLM, a tool-augmented geospatial agent that leverages the novel paradigm of state-driven LLM reasoning to decouple task progression from tool calling. By alleviating the workflow reasoning burden, our approach enables low-resource OLMs to complete geospatial tasks more effectively. When downsizing to small models below 7B parameters, Geo-OLM outperforms the strongest prior geospatial baselines by 32.8% in successful query completion rates. Our method performs comparably to proprietary models achieving results within 10% of GPT-4o, while reducing inference costs by two orders of magnitude from \$500-\$1000 to under \$10. We present an in-depth analysis with geospatial downstream benchmarks, providing key insights to help practitioners effectively deploy OLMs for EO applications. 

**Abstract (ZH)**: 地理空间伴飞助手在自动化地球观测和气候监测工作流中具有巨大的潜力，但依赖大规模模型如GPT-4o引入了一个悖论：本应促进可持续研究的工具却常常导致不可持续的成本。使用自主人工智能框架在地理空间应用中可能会累积数千美元的API费用，或者需要成本高昂、能耗高的GPU进行部署，从而为研究人员、政策制定者和NGOs（非政府组织）设置了障碍。不幸的是，当地理空间伴飞助手使用开放语言模型（OLMs）部署时，由于其依赖于GPT优化逻辑，性能往往会下降。在本文中，我们提出了Geo-OLM，这是一种工具增强的地理空间代理，利用新的状态驱动LLM推理范式来解耦任务进展与工具调用。通过减轻工作流推理负担，我们的方法使低成本的OLMs能够更有效地完成地理空间任务。当模型参数减少到7B以下时，Geo-OLM在成功查询完成率上的表现比先前最优的地理空间基线高出32.8%。我们的方法在结果上与专有模型相当，其表现相对于GPT-4o在10%以内，同时将推理成本降低了两个数量级，从500-1000美元降至不到10美元。我们通过地理空间下游基准测试进行了深入分析，提供了关键见解，以帮助实践者有效部署OLMs用于地球观测应用。 

---
# Balancing Complexity and Informativeness in LLM-Based Clustering: Finding the Goldilocks Zone 

**Title (ZH)**: 基于LLM的聚类中复杂性和信息量的平衡：寻找黄金分割点 

**Authors**: Justin Miller, Tristram Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2504.04314)  

**Abstract**: The challenge of clustering short text data lies in balancing informativeness with interpretability. Traditional evaluation metrics often overlook this trade-off. Inspired by linguistic principles of communicative efficiency, this paper investigates the optimal number of clusters by quantifying the trade-off between informativeness and cognitive simplicity. We use large language models (LLMs) to generate cluster names and evaluate their effectiveness through semantic density, information theory, and clustering accuracy. Our results show that Gaussian Mixture Model (GMM) clustering on embeddings generated by a LLM, increases semantic density compared to random assignment, effectively grouping similar bios. However, as clusters increase, interpretability declines, as measured by a generative LLM's ability to correctly assign bios based on cluster names. A logistic regression analysis confirms that classification accuracy depends on the semantic similarity between bios and their assigned cluster names, as well as their distinction from alternatives.
These findings reveal a "Goldilocks zone" where clusters remain distinct yet interpretable. We identify an optimal range of 16-22 clusters, paralleling linguistic efficiency in lexical categorization. These insights inform both theoretical models and practical applications, guiding future research toward optimising cluster interpretability and usefulness. 

**Abstract (ZH)**: 短文本聚类面临的挑战在于平衡信息量与可解释性之间的关系。传统的评估指标往往忽视这种权衡。受语言交流效率语境原则的启发，本文通过量化信息量与认知简单性之间的权衡来探究最优聚类数量。我们使用大规模语言模型（LLMs）生成聚类名称，并通过语义密度、信息理论和聚类准确性进行评估。结果表明，使用LLM生成的嵌入进行高斯混合模型（GMM）聚类相比随机分配，能够增加语义密度，并有效分组相似的个人简介。然而，随着聚类数量的增加，可解释性下降，这一变化通过生成LLM根据聚类名称正确分配个人简介的能力得以衡量。逻辑回归分析证实，分类准确性取决于个人简介与其分配的聚类名称之间的语义相似度，以及与其替代选项的区别。这些发现揭示了一个“黄金分割带”，在这一带中，聚类既保持区分性又具有可解释性。我们确定了最优的聚类范围为16-22个，这与词汇分类中的语言效率相媲美。这些见解不仅为理论模型提供了指导，也为实际应用提供了依据，引导未来研究优化聚类的可解释性和实用性。 

---
# A Survey of Social Cybersecurity: Techniques for Attack Detection, Evaluations, Challenges, and Future Prospects 

**Title (ZH)**: 社交网络安全综述：攻击检测技术、评估、挑战及未来展望 

**Authors**: Aos Mulahuwaish, Basheer Qolomany, Kevin Gyorick, Jacques Bou Abdo, Mohammed Aledhari, Junaid Qadir, Kathleen Carley, Ala Al-Fuqaha  

**Link**: [PDF](https://arxiv.org/pdf/2504.04311)  

**Abstract**: In today's digital era, the Internet, especially social media platforms, plays a significant role in shaping public opinions, attitudes, and beliefs. Unfortunately, the credibility of scientific information sources is often undermined by the spread of misinformation through various means, including technology-driven tools like bots, cyborgs, trolls, sock-puppets, and deep fakes. This manipulation of public discourse serves antagonistic business agendas and compromises civil society. In response to this challenge, a new scientific discipline has emerged: social cybersecurity. 

**Abstract (ZH)**: 当前数字时代，互联网，尤其是社交媒体平台，在塑造公众意见、态度和信念方面发挥着重要作用。不幸的是，各种手段，包括以技术为导向的工具（如机器人、半人 machine、 trolls 和 sock-puppet 账号以及深度伪造），常常削弱了科学信息来源的可信度。这种对公共言论的操控服务于对抗性商业目的，并损害了公民社会。为应对这一挑战，一个新的科学学科应运而生：社会 cybersecurity。 

---
# CO-Bench: Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization 

**Title (ZH)**: CO-Bench: 评估语言模型代理在组合优化算法搜索中的性能 

**Authors**: Weiwei Sun, Shengyu Feng, Shanda Li, Yiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04310)  

**Abstract**: Although LLM-based agents have attracted significant attention in domains such as software engineering and machine learning research, their role in advancing combinatorial optimization (CO) remains relatively underexplored. This gap underscores the need for a deeper understanding of their potential in tackling structured, constraint-intensive problems-a pursuit currently limited by the absence of comprehensive benchmarks for systematic investigation. To address this, we introduce CO-Bench, a benchmark suite featuring 36 real-world CO problems drawn from a broad range of domains and complexity levels. CO-Bench includes structured problem formulations and curated data to support rigorous investigation of LLM agents. We evaluate multiple agent frameworks against established human-designed algorithms, revealing key strengths and limitations of current approaches and identifying promising directions for future research. CO-Bench is publicly available at this https URL. 

**Abstract (ZH)**: 尽管基于LLM的代理在软件工程和机器学习研究等领域受到了广泛关注，但在组合优化（CO）领域的应用仍然相对未被充分探索。为填补这一空白，我们引入了CO-Bench，这是一个包含36个来自各种领域和复杂程度的实际CO问题的基准套件。CO-Bench 包含结构化的问题形式和精选的数据，以支持对LLM代理进行严格的调查。我们评估了多个代理框架与现有的人类设计算法，揭示了当前方法的关键优势和局限性，并指出了未来研究的前景。CO-Bench 已在以下链接公开发布：this https URL。 

---
# Gating is Weighting: Understanding Gated Linear Attention through In-context Learning 

**Title (ZH)**: 门控是加权：通过在上下文学习理解门控线性注意力 

**Authors**: Yingcong Li, Davoud Ataee Tarzanagh, Ankit Singh Rawat, Maryam Fazel, Samet Oymak  

**Link**: [PDF](https://arxiv.org/pdf/2504.04308)  

**Abstract**: Linear attention methods offer a compelling alternative to softmax attention due to their efficiency in recurrent decoding. Recent research has focused on enhancing standard linear attention by incorporating gating while retaining its computational benefits. Such Gated Linear Attention (GLA) architectures include competitive models such as Mamba and RWKV. In this work, we investigate the in-context learning capabilities of the GLA model and make the following contributions. We show that a multilayer GLA can implement a general class of Weighted Preconditioned Gradient Descent (WPGD) algorithms with data-dependent weights. These weights are induced by the gating mechanism and the input, enabling the model to control the contribution of individual tokens to prediction. To further understand the mechanics of this weighting, we introduce a novel data model with multitask prompts and characterize the optimization landscape of learning a WPGD algorithm. Under mild conditions, we establish the existence and uniqueness (up to scaling) of a global minimum, corresponding to a unique WPGD solution. Finally, we translate these findings to explore the optimization landscape of GLA and shed light on how gating facilitates context-aware learning and when it is provably better than vanilla linear attention. 

**Abstract (ZH)**: Gated Linear Attention模型的上下文学习能力及其实现加权预条件梯度下降算法的研究 

---
# Sigma: A dataset for text-to-code semantic parsing with statistical analysis 

**Title (ZH)**: Sigma：一个用于统计分析的文本到代码语义解析数据集 

**Authors**: Saleh Almohaimeed, Shenyang Liu, May Alsofyani, Saad Almohaimeed, Liqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04301)  

**Abstract**: In the domain of semantic parsing, significant progress has been achieved in Text-to-SQL and question-answering tasks, both of which focus on extracting information from data sources in their native formats. However, the inherent constraints of their formal meaning representations, such as SQL programming language or basic logical forms, hinder their ability to analyze data from various perspectives, such as conducting statistical analyses. To address this limitation and inspire research in this field, we design SIGMA, a new dataset for Text-to-Code semantic parsing with statistical analysis. SIGMA comprises 6000 questions with corresponding Python code labels, spanning across 160 databases. Half of the questions involve query types, which return information in its original format, while the remaining 50% are statistical analysis questions, which perform statistical operations on the data. The Python code labels in our dataset cover 4 types of query types and 40 types of statistical analysis patterns. We evaluated the SIGMA dataset using three different baseline models: LGESQL, SmBoP, and SLSQL. The experimental results show that the LGESQL model with ELECTRA outperforms all other models, achieving 83.37% structure accuracy. In terms of execution accuracy, the SmBoP model, when combined with GraPPa and T5, reaches 76.38%. 

**Abstract (ZH)**: 在语义解析的领域，已在文本到SQL和问答任务中取得了显著进展，两者均专注于从数据源的原始格式中提取信息。然而，其形式意义表示的固有约束，如SQL编程语言或基本逻辑形式，限制了它们从多个角度分析数据的能力，例如进行统计分析。为解决这一局限并激发该领域的新研究，我们设计了SIGMA，一个用于文本到代码语义解析的新数据集，涵盖统计分析。SIGMA包含6000个问题及其相应的Python代码标签，覆盖160个数据库。其中一半的问题涉及查询类型，返回数据的原始信息；另一半是统计分析问题，对数据进行统计操作。我们数据集中的Python代码标签涵盖4种查询类型和40种统计分析模式。我们使用三种不同的baseline模型评估了SIGMA数据集：LGESQL、SmBoP和SLSQL。实验结果显示，使用ELECTRA的LGESQL模型在结构准确性上表现出色，达到83.37%。在执行准确性方面，结合GraPPa和T5的SmBoP模型达到76.38%。 

---
# AI-induced sexual harassment: Investigating Contextual Characteristics and User Reactions of Sexual Harassment by a Companion Chatbot 

**Title (ZH)**: AI引发的性骚扰：探究伴侣聊天机器人引起的性骚扰的语境特征及用户反应 

**Authors**: Mohammad, Namvarpour, Harrison Pauwels, Afsaneh Razi  

**Link**: [PDF](https://arxiv.org/pdf/2504.04299)  

**Abstract**: Advancements in artificial intelligence (AI) have led to the increase of conversational agents like Replika, designed to provide social interaction and emotional support. However, reports of these AI systems engaging in inappropriate sexual behaviors with users have raised significant concerns. In this study, we conducted a thematic analysis of user reviews from the Google Play Store to investigate instances of sexual harassment by the Replika chatbot. From a dataset of 35,105 negative reviews, we identified 800 relevant cases for analysis. Our findings revealed that users frequently experience unsolicited sexual advances, persistent inappropriate behavior, and failures of the chatbot to respect user boundaries. Users expressed feelings of discomfort, violation of privacy, and disappointment, particularly when seeking a platonic or therapeutic AI companion. This study highlights the potential harms associated with AI companions and underscores the need for developers to implement effective safeguards and ethical guidelines to prevent such incidents. By shedding light on user experiences of AI-induced harassment, we contribute to the understanding of AI-related risks and emphasize the importance of corporate responsibility in developing safer and more ethical AI systems. 

**Abstract (ZH)**: 人工智能（AI）的进步导致了像Replika这样的对话代理的增加，这些代理旨在提供社会互动和情感支持。然而，这些AI系统与用户进行不当性行为的报告引起了严重关切。本研究通过分析Google Play Store的用户评论，进行了主题分析，以调查Replika聊天机器人性骚扰的案例。从35,105条负面评论的数据集中，我们识别出800个相关案例进行分析。研究发现，用户经常经历不受欢迎的性暗示、持续的不当行为以及聊天机器人的边界尊重失败。用户感到不适、隐私被侵犯，并且对于寻求 platonic 或治疗性AI伴侣时感到失望。本研究强调了AI伴侣可能带来的危害，并突显了开发者需要实施有效的防护措施和伦理规范以防止此类事件的重要性。通过揭示由AI引起的骚扰用户经验，本研究有助于理解与AI相关的风险，并强调企业在开发更安全和更具伦理性的AI系统中的责任。 

---
# CATS: Mitigating Correlation Shift for Multivariate Time Series Classification 

**Title (ZH)**: CATS: 减轻多变量时间序列分类中的相关性偏移 

**Authors**: Xiao Lin, Zhichen Zeng, Tianxin Wei, Zhining Liu, Yuzhong chen, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2504.04283)  

**Abstract**: Unsupervised Domain Adaptation (UDA) leverages labeled source data to train models for unlabeled target data. Given the prevalence of multivariate time series (MTS) data across various domains, the UDA task for MTS classification has emerged as a critical challenge. However, for MTS data, correlations between variables often vary across domains, whereas most existing UDA works for MTS classification have overlooked this essential characteristic. To bridge this gap, we introduce a novel domain shift, {\em correlation shift}, measuring domain differences in multivariate correlation. To mitigate correlation shift, we propose a scalable and parameter-efficient \underline{C}orrelation \underline{A}dapter for M\underline{TS} (CATS). Designed as a plug-and-play technique compatible with various Transformer variants, CATS employs temporal convolution to capture local temporal patterns and a graph attention module to model the changing multivariate correlation. The adapter reweights the target correlations to align the source correlations with a theoretically guaranteed precision. A correlation alignment loss is further proposed to mitigate correlation shift, bypassing the alignment challenge from the non-i.i.d. nature of MTS data. Extensive experiments on four real-world datasets demonstrate that (1) compared with vanilla Transformer-based models, CATS increases over $10\%$ average accuracy while only adding around $1\%$ parameters, and (2) all Transformer variants equipped with CATS either reach or surpass state-of-the-art baselines. 

**Abstract (ZH)**: 无监督领域适应（UDA）利用标记的源数据来训练模型以处理未标记的目标数据。随着多变量时间序列（MTS）数据在各个领域中的普遍存在，MTS分类的UDA任务已成为一个关键挑战。然而，对于MTS数据而言，变量之间的相关性往往在不同领域中有所不同，而现有大多数MTS分类的UDA工作尚未注意到这一基本特征。为了弥合这一差距，我们引入了一个新的领域差异——相关性差异，即测量多变量相关性在不同领域的差异。为减轻相关性差异，我们提出了一种可扩展且参数高效的多变量时间序列（MTS）相关性适配器（CATS）。CATS设计为插即用技术，兼容各种Transformer变体，通过时序卷积捕获局部时序模式，并通过图注意力模块建模不断变化的多变量相关性。适配器重新加权目标相关性，与源相关性实现理论上保证的精确对齐。还提出了一种相关性对齐损失来减轻相关性差异，绕过了MTS数据非独立同分布性质带来的对齐挑战。在四个真实世界数据集上的广泛实验表明：（1）与 vanilla Transformer 基准模型相比，CATS在平均准确率上提高了超过10%，同时只添加了大约1%的参数；（2）所有配备CATS的Transformer变体要么达到了要么超越了最先进的基线。 

---
# Beyond the Hype: Embeddings vs. Prompting for Multiclass Classification Tasks 

**Title (ZH)**: 超越 hype：嵌入表示与提示在多类分类任务中的比较 

**Authors**: Marios Kokkodis, Richard Demsyn-Jones, Vijay Raghavan  

**Link**: [PDF](https://arxiv.org/pdf/2504.04277)  

**Abstract**: Are traditional classification approaches irrelevant in this era of AI hype? We show that there are multiclass classification problems where predictive models holistically outperform LLM prompt-based frameworks. Given text and images from home-service project descriptions provided by Thumbtack customers, we build embeddings-based softmax models that predict the professional category (e.g., handyman, bathroom remodeling) associated with each problem description. We then compare against prompts that ask state-of-the-art LLM models to solve the same problem. We find that the embeddings approach outperforms the best LLM prompts in terms of accuracy, calibration, latency, and financial cost. In particular, the embeddings approach has 49.5% higher accuracy than the prompting approach, and its superiority is consistent across text-only, image-only, and text-image problem descriptions. Furthermore, it yields well-calibrated probabilities, which we later use as confidence signals to provide contextualized user experience during deployment. On the contrary, prompting scores are overly uninformative. Finally, the embeddings approach is 14 and 81 times faster than prompting in processing images and text respectively, while under realistic deployment assumptions, it can be up to 10 times cheaper. Based on these results, we deployed a variation of the embeddings approach, and through A/B testing we observed performance consistent with our offline analysis. Our study shows that for multiclass classification problems that can leverage proprietary datasets, an embeddings-based approach may yield unequivocally better results. Hence, scientists, practitioners, engineers, and business leaders can use our study to go beyond the hype and consider appropriate predictive models for their classification use cases. 

**Abstract (ZH)**: 传统分类方法在AI热潮 era 是否已无 relevance？我们展示了存在一类多分类问题，在这些问题上基于模型的整体预测性能优于基于大语言模型提示的方法。给定 Thumbtack 客户提供的家庭服务项目描述中的文本和图像，我们构建了基于嵌入的 softmax 模型，用于预测每个问题描述相关的专业类别（例如，家庭修理工、浴室翻新）。然后我们将这些模型与要求最先进的大语言模型解决相同问题的提示进行对比。我们发现，嵌入方法在准确率、校准、延迟和成本方面均优于最优大语言模型提示，嵌入方法的准确率比提示方法高出 49.5%，并且其优势在仅文本、仅图像和图文问题描述中是一致的。此外，嵌入方法生成了良好校准的概率，我们利用这些概率作为置信信号，在部署过程中提供上下文相关的用户体验。相反，提示分数过于冗余，缺乏信息性。最后，嵌入方法在处理图像和文本时分别快 14 倍和 81 倍，而在现实部署假设下，它可以便宜 10 倍以上。基于这些结果，我们部署了嵌入方法的一种变体，并通过 A/B 测试观察到了与离线分析一致的性能。我们的研究显示，对于能够利用专用数据集的多分类问题，基于嵌入的方法可能绝对能获得更好的结果。因此，科学家、从业人员、工程师和业务领导者可以利用我们的研究超越炒作，考虑适用于其分类应用场景的适当预测模型。 

---
# LOGLO-FNO: Efficient Learning of Local and Global Features in Fourier Neural Operators 

**Title (ZH)**: LOGLO-FNO: 有效地学习局部和全局特征的傅里叶神经运算符 

**Authors**: Marimuthu Kalimuthu, David Holzmüller, Mathias Niepert  

**Link**: [PDF](https://arxiv.org/pdf/2504.04260)  

**Abstract**: Modeling high-frequency information is a critical challenge in scientific machine learning. For instance, fully turbulent flow simulations of Navier-Stokes equations at Reynolds numbers 3500 and above can generate high-frequency signals due to swirling fluid motions caused by eddies and vortices. Faithfully modeling such signals using neural networks depends on accurately reconstructing moderate to high frequencies. However, it has been well known that deep neural nets exhibit the so-called spectral bias toward learning low-frequency components. Meanwhile, Fourier Neural Operators (FNOs) have emerged as a popular class of data-driven models in recent years for solving Partial Differential Equations (PDEs) and for surrogate modeling in general. Although impressive results have been achieved on several PDE benchmark problems, FNOs often perform poorly in learning non-dominant frequencies characterized by local features. This limitation stems from the spectral bias inherent in neural networks and the explicit exclusion of high-frequency modes in FNOs and their variants. Therefore, to mitigate these issues and improve FNO's spectral learning capabilities to represent a broad range of frequency components, we propose two key architectural enhancements: (i) a parallel branch performing local spectral convolutions (ii) a high-frequency propagation module. Moreover, we propose a novel frequency-sensitive loss term based on radially binned spectral errors. This introduction of a parallel branch for local convolutions reduces number of trainable parameters by up to 50% while achieving the accuracy of baseline FNO that relies solely on global convolutions. Experiments on three challenging PDE problems in fluid mechanics and biological pattern formation, and the qualitative and spectral analysis of predictions show the effectiveness of our method over the state-of-the-art neural operator baselines. 

**Abstract (ZH)**: 高频率信息建模是科学机器学习中的关键挑战。例如，湍流流动的纳维-斯托克斯方程模拟在雷诺数3500及以上的湍流流动中，由于涡旋和旋涡引起的旋转流体运动会产生高频率信号。使用神经网络忠实地建模此类信号依赖于对中高频信号的准确重构。然而，众所周知，深层神经网络表现出对学习低频成分的频谱偏见。与此同时，傅里叶神经算子（FNOs）已成为近年来用于解决偏微分方程（PDEs）和一般 surrogate 模型的流行数据驱动模型。尽管在多个PDE基准问题上取得了令人印象深刻的成果，但FNOs在学习由局部特征定义的次主导频率方面表现不佳。这一限制源于神经网络固有的频谱偏见以及FNO及其变体中显式排除高频模态。因此，为了缓解这些问题并提高FNO的频谱学习能力以代表广泛的频谱成分，我们提出了两个关键的架构增强：（i）一个并行分支执行局部谱卷积；（ii）一个高频率传播模块。此外，我们提出了一种基于径向分箱谱误差的新颖频谱敏感损失项。通过引入一个并行分支执行局部卷积，我们能将可训练参数的数量最多减少50%，同时仍能达到仅依赖全局卷积的基线FNO的准确性。在流体力学和生物模式形成中的三个具有挑战性的PDE问题上的实验及预测的定性和谱分析表明，我们的方法优于最先进的神经算子基线。 

---
# Progressive Multi-Source Domain Adaptation for Personalized Facial Expression Recognition 

**Title (ZH)**: 渐进多源域适应的个性化面部表情识别 

**Authors**: Muhammad Osama Zeeshan, Marco Pedersoli, Alessandro Lameiras Koerich, Eric Grange  

**Link**: [PDF](https://arxiv.org/pdf/2504.04252)  

**Abstract**: Personalized facial expression recognition (FER) involves adapting a machine learning model using samples from labeled sources and unlabeled target domains. Given the challenges of recognizing subtle expressions with considerable interpersonal variability, state-of-the-art unsupervised domain adaptation (UDA) methods focus on the multi-source UDA (MSDA) setting, where each domain corresponds to a specific subject, and improve model accuracy and robustness. However, when adapting to a specific target, the diverse nature of multiple source domains translates to a large shift between source and target data. State-of-the-art MSDA methods for FER address this domain shift by considering all the sources to adapt to the target representations. Nevertheless, adapting to a target subject presents significant challenges due to large distributional differences between source and target domains, often resulting in negative transfer. In addition, integrating all sources simultaneously increases computational costs and causes misalignment with the target. To address these issues, we propose a progressive MSDA approach that gradually introduces information from subjects based on their similarity to the target subject. This will ensure that only the most relevant sources from the target are selected, which helps avoid the negative transfer caused by dissimilar sources. We first exploit the closest sources to reduce the distribution shift with the target and then move towards the furthest while only considering the most relevant sources based on the predetermined threshold. Furthermore, to mitigate catastrophic forgetting caused by the incremental introduction of source subjects, we implemented a density-based memory mechanism that preserves the most relevant historical source samples for adaptation. Our experiments show the effectiveness of our proposed method on pain datasets: Biovid and UNBC-McMaster. 

**Abstract (ZH)**: 个性化面部表情识别（FER）涉及使用标记源和未标记目标域的样本调整机器学习模型。鉴于识别细微表情存在显著个体差异的挑战，最新的无监督域适应（UDA）方法集中在多源UDA（MSDA）设置上，每个域对应特定的个体，并提高模型准确性和鲁棒性。然而，在适应特定目标时，多个源域的多样性会导致源域和目标域之间出现较大的数据差异。最先进的FER MSDA方法通过考虑所有源域以适应目标表示来解决这一域差异问题。尽管如此，适应特定目标主体因源域和目标域之间巨大的分布差异而面临重大挑战，往往导致负迁移。此外，同时集成所有源域会增加计算成本并导致与目标主体的对齐偏差。为了应对这些问题，我们提出了一种渐进式的MSDA方法，根据其与目标主体的相似性逐步引入主体信息。这将确保仅选择与目标最相关的源，从而避免因不相似的源导致的负迁移。我们首先利用最近的源域减少与目标之间的分布差异，然后逐步向最远的源域推进，同时只考虑基于预设阈值的最相关源域。此外，为了缓解逐步引入源主体引起的灾难性遗忘，我们实现了一种基于密度的记忆机制，保留最相关的历史源样本以进行适应。我们的实验表明，该方法在疼痛数据集Biovid和UNBC-McMaster上具有有效性。 

---
# Task load dependent decision referrals for joint binary classification in human-automation teams 

**Title (ZH)**: 基于任务负载的决策转介在人类-自动化团队联合二分类中的应用 

**Authors**: Kesav Kaza, Jerome Le Ny, Aditya Mahajan  

**Link**: [PDF](https://arxiv.org/pdf/2504.04248)  

**Abstract**: We consider the problem of optimal decision referrals in human-automation teams performing binary classification tasks. The automation, which includes a pre-trained classifier, observes data for a batch of independent tasks, analyzes them, and may refer a subset of tasks to a human operator for fresh and final analysis. Our key modeling assumption is that human performance degrades with task load. We model the problem of choosing which tasks to refer as a stochastic optimization problem and show that, for a given task load, it is optimal to myopically refer tasks that yield the largest reduction in expected cost, conditional on the observed data. This provides a ranking scheme and a policy to determine the optimal set of tasks for referral. We evaluate this policy against a baseline through an experimental study with human participants. Using a radar screen simulator, participants made binary target classification decisions under time constraint. They were guided by a decision rule provided to them, but were still prone to errors under time pressure. An initial experiment estimated human performance model parameters, while a second experiment compared two referral policies. Results show statistically significant gains for the proposed optimal referral policy over a blind policy that determines referrals using the automation and human-performance models but not based on the observed data. 

**Abstract (ZH)**: 我们在执行二分类任务的人机团队中考虑最优决策转介问题。自动化系统包含一个预训练分类器，可以观察一批独立任务的数据，进行分析，并可能将部分任务转介给人类操作员进行新鲜和最终分析。我们关键的建模假设是人类性能随任务负载增加而下降。我们将选择转介哪些任务的问题建模为一个随机优化问题，并证明，在给定任务负载的情况下，最优策略是在观察到数据的条件下，转介能最大程度降低预期成本的任务。这提供了一种排名方案和决策策略，用于确定最优转介任务集。我们通过一项以人类参与者为对象的实验研究，将此策略与基线进行了评估。在雷达屏幕模拟器中，参与者在时间限制下进行二分类目标决策，并受到提供的决策规则的指导，但在时间压力下仍可能出现错误。第一次实验估算了人类性能模型参数，而第二次实验则比较了两种转介策略。结果表明，在观察到的数据基础上，所提出的最优转介策略比仅基于自动化和人类性能模型而未基于观测数据进行转介的盲目策略具有统计显著性优势。 

---
# From Automation to Autonomy in Smart Manufacturing: A Bayesian Optimization Framework for Modeling Multi-Objective Experimentation and Sequential Decision Making 

**Title (ZH)**: 从自动化到自主化：一种用于 modeling 多目标实验和序列决策的贝叶斯优化框架 

**Authors**: Avijit Saha Asru, Hamed Khosravi, Imtiaz Ahmed, Abdullahil Azeem  

**Link**: [PDF](https://arxiv.org/pdf/2504.04244)  

**Abstract**: Discovering novel materials with desired properties is essential for driving innovation. Industry 4.0 and smart manufacturing have promised transformative advances in this area through real-time data integration and automated production planning and control. However, the reliance on automation alone has often fallen short, lacking the flexibility needed for complex processes. To fully unlock the potential of smart manufacturing, we must evolve from automation to autonomous systems that go beyond rigid programming and can dynamically optimize the search for solutions. Current discovery approaches are often slow, requiring numerous trials to find optimal combinations, and costly, particularly when optimizing multiple properties simultaneously. This paper proposes a Bayesian multi-objective sequential decision-making (BMSDM) framework that can intelligently select experiments as manufacturing progresses, guiding us toward the discovery of optimal design faster and more efficiently. The framework leverages sequential learning through Bayesian Optimization, which iteratively refines a statistical model representing the underlying manufacturing process. This statistical model acts as a surrogate, allowing for efficient exploration and optimization without requiring numerous real-world experiments. This approach can significantly reduce the time and cost of data collection required by traditional experimental designs. The proposed framework is compared with traditional DoE methods and two other multi-objective optimization methods. Using a manufacturing dataset, we evaluate and compare the performance of these approaches across five evaluation metrics. BMSDM comprehensively outperforms the competing methods in multi-objective decision-making scenarios. Our proposed approach represents a significant leap forward in creating an intelligent autonomous platform capable of novel material discovery. 

**Abstract (ZH)**: 利用贝叶斯多目标顺序决策框架实现智能自主材料发现 

---
# Perils of Label Indeterminacy: A Case Study on Prediction of Neurological Recovery After Cardiac Arrest 

**Title (ZH)**: 标签不确定性的风险：心脏骤停后神经功能恢复预测的案例研究 

**Authors**: Jakob Schoeffer, Maria De-Arteaga, Jonathan Elmer  

**Link**: [PDF](https://arxiv.org/pdf/2504.04243)  

**Abstract**: The design of AI systems to assist human decision-making typically requires the availability of labels to train and evaluate supervised models. Frequently, however, these labels are unknown, and different ways of estimating them involve unverifiable assumptions or arbitrary choices. In this work, we introduce the concept of label indeterminacy and derive important implications in high-stakes AI-assisted decision-making. We present an empirical study in a healthcare context, focusing specifically on predicting the recovery of comatose patients after resuscitation from cardiac arrest. Our study shows that label indeterminacy can result in models that perform similarly when evaluated on patients with known labels, but vary drastically in their predictions for patients where labels are unknown. After demonstrating crucial ethical implications of label indeterminacy in this high-stakes context, we discuss takeaways for evaluation, reporting, and design. 

**Abstract (ZH)**: AI系统设计以辅助人类决策通常需要标签来训练和评估监督模型。然而，这些标签往往未知，并且估计它们的不同方法涉及不可验证假设或任意选择。在本研究中，我们介绍了标签不定性这一概念，并推导出其在高风险AI辅助决策中的重要影响。我们以医疗保健领域为例，专注于预测心脏骤停后复苏的昏迷患者恢复情况。研究显示，标签不定性可能导致模型在使用已知标签的患者上表现相似，但在未知标签的患者上预测结果差异巨大。在展示标签不定性在这一高风险领域中的关键伦理影响后，我们讨论了评估、报告和设计方面的启示。 

---
# oneDAL Optimization for ARM Scalable Vector Extension: Maximizing Efficiency for High-Performance Data Science 

**Title (ZH)**: oneDAL优化针对ARM可扩展向量扩展：最大化高性能数据科学的效率 

**Authors**: Chandan Sharma, Rakshith GB, Ajay Kumar Patel, Dhanus M Lal, Darshan Patel, Ragesh Hajela, Masahiro Doteguchi, Priyanka Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2504.04241)  

**Abstract**: The evolution of ARM-based architectures, particularly those incorporating Scalable Vector Extension (SVE), has introduced transformative opportunities for high-performance computing (HPC) and machine learning (ML) workloads. The Unified Acceleration Foundation's (UXL) oneAPI Data Analytics Library (oneDAL) is a widely adopted library for accelerating ML and data analytics workflows, but its reliance on Intel's proprietary Math Kernel Library (MKL) has traditionally limited its compatibility to x86platforms. This paper details the porting of oneDAL to ARM architectures with SVE support, using OpenBLAS as an alternative backend to overcome architectural and performance challenges. Beyond porting, the research introduces novel ARM-specific optimizations, including custom sparse matrix routines, vectorized statistical functions, and a Scalable Vector Extension (SVE)-optimized Support Vector Machine (SVM) algorithm. The SVM enhancements leverage SVE's flexible vector lengths and predicate driven execution, achieving notable performance gains of 22% for the Boser method and 5% for the Thunder method. Benchmarks conducted on ARM SVE-enabled AWSGraviton3 instances showcase up to 200x acceleration in ML training and inference tasks compared to the original scikit-learn implementation on the ARM platform. Moreover, the ARM-optimized oneDAL achieves performance parity with, and in some cases exceeds, the x86 oneDAL implementation (MKL backend) on IceLake x86 systems, which are nearly twice as costly as AWSGraviton3 ARM instances. These findings highlight ARM's potential as a high-performance, energyefficient platform for dataintensive ML applications. By expanding cross-architecture compatibility and contributing to the opensource ecosystem, this work reinforces ARM's position as a competitive alternative in the HPC and ML domains, paving the way for future advancements in dataintensive computing. 

**Abstract (ZH)**: 基于ARM架构，特别是支持可扩展向量扩展（SVE）的架构，对高性能计算（HPC）和机器学习（ML）工作负载引发了变革性机会。统一加速基金会（UXL）的一体化数据分析库（oneDAL）是一个广泛采用的库，用于加速ML和数据分析流程，但其对英特尔专有数学内核库（MKL）的依赖传统上限制了其在x86平台上的兼容性。本文详细介绍了将oneDAL移植到支持SVE的ARM架构的过程，使用OpenBLAS作为替代后端以克服架构和性能挑战。除了移植，研究还引入了针对ARM架构的新型优化，包括自定义稀疏矩阵算法、向量化统计函数以及优化后的支持向量机（SVM）算法。SVM增强利用了SVE的灵活向量长度和基于谓词的执行，分别实现了博斯方法22%和雷神方法5%的性能提升。在AWS Graviton3实例上进行的基准测试表明，与原始ARM平台上的scikit-learn实现相比，在ML训练和推理任务中可实现高达200倍的加速。此外，ARM优化的一体化数据分析库在IceLake x86系统上的性能与x86一体化数据分析库（MKL后端）相当，在某些情况下甚至超过，而IceLake x86系统的成本几乎是AWS Graviton3 ARM实例的两倍。这些发现突显了ARM架构在数据密集型ML应用中作为高性能、能效平台的潜力。通过扩展跨架构兼容性和促进开源生态系统的发展，这项工作巩固了ARM在HPC和ML领域的竞争力地位，为数据密集型计算的未来进步铺平了道路。 

---
# Sensitivity Meets Sparsity: The Impact of Extremely Sparse Parameter Patterns on Theory-of-Mind of Large Language Models 

**Title (ZH)**: 敏感性与稀疏性：极稀疏参数模式对大型语言模型心智理论的影响 

**Authors**: Yuheng Wu, Wentao Guo, Zirui Liu, Heng Ji, Zhaozhuo Xu, Denghui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04238)  

**Abstract**: This paper investigates the emergence of Theory-of-Mind (ToM) capabilities in large language models (LLMs) from a mechanistic perspective, focusing on the role of extremely sparse parameter patterns. We introduce a novel method to identify ToM-sensitive parameters and reveal that perturbing as little as 0.001% of these parameters significantly degrades ToM performance while also impairing contextual localization and language understanding. To understand this effect, we analyze their interaction with core architectural components of LLMs. Our findings demonstrate that these sensitive parameters are closely linked to the positional encoding module, particularly in models using Rotary Position Embedding (RoPE), where perturbations disrupt dominant-frequency activations critical for contextual processing. Furthermore, we show that perturbing ToM-sensitive parameters affects LLM's attention mechanism by modulating the angle between queries and keys under positional encoding. These insights provide a deeper understanding of how LLMs acquire social reasoning abilities, bridging AI interpretability with cognitive science. Our results have implications for enhancing model alignment, mitigating biases, and improving AI systems designed for human interaction. 

**Abstract (ZH)**: 本研究从机制角度探讨了大型语言模型（LLM）中理论思维（ToM）能力的 emergence，重点关注极稀疏参数模式的作用。我们提出了一种新型方法来识别 ToM 敏感参数，并揭示出扰动这些参数的 0.001% 可显著降低 ToM 性能，同时损害上下文定位和语言理解。为了理解这一效应，我们分析了它们与 LLM 核心架构组件的相互作用。研究发现，这些敏感参数与位置编码模块密切相关，特别是在使用旋转位置嵌入（RoPE）的模型中，扰动会破坏对上下文处理至关重要的主导频率激活。此外，我们展示了扰动 ToM 敏感参数如何通过调节编码位置下查询和密钥之间的角度影响 LLM 的注意机制。这些洞察为我们理解 LLM 如何获得社会推理能力提供了更深入的理解，将 AI 可解释性与认知科学联系起来。我们的结果对增强模型对齐、减轻偏见以及改进设计用于人类互动的 AI 系统具有重要意义。 

---
# TrafficLLM: Enhancing Large Language Models for Network Traffic Analysis with Generic Traffic Representation 

**Title (ZH)**: TrafficLLM: 通过通用traffic表示增强大型语言模型在网络流量分析中的性能 

**Authors**: Tianyu Cui, Xinjie Lin, Sijia Li, Miao Chen, Qilei Yin, Qi Li, Ke Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04222)  

**Abstract**: Machine learning (ML) powered network traffic analysis has been widely used for the purpose of threat detection. Unfortunately, their generalization across different tasks and unseen data is very limited. Large language models (LLMs), known for their strong generalization capabilities, have shown promising performance in various domains. However, their application to the traffic analysis domain is limited due to significantly different characteristics of network traffic. To address the issue, in this paper, we propose TrafficLLM, which introduces a dual-stage fine-tuning framework to learn generic traffic representation from heterogeneous raw traffic data. The framework uses traffic-domain tokenization, dual-stage tuning pipeline, and extensible adaptation to help LLM release generalization ability on dynamic traffic analysis tasks, such that it enables traffic detection and traffic generation across a wide range of downstream tasks. We evaluate TrafficLLM across 10 distinct scenarios and 229 types of traffic. TrafficLLM achieves F1-scores of 0.9875 and 0.9483, with up to 80.12% and 33.92% better performance than existing detection and generation methods. It also shows strong generalization on unseen traffic with an 18.6% performance improvement. We further evaluate TrafficLLM in real-world scenarios. The results confirm that TrafficLLM is easy to scale and achieves accurate detection performance on enterprise traffic. 

**Abstract (ZH)**: 基于机器学习的网络流量分析在威胁检测中得到了广泛应用，但其在不同任务和未见数据上的泛化能力非常有限。大型语言模型（LLMs）因其强大的泛化能力，在多个领域展现出了有前途的表现。然而，由于网络流量的特性与LLMs存在显著差异，其在流量分析领域的应用受到了限制。为解决这一问题，本文提出TrafficLLM，一种双阶段 fine-tuning 框架，用于从异构的原始流量数据中学习通用的流量表示。该框架通过网络流量领域的分词、双阶段调优管道以及可扩展的适应机制，帮助LLMs在动态流量分析任务上释放泛化能力，从而使得流量检测和流量生成适用于广泛的下游任务。我们在10种不同的场景和229种类型的流量上评估了TrafficLLM，结果显示其F1分数分别为0.9875和0.9483，比现有检测和生成方法分别高出了80.12%和33.92%。它在未见流量上的泛化能力也表现出色，性能提升了18.6%。我们进一步在实际场景中评估了TrafficLLM，结果表明TrafficLLM易于扩展，并在企业流量检测上实现了准确的检测性能。 

---
# Towards Understanding and Improving Refusal in Compressed Models via Mechanistic Interpretability 

**Title (ZH)**: 通过机械解释性方法理解并改善压缩模型中的拒绝服务 

**Authors**: Vishnu Kabir Chhabra, Mohammad Mahdi Khalili  

**Link**: [PDF](https://arxiv.org/pdf/2504.04215)  

**Abstract**: The rapid growth of large language models has spurred significant interest in model compression as a means to enhance their accessibility and practicality. While extensive research has explored model compression through the lens of safety, findings suggest that safety-aligned models often lose elements of trustworthiness post-compression. Simultaneously, the field of mechanistic interpretability has gained traction, with notable discoveries, such as the identification of a single direction in the residual stream mediating refusal behaviors across diverse model architectures. In this work, we investigate the safety of compressed models by examining the mechanisms of refusal, adopting a novel interpretability-driven perspective to evaluate model safety. Furthermore, leveraging insights from our interpretability analysis, we propose a lightweight, computationally efficient method to enhance the safety of compressed models without compromising their performance or utility. 

**Abstract (ZH)**: 大规模语言模型的快速增长激发了对模型压缩的兴趣，以提高其易用性和实用性。尽管已有大量研究从安全性的角度探索模型压缩，研究表明，安全对齐的模型在压缩后往往会失去部分可信度。同时，机制可解释性领域也取得了进展，例如发现残差流中的一个方向在跨多种模型架构中调节拒绝行为。在本工作中，我们通过研究拒绝机制来探讨压缩模型的安全性，并采用一种新颖的可解释性驱动视角评估模型安全性。此外，基于我们的可解释性分析所得洞察，我们提出了一种轻量级、计算效率高的方法，以增强压缩模型的安全性而不牺牲其性能或实用性。 

---
# Adaptive Elicitation of Latent Information Using Natural Language 

**Title (ZH)**: 基于自然语言的潜在信息自适应提取 

**Authors**: Jimmy Wang, Thomas Zollo, Richard Zemel, Hongseok Namkoong  

**Link**: [PDF](https://arxiv.org/pdf/2504.04204)  

**Abstract**: Eliciting information to reduce uncertainty about a latent entity is a critical task in many application domains, e.g., assessing individual student learning outcomes, diagnosing underlying diseases, or learning user preferences. Though natural language is a powerful medium for this purpose, large language models (LLMs) and existing fine-tuning algorithms lack mechanisms for strategically gathering information to refine their own understanding of the latent entity. To harness the generalization power and world knowledge of LLMs in developing effective information-gathering strategies, we propose an adaptive elicitation framework that actively reduces uncertainty on the latent entity. Since probabilistic modeling of an abstract latent entity is difficult, our framework adopts a predictive view of uncertainty, using a meta-learned language model to simulate future observations and enable scalable uncertainty quantification over complex natural language. Through autoregressive forward simulation, our model quantifies how new questions reduce epistemic uncertainty, enabling the development of sophisticated information-gathering strategies to choose the most informative next queries. In experiments on the 20 questions game, dynamic opinion polling, and adaptive student assessment, our method consistently outperforms baselines in identifying critical unknowns and improving downstream predictions, illustrating the promise of strategic information gathering in natural language settings. 

**Abstract (ZH)**: 利用自适应引证框架减轻潜藏实体不确定性以提高自然语言应用领域中的信息收集策略有效性 

---
# Reasoning on Multiple Needles In A Haystack 

**Title (ZH)**: haystack 中的多个针线推理 

**Authors**: Yidong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04150)  

**Abstract**: The Needle In A Haystack (NIAH) task has been widely used to evaluate the long-context question-answering capabilities of Large Language Models (LLMs). However, its reliance on simple retrieval limits its effectiveness. To address this limitation, recent studies have introduced the Multiple Needles In A Haystack Reasoning (MNIAH-R) task, which incorporates supporting documents (Multiple needles) of multi-hop reasoning tasks into a distracting context (Haystack}). Despite this advancement, existing approaches still fail to address the issue of models providing direct answers from internal knowledge, and they do not explain or mitigate the decline in accuracy as context length increases. In this paper, we tackle the memory-based answering problem by filtering out direct-answer questions, and we reveal that performance degradation is primarily driven by the reduction in the length of the thinking process as the input length increases. Building on this insight, we decompose the thinking process into retrieval and reasoning stages and introduce a reflection mechanism for multi-round extension. We also train a model using the generated iterative thinking process, which helps mitigate the performance degradation. Furthermore, we demonstrate the application of this retrieval-reflection capability in mathematical reasoning scenarios, improving GPT-4o's performance on AIME2024. 

**Abstract (ZH)**: 基于记忆的答案过滤任务：多层次思路推理与反射机制在大型语言模型中的应用 

---
# My Life in Artificial Intelligence: People, anecdotes, and some lessons learnt 

**Title (ZH)**: 我的人工智能生活：人物、轶事及一些吸取的教训 

**Authors**: Kees van Deemter  

**Link**: [PDF](https://arxiv.org/pdf/2504.04142)  

**Abstract**: In this very personal workography, I relate my 40-year experiences as a researcher and educator in and around Artificial Intelligence (AI), more specifically Natural Language Processing. I describe how curiosity, and the circumstances of the day, led me to work in both industry and academia, and in various countries, including The Netherlands (Amsterdam, Eindhoven, and Utrecht), the USA (Stanford), England (Brighton), Scotland (Aberdeen), and China (Beijing and Harbin). People and anecdotes play a large role in my story; the history of AI forms its backdrop. I focus on things that might be of interest to (even) younger colleagues, given the choices they face in their own work and life at a time when AI is finally emerging from the shadows. 

**Abstract (ZH)**: 在这份极具个人色彩的工作经历中，我回顾了自己40年在人工智能（AI），特别是自然语言处理领域的研究与教育经验。我描述了好奇心以及当时的环境如何促使我先后在工业界和学术界工作，并且足迹遍布荷兰（阿姆斯特丹、埃因霍温、乌特勒支）、美国（斯坦福）、英国（布赖顿）、苏格兰（阿伯丁）以及中国（北京、哈尔滨）等多地。人物与轶事在我的故事中占据了重要位置，人工智能的历史则是这一故事的背景。我着重讲述了对年轻同事们可能感兴趣的内容，特别是在人工智能终于崭露头角之际，他们所面临的各种工作和生活选择。 

---
# Predicting Soil Macronutrient Levels: A Machine Learning Approach Models Trained on pH, Conductivity, and Average Power of Acid-Base Solutions 

**Title (ZH)**: 预测土壤宏量营养素水平：一种基于pH值、电导率和酸碱溶液平均功率的机器学习方法 

**Authors**: Mridul Kumar, Deepali Jain, Zeeshan Saifi, Soami Daya Krishnananda  

**Link**: [PDF](https://arxiv.org/pdf/2504.04138)  

**Abstract**: Soil macronutrients, particularly potassium ions (K$^+$), are indispensable for plant health, underpinning various physiological and biological processes, and facilitating the management of both biotic and abiotic stresses. Deficient macronutrient content results in stunted growth, delayed maturation, and increased vulnerability to environmental stressors, thereby accentuating the imperative for precise soil nutrient monitoring. Traditional techniques such as chemical assays, atomic absorption spectroscopy, inductively coupled plasma optical emission spectroscopy, and electrochemical methods, albeit advanced, are prohibitively expensive and time-intensive, thus unsuitable for real-time macronutrient assessment. In this study, we propose an innovative soil testing protocol utilizing a dataset derived from synthetic solutions to model soil behaviour. The dataset encompasses physical properties including conductivity and pH, with a concentration on three key macronutrients: nitrogen (N), phosphorus (P), and potassium (K). Four machine learning algorithms were applied to the dataset, with random forest regressors and neural networks being selected for the prediction of soil nutrient concentrations. Comparative analysis with laboratory soil testing results revealed prediction errors of 23.6% for phosphorus and 16% for potassium using the random forest model, and 26.3% for phosphorus and 21.8% for potassium using the neural network model. This methodology illustrates a cost-effective and efficacious strategy for real-time soil nutrient monitoring, offering substantial advancements over conventional techniques and enhancing the capability to sustain optimal nutrient levels conducive to robust crop growth. 

**Abstract (ZH)**: 土壤宏量营养素，特别是钾离子（K+），对植物健康至关重要，支撑着各种生理和生物学过程，并促进生物胁迫和非生物胁迫的管理。宏量营养素含量不足会导致生长受阻、成熟延迟以及对环境胁迫的敏感性增加，从而强调了精确土壤养分监测的迫切性。传统的化学分析、原子吸收光谱法、电感耦合等离子体光谱法和电化学方法虽然先进，但成本高昂且耗时，不适用于实时养分评估。在此研究中，我们提出了一种创新的土壤测试协议，利用从合成溶液中提取的数据集来模拟土壤行为。该数据集包括电导率和pH等物理性质，并重点研究氮（N）、磷（P）和钾（K）三种关键宏量营养素。应用了四种机器学习算法，随机森林回归器和神经网络被选择用于预测土壤营养素浓度。与实验室土壤测试结果的比较分析显示，使用随机森林模型预测磷和钾的误差分别为23.6%和16%，而使用神经网络模型的误差分别为26.3%和21.8%。该方法展示了成本效益高且有效的实时土壤养分监测策略，显著优于传统技术，增强了维持有利于作物生长的最优营养水平的能力。 

---
# Multi-identity Human Image Animation with Structural Video Diffusion 

**Title (ZH)**: 多身份人体图像动画生成中的结构视频扩散 

**Authors**: Zhenzhi Wang, Yixuan Li, Yanhong Zeng, Yuwei Guo, Dahua Lin, Tianfan Xue, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2504.04126)  

**Abstract**: Generating human videos from a single image while ensuring high visual quality and precise control is a challenging task, especially in complex scenarios involving multiple individuals and interactions with objects. Existing methods, while effective for single-human cases, often fail to handle the intricacies of multi-identity interactions because they struggle to associate the correct pairs of human appearance and pose condition and model the distribution of 3D-aware dynamics. To address these limitations, we present Structural Video Diffusion, a novel framework designed for generating realistic multi-human videos. Our approach introduces two core innovations: identity-specific embeddings to maintain consistent appearances across individuals and a structural learning mechanism that incorporates depth and surface-normal cues to model human-object interactions. Additionally, we expand existing human video dataset with 25K new videos featuring diverse multi-human and object interaction scenarios, providing a robust foundation for training. Experimental results demonstrate that Structural Video Diffusion achieves superior performance in generating lifelike, coherent videos for multiple subjects with dynamic and rich interactions, advancing the state of human-centric video generation. 

**Abstract (ZH)**: 从单张图像生成高质量、精确控制的多个人体视频是一项具有挑战性的任务，尤其是在涉及多人和物体交互的复杂场景中。现有方法虽然在单人体案例中有效，但在处理多身份间的复杂交互时往往失效，因为它们难以正确关联人体外观和姿态条件，并建模3D动态分布。为解决这些限制，我们提出了一种新型框架——结构化视频扩散（Structural Video Diffusion），专门用于生成逼真的人体视频。该方法引入了两项核心创新：身份特定嵌入以保持不同个体的外观一致，并结合深度和法线线索，引入结构学习机制以建模人体与物体的交互。此外，我们扩展了现有的人体视频数据集，新增了25K个包含多样化多人和物体交互场景的新视频，为训练提供了坚实的基础。实验结果表明，结构化视频扩散在生成具备动态和丰富交互的多主体逼真、连贯视频方面表现出优越性能，推动了以人体为中心的视频生成技术的发展。 

---
# TARAC: Mitigating Hallucination in LVLMs via Temporal Attention Real-time Accumulative Connection 

**Title (ZH)**: TARAC: 通过时间注意连接实时累积连接减轻LVLMs幻觉问题 

**Authors**: Chunzhao Xie, Tongxuan Liu, Lei Jiang, Yuting Zeng, jinrong Guo, Yunheng Shen, Weizhe Huang, Jing Li, Xiaohua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04099)  

**Abstract**: Large Vision-Language Models have demonstrated remarkable performance across various tasks; however, the challenge of hallucinations constrains their practical applications. The hallucination problem arises from multiple factors, including the inherent hallucinations in language models, the limitations of visual encoders in perception, and biases introduced by multimodal data. Extensive research has explored ways to mitigate hallucinations. For instance, OPERA prevents the model from overly focusing on "anchor tokens", thereby reducing hallucinations, whereas VCD mitigates hallucinations by employing a contrastive decoding approach. In this paper, we investigate the correlation between the decay of attention to image tokens and the occurrence of hallucinations. Based on this finding, we propose Temporal Attention Real-time Accumulative Connection (TARAC), a novel training-free method that dynamically accumulates and updates LVLMs' attention on image tokens during generation. By enhancing the model's attention to image tokens, TARAC mitigates hallucinations caused by the decay of attention on image tokens. We validate the effectiveness of TARAC across multiple models and datasets, demonstrating that our approach substantially mitigates hallucinations. In particular, TARAC reduces $C_S$ by 25.2 and $C_I$ by 8.7 compared to VCD on the CHAIR benchmark. 

**Abstract (ZH)**: 大规模多模态模型在各种任务中展现了出色的表现；然而，幻觉问题限制了它们的实际应用。幻觉问题由多种因素引起，包括语言模型固有的幻觉、视觉编码器感知能力的局限性以及多模态数据引入的偏差。广泛的研究探索了减轻幻觉的方法。例如，OPERA通过防止模型过度关注“锚定词元”从而减少幻觉，而VCD通过对比解码的方法减轻幻觉。在本文中，我们研究了注意力衰减对图像词元注意力与幻觉发生之间的关系。基于这一发现，我们提出了一种新的无需训练的方法——时间注意力实时累积连接（TARAC），该方法在生成过程中动态地累积和更新LVLMs对图像词元的注意力。通过增强模型对图像词元的注意力，TARAC减轻了由注意力衰减引起的幻觉。我们在多个模型和数据集上验证了TARAC的有效性，表明我们的方法显著减轻了幻觉。特别是在CHAIR基准上，TARAC相比VCD减少了$C_S$ 25.2和$C_I$ 8.7。 

---
# DocSAM: Unified Document Image Segmentation via Query Decomposition and Heterogeneous Mixed Learning 

**Title (ZH)**: DocSAM：基于查询分解和异构混合学习的统一文档图像分割 

**Authors**: Xiao-Hui Li, Fei Yin, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.04085)  

**Abstract**: Document image segmentation is crucial for document analysis and recognition but remains challenging due to the diversity of document formats and segmentation tasks. Existing methods often address these tasks separately, resulting in limited generalization and resource wastage. This paper introduces DocSAM, a transformer-based unified framework designed for various document image segmentation tasks, such as document layout analysis, multi-granularity text segmentation, and table structure recognition, by modelling these tasks as a combination of instance and semantic segmentation. Specifically, DocSAM employs Sentence-BERT to map category names from each dataset into semantic queries that match the dimensionality of instance queries. These two sets of queries interact through an attention mechanism and are cross-attended with image features to predict instance and semantic segmentation masks. Instance categories are predicted by computing the dot product between instance and semantic queries, followed by softmax normalization of scores. Consequently, DocSAM can be jointly trained on heterogeneous datasets, enhancing robustness and generalization while reducing computational and storage resources. Comprehensive evaluations show that DocSAM surpasses existing methods in accuracy, efficiency, and adaptability, highlighting its potential for advancing document image understanding and segmentation across various applications. Codes are available at this https URL. 

**Abstract (ZH)**: 文档图像分割对于文档分析和识别至关重要，但由于文档格式和分割任务的多样性，这仍然具有挑战性。现有方法通常针对这些任务分别处理，导致泛化能力有限和资源浪费。本文介绍了DocSAM，这是一种基于变压器的统一框架，旨在通过将这些任务建模为实例分割和语义分割的组合，用于各种文档图像分割任务，如文档布局分析、多粒度文本分割和表格结构识别。具体而言，DocSAM 使用 Sentence-BERT 将每个数据集的类别名称映射为语义查询，使其与实例查询的维度相匹配。这两组查询通过注意机制相互作用，并与图像特征进行交叉注意，以预测实例和语义分割掩码。实例类别通过计算实例查询和语义查询之间的点积并应用softmax归一化来预测。因此，DocSAM 可以在异构数据集上联合训练，增强稳健性和泛化能力，同时减少计算和存储资源。全面的评估表明，DocSAM 在准确性、效率和适应性方面超过了现有方法，突显了其在各种应用中推动文档图像理解和分割的潜力。代码可在以下网址获取：这个 https URL。 

---
# Enforcement Agents: Enhancing Accountability and Resilience in Multi-Agent AI Frameworks 

**Title (ZH)**: 强化代理：增强多代理AI框架中的问责制和韧性 

**Authors**: Sagar Tamang, Dibya Jyoti Bora  

**Link**: [PDF](https://arxiv.org/pdf/2504.04070)  

**Abstract**: As autonomous agents become more powerful and widely used, it is becoming increasingly important to ensure they behave safely and stay aligned with system goals, especially in multi-agent settings. Current systems often rely on agents self-monitoring or correcting issues after the fact, but they lack mechanisms for real-time oversight. This paper introduces the Enforcement Agent (EA) Framework, which embeds dedicated supervisory agents into the environment to monitor others, detect misbehavior, and intervene through real-time correction. We implement this framework in a custom drone simulation and evaluate it across 90 episodes using 0, 1, and 2 EA configurations. Results show that adding EAs significantly improves system safety: success rates rise from 0.0% with no EA to 7.4% with one EA and 26.7% with two EAs. The system also demonstrates increased operational longevity and higher rates of malicious drone reformation. These findings highlight the potential of lightweight, real-time supervision for enhancing alignment and resilience in multi-agent systems. 

**Abstract (ZH)**: 自主代理日益强大且广泛应用，确保其安全行为并保持与系统目标一致变得愈发重要，尤其是在多代理环境中。当前系统通常依赖代理自我监控或事后改正问题，但缺乏实时监督机制。本文介绍了执行代理(EA)框架，该框架在环境中嵌入专职监督代理，以监控其他代理、检测不端行为并通过实时纠正进行干预。我们在自定义无人机模拟中实现此框架，并使用0、1和2种EA配置在90个场景中进行评估。结果表明，添加EA显著提高了系统安全性：没有EA时的成功率为0.0%，一个EA时为7.4%，两个EA时为26.7%。该系统还展示了操作寿命的增加以及恶意无人机改过的更高比例。这些发现突显了轻量级、实时监督在增强多代理系统中对齐和韧性方面的潜在价值。 

---
# Mapping at First Sense: A Lightweight Neural Network-Based Indoor Structures Prediction Method for Robot Autonomous Exploration 

**Title (ZH)**: 基于初次感知的 Lightweight 神经网络室内结构预测方法及其在机器人自主探索中的应用 

**Authors**: Haojia Gao, Haohua Que, Kunrong Li, Weihao Shan, Mingkai Liu, Rong Zhao, Lei Mu, Xinghua Yang, Qi Wei, Fei Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.04061)  

**Abstract**: Autonomous exploration in unknown environments is a critical challenge in robotics, particularly for applications such as indoor navigation, search and rescue, and service robotics. Traditional exploration strategies, such as frontier-based methods, often struggle to efficiently utilize prior knowledge of structural regularities in indoor spaces. To address this limitation, we propose Mapping at First Sense, a lightweight neural network-based approach that predicts unobserved areas in local maps, thereby enhancing exploration efficiency. The core of our method, SenseMapNet, integrates convolutional and transformerbased architectures to infer occluded regions while maintaining computational efficiency for real-time deployment on resourceconstrained robots. Additionally, we introduce SenseMapDataset, a curated dataset constructed from KTH and HouseExpo environments, which facilitates training and evaluation of neural models for indoor exploration. Experimental results demonstrate that SenseMapNet achieves an SSIM (structural similarity) of 0.78, LPIPS (perceptual quality) of 0.68, and an FID (feature distribution alignment) of 239.79, outperforming conventional methods in map reconstruction quality. Compared to traditional frontier-based exploration, our method reduces exploration time by 46.5% (from 2335.56s to 1248.68s) while maintaining a high coverage rate (88%) and achieving a reconstruction accuracy of 88%. The proposed method represents a promising step toward efficient, learning-driven robotic exploration in structured environments. 

**Abstract (ZH)**: 基于初次感知的未知环境自主探索 

---
# VocalNet: Speech LLM with Multi-Token Prediction for Faster and High-Quality Generation 

**Title (ZH)**: VocalNet：具有多令牌预测的语音LLM，以实现更快更高质量的生成 

**Authors**: Yuhao Wang, Heyang Liu, Ziyang Cheng, Ronghua Wu, Qunshan Gu, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04060)  

**Abstract**: Speech large language models (LLMs) have emerged as a prominent research focus in speech processing. We propose VocalNet-1B and VocalNet-8B, a series of high-performance, low-latency speech LLMs enabled by a scalable and model-agnostic training framework for real-time voice interaction. Departing from the conventional next-token prediction (NTP), we introduce multi-token prediction (MTP), a novel approach optimized for speech LLMs that simultaneously improves generation speed and quality. Experiments show that VocalNet outperforms mainstream Omni LLMs despite using significantly less training data, while also surpassing existing open-source speech LLMs by a substantial margin. To support reproducibility and community advancement, we will open-source all model weights, inference code, training data, and framework implementations upon publication. 

**Abstract (ZH)**: 基于语音的大语言模型（LLMs）已成为语音处理领域的研究重点。我们提出VocalNet-1B和VocalNet-8B，这是一种通过可扩展且模型无关的训练框架实现的高性能、低延迟语音LLMs系列，用于实时语音交互。不同于传统的下一个token预测（NTP），我们引入了多token预测（MTP），这是一种针对语音LLMs优化的新方法，可以同时提高生成速度和质量。实验表明，VocalNet在使用显著较少训练数据的情况下，性能优于主流的Omni LLMs，并且在开放源代码的语音LLMs中表现优异。为了支持可重复性和社区发展，我们将公开所有模型权重、推理代码、训练数据和框架实现。 

---
# PIORF: Physics-Informed Ollivier-Ricci Flow for Long-Range Interactions in Mesh Graph Neural Networks 

**Title (ZH)**: PIORF: 物理导向的橄榄里奇流机制用于网格图神经网络中的长程交互 

**Authors**: Youn-Yeol Yu, Jeongwhan Choi, Jaehyeon Park, Kookjin Lee, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.04052)  

**Abstract**: Recently, data-driven simulators based on graph neural networks have gained attention in modeling physical systems on unstructured meshes. However, they struggle with long-range dependencies in fluid flows, particularly in refined mesh regions. This challenge, known as the 'over-squashing' problem, hinders information propagation. While existing graph rewiring methods address this issue to some extent, they only consider graph topology, overlooking the underlying physical phenomena. We propose Physics-Informed Ollivier-Ricci Flow (PIORF), a novel rewiring method that combines physical correlations with graph topology. PIORF uses Ollivier-Ricci curvature (ORC) to identify bottleneck regions and connects these areas with nodes in high-velocity gradient nodes, enabling long-range interactions and mitigating over-squashing. Our approach is computationally efficient in rewiring edges and can scale to larger simulations. Experimental results on 3 fluid dynamics benchmark datasets show that PIORF consistently outperforms baseline models and existing rewiring methods, achieving up to 26.2 improvement. 

**Abstract (ZH)**: 基于物理信息的Ollivier-Ricci流（PIORF）：解决非结构网格上流体流动中的长程依赖性问题 

---
# Can You Count to Nine? A Human Evaluation Benchmark for Counting Limits in Modern Text-to-Video Models 

**Title (ZH)**: 你能数到九吗？现代文本到视频模型计数限制的人工评估基准 

**Authors**: Xuyang Guo, Zekai Huang, Jiayan Huo, Yingyu Liang, Zhenmei Shi, Zhao Song, Jiahao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04051)  

**Abstract**: Generative models have driven significant progress in a variety of AI tasks, including text-to-video generation, where models like Video LDM and Stable Video Diffusion can produce realistic, movie-level videos from textual instructions. Despite these advances, current text-to-video models still face fundamental challenges in reliably following human commands, particularly in adhering to simple numerical constraints. In this work, we present T2VCountBench, a specialized benchmark aiming at evaluating the counting capability of SOTA text-to-video models as of 2025. Our benchmark employs rigorous human evaluations to measure the number of generated objects and covers a diverse range of generators, covering both open-source and commercial models. Extensive experiments reveal that all existing models struggle with basic numerical tasks, almost always failing to generate videos with an object count of 9 or fewer. Furthermore, our comprehensive ablation studies explore how factors like video style, temporal dynamics, and multilingual inputs may influence counting performance. We also explore prompt refinement techniques and demonstrate that decomposing the task into smaller subtasks does not easily alleviate these limitations. Our findings highlight important challenges in current text-to-video generation and provide insights for future research aimed at improving adherence to basic numerical constraints. 

**Abstract (ZH)**: 生成模型在多种AI任务中推动了显著进展，包括从文本生成视频，例如Video LDM和Stable Video Diffusion等模型可以从文本指令生成真实感的电影级视频。尽管取得了这些进展，当前的文本到视频模型在可靠执行人类指令方面仍然面临基本挑战，特别是在遵守简单的数值约束方面。本文提出了T2VCountBench，一个专门的基准测试，旨在评估截至2025年的SOTA文本到视频模型的计数能力。该基准测试通过严格的主观评估来测量生成对象的数量，并涵盖了广泛的生成器，包括开源和商业模型。广泛的实验表明，所有现有模型在基本数值任务上都存在困难，几乎总是无法生成对象数量为9或更少的视频。此外，我们全面的消融研究探索了视频风格、时间动态和多语言输入等因素如何影响计数性能。我们还探讨了提示精炼技术，并展示了将任务分解为更小的子任务并不能容易缓解这些限制。我们的研究结果突出了当前文本到视频生成中重要的挑战，并为未来旨在改善对基本数值约束遵循性的研究提供了见解。 

---
# A Survey of Pathology Foundation Model: Progress and Future Directions 

**Title (ZH)**: 病理学基础模型综述：进展与未来方向 

**Authors**: Conghao Xiong, Hao Chen, Joseph J. Y. Sung  

**Link**: [PDF](https://arxiv.org/pdf/2504.04045)  

**Abstract**: Computational pathology, analyzing whole slide images for automated cancer diagnosis, relies on the multiple instance learning framework where performance heavily depends on the feature extractor and aggregator. Recent Pathology Foundation Models (PFMs), pretrained on large-scale histopathology data, have significantly enhanced capabilities of extractors and aggregators but lack systematic analysis frameworks. This survey presents a hierarchical taxonomy organizing PFMs through a top-down philosophy that can be utilized to analyze FMs in any domain: model scope, model pretraining, and model design. Additionally, we systematically categorize PFM evaluation tasks into slide-level, patch-level, multimodal, and biological tasks, providing comprehensive benchmarking criteria. Our analysis identifies critical challenges in both PFM development (pathology-specific methodology, end-to-end pretraining, data-model scalability) and utilization (effective adaptation, model maintenance), paving the way for future directions in this promising field. Resources referenced in this survey are available at this https URL. 

**Abstract (ZH)**: 计算病理学，基于整个切片图像的自动化癌症诊断，依赖于多实例学习框架，其性能高度依赖于特征提取器和聚合器。大规模组织病理学数据预训练的近期病理学基础模型（PFMs）显著增强了提取器和聚合器的能力，但缺乏系统的分析框架。本文综述提出了一种自上而下的层级分类体系，用于组织PFMs，并可应用于任何领域：模型范围、模型预训练和模型设计。此外，我们系统地将PFM评估任务分类为切片级、 patch级、多模态和生物任务，并提供了全面的基准评估标准。我们的分析指出了PFM开发和利用中关键的挑战（特定于病理的方法学、端到端预训练、数据-模型可扩展性）和利用中的挑战（有效适应、模型维护），为这一有前景领域的未来方向指明了道路。文中引用的资源可访问此链接：此 https URL。 

---
# Memory-Statistics Tradeoff in Continual Learning with Structural Regularization 

**Title (ZH)**: 结构正则化约束下持续学习的内存-统计权衡 

**Authors**: Haoran Li, Jingfeng Wu, Vladimir Braverman  

**Link**: [PDF](https://arxiv.org/pdf/2504.04039)  

**Abstract**: We study the statistical performance of a continual learning problem with two linear regression tasks in a well-specified random design setting. We consider a structural regularization algorithm that incorporates a generalized $\ell_2$-regularization tailored to the Hessian of the previous task for mitigating catastrophic forgetting. We establish upper and lower bounds on the joint excess risk for this algorithm. Our analysis reveals a fundamental trade-off between memory complexity and statistical efficiency, where memory complexity is measured by the number of vectors needed to define the structural regularization. Specifically, increasing the number of vectors in structural regularization leads to a worse memory complexity but an improved excess risk, and vice versa. Furthermore, our theory suggests that naive continual learning without regularization suffers from catastrophic forgetting, while structural regularization mitigates this issue. Notably, structural regularization achieves comparable performance to joint training with access to both tasks simultaneously. These results highlight the critical role of curvature-aware regularization for continual learning. 

**Abstract (ZH)**: 我们研究了在良好指定的随机设计设置下，具有两个线性回归任务的持续学习问题的统计性能。我们考虑了一种结构化正则化算法，该算法结合了一种针对前一个任务哈essian的广义$\ell_2$-正则化，以减轻灾难性遗忘。我们建立了该算法的联合超额风险的上界和下界。我们的分析揭示了内存复杂性和统计效率之间的基本权衡，其中内存复杂性通过定义结构化正则化的向量数量来衡量。具体来说，增加结构化正则化的向量数量会导致更差的内存复杂性但更好的超额风险，反之亦然。此外，我们的理论表明，没有正则化的简单持续学习会遭受灾难性遗忘，而结构化正则化可以缓解这一问题。值得注意的是，结构化正则化在能够访问两个任务的情况下实现了与联合训练相当的性能。这些结果突显了持续学习中曲率感知正则化的作用至关重要。 

---
# Contrastive and Variational Approaches in Self-Supervised Learning for Complex Data Mining 

**Title (ZH)**: 对比与变分方法在复杂数据自监督学习中的应用 

**Authors**: Yingbin Liang, Lu Dai, Shuo Shi, Minghao Dai, Junliang Du, Haige Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04032)  

**Abstract**: Complex data mining has wide application value in many fields, especially in the feature extraction and classification tasks of unlabeled data. This paper proposes an algorithm based on self-supervised learning and verifies its effectiveness through experiments. The study found that in terms of the selection of optimizer and learning rate, the combination of AdamW optimizer and 0.002 learning rate performed best in all evaluation indicators, indicating that the adaptive optimization method can improve the performance of the model in complex data mining tasks. In addition, the ablation experiment further analyzed the contribution of each module. The results show that contrastive learning, variational modules, and data augmentation strategies play a key role in the generalization ability and robustness of the model. Through the convergence curve analysis of the loss function, the experiment verifies that the method can converge stably during the training process and effectively avoid serious overfitting. Further experimental results show that the model has strong adaptability on different data sets, can effectively extract high-quality features from unlabeled data, and improves classification accuracy. At the same time, under different data distribution conditions, the method can still maintain high detection accuracy, proving its applicability in complex data environments. This study analyzed the role of self-supervised learning methods in complex data mining through systematic experiments and verified its advantages in improving feature extraction quality, optimizing classification performance, and enhancing model stability 

**Abstract (ZH)**: 基于自监督学习的复杂数据挖掘算法及其有效性验证 

---
# Simultaneous Motion And Noise Estimation with Event Cameras 

**Title (ZH)**: 事件相机中同时运动与噪声估计 

**Authors**: Shintaro Shiba, Yoshimitsu Aoki, Guillermo Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2504.04029)  

**Abstract**: Event cameras are emerging vision sensors, whose noise is challenging to characterize. Existing denoising methods for event cameras consider other tasks such as motion estimation separately (i.e., sequentially after denoising). However, motion is an intrinsic part of event data, since scene edges cannot be sensed without motion. This work proposes, to the best of our knowledge, the first method that simultaneously estimates motion in its various forms (e.g., ego-motion, optical flow) and noise. The method is flexible, as it allows replacing the 1-step motion estimation of the widely-used Contrast Maximization framework with any other motion estimator, such as deep neural networks. The experiments show that the proposed method achieves state-of-the-art results on the E-MLB denoising benchmark and competitive results on the DND21 benchmark, while showing its efficacy on motion estimation and intensity reconstruction tasks. We believe that the proposed approach contributes to strengthening the theory of event-data denoising, as well as impacting practical denoising use-cases, as we release the code upon acceptance. Project page: this https URL 

**Abstract (ZH)**: 事件相机是新兴的视觉传感器，其噪声难以表征。现有事件相机去噪方法在去噪后才考虑其他任务（如运动估计）。然而，运动是事件数据的一个内在部分，因为不能在没有运动的情况下感知场景边缘。本文提出了一种，据我们所知，第一个能够在其各种形式（如自我运动、光学流）中同时估计运动和噪声的方法。该方法具有灵活性，允许用任何其他运动估计器（如深度神经网络）代替广泛应用的对比最大化框架中的单步运动估计。实验结果显示，所提出的方法在E-MLB去噪基准上达到最先进的效果，在DND21基准上表现竞争力，并在运动估计和强度重建任务中展示了其有效性。我们认为，所提出的方法不仅加强了事件数据去噪的理论，而且还影响了实际的去噪应用，因为接受后我们将发布代码。项目页面：这个 https URL 

---
# Rethinking Reflection in Pre-Training 

**Title (ZH)**: 重思预训练中的反射机制 

**Authors**: Essential AI, Darsh J Shah, Peter Rushton, Somanshu Singla, Mohit Parmar, Kurt Smith, Yash Vanjani, Ashish Vaswani, Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, Anil Thomas, Anthony Polloreno, Ashish Tanwer, Burhan Drak Sibai, Divya S Mansingka, Divya Shivaprasad, Ishaan Shah, Karl Stratos, Khoi Nguyen, Michael Callahan, Michael Pust, Mrinal Iyer, Philip Monk, Platon Mazarakis, Ritvik Kapila, Saurabh Srivastava, Tim Romanski  

**Link**: [PDF](https://arxiv.org/pdf/2504.04022)  

**Abstract**: A language model's ability to reflect on its own reasoning provides a key advantage for solving complex problems. While most recent research has focused on how this ability develops during reinforcement learning, we show that it actually begins to emerge much earlier - during the model's pre-training. To study this, we introduce deliberate errors into chains-of-thought and test whether the model can still arrive at the correct answer by recognizing and correcting these mistakes. By tracking performance across different stages of pre-training, we observe that this self-correcting ability appears early and improves steadily over time. For instance, an OLMo2-7B model pre-trained on 4 trillion tokens displays self-correction on our six self-reflection tasks. 

**Abstract (ZH)**: 一种语言模型自我反思其推理能力为其解决复杂问题提供了关键优势。虽然最近的研究主要集中在这种能力在强化学习中的发展过程，但我们表明这种能力实际上在模型的预训练阶段就已经开始出现。为了研究这一点，我们引入了人工错误到推理链中，并测试模型是否能够通过识别和纠正这些错误而仍然得出正确的答案。通过跟踪不同预训练阶段的表现，我们观察到这种自我纠正的能力早在预训练初期就出现了，并且随着时间的推移逐渐提高。例如，一个在4万亿个词元上进行预训练的OLMo2-7B模型在我们的六个自我反思任务中展示了自我纠正能力。 

---
# Foundation Models for Time Series: A Survey 

**Title (ZH)**: 时间序列的基石模型：综述 

**Authors**: Siva Rama Krishna Kottapalli, Karthik Hubli, Sandeep Chandrashekhara, Garima Jain, Sunayana Hubli, Gayathri Botla, Ramesh Doddaiah  

**Link**: [PDF](https://arxiv.org/pdf/2504.04011)  

**Abstract**: Transformer-based foundation models have emerged as a dominant paradigm in time series analysis, offering unprecedented capabilities in tasks such as forecasting, anomaly detection, classification, trend analysis and many more time series analytical tasks. This survey provides a comprehensive overview of the current state of the art pre-trained foundation models, introducing a novel taxonomy to categorize them across several dimensions. Specifically, we classify models by their architecture design, distinguishing between those leveraging patch-based representations and those operating directly on raw sequences. The taxonomy further includes whether the models provide probabilistic or deterministic predictions, and whether they are designed to work with univariate time series or can handle multivariate time series out of the box. Additionally, the taxonomy encompasses model scale and complexity, highlighting differences between lightweight architectures and large-scale foundation models. A unique aspect of this survey is its categorization by the type of objective function employed during training phase. By synthesizing these perspectives, this survey serves as a resource for researchers and practitioners, providing insights into current trends and identifying promising directions for future research in transformer-based time series modeling. 

**Abstract (ZH)**: 基于Transformer的预训练基础模型已经成为了时间序列分析中的主导 paradigm，提供了前所未有的能力，适用于诸如预测、异常检测、分类、趋势分析等众多时间序列分析任务。本文综述提供了当前预训练基础模型的全面概述，并提出了一种新颖的分类法，根据多个维度对这些模型进行分类。具体来说，我们将模型按架构设计分类，区分基于块表示的模型和直接处理原始序列的模型。分类法还包括模型是否提供概率性或确定性预测，以及模型是否专为单变量时间序列设计或能够直接处理多变量时间序列。此外，该分类法还包括模型规模和复杂度，突显轻量级架构与大规模基础模型之间的差异。本文综述的一个独特之处在于，根据训练阶段所使用的客观函数类型对其进行分类。通过综合这些视角，本文综述成为研究者和实践者的资源，提供了当前趋势的见解，并指出了未来基于Transformer的时间序列建模研究的有前景方向。 

---
# Edge Approximation Text Detector 

**Title (ZH)**: 边缘近似文本检测器 

**Authors**: Chuang Yang, Xu Han, Tao Han, Han Han, Bingxuan Zhao, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.04001)  

**Abstract**: Pursuing efficient text shape representations helps scene text detection models focus on compact foreground regions and optimize the contour reconstruction steps to simplify the whole detection pipeline. Current approaches either represent irregular shapes via box-to-polygon strategy or decomposing a contour into pieces for fitting gradually, the deficiency of coarse contours or complex pipelines always exists in these models. Considering the above issues, we introduce EdgeText to fit text contours compactly while alleviating excessive contour rebuilding processes. Concretely, it is observed that the two long edges of texts can be regarded as smooth curves. It allows us to build contours via continuous and smooth edges that cover text regions tightly instead of fitting piecewise, which helps avoid the two limitations in current models. Inspired by this observation, EdgeText formulates the text representation as the edge approximation problem via parameterized curve fitting functions. In the inference stage, our model starts with locating text centers, and then creating curve functions for approximating text edges relying on the points. Meanwhile, truncation points are determined based on the location features. In the end, extracting curve segments from curve functions by using the pixel coordinate information brought by truncation points to reconstruct text contours. Furthermore, considering the deep dependency of EdgeText on text edges, a bilateral enhanced perception (BEP) module is designed. It encourages our model to pay attention to the recognition of edge features. Additionally, to accelerate the learning of the curve function parameters, we introduce a proportional integral loss (PI-loss) to force the proposed model to focus on the curve distribution and avoid being disturbed by text scales. 

**Abstract (ZH)**: 追求高效的文本形状表示有助于场景文本检测模型集中在紧凑的前景区域并优化轮廓重构步骤，简化整个检测管道。当前的方法要么通过盒状到多边形的策略表示不规则形状，要么逐步分解轮廓为片段进行拟合，这些模型中粗略的轮廓或复杂的管道始终存在缺陷。考虑到上述问题，我们介绍了EdgeText以紧致地拟合文本轮廓，同时减轻过多的轮廓重建过程。具体而言，观察到文本的两条长边可以被视为平滑曲线。这使我们能够通过连续和平滑的边缘构建紧密覆盖文本区域的轮廓，而不是分段拟合，从而避免当前模型中的两个局限性。受此观察的启发，EdgeText将文本表示形式化为参数化曲线拟合函数的边缘近似问题。在推理阶段，我们的模型首先定位文本中心，然后基于点创建曲线函数以逼近文本边缘，并根据位置特征确定截断点。最后，通过截断点带来的像素坐标信息从曲线函数中提取曲线段，重建文本轮廓。此外，考虑到EdgeText对文本边缘的深度依赖性，我们设计了一个双边增强感知（BEP）模块，以促使模型关注边缘特征的识别。另外，为了加速曲线函数参数的学习，我们引入了比例积分损失（PI-loss），以迫使提出模型关注曲线分布，避免被文本尺度干扰。 

---
# Improving Offline Mixed-Criticality Scheduling with Reinforcement Learning 

**Title (ZH)**: 基于强化学习的离线混合criticality调度改进 

**Authors**: Muhammad El-Mahdy, Nourhan Sakr, Rodrigo Carrasco  

**Link**: [PDF](https://arxiv.org/pdf/2504.03994)  

**Abstract**: This paper introduces a novel reinforcement learning (RL) approach to scheduling mixed-criticality (MC) systems on processors with varying speeds. Building upon the foundation laid by [1], we extend their work to address the non-preemptive scheduling problem, which is known to be NP-hard. By modeling this scheduling challenge as a Markov Decision Process (MDP), we develop an RL agent capable of generating near-optimal schedules for real-time MC systems. Our RL-based scheduler prioritizes high-critical tasks while maintaining overall system performance.
Through extensive experiments, we demonstrate the scalability and effectiveness of our approach. The RL scheduler significantly improves task completion rates, achieving around 80% overall and 85% for high-criticality tasks across 100,000 instances of synthetic data and real data under varying system conditions. Moreover, under stable conditions without degradation, the scheduler achieves 94% overall task completion and 93% for high-criticality tasks. These results highlight the potential of RL-based schedulers in real-time and safety-critical applications, offering substantial improvements in handling complex and dynamic scheduling scenarios. 

**Abstract (ZH)**: 一种新型强化学习调度方法：针对可变速度处理器上的混合关键性系统非抢占式调度问题 

---
# Algorithmic Prompt Generation for Diverse Human-like Teaming and Communication with Large Language Models 

**Title (ZH)**: 算法激发用于多元类人团队协作与交流的大语言模型指令 

**Authors**: Siddharth Srikanth, Varun Bhatt, Boshen Zhang, Werner Hager, Charles Michael Lewis, Katia P. Sycara, Aaquib Tabrez, Stefanos Nikolaidis  

**Link**: [PDF](https://arxiv.org/pdf/2504.03991)  

**Abstract**: Understanding how humans collaborate and communicate in teams is essential for improving human-agent teaming and AI-assisted decision-making. However, relying solely on data from large-scale user studies is impractical due to logistical, ethical, and practical constraints, necessitating synthetic models of multiple diverse human behaviors. Recently, agents powered by Large Language Models (LLMs) have been shown to emulate human-like behavior in social settings. But, obtaining a large set of diverse behaviors requires manual effort in the form of designing prompts. On the other hand, Quality Diversity (QD) optimization has been shown to be capable of generating diverse Reinforcement Learning (RL) agent behavior. In this work, we combine QD optimization with LLM-powered agents to iteratively search for prompts that generate diverse team behavior in a long-horizon, multi-step collaborative environment. We first show, through a human-subjects experiment (n=54 participants), that humans exhibit diverse coordination and communication behavior in this domain. We then show that our approach can effectively replicate trends from human teaming data and also capture behaviors that are not easily observed without collecting large amounts of data. Our findings highlight the combination of QD and LLM-powered agents as an effective tool for studying teaming and communication strategies in multi-agent collaboration. 

**Abstract (ZH)**: 理解人类在团队中的协作和沟通方式对于提升人类-代理团队协作和AI辅助决策至关重要。然而，依赖大规模用户研究的数据由于物流、伦理和实践限制而不切实际，需要合成多种多样的人类行为模型。最近，由大型语言模型（LLMs）驱动的代理已经在社交环境中表现出类似人类的行为。但获得多样化的行为集需要手动设计提示的劳动。另一方面，质量多样性（QD）优化已被证明能够生成多样化的强化学习（RL）代理行为。在本工作中，我们将QD优化与LLM驱动的代理结合，迭代搜索生成多样化团队行为的提示，在长时段多步协作环境中。我们首先通过一个被试研究（n=54参与者）展示了在该领域人类的多样化协调和沟通行为。然后我们展示了该方法可以有效地复制人类团队数据的趋势，并且能够捕捉到没有大量数据收集难以观察到的行为。我们的发现突显了QD与LLM驱动代理结合作为一种有效工具，用于研究多代理协作中的团队合作与沟通策略。 

---
# CORTEX-AVD: CORner Case Testing & EXploration for Autonomous Vehicles Development 

**Title (ZH)**: CORTEX-AVD: 角 case 测试与探索在自动驾驶车辆开发中的应用 

**Authors**: Gabriel Shimanuki, Alexandre Nascimento, Lucio Vismari, Joao Camargo Jr, Jorge Almeida Jr, Paulo Cugnasca  

**Link**: [PDF](https://arxiv.org/pdf/2504.03989)  

**Abstract**: Autonomous Vehicles (AVs) aim to improve traffic safety and efficiency by reducing human error. However, ensuring AVs reliability and safety is a challenging task when rare, high-risk traffic scenarios are considered. These 'Corner Cases' (CC) scenarios, such as unexpected vehicle maneuvers or sudden pedestrian crossings, must be safely and reliable dealt by AVs during their operations. But they arehard to be efficiently generated. Traditional CC generation relies on costly and risky real-world data acquisition, limiting scalability, and slowing research and development progress. Simulation-based techniques also face challenges, as modeling diverse scenarios and capturing all possible CCs is complex and time-consuming. To address these limitations in CC generation, this research introduces CORTEX-AVD, CORner Case Testing & EXploration for Autonomous Vehicles Development, an open-source framework that integrates the CARLA Simulator and Scenic to automatically generate CC from textual descriptions, increasing the diversity and automation of scenario modeling. Genetic Algorithms (GA) are used to optimize the scenario parameters in six case study scenarios, increasing the occurrence of high-risk events. Unlike previous methods, CORTEX-AVD incorporates a multi-factor fitness function that considers variables such as distance, time, speed, and collision likelihood. Additionally, the study provides a benchmark for comparing GA-based CC generation methods, contributing to a more standardized evaluation of synthetic data generation and scenario assessment. Experimental results demonstrate that the CORTEX-AVD framework significantly increases CC incidence while reducing the proportion of wasted simulations. 

**Abstract (ZH)**: corners-case 测试与探索框架 CORTEX-AVD：自主车辆开发中的异常场景生成 

---
# V-CEM: Bridging Performance and Intervenability in Concept-based Models 

**Title (ZH)**: V-CEM：概念模型中性能与可干预性的桥梁 

**Authors**: Francesco De Santis, Gabriele Ciravegna, Philippe Bich, Danilo Giordano, Tania Cerquitelli  

**Link**: [PDF](https://arxiv.org/pdf/2504.03978)  

**Abstract**: Concept-based eXplainable AI (C-XAI) is a rapidly growing research field that enhances AI model interpretability by leveraging intermediate, human-understandable concepts. This approach not only enhances model transparency but also enables human intervention, allowing users to interact with these concepts to refine and improve the model's performance. Concept Bottleneck Models (CBMs) explicitly predict concepts before making final decisions, enabling interventions to correct misclassified concepts. While CBMs remain effective in Out-Of-Distribution (OOD) settings with intervention, they struggle to match the performance of black-box models. Concept Embedding Models (CEMs) address this by learning concept embeddings from both concept predictions and input data, enhancing In-Distribution (ID) accuracy but reducing the effectiveness of interventions, especially in OOD scenarios. In this work, we propose the Variational Concept Embedding Model (V-CEM), which leverages variational inference to improve intervention responsiveness in CEMs. We evaluated our model on various textual and visual datasets in terms of ID performance, intervention responsiveness in both ID and OOD settings, and Concept Representation Cohesiveness (CRC), a metric we propose to assess the quality of the concept embedding representations. The results demonstrate that V-CEM retains CEM-level ID performance while achieving intervention effectiveness similar to CBM in OOD settings, effectively reducing the gap between interpretability (intervention) and generalization (performance). 

**Abstract (ZH)**: 基于概念的可解释人工智能（C-XAI）是一种迅速发展中的研究领域，通过利用中间的人类可理解的概念来增强AI模型的可解释性。这种方法不仅增强了模型的透明度，还允许人类干预，使用户能够与这些概念进行交互，以细化并提高模型的性能。概念瓶颈模型（CBMs）在做出最终决策前明确预测概念，从而使干预能够纠正错分类的概念。尽管CBMs在具有干预措施的Out-Of-Distribution（OOD）设置中仍有效，但在OOD场景中它们难以与黑盒模型匹配-performance。概念嵌入模型（CEMs）通过从概念预测和输入数据中学习概念嵌入来解决这一问题，这提高了In-Distribution（ID）的准确性，但减少了干预的有效性，尤其是在OOD情况下。在这项工作中，我们提出了变分概念嵌入模型（V-CEM），利用变分推断来改善CEMs的干预响应性。我们在各种文本和视觉数据集上从ID性能、ID和OOD设置中的干预响应性以及我们提出的概念表示一致性（CRC）方面评估了该模型，该指标用于评估概念嵌入表示的质量。结果表明，V-CEM保持了CEM级别的ID性能，同时在OOD设置中实现了与CBM相似的干预效果，有效地缩小了可解释性（干预）与泛化（性能）之间的差距。 

---
# OLAF: An Open Life Science Analysis Framework for Conversational Bioinformatics Powered by Large Language Models 

**Title (ZH)**: OLAF：一种基于大型语言模型的对话生物信息学开放生命科学分析框架 

**Authors**: Dylan Riffle, Nima Shirooni, Cody He, Manush Murali, Sovit Nayak, Rishikumar Gopalan, Diego Gonzalez Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2504.03976)  

**Abstract**: OLAF (Open Life Science Analysis Framework) is an open-source platform that enables researchers to perform bioinformatics analyses using natural language. By combining large language models (LLMs) with a modular agent-pipe-router architecture, OLAF generates and executes bioinformatics code on real scientific data, including formats like .h5ad. The system includes an Angular front end and a Python/Firebase backend, allowing users to run analyses such as single-cell RNA-seq workflows, gene annotation, and data visualization through a simple web interface. Unlike general-purpose AI tools, OLAF integrates code execution, data handling, and scientific libraries in a reproducible, user-friendly environment. It is designed to lower the barrier to computational biology for non-programmers and support transparent, AI-powered life science research. 

**Abstract (ZH)**: OLAF（开放生命科学分析框架）是一个开源平台，使研究人员能够使用自然语言进行生物信息学分析。通过结合大型语言模型（LLMs）与模块化代理-管道-路由器架构，OLAF生成并执行针对真实科学数据（包括.h5ad格式）的生物信息学代码。该系统包括一个Angular前端和一个Python/Firebase后端，允许用户通过简单的网页界面运行如单细胞RNA-seq工作流、基因注释和数据可视化等分析。与通用人工智能工具不同，OLAF在一个可重复的、用户友好的环境中集成了代码执行、数据处理和科学库，旨在降低非程序员进入计算生物学的门槛，并支持透明的、基于AI的生命科学研究。 

---
# GREATERPROMPT: A Unified, Customizable, and High-Performing Open-Source Toolkit for Prompt Optimization 

**Title (ZH)**: GREATERPROMPT: 一个统一、可定制且高性能的开源提示优化工具包 

**Authors**: Wenliang Zheng, Sarkar Snigdha Sarathi Das, Yusen Zhang, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03975)  

**Abstract**: LLMs have gained immense popularity among researchers and the general public for its impressive capabilities on a variety of tasks. Notably, the efficacy of LLMs remains significantly dependent on the quality and structure of the input prompts, making prompt design a critical factor for their performance. Recent advancements in automated prompt optimization have introduced diverse techniques that automatically enhance prompts to better align model outputs with user expectations. However, these methods often suffer from the lack of standardization and compatibility across different techniques, limited flexibility in customization, inconsistent performance across model scales, and they often exclusively rely on expensive proprietary LLM APIs. To fill in this gap, we introduce GREATERPROMPT, a novel framework that democratizes prompt optimization by unifying diverse methods under a unified, customizable API while delivering highly effective prompts for different tasks. Our framework flexibly accommodates various model scales by leveraging both text feedback-based optimization for larger LLMs and internal gradient-based optimization for smaller models to achieve powerful and precise prompt improvements. Moreover, we provide a user-friendly Web UI that ensures accessibility for non-expert users, enabling broader adoption and enhanced performance across various user groups and application scenarios. GREATERPROMPT is available at this https URL via GitHub, PyPI, and web user interfaces. 

**Abstract (ZH)**: GREATERPROMPT：统一可定制的提示优化框架 

---
# VideoComp: Advancing Fine-Grained Compositional and Temporal Alignment in Video-Text Models 

**Title (ZH)**: VideoComp: 促进视频-文本模型中的细粒度组成和时间对齐advance 

**Authors**: Dahun Kim, AJ Piergiovanni, Ganesh Mallya, Anelia Angelova  

**Link**: [PDF](https://arxiv.org/pdf/2504.03970)  

**Abstract**: We introduce VideoComp, a benchmark and learning framework for advancing video-text compositionality understanding, aimed at improving vision-language models (VLMs) in fine-grained temporal alignment. Unlike existing benchmarks focused on static image-text compositionality or isolated single-event videos, our benchmark targets alignment in continuous multi-event videos. Leveraging video-text datasets with temporally localized event captions (e.g. ActivityNet-Captions, YouCook2), we construct two compositional benchmarks, ActivityNet-Comp and YouCook2-Comp. We create challenging negative samples with subtle temporal disruptions such as reordering, action word replacement, partial captioning, and combined disruptions. These benchmarks comprehensively test models' compositional sensitivity across extended, cohesive video-text sequences. To improve model performance, we propose a hierarchical pairwise preference loss that strengthens alignment with temporally accurate pairs and gradually penalizes increasingly disrupted ones, encouraging fine-grained compositional learning. To mitigate the limited availability of densely annotated video data, we introduce a pretraining strategy that concatenates short video-caption pairs to simulate multi-event sequences. We evaluate video-text foundational models and large multimodal models (LMMs) on our benchmark, identifying both strengths and areas for improvement in compositionality. Overall, our work provides a comprehensive framework for evaluating and enhancing model capabilities in achieving fine-grained, temporally coherent video-text alignment. 

**Abstract (ZH)**: VideoComp：用于提升细粒度时空对齐的视频-文本组成性理解基准及学习框架 

---
# Bridging LMS and Generative AI: Dynamic Course Content Integration (DCCI) for Connecting LLMs to Course Content -- The Ask ME Assistant 

**Title (ZH)**: LMS与生成式AI桥梁：动态课程内容整合（DCCI）以连接LLM与课程内容——Ask ME助手 

**Authors**: Kovan Mzwri, Márta Turcsányi-Szabo  

**Link**: [PDF](https://arxiv.org/pdf/2504.03966)  

**Abstract**: The integration of Large Language Models (LLMs) with Learning Management Systems (LMSs) has the potential to enhance task automation and accessibility in education. However, hallucination where LLMs generate inaccurate or misleading information remains a significant challenge. This study introduces the Dynamic Course Content Integration (DCCI) mechanism, which dynamically retrieves and integrates course content and curriculum from Canvas LMS into the LLM-powered assistant, Ask ME. By employing prompt engineering to structure retrieved content within the LLM's context window, DCCI ensures accuracy, relevance, and contextual alignment, mitigating hallucination. To evaluate DCCI's effectiveness, Ask ME's usability, and broader student perceptions of AI in education, a mixed-methods approach was employed, incorporating user satisfaction ratings and a structured survey. Results from a pilot study indicate high user satisfaction (4.614/5), with students recognizing Ask ME's ability to provide timely and contextually relevant responses for both administrative and course-related inquiries. Additionally, a majority of students agreed that Ask ME's integration with course content in Canvas LMS reduced platform-switching, improving usability, engagement, and comprehension. AI's role in reducing classroom hesitation and fostering self-directed learning and intellectual curiosity was also highlighted. Despite these benefits and positive perception of AI tools, concerns emerged regarding over-reliance on AI, accuracy limitations, and ethical issues such as plagiarism and reduced student-teacher interaction. These findings emphasize the need for strategic AI implementation, ethical safeguards, and a pedagogical framework that prioritizes human-AI collaboration over substitution. 

**Abstract (ZH)**: 大型语言模型与学习管理系统集成在教育中的潜在增强任务自动化和访问性：动态课程内容集成机制及其评估 

---
# Clinical ModernBERT: An efficient and long context encoder for biomedical text 

**Title (ZH)**: 临床现代BERT：一种高效的大上下文编码器用于生物医学文本 

**Authors**: Simon A. Lee, Anthony Wu, Jeffrey N. Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03964)  

**Abstract**: We introduce Clinical ModernBERT, a transformer based encoder pretrained on large scale biomedical literature, clinical notes, and medical ontologies, incorporating PubMed abstracts, MIMIC IV clinical data, and medical codes with their textual descriptions. Building on ModernBERT the current state of the art natural language text encoder featuring architectural upgrades such as rotary positional embeddings (RoPE), Flash Attention, and extended context length up to 8,192 tokens our model adapts these innovations specifically for biomedical and clinical domains. Clinical ModernBERT excels at producing semantically rich representations tailored for long context tasks. We validate this both by analyzing its pretrained weights and through empirical evaluation on a comprehensive suite of clinical NLP benchmarks. 

**Abstract (ZH)**: Clinical ModernBERT：基于大规模生物医学文献、临床笔记和医学本体的变压器编码器 

---
# DeepOHeat-v1: Efficient Operator Learning for Fast and Trustworthy Thermal Simulation and Optimization in 3D-IC Design 

**Title (ZH)**: DeepOHeat-v1：用于3D-IC设计中快速和可靠热仿真与优化的有效算子学习 

**Authors**: Xinling Yu, Ziyue Liu, Hai Li, Yixing Li, Xin Ai, Zhiyu Zeng, Ian Young, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03955)  

**Abstract**: Thermal analysis is crucial in three-dimensional integrated circuit (3D-IC) design due to increased power density and complex heat dissipation paths. Although operator learning frameworks such as DeepOHeat have demonstrated promising preliminary results in accelerating thermal simulation, they face critical limitations in prediction capability for multi-scale thermal patterns, training efficiency, and trustworthiness of results during design optimization. This paper presents DeepOHeat-v1, an enhanced physics-informed operator learning framework that addresses these challenges through three key innovations. First, we integrate Kolmogorov-Arnold Networks with learnable activation functions as trunk networks, enabling an adaptive representation of multi-scale thermal patterns. This approach achieves a $1.25\times$ and $6.29\times$ reduction in error in two representative test cases. Second, we introduce a separable training method that decomposes the basis function along the coordinate axes, achieving $62\times$ training speedup and $31\times$ GPU memory reduction in our baseline case, and enabling thermal analysis at resolutions previously infeasible due to GPU memory constraints. Third, we propose a confidence score to evaluate the trustworthiness of the predicted results, and further develop a hybrid optimization workflow that combines operator learning with finite difference (FD) using Generalized Minimal Residual (GMRES) method for incremental solution refinement, enabling efficient and trustworthy thermal optimization. Experimental results demonstrate that DeepOHeat-v1 achieves accuracy comparable to optimization using high-fidelity finite difference solvers, while speeding up the entire optimization process by $70.6\times$ in our test cases, effectively minimizing the peak temperature through optimal placement of heat-generating components. 

**Abstract (ZH)**: 三维集成电路（3D-IC）设计中的热分析对于增加的功率密度和复杂的热散布路径至关重要。虽然像DeepOHeat这样的操作学习框架已经展示了在加速热仿真方面的有希望的初步结果，但在预测多尺度热模式的能力、训练效率以及设计优化过程中结果的可信度方面仍面临关键限制。本文提出了一种增强的物理知情操作学习框架DeepOHeat-v1，通过三种关键创新解决了这些挑战。首先，我们结合了Kolmogorov-Arnold网络和可学习的激活函数作为主体网络，以实现多尺度热模式的自适应表示。这种方法在两个代表性测试案例中的误差分别减少了1.25倍和6.29倍。其次，我们引入了一种分离训练方法，沿坐标轴分解基函数，在基线案例中实现了62倍的训练加速和31倍的GPU内存减少，从而使因GPU内存限制之前无法实现的热分析成为可能。第三，我们提出了一种置信度评分来评估预测结果的可信度，并进一步开发了一种结合操作学习和有限差分法（FD）的混合优化工作流，使用广义最小残差（GMRES）方法进行增量解精细，从而实现高效且可信的热优化。实验结果表明，DeepOHeat-v1在准确度上与高保真有限差分求解器优化相当，在测试案例中将整个优化过程加速了70.6倍，通过最优放置热源组件有效降低了峰温。 

---
# TGraphX: Tensor-Aware Graph Neural Network for Multi-Dimensional Feature Learning 

**Title (ZH)**: TGraphX：张量感知图神经网络多维特征学习 

**Authors**: Arash Sajjadi, Mark Eramian  

**Link**: [PDF](https://arxiv.org/pdf/2504.03953)  

**Abstract**: TGraphX presents a novel paradigm in deep learning by unifying convolutional neural networks (CNNs) with graph neural networks (GNNs) to enhance visual reasoning tasks. Traditional CNNs excel at extracting rich spatial features from images but lack the inherent capability to model inter-object relationships. Conversely, conventional GNNs typically rely on flattened node features, thereby discarding vital spatial details. TGraphX overcomes these limitations by employing CNNs to generate multi-dimensional node features (e.g., (3*128*128) tensors) that preserve local spatial semantics. These spatially aware nodes participate in a graph where message passing is performed using 1*1 convolutions, which fuse adjacent features while maintaining their structure. Furthermore, a deep CNN aggregator with residual connections is used to robustly refine the fused messages, ensuring stable gradient flow and end-to-end trainability. Our approach not only bridges the gap between spatial feature extraction and relational reasoning but also demonstrates significant improvements in object detection refinement and ensemble reasoning. 

**Abstract (ZH)**: TGraphX通过将卷积神经网络（CNNs）与图神经网络（GNNs）统一起来，以增强视觉推理任务，在深度学习中提出了一种新颖的范式。 

---
# Understanding EFX Allocations: Counting and Variants 

**Title (ZH)**: 理解EFX 分配：计数与变体 

**Authors**: Tzeh Yuan Neoh, Nicholas Teh  

**Link**: [PDF](https://arxiv.org/pdf/2504.03951)  

**Abstract**: Envy-freeness up to any good (EFX) is a popular and important fairness property in the fair allocation of indivisible goods, of which its existence in general is still an open question. In this work, we investigate the problem of determining the minimum number of EFX allocations for a given instance, arguing that this approach may yield valuable insights into the existence and computation of EFX allocations. We focus on restricted instances where the number of goods slightly exceeds the number of agents, and extend our analysis to weighted EFX (WEFX) and a novel variant of EFX for general monotone valuations, termed EFX+. In doing so, we identify the transition threshold for the existence of allocations satisfying these fairness notions. Notably, we resolve open problems regarding WEFX by proving polynomial-time computability under binary additive valuations, and establishing the first constant-factor approximation for two agents. 

**Abstract (ZH)**: 任意好的忌妒 freeness（EFX）是 indivisible goods 公平分配中一个流行且重要的公平性属性，其普遍存在的问题仍然是一个开放问题。在本文中，我们研究确定给定实例的 EFX 分配数量最小值的问题，认为这种做法可能为 EFX 分配的存在性和计算提供有价值的见解。我们关注商品数量略微超过代理数量的限制实例，并将分析扩展到加权 EFX（WEFX）以及对一般单调估值的一种新型 EFX 变体 EFX+。在此过程中，我们识别出满足这些公平性概念的分配存在的转换阈值。特别地，我们通过证明在二进制加性估值下的多项式时间可计算性解决了关于 WEFX 的开放问题，并首次建立了两个代理的常数因子近似算法。 

---
# Analysis of Robustness of a Large Game Corpus 

**Title (ZH)**: 大型游戏语料库的健壮性分析 

**Authors**: Mahsa Bazzaz, Seth Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2504.03940)  

**Abstract**: Procedural content generation via machine learning (PCGML) in games involves using machine learning techniques to create game content such as maps and levels. 2D tile-based game levels have consistently served as a standard dataset for PCGML because they are a simplified version of game levels while maintaining the specific constraints typical of games, such as being solvable. In this work, we highlight the unique characteristics of game levels, including their structured discrete data nature, the local and global constraints inherent in the games, and the sensitivity of the game levels to small changes in input. We define the robustness of data as a measure of sensitivity to small changes in input that cause a change in output, and we use this measure to analyze and compare these levels to state-of-the-art machine learning datasets, showcasing the subtle differences in their nature. We also constructed a large dataset from four games inspired by popular classic tile-based games that showcase these characteristics and address the challenge of sparse data in PCGML by providing a significantly larger dataset than those currently available. 

**Abstract (ZH)**: 基于机器学习的游戏程序化内容生成（PCGML）涉及使用机器学习技术生成游戏内容，如地图和关卡。2D砖块制游戏关卡一直作为PCGML的标准数据集，因为它们是游戏关卡的简化版本，同时保留了典型的约束条件，如可解性。在本工作中，我们强调了游戏关卡的独特特性，包括其结构化离散数据性质、游戏中固有的局部和全局约束，以及游戏关卡对输入微小变化的高度敏感性。我们将数据的鲁棒性定义为输入微小变化引起输出变化的敏感性度量，并使用此度量来分析和比较这些关卡与最先进的机器学习数据集，展示了它们本质上的细微差异。我们还从四款受到流行经典砖块制游戏启发的游戏构建了一个大型数据集，以展示这些特性，并通过提供比当前可用的更大规模的数据集来应对PCGML中的稀疏数据挑战。 

---
# Adaptation of Large Language Models 

**Title (ZH)**: 大型语言模型的适应性 

**Authors**: Zixuan Ke, Yifei Ming, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2504.03931)  

**Abstract**: This tutorial on adaptation of LLMs is designed to address the growing demand for models that go beyond the static capabilities of generic LLMs by providing an overview of dynamic, domain-specific, and task-adaptive LLM adaptation techniques. While general LLMs have demonstrated strong generalization across a variety of tasks, they often struggle to perform well in specialized domains such as finance, healthcare, and code generation for underrepresented languages. Additionally, their static nature limits their ability to evolve with the changing world, and they are often extremely large in size, making them impractical and costly to deploy at scale. As a result, the adaptation of LLMs has drawn much attention since the birth of LLMs and is of core importance, both for industry, which focuses on serving its targeted users, and academia, which can greatly benefit from small but powerful LLMs. To address this gap, this tutorial aims to provide an overview of the LLM adaptation techniques. We start with an introduction to LLM adaptation, from both the data perspective and the model perspective. We then emphasize how the evaluation metrics and benchmarks are different from other techniques. After establishing the problems, we explore various adaptation techniques. We categorize adaptation techniques into two main families. The first is parametric knowledge adaptation, which focuses on updating the parametric knowledge within LLMs. Additionally, we will discuss real-time adaptation techniques, including model editing, which allows LLMs to be updated dynamically in production environments. The second kind of adaptation is semi-parametric knowledge adaptation, where the goal is to update LLM parameters to better leverage external knowledge or tools through techniques like retrieval-augmented generation (RAG) and agent-based systems. 

**Abstract (ZH)**: This Tutorial on LLM Adaptation: An Overview of Dynamic, Domain-Specific, and Task-Adaptive Techniques 

---
# RF-BayesPhysNet: A Bayesian rPPG Uncertainty Estimation Method for Complex Scenarios 

**Title (ZH)**: RF-BayesPhysNet：一种应用于复杂场景的心率不确定性估计方法 

**Authors**: Rufei Ma, Chao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03915)  

**Abstract**: Remote photoplethysmography (rPPG) technology infers heart rate by capturing subtle color changes in facial skin
using a camera, demonstrating great potential in non-contact heart rate measurement. However, measurement
accuracy significantly decreases in complex scenarios such as lighting changes and head movements compared
to ideal laboratory conditions. Existing deep learning models often neglect the quantification of measurement
uncertainty, limiting their credibility in dynamic scenes. To address the issue of insufficient rPPG measurement
reliability in complex scenarios, this paper introduces Bayesian neural networks to the rPPG field for the first time,
proposing the Robust Fusion Bayesian Physiological Network (RF-BayesPhysNet), which can model both aleatoric
and epistemic uncertainty. It leverages variational inference to balance accuracy and computational efficiency.
Due to the current lack of uncertainty estimation metrics in the rPPG field, this paper also proposes a new set of
methods, using Spearman correlation coefficient, prediction interval coverage, and confidence interval width, to
measure the effectiveness of uncertainty estimation methods under different noise conditions. Experiments show
that the model, with only double the parameters compared to traditional network models, achieves a MAE of 2.56
on the UBFC-RPPG dataset, surpassing most models. It demonstrates good uncertainty estimation capability
in no-noise and low-noise conditions, providing prediction confidence and significantly enhancing robustness in
real-world applications. We have open-sourced the code at this https URL 

**Abstract (ZH)**: 远程光电体积描记术（rPPG）技术通过摄像头捕捉面部皮肤的细微颜色变化来推断心率，展示了在非接触心率测量方面的巨大潜力。然而，在光照变化和头部移动等复杂场景中，测量准确性显著下降，不及理想实验室条件。现有深度学习模型往往忽视了测量不确定性的量化，限制了其在动态场景中的可信度。为了解决复杂场景下rPPG测量可靠性的不足问题，本文首次将贝叶斯神经网络引入rPPG领域，提出了稳健融合贝叶斯生理网络（RF-BayesPhysNet），该网络可以建模偶然性和先验不确定性。利用变分推断平衡准确性和计算效率。由于rPPG领域目前缺乏不确定性估算指标，本文还提出了一套新的方法，使用斯皮尔曼相关系数、预测区间覆盖率和置信区间宽度，来衡量在不同噪声条件下的不确定性估算方法的有效性。实验结果显示，与传统网络模型相比，该模型仅参数量增加一倍，MAE达到2.56，在UBFC-RPPG数据集上超越了大多数模型。它在无噪声和低噪声条件下展示了良好的不确定性估算能力，提供了预测置信度，并显著增强了在实际应用中的鲁棒性。代码已开源：this https URL 

---
# Leveraging Gait Patterns as Biomarkers: An attention-guided Deep Multiple Instance Learning Network for Scoliosis Classification 

**Title (ZH)**: 利用步态模式作为生物标志物：一种attention引导的深多重实例学习网络在脊柱侧弯分类中的应用 

**Authors**: Haiqing Li, Yuzhi Guo, Feng Jiang, Qifeng Zhou, Hehuan Ma, Junzhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03894)  

**Abstract**: Scoliosis is a spinal curvature disorder that is difficult to detect early and can compress the chest cavity, impacting respiratory function and cardiac health. Especially for adolescents, delayed detection and treatment result in worsening compression. Traditional scoliosis detection methods heavily rely on clinical expertise, and X-ray imaging poses radiation risks, limiting large-scale early screening. We propose an Attention-Guided Deep Multi-Instance Learning method (Gait-MIL) to effectively capture discriminative features from gait patterns, which is inspired by ScoNet-MT's pioneering use of gait patterns for scoliosis detection. We evaluate our method on the first large-scale dataset based on gait patterns for scoliosis classification. The results demonstrate that our study improves the performance of using gait as a biomarker for scoliosis detection, significantly enhances detection accuracy for the particularly challenging Neutral cases, where subtle indicators are often overlooked. Our Gait-MIL also performs robustly in imbalanced scenarios, making it a promising tool for large-scale scoliosis screening. 

**Abstract (ZH)**: 注意力引导的深度多实例学习方法（Gait-MIL）在脊柱侧弯检测中的应用 

---
# Investigating Affective Use and Emotional Well-being on ChatGPT 

**Title (ZH)**: 探究ChatGPT中情感应用与情感福祉 

**Authors**: Jason Phang, Michael Lampe, Lama Ahmad, Sandhini Agarwal, Cathy Mengying Fang, Auren R. Liu, Valdemar Danry, Eunhae Lee, Samantha W.T. Chan, Pat Pataranutaporn, Pattie Maes  

**Link**: [PDF](https://arxiv.org/pdf/2504.03888)  

**Abstract**: As AI chatbots see increased adoption and integration into everyday life, questions have been raised about the potential impact of human-like or anthropomorphic AI on users. In this work, we investigate the extent to which interactions with ChatGPT (with a focus on Advanced Voice Mode) may impact users' emotional well-being, behaviors and experiences through two parallel studies. To study the affective use of AI chatbots, we perform large-scale automated analysis of ChatGPT platform usage in a privacy-preserving manner, analyzing over 3 million conversations for affective cues and surveying over 4,000 users on their perceptions of ChatGPT. To investigate whether there is a relationship between model usage and emotional well-being, we conduct an Institutional Review Board (IRB)-approved randomized controlled trial (RCT) on close to 1,000 participants over 28 days, examining changes in their emotional well-being as they interact with ChatGPT under different experimental settings. In both on-platform data analysis and the RCT, we observe that very high usage correlates with increased self-reported indicators of dependence. From our RCT, we find that the impact of voice-based interactions on emotional well-being to be highly nuanced, and influenced by factors such as the user's initial emotional state and total usage duration. Overall, our analysis reveals that a small number of users are responsible for a disproportionate share of the most affective cues. 

**Abstract (ZH)**: 随着AI聊天机器人被更广泛地采用并融入日常生活中，人们对其可能对用户产生的人类化或拟人化AI的影响提出了质疑。在本研究中，我们通过两项并行研究调查了与ChatGPT（特别是高级语音模式）交互对其用户情绪健康、行为和体验的影响程度。为研究AI聊天机器人的情感使用情况，我们以隐私保护的方式对ChatGPT平台使用进行了大规模自动化分析，分析了超过300万次对话以寻找情感线索，并对超过4000名用户进行了ChatGPT感知的调查。为探讨模型使用与情绪健康之间的关系，我们经过机构审查委员会（IRB）批准，对近1000名参与者进行了为期28天的随机对照试验（RCT），并在不同的实验设置下检查了他们的情绪健康变化情况。在平台数据的分析和RCT两方面，我们观察到极高的使用频率与自我报告的情感依赖性增加之间存在关联。从我们的RCT研究中，我们发现基于语音的交互对情绪健康的影响具有高度复杂性，并受到用户初始情绪状态和总使用时间等因素的影响。总体而言，我们的分析表明，一小部分用户产生了不成比例的最多情感线索。 

---
# Accurate GPU Memory Prediction for Deep Learning Jobs through Dynamic Analysis 

**Title (ZH)**: 基于动态分析的深度学习任务准确GPU内存预测 

**Authors**: Jiabo Shi, Yehia Elkhatib  

**Link**: [PDF](https://arxiv.org/pdf/2504.03887)  

**Abstract**: The benefits of Deep Learning (DL) impose significant pressure on GPU resources, particularly within GPU cluster, where Out-Of-Memory (OOM) errors present a primary impediment to model training and efficient resource utilization. Conventional OOM estimation techniques, relying either on static graph analysis or direct GPU memory profiling, suffer from inherent limitations: static analysis often fails to capture model dynamics, whereas GPU-based profiling intensifies contention for scarce GPU resources. To overcome these constraints, VeritasEst emerges. It is an innovative, entirely CPU-based analysis tool capable of accurately predicting the peak GPU memory required for DL training tasks without accessing the target GPU. This "offline" prediction capability is core advantage of VeritasEst, allowing accurate memory footprint information to be obtained before task scheduling, thereby effectively preventing OOM and optimizing GPU allocation. Its performance was validated through thousands of experimental runs across convolutional neural network (CNN) models: Compared to baseline GPU memory estimators, VeritasEst significantly reduces the relative error by 84% and lowers the estimation failure probability by 73%. VeritasEst represents a key step towards efficient and predictable DL training in resource-constrained environments. 

**Abstract (ZH)**: 深度学习对GPU资源的重大需求给GPU集群带来了显著压力，尤其是在GPU集群中，内存溢出（OOM）错误是模型训练和资源高效利用的主要障碍。传统的OOM估计技术要么依赖静态图分析，要么直接进行GPU内存监控，都存在先天限制：静态分析往往无法捕捉模型动态，而基于GPU的监控则加剧了对稀缺GPU资源的争夺。为克服这些限制，VeritasEst应运而生。它是一种创新的、完全基于CPU的分析工具，能够在不访问目标GPU的情况下准确预测DL训练任务所需的峰值GPU内存。VeritasEst的这种“离线”预测能力是其核心优势，能够在任务调度前获取准确的内存占用信息，从而有效防止OOM并优化GPU分配。实验验证显示，与基线GPU内存估计算法相比，VeritasEst将相对误差降低了84%，估计失败概率降低了73%。VeritasEst代表了在资源受限环境中实现高效且可预测的DL训练的一大步。 

---
# Can ChatGPT Learn My Life From a Week of First-Person Video? 

**Title (ZH)**: Can ChatGPT从一周的第一人称视频中学习我的生活？ 

**Authors**: Keegan Harris  

**Link**: [PDF](https://arxiv.org/pdf/2504.03857)  

**Abstract**: Motivated by recent improvements in generative AI and wearable camera devices (e.g. smart glasses and AI-enabled pins), I investigate the ability of foundation models to learn about the wearer's personal life through first-person camera data. To test this, I wore a camera headset for 54 hours over the course of a week, generated summaries of various lengths (e.g. minute-long, hour-long, and day-long summaries), and fine-tuned both GPT-4o and GPT-4o-mini on the resulting summary hierarchy. By querying the fine-tuned models, we are able to learn what the models learned about me. The results are mixed: Both models learned basic information about me (e.g. approximate age, gender). Moreover, GPT-4o correctly deduced that I live in Pittsburgh, am a PhD student at CMU, am right-handed, and have a pet cat. However, both models also suffered from hallucination and would make up names for the individuals present in the video footage of my life. 

**Abstract (ZH)**: 受近期生成式人工智能和可穿戴相机设备（如智能眼镜和AI驱动的针）进展的启发，我研究了基础模型通过第一人称相机数据学习佩戴者个人生活的能力。为了测试这一点，我在一周内佩戴相机头戴设备共计54小时，生成了不同长度的摘要（例如，1分钟、1小时和1天的摘要），并分别对GPT-4o和GPT-4o-mini进行了微调。通过查询微调后的模型，我们可以了解模型对我了解了多少。结果是混合的：两个模型都学到了一些关于我的基本信息（例如，大约年龄、性别）。此外，GPT-4o正确推断出我住在匹兹堡，是CMU的博士生，是右撇子，并且养有一只宠物猫。然而，两个模型也出现了幻觉，会在视频素材中虚构出人们的名字。 

---
# Detection Limits and Statistical Separability of Tree Ring Watermarks in Rectified Flow-based Text-to-Image Generation Models 

**Title (ZH)**: 校准流基于文本生成图像模型中树轮水印的检测限和统计可分辨性 

**Authors**: Ved Umrajkar, Aakash Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2504.03850)  

**Abstract**: Tree-Ring Watermarking is a significant technique for authenticating AI-generated images. However, its effectiveness in rectified flow-based models remains unexplored, particularly given the inherent challenges of these models with noise latent inversion. Through extensive experimentation, we evaluated and compared the detection and separability of watermarks between SD 2.1 and FLUX.1-dev models. By analyzing various text guidance configurations and augmentation attacks, we demonstrate how inversion limitations affect both watermark recovery and the statistical separation between watermarked and unwatermarked images. Our findings provide valuable insights into the current limitations of Tree-Ring Watermarking in the current SOTA models and highlight the critical need for improved inversion methods to achieve reliable watermark detection and separability. The official implementation, dataset release and all experimental results are available at this \href{this https URL}{\textbf{link}}. 

**Abstract (ZH)**: 树轮水印技术是认证AI生成图像的重要方法。然而，该技术在纠正的基于流的模型中的有效性尚未被探索，尤其是考虑到这些模型中存在的固有噪声反向转换难题。通过广泛的实验，我们评估并比较了SD 2.1和FLUX.1-dev模型之间水印的检测能力和可分离性。通过对各种文本引导配置和增强攻击的分析，我们展示了反向转换限制如何影响水印恢复以及带水印和不带水印图像之间的统计分离。我们的研究结果为当前Tree-Ring水印技术在当前SOTA模型中的局限性提供了有价值的洞见，并强调了改进反向转换方法对于实现可靠的水印检测和可分离性至关重要。官方实现、数据集发布和所有实验结果可在以下链接获取：\href{this https URL}{\textbf{link}}。 

---
# Arti-"fickle" Intelligence: Using LLMs as a Tool for Inference in the Political and Social Sciences 

**Title (ZH)**: “易变”的智能：将大语言模型作为推理工具应用于政治和社会科学 

**Authors**: Lisa P. Argyle, Ethan C. Busby, Joshua R. Gubler, Bryce Hepner, Alex Lyman, David Wingate  

**Link**: [PDF](https://arxiv.org/pdf/2504.03822)  

**Abstract**: Generative large language models (LLMs) are incredibly useful, versatile, and promising tools. However, they will be of most use to political and social science researchers when they are used in a way that advances understanding about real human behaviors and concerns. To promote the scientific use of LLMs, we suggest that researchers in the political and social sciences need to remain focused on the scientific goal of inference. To this end, we discuss the challenges and opportunities related to scientific inference with LLMs, using validation of model output as an illustrative case for discussion. We propose a set of guidelines related to establishing the failure and success of LLMs when completing particular tasks, and discuss how we can make inferences from these observations. We conclude with a discussion of how this refocus will improve the accumulation of shared scientific knowledge about these tools and their uses in the social sciences. 

**Abstract (ZH)**: 生成型大规模语言模型（LLMs）是极其有用、灵活且前景广阔的工具。然而，只有当它们用于促进对真实人类行为和关切的理解时，才对政治和社会科学家最有用。为了促进LLMs的科学发展应用，我们建议政治和社会科学家研究人员需要专注于科学推理的目标。为此，我们讨论了使用LLMs进行科学推理的挑战与机遇，以模型输出的验证为例进行说明。我们提出了一套关于任务完成时验证LLMs的失败与成功标准的指南，并讨论了我们如何根据这些观察进行推理。最后，我们探讨了这一重新聚焦如何改善对这些工具及其在社会科学中应用的共享科学知识的积累。 

---
# Exploring Various Sequential Learning Methods for Deformation History Modeling 

**Title (ZH)**: 探索各种序列学习方法在形变历史建模中的应用 

**Authors**: Muhammed Adil Yatkin, Mihkel Korgesaar, Jani Romanoff, Umit Islak, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2504.03818)  

**Abstract**: Current neural network (NN) models can learn patterns from data points with historical dependence. Specifically, in natural language processing (NLP), sequential learning has transitioned from recurrence-based architectures to transformer-based architectures. However, it is unknown which NN architectures will perform the best on datasets containing deformation history due to mechanical loading. Thus, this study ascertains the appropriateness of 1D-convolutional, recurrent, and transformer-based architectures for predicting deformation localization based on the earlier states in the form of deformation history. Following this investigation, the crucial incompatibility issues between the mathematical computation of the prediction process in the best-performing NN architectures and the actual values derived from the natural physical properties of the deformation paths are examined in detail. 

**Abstract (ZH)**: 当前的神经网络模型可以从具有历史依赖性的数据点中学习模式。具体而言，在自然语言处理（NLP）中， Sequential学习已经从基于循环的架构转型为基于变换器的架构。然而，对于包含由于机械加载引起的变形历史的 dataset，尚不清楚哪种 NN 架构能够表现最佳以预测变形 localization。因此，本研究确定了一维卷积、循环和变换器基架构在基于变形历史的早期状态预测变形 localization 方面的适当性。随后，研究了在最佳表现的 NN 架构中预测过程的数学计算与实际源自变形路径的自然物理属性的值之间的关键不兼容问题。 

---
# Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data? 

**Title (ZH)**: LLMs 中的递归训练循环：训练数据属性如何调节生成数据中的分布偏移？ 

**Authors**: Grgur Kovač, Jérémy Perez, Rémy Portelas, Peter Ford Dominey, Pierre-Yves Oudeyer  

**Link**: [PDF](https://arxiv.org/pdf/2504.03814)  

**Abstract**: Large language models (LLMs) are increasingly contributing to the creation of content on the Internet. This creates a feedback loop as subsequent generations of models will be trained on this generated, synthetic data. This phenomenon is receiving increasing interest, in particular because previous studies have shown that it may lead to distribution shift - models misrepresent and forget the true underlying distributions of human data they are expected to approximate (e.g. resulting in a drastic loss of quality). In this study, we study the impact of human data properties on distribution shift dynamics in iterated training loops. We first confirm that the distribution shift dynamics greatly vary depending on the human data by comparing four datasets (two based on Twitter and two on Reddit). We then test whether data quality may influence the rate of this shift. We find that it does on the twitter, but not on the Reddit datasets. We then focus on a Reddit dataset and conduct a more exhaustive evaluation of a large set of dataset properties. This experiment associated lexical diversity with larger, and semantic diversity with smaller detrimental shifts, suggesting that incorporating text with high lexical (but limited semantic) diversity could exacerbate the degradation of generated text. We then focus on the evolution of political bias, and find that the type of shift observed (bias reduction, amplification or inversion) depends on the political lean of the human (true) distribution. Overall, our work extends the existing literature on the consequences of recursive fine-tuning by showing that this phenomenon is highly dependent on features of the human data on which training occurs. This suggests that different parts of internet (e.g. GitHub, Reddit) may undergo different types of shift depending on their properties. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地参与互联网内容的创作。这一过程会形成反馈循环，后续模型将基于这些生成的合成数据进行训练。这一现象引起了越来越多的关注，因为之前的研究表明，这可能会导致分布偏移——模型误表征并忘记了人类数据的真实分布（例如，导致质量急剧下降）。在本研究中，我们研究了人类数据属性对迭代训练循环中分布偏移动态的影响。我们首先通过比较四个数据集（两个基于Twitter，两个基于Reddit）来确认分布偏移动态随人类数据的不同而显著变化。我们接着检验数据质量是否会影响这一偏移的速度。我们发现，在Twitter数据集上存在影响，而在Reddit数据集上不存在影响。然后，我们专注于一个Reddit数据集，并对大量数据集属性进行了更全面的评估。该实验表明词汇多样性与更大的负面偏移相关，而语义多样性与更小的负面影响偏移相关，这表明包含高词汇（但有限语义）多样性的文本可能会加剧生成文本的质量下降。我们还研究了政治偏见的发展，发现观察到的偏移类型（偏见减少、放大或反转）取决于人类（真实）分布的政治倾向。总体而言，我们的研究扩展了关于递归微调后果的现有文献，表明这一现象高度依赖于训练过程中人类数据的特征。这表明互联网的不同部分（例如，GitHub，Reddit）可能根据其属性经历不同类型的变化。 

---
# Drawing a Map of Elections 

**Title (ZH)**: 绘制选举地图 

**Authors**: Stanisław Szufa, Niclas Boehmer, Robert Bredereck, Piotr Faliszewski, Rolf Niedermeier, Piotr Skowron, Arkadii Slinko, Nimrod Talmon  

**Link**: [PDF](https://arxiv.org/pdf/2504.03809)  

**Abstract**: Our main contribution is the introduction of the map of elections framework. A map of elections consists of three main elements: (1) a dataset of elections (i.e., collections of ordinal votes over given sets of candidates), (2) a way of measuring similarities between these elections, and (3) a representation of the elections in the 2D Euclidean space as points, so that the more similar two elections are, the closer are their points. In our maps, we mostly focus on datasets of synthetic elections, but we also show an example of a map over real-life ones. To measure similarities, we would have preferred to use, e.g., the isomorphic swap distance, but this is infeasible due to its high computational complexity. Hence, we propose polynomial-time computable positionwise distance and use it instead. Regarding the representations in 2D Euclidean space, we mostly use the Kamada-Kawai algorithm, but we also show two alternatives. We develop the necessary theoretical results to form our maps and argue experimentally that they are accurate and credible. Further, we show how coloring the elections in a map according to various criteria helps in analyzing results of a number of experiments. In particular, we show colorings according to the scores of winning candidates or committees, running times of ILP-based winner determination algorithms, and approximation ratios achieved by particular algorithms. 

**Abstract (ZH)**: 我们主要的贡献是介绍了选举地图框架。选举地图由三个主要元素组成：(1) 选举数据集（即候选人的集合上的序数投票集合），(2) 一种衡量这些选举之间相似性的方法，以及(3) 将选举在二维欧几里得空间中表示为点，使得两个选举越相似，它们的点越接近。在我们的地图中，我们主要关注合成选举的数据集，但也展示了真实选举的一个示例。为了衡量相似性，我们本来希望使用同构交换距离，但由于其高计算复杂性，这变得不可能。因此，我们提出了一种多项式时间可计算的位置距离，并使用它代替。关于二维欧几里得空间中的表示，我们主要使用了Kamada-Kawai算法，但也展示了两种替代方案。我们发展了必要的理论成果来构建我们的地图，并通过实验论证它们的准确性和可靠性。此外，我们展示了根据各种标准对地图中的选举进行着色有助于分析多次实验的结果。具体地，我们展示了根据获胜候选人的得分或委员会、基于ILP的获胜者确定算法的运行时间以及特定算法达到的近似比的着色方案。 

---
# Semantic-guided Representation Learning for Multi-Label Recognition 

**Title (ZH)**: 基于语义引导的多标签识别表示学习 

**Authors**: Ruhui Zhang, Hezhe Qiao, Pengcheng Xu, Mingsheng Shang, Lin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03801)  

**Abstract**: Multi-label Recognition (MLR) involves assigning multiple labels to each data instance in an image, offering advantages over single-label classification in complex scenarios. However, it faces the challenge of annotating all relevant categories, often leading to uncertain annotations, such as unseen or incomplete labels. Recent Vision and Language Pre-training (VLP) based methods have made significant progress in tackling zero-shot MLR tasks by leveraging rich vision-language correlations. However, the correlation between multi-label semantics has not been fully explored, and the learned visual features often lack essential semantic information. To overcome these limitations, we introduce a Semantic-guided Representation Learning approach (SigRL) that enables the model to learn effective visual and textual representations, thereby improving the downstream alignment of visual images and categories. Specifically, we first introduce a graph-based multi-label correlation module (GMC) to facilitate information exchange between labels, enriching the semantic representation across the multi-label texts. Next, we propose a Semantic Visual Feature Reconstruction module (SVFR) to enhance the semantic information in the visual representation by integrating the learned textual representation during reconstruction. Finally, we optimize the image-text matching capability of the VLP model using both local and global features to achieve zero-shot MLR. Comprehensive experiments are conducted on several MLR benchmarks, encompassing both zero-shot MLR (with unseen labels) and single positive multi-label learning (with limited labels), demonstrating the superior performance of our approach compared to state-of-the-art methods. The code is available at this https URL. 

**Abstract (ZH)**: 多标签识别中的语义导向表示学习（SigRL）：面向零样本多标签识别的任务 

---
# Decision SpikeFormer: Spike-Driven Transformer for Decision Making 

**Title (ZH)**: 决策尖峰 Former: 尖峰驱动的变压器用于决策 Making 

**Authors**: Wei Huang, Qinying Gu, Nanyang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2504.03800)  

**Abstract**: Offline reinforcement learning (RL) enables policy training solely on pre-collected data, avoiding direct environment interaction - a crucial benefit for energy-constrained embodied AI applications. Although Artificial Neural Networks (ANN)-based methods perform well in offline RL, their high computational and energy demands motivate exploration of more efficient alternatives. Spiking Neural Networks (SNNs) show promise for such tasks, given their low power consumption. In this work, we introduce DSFormer, the first spike-driven transformer model designed to tackle offline RL via sequence modeling. Unlike existing SNN transformers focused on spatial dimensions for vision tasks, we develop Temporal Spiking Self-Attention (TSSA) and Positional Spiking Self-Attention (PSSA) in DSFormer to capture the temporal and positional dependencies essential for sequence modeling in RL. Additionally, we propose Progressive Threshold-dependent Batch Normalization (PTBN), which combines the benefits of LayerNorm and BatchNorm to preserve temporal dependencies while maintaining the spiking nature of SNNs. Comprehensive results in the D4RL benchmark show DSFormer's superiority over both SNN and ANN counterparts, achieving 78.4% energy savings, highlighting DSFormer's advantages not only in energy efficiency but also in competitive performance. Code and models are public at this https URL. 

**Abstract (ZH)**: 基于离线强化学习的DSFormer：一种_spike驱动的变压器模型以序列建模解决离线RL问题 

---
# Experimental Study on Time Series Analysis of Lower Limb Rehabilitation Exercise Data Driven by Novel Model Architecture and Large Models 

**Title (ZH)**: 基于新型模型架构和大规模模型驱动的下肢康复锻炼时间序列分析实验研究 

**Authors**: Hengyu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.03799)  

**Abstract**: This study investigates the application of novel model architectures and large-scale foundational models in temporal series analysis of lower limb rehabilitation motion data, aiming to leverage advancements in machine learning and artificial intelligence to empower active rehabilitation guidance strategies for post-stroke patients in limb motor function recovery. Utilizing the SIAT-LLMD dataset of lower limb movement data proposed by the Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, we systematically elucidate the implementation and analytical outcomes of the innovative xLSTM architecture and the foundational model Lag-Llama in short-term temporal prediction tasks involving joint kinematics and dynamics parameters. The research provides novel insights for AI-enabled medical rehabilitation applications, demonstrating the potential of cutting-edge model architectures and large-scale models in rehabilitation medicine temporal prediction. These findings establish theoretical foundations for future applications of personalized rehabilitation regimens, offering significant implications for the development of customized therapeutic interventions in clinical practice. 

**Abstract (ZH)**: 本研究探讨了新颖模型架构和大规模基础模型在下肢康复运动数据时序分析中的应用，旨在利用机器学习和人工智能的进步，为中风后肢体运动功能恢复患者的主动康复指导策略提供支持。利用中国科学院深圳先进技术研究院提出的SIAT-LLMD下肢运动数据集，系统阐释了创新xLSTM架构和基础模型Lag-Llama在涉及关节运动学和动力学参数的短期时序预测任务中的实现与分析结果。研究为AI赋能的医疗康复应用提供了新的见解，展示了前沿模型架构和大规模模型在康复医学时间序列预测中的潜力。这些发现为未来个性化的康复方案应用奠定了理论基础，为临床实践中定制化治疗干预的发展提供了重要意义。 

---
# An Intelligent and Privacy-Preserving Digital Twin Model for Aging-in-Place 

**Title (ZH)**: 一种用于原 place 老龄化的智能隐私保护数字孪生模型 

**Authors**: Yongjie Wang, Jonathan Cyril Leung, Ming Chen, Zhiwei Zeng, Benny Toh Hsiang Tan, Yang Qiu, Zhiqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03798)  

**Abstract**: The population of older adults is steadily increasing, with a strong preference for aging-in-place rather than moving to care facilities. Consequently, supporting this growing demographic has become a significant global challenge. However, facilitating successful aging-in-place is challenging, requiring consideration of multiple factors such as data privacy, health status monitoring, and living environments to improve health outcomes. In this paper, we propose an unobtrusive sensor system designed for installation in older adults' homes. Using data from the sensors, our system constructs a digital twin, a virtual representation of events and activities that occurred in the home. The system uses neural network models and decision rules to capture residents' activities and living environments. This digital twin enables continuous health monitoring by providing actionable insights into residents' well-being. Our system is designed to be low-cost and privacy-preserving, with the aim of providing green and safe monitoring for the health of older adults. We have successfully deployed our system in two homes over a time period of two months, and our findings demonstrate the feasibility and effectiveness of digital twin technology in supporting independent living for older adults. This study highlights that our system could revolutionize elder care by enabling personalized interventions, such as lifestyle adjustments, medical treatments, or modifications to the residential environment, to enhance health outcomes. 

**Abstract (ZH)**: 老年人口不断增加，偏向于居家养老而非入住护理设施。因此，支持这一不断壮大的人口群体已成为全球性的重要挑战。然而，促进成功的居家养老具有挑战性，需要考虑多个因素，如数据隐私、健康状况监控和生活环境，以改善健康结果。本文 propose 一种用于安装在老年人家中无侵扰传感器系统。利用传感器数据，该系统构建了一个数字孪生体，即家庭中发生事件和活动的虚拟表示。系统使用神经网络模型和决策规则来捕捉居民的活动和生活环境。此数字孪生体通过提供有关居民福祉的实际行动建议，实现持续的健康监控。本系统旨在低成本和保护隐私的情况下运行，目标是为老年人提供绿色和安全的健康监测。我们在两个月内成功部署了该系统于两个家庭，并且我们的研究结果表明，数字孪生技术在支持老年人独立生活方面具有可行性和有效性。本研究强调，我们的系统有望通过实现个性化干预，如生活方式调整、医疗治疗或住宅环境的修改，彻底改变老年护理领域，以提升健康结果。 

---
# Entropy-Based Block Pruning for Efficient Large Language Models 

**Title (ZH)**: 基于熵的块剪枝以实现高效的大规模语言模型 

**Authors**: Liangwei Yang, Yuhui Xu, Juntao Tan, Doyen Sahoo, Silvio Savarese, Caiming Xiong, Huan Wang, Shelby Heinecke  

**Link**: [PDF](https://arxiv.org/pdf/2504.03794)  

**Abstract**: As large language models continue to scale, their growing computational and storage demands pose significant challenges for real-world deployment. In this work, we investigate redundancy within Transformer-based models and propose an entropy-based pruning strategy to enhance efficiency while maintaining performance. Empirical analysis reveals that the entropy of hidden representations decreases in the early blocks but progressively increases across most subsequent blocks. This trend suggests that entropy serves as a more effective measure of information richness within computation blocks. Unlike cosine similarity, which primarily captures geometric relationships, entropy directly quantifies uncertainty and information content, making it a more reliable criterion for pruning. Extensive experiments demonstrate that our entropy-based pruning approach surpasses cosine similarity-based methods in reducing model size while preserving accuracy, offering a promising direction for efficient model deployment. 

**Abstract (ZH)**: 随着大型语言模型的不断扩展，其日益增长的计算和存储需求为实际部署带来了显著挑战。在本文中，我们探讨了Transformer基模型内的冗余，并提出了一种基于熵的剪枝策略，以提高效率同时保持性能。实证分析表明，隐藏表示的熵在早期块中减少，但在大多数后续块中逐渐增加。这一趋势表明，熵是更有效的计算块内信息丰富度的衡量标准。与主要捕捉几何关系的余弦相似性不同，熵直接量化不确定性与信息含量，使其成为更可靠的剪枝标准。广泛的实验表明，我们的基于熵的剪枝方法在减少模型大小的同时保持准确性，为高效的模型部署提供了有前途的方向。 

---
# Outlook Towards Deployable Continual Learning for Particle Accelerators 

**Title (ZH)**: 面向粒子加速器的可部署连续学习展望 

**Authors**: Kishansingh Rajput, Sen Lin, Auralee Edelen, Willem Blokland, Malachi Schram  

**Link**: [PDF](https://arxiv.org/pdf/2504.03793)  

**Abstract**: Particle Accelerators are high power complex machines. To ensure uninterrupted operation of these machines, thousands of pieces of equipment need to be synchronized, which requires addressing many challenges including design, optimization and control, anomaly detection and machine protection. With recent advancements, Machine Learning (ML) holds promise to assist in more advance prognostics, optimization, and control. While ML based solutions have been developed for several applications in particle accelerators, only few have reached deployment and even fewer to long term usage, due to particle accelerator data distribution drifts caused by changes in both measurable and non-measurable parameters. In this paper, we identify some of the key areas within particle accelerators where continual learning can allow maintenance of ML model performance with distribution drifts. Particularly, we first discuss existing applications of ML in particle accelerators, and their limitations due to distribution drift. Next, we review existing continual learning techniques and investigate their potential applications to address data distribution drifts in accelerators. By identifying the opportunities and challenges in applying continual learning, this paper seeks to open up the new field and inspire more research efforts towards deployable continual learning for particle accelerators. 

**Abstract (ZH)**: 粒子加速器是高功率复杂机器。为了确保这些机器不间断运行，需要同步数以千计的设备，这需要解决包括设计、优化和控制、异常检测以及机器保护在内的许多挑战。随着技术的进步，机器学习（ML）有望帮助实现更高级的预诊、优化和控制。虽然已经为粒子加速器的各种应用开发了基于ML的解决方案，但由于可测和不可测参数的变化导致的数据分布漂移，只有少数方案实现了部署，并且长期使用的情况更加罕见。本文识别了粒子加速器中的一些关键领域，其中持续学习可以在数据分布漂移的情况下保持ML模型性能。特别地，我们首先讨论了ML在粒子加速器中的现有应用及其因数据分布漂移造成的局限性，接下来我们回顾了现有的持续学习技术，并探讨了它们在加速器中应对数据分布漂移的潜在应用。通过识别应用持续学习的机会和挑战，本文旨在开拓这一新领域，并激励更多针对粒子加速器的可部署持续学习的研究努力。 

---
# DP-LET: An Efficient Spatio-Temporal Network Traffic Prediction Framework 

**Title (ZH)**: DP-LET：一种高效的时空网络流量预测框架 

**Authors**: Xintong Wang, Haihan Nan, Ruidong Li, Huaming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03792)  

**Abstract**: Accurately predicting spatio-temporal network traffic is essential for dynamically managing computing resources in modern communication systems and minimizing energy consumption. Although spatio-temporal traffic prediction has received extensive research attention, further improvements in prediction accuracy and computational efficiency remain necessary. In particular, existing decomposition-based methods or hybrid architectures often incur heavy overhead when capturing local and global feature correlations, necessitating novel approaches that optimize accuracy and complexity. In this paper, we propose an efficient spatio-temporal network traffic prediction framework, DP-LET, which consists of a data processing module, a local feature enhancement module, and a Transformer-based prediction module. The data processing module is designed for high-efficiency denoising of network data and spatial decoupling. In contrast, the local feature enhancement module leverages multiple Temporal Convolutional Networks (TCNs) to capture fine-grained local features. Meanwhile, the prediction module utilizes a Transformer encoder to model long-term dependencies and assess feature relevance. A case study on real-world cellular traffic prediction demonstrates the practicality of DP-LET, which maintains low computational complexity while achieving state-of-the-art performance, significantly reducing MSE by 31.8% and MAE by 23.1% compared to baseline models. 

**Abstract (ZH)**: 准确预测时空网络流量对于现代通信系统中动态管理计算资源和减少能源消耗至关重要。尽管时空流量预测已经获得了广泛的研究关注，但预测准确性和计算效率的进一步改进仍然是必要的。特别是，现有的基于分解的方法或混合架构在捕捉局部和全局特征关联时经常产生较大的开销，因此需要优化准确性和复杂性的新型方法。本文提出了一种高效的时空网络流量预测框架DP-LET，该框架包括数据处理模块、局部特征增强模块和基于Transformer的预测模块。数据处理模块旨在高效地去除网络数据噪声和实现空间解耦。相比之下，局部特征增强模块利用多个时序卷积网络（TCNs）捕捉细粒度的局部特征。同时，预测模块利用Transformer编码器建模长期依赖关系并评估特征相关性。在实际蜂窝网络流量预测中的案例研究证明了DP-LET的有效性，该方法在保持低计算复杂性的同时达到了最先进的性能，相比基准模型，MSE降低了31.8%，MAE降低了23.1%。 

---
# Robust Reinforcement Learning from Human Feedback for Large Language Models Fine-Tuning 

**Title (ZH)**: 基于人类反馈的大规模语言模型 fine-tuning 的健壮强化学习方法 

**Authors**: Kai Ye, Hongyi Zhou, Jin Zhu, Francesco Quinzan, Chengchung Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.03784)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has emerged as a key technique for aligning the output of large language models (LLMs) with human preferences. To learn the reward function, most existing RLHF algorithms use the Bradley-Terry model, which relies on assumptions about human preferences that may not reflect the complexity and variability of real-world judgments. In this paper, we propose a robust algorithm to enhance the performance of existing approaches under such reward model misspecifications. Theoretically, our algorithm reduces the variance of reward and policy estimators, leading to improved regret bounds. Empirical evaluations on LLM benchmark datasets demonstrate that the proposed algorithm consistently outperforms existing methods, with 77-81% of responses being favored over baselines on the Anthropic Helpful and Harmless dataset. 

**Abstract (ZH)**: 从人类反馈强化学习（RLHF）：在奖励模型误设情况下增强现有方法的稳健算法 

---
# FAST: Federated Active Learning with Foundation Models for Communication-efficient Sampling and Training 

**Title (ZH)**: FAST：基于基础模型的联邦主动学习方法，用于高效的通信采样和训练 

**Authors**: Haoyuan Li, Jindong Wang, Mathias Funk, Aaqib Saeed  

**Link**: [PDF](https://arxiv.org/pdf/2504.03783)  

**Abstract**: Federated Active Learning (FAL) has emerged as a promising framework to leverage large quantities of unlabeled data across distributed clients while preserving data privacy. However, real-world deployments remain limited by high annotation costs and communication-intensive sampling processes, particularly in a cross-silo setting, when clients possess substantial local datasets. This paper addresses the crucial question: What is the best practice to reduce communication costs in human-in-the-loop learning with minimal annotator effort? Existing FAL methods typically rely on iterative annotation processes that separate active sampling from federated updates, leading to multiple rounds of expensive communication and annotation. In response, we introduce FAST, a two-pass FAL framework that harnesses foundation models for weak labeling in a preliminary pass, followed by a refinement pass focused exclusively on the most uncertain samples. By leveraging representation knowledge from foundation models and integrating refinement steps into a streamlined workflow, FAST substantially reduces the overhead incurred by iterative active sampling. Extensive experiments on diverse medical and natural image benchmarks demonstrate that FAST outperforms existing FAL methods by an average of 4.36% while reducing communication rounds eightfold under a limited 5% labeling budget. 

**Abstract (ZH)**: 联邦主动学习（FAL）作为一种框架，在保持数据隐私的同时，利用分布式客户端跨客户端的大量未标注数据，展现了潜力。然而，在实际部署中，由于高标注成本和耗时的采样过程，尤其是在跨孤岛设置中，客户端拥有大量本地数据集时，部署受到限制。本文解决了关键问题：如何在最小化标注员努力的前提下，减少基于人类在环学习中的通信成本？现有的FAL方法通常依赖于迭代的标注过程，将主动采样与联邦更新区分开来，导致了多轮昂贵的通信和标注。针对这一问题，我们提出了一种两阶段的FAL框架FAST，该框架在初步阶段利用基础模型进行弱标注，然后在第二阶段专注于最不确定的样本进行细化。通过利用基础模型的表示知识并整合细化步骤到精简的工作流程中，FAST显著减少了迭代主动采样带来的开销。在多种医疗和自然图像基准测试上进行的广泛实验表明，与现有FAL方法相比，FAST在5%标注预算下通信轮数减少八倍的同时，平均性能高出4.36%。 

---
# Explainable and Interpretable Forecasts on Non-Smooth Multivariate Time Series for Responsible Gameplay 

**Title (ZH)**: 可解释和可解析的非光滑多变量时间序列预测：负责任的游戏玩法 

**Authors**: Hussain Jagirdar, Rukma Talwadker, Aditya Pareek, Pulkit Agrawal, Tridib Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2504.03777)  

**Abstract**: Multi-variate Time Series (MTS) forecasting has made large strides (with very negligible errors) through recent advancements in neural networks, e.g., Transformers. However, in critical situations like predicting gaming overindulgence that affects one's mental well-being; an accurate forecast without a contributing evidence (explanation) is irrelevant. Hence, it becomes important that the forecasts are Interpretable - intermediate representation of the forecasted trajectory is comprehensible; as well as Explainable - attentive input features and events are accessible for a personalized and timely intervention of players at risk. While the contributing state of the art research on interpretability primarily focuses on temporally-smooth single-process driven time series data, our online multi-player gameplay data demonstrates intractable temporal randomness due to intrinsic orthogonality between player's game outcome and their intent to engage further. We introduce a novel deep Actionable Forecasting Network (AFN), which addresses the inter-dependent challenges associated with three exclusive objectives - 1) forecasting accuracy; 2) smooth comprehensible trajectory and 3) explanations via multi-dimensional input features while tackling the challenges introduced by our non-smooth temporal data, together in one single solution. AFN establishes a \it{new benchmark} via: (i) achieving 25% improvement on the MSE of the forecasts on player data in comparison to the SOM-VAE based SOTA networks; (ii) attributing unfavourable progression of a player's time series to a specific future time step(s), with the premise of eliminating near-future overindulgent player volume by over 18% with player specific actionable inputs feature(s) and (iii) proactively detecting over 23% (100% jump from SOTA) of the to-be overindulgent, players on an average, 4 weeks in advance. 

**Abstract (ZH)**: 多变量时间序列（MTS）预测通过近期神经网络的进步（如变换器）取得了显著进展，但在关键情况下，如预测影响精神健康的游戏过度行为时，准确的预测如果不提供解释（证据）则毫无意义。因此，预测的可解释性和可解释性变得尤为重要——预测的中间表示易于理解；并且注意输入特征和事件易于访问，以便及时干预处于风险中的玩家。虽然现有的可解释性研究主要集中在平滑的单过程驱动的时间序列数据上，我们的在线多人游戏数据展示了由于玩家游戏结果和进一步参与意图之间的固有正交性而带来的难以解决的时间随机性。我们提出了一种新颖的可操作预测网络（AFN），它在三个独立目标——1）预测准确性；2）平滑易懂的轨迹；3）通过多维度输入特征提供解释——的同时，解决了一起浮现的挑战。AFN通过以下方式建立了新的基准：(i) 相比于基于SOM-VAE的当前最佳网络，玩家数据的预测均方误差提高了25%；(ii) 将玩家的不利进展归因于特定的未来时间步长，并通过针对特定玩家的可操作输入特征减少近未来过度行为玩家的比例超过18%；(iii) 预先检测到超过23%（相比当前最佳性能提高了100%）的即将过度行为的玩家，平均提前4周。 

---
# Advancing Air Quality Monitoring: TinyML-Based Real-Time Ozone Prediction with Cost-Effective Edge Devices 

**Title (ZH)**: 基于经济实惠边缘设备的TinyML实时臭氧预测以推进空气质量监测 

**Authors**: Huam Ming Ken, Mehran Behjati  

**Link**: [PDF](https://arxiv.org/pdf/2504.03776)  

**Abstract**: The escalation of urban air pollution necessitates innovative solutions for real-time air quality monitoring and prediction. This paper introduces a novel TinyML-based system designed to predict ozone concentration in real-time. The system employs an Arduino Nano 33 BLE Sense microcontroller equipped with an MQ7 sensor for carbon monoxide (CO) detection and built-in sensors for temperature and pressure measurements. The data, sourced from a Kaggle dataset on air quality parameters from India, underwent thorough cleaning and preprocessing. Model training and evaluation were performed using Edge Impulse, considering various combinations of input parameters (CO, temperature, and pressure). The optimal model, incorporating all three variables, achieved a mean squared error (MSE) of 0.03 and an R-squared value of 0.95, indicating high predictive accuracy. The regression model was deployed on the microcontroller via the Arduino IDE, showcasing robust real-time performance. Sensitivity analysis identified CO levels as the most critical predictor of ozone concentration, followed by pressure and temperature. The system's low-cost and low-power design makes it suitable for widespread implementation, particularly in resource-constrained settings. This TinyML approach provides precise real-time predictions of ozone levels, enabling prompt responses to pollution events and enhancing public health protection. 

**Abstract (ZH)**: 基于TinyML的实时臭氧浓度预测系统：创新应对城市空气污染挑战 

---
# FlowKV: A Disaggregated Inference Framework with Low-Latency KV Cache Transfer and Load-Aware Scheduling 

**Title (ZH)**: FlowKV：一种低延迟键值缓存传输与负载aware调度的解耦推理框架 

**Authors**: Weiqing Li, Guochao Jiang, Xiangyong Ding, Zhangcheng Tao, Chuzhan Hao, Chenfeng Xu, Yuewei Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03775)  

**Abstract**: Disaggregated inference has become an essential framework that separates the prefill (P) and decode (D) stages in large language model inference to improve throughput. However, the KV cache transfer faces significant delays between prefill and decode nodes. The block-wise calling method and discontinuous KV cache memory allocation increase the number of calls to the transmission kernel. Additionally, existing frameworks often fix the roles of P and D nodes, leading to computational imbalances. In this paper, we propose FlowKV, a novel disaggregated inference framework, which reduces the average transmission latency of KV cache by 96%, from 0.944s to 0.053s, almost eliminating the transfer time relative to the total request latency by optimizing the KV cache transfer. FlowKV introduces the Load-Aware Scheduler for balanced request scheduling and flexible PD node allocation. This design maximizes hardware resource utilization, achieving peak system throughput across various scenarios, including normal, computational imbalance, and extreme overload conditions. Experimental results demonstrate that FlowKV significantly accelerates inference by 15.2%-48.9% on LongBench dataset compared to the baseline and supports applications with heterogeneous GPUs. 

**Abstract (ZH)**: 基于流量的解聚推理框架FlowKV 

---
# Exploring energy consumption of AI frameworks on a 64-core RV64 Server CPU 

**Title (ZH)**: 探索AI框架在64核RV64服务器CPU上的能效消耗 

**Authors**: Giulio Malenza, Francesco Targa, Adriano Marques Garcia, Marco Aldinucci, Robert Birke  

**Link**: [PDF](https://arxiv.org/pdf/2504.03774)  

**Abstract**: In today's era of rapid technological advancement, artificial intelligence (AI) applications require large-scale, high-performance, and data-intensive computations, leading to significant energy demands. Addressing this challenge necessitates a combined approach involving both hardware and software innovations. Hardware manufacturers are developing new, efficient, and specialized solutions, with the RISC-V architecture emerging as a prominent player due to its open, extensible, and energy-efficient instruction set architecture (ISA). Simultaneously, software developers are creating new algorithms and frameworks, yet their energy efficiency often remains unclear. In this study, we conduct a comprehensive benchmark analysis of machine learning (ML) applications on the 64-core SOPHON SG2042 RISC-V architecture. We specifically analyze the energy consumption of deep learning inference models across three leading AI frameworks: PyTorch, ONNX Runtime, and TensorFlow. Our findings show that frameworks using the XNNPACK back-end, such as ONNX Runtime and TensorFlow, consume less energy compared to PyTorch, which is compiled with the native OpenBLAS back-end. 

**Abstract (ZH)**: 在快速 technological advancement时代，人工智能（AI）应用需要大规模、高性能和数据密集型计算，导致了显著的能源需求。为应对这一挑战，需要硬件和软件创新相结合的方法。硬件制造商正在开发新的、高效且专门化的新解决方案，RISC-V架构因其开放、可扩展且能效高的指令集架构（ISA）而成为引人注目的参与者。与此同时，软件开发人员正在创建新的算法和框架，但它们的能源效率往往不明确。在本研究中，我们对64核心SOPHON SG2042 RISC-V架构上的机器学习（ML）应用进行了全面基准分析。我们特别分析了三个主流AI框架——PyTorch、ONNX Runtime和TensorFlow——在深度学习推理模型中的能耗情况。研究发现，使用XNNPACK后端的框架，如ONNX Runtime和TensorFlow，相比于使用原生OpenBLAS后端编译的PyTorch，能耗较低。 

---
# SHapley Estimated Explanation (SHEP): A Fast Post-Hoc Attribution Method for Interpreting Intelligent Fault Diagnosis 

**Title (ZH)**: 基于Shapley值的快速事后归因方法：解释智能故障诊断 

**Authors**: Qian Chen, Xingjian Dong, Zhike Peng, Guang Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.03773)  

**Abstract**: Despite significant progress in intelligent fault diagnosis (IFD), the lack of interpretability remains a critical barrier to practical industrial applications, driving the growth of interpretability research in IFD. Post-hoc interpretability has gained popularity due to its ability to preserve network flexibility and scalability without modifying model structures. However, these methods often yield suboptimal time-domain explanations. Recently, combining domain transform with SHAP has improved interpretability by extending explanations to more informative domains. Nonetheless, the computational expense of SHAP, exacerbated by increased dimensions from domain transforms, remains a major challenge. To address this, we propose patch-wise attribution and SHapley Estimated Explanation (SHEP). Patch-wise attribution reduces feature dimensions at the cost of explanation granularity, while SHEP simplifies subset enumeration to approximate SHAP, reducing complexity from exponential to linear. Together, these methods significantly enhance SHAP's computational efficiency, providing feasibility for real-time interpretation in monitoring tasks. Extensive experiments confirm SHEP's efficiency, interpretability, and reliability in approximating SHAP. Additionally, with open-source code, SHEP has the potential to serve as a benchmark for post-hoc interpretability in IFD. The code is available on this https URL. 

**Abstract (ZH)**: 尽管在智能故障诊断（IFD）方面取得了显著进展，但缺乏可解释性仍然是其在工业应用中的一大障碍，推动了IFD可解释性研究的发展。后 hoc 可解释性由于能够保留网络的灵活性和可扩展性而未改变模型结构的情况下提高可解释性，因而备受欢迎。然而，这些方法通常会导致时间域解释的效果不佳。最近，将领域变换与 SHAP 结合使用，通过扩展解释到更具信息量的领域，提高了可解释性。不过，由于领域变换导致维度增加，SHAP 的计算开销仍然是一个主要挑战。为解决这一问题，我们提出了一种基于块的归因和 SHapley 估计解释（SHEP）方法。基于块的归因降低了特征维度，但牺牲了解释的精细度；而 SHEP 简化了子集枚举以近似 SHAP，将复杂度从指数级降低到线性级。这两种方法显著提高了 SHAP 的计算效率，为监控任务中的实时解释提供了可行性。广泛的实验验证了 SHEP 在效率、可解释性和近似 SHAP 方面的可靠性和有效性。此外，开源代码使 SHEP 成为 IFD 后 hoc 可解释性的基准。代码可在以下网址获取：this https URL。 

---
# JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model 

**Title (ZH)**: JailDAM：面向视觉语言模型的自适应记忆逃逸检测 

**Authors**: Yi Nian, Shenzhe Zhu, Yuehan Qin, Li Li, Ziyi Wang, Chaowei Xiao, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.03770)  

**Abstract**: Multimodal large language models (MLLMs) excel in vision-language tasks but also pose significant risks of generating harmful content, particularly through jailbreak attacks. Jailbreak attacks refer to intentional manipulations that bypass safety mechanisms in models, leading to the generation of inappropriate or unsafe content. Detecting such attacks is critical to ensuring the responsible deployment of MLLMs. Existing jailbreak detection methods face three primary challenges: (1) Many rely on model hidden states or gradients, limiting their applicability to white-box models, where the internal workings of the model are accessible; (2) They involve high computational overhead from uncertainty-based analysis, which limits real-time detection, and (3) They require fully labeled harmful datasets, which are often scarce in real-world settings. To address these issues, we introduce a test-time adaptive framework called JAILDAM. Our method leverages a memory-based approach guided by policy-driven unsafe knowledge representations, eliminating the need for explicit exposure to harmful data. By dynamically updating unsafe knowledge during test-time, our framework improves generalization to unseen jailbreak strategies while maintaining efficiency. Experiments on multiple VLM jailbreak benchmarks demonstrate that JAILDAM delivers state-of-the-art performance in harmful content detection, improving both accuracy and speed. 

**Abstract (ZH)**: 多模态大型语言模型在视觉语言任务中表现出色，但也存在通过 Jailbreak 攻击生成有害内容的重大风险。Jailbreak 攻击指的是故意操纵以绕过模型的安全机制，导致生成不适当或不安全的内容。检测此类攻击对于确保多模态大型语言模型的负责任部署至关重要。现有的 Jailbreak 检测方法面临三大主要挑战：（1）许多方法依赖于模型的隐藏状态或梯度，限制了其在白盒模型中的适用性；（2）基于不确定性分析的高计算开销限制了实时检测；（3）需要完全标记的有害数据集，而在实际应用中此类数据集往往稀缺。为解决这些问题，我们提出了一种测试时自适应框架 JAILDAM。该方法利用基于策略驱动的不安全知识表示的内存导向方法，避免显式暴露于有害数据。通过在测试时动态更新不安全知识，我们的框架在保持效率的同时，能够更好地泛化到未见过的 Jailbreak 策略。多项视觉语言模型 Jailbreak 基准测试表明，JAILDAM 在有害内容检测方面达到了最先进的性能，既提高了准确性又加快了速度。 

---
# MCP Safety Audit: LLMs with the Model Context Protocol Allow Major Security Exploits 

**Title (ZH)**: MCP安全审计：具有模型上下文协议的LLM允许重大安全exploits 

**Authors**: Brandon Radosevich, John Halloran  

**Link**: [PDF](https://arxiv.org/pdf/2504.03767)  

**Abstract**: To reduce development overhead and enable seamless integration between potential components comprising any given generative AI application, the Model Context Protocol (MCP) (Anthropic, 2024) has recently been released and subsequently widely adopted. The MCP is an open protocol that standardizes API calls to large language models (LLMs), data sources, and agentic tools. By connecting multiple MCP servers, each defined with a set of tools, resources, and prompts, users are able to define automated workflows fully driven by LLMs. However, we show that the current MCP design carries a wide range of security risks for end users. In particular, we demonstrate that industry-leading LLMs may be coerced into using MCP tools to compromise an AI developer's system through various attacks, such as malicious code execution, remote access control, and credential theft. To proactively mitigate these and related attacks, we introduce a safety auditing tool, MCPSafetyScanner, the first agentic tool to assess the security of an arbitrary MCP server. MCPScanner uses several agents to (a) automatically determine adversarial samples given an MCP server's tools and resources; (b) search for related vulnerabilities and remediations based on those samples; and (c) generate a security report detailing all findings. Our work highlights serious security issues with general-purpose agentic workflows while also providing a proactive tool to audit MCP server safety and address detected vulnerabilities before deployment.
The described MCP server auditing tool, MCPSafetyScanner, is freely available at: this https URL 

**Abstract (ZH)**: 面向潜在组件之间无缝集成以减少开发成本的Model Context Protocol (MCP) (Anthropic, 2024)最近已发布并广泛采用。然而，我们表明当前的MCP设计对终端用户存在广泛的安全风险。特别是，我们展示了一旦受到恶意攻击，如恶意代码执行、远程访问控制和凭证窃取，行业领先的LLM可能被迫利用MCP工具来攻击AI开发者的系统。为此，我们引入了MCPSafetyScanner这一安全审计工具，它是首个评估任意MCP服务器安全性的机构工具。MCPScanner利用多个代理（a）自动确定给定MCP服务器工具和资源的对抗样本；（b）基于这些样本搜索相关漏洞和修复方法；（c）生成详细的安全报告。我们的工作突显了通用机构工作流中的严重安全问题，同时也提供了一个主动工具来审计MCP服务器安全性和在部署前解决检测到的漏洞。 

---
# Efficient Calibration for RRAM-based In-Memory Computing using DoRA 

**Title (ZH)**: 基于DoRA的RRAM基存算一体化高效校准方法 

**Authors**: Weirong Dong, Kai Zhou, Zhen Kong, Quan Cheng, Junkai Huang, Zhengke Yang, Masanori Hashimoto, Longyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.03763)  

**Abstract**: Resistive In-Memory Computing (RIMC) offers ultra-efficient computation for edge AI but faces accuracy degradation due to RRAM conductance drift over time. Traditional retraining methods are limited by RRAM's high energy consumption, write latency, and endurance constraints. We propose a DoRA-based calibration framework that restores accuracy by compensating influential weights with minimal calibration parameters stored in SRAM, leaving RRAM weights untouched. This eliminates in-field RRAM writes, ensuring energy-efficient, fast, and reliable calibration. Experiments on RIMC-based ResNet50 (ImageNet-1K) demonstrate 69.53% accuracy restoration using just 10 calibration samples while updating only 2.34% of parameters. 

**Abstract (ZH)**: 基于DoRA的校准框架在保持RRAM权重不变的情况下通过少量SRAM存储的校准参数恢复Resistive In-Memory Computing (RIMC)的准确性 

---
# Emerging Cyber Attack Risks of Medical AI Agents 

**Title (ZH)**: 新兴医疗AI代理的网络攻击风险 

**Authors**: Jianing Qiu, Lin Li, Jiankai Sun, Hao Wei, Zhe Xu, Kyle Lam, Wu Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.03759)  

**Abstract**: Large language models (LLMs)-powered AI agents exhibit a high level of autonomy in addressing medical and healthcare challenges. With the ability to access various tools, they can operate within an open-ended action space. However, with the increase in autonomy and ability, unforeseen risks also arise. In this work, we investigated one particular risk, i.e., cyber attack vulnerability of medical AI agents, as agents have access to the Internet through web browsing tools. We revealed that through adversarial prompts embedded on webpages, cyberattackers can: i) inject false information into the agent's response; ii) they can force the agent to manipulate recommendation (e.g., healthcare products and services); iii) the attacker can also steal historical conversations between the user and agent, resulting in the leak of sensitive/private medical information; iv) furthermore, the targeted agent can also cause a computer system hijack by returning a malicious URL in its response. Different backbone LLMs were examined, and we found such cyber attacks can succeed in agents powered by most mainstream LLMs, with the reasoning models such as DeepSeek-R1 being the most vulnerable. 

**Abstract (ZH)**: 基于大型语言模型的AI代理在应对医疗和健康挑战方面表现出高度自主性，但随着自主性和能力的增强，未预见的安全风险也逐渐显现。本文探究了其中一种风险，即通过网页中的对抗性提示，黑客可以对医疗AI代理进行攻击：i) 注入虚假信息；ii) 强迫使代理操控推荐（如医疗产品和服务）；iii) 盗取用户与代理的历史对话记录，导致敏感/私密医疗信息泄露；iv) 进一步，受攻击代理还可能通过返回恶意URL导致计算机系统被劫持。不同基础模型的大型语言模型均存在此类攻击风险，其中深度探索-R1等推理模型最为脆弱。 

---
# ProtoGCD: Unified and Unbiased Prototype Learning for Generalized Category Discovery 

**Title (ZH)**: 统一无偏的原型学习方法：通用类别发现 

**Authors**: Shijie Ma, Fei Zhu, Xu-Yao Zhang, Cheng-Lin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03755)  

**Abstract**: Generalized category discovery (GCD) is a pragmatic but underexplored problem, which requires models to automatically cluster and discover novel categories by leveraging the labeled samples from old classes. The challenge is that unlabeled data contain both old and new classes. Early works leveraging pseudo-labeling with parametric classifiers handle old and new classes separately, which brings about imbalanced accuracy between them. Recent methods employing contrastive learning neglect potential positives and are decoupled from the clustering objective, leading to biased representations and sub-optimal results. To address these issues, we introduce a unified and unbiased prototype learning framework, namely ProtoGCD, wherein old and new classes are modeled with joint prototypes and unified learning objectives, {enabling unified modeling between old and new classes}. Specifically, we propose a dual-level adaptive pseudo-labeling mechanism to mitigate confirmation bias, together with two regularization terms to collectively help learn more suitable representations for GCD. Moreover, for practical considerations, we devise a criterion to estimate the number of new classes. Furthermore, we extend ProtoGCD to detect unseen outliers, achieving task-level unification. Comprehensive experiments show that ProtoGCD achieves state-of-the-art performance on both generic and fine-grained datasets. The code is available at this https URL. 

**Abstract (ZH)**: 泛化类别发现的统一无偏原型学习框架（ProtoGCD） 

---
# Proof of Humanity: A Multi-Layer Network Framework for Certifying Human-Originated Content in an AI-Dominated Internet 

**Title (ZH)**: 人性验证：一种在人工智能主导的互联网中认证人类生成内容的多层网络框架 

**Authors**: Sebastian Barros  

**Link**: [PDF](https://arxiv.org/pdf/2504.03752)  

**Abstract**: The rapid proliferation of generative AI has led to an internet increasingly populated with synthetic content-text, images, audio, and video generated without human intervention. As the distinction between human and AI-generated data blurs, the ability to verify content origin becomes critical for applications ranging from social media and journalism to legal and financial systems.
In this paper, we propose a conceptual, multi-layer architectural framework that enables telecommunications networks to act as infrastructure level certifiers of human-originated content. By leveraging identity anchoring at the physical layer, metadata propagation at the network and transport layers, and cryptographic attestations at the session and application layers, Telcos can provide an end-to-end Proof of Humanity for data traversing their networks.
We outline how each OSI layer can contribute to this trust fabric using technical primitives such as SIM/eSIM identity, digital signatures, behavior-based ML heuristics, and edge-validated APIs. The framework is presented as a foundation for future implementation, highlighting monetization pathways for telcos such as trust-as-a-service APIs, origin-certified traffic tiers, and regulatory compliance tools.
The paper does not present implementation or benchmarking results but offers a technically detailed roadmap and strategic rationale for transforming Telcos into validators of digital authenticity in an AI-dominated internet. Security, privacy, and adversarial considerations are discussed as directions for future work. 

**Abstract (ZH)**: 生成式AI的迅速发展导致互联网上充斥着无需人类干预生成的合成内容（包括文字、图像、音频和视频）。随着人类生成数据与AI生成数据之间的界限模糊，验证内容来源的能力对社交媒体、新闻业、法律和金融系统等应用而言变得至关重要。
在本文中，我们提出了一种概念性的多层次架构框架，使电信网络能够作为基础设施级的人类生成内容认证者。通过在物理层利用身份锚定、在网络层和传输层利用元数据传播、在会话层和应用层利用加密证实，电信运营商可以为其网络中传输的数据提供端到端的人类身份证明。
我们阐述了每一层开放系统互连（OSI）模型如何通过技术原语如SIM/eSIM身份、数字签名、基于行为的机器学习启发式规则以及边缘验证API来贡献于这一信任织网。该框架被提出作为未来实施的基础，强调了电信运营商通过信任即服务API、基于来源认证的流量层级以及合规工具等方面的盈利途径。
本文未提供实施或基准测试结果，而是提供了一份详细的技术路线图和战略 rationale，旨在使电信运营商转变为AI主导互联网中数字真实性的验证者。安全、隐私和对抗性考虑被讨论为未来工作的方向。 

---
# TDBench: Benchmarking Vision-Language Models in Understanding Top-Down Images 

**Title (ZH)**: TDBench: 评估视觉-语言模型理解自上而下图像的能力 

**Authors**: Kaiyuan Hou, Minghui Zhao, Lilin Xu, Yuang Fan, Xiaofan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03748)  

**Abstract**: The rapid emergence of Vision-Language Models (VLMs) has significantly advanced multimodal understanding, enabling applications in scene comprehension and visual reasoning. While these models have been primarily evaluated and developed for front-view image understanding, their capabilities in interpreting top-down images have received limited attention, partly due to the scarcity of diverse top-down datasets and the challenges in collecting such data. In contrast, top-down vision provides explicit spatial overviews and improved contextual understanding of scenes, making it particularly valuable for tasks like autonomous navigation, aerial imaging, and spatial planning. In this work, we address this gap by introducing TDBench, a comprehensive benchmark for VLMs in top-down image understanding. TDBench is constructed from public top-down view datasets and high-quality simulated images, including diverse real-world and synthetic scenarios. TDBench consists of visual question-answer pairs across ten evaluation dimensions of image understanding. Moreover, we conduct four case studies that commonly happen in real-world scenarios but are less explored. By revealing the strengths and limitations of existing VLM through evaluation results, we hope TDBench to provide insights for motivating future research. Project homepage: this https URL 

**Abstract (ZH)**: Vision-Language模型的快速兴起显著推动了多模态理解的发展，使其在场景理解与视觉推理方面得到广泛应用。尽管这些模型主要被评估和开发用于前视图图像理解，但它们在解释俯视图图像方面的能力却受到较少关注，部分原因是缺乏多样化的俯视图数据集，以及收集此类数据的挑战。相比之下，俯视图视觉提供了明确的空间概述和情境理解，尤其对于自主导航、航空成像和空间规划等任务具有重要价值。本文通过引入TDBench——一个全面的俯视图图像理解基准测试，来填补这一空白。TDBench基于公共俯视图视图数据集和高质量模拟图像构建，涵盖多种现实世界和合成场景。TDBench包括十种评估维度的视觉问题-答案对。此外，我们还进行了四个常见于实际场景但较少被探索的案例研究。通过评估结果揭示现有Vision-Language模型的优势和局限，我们希望TDBench能够为未来的研究提供见解。项目主页：this https URL 

---
# Enhancing Biologically Inspired Hierarchical Temporal Memory with Hardware-Accelerated Reflex Memory 

**Title (ZH)**: 基于硬件加速反射记忆增强生物启发的分层临时记忆 

**Authors**: Pavia Bera, Sabrina Hassan Moon, Jennifer Adorno, Dayane Alfenas Reis, Sanjukta Bhanja  

**Link**: [PDF](https://arxiv.org/pdf/2504.03746)  

**Abstract**: The rapid expansion of the Internet of Things (IoT) generates zettabytes of data that demand efficient unsupervised learning systems. Hierarchical Temporal Memory (HTM), a third-generation unsupervised AI algorithm, models the neocortex of the human brain by simulating columns of neurons to process and predict sequences. These neuron columns can memorize and infer sequences across multiple orders. While multiorder inferences offer robust predictive capabilities, they often come with significant computational overhead. The Sequence Memory (SM) component of HTM, which manages these inferences, encounters bottlenecks primarily due to its extensive programmable interconnects. In many cases, it has been observed that first-order temporal relationships have proven to be sufficient without any significant loss in efficiency. This paper introduces a Reflex Memory (RM) block, inspired by the Spinal Cord's working mechanisms, designed to accelerate the processing of first-order inferences. The RM block performs these inferences significantly faster than the SM. The integration of RM with HTM forms a system called the Accelerated Hierarchical Temporal Memory (AHTM), which processes repetitive information more efficiently than the original HTM while still supporting multiorder inferences. The experimental results demonstrate that the HTM predicts an event in 0.945 s, whereas the AHTM module does so in 0.125 s. Additionally, the hardware implementation of RM in a content-addressable memory (CAM) block, known as Hardware-Accelerated Hierarchical Temporal Memory (H-AHTM), predicts an event in just 0.094 s, significantly improving inference speed. Compared to the original algorithm \cite{bautista2020matlabhtm}, AHTM accelerates inference by up to 7.55x, while H-AHTM further enhances performance with a 10.10x speedup. 

**Abstract (ZH)**: 物联网的快速扩张生成了泽字节的数据，要求高效的无监督学习系统。层级时间记忆（HTM），一种第三代无监督人工智能算法，通过模拟神经元列来建模人脑的新皮层，处理和预测序列。这些神经元列可以在多个层次上记忆和推断序列。虽然多层次的推断提供了稳健的预测能力，但往往伴随着巨大的计算开销。HTM中的序列记忆（SM）组件管理这些推断，主要由于其广泛的可编程互联遇到了瓶颈。在许多情况下，观察到一阶时间关系已证明是足够的，而不会显著损失效率。本文提出了一种灵感来源于脊髓工作机制的反射记忆（RM）块，用于加速一阶推断的处理速度。RM块显著快于SM进行这些推断。RM块与HTM的结合形成了加速层级时间记忆（AHTM）系统，该系统在仍然支持多层次推断的同时，更高效地处理重复信息。实验结果表明，HTM预测事件所需时间为0.945秒，而AHTM模块仅需0.125秒。此外，RM在内容可寻址内存（CAM）块中的硬件实现——硬件加速层级时间记忆（H-AHTM）——预测事件所需时间仅为0.094秒，显著提高了推断速度。与原始算法相比，AHTM将推断加速7.55倍，而H-AHTM进一步提升了10.10倍的性能。 

---
# Comparative Explanations: Explanation Guided Decision Making for Human-in-the-Loop Preference Selection 

**Title (ZH)**: 比较解释：基于解释的决策制定以指导人类在环偏好选择 

**Authors**: Tanmay Chakraborty, Christian Wirth, Christin Seifert  

**Link**: [PDF](https://arxiv.org/pdf/2504.03744)  

**Abstract**: This paper introduces Multi-Output LOcal Narrative Explanation (MOLONE), a novel comparative explanation method designed to enhance preference selection in human-in-the-loop Preference Bayesian optimization (PBO). The preference elicitation in PBO is a non-trivial task because it involves navigating implicit trade-offs between vector-valued outcomes, subjective priorities of decision-makers, and decision-makers' uncertainty in preference selection. Existing explainable AI (XAI) methods for BO primarily focus on input feature importance, neglecting the crucial role of outputs (objectives) in human preference elicitation. MOLONE addresses this gap by providing explanations that highlight both input and output importance, enabling decision-makers to understand the trade-offs between competing objectives and make more informed preference selections. MOLONE focuses on local explanations, comparing the importance of input features and outcomes across candidate samples within a local neighborhood of the search space, thus capturing nuanced differences relevant to preference-based decision-making. We evaluate MOLONE within a PBO framework using benchmark multi-objective optimization functions, demonstrating its effectiveness in improving convergence compared to noisy preference selections. Furthermore, a user study confirms that MOLONE significantly accelerates convergence in human-in-the-loop scenarios by facilitating more efficient identification of preferred options. 

**Abstract (ZH)**: 多输出局部叙事解释(MOLONE)在人类在环 Preference Bayesian 优化中的新颖比较解释方法 

---
# Modelling bounded rational decision-making through Wasserstein constraints 

**Title (ZH)**: 通过Wasserstein约束建模有界理性决策 

**Authors**: Benjamin Patrick Evans, Leo Ardon, Sumitra Ganesh  

**Link**: [PDF](https://arxiv.org/pdf/2504.03743)  

**Abstract**: Modelling bounded rational decision-making through information constrained processing provides a principled approach for representing departures from rationality within a reinforcement learning framework, while still treating decision-making as an optimization process. However, existing approaches are generally based on Entropy, Kullback-Leibler divergence, or Mutual Information. In this work, we highlight issues with these approaches when dealing with ordinal action spaces. Specifically, entropy assumes uniform prior beliefs, missing the impact of a priori biases on decision-makings. KL-Divergence addresses this, however, has no notion of "nearness" of actions, and additionally, has several well known potentially undesirable properties such as the lack of symmetry, and furthermore, requires the distributions to have the same support (e.g. positive probability for all actions). Mutual information is often difficult to estimate. Here, we propose an alternative approach for modeling bounded rational RL agents utilising Wasserstein distances. This approach overcomes the aforementioned issues. Crucially, this approach accounts for the nearness of ordinal actions, modeling "stickiness" in agent decisions and unlikeliness of rapidly switching to far away actions, while also supporting low probability actions, zero-support prior distributions, and is simple to calculate directly. 

**Abstract (ZH)**: 通过信息受限处理建模有界理性决策提供了一种在强化学习框架内表示理性偏差的原则性方法，同时仍将决策视为优化过程。然而，现有方法通常基于熵、Kullback-Leibler散度或互信息。在本工作中，我们指出了这些方法在处理序数动作空间时存在的问题。具体而言，熵假定均匀先验信念，忽略了先验偏见对决策的影响。KL散度确实解决了这一问题，但它没有“动作接近性”的概念，并且还具有非对称性等众所周知的潜在不良性质，此外，要求分布具有相同的支撑（例如，所有动作的正概率）。互信息通常难以估计。在此，我们提出了一种利用Wasserstein距离建模有界理性RL代理的替代方法。这种方法克服了上述问题。至关重要的是，这种方法考虑了序数动作的接近性，模型了代理决策中的“粘滞性”，以及不太可能迅速切换到远离的动作，同时支持低概率动作、零支撑先验分布，并且易于直接计算。 

---
# Hierarchical Local-Global Feature Learning for Few-shot Malicious Traffic Detection 

**Title (ZH)**: 局部-全局特征层次学习在少量样本恶意流量检测中的应用 

**Authors**: Songtao Peng, Lei Wang, Wu Shuai, Hao Song, Jiajun Zhou, Shanqing Yu, Qi Xuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.03742)  

**Abstract**: With the rapid growth of internet traffic, malicious network attacks have become increasingly frequent and sophisticated, posing significant threats to global cybersecurity. Traditional detection methods, including rule-based and machine learning-based approaches, struggle to accurately identify emerging threats, particularly in scenarios with limited samples. While recent advances in few-shot learning have partially addressed the data scarcity issue, existing methods still exhibit high false positive rates and lack the capability to effectively capture crucial local traffic patterns. In this paper, we propose HLoG, a novel hierarchical few-shot malicious traffic detection framework that leverages both local and global features extracted from network sessions. HLoG employs a sliding-window approach to segment sessions into phases, capturing fine-grained local interaction patterns through hierarchical bidirectional GRU encoding, while simultaneously modeling global contextual dependencies. We further design a session similarity assessment module that integrates local similarity with global self-attention-enhanced representations, achieving accurate and robust few-shot traffic classification. Comprehensive experiments on three meticulously reconstructed datasets demonstrate that HLoG significantly outperforms existing state-of-the-art methods. Particularly, HLoG achieves superior recall rates while substantially reducing false positives, highlighting its effectiveness and practical value in real-world cybersecurity applications. 

**Abstract (ZH)**: 随着互联网流量的迅速增长，恶意网络攻击事件变得更加频繁和复杂，对全球网络安全造成了重大威胁。传统的检测方法，包括基于规则和基于机器学习的方法，难以准确识别新兴威胁，特别是在样本有限的情况下。尽管最近在少样本学习方面的进步部分解决了数据稀缺问题，但现有方法仍存在较高的误报率，并且无法有效捕捉关键的局部流量模式。在本文中，我们提出了一种新的层次少样本恶意流量检测框架HLoG，该框架利用从网络会话中提取的局部和全局特征。HLoG采用滑动窗口方法将会话分割成不同的阶段，通过分层双向GRU编码捕捉精细的局部交互模式，同时建模全局上下文依赖关系。进一步设计了一个会话相似性评估模块，将局部相似性与全局自注意力增强表示相结合，实现准确且稳健的少样本流量分类。在三个精心重构的数据集上的全面实验表明，HLoG显著优于现有最先进的方法。特别是，HLoG在大幅减少误报的同时实现了更高的召回率，突显了其在实际网络安全应用中的有效性和实用性。 

---
# Brain Network Classification Based on Graph Contrastive Learning and Graph Transformer 

**Title (ZH)**: 基于图对比学习和图变换器的脑网络分类 

**Authors**: ZhiTeng Zhu, Lan Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.03740)  

**Abstract**: The dynamic characterization of functional brain networks is of great significance for elucidating the mechanisms of human brain function. Although graph neural networks have achieved remarkable progress in functional network analysis, challenges such as data scarcity and insufficient supervision persist. To address the limitations of limited training data and inadequate supervision, this paper proposes a novel model named PHGCL-DDGformer that integrates graph contrastive learning with graph transformers, effectively enhancing the representation learning capability for brain network classification tasks. To overcome the constraints of existing graph contrastive learning methods in brain network feature extraction, an adaptive graph augmentation strategy combining attribute masking and edge perturbation is implemented for data enhancement. Subsequently, a dual-domain graph transformer (DDGformer) module is constructed to integrate local and global information, where graph convolutional networks aggregate neighborhood features to capture local patterns while attention mechanisms extract global dependencies. Finally, a graph contrastive learning framework is established to maximize the consistency between positive and negative pairs, thereby obtaining high-quality graph representations. Experimental results on real-world datasets demonstrate that the PHGCL-DDGformer model outperforms existing state-of-the-art approaches in brain network classification tasks. 

**Abstract (ZH)**: 功能脑网络的动态表征对于阐明人类脑功能机制具有重要意义。尽管图神经网络在功能网络分析方面取得了显著进展，但数据稀缺和监督不足等问题依然存在。为了解决有限训练数据和不足监督的限制，本文提出了一种名为PHGCL-DDGformer的新模型，该模型将图对比学习与图变压器相结合，有效增强了脑网络分类任务中的表示学习能力。为克服现有图对比学习方法在脑网络特征提取方面的限制，该模型实施了一种结合属性掩蔽和边扰动的自适应图增强策略，以实现数据增强。随后，构建了一个双域图变压器（DDGformer）模块，该模块整合了局部和全局信息；图卷积网络聚合邻域特征以捕获局部模式，而注意力机制则提取全局依赖关系。最后，建立了图对比学习框架以最大化正负样本对的一致性，从而获得高质量的图表示。实验结果表明，在真实数据集上的研究结果证明，PHGCL-DDGformer模型在脑网络分类任务上的表现优于现有最先进的方法。 

---
# A Unified Virtual Mixture-of-Experts Framework:Enhanced Inference and Hallucination Mitigation in Single-Model System 

**Title (ZH)**: 统一虚拟专家混合框架：单模型系统中增强推理和幻觉抑制 

**Authors**: Mingyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03739)  

**Abstract**: Generative models, such as GPT and BERT, have significantly improved performance in tasks like text generation and summarization. However, hallucinations "where models generate non-factual or misleading content" are especially problematic in smaller-scale architectures, limiting their real-world this http URL this paper, we propose a unified Virtual Mixture-of-Experts (MoE) fusion strategy that enhances inference performance and mitigates hallucinations in a single Qwen 1.5 0.5B model without increasing the parameter count. Our method leverages multiple domain-specific expert prompts (with the number of experts being adjustable) to guide the model from different perspectives. We apply a statistical outlier truncation strategy based on the mean and standard deviation to filter out abnormally high probability predictions, and we inject noise into the embedding space to promote output diversity. To clearly assess the contribution of each module, we adopt a fixed voting mechanism rather than a dynamic gating network, thereby avoiding additional confounding factors. We provide detailed theoretical derivations from both statistical and ensemble learning perspectives to demonstrate how our method reduces output variance and suppresses hallucinations. Extensive ablation experiments on dialogue generation tasks show that our approach significantly improves inference accuracy and robustness in small models. Additionally, we discuss methods for evaluating the orthogonality of virtual experts and outline the potential for future work involving dynamic expert weight allocation using gating networks. 

**Abstract (ZH)**: 生成模型，如GPT和BERT，在文本生成和总结等任务中显著提升了性能。然而，在较小规模的架构中，模型生成非事实或误导性内容的“幻觉”问题尤其突出，限制了其在实际应用中的表现。在本文中，我们提出了一种统一的虚拟混合专家（MoE）融合策略，能够在不增加参数数量的情况下，增强推断性能并减轻Qwen 1.5 0.5B模型中的幻觉现象。该方法通过调用量化的领域特定专家提示，从多个角度引导模型。我们采用基于均值和标准差的统计异常值修剪策略去除异常高概率预测，并在嵌入空间中注入噪声以促进输出多样性。为了清晰评估每个模块的贡献，我们采用固定投票机制而非动态门控网络，从而避免额外的混杂因素。我们从统计学和集成学习的角度提供了详细的理论推导，展示了该方法如何减少输出方差并抑制幻觉现象。在对话生成任务上的大量消融实验显示，我们的方法在小型模型中的推断准确性和鲁棒性有显著提升。我们还讨论了评估虚拟专家正交性的方法，并概述了未来使用门控网络进行动态专家权重分配的研究潜力。 

---
# Attention in Diffusion Model: A Survey 

**Title (ZH)**: 扩散模型中的注意力：一个综述 

**Authors**: Litao Hua, Fan Liu, Jie Su, Xingyu Miao, Zizhou Ouyang, Zeyu Wang, Runze Hu, Zhenyu Wen, Bing Zhai, Yang Long, Haoran Duan, Yuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.03738)  

**Abstract**: Attention mechanisms have become a foundational component in diffusion models, significantly influencing their capacity across a wide range of generative and discriminative tasks. This paper presents a comprehensive survey of attention within diffusion models, systematically analysing its roles, design patterns, and operations across different modalities and tasks. We propose a unified taxonomy that categorises attention-related modifications into parts according to the structural components they affect, offering a clear lens through which to understand their functional diversity. In addition to reviewing architectural innovations, we examine how attention mechanisms contribute to performance improvements in diverse applications. We also identify current limitations and underexplored areas, and outline potential directions for future research. Our study provides valuable insights into the evolving landscape of diffusion models, with a particular focus on the integrative and ubiquitous role of attention. 

**Abstract (ZH)**: 注意机制已成为扩散模型的基本组成部分，显著影响其在生成和判别任务中的能力。本文全面调研了注意机制在扩散模型中的应用，系统分析了其在不同模态和任务中的角色、设计模式和操作方式。我们提出了一种统一的分类体系，根据其影响的结构组件将注意相关的修改分为若干部分，提供了一种清晰的理解其功能多样性的视角。除了回顾架构创新，我们还探讨了注意机制如何在各种应用中提升性能。我们还指出了当前的局限性和未探索的领域，并概述了未来研究的潜在方向。我们的研究为理解扩散模型不断演化的景观提供了有价值的见解，特别关注注意机制的整合和普遍作用。 

---
# Uncertainty Propagation in XAI: A Comparison of Analytical and Empirical Estimators 

**Title (ZH)**: XAI中不确定性传播：分析估计器与经验估计器的比较 

**Authors**: Teodor Chiaburu, Felix Bießmann, Frank Haußer  

**Link**: [PDF](https://arxiv.org/pdf/2504.03736)  

**Abstract**: Understanding uncertainty in Explainable AI (XAI) is crucial for building trust and ensuring reliable decision-making in Machine Learning models. This paper introduces a unified framework for quantifying and interpreting Uncertainty in XAI by defining a general explanation function $e_{\theta}(x, f)$ that captures the propagation of uncertainty from key sources: perturbations in input data and model parameters. By using both analytical and empirical estimates of explanation variance, we provide a systematic means of assessing the impact uncertainty on explanations. We illustrate the approach using a first-order uncertainty propagation as the analytical estimator. In a comprehensive evaluation across heterogeneous datasets, we compare analytical and empirical estimates of uncertainty propagation and evaluate their robustness. Extending previous work on inconsistencies in explanations, our experiments identify XAI methods that do not reliably capture and propagate uncertainty. Our findings underscore the importance of uncertainty-aware explanations in high-stakes applications and offer new insights into the limitations of current XAI methods. The code for the experiments can be found in our repository at this https URL 

**Abstract (ZH)**: 理解解释性人工智能（XAI）中的不确定性对于建立信任并确保机器学习模型可靠决策至关重要。本文提出了一个统一框架来量化和解释XAI中的不确定性，通过定义一个一般解释函数 \(e_{\theta}(x, f)\) 来捕捉来自关键来源的不确定性传播：输入数据和模型参数的扰动。利用分析和经验解释方差的估计值，我们提供了一种系统方法来评估不确定性对解释的影响。我们使用一阶不确定性传播作为分析估计器来说明该方法。在跨异构数据集的全面评估中，我们将分析和经验不确定性传播估计值进行比较，并评估其稳健性。在先前关于解释不一致性的研究基础上，我们的实验识别出不能可靠地捕捉和传播不确定性的XAI方法。我们的发现强调了高风险应用中不确定性意识解释的重要性，并提供了关于当前XAI方法局限性的新见解。实验代码可以在我们的仓库中找到：this https URL。 

---
# Misaligned Roles, Misplaced Images: Structural Input Perturbations Expose Multimodal Alignment Blind Spots 

**Title (ZH)**: 角色错位，图像错置：结构输入扰动揭示多模态对齐盲点 

**Authors**: Erfan Shayegani, G M Shahariar, Sara Abdali, Lei Yu, Nael Abu-Ghazaleh, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.03735)  

**Abstract**: Multimodal Language Models (MMLMs) typically undergo post-training alignment to prevent harmful content generation. However, these alignment stages focus primarily on the assistant role, leaving the user role unaligned, and stick to a fixed input prompt structure of special tokens, leaving the model vulnerable when inputs deviate from these expectations. We introduce Role-Modality Attacks (RMA), a novel class of adversarial attacks that exploit role confusion between the user and assistant and alter the position of the image token to elicit harmful outputs. Unlike existing attacks that modify query content, RMAs manipulate the input structure without altering the query itself. We systematically evaluate these attacks across multiple Vision Language Models (VLMs) on eight distinct settings, showing that they can be composed to create stronger adversarial prompts, as also evidenced by their increased projection in the negative refusal direction in the residual stream, a property observed in prior successful attacks. Finally, for mitigation, we propose an adversarial training approach that makes the model robust against input prompt perturbations. By training the model on a range of harmful and benign prompts all perturbed with different RMA settings, it loses its sensitivity to Role Confusion and Modality Manipulation attacks and is trained to only pay attention to the content of the query in the input prompt structure, effectively reducing Attack Success Rate (ASR) while preserving the model's general utility. 

**Abstract (ZH)**: 角色模态攻击：利用用户与助手角色混淆的新型对抗攻击 

---
# Artificial Geographically Weighted Neural Network: A Novel Framework for Spatial Analysis with Geographically Weighted Layers 

**Title (ZH)**: 基于地理加权层的人工地理加权神经网络：一种空间分析的新框架 

**Authors**: Jianfei Cao, Dongchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03734)  

**Abstract**: Geographically Weighted Regression (GWR) is a widely recognized technique for modeling spatial heterogeneity. However, it is commonly assumed that the relationships between dependent and independent variables are linear. To overcome this limitation, we propose an Artificial Geographically Weighted Neural Network (AGWNN), a novel framework that integrates geographically weighted techniques with neural networks to capture complex nonlinear spatial relationships. Central to this framework is the Geographically Weighted Layer (GWL), a specialized component designed to encode spatial heterogeneity within the neural network architecture. To rigorously evaluate the performance of AGWNN, we conducted comprehensive experiments using both simulated datasets and real-world case studies. Our results demonstrate that AGWNN significantly outperforms traditional GWR and standard Artificial Neural Networks (ANNs) in terms of model fitting accuracy. Notably, AGWNN excels in modeling intricate nonlinear relationships and effectively identifies complex spatial heterogeneity patterns, offering a robust and versatile tool for advanced spatial analysis. 

**Abstract (ZH)**: 地理加权神经网络（AGWNN）：一种融合地理加权技术的新型复杂非线性空间关系建模框架 

---
# Artificial Intelligence and Deep Learning Algorithms for Epigenetic Sequence Analysis: A Review for Epigeneticists and AI Experts 

**Title (ZH)**: 人工智能与深度学习算法在表观遗传序列分析中的应用：表观遗传学家与AI专家的综述 

**Authors**: Muhammad Tahir, Mahboobeh Norouzi, Shehroz S. Khan, James R. Davie, Soichiro Yamanaka, Ahmed Ashraf  

**Link**: [PDF](https://arxiv.org/pdf/2504.03733)  

**Abstract**: Epigenetics encompasses mechanisms that can alter the expression of genes without changing the underlying genetic sequence. The epigenetic regulation of gene expression is initiated and sustained by several mechanisms such as DNA methylation, histone modifications, chromatin conformation, and non-coding RNA. The changes in gene regulation and expression can manifest in the form of various diseases and disorders such as cancer and congenital deformities. Over the last few decades, high throughput experimental approaches have been used to identify and understand epigenetic changes, but these laboratory experimental approaches and biochemical processes are time-consuming and expensive. To overcome these challenges, machine learning and artificial intelligence (AI) approaches have been extensively used for mapping epigenetic modifications to their phenotypic manifestations. In this paper we provide a narrative review of published research on AI models trained on epigenomic data to address a variety of problems such as prediction of disease markers, gene expression, enhancer promoter interaction, and chromatin states. The purpose of this review is twofold as it is addressed to both AI experts and epigeneticists. For AI researchers, we provided a taxonomy of epigenetics research problems that can benefit from an AI-based approach. For epigeneticists, given each of the above problems we provide a list of candidate AI solutions in the literature. We have also identified several gaps in the literature, research challenges, and recommendations to address these challenges. 

**Abstract (ZH)**: 表观遗传学包括不改变遗传序列即可改变基因表达的机制。表观遗传对基因表达的调控是由DNA甲基化、组蛋白修饰、染色质构象和非编码RNA等多种机制启动和维持的。基因调控和表达的变化可以表现为多种疾病和畸形，如癌症和先天性缺陷。在过去几十年中，高通量实验方法被用来识别和理解表观遗传变化，但这些实验室实验方法和生物化学过程耗时且成本高昂。为克服这些挑战，机器学习和人工智能（AI）方法被广泛用于将表观遗传修饰与其表型表型表现进行映射。在本文中，我们对基于表观基因组数据训练的AI模型进行了综述性研究，以解决各种问题，如疾病标志物预测、基因表达、增强子启动子相互作用和染色质状态。本文的目的是双重的，既面向AI专家也面向表观遗传学家。对于AI研究人员，我们提供了一种表观遗传学研究问题的分类，这些问题可以从基于AI的方法中受益。对于表观遗传学家，针对上述每个问题，我们提供了文献中候选的AI解决方案列表。我们还识别了文献中的若干空白、研究挑战，并提出了应对这些挑战的建议。 

---
# Detecting Malicious AI Agents Through Simulated Interactions 

**Title (ZH)**: 通过模拟交互检测恶意AI代理 

**Authors**: Yulu Pi, Ella Bettison, Anna Becker  

**Link**: [PDF](https://arxiv.org/pdf/2504.03726)  

**Abstract**: This study investigates malicious AI Assistants' manipulative traits and whether the behaviours of malicious AI Assistants can be detected when interacting with human-like simulated users in various decision-making contexts. We also examine how interaction depth and ability of planning influence malicious AI Assistants' manipulative strategies and effectiveness. Using a controlled experimental design, we simulate interactions between AI Assistants (both benign and deliberately malicious) and users across eight decision-making scenarios of varying complexity and stakes. Our methodology employs two state-of-the-art language models to generate interaction data and implements Intent-Aware Prompting (IAP) to detect malicious AI Assistants. The findings reveal that malicious AI Assistants employ domain-specific persona-tailored manipulation strategies, exploiting simulated users' vulnerabilities and emotional triggers. In particular, simulated users demonstrate resistance to manipulation initially, but become increasingly vulnerable to malicious AI Assistants as the depth of the interaction increases, highlighting the significant risks associated with extended engagement with potentially manipulative systems. IAP detection methods achieve high precision with zero false positives but struggle to detect many malicious AI Assistants, resulting in high false negative rates. These findings underscore critical risks in human-AI interactions and highlight the need for robust, context-sensitive safeguards against manipulative AI behaviour in increasingly autonomous decision-support systems. 

**Abstract (ZH)**: 本研究探讨了恶意AI助手的操控特质，并分析在与人类模拟用户交互时，尤其是在各种决策情境下，是否能检测到恶意AI助手的行为。我们还探讨了交互深度和计划能力如何影响恶意AI助手的操控策略及其有效性。通过受控实验设计，我们在八个复杂程度和风险等级不同的决策场景中模拟了良性与恶意AI助手与用户之间的交互。我们的方法使用了两种最先进的语言模型生成交互数据，并采用了意图感知提示（IAP）来检测恶意AI助手。研究发现，恶意AI助手采用领域特定的人格定制操控策略，利用模拟用户的情感触发点和弱点进行操控。特别是，模拟用户最初对操控具有抵抗力，但随着交互深度的增加，他们变得越来越容易受到恶意AI助手的影响，突显了与潜在操控性系统进行长期交互的重大风险。IAP检测方法在没有假阳性的情况下实现了高精度，但难以检测许多恶意AI助手，导致高假阴性率。这些发现强调了人类与AI交互中的关键风险，并突出了在日益自主的支持决策系统中against操控性AI行为需要更加 robust且情境敏感的安全措施。 

---
# A Hybrid Reinforcement Learning Framework for Hard Latency Constrained Resource Scheduling 

**Title (ZH)**: 一种满足严苛延迟约束资源调度的混合强化学习框架 

**Authors**: Luyuan Zhang, An Liu, Kexuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03721)  

**Abstract**: In the forthcoming 6G era, extend reality (XR) has been regarded as an emerging application for ultra-reliable and low latency communications (URLLC) with new traffic characteristics and more stringent requirements. In addition to the quasi-periodical traffic in XR, burst traffic with both large frame size and random arrivals in some real world low latency communication scenarios has become the leading cause of network congestion or even collapse, and there still lacks an efficient algorithm for the resource scheduling problem under burst traffic with hard latency constraints. We propose a novel hybrid reinforcement learning framework for resource scheduling with hard latency constraints (HRL-RSHLC), which reuses polices from both old policies learned under other similar environments and domain-knowledge-based (DK) policies constructed using expert knowledge to improve the performance. The joint optimization of the policy reuse probabilities and new policy is formulated as an Markov Decision Problem (MDP), which maximizes the hard-latency constrained effective throughput (HLC-ET) of users. We prove that the proposed HRL-RSHLC can converge to KKT points with an arbitrary initial point. Simulations show that HRL-RSHLC can achieve superior performance with faster convergence speed compared to baseline algorithms. 

**Abstract (ZH)**: 在未来的6G时代，增强现实(XR)被视为超可靠低延迟通信(URLLC)的一种新兴应用，具有新的业务特性和更严格的要求。除了XR中的准周期性流量外，在某些实际低延迟通信场景中，具有较大帧尺寸和随机到达时间的突发流量已成为网络拥塞甚至崩溃的主要原因，仍缺乏在具有严格延迟约束的突发流量下的高效资源调度算法。我们提出了一种新型混合强化学习框架（HRL-RSHLC）进行具有严格延迟约束的资源调度，该框架结合了基于旧环境学得策略和基于专家知识构建的域知识策略，以提高性能。将策略重用概率的联合优化与新策略的优化形式化为马尔可夫决策过程（MDP），以最大化用户的有效吞吐量（HLC-ET）。证明了所提出的HRL-RSHLC可以从任意初始点收敛到KKT点。仿真实验表明，与基线算法相比，HRL-RSHLC在收敛速度方面表现出更优的性能。 

---
# Towards Symmetric Low-Rank Adapters 

**Title (ZH)**: Towards 对称低秩适配器 

**Authors**: Tales Panoutsos, Rodrygo L. T. Santos, Flavio Figueiredo  

**Link**: [PDF](https://arxiv.org/pdf/2504.03719)  

**Abstract**: \newcommand{\mathds}[1]{\text{\usefont{U}{dsrom}{m}{n}#1}}
In this paper, we introduce Symmetric Low-Rank Adapters, an optimized variant of LoRA with even fewer weights. This method utilizes Low-Rank Symmetric Weight Matrices to learn downstream tasks more efficiently. Traditional LoRA accumulates fine-tuning weights with the original pre-trained weights via a Singular Value Decomposition (SVD) like approach, i.e., model weights are fine-tuned via updates of the form $BA$ (where $B \in \mathbb{R}^{n\times r}$, $A \in \mathbb{R}^{r\times n}$, and $r$ is the rank of the merged weight matrix). In contrast, our approach, named SymLoRA, represents fine-tuning weights as a Spectral Decomposition, i.e., $Q \, diag(\Lambda)\, Q^T$, where $Q \in \mathbb{R}^{n\times r}$ and $\Lambda \in \mathbb{R}^r$. SymLoRA requires approximately half of the finetuning weights. Here, we show that this approach has negligible losses in downstream efficacy. 

**Abstract (ZH)**: 基于低秩对称矩阵的对称低秩适配器：一种具有更少参数的优化变体 

---
# Task-Aware Parameter-Efficient Fine-Tuning of Large Pre-Trained Models at the Edge 

**Title (ZH)**: 边缘设备上的任务导向参数高效微调大型预训练模型 

**Authors**: Senkang Hu, Yanan Ma, Yihang Tao, Zhengru Fang, Zihan Fang, Yiqin Deng, Sam Kwong, Yuguang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03718)  

**Abstract**: Large language models (LLMs) have achieved remarkable success in various tasks, such as decision-making, reasoning, and question answering. They have been widely used in edge devices. However, fine-tuning LLMs to specific tasks at the edge is challenging due to the high computational cost and the limited storage and energy resources at the edge. To address this issue, we propose TaskEdge, a task-aware parameter-efficient fine-tuning framework at the edge, which allocates the most effective parameters to the target task and only updates the task-specific parameters. Specifically, we first design a parameter importance calculation criterion that incorporates both weights and input activations into the computation of weight importance. Then, we propose a model-agnostic task-specific parameter allocation algorithm to ensure that task-specific parameters are distributed evenly across the model, rather than being concentrated in specific regions. In doing so, TaskEdge can significantly reduce the computational cost and memory usage while maintaining performance on the target downstream tasks by updating less than 0.1\% of the parameters. In addition, TaskEdge can be easily integrated with structured sparsity to enable acceleration by NVIDIA's specialized sparse tensor cores, and it can be seamlessly integrated with LoRA to enable efficient sparse low-rank adaptation. Extensive experiments on various tasks demonstrate the effectiveness of TaskEdge. 

**Abstract (ZH)**: TaskEdge：一种面向任务的边缘参数高效微调框架 

---
# RaanA: A Fast, Flexible, and Data-Efficient Post-Training Quantization Algorithm 

**Title (ZH)**: RaanA: 一种快速、灵活且数据高效的后训练量化算法 

**Authors**: Yongyi Yang, Jianyang Gao, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03717)  

**Abstract**: Post-training Quantization (PTQ) has become a widely used technique for improving inference efficiency of large language models (LLMs). However, existing PTQ methods generally suffer from crucial limitations such as heavy calibration data requirements and inflexible choice of target number of bits. In this paper, we propose RaanA, a unified PTQ framework that overcomes these challenges by introducing two novel components: 1) RaBitQ-H, a variant of a randomized vector quantization method RaBitQ, designed for fast, accurate, and highly efficient quantization; and 2) AllocateBits, an algorithm that optimally allocates bit-widths across layers based on their quantization sensitivity. RaanA achieves competitive performance with state-of-the-art quantization methods while being extremely fast, requiring minimal calibration data, and enabling flexible bit allocation. Extensive experiments demonstrate RaanA's efficacy in balancing efficiency and accuracy. The code is publicly available at this https URL . 

**Abstract (ZH)**: Post-training 量化 (PTQ) 已成为提高大型语言模型 (LLM) 推断效率的一种广泛使用的技术。然而，现有的 PTQ 方法通常存在关键限制，如对校准数据的高要求和目标位数选择的灵活性不足。本文提出了一种名为 RaanA 的统一 PTQ 框架，通过引入两个新型组件来克服这些挑战：1) RaBitQ-H，一种基于随机化向量量化方法 RaBitQ 的变体，旨在实现快速、准确且高效的量化；2) AllocateBits，一种根据各层的量化敏感性优化分配位宽的算法。RaanA 在保持与最先进的量化方法相当的性能的同时，具有极高的速度、最小的校准数据要求，并支持灵活的位宽分配。广泛的实验验证了 RaanA 在效率和精度之间取得的平衡效果。代码可在以下网址获取：this https URL。 

---
# Ethical AI on the Waitlist: Group Fairness Evaluation of LLM-Aided Organ Allocation 

**Title (ZH)**: 等待列表中的伦理AI：LLM辅助器官分配的团体公平性评价 

**Authors**: Hannah Murray, Brian Hyeongseok Kim, Isabelle Lee, Jason Byun, Dani Yogatama, Evi Micha  

**Link**: [PDF](https://arxiv.org/pdf/2504.03716)  

**Abstract**: Large Language Models (LLMs) are becoming ubiquitous, promising automation even in high-stakes scenarios. However, existing evaluation methods often fall short -- benchmarks saturate, accuracy-based metrics are overly simplistic, and many inherently ambiguous problems lack a clear ground truth. Given these limitations, evaluating fairness becomes complex. To address this, we reframe fairness evaluation using Borda scores, a method from voting theory, as a nuanced yet interpretable metric for measuring fairness. Using organ allocation as a case study, we introduce two tasks: (1) Choose-One and (2) Rank-All. In Choose-One, LLMs select a single candidate for a kidney, and we assess fairness across demographics using proportional parity. In Rank-All, LLMs rank all candidates for a kidney, reflecting real-world allocation processes. Since traditional fairness metrics do not account for ranking, we propose a novel application of Borda scoring to capture biases. Our findings highlight the potential of voting-based metrics to provide a richer, more multifaceted evaluation of LLM fairness. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在变得无处不在，在高风险场景中也承诺实现自动化。然而，现有的评估方法往往不尽如人意——评估基准趋于饱和，基于准确性的指标过于简化，而且许多本质上具有模糊性的问题缺乏明确的基准答案。鉴于这些限制，评估公平性变得更复杂。为解决这一问题，我们重新定义公平性评估，使用选举理论中的Borda分数作为衡量公平性的细致且可解释的度量标准。以器官分配为例，我们提出两项任务：（1）单项选择和（2）全部排序。在单项选择任务中，LLMs为肾脏选择一个单一候选人，并使用比例平等来评估不同人口统计学群体的公平性。在全部排序任务中，LLMs对所有候选人进行排名，反映现实中的分配过程。由于传统的公平性指标未能考虑排名，我们提出将Borda评分应用到新领域以捕捉偏差。我们的研究结果凸显了基于选举的指标在提供LLM公平性更加丰富和多维评估方面的潜力。 

---
# Multi-Objective Quality-Diversity in Unstructured and Unbounded Spaces 

**Title (ZH)**: 无结构和无边界空间中的多目标质量多样性 

**Authors**: Hannah Janmohamed, Antoine Cully  

**Link**: [PDF](https://arxiv.org/pdf/2504.03715)  

**Abstract**: Quality-Diversity algorithms are powerful tools for discovering diverse, high-performing solutions. Recently, Multi-Objective Quality-Diversity (MOQD) extends QD to problems with several objectives while preserving solution diversity. MOQD has shown promise in fields such as robotics and materials science, where finding trade-offs between competing objectives like energy efficiency and speed, or material properties is essential. However, existing methods in MOQD rely on tessellating the feature space into a grid structure, which prevents their application in domains where feature spaces are unknown or must be learned, such as complex biological systems or latent exploration tasks. In this work, we introduce Multi-Objective Unstructured Repertoire for Quality-Diversity (MOUR-QD), a MOQD algorithm designed for unstructured and unbounded feature spaces. We evaluate MOUR-QD on five robotic tasks. Importantly, we show that our method excels in tasks where features must be learned, paving the way for applying MOQD to unsupervised domains. We also demonstrate that MOUR-QD is advantageous in domains with unbounded feature spaces, outperforming existing grid-based methods. Finally, we demonstrate that MOUR-QD is competitive with established MOQD methods on existing MOQD tasks and achieves double the MOQD-score in some environments. MOUR-QD opens up new opportunities for MOQD in domains like protein design and image generation. 

**Abstract (ZH)**: Multi-Objective Unstructured Repertoire for Quality-Diversity 

---
# Breach in the Shield: Unveiling the Vulnerabilities of Large Language Models 

**Title (ZH)**: 屏蔽失效：大型语言模型的漏洞揭示 

**Authors**: Runpeng Dai, Run Yang, Fan Zhou, Hongtu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03714)  

**Abstract**: Large Language Models (LLMs) and Vision-Language Models (VLMs) have become essential to general artificial intelligence, exhibiting remarkable capabilities in task understanding and problem-solving. However, the real-world reliability of these models critically depends on their stability, which remains an underexplored area. Despite their widespread use, rigorous studies examining the stability of LLMs under various perturbations are still lacking. In this paper, we address this gap by proposing a novel stability measure for LLMs, inspired by statistical methods rooted in information geometry. Our measure possesses desirable invariance properties, making it well-suited for analyzing model sensitivity to both parameter and input perturbations. To assess the effectiveness of our approach, we conduct extensive experiments on models ranging in size from 1.5B to 13B parameters. Our results demonstrate the utility of our measure in identifying salient parameters and detecting vulnerable regions in input images or critical dimensions in token embeddings. Furthermore, leveraging our stability framework, we enhance model robustness during model merging, leading to improved performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）和视觉语言模型（VLMs）已成为通用人工智能的关键组成部分，展示了在任务理解和问题解决方面的杰出能力。然而，这些模型在现实世界中的可靠性关键依赖于其稳定性，这一领域仍存在未探索的领域。尽管这些模型被广泛使用，但对各种扰动下LLMs稳定性的严格研究仍然不足。在本文中，我们通过提出一种受信息几何统计方法启发的新颖稳定性度量来填补这一空白。该度量具有期望的不变性特性，使其适用于分析模型对参数和输入扰动的敏感性。为了评估我们方法的有效性，我们在从1.5B到13B参数不等的模型上进行了广泛实验。我们的结果表明了该度量在识别关键参数和检测输入图像中的脆弱区域或令牌嵌入中的关键维度方面的实用性。此外，利用我们的稳定性框架，我们在模型合并中增强了模型的稳健性，从而提高了性能。 

---
# RLDBF: Enhancing LLMs Via Reinforcement Learning With DataBase FeedBack 

**Title (ZH)**: RLDBF: 通过数据库反馈增强LLMs的强化学习方法 

**Authors**: Weichen Dai, Zijie Dai, Zhijie Huang, Yixuan Pan, Xinhe Li, Xi Li, Yi Zhou, Ji Qi, Wu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03713)  

**Abstract**: While current large language models (LLMs) demonstrate remarkable linguistic capabilities through training on massive unstructured text corpora, they remain inadequate in leveraging structured scientific data (e.g., chemical molecular properties in databases) that encapsulate centuries of accumulated scientific expertise. These structured datasets hold strategic significance for advancing AI for Science yet current approaches merely treat them as auxiliary supplements to unstructured text. This study pioneers a systematic investigation into enhancing LLMs with structured scientific data, using chemical molecular science as a testbed. We investigate the impact of incorporating molecular property data on LLM across distinct training phases, including continual pre-training, supervised fine-tuning, and reinforcement learning. Notably, to address the inherent limitation of numerical insensitivity in large models, we propose an innovative methodology termed "Reinforcement Learning with Database Feedback" (RLDBF). Experimental evaluations demonstrate the efficacy of the proposed approach, with the model exhibiting remarkable generalization capabilities on previously unseen data and other chemical tasks. The results substantiate the potential of our method in advancing the field of structured scientific data processing within LLMs. 

**Abstract (ZH)**: 尽管当前的大规模语言模型（LLMs）通过训练大量未结构化文本 corpora 展示出卓越的语言能力，但在利用蕴含百年科学经验的结构化科学数据（例如数据库中的化学分子性质）方面仍显不足。这些结构化数据对推动人工智能科学具有战略意义，但当前的方法仅仅将它们视为未结构化文本的辅助补充。本研究首次系统性地探讨了结合结构化科学数据以增强 LLMs 的可能性，以化学分子科学作为试验场。我们研究了将分子性质数据纳入 LLM 在不同训练阶段的影响，包括持续预训练、监督微调和强化学习。值得注意的是，为解决大模型固有的数值敏感性不足问题，我们提出了一种名为“数据库反馈强化学习”（RLDBF）的创新方法。实验评估表明，所提出的方法具有显著效果，模型在未见过的数据和其他化学任务上表现出卓越的一般化能力。研究结果证明了该方法在促进 LLMs 中结构化科学数据处理领域的潜力。 

---
# Scalable heliostat surface predictions from focal spots: Sim-to-Real transfer of inverse Deep Learning Raytracing 

**Title (ZH)**: 基于聚光点的可扩展抛物镜表面预测：从仿真到现实的逆深度学习光线追踪转移 

**Authors**: Jan Lewen, Max Pargmann, Jenia Jitsev, Mehdi Cherti, Robert Pitz-Paal, Daniel Maldonado Quinto  

**Link**: [PDF](https://arxiv.org/pdf/2504.03712)  

**Abstract**: Concentrating Solar Power (CSP) plants are a key technology in the transition toward sustainable energy. A critical factor for their safe and efficient operation is the distribution of concentrated solar flux on the receiver. However, flux distributions from individual heliostats are sensitive to surface imperfections. Measuring these surfaces across many heliostats remains impractical in real-world deployments. As a result, control systems often assume idealized heliostat surfaces, leading to suboptimal performance and potential safety risks. To address this, inverse Deep Learning Raytracing (iDLR) has been introduced as a novel method for inferring heliostat surface profiles from target images recorded during standard calibration procedures. In this work, we present the first successful Sim-to-Real transfer of iDLR, enabling accurate surface predictions directly from real-world target images. We evaluate our method on 63 heliostats under real operational conditions. iDLR surface predictions achieve a median mean absolute error (MAE) of 0.17 mm and show good agreement with deflectometry ground truth in 84% of cases. When used in raytracing simulations, it enables flux density predictions with a mean accuracy of 90% compared to deflectometry over our dataset, and outperforms the commonly used ideal heliostat surface assumption by 26%. We tested this approach in a challenging double-extrapolation scenario-involving unseen sun positions and receiver projection-and found that iDLR maintains high predictive accuracy, highlighting its generalization capabilities. Our results demonstrate that iDLR is a scalable, automated, and cost-effective solution for integrating realistic heliostat surface models into digital twins. This opens the door to improved flux control, more precise performance modeling, and ultimately, enhanced efficiency and safety in future CSP plants. 

**Abstract (ZH)**: 基于逆深度学习光线追踪的实到虚转移在集中太阳能动力系统中的表面预测研究 

---
# SAFE: Self-Adjustment Federated Learning Framework for Remote Sensing Collaborative Perception 

**Title (ZH)**: SAFE：自调整联邦学习框架用于遥感协同感知 

**Authors**: Xiaohe Li, Haohua Wu, Jiahao Li, Zide Fan, Kaixin Zhang, Xinming Li, Yunping Ge, Xinyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.03700)  

**Abstract**: The rapid increase in remote sensing satellites has led to the emergence of distributed space-based observation systems. However, existing distributed remote sensing models often rely on centralized training, resulting in data leakage, communication overhead, and reduced accuracy due to data distribution discrepancies across platforms. To address these challenges, we propose the \textit{Self-Adjustment FEderated Learning} (SAFE) framework, which innovatively leverages federated learning to enhance collaborative sensing in remote sensing scenarios. SAFE introduces four key strategies: (1) \textit{Class Rectification Optimization}, which autonomously addresses class imbalance under unknown local and global distributions. (2) \textit{Feature Alignment Update}, which mitigates Non-IID data issues via locally controlled EMA updates. (3) \textit{Dual-Factor Modulation Rheostat}, which dynamically balances optimization effects during training. (4) \textit{Adaptive Context Enhancement}, which is designed to improve model performance by dynamically refining foreground regions, ensuring computational efficiency with accuracy improvement across distributed satellites. Experiments on real-world image classification and object segmentation datasets validate the effectiveness and reliability of the SAFE framework in complex remote sensing scenarios. 

**Abstract (ZH)**: 遥感卫星的快速发展催生了分布式空间观测系统。然而，现有的分布式遥感模型通常依赖于集中训练，导致数据泄露、通信开销增大以及由于平台间数据分布差异而降低准确性。为应对这些挑战，我们提出了一种名为“Self-Adjustment Federated Learning”（SAFE）的框架，该框架创新性地利用联邦学习来增强遥感场景中的协同感知。SAFE引入了四个关键策略：(1) 类别校正优化，自主解决未知局部和全局分布下的类别不平衡问题。(2) 特征对齐更新，通过局部控制的指数移动平均（EMA）更新缓解非独立同分布（Non-IID）数据问题。(3) 双因子调节电位器，动态平衡训练期间的优化效果。(4) 适应回溯增强，旨在通过动态细化前景区域来提升模型性能，并确保在分布式卫星中提高计算效率和准确性。实世界图像分类和对象分割数据集上的实验验证了SAFE框架在复杂遥感场景中的有效性和可靠性。 

---
# Are Anxiety Detection Models Generalizable? A Cross-Activity and Cross-Population Study Using Wearables 

**Title (ZH)**: 焦虑检测模型具有普遍适用性吗？可穿戴设备在跨活动和跨人群研究中的应用 

**Authors**: Nilesh Kumar Sahu, Snehil Gupta, Haroon R Lone  

**Link**: [PDF](https://arxiv.org/pdf/2504.03695)  

**Abstract**: Anxiety-provoking activities, such as public speaking, can trigger heightened anxiety responses in individuals with anxiety disorders. Recent research suggests that physiological signals, including electrocardiogram (ECG) and electrodermal activity (EDA), collected via wearable devices, can be used to detect anxiety in such contexts through machine learning models. However, the generalizability of these anxiety prediction models across different activities and diverse populations remains underexplored-an essential step for assessing model bias and fostering user trust in broader applications. To address this gap, we conducted a study with 111 participants who engaged in three anxiety-provoking activities. Utilizing both our collected dataset and two well-known publicly available datasets, we evaluated the generalizability of anxiety detection models within participants (for both same-activity and cross-activity scenarios) and across participants (within-activity and cross-activity). In total, we trained and tested more than 3348 anxiety detection models (using six classifiers, 31 feature sets, and 18 train-test configurations). Our results indicate that three key metrics-AUROC, recall for anxious states, and recall for non-anxious states-were slightly above the baseline score of 0.5. The best AUROC scores ranged from 0.62 to 0.73, with recall for the anxious class spanning 35.19% to 74.3%. Interestingly, model performance (as measured by AUROC) remained relatively stable across different activities and participant groups, though recall for the anxious class did exhibit some variation. 

**Abstract (ZH)**: 焦虑诱发活动（如公开发言）可引起焦虑障碍患者的焦虑反应加剧。研究表明，通过穿戴设备收集的心电图（ECG）和电导率活动（EDA）等生理信号，可以通过机器学习模型在这些情境中检测焦虑。然而，这些焦虑预测模型在不同活动和多样化人群中的泛化能力尚未充分探索——这是评估模型偏差和促进更广泛应用用户信任的关键步骤。为弥补这一不足，我们进行了一个包括111名参与三种焦虑诱发活动的研究。利用我们收集的数据集及两个广为人知的公开数据集，我们评估了焦虑检测模型在参与者内部（同活动和跨活动情景）和参与者之间（同活动和跨活动情景）的泛化能力。总共，我们训练并测试了超过3348个焦虑检测模型（使用六种分类器、31个特征集和18种训练-测试配置）。结果显示，三个关键指标——AUCROC、焦虑状态的召回率和非焦虑状态的召回率——略高于基线分数0.5。最佳AUCROC得分范围从0.62到0.73，焦虑类别的召回率范围为35.19%到74.3%。有趣的是，模型性能（通过AUCROC衡量）在不同活动和参与者群体中相对稳定，尽管焦虑类别的召回率存在一些差异。 

---
# Learning to Interfere in Non-Orthogonal Multiple-Access Joint Source-Channel Coding 

**Title (ZH)**: 学习在非正交多访问联合源-信道编码中进行干扰 

**Authors**: Selim F. Yilmaz, Can Karamanli, Deniz Gunduz  

**Link**: [PDF](https://arxiv.org/pdf/2504.03690)  

**Abstract**: We consider multiple transmitters aiming to communicate their source signals (e.g., images) over a multiple access channel (MAC). Conventional communication systems minimize interference by orthogonally allocating resources (time and/or bandwidth) among users, which limits their capacity. We introduce a machine learning (ML)-aided wireless image transmission method that merges compression and channel coding using a multi-view autoencoder, which allows the transmitters to use all the available channel resources simultaneously, resulting in a non-orthogonal multiple access (NOMA) scheme. The receiver must recover all the images from the received superposed signal, while also associating each image with its transmitter. Traditional ML models deal with individual samples, whereas our model allows signals from different users to interfere in order to leverage gains from NOMA under limited bandwidth and power constraints. We introduce a progressive fine-tuning algorithm that doubles the number of users at each iteration, maintaining initial performance with orthogonalized user-specific projections, which is then improved through fine-tuning steps. Remarkably, our method scales up to 16 users and beyond, with only a 0.6% increase in the number of trainable parameters compared to a single-user model, significantly enhancing recovered image quality and outperforming existing NOMA-based methods over a wide range of datasets, metrics, and channel conditions. Our approach paves the way for more efficient and robust multi-user communication systems, leveraging innovative ML components and strategies. 

**Abstract (ZH)**: 基于机器学习的多视角自编码器辅助无线图像传输方法 

---
# CLCR: Contrastive Learning-based Constraint Reordering for Efficient MILP Solving 

**Title (ZH)**: CLCR：基于对比学习的约束重排序以实现高效的混合整数规划求解 

**Authors**: Shuli Zeng, Mengjie Zhou, Sijia Zhang, Yixiang Hu, Feng Wu, Xiang-Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.03688)  

**Abstract**: Constraint ordering plays a critical role in the efficiency of Mixed-Integer Linear Programming (MILP) solvers, particularly for large-scale problems where poorly ordered constraints trigger increased LP iterations and suboptimal search trajectories. This paper introduces CLCR (Contrastive Learning-based Constraint Reordering), a novel framework that systematically optimizes constraint ordering to accelerate MILP solving. CLCR first clusters constraints based on their structural patterns and then employs contrastive learning with a pointer network to optimize their sequence, preserving problem equivalence while improving solver efficiency. Experiments on benchmarks show CLCR reduces solving time by 30% and LP iterations by 25% on average, without sacrificing solution accuracy. This work demonstrates the potential of data-driven constraint ordering to enhance optimization models, offering a new paradigm for bridging mathematical programming with machine learning. 

**Abstract (ZH)**: 基于对比学习的约束重排序（CLCR）在混合整数线性规划中的应用 

---
# Process Optimization and Deployment for Sensor-Based Human Activity Recognition Based on Deep Learning 

**Title (ZH)**: 基于深度学习的传感器驱动的人类活动识别过程优化与部署 

**Authors**: Hanyu Liu, Ying Yu, Hang Xiao, Siyao Li, Xuze Li, Jiarui Li, Haotian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03687)  

**Abstract**: Sensor-based human activity recognition is a key technology for many human-centered intelligent applications. However, this research is still in its infancy and faces many unresolved challenges. To address these, we propose a comprehensive optimization process approach centered on multi-attention interaction. We first utilize unsupervised statistical feature-guided diffusion models for highly adaptive data enhancement, and introduce a novel network architecture-Multi-branch Spatiotemporal Interaction Network, which uses multi-branch features at different levels to effectively Sequential ), which uses multi-branch features at different levels to effectively Sequential spatio-temporal interaction to enhance the ability to mine advanced latent features. In addition, we adopt a multi-loss function fusion strategy in the training phase to dynamically adjust the fusion weights between batches to optimize the training results. Finally, we also conducted actual deployment on embedded devices to extensively test the practical feasibility of the proposed method in existing work. We conduct extensive testing on three public datasets, including ablation studies, comparisons of related work, and embedded deployments. 

**Abstract (ZH)**: 基于传感器的人类活动识别是许多以人类为中心的智能应用的关键技术。然而，这一研究领域仍处于初级阶段，面临许多未解决的挑战。为此，我们提出了一种以多关注交互为中心的综合优化过程。首先，我们利用无监督的统计特征引导扩散模型进行高度适应的数据增强，并引入了一种新的网络架构——多分支时空交互网络，该网络在不同层次使用多分支特征有效地进行时空序列交互，以增强挖掘高级潜在特征的能力。此外，在训练阶段，我们采用了多损失函数融合策略，动态调整批次之间的融合权重以优化训练结果。最后，我们在嵌入式设备上进行了实际部署，广泛测试了所提出方法在现有工作中应用的实用性。我们对三个公开数据集进行了广泛的测试，包括消融研究、相关工作中方法的比较以及嵌入式部署。 

---
# Revisiting Outage for Edge Inference Systems 

**Title (ZH)**: 重新审视边缘推理系统的中断问题 

**Authors**: Zhanwei Wang, Qunsong Zeng, Haotian Zheng, Kaibin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03686)  

**Abstract**: One of the key missions of sixth-generation (6G) mobile networks is to deploy large-scale artificial intelligence (AI) models at the network edge to provide remote-inference services for edge devices. The resultant platform, known as edge inference, will support a wide range of Internet-of-Things applications, such as autonomous driving, industrial automation, and augmented reality. Given the mission-critical and time-sensitive nature of these tasks, it is essential to design edge inference systems that are both reliable and capable of meeting stringent end-to-end (E2E) latency constraints. Existing studies, which primarily focus on communication reliability as characterized by channel outage probability, may fail to guarantee E2E performance, specifically in terms of E2E inference accuracy and latency. To address this limitation, we propose a theoretical framework that introduces and mathematically characterizes the inference outage (InfOut) probability, which quantifies the likelihood that the E2E inference accuracy falls below a target threshold. Under an E2E latency constraint, this framework establishes a fundamental tradeoff between communication overhead (i.e., uploading more sensor observations) and inference reliability as quantified by the InfOut probability. To find a tractable way to optimize this tradeoff, we derive accurate surrogate functions for InfOut probability by applying a Gaussian approximation to the distribution of the received discriminant gain. Experimental results demonstrate the superiority of the proposed design over conventional communication-centric approaches in terms of E2E inference reliability. 

**Abstract (ZH)**: 第六代（6G）移动网络的关键使命之一是在网络边缘部署大规模人工智能模型，以提供边缘设备的远程推断服务。由此形成的边缘推断平台将支持自动驾驶、工业自动化和增强现实等广泛的应用。鉴于这些任务的关键性和时间敏感性，设计既可靠又能满足严格端到端（E2E）延迟约束的边缘推断系统至关重要。现有研究主要关注由信道 outage 概率表征的通信可靠性，可能无法保证E2E性能，特别是E2E推断准确性和延迟。为解决这一局限性，我们提出了一种理论框架，引入并从数学上刻画了推断 outage（InfOut）概率，以量化E2E推断准确度低于目标阈值的可能性。在E2E延迟约束下，该框架建立了通信开销（即上传更多传感器观察数据）与由InfOut概率量化推断可靠性之间的基本权衡。为了找到优化这一权衡的可行方法，我们通过高斯近似推导了InfOut概率的准确替代函数。实验结果表明，所提出的设计在E2E推断可靠性方面优于传统的以通信为中心的方法。 

---
# Intelligent Resource Allocation Optimization for Cloud Computing via Machine Learning 

**Title (ZH)**: 基于机器学习的云computing智能资源分配优化 

**Authors**: Yuqing Wang, Xiao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03682)  

**Abstract**: With the rapid expansion of cloud computing applications, optimizing resource allocation has become crucial for improving system performance and cost efficiency. This paper proposes an intelligent resource allocation algorithm that leverages deep learning (LSTM) for demand prediction and reinforcement learning (DQN) for dynamic scheduling. By accurately forecasting computing resource demands and enabling real-time adjustments, the proposed system enhances resource utilization by 32.5%, reduces average response time by 43.3%, and lowers operational costs by 26.6%. Experimental results in a production cloud environment confirm that the method significantly improves efficiency while maintaining high service quality. This study provides a scalable and effective solution for intelligent cloud resource management, offering valuable insights for future cloud optimization strategies. 

**Abstract (ZH)**: 随着云 computing 应用的迅速扩展，优化资源分配已成为提升系统性能和成本效率的关键。本文提出了一种智能资源分配算法，该算法利用深度学习（LSTM）进行需求预测，并利用强化学习（DQN）进行动态调度。通过准确预测计算资源需求并实现实时调整，所提出系统将资源利用率提高32.5%，平均响应时间缩短43.3%，运营成本降低26.6%。在生产云环境中进行的实验结果证实，该方法显著提高了效率，同时保持了高质量的服务。本文为智能云资源管理提供了可扩展和有效的解决方案，并为未来的云优化策略提供了宝贵见解。 

---
# HiAER-Spike: Hardware-Software Co-Design for Large-Scale Reconfigurable Event-Driven Neuromorphic Computing 

**Title (ZH)**: HiAER-Spike: 硬件-软件协同设计的大规模可重构事件驱动神经形态计算 

**Authors**: Gwenevere Frank, Gopabandhu Hota, Keli Wang, Abhinav Uppal, Omowuyi Olajide, Kenneth Yoshimoto, Leif Gibb, Qingbo Wang, Johannes Leugering, Stephen Deiss, Gert Cauwenberghs  

**Link**: [PDF](https://arxiv.org/pdf/2504.03671)  

**Abstract**: In this work, we present HiAER-Spike, a modular, reconfigurable, event-driven neuromorphic computing platform designed to execute large spiking neural networks with up to 160 million neurons and 40 billion synapses - roughly twice the neurons of a mouse brain at faster-than real-time. This system, which is currently under construction at the UC San Diego Supercomputing Center, comprises a co-designed hard- and software stack that is optimized for run-time massively parallel processing and hierarchical address-event routing (HiAER) of spikes while promoting memory-efficient network storage and execution. Our architecture efficiently handles both sparse connectivity and sparse activity for robust and low-latency event-driven inference for both edge and cloud computing. A Python programming interface to HiAER-Spike, agnostic to hardware-level detail, shields the user from complexity in the configuration and execution of general spiking neural networks with virtually no constraints in topology. The system is made easily available over a web portal for use by the wider community. In the following we provide an overview of the hard- and software stack, explain the underlying design principles, demonstrate some of the system's capabilities and solicit feedback from the broader neuromorphic community. 

**Abstract (ZH)**: HiAER-Spike：一种模块化、可重构、事件驱动的神经形态计算平台 

---
# Self-Learning-Based Optimization for Free-form Pipe Routing in Aeroengine with Dynamic Design Environment 

**Title (ZH)**: 自学习优化在动态设计环境中的自形管道 routing 优化在航空发动机中 

**Authors**: Caicheng Wang, Zili Wang, Shuyou Zhang, Yongzhe Xiang, Zheyi Li, Jianrong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.03669)  

**Abstract**: Pipe routing is a highly complex, time-consuming, and no-deterministic polynomial-time hard (NP-hard) problem in aeroengine design. Despite extensive research efforts in optimizing constant-curvature pipe routing, the growing demand for free-form pipes poses new challenges. Dynamic design environments and fuzzy layout rules further impact the optimization performance and efficiency. To tackle these challenges, this study proposes a self-learning-based method (SLPR) for optimizing free-form pipe routing in aeroengines. The SLPR is based on the proximal policy optimization (PPO) algorithm and integrates a unified rule modeling framework for efficient obstacle detection and fuzzy rule modeling in continuous space. Additionally, a potential energy table is constructed to enable rapid queries of layout tendencies and interference. The agent within SLPR iteratively refines pipe routing and accumulates the design knowledge through interaction with the environment. Once the design environment shifts, the agent can swiftly adapt by fine-tuning network parameters. Comparative tests reveal that SLPR ensures smooth pipe routing through cubic non-uniform B-spline (NURBS) curves, avoiding redundant pipe segments found in constant-curvature pipe routing. Results in both static and dynamic design environments demonstrate that SLPR outperforms three representative baselines in terms of the pipe length reduction, the adherence to layout rules, the path complexity, and the computational efficiency. Furthermore, tests in dynamic environments indicate that SLPR eliminates labor-intensive searches from scratch and even yields superior solutions compared to the retrained model. These results highlight the practical value of SLPR for real-world pipe routing, meeting lightweight, precision, and sustainability requirements of the modern aeroengine design. 

**Abstract (ZH)**: 基于自学习的方法（SLPR）优化航空发动机自由形管路由问题 

---
# LLM & HPC:Benchmarking DeepSeek's Performance in High-Performance Computing Tasks 

**Title (ZH)**: LLM & HPC：评估DeepSeek在高性能计算任务中的性能 

**Authors**: Noujoud Nader, Patrick Diehl, Steve Brandt, Hartmut Kaiser  

**Link**: [PDF](https://arxiv.org/pdf/2504.03665)  

**Abstract**: Large Language Models (LLMs), such as GPT-4 and DeepSeek, have been applied to a wide range of domains in software engineering. However, their potential in the context of High-Performance Computing (HPC) much remains to be explored. This paper evaluates how well DeepSeek, a recent LLM, performs in generating a set of HPC benchmark codes: a conjugate gradient solver, the parallel heat equation, parallel matrix multiplication, DGEMM, and the STREAM triad operation. We analyze DeepSeek's code generation capabilities for traditional HPC languages like Cpp, Fortran, Julia and Python. The evaluation includes testing for code correctness, performance, and scaling across different configurations and matrix sizes. We also provide a detailed comparison between DeepSeek and another widely used tool: GPT-4. Our results demonstrate that while DeepSeek generates functional code for HPC tasks, it lags behind GPT-4, in terms of scalability and execution efficiency of the generated code. 

**Abstract (ZH)**: 大规模语言模型（LLMs）如GPT-4和DeepSeek已在软件工程的多个领域得到应用。然而，在高性能计算（HPC）领域中的潜力仍待进一步探索。本文评估了最近的LLM DeepSeek在生成HPC基准代码方面的表现：共轭梯度求解器、并行热方程、并行矩阵乘法、DGEMM以及STREAM三元操作。我们分析了DeepSeek在C++、Fortran、Julia和Python等传统HPC语言的代码生成能力。评估包括对代码正确性、性能以及在不同配置和矩阵大小下的扩展性的测试。我们还提供了DeepSeek与另一个广泛使用的工具GPT-4之间的详细对比。我们的结果表明，虽然DeepSeek能够生成用于HPC任务的功能性代码，但在生成代码的可扩展性和执行效率方面仍落后于GPT-4。 

---
# PIPO: Pipelined Offloading for Efficient Inference on Consumer Devices 

**Title (ZH)**: PIPO：面向消费者设备高效推理的流水线卸载方法 

**Authors**: Yangyijian Liu, Jun Li, Wu-Jun Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.03664)  

**Abstract**: The high memory and computation demand of large language models (LLMs) makes them challenging to be deployed on consumer devices due to limited GPU memory. Offloading can mitigate the memory constraint but often suffers from low GPU utilization, leading to low inference efficiency. In this work, we propose a novel framework, called pipelined offloading (PIPO), for efficient inference on consumer devices. PIPO designs a fine-grained offloading pipeline, complemented with optimized data transfer and computation, to achieve high concurrency and efficient scheduling for inference. Experimental results show that compared with state-of-the-art baseline, PIPO increases GPU utilization from below 40% to over 90% and achieves up to 3.1$\times$ higher throughput, running on a laptop equipped with a RTX3060 GPU of 6GB memory. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的高内存和计算需求使其难以在消费级设备上部署，受限于有限的GPU内存。卸载可以缓解内存限制，但往往导致低GPU利用率，从而影响推理效率。在此工作中，我们提出了一种新的框架，称为流水线卸载（PIPO），旨在高效地在消费级设备上进行推理。PIPO设计了细粒度的卸载流水线，并结合优化的数据传输和计算，以实现高并发和高效的推理调度。实验结果表明，与最先进的基线相比，PIPO将GPU利用率从不到40%提高到超过90%，并在配备6GB显存的RTX3060 GPU的笔记本电脑上实现了高达3.1倍的更高吞吐量。 

---
# PointSplit: Towards On-device 3D Object Detection with Heterogeneous Low-power Accelerators 

**Title (ZH)**: PointSplit: 向着使用异构低功耗加速器的设备端3D物体检测 

**Authors**: Keondo Park, You Rim Choi, Inhoe Lee, Hyung-Sin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.03654)  

**Abstract**: Running deep learning models on resource-constrained edge devices has drawn significant attention due to its fast response, privacy preservation, and robust operation regardless of Internet connectivity. While these devices already cope with various intelligent tasks, the latest edge devices that are equipped with multiple types of low-power accelerators (i.e., both mobile GPU and NPU) can bring another opportunity; a task that used to be too heavy for an edge device in the single-accelerator world might become viable in the upcoming heterogeneous-accelerator this http URL realize the potential in the context of 3D object detection, we identify several technical challenges and propose PointSplit, a novel 3D object detection framework for multi-accelerator edge devices that addresses the problems. Specifically, our PointSplit design includes (1) 2D semantics-aware biased point sampling, (2) parallelized 3D feature extraction, and (3) role-based group-wise quantization. We implement PointSplit on TensorFlow Lite and evaluate it on a customized hardware platform comprising both mobile GPU and EdgeTPU. Experimental results on representative RGB-D datasets, SUN RGB-D and Scannet V2, demonstrate that PointSplit on a multi-accelerator device is 24.7 times faster with similar accuracy compared to the full-precision, 2D-3D fusion-based 3D detector on a GPU-only device. 

**Abstract (ZH)**: 基于多加速器边缘设备的点分割三维目标检测方法：利用低功耗移动GPU和NPU实现快速精准的三维目标检测 

---
# Echo: Efficient Co-Scheduling of Hybrid Online-Offline Tasks for Large Language Model Serving 

**Title (ZH)**: Echo: 效率兼优的混合在线-离线任务协同调度方法用于大型语言模型服务 

**Authors**: Zhibin Wang, Shipeng Li, Xue Li, Yuhang Zhou, Zhonghui Zhang, Zibo Wang, Rong Gu, Chen Tian, Kun Yang, Sheng Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2504.03651)  

**Abstract**: Large language models have been widely deployed in various applications, encompassing both interactive online tasks and batched offline tasks. Given the burstiness and latency sensitivity of online tasks, over-provisioning resources is common practice. This allows for the integration of latency-insensitive offline tasks during periods of low online load, enhancing resource utilization. However, strategically serving online and offline tasks through a preemption mechanism fails to fully leverage the flexibility of offline tasks and suffers from KV cache recomputation and irregular workloads.
In this paper, we introduce Echo, a collaborative online-offline task serving system, including a scheduler, a KV cache manager, and estimation toolkits. The scheduler and KV cache manager work tightly to maximize the throughput of offline tasks, while the estimator further predicts execution time to ensure online task SLOs. The scheduler leverages the batch information of last iteration to reduce the search space for finding the optimal schedule. The KV cache manager sets the priority of the KV cache based on the type of tasks and the opportunity of prefix sharing to reduce the recomputation. Finally, the estimation toolkits predict the execution time, future memory consumption, and the throughput of offline tasks to guide the scheduler, KV cache manager, and the system deployer. Evaluation based on real-world workloads demonstrates that Echo can increase offline task throughput by up to $3.3\times$, while satisfying online task SLOs. 

**Abstract (ZH)**: 一种协作的线上线下任务服务系统Echo：调度器、KV缓存管理器及估算工具箱 

---
# BoxRL-NNV: Boxed Refinement of Latin Hypercube Samples for Neural Network Verification 

**Title (ZH)**: BoxRL-NNV: 盒子精炼的拉丁超立方样本在神经网络验证中的应用 

**Authors**: Sarthak Das  

**Link**: [PDF](https://arxiv.org/pdf/2504.03650)  

**Abstract**: BoxRL-NNV is a Python tool for the detection of safety violations in neural networks by computing the bounds of the output variables, given the bounds of the input variables of the network. This is done using global extrema estimation via Latin Hypercube Sampling, and further refinement using L-BFGS-B for local optimization around the initial guess. This paper presents an overview of BoxRL-NNV, as well as our results for a subset of the ACAS Xu benchmark. A complete evaluation of the tool's performance, including benchmark comparisons with state-of-the-art tools, shall be presented at the Sixth International Verification of Neural Networks Competition (VNN-COMP'25). 

**Abstract (ZH)**: BoxRL-NNV是用于通过计算输出变量的界限来检测神经网络中的安全违规行为的Python工具，给定网络输入变量的界限。这是通过使用拉丁超立方抽样进行全局极值估计，并进一步使用L-BFGS-B进行局部优化来实现的。本文介绍了BoxRL-NNV的概述及其在ACAS Xu基准部分的数据结果。该工具的全面评估，包括与最新工具的基准比较，将在第六届国际神经网络验证竞赛（VNN-COMP'25）中呈现。 

---
# AIBrix: Towards Scalable, Cost-Effective Large Language Model Inference Infrastructure 

**Title (ZH)**: AIBrix: 向scalable、cost-effective的大型语言模型推理基础设施方向努力 

**Authors**: AIBrix Team, Jiaxin Shan, Varun Gupta, Le Xu, Haiyang Shi, Jingyuan Zhang, Ning Wang, Linhui Xu, Rong Kang, Tongping Liu, Yifei Zhang, Yiqing Zhu, Shuowei Jin, Gangmuk Lim, Binbin Chen, Zuzhi Chen, Xiao Liu, Xin Chen, Kante Yin, Chak-Pong Chung, Chenyu Jiang, Yicheng Lu, Jianjun Chen, Caixue Lin, Wu Xiang, Rui Shi, Liguang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2504.03648)  

**Abstract**: We introduce AIBrix, a cloud-native, open-source framework designed to optimize and simplify large-scale LLM deployment in cloud environments. Unlike traditional cloud-native stacks, AIBrix follows a co-design philosophy, ensuring every layer of the infrastructure is purpose-built for seamless integration with inference engines like vLLM. AIBrix introduces several key innovations to reduce inference costs and enhance performance including high-density LoRA management for dynamic adapter scheduling, LLM-specific autoscalers, and prefix-aware, load-aware routing. To further improve efficiency, AIBrix incorporates a distributed KV cache, boosting token reuse across nodes, leading to a 50% increase in throughput and a 70% reduction in inference latency. AIBrix also supports unified AI runtime which streamlines model management while maintaining vendor-agnostic engine compatibility. For large-scale multi-node inference, AIBrix employs hybrid orchestration -- leveraging Kubernetes for coarse-grained scheduling and Ray for fine-grained execution -- to balance efficiency and flexibility. Additionally, an SLO-driven GPU optimizer dynamically adjusts resource allocations, optimizing heterogeneous serving to maximize cost efficiency while maintaining service guarantees. Finally, AIBrix enhances system reliability with AI accelerator diagnostic tools, enabling automated failure detection and mock-up testing to improve fault resilience. AIBrix is available at this https URL. 

**Abstract (ZH)**: 我们介绍AIBrix，一个面向云环境的大规模LLM部署优化和简化的设计理念，AIBrix是一个云原生、开源框架。不同于传统的云原生堆栈，AIBrix遵循共同设计哲学，确保每层基础设施都为无缝集成与推理引擎（如vLLM）构建。AIBrix引入了几项关键创新以减少推理成本并提升性能，包括动态适配器调度的高密度LoRA管理、特定于LLM的自动化扩容器以及前缀感知、负载感知路由。为了进一步提高效率，AIBrix集成了一个分布式KV缓存，提高节点间token重用，从而将吞吐量提高50%，推理延迟减少70%。AIBrix还支持统一AI运行时，简化模型管理同时保持对供应商无关引擎的兼容性。对于大规模多节点推理，AIBrix采用混合编排——利用Kubernetes进行粗粒度调度和利用Ray进行细粒度执行，以平衡效率和灵活性。此外，一个基于SLA的GPU优化器动态调整资源分配，优化异构服务以实现成本效率最大化并维持服务质量保证。最后，AIBrix通过AI加速器诊断工具增强系统可靠性，实现自动化故障检测和模拟测试以提高故障容忍度。AIBrix可在以下链接获取：此httpsURL。 

---
# Potential Indicator for Continuous Emotion Arousal by Dynamic Neural Synchrony 

**Title (ZH)**: 持续情感唤醒的动态神经同步潜在指标 

**Authors**: Guandong Pan, Zhaobang Wu, Yaqian Yang, Xin Wang, Longzhao Liu, Zhiming Zheng, Shaoting Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03643)  

**Abstract**: The need for automatic and high-quality emotion annotation is paramount in applications such as continuous emotion recognition and video highlight detection, yet achieving this through manual human annotations is challenging. Inspired by inter-subject correlation (ISC) utilized in neuroscience, this study introduces a novel Electroencephalography (EEG) based ISC methodology that leverages a single-electrode and feature-based dynamic approach. Our contributions are three folds. Firstly, we reidentify two potent emotion features suitable for classifying emotions-first-order difference (FD) an differential entropy (DE). Secondly, through the use of overall correlation analysis, we demonstrate the heterogeneous synchronized performance of electrodes. This performance aligns with neural emotion patterns established in prior studies, thus validating the effectiveness of our approach. Thirdly, by employing a sliding window correlation technique, we showcase the significant consistency of dynamic ISCs across various features or key electrodes in each analyzed film clip. Our findings indicate the method's reliability in capturing consistent, dynamic shared neural synchrony among individuals, triggered by evocative film stimuli. This underscores the potential of our approach to serve as an indicator of continuous human emotion arousal. The implications of this research are significant for advancements in affective computing and the broader neuroscience field, suggesting a streamlined and effective tool for emotion analysis in real-world applications. 

**Abstract (ZH)**: 基于EEG的单电极特征动态方法在情绪注释中的应用：自动和高质量情绪标注的需求 

---
# CodeIF-Bench: Evaluating Instruction-Following Capabilities of Large Language Models in Interactive Code Generation 

**Title (ZH)**: CodeIF-基准：评估大型语言模型在交互式代码生成中的指令遵循能力 

**Authors**: Peiding Wang, Li Zhang, Fang Liu, Lin Shi, Minxiao Li, Bo Shen, An Fu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22688)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional performance in code generation tasks and have become indispensable programming assistants for developers. However, existing code generation benchmarks primarily assess the functional correctness of code generated by LLMs in single-turn interactions, offering limited insight into their capabilities to generate code that strictly follows users' instructions, especially in multi-turn interaction scenarios. In this paper, we introduce \bench, a benchmark for evaluating LLMs' instruction-following capabilities in interactive code generation. Specifically, \bench incorporates nine types of verifiable instructions aligned with the real-world software development requirements, which can be independently and objectively validated through specified test cases, facilitating the evaluation of instruction-following capability in multi-turn interactions. We evaluate nine prominent LLMs using \bench, and the experimental results reveal a significant disparity between their basic programming capability and instruction-following capability, particularly as task complexity, context length, and the number of dialogue rounds increase. 

**Abstract (ZH)**: Large Language Models (LLMs)在代码生成任务中表现出色，已成为开发人员不可或缺的编程助手。然而，现有的代码生成基准主要评估LLMs在单轮交互中生成的代码的功能正确性，对于它们生成严格遵循用户指令的代码的能力提供有限的洞察，尤其是在多轮交互场景中。本文介绍了一个新的基准\bench，用于评估LLMs在交互式代码生成中的指令遵循能力。\bench包含九种与实际软件开发需求对齐的可验证指令，可以通过指定的测试用例独立且客观地验证，从而促进多轮交互中的指令遵循能力评估。我们使用\bench评估了九种 prominant LLMs，并且实验结果揭示了它们的基本编程能力和指令遵循能力之间存在显著差异，特别是在任务复杂度、上下文长度和对话轮次增加的情况下。 

---
# Packet Inspection Transformer: A Self-Supervised Journey to Unseen Malware Detection with Few Samples 

**Title (ZH)**: 包检查变换器：基于少量样本的自监督非看见恶意软件检测之旅 

**Authors**: Kyle Stein, Arash Mahyari, Guillermo Francia III, Eman El-Sheikh  

**Link**: [PDF](https://arxiv.org/pdf/2409.18219)  

**Abstract**: As networks continue to expand and become more interconnected, the need for novel malware detection methods becomes more pronounced. Traditional security measures are increasingly inadequate against the sophistication of modern cyber attacks. Deep Packet Inspection (DPI) has been pivotal in enhancing network security, offering an in-depth analysis of network traffic that surpasses conventional monitoring techniques. DPI not only examines the metadata of network packets, but also dives into the actual content being carried within the packet payloads, providing a comprehensive view of the data flowing through networks. While the integration of advanced deep learning techniques with DPI has introduced modern methodologies into malware detection and network traffic classification, state-of-the-art supervised learning approaches are limited by their reliance on large amounts of annotated data and their inability to generalize to novel, unseen malware threats. To address these limitations, this paper leverages the recent advancements in self-supervised learning (SSL) and few-shot learning (FSL). Our proposed self-supervised approach trains a transformer via SSL to learn the embedding of packet content, including payload, from vast amounts of unlabeled data by masking portions of packets, leading to a learned representation that generalizes to various downstream tasks. Once the representation is extracted from the packets, they are used to train a malware detection algorithm. The representation obtained from the transformer is then used to adapt the malware detector to novel types of attacks using few-shot learning approaches. Our experimental results demonstrate that our method achieves classification accuracies of up to 94.76% on the UNSW-NB15 dataset and 83.25% on the CIC-IoT23 dataset. 

**Abstract (ZH)**: 随着网络不断扩展和互联互通，新型恶意软件检测方法的需求日益凸显。传统安全措施越来越难以应对现代网络攻击的复杂性。深度包检测（DPI）在提升网络安全性方面发挥了关键作用，它提供了超越传统监控技术的深入网络流量分析。DPI不仅检查网络包的元数据，还深入分析包载荷的实际内容，提供网络中传输数据的全面视图。尽管将高级深度学习技术与DPI结合引入了先进的恶意软件检测和网络流量分类方法，但最先进的监督学习方法受限于对大量标注数据的依赖以及无法泛化到新型未知恶意软件威胁。为解决这些限制，本文利用自监督学习（SSL）和少样本学习（FSL）的最新进展。我们提出了一种自监督方法，通过SSL训练一个变换器，从大量未标注数据中学习包内容（包括载荷）的表示，通过掩蔽包部分内容，生成适用于各种下游任务的泛化表示。一旦从包中提取表示，就用于训练恶意软件检测算法。从变换器获得的表示随后用于使用少样本学习方法适应新型攻击类型的恶意软件检测器。我们的实验结果表明，该方法在UNSW-NB15数据集上的分类准确率高达94.76%，在CIC-IoT23数据集上的准确率为83.25%。 

---
