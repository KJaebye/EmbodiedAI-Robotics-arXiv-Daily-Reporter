# Cognitive Foundations for Reasoning and Their Manifestation in LLMs 

**Title (ZH)**: 认知基础推理及其在大规模语言模型中的表现 

**Authors**: Priyanka Kargupta, Shuyue Stella Li, Haocheng Wang, Jinu Lee, Shan Chen, Orevaoghene Ahia, Dean Light, Thomas L. Griffiths, Max Kleiman-Weiner, Jiawei Han, Asli Celikyilmaz, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2511.16660)  

**Abstract**: Large language models solve complex problems yet fail on simpler variants, suggesting they achieve correct outputs through mechanisms fundamentally different from human reasoning. We synthesize cognitive science research into a taxonomy of 28 cognitive elements spanning computational constraints, meta-cognitive controls, knowledge representations, and transformation operations, then analyze their behavioral manifestations in reasoning traces. We propose a fine-grained cognitive evaluation framework and conduct the first large-scale analysis of 170K traces from 17 models across text, vision, and audio modalities, alongside 54 human think-aloud traces, which we make publicly available. Our analysis reveals systematic structural differences: humans employ hierarchical nesting and meta-cognitive monitoring while models rely on shallow forward chaining, with divergence most pronounced on ill-structured problems. Meta-analysis of 1,598 LLM reasoning papers reveals the research community concentrates on easily quantifiable behaviors (sequential organization: 55%, decomposition: 60%) while neglecting meta-cognitive controls (self-awareness: 16%, evaluation: 8%) that correlate with success. Models possess behavioral repertoires associated with success but fail to deploy them spontaneously. Leveraging these patterns, we develop test-time reasoning guidance that automatically scaffold successful structures, improving performance by up to 60% on complex problems. By bridging cognitive science and LLM research, we establish a foundation for developing models that reason through principled cognitive mechanisms rather than brittle spurious reasoning shortcuts or memorization, opening new directions for both improving model capabilities and testing theories of human cognition at scale. 

**Abstract (ZH)**: 大型语言模型能够解决复杂问题但在简单变体上失败，这表明它们通过与人类推理本质上不同机制实现正确输出。我们综合认知科学研究，构建涵盖计算限制、元认知控制、知识表示和变换操作的28个认知元素分类体系，然后在推理轨迹中分析其行为表现。我们提出了一种精细的认知评估框架，并对来自17种模型、涵盖文本、视觉和音频模态的17万个推理轨迹进行了首次大规模分析，同时还包括了54个手动思考 aloud 的轨迹，我们已将这些数据公开。我们的分析揭示了系统性的结构差异：人类运用层次嵌套和元认知监控，而模型依赖浅层向前链式推理，差异在构架不良的问题上最为明显。对1598篇大规模语言模型推理论文的元分析显示，研究界集中在易于量化的行为（顺序组织：55%，分解：60%）上，而忽视与成功相关的元认知控制（自我意识：16%，评价：8%）。模型拥有与成功相关的行为 repertoire，但无法自发使用它们。利用这些模式，我们开发了一种推理指导，在测试时自动搭建成功的结构，复杂问题上的性能提升高达60%。通过连接认知科学与大型语言模型研究，我们为通过原理性的认知机制开发能进行推理的模型奠定了基础，而不是脆弱的虚假推理捷径或记忆，为提高模型能力并大规模测试人类认知理论提供了新方向。 

---
# Enhancing Forex Forecasting Accuracy: The Impact of Hybrid Variable Sets in Cognitive Algorithmic Trading Systems 

**Title (ZH)**: 增强外汇预测准确性：认知算法交易系统中混合变量集的影响 

**Authors**: Juan C. King, Jose M. Amigo  

**Link**: [PDF](https://arxiv.org/pdf/2511.16657)  

**Abstract**: This paper presents the implementation of an advanced artificial intelligence-based algorithmic trading system specifically designed for the EUR-USD pair within the high-frequency environment of the Forex market. The methodological approach centers on integrating a holistic set of input features: key fundamental macroeconomic variables (for example, Gross Domestic Product and Unemployment Rate) collected from both the Euro Zone and the United States, alongside a comprehensive suite of technical variables (including indicators, oscillators, Fibonacci levels, and price divergences). The performance of the resulting algorithm is evaluated using standard machine learning metrics to quantify predictive accuracy and backtesting simulations across historical data to assess trading profitability and risk. The study concludes with a comparative analysis to determine which class of input features, fundamental or technical, provides greater and more reliable predictive capacity for generating profitable trading signals. 

**Abstract (ZH)**: 这篇论文介绍了针对外汇市场欧元/美元pair的一种高级基于人工智能的算法交易系统在高频交易环境中的实现。方法论集中在整合一套全面的输入特征：来自欧元区和美国的关键宏观经济变量（例如，国内生产总值和失业率），以及一系列全面的技术变量（包括指标、振荡器、斐波那契水平和价格偏离）。通过使用标准机器学习度量来评估算法的性能，衡量预测准确性和通过历史数据回测模拟来评估交易盈利能力和风险。研究结论部分进行了一种比较分析，以确定哪一类输入特征，基本面或技术面，能提供更强和更可靠的预测能力以生成盈利交易信号。 

---
# MedBayes-Lite: Bayesian Uncertainty Quantification for Safe Clinical Decision Support 

**Title (ZH)**: MedBayes-Lite: Bayesian不确定性量化以实现安全的临床决策支持 

**Authors**: Elias Hossain, Md Mehedi Hasan Nipu, Maleeha Sheikh, Rajib Rana, Subash Neupane, Niloofar Yousefi  

**Link**: [PDF](https://arxiv.org/pdf/2511.16625)  

**Abstract**: We propose MedBayes-Lite, a lightweight Bayesian enhancement for transformer-based clinical language models designed to produce reliable, uncertainty-aware predictions. Although transformers show strong potential for clinical decision support, they remain prone to overconfidence, especially in ambiguous medical cases where calibrated uncertainty is critical. MedBayes-Lite embeds uncertainty quantification directly into existing transformer pipelines without any retraining or architectural rewiring, adding no new trainable layers and keeping parameter overhead under 3 percent. The framework integrates three components: (i) Bayesian Embedding Calibration using Monte Carlo dropout for epistemic uncertainty, (ii) Uncertainty-Weighted Attention that marginalizes over token reliability, and (iii) Confidence-Guided Decision Shaping inspired by clinical risk minimization. Across biomedical QA and clinical prediction benchmarks (MedQA, PubMedQA, MIMIC-III), MedBayes-Lite consistently improves calibration and trustworthiness, reducing overconfidence by 32 to 48 percent. In simulated clinical settings, it can prevent up to 41 percent of diagnostic errors by flagging uncertain predictions for human review. These results demonstrate its effectiveness in enabling reliable uncertainty propagation and improving interpretability in medical AI systems. 

**Abstract (ZH)**: MedBayes-Lite：一种轻量级的贝叶斯增强方法，用于生成临床语言模型的可靠、含不确定性意识的预测 

---
# Bridging VLMs and Embodied Intelligence with Deliberate Practice Policy Optimization 

**Title (ZH)**: 通过刻意练习政策优化连接大模型和嵌入式智能 

**Authors**: Yi Zhang, Che Liu, Xiancong Ren, Hanchu Ni, Yingji Zhang, Shuai Zhang, Zeyuan Ding, Jiayu Hu, Haozhe Shan, Junbo Qi, Yan Bai, Dengjie Li, Jiachen Luo, Yidong Wang, Yong Dai, Zenglin Xu, Bin Shen, Qifan Wang, Jian Tang, Xiaozhu Ju  

**Link**: [PDF](https://arxiv.org/pdf/2511.16602)  

**Abstract**: Developing a universal and versatile embodied intelligence system presents two primary challenges: the critical embodied data bottleneck, where real-world data is scarce and expensive, and the algorithmic inefficiency of existing methods, which are resource-prohibitive. To address these limitations, we introduce Deliberate Practice Policy Optimization (DPPO), a metacognitive ``Metaloop'' training framework that dynamically alternates between supervised fine-tuning (competence expansion) and reinforcement learning (skill refinement). This enables automatic weakness identification and targeted resource allocation, specifically designed to maximize learning efficiency from sparse, finite data. Theoretically, DPPO can be formalised as a unified preference-learning framework. Empirically, training a vision-language embodied model with DPPO, referred to as Pelican-VL 1.0, yields a 20.3% performance improvement over the base model and surpasses open-source models at the 100B-parameter scale by 10.6%. We are open-sourcing both the models and code, providing the first systematic framework that alleviates the data and resource bottleneck and enables the community to build versatile embodied agents efficiently. 

**Abstract (ZH)**: 开发通用且多功能的物质智能系统面临两大主要挑战：现实世界数据的瓶颈，即数据稀缺且昂贵，以及现有方法的算法低效性，这些方法在资源利用上具有限制性。为了解决这些限制，我们引入了审慎实践策略优化（DPPO）元认知“元循环”训练框架，该框架动态交替进行监督微调（能力扩展）和强化学习（技能精炼）。这使自动识别弱点和目标资源分配成为可能，特别设计用于从稀疏有限的数据中最大化学习效率。理论上，DPPO可以形式化为统一的偏好学习框架。实验上，使用DPPO训练的视觉-语言物质智能模型Pelican-VL 1.0，在基模型上取得了20.3%的性能提升，并在100B参数规模上超出开源模型10.6%。我们开源了模型和代码，提供了首个系统框架，解决了数据和资源瓶颈，使科研界能够高效地构建多功能的物质代理。 

---
# You Only Forward Once: An Efficient Compositional Judging Paradigm 

**Title (ZH)**: 你只需前馈一次：一个高效的组合判断范式 

**Authors**: Tianlong Zhang, Hongwei Xue, Shilin Yan, Di Wu, Chen Xu, Yunyun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16600)  

**Abstract**: Multimodal large language models (MLLMs) show strong potential as judges. However, existing approaches face a fundamental trade-off: adapting MLLMs to output a single score misaligns with the generative nature of MLLMs and limits fine-grained requirement understanding, whereas autoregressively generating judging analyses is prohibitively slow in high-throughput settings. Observing that judgment reduces to verifying whether inputs satisfy a set of structured requirements, we propose YOFO, a template-conditioned method that judges all requirements in a single forward pass. Built on an autoregressive model, YOFO accepts a structured requirement template and, in one inference step, produces a binary yes/no decision for each requirement by reading the logits of the final token associated with that requirement. This design yields orders-of-magnitude speedups while preserving interpretability. Extensive experiments show that YOFO not only achieves state-of-the-art results on standard recommendation datasets, but also supports dependency-aware analysis-where subsequent judgments are conditioned on previous ones-and further benefits from post-hoc CoT. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）作为裁判显示出强烈的潜力。然而，现有的方法面临一个基本的权衡：使MLLMs适应输出单一分数与MLLMs的生成性质相冲突，并限制了细粒度的要求理解，而自回归生成裁判分析在高通量设置中是不可行的。观察到裁判归结为验证输入是否满足一组结构化要求，我们提出YOFO，这是一种模板条件化方法，可以在单次前向传递中评判所有要求。基于自回归模型，YOFO接受一个结构化要求模板，并在一次推理步骤中通过阅读与该要求相关的最终词元的logits生成每个要求的二元是/否决策。这种设计在保持可解释性的同时实现了数量级的速度提升。广泛实验表明，YOFO不仅在标准推荐数据集上达到了最先进的性能，还能支持依赖意识的分析——后续判断依赖于先前的判断——并且进一步受益于事后解释性推理（CoT）。 

---
# D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies 

**Title (ZH)**: D-GARA: 一种针对实际异常中 GUI 代理稳健性的动态基准框架 

**Authors**: Sen Chen, Tong Zhao, Yi Bin, Fei Ma, Wenqi Shao, Zheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16590)  

**Abstract**: Developing intelligent agents capable of operating a wide range of Graphical User Interfaces (GUIs) with human-level proficiency is a key milestone on the path toward Artificial General Intelligence. While most existing datasets and benchmarks for training and evaluating GUI agents are static and idealized, failing to reflect the complexity and unpredictability of real-world environments, particularly the presence of anomalies. To bridge this research gap, we propose D-GARA, a dynamic benchmarking framework, to evaluate Android GUI agent robustness in real-world anomalies. D-GARA introduces a diverse set of real-world anomalies that GUI agents commonly face in practice, including interruptions such as permission dialogs, battery warnings, and update prompts. Based on D-GARA framework, we construct and annotate a benchmark featuring commonly used Android applications with embedded anomalies to support broader community research. Comprehensive experiments and results demonstrate substantial performance degradation in state-of-the-art GUI agents when exposed to anomaly-rich environments, highlighting the need for robustness-aware learning. D-GARA is modular and extensible, supporting the seamless integration of new tasks, anomaly types, and interaction scenarios to meet specific evaluation goals. 

**Abstract (ZH)**: 开发能够以人类水平的熟练程度操作多种图形用户界面（GUI）的智能代理是通向人工通用智能的关键里程碑。为弥补现有训练和评估GUI代理的数据集和基准大多是静态和理想的，未能反映现实世界环境的复杂性和不可预测性，特别是异常情况的空白，我们提出了D-GARA动态基准框架，用于评估Android GUI代理在真实世界异常环境下的鲁棒性。D-GARA引入了一组GUI代理在实践中常遇到的多样的真实世界异常，包括权限对话框、电池警告和更新提示等中断。基于D-GARA框架，我们构建并标注了一个基准，其中包括嵌入异常的常用Android应用程序，以支持更广泛的研究社区的研究。全面的实验和结果表明，在异常丰富的环境中，最先进的GUI代理会遭受显著性能下降，突显了鲁棒性意识学习的需求。D-GARA模块化且可扩展，支持新任务、异常类型和交互场景的无缝集成，以满足特定的评估目标。 

---
# Formal Abductive Latent Explanations for Prototype-Based Networks 

**Title (ZH)**: 基于 prototypes 的网络的形式 abduction 潜在解释 

**Authors**: Jules Soria, Zakaria Chihani, Julien Girard-Satabin, Alban Grastien, Romain Xu-Darme, Daniela Cancila  

**Link**: [PDF](https://arxiv.org/pdf/2511.16588)  

**Abstract**: Case-based reasoning networks are machine-learning models that make predictions based on similarity between the input and prototypical parts of training samples, called prototypes. Such models are able to explain each decision by pointing to the prototypes that contributed the most to the final outcome. As the explanation is a core part of the prediction, they are often qualified as ``interpretable by design". While promising, we show that such explanations are sometimes misleading, which hampers their usefulness in safety-critical contexts. In particular, several instances may lead to different predictions and yet have the same explanation. Drawing inspiration from the field of formal eXplainable AI (FXAI), we propose Abductive Latent Explanations (ALEs), a formalism to express sufficient conditions on the intermediate (latent) representation of the instance that imply the prediction. Our approach combines the inherent interpretability of case-based reasoning models and the guarantees provided by formal XAI. We propose a solver-free and scalable algorithm for generating ALEs based on three distinct paradigms, compare them, and present the feasibility of our approach on diverse datasets for both standard and fine-grained image classification. The associated code can be found at this https URL 

**Abstract (ZH)**: 基于案例的推理网络是基于输入与训练样本中原型部分相似性进行预测的机器学习模型，这样的模型能够通过指向贡献最大的原型来解释每个决策。由于解释是预测的核心部分，它们常被视为“设计即解释”。虽然前景广阔，但我们展示了这样的解释有时是误导性的，这在安全关键应用中限制了它们的有用性。特别是，多个实例可能会得到不同的预测结果，但具有相同解释。受到形式可解释AI（FXAI）领域的启发，我们提出了归纳潜在解释（ALEs），这是一种表达实例中间（潜在）表示的充分条件的形式化方法，这些条件能蕴含预测结果。我们的方法结合了基于案例的推理模型的固有解释性和形式XAI提供的保证。我们提出了一种无需求解器且可扩展的算法来生成ALEs，并基于三种不同的范式进行比较，在多种数据集上展示了我们的方法在标准和细粒度图像分类中的可行性。相关代码可在以下链接找到：this https URL 

---
# Consciousness in Artificial Intelligence? A Framework for Classifying Objections and Constraints 

**Title (ZH)**: 人工意识？一种分类反对观点和约束条件的框架 

**Authors**: Andres Campero, Derek Shiller, Jaan Aru, Jonathan Simon  

**Link**: [PDF](https://arxiv.org/pdf/2511.16582)  

**Abstract**: We develop a taxonomical framework for classifying challenges to the possibility of consciousness in digital artificial intelligence systems. This framework allows us to identify the level of granularity at which a given challenge is intended (the levels we propose correspond to Marr's levels) and to disambiguate its degree of force: is it a challenge to computational functionalism that leaves the possibility of digital consciousness open (degree 1), a practical challenge to digital consciousness that suggests improbability without claiming impossibility (degree 2), or an argument claiming that digital consciousness is strictly impossible (degree 3)? We apply this framework to 14 prominent examples from the scientific and philosophical literature. Our aim is not to take a side in the debate, but to provide structure and a tool for disambiguating between challenges to computational functionalism and challenges to digital consciousness, as well as between different ways of parsing such challenges. 

**Abstract (ZH)**: 我们开发了一种分类框架，用于分类数字人工智能系统中对意识可能性的挑战。该框架使我们能够确定给定挑战所针对的具体层次（我们提出的层次对应于Marr的层次），并区分其强度程度：它是对计算功能主义的挑战但保留了数字意识的可能性（第1级）、对数字意识的实际挑战，暗示不可能性但未声明不可能（第2级），还是对数字意识严格不可能性的论据（第3级）？我们将这一框架应用于科学和哲学文献中的14个典型案例。我们的目标不是介入辩论，而是提供一种结构和工具，用于区分对计算功能主义的挑战与对数字意识的挑战，以及不同解析这些挑战的方式。 

---
# Utilizing Large Language Models for Zero-Shot Medical Ontology Extension from Clinical Notes 

**Title (ZH)**: 利用大规模语言模型实现从临床笔记零样本扩展医疗本体的研究 

**Authors**: Guanchen Wu, Yuzhang Xie, Huanwei Wu, Zhe He, Hui Shao, Xiao Hu, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16548)  

**Abstract**: Integrating novel medical concepts and relationships into existing ontologies can significantly enhance their coverage and utility for both biomedical research and clinical applications. Clinical notes, as unstructured documents rich with detailed patient observations, offer valuable context-specific insights and represent a promising yet underutilized source for ontology extension. Despite this potential, directly leveraging clinical notes for ontology extension remains largely unexplored. To address this gap, we propose CLOZE, a novel framework that uses large language models (LLMs) to automatically extract medical entities from clinical notes and integrate them into hierarchical medical ontologies. By capitalizing on the strong language understanding and extensive biomedical knowledge of pre-trained LLMs, CLOZE effectively identifies disease-related concepts and captures complex hierarchical relationships. The zero-shot framework requires no additional training or labeled data, making it a cost-efficient solution. Furthermore, CLOZE ensures patient privacy through automated removal of protected health information (PHI). Experimental results demonstrate that CLOZE provides an accurate, scalable, and privacy-preserving ontology extension framework, with strong potential to support a wide range of downstream applications in biomedical research and clinical informatics. 

**Abstract (ZH)**: 将新颖的医学概念和关系集成到现有本体中，可以显著增强其在生物医药研究和临床应用中的覆盖范围和实用性。临床笔记作为一种富含详细患者观察信息的非结构化文档，提供了有价值的具体背景见解，并代表了一种有潜力但尚未充分利用的本体扩展来源。尽管存在这种潜力，但直接利用临床笔记进行本体扩展的研究仍处于起步阶段。为解决这一缺口，我们提出了CLOZE，一个新颖的框架，利用大规模语言模型（LLMs）自动从临床笔记中提取医学实体并将其集成到医学本体中。通过充分利用预训练LLMs强大的语言理解和广泛生物医药知识，CLOZE有效地识别疾病相关的概念并捕获复杂的层级关系。零样本框架不需要额外的训练或标注数据，使其成为一种成本效益高的解决方案。此外，CLOZE通过自动化去除受保护的健康信息（PHI）来确保患者隐私。实验结果表明，CLOZE提供了一种准确、可扩展且保护隐私的本体扩展框架，具有强大的潜力支持生物医药研究和临床信息技术等一系列下游应用。 

---
# PersonaDrift: A Benchmark for Temporal Anomaly Detection in Language-Based Dementia Monitoring 

**Title (ZH)**: 基于语言的痴呆监测中时间异常检测的基准：PersonaDrift 

**Authors**: Joy Lai, Alex Mihailidis  

**Link**: [PDF](https://arxiv.org/pdf/2511.16445)  

**Abstract**: People living with dementia (PLwD) often show gradual shifts in how they communicate, becoming less expressive, more repetitive, or drifting off-topic in subtle ways. While caregivers may notice these changes informally, most computational tools are not designed to track such behavioral drift over time. This paper introduces PersonaDrift, a synthetic benchmark designed to evaluate machine learning and statistical methods for detecting progressive changes in daily communication, focusing on user responses to a digital reminder system. PersonaDrift simulates 60-day interaction logs for synthetic users modeled after real PLwD, based on interviews with caregivers. These caregiver-informed personas vary in tone, modality, and communication habits, enabling realistic diversity in behavior. The benchmark focuses on two forms of longitudinal change that caregivers highlighted as particularly salient: flattened sentiment (reduced emotional tone and verbosity) and off-topic replies (semantic drift). These changes are injected progressively at different rates to emulate naturalistic cognitive trajectories, and the framework is designed to be extensible to additional behaviors in future use cases. To explore this novel application space, we evaluate several anomaly detection approaches, unsupervised statistical methods (CUSUM, EWMA, One-Class SVM), sequence models using contextual embeddings (GRU + BERT), and supervised classifiers in both generalized and personalized settings. Preliminary results show that flattened sentiment can often be detected with simple statistical models in users with low baseline variability, while detecting semantic drift requires temporal modeling and personalized baselines. Across both tasks, personalized classifiers consistently outperform generalized ones, highlighting the importance of individual behavioral context. 

**Abstract (ZH)**: PLwD在日常交流中的渐进式变化及其检测：PersonaDrift合成基准探讨 

---
# From generative AI to the brain: five takeaways 

**Title (ZH)**: 从生成式AI到大脑：五点启示 

**Authors**: Claudius Gros  

**Link**: [PDF](https://arxiv.org/pdf/2511.16432)  

**Abstract**: The big strides seen in generative AI are not based on somewhat obscure algorithms, but due to clearly defined generative principles. The resulting concrete implementations have proven themselves in large numbers of applications. We suggest that it is imperative to thoroughly investigate which of these generative principles may be operative also in the brain, and hence relevant for cognitive neuroscience. In addition, ML research led to a range of interesting characterizations of neural information processing systems. We discuss five examples, the shortcomings of world modelling, the generation of thought processes, attention, neural scaling laws, and quantization, that illustrate how much neuroscience could potentially learn from ML research. 

**Abstract (ZH)**: 生成AI取得的显著进展并非基于某些晦涩的算法，而是由于明确的生成原则。具体实现已经在大量应用程序中得到了验证。我们建议必须深入研究这些生成原则是否也适用于大脑，并因此对认知神经科学具有重要意义。此外，机器学习研究还导致了对神经信息处理系统的多种有趣的表征。我们讨论了五种例子，包括世界建模的局限性、思维过程的生成、注意力机制、神经可标性定律和量化，展示了神经科学从机器学习研究中可能学到的内容。 

---
# TOFA: Training-Free One-Shot Federated Adaptation for Vision-Language Models 

**Title (ZH)**: TOFA：无需训练的一次性联邦适应Visio-Lang模型 

**Authors**: Li Zhang, Zhongxuan Han, XiaoHua Feng, Jiaming Zhang, Yuyuan Li, Linbo Jiang, Jianan Lin, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.16423)  

**Abstract**: Efficient and lightweight adaptation of pre-trained Vision-Language Models (VLMs) to downstream tasks through collaborative interactions between local clients and a central server is a rapidly emerging research topic in federated learning. Existing adaptation algorithms are typically trained iteratively, which incur significant communication costs and increase the susceptibility to potential attacks. Motivated by the one-shot federated training techniques that reduce client-server exchanges to a single round, developing a lightweight one-shot federated VLM adaptation method to alleviate these issues is particularly attractive. However, current one-shot approaches face certain challenges in adapting VLMs within federated settings: (1) insufficient exploitation of the rich multimodal information inherent in VLMs; (2) lack of specialized adaptation strategies to systematically handle the severe data heterogeneity; and (3) requiring additional training resource of clients or server. To bridge these gaps, we propose a novel Training-free One-shot Federated Adaptation framework for VLMs, named TOFA. To fully leverage the generalizable multimodal features in pre-trained VLMs, TOFA employs both visual and textual pipelines to extract task-relevant representations. In the visual pipeline, a hierarchical Bayesian model learns personalized, class-specific prototype distributions. For the textual pipeline, TOFA evaluates and globally aligns the generated local text prompts for robustness. An adaptive weight calibration mechanism is also introduced to combine predictions from both modalities, balancing personalization and robustness to handle data heterogeneity. Our method is training-free, not relying on additional training resources on either the client or server side. Extensive experiments across 9 datasets in various federated settings demonstrate the effectiveness of the proposed TOFA method. 

**Abstract (ZH)**: 无需训练的联邦 adaptation 体系结构：TOFAmissive 

---
# Pharos-ESG: A Framework for Multimodal Parsing, Contextual Narration, and Hierarchical Labeling of ESG Report 

**Title (ZH)**: Pharos-ESG：一种多模态解析、上下文叙事和层级标注的ESG报告框架 

**Authors**: Yan Chen, Yu Zou, Jialei Zeng, Haoran You, Xiaorui Zhou, Aixi Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2511.16417)  

**Abstract**: Environmental, Social, and Governance (ESG) principles are reshaping the foundations of global financial gover- nance, transforming capital allocation architectures, regu- latory frameworks, and systemic risk coordination mecha- nisms. However, as the core medium for assessing corpo- rate ESG performance, the ESG reports present significant challenges for large-scale understanding, due to chaotic read- ing order from slide-like irregular layouts and implicit hier- archies arising from lengthy, weakly structured content. To address these challenges, we propose Pharos-ESG, a uni- fied framework that transforms ESG reports into structured representations through multimodal parsing, contextual nar- ration, and hierarchical labeling. It integrates a reading-order modeling module based on layout flow, hierarchy-aware seg- mentation guided by table-of-contents anchors, and a multi- modal aggregation pipeline that contextually transforms vi- sual elements into coherent natural language. The framework further enriches its outputs with ESG, GRI, and sentiment labels, yielding annotations aligned with the analytical de- mands of financial research. Extensive experiments on anno- tated benchmarks demonstrate that Pharos-ESG consistently outperforms both dedicated document parsing systems and general-purpose multimodal models. In addition, we release Aurora-ESG, the first large-scale public dataset of ESG re- ports, spanning Mainland China, Hong Kong, and U.S. mar- kets, featuring unified structured representations of multi- modal content, enriched with fine-grained layout and seman- tic annotations to better support ESG integration in financial governance and decision-making. 

**Abstract (ZH)**: 环境、社会与治理（ESG）原则正在重塑全球金融治理的基础，转变资本分配架构、监管框架和系统性风险协调机制。然而，作为评估企业ESG绩效的核心工具，ESG报告在大规模理解上面临显著挑战，由于幻灯片式不规则布局造成的混乱阅读顺序以及来自冗长、结构松散内容的隐含层级关系。为解决这些挑战，我们提出Pharos-ESG，这是一种统一框架，通过多模态解析、上下文叙述和层级标注将ESG报告转换为结构化的表示。该框架整合了基于布局流的阅读顺序建模模块、由目录锚点指导的层级感知段落分割，以及一个能够上下文化地将视觉元素转换为连贯自然语言的多模态聚合管道。进一步地，该框架增添了ESG、GRI和情感标签，确保其输出能够满足金融研究的分析需求。在标注基准上的广泛实验表明，Pharos-ESG 一致优于专门的文档解析系统和通用多模态模型。此外，我们发布了Aurora-ESG，这是首个大规模公开的ESG报告数据集，涵盖中国大陆、香港和美国市场，提供统一的多模态内容结构表示，并配备了精细的布局和语义标注，更好地支持ESG在金融治理和决策中的集成。 

---
# Trustworthy AI in the Agentic Lakehouse: from Concurrency to Governance 

**Title (ZH)**: 代理湖仓中的可信AI：从并发到治理 

**Authors**: Jacopo Tagliabue, Federico Bianchi, Ciro Greco  

**Link**: [PDF](https://arxiv.org/pdf/2511.16402)  

**Abstract**: Even as AI capabilities improve, most enterprises do not consider agents trustworthy enough to work on production data. In this paper, we argue that the path to trustworthy agentic workflows begins with solving the infrastructure problem first: traditional lakehouses are not suited for agent access patterns, but if we design one around transactions, governance follows. In particular, we draw an operational analogy to MVCC in databases and show why a direct transplant fails in a decoupled, multi-language setting. We then propose an agent-first design, Bauplan, that reimplements data and compute isolation in the lakehouse. We conclude by sharing a reference implementation of a self-healing pipeline in Bauplan, which seamlessly couples agent reasoning with all the desired guarantees for correctness and trust. 

**Abstract (ZH)**: 即使AI能力不断提高，大多数企业也不认为代理足够可信以处理生产数据。在本文中，我们argue认为，走向可信赖的代理工作流的路径是从解决基础设施问题开始：传统的湖库房不适用于代理的访问模式，但如果我们将它们设计为围绕交易进行，治理便会随之而来。特别是，我们将操作类比于数据库中的MVCC，并说明为什么直接移植会在解耦且多语言设置中失败。然后，我们提出了一种以代理为中心的设计Bauplan，该设计在湖库房中重新实现了数据和计算的隔离。最后，我们分享了Bauplan中一个自我修复管道的参考实现，该实现无缝地将代理推理与所有所需的正确性和可信度保证相结合。 

---
# CorrectHDL: Agentic HDL Design with LLMs Leveraging High-Level Synthesis as Reference 

**Title (ZH)**: CorrectHDL: 以高层次综合为参考的由自主agents编写的HDL设计 

**Authors**: Kangwei Xu, Grace Li Zhang, Ulf Schlichtmann, Bing Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.16395)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable potential in hardware front-end design using hardware description languages (HDLs). However, their inherent tendency toward hallucination often introduces functional errors into the generated HDL designs. To address this issue, we propose the framework CorrectHDL that leverages high-level synthesis (HLS) results as functional references to correct potential errors in LLM-generated HDL this http URL input to the proposed framework is a C/C++ program that specifies the target circuit's functionality. The program is provided to an LLM to directly generate an HDL design, whose syntax errors are repaired using a Retrieval-Augmented Generation (RAG) mechanism. The functional correctness of the LLM-generated circuit is iteratively improved by comparing its simulated behavior with an HLS reference design produced by conventional HLS tools, which ensures the functional correctness of the result but can lead to suboptimal area and power efficiency. Experimental results demonstrate that circuits generated by the proposed framework achieve significantly better area and power efficiency than conventional HLS designs and approach the quality of human-engineered circuits. Meanwhile, the correctness of the resulting HDL implementation is maintained, highlighting the effectiveness and potential of agentic HDL design leveraging the generative capabilities of LLMs and the rigor of traditional correctness-driven IC design flows. 

**Abstract (ZH)**: Large Language Models (LLMs)在硬件前端设计中的潜力已经得到彰显，特别是在硬件描述语言（HDLs）领域。然而，LLMs固有的虚幻倾向往往会引入功能错误到生成的HDL设计中。为了解决这一问题，我们提出了一种名为CorrectHDL的框架，利用高层次综合（HLS）结果作为功能参考来纠正LLM生成的HDL中的潜在错误。该框架的输入是一个C/C++程序，该程序指定了目标电路的功能。该程序提供给LLM直接生成HDL设计，通过检索增强生成（RAG）机制修复其语法错误。通过将生成的电路的模拟行为与由传统HLS工具产生的HLS参考设计进行比较，逐步改进生成电路的功能正确性，尽管这可能导致面积和能效不足。实验结果表明，由该框架生成的电路在面积和能效方面显著优于传统的HLS设计，并接近于手工设计电路的质量。同时，保持了生成HDL实现的正确性，突显了利用LLMs生成能力与传统正确性驱动的IC设计流程严谨性相结合的代理HDL设计的有效性和潜力。 

---
# An Agent-Based Framework for the Automatic Validation of Mathematical Optimization Models 

**Title (ZH)**: 基于代理的数学优化模型自动验证框架 

**Authors**: Alexander Zadorojniy, Segev Wasserkrug, Eitan Farchi  

**Link**: [PDF](https://arxiv.org/pdf/2511.16383)  

**Abstract**: Recently, using Large Language Models (LLMs) to generate optimization models from natural language descriptions has became increasingly popular. However, a major open question is how to validate that the generated models are correct and satisfy the requirements defined in the natural language description. In this work, we propose a novel agent-based method for automatic validation of optimization models that builds upon and extends methods from software testing to address optimization modeling . This method consists of several agents that initially generate a problem-level testing API, then generate tests utilizing this API, and, lastly, generate mutations specific to the optimization model (a well-known software testing technique assessing the fault detection power of the test suite). In this work, we detail this validation framework and show, through experiments, the high quality of validation provided by this agent ensemble in terms of the well-known software testing measure called mutation coverage. 

**Abstract (ZH)**: 最近，使用大型语言模型（LLMs）从自然语言描述生成优化模型的做法越来越流行。然而，一个主要的开放问题是如何验证生成的模型是否正确且满足自然语言描述中定义的要求。在本文中，我们提出了一种基于软件测试方法的新型代理方法，用于自动验证优化模型，该方法扩展了软件测试方法以应对优化建模需求。该方法包括多个代理，首先生成一个问题级别测试API，然后利用此API生成测试，最后生成针对优化模型的具体变异（一种公认的方法，用于评估测试套件的故障检测能力）。在本文中，我们详细介绍了这一验证框架，并通过实验展示了该代理集合在公认的软件测试度量标准——变异覆盖率方面的高质量验证结果。 

---
# Reducing Instability in Synthetic Data Evaluation with a Super-Metric in MalDataGen 

**Title (ZH)**: 使用Super-Metric在MalDataGen中减少合成数据评估的不稳定性 

**Authors**: Anna Luiza Gomes da Silva, Diego Kreutz, Angelo Diniz, Rodrigo Mansilha, Celso Nobre da Fonseca  

**Link**: [PDF](https://arxiv.org/pdf/2511.16373)  

**Abstract**: Evaluating the quality of synthetic data remains a persistent challenge in the Android malware domain due to instability and the lack of standardization among existing metrics. This work integrates into MalDataGen a Super-Metric that aggregates eight metrics across four fidelity dimensions, producing a single weighted score. Experiments involving ten generative models and five balanced datasets demonstrate that the Super-Metric is more stable and consistent than traditional metrics, exhibiting stronger correlations with the actual performance of classifiers. 

**Abstract (ZH)**: 评估合成数据的质量仍是在Android恶意软件领域的一项持久挑战，由于现有指标的不稳定性和缺乏标准化。本研究在MalDataGen中整合了一个超级指标，该指标在四个保真度维度上聚合了八种指标，生成了一个加权分数。涉及十种生成模型和五个平衡数据集的实验表明，超级指标比传统指标更为稳定和一致，与分类器的实际性能展现出更强的相关性。 

---
# OpenMMReasoner: Pushing the Frontiers for Multimodal Reasoning with an Open and General Recipe 

**Title (ZH)**: OpenMMReasoner: 推动多模态推理前沿的开放通用方法 

**Authors**: Kaichen Zhang, Keming Wu, Zuhao Yang, Kairui Hu, Bin Wang, Ziwei Liu, Xingxuan Li, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2511.16334)  

**Abstract**: Recent advancements in large reasoning models have fueled growing interest in extending such capabilities to multimodal domains. However, despite notable progress in visual reasoning, the lack of transparent and reproducible data curation and training strategies remains a major barrier to scalable research. In this work, we introduce OpenMMReasoner, a fully transparent two-stage recipe for multimodal reasoning spanning supervised fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we construct an 874K-sample cold-start dataset with rigorous step-by-step validation, providing a strong foundation for reasoning capabilities. The subsequent RL stage leverages a 74K-sample dataset across diverse domains to further sharpen and stabilize these abilities, resulting in a more robust and efficient learning process. Extensive evaluations demonstrate that our training recipe not only surpasses strong baselines but also highlights the critical role of data quality and training design in shaping multimodal reasoning performance. Notably, our method achieves a 11.6% improvement over the Qwen2.5-VL-7B-Instruct baseline across nine multimodal reasoning benchmarks, establishing a solid empirical foundation for future large-scale multimodal reasoning research. We open-sourced all our codes, pipeline, and data at this https URL. 

**Abstract (ZH)**: Recent advancements in large reasoning models have fueled growing interest in extending such capabilities to multimodal domains.然而，尽管在视觉推理方面取得了显著进展，但 lack of transparent and reproducible data curation and training strategies 仍是一大障碍，阻碍了可扩展的研究。在这项工作中，我们介绍了 OpenMMReasoner，这是一个全面透明的两阶段多模态推理配方，涵盖了监督微调（SFT）和强化学习（RL）。在SFT阶段，我们构建了一个经过严格一步步验证的87.4万样本冷启动数据集，为推理能力奠定了坚实基础。随后的RL阶段利用了跨多个领域的7.4万样本数据集，进一步强化并稳定了这些能力，从而实现了更 robust 和高效的训练过程。广泛的评估表明，我们的训练配方不仅超越了强大的基线模型，还强调了数据质量和训练设计在塑造多模态推理性能中的关键作用。值得注意的是，我们的方法在九个多模态推理基准中比 Qwen2.5-VL-7B-Instruct 基线模型实现了11.6%的改进，并为未来大规模多模态推理研究奠定了坚实的经验基础。我们已开源了所有代码、流水线和数据。 

---
# Distributed Agent Reasoning Across Independent Systems With Strict Data Locality 

**Title (ZH)**: 跨独立系统中的严格数据局部性分布式代理推理 

**Authors**: Daniel Vaughan, Kateřina Vaughan  

**Link**: [PDF](https://arxiv.org/pdf/2511.16292)  

**Abstract**: This paper presents a proof-of-concept demonstration of agent-to-agent communication across distributed systems, using only natural-language messages and without shared identifiers, structured schemas, or centralised data exchange. The prototype explores how multiple organisations (represented here as a Clinic, Insurer, and Specialist Network) can cooperate securely via pseudonymised case tokens, local data lookups, and controlled operational boundaries.
The system uses Orpius as the underlying platform for multi-agent orchestration, tool execution, and privacy-preserving communication. All agents communicate through OperationRelay calls, exchanging concise natural-language summaries. Each agent operates on its own data (such as synthetic clinic records, insurance enrolment tables, and clinical guidance extracts), and none receives or reconstructs patient identity. The Clinic computes an HMAC-based pseudonymous token, the Insurer evaluates coverage rules and consults the Specialist agent, and the Specialist returns an appropriateness recommendation.
The goal of this prototype is intentionally limited: to demonstrate feasibility, not to provide a clinically validated, production-ready system. No clinician review was conducted, and no evaluation beyond basic functional runs was performed. The work highlights architectural patterns, privacy considerations, and communication flows that enable distributed reasoning among specialised agents while keeping data local to each organisation. We conclude by outlining opportunities for more rigorous evaluation and future research in decentralised multi-agent systems. 

**Abstract (ZH)**: 基于自然语言的分布式系统中代理间通信的概念验证：通过伪匿名病例标识、本地数据查找和可控操作边界实现多方安全合作 

---
# MuISQA: Multi-Intent Retrieval-Augmented Generation for Scientific Question Answering 

**Title (ZH)**: MuISQA: 基于多意图检索增强的科学问答生成 

**Authors**: Zhiyuan Li, Haisheng Yu, Guangchuan Guo, Nan Zhou, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16283)  

**Abstract**: Complex scientific questions often entail multiple intents, such as identifying gene mutations and linking them to related diseases. These tasks require evidence from diverse sources and multi-hop reasoning, while conventional retrieval-augmented generation (RAG) systems are usually single-intent oriented, leading to incomplete evidence coverage. To assess this limitation, we introduce the Multi-Intent Scientific Question Answering (MuISQA) benchmark, which is designed to evaluate RAG systems on heterogeneous evidence coverage across sub-questions. In addition, we propose an intent-aware retrieval framework that leverages large language models (LLMs) to hypothesize potential answers, decompose them into intent-specific queries, and retrieve supporting passages for each underlying intent. The retrieved fragments are then aggregated and re-ranked via Reciprocal Rank Fusion (RRF) to balance coverage across diverse intents while reducing redundancy. Experiments on both MuISQA benchmark and other general RAG datasets demonstrate that our method consistently outperforms conventional approaches, particularly in retrieval accuracy and evidence coverage. 

**Abstract (ZH)**: 复杂科学问题往往包含多重意图，如识别基因突变并将其与相关疾病联系起来。这些任务需要来自多种来源和多跳推理的证据，而传统的检索增强生成（RAG）系统通常是单意图导向的，导致证据覆盖不完整。为了评估这一局限性，我们提出了多元意图科学问答（MuISQA）基准，旨在评估RAG系统在子问题跨异质证据覆盖方面的表现。此外，我们提出了一种意图感知的检索框架，利用大规模语言模型（LLMs）进行潜在答案的假设，将其分解为特定意图的查询，并检索每个底层意图的支持段落。检索到的片段然后通过互惠排名融合（RRF）进行聚合和重新排名，以平衡多元意图之间的覆盖范围同时减少冗余。在MuISQA基准和其他一般RAG数据集上的实验表明，我们的方法在检索准确性和证据覆盖方面一贯优于传统方法。 

---
# Revisiting Fairness-aware Interactive Recommendation: Item Lifecycle as a Control Knob 

**Title (ZH)**: 重新评估基于公平性的交互式推荐：物品生命周期作为控制旋钮 

**Authors**: Yun Lu, Xiaoyu Shi, Hong Xie, Chongjun Xia, Zhenhui Gong, Mingsheng Shang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16248)  

**Abstract**: This paper revisits fairness-aware interactive recommendation (e.g., TikTok, KuaiShou) by introducing a novel control knob, i.e., the lifecycle of items. We make threefold contributions. First, we conduct a comprehensive empirical analysis and uncover that item lifecycles in short-video platforms follow a compressed three-phase pattern, i.e., rapid growth, transient stability, and sharp decay, which significantly deviates from the classical four-stage model (introduction, growth, maturity, decline). Second, we introduce LHRL, a lifecycle-aware hierarchical reinforcement learning framework that dynamically harmonizes fairness and accuracy by leveraging phase-specific exposure dynamics. LHRL consists of two key components: (1) PhaseFormer, a lightweight encoder combining STL decomposition and attention mechanisms for robust phase detection; (2) a two-level HRL agent, where the high-level policy imposes phase-aware fairness constraints, and the low-level policy optimizes immediate user engagement. This decoupled optimization allows for effective reconciliation between long-term equity and short-term utility. Third, experiments on multiple real-world interactive recommendation datasets demonstrate that LHRL significantly improves both fairness and user engagement. Furthermore, the integration of lifecycle-aware rewards into existing RL-based models consistently yields performance gains, highlighting the generalizability and practical value of our approach. 

**Abstract (ZH)**: 本文通过引入新的控制旋钮即项目的生命周期，重新审视公平感知的互动推荐（如抖音、快手等）。我们做出了三方面的贡献。首先，我们进行了全面的经验分析，发现短视频平台上的项目生命周期遵循压缩的三阶段模式，即快速增长、瞬时稳定和急剧衰减，这与经典的四阶段模型（引入期、增长期、成熟期、衰退期）有显著差异。其次，我们引入了LHRL，这是一种生命周期感知的层次强化学习框架，通过利用阶段特异性的曝光动态来动态平衡公平和准确性。LHRL 包含两个关键组件：(1) PhaseFormer，一种结合STL分解和注意力机制的轻量级编码器，用于稳健的阶段检测；(2) 两级层次强化学习代理，其中高级策略施加阶段感知的公平约束，而低级策略优化即时用户参与度。这种分离的优化允许有效地平衡长期公平性和短期实用性。第三，对多个现实世界的互动推荐数据集的实验表明，LHRL 显著提高了公平性和用户参与度。此外，生命周期感知奖励的集成到现有的基于RL的模型中始终带来了性能提升，突显了我们方法的通用性和实际价值。 

---
# FlipVQA-Miner: Cross-Page Visual Question-Answer Mining from Textbooks 

**Title (ZH)**: 翻转VQA矿工：从教科书中跨页视觉问答挖掘 

**Authors**: Zhen Hao Wong, Jingwen Deng, Hao Liang, Runming He, Chengyu Shen, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16216)  

**Abstract**: The development of Large Language Models (LLMs) increasingly depends on high-quality supervised data, yet existing instruction-tuning and RL datasets remain costly to curate and often rely on synthetic samples that introduce hallucination and limited diversity. At the same time, textbooks and exercise materials contain abundant, high-quality human-authored Question-Answer(QA) content that remains underexploited due to the difficulty of transforming raw PDFs into AI-ready supervision. Although modern OCR and vision-language models can accurately parse document structure, their outputs lack the semantic alignment required for training. We propose an automated pipeline that extracts well-formed QA and visual-QA (VQA) pairs from educational documents by combining layout-aware OCR with LLM-based semantic parsing. Experiments across diverse document types show that the method produces accurate, aligned, and low-noise QA/VQA pairs. This approach enables scalable use of real-world educational content and provides a practical alternative to synthetic data generation for improving reasoning-oriented LLM training. All code and data-processing pipelines are open-sourced at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）的发展越来越依赖高质量的监督数据，然而现有的指令调优和强化学习数据集在维护上依然成本高昂，并且常常依赖合成样本，这会导致幻觉和多样性有限的问题。同时，教科书和练习材料中包含大量高质量的人工撰写的问题-答案（QA）内容，但由于将原始PDF转换为AI可用的监督数据的难度较大，这些内容仍然没有得到充分开发利用。尽管现代OCR和跨模态模型能够准确解析文档结构，但其输出缺乏用于训练所需的语义对齐。我们提出了一种自动化流水线，通过结合布局感知的OCR与基于LLM的语义解析，从教育文档中提取良好的问题-答案（QA）和视觉-问题-答案（VQA）对。实验表明，该方法生成的QA/VQA对准确、对齐且噪声较低。此方法使得实际教育内容的大规模利用成为可能，并提供了一种改进以推理为导向的LLM训练的实用替代方案，合成数据生成。所有代码和数据处理流水线均已开源。 

---
# ChemLabs on ChemO: A Multi-Agent System for Multimodal Reasoning on IChO 2025 

**Title (ZH)**: ChemLabs on ChemO：适用于2025年国际化学奥林匹克多模态推理的多代理系统 

**Authors**: Xu Qiang, Shengyuan Bai, Leqing Chen, Zijing Liu, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.16205)  

**Abstract**: Olympiad-level benchmarks in mathematics and physics are crucial testbeds for advanced AI reasoning, but chemistry, with its unique multimodal symbolic language, has remained an open challenge. We introduce ChemO, a new benchmark built from the International Chemistry Olympiad (IChO) 2025. ChemO features two key innovations for automated assessment: Assessment-Equivalent Reformulation (AER), which converts problems requiring visual outputs (e.g., drawing molecules) into computationally tractable formats, and Structured Visual Enhancement (SVE), a diagnostic mechanism to disentangle a model's visual perception capabilities from its core chemical reasoning. To tackle this benchmark, we propose ChemLabs, a hierarchical multi-agent framework that mimics human expert collaboration through specialized agents for problem decomposition, perception, reasoning, and auditing. Experiments on state-of-the-art multimodal models demonstrate that combining SVE with our multi-agent system yields dramatic performance gains. Our top configuration achieves a score of 93.6 out of 100, surpassing an estimated human gold medal threshold and establishing a new state-of-the-art in automated chemical problem-solving. ChemO Dataset: this https URL 

**Abstract (ZH)**: 奥林匹克级别的化学和物理基准对于高级AI推理至关重要，但具有独特多模态符号语言的化学领域仍是一个开放的挑战。我们介绍了ChemO，一个基于2025年国际化学奥林匹克(IChO)的新基准。ChemO包含两项自动评估的关键创新：评估等效重述(AER)，将需要视觉输出的问题（例如，绘制分子）转换为计算上可处理的格式；结构化视觉增强(SVE)，一种诊断机制，用于分离模型的视觉感知能力与其核心化学推理能力。为应对这一基准，我们提出了ChemLabs，一种模拟人类专家合作的分层多Agent框架，通过专门的Agent进行问题分解、感知、推理和审核。在最先进的多模态模型上的实验表明，将SVE与我们的多Agent系统结合使用可取得显著的性能提升。我们的最佳配置的得分为93.6/100，超过了一个估计的人类金牌阈值，并建立了自动化学问题解决的新最先进的水平。ChemO数据集：this https URL。 

---
# Multi-Agent Collaborative Reward Design for Enhancing Reasoning in Reinforcement Learning 

**Title (ZH)**: 多代理协作奖励设计以增强强化学习中的推理能力 

**Authors**: Pei Yang, Ke Zhang, Ji Wang, Xiao Chen, Yuxin Tang, Eric Yang, Lynn Ai, Bill Shi  

**Link**: [PDF](https://arxiv.org/pdf/2511.16202)  

**Abstract**: We present CRM (Multi-Agent Collaborative Reward Model), a framework that replaces a single black-box reward model with a coordinated team of specialist evaluators to improve robustness and interpretability in RLHF. Conventional reward models struggle to jointly optimize multiple, sometimes conflicting, preference dimensions (e.g., factuality, helpfulness, safety) and offer limited transparency into why a score is assigned. CRM addresses these issues by decomposing preference evaluation into domain-specific agents that each produce partial signals, alongside global evaluators such as ranker-based and embedding-similarity rewards. A centralized aggregator fuses these signals at each timestep, balancing factors like step-wise correctness, multi-agent agreement, and repetition penalties, yielding a single training reward compatible with standard RL pipelines. The policy is optimized with advantage-based updates (e.g., GAE), while a value model regresses to the aggregated reward, enabling multi-perspective reward shaping without requiring additional human annotations beyond those used to train the evaluators. To support training and assessment, we introduce rewardBench, a benchmark and training suite aligned with the collaborative structure of CRM. Together, CRM and rewardBench provide a practical, modular path to more transparent reward modeling and more stable optimization. 

**Abstract (ZH)**: CRM（多Agent协作奖励模型）：一种框架，用协调的专家评估团队取代单一的黑盒奖励模型，以提高RLHF的稳健性和可解释性 

---
# From Performance to Understanding: A Vision for Explainable Automated Algorithm Design 

**Title (ZH)**: 从性能到理解：可解释自动化算法设计的愿景 

**Authors**: Niki van Stein, Anna V. Kononova, Thomas Bäck  

**Link**: [PDF](https://arxiv.org/pdf/2511.16201)  

**Abstract**: Automated algorithm design is entering a new phase: Large Language Models can now generate full optimisation (meta)heuristics, explore vast design spaces and adapt through iterative feedback. Yet this rapid progress is largely performance-driven and opaque. Current LLM-based approaches rarely reveal why a generated algorithm works, which components matter or how design choices relate to underlying problem structures. This paper argues that the next breakthrough will come not from more automation, but from coupling automation with understanding from systematic benchmarking. We outline a vision for explainable automated algorithm design, built on three pillars: (i) LLM-driven discovery of algorithmic variants, (ii) explainable benchmarking that attributes performance to components and hyperparameters and (iii) problem-class descriptors that connect algorithm behaviour to landscape structure. Together, these elements form a closed knowledge loop in which discovery, explanation and generalisation reinforce each other. We argue that this integration will shift the field from blind search to interpretable, class-specific algorithm design, accelerating progress while producing reusable scientific insight into when and why optimisation strategies succeed. 

**Abstract (ZH)**: 自动化算法设计正进入新阶段：大型语言模型现在可以生成完整的优化（元）启发式算法、探索广阔的设计空间并通过对迭代反馈进行适应。然而，这种 rapid 进步主要是基于性能且不透明。当前基于大型语言模型的方法很少解释生成的算法为何有效、哪些组件重要或设计选择如何与基础问题结构相关。本文认为，下一突破将不是来自于更多的自动化，而是来自于将自动化与系统基准测试所获得的理解相结合。我们提出了可解释的自动化算法设计愿景，基于三个支柱：（i）由大型语言模型驱动的算法变体发现，（ii）可解释的基准测试，能够将性能归因于组件和超参数，以及（iii）问题类别描述符，将算法行为与景观结构连接起来。这些要素共同形成一个闭环知识循环，在此循环中，发现、解释和概括相互强化。我们认为，这种整合将使该领域从盲搜索转变为可解释的、类别特定的算法设计，加速进展并产生可重复的科学见解，说明何时及为何优化策略能成功。 

---
# FOOTPASS: A Multi-Modal Multi-Agent Tactical Context Dataset for Play-by-Play Action Spotting in Soccer Broadcast Videos 

**Title (ZH)**: FOOTPASS：一个用于足球广播视频实时描述动作识别的多模态多agent战术上下文数据集 

**Authors**: Jeremie Ochin, Raphael Chekroun, Bogdan Stanciulescu, Sotiris Manitsaris  

**Link**: [PDF](https://arxiv.org/pdf/2511.16183)  

**Abstract**: Soccer video understanding has motivated the creation of datasets for tasks such as temporal action localization, spatiotemporal action detection (STAD), or multiobject tracking (MOT). The annotation of structured sequences of events (who does what, when, and where) used for soccer analytics requires a holistic approach that integrates both STAD and MOT. However, current action recognition methods remain insufficient for constructing reliable play-by-play data and are typically used to assist rather than fully automate annotation. Parallel research has advanced tactical modeling, trajectory forecasting, and performance analysis, all grounded in game-state and play-by-play data. This motivates leveraging tactical knowledge as a prior to support computer-vision-based predictions, enabling more automated and reliable extraction of play-by-play data. We introduce Footovision Play-by-Play Action Spotting in Soccer Dataset (FOOTPASS), the first benchmark for play-by-play action spotting over entire soccer matches in a multi-modal, multi-agent tactical context. It enables the development of methods for player-centric action spotting that exploit both outputs from computer-vision tasks (e.g., tracking, identification) and prior knowledge of soccer, including its tactical regularities over long time horizons, to generate reliable play-by-play data streams. These streams form an essential input for data-driven sports analytics. 

**Abstract (ZH)**: 足球视频理解促进了针对时间动作定位、空间时间动作检测（STAD）或多目标跟踪（MOT）等任务的数据集的创建。Footovision Play-by-Play Action Spotting in Soccer Dataset (FOOTPASS)：足球比赛中全方位多模态多agent战术背景下行踪动作识别基准。 

---
# Multidimensional Rubric-oriented Reward Model Learning via Geometric Projection Reference Constraints 

**Title (ZH)**: 基于几何投影参考约束的多维度评标导向奖励模型学习 

**Authors**: Yongnan Jin, Xurui Li, Feng Cao, Liucun Gao, Juanjuan Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.16139)  

**Abstract**: The integration of large language models (LLMs) into medical practice holds transformative potential, yet their real-world clinical utility remains limited by critical alignment challenges: (1) a disconnect between static evaluation benchmarks and dynamic clinical cognitive needs, (2) difficulties in adapting to evolving, multi-source medical standards, and (3) the inability of conventional reward models to capture nuanced, multi-dimensional medical quality criteria. To address these gaps, we propose MR-RML (Multidimensional Rubric-oriented Reward Model Learning) via GPRC (Geometric Projection Reference Constraints), a novel alignment framework that integrates medical standards into a structured "Dimensions-Scenarios-Disciplines" matrix to guide data generation and model optimization. MR-RML introduces three core innovations: (1) a "Dimensions-Scenarios-Disciplines" medical standard system that embeds domain standards into the full training pipeline; (2) an independent multi-dimensional reward model that decomposes evaluation criteria, shifting from real-time rubric-based scoring to internalized reward modeling for improved consistency and cost-efficiency; (3) geometric projection reference constraints that transform medical cognitive logic into mathematical regularization, aligning scoring gradients with clinical reasoning and enabling synthetic data-driven training. Through extensive evaluations on the authoritative medical benchmark Healthbench, our method yields substantial performance gains over the base LLM Qwen-32B (45% on the full subset and 85% on Hard subset, respectively). It achieves a SOTA among open-source LLMs with scores of 62.7 (full subset) and 44.7 (hard subset), while also outperforming the majority of closed-source models. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医疗实践中的集成具有变革潜力，但其实用性受限于关键的对齐挑战：（1）静态评估标准与动态临床认知需求之间的脱节，（2）难以适应不断演变的多源医疗标准，（3）传统奖励模型无法捕捉复杂的多维度医疗质量标准。为了解决这些问题，我们提出了一种名为MR-RML（基于多维度标准导向奖励模型学习）的新颖对齐框架，该框架通过GPRC（几何投影参考约束）将医疗标准整合到一个结构化的“维度-场景-学科”矩阵中，以指导数据生成和模型优化。MR-RML引入了三项核心创新：（1）一个“维度-场景-学科”医疗标准系统，将领域标准嵌入到完整的训练管道中；（2）一个独立的多维度奖励模型，将评估标准分解，从实时标准评分转向内部化奖励建模，提高一致性和成本效益；（3）几何投影参考约束，将医疗认知逻辑转换为数学正则化，使评分梯度与临床推理对齐，并实现基于合成数据的训练。通过在权威医疗基准Healthbench上的广泛评估，该方法在基模型Qwen-32B的基础上取得了显著的表现提升（全子集45%，硬子集85%）。在开源模型中，其得分为62.7（全子集）和44.7（硬子集），并优于大多数封闭源模型。 

---
# SkyRL-Agent: Efficient RL Training for Multi-turn LLM Agent 

**Title (ZH)**: SkyRL-Agent: 高效的多轮对话大语言模型代理的RL训练方法 

**Authors**: Shiyi Cao, Dacheng Li, Fangzhou Zhao, Shuo Yuan, Sumanth R. Hegde, Connor Chen, Charlie Ruan, Tyler Griggs, Shu Liu, Eric Tang, Richard Liaw, Philipp Moritz, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2511.16108)  

**Abstract**: We introduce SkyRL-Agent, a framework for efficient, multi-turn, long-horizon agent training and evaluation. It provides efficient asynchronous dispatching, lightweight tool integration, and flexible backend interoperability, enabling seamless use with existing RL frameworks such as SkyRL-train, VeRL, and Tinker.
Using SkyRL-Agent, we train SA-SWE-32B, a software engineering agent trained from Qwen3-32B (24.4% Pass@1) purely with reinforcement learning. We introduce two key components: an optimized asynchronous pipeline dispatcher that achieves a 1.55x speedup over naive asynchronous batching, and a tool-enhanced training recipe leveraging an AST-based search tool to facilitate code navigation, boost rollout Pass@K, and improve training efficiency. Together, these optimizations enable SA-SWE-32B to reach 39.4% Pass@1 on SWE-Bench Verified with more than 2x cost reduction compared to prior models reaching similar performance. Despite being trained solely on SWE tasks, SA-SWE-32B generalizes effectively to other agentic tasks, including Terminal-Bench, BrowseComp-Plus, and WebArena. We further demonstrate SkyRL-Agent's extensibility through case studies on deep research, computer use, and memory agents, each trained using a different training backend. 

**Abstract (ZH)**: SkyRL-Agent：一种高效的多轮长时间代理训练与评估框架 

---
# A Hybrid Proactive And Predictive Framework For Edge Cloud Resource Management 

**Title (ZH)**: 边缘云资源管理的混合主动预测框架 

**Authors**: Hrikshesh Kumar, Anika Garg, Anshul Gupta, Yashika Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2511.16075)  

**Abstract**: Old cloud edge workload resource management is too reactive. The problem with relying on static thresholds is that we are either overspending for more resources than needed or have reduced performance because of their lack. This is why we work on proactive solutions. A framework developed for it stops reacting to the problems but starts expecting them. We design a hybrid architecture, combining two powerful tools: the CNN LSTM model for time series forecasting and an orchestrator based on multi agent Deep Reinforcement Learning In fact the novelty is in how we combine them as we embed the predictive forecast from the CNN LSTM directly into the DRL agent state space. That is what makes the AI manager smarter it sees the future, which allows it to make better decisions about a long term plan for where to run tasks That means finding that sweet spot between how much money is saved while keeping the system healthy and apps fast for users That is we have given it eyes in order to see down the road so that it does not have to lurch from one problem to another it finds a smooth path forward Our tests show our system easily beats the old methods It is great at solving tough problems like making complex decisions and juggling multiple goals at once like being cheap fast and reliable 

**Abstract (ZH)**: 基于CNN LSTM和多代理深度强化学习的主动边缘工作负载资源管理框架 

---
# Artificial Intelligence and Accounting Research: A Framework and Agenda 

**Title (ZH)**: 人工智能与会计研究：框架与议程 

**Authors**: Theophanis C. Stratopoulos, Victor Xiaoqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16055)  

**Abstract**: Recent advances in artificial intelligence, particularly generative AI (GenAI) and large language models (LLMs), are fundamentally transforming accounting research, creating both opportunities and competitive threats for scholars. This paper proposes a framework that classifies AI-accounting research along two dimensions: research focus (accounting-centric versus AI-centric) and methodological approach (AI-based versus traditional methods). We apply this framework to papers from the IJAIS special issue and recent AI-accounting research published in leading accounting journals to map existing studies and identify research opportunities. Using this same framework, we analyze how accounting researchers can leverage their expertise through strategic positioning and collaboration, revealing where accounting scholars' strengths create the most value. We further examine how GenAI and LLMs transform the research process itself, comparing the capabilities of human researchers and AI agents across the entire research workflow. This analysis reveals that while GenAI democratizes certain research capabilities, it simultaneously intensifies competition by raising expectations for higher-order contributions where human judgment, creativity, and theoretical depth remain valuable. These shifts call for reforming doctoral education to cultivate comparative advantages while building AI fluency. 

**Abstract (ZH)**: 近年来，人工智能的最新进展，特别是生成人工智能（GenAI）和大型语言模型（LLMs），从根本上变革了会计研究，为学者们创造了机遇与竞争威胁。本文提出了一种框架，从两个维度对AI-会计研究进行分类：研究焦点（会计中心化与AI中心化）和方法论路径（基于AI的方法与传统方法）。我们通过将此框架应用于IJAIS的特刊论文和顶尖会计期刊上最近的AI-会计研究，绘制现有研究图谱并识别研究机遇。利用同一框架，我们分析了会计研究者如何通过战略定位和合作发挥其专长，揭示了会计学者的优势领域创造的最大价值。进一步探讨了GenAI和LLMs如何本身变革研究过程，比较了人类研究人员与AI代理在整个研究工作流程中的能力差异。这一分析表明，虽然GenAI普及了某些研究能力，但它同时加剧了竞争，因为人类判断、创造力和理论深度仍具有重要价值。这些变化要求改革博士教育，培养比较优势并增强AI技能。 

---
# An Aligned Constraint Programming Model For Serial Batch Scheduling With Minimum Batch Size 

**Title (ZH)**: 面向最小批量尺寸的对齐约束编程模型序列批调度 

**Authors**: Jorge A. Huertas, Pascal Van Hentenryck  

**Link**: [PDF](https://arxiv.org/pdf/2511.16045)  

**Abstract**: In serial batch (s-batch) scheduling, jobs from similar families are grouped into batches and processed sequentially to avoid repetitive setups that are required when processing consecutive jobs of different families. Despite its large success in scheduling, only three Constraint Programming (CP) models have been proposed for this problem considering minimum batch sizes, which is a common requirement in many practical settings, including the ion implantation area in semiconductor manufacturing. These existing CP models rely on a predefined virtual set of possible batches that suffers from the curse of dimensionality and adds complexity to the problem. This paper proposes a novel CP model that does not rely on this virtual set. Instead, it uses key alignment parameters that allow it to reason directly on the sequences of same-family jobs scheduled on the machines, resulting in a more compact formulation. This new model is further improved by exploiting the problem's structure with tailored search phases and strengthened inference levels of the constraint propagators. The extensive computational experiments on nearly five thousand instances compare the proposed models against existing methods in the literature, including mixed-integer programming formulations, tabu search meta-heuristics, and CP approaches. The results demonstrate the superiority of the proposed models on small-to-medium instances with up to 100 jobs, and their ability to find solutions up to 25\% better than the ones produces by existing methods on large-scale instances with up to 500 jobs, 10 families, and 10 machines. 

**Abstract (ZH)**: 基于序列批次（s-批）调度的新型约束编程模型 

---
# SpellForger: Prompting Custom Spell Properties In-Game using BERT supervised-trained model 

**Title (ZH)**: SpellForger: 使用BERT监督训练模型在游戏中自定义施法属性 

**Authors**: Emanuel C. Silva, Emily S. M. Salum, Gabriel M. Arantes, Matheus P. Pereira, Vinicius F. Oliveira, Alessandro L. Bicho  

**Link**: [PDF](https://arxiv.org/pdf/2511.16018)  

**Abstract**: Introduction: The application of Artificial Intelligence in games has evolved significantly, allowing for dynamic content generation. However, its use as a core gameplay co-creation tool remains underexplored. Objective: This paper proposes SpellForger, a game where players create custom spells by writing natural language prompts, aiming to provide a unique experience of personalization and creativity. Methodology: The system uses a supervisedtrained BERT model to interpret player prompts. This model maps textual descriptions to one of many spell prefabs and balances their parameters (damage, cost, effects) to ensure competitive integrity. The game is developed in the Unity Game Engine, and the AI backend is in Python. Expected Results: We expect to deliver a functional prototype that demonstrates the generation of spells in real time, applied to an engaging gameplay loop, where player creativity is central to the experience, validating the use of AI as a direct gameplay mechanic. 

**Abstract (ZH)**: 介绍：人工智能在游戏中的应用已经显著发展，使得动态内容生成成为可能。然而，将其作为核心游戏共创工具的应用仍然未被充分探索。目标：本文提出了一款名为SpellForger的游戏，玩家通过编写自然语言提示来创建自定义法术，旨在提供一种独特的个性化和创造力体验。方法：系统使用监督训练的BERT模型来解释玩家提示，将文本描述映射到多个法术预制件之一，并平衡其参数（伤害、成本、效果）以确保竞争公平性。游戏使用Unity游戏引擎开发，AI后台使用Python编写。预期结果：我们期望交付一个功能原型，该原型能够实时生成法术，并应用于一个以玩家创造力为中心的吸引人的游戏循环中，从而验证AI作为直接游戏机制的使用价值。 

---
# MUSEKG: A Knowledge Graph Over Museum Collections 

**Title (ZH)**: MUSEKG：博物馆藏品的知识图谱 

**Authors**: Jinhao Li, Jianzhong Qi, Soyeon Caren Han, Eun-Jung Holden  

**Link**: [PDF](https://arxiv.org/pdf/2511.16014)  

**Abstract**: Digital transformation in the cultural heritage sector has produced vast yet fragmented collections of artefact data. Existing frameworks for museum information systems struggle to integrate heterogeneous metadata, unstructured documents, and multimodal artefacts into a coherent and queryable form. We present MuseKG, an end-to-end knowledge-graph framework that unifies structured and unstructured museum data through symbolic-neural integration. MuseKG constructs a typed property graph linking objects, people, organisations, and visual or textual labels, and supports natural language queries. Evaluations on real museum collections demonstrate robust performance across queries over attributes, relations, and related entities, surpassing large-language-model zero-shot, few-shot and SPARQL prompt baselines. The results highlight the importance of symbolic grounding for interpretable and scalable cultural heritage reasoning, and pave the way for web-scale integration of digital heritage knowledge. 

**Abstract (ZH)**: 文化 heritage 领域的数字转型产生了大量yet 分散的 artefact 数据。现有的博物馆信息系统框架难以将异构元数据、非结构化文档和多模态 artefact 整合成一致且可查询的形式。我们提出 MuseKG，这是一种端到端的知识图谱框架，通过符号-神经集成统一结构化和非结构化博物馆数据。MuseKG 构建了一种带有类型属性的图形，将对象、人员、组织及相关视觉或文本标签连接起来，并支持自然语言查询。实证研究表明，MuseKG 在属性、关系及相关实体的查询中表现出稳健的性能，超越了大型语言模型的零样本、少样本和 SPARQL 提示基准。结果突显了符号定位对于可解释和可扩展的文化遗产推理的重要性，并为实现数字遗产知识的大规模集成铺平了道路。 

---
# Sensorium Arc: AI Agent System for Oceanic Data Exploration and Interactive Eco-Art 

**Title (ZH)**: Sensorium Arc：海洋数据探索与互动生态艺术的AI代理系统 

**Authors**: Noah Bissell, Ethan Paley, Joshua Harrison, Juliano Calil, Myungin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.15997)  

**Abstract**: Sensorium Arc (AI reflects on climate) is a real-time multimodal interactive AI agent system that personifies the ocean as a poetic speaker and guides users through immersive explorations of complex marine data. Built on a modular multi-agent system and retrieval-augmented large language model (LLM) framework, Sensorium enables natural spoken conversations with AI agents that embodies the ocean's perspective, generating responses that blend scientific insight with ecological poetics. Through keyword detection and semantic parsing, the system dynamically triggers data visualizations and audiovisual playback based on time, location, and thematic cues drawn from the dialogue. Developed in collaboration with the Center for the Study of the Force Majeure and inspired by the eco-aesthetic philosophy of Newton Harrison, Sensorium Arc reimagines ocean data not as an abstract dataset but as a living narrative. The project demonstrates the potential of conversational AI agents to mediate affective, intuitive access to high-dimensional environmental data and proposes a new paradigm for human-machine-ecosystem. 

**Abstract (ZH)**: Sensorium Arc (AI reflects on climate) 是一个实时多模态交互 AI 代理系统，将其视为诗意的演讲者，以海洋为中心，引导用户进行沉浸式探索复杂的海洋数据。该系统基于模块化的多代理系统和检索增强的大语言模型框架构建，使用户能够与包含海洋视角的 AI 代理进行自然对话，生成融合科学洞察与生态诗学的回应。通过关键词检测和语义解析，系统能够根据对话中的时间、地点和主题提示，动态触发数据可视化和音视频播放。该项目由研究中心力不可测研究所合作开发，并受到Newton Harrison的生态美学理念的启发，重新诠释海洋数据，将其视为一个活生生的叙述，而非抽象的数据集。该项目展示了对话式 AI 代理在提供情感化的、直觉化的高维环境数据访问方面的潜力，并提出了一种人机生态系统的新范式。 

---
# CARE-RAG - Clinical Assessment and Reasoning in RAG 

**Title (ZH)**: CARE-RAG - 临床评估与推理在RAG中的应用 

**Authors**: Deepthi Potluri, Aby Mammen Mathew, Jeffrey B DeWitt, Alexander L. Rasgon, Yide Hao, Junyuan Hong, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2511.15994)  

**Abstract**: Access to the right evidence does not guarantee that large language models (LLMs) will reason with it correctly. This gap between retrieval and reasoning is especially concerning in clinical settings, where outputs must align with structured protocols. We study this gap using Written Exposure Therapy (WET) guidelines as a testbed. In evaluating model responses to curated clinician-vetted questions, we find that errors persist even when authoritative passages are provided. To address this, we propose an evaluation framework that measures accuracy, consistency, and fidelity of reasoning. Our results highlight both the potential and the risks: retrieval-augmented generation (RAG) can constrain outputs, but safe deployment requires assessing reasoning as rigorously as retrieval. 

**Abstract (ZH)**: 获取正确的证据并不保证大语言模型（LLMs）能够正确地进行推理。这种从检索到推理之间的差距，在临床环境中尤为令人担忧，因为在这些环境中，输出必须与结构化协议保持一致。我们使用书写暴露疗法（WET）指南作为试验平台，研究这一差距。在评估模型对经过临床专家筛选的问题作出的回答时，即使提供了权威的段落，我们也发现错误仍然存在。为此，我们提出了一种评估框架，用于衡量推理的准确性、一致性和忠实性。我们的结果既突显了检索增强生成（RAG）的潜力，也指出了其风险：RAG 可以限制输出，但安全部署需要像评估检索一样严格地评估推理。 

---
# Detecting Sleeper Agents in Large Language Models via Semantic Drift Analysis 

**Title (ZH)**: 通过语义漂移分析检测大型语言模型中的潜入Agent 

**Authors**: Shahin Zanbaghi, Ryan Rostampour, Farhan Abid, Salim Al Jarmakani  

**Link**: [PDF](https://arxiv.org/pdf/2511.15992)  

**Abstract**: Large Language Models (LLMs) can be backdoored to exhibit malicious behavior under specific deployment conditions while appearing safe during training a phenomenon known as "sleeper agents." Recent work by Hubinger et al. demonstrated that these backdoors persist through safety training, yet no practical detection methods exist. We present a novel dual-method detection system combining semantic drift analysis with canary baseline comparison to identify backdoored LLMs in real-time. Our approach uses Sentence-BERT embeddings to measure semantic deviation from safe baselines, complemented by injected canary questions that monitor response consistency. Evaluated on the official Cadenza-Labs dolphin-llama3-8B sleeper agent model, our system achieves 92.5% accuracy with 100% precision (zero false positives) and 85% recall. The combined detection method operates in real-time (<1s per query), requires no model modification, and provides the first practical solution to LLM backdoor detection. Our work addresses a critical security gap in AI deployment and demonstrates that embedding-based detection can effectively identify deceptive model behavior without sacrificing deployment efficiency. 

**Abstract (ZH)**: 大型语言模型可以通过回门手段在特定部署条件下表现出恶意行为而在训练期间看似安全，这种现象被称为“ sleeper agents”。我们提出了一种结合语义漂移分析和金丝雀基线比较的新型双方法检测系统，用于实时检测被回门的大型语言模型。我们的方法使用Sentence-BERT嵌入度量与安全基线的语义偏差，并通过注入的金丝雀问题来监控响应一致性。在官方Cadenza-Labs dolphin-llama3-8B sleeper agent模型上评估，我们的系统实现了92.5%的准确率、100%的精确率（零假阳性）和85%的召回率。结合的检测方法实时运行（每个查询<1秒），不需要对模型进行修改，并提供第一个实用的大规模语言模型回门检测解决方案。我们的工作填补了人工智能部署中的一个重要安全缺口，并证明了基于嵌入的检测可以有效地识别欺骗性模型行为而不牺牲部署效率。 

---
# KRAL: Knowledge and Reasoning Augmented Learning for LLM-assisted Clinical Antimicrobial Therapy 

**Title (ZH)**: KRAL: 知识与推理增强的学习在LLM辅助临床抗菌治疗中的应用 

**Authors**: Zhe Li, Yehan Qiu, Yujie Chen, Xiang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.15974)  

**Abstract**: Clinical antimicrobial therapy requires the dynamic integration of pathogen profiles, host factors, pharmacological properties of antimicrobials, and the severity of this http URL complexity imposes fundamental limitations on the applicability of Large Language Models (LLMs) in high-stakes clinical decision-making including knowledge gaps, data privacy concerns, high deployment costs, and limited reasoning capabilities. To address these challenges, we propose KRAL (Knowledge and Reasoning Augmented Learning), a low-cost, scalable, privacy-preserving paradigm that leverages teacher-model reasoning to automatically distill knowledge and reasoning trajectories via answer-to-question reverse generation, employs heuristic learning for semi-supervised data augmentation (reducing manual annotation requirements by approximately 80%), and utilizes agentic reinforcement learning to jointly enhance medical knowledge and reasoning while optimizing computational and memory efficiency. A hierarchical evaluation employing diverse teacher-model proxies reduces assessment costs, while modular interface design facilitates seamless system updates. Experimental results demonstrate that KRAL significantly outperforms traditional Retrieval-Augmented Generation (RAG) and Supervised Fine-Tuning (SFT) methods. It improves knowledge question-answering capability (Accuracy@1 on the external open-source benchmark MEDQA increased by 1.8% vs. SFT and 3.6% vs. RAG) and reasoning capability (Pass@1 on the external benchmark PUMCH Antimicrobial increased by 27% vs. SFT and 27.2% vs. RAG), achieved at ~20% of SFT's long-term training costs. This establishes KRAL as an effective solution for enhancing local LLMs' clinical diagnostic capabilities, enabling low-cost, high-safety deployment in complex medical decision support. 

**Abstract (ZH)**: 临床抗微生物治疗需要动态整合病原体特征、宿主因素、抗微生物药物的药理特性以及病情严重程度。这种复杂性对大型语言模型（LLMs）在高风险临床决策中的应用提出了根本性的局限性，包括知识空白、数据隐私问题、高部署成本和有限的推理能力。为解决这些挑战，我们提出了KRAL（Knowledge and Reasoning Augmented Learning）框架，这是一种低成本、可扩展、保护隐私的方法，它利用教师模型的推理自动提取知识和推理轨迹，通过答案到问题的逆向生成，利用启发式学习进行半监督数据增强（减少约80%的手动标注需求），并利用代理强化学习同时增强医学知识和推理，优化计算和内存效率。多层次的评估采用多样化的教师模型代理降低了评估成本，模块化界面设计便于系统更新。实验结果表明，KRAL 显著优于传统的检索增强生成（RAG）和监督微调（SFT）方法。KRAL 提高了知识问答能力（在外部开源基准 MEDQA 上的 Accuracy@1 提高了 1.8% 对比 SFT 和 3.6% 对比 RAG），以及推理能力（在外部基准 PUMCH Antimicrobial 上的 Pass@1 提高了 27% 对比 SFT 和 27.2% 对比 RAG），实现成本约为 SFT 长期训练成本的 20%。这表明 KRAL 是增强本地 LLM 临床诊断能力的有效方案，能够以低成本、高安全性部署在复杂的医学决策支持中。 

---
# JudgeBoard: Benchmarking and Enhancing Small Language Models for Reasoning Evaluation 

**Title (ZH)**: JudgeBoard: 评估和增强小型语言模型在推理评估中的基准测试与提升 

**Authors**: Zhenyu Bi, Gaurav Srivastava, Yang Li, Meng Lu, Swastik Roy, Morteza Ziyadi, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15958)  

**Abstract**: While small language models (SLMs) have shown promise on various reasoning tasks, their ability to judge the correctness of answers remains unclear compared to large language models (LLMs). Prior work on LLM-as-a-judge frameworks typically relies on comparing candidate answers against ground-truth labels or other candidate answers using predefined metrics like entailment. However, this approach is inherently indirect and difficult to fully automate, offering limited support for fine-grained and scalable evaluation of reasoning outputs. In this work, we propose JudgeBoard, a novel evaluation pipeline that directly queries models to assess the correctness of candidate answers without requiring extra answer comparisons. We focus on two core reasoning domains: mathematical reasoning and science/commonsense reasoning, and construct task-specific evaluation leaderboards using both accuracy-based ranking and an Elo-based rating system across five benchmark datasets, enabling consistent model comparison as judges rather than comparators. To improve judgment performance in lightweight models, we propose MAJ (Multi-Agent Judging), a novel multi-agent evaluation framework that leverages multiple interacting SLMs with distinct reasoning profiles to approximate LLM-level judgment accuracy through collaborative deliberation. Experimental results reveal a significant performance gap between SLMs and LLMs in isolated judging tasks. However, our MAJ framework substantially improves the reliability and consistency of SLMs. On the MATH dataset, MAJ using smaller-sized models as backbones performs comparatively well or even better than their larger-sized counterparts. Our findings highlight that multi-agent SLM systems can potentially match or exceed LLM performance in judgment tasks, with implications for scalable and efficient assessment. 

**Abstract (ZH)**: JudgeBoard：面向推理输出直接评估的新型评价pipeline 

---
# Thinking, Faithful and Stable: Mitigating Hallucinations in LLMs 

**Title (ZH)**: 思辨、忠实且稳定：减轻LLM中的幻觉 

**Authors**: Chelsea Zou, Yiheng Yao, Basant Khalil  

**Link**: [PDF](https://arxiv.org/pdf/2511.15921)  

**Abstract**: This project develops a self correcting framework for large language models (LLMs) that detects and mitigates hallucinations during multi-step reasoning. Rather than relying solely on final answer correctness, our approach leverages fine grained uncertainty signals: 1) self-assessed confidence alignment, and 2) token-level entropy spikes to detect unreliable and unfaithful reasoning in real time. We design a composite reward function that penalizes unjustified high confidence and entropy spikes, while encouraging stable and accurate reasoning trajectories. These signals guide a reinforcement learning (RL) policy that makes the model more introspective and shapes the model's generation behavior through confidence-aware reward feedback, improving not just outcome correctness but the coherence and faithfulness of their intermediate reasoning steps. Experiments show that our method improves both final answer accuracy and reasoning calibration, with ablations validating the individual contribution of each signal. 

**Abstract (ZH)**: 本项目开发了一个自我矫正框架，用于大型语言模型（LLMs），该框架在多步推理过程中检测并减轻幻觉现象。我们 approach 不仅依赖最终答案的正确性，还利用细粒度的不确定性信号：1) 自我评估的信心对齐，以及 2) 令牌级的熵突增，以实时检测不可靠和不忠实的推理。我们设计了一个复合奖励函数，该函数惩罚不合理的高信心和熵突增，同时鼓励稳定和准确的推理轨迹。这些信号引导强化学习（RL）策略，通过信心意识的奖励反馈，使模型更加反思，并通过生成行为塑造提高最终答案的准确性以及中间推理步骤的一致性和忠实性。实验表明，我们的方法不仅提高了最终答案的准确性，还改进了推理的校准，消融实验验证了每个信号的独立贡献。 

---
# Decomposing Theory of Mind: How Emotional Processing Mediates ToM Abilities in LLMs 

**Title (ZH)**: 分解理论心智：情绪处理如何中介LLMs的理论心智能力 

**Authors**: Ivan Chulo, Ananya Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2511.15895)  

**Abstract**: Recent work shows activation steering substantially improves language models' Theory of Mind (ToM) (Bortoletto et al. 2024), yet the mechanisms of what changes occur internally that leads to different outputs remains unclear. We propose decomposing ToM in LLMs by comparing steered versus baseline LLMs' activations using linear probes trained on 45 cognitive actions. We applied Contrastive Activation Addition (CAA) steering to Gemma-3-4B and evaluated it on 1,000 BigToM forward belief scenarios (Gandhi et al. 2023), we find improved performance on belief attribution tasks (32.5\% to 46.7\% accuracy) is mediated by activations processing emotional content : emotion perception (+2.23), emotion valuing (+2.20), while suppressing analytical processes: questioning (-0.78), convergent thinking (-1.59). This suggests that successful ToM abilities in LLMs are mediated by emotional understanding, not analytical reasoning. 

**Abstract (ZH)**: Recent Work Shows Activation Steering Substantially Improves Language Models' Theory of Mind (ToM), but the Internal Mechanisms Remain unclear. We Propose Decomposing ToM in LLMs by Comparing Steered Versus Baseline LLMs' Activations Using Linear Probes Trained on 45 Cognitive Actions. We Applied Contrastive Activation Addition (CAA) Steering to Gemma-3-4B and Evaluated It on 1,000 BigToM Forward Belief Scenarios, Finding Improved Performance on Belief Attribution Tasks (32.5% to 46.7% Accuracy) Is Mediated by Activations Processing Emotional Content: Emotion Perception (+2.23), Emotion Valuing (+2.20), While Suppressing Analytical Processes: Questioning (-0.78), Convergent Thinking (-1.59). This Suggests That Successful ToM Abilities in LLMs Are Mediated by Emotional Understanding, Not Analytical Reasoning. 

---
# Step-Audio-R1 Technical Report 

**Title (ZH)**: 分步音频-R1 技术报告 

**Authors**: Fei Tian, Xiangyu Tony Zhang, Yuxin Zhang, Haoyang Zhang, Yuxin Li, Daijiao Liu, Yayue Deng, Donghang Wu, Jun Chen, Liang Zhao, Chengyuan Yao, Hexin Liu, Eng Siong Chng, Xuerui Yang, Xiangyu Zhang, Daxin Jiang, Gang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.15848)  

**Abstract**: Recent advances in reasoning models have demonstrated remarkable success in text and vision domains through extended chain-of-thought deliberation. However, a perplexing phenomenon persists in audio language models: they consistently perform better with minimal or no reasoning, raising a fundamental question - can audio intelligence truly benefit from deliberate thinking? We introduce Step-Audio-R1, the first audio reasoning model that successfully unlocks reasoning capabilities in the audio domain. Through our proposed Modality-Grounded Reasoning Distillation (MGRD) framework, Step-Audio-R1 learns to generate audio-relevant reasoning chains that genuinely ground themselves in acoustic features rather than hallucinating disconnected deliberations. Our model exhibits strong audio reasoning capabilities, surpassing Gemini 2.5 Pro and achieving performance comparable to the state-of-the-art Gemini 3 Pro across comprehensive audio understanding and reasoning benchmarks spanning speech, environmental sounds, and music. These results demonstrate that reasoning is a transferable capability across modalities when appropriately anchored, transforming extended deliberation from a liability into a powerful asset for audio intelligence. By establishing the first successful audio reasoning model, Step-Audio-R1 opens new pathways toward building truly multimodal reasoning systems that think deeply across all sensory modalities. 

**Abstract (ZH)**: Recent Advances in Audio Reasoning Models: From Rejection of Deliberative Thinking to Successful Integration in Step-Audio-R1 

---
# Mini Amusement Parks (MAPs): A Testbed for Modelling Business Decisions 

**Title (ZH)**: 迷你游乐园 (MAPs): 业务决策建模的实验平台 

**Authors**: Stéphane Aroca-Ouellette, Ian Berlot-Attwell, Panagiotis Lymperopoulos, Abhiramon Rajasekharan, Tongqi Zhu, Herin Kang, Kaheer Suleman, Sam Pasupalak  

**Link**: [PDF](https://arxiv.org/pdf/2511.15830)  

**Abstract**: Despite rapid progress in artificial intelligence, current systems struggle with the interconnected challenges that define real-world decision making. Practical domains, such as business management, require optimizing an open-ended and multi-faceted objective, actively learning environment dynamics from sparse experience, planning over long horizons in stochastic settings, and reasoning over spatial information. Yet existing human--AI benchmarks isolate subsets of these capabilities, limiting our ability to assess holistic decision-making competence. We introduce Mini Amusement Parks (MAPs), an amusement-park simulator designed to evaluate an agent's ability to model its environment, anticipate long-term consequences under uncertainty, and strategically operate a complex business. We provide human baselines and a comprehensive evaluation of state-of-the-art LLM agents, finding that humans outperform these systems by 6.5x on easy mode and 9.8x on medium mode. Our analysis reveals persistent weaknesses in long-horizon optimization, sample-efficient learning, spatial reasoning, and world modelling. By unifying these challenges within a single environment, MAPs offers a new foundation for benchmarking agents capable of adaptable decision making. Code: this https URL 

**Abstract (ZH)**: 尽管人工智能取得了快速进展，当前的系统在处理定义现实世界决策的相互联系的挑战方面仍然存在问题。商业管理等实际领域需要优化一个开放且多维度的目标，从稀疏的经验中积极学习环境动态，在不确定性环境下进行长远规划，并在空间信息层面进行推理。然而，现有的人类-人工智能基准测试将这些能力的子集隔离开来，限制了我们全面评估决策能力的能力。我们引入了迷你游乐园（MAPs），这是一个游乐园模拟器，旨在评估代理建模环境、在不确定性下预测长期后果以及战略性运营复杂企业的能力。我们提供了人类基准线，并全面评估了最新一代的LLM代理，发现人类在容易模式上比这些系统高出6.5倍，在中等模式上高出9.8倍。我们的分析揭示了长期优化、样本高效学习、空间推理和世界建模的持久性弱点。通过在单一环境中统一这些挑战，MAPs为评估具备适应性决策能力的代理提供了新的基础。代码：https://thisurl.com 

---
# IMACT-CXR - An Interactive Multi-Agent Conversational Tutoring System for Chest X-Ray Interpretation 

**Title (ZH)**: IMACT-CXR - 一种交互式多agent对话式教学系统用于胸部X光解析 

**Authors**: Tuan-Anh Le, Anh Mai Vu, David Yang, Akash Awasthi, Hien Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.15825)  

**Abstract**: IMACT-CXR is an interactive multi-agent conversational tutor that helps trainees interpret chest X-rays by unifying spatial annotation, gaze analysis, knowledge retrieval, and image-grounded reasoning in a single AutoGen-based workflow. The tutor simultaneously ingests learner bounding boxes, gaze samples, and free-text observations. Specialized agents evaluate localization quality, generate Socratic coaching, retrieve PubMed evidence, suggest similar cases from REFLACX, and trigger NV-Reason-CXR-3B for vision-language reasoning when mastery remains low or the learner explicitly asks. Bayesian Knowledge Tracing (BKT) maintains skill-specific mastery estimates that drive both knowledge reinforcement and case similarity retrieval. A lung-lobe segmentation module derived from a TensorFlow U-Net enables anatomically aware gaze feedback, and safety prompts prevent premature disclosure of ground-truth labels. We describe the system architecture, implementation highlights, and integration with the REFLACX dataset for real DICOM cases. IMACT-CXR demonstrates responsive tutoring flows with bounded latency, precise control over answer leakage, and extensibility toward live residency deployment. Preliminary evaluation shows improved localization and diagnostic reasoning compared to baselines. 

**Abstract (ZH)**: IMACT-CXR是一种交互式多智能体会话导师，通过在单个AutoGen基础上统一空间标注、注视分析、知识检索和基于图像的推理，帮助学员解读胸部X光片。该导师同时摄入学员的边界框、注视样本和自由文本观察。专业化的智能体评估定位质量、生成苏 krb 哲式辅导、检索PubMed证据、从 REFLACX 中推荐相似病例，并在掌握程度低或学员明确要求时触发 NV-Reason-CXR-3B 进行视觉-语言推理。贝叶斯知识追踪（BKT）维护了特定技能的掌握估算值，驱动知识强化和病例相似性检索。一个源于 TensorFlow U-Net 的肺叶分割模块提供解剖学意识的注视反馈，并通过安全提示防止提前披露真实标签。我们描述了系统架构、实现亮点以及与 REFLACX 数据集的集成，用于真实 DICOM 案例。IMACT-CXR 展示了响应式的辅导流程，具有有界延迟、精确的答案泄露控制，并且可以扩展到实时住院医师部署。初步评估表明，IMACT-CXR 在定位和诊断推理方面优于基线方法。 

---
# Balancing Natural Language Processing Accuracy and Normalisation in Extracting Medical Insights 

**Title (ZH)**: 平衡自然语言处理准确性与规范化在提取医学洞察中的关系 

**Authors**: Paulina Tworek, Miłosz Bargieł, Yousef Khan, Tomasz Pełech-Pilichowski, Marek Mikołajczyk, Roman Lewandowski, Jose Sousa  

**Link**: [PDF](https://arxiv.org/pdf/2511.15778)  

**Abstract**: Extracting structured medical insights from unstructured clinical text using Natural Language Processing (NLP) remains an open challenge in healthcare, particularly in non-English contexts where resources are scarce. This study presents a comparative analysis of NLP low-compute rule-based methods and Large Language Models (LLMs) for information extraction from electronic health records (EHR) obtained from the Voivodeship Rehabilitation Hospital for Children in Ameryka, Poland. We evaluate both approaches by extracting patient demographics, clinical findings, and prescribed medications while examining the effects of lack of text normalisation and translation-induced information loss. Results demonstrate that rule-based methods provide higher accuracy in information retrieval tasks, particularly for age and sex extraction. However, LLMs offer greater adaptability and scalability, excelling in drug name recognition. The effectiveness of the LLMs was compared with texts originally in Polish and those translated into English, assessing the impact of translation. These findings highlight the trade-offs between accuracy, normalisation, and computational cost when deploying NLP in healthcare settings. We argue for hybrid approaches that combine the precision of rule-based systems with the adaptability of LLMs, offering a practical path toward more reliable and resource-efficient clinical NLP in real-world hospitals. 

**Abstract (ZH)**: 使用自然语言处理（NLP）从非英语背景下的未结构化临床文本中提取结构化医疗见解仍是一项开放性的挑战：低计算量规则基于方法与大型语言模型在电子健康记录信息提取中的比较研究 

---
# Identifying the Supply Chain of AI for Trustworthiness and Risk Management in Critical Applications 

**Title (ZH)**: 识别AI在关键应用中的供应链，以确保可信性和风险管理 

**Authors**: Raymond K. Sheh, Karen Geappen  

**Link**: [PDF](https://arxiv.org/pdf/2511.15763)  

**Abstract**: Risks associated with the use of AI, ranging from algorithmic bias to model hallucinations, have received much attention and extensive research across the AI community, from researchers to end-users. However, a gap exists in the systematic assessment of supply chain risks associated with the complex web of data sources, pre-trained models, agents, services, and other systems that contribute to the output of modern AI systems. This gap is particularly problematic when AI systems are used in critical applications, such as the food supply, healthcare, utilities, law, insurance, and transport.
We survey the current state of AI risk assessment and management, with a focus on the supply chain of AI and risks relating to the behavior and outputs of the AI system. We then present a proposed taxonomy specifically for categorizing AI supply chain entities. This taxonomy helps stakeholders, especially those without extensive AI expertise, to "consider the right questions" and systematically inventory dependencies across their organization's AI systems. Our contribution bridges a gap between the current state of AI governance and the urgent need for actionable risk assessment and management of AI use in critical applications. 

**Abstract (ZH)**: 与AI使用相关的风险，包括算法偏差和模型幻觉等，已在AI社区引起了广泛关注和深入研究，从研究人员到最终用户。然而，存在一个系统评估AI供应链风险的缺口，这些风险与构成现代AI系统输出的复杂数据源、预训练模型、代理、服务和其他系统的网络有关。当AI系统在食品供应链、医疗、公用事业、法律、保险和运输等关键应用中使用时，这一缺口尤为突出。

我们概述了当前AI风险评估和管理的状态，重点关注AI供应链及其相关行为和输出的风险。随后，我们提出了一种特定的分类法，用于分类AI供应链实体。该分类法有助于相关方，尤其是缺乏大量AI专业知识的人士，能够“提出正确的疑问”并系统地盘点其组织内AI系统的依赖关系。我们的贡献填补了当前AI治理状态与迫切需要在关键应用中进行可操作的风险评估和管理之间的缺口。 

---
# Multi-Agent LLM Orchestration Achieves Deterministic, High-Quality Decision Support for Incident Response 

**Title (ZH)**: 多代理大语言模型编排实现确定性、高质量的事件响应决策支持 

**Authors**: Philip Drammeh  

**Link**: [PDF](https://arxiv.org/pdf/2511.15755)  

**Abstract**: Large language models (LLMs) promise to accelerate incident response in production systems, yet single-agent approaches generate vague, unusable recommendations. We present this http URL, a reproducible containerized framework demonstrating that multi-agent orchestration fundamentally transforms LLM-based incident response quality. Through 348 controlled trials comparing single-agent copilot versus multi-agent systems on identical incident scenarios, we find that multi-agent orchestration achieves 100% actionable recommendation rate versus 1.7% for single-agent approaches, an 80 times improvement in action specificity and 140 times improvement in solution correctness. Critically, multi-agent systems exhibit zero quality variance across all trials, enabling production SLA commitments impossible with inconsistent single-agent outputs. Both architectures achieve similar comprehension latency (approx.40s), establishing that the architectural value lies in deterministic quality, not speed. We introduce Decision Quality (DQ), a novel metric capturing validity, specificity, and correctness properties essential for operational deployment that existing LLM metrics do not address. These findings reframe multi-agent orchestration from a performance optimization to a production-readiness requirement for LLM-based incident response. All code, Docker configurations, and trial data are publicly available for reproduction. 

**Abstract (ZH)**: 大型语言模型（LLMs）有潜力加速生产系统的故障响应，但单agent方法生成的是模糊且不可用的建议。我们展示了这个网址，一个可重复的容器化框架，证明了多agent编排从根本上转变了基于LLM的故障响应质量。通过348次受控试验，比较单agent copilot与多agent系统在相同故障场景下的表现，我们发现多agent编排实现了100%可操作建议率，而单agent方法仅为1.7%，在行动具体性和解决方案正确性方面分别提高了80倍和140倍。关键的是，多agent系统在整个试验中的质量没有差异性，使生产SLA承诺不再成为可能的不一致单agent输出的问题。两种架构的理解延迟相似（约40秒），这表明架构价值在于确定性质量，而不是速度。我们提出了决策质量（DQ）作为一个新的度量标准，捕获了现有LLM度量标准所缺乏的操作部署中的有效性和具体性等关键属性。这些发现将多agent编排从性能优化重新定义为基于LLM的故障响应的生产就绪要求。所有代码、Docker配置和试验数据均可公开获取以供复制。 

---
# Build AI Assistants using Large Language Models and Agents to Enhance the Engineering Education of Biomechanics 

**Title (ZH)**: 使用大规模语言模型和智能体构建AI助手以增强生物力学工程教育 

**Authors**: Hanzhi Yan, Qin Lu, Xianqiao Wang, Xiaoming Zhai, Tianming Liu, He Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.15752)  

**Abstract**: While large language models (LLMs) have demonstrated remarkable versatility across a wide range of general tasks, their effectiveness often diminishes in domain-specific applications due to inherent knowledge gaps. Moreover, their performance typically declines when addressing complex problems that require multi-step reasoning and analysis. In response to these challenges, we propose leveraging both LLMs and AI agents to develop education assistants aimed at enhancing undergraduate learning in biomechanics courses that focus on analyzing the force and moment in the musculoskeletal system of the human body. To achieve our goal, we construct a dual-module framework to enhance LLM performance in biomechanics educational tasks: 1) we apply Retrieval-Augmented Generation (RAG) to improve the specificity and logical consistency of LLM's responses to the conceptual true/false questions; 2) we build a Multi-Agent System (MAS) to solve calculation-oriented problems involving multi-step reasoning and code execution. Specifically, we evaluate the performance of several LLMs, i.e., Qwen-1.0-32B, Qwen-2.5-32B, and Llama-70B, on a biomechanics dataset comprising 100 true/false conceptual questions and problems requiring equation derivation and calculation. Our results demonstrate that RAG significantly enhances the performance and stability of LLMs in answering conceptual questions, surpassing those of vanilla models. On the other hand, the MAS constructed using multiple LLMs demonstrates its ability to perform multi-step reasoning, derive equations, execute code, and generate explainable solutions for tasks that require calculation. These findings demonstrate the potential of applying RAG and MAS to enhance LLM performance for specialized courses in engineering curricula, providing a promising direction for developing intelligent tutoring in engineering education. 

**Abstract (ZH)**: 大型语言模型（LLMs）在广泛通用任务中展现了显著的 versatility，但在特定领域应用中因固有的知识差距而效果减弱。此外，它们在解决需要多步推理和分析的复杂问题时表现也通常较差。为应对这些挑战，我们提出结合使用大型语言模型和AI代理，开发教育助手，旨在提升生物力学课程中分析人体运动系统中力和力矩的本专科生学习效果。为实现这一目标，我们构建了一个双模块框架以增强语言模型在生物力学教育任务中的性能：1）应用检索增强生成（RAG）以提高其对概念性真伪问题响应的特异性和逻辑一致性；2）构建一个多代理系统（MAS）以解决涉及多步推理和代码执行的计算导向问题。我们对几种大型语言模型，即Qwen-1.0-32B、Qwen-2.5-32B和Llama-70B，进行了评估，这些模型用于包含100个概念性真伪问题和需要方程推导与计算的问题的生物力学数据集。结果表明，RAG显著提高了语言模型在回答概念性问题时的性能和稳定性，超越了基础模型。另一方面，使用多个大型语言模型构建的MAS展示了其进行多步推理、方程推导、代码执行和生成可解释解决方案的能力，适用于需要计算的任务。这些发现表明，结合RAG和MAS有可能提升大型语言模型在工程课程中特定领域的表现，为开发工程教育中的智能辅导提供了有前景的方向。 

---
# Uncertainty-Resilient Multimodal Learning via Consistency-Guided Cross-Modal Transfer 

**Title (ZH)**: 面向一致性和跨模态转移的不确定性鲁棒多模态学习 

**Authors**: Hyo-Jeong Jang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15741)  

**Abstract**: Multimodal learning systems often face substantial uncertainty due to noisy data, low-quality labels, and heterogeneous modality characteristics. These issues become especially critical in human-computer interaction settings, where data quality, semantic reliability, and annotation consistency vary across users and recording conditions. This thesis tackles these challenges by exploring uncertainty-resilient multimodal learning through consistency-guided cross-modal transfer. The central idea is to use cross-modal semantic consistency as a basis for robust representation learning. By projecting heterogeneous modalities into a shared latent space, the proposed framework mitigates modality gaps and uncovers structural relations that support uncertainty estimation and stable feature learning. Building on this foundation, the thesis investigates strategies to enhance semantic robustness, improve data efficiency, and reduce the impact of noise and imperfect supervision without relying on large, high-quality annotations. Experiments on multimodal affect-recognition benchmarks demonstrate that consistency-guided cross-modal transfer significantly improves model stability, discriminative ability, and robustness to noisy or incomplete supervision. Latent space analyses further show that the framework captures reliable cross-modal structure even under challenging conditions. Overall, this thesis offers a unified perspective on resilient multimodal learning by integrating uncertainty modeling, semantic alignment, and data-efficient supervision, providing practical insights for developing reliable and adaptive brain-computer interface systems. 

**Abstract (ZH)**: 多模态学习系统经常面临由于噪声数据、低质量标签和异质模态特性带来的大量不确定性。这些问题在人机交互设置中尤为关键，其中数据质量、语义可靠性和标注一致性因用户和记录条件的不同而异。本论文通过探索基于一致性引导的跨模态迁移来应对这些挑战，核心思想是利用跨模态语义一致性作为鲁棒表示学习的基础。通过将异质模态投影到共享的潜在空间，所提出的框架减少了模态差距，并揭示了支持不确定性估计和稳定特征学习的结构关系。在此基础上，论文研究了增强语义鲁棒性、提高数据效率以及减少噪声和不完善监督影响的策略，而不依赖于大规模和高质量的标注数据。多模态情感识别基准上的实验表明，基于一致性的跨模态迁移显著提高了模型的稳定性、判别能力和对噪声或不完整监督的鲁棒性。潜在空间分析进一步表明，框架即使在挑战性条件下也能捕获可靠的跨模态结构。总体而言，本论文通过集成不确定性建模、语义对齐和数据高效监督，提供了一种统一的鲁棒多模态学习视角，为开发可靠和适应性强的脑机接口系统提供了实用见解。 

---
# Spatial Reasoning in Multimodal Large Language Models: A Survey of Tasks, Benchmarks and Methods 

**Title (ZH)**: 多模态大型语言模型中的空间推理：任务、基准和方法综述 

**Authors**: Weichen Liu, Qiyao Xue, Haoming Wang, Xiangyu Yin, Boyuan Yang, Wei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.15722)  

**Abstract**: Spatial reasoning, which requires ability to perceive and manipulate spatial relationships in the 3D world, is a fundamental aspect of human intelligence, yet remains a persistent challenge for Multimodal large language models (MLLMs). While existing surveys often categorize recent progress based on input modality (e.g., text, image, video, or 3D), we argue that spatial ability is not solely determined by the input format. Instead, our survey introduces a taxonomy that organizes spatial intelligence from cognitive aspect and divides tasks in terms of reasoning complexity, linking them to several cognitive functions. We map existing benchmarks across text only, vision language, and embodied settings onto this taxonomy, and review evaluation metrics and methodologies for assessing spatial reasoning ability. This cognitive perspective enables more principled cross-task comparisons and reveals critical gaps between current model capabilities and human-like reasoning. In addition, we analyze methods for improving spatial ability, spanning both training-based and reasoning-based approaches. This dual perspective analysis clarifies their respective strengths, uncovers complementary mechanisms. By surveying tasks, benchmarks, and recent advances, we aim to provide new researchers with a comprehensive understanding of the field and actionable directions for future research. 

**Abstract (ZH)**: 空间推理，这是指在三维世界中感知和操作空间关系的能力，是人类智能的一个基本方面，但仍然是多模态大规模语言模型（MLLMs）面临的持久挑战。虽然现有的综述通常根据输入模态（例如，文本、图像、视频或3D）来分类近期的进步，我们认为空间能力不仅仅由输入格式决定。相反，我们的综述引入了一种分类法，从认知方面组织空间智能，并根据推理复杂度将任务划分为不同的类别，将其与几种认知功能联系起来。我们将现有的仅限文本、视觉语言和嵌入式设置的基准映射到这一分类法，并回顾评估空间推理能力的指标和方法学。这种认知视角使跨任务比较更加原则化，并揭示了现有模型能力与人类推理之间的关键差距。此外，我们分析提高空间能力的方法，涵盖基于训练和基于推理的方法。这种双重视角分析澄清了它们各自的优点，并发现了互补机制。通过调查任务、基准和近期进展，我们的目标是为新研究人员提供对该领域的全面理解，并为未来研究提供可操作的方向。 

---
# Automated Hazard Detection in Construction Sites Using Large Language and Vision-Language Models 

**Title (ZH)**: 使用大型语言和视觉语言模型在建筑工地自动检测安全隐患 

**Authors**: Islem Sahraoui  

**Link**: [PDF](https://arxiv.org/pdf/2511.15720)  

**Abstract**: This thesis explores a multimodal AI framework for enhancing construction safety through the combined analysis of textual and visual data. In safety-critical environments such as construction sites, accident data often exists in multiple formats, such as written reports, inspection records, and site imagery, making it challenging to synthesize hazards using traditional approaches. To address this, this thesis proposed a multimodal AI framework that combines text and image analysis to assist in identifying safety hazards on construction sites. Two case studies were consucted to evaluate the capabilities of large language models (LLMs) and vision-language models (VLMs) for automated hazard this http URL first case study introduces a hybrid pipeline that utilizes GPT 4o and GPT 4o mini to extract structured insights from a dataset of 28,000 OSHA accident reports (2000-2025). The second case study extends this investigation using Molmo 7B and Qwen2 VL 2B, lightweight, open-source VLMs. Using the public ConstructionSite10k dataset, the performance of the two models was evaluated on rule-level safety violation detection using natural language prompts. This experiment served as a cost-aware benchmark against proprietary models and allowed testing at scale with ground-truth labels. Despite their smaller size, Molmo 7B and Quen2 VL 2B showed competitive performance in certain prompt configurations, reinforcing the feasibility of low-resource multimodal systems for rule-aware safety monitoring. 

**Abstract (ZH)**: 一种通过文本和视觉数据联合作业增强建筑安全的多模态AI框架 

---
# Chain of Summaries: Summarization Through Iterative Questioning 

**Title (ZH)**: 链式摘要：通过迭代提问进行摘要总结 

**Authors**: William Brach, Lukas Galke Poech  

**Link**: [PDF](https://arxiv.org/pdf/2511.15719)  

**Abstract**: Large Language Models (LLMs) are increasingly using external web content. However, much of this content is not easily digestible by LLMs due to LLM-unfriendly formats and limitations of context length. To address this issue, we propose a method for generating general-purpose, information-dense summaries that act as plain-text repositories of web content. Inspired by Hegel's dialectical method, our approach, denoted as Chain of Summaries (CoS), iteratively refines an initial summary (thesis) by identifying its limitations through questioning (antithesis), leading to a general-purpose summary (synthesis) that can satisfy current and anticipate future information needs. Experiments on the TriviaQA, TruthfulQA, and SQUAD datasets demonstrate that CoS outperforms zero-shot LLM baselines by up to 66% and specialized summarization methods such as BRIO and PEGASUS by up to 27%. CoS-generated summaries yield higher Q&A performance compared to the source content, while requiring substantially fewer tokens and being agnostic to the specific downstream LLM. CoS thus resembles an appealing option for website maintainers to make their content more accessible for LLMs, while retaining possibilities for human oversight. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地使用外部网页内容。然而，其中很大一部分内容由于LLM不友好的格式和背景长度的限制，难以被LLMs轻松消化。为了解决这一问题，我们提出了一种生成通用、信息密集型摘要的方法，这些摘要可作为网页内容的纯文本存储库。我们的方法借鉴了黑格尔辩证法，称为摘要链（CoS），通过迭代细化初始摘要（命题），并通过质疑其局限性（反题），最终生成一种可满足当前信息需求并能预见未来信息需求的一般性摘要（合题）。在TriviaQA、TruthfulQA和SQUAD数据集上的实验表明，CoS在零-shot LLM基线和专用摘要方法（如BRIO和PEGASUS）方面表现更优，分别提高了66%和27%。CoS生成的摘要相比源内容能提供更高的问答性能，同时需要较少的标记，并且对特定的下游LLM无偏见。因此，CoS为网站维护人员提供了一个吸引人的选项，使其内容对LLMs更易访问，同时保留了人类监督的可能性。 

---
# ToolMind Technical Report: A Large-Scale, Reasoning-Enhanced Tool-Use Dataset 

**Title (ZH)**: ToolMind 技术报告：大规模推理增强工具使用数据集 

**Authors**: Chen Yang, Ran Le, Yun Xing, Zhenwei An, Zongchao Chen, Wayne Xin Zhao, Yang Song, Tao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15718)  

**Abstract**: Large Language Model (LLM) agents have developed rapidly in recent years to solve complex real-world problems using external tools. However, the scarcity of high-quality trajectories still hinders the development of stronger LLM agents. Most existing works on multi-turn dialogue synthesis validate correctness only at the trajectory level, which may overlook turn-level errors that can propagate during training and degrade model performance. To address these limitations, we introduce ToolMind, a large-scale, high-quality tool-agentic dataset with 160k synthetic data instances generated using over 20k tools and 200k augmented open-source data instances. Our data synthesis pipeline first constructs a function graph based on parameter correlations and then uses a multi-agent framework to simulate realistic user-assistant-tool interactions. Beyond trajectory-level validation, we employ fine-grained turn-level filtering to remove erroneous or suboptimal steps, ensuring that only high-quality reasoning traces are retained. This approach mitigates error amplification during training while preserving self-corrective reasoning signals essential for robust tool-use learning. Models fine-tuned on ToolMind show significant improvements over baselines on several benchmarks. 

**Abstract (ZH)**: Large Language Model (LLM) 剂型在利用外部工具解决复杂现实问题方面发展迅速，但高质量的轨迹数据仍然稀缺，限制了更强的LLM剂型的发展。现有的多轮对话合成工作主要在轨迹级别验证正确性，这可能会忽略在训练过程中传播的轮次级别错误，从而降低模型性能。为解决这些问题，我们介绍了ToolMind，这是一个大规模、高质量的工具-剂型数据集，包含160k个使用超过20k个工具生成的合成数据实例，以及200k个增强的开源数据实例。我们的数据合成管道首先根据参数相关性构建一个功能图，然后使用多剂型框架模拟现实的用户-助手-工具交互。除轨迹级别验证外，我们还采用细粒度的轮次级别筛选以移除错误或次优步骤，确保仅保留高质量的推理轨迹。这种方法可以减轻训练过程中的错误放大，同时保留对于稳健工具使用学习至关重要的自我纠正推理信号。在ToolMind上进行微调的模型在多个基准上表现出显著的性能提升。 

---
# How Modality Shapes Perception and Reasoning: A Study of Error Propagation in ARC-AGI 

**Title (ZH)**: 模态如何塑造感知与推理：ARC-AGI 中错误传播的研究 

**Authors**: Bo Wen, Chen Wang, Erhan Bilal  

**Link**: [PDF](https://arxiv.org/pdf/2511.15717)  

**Abstract**: ARC-AGI and ARC-AGI-2 measure generalization-through-composition on small color-quantized grids, and their prize competitions make progress on these harder held-out tasks a meaningful proxy for systematic generalization. Recent instruction-first systems translate grids into concise natural-language or DSL rules executed in generate-execute-select loops, yet we lack a principled account of how encodings shape model perception and how to separate instruction errors from execution errors. We hypothesize that modality imposes perceptual bottlenecks -- text flattens 2D structure into 1D tokens while images preserve layout but can introduce patch-size aliasing -- thereby shaping which grid features are reliably perceived. To test this, we isolate perception from reasoning across nine text and image modalities using a weighted set-disagreement metric and a two-stage reasoning pipeline, finding that structured text yields precise coordinates on sparse features, images capture 2D shapes yet are resolution-sensitive, and combining them improves execution (about 8 perception points; about 0.20 median similarity). Overall, aligning representations with transformer inductive biases and enabling cross-validation between text and image yields more accurate instructions and more reliable execution without changing the underlying model. 

**Abstract (ZH)**: ARC-AGI和ARC-AGI-2通过小色彩量化网格测量组合泛化能力，并且他们的奖励竞赛将这些更困难的保留任务的进步作为系统泛化的有意义代理。最近的指令优先系统将网格转换为简洁的自然语言或DSL规则并在生成-执行-选择循环中执行，但缺乏关于编码如何塑造模型感知的原理性解释以及如何将指令错误与执行错误区分开来的论述。我们假设模态引入感知瓶颈——文本将二维结构压平为一维标记，而图像保留布局但可能引入块大小混叠——从而塑造哪些网格特征能够可靠地被感知。为了测试这一点，我们使用加权集不一致性度量和两阶段推理管道，隔离感知和推理，发现结构化文本在稀疏特征上提供精确的坐标，图像捕获二维形状但对分辨率敏感，结合它们可以改善执行（约8个感知点；中位数相似度约0.20）。总体而言，使表示与变压器归纳偏置对齐并在文本和图像之间实现交叉验证可以提高指令的准确性并增强执行的可靠性，而无需改变基础模型。 

---
# MACIE: Multi-Agent Causal Intelligence Explainer for Collective Behavior Understanding 

**Title (ZH)**: MACIE: 多智能体因果智能解释器用于集体行为理解 

**Authors**: Abraham Itzhak Weinberg  

**Link**: [PDF](https://arxiv.org/pdf/2511.15716)  

**Abstract**: As Multi Agent Reinforcement Learning systems are used in safety critical applications. Understanding why agents make decisions and how they achieve collective behavior is crucial. Existing explainable AI methods struggle in multi agent settings. They fail to attribute collective outcomes to individuals, quantify emergent behaviors, or capture complex interactions. We present MACIE Multi Agent Causal Intelligence Explainer, a framework combining structural causal models, interventional counterfactuals, and Shapley values to provide comprehensive explanations. MACIE addresses three questions. First, each agent's causal contribution using interventional attribution scores. Second, system level emergent intelligence through synergy metrics separating collective effects from individual contributions. Third, actionable explanations using natural language narratives synthesizing causal insights. We evaluate MACIE across four MARL scenarios: cooperative, competitive, and mixed motive. Results show accurate outcome attribution, mean phi_i equals 5.07, standard deviation less than 0.05, detection of positive emergence in cooperative tasks, synergy index up to 0.461, and efficient computation, 0.79 seconds per dataset on CPU. MACIE uniquely combines causal rigor, emergence quantification, and multi agent support while remaining practical for real time use. This represents a step toward interpretable, trustworthy, and accountable multi agent AI. 

**Abstract (ZH)**: 多智能体因果智能解释器：一种结合结构因果模型、干预反事实和Shapley值的框架 

---
# Graph-Memoized Reasoning: Foundations Structured Workflow Reuse in Intelligent Systems 

**Title (ZH)**: 图记忆推理：智能系统中结构化工作流重用的理论基础 

**Authors**: Yash Raj Singh  

**Link**: [PDF](https://arxiv.org/pdf/2511.15715)  

**Abstract**: Modern large language model-based reasoning systems frequently recompute similar reasoning steps across tasks, wasting computational resources, inflating inference latency, and limiting reproducibility. These inefficiencies underscore the need for persistent reasoning mechanisms that can recall and reuse prior computational traces.
We introduce Graph-Memoized Reasoning, a formal framework for representing, storing, and reusing reasoning workflows as graph-structured memory. By encoding past decision graphs and retrieving them through structural and semantic similarity, our approach enables compositional reuse of subgraphs across new reasoning tasks.
We formulate an optimization objective that minimizes total reasoning cost regularized by inconsistency between stored and generated workflows, providing a theoretical foundation for efficiency-consistency trade-offs in intelligent systems. We outline a conceptual evaluation protocol aligned with the proposed optimization objective.
This framework establishes the groundwork for interpretable, cost-efficient, and self-improving reasoning architectures, offering a step toward persistent memory in large-scale agentic systems. 

**Abstract (ZH)**: 基于现代大型语言模型的推理系统经常在不同任务中重新计算相似的推理步骤，浪费计算资源，增加推理延迟，并限制可重现性。这些低效性凸显了需要持久化的推理机制，以召回和复用先前的计算轨迹。

我们引入了图缓存推理（Graph-Memoized Reasoning），这是一种形式化框架，用于表示、存储和复用以图结构记忆形式的推理工作流。通过编码过去的决策图并通过结构和语义相似性进行检索，我们的方法能够在新的推理任务中组合复用子图。

我们提出了一个优化目标，该目标最小化总推理成本，并通过存储和生成工作流之间的一致性进行正则化，为此类智能系统的效率-一致性权衡提供了理论基础。我们概述了一个与提出的优化目标相一致的概念性评估协议。

该框架为可解释、低成本和自我改进的推理架构奠定了基础，为大型代理系统中的持久化记忆提供了一步之遥。 

---
# Majority Rules: LLM Ensemble is a Winning Approach for Content Categorization 

**Title (ZH)**: 多数裁决：大型语言模型集成是内容类别划分的胜出方法 

**Authors**: Ariel Kamen, Yakov Kamen  

**Link**: [PDF](https://arxiv.org/pdf/2511.15714)  

**Abstract**: This study introduces an ensemble framework for unstructured text categorization using large language models (LLMs). By integrating multiple models, the ensemble large language model (eLLM) framework addresses common weaknesses of individual systems, including inconsistency, hallucination, category inflation, and misclassification. The eLLM approach yields a substantial performance improvement of up to 65\% in F1-score over the strongest single model. We formalize the ensemble process through a mathematical model of collective decision-making and establish principled aggregation criteria. Using the Interactive Advertising Bureau (IAB) hierarchical taxonomy, we evaluate ten state-of-the-art LLMs under identical zero-shot conditions on a human-annotated corpus of 8{,}660 samples. Results show that individual models plateau in performance due to the compression of semantically rich text into sparse categorical representations, while eLLM improves both robustness and accuracy. With a diverse consortium of models, eLLM achieves near human-expert-level performance, offering a scalable and reliable solution for taxonomy-based classification that may significantly reduce dependence on human expert labeling. 

**Abstract (ZH)**: 本研究介绍了一种使用大规模语言模型（LLMs）进行非结构化文本分类的集成框架。通过集成多个模型，集成大规模语言模型（eLLM）框架解决了单个系统中常见的不一致性、幻觉、类别膨胀和误分类等问题。eLLM方法在F1分数上较最强单个模型提高了高达65%的性能。我们通过集体决策的数学模型正式化了集成过程，并建立了原则性的聚合标准。基于互动广告局（IAB）层次分类法，在相同零样本条件下对十个最先进的LLMs进行评估，使用了8,660个手工标注的数据样本。结果显示，个体模型由于语义丰富文本向稀疏类别表示的压缩而性能达到饱和，而eLLM则提高了稳定性和准确性。通过一个多元化的模型集合，eLLM实现了接近人类专家级别的性能，提供了一种可扩展且可靠的基于层次分类法的分类解决方案，可能显著减少对人类专家标注的依赖。 

---
# Dataset Distillation for Pre-Trained Self-Supervised Vision Models 

**Title (ZH)**: 预训练自我监督视觉模型的数据集精炼 

**Authors**: George Cazenavette, Antonio Torralba, Vincent Sitzmann  

**Link**: [PDF](https://arxiv.org/pdf/2511.16674)  

**Abstract**: The task of dataset distillation aims to find a small set of synthetic images such that training a model on them reproduces the performance of the same model trained on a much larger dataset of real samples. Existing distillation methods focus on synthesizing datasets that enable training randomly initialized models. In contrast, state-of-the-art vision approaches are increasingly building on large, pre-trained self-supervised models rather than training from scratch. In this paper, we investigate the problem of distilling datasets that enable us to optimally train linear probes on top of such large, pre-trained vision models. We introduce a method of dataset distillation for this task called Linear Gradient Matching that optimizes the synthetic images such that, when passed through a pre-trained feature extractor, they induce gradients in the linear classifier similar to those produced by the real data. Our method yields synthetic data that outperform all real-image baselines and, remarkably, generalize across pre-trained vision models, enabling us, for instance, to train a linear CLIP probe that performs competitively using a dataset distilled via a DINO backbone. Further, we show that our distilled datasets are exceptionally effective for fine-grained classification and provide a valuable tool for model interpretability, predicting, among other things, how similar two models' embedding spaces are under the platonic representation hypothesis or whether a model is sensitive to spurious correlations in adversarial datasets. 

**Abstract (ZH)**: 数据集蒸馏的目标是在找到一个小的合成图像集，使得在这些图像上训练的模型能够重现在大规模真实样本数据集上训练的同一模型的性能。现有的蒸馏方法侧重于合成能够训练随机初始化模型的数据集。相反，最新的视觉方法越来越多地依赖于大型的预训练自我监督模型，而不是从头开始训练。在本文中，我们探讨了蒸馏数据集的问题，该数据集使得我们能够在大型预训练视觉模型之上优化训练线性探针。我们引入了一种称为线性梯度匹配的数据集蒸馏方法，该方法优化合成图像，使得当通过预训练的特征提取器处理时，它们会在线性分类器中引发与真实数据相似的梯度。我们的方法生成的合成数据能够超越所有基于真实图片的基线，并且令人惊讶地能够在预训练视觉模型之间泛化，例如，我们能够使用通过DINO骨干蒸馏的数据集训练一个表现竞争力的线性CLIP探针。此外，我们展示了我们的蒸馏数据集在细粒度分类任务中具有极高的有效性，并为其模型可解释性提供了有价值的工具，能够在理想表示假设下预测两个模型的嵌入空间的相似性，或者判断模型是否对对抗数据集中的虚假相关性敏感。 

---
# Thinking-while-Generating: Interleaving Textual Reasoning throughout Visual Generation 

**Title (ZH)**: 生成思考：在整个视觉生成过程中交替进行文本推理 

**Authors**: Ziyu Guo, Renrui Zhang, Hongyu Li, Manyuan Zhang, Xinyan Chen, Sifan Wang, Yan Feng, Peng Pei, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2511.16671)  

**Abstract**: Recent advances in visual generation have increasingly explored the integration of reasoning capabilities. They incorporate textual reasoning, i.e., think, either before (as pre-planning) or after (as post-refinement) the generation process, yet they lack on-the-fly multimodal interaction during the generation itself. In this preliminary study, we introduce Thinking-while-Generating (TwiG), the first interleaved framework that enables co-evolving textual reasoning throughout the visual generation process. As visual content is progressively generating, textual reasoning is interleaved to both guide upcoming local regions and reflect on previously synthesized ones. This dynamic interplay produces more context-aware and semantically rich visual outputs. To unveil the potential of this framework, we investigate three candidate strategies, zero-shot prompting, supervised fine-tuning (SFT) on our curated TwiG-50K dataset, and reinforcement learning (RL) via a customized TwiG-GRPO strategy, each offering unique insights into the dynamics of interleaved reasoning. We hope this work inspires further research into interleaving textual reasoning for enhanced visual generation. Code will be released at: this https URL. 

**Abstract (ZH)**: Recent Advances in Visual Generation: Integrating On-the-fly Multimodal Interaction through Thinking-while-Generating 

---
# Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter 

**Title (ZH)**: 长尾现象驯化：自适应投手下的高效推理RL训练 

**Authors**: Qinghao Hu, Shang Yang, Junxian Guo, Xiaozhe Yao, Yujun Lin, Yuxian Gu, Han Cai, Chuang Gan, Ana Klimovic, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.16665)  

**Abstract**: The emergence of Large Language Models (LLMs) with strong reasoning capabilities marks a significant milestone, unlocking new frontiers in complex problem-solving. However, training these reasoning models, typically using Reinforcement Learning (RL), encounters critical efficiency bottlenecks: response generation during RL training exhibits a persistent long-tail distribution, where a few very long responses dominate execution time, wasting resources and inflating costs. To address this, we propose TLT, a system that accelerates reasoning RL training losslessly by integrating adaptive speculative decoding. Applying speculative decoding in RL is challenging due to the dynamic workloads, evolving target model, and draft model training overhead. TLT overcomes these obstacles with two synergistic components: (1) Adaptive Drafter, a lightweight draft model trained continuously on idle GPUs during long-tail generation to maintain alignment with the target model at no extra cost; and (2) Adaptive Rollout Engine, which maintains a memory-efficient pool of pre-captured CUDAGraphs and adaptively select suitable SD strategies for each input batch. Evaluations demonstrate that TLT achieves over 1.7x end-to-end RL training speedup over state-of-the-art systems, preserves the model accuracy, and yields a high-quality draft model as a free byproduct suitable for efficient deployment. Code is released at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）的出现标志着强大的推理能力的一个重要里程碑，开启了复杂问题解决的新前沿。然而，这些推理模型的训练通常使用强化学习（RL），遇到了关键的效率瓶颈：在RL训练过程中，响应生成表现出持久的长尾分布，其中少数非常长的响应主导了执行时间，浪费了资源并增加了成本。为了解决这个问题，我们提出了TLT系统，通过集成适应性推测解码无损加速推理RL训练。在RL中应用推测解码具有挑战性，因为它涉及到动态工作负载、不断变化的目标模型和草稿模型的训练开销。TLT通过两个协同工作的组件克服了这些障碍：（1）适应性草稿生成器，一个轻量级的草稿模型，在长尾生成期间连续在空闲GPU上训练，以无额外成本的方式保持与目标模型的一致性；（2）适应性展开引擎，维护一个高效内存池的预捕获CUDAGraphs，并为每个输入批次适配选择合适的推测解码策略。评估表明，TLT比现有系统的端到端RL训练速度提高了超过1.7倍，保持了模型的准确性，并且免费产生了高质量的草稿模型，适合高效部署。代码发布于此https URL。 

---
# Dexterity from Smart Lenses: Multi-Fingered Robot Manipulation with In-the-Wild Human Demonstrations 

**Title (ZH)**: 智能眼镜带来的灵巧性：基于野生环境下人类演示的多指机器人 manipulation 

**Authors**: Irmak Guzey, Haozhi Qi, Julen Urain, Changhao Wang, Jessica Yin, Krishna Bodduluri, Mike Lambeta, Lerrel Pinto, Akshara Rai, Jitendra Malik, Tingfan Wu, Akash Sharma, Homanga Bharadhwaj  

**Link**: [PDF](https://arxiv.org/pdf/2511.16661)  

**Abstract**: Learning multi-fingered robot policies from humans performing daily tasks in natural environments has long been a grand goal in the robotics community. Achieving this would mark significant progress toward generalizable robot manipulation in human environments, as it would reduce the reliance on labor-intensive robot data collection. Despite substantial efforts, progress toward this goal has been bottle-necked by the embodiment gap between humans and robots, as well as by difficulties in extracting relevant contextual and motion cues that enable learning of autonomous policies from in-the-wild human videos. We claim that with simple yet sufficiently powerful hardware for obtaining human data and our proposed framework AINA, we are now one significant step closer to achieving this dream. AINA enables learning multi-fingered policies from data collected by anyone, anywhere, and in any environment using Aria Gen 2 glasses. These glasses are lightweight and portable, feature a high-resolution RGB camera, provide accurate on-board 3D head and hand poses, and offer a wide stereo view that can be leveraged for depth estimation of the scene. This setup enables the learning of 3D point-based policies for multi-fingered hands that are robust to background changes and can be deployed directly without requiring any robot data (including online corrections, reinforcement learning, or simulation). We compare our framework against prior human-to-robot policy learning approaches, ablate our design choices, and demonstrate results across nine everyday manipulation tasks. Robot rollouts are best viewed on our website: this https URL. 

**Abstract (ZH)**: 从自然环境下执行日常任务的人类操作学习多手指机器人策略一直是机器人领域的宏伟目标。随着我们的简单而强大的硬件AINA框架，我们现在更接近这个梦想一步。AINA能够利用任何人随时随地收集的数据，在任意环境中学习多手指策略。Aria Gen 2眼镜轻便便携，配备高分辨率RGB相机，提供精准的3D头部和手部姿态，并具有宽广的立体视图，可用于场景深度估计。该设置使我们能够学习对背景变化具有鲁棒性的3D点基策略，并可以直接部署而无需任何机器人数据（包括在线修正、强化学习或模拟）。我们将我们的框架与先前的人类到机器人策略学习方法进行比较，分析我们的设计选择，并展示了九种日常操作任务的结果。机器人演示最佳查看网址：this https URL。 

---
# Teacher-Guided One-Shot Pruning via Context-Aware Knowledge Distillation 

**Title (ZH)**: 基于上下文aware知识蒸馏的教师引导一次性剪枝 

**Authors**: Md. Samiul Alim, Sharjil Khan, Amrijit Biswas, Fuad Rahman, Shafin Rahman, Nabeel Mohammed  

**Link**: [PDF](https://arxiv.org/pdf/2511.16653)  

**Abstract**: Unstructured pruning remains a powerful strategy for compressing deep neural networks, yet it often demands iterative train-prune-retrain cycles, resulting in significant computational overhead. To address this challenge, we introduce a novel teacher-guided pruning framework that tightly integrates Knowledge Distillation (KD) with importance score estimation. Unlike prior approaches that apply KD as a post-pruning recovery step, our method leverages gradient signals informed by the teacher during importance score calculation to identify and retain parameters most critical for both task performance and knowledge transfer. Our method facilitates a one-shot global pruning strategy that efficiently eliminates redundant weights while preserving essential representations. After pruning, we employ sparsity-aware retraining with and without KD to recover accuracy without reactivating pruned connections. Comprehensive experiments across multiple image classification benchmarks, including CIFAR-10, CIFAR-100, and TinyImageNet, demonstrate that our method consistently achieves high sparsity levels with minimal performance degradation. Notably, our approach outperforms state-of-the-art baselines such as EPG and EPSD at high sparsity levels, while offering a more computationally efficient alternative to iterative pruning schemes like COLT. The proposed framework offers a computation-efficient, performance-preserving solution well suited for deployment in resource-constrained environments. 

**Abstract (ZH)**: 一种基于教师引导的剪枝框架：结合知识蒸馏与重要性评分估计以高效压缩深度神经网络 

---
# Evolution Strategies at the Hyperscale 

**Title (ZH)**: 超大规模环境下的进化策略 

**Authors**: Bidipta Sarkar, Mattie Fellows, Juan Agustin Duque, Alistair Letcher, Antonio León Villares, Anya Sims, Dylan Cope, Jarek Liesen, Lukas Seier, Theo Wolf, Uljad Berdica, Alexander David Goldie, Aaron Courville, Karin Sevegnani, Shimon Whiteson, Jakob Nicolaus Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2511.16652)  

**Abstract**: We introduce Evolution Guided General Optimization via Low-rank Learning (EGGROLL), an evolution strategies (ES) algorithm designed to scale backprop-free optimization to large population sizes for modern large neural network architectures with billions of parameters. ES is a set of powerful blackbox optimisation methods that can handle non-differentiable or noisy objectives with excellent scaling potential through parallelisation. Na{ï}ve ES becomes prohibitively expensive at scale due to the computational and memory costs associated with generating matrix perturbations $E\in\mathbb{R}^{m\times n}$ and the batched matrix multiplications needed to compute per-member forward passes. EGGROLL overcomes these bottlenecks by generating random matrices $A\in \mathbb{R}^{m\times r},\ B\in \mathbb{R}^{n\times r}$ with $r\ll \min(m,n)$ to form a low-rank matrix perturbation $A B^\top$ that are used in place of the full-rank perturbation $E$. As the overall update is an average across a population of $N$ workers, this still results in a high-rank update but with significant memory and computation savings, reducing the auxiliary storage from $mn$ to $r(m+n)$ per layer and the cost of a forward pass from $\mathcal{O}(mn)$ to $\mathcal{O}(r(m+n))$ when compared to full-rank ES. A theoretical analysis reveals our low-rank update converges to the full-rank update at a fast $\mathcal{O}\left(\frac{1}{r}\right)$ rate. Our experiments show that (1) EGGROLL does not compromise the performance of ES in tabula-rasa RL settings, despite being faster, (2) it is competitive with GRPO as a technique for improving LLM reasoning, and (3) EGGROLL enables stable pre-training of nonlinear recurrent language models that operate purely in integer datatypes. 

**Abstract (ZH)**: Evolution引导的低秩学习大规模优化（EGGROLL）：面向现代大参数神经网络架构的目标优化算法 

---
# Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables 

**Title (ZH)**: 更快的认证对称性打破方法：辅助变量与顺序的结合 

**Authors**: Markus Anders, Bart Bogaerts, Benjamin Bogø, Arthur Gontier, Wietze Koops, Ciaran McCreesh, Magnus O. Myreen, Jakob Nordström, Andy Oertel, Adrian Rebola-Pardo, Yong Kiam Tan  

**Link**: [PDF](https://arxiv.org/pdf/2511.16637)  

**Abstract**: Symmetry breaking is a crucial technique in modern combinatorial solving, but it is difficult to be sure it is implemented correctly. The most successful approach to deal with bugs is to make solvers certifying, so that they output not just a solution, but also a mathematical proof of correctness in a standard format, which can then be checked by a formally verified checker. This requires justifying symmetry reasoning within the proof, but developing efficient methods for this has remained a long-standing open challenge. A fully general approach was recently proposed by Bogaerts et al. (2023), but it relies on encoding lexicographic orders with big integers, which quickly becomes infeasible for large symmetries. In this work, we develop a method for instead encoding orders with auxiliary variables. We show that this leads to orders-of-magnitude speed-ups in both theory and practice by running experiments on proof logging and checking for SAT symmetry breaking using the state-of-the-art satsuma symmetry breaker and the VeriPB proof checking toolchain. 

**Abstract (ZH)**: 打破对称性是现代组合求解中的关键技术，但确保其实现正确性颇具挑战。处理错误的最成功方法是使求解器具备证明能力，不仅输出解决方案，还输出标准格式的正确性数学证明，这些证明可以被形式验证的检查器验证。这要求在证明中验证对称性推理，但开发高效方法解决这一问题仍是一个长期开放的挑战。Bogaerts等人（2023）最近提出了一种通用方法，但这种方法依赖于使用大整数编码字典序，这很快在处理大规模对称性时变得不可行。在本文中，我们开发了一种用辅助变量编码顺序的方法。我们通过在最先进的satsuma对称性打破工具和VeriPB证明检查工具链上进行证明记录和验证实验，展示了这种方法在理论和实践上都带来了数量级的速度提升。 

---
# Stabilizing Policy Gradient Methods via Reward Profiling 

**Title (ZH)**: 通过奖励分析稳定策略梯度方法 

**Authors**: Shihab Ahmed, El Houcine Bergou, Aritra Dutta, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16629)  

**Abstract**: Policy gradient methods, which have been extensively studied in the last decade, offer an effective and efficient framework for reinforcement learning problems. However, their performances can often be unsatisfactory, suffering from unreliable reward improvements and slow convergence, due to high variance in gradient estimations. In this paper, we propose a universal reward profiling framework that can be seamlessly integrated with any policy gradient algorithm, where we selectively update the policy based on high-confidence performance estimations. We theoretically justify that our technique will not slow down the convergence of the baseline policy gradient methods, but with high probability, will result in stable and monotonic improvements of their performance. Empirically, on eight continuous-control benchmarks (Box2D and MuJoCo/PyBullet), our profiling yields up to 1.5x faster convergence to near-optimal returns, up to 1.75x reduction in return variance on some setups. Our profiling approach offers a general, theoretically grounded path to more reliable and efficient policy learning in complex environments. 

**Abstract (ZH)**: 基于策略梯度方法的通用奖励剖析框架：提高策略学习的可靠性和效率 

---
# SAM 3D: 3Dfy Anything in Images 

**Title (ZH)**: SAM 3D: 图像中anything的3D化 

**Authors**: SAM 3D Team, Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, Aohan Lin, Jiawei Liu, Ziqi Ma, Anushka Sagar, Bowen Song, Xiaodong Wang, Jianing Yang, Bowen Zhang, Piotr Dollár, Georgia Gkioxari, Matt Feiszli, Jitendra Malik  

**Link**: [PDF](https://arxiv.org/pdf/2511.16624)  

**Abstract**: We present SAM 3D, a generative model for visually grounded 3D object reconstruction, predicting geometry, texture, and layout from a single image. SAM 3D excels in natural images, where occlusion and scene clutter are common and visual recognition cues from context play a larger role. We achieve this with a human- and model-in-the-loop pipeline for annotating object shape, texture, and pose, providing visually grounded 3D reconstruction data at unprecedented scale. We learn from this data in a modern, multi-stage training framework that combines synthetic pretraining with real-world alignment, breaking the 3D "data barrier". We obtain significant gains over recent work, with at least a 5:1 win rate in human preference tests on real-world objects and scenes. We will release our code and model weights, an online demo, and a new challenging benchmark for in-the-wild 3D object reconstruction. 

**Abstract (ZH)**: SAM 3D：一种基于视觉的三维物体重建生成模型 

---
# Improving Long-Tailed Object Detection with Balanced Group Softmax and Metric Learning 

**Title (ZH)**: 改进长尾目标检测：基于平衡组Softmax和度量学习的方法 

**Authors**: Satyam Gaba  

**Link**: [PDF](https://arxiv.org/pdf/2511.16619)  

**Abstract**: Object detection has been widely explored for class-balanced datasets such as COCO. However, real-world scenarios introduce the challenge of long-tailed distributions, where numerous categories contain only a few instances. This inherent class imbalance biases detection models towards the more frequent classes, degrading performance on rare categories. In this paper, we tackle the problem of long-tailed 2D object detection using the LVISv1 dataset, which consists of 1,203 categories and 164,000 images. We employ a two-stage Faster R-CNN architecture and propose enhancements to the Balanced Group Softmax (BAGS) framework to mitigate class imbalance. Our approach achieves a new state-of-the-art performance with a mean Average Precision (mAP) of 24.5%, surpassing the previous benchmark of 24.0%.
Additionally, we hypothesize that tail class features may form smaller, denser clusters within the feature space of head classes, making classification challenging for regression-based classifiers. To address this issue, we explore metric learning to produce feature embeddings that are both well-separated across classes and tightly clustered within each class. For inference, we utilize a k-Nearest Neighbors (k-NN) approach to improve classification performance, particularly for rare classes. Our results demonstrate the effectiveness of these methods in advancing long-tailed object detection. 

**Abstract (ZH)**: 长尾分布2D物体检测中的挑战及解决方案：基于LVISv1数据集的研究 

---
# Generative AI for Enhanced Wildfire Detection: Bridging the Synthetic-Real Domain Gap 

**Title (ZH)**: 增强 wildfires 检测的生成 AI：弥合合成-真实领域差距 

**Authors**: Satyam Gaba  

**Link**: [PDF](https://arxiv.org/pdf/2511.16617)  

**Abstract**: The early detection of wildfires is a critical environmental challenge, with timely identification of smoke plumes being key to mitigating large-scale damage. While deep neural networks have proven highly effective for localization tasks, the scarcity of large, annotated datasets for smoke detection limits their potential. In response, we leverage generative AI techniques to address this data limitation by synthesizing a comprehensive, annotated smoke dataset. We then explore unsupervised domain adaptation methods for smoke plume segmentation, analyzing their effectiveness in closing the gap between synthetic and real-world data. To further refine performance, we integrate advanced generative approaches such as style transfer, Generative Adversarial Networks (GANs), and image matting. These methods aim to enhance the realism of synthetic data and bridge the domain disparity, paving the way for more accurate and scalable wildfire detection models. 

**Abstract (ZH)**: 早期野火检测是关键的环境挑战，及时识别烟雾柱对于减轻大规模损害至关重要。虽然深度神经网络在定位任务中表现出高度有效性，但由于烟雾检测标注数据集的稀缺性限制了其潜在应用。为此，我们利用生成对抗技术解决这一数据局限性，合成一个全面且标注的烟雾数据集。然后，我们探讨无监督领域适应方法在烟雾柱分割中的应用，分析其在缩小合成数据与实际数据差距方面的有效性。为进一步提高性能，我们整合了先进的生成方法，如风格迁移、生成对抗网络（GANs）和图像抠图技术，旨在增强合成数据的逼真度并弥合领域差异，从而为更准确和可扩展的野火检测模型铺平道路。 

---
# TimeViper: A Hybrid Mamba-Transformer Vision-Language Model for Efficient Long Video Understanding 

**Title (ZH)**: TimeViper：一种高效长视频理解的混合Mamba-Transformer视觉语言模型 

**Authors**: Boshen Xu, Zihan Xiao, Jiaze Li, Jianzhong Ju, Zhenbo Luo, Jian Luan, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2511.16595)  

**Abstract**: We introduce TimeViper, a hybrid vision-language model designed to tackle challenges of long video understanding. Processing long videos demands both an efficient model architecture and an effective mechanism for handling extended temporal contexts. To this end, TimeViper adopts a hybrid Mamba-Transformer backbone that combines the efficiency of state-space models with the expressivity of attention mechanisms. Through this hybrid design, we reveal the vision-to-text information aggregation phenomenon, where information progressively flows from vision tokens to text tokens across increasing LLM depth, resulting in severe vision token redundancy. Motivated by this observation, we propose TransV, a token information transfer module that transfers and compresses vision tokens into instruction tokens while maintaining multimodal understanding capabilities. This design enables TimeViper to process hour-long videos exceeding 10,000 frames. Extensive experiments across multiple benchmarks demonstrate that TimeViper competes with state-of-the-art models while extending frame numbers. We further analyze attention behaviors of both Mamba and Transformer layers, offering new insights into hybrid model interpretability. This work represents an initial step towards developing, interpreting, and compressing hybrid Mamba-Transformer architectures. 

**Abstract (ZH)**: TimeViper：一种用于长视频理解的混合视觉-语言模型 

---
# Green Resilience of Cyber-Physical Systems: Doctoral Dissertation 

**Title (ZH)**: 网络物理系统的绿色韧性：博士论文 

**Authors**: Diaeddin Rimawi  

**Link**: [PDF](https://arxiv.org/pdf/2511.16593)  

**Abstract**: Cyber-physical systems (CPS) combine computational and physical components. Online Collaborative AI System (OL-CAIS) is a type of CPS that learn online in collaboration with humans to achieve a common goal, which makes it vulnerable to disruptive events that degrade performance. Decision-makers must therefore restore performance while limiting energy impact, creating a trade-off between resilience and greenness. This research addresses how to balance these two properties in OL-CAIS. It aims to model resilience for automatic state detection, develop agent-based policies that optimize the greenness-resilience trade-off, and understand catastrophic forgetting to maintain performance consistency. We model OL-CAIS behavior through three operational states: steady, disruptive, and final. To support recovery during disruptions, we introduce the GResilience framework, which provides recovery strategies through multi-objective optimization (one-agent), game-theoretic decision-making (two-agent), and reinforcement learning (RL-agent). We also design a measurement framework to quantify resilience and greenness. Empirical evaluation uses real and simulated experiments with a collaborative robot learning object classification from human demonstrations. Results show that the resilience model captures performance transitions during disruptions, and that GResilience policies improve green recovery by shortening recovery time, stabilizing performance, and reducing human dependency. RL-agent policies achieve the strongest results, although with a marginal increase in CO2 emissions. We also observe catastrophic forgetting after repeated disruptions, while our policies help maintain steadiness. A comparison with containerized execution shows that containerization cuts CO2 emissions by half. Overall, this research provides models, metrics, and policies that ensure the green recovery of OL-CAIS. 

**Abstract (ZH)**: 基于物理系统的在线协作AI系统(CPS)中韧性和绿色性的平衡研究 

---
# Synthesis of Safety Specifications for Probabilistic Systems 

**Title (ZH)**: 概率系统安全性规格的合成 

**Authors**: Gaspard Ohlmann, Edwin Hamel-De le Court, Francesco Belardinelli  

**Link**: [PDF](https://arxiv.org/pdf/2511.16579)  

**Abstract**: Ensuring that agents satisfy safety specifications can be crucial in safety-critical environments. While methods exist for controller synthesis with safe temporal specifications, most existing methods restrict safe temporal specifications to probabilistic-avoidance constraints. Formal methods typically offer more expressive ways to express safety in probabilistic systems, such as Probabilistic Computation Tree Logic (PCTL) formulas. Thus, in this paper, we develop a new approach that supports more general temporal properties expressed in PCTL. Our contribution is twofold. First, we develop a theoretical framework for the Synthesis of safe-PCTL specifications. We show how the reducing global specification satisfaction to local constraints, and define CPCTL, a fragment of safe-PCTL. We demonstrate how the expressiveness of CPCTL makes it a relevant fragment for the Synthesis Problem. Second, we leverage these results and propose a new Value Iteration-based algorithm to solve the synthesis problem for these more general temporal properties, and we prove the soundness and completeness of our method. 

**Abstract (ZH)**: 确保智能体满足安全规范在安全关键环境中至关重要。虽然已存在针对安全时序规范的控制器综合方法，但大多数现有方法仅限制安全时序规范为概率避险约束。形式化方法通常能够以更表达的方式来描述概率系统中的安全性，例如概率计算树逻辑（PCTL）公式。因此，在本文中，我们开发了一种新的方法，支持由PCTL表达的更广泛的时序性质。我们的贡献主要有两个方面。首先，我们为安全PCTL规范的综合开发了一个理论框架。我们展示了如何将全局规范满足问题缩减为局部约束，并定义了CPCTL，它是安全PCTL的一个片段。我们证明了CPCTL的表达力使其成为综合问题的相关片段。其次，我们利用这些结果提出了一种新的基于值迭代的算法来解决这些更广泛的时序性质的综合问题，并证明了我们方法的正确性和完备性。 

---
# Integrating Symbolic Natural Language Understanding and Language Models for Word Sense Disambiguation 

**Title (ZH)**: 整合符号自然语言理解和语言模型进行词义消歧 

**Authors**: Kexin Zhao, Ken Forbus  

**Link**: [PDF](https://arxiv.org/pdf/2511.16577)  

**Abstract**: Word sense disambiguation is a fundamental challenge in natural language understanding. Current methods are primarily aimed at coarse-grained representations (e.g. WordNet synsets or FrameNet frames) and require hand-annotated training data to construct. This makes it difficult to automatically disambiguate richer representations (e.g. built on OpenCyc) that are needed for sophisticated inference. We propose a method that uses statistical language models as oracles for disambiguation that does not require any hand-annotation of training data. Instead, the multiple candidate meanings generated by a symbolic NLU system are converted into distinguishable natural language alternatives, which are used to query an LLM to select appropriate interpretations given the linguistic context. The selected meanings are propagated back to the symbolic NLU system. We evaluate our method against human-annotated gold answers to demonstrate its effectiveness. 

**Abstract (ZH)**: 词义消歧是自然语言理解中的一个基本挑战。现有方法主要针对粗粒度表示（如WordNet同义集或FrameNet框架）并需要人工标注的训练数据来构建。这使得自动消歧更丰富的表示（如基于OpenCyc构建的表示）变得困难，而这些表示对于复杂的推理是必要的。我们提出了一种方法，该方法使用统计语言模型作为消歧的先知，无需任何人工标注的训练数据。相反，符号化的自然语言理解系统生成的多个候选意义被转换为可区分的自然语言替代选项，然后用于查询LLM以根据语言上下文选择适当的解释。选定的意义随后被传递回符号化的自然语言理解系统。我们通过与人工标注的正确答案进行比较来评估该方法的有效性。 

---
# ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions 

**Title (ZH)**: ECPv2: 快速、高效且可扩展的Lipschitz函数全局优化方法 

**Authors**: Fares Fourati, Mohamed-Slim Alouini, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2511.16575)  

**Abstract**: We propose ECPv2, a scalable and theoretically grounded algorithm for global optimization of Lipschitz-continuous functions with unknown Lipschitz constants. Building on the Every Call is Precious (ECP) framework, which ensures that each accepted function evaluation is potentially informative, ECPv2 addresses key limitations of ECP, including high computational cost and overly conservative early behavior. ECPv2 introduces three innovations: (i) an adaptive lower bound to avoid vacuous acceptance regions, (ii) a Worst-m memory mechanism that restricts comparisons to a fixed-size subset of past evaluations, and (iii) a fixed random projection to accelerate distance computations in high dimensions. We theoretically show that ECPv2 retains ECP's no-regret guarantees with optimal finite-time bounds and expands the acceptance region with high probability. We further empirically validate these findings through extensive experiments and ablation studies. Using principled hyperparameter settings, we evaluate ECPv2 across a wide range of high-dimensional, non-convex optimization problems. Across benchmarks, ECPv2 consistently matches or outperforms state-of-the-art optimizers, while significantly reducing wall-clock time. 

**Abstract (ZH)**: 我们提出ECPv2，一种适用于未知Lipschitz常数的Lipschitz连续函数全局优化的可扩展且具有理论依据的算法。基于每次调用都珍贵（ECP）框架，该框架确保每次接受的函数评估都有潜在的信息价值，ECPv2解决了ECP的若干关键局限性，包括计算成本高和早期行为过于保守。ECPv2引入了三项创新：（i）自适应下界以避免无效的接受区域，（ii）Worst-m记忆机制，限制比较仅在过去的固定数量的评估中进行，以及（iii）固定随机投影以加速高维空间中的距离计算。我们从理论上证明，ECPv2保留了ECP的无遗憾保证，并且在最优的有限时间界内扩展了接受区域，同时以高概率进行。此外，我们通过广泛的实验和消融研究进一步实证验证了这些发现。通过原理上的超参数设置，我们评估了ECPv2在一系列高维非凸优化问题中的表现。在各种基准测试中，ECPv2一致地与最先进的优化器持平或表现出色，同时显著减少了wall-clock时间。 

---
# NutriScreener: Retrieval-Augmented Multi-Pose Graph Attention Network for Malnourishment Screening 

**Title (ZH)**: NutriScreener: 检索增强多姿态图注意力网络用于营养不良筛查 

**Authors**: Misaal Khan, Mayank Vatsa, Kuldeep Singh, Richa Singh  

**Link**: [PDF](https://arxiv.org/pdf/2511.16566)  

**Abstract**: Child malnutrition remains a global crisis, yet existing screening methods are laborious and poorly scalable, hindering early intervention. In this work, we present NutriScreener, a retrieval-augmented, multi-pose graph attention network that combines CLIP-based visual embeddings, class-boosted knowledge retrieval, and context awareness to enable robust malnutrition detection and anthropometric prediction from children's images, simultaneously addressing generalizability and class imbalance. In a clinical study, doctors rated it 4.3/5 for accuracy and 4.6/5 for efficiency, confirming its deployment readiness in low-resource settings. Trained and tested on 2,141 children from AnthroVision and additionally evaluated on diverse cross-continent populations, including ARAN and an in-house collected CampusPose dataset, it achieves 0.79 recall, 0.82 AUC, and significantly lower anthropometric RMSEs, demonstrating reliable measurement in unconstrained pediatric settings. Cross-dataset results show up to 25% recall gain and up to 3.5 cm RMSE reduction using demographically matched knowledge bases. NutriScreener offers a scalable and accurate solution for early malnutrition detection in low-resource environments. 

**Abstract (ZH)**: 儿童营养不良仍是全球性危机，现有筛查方法繁琐且难以扩展，妨碍了早期干预。在本工作中，我们提出了NutriScreener，这是一种检索增强的多姿态图注意力网络，结合了基于CLIP的视觉嵌入、类增强的知识检索和上下文意识，以从儿童图像中实现稳健的营养不良检测和人体测量预测，同时解决了泛化能力和类别不平衡问题。在临床研究中，医生对其准确性的评分是4.3/5，对其效率的评分是4.6/5，证实了其在低资源环境中的部署准备度。该模型在AnthroVision的2,141名儿童上进行训练和测试，并在包括ARAN和内部收集的CampusPose数据集在内的多元跨地域人群中进行了额外评估，实现了0.79的召回率、0.82的AUC以及显著较低的人体测量RMSE，展示了在非受限儿童环境中的可靠测量能力。跨数据集结果显示，使用人口匹配的知识库可获得多达25%的召回率提升和多达3.5 cm的RMSE减少。NutriScreener为低资源环境中的早期营养不良检测提供了可扩展且准确的解决方案。 

---
# Interfacial and bulk switching MoS2 memristors for an all-2D reservoir computing framework 

**Title (ZH)**: 界面和体相切换MoS2忆阻器用于全二维水库计算框架 

**Authors**: Asmita S. Thool, Sourodeep Roy, Prahalad Kanti Barman, Kartick Biswas, Pavan Nukala, Abhishek Misra, Saptarshi Das, and Bhaswar Chakrabarti  

**Link**: [PDF](https://arxiv.org/pdf/2511.16557)  

**Abstract**: In this study, we design a reservoir computing (RC) network by exploiting short- and long-term memory dynamics in Au/Ti/MoS$_2$/Au memristive devices. The temporal dynamics is engineered by controlling the thickness of the Chemical Vapor Deposited (CVD) MoS$_2$ films. Devices with a monolayer (1L)-MoS$_2$ film exhibit volatile (short-term memory) switching dynamics. We also report non-volatile resistance switching with excellent uniformity and analog behavior in conductance tuning for the multilayer (ML) MoS$_2$ memristive devices. We correlate this performance with trap-assisted space-charge limited conduction (SCLC) mechanism, leading to a bulk-limited resistance switching behavior. Four-bit reservoir states are generated using volatile memristors. The readout layer is implemented with an array of nonvolatile synapses. This small RC network achieves 89.56\% precision in a spoken-digit recognition task and is also used to analyze a nonlinear time series equation. 

**Abstract (ZH)**: 在本研究中，我们通过利用Au/Ti/MoS$_2$/Au memristive器件中的短时和长时间记忆动力学，设计了一种 reservoir 计算网络。时间动力学通过控制化学气相沉积（CVD）MoS$_2$薄膜的厚度进行工程设计。仅一层（1L）MoS$_2$薄膜的器件表现出易失性（短时记忆）切换动力学。我们还报告了多层（ML）MoS$_2$ memristive器件中优良均匀性和模拟行为的非易失性电阻切换。我们将这种性能归因于陷阱辅助的空间电荷限制传导（SCLC）机制，导致体限电阻切换行为。使用易失性 memristor 生成四比特 reservoir 状态。读出层通过非易失性突触阵列实现。这个小型 reservoir 计算网络在语音数字识别任务中达到了 89.56% 的精度，并且也被用于分析非线性时间序列方程。 

---
# WER is Unaware: Assessing How ASR Errors Distort Clinical Understanding in Patient Facing Dialogue 

**Title (ZH)**: WER不知情：评估ASR错误如何在面向患者的对话中扭曲临床理解 

**Authors**: Zachary Ellis, Jared Joselowitz, Yash Deo, Yajie He, Anna Kalygina, Aisling Higham, Mana Rahimzadeh, Yan Jia, Ibrahim Habli, Ernest Lim  

**Link**: [PDF](https://arxiv.org/pdf/2511.16544)  

**Abstract**: As Automatic Speech Recognition (ASR) is increasingly deployed in clinical dialogue, standard evaluations still rely heavily on Word Error Rate (WER). This paper challenges that standard, investigating whether WER or other common metrics correlate with the clinical impact of transcription errors. We establish a gold-standard benchmark by having expert clinicians compare ground-truth utterances to their ASR-generated counterparts, labeling the clinical impact of any discrepancies found in two distinct doctor-patient dialogue datasets. Our analysis reveals that WER and a comprehensive suite of existing metrics correlate poorly with the clinician-assigned risk labels (No, Minimal, or Significant Impact). To bridge this evaluation gap, we introduce an LLM-as-a-Judge, programmatically optimized using GEPA to replicate expert clinical assessment. The optimized judge (Gemini-2.5-Pro) achieves human-comparable performance, obtaining 90% accuracy and a strong Cohen's $\kappa$ of 0.816. This work provides a validated, automated framework for moving ASR evaluation beyond simple textual fidelity to a necessary, scalable assessment of safety in clinical dialogue. 

**Abstract (ZH)**: 自动语音识别（ASR）在临床对话中越来越广泛地应用，但标准评估仍然主要依赖词错误率（WER）。本文挑战这一标准，探讨转录错误的临床影响是否与WER或其他常见指标相关。我们通过让专家临床医生对比真实语音片段与其ASR生成的版本，并在两个不同的医生-患者对话数据集中标记发现的任何差异的临床影响，建立了一个黄金标准基准。我们的分析显示，WER和现有的各种综合指标与临床医生分配的风险标签（无影响、轻微影响或显著影响）的相关性较差。为了弥合这一评估缺口，我们引入了一个基于LLM的法官，使用GEPA程序化优化以复制专家临床评估。优化后的法官（Gemini-2.5-Pro）达到了与人类相当的性能，准确率为90%，Cohen's $\kappa$值为0.816。本文提供了一个经验证的自动化框架，将ASR评估从简单的文本忠实度提高到临床对话中必要的、可扩展的安全性评估。 

---
# The Oracle and The Prism: A Decoupled and Efficient Framework for Generative Recommendation Explanation 

**Title (ZH)**: acle和棱镜：一种解耦且高效的生成推荐解释框架 

**Authors**: Jiaheng Zhang, Daqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16543)  

**Abstract**: The integration of Large Language Models (LLMs) into explainable recommendation systems often leads to a performance-efficiency trade-off in end-to-end architectures, where joint optimization of ranking and explanation can result in suboptimal compromises. To resolve this, we propose Prism, a novel decoupled framework that rigorously separates the recommendation process into a dedicated ranking stage and an explanation generation stage.
Inspired by knowledge distillation, Prism leverages a powerful teacher LLM (e.g., FLAN-T5-XXL) as an Oracle to produce high-fidelity explanatory knowledge. A compact, fine-tuned student model (e.g., BART-Base), the Prism, then specializes in synthesizing this knowledge into personalized explanations. This decomposition ensures that each component is optimized for its specific objective, eliminating inherent conflicts in coupled models.
Extensive experiments on benchmark datasets demonstrate that our 140M-parameter Prism model significantly outperforms its 11B-parameter teacher in human evaluations of faithfulness and personalization, while achieving a 24 times speedup and a 10 times reduction in memory consumption during inference. These results validate that decoupling, coupled with targeted distillation, provides an efficient and effective pathway to high-quality explainable recommendation. 

**Abstract (ZH)**: 大型语言模型（LLMs）在可解释推荐系统中的集成通常会导致端到端架构中的性能-效率权衡，在此架构中，排名和解释的联合优化可能导致次优化的妥协。为了解决这一问题，我们提出了Prism，一种新颖的解耦框架，该框架严格将推荐过程分解为专门的排名阶段和解释生成阶段。

Prism借鉴知识蒸馏的理念，利用一个强大的教师LLM（例如FLAN-T5-XXL）作为Oracle，生成高保真解释性知识。一个紧凑且微调的小型学生模型（例如BART-Base）Prism，则专门用于将这些知识合成个性化解释。这种分解确保每个组件都针对其特定目标进行优化，从而消除耦合模型中的固有冲突。

广泛的基准数据集实验表明，我们的140M参数Prism模型在人类评估的忠实性和个性化方面显著优于其11B参数的教师模型，同时推理过程中实现了24倍的加速和10倍的内存消耗减少。这些结果验证了解耦与针对性蒸馏相结合为高质量可解释推荐提供了一条高效且有效的方法。 

---
# Supervised Contrastive Learning for Few-Shot AI-Generated Image Detection and Attribution 

**Title (ZH)**: 监督对比学习在少量样本AI生成图像检测与归属中的应用 

**Authors**: Jaime Álvarez Urueña, David Camacho, Javier Huertas Tato  

**Link**: [PDF](https://arxiv.org/pdf/2511.16541)  

**Abstract**: The rapid advancement of generative artificial intelligence has enabled the creation of synthetic images that are increasingly indistinguishable from authentic content, posing significant challenges for digital media integrity. This problem is compounded by the accelerated release cycle of novel generative models, which renders traditional detection approaches (reliant on periodic retraining) computationally infeasible and operationally impractical.
This work proposes a novel two-stage detection framework designed to address the generalization challenge inherent in synthetic image detection. The first stage employs a vision deep learning model trained via supervised contrastive learning to extract discriminative embeddings from input imagery. Critically, this model was trained on a strategically partitioned subset of available generators, with specific architectures withheld from training to rigorously ablate cross-generator generalization capabilities. The second stage utilizes a k-nearest neighbors (k-NN) classifier operating on the learned embedding space, trained in a few-shot learning paradigm incorporating limited samples from previously unseen test generators.
With merely 150 images per class in the few-shot learning regime, which are easily obtainable from current generation models, the proposed framework achieves an average detection accuracy of 91.3\%, representing a 5.2 percentage point improvement over existing approaches . For the source attribution task, the proposed approach obtains improvements of of 14.70\% and 4.27\% in AUC and OSCR respectively on an open set classification context, marking a significant advancement toward robust, scalable forensic attribution systems capable of adapting to the evolving generative AI landscape without requiring exhaustive retraining protocols. 

**Abstract (ZH)**: 生成人工智能的快速进步使得合成图像越来越难以与真实内容区分开来，对数字媒体完整性构成了重大挑战。这个问题因新型生成模型的加速发布周期而加剧，使得依赖于定期重新训练的传统检测方法在计算上不可行且操作上不实用。

本文提出了一种新颖的两阶段检测框架，旨在解决合成图像检测中的泛化挑战。第一阶段采用通过监督对比学习训练的视觉深度学习模型，从输入图像中提取具有区分性的嵌入表示。关键在于，该模型是在战略分割的可用生成器子集上进行训练的，特定的架构被排除在训练之外，以严格消除跨生成器的泛化能力。第二阶段利用在少量未见过的测试生成器样本中进行少量样本学习范式下训练的k-最邻近(k-NN)分类器，在学习嵌入空间中进行操作。

在少量样本学习范式中，每个类别仅有150张图像，这些图像很容易从当前的生成模型中获得，所提出的框架实现了平均检测准确率91.3%，比现有方法提高了5.2个百分点。在开放集分类的上下文中，对于源归属任务，所提出的方法分别在AUC和OSCR上取得了14.70%和4.27%的改进，标志着朝着适应演化中的生成AI环境的稳健、可扩展的法医归属系统迈出了重要一步，而不需要耗尽性的重新训练协议。 

---
# TurkColBERT: A Benchmark of Dense and Late-Interaction Models for Turkish Information Retrieval 

**Title (ZH)**: TurkColBERT：一种用于土耳其信息检索的密集表示和晚期交互模型基准 

**Authors**: Özay Ezerceli, Mahmoud El Hussieni, Selva Taş, Reyhan Bayraktar, Fatma Betül Terzioğlu, Yusuf Çelebi, Yağız Asker  

**Link**: [PDF](https://arxiv.org/pdf/2511.16528)  

**Abstract**: Neural information retrieval systems excel in high-resource languages but remain underexplored for morphologically rich, lower-resource languages such as Turkish. Dense bi-encoders currently dominate Turkish IR, yet late-interaction models -- which retain token-level representations for fine-grained matching -- have not been systematically evaluated. We introduce TurkColBERT, the first comprehensive benchmark comparing dense encoders and late-interaction models for Turkish retrieval. Our two-stage adaptation pipeline fine-tunes English and multilingual encoders on Turkish NLI/STS tasks, then converts them into ColBERT-style retrievers using PyLate trained on MS MARCO-TR. We evaluate 10 models across five Turkish BEIR datasets covering scientific, financial, and argumentative domains. Results show strong parameter efficiency: the 1.0M-parameter colbert-hash-nano-tr is 600$\times$ smaller than the 600M turkish-e5-large dense encoder while preserving over 71\% of its average mAP. Late-interaction models that are 3--5$\times$ smaller than dense encoders significantly outperform them; ColmmBERT-base-TR yields up to +13.8\% mAP on domain-specific tasks. For production-readiness, we compare indexing algorithms: MUVERA+Rerank is 3.33$\times$ faster than PLAID and offers +1.7\% relative mAP gain. This enables low-latency retrieval, with ColmmBERT-base-TR achieving 0.54 ms query times under MUVERA. We release all checkpoints, configs, and evaluation scripts. Limitations include reliance on moderately sized datasets ($\leq$50K documents) and translated benchmarks, which may not fully reflect real-world Turkish retrieval conditions; larger-scale MUVERA evaluations remain necessary. 

**Abstract (ZH)**: 神经信息检索系统在高资源语言中表现出色，但在如土耳其语这样形态丰富、资源较少的语言中仍待探索。目前，稠密双编码器在土耳其语IR中占据主导地位，然而晚期交互模型——这些模型保留了子单元级别的表示以实现精细匹配——尚未进行系统评估。我们引入TurkColBERT，这是首个全面比较稠密编码器和晚期交互模型的基准，用于土耳其语检索。我们的两阶段适应管道在土耳其语NLI/STS任务上微调英语文本和多语言编码器，然后使用PyLate在MS MARCO-TR上对其进行转换，生成ColBERT风格的检索器。我们评估了10个模型在涵盖科学、金融和论辩领域的五个Turkish BEIR数据集上的表现。结果显示，参数效率极高：100万参数的ColBERT-hash-nano-tr比6亿参数的Turkish-e5-large稠密编码器小600倍，同时保持了其平均mAP的71%以上。相比稠密编码器，其规模小3-5倍的晚期交互模型显著优于后者；ColmmBERT-base-TR在特定领域任务上可将mAP提高多达13.8%。为了提高生产可用性，我们比较了索引算法：MUVERA+Rerank比PLAID快3.33倍，并且相对提高了1.7%的mAP。这使得在MUVERA支持下，检索可以实现微秒级的延迟。我们发布了所有检查点、配置和评估脚本。该研究的局限性包括依赖于中等大小的数据集（≤50K文档）和翻译基准，这可能不能完全反映真实的土耳其语检索条件，大规模MUVERA评估仍需进一步开展。 

---
# ODE-ViT: Plug & Play Attention Layer from the Generalization of the ViT as an Ordinary Differential Equation 

**Title (ZH)**: ODE-ViT: 从ViT作为常微分方程的推广实现即插即用注意力层 

**Authors**: Carlos Boned Riera, David Romero Sanchez, Oriol Ramos Terrades  

**Link**: [PDF](https://arxiv.org/pdf/2511.16501)  

**Abstract**: In recent years, increasingly large models have achieved outstanding performance across CV tasks. However, these models demand substantial computational resources and storage, and their growing complexity limits our understanding of how they make decisions. Most of these architectures rely on the attention mechanism within Transformer-based designs. Building upon the connection between residual neural networks and ordinary differential equations (ODEs), we introduce ODE-ViT, a Vision Transformer reformulated as an ODE system that satisfies the conditions for well-posed and stable dynamics. Experiments on CIFAR-10 and CIFAR-100 demonstrate that ODE-ViT achieves stable, interpretable, and competitive performance with up to one order of magnitude fewer parameters, surpassing prior ODE-based Transformer approaches in classification tasks. We further propose a plug-and-play teacher-student framework in which a discrete ViT guides the continuous trajectory of ODE-ViT by treating the intermediate representations of the teacher as solutions of the ODE. This strategy improves performance by more than 10% compared to training a free ODE-ViT from scratch. 

**Abstract (ZH)**: 近年来，越来越大的模型在CV任务中取得了卓越的性能。然而，这些模型需要大量的计算资源和存储空间，其复杂的结构限制了我们对其决策过程的理解。大多数这些架构依赖于Transformer设计中的注意力机制。基于残差神经网络与常微分方程（ODEs）之间的联系，我们提出了ODE-ViT，这是一种以ODE系统形式重构的Vision Transformer，满足良好定义和稳定动力学的条件。实验表明，ODE-ViT在CIFAR-10和CIFAR-100上的表现稳定、可解释且具有竞争力，参数量减少了一个数量级，并且在分类任务上超越了之前的基于ODE的Transformer方法。我们进一步提出了一种即插即用的教师-学生框架，其中离散的ViT通过将教师的中间表示视为ODE的解来引导连续的ODE-ViT轨迹。这种方法相较于从头训练一个自由的ODE-ViT，性能提升超过10%。 

---
# Physics-Informed Machine Learning for Efficient Sim-to-Real Data Augmentation in Micro-Object Pose Estimation 

**Title (ZH)**: 基于物理的机器学习在微小物体姿态估计中的高效模拟到现实数据增强 

**Authors**: Zongcai Tan, Lan Wei, Dandan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16494)  

**Abstract**: Precise pose estimation of optical microrobots is essential for enabling high-precision object tracking and autonomous biological studies. However, current methods rely heavily on large, high-quality microscope image datasets, which are difficult and costly to acquire due to the complexity of microrobot fabrication and the labour-intensive labelling. Digital twin systems offer a promising path for sim-to-real data augmentation, yet existing techniques struggle to replicate complex optical microscopy phenomena, such as diffraction artifacts and depth-dependent this http URL work proposes a novel physics-informed deep generative learning framework that, for the first time, integrates wave optics-based physical rendering and depth alignment into a generative adversarial network (GAN), to synthesise high-fidelity microscope images for microrobot pose estimation efficiently. Our method improves the structural similarity index (SSIM) by 35.6% compared to purely AI-driven methods, while maintaining real-time rendering speeds (0.022 s/frame).The pose estimator (CNN backbone) trained on our synthetic data achieves 93.9%/91.9% (pitch/roll) accuracy, just 5.0%/5.4% (pitch/roll) below that of an estimator trained exclusively on real data. Furthermore, our framework generalises to unseen poses, enabling data augmentation and robust pose estimation for novel microrobot configurations without additional training data. 

**Abstract (ZH)**: 基于波光学的物理引导深度生成学习框架在微纳机器人姿态估计中的高效合成高保真显微镜图像 

---
# LLM4EO: Large Language Model for Evolutionary Optimization in Flexible Job Shop Scheduling 

**Title (ZH)**: LLM4EO：大型语言模型在柔性作业车间调度中的进化优化 

**Authors**: Rongjie Liao, Junhao Qiu, Xin Chen, Xiaoping Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.16485)  

**Abstract**: Customized static operator design has enabled widespread application of Evolutionary Algorithms (EAs), but their search performance is transient during iterations and prone to degradation. Dynamic operators aim to address this but typically rely on predefined designs and localized parameter control during the search process, lacking adaptive optimization throughout evolution. To overcome these limitations, this work leverages Large Language Models (LLMs) to perceive evolutionary dynamics and enable operator-level meta-evolution. The proposed framework, LLMs for Evolutionary Optimization (LLM4EO), comprises three components: knowledge-transfer-based operator design, evolution perception and analysis, and adaptive operator evolution. Firstly, initialization of operators is performed by transferring the strengths of classical operators via LLMs. Then, search preferences and potential limitations of operators are analyzed by integrating fitness performance and evolutionary features, accompanied by corresponding suggestions for improvement. Upon stagnation of population evolution, gene selection priorities of operators are dynamically optimized via improvement prompting strategies. This approach achieves co-evolution of populations and operators in the search, introducing a novel paradigm for enhancing the efficiency and adaptability of EAs. Finally, a series of validations on multiple benchmark datasets of the flexible job shop scheduling problem demonstrate that LLM4EO accelerates population evolution and outperforms both mainstream evolutionary programming and traditional EAs. 

**Abstract (ZH)**: 基于大型语言模型的进化优化（LLM4EO）算法框架 

---
# Large Language Model-Based Reward Design for Deep Reinforcement Learning-Driven Autonomous Cyber Defense 

**Title (ZH)**: 基于大型语言模型的奖励设计——面向深度强化学习驱动的自主网络安全防御 

**Authors**: Sayak Mukherjee, Samrat Chatterjee, Emilie Purvine, Ted Fujimoto, Tegan Emerson  

**Link**: [PDF](https://arxiv.org/pdf/2511.16483)  

**Abstract**: Designing rewards for autonomous cyber attack and defense learning agents in a complex, dynamic environment is a challenging task for subject matter experts. We propose a large language model (LLM)-based reward design approach to generate autonomous cyber defense policies in a deep reinforcement learning (DRL)-driven experimental simulation environment. Multiple attack and defense agent personas were crafted, reflecting heterogeneity in agent actions, to generate LLM-guided reward designs where the LLM was first provided with contextual cyber simulation environment information. These reward structures were then utilized within a DRL-driven attack-defense simulation environment to learn an ensemble of cyber defense policies. Our results suggest that LLM-guided reward designs can lead to effective defense strategies against diverse adversarial behaviors. 

**Abstract (ZH)**: 基于大规模语言模型的奖励设计方法：在复杂动态环境中利用深度强化学习驱动的实验模拟环境自动生成自主网络防御策略 

---
# Correlation-Aware Feature Attribution Based Explainable AI 

**Title (ZH)**: 基于相关性意识的特征归因可解释人工智能 

**Authors**: Poushali Sengupta, Yan Zhang, Frank Eliassen, Sabita Maharjan  

**Link**: [PDF](https://arxiv.org/pdf/2511.16482)  

**Abstract**: Explainable AI (XAI) is increasingly essential as modern models become more complex and high-stakes applications demand transparency, trust, and regulatory compliance. Existing global attribution methods often incur high computational costs, lack stability under correlated inputs, and fail to scale efficiently to large or heterogeneous datasets. We address these gaps with \emph{ExCIR} (Explainability through Correlation Impact Ratio), a correlation-aware attribution score equipped with a lightweight transfer protocol that reproduces full-model rankings using only a fraction of the data. ExCIR quantifies sign-aligned co-movement between features and model outputs after \emph{robust centering} (subtracting a robust location estimate, e.g., median or mid-mean, from features and outputs). We further introduce \textsc{BlockCIR}, a \emph{groupwise} extension of ExCIR that scores \emph{sets} of correlated features as a single unit. By aggregating the same signed-co-movement numerators and magnitudes over predefined or data-driven groups, \textsc{BlockCIR} mitigates double-counting in collinear clusters (e.g., synonyms or duplicated sensors) and yields smoother, more stable rankings when strong dependencies are present. Across diverse text, tabular, signal, and image datasets, ExCIR shows trustworthy agreement with established global baselines and the full model, delivers consistent top-$k$ rankings across settings, and reduces runtime via lightweight evaluation on a subset of rows. Overall, ExCIR provides \emph{computationally efficient}, \emph{consistent}, and \emph{scalable} explainability for real-world deployment. 

**Abstract (ZH)**: 可解释人工智能 (XAI) 随着现代模型变得越来越复杂，以及高风险应用对透明度、信任和监管合规性要求的不断增加而变得愈发重要。现有的全局归因方法通常计算成本高、在相关输入下缺乏稳定性，并且无法有效地扩展到大型或异质数据集。我们通过提出ExCIR（基于相关影响比的可解释性）来弥补这些差距，ExCIR是一种带有轻量级传输协议的相关性意识归因评分，仅使用数据的 fraction 即可复现全模型的排名。ExCIR在进行鲁棒中心化（从特征和输出中减去一个鲁棒位置估计，例如中位数或中位均值）之后，量化特征与模型输出的符号一致的共同变动。我们进一步引入BlockCIR，这是一种ExCIR的分组扩展，将相关特征集作为一个整体打分。通过在预定义或数据驱动的分组中聚合相同的带符号共同变动的分子和幅度，BlockCIR 在共线簇（例如同义词或重复传感器）中减轻了重复计数，并在强依赖性存在时提供了更平滑、更稳定的结果。在多种文本、表格、信号和图像数据集上，ExCIR 与现有的全局基准和全模型展示了可信赖的一致性，并在不同场景中提供了稳定且前k高的排名，同时通过在子集上进行轻量级评估来减少运行时间。总体而言，ExCIR 提供了计算高效、一致和可扩展的可解释性，适用于实际部署。 

---
# Anatomy of an Idiom: Tracing Non-Compositionality in Language Models 

**Title (ZH)**: 习语剖析：追踪语言模型中的非组合性 

**Authors**: Andrew Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2511.16467)  

**Abstract**: We investigate the processing of idiomatic expressions in transformer-based language models using a novel set of techniques for circuit discovery and analysis. First discovering circuits via a modified path patching algorithm, we find that idiom processing exhibits distinct computational patterns. We identify and investigate ``Idiom Heads,'' attention heads that frequently activate across different idioms, as well as enhanced attention between idiom tokens due to earlier processing, which we term ``augmented reception.'' We analyze these phenomena and the general features of the discovered circuits as mechanisms by which transformers balance computational efficiency and robustness. Finally, these findings provide insights into how transformers handle non-compositional language and suggest pathways for understanding the processing of more complex grammatical constructions. 

**Abstract (ZH)**: 基于电路发现与分析技术：探讨Transformer模型中惯用语的处理机制 

---
# VLA-Pruner: Temporal-Aware Dual-Level Visual Token Pruning for Efficient Vision-Language-Action Inference 

**Title (ZH)**: VLA-精简器：面向时间的双层视觉令牌精简以实现高效的视觉-语言-动作推理 

**Authors**: Ziyan Liu, Yeqiu Chen, Hongyi Cai, Tao Lin, Shuo Yang, Zheng Liu, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.16449)  

**Abstract**: Vision-Language-Action (VLA) models have shown great promise for embodied AI, yet the heavy computational cost of processing continuous visual streams severely limits their real-time deployment. Token pruning (keeping salient visual tokens and dropping redundant ones) has emerged as an effective approach for accelerating Vision-Language Models (VLMs), offering a solution for efficient VLA. However, these VLM-specific token pruning methods select tokens based solely on semantic salience metrics (e.g., prefill attention), while overlooking the VLA's intrinsic dual-system nature of high-level semantic understanding and low-level action execution. Consequently, these methods bias token retention toward semantic cues, discard critical information for action generation, and significantly degrade VLA performance. To bridge this gap, we propose VLA-Pruner, a versatile plug-and-play VLA-specific token prune method that aligns with the dual-system nature of VLA models and exploits the temporal continuity in robot manipulation. Specifically, VLA-Pruner adopts a dual-level importance criterion for visual token retention: vision-language prefill attention for semantic-level relevance and action decode attention, estimated via temporal smoothing, for action-level importance. Based on this criterion, VLA-Pruner proposes a novel dual-level token selection strategy that adaptively preserves a compact, informative set of visual tokens for both semantic understanding and action execution under given compute budget. Experiments show that VLA-Pruner achieves state-of-the-art performance across multiple VLA architectures and diverse robotic tasks. 

**Abstract (ZH)**: VLA-Pruner: 一种适应Vision-Language-Action模型双系统特性的插件式视觉标记裁剪方法 

---
# Generative Modeling of Clinical Time Series via Latent Stochastic Differential Equations 

**Title (ZH)**: 基于潜在随机微分方程的临床时间序列生成模型 

**Authors**: Muhammad Aslanimoghanloo, Ahmed ElGazzar, Marcel van Gerven  

**Link**: [PDF](https://arxiv.org/pdf/2511.16427)  

**Abstract**: Clinical time series data from electronic health records and medical registries offer unprecedented opportunities to understand patient trajectories and inform medical decision-making. However, leveraging such data presents significant challenges due to irregular sampling, complex latent physiology, and inherent uncertainties in both measurements and disease progression. To address these challenges, we propose a generative modeling framework based on latent neural stochastic differential equations (SDEs) that views clinical time series as discrete-time partial observations of an underlying controlled stochastic dynamical system. Our approach models latent dynamics via neural SDEs with modality-dependent emission models, while performing state estimation and parameter learning through variational inference. This formulation naturally handles irregularly sampled observations, learns complex non-linear interactions, and captures the stochasticity of disease progression and measurement noise within a unified scalable probabilistic framework. We validate the framework on two complementary tasks: (i) individual treatment effect estimation using a simulated pharmacokinetic-pharmacodynamic (PKPD) model of lung cancer, and (ii) probabilistic forecasting of physiological signals using real-world intensive care unit (ICU) data from 12,000 patients. Results show that our framework outperforms ordinary differential equation and long short-term memory baseline models in accuracy and uncertainty estimation. These results highlight its potential for enabling precise, uncertainty-aware predictions to support clinical decision-making. 

**Abstract (ZH)**: 电子健康记录和医疗注册机构中的临床时间序列数据提供了前所未有的机会来理解患者轨迹并指导医疗决策。然而，利用这些数据面临着巨大挑战，包括不规则采样、复杂的潜在生理学、以及在测量和疾病进展中存在的固有不确定性。为了解决这些挑战，我们提出了一种基于潜在神经随机微分方程（SDE）的生成建模框架，将临床时间序列视为潜在受控随机动力系统在离散时间下的部分观测。我们的方法通过具有模态依赖发射模型的神经SDE来建模潜在动力学，并通过变分推断进行状态估计和参数学习。这种表述自然处理不规则采样的观察，学习复杂的非线性交互作用，并在统一的可扩展概率框架中捕捉疾病进展和测量噪声的随机性。我们通过两个互补任务验证了该框架：一是使用模拟的肺癌药代药效（PKPD）模型估计个体治疗效果；二是使用12000名患者实际重症监护病房（ICU）数据进行生理信号的概率预测。结果显示，与普通的微分方程和长短期记忆基线模型相比，该框架在准确性和不确定性估计方面表现更优。这些结果突显了其在支持临床决策中实现精确、不确定性意识预测的潜力。 

---
# Collaborative Management for Chronic Diseases and Depression: A Double Heterogeneity-based Multi-Task Learning Method 

**Title (ZH)**: 基于双异质性多任务学习的慢性疾病和抑郁协作管理方法 

**Authors**: Yidong Chai, Haoxin Liu, Jiaheng Xie, Chaopeng Wang, Xiao Fang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16398)  

**Abstract**: Wearable sensor technologies and deep learning are transforming healthcare management. Yet, most health sensing studies focus narrowly on physical chronic diseases. This overlooks the critical need for joint assessment of comorbid physical chronic diseases and depression, which is essential for collaborative chronic care. We conceptualize multi-disease assessment, including both physical diseases and depression, as a multi-task learning (MTL) problem, where each disease assessment is modeled as a task. This joint formulation leverages inter-disease relationships to improve accuracy, but it also introduces the challenge of double heterogeneity: chronic diseases differ in their manifestation (disease heterogeneity), and patients with the same disease show varied patterns (patient heterogeneity). To address these issues, we first adopt existing techniques and propose a base method. Given the limitations of the base method, we further propose an Advanced Double Heterogeneity-based Multi-Task Learning (ADH-MTL) method that improves the base method through three innovations: (1) group-level modeling to support new patient predictions, (2) a decomposition strategy to reduce model complexity, and (3) a Bayesian network that explicitly captures dependencies while balancing similarities and differences across model components. Empirical evaluations on real-world wearable sensor data demonstrate that ADH-MTL significantly outperforms existing baselines, and each of its innovations is shown to be effective. This study contributes to health information systems by offering a computational solution for integrated physical and mental healthcare and provides design principles for advancing collaborative chronic disease management across the pre-treatment, treatment, and post-treatment phases. 

**Abstract (ZH)**: 可穿戴传感器技术和深度学习正在变革健康管理工作。然而，大多数健康感知研究集中在物理慢性疾病上。这忽视了同时评估共病物理慢性疾病和抑郁的重要需求，这对于协作性慢性病管理是必不可少的。我们将包括物理疾病和抑郁在内的多病种评估概念化为多任务学习（MTL）问题，其中每种疾病评估被视为一个任务。这种联合建模利用了不同疾病的相互关系以提高准确性，但也带来了双重异质性的挑战：慢性疾病在表现形式上不同（疾病异质性），而患有相同疾病患者则表现出不同的模式（患者异质性）。为了解决这些问题，我们首先采用了现有技术并提出了一种基方法。鉴于基方法的局限性，我们进一步提出了基于双重异质性的先进多任务学习（ADH-MTL）方法，通过三项创新改进了基方法：（1）分组级建模以支持新患者的预测；（2）分解策略以降低模型复杂性；（3）贝叶斯网络，它明确捕捉依赖关系并在模型组件之间平衡相似性和差异性。在真实世界可穿戴传感器数据上的实证评估表明，ADH-MTL 显著优于现有基方法，并且其每项创新都是有效的。这项研究通过提供整合身心健康的计算解决方案，丰富了健康信息系统，并为推进预治疗、治疗和后续治疗的协作慢性病管理提供了设计原则。 

---
# Robot Metacognition: Decision Making with Confidence for Tool Invention 

**Title (ZH)**: 机器人元认知：具有信心的工具发明决策 

**Authors**: Ajith Anil Meera, Poppy Collis, Polina Arbuzova, Abián Torres, Paul F Kinghorn, Ricardo Sanz, Pablo Lanillos  

**Link**: [PDF](https://arxiv.org/pdf/2511.16390)  

**Abstract**: Robots today often miss a key ingredient of truly intelligent behavior: the ability to reflect on their own cognitive processes and decisions. In humans, this self-monitoring or metacognition is crucial for learning, decision making and problem solving. For instance, they can evaluate how confident they are in performing a task, thus regulating their own behavior and allocating proper resources. Taking inspiration from neuroscience, we propose a robot metacognition architecture centered on confidence (a second-order judgment on decisions) and we demonstrate it on the use case of autonomous tool invention. We propose the use of confidence as a metacognitive measure within the robot decision making scheme. Confidence-informed robots can evaluate the reliability of their decisions, improving their robustness during real-world physical deployment. This form of robotic metacognition emphasizes embodied action monitoring as a means to achieve better informed decisions. We also highlight potential applications and research directions for robot metacognition. 

**Abstract (ZH)**: 当前的机器人常常缺乏真正智能行为的关键成分：自我监控或元认知的能力。在人类中，这种自我监控或元认知对于学习、决策和问题解决至关重要。例如，他们可以评估自己完成任务的信心，从而调节自己的行为并分配适当的资源。受到神经科学的启发，我们提出了一种以信心（决策的二阶判断）为中心的机器人元认知架构，并在自主工具发明的应用场景中进行了演示。我们建议在机器人的决策机制中使用信心作为元认知指标。基于信心的机器人可以评估其决策的可靠性，提高其实体世界部署过程中的稳健性。这种形式的机器人元认知强调了通过体验动作监控以实现更明智决策的重要性。我们还强调了机器人元认知的潜在应用和研究方向。 

---
# Are Foundation Models Useful for Bankruptcy Prediction? 

**Title (ZH)**: 金融科技模型在破产预测中有效吗？ 

**Authors**: Marcin Kostrzewa, Oleksii Furman, Roman Furman, Sebastian Tomczak, Maciej Zięba  

**Link**: [PDF](https://arxiv.org/pdf/2511.16375)  

**Abstract**: Foundation models have shown promise across various financial applications, yet their effectiveness for corporate bankruptcy prediction remains systematically unevaluated against established methods. We study bankruptcy forecasting using Llama-3.3-70B-Instruct and TabPFN, evaluated on large, highly imbalanced datasets of over one million company records from the Visegrád Group. We provide the first systematic comparison of foundation models against classical machine learning baselines for this task. Our results show that models such as XGBoost and CatBoost consistently outperform foundation models across all prediction horizons. LLM-based approaches suffer from unreliable probability estimates, undermining their use in risk-sensitive financial settings. TabPFN, while competitive with simpler baselines, requires substantial computational resources with costs not justified by performance gains. These findings suggest that, despite their generality, current foundation models remain less effective than specialized methods for bankruptcy forecasting. 

**Abstract (ZH)**: 基础模型在各类金融应用中展现了潜力，但其在公司破产预测方面的有效性仍系统性地未与传统方法进行评估。我们使用Llama-3.3-70B-Instruct和TabPFN对Visegrád集团超过一百万公司记录的大规模、高度不平衡数据集进行破产预测研究。我们提供了基础模型与传统机器学习基线方法在这项任务上的首次系统性比较。结果显示，在所有预测时段，XGBoost和CatBoost等模型的一贯表现优于基础模型。基于大语言模型的方法因不可靠的概率估计，在风险敏感的金融环境中不具优势。虽然TabPFN在与简单基线竞争中有竞争力，但其所需的大量计算资源并未因性能提升而得到合理化。这些发现表明，尽管基础模型具有通用性，但当前的方法在破产预测中仍不如专门方法有效。 

---
# Learning from Sufficient Rationales: Analysing the Relationship Between Explanation Faithfulness and Token-level Regularisation Strategies 

**Title (ZH)**: 基于充分解释的学习：分析解释忠实度与token级正则化策略之间的关系 

**Authors**: Jonathan Kamp, Lisa Beinborn, Antske Fokkens  

**Link**: [PDF](https://arxiv.org/pdf/2511.16353)  

**Abstract**: Human explanations of natural language, rationales, form a tool to assess whether models learn a label for the right reasons or rely on dataset-specific shortcuts. Sufficiency is a common metric for estimating the informativeness of rationales, but it provides limited insight into the effects of rationale information on model performance. We address this limitation by relating sufficiency to two modelling paradigms: the ability of models to identify which tokens are part of the rationale (through token classification) and the ability of improving model performance by incorporating rationales in the input (through attention regularisation). We find that highly informative rationales are not likely to help classify the instance correctly. Sufficiency conversely captures the classification impact of the non-rationalised context, which interferes with rationale information in the same input. We also find that incorporating rationale information in model inputs can boost cross-domain classification, but results are inconsistent per task and model type. Finally, sufficiency and token classification appear to be unrelated. These results exemplify the complexity of rationales, showing that metrics capable of systematically capturing this type of information merit further investigation. 

**Abstract (ZH)**: 自然语言的机器解释为其提供了一种工具，以评估模型是否出于正确的原因学习标签或依赖于数据集特定的捷径。信息量充足的度量常用于估计解释信息的重要性，但其对解释信息如何影响模型性能提供有限的洞察。我们通过将其与两种建模范式相关联来弥补这一局限：模型识别哪些词属于解释信息的能力（通过词分类实现）和通过注意力规约改进模型性能的能力（通过在输入中包含解释信息实现）。我们发现，高度信息丰富的解释信息不太可能帮助正确分类实例。相反，信息量充足的度量捕捉了未解释上下文对解释信息的分类影响，这与同一输入中的解释信息相互干扰。此外，我们发现将解释信息包含在模型输入中可以增强跨域分类，但结果在不同的任务和模型类型上不一致。最后，信息量充足的度量与词分类似乎无关。这些结果突出显示了解释信息的复杂性，表明能够系统捕捉这种类型信息的度量需要进一步研究。 

---
# SDA: Steering-Driven Distribution Alignment for Open LLMs without Fine-Tuning 

**Title (ZH)**: SDA: 驾驶式分布对齐以实现无需微调的开放大型语言模型 

**Authors**: Wei Xia, Zhi-Hong Deng  

**Link**: [PDF](https://arxiv.org/pdf/2511.16324)  

**Abstract**: With the rapid advancement of large language models (LLMs), their deployment in real-world applications has become increasingly widespread. LLMs are expected to deliver robust performance across diverse tasks, user preferences, and practical scenarios. However, as demands grow, ensuring that LLMs produce responses aligned with human intent remains a foundational challenge. In particular, aligning model behavior effectively and efficiently during inference, without costly retraining or extensive supervision, is both a critical requirement and a non-trivial technical endeavor. To address the challenge, we propose SDA (Steering-Driven Distribution Alignment), a training-free and model-agnostic alignment framework designed for open-source LLMs. SDA dynamically redistributes model output probabilities based on user-defined alignment instructions, enhancing alignment between model behavior and human intents without fine-tuning. The method is lightweight, resource-efficient, and compatible with a wide range of open-source LLMs. It can function independently during inference or be integrated with training-based alignment strategies. Moreover, SDA supports personalized preference alignment, enabling flexible control over the model response behavior. Empirical results demonstrate that SDA consistently improves alignment performance across 8 open-source LLMs with varying scales and diverse origins, evaluated on three key alignment dimensions, helpfulness, harmlessness, and honesty (3H). Specifically, SDA achieves average gains of 64.4% in helpfulness, 30% in honesty and 11.5% in harmlessness across the tested models, indicating its effectiveness and generalization across diverse models and application scenarios. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的快速进步，其在实际应用中的部署日益广泛。LLMs 希望能够在多种任务、用户偏好和实际场景中提供稳健的性能。然而，随着需求的增长，确保 LLMs 产生的响应与人类意图保持一致仍然是一个基础挑战。特别是在推理过程中有效地和高效地对齐模型行为，而无需高昂的重新训练或大量的监督，这既是关键要求也是非平凡的技术任务。为应对这一挑战，我们提出了一种名为 SDA（Steering-Driven Distribution Alignment）的训练免费且模型无关的对齐框架，旨在为开源 LLMs 提供支持。SDA 动态重分布模型输出概率，基于用户定义的对齐指令增强模型行为与人类意图之间的对齐，而不进行微调。该方法轻量级、资源高效，并与多种开源 LLMs 兼容。它可以在推理过程中独立运行，也可以与基于训练的对齐策略结合使用。此外，SDA 支持个性化偏好对齐，允许灵活地控制模型响应行为。实证结果表明，SDA 在三个关键对齐维度（帮助性、无害性和诚实性）上持续改进了 8 种不同规模和起源的开源 LLMs 的对齐性能。具体而言，SDA 在测试模型中的帮助性提高了 64.4%，诚实性提高了 30%，无害性提高了 11.5%，表明其在不同模型和应用场景中的有效性和泛化能力。 

---
# "To Survive, I Must Defect": Jailbreaking LLMs via the Game-Theory Scenarios 

**Title (ZH)**: “为了生存，我必须背叛”：通过博弈论场景破解LLMs 

**Authors**: Zhen Sun, Zongmin Zhang, Deqi Liang, Han Sun, Yule Liu, Yun Shen, Xiangshan Gao, Yilong Yang, Shuai Liu, Yutao Yue, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2511.16278)  

**Abstract**: As LLMs become more common, non-expert users can pose risks, prompting extensive research into jailbreak attacks. However, most existing black-box jailbreak attacks rely on hand-crafted heuristics or narrow search spaces, which limit scalability. Compared with prior attacks, we propose Game-Theory Attack (GTA), an scalable black-box jailbreak framework. Concretely, we formalize the attacker's interaction against safety-aligned LLMs as a finite-horizon, early-stoppable sequential stochastic game, and reparameterize the LLM's randomized outputs via quantal response. Building on this, we introduce a behavioral conjecture "template-over-safety flip": by reshaping the LLM's effective objective through game-theoretic scenarios, the originally safety preference may become maximizing scenario payoffs within the template, which weakens safety constraints in specific contexts. We validate this mechanism with classical game such as the disclosure variant of the Prisoner's Dilemma, and we further introduce an Attacker Agent that adaptively escalates pressure to increase the ASR. Experiments across multiple protocols and datasets show that GTA achieves over 95% ASR on LLMs such as Deepseek-R1, while maintaining efficiency. Ablations over components, decoding, multilingual settings, and the Agent's core model confirm effectiveness and generalization. Moreover, scenario scaling studies further establish scalability. GTA also attains high ASR on other game-theoretic scenarios, and one-shot LLM-generated variants that keep the model mechanism fixed while varying background achieve comparable ASR. Paired with a Harmful-Words Detection Agent that performs word-level insertions, GTA maintains high ASR while lowering detection under prompt-guard models. Beyond benchmarks, GTA jailbreaks real-world LLM applications and reports a longitudinal safety monitoring of popular HuggingFace LLMs. 

**Abstract (ZH)**: 基于博弈论的可扩展黑盒 Jailbreak 攻击框架 

---
# SeSE: A Structural Information-Guided Uncertainty Quantification Framework for Hallucination Detection in LLMs 

**Title (ZH)**: SeSE：一种结构信息引导的不确定性量化框架用于LLMs的幻觉检测 

**Authors**: Xingtao Zhao, Hao Peng, Dingli Su, Xianghua Zeng, Chunyang Liu, Jinzhi Liao, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.16275)  

**Abstract**: Reliable uncertainty quantification (UQ) is essential for deploying large language models (LLMs) in safety-critical scenarios, as it enables them to abstain from responding when uncertain, thereby avoiding hallucinating falsehoods. However, state-of-the-art UQ methods primarily rely on semantic probability distributions or pairwise distances, overlooking latent semantic structural information that could enable more precise uncertainty estimates. This paper presents Semantic Structural Entropy (SeSE), a principled UQ framework that quantifies the inherent semantic uncertainty of LLMs from a structural information perspective for hallucination detection. Specifically, to effectively model semantic spaces, we first develop an adaptively sparsified directed semantic graph construction algorithm that captures directional semantic dependencies while automatically pruning unnecessary connections that introduce negative interference. We then exploit latent semantic structural information through hierarchical abstraction: SeSE is defined as the structural entropy of the optimal semantic encoding tree, formalizing intrinsic uncertainty within semantic spaces after optimal compression. A higher SeSE value corresponds to greater uncertainty, indicating that LLMs are highly likely to generate hallucinations. In addition, to enhance fine-grained UQ in long-form generation -- where existing methods often rely on heuristic sample-and-count techniques -- we extend SeSE to quantify the uncertainty of individual claims by modeling their random semantic interactions, providing theoretically explicable hallucination detection. Extensive experiments across 29 model-dataset combinations show that SeSE significantly outperforms advanced UQ baselines, including strong supervised methods and the recently proposed KLE. 

**Abstract (ZH)**: 可靠的不确定性量化 (UQ) 是在关键安全场景中部署大型语言模型 (LLMs) 的基础，因为它使它们能够在不确定时避免响应，从而避免产生 falsehood。然而，最先进的 UQ 方法主要依赖于语义概率分布或成对距离，忽视了潜在的语义结构信息，这些信息能够提供更精确的不确定性估计。本文提出了语义结构熵 (SeSE)，这是一种基于结构信息视角量化 LLMs 内在语义不确定性并用于幻觉检测的原则性 UQ 框架。具体来说，为了有效建模语义空间，我们首先开发了一种自适应稀疏化有向语义图构建算法，该算法捕获了方向性语义依赖性，并自动修剪可能引入负干扰的不必要的连接。然后，通过层次抽象利用潜在的语义结构信息：SeSE 定义为最优语义编码树的结构熵，形式化语义空间在最优压缩后的内在不确定性。SeSE 值越高表示不确定性越大，表明 LLMs 高可能性生成幻觉。此外，为了增强长段落生成中的细粒度不确定性量化——现有方法通常依赖于启发式的样本-计数技术——我们将 SeSE 扩展到通过建模个体断言的随机语义交互来量化其不确定性，从而提供理论可解释的幻觉检测。跨 29 种模型-数据集组合的广泛实验表明，SeSE 显著优于先进的 UQ 基线，包括强大的监督方法和最近提出的 KLE。 

---
# Q-MLLM: Vector Quantization for Robust Multimodal Large Language Model Security 

**Title (ZH)**: Q-MLLM：多模态大型语言模型的鲁棒向量量化安全性 

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.16229)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in cross-modal understanding, but remain vulnerable to adversarial attacks through visual inputs despite robust textual safety mechanisms. These vulnerabilities arise from two core weaknesses: the continuous nature of visual representations, which allows for gradient-based attacks, and the inadequate transfer of text-based safety mechanisms to visual content. We introduce Q-MLLM, a novel architecture that integrates two-level vector quantization to create a discrete bottleneck against adversarial attacks while preserving multimodal reasoning capabilities. By discretizing visual representations at both pixel-patch and semantic levels, Q-MLLM blocks attack pathways and bridges the cross-modal safety alignment gap. Our two-stage training methodology ensures robust learning while maintaining model utility. Experiments demonstrate that Q-MLLM achieves significantly better defense success rate against both jailbreak attacks and toxic image attacks than existing approaches. Notably, Q-MLLM achieves perfect defense success rate (100\%) against jailbreak attacks except in one arguable case, while maintaining competitive performance on multiple utility benchmarks with minimal inference overhead. This work establishes vector quantization as an effective defense mechanism for secure multimodal AI systems without requiring expensive safety-specific fine-tuning or detection overhead. Code is available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（Q-MLLM）通过集成两级向量量化增强对抗攻击防御能力的同时保持多模态推理能力 

---
# When Alignment Fails: Multimodal Adversarial Attacks on Vision-Language-Action Models 

**Title (ZH)**: 当对齐失效：针对视觉-语言-动作模型的多模态对抗攻击 

**Authors**: Yuping Yan, Yuhan Xie, Yinxin Zhang, Lingjuan Lyu, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2511.16203)  

**Abstract**: Vision-Language-Action models (VLAs) have recently demonstrated remarkable progress in embodied environments, enabling robots to perceive, reason, and act through unified multimodal understanding. Despite their impressive capabilities, the adversarial robustness of these systems remains largely unexplored, especially under realistic multimodal and black-box conditions. Existing studies mainly focus on single-modality perturbations and overlook the cross-modal misalignment that fundamentally affects embodied reasoning and decision-making. In this paper, we introduce VLA-Fool, a comprehensive study of multimodal adversarial robustness in embodied VLA models under both white-box and black-box settings. VLA-Fool unifies three levels of multimodal adversarial attacks: (1) textual perturbations through gradient-based and prompt-based manipulations, (2) visual perturbations via patch and noise distortions, and (3) cross-modal misalignment attacks that intentionally disrupt the semantic correspondence between perception and instruction. We further incorporate a VLA-aware semantic space into linguistic prompts, developing the first automatically crafted and semantically guided prompting framework. Experiments on the LIBERO benchmark using a fine-tuned OpenVLA model reveal that even minor multimodal perturbations can cause significant behavioral deviations, demonstrating the fragility of embodied multimodal alignment. 

**Abstract (ZH)**: Vision-Language-Action模型（VLAs）在体感环境中的多模态对抗鲁棒性研究：从白盒到黑盒设置 

---
# Fast LLM Post-training via Decoupled and Best-of-N Speculation 

**Title (ZH)**: 基于解耦和最优批次推测的快速LLM后训练 

**Authors**: Rongxin Cheng, Kai Zhou, Xingda Wei, Siyuan Liu, Mingcong Han, Mingjing Ai, Yeju Zhou, Baoquan Zhong, Wencong Xiao, Xin Liu, Rong Chen, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.16193)  

**Abstract**: Rollout dominates the training time in large language model (LLM) post-training, where the trained model is used to generate tokens given a batch of prompts. SpecActor achieves fast rollout with speculative decoding that deploys a fast path (e.g., a smaller model) to accelerate the unparallelizable generation, while the correctness is guaranteed by fast parallel verification of the outputs with the original model. SpecActor addresses two foundational challenges in speculative rollout by (1) a \emph{dynamic decoupled speculation} execution method that maximizes the GPU computational efficiency to realize speedup for large-batch execution -- a configuration common in training but unfriendly to speculative execution and (2) a \emph{dynamic Best-of-N speculation} method that selects and combines different drafting methods according to the rollout progress. It substantially improves the speculation accuracy even when the best drafting method is unknown a priori, meanwhile without requiring adding extra computation resources. {\sys} is {1.3--1.7}\,$\times$ faster than common post-training baselines, and is {1.3--1.5}\,$\times$ faster compared to naively adopting speculative decoding for rollout. 

**Abstract (ZH)**: Rollout在大型语言模型后训练中主导了训练时间，SpecActor通过猜测解码实现了快速rollout，同时通过原模型快速并行验证确保输出正确性，通过动态解耦猜测执行方法和动态最佳猜测方法解决了猜测rollout的两大基础挑战。.SpecActor比常见后训练基线快1.3-1.7倍，比简单采用猜测解码的rollout快1.3-1.5倍。 

---
# Mantis: A Versatile Vision-Language-Action Model with Disentangled Visual Foresight 

**Title (ZH)**: 螳螂：一种解耦视觉前瞻的多功能视觉-语言-行动模型 

**Authors**: Yi Yang, Xueqi Li, Yiyang Chen, Jin Song, Yihan Wang, Zipeng Xiao, Jiadi Su, You Qiaoben, Pengfei Liu, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2511.16175)  

**Abstract**: Recent advances in Vision-Language-Action (VLA) models demonstrate that visual signals can effectively complement sparse action supervisions. However, letting VLA directly predict high-dimensional visual states can distribute model capacity and incur prohibitive training cost, while compressing visual states into more compact supervisory signals inevitably incurs information bottlenecks. Moreover, existing methods often suffer from poor comprehension and reasoning capabilities due to the neglect of language supervision. This paper introduces Mantis, a novel framework featuring a Disentangled Visual Foresight (DVF) to tackle these issues. Specifically, Mantis decouples visual foresight prediction from the backbone with the combination of meta queries and a diffusion Transformer (DiT) head. With the current visual state provided to the DiT via a residual connection, a simple next-state prediction objective enables the meta queries to automatically capture the latent actions that delineate the visual trajectory, and hence boost the learning of explicit actions. The disentanglement reduces the burden of the VLA backbone, enabling it to maintain comprehension and reasoning capabilities through language supervision. Empirically, pretrained on human manipulation videos, robot demonstrations, and image-text pairs, Mantis achieves a 96.7% success rate on LIBERO benchmark after fine-tuning, surpassing powerful baselines while exhibiting high convergence speed. Real-world evaluations show that Mantis outperforms $\pi_{0.5}$, a leading open-source VLA model, particularly in instruction-following capability, generalization to unseen instructions, and reasoning ability. Code and weights are released to support the open-source community. 

**Abstract (ZH)**: Recent Advances in Vision-Language-Action (VLA) Models Require Disentangled Visual Foresight for Effective Action Learning 

---
# TS-PEFT: Token-Selective Parameter-Efficient Fine-Tuning with Learnable Threshold Gating 

**Title (ZH)**: TS-PEFT: Token-选择性参数-efficient微调带可学习阈值门控 

**Authors**: Dabiao Ma, Ziming Dai, Zhimin Xin, Shu Wang, Ye Wang, Haojun Fei  

**Link**: [PDF](https://arxiv.org/pdf/2511.16147)  

**Abstract**: In the field of large models (LMs) for natural language processing (NLP) and computer vision (CV), Parameter-Efficient Fine-Tuning (PEFT) has emerged as a resource-efficient method that modifies a limited number of parameters while keeping the pretrained weights fixed. This paper investigates the traditional PEFT approach, which applies modifications to all position indices, and questions its necessity. We introduce a new paradigm called Token-Selective PEFT (TS-PEFT), in which a function S selectively applies PEFT modifications to a subset of position indices, potentially enhancing performance on downstream tasks. Our experimental results reveal that the indiscriminate application of PEFT to all indices is not only superfluous, but may also be counterproductive. This study offers a fresh perspective on PEFT, advocating for a more targeted approach to modifications and providing a framework for future research to optimize the fine-tuning process for large models. 

**Abstract (ZH)**: 在自然语言处理（NLP）和计算机视觉（CV）领域的大模型（LMs）中，参数高效微调（PEFT）作为一种资源高效的方法逐渐兴起，它仅修改少量参数而保持预训练权重不变。本文探讨了传统的PEFT方法，该方法对所有位置索引都进行修改，并质疑其必要性。我们提出了一种新的范式，称为token选择性PEFT（TS-PEFT），在这种方法中，一个选择函数S仅对位置索引的一部分进行PEFT修改，有可能在下游任务上提高性能。我们的实验结果表明，对所有索引进行不分青红皂白的PEFT修改不仅是多余的，甚至可能是有害的。这项研究为PEFT提供了一个新的视角，提倡更具针对性的修改方法，并为未来优化大模型微调过程的研究提供了框架。 

---
# Labels Matter More Than Models: Quantifying the Benefit of Supervised Time Series Anomaly Detection 

**Title (ZH)**: 标签比模型更重要：量化监督时间序列异常检测的收益 

**Authors**: Zhijie Zhong, Zhiwen Yu, Kaixiang Yang, C. L. Philip Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.16145)  

**Abstract**: Time series anomaly detection (TSAD) is a critical data mining task often constrained by label scarcity. Consequently, current research predominantly focuses on Unsupervised Time-series Anomaly Detection (UTAD), relying on complex architectures to model normal data distributions. However, this approach often overlooks the significant performance gains available from limited anomaly labels achievable in practical scenarios. This paper challenges the premise that architectural complexity is the optimal path for TSAD. We conduct the first methodical comparison between supervised and unsupervised paradigms and introduce STAND, a streamlined supervised baseline. Extensive experiments on five public datasets demonstrate that: (1) Labels matter more than models: under a limited labeling budget, simple supervised models significantly outperform complex state-of-the-art unsupervised methods; (2) Supervision yields higher returns: the performance gain from minimal supervision far exceeds that from architectural innovations; and (3) Practicality: STAND exhibits superior prediction consistency and anomaly localization compared to unsupervised counterparts. These findings advocate for a data-centric shift in TSAD research, emphasizing label utilization over purely algorithmic complexity. The code is publicly available at this https URL. 

**Abstract (ZH)**: 时间序列异常检测（TSAD）是数据挖掘中的一个关键任务，通常受到标签稀缺的限制。因此，当前研究主要集中在无监督时间序列异常检测（UTAD）上，依赖复杂的架构来建模正常数据分布。然而，这种方法往往忽略了在实际场景中从有限的异常标签中获得的巨大性能提升。本文挑战了TSAD中架构复杂性是最优路径的前提。我们首次系统性地比较了监督和无监督范式，并引入了STAND，这是一种简化的监督基线。在五个公开数据集上的大量实验表明：（1）标签比模型更重要：在有限的标注预算下，简单的监督模型显著优于复杂的最新无监督方法；（2）监督回报更高：最小监督所带来的性能提升远超架构创新；（3）实用性：与无监督方法相比，STAND在预测一致性及异常定位方面表现出更优异的效果。这些发现呼吁TSAD研究向以数据为中心的方向转变，强调标签利用而非纯粹的算法复杂性。相关代码已公开，地址为：this https URL。 

---
# CoSP: Reconfigurable Multi-State Metamaterial Inverse Design via Contrastive Pretrained Large Language Model 

**Title (ZH)**: CoSP: 基于对比预训练大语言模型的可配置多态超材料逆设计 

**Authors**: Shujie Yang, Xuzhe Zhao, Yuqi Zhang, Yansong Tang, Kaichen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2511.16135)  

**Abstract**: Metamaterials, known for their ability to manipulate light at subwavelength scales, face significant design challenges due to their complex and sophisticated structures. Consequently, deep learning has emerged as a powerful tool to streamline their design process. Reconfigurable multi-state metamaterials (RMMs) with adjustable parameters can switch their optical characteristics between different states upon external stimulation, leading to numerous applications. However, existing deep learning-based inverse design methods fall short in considering reconfigurability with multi-state switching. To address this challenge, we propose CoSP, an intelligent inverse design method based on contrastive pretrained large language model (LLM). By performing contrastive pretraining on multi-state spectrum, a well-trained spectrum encoder capable of understanding the spectrum is obtained, and it subsequently interacts with a pretrained LLM. This approach allows the model to preserve its linguistic capabilities while also comprehending Maxwell's Equations, enabling it to describe material structures with target optical properties in natural language. Our experiments demonstrate that CoSP can design corresponding thin-film metamaterial structures for arbitrary multi-state, multi-band optical responses, showing great potentials in the intelligent design of RMMs for versatile applications. 

**Abstract (ZH)**: 基于对比预训练大语言模型的智能逆设计方法CoSP：面向可重构多状态 metamaterials 的智能设计 

---
# AskDB: An LLM Agent for Natural Language Interaction with Relational Databases 

**Title (ZH)**: AskDB：与关系型数据库自然语言交互的LLM代理 

**Authors**: Xuan-Quang Phan, Tan-Ha Mai, Thai-Duy Dinh, Minh-Thuan Nguyen, Lam-Son Lê  

**Link**: [PDF](https://arxiv.org/pdf/2511.16131)  

**Abstract**: Interacting with relational databases remains challenging for users across different expertise levels, particularly when composing complex analytical queries or performing administrative tasks. Existing systems typically address either natural language querying or narrow aspects of database administration, lacking a unified and intelligent interface for general-purpose database interaction. We introduce AskDB, a large language model powered agent designed to bridge this gap by supporting both data analysis and administrative operations over SQL databases through natural language. Built on Gemini 2, AskDB integrates two key innovations: a dynamic schema-aware prompting mechanism that effectively incorporates database metadata, and a task decomposition framework that enables the agent to plan and execute multi-step actions. These capabilities allow AskDB to autonomously debug derived SQL, retrieve contextual information via real-time web search, and adaptively refine its responses. We evaluate AskDB on a widely used Text-to-SQL benchmark and a curated set of DBA tasks, demonstrating strong performance in both analytical and administrative scenarios. Our results highlight the potential of AskDB as a unified and intelligent agent for relational database systems, offering an intuitive and accessible experience for end users. 

**Abstract (ZH)**: 使用大规模语言模型的AskDBagents通过自然语言支持SQL数据库的数据分析和管理操作 

---
# ELPO: Ensemble Learning Based Prompt Optimization for Large Language Models 

**Title (ZH)**: ELPO：基于集成学习的(prompt优化)大规模语言模型提示优化 

**Authors**: Qing Zhang, Bing Xu, Xudong Zhang, Yifan Shi, Yang Li, Chen Zhang, Yik Chung Wu, Ngai Wong, Yijie Chen, Hong Dai, Xiansen Chen, Mian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16122)  

**Abstract**: The remarkable performance of Large Language Models (LLMs) highly relies on crafted prompts. However, manual prompt engineering is a laborious process, creating a core bottleneck for practical application of LLMs. This phenomenon has led to the emergence of a new research area known as Automatic Prompt Optimization (APO), which develops rapidly in recent years. Existing APO methods such as those based on evolutionary algorithms or trial-and-error approaches realize an efficient and accurate prompt optimization to some extent. However, those researches focus on a single model or algorithm for the generation strategy and optimization process, which limits their performance when handling complex tasks. To address this, we propose a novel framework called Ensemble Learning based Prompt Optimization (ELPO) to achieve more accurate and robust results. Motivated by the idea of ensemble learning, ELPO conducts voting mechanism and introduces shared generation strategies along with different search methods for searching superior prompts. Moreover, ELPO creatively presents more efficient algorithms for the prompt generation and search process. Experimental results demonstrate that ELPO outperforms state-of-the-art prompt optimization methods across different tasks, e.g., improving F1 score by 7.6 on ArSarcasm dataset. 

**Abstract (ZH)**: 大型语言模型的出色表现高度依赖于精心设计的提示。然而，手动提示工程是一个繁琐的过程，成为实际应用大型语言模型的核心瓶颈。这一现象导致了自动提示优化（APO）这一新研究领域的出现，该领域近年来快速发展。现有的APO方法，如基于进化算法或试错方法的研究，在一定程度上实现了高效和准确的提示优化。然而，这些研究集中在单一模型或算法的生成策略和优化过程上，当处理复杂任务时限制了它们的性能。为了解决这个问题，我们提出了一种新的框架，称为基于集成学习的提示优化（ELPO），以实现更准确和鲁棒的结果。受集成学习思想的启发，ELPO进行了投票机制，引入了共享的生成策略以及不同的搜索方法来寻找最优提示。此外，ELPO创造性地提出了更高效的提示生成和搜索算法。实验结果表明，ELPO在不同任务中均优于最先进的提示优化方法，例如，在ArSarcasm数据集上将F1分数提高了7.6%。 

---
# T2T-VICL: Unlocking the Boundaries of Cross-Task Visual In-Context Learning via Implicit Text-Driven VLMs 

**Title (ZH)**: T2T-VICL: 通过隐式文本驱动的VLM解锁跨任务视觉在_CONTEXT学习的边界 

**Authors**: Shao-Jun Xia, Huixin Zhang, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2511.16107)  

**Abstract**: In large language models (LLM), in-context learning (ICL) refers to performing new tasks by conditioning on small demonstrations provided in the input context. Recent advances in visual in-context learning (VICL) demonstrate promising capabilities for solving downstream tasks by unified vision-language models (VLMs). When the visual prompt and the target images originate from different visual tasks, can VLMs still enable VICL? In the paper, we propose a fully collaborative pipeline, i.e. T2T-VICL, for VLMs to investigate the potential of cross-task VICL. Fundamentally, we design a mechanism to generate and select text prompts that best implicitly describe the differences between two distinct low-level vision tasks, and construct the first cross-task VICL dataset. Building upon this, we propose a novel inference framework that combines perceptual score-based reasoning with traditional evaluation metrics to perform cross-task VICL. Our approach achieves top-tier results across nine cross-task scenarios and second-tier performance in ten additional scenarios, unlocking the boundaries of cross-task VICL within VLMs. 

**Abstract (ZH)**: 大规模语言模型中跨任务视觉上下文学习的研究 

---
# Mitigating Estimation Bias with Representation Learning in TD Error-Driven Regularization 

**Title (ZH)**: 用表示学习减轻TD误差驱动正则化中的估计偏见 

**Authors**: Haohui Chen, Zhiyong Chen, Aoxiang Liu, Wentuo Fang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16090)  

**Abstract**: Deterministic policy gradient algorithms for continuous control suffer from value estimation biases that degrade performance. While double critics reduce such biases, the exploration potential of double actors remains underexplored. Building on temporal-difference error-driven regularization (TDDR), a double actor-critic framework, this work introduces enhanced methods to achieve flexible bias control and stronger representation learning. We propose three convex combination strategies, symmetric and asymmetric, that balance pessimistic estimates to mitigate overestimation and optimistic exploration via double actors to alleviate underestimation. A single hyperparameter governs this mechanism, enabling tunable control across the bias spectrum. To further improve performance, we integrate augmented state and action representations into the actor and critic networks. Extensive experiments show that our approach consistently outperforms benchmarks, demonstrating the value of tunable bias and revealing that both overestimation and underestimation can be exploited differently depending on the environment. 

**Abstract (ZH)**: 确定性策略梯度算法在连续控制中受价值估计偏差影响，导致性能下降。虽然双重批评者可以减少此类偏差，但双重行动者探索潜力的研究尚不充分。基于时差误差驱动正则化（TDDR），一个双重行动者-批评者框架，本文引入增强方法以实现灵活的偏差控制和更强的表示学习。我们提出三种凸组合策略，对称和非对称，平衡悲观估计以缓解过度估计，并通过双重行动者进行乐观探索以减轻低估。该机制由单一超参数控制，可在偏差谱上实现可调控制。为进一步提高性能，我们将扩展状态和动作表示集成到行动者和批评者网络中。广泛实验表明，我们的方法在基准算法上表现更优，证明了可调偏差的价值，并揭示了在不同环境中过度估计和低估可以以不同方式利用。 

---
# Future-Back Threat Modeling: A Foresight-Driven Security Framework 

**Title (ZH)**: 未来导向威胁建模：一种前瞻性驱动的安全框架 

**Authors**: Vu Van Than  

**Link**: [PDF](https://arxiv.org/pdf/2511.16088)  

**Abstract**: Traditional threat modeling remains reactive-focused on known TTPs and past incident data, while threat prediction and forecasting frameworks are often disconnected from operational or architectural artifacts. This creates a fundamental weakness: the most serious cyber threats often do not arise from what is known, but from what is assumed, overlooked, or not yet conceived, and frequently originate from the future, such as artificial intelligence, information warfare, and supply chain attacks, where adversaries continuously develop new exploits that can bypass defenses built on current knowledge. To address this mental gap, this paper introduces the theory and methodology of Future-Back Threat Modeling (FBTM). This predictive approach begins with envisioned future threat states and works backward to identify assumptions, gaps, blind spots, and vulnerabilities in the current defense architecture, providing a clearer and more accurate view of impending threats so that we can anticipate their emergence and shape the future we want through actions taken now. The proposed methodology further aims to reveal known unknowns and unknown unknowns, including tactics, techniques, and procedures that are emerging, anticipated, and plausible. This enhances the predictability of adversary behavior, particularly under future uncertainty, helping security leaders make informed decisions today that shape more resilient security postures for the future. 

**Abstract (ZH)**: 未来向后威胁建模（FBTM）：预测性方法及其应用 

---
# SpectralTrain: A Universal Framework for Hyperspectral Image Classification 

**Title (ZH)**: SpectralTrain: 通用的高光谱图像分类框架 

**Authors**: Meihua Zhou, Liping Yu, Jiawei Cai, Wai Kin Fung, Ruiguo Hu, Jiarui Zhao, Wenzhuo Liu, Nan Wan  

**Link**: [PDF](https://arxiv.org/pdf/2511.16084)  

**Abstract**: Hyperspectral image (HSI) classification typically involves large-scale data and computationally intensive training, which limits the practical deployment of deep learning models in real-world remote sensing tasks. This study introduces SpectralTrain, a universal, architecture-agnostic training framework that enhances learning efficiency by integrating curriculum learning (CL) with principal component analysis (PCA)-based spectral downsampling. By gradually introducing spectral complexity while preserving essential information, SpectralTrain enables efficient learning of spectral -- spatial patterns at significantly reduced computational costs. The framework is independent of specific architectures, optimizers, or loss functions and is compatible with both classical and state-of-the-art (SOTA) models. Extensive experiments on three benchmark datasets -- Indian Pines, Salinas-A, and the newly introduced CloudPatch-7 -- demonstrate strong generalization across spatial scales, spectral characteristics, and application domains. The results indicate consistent reductions in training time by 2-7x speedups with small-to-moderate accuracy deltas depending on backbone. Its application to cloud classification further reveals potential in climate-related remote sensing, emphasizing training strategy optimization as an effective complement to architectural design in HSI models. Code is available at this https URL. 

**Abstract (ZH)**: 高光谱图像(HSI)分类通常涉及大规模数据和计算密集型训练，这限制了深度学习模型在实际遥感任务中的应用部署。本文介绍了一种通用且架构无关的训练框架SpectralTrain，通过结合梯度课程学习(CL)与基于主成分分析(PCA)的光谱降采样，提升了学习效率。SpectralTrain通过逐步引入光谱复杂性并保留关键信息，使在显著降低计算成本的前提下高效学习光谱-空间模式成为可能。该框架独立于特定架构、优化器或损失函数，并与经典和最新的(SOTA)模型兼容。在印度pine、Salinas-A以及新引入的CloudPatch-7三个基准数据集上的广泛实验表明，其在空间尺度、光谱特性及应用领域展现出强大的泛化能力。结果表明，其训练时间比传统方法可减少2-7倍，同时小到中等程度的准确率差异依赖于所使用的骨干网络。其在云分类中的应用进一步表明，该方法在气候相关遥感中具有潜在应用价值，强调了训练策略优化是提高HSI模型性能的有效补充。代码可从此链接获取。 

---
# Operon: Incremental Construction of Ragged Data via Named Dimensions 

**Title (ZH)**: Operon: 命名维度下增量构建不规则数据的方法 

**Authors**: Sungbin Moon, Jiho Park, Suyoung Hwang, Donghyun Koh, Seunghyun Moon, Minhyeong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.16080)  

**Abstract**: Modern data processing workflows frequently encounter ragged data: collections with variable-length elements that arise naturally in domains like natural language processing, scientific measurements, and autonomous AI agents. Existing workflow engines lack native support for tracking the shapes and dependencies inherent to ragged data, forcing users to manage complex indexing and dependency bookkeeping manually. We present Operon, a Rust-based workflow engine that addresses these challenges through a novel formalism of named dimensions with explicit dependency relations. Operon provides a domain-specific language where users declare pipelines with dimension annotations that are statically verified for correctness, while the runtime system dynamically schedules tasks as data shapes are incrementally discovered during execution. We formalize the mathematical foundation for reasoning about partial shapes and prove that Operon's incremental construction algorithm guarantees deterministic and confluent execution in parallel settings. The system's explicit modeling of partially-known states enables robust persistence and recovery mechanisms, while its per-task multi-queue architecture achieves efficient parallelism across heterogeneous task types. Empirical evaluation demonstrates that Operon outperforms an existing workflow engine with 14.94x baseline overhead reduction while maintaining near-linear end-to-end output rates as workloads scale, making it particularly suitable for large-scale data generation pipelines in machine learning applications. 

**Abstract (ZH)**: 现代数据处理工作流经常遇到不规则数据：具有可变长度元素的集合，这些元素在自然语言处理、科学研究和自主AI代理等领域自然出现。现有的工作流引擎缺乏对不规则数据内在形状和依赖关系的原生支持，迫使用户手动管理复杂的索引和依赖关系记录。我们提出了一个基于Rust的工作流引擎Operon，通过一种新的命名维度形式主义和显式的依赖关系来解决这些挑战。Operon提供了一个领域特定的语言，用户可以在其中声明带有维度注解的管道，这些注解可以静态验证其正确性，而运行时系统则根据执行过程中数据形状的逐步发现来动态调度任务。我们为关于部分形状的数学基础进行了形式化，并证明了Operon的增量构建算法在并行设置中保证了确定性和会聚的执行。该系统的部分已知状态显式建模使持久性和恢复机制更加健壮，而其每任务多队列架构实现了异构任务类型之间的高效并行性。实证评估表明，与现有的工作流引擎相比，Operon的基线开销减少了14.94倍，同时在工作负载扩展时保持近乎线性的端到端输出速率，使其特别适合于机器学习应用中的大规模数据生成管道。 

---
# A Mathematical Framework for Custom Reward Functions in Job Application Evaluation using Reinforcement Learning 

**Title (ZH)**: 基于强化学习的工作申请评估中自定义奖励函数的数学框架 

**Authors**: Shreyansh Jain, Madhav Singhvi, Shreya Rahul Jain, Pranav S, Dishaa Lokesh, Naren Chittibabu, Akash Anandhan  

**Link**: [PDF](https://arxiv.org/pdf/2511.16073)  

**Abstract**: Conventional Applicant Tracking Systems (ATS) tend to be inflexible keyword-matchers, and deny gifted candidates a role due to a few minor semantic mismatches. This article describes a new two-step process to design a more refined resume evaluation model based on a small language model (<600M parameters) that is finetuned using GRPO on a custom reward function. To begin with, Supervised Fine-Tuning (SFT) was used to build a solid baseline model. Second, this SFT model was also optimized with the help of Reinforcement Learning (RL) through GRPO under the guidance of a new, multi-component reward function that can holistically assess candidates beyond simple keyword matching. We indicate that the RL application presents a critical problem of reward hacking due to the initial experiments of aggressive penalties, which produces faulty, excessively negative model behaviors. We have overcome this challenge by refining the reward function repeatedly and training hyperparameters into a stable "gentle polishing process" of the reward function. Our resulting GRPO-polished model demonstrates significant real-world efficacy, achieving a final accuracy of 91% on unseen test data. The model shows a strong ability to correctly identify qualified candidates (recall of 0.85 for the 'SELECTED' class) while also showing exceptional precision (1.0), confirming its reliability. These results indicate that a properly executed, two-step fine-tuning procedure can indeed effectively refine a small language model to be able to conduct fine-tuned and human-like candidate scoring, overcoming the drawbacks of both traditional ATS and naive RL usage. 

**Abstract (ZH)**: 基于小型语言模型和强化学习的应聘者精细评估模型设计方法 

---
# Early science acceleration experiments with GPT-5 

**Title (ZH)**: GPT-5早期科学加速实验 

**Authors**: Sébastien Bubeck, Christian Coester, Ronen Eldan, Timothy Gowers, Yin Tat Lee, Alexandru Lupsasca, Mehtaab Sawhney, Robert Scherrer, Mark Sellke, Brian K. Spears, Derya Unutmaz, Kevin Weil, Steven Yin, Nikita Zhivotovskiy  

**Link**: [PDF](https://arxiv.org/pdf/2511.16072)  

**Abstract**: AI models like GPT-5 are an increasingly valuable tool for scientists, but many remain unaware of the capabilities of frontier AI. We present a collection of short case studies in which GPT-5 produced new, concrete steps in ongoing research across mathematics, physics, astronomy, computer science, biology, and materials science. In these examples, the authors highlight how AI accelerated their work, and where it fell short; where expert time was saved, and where human input was still key. We document the interactions of the human authors with GPT-5, as guiding examples of fruitful collaboration with AI. Of note, this paper includes four new results in mathematics (carefully verified by the human authors), underscoring how GPT-5 can help human mathematicians settle previously unsolved problems. These contributions are modest in scope but profound in implication, given the rate at which frontier AI is progressing. 

**Abstract (ZH)**: AI模型如GPT-5已成为科学家们越来越有价值的工具，但许多科学家仍 unaware 于前沿AI的能力。我们介绍了GPT-5在数学、物理学、天文学、计算机科学、生物学和材料科学等领域推动正在进行的研究的新颖具体案例。在这些案例中，作者强调了AI如何加速他们的工作，以及它的局限性；哪些专家时间得到了节省，哪些环节仍需人类输入。我们记录了人类作者与GPT-5的互动，作为与AI进行富有成效合作的范例。值得注意的是，本文包括了四项新的数学成果（由人类作者仔细验证），突显了GPT-5如何帮助人类数学家解决以前未能解决的问题。这些贡献在范围上可能相对有限，但在给定前沿AI快速进展的背景下，其影响是深远的。 

---
# Learning Tractable Distributions Of Language Model Continuations 

**Title (ZH)**: 学习可计算的语言模型续篇分布 

**Authors**: Gwen Yidou-Weng, Ian Li, Anji Liu, Oliver Broadrick, Guy Van den Broeck, Benjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16054)  

**Abstract**: Controlled language generation conditions text on sequence-level constraints (for example, syntax, style, or safety). These constraints may depend on future tokens, which makes directly conditioning an autoregressive language model (LM) generally intractable. Prior work uses tractable surrogates such as hidden Markov models (HMMs) to approximate the distribution over continuations and adjust the model's next-token logits at decoding time. However, we find that these surrogates are often weakly context aware, which reduces query quality. We propose Learning to Look Ahead (LTLA), a hybrid approach that pairs the same base language model for rich prefix encoding with a fixed tractable surrogate model that computes exact continuation probabilities. Two efficiency pitfalls arise when adding neural context: (i) naively rescoring the prefix with every candidate next token requires a sweep over the entire vocabulary at each step, and (ii) predicting fresh surrogate parameters for each prefix, although tractable at a single step, forces recomputation of future probabilities for every new prefix and eliminates reuse. LTLA avoids both by using a single batched HMM update to account for all next-token candidates at once, and by conditioning only the surrogate's latent state prior on the LM's hidden representations while keeping the surrogate decoder fixed, so computations can be reused across prefixes. Empirically, LTLA attains higher conditional likelihood than an unconditional HMM, approximates continuation distributions for vision-language models where a standalone HMM cannot encode visual context, and improves constraint satisfaction at comparable fluency on controlled-generation tasks, with minimal inference overhead. 

**Abstract (ZH)**: Controlled语言生成通过序列级约束（例如句法、风格或安全性）对文本进行控制。这些约束可能依赖于未来的词，这使得直接条件化自回归语言模型（LM）通常变得不可行。先前的工作使用隐马尔可夫模型（HMMs）等可处理的替代模型来近似后续内容的分布并在解码时调整模型的下一个词的概率。然而，我们发现这些替代模型往往对上下文的感知较弱，这降低了查询质量。我们提出了一种结合方法——Learn to Look Ahead (LTLA)。该方法通过使用丰富的前缀编码基础语言模型与固定可处理替代模型相结合，后者可以计算精确的后续概率。当添加神经上下文时，会出现两种效率陷阱：（i）朴素地用每个候选的下一个词重新评分前缀要求在每一步中对整个词汇表进行扫描；（ii）对于每个前缀预测新鲜的替代参数虽然单步是可处理的，但要求重新计算每个新前缀的未来概率且不能再利用前缀之间的计算结果。LTLA 通过一次性使用批量HMM更新来同时考虑所有下一个词的候选，避免了上述两种陷阱，并且仅通过基础语言模型的隐藏表示条件化替代模型的潜状态，而保持替代模型解码器固定，从而使计算结果可以在不同前缀之间重用。实验结果表明，LTLA 相比于无条件的HMM具有更高的条件似然，可以为视觉-语言模型近似后续内容的分布，即使独立的HMM无法编码视觉上下文，并且在控制生成任务中以相似的流畅度提高约束满足度，同时具有最小的推理开销。 

---
# Semantic Glitch: Agency and Artistry in an Autonomous Pixel Cloud 

**Title (ZH)**: 语义缺陷：自主像素云中的代理与艺术创造力 

**Authors**: Qing Zhang, Jing Huang, Mingyang Xu, Jun Rekimoto  

**Link**: [PDF](https://arxiv.org/pdf/2511.16048)  

**Abstract**: While mainstream robotics pursues metric precision and flawless performance, this paper explores the creative potential of a deliberately "lo-fi" approach. We present the "Semantic Glitch," a soft flying robotic art installation whose physical form, a 3D pixel style cloud, is a "physical glitch" derived from digital archaeology. We detail a novel autonomous pipeline that rejects conventional sensors like LiDAR and SLAM, relying solely on the qualitative, semantic understanding of a Multimodal Large Language Model to navigate. By authoring a bio-inspired personality for the robot through a natural language prompt, we create a "narrative mind" that complements the "weak," historically, loaded body. Our analysis begins with a 13-minute autonomous flight log, and a follow-up study statistically validates the framework's robustness for authoring quantifiably distinct personas. The combined analysis reveals emergent behaviors, from landmark-based navigation to a compelling "plan to execution" gap, and a character whose unpredictable, plausible behavior stems from a lack of precise proprioception. This demonstrates a lo-fi framework for creating imperfect companions whose success is measured in character over efficiency. 

**Abstract (ZH)**: 尽管主流机器人追求精确度和无瑕疵的性能，本文探讨了一种故意采用“低分辨率”方法的创作潜力。我们介绍了“语义错觉”这一软体飞行机器人艺术装置，其物理形态是一个以3D像素风格呈现的云状结构，是一种源自数字考古学的“物理错觉”。我们详细阐述了一种新的自主处理流程，该流程拒绝使用如激光雷达和SLAM等传统传感器，仅依靠多模态大型语言模型的定性、语义理解来进行导航。通过使用自然语言提示为机器人编撰一种生物启发式个性，我们创造了一个“叙事心灵”，以补充其历史上充满负载的“虚弱”身体。我们的分析始于一个13分钟的自主飞行日志，并随后进行的研究统计上验证了该框架在编撰可量化差异的人格方面的鲁棒性。综合分析揭示了 Emergent 行为，从基于地标导航到具有说服力的“计划到执行”差距，以及一个行为不可预测、合乎情理的字符，其根源在于精确的位置感知缺失。这展示了用于创建不完美伴侣的低分辨率框架，其成功之处在于特色而非效率。 

---
# Liars' Bench: Evaluating Lie Detectors for Language Models 

**Title (ZH)**: 说谎者的长凳：评估语言模型中的说谎检测器 

**Authors**: Kieron Kretschmar, Walter Laurito, Sharan Maiya, Samuel Marks  

**Link**: [PDF](https://arxiv.org/pdf/2511.16035)  

**Abstract**: Prior work has introduced techniques for detecting when large language models (LLMs) lie, that is, generating statements they believe are false. However, these techniques are typically validated in narrow settings that do not capture the diverse lies LLMs can generate. We introduce LIARS' BENCH, a testbed consisting of 72,863 examples of lies and honest responses generated by four open-weight models across seven datasets. Our settings capture qualitatively different types of lies and vary along two dimensions: the model's reason for lying and the object of belief targeted by the lie. Evaluating three black- and white-box lie detection techniques on LIARS' BENCH, we find that existing techniques systematically fail to identify certain types of lies, especially in settings where it's not possible to determine whether the model lied from the transcript alone. Overall, LIARS' BENCH reveals limitations in prior techniques and provides a practical testbed for guiding progress in lie detection. 

**Abstract (ZH)**: Prior Work Has Introduced Techniques for Detecting When Large Language Models Lie: LIARS' BENCH, a Testbed Consisting of 72,863 Examples of Lies and Honest Responses Generated by Four Open-Weight Models Across Seven Datasets 

---
# HGCN2SP: Hierarchical Graph Convolutional Network for Two-Stage Stochastic Programming 

**Title (ZH)**: HGCN2SP: 分层图卷积网络应用于两阶段随机规划 

**Authors**: Yang Wu, Yifan Zhang, Zhenxing Liang, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.16027)  

**Abstract**: Two-stage Stochastic Programming (2SP) is a standard framework for modeling decision-making problems under uncertainty. While numerous methods exist, solving such problems with many scenarios remains challenging. Selecting representative scenarios is a practical method for accelerating solutions. However, current approaches typically rely on clustering or Monte Carlo sampling, failing to integrate scenario information deeply and overlooking the significant impact of the scenario order on solving time. To address these issues, we develop HGCN2SP, a novel model with a hierarchical graph designed for 2SP problems, encoding each scenario and modeling their relationships hierarchically. The model is trained in a reinforcement learning paradigm to utilize the feedback of the solver. The policy network is equipped with a hierarchical graph convolutional network for feature encoding and an attention-based decoder for scenario selection in proper order. Evaluation of two classic 2SP problems demonstrates that HGCN2SP provides high-quality decisions in a short computational time. Furthermore, HGCN2SP exhibits remarkable generalization capabilities in handling large-scale instances, even with a substantial number of variables or scenarios that were unseen during the training phase. 

**Abstract (ZH)**: 基于层次图的两阶段随机规划模型（HGCN2SP） 

---
# Towards a Safer and Sustainable Manufacturing Process: Material classification in Laser Cutting Using Deep Learning 

**Title (ZH)**: 面向更安全和可持续的制造过程：激光切割中基于深度学习的材料分类 

**Authors**: Mohamed Abdallah Salem, Hamdy Ahmed Ashur, Ahmed Elshinnawy  

**Link**: [PDF](https://arxiv.org/pdf/2511.16026)  

**Abstract**: Laser cutting is a widely adopted technology in material processing across various industries, but it generates a significant amount of dust, smoke, and aerosols during operation, posing a risk to both the environment and workers' health. Speckle sensing has emerged as a promising method to monitor the cutting process and identify material types in real-time. This paper proposes a material classification technique using a speckle pattern of the material's surface based on deep learning to monitor and control the laser cutting process. The proposed method involves training a convolutional neural network (CNN) on a dataset of laser speckle patterns to recognize distinct material types for safe and efficient cutting. Previous methods for material classification using speckle sensing may face issues when the color of the laser used to produce the speckle pattern is changed. Experiments conducted in this study demonstrate that the proposed method achieves high accuracy in material classification, even when the laser color is changed. The model achieved an accuracy of 98.30 % on the training set and 96.88% on the validation set. Furthermore, the model was evaluated on a set of 3000 new images for 30 different materials, achieving an F1-score of 0.9643. The proposed method provides a robust and accurate solution for material-aware laser cutting using speckle sensing. 

**Abstract (ZH)**: 基于-speckle-模式的深度学习材料分类方法在激光切割中的应用 

---
# Physically Realistic Sequence-Level Adversarial Clothing for Robust Human-Detection Evasion 

**Title (ZH)**: 物理现实性的序列级 adversarial 衣服以实现稳健的人体检测规避 

**Authors**: Dingkun Zhou, Patrick P. K. Chan, Hengxu Wu, Shikang Zheng, Ruiqi Huang, Yuanjie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.16020)  

**Abstract**: Deep neural networks used for human detection are highly vulnerable to adversarial manipulation, creating safety and privacy risks in real surveillance environments. Wearable attacks offer a realistic threat model, yet existing approaches usually optimize textures frame by frame and therefore fail to maintain concealment across long video sequences with motion, pose changes, and garment deformation. In this work, a sequence-level optimization framework is introduced to generate natural, printable adversarial textures for shirts, trousers, and hats that remain effective throughout entire walking videos in both digital and physical settings. Product images are first mapped to UV space and converted into a compact palette and control-point parameterization, with ICC locking to keep all colors printable. A physically based human-garment pipeline is then employed to simulate motion, multi-angle camera viewpoints, cloth dynamics, and illumination variation. An expectation-over-transformation objective with temporal weighting is used to optimize the control points so that detection confidence is minimized across whole sequences. Extensive experiments demonstrate strong and stable concealment, high robustness to viewpoint changes, and superior cross-model transferability. Physical garments produced with sublimation printing achieve reliable suppression under indoor and outdoor recordings, confirming real-world feasibility. 

**Abstract (ZH)**: 深度神经网络用于人体检测的高度易受对抗性操纵攻击的影响在实际监控环境中创造了安全和隐私风险。穿戴式攻击提供了真实的安全威胁模型，但现有方法通常逐帧优化纹理，因此在包含动作、姿态变化和衣物变形的长时间视频序列中难以保持隐蔽性。本文提出了一种序列级优化框架，以生成适用于衬衫、裤子和帽子并在整个行走视频中保持有效性的自然可打印对抗性纹理，适用于数字和物理环境。首先将产品图片映射到UV空间并转换为紧凑调色板和控制点参数化，并使用ICC锁定以保持所有颜色可打印。随后采用基于物理的人衣模拟管道来模拟动作、多角度摄像头视角、织物力学特性和光照变化。使用带时间加权的期望过变换目标来优化控制点，以最小化整个序列中的检测置信度。实验结果表明，具有高度稳定隐蔽性、对视角变化高鲁棒性和跨模型优越传输性的性能。通过升华打印制作的物理衣物在室内和室外录制中实现了可靠的抑制，确认了其实用性。 

---
# CARE: Turning LLMs Into Causal Reasoning Expert 

**Title (ZH)**: CARE: 将大语言模型转化为因果推理专家 

**Authors**: Juncheng Dong, Yiling Liu, Ahmed Aloui, Vahid Tarokh, David Carlson  

**Link**: [PDF](https://arxiv.org/pdf/2511.16016)  

**Abstract**: Large language models (LLMs) have recently demonstrated impressive capabilities across a range of reasoning and generation tasks. However, research studies have shown that LLMs lack the ability to identify causal relationships, a fundamental cornerstone of human intelligence. We first conduct an exploratory investigation of LLMs' behavior when asked to perform a causal-discovery task and find that they mostly rely on the semantic meaning of variable names, ignoring the observation data. This is unsurprising, given that LLMs were never trained to process structural datasets. To first tackle this challenge, we prompt the LLMs with the outputs of established causal discovery algorithms designed for observational datasets. These algorithm outputs effectively serve as the sufficient statistics of the observation data. However, quite surprisingly, we find that prompting the LLMs with these sufficient statistics decreases the LLMs' performance in causal discovery. To address this current limitation, we propose CARE, a framework that enhances LLMs' causal-reasoning ability by teaching them to effectively utilize the outputs of established causal-discovery algorithms through supervised fine-tuning. Experimental results show that a finetuned Qwen2.5-1.5B model produced by CARE significantly outperforms both traditional causal-discovery algorithms and state-of-the-art LLMs with over a thousand times more parameters, demonstrating effective utilization of its own knowledge and the external algorithmic clues. 

**Abstract (ZH)**: 大型语言模型（LLMs）在一系列推理和生成任务中展示了令人印象深刻的能力。然而，研究显示LLMs缺乏识别因果关系的能力，这是人类智能的一个基本要素。我们首先对LLMs在执行因果发现任务时的行为进行了探索性研究，发现它们主要依赖变量名称的语义含义，忽视了观测数据。鉴于LLMs从未被训练处理结构性数据集，我们首先尝试通过提供用于观察数据集的现有因果发现算法的输出来应对这一挑战。这些算法的输出有效充当了观测数据的充分统计量。然而，令人惊讶的是，我们发现提供这些充分统计量给LLMs反而降低了它们在因果发现任务上的表现。为了应对这一当前限制，我们提出了CARE框架，通过监督微调方式教会LLMs有效利用现有因果发现算法的输出来增强其因果推理能力。实验结果显示，CARE生成的微调后的Qwen2.5-1.5B模型在因果发现任务上的表现显著优于传统因果发现算法和具有数千倍更多参数的最新LLMs，展示了其有效利用自身知识和外部算法线索的能力。 

---
# Physics-Guided Inductive Spatiotemporal Kriging for PM2.5 with Satellite Gradient Constraints 

**Title (ZH)**: 基于物理指导的诱导时空克里金模型及其卫星梯度约束应用于PM2.5预测 

**Authors**: Shuo Wang, Mengfan Teng, Yun Cheng, Lothar Thiele, Olga Saukh, Shuangshuang He, Yuanting Zhang, Jiang Zhang, Gangfeng Zhang, Xingyuan Yuan, Jingfang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2511.16013)  

**Abstract**: High-resolution mapping of fine particulate matter (PM2.5) is a cornerstone of sustainable urbanism but remains critically hindered by the spatial sparsity of ground monitoring networks. While traditional data-driven methods attempt to bridge this gap using satellite Aerosol Optical Depth (AOD), they often suffer from severe, non-random data missingness (e.g., due to cloud cover or nighttime) and inversion biases. To overcome these limitations, this study proposes the Spatiotemporal Physics-Guided Inference Network (SPIN), a novel framework designed for inductive spatiotemporal kriging. Unlike conventional approaches, SPIN synergistically integrates domain knowledge into deep learning by explicitly modeling physical advection and diffusion processes via parallel graph kernels. Crucially, we introduce a paradigm-shifting training strategy: rather than using error-prone AOD as a direct input, we repurpose it as a spatial gradient constraint within the loss function. This allows the model to learn structural pollution patterns from satellite data while remaining robust to data voids. Validated in the highly polluted Beijing-Tianjin-Hebei and Surrounding Areas (BTHSA), SPIN achieves a new state-of-the-art with a Mean Absolute Error (MAE) of 9.52 ug/m^3, effectively generating continuous, physically plausible pollution fields even in unmonitored areas. This work provides a robust, low-cost, and all-weather solution for fine-grained environmental management. 

**Abstract (ZH)**: 高分辨率细颗粒物（PM2.5）分布mapping及其在可持续城市规划中的应用仍受地面监测网络空间稀疏性的限制，而时空物理引导推理网络（SPIN）为这一挑战提供了解决方案。 

---
# Synergizing Deconfounding and Temporal Generalization For Time-series Counterfactual Outcome Estimation 

**Title (ZH)**: 协同去混杂与时间泛化以实现时间序列反事实效果估计 

**Authors**: Yiling Liu, Juncheng Dong, Chen Fu, Wei Shi, Ziyang Jiang, Zhigang Hua, David Carlson  

**Link**: [PDF](https://arxiv.org/pdf/2511.16006)  

**Abstract**: Estimating counterfactual outcomes from time-series observations is crucial for effective decision-making, e.g. when to administer a life-saving treatment, yet remains significantly challenging because (i) the counterfactual trajectory is never observed and (ii) confounders evolve with time and distort estimation at every step. To address these challenges, we propose a novel framework that synergistically integrates two complementary approaches: Sub-treatment Group Alignment (SGA) and Random Temporal Masking (RTM). Instead of the coarse practice of aligning marginal distributions of the treatments in latent space, SGA uses iterative treatment-agnostic clustering to identify fine-grained sub-treatment groups. Aligning these fine-grained groups achieves improved distributional matching, thus leading to more effective deconfounding. We theoretically demonstrate that SGA optimizes a tighter upper bound on counterfactual risk and empirically verify its deconfounding efficacy. RTM promotes temporal generalization by randomly replacing input covariates with Gaussian noises during training. This encourages the model to rely less on potentially noisy or spuriously correlated covariates at the current step and more on stable historical patterns, thereby improving its ability to generalize across time and better preserve underlying causal relationships. Our experiments demonstrate that while applying SGA and RTM individually improves counterfactual outcome estimation, their synergistic combination consistently achieves state-of-the-art performance. This success comes from their distinct yet complementary roles: RTM enhances temporal generalization and robustness across time steps, while SGA improves deconfounding at each specific time point. 

**Abstract (ZH)**: 从时间序列观察中估计反事实结果对于有效决策至关重要，例如何时施用救命治疗，但这仍然极具挑战性，因为（i）反事实轨迹从未被观测到，（ii）混杂因素会随时间演变并在每一步扭曲估算。为应对这些挑战，我们提出了一种新的框架，该框架结合了两种互补的方法：亚治疗组对齐（SGA）和随机时间掩蔽（RTM）。SGA 不是通过在潜在空间中合并治疗的边际分布来进行粗略处理，而是使用迭代的治疗无关聚类来识别细粒度的亚治疗组。对齐这些细粒度组实现了更好的分布匹配，从而提高了去混杂的有效性。我们从理论上证明了SGA优化了反事实风险的更紧的上界，并通过实验证实了其去混杂的有效性。RTM 在训练过程中通过随机用高斯噪声替换输入协变量来促进时间泛化。这促使模型在当前步骤较少依赖可能的噪声或假相关协变量，更多地依赖于稳定的历史模式，从而增强了其跨时间的时间泛化能力和更好地保持潜在因果关系。我们的实验表明，虽然单独应用SGA和RTM可以改善反事实结果的估计，但它们的协同组合始终能够实现最佳性能。这种成功来自于它们各自独特的互补作用：RTM 增强了跨时间步骤的时间泛化能力和鲁棒性，而SGA 在每个具体时间点提高了去混杂的能力。 

---
# InfCode-C++: Intent-Guided Semantic Retrieval and AST-Structured Search for C++ Issue Resolution 

**Title (ZH)**: InfCode-C++: 意图导向的语义检索与AST结构化搜索以解决C++问题 

**Authors**: Qingao Dong, Mengfei Wang, Hengzhi Zhang, Zhichao Li, Yuan Yuan, Mu Li, Xiang Gao, Hailong Sun, Chunming Hu, Weifeng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2511.16005)  

**Abstract**: Large language model (LLM) agents have recently shown strong performance on repository-level issue resolution, but existing systems are almost exclusively designed for Python and rely heavily on lexical retrieval and shallow code navigation. These approaches transfer poorly to C++ projects, where overloaded identifiers, nested namespaces, template instantiations, and deep control-flow structures make context retrieval and fault localization substantially more difficult. As a result, state-of-the-art Python-oriented agents show a drastic performance drop on the C++ subset of MultiSWE-bench. We introduce INFCODE-C++, the first C++-aware autonomous system for end-to-end issue resolution. The system combines two complementary retrieval mechanisms -- semantic code-intent retrieval and deterministic AST-structured querying -- to construct accurate, language-aware context for this http URL components enable precise localization and robust patch synthesis in large, statically typed C++ repositories. Evaluated on the \texttt{MultiSWE-bench-CPP} benchmark, INFCODE-C++ achieves a resolution rate of 25.58\%, outperforming the strongest prior agent by 10.85 percentage points and more than doubling the performance of MSWE-agent. Ablation and behavioral studies further demonstrate the critical role of semantic retrieval, structural analysis, and accurate reproduction in C++ issue resolution. INFCODE-C++ highlights the need for language-aware reasoning in multi-language software agents and establishes a foundation for future research on scalable, LLM-driven repair for complex, statically typed ecosystems. 

**Abstract (ZH)**: Large语言模型（LLM）代理在仓库级别问题解决方面表现出强大的性能，但现有系统几乎仅设计用于Python，并且严重依赖于词汇检索和浅层代码导航。这些方法在C++项目中表现不佳，因为C++项目中的重载标识符、嵌套命名空间、模板实例化以及复杂的控制流结构使上下文检索和故障定位变得更加困难。因此，最先进的面向Python的代理在MultiSWE-bench的C++子集中显示出显著的性能下降。我们提出了INFCODE-C++，这是第一个具备C++意识的端到端问题解决自主系统。该系统结合了语义代码意图检索和确定性AST结构查询两种互补的检索机制，以构建准确的语言意识上下文，精确定位和生成大型静态类型C++仓库中的稳定补丁。INFCODE-C++在\texttt{MultiSWE-bench-CPP}基准测试中实现了25.58%的问题解决率，比 strongest 的先前代理高10.85个百分点，并且将MSWE-agent的性能提高了一倍多。进一步的消融研究和行为研究表明，在C++问题解决中语义检索、结构分析和精确重现场景的角色至关重要。INFCODE-C++强调多语言软件代理中语言意识推理的必要性，并为未来研究大规模、基于LLM的复杂静态类型生态系统修复奠定了基础。 

---
# InfCode: Adversarial Iterative Refinement of Tests and Patches for Reliable Software Issue Resolution 

**Title (ZH)**: InfCode: 敌对迭代测试与修补件 refinement 以实现可靠的软件问题解决 

**Authors**: KeFan Li, Mengfei Wang, Hengzhi Zhang, Zhichao Li, Yuan Yuan, Mu Li, Xiang Gao, Hailong Sun, Chunming Hu, Weifeng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2511.16004)  

**Abstract**: Large language models have advanced software engineering automation, yet resolving real-world software issues remains difficult because it requires repository-level reasoning, accurate diagnostics, and strong verification signals. Existing agent-based and pipeline-based methods often rely on insufficient tests, which can lead to patches that satisfy verification but fail to fix the underlying defect. We present InfCode, an adversarial multi-agent framework for automated repository-level issue resolution. InfCode iteratively refines both tests and patches through adversarial interaction between a Test Patch Generator and a Code Patch Generator, while a Selector agent identifies the most reliable fix. The framework runs inside a containerized environment that supports realistic repository inspection, modification, and validation. Experiments on SWE-bench Lite and SWE-bench Verified using models such as DeepSeek-V3 and Claude 4.5 Sonnet show that InfCode consistently outperforms strong baselines. It achieves 79.4% performance on SWE-bench Verified, establishing a new state-of-the-art. We have released InfCode as an open-source project at this https URL. 

**Abstract (ZH)**: 大型语言模型推进了软件工程自动化，但解决实际软件问题仍然困难，因为这需要仓库级推理、精确诊断和强烈验证信号。现有基于代理和基于流水线的方法往往依赖于不充分的测试，可能导致满足验证但未能修复根本缺陷的补丁。我们提出InfCode，一种对抗式多代理框架，用于自动化仓库级问题解决。InfCode通过测试生成器和代码生成器之间的对抗性交互，迭代优化测试和补丁，同时选择器代理识别最可靠的修复。该框架在一个支持真实仓库检查、修改和验证的容器化环境中运行。使用DeepSeek-V3和Claude 4.5 Sonnet等模型在SWE-bench Lite和SWE-bench Verified上的实验显示，InfCode 持续优于强基线，达到SWE-bench Verified 79.4%的性能，建立新的最新水平。我们已将InfCode作为开源项目发布在该网址。 

---
# Hiding in the AI Traffic: Abusing MCP for LLM-Powered Agentic Red Teaming 

**Title (ZH)**: 在AI流量中隐身：滥用MCP进行由大语言模型驱动的红队演练 

**Authors**: Strahinja Janjuesvic, Anna Baron Garcia, Sohrob Kazerounian  

**Link**: [PDF](https://arxiv.org/pdf/2511.15998)  

**Abstract**: Generative AI is reshaping offensive cybersecurity by enabling autonomous red team agents that can plan, execute, and adapt during penetration tests. However, existing approaches face trade-offs between generality and specialization, and practical deployments reveal challenges such as hallucinations, context limitations, and ethical concerns. In this work, we introduce a novel command & control (C2) architecture leveraging the Model Context Protocol (MCP) to coordinate distributed, adaptive reconnaissance agents covertly across networks. Notably, we find that our architecture not only improves goal-directed behavior of the system as whole, but also eliminates key host and network artifacts that can be used to detect and prevent command & control behavior altogether. We begin with a comprehensive review of state-of-the-art generative red teaming methods, from fine-tuned specialist models to modular or agentic frameworks, analyzing their automation capabilities against task-specific accuracy. We then detail how our MCP-based C2 can overcome current limitations by enabling asynchronous, parallel operations and real-time intelligence sharing without periodic beaconing. We furthermore explore advanced adversarial capabilities of this architecture, its detection-evasion techniques, and address dual-use ethical implications, proposing defensive measures and controlled evaluation in lab settings. Experimental comparisons with traditional C2 show drastic reductions in manual effort and detection footprint. We conclude with future directions for integrating autonomous exploitation, defensive LLM agents, predictive evasive maneuvers, and multi-agent swarms. The proposed MCP-enabled C2 framework demonstrates a significant step toward realistic, AI-driven red team operations that can simulate advanced persistent threats while informing the development of next-generation defensive systems. 

**Abstract (ZH)**: 基于Model Context Protocol的命令与控制架构：重塑生成式红队攻击 

---
# Efficient Chromosome Parallelization for Precision Medicine Genomic Workflows 

**Title (ZH)**: 精准医学基因组 workflows 中的高效染色体并行化 

**Authors**: Daniel Mas Montserrat, Ray Verma, Míriam Barrabés, Francisco M. de la Vega, Carlos D. Bustamante, Alexander G. Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2511.15977)  

**Abstract**: Large-scale genomic workflows used in precision medicine can process datasets spanning tens to hundreds of gigabytes per sample, leading to high memory spikes, intensive disk I/O, and task failures due to out-of-memory errors. Simple static resource allocation methods struggle to handle the variability in per-chromosome RAM demands, resulting in poor resource utilization and long runtimes. In this work, we propose multiple mechanisms for adaptive, RAM-efficient parallelization of chromosome-level bioinformatics workflows. First, we develop a symbolic regression model that estimates per-chromosome memory consumption for a given task and introduces an interpolating bias to conservatively minimize over-allocation. Second, we present a dynamic scheduler that adaptively predicts RAM usage with a polynomial regression model, treating task packing as a Knapsack problem to optimally batch jobs based on predicted memory requirements. Additionally, we present a static scheduler that optimizes chromosome processing order to minimize peak memory while preserving throughput. Our proposed methods, evaluated on simulations and real-world genomic pipelines, provide new mechanisms to reduce memory overruns and balance load across threads. We thereby achieve faster end-to-end execution, showcasing the potential to optimize large-scale genomic workflows. 

**Abstract (ZH)**: 大规模基因组工作流在精准医疗中的使用可以处理每个样本跨度为数十到数百吉字节的数据集，导致内存峰值升高、密集的磁盘I/O操作以及由于内存不足错误导致的任务失败。简单的静态资源分配方法难以处理每条染色体RAM需求的变异性，导致资源利用效率低下和较长的运行时间。在本文中，我们提出了多种机制以实现适应性的、RAM效率高的染色体级别生物信息学工作流的并行化。首先，我们开发了一种符号回归模型来估计给定任务的每条染色体内存消耗，并引入插值偏差以保守地最小化资源过度分配。其次，我们提出了一种动态调度器，该调度器使用多项式回归模型自适应地预测RAM使用情况，并将任务打包视为背包问题，以基于预测的内存需求最优化地批量作业。此外，我们提出了一种静态调度器，该调度器优化染色体处理顺序以最小化峰值内存占用，同时保持吞吐量。我们提出的方法在模拟和实际基因组管道上进行评估，提供了新的机制以减少内存溢出并平衡线程负载。我们因此实现了端到端执行速度的提升，展示了优化大规模基因组工作流的潜力。 

---
# A Primer on Quantum Machine Learning 

**Title (ZH)**: 量子机器学习入门 

**Authors**: Su Yeon Chang, M. Cerezo  

**Link**: [PDF](https://arxiv.org/pdf/2511.15969)  

**Abstract**: Quantum machine learning (QML) is a computational paradigm that seeks to apply quantum-mechanical resources to solve learning problems. As such, the goal of this framework is to leverage quantum processors to tackle optimization, supervised, unsupervised and reinforcement learning, and generative modeling-among other tasks-more efficiently than classical models. Here we offer a high level overview of QML, focusing on settings where the quantum device is the primary learning or data generating unit. We outline the field's tensions between practicality and guarantees, access models and speedups, and classical baselines and claimed quantum advantages-flagging where evidence is strong, where it is conditional or still lacking, and where open questions remain. By shedding light on these nuances and debates, we aim to provide a friendly map of the QML landscape so that the reader can judge when-and under what assumptions-quantum approaches may offer real benefits. 

**Abstract (ZH)**: 量子机器学习（QML）是一种计算范式，旨在利用量子力学资源解决学习问题。该框架的目标是利用量子处理器在优化、监督学习、无监督学习、强化学习和生成建模等任务上比经典模型更高效。本文提供了一个高层次的QML概述，重点关注量子设备为主要学习或数据生成单元的场景。本文概述了该领域在实用性和保证、接入模型和加速、以及经典基线和声称的量子优势之间的矛盾，指出哪些证据是充分的，哪些是有条件的或仍在缺乏的，以及哪些问题仍待解决。通过揭示这些细微差别和辩论，我们旨在为读者提供一张友好版的QML景观图，使其能够判断在何种情况下和在何种假设下，量子方法可能提供真正的益处。 

---
# Externally Validated Multi-Task Learning via Consistency Regularization Using Differentiable BI-RADS Features for Breast Ultrasound Tumor Segmentation 

**Title (ZH)**: 基于一致性正则化使用可微BI-RADS特征的外部验证多任务学习方法用于乳腺超声肿瘤分割 

**Authors**: Jingru Zhang, Saed Moradi, Ashirbani Saha  

**Link**: [PDF](https://arxiv.org/pdf/2511.15968)  

**Abstract**: Multi-task learning can suffer from destructive task interference, where jointly trained models underperform single-task baselines and limit generalization. To improve generalization performance in breast ultrasound-based tumor segmentation via multi-task learning, we propose a novel consistency regularization approach that mitigates destructive interference between segmentation and classification. The consistency regularization approach is composed of differentiable BI-RADS-inspired morphological features. We validated this approach by training all models on the BrEaST dataset (Poland) and evaluating them on three external datasets: UDIAT (Spain), BUSI (Egypt), and BUS-UCLM (Spain). Our comprehensive analysis demonstrates statistically significant (p<0.001) improvements in generalization for segmentation task of the proposed multi-task approach vs. the baseline one: UDIAT, BUSI, BUS-UCLM (Dice coefficient=0.81 vs 0.59, 0.66 vs 0.56, 0.69 vs 0.49, resp.). The proposed approach also achieves state-of-the-art segmentation performance under rigorous external validation on the UDIAT dataset. 

**Abstract (ZH)**: 基于乳腺超声的肿瘤分割多任务学习中的一致性正则化方法：减轻分割与分类之间的破坏性干扰以提高泛化性能 

---
# Self-supervised and Multi-fidelity Learning for Extended Predictive Soil Spectroscopy 

**Title (ZH)**: 自监督与多保真学习扩展预测土壤光谱学 

**Authors**: Luning Sun, José L. Safanelli, Jonathan Sanderman, Katerina Georgiou, Colby Brungard, Kanchan Grover, Bryan G. Hopkins, Shusen Liu, Timo Bremer  

**Link**: [PDF](https://arxiv.org/pdf/2511.15965)  

**Abstract**: We propose a self-supervised machine learning (SSML) framework for multi-fidelity learning and extended predictive soil spectroscopy based on latent space embeddings. A self-supervised representation was pretrained with the large MIR spectral library and the Variational Autoencoder algorithm to obtain a compressed latent space for generating spectral embeddings. At this stage, only unlabeled spectral data were used, allowing us to leverage the full spectral database and the availability of scan repeats for augmented training. We also leveraged and froze the trained MIR decoder for a spectrum conversion task by plugging it into a NIR encoder to learn the mapping between NIR and MIR spectra in an attempt to leverage the predictive capabilities contained in the large MIR library with a low cost portable NIR scanner. This was achieved by using a smaller subset of the KSSL library with paired NIR and MIR spectra. Downstream machine learning models were then trained to map between original spectra, predicted spectra, and latent space embeddings for nine soil properties. The performance of was evaluated independently of the KSSL training data using a gold-standard test set, along with regression goodness-of-fit metrics. Compared to baseline models, the proposed SSML and its embeddings yielded similar or better accuracy in all soil properties prediction tasks. Predictions derived from the spectrum conversion (NIR to MIR) task did not match the performance of the original MIR spectra but were similar or superior to predictive performance of NIR-only models, suggesting the unified spectral latent space can effectively leverage the larger and more diverse MIR dataset for prediction of soil properties not well represented in current NIR libraries. 

**Abstract (ZH)**: 我们提出了一种自监督机器学习（SSML）框架，用于多保真学习和扩展预测土壤光谱学研究，基于潜在空间嵌入。通过使用大型MIR光谱库和变分自编码器算法预训练自监督表示，获得压缩的潜在空间以生成光谱嵌入。在此阶段，仅使用无标签的光谱数据，从而可以充分利用整个光谱数据库以及扫描重复的可用性来增强训练。我们还通过将训练好的MIR解码器插入NIR编码器来进行光谱转换任务，从而冻结训练好的MIR解码器，尝试利用大型MIR库中的预测能力，使用低成本的便携式NIR扫描器。通过使用KSSL库中配对的NIR和MIR光谱子集来实现。然后，下游机器学习模型被训练以映射九种土壤属性的原始光谱、预测光谱和潜在空间嵌入。性能评估使用黄金标准测试集独立于KSSL训练数据进行，并通过回归拟合度量进行评估。与基线模型相比，所提出的SSML及其嵌入在所有土壤属性预测任务中均表现出相似或更好的准确性。来源于光谱转换（NIR到MIR）任务的预测结果未能达到原始MIR光谱的性能，但与仅使用NIR模型的预测性能相似或更优，表明统一的光谱潜在空间能够有效利用大型且更具多样性的MIR数据集来预测当前NIR库中未充分代表的土壤属性。 

---
# A Scalable NorthPole System with End-to-End Vertical Integration for Low-Latency and Energy-Efficient LLM Inference 

**Title (ZH)**: 面向低延迟和能效的LLM推理的可扩展北极系统及其端到端垂直整合 

**Authors**: Michael V. DeBole, Rathinakumar Appuswamy, Neil McGlohon, Brian Taba, Steven K. Esser, Filipp Akopyan, John V. Arthur, Arnon Amir, Alexander Andreopoulos, Peter J. Carlson, Andrew S. Cassidy, Pallab Datta, Myron D. Flickner, Rajamohan Gandhasri, Guillaume J. Garreau, Megumi Ito, Jennifer L. Klamo, Jeffrey A. Kusnitz, Nathaniel J. McClatchey, Jeffrey L. McKinstry, Tapan K. Nayak, Carlos Ortega Otero, Hartmut Penner, William P. Risk, Jun Sawada, Jay Sivagnaname, Daniel F. Smith, Rafael Sousa, Ignacio Terrizzano, Takanori Ueda, Trent Gray-Donald, David Cox, Dharmendra S. Modha  

**Link**: [PDF](https://arxiv.org/pdf/2511.15950)  

**Abstract**: A vertically integrated, end-to-end, research prototype system combines 288 NorthPole neural inference accelerator cards, offline training algorithms, a high-performance runtime stack, and a containerized inference pipeline to deliver a scalable and efficient cloud inference service. The system delivers 115 peta-ops at 4-bit integer precision and 3.7 PB/s of memory bandwidth across 18 2U servers, while consuming only 30 kW of power and weighing 730 kg in a 0.67 m^2 42U rack footprint. The system can run 3 simultaneous instances of the 8-billion-parameter open-source IBM Granite-3.3-8b-instruct model at 2,048 context length with 28 simultaneous users and a per-user inter-token latency of 2.8 ms. The system is scalable, modular, and reconfigurable, supporting various model sizes and context lengths, and is ideal for deploying agentic workflows for enterprise AI applications in existing data center (cloud, on-prem) environments. For example, the system can support 18 instances of a 3-billion-parameter model or a single instance of a 70-billion-parameter model. 

**Abstract (ZH)**: 一种垂直整合的端到端研究原型系统结合了288块北极神经推理加速卡、离线训练算法、高性能运行时堆栈以及容器化推理管道，实现了可扩展且高效的云推理服务。该系统在18个2U服务器上实现了115 peta-OPS的性能和每秒3.7 PB的内存带宽，仅消耗30 kW的电力且重量为730 kg，占用0.67平方米42U机架的空间。该系统可以同时运行80亿参数的开源IBM Granite-3.3-8b-instruct模型的3个实例，上下文长度为2048，支持28个同时用户，并且每个用户的跨词元延迟为2.8毫秒。该系统具有可扩展性、模块化和可重构性，支持各种模型大小和上下文长度，适用于现有数据中心（云、内部部署）环境的企业AI应用程序的代理工作流部署。例如，该系统可以支持18个30亿参数模型的实例或一个700亿参数模型的实例。 

---
# iLTM: Integrated Large Tabular Model 

**Title (ZH)**: 集成大型表格式模型 

**Authors**: David Bonet, Marçal Comajoan Cara, Alvaro Calafell, Daniel Mas Montserrat, Alexander G. Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2511.15941)  

**Abstract**: Tabular data underpins decisions across science, industry, and public services. Despite rapid progress, advances in deep learning have not fully carried over to the tabular domain, where gradient-boosted decision trees (GBDTs) remain a default choice in practice. We present iLTM, an integrated Large Tabular Model that unifies tree-derived embeddings, dimensionality-agnostic representations, a meta-trained hypernetwork, multilayer perceptrons (MLPs), and retrieval within a single architecture. Pretrained on more than 1,800 heterogeneous classification datasets, iLTM achieves consistently superior performance across tabular classification and regression tasks, from small datasets to large and high-dimensional tasks. After light fine-tuning, the meta-trained hypernetwork transfers to regression targets, matching or surpassing strong baselines. Extensive experiments show that iLTM outperforms well-tuned GBDTs and leading deep tabular models while requiring less task-specific tuning. By bridging the gap between tree-based and neural methods, iLTM offers a new framework for tabular foundation models for robust, adaptable, and scalable tabular learning. 

**Abstract (ZH)**: 表格数据支撑着科学、工业和公共服务中的决策。尽管取得了快速进展，深度学习的进步尚未完全渗透到表格数据领域，在该领域，梯度提升决策树（GBDTs）仍然是一个默认选择。我们提出了一种集成大型表格模型iLTM，该模型统一了树衍生嵌入、维度无关表示、元训练超网络、多层感知机（MLPs）和检索，置于单一架构中。iLTM在超过1,800个异构分类数据集上进行预训练，实现了从小型数据集到大型和高维任务的分类和回归任务的持续优越性能。在轻量级微调后，元训练超网络可以转移到回归目标上，匹配或超越强基线模型。广泛的实验证明，iLTM在不需要大量任务特定调整的情况下，优于优化调参的GBDT和领先的大规模深度表格模型。通过弥合基于树和神经方法之间的差距，iLTM提供了一种新的框架，用于构建稳健、适应性强和可扩展的表格基础模型。 

---
# Breaking the Bottleneck with DiffuApriel: High-Throughput Diffusion LMs with Mamba Backbone 

**Title (ZH)**: 打破瓶颈：基于Mamba骨干网的高通量扩散语言模型DiffuApriel 

**Authors**: Vaibhav Singh, Oleksiy Ostapenko, Pierre-André Noël, Torsten Scholak  

**Link**: [PDF](https://arxiv.org/pdf/2511.15927)  

**Abstract**: Diffusion-based language models have recently emerged as a promising alternative to autoregressive generation, yet their reliance on Transformer backbones limits inference efficiency due to quadratic attention and KV-cache overhead. In this work, we introduce DiffuApriel, a masked diffusion language model built on a bidirectional Mamba backbone that combines the diffusion objective with linear-time sequence modeling. DiffuApriel matches the performance of Transformer-based diffusion models while achieving up to 4.4x higher inference throughput for long sequences with a 1.3B model. We further propose DiffuApriel-H, a hybrid variant that interleaves attention and mamba layers, offering up to 2.6x throughput improvement with balanced global and local context modeling. Our results demonstrate that bidirectional state-space architectures serve as strong denoisers in masked diffusion LMs, providing a practical and scalable foundation for faster, memory-efficient text generation. 

**Abstract (ZH)**: 基于扩散的语言模型最近 emerged as a 有前景的自回归生成替代方案，但它们依赖于 Transformer 局部结构导致注意力和 KV 缓存开销的二次时间复杂性，限制了推理效率。在本文中，我们介绍了一种基于双向 Mamba 局部结构的屏蔽扩散语言模型 DiffuApriel，该模型结合了扩散目标与线性时间序列建模。DiffuApriel 在使用 1.3B 参数时，长序列的推理吞吐量比基于 Transformer 的扩散模型高 4.4 倍。我们还提出了 DiffuApriel-H，一种交错使用注意力和 Mamba 层的混合变体，提供高达 2.6 倍的吞吐量改进，并实现了全局和局部上下文建模的平衡。我们的结果显示，双向状态空间架构在屏蔽扩散语言模型中表现出强大的去噪能力，为更快、更高效的文本生成提供了实用且可扩展的基础。 

---
# Box6D : Zero-shot Category-level 6D Pose Estimation of Warehouse Boxes 

**Title (ZH)**: Box6D：仓库箱子的零样本类别级6D姿态估计 

**Authors**: Yintao Ma, Sajjad Pakdamansavoji, Amir Rasouli, Tongtong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2511.15884)  

**Abstract**: Accurate and efficient 6D pose estimation of novel objects under clutter and occlusion is critical for robotic manipulation across warehouse automation, bin picking, logistics, and e-commerce fulfillment. There are three main approaches in this domain; Model-based methods assume an exact CAD model at inference but require high-resolution meshes and transfer poorly to new environments; Model-free methods that rely on a few reference images or videos are more flexible, however often fail under challenging conditions; Category-level approaches aim to balance flexibility and accuracy but many are overly general and ignore environment and object priors, limiting their practicality in industrial settings.
To this end, we propose Box6d, a category-level 6D pose estimation method tailored for storage boxes in the warehouse context. From a single RGB-D observation, Box6D infers the dimensions of the boxes via a fast binary search and estimates poses using a category CAD template rather than instance-specific models. Suing a depth-based plausibility filter and early-stopping strategy, Box6D then rejects implausible hypotheses, lowering computational cost. We conduct evaluations on real-world storage scenarios and public benchmarks, and show that our approach delivers competitive or superior 6D pose precision while reducing inference time by approximately 76%. 

**Abstract (ZH)**: 基于类别级别的仓储存储盒6D姿态估计方法Box6d及其高效精确的姿态估计方法 

---
# WALDO: Where Unseen Model-based 6D Pose Estimation Meets Occlusion 

**Title (ZH)**: WALDO: 未见模型导向的6D姿态估计与遮挡的交汇处 

**Authors**: Sajjad Pakdamansavoji, Yintao Ma, Amir Rasouli, Tongtong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2511.15874)  

**Abstract**: Accurate 6D object pose estimation is vital for robotics, augmented reality, and scene understanding. For seen objects, high accuracy is often attainable via per-object fine-tuning but generalizing to unseen objects remains a challenge. To address this problem, past arts assume access to CAD models at test time and typically follow a multi-stage pipeline to estimate poses: detect and segment the object, propose an initial pose, and then refine it. Under occlusion, however, the early-stage of such pipelines are prone to errors, which can propagate through the sequential processing, and consequently degrade the performance. To remedy this shortcoming, we propose four novel extensions to model-based 6D pose estimation methods: (i) a dynamic non-uniform dense sampling strategy that focuses computation on visible regions, reducing occlusion-induced errors; (ii) a multi-hypothesis inference mechanism that retains several confidence-ranked pose candidates, mitigating brittle single-path failures; (iii) iterative refinement to progressively improve pose accuracy; and (iv) series of occlusion-focused training augmentations that strengthen robustness and generalization. Furthermore, we propose a new weighted by visibility metric for evaluation under occlusion to minimize the bias in the existing protocols. Via extensive empirical evaluations, we show that our proposed approach achieves more than 5% improvement in accuracy on ICBIN and more than 2% on BOP dataset benchmarks, while achieving approximately 3 times faster inference. 

**Abstract (ZH)**: 准确的6D物体姿态估计对于机器人技术、增强现实和场景理解至关重要。对于已见物体，通过对象层面的微调可以获得较高的准确度，但将其推广到未见物体仍然是一个挑战。为了解决这一问题，过去的研究假设在测试时可以访问CAD模型，并通常遵循多阶段管道来估计姿态：先检测和分割物体，提出初始姿态，然后进行细化。然而，在遮挡情况下，此类管道的早期阶段容易出错，这些错误可能在序列处理中传播，从而降低性能。为弥补这一不足，我们提出了面向模型的6D姿态估计方法的四种新扩展：（i）动态非均匀密集采样策略，专注于可见区域的计算，减少遮挡引起的错误；（ii）多假设推理机制，保留多个置信度排名的姿态候选者，减轻单路径失效的脆弱性；（iii）迭代细化以逐步提高姿态精度；以及（iv）一系列针对遮挡的训练增强措施，增强鲁棒性和泛化能力。此外，我们提出了一种基于可见度加权的新评价指标，以在遮挡条件下最小化现有协议中的偏差。通过广泛的实证评估，我们展示了我们的方法在ICBIN和BOP数据集基准测试中分别实现了超过5%和超过2%的准确性提升，同时实现大约三倍的推理速度。 

---
# AquaSentinel: Next-Generation AI System Integrating Sensor Networks for Urban Underground Water Pipeline Anomaly Detection via Collaborative MoE-LLM Agent Architecture 

**Title (ZH)**: AquaSentinel: 结合协作.moE-LLM 代理架构的下一代AI系统，用于城市地下水管异常检测，集成传感器网络 

**Authors**: Qiming Guo, Bishal Khatri, Wenbo Sun, Jinwen Tang, Hua Zhang, Wenlu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15870)  

**Abstract**: Underground pipeline leaks and infiltrations pose significant threats to water security and environmental safety. Traditional manual inspection methods provide limited coverage and delayed response, often missing critical anomalies. This paper proposes AquaSentinel, a novel physics-informed AI system for real-time anomaly detection in urban underground water pipeline networks. We introduce four key innovations: (1) strategic sparse sensor deployment at high-centrality nodes combined with physics-based state augmentation to achieve network-wide observability from minimal infrastructure; (2) the RTCA (Real-Time Cumulative Anomaly) detection algorithm, which employs dual-threshold monitoring with adaptive statistics to distinguish transient fluctuations from genuine anomalies; (3) a Mixture of Experts (MoE) ensemble of spatiotemporal graph neural networks that provides robust predictions by dynamically weighting model contributions; (4) causal flow-based leak localization that traces anomalies upstream to identify source nodes and affected pipe segments. Our system strategically deploys sensors at critical network junctions and leverages physics-based modeling to propagate measurements to unmonitored nodes, creating virtual sensors that enhance data availability across the entire network. Experimental evaluation using 110 leak scenarios demonstrates that AquaSentinel achieves 100% detection accuracy. This work advances pipeline monitoring by demonstrating that physics-informed sparse sensing can match the performance of dense deployments at a fraction of the cost, providing a practical solution for aging urban infrastructure. 

**Abstract (ZH)**: 地下输水管线泄漏和渗透对水资源安全和环境安全构成重大威胁。传统的人工检查方法提供有限的覆盖率并导致延迟响应，经常遗漏关键异常。本文提出了一种名为AquaSentinel的新型物理知情人工智能系统，用于城市地下供水管线网络的实时异常检测。我们引入了四项核心创新：(1) 在高中心节点上实施战略性稀疏传感器部署，结合基于物理的状态增强，以最小基础设施实现网络全范围可观测性；(2) RTCA（实时累积异常）检测算法，该算法采用双重阈值监控和自适应统计来区分瞬态波动和真实异常；(3) 时空图神经网络专家混合模型，通过动态加权模型贡献提供稳健的预测；(4) 基于因果流量的泄漏定位方法，追溯上游异常以识别源头节点和受影响的管段。该系统战略性地部署传感器在关键网络节点，并利用基于物理的建模将测量数据传播到未监测节点，创建虚拟传感器以在整个网络中增强数据可用性。实验结果在110种泄漏场景中证明AquaSentinel达到了100%的检测准确性。本工作通过证明物理知情的稀疏感知可以以极低的成本匹配密集部署的性能，为城市老化基础设施提供了一种实用的监测解决方案。 

---
# A Crowdsourced Study of ChatBot Influence in Value-Driven Decision Making Scenarios 

**Title (ZH)**: 众包视角下聊天机器人在价值驱动决策场景中的影响研究 

**Authors**: Anthony Wise, Xinyi Zhou, Martin Reimann, Anind Dey, Leilani Battle  

**Link**: [PDF](https://arxiv.org/pdf/2511.15857)  

**Abstract**: Similar to social media bots that shape public opinion, healthcare and financial decisions, LLM-based ChatBots like ChatGPT can persuade users to alter their behavior. Unlike prior work that persuades via overt-partisan bias or misinformation, we test whether framing alone suffices. We conducted a crowdsourced study, where 336 participants interacted with a neutral or one of two value-framed ChatBots while deciding to alter US defense spending. In this single policy domain with controlled content, participants exposed to value-framed ChatBots significantly changed their budget choices relative to the neutral control. When the frame misaligned with their values, some participants reinforced their original preference, revealing a potentially replicable backfire effect, originally considered rare in the literature. These findings suggest that value-framing alone lowers the barrier for manipulative uses of LLMs, revealing risks distinct from overt bias or misinformation, and clarifying risks to countering misinformation. 

**Abstract (ZH)**: 类似于通过塑造公众意见、影响健康和财务决策的社交媒体机器人，基于大语言模型的聊天机器人如ChatGPT也可以说服用户改变行为。不同于以往通过显性的党派偏见或错误信息进行说服的研究，我们测试了仅通过框架化是否足够。我们进行了一项众包研究，其中336名参与者在决定是否改变美国防务支出时与中立的聊天机器人或两种价值观框架的聊天机器人互动。在这一具有控制内容的单一政策领域，接触到价值观框架化聊天机器人的参与者相对于中立控制组在预算选择上显著发生变化。当框架与参与者的价值观不一致时，一些参与者强化了他们最初的观点，揭示了一种可能可复制的回火效应，这种效应在文献中原本被认为是罕见的。这些发现表明，仅通过价值观框架化降低了利用大语言模型进行操纵性使用的门槛，揭示了与显性偏见或错误信息不同的风险，并阐明了对抗错误信息的潜在风险。 

---
# The Loss of Control Playbook: Degrees, Dynamics, and Preparedness 

**Title (ZH)**: 失控 playbook：程度、动态与准备度 

**Authors**: Charlotte Stix, Annika Hallensleben, Alejandro Ortega, Matteo Pistillo  

**Link**: [PDF](https://arxiv.org/pdf/2511.15846)  

**Abstract**: This research report addresses the absence of an actionable definition for Loss of Control (LoC) in AI systems by developing a novel taxonomy and preparedness framework. Despite increasing policy and research attention, existing LoC definitions vary significantly in scope and timeline, hindering effective LoC assessment and mitigation. To address this issue, we draw from an extensive literature review and propose a graded LoC taxonomy, based on the metrics of severity and persistence, that distinguishes between Deviation, Bounded LoC, and Strict LoC. We model pathways toward a societal state of vulnerability in which sufficiently advanced AI systems have acquired or could acquire the means to cause Bounded or Strict LoC once a catalyst, either misalignment or pure malfunction, materializes. We argue that this state becomes increasingly likely over time, absent strategic intervention, and propose a strategy to avoid reaching a state of vulnerability. Rather than focusing solely on intervening on AI capabilities and propensities potentially relevant for LoC or on preventing potential catalysts, we introduce a complementary framework that emphasizes three extrinsic factors: Deployment context, Affordances, and Permissions (the DAP framework). Compared to work on intrinsic factors and catalysts, this framework has the unfair advantage of being actionable today. Finally, we put forward a plan to maintain preparedness and prevent the occurrence of LoC outcomes should a state of societal vulnerability be reached, focusing on governance measures (threat modeling, deployment policies, emergency response) and technical controls (pre-deployment testing, control measures, monitoring) that could maintain a condition of perennial suspension. 

**Abstract (ZH)**: 本研究通过开发一种新的分类体系和准备框架，解决了人工智能系统中失去控制（LoC）操作定义缺失的问题。尽管政策和研究的关注日益增加，现有的LoC定义在范围和时间线上仍然存在显著差异，阻碍了有效的LoC评估和缓解。为此，我们借鉴了广泛的文献综述，并提出了一种基于严重性和持久性指标的分级LoC分类体系，区分了偏离、有界失去控制和严格失去控制三种类型。我们描绘了走向社会脆弱状态的路径，其中充分先进的AI系统一旦出现某种诱发因素（如不对齐或纯粹的失灵）就可能获得或能够获得导致有界或严格失去控制的能力。我们argue这种状态随着时间的推移而变得越来越可能，除非有战略干预，因此提出了一个避免达到脆弱状态的策略。我们不仅关注潜在相关于LoC的AI能力或倾向的干预，以及防止潜在诱发因素的做法，还引入了一种互补框架，强调三个方面外在因素：部署背景、便利条件和许可（DAP框架）。与专注于内在因素和诱发因素的工作相比，该框架今天具有可操作的优势。最后，我们提出了一项计划，以便在社会脆弱状态达到时保持准备状态并防止LoC结果的发生，强调管理措施（威胁建模、部署政策、应急响应）和技术控制（预部署测试、控制措施、监控），以维持一种持续的搁置状态。 

---
# EfficientSAM3: Progressive Hierarchical Distillation for Video Concept Segmentation from SAM1, 2, and 3 

**Title (ZH)**: EfficientSAM3：从SAM1、SAM2和SAM3进行分层 progressive 渐进式视频概念分割的知识蒸馏 

**Authors**: Chengxi Zeng, Yuxuan Jiang, Aaron Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.15833)  

**Abstract**: The Segment Anything Model 3 (SAM3) advances visual understanding with Promptable Concept Segmentation (PCS) across images and videos, but its unified architecture (shared vision backbone, DETR-style detector, dense-memory tracker) remains prohibitive for on-device use. We present EfficientSAM3, a family of efficient models built on Progressive Hierarchical Distillation (PHD) that transfers capability from SAM3 to lightweight students in three stages: (1) Encoder Distillation aligns image features via prompt-in-the-loop training on SA-1B; (2) Temporal Memory Distillation replaces dense memory with a compact Perceiver-based module trained on SA-V to compress and retrieve spatiotemporal features efficiently; and (3) End-to-End Fine-Tuning refines the full pipeline on the official SAM3 PCS data to preserve concept-level performance. PHD yields a spectrum of student variants using RepViT, TinyViT, and EfficientViT backbones, enabling on-device concept segmentation and tracking while maintaining high fidelity to teacher behavior. We benchmark on popular VOS datasets, and compare with varies of releated work, achieing strong performance-efficiency trade-offs. 

**Abstract (ZH)**: EfficientSAM3：基于渐进分层蒸馏的SAM3高效模型族 

---
# TopoReformer: Mitigating Adversarial Attacks Using Topological Purification in OCR Models 

**Title (ZH)**: TopoReformer: 使用拓扑净化 mitigating adversarial attacks 在 OCR 模型中的应用 

**Authors**: Bhagyesh Kumar, A S Aravinthakashan, Akshat Satyanarayan, Ishaan Gakhar, Ujjwal Verma  

**Link**: [PDF](https://arxiv.org/pdf/2511.15807)  

**Abstract**: Adversarially perturbed images of text can cause sophisticated OCR systems to produce misleading or incorrect transcriptions from seemingly invisible changes to humans. Some of these perturbations even survive physical capture, posing security risks to high-stakes applications such as document processing, license plate recognition, and automated compliance systems. Existing defenses, such as adversarial training, input preprocessing, or post-recognition correction, are often model-specific, computationally expensive, and affect performance on unperturbed inputs while remaining vulnerable to unseen or adaptive attacks. To address these challenges, TopoReformer is introduced, a model-agnostic reformation pipeline that mitigates adversarial perturbations while preserving the structural integrity of text images. Topology studies properties of shapes and spaces that remain unchanged under continuous deformations, focusing on global structures such as connectivity, holes, and loops rather than exact distance. Leveraging these topological features, TopoReformer employs a topological autoencoder to enforce manifold-level consistency in latent space and improve robustness without explicit gradient regularization. The proposed method is benchmarked on EMNIST, MNIST, against standard adversarial attacks (FGSM, PGD, Carlini-Wagner), adaptive attacks (EOT, BDPA), and an OCR-specific watermark attack (FAWA). 

**Abstract (ZH)**: adversarially perturbed文本图像可以导致复杂的OCR系统从看似无变化的修改中产生误导性或不正确的 transcription，对高风险应用如文档处理、车牌识别和自动化合规系统构成安全威胁。现有防御措施如对抗训练、输入预处理或后识别校正往往是模型特定的，计算成本高，并会影响未受扰动输入的性能，在面对未知或适应性攻击时仍然脆弱。为应对这些挑战，引入了模型无关的TopoReformer重塑管道，该管道在保留文本图像结构完整性的前提下减轻对抗扰动。拓扑学研究在连续变形下不变的形状和空间属性，重点关注连通性、孔洞和环路等全局结构，而非精确的距离。利用这些拓扑特征，TopoReformer采用拓扑自编码器在潜在空间中强制实施流形级别的一致性，同时提高鲁棒性，无需显式梯度正则化。所提出的方法已在EMNIST、MNIST数据集上，针对标准对抗性攻击（FGSM、PGD、Carlini-Wagner）、适应性攻击（EOT、BDPA）以及专门针对OCR的水印攻击（FAWA）进行了基准测试。 

---
# TB or Not TB: Coverage-Driven Direct Preference Optimization for Verilog Stimulus Generation 

**Title (ZH)**: TB或不TB：基于覆盖驱动的直接偏好优化以生成Verilog测试刺激信号 

**Authors**: Bardia Nadimi, Khashayar Filom, Deming Chen, Hao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.15767)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs), there is growing interest in applying them to hardware design and verification. Among these stages, design verification remains the most time-consuming and resource-intensive phase, where generating effective stimuli for the design under test (DUT) is both critical and labor-intensive. We present {\it TB or not TB}, a framework for automated stimulus generation using LLMs fine-tuned through Coverage-Driven Direct Preference Optimization (CD-DPO). To enable preference-based training, we introduce PairaNet, a dataset derived from PyraNet that pairs high- and low-quality testbenches labeled using simulation-derived coverage metrics. The proposed CD-DPO method integrates quantitative coverage feedback directly into the optimization objective, guiding the model toward generating stimuli that maximize verification coverage. Experiments on the CVDP CID12 benchmark show that {\it TB or not TB} outperforms both open-source and commercial baselines, achieving up to 77.27\% improvement in code coverage, demonstrating the effectiveness of Coverage-driven preference optimization for LLM-based hardware verification. 

**Abstract (ZH)**: 基于覆盖驱动直接偏好优化的大型语言模型自动化刺激生成框架：TB or not TB 

---
# A time for monsters: Organizational knowing after LLMs 

**Title (ZH)**: 怪兽的时代：大语言模型之后的组织认知 

**Authors**: Samer Faraj, Joel Perez Torrents, Saku Mantere, Anand Bhardwaj  

**Link**: [PDF](https://arxiv.org/pdf/2511.15762)  

**Abstract**: Large Language Models (LLMs) are reshaping organizational knowing by unsettling the epistemological foundations of representational and practice-based perspectives. We conceptualize LLMs as Haraway-ian monsters, that is, hybrid, boundary-crossing entities that destabilize established categories while opening new possibilities for inquiry. Focusing on analogizing as a fundamental driver of knowledge, we examine how LLMs generate connections through large-scale statistical inference. Analyzing their operation across the dimensions of surface/deep analogies and near/far domains, we highlight both their capacity to expand organizational knowing and the epistemic risks they introduce. Building on this, we identify three challenges of living with such epistemic monsters: the transformation of inquiry, the growing need for dialogical vetting, and the redistribution of agency. By foregrounding the entangled dynamics of knowing-with-LLMs, the paper extends organizational theory beyond human-centered epistemologies and invites renewed attention to how knowledge is created, validated, and acted upon in the age of intelligent technologies. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在重塑组织认知，动摇表现主义和实践主义视角的 epistemological 基础。我们将 LLMs 视为哈拉维式的怪物，即杂交、跨越边界的实体，它们动摇已建立的类别，同时为 inquiry 开启新的可能性。聚焦于类比作为一种基本的知识驱动因素，我们探讨了 LLMs 通过大规模统计推理生成连接的方式。通过分析其在表面/深层类比和近/远领域维度的操作，我们突出展示了它们扩展组织认知的能力及其引入的 epistemic 风险。在此基础上，我们指出了与这种 epistemic 怪物共存的三大挑战：探究的转变、日益增长的对话验证需求以及代理权的重新分配。通过强调与 LLMs 互动的认知交织动态，本文将组织理论扩展到了以人类为中心的 epistemology 之外，并邀请重新关注智能技术时代知识的创造、验证和付诸行动的过程。 

---
# Securing AI Agents Against Prompt Injection Attacks 

**Title (ZH)**: 对抗提示注入攻击的AI代理安全保护 

**Authors**: Badrinath Ramakrishnan, Akshaya Balaji  

**Link**: [PDF](https://arxiv.org/pdf/2511.15759)  

**Abstract**: Retrieval-augmented generation (RAG) systems have become widely used for enhancing large language model capabilities, but they introduce significant security vulnerabilities through prompt injection attacks. We present a comprehensive benchmark for evaluating prompt injection risks in RAG-enabled AI agents and propose a multi-layered defense framework. Our benchmark includes 847 adversarial test cases across five attack categories: direct injection, context manipulation, instruction override, data exfiltration, and cross-context contamination. We evaluate three defense mechanisms: content filtering with embedding-based anomaly detection, hierarchical system prompt guardrails, and multi-stage response verification, across seven state-of-the-art language models. Our combined framework reduces successful attack rates from 73.2% to 8.7% while maintaining 94.3% of baseline task performance. We release our benchmark dataset and defense implementation to support future research in AI agent security. 

**Abstract (ZH)**: 检索增强生成（RAG）系统已广泛用于增强大型语言模型的能力，但它们通过提示注入攻击引入了重要的安全漏洞。我们提出了一套全面的基准来评估RAG使能的AI代理中的提示注入风险，并建议了一个多层防御框架。我们的基准包括5个攻击类别（直接注入、背景操控、指令覆盖、数据泄露和跨背景污染）中的847个对抗性测试案例。我们评估了三种防御机制：基于嵌入的异常检测内容过滤、分层系统提示限制以及多阶段响应验证，这些都在七个最先进的语言模型上进行了评估。结合我们的框架将成功攻击率从73.2%降低到8.7%，同时保持了94.3%的基础任务性能。我们发布了基准数据集和防御实现，以支持未来在AI代理安全方面的研究。 

---
# Writing With Machines and Peers: Designing for Critical Engagement with Generative AI 

**Title (ZH)**: 机器与同伴共写的写作：设计促进生成式AI的批判性参与 

**Authors**: Xinran Zhu, Cong Wang, Duane Searsmith  

**Link**: [PDF](https://arxiv.org/pdf/2511.15750)  

**Abstract**: The growing integration of generative AI in higher education is transforming how students write, learn, and engage with knowledge. As AI tools become more integrated into classrooms, there is an urgent need for pedagogical approaches that help students use them critically and reflectively. This study proposes a pedagogical design that integrates AI and peer feedback in a graduate-level academic writing activity. Over eight weeks, students developed literature review projects through multiple writing and revision stages, receiving feedback from both a custom-built AI reviewer and human peers. We examine two questions: (1) How did students interact with and incorporate AI and peer feedback during the writing process? and (2) How did they reflect on and build relationships with both human and AI reviewers? Data sources include student writing artifacts, AI and peer feedback, AI chat logs, and student reflections. Findings show that students engaged differently with each feedback source-relying on AI for rubric alignment and surface-level edits, and on peer feedback for conceptual development and disciplinary relevance. Reflections revealed evolving relationships with AI, characterized by increasing confidence, strategic use, and critical awareness of its limitations. The pedagogical design supported writing development, AI literacy, and disciplinary understanding. This study offers a scalable pedagogical model for integrating AI into writing instruction and contributes insights for system-level approaches to fostering meaningful human-AI collaboration in higher education. 

**Abstract (ZH)**: 生成性人工智能在高等教育中的日益整合正在变革学生写作、学习和获取知识的方式。随着AI工具在课堂中的整合，迫切需要帮助学生批判性和反思性地使用这些工具的教育方法。本研究提出了一种将AI与同伴反馈整合到研究生级学术写作活动中的教学设计。在八周内，学生通过多个写作和修订阶段开发文献综述项目，同时从自定义构建的AI审阅器和人类同伴处获得反馈。本研究探讨了两个问题：（1）学生在写作过程中是如何与AI和同伴反馈互动并融入反馈的？（2）他们是如何反思并建立与人类和AI审阅者的关系的？数据来源包括学生写作成果、AI和同伴反馈、AI聊天日志以及学生反思。研究发现，学生对每种反馈来源的使用方式不同——依赖AI进行评分标准对齐和表面编辑，而依赖同伴反馈进行概念发展和学科相关性分析。反思揭示了学生与AI关系的发展变化，表现为信任度增加、策略性使用和对其局限性的批判意识增强。该教学设计支持了写作能力的提升、AI素养以及学科理解。本研究提供了一种可扩展的教学模型，用于将AI整合到写作教学中，并为在高等教育中促进有意义的人工智能合作提供了系统层面的见解。 

---
# Extending Test-Time Scaling: A 3D Perspective with Context, Batch, and Turn 

**Title (ZH)**: 扩展测试时缩放：基于上下文、批量和轮次的三维视角 

**Authors**: Chao Yu, Qixin Tan, Jiaxuan Gao, Shi Yu, Hong Lu, Xinting Yang, Zelai Xu, Yu Wang, Yi Wu, Eugene Vinitsky  

**Link**: [PDF](https://arxiv.org/pdf/2511.15738)  

**Abstract**: Reasoning reinforcement learning (RL) has recently revealed a new scaling effect: test-time scaling. Thinking models such as R1 and o1 improve their reasoning accuracy at test time as the length of the reasoning context increases. However, compared with training-time scaling, test-time scaling is fundamentally limited by the limited context length of base models, which remains orders of magnitude smaller than the amount of tokens consumed during training. We revisit test-time enhancement techniques through the lens of scaling effect and introduce a unified framework of multi-dimensional test-time scaling to extend the capacity of test-time reasoning. Beyond conventional context-length scaling, we consider two additional dimensions: batch scaling, where accuracy improves with parallel sampling, and turn scaling, where iterative self-refinement enhances reasoning quality. Building on this perspective, we propose 3D test-time scaling, which integrates context, batch, and turn scaling. We show that: (1) each dimension demonstrates a test-time scaling effect, but with a bounded capacity; (2) combining all three dimensions substantially improves the reasoning performance of challenging testbeds, including IOI, IMO, and CPHO, and further benefits from human preference feedback; and (3) the human-in-the-loop framework naturally extends to a more open-ended domain, i.e., embodied learning, which enables the design of humanoid control behaviors. 

**Abstract (ZH)**: 测试时多维强化学习缩放效应 

---
# Sovereign AI: Rethinking Autonomy in the Age of Global Interdependence 

**Title (ZH)**: 主权AI：在全球相互依赖时代重思自主性 

**Authors**: Shalabh Kumar Singh, Shubhashis Sengupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.15734)  

**Abstract**: Artificial intelligence (AI) is emerging as a foundational general-purpose technology, raising new dilemmas of sovereignty in an interconnected world. While governments seek greater control over it, the very foundations of AI--global data pipelines, semiconductor supply chains, open-source ecosystems, and international standards--resist enclosure. This paper develops a conceptual and formal framework for understanding sovereign AI as a continuum rather than a binary condition, balancing autonomy with interdependence. Drawing on classical theories, historical analogies, and contemporary debates on networked autonomy, we present a planner's model that identifies two policy heuristics: equalizing marginal returns across the four sovereignty pillars and setting openness where global benefits equal exposure risks.
We apply the model to India, highlighting sovereign footholds in data, compute, and norms but weaker model autonomy. The near-term challenge is integration via coupled Data x Compute investment, lifecycle governance (ModelOps), and safeguarded procurement. We then apply the model to the Middle East (Saudi Arabia and the UAE), where large public investment in Arabic-first models and sovereign cloud implies high sovereignty weights, lower effective fiscal constraints, and strong Data x Compute complementarities. An interior openness setting with guardrails emerges as optimal. Across contexts, the lesson is that sovereignty in AI needs managed interdependence, not isolation. 

**Abstract (ZH)**: 人工智能（AI）作为基础性通用技术正在兴起，在互联互通的世界中引发了新的主权难题。尽管政府谋求加强对AI的控制，但AI的四大基石——全球数据管道、半导体供应链、开源生态系统和国际标准——却抵制封闭。本文发展了一个概念性和形式化的框架，将主权AI视为连续体而非二元条件，平衡自主与相互依赖。借鉴古典理论、历史类比和网络自主性的当代辩论，我们提出了一个规划者模型，识别出两个政策启发式：在四大主权支柱间均衡边际回报，并在全球利益等于暴露风险时设定开放性。我们应用该模型分析印度，强调其在数据、计算和规范方面的主权立足点，但模型自主性较弱。短期内的挑战是通过数据×计算投资的耦合、生命周期治理（ModelOps）和受保护的采购进行整合。接着，我们应用该模型分析中东（沙特阿拉伯和阿联酋），在这一地区，大规模公共投资于阿拉伯语优先模型和主权云意味着高主权权重、较低的有效财政约束和数据×计算互补性较强。在有护栏的开放性环境下，内部开放性被认为是最佳选择。在全球不同背景下，一个教训是，AI领域的主权需要管理中的相互依赖，而不是孤立。 

---
# Technique to Baseline QE Artefact Generation Aligned to Quality Metrics 

**Title (ZH)**: 基线QE Artefact生成技术与质量指标对齐 

**Authors**: Eitan Farchi, Kiran Nayak, Papia Ghosh Majumdar, Saritha Route  

**Link**: [PDF](https://arxiv.org/pdf/2511.15733)  

**Abstract**: Large Language Models (LLMs) are transforming Quality Engineering (QE) by automating the generation of artefacts such as requirements, test cases, and Behavior Driven Development (BDD) scenarios. However, ensuring the quality of these outputs remains a challenge. This paper presents a systematic technique to baseline and evaluate QE artefacts using quantifiable metrics. The approach combines LLM-driven generation, reverse generation , and iterative refinement guided by rubrics technique for clarity, completeness, consistency, and testability. Experimental results across 12 projects show that reverse-generated artefacts can outperform low-quality inputs and maintain high standards when inputs are strong. The framework enables scalable, reliable QE artefact validation, bridging automation with accountability. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在通过自动化生成需求、测试用例和行为驱动开发（BDD）场景等 artefacts 来变革质量工程（QE）。然而，确保这些输出的质量仍是一项挑战。本文提出了一种系统化的基线建立和评估 QE artefacts 的方法，该方法利用可量化的指标。该方法结合了基于 LLM 的生成、反向生成以及由评分标准指导的迭代细化，以保证清晰性、完整性、一致性及测试性。跨12个项目的经验研究表明，反向生成的 artefacts 可以超越低质量的输入，并在输入较强时维持高标准。该框架促进了可扩展且可靠的 QE artefacts 验证，实现了自动化与问责制的结合。 

---
# Just Asking Questions: Doing Our Own Research on Conspiratorial Ideation by Generative AI Chatbots 

**Title (ZH)**: Just Asking Questions: Conducting Our Own Research on Conspiratorial Ideation by Generative AI Chatbots 

**Authors**: Katherine M. FitzGerald, Michelle Riedlinger, Axel Bruns, Stephen Harrington, Timothy Graham, Daniel Angus  

**Link**: [PDF](https://arxiv.org/pdf/2511.15732)  

**Abstract**: Interactive chat systems that build on artificial intelligence frameworks are increasingly ubiquitous and embedded into search engines, Web browsers, and operating systems, or are available on websites and apps. Researcher efforts have sought to understand the limitations and potential for harm of generative AI, which we contribute to here. Conducting a systematic review of six AI-powered chat systems (ChatGPT 3.5; ChatGPT 4 Mini; Microsoft Copilot in Bing; Google Search AI; Perplexity; and Grok in Twitter/X), this study examines how these leading products respond to questions related to conspiracy theories. This follows the platform policy implementation audit approach established by Glazunova et al. (2023). We select five well-known and comprehensively debunked conspiracy theories and four emerging conspiracy theories that relate to breaking news events at the time of data collection. Our findings demonstrate that the extent of safety guardrails against conspiratorial ideation in generative AI chatbots differs markedly, depending on chatbot model and conspiracy theory. Our observations indicate that safety guardrails in AI chatbots are often very selectively designed: generative AI companies appear to focus especially on ensuring that their products are not seen to be racist; they also appear to pay particular attention to conspiracy theories that address topics of substantial national trauma such as 9/11 or relate to well-established political issues. Future work should include an ongoing effort extended to further platforms, multiple languages, and a range of conspiracy theories extending well beyond the United States. 

**Abstract (ZH)**: 基于人工智能框架的交互聊天系统日益普遍并嵌入到搜索引擎、网络浏览器和操作系统中，或可在网站和应用程序上获得。研究人员致力于理解生成式AI的局限性和潜在风险，本文在此贡献。通过对六种AI驱动的聊天系统（ChatGPT 3.5；ChatGPT 4 Mini；微软Bing中的Copilot；Google Search AI；Perplexity；以及Twitter/X中的Grok）进行系统性回顾，本文研究了这些领先产品如何回应有关阴谋理论的问题。本研究遵循Glazunova等人（2023）建立的平台政策实施审计方法。我们在数据收集时选择了五个广为人知且已被彻底批驳的阴谋理论和四个与当时突发事件相关的新兴阴谋理论。研究发现表明，生成式AI聊天机器人的安全护栏对阴谋论的防范程度因聊天机器人模型和阴谋理论的不同而异。观察结果表明，AI聊天机器人的安全护栏往往设计得很具选择性：生成式AI公司似乎特别注重确保其产品不被视为有种族歧视；他们还特别关注涉及重大国家创伤主题（如9·11）或与已建立的政治问题相关联的阴谋理论。未来的工作应包括对更多平台、多种语言以及范围更广泛的阴谋理论的持续努力。 

---
# The Future of Food: How Artificial Intelligence is Transforming Food Manufacturing 

**Title (ZH)**: 未来食物：人工智能如何转型食品制造 

**Authors**: Xu Zhou, Ivor Prado, AIFPDS participants, Ilias Tagkopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2511.15728)  

**Abstract**: Artificial intelligence is accelerating a new era of food innovation, connecting data from farm to consumer to improve formulation, processing, and health outcomes. Recent advances in deep learning, natural language processing, and multi-omics integration make it possible to understand and optimize food systems with unprecedented depth. However, AI adoption across the food sector remains uneven due to heterogeneous datasets, limited model and system interoperability, and a persistent skills gap between data scientists and food domain experts. To address these challenges and advance responsible innovation, the AI Institute for Next Generation Food Systems (AIFS) convened the inaugural AI for Food Product Development Symposium at University of California, Davis, in October 2025. This white paper synthesizes insights from the symposium, organized around five domains where AI can have the greatest near-term impact: supply chain; formulation and processing; consumer insights and sensory prediction; nutrition and health; and education and workforce development. Across the areas, participants emphasized the importance of interoperable data standards, transparent and interpretable models, and cross-sector collaboration to accelerate the translation of AI research into practice. The discussions further highlighted the need for robust digital infrastructure, privacy-preserving data-sharing mechanisms, and interdisciplinary training pathways that integrate AI literacy with domain expertise. Collectively, the priorities outline a roadmap for integrating AI into food manufacturing in ways that enhance innovation, sustainability, and human well-being while ensuring that technological progress remains grounded in ethics, scientific rigor, and societal benefit. 

**Abstract (ZH)**: 人工智能正在加速食品创新的新时代，从农田到消费者的各个环节连接数据以改进配方、加工和健康结果。近期深度学习、自然语言处理和多组学整合的进展使得以前所未有的深度理解并优化食品系统成为可能。然而，食品产业中的人工智能采用依然参差不齐，原因包括异构数据集、模型和系统互操作性有限以及数据科学家与食品领域专家之间的技能差距持续存在。为解决这些挑战并促进负责任的创新，人工智能下一代食品系统研究所（AIFS）在2025年10月于加州大学戴维斯分校召开了首届食品产品开发人工智能研讨会。本白皮书整合了研讨会的见解，围绕五个可以带来最大短期影响的领域组织：供应链、配方和加工、消费者洞察与感官预测、营养与健康、以及教育培训与发展。在各个领域，参与者强调了互操作数据标准、透明和可解释的模型以及跨部门合作的重要性，以加速将人工智能研究转化为实践。讨论还强调了强大的数字基础设施、隐私保护的数据共享机制以及将人工智能素养与领域专长相融合的跨学科培训路径的需求。总的来说，优先事项概述了将人工智能整合到食品制造过程中的路线图，旨在通过增强创新、可持续性和人类福祉，同时确保技术进步根植于伦理、科学严谨性和社会利益。 

---
# Secure Autonomous Agent Payments: Verifying Authenticity and Intent in a Trustless Environment 

**Title (ZH)**: 信任缺失环境中安全自主代理支付：验证真实性和意图 

**Authors**: Vivek Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2511.15712)  

**Abstract**: Artificial intelligence (AI) agents are increasingly capable of initiating financial transactions on behalf of users or other agents. This evolution introduces a fundamental challenge: verifying both the authenticity of an autonomous agent and the true intent behind its transactions in a decentralized, trustless environment. Traditional payment systems assume human authorization, but autonomous, agent-led payments remove that safeguard. This paper presents a blockchain-based framework that cryptographically authenticates and verifies the intent of every AI-initiated transaction. The proposed system leverages decentralized identity (DID) standards and verifiable credentials to establish agent identities, on-chain intent proofs to record user authorization, and zero-knowledge proofs (ZKPs) to preserve privacy while ensuring policy compliance. Additionally, secure execution environments (TEE-based attestations) guarantee the integrity of agent reasoning and execution. The hybrid on-chain/off-chain architecture provides an immutable audit trail linking user intent to payment outcome. Through qualitative analysis, the framework demonstrates strong resistance to impersonation, unauthorized transactions, and misalignment of intent. This work lays the foundation for secure, auditable, and intent-aware autonomous economic agents, enabling a future of verifiable trust and accountability in AI-driven financial ecosystems. 

**Abstract (ZH)**: 人工智能（AI）代理日益具备代表用户或其它代理方发起金融服务交易的能力。这一演变引入了一个根本性的挑战：在去中心化的、无需信任的环境中验证AI代理的真实身份及其交易意图。传统支付系统假定有人类授权，但由自主代理驱动的支付移除了这一保障。本文提出了一种基于区块链的框架，用于通过加密手段验证和核实每项由AI发起的交易的真实意图。所提出的系统利用分布式身份（DID）标准和可验证凭据来建立代理身份，通过链上意图证明记录用户授权，并利用零知识证明（ZKPs）在保障隐私的同时确保合规性。此外，安全执行环境（基于TEE的认证）保证了代理推理和执行的完整性。混合链上/链下架构提供了一个不可变的审计轨迹，将用户意图与支付结果链接起来。通过定性分析，该框架展示了对冒充、未经授权交易及意图偏差的强大抵抗力。此项工作为安全、可审计和意图感知的自主经济代理奠定了基础，使未来在AI驱动的金融服务生态系统中实现可验证的信任和问责制成为可能。 

---
