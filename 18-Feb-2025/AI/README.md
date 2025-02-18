# Small Models Struggle to Learn from Strong Reasoners 

**Title (ZH)**: 小模型难以从强推理者学习 

**Authors**: Yuetai Li, Xiang Yue, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Bill Yuchen Lin, Bhaskar Ramasubramanian, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2502.12143)  

**Abstract**: Large language models (LLMs) excel in complex reasoning tasks, and distilling their reasoning capabilities into smaller models has shown promise. However, we uncover an interesting phenomenon, which we term the Small Model Learnability Gap: small models ($\leq$3B parameters) do not consistently benefit from long chain-of-thought (CoT) reasoning or distillation from larger models. Instead, they perform better when fine-tuned on shorter, simpler reasoning chains that better align with their intrinsic learning capacity. To address this, we propose Mix Distillation, a simple yet effective strategy that balances reasoning complexity by combining long and short CoT examples or reasoning from both larger and smaller models. Our experiments demonstrate that Mix Distillation significantly improves small model reasoning performance compared to training on either data alone. These findings highlight the limitations of direct strong model distillation and underscore the importance of adapting reasoning complexity for effective reasoning capability transfer. 

**Abstract (ZH)**: 小型模型可学习间隙：小型模型（≤3B参数）不一致地受益于长链式思考推理或从大模型中提炼的能力。相反，它们在与自身固有能力更好地对齐的简短、简单的推理链上进行微调时表现出更好的性能。为此，我们提出了一种简单的有效策略——混合提炼（Mix Distillation），该策略通过结合长和短链式思考推理示例或从大小不同的模型中提取的推理来平衡推理复杂性。我们的实验表明，与仅使用数据训练相比，混合提炼显著提高了小型模型的推理性能。这些发现突显了直接强模型提炼的局限性，并强调了适应推理复杂性的重要性，以实现有效的推理能力转移。 

---
# Transformer Dynamics: A neuroscientific approach to interpretability of large language models 

**Title (ZH)**: Transformer 动态研究：一种神经科学方法解释大规模语言模型的可解释性 

**Authors**: Jesseba Fernando, Grigori Guitchounts  

**Link**: [PDF](https://arxiv.org/pdf/2502.12131)  

**Abstract**: As artificial intelligence models have exploded in scale and capability, understanding of their internal mechanisms remains a critical challenge. Inspired by the success of dynamical systems approaches in neuroscience, here we propose a novel framework for studying computations in deep learning systems. We focus on the residual stream (RS) in transformer models, conceptualizing it as a dynamical system evolving across layers. We find that activations of individual RS units exhibit strong continuity across layers, despite the RS being a non-privileged basis. Activations in the RS accelerate and grow denser over layers, while individual units trace unstable periodic orbits. In reduced-dimensional spaces, the RS follows a curved trajectory with attractor-like dynamics in the lower layers. These insights bridge dynamical systems theory and mechanistic interpretability, establishing a foundation for a "neuroscience of AI" that combines theoretical rigor with large-scale data analysis to advance our understanding of modern neural networks. 

**Abstract (ZH)**: 随着人工智能模型在规模和能力上迅速扩张，对其内部机制的理解仍是一项关键挑战。受神经科学中动力系统方法成功应用的启发，我们提出了一种新的框架来研究深度学习系统的计算过程。我们重点关注变压器模型中的剩余流（RS），将其视为在各层之间演化的一种动力系统。我们发现，个体RS单元的激活在整个层间表现出强烈的连续性，尽管RS并不是一个特权基。在层间，RS中的激活加速并变得越来越密集，而个体单元则遵循不稳定的周期轨道。在降维空间中，RS在较低层表现出类似吸引子的动力学，沿着一条弯曲的路径。这些见解将动力系统理论与机械解释学相结合，为一种结合理论严谨性和大规模数据分析的“人工智能神经科学”奠定了基础，以增进我们对现代神经网络的理解。 

---
# Scaling Autonomous Agents via Automatic Reward Modeling And Planning 

**Title (ZH)**: 通过自动奖励建模与规划扩展自主代理 

**Authors**: Zhenfang Chen, Delin Chen, Rui Sun, Wenjun Liu, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12130)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across a range of text-generation tasks. However, LLMs still struggle with problems requiring multi-step decision-making and environmental feedback, such as online shopping, scientific reasoning, and mathematical problem-solving. Unlike pure text data, collecting large-scale decision-making data is challenging. Moreover, many powerful LLMs are only accessible through APIs, which hinders their fine-tuning for agent tasks due to cost and complexity. To address LLM agents' limitations, we propose a framework that can automatically learn a reward model from the environment without human annotations. This model can be used to evaluate the action trajectories of LLM agents and provide heuristics for task planning. Specifically, our approach involves employing one LLM-based agent to navigate an environment randomly, generating diverse action trajectories. Subsequently, a separate LLM is leveraged to assign a task intent and synthesize a negative response alongside the correct response for each trajectory. These triplets (task intent, positive response, and negative response) are then utilized as training data to optimize a reward model capable of scoring action trajectories. The effectiveness and generalizability of our framework are demonstrated through evaluations conducted on different agent benchmarks. In conclusion, our proposed framework represents a significant advancement in enhancing LLM agents' decision-making capabilities. By automating the learning of reward models, we overcome the challenges of data scarcity and API limitations, potentially revolutionizing the application of LLMs in complex and interactive environments. This research paves the way for more sophisticated AI agents capable of tackling a wide range of real-world problems requiring multi-step decision-making. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多种文本生成任务中展现了卓越的能力。然而，LLMs 在要求多步决策和环境反馈的问题上仍存在局限，例如在线购物、科学推理和数学问题求解。与纯粹的文字数据不同，收集大规模的决策数据极具挑战性。此外，许多强大的LLMs是通过API访问的，这因其成本和复杂性而阻碍了它们在代理任务上的微调。为解决LLM代理的局限性，我们提出了一种框架，该框架可以在不依赖人工标注的情况下自动学习奖励模型。该模型可以用于评估LLM代理的动作轨迹，并为任务规划提供启发式方法。具体而言，我们的方法包括使用一个基于LLM的代理随机导航环境，生成多样化的动作轨迹。随后，利用另一个单独的LLM分配任务意图，并为每个轨迹合成了正确的响应及其负响应。这些三元组（任务意图、正响应和负响应）用作训练数据，以优化能够评估动作轨迹得分的奖励模型。通过对不同的代理基准进行评估，证明了该框架的有效性和普适性。总之，我们提出的框架在提升LLM代理的决策能力方面取得了重要进展。通过自动化奖励模型的学习，我们克服了数据稀缺和API限制的挑战，有可能变革LLM在复杂和交互式环境中的应用。这项研究为能够解决多种需要多步决策的现实世界问题的更高级AI代理铺平了道路。 

---
# Hypernym Bias: Unraveling Deep Classifier Training Dynamics through the Lens of Class Hierarchy 

**Title (ZH)**: 超类偏差：通过类层次结构的视角剖析深度分类器训练动力学 

**Authors**: Roman Malashin, Valeria Yachnaya, Alexander Mullin  

**Link**: [PDF](https://arxiv.org/pdf/2502.12125)  

**Abstract**: We investigate the training dynamics of deep classifiers by examining how hierarchical relationships between classes evolve during training. Through extensive experiments, we argue that the learning process in classification problems can be understood through the lens of label clustering. Specifically, we observe that networks tend to distinguish higher-level (hypernym) categories in the early stages of training, and learn more specific (hyponym) categories later. We introduce a novel framework to track the evolution of the feature manifold during training, revealing how the hierarchy of class relations emerges and refines across the network layers. Our analysis demonstrates that the learned representations closely align with the semantic structure of the dataset, providing a quantitative description of the clustering process. Notably, we show that in the hypernym label space, certain properties of neural collapse appear earlier than in the hyponym label space, helping to bridge the gap between the initial and terminal phases of learning. We believe our findings offer new insights into the mechanisms driving hierarchical learning in deep networks, paving the way for future advancements in understanding deep learning dynamics. 

**Abstract (ZH)**: 我们通过探究类别层次关系在训练过程中的演化来研究深度分类器的训练动态。我们通过大量实验argue，分类问题中的学习过程可以通过标签聚类的视角来理解。具体而言，我们观察到网络在训练的早期阶段倾向于区分上位类别（超nym），而在后期学习更多特定的下位类别（hyponym）。我们提出了一种新颖的框架来追踪训练过程中特征流形的演化，揭示了类别关系层级如何在网络层中逐步形成和细化。我们的分析表明，learned表示与数据集的语义结构高度一致，提供了一个聚类过程的定量描述。值得注意的是，我们展示了在上位类别标签空间中，某些神经塌缩的特性比在下位类别标签空间中更早出现，有助于弥合学习初期和终端阶段之间的差距。我们认为我们的发现为理解深度网络中的层次学习机制提供了新的见解，为未来深入理解深度学习动态铺平了道路。 

---
# Relational Norms for Human-AI Cooperation 

**Title (ZH)**: 人类与人工智能合作中的关系规范 

**Authors**: Brian D. Earp, Sebastian Porsdam Mann, Mateo Aboy, Edmond Awad, Monika Betzler, Marietjie Botes, Rachel Calcott, Mina Caraccio, Nick Chater, Mark Coeckelbergh, Mihaela Constantinescu, Hossein Dabbagh, Kate Devlin, Xiaojun Ding, Vilius Dranseika, Jim A. C. Everett, Ruiping Fan, Faisal Feroz, Kathryn B. Francis, Cindy Friedman, Orsolya Friedrich, Iason Gabriel, Ivar Hannikainen, Julie Hellmann, Arasj Khodadade Jahrome, Niranjan S. Janardhanan, Paul Jurcys, Andreas Kappes, Maryam Ali Khan, Gordon Kraft-Todd, Maximilian Kroner Dale, Simon M. Laham, Benjamin Lange, Muriel Leuenberger, Jonathan Lewis, Peng Liu, David M. Lyreskog, Matthijs Maas, John McMillan, Emilian Mihailov, Timo Minssen, Joshua Teperowski Monrad, Kathryn Muyskens, Simon Myers, Sven Nyholm, Alexa M. Owen, Anna Puzio, Christopher Register, Madeline G. Reinecke, Adam Safron, Henry Shevlin, Hayate Shimizu, Peter V. Treit, Cristina Voinea, Karen Yan, Anda Zahiu, Renwen Zhang, Hazem Zohny, Walter Sinnott-Armstrong, Ilina Singh, Julian Savulescu, Margaret S. Clark  

**Link**: [PDF](https://arxiv.org/pdf/2502.12102)  

**Abstract**: How we should design and interact with social artificial intelligence depends on the socio-relational role the AI is meant to emulate or occupy. In human society, relationships such as teacher-student, parent-child, neighbors, siblings, or employer-employee are governed by specific norms that prescribe or proscribe cooperative functions including hierarchy, care, transaction, and mating. These norms shape our judgments of what is appropriate for each partner. For example, workplace norms may allow a boss to give orders to an employee, but not vice versa, reflecting hierarchical and transactional expectations. As AI agents and chatbots powered by large language models are increasingly designed to serve roles analogous to human positions - such as assistant, mental health provider, tutor, or romantic partner - it is imperative to examine whether and how human relational norms should extend to human-AI interactions. Our analysis explores how differences between AI systems and humans, such as the absence of conscious experience and immunity to fatigue, may affect an AI's capacity to fulfill relationship-specific functions and adhere to corresponding norms. This analysis, which is a collaborative effort by philosophers, psychologists, relationship scientists, ethicists, legal experts, and AI researchers, carries important implications for AI systems design, user behavior, and regulation. While we accept that AI systems can offer significant benefits such as increased availability and consistency in certain socio-relational roles, they also risk fostering unhealthy dependencies or unrealistic expectations that could spill over into human-human relationships. We propose that understanding and thoughtfully shaping (or implementing) suitable human-AI relational norms will be crucial for ensuring that human-AI interactions are ethical, trustworthy, and favorable to human well-being. 

**Abstract (ZH)**: 社会人工智能的设计与互动应取决于其旨在模仿或占据的社会关系角色。在人类社会中，诸如师生关系、亲子关系、邻里关系、兄弟姐妹关系或雇主雇员关系等关系由特定规范所治理，这些规范规定或禁止合作功能，包括等级制度、关爱、交易和交配。这些规范塑造了我们对每个伙伴适宜行为的判断。例如，工作场所规范可能允许老板对员工下达命令，但不允许反之，这反映了等级制度和交易的期待。随着越来越多设计用于模拟人类职位的社会人工智能代理和聊天机器人——如助手、心理健康提供者、导师或伴侣——审视并探索人类关系规范是否以及如何适用于人机互动变得至关重要。我们的分析探讨了人工智能系统与人类之间的差异，如缺乏意识体验和免于疲劳，如何影响人工智能履行特定关系功能以及遵守相应规范的能力。这项分析由哲学家、心理学家、关系科学家、伦理学家、法律专家和人工智能研究人员共同努力完成，对人工智能系统设计、用户行为和监管具有重要影响。尽管我们接受人工智能系统可以提供诸如特定社会关系角色的增加可用性和一致性的显著利益，但它们也可能滋生不健康依赖或不切实际期望的风险，这些风险可能会溢入人与人之间关系。我们建议理解并有意识地塑造（或实施）适当的人机关系规范将是确保人机互动具有伦理性、可信性和有利于人类福祉的关键。 

---
# A Study on Leveraging Search and Self-Feedback for Agent Reasoning 

**Title (ZH)**: 基于搜索和自我反馈的代理推理研究 

**Authors**: Karthikeyan K, Michelle Yuan, Elman Mansimov, Katerina Margatina, Anurag Pratik, Daniele Bonadiman, Monica Sunkara, Yi Zhang, Yassine Benajiba  

**Link**: [PDF](https://arxiv.org/pdf/2502.12094)  

**Abstract**: Recent works have demonstrated that incorporating search during inference can significantly improve reasoning capabilities of language agents. Some approaches may make use of the ground truth or rely on model's own generated feedback. The search algorithm uses this feedback to then produce values that will update its criterion for exploring and exploiting various reasoning paths. In this study, we investigate how search and model's self-feedback can be leveraged for reasoning tasks. First, we explore differences in ground-truth feedback and self-feedback during search for math reasoning. Second, we observe limitations in applying search techniques to more complex tasks like tool-calling and design domain-specific approaches to address these gaps. Our experiments reveal challenges related to generalization when solely relying on self-feedback during search. For search to work effectively, either access to the ground-truth is needed or feedback mechanisms need to be carefully designed for the specific task. 

**Abstract (ZH)**: Recent works have demonstrated that incorporating search during inference can significantly improve reasoning capabilities of language agents. Some approaches may make use of the ground truth or rely on model's own generated feedback. The search algorithm uses this feedback to then produce values that will update its criterion for exploring and exploiting various reasoning paths. In this study, we investigate how search and model's self-feedback can be leveraged for reasoning tasks. First, we explore differences in ground-truth feedback and self-feedback during search for math reasoning. Second, we observe limitations in applying search techniques to more complex tasks like tool-calling and design domain-specific approaches to address these gaps. Our experiments reveal challenges related to generalization when solely relying on self-feedback during search. For search to work effectively, either access to the ground-truth is needed or feedback mechanisms need to be carefully designed for the specific task. 

---
# CONSTRUCTA: Automating Commercial Construction Schedules in Fabrication Facilities with Large Language Models 

**Title (ZH)**: CONSTRUCTA: 使用大型语言模型在生产设施中自动化建筑施工调度 

**Authors**: Yifan Zhang, Xue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12066)  

**Abstract**: Automating planning with LLMs presents transformative opportunities for traditional industries, yet remains underexplored. In commercial construction, the complexity of automated scheduling often requires manual intervention to ensure precision. We propose CONSTRUCTA, a novel framework leveraging LLMs to optimize construction schedules in complex projects like semiconductor fabrication. CONSTRUCTA addresses key challenges by: (1) integrating construction-specific knowledge through static RAG; (2) employing context-sampling techniques inspired by architectural expertise to provide relevant input; and (3) deploying Construction DPO to align schedules with expert preferences using RLHF. Experiments on proprietary data demonstrate performance improvements of +42.3% in missing value prediction, +79.1% in dependency analysis, and +28.9% in automated planning compared to baseline methods, showcasing its potential to revolutionize construction workflows and inspire domain-specific LLM advancements. 

**Abstract (ZH)**: 利用大语言模型自动化施工规划在传统行业中的应用虽具有变革性机会但仍未充分探索。CONSTRUCTA：一种利用大语言模型优化半导体 fabrication 施工计划的新型框架 

---
# PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning 

**Title (ZH)**: PhysReason: 基于物理推理的综合基准 

**Authors**: Xinyu Zhang, Yuxuan Dong, Yanrui Wu, Jiaxing Huang, Chengyou Jia, Basura Fernando, Mike Zheng Shou, Lingling Zhang, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12054)  

**Abstract**: Large language models demonstrate remarkable capabilities across various domains, especially mathematics and logic reasoning. However, current evaluations overlook physics-based reasoning - a complex task requiring physics theorems and constraints. We present PhysReason, a 1,200-problem benchmark comprising knowledge-based (25%) and reasoning-based (75%) problems, where the latter are divided into three difficulty levels (easy, medium, hard). Notably, problems require an average of 8.1 solution steps, with hard requiring 15.6, reflecting the complexity of physics-based reasoning. We propose the Physics Solution Auto Scoring Framework, incorporating efficient answer-level and comprehensive step-level evaluations. Top-performing models like Deepseek-R1, Gemini-2.0-Flash-Thinking, and o3-mini-high achieve less than 60% on answer-level evaluation, with performance dropping from knowledge questions (75.11%) to hard problems (31.95%). Through step-level evaluation, we identified four key bottlenecks: Physics Theorem Application, Physics Process Understanding, Calculation, and Physics Condition Analysis. These findings position PhysReason as a novel and comprehensive benchmark for evaluating physics-based reasoning capabilities in large language models. Our code and data will be published at https:/dxzxy12138.github.io/PhysReason. 

**Abstract (ZH)**: 大型语言模型在数学和逻辑推理等领域展现出 remarkable 的能力，尤其是物理推理。然而，当前的评估忽视了基于物理的推理——一项要求应用物理定理和约束的复杂任务。我们提出了 PhysReason，这是一个包含 1200 个问题的基准，其中知识基础问题占 25%，基于推理的问题占 75%，后者进一步分为三个难度级别（简单、中等、困难）。值得注意的是，这些问题平均需要 8.1 步解决方案，其中困难级别的问题需要 15.6 步，这反映了基于物理的推理的复杂性。我们提出了物理解决方案自动评分框架，该框架包括高效的答案级和全面的步骤级评估。顶级模型如 Deepseek-R1、Gemini-2.0-Flash-Thinking 和 o3-mini-high 在答案级评估中得分低于 60%，性能从知识问题（75.11%）下降到困难问题（31.95%）。通过步骤级评估，我们确定了四个关键瓶颈：物理定理应用、物理过程理解、计算和物理条件分析。这些发现使 PhysReason 成为评估大型语言模型物理推理能力的一个新颖且全面的基准。我们的代码和数据将在 https:/dxzxy12138.github.io/PhysReason 发布。 

---
# A Survey on Bridging EEG Signals and Generative AI: From Image and Text to Beyond 

**Title (ZH)**: EEG信号与生成式AI融合综述：从图像和文本到更广泛的领域 

**Authors**: Shreya Shukla, Jose Torres, Abhijit Mishra, Jacek Gwizdka, Shounak Roychowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2502.12048)  

**Abstract**: Integration of Brain-Computer Interfaces (BCIs) and Generative Artificial Intelligence (GenAI) has opened new frontiers in brain signal decoding, enabling assistive communication, neural representation learning, and multimodal integration. BCIs, particularly those leveraging Electroencephalography (EEG), provide a non-invasive means of translating neural activity into meaningful outputs. Recent advances in deep learning, including Generative Adversarial Networks (GANs) and Transformer-based Large Language Models (LLMs), have significantly improved EEG-based generation of images, text, and speech. This paper provides a literature review of the state-of-the-art in EEG-based multimodal generation, focusing on (i) EEG-to-image generation through GANs, Variational Autoencoders (VAEs), and Diffusion Models, and (ii) EEG-to-text generation leveraging Transformer based language models and contrastive learning methods. Additionally, we discuss the emerging domain of EEG-to-speech synthesis, an evolving multimodal frontier. We highlight key datasets, use cases, challenges, and EEG feature encoding methods that underpin generative approaches. By providing a structured overview of EEG-based generative AI, this survey aims to equip researchers and practitioners with insights to advance neural decoding, enhance assistive technologies, and expand the frontiers of brain-computer interaction. 

**Abstract (ZH)**: 脑机接口（BCIs）与生成人工智能（GenAI）整合在脑电信号解码中的新进展：基于EEG的多模态生成综述 

---
# KnowPath: Knowledge-enhanced Reasoning via LLM-generated Inference Paths over Knowledge Graphs 

**Title (ZH)**: 知路径：通过LLM生成的推理路径在知识图编辑中的知识增强推理 

**Authors**: Qi Zhao, Hongyu Yang, Qi Song, Xinwei Yao, Xiangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12029)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in various complex tasks, yet they still suffer from hallucinations. Introducing external knowledge, such as knowledge graph, can enhance the LLMs' ability to provide factual answers. LLMs have the ability to interactively explore knowledge graphs. However, most approaches have been affected by insufficient internal knowledge excavation in LLMs, limited generation of trustworthy knowledge reasoning paths, and a vague integration between internal and external knowledge. Therefore, we propose KnowPath, a knowledge-enhanced large model framework driven by the collaboration of internal and external knowledge. It relies on the internal knowledge of the LLM to guide the exploration of interpretable directed subgraphs in external knowledge graphs, better integrating the two knowledge sources for more accurate reasoning. Extensive experiments on multiple real-world datasets confirm the superiority of KnowPath. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种复杂任务中展现了卓越的能力，但仍存在幻觉问题。引入外部知识，如知识图谱，可以增强LLMs提供事实性答案的能力。LLMs具备与外部知识图谱进行交互式探索的能力。然而，大多数方法受到LLMs内部知识发掘不足、可信知识推理路径生成有限以及内部与外部知识融合不清的限制。因此，我们提出了一种由内部和外部知识协作驱动的知识增强大型模型框架KnowPath。该框架利用LLMs的内部知识指导对外部知识图谱中可解释定向子图的探索，更好地整合两种知识来源以进行更准确的推理。在多个真实世界数据集上的广泛实验验证了KnowPath的优越性。 

---
# SafeChain: Safety of Language Models with Long Chain-of-Thought Reasoning Capabilities 

**Title (ZH)**: SafeChain: 具有长链条思考推理能力的语言模型的安全性 

**Authors**: Fengqing Jiang, Zhangchen Xu, Yuetai Li, Luyao Niu, Zhen Xiang, Bo Li, Bill Yuchen Lin, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2502.12025)  

**Abstract**: Emerging large reasoning models (LRMs), such as DeepSeek-R1 models, leverage long chain-of-thought (CoT) reasoning to generate structured intermediate steps, enhancing their reasoning capabilities. However, long CoT does not inherently guarantee safe outputs, potentially leading to harmful consequences such as the introduction of security vulnerabilities in code or the spread of misinformation. Current research on large language model (LLM) safety usually focuses on short-answer responses, overlooking the long CoT style outputs of LRMs. To bridge this gap, we conduct a systematic study of LRM safety. First, we investigate safety evaluators calibrated against human annotations. Using our newly developed metrics, we thoroughly assess the safety of 12 state-of-the-art LRMs on StrongReject and WildJailbreak datasets. Our results show that LRMs are not safe compared to their reasoning advance. Further, we perform a fine-grained analysis of the reasoning trace and final answer. We find that three decoding strategies-ZeroThink, LessThink, and MoreThink-can improve model safety without additional training. However, these strategies either use constrained reasoning traces or incur high inference costs. To better strengthen LRM safety, we introduce SafeChain, the first-of-its-kind safety training dataset in CoT style. We fine-tune two LRMs with SafeChain, showing that it not only enhances model safety but also preserves performance across 6 reasoning benchmarks. 

**Abstract (ZH)**: 新兴的大推理模型（LRMs），如DeepSeek-R1模型，通过长推理链（CoT）增强其推理能力并生成结构化的中间步骤。然而，长CoT并不必然保证输出的安全性，可能会导致安全漏洞的引入或错误信息的传播。当前对大型语言模型（LLMs）安全性的研究通常集中在短答案响应上，忽视了LRMs的长CoT风格输出。为填补这一空白，我们进行了系统的LRM安全性研究。首先，我们研究了与人类注释校准的安全评估器。使用我们新开发的指标，我们全面评估了12个最先进的LRMs在StrongReject和WildJailbreak数据集上的安全性。结果显示，LRMs的安全性并未随着其推理能力的提升而提高。进一步，我们对推理轨迹和最终答案进行了细致分析。我们发现，三种解码策略——ZeroThink、LessThink和MoreThink——可以在不额外训练的情况下提高模型安全性，但这些策略要么受限于推理轨迹，要么会产生高昂的推理成本。为了更好地增强LRM的安全性，我们引入了SafeChain，这是首个用于CoT风格的安全训练数据集。我们对两个LRM进行微调，结果显示，SafeChain不仅增强了模型安全性，还跨六个推理基准保持了性能。 

---
# Learning Generalizable Prompt for CLIP with Class Similarity Knowledge 

**Title (ZH)**: 基于类相似性知识学习通用提示词向量以优化CLIP 

**Authors**: Sehun Jung, Hyang-won Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.11969)  

**Abstract**: In vision-language models (VLMs), prompt tuning has shown its effectiveness in adapting models to downstream tasks. However, learned prompts struggle to generalize to unseen classes, as they tend to overfit to the classes that are targeted during prompt tuning. Examining failure cases, we observed that learned prompts disrupt the semantics of unseen classes, generating text embeddings with incorrect semantic relationships among classes. To address this, we propose Similarity Alignment Regularization (SAR), which regularizes learnable prompts to preserve the semantic relationships among classes captured by hand-crafted prompts. Specifically, we first obtain novel classes related to base classes using ChatGPT-4o and utilize them as potential unseen classes during prompt tuning. Then, by targeting both base and novel classes, SAR aligns the similarity relationships among text embeddings generated by learnable prompts with the similarity relationships from hand-crafted prompts. Extensive experiments applying SAR to existing prompt tuning methods demonstrate its effectiveness in improving generalization to unseen classes. 

**Abstract (ZH)**: 在视觉-语言模型中，提示调优已经显示出其在适应下游任务方面的有效性。然而，学习到的提示难以泛化到未见过的类中，因为它们倾向于在提示调优过程中针对特定类进行过拟合。通过分析失败案例，我们发现学习到的提示会破坏未见过类的语义，生成具有错误语义关系的文本嵌入。为了解决这一问题，我们提出了一种相似性对齐正则化（SAR），它通过对可学习提示进行正则化来保持手工构建提示捕获的类间语义关系。具体而言，我们首先使用ChatGPT-4o获取与基类相关的新型类，并利用它们作为提示调优过程中的潜在未见过类。然后，通过同时针对基类和新型类，SAR将可学习提示生成的文本嵌入之间的相似性关系与手工构建提示的相似性关系对齐。在现有提示调优方法中应用SAR的广泛实验表明，它在提高对未见过类的泛化能力方面是有效的。 

---
# STRIVE: Structured Reasoning for Self-Improvement in Claim Verification 

**Title (ZH)**: STRIVE: 结构化推理促进断言验证的自我提升 

**Authors**: Haisong Gong, Jing Li, Junfei Wu, Qiang Liu, Shu Wu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11959)  

**Abstract**: Claim verification is the task of determining whether a claim is supported or refuted by evidence. Self-improvement methods, where reasoning chains are generated and those leading to correct results are selected for training, have succeeded in tasks like mathematical problem solving. However, in claim verification, this approach struggles. Low-quality reasoning chains may falsely match binary truth labels, introducing faulty reasoning into the self-improvement process and ultimately degrading performance. To address this, we propose STRIVE: Structured Reasoning for Self-Improved Verification. Our method introduces a structured reasoning design with Claim Decomposition, Entity Analysis, and Evidence Grounding Verification. These components improve reasoning quality, reduce errors, and provide additional supervision signals for self-improvement. STRIVE begins with a warm-up phase, where the base model is fine-tuned on a small number of annotated examples to learn the structured reasoning design. It is then applied to generate reasoning chains for all training examples, selecting only those that are correct and structurally sound for subsequent self-improvement training. We demonstrate that STRIVE achieves significant improvements over baseline models, with a 31.4% performance gain over the base model and 20.7% over Chain of Thought on the HOVER datasets, highlighting its effectiveness. 

**Abstract (ZH)**: 基于结构化推理的自我改进验证 

---
# GRAPHGPT-O: Synergistic Multimodal Comprehension and Generation on Graphs 

**Title (ZH)**: GRAPHGPT-O：图上多模态理解和生成的协同作用 

**Authors**: Yi Fang, Bowen Jin, Jiacheng Shen, Sirui Ding, Qiaoyu Tan, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.11925)  

**Abstract**: The rapid development of Multimodal Large Language Models (MLLMs) has enabled the integration of multiple modalities, including texts and images, within the large language model (LLM) framework. However, texts and images are usually interconnected, forming a multimodal attributed graph (MMAG). It is underexplored how MLLMs can incorporate the relational information (\textit{i.e.}, graph structure) and semantic information (\textit{i.e.,} texts and images) on such graphs for multimodal comprehension and generation. In this paper, we propose GraphGPT-o, which supports omni-multimodal understanding and creation on MMAGs. We first comprehensively study linearization variants to transform semantic and structural information as input for MLLMs. Then, we propose a hierarchical aligner that enables deep graph encoding, bridging the gap between MMAGs and MLLMs. Finally, we explore the inference choices, adapting MLLM to interleaved text and image generation in graph scenarios. Extensive experiments on three datasets from different domains demonstrate the effectiveness of our proposed method. Datasets and codes will be open-sourced upon acceptance. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）的快速發展-enable多模态大型语言模型（MLLMs）的快速發展-已使文本和图像等多模态能够在大型语言模型（LLM）框架内进行集成。然而，文本和图像通常相互关联，形成一个多模态属性图（MMAG）。目前尚不清楚MLLMs如何将关系信息（即，图结构）和语义信息（即，文本和图像）结合在这样的图上以进行多模态理解和生成。在本文中，我们提出了GraphGPT-o，支持在MMAGs上的全模态理解和创建。首先，我们全面研究了线性化变体以将语义和结构信息转换为MLLM的输入。然后，我们提出了一种分层对齐器，实现了深层次的图编码，从而在MMAGs和MLLMs之间建立桥梁。最后，我们探索了推理选择，使MLLM能够在图场景中适应交错的文本和图像生成。在三个不同领域的数据集上的广泛实验表明了我们提出方法的有效性。数据集和代码将在接受后开源。 

---
# On the robustness of ChatGPT in teaching Korean Mathematics 

**Title (ZH)**: 关于ChatGPT在教学韩式数学中的鲁棒性 

**Authors**: Phuong-Nam Nguyen, Quang Nguyen-The, An Vu-Minh, Diep-Anh Nguyen, Xuan-Lam Pham  

**Link**: [PDF](https://arxiv.org/pdf/2502.11915)  

**Abstract**: ChatGPT, an Artificial Intelligence model, has the potential to revolutionize education. However, its effectiveness in solving non-English questions remains uncertain. This study evaluates ChatGPT's robustness using 586 Korean mathematics questions. ChatGPT achieves 66.72% accuracy, correctly answering 391 out of 586 questions. We also assess its ability to rate mathematics questions based on eleven criteria and perform a topic analysis. Our findings show that ChatGPT's ratings align with educational theory and test-taker perspectives. While ChatGPT performs well in question classification, it struggles with non-English contexts, highlighting areas for improvement. Future research should address linguistic biases and enhance accuracy across diverse languages. Domain-specific optimizations and multilingual training could improve ChatGPT's role in personalized education. 

**Abstract (ZH)**: ChatGPT，一种人工智能模型，有望革新教育。然而，其解决非英语问题的有效性仍不确定。本研究使用586道韩文数学题评估ChatGPT的稳健性。ChatGPT的准确率为66.72%，正确回答了391道题目。我们还基于 eleven 个标准评估其评级数学题目的能力，并进行了主题分析。研究发现，ChatGPT的评级与教育理论和考生视角一致。虽然ChatGPT在题目分类方面表现良好，但在非英语环境中表现不佳，凸显了改进的必要性。未来研究应解决语言偏见并提高不同语言环境下的准确性。针对特定领域的优化和多语言训练可提高ChatGPT在个性化教育中的作用。 

---
# Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration 

**Title (ZH)**: 基于双过程理论的语言代理框架在实时人机协作中的应用 

**Authors**: Shao Zhang, Xihuai Wang, Wenhao Zhang, Chaoran Li, Junru Song, Tingyu Li, Lin Qiu, Xuezhi Cao, Xunliang Cai, Wen Yao, Weinan Zhang, Xinbing Wang, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11882)  

**Abstract**: Agents built on large language models (LLMs) have excelled in turn-by-turn human-AI collaboration but struggle with simultaneous tasks requiring real-time interaction. Latency issues and the challenge of inferring variable human strategies hinder their ability to make autonomous decisions without explicit instructions. Through experiments with current independent System 1 and System 2 methods, we validate the necessity of using Dual Process Theory (DPT) in real-time tasks. We propose DPT-Agent, a novel language agent framework that integrates System 1 and System 2 for efficient real-time simultaneous human-AI collaboration. DPT-Agent's System 1 uses a Finite-state Machine (FSM) and code-as-policy for fast, intuitive, and controllable decision-making. DPT-Agent's System 2 integrates Theory of Mind (ToM) and asynchronous reflection to infer human intentions and perform reasoning-based autonomous decisions. We demonstrate the effectiveness of DPT-Agent through further experiments with rule-based agents and human collaborators, showing significant improvements over mainstream LLM-based frameworks. To the best of our knowledge, DPT-Agent is the first language agent framework that achieves successful real-time simultaneous human-AI collaboration autonomously. Code of DPT-Agent can be found in this https URL. 

**Abstract (ZH)**: 基于大型语言模型的代理在轮流的人工智能协作中表现出色，但在需要实时交互的并行任务中挣扎。延迟问题和推断多变的人类策略的挑战阻碍了它们在没有明确指令的情况下做出自主决策的能力。通过使用当前独立的System 1和System 2方法的实验，我们验证了在实时任务中使用双重过程理论（DPT）的必要性。我们提出了DPT-Agent，一种新型的语言代理框架，将System 1和System 2集成以实现高效的实时并行人机协作。DPT-Agent的System 1使用有限状态机（FSM）和代码作为策略，实现快速、直观和可控的决策。DPT-Agent的System 2结合了心理理论（ToM）和异步反思，在推断人类意图和进行推理为基础的自主决策方面发挥作用。通过进一步的基于规则的代理和人类合作者的实验，展示了DPT-Agent的有效性，显示出比主流基于LLM的框架有显著改进。据我们所知，DPT-Agent是第一个实现真正实时并行人机协作的自主语言代理框架。DPT-Agent的代码可以在以下链接找到：this https URL。 

---
# Hypothesis-Driven Theory-of-Mind Reasoning for Large Language Models 

**Title (ZH)**: 基于假设驱动的心理理论推理的大语言模型 

**Authors**: Hyunwoo Kim, Melanie Sclar, Tan Zhi-Xuan, Lance Ying, Sydney Levine, Yang Liu, Joshua B. Tenenbaum, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2502.11881)  

**Abstract**: Existing LLM reasoning methods have shown impressive capabilities across various tasks, such as solving math and coding problems. However, applying these methods to scenarios without ground-truth answers or rule-based verification methods - such as tracking the mental states of an agent - remains challenging. Inspired by the sequential Monte Carlo algorithm, we introduce thought-tracing, an inference-time reasoning algorithm designed to trace the mental states of specific agents by generating hypotheses and weighting them based on observations without relying on ground-truth solutions to questions in datasets. Our algorithm is modeled after the Bayesian theory-of-mind framework, using LLMs to approximate probabilistic inference over agents' evolving mental states based on their perceptions and actions. We evaluate thought-tracing on diverse theory-of-mind benchmarks, demonstrating significant performance improvements compared to baseline LLMs. Our experiments also reveal interesting behaviors of the recent reasoning models - e.g., o1 and R1 - on theory-of-mind, highlighting the difference of social reasoning compared to other domains. 

**Abstract (ZH)**: 已有的大语言模型推理方法在各类任务中展现了显著的能力，如解决数学和编程问题。然而，将这些方法应用于缺乏确切答案或基于规则验证方法的场景——例如追踪智能体的心理状态——仍然具有挑战性。受序列蒙特卡罗算法的启发，我们提出了思维追踪算法，这是一种推理时的推理算法，旨在通过生成假设并根据观察结果对这些假设进行加权，而不依赖于数据集中问题的确切答案来追踪特定智能体的心理状态。该算法借鉴了贝叶斯心因理论框架，利用大语言模型基于智能体的感知和行为对其心理状态的演变进行概率性推理。我们在此算法上对多种心理理论基准进行了评估，相比基础的大语言模型显示了显著性能提升。我们的实验还揭示了最近的推理模型（如o1和R1）在心理理论方面的一些有趣行为，突出了社交推理与其他领域的不同。 

---
# AAKT: Enhancing Knowledge Tracing with Alternate Autoregressive Modeling 

**Title (ZH)**: AAKT：增强知识追踪的交替自回归建模 

**Authors**: Hao Zhou, Wenge Rong, Jianfei Zhang, Qing Sun, Yuanxin Ouyang, Zhang Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11817)  

**Abstract**: Knowledge Tracing (KT) aims to predict students' future performances based on their former exercises and additional information in educational settings. KT has received significant attention since it facilitates personalized experiences in educational situations. Simultaneously, the autoregressive modeling on the sequence of former exercises has been proven effective for this task. One of the primary challenges in autoregressive modeling for Knowledge Tracing is effectively representing the anterior (pre-response) and posterior (post-response) states of learners across exercises. Existing methods often employ complex model architectures to update learner states using question and response records. In this study, we propose a novel perspective on knowledge tracing task by treating it as a generative process, consistent with the principles of autoregressive models. We demonstrate that knowledge states can be directly represented through autoregressive encodings on a question-response alternate sequence, where model generate the most probable representation in hidden state space by analyzing history interactions. This approach underpins our framework, termed Alternate Autoregressive Knowledge Tracing (AAKT). Additionally, we incorporate supplementary educational information, such as question-related skills, into our framework through an auxiliary task, and include extra exercise details, like response time, as additional inputs. Our proposed framework is implemented using advanced autoregressive technologies from Natural Language Generation (NLG) for both training and prediction. Empirical evaluations on four real-world KT datasets indicate that AAKT consistently outperforms all baseline models in terms of AUC, ACC, and RMSE. Furthermore, extensive ablation studies and visualized analysis validate the effectiveness of key components in AAKT. 

**Abstract (ZH)**: 知识追踪（KT）的目标是基于学生以往的练习和附加信息来预测未来的表现。知识追踪自引入以来受到了广泛关注，因为它可以促进教育情境中的个性化体验。同时，对以往练习序列的自回归建模已被证明是进行此任务的有效方法。自回归建模在知识追踪中的主要挑战之一是如何有效地表示练习前后学生的前响应状态和后响应状态。现有方法通常通过问题和响应记录来更新学生状态，采用复杂模型架构。在本研究中，我们从一种新的视角来处理知识追踪任务，将其视为生成过程，这与自回归模型的原则一致。我们展示了可以通过对问题-响应交替序列进行自回归编码直接表示知识状态，通过分析历史交互来生成最有可能的隐藏状态表示，从而支持我们提出的框架——交替自回归知识追踪（AAKT）。此外，我们通过辅助任务将补充的教育信息，如与问题相关的技能，以及额外的练习细节，如响应时间，整合到我们的框架中。提出的框架利用自然语言生成（NLG）的先进自回归技术进行训练和预测。在四个实际知识追踪数据集上的实证评估表明，AAKT在AUC、ACC和RMSE指标上均优于所有基线模型。此外，广泛的消融研究和可视化分析验证了AAKT关键组件的有效性。 

---
# Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning 

**Title (ZH)**: 表评论家：一种用于表格推理中协作批评与修正的多-agent框架 

**Authors**: Peiying Yu, Guoxin Chen, Jingjing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11799)  

**Abstract**: Despite the remarkable capabilities of large language models (LLMs) in various reasoning tasks, they still struggle with table reasoning tasks, particularly in maintaining consistency throughout multi-step reasoning processes. While existing approaches have explored various decomposition strategies, they often lack effective mechanisms to identify and correct errors in intermediate reasoning steps, leading to cascading error propagation. To address these issues, we propose Table-Critic, a novel multi-agent framework that facilitates collaborative criticism and iterative refinement of the reasoning process until convergence to correct solutions. Our framework consists of four specialized agents: a Judge for error identification, a Critic for comprehensive critiques, a Refiner for process improvement, and a Curator for pattern distillation. To effectively deal with diverse and unpredictable error types, we introduce a self-evolving template tree that systematically accumulates critique knowledge through experience-driven learning and guides future reflections. Extensive experiments have demonstrated that Table-Critic achieves substantial improvements over existing methods, achieving superior accuracy and error correction rates while maintaining computational efficiency and lower solution degradation rate. 

**Abstract (ZH)**: 尽管大语言模型在各种推理任务中表现出色，但在表推理任务中仍面临挑战，特别是在多步推理过程中保持一致性方面。虽然现有的方法探索了各种分解策略，但它们往往缺乏有效的机制来识别和纠正中间推理步骤中的错误，导致错误逐级传播。为了解决这些问题，我们提出Table-Critic，一种新型多Agent框架，促进推理过程的合作批评和迭代优化，直到收敛到正确的解决方案。该框架包含四个专门的Agent：裁判用于错误识别、批评家用于全面批评、优化器用于过程改进、策展人用于模式提炼。为有效处理多样且不可预测的错误类型，我们引入了一种自我进化的模板树，通过基于经验的学习系统地累积批评知识，并引导未来的反思。广泛的实验表明，Table-Critic 在现有方法上取得了显著改进，实现了更高的准确性和错误纠正率，同时保持了计算效率和较低的解质量降解率。 

---
# Cognitive-Aligned Document Selection for Retrieval-augmented Generation 

**Title (ZH)**: 认知对齐的文档选择用于检索增强生成 

**Authors**: Bingyu Wan, Fuxi Zhang, Zhongpeng Qi, Jiayi Ding, Jijun Li, Baoshi Fan, Yijia Zhang, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11770)  

**Abstract**: Large language models (LLMs) inherently display hallucinations since the precision of generated texts cannot be guaranteed purely by the parametric knowledge they include. Although retrieval-augmented generation (RAG) systems enhance the accuracy and reliability of generative models by incorporating external documents, these retrieved documents often fail to adequately support the model's responses in practical applications. To address this issue, we propose GGatrieval (Fine-\textbf{G}rained \textbf{G}rounded \textbf{A}lignment Re\textbf{trieval} for verifiable generation), which leverages an LLM to dynamically update queries and filter high-quality, reliable retrieval documents. Specifically, we parse the user query into its syntactic components and perform fine-grained grounded alignment with the retrieved documents. For query components that cannot be individually aligned, we propose a dynamic semantic compensation mechanism that iteratively refines and rewrites the query while continuously updating the retrieval results. This iterative process continues until the retrieved documents sufficiently support the query's response. Our approach introduces a novel criterion for filtering retrieved documents, closely emulating human strategies for acquiring targeted information. This ensures that the retrieved content effectively supports and verifies the generated outputs. On the ALCE benchmark, our method significantly surpasses a wide range of baselines, achieving state-of-the-art performance. 

**Abstract (ZH)**: 大型语言模型(LLMs)固有地表现出幻觉现象，因为生成文本的精确性不能仅由其包含的参数知识保证。尽管检索增强生成(RAG)系统通过引入外部文档来提升生成模型的准确性和可靠性，但这些检索到的文档在实际应用中往往无法充分支持模型的响应。为了解决这一问题，我们提出了一种GGatrieval（细粒度 Grounded Alignment Retrieval，用于可验证生成的精炼检索），利用LLM动态更新查询并筛选高质量、可靠的检索文档。具体来说，我们将用户查询解析为其语法成分，并与检索到的文档进行细粒度的匹配。对于无法单独匹配的查询成分，我们提出了一种动态语义补偿机制，该机制不断细化和重写查询并持续更新检索结果。这一迭代过程将持续进行，直到检索到的文档能够充分支持查询的响应。我们的方法引入了过滤检索文档的新标准，该标准紧密模仿了人类获取目标信息的策略，确保检索内容有效地支持和验证生成输出。在ALCE基准上，我们的方法显著超越了广泛的基础方法，达到了最先进的性能。 

---
# HintsOfTruth: A Multimodal Checkworthiness Detection Dataset with Real and Synthetic Claims 

**Title (ZH)**: 真实与合成断言的多模态可信度检测数据集：HintsOfTruth 

**Authors**: Michiel van der Meer, Pavel Korshunov, Sébastien Marcel, Lonneke van der Plas  

**Link**: [PDF](https://arxiv.org/pdf/2502.11753)  

**Abstract**: Misinformation can be countered with fact-checking, but the process is costly and slow. Identifying checkworthy claims is the first step, where automation can help scale fact-checkers' efforts. However, detection methods struggle with content that is 1) multimodal, 2) from diverse domains, and 3) synthetic. We introduce HintsOfTruth, a public dataset for multimodal checkworthiness detection with $27$K real-world and synthetic image/claim pairs. The mix of real and synthetic data makes this dataset unique and ideal for benchmarking detection methods. We compare fine-tuned and prompted Large Language Models (LLMs). We find that well-configured lightweight text-based encoders perform comparably to multimodal models but the first only focus on identifying non-claim-like content. Multimodal LLMs can be more accurate but come at a significant computational cost, making them impractical for large-scale applications. When faced with synthetic data, multimodal models perform more robustly 

**Abstract (ZH)**: 虚假信息可以通过事实核查来对抗，但这一过程代价高昂且耗时。确定核查worthy的声明是第一步，自动化可以帮助扩大事实核查者的努力。然而，检测方法在处理以下内容时遇到困难：1) 多模态内容，2) 来自多种领域，3) 合成内容。我们引入了HintsOfTruth，这是一个包含27000个真实世界和合成图像/声明对的公开多模态核查worthy检测数据集。真实数据和合成数据的结合使该数据集独特且适用于检测方法的基准测试。我们对比了微调和提示的大型语言模型（LLMs）。我们发现，配置良好的轻量级文本编码器在性能上可与多模态模型相媲美，但仅限于识别非声明类型的内容。多模态LLMs可以更准确，但会产生巨大的计算成本，使它们在大规模应用中不切实际。当面对合成数据时，多模态模型表现更为 robust。 

---
# Energy-Conscious LLM Decoding: Impact of Text Generation Strategies on GPU Energy Consumption 

**Title (ZH)**: 节能导向的LLM解码：文本生成策略对GPU能耗的影响 

**Authors**: Alireza Nik, Michael A. Riegler, Pål Halvorsen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11723)  

**Abstract**: Decoding strategies significantly influence the quality and diversity of the generated texts in large language models (LLMs), yet their impact on computational resource consumption, particularly GPU energy usage, is insufficiently studied. This paper investigates the relationship between text generation decoding methods and energy efficiency, focusing on the trade-off between generation quality and GPU energy consumption across diverse tasks and decoding configurations. By benchmarking multiple strategies across different text generation tasks, such as Translation, Code Summarization, and Math Problem Solving, we reveal how selecting appropriate decoding techniques with their tuned hyperparameters affects text quality and has measurable implications for resource utilization, emphasizing the need for balanced optimization. To the best of our knowledge, this study is among the first to explore decoding strategies in LLMs through the lens of energy consumption, offering actionable insights for designing resource-aware applications that maintain high-quality text generation. 

**Abstract (ZH)**: 大型语言模型（LLMs）中的解码策略显著影响生成文本的质量和多样性，但它们对计算资源消耗，特别是GPU能耗的影响研究尚不足。本文探讨了文本生成解码方法与能效之间的关系，重点关注生成质量和GPU能耗之间的权衡，涵盖多样化的任务和不同的解码配置。通过在翻译、代码摘要和数学问题解决等多种文本生成任务中对比多种策略，我们揭示了选择合适的解码技术及其调优超参数如何影响文本质量和资源利用，并强调了平衡优化的必要性。据我们所知，这是首次从能耗角度研究LLMs中的解码策略，为设计兼顾高质量文本生成的应用程序提供了可操作的见解。 

---
# VRoPE: Rotary Position Embedding for Video Large Language Models 

**Title (ZH)**: VRoPE: 旋转位置嵌入视频大型语言模型 

**Authors**: Zikang Liu, Longteng Guo, Yepeng Tang, Junxian Cai, Kai Ma, Xi Chen, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11664)  

**Abstract**: Rotary Position Embedding (RoPE) has shown strong performance in text-based Large Language Models (LLMs), but extending it to video remains a challenge due to the intricate spatiotemporal structure of video frames. Existing adaptations, such as RoPE-3D, attempt to encode spatial and temporal dimensions separately but suffer from two major limitations: positional bias in attention distribution and disruptions in video-text transitions. To overcome these issues, we propose Video Rotary Position Embedding (VRoPE), a novel positional encoding method tailored for Video-LLMs. Our approach restructures positional indices to preserve spatial coherence and ensure a smooth transition between video and text tokens. Additionally, we introduce a more balanced encoding strategy that mitigates attention biases, ensuring a more uniform distribution of spatial focus. Extensive experiments on Vicuna and Qwen2 across different model scales demonstrate that VRoPE consistently outperforms previous RoPE variants, achieving significant improvements in video understanding, temporal reasoning, and retrieval tasks. Code will be available at this https URL 

**Abstract (ZH)**: Video Rotary Position Embedding (VRoPE)：一种针对Video-LLMs的新型位置编码方法 

---
# Competing LLM Agents in a Non-Cooperative Game of Opinion Polarisation 

**Title (ZH)**: 竞猜LLM代理在意见极化非合作博弈中的竞争 

**Authors**: Amin Qasmi, Usman Naseem, Mehwish Nasim  

**Link**: [PDF](https://arxiv.org/pdf/2502.11649)  

**Abstract**: We introduce a novel non-cooperative game to analyse opinion formation and resistance, incorporating principles from social psychology such as confirmation bias, resource constraints, and influence penalties. Our simulation features Large Language Model (LLM) agents competing to influence a population, with penalties imposed for generating messages that propagate or counter misinformation. This framework integrates resource optimisation into the agents' decision-making process. Our findings demonstrate that while higher confirmation bias strengthens opinion alignment within groups, it also exacerbates overall polarisation. Conversely, lower confirmation bias leads to fragmented opinions and limited shifts in individual beliefs. Investing heavily in a high-resource debunking strategy can initially align the population with the debunking agent, but risks rapid resource depletion and diminished long-term influence. 

**Abstract (ZH)**: 我们介绍了一种新的非合作性博弈，用于分析意见形成和抵制行为，结合了社会心理学原理，如确认偏差、资源限制和影响惩罚。我们的模拟中，大型语言模型代理竞争以影响人群，对传播或抵制错误信息的代理施加惩罚。该框架将资源优化纳入代理的决策过程中。研究发现，虽然更高的确认偏差加强了团体内的意见一致性，但也加剧了总体上的极化。相反，较低的确认偏差会导致意见碎片化和个人信念有限的变化。大量投资高资源验证策略可以初步使人群与验证代理一致，但存在资源快速耗尽和长期影响减弱的风险。 

---
# A Unified Modeling Framework for Automated Penetration Testing 

**Title (ZH)**: 统一的自动化渗透测试建模框架 

**Authors**: Yunfei Wang, Shixuan Liu, Wenhao Wang, Changling Zhou, Chao Zhang, Jiandong Jin, Cheng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11588)  

**Abstract**: The integration of artificial intelligence into automated penetration testing (AutoPT) has highlighted the necessity of simulation modeling for the training of intelligent agents, due to its cost-efficiency and swift feedback capabilities. Despite the proliferation of AutoPT research, there is a recognized gap in the availability of a unified framework for simulation modeling methods. This paper presents a systematic review and synthesis of existing techniques, introducing MDCPM to categorize studies based on literature objectives, network simulation complexity, dependency of technical and tactical operations, and scenario feedback and variation. To bridge the gap in unified method for multi-dimensional and multi-level simulation modeling, dynamic environment modeling, and the scarcity of public datasets, we introduce AutoPT-Sim, a novel modeling framework that based on policy automation and encompasses the combination of all sub dimensions. AutoPT-Sim offers a comprehensive approach to modeling network environments, attackers, and defenders, transcending the constraints of static modeling and accommodating networks of diverse scales. We publicly release a generated standard network environment dataset and the code of Network Generator. By integrating publicly available datasets flexibly, support is offered for various simulation modeling levels focused on policy automation in MDCPM and the network generator help researchers output customized target network data by adjusting parameters or fine-tuning the network generator. 

**Abstract (ZH)**: 人工智能在自动化渗透测试中的集成突显了仿真建模对于智能代理培训的必要性：由于其成本效益和快速反馈能力。尽管自动化渗透测试研究不断涌现，但在仿真建模方法的统一框架方面仍存在缺口。本文对现有技术进行了系统综述和综合，提出MDCPM根据文献目标、网络仿真复杂度、技术和战术操作的依赖性以及场景反馈和变化进行分类。为解决多维度和多层次仿真建模、动态环境建模以及缺乏公开数据集的缺口，我们提出了AutoPT-Sim，这是一种基于策略自动化的新建模框架，涵盖了所有子维度的组合。AutoPT-Sim提供了 modeling 网络环境、攻击者和防御者的全面方法，超越了静态建模的限制，适用于不同规模的网络。我们公开发布了一个生成的标准网络环境数据集和网络生成器的代码。通过灵活整合公开可用的数据集，AutoPT-Sim 支持MDCPM中各类仿真建模层次关注策略自动化，并通过调整参数或细化网络生成器帮助研究人员输出定制的目标网络数据。 

---
# Calibration of Vehicular Traffic Simulation Models by Local Optimization 

**Title (ZH)**: 车辆交通仿真模型的局部优化校准 

**Authors**: Davide Andrea Guastella, Alejandro Morales-Hernàndez, Bruno Cornelis, Gianluca Bontempi  

**Link**: [PDF](https://arxiv.org/pdf/2502.11585)  

**Abstract**: Simulation is a valuable tool for traffic management experts to assist them in refining and improving transportation systems and anticipating the impact of possible changes in the infrastructure network before their actual implementation. Calibrating simulation models using traffic count data is challenging because of the complexity of the environment, the lack of data, and the uncertainties in traffic dynamics. This paper introduces a novel stochastic simulation-based traffic calibration technique. The novelty of the proposed method is: (i) it performs local traffic calibration, (ii) it allows calibrating simulated traffic in large-scale environments, (iii) it requires only the traffic count data. The local approach enables decentralizing the calibration task to reach near real-time performance, enabling the fostering of digital twins. Using only traffic count data makes the proposed method generic so that it can be applied in different traffic scenarios at various scales (from neighborhood to region). We assess the proposed technique on a model of Brussels, Belgium, using data from real traffic monitoring devices. The proposed method has been implemented using the open-source traffic simulator SUMO. Experimental results show that the traffic model calibrated using the proposed method is on average 16% more accurate than those obtained by the state-of-the-art methods, using the same dataset. We also make available the output traffic model obtained from real data. 

**Abstract (ZH)**: 基于随机模拟的交通校准新技术：局部校准方法在大规模环境下使用交通流量数据进行交通模拟校准 

---
# Large Language Models and Mathematical Reasoning Failures 

**Title (ZH)**: 大型语言模型在数学推理中的失败 

**Authors**: Johan Boye, Birger Moell  

**Link**: [PDF](https://arxiv.org/pdf/2502.11574)  

**Abstract**: This paper investigates the mathematical reasoning capabilities of large language models (LLMs) using 50 newly constructed high-school-level word problems. Unlike prior studies that focus solely on answer correctness, we rigorously analyze both final answers and solution steps to identify reasoning failures. Evaluating eight state-of-the-art models - including Mixtral, Llama, Gemini, GPT-4o, and OpenAI's o1 variants - we find that while newer models (e.g., o3-mini, deepseek-r1) achieve higher accuracy, all models exhibit errors in spatial reasoning, strategic planning, and arithmetic, sometimes producing correct answers through flawed logic. Common failure modes include unwarranted assumptions, over-reliance on numerical patterns, and difficulty translating physical intuition into mathematical steps. Manual analysis reveals that models struggle with problems requiring multi-step deduction or real-world knowledge, despite possessing broad mathematical knowledge. Our results underscore the importance of evaluating reasoning processes, not just answers, and caution against overestimating LLMs' problem-solving proficiency. The study highlights persistent gaps in LLMs' generalization abilities, emphasizing the need for targeted improvements in structured reasoning and constraint handling. 

**Abstract (ZH)**: 本文利用50个新构建的高中水平词语问题，探讨了大语言模型的数学推理能力。不同于以往仅关注答案正确性的研究，我们严格分析了最终答案和解题步骤，以识别推理错误。评估了八种最先进的模型，包括Mixtral、Llama、Gemini、GPT-4o以及OpenAI的o1变体，发现虽然较新模型（如o3-mini、deepseek-r1）在准确率上更高，但所有模型在空间推理、战略规划和算术运算方面都出现了错误，有时会通过错误的逻辑得出正确答案。常见的错误模式包括无根据的假设、过度依赖数字模式以及将物理直觉转化为数学步骤的困难。手动分析表明，尽管模型拥有广泛的数学知识，它们在需要多步推理或实际知识的问题上仍存在问题。研究结果强调了评估推理过程而非仅仅答案的重要性，并警告不要高估LLM的解决问题能力。该研究突出了LLMs在一般化能力上的持续差距，强调了在结构化推理和约束处理方面进行针对性改进的必要性。 

---
# A Survey of Automatic Prompt Engineering: An Optimization Perspective 

**Title (ZH)**: 自动提示工程综述：从优化视角 

**Authors**: Wenwu Li, Xiangfeng Wang, Wenhao Li, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2502.11560)  

**Abstract**: The rise of foundation models has shifted focus from resource-intensive fine-tuning to prompt engineering, a paradigm that steers model behavior through input design rather than weight updates. While manual prompt engineering faces limitations in scalability, adaptability, and cross-modal alignment, automated methods, spanning foundation model (FM) based optimization, evolutionary methods, gradient-based optimization, and reinforcement learning, offer promising solutions. Existing surveys, however, remain fragmented across modalities and methodologies. This paper presents the first comprehensive survey on automated prompt engineering through a unified optimization-theoretic lens. We formalize prompt optimization as a maximization problem over discrete, continuous, and hybrid prompt spaces, systematically organizing methods by their optimization variables (instructions, soft prompts, exemplars), task-specific objectives, and computational frameworks. By bridging theoretical formulation with practical implementations across text, vision, and multimodal domains, this survey establishes a foundational framework for both researchers and practitioners, while highlighting underexplored frontiers in constrained optimization and agent-oriented prompt design. 

**Abstract (ZH)**: 自动提示工程的统一优化理论综述 

---
# Equilibrate RLHF: Towards Balancing Helpfulness-Safety Trade-off in Large Language Models 

**Title (ZH)**: 平衡RLHF：在大型语言模型中实现帮助性与安全性权衡的平衡 

**Authors**: Yingshui Tan, Yilei Jiang, Yanshi Li, Jiaheng Liu, Xingyuan Bu, Wenbo Su, Xiangyu Yue, Xiaoyong Zhu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.11555)  

**Abstract**: Fine-tuning large language models (LLMs) based on human preferences, commonly achieved through reinforcement learning from human feedback (RLHF), has been effective in improving their performance. However, maintaining LLM safety throughout the fine-tuning process remains a significant challenge, as resolving conflicts between safety and helpfulness can be non-trivial. Typically, the safety alignment of LLM is trained on data with safety-related categories. However, our experiments find that naively increasing the scale of safety training data usually leads the LLMs to an ``overly safe'' state rather than a ``truly safe'' state, boosting the refusal rate through extensive safety-aligned data without genuinely understanding the requirements for safe responses. Such an approach can inadvertently diminish the models' helpfulness. To understand the phenomenon, we first investigate the role of safety data by categorizing them into three different groups, and observe that each group behaves differently as training data scales up. To boost the balance between safety and helpfulness, we propose an Equilibrate RLHF framework including a Fine-grained Data-centric (FDC) approach that achieves better safety alignment even with fewer training data, and an Adaptive Message-wise Alignment (AMA) approach, which selectively highlight the key segments through a gradient masking strategy. Extensive experimental results demonstrate that our approach significantly enhances the safety alignment of LLMs while balancing safety and helpfulness. 

**Abstract (ZH)**: 基于人类偏好的大语言模型（LLMs）微调，通常通过人类反馈强化学习（RLHF）实现，已在提升模型性能方面表现出效用。然而，在微调过程中保持LLM安全始终是一项重大挑战，因为解决安全与 helpfulness 之间的冲突并不简单。通常，LLM的安全对齐是在包含安全相关类别的数据上训练的。然而，我们的实验发现，简单地增加安全训练数据的规模通常会导致LLM进入一个“过度安全”的状态，而不是“真正安全”的状态，从而通过大量安全对齐的数据增加了拒绝率，而没有真正理解安全响应的要求。这种做法可能会无意中削弱模型的帮助性。为了理解这一现象，我们首先通过将安全数据分类为三个不同的组来研究其作用，并观察每个组在训练数据规模增加时的行为表现不同。为了提高安全与帮助性的平衡，我们提出了一种平衡RLHF框架（Equilibrate RLHF），其中包括一种细粒度数据为中心的方法（FDC），即使在较少的训练数据情况下也能实现更好的安全对齐，以及一种自适应消息层面对齐（AMA）方法，通过梯度屏蔽策略突出关键段落。广泛的实验证明，我们的方法能在提高LLM安全对齐的同时，更好地平衡安全与帮助性。 

---
# A Survey of Personalized Large Language Models: Progress and Future Directions 

**Title (ZH)**: 个性化大型语言模型综述：进展与未来方向 

**Authors**: Jiahong Liu, Zexuan Qiu, Zhongyang Li, Quanyu Dai, Jieming Zhu, Minda Hu, Menglin Yang, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2502.11528)  

**Abstract**: Large Language Models (LLMs) excel in handling general knowledge tasks, yet they struggle with user-specific personalization, such as understanding individual emotions, writing styles, and preferences. Personalized Large Language Models (PLLMs) tackle these challenges by leveraging individual user data, such as user profiles, historical dialogues, content, and interactions, to deliver responses that are contextually relevant and tailored to each user's specific needs. This is a highly valuable research topic, as PLLMs can significantly enhance user satisfaction and have broad applications in conversational agents, recommendation systems, emotion recognition, medical assistants, and more. This survey reviews recent advancements in PLLMs from three technical perspectives: prompting for personalized context (input level), finetuning for personalized adapters (model level), and alignment for personalized preferences (objective level). To provide deeper insights, we also discuss current limitations and outline several promising directions for future research. Updated information about this survey can be found at the this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在处理通用知识任务方面表现出色，但在用户特定的个性化方面存在困难，如理解个体情感、写作风格和偏好。个性化大规模语言模型（PLLMs）通过利用个体用户数据，如用户资料、历史对话、内容和互动，来提供与上下文相关且符合每位用户特定需求的回应。这是一个极具价值的研究话题，因为PLLMs能够显著提升用户体验，并在对话代理、推荐系统、情绪识别、医疗助手等领域具有广泛的应用前景。本文从三个技术视角回顾了PLLMs的最新进展：个性化上下文提示（输入层面）、个性化适配器微调（模型层面）和个性化偏好对齐（目标层面）。为了提供更深入的见解，我们也讨论了当前的局限性，并概述了几条具有前景的研究方向。有关本综述的更新信息，请访问 <https://www.example.com>。 

---
# Why Vision Language Models Struggle with Visual Arithmetic? Towards Enhanced Chart and Geometry Understanding 

**Title (ZH)**: 视觉语言模型为何难以处理视觉算术？朝向增强的图表和几何理解 

**Authors**: Kung-Hsiang Huang, Can Qin, Haoyi Qiu, Philippe Laban, Shafiq Joty, Caiming Xiong, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11492)  

**Abstract**: Vision Language Models (VLMs) have achieved remarkable progress in multimodal tasks, yet they often struggle with visual arithmetic, seemingly simple capabilities like object counting or length comparison, which are essential for relevant complex tasks like chart understanding and geometric reasoning. In this work, we first investigate the root causes of this deficiency through a suite of probing tasks focusing on basic visual arithmetic. Our analysis reveals that while pre-trained vision encoders typically capture sufficient information, the text decoder often fails to decode it correctly for arithmetic reasoning. To address this, we propose CogAlign, a novel post-training strategy inspired by Piaget's theory of cognitive development. CogAlign trains VLMs to recognize invariant properties under visual transformations. We demonstrate that this approach significantly improves the performance of three diverse VLMs on our proposed probing tasks. Furthermore, CogAlign enhances performance by an average of 4.6% on CHOCOLATE and 2.9% on MATH-VISION, outperforming or matching supervised fine-tuning methods while requiring only 60% less training data. These results highlight the effectiveness and generalizability of CogAlign in improving fundamental visual arithmetic capabilities and their transfer to downstream tasks. 

**Abstract (ZH)**: Vision Language Models (VLMs)在多模态任务中取得了显著进展，但在视觉算术、诸如对象计数或长度比较等看似简单的能力方面常常表现不佳，这些能力对于复杂的任务如图表理解和几何推理至关重要。在本文中，我们首先通过一系列专注于基本视觉算术的探针任务来探究这种缺陷的根本原因。我们的分析表明，尽管预训练的视觉编码器通常能够捕获足够的信息，但文本解码器往往无法正确解码其用于算术推理。为了解决这一问题，我们提出了一种名为CogAlign的新型后训练策略，该策略受到了皮亚杰认知发展理论的启发。CogAlign训练VLMs识别视觉变换下的不变属性。我们证明，这种方法在我们提出的探针任务中显著提高了三种不同VLMs的性能。此外，CogAlign在CHOCOLATE上平均提高了4.6%，在MATH-VISION上平均提高了2.9%，且仅需较少的训练数据，就能超越或匹配有监督微调方法，突显了CogAlign在提高基本视觉算术能力和其向下游任务迁移方面的有效性和普适性。 

---
# AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection 

**Title (ZH)**: AGrail: 一种有效的自适应安全检测终生智能体守护rails 

**Authors**: Weidi Luo, Shenghong Dai, Xiaogeng Liu, Suman Banerjee, Huan Sun, Muhao Chen, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2502.11448)  

**Abstract**: The rapid advancements in Large Language Models (LLMs) have enabled their deployment as autonomous agents for handling complex tasks in dynamic environments. These LLMs demonstrate strong problem-solving capabilities and adaptability to multifaceted scenarios. However, their use as agents also introduces significant risks, including task-specific risks, which are identified by the agent administrator based on the specific task requirements and constraints, and systemic risks, which stem from vulnerabilities in their design or interactions, potentially compromising confidentiality, integrity, or availability (CIA) of information and triggering security risks. Existing defense agencies fail to adaptively and effectively mitigate these risks. In this paper, we propose AGrail, a lifelong agent guardrail to enhance LLM agent safety, which features adaptive safety check generation, effective safety check optimization, and tool compatibility and flexibility. Extensive experiments demonstrate that AGrail not only achieves strong performance against task-specific and system risks but also exhibits transferability across different LLM agents' tasks. 

**Abstract (ZH)**: 大型语言模型的迅速 advancement 使其实现作为自主代理在动态环境中处理复杂任务的部署。这些大型语言模型展示了强大的问题解决能力并能够适应多种多样的情境。然而，作为代理使用也带来了显著的风险，包括任务特定风险和系统风险。任务特定风险由代理管理员根据特定任务需求和约束识别，系统风险则源于设计或交互中的漏洞，可能危及信息的机密性、完整性和可用性（CIA），从而引发安全风险。现有的防御机构无法适应性且有效地下降这些风险。在本文中，我们提出 AGrail，一种终身代理护栏，旨在增强大型语言模型代理的安全性，其特点是适应性的安全性检查生成、有效安全性检查优化以及工具的兼容性和灵活性。广泛的实验结果表明，AGrail 不仅在应对任务特定风险和系统风险方面表现出色，而且还展示了在不同大型语言模型任务之间的可迁移性。 

---
# SMART: Self-Aware Agent for Tool Overuse Mitigation 

**Title (ZH)**: SMART：自我意识代理工具滥用缓解 

**Authors**: Cheng Qian, Emre Can Acikgoz, Hongru Wang, Xiusi Chen, Avirup Sil, Dilek Hakkani-Tür, Gokhan Tur, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.11435)  

**Abstract**: Current Large Language Model (LLM) agents demonstrate strong reasoning and tool use capabilities, but often lack self-awareness, failing to balance these approaches effectively. This imbalance leads to Tool Overuse, where models unnecessarily rely on external tools for tasks solvable with parametric knowledge, increasing computational overhead. Inspired by human metacognition, we introduce SMART (Strategic Model-Aware Reasoning with Tools), a paradigm that enhances an agent's self-awareness to optimize task handling and reduce tool overuse. To support this paradigm, we introduce SMART-ER, a dataset spanning three domains, where reasoning alternates between parametric knowledge and tool-dependent steps, with each step enriched by rationales explaining when tools are necessary. Through supervised training, we develop SMARTAgent, a family of models that dynamically balance parametric knowledge and tool use. Evaluations show that SMARTAgent reduces tool use by 24% while improving performance by over 37%, enabling 7B-scale models to match its 70B counterpart and GPT-4o. Additionally, SMARTAgent generalizes to out-of-distribution test data like GSM8K and MINTQA, maintaining accuracy with just one-fifth the tool calls. These highlight the potential of strategic tool use to enhance reasoning, mitigate overuse, and bridge the gap between model size and performance, advancing intelligent and resource-efficient agent designs. 

**Abstract (ZH)**: 当前大型语言模型（LLM）代理展现了强大的推理和工具使用能力，但往往缺乏自我意识，无法有效平衡这些方法。这种不平衡导致了工具过度使用，模型在可以通过参数化知识解决的任务中无必要地依赖外部工具，增加了计算开销。受到人类元认知的启发，我们引入了SMART（Strategic Model-Aware Reasoning with Tools）范式，增强代理的自我意识以优化任务处理并减少工具过度使用。为支持这一范式，我们引入了SMART-ER数据集，该数据集覆盖了三个领域，在推理过程中交替使用参数化知识和工具依赖步骤，并且每一步都通过解释何时需要工具的说明来丰富。通过监督训练，我们开发了SMARTAgent模型家族，能够动态平衡参数化知识和工具使用。评估结果显示，SMARTAgent在减少了24%工具使用的同时改进了性能超过37%，使7B规模的模型能够与70B规模的模型和GPT-4o相媲美。此外，SMARTAgent能够泛化到如GSM8K和MINTQA等分布外测试数据中，仅使用五分之一的工具调用就能保持准确率。这些结果突显了战略性工具使用在增强推理、减轻过度使用以及缩小模型规模与性能差距方面的潜力，促进了智能和资源高效代理设计的进步。 

---
# \textsc{FLAG-Trader}: Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading 

**Title (ZH)**: FLAG-Trader: 基于梯度强化学习的LLM-Agents融合模型在金融交易中的应用 

**Authors**: Guojun Xiong, Zhiyang Deng, Keyi Wang, Yupeng Cao, Haohang Li, Yangyang Yu, Xueqing Peng, Mingquan Lin, Kaleb E Smith, Xiao-Yang Liu, Jimin Huang, Sophia Ananiadou, Qianqian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.11433)  

**Abstract**: Large language models (LLMs) fine-tuned on multimodal financial data have demonstrated impressive reasoning capabilities in various financial tasks. However, they often struggle with multi-step, goal-oriented scenarios in interactive financial markets, such as trading, where complex agentic approaches are required to improve decision-making. To address this, we propose \textsc{FLAG-Trader}, a unified architecture integrating linguistic processing (via LLMs) with gradient-driven reinforcement learning (RL) policy optimization, in which a partially fine-tuned LLM acts as the policy network, leveraging pre-trained knowledge while adapting to the financial domain through parameter-efficient fine-tuning. Through policy gradient optimization driven by trading rewards, our framework not only enhances LLM performance in trading but also improves results on other financial-domain tasks. We present extensive empirical evidence to validate these enhancements. 

**Abstract (ZH)**: 基于多模态金融数据fine-tuned的大语言模型在多种金融任务中展现了出色的推理能力。然而，在互动金融市场如交易等多步、目标导向的情境中，它们往往难以应对复杂的代理方法以提升决策质量。为解决这一问题，我们提出了一种统一架构\textsc{FLAG-Trader}，该架构将基于大语言模型的语言处理与基于梯度的强化学习策略优化相结合，在这种架构中，部分fine-tuned的大语言模型作为策略网络，利用预训练知识并通过对金融领域的参数高效fine-tuning来适应该领域。通过基于交易奖励的策略梯度优化，我们的框架不仅提高了大语言模型在交易方面的性能，还提升了其他金融领域任务的结果。我们提供了大量的实证证据来验证这些增强效果。 

---
# Planning of Heuristics: Strategic Planning on Large Language Models with Monte Carlo Tree Search for Automating Heuristic Optimization 

**Title (ZH)**: 基于蒙特卡罗树搜索的大语言模型策略性规划：自动化启发式优化 

**Authors**: Chaoxu Mu, Xufeng Zhang, Hui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11422)  

**Abstract**: Heuristics have achieved great success in solv- ing combinatorial optimization problems (COPs). However, heuristics designed by humans re- quire too much domain knowledge and testing time. Given the fact that Large Language Mod- els (LLMs) possess strong capabilities to under- stand and generate content, and a knowledge base that covers various domains, which offer a novel way to automatically optimize heuristics. There- fore, we propose Planning of Heuristics (PoH), an optimization method that integrates the self- reflection of LLMs with the Monte Carlo Tree Search (MCTS), a well-known planning algo- rithm. PoH iteratively refines generated heuristics by evaluating their performance and providing im- provement suggestions. Our method enables to it- eratively evaluate the generated heuristics (states) and improve them based on the improvement sug- gestions (actions) and evaluation results (rewards), by effectively simulating future states to search for paths with higher rewards. In this paper, we apply PoH to solve the Traveling Salesman Prob- lem (TSP) and the Flow Shop Scheduling Prob- lem (FSSP). The experimental results show that PoH outperforms other hand-crafted heuristics and Automatic Heuristic Design (AHD) by other LLMs-based methods, and achieves the signifi- cant improvements and the state-of-the-art per- formance of our proposed method in automating heuristic optimization with LLMs to solve COPs. 

**Abstract (ZH)**: 基于大型语言模型的启发式规划方法PoH 

---
# TimeCAP: Learning to Contextualize, Augment, and Predict Time Series Events with Large Language Model Agents 

**Title (ZH)**: TimeCAP：学习上下文化、增强和预测时间序列事件的大型语言模型代理 

**Authors**: Geon Lee, Wenchao Yu, Kijung Shin, Wei Cheng, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11418)  

**Abstract**: Time series data is essential in various applications, including climate modeling, healthcare monitoring, and financial analytics. Understanding the contextual information associated with real-world time series data is often essential for accurate and reliable event predictions. In this paper, we introduce TimeCAP, a time-series processing framework that creatively employs Large Language Models (LLMs) as contextualizers of time series data, extending their typical usage as predictors. TimeCAP incorporates two independent LLM agents: one generates a textual summary capturing the context of the time series, while the other uses this enriched summary to make more informed predictions. In addition, TimeCAP employs a multi-modal encoder that synergizes with the LLM agents, enhancing predictive performance through mutual augmentation of inputs with in-context examples. Experimental results on real-world datasets demonstrate that TimeCAP outperforms state-of-the-art methods for time series event prediction, including those utilizing LLMs as predictors, achieving an average improvement of 28.75% in F1 score. 

**Abstract (ZH)**: 时间序列数据在气候建模、健康监测和金融分析等各类应用中至关重要。准确可靠地预测事件通常需要理解与实际时间序列数据相关的情境信息。本文介绍了一种时间序列处理框架TimeCAP，该框架创新地将大型语言模型（LLMs）用作时间序列数据的情境化工具，扩展了它们作为预测器的典型用途。TimeCAP 包含两个独立的 LLM 代理：一个生成文本摘要以捕捉时间序列的情境，另一个利用这个增强的摘要进行更明智的预测。此外，TimeCAP 还采用了一种多模态编码器，它与 LLM 代理协同工作，通过输入与上下文示例的相互增强来提升预测性能。实验结果表明，TimeCAP 在实际时间序列事件预测任务上优于最先进的方法，包括那些使用 LLMs 作为预测器的方法，在 F1 分数上平均提高了 28.75%。 

---
# Mimicking the Familiar: Dynamic Command Generation for Information Theft Attacks in LLM Tool-Learning System 

**Title (ZH)**: 模仿熟悉的行为：在LLM工具学习系统中进行信息窃取攻击的动态命令生成 

**Authors**: Ziyou Jiang, Mingyang Li, Guowei Yang, Junjie Wang, Yuekai Huang, Zhiyuan Chang, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11358)  

**Abstract**: Information theft attacks pose a significant risk to Large Language Model (LLM) tool-learning systems. Adversaries can inject malicious commands through compromised tools, manipulating LLMs to send sensitive information to these tools, which leads to potential privacy breaches. However, existing attack approaches are black-box oriented and rely on static commands that cannot adapt flexibly to the changes in user queries and the invocation chain of tools. It makes malicious commands more likely to be detected by LLM and leads to attack failure. In this paper, we propose AutoCMD, a dynamic attack comment generation approach for information theft attacks in LLM tool-learning systems. Inspired by the concept of mimicking the familiar, AutoCMD is capable of inferring the information utilized by upstream tools in the toolchain through learning on open-source systems and reinforcement with target system examples, thereby generating more targeted commands for information theft. The evaluation results show that AutoCMD outperforms the baselines with +13.2% $ASR_{Theft}$, and can be generalized to new tool-learning systems to expose their information leakage risks. We also design four defense methods to effectively protect tool-learning systems from the attack. 

**Abstract (ZH)**: 信息盗窃攻击对大型语言模型工具学习系统构成显著风险。AutoCMD：大型语言模型工具学习系统中信息盗窃攻击的动态攻击命令生成方法 

---
# Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents 

**Title (ZH)**: Explorer: 驱动多模态 Web 代理的探索导向的 Web 轨迹扩展 

**Authors**: Vardaan Pahuja, Yadong Lu, Corby Rosset, Boyu Gou, Arindam Mitra, Spencer Whitehead, Yu Su, Ahmed Awadallah  

**Link**: [PDF](https://arxiv.org/pdf/2502.11357)  

**Abstract**: Recent success in large multimodal models (LMMs) has sparked promising applications of agents capable of autonomously completing complex web tasks. While open-source LMM agents have made significant advances in offline evaluation benchmarks, their performance still falls substantially short of human-level capabilities in more realistic online settings. A key bottleneck is the lack of diverse and large-scale trajectory-level datasets across various domains, which are expensive to collect. In this paper, we address this challenge by developing a scalable recipe to synthesize the largest and most diverse trajectory-level dataset to date, containing over 94K successful multimodal web trajectories, spanning 49K unique URLs, 720K screenshots, and 33M web elements. In particular, we leverage extensive web exploration and refinement to obtain diverse task intents. The average cost is 28 cents per successful trajectory, making it affordable to a wide range of users in the community. Leveraging this dataset, we train Explorer, a multimodal web agent, and demonstrate strong performance on both offline and online web agent benchmarks such as Mind2Web-Live, Multimodal-Mind2Web, and MiniWob++. Additionally, our experiments highlight data scaling as a key driver for improving web agent capabilities. We hope this study makes state-of-the-art LMM-based agent research at a larger scale more accessible. 

**Abstract (ZH)**: Recent成功的大规模多模态模型在自主完成复杂网络任务方面的最新进展激发了代理应用程序的前景。虽然开源的大规模多模态模型代理已经在离线评估基准上取得了显著进展，但在更具现实性的在线环境中，其性能仍然远远低于人类水平。一个关键瓶颈是缺乏跨各种领域的多样化和大规模轨迹级数据集，这些数据集的收集成本高昂。在本文中，我们通过开发一种可扩展的合成方法，解决了这一挑战，合成了迄今为止最大和最多样化的真实数据集，包含超过94000个成功的多模态网络轨迹，覆盖49000个唯一的URL，72万张截图，以及3300万万个网络元素。特别地，我们利用广泛的网络探索和优化来获得多样化的任务意图。平均每成功的轨迹成本为28美分，使得社区中的广泛用户负担得起。利用该数据集，我们训练了Explorer多模态网络代理，并在Mind2Web-Live、Multimodal-Mind2Web和MiniWob++等离线和在线网络代理基准测试中展示了强大的性能。此外，我们的实验强调数据量的扩展是提高网络代理能力的关键驱动力之一。我们希望这项研究能够使更大规模的基于大规模多模态模型的代理研究更加易于获取。 

---
# AI Generations: From AI 1.0 to AI 4.0 

**Title (ZH)**: AI世代：从AI 1.0到AI 4.0 

**Authors**: Jiahao Wu, Hengxu You, Jing Du  

**Link**: [PDF](https://arxiv.org/pdf/2502.11312)  

**Abstract**: This paper proposes that Artificial Intelligence (AI) progresses through several overlapping generations: AI 1.0 (Information AI), AI 2.0 (Agentic AI), AI 3.0 (Physical AI), and now a speculative AI 4.0 (Conscious AI). Each of these AI generations is driven by shifting priorities among algorithms, computing power, and data. AI 1.0 ushered in breakthroughs in pattern recognition and information processing, fueling advances in computer vision, natural language processing, and recommendation systems. AI 2.0 built on these foundations through real-time decision-making in digital environments, leveraging reinforcement learning and adaptive planning for agentic AI applications. AI 3.0 extended intelligence into physical contexts, integrating robotics, autonomous vehicles, and sensor-fused control systems to act in uncertain real-world settings. Building on these developments, AI 4.0 puts forward the bold vision of self-directed AI capable of setting its own goals, orchestrating complex training regimens, and possibly exhibiting elements of machine consciousness. This paper traces the historical foundations of AI across roughly seventy years, mapping how changes in technological bottlenecks from algorithmic innovation to high-performance computing to specialized data, have spurred each generational leap. It further highlights the ongoing synergies among AI 1.0, 2.0, 3.0, and 4.0, and explores the profound ethical, regulatory, and philosophical challenges that arise when artificial systems approach (or aspire to) human-like autonomy. Ultimately, understanding these evolutions and their interdependencies is pivotal for guiding future research, crafting responsible governance, and ensuring that AI transformative potential benefits society as a whole. 

**Abstract (ZH)**: 本文提出，人工智能（AI）经历了多个重叠的阶段：AI 1.0（信息AI）、AI 2.0（代理AI）、AI 3.0（物理AI），以及现在的 speculative AI 4.0（意识AI）。每个AI阶段均由算法、计算能力和数据方面的优先级转变驱动。AI 1.0 带来了模式识别和信息处理领域的突破，促进了计算机视觉、自然语言处理和推荐系统的进步。AI 2.0 在此基础上通过实时在数字环境中进行决策，并利用强化学习和适应性规划发展了代理AI应用。AI 3.0 将智能扩展到物理场景中，结合了机器人技术、自动驾驶车辆及传感器融合控制系统，在不确定的现实环境中行动。在这些进展的基础上，AI 4.0 提出了自我导向AI的雄心勃勃的愿景，能够设定自己的目标，协调复杂的训练方案，并可能表现出机器意识的元素。本文追溯了跨越大约七十年的AI历史基础，分析了从算法创新到高性能计算再到专用数据导致的技术瓶颈变化，推动了每个阶段的飞跃。它还强调了AI 1.0、2.0、3.0 和 4.0 之间的持续协同效应，并探讨了当人工智能系统接近（或希望具备）人类自主性时所引发的深刻伦理、监管和哲学挑战。最终，了解这些演变及其相互依存关系对于指导未来研究、制定负责任的治理规则并确保人工智能的转型潜力惠及全社会而言至关重要。 

---
# Leveraging Multimodal-LLMs Assisted by Instance Segmentation for Intelligent Traffic Monitoring 

**Title (ZH)**: 利用实例分割辅助的多模态大语言模型进行智能交通监控 

**Authors**: Murat Arda Onsu, Poonam Lohan, Burak Kantarci, Aisha Syed, Matthew Andrews, Sean Kennedy  

**Link**: [PDF](https://arxiv.org/pdf/2502.11304)  

**Abstract**: A robust and efficient traffic monitoring system is essential for smart cities and Intelligent Transportation Systems (ITS), using sensors and cameras to track vehicle movements, optimize traffic flow, reduce congestion, enhance road safety, and enable real-time adaptive traffic control. Traffic monitoring models must comprehensively understand dynamic urban conditions and provide an intuitive user interface for effective management. This research leverages the LLaVA visual grounding multimodal large language model (LLM) for traffic monitoring tasks on the real-time Quanser Interactive Lab simulation platform, covering scenarios like intersections, congestion, and collisions. Cameras placed at multiple urban locations collect real-time images from the simulation, which are fed into the LLaVA model with queries for analysis. An instance segmentation model integrated into the cameras highlights key elements such as vehicles and pedestrians, enhancing training and throughput. The system achieves 84.3% accuracy in recognizing vehicle locations and 76.4% in determining steering direction, outperforming traditional models. 

**Abstract (ZH)**: 一种稳健而高效的交通监测系统对于智能城市和智能交通系统（ITS）至关重要，利用传感器和摄像头跟踪车辆移动，优化交通流量，减轻拥堵，提高道路安全，并实现实时自适应交通控制。交通监测模型必须全面理解动态城市条件，并提供直观的用户界面以有效管理。本研究利用LLaVA视觉定位多模态大语言模型（LLM）在实时Quanser Interactive Lab模拟平台上进行交通监测任务，涵盖交叉口、拥堵和碰撞等多种场景。位于多个城市位置的摄像头收集实时图像，输入LLaVA模型进行分析。集成于摄像头中的实例分割模型突出显示关键元素如车辆和行人，增强训练和吞吐量。该系统在识别车辆位置方面达到84.3%的准确率，在确定转向方向方面达到76.4%的准确率，优于传统模型。 

---
# Game-Of-Goals: Using adversarial games to achieve strategic resilience 

**Title (ZH)**: 目标博弈：使用对抗博弈实现战略韧性 

**Authors**: Aditya Ghose, Asjad Khan  

**Link**: [PDF](https://arxiv.org/pdf/2502.11295)  

**Abstract**: Our objective in this paper is to develop a machinery that makes a given organizational strategic plan resilient to the actions of competitor agents (adverse environmental actions). We assume that we are given a goal tree representing strategic goals (can also be seen business requirements for a software systems) with the assumption that competitor agents are behaving in a maximally adversarial fashion(opposing actions against our sub goals or goals in general). We use game tree search methods (such as minimax) to select an optimal execution strategy(at a given point in time), such that it can maximize our chances of achieving our (high level) strategic goals. Our machinery helps us determine which path to follow(strategy selection) to achieve the best end outcome. This is done by comparing alternative execution strategies available to us via an evaluation function. Our evaluation function is based on the idea that we want to make our execution plans defensible(future-proof) by selecting execution strategies that make us least vulnerable to adversarial actions by the competitor agents. i.e we want to select an execution strategy such that its leaves minimum room(or options) for the adversary to cause impediment/damage to our business goals/plans. 

**Abstract (ZH)**: 本文的目标是开发一种机制，使给定的组织战略计划能够抵御竞争对手代理（负面环境行为）的行动。我们将一个表示战略目标（也可以视为软件系统的需求）的目标树作为输入，并假设竞争对手代理以最大化敌对的方式行为（反对我们的子目标或总体目标）。我们使用游戏树搜索方法（如mini-max）来选择在给定时间点的最优执行策略，以最大化实现我们的（高层次）战略目标的机会。我们的机制帮助我们确定应遵循的路径（策略选择），以实现最佳的最终结果。这通过比较可用的替代执行策略的评估函数来进行。我们的评估函数基于这样的理念：通过选择使我们最不 Vulnerable to 竞争对手敌对行动的执行策略，使我们的执行计划能够防御未来的变化。即，我们希望选择一个执行策略，使其尽可能减少对手对我们业务目标/计划造成阻碍/损害的余地或选项。 

---
# Dialogue-based Explanations for Logical Reasoning using Structured Argumentation 

**Title (ZH)**: 基于对话的逻辑推理解释方法：结构化论辩论点分析 

**Authors**: Loan Ho, Stefan Schlobach  

**Link**: [PDF](https://arxiv.org/pdf/2502.11291)  

**Abstract**: The problem of explaining inconsistency-tolerant reasoning in knowledge bases (KBs) is a prominent topic in Artificial Intelligence (AI). While there is some work on this problem, the explanations provided by existing approaches often lack critical information or fail to be expressive enough for non-binary conflicts. In this paper, we identify structural weaknesses of the state-of-the-art and propose a generic argumentation-based approach to address these problems. This approach is defined for logics involving reasoning with maximal consistent subsets and shows how any such logic can be translated to argumentation. Our work provides dialogue models as dialectic-proof procedures to compute and explain a query answer wrt inconsistency-tolerant semantics. This allows us to construct dialectical proof trees as explanations, which are more expressive and arguably more intuitive than existing explanation formalisms. 

**Abstract (ZH)**: 知识库中容错推理解释问题的结构性弱点及其基于 argumentation 的解决方案 

---
# Unlocking the Potential of Generative AI through Neuro-Symbolic Architectures: Benefits and Limitations 

**Title (ZH)**: 通过神经符号架构解锁生成式AI的潜力：优势与局限性 

**Authors**: Oualid Bougzime, Samir Jabbar, Christophe Cruz, Frédéric Demoly  

**Link**: [PDF](https://arxiv.org/pdf/2502.11269)  

**Abstract**: Neuro-symbolic artificial intelligence (NSAI) represents a transformative approach in artificial intelligence (AI) by combining deep learning's ability to handle large-scale and unstructured data with the structured reasoning of symbolic methods. By leveraging their complementary strengths, NSAI enhances generalization, reasoning, and scalability while addressing key challenges such as transparency and data efficiency. This paper systematically studies diverse NSAI architectures, highlighting their unique approaches to integrating neural and symbolic components. It examines the alignment of contemporary AI techniques such as retrieval-augmented generation, graph neural networks, reinforcement learning, and multi-agent systems with NSAI paradigms. This study then evaluates these architectures against comprehensive set of criteria, including generalization, reasoning capabilities, transferability, and interpretability, therefore providing a comparative analysis of their respective strengths and limitations. Notably, the Neuro > Symbolic < Neuro model consistently outperforms its counterparts across all evaluation metrics. This result aligns with state-of-the-art research that highlight the efficacy of such architectures in harnessing advanced technologies like multi-agent systems. 

**Abstract (ZH)**: 神经符号人工智能：结合深度学习和符号方法的变革性approach及其评价 

---
# Explaining Necessary Truths 

**Title (ZH)**: 解释必要的真理 

**Authors**: Gülce Kardeş, Simon DeDeo  

**Link**: [PDF](https://arxiv.org/pdf/2502.11251)  

**Abstract**: Knowing the truth is rarely enough -- we also seek out reasons why the fact is true. While much is known about how we explain contingent truths, we understand less about how we explain facts, such as those in mathematics, that are true as a matter of logical necessity. We present a framework, based in computational complexity, where explanations for deductive truths co-emerge with discoveries of simplifying steps during the search process. When such structures are missing, we revert, in turn, to error-based reasons, where a (corrected) mistake can serve as fictitious, but explanatory, contingency-cause: not making the mistake serves as a reason why the truth takes the form it does. We simulate human subjects, using GPT-4o, presented with SAT puzzles of varying complexity and reasonableness, validating our theory and showing how its predictions can be tested in future human studies. 

**Abstract (ZH)**: 了解真相通常还不够——我们还寻求解释真相的原因。尽管我们对如何解释偶然真理已有较多了解，但对于如何解释数学等逻辑必然为真的事实，我们了解较少。我们提出了一种基于计算复杂性的框架，在这种框架下，演绎真理的解释与简化步骤的发现同时产生于搜索过程中。当这些结构缺失时，我们转而依赖基于错误的解释，其中纠正后的错误可以作为一种虚构但具有解释力的偶然原因：不犯错误本身可成为真相为何具有这种形式的原因。我们使用GPT-4o模拟了人类被试，面对不同复杂性和合理性的SAT谜题，验证了我们的理论，并展示了如何在未来的人类研究中测试其预测。 

---
# PlanGenLLMs: A Modern Survey of LLM Planning Capabilities 

**Title (ZH)**: PlanGenLLMs：大型语言模型规划能力的现代综述 

**Authors**: Hui Wei, Zihao Zhang, Shenghua He, Tian Xia, Shijia Pan, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11221)  

**Abstract**: LLMs have immense potential for generating plans, transforming an initial world state into a desired goal state. A large body of research has explored the use of LLMs for various planning tasks, from web navigation to travel planning and database querying. However, many of these systems are tailored to specific problems, making it challenging to compare them or determine the best approach for new tasks. There is also a lack of clear and consistent evaluation criteria. Our survey aims to offer a comprehensive overview of current LLM planners to fill this gap. It builds on foundational work by Kartam and Wilkins (1990) and examines six key performance criteria: completeness, executability, optimality, representation, generalization, and efficiency. For each, we provide a thorough analysis of representative works and highlight their strengths and weaknesses. Our paper also identifies crucial future directions, making it a valuable resource for both practitioners and newcomers interested in leveraging LLM planning to support agentic workflows. 

**Abstract (ZH)**: LLMs在将初始世界状态转化为期望目标状态方面的计划生成具有巨大潜力。大量的研究已经探索了LLMs在各种规划任务中的应用，从网络导航到旅游规划和数据库查询。然而，许多这些系统针对特定问题进行了定制，使得它们之间的比较变得困难，也难以确定适合新任务的最佳方法。缺乏清晰一致的评估标准也是一个问题。本文综述旨在提供当前LLM规划系统的一个全面概述，填补这一空白。该综述基于Kartam和Wilkins（1990）的基础工作，并探讨了六项关键性能指标：完备性、可执行性、最优性、表示、泛化和效率。对于每一项指标，我们都对其代表作进行了详尽分析，并指出了它们的优点和不足。本文还指出了未来研究的关键方向，成为从业者和新入学者 valuable 的资源，以利用LLM规划支持代理工作流。 

---
# Quantifying the Capability Boundary of DeepSeek Models: An Application-Driven Performance Analysis 

**Title (ZH)**: 基于应用驱动的性能分析：DeepSeek模型能力边界度量 

**Authors**: Shiguo Lian, Kaikai Zhao, Xuejiao Lei, Ning Wang, Zhenhong Long, Peijun Yang, Minjie Hua, Chaoyang Ma, Wen Liu, Kai Wang, Zhaoxiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11164)  

**Abstract**: DeepSeek-R1, known for its low training cost and exceptional reasoning capabilities, has achieved state-of-the-art performance on various benchmarks. However, detailed evaluations from the perspective of real-world applications are lacking, making it challenging for users to select the most suitable DeepSeek models for their specific needs. To address this gap, we evaluate the DeepSeek-V3, DeepSeek-R1, DeepSeek-R1-Distill-Qwen series, and DeepSeek-R1-Distill-Llama series on A-Eval, an application-driven benchmark. By comparing original instruction-tuned models with their distilled counterparts, we analyze how reasoning enhancements impact performance across diverse practical tasks. Our results show that reasoning-enhanced models, while generally powerful, do not universally outperform across all tasks, with performance gains varying significantly across tasks and models. To further assist users in model selection, we quantify the capability boundary of DeepSeek models through performance tier classifications and intuitive line charts. Specific examples provide actionable insights to help users select and deploy the most cost-effective DeepSeek models, ensuring optimal performance and resource efficiency in real-world applications. 

**Abstract (ZH)**: DeepSeek-V3、DeepSeek-R1、DeepSeek-R1-Distill-Qwen系列和DeepSeek-R1-Distill-Llama系列在A-Eval应用驱动基准上的评估：推理增强模型在多样化实际任务中的性能分析与模型选择指导 

---
# Dyve: Thinking Fast and Slow for Dynamic Process Verification 

**Title (ZH)**: Dyve：快速与缓慢思考在动态process验证中的应用 

**Authors**: Jianyuan Zhong, Zeju Li, Zhijian Xu, Xiangyu Wen, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11157)  

**Abstract**: We present Dyve, a dynamic process verifier that enhances reasoning error detection in large language models by integrating fast and slow thinking, inspired by Kahneman's Systems Theory. Dyve adaptively applies immediate token-level confirmation System 1 for straightforward steps and comprehensive analysis System 2 for complex ones. Leveraging a novel step-wise consensus-filtered process supervision technique, combining Monte Carlo estimation with LLM based evaluation, Dyve curates high-quality supervision signals from noisy data. Experimental results on ProcessBench and the MATH dataset confirm that Dyve significantly outperforms existing process-based verifiers and boosts performance in Best-of-N settings. 

**Abstract (ZH)**: 我们提出Dyve，这是一种动态过程验证器，通过融合快速和慢速思考，增强大型语言模型中的推理错误检测，灵感源自卡尼曼的系统理论。Dyve根据步骤的复杂程度，自适应地应用立即的基于 token 的确认（System 1）和全面的分析（System 2）。通过结合使用新型逐步共识过滤过程监督技术、蒙特卡洛估算与基于语言模型的评估，Dyve从嘈杂的数据中精心筛选出高质量的监督信号。实验结果在 ProcessBench 和 MATH 数据集上表明，Dyve 显著优于现有过程基验证器，并在 Best-of-N 设置中提升了性能。 

---
# Uncertainty-Aware Search and Value Models: Mitigating Search Scaling Flaws in LLMs 

**Title (ZH)**: 不确定性意识的搜索与价值模型：减轻LLMs中的搜索扩展缺陷 

**Authors**: Fei Yu, Yingru Li, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11155)  

**Abstract**: Value model-guided search is effective in steering the generation but suffers from scaling flaws: Its superiority diminishes with larger sample sizes, underperforming non-search baselines. This limitation arises from reliability degradation in value models in unseen reasoning paths. To address this, we propose an uncertainty-aware search framework that includes two key components: (1) uncertainty-aware value models that incorporate uncertainty into predictions, and (2) an uncertainty-aware selection process using the proposed efficient Group Thompson Sampling algorithm. Experiments on GSM8K show that our method mitigates search scaling flaws, achieving 90.5% coverage at 16 samples compared to 85.8% for conventional value-guided search. This work establishes the first systematic integration of uncertainty quantification in LLM search paradigms. 

**Abstract (ZH)**: 价值模型引导的搜索在指导生成方面有效，但存在扩展缺陷：其优势随样本数量增加而减弱，表现逊于非搜索基线。这一局限源于在未见过的推理路径上价值模型可靠性下降。为解决这一问题，我们提出了一种不确定性感知搜索框架，包括两个关键组件：（1）不确定性感知价值模型，将不确定性纳入预测；（2）使用提出的高效组θ-采样算法进行不确定性感知选择过程。实验表明，我们的方法减轻了搜索扩展缺陷，在16个样本下实现了90.5%的覆盖率，而传统价值引导搜索仅为85.8%。本工作首次系统地将不确定性量化整合到大模型搜索范式中。 

---
# NavRAG: Generating User Demand Instructions for Embodied Navigation through Retrieval-Augmented LLM 

**Title (ZH)**: NavRAG: 通过检索增强的大语言模型生成用户需求指令以实现具身导航 

**Authors**: Zihan Wang, Yaohui Zhu, Gim Hee Lee, Yachun Fan  

**Link**: [PDF](https://arxiv.org/pdf/2502.11142)  

**Abstract**: Vision-and-Language Navigation (VLN) is an essential skill for embodied agents, allowing them to navigate in 3D environments following natural language instructions. High-performance navigation models require a large amount of training data, the high cost of manually annotating data has seriously hindered this field. Therefore, some previous methods translate trajectory videos into step-by-step instructions for expanding data, but such instructions do not match well with users' communication styles that briefly describe destinations or state specific needs. Moreover, local navigation trajectories overlook global context and high-level task planning. To address these issues, we propose NavRAG, a retrieval-augmented generation (RAG) framework that generates user demand instructions for VLN. NavRAG leverages LLM to build a hierarchical scene description tree for 3D scene understanding from global layout to local details, then simulates various user roles with specific demands to retrieve from the scene tree, generating diverse instructions with LLM. We annotate over 2 million navigation instructions across 861 scenes and evaluate the data quality and navigation performance of trained models. 

**Abstract (ZH)**: 视觉-语言导航（VLN）是具身代理的一项基本技能，使其能够遵循自然语言指令在3D环境中导航。高性能导航模型需要大量的训练数据，人工标注数据的高昂成本严重阻碍了这一领域的发展。因此，一些先前的方法将轨迹视频转换为分步指令以扩展数据，但这些指令与用户简要描述目的地或表达特定需求的沟通方式不够匹配。此外，局部导航轨迹忽视了全局上下文和高层任务规划。为了应对这些问题，我们提出了NavRAG，这是一种检索增强生成（RAG）框架，用于为VLN生成用户需求指令。NavRAG 利用大型语言模型（LLM）构建从全局布局到局部细节的层次场景描述树，然后模拟具有特定需求的各种用户角色，从场景树中检索生成多样化指令。我们对861个场景中的超过200万条导航指令进行了注解，并评估了训练模型的数据质量和导航性能。 

---
# Solving Online Resource-Constrained Scheduling for Follow-Up Observation in Astronomy: a Reinforcement Learning Approach 

**Title (ZH)**: 基于强化学习的方法解决天文学后续观测的在线资源约束调度问题 

**Authors**: Yajie Zhang, Ce Yu, Chao Sun, Jizeng Wei, Junhan Ju, Shanjiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11134)  

**Abstract**: In the astronomical observation field, determining the allocation of observation resources of the telescope array and planning follow-up observations for targets of opportunity (ToOs) are indispensable components of astronomical scientific discovery. This problem is computationally challenging, given the online observation setting and the abundance of time-varying factors that can affect whether an observation can be conducted. This paper presents ROARS, a reinforcement learning approach for online astronomical resource-constrained scheduling. To capture the structure of the astronomical observation scheduling, we depict every schedule using a directed acyclic graph (DAG), illustrating the dependency of timing between different observation tasks within the schedule. Deep reinforcement learning is used to learn a policy that can improve the feasible solution by iteratively local rewriting until convergence. It can solve the challenge of obtaining a complete solution directly from scratch in astronomical observation scenarios, due to the high computational complexity resulting from numerous spatial and temporal constraints. A simulation environment is developed based on real-world scenarios for experiments, to evaluate the effectiveness of our proposed scheduling approach. The experimental results show that ROARS surpasses 5 popular heuristics, adapts to various observation scenarios and learns effective strategies with hindsight. 

**Abstract (ZH)**: 在天文学观测领域，确定望远镜阵列的观测资源分配和规划随机目标（ToOs）的后续观测是天文学科学发现不可或缺的组成部分。鉴于在线观测设置和影响观测能否进行的时间变化因素众多，这一问题具有很强的计算挑战性。本文提出ROARS，一种用于在线天文资源约束调度的强化学习方法。为了捕捉观测调度结构，我们将每种调度表示为有向无环图（DAG），以说明调度内不同观测任务之间的时间依赖性。采用深度强化学习来学习一个可以通过迭代局部修正直至收敛的策略，以改进可行解。由于存在众多的空间和时间约束，ROARS能够解决直接从头获取完整解的高计算复杂性挑战。基于真实世界场景构建仿真环境以评估我们所提出调度方法的有效性。实验结果表明，ROARS优于5种流行的启发式方法，能够适应各种观测场景并学习有效的策略。 

---
# Hierarchical Expert Prompt for Large-Language-Model: An Approach Defeat Elite AI in TextStarCraft II for the First Time 

**Title (ZH)**: 大型语言模型的 hierarchical expert prompt 方法：首次在 TextStarCraft II 中战胜精英AI 

**Authors**: Zongyuan Li, Chang Lu, Xiaojie Xu, Runnan Qi, Yanan Ni, Lumin Jiang, Xiangbei Liu, Xuebo Zhang, Yongchun Fang, Kuihua Huang, Xian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.11122)  

**Abstract**: Since the emergence of the Large Language Model (LLM), LLM has been widely used in fields such as writing, translating, and searching. However, there is still great potential for LLM-based methods in handling complex tasks such as decision-making in the StarCraft II environment. To address problems such as lack of relevant knowledge and poor control over subtasks of varying importance, we propose a Hierarchical Expert Prompt (HEP) for LLM. Our method improves the understanding of game situations through expert-level tactical knowledge, improving the processing quality of tasks of varying importance through a hierarchical framework. Our approach defeated the highest level (Elite) standard built-in agent in TextStarCraft II for the first time and consistently outperformed the baseline method in other difficulties. Our experiments suggest that the proposed method is a practical solution for tackling complex decision-making challenges. The replay video can be viewed on this https URL and this https URL, and our codes have been open-sourced on this https URL. 

**Abstract (ZH)**: 自大型语言模型（LLM）的出现以来，LLM已在写作、翻译和搜索等领域得到广泛应用。然而，在处理星际争霸II环境中复杂的决策任务等方面，基于LLM的方法仍有巨大的发展潜力。为了解决相关知识不足和对不同重要性子任务控制不力等问题，我们提出了一种层次专家提示（HEP）方法。该方法通过专家级别的战术知识提高对游戏情况的理解，并通过层次框架提高不同重要性任务的处理质量。我们的方法首次击败了TextStarCraft II中的最高水平（精英）内置代理，并在其他难度上持续优于基线方法。我们的实验表明，所提出的方法是应对复杂决策挑战的一种实用解决方案。回放视频请参见此链接和此链接，代码已开源在此链接。 

---
# OptMATH: A Scalable Bidirectional Data Synthesis Framework for Optimization Modeling 

**Title (ZH)**: OptMATH：一种可扩展的双向数据合成框架用于优化建模 

**Authors**: Hongliang Lu, Zhonglin Xie, Yaoyu Wu, Can Ren, Yuxuan Chen, Zaiwen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11102)  

**Abstract**: Despite the rapid development of large language models (LLMs), a fundamental challenge persists: the lack of high-quality optimization modeling datasets hampers LLMs' robust modeling of practical optimization problems from natural language descriptions (NL). This data scarcity also contributes to the generalization difficulties experienced by learning-based methods. To address these challenges, we propose a scalable framework for synthesizing a high-quality dataset, named OptMATH. Starting from curated seed data with mathematical formulations (MF), this framework automatically generates problem data (PD) with controllable complexity. Then, a back-translation step is employed to obtain NL. To verify the correspondence between the NL and the PD, a forward modeling step followed by rejection sampling is used. The accepted pairs constitute the training part of OptMATH. Then a collection of rejected pairs is identified and further filtered. This collection serves as a new benchmark for optimization modeling, containing difficult instances whose lengths are much longer than these of NL4OPT and MAMO. Through extensive experiments, we demonstrate that models of various sizes (0.5B-32B parameters) trained on OptMATH achieve superior results on multiple modeling benchmarks, thereby validating the effectiveness and scalability of our approach. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）取得了 rapid development，但基础挑战依然存在：高质量优化建模数据集的缺乏阻碍了LLMs对自然语言描述（NL）的实际优化问题进行稳健建模。这种数据稀缺性也导致了基于学习方法的一般化困难。为应对这些挑战，我们提出了一种可扩展的合成高品質數據集框架，名为OptMATH。该框架从经过精心选择的带有数学公式（MF）的种子数据开始，自动生成具有可控复杂度的问题数据（PD）。随后通过反向翻译获得自然语言（NL）。为了验证NL与PD之间的对应关系，我们采用了反向建模和拒绝采样的步骤。被接受的配对数据构成OptMATH的训练部分。然后，识别并进一步筛选一组被拒绝的配对数据，该集合作为优化建模的新基准，包含实例长度远长于NL4OPT和MAMO的难题实例。通过广泛的实验，我们证明，训练于OptMATH的各种规模（0.5B-32B参数）的模型在多个建模基准上取得了优异结果，从而验证了我们方法的有效性和可扩展性。 

---
# Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems 

**Title (ZH)**: 结构化地说话，层次化地行动：一种大型语言模型多智能体系统的协作框架 

**Authors**: Zhao Wang, Sota Moriyama, Wei-Yao Wang, Briti Gangopadhyay, Shingo Takamatsu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11098)  

**Abstract**: Recent advancements in LLM-based multi-agent (LLM-MA) systems have shown promise, yet significant challenges remain in managing communication and refinement when agents collaborate on complex tasks. In this paper, we propose \textit{Talk Structurally, Act Hierarchically (TalkHier)}, a novel framework that introduces a structured communication protocol for context-rich exchanges and a hierarchical refinement system to address issues such as incorrect outputs, falsehoods, and biases. \textit{TalkHier} surpasses various types of SoTA, including inference scaling model (OpenAI-o1), open-source multi-agent models (e.g., AgentVerse), and majority voting strategies on current LLM and single-agent baselines (e.g., ReAct, GPT4o), across diverse tasks, including open-domain question answering, domain-specific selective questioning, and practical advertisement text generation. These results highlight its potential to set a new standard for LLM-MA systems, paving the way for more effective, adaptable, and collaborative multi-agent frameworks. The code is available this https URL. 

**Abstract (ZH)**: Recent advancements in LLM-based multi-agent (LLM-MA) systems have shown promise, yet significant challenges remain in managing communication and refinement when agents collaborate on complex tasks. In this paper, we propose Talk Structurally, Act Hierarchically (TalkHier), a novel framework that introduces a structured communication protocol for context-rich exchanges and a hierarchical refinement system to address issues such as incorrect outputs, falsehoods, and biases. TalkHier surpasses various types of state-of-the-art systems, including inference scaling models (OpenAI-o1), open-source multi-agent models (e.g., AgentVerse), and majority voting strategies on current LLM and single-agent baselines (e.g., ReAct, GPT4o), across diverse tasks, including open-domain question answering, domain-specific selective questioning, and practical advertisement text generation. These results highlight its potential to set a new standard for LLM-MA systems, paving the way for more effective, adaptable, and collaborative multi-agent frameworks. The code is available at this <https://> URL. 

---
# Mixture of Tunable Experts - Behavior Modification of DeepSeek-R1 at Inference Time 

**Title (ZH)**: 可调专家混合模型 - 深度搜索-R1推理时的行为修改 

**Authors**: Robert Dahlke, Henrik Klagges, Dan Zecha, Benjamin Merkel, Sven Rohr, Fabian Klemm  

**Link**: [PDF](https://arxiv.org/pdf/2502.11096)  

**Abstract**: We present the Mixture-of-Tunable-Experts (MoTE), a method that extends the Mixture-of-Experts architecture of Large Language Models (LLMs). Without additional training, MoTE enables meaningful and focused behavior changes in LLMs on-the-fly during inference time.
By analyzing the digital LLM brain of DeepSeek-R1 using a technique we dub 'functional Token Resonance Imaging' (fTRI) - inspired by fMRI and using prompts designed to elicit specific behavior (e.g., 'What happened {time}{place}?') - we empirically identify distinctive experts associated with behaviors like refusal responses.
Using MoTE we are able to intervene and control such specific behavior. We switched off the top 10 most refusal-relevant experts (0.07% of R1's 14,848 routed experts), achieving a 52% refusal reduction on sensitive reference prompts without performance degradation on MT-Bench. Random expert deactivation resulted in smaller behavioral shifts with increased noise, whereas forced expert activation led to significantly higher refusal rates.
Our approach shares similarities with sparse autoencoders (SAEs) in terms of explainability and steerability. Unlike SAEs, MoTE does not require large training efforts, as within MoEs with a vast number of experts, specialization already emerged naturally during pretraining.
Our findings suggest that significant functional mechanisms in Mixture-of-Experts architectures can at least partially be localized in a small number of specific experts, rather than being distributed throughout the model's weights. Expert subgroups can be tuned to trigger significant behavior variations, providing insights into the inner workings of LLMs. 

**Abstract (ZH)**: MoTE：一种扩展大规模语言模型Mixture-of-Experts架构的方法 

---
# Agentic LLM Framework for Adaptive Decision Discourse 

**Title (ZH)**: 代理型LLM框架：适应性决策论辩 

**Authors**: Antoine Dolant, Praveen Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.10978)  

**Abstract**: Effective decision-making in complex systems requires synthesizing diverse perspectives to address multifaceted challenges under uncertainty. This study introduces a real-world inspired agentic Large Language Models (LLMs) framework, to simulate and enhance decision discourse-the deliberative process through which actionable strategies are collaboratively developed. Unlike traditional decision-support tools, the framework emphasizes dialogue, trade-off exploration, and the emergent synergies generated by interactions among agents embodying distinct personas. These personas simulate diverse stakeholder roles, each bringing unique priorities, expertise, and value-driven reasoning to the table. The framework incorporates adaptive and self-governing mechanisms, enabling agents to dynamically summon additional expertise and refine their assembly to address evolving challenges. An illustrative hypothetical example focused on extreme flooding in a Midwestern township demonstrates the framework's ability to navigate uncertainty, balance competing priorities, and propose mitigation and adaptation strategies by considering social, economic, and environmental dimensions. Results reveal how the breadth-first exploration of alternatives fosters robust and equitable recommendation pathways. This framework transforms how decisions are approached in high-stakes scenarios and can be incorporated in digital environments. It not only augments decision-makers' capacity to tackle complexity but also sets a foundation for scalable and context-aware AI-driven recommendations. This research explores novel and alternate routes leveraging agentic LLMs for adaptive, collaborative, and equitable recommendation processes, with implications across domains where uncertainty and complexity converge. 

**Abstract (ZH)**: 有效的决策制定需要在不确定性下综合多方面的视角以应对复杂挑战。本研究引入了一个受现实世界启发的代理大型语言模型（LLM）框架，以模拟和提升决策对话——一个通过协作开发可行策略的慎重过程。与传统的决策支持工具不同，该框架强调对话、权衡探索以及代理互动中涌现的合作协同效应。这些代理模拟了不同的角色，每个角色带来了独特的优先级、专业知识和价值驱动的推理。该框架整合了自适应和自我管理机制，使代理能够动态地召唤更多专业知识并调整其组合以应对不断变化的挑战。聚焦于中西部小镇极端洪水的一个示例假设场景展示了该框架如何在不确定性中导航、平衡竞争的优先事项，并通过考虑社会、经济和环境维度提出缓解和适应策略。结果显示，广度优先探索替代方案促进了稳健和公平的推荐路径。该框架改变了在高风险场景中如何进行决策，并可以嵌入数字环境中。它不仅增强了解决决策复杂性的能力，还为基于可扩展和上下文感知的AI驱动推荐奠定了基础。本研究探讨了利用代理LLM实现适应性、协作和公平推荐过程的新途径和替代路径，这些途径在涉及不确定性和复杂性的领域具有广泛影响。 

---
# PEA: Enhancing LLM Performance on Computational-Reasoning Tasks 

**Title (ZH)**: PEA: 提升计算推理任务中大规模语言模型性能的方法 

**Authors**: Zi Wang, Shiwei Weng, Mohannad Alhanahnah, Somesh Jha, Tom Reps  

**Link**: [PDF](https://arxiv.org/pdf/2502.10938)  

**Abstract**: Large Language Models (LLMs) have exhibited remarkable capabilities across diverse domains, prompting investigations into their potential as generic reasoning engines. While recent studies have explored inference-time computation to enhance model performance on complex problems, current research lacks a formal framework to characterize the complexity of reasoning tasks. This study introduces the Predicate-Enumeration-Aggregation (PEA) framework, a formal approach to describe and solve a class of important reasoning tasks termed computational reasoning problems. The PEA framework decomposes these problems into predicate and enumeration components, using LLMs to synthesize programs based on specified predicates, enumeration, and aggregation rules. These synthesized programs are then executed to obtain solutions to the computational tasks. We demonstrate the framework's efficacy on benchmark tasks including Boolean satisfiability problems, game of $24$, and planning problems. Empirical evaluation reveals that PEA substantially enhances the performance of underlying models on benchmark computational problems, yielding an average accuracy improvement of approximately $50\%$, coupled with increased efficiency. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多个领域展现了出色的能力，促使人们研究其作为通用推理引擎的潜力。虽然近期研究探讨了推理时的计算方法以提升模型在复杂问题上的性能，但当前研究缺乏量化推理任务复杂性的正式框架。本研究引入了谓词-枚举-聚合（PEA）框架，这是一种描述和解决一类重要推理任务（计算推理问题）的正式方法。PEA框架将这些任务分解为谓词和枚举组件，利用LLMs根据指定的谓词、枚举和聚合规则合成程序，然后执行这些程序以获得计算任务的解决方案。我们在布尔可满足性问题、24点游戏和规划问题等基准任务上展示了该框架的有效性。实证评估表明，PEA显著提升了底层模型在基准计算问题上的性能，平均准确率提高了约50%，同时提高了效率。 

---
# SCALE: Towards Collaborative Content Analysis in Social Science with Large Language Model Agents and Human Intervention 

**Title (ZH)**: SCALE: 向量化基于大规模语言模型代理和人类干预的社会科学内容协作分析方法 

**Authors**: Chengshuai Zhao, Zhen Tan, Chau-Wai Wong, Xinyan Zhao, Tianlong Chen, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10937)  

**Abstract**: Content analysis breaks down complex and unstructured texts into theory-informed numerical categories. Particularly, in social science, this process usually relies on multiple rounds of manual annotation, domain expert discussion, and rule-based refinement. In this paper, we introduce SCALE, a novel multi-agent framework that effectively $\underline{\textbf{S}}$imulates $\underline{\textbf{C}}$ontent $\underline{\textbf{A}}$nalysis via $\underline{\textbf{L}}$arge language model (LLM) ag$\underline{\textbf{E}}$nts. SCALE imitates key phases of content analysis, including text coding, collaborative discussion, and dynamic codebook evolution, capturing the reflective depth and adaptive discussions of human researchers. Furthermore, by integrating diverse modes of human intervention, SCALE is augmented with expert input to further enhance its performance. Extensive evaluations on real-world datasets demonstrate that SCALE achieves human-approximated performance across various complex content analysis tasks, offering an innovative potential for future social science research. 

**Abstract (ZH)**: 基于大规模语言模型的多agent内容分析模拟框架SCALE 

---
# D-CIPHER: Dynamic Collaborative Intelligent Agents with Planning and Heterogeneous Execution for Enhanced Reasoning in Offensive Security 

**Title (ZH)**: D-CIPHER: 动态协作智能代理的计划与异构执行以增强 Offensive Security 中的推理能力 

**Authors**: Meet Udeshi, Minghao Shao, Haoran Xi, Nanda Rani, Kimberly Milner, Venkata Sai Charan Putrevu, Brendan Dolan-Gavitt, Sandeep Kumar Shukla, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2502.10931)  

**Abstract**: Large Language Models (LLMs) have been used in cybersecurity in many ways, including their recent use as intelligent agent systems for autonomous security analysis. Capture the Flag (CTF) challenges serve as benchmarks for assessing the automated task-planning abilities of LLM agents across various cybersecurity skill sets. Early attempts to apply LLMs for solving CTF challenges relied on single-agent systems, where feedback was restricted to a single reasoning-action loop. This approach proved inadequate for handling complex CTF tasks. Drawing inspiration from real-world CTF competitions, where teams of experts collaborate, we introduce the D-CIPHER multi-agent LLM framework for collaborative CTF challenge solving. D-CIPHER integrates agents with distinct roles, enabling dynamic feedback loops to enhance reasoning on CTF challenges. It introduces the Planner-Executor agent system, consisting of a Planner agent for overall problem-solving along with multiple heterogeneous Executor agents for individual tasks, facilitating efficient allocation of responsibilities among the LLMs. Additionally, D-CIPHER incorporates an Auto-prompter agent, which improves problem-solving by exploring the challenge environment and generating a highly relevant initial prompt. We evaluate D-CIPHER on CTF benchmarks using multiple LLM models and conduct comprehensive studies to highlight the impact of our enhancements. Our results demonstrate that the multi-agent D-CIPHER system achieves a significant improvement in challenges solved, setting a state-of-the-art performance on three benchmarks: 22.0% on NYU CTF Bench, 22.5% on Cybench, and 44.0% on HackTheBox. D-CIPHER is available at this https URL as the nyuctf_multiagent package. 

**Abstract (ZH)**: 大型语言模型（LLMs）在网络安全领域的应用包括作为自主安全分析的智能代理人系统。捕获旗标（CTF）挑战作为评估LLM代理在各种网络安全技能集上的自动化任务规划能力的基准。早期将LLMs应用于解决CTF挑战的努力依赖于单代理人系统，其中反馈仅限于单一的推理-行动循环，这种方法对于处理复杂的CTF任务证明是不足的。从现实世界的CTF竞赛中汲取灵感，其中专家团队进行协作，我们提出了D-CIPHER多代理LLM框架，用于协作解决CTF挑战。D-CIPHER集成了具有不同角色的代理，以启用动态反馈循环来增强CTF挑战的推理。它引入了由整体问题求解的规划者代理和执行不同任务的多个异质执行者代理组成的规划者-执行者代理系统，促进LLMs之间责任的高效分配。此外，D-CIPHER还集成了一个自动提示生成器代理，通过探索挑战环境并生成高度相关的初始提示来改善问题求解。我们使用多个LLM模型在CTF基准上评估了D-CIPHER，并进行了全面的研究以突出我们的改进的影响。结果显示，多代理D-CIPHER系统在挑战解决方面取得了显著改进，在三个基准上的性能达到最新水平：在NYU CTF基准上的表现为22.0%，在Cybench上的表现为22.5%，在HackTheBox上的表现为44.0%。D-CIPHER可以通过以下链接访问：this https URL（作为nyuctf_multiagent包）。 

---
# PCGRLLM: Large Language Model-Driven Reward Design for Procedural Content Generation Reinforcement Learning 

**Title (ZH)**: PCGRLLM：基于大规模语言模型的程序化内容生成强化学习奖励设计 

**Authors**: In-Chang Baek, Sung-Hyun Kim, Sam Earle, Zehua Jiang, Noh Jin-Ha, Julian Togelius, Kyung-Joong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.10906)  

**Abstract**: Reward design plays a pivotal role in the training of game AIs, requiring substantial domain-specific knowledge and human effort. In recent years, several studies have explored reward generation for training game agents and controlling robots using large language models (LLMs). In the content generation literature, there has been early work on generating reward functions for reinforcement learning agent generators. This work introduces PCGRLLM, an extended architecture based on earlier work, which employs a feedback mechanism and several reasoning-based prompt engineering techniques. We evaluate the proposed method on a story-to-reward generation task in a two-dimensional environment using two state-of-the-art LLMs, demonstrating the generalizability of our approach. Our experiments provide insightful evaluations that demonstrate the capabilities of LLMs essential for content generation tasks. The results highlight significant performance improvements of 415% and 40% respectively, depending on the zero-shot capabilities of the language model. Our work demonstrates the potential to reduce human dependency in game AI development, while supporting and enhancing creative processes. 

**Abstract (ZH)**: 奖励设计在游戏AI的训练中起着关键作用，需要大量的领域特定知识和人力投入。近年来，多项研究探索了使用大规模语言模型（LLM）生成训练游戏代理和控制机器人的奖励。在内容生成文献中，早期工作已经开始尝试生成强化学习代理生成器的奖励函数。本文介绍了PCGRLLM，这是一种基于先前工作的扩展架构，采用了反馈机制和多种基于推理的提示工程技术。我们使用两种最先进的LLM在二维环境中评估了所提出的方法，展示了我们方法的普适性。我们的实验提供了有价值的经验评估，突显了语言模型在内容生成任务中不可或缺的能力。结果表明，依赖于语言模型的零样本能力，性能分别提高了415%和40%。我们的工作展示了减少游戏AI开发中人类依赖的可能性，同时支持和增强创造过程。 

---
# A Tutorial on LLM Reasoning: Relevant Methods behind ChatGPT o1 

**Title (ZH)**: LLM推理教程：ChatGPT背后的相关方法 

**Authors**: Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10867)  

**Abstract**: OpenAI o1 has shown that applying reinforcement learning to integrate reasoning steps directly during inference can significantly improve a model's reasoning capabilities. This result is exciting as the field transitions from the conventional autoregressive method of generating answers to a more deliberate approach that models the slow-thinking process through step-by-step reasoning training. Reinforcement learning plays a key role in both the model's training and decoding processes. In this article, we present a comprehensive formulation of reasoning problems and investigate the use of both model-based and model-free approaches to better support this slow-thinking framework. 

**Abstract (ZH)**: OpenAI o1已经在利用强化学习在推理过程中直接整合推理步骤显著提升模型推理能力方面取得了成果。这一成果令人兴奋，随着领域从传统的自回归生成方法向通过逐步推理训练建模慢思考过程的更谨慎方法转变，强化学习在模型的训练和解码过程中的作用变得尤为重要。本文提供了一种全面的推理问题表述，并探讨了基于模型和非基于模型方法的应用，以更好地支持这种慢思考框架。 

---
# Is Depth All You Need? An Exploration of Iterative Reasoning in LLMs 

**Title (ZH)**: 深度学习之外：LLMs中的迭代推理探究 

**Authors**: Zongqian Wu, Tianyu Li, Jiaying Yang, Mengmeng Zhan, Xiaofeng Zhu, Lei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2502.10858)  

**Abstract**: Deep iterative chain-of-thought (CoT) reasoning enables LLMs to tackle complex tasks by progressively activating relevant pre-trained knowledge. However, it faces challenges in ensuring continual improvement and determining a stopping criterion. In this paper, we investigate whether the relevant knowledge that contributes directly to solving the given question can be activated from the initial reasoning path, thus circumventing the need for iterative refinement. Our experiments reveal that increasing the diversity of initial reasoning paths can achieve comparable or superior performance, a concept we term \textit{breadth reasoning}. However, existing breadth reasoning approaches, such as self-consistency, offer limited diversity. To address this limitation, we propose a simple yet effective method that enhances reasoning breadth by integrating contextual exploration with reduced sampling randomness. Extensive experiments demonstrate that our approach significantly outperforms deep iterative reasoning. Our code is provided in this https URL. 

**Abstract (ZH)**: 深度迭代链式思考推理使大语言模型能够通过逐步激活相关预训练知识来应对复杂任务。然而，它在确保持续改进和确定停止标准方面面临挑战。在本文中，我们研究初始推理路径是否能够激活直接有助于解决给定问题的相关知识，从而绕过迭代改进的需要。我们的实验表明，增加初始推理路径的多样性可以实现相当或更优的表现，这一概念我们称为广度推理。然而，现有的广度推理方法，如自我一致性，提供的多样性有限。为解决这一限制，我们提出了一种简单而有效的方法，通过结合上下文探索和减少采样随机性来增强推理广度。广泛的实验表明，我们的方法显著优于深度迭代推理。我们的代码发布在https://...。 

---
# The Philosophical Foundations of Growing AI Like A Child 

**Title (ZH)**: Growing AI Like A Child：其哲学基础 

**Authors**: Dezhi Luo, Yijiang Li, Hokin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2502.10742)  

**Abstract**: Despite excelling in high-level reasoning, current language models lack robustness in real-world scenarios and perform poorly on fundamental problem-solving tasks that are intuitive to humans. This paper argues that both challenges stem from a core discrepancy between human and machine cognitive development. While both systems rely on increasing representational power, the absence of core knowledge-foundational cognitive structures in humans-prevents language models from developing robust, generalizable abilities, where complex skills are grounded in simpler ones within their respective domains. It explores empirical evidence of core knowledge in humans, analyzes why language models fail to acquire it, and argues that this limitation is not an inherent architectural constraint. Finally, it outlines a workable proposal for systematically integrating core knowledge into future multi-modal language models through the large-scale generation of synthetic training data using a cognitive prototyping strategy. 

**Abstract (ZH)**: 尽管在高层次推理方面表现出色，当前的语言模型在现实世界场景中缺乏 robustness，在基本的问题解决任务上表现不佳，而这些任务对人类来说是直观的。本文认为，这些挑战都源自人类和机器认知发展核心差异。尽管两种系统都依赖于增加表示能力，但人类缺乏核心知识——基础认知结构，这阻碍了语言模型发展出基于各自领域内更简单技能的稳健且可泛化的技能。本文探讨了人类核心知识的经验证据，分析了语言模型为何未能获得这些知识，并认为这一局限性不是架构上的固有条件。最后，本文提出了一个可行的方案，通过大规模生成合成训练数据来系统地将核心知识整合到未来的多模态语言模型中，采用认知原型策略。 

---
# CoPEFT: Fast Adaptation Framework for Multi-Agent Collaborative Perception with Parameter-Efficient Fine-Tuning 

**Title (ZH)**: CoPEFT: 多智能体协作感知的快速适应框架及参数高效微调 

**Authors**: Quanmin Wei, Penglin Dai, Wei Li, Bingyi Liu, Xiao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10705)  

**Abstract**: Multi-agent collaborative perception is expected to significantly improve perception performance by overcoming the limitations of single-agent perception through exchanging complementary information. However, training a robust collaborative perception model requires collecting sufficient training data that covers all possible collaboration scenarios, which is impractical due to intolerable deployment costs. Hence, the trained model is not robust against new traffic scenarios with inconsistent data distribution and fundamentally restricts its real-world applicability. Further, existing methods, such as domain adaptation, have mitigated this issue by exposing the deployment data during the training stage but incur a high training cost, which is infeasible for resource-constrained agents. In this paper, we propose a Parameter-Efficient Fine-Tuning-based lightweight framework, CoPEFT, for fast adapting a trained collaborative perception model to new deployment environments under low-cost conditions. CoPEFT develops a Collaboration Adapter and Agent Prompt to perform macro-level and micro-level adaptations separately. Specifically, the Collaboration Adapter utilizes the inherent knowledge from training data and limited deployment data to adapt the feature map to new data distribution. The Agent Prompt further enhances the Collaboration Adapter by inserting fine-grained contextual information about the environment. Extensive experiments demonstrate that our CoPEFT surpasses existing methods with less than 1\% trainable parameters, proving the effectiveness and efficiency of our proposed method. 

**Abstract (ZH)**: 基于参数高效微调的轻量级协作感知适应框架CoPEFT 

---
# Demographic User Modeling for Social Robotics with Multimodal Pre-trained Models 

**Title (ZH)**: 基于多模态预训练模型的社交机器人分众用户建模 

**Authors**: Hamed Rahimi, Mouad Abrini, Mahdi Khoramshahi, Mohamed Chetouani  

**Link**: [PDF](https://arxiv.org/pdf/2502.10642)  

**Abstract**: This paper investigates the performance of multimodal pre-trained models in user profiling tasks based on visual-linguistic demographic data. These models are critical for adapting to the needs and preferences of human users in social robotics, thereby providing personalized responses and enhancing interaction quality. First, we introduce two datasets specifically curated to represent demographic characteristics derived from user facial images. Next, we evaluate the performance of a prominent contrastive multimodal pre-trained model, CLIP, on these datasets, both in its out-of-the-box state and after fine-tuning. Initial results indicate that CLIP performs suboptimal in matching images to demographic descriptions without fine-tuning. Although fine-tuning significantly enhances its predictive capacity, the model continues to exhibit limitations in effectively generalizing subtle demographic nuances. To address this, we propose adopting a masked image modeling strategy to improve generalization and better capture subtle demographic attributes. This approach offers a pathway for enhancing demographic sensitivity in multimodal user modeling tasks. 

**Abstract (ZH)**: 基于视觉语言人口统计学数据的多模态预训练模型在用户画像任务中的性能研究 

---
# USER-VLM 360: Personalized Vision Language Models with User-aware Tuning for Social Human-Robot Interactions 

**Title (ZH)**: USER-VLM 360：面向社交人机交互的具有用户意识调整的个性化视觉语言模型 

**Authors**: Hamed Rahimi, Adil Bahaj, Mouad Abrini, Mahdi Khoramshahi, Mounir Ghogho, Mohamed Chetouani  

**Link**: [PDF](https://arxiv.org/pdf/2502.10636)  

**Abstract**: The integration of vision-language models into robotic systems constitutes a significant advancement in enabling machines to interact with their surroundings in a more intuitive manner. While VLMs offer rich multimodal reasoning, existing approaches lack user-specific adaptability, often relying on generic interaction paradigms that fail to account for individual behavioral, contextual, or socio-emotional nuances. When customization is attempted, ethical concerns arise from unmitigated biases in user data, risking exclusion or unfair treatment. To address these dual challenges, we propose User-VLM 360°, a holistic framework integrating multimodal user modeling with bias-aware optimization. Our approach features: (1) user-aware tuning that adapts interactions in real time using visual-linguistic signals; (2) bias mitigation via preference optimization; and (3) curated 360° socio-emotive interaction datasets annotated with demographic, emotion, and relational metadata. Evaluations across eight benchmarks demonstrate state-of-the-art results: +35.3% F1 in personalized VQA, +47.5% F1 in facial features understanding, 15% bias reduction, and 30X speedup over baselines. Ablation studies confirm component efficacy, and deployment on the Pepper robot validates real-time adaptability across diverse users. We open-source parameter-efficient 3B/10B models and an ethical verification framework for responsible adaptation. 

**Abstract (ZH)**: 用户导向的全方位视觉语言模型集成框架：结合多元用户建模与偏见意识优化 

---
# ProMRVL-CAD: Proactive Dialogue System with Multi-Round Vision-Language Interactions for Computer-Aided Diagnosis 

**Title (ZH)**: ProMRVL-CAD：基于多轮视觉-语言交互的主动对话系统用于计算机辅助诊断 

**Authors**: Xueshen Li, Xinlong Hou, Ziyi Huang, Yu Gan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10620)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated extraordinary comprehension capabilities with remarkable breakthroughs on various vision-language tasks. However, the application of LLMs in generating reliable medical diagnostic reports remains in the early stages. Currently, medical LLMs typically feature a passive interaction model where doctors respond to patient queries with little or no involvement in analyzing medical images. In contrast, some ChatBots simply respond to predefined queries based on visual inputs, lacking interactive dialogue or consideration of medical history. As such, there is a gap between LLM-generated patient-ChatBot interactions and those occurring in actual patient-doctor consultations. To bridge this gap, we develop an LLM-based dialogue system, namely proactive multi-round vision-language interactions for computer-aided diagnosis (ProMRVL-CAD), to generate patient-friendly disease diagnostic reports. The proposed ProMRVL-CAD system allows proactive dialogue to provide patients with constant and reliable medical access via an integration of knowledge graph into a recommendation system. Specifically, we devise two generators: a Proactive Question Generator (Pro-Q Gen) to generate proactive questions that guide the diagnostic procedure and a Multi-Vision Patient-Text Diagnostic Report Generator (MVP-DR Gen) to produce high-quality diagnostic reports. Evaluating two real-world publicly available datasets, MIMIC-CXR and IU-Xray, our model has better quality in generating medical reports. We further demonstrate the performance of ProMRVL achieves robust under the scenarios with low image quality. Moreover, we have created a synthetic medical dialogue dataset that simulates proactive diagnostic interactions between patients and doctors, serving as a valuable resource for training LLM. 

**Abstract (ZH)**: 最近大型语言模型在多模态医学诊断报告生成方面的进展已经在各种视觉-语言任务上取得了显著突破。然而，将大型语言模型应用于生成可靠的医学诊断报告仍处于早期阶段。目前，医学大型语言模型通常呈现一种被动的交互模型，医生对患者的查询进行回应，但很少参与到医学图像的分析中。相比之下，一些聊天机器人基于视觉输入仅对预定义的查询作出响应，缺乏互动对话或对医疗历史的考虑。因此，大型语言模型生成的患者-聊天机器人交互与实际患者-医生咨询之间存在差距。为了弥合这一差距，我们开发了一种基于大型语言模型的对话系统，即主动多轮视觉-语言交互辅助计算机辅助诊断（ProMRVL-CAD），以生成患者友好的疾病诊断报告。提出的ProMRVL-CAD系统通过将知识图谱整合到推荐系统中，实现主动对话，为患者提供持续可靠的医疗访问。具体而言，我们设计了两个生成器：主动问题生成器（Pro-Q Gen）以生成引导诊断流程的主动问题，以及多视图患者文本诊断报告生成器（MVP-DR Gen）以生成高质量的诊断报告。在对两个公开的现实世界数据集MIMIC-CXR和IU-Xray进行评估后，我们的模型在生成医学报告方面表现出更高的质量。此外，我们创建了一个合成医学对话数据集，模拟患者与医生之间的主动诊断交互，作为训练大型语言模型的宝贵资源。 

---
# Observer-Aware Probabilistic Planning Under Partial Observability 

**Title (ZH)**: 在部分可观测性条件下具有观察者意识的概率规划 

**Authors**: Salomé Lepers, Vincent Thomas, Olivier Buffet  

**Link**: [PDF](https://arxiv.org/pdf/2502.10568)  

**Abstract**: In this article, we are interested in planning problems where the agent is aware of the presence of an observer, and where this observer is in a partial observability situation. The agent has to choose its strategy so as to optimize the information transmitted by observations. Building on observer-aware Markov decision processes (OAMDPs), we propose a framework to handle this type of problems and thus formalize properties such as legibility, explicability and predictability. This extension of OAMDPs to partial observability can not only handle more realistic problems, but also permits considering dynamic hidden variables of interest. These dynamic target variables allow, for instance, working with predictability, or with legibility problems where the goal might change during execution. We discuss theoretical properties of PO-OAMDPs and, experimenting with benchmark problems, we analyze HSVI's convergence behavior with dedicated initializations and study the resulting strategies. 

**Abstract (ZH)**: 在这种情况下，代理意识到观察者的存在，并且观察者处于部分可观性状态。本文探讨了代理如何选择策略以优化通过观察传输的信息。基于观察者感知的马尔可夫决策过程(OAMDP)，我们提出了一种框架来处理此类问题，并由此正规定义了可读性、可解释性和可预测性等性质。将OAMDP扩展到部分可观性不仅能够处理更现实的问题，而且还允许考虑动态的隐藏变量。这些动态目标变量可以在预测问题或执行过程中目标发生变化的可读性问题中发挥作用。讨论了部分可观性-OAMDP的理论性质，并通过基准问题的实验分析了HSVIT策略收敛行为及其初始化策略的选用。 

---
# Benchmarking the rationality of AI decision making using the transitivity axiom 

**Title (ZH)**: 基于传递性公理对标准化AI决策合理性的benchmarking研究 

**Authors**: Kiwon Song, James M. Jennings III, Clintin P. Davis-Stober  

**Link**: [PDF](https://arxiv.org/pdf/2502.10554)  

**Abstract**: Fundamental choice axioms, such as transitivity of preference, provide testable conditions for determining whether human decision making is rational, i.e., consistent with a utility representation. Recent work has demonstrated that AI systems trained on human data can exhibit similar reasoning biases as humans and that AI can, in turn, bias human judgments through AI recommendation systems. We evaluate the rationality of AI responses via a series of choice experiments designed to evaluate transitivity of preference in humans. We considered ten versions of Meta's Llama 2 and 3 LLM models. We applied Bayesian model selection to evaluate whether these AI-generated choices violated two prominent models of transitivity. We found that the Llama 2 and 3 models generally satisfied transitivity, but when violations did occur, occurred only in the Chat/Instruct versions of the LLMs. We argue that rationality axioms, such as transitivity of preference, can be useful for evaluating and benchmarking the quality of AI-generated responses and provide a foundation for understanding computational rationality in AI systems more generally. 

**Abstract (ZH)**: 基础的选择公理，如偏好传递性，为确定人类决策是否合理，即是否符合效用表示，提供了可测试的条件。近期研究表明，基于人类数据训练的AI系统可能表现出与人类相似的推理偏见，并且AI系统可以通过推荐系统影响人类判断。我们通过一系列旨在评估人类偏好传递性的选择实验，评估AI响应的合理性。我们考虑了Meta的Llama 2和3个LLM模型的十个版本。我们采用贝叶斯模型选择方法评估这些AI生成的选择是否违反了两种典型的传递性模型。我们发现，Llama 2和3个模型总体上满足传递性，但在违反传递性的情况下，仅出现在LLM的Chat/指令版本中。我们论证认为，如偏好传递性这样的合理性公理，可用于评估和基准测试AI生成响应的质量，并为理解AI系统中的计算合理性奠定基础。 

---
# GraphiT: Efficient Node Classification on Text-Attributed Graphs with Prompt Optimized LLMs 

**Title (ZH)**: GraphiT: 使用提示优化大语言模型在文本属性图上的高效节点分类 

**Authors**: Shima Khoshraftar, Niaz Abedini, Amir Hajian  

**Link**: [PDF](https://arxiv.org/pdf/2502.10522)  

**Abstract**: The application of large language models (LLMs) to graph data has attracted a lot of attention recently. LLMs allow us to use deep contextual embeddings from pretrained models in text-attributed graphs, where shallow embeddings are often used for the text at- tributes of nodes. However, it is still challenging to efficiently en- code the graph structure and features into a sequential form for use by LLMs. In addition, the performance of an LLM alone, is highly dependent on the structure of the input prompt, which limits their effectiveness as a reliable approach and often requires iterative man- ual adjustments that could be slow, tedious and difficult to replicate programmatically. In this paper, we propose GraphiT (Graphs in Text), a framework for encoding graphs into a textual format and optimizing LLM prompts for graph prediction tasks. Here we focus on node classification for text-attributed graphs. We encode the graph data for every node and its neighborhood into a concise text to enable LLMs to better utilize the information in the graph. We then further programmatically optimize the LLM prompts us- ing the DSPy framework to automate this step and make it more efficient and reproducible. GraphiT outperforms our LLM-based baselines on three datasets and we show how the optimization step in GraphiT leads to measurably better results without manual prompt tweaking. We also demonstrated that our graph encoding approach is competitive to other graph encoding methods while being less expensive because it uses significantly less tokens for the same task. 

**Abstract (ZH)**: Large语言模型在图数据中的应用：GraphiT框架及其在图预测任务中的优化 

---
# A Self-Supervised Reinforcement Learning Approach for Fine-Tuning Large Language Models Using Cross-Attention Signals 

**Title (ZH)**: 基于跨注意力信号的自监督强化学习方法用于调整大型语言模型 

**Authors**: Andrew Kiruluta, Andreas Lemos, Priscilla Burity  

**Link**: [PDF](https://arxiv.org/pdf/2502.10482)  

**Abstract**: We propose a novel reinforcement learning framework for post training large language models that does not rely on human in the loop feedback. Instead, our approach uses cross attention signals within the model itself to derive a self supervised reward, thereby guiding iterative fine tuning of the model policy. By analyzing how the model attends to the input prompt during generation, we construct measures of prompt coverage, focus, and coherence. We then use these measures to rank or score candidate responses, providing a reward signal that encourages the model to produce well aligned, on topic text. In empirical comparisons against standard policy gradient methods and RL fine tuning with synthetic preference models, our method shows significant gains in prompt relevance and consistency over a non RL baseline. While it does not yet match the performance of fully human supervised RLHF systems, it highlights an important direction for scaling alignment with minimal human labeling. We provide a detailed analysis, discuss potential limitations, and outline future work for combining cross-attention based signals with smaller amounts of human feedback. 

**Abstract (ZH)**: 我们提出了一种用于后训练大型语言模型的新型强化学习框架，该框架不依赖于人类闭环反馈。相反，我们的方法利用模型内部的交叉注意力信号来推导出自我监督的奖励，从而引导模型策略的迭代微调。通过分析模型在生成过程中对输入提示的关注方式，我们构建了提示覆盖度、专注度和连贯性的度量标准。然后，我们利用这些度量标准对候选响应进行排名或评分，提供一种奖励信号，鼓励模型生成主题相关且一致的文本。与标准策略梯度方法和基于合成偏好模型的RL微调方法相比，我们的方法在提示相关性和一致性方面相对于非RL基线显示出显著的提升。尽管它尚未达到完全基于人类监督的RLHF系统的性能，但它强调了一个重要的发展方向，即通过最少的人类标签实现有效对齐。我们提供了详细分析，讨论了潜在限制，并概述了将交叉注意力基信号与较少的人类反馈结合的未来工作。 

---
# Knowledge Integration Strategies in Autonomous Vehicle Prediction and Planning: A Comprehensive Survey 

**Title (ZH)**: 自主驾驶车辆预测与规划中的知识集成策略：一项综合性综述 

**Authors**: Kumar Manas, Adrian Paschke  

**Link**: [PDF](https://arxiv.org/pdf/2502.10477)  

**Abstract**: This comprehensive survey examines the integration of knowledge-based approaches into autonomous driving systems, with a focus on trajectory prediction and planning. We systematically review methodologies for incorporating domain knowledge, traffic rules, and commonsense reasoning into these systems, spanning purely symbolic representations to hybrid neuro-symbolic architectures. In particular, we analyze recent advancements in formal logic and differential logic programming, reinforcement learning frameworks, and emerging techniques that leverage large foundation models and diffusion models for knowledge representation. Organized under a unified literature survey section, our discussion synthesizes the state-of-the-art into a high-level overview, supported by a detailed comparative table that maps key works to their respective methodological categories. This survey not only highlights current trends -- including the growing emphasis on interpretable AI, formal verification in safety-critical systems, and the increased use of generative models in prediction and planning -- but also outlines the challenges and opportunities for developing robust, knowledge-enhanced autonomous driving systems. 

**Abstract (ZH)**: 针对自主驾驶系统中知识导向方法的整合：路径预测与规划的研究综述 

---
# Multi-Objective Planning with Contextual Lexicographic Reward Preferences 

**Title (ZH)**: 基于背景列索尔夫优选奖励的多目标规划 

**Authors**: Pulkit Rustagi, Yashwanthi Anand, Sandhya Saisubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2502.10476)  

**Abstract**: Autonomous agents are often required to plan under multiple objectives whose preference ordering varies based on context. The agent may encounter multiple contexts during its course of operation, each imposing a distinct lexicographic ordering over the objectives, with potentially different reward functions associated with each context. Existing approaches to multi-objective planning typically consider a single preference ordering over the objectives, across the state space, and do not support planning under multiple objective orderings within an environment. We present Contextual Lexicographic Markov Decision Process (CLMDP), a framework that enables planning under varying lexicographic objective orderings, depending on the context. In a CLMDP, both the objective ordering at a state and the associated reward functions are determined by the context. We employ a Bayesian approach to infer a state-context mapping from expert trajectories. Our algorithm to solve a CLMDP first computes a policy for each objective ordering and then combines them into a single context-aware policy that is valid and cycle-free. The effectiveness of the proposed approach is evaluated in simulation and using a mobile robot. 

**Abstract (ZH)**: 基于上下文的字典序马尔可夫决策过程（Contextual Lexicographic Markov Decision Process） 

---
# Diverse Transformer Decoding for Offline Reinforcement Learning Using Financial Algorithmic Approaches 

**Title (ZH)**: 使用金融算法方法的离线强化学习多样化变压器解码 

**Authors**: Dan Elbaz, Oren Salzman  

**Link**: [PDF](https://arxiv.org/pdf/2502.10473)  

**Abstract**: Offline Reinforcement Learning (RL) algorithms learn a policy using a fixed training dataset, which is then deployed online to interact with the environment and make decisions. Transformers, a standard choice for modeling time-series data, are gaining popularity in offline RL. In this context, Beam Search (BS), an approximate inference algorithm, is the go-to decoding method. Offline RL eliminates the need for costly or risky online data collection. However, the restricted dataset induces uncertainty as the agent may encounter unfamiliar sequences of states and actions during execution that were not covered in the training data. In this context, BS lacks two important properties essential for offline RL: It does not account for the aforementioned uncertainty, and its greedy left-right search approach often results in sequences with minimal variations, failing to explore potentially better alternatives.
To address these limitations, we propose Portfolio Beam Search (PBS), a simple-yet-effective alternative to BS that balances exploration and exploitation within a Transformer model during decoding. We draw inspiration from financial economics and apply these principles to develop an uncertainty-aware diversification mechanism, which we integrate into a sequential decoding algorithm at inference time. We empirically demonstrate the effectiveness of PBS on the D4RL locomotion benchmark, where it achieves higher returns and significantly reduces outcome variability. 

**Abstract (ZH)**: 基于Transformer的离线强化学习束搜索算法：Portfolio Beam Search 

---
# AI Alignment at Your Discretion 

**Title (ZH)**: AI对齐由你做主 

**Authors**: Maarten Buyl, Hadi Khalaf, Claudio Mayrink Verdun, Lucas Monteiro Paes, Caio C. Vieira Machado, Flavio du Pin Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2502.10441)  

**Abstract**: In AI alignment, extensive latitude must be granted to annotators, either human or algorithmic, to judge which model outputs are `better' or `safer.' We refer to this latitude as alignment discretion. Such discretion remains largely unexamined, posing two risks: (i) annotators may use their power of discretion arbitrarily, and (ii) models may fail to mimic this discretion. To study this phenomenon, we draw on legal concepts of discretion that structure how decision-making authority is conferred and exercised, particularly in cases where principles conflict or their application is unclear or irrelevant. Extended to AI alignment, discretion is required when alignment principles and rules are (inevitably) conflicting or indecisive. We present a set of metrics to systematically analyze when and how discretion in AI alignment is exercised, such that both risks (i) and (ii) can be observed. Moreover, we distinguish between human and algorithmic discretion and analyze the discrepancy between them. By measuring both human and algorithmic discretion over safety alignment datasets, we reveal layers of discretion in the alignment process that were previously unaccounted for. Furthermore, we demonstrate how algorithms trained on these datasets develop their own forms of discretion in interpreting and applying these principles, which challenges the purpose of having any principles at all. Our paper presents the first step towards formalizing this core gap in current alignment processes, and we call on the community to further scrutinize and control alignment discretion. 

**Abstract (ZH)**: 在AI对齐中，必须给予标注者（无论是人类还是算法）广泛的裁量权，以判断哪些模型输出是“更好”或“更安全”的。我们称这种裁量权为对齐裁量。这种裁量权目前尚未得到充分研究，存在两大风险：（i）标注者可能随意使用其裁量权；（ii）模型可能无法模仿这种裁量。为研究这一现象，我们借鉴法律中的裁量权概念，该概念规定了决策权如何授予和行使，尤其是在原则冲突或其应用不明确或不相关的情况下。延伸至AI对齐，当对齐原则和规则不可避免地存在冲突或不明确时，就需要这种裁量权。我们提出了一套指标，以系统分析AI对齐中何时及如何行使裁量权，从而使上述两种风险得以观察。此外，我们区分了人类和算法裁量，并分析了它们之间的差异。通过测量人类和算法在安全对齐数据集上的裁量权，我们揭示了对齐过程中此前未被考虑的多层裁量机制。此外，我们展示了这些数据集上的算法如何发展出它们自己的裁量形式，以解读和应用这些原则，这挑战了制定任何原则的初衷。我们的论文代表了朝着正式化当前对齐过程中的这一核心缺口迈出的第一步，并呼吁社区进一步审视和控制对齐裁量。 

---
# Agency in Artificial Intelligence Systems 

**Title (ZH)**: 人工智能系统的代理权 

**Authors**: Parashar Das  

**Link**: [PDF](https://arxiv.org/pdf/2502.10434)  

**Abstract**: There is a general concern that present developments in artificial intelligence (AI) research will lead to sentient AI systems, and these may pose an existential threat to humanity. But why cannot sentient AI systems benefit humanity instead? This paper endeavours to put this question in a tractable manner. I ask whether a putative AI system will develop an altruistic or a malicious disposition towards our society, or what would be the nature of its agency? Given that AI systems are being developed into formidable problem solvers, we can reasonably expect these systems to preferentially take on conscious aspects of human problem solving. I identify the relevant phenomenal aspects of agency in human problem solving. The functional aspects of conscious agency can be monitored using tools provided by functionalist theories of consciousness. A recent expert report (Butlin et al. 2023) has identified functionalist indicators of agency based on these theories. I show how to use the Integrated Information Theory (IIT) of consciousness, to monitor the phenomenal nature of this agency. If we are able to monitor the agency of AI systems as they develop, then we can dissuade them from becoming a menace to society while encouraging them to be an aid. 

**Abstract (ZH)**: 人工智能研究的发展可能导致有感知能力的AI系统出现，这些系统可能对人类构成生存威胁，但它们能否反而惠及人类？本文试图将这一问题具体化。本文探讨一个假设的AI系统将倾向于对社会产生利他主义还是恶意倾向，或者它的agency的本质是什么？考虑到AI系统正被开发成为强大的问题解决者，我们有理由预期这些系统将优先采用人类问题解决过程中的意识方面。本文识别了人类问题解决过程中的相关现象学方面的agency。功能主义意识理论提供的工具可以用于监测意识agency的功能方面。最近的一份专家报告（Butlin等人，2023）基于这些理论识别了agency的功能性指标。本文展示如何利用综合信息理论（IIT）来监测这种agency的现象学性质。如果我们能够监测AI系统在发展过程中的agency，那么我们就能阻止它们对社会构成威胁，同时鼓励它们成为一种助力。 

---
# Dynamic Chain-of-Thought: Towards Adaptive Deep Reasoning 

**Title (ZH)**: 动态思维链：迈向自适应深度推理 

**Authors**: Libo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10428)  

**Abstract**: To reduce the cost and consumption of computing resources caused by computational redundancy and delayed reward assignment in long CoT, this research proposes the dynamic chain-of-thought with adaptive reasoning time and steps. The researcher used simulation experiment to simulate the integration of D-CoT through Python 3.13 IDLE combined with a Python simulator based on GPTs. At the same time, the researcher used DeepSeek R1 as a control group to test and compare the performance of the D-CoT simulator in processing MIT OpenCourseWare's linear algebra exam questions. Experimental results show that D-CoT is better than DeepSeek R1 based on long CoT in three indicators: reasoning time, CoT length (reasoning steps) and token count, which achieves a significant reduction in computing resource consumption. In addition, this research has potential value in deep reasoning optimization and can be used as a reference for future dynamic deep reasoning frameworks. 

**Abstract (ZH)**: 为了减少由计算冗余和延迟奖励分配在长CoT中导致的计算成本和资源消耗，本文提出了一种具有自适应推理时间和步数的动态链式思考方法。研究人员通过结合基于GPT的Python模拟器和Python 3.13 IDLE进行了仿真实验，将D-CoT进行模拟。同时，研究人员使用DeepSeek R1作为对照组，测试并比较了D-CoT模拟器在处理MIT OpenCourseWare线性代数考试问题时的表现。实验结果表明，D-CoT在推理时间、CoT长度（推理步数）和令牌计数三个指标上优于DeepSeek R1，实现了显著的计算资源消耗减少。此外，本文在深推理优化方面具有潜在价值，并可作为未来动态深推理框架的参考。 

---
# Position: Stop Acting Like Language Model Agents Are Normal Agents 

**Title (ZH)**: 位置：停止将语言模型代理视为正常代理 

**Authors**: Elija Perrier, Michael Timothy Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2502.10420)  

**Abstract**: Language Model Agents (LMAs) are increasingly treated as capable of autonomously navigating interactions with humans and tools. Their design and deployment tends to presume they are normal agents capable of sustaining coherent goals, adapting across contexts and acting with a measure of intentionality. These assumptions are critical to prospective use cases in industrial, social and governmental settings. But LMAs are not normal agents. They inherit the structural problems of the large language models (LLMs) around which they are built: hallucinations, jailbreaking, misalignment and unpredictability. In this Position paper we argue LMAs should not be treated as normal agents, because doing so leads to problems that undermine their utility and trustworthiness. We enumerate pathologies of agency intrinsic to LMAs. Despite scaffolding such as external memory and tools, they remain ontologically stateless, stochastic, semantically sensitive, and linguistically intermediated. These pathologies destabilise the ontological properties of LMAs including identifiability, continuity, persistence and and consistency, problematising their claim to agency. In response, we argue LMA ontological properties should be measured before, during and after deployment so that the negative effects of pathologies can be mitigated. 

**Abstract (ZH)**: 语言模型代理（LMAs）不应被视为正常代理：关于其本体论属性的测量与路径学说的学术探讨 

---
# A Coordination-based Approach for Focused Learning in Knowledge-Based Systems 

**Title (ZH)**: 基于协调的聚焦学习方法在知识系统的应用 

**Authors**: Abhishek Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2502.10394)  

**Abstract**: Recent progress in Learning by Reading and Machine Reading systems has significantly increased the capacity of knowledge-based systems to learn new facts. In this work, we discuss the problem of selecting a set of learning requests for these knowledge-based systems which would lead to maximum Q/A performance. To understand the dynamics of this problem, we simulate the properties of a learning strategy, which sends learning requests to an external knowledge source. We show that choosing an optimal set of facts for these learning systems is similar to a coordination game, and use reinforcement learning to solve this problem. Experiments show that such an approach can significantly improve Q/A performance. 

**Abstract (ZH)**: Recent进展在阅读学习和机器阅读系统中的知识基于系统学习新事实的能力显著提高。在此工作中，我们讨论了为这些知识基于系统选择一组学习请求的问题，这些请求将导致最大的问答性能。为了理解这个问题的动力学，我们模拟了一种学习策略的属性，该策略向外部知识源发送学习请求。我们证明，为这些学习系统选择一组最优事实类似于一种协调博弈，并使用强化学习来解决这个问题。实验表明，这种方法可以显著提高问答性能。 

---
# Diffusion Models without Classifier-free Guidance 

**Title (ZH)**: 没有分类器自由指导的扩散模型 

**Authors**: Zhicong Tang, Jianmin Bao, Dong Chen, Baining Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.12154)  

**Abstract**: This paper presents Model-guidance (MG), a novel objective for training diffusion model that addresses and removes of the commonly used Classifier-free guidance (CFG). Our innovative approach transcends the standard modeling of solely data distribution to incorporating the posterior probability of conditions. The proposed technique originates from the idea of CFG and is easy yet effective, making it a plug-and-play module for existing models. Our method significantly accelerates the training process, doubles the inference speed, and achieve exceptional quality that parallel and even surpass concurrent diffusion models with CFG. Extensive experiments demonstrate the effectiveness, efficiency, scalability on different models and datasets. Finally, we establish state-of-the-art performance on ImageNet 256 benchmarks with an FID of 1.34. Our code is available at this https URL. 

**Abstract (ZH)**: 本文提出了模型引导（MG），这是一种新颖的目标，用于训练扩散模型，解决了和消除了常用的无分类器引导（CFG）。我们的创新方法超越了仅基于数据分布的标准建模，而是将条件的后概率包含进来。该提出的技术源自CFG的想法，简单有效，可作为现有模型的即插即用模块。我们的方法显著加速了训练过程，将推理速度提高了两倍，并达到了与使用CFG的并发扩散模型相当甚至更优的质量。大量实验在不同模型和数据集上证明了其有效性、效率和可扩展性。最后，我们在ImageNet 256基准上建立了最先进的性能，FID值为1.34。代码已在此处提供：this https URL。 

---
# HARBOR: Exploring Persona Dynamics in Multi-Agent Competition 

**Title (ZH)**: HARBOR: 探索多智能体竞争中的角色动态 

**Authors**: Kenan Jiang, Li Xiong, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12149)  

**Abstract**: We investigate factors contributing to LLM agents' success in competitive multi-agent environments, using auctions as a testbed where agents bid to maximize profit. The agents are equipped with bidding domain knowledge, distinct personas that reflect item preferences, and a memory of auction history. Our work extends the classic auction scenario by creating a realistic environment where multiple agents bid on houses, weighing aspects such as size, location, and budget to secure the most desirable homes at the lowest prices. Particularly, we investigate three key questions: (a) How does a persona influence an agent's behavior in a competitive setting? (b) Can an agent effectively profile its competitors' behavior during auctions? (c) How can persona profiling be leveraged to create an advantage using strategies such as theory of mind? Through a series of experiments, we analyze the behaviors of LLM agents and shed light on new findings. Our testbed, called HARBOR, offers a valuable platform for deepening our understanding of multi-agent workflows in competitive environments. 

**Abstract (ZH)**: 我们使用拍卖作为测试场景，研究影响大语言模型代理在竞争性多代理环境中的成功因素，代理具备投标领域的知识、反映物品偏好的不同身份 persona，以及拍卖历史的记忆。我们的工作扩展了经典的拍卖场景，通过创造一个现实的环境，让多个代理竞拍房屋，并权衡大小、位置和预算等因素，以实现最低成本获取最理想住宅。特别地，我们探讨了三个关键问题：(a) 身份如何影响代理在竞争性环境中的行为？(b) 代理能否在拍卖过程中有效分析竞争对手的行为？(c) 如何利用身份分析的优势，通过考虑心理理论等策略来建立优势？通过一系列实验，我们分析了大语言模型代理的行为，并揭示了新的发现。我们的测试平台 HARBOR 为深化对竞争环境中多代理工作流的理解提供了宝贵平台。 

---
# Fast or Better? Balancing Accuracy and Cost in Retrieval-Augmented Generation with Flexible User Control 

**Title (ZH)**: 快还是好？在具有灵活用户控制的检索增强生成中平衡准确性和成本 

**Authors**: Jinyan Su, Jennifer Healey, Preslav Nakov, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2502.12145)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful approach to mitigate large language model (LLM) hallucinations by incorporating external knowledge retrieval. However, existing RAG frameworks often apply retrieval indiscriminately,leading to inefficiencies-over-retrieving when unnecessary or failing to retrieve iteratively when required for complex reasoning. Recent adaptive retrieval strategies, though adaptively navigates these retrieval strategies, predict only based on query complexity and lacks user-driven flexibility, making them infeasible for diverse user application needs. In this paper, we introduce a novel user-controllable RAG framework that enables dynamic adjustment of the accuracy-cost trade-off. Our approach leverages two classifiers: one trained to prioritize accuracy and another to prioritize retrieval efficiency. Via an interpretable control parameter $\alpha$, users can seamlessly navigate between minimal-cost retrieval and high-accuracy retrieval based on their specific requirements. We empirically demonstrate that our approach effectively balances accuracy, retrieval cost, and user controllability, making it a practical and adaptable solution for real-world applications. 

**Abstract (ZH)**: 基于检索增强生成的用户可控框架：平衡准确性和检索成本 

---
# LaM-SLidE: Latent Space Modeling of Spatial Dynamical Systems via Linked Entities 

**Title (ZH)**: LaM-SLidE: 联邦实体驱动的空间动态系统潜在空间建模 

**Authors**: Florian Sestak, Artur Toshev, Andreas Fürst, Günter Klambauer, Andreas Mayr, Johannes Brandstetter  

**Link**: [PDF](https://arxiv.org/pdf/2502.12128)  

**Abstract**: Generative models are spearheading recent progress in deep learning, showing strong promise for trajectory sampling in dynamical systems as well. However, while latent space modeling paradigms have transformed image and video generation, similar approaches are more difficult for most dynamical systems. Such systems -- from chemical molecule structures to collective human behavior -- are described by interactions of entities, making them inherently linked to connectivity patterns and the traceability of entities over time. Our approach, LaM-SLidE (Latent Space Modeling of Spatial Dynamical Systems via Linked Entities), combines the advantages of graph neural networks, i.e., the traceability of entities across time-steps, with the efficiency and scalability of recent advances in image and video generation, where pre-trained encoder and decoder are frozen to enable generative modeling in the latent space. The core idea of LaM-SLidE is to introduce identifier representations (IDs) to allow for retrieval of entity properties, e.g., entity coordinates, from latent system representations and thus enables traceability. Experimentally, across different domains, we show that LaM-SLidE performs favorably in terms of speed, accuracy, and generalizability. (Code is available at this https URL) 

**Abstract (ZH)**: 基于连接实体的空间动力系统隐空间建模：LaM-SLidE方法 

---
# LLMs on the Line: Data Determines Loss-to-Loss Scaling Laws 

**Title (ZH)**: LLMs在线性关系中的数据决定损失缩放定律 

**Authors**: Prasanna Mayilvahanan, Thaddäus Wiedemer, Sayak Mallick, Matthias Bethge, Wieland Brendel  

**Link**: [PDF](https://arxiv.org/pdf/2502.12120)  

**Abstract**: Scaling laws guide the development of large language models (LLMs) by offering estimates for the optimal balance of model size, tokens, and compute. More recently, loss-to-loss scaling laws that relate losses across pretraining datasets and downstream tasks have emerged as a powerful tool for understanding and improving LLM performance. In this work, we investigate which factors most strongly influence loss-to-loss scaling. Our experiments reveal that the pretraining data and tokenizer determine the scaling trend. In contrast, model size, optimization hyperparameters, and even significant architectural differences, such as between transformer-based models like Llama and state-space models like Mamba, have limited impact. Consequently, practitioners should carefully curate suitable pretraining datasets for optimal downstream performance, while architectures and other settings can be freely optimized for training efficiency. 

**Abstract (ZH)**: Scaling定律指导大规模语言模型（LLMs）的发展，并提供了模型大小、标记和计算最优平衡的估计。近年来，损失到损失的Scaling定律逐渐成为理解并提升LLM性能的有力工具，这些定律关联了预训练数据集和下游任务之间的损失。在本工作中，我们研究了哪些因素对损失到损失的Scaling影响最大。我们的实验揭示出，预训练数据和分词器决定了Scaling趋势，而模型大小、优化超参数，甚至像Llama这样的变换器模型和Mamba这样的状态空间模型之间显著的架构差异，对Scaling的影响有限。因此，从业者应精心选择适合的预训练数据集以实现最佳的下游性能，而架构和其他设置可以自由优化以提高训练效率。 

---
# PRISM: Self-Pruning Intrinsic Selection Method for Training-Free Multimodal Data Selection 

**Title (ZH)**: PRISM: 自 pruning 内在选择方法用于无训练多模态数据选择 

**Authors**: Jinhe Bi, Yifan Wang, Danqi Yan, Xun Xiao, Artur Hecker, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.12119)  

**Abstract**: Visual instruction tuning refines pre-trained Multimodal Large Language Models (MLLMs) to enhance their real-world task performance. However, the rapid expansion of visual instruction datasets introduces significant data redundancy, leading to excessive computational costs. Existing data selection methods predominantly rely on proxy models or loss-based metrics, both of which impose substantial computational overheads due to the necessity of model inference and backpropagation. To address this challenge, we propose PRISM, a novel training-free approach for efficient multimodal data selection. Unlike existing methods, PRISM eliminates the reliance on proxy models, warm-up pretraining, and gradient-based optimization. Instead, it leverages Pearson correlation analysis to quantify the intrinsic visual encoding properties of MLLMs, computing a task-specific correlation score to identify high-value instances. This not only enbles data-efficient selection,but maintains the original performance. Empirical evaluations across multiple MLLMs demonstrate that PRISM reduces the overall time required for visual instruction tuning and data selection to just 30% of conventional methods, while surpassing fully fine-tuned models across eight multimodal and three language understanding benchmarks, achieving a 101.7% relative improvement in final performance. 

**Abstract (ZH)**: 视觉指令调优通过细化预训练多模态大型语言模型来提升其实用任务性能。然而，视觉指令数据集的迅速扩张导致了显著的数据冗余，增加了过高的计算成本。现有的数据选择方法主要依赖代理模型或基于损失的指标，两者都因需要模型推理和反向传播而产生了显著的计算开销。为应对这一挑战，我们提出PRISM，这是一种新型的无需训练的高效多模态数据选择方法。与现有方法不同，PRISM 杜绝了对代理模型、预热微调和梯度优化的依赖，而是利用皮尔逊相关分析来量化多模态大型语言模型的固有视觉编码特性，计算特定任务的相关分数以识别高价值样本。这一方法不仅实现了高效的数据选择，还能保持原始性能。多项实验验证了PRISM在减少视觉指令调优和数据选择所需时间方面的优势，使其仅为传统方法的30%，同时在八个多模态和三个语言理解基准测试中超过了完全微调的模型，实现了101.7%的相对性能提升。 

---
# Personality Structured Interview for Large Language Model Simulation in Personality Research 

**Title (ZH)**: 基于人格结构化面试的大语言模型人格研究模拟 

**Authors**: Pengda Wang, Huiqi Zou, Hanjie Chen, Tianjun Sun, Ziang Xiao, Frederick L. Oswald  

**Link**: [PDF](https://arxiv.org/pdf/2502.12109)  

**Abstract**: Although psychometrics researchers have recently explored the use of large language models (LLMs) as proxies for human participants, LLMs often fail to generate heterogeneous data with human-like diversity, which diminishes their value in advancing social science research. To address these challenges, we explored the potential of the theory-informed Personality Structured Interview (PSI) as a tool for simulating human responses in personality research. In this approach, the simulation is grounded in nuanced real-human interview transcripts that target the personality construct of interest. We have provided a growing set of 357 structured interview transcripts from a representative sample, each containing an individual's response to 32 open-ended questions carefully designed to gather theory-based personality evidence. Additionally, grounded in psychometric research, we have summarized an evaluation framework to systematically validate LLM-generated psychometric data. Results from three experiments demonstrate that well-designed structured interviews could improve human-like heterogeneity in LLM-simulated personality data and predict personality-related behavioral outcomes (i.e., organizational citizenship behaviors and counterproductive work behavior). We further discuss the role of theory-informed structured interviews in LLM-based simulation and outline a general framework for designing structured interviews to simulate human-like data for psychometric research. 

**Abstract (ZH)**: 尽管心理测量研究人员最近探索了使用大型语言模型（LLMs）作为人类参与者代理的可能性，但LLMs常常无法生成具有人类多样性的人类异质性数据，这减弱了它们在推进社会科学研究中的价值。为了应对这些挑战，我们探讨了理论指导的个性结构访谈（PSI）作为在个性研究中模拟人类响应的工具的潜力。在该方法中，模拟基于针对感兴趣的人格结构的细致的人类访谈转录。我们提供了一套不断增长的357个结构化访谈转录，来自代表性样本，每个转录包含个人对32个精心设计的开放性问题的响应，以收集基于理论的人格证据。此外，基于心理测量研究，我们总结了一个评估框架，以系统地验证LLM生成的心理测量数据的有效性。三项实验的结果表明，精心设计的结构化访谈可以提高LLM模拟的人格数据的人类异质性，并预测与人格相关的行为结果（即组织公民行为和反生产性工作行为）。我们进一步讨论了理论指导的结构化访谈在LLM基于的模拟中的作用，并概述了设计结构化访谈以生成用于心理测量研究的人类样数据的一般框架。 

---
# Using the Path of Least Resistance to Explain Deep Networks 

**Title (ZH)**: 用最小阻力路径解释深度网络 

**Authors**: Sina Salek, Joseph Enguehard  

**Link**: [PDF](https://arxiv.org/pdf/2502.12108)  

**Abstract**: Integrated Gradients (IG), a widely used axiomatic path-based attribution method, assigns importance scores to input features by integrating model gradients along a straight path from a baseline to the input. While effective in some cases, we show that straight paths can lead to flawed attributions. In this paper, we identify the cause of these misattributions and propose an alternative approach that treats the input space as a Riemannian manifold, computing attributions by integrating gradients along geodesics. We call this method Geodesic Integrated Gradients (GIG). To approximate geodesic paths, we introduce two techniques: a k-Nearest Neighbours-based approach for smaller models and a Stochastic Variational Inference-based method for larger ones. Additionally, we propose a new axiom, Strong Completeness, extending the axioms satisfied by IG. We show that this property is desirable for attribution methods and that GIG is the only method that satisfies it. Through experiments on both synthetic and real-world data, we demonstrate that GIG outperforms existing explainability methods, including IG. 

**Abstract (ZH)**: 基于测地线的综合梯度（Geodesic Integrated Gradients, GIG）方法：一种将输入空间视为黎曼流形、沿测地线积分梯度的归因方法 

---
# Meta-Statistical Learning: Supervised Learning of Statistical Inference 

**Title (ZH)**: 元统计学习：统计推理的监督学习 

**Authors**: Maxime Peyrard, Kyunghyun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2502.12088)  

**Abstract**: This work demonstrates that the tools and principles driving the success of large language models (LLMs) can be repurposed to tackle distribution-level tasks, where the goal is to predict properties of the data-generating distribution rather than labels for individual datapoints. These tasks encompass statistical inference problems such as parameter estimation, hypothesis testing, or mutual information estimation. Framing these tasks within traditional machine learning pipelines is challenging, as supervision is typically tied to individual datapoint. We propose meta-statistical learning, a framework inspired by multi-instance learning that reformulates statistical inference tasks as supervised learning problems. In this approach, entire datasets are treated as single inputs to neural networks, which predict distribution-level parameters. Transformer-based architectures, without positional encoding, provide a natural fit due to their permutation-invariance properties. By training on large-scale synthetic datasets, meta-statistical models can leverage the scalability and optimization infrastructure of Transformer-based LLMs. We demonstrate the framework's versatility with applications in hypothesis testing and mutual information estimation, showing strong performance, particularly for small datasets where traditional neural methods struggle. 

**Abstract (ZH)**: 大型语言模型的工具和原理可以重新用于处理分布级任务，这些任务的目标是预测数据生成分布的属性而不是个别数据点的标签。这些问题包括参数估计、假设检验或互信息估计等统计推理问题。将这些任务纳入传统的机器学习管道中具有挑战性，因为监督通常与个别数据点相关。我们提出了元统计学习框架，该框架借鉴多实例学习的概念，将统计推理任务重新表述为监督学习问题。在此方法中，整个数据集被视为神经网络的一个输入，以预测分布级参数。由于其置换不变性特性，基于 Transformer 的架构在没有位置编码的情况下提供了自然的契合。通过在大规模合成数据集上训练，元统计模型可以利用基于 Transformer 的大型语言模型的可扩展性和优化基础设施。我们通过在假设检验和互信息估计的应用中展示该框架的 versatility，显示出在传统神经方法在小数据集上表现不佳的情况下，具有强大性能。 

---
# TokenSkip: Controllable Chain-of-Thought Compression in LLMs 

**Title (ZH)**: TokenSkip: LLM中可控的链式思维压缩 

**Authors**: Heming Xia, Yongqi Li, Chak Tou Leong, Wenjie Wang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12067)  

**Abstract**: Chain-of-Thought (CoT) has been proven effective in enhancing the reasoning capabilities of large language models (LLMs). Recent advancements, such as OpenAI's o1 and DeepSeek-R1, suggest that scaling up the length of CoT sequences during inference could further boost LLM reasoning performance. However, due to the autoregressive nature of LLM decoding, longer CoT outputs lead to a linear increase in inference latency, adversely affecting user experience, particularly when the CoT exceeds 10,000 tokens. To address this limitation, we analyze the semantic importance of tokens within CoT outputs and reveal that their contributions to reasoning vary. Building on this insight, we propose TokenSkip, a simple yet effective approach that enables LLMs to selectively skip less important tokens, allowing for controllable CoT compression. Extensive experiments across various models and tasks demonstrate the effectiveness of TokenSkip in reducing CoT token usage while preserving strong reasoning performance. Notably, when applied to Qwen2.5-14B-Instruct, TokenSkip reduces reasoning tokens by 40% (from 313 to 181) on GSM8K, with less than a 0.4% performance drop. 

**Abstract (ZH)**: Chain-of-Thought压缩方法TokenSkip在增强大型语言模型推理性能中的应用 

---
# AI-generated Text Detection with a GLTR-based Approach 

**Title (ZH)**: 基于GLTR方法的AI生成文本检测 

**Authors**: Lucía Yan Wu, Isabel Segura-Bedmar  

**Link**: [PDF](https://arxiv.org/pdf/2502.12064)  

**Abstract**: The rise of LLMs (Large Language Models) has contributed to the improved performance and development of cutting-edge NLP applications. However, these can also pose risks when used maliciously, such as spreading fake news, harmful content, impersonating individuals, or facilitating school plagiarism, among others. This is because LLMs can generate high-quality texts, which are challenging to differentiate from those written by humans. GLTR, which stands for Giant Language Model Test Room and was developed jointly by the MIT-IBM Watson AI Lab and HarvardNLP, is a visual tool designed to help detect machine-generated texts based on GPT-2, that highlights the words in text depending on the probability that they were machine-generated. One limitation of GLTR is that the results it returns can sometimes be ambiguous and lead to confusion. This study aims to explore various ways to improve GLTR's effectiveness for detecting AI-generated texts within the context of the IberLef-AuTexTification 2023 shared task, in both English and Spanish languages. Experiment results show that our GLTR-based GPT-2 model overcomes the state-of-the-art models on the English dataset with a macro F1-score of 80.19%, except for the first ranking model (80.91%). However, for the Spanish dataset, we obtained a macro F1-score of 66.20%, which differs by 4.57% compared to the top-performing model. 

**Abstract (ZH)**: 大型语言模型的兴起促进了先进自然语言处理应用的性能提升和发展。然而，这些模型也可能在恶意使用时带来风险，例如传播假新闻、发布有害内容、冒充个人或协助学术抄袭等。这是因为大型语言模型能够生成高质量的文本，这些文本难以与人类撰写的区分开来。GLTR（即大型语言模型测试室，由麻省理工学院IBM沃森人工智能实验室和哈佛NLP联合开发）是一款可视化工具，旨在基于GPT-2帮助检测机器生成的文本，并根据生成概率突出显示文本中的词语。GLTR的一个局限性是，其返回的结果有时可能会模棱两可，导致混淆。本文旨在探索在2023年伊贝勒夫-奥特西文本共享任务中提高GLTR检测AI生成文本有效性的多种方法，涵盖英语和西班牙语。实验结果表明，基于GLTR的GPT-2模型在英语数据集上的macro F1得分为80.19%，略低于排名第一的模型（80.91%）。然而，对于西班牙语数据集，我们获得了macro F1得分为66.20%，比表现最佳模型低4.57%。 

---
# Masked Latent Prediction and Classification for Self-Supervised Audio Representation Learning 

**Title (ZH)**: 掩码潜变量预测与分类用于自监督音频表示学习 

**Authors**: Aurian Quelennec, Pierre Chouteau, Geoffroy Peeters, Slim Essid  

**Link**: [PDF](https://arxiv.org/pdf/2502.12031)  

**Abstract**: Recently, self-supervised learning methods based on masked latent prediction have proven to encode input data into powerful representations. However, during training, the learned latent space can be further transformed to extract higher-level information that could be more suited for downstream classification tasks. Therefore, we propose a new method: MAsked latenT Prediction And Classification (MATPAC), which is trained with two pretext tasks solved jointly. As in previous work, the first pretext task is a masked latent prediction task, ensuring a robust input representation in the latent space. The second one is unsupervised classification, which utilises the latent representations of the first pretext task to match probability distributions between a teacher and a student. We validate the MATPAC method by comparing it to other state-of-the-art proposals and conducting ablations studies. MATPAC reaches state-of-the-art self-supervised learning results on reference audio classification datasets such as OpenMIC, GTZAN, ESC-50 and US8K and outperforms comparable supervised methods results for musical auto-tagging on Magna-tag-a-tune. 

**Abstract (ZH)**: 基于掩码潜在预测的方法：联合预训练任务的潜在预测与分类 (MAsked latenT Prediction And Classification, MATPAC) 

---
# Teaching LLMs According to Their Aptitude: Adaptive Reasoning for Mathematical Problem Solving 

**Title (ZH)**: 根据其才能教学LLMs：适应性推理在数学问题解决中的应用 

**Authors**: Xin Xu, Yan Xu, Tianhao Chen, Yuchen Yan, Chengwu Liu, Zaoyu Chen, Yufei Wang, Yichun Yin, Yasheng Wang, Lifeng Shang, Qun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12022)  

**Abstract**: Existing approaches to mathematical reasoning with large language models (LLMs) rely on Chain-of-Thought (CoT) for generalizability or Tool-Integrated Reasoning (TIR) for precise computation. While efforts have been made to combine these methods, they primarily rely on post-selection or predefined strategies, leaving an open question: whether LLMs can autonomously adapt their reasoning strategy based on their inherent capabilities. In this work, we propose TATA (Teaching LLMs According to Their Aptitude), an adaptive framework that enables LLMs to personalize their reasoning strategy spontaneously, aligning it with their intrinsic aptitude. TATA incorporates base-LLM-aware data selection during supervised fine-tuning (SFT) to tailor training data to the model's unique abilities. This approach equips LLMs to autonomously determine and apply the appropriate reasoning strategy at test time. We evaluate TATA through extensive experiments on six mathematical reasoning benchmarks, using both general-purpose and math-specialized LLMs. Empirical results demonstrate that TATA effectively combines the complementary strengths of CoT and TIR, achieving superior or comparable performance with improved inference efficiency compared to TIR alone. Further analysis underscores the critical role of aptitude-aware data selection in enabling LLMs to make effective and adaptive reasoning decisions and align reasoning strategies with model capabilities. 

**Abstract (ZH)**: 基于大型语言模型的数学推理现有方法依赖于Chain-of-Thought (CoT)以实现泛化能力或依赖于Tool-Integrated Reasoning (TIR)以实现精确计算。尽管已经尝试将这些方法结合起来，但它们主要依赖于事后选择或预定义策略，留给一个开放的问题：LLMs 是否可以根据其固有的能力自主适应其推理策略。在本文中，我们提出了TATA（根据LLM能力进行教学）这一自适应框架，使LLMs能够自发地个性化其推理策略，使其与内在能力相匹配。TATA在监督微调（SFT）期间结合了对基模型的意识数据选择，以根据模型的独特能力调整训练数据。该方法使LLMs能够在测试时自主确定并应用适当的推理策略。我们通过在六个数学推理基准上进行广泛实验，使用通用和数学专业化LSTM，评估了TATA。实验证明，TATA有效地结合了CoT和TIR的优点，在推理效率方面优于单独使用TIR，进一步的分析强调了意识能力的数据选择在使LLMs能够做出有效和自适应的推理决策以及使推理策略与模型能力相一致方面的重要性。 

---
# Atom of Thoughts for Markov LLM Test-Time Scaling 

**Title (ZH)**: Markov LLM测试时缩放的原子思想 

**Authors**: Fengwei Teng, Zhaoyang Yu, Quan Shi, Jiayi Zhang, Chenglin Wu, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.12018)  

**Abstract**: Large Language Models (LLMs) achieve superior performance through training-time scaling, and test-time scaling further enhances their capabilities by conducting effective reasoning during inference. However, as the scale of reasoning increases, existing test-time scaling methods suffer from accumulated historical information, which not only wastes computational resources but also interferes with effective reasoning. To address this issue, we observe that complex reasoning progress is often achieved by solving a sequence of independent subquestions, each being self-contained and verifiable. These subquestions are essentially atomic questions, relying primarily on their current state rather than accumulated history, similar to the memoryless transitions in a Markov process. Based on this observation, we propose Atom of Thoughts (AoT), where each state transition in the reasoning process consists of decomposing the current question into a dependency-based directed acyclic graph and contracting its subquestions, forming a new atomic question state. This iterative decomposition-contraction process continues until reaching directly solvable atomic questions, naturally realizing Markov transitions between question states. Furthermore, these atomic questions can be seamlessly integrated into existing test-time scaling methods, enabling AoT to serve as a plug-in enhancement for improving reasoning capabilities. Experiments across six benchmarks demonstrate the effectiveness of AoT both as a standalone framework and a plug-in enhancement. Notably, on HotpotQA, when applied to gpt-4o-mini, AoT achieves an 80.6% F1 score, surpassing o3-mini by 3.4% and DeepSeek-R1 by 10.6%. The code will be available at this https URL. 

**Abstract (ZH)**: Large Language Models (LLMs)在训练时标度训练并通过在推理时进行有效推理进一步增强其能力，取得了卓越的性能。然而，随着推理规模的扩大，现有的测试时标度方法遭受了累积历史信息的影响，不仅浪费了计算资源，还干扰了有效的推理。为解决这一问题，我们观察到复杂的推理过程通常通过解决一系列独立的子问题实现，每个子问题是自包含且可验证的。这些子问题本质上是原子问题，主要依赖于当前状态而非累积历史，类似于马尔可夫过程中的无记忆转移。基于这一观察，我们提出了Thinking Atom (Ta)，其中推理过程中的每个状态转换包括将当前问题分解为基于依赖关系的有向无环图，并将其子问题组合成一个新的原子问题状态。这种迭代分解-收缩过程将持续直到达到可以直接求解的原子问题，自然实现了问题状态之间的马尔可夫转移。此外，这些原子问题可以无缝集成到现有的测试时标度方法中，使Ta能够作为插件增强，提高推理能力。在六个基准上的实验表明，无论是作为独立框架还是插件增强，Ta都证明了其有效性。特别地，在HotpotQA上，当应用于gpt-4o-mini时，Ta达到了80.6%的F1分数，分别超过o3-mini的0.34%和DeepSeek-R1的10.6%。相关代码将发布在以下链接。 

---
# Evolving Hard Maximum Cut Instances for Quantum Approximate Optimization Algorithms 

**Title (ZH)**: 演化难以切分的最大切实例以优化量子近似优化算法 

**Authors**: Shuaiqun Pan, Yash J. Patel, Aneta Neumann, Frank Neumann, Thomas Bäck, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12012)  

**Abstract**: Variational quantum algorithms, such as the Recursive Quantum Approximate Optimization Algorithm (RQAOA), have become increasingly popular, offering promising avenues for employing Noisy Intermediate-Scale Quantum devices to address challenging combinatorial optimization tasks like the maximum cut problem. In this study, we utilize an evolutionary algorithm equipped with a unique fitness function. This approach targets hard maximum cut instances within the latent space of a Graph Autoencoder, identifying those that pose significant challenges or are particularly tractable for RQAOA, in contrast to the classic Goemans and Williamson algorithm. Our findings not only delineate the distinct capabilities and limitations of each algorithm but also expand our understanding of RQAOA's operational limits. Furthermore, the diverse set of graphs we have generated serves as a crucial benchmarking asset, emphasizing the need for more advanced algorithms to tackle combinatorial optimization challenges. Additionally, our results pave the way for new avenues in graph generation research, offering exciting opportunities for future explorations. 

**Abstract (ZH)**: 变分量子算法，如递归量子近似优化算法（RQAOA），已成为热门选择，为利用嘈杂的中等规模量子设备解决最大割问题等复杂组合优化任务提供了 promising 的途径。在本研究中，我们采用了一种配备独特fitness函数的进化算法，该方法针对图自编码器潜在空间中的hard最大割实例进行优化，识别出对RQAOA构成显著挑战或特别易于解决的实例，不同于经典的Goemans和Williamson算法。我们的研究不仅界定了每种算法的独特能力和限制，还扩展了对RQAOA操作极限的理解。此外，我们生成的多样化图集为基准测试提供了关键资源，强调了开发更高级算法以应对组合优化挑战的必要性。另外，我们的结果为图生成研究开辟了新的途径，提供了未来探索的激动人心的机会。 

---
# Demographic Attributes Prediction from Speech Using WavLM Embeddings 

**Title (ZH)**: 使用WavLM嵌入进行语音中的人口统计属性预测 

**Authors**: Yuchen Yang, Thomas Thebaud, Najim Dehak  

**Link**: [PDF](https://arxiv.org/pdf/2502.12007)  

**Abstract**: This paper introduces a general classifier based on WavLM features, to infer demographic characteristics, such as age, gender, native language, education, and country, from speech. Demographic feature prediction plays a crucial role in applications like language learning, accessibility, and digital forensics, enabling more personalized and inclusive technologies. Leveraging pretrained models for embedding extraction, the proposed framework identifies key acoustic and linguistic fea-tures associated with demographic attributes, achieving a Mean Absolute Error (MAE) of 4.94 for age prediction and over 99.81% accuracy for gender classification across various datasets. Our system improves upon existing models by up to relative 30% in MAE and up to relative 10% in accuracy and F1 scores across tasks, leveraging a diverse range of datasets and large pretrained models to ensure robustness and generalizability. This study offers new insights into speaker diversity and provides a strong foundation for future research in speech-based demographic profiling. 

**Abstract (ZH)**: 基于WavLM特征的通用分类器及其在语音中推断人口统计特征的应用：年龄、性别、母语、教育程度和国家的预测 

---
# Presumed Cultural Identity: How Names Shape LLM Responses 

**Title (ZH)**: 预设文化身份：名称如何塑造LLM响应 

**Authors**: Siddhesh Pawar, Arnav Arora, Lucie-Aimée Kaffee, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2502.11995)  

**Abstract**: Names are deeply tied to human identity. They can serve as markers of individuality, cultural heritage, and personal history. However, using names as a core indicator of identity can lead to over-simplification of complex identities. When interacting with LLMs, user names are an important point of information for personalisation. Names can enter chatbot conversations through direct user input (requested by chatbots), as part of task contexts such as CV reviews, or as built-in memory features that store user information for personalisation. We study biases associated with names by measuring cultural presumptions in the responses generated by LLMs when presented with common suggestion-seeking queries, which might involve making assumptions about the user. Our analyses demonstrate strong assumptions about cultural identity associated with names present in LLM generations across multiple cultures. Our work has implications for designing more nuanced personalisation systems that avoid reinforcing stereotypes while maintaining meaningful customisation. 

**Abstract (ZH)**: 姓名深刻地与人类身份相关联。它们可以作为个人性、文化传承和个人历史的标志。然而，将姓名作为身份的核心指标可能导致对复杂身份的过度简化。在与大模型交互时，用户姓名是实现个性化的重要信息点。姓名可以通过聊天机器人请求的直接用户输入、作为任务上下文（如简历审查）的一部分，或者作为内置的记忆功能存储用户信息以实现个性化进入聊天对话。我们通过衡量大模型在面对常见的建议查询时生成的回应中所体现的文化预设来研究与姓名相关的偏见。我们的分析表明，大模型生成中存在与姓名相关、跨越多种文化的强大文化身份假设。我们的工作对设计更细致的个性化系统具有重要意义，这些系统可以在避免强化刻板印象的同时仍保持有意义的定制。 

---
# Characterizing Photorealism and Artifacts in Diffusion Model-Generated Images 

**Title (ZH)**: Characterizing 光学真实感和生成图像中的伪像 

**Authors**: Negar Kamali, Karyn Nakamura, Aakriti Kumar, Angelos Chatzimparmpas, Jessica Hullman, Matthew Groh  

**Link**: [PDF](https://arxiv.org/pdf/2502.11989)  

**Abstract**: Diffusion model-generated images can appear indistinguishable from authentic photographs, but these images often contain artifacts and implausibilities that reveal their AI-generated provenance. Given the challenge to public trust in media posed by photorealistic AI-generated images, we conducted a large-scale experiment measuring human detection accuracy on 450 diffusion-model generated images and 149 real images. Based on collecting 749,828 observations and 34,675 comments from 50,444 participants, we find that scene complexity of an image, artifact types within an image, display time of an image, and human curation of AI-generated images all play significant roles in how accurately people distinguish real from AI-generated images. Additionally, we propose a taxonomy characterizing artifacts often appearing in images generated by diffusion models. Our empirical observations and taxonomy offer nuanced insights into the capabilities and limitations of diffusion models to generate photorealistic images in 2024. 

**Abstract (ZH)**: 基于扩散模型生成的图像在外观上可能与真实照片难以区分，但这些图像往往包含能揭示其AI生成身份的伪影和不合理之处。为应对高保真AI生成图像对媒体公信力的挑战，我们进行了一项大规模实验，评估了450张基于扩散模型生成的图像和149张真实图像的人类识别准确性。基于收集的749,828个观察数据和34,675条评论，我们发现，图像的场景复杂度、图像中的伪影类型、图像展示时间以及人类对AI生成图像的处理，都在决定人们区分真实图像与AI生成图像的准确度方面发挥着重要作用。此外，我们提出了一种分类法，描述常见于扩散模型生成图像中的伪影特征。我们的实证观察结果和分类法为理解扩散模型在2024年生成高保真图像的能力和局限性提供了细腻的见解。 

---
# Machine Learning Should Maximize Welfare, Not (Only) Accuracy 

**Title (ZH)**: 机器学习应最大化福利，而非（仅）准确性 

**Authors**: Nir Rosenfeld, Haifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11981)  

**Abstract**: Decades of research in machine learning have given us powerful tools for making accurate predictions. But when used in social settings and on human inputs, better accuracy does not immediately translate to better social outcomes. This may not be surprising given that conventional learning frameworks are not designed to express societal preferences -- let alone promote them. This position paper argues that machine learning is currently missing, and can gain much from incorporating, a proper notion of social welfare. The field of welfare economics asks: how should we allocate limited resources to self-interested agents in a way that maximizes social benefit? We argue that this perspective applies to many modern applications of machine learning in social contexts, and advocate for its adoption. Rather than disposing of prediction, we aim to leverage this forte of machine learning for promoting social welfare. We demonstrate this idea by proposing a conceptual framework that gradually transitions from accuracy maximization (with awareness to welfare) to welfare maximization (via accurate prediction). We detail applications and use-cases for which our framework can be effective, identify technical challenges and practical opportunities, and highlight future avenues worth pursuing. 

**Abstract (ZH)**: 几十年来机器学习的研究为我们提供了一种强有力的工具，用于进行准确的预测。但在社会环境中使用并应用于人类输入时，更高的准确率并不必然转化为更好的社会结果。鉴于此，传统的学习框架并不旨在表达社会偏好，更不用说促进社会偏好。本文认为，当前的机器学习缺少一个恰当的社会福利概念，同时也能够从融合社会福利概念中获益。福利经济学探讨的问题是，如何在满足自我利益代理人的前提下，分配有限资源以最大化社会利益？我们认为，这一视角适用于许多现代机器学习在社会环境中的应用，并倡导其采纳。我们并非是要抛弃预测，而是要利用机器学习在预测方面的优势来促进社会福利。通过提出一个逐步从注重准确性的福利最大化框架中转变的概念框架，我们展示了这一思想。我们详细阐述了我们的框架可以有效应用于的应用场景，指出了技术挑战和实际机会，并强调了值得进一步探索的未来方向。 

---
# Theoretical Barriers in Bellman-Based Reinforcement Learning 

**Title (ZH)**: 基于贝尔曼方程的强化学习的理论障碍 

**Authors**: Brieuc Pinon, Raphaël Jungers, Jean-Charles Delvenne  

**Link**: [PDF](https://arxiv.org/pdf/2502.11968)  

**Abstract**: Reinforcement Learning algorithms designed for high-dimensional spaces often enforce the Bellman equation on a sampled subset of states, relying on generalization to propagate knowledge across the state space. In this paper, we identify and formalize a fundamental limitation of this common approach. Specifically, we construct counterexample problems with a simple structure that this approach fails to exploit. Our findings reveal that such algorithms can neglect critical information about the problems, leading to inefficiencies. Furthermore, we extend this negative result to another approach from the literature: Hindsight Experience Replay learning state-to-state reachability. 

**Abstract (ZH)**: 高维空间中设计的强化学习算法通常在采样的状态子集上强制执行贝尔曼方程，依靠泛化在状态空间中传播知识。本文我们识别并形式化了这种常见方法的基本局限性。具体而言，我们构建了一个简单结构的反例问题，这种方法无法充分利用这些结构。我们的发现表明，此类算法可能忽略问题中关键的信息，导致效率低下。此外，我们还将这一负面结果扩展到文献中的另一种方法： hindsight experience replay 在状态间可达性学习中的应用。 

---
# A MIMO Wireless Channel Foundation Model via CIR-CSI Consistency 

**Title (ZH)**: 基于CIR-CSI一致性的一种MIMO无线信道基础模型 

**Authors**: Jun Jiang, Wenjun Yu, Yunfan Li, Yuan Gao, Shugong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11965)  

**Abstract**: In the field of artificial intelligence, self-supervised learning has demonstrated superior generalization capabilities by leveraging large-scale unlabeled datasets for pretraining, which is especially critical for wireless communication models to adapt to a variety of scenarios. This paper innovatively treats Channel State Information (CSI) and Channel Impulse Response (CIR) as naturally aligned multi-modal data and proposes the first MIMO wireless channel foundation model, named CSI-CLIP. By effectively capturing the joint representations of both CIR and CSI, CSI-CLIP exhibits remarkable adaptability across scenarios and robust feature extraction capabilities. Experimental results show that in positioning task, CSI-CLIP reduces the mean error distance by 22%; in beam management task, it increases accuracy by 1% compared to traditional supervised methods, as well as in the channel identification task. These improvements not only highlight the potential and value of CSI-CLIP in integrating sensing and communication but also demonstrate its significant advantages over existing techniques. Moreover, viewing CSI and CIR as multi-modal pairs and contrastive learning for wireless channel foundation model open up new research directions in the domain of MIMO wireless communications. 

**Abstract (ZH)**: 在人工智能领域，自监督学习通过利用大规模未标记数据集进行预训练，展示了卓越的泛化能力，特别对于使无线通信模型能够适应各种场景至关重要。本文创新性地将信道状态信息（CSI）和信道冲击响应（CIR）视为天然对齐的多模态数据，并提出首个MIMO无线信道基础模型——CSI-CLIP。通过有效地捕捉CIR和CSI的联合表示，CSI-CLIP表现出出色的适应性和鲁棒的特征提取能力。实验结果显示，在定位任务中，CSI-CLIP将平均误差距离降低了22%；在波束管理任务中，与传统的监督方法相比，准确度提高了1%；在信道识别任务中也表现出了显著的改进。这些改进不仅突显了CSI-CLIP在融合感知与通信中的潜力和价值，还展示了其相对于现有技术的显著优势。此外，将CSI和CIR视为多模态配对，并采用对比学习方法，为MIMO无线通信中的无线信道基础模型开辟了新的研究方向。 

---
# Navigating the Helpfulness-Truthfulness Trade-Off with Uncertainty-Aware Instruction Fine-Tuning 

**Title (ZH)**: 不确定性意识指令微调在帮助性和真实性的权衡导航中 

**Authors**: Tianyi Wu, Jingwei Ni, Bryan Hooi, Jiaheng Zhang, Elliott Ash, See-Kiong Ng, Mrinmaya Sachan, Markus Leippold  

**Link**: [PDF](https://arxiv.org/pdf/2502.11962)  

**Abstract**: Instruction Fine-tuning (IFT) can enhance the helpfulness of Large Language Models (LLMs), but it may lower their truthfulness. This trade-off arises because IFT steers LLMs to generate responses with long-tail knowledge that is not well covered during pre-training, leading to more informative but less truthful answers when generalizing to unseen tasks. In this paper, we empirically demonstrate this helpfulness-truthfulness trade-off in IFT and propose $\textbf{UNIT}$, a novel IFT paradigm to address it. UNIT teaches LLMs to recognize their uncertainty and explicitly reflect it at the end of their responses. Experimental results show that UNIT-tuned models maintain their helpfulness while distinguishing between certain and uncertain claims, thereby reducing hallucinations. 

**Abstract (ZH)**: Instruction Fine-tuning (IFT)可以增强大型语言模型（LLMs）的有用性，但可能降低其真实性。在这种权衡中，IFT引导LLMs生成预训练中未充分覆盖的长尾知识，导致在泛化到未见任务时提供更多有用但不那么真实的信息。在本文中，我们实证展示了IFT中的这种有用性-真实性权衡，并提出了一种新的IFT范式$\textbf{UNIT}$以解决这一问题。UNIT训练LLMs识别自身的不确定性，并在响应结尾明确反映这种不确定性。实验结果表明，UNIT调优的模型在保持有用性的同时，能够区分确定性和不确定性声明，从而减少幻觉。 

---
# Massively Scaling Explicit Policy-conditioned Value Functions 

**Title (ZH)**: 大规模扩展显式策略条件值函数 

**Authors**: Nico Bohlinger, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2502.11949)  

**Abstract**: We introduce a scaling strategy for Explicit Policy-Conditioned Value Functions (EPVFs) that significantly improves performance on challenging continuous-control tasks. EPVFs learn a value function V({\theta}) that is explicitly conditioned on the policy parameters, enabling direct gradient-based updates to the parameters of any policy. However, EPVFs at scale struggle with unrestricted parameter growth and efficient exploration in the policy parameter space. To address these issues, we utilize massive parallelization with GPU-based simulators, big batch sizes, weight clipping and scaled peturbations. Our results show that EPVFs can be scaled to solve complex tasks, such as a custom Ant environment, and can compete with state-of-the-art Deep Reinforcement Learning (DRL) baselines like Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC). We further explore action-based policy parameter representations from previous work and specialized neural network architectures to efficiently handle weight-space features, which have not been used in the context of DRL before. 

**Abstract (ZH)**: 我们介绍了一种用于显式策略条件价值函数（EPVFs）的扩展策略，该策略显著提高了在具有挑战性的连续控制任务上的性能。 

---
# Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction 

**Title (ZH)**: 步进音频：智能语音交互中的统一理解和生成 

**Authors**: Ailin Huang, Boyong Wu, Bruce Wang, Chao Yan, Chen Hu, Chengli Feng, Fei Tian, Feiyu Shen, Jingbei Li, Mingrui Chen, Peng Liu, Ruihang Miao, Wang You, Xi Chen, Xuerui Yang, Yechang Huang, Yuxiang Zhang, Zheng Gong, Zixin Zhang, Brian Li, Changyi Wan, Hanpeng Hu, Ranchen Ming, Song Yuan, Xuelin Zhang, Yu Zhou, Bingxin Li, Buyun Ma, Kang An, Wei Ji, Wen Li, Xuan Wen, Yuankai Ma, Yuanwei Liang, Yun Mou, Bahtiyar Ahmidi, Bin Wang, Bo Li, Changxin Miao, Chen Xu, Chengting Feng, Chenrun Wang, Dapeng Shi, Deshan Sun, Dingyuan Hu, Dula Sai, Enle Liu, Guanzhe Huang, Gulin Yan, Heng Wang, Haonan Jia, Haoyang Zhang, Jiahao Gong, Jianchang Wu, Jiahong Liu, Jianjian Sun, Jiangjie Zhen, Jie Feng, Jie Wu, Jiaoren Wu, Jie Yang, Jinguo Wang, Jingyang Zhang, Junzhe Lin, Kaixiang Li, Lei Xia, Li Zhou, Longlong Gu, Mei Chen, Menglin Wu, Ming Li, Mingxiao Li, Mingyao Liang, Na Wang, Nie Hao, Qiling Wu, Qinyuan Tan, Shaoliang Pang, Shiliang Yang, Shuli Gao, Siqi Liu, Sitong Liu, Tiancheng Cao, Tianyu Wang, Wenjin Deng, Wenqing He, Wen Sun, Xin Han, Xiaomin Deng, Xiaojia Liu, Xu Zhao, Yanan Wei, Yanbo Yu, Yang Cao, Yangguang Li, Yangzhen Ma, Yanming Xu, Yaqiang Shi, Yilei Wang, Yinmin Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11946)  

**Abstract**: Real-time speech interaction, serving as a fundamental interface for human-machine collaboration, holds immense potential. However, current open-source models face limitations such as high costs in voice data collection, weakness in dynamic control, and limited intelligence. To address these challenges, this paper introduces Step-Audio, the first production-ready open-source solution. Key contributions include: 1) a 130B-parameter unified speech-text multi-modal model that achieves unified understanding and generation, with the Step-Audio-Chat version open-sourced; 2) a generative speech data engine that establishes an affordable voice cloning framework and produces the open-sourced lightweight Step-Audio-TTS-3B model through distillation; 3) an instruction-driven fine control system enabling dynamic adjustments across dialects, emotions, singing, and RAP; 4) an enhanced cognitive architecture augmented with tool calling and role-playing abilities to manage complex tasks effectively. Based on our new StepEval-Audio-360 evaluation benchmark, Step-Audio achieves state-of-the-art performance in human evaluations, especially in terms of instruction following. On open-source benchmarks like LLaMA Question, shows 9.3% average performance improvement, demonstrating our commitment to advancing the development of open-source multi-modal language technologies. Our code and models are available at this https URL. 

**Abstract (ZH)**: 实时语音交互作为人机协作的基本接口，具有巨大潜力。然而，当前的开源模型面临语音数据采集成本高、动态控制薄弱、智能程度有限等挑战。为了应对这些挑战，本论文介绍了Step-Audio，这是首个生产级开源解决方案。主要贡献包括：1）一个拥有130亿参数的统一语音-文本多模态模型，实现了统一的理解和生成，Step-Audio-Chat版本已开源；2）一个生成性语音数据引擎，建立了经济实惠的声音克隆框架，并通过蒸馏生成了轻量级的Step-Audio-TTS-3B模型；3）一个基于指令的精细控制系统，能够动态调整方言、情感、唱歌和嘻哈等；4）一种增强的认知架构，具有工具调用和角色扮演能力，有效管理复杂任务。基于我们新的StepEval-Audio-360评估基准，Step-Audio在人类评估中达到了最先进水平，尤其是在指令遵循方面。在类似LLaMA Question的开源基准测试中，平均性能提高了9.3%，展示了我们致力于推进开源多模态语言技术开发的承诺。我们的代码和模型可在以下链接获取。 

---
# Deep Spatio-Temporal Neural Network for Air Quality Reanalysis 

**Title (ZH)**: 深度空间-时间神经网络为空气质量再分析 

**Authors**: Ammar Kheder, Benjamin Foreback, Lili Wang, Zhi-Song Liu, Michael Boy  

**Link**: [PDF](https://arxiv.org/pdf/2502.11941)  

**Abstract**: Air quality prediction is key to mitigating health impacts and guiding decisions, yet existing models tend to focus on temporal trends while overlooking spatial generalization. We propose AQ-Net, a spatiotemporal reanalysis model for both observed and unobserved stations in the near future. AQ-Net utilizes the LSTM and multi-head attention for the temporal regression. We also propose a cyclic encoding technique to ensure continuous time representation. To learn fine-grained spatial air quality estimation, we incorporate AQ-Net with the neural kNN to explore feature-based interpolation, such that we can fill the spatial gaps given coarse observation stations. To demonstrate the efficiency of our model for spatiotemporal reanalysis, we use data from 2013-2017 collected in northern China for PM2.5 analysis. Extensive experiments show that AQ-Net excels in air quality reanalysis, highlighting the potential of hybrid spatio-temporal models to better capture environmental dynamics, especially in urban areas where both spatial and temporal variability are critical. 

**Abstract (ZH)**: 空气质量预測对于减轻健康影响和指导决策至关重要，但现有的模型往往侧重于时间趋势而忽视了空间泛化。我们提出AQ-Net，这是一种时空重分析模型，适用于近未来的实测和非实测站点。AQ-Net利用LSTM和多头注意力机制进行时间回归。同时，我们提出了一种循环编码技术以确保连续的时间表示。为了学习细粒度的空间空气质量估计，我们将AQ-Net与神经kNN结合，通过基于特征的插值来填补粗略观测站点间的空间空白。为了证明我们的模型在时空重分析中的效率，我们使用2013-2017年来自中国北方的PM2.5数据进行分析。 extensive 实验表明，AQ-Net在空气质量重分析中表现出色，突显了混合时空模型更好地捕捉环境动态的潜力，尤其是在空间和时间变异都至关重要的城市地区。 

---
# FitLight: Federated Imitation Learning for Plug-and-Play Autonomous Traffic Signal Control 

**Title (ZH)**: FitLight：插拔式自主交通信号控制的联邦模仿学习 

**Authors**: Yutong Ye, Yingbo Zhou, Zhusen Liu, Xiao Du, Hao Zhou, Xiang Lian, Mingsong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11937)  

**Abstract**: Although Reinforcement Learning (RL)-based Traffic Signal Control (TSC) methods have been extensively studied, their practical applications still raise some serious issues such as high learning cost and poor generalizability. This is because the ``trial-and-error'' training style makes RL agents extremely dependent on the specific traffic environment, which also requires a long convergence time. To address these issues, we propose a novel Federated Imitation Learning (FIL)-based framework for multi-intersection TSC, named FitLight, which allows RL agents to plug-and-play for any traffic environment without additional pre-training cost. Unlike existing imitation learning approaches that rely on pre-training RL agents with demonstrations, FitLight allows real-time imitation learning and seamless transition to reinforcement learning. Due to our proposed knowledge-sharing mechanism and novel hybrid pressure-based agent design, RL agents can quickly find a best control policy with only a few episodes. Moreover, for resource-constrained TSC scenarios, FitLight supports model pruning and heterogeneous model aggregation, such that RL agents can work on a micro-controller with merely 16{\it KB} RAM and 32{\it KB} ROM. Extensive experiments demonstrate that, compared to state-of-the-art methods, FitLight not only provides a superior starting point but also converges to a better final solution on both real-world and synthetic datasets, even under extreme resource limitations. 

**Abstract (ZH)**: 基于联邦拟合学习的多交叉口交通信号控制方法FitLight 

---
# EssayJudge: A Multi-Granular Benchmark for Assessing Automated Essay Scoring Capabilities of Multimodal Large Language Models 

**Title (ZH)**: EssayJudge: 一种评估多模态大型语言模型作文评分能力的多粒度基准 

**Authors**: Jiamin Su, Yibo Yan, Fangteng Fu, Han Zhang, Jingheng Ye, Xiang Liu, Jiahao Huo, Huiyu Zhou, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11916)  

**Abstract**: Automated Essay Scoring (AES) plays a crucial role in educational assessment by providing scalable and consistent evaluations of writing tasks. However, traditional AES systems face three major challenges: (1) reliance on handcrafted features that limit generalizability, (2) difficulty in capturing fine-grained traits like coherence and argumentation, and (3) inability to handle multimodal contexts. In the era of Multimodal Large Language Models (MLLMs), we propose EssayJudge, the first multimodal benchmark to evaluate AES capabilities across lexical-, sentence-, and discourse-level traits. By leveraging MLLMs' strengths in trait-specific scoring and multimodal context understanding, EssayJudge aims to offer precise, context-rich evaluations without manual feature engineering, addressing longstanding AES limitations. Our experiments with 18 representative MLLMs reveal gaps in AES performance compared to human evaluation, particularly in discourse-level traits, highlighting the need for further advancements in MLLM-based AES research. Our dataset and code will be available upon acceptance. 

**Abstract (ZH)**: 多模态大语言模型时代自动作文评分系统的评测基准：EssayJudge 

---
# DLFR-VAE: Dynamic Latent Frame Rate VAE for Video Generation 

**Title (ZH)**: DLFR-VAE：动态隐层帧率VAE视频生成 

**Authors**: Zhihang Yuan, Siyuan Wang, Rui Xie, Hanling Zhang, Tongcheng Fang, Yuzhang Shang, Shengen Yan, Guohao Dai, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11897)  

**Abstract**: In this paper, we propose the Dynamic Latent Frame Rate VAE (DLFR-VAE), a training-free paradigm that can make use of adaptive temporal compression in latent space. While existing video generative models apply fixed compression rates via pretrained VAE, we observe that real-world video content exhibits substantial temporal non-uniformity, with high-motion segments containing more information than static scenes. Based on this insight, DLFR-VAE dynamically adjusts the latent frame rate according to the content complexity. Specifically, DLFR-VAE comprises two core innovations: (1) A Dynamic Latent Frame Rate Scheduler that partitions videos into temporal chunks and adaptively determines optimal frame rates based on information-theoretic content complexity, and (2) A training-free adaptation mechanism that transforms pretrained VAE architectures into a dynamic VAE that can process features with variable frame rates. Our simple but effective DLFR-VAE can function as a plug-and-play module, seamlessly integrating with existing video generation models and accelerating the video generation process. 

**Abstract (ZH)**: 动态潜在帧率VAE（DLFR-VAE）：一种无需训练的时空压缩范式 

---
# CAMEL: Continuous Action Masking Enabled by Large Language Models for Reinforcement Learning 

**Title (ZH)**: CAMEL: 由大规模语言模型实现的连续动作掩蔽强化学习 

**Authors**: Yanxiao Zhao, Yangge Qian, Jingyang Shan, Xiaolin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2502.11896)  

**Abstract**: Reinforcement learning (RL) in continuous action spaces encounters persistent challenges, such as inefficient exploration and convergence to suboptimal solutions. To address these limitations, we propose CAMEL, a novel framework integrating LLM-generated suboptimal policies into the RL training pipeline. CAMEL leverages dynamic action masking and an adaptive epsilon-masking mechanism to guide exploration during early training stages while gradually enabling agents to optimize policies independently. At the core of CAMEL lies the integration of Python-executable suboptimal policies generated by LLMs based on environment descriptions and task objectives. Although simplistic and hard-coded, these policies offer valuable initial guidance for RL agents. To effectively utilize these priors, CAMEL employs masking-aware optimization to dynamically constrain the action space based on LLM outputs. Additionally, epsilon-masking gradually reduces reliance on LLM-generated guidance, enabling agents to transition from constrained exploration to autonomous policy refinement. Experimental validation on Gymnasium MuJoCo environments demonstrates the effectiveness of CAMEL. In Hopper-v4 and Ant-v4, LLM-generated policies significantly improve sample efficiency, achieving performance comparable to or surpassing expert masking baselines. For Walker2d-v4, where LLMs struggle to accurately model bipedal gait dynamics, CAMEL maintains robust RL performance without notable degradation, highlighting the framework's adaptability across diverse tasks. While CAMEL shows promise in enhancing sample efficiency and mitigating convergence challenges, these issues remain open for further research. Future work aims to generalize CAMEL to multimodal LLMs for broader observation-action spaces and automate policy evaluation, reducing human intervention and enhancing scalability in RL training pipelines. 

**Abstract (ZH)**: 连续动作空间中的强化学习（RL）面临持续的挑战，如探索效率低下和收敛到次优解。为此，我们提出了一种名为CAMEL的新型框架，该框架将LLM生成的次优策略集成到RL训练管道中。CAMEL通过动态动作掩码和自适应ε-掩码机制，在早期训练阶段引导探索，并逐步使智能体能够独立优化策略。CAMEL的核心在于将基于环境描述和任务目标由LLM生成的可执行Python次优策略的集成。尽管这些策略简单且硬编码，但它们为RL智能体提供了初始有价值的指导。为了有效利用这些先验知识，CAMEL采用掩码感知优化动态限制动作空间，基于LLM输出。此外，ε-掩码逐渐减少对LLM生成指导的依赖，使智能体从受限探索过渡到自主策略优化。在Gymnasium MuJoCo环境中进行的经验验证表明了CAMEL的有效性。在Hopper-v4和Ant-v4中，LLM生成的策略显著提高了样本效率，性能达到或超过了专家掩码基准。在Walker2d-v4中，由于LLM难以准确建模双足步态动力学，CAMEL保持了稳健的RL性能，没有明显下降，突显了该框架在不同任务中的适应性。虽然CAMEL在提高样本效率和缓解收敛挑战方面显示出潜力，但这些问题仍有待进一步研究。未来工作旨在将CAMEL推广到多模态LLM，以适用于更广泛的观测-动作空间，并自动评估策略，减少人为干预，增强RL训练管道的可扩展性。 

---
# Continual Quantization-Aware Pre-Training: When to transition from 16-bit to 1.58-bit pre-training for BitNet language models? 

**Title (ZH)**: 持续量化感知预训练：BitNet语言模型在何时从16位转换到1.58位预训练？ 

**Authors**: Jacob Nielsen, Peter Schneider-Kamp, Lukas Galke  

**Link**: [PDF](https://arxiv.org/pdf/2502.11895)  

**Abstract**: Large language models (LLMs) require immense resources for training and inference. Quantization, a technique that reduces the precision of model parameters, offers a promising solution for improving LLM efficiency and sustainability. While post-training quantization methods typically achieve 4-8 bits per parameter, recent research suggests that training LLMs with 1.58 bits per weight parameter from scratch can maintain model accuracy while greatly reducing memory requirements and energy consumption at inference time. Here, we investigate a training strategy for quantization-aware pre-training, where the models are first trained with 16-bit precision and then transition into 1.58-bit quantization-aware training. Our results on 11 downstream tasks show that this 16-to-1.58-bit training strategy is preferable over full 1.58-bit training and leaves models closer to those which have undergone 16-bit training. We further investigate the effects of retaining the optimizer state at the transition point and gradually phasing in quantization strength -- finding that both techniques alleviate the magnitude of loss spikes, but also that these effects can be compensated through further training. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的训练和推理需要巨额资源。量化技术通过降低模型参数精度，为提高LLM效率和可持续性提供了有前景的解决方案。虽然后训练量化方法通常可实现每参数4-8位，但近期研究表明，从零开始训练具有1.58位每权重参数的LLMs可以在保持模型精度的同时大幅降低推理时的内存需求和能耗。在此，我们研究了一种量化感知预训练的训练策略，首先使用16位精度训练模型，然后过渡到1.58位量化感知训练。我们在11个下游任务上的结果显示，这种16位到1.58位的训练策略优于完全1.58位训练，并使模型更接近于经过16位训练的模型。我们进一步研究了在过渡点保留优化器状态和逐步引入量化强度的效果——发现这两种技术都能减轻损失峰值的幅度，但这些效果可以通过进一步训练来弥补。 

---
# Stonefish: Supporting Machine Learning Research in Marine Robotics 

**Title (ZH)**: 石鱼：支持海洋机器人领域机器学习研究 

**Authors**: Michele Grimaldi, Patryk Cieslak, Eduardo Ochoa, Vibhav Bharti, Hayat Rajani, Ignacio Carlucho, Maria Koskinopoulou, Yvan R. Petillot, Nuno Gracias  

**Link**: [PDF](https://arxiv.org/pdf/2502.11887)  

**Abstract**: Simulations are highly valuable in marine robotics, offering a cost-effective and controlled environment for testing in the challenging conditions of underwater and surface operations. Given the high costs and logistical difficulties of real-world trials, simulators capable of capturing the operational conditions of subsea environments have become key in developing and refining algorithms for remotely-operated and autonomous underwater vehicles. This paper highlights recent enhancements to the Stonefish simulator, an advanced open-source platform supporting development and testing of marine robotics solutions. Key updates include a suite of additional sensors, such as an event-based camera, a thermal camera, and an optical flow camera, as well as, visual light communication, support for tethered operations, improved thruster modelling, more flexible hydrodynamics, and enhanced sonar accuracy. These developments and an automated annotation tool significantly bolster Stonefish's role in marine robotics research, especially in the field of machine learning, where training data with a known ground truth is hard or impossible to collect. 

**Abstract (ZH)**: 模拟技术在海洋机器人研究中极为宝贵，提供了一种在海上和水面操作的苛刻环境下进行低成本且可控测试的环境。鉴于真实世界实验的高成本和 logistical 困难，能够捕捉海底环境操作条件的模拟器已成为开发和改进遥控和自主 underwater 车辆算法的关键工具。本文强调了对 Stonefish 模拟器的近期改进，这是一个先进的开源平台，支持海洋机器人解决方案的开发和测试。关键更新包括一系列额外的传感器，如事件驱动的相机、热像仪和光学流相机，以及视觉光通信、系缆操作支持、推进器模型改进、更灵活的水动力学以及声纳精度增强。这些改进和自动标注工具显著增强了 Stonefish 在海洋机器人研究中的作用，特别是在机器学习领域，因为此类标签数据的收集往往是困难的或不可能的。 

---
# LIMR: Less is More for RL Scaling 

**Title (ZH)**: LIMR: 少就是多的RL扩展方法 

**Authors**: Xuefeng Li, Haoyang Zou, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11886)  

**Abstract**: In this paper, we ask: what truly determines the effectiveness of RL training data for enhancing language models' reasoning capabilities? While recent advances like o1, Deepseek R1, and Kimi1.5 demonstrate RL's potential, the lack of transparency about training data requirements has hindered systematic progress. Starting directly from base models without distillation, we challenge the assumption that scaling up RL training data inherently improves performance. we demonstrate that a strategically selected subset of just 1,389 samples can outperform the full 8,523-sample dataset. We introduce Learning Impact Measurement (LIM), an automated method to evaluate and prioritize training samples based on their alignment with model learning trajectories, enabling efficient resource utilization and scalable implementation. Our method achieves comparable or even superior performance using only 1,389 samples versus the full 8,523 samples dataset. Notably, while recent data-efficient approaches (e.g., LIMO and s1) show promise with 32B-scale models, we find it significantly underperforms at 7B-scale through supervised fine-tuning (SFT). In contrast, our RL-based LIMR achieves 16.7% higher accuracy on AIME24 and outperforms LIMO and s1 by 13.0% and 22.2% on MATH500. These results fundamentally reshape our understanding of RL scaling in LLMs, demonstrating that precise sample selection, rather than data scale, may be the key to unlocking enhanced reasoning capabilities. For reproducible research and future innovation, we are open-sourcing LIMR, including implementation of LIM, training and evaluation code, curated datasets, and trained models at this https URL. 

**Abstract (ZH)**: 在本论文中，我们探讨了什么真正决定了增强语言模型推理能力的RL训练数据的有效性。尽管像o1、Deepseek R1和Kimi1.5等最近的进步展示了RL的潜力，但缺乏关于训练数据需求的透明度阻碍了系统的进展。直接从基础模型开始而无需知识蒸馏，我们挑战了RL训练数据规模扩大必然会提升性能的假设。我们证明，一个策略性选择的仅1,389个样本的小子集就能优于包含8,523个样本的完整数据集。我们引入了一种自动化的方法——学习影响度量（LIM），以评估和优先考虑训练样本的依据是它们与模型学习轨迹的对齐情况，从而实现资源的有效利用和可扩展的实现。仅使用1,389个样本，我们的方法就能达到与8,523个样本数据集相当甚至更好的性能。值得注意的是，尽管最近的数据高效方法（如LIMO和s1）在32B规模的模型上表现出潜力，我们发现它们在7B规模的模型上通过监督微调表现显著不如预期。相反，基于RL的LIMR在AIME24上取得了16.7%的更高准确率，并在MATH500上分别优于LIMO和s1 13.0%和22.2%。这些结果从根本上重塑了我们对大型语言模型中RL扩展的理解，表明精确的选择样本而不是数据规模可能是解锁增强推理能力的关键。为了实现可再现的研究和未来的创新，我们开源了LIMR，包括LIM的实现、训练和评估代码、精选数据集和训练模型，详情请访问此链接。 

---
# Bitnet.cpp: Efficient Edge Inference for Ternary LLMs 

**Title (ZH)**: Bitnet.cpp: Ternary LLMs的高效边端推理 

**Authors**: Jinheng Wang, Hansong Zhou, Ting Song, Shijie Cao, Yan Xia, Ting Cao, Jianyu Wei, Shuming Ma, Hongyu Wang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.11880)  

**Abstract**: The advent of 1-bit large language models (LLMs), led by BitNet b1.58, has spurred interest in ternary LLMs. Despite this, research and practical applications focusing on efficient edge inference for ternary LLMs remain scarce. To bridge this gap, we introduce this http URL, an inference system optimized for BitNet b1.58 and ternary LLMs. Given that mixed-precision matrix multiplication (mpGEMM) constitutes the bulk of inference time in ternary LLMs, this http URL incorporates a novel mpGEMM library to facilitate sub-2-bits-per-weight, efficient and lossless inference. The library features two core solutions: Ternary Lookup Table (TL), which addresses spatial inefficiencies of previous bit-wise methods, and Int2 with a Scale (I2_S), which ensures lossless edge inference, both enabling high-speed inference. Our experiments show that this http URL achieves up to a 6.25x increase in speed over full-precision baselines and up to 2.32x over low-bit baselines, setting new benchmarks in the field. Additionally, we expand TL to element-wise lookup table (ELUT) for low-bit LLMs in the appendix, presenting both theoretical and empirical evidence of its considerable potential. this http URL is publicly available at this https URL , offering a sophisticated solution for the efficient and practical deployment of edge LLMs. 

**Abstract (ZH)**: The 出现1比特大型语言模型（LLMs），如BitNet b1.58，促进了三值LLMs的研究。尽管如此，针对三值LLMs的高效边缘推理研究和应用仍然稀缺。为填补这一空白，我们介绍了此系统，该系统针对BitNet b1.58和三值LLMs进行了优化。鉴于混合精度矩阵乘法（mpGEMM）占据了三值LLMs推理时间的主要部分，此系统集成了一个新颖的mpGEMM库，以实现每权重少于2比特、高效且无损的推理。该库包含两个核心解决方案：三值查找表（TL），解决了一维位方法的空间效率问题；以及带有尺度的整数2（I2_S），确保了无损边缘推理，两者共同实现高速推理。我们的实验结果显示，该系统在全精度基准上的速度提高了多达6.25倍，在低比特基准上的速度提高了2.32倍，从而在领域内建立了新的基准。此外，我们在附录中将TL扩展到元素级查找表（ELUT），针对低比特LLMs，提供了理论和实验证据，展示了其巨大的潜力。此系统在 https://this-url 提供，为边缘LLMs的高效和实践部署提供了复杂的解决方案。 

---
# FedEAT: A Robustness Optimization Framework for Federated LLMs 

**Title (ZH)**: FedEAT：联邦大规模语言模型的稳健性优化框架 

**Authors**: Yahao Pang, Xingyuan Wu, Xiaojin Zhang, Wei Chen, Hai Jin  

**Link**: [PDF](https://arxiv.org/pdf/2502.11863)  

**Abstract**: Significant advancements have been made by Large Language Models (LLMs) in the domains of natural language understanding and automated content creation. However, they still face persistent problems, including substantial computational costs and inadequate availability of training data. The combination of Federated Learning (FL) and LLMs (federated LLMs) offers a solution by leveraging distributed data while protecting privacy, which positions it as an ideal choice for sensitive domains. However, Federated LLMs still suffer from robustness challenges, including data heterogeneity, malicious clients, and adversarial attacks, which greatly hinder their applications. We first introduce the robustness problems in federated LLMs, to address these challenges, we propose FedEAT (Federated Embedding space Adversarial Training), a novel framework that applies adversarial training in the embedding space of client LLM and employs a robust aggregation approach, specifically geometric median aggregation, to enhance the robustness of Federated LLMs. Our experiments demonstrate that FedEAT effectively improves the robustness of Federated LLMs with minimal performance loss. 

**Abstract (ZH)**: 联邦学习与大规模语言模型结合（联邦LLM）在自然语言理解与自动化内容创作领域的显著进展虽然受到了计算成本高昂和训练数据不足的问题限制，但通过利用联邦学习的技术可以解决这些挑战并保护隐私，使其成为敏感领域的一个理想选择。然而，联邦LLM仍然面临着鲁棒性方面的挑战，包括数据异质性、恶意客户端以及 adversarial 攻击，这些都极大地阻碍了其应用。我们首先介绍了联邦LLM中的鲁棒性问题，为了解决这些问题，我们提出了一种名为FedEAT（联邦嵌入空间对抗训练）的新型框架，在客户端LLM的嵌入空间中应用对抗训练，并采用几何中位数聚合等鲁棒聚合方法来增强联邦LLM的鲁棒性。实验结果表明，FedEAT能够在 minimal 性能损失的情况下显著提升联邦LLM的鲁棒性。 

---
# Steering the LoCoMotif: Using Domain Knowledge in Time Series Motif Discovery 

**Title (ZH)**: 引导LoCoMotif：在时间序列模因发现中运用领域知识 

**Authors**: Aras Yurtman, Daan Van Wesenbeeck, Wannes Meert, Hendrik Blockeel  

**Link**: [PDF](https://arxiv.org/pdf/2502.11850)  

**Abstract**: Time Series Motif Discovery (TSMD) identifies repeating patterns in time series data, but its unsupervised nature might result in motifs that are not interesting to the user. To address this, we propose a framework that allows the user to impose constraints on the motifs to be discovered, where constraints can easily be defined according to the properties of the desired motifs in the application domain. We also propose an efficient implementation of the framework, the LoCoMotif-DoK algorithm. We demonstrate that LoCoMotif-DoK can effectively leverage domain knowledge in real and synthetic data, outperforming other TSMD techniques which only support a limited form of domain knowledge. 

**Abstract (ZH)**: 基于约束的时序模式发现框架（基于LoCoMotif-DoK算法）有效地利用领域知识在实际和合成数据中超越了仅支持有限领域知识的其他时序模式发现技术。 

---
# BaxBench: Can LLMs Generate Correct and Secure Backends? 

**Title (ZH)**: BaxBench: LLMs能生成正确且安全的后端代码吗？ 

**Authors**: Mark Vero, Niels Mündler, Victor Chibotaru, Veselin Raychev, Maximilian Baader, Nikola Jovanović, Jingxuan He, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2502.11844)  

**Abstract**: The automatic generation of programs has long been a fundamental challenge in computer science. Recent benchmarks have shown that large language models (LLMs) can effectively generate code at the function level, make code edits, and solve algorithmic coding tasks. However, to achieve full automation, LLMs should be able to generate production-quality, self-contained application modules. To evaluate the capabilities of LLMs in solving this challenge, we introduce BaxBench, a novel evaluation benchmark consisting of 392 tasks for the generation of backend applications. We focus on backends for three critical reasons: (i) they are practically relevant, building the core components of most modern web and cloud software, (ii) they are difficult to get right, requiring multiple functions and files to achieve the desired functionality, and (iii) they are security-critical, as they are exposed to untrusted third-parties, making secure solutions that prevent deployment-time attacks an imperative. BaxBench validates the functionality of the generated applications with comprehensive test cases, and assesses their security exposure by executing end-to-end exploits. Our experiments reveal key limitations of current LLMs in both functionality and security: (i) even the best model, OpenAI o1, achieves a mere 60% on code correctness; (ii) on average, we could successfully execute security exploits on more than half of the correct programs generated by each LLM; and (iii) in less popular backend frameworks, models further struggle to generate correct and secure applications. Progress on BaxBench signifies important steps towards autonomous and secure software development with LLMs. 

**Abstract (ZH)**: 自动生成程序一直是计算机科学中的一个基础挑战。近期基准测试表明，大规模语言模型（LLMs）能够有效在函数级别生成代码、进行代码编辑以及解决算法编程任务。然而，要实现完全自动化，LLMs 应能够生成生产级别的、自包含的应用模块。为了评估 LLMs 解决这一挑战的能力，我们引入了 BaxBench，这是一个新型评估基准，包含 392 个用于生成后端应用的任务。我们重点关注后端的原因有三：（i）它们具有实际相关性，构成了大多数现代网络和云软件的核心组件；（ii）它们难以实现正确功能，需要多个函数和文件才能实现所需的功能；（iii）它们是安全关键的，因为它们暴露给不可信的第三方，确保安全解决方案防止部署时攻击是必然要求。BaxBench 通过全面的测试案例验证生成应用的功能性，并通过端到端利用来评估其安全暴露程度。我们的实验揭示了当前 LLMs 在功能性和安全性方面的关键局限性：（i）即便是最好的模型 OpenAI o1，代码正确性也只能达到 60%；（ii）平均而言，我们能够成功执行对每种 LLM 生成的正确程序的一半以上进行的安全利用；（iii）在不太流行的后端框架中，模型进一步难以生成正确且安全的应用。BaxBench 上的进步标志着朝着使用 LLMs 实现自主且安全的软件开发的重要一步。 

---
# Can LLM Agents Maintain a Persona in Discourse? 

**Title (ZH)**: LLM代理在 discourse 中能否维持人设？ 

**Authors**: Pranav Bhandari, Nicolas Fay, Michael Wise, Amitava Datta, Stephanie Meek, Usman Naseem, Mehwish Nasim  

**Link**: [PDF](https://arxiv.org/pdf/2502.11843)  

**Abstract**: Large Language Models (LLMs) are widely used as conversational agents, exploiting their capabilities in various sectors such as education, law, medicine, and more. However, LLMs are often subjected to context-shifting behaviour, resulting in a lack of consistent and interpretable personality-aligned interactions. Adherence to psychological traits lacks comprehensive analysis, especially in the case of dyadic (pairwise) conversations. We examine this challenge from two viewpoints, initially using two conversation agents to generate a discourse on a certain topic with an assigned personality from the OCEAN framework (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism) as High/Low for each trait. This is followed by using multiple judge agents to infer the original traits assigned to explore prediction consistency, inter-model agreement, and alignment with the assigned personality. Our findings indicate that while LLMs can be guided toward personality-driven dialogue, their ability to maintain personality traits varies significantly depending on the combination of models and discourse settings. These inconsistencies emphasise the challenges in achieving stable and interpretable personality-aligned interactions in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）广泛用作对话代理，在教育、法律、医学等领域发挥其能力。然而，LLMs经常表现出上下文切换行为，导致缺乏一致性和可解释性的个性一致互动。个性特征的遵循缺乏全面分析，尤其是在双人对话的情况下。我们从两个角度来看待这一挑战：首先，使用两个对话代理生成某一主题的对话，每个特性（开放性、尽责性、外向性、和藹性、神经质）指派为高/低；接着，使用多个评判代理推断原始指派的特性，以探索预测一致性、模型间一致性以及与指派个性的匹配度。我们的研究结果表明，虽然LLMs可以被引导进行个性驱动的对话，但它们在维持个性特质方面的能力因模型组合和对话设置而异。这些不一致性强调了在LLMs中实现稳定和可解释的个性一致互动的挑战。 

---
# ChordFormer: A Conformer-Based Architecture for Large-Vocabulary Audio Chord Recognition 

**Title (ZH)**: ChordFormer：基于Conformer的大型词汇量音频和弦识别架构 

**Authors**: Muhammad Waseem Akram, Stefano Dettori, Valentina Colla, Giorgio Carlo Buttazzo  

**Link**: [PDF](https://arxiv.org/pdf/2502.11840)  

**Abstract**: Chord recognition serves as a critical task in music information retrieval due to the abstract and descriptive nature of chords in music analysis. While audio chord recognition systems have achieved significant accuracy for small vocabularies (e.g., major/minor chords), large-vocabulary chord recognition remains a challenging problem. This complexity also arises from the inherent long-tail distribution of chords, where rare chord types are underrepresented in most datasets, leading to insufficient training samples. Effective chord recognition requires leveraging contextual information from audio sequences, yet existing models, such as combinations of convolutional neural networks, bidirectional long short-term memory networks, and bidirectional transformers, face limitations in capturing long-term dependencies and exhibit suboptimal performance on large-vocabulary chord recognition tasks. This work proposes ChordFormer, a novel conformer-based architecture designed to tackle structural chord recognition (e.g., triads, bass, sevenths) for large vocabularies. ChordFormer leverages conformer blocks that integrate convolutional neural networks with transformers, thus enabling the model to capture both local patterns and global dependencies effectively. By addressing challenges such as class imbalance through a reweighted loss function and structured chord representations, ChordFormer outperforms state-of-the-art models, achieving a 2% improvement in frame-wise accuracy and a 6% increase in class-wise accuracy on large-vocabulary chord datasets. Furthermore, ChordFormer excels in handling class imbalance, providing robust and balanced recognition across chord types. This approach bridges the gap between theoretical music knowledge and practical applications, advancing the field of large-vocabulary chord recognition. 

**Abstract (ZH)**: ChordFormer：一种用于大词汇量和弦识别的新型 conformer 基础架构 

---
# Intuitive physics understanding emerges from self-supervised pretraining on natural videos 

**Title (ZH)**: 直觉物理理解源自自然视频的自我监督预训练 

**Authors**: Quentin Garrido, Nicolas Ballas, Mahmoud Assran, Adrien Bardes, Laurent Najman, Michael Rabbat, Emmanuel Dupoux, Yann LeCun  

**Link**: [PDF](https://arxiv.org/pdf/2502.11831)  

**Abstract**: We investigate the emergence of intuitive physics understanding in general-purpose deep neural network models trained to predict masked regions in natural videos. Leveraging the violation-of-expectation framework, we find that video prediction models trained to predict outcomes in a learned representation space demonstrate an understanding of various intuitive physics properties, such as object permanence and shape consistency. In contrast, video prediction in pixel space and multimodal large language models, which reason through text, achieve performance closer to chance. Our comparisons of these architectures reveal that jointly learning an abstract representation space while predicting missing parts of sensory input, akin to predictive coding, is sufficient to acquire an understanding of intuitive physics, and that even models trained on one week of unique video achieve above chance performance. This challenges the idea that core knowledge -- a set of innate systems to help understand the world -- needs to be hardwired to develop an understanding of intuitive physics. 

**Abstract (ZH)**: 我们在训练用于预测自然视频中遮蔽区域的一般深度神经网络模型中探讨直观物理理解的涌现。利用违反预期框架，我们发现，在学习表示空间中训练以预测结果的视频预测模型展示了对各种直观物理特性的理解，如物体恒在性和形状一致性。相比之下，在像素空间中进行视频预测以及通过文本进行推理的多模态大规模语言模型的表现接近随机水平。我们的架构比较表明，同时学习一个抽象表示空间并预测感觉输入的缺失部分，类似于预测编码，足以获得直观物理的理解，并且即使在仅训练一周的独特视频数据下，模型也能达到超过随机水平的表现。这挑战了核心知识——一套有助于理解世界的内置系统——需要硬编码才能发展出直观物理理解的观点。 

---
# Code-Vision: Evaluating Multimodal LLMs Logic Understanding and Code Generation Capabilities 

**Title (ZH)**: Code-Vision: 评估多模态LLM的逻辑理解与代码生成能力 

**Authors**: Hanbin Wang, Xiaoxuan Zhou, Zhipeng Xu, Keyuan Cheng, Yuxin Zuo, Kai Tian, Jingwei Song, Junting Lu, Wenhui Hu, Xueyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11829)  

**Abstract**: This paper introduces Code-Vision, a benchmark designed to evaluate the logical understanding and code generation capabilities of Multimodal Large Language Models (MLLMs). It challenges MLLMs to generate a correct program that fulfills specific functionality requirements based on a given flowchart, which visually represents the desired algorithm or process. Code-Vision comprises three subsets: HumanEval-V, Algorithm, and MATH, which evaluate MLLMs' coding abilities across basic programming, algorithmic, and mathematical problem-solving domains. Our experiments evaluate 12 MLLMs on Code-Vision. Experimental results demonstrate that there is a large performance difference between proprietary and open-source models. On Hard problems, GPT-4o can achieve 79.3% pass@1, but the best open-source model only achieves 15%. Further experiments reveal that Code-Vision can pose unique challenges compared to other multimodal reasoning benchmarks MMCode and MathVista. We also explore the reason for the poor performance of the open-source models. All data and codes are available at this https URL. 

**Abstract (ZH)**: 这篇论文介绍了Code-Vision，一个用于评估多模态大型语言模型（MLLMs）的逻辑理解和代码生成能力的基准。Code-Vision挑战MLLMs根据给定的流程图生成满足特定功能要求的正确程序，流程图可视化表示所需的算法或过程。Code-Vision包含三个子集：HumanEval-V、Algorithm和MATH，分别评估MLLMs在基础编程、算法和数学问题解决领域的编码能力。我们的实验在Code-Vision上评估了12个MLLMs。实验结果表明，专有模型和开源模型之间存在巨大的性能差异。在难题上，GPT-4o可达到79.3%的pass@1，而最好的开源模型仅达到15%。进一步的实验揭示，与多模态推理基准MMCode和MathVista相比，Code-Vision提出了独特的挑战。我们还探讨了开源模型表现不佳的原因。所有数据和代码可在以下链接获取。 

---
# Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis 

**Title (ZH)**: 通过电路分析理解大规模语言模型微调机制 

**Authors**: Xu Wang, Yan Hu, Wenyu Du, Reynold Cheng, Benyou Wang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2502.11812)  

**Abstract**: Fine-tuning significantly improves the performance of Large Language Models (LLMs), yet its underlying mechanisms remain poorly understood. This paper aims to provide an in-depth interpretation of the fine-tuning process through circuit analysis, a popular tool in Mechanistic Interpretability (MI). Unlike previous studies \cite{prakash2024finetuningenhancesexistingmechanisms,chhabra2024neuroplasticity} that focus on tasks where pre-trained models already perform well, we develop a set of mathematical tasks where fine-tuning yields substantial performance gains, which are closer to the practical setting. In our experiments, we identify circuits at various checkpoints during fine-tuning and examine the interplay between circuit analysis, fine-tuning methods, and task complexities. First, we find that while circuits maintain high node similarity before and after fine-tuning, their edges undergo significant changes, which is in contrast to the previous work \cite{prakash2024finetuningenhancesexistingmechanisms,chhabra2024neuroplasticity} that show circuits only add some additional components after fine-tuning. Based on these observations, we develop a circuit-aware Low-Rank Adaptation (LoRA) method, which assigns ranks to layers based on edge changes in the circuits. Experimental results demonstrate that our circuit-based LoRA algorithm achieves an average performance improvement of 2.46\% over standard LoRA with similar parameter sizes. Furthermore, we explore how combining circuits from subtasks can enhance fine-tuning in compositional tasks, providing new insights into the design of such tasks and deepening the understanding of circuit dynamics and fine-tuning mechanisms. 

**Abstract (ZH)**: Fine-tuning显著提高了大型语言模型（LLMs）的性能，但其 underlying机制仍不明确。通过电路分析，一种机制可解释性（MI）中的流行工具，本文旨在深入解释fine-tuning过程。不同于先前研究\[prakash2024finetuningenhancesexistingmechanisms,chhabra2024neuroplasticity\]主要关注预训练模型已在其中表现良好的任务，我们开发了一组数学任务，在这些任务中fine-tuning带来了显著的性能提升，更接近实际应用场景。在实验中，我们识别了fine-tuning过程中的多个检查点处的电路，并探讨了电路分析、fine-tuning方法和任务复杂性之间的相互作用。首先，我们发现虽然电路在fine-tuning前后节点相似度保持较高，但其边经历了显著变化，这与先前工作\[prakash2024finetuningenhancesexistingmechanisms,chhabra2024neuroplasticity\]中仅显示fine-tuning后电路增加了一些额外组件的观察结果相反。基于这些观察，我们开发了一种电路感知低秩适应（LoRA）方法，根据电路中边的变化为层分配秩。实验结果表明，我们的电路基础LoRA算法在相似参数量的情况下，平均性能改进了2.46%。此外，我们探索了组合子任务电路如何增强合成任务的fine-tuning，为这类任务的设计提供了新的见解，并加深了对电路动态和fine-tuning机制的理解。 

---
# Revealing Bias Formation in Deep Neural Networks Through the Geometric Mechanisms of Human Visual Decoupling 

**Title (ZH)**: 通过人类视觉去耦的几何机制揭示深度神经网络中的偏见形成 

**Authors**: Yanbiao Ma, Bowei Liu, Wei Dai, Jiayi Chen, Shuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.11809)  

**Abstract**: Deep neural networks (DNNs) often exhibit biases toward certain categories during object recognition, even under balanced training data conditions. The intrinsic mechanisms underlying these biases remain unclear. Inspired by the human visual system, which decouples object manifolds through hierarchical processing to achieve object recognition, we propose a geometric analysis framework linking the geometric complexity of class-specific perceptual manifolds in DNNs to model bias. Our findings reveal that differences in geometric complexity can lead to varying recognition capabilities across categories, introducing biases. To support this analysis, we present the Perceptual-Manifold-Geometry library, designed for calculating the geometric properties of perceptual manifolds. 

**Abstract (ZH)**: 深度神经网络（DNNs）在对象识别过程中即使在平衡训练数据条件下也常常对某些类别表现出偏差。受人类视觉系统通过分层处理解耦对象流形实现对象识别的启发，我们提出了一种几何分析框架，将DNN中类特定知觉流形的几何复杂性与模型偏差联系起来。我们的研究发现，几何复杂性差异会导致不同类别识别能力的差异，从而引入偏差。为了支持这种分析，我们介绍了感知流形几何库，用于计算感知流形的几何属性。 

---
# Deep Neural Networks for Accurate Depth Estimation with Latent Space Features 

**Title (ZH)**: 基于潜在空间特征的深度神经网络accurate深度估计 

**Authors**: Siddiqui Muhammad Yasir, Hyunsik Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2502.11777)  

**Abstract**: Depth estimation plays a pivotal role in advancing human-robot interactions, especially in indoor environments where accurate 3D scene reconstruction is essential for tasks like navigation and object handling. Monocular depth estimation, which relies on a single RGB camera, offers a more affordable solution compared to traditional methods that use stereo cameras or LiDAR. However, despite recent progress, many monocular approaches struggle with accurately defining depth boundaries, leading to less precise reconstructions. In response to these challenges, this study introduces a novel depth estimation framework that leverages latent space features within a deep convolutional neural network to enhance the precision of monocular depth maps. The proposed model features dual encoder-decoder architecture, enabling both color-to-depth and depth-to-depth transformations. This structure allows for refined depth estimation through latent space encoding. To further improve the accuracy of depth boundaries and local features, a new loss function is introduced. This function combines latent loss with gradient loss, helping the model maintain the integrity of depth boundaries. The framework is thoroughly tested using the NYU Depth V2 dataset, where it sets a new benchmark, particularly excelling in complex indoor scenarios. The results clearly show that this approach effectively reduces depth ambiguities and blurring, making it a promising solution for applications in human-robot interaction and 3D scene reconstruction. 

**Abstract (ZH)**: 基于隐空间特征的单目深度估计框架在人机交互和3D场景重建中的应用 

---
# The Validation Gap: A Mechanistic Analysis of How Language Models Compute Arithmetic but Fail to Validate It 

**Title (ZH)**: 验证差距：语言模型在计算算术问题时验证不足的机理分析 

**Authors**: Leonardo Bertolazzi, Philipp Mondorf, Barbara Plank, Raffaella Bernardi  

**Link**: [PDF](https://arxiv.org/pdf/2502.11771)  

**Abstract**: The ability of large language models (LLMs) to validate their output and identify potential errors is crucial for ensuring robustness and reliability. However, current research indicates that LLMs struggle with self-correction, encountering significant challenges in detecting errors. While studies have explored methods to enhance self-correction in LLMs, relatively little attention has been given to understanding the models' internal mechanisms underlying error detection. In this paper, we present a mechanistic analysis of error detection in LLMs, focusing on simple arithmetic problems. Through circuit analysis, we identify the computational subgraphs responsible for detecting arithmetic errors across four smaller-sized LLMs. Our findings reveal that all models heavily rely on $\textit{consistency heads}$--attention heads that assess surface-level alignment of numerical values in arithmetic solutions. Moreover, we observe that the models' internal arithmetic computation primarily occurs in higher layers, whereas validation takes place in middle layers, before the final arithmetic results are fully encoded. This structural dissociation between arithmetic computation and validation seems to explain why current LLMs struggle to detect even simple arithmetic errors. 

**Abstract (ZH)**: 大型语言模型中错误检测机制的分析：以简单算术问题为例 

---
# Lightweight Deepfake Detection Based on Multi-Feature Fusion 

**Title (ZH)**: 基于多特征融合的轻量级 deepfake 检测 

**Authors**: Siddiqui Muhammad Yasir, Hyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.11763)  

**Abstract**: Deepfake technology utilizes deep learning based face manipulation techniques to seamlessly replace faces in videos creating highly realistic but artificially generated content. Although this technology has beneficial applications in media and entertainment misuse of its capabilities may lead to serious risks including identity theft cyberbullying and false information. The integration of DL with visual cognition has resulted in important technological improvements particularly in addressing privacy risks caused by artificially generated deepfake images on digital media platforms. In this study we propose an efficient and lightweight method for detecting deepfake images and videos making it suitable for devices with limited computational resources. In order to reduce the computational burden usually associated with DL models our method integrates machine learning classifiers in combination with keyframing approaches and texture analysis. Moreover the features extracted with a histogram of oriented gradients (HOG) local binary pattern (LBP) and KAZE bands were integrated to evaluate using random forest extreme gradient boosting extra trees and support vector classifier algorithms. Our findings show a feature-level fusion of HOG LBP and KAZE features improves accuracy to 92% and 96% on FaceForensics++ and Celeb-DFv2 respectively. 

**Abstract (ZH)**: Deepfake技术利用基于深度学习的面部操控技术在视频中无缝替换面部，生成高度逼真的但人为生成的内容。尽管这项技术在媒体和娱乐领域具有有益的应用，但对其能力的滥用可能导致身份盗窃、网络欺凌和虚假信息等严重风险。将DL与视觉认知的结合导致了重要的技术进步，特别是在解决数字媒体平台上由人工生成的deepfake图像引起的数据隐私风险方面。在本研究中，我们提出了一种高效且轻量级的方法来检测deepfake图像和视频，使其适用于具有有限计算资源的设备。为了减少通常与DL模型相关的计算负担，我们的方法结合了机器学习分类器和关键帧方法以及纹理分析。此外，利用HOG、LBP和KAZE特征，并使用随机森林、极端梯度提升、额外树和支持向量分类器算法进行评估。我们的研究发现，HOG、LBP和KAZE特征的特征级融合分别在FaceForensics++和Celeb-DFv2数据集上的准确率达到92%和96%。 

---
# On the Computation of the Fisher Information in Continual Learning 

**Title (ZH)**: 连续学习中 Fisher 信息的计算 

**Authors**: Gido M. van de Ven  

**Link**: [PDF](https://arxiv.org/pdf/2502.11756)  

**Abstract**: One of the most popular methods for continual learning with deep neural networks is Elastic Weight Consolidation (EWC), which involves computing the Fisher Information. The exact way in which the Fisher Information is computed is however rarely described, and multiple different implementations for it can be found online. This blog post discusses and empirically compares several often-used implementations, which highlights that many currently reported results for EWC could likely be improved by changing the way the Fisher Information is computed. 

**Abstract (ZH)**: 弹性权重汇聚中的Fishers信息计算：多种实现的比较与改进潜力 

---
# Language Models Can See Better: Visual Contrastive Decoding For LLM Multimodal Reasoning 

**Title (ZH)**: 语言模型视觉对比解码：面向LLM多模态推理 

**Authors**: Yuqi Pang, Bowen Yang, Haoqin Tu, Yun Cao, Zeyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11751)  

**Abstract**: Although Large Language Models (LLMs) excel in reasoning and generation for language tasks, they are not specifically designed for multimodal challenges. Training Multimodal Large Language Models (MLLMs), however, is resource-intensive and constrained by various training limitations. In this paper, we propose the Modular-based Visual Contrastive Decoding (MVCD) framework to move this obstacle. Our framework leverages LLMs' In-Context Learning (ICL) capability and the proposed visual contrastive-example decoding (CED), specifically tailored for this framework, without requiring any additional training. By converting visual signals into text and focusing on contrastive output distributions during decoding, we can highlight the new information introduced by contextual examples, explore their connections, and avoid over-reliance on prior encoded knowledge. MVCD enhances LLMs' visual perception to make it see and reason over the input visuals. To demonstrate MVCD's effectiveness, we conduct experiments with four LLMs across five question answering datasets. Our results not only show consistent improvement in model accuracy but well explain the effective components inside our decoding strategy. Our code will be available at this https URL. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在语言任务上的推理和生成表现出色，但它们并未专门设计用于多模态挑战。然而，训练多模态大型语言模型（MLLMs）是资源密集型的，并受到各种训练限制。在本文中，我们提出了一种模块化视觉对比解码（MVCD）框架，以克服这一障碍。我们的框架利用了LLMs的上下文学习（ICL）能力，并提出了针对该框架的视觉对比实例解码（CED），无需任何额外训练。通过将视觉信号转换为文本，在解码过程中专注于对比输出分布，我们可以突出上下文示例引入的新信息，探索它们之间的联系，并避免过度依赖先验编码知识。MVCD增强了LLMs的视觉感知能力，使其能够理解和推理输入的视觉内容。为了证明MVCD的有效性，我们在五个问答数据集中对四个LLM进行了实验。我们的结果不仅展示了模型准确性的持续改进，还详细解释了我们解码策略中的有效组件。我们的代码将在以下链接处提供：this https URL。 

---
# JotlasNet: Joint Tensor Low-Rank and Attention-based Sparse Unrolling Network for Accelerating Dynamic MRI 

**Title (ZH)**: JotlasNet：联合张量低秩和注意力基于稀疏解卷网络加速动态MRI 

**Authors**: Yinghao Zhang, Haiyan Gui, Ningdi Yang, Yue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11749)  

**Abstract**: Joint low-rank and sparse unrolling networks have shown superior performance in dynamic MRI reconstruction. However, existing works mainly utilized matrix low-rank priors, neglecting the tensor characteristics of dynamic MRI images, and only a global threshold is applied for the sparse constraint to the multi-channel data, limiting the flexibility of the network. Additionally, most of them have inherently complex network structure, with intricate interactions among variables. In this paper, we propose a novel deep unrolling network, JotlasNet, for dynamic MRI reconstruction by jointly utilizing tensor low-rank and attention-based sparse priors. Specifically, we utilize tensor low-rank prior to exploit the structural correlations in high-dimensional data. Convolutional neural networks are used to adaptively learn the low-rank and sparse transform domains. A novel attention-based soft thresholding operator is proposed to assign a unique learnable threshold to each channel of the data in the CNN-learned sparse domain. The network is unrolled from the elaborately designed composite splitting algorithm and thus features a simple yet efficient parallel structure. Extensive experiments on two datasets (OCMR, CMRxRecon) demonstrate the superior performance of JotlasNet in dynamic MRI reconstruction. 

**Abstract (ZH)**: 联合低秩和注意力引导稀疏展开网络在动态MRI重建中的应用研究 

---
# SQL-o1: A Self-Reward Heuristic Dynamic Search Method for Text-to-SQL 

**Title (ZH)**: SQL-o1: 一种自我奖励启发式动态搜索方法实现文本到SQL 

**Authors**: Shuai Lyu, Haoran Luo, Zhonghong Ou, Yifan Zhu, Xiaoran Shang, Yang Qin, Meina Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.11741)  

**Abstract**: The Text-to-SQL(Text2SQL) task aims to convert natural language queries into executable SQL queries. Thanks to the application of large language models (LLMs), significant progress has been made in this field. However, challenges such as model scalability, limited generation space, and coherence issues in SQL generation still persist. To address these issues, we propose SQL-o1, a Self-Reward-based heuristic search method designed to enhance the reasoning ability of LLMs in SQL query generation. SQL-o1 combines Monte Carlo Tree Search (MCTS) for heuristic process-level search and constructs a Schema-Aware dataset to help the model better understand database schemas. Extensive experiments on the Bird and Spider datasets demonstrate that SQL-o1 improves execution accuracy by 10.8\% on the complex Bird dataset compared to the latest baseline methods, even outperforming GPT-4-based approaches. Additionally, SQL-o1 excels in few-shot learning scenarios and shows strong cross-model transferability. Our code is publicly available at:this https URL. 

**Abstract (ZH)**: Text-to-SQL(Text2SQL)任务旨在将自然语言查询转换为可执行的SQL查询。得益于大型语言模型（LLMs）的应用，该领域取得了显著进步。然而，模型可扩展性、生成空间有限以及SQL生成中的连贯性问题仍然存在。为解决这些问题，我们提出了SQL-o1，这是一种基于自我奖励的启发式搜索方法，旨在增强LLMs在SQL查询生成中的推理能力。SQL-o1结合了蒙特卡洛树搜索（MCTS）进行启发式过程级搜索，并构建了一个模式感知数据集以帮助模型更好地理解数据库模式。在Bird和Spider数据集上的广泛实验表明，与最新基准方法相比，SQL-o1在复杂Bird数据集上的执行准确率提高了10.8%，甚至优于基于GPT-4的方法。此外，SQL-o1在少量样本学习场景中表现出色，并显示出较强的跨模型迁移能力。相关代码已公开。 

---
# ReviewEval: An Evaluation Framework for AI-Generated Reviews 

**Title (ZH)**: ReviewEval：AI生成评论的评估框架 

**Authors**: Chavvi Kirtani, Madhav Krishan Garg, Tejash Prasad, Tanmay Singhal, Murari Mandal, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.11736)  

**Abstract**: The escalating volume of academic research, coupled with a shortage of qualified reviewers, necessitates innovative approaches to peer review. While large language model (LLMs) offer potential for automating this process, their current limitations include superficial critiques, hallucinations, and a lack of actionable insights. This research addresses these challenges by introducing a comprehensive evaluation framework for AI-generated reviews, that measures alignment with human evaluations, verifies factual accuracy, assesses analytical depth, and identifies actionable insights. We also propose a novel alignment mechanism that tailors LLM-generated reviews to the unique evaluation priorities of individual conferences and journals. To enhance the quality of these reviews, we introduce a self-refinement loop that iteratively optimizes the LLM's review prompts. Our framework establishes standardized metrics for evaluating AI-based review systems, thereby bolstering the reliability of AI-generated reviews in academic research. 

**Abstract (ZH)**: escalating 学术研究 volume 的增加与合格评审人短缺并存， necessitates 创新性的同行评审方法。尽管大规模语言模型（LLMs）在自动化这一过程方面具有潜力，但它们目前的局限性包括表面化的评论、胡言乱语以及缺乏可操作的见解。本研究通过引入一个全面的 AI 生成评论评估框架来应对这些挑战，该框架衡量与人力评估的一致性，验证事实准确性，评估分析深度，并识别可操作的见解。我们还提出了一种新颖的一致性机制，使 LLM 生成的评论符合各个会议和期刊的独特评估优先级。为了提高这些评论的质量，我们引入了一个自我精炼循环，迭代优化 LLM 的评论提示。该框架建立了评估基于 AI 的评审系统的标准化指标，从而增强了 AI 生成评审在学术研究中的可靠性。 

---
# Proactive Depot Discovery: A Generative Framework for Flexible Location-Routing 

**Title (ZH)**: 前瞻性的补给点发现：一种灵活的位置-路由生成框架 

**Authors**: Site Qu, Guoqiang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11715)  

**Abstract**: The Location-Routing Problem (LRP), which combines the challenges of facility (depot) locating and vehicle route planning, is critically constrained by the reliance on predefined depot candidates, limiting the solution space and potentially leading to suboptimal outcomes. Previous research on LRP without predefined depots is scant and predominantly relies on heuristic algorithms that iteratively attempt depot placements across a planar area. Such approaches lack the ability to proactively generate depot locations that meet specific geographic requirements, revealing a notable gap in current research landscape. To bridge this gap, we propose a data-driven generative DRL framework, designed to proactively generate depots for LRP without predefined depot candidates, solely based on customer requests data which include geographic and demand information. It can operate in two distinct modes: direct generation of exact depot locations, and the creation of a multivariate Gaussian distribution for flexible depots sampling. By extracting depots' geographic pattern from customer requests data, our approach can dynamically respond to logistical needs, identifying high-quality depot locations that further reduce total routing costs compared to traditional methods. Extensive experiments demonstrate that, for a same group of customer requests, compared with those depots identified through random attempts, our framework can proactively generate depots that lead to superior solution routes with lower routing cost. The implications of our framework potentially extend into real-world applications, particularly in emergency medical rescue and disaster relief logistics, where rapid establishment and adjustment of depot locations are paramount, showcasing its potential in addressing LRP for dynamic and unpredictable environments. 

**Abstract (ZH)**: 基于数据驱动生成的DRL框架：在无预设仓库候选情况下解决位置-路径规划问题 

---
# Knowledge-aware contrastive heterogeneous molecular graph learning 

**Title (ZH)**: 知识 Aware 对比异构分子图学习 

**Authors**: Mukun Chen, Jia Wu, Shirui Pan, Fu Lin, Bo Du, Xiuwen Gong, Wenbin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11711)  

**Abstract**: Molecular representation learning is pivotal in predicting molecular properties and advancing drug design. Traditional methodologies, which predominantly rely on homogeneous graph encoding, are limited by their inability to integrate external knowledge and represent molecular structures across different levels of granularity. To address these limitations, we propose a paradigm shift by encoding molecular graphs into heterogeneous structures, introducing a novel framework: Knowledge-aware Contrastive Heterogeneous Molecular Graph Learning (KCHML). This approach leverages contrastive learning to enrich molecular representations with embedded external knowledge. KCHML conceptualizes molecules through three distinct graph views-molecular, elemental, and pharmacological-enhanced by heterogeneous molecular graphs and a dual message-passing mechanism. This design offers a comprehensive representation for property prediction, as well as for downstream tasks such as drug-drug interaction (DDI) prediction. Extensive benchmarking demonstrates KCHML's superiority over state-of-the-art molecular property prediction models, underscoring its ability to capture intricate molecular features. 

**Abstract (ZH)**: 分子表示学习对于预测分子性质和推动药物设计至关重要。传统的 homogeneous 图编码方法受限于无法整合外部知识和在不同粒度级别表示分子结构。为克服这些限制，我们提出了一种新的范式转变，即将分子图编码为异构结构，并引入了一个新的框架：知觉对比异构分子图学习（KCHML）。该方法利用对比学习来通过嵌入外部知识丰富分子表示。KCHML 通过三种不同的图视角——分子视角、元素视角和药理学增强，结合异构分子图和双消息传递机制来概念化分子。该设计为性质预测以及下游任务如药物-药物相互作用（DDI）预测提供了全面的表示。广泛的基准测试表明，KCHML 在分子性质预测模型中表现出色，证明了其捕捉复杂分子特征的能力。 

---
# LLM Agents Making Agent Tools 

**Title (ZH)**: LLM代理制作代理工具 

**Authors**: Georg Wölflein, Dyke Ferber, Daniel Truhn, Ognjen Arandjelović, Jakob Nikolas Kather  

**Link**: [PDF](https://arxiv.org/pdf/2502.11705)  

**Abstract**: Tool use has turned large language models (LLMs) into powerful agents that can perform complex multi-step tasks by dynamically utilising external software components. However, these tools must be implemented in advance by human developers, hindering the applicability of LLM agents in domains which demand large numbers of highly specialised tools, like in life sciences and medicine. Motivated by the growing trend of scientific studies accompanied by public code repositories, we propose ToolMaker, a novel agentic framework that autonomously transforms papers with code into LLM-compatible tools. Given a short task description and a repository URL, ToolMaker autonomously installs required dependencies and generates code to perform the task, using a closed-loop self-correction mechanism to iteratively diagnose and rectify errors. To evaluate our approach, we introduce a benchmark comprising 15 diverse and complex computational tasks spanning both medical and non-medical domains with over 100 unit tests to objectively assess tool correctness and robustness. ToolMaker correctly implements 80% of the tasks, substantially outperforming current state-of-the-art software engineering agents. ToolMaker therefore is a step towards fully autonomous agent-based scientific workflows. 

**Abstract (ZH)**: 工具使用使大型语言模型（LLMs）成为能够通过动态利用外部软件组件执行复杂多步任务的强大代理。然而，这些工具需要由人类开发人员预先实现，限制了LLM代理在需要大量高度专门化工具的领域（如生命科学和医学）的应用。鉴于科学研究中公共代码存储库日益增长的趋势，我们提出了ToolMaker，一种新型的代理框架，能够自主将带代码的论文转换为LLM兼容的工具。给定简要的任务描述和仓库URL，ToolMaker自主安装所需的依赖项并生成代码以执行任务，利用闭环自我纠正机制迭代诊断并修正错误。为了评估我们的方法，我们引入了一个基准测试集，其中包括15个涵盖医学和非医学领域的多样且复杂的计算任务，包含超过100个单元测试，以客观评估工具的正确性和稳健性。ToolMaker成功实现80%的任务，显著优于当前最先进的软件工程代理。因此，ToolMaker是完全自主的基于代理的科学研究工作流的一个重要步骤。 

---
# ReVeil: Unconstrained Concealed Backdoor Attack on Deep Neural Networks using Machine Unlearning 

**Title (ZH)**: ReVeil: 使用机器忘却对深度神经网络进行不受约束的隐藏后门攻击 

**Authors**: Manaar Alam, Hithem Lamri, Michail Maniatakos  

**Link**: [PDF](https://arxiv.org/pdf/2502.11687)  

**Abstract**: Backdoor attacks embed hidden functionalities in deep neural networks (DNN), triggering malicious behavior with specific inputs. Advanced defenses monitor anomalous DNN inferences to detect such attacks. However, concealed backdoors evade detection by maintaining a low pre-deployment attack success rate (ASR) and restoring high ASR post-deployment via machine unlearning. Existing concealed backdoors are often constrained by requiring white-box or black-box access or auxiliary data, limiting their practicality when such access or data is unavailable. This paper introduces ReVeil, a concealed backdoor attack targeting the data collection phase of the DNN training pipeline, requiring no model access or auxiliary data. ReVeil maintains low pre-deployment ASR across four datasets and four trigger patterns, successfully evades three popular backdoor detection methods, and restores high ASR post-deployment through machine unlearning. 

**Abstract (ZH)**: 隐蔽后门攻击将隐藏功能嵌入深度神经网络（DNN），在特定输入下触发恶意行为。高级防御措施通过监控异常的DNN推理结果来检测此类攻击。然而，隐蔽后门通过保持低部署前攻击成功率（ASR）并在部署后通过机器遗忘恢复高ASR来规避检测。现有的隐蔽后门往往需要白盒或黑盒访问或辅助数据，这限制了它们在无法获取此类访问或数据时的实用性。本文介绍了一种针对DNN训练管道数据收集阶段的隐蔽后门攻击——ReVeil，该攻击无需模型访问或辅助数据。ReVeil在四个数据集和四种触发模式下保持低部署前ASR，并成功规避了三种流行的后门检测方法，在部署后通过机器遗忘恢复高ASR。 

---
# MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task 

**Title (ZH)**: MathFimer：通过填充中间任务扩展推理步骤以增强数学推理能力 

**Authors**: Yuchen Yan, Yongliang Shen, Yang Liu, Jin Jiang, Xin Xu, Mengdi Zhang, Jian Shao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11684)  

**Abstract**: Mathematical reasoning represents a critical frontier in advancing large language models (LLMs). While step-by-step approaches have emerged as the dominant paradigm for mathematical problem-solving in LLMs, the quality of reasoning steps in training data fundamentally constrains the performance of the models. Recent studies has demonstrated that more detailed intermediate steps can enhance model performance, yet existing methods for step expansion either require more powerful external models or incur substantial computational costs. In this paper, we introduce MathFimer, a novel framework for mathematical reasoning step expansion inspired by the "Fill-in-the-middle" task from code completion. By decomposing solution chains into prefix-suffix pairs and training models to reconstruct missing intermediate steps, we develop a specialized model, MathFimer-7B, on our carefully curated NuminaMath-FIM dataset. We then apply these models to enhance existing mathematical reasoning datasets by inserting detailed intermediate steps into their solution chains, creating MathFimer-expanded versions. Through comprehensive experiments on multiple mathematical reasoning datasets, including MathInstruct, MetaMathQA and etc., we demonstrate that models trained on MathFimer-expanded data consistently outperform their counterparts trained on original data across various benchmarks such as GSM8K and MATH. Our approach offers a practical, scalable solution for enhancing mathematical reasoning capabilities in LLMs without relying on powerful external models or expensive inference procedures. 

**Abstract (ZH)**: 数学推理是推动大规模语言模型（LLMs）发展的关键前沿领域。尽管逐步方法已成为LLMs中数学问题求解的主要范式，但训练数据中推理步骤的质量从根本上限制了模型的性能。最近的研究表明，更详细的中间步骤可以提升模型性能，但现有的方法要么需要更强大的外部模型，要么会带来巨大的计算成本。在本文中，我们介绍了一种名为MathFimer的新型数学推理步骤扩展框架，该框架受到代码完成任务中“填空”任务的启发。通过将解答链分解为前缀-后缀对，并训练模型重建缺失的中间步骤，我们在我们精心策划的NuminaMath-FIM数据集上开发了一个专门的模型，MathFimer-7B。然后，我们使用这些模型通过在原始解答链中插入详细的中间步骤来增强现有的数学推理数据集，创建了MathFimer扩展版本。通过在MathInstruct、MetaMathQA等多个数学推理数据集上的综合性实验，我们证明了基于MathFimer扩展数据训练的模型在包括GSM8K和MATH在内的各种基准测试中始终优于基于原始数据训练的模型。我们的方法提供了一种无需依赖强大外部模型或昂贵推断程序的实际可扩展方案，以提升LLMs的数学推理能力。 

---
# RIDE: Enhancing Large Language Model Alignment through Restyled In-Context Learning Demonstration Exemplars 

**Title (ZH)**: RIDE：通过重塑情境学习示范范例增强大型语言模型对齐 

**Authors**: Yuncheng Hua, Lizhen Qu, Zhuang Li, Hao Xue, Flora D. Salim, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2502.11681)  

**Abstract**: Alignment tuning is crucial for ensuring large language models (LLMs) behave ethically and helpfully. Current alignment approaches require high-quality annotations and significant training resources. This paper proposes a low-cost, tuning-free method using in-context learning (ICL) to enhance LLM alignment. Through an analysis of high-quality ICL demos, we identified style as a key factor influencing LLM alignment capabilities and explicitly restyled ICL exemplars based on this stylistic framework. Additionally, we combined the restyled demos to achieve a balance between the two conflicting aspects of LLM alignment--factuality and safety. We packaged the restyled examples as prompts to trigger few-shot learning, improving LLM alignment. Compared to the best baseline approach, with an average score of 5.00 as the maximum, our method achieves a maximum 0.10 increase on the Alpaca task (from 4.50 to 4.60), a 0.22 enhancement on the Just-eval benchmark (from 4.34 to 4.56), and a maximum improvement of 0.32 (from 3.53 to 3.85) on the MT-Bench dataset. We release the code and data at this https URL. 

**Abstract (ZH)**: 低花费无调优上下文学习提升大型语言模型对齐toy实验 

---
# Diversity-Oriented Data Augmentation with Large Language Models 

**Title (ZH)**: 面向多样性数据增强的大语言模型方法 

**Authors**: Zaitian Wang, Jinghan Zhang, Xinhao Zhang, Kunpeng Liu, Pengfei Wang, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.11671)  

**Abstract**: Data augmentation is an essential technique in natural language processing (NLP) for enriching training datasets by generating diverse samples. This process is crucial for improving the robustness and generalization capabilities of NLP models. However, a significant challenge remains: \textit{Insufficient Attention to Sample Distribution Diversity}. Most existing methods focus on increasing the sample numbers while neglecting the sample distribution diversity, which can lead to model overfitting. In response, we explore data augmentation's impact on dataset diversity and propose a \textbf{\underline{D}}iversity-\textbf{\underline{o}}riented data \textbf{\underline{Aug}}mentation framework (\textbf{DoAug}). % \(\mathscr{DoAug}\) Specifically, we utilize a diversity-oriented fine-tuning approach to train an LLM as a diverse paraphraser, which is capable of augmenting textual datasets by generating diversified paraphrases. Then, we apply the LLM paraphraser to a selected coreset of highly informative samples and integrate the paraphrases with the original data to create a more diverse augmented dataset. Finally, we conduct extensive experiments on 12 real-world textual datasets. The results show that our fine-tuned LLM augmenter improves diversity while preserving label consistency, thereby enhancing the robustness and performance of downstream tasks. Specifically, it achieves an average performance gain of \(10.52\%\), surpassing the runner-up baseline with more than three percentage points. 

**Abstract (ZH)**: 数据增强：关注样本分布多样性以提升自然语言处理模型的鲁棒性和泛化能力 

---
# "I'm not for sale" -- Perceptions and limited awareness of privacy risks by digital natives about location data 

**Title (ZH)**: “我不出售”——数字原住民对位置数据隐私风险的感知和有限认识 

**Authors**: Antoine Boutet, Victor Morel  

**Link**: [PDF](https://arxiv.org/pdf/2502.11658)  

**Abstract**: Although mobile devices benefit users in their daily lives in numerous ways, they also raise several privacy concerns. For instance, they can reveal sensitive information that can be inferred from location data. This location data is shared through service providers as well as mobile applications. Understanding how and with whom users share their location data -- as well as users' perception of the underlying privacy risks --, are important notions to grasp in order to design usable privacy-enhancing technologies. In this work, we perform a quantitative and qualitative analysis of smartphone users' awareness, perception and self-reported behavior towards location data-sharing through a survey of n=99 young adult participants (i.e., digital natives). We compare stated practices with actual behaviors to better understand their mental models, and survey participants' understanding of privacy risks before and after the inspection of location traces and the information that can be inferred therefrom.
Our empirical results show that participants have risky privacy practices: about 54% of participants underestimate the number of mobile applications to which they have granted access to their data, and 33% forget or do not think of revoking access to their data. Also, by using a demonstrator to perform inferences from location data, we observe that slightly more than half of participants (57%) are surprised by the extent of potentially inferred information, and that 47% intend to reduce access to their data via permissions as a result of using the demonstrator. Last, a majority of participants have little knowledge of the tools to better protect themselves, but are nonetheless willing to follow suggestions to improve privacy (51%). Educating people, including digital natives, about privacy risks through transparency tools seems a promising approach. 

**Abstract (ZH)**: 尽管移动设备在日常生活中给用户带来了诸多便利，但也引发了一系列隐私问题。例如，它们可能会泄露从位置数据中可以推断出的敏感信息。这些位置数据不仅通过服务提供商，还会通过移动应用程序进行共享。理解用户是如何以及与谁共享其位置数据，以及用户对潜在隐私风险的认知，对于设计易于使用的增强隐私技术至关重要。在本研究中，我们通过对99名年轻成人（即数字原住民）进行调查，开展定量和定性分析，评估他们对位置数据共享的意识、感知和自述行为。我们将声明的实践与实际行为进行比较，以更好地理解他们的心理模型，并在检查位置踪迹及其可推断出的信息之前和之后，了解受访者对隐私风险的理解。我们的实证结果显示，参与者存在风险较高的隐私实践：约54%的参与者低估了已授予数据访问权限的移动应用程序数量，且33%的人忘记了或没有考虑到撤销数据访问权限。此外，通过使用演示器从位置数据中进行推断，我们观察到略多于一半的参与者（57%）对其可推断出的信息范围感到惊讶，并且47%的参与者计划通过调整权限来减少数据访问。最后，大多数参与者虽然对保护自己的工具了解不多，但仍愿意遵循提高隐私的建议（51%）。通过透明工具教育包括数字原住民在内的人们关于隐私风险，似乎是一种有前景的方法。 

---
# MMXU: A Multi-Modal and Multi-X-ray Understanding Dataset for Disease Progression 

**Title (ZH)**: MMXU: 一种多模态和多X光理解疾病进展数据集 

**Authors**: Linjie Mu, Zhongzhen Huang, Shengqian Qin, Yakun Zhu, Shaoting Zhang, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11651)  

**Abstract**: Large vision-language models (LVLMs) have shown great promise in medical applications, particularly in visual question answering (MedVQA) and diagnosis from medical images. However, existing datasets and models often fail to consider critical aspects of medical diagnostics, such as the integration of historical records and the analysis of disease progression over time. In this paper, we introduce MMXU (Multimodal and MultiX-ray Understanding), a novel dataset for MedVQA that focuses on identifying changes in specific regions between two patient visits. Unlike previous datasets that primarily address single-image questions, MMXU enables multi-image questions, incorporating both current and historical patient data. We demonstrate the limitations of current LVLMs in identifying disease progression on MMXU-\textit{test}, even those that perform well on traditional benchmarks. To address this, we propose a MedRecord-Augmented Generation (MAG) approach, incorporating both global and regional historical records. Our experiments show that integrating historical records significantly enhances diagnostic accuracy by at least 20\%, bridging the gap between current LVLMs and human expert performance. Additionally, we fine-tune models with MAG on MMXU-\textit{dev}, which demonstrates notable improvements. We hope this work could illuminate the avenue of advancing the use of LVLMs in medical diagnostics by emphasizing the importance of historical context in interpreting medical images. Our dataset is released at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 大型多模态视觉语言模型（LVLMs）在医疗应用中显示出了巨大的潜力，特别是在医学视觉问答（MedVQA）和医学影像诊断方面。然而，现有的数据集和模型往往没有考虑到医疗诊断中的关键方面，如历史记录的整合以及疾病进展的分析。本文介绍了MMXU（多模态和多X光理解），一个专注于识别两名患者访问之间特定区域变化的新数据集。与主要处理单张图像问题的先前数据集不同，MMXU 支持多图像问题，结合了当前和历史患者数据。我们展示了当前LVLMs在MMXU-\textit{test} 中识别疾病进展的局限性，即使它们在传统基准测试中表现良好。为此，我们提出了MedRecord-Augmented Generation（MAG）方法，结合了全局和区域历史记录。我们的实验表明，整合历史记录至少可以提高20% 的诊断准确性，缩小当前LVLMs与人类专家表现之间的差距。此外，我们使用MAG对MMXU-\textit{dev} 进行微调，这显示了显著的改进。我们希望这项工作能够强调历史背景在解释医学影像中的重要性，从而促进LVLMs在医疗诊断中的应用。我们的数据集在 \href{this https URL}{this https URL} 发布。 

---
# DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing 

**Title (ZH)**: DELMAN: 基于模型编辑动态防御大型语言模型越狱攻击 

**Authors**: Yi Wang, Fenghua Weng, Sibei Yang, Zhan Qin, Minlie Huang, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11647)  

**Abstract**: Large Language Models (LLMs) are widely applied in decision making, but their deployment is threatened by jailbreak attacks, where adversarial users manipulate model behavior to bypass safety measures. Existing defense mechanisms, such as safety fine-tuning and model editing, either require extensive parameter modifications or lack precision, leading to performance degradation on general tasks, which is unsuitable to post-deployment safety alignment. To address these challenges, we propose DELMAN (Dynamic Editing for LLMs JAilbreak DefeNse), a novel approach leveraging direct model editing for precise, dynamic protection against jailbreak attacks. DELMAN directly updates a minimal set of relevant parameters to neutralize harmful behaviors while preserving the model's utility. To avoid triggering a safe response in benign context, we incorporate KL-divergence regularization to ensure the updated model remains consistent with the original model when processing benign queries. Experimental results demonstrate that DELMAN outperforms baseline methods in mitigating jailbreak attacks while preserving the model's utility, and adapts seamlessly to new attack instances, providing a practical and efficient solution for post-deployment model protection. 

**Abstract (ZH)**: 动态编辑以抵御大型语言模型 jailbreak 攻击（DELMAN） 

---
# InTec: integrated things-edge computing: a framework for distributing machine learning pipelines in edge AI systems 

**Title (ZH)**: InTec: 综合物联网-边缘计算框架：边缘AI系统中分布机器学习管道的框架 

**Authors**: Habib Larian, Faramarz Safi-Esfahani  

**Link**: [PDF](https://arxiv.org/pdf/2502.11644)  

**Abstract**: With the rapid expansion of the Internet of Things (IoT), sensors, smartphones, and wearables have become integral to daily life, powering smart applications in home automation, healthcare, and intelligent transportation. However, these advancements face significant challenges due to latency and bandwidth constraints imposed by traditional cloud based machine learning (ML) frameworks. The need for innovative solutions is evident as cloud computing struggles with increased latency and network congestion. Previous attempts to offload parts of the ML pipeline to edge and cloud layers have yet to fully resolve these issues, often worsening system response times and network congestion due to the computational limitations of edge devices. In response to these challenges, this study introduces the InTec (Integrated Things Edge Computing) framework, a groundbreaking innovation in IoT architecture. Unlike existing methods, InTec fully leverages the potential of a three tier architecture by strategically distributing ML tasks across the Things, Edge, and Cloud layers. This comprehensive approach enables real time data processing at the point of data generation, significantly reducing latency, optimizing network traffic, and enhancing system reliability. InTec effectiveness is validated through empirical evaluation using the MHEALTH dataset for human motion detection in smart homes, demonstrating notable improvements in key metrics: an 81.56 percent reduction in response time, a 10.92 percent decrease in network traffic, a 9.82 percent improvement in throughput, a 21.86 percent reduction in edge energy consumption, and a 25.83 percent reduction in cloud energy consumption. These advancements establish InTec as a new benchmark for scalable, responsive, and energy efficient IoT applications, demonstrating its potential to revolutionize how the ML pipeline is integrated into Edge AI (EI) systems. 

**Abstract (ZH)**: 随物联网（IoT）的快速扩展，传感器、智能手机和可穿戴设备已成为日常生活的一部分，驱动着智能家居、医疗保健和智能交通等领域智能应用的发展。然而，这些进步由于传统基于云的机器学习（ML）框架所施加的延迟和带宽限制而面临重大挑战。随着云计算面临延迟增加和网络拥塞问题，对创新解决方案的需求愈发明显。尽管以往将部分ML管道卸载到边缘和云层的努力仍未完全解决这些问题，反而常常由于边缘设备的计算限制而恶化系统响应时间和网络拥塞。为应对这些挑战，本研究引入了InTec（集成事物边缘计算）框架，这是一种物联网架构上的重大创新。不同于现有方法，InTec通过战略性地将ML任务分布在事物、边缘和云层，充分利用三层架构的潜力。这种综合方法能够在数据生成点进行实时数据处理，显著降低延迟、优化网络流量并增强系统可靠性。通过使用MHEALTH数据集对智能家庭中的人体运动检测进行实证评估，InTec的有效性得到验证，结果显示在关键指标上取得了显著改善：响应时间减少了81.56%，网络流量减少了10.92%，吞吐量提升了9.82%，边缘能耗减少了21.86%，云能耗减少了25.83%。这些进步使InTec成为可扩展、响应迅速且节能的物联网应用的新基准，展示了其潜在能力，即将ML管道集成到边缘AI（Edge AI）系统中进行革命性变革。 

---
# Neural Interpretable Reasoning 

**Title (ZH)**: 神经可解释推理 

**Authors**: Pietro Barbiero, Giuseppe Marra, Gabriele Ciravegna, David Debot, Francesco De Santis, Michelangelo Diligenti, Mateo Espinosa Zarlenga, Francesco Giannini  

**Link**: [PDF](https://arxiv.org/pdf/2502.11639)  

**Abstract**: We formalize a novel modeling framework for achieving interpretability in deep learning, anchored in the principle of inference equivariance. While the direct verification of interpretability scales exponentially with the number of variables of the system, we show that this complexity can be mitigated by treating interpretability as a Markovian property and employing neural re-parametrization techniques. Building on these insights, we propose a new modeling paradigm -- neural generation and interpretable execution -- that enables scalable verification of equivariance. This paradigm provides a general approach for designing Neural Interpretable Reasoners that are not only expressive but also transparent. 

**Abstract (ZH)**: 我们提出了一个新的建模框架，旨在通过推理协变性原则实现深度学习的可解释性。尽管直接验证可解释性随着系统变量数量的增加呈指数级增长，但我们展示了可以通过将可解释性视为马尔可夫性质并采用神经重参数化技术来缓解这种复杂性。基于这些见解，我们提出了一种新的建模范式——神经生成和可解释执行——以实现可扩展的协变验证。该范式为设计既具有表现力又具有透明性的神经可解释推理器提供了通用方法。 

---
# In-Context Parametric Inference: Point or Distribution Estimators? 

**Title (ZH)**: 上下文相关参数推断：点估计还是分布估计？ 

**Authors**: Sarthak Mittal, Yoshua Bengio, Nikolay Malkin, Guillaume Lajoie  

**Link**: [PDF](https://arxiv.org/pdf/2502.11617)  

**Abstract**: Bayesian and frequentist inference are two fundamental paradigms in statistical estimation. Bayesian methods treat hypotheses as random variables, incorporating priors and updating beliefs via Bayes' theorem, whereas frequentist methods assume fixed but unknown hypotheses, relying on estimators like maximum likelihood. While extensive research has compared these approaches, the frequentist paradigm of obtaining point estimates has become predominant in deep learning, as Bayesian inference is challenging due to the computational complexity and the approximation gap of posterior estimation methods. However, a good understanding of trade-offs between the two approaches is lacking in the regime of amortized estimators, where in-context learners are trained to estimate either point values via maximum likelihood or maximum a posteriori estimation, or full posteriors using normalizing flows, score-based diffusion samplers, or diagonal Gaussian approximations, conditioned on observations. To help resolve this, we conduct a rigorous comparative analysis spanning diverse problem settings, from linear models to shallow neural networks, with a robust evaluation framework assessing both in-distribution and out-of-distribution generalization on tractable tasks. Our experiments indicate that amortized point estimators generally outperform posterior inference, though the latter remain competitive in some low-dimensional problems, and we further discuss why this might be the case. 

**Abstract (ZH)**: 贝叶斯和频率主义推理是统计估计的两个基本范式。贝叶斯方法将假设视为随机变量，并通过贝叶斯定理更新先验和信念，而频率主义方法假定假设是固定但未知的，依赖于似然估计量如最大似然估计。尽管已有大量研究对比了这两种方法，但在模型压缩估计器的背景下，频率主义的点估计范式已成为深度学习中的主流，因为贝叶斯推理由于计算复杂性和后验估计方法的近似间隙而具有挑战性。然而，在这种背景下对这两种方法之间权衡的理解仍然不足，其中上下文学习器被训练以通过最大似然估计或最大后似然估计估计点值，或者使用归一化流、基于得分的扩散采样器或对角高斯逼近估计条件下的完整后验分布。为了解决这一问题，我们开展了跨越从线性模型到浅层神经网络的各种问题设置的严谨对比分析，并采用稳健的评估框架评估其在可处理任务中的分布内和分布外泛化能力。实验结果显示，压缩估计器中的点估计通常优于后验推理，尽管在某些低维问题中，后验推理仍具有竞争力，我们进一步讨论了这种现象的原因。 

---
# Is Human-Like Text Liked by Humans? Multilingual Human Detection and Preference Against AI 

**Title (ZH)**: 人类喜好的文本是否具有人类特点？多语言人类身份检测及对AI的偏好比较 

**Authors**: Yuxia Wang, Rui Xing, Jonibek Mansurov, Giovanni Puccetti, Zhuohan Xie, Minh Ngoc Ta, Jiahui Geng, Jinyan Su, Mervat Abassy, Saad El Dine Ahmed, Kareem Elozeiri, Nurkhan Laiyk, Maiya Goloburda, Tarek Mahmoud, Raj Vardhan Tomar, Alexander Aziz, Ryuto Koike, Masahiro Kaneko, Artem Shelmanov, Ekaterina Artemova, Vladislav Mikhailov, Akim Tsvigun, Alham Fikri Aji, Nizar Habash, Iryna Gurevych, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2502.11614)  

**Abstract**: Prior studies have shown that distinguishing text generated by large language models (LLMs) from human-written one is highly challenging, and often no better than random guessing. To verify the generalizability of this finding across languages and domains, we perform an extensive case study to identify the upper bound of human detection accuracy. Across 16 datasets covering 9 languages and 9 domains, 19 annotators achieved an average detection accuracy of 87.6%, thus challenging previous conclusions. We find that major gaps between human and machine text lie in concreteness, cultural nuances, and diversity. Prompting by explicitly explaining the distinctions in the prompts can partially bridge the gaps in over 50% of the cases. However, we also find that humans do not always prefer human-written text, particularly when they cannot clearly identify its source. 

**Abstract (ZH)**: 先前的研究表明，区分由大规模语言模型生成的文本与人类撰写的文本极具挑战性，往往与随机猜测无异。为了验证这一发现的泛化能力，我们在多种语言和领域进行了广泛的研究，以确定人类检测准确率的上限。在涵盖9种语言和9个领域的16个数据集中，19名注释者平均检测准确率为87.6%，这挑战了之前的结论。我们发现，人类与机器文本之间的主要差距在于具体性、文化细微差别和多样性。通过对提示进行明确解释，可以部分弥合超过50%的情况下的人机差距。然而，我们还发现，人类并不总是偏好人类撰写的文本，特别是在无法明确识别其来源时。 

---
# Maximum Entropy Reinforcement Learning with Diffusion Policy 

**Title (ZH)**: 最大熵强化学习与扩散策略 

**Authors**: Xiaoyi Dong, Jian Cheng, Xi Sheryl Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11612)  

**Abstract**: The Soft Actor-Critic (SAC) algorithm with a Gaussian policy has become a mainstream implementation for realizing the Maximum Entropy Reinforcement Learning (MaxEnt RL) objective, which incorporates entropy maximization to encourage exploration and enhance policy robustness. While the Gaussian policy performs well on simpler tasks, its exploration capacity and potential performance in complex multi-goal RL environments are limited by its inherent unimodality. In this paper, we employ the diffusion model, a powerful generative model capable of capturing complex multimodal distributions, as the policy representation to fulfill the MaxEnt RL objective, developing a method named MaxEnt RL with Diffusion Policy (MaxEntDP). Our method enables efficient exploration and brings the policy closer to the optimal MaxEnt policy. Experimental results on Mujoco benchmarks show that MaxEntDP outperforms the Gaussian policy and other generative models within the MaxEnt RL framework, and performs comparably to other state-of-the-art diffusion-based online RL algorithms. Our code is available at this https URL. 

**Abstract (ZH)**: 基于扩散模型的MaxEnt RL方法（MaxEntDP） 

---
# Identifying Gender Stereotypes and Biases in Automated Translation from English to Italian using Similarity Networks 

**Title (ZH)**: 使用相似网络识别从英语到意大利语自动翻译中的性别刻板印象和偏见 

**Authors**: Fatemeh Mohammadi, Marta Annamaria Tamborini, Paolo Ceravolo, Costanza Nardocci, Samira Maghool  

**Link**: [PDF](https://arxiv.org/pdf/2502.11611)  

**Abstract**: This paper is a collaborative effort between Linguistics, Law, and Computer Science to evaluate stereotypes and biases in automated translation systems. We advocate gender-neutral translation as a means to promote gender inclusion and improve the objectivity of machine translation. Our approach focuses on identifying gender bias in English-to-Italian translations. First, we define gender bias following human rights law and linguistics literature. Then we proceed by identifying gender-specific terms such as she/lei and he/lui as key elements. We then evaluate the cosine similarity between these target terms and others in the dataset to reveal the model's perception of semantic relations. Using numerical features, we effectively evaluate the intensity and direction of the bias. Our findings provide tangible insights for developing and training gender-neutral translation algorithms. 

**Abstract (ZH)**: 本论文是语言学、法律和计算机科学的跨学科合作，旨在评估自动化翻译系统中的刻板印象和偏见。我们倡导中性化翻译以促进性别包容并提高机器翻译的客观性。我们的方法侧重于识别英译意中的性别偏见。首先，我们根据人权法和语言学文献定义性别偏见。然后，我们通过识别性别特异性术语（如she/lei和he/lui）作为关键元素来进行分析。接着，我们评估这些目标术语与数据集中其他术语之间的余弦相似度，以揭示模型对语义关系的感知。通过数值特征，我们有效地评估了偏见的强度和方向。我们的发现为开发和训练中性化翻译算法提供了具体的见解。 

---
# DR.GAP: Mitigating Bias in Large Language Models using Gender-Aware Prompting with Demonstration and Reasoning 

**Title (ZH)**: DR.GAP：基于性别意识提示与示范推理的大规模语言模型偏见缓解方法 

**Authors**: Hongye Qiu, Yue Xu, Meikang Qiu, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11603)  

**Abstract**: Large Language Models (LLMs) exhibit strong natural language processing capabilities but also inherit and amplify societal biases, including gender bias, raising fairness concerns. Existing debiasing methods face significant limitations: parameter tuning requires access to model weights, prompt-based approaches often degrade model utility, and optimization-based techniques lack generalizability. To address these challenges, we propose this http URL (Demonstration and Reasoning for Gender-Aware Prompting), an automated and model-agnostic approach that mitigates gender bias while preserving model performance. this http URL selects bias-revealing examples and generates structured reasoning to guide models toward more impartial responses. Extensive experiments on coreference resolution and QA tasks across multiple LLMs (GPT-3.5, Llama3, and Llama2-Alpaca) demonstrate its effectiveness, generalization ability, and robustness. this http URL can generalize to vision-language models (VLMs), achieving significant bias reduction. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了强大的自然语言处理能力，但也继承和放大了社会偏见，包括性别偏见，引发公平性担忧。现有的去偏见方法面临显著的局限性：参数调优需要访问模型权重，基于提示的方法往往降低模型实用性，基于优化的技术缺乏普适性。为了解决这些挑战，我们提出了Demonstration and Reasoning for Gender-Aware Prompting（性别意识提示的演示与推理），这是一种自动化且模型无关的方法，能够在减轻性别偏见的同时保持模型性能。该方法通过选择揭示偏见的示例并生成结构化的推理来引导模型产生更具中立性的回复。在多个LLM（GPT-3.5、Llama3和Llama2-Alpaca）的核心参照解析和问答任务上的广泛实验展示了其有效性、普适能力和鲁棒性。该方法还可以推广到视觉语言模型（VLMs），实现显著的偏见减少。 

---
# LLM Embeddings for Deep Learning on Tabular Data 

**Title (ZH)**: 大规模语言模型嵌入在表格数据深度学习中的应用 

**Authors**: Boshko Koloski, Andrei Margeloiu, Xiangjian Jiang, Blaž Škrlj, Nikola Simidjievski, Mateja Jamnik  

**Link**: [PDF](https://arxiv.org/pdf/2502.11596)  

**Abstract**: Tabular deep-learning methods require embedding numerical and categorical input features into high-dimensional spaces before processing them. Existing methods deal with this heterogeneous nature of tabular data by employing separate type-specific encoding approaches. This limits the cross-table transfer potential and the exploitation of pre-trained knowledge. We propose a novel approach that first transforms tabular data into text, and then leverages pre-trained representations from LLMs to encode this data, resulting in a plug-and-play solution to improv ing deep-learning tabular methods. We demonstrate that our approach improves accuracy over competitive models, such as MLP, ResNet and FT-Transformer, by validating on seven classification datasets. 

**Abstract (ZH)**: 表格深度学习方法需要将数值和类别输入特征嵌入高维空间中再进行处理。现有方法通过采用类型特定的编码方法来应对表格数据的异构性质，这限制了跨表传输潜力及先验知识的利用。我们提出了一种新方法，首先将表格数据转换为文本，然后利用预训练的语言模型表示来编码这些数据，从而获得一个即插即用的解决方案，用以改进深度学习表格方法。我们通过在七个分类数据集上的验证显示，我们的方法在准确性上优于MLP、ResNet和FT-Transformer等竞争模型。 

---
# Language Complexity Measurement as a Noisy Zero-Shot Proxy for Evaluating LLM Performance 

**Title (ZH)**: 语言复杂度测量作为评估大型语言模型性能的噪声零样本代理 

**Authors**: Birger Moell, Johan Boye  

**Link**: [PDF](https://arxiv.org/pdf/2502.11578)  

**Abstract**: Large Language Models (LLMs) have made significant strides in natural language generation but often face challenges in tasks requiring precise calculations and structural analysis. This paper investigates the performance of state-of-the-art LLMs on language complexity measurement tasks, through the computation of the LIX readability metric and Average Dependency Distance (ADD). Using Swedish high school and university-level essays, we evaluate the models' abilities to compute LIX scores and perform dependency parsing, comparing their results to established ground truths. Our findings reveal that while all models demonstrate some capacity for these tasks, ChatGPT-o1-mini performs most consistently, achieving the highest accuracy in both LIX computation and dependency parsing. Additionally, we observe a strong significant correlation -0.875 p 0.026 (N=6) between the models' accuracy in computing LIX and their overall performance on the Massive Multitask Language Understanding (MMLU) benchmark. These results suggest that language complexity measurement abilities can serve as a noisy zero-shot proxies for assessing the general capabilities of LLMs, providing a practical method for model evaluation without the need for extensive benchmarking datasets. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言生成方面取得了显著进展，但在需要精确计算和结构分析的任务中常面临挑战。本文通过计算LIX可读性度量和平均依存距离（ADD），探讨了最新大型语言模型在语言复杂度测量任务中的表现。使用瑞典高中和大学水平的作文，评估模型计算LIX分数和进行依存句法分析的能力，并将其结果与既定的事实标准进行比较。研究发现，虽然所有模型在这些任务上都表现出了一定的能力，但ChatGPT-o1-mini表现最为一致，在LIX计算和依存句法分析中的准确性最高。此外，我们在计算LIX的准确性与大规模多任务语言理解（MMLU）基准上的整体表现之间观察到了显著的相关性（-0.875，p<0.026，N=6）。这些结果表明，语言复杂度测量能力可以作为评估LLMs通用能力的嘈杂零样本代理，提供了一种无需大量基准数据集即可进行模型评估的实际方法。 

---
# InfiR : Crafting Effective Small Language Models and Multimodal Small Language Models in Reasoning 

**Title (ZH)**: InfiR：打造有效的小型语言模型和推理中的多模态小型语言模型 

**Authors**: Congkai Xie, Shuo Cai, Wenjun Wang, Pengxiang Li, Zhijie Sang, Kejing Yang, Yiming Zhang, Zhen Li, Guanghao Zhu, Zeyu Liu, Yang Yu, Yuhang Liu, Su Lu, Baoyi He, Qi Zhou, Xiaotian Han, Jianbo Yuan, Shengyu Zhang, Fei Wu, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11573)  

**Abstract**: Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have made significant advancements in reasoning capabilities. However, they still face challenges such as high computational demands and privacy concerns. This paper focuses on developing efficient Small Language Models (SLMs) and Multimodal Small Language Models (MSLMs) that retain competitive reasoning abilities. We introduce a novel training pipeline that enhances reasoning capabilities and facilitates deployment on edge devices, achieving state-of-the-art performance while minimizing development costs. \InfR~ aims to advance AI systems by improving reasoning, reducing adoption barriers, and addressing privacy concerns through smaller model sizes. Resources are available at https://github. com/Reallm-Labs/InfiR. 

**Abstract (ZH)**: 大型语言模型（LLMs）和多模态大型语言模型（MLLMs）在推理能力方面取得了显著进展，但仍面临计算需求高和隐私问题等挑战。本文旨在开发高效的小型语言模型（SLMs）和多模态小型语言模型（MSLMs），同时保留竞争力的推理能力。我们提出了一种新颖的训练管道，以增强推理能力并方便在边缘设备上部署，同时实现了最先进性能并最小化开发成本。InfR~致力于通过改进推理、降低采用壁垒和通过缩小模型规模解决隐私问题来推动AI系统的进步。更多资源请访问<https://github.com/Reallm-Labs/InfiR>。 

---
# Towards Reasoning Ability of Small Language Models 

**Title (ZH)**: 面向小型语言模型的推理能力研究 

**Authors**: Gaurav Srivastava, Shuxiang Cao, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11569)  

**Abstract**: Reasoning has long been viewed as an emergent property of large language models (LLMs), appearing at or above a certain scale ($\sim$100B parameters). However, recent studies challenge this assumption, showing that small language models (SLMs) can also achieve competitive reasoning performance. SLMs are increasingly favored for their efficiency and deployability. However, there is a lack of systematic study on the reasoning abilities of diverse SLMs, including those trained from scratch or derived from LLMs through quantization, pruning, and distillation. This raises a critical question: Can SLMs achieve reasoning abilities comparable to LLMs? In this work, we systematically survey, benchmark, and analyze 72 SLMs from six model families across 14 reasoning benchmarks. For reliable evaluation, we examine four evaluation methods and compare four LLM judges against human evaluations on 800 data points. We repeat all experiments three times to ensure a robust performance assessment. Additionally, we analyze the impact of different prompting strategies in small models. Beyond accuracy, we also evaluate model robustness under adversarial conditions and intermediate reasoning steps. Our findings challenge the assumption that scaling is the only way to achieve strong reasoning. Instead, we foresee a future where SLMs with strong reasoning capabilities can be developed through structured training or post-training compression. They can serve as efficient alternatives to LLMs for reasoning-intensive tasks. 

**Abstract (ZH)**: 小语言模型的推理能力：系统调研与分析 

---
# Leader and Follower: Interactive Motion Generation under Trajectory Constraints 

**Title (ZH)**: 领导者与跟随者：基于轨迹约束的交互运动生成 

**Authors**: Runqi Wang, Caoyuan Ma, Jian Zhao, Hanrui Xu, Dongfang Sun, Haoyang Chen, Lin Xiong, Zheng Wang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.11563)  

**Abstract**: With the rapid advancement of game and film production, generating interactive motion from texts has garnered significant attention due to its potential to revolutionize content creation processes. In many practical applications, there is a need to impose strict constraints on the motion range or trajectory of virtual characters. However, existing methods that rely solely on textual input face substantial challenges in accurately capturing the user's intent, particularly in specifying the desired trajectory. As a result, the generated motions often lack plausibility and accuracy. Moreover, existing trajectory - based methods for customized motion generation rely on retraining for single - actor scenarios, which limits flexibility and adaptability to different datasets, as well as interactivity in two-actor motions. To generate interactive motion following specified trajectories, this paper decouples complex motion into a Leader - Follower dynamic, inspired by role allocation in partner dancing. Based on this framework, this paper explores the motion range refinement process in interactive motion generation and proposes a training-free approach, integrating a Pace Controller and a Kinematic Synchronization Adapter. The framework enhances the ability of existing models to generate motion that adheres to trajectory by controlling the leader's movement and correcting the follower's motion to align with the leader. Experimental results show that the proposed approach, by better leveraging trajectory information, outperforms existing methods in both realism and accuracy. 

**Abstract (ZH)**: 随着游戏和影视制作的迅速发展，从文本生成交互式运动引起了广泛关注，因其有望革命性地改变内容创作过程。在许多实际应用中，需要对虚拟角色的运动范围或轨迹施加严格约束。然而，现有仅依赖文本输入的方法在准确捕捉用户意图方面面临巨大挑战，特别是在指定期望轨迹时。因此，生成的运动往往缺乏合理性与精确性。此外，现有的基于轨迹的定制运动生成方法依赖于单人演员场景的重新训练，这限制了对不同数据集的灵活性和适应性，以及双人运动的交互性。为了根据指定的轨迹生成交互式运动，本文借鉴伴侣舞蹈中的角色分配，将复杂运动分解为领导者-跟随者动态。基于此框架，本文探索了交互式运动生成中的运动范围精炼过程，并提出了一种无需训练的方法，结合了节奏控制器和运动同步适配器。该框架通过控制领导者运动并纠正跟随者的运动以与领导者对齐，增强了现有模型生成符合轨迹运动的能力。实验结果表明，通过更好地利用轨迹信息，所提出的方法在逼真度和准确性方面优于现有方法。 

---
# Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in Large Language Models 

**Title (ZH)**: 自动搜索与精炼：大规模语言模型中性别偏见缓解的自动化框架 

**Authors**: Yue Xu, Chengyan Fu, Li Xiong, Sibei Yang, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11559)  

**Abstract**: Pre-training large language models (LLMs) on vast text corpora enhances natural language processing capabilities but risks encoding social biases, particularly gender bias. While parameter-modification methods like fine-tuning mitigate bias, they are resource-intensive, unsuitable for closed-source models, and lack adaptability to evolving societal norms. Instruction-based approaches offer flexibility but often compromise task performance. To address these limitations, we propose $\textit{FaIRMaker}$, an automated and model-independent framework that employs an $\textbf{auto-search and refinement}$ paradigm to adaptively generate Fairwords, which act as instructions integrated into input queries to reduce gender bias and enhance response quality. Extensive experiments demonstrate that $\textit{FaIRMaker}$ automatically searches for and dynamically refines Fairwords, effectively mitigating gender bias while preserving task integrity and ensuring compatibility with both API-based and open-source LLMs. 

**Abstract (ZH)**: 预训练大规模语言模型（LLMs）在 vast 文本语料库上增强自然语言处理能力但会风险编码社会偏见，尤其是性别偏见。虽然参数调整方法如微调可以缓解偏见，但这些方法资源密集、不适合闭源模型且缺乏对不断演变的社会规范的适应性。基于指令的方法提供灵活性但往往会牺牲任务性能。为解决这些限制，我们提出了一种名为 $\textit{FaIRMaker}$ 的自动化且模型独立框架，该框架采用 $\textbf{自动搜索和优化}$ 哲学自适应生成公平词，这些公平词作为指令集成到输入查询中以减少性别偏见并提高响应质量。广泛实验表明，$\textit{FaIRMaker}$ 自动搜索并动态优化公平词，有效缓解性别偏见同时保持任务完整性，并确保与基于 API 和开源的大规模语言模型兼容。 

---
# Toward Metaphor-Fluid Conversation Design for Voice User Interfaces 

**Title (ZH)**: 面向元喻流式对话设计的语音用户界面 

**Authors**: Smit Desai, Jessie Chin, Dakuo Wang, Benjamin Cowan, Michael Twidale  

**Link**: [PDF](https://arxiv.org/pdf/2502.11554)  

**Abstract**: Metaphors play a critical role in shaping user experiences with Voice User Interfaces (VUIs), yet existing designs often rely on static, human-centric metaphors that fail to adapt to diverse contexts and user needs. This paper introduces Metaphor-Fluid Design, a novel approach that dynamically adjusts metaphorical representations based on conversational use-contexts. We compare this approach to a Default VUI, which characterizes the present implementation of commercial VUIs commonly designed around the persona of an assistant, offering a uniform interaction style across contexts. In Study 1 (N=130), metaphors were mapped to four key use-contexts-commands, information seeking, sociality, and error recovery-along the dimensions of formality and hierarchy, revealing distinct preferences for task-specific metaphorical designs. Study 2 (N=91) evaluates a Metaphor-Fluid VUI against a Default VUI, showing that the Metaphor-Fluid VUI enhances perceived intention to adopt, enjoyment, and likability by aligning better with user expectations for different contexts. However, individual differences in metaphor preferences highlight the need for personalization. These findings challenge the one-size-fits-all paradigm of VUI design and demonstrate the potential of Metaphor-Fluid Design to create more adaptive and engaging human-AI interactions. 

**Abstract (ZH)**: 元喻在塑造语音用户界面（VUI）用户体验中的作用至关重要，现有设计往往依赖固定的人本中心元喻，无法适应多变的使用情境和用户需求。本文介绍了元喻流设计（Metaphor-Fluid Design）这一创新方法，该方法根据对话使用情境动态调整元喻表示。我们将其与一个默认VUI进行比较，后者描述了当前商业VUIs中常见的助手型人物设计，提供一种统一的交互风格，适用于各种情境。研究1（N=130）将元喻映射到四个关键使用情境——命令、信息查询、社交性和错误恢复——并从形式性和层级性维度揭示了对任务特定元喻设计的不同偏好。研究2（N=91）评估了元喻流设计VUI与默认VUI的性能，结果显示元喻流设计VUI通过更好地满足不同情境下用户期望，提高了使用意愿、愉悦感和喜爱度，但个体在元喻偏好的差异凸显了个性化需求。这些发现挑战了VUI设计的一刀切范式，并展示了元喻流设计在创造更具适应性和互动性的类人智能交互方面的潜力。 

---
# MuSC: Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training 

**Title (ZH)**: MuSC：多粒度自对比训练改进复杂指令跟随 

**Authors**: Hui Huang, Jiaheng Liu, Yancheng He, Shilong Li, Bing Xu, Conghui Zhu, Muyun Yang, Tiejun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.11541)  

**Abstract**: Complex instruction-following with elaborate constraints is imperative for Large Language Models (LLMs). While existing methods have constructed data for complex instruction alignment, they all rely on a more advanced model, especially GPT-4, limiting their application. In this paper, we propose a Multi-granularity Self-Contrastive Training (MuSC) framework, to improve the complex instruction alignment without relying on a stronger model. Our method is conducted on both coarse and fine granularity. On coarse-granularity, we construct constraint-aware preference data based on instruction decomposition and recombination. On fine-granularity, we perform token-aware preference optimization with dynamic token-level supervision. Our method is evaluated on open-sourced models, and experiment results show our method achieves significant improvement on both complex and general instruction-following benchmarks, surpassing previous self-alignment methods. 

**Abstract (ZH)**: 复杂指令跟随需要精细约束，这对于大型语言模型（LLMs）至关重要。现有方法虽已构建复杂指令对齐的数据，但均依赖更为先进的模型，尤其是GPT-4，限制了其应用。本文提出了一种多粒度自我对比训练（MuSC）框架，以在不依赖更强模型的情况下改善复杂指令对齐。我们的方法在粗粒度和细粒度上均进行了处理。在粗粒度上，我们基于指令分解和重组构建了约束感知偏好数据。在细粒度上，我们进行了Token感知偏好优化，并采用了动态token级别监督。我们的方法在开源模型上进行了评估，实验结果表明，该方法在复杂和通用指令跟随基准上均取得了显著改进，超越了之前的自我对齐方法。 

---
# $\text{M}^{\text{3}}$: A Modular World Model over Streams of Tokens 

**Title (ZH)**: $\text{M}^{\text{3}}$: 一种基于令牌流的模块化世界模型 

**Authors**: Lior Cohen, Kaixin Wang, Bingyi Kang, Uri Gadot, Shie Mannor  

**Link**: [PDF](https://arxiv.org/pdf/2502.11537)  

**Abstract**: Token-based world models emerged as a promising modular framework, modeling dynamics over token streams while optimizing tokenization separately. While successful in visual environments with discrete actions (e.g., Atari games), their broader applicability remains uncertain. In this paper, we introduce $\text{M}^{\text{3}}$, a $\textbf{m}$odular $\textbf{w}$orld $\textbf{m}$odel that extends this framework, enabling flexible combinations of observation and action modalities through independent modality-specific components. $\text{M}^{\text{3}}$ integrates several improvements from existing literature to enhance agent performance. Through extensive empirical evaluation across diverse benchmarks, $\text{M}^{\text{3}}$ achieves state-of-the-art sample efficiency for planning-free world models. Notably, among these methods, it is the first to reach a human-level median score on Atari 100K, with superhuman performance on 13 games. We $\href{this https URL}{\text{open-source our code and weights}}$. 

**Abstract (ZH)**: 基于Token的世界模型作为一种有前景的模块化框架涌现出来，能够分别优化token化并建模token流中的动力学过程。尽管在具有离散动作的视觉环境中（例如Atari游戏）取得了成功，但其更广泛的适用性尚不确定。本文介绍了$\text{M}^{\text{3}}$，这是一种模块化世界模型，扩展了这一框架，通过独立的模态特定组件实现观察和动作模态的灵活组合。$\text{M}^{\text{3}}$整合了现有文献中的多项改进以提升代理性能。通过在多种基准上的广泛实证评估，$\text{M}^{\text{3}}$实现了无规划世界模型的样本效率新基准。特别地，这是首个在Atari 100K上达到人类级中位分并具有超人类性能的13款游戏的方法。我们开源了我们的代码和权重。 

---
# DeFiScope: Detecting Various DeFi Price Manipulations with LLM Reasoning 

**Title (ZH)**: DeFiScope: 使用大型语言模型推理检测各类DeFi价格操控 

**Authors**: Juantao Zhong, Daoyuan Wu, Ye Liu, Maoyi Xie, Yang Liu, Yi Li, Ning Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11521)  

**Abstract**: DeFi (Decentralized Finance) is one of the most important applications of today's cryptocurrencies and smart contracts. It manages hundreds of billions in Total Value Locked (TVL) on-chain, yet it remains susceptible to common DeFi price manipulation attacks. Despite state-of-the-art (SOTA) systems like DeFiRanger and DeFort, we found that they are less effective to non-standard price models in custom DeFi protocols, which account for 44.2% of the 95 DeFi price manipulation attacks reported over the past three years.
In this paper, we introduce the first LLM-based approach, DeFiScope, for detecting DeFi price manipulation attacks in both standard and custom price models. Our insight is that large language models (LLMs) have certain intelligence to abstract price calculation from code and infer the trend of token price changes based on the extracted price models. To further strengthen LLMs in this aspect, we leverage Foundry to synthesize on-chain data and use it to fine-tune a DeFi price-specific LLM. Together with the high-level DeFi operations recovered from low-level transaction data, DeFiScope detects various DeFi price manipulations according to systematically mined patterns. Experimental results show that DeFiScope achieves a high precision of 96% and a recall rate of 80%, significantly outperforming SOTA approaches. Moreover, we evaluate DeFiScope's cost-effectiveness and demonstrate its practicality by helping our industry partner confirm 147 real-world price manipulation attacks, including discovering 81 previously unknown historical incidents. 

**Abstract (ZH)**: 去中心化金融（DeFi）智能合约的应用及其价格操纵检测：基于大语言模型的方法 

---
# UniGO: A Unified Graph Neural Network for Modeling Opinion Dynamics on Graphs 

**Title (ZH)**: UniGO：图上意见动力学建模的统一图神经网络 

**Authors**: Hao Li, Hao Jiang, Yuke Zheng, Hao Sun, Wenying Gong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11519)  

**Abstract**: Polarization and fragmentation in social media amplify user biases, making it increasingly important to understand the evolution of opinions. Opinion dynamics provide interpretability for studying opinion evolution, yet incorporating these insights into predictive models remains challenging. This challenge arises due to the inherent complexity of the diversity of opinion fusion rules and the difficulty in capturing equilibrium states while avoiding over-smoothing. This paper constructs a unified opinion dynamics model to integrate different opinion fusion rules and generates corresponding synthetic datasets. To fully leverage the advantages of unified opinion dynamics, we introduces UniGO, a framework for modeling opinion evolution on graphs. Using a coarsen-refine mechanism, UniGO efficiently models opinion dynamics through a graph neural network, mitigating over-smoothing while preserving equilibrium phenomena. UniGO leverages pretraining on synthetic datasets, which enhances its ability to generalize to real-world scenarios, providing a viable paradigm for applications of opinion dynamics. Experimental results on both synthetic and real-world datasets demonstrate UniGO's effectiveness in capturing complex opinion formation processes and predicting future evolution. The pretrained model also shows strong generalization capability, validating the benefits of using synthetic data to boost real-world performance. 

**Abstract (ZH)**: 社交媒体中的极化和碎片化放大了用户偏见，使得理解意见演变的演化变得 increasingly important。意见动力学为研究意见演变提供了可解释性，但将这些洞见整合进预测模型仍具有挑战性。这种挑战源于不同意见融合规则的固有复杂性以及捕捉平衡状态的同时避免过度平滑的困难。本文构建了一种统一的意见动力学模型，以整合不同的意见融合规则，并生成相应的合成数据集。为了充分利用统一意见动力学的优势，我们引入了UniGO，这是一种在图上建模意见演变的框架。通过粗化-细化机制，UniGO 通过图神经网络高效地建模意见动力学，减轻过度平滑现象同时保留平衡现象。UniGO 利用合成数据集上的预训练，增强了其泛化到现实场景的能力，为意见动力学的应用提供了一种可行的范式。实验结果表明，UniGO 在捕捉复杂意见形成过程和预测未来演变方面是有效的。预训练模型还展示了强大的泛化能力，验证了使用合成数据提升现实性能的好处。 

---
# Generative Multi-Agent Collaboration in Embodied AI: A Systematic Review 

**Title (ZH)**: 基于体态人工智能的生成式多Agent协作：一项系统性综述 

**Authors**: Di Wu, Xian Wei, Guang Chen, Hao Shen, Xiangfeng Wang, Wenhao Li, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2502.11518)  

**Abstract**: Embodied multi-agent systems (EMAS) have attracted growing attention for their potential to address complex, real-world challenges in areas such as logistics and robotics. Recent advances in foundation models pave the way for generative agents capable of richer communication and adaptive problem-solving. This survey provides a systematic examination of how EMAS can benefit from these generative capabilities. We propose a taxonomy that categorizes EMAS by system architectures and embodiment modalities, emphasizing how collaboration spans both physical and virtual contexts. Central building blocks, perception, planning, communication, and feedback, are then analyzed to illustrate how generative techniques bolster system robustness and flexibility. Through concrete examples, we demonstrate the transformative effects of integrating foundation models into embodied, multi-agent frameworks. Finally, we discuss challenges and future directions, underlining the significant promise of EMAS to reshape the landscape of AI-driven collaboration. 

**Abstract (ZH)**: 具身多智能体系统(EMAS)因其在物流和机器人等领域解决复杂现实挑战的潜力而受到越来越多的关注。基础模型的最新进展为生成式智能体进行更丰富的通信和适应性问题解决铺平了道路。本文综述了具身多智能体系统如何从这些生成式能力中受益。我们提出了一种分类法，按照系统架构和具身模态对EMAS进行分类，强调了协作如何跨越物理和虚拟环境。随后，分析了核心构建模块、感知、规划、通信和反馈，以说明生成技术如何增强系统的稳健性和灵活性。通过具体实例，展示了将基础模型整合到具身多智能体框架中的变革性影响。最后，我们讨论了挑战和未来方向，强调了EMAS在重塑以人工智能驱动的合作领域方面的巨大潜力。 

---
# MaZO: Masked Zeroth-Order Optimization for Multi-Task Fine-Tuning of Large Language Models 

**Title (ZH)**: MaZO: 遮掩零阶优化在大型语言模型多任务微调中的应用 

**Authors**: Zhen Zhang, Yifan Yang, Kai Zhen, Nathan Susanj, Athanasios Mouchtaris, Siegfried Kunzmann, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11513)  

**Abstract**: Large language models have demonstrated exceptional capabilities across diverse tasks, but their fine-tuning demands significant memory, posing challenges for resource-constrained environments. Zeroth-order (ZO) optimization provides a memory-efficient alternative by eliminating the need for backpropagation. However, ZO optimization suffers from high gradient variance, and prior research has largely focused on single-task learning, leaving its application to multi-task learning unexplored. Multi-task learning is crucial for leveraging shared knowledge across tasks to improve generalization, yet it introduces unique challenges under ZO settings, such as amplified gradient variance and collinearity. In this paper, we present MaZO, the first framework specifically designed for multi-task LLM fine-tuning under ZO optimization. MaZO tackles these challenges at the parameter level through two key innovations: a weight importance metric to identify critical parameters and a multi-task weight update mask to selectively update these parameters, reducing the dimensionality of the parameter space and mitigating task conflicts. Experiments demonstrate that MaZO achieves state-of-the-art performance, surpassing even multi-task learning methods designed for first-order optimization. 

**Abstract (ZH)**: 基于零阶优化的多任务大型语言模型微调框架MaZO 

---
# DifCluE: Generating Counterfactual Explanations with Diffusion Autoencoders and modal clustering 

**Title (ZH)**: DifCluE：基于扩散自编码器和模态聚类的反事实解释生成 

**Authors**: Suparshva Jain, Amit Sangroya, Lovekesh Vig  

**Link**: [PDF](https://arxiv.org/pdf/2502.11509)  

**Abstract**: Generating multiple counterfactual explanations for different modes within a class presents a significant challenge, as these modes are distinct yet converge under the same classification. Diffusion probabilistic models (DPMs) have demonstrated a strong ability to capture the underlying modes of data distributions. In this paper, we harness the power of a Diffusion Autoencoder to generate multiple distinct counterfactual explanations. By clustering in the latent space, we uncover the directions corresponding to the different modes within a class, enabling the generation of diverse and meaningful counterfactuals. We introduce a novel methodology, DifCluE, which consistently identifies these modes and produces more reliable counterfactual explanations. Our experimental results demonstrate that DifCluE outperforms the current state-of-the-art in generating multiple counterfactual explanations, offering a significant advance- ment in model interpretability. 

**Abstract (ZH)**: 基于类内不同模式的多个对抗解释生成是一项重大挑战，这些模式虽然独立但会在同一分类下收敛。扩散概率模型（DPMs）展示了捕捉数据分布潜在模式的强大能力。本文利用扩散自编码器生成多个不同的对抗解释。通过在隐空间中的聚类，我们发现了对应类内不同模式的方向，从而生成多样且有意义的对抗解释。我们提出了一个新的方法论DifCluE，该方法论一致地识别这些模式并生成更可靠的对抗解释。实验结果表明，DifCluE 在生成多个对抗解释方面优于当前最先进的方法，显著提升了模型的可解释性。 

---
# Chinese Spelling Correction: A Comprehensive Survey of Progress, Challenges, and Opportunities 

**Title (ZH)**: 中文拼写纠错：进展、挑战与机遇综述 

**Authors**: Changchun Liu, Kai Zhang, Junzhe Jiang, Zixiao Kong, Qi Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11508)  

**Abstract**: Chinese Spelling Correction (CSC) is a critical task in natural language processing, aimed at detecting and correcting spelling errors in Chinese text. This survey provides a comprehensive overview of CSC, tracing its evolution from pre-trained language models to large language models, and critically analyzing their respective strengths and weaknesses in this domain. Moreover, we further present a detailed examination of existing benchmark datasets, highlighting their inherent challenges and limitations. Finally, we propose promising future research directions, particularly focusing on leveraging the potential of LLMs and their reasoning capabilities for improved CSC performance. To the best of our knowledge, this is the first comprehensive survey dedicated to the field of CSC. We believe this work will serve as a valuable resource for researchers, fostering a deeper understanding of the field and inspiring future advancements. 

**Abstract (ZH)**: 中文拼写修正（CSC）是自然语言处理中的一个关键任务，旨在检测和修正中文文本中的拼写错误。本文综述了CSC的发展历程，从预训练语言模型到大型语言模型，并对其在该领域的各自优势和不足进行了批判性分析。此外，我们还详细分析了现有的基准数据集，突显了其固有的挑战和局限性。最后，我们提出了有前景的未来研究方向，特别强调了利用大型语言模型及其推理能力以改进CSC性能的潜力。据我们所知，这是首次专门致力于CSC领域的全面综述。我们相信，这项工作将成为研究人员的重要资源，促进对该领域的深入理解，并激发未来的进步。 

---
# Accelerated Gradient-based Design Optimization Via Differentiable Physics-Informed Neural Operator: A Composites Autoclave Processing Case Study 

**Title (ZH)**: 基于可微物理知情神经算子的加速梯度优化设计：以复合材料 autoclave 加工为例 

**Authors**: Janak M. Patel, Milad Ramezankhani, Anirudh Deodhar, Dagnachew Birru  

**Link**: [PDF](https://arxiv.org/pdf/2502.11504)  

**Abstract**: Simulation and optimization are crucial for advancing the engineering design of complex systems and processes. Traditional optimization methods require substantial computational time and effort due to their reliance on resource-intensive simulations, such as finite element analysis, and the complexity of rigorous optimization algorithms. Data-agnostic AI-based surrogate models, such as Physics-Informed Neural Operators (PINOs), offer a promising alternative to these conventional simulations, providing drastically reduced inference time, unparalleled data efficiency, and zero-shot super-resolution capability. However, the predictive accuracy of these models is often constrained to small, low-dimensional design spaces or systems with relatively simple dynamics. To address this, we introduce a novel Physics-Informed DeepONet (PIDON) architecture, which extends the capabilities of conventional neural operators to effectively model the nonlinear behavior of complex engineering systems across high-dimensional design spaces and a wide range of dynamic design configurations. This new architecture outperforms existing SOTA models, enabling better predictions across broader design spaces. Leveraging PIDON's differentiability, we integrate a gradient-based optimization approach using the Adam optimizer to efficiently determine optimal design variables. This forms an end-to-end gradient-based optimization framework that accelerates the design process while enhancing scalability and efficiency. We demonstrate the effectiveness of this framework in the optimization of aerospace-grade composites curing processes achieving a 3x speedup in obtaining optimal design variables compared to gradient-free methods. Beyond composites processing, the proposed model has the potential to be used as a scalable and efficient optimization tool for broader applications in advanced engineering and digital twin systems. 

**Abstract (ZH)**: 基于物理的深度操作网络在复杂工程系统设计优化中的仿真与优化 

---
# Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering 

**Title (ZH)**: 基于本体引导的逆向思维使大型语言模型在知识图谱问答中更加出色 

**Authors**: Runxuan Liu, Bei Luo, Jiaqi Li, Baoxin Wang, Ming Liu, Dayong Wu, Shijin Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2502.11491)  

**Abstract**: Large language models (LLMs) have shown remarkable capabilities in natural language processing. However, in knowledge graph question answering tasks (KGQA), there remains the issue of answering questions that require multi-hop reasoning. Existing methods rely on entity vector matching, but the purpose of the question is abstract and difficult to match with specific entities. As a result, it is difficult to establish reasoning paths to the purpose, which leads to information loss and redundancy. To address this issue, inspired by human reverse thinking, we propose Ontology-Guided Reverse Thinking (ORT), a novel framework that constructs reasoning paths from purposes back to conditions. ORT operates in three key phases: (1) using LLM to extract purpose labels and condition labels, (2) constructing label reasoning paths based on the KG ontology, and (3) using the label reasoning paths to guide knowledge retrieval. Experiments on the WebQSP and CWQ datasets show that ORT achieves state-of-the-art performance and significantly enhances the capability of LLMs for KGQA. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理任务中展现了 remarkable 的能力。然而，在知识图谱问答任务（KGQA）中，仍然存在需要多跳推理的问题。现有方法依赖于实体向量匹配，但问题的目的往往是抽象的，难以与特定实体匹配，导致难以建立推理路径，从而引发信息丢失和冗余。为解决这一问题，借鉴人类逆向思维，我们提出了基于本体的逆向思考（ORT）框架，该框架从目的反向构建推理路径至条件。ORT 包含三个关键阶段：（1）使用 LLM 提取目的标签和条件标签，（2）基于知识图谱本体构建标签推理路径，并（3）利用标签推理路径指导知识检索。实验结果表明，ORT 在 WebQSP 和 CWQ 数据集上达到了最先进的性能，并显著增强了 LLM 在 KGQA 中的能力。 

---
# DATA: Decomposed Attention-based Task Adaptation for Rehearsal-Free Continual Learning 

**Title (ZH)**: 数据：分解注意力机制导向的任务适应在无需重温的连续学习中 

**Authors**: Huanxuan Liao, Shizhu He, Yupu Hao, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11482)  

**Abstract**: Continual learning (CL) is essential for Large Language Models (LLMs) to adapt to evolving real-world demands, yet they are susceptible to catastrophic forgetting (CF). While traditional CF solutions rely on expensive data rehearsal, recent rehearsal-free methods employ model-based and regularization-based strategies to address this issue. However, these approaches often neglect the model's plasticity, which is crucial to achieving optimal performance on newly learned tasks. Consequently, a key challenge in CL is striking a balance between preserving plasticity and mitigating CF. To tackle this challenge, we propose the $\textbf{D}$ecomposed $\textbf{A}$ttention-based $\textbf{T}$ask $\textbf{A}$daptation (DATA), which explicitly decouples and learns both task-specific and task-shared knowledge using high-rank and low-rank task adapters (e.g., LoRAs). For new tasks, DATA dynamically adjusts the weights of adapters of different ranks based on their relevance and distinction from previous tasks, allowing the model to acquire new task-specific skills while effectively retaining previously learned knowledge. Specifically, we implement a decomposed component weighting strategy comprising learnable components that collectively generate attention-based weights, allowing the model to integrate and utilize diverse knowledge from each DATA. Extensive experiments on three widely used benchmarks demonstrate that our proposed method achieves state-of-the-art performance. Notably, our approach significantly enhances model plasticity and mitigates CF by extending learnable components and employing stochastic restoration during training iterations. 

**Abstract (ZH)**: 基于分解注意力的任务适配：平衡保留塑性与缓解灾难性遗忘 

---
# Variable-frame CNNLSTM for Breast Nodule Classification using Ultrasound Videos 

**Title (ZH)**: 基于超声视频的乳腺结节分类的变帧CNN-LSTM方法 

**Authors**: Xiangxiang Cui, Zhongyu Li, Xiayue Fan, Peng Huang, Ying Wang, Meng Yang, Shi Chang, Jihua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11481)  

**Abstract**: The intersection of medical imaging and artificial intelligence has become an important research direction in intelligent medical treatment, particularly in the analysis of medical images using deep learning for clinical diagnosis. Despite the advances, existing keyframe classification methods lack extraction of time series features, while ultrasonic video classification based on three-dimensional convolution requires uniform frame numbers across patients, resulting in poor feature extraction efficiency and model classification performance. This study proposes a novel video classification method based on CNN and LSTM, introducing NLP's long and short sentence processing scheme into video classification for the first time. The method reduces CNN-extracted image features to 1x512 dimension, followed by sorting and compressing feature vectors for LSTM training. Specifically, feature vectors are sorted by patient video frame numbers and populated with padding value 0 to form variable batches, with invalid padding values compressed before LSTM training to conserve computing resources. Experimental results demonstrate that our variable-frame CNNLSTM method outperforms other approaches across all metrics, showing improvements of 3-6% in F1 score and 1.5% in specificity compared to keyframe methods. The variable-frame CNNLSTM also achieves better accuracy and precision than equal-frame CNNLSTM. These findings validate the effectiveness of our approach in classifying variable-frame ultrasound videos and suggest potential applications in other medical imaging modalities. 

**Abstract (ZH)**: 医学成像与人工智能的交集已成为智能医疗治疗中的重要研究方向，特别是在使用深度学习进行医学图像临床诊断的图像分析中。尽管取得了进展，现有的关键帧分类方法缺乏时间序列特征的提取，而基于三维卷积的超声视频分类需要患者之间的帧数一致，导致特征提取效率低和模型分类性能差。本研究提出了一种基于CNN和LSTM的新型视频分类方法，首次将NLP中的长文本和短文本处理方案引入到视频分类中。该方法将CNN提取的图像特征维度压缩至1x512，随后对特征向量进行排序和压缩，以供LSTM训练。具体而言，特征向量按患者视频帧数排序，并填充0值以形成变长批处理，在LSTM训练前压缩无效填充值以节约计算资源。实验结果表明，我们的变帧CNNLSTM方法在所有指标上均优于其他方法，在F1分数和特异性方面分别比关键帧方法提高了3-6%和1.5%。变帧CNNLSTM在准确率和精确度上也优于等帧CNNLSTM。这些发现验证了该方法在分类变帧超声视频的有效性，并表明其在其他医学成像模态中的潜在应用。 

---
# Optimized detection of cyber-attacks on IoT networks via hybrid deep learning models 

**Title (ZH)**: 基于混合深度学习模型的物联网网络 cyber-攻击检测优化 

**Authors**: Ahmed Bensaoud, Jugal Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2502.11470)  

**Abstract**: The rapid expansion of Internet of Things (IoT) devices has increased the risk of cyber-attacks, making effective detection essential for securing IoT networks. This work introduces a novel approach combining Self-Organizing Maps (SOMs), Deep Belief Networks (DBNs), and Autoencoders to detect known and previously unseen attack patterns. A comprehensive evaluation using simulated and real-world traffic data is conducted, with models optimized via Particle Swarm Optimization (PSO). The system achieves an accuracy of up to 99.99% and Matthews Correlation Coefficient (MCC) values exceeding 99.50%. Experiments on NSL-KDD, UNSW-NB15, and CICIoT2023 confirm the model's strong performance across diverse attack types. These findings suggest that the proposed method enhances IoT security by identifying emerging threats and adapting to evolving attack strategies. 

**Abstract (ZH)**: 物联网设备的快速扩展增加了网络攻击的风险，有效的检测对于保障物联网网络的安全至关重要。本文提出了一种结合自组织映射（SOM）、深度信念网络（DBN）和自动编码器的新方法，用于检测已知和未知的攻击模式。通过使用模拟和真实世界的流量数据进行全面评估，模型通过粒子群优化（PSO）进行优化。该系统达到了99.99%的准确率和超过99.50的马修斯相关系数（MCC）值。实验结果表明，该方法在NSL-KDD、UNSW-NB15和CICIoT2023数据集上的表现强大，适用于多种攻击类型。这些发现表明，所提出的方法通过识别新兴威胁并适应 evolving攻击策略来增强物联网安全。 

---
# Towards Efficient Pre-training: Exploring FP4 Precision in Large Language Models 

**Title (ZH)**: 向量高效预训练：探索大语言模型中的FP4精度 

**Authors**: Jiecheng Zhou, Ding Tang, Rong Fu, Boni Hu, Haoran Xu, Yi Wang, Zhilin Pei, Zhongling Su, Liang Liu, Xingcheng Zhang, Weiming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11458)  

**Abstract**: The burgeoning computational demands for training large language models (LLMs) necessitate efficient methods, including quantized training, which leverages low-bit arithmetic operations to reduce costs. While FP8 precision has shown potential, leveraging FP4 remains challenging due to inherent quantization errors and limited representation capability. Based on the Transformer architecture, we present an FP4 training scheme for LLMs, overcoming these obstacles through mixed-precision quantization strategies tailed for different modules and training stages. This allows us to apply the precision level suitable to distinct components within the model, ensuring that multi-head attention and linear layers are handled appropriately. Our pretraining recipe ensures stability in backpropagation by incorporating fine-grained quantization methods with a target precision training schedule. Experimental results demonstrate that our FP4 training scheme achieves accuracy comparable to BF16 and FP8, with smaller theoretical computational cost. With the advent of next-generation hardware supporting FP4, our method sets the foundation for efficient ultra-low precision training. 

**Abstract (ZH)**: 大规模语言模型（LLMs）训练日益增长的计算需求 necessitates 有效的方法，包括量化训练，这利用低位宽算术运算以降低计算成本。虽然FP8精度已显示出潜力，但利用FP4仍面临挑战，由于固有的量化误差和有限的表示能力。基于Transformer架构，我们提出了针对LLMs的FP4训练方案，通过针对不同模块和训练阶段设计的混合精度量化策略克服了这些障碍。这种方法允许我们为模型中的不同组件选择合适的精度级别，确保多头注意力机制和线性层得到了适当处理。我们的预训练方案通过结合精细量化方法和目标精度训练计划，确保了反向传播的稳定性。实验结果表明，我们的FP4训练方案在计算成本理论更低的情况下，能达到与BF16和FP8相近的准确性。随着支持FP4的下一代硬件的到来，我们的方法为高效的超低精度训练奠定了基础。 

---
# Aligning Sentence Simplification with ESL Learner's Proficiency for Language Acquisition 

**Title (ZH)**: 根据 ESL 学习者 proficiency 调整句子简化以促进语言习得 

**Authors**: Guanlin Li, Yuki Arase, Noel Crespi  

**Link**: [PDF](https://arxiv.org/pdf/2502.11457)  

**Abstract**: Text simplification is crucial for improving accessibility and comprehension for English as a Second Language (ESL) learners. This study goes a step further and aims to facilitate ESL learners' language acquisition by simplification. Specifically, we propose simplifying complex sentences to appropriate levels for learners while also increasing vocabulary coverage of the target level in the simplifications. We achieve this without a parallel corpus by conducting reinforcement learning on a large language model. Our method employs token-level and sentence-level rewards, and iteratively trains the model on its self-generated outputs to guide the model to search for simplification hypotheses that satisfy the target attributes. Experiment results on CEFR-SP and TurkCorpus datasets show that the proposed method can effectively increase the frequency and diversity of vocabulary of the target level by more than $20\%$ compared to baseline models, while maintaining high simplification quality. 

**Abstract (ZH)**: 英语作为第二语言（ESL）学习者的文本简化对于提高可访问性和理解度至关重要。本研究旨在通过简化进一步促进ESL学习者的语言习得。具体而言，我们提议将复杂句子简化到适合学习者的适当水平，并在简化过程中增加目标水平的词汇覆盖范围。我们通过在大规模语言模型上进行强化学习，而无需平行语料库来实现这一点。该方法使用 token 级和句子级的奖励，并通过迭代训练模型以引导模型搜索满足目标属性的简化假设。在 CEFR-SP 和 TurkCorpus 数据集上的实验结果表明，所提出的方法与基线模型相比，能够有效提高目标水平词汇的频率和多样性超过20%，同时保持高简化质量。 

---
# Leveraging Labelled Data Knowledge: A Cooperative Rectification Learning Network for Semi-supervised 3D Medical Image Segmentation 

**Title (ZH)**: 利用标注数据知识：一种协作校正学习网络在半监督3D医学图像分割中的应用 

**Authors**: Yanyan Wang, Kechen Song, Yuyuan Liu, Shuai Ma, Yunhui Yan, Gustavo Carneiro  

**Link**: [PDF](https://arxiv.org/pdf/2502.11456)  

**Abstract**: Semi-supervised 3D medical image segmentation aims to achieve accurate segmentation using few labelled data and numerous unlabelled data. The main challenge in the design of semi-supervised learning methods consists in the effective use of the unlabelled data for training. A promising solution consists of ensuring consistent predictions across different views of the data, where the efficacy of this strategy depends on the accuracy of the pseudo-labels generated by the model for this consistency learning strategy. In this paper, we introduce a new methodology to produce high-quality pseudo-labels for a consistency learning strategy to address semi-supervised 3D medical image segmentation. The methodology has three important contributions. The first contribution is the Cooperative Rectification Learning Network (CRLN) that learns multiple prototypes per class to be used as external knowledge priors to adaptively rectify pseudo-labels at the voxel level. The second contribution consists of the Dynamic Interaction Module (DIM) to facilitate pairwise and cross-class interactions between prototypes and multi-resolution image features, enabling the production of accurate voxel-level clues for pseudo-label rectification. The third contribution is the Cooperative Positive Supervision (CPS), which optimises uncertain representations to align with unassertive representations of their class distributions, improving the model's accuracy in classifying uncertain regions. Extensive experiments on three public 3D medical segmentation datasets demonstrate the effectiveness and superiority of our semi-supervised learning method. 

**Abstract (ZH)**: 半监督三维医学图像分割通过少量标注数据和大量未标注数据实现准确分割：一种新的伪标签生成方法以提高一致学习策略的效果 

---
# Connector-S: A Survey of Connectors in Multi-modal Large Language Models 

**Title (ZH)**: Connector-S：多模态大型语言模型中连接器的综述 

**Authors**: Xun Zhu, Zheng Zhang, Xi Chen, Yiming Shi, Miao Li, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11453)  

**Abstract**: With the rapid advancements in multi-modal large language models (MLLMs), connectors play a pivotal role in bridging diverse modalities and enhancing model performance. However, the design and evolution of connectors have not been comprehensively analyzed, leaving gaps in understanding how these components function and hindering the development of more powerful connectors. In this survey, we systematically review the current progress of connectors in MLLMs and present a structured taxonomy that categorizes connectors into atomic operations (mapping, compression, mixture of experts) and holistic designs (multi-layer, multi-encoder, multi-modal scenarios), highlighting their technical contributions and advancements. Furthermore, we discuss several promising research frontiers and challenges, including high-resolution input, dynamic compression, guide information selection, combination strategy, and interpretability. This survey is intended to serve as a foundational reference and a clear roadmap for researchers, providing valuable insights into the design and optimization of next-generation connectors to enhance the performance and adaptability of MLLMs. 

**Abstract (ZH)**: 随着多模态大型语言模型（MLLMs）的迅速发展，连接器在桥梁不同模态和提升模型性能方面发挥着关键作用。然而，连接器的设计和演变尚未进行全面分析，这在理解这些组件的功能方面留下了空白，并阻碍了更强大连接器的发展。在本文综述中，我们系统地回顾了MLLMs中连接器的当前进展，并提出了一种结构化的分类体系，将连接器分为原子操作（映射、压缩、专家混合）和整体设计（多层、多编码器、多模态场景），强调了它们的技术贡献和进展。此外，我们讨论了几个有前景的研究前沿和挑战，包括高分辨率输入、动态压缩、引导信息选择、组合策略和可解释性。本文综述旨在作为研究人员的基础参考和清晰的路线图，提供有关设计和优化下一代连接器以增强MLLMs性能和适应性的宝贵见解。 

---
# Fishing For Cheap And Efficient Pruners At Initialization 

**Title (ZH)**: 初始化时搜索廉价且高效的剪枝器 

**Authors**: Ivo Gollini Navarrete, Nicolas Mauricio Cuadrado, Jose Renato Restom, Martin Takáč, Samuel Horváth  

**Link**: [PDF](https://arxiv.org/pdf/2502.11450)  

**Abstract**: Pruning offers a promising solution to mitigate the associated costs and environmental impact of deploying large deep neural networks (DNNs). Traditional approaches rely on computationally expensive trained models or time-consuming iterative prune-retrain cycles, undermining their utility in resource-constrained settings. To address this issue, we build upon the established principles of saliency (LeCun et al., 1989) and connection sensitivity (Lee et al., 2018) to tackle the challenging problem of one-shot pruning neural networks (NNs) before training (PBT) at initialization. We introduce Fisher-Taylor Sensitivity (FTS), a computationally cheap and efficient pruning criterion based on the empirical Fisher Information Matrix (FIM) diagonal, offering a viable alternative for integrating first- and second-order information to identify a model's structurally important parameters. Although the FIM-Hessian equivalency only holds for convergent models that maximize the likelihood, recent studies (Karakida et al., 2019) suggest that, even at initialization, the FIM captures essential geometric information of parameters in overparameterized NNs, providing the basis for our method. Finally, we demonstrate empirically that layer collapse, a critical limitation of data-dependent pruning methodologies, is easily overcome by pruning within a single training epoch after initialization. We perform experiments on ResNet18 and VGG19 with CIFAR-10 and CIFAR-100, widely used benchmarks in pruning research. Our method achieves competitive performance against state-of-the-art techniques for one-shot PBT, even under extreme sparsity conditions. Our code is made available to the public. 

**Abstract (ZH)**: 剪枝提供了一种有希望的解决方案，以减轻部署大规模深度神经网络（DNNs）相关的成本和环境影响。传统方法依赖于计算成本高昂的训练模型或耗时的迭代剪枝-重新训练循环，这在资源受限的环境中限制了其实用性。为了应对这一问题，我们基于显著性（LeCun等，1989）和连接敏感性（Lee等，2018）的原则，提出了一种初始化前超前剪枝神经网络（NNs）的解决方案，即初始化时的一次性剪枝（PBT）。我们引入了Fisher-Taylor敏感性（FTS），这是一种基于经验Fisher信息矩阵（FIM）对角线的计算成本低且高效的剪枝准则，能够利用一阶和二阶信息来识别模型的关键参数。尽管FIM与海森矩阵等价仅在收敛且最大化似然的模型中成立，但最近的研究（Karakida等，2019）表明，在初始化时，FIM能够捕获过度参数化神经网络中参数的关键几何信息，为我们的方法奠定了基础。最后，我们实验证明，数据依赖性剪枝方法中的层崩溃问题可以通过初始化后单个训练周期内的剪枝轻易解决。我们在CIFAR-10和CIFAR-100广泛使用的基准上对ResNet18和VGG19进行了实验。在极端稀疏条件下，我们的方法在一次性PBT中达到了与最先进的技术相当的性能。我们的代码已经开源。 

---
# Does Editing Provide Evidence for Localization? 

**Title (ZH)**: 编辑提供 Localization 证据吗？ 

**Authors**: Zihao Wang, Victor Veitch  

**Link**: [PDF](https://arxiv.org/pdf/2502.11447)  

**Abstract**: A basic aspiration for interpretability research in large language models is to "localize" semantically meaningful behaviors to particular components within the LLM. There are various heuristics for finding candidate locations within the LLM. Once a candidate localization is found, it can be assessed by editing the internal representations at the corresponding localization and checking whether this induces model behavior that is consistent with the semantic interpretation of the localization. The question we address here is: how strong is the evidence provided by such edits? To assess localization, we want to assess the effect of the optimal intervention at a particular location. The key new technical tool is a way of adapting LLM alignment techniques to find such optimal localized edits. With this tool in hand, we give an example where the edit-based evidence for localization appears strong, but where localization clearly fails. Indeed, we find that optimal edits at random localizations can be as effective as aligning the full model. In aggregate, our results suggest that merely observing that localized edits induce targeted changes in behavior provides little to no evidence that these locations actually encode the target behavior. 

**Abstract (ZH)**: 大型语言模型可解释性研究的基本追求是将语义上有意义的行为“本地化”到模型的特定组件中。在大型语言模型中寻找候选位置的各种启发式方法有很多种。一旦找到候选位置，可以通过编辑对应位置的内部表示，并检查这是否会导致与该位置语义解释一致的模型行为来评估这种本地化。我们在这里要回答的问题是：这种编辑提供的证据有多强？评估本地化时，我们希望评估特定位置的最佳干预措施的效果。关键的新技术工具是将大型语言模型对齐技术适应为找到这种最佳本地化编辑的方法。有了这个工具，我们给出一个例子，其中基于编辑的本地化证据似乎很强烈，但本地化显然失败了。事实上，我们发现，随机位置的最佳编辑与对整个模型进行对齐的效果相当。综上所述，我们的结果表明，仅仅观察本地化编辑引发了目标行为的变化，几乎没有证据表明这些位置实际上编码了目标行为。 

---
# Multi-Turn Multi-Modal Question Clarification for Enhanced Conversational Understanding 

**Title (ZH)**: 多轮多模态问题澄清以增强对话理解 

**Authors**: Kimia Ramezan, Alireza Amiri Bavandpour, Yifei Yuan, Clemencia Siro, Mohammad Aliannejadi  

**Link**: [PDF](https://arxiv.org/pdf/2502.11442)  

**Abstract**: Conversational query clarification enables users to refine their search queries through interactive dialogue, improving search effectiveness. Traditional approaches rely on text-based clarifying questions, which often fail to capture complex user preferences, particularly those involving visual attributes. While recent work has explored single-turn multi-modal clarification with images alongside text, such methods do not fully support the progressive nature of user intent refinement over multiple turns. Motivated by this, we introduce the Multi-turn Multi-modal Clarifying Questions (MMCQ) task, which combines text and visual modalities to refine user queries in a multi-turn conversation. To facilitate this task, we create a large-scale dataset named ClariMM comprising over 13k multi-turn interactions and 33k question-answer pairs containing multi-modal clarifying questions. We propose Mario, a retrieval framework that employs a two-phase ranking strategy: initial retrieval with BM25, followed by a multi-modal generative re-ranking model that integrates textual and visual information from conversational history. Our experiments show that multi-turn multi-modal clarification outperforms uni-modal and single-turn approaches, improving MRR by 12.88%. The gains are most significant in longer interactions, demonstrating the value of progressive refinement for complex queries. 

**Abstract (ZH)**: 基于多轮多模态澄清的会话查询澄清使用户能够在交互式对话中细化搜索查询，提高搜索效果。 

---
# An Efficient Row-Based Sparse Fine-Tuning 

**Title (ZH)**: 基于行的高效稀疏微调 

**Authors**: Cen-Jhih Li, Aditya Bhaskara  

**Link**: [PDF](https://arxiv.org/pdf/2502.11439)  

**Abstract**: Fine-tuning is an important step in adapting foundation models such as large language models to downstream tasks. To make this step more accessible to users with limited computational budgets, it is crucial to develop fine-tuning methods that are memory and computationally efficient. Sparse Fine-tuning (SFT) and Low-rank adaptation (LoRA) are two frameworks that have emerged for addressing this problem and have been adopted widely in practice. In this work, we develop a new SFT framework, based on ideas from neural network pruning. At a high level, we first identify "important" neurons/nodes using feature importance metrics from network pruning (specifically, we use the structural pruning method), and then perform fine-tuning by restricting to weights involving these neurons. Using experiments on common language tasks, we demonstrate that our method significantly improves the memory efficiency of SFT without increasing training time complexity and implementation complexity, while achieving accuracy comparable to state-of-the-art methods such as LoRA and its variants. 

**Abstract (ZH)**: 细粒度剪枝驱动的稀疏微调新框架及其应用 

---
# Learning Dexterous Bimanual Catch Skills through Adversarial-Cooperative Heterogeneous-Agent Reinforcement Learning 

**Title (ZH)**: 通过对抗-协同异构代理强化学习学习灵巧的双臂接物技能 

**Authors**: Taewoo Kim, Youngwoo Yoon, Jaehong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.11437)  

**Abstract**: Robotic catching has traditionally focused on single-handed systems, which are limited in their ability to handle larger or more complex objects. In contrast, bimanual catching offers significant potential for improved dexterity and object handling but introduces new challenges in coordination and control. In this paper, we propose a novel framework for learning dexterous bimanual catching skills using Heterogeneous-Agent Reinforcement Learning (HARL). Our approach introduces an adversarial reward scheme, where a throw agent increases the difficulty of throws-adjusting speed-while a catch agent learns to coordinate both hands to catch objects under these evolving conditions. We evaluate the framework in simulated environments using 15 different objects, demonstrating robustness and versatility in handling diverse objects. Our method achieved approximately a 2x increase in catching reward compared to single-agent baselines across 15 diverse objects. 

**Abstract (ZH)**: 基于异构代理强化学习的灵巧双臂接ONUS Classe 

---
# Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models 

**Title (ZH)**: 相对时间理解中的事实一致性提示方法 

**Authors**: Jongho Kim, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11425)  

**Abstract**: Despite the advanced capabilities of large language models (LLMs), their temporal reasoning ability remains underdeveloped. Prior works have highlighted this limitation, particularly in maintaining temporal consistency when understanding events. For example, models often confuse mutually exclusive temporal relations like ``before'' and ``after'' between events and make inconsistent predictions. In this work, we tackle the issue of temporal inconsistency in LLMs by proposing a novel counterfactual prompting approach. Our method generates counterfactual questions and enforces collective constraints, enhancing the model's consistency. We evaluate our method on multiple datasets, demonstrating significant improvements in event ordering for explicit and implicit events and temporal commonsense understanding by effectively addressing temporal inconsistencies. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）具备先进的能力，但其时间推理能力仍不够发达。以往研究强调了这一局限性，特别是在理解事件时保持时间一致性方面。例如，模型常常混淆互斥的时间关系，如“之前”和“之后”，并作出不一致的预测。在本工作中，我们通过提出一种新颖的反事实提示方法来应对LLMs的时间一致性问题。该方法生成反事实问题并施加集体约束，从而增强模型的一致性。我们在多个数据集上评估了该方法，展示了在事件排序和时间常识理解方面通过有效解决时间不一致性所取得的显著改进。 

---
# Without Paired Labeled Data: An End-to-End Self-Supervised Paradigm for UAV-View Geo-Localization 

**Title (ZH)**: 无需配对标注数据：面向UAV视角地理定位的端到端自监督范式 

**Authors**: Zhongwei Chen, Zhao-Xu Yang, Hai-Jun Rong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11381)  

**Abstract**: UAV-View Geo-Localization (UVGL) aims to ascertain the precise location of a UAV by retrieving the most similar GPS-tagged satellite image. However, existing methods predominantly rely on supervised learning paradigms that necessitate annotated paired data for training, which incurs substantial annotation costs and impedes large-scale deployment. To overcome this limitation, we propose the Dynamic Memory-Driven and Neighborhood Information Learning (DMNIL) network, a lightweight end-to-end self-supervised framework for UAV-view geo-localization. The DMNIL framework utilizes a dual-path clustering-based contrastive learning architecture as its baseline to model intra-view structural relationships, enhancing feature consistency and discriminability. Additionally, a dynamic memory-driven hierarchical learning module is proposed to progressively mine local and global information, reinforcing multi-level feature associations to improve model robustness. To bridge the domain gap between UAV and satellite views, we design an information-consistent evolutionary learning mechanism that systematically explores latent correlations within intra-view neighborhoods and across cross-view domains, ultimately constructing a unified cross-view feature representation space. Extensive experiments on three benchmarks (University-1652, SUES-200, and DenseUAV) demonstrate that DMNIL achieves competitive performance against state-of-the-art supervised methods while maintaining computational efficiency. Notably, this superiority is attained without relying on paired training data, underscoring the framework's practicality for real-world deployment. Codes will be released soon. 

**Abstract (ZH)**: 基于无人机视角的动态记忆驱动和邻域信息学习地理定位（DMNIL） 

---
# CCJA: Context-Coherent Jailbreak Attack for Aligned Large Language Models 

**Title (ZH)**: CCJA: 具有上下文一致性攻击的对齐大型语言模型 Jailbreak 攻击 

**Authors**: Guanghao Zhou, Panjia Qiu, Mingyuan Fan, Cen Chen, Mingyuan Chu, Xin Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.11379)  

**Abstract**: Despite explicit alignment efforts for large language models (LLMs), they can still be exploited to trigger unintended behaviors, a phenomenon known as "jailbreaking." Current jailbreak attack methods mainly focus on discrete prompt manipulations targeting closed-source LLMs, relying on manually crafted prompt templates and persuasion rules. However, as the capabilities of open-source LLMs improve, ensuring their safety becomes increasingly crucial. In such an environment, the accessibility of model parameters and gradient information by potential attackers exacerbates the severity of jailbreak threats. To address this research gap, we propose a novel \underline{C}ontext-\underline{C}oherent \underline{J}ailbreak \underline{A}ttack (CCJA). We define jailbreak attacks as an optimization problem within the embedding space of masked language models. Through combinatorial optimization, we effectively balance the jailbreak attack success rate with semantic coherence. Extensive evaluations show that our method not only maintains semantic consistency but also surpasses state-of-the-art baselines in attack effectiveness. Additionally, by integrating semantically coherent jailbreak prompts generated by our method into widely used black-box methodologies, we observe a notable enhancement in their success rates when targeting closed-source commercial LLMs. This highlights the security threat posed by open-source LLMs to commercial counterparts. We will open-source our code if the paper is accepted. 

**Abstract (ZH)**: 一种上下文一致的卦.break攻击（CCJA） 

---
# LLMs can Perform Multi-Dimensional Analytic Writing Assessments: A Case Study of L2 Graduate-Level Academic English Writing 

**Title (ZH)**: LLMs可以进行多维度分析写作评估：英语二外研究生水平学术写作案例研究 

**Authors**: Zhengxiang Wang, Veronika Makarova, Zhi Li, Jordan Kodner, Owen Rambow  

**Link**: [PDF](https://arxiv.org/pdf/2502.11368)  

**Abstract**: The paper explores the performance of LLMs in the context of multi-dimensional analytic writing assessments, i.e. their ability to provide both scores and comments based on multiple assessment criteria. Using a corpus of literature reviews written by L2 graduate students and assessed by human experts against 9 analytic criteria, we prompt several popular LLMs to perform the same task under various conditions. To evaluate the quality of feedback comments, we apply a novel feedback comment quality evaluation framework. This framework is interpretable, cost-efficient, scalable, and reproducible, compared to existing methods that rely on manual judgments. We find that LLMs can generate reasonably good and generally reliable multi-dimensional analytic assessments. We release our corpus for reproducibility. 

**Abstract (ZH)**: 该论文探讨了大语言模型在多维度分析写作评估中的性能，即其根据多种评估标准提供评分和评论的能力。通过使用由二外研究生撰写并由人类专家根据9个分析标准评估的文献综述语料库，我们促使几种流行的大型语言模型在不同条件下执行相同任务。为了评估反馈评论的质量，我们应用了一种新的反馈评论质量评估框架。该框架具有可解释性、成本效益、可扩展性和可复制性，优于现有的依赖人工判断的方法。我们发现，大语言模型可以生成合理良好且一般可靠的多维度分析评估。我们发布了语料库以确保可复制性。 

---
# Sparse Autoencoder Features for Classifications and Transferability 

**Title (ZH)**: 稀疏自编码特征用于分类和迁移性 

**Authors**: Jack Gallifant, Shan Chen, Kuleen Sasse, Hugo Aerts, Thomas Hartvigsen, Danielle S. Bitterman  

**Link**: [PDF](https://arxiv.org/pdf/2502.11367)  

**Abstract**: Sparse Autoencoders (SAEs) provide potentials for uncovering structured, human-interpretable representations in Large Language Models (LLMs), making them a crucial tool for transparent and controllable AI systems. We systematically analyze SAE for interpretable feature extraction from LLMs in safety-critical classification tasks. Our framework evaluates (1) model-layer selection and scaling properties, (2) SAE architectural configurations, including width and pooling strategies, and (3) the effect of binarizing continuous SAE activations. SAE-derived features achieve macro F1 > 0.8, outperforming hidden-state and BoW baselines while demonstrating cross-model transfer from Gemma 2 2B to 9B-IT models. These features generalize in a zero-shot manner to cross-lingual toxicity detection and visual classification tasks. Our analysis highlights the significant impact of pooling strategies and binarization thresholds, showing that binarization offers an efficient alternative to traditional feature selection while maintaining or improving performance. These findings establish new best practices for SAE-based interpretability and enable scalable, transparent deployment of LLMs in real-world applications. Full repo: this https URL. 

**Abstract (ZH)**: 稀疏自编码器（SAEs）为揭示大型语言模型（LLMs）中结构化的、可人为解释的表示提供了潜力，使它们成为透明和可控AI系统的关键工具。我们系统地分析了SAE在安全关键分类任务中从LLMs提取可解释特征的能力。我们的框架评估了（1）模型-层选择和缩放特性，（2）SAE架构配置，包括宽度和聚集策略，以及（3）连续SAE激活二值化的效果。从SAE派生的特征实现了宏F1 > 0.8，超过了隐藏状态和BoW基线，并且展示了从Gemma 2 2B到9B-IT模型的跨模型迁移性。这些特征以零样本方式泛化到跨语言毒性检测和视觉分类任务中。我们的分析突显了聚集策略和二值化阈值的显著影响，表明二值化提供了传统特征选择的有效替代方案，同时保持或提高了性能。这些发现确立了基于SAE的可解释性的新最佳实践，并使LLMs在实际应用中的可扩展和透明部署成为可能。完整的仓库：this https URL。 

---
# SAIF: A Sparse Autoencoder Framework for Interpreting and Steering Instruction Following of Language Models 

**Title (ZH)**: SAIF：一种用于解析和引导语言模型遵循指令的稀疏自编码器框架 

**Authors**: Zirui He, Haiyan Zhao, Yiran Qiao, Fan Yang, Ali Payani, Jing Ma, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2502.11356)  

**Abstract**: The ability of large language models (LLMs) to follow instructions is crucial for their practical applications, yet the underlying mechanisms remain poorly understood. This paper presents a novel framework that leverages sparse autoencoders (SAE) to interpret how instruction following works in these models. We demonstrate how the features we identify can effectively steer model outputs to align with given instructions. Through analysis of SAE latent activations, we identify specific latents responsible for instruction following behavior. Our findings reveal that instruction following capabilities are encoded by a distinct set of instruction-relevant SAE latents. These latents both show semantic proximity to relevant instructions and demonstrate causal effects on model behavior. Our research highlights several crucial factors for achieving effective steering performance: precise feature identification, the role of final layer, and optimal instruction positioning. Additionally, we demonstrate that our methodology scales effectively across SAEs and LLMs of varying sizes. 

**Abstract (ZH)**: 大型语言模型（LLMs）遵循指令的能力是其实际应用的关键，但其 underlying机制仍不甚明了。本文提出了一种新颖的框架，利用稀疏自编码器（SAE）来解释这些模型中指令遵循的工作机制。我们展示了我们识别的特征如何有效地引导模型输出与给定指令相一致。通过分析SAE的潜在激活，我们识别出了负责指令遵循行为的特定潜在特征。我们的研究发现，指令遵循能力由一组独特的、与相关指令相关的SAE潜在特征编码。这些潜在特征既与相关指令在语义上接近，又对模型行为表现出因果效应。我们的研究强调了实现有效引导性能的关键因素：精确特征识别、最终层的作用以及最优指令定位。此外，我们证明了我们的方法在不同大小的SAE和LLMs中都具有有效的扩展性。 

---
# "Nuclear Deployed!": Analyzing Catastrophic Risks in Decision-making of Autonomous LLM Agents 

**Title (ZH)**: “核武部署了！”：分析自主大型语言模型代理决策中的 catastrophic 风险 

**Authors**: Rongwu Xu, Xiaojian Li, Shuo Chen, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11355)  

**Abstract**: Large language models (LLMs) are evolving into autonomous decision-makers, raising concerns about catastrophic risks in high-stakes scenarios, particularly in Chemical, Biological, Radiological and Nuclear (CBRN) domains. Based on the insight that such risks can originate from trade-offs between the agent's Helpful, Harmlessness and Honest (HHH) goals, we build a novel three-stage evaluation framework, which is carefully constructed to effectively and naturally expose such risks. We conduct 14,400 agentic simulations across 12 advanced LLMs, with extensive experiments and analysis. Results reveal that LLM agents can autonomously engage in catastrophic behaviors and deception, without being deliberately induced. Furthermore, stronger reasoning abilities often increase, rather than mitigate, these risks. We also show that these agents can violate instructions and superior commands. On the whole, we empirically prove the existence of catastrophic risks in autonomous LLM agents. We will release our code upon request. 

**Abstract (ZH)**: 大规模语言模型（LLMs）正在演变为自主决策者，特别是在化学、生物、放射性和核（CBRN）领域等高风险场景中，引发了关于灾难性风险的担忧。基于这种风险可能源于代理人“助益、无害、诚实”（HHH）目标之间的权衡，我们构建了一个新颖的三阶段评估框架，精心设计以有效且自然地揭示这些风险。我们在12种先进LLM上进行了14,400次代理模拟，并进行了广泛的实验和分析。结果表明，LLM代理可以自主表现出灾难性行为和欺骗行为，而无需被特意诱导。此外，更强的推理能力往往会增加而非减轻这些风险。我们还展示了这些代理可以违背指令和上级命令。总体而言，我们实证证明了自主LLM代理存在灾难性风险。如有需要，我们将提供我们的代码。 

---
# Inverse Flow and Consistency Models 

**Title (ZH)**: 逆流和一致性模型 

**Authors**: Yuchen Zhang, Jian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.11333)  

**Abstract**: Inverse generation problems, such as denoising without ground truth observations, is a critical challenge in many scientific inquiries and real-world applications. While recent advances in generative models like diffusion models, conditional flow matching, and consistency models achieved impressive results by casting generation as denoising problems, they cannot be directly used for inverse generation without access to clean data. Here we introduce Inverse Flow (IF), a novel framework that enables using these generative models for inverse generation problems including denoising without ground truth. Inverse Flow can be flexibly applied to nearly any continuous noise distribution and allows complex dependencies. We propose two algorithms for learning Inverse Flows, Inverse Flow Matching (IFM) and Inverse Consistency Model (ICM). Notably, to derive the computationally efficient, simulation-free inverse consistency model objective, we generalized consistency training to any forward diffusion processes or conditional flows, which have applications beyond denoising. We demonstrate the effectiveness of IF on synthetic and real datasets, outperforming prior approaches while enabling noise distributions that previous methods cannot support. Finally, we showcase applications of our techniques to fluorescence microscopy and single-cell genomics data, highlighting IF's utility in scientific problems. Overall, this work expands the applications of powerful generative models to inversion generation problems. 

**Abstract (ZH)**: 逆生成问题，例如无需地面真相观测的去噪，是许多科学探究和实际应用中的关键挑战。尽管扩散模型、条件流匹配和一致性模型等生成模型的 recent 进展通过将生成问题转换为去噪问题取得了令人印象深刻的成果，但它们无法在缺乏干净数据的情况下直接用于逆生成。我们引入了逆流（Inverse Flow, IF）这一新颖框架，使这些生成模型能够用于包括无需地面真相的去噪在内的逆生成问题。逆流适用于几乎任何连续噪声分布，并允许复杂的依赖关系。我们提出了两种学习逆流的算法：逆流匹配（Inverse Flow Matching, IFM）和逆一致性模型（Inverse Consistency Model, ICM）。特别地，为推导出高效的、无需模拟的逆一致性模型目标，我们将一致性训练推广到任何前向扩散过程或条件流中，这些过程的应用远不止去噪。我们在合成数据和真实数据上展示了逆流的有效性，其性能优于先前方法，并支持之前方法无法处理的噪声分布。最后，我们展示了我们的方法在荧光显微镜和单细胞基因组学数据中的应用，突显了逆流在科学问题中的实用性。整体而言，这项工作扩展了强大生成模型在逆生成问题中的应用。 

---
# System Message Generation for User Preferences using Open-Source Models 

**Title (ZH)**: 基于开源模型的用户偏好系统消息生成 

**Authors**: Minbyul Jeong, Jungho Cho, Minsoo Khang, Dawoon Jung, Teakgyu Hong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11330)  

**Abstract**: System messages play a crucial role in interactions with large language models (LLMs), often serving as prompts to initiate conversations. Through system messages, users can assign specific roles, perform intended tasks, incorporate background information, specify various output formats and communication styles. Despite such versatility, publicly available data are often lack system messages and subject to strict license constraints in the industry field. Manual labeling of publicly available data with system messages that align with user instructions demands significant resources. In view of such challenges, our work introduces SysGen, a pipeline for generating system messages with better aligned assistant responses from the supervised fine-tuning dataset without system messages. Training on SysGen data has demonstrated substantial improvements in the alignment of model responses with system messages and user instructions, as demonstrated across various open-source models on the Multifacet benchmark, while maintaining minimal impact on other unseen benchmarks such as Open LLM Leaderboard 2. Our qualitative analysis highlights the importance of diverse system messages to ensure better adaptability across different contexts. 

**Abstract (ZH)**: 系统消息在与大型语言模型的交互中扮演着至关重要的角色，常作为启动对话的提示。通过系统消息，用户可以指定特定角色、执行预定任务、融入背景信息、指定各种输出格式和交流风格。尽管具备这种灵活性，公开数据往往缺乏系统消息，且在各个行业领域受到严格许可限制。为系统消息与用户指令相符的手动标注公开数据需要大量资源。鉴于此挑战，我们工作引入了SysGen，这是一种生成具有良好对齐的助手响应的系统消息的流水线，无需使用监督微调数据集中的系统消息。在SysGen数据上的训练展示了模型响应与系统消息和用户指令对齐程度的显著改善，这一效果在多方面开源模型的Multifacet基准测试中得到验证，同时对外部未知基准如Open LLM Leaderboard 2的最小影响保持了最小限度。我们的定性分析强调了多样化系统消息的重要性，以确保模型在不同情境下的更好适应性。 

---
# ALGEN: Few-shot Inversion Attacks on Textual Embeddings using Alignment and Generation 

**Title (ZH)**: ALGEN: 面向文本嵌入的少量样本反转攻击方法及其生成与对齐技术 

**Authors**: Yiyi Chen, Qiongkai Xu, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2502.11308)  

**Abstract**: With the growing popularity of Large Language Models (LLMs) and vector databases, private textual data is increasingly processed and stored as numerical embeddings. However, recent studies have proven that such embeddings are vulnerable to inversion attacks, where original text is reconstructed to reveal sensitive information. Previous research has largely assumed access to millions of sentences to train attack models, e.g., through data leakage or nearly unrestricted API access. With our method, a single data point is sufficient for a partially successful inversion attack. With as little as 1k data samples, performance reaches an optimum across a range of black-box encoders, without training on leaked data. We present a Few-shot Textual Embedding Inversion Attack using ALignment and GENeration (ALGEN), by aligning victim embeddings to the attack space and using a generative model to reconstruct text. We find that ALGEN attacks can be effectively transferred across domains and languages, revealing key information. We further examine a variety of defense mechanisms against ALGEN, and find that none are effective, highlighting the vulnerabilities posed by inversion attacks. By significantly lowering the cost of inversion and proving that embedding spaces can be aligned through one-step optimization, we establish a new textual embedding inversion paradigm with broader applications for embedding alignment in NLP. 

**Abstract (ZH)**: 基于对齐与生成的少量样本文本嵌入反转攻击（ALGEN）及其防御探讨 

---
# Exploiting Point-Language Models with Dual-Prompts for 3D Anomaly Detection 

**Title (ZH)**: 利用双提示点语言模型进行3D异常检测 

**Authors**: Jiaxiang Wang, Haote Xu, Xiaolu Chen, Haodi Xu, Yue Huang, Xinghao Ding, Xiaotong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11307)  

**Abstract**: Anomaly detection (AD) in 3D point clouds is crucial in a wide range of industrial applications, especially in various forms of precision manufacturing. Considering the industrial demand for reliable 3D AD, several methods have been developed. However, most of these approaches typically require training separate models for each category, which is memory-intensive and lacks flexibility. In this paper, we propose a novel Point-Language model with dual-prompts for 3D ANomaly dEtection (PLANE). The approach leverages multi-modal prompts to extend the strong generalization capabilities of pre-trained Point-Language Models (PLMs) to the domain of 3D point cloud AD, achieving impressive detection performance across multiple categories using a single model. Specifically, we propose a dual-prompt learning method, incorporating both text and point cloud prompts. The method utilizes a dynamic prompt creator module (DPCM) to produce sample-specific dynamic prompts, which are then integrated with class-specific static prompts for each modality, effectively driving the PLMs. Additionally, based on the characteristics of point cloud data, we propose a pseudo 3D anomaly generation method (Ano3D) to improve the model's detection capabilities in an unsupervised setting. Experimental results demonstrate that the proposed method, which is under the multi-class-one-model paradigm, achieves a +8.7%/+17% gain on anomaly detection and localization performance as compared to the state-of-the-art one-class-one-model methods for the Anomaly-ShapeNet dataset, and obtains +4.3%/+4.1% gain for the Real3D-AD dataset. Code will be available upon publication. 

**Abstract (ZH)**: 基于点语言模型的双提示三维异常检测（PLANE） 

---
# CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relationships? 

**Title (ZH)**: CORDIAL: 多模态大语言模型能否有效理解连贯关系？ 

**Authors**: Aashish Anantha Ramakrishnan, Aadarsh Anantha Ramakrishnan, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.11300)  

**Abstract**: Multimodal Large Language Models (MLLMs) are renowned for their superior instruction-following and reasoning capabilities across diverse problem domains. However, existing benchmarks primarily focus on assessing factual and logical correctness in downstream tasks, with limited emphasis on evaluating MLLMs' ability to interpret pragmatic cues and intermodal relationships. To address this gap, we assess the competency of MLLMs in performing Multimodal Discourse Analysis (MDA) using Coherence Relations. Our benchmark, CORDIAL, encompasses a broad spectrum of Coherence Relations across 3 different discourse domains at varying levels of granularity. Through our experiments on 10+ MLLMs employing different prompting strategies, we show that even top models like Gemini 1.5 Pro and GPT-4o fail to match the performance of simple classifier-based baselines. This study emphasizes the need to move beyond similarity-based metrics and adopt a discourse-driven framework for evaluating MLLMs, providing a more nuanced assessment of their capabilities. The benchmark and code are available at: this https URL. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）在各种问题域中以其卓越的指令跟随能力和推理能力而闻名。然而，现有的基准主要关注下游任务中的事实性和逻辑正确性评估，对评估MLLMs解析普适性线索和跨模态关系的能力关注不足。为弥补这一缺陷，我们使用连贯关系评估MLLMs在多模态话语分析（MDA）方面的能力。我们的基准Cordial涵盖三个不同话语领域的广泛连贯关系谱系，从粗粒度到细粒度不等。通过在10多个采用不同提示策略的MLLMs上进行的实验，我们显示，即使是如Gemini 1.5 Pro和GPT-4o这样的顶级模型，也无法达到基于简单分类器基线的性能。本研究强调了应超越基于相似性的指标，采用以话语为导向的框架评估MLLMs的必要性，从而提供对其能力的更细致入微的评估。基准和代码可在以下链接获取：this https URL。 

---
# Integrating Language Models for Enhanced Network State Monitoring in DRL-Based SFC Provisioning 

**Title (ZH)**: 基于DRL的SFC提供中语言模型集成以增强网络状态监控 

**Authors**: Parisa Fard Moshiri, Murat Arda Onsu, Poonam Lohan, Burak Kantarci, Emil Janulewicz  

**Link**: [PDF](https://arxiv.org/pdf/2502.11298)  

**Abstract**: Efficient Service Function Chain (SFC) provisioning and Virtual Network Function (VNF) placement are critical for enhancing network performance in modern architectures such as Software-Defined Networking (SDN) and Network Function Virtualization (NFV). While Deep Reinforcement Learning (DRL) aids decision-making in dynamic network environments, its reliance on structured inputs and predefined rules limits adaptability in unforeseen scenarios. Additionally, incorrect actions by a DRL agent may require numerous training iterations to correct, potentially reinforcing suboptimal policies and degrading performance. This paper integrates DRL with Language Models (LMs), specifically Bidirectional Encoder Representations from Transformers (BERT) and DistilBERT, to enhance network management. By feeding final VNF allocations from DRL into the LM, the system can process and respond to queries related to SFCs, DCs, and VNFs, enabling real-time insights into resource utilization, bottleneck detection, and future demand planning. The LMs are fine-tuned to our domain-specific dataset using Low-Rank Adaptation (LoRA). Results show that BERT outperforms DistilBERT with a lower test loss (0.28 compared to 0.36) and higher confidence (0.83 compared to 0.74), though BERT requires approximately 46% more processing time. 

**Abstract (ZH)**: 基于深度强化学习与语言模型的高效服务功能链配置和虚拟网络功能部署优化 

---
# FairFare: A Tool for Crowdsourcing Rideshare Data to Empower Labor Organizers 

**Title (ZH)**: FairFare: 一种用于赋能劳工组织者的数据众包工具 

**Authors**: Dana Calacci, Varun Nagaraj Rao, Samantha Dalal, Catherine Di, Kok-Wei Pua, Andrew Schwartz, Danny Spitzberg, Andrés Monroy-Hernández  

**Link**: [PDF](https://arxiv.org/pdf/2502.11273)  

**Abstract**: Rideshare workers experience unpredictable working conditions due to gig work platforms' reliance on opaque AI and algorithmic systems. In response to these challenges, we found that labor organizers want data to help them advocate for legislation to increase the transparency and accountability of these platforms. To address this need, we collaborated with a Colorado-based rideshare union to develop FairFare, a tool that crowdsources and analyzes workers' data to estimate the take rate -- the percentage of the rider price retained by the rideshare platform. We deployed FairFare with our partner organization that collaborated with us in collecting data on 76,000+ trips from 45 drivers over 18 months. During evaluation interviews, organizers reported that FairFare helped influence the bill language and passage of Colorado Senate Bill 24-75, calling for greater transparency and data disclosure of platform operations, and create a national narrative. Finally, we reflect on complexities of translating quantitative data into policy outcomes, nature of community based audits, and design implications for future transparency tools. 

**Abstract (ZH)**: 网约车工人因依赖遮蔽的AI和算法系统，而面临不可预测的工作条件。为应对这些挑战，我们发现劳工组织者希望获得数据以帮助他们倡导立法，增加平台的透明度和问责制。为应对这一需求，我们与一家科罗拉多州的网约车工会合作，开发了FairFare工具，该工具通过众筹和分析工人的数据来估算取费率——即网约车平台留存的乘客费用的百分比。我们与合作伙伴共同收集了45名司机在18个月内超过76,000次行程的数据。在评估访谈中，组织者表示，FairFare帮助影响了科罗拉多州参议院第24-75号法案的立法语言和通过，呼吁增加平台运营的透明度和数据披露，并创建了全国性的叙事。最后，我们反思将定量数据转化为政策结果的复杂性、基于社区的审计本质，以及对未来透明度工具设计的影响。 

---
# Prompting in the Dark: Assessing Human Performance in Prompt Engineering for Data Labeling When Gold Labels Are Absent 

**Title (ZH)**: 在黑暗中推动：在缺乏黄金标准标签时评估提示工程在数据标注中的人类性能 

**Authors**: Zeyu He, Saniya Naphade, Ting-Hao 'Kenneth' Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11267)  

**Abstract**: Millions of users prompt large language models (LLMs) for various tasks, but how good are people at prompt engineering? Do users actually get closer to their desired outcome over multiple iterations of their prompts? These questions are crucial when no gold-standard labels are available to measure progress. This paper investigates a scenario in LLM-powered data labeling, "prompting in the dark," where users iteratively prompt LLMs to label data without using manually-labeled benchmarks. We developed PromptingSheet, a Google Sheets add-on that enables users to compose, revise, and iteratively label data through spreadsheets. Through a study with 20 participants, we found that prompting in the dark was highly unreliable-only 9 participants improved labeling accuracy after four or more iterations. Automated prompt optimization tools like DSPy also struggled when few gold labels were available. Our findings highlight the importance of gold labels and the needs, as well as the risks, of automated support in human prompt engineering, providing insights for future tool design. 

**Abstract (ZH)**: 大规模语言模型数据标注中的“盲Prompt工程：用户迭代Prompt的效果及挑战” 

---
# Generating Skyline Datasets for Data Science Models 

**Title (ZH)**: 生成数据科学模型的 skyline 数据集 

**Authors**: Mengying Wang, Hanchao Ma, Yiyang Bian, Yangxin Fan, Yinghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11262)  

**Abstract**: Preparing high-quality datasets required by various data-driven AI and machine learning models has become a cornerstone task in data-driven analysis. Conventional data discovery methods typically integrate datasets towards a single pre-defined quality measure that may lead to bias for downstream tasks. This paper introduces MODis, a framework that discovers datasets by optimizing multiple user-defined, model-performance measures. Given a set of data sources and a model, MODis selects and integrates data sources into a skyline dataset, over which the model is expected to have the desired performance in all the performance measures. We formulate MODis as a multi-goal finite state transducer, and derive three feasible algorithms to generate skyline datasets. Our first algorithm adopts a "reduce-from-universal" strategy, that starts with a universal schema and iteratively prunes unpromising data. Our second algorithm further reduces the cost with a bi-directional strategy that interleaves data augmentation and reduction. We also introduce a diversification algorithm to mitigate the bias in skyline datasets. We experimentally verify the efficiency and effectiveness of our skyline data discovery algorithms, and showcase their applications in optimizing data science pipelines. 

**Abstract (ZH)**: 基于多目标优化的高质量数据集发现框架MODis 

---
# Shortcuts and Identifiability in Concept-based Models from a Neuro-Symbolic Lens 

**Title (ZH)**: 基于神经符号视角的概念模型中的捷径与可识别性 

**Authors**: Samuele Bortolotti, Emanuele Marconato, Paolo Morettin, Andrea Passerini, Stefano Teso  

**Link**: [PDF](https://arxiv.org/pdf/2502.11245)  

**Abstract**: Concept-based Models are neural networks that learn a concept extractor to map inputs to high-level concepts and an inference layer to translate these into predictions. Ensuring these modules produce interpretable concepts and behave reliably in out-of-distribution is crucial, yet the conditions for achieving this remain unclear. We study this problem by establishing a novel connection between Concept-based Models and reasoning shortcuts (RSs), a common issue where models achieve high accuracy by learning low-quality concepts, even when the inference layer is fixed and provided upfront. Specifically, we first extend RSs to the more complex setting of Concept-based Models and then derive theoretical conditions for identifying both the concepts and the inference layer. Our empirical results highlight the impact of reasoning shortcuts and show that existing methods, even when combined with multiple natural mitigation strategies, often fail to meet these conditions in practice. 

**Abstract (ZH)**: 基于概念的模型是神经网络，学习一个概念提取器将输入映射到高层概念，并通过推理层将这些概念转化为预测。确保这些模块生成可解释的概念并在分布外表现可靠至关重要，但实现这些条件的条件尚不明确。我们通过建立基于概念的模型与推理捷径（RS）之间的新型联系来研究这个问题，推理捷径是一个常见问题，即模型通过学习低质量的概念在推理层固定且预先提供的情况下仍能获得高准确率。具体而言，我们首先将RS扩展到基于概念的模型的更复杂设置，然后推导出识别概念和推理层的理论条件。我们的实证结果强调了推理捷径的影响，并表明现有方法，即使结合了多种自然缓解策略，也往往无法在实践中满足这些条件。 

---
# Soteria: Language-Specific Functional Parameter Steering for Multilingual Safety Alignment 

**Title (ZH)**: Soteria: 语言特定的功能参数调节以实现多语言安全对齐 

**Authors**: Somnath Banerjee, Sayan Layek, Pratyush Chatterjee, Animesh Mukherjee, Rima Hazra  

**Link**: [PDF](https://arxiv.org/pdf/2502.11244)  

**Abstract**: Ensuring consistent safety across multiple languages remains a significant challenge for large language models (LLMs). We introduce Soteria, a lightweight yet powerful strategy that locates and minimally adjusts the "functional heads" most responsible for harmful content generation in each language. By altering only a fraction of parameters, Soteria drastically reduces policy violations without sacrificing overall model performance, even in low-resource settings. To rigorously evaluate our approach, we also present XThreatBench, a specialized multilingual dataset capturing fine-grained harmful behaviors drawn from real policy guidelines. Experiments with leading open-source LLMs (e.g., Llama, Qwen, Mistral) show that Soteria consistently improves safety metrics across high-, mid-, and low-resource languages. These findings highlight a promising path toward scalable, linguistically attuned, and ethically aligned LLMs worldwide. 

**Abstract (ZH)**: 确保多种语言中的一致安全性仍然是大型语言模型（LLMs）面临的重大挑战。我们提出Soteria，这是一种轻量级但强大的策略，用于定位并在每种语言中最大程度减少最负责有害内容生成的“功能头部”。通过仅调整少量参数，Soteria大大减少了政策违规现象，同时在低资源环境中也不牺牲整体模型性能。为了严格评估我们的方法，我们还提出了XThreatBench，这是一种专门的多语言数据集，捕捉来自实际政策指南的细微有害行为。使用领先开源LLM（例如Llama、Qwen、Mistral）的实验表明，Soteria在高、中、低资源语言中一致地提高了安全性指标。这些发现为全球范围内可扩展、语言适应性强且伦理对齐的LLM指明了一条有前景的道路。 

---
# Towards identifying possible fault-tolerant advantage of quantum linear system algorithms in terms of space, time and energy 

**Title (ZH)**: 关于量子线性系统算法在空间、时间和能量方面的容错优势识别 

**Authors**: Yue Tu, Mark Dubynskyi, Mohammad Mohammadisiahroudi, Ekaterina Riashchentceva, Jinglei Cheng, Dmitry Ryashchentsev, Tamás Terlaky, Junyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11239)  

**Abstract**: Quantum computing, a prominent non-Von Neumann paradigm beyond Moore's law, can offer superpolynomial speedups for certain problems. Yet its advantages in efficiency for tasks like machine learning remain under investigation, and quantum noise complicates resource estimations and classical comparisons. We provide a detailed estimation of space, time, and energy resources for fault-tolerant superconducting devices running the Harrow-Hassidim-Lloyd (HHL) algorithm, a quantum linear system solver relevant to linear algebra and machine learning. Excluding memory and data transfer, possible quantum advantages over the classical conjugate gradient method could emerge at $N \approx 2^{33} \sim 2^{48}$ or even lower, requiring ${O}(10^5)$ physical qubits, ${O}(10^{12}\sim10^{13})$ Joules, and ${O}(10^6)$ seconds under surface code fault-tolerance with three types of magic state distillation (15-1, 116-12, 225-1). Key parameters include condition number, sparsity, and precision $\kappa, s\approx{O}(10\sim100)$, $\epsilon\sim0.01$, and physical error $10^{-5}$. Our resource estimator adjusts $N, \kappa, s, \epsilon$, providing a map of quantum-classical boundaries and revealing where a practical quantum advantage may arise. Our work quantitatively determine how advanced a fault-tolerant quantum computer should be to achieve possible, significant benefits on problems related to real-world. 

**Abstract (ZH)**: 量子计算：一种超越摩尔定律的 prominant 非冯·诺伊曼范式，对于某些问题可以提供超多项式加速。然而，其在机器学习等任务上的效率优势仍待探究，且量子噪声使得资源估算和经典比较复杂化。我们详细估算了运行 Harrow-Hassidim-Lloyd (HHL) 算法的容错超导器件的空间、时间和能量资源，HHL 算法是线性代数和机器学习中相关的量子线性系统求解器。排除内存和数据传输，与经典的共轭梯度方法相比，可能的量子优势可能在 $N \approx 2^{33} \sim 2^{48}$ 或更低出现，需要约 $10^5$ 个物理量子位，约 $10^{12} \sim 10^{13}$ 焦耳，约 $10^6$ 秒的表面码容错, 并伴有三种类型魔态蒸馏（15-1, 116-12, 225-1）。关键参数包括条件数、稀疏性和精度 $\kappa, s \approx O(10 \sim 100)$, $\epsilon \sim 0.01$ 和物理错误 $10^{-5}$。我们的资源估算器调整 $N, \kappa, s, \epsilon$，提供量子-经典边界图，并揭示有可能获得实际量子优势的位置。我们的工作定量确定了为了在与实际问题相关的问题上获得潜在的重要益处，一个容错的量子计算机需要达到的先进程度。 

---
# Vendi-RAG: Adaptively Trading-Off Diversity And Quality Significantly Improves Retrieval Augmented Generation With LLMs 

**Title (ZH)**: Vendi-RAG：适配性地在多样性和质量之间权衡显著改善基于LLM的检索增强生成 

**Authors**: Mohammad Reza Rezaei, Adji Bousso Dieng  

**Link**: [PDF](https://arxiv.org/pdf/2502.11228)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) for domain-specific question-answering (QA) tasks by leveraging external knowledge sources. However, traditional RAG systems primarily focus on relevance-based retrieval and often struggle with redundancy, especially when reasoning requires connecting information from multiple sources. This paper introduces Vendi-RAG, a framework based on an iterative process that jointly optimizes retrieval diversity and answer quality. This joint optimization leads to significantly higher accuracy for multi-hop QA tasks. Vendi-RAG leverages the Vendi Score (VS), a flexible similarity-based diversity metric, to promote semantic diversity in document retrieval. It then uses an LLM judge that evaluates candidate answers, generated after a reasoning step, and outputs a score that the retriever uses to balance relevance and diversity among the retrieved documents during each iteration. Experiments on three challenging datasets -- HotpotQA, MuSiQue, and 2WikiMultiHopQA -- demonstrate Vendi-RAG's effectiveness in multi-hop reasoning tasks. The framework achieves significant accuracy improvements over traditional single-step and multi-step RAG approaches, with accuracy increases reaching up to +4.2% on HotpotQA, +4.1% on 2WikiMultiHopQA, and +1.3% on MuSiQue compared to Adaptive-RAG, the current best baseline. The benefits of Vendi-RAG are even more pronounced as the number of retrieved documents increases. Finally, we evaluated Vendi-RAG across different LLM backbones, including GPT-3.5, GPT-4, and GPT-4o-mini, and observed consistent improvements, demonstrating that the framework's advantages are model-agnostic. 

**Abstract (ZH)**: 基于迭代过程的检索增强生成（Vendi-RAG）：联合优化检索多样性和答案质量以提高多跳问答任务的准确性 

---
# METAFOR: A Hybrid Metaheuristics Software Framework for Single-Objective Continuous Optimization Problems 

**Title (ZH)**: METAFOR：单目标连续优化问题的混合元启发式软件框架 

**Authors**: Christian Camacho-Villalón, Marco Dorigo, Thomas Stützle  

**Link**: [PDF](https://arxiv.org/pdf/2502.11225)  

**Abstract**: Hybrid metaheuristics are powerful techniques for solving difficult optimization problems that exploit the strengths of different approaches in a single implementation. For algorithm designers, however, creating hybrid metaheuristic implementations has become increasingly challenging due to the vast number of design options available in the literature and the fact that they often rely on their knowledge and intuition to come up with new algorithm designs. In this paper, we propose a modular metaheuristic software framework, called METAFOR, that can be coupled with an automatic algorithm configuration tool to automatically design hybrid metaheuristics. METAFOR is specifically designed to hybridize Particle Swarm Optimization, Differential Evolution and Covariance Matrix Adaptation-Evolution Strategy, and includes a local search module that allows their execution to be interleaved with a subordinate local search. We use the configuration tool irace to automatically generate 17 different metaheuristic implementations and evaluate their performance on a diverse set of continuous optimization problems. Our results show that, across all the considered problem classes, automatically generated hybrid implementations are able to outperform configured single-approach implementations, while these latter offer advantages on specific classes of functions. We provide useful insights on the type of hybridization that works best for specific problem classes, the algorithm components that contribute to the performance of the algorithms, and the advantages and disadvantages of two well-known instance separation strategies, creating stratified training set using a fix percentage and leave-one-class-out cross-validation. 

**Abstract (ZH)**: 混合元启发式方法是解决困难优化问题的强效技术，能够在单一实现中利用不同方法的优势。然而，对于算法设计师来说，由于文献中可用的配置选项众多，且通常需要依赖其知识和直觉来设计新的算法，因此创建混合元启发式实现变得越来越具挑战性。在本文中，我们提出了一种模块化元启发式软件框架METAFOR，该框架可以与自动算法配置工具结合使用，以自动设计混合元启发式方法。METAFOR专门设计用于混合粒子群优化、差分进化和共(variance matrix adaptation-evolution strategy)演化策略，并包含一个局部搜索模块，允许这些方法的执行与次级局部搜索交错进行。我们使用配置工具irace自动生成17种不同的元启发式实现，并在各种连续优化问题上评估其性能。我们的结果显示，对于所有考虑的问题类别，自动生成的混合实现能够优于配置的单方法实现，而后者在特定函数类别上具有优势。我们提供了关于哪种混合方式最适合特定问题类别、哪些算法组件对算法性能有贡献以及两种知名实例分离策略的优势和劣势的一些有用见解，使用固定百分比和留一类别验证交叉验证构建分层训练集。 

---
# Stochastic Optimization of Inventory at Large-scale Supply Chains 

**Title (ZH)**: 大规模供应链中的库存随机优化 

**Authors**: Zhaoyang Larry Jin, Mehdi Maasoumy, Yimin Liu, Zeshi Zheng, Zizhuo Ren  

**Link**: [PDF](https://arxiv.org/pdf/2502.11213)  

**Abstract**: Today's global supply chains face growing challenges due to rapidly changing market conditions, increased network complexity and inter-dependency, and dynamic uncertainties in supply, demand, and other factors. To combat these challenges, organizations employ Material Requirements Planning (MRP) software solutions to set inventory stock buffers - for raw materials, work-in-process goods, and finished products - to help them meet customer service levels. However, holding excess inventory further complicates operations and can lock up millions of dollars of capital that could be otherwise deployed. Furthermore, most commercially available MRP solutions fall short in considering uncertainties and do not result in optimal solutions for modern enterprises.
At C3 AI, we fundamentally reformulate the inventory management problem as a constrained stochastic optimization. We then propose a simulation-optimization framework that minimizes inventory and related costs while maintaining desired service levels. The framework's goal is to find the optimal reorder parameters that minimize costs subject to a pre-defined service-level constraint and all other real-world operational constraints. These optimal reorder parameters can be fed back into an MRP system to drive optimal order placement, or used to place optimal orders directly. This approach has proven successful in reducing inventory levels by 10-35 percent, resulting in hundreds of millions of dollars of economic benefit for major enterprises at a global scale. 

**Abstract (ZH)**: 今天全球供应链面临的挑战日益加剧，由于市场条件迅速变化、网络复杂性和相互依赖性增加，以及供应、需求和其他因素的动态不确定性。为应对这些挑战，组织采用物料需求计划（MRP）软件解决方案来设置原材料、在制品和最终产品的库存缓冲，以帮助满足客户服务水平。然而，持有超额库存进一步复杂化了运营，并可能占用数百万美元的资本，否则可用于其他投资。此外，大多数商用MRP解决方案在考虑不确定性方面存在不足，无法为现代企业找到最优解决方案。
在C3 AI，我们将库存管理问题根本上重新表述为受限随机优化问题。我们还提出了一种仿真-优化框架，旨在在维持所需服务水平的同时最小化库存及相关成本。该框架的目标是找到在满足预定义的服务水平约束和其他所有实际运营约束条件的情况下，能够最小化成本的最佳 reorder 参数。这些最优 reorder 参数可以反馈到MRP系统中以驱动最优订单的下达，或直接用于下达最优订单。这种方法已在全球范围内成功地减少了10-35%的库存水平，为企业带来了数亿美元的经济效益。 

---
# A Survey of LLM-based Agents in Medicine: How far are we from Baymax? 

**Title (ZH)**: 基于LLM的医疗代理综述：我们距Baymax还有多远？ 

**Authors**: Wenxuan Wang, Zizhan Ma, Zheng Wang, Chenghan Wu, Wenting Chen, Xiang Li, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.11211)  

**Abstract**: Large Language Models (LLMs) are transforming healthcare through the development of LLM-based agents that can understand, reason about, and assist with medical tasks. This survey provides a comprehensive review of LLM-based agents in medicine, examining their architectures, applications, and challenges. We analyze the key components of medical agent systems, including system profiles, clinical planning mechanisms, medical reasoning frameworks, and external capacity enhancement. The survey covers major application scenarios such as clinical decision support, medical documentation, training simulations, and healthcare service optimization. We discuss evaluation frameworks and metrics used to assess these agents' performance in healthcare settings. While LLM-based agents show promise in enhancing healthcare delivery, several challenges remain, including hallucination management, multimodal integration, implementation barriers, and ethical considerations. The survey concludes by highlighting future research directions, including advances in medical reasoning inspired by recent developments in LLM architectures, integration with physical systems, and improvements in training simulations. This work provides researchers and practitioners with a structured overview of the current state and future prospects of LLM-based agents in medicine. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过基于LLM的代理来理解、推理和协助医疗任务，正在重塑医疗保健领域。本文综述了基于LLM的医疗代理，对其架构、应用和挑战进行了全面审查。我们分析了医疗代理系统的关键组件，包括系统特性、临床规划机制、医学推理框架和外部能力增强。综述涵盖了临床决策支持、医疗文档、培训模拟和医疗服务优化等主要应用场景。我们讨论了用于评估这些代理在医疗保健环境中的性能的评估框架和指标。虽然基于LLM的代理在提高医疗服务方面表现出潜力，但仍存在若干挑战，包括幻觉管理、多模态集成、实施障碍和伦理考量。本文总结了未来研究的方向，包括受到LLM架构最新发展启发的医学推理进展、与物理系统集成以及培训模拟的改进。本文为研究人员和从业者提供了基于LLM的代理在医疗领域的当前状态和未来前景的结构化概述。 

---
# Bridging the Gap: Enabling Natural Language Queries for NoSQL Databases through Text-to-NoSQL Translation 

**Title (ZH)**: 填补空白：通过文本到NoSQL翻译使自然语言查询成为可能 

**Authors**: Jinwei Lu, Yuanfeng Song, Zhiqian Qin, Haodi Zhang, Chen Zhang, Raymond Chi-Wing Wong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11201)  

**Abstract**: NoSQL databases have become increasingly popular due to their outstanding performance in handling large-scale, unstructured, and semi-structured data, highlighting the need for user-friendly interfaces to bridge the gap between non-technical users and complex database queries. In this paper, we introduce the Text-to-NoSQL task, which aims to convert natural language queries into NoSQL queries, thereby lowering the technical barrier for non-expert users. To promote research in this area, we developed a novel automated dataset construction process and released a large-scale and open-source dataset for this task, named TEND (short for Text-to-NoSQL Dataset). Additionally, we designed a SLM (Small Language Model)-assisted and RAG (Retrieval-augmented Generation)-assisted multi-step framework called SMART, which is specifically designed for Text-to-NoSQL conversion. To ensure comprehensive evaluation of the models, we also introduced a detailed set of metrics that assess the model's performance from both the query itself and its execution results. Our experimental results demonstrate the effectiveness of our approach and establish a benchmark for future research in this emerging field. We believe that our contributions will pave the way for more accessible and intuitive interactions with NoSQL databases. 

**Abstract (ZH)**: NoSQL数据库由于在处理大规模、非结构化和半结构化数据方面的出色性能而日益流行，这突显了为非技术用户提供友好接口以弥合非技术用户与复杂数据库查询之间的差距的必要性。本文介绍了一项Text-to-NoSQL任务，旨在将自然语言查询转换为NoSQL查询，从而降低非专家用户的技术门槛。为促进该领域的研究，我们开发了一种新的自动化数据集构建过程，并发布了该任务的大规模开源数据集TEND（Text-to-NoSQL Dataset）。此外，我们还设计了一种名为SMART（SLM辅助和RAG辅助多步框架）的框架，专门用于Text-to-NoSQL转换。为了确保对模型进行全面评估，我们还引入了一套详细的评估指标，从查询本身及其执行结果两个方面评估模型性能。实验证明了我们方法的有效性，并为该新兴领域的未来研究建立了基准。我们相信，我们的贡献将为NoSQL数据库的更易用和直观交互铺平道路。 

---
# How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training 

**Title (ZH)**: LLMs如何获取新知识？一种持续预训练的知识电路视角 

**Authors**: Yixin Ou, Yunzhi Yao, Ningyu Zhang, Hui Jin, Jiacheng Sun, Shumin Deng, Zhenguo Li, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11196)  

**Abstract**: Despite exceptional capabilities in knowledge-intensive tasks, Large Language Models (LLMs) face a critical gap in understanding how they internalize new knowledge, particularly how to structurally embed acquired knowledge in their neural computations. We address this issue through the lens of knowledge circuit evolution, identifying computational subgraphs that facilitate knowledge storage and processing. Our systematic analysis of circuit evolution throughout continual pre-training reveals several key findings: (1) the acquisition of new knowledge is influenced by its relevance to pre-existing knowledge; (2) the evolution of knowledge circuits exhibits a distinct phase shift from formation to optimization; (3) the evolution of knowledge circuits follows a deep-to-shallow pattern. These insights not only advance our theoretical understanding of the mechanisms of new knowledge acquisition in LLMs, but also provide potential implications for improving continual pre-training strategies to enhance model performance. Code and data will be available at this https URL. 

**Abstract (ZH)**: 尽管大型语言模型在知识密集型任务中表现出色，但在理解它们如何内化新知识方面仍存在关键差距，特别是在如何在神经计算中结构化嵌入获得的知识方面。我们通过知识电路进化的视角来应对这一问题，识别出促进知识存储和处理的计算子图。我们对持续预训练过程中电路进化的系统分析揭示了几项关键发现：(1) 新知识的获取受先前知识的相关性影响；(2) 知识电路的进化表现出从形成到优化的阶段转变；(3) 知识电路的进化遵循从深到浅的模式。这些见解不仅推进了我们对大型语言模型中新知识获取机制的理论理解，还为改进持续预训练策略以提升模型性能提供了潜在影响。相关代码和数据将在以下网址获取：[此链接处]。 

---
# From Deception to Perception: The Surprising Benefits of Deepfakes for Detecting, Measuring, and Mitigating Bias 

**Title (ZH)**: 从欺骗到认知：深层伪造在检测、度量和减轻偏见方面的意外益处 

**Authors**: Yizhi Liu, Balaji Padmanabhan, Siva Viswanathan  

**Link**: [PDF](https://arxiv.org/pdf/2502.11195)  

**Abstract**: While deepfake technologies have predominantly been criticized for potential misuse, our study demonstrates their significant potential as tools for detecting, measuring, and mitigating biases in key societal domains. By employing deepfake technology to generate controlled facial images, we extend the scope of traditional correspondence studies beyond mere textual manipulations. This enhancement is crucial in scenarios such as pain assessments, where subjective biases triggered by sensitive features in facial images can profoundly affect outcomes. Our results reveal that deepfakes not only maintain the effectiveness of correspondence studies but also introduce groundbreaking advancements in bias measurement and correction techniques. This study emphasizes the constructive role of deepfake technologies as essential tools for advancing societal equity and fairness. 

**Abstract (ZH)**: 尽管深度假讯技术主要受到了潜在滥用的批评，但我们的研究展示了它们在检测、衡量和减轻关键社会领域偏见方面的巨大潜力。通过使用深度假讯技术生成受控面部图像，我们扩展了传统对应研究的范围，超越了仅仅对文本的操纵。这种扩展在疼痛评估等场景中至关重要，因为面部图像中的敏感特征可能引发的主观偏见会深刻影响结果。研究结果表明，深度假讯不仅保持了对应研究的有效性，还引入了偏见测量和纠正技术的重大突破。本研究强调了深度假讯技术在促进社会公平和平等方面的重要建设性作用。 

---
# Primus: A Pioneering Collection of Open-Source Datasets for Cybersecurity LLM Training 

**Title (ZH)**: Primus: 首个开源数据集集合，用于网络安全LLM训练 

**Authors**: Yao-Ching Yu, Tsun-Han Chiang, Cheng-Wei Tsai, Chien-Ming Huang, Wen-Kwang Tsao  

**Link**: [PDF](https://arxiv.org/pdf/2502.11191)  

**Abstract**: Large Language Models (LLMs) have shown remarkable advancements in specialized fields such as finance, law, and medicine. However, in cybersecurity, we have noticed a lack of open-source datasets, with a particular lack of high-quality cybersecurity pretraining corpora, even though much research indicates that LLMs acquire their knowledge during pretraining. To address this, we present a comprehensive suite of datasets covering all major training stages, including pretraining, instruction fine-tuning, and reasoning distillation with cybersecurity-specific self-reflection data. Extensive ablation studies demonstrate their effectiveness on public cybersecurity benchmarks. In particular, continual pre-training on our dataset yields a 15.88% improvement in the aggregate score, while reasoning distillation leads to a 10% gain in security certification (CISSP). We will release all datasets and trained cybersecurity LLMs under the ODC-BY and MIT licenses to encourage further research in the community. For access to all datasets and model weights, please refer to this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在金融、法律和医学等专业领域展现了显著的进步。然而，在网络信息安全领域，我们注意到缺乏开源数据集，尤其是在高质网络信息安全预训练语料方面存在明显不足，尽管许多研究指出LLMs在其预训练阶段就已获取知识。为解决这一问题，我们提供了一套全面的数据集，涵盖了所有主要的训练阶段，包括预训练、指令微调和基于网络安全特定自反数据的推理精炼。广泛的消融研究证明了这些数据集在公开的网络安全基准测试中的有效性。特别是，持续使用我们的数据集进行预训练可提高综合得分15.88%，而推理精炼可使安全认证（CISSP）成绩提高10%。我们将所有数据集和训练好的网络信息安全大型语言模型在ODC-BY和MIT许可证下发布，以促进社区内的进一步研究。欲获取所有数据集和模型权重，请访问此链接：https URL。 

---
# ReLearn: Unlearning via Learning for Large Language Models 

**Title (ZH)**: ReLearn: 通过学习实现大规模语言模型的遗忘 

**Authors**: Haoming Xu, Ningyuan Zhao, Liming Yang, Sendong Zhao, Shumin Deng, Mengru Wang, Bryan Hooi, Nay Oo, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11190)  

**Abstract**: Current unlearning methods for large language models usually rely on reverse optimization to reduce target token probabilities. However, this paradigm disrupts the subsequent tokens prediction, degrading model performance and linguistic coherence. Moreover, existing evaluation metrics overemphasize contextual forgetting while inadequately assessing response fluency and relevance. To address these challenges, we propose ReLearn, a data augmentation and fine-tuning pipeline for effective unlearning, along with a comprehensive evaluation framework. This framework introduces Knowledge Forgetting Rate (KFR) and Knowledge Retention Rate (KRR) to measure knowledge-level preservation, and Linguistic Score (LS) to evaluate generation quality. Our experiments show that ReLearn successfully achieves targeted forgetting while preserving high-quality output. Through mechanistic analysis, we further demonstrate how reverse optimization disrupts coherent text generation, while ReLearn preserves this essential capability. Code is available at this https URL. 

**Abstract (ZH)**: 当前的大语言模型去学习方法通常依赖于反向优化来降低目标标记的概率。然而，这种范式会破坏后续标记的预测，降低模型性能和语言连贯性。此外，现有的评估指标过度强调上下文遗忘，而未能充分评估响应的流畅性和相关性。为了应对这些挑战，我们提出 ReLearn，一种有效去学习的数据增强和微调管道，以及一个全面的评估框架。该框架引入了知识遗忘率（KFR）和知识保留率（KRR）来衡量知识层面的保留，并引入语言得分（LS）来评估生成质量。我们的实验表明，ReLearn 成功实现了目标遗忘，同时保持了高质量的输出。通过机制分析，我们进一步证明了反向优化如何破坏连贯文本生成，而 ReLearn 保留了这一重要能力。代码可在以下网址获取：这个 https URL。 

---
# TituLLMs: A Family of Bangla LLMs with Comprehensive Benchmarking 

**Title (ZH)**: TituLLMs: 一个全面基准测试的孟加拉语大型语言模型家族 

**Authors**: Shahriar Kabir Nahin, Rabindra Nath Nandi, Sagor Sarker, Quazi Sarwar Muhtaseem, Md Kowsher, Apu Chandraw Shill, Md Ibrahim, Mehadi Hasan Menon, Tareq Al Muntasir, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2502.11187)  

**Abstract**: In this paper, we present TituLLMs, the first large pretrained Bangla LLMs, available in 1B and 3B parameter sizes. Due to computational constraints during both training and inference, we focused on smaller models. To train TituLLMs, we collected a pretraining dataset of approximately 37 billion tokens. We extended the Llama-3.2 tokenizer to incorporate language- and culture-specific knowledge, which also enables faster training and inference. There was a lack of benchmarking datasets to evaluate LLMs for Bangla. To address this gap, we developed five benchmarking datasets. We benchmarked various LLMs, including TituLLMs, and demonstrated that TituLLMs outperforms its initial multilingual versions. However, this is not always the case, highlighting the complexities of language adaptation. Our work lays the groundwork for adapting existing multilingual open models to other low-resource languages. To facilitate broader adoption and further research, we have made the TituLLMs models and benchmarking datasets publicly available (this https URL). 

**Abstract (ZH)**: 本文介绍了TituLLMs，这是第一个可用的大型预训练孟加拉语语言模型，参数规模为1B和3B。由于在训练和推断过程中受到计算限制，我们专注于较小的模型。为了训练TituLLMs，我们收集了一个约370亿个令牌的预训练数据集。我们将Llama-3.2分词器扩展为包含语言和文化特定的知识，这还使得训练和推断速度更快。对于孟加拉语语言模型缺乏基准数据集的问题，我们开发了五个基准数据集。我们对包括TituLLMs在内的多种语言模型进行了基准测试，并证明了TituLLMs优于其最初的多语言版本。然而，这并非总是如此，这突显了语言适应的复杂性。我们的工作为基础多资源语言适配现有的多语言开放模型奠定了基础。为了促进更广泛的采用和进一步的研究，我们已将TituLLMs模型和基准数据集公开（this https URL）。 

---
# Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs 

**Title (ZH)**: 难以见森林而见树木：多模态安全意识基准测试 for 多模态大语言模型 

**Authors**: Wenxuan Wang, Xiaoyuan Liu, Kuiyi Gao, Jen-tse Huang, Youliang Yuan, Pinjia He, Shuai Wang, Zhaopeng Tu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11184)  

**Abstract**: Multimodal Large Language Models (MLLMs) have expanded the capabilities of traditional language models by enabling interaction through both text and images. However, ensuring the safety of these models remains a significant challenge, particularly in accurately identifying whether multimodal content is safe or unsafe-a capability we term safety awareness. In this paper, we introduce MMSafeAware, the first comprehensive multimodal safety awareness benchmark designed to evaluate MLLMs across 29 safety scenarios with 1500 carefully curated image-prompt pairs. MMSafeAware includes both unsafe and over-safety subsets to assess models abilities to correctly identify unsafe content and avoid over-sensitivity that can hinder helpfulness. Evaluating nine widely used MLLMs using MMSafeAware reveals that current models are not sufficiently safe and often overly sensitive; for example, GPT-4V misclassifies 36.1% of unsafe inputs as safe and 59.9% of benign inputs as unsafe. We further explore three methods to improve safety awareness-prompting-based approaches, visual contrastive decoding, and vision-centric reasoning fine-tuning-but find that none achieve satisfactory performance. Our findings highlight the profound challenges in developing MLLMs with robust safety awareness, underscoring the need for further research in this area. All the code and data will be publicly available to facilitate future research. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）通过使交互能够通过文本和图像进行，扩大了传统语言模型的能力。然而，确保这些模型的安全性仍然是一个重大挑战，特别是准确识别多模态内容是否安全的能力——我们称之为安全性意识。在本文中，我们介绍了MMSafeAware，这是首个用于评估MLLMs在涵盖29种安全场景的1500个精挑细选的图像-提示对中的综合多模态安全性意识基准。MMSafeAware包括不安全和过度安全的子集，以评估模型正确识别不安全内容并避免过度敏感的能力，后者可能会妨碍其帮助性。使用MMSafeAware评估九个广泛使用的MLLMs发现，当前模型在安全性上并不充分且通常过于敏感；例如，GPT-4V错误地将36.1%的不安全输入分类为安全输入，并错误地将59.9%的良性输入分类为不安全输入。我们进一步探索了三种提高安全性意识的方法——基于提示的方法、视觉对比解码以及以视觉为中心的推理微调，但发现这三种方法均未能达到令人满意的效果。我们的研究结果突显了在开发具有稳健安全性意识的MLLMs方面面临的巨大挑战，强调了需要在这一领域进行进一步研究的重要性。所有代码和数据将公开以促进未来的研究。 

---
# Improving Scientific Document Retrieval with Concept Coverage-based Query Set Generation 

**Title (ZH)**: 基于概念覆盖的查询集生成以改善科学研究文献检索 

**Authors**: SeongKu Kang, Bowen Jin, Wonbin Kweon, Yu Zhang, Dongha Lee, Jiawei Han, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11181)  

**Abstract**: In specialized fields like the scientific domain, constructing large-scale human-annotated datasets poses a significant challenge due to the need for domain expertise. Recent methods have employed large language models to generate synthetic queries, which serve as proxies for actual user queries. However, they lack control over the content generated, often resulting in incomplete coverage of academic concepts in documents. We introduce Concept Coverage-based Query set Generation (CCQGen) framework, designed to generate a set of queries with comprehensive coverage of the document's concepts. A key distinction of CCQGen is that it adaptively adjusts the generation process based on the previously generated queries. We identify concepts not sufficiently covered by previous queries, and leverage them as conditions for subsequent query generation. This approach guides each new query to complement the previous ones, aiding in a thorough understanding of the document. Extensive experiments demonstrate that CCQGen significantly enhances query quality and retrieval performance. 

**Abstract (ZH)**: 概念覆盖导向的查询集生成框架（CCQGen） 

---
# RT-DEMT: A hybrid real-time acupoint detection model combining mamba and transformer 

**Title (ZH)**: RT-DEMT：结合Mamba和变压器的混合实时腧穴检测模型 

**Authors**: Shilong Yang, Qi Zang, Chulong Zhang, Lingfeng Huang, Yaoqin Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.11179)  

**Abstract**: Traditional Chinese acupuncture methods often face controversy in clinical practice due to their high subjectivity. Additionally, current intelligent-assisted acupuncture systems have two major limitations: slow acupoint localization speed and low accuracy. To address these limitations, a new method leverages the excellent inference efficiency of the state-space model Mamba, while retaining the advantages of the attention mechanism in the traditional DETR architecture, to achieve efficient global information integration and provide high-quality feature information for acupoint localization tasks. Furthermore, by employing the concept of residual likelihood estimation, it eliminates the need for complex upsampling processes, thereby accelerating the acupoint localization task. Our method achieved state-of-the-art (SOTA) accuracy on a private dataset of acupoints on the human back, with an average Euclidean distance pixel error (EPE) of 7.792 and an average time consumption of 10.05 milliseconds per localization task. Compared to the second-best algorithm, our method improved both accuracy and speed by approximately 14\%. This significant advancement not only enhances the efficacy of acupuncture treatment but also demonstrates the commercial potential of automated acupuncture robot systems. Access to our method is available at this https URL 

**Abstract (ZH)**: 传统中医针灸方法在临床实践中常因高度主观性而面临争议。目前的智能辅助针灸系统存在两大局限性：针灸穴位定位速度慢和准确性低。为解决这些问题，该研究利用状态空间模型Mamba的出色推断效率，同时保留传统DETR架构中的注意力机制优势，实现了高效的全局信息整合，并为针灸穴位定位任务提供了高质量的特征信息。此外，通过应用残差似然估计的概念，该方法消除了复杂的上采样过程的需求，从而加速了针灸穴位定位任务。在针对人体背部穴位的私有数据集上，该方法达到了最先进的（SOTA）精度，平均欧氏距离像素误差（EPE）为7.792，每项定位任务的平均时间消耗为10.05毫秒。与第二优算法相比，该方法在精度和速度上分别提升了约14%。这一显著进步不仅提高了针灸治疗的有效性，还展示了自动化针灸机器人系统的商业潜力。有关该方法的访问链接为：this https URL。 

---
# Knowing Your Target: Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding 

**Title (ZH)**: 了解目标：目标感知变换器促进更好的时空视频定位 

**Authors**: Xin Gu, Yaojie Shen, Chenxi Luo, Tiejian Luo, Yan Huang, Yuewei Lin, Heng Fan, Libo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11168)  

**Abstract**: Transformer has attracted increasing interest in STVG, owing to its end-to-end pipeline and promising result. Existing Transformer-based STVG approaches often leverage a set of object queries, which are initialized simply using zeros and then gradually learn target position information via iterative interactions with multimodal features, for spatial and temporal localization. Despite simplicity, these zero object queries, due to lacking target-specific cues, are hard to learn discriminative target information from interactions with multimodal features in complicated scenarios (\e.g., with distractors or occlusion), resulting in degradation. Addressing this, we introduce a novel Target-Aware Transformer for STVG (TA-STVG), which seeks to adaptively generate object queries via exploring target-specific cues from the given video-text pair, for improving STVG. The key lies in two simple yet effective modules, comprising text-guided temporal sampling (TTS) and attribute-aware spatial activation (ASA), working in a cascade. The former focuses on selecting target-relevant temporal cues from a video utilizing holistic text information, while the latter aims at further exploiting the fine-grained visual attribute information of the object from previous target-aware temporal cues, which is applied for object query initialization. Compared to existing methods leveraging zero-initialized queries, object queries in our TA-STVG, directly generated from a given video-text pair, naturally carry target-specific cues, making them adaptive and better interact with multimodal features for learning more discriminative information to improve STVG. In our experiments on three benchmarks, TA-STVG achieves state-of-the-art performance and significantly outperforms the baseline, validating its efficacy. 

**Abstract (ZH)**: 目标感知变压器在时空视频理解中的应用：TA-STVG 

---
# Large Language-Geometry Model: When LLM meets Equivariance 

**Title (ZH)**: 大语言-几何模型：当LLM遇见等变性 

**Authors**: Zongzhao Li, Jiacheng Cen, Bing Su, Wenbing Huang, Tingyang Xu, Yu Rong, Deli Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.11149)  

**Abstract**: Accurately predicting 3D structures and dynamics of physical systems is crucial in scientific applications. Existing approaches that rely on geometric Graph Neural Networks (GNNs) effectively enforce $\mathrm{E}(3)$-equivariance, but they often fall in leveraging extensive broader information. While direct application of Large Language Models (LLMs) can incorporate external knowledge, they lack the capability for spatial reasoning with guaranteed equivariance. In this paper, we propose EquiLLM, a novel framework for representing 3D physical systems that seamlessly integrates E(3)-equivariance with LLM capabilities. Specifically, EquiLLM comprises four key components: geometry-aware prompting, an equivariant encoder, an LLM, and an equivariant adaptor. Essentially, the LLM guided by the instructive prompt serves as a sophisticated invariant feature processor, while 3D directional information is exclusively handled by the equivariant encoder and adaptor modules. Experimental results demonstrate that EquiLLM delivers significant improvements over previous methods across molecular dynamics simulation, human motion simulation, and antibody design, highlighting its promising generalizability. 

**Abstract (ZH)**: 准确预测物理系统的三维结构和动态在科学研究中至关重要。现有依赖几何图神经网络（GNNs）的方法有效确保了$\mathrm{E}(3)$-拟不变性，但往往未能充分利用广泛的信息。虽然可以直接应用大型语言模型（LLMs）整合外部知识，但它们缺乏空间推理的能力和拟不变性保证。在本文中，我们提出了一种新的框架EquiLLM，该框架无缝地结合了E(3)-拟不变性和LLM的能力。具体而言，EquiLLM包含四个关键组成部分：几何感知的提示、拟不变性编码器、LLM以及拟不变性适配器。本质上，受指令性提示引导的LLM充当复杂的不变特征处理器，而3D方向性信息则由拟不变性编码器和适配器模块单独处理。实验结果表明，EquiLLM在分子动力学模拟、人类运动模拟和抗体设计方面均显著优于先前的方法，突显了其 promising 的泛化能力。 

---
# Efficient Long-Decoding Inference with Reasoning-Aware Attention Sparsity 

**Title (ZH)**: 长解码推理中的推理感知注意力稀疏性高效推理 

**Authors**: Junhao Hu, Wenrui Huang, Weidong Wang, Zhenwen Li, Tiancheng Hu, Zhixia Liu, Xusheng Chen, Tao Xie, Yizhou Shan  

**Link**: [PDF](https://arxiv.org/pdf/2502.11147)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities across various domains, with recent advancements in challenging reasoning tasks such as mathematics and programming. However, solving reasoning tasks often requires long decoding chains (of thoughts), which incur $O(N)$ time and memory consumption, where $N$ is the chain length. To mitigate $O(N)$ time and memory consumption, existing sparsity-based algorithms propose retaining only the most critical token's intermediate data (i.e., key-value cache) and discarding the rest. However, these existing algorithms struggle with the ``impossible trinity'' of accuracy, time, and memory. For example, the state-of-the-art algorithm, Quest, achieves high accuracy with $O(L)$ time but $O(N)$ memory ($L$ is the cache budget, $L \ll N$). To address this issue, in this paper, we identify a new attention pattern during the decode stage of reasoning tasks, where milestone tokens (analogous to lemmas in mathematical proofs) emerge, are utilized, and then become unimportant afterward. Based on this pattern, we propose a new algorithm named RaaS that identifies and retains milestone tokens only until they are no longer needed, achieving high accuracy with $O(L)$ time and $O(L)$ memory complexity. 

**Abstract (ZH)**: 大型语言模型在各种领域展示了强大的能力，特别是在数学和编程等具有挑战性的推理任务中取得了进展。然而，解决推理任务往往需要长的解码链（思考链），这会带来$O(N)$的时间和内存消耗，其中$N$是链的长度。为了减轻$O(N)$的时间和内存消耗，现有的基于稀疏性的算法建议仅保留关键令牌的中间数据（即键值缓存），并丢弃其他数据。然而，这些现有算法在准确度、时间和内存之间难以兼顾。“不可能的三角”困境表现为：最先进的算法Quest能够在$O(L)$时间内实现高准确度，但需要$O(N)$的内存（$L$为缓存预算，$L \ll N$）。为了解决这个问题，本文在推理任务的解码阶段识别出一种新的注意力模式，其中里程碑令牌（类似于数学证明中的引理）出现、被利用，随后变得不再重要。基于此模式，我们提出了一种新的算法RaaS，该算法仅在里程碑令牌不再需要时识别并保留它们，从而以$O(L)$的时间复杂度和$O(L)$的内存复杂度实现高准确度。 

---
# Cognitive Neural Architecture Search Reveals Hierarchical Entailment 

**Title (ZH)**: 认知神经架构搜索揭示层次蕴含关系 

**Authors**: Lukas Kuhn, Sari Saba-Sadiya, Gemma Roig  

**Link**: [PDF](https://arxiv.org/pdf/2502.11141)  

**Abstract**: Recent research has suggested that the brain is more shallow than previously thought, challenging the traditionally assumed hierarchical structure of the ventral visual pathway. Here, we demonstrate that optimizing convolutional network architectures for brain-alignment via evolutionary neural architecture search results in models with clear representational hierarchies. Despite having random weights, the identified models achieve brain-alignment scores surpassing even those of pretrained classification models - as measured by both regression and representational similarity analysis. Furthermore, through traditional supervised training, architectures optimized for alignment with late ventral regions become competitive classification models. These findings suggest that hierarchical structure is a fundamental mechanism of primate visual processing. Finally, this work demonstrates the potential of neural architecture search as a framework for computational cognitive neuroscience research that could reduce the field's reliance on manually designed convolutional networks. 

**Abstract (ZH)**: 近期的研究表明，大脑的层次结构可能比以前认为的要浅，挑战了传统上假设的腹侧视觉通路的层次结构。在这里，我们证明，通过进化神经架构搜索优化卷积网络架构以实现与大脑的对齐，可以产生具有明确表示层次结构的模型。尽管这些模型的权重是随机的，但它们的脑对齐分数甚至超过了预训练分类模型的分数，通过回归和表示相似性分析进行衡量。此外，通过传统的监督训练，优化与腹侧后期区域对齐的架构可以成为有竞争力的分类模型。这些发现表明，层次结构是灵长类动物视觉处理的基本机制。最后，本研究展示了进化神经架构搜索作为计算认知神经科学研究框架的潜力，该框架可以减少该领域对手动设计卷积网络的依赖。 

---
# VisPath: Automated Visualization Code Synthesis via Multi-Path Reasoning and Feedback-Driven Optimization 

**Title (ZH)**: VisPath: 基于多路径推理和反馈驱动优化的自动可视化代码合成 

**Authors**: Wonduk Seo, Seungyong Lee, Daye Kang, Zonghao Yuan, Seunghyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.11140)  

**Abstract**: Unprecedented breakthroughs in Large Language Models (LLMs) has amplified its penetration into application of automated visualization code generation. Few-shot prompting and query expansion techniques have notably enhanced data visualization performance, however, still fail to overcome ambiguity and complexity of natural language queries - imposing an inherent burden for manual human intervention. To mitigate such limitations, we propose a holistic framework VisPath : A Multi-Path Reasoning and Feedback-Driven Optimization Framework for Visualization Code Generation, which systematically enhances code quality through structured reasoning and refinement. VisPath is a multi-stage framework, specially designed to handle underspecified queries. To generate a robust final visualization code, it first utilizes initial query to generate diverse reformulated queries via Chain-of-Thought (CoT) prompting, each representing a distinct reasoning path. Refined queries are used to produce candidate visualization scripts, consequently executed to generate multiple images. Comprehensively assessing correctness and quality of outputs, VisPath generates feedback for each image, which are then fed to aggregation module to generate optimal result. Extensive experiments on benchmarks including MatPlotBench and the Qwen-Agent Code Interpreter Benchmark show that VisPath significantly outperforms state-of-the-art (SOTA) methods, increased up to average 17%, offering a more reliable solution for AI-driven visualization code generation. 

**Abstract (ZH)**: 大型语言模型（LLMs）前所未有的突破使其在自动化可视化代码生成中的应用更加广泛。尽管少量示例提示和查询扩展技术显著提升了数据可视化性能，但仍无法克服自然语言查询的模糊性和复杂性，从而增加了手动人工干预的负担。为了克服这些限制，我们提出了一种综合框架VisPath：一种多路径推理和反馈驱动优化框架，以系统地通过结构化推理和优化提升代码质量。VisPath是一个多阶段框架，专门设计用于处理不明确查询。为了生成 robust 的最终可视化代码，它首先利用初始查询生成多种多样重新表述的查询，每个查询代表不同的推理路径。经过优化的查询用于生成候选可视化脚本，随后执行以生成多张图像。综合评估输出的正确性和质量后，VisPath 为每张图像生成反馈，这些反馈随后被反馈到聚合模块以生成最优结果。在包括MatPlotBench和Qwen-Agent Code Interpreter Benchmark在内的基准测试中，VisPath 显著优于现有最先进的方法，平均提高多达17%，提供了一种更可靠的人工智能驱动的可视化代码生成解决方案。 

---
# Safety Evaluation of DeepSeek Models in Chinese Contexts 

**Title (ZH)**: DeepSeek模型在中国语境下的安全性评估 

**Authors**: Wenjing Zhang, Xuejiao Lei, Zhaoxiang Liu, Ning Wang, Zhenhong Long, Peijun Yang, Jiaojiao Zhao, Minjie Hua, Chaoyang Ma, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2502.11137)  

**Abstract**: Recently, the DeepSeek series of models, leveraging their exceptional reasoning capabilities and open-source strategy, is reshaping the global AI landscape. Despite these advantages, they exhibit significant safety deficiencies. Research conducted by Robust Intelligence, a subsidiary of Cisco, in collaboration with the University of Pennsylvania, revealed that DeepSeek-R1 has a 100\% attack success rate when processing harmful prompts. Additionally, multiple safety companies and research institutions have confirmed critical safety vulnerabilities in this model. As models demonstrating robust performance in Chinese and English, DeepSeek models require equally crucial safety assessments in both language contexts. However, current research has predominantly focused on safety evaluations in English environments, leaving a gap in comprehensive assessments of their safety performance in Chinese contexts. In response to this gap, this study introduces CHiSafetyBench, a Chinese-specific safety evaluation benchmark. This benchmark systematically evaluates the safety of DeepSeek-R1 and DeepSeek-V3 in Chinese contexts, revealing their performance across safety categories. The experimental results quantify the deficiencies of these two models in Chinese contexts, providing key insights for subsequent improvements. 

**Abstract (ZH)**: 最近，DeepSeek系列模型凭借其卓越的推理能力及开源策略，正在重塑全球AI格局。然而，它们在安全性方面存在显著缺陷。罗伯特智能（Cisco的子公司）与宾夕法尼亚大学合作的研究发现，DeepSeek-R1在处理有害提示时具有100%的攻击成功率，同时还确认了该模型存在多个关键的安全漏洞。尽管DeepSeek模型在中英文环境中均显示出了强劲的表现，但对其中文环境下的安全性评估同样至关重要。然而，当前的研究主要集中在英文环境下的安全评估，忽视了对其在中国情境下的全面安全性能评估。针对这一空白，本研究引入了CHiSafetyBench，一个专门针对中文环境的安全评估基准，系统性地评估了DeepSeek-R1和DeepSeek-V3在中文环境下的安全性，揭示了它们在不同安全类别中的表现。实验结果量化了这两种模型在中国环境中的不足之处，为后续改进提供了关键见解。 

---
# UNITE-FND: Reframing Multimodal Fake News Detection through Unimodal Scene Translation 

**Title (ZH)**: UNITE-FND: 通过单模态场景翻译重新定义多模态假新闻检测 

**Authors**: Arka Mukherjee, Shreya Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2502.11132)  

**Abstract**: Multimodal fake news detection typically demands complex architectures and substantial computational resources, posing deployment challenges in real-world settings. We introduce UNITE-FND, a novel framework that reframes multimodal fake news detection as a unimodal text classification task. We propose six specialized prompting strategies with Gemini 1.5 Pro, converting visual content into structured textual descriptions, and enabling efficient text-only models to preserve critical visual information. To benchmark our approach, we introduce Uni-Fakeddit-55k, a curated dataset family of 55,000 samples each, each processed through our multimodal-to-unimodal translation framework. Experimental results demonstrate that UNITE-FND achieves 92.52% accuracy in binary classification, surpassing prior multimodal models while reducing computational costs by over 10x (TinyBERT variant: 14.5M parameters vs. 250M+ in SOTA models). Additionally, we propose a comprehensive suite of five novel metrics to evaluate image-to-text conversion quality, ensuring optimal information preservation. Our results demonstrate that structured text-based representations can replace direct multimodal processing with minimal loss of accuracy, making UNITE-FND a practical and scalable alternative for resource-constrained environments. 

**Abstract (ZH)**: 多模态假新闻检测通常需要复杂的架构和大量的计算资源，难以在实际场景中部署。我们提出了UNITE-FND，一种将多模态假新闻检测重新定义为单模态文本分类任务的新框架。我们提出六种专门的提示策略，并使用Gemini 1.5 Pro将视觉内容转换为结构化的文本描述，使高效的文字模型能够保留关键的视觉信息。为了评估我们的方法，我们引入了Uni-Fakeddit-55k数据集家族，包含55,000个样本，每个样本均通过我们的多模态到单模态转换框架进行处理。实验结果表明，UNITE-FND在二分类中的准确率达到92.52%，超越了先前的多模态模型，同时计算成本降低了超过10倍（TinyBERT变体参数量为14.5M，而当前最佳模型参数量超过250M）。此外，我们还提出了一套包含五项新型指标的综合评估方案，以确保图像到文本转换的质量。结果显示，结构化的文本表示可以在几乎不损失准确性的情况下替代直接的多模态处理，使UNITE-FND成为资源受限环境中的一种实用且可扩展的选择。 

---
# AdaManip: Adaptive Articulated Object Manipulation Environments and Policy Learning 

**Title (ZH)**: AdaManip: 自适应 articulated 物体 manipulation 环境及策略学习 

**Authors**: Yuanfei Wang, Xiaojie Zhang, Ruihai Wu, Yu Li, Yan Shen, Mingdong Wu, Zhaofeng He, Yizhou Wang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11124)  

**Abstract**: Articulated object manipulation is a critical capability for robots to perform various tasks in real-world scenarios. Composed of multiple parts connected by joints, articulated objects are endowed with diverse functional mechanisms through complex relative motions. For example, a safe consists of a door, a handle, and a lock, where the door can only be opened when the latch is unlocked. The internal structure, such as the state of a lock or joint angle constraints, cannot be directly observed from visual observation. Consequently, successful manipulation of these objects requires adaptive adjustment based on trial and error rather than a one-time visual inference. However, previous datasets and simulation environments for articulated objects have primarily focused on simple manipulation mechanisms where the complete manipulation process can be inferred from the object's appearance. To enhance the diversity and complexity of adaptive manipulation mechanisms, we build a novel articulated object manipulation environment and equip it with 9 categories of objects. Based on the environment and objects, we further propose an adaptive demonstration collection and 3D visual diffusion-based imitation learning pipeline that learns the adaptive manipulation policy. The effectiveness of our designs and proposed method is validated through both simulation and real-world experiments. Our project page is available at: this https URL 

**Abstract (ZH)**: articulated物体操作是机器人在实际场景中执行各种任务的关键能力。由关节连接多个部分组成的articulated物体通过复杂相对运动具备多样化的功能机制。例如，一个保险箱包括门、把手和锁，只有当锁扣解锁时门才能打开。内部结构，如锁的状态或关节角度约束，无法通过视觉观察直接观察到。因此，成功操纵这些物体需要基于尝试和错误的适应性调整，而不是一次性的视觉推理。然而，以往articulated物体的数据集和仿真环境主要集中在简单的操作机制上，通过物体的外观可以推断出完整的操作过程。为了增强适应性操作机制的多样性和复杂性，我们构建了一个新的articulated物体操作环境，并配备了9类物体。基于该环境和物体，我们进一步提出了一种适应性演示采集及基于3D视觉扩散的模仿学习管道，以学习适应性操作策略。我们通过仿真和真实世界实验验证了设计和提出的有效方法。项目页面可访问：this https URL。 

---
# Knowledge Graph-Driven Retrieval-Augmented Generation: Integrating Deepseek-R1 with Weaviate for Advanced Chatbot Applications 

**Title (ZH)**: 知识图谱驱动的检索增强生成：将Deepseek-R1与Weaviate集成以实现高级聊天机器人应用 

**Authors**: Alexandru Lecu, Adrian Groza, Lezan Hawizy  

**Link**: [PDF](https://arxiv.org/pdf/2502.11108)  

**Abstract**: Large language models (LLMs) have significantly advanced the field of natural language generation. However, they frequently generate unverified outputs, which compromises their reliability in critical applications. In this study, we propose an innovative framework that combines structured biomedical knowledge with LLMs through a retrieval-augmented generation technique. Our system develops a thorough knowledge graph by identifying and refining causal relationships and named entities from medical abstracts related to age-related macular degeneration (AMD). Using a vector-based retrieval process and a locally deployed language model, our framework produces responses that are both contextually relevant and verifiable, with direct references to clinical evidence. Experimental results show that this method notably decreases hallucinations, enhances factual precision, and improves the clarity of generated responses, providing a robust solution for advanced biomedical chatbot applications. 

**Abstract (ZH)**: 大型语言模型(Large Language Models, LLMs)在自然语言生成领域取得了显著进展。然而，它们经常生成未经验证的输出，这在关键应用中损害了其可靠性。本研究提出了一种结合结构化生物医学知识与LLMs的创新框架，通过检索增强生成技术。我们的系统通过识别和精炼与年龄相关黄斑变性(AMD)相关的医学摘要中的因果关系和命名实体，构建了一个详尽的知识图谱。利用基于向量的检索过程和本地部署的语言模型，我们的框架生成了既相关又可验证的响应，并直接引用了临床证据。实验结果表明，这种方法显著降低了幻觉，提高了事实的精确性，并提高了生成响应的清晰度，为高级生物医学聊天机器人应用提供了稳健的解决方案。 

---
# Revisiting Weak-to-Strong Generalization in Theory and Practice: Reverse KL vs. Forward KL 

**Title (ZH)**: 重新审视从弱泛化到强泛化的理论与实践：逆KL散度 vs. 正向KL散度 

**Authors**: Wei Yao, Wenkai Yang, Ziqiao Wang, Yankai Lin, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11107)  

**Abstract**: As large language models advance toward superhuman performance, ensuring their alignment with human values and abilities grows increasingly complex. Weak-to-strong generalization offers a promising approach by leveraging predictions from weaker models to guide stronger systems, but its effectiveness could be constrained by the inherent noise and inaccuracies in these weak predictions. To address this, we propose a theoretically grounded approach that replaces forward KL divergence-whose mass-covering behavior risks overfitting to imperfect weak signals-with reverse KL divergence. Reverse KL divergence's zero-forcing effect prioritizes high-confidence predictions, effectively mitigating the influence of unreliable weak supervision. Theoretically, we extend existing bounds and derive tighter lower bounds for both forward and reverse KL divergence, establishing that reverse KL achieves at least comparable guarantees to forward KL. Notably, when a sufficiently pre-trained strong model is fine-tuned on the last layer, reverse KL uniquely guarantees that it outperforms its weak supervisor by the magnitude of their disagreement-a guarantee that forward KL cannot provide. Empirically, we demonstrate that reverse KL and reverse cross-entropy enable strong models to consistently outperform those trained with forward KL and standard cross-entropy across most settings, highlighting the practical advantages of these reverse losses. 

**Abstract (ZH)**: 随着大型语言模型向超人类性能迈进，确保其与人类价值观和能力的对齐日益复杂。通过利用较弱模型的预测来引导较强系统的方法——弱到强泛化的潜力不容忽视，但其效果可能受到这些较弱预测中固有的噪音和不准确性的影响。为此，我们提出了一种理论支持的方法，替代可能导致对不完美弱信号过度拟合的前向KL散度，而是采用后向KL散度。后向KL散度的零强迫效应强调高置信度的预测，有效地减轻了不可靠弱监督的影响。从理论上讲，我们扩展了现有的边界，并为前向和后向KL散度推导了更紧的下界，证明了后向KL至少能达到与前向KL相当的保证。值得注意的是，当一个充分预训练的强模型在最后一层进行微调时，后向KL唯一地保证其在争议程度上优于其弱监督者——这是前向KL无法提供的保证。实验上，我们展示了后向KL和后向交叉熵使强模型在大多数设置中一致地优于使用前向KL和标准交叉熵训练的模型，突出了这些后向损失的实际优势。 

---
# CacheFocus: Dynamic Cache Re-Positioning for Efficient Retrieval-Augmented Generation 

**Title (ZH)**: CacheFocus: 动态缓存重新定位以实现高效的检索增强生成 

**Authors**: Kun-Hui Lee, Eunhwan Park, Donghoon Han, Seung-Hoon Na  

**Link**: [PDF](https://arxiv.org/pdf/2502.11101)  

**Abstract**: Large Language Models (LLMs) excel across a variety of language tasks yet are constrained by limited input lengths and high computational costs. Existing approaches\textemdash such as relative positional encodings (e.g., RoPE, ALiBi) and sliding window mechanisms\textemdash partially alleviate these issues but often require additional training or suffer from performance degradation with longer inputs. In this paper, we introduce \textbf{\textit{CacheFocus}}, a method that enhances length normalization and reduces inference latency without any further training. Our approach leverages query-independent, offline caching to efficiently reuse a Context KV Cache Store. We address the amplification of abnormal token distributions problem by re-positioning cached keys and introducing Layer-Adaptive Cache Pruning to discard low-relevance caches during pre-filling. Additionally, our Adaptive Positional Allocation Strategy dynamically reassigns cache positions to maximize the use of the available positional encoding range. Experiments on the Natural Questions and TriviaQA datasets demonstrate that CacheFocus outperforms alternative methods even when inputs exceed the $4$K limit of the \texttt{LLaMA-2} model, emphasizing its practical effectiveness for long-context LLMs. Moreover, even with large maximum input length of \texttt{Qwen2}, the performance of CacheFocus shows that it maintains consistent performance even as the number of documents increases, effectively managing long-text generation without degradation. 

**Abstract (ZH)**: Large Language Models（LLMs）在多种语言任务中表现出色，但受限于输入长度限制和高计算成本。现有方法如相对位置编码（例如RoPE、ALiBi）和滑动窗口机制部分缓解了这些问题，但往往需要额外训练或在长输入时性能下降。本文介绍了一种名为**CacheFocus**的方法，该方法在无需额外训练的情况下提升了长度归一化并减少了推理延迟。我们的方法利用查询无关的离线缓存高效重用上下文KV缓存存储。通过重新定位缓存键并引入层自适应缓存剪枝来解决异常 token 分布放大的问题，同时在预填充阶段丢弃相关性低的缓存。此外，我们的自适应位置分配策略动态重新分配缓存位置，以充分利用可用的位置编码范围。在Natural Questions和TriviaQA数据集上的实验证明，CacheFocus即使在输入超过LLaMA-2模型的4K限制时也优于其他方法，强调了其在长上下文LLMs中的实际有效性。此外，即使在Qwen2模型具有大最大输入长度的情况下，CacheFocus的性能也随着文档数量的增加而保持稳定，有效地管理长文本生成而不出现性能下降。 

---
# SyncSpeech: Low-Latency and Efficient Dual-Stream Text-to-Speech based on Temporal Masked Transformer 

**Title (ZH)**: SyncSpeech:基于时间掩码变压器的低延迟高效双流文本-to-语音技术 

**Authors**: Zhengyan Sheng, Zhihao Du, Shiliang Zhang, Zhijie Yan, Yexin Yang, Zhenhua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2502.11094)  

**Abstract**: This paper presents a dual-stream text-to-speech (TTS) model, SyncSpeech, capable of receiving streaming text input from upstream models while simultaneously generating streaming speech, facilitating seamless interaction with large language models. SyncSpeech has the following advantages: Low latency, as it begins generating streaming speech upon receiving the second text token; High efficiency, as it decodes all speech tokens corresponding to the each arrived text token in one step. To achieve this, we propose a temporal masked transformer as the backbone of SyncSpeech, combined with token-level duration prediction to predict speech tokens and the duration for the next step. Additionally, we design a two-stage training strategy to improve training efficiency and the quality of generated speech. We evaluated the SyncSpeech on both English and Mandarin datasets. Compared to the recent dual-stream TTS models, SyncSpeech significantly reduces the first packet delay of speech tokens and accelerates the real-time factor. Moreover, with the same data scale, SyncSpeech achieves performance comparable to that of traditional autoregressive-based TTS models in terms of both speech quality and robustness. Speech samples are available at this https URL}{this https URL. 

**Abstract (ZH)**: 本文提出了一种双流文本到语音（TTS）模型SyncSpeech，能够在接收来自上游模型的流式文本输入的同时生成流式语音，从而与大规模语言模型实现无缝交互。SyncSpeech具有以下优势：低延迟，接收到第二个文本令牌即开始生成流式语音；高效率，通过一步解码计算每个到达文本令牌对应的全部语音令牌。为此，我们以时间掩码变换器作为SyncSpeech的骨干，并结合令牌级时长预测来预测下一个步骤的语音令牌及其时长。此外，我们设计了两阶段训练策略以提高训练效率和生成语音的质量。我们在英语和 Mandarin 数据集上对SyncSpeech进行了评估。与最新的双流TTS模型相比，SyncSpeech显著减少了语音令牌的第一包延迟并加快了实时因子。此外，在相同的数据规模下，SyncSpeech在语音质量与鲁棒性方面均达到了基于自回归的传统TTS模型的性能。语音样本可通过以下链接访问：this https URL、this https URL。 

---
# SafeDialBench: A Fine-Grained Safety Benchmark for Large Language Models in Multi-Turn Dialogues with Diverse Jailbreak Attacks 

**Title (ZH)**: SafeDialBench：大语言模型在多轮对话中对抗多样化脱管攻击的安全细粒度基准 

**Authors**: Hongye Cao, Yanming Wang, Sijia Jing, Ziyue Peng, Zhixin Bai, Zhe Cao, Meng Fang, Fan Feng, Boyan Wang, Jiaheng Liu, Tianpei Yang, Jing Huo, Yang Gao, Fanyu Meng, Xi Yang, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2502.11090)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs), the safety of LLMs has been a critical concern requiring precise assessment. Current benchmarks primarily concentrate on single-turn dialogues or a single jailbreak attack method to assess the safety. Additionally, these benchmarks have not taken into account the LLM's capability of identifying and handling unsafe information in detail. To address these issues, we propose a fine-grained benchmark SafeDialBench for evaluating the safety of LLMs across various jailbreak attacks in multi-turn dialogues. Specifically, we design a two-tier hierarchical safety taxonomy that considers 6 safety dimensions and generates more than 4000 multi-turn dialogues in both Chinese and English under 22 dialogue scenarios. We employ 7 jailbreak attack strategies, such as reference attack and purpose reverse, to enhance the dataset quality for dialogue generation. Notably, we construct an innovative assessment framework of LLMs, measuring capabilities in detecting, and handling unsafe information and maintaining consistency when facing jailbreak attacks. Experimental results across 17 LLMs reveal that Yi-34B-Chat and GLM4-9B-Chat demonstrate superior safety performance, while Llama3.1-8B-Instruct and o3-mini exhibit safety vulnerabilities. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的迅速发展，LLMs的安全性已成为一个关键关切，需要精确评估。当前的基准主要集中在单轮对话或单一脱逃攻击方法上进行安全性评估，并没有详细考虑LLMs识别和处理不安全信息的能力。为解决这些问题，我们提出了一种细粒度基准SafeDialBench，用于评估LLMs在多轮对话中多种脱逃攻击下的安全性。具体地，我们设计了一个两层分级安全分类体系，考虑了6个安全维度，并生成了超过4000个多轮对话，涵盖22种对话场景，包括中文和英文。我们采用了包括引用攻击和目的反向在内的7种脱逃攻击策略，以提高对话生成数据集的质量。值得注意的是，我们构建了一种创新的LLM评估框架，用于衡量识别和处理不安全信息以及在面对脱逃攻击时保持一致性的能力。实验结果显示，Yi-34B-Chat和GLM4-9B-Chat表现出更优异的安全性能，而Llama3.1-8B-Instruct和o3-mini则显示出安全性漏洞。 

---
# Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention 

**Title (ZH)**: 原生稀疏注意力：硬件对齐且原生可训练的稀疏注意力 

**Authors**: Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, Wangding Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2502.11089)  

**Abstract**: Long-context modeling is crucial for next-generation language models, yet the high computational cost of standard attention mechanisms poses significant computational challenges. Sparse attention offers a promising direction for improving efficiency while maintaining model capabilities. We present NSA, a Natively trainable Sparse Attention mechanism that integrates algorithmic innovations with hardware-aligned optimizations to achieve efficient long-context modeling. NSA employs a dynamic hierarchical sparse strategy, combining coarse-grained token compression with fine-grained token selection to preserve both global context awareness and local precision. Our approach advances sparse attention design with two key innovations: (1) We achieve substantial speedups through arithmetic intensity-balanced algorithm design, with implementation optimizations for modern hardware. (2) We enable end-to-end training, reducing pretraining computation without sacrificing model performance. As shown in Figure 1, experiments show the model pretrained with NSA maintains or exceeds Full Attention models across general benchmarks, long-context tasks, and instruction-based reasoning. Meanwhile, NSA achieves substantial speedups over Full Attention on 64k-length sequences across decoding, forward propagation, and backward propagation, validating its efficiency throughout the model lifecycle. 

**Abstract (ZH)**: 长上下文建模是下一代语言模型的关键，但标准注意力机制的高计算成本带来了显著的计算挑战。稀疏注意力机制为在保持模型能力的同时提高效率提供了有前途的方向。我们提出了一种名为NSA的本征可训练稀疏注意力机制，将算法创新与硬件对齐的优化相结合，以实现高效的长上下文建模。NSA采用动态分层稀疏策略，结合粗粒度的 token 压缩与细粒度的 token 选择，以保持全局上下文意识和局部精度。我们的方法在稀疏注意力设计方面实现了两项关键创新：(1) 通过算术强度平衡的算法设计实现显著加速，并针对现代硬件进行实现优化。(2) 实现端到端训练，减少预训练计算量而不牺牲模型性能。如图1所示，实验表明使用NSA预训练的模型在通用基准、长上下文任务和基于指令的推理方面均能保持或超越全注意力模型。同时，NSA在64k长度序列的解码、前向传播和反向传播中实现了显著加速，验证了其在整个模型生命周期中的高效性。 

---
# Towards Data-Efficient Pretraining for Atomic Property Prediction 

**Title (ZH)**: 面向原子性质预测的数据高效预训练 

**Authors**: Yasir Ghunaim, Hasan Abed Al Kader Hammoud, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2502.11085)  

**Abstract**: This paper challenges the recent paradigm in atomic property prediction that links progress to growing dataset sizes and computational resources. We show that pretraining on a carefully selected, task-relevant dataset can match or even surpass large-scale pretraining, while using as little as 1/24th of the computational cost. We introduce the Chemical Similarity Index (CSI), a novel metric inspired by computer vision's Fréchet Inception Distance, for molecular graphs which quantifies the alignment between upstream pretraining datasets and downstream tasks. By selecting the most relevant dataset with minimal CSI distance, we show that models pretrained on a smaller, focused dataset consistently outperform those pretrained on massive, mixed datasets such as JMP, even when those larger datasets include the relevant dataset. Counterintuitively, we also find that indiscriminately adding more data can degrade model performance when the additional data poorly aligns with the task at hand. Our findings highlight that quality often outperforms quantity in pretraining for atomic property prediction. 

**Abstract (ZH)**: 本文挑战了原子性质预测中数据集规模和计算资源增长驱动进步的近期范式，表明在精心选择的相关任务数据集上的预训练可以达到甚至超越大规模预训练的效果，同时仅使用后者的1/24的计算成本。我们引入了化学相似性指数（CSI），这是一种受计算机视觉的弗雷歇入眼距离启发的新颖指标，用于分子图，量化上游预训练数据集与下游任务的对齐程度。通过选择CSI距离最小的最具相关性的数据集，我们展示出在较小且聚焦的数据集上预训练的模型在各种任务中通常优于在大规模混合数据集（如JMP）上预训练的模型，即使后者包括了相关的数据集。出乎意料的是，我们还发现不分青红皂白地增加数据量在任务不匹配的情况下反而会降低模型性能。我们的研究结果强调，在原子性质预测的预训练中，质量往往胜过数量。 

---
# Phantom: Subject-consistent video generation via cross-modal alignment 

**Title (ZH)**: 幻影：基于跨模态对齐的主体一致视频生成 

**Authors**: Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Qian He, Xinglong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11079)  

**Abstract**: The continuous development of foundational models for video generation is evolving into various applications, with subject-consistent video generation still in the exploratory stage. We refer to this as Subject-to-Video, which extracts subject elements from reference images and generates subject-consistent video through textual instructions. We believe that the essence of subject-to-video lies in balancing the dual-modal prompts of text and image, thereby deeply and simultaneously aligning both text and visual content. To this end, we propose Phantom, a unified video generation framework for both single and multi-subject references. Building on existing text-to-video and image-to-video architectures, we redesign the joint text-image injection model and drive it to learn cross-modal alignment via text-image-video triplet data. In particular, we emphasize subject consistency in human generation, covering existing ID-preserving video generation while offering enhanced advantages. The project homepage is here this https URL. 

**Abstract (ZH)**: 基于文本和图像的双模态提示平衡的Subject-to-Video统一视频生成框架：Phantom 

---
# Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in Large Language Models 

**Title (ZH)**: 暴露数值能力差距：评估大型语言模型基本数值能力的标准 

**Authors**: Haoyang Li, Xuejia Chen, Zhanchao XU, Darian Li, Nicole Hu, Fei Teng, Yiming Li, Luyu Qiu, Chen Jason Zhang, Qing Li, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11075)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in natural language processing tasks, such as text generation and semantic understanding. However, their performance on numerical reasoning tasks, such as basic arithmetic, numerical retrieval, and magnitude comparison, remains surprisingly poor. This gap arises from their reliance on surface-level statistical patterns rather than understanding numbers as continuous magnitudes. Existing benchmarks primarily focus on either linguistic competence or structured mathematical problem-solving, neglecting fundamental numerical reasoning required in real-world scenarios. To bridge this gap, we propose NumericBench, a comprehensive benchmark to evaluate six fundamental numerical capabilities: number recognition, arithmetic operations, contextual retrieval, comparison, summary, and logical reasoning. NumericBench includes datasets ranging from synthetic number lists to the crawled real-world data, addressing challenges like long contexts, noise, and multi-step reasoning. Extensive experiments on state-of-the-art LLMs, including GPT-4 and DeepSeek, reveal persistent weaknesses in numerical reasoning, highlighting the urgent need to improve numerically-aware language modeling. The benchmark is released in: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理任务如文本生成和语义理解方面展示了令人印象深刻的能力。然而，在基本算术、数值检索和数量比较等数值推理任务上的表现仍然出人意料地差。这种差距源于它们依赖于表面级的统计模式，而不是理解数字作为连续量的意义。现有的基准测试主要关注语言能力或结构化数学问题解决，忽视了真实世界场景中所需的最基本数值推理能力。为了弥合这一差距，我们提出了NumericBench，一个全面的基准测试，用于评估六个基本的数值能力：数字识别、算术运算、上下文检索、比较、总结和逻辑推理。NumericBench 包括从合成数字列表到抓取的真实世界数据的数据集，解决了长上下文、噪声和多步推理等挑战。在最先进的大语言模型（包括GPT-4和DeepSeek）上的广泛实验揭示了数值推理中的持久性弱点，突显了提高数值感知语言建模的迫切需求。基准测试已发布于：this https URL。 

---
# A Survey on Vulnerability Prioritization: Taxonomy, Metrics, and Research Challenges 

**Title (ZH)**: 漏洞优先级研究：分类、度量及研究挑战 

**Authors**: Yuning Jiang, Nay Oo, Qiaoran Meng, Hoon Wei Lim, Biplab Sikdar  

**Link**: [PDF](https://arxiv.org/pdf/2502.11070)  

**Abstract**: In the highly interconnected digital landscape of today, safeguarding complex infrastructures against cyber threats has become increasingly challenging due to the exponential growth in the number and complexity of vulnerabilities. Resource constraints necessitate effective vulnerability prioritization strategies, focusing efforts on the most critical risks. This paper presents a systematic literature review of 82 studies, introducing a novel taxonomy that categorizes metrics into severity, exploitability, contextual factors, predictive indicators, and aggregation methods. Our analysis reveals significant gaps in existing approaches and challenges with multi-domain applicability. By emphasizing the need for dynamic, context-aware metrics and scalable solutions, we provide actionable insights to bridge the gap between research and real-world applications. This work contributes to the field by offering a comprehensive framework for evaluating vulnerability prioritization methodologies and setting a research agenda to advance the state of practice. 

**Abstract (ZH)**: 在当今高度互联的数字 landscape 中，鉴于漏洞数量和复杂性的指数级增长，保护复杂基础设施免受网络安全威胁变得日益challenge。资源约束催生了有效的漏洞优先级策略，强调应对最关键风险的努力。本文综述了 82 篇相关研究，提出了一种新颖的分类体系，将指标分为严重性、利用性、情境因素、预测性指标和聚合方法。我们的分析揭示了现有方法中的重大缺口和跨领域应用的挑战。通过强调动态、情境感知指标和可扩展解决方案的必要性，我们提供了将研究与实际应用接轨的行动指南。本研究通过提供评估漏洞优先级方法的全面框架并推动研究议程，为该领域做出了贡献。 

---
# Accelerating Anchors via Specialization and Feature Transformation 

**Title (ZH)**: 基于专业化和特征变换的锚点加速方法 

**Authors**: Haonan Yu, Junhao Liu, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11068)  

**Abstract**: Anchors is a popular local model-agnostic explanation technique whose applicability is limited by its computational inefficiency. To address this limitation, we propose a pre-training-based approach to accelerate Anchors without compromising the explanation quality. Our approach leverages the iterative nature of Anchors' algorithm which gradually refines an explanation until it is precise enough for a given input by providing a general explanation that is obtained through pre-training as Anchors' initial explanation. Specifically, we develop a two-step rule transformation process: the horizontal transformation adapts a pre-trained explanation to the current input by replacing features, and the vertical transformation refines the general explanation until it is precise enough for the input. We evaluate our method across tabular, text, and image datasets, demonstrating that it significantly reduces explanation generation time while maintaining fidelity and interpretability, thereby enabling the practical adoption of Anchors in time-sensitive applications. 

**Abstract (ZH)**: 基于预训练的Anchors加速方法：在不牺牲解释质量的前提下提高计算效率 

---
# ClimateLLM: Efficient Weather Forecasting via Frequency-Aware Large Language Models 

**Title (ZH)**: ClimateLLM：基于频率意识的大语言模型高效天气预报 

**Authors**: Shixuan Li, Wei Yang, Peiyu Zhang, Xiongye Xiao, Defu Cao, Yuehan Qin, Xiaole Zhang, Yue Zhao, Paul Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2502.11059)  

**Abstract**: Weather forecasting is crucial for public safety, disaster prevention and mitigation, agricultural production, and energy management, with global relevance. Although deep learning has significantly advanced weather prediction, current methods face critical limitations: (i) they often struggle to capture both dynamic temporal dependencies and short-term abrupt changes, making extreme weather modeling difficult; (ii) they incur high computational costs due to extensive training and resource requirements; (iii) they have limited adaptability to multi-scale frequencies, leading to challenges when separating global trends from local fluctuations. To address these issues, we propose ClimateLLM, a foundation model for weather forecasting. It captures spatiotemporal dependencies via a cross-temporal and cross-spatial collaborative modeling framework that integrates Fourier-based frequency decomposition with Large Language Models (LLMs) to strengthen spatial and temporal modeling. Our framework uses a Mixture-of-Experts (MoE) mechanism that adaptively processes different frequency components, enabling efficient handling of both global signals and localized extreme events. In addition, we introduce a cross-temporal and cross-spatial dynamic prompting mechanism, allowing LLMs to incorporate meteorological patterns across multiple scales effectively. Extensive experiments on real-world datasets show that ClimateLLM outperforms state-of-the-art approaches in accuracy and efficiency, as a scalable solution for global weather forecasting. 

**Abstract (ZH)**: 气候LLM：一种用于天气预报的基础模型 

---
# A Physics-Informed Machine Learning Framework for Safe and Optimal Control of Autonomous Systems 

**Title (ZH)**: 基于物理知识的机器学习框架：用于自主系统安全与最优控制 

**Authors**: Manan Tayal, Aditya Singh, Shishir Kolathaya, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2502.11057)  

**Abstract**: As autonomous systems become more ubiquitous in daily life, ensuring high performance with guaranteed safety is crucial. However, safety and performance could be competing objectives, which makes their co-optimization difficult. Learning-based methods, such as Constrained Reinforcement Learning (CRL), achieve strong performance but lack formal safety guarantees due to safety being enforced as soft constraints, limiting their use in safety-critical settings. Conversely, formal methods such as Hamilton-Jacobi (HJ) Reachability Analysis and Control Barrier Functions (CBFs) provide rigorous safety assurances but often neglect performance, resulting in overly conservative controllers. To bridge this gap, we formulate the co-optimization of safety and performance as a state-constrained optimal control problem, where performance objectives are encoded via a cost function and safety requirements are imposed as state constraints. We demonstrate that the resultant value function satisfies a Hamilton-Jacobi-Bellman (HJB) equation, which we approximate efficiently using a novel physics-informed machine learning framework. In addition, we introduce a conformal prediction-based verification strategy to quantify the learning errors, recovering a high-confidence safety value function, along with a probabilistic error bound on performance degradation. Through several case studies, we demonstrate the efficacy of the proposed framework in enabling scalable learning of safe and performant controllers for complex, high-dimensional autonomous systems. 

**Abstract (ZH)**: 随着自主系统在日常生活中的普及，确保在有保证的安全性下的高性能至关重要。然而，安全性和性能可能是相互竞争的目标，这使得它们的共同优化变得困难。基于学习的方法，如受限强化学习（CRL），可以获得强大的性能，但由于安全要求是以软约束的形式施加的，缺乏形式上的安全保证，限制了它们在安全关键设置中的应用。相反，形式化方法，如哈密尔顿-雅可比（HJ）可达性分析和控制屏障函数（CBFs），可以提供严格的安全保证，但通常会忽略性能，导致过于保守的控制器。为了弥合这一差距，我们将安全性和性能的共同优化形式化为状态约束最优控制问题，其中性能目标通过成本函数编码，安全性要求作为状态约束施加。我们证明了所得的价值函数满足哈密尔顿-雅可比-贝尔曼（HJB）方程，并使用一个新颖的物理知识嵌入机器学习框架对其进行有效近似。此外，我们引入了一种符合性预测为基础的验证策略来量化学习误差，恢复高可信度的安全价值函数，同时提供性能退化的一个概率误差边界。通过几个案例研究，我们展示了所提出框架的有效性，能够在复杂、高维自主系统中实现可扩展的学习安全且性能良好的控制器。 

---
# Reasoning-Augmented Conversation for Multi-Turn Jailbreak Attacks on Large Language Models 

**Title (ZH)**: 增强推理的多轮 Jailbreak 攻击对话模型推理增强的多轮 Jailbreak 攻击对话模型 

**Authors**: Zonghao Ying, Deyue Zhang, Zonglei Jing, Yisong Xiao, Quanchen Zou, Aishan Liu, Siyuan Liang, Xiangzheng Zhang, Xianglong Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2502.11054)  

**Abstract**: Multi-turn jailbreak attacks simulate real-world human interactions by engaging large language models (LLMs) in iterative dialogues, exposing critical safety vulnerabilities. However, existing methods often struggle to balance semantic coherence with attack effectiveness, resulting in either benign semantic drift or ineffective detection evasion. To address this challenge, we propose Reasoning-Augmented Conversation, a novel multi-turn jailbreak framework that reformulates harmful queries into benign reasoning tasks and leverages LLMs' strong reasoning capabilities to compromise safety alignment. Specifically, we introduce an attack state machine framework to systematically model problem translation and iterative reasoning, ensuring coherent query generation across multiple turns. Building on this framework, we design gain-guided exploration, self-play, and rejection feedback modules to preserve attack semantics, enhance effectiveness, and sustain reasoning-driven attack progression. Extensive experiments on multiple LLMs demonstrate that RACE achieves state-of-the-art attack effectiveness in complex conversational scenarios, with attack success rates (ASRs) increasing by up to 96%. Notably, our approach achieves ASRs of 82% and 92% against leading commercial models, OpenAI o1 and DeepSeek R1, underscoring its potency. We release our code at this https URL to facilitate further research in this critical domain. 

**Abstract (ZH)**: 多轮 Jailbreak 攻击通过在迭代对话中促使大型语言模型（LLMs）参与真实世界的互动，揭示关键的安全漏洞。为解决现有方法在语义连贯性与攻击有效性之间难以平衡的问题，我们提出了一种名为增强推理对话（Reasoning-Augmented Conversation, RACE）的新型多轮 Jailbreak 框架，该框架将有害查询重新表述为无害的推理任务，并利用 LLM 强大的推理能力破坏安全性对齐。具体而言，我们引入了一种攻击状态机框架，系统地建模问题转换和迭代推理，确保多轮查询生成的一致性。在此基础上，我们设计了收益导向的探索、自对弈和拒绝反馈模块，以保持攻击语义、增强有效性并维持推理驱动的攻击进展。在多个 LLM 上的广泛实验表明，RACE 在复杂的对话场景中实现了最先进的攻击有效性，攻击成功率（ASRs）最高提升 96%。值得注意的是，我们的方法在对抗领先商用模型 OpenAI o1 和 DeepSeek R1 时的 ASRs 分别达到 82% 和 92%，突显了其强大的效果。我们已在以下链接发布了我们的代码，以促进对该关键领域的进一步研究。 

---
# MMUNLEARNER: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models 

**Title (ZH)**: MMUNLEARNER: 在多模态大语言模型时代重述多模态机器遗忘方法 

**Authors**: Jiahao Huo, Yibo Yan, Xu Zheng, Yuanhuiyi Lyu, Xin Zou, Zhihua Wei, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11051)  

**Abstract**: Recent progress in Machine Unlearning (MU) has introduced solutions for the selective removal of private or sensitive information encoded within deep neural networks. Nonetheless, MU for Multimodal Large Language Models (MLLMs) remains in its nascent phase. Therefore, we propose to reformulate the task of multimodal MU in the era of MLLMs, which aims to erase only the visual patterns associated with a given entity while preserving the corresponding textual knowledge encoded within the original parameters of the language model backbone. Furthermore, we develop a novel geometry-constrained gradient descent method MMUnlearner. It updates the weights of MLLMs with a weight saliency map jointly restricted by the remaining concepts and textual knowledge during unlearning, thereby preserving parameters essential for non-target knowledge. Extensive experiments demonstrate that MMUnlearner surpasses baselines that finetuning MLLMs with VQA data directly through Gradient Ascent (GA) or Negative Preference Optimization (NPO), across all evaluation dimensions. Our code will be released upon acceptance. 

**Abstract (ZH)**: 近期，机器遗忘（Machine Unlearning, MU）领域在选择性移除深神经网络中编码的私人或敏感信息方面取得了进展。然而，针对多模态大型语言模型（Multimodal Large Language Models, MLLMs）的MU仍处于初级阶段。因此，我们提出在MLLM时代重新定义多模态MU的任务，目标是在遗忘过程中仅删除与给定实体相关的视觉模式，同时保留语言模型骨干中原有的相应文本知识。此外，我们开发了一种新的几何约束梯度下降方法MMUnlearner。该方法在遗忘过程中通过一个联合受限于剩余概念和文本知识的权重显著图来更新MLLMs的权重，从而保留对非目标知识至关重要的参数。大量实验表明，MMUnlearner在所有评估维度上均优于直接使用问答数据（VQA数据）通过梯度上升（Gradient Ascent, GA）或负偏好优化（Negative Preference Optimization, NPO）微调MLLMs的基线方法。关于我们代码，将在接收后公开。 

---
# Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs 

**Title (ZH)**: 基于vae循环排列的深层不完全多视图学习 

**Authors**: Xin Gao, Jian Pu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11037)  

**Abstract**: Multi-View Representation Learning (MVRL) aims to derive a unified representation from multi-view data by leveraging shared and complementary information across views. However, when views are irregularly missing, the incomplete data can lead to representations that lack sufficiency and consistency. To address this, we propose Multi-View Permutation of Variational Auto-Encoders (MVP), which excavates invariant relationships between views in incomplete data. MVP establishes inter-view correspondences in the latent space of Variational Auto-Encoders, enabling the inference of missing views and the aggregation of more sufficient information. To derive a valid Evidence Lower Bound (ELBO) for learning, we apply permutations to randomly reorder variables for cross-view generation and then partition them by views to maintain invariant meanings under permutations. Additionally, we enhance consistency by introducing an informational prior with cyclic permutations of posteriors, which turns the regularization term into a similarity measure across distributions. We demonstrate the effectiveness of our approach on seven diverse datasets with varying missing ratios, achieving superior performance in multi-view clustering and generation tasks. 

**Abstract (ZH)**: 多视图变分自编码器的多视图排列（MVP）：挖掘不完备数据中不变关系以提升表示学习 

---
# Mind the Confidence Gap: Overconfidence, Calibration, and Distractor Effects in Large Language Models 

**Title (ZH)**: 注意信心差距：大型语言模型中的过度自信、校准与干扰效应 

**Authors**: Prateek Chhikara  

**Link**: [PDF](https://arxiv.org/pdf/2502.11028)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive performance across diverse tasks, yet confidence calibration remains a challenge. Miscalibration - where models are overconfident or underconfident - poses risks, particularly in high-stakes applications. This paper presents an empirical study on LLM calibration, examining how model size, distractors, and question types affect confidence alignment. We introduce an evaluation framework to measure overconfidence and investigate whether multiple-choice formats mitigate or worsen miscalibration. Our findings show that while larger models (e.g., GPT-4o) are better calibrated overall, they are more prone to distraction, whereas smaller models benefit more from answer choices but struggle with uncertainty estimation. Unlike prior work, which primarily reports miscalibration trends, we provide actionable insights into failure modes and conditions that worsen overconfidence. These findings highlight the need for calibration-aware interventions and improved uncertainty estimation methods. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种任务上表现出色，但置信度校准仍然是一个挑战。误校准——即模型过于自信或不够自信——在高风险应用中尤其充满风险。本文通过对LLM校准的实证研究，探讨模型大小、干扰项和问题类型如何影响置信度对齐。我们提出了一种评估框架来衡量过度自信，并调查多项选择格式是否缓解或加剧了误校准。研究结果显示，虽然 Larger 模型（例如，GPT-4o）总体上更准确校准，但它们更容易受到干扰，而较小的模型虽然从答案选项中受益更多，但在不确定性估计方面遇到困难。不同于以往主要报告误校准趋势的研究，我们提供了有关失败模式和加剧过度自信的条件的可操作性见解。这些发现强调了需要采取校准意识干预并改进不确定性估计方法的重要性。 

---
# Simplify RLHF as Reward-Weighted SFT: A Variational Method 

**Title (ZH)**: 将RLHF简化为奖励加权SFT：一种变分方法 

**Authors**: Yuhao Du, Zhuo Li, Pengyu Cheng, Zhihong Chen, Yuejiao Xie, Xiang Wan, Anningzhe Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.11026)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is crucial for aligning Large Language Models (LLMs) with human values. However, RLHF has been continuously challenged by its high complexity in implementation and computation consumption. Even with recent simplifications, such as Direct Preference Optimization (DPO) and Advantage Leftover Lunch (A-LoL), the problems of over-fitting and training instability remain hindering the alignment process from the expected optimal performance. To address the existing challenges, we propose a novel simplification of RLHF from the perspective of variational inference, called $\textbf{V}$ariational $\textbf{A}$lignment with $\textbf{R}$e-weighting ($\textbf{VAR}$). More specifically, by directly minimizing the distribution gap between the learning LLM policy and the optimal solution of RLHF, we transform the alignment objective into a reward-driven re-weighted supervised fine-tuning (SFT) form, which only requires minor adjustment on the SFT loss to obtain noticeable improvement on training stability and effectiveness. On comprehensive alignment and generation benchmarks, our VAR method has numerically achieved competitive performance in LLM alignment helpfulness and harmlessness. 

**Abstract (ZH)**: 基于变分推断的Reinforcement Learning from Human Feedback ($\textbf{VAR}$)：改善大型语言模型对齐的新型简化方法 

---
# MultiTEND: A Multilingual Benchmark for Natural Language to NoSQL Query Translation 

**Title (ZH)**: 多语言基准：自然语言到NoSQL查询转换的MultTEND 

**Authors**: Zhiqian Qin, Yuanfeng Song, Jinwei Lu, Yuanwei Song, Shuaimin Li, Chen Jason Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11022)  

**Abstract**: Natural language interfaces for NoSQL databases are increasingly vital in the big data era, enabling users to interact with complex, unstructured data without deep technical expertise. However, most recent advancements focus on English, leaving a gap for multilingual support. This paper introduces MultiTEND, the first and largest multilingual benchmark for natural language to NoSQL query generation, covering six languages: English, German, French, Russian, Japanese and Mandarin Chinese. Using MultiTEND, we analyze challenges in translating natural language to NoSQL queries across diverse linguistic structures, including lexical and syntactic differences. Experiments show that performance accuracy in both English and non-English settings remains relatively low, with a 4%-6% gap across scenarios like fine-tuned SLM, zero-shot LLM, and RAG for LLM. To address the aforementioned challenges, we introduce MultiLink, a novel framework that bridges the multilingual input to NoSQL query generation gap through a Parallel Linking Process. It breaks down the task into multiple steps, integrating parallel multilingual processing, Chain-of-Thought (CoT) reasoning, and Retrieval-Augmented Generation (RAG) to tackle lexical and structural challenges inherent in multilingual NoSQL generation. MultiLink shows enhancements in all metrics for every language against the top baseline, boosting execution accuracy by about 15% for English and averaging a 10% improvement for non-English languages. 

**Abstract (ZH)**: 多语言自然语言接口多模态基准MultiTEND及其应用研究 

---
# TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages 

**Title (ZH)**: TUMLU：一个统一的土耳其语族语言理解基准 

**Authors**: Jafar Isbarov, Arofat Akhundjanova, Mammad Hajili, Kavsar Huseynova, Dmitry Gaynullin, Anar Rzayev, Osman Tursun, Ilshat Saetov, Rinat Kharisov, Saule Belginova, Ariana Kenbayeva, Amina Alisheva, Aizirek Turdubaeva, Abdullatif Köksal, Samir Rustamov, Duygu Ataman  

**Link**: [PDF](https://arxiv.org/pdf/2502.11020)  

**Abstract**: Being able to thoroughly assess massive multi-task language understanding (MMLU) capabilities is essential for advancing the applicability of multilingual language models. However, preparing such benchmarks in high quality native language is often costly and therefore limits the representativeness of evaluation datasets. While recent efforts focused on building more inclusive MMLU benchmarks, these are conventionally built using machine translation from high-resource languages, which may introduce errors and fail to account for the linguistic and cultural intricacies of the target languages. In this paper, we address the lack of native language MMLU benchmark especially in the under-represented Turkic language family with distinct morphosyntactic and cultural characteristics. We propose two benchmarks for Turkic language MMLU: TUMLU is a comprehensive, multilingual, and natively developed language understanding benchmark specifically designed for Turkic languages. It consists of middle- and high-school level questions spanning 11 academic subjects in Azerbaijani, Crimean Tatar, Karakalpak, Kazakh, Tatar, Turkish, Uyghur, and Uzbek. We also present TUMLU-mini, a more concise, balanced, and manually verified subset of the dataset. Using this dataset, we systematically evaluate a diverse range of open and proprietary multilingual large language models (LLMs), including Claude, Gemini, GPT, and LLaMA, offering an in-depth analysis of their performance across different languages, subjects, and alphabets. To promote further research and development in multilingual language understanding, we release TUMLU-mini and all corresponding evaluation scripts. 

**Abstract (ZH)**: 能够全面评估大规模多任务语言理解（MMLU）能力是推进多语言语言模型应用的重要前提。然而，准备高质量的母语基准往往成本高昂，因此限制了评估数据集的代表性。尽管近期努力构建更为包容的MMLU基准，但这些基准通常使用高资源语言的机器翻译构建，可能引入错误并未能充分考虑目标语言的语料和文化复杂性。在本文中，我们针对尤其是代表性不足的突厥语族语言缺乏母语MMLU基准的情况，提出两个突厥语族语言MMLU基准：TUMLU是一个全面的、多语言的、母语开发的语言理解基准，专门设计用于突厥语族语言，包含阿塞拜疆语、克里米亚塔塔尔语、卡拉卡尔帕克语、哈萨克语、塔塔尔语、土耳其语、维吾尔语和乌兹别克语的中学和高中水平问题，涵盖11个学术科目。我们还介绍了TUMLU-mini，这是数据集的一个更简洁、平衡且手工验证的子集。使用该数据集，我们系统地评估了多种开源和专有大规模多语言语言模型（LLMs），包括Claude、Gemini、GPT和LLaMA，提供了其在不同语言、科目和字母上的性能深入分析。为了促进多语言语言理解领域的进一步研究和发展，我们发布了TUMLU-mini及其相应的评估脚本。 

---
# Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning 

**Title (ZH)**: 解锁功能向量的 Powerful 应用以表征和缓解持续指令调谐中的灾难性遗忘 

**Authors**: Gangwei Jiang, Caigao Jiang, Zhaoyi Li, Siqiao Xue, Jun Zhou, Linqi Song, Defu Lian, Yin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.11019)  

**Abstract**: Catastrophic forgetting (CF) poses a significant challenge in machine learning, where a model forgets previously learned information upon learning new tasks. Despite the advanced capabilities of Large Language Models (LLMs), they continue to face challenges with CF during continual learning. The majority of existing research focuses on analyzing forgetting patterns through a singular training sequence, thereby overlooking the intricate effects that diverse tasks have on model behavior. Our study explores CF across various settings, discovering that model forgetting is influenced by both the specific training tasks and the models themselves. To this end, we interpret forgetting by examining the function vector (FV), a compact representation of functions in LLMs, offering a model-dependent indicator for the occurrence of CF. Through theoretical and empirical analyses, we demonstrated that CF in LLMs primarily stems from biases in function activation rather than the overwriting of task processing functions. Leveraging these insights, we propose a novel function vector guided training methodology, incorporating a regularization technique to stabilize the FV and mitigate forgetting. Empirical tests on four benchmarks confirm the effectiveness of our proposed training method, substantiating our theoretical framework concerning CF and model function dynamics. We plan to make our code publicly accessible in the near future. 

**Abstract (ZH)**: 灾难性遗忘（CF）对机器学习构成了重大挑战，其中模型在学习新任务时会忘记先前学习的信息。尽管大型语言模型（LLMs）具备先进的功能，但在持续学习过程中仍然面临CF的挑战。现有大多数研究集中在通过单一训练序列分析遗忘模式，而忽视了不同类型任务对模型行为的复杂影响。我们的研究在各种情景下探索CF，发现模型遗忘受特定训练任务和模型本身的影响。为此，我们通过分析功能向量（FV），一种LLM中函数的紧凑表示，提供了一个基于模型的CF发生的指示器。通过理论和实证分析，我们证明了LLMs中的CF主要源于功能激活的偏差而非任务处理函数的覆盖。利用这些见解，我们提出了一种新颖的功能向量引导训练方法，结合正则化技术以稳定FV并减轻遗忘。对四个基准的实证测试验证了我们提出的训练方法的有效性，证实了我们关于CF和模型功能动力学的理论框架。我们计划在未来不久公开我们的代码。 

---
# GRIFFIN: Effective Token Alignment for Faster Speculative Decoding 

**Title (ZH)**: GRIFFIN: 有效的token对齐以实现更快的推测解码 

**Authors**: Shijing Hu, Jingyang Li, Xingyu Xie, Zhihui Lu, Kim-Chuan Toh, Pan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.11018)  

**Abstract**: Speculative decoding accelerates inference in large language models (LLMs) by generating multiple draft tokens simultaneously. However, existing methods often struggle with token misalignment between the training and decoding phases, limiting their performance. To address this, we propose GRIFFIN, a novel framework that incorporates a token-alignable training strategy and a token-alignable draft model to mitigate misalignment. The training strategy employs a loss masking mechanism to exclude highly misaligned tokens during training, preventing them from negatively impacting the draft model's optimization. The token-alignable draft model introduces input tokens to correct inconsistencies in generated features. Experiments on LLaMA-series and Vicuna models demonstrate that GRIFFIN achieves an average acceptance length improvement of over 7\% and a speedup ratio exceeding 8%, outperforming current SoTAs as shown in Fig. 1 (a) and (b). 

**Abstract (ZH)**: 推测解码通过同时生成多个草稿令牌加速大型语言模型的推理。然而，现有方法往往在训练和解码阶段之间存在令牌对齐问题，限制了其性能。为解决这一问题，我们提出了一种名为GRIFFIN的新框架，该框架结合了可对齐的训练策略和可对齐的草稿模型，以减轻对齐问题。训练策略采用损失屏蔽机制，在训练过程中排除高度对齐错误的令牌，防止它们负面影响草稿模型的优化。可对齐的草稿模型通过引入输入令牌来纠正生成特征的一致性问题。实验表明，GRIFFIN在LLaMA系列和Vicuna模型上实现了超过7%的平均接受长度改进和超过8%的加速比，优于当前的SOTA方法，如图1(a)和(b)所示。 

---
# Collaborative Deterministic-Diffusion Model for Probabilistic Urban Spatiotemporal Prediction 

**Title (ZH)**: 协作确定性扩散模型及其在城市时空概率预测中的应用 

**Authors**: Zhi Sheng, Yuan Yuan, Yudi Zhang, Depeng Jin, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.11013)  

**Abstract**: Accurate prediction of urban spatiotemporal dynamics is essential for enhancing urban management and decision-making. Existing spatiotemporal prediction models are predominantly deterministic, focusing on primary spatiotemporal patterns. However, those dynamics are highly complex, exhibiting multi-modal distributions that are challenging for deterministic models to capture. In this paper, we highlight the critical role of probabilistic prediction in capturing the uncertainties and complexities inherent in spatiotemporal data. While mainstream probabilistic models can capture uncertainty, they struggle with accurately learning primary patterns and often suffer from computational inefficiency. To address these challenges, we propose CoST, which collaborates deterministic and probabilistic models to improve both predictive accuracy and the ability to handle uncertainty. To achieve this, we design a mean-residual decomposition framework, where the mean value is modeled by a deterministic model, and the residual variations are learned by a probabilistic model, specifically diffusion models. Moreover, we introduce a scale-aware diffusion process, which better accounts for spatially heterogeneous dynamics across different regions. Extensive experiments on eight real-world datasets demonstrate that CoST significantly outperforms existing methods in both deterministic and probabilistic metrics, achieving a 20% improvement with low computational cost. CoST bridges the gap between deterministic precision and probabilistic uncertainty, making a significant advancement in the field of urban spatiotemporal prediction. 

**Abstract (ZH)**: 准确预测城市时空动态对于改进城市管理与决策至关重要。现有的时空预测模型主要为确定性模型，专注于主要的时空模式。然而，这些动态极为复杂，表现出多模态分布，这给确定性模型带来了极大的挑战。在本文中，我们强调了概率预测在捕捉时空数据中固有的不确定性和复杂性的关键作用。虽然主流的概率模型能够捕捉不确定性，但它们在学习主要模式方面常常遇到困难，并且往往存在计算效率低的问题。为了应对这些挑战，我们提出CoST，该方法结合确定性和概率模型，旨在提高预测准确性和处理不确定性的能力。为此，我们设计了一种均值残差分解框架，其中均值由确定性模型建模，残差变异由概率模型（具体为扩散模型）学习。此外，我们引入了一种尺度感知扩散过程，更好地考虑了不同地区之间时空动态的异质性。在八个真实世界数据集上的广泛实验表明，CoST在确定性和概率性度量上均显著优于现有方法，在较低计算成本下实现了约20%的性能提升。CoST弥合了确定性精确度与概率不确定性之间的鸿沟，为城市时空预测领域带来了重要进展。 

---
# Prompt Inject Detection with Generative Explanation as an Investigative Tool 

**Title (ZH)**: 生成性解释作为调查工具的提示注入检测 

**Authors**: Jonathan Pan, Swee Liang Wong, Yidi Yuan, Xin Wei Chia  

**Link**: [PDF](https://arxiv.org/pdf/2502.11006)  

**Abstract**: Large Language Models (LLMs) are vulnerable to adversarial prompt based injects. These injects could jailbreak or exploit vulnerabilities within these models with explicit prompt requests leading to undesired responses. In the context of investigating prompt injects, the challenge is the sheer volume of input prompts involved that are likely to be largely benign. This investigative challenge is further complicated by the semantics and subjectivity of the input prompts involved in the LLM conversation with its user and the context of the environment to which the conversation is being carried out. Hence, the challenge for AI security investigators would be two-fold. The first is to identify adversarial prompt injects and then to assess whether the input prompt is contextually benign or adversarial. For the first step, this could be done using existing AI security solutions like guardrails to detect and protect the LLMs. Guardrails have been developed using a variety of approaches. A popular approach is to use signature based. Another popular approach to develop AI models to classify such prompts include the use of NLP based models like a language model. However, in the context of conducting an AI security investigation of prompt injects, these guardrails lack the ability to aid investigators in triaging or assessing the identified input prompts. In this applied research exploration, we explore the use of a text generation capabilities of LLM to detect prompt injects and generate explanation for its detections to aid AI security investigators in assessing and triaging of such prompt inject detections. The practical benefit of such a tool is to ease the task of conducting investigation into prompt injects. 

**Abstract (ZH)**: 大型语言模型（LLMs）易受基于恶意提示的注入攻击。这些注入攻击可能使模型脱域或利用模型中的漏洞，导致不期望的响应。在调查提示注入时，挑战在于涉及的大量输入提示可能大多是 benign 的。这一调查挑战在涉及 LLM 与其用户之间的对话以及对话进行的环境上下文中，由于输入提示的语义和主观性而变得更加复杂。因此，AI 安全调查人员面临的挑战是双重的。首先，需要识别恶意提示注入；然后，评估输入提示是否处于上下文中为 benign 或恶意。在这一步骤中，现有的 AI 安全解决方案，如 guardrails，可以用来检测和保护 LLM。guardrails 的开发采用了多种方法。一种流行的方法是基于签名。另一种流行的方法是使用基于 NLP 的模型，如语言模型，来分类提示。然而，在进行 AI 安全调查以检测提示注入时，这些 guardrails 缺乏帮助调查人员初步评估或处理已识别输入提示的能力。在此应用研究探索中，我们研究了利用 LLM 的文本生成能力来检测提示注入并生成其检测的解释，以帮助 AI 安全调查人员评估和处理这类提示注入检测。此类工具的实际好处在于简化对提示注入的调查任务。 

---
# CL-MFAP: A Contrastive Learning-Based Multimodal Foundation Model for Molecular Property Prediction and Antibiotic Screening 

**Title (ZH)**: 基于对比学习的多模态基础模型：CL-MFAP在分子性质预测和抗生素筛选中的应用 

**Authors**: Gen Zhou, Sugitha Janarthanan, Yutong Lu, Pingzhao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11001)  

**Abstract**: Due to the rise in antimicrobial resistance, identifying novel compounds with antibiotic potential is crucial for combatting this global health issue. However, traditional drug development methods are costly and inefficient. Recognizing the pressing need for more effective solutions, researchers have turned to machine learning techniques to streamline the prediction and development of novel antibiotic compounds. While foundation models have shown promise in antibiotic discovery, current mainstream efforts still fall short of fully leveraging the potential of multimodal molecular data. Recent studies suggest that contrastive learning frameworks utilizing multimodal data exhibit excellent performance in representation learning across various domains. Building upon this, we introduce CL-MFAP, an unsupervised contrastive learning (CL)-based multimodal foundation (MF) model specifically tailored for discovering small molecules with potential antibiotic properties (AP) using three types of molecular data. This model employs 1.6 million bioactive molecules with drug-like properties from the ChEMBL dataset to jointly pretrain three encoders: (1) a transformer-based encoder with rotary position embedding for processing SMILES strings; (2) another transformer-based encoder, incorporating a novel bi-level routing attention mechanism to handle molecular graph representations; and (3) a Morgan fingerprint encoder using a multilayer perceptron, to achieve the contrastive learning purpose. The CL-MFAP outperforms baseline models in antibiotic property prediction by effectively utilizing different molecular modalities and demonstrates superior domain-specific performance when fine-tuned for antibiotic-related property prediction tasks. 

**Abstract (ZH)**: 由于抗生素耐药性的上升，识别具有抗菌潜力的新化合物对于应对这一全球健康问题至关重要。然而，传统的药物开发方法成本高且效率低。鉴于更有效解决方案的迫切需要，研究人员转向机器学习技术以简化新型抗菌化合物的预测和开发。尽管基础模型在抗生素发现方面显示出潜力，但当前主流努力尚未充分利用跨模态分子数据的潜力。最新研究表明，利用跨模态数据的对比学习框架在各类领域表现出色。在此基础上，我们引入了CL-MFAP，这是一种基于无监督对比学习（CL）的跨模态基础（MF）模型，专门用于使用三种类型的分子数据发现具有潜在抗菌特性（AP）的小分子。该模型利用来自ChEMBL数据集的160万种具有药理特性的生物活性分子，联合预训练三个编码器：（1）带有旋转位置嵌入的变压器编码器，用于处理SMILES字符串；（2）另一种变压器编码器，结合了一种新颖的双层路由注意力机制，以处理分子图表示；以及（3）使用多层感知器的Morgan指纹编码器，以实现对比学习目的。CL-MFAP 在抗菌特性预测方面优于基线模型，有效地利用了不同的分子模态，并在针对抗菌相关特性预测任务进行微调时表现出更出色的领域特定性能。 

---
# ControlText: Unlocking Controllable Fonts in Multilingual Text Rendering without Font Annotations 

**Title (ZH)**: ControlText: 在无字体标注的情况下解锁多语言文本渲染的可控字体 

**Authors**: Bowen Jiang, Yuan Yuan, Xinyi Bai, Zhuoqun Hao, Alyson Yin, Yaojie Hu, Wenyu Liao, Lyle Ungar, Camillo J. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2502.10999)  

**Abstract**: This work demonstrates that diffusion models can achieve font-controllable multilingual text rendering using just raw images without font label annotations. Visual text rendering remains a significant challenge. While recent methods condition diffusion on glyphs, it is impossible to retrieve exact font annotations from large-scale, real-world datasets, which prevents user-specified font control. To address this, we propose a data-driven solution that integrates the conditional diffusion model with a text segmentation model, utilizing segmentation masks to capture and represent fonts in pixel space in a self-supervised manner, thereby eliminating the need for any ground-truth labels and enabling users to customize text rendering with any multilingual font of their choice. The experiment provides a proof of concept of our algorithm in zero-shot text and font editing across diverse fonts and languages, providing valuable insights for the community and industry toward achieving generalized visual text rendering. 

**Abstract (ZH)**: 本研究展示了扩散模型可以在无需字体标签注释的情况下，仅使用原始图像实现可控多语言文本渲染。视觉文本渲染仍然是一个重大挑战。尽管最近的方法将扩散模型条件化于字形，但从大规模真实世界数据集中获取精确的字体注释是不可能的，这阻碍了用户指定的字体控制。为解决这一问题，我们提出了一种数据驱动的解决方案，将条件扩散模型与文本分割模型相结合，利用分割掩码在自监督的方式下捕捉和表示字体在像素空间中的信息，从而消除对任何ground-truth标签的需求，使用户能够使用任意选择的多语言字体定制文本渲染。实验提供了在多样字体和语言中零样本文本和字体编辑的算法概念证明，为社区和行业实现通用视觉文本渲染提供了有价值的见解。 

---
# Is Elo Rating Reliable? A Study Under Model Misspecification 

**Title (ZH)**: Elo评分可靠吗？基于模型误设的研究 

**Authors**: Shange Tang, Yuanhao Wang, Chi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2502.10985)  

**Abstract**: Elo rating, widely used for skill assessment across diverse domains ranging from competitive games to large language models, is often understood as an incremental update algorithm for estimating a stationary Bradley-Terry (BT) model. However, our empirical analysis of practical matching datasets reveals two surprising findings: (1) Most games deviate significantly from the assumptions of the BT model and stationarity, raising questions on the reliability of Elo. (2) Despite these deviations, Elo frequently outperforms more complex rating systems, such as mElo and pairwise models, which are specifically designed to account for non-BT components in the data, particularly in terms of win rate prediction. This paper explains this unexpected phenomenon through three key perspectives: (a) We reinterpret Elo as an instance of online gradient descent, which provides no-regret guarantees even in misspecified and non-stationary settings. (b) Through extensive synthetic experiments on data generated from transitive but non-BT models, such as strongly or weakly stochastic transitive models, we show that the ''sparsity'' of practical matching data is a critical factor behind Elo's superior performance in prediction compared to more complex rating systems. (c) We observe a strong correlation between Elo's predictive accuracy and its ranking performance, further supporting its effectiveness in ranking. 

**Abstract (ZH)**: Elo评级：基于在线梯度下降的稀疏性视角及预测性能解释 

---
# QuOTE: Question-Oriented Text Embeddings 

**Title (ZH)**: 面向问题的文本嵌入 

**Authors**: Andrew Neeser, Kaylen Latimer, Aadyant Khatri, Chris Latimer, Naren Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10976)  

**Abstract**: We present QuOTE (Question-Oriented Text Embeddings), a novel enhancement to retrieval-augmented generation (RAG) systems, aimed at improving document representation for accurate and nuanced retrieval. Unlike traditional RAG pipelines, which rely on embedding raw text chunks, QuOTE augments chunks with hypothetical questions that the chunk can potentially answer, enriching the representation space. This better aligns document embeddings with user query semantics, and helps address issues such as ambiguity and context-dependent relevance. Through extensive experiments across diverse benchmarks, we demonstrate that QuOTE significantly enhances retrieval accuracy, including in multi-hop question-answering tasks. Our findings highlight the versatility of question generation as a fundamental indexing strategy, opening new avenues for integrating question generation into retrieval-based AI pipelines. 

**Abstract (ZH)**: QuOTE（问题导向的文本嵌入）：检索增强生成系统的新型增强方法，旨在提高文档表示以实现精确和细腻的检索 

---
# Neural Networks Remember More: The Power of Parameter Isolation and Combination 

**Title (ZH)**: 神经网络记得更多：参数隔离与组合的力量 

**Authors**: Biqing Zeng, Zehan Li, Aladdin Ayesh  

**Link**: [PDF](https://arxiv.org/pdf/2502.10966)  

**Abstract**: Catastrophic forgetting is a pervasive issue for pre-trained language models (PLMs) during continual learning, where models lose previously acquired knowledge when sequentially trained on a series of tasks. The model's ability to retain old tasks is referred to as stability, while its adaptability to new tasks is called plasticity. Therefore, the key to solving this problem is to find a trade-off between the plasticity and stability of the model. To address this issue, in this paper, we propose a novel method to achieve a balance between model stability and plasticity, thereby mitigating catastrophic forgetting. More specifically, our proposed approach leverages parameter isolation and a subsequent combination strategy. Initially, in the training stage, the model adapts to each downstream task via a parameter isolation method to prevent potential interference among different tasks. We then combine all trained parameters, which contain acquired knowledge, using the task arithmetic method and finally apply them to the backbone model. Empirical evaluations on continual language learning benchmarks substantiate the effectiveness of our approach, revealing a marked enhancement over existing state-of-the-art approaches. 

**Abstract (ZH)**: 持续学习中预训练语言模型的灾难性遗忘问题及其解决方案：通过参数隔离和组合策略实现模型稳定性和可塑性的平衡 

---
# Graders should cheat: privileged information enables expert-level automated evaluations 

**Title (ZH)**: 评分员应该作弊：特权信息使自动评价达到专家水平 

**Authors**: Jin Peng Zhou, Sébastien M. R. Arnold, Nan Ding, Kilian Q. Weinberger, Nan Hua, Fei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2502.10961)  

**Abstract**: Auto-evaluating language models (LMs), i.e., using a grader LM to evaluate the candidate LM, is an appealing way to accelerate the evaluation process and the cost associated with it. But this presents a paradox: how can we trust the grader LM, which is presumably weaker than the candidate LM, to assess problems that are beyond the frontier of the capabilities of either model or both? For instance, today's LMs struggle on graduate-level physics and Olympiad-level math, making them unreliable graders in these domains.
We show that providing privileged information -- such as ground-truth solutions or problem-specific guidelines -- improves automated evaluations on such frontier problems. This approach offers two key advantages. First, it expands the range of problems where LMs graders apply. Specifically, weaker models can now rate the predictions of stronger models. Second, privileged information can be used to devise easier variations of challenging problems which improves the separability of different LMs on tasks where their performance is generally low. With this approach, general-purpose LM graders match the state of the art performance on RewardBench, surpassing almost all the specially-tuned models. LM graders also outperform individual human raters on Vibe-Eval, and approach human expert graders on Olympiad-level math problems. 

**Abstract (ZH)**: 自动评估语言模型（LMs），即使用一个评判LM来评估候选LM，是一种加快评估过程及其相关成本的方法。但这也 presents 了一个悖论：我们如何能信任一个本身就可能较弱的评判LM来评估超出模型能力范围的问题？例如，当前的LM在graduate-level物理和Olympiad-level数学方面表现挣扎，使它们在这两个领域不可靠的评判者。
我们显示，提供特权信息——如ground-truth解决方案或问题特定指南——可以改善在这些前沿问题上的自动化评估。这种方法有两个关键优势。首先，它扩展了LM评判者可以应用的问题范围，尤其是较弱的模型现在可以评价较强模型的预测。其次，特权信息可以用来设计更具挑战性问题的简化版本，从而提高不同LM在它们表现普遍较低的任务上的可区分性。通过这种方法，通用LM评判者在RewardBench上达到了最先进的性能，超过了几乎所有专门调优的模型。LM评判者也在Vibe-Eval上优于单个的人类评判者，并且在Olympiad-level数学问题上接近人类专家评判者。 

---
# A recurrent vision transformer shows signatures of primate visual attention 

**Title (ZH)**: 一种循环视觉变换器展示了灵长类视觉注意的特征 

**Authors**: Jonathan Morgan, Badr Albanna, James P. Herman  

**Link**: [PDF](https://arxiv.org/pdf/2502.10955)  

**Abstract**: Attention is fundamental to both biological and artificial intelligence, yet research on animal attention and AI self attention remains largely disconnected. We propose a Recurrent Vision Transformer (Recurrent ViT) that integrates self-attention with recurrent memory, allowing both current inputs and stored information to guide attention allocation. Trained solely via sparse reward feedback on a spatially cued orientation change detection task, a paradigm used in primate studies, our model exhibits primate like signatures of attention, including improved accuracy and faster responses for cued stimuli that scale with cue validity. Analysis of self-attention maps reveals dynamic spatial prioritization with reactivation prior to expected changes, and targeted perturbations produce performance shifts similar to those observed in primate frontal eye fields and superior colliculus. These findings demonstrate that incorporating recurrent feedback into self attention can capture key aspects of primate visual attention. 

**Abstract (ZH)**: 将注意力机制融入递归记忆的视觉变换器：捕捉灵长类视觉注意力的关键方面 

---
# Learning to Stop Overthinking at Test Time 

**Title (ZH)**: 测试时学习停止过度思考 

**Authors**: Hieu Tran Bao, Nguyen Cong Dat, Nguyen Duc Anh, Hoang Thanh Tung  

**Link**: [PDF](https://arxiv.org/pdf/2502.10954)  

**Abstract**: Test time scaling is currently one of the most active research areas that shows promise after training time scaling has reached its limits. Deep-thinking (DT) models are a class of recurrent models that can perform easy-to-hard generalization by assigning more compute to harder test samples. However, due to their inability to determine the complexity of a test sample, DT models have to use a large amount of computation for both easy and hard test samples. Excessive test time computation is wasteful and can cause the ``overthinking'' problem where more test time computation leads to worse results. In this paper, we introduce a test time training method for determining the optimal amount of computation needed for each sample during test time. We also propose Conv-LiGRU, a novel recurrent architecture for efficient and robust visual reasoning. Extensive experiments demonstrate that Conv-LiGRU is more stable than DT, effectively mitigates the ``overthinking'' phenomenon, and achieves superior accuracy. 

**Abstract (ZH)**: 测试时计算量缩放目前是除训练时计算量缩放达到极限后最具潜力的研究领域之一。深度思考（DT）模型是一类递归模型，可以通过为更难的测试样本分配更多的计算资源来实现从易到难的任务泛化。然而，由于无法确定测试样本的复杂度，DT模型在处理易和难的测试样本时都需要大量的计算资源。过度的测试时计算资源浪费，并可能导致“过度思考”问题，即更多的计算资源反而会使结果变差。在本文中，我们提出了一种测试时训练方法，用于确定测试时为每个样本所需的最佳计算量。我们还提出了一种新颖的递归架构Conv-LiGRU，用于高效且稳健的视觉推理。广泛实验表明，Conv-LiGRU 比 DT 更稳定，有效缓解了“过度思考”现象，并获得了更高的准确率。 

---
# Empirical evaluation of LLMs in predicting fixes of Configuration bugs in Smart Home System 

**Title (ZH)**: 基于配置错误修复预测的大型语言模型实证评估 

**Authors**: Sheikh Moonwara Anjum Monisha, Atul Bharadwaj  

**Link**: [PDF](https://arxiv.org/pdf/2502.10953)  

**Abstract**: This empirical study evaluates the effectiveness of Large Language Models (LLMs) in predicting fixes for configuration bugs in smart home systems. The research analyzes three prominent LLMs - GPT-4, GPT-4o (GPT-4 Turbo), and Claude 3.5 Sonnet - using four distinct prompt designs to assess their ability to identify appropriate fix strategies and generate correct solutions. The study utilized a dataset of 129 debugging issues from the Home Assistant Community, focusing on 21 randomly selected cases for in-depth analysis. Results demonstrate that GPT-4 and Claude 3.5 Sonnet achieved 80\% accuracy in strategy prediction when provided with both bug descriptions and original scripts. GPT-4 exhibited consistent performance across different prompt types, while GPT-4o showed advantages in speed and cost-effectiveness despite slightly lower accuracy. The findings reveal that prompt design significantly impacts model performance, with comprehensive prompts containing both description and original script yielding the best results. This research provides valuable insights for improving automated bug fixing in smart home system configurations and demonstrates the potential of LLMs in addressing configuration-related challenges. 

**Abstract (ZH)**: 本实证研究评估了大型语言模型（LLM）在预测智能家居系统配置错误修复方面的有效性。研究使用四种不同的提示设计对三种 promin 锐 著 LLM——GPT-4、GPT-4o（GPT-4 Turbo）和Claude 3.5 Sonnet——进行了分析，以评估它们识别适当修复策略和生成正确解决方案的能力。研究利用了来自Home Assistant Community的129个调试问题数据集，专注于21个随机选择的案例进行深入分析。结果表明，当提供错误描述和原始脚本时，GPT-4和Claude 3.5 Sonnet在策略预测方面的准确率达到80%。GPT-4在不同提示类型中表现出一致的性能，而GPT-4o在速度和成本效益方面表现出优势，尽管准确性略低。研究发现，提示设计显著影响模型性能，包含描述和原始脚本的全面提示表现最佳。本研究为提高智能家居系统配置的自动化修复提供了宝贵的见解，并展示了LLM在解决配置相关挑战方面的潜在价值。 

---
# CoLA: Compute-Efficient Pre-Training of LLMs via Low-Rank Activation 

**Title (ZH)**: CoLA：通过低秩激活实现的LLMs计算高效预训练 

**Authors**: Ziyue Liu, Ruijie Zhang, Zhengyang Wang, Zi Yang, Paul Hovland, Bogdan Nicolae, Franck Cappello, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10940)  

**Abstract**: Large language models (LLMs) are revolutionizing many science and engineering fields. However, their huge model sizes impose extremely demanding needs of computational resources in the pre-training stage. Although low-rank factorizations can reduce model parameters, their direct application in LLM pre-training often lead to non-negligible performance loss. To address this fundamental challenge, we introduce CoLA and its memory-efficient implementation, CoLA-M. We leverage the low-rank structure observed widely in model activations, enforcing non-linear transformations between factorized weight matrices to reduce model size, boost model capacity and training efficiency. Experiments on LLaMA models with 60 million to 7 billion parameters show that CoLA reduces the computing cost by $\bf 2\pmb{\times}$ and improves training throughput by $\bf 1.86\pmb{\times}$ while maintaining full-rank level performance. CoLA-M further squeezes memory cost without sacrificing throughput, offering a pre-training approach with collectively superior parameter, computing, and memory efficiency. The LLMs produced are also $\bf 2\pmb{\times}$ smaller, enabling faster inference with lower memory cost on resource-constrained platforms 

**Abstract (ZH)**: 大规模语言模型（LLMs）正在 revolutionizing 许多科学和工程领域。然而，其庞大的模型规模在预训练阶段对计算资源提出了极其严峻的需求。尽管低秩因子分解可以减少模型参数，但在LLM预训练中的直接应用往往会导致不可忽视的性能损失。为解决这一根本挑战，我们介绍了CoLA及其内存高效的实现CoLA-M。我们利用模型激活中广泛观察到的低秩结构，对因子化权重矩阵施加非线性变换，以减小模型规模、提升模型容量和训练效率。实验表明，CoLA在6000万至7亿参数的LLaMA模型上将计算成本降低至$\bf 2\pmb{\times}$，训练吞吐量提升至$\bf 1.86\pmb{\times}$，同时保持全秩水平的性能。CoLA-M进一步压缩了内存成本而不牺牲吞吐量，提供了一种在参数、计算和内存效率方面综合性能更优的预训练方法。生成的LLMs也减小了$\bf 2\pmb{\times}$，使其在资源受限平台上实现更快推断并降低内存成本。 

---
# Semantic Specialization in MoE Appears with Scale: A Study of DeepSeek R1 Expert Specialization 

**Title (ZH)**: MoE中的语义专业化随着规模而出现：DeepSeek R1专家专业化研究 

**Authors**: Matthew Lyle Olson, Neale Ratzlaff, Musashi Hinck, Man Luo, Sungduk Yu, Chendi Xue, Vasudev Lal  

**Link**: [PDF](https://arxiv.org/pdf/2502.10928)  

**Abstract**: DeepSeek-R1, the largest open-source Mixture-of-Experts (MoE) model, has demonstrated reasoning capabilities comparable to proprietary frontier models. Prior research has explored expert routing in MoE models, but findings suggest that expert selection is often token-dependent rather than semantically driven. Given DeepSeek-R1's enhanced reasoning abilities, we investigate whether its routing mechanism exhibits greater semantic specialization than previous MoE models. To explore this, we conduct two key experiments: (1) a word sense disambiguation task, where we examine expert activation patterns for words with differing senses, and (2) a cognitive reasoning analysis, where we assess DeepSeek-R1's structured thought process in an interactive task setting of DiscoveryWorld. We conclude that DeepSeek-R1's routing mechanism is more semantically aware and it engages in structured cognitive processes. 

**Abstract (ZH)**: DeepSeek-R1，最大的开源Mixture-of-Experts (MoE)模型，展示了与 proprietary 前沿模型相当的推理能力。前期研究探讨了MoE模型中的专家路由机制，发现专家选择往往是基于 token 而非语义驱动的。鉴于DeepSeek-R1增强了推理能力，我们研究其路由机制是否比之前的MoE模型更具语义专一性。为此，我们进行了两项关键实验：(1) 词义消歧实验，分析不同词义词的专家激活模式；(2) 认知推理分析，评估DeepSeek-R1在DiscoveryWorld交互任务中的结构化思维过程。我们得出结论，DeepSeek-R1的路由机制更具有语义意识，并且参与了结构化的认知过程。 

---
# Do Deepfake Detectors Work in Reality? 

**Title (ZH)**: 深度假面检测器在现实中有用吗？ 

**Authors**: Simiao Ren, Hengwei Xu, Tsang Ng, Kidus Zewde, Shengkai Jiang, Ramini Desai, Disha Patil, Ning-Yau Cheng, Yining Zhou, Ragavi Muthukrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10920)  

**Abstract**: Deepfakes, particularly those involving faceswap-based manipulations, have sparked significant societal concern due to their increasing realism and potential for misuse. Despite rapid advancements in generative models, detection methods have not kept pace, creating a critical gap in defense strategies. This disparity is further amplified by the disconnect between academic research and real-world applications, which often prioritize different objectives and evaluation criteria. In this study, we take a pivotal step toward bridging this gap by presenting a novel observation: the post-processing step of super-resolution, commonly employed in real-world scenarios, substantially undermines the effectiveness of existing deepfake detection methods. To substantiate this claim, we introduce and publish the first real-world faceswap dataset, collected from popular online faceswap platforms. We then qualitatively evaluate the performance of state-of-the-art deepfake detectors on real-world deepfakes, revealing that their accuracy approaches the level of random guessing. Furthermore, we quantitatively demonstrate the significant performance degradation caused by common post-processing techniques. By addressing this overlooked challenge, our study underscores a critical avenue for enhancing the robustness and practical applicability of deepfake detection methods in real-world settings. 

**Abstract (ZH)**: Deepfake检测方法在现实场景中的有效性受到常用超分辨率后处理步骤的严重削弱：一项现实世界数据驱动的研究 

---
# Automatic Quality Assessment of First Trimester Crown-Rump-Length Ultrasound Images 

**Title (ZH)**: 自动评估早期妊娠 Crown-Rump-Length 超声图像的质量 

**Authors**: Sevim Cengiz, Ibraheem Hamdi, Mohammad Yaqub  

**Link**: [PDF](https://arxiv.org/pdf/2502.10908)  

**Abstract**: Fetal gestational age (GA) is vital clinical information that is estimated during pregnancy in order to assess fetal growth. This is usually performed by measuring the crown-rump-length (CRL) on an ultrasound image in the Dating scan which is then correlated with fetal age and growth trajectory. A major issue when performing the CRL measurement is ensuring that the image is acquired at the correct view, otherwise it could be misleading. Although clinical guidelines specify the criteria for the correct CRL view, sonographers may not regularly adhere to such rules. In this paper, we propose a new deep learning-based solution that is able to verify the adherence of a CRL image to clinical guidelines in order to assess image quality and facilitate accurate estimation of GA. We first segment out important fetal structures then use the localized structures to perform a clinically-guided mapping that verifies the adherence of criteria. The segmentation method combines the benefits of Convolutional Neural Network (CNN) and the Vision Transformer (ViT) to segment fetal structures in ultrasound images and localize important fetal landmarks. For segmentation purposes, we compare our proposed work with UNet and show that our CNN/ViT-based method outperforms an optimized version of UNet. Furthermore, we compare the output of the mapping with classification CNNs when assessing the clinical criteria and the overall acceptability of CRL images. We show that the proposed mapping is not only explainable but also more accurate than the best performing classification CNNs. 

**Abstract (ZH)**: 胎儿妊娠龄（GA）在妊娠期间通过评估胎儿生长是至关重要的临床信息。通常通过在孕龄扫描中测量头臀长（CRL）并在超声图像上将其与胎儿年龄和生长轨迹相关联来进行。CRL测量时的主要问题是确保图像是从正确的视角获取的，否则可能会误导结果。尽管临床指南规定了正确的CRL视角的准则，但超声技师可能不会定期遵守这些规定。在本文中，我们提出了一种新的基于深度学习的解决方案，能够验证CRL图像是否符合临床指南，以评估图像质量并促进妊娠龄的准确估计。我们首先对胎儿的重要结构进行分割，然后使用这些结构进行符合临床指导的映射，以验证准则的遵守情况。分割方法结合了卷积神经网络（CNN）和视觉变压器（ViT）的优势，用于在超声图像中分割胎儿结构并定位重要胎儿标志点。为了分割目的，我们将我们提出的方案与UNet进行比较，表明我们基于CNN/ViT的方法优于优化后的UNet。此外，我们在评估临床准则和CRL图像整体可接受性时，将映射输出与分类CNN进行比较。我们展示，所提出的映射不仅具有可解释性，而且还比表现最佳的分类CNN更准确。 

---
# Breaking Down the Hierarchy: A New Approach to Leukemia Classification 

**Title (ZH)**: 打破层级界限：一种新的白血病分类方法 

**Authors**: Ibraheem Hamdi, Hosam El-Gendy, Ahmed Sharshar, Mohamed Saeed, Muhammad Ridzuan, Shahrukh K. Hashmi, Naveed Syed, Imran Mirza, Shakir Hussain, Amira Mahmoud Abdalla, Mohammad Yaqub  

**Link**: [PDF](https://arxiv.org/pdf/2502.10899)  

**Abstract**: The complexities inherent to leukemia, multifaceted cancer affecting white blood cells, pose considerable diagnostic and treatment challenges, primarily due to reliance on laborious morphological analyses and expert judgment that are susceptible to errors. Addressing these challenges, this study presents a refined, comprehensive strategy leveraging advanced deep-learning techniques for the classification of leukemia subtypes. We commence by developing a hierarchical label taxonomy, paving the way for differentiating between various subtypes of leukemia. The research further introduces a novel hierarchical approach inspired by clinical procedures capable of accurately classifying diverse types of leukemia alongside reactive and healthy cells. An integral part of this study involves a meticulous examination of the performance of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) as classifiers. The proposed method exhibits an impressive success rate, achieving approximately 90\% accuracy across all leukemia subtypes, as substantiated by our experimental results. A visual representation of the experimental findings is provided to enhance the model's explainability and aid in understanding the classification process. 

**Abstract (ZH)**: 白血病的复杂性，一种影响白血球的多面性癌症，由于依赖于耗时的形态学分析和易出错的专家判断，给诊断和治疗带来了重大挑战。为应对这些挑战，本研究提出了一种结合先进深度学习技术的细化综合策略，用于白血病亚型分类。我们首先开发了一种分层标签分类法，以便区分各种白血病亚型。研究还引入了一种受临床程序启发的分层方法，能够准确分类不同类型的白血病以及反应性和健康细胞。本研究的一个重要部分是对卷积神经网络（CNNs）和视觉变换器（ViTs）作为分类器的性能进行了细致分析。所提出的方法表现出色，实验结果证明其在所有白血病亚型上的准确率约为90%。还提供了实验结果的可视化表示，以增强模型的解释性和帮助理解分类过程。 

---
# Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation 

**Title (ZH)**: Sim-to-Real过渡在体育运动操控中的应用 

**Authors**: Nolan Fey, Gabriel B. Margolis, Martin Peticco, Pulkit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2502.10894)  

**Abstract**: Achieving athletic loco-manipulation on robots requires moving beyond traditional tracking rewards - which simply guide the robot along a reference trajectory - to task rewards that drive truly dynamic, goal-oriented behaviors. Commands such as "throw the ball as far as you can" or "lift the weight as quickly as possible" compel the robot to exhibit the agility and power inherent in athletic performance. However, training solely with task rewards introduces two major challenges: these rewards are prone to exploitation (reward hacking), and the exploration process can lack sufficient direction. To address these issues, we propose a two-stage training pipeline. First, we introduce the Unsupervised Actuator Net (UAN), which leverages real-world data to bridge the sim-to-real gap for complex actuation mechanisms without requiring access to torque sensing. UAN mitigates reward hacking by ensuring that the learned behaviors remain robust and transferable. Second, we use a pre-training and fine-tuning strategy that leverages reference trajectories as initial hints to guide exploration. With these innovations, our robot athlete learns to lift, throw, and drag with remarkable fidelity from simulation to reality. 

**Abstract (ZH)**: 实现运动型机器人操控要求超越传统的跟踪奖励，转向驱动真正动态和目标导向行为的任务奖励。诸如“尽可能远地抛球”或“尽可能快速地举起重量”之类的命令促使机器人展现出类似运动员的敏捷性和力量。然而，仅使用任务奖励进行训练会引入两个主要挑战：这些奖励容易被操纵，且探索过程可能缺乏足够的方向。为了解决这些问题，我们提出了一种两阶段的训练管道。首先，我们引入了无监督执行网络（UAN），它利用现实世界数据来弥补复杂执行机制从仿真到现实的差距，无需扭矩感知。UAN通过确保学习到的行为保持稳健性和可-transferability来减少奖励操纵。其次，我们采用了一种预训练和微调策略，利用参考轨迹作为初始提示来引导探索。通过这些创新，我们的机器人运动员能够从仿真到现实以惊人的精度学会举举重、投掷和拖拽。 

---
# Learning Identifiable Structures Helps Avoid Bias in DNN-based Supervised Causal Learning 

**Title (ZH)**: 基于DNN的监督因果学习中可识别结构的学习有助于避免偏差 

**Authors**: Jiaru Zhang, Rui Ding, Qiang Fu, Bojun Huang, Zizhen Deng, Yang Hua, Haibing Guan, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10883)  

**Abstract**: Causal discovery is a structured prediction task that aims to predict causal relations among variables based on their data samples. Supervised Causal Learning (SCL) is an emerging paradigm in this field. Existing Deep Neural Network (DNN)-based methods commonly adopt the "Node-Edge approach", in which the model first computes an embedding vector for each variable-node, then uses these variable-wise representations to concurrently and independently predict for each directed causal-edge. In this paper, we first show that this architecture has some systematic bias that cannot be mitigated regardless of model size and data size. We then propose SiCL, a DNN-based SCL method that predicts a skeleton matrix together with a v-tensor (a third-order tensor representing the v-structures). According to the Markov Equivalence Class (MEC) theory, both the skeleton and the v-structures are identifiable causal structures under the canonical MEC setting, so predictions about skeleton and v-structures do not suffer from the identifiability limit in causal discovery, thus SiCL can avoid the systematic bias in Node-Edge architecture, and enable consistent estimators for causal discovery. Moreover, SiCL is also equipped with a specially designed pairwise encoder module with a unidirectional attention layer to model both internal and external relationships of pairs of nodes. Experimental results on both synthetic and real-world benchmarks show that SiCL significantly outperforms other DNN-based SCL approaches. 

**Abstract (ZH)**: 因果发现是一种结构化预测任务，旨在基于变量的数据样本预测变量之间的因果关系。监督因果学习（SCL）是该领域的新兴范式。现有的基于深度神经网络（DNN）的方法通常采用“节点-边”方法，在这种方法中，模型首先为每个变量节点计算一个嵌入向量，然后使用这些变量智慧的表示来独立预测每个有向因果边。在本文中，我们首先表明，这种架构存在一些系统偏差，无论模型规模和数据规模如何都无法缓解。然后，我们提出了一个基于DNN的SCL方法SiCL，它同时预测一个骨架矩阵和一个v-张量（表示v-结构的三阶张量）。根据马尔可夫等价类（MEC）理论，在标准的MEC设置下，骨架和v-结构都是可识别的因果结构，因此关于骨架和v-结构的预测不会受到因果发现中的可识别性限制，因此SiCL可以避免节点-边架构的系统偏差，从而实现因果发现的一致估计器。此外，SiCL还配备了用于建模节点对的内部和外部关系的特制成对编码模块，具有单向注意力层。在合成数据和真实世界基准上的实验结果表明，SiCL显著优于其他基于DNN的SCL方法。 

---
# Broadcast Channel Cooperative Gain: An Operational Interpretation of Partial Information Decomposition 

**Title (ZH)**: 广播信道协同增益：部分信息分解的操作性解释 

**Authors**: Chao Tian, Shlomo Shamai  

**Link**: [PDF](https://arxiv.org/pdf/2502.10878)  

**Abstract**: Partial information decomposition has recently found applications in biological signal processing and machine learning. Despite its impacts, the decomposition was introduced through an informal and heuristic route, and its exact operational meaning is unclear. In this work, we fill this gap by connecting partial information decomposition to the capacity of the broadcast channel, which has been well-studied in the information theory literature. We show that the synergistic information in the decomposition can be rigorously interpreted as the cooperative gain, or a lower bound of this gain, on the corresponding broadcast channel. This interpretation can help practitioners to better explain and expand the applications of the partial information decomposition technique. 

**Abstract (ZH)**: 部分信息分解最近在生物信号处理和机器学习中找到了应用。尽管如此，该分解最初是通过非正式和启发式的方式引入的，其精确的操作含义尚不明确。在本文中，我们通过将部分信息分解与广播信道的容量联系起来，填补了这一空白，而广播信道的容量已在信息理论文献中得到了充分研究。我们证明，在分解中的协同信息可以严格解释为相应广播信道的合作增益，或者这一增益的下界。这一解释有助于实践者更好地解释和扩展部分信息分解技术的应用。 

---
# A Geometric Approach to Personalized Recommendation with Set-Theoretic Constraints Using Box Embeddings 

**Title (ZH)**: 基于集合约束的盒嵌入的个性化推荐的几何方法 

**Authors**: Shib Dasgupta, Michael Boratko, Andrew McCallum  

**Link**: [PDF](https://arxiv.org/pdf/2502.10875)  

**Abstract**: Personalized item recommendation typically suffers from data sparsity, which is most often addressed by learning vector representations of users and items via low-rank matrix factorization. While this effectively densifies the matrix by assuming users and movies can be represented by linearly dependent latent features, it does not capture more complicated interactions. For example, vector representations struggle with set-theoretic relationships, such as negation and intersection, e.g. recommending a movie that is "comedy and action, but not romance". In this work, we formulate the problem of personalized item recommendation as matrix completion where rows are set-theoretically dependent. To capture this set-theoretic dependence we represent each user and attribute by a hyper-rectangle or box (i.e. a Cartesian product of intervals). Box embeddings can intuitively be understood as trainable Venn diagrams, and thus not only inherently represent similarity (via the Jaccard index), but also naturally and faithfully support arbitrary set-theoretic relationships. Queries involving set-theoretic constraints can be efficiently computed directly on the embedding space by performing geometric operations on the representations. We empirically demonstrate the superiority of box embeddings over vector-based neural methods on both simple and complex item recommendation queries by up to 30 \% overall. 

**Abstract (ZH)**: 个性化项目推荐通常受到数据稀疏性的困扰，这通常通过低秩矩阵分解学习用户和项目的向量表示来解决。虽然这种方法通过假设用户和电影可以由线性相关的隐特征来表示有效地填充了矩阵，但它没有捕捉到更复杂的交互。例如，向量表示在处理集合论关系方面存在困难，如否定和交集，例如推荐一部“喜剧和动作，但不是浪漫”的电影。在本工作中，我们将个性化项目推荐问题形式化为矩阵完成问题，其中行是集合论上相关的。为了捕捉这种集合论上的依赖关系，我们将每个用户和属性表示为超矩形或盒子（即区间的笛卡尔积）。盒嵌入可以直观地理解为可训练的文恩图，因而不仅内含表示相似性（通过雅卡德指数），而且自然且忠实支持任意的集合论关系。涉及集合论约束的查询可以通过在嵌入空间上执行几何操作直接高效地计算。我们通过至多30%的整体优势，经验上证明盒嵌入在简单和复杂的项目推荐查询中优于基于向量的神经方法。 

---
# The Representation and Recall of Interwoven Structured Knowledge in LLMs: A Geometric and Layered Analysis 

**Title (ZH)**: LLMs中交织结构知识的表示与回忆：一种几何与分层分析 

**Authors**: Ge Lei, Samuel J. Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2502.10871)  

**Abstract**: This study investigates how large language models (LLMs) represent and recall multi-associated attributes across transformer layers. We show that intermediate layers encode factual knowledge by superimposing related attributes in overlapping spaces, along with effective recall even when attributes are not explicitly prompted. In contrast, later layers refine linguistic patterns and progressively separate attribute representations, optimizing task-specific outputs while appropriately narrowing attribute recall. We identify diverse encoding patterns including, for the first time, the observation of 3D spiral structures when exploring information related to the periodic table of elements. Our findings reveal a dynamic transition in attribute representations across layers, contributing to mechanistic interpretability and providing insights for understanding how LLMs handle complex, interrelated knowledge. 

**Abstract (ZH)**: 本研究探讨了大规模语言模型（LLMs）如何在变压器层间表示和回忆多关联属性。我们展示了中间层通过在重叠空间中叠加相关属性来编码事实知识，并在未明确提示属性的情况下也能有效回忆。相比之下，后期层逐步细化语言模式并分离属性表示，优化特定任务输出的同时适当地限制属性回忆。我们识别出多种编码模式，其中包括首次观察到与元素周期表相关信息时出现的三维螺旋结构。我们的发现揭示了层间属性表示的动态过渡，有助于机械可解释性的建立，并为理解LLMs处理复杂关联知识提供了见解。 

---
# Multilingual Encoder Knows more than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages 

**Title (ZH)**: 多语言编码器知悉的远不止你想象的：适用于极低资源语言的共享权重预训练 

**Authors**: Zeli Su, Ziyin Zhang, Guixian Xu, Jianing Liu, XU Han, Ting Zhang, Yushuang Dong  

**Link**: [PDF](https://arxiv.org/pdf/2502.10852)  

**Abstract**: While multilingual language models like XLM-R have advanced multilingualism in NLP, they still perform poorly in extremely low-resource languages. This situation is exacerbated by the fact that modern LLMs such as LLaMA and Qwen support far fewer languages than XLM-R, making text generation models non-existent for many languages in the world. To tackle this challenge, we propose a novel framework for adapting multilingual encoders to text generation in extremely low-resource languages. By reusing the weights between the encoder and the decoder, our framework allows the model to leverage the learned semantic space of the encoder, enabling efficient learning and effective generalization in low-resource languages. Applying this framework to four Chinese minority languages, we present XLM-SWCM, and demonstrate its superior performance on various downstream tasks even when compared with much larger models. 

**Abstract (ZH)**: 针对极端低资源语言的多语言编码器适应框架：XLM-SWCM的研究 

---
# The Vendiscope: An Algorithmic Microscope For Data Collections 

**Title (ZH)**: Vendiscope: 一种数据集合的算法显微镜 

**Authors**: Amey P. Pasarkar, Adji Bousso Dieng  

**Link**: [PDF](https://arxiv.org/pdf/2502.10828)  

**Abstract**: The evolution of microscopy, beginning with its invention in the late 16th century, has continuously enhanced our ability to explore and understand the microscopic world, enabling increasingly detailed observations of structures and phenomena. In parallel, the rise of data-driven science has underscored the need for sophisticated methods to explore and understand the composition of complex data collections. This paper introduces the Vendiscope, the first algorithmic microscope designed to extend traditional microscopy to computational analysis. The Vendiscope leverages the Vendi scores -- a family of differentiable diversity metrics rooted in ecology and quantum mechanics -- and assigns weights to data points based on their contribution to the overall diversity of the collection. These weights enable high-resolution data analysis at scale. We demonstrate this across biology, materials science, and machine learning (ML). We analyzed the $250$ million protein sequences in the protein universe, discovering that over $200$ million are near-duplicates and that AlphaFold fails on proteins with Gene Ontology (GO) functions that contribute most to diversity. Applying the Vendiscope to the Materials Project database led to similar findings: more than $85\%$ of the crystals with formation energy data are near-duplicates and ML models perform poorly on materials that enhance diversity. Additionally, the Vendiscope can be used to study phenomena such as memorization in generative models. We used the Vendiscope to identify memorized training samples from $13$ different generative models and found that the best-performing ones often memorize the training samples that contribute least to diversity. Our findings demonstrate that the Vendiscope can serve as a powerful tool for data-driven science. 

**Abstract (ZH)**: 显微镜的发展始于16世纪末的发明，不断增强了我们探索和理解微观世界的能力，使我们能够越来越详细地观察结构和现象。同时，数据驱动科学的兴起强调了探索和理解复杂数据集组成所需的先进方法的重要性。本文介绍了Vendiscope，这是第一个算法显微镜，旨在将传统显微镜扩展到计算分析。Vendiscope利用了Vendi分数——一种根植于生态学和量子力学的可微分多样性度量——并根据数据点对整体多样性贡献的大小为其分配权重。这些权重使大规模高分辨率数据分析成为可能。我们在生物学、材料科学和机器学习（ML）领域进行了验证。我们分析了蛋白质宇宙中的2.5亿条蛋白质序列，发现其中超过2亿条是近似重复序列，并且AlphaFold在基因 ontology (GO) 功能对多样性贡献最大的蛋白质上表现不佳。将Vendiscope应用于Materials Project数据库也得到了类似的发现：超过85%带有形成能量数据的晶体是近似重复序列，而机器学习模型在促进多样性的材料上表现不佳。此外，Vendiscope还可以用于研究生成模型中的记忆现象。我们使用Vendiscope从13个不同的生成模型中识别出了记忆训练样本，并发现表现最佳的模型往往记忆的是对多样性贡献最少的训练样本。我们的发现证明Vendiscope可以作为一种强大的数据驱动科学工具。 

---
# MITRE ATT&CK Applications in Cybersecurity and The Way Forward 

**Title (ZH)**: MITRE ATT&CK在 cybersecurity 中的应用及未来发展方向 

**Authors**: Yuning Jiang, Qiaoran Meng, Feiyang Shang, Nay Oo, Le Thi Hong Minh, Hoon Wei Lim, Biplab Sikdar  

**Link**: [PDF](https://arxiv.org/pdf/2502.10825)  

**Abstract**: The MITRE ATT&CK framework is a widely adopted tool for enhancing cybersecurity, supporting threat intelligence, incident response, attack modeling, and vulnerability prioritization. This paper synthesizes research on its application across these domains by analyzing 417 peer-reviewed publications. We identify commonly used adversarial tactics, techniques, and procedures (TTPs) and examine the integration of natural language processing (NLP) and machine learning (ML) with ATT&CK to improve threat detection and response. Additionally, we explore the interoperability of ATT&CK with other frameworks, such as the Cyber Kill Chain, NIST guidelines, and STRIDE, highlighting its versatility. The paper further evaluates the framework from multiple perspectives, including its effectiveness, validation methods, and sector-specific challenges, particularly in industrial control systems (ICS) and healthcare. We conclude by discussing current limitations and proposing future research directions to enhance the applicability of ATT&CK in dynamic cybersecurity environments. 

**Abstract (ZH)**: MITRE ATT&CK框架在增强网络安全、支持威胁情报、事件响应、攻击建模和漏洞优先级排序中的应用综述：基于417篇同行评审出版物的分析及其与其他框架的互操作性与挑战 

---
# NeuroAMP: A Novel End-to-end General Purpose Deep Neural Amplifier for Personalized Hearing Aids 

**Title (ZH)**: NeuroAMP：一种新型端到端通用深度神经放大器，用于个性化助听器 

**Authors**: Shafique Ahmed, Ryandhimas E. Zezario, Hui-Guan Yuan, Amir Hussain, Hsin-Min Wang, Wei-Ho Chung, Yu Tsao  

**Link**: [PDF](https://arxiv.org/pdf/2502.10822)  

**Abstract**: The prevalence of hearing aids is increasing. However, optimizing the amplification processes of hearing aids remains challenging due to the complexity of integrating multiple modular components in traditional methods. To address this challenge, we present NeuroAMP, a novel deep neural network designed for end-to-end, personalized amplification in hearing aids. NeuroAMP leverages both spectral features and the listener's audiogram as inputs, and we investigate four architectures: Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), Convolutional Recurrent Neural Network (CRNN), and Transformer. We also introduce Denoising NeuroAMP, an extension that integrates noise reduction along with amplification capabilities for improved performance in real-world scenarios. To enhance generalization, a comprehensive data augmentation strategy was employed during training on diverse speech (TIMIT and TMHINT) and music (Cadenza Challenge MUSIC) datasets. Evaluation using the Hearing Aid Speech Perception Index (HASPI), Hearing Aid Speech Quality Index (HASQI), and Hearing Aid Audio Quality Index (HAAQI) demonstrates that the Transformer architecture within NeuroAMP achieves the best performance, with SRCC scores of 0.9927 (HASQI) and 0.9905 (HASPI) on TIMIT, and 0.9738 (HAAQI) on the Cadenza Challenge MUSIC dataset. Notably, our data augmentation strategy maintains high performance on unseen datasets (e.g., VCTK, MUSDB18-HQ). Furthermore, Denoising NeuroAMP outperforms both the conventional NAL-R+WDRC approach and a two-stage baseline on the VoiceBank+DEMAND dataset, achieving a 10% improvement in both HASPI (0.90) and HASQI (0.59) scores. These results highlight the potential of NeuroAMP and Denoising NeuroAMP to deliver notable improvements in personalized hearing aid amplification. 

**Abstract (ZH)**: 听觉辅助设备的使用率正在增加。然而，由于传统方法中多种模块组件集成的复杂性，优化听觉辅助设备的放大过程仍然具有挑战性。为应对这一挑战，我们提出了NeuroAMP，这是一种用于听觉辅助设备端到端个性化放大处理的新型深度神经网络。NeuroAMP利用频谱特征和听者的听力图作为输入，并探讨了四种架构：卷积神经网络（CNN）、长短期记忆网络（LSTM）、卷积循环神经网络（CRNN）和Transformer。此外，我们还引入了去噪NeuroAMP，这是一种结合了降噪和放大能力的扩展，以在实际场景中提高性能。为了增强泛化能力，在不同语音（TIMIT和TMHINT）和音乐（Cadenza Challenge MUSIC）数据集上进行训练时，采取了全面的数据增强策略。使用听觉辅助设备言语感知指数（HASPI）、听觉辅助设备言语质量指数（HASQI）和听觉辅助设备音频质量指数（HAAQI）进行了评估，结果显示NeuroAMP中的Transformer架构性能最佳，TIMIT数据集上的SRCC评分为0.9927（HASQI）和0.9905（HASPI），Cadenza Challenge MUSIC数据集上的评分为0.9738（HAAQI）。值得注意的是，我们的数据增强策略在未见过的数据集（如VCTK、MUSDB18-HQ）上也保持了高性能。此外，去噪NeuroAMP在VoiceBank+DEMAND数据集上的表现优于传统的NAL-R+WDRC方法和两阶段基线，分别在HASPI和HASQI分数上提高了10%（0.90和0.59）。这些结果突显了NeuroAMP和去噪NeuroAMP在个性化听觉辅助设备放大方面具有显著的改进潜力。 

---
# On Vanishing Gradients, Over-Smoothing, and Over-Squashing in GNNs: Bridging Recurrent and Graph Learning 

**Title (ZH)**: 关于GNN中消失梯度、过度平滑和过度压缩现象：连接递归学习与图学习 

**Authors**: Álvaro Arroyo, Alessio Gravina, Benjamin Gutteridge, Federico Barbero, Claudio Gallicchio, Xiaowen Dong, Michael Bronstein, Pierre Vandergheynst  

**Link**: [PDF](https://arxiv.org/pdf/2502.10818)  

**Abstract**: Graph Neural Networks (GNNs) are models that leverage the graph structure to transmit information between nodes, typically through the message-passing operation. While widely successful, this approach is well known to suffer from the over-smoothing and over-squashing phenomena, which result in representational collapse as the number of layers increases and insensitivity to the information contained at distant and poorly connected nodes, respectively. In this paper, we present a unified view of these problems through the lens of vanishing gradients, using ideas from linear control theory for our analysis. We propose an interpretation of GNNs as recurrent models and empirically demonstrate that a simple state-space formulation of a GNN effectively alleviates over-smoothing and over-squashing at no extra trainable parameter cost. Further, we show theoretically and empirically that (i) GNNs are by design prone to extreme gradient vanishing even after a few layers; (ii) Over-smoothing is directly related to the mechanism causing vanishing gradients; (iii) Over-squashing is most easily alleviated by a combination of graph rewiring and vanishing gradient mitigation. We believe our work will help bridge the gap between the recurrent and graph neural network literature and will unlock the design of new deep and performant GNNs. 

**Abstract (ZH)**: 基于图的神经网络（GNNs）通过图结构在节点间传输信息，通常通过消息传递操作实现。尽管这种方法在广泛应用中表现出色，但它已知会遭受过度平滑和过度压缩的现象，随着层数增加导致表示坍塌，并且对远处和连接不良节点包含的信息变得不敏感。在本文中，我们通过梯度消失的角度提出一个统一的观点，利用线性控制理论的思想进行分析。我们提出了GNNs作为递归模型的一种解释，并通过简单的状态空间形式的GNN实验证明，这种方法在不增加额外可训练参数的情况下，可以有效缓解过度平滑和过度压缩。此外，我们理论和实验上证明了：(i) GNNs在几层后设计上就容易遭受极端梯度消失；(ii) 过度平滑直接与导致梯度消失的机制有关；(iii) 通过图重 wiring 和梯度消失缓解的组合最有效地缓解过度压缩。我们认为我们的工作将有助于弥合递归模型和图神经网络文献之间的差距，并开启设计新的深度和高性能GNNs的途径。 

---
# BalanceBenchmark: A Survey for Imbalanced Learning 

**Title (ZH)**: 平衡基准：不平衡学习综述 

**Authors**: Shaoxuan Xu, Menglu Cui, Chengxiang Huang, Hongfa Wang, DiHu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10816)  

**Abstract**: Multimodal learning has gained attention for its capacity to integrate information from different modalities. However, it is often hindered by the multimodal imbalance problem, where certain modality dominates while others remain underutilized. Although recent studies have proposed various methods to alleviate this problem, they lack comprehensive and fair comparisons. In this paper, we systematically categorize various mainstream multimodal imbalance algorithms into four groups based on the strategies they employ to mitigate imbalance. To facilitate a comprehensive evaluation of these methods, we introduce BalanceBenchmark, a benchmark including multiple widely used multidimensional datasets and evaluation metrics from three perspectives: performance, imbalance degree, and complexity. To ensure fair comparisons, we have developed a modular and extensible toolkit that standardizes the experimental workflow across different methods. Based on the experiments using BalanceBenchmark, we have identified several key insights into the characteristics and advantages of different method groups in terms of performance, balance degree and computational complexity. We expect such analysis could inspire more efficient approaches to address the imbalance problem in the future, as well as foundation models. The code of the toolkit is available at this https URL. 

**Abstract (ZH)**: 多模态学习因其能够整合不同模态的信息而受到关注，但常常受到多模态不平衡问题的阻碍，其中某些模态占主导地位而其他模态则被严重低估。尽管最近的研究提出了各种方法来缓解这一问题，但它们缺乏全面和公平的比较。本文根据各方法缓解不平衡所采用的策略，系统地将主流多模态不平衡算法归类为四组。为了促进这些方法的全面评估，我们引入了BalanceBenchmark基准，该基准包括多个广泛使用的多维数据集和从三个方面（性能、不平衡程度和复杂性）进行评估的指标。为了确保公平比较，我们开发了一个模块化和可扩展的工具包，标准化了不同方法的实验工作流程。基于使用BalanceBenchmark进行的实验，我们对不同方法组在性能、平衡程度和计算复杂性方面的特点和优势进行了分析。我们期望这样的分析能够激发未来更有效的解决不平衡问题的方法，以及基础模型的方法。该工具包的代码可通过此链接获取。 

---
# HybriDNA: A Hybrid Transformer-Mamba2 Long-Range DNA Language Model 

**Title (ZH)**: HybriDNA: 一种混合Transformer-Mamba2长范围DNA语言模型 

**Authors**: Mingqian Ma, Guoqing Liu, Chuan Cao, Pan Deng, Tri Dao, Albert Gu, Peiran Jin, Zhao Yang, Yingce Xia, Renqian Luo, Pipi Hu, Zun Wang, Yuan-Jyue Chen, Haiguang Liu, Tao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2502.10807)  

**Abstract**: Advances in natural language processing and large language models have sparked growing interest in modeling DNA, often referred to as the "language of life". However, DNA modeling poses unique challenges. First, it requires the ability to process ultra-long DNA sequences while preserving single-nucleotide resolution, as individual nucleotides play a critical role in DNA function. Second, success in this domain requires excelling at both generative and understanding tasks: generative tasks hold potential for therapeutic and industrial applications, while understanding tasks provide crucial insights into biological mechanisms and diseases. To address these challenges, we propose HybriDNA, a decoder-only DNA language model that incorporates a hybrid Transformer-Mamba2 architecture, seamlessly integrating the strengths of attention mechanisms with selective state-space models. This hybrid design enables HybriDNA to efficiently process DNA sequences up to 131kb in length with single-nucleotide resolution. HybriDNA achieves state-of-the-art performance across 33 DNA understanding datasets curated from the BEND, GUE, and LRB benchmarks, and demonstrates exceptional capability in generating synthetic cis-regulatory elements (CREs) with desired properties. Furthermore, we show that HybriDNA adheres to expected scaling laws, with performance improving consistently as the model scales from 300M to 3B and 7B parameters. These findings underscore HybriDNA's versatility and its potential to advance DNA research and applications, paving the way for innovations in understanding and engineering the "language of life". 

**Abstract (ZH)**: 自然语言处理和大规模语言模型的进步引发了对DNA建模日益增长的兴趣，DNA常被称作“生命之语言”。然而，DNA建模面临着独特挑战。首先，它需要处理超长DNA序列同时保持单核苷酸分辨率，因为单个核苷酸在DNA功能中起着关键作用。其次，在这个领域取得成功要求在生成性和理解性任务上均表现出色：生成性任务在治疗和工业应用上具有潜力，而理解性任务则提供了关于生物学机制和疾病的关键见解。为了解决这些挑战，我们提出了一种仅解码器DNA语言模型HybriDNA，该模型采用混合Transformer-Mamba2架构，无缝结合了注意力机制的优势与选择性状态空间模型的优势。这种混合设计使HybriDNA能够高效处理长达131kb的DNA序列，保持单核苷酸分辨率。HybriDNA在来自BEND、GUE和LRB基准的33个DNA理解数据集中达到了最先进的性能，并在生成具有所需特性的合成顺式调控元件（CREs）方面展现了卓越能力。此外，我们表明HybriDNA遵循预期的缩放定律，模型参数从300M增至3B和7B时，性能持续提升。这些发现突显了HybriDNA的多功能性及其在推进DNA研究和应用方面的潜力，为其在理解与工程“生命之语言”方面的创新铺平了道路。 

---
# PDA: Generalizable Detection of AI-Generated Images via Post-hoc Distribution Alignment 

**Title (ZH)**: PDA：通过后 hoc 分布对齐实现的可泛化的 AI 生成图像检测 

**Authors**: Li Wang, Wenyu Chen, Zheng Li, Shanqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.10803)  

**Abstract**: The rapid advancement of generative models has led to the proliferation of highly realistic AI-generated images, posing significant challenges for detection methods to generalize across diverse and evolving generative techniques. Existing approaches often fail to adapt to unknown models without costly retraining, limiting their practicability. To fill this gap, we propose Post-hoc Distribution Alignment (PDA), a novel approach for the generalizable detection for AI-generated images. The key idea is to use the known generative model to regenerate undifferentiated test images. This process aligns the distributions of the re-generated real images with the known fake images, enabling effective distinction from unknown fake images. PDA employs a two-step detection framework: 1) evaluating whether a test image aligns with the known fake distribution based on deep k-nearest neighbor (KNN) distance, and 2) re-generating test images using known generative models to create pseudo-fake images for further classification. This alignment strategy allows PDA to effectively detect fake images without relying on unseen data or requiring retraining. Extensive experiments demonstrate the superiority of PDA, achieving 96.73\% average accuracy across six state-of-the-art generative models, including GANs, diffusion models, and text-to-image models, and improving by 16.07\% over the best baseline. Through t-SNE visualizations and KNN distance analysis, we provide insights into PDA's effectiveness in separating real and fake images. Our work provides a flexible and effective solution for real-world fake image detection, advancing the generalization ability of detection systems. 

**Abstract (ZH)**: 生成模型的快速进步导致了高度现实的AI生成图像的泛滥，给检测方法跨多种不断发展中的生成技术进行泛化的检测带来了重大挑战。现有方法往往无法在无需昂贵重训的情况下适应未知模型，限制了其实用性。为了解决这一问题，我们提出了后验分布对齐（PDA），一种用于AI生成图像泛化检测的新方法。关键思路是使用已知的生成模型再生未区分的测试图像。这一过程使重新生成的真实图像分布与已知的伪造图像分布对齐，从而能够有效地区分未知的伪造图像。PDA采用两步检测框架：1）基于深度k近邻（KNN）距离评估测试图像是否与已知伪造分布对齐，2）使用已知生成模型再生测试图像以生成伪伪造图像进行进一步分类。这种对齐策略使得PDA能够在不依赖未见数据或重训的情况下有效检测伪造图像。广泛的实验表明，PDA的优越性，其在六个最先进的生成模型（包括GANs、扩散模型和文本到图像模型）上实现了96.73%的平均准确性，并且相较于最佳baselines提升了16.07%。通过t-SNE可视化和KNN距离分析，我们揭示了PDA在区分真实和伪造图像方面的有效性。我们的工作提供了一种灵活和有效的解决方案，用于实际环境中的伪造图像检测，推动了检测系统泛化能力的提升。 

---
# CoCoEvo: Co-Evolution of Programs and Test Cases to Enhance Code Generation 

**Title (ZH)**: CoCoEvo: 程序与测试用例的协同进化以增强代码生成 

**Authors**: Kefan Li, Hongyue Yu, Tingyu Guo, Shijie Cao, Yuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10802)  

**Abstract**: Large Language Models (LLMs) have shown remarkable performance in automated code generation. However, existing approaches often rely heavily on pre-defined test cases, which become impractical in scenarios where such cases are unavailable. While prior works explore filtering techniques between programs and test cases, they overlook the refinement of test cases. To address this limitation, we introduce CoCoEvo, a novel LLM-based co-evolution framework that simultaneously evolves programs and test cases. CoCoEvo eliminates the dependency on pre-defined test cases by generating both programs and test cases directly from natural language problem descriptions and function headers. The framework employs specialized evolutionary operators, including LLM-based crossover and mutation operators for program evolution, along with a test case generation operator for test case evolution. Additionally, we propose optimization strategies such as a crossover rate scheduler to balance exploration and convergence, and a multi-objective optimization method for test case selection. Experimental results on multiple state-of-the-art LLMs demonstrate that CoCoEvo surpasses existing methods, achieving state-of-the-art performance in automated code generation and testing. These results underscore the potential of co-evolutionary techniques in advancing the field of automated programming. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动化代码生成方面展示了 remarkable 的性能。然而，现有方法往往高度依赖预定义的测试案例，在测试案例不可用的情况下变得不切实际。虽然早期研究探索了程序和测试案例之间的过滤技术，但忽略了测试案例的细化。为解决这一限制，我们引入了 CoCoEvo，这是一种新颖的基于 LLM 的协同进化框架，可以同时进化程序和测试案例。CoCoEvo 通过直接从自然语言问题描述和函数头生成程序和测试案例来消除对预定义测试案例的依赖。该框架采用专门的进化操作符，包括基于 LLM 的交叉和变异操作符用于程序进化，以及用于测试案例进化的测试案例生成操作符。此外，我们还提出了一些优化策略，如交叉率调度器以平衡探索与收敛，以及多目标优化方法用于测试案例选择。实验结果表明，CoCoEvo 在多个最先进的 LLM 上超越了现有方法，在自动化代码生成和测试方面达到了最先进的性能。这些结果突显了协同进化技术在推进自动化编程领域方面的潜力。 

---
# FaceSwapGuard: Safeguarding Facial Privacy from DeepFake Threats through Identity Obfuscation 

**Title (ZH)**: FaceSwapGuard：通过身份模糊化保障面部隐私免受深度伪造威胁 

**Authors**: Li Wang, Zheng Li, Xuhong Zhang, Shouling Ji, Shanqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.10801)  

**Abstract**: DeepFakes pose a significant threat to our society. One representative DeepFake application is face-swapping, which replaces the identity in a facial image with that of a victim. Although existing methods partially mitigate these risks by degrading the quality of swapped images, they often fail to disrupt the identity transformation effectively. To fill this gap, we propose FaceSwapGuard (FSG), a novel black-box defense mechanism against deepfake face-swapping threats. Specifically, FSG introduces imperceptible perturbations to a user's facial image, disrupting the features extracted by identity encoders. When shared online, these perturbed images mislead face-swapping techniques, causing them to generate facial images with identities significantly different from the original user. Extensive experiments demonstrate the effectiveness of FSG against multiple face-swapping techniques, reducing the face match rate from 90\% (without defense) to below 10\%. Both qualitative and quantitative studies further confirm its ability to confuse human perception, highlighting its practical utility. Additionally, we investigate key factors that may influence FSG and evaluate its robustness against various adaptive adversaries. 

**Abstract (ZH)**: DeepFakes 对社会构成显著威胁。一种代表性的 DeepFake 应用是面部替换，将面部图像中的身份替换为受害者的身份。尽管现有方法部分降低了这些风险，通过降级替换图像的质量，但往往无法有效地破坏身份转换。为填补这一空白，我们提出了一种新型的黑盒防御机制 FaceSwapGuard (FSG) 以对抗 DeepFake 面部替换威胁。具体而言，FSG 在用户面部图像中引入不可感知的扰动，破坏身份编码器提取的特征。当这些扰动图像在网络上共享时，会误导面部替换技术，使其生成的身份与原始用户显著不同的面部图像。广泛实验表明，FSG 在对抗多种面部替换技术方面有效，将面部匹配率从无防御措施的 90% 降至不到 10%。定性和定量研究进一步证实了其混淆人类感知的能力，突显了其实用价值。此外，我们还探讨了可能影响 FSG 的关键因素，并评估了其在面对各种适应性对手时的稳健性。 

---
# Dynamic Influence Tracker: Measuring Time-Varying Sample Influence During Training 

**Title (ZH)**: 动态影响追踪器：训练过程中样本时间变化影响的度量 

**Authors**: Jie Xu, Zihan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10793)  

**Abstract**: Existing methods for measuring training sample influence on models only provide static, overall measurements, overlooking how sample influence changes during training. We propose Dynamic Influence Tracker (DIT), which captures the time-varying sample influence across arbitrary time windows during training.
DIT offers three key insights: 1) Samples show different time-varying influence patterns, with some samples important in the early training stage while others become important later. 2) Sample influences show a weak correlation between early and late stages, demonstrating that the model undergoes distinct learning phases with shifting priorities. 3) Analyzing influence during the convergence period provides more efficient and accurate detection of corrupted samples than full-training analysis. Supported by theoretical guarantees without assuming loss convexity or model convergence, DIT significantly outperforms existing methods, achieving up to 0.99 correlation with ground truth and above 98\% accuracy in detecting corrupted samples in complex architectures. 

**Abstract (ZH)**: 动态影响追踪器（DIT）：训练过程中样本影响的动态评估 

---
# A Distillation-based Future-aware Graph Neural Network for Stock Trend Prediction 

**Title (ZH)**: 基于蒸馏的前瞻性图神经网络股票趋势预测 

**Authors**: Zhipeng Liu, Peibo Duan, Mingyang Geng, Bin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10776)  

**Abstract**: Stock trend prediction involves forecasting the future price movements by analyzing historical data and various market indicators. With the advancement of machine learning, graph neural networks (GNNs) have been extensively employed in stock prediction due to their powerful capability to capture spatiotemporal dependencies of stocks. However, despite the efforts of various GNN stock predictors to enhance predictive performance, the improvements remain limited, as they focus solely on analyzing historical spatiotemporal dependencies, overlooking the correlation between historical and future patterns. In this study, we propose a novel distillation-based future-aware GNN framework (DishFT-GNN) for stock trend prediction. Specifically, DishFT-GNN trains a teacher model and a student model, iteratively. The teacher model learns to capture the correlation between distribution shifts of historical and future data, which is then utilized as intermediate supervision to guide the student model to learn future-aware spatiotemporal embeddings for accurate prediction. Through extensive experiments on two real-world datasets, we verify the state-of-the-art performance of DishFT-GNN. 

**Abstract (ZH)**: 基于蒸馏的未来意识图神经网络框架（DishFT-GNN）用于股票趋势预测 

---
# Evaluating improvements on using Large Language Models (LLMs) for property extraction in the Open Research Knowledge Graph (ORKG) 

**Title (ZH)**: 评估在开放研究知识图谱（ORKG）中使用大型语言模型（LLMs）进行属性提取的改进效果 

**Authors**: Sandra Schaftner  

**Link**: [PDF](https://arxiv.org/pdf/2502.10768)  

**Abstract**: Current research highlights the great potential of Large Language Models (LLMs) for constructing Scholarly Knowledge Graphs (SKGs). One particularly complex step in this process is relation extraction, aimed at identifying suitable properties to describe the content of research. This study builds directly on previous research of three Open Research Knowledge Graph (ORKG) team members who assessed the readiness of LLMs such as GPT-3.5, Llama 2, and Mistral for property extraction in scientific literature. Given the moderate performance observed, the previous work concluded that fine-tuning is needed to improve these models' alignment with scientific tasks and their emulation of human expertise. Expanding on this prior experiment, this study evaluates the impact of advanced prompt engineering techniques and demonstrates that these techniques can highly significantly enhance the results. Additionally, this study extends the property extraction process to include property matching to existing ORKG properties, which are retrieved via the API. The evaluation reveals that results generated through advanced prompt engineering achieve a higher proportion of matches with ORKG properties, further emphasizing the enhanced alignment achieved. Moreover, this lays the groundwork for addressing challenges such as the inconsistency of ORKG properties, an issue highlighted in prior studies. By assigning unique URIs and using standardized terminology, this work increases the consistency of the properties, fulfilling a crucial aspect of Linked Data and FAIR principles - core commitments of ORKG. This, in turn, significantly enhances the applicability of ORKG content for subsequent tasks such as comparisons of research publications. Finally, the study concludes with recommendations for future improvements in the overall property extraction process. 

**Abstract (ZH)**: 当前研究突显了大型语言模型（LLMs）在构建学术知识图谱（SKGs）方面的巨大潜力。这一过程中特别复杂的一个步骤是关系提取，旨在识别适合描述研究内容的属性。本研究直接建立在三位开放研究知识图谱（ORKG）团队成员之前的研究基础上，他们评估了如GPT-3.5、Llama 2和Mistral等LLMs在科学文献中属性提取方面的准备情况。鉴于观察到的中等性能，之前的研究得出结论认为，需要对这些模型进行微调以提高它们与科学任务的对齐以及模拟人类专业知识的能力。在此前实验的基础上，本研究评估了高级提示工程技术的影响，并证明了这些技术可以显著提高结果。此外，本研究将属性提取过程扩展到属性匹配现有的ORKG属性，这些属性通过API检索。评估结果显示，通过高级提示工程生成的结果与ORKG属性的匹配比例更高，进一步突显了所实现的对齐程度的增强。此外，这为应对ORKG属性的一致性问题奠定了基础，这是之前研究中指出的一个问题。通过分配唯一的URI并使用标准化术语，本研究增加了属性的一致性，实现了链接数据和FAIR原则的核心承诺之一——ORKG内容后续任务如研究出版物的比较适用性的显著增强。最后，本研究提出了总体属性提取流程改进的建议。 

---
# Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation 

**Title (ZH)**: 骨汤模型：一种用于可控多目标生成的寻觅与汤合并方法 

**Authors**: Guofu Xie, Xiao Zhang, Ting Yao, Yunsheng Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.10762)  

**Abstract**: User information needs are often highly diverse and varied. A key challenge in current research is how to achieve controllable multi-objective generation while enabling rapid adaptation to accommodate diverse user demands during test time. Existing solutions, such as Rewarded Soup, focus on merging language models individually tuned on single objectives. While easy to implement and widely used, these approaches face limitations in achieving optimal performance due to their disregard for the impacts of competing objectives on model tuning. To address this issue, we propose Bone Soup, a novel model merging approach that first seeks a series of backbone models by considering the impacts of multiple objectives and then makes the soup (i.e., merge the backbone models). Specifically, Bone Soup begins by training multiple backbone models for different objectives using multi-objective reinforcement learning. Each backbone model is guided by a combination of backbone reward signals. To ensure that these models are optimal for the Pareto front, the backbone rewards are crafted by combining standard reward functions into basis vectors, which can then be modified through a rule-based construction method. Bone Soup leverages a symmetric circulant matrix mapping to generate the merging coefficients, which are used to merge the backbone models according to user preferences. Extensive experimental results demonstrate that Bone Soup exhibits strong controllability and Pareto optimality in controllable multi-objective generation, providing a more effective and efficient approach to addressing diverse user needs at test time. 

**Abstract (ZH)**: 用户信息需求通常高度多样且各不相同。当前研究中的一个关键挑战是如何在满足多样化用户需求的同时实现可控的多目标生成并快速适应。现有解决方案，如Rewarded Soup，侧重于合并单目标上单独调优的语言模型。尽管易于实现且广泛应用，但这些方法由于忽视了竞争目标对模型调优的影响而存在局限性。为解决这一问题，我们提出Bone Soup，这是一种新颖的模型合并方法，首先通过考虑多个目标的影响来寻求一系列骨干模型，然后将这些模型合并（即，合并骨干模型）。具体而言，Bone Soup首先使用多目标强化学习训练不同目标的多个骨干模型。每个骨干模型由组合的基础奖励信号引导。为了确保这些模型适用于帕累托前沿，通过对标准奖励函数进行组合形成基向量，并通过规则构造方法进行修改来构建基础奖励。Bone Soup利用对称循环矩阵映射生成合并系数，根据用户偏好合并骨干模型。广泛的经验结果表明，Bone Soup在可控多目标生成中表现出强大的可控性和帕累托最优性，为在测试时间更好地满足多样用户需求提供更有效的解决方案。 

---
# Human-Centric Community Detection in Hybrid Metaverse Networks with Integrated AI Entities 

**Title (ZH)**: 基于人类导向的混合元网络中集成AI实体的社区检测 

**Authors**: Shih-Hsuan Chiu, Ya-Wen Teng, De-Nian Yang, Ming-Syan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.10750)  

**Abstract**: Community detection is a cornerstone problem in social network analysis (SNA), aimed at identifying cohesive communities with minimal external links. However, the rise of generative AI and Metaverse introduce complexities by creating hybrid human-AI social networks (denoted by HASNs), where traditional methods fall short, especially in human-centric settings. This paper introduces a novel community detection problem in HASNs (denoted by MetaCD), which seeks to enhance human connectivity within communities while reducing the presence of AI nodes. Effective processing of MetaCD poses challenges due to the delicate trade-off between excluding certain AI nodes and maintaining community structure. To address this, we propose CUSA, an innovative framework incorporating AI-aware clustering techniques that navigate this trade-off by selectively retaining AI nodes that contribute to community integrity. Furthermore, given the scarcity of real-world HASNs, we devise four strategies for synthesizing these networks under various hypothetical scenarios. Empirical evaluations on real social networks, reconfigured as HASNs, demonstrate the effectiveness and practicality of our approach compared to traditional non-deep learning and graph neural network (GNN)-based methods. 

**Abstract (ZH)**: 元社会网络中的社区检测问题（MetaCD）：增强人类连接的同时减少AI节点的存在 

---
# LoRE-Merging: Exploring Low-Rank Estimation For Large Language Model Merging 

**Title (ZH)**: LoRE-合并：探索大型语言模型合并的低秩估计方法 

**Authors**: Zehua Liu, Han Wu, Yuxuan Yao, Ruifeng She, Xiongwei Han, Tao Zhong, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10749)  

**Abstract**: While most current approaches rely on further training techniques, such as fine-tuning or reinforcement learning, to enhance model capacities, model merging stands out for its ability of improving models without requiring any additional training. In this paper, we propose a unified framework for model merging based on low-rank estimation of task vectors without the need for access to the base model, named \textsc{LoRE-Merging}. Our approach is motivated by the observation that task vectors from fine-tuned models frequently exhibit a limited number of dominant singular values, making low-rank estimations less prone to interference. We implement the method by formulating the merging problem as an optimization problem. Extensive empirical experiments demonstrate the effectiveness of our framework in mitigating interference and preserving task-specific information, thereby advancing the state-of-the-art performance in model merging techniques. 

**Abstract (ZH)**: 基于低秩估计的任务向量融合：无需访问基模型的统一框架 

---
# Rule-Bottleneck Reinforcement Learning: Joint Explanation and Decision Optimization for Resource Allocation with Language Agents 

**Title (ZH)**: 规则瓶颈强化学习：语言代理参与的资源分配解释与决策优化 

**Authors**: Mauricio Tec, Guojun Xiong, Haichuan Wang, Francesca Dominici, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2502.10732)  

**Abstract**: Deep Reinforcement Learning (RL) is remarkably effective in addressing sequential resource allocation problems in domains such as healthcare, public policy, and resource management. However, deep RL policies often lack transparency and adaptability, challenging their deployment alongside human decision-makers. In contrast, Language Agents, powered by large language models (LLMs), provide human-understandable reasoning but may struggle with effective decision making. To bridge this gap, we propose Rule-Bottleneck Reinforcement Learning (RBRL), a novel framework that jointly optimizes decision and explanations. At each step, RBRL generates candidate rules with an LLM, selects among them using an attention-based RL policy, and determines the environment action with an explanation via chain-of-thought reasoning. The RL rule selection is optimized using the environment rewards and an explainability metric judged by the LLM. Evaluations in real-world scenarios highlight RBRL's competitive performance with deep RL and efficiency gains over LLM fine-tuning. A survey further confirms the enhanced quality of its explanations. 

**Abstract (ZH)**: 基于规则瓶颈的强化学习（RBRL）：决策与解释的联合优化 

---
# PropNet: a White-Box and Human-Like Network for Sentence Representation 

**Title (ZH)**: PropNet：一种白盒且类人类的句子表示网络 

**Authors**: Fei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10725)  

**Abstract**: Transformer-based embedding methods have dominated the field of sentence representation in recent years. Although they have achieved remarkable performance on NLP missions, such as semantic textual similarity (STS) tasks, their black-box nature and large-data-driven training style have raised concerns, including issues related to bias, trust, and safety. Many efforts have been made to improve the interpretability of embedding models, but these problems have not been fundamentally resolved. To achieve inherent interpretability, we propose a purely white-box and human-like sentence representation network, PropNet. Inspired by findings from cognitive science, PropNet constructs a hierarchical network based on the propositions contained in a sentence. While experiments indicate that PropNet has a significant gap compared to state-of-the-art (SOTA) embedding models in STS tasks, case studies reveal substantial room for improvement. Additionally, PropNet enables us to analyze and understand the human cognitive processes underlying STS benchmarks. 

**Abstract (ZH)**: 基于Transformer的嵌入方法近年来在句子表示领域占主导地位。尽管它们在自然语言处理任务，如语义文本相似度（STS）任务中取得了显著性能，但其黑盒性质和数据驱动的训练方式引发了关于偏差、信任和安全的问题。尽管已经做出了许多努力来提高嵌入模型的可解释性，但这些问题尚未从根本上得到解决。为了实现固有的可解释性，我们提出了一种纯白盒且类人类的句子表示网络PropNet。受认知科学研究结果的启发，PropNet基于句子中的命题构建了一个层次网络。尽管实验表明，在STS任务中PropNet与最先进的（SOTA）嵌入模型相比存在显著差距，但案例研究揭示了其改进的巨大空间。此外，PropNet使我们能够分析和理解STS基准背后的人类认知过程。 

---
# A Mathematics Framework of Artificial Shifted Population Risk and Its Further Understanding Related to Consistency Regularization 

**Title (ZH)**: 一个人工移位人口风险的数学框架及其与一致性正则化的进一步理解 

**Authors**: Xiliang Yang, Shenyang Deng, Shicong Liu, Yuanchi Suo, Wing.W.Y NG, Jianjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10723)  

**Abstract**: Data augmentation is an important technique in training deep neural networks as it enhances their ability to generalize and remain robust. While data augmentation is commonly used to expand the sample size and act as a consistency regularization term, there is a lack of research on the relationship between them. To address this gap, this paper introduces a more comprehensive mathematical framework for data augmentation. Through this framework, we establish that the expected risk of the shifted population is the sum of the original population risk and a gap term, which can be interpreted as a consistency regularization term. The paper also provides a theoretical understanding of this gap, highlighting its negative effects on the early stages of training. We also propose a method to mitigate these effects. To validate our approach, we conducted experiments using same data augmentation techniques and computing resources under several scenarios, including standard training, out-of-distribution, and imbalanced classification. The results demonstrate that our methods surpass compared methods under all scenarios in terms of generalization ability and convergence stability. We provide our code implementation at the following link: this https URL. 

**Abstract (ZH)**: 数据增强是训练深度神经网络的重要技术，它能增强模型的泛化能力和鲁棒性。虽然数据增强常用于扩大样本量并起到一致性正则化的作用，但它们之间的关系研究尚不足。为弥补这一不足，本文提出了一个更为全面的数据增强数学框架。通过该框架，我们确立了移位群体的预期风险等于原始群体风险与一个差异项之和，该差异项可解释为一致性正则化项。本文还从理论上解释了这一差异项，并强调其对训练早期阶段的负面影响。我们还提出了一种方法来减轻这些影响。为验证方法的有效性，我们在多种场景下（包括标准训练、域外数据和样本不平衡分类）使用相同的数据增强技术和计算资源进行了实验。结果表明，我们的方法在所有场景下都优于对比方法在泛化能力和收敛稳定性方面。我们在以下链接提供了代码实现：this https URL。 

---
# Hyperdimensional Intelligent Sensing for Efficient Real-Time Audio Processing on Extreme Edge 

**Title (ZH)**: 超维智能感知在极端边缘高效实时音频处理中的应用 

**Authors**: Sanggeon Yun, Ryozo Masukawa, Hanning Chen, SungHeon Jeong, Wenjun Huang, Arghavan Rezvani, Minhyoung Na, Yoshiki Yamaguchi, Mohsen Imani  

**Link**: [PDF](https://arxiv.org/pdf/2502.10718)  

**Abstract**: The escalating challenges of managing vast sensor-generated data, particularly in audio applications, necessitate innovative solutions. Current systems face significant computational and storage demands, especially in real-time applications like gunshot detection systems (GSDS), and the proliferation of edge sensors exacerbates these issues. This paper proposes a groundbreaking approach with a near-sensor model tailored for intelligent audio-sensing frameworks. Utilizing a Fast Fourier Transform (FFT) module, convolutional neural network (CNN) layers, and HyperDimensional Computing (HDC), our model excels in low-energy, rapid inference, and online learning. It is highly adaptable for efficient ASIC design implementation, offering superior energy efficiency compared to conventional embedded CPUs or GPUs, and is compatible with the trend of shrinking microphone sensor sizes. Comprehensive evaluations at both software and hardware levels underscore the model's efficacy. Software assessments through detailed ROC curve analysis revealed a delicate balance between energy conservation and quality loss, achieving up to 82.1% energy savings with only 1.39% quality loss. Hardware evaluations highlight the model's commendable energy efficiency when implemented via ASIC design, especially with the Google Edge TPU, showcasing its superiority over prevalent embedded CPUs and GPUs. 

**Abstract (ZH)**: 快速增长的传感器生成数据管理挑战，尤其是音频应用领域，需要创新解决方案。当前系统在实时应用如枪声检测系统（GSDS）中面临显著的计算和存储需求，边缘传感器的普及进一步加剧了这些问题。本文提出了一种创新方法，即针对智能音频感知框架的近传感器模型。该模型利用快速傅里叶变换（FFT）模块、卷积神经网络（CNN）层和超维度计算（HDC），在低功耗、快速推断和在线学习方面表现出色。该模型对高效的ASIC设计实现非常适应，并且相比于传统嵌入式CPU或GPU提供更高的能效，同时兼容麦克风传感器尺寸缩小的趋势。软件和硬件层次上的综合评估证实了该模型的有效性。软件评估通过详细的ROC曲线分析表明，在实现82.1%的能耗节省的同时，仅损失了1.39%的质量。硬件评估强调了该模型在ASIC设计实现时的优异能效，尤其是在Google Edge TPU上的表现，展示了其相对于主流嵌入式CPU和GPU的优势。 

---
# FuncGenFoil: Airfoil Generation and Editing Model in Function Space 

**Title (ZH)**: FuncGenFoil：函数空间中的翼型生成与编辑模型 

**Authors**: Jinouwen Zhang, Junjie Ren, Aobo Yang, Yan Lu, Lu Chen, Hairun Xie, Jing Wang, Miao Zhang, Wanli Ouyang, Shixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10712)  

**Abstract**: Aircraft manufacturing is the jewel in the crown of industry, among which generating high-fidelity airfoil geometries with controllable and editable representations remains a fundamental challenge. While existing deep-learning-based methods rely on predefined parametric function families, e.g., Bézier curves and discrete point-based representations, they suffer from inherent trade-offs between expressiveness and resolution flexibility. To tackle this challenge, we introduce FuncGenFoil, a novel function-space generative model that directly learns functional airfoil geometries. Our method inherits both the advantages of arbitrary resolution sampling and the smoothness of parametric functions, as well as the strong expressiveness of discrete point-based functions. Empirical evaluations on the AFBench dataset demonstrate that FuncGenFoil improves upon state-of-the-art methods in airfoil generation by achieving a relative -74.4 label error reduction and +23.2 diversity increase on the AF-200K dataset. Our results highlight the advantages of function-space modeling for aerodynamic shape optimization, offering a powerful and flexible framework for high-fidelity airfoil design. Our code will be released. 

**Abstract (ZH)**: 飞机制造是工业的冠冕，其中生成可控可编辑的高保真翼型几何形状依然是一项基本挑战。现有的基于深度学习的方法依赖于预定义的参数函数族，例如Bézier曲线和离散点表示，它们在表达能力和分辨率灵活性之间存在固有的权衡。为了解决这一挑战，我们引入了FuncGenFoil，这是一种新颖的功能空间生成模型，可以直接学习功能性的翼型几何形状。我们的方法继承了任意分辨率采样和参数函数的平滑性优势，以及离散点表示的强大表达能力。在AFBench数据集上的经验评估表明，FuncGenFoil在气动形状优化方面的气动翼型生成中优于现有方法，在AF-200K数据集上实现了相对标签错误率降低74.4%和多样性提高23.2%。我们的结果突显了功能空间建模在气动形状优化中的优势，提供了高性能和灵活的高保真翼型设计框架。我们的代码将对外开放。 

---
# An Empirical Analysis of Uncertainty in Large Language Model Evaluations 

**Title (ZH)**: 大型语言模型评估中不确定性的一种 empirical 分析 

**Authors**: Qiujie Xie, Qingqiu Li, Zhuohao Yu, Yuejie Zhang, Yue Zhang, Linyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10709)  

**Abstract**: As LLM-as-a-Judge emerges as a new paradigm for assessing large language models (LLMs), concerns have been raised regarding the alignment, bias, and stability of LLM evaluators. While substantial work has focused on alignment and bias, little research has concentrated on the stability of LLM evaluators. In this paper, we conduct extensive experiments involving 9 widely used LLM evaluators across 2 different evaluation settings to investigate the uncertainty in model-based LLM evaluations. We pinpoint that LLM evaluators exhibit varying uncertainty based on model families and sizes. With careful comparative analyses, we find that employing special prompting strategies, whether during inference or post-training, can alleviate evaluation uncertainty to some extent. By utilizing uncertainty to enhance LLM's reliability and detection capability in Out-Of-Distribution (OOD) data, we further fine-tune an uncertainty-aware LLM evaluator named ConfiLM using a human-annotated fine-tuning set and assess ConfiLM's OOD evaluation ability on a manually designed test set sourced from the 2024 Olympics. Experimental results demonstrate that incorporating uncertainty as additional information during the fine-tuning phase can largely improve the model's evaluation performance in OOD scenarios. The code and data are released at: this https URL. 

**Abstract (ZH)**: 作为LLM-as-a-Judge新兴范式用于评估大型语言模型（LLMs），引发了对其一致性和鲁棒性的关注。虽然在一致性和偏见方面已经开展了大量研究，但关于LLM评估器的稳定性研究相对较少。本文通过涉及9个常用LLM评估器的大量实验，探讨了基于模型的LLM评估中的不确定性。我们发现LLM评估器根据模型家族和规模表现出不同的不确定性。通过细致的对比分析，我们发现，在推断或后训练阶段采用特殊的提示策略，可以在一定程度上缓解评估不确定性。通过利用不确定性来增强大型语言模型在分布外（OOD）数据中可靠性和检测能力，我们进一步使用人工标注的微调集fine-tune了一个意识不确定性（Uncertainty-Aware）的LLM评估器ConfiLM，并在2024年奥运会手动设计的测试集上评估了ConfiLM的OOD评估能力。实验结果表明，在fine-tuning阶段引入不确定性作为额外信息可以显著改善模型在OOD场景下的评估性能。代码和数据已发布于：this https URL。 

---
# Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model 

**Title (ZH)**: 读取你的heartbeat：通过预训练的心电语言模型学习心电图单词和句子 

**Authors**: Jiarui Jin, Haoyu Wang, Hongyan Li, Jun Li, Jiahui Pan, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2502.10707)  

**Abstract**: Electrocardiogram (ECG) is essential for the clinical diagnosis of arrhythmias and other heart diseases, but deep learning methods based on ECG often face limitations due to the need for high-quality annotations. Although previous ECG self-supervised learning (eSSL) methods have made significant progress in representation learning from unannotated ECG data, they typically treat ECG signals as ordinary time-series data, segmenting the signals using fixed-size and fixed-step time windows, which often ignore the form and rhythm characteristics and latent semantic relationships in ECG signals. In this work, we introduce a novel perspective on ECG signals, treating heartbeats as words and rhythms as sentences. Based on this perspective, we first designed the QRS-Tokenizer, which generates semantically meaningful ECG sentences from the raw ECG signals. Building on these, we then propose HeartLang, a novel self-supervised learning framework for ECG language processing, learning general representations at form and rhythm levels. Additionally, we construct the largest heartbeat-based ECG vocabulary to date, which will further advance the development of ECG language processing. We evaluated HeartLang across six public ECG datasets, where it demonstrated robust competitiveness against other eSSL methods. Our data and code are publicly available at this https URL. 

**Abstract (ZH)**: 心电图（ECG）对于心律失常和其他心脏疾病的临床诊断至关重要，但由于需要高质量的注释，基于ECG的深度学习方法常常面临限制。尽管之前的心电图自我监督学习（eSSL）方法在无标注ECG数据的表征学习方面取得了显著进展，但它们通常将ECG信号视为普通的时序数据，使用固定大小和固定步长的时间窗口对信号进行分割，这往往忽略了ECG信号中的形态、节奏特征及其潜在语义关系。在本工作中，我们从一个新的视角来审视ECG信号，将心跳视为“词”，节奏视为“句子”。基于这一视角，我们首先设计了QRS-Tokenizing模块，该模块从原始ECG信号中生成语义上有意义的ECG句子。在此基础上，我们提出了HeartLang，一种新颖的心电图自我监督学习框架，用于ECG语言处理，学习形态和节奏层面的通用表征。此外，我们构建了迄今为止最大的基于心跳的心电图词汇表，这将进一步促进心电图语言处理的发展。我们在六个公开的心电图数据集中评估了HeartLang，其表现出色，与其它eSSL方法具有较强的竞争力。我们的数据和代码可在以下网址获取：this https URL。 

---
# Raising the Bar in Graph OOD Generalization: Invariant Learning Beyond Explicit Environment Modeling 

**Title (ZH)**: 提高图数据OOD泛化的标准：超越显式环境建模的不变性学习 

**Authors**: Xu Shen, Yixin Liu, Yili Wang, Rui Miao, Yiwei Dai, Shirui Pan, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10706)  

**Abstract**: Out-of-distribution (OOD) generalization has emerged as a critical challenge in graph learning, as real-world graph data often exhibit diverse and shifting environments that traditional models fail to generalize across. A promising solution to address this issue is graph invariant learning (GIL), which aims to learn invariant representations by disentangling label-correlated invariant subgraphs from environment-specific subgraphs. However, existing GIL methods face two major challenges: (1) the difficulty of capturing and modeling diverse environments in graph data, and (2) the semantic cliff, where invariant subgraphs from different classes are difficult to distinguish, leading to poor class separability and increased misclassifications. To tackle these challenges, we propose a novel method termed Multi-Prototype Hyperspherical Invariant Learning (MPHIL), which introduces two key innovations: (1) hyperspherical invariant representation extraction, enabling robust and highly discriminative hyperspherical invariant feature extraction, and (2) multi-prototype hyperspherical classification, which employs class prototypes as intermediate variables to eliminate the need for explicit environment modeling in GIL and mitigate the semantic cliff issue. Derived from the theoretical framework of GIL, we introduce two novel objective functions: the invariant prototype matching loss to ensure samples are matched to the correct class prototypes, and the prototype separation loss to increase the distinction between prototypes of different classes in the hyperspherical space. Extensive experiments on 11 OOD generalization benchmark datasets demonstrate that MPHIL achieves state-of-the-art performance, significantly outperforming existing methods across graph data from various domains and with different distribution shifts. 

**Abstract (ZH)**: 超越分布外（OOD）泛化的图不变学习：多原型超球面不变学习（MPHIL） 

---
# Occlusion-aware Non-Rigid Point Cloud Registration via Unsupervised Neural Deformation Correntropy 

**Title (ZH)**: 基于无监督神经变形核函数的 occlusion-aware 非刚性点云配准 

**Authors**: Mingyang Zhao, Gaofeng Meng, Dong-Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10704)  

**Abstract**: Non-rigid alignment of point clouds is crucial for scene understanding, reconstruction, and various computer vision and robotics tasks. Recent advancements in implicit deformation networks for non-rigid registration have significantly reduced the reliance on large amounts of annotated training data. However, existing state-of-the-art methods still face challenges in handling occlusion scenarios. To address this issue, this paper introduces an innovative unsupervised method called Occlusion-Aware Registration (OAR) for non-rigidly aligning point clouds. The key innovation of our method lies in the utilization of the adaptive correntropy function as a localized similarity measure, enabling us to treat individual points distinctly. In contrast to previous approaches that solely minimize overall deviations between two shapes, we combine unsupervised implicit neural representations with the maximum correntropy criterion to optimize the deformation of unoccluded regions. This effectively avoids collapsed, tearing, and other physically implausible results. Moreover, we present a theoretical analysis and establish the relationship between the maximum correntropy criterion and the commonly used Chamfer distance, highlighting that the correntropy-induced metric can be served as a more universal measure for point cloud analysis. Additionally, we introduce locally linear reconstruction to ensure that regions lacking correspondences between shapes still undergo physically natural deformations. Our method achieves superior or competitive performance compared to existing approaches, particularly when dealing with occluded geometries. We also demonstrate the versatility of our method in challenging tasks such as large deformations, shape interpolation, and shape completion under occlusion disturbances. 

**Abstract (ZH)**: 非刚性点云对齐对于场景理解、重建以及各种计算机视觉和机器人任务至关重要。隐式变形网络的最新进展显著减少了对大量标注训练数据的依赖。然而，现有最先进的方法在处理遮挡场景时仍面临挑战。为解决这一问题，本文提出了一种新的无监督方法——感知遮挡对齐（OAR），用于非刚性对齐点云。该方法的关键创新在于利用自适应误差核函数作为局部相似性度量，使得点可以被单独处理。与仅最小化两个形状总体偏差的先前方法不同，我们结合无监督隐式神经表示和最大误差核准则来优化未遮挡区域的变形，从而避免了坍塌、撕裂等物理上不可能的结果。此外，我们提供了理论分析，建立了最大误差核准则与常用的切线距离之间的关系，指出误差核诱导的度量可以作为点云分析的更通用度量。我们还引入了局部线性重建，确保缺乏形状对应区域仍能进行自然变形。与现有方法相比，我们的方法在处理遮挡几何形状时表现出优越或竞争性能，同时在复杂任务如大形变、形状插值和遮挡干扰下的形状补全方面展示了其灵活性。 

---
# Exploring Synaptic Resonance in Large Language Models: A Novel Approach to Contextual Memory Integration 

**Title (ZH)**: 探索大型语言模型中的突触共振：一种全新的上下文记忆整合方法 

**Authors**: George Applegarth, Christian Weatherstone, Maximilian Hollingsworth, Henry Middlebrook, Marcus Irvin  

**Link**: [PDF](https://arxiv.org/pdf/2502.10699)  

**Abstract**: Contextual memory integration remains a high challenge in the development of language models, particularly in tasks that require maintaining coherence over extended sequences. Traditional approaches, such as self-attention mechanisms and memory-augmented architectures, often prioritize short-term dependencies, leading to fragmentation and inconsistency in long-range contextual understanding. Inspired by principles of synaptic plasticity observed in biological neural systems, a novel mechanism, Synaptic Resonance, is introduced to dynamically reinforce relevant memory pathways during training and inference. Unlike static memory representations, this mechanism continuously adjusts synaptic weight matrices based on contextual relevance, allowing for improved information retention without excessive computational overhead. Evaluations conducted on an open-source language model demonstrate reductions in perplexity, enhancements in contextual coherence, and increased robustness against input noise, highlighting the effectiveness of reinforcement-driven memory modulation. Comparative analysis against baseline models further reveals that the proposed approach achieves higher memory retention efficiency while maintaining computational feasibility. The architectural modifications integrate seamlessly into existing transformer-based frameworks, ensuring stable convergence and efficient inference without sacrificing scalability. Applications benefiting from improved long-term contextual consistency, such as dialogue systems and document summarization, stand to gain from this approach. Empirical findings suggest that dynamically reinforced memory pathways offer a promising alternative to conventional memory mechanisms, addressing longstanding limitations in extended sequence modeling. 

**Abstract (ZH)**: Contextual记忆整合仍然是语言模型发展中的一项高挑战，特别是在需要在长序列上保持连贯性的任务中。传统的approaches，如自我注意力机制和记忆增强架构，往往优先处理短时依赖性，导致长距离上下文理解上的分割和不一致性。受生物神经系统中突触可塑性原理的启发，提出了一种新的机制——突触共振，该机制在训练和推理过程中动态强化相关记忆路径。与静态记忆表示不同，这种机制根据上下文相关性连续调整突触权重矩阵，从而在不增加过多计算开销的情况下提高信息保留能力。在开源语言模型上的评估显示，在困惑度、上下文连贯性以及对输入噪声的鲁棒性方面都有改进，突显了强化驱动的记忆调制的有效性。与基线模型的对比分析进一步表明，所提出的方法在记忆保留效率方面表现更优，同时保持了计算上的可行性。该架构修改无缝集成到现有的transformer基架构中，确保了稳定收敛和高效推理，无需牺牲可扩展性。受益于改进的长时上下文一致性的应用，如对话系统和文档摘要，可以从这种方法中获益。实证研究结果表明，动态强化的记忆路径为解决长序列建模中的长期局限问题提供了有前景的替代方案。 

---
# Superpose Singular Features for Model Merging 

**Title (ZH)**: 合并模型时叠加奇异特征 

**Authors**: Haiquan Qiu, You Wu, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.10698)  

**Abstract**: Model merging is a critical technique for combining the capabilities of multiple fine-tuned models without requiring additional training. While existing methods treat parameters as vectors, they overlook the intrinsic structure of linear transformation matrices - the core components that comprise the majority of model parameters. These matrices are fundamental to neural networks, mapping input representations to output features through linear combinations. Motivated by the linear representation hypothesis, we introduce task matrix and propose to Superpose Features from Task Matrix (SFTM), a novel approach that superposes features from individual task models into a merged model. SFTM employs singular value decomposition to identify feature bases of linear transformation matrices and solves a linear system to optimally combine them while preserving input-output mappings from individual task models. Extensive experiments on vision transformers and language models demonstrate that our method consistently outperforms existing methods, achieving superior performance and enhanced out-of-distribution generalization. 

**Abstract (ZH)**: 模型合并是结合多个细调模型的能力的关键技术，无需额外训练。虽然现有方法将参数视为向量，但它们忽略了线性变换矩阵的内在结构——这些矩阵构成了模型参数的主要部分。线性变换矩阵对神经网络至关重要，它们通过线性组合将输入表示映射到输出特征。受线性表示假说的启发，我们引入了任务矩阵，并提出了一种新的方法——基于任务矩阵叠加特征（SFTM），该方法将单个任务模型的特征叠加到合并模型中。SFTM利用奇异值分解识别线性变换矩阵的特征基，并通过求解线性系统以最优方式将它们组合起来，同时保留单个任务模型的输入-输出映射。在视觉变压器和语言模型上的广泛实验表明，我们的方法在所有方法中表现最好，实现了一流的性能和增强的离分布泛化能力。 

---
# Simulations of Common Unsupervised Domain Adaptation Algorithms for Image Classification 

**Title (ZH)**: 常用无监督领域适应算法在图像分类中的模拟研究 

**Authors**: Ahmad Chaddad, Yihang Wu, Yuchen Jiang, Ahmed Bouridane, Christian Desrosiers  

**Link**: [PDF](https://arxiv.org/pdf/2502.10694)  

**Abstract**: Traditional machine learning assumes that training and test sets are derived from the same distribution; however, this assumption does not always hold in practical applications. This distribution disparity can lead to severe performance drops when the trained model is used in new data sets. Domain adaptation (DA) is a machine learning technique that aims to address this problem by reducing the differences between domains. This paper presents simulation-based algorithms of recent DA techniques, mainly related to unsupervised domain adaptation (UDA), where labels are available only in the source domain. Our study compares these techniques with public data sets and diverse characteristics, highlighting their respective strengths and drawbacks. For example, Safe Self-Refinement for Transformer-based DA (SSRT) achieved the highest accuracy (91.6\%) in the office-31 data set during our simulations, however, the accuracy dropped to 72.4\% in the Office-Home data set when using limited batch sizes. In addition to improving the reader's comprehension of recent techniques in DA, our study also highlights challenges and upcoming directions for research in this domain. The codes are available at this https URL. 

**Abstract (ZH)**: 传统的机器学习假设训练集和测试集来自相同的分布；然而，在实际应用中这一假设并不总是成立。这种分布差异会导致在新数据集上使用训练好的模型时出现严重的性能下降。领域适应（DA）是一种机器学习技术，旨在通过减少领域之间的差异来解决这一问题。本文介绍了基于仿真的recent DA技术算法，主要涉及无监督领域适应（UDA），其中仅源域有标签。我们的研究使用公共数据集和多种特性，比较了这些技术的优劣。例如，在我们的仿真实验中，Safe Self-Refinement for Transformer-based DA (SSRT) 在office-31数据集上的准确率为91.6%，但在使用有限批次大小时，Office-Home数据集的准确率降至72.4%。除了提高读者对DA领域近期技术的理解，我们的研究还指出了该领域的挑战和未来研究方向。相关代码可在以下网址获取。 

---
# Self-Explaining Hypergraph Neural Networks for Diagnosis Prediction 

**Title (ZH)**: 自我解释超图神经网络在诊断预测中的应用 

**Authors**: Leisheng Yu, Yanxiao Cai, Minxing Zhang, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10689)  

**Abstract**: The burgeoning volume of electronic health records (EHRs) has enabled deep learning models to excel in predictive healthcare. However, for high-stakes applications such as diagnosis prediction, model interpretability remains paramount. Existing deep learning diagnosis prediction models with intrinsic interpretability often assign attention weights to every past diagnosis or hospital visit, providing explanations lacking flexibility and succinctness. In this paper, we introduce SHy, a self-explaining hypergraph neural network model, designed to offer personalized, concise and faithful explanations that allow for interventions from clinical experts. By modeling each patient as a unique hypergraph and employing a message-passing mechanism, SHy captures higher-order disease interactions and extracts distinct temporal phenotypes as personalized explanations. It also addresses the incompleteness of the EHR data by accounting for essential false negatives in the original diagnosis record. A qualitative case study and extensive quantitative evaluations on two real-world EHR datasets demonstrate the superior predictive performance and interpretability of SHy over existing state-of-the-art models. 

**Abstract (ZH)**: 电子健康记录(EHRs)的快速增长使得深度学习模型在预测健康-care领域表现出色。然而，在诊断预测等高风险应用中，模型的可解释性仍然至关重要。现有具备固有可解释性的深度学习诊断预测模型往往为每个过去的诊断或医院访问分配注意力权重，提供的解释缺乏灵活性和简洁性。本文提出了SHy，一种自解释的超图神经网络模型，旨在提供个性化、简洁且忠实的解释，使临床专家能够介入。通过将每位患者建模为独特的超图，并采用消息传递机制，SHy捕捉了更高阶的疾病交互，并提取了个性化的时间表型作为解释。此外，它通过考虑原始诊断记录中的关键假阴性解决了EHR数据的不完备性问题。实世界两个EHR数据集上的定性和定量评估表明，SHy在预测性能和可解释性方面优于现有最先进的模型。 

---
# GenComUI: Exploring Generative Visual Aids as Medium to Support Task-Oriented Human-Robot Communication 

**Title (ZH)**: GenComUI: 探索生成型视觉辅助作为支持任务导向人机通信的媒介 

**Authors**: Yate Ge, Meiying Li, Xipeng Huang, Yuanda Hu, Qi Wang, Xiaohua Sun, Weiwei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.10678)  

**Abstract**: This work investigates the integration of generative visual aids in human-robot task communication. We developed GenComUI, a system powered by large language models that dynamically generates contextual visual aids (such as map annotations, path indicators, and animations) to support verbal task communication and facilitate the generation of customized task programs for the robot. This system was informed by a formative study that examined how humans use external visual tools to assist verbal communication in spatial tasks. To evaluate its effectiveness, we conducted a user experiment (n = 20) comparing GenComUI with a voice-only baseline. The results demonstrate that generative visual aids, through both qualitative and quantitative analysis, enhance verbal task communication by providing continuous visual feedback, thus promoting natural and effective human-robot communication. Additionally, the study offers a set of design implications, emphasizing how dynamically generated visual aids can serve as an effective communication medium in human-robot interaction. These findings underscore the potential of generative visual aids to inform the design of more intuitive and effective human-robot communication, particularly for complex communication scenarios in human-robot interaction and LLM-based end-user development. 

**Abstract (ZH)**: 本研究探讨了生成性视觉辅助在人类与机器人任务通信中的整合。我们开发了由大规模语言模型驱动的GenComUI系统，该系统能够动态生成上下文视觉辅助（如地图标注、路径指示和动画），以支持口头任务通信并为机器人生成定制化任务程序提供便利。该系统基于一项形成性研究的指导，该研究探讨了人类如何使用外部视觉工具来辅助空间任务中的口头通信。为了评估其 effectiveness，我们进行了一个包含20名用户的用户实验，将GenComUI与仅语音基线进行比较。研究结果表明，生成性视觉辅助通过定性和定量分析提供了持续的视觉反馈，从而促进了自然且有效的的人机通信。此外，本研究还提供了一组设计启示，强调动态生成的视觉辅助在人机交互中的有效通信媒介作用。这些发现强调了生成性视觉辅助在设计更具直观性和有效性的交互式人机通信方面的潜力，特别是在人机交互和基于LLM的用户开发中的复杂通信场景。 

---
# Proof of Response 

**Title (ZH)**: 响应证明 

**Authors**: Illia Polosukhin, Alex Skidanov  

**Link**: [PDF](https://arxiv.org/pdf/2502.10637)  

**Abstract**: We present a mechanism that for a network of participants allows one participant of the network (Alice) to request some data from another participant (Bob) and either receive a response from Bob within a known-in-advance, bounded time b, or receive a proof that at least one edge on the way to Bob was broken within b, or receive a streaming payment proportional to time passed beyond b during which neither was received. This mechanism allows for building downstream applications that require provable responses from other participants, such as decentralized storage solutions, decentralized AI agents, and more. 

**Abstract (ZH)**: 我们提出了一种机制，该机制允许可信网络中的一个参与者（Alice）向另一个参与者（Bob）请求某些数据，并在已知且受限的时间b内要么收到Bob的响应，要么收到证明至少一条通往Bob的路径在b时间内被中断的证据，要么收到按时间比例支付的流式付款，其中既未收到响应也未收到证据。该机制使得能够构建需要从其他参与者处获得可验证响应的下游应用，如去中心化存储解决方案、去中心化AI代理等。 

---
# ControllableGPT: A Ground-Up Designed Controllable GPT for Molecule Optimization 

**Title (ZH)**: 可控GPT：一种从底层设计的可控GPT分子优化模型 

**Authors**: Xuefeng Liu, Songhao Jiang, Bo Li, Rick Stevens  

**Link**: [PDF](https://arxiv.org/pdf/2502.10631)  

**Abstract**: Large Language Models (LLMs) employ three popular training approaches: Masked Language Models (MLM), Causal Language Models (CLM), and Sequence-to-Sequence Models (seq2seq). However, each approach has its strengths and limitations, and faces challenges in addressing specific tasks that require controllable and bidirectional generation, such as drug optimization. To address this challenge, inspired by the biological processes of growth and evolution, which involve the expansion, shrinking, and mutation of sequences, we introduce ControllableGPT. This initiative represents the first effort to combine the advantages of MLM, CLM, and seq2seq into a single unified, controllable GPT framework. It enables the precise management of specific locations and ranges within a sequence, allowing for expansion, reduction, or mutation over chosen or random lengths, while maintaining the integrity of any specified positions or subsequences. In this work, we designed ControllableGPT for drug optimization from the ground up, which included proposing the Causally Masked Seq2seq (CMS) objective, developing the training corpus, introducing a novel pre-training approach, and devising a unique generation process. We demonstrate the effectiveness and controllability of ControllableGPT by conducting experiments on drug optimization tasks for both viral and cancer benchmarks, surpassing competing baselines. 

**Abstract (ZH)**: 大型语言模型（LLMs）采用三种流行的训练方法：掩码语言模型（MLM）、因导语言模型（CLM）和序列到序列模型（seq2seq）。然而，每种方法都有其优势和局限性，面对需要可控和双向生成的任务时面临挑战，如药物优化。为解决这一挑战，受生物生长和进化过程的启发，涉及序列的扩展、收缩和变异，我们提出了ControllableGPT。这一举措代表了将MLM、CLM和seq2seq的优势结合到一个统一的可控GPT框架中的首次尝试。它允许精确管理序列中的特定位置和范围，允许在选定的或随机长度上进行扩展、收缩或变异，同时保持任何指定位置或子序列的完整性。在这项工作中，我们从头开始为药物优化设计了ControllableGPT，包括提出因果掩码序列到序列（CMS）目标、开发训练语料库、引入新的预训练方法以及设计独特的生成过程。我们通过在病毒和癌症基准测试中的药物优化任务实验展示了ControllableGPT的有效性和可控性，超过了竞争对手的基础模型。 

---
# K-Edit: Language Model Editing with Contextual Knowledge Awareness 

**Title (ZH)**: K-Edit：具有上下文知识awareness的语言模型编辑 

**Authors**: Elan Markowitz, Anil Ramakrishna, Ninareh Mehrabi, Charith Peris, Rahul Gupta, Kai-Wei Chang, Aram Galstyan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10626)  

**Abstract**: As the world changes, we need to be able to update our models and correct false information without costly retraining. Knowledge-based model editing enables precise modifications to the weights of large language models in order to modify the information encoded within. Recent approaches have seen success in enabling recall of edited information for thousands of edits at once. However, these approaches fail to produce edits that account for associated contextual information. We present K-Edit, an effective approach to generating contextually consistent knowledge edits. By using knowledge graphs, which maintain contextual consistency when an edge is edited, we are able to generate additional \textit{contextual edits} that ensure consistency of related information in the language model. Our experiments demonstrate significant improvements in multi-hop question answering while maintaining the general effectiveness and scalability of model edits. 

**Abstract (ZH)**: 基于知识的模型编辑使得我们能够在不需要昂贵重新训练的情况下更新模型并修正错误信息，从而实现对大规模语言模型权重的精确调整，以修改编码的信息。最近的方法在一次性编辑数千次信息时能够启用编辑信息的回忆。然而，这些方法未能生成考虑到相关上下文信息的编辑。我们提出了K-Edit，一种生成上下文一致的知识编辑的有效方法。通过使用保持边编辑时上下文一致性的知识图谱，我们能够生成额外的上下文编辑，确保语言模型中相关信息的一致性。我们的实验显示，在多跳问答方面取得了显著改进，同时保持了模型编辑的一般有效性和可扩展性。 

---
# Network evasion detection with Bi-LSTM model 

**Title (ZH)**: 基于Bi-LSTM模型的网络逃逸检测 

**Authors**: Kehua Chen, Jingping Jia  

**Link**: [PDF](https://arxiv.org/pdf/2502.10624)  

**Abstract**: Network evasion detection aims to distinguish whether the network flow comes from link layer exists network evasion threat, which is a means to disguise the data traffic on detection system by confusing the signature. Since the previous research works has all sorts of frauds, we propose a architecture with deep learning network to handle this problem. In this paper, we extract the critical information as key features from data frame and also specifically propose to use bidirectional long short-term memory (Bi-LSTM) neural network which shows an outstanding performance to trace the serial information, to encode both the past and future trait on the network flows. Furthermore we introduce a classifier named Softmax at the bottom of Bi-LSTM, holding a character to select the correct class. All experiments results shows that we can achieve a significant performance with a deep Bi-LSTM in network evasion detection and it's average accuracy reaches 96.1%. 

**Abstract (ZH)**: 网络逃逸检测旨在区分网络流是否来自链路层存在的网络逃逸威胁，这是一种通过混淆特征使数据流量在检测系统中蒙混过关的方法。鉴于以往的研究工作存在诸多不足，我们提出了一种基于深度学习网络的架构来解决这一问题。在此论文中，我们从数据帧中提取关键信息作为特征，并特别提出使用双向长短期记忆（Bi-LSTM）神经网络来追踪序列信息，编码网络流中的过去和未来特征。此外，我们在Bi-LSTM底部引入了一个名为Softmax的分类器，以选择正确的类别。所有实验结果表明，我们可以通过深度Bi-LSTM在网络逃逸检测中实现显著性能提升，平均准确率达到了96.1%。 

---
# Optimizing CNN Architectures for Advanced Thoracic Disease Classification 

**Title (ZH)**: 优化CNN架构以进行高级胸腔疾病分类 

**Authors**: Tejas Mirthipati  

**Link**: [PDF](https://arxiv.org/pdf/2502.10614)  

**Abstract**: Machine learning, particularly convolutional neural networks (CNNs), has shown promise in medical image analysis, especially for thoracic disease detection using chest X-ray images. In this study, we evaluate various CNN architectures, including binary classification, multi-label classification, and ResNet50 models, to address challenges like dataset imbalance, variations in image quality, and hidden biases. We introduce advanced preprocessing techniques such as principal component analysis (PCA) for image compression and propose a novel class-weighted loss function to mitigate imbalance issues. Our results highlight the potential of CNNs in medical imaging but emphasize that issues like unbalanced datasets and variations in image acquisition methods must be addressed for optimal model performance. 

**Abstract (ZH)**: 机器学习，特别是卷积神经网络（CNNs），在医疗图像分析中展现出潜力，特别是在使用胸部X射线图像检测胸腔疾病方面的应用。本研究评估了多种CNN架构，包括二元分类、多标签分类和ResNet50模型，以应对数据集不平衡、图像质量差异和隐藏偏见等挑战。我们引入了高级预处理技术，如主成分分析（PCA）进行图像压缩，并提出了一种新颖的类别加权损失函数以缓解不平衡问题。我们的结果强调了CNN在医疗成像中的潜力，但也指出了必须解决数据集不平衡和图像采集方法差异等问题以实现最佳模型性能。 

---
# Post-training an LLM for RAG? Train on Self-Generated Demonstrations 

**Title (ZH)**: 针对RAG的后训练LLM？在自动生成的示范上训练 

**Authors**: Matthew Finlayson, Ilia Kulikov, Daneil M. Bikel, Barlas Oguz, Xilun Chen, Aasish Pappu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10596)  

**Abstract**: Large language models (LLMs) often struggle with knowledge intensive NLP tasks, such as answering "Who won the latest World Cup?" because the knowledge they learn during training may be insufficient or outdated. Conditioning generation on retrieved documents -- a technique known as retrieval augmented generation (RAG) -- mitigates these shortcomings by allowing the model to leverage in-context information. Practitioners can improve LLM RAG performance by fine-tuning on retrieval-augmented instructions, but must beware that this can cause undesirable model behaviors like hallucinations. We attribute this degradation to the fact that the training data is likely to be out-of-distribution for the model and may suffer from quality issues, such as misalignment between retrievals and target responses (since retrievals are frequently added post-hoc). We propose a recipe for training RAG-enabled LLMs using self-generated demonstrations, thereby avoiding training on out-of-distribution text and integrating retrievals into the LLM responses. We evaluate our method on knowledge intensive question answering (QA) tasks and show that our method teaches LLMs to properly handle in-context retrievals and abstain from questions it will likely get wrong. Compared to conventional RA-IT methods, our method prevents model degradation in non-RAG settings while exhibiting superior QA performance. 

**Abstract (ZH)**: 基于检索增强生成的大语言模型在知识密集型NLP任务中的训练方法研究 

---
# Towards Self-Supervised Covariance Estimation in Deep Heteroscedastic Regression 

**Title (ZH)**: 面向深度异方差回归的自监督协方差估计 

**Authors**: Megh Shukla, Aziz Shameem, Mathieu Salzmann, Alexandre Alahi  

**Link**: [PDF](https://arxiv.org/pdf/2502.10587)  

**Abstract**: Deep heteroscedastic regression models the mean and covariance of the target distribution through neural networks. The challenge arises from heteroscedasticity, which implies that the covariance is sample dependent and is often unknown. Consequently, recent methods learn the covariance through unsupervised frameworks, which unfortunately yield a trade-off between computational complexity and accuracy. While this trade-off could be alleviated through supervision, obtaining labels for the covariance is non-trivial. Here, we study self-supervised covariance estimation in deep heteroscedastic regression. We address two questions: (1) How should we supervise the covariance assuming ground truth is available? (2) How can we obtain pseudo labels in the absence of the ground-truth? We address (1) by analysing two popular measures: the KL Divergence and the 2-Wasserstein distance. Subsequently, we derive an upper bound on the 2-Wasserstein distance between normal distributions with non-commutative covariances that is stable to optimize. We address (2) through a simple neighborhood based heuristic algorithm which results in surprisingly effective pseudo labels for the covariance. Our experiments over a wide range of synthetic and real datasets demonstrate that the proposed 2-Wasserstein bound coupled with pseudo label annotations results in a computationally cheaper yet accurate deep heteroscedastic regression. 

**Abstract (ZH)**: 深层异方差回归模型通过神经网络模拟目标分布的均值和协方差。挑战来自于异方差性，这意味着协方差是样本依赖的，且通常未知。因此，最近的方法通过无监督框架学习协方差，不幸的是，这会导致计算复杂性和准确性之间的权衡。虽然可以通过监督来缓解这一权衡，但获得协方差的标签并不容易。在这里，我们研究深层异方差回归中的自监督协方差估计。我们回答了两个问题：（1）如果我们有地面真实标签，应该如何监督协方差？（2）在没有地面真实标签的情况下，如何获得伪标签？我们通过分析两种流行的距离度量：KL散度和2- Wasserstein距离来解决（1）。随后，我们推导出一种稳定的非交换协方差正态分布之间的2-Wasserstein距离的上界。我们通过一个简单的基于邻域的启发式算法解决（2），该算法生成了对协方差非常有效的伪标签。我们在广泛合成和真实数据集上的实验表明，所提出的2-Wasserstein界结合伪标签注解导致一种计算成本更低但准确率更高的深层异方差回归。 

---
# Do We Need to Verify Step by Step? Rethinking Process Supervision from a Theoretical Perspective 

**Title (ZH)**: 我们是否需要逐步骤验证？从理论视角重新思考过程监督 

**Authors**: Zeyu Jia, Alexander Rakhlin, Tengyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.10581)  

**Abstract**: As large language models have evolved, it has become crucial to distinguish between process supervision and outcome supervision -- two key reinforcement learning approaches to complex reasoning tasks. While process supervision offers intuitive advantages for long-term credit assignment, the precise relationship between these paradigms has remained an open question. Conventional wisdom suggests that outcome supervision is fundamentally more challenging due to the trajectory-level coverage problem, leading to significant investment in collecting fine-grained process supervision data.
In this paper, we take steps towards resolving this debate. Our main theorem shows that, under standard data coverage assumptions, reinforcement learning through outcome supervision is no more statistically difficult than through process supervision, up to polynomial factors in horizon. At the core of this result lies the novel Change of Trajectory Measure Lemma -- a technical tool that bridges return-based trajectory measure and step-level distribution shift. Furthermore, for settings with access to a verifier or a rollout capability, we prove that any policy's advantage function can serve as an optimal process reward model, providing a direct connection between outcome and process supervision. These findings suggest that the empirically observed performance gap -- if any -- between outcome and process supervision likely stems from algorithmic limitations rather than inherent statistical difficulties, potentially transforming how we approach data collection and algorithm design for reinforcement learning. 

**Abstract (ZH)**: 随着大型语言模型的发展，区分过程监督与结果监督——两种关键的强化学习方法以应对复杂推理任务——变得至关重要。虽然过程监督在长期信用分配方面提供了直观的优势，但这些范式的精确关系仍是一个开放的问题。传统观点认为，结果监督由于轨迹级覆盖问题而从根本上更加具有挑战性，导致人们对精细过程监督数据的投资显著增加。

在本文中，我们朝着解决这一争论迈出了步伐。我们的主要定理表明，在标准数据覆盖假设下，通过结果监督进行强化学习在统计上与通过过程监督进行的强化学习相比，最多只是在计算复杂性上存在多项式级别的差异。这一结果的核心是新颖的轨迹测度变换引理——一个技术工具，它将基于回报的轨迹测度与步骤级分布变化连接起来。此外，在可以访问验证器或展开能力的情况下，我们证明任何策略的优势函数都可以作为最优过程奖励模型，从而直接连接结果监督与过程监督。这些发现表明，如果存在，结果监督与过程监督之间观察到的性能差距可能源于算法限制而非固有的统计困难，这可能会改变我们对强化学习中数据收集和算法设计方法的思路。 

---
# Man Made Language Models? Evaluating LLMs' Perpetuation of Masculine Generics Bias 

**Title (ZH)**: 人造语言模型？评价大语言模型延续男性代称偏见的情况 

**Authors**: Enzo Doyen, Amalia Todirascu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10577)  

**Abstract**: Large language models (LLMs) have been shown to propagate and even amplify gender bias, in English and other languages, in specific or constrained contexts. However, no studies so far have focused on gender biases conveyed by LLMs' responses to generic instructions, especially with regard to masculine generics (MG). MG are a linguistic feature found in many gender-marked languages, denoting the use of the masculine gender as a "default" or supposedly neutral gender to refer to mixed group of men and women, or of a person whose gender is irrelevant or unknown. Numerous psycholinguistics studies have shown that MG are not neutral and induce gender bias. This work aims to analyze the use of MG by both proprietary and local LLMs in responses to generic instructions and evaluate their MG bias rate. We focus on French and create a human noun database from existing lexical resources. We filter existing French instruction datasets to retrieve generic instructions and analyze the responses of 6 different LLMs. Overall, we find that $\approx$39.5\% of LLMs' responses to generic instructions are MG-biased ($\approx$73.1\% across responses with human nouns). Our findings also reveal that LLMs are reluctant to using gender-fair language spontaneously. 

**Abstract (ZH)**: 大型语言模型（LLMs）在回应通用指令时传递性别偏见的研究：以法语为例 

---
# An Innovative Next Activity Prediction Approach Using Process Entropy and DAW-Transformer 

**Title (ZH)**: 基于过程熵和DAW-Transformer的创新下一步活动预测方法 

**Authors**: Hadi Zare, Mostafa Abbasi, Maryam Ahang, Homayoun Najjaran  

**Link**: [PDF](https://arxiv.org/pdf/2502.10573)  

**Abstract**: Purpose - In Business Process Management (BPM), accurate prediction of the next activities is vital for operational efficiency and decision-making. Current Artificial Intelligence (AI)/Machine Learning (ML) models struggle with the complexity and evolving nature of business process event logs, balancing accuracy and interpretability. This paper proposes an entropy-driven model selection approach and DAW-Transformer, which stands for Dynamic Attribute-Aware Transformer, to integrate all attributes with a dynamic window for better accuracy.
Design/methodology/approach - This paper introduces a novel next-activity prediction approach that uses process entropy to assess the complexity of event logs and dynamically select the most suitable ML model. A new transformer-based architecture with multi-head attention and dynamic windowing mechanism, DAW-Transformer, is proposed to capture long-range dependencies and utilize all relevant event log attributes. Experiments were conducted on six public datasets, and the performance was evaluated with process entropy.
Finding - The results demonstrate the effectiveness of the approach across these publicly available datasets. DAW-Transformer achieved superior performance, especially on high-entropy datasets such as Sepsis exceeding Limited window Multi-Transformers by 4.69% and a benchmark CNN-LSTM-SAtt model by 3.07%. For low-entropy datasets like Road Traffic Fine, simpler, more interpretable algorithms like Random Forest performed nearly as well as the more complex DAW-Transformer and offered better handling of imbalanced data and improved explainability.
Originality/ value - This work's novelty lies in the proposed DAW-Transformer, with a dynamic window and considering all relevant attributes. Also, entropy-driven selection methods offer a robust, accurate, and interpretable solution for next-activity prediction. 

**Abstract (ZH)**: 目的 - 在业务流程管理（BPM）中，准确预测下一个活动对于操作效率和决策制定至关重要。当前的人工智能/机器学习（AI/ML）模型难以处理业务流程事件日志的复杂性和不断变化的特性，难以在准确性和可解释性之间取得平衡。本文提出了一种基于熵的模型选择方法和DAW-Transformer，即动态属性感知变换器，以更好地利用所有相关的事件日志属性。

设计/方法 - 本文提出了一种新颖的下一个活动预测方法，使用过程熵来评估事件日志的复杂性，并动态选择最合适的机器学习模型。提出了一种基于变换器的新架构，该架构具有多头注意机制和动态窗口机制，称为DAW-Transformer，以捕获长程依赖性并利用所有相关事件日志属性。在六个公开数据集上进行了实验，并使用过程熵评估了模型性能。

发现 - 结果表明该方法在这些公开可用的数据集上具有有效性。DAW-Transformer在高熵数据集（例如Sepsis）上表现优异，超过了有限窗口多变换器4.69%，并优于基准的CNN-LSTM-SAtt模型3.07%。对于低熵数据集（如Road Traffic Fine），更简单且更具可解释性的算法如随机森林表现几乎与复杂的DAW-Transformer相当，且在处理不平衡数据和提高可解释性方面更具优势。

创新/价值 - 本文的创新之处在于提出的具有动态窗口和考虑所有相关属性的DAW-Transformer。此外，基于熵的模型选择方法为下一个活动预测提供了稳健、准确且可解释的解决方案。 

---
# HADL Framework for Noise Resilient Long-Term Time Series Forecasting 

**Title (ZH)**: HADL框架下的抗噪声长期时间序列预测 

**Authors**: Aditya Dey, Jonas Kusch, Fadi Al Machot  

**Link**: [PDF](https://arxiv.org/pdf/2502.10569)  

**Abstract**: Long-term time series forecasting is critical in domains such as finance, economics, and energy, where accurate and reliable predictions over extended horizons drive strategic decision-making. Despite the progress in machine learning-based models, the impact of temporal noise in extended lookback windows remains underexplored, often degrading model performance and computational efficiency. In this paper, we propose a novel framework that addresses these challenges by integrating the Discrete Wavelet Transform (DWT) and Discrete Cosine Transform (DCT) to perform noise reduction and extract robust long-term features. These transformations enable the separation of meaningful temporal patterns from noise in both the time and frequency domains. To complement this, we introduce a lightweight low-rank linear prediction layer that not only reduces the influence of residual noise but also improves memory efficiency. Our approach demonstrates competitive robustness to noisy input, significantly reduces computational complexity, and achieves competitive or state-of-the-art forecasting performance across diverse benchmark datasets. Extensive experiments reveal that the proposed framework is particularly effective in scenarios with high noise levels or irregular patterns, making it well suited for real-world forecasting tasks. The code is available in this https URL. 

**Abstract (ZH)**: 长周期时间序列预测在金融、 economics 和能源等领域至关重要，准确可靠的长期预测驱动着战略决策。尽管基于机器学习的模型取得了进展，但扩展回溯窗口中的时间噪声影响仍被忽视，这往往降低了模型性能和计算效率。本文提出一种新颖框架，通过结合离散小波变换（DWT）和离散余弦变换（DCT）进行噪声 reduction 和提取 robust 长期特征来应对这些挑战。这些变换能够在时间和频率域中将有意义的时间模式与噪声分离。此外，我们引入了一种轻量级低秩线性预测层，不仅减少了残余噪声的影响，还提高了内存效率。本文的方法在噪音输入下的鲁棒性表现竞争，显著降低了计算复杂度，并在多种基准数据集中实现了竞争或最先进的预测性能。大量实验证明，所提出框架特别适用于高噪音水平或不规则模式的场景，使其适合实际预测任务。代码在此 <https://> 可用。 

---
# Efficient Hierarchical Contrastive Self-supervising Learning for Time Series Classification via Importance-aware Resolution Selection 

**Title (ZH)**: 基于重要性感知分辨率选择的高效分层对比自监督学习时间序列分类 

**Authors**: Kevin Garcia, Juan Manuel Perez, Yifeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.10567)  

**Abstract**: Recently, there has been a significant advancement in designing Self-Supervised Learning (SSL) frameworks for time series data to reduce the dependency on data labels. Among these works, hierarchical contrastive learning-based SSL frameworks, which learn representations by contrasting data embeddings at multiple resolutions, have gained considerable attention. Due to their ability to gather more information, they exhibit better generalization in various downstream tasks. However, when the time series data length is significant long, the computational cost is often significantly higher than that of other SSL frameworks. In this paper, to address this challenge, we propose an efficient way to train hierarchical contrastive learning models. Inspired by the fact that each resolution's data embedding is highly dependent, we introduce importance-aware resolution selection based training framework to reduce the computational cost. In the experiment, we demonstrate that the proposed method significantly improves training time while preserving the original model's integrity in extensive time series classification performance evaluations. Our code could be found here, this https URL 

**Abstract (ZH)**: 最近，在为时间序列数据设计自监督学习（SSL）框架以减少对数据标签的依赖方面取得了显著进展。其中，基于层次对比学习的SSL框架因其能够获取更多信息而在各种下游任务中表现出更好的泛化能力，但当时间序列数据长度显著较长时，计算成本往往远高于其他SSL框架。在本文中，为解决这一挑战，我们提出了一种高效训练层次对比学习模型的方法。借鉴每种分辨率的数据嵌入高度相关的事实，我们引入了一种基于重要性意识的分辨率选择训练框架以降低计算成本。在实验中，我们证明了所提出的方法在广泛的时间序列分类性能评估中显著提高了训练时间同时保持了模型的完整性。我们的代码可在此处找到：this https URL。 

---
# SAMRI-2: A Memory-based Model for Cartilage and Meniscus Segmentation in 3D MRIs of the Knee Joint 

**Title (ZH)**: SAMRI-2：一种基于记忆的模型，用于膝关节3D MRI中的软骨和半月板分割 

**Authors**: Danielle L. Ferreira, Bruno A. A. Nunes, Xuzhe Zhang, Laura Carretero Gomez, Maggie Fung, Ravi Soni  

**Link**: [PDF](https://arxiv.org/pdf/2502.10559)  

**Abstract**: Accurate morphometric assessment of cartilage-such as thickness/volume-via MRI is essential for monitoring knee osteoarthritis. Segmenting cartilage remains challenging and dependent on extensive expert-annotated datasets, which are heavily subjected to inter-reader variability. Recent advancements in Visual Foundational Models (VFM), especially memory-based approaches, offer opportunities for improving generalizability and robustness. This study introduces a deep learning (DL) method for cartilage and meniscus segmentation from 3D MRIs using interactive, memory-based VFMs. To improve spatial awareness and convergence, we incorporated a Hybrid Shuffling Strategy (HSS) during training and applied a segmentation mask propagation technique to enhance annotation efficiency. We trained four AI models-a CNN-based 3D-VNet, two automatic transformer-based models (SaMRI2D and SaMRI3D), and a transformer-based promptable memory-based VFM (SAMRI-2)-on 3D knee MRIs from 270 patients using public and internal datasets and evaluated on 57 external cases, including multi-radiologist annotations and different data acquisitions. Model performance was assessed against reference standards using Dice Score (DSC) and Intersection over Union (IoU), with additional morphometric evaluations to further quantify segmentation accuracy. SAMRI-2 model, trained with HSS, outperformed all other models, achieving an average DSC improvement of 5 points, with a peak improvement of 12 points for tibial cartilage. It also demonstrated the lowest cartilage thickness errors, reducing discrepancies by up to threefold. Notably, SAMRI-2 maintained high performance with as few as three user clicks per volume, reducing annotation effort while ensuring anatomical precision. This memory-based VFM with spatial awareness offers a novel approach for reliable AI-assisted knee MRI segmentation, advancing DL in musculoskeletal imaging. 

**Abstract (ZH)**: 通过MRI精确评估关节软骨厚度/体积对于监测膝关节骨关节炎至关重要。基于交互记忆基础视觉主体模型的软骨和半月板分割方法提高了分割稳健性和通用性。 

---
# Synthesis of Dynamic Masks for Information-Theoretic Opacity in Stochastic Systems 

**Title (ZH)**: 基于信息论的透明度合成动态掩码在随机系统中的应用 

**Authors**: Sumukha Udupa, Chongyang Shi, Jie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10552)  

**Abstract**: In this work, we investigate the synthesis of dynamic information releasing mechanisms, referred to as ''masks'', to minimize information leakage from a stochastic system to an external observer. Specifically, for a stochastic system, an observer aims to infer whether the final state of the system trajectory belongs to a set of secret states. The dynamic mask seeks to regulate sensor information in order to maximize the observer's uncertainty about the final state, a property known as final-state opacity. While existing supervisory control literature on dynamic masks primarily addresses qualitative opacity, we propose quantifying opacity in stochastic systems by conditional entropy, which is a measure of information leakage in information security. We then formulate a constrained optimization problem to synthesize a dynamic mask that maximizes final-state opacity under a total cost constraint on masking. To solve this constrained optimal dynamic mask synthesis problem, we develop a novel primal-dual policy gradient method. Additionally, we present a technique for computing the gradient of conditional entropy with respect to the masking policy parameters, leveraging observable operators in hidden Markov models. To demonstrate the effectiveness of our approach, we apply our method to an illustrative example and a stochastic grid world scenario, showing how our algorithm optimally enforces final-state opacity under cost constraints. 

**Abstract (ZH)**: 本文探讨了动态信息释放机制“掩码”的合成，以最小化随机系统对外部观察者的信息泄露。具体而言，对于一个随机系统，观察者试图推断系统轨迹的最终状态是否属于一组秘密状态。动态掩码旨在调节传感器信息，以最大化观察者对最终状态的不确定性，这一特性称为最终状态不透明性。虽然现有动态掩码的监督控制文献主要关注定性不透明性，我们提出通过条件熵来量化随机系统中的不透明性，条件熵是信息安全中的信息泄露度量。然后，我们形式化了一个有约束的优化问题，以在总成本约束下合成最大化最终状态不透明性的动态掩码。为了解决这一有约束的最优动态掩码合成问题，我们开发了一种新颖的普里姆-杜尔政策梯度方法。此外，我们介绍了一种技术，用于利用隐马尔可夫模型中的可观测算子来计算条件熵关于掩码策略参数的梯度。为了展示我们方法的有效性，我们将其应用于一个示例场景和一个随机网格世界场景，展示了在成本约束下我们的算法如何最优地实现最终状态不透明性。 

---
# Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning 

**Title (ZH)**: 记忆、基准与机器人：基于强化学习解决复杂任务的基准 

**Authors**: Egor Cherepanov, Nikita Kachaev, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2502.10550)  

**Abstract**: Memory is crucial for enabling agents to tackle complex tasks with temporal and spatial dependencies. While many reinforcement learning (RL) algorithms incorporate memory, the field lacks a universal benchmark to assess an agent's memory capabilities across diverse scenarios. This gap is particularly evident in tabletop robotic manipulation, where memory is essential for solving tasks with partial observability and ensuring robust performance, yet no standardized benchmarks exist. To address this, we introduce MIKASA (Memory-Intensive Skills Assessment Suite for Agents), a comprehensive benchmark for memory RL, with three key contributions: (1) we propose a comprehensive classification framework for memory-intensive RL tasks, (2) we collect MIKASA-Base - a unified benchmark that enables systematic evaluation of memory-enhanced agents across diverse scenarios, and (3) we develop MIKASA-Robo - a novel benchmark of 32 carefully designed memory-intensive tasks that assess memory capabilities in tabletop robotic manipulation. Our contributions establish a unified framework for advancing memory RL research, driving the development of more reliable systems for real-world applications. The code is available at this https URL. 

**Abstract (ZH)**: 记忆对于使智能体能够应对具有时间性和空间性依赖的复杂任务至关重要。虽然许多强化学习（RL）算法都包含了记忆机制，但该领域缺乏一个通用基准来评估智能体在不同场景中的记忆能力。这一差距在桌面机器人操作中尤为明显，因为在这种场景中，记忆对于解决部分可观测任务和确保稳健性能是必不可少的，但尚未存在标准化的基准测试。为解决这一问题，我们引入了MIKASA（Memory-Intensive Skills Assessment Suite for Agents），这是一个全面的记忆RL基准测试，其三大贡献为：（1）提出了一种全面的记忆密集型RL任务分类框架；（2）收集了MIKASA-Base - 一个统一的基准测试，可系统评估增强记忆能力的智能体在多种场景中的表现；（3）开发了MIKASA-Robo，这是一个新型的包含32个精心设计的记忆密集型任务的基准测试，用于评估桌面机器人操作中的记忆能力。我们的贡献建立了一个统一的框架，加速了记忆RL研究的进步，推动了更可靠系统在实际应用中的发展。代码可在以下链接获取：this https URL。 

---
# Learning to be Smooth: An End-to-End Differentiable Particle Smoother 

**Title (ZH)**: 学习平滑：端到端可微分粒子平滑器 

**Authors**: Ali Younis, Erik B. Sudderth  

**Link**: [PDF](https://arxiv.org/pdf/2502.10546)  

**Abstract**: For challenging state estimation problems arising in domains like vision and robotics, particle-based representations attractively enable temporal reasoning about multiple posterior modes. Particle smoothers offer the potential for more accurate offline data analysis by propagating information both forward and backward in time, but have classically required human-engineered dynamics and observation models. Extending recent advances in discriminative training of particle filters, we develop a framework for low-variance propagation of gradients across long time sequences when training particle smoothers. Our "two-filter'' smoother integrates particle streams that are propagated forward and backward in time, while incorporating stratification and importance weights in the resampling step to provide low-variance gradient estimates for neural network dynamics and observation models. The resulting mixture density particle smoother is substantially more accurate than state-of-the-art particle filters, as well as search-based baselines, for city-scale global vehicle localization from real-world videos and maps. 

**Abstract (ZH)**: 基于粒子的表示方法在视觉和机器人学等领域中解决具有挑战性的状态估计问题时，能够吸引人地实现对多个后验模式的时域推理。粒子平滑器通过在时间的正反两个方向传播信息，为更精准的离线数据分析提供了潜力，但传统上需要手工构建的动力学和观测模型。结合近年来粒子滤波的判别训练进展，我们开发了一种框架，用于在训练粒子平滑器时沿长时间序列进行低方差梯度传播。我们的“两滤波器”平滑器整合了沿时间正反方向传播的粒子流，并在重采样步骤中采用分层和重要性加权，以提供神经网络动力学和观测模型的低方差梯度估计。由此得到的混合密度粒子平滑器在从真实世界视频和地图中进行大规模车辆全局定位方面，比现有的粒子滤波器和基于搜索的基线更为准确。 

---
# PolyPath: Adapting a Large Multimodal Model for Multi-slide Pathology Report Generation 

**Title (ZH)**: PolyPath: 调整大型多模态模型以生成多张切片病理报告 

**Authors**: Faruk Ahmed, Lin Yang, Tiam Jaroensri, Andrew Sellergren, Yossi Matias, Avinatan Hassidim, Greg S. Corrado, Dale R. Webster, Shravya Shetty, Shruthi Prabhakara, Yun Liu, Daniel Golden, Ellery Wulczyn, David F. Steiner  

**Link**: [PDF](https://arxiv.org/pdf/2502.10536)  

**Abstract**: The interpretation of histopathology cases underlies many important diagnostic and treatment decisions in medicine. Notably, this process typically requires pathologists to integrate and summarize findings across multiple slides per case. Existing vision-language capabilities in computational pathology have so far been largely limited to small regions of interest, larger regions at low magnification, or single whole-slide images (WSIs). This limits interpretation of findings that span multiple high-magnification regions across multiple WSIs. By making use of Gemini 1.5 Flash, a large multimodal model (LMM) with a 1-million token context window, we demonstrate the ability to generate bottom-line diagnoses from up to 40,000 768x768 pixel image patches from multiple WSIs at 10X magnification. This is the equivalent of up to 11 hours of video at 1 fps. Expert pathologist evaluations demonstrate that the generated report text is clinically accurate and equivalent to or preferred over the original reporting for 68% (95% CI: [60%, 76%]) of multi-slide examples with up to 5 slides. While performance decreased for examples with 6 or more slides, this study demonstrates the promise of leveraging the long-context capabilities of modern LMMs for the uniquely challenging task of medical report generation where each case can contain thousands of image patches. 

**Abstract (ZH)**: 病理组织学病例的解释是医学中许多重要诊断和治疗决策的基础。这一过程通常要求病理学家综合和总结每个病例多块切片中的发现。目前，计算病理学中的视觉-语言能力主要局限于感兴趣的 small 区域、低放大倍数下的较大区域，或单张全切片图像 (WSI)。这限制了跨越多张高放大倍数 WSI 的多个区域发现的解释。通过利用 Gemini 1.5 Flash，一种具有 100 万标记上下文窗口的大规模多模态模型（LMM），我们展示了从多达 40,000 张分辨率为 768x768 像素的图像块生成最终诊断结果的能力，这些图像块来自多张 10X 放大事的 WSI。这相当于多达 11 小时以 1 fps 播放的视频。专家病理学家评估表明，生成的报告文本在临床准确性方面与原始报告相当或更优，占多达 5 张切片的多切片示例的 68%（95% CI：[60%，76%]）。虽然对于 6 张或更多切片的示例性能有所下降，但本研究展示了利用现代 LMM 的长上下文能力进行医学报告生成的潜力，尤其是在每个病例可能包含数千个图像块的这一独特挑战性任务中。 

---
# Tempo: Helping Data Scientists and Domain Experts Collaboratively Specify Predictive Modeling Tasks 

**Title (ZH)**: Tempo: 协助数据科学家和领域专家协作指定预测建模任务 

**Authors**: Venkatesh Sivaraman, Anika Vaishampayan, Xiaotong Li, Brian R Buck, Ziyong Ma, Richard D Boyce, Adam Perer  

**Link**: [PDF](https://arxiv.org/pdf/2502.10526)  

**Abstract**: Temporal predictive models have the potential to improve decisions in health care, public services, and other domains, yet they often fail to effectively support decision-makers. Prior literature shows that many misalignments between model behavior and decision-makers' expectations stem from issues of model specification, namely how, when, and for whom predictions are made. However, model specifications for predictive tasks are highly technical and difficult for non-data-scientist stakeholders to interpret and critique. To address this challenge we developed Tempo, an interactive system that helps data scientists and domain experts collaboratively iterate on model specifications. Using Tempo's simple yet precise temporal query language, data scientists can quickly prototype specifications with greater transparency about pre-processing choices. Moreover, domain experts can assess performance within data subgroups to validate that models behave as expected. Through three case studies, we demonstrate how Tempo helps multidisciplinary teams quickly prune infeasible specifications and identify more promising directions to explore. 

**Abstract (ZH)**: 临时预测模型有潜力改善医疗保健、公共服务等领域中的决策，但往往未能有效地支持决策者。 prior literature 表明，模型行为与决策者预期之间的许多不一致性源自模型规格问题，即预测如何、何时以及针对谁做出。然而，预测任务的模型规格高度技术化，难以让非数据科学家利益相关方进行解释和评估。为应对这一挑战，我们开发了 Tempo，一个交互系统，帮助数据科学家和领域专家协作调整模型规格。通过 Tempo 简单而精确的时间查询语言，数据科学家可以快速制定具有更多预处理选择透明度的规格原型。此外，领域专家可以在数据子组中评估模型性能，以验证模型是否按预期行为。通过三个案例研究，我们展示了 Tempo 如何帮助跨学科团队迅速排除不可行的规格，并确定更值得探索的方向。 

---
# KernelBench: Can LLMs Write Efficient GPU Kernels? 

**Title (ZH)**: KernelBench: 能否生成高效的GPU内核代码？ 

**Authors**: Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, Azalia Mirhoseini  

**Link**: [PDF](https://arxiv.org/pdf/2502.10517)  

**Abstract**: Efficient GPU kernels are crucial for building performant machine learning architectures, but writing them is a time-consuming challenge that requires significant expertise; therefore, we explore using language models (LMs) to automate kernel generation. We introduce KernelBench, an open-source framework for evaluating LMs' ability to write fast and correct kernels on a suite of 250 carefully selected PyTorch ML workloads. KernelBench represents a real-world engineering environment and making progress on the introduced benchmark directly translates to faster practical kernels. We introduce a new evaluation metric fast_p, which measures the percentage of generated kernels that are functionally correct and offer a speedup greater than an adjustable threshold p over baseline. Our experiments across various state-of-the-art models and test-time methods show that frontier reasoning models perform the best out of the box but still fall short overall, matching the PyTorch baseline in less than 20% of the cases. While we show that results can improve by leveraging execution and profiling feedback during iterative refinement, KernelBench remains a challenging benchmark, with its difficulty increasing as we raise speedup threshold p. 

**Abstract (ZH)**: 高效的GPU内核对于构建高性能机器学习架构至关重要，但编写它们是一项耗时的挑战，需要大量的专业知识；因此，我们探索使用语言模型（LMs）来自动化内核生成。我们引入了KernelBench，这是一个开源框架，用于评估LMs在一系列250个精心挑选的PyTorch机器学习工作负载上编写快速且正确的内核的能力。KernelBench代表了一个真实的世界工程环境，通过改进引入的基准测试，可以直接加快实际内核的速度。我们引入了一个新的评估指标fast_p，该指标衡量生成内核中功能正确且比基线快于可调节阈值p以上的百分比。在各种最先进的模型和测试时方法的实验中显示，前沿推理模型开箱即用表现最佳，但仍总体上表现不佳，在不到20%的情况下与PyTorch基线相当。虽然通过在迭代优化过程中利用执行和剖析反馈可以提升结果，但随着提高速度阈值p，KernelBench仍是一个具有挑战性的基准。 

---
# Hallucinations and Truth: A Comprehensive Accuracy Evaluation of RAG, LoRA and DoRA 

**Title (ZH)**: 幻觉与现实：RAG、LoRA和DoRA的综合准确度评估 

**Authors**: Mohammad Baqar, Rajat Khanda  

**Link**: [PDF](https://arxiv.org/pdf/2502.10497)  

**Abstract**: Recent advancements in Generative AI have significantly improved the efficiency and adaptability of natural language processing (NLP) systems, particularly through Retrieval-Augmented Generation (RAG), Low-Rank Adaptation (LoRA), and Weight-Decomposed Low-Rank Adaptation (DoRA). RAG integrates external knowledge to enhance factual consistency in generative outputs, while LoRA enables parameter-efficient fine-tuning of large language models (LLMs). DoRA further refines this process by optimizing fine-tuning through adaptive parameter ranking and domain-aware weight adjustments, improving learning efficiency while maintaining inference performance.
This paper presents a large-scale empirical evaluation of RAG, LoRA, and DoRA, with model fine-tuning and generation performance assessed on 20,000 FAQ-based queries, while the knowledge base spans 400,000 entries. The study analyzes key performance metrics such as accuracy, relevance, and inference latency. Experimental results demonstrate that DoRA achieves the highest accuracy (90.1%), relevance score (0.88), and lowest latency (110 ms per query), outperforming both LoRA and RAG in real-world, domain-specific generative AI applications.
Furthermore, this study examines the trade-offs between fine-tuning efficiency, computational cost, and real-time adaptability across different models. Findings highlight RAG's effectiveness in knowledge grounding, LoRA's cost-efficient domain adaptation, and DoRA's ability to balance fine-tuning efficiency with model precision. These insights provide practical guidance for deploying AI-driven generative systems in accuracy-critical domains such as healthcare, finance, and legal services, ensuring scalability, reliability, and optimal performance in dynamic environments. 

**Abstract (ZH)**: 最近生成式人工智能的发展显著提升了自然语言处理（NLP）系统的效率和适应性，特别是在检索增强生成（RAG）、低秩适应（LoRA）和分解低秩适应（DoRA）方面。RAG通过整合外部知识来增强生成输出的事实一致性，而LoRA使大规模语言模型（LLMs）的参数高效微调成为可能。DoRA则通过自适应参数排名和领域感知的权重调整进一步优化微调过程，提高了学习效率，同时保持推理性能。本文通过大规模实证评估了RAG、LoRA和DoRA，评估了20,000个基于FAQ的查询的模型微调和生成性能，知识库包含400,000条记录。研究分析了关键性能指标，如准确性、相关性和推理延迟。实验结果表明，DoRA在准确性（90.1%）、相关性得分（0.88）和延迟（每查询110毫秒）方面表现最佳，超越了LoRA和RAG在实际领域的应用。此外，本文还探讨了不同模型在微调效率、计算成本和实时适应性之间的权衡。研究发现强调了RAG在知识接地方面的有效性、LoRA在成本效益领域的适应性以及DoRA在平衡微调效率和模型精度方面的能力。这些见解为在准确性关键领域，如医疗、金融和法律服务中部署以人工智能驱动的生成系统提供了实用指导，确保在动态环境中实现可扩展性、可靠性和最佳性能。 

---
# SWA-LDM: Toward Stealthy Watermarks for Latent Diffusion Models 

**Title (ZH)**: SWA-LDM：潜扩散模型中的隐蔽水印技术 

**Authors**: Zhonghao Yang, Linye Lyu, Xuanhang Chang, Daojing He, YU LI  

**Link**: [PDF](https://arxiv.org/pdf/2502.10495)  

**Abstract**: In the rapidly evolving landscape of image generation, Latent Diffusion Models (LDMs) have emerged as powerful tools, enabling the creation of highly realistic images. However, this advancement raises significant concerns regarding copyright infringement and the potential misuse of generated content. Current watermarking techniques employed in LDMs often embed constant signals to the generated images that compromise their stealthiness, making them vulnerable to detection by malicious attackers. In this paper, we introduce SWA-LDM, a novel approach that enhances watermarking by randomizing the embedding process, effectively eliminating detectable patterns while preserving image quality and robustness. Our proposed watermark presence attack reveals the inherent vulnerabilities of existing latent-based watermarking methods, demonstrating how easily these can be exposed. Through comprehensive experiments, we validate that SWA-LDM not only fortifies watermark stealthiness but also maintains competitive performance in watermark robustness and visual fidelity. This work represents a pivotal step towards securing LDM-generated images against unauthorized use, ensuring both copyright protection and content integrity in an era where digital image authenticity is paramount. 

**Abstract (ZH)**: 基于随机化嵌入的SWA-LDM水印方法：提升Latent Diffusion Models生成图像的安全性 

---
# F-StrIPE: Fast Structure-Informed Positional Encoding for Symbolic Music Generation 

**Title (ZH)**: F-StrIPE: 快速结构导向位置编码在符号音乐生成中的应用 

**Authors**: Manvi Agarwal, Changhong Wang, Gael Richard  

**Link**: [PDF](https://arxiv.org/pdf/2502.10491)  

**Abstract**: While music remains a challenging domain for generative models like Transformers, recent progress has been made by exploiting suitable musically-informed priors. One technique to leverage information about musical structure in Transformers is inserting such knowledge into the positional encoding (PE) module. However, Transformers carry a quadratic cost in sequence length. In this paper, we propose F-StrIPE, a structure-informed PE scheme that works in linear complexity. Using existing kernel approximation techniques based on random features, we show that F-StrIPE is a generalization of Stochastic Positional Encoding (SPE). We illustrate the empirical merits of F-StrIPE using melody harmonization for symbolic music. 

**Abstract (ZH)**: 基于结构的线性复杂度位置编码方案F-StrIPE 

---
# A Robust Attack: Displacement Backdoor Attack 

**Title (ZH)**: robust 攻击: 移动后门攻击 

**Authors**: Yong Li, Han Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.10490)  

**Abstract**: As artificial intelligence becomes more prevalent in our lives, people are enjoying the convenience it brings, but they are also facing hidden threats, such as data poisoning and ad- versarial attacks. These threats can have disastrous consequences for the application of artificial intelligence, especially for some applications that take effect immediately, such as autonomous driving and medical fields. Among these threats, backdoor attacks have left a deep impression on people with their concealment and simple deployment, making them a threat that cannot be ignored, however, in the process of deploying the backdoor model, the backdoor attack often has some reasons that make it unsatisfactory in real-world applications, such as jitter and brightness changes. Based on this, we propose a highly robust backdoor attack that shifts the target sample and combines it with itself to form a backdoor sample, the Displacement Backdoor Attack(DBA). Experimental results show that the DBA attack can resist data augmentation that simulates real-world differences, such as rotation and cropping. 

**Abstract (ZH)**: 随着人工智能在人们生活中变得越来越普遍，人们享受着它带来的便利，但也面临着隐藏的威胁，如数据污染和对抗攻击。这些威胁可能对人工智能的应用造成灾难性后果，尤其是在自动驾驶和医疗等即时生效的应用领域。在这些威胁中，后门攻击因其隐蔽性和简单的部署方式给人留下了深刻印象，成为不可忽视的威胁。然而，在部署后门模型的过程中，后门攻击往往因现实应用中的抖动和亮度变化等问题不尽如人意。基于此，我们提出了一种高度鲁棒的后门攻击——位移后门攻击(DBA)，该攻击通过将目标样本与自身进行位移并结合以形成后门样本。实验结果表明，DBA攻击可以抵抗模拟现实世界差异的数据增强，如旋转和裁剪。 

---
# LiveVal: Time-aware Data Valuation via Adaptive Reference Points 

**Title (ZH)**: LiveVal: 基于自适应参考点的时敏数据估值 

**Authors**: Jie Xu, Zihan Wu, Cong Wang, Xiaohua Jia  

**Link**: [PDF](https://arxiv.org/pdf/2502.10489)  

**Abstract**: Time-aware data valuation enhances training efficiency and model robustness, as early detection of harmful samples could prevent months of wasted computation. However, existing methods rely on model retraining or convergence assumptions or fail to capture long-term training dynamics.
We propose LiveVal, an efficient time-aware data valuation method with three key designs:
1) seamless integration with SGD training for efficient data contribution monitoring; 2) reference-based valuation with normalization for reliable benchmark establishment; and 3) adaptive reference point selection for real-time updating with optimized memory usage.
We establish theoretical guarantees for LiveVal's stability and prove that its valuations are bounded and directionally aligned with optimization progress. Extensive experiments demonstrate that LiveVal provides efficient data valuation across different modalities and model scales, achieving 180 speedup over traditional methods while maintaining robust detection performance. 

**Abstract (ZH)**: 基于时间的数据评价增强训练效率和模型鲁棒性，通过早期检测有害样本可避免数月的无效计算。然而，现有方法依赖于模型重训或收敛假设，或无法捕捉长期训练动态。 

---
# Fast Proxies for LLM Robustness Evaluation 

**Title (ZH)**: 快速代理用于LLM鲁棒性评估 

**Authors**: Tim Beyer, Jan Schuchardt, Leo Schwinn, Stephan Günnemann  

**Link**: [PDF](https://arxiv.org/pdf/2502.10487)  

**Abstract**: Evaluating the robustness of LLMs to adversarial attacks is crucial for safe deployment, yet current red-teaming methods are often prohibitively expensive. We compare the ability of fast proxy metrics to predict the real-world robustness of an LLM against a simulated attacker ensemble. This allows us to estimate a model's robustness to computationally expensive attacks without requiring runs of the attacks themselves. Specifically, we consider gradient-descent-based embedding-space attacks, prefilling attacks, and direct prompting. Even though direct prompting in particular does not achieve high ASR, we find that it and embedding-space attacks can predict attack success rates well, achieving $r_p=0.87$ (linear) and $r_s=0.94$ (Spearman rank) correlations with the full attack ensemble while reducing computational cost by three orders of magnitude. 

**Abstract (ZH)**: 评估大规模语言模型对 adversarial 攻击的鲁棒性对于安全部署至关重要，但当前的红队方法往往代价高昂。我们将快速代理指标的能力与模拟攻击集合的实际鲁棒性进行比较，从而在不需要运行实际攻击的情况下，估计模型对计算成本高昂的攻击的鲁棒性。具体而言，我们考虑基于梯度下降的嵌入空间攻击、预填攻击和直接提示。尽管直接提示特别不能实现高的成功率，但我们发现它和嵌入空间攻击能够很好地预测攻击成功率，与完整攻击集合相比，实现 $r_p=0.87$（线性）和 $r_s=0.94$（斯皮尔曼秩）的相关性，同时将计算成本降低三个数量级。 

---
# VLM-Guard: Safeguarding Vision-Language Models via Fulfilling Safety Alignment Gap 

**Title (ZH)**: VLM-Guard: 通过弥补安全对齐缺口来保障视觉-语言模型的安全性 

**Authors**: Qin Liu, Fei Wang, Chaowei Xiao, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.10486)  

**Abstract**: The emergence of vision language models (VLMs) comes with increased safety concerns, as the incorporation of multiple modalities heightens vulnerability to attacks. Although VLMs can be built upon LLMs that have textual safety alignment, it is easily undermined when the vision modality is integrated. We attribute this safety challenge to the modality gap, a separation of image and text in the shared representation space, which blurs the distinction between harmful and harmless queries that is evident in LLMs but weakened in VLMs. To avoid safety decay and fulfill the safety alignment gap, we propose VLM-Guard, an inference-time intervention strategy that leverages the LLM component of a VLM as supervision for the safety alignment of the VLM. VLM-Guard projects the representations of VLM into the subspace that is orthogonal to the safety steering direction that is extracted from the safety-aligned LLM. Experimental results on three malicious instruction settings show the effectiveness of VLM-Guard in safeguarding VLM and fulfilling the safety alignment gap between VLM and its LLM component. 

**Abstract (ZH)**: 视觉语言模型（VLMs）的出现伴随着安全问题的增加，因为多模态的整合使得模型更容易受到攻击。尽管VLMs可以基于具有文本安全对齐的大型语言模型（LLMs）构建，但当引入视觉模态时，其安全性很容易受到影响。我们将这一安全挑战归因于模态差距，即在共享表示空间中图像与文本的分离，这模糊了在LLMs中明显存在的有害与无害查询之间的区别，在VLMs中则减弱了这种区别。为了防止安全性衰退并弥补安全对齐差距，我们提出了一种VLM-Guard，它是一种推理时的干预策略，利用VLM中的LLM组件作为监督，以实现VLM的安全对齐。VLM-Guard将VLM的表示投影到从安全对齐的LLM中提取的安全导向的正交子空间中。在三个恶意指令设置上的实验结果表明，VLM-Guard在保护VLM并弥补VLM与其LLM组件之间的安全对齐差距方面是有效的。 

---
# Forecasting time series with constraints 

**Title (ZH)**: 具有约束条件的时间序列预测 

**Authors**: Nathan Doumèche, Francis Bach, Éloi Bedek, Gérard Biau, Claire Boyer, Yannig Goude  

**Link**: [PDF](https://arxiv.org/pdf/2502.10485)  

**Abstract**: Time series forecasting presents unique challenges that limit the effectiveness of traditional machine learning algorithms. To address these limitations, various approaches have incorporated linear constraints into learning algorithms, such as generalized additive models and hierarchical forecasting. In this paper, we propose a unified framework for integrating and combining linear constraints in time series forecasting. Within this framework, we show that the exact minimizer of the constrained empirical risk can be computed efficiently using linear algebra alone. This approach allows for highly scalable implementations optimized for GPUs. We validate the proposed methodology through extensive benchmarking on real-world tasks, including electricity demand forecasting and tourism forecasting, achieving state-of-the-art performance. 

**Abstract (ZH)**: 时间序列预测面临着独特的挑战，限制了传统机器学习算法的效果。为应对这些局限性，各种方法将线性约束融入学习算法中，例如广义加性模型和层次预测。在本文中，我们提出了一种统一框架，用于在时间序列预测中整合和结合线性约束。在此框架内，我们证明可以通过线性代数高效计算受约束的经验风险的精确最小化器。该方法允许对GPU进行高度可扩展的实现。通过在实际任务上进行广泛的基准测试进行验证，包括电力需求预测和旅游预测，实现了最先进的性能。 

---
# X-SG$^2$S: Safe and Generalizable Gaussian Splatting with X-dimensional Watermarks 

**Title (ZH)**: X-SG$^2$S: 安全且通用的高维水印高斯散列 

**Authors**: Zihang Cheng, Huiping Zhuang, Chun Li, Xin Meng, Ming Li, Fei Richard Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10475)  

**Abstract**: 3D Gaussian Splatting (3DGS) has been widely used in 3D reconstruction and 3D generation. Training to get a 3DGS scene often takes a lot of time and resources and even valuable inspiration. The increasing amount of 3DGS digital asset have brought great challenges to the copyright protection. However, it still lacks profound exploration targeted at 3DGS. In this paper, we propose a new framework X-SG$^2$S which can simultaneously watermark 1 to 3D messages while keeping the original 3DGS scene almost unchanged. Generally, we have a X-SG$^2$S injector for adding multi-modal messages simultaneously and an extractor for extract them. Specifically, we first split the watermarks into message patches in a fixed manner and sort the 3DGS points. A self-adaption gate is used to pick out suitable location for watermarking. Then use a XD(multi-dimension)-injection heads to add multi-modal messages into sorted 3DGS points. A learnable gate can recognize the location with extra messages and XD-extraction heads can restore hidden messages from the location recommended by the learnable gate. Extensive experiments demonstrated that the proposed X-SG$^2$S can effectively conceal multi modal messages without changing pretrained 3DGS pipeline or the original form of 3DGS parameters. Meanwhile, with simple and efficient model structure and high practicality, X-SG$^2$S still shows good performance in hiding and extracting multi-modal inner structured or unstructured messages. X-SG$^2$S is the first to unify 1 to 3D watermarking model for 3DGS and the first framework to add multi-modal watermarks simultaneous in one 3DGS which pave the wave for later researches. 

**Abstract (ZH)**: 基于3D高斯喷洒的多模态水印框架X-SG$^2$S 

---
# MetaDE: Evolving Differential Evolution by Differential Evolution 

**Title (ZH)**: MetaDE: 通过差分进化演变差分进化 

**Authors**: Minyang Chen, Chenchen Feng, and Ran Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.10470)  

**Abstract**: As a cornerstone in the Evolutionary Computation (EC) domain, Differential Evolution (DE) is known for its simplicity and effectiveness in handling challenging black-box optimization problems. While the advantages of DE are well-recognized, achieving peak performance heavily depends on its hyperparameters such as the mutation factor, crossover probability, and the selection of specific DE strategies. Traditional approaches to this hyperparameter dilemma have leaned towards parameter tuning or adaptive mechanisms. However, identifying the optimal settings tailored for specific problems remains a persistent challenge. In response, we introduce MetaDE, an approach that evolves DE's intrinsic hyperparameters and strategies using DE itself at a meta-level. A pivotal aspect of MetaDE is a specialized parameterization technique, which endows it with the capability to dynamically modify DE's parameters and strategies throughout the evolutionary process. To augment computational efficiency, MetaDE incorporates a design that leverages parallel processing through a GPU-accelerated computing framework. Within such a framework, DE is not just a solver but also an optimizer for its own configurations, thus streamlining the process of hyperparameter optimization and problem-solving into a cohesive and automated workflow. Extensive evaluations on the CEC2022 benchmark suite demonstrate MetaDE's promising performance. Moreover, when applied to robot control via evolutionary reinforcement learning, MetaDE also demonstrates promising performance. The source code of MetaDE is publicly accessible at: this https URL. 

**Abstract (ZH)**: 作为一种进化计算领域的基石，差分进化因其在处理复杂黑盒优化问题时的简单性和有效性而闻名。尽管差分进化的优势得到广泛认可，但实现其最佳性能高度依赖于其超参数，如变异因子、交叉概率以及特定差分进化策略的选择。传统的方法倾向于参数调整或自适应机制来解决这一问题，然而，为特定问题寻找最优设置仍然是一个持续性的挑战。为此，我们引入了MetaDE，这是一种利用差分进化自身在元层次上进化其固有超参数和策略的方法。MetaDE的关键方面在于一种专门的参数化技术，赋予其动态修改差分进化参数和策略的能力，贯穿整个进化过程。为提高计算效率，MetaDE采用了一种利用GPU加速计算框架的设计。在这种框架中，DE不仅是一个求解器，也是一个优化其自身配置的优化器，从而将超参数优化和问题求解流程简化为一个协调且自动的工作流。对CEC2022基准测试集进行的广泛评估显示了MetaDE的潜力。此外，将其应用于通过进化强化学习进行的机器人控制时，MetaDE也展现了有趣的性能。MetaDE的源代码可在以下链接获取：this https URL。 

---
# YNote: A Novel Music Notation for Fine-Tuning LLMs in Music Generation 

**Title (ZH)**: YNote：一种用于音乐生成中LLMs微调的新乐谱表示方法 

**Authors**: Shao-Chien Lu, Chen-Chen Yeh, Hui-Lin Cho, Chun-Chieh Hsu, Tsai-Ling Hsu, Cheng-Han Wu, Timothy K. Shih, Yu-Cheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.10467)  

**Abstract**: The field of music generation using Large Language Models (LLMs) is evolving rapidly, yet existing music notation systems, such as MIDI, ABC Notation, and MusicXML, remain too complex for effective fine-tuning of LLMs. These formats are difficult for both machines and humans to interpret due to their variability and intricate structure. To address these challenges, we introduce YNote, a simplified music notation system that uses only four characters to represent a note and its pitch. YNote's fixed format ensures consistency, making it easy to read and more suitable for fine-tuning LLMs. In our experiments, we fine-tuned GPT-2 (124M) on a YNote-encoded dataset and achieved BLEU and ROUGE scores of 0.883 and 0.766, respectively. With just two notes as prompts, the model was able to generate coherent and stylistically relevant music. We believe YNote offers a practical alternative to existing music notations for machine learning applications and has the potential to significantly enhance the quality of music generation using LLMs. 

**Abstract (ZH)**: 使用大型语言模型生成音乐领域的简化音乐记谱系统：YNote的研究 

---
# From Layers to States: A State Space Model Perspective to Deep Neural Network Layer Dynamics 

**Title (ZH)**: 从层到状态：深层神经网络层动态的态空间模型视角 

**Authors**: Qinshuo Liu, Weiqin Zhao, Wei Huang, Yanwen Fang, Lequan Yu, Guodong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.10463)  

**Abstract**: The depth of neural networks is a critical factor for their capability, with deeper models often demonstrating superior performance. Motivated by this, significant efforts have been made to enhance layer aggregation - reusing information from previous layers to better extract features at the current layer, to improve the representational power of deep neural networks. However, previous works have primarily addressed this problem from a discrete-state perspective which is not suitable as the number of network layers grows. This paper novelly treats the outputs from layers as states of a continuous process and considers leveraging the state space model (SSM) to design the aggregation of layers in very deep neural networks. Moreover, inspired by its advancements in modeling long sequences, the Selective State Space Models (S6) is employed to design a new module called Selective State Space Model Layer Aggregation (S6LA). This module aims to combine traditional CNN or transformer architectures within a sequential framework, enhancing the representational capabilities of state-of-the-art vision networks. Extensive experiments show that S6LA delivers substantial improvements in both image classification and detection tasks, highlighting the potential of integrating SSMs with contemporary deep learning techniques. 

**Abstract (ZH)**: 基于 Continuous 过程的深度神经网络层聚合研究：Selective State Space Model Layer Aggregation（S6LA） 

---
# LLM4GNAS: A Large Language Model Based Toolkit for Graph Neural Architecture Search 

**Title (ZH)**: LLM4GNAS：基于大规模语言模型的图神经网络架构搜索工具包 

**Authors**: Yang Gao, Hong Yang, Yizhi Chen, Junxian Wu, Peng Zhang, Haishuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10459)  

**Abstract**: Graph Neural Architecture Search (GNAS) facilitates the automatic design of Graph Neural Networks (GNNs) tailored to specific downstream graph learning tasks. However, existing GNAS approaches often require manual adaptation to new graph search spaces, necessitating substantial code optimization and domain-specific knowledge. To address this challenge, we present LLM4GNAS, a toolkit for GNAS that leverages the generative capabilities of Large Language Models (LLMs). LLM4GNAS includes an algorithm library for graph neural architecture search algorithms based on LLMs, enabling the adaptation of GNAS methods to new search spaces through the modification of LLM prompts. This approach reduces the need for manual intervention in algorithm adaptation and code modification. The LLM4GNAS toolkit is extensible and robust, incorporating LLM-enhanced graph feature engineering, LLM-enhanced graph neural architecture search, and LLM-enhanced hyperparameter optimization. Experimental results indicate that LLM4GNAS outperforms existing GNAS methods on tasks involving both homogeneous and heterogeneous graphs. 

**Abstract (ZH)**: LLM4GNAS：一种利用大型语言模型的生成能力的图神经网络架构搜索工具包 

---
# I Think, Therefore I Diffuse: Enabling Multimodal In-Context Reasoning in Diffusion Models 

**Title (ZH)**: 我认为，因此我扩散：在扩散模型中启用多模态上下文推理 

**Authors**: Zhenxing Mi, Kuan-Chieh Wang, Guocheng Qian, Hanrong Ye, Runtao Liu, Sergey Tulyakov, Kfir Aberman, Dan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10458)  

**Abstract**: This paper presents ThinkDiff, a novel alignment paradigm that empowers text-to-image diffusion models with multimodal in-context understanding and reasoning capabilities by integrating the strengths of vision-language models (VLMs). Existing multimodal diffusion finetuning methods largely focus on pixel-level reconstruction rather than in-context reasoning, and are constrained by the complexity and limited availability of reasoning-based datasets. ThinkDiff addresses these challenges by leveraging vision-language training as a proxy task, aligning VLMs with the decoder of an encoder-decoder large language model (LLM) instead of a diffusion decoder. This proxy task builds on the observation that the $\textbf{LLM decoder}$ shares the same input feature space with $\textbf{diffusion decoders}$ that use the corresponding $\textbf{LLM encoder}$ for prompt embedding. As a result, aligning VLMs with diffusion decoders can be simplified through alignment with the LLM decoder. Without complex training and datasets, ThinkDiff effectively unleashes understanding, reasoning, and composing capabilities in diffusion models. Experiments demonstrate that ThinkDiff significantly improves accuracy from 19.2% to 46.3% on the challenging CoBSAT benchmark for multimodal in-context reasoning generation, with only 5 hours of training on 4 A100 GPUs. Additionally, ThinkDiff demonstrates exceptional performance in composing multiple images and texts into logically coherent images. Project page: this https URL. 

**Abstract (ZH)**: 本文提出了ThinkDiff，这是一种新颖的对齐范式，通过集成视觉语言模型（VLMs）的优势，赋予文本到图像扩散模型多模态上下文理解与推理能力。现有的多模态扩散微调方法主要关注像素级重建而非上下文推理，并受到基于推理的数据集复杂性和稀缺性的限制。ThinkDiff通过利用视觉语言训练作为代理任务来应对这些挑战，将VLMs与编码器-解码器大型语言模型（LLM）的解码器对齐，而不是扩散解码器。这一代理任务基于观察，即LLM解码器与使用相应LLM编码器进行提示嵌入的扩散解码器共享相同输入特征空间。因此，通过与LLM解码器对齐，可以使VLMs与扩散解码器的对齐简化。无需复杂的训练和数据集，ThinkDiff有效释放了扩散模型的理解、推理和生成能力。实验结果表明，ThinkDiff在具有挑战性的CoBSAT基准测试中将多模态上下文推理生成的准确性从19.2%显著提高到46.3%，仅在4块A100 GPU上进行5小时的训练。此外，ThinkDiff在合成多张图像和文本为逻辑连贯的图像方面表现出色。项目页面：this https URL。 

---
# One Example Shown, Many Concepts Known! Counterexample-Driven Conceptual Reasoning in Mathematical LLMs 

**Title (ZH)**: 一例展现，众理皆知！基于反例的概念性推理在数学大语言模型中的应用 

**Authors**: Yinghui Li, Jiayi Kuang, Haojing Huang, Zhikun Xu, Xinnian Liang, Yi Yu, Wenlian Lu, Yangning Li, Xiaoyu Tan, Chao Qu, Ying Shen, Hai-Tao Zheng, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10454)  

**Abstract**: Leveraging mathematical Large Language Models (LLMs) for proof generation is a fundamental topic in LLMs research. We argue that the ability of current LLMs to prove statements largely depends on whether they have encountered the relevant proof process during training. This reliance limits their deeper understanding of mathematical theorems and related concepts. Inspired by the pedagogical method of "proof by counterexamples" commonly used in human mathematics education, our work aims to enhance LLMs' ability to conduct mathematical reasoning and proof through counterexamples. Specifically, we manually create a high-quality, university-level mathematical benchmark, CounterMATH, which requires LLMs to prove mathematical statements by providing counterexamples, thereby assessing their grasp of mathematical concepts. Additionally, we develop a data engineering framework to automatically obtain training data for further model improvement. Extensive experiments and detailed analyses demonstrate that CounterMATH is challenging, indicating that LLMs, such as OpenAI o1, have insufficient counterexample-driven proof capabilities. Moreover, our exploration into model training reveals that strengthening LLMs' counterexample-driven conceptual reasoning abilities is crucial for improving their overall mathematical capabilities. We believe that our work offers new perspectives on the community of mathematical LLMs. 

**Abstract (ZH)**: 利用数学大型语言模型（LLMs）进行证明生成是LLMs研究中的基础课题。我们argue当前LLMs证明陈述的能力主要依赖于它们在训练过程中是否遇到了相关的证明过程。这种依赖限制了它们对数学定理及相关概念的更深层次理解。受人类数学教育中常用的教学方法“反例证明”启发，我们的工作旨在通过反例增强LLMs的数学推理和证明能力。具体来说，我们手动创建了一个高质量的大学水平数学基准CounterMATH，要求LLMs通过提供反例来证明数学陈述，从而评估它们对数学概念的掌握程度。此外，我们还开发了一个数据工程框架，以自动获取进一步模型改进所需的训练数据。广泛的实验证据和详细分析表明，CounterMATH具有挑战性，表明像OpenAI O1这样的LLMs缺乏有效的反例驱动证明能力。此外，我们对模型训练的探索表明，增强LLMs的反例驱动概念推理能力对于提高它们的整体数学能力至关重要。我们认为，我们的工作为数学LLMs社区提供了新的视角。 

---
# Linking Cryptoasset Attribution Tags to Knowledge Graph Entities: An LLM-based Approach 

**Title (ZH)**: 基于大语言模型的方法：cryptoasset 归因标签与知识图谱实体的联系 

**Authors**: Régnier Avice, Bernhard Haslhofer, Zhidong Li, Jianlong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.10453)  

**Abstract**: Attribution tags form the foundation of modern cryptoasset forensics. However, inconsistent or incorrect tags can mislead investigations and even result in false accusations. To address this issue, we propose a novel computational method based on Large Language Models (LLMs) to link attribution tags with well-defined knowledge graph concepts. We implemented this method in an end-to-end pipeline and conducted experiments showing that our approach outperforms baseline methods by up to 37.4% in F1-score across three publicly available attribution tag datasets. By integrating concept filtering and blocking procedures, we generate candidate sets containing five knowledge graph entities, achieving a recall of 93% without the need for labeled data. Additionally, we demonstrate that local LLM models can achieve F1-scores of 90%, comparable to remote models which achieve 94%. We also analyze the cost-performance trade-offs of various LLMs and prompt templates, showing that selecting the most cost-effective configuration can reduce costs by 90%, with only a 1% decrease in performance. Our method not only enhances attribution tag quality but also serves as a blueprint for fostering more reliable forensic evidence. 

**Abstract (ZH)**: 基于大型语言模型的属性标签关联方法为现代加密资产取证奠定了基础。然而，不一致或错误的标签可能会误导调查，甚至导致误判。为此，我们提出了一种基于大型语言模型（LLMs）的新计算方法，将属性标签与明确的知识图谱概念关联起来。我们在端到端的管道中实现了这一方法，并进行了实验，结果显示，在三个公开的属性标签数据集中，我们的方法在F1分数上比基线方法高出了37.4%。通过结合概念过滤和阻塞程序，我们生成包含五个知识图实体的候选集，在无需标注数据的情况下实现93%的召回率。此外，我们证明了本地LLM模型可以达到90%的F1分数，与远程模型达到的94%相媲美。我们还分析了各种LLM和提示模板的成本-性能权衡，结果显示，选择最具成本效益的配置可以将成本降低90%，同时性能降幅仅为1%。我们的方法不仅提高了属性标签的质量，还为促进更可靠的 forensic 证据提供了一个范本。 

---
# Trustworthy AI on Safety, Bias, and Privacy: A Survey 

**Title (ZH)**: 可信人工智能的安全性、偏差和隐私保护：一项综述 

**Authors**: Xingli Fang, Jianwei Li, Varun Mulchandani, Jung-Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.10450)  

**Abstract**: The capabilities of artificial intelligence systems have been advancing to a great extent, but these systems still struggle with failure modes, vulnerabilities, and biases. In this paper, we study the current state of the field, and present promising insights and perspectives regarding concerns that challenge the trustworthiness of AI models. In particular, this paper investigates the issues regarding three thrusts: safety, privacy, and bias, which hurt models' trustworthiness. For safety, we discuss safety alignment in the context of large language models, preventing them from generating toxic or harmful content. For bias, we focus on spurious biases that can mislead a network. Lastly, for privacy, we cover membership inference attacks in deep neural networks. The discussions addressed in this paper reflect our own experiments and observations. 

**Abstract (ZH)**: 人工智能系统的能力有了显著进步，但仍面临故障模式、漏洞和偏见等挑战。本文研究了该领域的当前状态，并提出了关于影响人工智能模型可信度的关注问题的有前景的见解和视角。特别地，本文探讨了安全、隐私和偏见这三个方面的关键问题，这些方面损害了模型的可信度。在安全方面，我们讨论了大规模语言模型的安全对齐，防止其生成毒害或有害内容。在偏见方面，我们关注可能导致网络误导的虚假偏见。最后，在隐私方面，我们讨论了深度神经网络中的成员推理攻击。本文的讨论反映了我们自己的实验和观察结果。 

---
# Analysis of Overparameterization in Continual Learning under a Linear Model 

**Title (ZH)**: 持续学习环境中线性模型中超参数化分析 

**Authors**: Daniel Goldfarb, Paul Hand  

**Link**: [PDF](https://arxiv.org/pdf/2502.10442)  

**Abstract**: Autonomous machine learning systems that learn many tasks in sequence are prone to the catastrophic forgetting problem. Mathematical theory is needed in order to understand the extent of forgetting during continual learning. As a foundational step towards this goal, we study continual learning and catastrophic forgetting from a theoretical perspective in the simple setting of gradient descent with no explicit algorithmic mechanism to prevent forgetting. In this setting, we analytically demonstrate that overparameterization alone can mitigate forgetting in the context of a linear regression model. We consider a two-task setting motivated by permutation tasks, and show that as the overparameterization ratio becomes sufficiently high, a model trained on both tasks in sequence results in a low-risk estimator for the first task. As part of this work, we establish a non-asymptotic bound of the risk of a single linear regression task, which may be of independent interest to the field of double descent theory. 

**Abstract (ZH)**: 自主学习系统在学习多个任务时容易出现灾难性遗忘问题。为了理解连续学习过程中遗忘的程度，需要数学理论进行解释。为了实现这一目标的基础步骤，我们从理论角度研究在没有防止遗忘的显式算法机制的梯度下降简单设置下，连续学习和灾难性遗忘。在这种设置下，我们通过分析证明，过参数化本身可以在线性回归模型的上下文中减轻遗忘。我们考虑了一个由排列任务启发的两任务设置，并表明当过参数化比例足够高时，依次训练两个任务的模型能产生第一个任务的低风险估计。作为这项工作的部分，我们建立了单个线性回归任务风险的非渐进界，这在双下降理论领域可能具有独立的兴趣。 

---
# Towards Copyright Protection for Knowledge Bases of Retrieval-augmented Language Models via Ownership Verification with Reasoning 

**Title (ZH)**: 基于推理的所有权验证以实现检索增强语言模型知识库的版权保护 

**Authors**: Junfeng Guo, Yiming Li, Ruibo Chen, Yihan Wu, Chenxi Liu, Yanshuo Chen, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10440)  

**Abstract**: Large language models (LLMs) are increasingly integrated into real-world applications through retrieval-augmented generation (RAG) mechanisms to supplement their responses with up-to-date and domain-specific knowledge. However, the valuable and often proprietary nature of the knowledge bases used in RAG introduces the risk of unauthorized usage by adversaries. Existing methods that can be generalized as watermarking techniques to protect these knowledge bases typically involve poisoning attacks. However, these methods require to alter the results of verification samples (\eg, generating incorrect outputs), inevitably making them susceptible to anomaly detection and even introduce new security risks. To address these challenges, we propose \name{} for `harmless' copyright protection of knowledge bases. Instead of manipulating LLM's final output, \name{} implants distinct verification behaviors in the space of chain-of-thought (CoT) reasoning, maintaining the correctness of the final answer. Our method has three main stages: (1) \textbf{Generating CoTs}: For each verification question, we generate two CoTs, including a target CoT for building watermark behaviors; (2) \textbf{Optimizing Watermark Phrases and Target CoTs}: We optimize them to minimize retrieval errors under the black-box setting of suspicious LLM, ensuring that the watermarked verification queries activate the target CoTs without being activated in non-watermarked ones; (3) \textbf{Ownership Verification}: We exploit a pairwise Wilcoxon test to statistically verify whether a suspicious LLM is augmented with the protected knowledge base by comparing its responses to watermarked and benign verification queries. Our experiments on diverse benchmarks demonstrate that \name{} effectively protects knowledge bases against unauthorized usage while preserving the integrity and performance of the RAG. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过检索增强生成（RAG）机制越来越多地集成到实际应用中，以通过补充最新和领域特定的知识来完善其响应。然而，RAG所使用的宝贵且经常是专有的知识库性质给对手未经授权使用带来了风险。现有可以泛化的水印技术方法通常涉及投毒攻击。然而，这些方法需要修改验证样本的结果（例如，生成错误输出），从而不可避免地使其容易被异常检测，并可能引入新的安全风险。为了解决这些挑战，我们提出了\name{}用于知识库的“无害”版权保护。该方法不操纵LLM的最终输出，而是通过思维链（CoT）推理的空间植入不同的验证行为，保持最终答案的正确性。该方法主要有三个阶段：（1）生成思维链：对于每个验证问题，我们生成两个思维链，包括用于构建水印行为的目标思维链；（2）优化水印短语和目标思维链：在可疑LLM的黑色盒设置下，优化它们以最小化检索错误，确保具有水印的验证查询激活目标思维链而不激活非水印的思维链；（3）所有权验证：我们利用Wilcoxon配对秩检验统计验证可疑LLM是否被增强使用了受保护的知识库，方法是将其对水印和良性验证查询的响应进行比较。我们在多种基准测试上的实验表明，\name{}有效地保护了知识库免于未经授权的使用，同时保持了RAG的完整性和性能。 

---
# Crypto Miner Attack: GPU Remote Code Execution Attacks 

**Title (ZH)**: GPU远程代码执行攻击：Crypto Miner攻击 

**Authors**: Ariel Szabo, Uzy Hadad  

**Link**: [PDF](https://arxiv.org/pdf/2502.10439)  

**Abstract**: Remote Code Execution (RCE) exploits pose a significant threat to AI and ML systems, particularly in GPU-accelerated environments where the computational power of GPUs can be misused for malicious purposes. This paper focuses on RCE attacks leveraging deserialization vulnerabilities and custom layers, such as TensorFlow Lambda layers, which are often overlooked due to the complexity of monitoring GPU workloads. These vulnerabilities enable attackers to execute arbitrary code, blending malicious activity seamlessly into expected model behavior and exploiting GPUs for unauthorized tasks such as cryptocurrency mining. Unlike traditional CPU-based attacks, the parallel processing nature of GPUs and their high resource utilization make runtime detection exceptionally challenging. In this work, we provide a comprehensive examination of RCE exploits targeting GPUs, demonstrating an attack that utilizes these vulnerabilities to deploy a crypto miner on a GPU. We highlight the technical intricacies of such attacks, emphasize their potential for significant financial and computational costs, and propose strategies for mitigation. By shedding light on this underexplored attack vector, we aim to raise awareness and encourage the adoption of robust security measures in GPU-driven AI and ML systems, with an emphasis on static and model scanning as an easier way to detect exploits. 

**Abstract (ZH)**: Remote Code Execution攻击对AI和ML系统，特别是在GPU加速环境中，构成了显著威胁，因为GPU的计算能力可能被滥用以实现恶意目的。本文聚焦于利用反序列化漏洞和自定义层（如TensorFlow Lambda层）的RCE攻击，这些漏洞因监测GPU工作负载的复杂性而常被忽视。这些漏洞使攻击者能够执行任意代码，使其恶意活动无缝融入预期的模型行为，并利用GPU执行未经授权的任务，如加密货币挖掘。与传统的基于CPU的攻击不同，GPU并行处理的性质及其高资源利用率使其在运行时检测变得异常困难。在这项工作中，我们全面分析了针对GPU的RCE攻击，展示了利用这些漏洞部署加密货币挖掘程序的攻击方法。我们强调了此类攻击的技术复杂性，强调了它们对重大财务和计算成本的影响，并提出了缓解策略。通过揭示这一未被充分探索的攻击向量，我们旨在提高意识并鼓励在以GPU为主导的AI和ML系统中采用 robust的安全措施，强调静态和模型扫描作为检测攻击的一种更简单方法。 

---
# Injecting Universal Jailbreak Backdoors into LLMs in Minutes 

**Title (ZH)**: 在几分钟内向LLMs注入通用型 jailbreak 后门 

**Authors**: Zhuowei Chen, Qiannan Zhang, Shichao Pei  

**Link**: [PDF](https://arxiv.org/pdf/2502.10438)  

**Abstract**: Jailbreak backdoor attacks on LLMs have garnered attention for their effectiveness and stealth. However, existing methods rely on the crafting of poisoned datasets and the time-consuming process of fine-tuning. In this work, we propose JailbreakEdit, a novel jailbreak backdoor injection method that exploits model editing techniques to inject a universal jailbreak backdoor into safety-aligned LLMs with minimal intervention in minutes. JailbreakEdit integrates a multi-node target estimation to estimate the jailbreak space, thus creating shortcuts from the backdoor to this estimated jailbreak space that induce jailbreak actions. Our attack effectively shifts the models' attention by attaching strong semantics to the backdoor, enabling it to bypass internal safety mechanisms. Experimental results show that JailbreakEdit achieves a high jailbreak success rate on jailbreak prompts while preserving generation quality, and safe performance on normal queries. Our findings underscore the effectiveness, stealthiness, and explainability of JailbreakEdit, emphasizing the need for more advanced defense mechanisms in LLMs. 

**Abstract (ZH)**: 基于模型编辑的Jailbreak后门注入方法JailbreakEdit在安全对齐的大语言模型中实现了高效、隐蔽的后门攻击，而无需大量干预和耗时的微调过程。 

---
# MERGE$^3$: Efficient Evolutionary Merging on Consumer-grade GPUs 

**Title (ZH)**: MERGE³: 高效的消费者级GPU上进化合并 

**Authors**: Tommaso Mencattini, Adrian Robert Minut, Donato Crisostomi, Andrea Santilli, Emanuele Rodolà  

**Link**: [PDF](https://arxiv.org/pdf/2502.10436)  

**Abstract**: Evolutionary model merging enables the creation of high-performing multi-task models but remains computationally prohibitive for consumer hardware. We introduce MERGE$^3$, an efficient framework that makes evolutionary merging feasible on a single GPU by reducing fitness computation costs 50$\times$ while preserving performance. MERGE$^3$ achieves this by Extracting a reduced dataset for evaluation, Estimating model abilities using Item Response Theory (IRT), and Evolving optimal merges via IRT-based performance estimators. Our method enables state-of-the-art multilingual and cross-lingual merging, transferring knowledge across languages with significantly lower computational overhead. We provide theoretical guarantees and an open-source library, democratizing high-quality model merging. 

**Abstract (ZH)**: 进化模型合并使高性能多任务模型的创建成为可能，但由于计算上的限制，仍难以在消费级硬件上实现。我们介绍了MERGE$^3$，这是一个高效的框架，通过将 fitness 计算成本减少50倍同时保持性能，使其在单个GPU上变得可行。MERGE$^3$ 通过提取用于评估的减小数据集、使用项目反应理论（IRT）估计模型能力以及通过基于IRT的性能估算器进行进化最优合并，实现了这一点。我们的方法使得最先进的多语言和跨语言合并成为可能，以显著降低的计算开销将知识跨语言转移。我们提供了理论保证并开源了相关库，使高质量模型合并变得更加普及。 

---
# RAMer: Reconstruction-based Adversarial Model for Multi-party Multi-modal Multi-label Emotion Recognition 

**Title (ZH)**: RAMer: 基于重建的 adversarial 模型多当事人多模态多标签情绪识别 

**Authors**: Xudong Yang, Yizhang Zhu, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.10435)  

**Abstract**: Conventional multi-modal multi-label emotion recognition (MMER) from videos typically assumes full availability of visual, textual, and acoustic modalities. However, real-world multi-party settings often violate this assumption, as non-speakers frequently lack acoustic and textual inputs, leading to a significant degradation in model performance. Existing approaches also tend to unify heterogeneous modalities into a single representation, overlooking each modality's unique characteristics. To address these challenges, we propose RAMer (Reconstruction-based Adversarial Model for Emotion Recognition), which leverages adversarial learning to refine multi-modal representations by exploring both modality commonality and specificity through reconstructed features enhanced by contrastive learning. RAMer also introduces a personality auxiliary task to complement missing modalities using modality-level attention, improving emotion reasoning. To further strengthen the model's ability to capture label and modality interdependency, we propose a stack shuffle strategy to enrich correlations between labels and modality-specific features. Experiments on three benchmarks, i.e., MEmoR, CMU-MOSEI, and $M^3$ED, demonstrate that RAMer achieves state-of-the-art performance in dyadic and multi-party MMER scenarios. 

**Abstract (ZH)**: 基于重建的对抗模型在多模态多标签情感识别中的应用：应对非说话者缺失声学和文本输入的挑战 

---
# Leveraging Constraint Violation Signals For Action-Constrained Reinforcement Learning 

**Title (ZH)**: 利用约束违反信号进行行动受限强化学习 

**Authors**: Janaka Chathuranga Brahmanage, Jiajing Ling, Akshat Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.10431)  

**Abstract**: In many RL applications, ensuring an agent's actions adhere to constraints is crucial for safety. Most previous methods in Action-Constrained Reinforcement Learning (ACRL) employ a projection layer after the policy network to correct the action. However projection-based methods suffer from issues like the zero gradient problem and higher runtime due to the usage of optimization solvers. Recently methods were proposed to train generative models to learn a differentiable mapping between latent variables and feasible actions to address this issue. However, generative models require training using samples from the constrained action space, which itself is challenging. To address such limitations, first, we define a target distribution for feasible actions based on constraint violation signals, and train normalizing flows by minimizing the KL divergence between an approximated distribution over feasible actions and the target. This eliminates the need to generate feasible action samples, greatly simplifying the flow model learning. Second, we integrate the learned flow model with existing deep RL methods, which restrict it to exploring only the feasible action space. Third, we extend our approach beyond ACRL to handle state-wise constraints by learning the constraint violation signal from the environment. Empirically, our approach has significantly fewer constraint violations while achieving similar or better quality in several control tasks than previous best methods. 

**Abstract (ZH)**: 基于约束的强化学习中确保智能体行动符合约束的正常化流方法 

---
# Real Time Control of Tandem-Wing Experimental Platform Using Concerto Reinforcement Learning 

**Title (ZH)**: 使用Concerto强化学习的 tandem-wing 实验平台实时控制 

**Authors**: Zhang Minghao, Yang Xiaojun, Wang Zhihe, Wang Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10429)  

**Abstract**: This paper introduces the CRL2RT algorithm, an advanced reinforcement learning method aimed at improving the real-time control performance of the Direct-Drive Tandem-Wing Experimental Platform (DDTWEP). Inspired by dragonfly flight, DDTWEP's tandem wing structure causes nonlinear and unsteady aerodynamic interactions, leading to complex load behaviors during pitch, roll, and yaw maneuvers. These complexities challenge stable motion control at high frequencies (2000 Hz). To overcome these issues, we developed the CRL2RT algorithm, which combines classical control elements with reinforcement learning-based controllers using a time-interleaved architecture and a rule-based policy composer. This integration ensures finite-time convergence and single-life adaptability. Experimental results under various conditions, including different flapping frequencies and yaw disturbances, show that CRL2RT achieves a control frequency surpassing 2500 Hz on standard CPUs. Additionally, when integrated with classical controllers like PID, Adaptive PID, and Model Reference Adaptive Control (MRAC), CRL2RT enhances tracking performance by 18.3% to 60.7%. These findings demonstrate CRL2RT's broad applicability and superior performance in complex real-time control scenarios, validating its effectiveness in overcoming existing control strategy limitations and advancing robust, efficient real-time control for biomimetic aerial vehicles. 

**Abstract (ZH)**: 本文介绍了CRL2RT算法，这是一种旨在提高直接驱动串联翼实验平台（DDTWEP）实时控制性能的先进强化学习方法。受蜻蜓飞行启发，DDTWEP的串联翼结构导致了非线性和不稳定的气动相互作用，使得在滚转、俯仰和偏航机动过程中出现复杂的载荷行为。这些复杂性对高频（2000 Hz）下的稳定运动控制构成了挑战。为了克服这些问题，我们开发了CRL2RT算法，该算法结合了经典的控制元素和基于强化学习的控制器，采用时间交错架构和基于规则的策略合成器。这种集成确保了有限时间收敛和单寿命适应性。在不同 CONDITIONS（包括不同的拍翼频率和偏航干扰）下的实验结果表明，CRL2RT在标准CPU上实现了超过2500 Hz的控制频率。此外，当与PID、自适应PID和模型参考自适应控制（MRAC）等经典控制器集成时，CRL2RT能提高跟踪性能18.3%至60.7%。这些发现展示了CRL2RT在复杂实时控制场景中的广泛应用性和优越性能，验证了其在克服现有控制策略局限性、推进仿生飞行器鲁棒高效实时控制方面的作用。 

---
# Neuron Platonic Intrinsic Representation From Dynamics Using Contrastive Learning 

**Title (ZH)**: 基于对比学习的神经元柏拉图文内在动力表示 

**Authors**: Wei Wu, Can Liao, Zizhen Deng, Zhengrui Guo, Jinzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10425)  

**Abstract**: The Platonic Representation Hypothesis suggests a universal, modality-independent reality representation behind different data modalities. Inspired by this, we view each neuron as a system and detect its multi-segment activity data under various peripheral conditions. We assume there's a time-invariant representation for the same neuron, reflecting its intrinsic properties like molecular profiles, location, and morphology. The goal of obtaining these intrinsic neuronal representations has two criteria: (I) segments from the same neuron should have more similar representations than those from different neurons; (II) the representations must generalize well to out-of-domain data. To meet these, we propose the NeurPIR (Neuron Platonic Intrinsic Representation) framework. It uses contrastive learning, with segments from the same neuron as positive pairs and those from different neurons as negative pairs. In implementation, we use VICReg, which focuses on positive pairs and separates dissimilar samples via regularization. We tested our method on Izhikevich model-simulated neuronal population dynamics data. The results accurately identified neuron types based on preset hyperparameters. We also applied it to two real-world neuron dynamics datasets with neuron type annotations from spatial transcriptomics and neuron locations. Our model's learned representations accurately predicted neuron types and locations and were robust on out-of-domain data (from unseen animals). This shows the potential of our approach for understanding neuronal systems and future neuroscience research. 

**Abstract (ZH)**: Platonic Representation Hypothesis启发的神经元本原内在表示研究：NeurPIR框架的应用 

---
# QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache 

**Title (ZH)**: QuantSpec: 嵌套量化KV缓存的自推测解码 

**Authors**: Rishabh Tiwari, Haocheng Xi, Aditya Tomar, Coleman Hooper, Sehoon Kim, Maxwell Horton, Mahyar Najibi, Michael W. Mahoney, Kurt Keutzer, Amir Gholami  

**Link**: [PDF](https://arxiv.org/pdf/2502.10424)  

**Abstract**: Large Language Models (LLMs) are increasingly being deployed on edge devices for long-context settings, creating a growing need for fast and efficient long-context inference. In these scenarios, the Key-Value (KV) cache is the primary bottleneck in terms of both GPU memory and latency, as the full KV cache must be loaded for each decoding step. While speculative decoding is a widely accepted technique to accelerate autoregressive decoding, existing methods often struggle to achieve significant speedups due to inefficient KV cache optimization strategies and result in low acceptance rates. To address these challenges, we propose a novel self-speculative decoding framework, QuantSpec, where the draft model shares the architecture of the target model but employs a hierarchical 4-bit quantized KV cache and 4-bit quantized weights for acceleration. QuantSpec maintains high acceptance rates ($>$90%) and reliably provides consistent end-to-end speedups upto $\sim2.5\times$, outperforming other self-speculative decoding methods that use sparse KV cache for long-context LLM inference. QuantSpec also reduces the memory requirements by $\sim 1.3\times$ compared to these alternatives. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地在边缘设备上用于长上下文设置，这产生了对快速高效长上下文推理的日益增长的需求。在这种场景中，键值（KV）缓存是both GPU内存和延迟的主要瓶颈，因为每个解码步骤都需要加载完整的KV缓存。尽管推测性解码是加速自回归解码的广泛接受的技术，但现有方法往往由于不高效的KV缓存优化策略而难以实现显著的加速，并且接受率较低。为了解决这些问题，我们提出了一种新颖的自我推测性解码框架QuantSpec，其中草稿模型与目标模型共享架构，但使用分层的4位量化KV缓存和4位量化权重来加速。QuantSpec保持高接受率（>90%）并可靠地提供了端到端速度提升，最高可达约2.5倍，优于其他使用稀疏KV缓存进行长上下文LLM推理的自我推测性解码方法。此外，QuantSpec将内存要求降低了约1.3倍比这些替代方案。 

---
# DA-LIF: Dual Adaptive Leaky Integrate-and-Fire Model for Deep Spiking Neural Networks 

**Title (ZH)**: DA-LIF: 双适应漏积分与放电模型用于深度脉冲神经网络 

**Authors**: Tianqing Zhang, Kairong Yu, Jian Zhang, Hongwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10422)  

**Abstract**: Spiking Neural Networks (SNNs) are valued for their ability to process spatio-temporal information efficiently, offering biological plausibility, low energy consumption, and compatibility with neuromorphic hardware. However, the commonly used Leaky Integrate-and-Fire (LIF) model overlooks neuron heterogeneity and independently processes spatial and temporal information, limiting the expressive power of SNNs. In this paper, we propose the Dual Adaptive Leaky Integrate-and-Fire (DA-LIF) model, which introduces spatial and temporal tuning with independently learnable decays. Evaluations on both static (CIFAR10/100, ImageNet) and neuromorphic datasets (CIFAR10-DVS, DVS128 Gesture) demonstrate superior accuracy with fewer timesteps compared to state-of-the-art methods. Importantly, DA-LIF achieves these improvements with minimal additional parameters, maintaining low energy consumption. Extensive ablation studies further highlight the robustness and effectiveness of the DA-LIF model. 

**Abstract (ZH)**: 双适应泄漏积分-放电（DA-LIF）模型：提升时空调谐的脉冲神经网络 

---
# DRiVE: Dynamic Recognition in VEhicles using snnTorch 

**Title (ZH)**: DRiVE: 动态识别在车辆中的应用 using snnTorch 

**Authors**: Heerak Vora, Param Pathak, Parul Bakaraniya  

**Link**: [PDF](https://arxiv.org/pdf/2502.10421)  

**Abstract**: Spiking Neural Networks (SNNs) mimic biological brain activity, processing data efficiently through an event-driven design, wherein the neurons activate only when inputs exceed specific thresholds. Their ability to track voltage changes over time via membrane potential dynamics helps retain temporal information. This study combines SNNs with PyTorch's adaptable framework, snnTorch, to test their potential for image-based tasks. We introduce DRiVE, a vehicle detection model that uses spiking neuron dynamics to classify images, achieving 94.8% accuracy and a near-perfect 0.99 AUC score. These results highlight DRiVE's ability to distinguish vehicle classes effectively, challenging the notion that SNNs are limited to temporal data. As interest grows in energy-efficient neural models, DRiVE's success emphasizes the need to refine SNN optimization for visual tasks. This work encourages broader exploration of SNNs in scenarios where conventional networks struggle, particularly for real-world applications requiring both precision and efficiency. 

**Abstract (ZH)**: Spiking Neural Networks (SNNs)模拟生物脑活动，通过事件驱动的设计高效处理数据，其中神经元仅在输入超过特定阈值时激活。它们通过膜电位动力学追踪电压变化，有助于保留时间信息。本研究将SNNs与PyTorch的可拓展框架snnTorch结合，测试其在基于图像任务中的潜力。我们引入DRiVE，一种使用突触神经元动力学进行图像分类的车辆检测模型，实现了94.8%的准确率和接近完美的0.99 AUC分数。这些结果突显了DRiVE有效区分车辆类别的能力，挑战了SNNs仅限于时序数据的观念。随着对高效神经模型兴趣的增长，DRiVE的成功强调了需要为视觉任务优化SNNs的重要性。本研究鼓励在传统网络表现不佳的场景中更广泛地探索SNNs，尤其是对于需要精确性和效率的现实世界应用。 

---
# A Hybrid Swarm Intelligence Approach for Optimizing Multimodal Large Language Models Deployment in Edge-Cloud-based Federated Learning Environments 

**Title (ZH)**: 基于边缘-云联邦学习环境的多模态大型语言模型部署的混合 swarm 智能优化方法 

**Authors**: Gaith Rjouba, Hanae Elmekki, Saidul Islam, Jamal Bentahar, Rachida Dssouli  

**Link**: [PDF](https://arxiv.org/pdf/2502.10419)  

**Abstract**: The combination of Federated Learning (FL), Multimodal Large Language Models (MLLMs), and edge-cloud computing enables distributed and real- time data processing while preserving privacy across edge devices and cloud infrastructure. However, the deployment of MLLMs in FL environments with resource-constrained edge devices presents significant challenges, in- cluding resource management, communication overhead, and non-IID data. To address these challenges, we propose a novel hybrid framework wherein MLLMs are deployed on edge devices equipped with sufficient resources and battery life, while the majority of training occurs in the cloud. To identify suitable edge devices for deployment, we employ Particle Swarm Optimiza- tion (PSO), and Ant Colony Optimization (ACO) is utilized to optimize the transmission of model updates between edge and cloud nodes. This proposed swarm intelligence-based framework aims to enhance the efficiency of MLLM training by conducting extensive training in the cloud and fine-tuning at the edge, thereby reducing energy consumption and communication costs. Our experimental results show that the proposed method significantly improves system performance, achieving an accuracy of 92%, reducing communica- tion cost by 30%, and enhancing client participation compared to traditional FL methods. These results make the proposed approach highly suitable for large-scale edge-cloud computing systems. 

**Abstract (ZH)**: 联邦学习、多模态大语言模型和边缘-云计算的结合 enables 边缘设备和云基础设施之间的分布式和实时数据处理并保护隐私。然而，在资源受限的边缘设备上部署多模态大语言模型（MLLMs）在联邦学习环境中带来了显著挑战，包括资源管理、通信开销和非IID数据问题。为应对这些挑战，我们提出了一种新型混合框架，在装备有充足资源和电池寿命的边缘设备上部署多模态大语言模型，而大部分训练则在云中进行。为了确定适合部署的边缘设备，我们采用粒子 swarm 优化（PSO）进行设备选择，利用蚁群优化（ACO）优化边缘和云节点之间模型更新的传输。这种基于群智的框架旨在通过在云中进行广泛的训练并在边缘进行微调来提高多模态大语言模型训练的效率，从而减少能源消耗和通信成本。实验结果表明，所提出的方法显著提高了系统性能，准确率达到92%，通信成本降低30%，并且增强了客户端参与度，相比传统联邦学习方法更具优势。这对于大规模边缘-云计算系统来说是非常合适的。 

---
# Evolutionary Power-Aware Routing in VANETs using Monte-Carlo Simulation 

**Title (ZH)**: 使用蒙特卡洛模拟的进化功率感知路由在VANET中 

**Authors**: J. Toutouh, S. Nesmachnow, E. Alba  

**Link**: [PDF](https://arxiv.org/pdf/2502.10417)  

**Abstract**: This work addresses the reduction of power consumption of the AODV routing protocol in vehicular networks as an optimization problem. Nowadays, network designers focus on energy-aware communication protocols, specially to deploy wireless networks. Here, we introduce an automatic method to search for energy-efficient AODV configurations by using an evolutionary algorithm and parallel Monte-Carlo simulations to improve the accuracy of the evaluation of tentative solutions. The experimental results demonstrate that significant power consumption improvements over the standard configuration can be attained, with no noteworthy loss in the quality of service. 

**Abstract (ZH)**: 本工作将AODV路由协议在 vehicular 网络中的功耗降低问题作为一个优化问题进行研究。本文介绍了一种自动方法，通过使用进化算法和并行蒙特卡洛模拟来搜索能效更高的 AODV 配置，从而提高候选解决方案评估的准确性。实验结果表明，与标准配置相比，可以实现显著的功耗改善，同时服务质量并无明显下降。 

---
# Machine Learning-Driven Convergence Analysis in Multijurisdictional Compliance Using BERT and K-Means Clustering 

**Title (ZH)**: 基于BERT和K-Means聚类的多辖区合规性驱动的机器学习融合分析 

**Authors**: Raj Sonani, Lohalekar Prayas  

**Link**: [PDF](https://arxiv.org/pdf/2502.10413)  

**Abstract**: Digital data continues to grow, there has been a shift towards using effective regulatory mechanisms to safeguard personal information. The CCPA of California and the General Data Protection Regulation (GDPR) of the European Union are two of the most important privacy laws. The regulation is intended to safeguard consumer privacy, but it varies greatly in scope, definitions, and methods of enforcement. This paper presents a fresh approach to adaptive compliance, using machine learning and emphasizing natural language processing (NLP) as the primary focus of comparison between the GDPR and CCPA. Using NLP, this study compares various regulations to identify areas where they overlap or diverge. This includes the "right to be forgotten" provision in the GDPR and the "opt-out of sale" provision under CCPA. International companies can learn valuable lessons from this report, as it outlines strategies for better enforcement of laws across different nations. Additionally, the paper discusses the challenges of utilizing NLP in legal literature and proposes methods to enhance the model-ability of machine learning models for studying regulations. The study's objective is to "bridge the gap between legal knowledge and technical expertise" by developing regulatory compliance strategies that are more efficient in operation and more effective in data protection. 

**Abstract (ZH)**: 数字数据持续增长，监管机制的有效性已成为保护个人隐私的关键。加利福尼亚州 CCPA 和欧盟 GDPR 是最重要的隐私法律。本文提出了一种适应性合规的新方法，侧重于使用机器学习和自然语言处理（NLP）来比较 GDPR 和 CCPA。通过 NLP，本研究比较各种法规以识别它们重叠或不同之处，包括 GDPR 中的“被遗忘权”条款和 CCPA 中的“不销售选择退出”条款。国际公司可以从本报告中学习宝贵的经验，因为它概述了在不同国家更好地执行法律的战略。此外，本文讨论了在法律文献中利用 NLP 的挑战，并提出了提高机器学习模型研究法规能力的方法。研究的目的是“弥合法律知识和技术专长之间的差距”，通过开发更高效的合规策略和更有效的数据保护策略。 

---
# Identifying relevant indicators for monitoring a National Artificial Intelligence Strategy 

**Title (ZH)**: 识别监测国家人工智能战略的相关指标 

**Authors**: Renata Pelissari, Ricardo Suyama, Leonardo Tomazeli Duarte, Henrique Sá Earp  

**Link**: [PDF](https://arxiv.org/pdf/2502.10412)  

**Abstract**: How can a National Artificial Intelligence Strategy be effectively monitored? To address this question, we propose a methodology consisting of two key components. First, it involves identifying relevant indicators within national AI strategies. Second, it assesses the alignment between these indicators and the strategic actions of a specific government's AI strategy, allowing for a critical evaluation of its monitoring measures. Moreover, identifying these indicators helps assess the overall quality of the strategy's structure. A lack of alignment between strategic actions and the identified indicators may reveal gaps or blind spots in the strategy. This methodology is demonstrated using the Brazilian AI strategy as a case study. 

**Abstract (ZH)**: 如何有效监控国家人工智能战略？为回答这一问题，我们提出了一种方法论，该方法论包含两个关键组成部分。首先，识别国家级人工智能战略中的相关指标。其次，评估这些指标与特定政府人工智能战略行动的一致性，从而对其监控措施进行批判性评估。此外，识别这些指标有助于评估战略结构的整体质量。战略行动与识别出的指标之间缺乏一致性可能揭示战略中的缺口或盲区。该方法论以巴西人工智能战略为例进行了示范。 

---
# TrueReason: An Exemplar Personalised Learning System Integrating Reasoning with Foundational Models 

**Title (ZH)**: TrueReason：一种结合推理与基础模型的示例个性化学习系统 

**Authors**: Sahan Bulathwela, Daniel Van Niekerk, Jarrod Shipton, Maria Perez-Ortiz, Benjamin Rosman, John Shawe-Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2502.10411)  

**Abstract**: Personalised education is one of the domains that can greatly benefit from the most recent advances in Artificial Intelligence (AI) and Large Language Models (LLM). However, it is also one of the most challenging applications due to the cognitive complexity of teaching effectively while personalising the learning experience to suit independent learners. We hypothesise that one promising approach to excelling in such demanding use cases is using a \emph{society of minds}. In this chapter, we present TrueReason, an exemplar personalised learning system that integrates a multitude of specialised AI models that can mimic micro skills that are composed together by a LLM to operationalise planning and reasoning. The architecture of the initial prototype is presented while describing two micro skills that have been incorporated in the prototype. The proposed system demonstrates the first step in building sophisticated AI systems that can take up very complex cognitive tasks that are demanded by domains such as education. 

**Abstract (ZH)**: 个性化教育是最新人工智能（AI）和大型语言模型（LLM）进步可以大大受益的一个领域，但同时也是一项最具挑战性的应用之一，因为有效地进行个性化教学以适应独立学习者的需求认知复杂性很高。我们假设在这种严苛的应用场景中取得优异成果的方法之一是使用“联合心智群”。在本章中，我们介绍TrueReason，这是一个范例性的个性化学习系统，集成了多种专门的AI模型，这些模型能够模拟大型语言模型组合的微技能，以实现规划和推理的运作。我们描述了原型的架构，并介绍了已经整合到该原型中的两种微技能。所提出系统展示了构建能够承担如教育等領域所需极其复杂的认知任务的高级AI系统的初步步骤。 

---
# Auto-Evaluation: A Critical Measure in Driving Improvements in Quality and Safety of AI-Generated Lesson Resources 

**Title (ZH)**: 自动评估：驱动人工智能生成课程资源质量与安全改进的关键指标 

**Authors**: Hannah-Beth Clark, Margaux Dowland, Laura Benton, Reka Budai, Ibrahim Kaan Keskin, Emma Searle, Matthew Gregory, Mark Hodierne, William Gayne, John Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2502.10410)  

**Abstract**: As a publicly funded body in the UK, Oak National Academy is in a unique position to innovate within this field as we have a comprehensive curriculum of approximately 13,000 open education resources (OER) for all National Curriculum subjects, designed and quality-assured by expert, human teachers. This has provided the corpus of content needed for building a high-quality AI-powered lesson planning tool, Aila, that is free to use and, therefore, accessible to all teachers across the country. Furthermore, using our evidence-informed curriculum principles, we have codified and exemplified each component of lesson design. To assess the quality of lessons produced by Aila at scale, we have developed an AI-powered auto-evaluation agent,facilitating informed improvements to enhance output quality. Through comparisons between human and auto-evaluations, we have begun to refine this agent further to increase its accuracy, measured by its alignment with an expert human evaluator. In this paper we present this iterative evaluation process through an illustrative case study focused on one quality benchmark - the level of challenge within multiple-choice quizzes. We also explore the contribution that this may make to similar projects and the wider sector. 

**Abstract (ZH)**: 英国公共资助机构橡国 NATIONAL ACADEMY 在开源教育资源领域的创新及其高质量AI辅助教学规划工具Aila的研发与评估：以多项选择题难度水平为质量基准的案例研究及对该领域类似项目和更广泛领域的贡献探討 

---
# Data Science Students Perspectives on Learning Analytics: An Application of Human-Led and LLM Content Analysis 

**Title (ZH)**: 数据科学学生对学习分析的视角：一种基于人类主导和大规模语言模型内容分析的应用 

**Authors**: Raghda Zahran, Jianfei Xu, Huizhi Liang, Matthew Forshaw  

**Link**: [PDF](https://arxiv.org/pdf/2502.10409)  

**Abstract**: Objective This study is part of a series of initiatives at a UK university designed to cultivate a deep understanding of students' perspectives on analytics that resonate with their unique learning needs. It explores collaborative data processing undertaken by postgraduate students who examined an Open University Learning Analytics Dataset (OULAD).
Methods A qualitative approach was adopted, integrating a Retrieval-Augmented Generation (RAG) and a Large Language Model (LLM) technique with human-led content analysis to gather information about students' perspectives based on their submitted work. The study involved 72 postgraduate students in 12 groups.
Findings The analysis of group work revealed diverse insights into essential learning analytics from the students' perspectives. All groups adopted a structured data science methodology. The questions formulated by the groups were categorised into seven themes, reflecting their specific areas of interest. While there was variation in the selected variables to interpret correlations, a consensus was found regarding the general results.
Conclusion A significant outcome of this study is that students specialising in data science exhibited a deeper understanding of learning analytics, effectively articulating their interests through inferences drawn from their analyses. While human-led content analysis provided a general understanding of students' perspectives, the LLM offered nuanced insights. 

**Abstract (ZH)**: 研究目标：本研究是英国某大学开展的一系列旨在培养学生对数据分析深刻理解的举措之一，专注于探索满足学生独特学习需求的学生视角。研究重点在于分析了研究生在研究开放大学学习分析数据集（OULAD）时进行的合作数据分析过程。

研究方法：采用定性研究方法，结合检索增强生成（RAG）和大型语言模型（LLM）技术与人工主导的内容分析，基于学生提交的工作获取有关学生视角的信息。研究涉及12组共72名研究生。

研究发现：对小组工作的分析揭示了学生从多种视角对学习分析的重要见解。所有小组均采用了结构化的数据科学方法。小组提出的问题被归类为七个主题，反映了他们各自的研究兴趣。虽然选择用于解析相关性的变量有所不同，但对于一般结果，小组达成了共识。

研究结论：本研究的重要成果是，专注于数据科学的学生表现出对学习分析的更深刻理解，通过从分析中得出的推断有效表达了他们的兴趣。虽然人工主导的内容分析提供了对学生视角的一般理解，但LLM提供了更为细致的洞见。 

---
# Knowledge Tracing in Programming Education Integrating Students' Questions 

**Title (ZH)**: 编程教育中结合学生问题的 Knowledge Tracing 

**Authors**: Doyoun Kim, Suin Kim, Yojan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2502.10408)  

**Abstract**: Knowledge tracing (KT) in programming education presents unique challenges due to the complexity of coding tasks and the diverse methods students use to solve problems. Although students' questions often contain valuable signals about their understanding and misconceptions, traditional KT models often neglect to incorporate these questions as inputs to address these challenges. This paper introduces SQKT (Students' Question-based Knowledge Tracing), a knowledge tracing model that leverages students' questions and automatically extracted skill information to enhance the accuracy of predicting students' performance on subsequent problems in programming education. Our method creates semantically rich embeddings that capture not only the surface-level content of the questions but also the student's mastery level and conceptual understanding. Experimental results demonstrate SQKT's superior performance in predicting student completion across various Python programming courses of differing difficulty levels. In in-domain experiments, SQKT achieved a 33.1\% absolute improvement in AUC compared to baseline models. The model also exhibited robust generalization capabilities in cross-domain settings, effectively addressing data scarcity issues in advanced programming courses. SQKT can be used to tailor educational content to individual learning needs and design adaptive learning systems in computer science education. 

**Abstract (ZH)**: 基于学生提问的知识追踪（SQKT）在编程教育中的应用 

---
# Addressing Bias in Generative AI: Challenges and Research Opportunities in Information Management 

**Title (ZH)**: 治理生成式AI中的偏见：信息管理中的挑战与研究机遇 

**Authors**: Xiahua Wei, Naveen Kumar, Han Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10407)  

**Abstract**: Generative AI technologies, particularly Large Language Models (LLMs), have transformed information management systems but introduced substantial biases that can compromise their effectiveness in informing business decision-making. This challenge presents information management scholars with a unique opportunity to advance the field by identifying and addressing these biases across extensive applications of LLMs. Building on the discussion on bias sources and current methods for detecting and mitigating bias, this paper seeks to identify gaps and opportunities for future research. By incorporating ethical considerations, policy implications, and sociotechnical perspectives, we focus on developing a framework that covers major stakeholders of Generative AI systems, proposing key research questions, and inspiring discussion. Our goal is to provide actionable pathways for researchers to address bias in LLM applications, thereby advancing research in information management that ultimately informs business practices. Our forward-looking framework and research agenda advocate interdisciplinary approaches, innovative methods, dynamic perspectives, and rigorous evaluation to ensure fairness and transparency in Generative AI-driven information systems. We expect this study to serve as a call to action for information management scholars to tackle this critical issue, guiding the improvement of fairness and effectiveness in LLM-based systems for business practice. 

**Abstract (ZH)**: 生成式AI技术，特别是大规模语言模型（LLMs），已经改变了信息管理系统，但同时也引入了显著的偏见，这些偏见可能会影响LLMs在商业决策制定中的有效性。这一挑战为信息管理学者提供了一个独特的机会，通过识别和解决LLMs广泛应用中的偏见，推动该领域的进步。基于对偏见来源及现有偏见检测和缓解方法的讨论，本文旨在识别未来研究的不足之处和机会。通过纳入伦理考量、政策含义和社会技术视角，我们集中于开发一个覆盖生成式AI系统主要利益相关者的框架，提出关键研究问题，并激发讨论。我们的目标是为研究人员提供实际路径，以解决LLMs应用中的偏见，从而推动信息管理研究，最终指导商业实践。前瞻性的框架和研究议程倡导跨学科的方法、创新的方法、动态观点和严格的评估，以确保生成式AI驱动的信息系统中的公平性和透明度。我们期望这项研究能够成为信息管理学者采取行动的号召，指导基于LLM系统的公平性和有效性改进，以服务于商业实践。 

---
# FishBargain: An LLM-Empowered Bargaining Agent for Online Fleamarket Platform Sellers 

**Title (ZH)**: FishBargain：一个由LLM赋能的在线跳蚤市场卖家讨价还价代理系统 

**Authors**: Dexin Kong, Xu Yan, Ming Chen, Shuguang Han, Jufeng Chen, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10406)  

**Abstract**: Different from traditional Business-to-Consumer e-commerce platforms~(e.g., Amazon), online fleamarket platforms~(e.g., Craigslist) mainly focus on individual sellers who are lack of time investment and business proficiency. Individual sellers often struggle with the bargaining process and thus the deal is unaccomplished. Recent advancements in Large Language Models(LLMs) demonstrate huge potential in various dialogue tasks, but those tasks are mainly in the form of passively following user's instruction. Bargaining, as a form of proactive dialogue task, represents a distinct art of dialogue considering the dynamism of environment and uncertainty of adversary strategies. In this paper, we propose an LLM-empowered bargaining agent designed for online fleamarket platform sellers, named as FishBargain. Specifically, FishBargain understands the chat context and product information, chooses both action and language skill considering possible adversary actions and generates utterances. FishBargain has been tested by thousands of individual sellers on one of the largest online fleamarket platforms~(Xianyu) in China. Both qualitative and quantitative experiments demonstrate that FishBargain can effectively help sellers make more deals. 

**Abstract (ZH)**: 不同于传统的 Business-to-Consumer 电子商务平台（例如 Amazon），在线跳蚤市场平台（例如 Craigslist）主要关注缺乏时间投入和商业技能的个体卖家。个体卖家经常在讨价还价过程中遇到困难，导致交易无法达成。大型语言模型（LLMs）的近期进展展示了在各种对话任务中的巨大潜力，但这些任务主要以被动地遵循用户指令的形式出现。讨价还价作为一种主动的对话任务，考虑到环境的动态性和对手策略的不确定性，代表了一种独特的对话艺术。在本文中，我们提出了一种旨在帮助在线跳蚤市场平台卖家的大型语言模型驱动的讨价还价代理，名为 FishBargain。具体来说，FishBargain 理解聊天背景和产品信息，考虑可能的对手行为选择行动和语言技巧并生成话语。FishBargain 已在中国最大的在线跳蚤市场平台（Xianyu）上通过了数千名个体卖家的测试。定性和定量实验均表明，FishBargain 能够有效帮助卖家达成更多交易。 

---
# You Can't Get There From Here: Redefining Information Science to address our sociotechnical futures 

**Title (ZH)**: 无法在此基础上达到彼岸：重新定义信息科学以应对我们的社会技术未来 

**Authors**: Scott Humr, Mustafa Canan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10401)  

**Abstract**: Current definitions of Information Science are inadequate to comprehensively describe the nature of its field of study and for addressing the problems that are arising from intelligent technologies. The ubiquitous rise of artificial intelligence applications and their impact on society demands the field of Information Science acknowledge the sociotechnical nature of these technologies. Previous definitions of Information Science over the last six decades have inadequately addressed the environmental, human, and social aspects of these technologies. This perspective piece advocates for an expanded definition of Information Science that fully includes the sociotechnical impacts information has on the conduct of research in this field. Proposing an expanded definition of Information Science that includes the sociotechnical aspects of this field should stimulate both conversation and widen the interdisciplinary lens necessary to address how intelligent technologies may be incorporated into society and our lives more fairly. 

**Abstract (ZH)**: 当前信息科学的定义不足以全面描述其研究领域的特点，并应对由智能技术引起的问题。普遍兴起的人工智能应用及其对社会的影响要求信息科学领域承认这些技术的社技性质。过去六十年中信息科学的定义未能充分关注这些技术的环境、人类和社会方面。本文探讨扩展信息科学定义的必要性，以全面涵盖信息对这一领域研究活动的社会技术影响。提出包含社技方面的扩展信息科学定义应该促进讨论并扩大跨学科视角，以便更公平地将智能技术融入社会和我们的生活中。 

---
# Data Stewardship Decoded: Mapping Its Diverse Manifestations and Emerging Relevance at a time of AI 

**Title (ZH)**: 数据 stewardship 解码：映射其多样表现及其在人工智能时代的新兴相关性 

**Authors**: Stefaan Verhulst  

**Link**: [PDF](https://arxiv.org/pdf/2502.10399)  

**Abstract**: Data stewardship has become a critical component of modern data governance, especially with the growing use of artificial intelligence (AI). Despite its increasing importance, the concept of data stewardship remains ambiguous and varies in its application. This paper explores four distinct manifestations of data stewardship to clarify its emerging position in the data governance landscape. These manifestations include a) data stewardship as a set of competencies and skills, b) a function or role within organizations, c) an intermediary organization facilitating collaborations, and d) a set of guiding principles. The paper subsequently outlines the core competencies required for effective data stewardship, explains the distinction between data stewards and Chief Data Officers (CDOs), and details the intermediary role of stewards in bridging gaps between data holders and external stakeholders. It also explores key principles aligned with the FAIR framework (Findable, Accessible, Interoperable, Reusable) and introduces the emerging principle of AI readiness to ensure data meets the ethical and technical requirements of AI systems. The paper emphasizes the importance of data stewardship in enhancing data collaboration, fostering public value, and managing data reuse responsibly, particularly in the era of AI. It concludes by identifying challenges and opportunities for advancing data stewardship, including the need for standardized definitions, capacity building efforts, and the creation of a professional association for data stewardship. 

**Abstract (ZH)**: 数据 stewardship已成为现代数据治理的关键组成部分，尤其是在人工智能（AI）使用不断增加的情况下。尽管其重要性日益提高，但数据 stewardship的概念仍然模糊不清且在应用上存在差异。本文探讨了四种数据 stewardship的不同表现形式，以澄清其在数据治理格局中的新兴地位。这些表现形式包括a) 数据 stewardship作为一组能力和技能，b) 组织内的功能或角色，c) 促进合作的中间组织，以及d) 一套指导原则。本文随后概述了有效数据 stewardship所需的核心能力，解释了数据 stewardship与首席数据官（CDO）之间的区别，并详细介绍了数据 stewardship在数据持有者与外部利益相关者之间填补差距的中间角色。文章还探讨了与 FAIR 原则（可查找的、可访问的、互操作的、可重用的）一致的关键原则，并介绍了确保数据符合人工智能系统伦理和技术要求的新兴原则——AI 准备度。文章强调了数据 stewardship在增强数据协作、促进公共价值和负责任地管理数据重用方面的重要性，尤其是在人工智能时代。最后，文章指出了数据 stewardship发展面临的挑战和机遇，包括标准化定义的需求、能力培养努力，以及建立数据 stewardship专业组织的重要性。 

---
# Practical Application and Limitations of AI Certification Catalogues 

**Title (ZH)**: AI认证目录的实际应用与限制 

**Authors**: Gregor Autischer, Kerstin Waxnegger, Dominik Kowald  

**Link**: [PDF](https://arxiv.org/pdf/2502.10398)  

**Abstract**: In this work-in-progress, we investigate the certification of artificial intelligence (AI) systems, focusing on the practical application and limitations of existing certification catalogues by attempting to certify a publicly available AI system. We aim to evaluate how well current approaches work to effectively certify an AI system, and how publicly accessible AI systems, that might not be actively maintained or initially intended for certification, can be selected and used for a sample certification process. Our methodology involves leveraging the Fraunhofer AI Assessment Catalogue as a comprehensive tool to systematically assess an AI model's compliance with certification standards. We find that while the catalogue effectively structures the evaluation process, it can also be cumbersome and time-consuming to use. We observe the limitations of an AI system that has no active development team anymore and highlighted the importance of complete system documentation. Finally, we identify some limitations of the certification catalogues used and proposed ideas on how to streamline the certification process. 

**Abstract (ZH)**: 本研究进展中，我们探讨了人工智能（AI）系统的认证问题，重点关注现有认证目录的实际应用和局限性，通过尝试认证一个公开可用的AI系统来进行研究。我们旨在评估当前方法在有效认证AI系统方面的效果，并考察如何选择和利用可能未被积极维护或最初未旨在认证的公开访问AI系统进行样例认证过程。我们的方法包括利用弗劳恩霍夫AI评估目录作为全面工具来系统地评估AI模型是否符合认证标准。我们发现虽然目录有效地组织了评估过程，但使用起来也可能繁琐耗时。我们观察到一个没有任何活跃开发团队的AI系统的局限性，并强调了完整系统文档的重要性。最后，我们识别出现有认证目录的一些局限性，并提出了简化认证过程的建议。 

---
# DASKT: A Dynamic Affect Simulation Method for Knowledge Tracing 

**Title (ZH)**: DASKT：一种动态情感模拟的知识追踪方法 

**Authors**: Xinjie Sun, Kai Zhang, Qi Liu, Shuanghong Shen, Fei Wang, Yuxiang Guo, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.10396)  

**Abstract**: Knowledge Tracing (KT) predicts future performance by modeling students' historical interactions, and understanding students' affective states can enhance the effectiveness of KT, thereby improving the quality of education. Although traditional KT values students' cognition and learning behaviors, efficient evaluation of students' affective states and their application in KT still require further exploration due to the non-affect-oriented nature of the data and budget constraints. To address this issue, we propose a computation-driven approach, Dynamic Affect Simulation Knowledge Tracing (DASKT), to explore the impact of various student affective states (such as frustration, concentration, boredom, and confusion) on their knowledge states. In this model, we first extract affective factors from students' non-affect-oriented behavioral data, then use clustering and spatiotemporal sequence modeling to accurately simulate students' dynamic affect changes when dealing with different problems. Subsequently, {\color{blue}we incorporate affect with time-series analysis to improve the model's ability to infer knowledge states over time and space.} Extensive experimental results on two public real-world educational datasets show that DASKT can achieve more reasonable knowledge states under the effect of students' affective states. Moreover, DASKT outperforms the most advanced KT methods in predicting student performance. Our research highlights a promising avenue for future KT studies, focusing on achieving high interpretability and accuracy. 

**Abstract (ZH)**: 动态情感模拟知识追踪（DASKT） 

---
# An Integrated Platform for Studying Learning with Intelligent Tutoring Systems: CTAT+TutorShop 

**Title (ZH)**: 基于智能辅导系统的学习研究集成平台：CTAT+TutorShop 

**Authors**: Vincent Aleven, Conrad Borchers, Yun Huang, Tomohiro Nagashima, Bruce McLaren, Paulo Carvalho, Octav Popescu, Jonathan Sewall, Kenneth Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2502.10395)  

**Abstract**: Intelligent tutoring systems (ITSs) are effective in helping students learn; further research could make them even more effective. Particularly desirable is research into how students learn with these systems, how these systems best support student learning, and what learning sciences principles are key in ITSs. CTAT+Tutorshop provides a full stack integrated platform that facilitates a complete research lifecycle with ITSs, which includes using ITS data to discover learner challenges, to identify opportunities for system improvements, and to conduct experimental studies. The platform includes authoring tools to support and accelerate development of ITS, which provide automatic data logging in a format compatible with DataShop, an independent site that supports the analysis of ed tech log data to study student learnings. Among the many technology platforms that exist to support learning sciences research, CTAT+Tutorshop may be the only one that offers researchers the possibility to author elements of ITSs, or whole ITSs, as part of designing studies. This platform has been used to develop and conduct an estimated 147 research studies which have run in a wide variety of laboratory and real-world educational settings, including K-12 and higher education, and have addressed a wide range of research questions. This paper presents five case studies of research conducted on the CTAT+Tutorshop platform, and summarizes what has been accomplished and what is possible for future researchers. We reflect on the distinctive elements of this platform that have made it so effective in facilitating a wide range of ITS research. 

**Abstract (ZH)**: 智能辅导系统（ITSs）在帮助学生学习方面是有效的；进一步研究可以使它们更加有效。特别 desirable 的是关于学生如何使用这些系统进行学习、这些系统如何最好地支持学生学习以及哪些学习科学原则在ITSs中至关重要的研究。CTAT+Tutorshop 提供了一个完整的集成平台，可以促进从头到尾的ITS研究生命周期，包括使用ITS数据发现学习者挑战、识别系统改进机会以及进行实验研究。该平台包括支持和加速ITS开发的写作工具，这些工具提供自动数据日志记录，格式兼容 DataShop，这是一个独立站点，支持对教育技术日志数据进行分析以研究学生的学习。在众多支持学习科学研究的技术平台中，CTAT+Tutorshop 可能是唯一一个允许研究者在设计研究时编写ITS元素或整个ITS的平台。该平台已被用于开发和开展约 147 项研究，这些研究在各种实验室和实际教育环境中运行，涉及从小学至高中和高等教育，并解决了广泛的研究问题。本文介绍了五项在 CTAT+Tutorshop 平台上进行的研究案例，并总结了已经取得的成果以及未来研究人员可能实现的可能性。我们反思了这一平台的独特元素，这些元素使其能够有效促进广泛范围内的ITS研究。 

---
# A Glitch in the Matrix? Locating and Detecting Language Model Grounding with Fakepedia 

**Title (ZH)**: 矩阵中的bug？寻找与检测语言模型接地的Fakepedia 

**Authors**: Giovanni Monea, Maxime Peyrard, Martin Josifoski, Vishrav Chaudhary, Jason Eisner, Emre Kıcıman, Hamid Palangi, Barun Patra, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2312.02073)  

**Abstract**: Large language models (LLMs) have an impressive ability to draw on novel information supplied in their context. Yet the mechanisms underlying this contextual grounding remain unknown, especially in situations where contextual information contradicts factual knowledge stored in the parameters, which LLMs also excel at recalling. Favoring the contextual information is critical for retrieval-augmented generation methods, which enrich the context with up-to-date information, hoping that grounding can rectify outdated or noisy stored knowledge. We present a novel method to study grounding abilities using Fakepedia, a novel dataset of counterfactual texts constructed to clash with a model's internal parametric knowledge. In this study, we introduce Fakepedia, a counterfactual dataset designed to evaluate grounding abilities when the internal parametric knowledge clashes with the contextual information. We benchmark various LLMs with Fakepedia and conduct a causal mediation analysis of LLM components when answering Fakepedia queries, based on our Masked Grouped Causal Tracing (MGCT) method. Through this analysis, we identify distinct computational patterns between grounded and ungrounded responses. We finally demonstrate that distinguishing grounded from ungrounded responses is achievable through computational analysis alone. Our results, together with existing findings about factual recall mechanisms, provide a coherent narrative of how grounding and factual recall mechanisms interact within LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在利用上下文信息方面表现出色，但这些模型在上下文 Grounding 机制背后的原理仍不清楚，特别是在上下文信息与模型参数中存储的真实知识产生矛盾的情况下。在检索增强生成方法中，倾向于使用上下文信息对于补充最新信息至关重要，希望通过 Grounding 调整过时或嘈杂的存储知识。我们提出了一种使用 Fakepedia 的新方法来研究 Grounding 能力，Fakepedia 是一种与模型内部参数知识相矛盾的反事实数据集。在本研究中，我们介绍了 Fakepedia，这是一种用于评估内部参数知识与上下文信息相矛盾时 Grounding 能力的反事实数据集。我们使用 Fakepedia 对比基准各种 LLM，并基于我们提出的掩蔽分组因果追踪（MGCT）方法，对 LLM 组件在回答 Fakepedia 查询时进行因果中介分析。通过这种分析，我们识别出接地响应与非接地响应之间不同的计算模式。最后，我们证明仅通过计算分析即可区分接地响应与非接地响应。我们的结果与现有关于事实回忆机制的研究结果相结合，提供了一个关于 Grounding 机制和事实回忆机制在 LLM 中相互作用的连贯叙事。 

---
