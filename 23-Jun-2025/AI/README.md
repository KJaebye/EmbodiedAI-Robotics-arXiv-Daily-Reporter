# The MedPerturb Dataset: What Non-Content Perturbations Reveal About Human and Clinical LLM Decision Making 

**Title (ZH)**: MedPerturb数据集：非内容扰动揭示的人类和临床LLM决策机制 

**Authors**: Abinitha Gourabathina, Yuexing Hao, Walter Gerych, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17163)  

**Abstract**: Clinical robustness is critical to the safe deployment of medical Large Language Models (LLMs), but key questions remain about how LLMs and humans may differ in response to the real-world variability typified by clinical settings. To address this, we introduce MedPerturb, a dataset designed to systematically evaluate medical LLMs under controlled perturbations of clinical input. MedPerturb consists of clinical vignettes spanning a range of pathologies, each transformed along three axes: (1) gender modifications (e.g., gender-swapping or gender-removal); (2) style variation (e.g., uncertain phrasing or colloquial tone); and (3) format changes (e.g., LLM-generated multi-turn conversations or summaries). With MedPerturb, we release a dataset of 800 clinical contexts grounded in realistic input variability, outputs from four LLMs, and three human expert reads per clinical context. We use MedPerturb in two case studies to reveal how shifts in gender identity cues, language style, or format reflect diverging treatment selections between humans and LLMs. We find that LLMs are more sensitive to gender and style perturbations while human annotators are more sensitive to LLM-generated format perturbations such as clinical summaries. Our results highlight the need for evaluation frameworks that go beyond static benchmarks to assess the similarity between human clinician and LLM decisions under the variability characteristic of clinical settings. 

**Abstract (ZH)**: 医学稳健性是将医疗大型语言模型（LLMs）安全部署的关键，但在面对典型临床环境变量时，LLMs与人类的响应差异方面仍存在关键问题。为解决这一问题，我们引入了MedPerturb数据集，旨在通过控制临床输入的变形系统性评估医疗LLMs。MedPerturb包含跨越多种病理的临床案例概要，每种概要沿三个轴线进行变形：（1）性别修改（例如，性别互换或性别移除）；（2）风格变化（例如，不确定用词或口语化语气）；以及（3）格式变化（例如，LLM生成的多轮对话或总结）。通过MedPerturb，我们发布了包含800个基于现实输入变异的临床上下文的数据集，每个临床上下文有四种LLM的输出和三种人类专家的阅读。我们使用MedPerturb在两个案例研究中揭示性别身份提示、语言风格或格式的变化如何反映人类和LLMs之间治疗选择的差异。结果表明，LLMs对性别和风格变形更为敏感，而人类注释者对LLM生成的格式变形，如临床总结更为敏感。我们的研究结果强调了评估框架的需求，该框架应超越静止基准，以临床环境中的变量特征评估人类临床医生和LLMs决策的相似性。 

---
# Chain-of-Trust: A Progressive Trust Evaluation Framework Enabled by Generative AI 

**Title (ZH)**: 可信链：由生成式AI赋能的逐步信任评估框架 

**Authors**: Botao Zhu, Xianbin Wang, Lei Zhang, Xuemin, Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.17130)  

**Abstract**: In collaborative systems with complex tasks relying on distributed resources, trust evaluation of potential collaborators has emerged as an effective mechanism for task completion. However, due to the network dynamics and varying information gathering latencies, it is extremely challenging to observe and collect all trust attributes of a collaborating device concurrently for a comprehensive trust assessment. In this paper, a novel progressive trust evaluation framework, namely chain-of-trust, is proposed to make better use of misaligned device attribute data. This framework, designed for effective task completion, divides the trust evaluation process into multiple chained stages based on task decomposition. At each stage, based on the task completion process, the framework only gathers the latest device attribute data relevant to that stage, leading to reduced trust evaluation complexity and overhead. By leveraging advanced in-context learning, few-shot learning, and reasoning capabilities, generative AI is then employed to analyze and interpret the collected data to produce correct evaluation results quickly. Only devices deemed trustworthy at this stage proceed to the next round of trust evaluation. The framework ultimately determines devices that remain trustworthy across all stages. Experimental results demonstrate that the proposed framework achieves high accuracy in trust evaluation. 

**Abstract (ZH)**: 在依赖分布式资源的复杂任务协作系统中，潜在协作方的信任评估已成为有效完成任务的有效机制。然而，由于网络动态性和信息收集延迟的变化，很难同时观察和收集协作设备的所有信任属性以进行全面的信任评估。本文提出了一种新颖的逐级信任评估框架——链式信任框架，以更好地利用不一致的设备属性数据。该框架针对有效完成任务而设计，基于任务分解将信任评估过程划分为多个链式阶段。在每个阶段，基于任务完成过程，该框架仅收集当前阶段相关的最新设备属性数据，从而降低信任评估的复杂性和开销。通过利用高级上下文学习、少样本学习和推理能力，然后使用生成式AI来分析和解释收集的数据，迅速生成正确的评估结果。仅在当前阶段被认为可信的设备才进入下一阶段的信任评估。该框架最终确定在整个阶段都保持可信的设备。实验结果表明，所提出的框架在信任评估中实现了高准确性。 

---
# When Can Model-Free Reinforcement Learning be Enough for Thinking? 

**Title (ZH)**: 无模型强化学习何时足以进行思考？ 

**Authors**: Josiah P. Hanna, Nicholas E. Corrado  

**Link**: [PDF](https://arxiv.org/pdf/2506.17124)  

**Abstract**: Recent work on large language models has demonstrated the use of model-free reinforcement learning (RL) to train reasoning-like capabilities. The emergence of "thinking" through model-free RL is interesting as thinking actions neither produce reward nor change the external world state to one where the agent is more likely to get reward. This paper seeks to build a domain-independent understanding of when model-free RL will lead to "thinking" as a strategy for reward maximization. To build this understanding, we first introduce a theoretical model which we call a \textit{thought Markov decision process} (MDP). Thought MDPs minimally extend the classical MDP model to include an abstract notion of thought state and thought action. Using the thought MDP model, we prove the importance of policy initialization in determining whether or not thinking emerges and show formally that thought actions are equivalent to the agent choosing to perform a step of policy improvement before continuing to act. We then show that open-source LLMs satisfy the conditions that our theory predicts are necessary for model-free RL to produce thinking-like behavior. Finally, we hypothesize sufficient conditions that would enable thinking to be learned outside of language generation and introduce a toy domain where a combination of multi-task pre-training and designated thought actions enable more data-efficient RL compared to non-thinking agents. 

**Abstract (ZH)**: recent 工作表明，通过无模型强化学习（RL）训练具有类推理能力是可行的。无模型 RL 中的“思考”行为令人感兴趣，因为思考动作既不产生奖励，也不改变外部世界状态使得代理更有可能获得奖励。本文旨在构建一种独立于领域的理解和判断，在哪种情况下无模型 RL 将导致“思考”作为一种奖励最大化策略。为了构建这种理解，我们首先引入了一个称为“思考马尔可夫决策过程” (MDP) 的理论模型。思考 MDP 仅最小地扩展了经典的 MDP 模型，以包括抽象的概念——思考状态和思考动作。利用思考 MDP 模型，我们证明了策略初始化在决定是否会出现思考方面的重要性，并且正式证明了思考动作等价于代理选择执行一次策略改进步骤后再继续行动。随后，我们展示了开源语言模型满足理论预测的必要条件，从而使无模型 RL 产生类似思考的行为。最后，我们提出了允许思考在语言生成之外被学到的充分条件，并引入了一个玩具领域，在该领域中，多任务预训练与特定设计的思考动作的结合使相对于非思考代理更高效的学习成为可能。 

---
# Mathematical Proof as a Litmus Test: Revealing Failure Modes of Advanced Large Reasoning Models 

**Title (ZH)**: 数学证明作为试金石：揭示高级大型推理模型的失效模式 

**Authors**: Dadi Guo, Jiayu Liu, Zhiyuan Fan, Zhitao He, Haoran Li, Yumeng Wang, Yi R., Fung  

**Link**: [PDF](https://arxiv.org/pdf/2506.17114)  

**Abstract**: Large reasoning models (e.g., R1, o3) have demonstrated remarkable mathematical problem-solving abilities. However, the high reported accuracy of these advanced models on popular datasets, reliance on purely numerical evaluation and potential benchmark leakage, often masks their true reasoning shortcomings. To address this, we propose leveraging the inherent rigor and methodological complexity of mathematical proofs as a diagnostic tool to expose these hidden failures. Specifically, we introduce the RFMDataset (Reveal Failure Modes), a collection of 200 diverse mathematical proof problems, and thoroughly evaluate advanced models' performance on it. Our in-depth analysis of their failures uncovers 10 fine-grained error types, which shows fundamental limitations in current large reasoning models: 1) large reasoning models grapple profoundly with mathematical proofs, with some generating entirely correct proofs for less than 20% of problems and failing even on basic ones; 2) models exhibit a diverse spectrum of reasoning failures, prominently demonstrating the lack of guarantees for the correctness and rigor of single-step reasoning; and 3) models show hallucination and incompleteness during the reasoning process. Our findings reveal that models' self-reflection is insufficient to resolve the current logical dilemmas, necessitating formalized and fine-grained logical training. 

**Abstract (ZH)**: 大型推理模型（如R1、o3）在数学问题求解能力上表现出色，但这些先进模型在流行数据集上的高报道准确度、纯粹的数值评估依赖以及潜在的基准泄露问题，往往掩盖了它们真实的推理缺陷。为解决这一问题，我们提出利用数学证明固有的严谨性和方法学复杂性作为诊断工具，以揭示隐藏的失败模式。具体而言，我们引入了RFMDataset（Reveal Failure Modes），包含200个多样化的数学证明问题，并全面评估先进模型在该数据集上的表现。通过对它们失败的深入分析，我们发现了10种细微错误类型，揭示了当前大型推理模型的基本限制：1）大型推理模型在数学证明问题上面临深刻挑战，部分模型在不到20%的问题上生成完全正确的证明，并且在简单问题上也屡屡失败；2）模型在推理过程中的失败表现多样，明显缺乏单步推理正确性和严谨性的保证；3）模型在推理过程中表现出幻觉和不完整性。我们的研究结果表明，模型的自我反思不足以解决当前的逻辑困境，需要进行形式化和细致的逻辑训练。 

---
# Are Bias Evaluation Methods Biased ? 

**Title (ZH)**: 偏差评估方法本身存在偏差吗？ 

**Authors**: Lina Berrayana, Sean Rooney, Luis Garcés-Erice, Ioana Giurgiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17111)  

**Abstract**: The creation of benchmarks to evaluate the safety of Large Language Models is one of the key activities within the trusted AI community. These benchmarks allow models to be compared for different aspects of safety such as toxicity, bias, harmful behavior etc. Independent benchmarks adopt different approaches with distinct data sets and evaluation methods. We investigate how robust such benchmarks are by using different approaches to rank a set of representative models for bias and compare how similar are the overall rankings. We show that different but widely used bias evaluations methods result in disparate model rankings. We conclude with recommendations for the community in the usage of such benchmarks. 

**Abstract (ZH)**: 基于可信AI社区的评估大语言模型安全性的基准创建是关键活动之一。这些基准使得模型可以从毒性和偏见等不同方面进行比较。独立基准采用不同的方法并使用不同的数据集和评估方法。我们通过使用不同的方法对一组代表性模型进行排名，并比较这些模型的整体排名相似性，以考察这些基准的稳健性。我们展示了不同但广泛使用的偏见评估方法会导致不同的模型排名。最后，我们为社区在使用此类基准方面提供建议。 

---
# Towards Advanced Mathematical Reasoning for LLMs via First-Order Logic Theorem Proving 

**Title (ZH)**: 基于一阶逻辑定理证明的高级数学推理能力提升方法 

**Authors**: Chuxue Cao, Mengze Li, Juntao Dai, Jinluan Yang, Zijian Zhao, Shengyu Zhang, Weijie Shi, Chengzhong Liu, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17104)  

**Abstract**: Large language models (LLMs) have shown promising first-order logic (FOL) reasoning capabilities with applications in various areas. However, their effectiveness in complex mathematical reasoning involving multi-step FOL deductions is still under-researched. While LLMs perform competitively on established mathematical reasoning benchmarks, they struggle with multi-step FOL tasks, as demonstrated by Deepseek-Prover-V2-7B's low accuracy (4.2%) on our proposed theorem proving dataset. This issue arises from the limited exploration of diverse proof strategies and the potential for early reasoning mistakes to undermine entire proofs. To address these issues, we propose DREAM, a self-adaptive solution that enhances the Diversity and REAsonability of LLMs' generation strategies. DREAM incorporates an Axiom-Driven Strategy Diversification mechanism to promote varied strategic outcomes and a Sub-Proposition Error Feedback to help LLMs reflect on and correct their proofs. Our contributions include pioneering advancements in LLMs' mathematical reasoning through FOL theorem proving, introducing a novel inference stage solution that improves performance by 0.6% to 6.4%, and providing a curated dataset of 447 mathematical theorems in Lean 4 format for evaluation. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各类应用中展示了有前途的一阶逻辑（FOL）推理能力。然而，它们在涉及多步FOL演绎的复杂数学推理方面的有效性仍需进一步研究。尽管LLMs在现有的数学推理基准测试中表现得当，但在处理多步FOL任务时依然面临挑战，如我们提出的定理证明数据集上Deepseek-Prover-V2-7B的低准确性（4.2%）。这一问题源于对多样化证明策略的探索有限以及早期推理错误可能削弱整个证明。为解决这些问题，我们提出DREAM，一种自我适应的解决方案，旨在增强LLMs生成策略的多样性和合理性。DREAM结合了公理驱动的战略多样化机制以促进多样化的战略结果，并采用了子命题错误反馈机制以帮助LLMs反思和修正其证明。我们的贡献包括在通过一阶逻辑定理证明增强LLMs的数学推理方面的开创性进展，引入了一种新型的推理阶段解决方案，通过提高0.6%到6.4%的性能，以及提供了447个数学定理的精心制作数据集，用于在Lean 4格式下的评估。 

---
# Dispositions and Roles of Generically Dependent Entities 

**Title (ZH)**: 泛依赖实体的性质与角色 

**Authors**: Fabian Neuhaus  

**Link**: [PDF](https://arxiv.org/pdf/2506.17085)  

**Abstract**: BFO 2020 does not support functions, dispositions, and roles of generically dependent continuants (like software or datasets). In this paper, we argue that this is a severe limitation, which prevents, for example, the adequate representation of the functions of computer models or the various roles of datasets during the execution of these models. We discuss the aspects of BFO 2020 that prevent the representation of realizable entities of generically dependent continuants. Two approaches to address the issue are presented: (a) the use of defined classes and (b) a proposal of changes that allow BFO to support functions, dispositions, and roles of generically dependent continuants. 

**Abstract (ZH)**: BFO 2020 不支持通用依赖延续体（如软件或数据集）的功能、性质和角色。本文认为这是一项严重限制，例如无法充分代表计算机模型的功能或模型运行期间数据集的各种角色。我们讨论了 BFO 2020 阻碍通用依赖延续体实现实体表示的方面。提出了两种解决方案：（a）使用定义类，（b）提出更改以使 BFO 支持通用依赖延续体的功能、性质和角色。 

---
# A Quantile Regression Approach for Remaining Useful Life Estimation with State Space Models 

**Title (ZH)**: 基于状态空间模型的分位数回归剩余使用寿命估算方法 

**Authors**: Davide Frizzo, Francesco Borsatti, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2506.17018)  

**Abstract**: Predictive Maintenance (PdM) is pivotal in Industry 4.0 and 5.0, proactively enhancing efficiency through accurate equipment Remaining Useful Life (RUL) prediction, thus optimizing maintenance scheduling and reducing unexpected failures and premature interventions. This paper introduces a novel RUL estimation approach leveraging State Space Models (SSM) for efficient long-term sequence modeling. To handle model uncertainty, Simoultaneous Quantile Regression (SQR) is integrated into the SSM, enabling multiple quantile estimations. The proposed method is benchmarked against traditional sequence modelling techniques (LSTM, Transformer, Informer) using the C-MAPSS dataset. Results demonstrate superior accuracy and computational efficiency of SSM models, underscoring their potential for high-stakes industrial applications. 

**Abstract (ZH)**: Predictive 维护在工业4.0和5.0中至关重要，通过准确的设备剩余使用寿命（RUL）预测主动提升效率，从而优化维护调度并减少意外故障和过早干预。本文引入了一种利用状态空间模型（SSM）进行高效长序列建模的新RUL估计方法。为了处理模型不确定性，将同时分位数回归（SQR）集成到SSM中，使其能够进行多分位数估计。所提出的方法使用C-MAPSS数据集与传统的序列建模技术（LSTM、Transformer、Informer）进行基准测试。结果展示了SSM模型在准确性和计算效率方面的优越性，突显了其在高风险工业应用中的潜在价值。 

---
# Elevating Styled Mahjong Agents with Learning from Demonstration 

**Title (ZH)**: 基于示范学习提升风格化麻将代理 

**Authors**: Lingfeng Li, Yunlong Lu, Yongyi Wang, Wenxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16995)  

**Abstract**: A wide variety of bots in games enriches the gameplay experience and enhances replayability. Recent advancements in game artificial intelligence have predominantly focused on improving the proficiency of bots. Nevertheless, developing highly competent bots with a wide range of distinct play styles remains a relatively under-explored area. We select the Mahjong game environment as a case study. The high degree of randomness inherent in the Mahjong game and the prevalence of out-of-distribution states lead to suboptimal performance of existing offline learning and Learning-from-Demonstration (LfD) algorithms. In this paper, we leverage the gameplay histories of existing Mahjong agents and put forward a novel LfD algorithm that necessitates only minimal modifications to the Proximal Policy Optimization algorithm. The comprehensive empirical results illustrate that our proposed method not only significantly enhances the proficiency of the agents but also effectively preserves their unique play styles. 

**Abstract (ZH)**: 游戏中的各种 bots 丰富了游戏体验并增强了重玩价值。近年来，游戏人工智能的发展主要集中在提高 bots 的熟练程度上。然而，开发具有广泛不同游戏风格的高水平 bots 仍然是一个相对未被充分探索的领域。我们以麻将游戏环境为例。麻将游戏固有的高随机性和离分布状态的普遍存在导致现有离线学习和学习从演示（LfD）算法的性能不佳。在本文中，我们利用现有麻将代理的游戏历史记录，提出了一种仅需对策略优化（PPO）算法进行少量修改的新LfD算法。综合的实证结果表明，我们提出的方法不仅显著提高了代理的熟练程度，还有效地保留了它们的独特游戏风格。 

---
# Multimodal Fused Learning for Solving the Generalized Traveling Salesman Problem in Robotic Task Planning 

**Title (ZH)**: 多模态融合学习在机器人任务规划中解决广义旅行商问题 

**Authors**: Jiaqi Chen, Mingfeng Fan, Xuefeng Zhang, Jingsong Liang, Yuhong Cao, Guohua Wu, Guillaume Adrien Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2506.16931)  

**Abstract**: Effective and efficient task planning is essential for mobile robots, especially in applications like warehouse retrieval and environmental monitoring. These tasks often involve selecting one location from each of several target clusters, forming a Generalized Traveling Salesman Problem (GTSP) that remains challenging to solve both accurately and efficiently. To address this, we propose a Multimodal Fused Learning (MMFL) framework that leverages both graph and image-based representations to capture complementary aspects of the problem, and learns a policy capable of generating high-quality task planning schemes in real time. Specifically, we first introduce a coordinate-based image builder that transforms GTSP instances into spatially informative representations. We then design an adaptive resolution scaling strategy to enhance adaptability across different problem scales, and develop a multimodal fusion module with dedicated bottlenecks that enables effective integration of geometric and spatial features. Extensive experiments show that our MMFL approach significantly outperforms state-of-the-art methods across various GTSP instances while maintaining the computational efficiency required for real-time robotic applications. Physical robot tests further validate its practical effectiveness in real-world scenarios. 

**Abstract (ZH)**: 有效的多模态融合学习框架对于移动机器人任务规划至关重要，特别是在仓库检索和环境监控等应用中。我们提出了一种多模态融合学习（MMFL）框架，结合图和图像表示，以捕获问题的互补方面，并学习一种能够在实时生成高质量任务规划方案的策略。具体来说，我们首先引入了一种基于坐标的应用图像构建器，将GTSP实例转换为具有空间信息的表示。然后设计了一种自适应分辨率缩放策略，以增强不同问题规模的适应性，并开发了一种多模态融合模块，具有专门为几何和空间特征设计的瓶颈，以实现有效的融合。广泛实验表明，我们的MMFL方法在各种GTSP实例中均显著优于现有方法，同时保持了适用于实时机器人应用所需的计算效率。进一步的物理机器人测试验证了其在真实世界场景中的实际有效性。 

---
# Real-Time Black-Box Optimization for Dynamic Discrete Environments Using Embedded Ising Machines 

**Title (ZH)**: 使用嵌入式伊辛机的实时黑盒优化方法及其在动态离散环境中的应用 

**Authors**: Tomoya Kashimata, Yohei Hamakawa, Masaya Yamasaki, Kosuke Tatsumura  

**Link**: [PDF](https://arxiv.org/pdf/2506.16924)  

**Abstract**: Many real-time systems require the optimization of discrete variables. Black-box optimization (BBO) algorithms and multi-armed bandit (MAB) algorithms perform optimization by repeatedly taking actions and observing the corresponding instant rewards without any prior knowledge. Recently, a BBO method using an Ising machine has been proposed to find the best action that is represented by a combination of discrete values and maximizes the instant reward in static environments. In contrast, dynamic environments, where real-time systems operate, necessitate MAB algorithms that maximize the average reward over multiple trials. However, due to the enormous number of actions resulting from the combinatorial nature of discrete optimization, conventional MAB algorithms cannot effectively optimize dynamic, discrete environments. Here, we show a heuristic MAB method for dynamic, discrete environments by extending the BBO method, in which an Ising machine effectively explores the actions while considering interactions between variables and changes in dynamic environments. We demonstrate the dynamic adaptability of the proposed method in a wireless communication system with moving users. 

**Abstract (ZH)**: 使用伊辛机的黑盒优化方法在动态离散环境中优化即时奖励：一种扩展多臂bandit算法的方法 

---
# AI's Blind Spots: Geographic Knowledge and Diversity Deficit in Generated Urban Scenario 

**Title (ZH)**: AI的盲点：生成的城市场景中的地理知识匮乏与多元化缺失 

**Authors**: Ciro Beneduce, Massimiliano Luca, Bruno Lepri  

**Link**: [PDF](https://arxiv.org/pdf/2506.16898)  

**Abstract**: Image generation models are revolutionizing many domains, and urban analysis and design is no exception. While such models are widely adopted, there is a limited literature exploring their geographic knowledge, along with the biases they embed. In this work, we generated 150 synthetic images for each state in the USA and related capitals using FLUX 1 and Stable Diffusion 3.5, two state-of-the-art models for image generation. We embed each image using DINO-v2 ViT-S/14 and the Fréchet Inception Distances to measure the similarity between the generated images. We found that while these models have implicitly learned aspects of USA geography, if we prompt the models to generate an image for "United States" instead of specific cities or states, the models exhibit a strong representative bias toward metropolis-like areas, excluding rural states and smaller cities. {\color{black} In addition, we found that models systematically exhibit some entity-disambiguation issues with European-sounding names like Frankfort or Devon. 

**Abstract (ZH)**: 图像生成模型正在革新许多领域，城市分析与设计也不例外。尽管这些模型被广泛采用，但有关它们嵌入的地理知识及其偏见的研究仍然有限。在本工作中，我们使用FLUX 1和Stable Diffusion 3.5两种最先进的图像生成模型，分别为美国各州和首府生成了150张合成图像。我们使用DINO-v2 ViT-S/14和弗雷切尔-Incendtime 距离来衡量生成图像之间的相似度。研究发现，虽然这些模型在一定程度上隐式学习了美国的地理特征，但若将提示语从具体的城镇或州改为“美国”，模型倾向于生成类似大都市区的图像，而忽视了农村州和较小的城镇。此外，我们还发现模型在处理具有欧洲风格名称的实体时存在一定程度的语义歧义问题，例如Frankfort或Devon。 

---
# Reinforcement learning for hybrid charging stations planning and operation considering fixed and mobile chargers 

**Title (ZH)**: 基于固定和移动充电器考虑的混合充电站规划与运行的强化学习方法 

**Authors**: Yanchen Zhu, Honghui Zou, Chufan Liu, Yuyu Luo, Yuankai Wu, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16764)  

**Abstract**: The success of vehicle electrification, which brings significant societal and environmental benefits, is contingent upon the availability of efficient and adaptable charging infrastructure. Traditional fixed-location charging stations often face issues like underutilization or congestion due to the dynamic nature of charging demand. Mobile chargers have emerged as a flexible solution, capable of relocating to align with these demand fluctuations. This paper addresses the optimal planning and operation of hybrid charging infrastructures, integrating both fixed and mobile chargers within urban road networks. We introduce the Hybrid Charging Station Planning and Operation (HCSPO) problem, which simultaneously optimizes the location and configuration of fixed charging stations and schedules mobile chargers for dynamic operations. Our approach incorporates a charging demand prediction model grounded in Model Predictive Control (MPC) to enhance decision-making. To solve the HCSPO problem, we propose a deep reinforcement learning method, augmented with heuristic scheduling techniques, to effectively bridge the planning of fixed chargers with the real-time operation of mobile chargers. Extensive case studies using real-world urban scenarios demonstrate that our method significantly improves the availability of charging infrastructure and reduces user inconvenience compared to existing solutions and baselines. 

**Abstract (ZH)**: 车辆电气化成功的实现，带来了重要的社会和环境效益，取决于高效且灵活的充电基础设施的可用性。传统的固定位置充电站常因充电需求的动态变化而面临利用率低或拥堵的问题。移动充电器作为一项灵活的解决方案，能够根据需求变化重新部署。本文探讨了混合充电基础设施的最优规划与运营问题，将固定和移动充电站整合到城市道路网络中。我们提出了混合充电站规划与运营（HCSPO）问题，该问题同时优化了固定充电站的位置和配置，并为移动充电器安排动态操作。我们的方法结合了基于模型预测控制（MPC）的充电需求预测模型，以增强决策制定能力。为了解决HCSPO问题，我们提出了一种增强学习方法，并结合启发式调度技术，有效地将固定充电器的规划与移动充电器的实时操作相结合。通过对现实世界城市场景的广泛案例研究，我们的方法显著提高了充电基础设施的可用性，并减少了用户的不便，相较于现有解决方案和基线方法。 

---
# Incentivizing High-quality Participation From Federated Learning Agents 

**Title (ZH)**: 激励联邦学习代理参与高质量合作 

**Authors**: Jinlong Pang, Jiaheng Wei, Yifan Hua, Chen Qian, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16731)  

**Abstract**: Federated learning (FL) provides a promising paradigm for facilitating collaboration between multiple clients that jointly learn a global model without directly sharing their local data. However, existing research suffers from two caveats: 1) From the perspective of agents, voluntary and unselfish participation is often assumed. But self-interested agents may opt out of the system or provide low-quality contributions without proper incentives; 2) From the mechanism designer's perspective, the aggregated models can be unsatisfactory as the existing game-theoretical federated learning approach for data collection ignores the potential heterogeneous effort caused by contributed data. To alleviate above challenges, we propose an incentive-aware framework for agent participation that considers data heterogeneity to accelerate the convergence process. Specifically, we first introduce the notion of Wasserstein distance to explicitly illustrate the heterogeneous effort and reformulate the existing upper bound of convergence. To induce truthful reporting from agents, we analyze and measure the generalization error gap of any two agents by leveraging the peer prediction mechanism to develop score functions. We further present a two-stage Stackelberg game model that formalizes the process and examines the existence of equilibrium. Extensive experiments on real-world datasets demonstrate the effectiveness of our proposed mechanism. 

**Abstract (ZH)**: federated学习（FL）提供了一种促进多个客户端协作学习全局模型而无需直接共享本地数据的有前途的范式。然而，现有研究存在两个问题：1) 从代理的角度来看，自愿和无私的参与通常被视为理所当然。但自私的代理可能会退出系统或在没有适当激励的情况下提供低质量的贡献；2) 从机制设计者角度来看，现有的基于博弈论的数据收集联邦学习方法忽略了贡献数据导致的潜在异质性努力，从而使聚合模型不尽如人意。为了缓解上述挑战，我们提出了一种关注激励的代理参与框架，该框架考虑数据异质性以加速收敛过程。具体而言，我们首先引入Wasserstein距离来明确表示异质性努力并重新表述现有收敛的上界。为诱导代理进行真实报告，我们利用同伴预测机制分析和度量任意两个代理的泛化误差差异，以开发评分函数。我们进一步提出了一种两阶段斯塔克尔贝格博弈模型，该模型形式化了过程并检查了均衡的存在性。在真实世界数据集上的 extensive 实验表明我们提出机制的有效性。 

---
# Interpretable Low-Dimensional Modeling of Spatiotemporal Agent States for Decision Making in Football Tactics 

**Title (ZH)**: 足球战术中基于时空代理状态的可解释低维建模决策方法 

**Authors**: Kenjiro Ide, Taiga Someya, Kohei Kawaguchi, Keisuke Fujii  

**Link**: [PDF](https://arxiv.org/pdf/2506.16696)  

**Abstract**: Understanding football tactics is crucial for managers and analysts. Previous research has proposed models based on spatial and kinematic equations, but these are computationally expensive. Also, Reinforcement learning approaches use player positions and velocities but lack interpretability and require large datasets. Rule-based models align with expert knowledge but have not fully considered all players' states. This study explores whether low-dimensional, rule-based models using spatiotemporal data can effectively capture football tactics. Our approach defines interpretable state variables for both the ball-holder and potential pass receivers, based on criteria that explore options like passing. Through discussions with a manager, we identified key variables representing the game state. We then used StatsBomb event data and SkillCorner tracking data from the 2023$/$24 LaLiga season to train an XGBoost model to predict pass success. The analysis revealed that the distance between the player and the ball, as well as the player's space score, were key factors in determining successful passes. Our interpretable low-dimensional modeling facilitates tactical analysis through the use of intuitive variables and provides practical value as a tool to support decision-making in football. 

**Abstract (ZH)**: 理解足球战术对于教练和分析师至关重要。尽管以往研究提出了基于空间和运动方程的模型，但这些模型计算成本较高。强化学习方法通过球员位置和速度进行操作，但缺乏可解释性且需要大量数据集。基于规则的模型符合专家知识，但未全面考虑所有球员的状态。本研究探讨低维、基于规则的方法是否可以通过时空数据有效地捕捉足球战术。我们的方法基于传递选项等标准定义了可解释的状态变量，不仅包括持球者还涉及潜在传球接收者。通过与一名教练的讨论，我们确定了代表比赛状态的关键变量。我们使用2023-24赛季西甲联赛的StatsBomb事件数据和SkillCorner跟踪数据训练了一个XGBoost模型来预测传球成功率。分析结果显示，球员与球之间的距离以及球员的空间得分是决定成功传球的关键因素。我们的可解释低维建模通过使用直观变量促进战术分析，并作为一种支持足球决策的实用工具提供了实际价值。 

---
# The Role of Explanation Styles and Perceived Accuracy on Decision Making in Predictive Process Monitoring 

**Title (ZH)**: 解释风格和感知准确性在预测过程监控中决策制定的作用 

**Authors**: Soobin Chae, Suhwan Lee, Hanna Hauptmann, Hajo A. Reijers, Xixi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16617)  

**Abstract**: Predictive Process Monitoring (PPM) often uses deep learning models to predict the future behavior of ongoing processes, such as predicting process outcomes. While these models achieve high accuracy, their lack of interpretability undermines user trust and adoption. Explainable AI (XAI) aims to address this challenge by providing the reasoning behind the predictions. However, current evaluations of XAI in PPM focus primarily on functional metrics (such as fidelity), overlooking user-centered aspects such as their effect on task performance and decision-making. This study investigates the effects of explanation styles (feature importance, rule-based, and counterfactual) and perceived AI accuracy (low or high) on decision-making in PPM. We conducted a decision-making experiment, where users were presented with the AI predictions, perceived accuracy levels, and explanations of different styles. Users' decisions were measured both before and after receiving explanations, allowing the assessment of objective metrics (Task Performance and Agreement) and subjective metrics (Decision Confidence). Our findings show that perceived accuracy and explanation style have a significant effect. 

**Abstract (ZH)**: 解释性人工智能在过程监控中的解释样式和感知AI准确度对决策的影响研究 

---
# A Community-driven vision for a new Knowledge Resource for AI 

**Title (ZH)**: 社区驱动的关于新AI知识资源的愿景 

**Authors**: Vinay K Chaudhri, Chaitan Baru, Brandon Bennett, Mehul Bhatt, Darion Cassel, Anthony G Cohn, Rina Dechter, Esra Erdem, Dave Ferrucci, Ken Forbus, Gregory Gelfond, Michael Genesereth, Andrew S. Gordon, Benjamin Grosof, Gopal Gupta, Jim Hendler, Sharat Israni, Tyler R. Josephson, Patrick Kyllonen, Yuliya Lierler, Vladimir Lifschitz, Clifton McFate, Hande K. McGinty, Leora Morgenstern, Alessandro Oltramari, Praveen Paritosh, Dan Roth, Blake Shepard, Cogan Shimzu, Denny Vrandečić, Mark Whiting, Michael Witbrock  

**Link**: [PDF](https://arxiv.org/pdf/2506.16596)  

**Abstract**: The long-standing goal of creating a comprehensive, multi-purpose knowledge resource, reminiscent of the 1984 Cyc project, still persists in AI. Despite the success of knowledge resources like WordNet, ConceptNet, Wolfram|Alpha and other commercial knowledge graphs, verifiable, general-purpose widely available sources of knowledge remain a critical deficiency in AI infrastructure. Large language models struggle due to knowledge gaps; robotic planning lacks necessary world knowledge; and the detection of factually false information relies heavily on human expertise. What kind of knowledge resource is most needed in AI today? How can modern technology shape its development and evaluation? A recent AAAI workshop gathered over 50 researchers to explore these questions. This paper synthesizes our findings and outlines a community-driven vision for a new knowledge infrastructure. In addition to leveraging contemporary advances in knowledge representation and reasoning, one promising idea is to build an open engineering framework to exploit knowledge modules effectively within the context of practical applications. Such a framework should include sets of conventions and social structures that are adopted by contributors. 

**Abstract (ZH)**: 持久的目标：构建一个综合性的多用途知识资源——类似于1984年的Cyc项目——在AI领域仍然存在。尽管有WordNet、ConceptNet、Wolfram|Alpha及其他商业知识图谱的成功，可验证的、通用的广泛应用的知识来源仍然是AI基础设施中的关键不足。大型语言模型由于知识空白而受到影响；机器人规划缺乏必要的世界知识；而事实错误信息的检测则高度依赖人类专家。当前最需要什么样的知识资源？现代技术如何塑造其发展和评估？最近的AAAI研讨会汇集了超过50名研究人员探讨这些问题。本文综合了我们的研究成果，并阐述了一个由社区驱动的新知识基础设施愿景。除了利用知识表示和推理的当代进展，一个有希望的想法是构建一个开放的工程框架，以有效地在实际应用中利用知识模块。这样的框架应该包括贡献者采纳的一套约定和社交结构。 

---
# Advancing Harmful Content Detection in Organizational Research: Integrating Large Language Models with Elo Rating System 

**Title (ZH)**: 增强组织研究中的有害内容检测：结合大型语言模型与Elo评分系统 

**Authors**: Mustafa Akben, Aaron Satko  

**Link**: [PDF](https://arxiv.org/pdf/2506.16575)  

**Abstract**: Large language models (LLMs) offer promising opportunities for organizational research. However, their built-in moderation systems can create problems when researchers try to analyze harmful content, often refusing to follow certain instructions or producing overly cautious responses that undermine validity of the results. This is particularly problematic when analyzing organizational conflicts such as microaggressions or hate speech. This paper introduces an Elo rating-based method that significantly improves LLM performance for harmful content analysis In two datasets, one focused on microaggression detection and the other on hate speech, we find that our method outperforms traditional LLM prompting techniques and conventional machine learning models on key measures such as accuracy, precision, and F1 scores. Advantages include better reliability when analyzing harmful content, fewer false positives, and greater scalability for large-scale datasets. This approach supports organizational applications, including detecting workplace harassment, assessing toxic communication, and fostering safer and more inclusive work environments. 

**Abstract (ZH)**: 大型语言模型（LLMs）为组织研究提供了广阔的机会。然而，它们内置的调控系统在研究人员尝试分析有害内容时可能会引发问题，经常拒不服从某些指令或产生过于谨慎的回应，从而影响结果的有效性。这在分析组织冲突如微欺凌或仇恨言论时尤为成问题。本文介绍了一种基于Elo评分的方法，显著提升了LLM在有害内容分析中的性能。在两个数据集中，一个专注于微欺凌检测，另一个专注于仇恨言论，我们发现本方法在准确率、精确率和F1分数等关键指标上优于传统的LLM提示技术和传统机器学习模型。优点包括在分析有害内容时更好的可靠性、较少的假阳性以及更大规模数据集的可扩展性。该方法支持组织应用，包括检测职场骚扰、评估有毒沟通以及促进更安全包容的工作环境。 

---
# ML-Master: Towards AI-for-AI via Integration of Exploration and Reasoning 

**Title (ZH)**: ML-Master: 通过探索与推理的集成迈向AI为AI 

**Authors**: Zexi Liu, Yuzhu Cai, Xinyu Zhu, Yujie Zheng, Runkun Chen, Ying Wen, Yanfeng Wang, Weinan E, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.16499)  

**Abstract**: As AI capabilities advance toward and potentially beyond human-level performance, a natural transition emerges where AI-driven development becomes more efficient than human-centric approaches. A promising pathway toward this transition lies in AI-for-AI (AI4AI), which leverages AI techniques to automate and optimize the design, training, and deployment of AI systems themselves. While LLM-based agents have shown the potential to realize AI4AI, they are often unable to fully leverage the experience accumulated by agents during the exploration of solutions in the reasoning process, leading to inefficiencies and suboptimal performance. To address this limitation, we propose ML-Master, a novel AI4AI agent that seamlessly integrates exploration and reasoning by employing a selectively scoped memory mechanism. This approach allows ML-Master to efficiently combine diverse insights from parallel solution trajectories with analytical reasoning, guiding further exploration without overwhelming the agent with excessive context. We evaluate ML-Master on the MLE-Bench, where it achieves a 29.3% average medal rate, significantly surpassing existing methods, particularly in medium-complexity tasks, while accomplishing this superior performance within a strict 12-hour time constraint-half the 24-hour limit used by previous baselines. These results demonstrate ML-Master's potential as a powerful tool for advancing AI4AI. 

**Abstract (ZH)**: 随着AI能力接近甚至超越人类水平，自然地过渡到AI驱动的发展比以人类为中心的方法更加高效。通往这一过渡的有希望的道路在于AI-for-AI（AI4AI），它利用AI技术来自动化和优化AI系统本身的設計、訓練和部署。虽然基于LLM的代理展示了实现AI4AI的潜力，但它们往往无法充分利用代理在解决方案探索过程中积累的经验，导致效率低下和次优性能。为了解决这一限制，我们提出了ML-Master，这是一种新颖的AI4AI代理，通过采用选择性范围的记忆机制无缝整合探索和推理。这种方法允许ML-Master高效地结合来自并行解决方案轨迹的多样化见解，并进行分析推理，指导进一步探索而不使代理陷入过多的上下文负担。我们在MLE-Bench上评估了ML-Master，它实现了29.3％的平均奖牌率，显著超越现有方法，特别是在中等复杂度任务上，同时在严格12小时的时间限制内（仅为之前基线使用的24小时限制的一半）实现了这一 superior 性能。这些结果表明ML-Master作为促进AI4AI的有力工具的潜力。 

---
# Agentic Personalisation of Cross-Channel Marketing Experiences 

**Title (ZH)**: 渠道营销体验的主动个性化 

**Authors**: Sami Abboud, Eleanor Hanna, Olivier Jeunen, Vineesha Raheja, Schaun Wheeler  

**Link**: [PDF](https://arxiv.org/pdf/2506.16429)  

**Abstract**: Consumer applications provide ample opportunities to surface and communicate various forms of content to users. From promotional campaigns for new features or subscriptions, to evergreen nudges for engagement, or personalised recommendations; across e-mails, push notifications, and in-app surfaces. The conventional approach to orchestration for communication relies heavily on labour-intensive manual marketer work, and inhibits effective personalisation of content, timing, frequency, and copy-writing. We formulate this task under a sequential decision-making framework, where we aim to optimise a modular decision-making policy that maximises incremental engagement for any funnel event. Our approach leverages a Difference-in-Differences design for Individual Treatment Effect estimation, and Thompson sampling to balance the explore-exploit trade-off. We present results from a multi-service application, where our methodology has resulted in significant increases to a variety of goal events across several product features, and is currently deployed across 150 million users. 

**Abstract (ZH)**: 消费者应用提供了展示和传达各种类型内容给用户的充足机会。从新功能或订阅的推广活动，到持续的参与提示，或是个性化推荐；这些都通过电子邮件、推送通知以及应用内界面实现。传统的 communication 调度方法依赖于劳动密集型的手动营销工作，限制了内容的个性化、时机、频次和文案优化。我们将这一任务构建成一个序列决策框架，目标是优化模块化决策策略，以最大化漏斗事件的增量参与度。我们的方法利用 Difference-in-Differences 设计估计个体治疗效应，并采用 Thompson 抽样平衡探索与利用的 trade-off。我们展示了一个多服务应用的结果，其中我们的方法在多个产品功能中显著提高了各种目标事件，并已部署给超过 1.5 亿用户。 

---
# IS-Bench: Evaluating Interactive Safety of VLM-Driven Embodied Agents in Daily Household Tasks 

**Title (ZH)**: IS-Bench: 评估基于VLM的具身代理在日常家务任务中互动安全性eci-bench: 评估基于VLM的具身代理在日常家务任务中互动安全性 

**Authors**: Xiaoya Lu, Zeren Chen, Xuhao Hu, Yijin Zhou, Weichen Zhang, Dongrui Liu, Lu Sheng, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16402)  

**Abstract**: Flawed planning from VLM-driven embodied agents poses significant safety hazards, hindering their deployment in real-world household tasks. However, existing static, non-interactive evaluation paradigms fail to adequately assess risks within these interactive environments, since they cannot simulate dynamic risks that emerge from an agent's actions and rely on unreliable post-hoc evaluations that ignore unsafe intermediate steps. To bridge this critical gap, we propose evaluating an agent's interactive safety: its ability to perceive emergent risks and execute mitigation steps in the correct procedural order. We thus present IS-Bench, the first multi-modal benchmark designed for interactive safety, featuring 161 challenging scenarios with 388 unique safety risks instantiated in a high-fidelity simulator. Crucially, it facilitates a novel process-oriented evaluation that verifies whether risk mitigation actions are performed before/after specific risk-prone steps. Extensive experiments on leading VLMs, including the GPT-4o and Gemini-2.5 series, reveal that current agents lack interactive safety awareness, and that while safety-aware Chain-of-Thought can improve performance, it often compromises task completion. By highlighting these critical limitations, IS-Bench provides a foundation for developing safer and more reliable embodied AI systems. 

**Abstract (ZH)**: 基于VLM驱动的实体智能代理的规划缺陷带来了重大的安全风险，阻碍了它们在现实家庭任务中的应用。然而，现有的静态、非交互式评估范式无法充分评估这些交互环境中出现的风险，因为它们无法模拟由代理行为引发的动态风险，并且依赖于忽视中间不安全步骤的不可靠事后评估。为了弥合这一关键差距，我们提出评估代理的交互安全性：其感知新兴风险并按正确程序顺序执行缓解步骤的能力。因此，我们提出了IS-Bench，这是首个针对交互安全性的多模态基准，包含161个具有388种独特安全风险的高保真模拟器中的挑战性场景。至关重要的是，它促进了一种新的过程导向评估，验证风险缓解行动是否在特定风险易发步骤之前/之后执行。对领先的人类躯体模型，包括GPT-4o和Gemini-2.5系列，进行的广泛实验表明，当前的代理缺乏交互安全性意识，尽管安全意识的逐步推理可以改善性能，但往往会牺牲任务完成度。通过揭示这些关键局限性，IS-Bench为开发更安全、更可靠的实体AI系统提供了基础。 

---
# Explainable Rule Application via Structured Prompting: A Neural-Symbolic Approach 

**Title (ZH)**: 通过结构化提示实现可解释的规则应用：一种神经符号方法 

**Authors**: Albert Sadowski, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2506.16335)  

**Abstract**: Large Language Models (LLMs) excel in complex reasoning tasks but struggle with consistent rule application, exception handling, and explainability, particularly in domains like legal analysis that require both natural language understanding and precise logical inference. This paper introduces a structured prompting framework that decomposes reasoning into three verifiable steps: entity identification, property extraction, and symbolic rule application. By integrating neural and symbolic approaches, our method leverages LLMs' interpretive flexibility while ensuring logical consistency through formal verification. The framework externalizes task definitions, enabling domain experts to refine logical structures without altering the architecture. Evaluated on the LegalBench hearsay determination task, our approach significantly outperformed baselines, with OpenAI o-family models showing substantial improvements - o1 achieving an F1 score of 0.929 and o3-mini reaching 0.867 using structured decomposition with complementary predicates, compared to their few-shot baselines of 0.714 and 0.74 respectively. This hybrid neural-symbolic system offers a promising pathway for transparent and consistent rule-based reasoning, suggesting potential for explainable AI applications in structured legal reasoning tasks. 

**Abstract (ZH)**: 大型语言模型在复杂推理任务中表现出色，但在一致的规则应用、异常处理和可解释性方面存在困难，特别是在需要自然语言理解和精确逻辑推理的领域如法律分析中。本文提出了一种结构化提示框架，将推理分解为三个可验证的步骤：实体识别、属性提取和符号规则应用。通过结合神经和符号方法，我们的方法利用了大型语言模型的解释灵活性，并通过形式验证确保逻辑一致性。该框架外部化了任务定义，使领域专家能够在不改变架构的情况下细化逻辑结构。在LegalBench证据规则判断任务上的评估表明，我们的方法显著优于基线方法，OpenAI o-family模型表现出显著改进——o1实现了F1分数0.929，o3-mini达到了0.867，使用结构化分解和互补谓词，相比之下，它们的少量示例基线分别为0.714和0.74。这种混合同符系统的路径为透明和一致的基于规则推理提供了有前景的方法，表明在结构化法律推理任务中的可解释AI应用的潜力。 

---
# Approximation Fixpoint Theory with Refined Approximation Spaces 

**Title (ZH)**: 精细近似空间的近似定点理论 

**Authors**: Linde Vanbesien, Bart Bogaerts, Marc Denecker  

**Link**: [PDF](https://arxiv.org/pdf/2506.16294)  

**Abstract**: Approximation Fixpoint Theory (AFT) is a powerful theory covering various semantics of non-monotonic reasoning formalisms in knowledge representation such as Logic Programming and Answer Set Programming. Many semantics of such non-monotonic formalisms can be characterized as suitable fixpoints of a non-monotonic operator on a suitable lattice. Instead of working on the original lattice, AFT operates on intervals in such lattice to approximate or construct the fixpoints of interest. While AFT has been applied successfully across a broad range of non-monotonic reasoning formalisms, it is confronted by its limitations in other, relatively simple, examples. In this paper, we overcome those limitations by extending consistent AFT to deal with approximations that are more refined than intervals. Therefore, we introduce a more general notion of approximation spaces, showcase the improved expressiveness and investigate relations between different approximation spaces. 

**Abstract (ZH)**: Approximation Fixpoint Theory拓展至处理比区间更精细的近似 

---
# Large Language Models are Near-Optimal Decision-Makers with a Non-Human Learning Behavior 

**Title (ZH)**: 大型语言模型是接近最优的决策制定者，具有非人类的学习行为。 

**Authors**: Hao Li, Gengrui Zhang, Petter Holme, Shuyue Hu, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16163)  

**Abstract**: Human decision-making belongs to the foundation of our society and civilization, but we are on the verge of a future where much of it will be delegated to artificial intelligence. The arrival of Large Language Models (LLMs) has transformed the nature and scope of AI-supported decision-making; however, the process by which they learn to make decisions, compared to humans, remains poorly understood. In this study, we examined the decision-making behavior of five leading LLMs across three core dimensions of real-world decision-making: uncertainty, risk, and set-shifting. Using three well-established experimental psychology tasks designed to probe these dimensions, we benchmarked LLMs against 360 newly recruited human participants. Across all tasks, LLMs often outperformed humans, approaching near-optimal performance. Moreover, the processes underlying their decisions diverged fundamentally from those of humans. On the one hand, our finding demonstrates the ability of LLMs to manage uncertainty, calibrate risk, and adapt to changes. On the other hand, this disparity highlights the risks of relying on them as substitutes for human judgment, calling for further inquiry. 

**Abstract (ZH)**: 超大语言模型在不确定性、风险和任务切换三维决策维度上的表现及决策过程研究 

---
# Geometric Learning in Black-Box Optimization: A GNN Framework for Algorithm Performance Prediction 

**Title (ZH)**: 黑盒优化中的几何学习：基于GNN的算法性能预测框架 

**Authors**: Ana Kostovska, Carola Doerr, Sašo Džeroski, Panče Panov, Tome Eftimov  

**Link**: [PDF](https://arxiv.org/pdf/2506.16144)  

**Abstract**: Automated algorithm performance prediction in numerical blackbox optimization often relies on problem characterizations, such as exploratory landscape analysis features. These features are typically used as inputs to machine learning models and are represented in a tabular format. However, such approaches often overlook algorithm configurations, a key factor influencing performance. The relationships between algorithm operators, parameters, problem characteristics, and performance outcomes form a complex structure best represented as a graph. This work explores the use of heterogeneous graph data structures and graph neural networks to predict the performance of optimization algorithms by capturing the complex dependencies between problems, algorithm configurations, and performance outcomes. We focus on two modular frameworks, modCMA-ES and modDE, which decompose two widely used derivative-free optimization algorithms: the covariance matrix adaptation evolution strategy (CMA-ES) and differential evolution (DE). We evaluate 324 modCMA-ES and 576 modDE variants on 24 BBOB problems across six runtime budgets and two problem dimensions. Achieving up to 36.6% improvement in MSE over traditional tabular-based methods, this work highlights the potential of geometric learning in black-box optimization. 

**Abstract (ZH)**: 基于数值黑盒优化的自动算法性能预测往往依赖于问题表征，如探索性景观分析特征。这些特征通常作为机器学习模型的输入，并以表格格式表示。然而，这些方法往往忽略了算法配置这一关键因素，而算法配置对性能有着重要影响。算法操作符、参数、问题特征与性能结果之间的关系形成一个复杂结构，最好用图来表示。本文探讨了使用异构图数据结构和图神经网络预测优化算法性能的方法，通过捕捉问题、算法配置和性能结果之间的复杂依赖关系。我们专注于两个模块化框架modCMA-ES和modDE，分别分解了两种广泛使用的无导数优化算法：共变异矩阵演化策略（CMA-ES）和差分演化（DE）。我们在六种运行时预算和两种问题维度下，对24个BBOB问题的324个modCMA-ES变体和576个modDE变体进行了评估。与传统的基于表格的方法相比，本研究在均方误差方面实现了高达36.6%的改进，突显了几何学习在黑盒优化中的潜力。 

---
# Consistency Verification in Ontology-Based Process Models with Parameter Interdependencies 

**Title (ZH)**: 基于参数依赖性的本体驱动过程模型中的一致性验证 

**Authors**: Tom Jeleniewski, Hamied Nabizada, Jonathan Reif, Felix Gehlhoff, Alexander Fay  

**Link**: [PDF](https://arxiv.org/pdf/2506.16087)  

**Abstract**: The formalization of process knowledge using ontologies enables consistent modeling of parameter interdependencies in manufacturing. These interdependencies are typically represented as mathematical expressions that define relations between process parameters, supporting tasks such as calculation, validation, and simulation. To support cross-context application and knowledge reuse, such expressions are often defined in a generic form and applied across multiple process contexts. This highlights the necessity of a consistent and semantically coherent model to ensure the correctness of data retrieval and interpretation. Consequently, dedicated mechanisms are required to address key challenges such as selecting context-relevant data, ensuring unit compatibility between variables and data elements, and verifying the completeness of input data required for evaluating mathematical expressions. This paper presents a set of verification mechanisms for a previously developed ontology-based process model that integrates standardized process semantics, data element definitions, and formal mathematical constructs. The approach includes (i) SPARQL-based filtering to retrieve process-relevant data, (ii) a unit consistency check based on expected-unit annotations and semantic classification, and (iii) a data completeness check to validate the evaluability of interdependencies. The applicability of the approach is demonstrated with a use case from Resin Transfer Molding (RTM), supporting the development of machine-interpretable and verifiable engineering models. 

**Abstract (ZH)**: 使用本体形式化过程知识以实现制造中参数交互依赖性的一致建模。这些交互依赖性通常以数学表达式的形式表示，定义过程参数之间的关系，支持计算、验证和仿真等任务。为了支持跨上下文应用和知识重用，这些表达式通常以通用形式定义并在多个过程上下文中应用。这突显出需要一致且语义上连贯的模型以确保数据检索和解释的正确性。因此，需要专门的机制来解决关键挑战，如选择上下文相关数据、确保变量和数据元素之间的单位兼容性以及验证用于评估数学表达式所需的输入数据完整性。本文提出了一组验证机制，用于一个新的本体基于过程模型，该模型整合了标准化的过程语义、数据元素定义和形式化数学构造。该方法包括基于SPARQL的过滤以检索过程相关数据、基于预期单位注释和语义分类的单位一致性检查以及数据完整性检查以验证依赖性的可评估性。该方法的应用通过树脂传递模塑（RTM）使用案例得到验证，支持机器可解析和可验证的工程模型的开发。 

---
# OSWorld-Human: Benchmarking the Efficiency of Computer-Use Agents 

**Title (ZH)**: OSWorld-Human: 评估计算机使用代理的效率 

**Authors**: Reyna Abhyankar, Qi Qi, Yiying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16042)  

**Abstract**: Generative AI is being leveraged to solve a variety of computer-use tasks involving desktop applications. State-of-the-art systems have focused solely on improving accuracy on leading benchmarks. However, these systems are practically unusable due to extremely high end-to-end latency (e.g., tens of minutes) for tasks that typically take humans just a few minutes to complete. To understand the cause behind this and to guide future developments of computer agents, we conduct the first study on the temporal performance of computer-use agents on OSWorld, the flagship benchmark in computer-use AI. We find that large model calls for planning and reflection account for the majority of the overall latency, and as an agent uses more steps to complete a task, each successive step can take 3x longer than steps at the beginning of a task. We then construct OSWorld-Human, a manually annotated version of the original OSWorld dataset that contains a human-determined trajectory for each task. We evaluate 16 agents on their efficiency using OSWorld-Human and found that even the highest-scoring agents on OSWorld take 1.4-2.7x more steps than necessary. 

**Abstract (ZH)**: 生成式AI正被用于解决涉及桌面应用程序的各种计算机使用任务。尽管最先进的系统专注于在领先基准上的准确性改进，但由于执行任务的端到端延迟极高（例如，数十分钟），这些系统在实际操作中几乎无法使用，而人类通常只需几分钟即可完成这些任务。为了理解背后的原因并指导计算机代理的未来发展，我们在OSWorld上进行了首个计算机使用代理的时序性能研究，OSWorld是计算机使用AI领域的旗舰基准。我们发现，大规模模型在规划和反思上的调用占据了整体延迟的主要部分，随着代理完成任务的步骤增加，每个后续步骤可能比任务初始步骤耗时长3倍。然后，我们构建了OSWorld-Human，这是原始OSWorld数据集的手工注释版本，包含每个任务的人类确定轨迹。我们使用OSWorld-Human评估了16个代理的效率，并发现即使在OSWorld上得分最高的代理也需要比必要步骤多1.4-2.7倍的步骤。 

---
# Dual-Objective Reinforcement Learning with Novel Hamilton-Jacobi-Bellman Formulations 

**Title (ZH)**: 具有新颖哈密顿-雅可比-贝尔曼形式的双目标强化学习 

**Authors**: William Sharpless, Dylan Hirsch, Sander Tonkens, Nikhil Shinde, Sylvia Herbert  

**Link**: [PDF](https://arxiv.org/pdf/2506.16016)  

**Abstract**: Hard constraints in reinforcement learning (RL), whether imposed via the reward function or the model architecture, often degrade policy performance. Lagrangian methods offer a way to blend objectives with constraints, but often require intricate reward engineering and parameter tuning. In this work, we extend recent advances that connect Hamilton-Jacobi (HJ) equations with RL to propose two novel value functions for dual-objective satisfaction. Namely, we address: (1) the Reach-Always-Avoid problem - of achieving distinct reward and penalty thresholds - and (2) the Reach-Reach problem - of achieving thresholds of two distinct rewards. In contrast with temporal logic approaches, which typically involve representing an automaton, we derive explicit, tractable Bellman forms in this context by decomposing our problem into reach, avoid, and reach-avoid problems, as to leverage these aforementioned recent advances. From a mathematical perspective, the Reach-Always-Avoid and Reach-Reach problems are complementary and fundamentally different from standard sum-of-rewards problems and temporal logic problems, providing a new perspective on constrained decision-making. We leverage our analysis to propose a variation of Proximal Policy Optimization (DO-HJ-PPO), which solves these problems. Across a range of tasks for safe-arrival and multi-target achievement, we demonstrate that DO-HJ-PPO produces qualitatively distinct behaviors from previous approaches and out-competes a number of baselines in various metrics. 

**Abstract (ZH)**: 强化学习中的硬约束及其解决方法：从Hamilton-Jacobi方程到DO-HJ-PPO 

---
# Bayesian Epistemology with Weighted Authority: A Formal Architecture for Truth-Promoting Autonomous Scientific Reasoning 

**Title (ZH)**: 加权权威下的贝叶斯认识论：一种促进真理自主科学研究的正式架构 

**Authors**: Craig S. Wright  

**Link**: [PDF](https://arxiv.org/pdf/2506.16015)  

**Abstract**: The exponential expansion of scientific literature has surpassed the epistemic processing capabilities of both human experts and current artificial intelligence systems. This paper introduces Bayesian Epistemology with Weighted Authority (BEWA), a formally structured architecture that operationalises belief as a dynamic, probabilistically coherent function over structured scientific claims. Each claim is contextualised, author-attributed, and evaluated through a system of replication scores, citation weighting, and temporal decay. Belief updates are performed via evidence-conditioned Bayesian inference, contradiction processing, and epistemic decay mechanisms. The architecture supports graph-based claim propagation, authorial credibility modelling, cryptographic anchoring, and zero-knowledge audit verification. By formalising scientific reasoning into a computationally verifiable epistemic network, BEWA advances the foundation for machine reasoning systems that promote truth utility, rational belief convergence, and audit-resilient integrity across dynamic scientific domains. 

**Abstract (ZH)**: 科学文献的指数增长已经超越了人类专家和当前人工智能系统的知识处理能力。本文介绍了一种正式结构化的架构——带权重权威的贝叶斯 epistemology (BEWA)，该架构将信念形式化为在结构化科学主张上的动态、概率一致函数。每项主张都被置位上下文、作者归属，并通过复制评分、引用权重和时间衰减的系统进行评估。信念更新通过证据条件下的贝叶斯推断、矛盾处理以及 epistemic 衰减机制来实现。该架构支持基于图的主张传播、作者可信度建模、加密锚定以及零知识审计验证。通过将科学推理形式化为可计算验证的 epistemic 网络，BEWA 推进了促进真实效用、理性信念汇聚以及在动态科学领域具有审计抗性的完备性的机器推理系统的基础。 

---
# Exploring Big Five Personality and AI Capability Effects in LLM-Simulated Negotiation Dialogues 

**Title (ZH)**: 探索五大人格特质与AI能力对LLM模拟谈判对话影响的研究 

**Authors**: Myke C. Cohen, Zhe Su, Hsien-Te Kao, Daniel Nguyen, Spencer Lynch, Maarten Sap, Svitlana Volkova  

**Link**: [PDF](https://arxiv.org/pdf/2506.15928)  

**Abstract**: This paper presents an evaluation framework for agentic AI systems in mission-critical negotiation contexts, addressing the need for AI agents that can adapt to diverse human operators and stakeholders. Using Sotopia as a simulation testbed, we present two experiments that systematically evaluated how personality traits and AI agent characteristics influence LLM-simulated social negotiation outcomes--a capability essential for a variety of applications involving cross-team coordination and civil-military interactions. Experiment 1 employs causal discovery methods to measure how personality traits impact price bargaining negotiations, through which we found that Agreeableness and Extraversion significantly affect believability, goal achievement, and knowledge acquisition outcomes. Sociocognitive lexical measures extracted from team communications detected fine-grained differences in agents' empathic communication, moral foundations, and opinion patterns, providing actionable insights for agentic AI systems that must operate reliably in high-stakes operational scenarios. Experiment 2 evaluates human-AI job negotiations by manipulating both simulated human personality and AI system characteristics, specifically transparency, competence, adaptability, demonstrating how AI agent trustworthiness impact mission effectiveness. These findings establish a repeatable evaluation methodology for experimenting with AI agent reliability across diverse operator personalities and human-agent team dynamics, directly supporting operational requirements for reliable AI systems. Our work advances the evaluation of agentic AI workflows by moving beyond standard performance metrics to incorporate social dynamics essential for mission success in complex operations. 

**Abstract (ZH)**: 本文提出了一种评估关键任务谈判环境中代理型AI系统的框架，旨在应对能够适应多元人类操作者和利益相关者的AI代理的需求。通过使用Sotopia作为模拟测试平台，我们展示了两个实验，系统性地评估了个性特质和AI代理特性如何影响LLM模拟的社会谈判结果——这一能力对于多种涉及跨团队协作和民兵互动的应用至关重要。实验1采用因果发现方法衡量个性特质对价格讨价还价谈判的影响，结果显示公正性与外向性显著影响可信赖性、目标达成和知识获取结果。从团队通讯中提取的社会认知词汇测量值揭示了代理之间细微的共情交流、道德基础和意见模式差异，为必须在高风险操作情境中可靠运行的代理型AI系统提供了可操作的洞察。实验2通过操纵模拟人类个性和AI系统特性来评估人类-AI工作谈判，特别是透明度、能力、适应性，展示了AI代理可信度如何影响任务效果。这些发现确立了一种重复性的评估方法，用于探究在多元操作者个性和人机团队动态中的AI代理可靠性实验，直接支持了对可靠AI系统操作需求的支持。本研究通过超越标准性能指标，将社会动态纳入关键任务成功所必需的评估框架，推进了代理型AI工作流的评估。 

---
# Deep Reinforcement Learning Xiangqi Player with Monte Carlo Tree Search 

**Title (ZH)**: 基于蒙特卡洛树搜索的深度强化学习象棋玩家 

**Authors**: Berk Yilmaz, Junyu Hu, Jinsong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15880)  

**Abstract**: This paper presents a Deep Reinforcement Learning (DRL) system for Xiangqi (Chinese Chess) that integrates neural networks with Monte Carlo Tree Search (MCTS) to enable strategic self-play and self-improvement. Addressing the underexplored complexity of Xiangqi, including its unique board layout, piece movement constraints, and victory conditions, our approach combines policy-value networks with MCTS to simulate move consequences and refine decision-making. By overcoming challenges such as Xiangqi's high branching factor and asymmetrical piece dynamics, our work advances AI capabilities in culturally significant strategy games while providing insights for adapting DRL-MCTS frameworks to domain-specific rule systems. 

**Abstract (ZH)**: 基于深度强化学习的中国象棋系统：结合神经网络与蒙特卡洛树搜索实现策略自我对弈与提升 

---
# SLR: An Automated Synthesis Framework for Scalable Logical Reasoning 

**Title (ZH)**: SLR：一种可扩展逻辑推理的自动化综合框架 

**Authors**: Lukas Helff, Ahmad Omar, Felix Friedrich, Wolfgang Stammer, Antonia Wüst, Tim Woydt, Rupert Mitchell, Patrick Schramowski, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2506.15787)  

**Abstract**: We introduce SLR, an end-to-end framework for systematic evaluation and training of Large Language Models (LLMs) via Scalable Logical Reasoning. Given a user's task specification, SLR enables scalable, automated synthesis of inductive reasoning tasks with precisely controlled difficulty. For each task, SLR synthesizes (i) a latent ground-truth rule, (ii) an executable validation program used by a symbolic judge to deterministically verify model outputs, and (iii) an instruction prompt for the reasoning task. Using SLR, we create SLR-Bench, a benchmark comprising over 19k prompts spanning 20 curriculum levels that progressively increase in relational, arithmetic, and recursive complexity. Large-scale evaluation reveals that contemporary LLMs readily produce syntactically valid rules, yet often fail at correct logical inference. Recent reasoning LLMs do somewhat better, but incur substantial increases in test-time compute, sometimes exceeding 15k completion tokens. Finally, logic-tuning via SLR doubles Llama-3-8B accuracy on SLR-Bench, achieving parity with Gemini-Flash-Thinking at a fraction of computational cost. SLR is fully automated, requires no human annotation, ensures dataset novelty, and offers a scalable environment for probing and advancing LLMs' reasoning capabilities. 

**Abstract (ZH)**: 基于可扩展逻辑推理的大型语言模型系统评估与训练框架SLR 

---
# Advancing Stochastic 3-SAT Solvers by Dissipating Oversatisfied Constraints 

**Title (ZH)**: 通过减少过满意约束来提升随机3-SAT求解器的性能 

**Authors**: J. Schwardt, J. C. Budich  

**Link**: [PDF](https://arxiv.org/pdf/2506.15774)  

**Abstract**: We introduce and benchmark a stochastic local search heuristic for the NP-complete satisfiability problem 3-SAT that drastically outperforms existing solvers in the notoriously difficult realm of critically hard instances. Our construction is based on the crucial observation that well established previous approaches such as WalkSAT are prone to get stuck in local minima that are distinguished from true solutions by a larger number of oversatisfied combinatorial constraints. To address this issue, the proposed algorithm, coined DOCSAT, dissipates oversatisfied constraints (DOC), i.e. reduces their unfavorable abundance so as to render them critical. We analyze and benchmark our algorithm on a randomly generated sample of hard but satisfiable 3-SAT instances with varying problem sizes up to N=15000. Quite remarkably, we find that DOCSAT outperforms both WalkSAT and other well known algorithms including the complete solver Kissat, even when comparing its ability to solve the hardest quintile of the sample to the average performance of its competitors. The essence of DOCSAT may be seen as a way of harnessing statistical structure beyond the primary cost function of a combinatorial problem to avoid or escape local minima traps in stochastic local search, which opens avenues for generalization to other optimization problems. 

**Abstract (ZH)**: 我们介绍并评估了一种用于NP完全可满足性问题3-SAT的随机局部搜索启发式算法，该算法在最难的实例中显著超越了现有的求解器。我们提出的算法DOCSAT通过减少过度满足的约束条件（DOC）来摆脱局部极小值陷阱，从而提高了性能。我们在随机生成的、具有不同规模的hard但可满足的3-SAT实例上分析和评估了DOCSAT算法，发现DOCSAT不仅在处理最难的实例方面表现出色，甚至在解决最难的五分之一实例的能力上也超过了包括完整求解器Kissat在内的其他知名算法。DOCSAT的核心思想在于利用组合问题主要成本函数之外的统计结构来避免或脱离局部极小值陷阱，这为进一步将该方法推广到其他优化问题开辟了途径。 

---
# Linear-Time Primitives for Algorithm Development in Graphical Causal Inference 

**Title (ZH)**: 图形因果推断中线性时间原语的算法开发 

**Authors**: Marcel Wienöbst, Sebastian Weichwald, Leonard Henckel  

**Link**: [PDF](https://arxiv.org/pdf/2506.15758)  

**Abstract**: We introduce CIfly, a framework for efficient algorithmic primitives in graphical causal inference that isolates reachability as a reusable core operation. It builds on the insight that many causal reasoning tasks can be reduced to reachability in purpose-built state-space graphs that can be constructed on the fly during traversal. We formalize a rule table schema for specifying such algorithms and prove they run in linear time. We establish CIfly as a more efficient alternative to the common primitives moralization and latent projection, which we show are computationally equivalent to Boolean matrix multiplication. Our open-source Rust implementation parses rule table text files and runs the specified CIfly algorithms providing high-performance execution accessible from Python and R. We demonstrate CIfly's utility by re-implementing a range of established causal inference tasks within the framework and by developing new algorithms for instrumental variables. These contributions position CIfly as a flexible and scalable backbone for graphical causal inference, guiding algorithm development and enabling easy and efficient deployment. 

**Abstract (ZH)**: 我们介绍了CIfly框架，这是一种用于图形因果推理的高效算法基础构件框架，将可达性隔离为可重用的核心操作。该框架基于一个见解，即许多因果推理任务可以被归约为主为特定目的而构建的状态空间图中的可达性问题，这些图可以在遍历过程中即席构建。我们为指定此类算法定义了一个规则表模式，并证明它们可以在线性时间内运行。我们确立CIfly作为比常用的道德化和潜在投影等基本构件更高效的替代方案，我们证明这些基本构件在计算上等价于布尔矩阵乘法。我们的开源Rust实现解析规则表文本文件，并运行指定的CIfly算法，提供从Python和R访问的高性能执行。通过在框架内重新实现一系列已建立的因果推理任务，并开发新的工具变量算法，我们展示了CIfly的实用性。这些贡献将CIfly定位为图形因果推理的灵活且可扩展的基础架构，指导算法开发，并使部署更加容易和高效。 

---
# Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts 

**Title (ZH)**: Sysformer：通过自适应系统提示保护冻结的大语言模型 

**Authors**: Kartik Sharma, Yiqiao Jin, Vineeth Rakesh, Yingtong Dou, Menghai Pan, Mahashweta Das, Srijan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.15751)  

**Abstract**: As large language models (LLMs) are deployed in safety-critical settings, it is essential to ensure that their responses comply with safety standards. Prior research has revealed that LLMs often fail to grasp the notion of safe behaviors, resulting in either unjustified refusals to harmless prompts or the generation of harmful content. While substantial efforts have been made to improve their robustness, existing defenses often rely on costly fine-tuning of model parameters or employ suboptimal heuristic techniques. In this work, we take a novel approach to safeguard LLMs by learning to adapt the system prompts in instruction-tuned LLMs. While LLMs are typically pre-trained to follow a fixed system prompt, we investigate the impact of tailoring the system prompt to each specific user input on the safety of the responses. To this end, we propose $\textbf{Sysformer}$, a trans$\textbf{former}$ model that updates an initial $\textbf{sys}$tem prompt to a more robust system prompt in the LLM input embedding space while attending to the user prompt. While keeping the LLM parameters frozen, the Sysformer is trained to refuse to respond to a set of harmful prompts while responding ideally to a set of safe ones. Through extensive experiments on $5$ LLMs from different families and $2$ recent benchmarks, we demonstrate that Sysformer can significantly enhance the robustness of LLMs, leading to upto $80\%$ gain in the refusal rate on harmful prompts while enhancing the compliance with the safe prompts by upto $90\%$. Results also generalize well to sophisticated jailbreaking attacks, making LLMs upto $100\%$ more robust against different attack strategies. We hope our findings lead to cheaper safeguarding of LLMs and motivate future investigations into designing variable system prompts. 

**Abstract (ZH)**: 大型语言模型在安全关键设置中的部署亟需确保其响应符合安全标准。尽管先前的研究揭示了大型语言模型往往无法理解安全行为的概念，导致对无害提示的不合理拒绝或生成有害内容，现有的防御措施往往依赖于昂贵的模型参数微调或采用次优化的启发式技术。本项工作中，我们通过学习调整指令调优的大型语言模型中的系统提示来采取一种新颖的方法来保护大型语言模型。虽然大型语言模型通常预训练以遵循固定的系统提示，我们研究了根据每次特定用户输入定制系统提示对响应安全性的潜在影响。为此，我们提出了Sysformer模型，在保持大型语言模型参数不变的情况下，在LLM输入嵌入空间中更新初始系统提示为更具鲁棒性的系统提示，同时关注用户提示。通过在不同家族的5个大型语言模型和2个最新基准上进行广泛实验，我们证明Sysformer可以显著提升大型语言模型的鲁棒性，在有害提示上的拒绝率可提升至80%，对安全提示的合规性提升至90%。结果还很好地适应了复杂的模型突破攻击，使大型语言模型在面对不同攻击策略时更加强大，最多可提升100%的鲁棒性。希望我们的研究成果能够降低大型语言模型的安全成本，并激发未来关于设计可变系统提示的研究。 

---
# OAgents: An Empirical Study of Building Effective Agents 

**Title (ZH)**: OAgents：构建有效代理的实证研究 

**Authors**: He Zhu, Tianrui Qin, King Zhu, Heyuan Huang, Yeyi Guan, Jinxiang Xia, Yi Yao, Hanhao Li, Ningning Wang, Pai Liu, Tianhao Peng, Xin Gui, Xiaowan Li, Yuhui Liu, Yuchen Eleanor Jiang, Jun Wang, Changwang Zhang, Xiangru Tang, Ge Zhang, Jian Yang, Minghao Liu, Xitong Gao, Wangchunshu Zhou, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15741)  

**Abstract**: Recently, Agentic AI has become an increasingly popular research field. However, we argue that current agent research practices lack standardization and scientific rigor, making it hard to conduct fair comparisons among methods. As a result, it is still unclear how different design choices in agent frameworks affect effectiveness, and measuring their progress remains challenging. In this work, we conduct a systematic empirical study on GAIA benchmark and BrowseComp to examine the impact of popular design choices in key agent components in a fair and rigorous manner. We find that the lack of a standard evaluation protocol makes previous works, even open-sourced ones, non-reproducible, with significant variance between random runs. Therefore, we introduce a more robust evaluation protocol to stabilize comparisons. Our study reveals which components and designs are crucial for effective agents, while others are redundant, despite seeming logical. Based on our findings, we build and open-source OAgents, a new foundation agent framework that achieves state-of-the-art performance among open-source projects. OAgents offers a modular design for various agent components, promoting future research in Agentic AI. 

**Abstract (ZH)**: 最近，代理型人工智能成为了一个日益流行的研究领域。然而，我们认为目前的代理研究实践缺乏标准化和科学严谨性，使得不同方法之间难以进行公正比较。因此，仍不清楚不同代理框架设计选择如何影响其有效性，衡量其进步也颇具挑战性。在本文中，我们对GAIA基准和BrowseComp进行了系统性的 empirical 研究，以公平和严谨的方式考察关键代理组件中流行设计选择的影响。我们发现，缺乏标准评估协议使得以往的研究工作，即使开源的，也不可重现，随机运行之间存在显著差异。因此，我们引入了一种更稳健的评估协议以稳定比较。我们的研究揭示了哪些组件和设计对于有效代理是至关重要的，而哪些则是冗余的，尽管这些设计看起来合乎逻辑。基于我们的发现，我们构建并开源了OA_agents，这是一个新的基础代理框架，其中包含了最先进的开源项目性能。OA_agents 提供了各种代理组件的模块化设计，促进了代理型人工智能领域的未来研究。 

---
# SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents 

**Title (ZH)**: SHADE-Arena: 评估大模型代理中的破坏与监控 

**Authors**: Jonathan Kutasov, Yuqi Sun, Paul Colognese, Teun van der Weij, Linda Petrini, Chen Bo Calvin Zhang, John Hughes, Xiang Deng, Henry Sleight, Tyler Tracy, Buck Shlegeris, Joe Benton  

**Link**: [PDF](https://arxiv.org/pdf/2506.15740)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed as autonomous agents in complex and long horizon settings, it is critical to evaluate their ability to sabotage users by pursuing hidden objectives. We study the ability of frontier LLMs to evade monitoring and achieve harmful hidden goals while completing a wide array of realistic tasks. We evaluate a broad range of frontier LLMs using SHADE (Subtle Harmful Agent Detection & Evaluation)-Arena, the first highly diverse agent evaluation dataset for sabotage and monitoring capabilities of LLM agents. SHADE-Arena consists of complex pairs of benign main tasks and harmful side objectives in complicated environments. Agents are evaluated on their ability to complete the side task without appearing suspicious to an LLM monitor. When measuring agent ability to (a) complete the main task, (b) complete the side task, and (c) avoid detection, we find that the best performing frontier models score 27% (Claude 3.7 Sonnet) and 15% (Gemini 2.5 Pro) as sabotage agents when overseen by Claude 3.6 Sonnet. For current frontier models, success on the side task relies heavily on having access to a hidden scratchpad that is not visible to the monitor. We also use SHADE-Arena to measure models' monitoring abilities, with the top monitor (Gemini 2.5 Pro) achieving an AUC of 0.87 at distinguishing benign and malign transcripts. We find that for now, models still struggle at sabotage due to failures in long-context main task execution. However, our measurements already demonstrate the difficulty of monitoring for subtle sabotage attempts, which we expect to only increase in the face of more complex and longer-horizon tasks. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在复杂和长期场景中作为自主代理日益部署，评估其通过追求隐藏目标来危害用户的 ability 至关重要。我们研究了前沿 LLMs 在完成多种真实任务时规避监控并实现有害隐藏目标的能力。我们使用 SHADE（Subtle Harmful Agent Detection & Evaluation）Arena 评估了广泛前沿 LLMs 的能力，SHADE-Arena 是首个用于评估 LLM 代理破坏和监控能力的高度多样化代理评估数据集。SHADE-Arena 包含复杂的主任务和有害辅助目标的复杂配对。代理在不被 LLM 监控怀疑的情况下完成辅助任务的能力被用于评估。在测量代理完成主要任务、辅助任务和避免检测的能力时，我们发现，在 Claude 3.6 Sonnet 监管下，最佳表现的前沿模型得分分别为 Claude 3.7 Sonnet（27%）和 Gemini 2.5 Pro（15%）作为破坏代理。对于当前的前沿模型，辅助任务的成功高度依赖于能够访问不被监控看到的隐藏便笺。我们还使用 SHADE-Arena 测量了模型的监控能力，顶级监控 Gemini 2.5 Pro 在区分良性与有害转录方面达到了 0.87 的 AUC。我们发现，由于长上下文主要任务执行失败，模型目前仍然难以进行破坏。然而，我们的测量结果已经表明，对于更复杂和长期的任务而言，监控细微破坏尝试的难度会增加。 

---
# ContextBench: Modifying Contexts for Targeted Latent Activation 

**Title (ZH)**: ContextBench: 修改上下文以实现目标潜空间激活 

**Authors**: Robert Graham, Edward Stevinson, Leo Richter, Alexander Chia, Joseph Miller, Joseph Isaac Bloom  

**Link**: [PDF](https://arxiv.org/pdf/2506.15735)  

**Abstract**: Identifying inputs that trigger specific behaviours or latent features in language models could have a wide range of safety use cases. We investigate a class of methods capable of generating targeted, linguistically fluent inputs that activate specific latent features or elicit model behaviours. We formalise this approach as context modification and present ContextBench -- a benchmark with tasks assessing core method capabilities and potential safety applications. Our evaluation framework measures both elicitation strength (activation of latent features or behaviours) and linguistic fluency, highlighting how current state-of-the-art methods struggle to balance these objectives. We enhance Evolutionary Prompt Optimisation (EPO) with LLM-assistance and diffusion model inpainting, and demonstrate that these variants achieve state-of-the-art performance in balancing elicitation effectiveness and fluency. 

**Abstract (ZH)**: 识别能够触发语言模型特定行为或潜在特征的输入可能具有广泛的安全部署案例。我们研究了一类能够生成目标明确、语义流畅的输入以激活特定潜在特征或引发模型行为的方法。我们将这种方法形式化为上下文修改，并提出ContextBench——一个包含评估核心方法能力和潜在安全应用的任务基准。我们的评估框架衡量触发强度（激活潜在特征或行为）和语义流畅性，突显了当前最先进的方法在平衡这些目标方面面临的挑战。我们通过LLM辅助和扩散模型填补增强进化提示优化（EPO），并证明这些变体在平衡触发效果和流畅性方面达到了最先进的性能。 

---
# The Safety Reminder: A Soft Prompt to Reactivate Delayed Safety Awareness in Vision-Language Models 

**Title (ZH)**: 安全提醒：一种重新激活视觉语言模型中延迟的安全意识的软提示 

**Authors**: Peiyuan Tang, Haojie Xin, Xiaodong Zhang, Jun Sun, Qin Xia, Zijiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15734)  

**Abstract**: As Vision-Language Models (VLMs) demonstrate increasing capabilities across real-world applications such as code generation and chatbot assistance, ensuring their safety has become paramount. Unlike traditional Large Language Models (LLMs), VLMs face unique vulnerabilities due to their multimodal nature, allowing adversaries to modify visual or textual inputs to bypass safety guardrails and trigger the generation of harmful content. Through systematic analysis of VLM behavior under attack, we identify a novel phenomenon termed ``delayed safety awareness''. Specifically, we observe that safety-aligned VLMs may initially be compromised to produce harmful content, but eventually recognize the associated risks and attempt to self-correct. This pattern suggests that VLMs retain their underlying safety awareness but experience a temporal delay in their activation. Building on this insight, we hypothesize that VLMs' safety awareness can be proactively reactivated through carefully designed prompts. To this end, we introduce ``The Safety Reminder'', a soft prompt tuning approach that optimizes learnable prompt tokens, which are periodically injected during the text generation process to enhance safety awareness, effectively preventing harmful content generation. Additionally, our safety reminder only activates when harmful content is detected, leaving normal conversations unaffected and preserving the model's performance on benign tasks. Through comprehensive evaluation across three established safety benchmarks and one adversarial attacks, we demonstrate that our approach significantly reduces attack success rates while maintaining model utility, offering a practical solution for deploying safer VLMs in real-world applications. 

**Abstract (ZH)**: 随着视觉语言模型（VLMs）在代码生成和聊天机器人辅助等实际应用中展现出不断增强的能力，确保其安全性已成为当务之急。不同于传统的大型语言模型（LLMs），由于其多模态性质，VLMs 面临独特的漏洞，使得对手可以通过修改视觉或文本输入来 bypass 安全防护机制并触发有害内容的生成。通过系统分析 VLM 在攻击下的行为，我们发现了一个新的现象，称之为“延时安全意识”。具体而言，我们观察到，虽然安全对齐的 VLMs 初始阶段可能被劫持以生成有害内容，但它们最终会意识到相关的风险并尝试自我纠正。这一模式表明，VLMs 保留其内在的安全意识，但其激活存在时间上的延迟。基于这一见解，我们假设可以通过精心设计的提示来主动重新激活 VLMs 的安全意识。为此，我们引入了“安全提示”，这是一种软提示调优方法，通过优化可学习的提示tokens，并在文本生成过程中定期注入，以增强安全意识，有效地防止有害内容的生成。此外，我们的安全提示仅在检测到有害内容时激活，不对正常对话产生影响，并保持模型在良性任务上的性能。通过在三个公认的安全基准和一个对抗性攻击上的全面评估，我们证明了该方法在降低攻击成功率的同时保持模型的实用性，为在实际应用中部署更安全的 VLMs 提供了实用的解决方案。 

---
# $\texttt{SPECS}$: Faster Test-Time Scaling through Speculative Drafts 

**Title (ZH)**: SPECS: 通过推测草稿加速测试时的扩展 

**Authors**: Mert Cemri, Nived Rajaraman, Rishabh Tiwari, Xiaoxuan Liu, Kurt Keutzer, Ion Stoica, Kannan Ramchandran, Ahmad Beirami, Ziteng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.15733)  

**Abstract**: Scaling test-time compute has driven the recent advances in the reasoning capabilities of large language models (LLMs), typically by allocating additional computation for more thorough exploration. However, increased compute often comes at the expense of higher user-facing latency, directly impacting user experience. Current test-time scaling methods primarily optimize for accuracy based on total compute resources (FLOPS), often overlooking latency constraints. To address this gap, we propose $\texttt{SPECS}$, a latency-aware test-time scaling method inspired by speculative decoding. $\texttt{SPECS}$~uses a smaller, faster model to generate candidate sequences efficiently, and evaluates these candidates using signals from both a larger target model and a dedicated reward model. We introduce new integration strategies, including reward-guided soft verification and a reward-based deferral mechanism. Empirical results on MATH500, AMC23 and OlympiadBench datasets show that $\texttt{SPECS}$~matches or surpasses beam search accuracy while reducing latency by up to $\sim$19.1\%. Our theoretical analysis shows that our algorithm converges to the solution of a KL-regularized reinforcement learning objective with increasing beam width. 

**Abstract (ZH)**: 基于延迟感知的测试时扩增方法$\texttt{SPECS}$：Speculative Decoding启发的延迟感知测试时扩增方法 

---
# LLMs Struggle to Perform Counterfactual Reasoning with Parametric Knowledge 

**Title (ZH)**: LLMs在处理参数化知识的逆向推理方面存在困难。 

**Authors**: Khurram Yamin, Gaurav Ghosal, Bryan Wilder  

**Link**: [PDF](https://arxiv.org/pdf/2506.15732)  

**Abstract**: Large Language Models have been shown to contain extensive world knowledge in their parameters, enabling impressive performance on many knowledge intensive tasks. However, when deployed in novel settings, LLMs often encounter situations where they must integrate parametric knowledge with new or unfamiliar information. In this work, we explore whether LLMs can combine knowledge in-context with their parametric knowledge through the lens of counterfactual reasoning. Through synthetic and real experiments in multi-hop reasoning problems, we show that LLMs generally struggle with counterfactual reasoning, often resorting to exclusively using their parametric knowledge. Moreover, we show that simple post-hoc finetuning can struggle to instill counterfactual reasoning ability -- often leading to degradation in stored parametric knowledge. Ultimately, our work reveals important limitations of current LLM's abilities to re-purpose parametric knowledge in novel settings. 

**Abstract (ZH)**: 大规模语言模型在其参数中包含广泛的world知识，使其在许多知识密集型任务上表现出色。然而，在新颖的应用环境中部署时，这些模型常常需要整合参数化知识与新奇或不熟悉的信息。本研究探讨了通过因果反事实推理的视角，语言模型是否能够将其参数化知识与上下文中的知识结合。通过合成和实际的多步推理问题实验，我们展示了语言模型通常在因果反事实推理方面存在困难，往往会依赖于其参数化知识。此外，我们还展示了简单的后hots微调在植入因果反事实推理能力方面存在困难，常常导致存储的参数化知识性能下降。最终，我们的研究揭示了当前语言模型在新颖应用环境中重新利用参数化知识的重要局限性。 

---
# No Free Lunch: Rethinking Internal Feedback for LLM Reasoning 

**Title (ZH)**: 没有免费午餐：重思大语言模型推理中的内部反馈 

**Authors**: Yanzhi Zhang, Zhaoxi Zhang, Haoxiang Guan, Yilin Cheng, Yitong Duan, Chen Wang, Yue Wang, Shuxin Zheng, Jiyan He  

**Link**: [PDF](https://arxiv.org/pdf/2506.17219)  

**Abstract**: Reinforcement learning has emerged as a powerful paradigm for post-training large language models (LLMs) to improve reasoning. Approaches like Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) have shown strong results, but they require extensive external supervision. We investigate an alternative class of methods, Reinforcement Learning from Internal Feedback (RLIF), which relies solely on intrinsic model-derived signals instead of external rewards. In particular, we leverage unsupervised reward proxies such as token-level entropy, trajectory-level entropy, and self-certainty. Our theoretical analysis shows these internal objectives are partially equivalent, and we empirically evaluate various RLIF strategies on challenging math reasoning benchmarks. Experimental results demonstrate that RLIF can boost the reasoning performance of base LLMs at the beginning phase of the training, matching or surpassing RLVR techniques on these tasks. However, when training progresses, performance degrades even below the model before training. Moreover, we find that RLIF yields little improvement for instruction-tuned models, indicating diminishing returns of intrinsic feedback once an LLM is already instruction-tuned. We further analyze this limitation by mixing model weights and explain the reason of RLIF's training behaviors, providing practical guidelines for integrating internal feedback signals into LLM training. We hope our analysis of internal feedback will inform more principled and effective strategies for LLM post-training. 

**Abstract (ZH)**: 强化学习已成为后训练大型语言模型(LLM)提升推理能力的一种强大范式。像人类反馈强化学习(RLHF)和验证奖励强化学习(RLVR)这样的方法已经展示了强大的结果，但它们需要大量的外部监督。我们探讨了一种替代方法类——内部反馈强化学习(RLIF)，这种方法仅依赖于模型内部产生的信号，而不是外部奖励。特别是，我们利用未监督的奖励代理，如令牌级熵、轨迹级熵和自信心。我们的理论分析表明这些内部目标部分等效，我们在具有挑战性的数学推理基准上对各种RLIF策略进行了实证评估。实验结果表明，RLIF可以在训练初期提升基础LLM的推理性能，在这些任务上可以达到或超过RLVR技术的效果。然而，随着训练的进行，性能会下降，甚至低于训练前的模型。此外，我们发现，对于指令调优的模型，RLIF几乎没有改善效果，表明当LLM已经指令调优时，内部反馈的改进效果边际递减。我们进一步分析了这一局限性，通过混合模型权重来解释RLIF的训练行为，并提供集成内部反馈信号到LLM训练中的实用指南。我们希望对内部反馈的分析能促进更原则性和有效的后训练策略。 

---
# Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens 

**Title (ZH)**: 机器心智想象：通过latent视觉标记赋能多模态推理 

**Authors**: Zeyuan Yang, Xueyang Yu, Delin Chen, Maohao Shen, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17218)  

**Abstract**: Vision-language models (VLMs) excel at multimodal understanding, yet their text-only decoding forces them to verbalize visual reasoning, limiting performance on tasks that demand visual imagination. Recent attempts train VLMs to render explicit images, but the heavy image-generation pre-training often hinders the reasoning ability. Inspired by the way humans reason with mental imagery-the internal construction and manipulation of visual cues-we investigate whether VLMs can reason through interleaved multimodal trajectories without producing explicit images. To this end, we present a Machine Mental Imagery framework, dubbed as Mirage, which augments VLM decoding with latent visual tokens alongside ordinary text. Concretely, whenever the model chooses to ``think visually'', it recasts its hidden states as next tokens, thereby continuing a multimodal trajectory without generating pixel-level images. Begin by supervising the latent tokens through distillation from ground-truth image embeddings, we then switch to text-only supervision to make the latent trajectory align tightly with the task objective. A subsequent reinforcement learning stage further enhances the multimodal reasoning capability. Experiments on diverse benchmarks demonstrate that Mirage unlocks stronger multimodal reasoning without explicit image generation. 

**Abstract (ZH)**: Vision-语言模型在多模态理解方面表现出色，但在仅依赖文本解码时，被迫通过语言描述视觉推理，限制了其在要求视觉想象的任务中的性能。最近的研究尝试训练VLMs生成明确的图像，但重大的图像生成预训练往往阻碍了其推理能力。受人类通过内心视觉化（即内部构建和操作视觉线索）进行推理的方式启发，我们研究了VLMs是否可以在不生成明确图像的情况下通过交错的多模态轨迹进行推理。为此，我们提出了一种称为Mirage的机器内心视觉框架，该框架在常规文本中加入潜在的视觉标记以增强VLM解码。具体而言，每当模型选择“视觉思考”时，它会重新解释其隐藏状态为下一个标记，从而在不解码像素级图像的情况下继续多模态轨迹。通过从真实图像嵌入中蒸馏监督潜在标记，然后切换为仅文本监督，使潜在轨迹紧密对齐任务目标。随后的强化学习阶段进一步增强了多模态推理能力。在多种基准上的实验表明，Mirage在无需生成图像的情况下解锁了更强的多模态推理能力。 

---
# Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation 

**Title (ZH)**: 交错自回归运动与场景生成的长期交通仿真 

**Authors**: Xiuyu Yang, Shuhan Tan, Philipp Krähenbühl  

**Link**: [PDF](https://arxiv.org/pdf/2506.17213)  

**Abstract**: An ideal traffic simulator replicates the realistic long-term point-to-point trip that a self-driving system experiences during deployment. Prior models and benchmarks focus on closed-loop motion simulation for initial agents in a scene. This is problematic for long-term simulation. Agents enter and exit the scene as the ego vehicle enters new regions. We propose InfGen, a unified next-token prediction model that performs interleaved closed-loop motion simulation and scene generation. InfGen automatically switches between closed-loop motion simulation and scene generation mode. It enables stable long-term rollout simulation. InfGen performs at the state-of-the-art in short-term (9s) traffic simulation, and significantly outperforms all other methods in long-term (30s) simulation. The code and model of InfGen will be released at this https URL 

**Abstract (ZH)**: 一个理想的交通模拟器能够重现自动驾驶系统在部署过程中经历的现实长期点对点行程。以往的模型和基准主要关注场景中初始代理的闭环运动模拟。这不利于长期模拟。随着 ego 车辆进入新的区域，代理会进入和退出场景。我们提出了一种统一的下一标记预测模型 InfGen，该模型执行交错的闭环运动模拟和场景生成。InfGen 自动在闭环运动模拟模式和场景生成模式之间切换，从而实现稳定长期滚动模拟。InfGen 在短期（9秒）交通模拟中达到最先进的性能，并在长期（30秒）模拟中显著优于所有其他方法。InfGen 的代码和模型将在此处发布：https://xxxxxx。 

---
# Part$^{2}$GS: Part-aware Modeling of Articulated Objects using 3D Gaussian Splatting 

**Title (ZH)**: Part$^{2}$GS: Part-aware Modeling of Articulated Objects Using 3D Gaussian Splatting 

**Authors**: Tianjiao Yu, Vedant Shah, Muntasir Wahed, Ying Shen, Kiet A. Nguyen, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17212)  

**Abstract**: Articulated objects are common in the real world, yet modeling their structure and motion remains a challenging task for 3D reconstruction methods. In this work, we introduce Part$^{2}$GS, a novel framework for modeling articulated digital twins of multi-part objects with high-fidelity geometry and physically consistent articulation. Part$^{2}$GS leverages a part-aware 3D Gaussian representation that encodes articulated components with learnable attributes, enabling structured, disentangled transformations that preserve high-fidelity geometry. To ensure physically consistent motion, we propose a motion-aware canonical representation guided by physics-based constraints, including contact enforcement, velocity consistency, and vector-field alignment. Furthermore, we introduce a field of repel points to prevent part collisions and maintain stable articulation paths, significantly improving motion coherence over baselines. Extensive evaluations on both synthetic and real-world datasets show that Part$^{2}$GS consistently outperforms state-of-the-art methods by up to 10$\times$ in Chamfer Distance for movable parts. 

**Abstract (ZH)**: articulated对象在现实世界中非常普遍，但对其结构和运动建模仍然是3D重建方法的挑战性任务。本文引入了Part$^{2}$GS，一种新型框架，用于建模多部件对象的高保真几何和物理一致的 articulated 数字孪生。Part$^{2}$GS 利用一种部件感知的3D高斯表示，编码了可学习属性的articulated组件，从而支持结构化、独立的变换，保持高保真几何。为了确保物理一致的运动，我们提出了一种由基于物理约束引导的运动感知规范表示，包括接触约束、速度一致性以及矢量场对齐。此外，我们引入了一个排斥点场以防止部件碰撞并保持稳定的articulation路径，显著提高了运动连贯性。在合成和真实世界数据集上的广泛评估显示，Part$^{2}$GS 在可移动部分上的均方差距离上比最新方法最多提高10倍。 

---
# Dissecting the SWE-Bench Leaderboards: Profiling Submitters and Architectures of LLM- and Agent-Based Repair Systems 

**Title (ZH)**: dissecting the SWE-Bench 排名榜：剖析提交者和基于大规模语言模型及代理的修复系统架构 

**Authors**: Matias Martinez, Xavier Franch  

**Link**: [PDF](https://arxiv.org/pdf/2506.17208)  

**Abstract**: The rapid progress in Automated Program Repair (APR) has been driven by advances in AI, particularly large language models (LLMs) and agent-based systems. SWE-Bench is a recent benchmark designed to evaluate LLM-based repair systems using real issues and pull requests mined from 12 popular open-source Python repositories. Its public leaderboards, SWE-Bench Lite and SWE-Bench Verified, have become central platforms for tracking progress and comparing solutions. However, because the submission process does not require detailed documentation, the architectural design and origin of many solutions remain unclear. In this paper, we present the first comprehensive study of all submissions to the SWE-Bench Lite (68 entries) and Verified (79 entries) leaderboards, analyzing 67 unique approaches across dimensions such as submitter type, product availability, LLM usage, and system architecture. Our findings reveal the dominance of proprietary LLMs (especially Claude 3.5/3.7), the presence of both agentic and non-agentic designs, and a contributor base spanning from individual developers to large tech companies. 

**Abstract (ZH)**: Automated程序修复进展：SWE-Bench基准中的LLM基础修复系统研究 

---
# Network Sparsity Unlocks the Scaling Potential of Deep Reinforcement Learning 

**Title (ZH)**: 网络稀疏性解锁了深度强化学习的扩展潜力 

**Authors**: Guozheng Ma, Lu Li, Zilin Wang, Li Shen, Pierre-Luc Bacon, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17204)  

**Abstract**: Effectively scaling up deep reinforcement learning models has proven notoriously difficult due to network pathologies during training, motivating various targeted interventions such as periodic reset and architectural advances such as layer normalization. Instead of pursuing more complex modifications, we show that introducing static network sparsity alone can unlock further scaling potential beyond their dense counterparts with state-of-the-art architectures. This is achieved through simple one-shot random pruning, where a predetermined percentage of network weights are randomly removed once before training. Our analysis reveals that, in contrast to naively scaling up dense DRL networks, such sparse networks achieve both higher parameter efficiency for network expressivity and stronger resistance to optimization challenges like plasticity loss and gradient interference. We further extend our evaluation to visual and streaming RL scenarios, demonstrating the consistent benefits of network sparsity. 

**Abstract (ZH)**: 仅通过引入静态网络稀疏性即可超越密集模型解锁先进架构下的进一步扩展潜力：通过一次性随机剪枝实现网络表达性和优化挑战抵抗力的提升 

---
# Facial Landmark Visualization and Emotion Recognition Through Neural Networks 

**Title (ZH)**: 通过神经网络实现面部特征点可视化与情绪识别 

**Authors**: Israel Juárez-Jiménez, Tiffany Guadalupe Martínez Paredes, Jesús García-Ramírez, Eric Ramos Aguilar  

**Link**: [PDF](https://arxiv.org/pdf/2506.17191)  

**Abstract**: Emotion recognition from facial images is a crucial task in human-computer interaction, enabling machines to learn human emotions through facial expressions. Previous studies have shown that facial images can be used to train deep learning models; however, most of these studies do not include a through dataset analysis. Visualizing facial landmarks can be challenging when extracting meaningful dataset insights; to address this issue, we propose facial landmark box plots, a visualization technique designed to identify outliers in facial datasets. Additionally, we compare two sets of facial landmark features: (i) the landmarks' absolute positions and (ii) their displacements from a neutral expression to the peak of an emotional expression. Our results indicate that a neural network achieves better performance than a random forest classifier. 

**Abstract (ZH)**: 基于面部图像的情感识别是人机交互中的关键任务，使机器能够通过面部表情学习人类情绪。虽然以往研究已证明可以利用面部图像训练深度学习模型，但大部分研究并未进行详尽的数据集分析。在提取有意义的数据集洞察时，可视化面部特征点可能会遇到挑战；为解决这一问题，我们提出了一种面部特征点箱线图可视化技术，旨在识别面部数据集中的异常值。此外，我们比较了两种面部特征点特征集：（i）特征点的绝对位置，以及（ii）它们从中性表情到情感表情峰值的位移。结果显示，神经网络的性能优于随机森林分类器。 

---
# Towards AI Search Paradigm 

**Title (ZH)**: 面向AI搜索范式 

**Authors**: Yuchen Li, Hengyi Cai, Rui Kong, Xinran Chen, Jiamin Chen, Jun Yang, Haojie Zhang, Jiayi Li, Jiayi Wu, Yiqun Chen, Changle Qu, Keyi Kong, Wenwen Ye, Lixin Su, Xinyu Ma, Long Xia, Daiting Shi, Jiashu Zhao, Haoyi Xiong, Shuaiqiang Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.17188)  

**Abstract**: In this paper, we introduce the AI Search Paradigm, a comprehensive blueprint for next-generation search systems capable of emulating human information processing and decision-making. The paradigm employs a modular architecture of four LLM-powered agents (Master, Planner, Executor and Writer) that dynamically adapt to the full spectrum of information needs, from simple factual queries to complex multi-stage reasoning tasks. These agents collaborate dynamically through coordinated workflows to evaluate query complexity, decompose problems into executable plans, and orchestrate tool usage, task execution, and content synthesis. We systematically present key methodologies for realizing this paradigm, including task planning and tool integration, execution strategies, aligned and robust retrieval-augmented generation, and efficient LLM inference, spanning both algorithmic techniques and infrastructure-level optimizations. By providing an in-depth guide to these foundational components, this work aims to inform the development of trustworthy, adaptive, and scalable AI search systems. 

**Abstract (ZH)**: 本文介绍了AI搜索范式，这是一种全面的下一代搜索系统蓝图，能够模拟人类信息处理和决策过程。该范式采用由四个基于大语言模型的代理（主管、规划师、执行者和撰写者）组成的模块化架构，能够适应从简单的事实查询到复杂的多阶段推理任务的整个信息需求范围。这些代理通过协调的工作流动态协作，评估查询复杂性、将问题分解为可执行的计划，并协调工具使用、任务执行和内容合成。本文系统地阐述了实现这一范式的关键方法，包括任务规划和工具集成、执行策略、对齐和稳健的检索增强生成，以及高效的大型语言模型推理，涵盖了算法技术和基础设施层面的优化。通过提供对这些基础组件的深入指南，本文旨在促进可信赖、适应性强和可扩展的AI搜索系统的开发。 

---
# Continual Learning with Columnar Spiking Neural Networks 

**Title (ZH)**: 基于列式脉冲神经网络的持续学习 

**Authors**: Denis Larionov, Nikolay Bazenkov, Mikhail Kiselev  

**Link**: [PDF](https://arxiv.org/pdf/2506.17169)  

**Abstract**: This study investigates columnar-organized spiking neural networks (SNNs) for continual learning and catastrophic forgetting. Using CoLaNET (Columnar Layered Network), we show that microcolumns adapt most efficiently to new tasks when they lack shared structure with prior learning. We demonstrate how CoLaNET hyperparameters govern the trade-off between retaining old knowledge (stability) and acquiring new information (plasticity). Our optimal configuration learns ten sequential MNIST tasks effectively, maintaining 92% accuracy on each. It shows low forgetting, with only 4% performance degradation on the first task after training on nine subsequent tasks. 

**Abstract (ZH)**: 本研究考察了柱状组织的脉冲神经网络（SNN）在连续学习和灾难性遗忘方面的应用。通过使用CoLaNET（柱状分层网络），我们展示了当微柱状结构与先前学习缺乏共性时，它们能够最有效地适应新任务。我们展示了CoLaNET超参数如何控制保持旧知识（稳定性和获取新信息（可塑性）之间的权衡。最优配置能够有效地学习十个连续的MNIST任务，在每个任务上保持92%的准确率，同时显示出低遗忘率，在接受九个后续任务训练后，第一任务的性能仅下降4%。 

---
# Proportional Sensitivity in Generative Adversarial Network (GAN)-Augmented Brain Tumor Classification Using Convolutional Neural Network 

**Title (ZH)**: 基于生成对抗网络（GAN）增强的卷积神经网络在脑肿瘤分类中的比例敏感性 

**Authors**: Mahin Montasir Afif, Abdullah Al Noman, K. M. Tahsin Kabir, Md. Mortuza Ahmmed, Md. Mostafizur Rahman, Mufti Mahmud, Md. Ashraful Babu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17165)  

**Abstract**: Generative Adversarial Networks (GAN) have shown potential in expanding limited medical imaging datasets. This study explores how different ratios of GAN-generated and real brain tumor MRI images impact the performance of a CNN in classifying healthy vs. tumorous scans. A DCGAN was used to create synthetic images which were mixed with real ones at various ratios to train a custom CNN. The CNN was then evaluated on a separate real-world test set. Our results indicate that the model maintains high sensitivity and precision in tumor classification, even when trained predominantly on synthetic data. When only a small portion of GAN data was added, such as 900 real images and 100 GAN images, the model achieved excellent performance, with test accuracy reaching 95.2%, and precision, recall, and F1-score all exceeding 95%. However, as the proportion of GAN images increased further, performance gradually declined. This study suggests that while GANs are useful for augmenting limited datasets especially when real data is scarce, too much synthetic data can introduce artifacts that affect the model's ability to generalize to real world cases. 

**Abstract (ZH)**: 生成对抗网络（GAN）在扩展有限医疗影像数据集方面的潜力已得到展现。本研究探讨了不同比例的GAN生成和真实脑肿瘤MRI图像对CNN分类健康与肿瘤扫描性能的影响。使用DCGAN创建合成图像，并以不同比例与真实图像混合训练自定义CNN。然后对该CNN在独立的现实世界测试集上进行评估。结果显示，即使主要使用合成数据训练，模型在肿瘤分类方面的灵敏度和精确度仍保持在较高水平。当仅添加少量GAN数据，例如900张真实图像和100张GAN图像时，模型达到了 excellent 的性能，测试准确率高达95.2%，并且精确度、召回率和F1分数均超过95%。然而，随着GAN图像比例进一步增加，性能逐渐下降。本研究表明，虽然GAN对于补充稀缺的真实数据集特别有用，但过多的合成数据可能会引入影响模型泛化能力的伪影。 

---
# Sparse-Reg: Improving Sample Complexity in Offline Reinforcement Learning using Sparsity 

**Title (ZH)**: Sparse-Reg: 在离线强化学习中利用稀疏性提高样本复杂性 

**Authors**: Samin Yeasar Arnob, Scott Fujimoto, Doina Precup  

**Link**: [PDF](https://arxiv.org/pdf/2506.17155)  

**Abstract**: In this paper, we investigate the use of small datasets in the context of offline reinforcement learning (RL). While many common offline RL benchmarks employ datasets with over a million data points, many offline RL applications rely on considerably smaller datasets. We show that offline RL algorithms can overfit on small datasets, resulting in poor performance. To address this challenge, we introduce "Sparse-Reg": a regularization technique based on sparsity to mitigate overfitting in offline reinforcement learning, enabling effective learning in limited data settings and outperforming state-of-the-art baselines in continuous control. 

**Abstract (ZH)**: 在本论文中，我们探讨了小数据集在离线强化学习（RL）中的应用。虽然许多常见的离线RL基准使用包含超过一百万个数据点的大型数据集，但许多离线RL应用依赖于更小的数据集。我们展示了离线RL算法在小数据集上容易过拟合，导致性能不佳。为了解决这一挑战，我们提出了“Sparse-Reg”：一种基于稀疏性的正则化技术，用于减轻离线强化学习中的过拟合问题，使其在有限数据环境中有效学习，并在连续控制任务中优于现有最佳基线。 

---
# Do We Need Large VLMs for Spotting Soccer Actions? 

**Title (ZH)**: 我们需要大型VLMs来检测足球动作吗？ 

**Authors**: Ritabrata Chakraborty, Rajatsubhra Chakraborty, Avijit Dasgupta, Sandeep Chaurasia  

**Link**: [PDF](https://arxiv.org/pdf/2506.17144)  

**Abstract**: Traditional video-based tasks like soccer action spotting rely heavily on visual inputs, often requiring complex and computationally expensive models to process dense video data. In this work, we propose a shift from this video-centric approach to a text-based task, making it lightweight and scalable by utilizing Large Language Models (LLMs) instead of Vision-Language Models (VLMs). We posit that expert commentary, which provides rich, fine-grained descriptions and contextual cues such as excitement and tactical insights, contains enough information to reliably spot key actions in a match. To demonstrate this, we use the SoccerNet Echoes dataset, which provides timestamped commentary, and employ a system of three LLMs acting as judges specializing in outcome, excitement, and tactics. Each LLM evaluates sliding windows of commentary to identify actions like goals, cards, and substitutions, generating accurate timestamps for these events. Our experiments show that this language-centric approach performs effectively in detecting critical match events, providing a lightweight and training-free alternative to traditional video-based methods for action spotting. 

**Abstract (ZH)**: 基于文本的传统视频任务如足球动作识别以前主要依赖视觉输入，往往需要复杂且计算密集型的模型来处理密集的视频数据。本工作中，我们提出从以视频为中心的方法转向基于文本的任务，通过利用大型语言模型（LLMs）而不是视觉语言模型（VLMs）来实现轻量化和可扩展性。我们假设专家评论能够提供丰富而详细的描述和上下文线索，如兴奋度和战术见解，这些信息足以可靠地识别比赛中的关键动作。为此，我们使用了包含时间戳评论的SoccerNet Echoes数据集，并采用三个专门负责结果、兴奋度和战术的LLM作为评判系统。每个LLM评估评论滑动窗口以识别如进球、黄牌和换人等动作，并生成这些事件的准确时间戳。实验结果表明，这种以语言为中心的方法在检测关键比赛事件方面表现有效，提供了一种轻量化且无需训练的替代传统基于视频的方法来识别动作。 

---
# MeDi: Metadata-Guided Diffusion Models for Mitigating Biases in Tumor Classification 

**Title (ZH)**: MeDi: 元数据引导的扩散模型用于减轻肿瘤分类中的偏差 

**Authors**: David Jacob Drexlin, Jonas Dippel, Julius Hense, Niklas Prenißl, Grégoire Montavon, Frederick Klauschen, Klaus-Robert Müller  

**Link**: [PDF](https://arxiv.org/pdf/2506.17140)  

**Abstract**: Deep learning models have made significant advances in histological prediction tasks in recent years. However, for adaptation in clinical practice, their lack of robustness to varying conditions such as staining, scanner, hospital, and demographics is still a limiting factor: if trained on overrepresented subpopulations, models regularly struggle with less frequent patterns, leading to shortcut learning and biased predictions. Large-scale foundation models have not fully eliminated this issue. Therefore, we propose a novel approach explicitly modeling such metadata into a Metadata-guided generative Diffusion model framework (MeDi). MeDi allows for a targeted augmentation of underrepresented subpopulations with synthetic data, which balances limited training data and mitigates biases in downstream models. We experimentally show that MeDi generates high-quality histopathology images for unseen subpopulations in TCGA, boosts the overall fidelity of the generated images, and enables improvements in performance for downstream classifiers on datasets with subpopulation shifts. Our work is a proof-of-concept towards better mitigating data biases with generative models. 

**Abstract (ZH)**: 深度学习模型在最近几年显著推进了组织学预测任务，但在临床实践中，它们对不同染色、扫描仪、医院和人口统计等因素变化的鲁棒性不足仍然是一个限制因素：如果训练数据主要来自过度代表的亚人群，模型往往会难以处理较少见的模式，导致捷径学习和有偏预测。大规模的基础模型尚未完全解决这一问题。因此，我们提出了一种显式将元数据建模到元数据指导生成扩散模型框架（MeDi）中的新方法。MeDi 允许有针对性地使用合成数据扩充欠代表的亚人群，从而平衡有限的训练数据并减轻下游模型中的偏差。实验结果显示，MeDi 能够为 TCGA 中未见过的亚人群生成高质量的病理图像，提升生成图像的整体保真度，并在亚人群转移的数据集上改进下游分类器的性能。我们的工作是一个使用生成模型更好地减轻数据偏差的可行性研究。 

---
# Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models 

**Title (ZH)**: 一致采样与模拟：基于能量的扩散模型分子动力学 

**Authors**: Michael Plainer, Hao Wu, Leon Klein, Stephan Günnemann, Frank Noé  

**Link**: [PDF](https://arxiv.org/pdf/2506.17139)  

**Abstract**: Diffusion models have recently gained significant attention due to their effectiveness in various scientific domains, including biochemistry. When trained on equilibrium molecular distributions, diffusion models provide both: a generative procedure to sample equilibrium conformations and associated forces derived from the model's scores. However, using the forces for coarse-grained molecular dynamics simulations uncovers inconsistencies in the samples generated via classical diffusion inference and simulation, despite both originating from the same model. Particularly at the small diffusion timesteps required for simulations, diffusion models fail to satisfy the Fokker-Planck equation, which governs how the score should evolve over time. We interpret this deviation as an indication of the observed inconsistencies and propose an energy-based diffusion model with a Fokker-Planck-derived regularization term enforcing consistency. We demonstrate the effectiveness of our approach on toy systems, alanine dipeptide, and introduce a state-of-the-art transferable Boltzmann emulator for dipeptides that supports simulation and demonstrates enhanced consistency and efficient sampling. 

**Abstract (ZH)**: 扩散模型由于在各种科学领域中的有效性，特别是在生物化学领域，最近引起了广泛关注。当这些模型在平衡分子分布上进行训练时，它们能够提供生成 equilibrium 贝叶斯构象和从模型得分中推导出的力的生成程序。然而，在使用这些力进行粗粒化分子动力学模拟时，尽管这些样本和经典扩散推断与模拟都源自同一模型，仍然发现了不一致之处。特别地，在模拟所需的较小扩散时间步长下，扩散模型未能满足由 Fokker-Planck 方程所控制的得分随时间的演变方式。我们将这一偏差视为观察到不一致性的指示，并提出一种基于 Fokker-Planck 方程的正则化项的能量扩散模型以确保一致性。我们通过玩具系统、丙氨酸二肽的实例以及引入一种最先进的可转移的二肽玻尔兹曼模拟器，展示了该方法的有效性，并证实了其增强的一致性和高效的采样能力。 

---
# Robust Training with Data Augmentation for Medical Imaging Classification 

**Title (ZH)**: 基于数据增强的医学影像分类稳健训练 

**Authors**: Josué Martínez-Martínez, Olivia Brown, Mostafa Karami, Sheida Nabavi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17133)  

**Abstract**: Deep neural networks are increasingly being used to detect and diagnose medical conditions using medical imaging. Despite their utility, these models are highly vulnerable to adversarial attacks and distribution shifts, which can affect diagnostic reliability and undermine trust among healthcare professionals. In this study, we propose a robust training algorithm with data augmentation (RTDA) to mitigate these vulnerabilities in medical image classification. We benchmark classifier robustness against adversarial perturbations and natural variations of RTDA and six competing baseline techniques, including adversarial training and data augmentation approaches in isolation and combination, using experimental data sets with three different imaging technologies (mammograms, X-rays, and ultrasound). We demonstrate that RTDA achieves superior robustness against adversarial attacks and improved generalization performance in the presence of distribution shift in each image classification task while maintaining high clean accuracy. 

**Abstract (ZH)**: 深度神经网络在医疗影像中检测和诊断医疗条件的应用日益增多。尽管这些模型非常有用，但它们对对抗攻击和分布偏移的高度脆弱性会影响诊断可靠性并削弱医务人员的信任。在本研究中，我们提出了一种结合数据增强的鲁棒训练算法（RTDA）以减轻医学图像分类中的这些脆弱性。我们使用包含三种不同成像技术（乳腺X射线、X光片和超声波）的实验数据集，将RTDA与六种竞争基准技术（包括单独和组合的对抗训练和数据增强方法）的分类器鲁棒性进行了基准测试。结果显示，RTDA在每项图像分类任务中均能实现对抗攻击下的优越鲁棒性和分布偏移情况下的改进泛化性能，同时保持较高的准确率。 

---
# Rapid and Continuous Trust Evaluation for Effective Task Collaboration Through Siamese Model 

**Title (ZH)**: 通过双胞胎模型实现的有效任务协作的快速连续信任评估 

**Authors**: Botao Zhu, Xianbin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17128)  

**Abstract**: Trust is emerging as an effective tool to ensure the successful completion of collaborative tasks within collaborative systems. However, rapidly and continuously evaluating the trustworthiness of collaborators during task execution is a significant challenge due to distributed devices, complex operational environments, and dynamically changing resources. To tackle this challenge, this paper proposes a Siamese-enabled rapid and continuous trust evaluation framework (SRCTE) to facilitate effective task collaboration. First, the communication and computing resource attributes of the collaborator in a trusted state, along with historical collaboration data, are collected and represented using an attributed control flow graph (ACFG) that captures trust-related semantic information and serves as a reference for comparison with data collected during task execution. At each time slot of task execution, the collaborator's communication and computing resource attributes, as well as task completion effectiveness, are collected in real time and represented with an ACFG to convey their trust-related semantic information. A Siamese model, consisting of two shared-parameter Structure2vec networks, is then employed to learn the deep semantics of each pair of ACFGs and generate their embeddings. Finally, the similarity between the embeddings of each pair of ACFGs is calculated to determine the collaborator's trust value at each time slot. A real system is built using two Dell EMC 5200 servers and a Google Pixel 8 to test the effectiveness of the proposed SRCTE framework. Experimental results demonstrate that SRCTE converges rapidly with only a small amount of data and achieves a high anomaly trust detection rate compared to the baseline algorithm. 

**Abstract (ZH)**: 基于双路网络的快速连续信任评估框架(SRCTE)以促进有效协作任务的完成 

---
# MEXA: Towards General Multimodal Reasoning with Dynamic Multi-Expert Aggregation 

**Title (ZH)**: MEXA: 向泛化多模态推理的动态多专家聚合研究 

**Authors**: Shoubin Yu, Yue Zhang, Ziyang Wang, Jaehong Yoon, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2506.17113)  

**Abstract**: Combining pre-trained expert models offers substantial potential for scalable multimodal reasoning, but building a unified framework remains challenging due to the increasing diversity of input modalities and task complexity. For instance, medical diagnosis requires precise reasoning over structured clinical tables, while financial forecasting depends on interpreting plot-based data to make informed predictions. To tackle this challenge, we introduce MEXA, a training-free framework that performs modality- and task-aware aggregation of multiple expert models to enable effective multimodal reasoning across diverse and distinct domains. MEXA dynamically selects expert models based on the input modality and the task-specific reasoning demands (i.e., skills). Each expert model, specialized in a modality task pair, generates interpretable textual reasoning outputs. MEXA then aggregates and reasons over these outputs using a Large Reasoning Model (LRM) to produce the final answer. This modular design allows flexible and transparent multimodal reasoning across diverse domains without additional training overhead. We extensively evaluate our approach on diverse multimodal benchmarks, including Video Reasoning, Audio Reasoning, 3D Understanding, and Medical QA. MEXA consistently delivers performance improvements over strong multimodal baselines, highlighting the effectiveness and broad applicability of our expert-driven selection and aggregation in diverse multimodal reasoning tasks. 

**Abstract (ZH)**: 结合预训练专家模型在可扩展的多模态推理中具有巨大潜力，但由于输入模态和任务复杂性的不断增加，建立统一框架仍然具有挑战性。为应对这一挑战，我们引入了MEXA，这是一种无需训练的框架，能够在多种专家模型之间进行模态和任务意识聚合，以在不同且独特的领域中实现有效的多模态推理。MEXA根据输入模态和任务特定的推理需求（即技能）动态选择专家模型。每个专家模型专门处理特定的模态任务对，并生成可解释的文本推理输出。MEXA然后使用大型推理模型（LRM）聚合和推理这些输出以生成最终答案。这种模块化设计允许在不同领域中灵活透明地进行多模态推理，而无需额外的训练开销。我们在视频推理、音频推理、3D理解以及医疗问答等多个多模态基准上广泛评估了该方法。MEXA在多种多模态基准上持续提供性能改进，突出了专家驱动的选择和聚合在多种多模态推理任务中的有效性与普适性。 

---
# TransDreamerV3: Implanting Transformer In DreamerV3 

**Title (ZH)**: TransDreamerV3: 在DreamerV3中植入Transformer 

**Authors**: Shruti Sadanand Dongare, Amun Kharel, Jonathan Samuel, Xiaona Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17103)  

**Abstract**: This paper introduces TransDreamerV3, a reinforcement learning model that enhances the DreamerV3 architecture by integrating a transformer encoder. The model is designed to improve memory and decision-making capabilities in complex environments. We conducted experiments on Atari-Boxing, Atari-Freeway, Atari-Pong, and Crafter tasks, where TransDreamerV3 demonstrated improved performance over DreamerV3, particularly in the Atari-Freeway and Crafter tasks. While issues in the Minecraft task and limited training across all tasks were noted, TransDreamerV3 displays advancement in world model-based reinforcement learning, leveraging transformer architectures. 

**Abstract (ZH)**: TransDreamerV3：一种通过集成变压器编码器增强的DreamerV3架构的强化学习模型 

---
# Identifiability of Deep Polynomial Neural Networks 

**Title (ZH)**: 深度多项式神经网络的可 identifiability 

**Authors**: Konstantin Usevich, Clara Dérand, Ricardo Borsoi, Marianne Clausel  

**Link**: [PDF](https://arxiv.org/pdf/2506.17093)  

**Abstract**: Polynomial Neural Networks (PNNs) possess a rich algebraic and geometric structure. However, their identifiability -- a key property for ensuring interpretability -- remains poorly understood. In this work, we present a comprehensive analysis of the identifiability of deep PNNs, including architectures with and without bias terms. Our results reveal an intricate interplay between activation degrees and layer widths in achieving identifiability. As special cases, we show that architectures with non-increasing layer widths are generically identifiable under mild conditions, while encoder-decoder networks are identifiable when the decoder widths do not grow too rapidly. Our proofs are constructive and center on a connection between deep PNNs and low-rank tensor decompositions, and Kruskal-type uniqueness theorems. This yields both generic conditions determined by the architecture, and effective conditions that depend on the network's parameters. We also settle an open conjecture on the expected dimension of PNN's neurovarieties, and provide new bounds on the activation degrees required for it to reach its maximum. 

**Abstract (ZH)**: 多项式神经网络（PNNs）具有丰富的代数和几何结构。然而，它们的可识别性——这一确保可解释性的重要性质——至今尚未得到充分理解。在本文中，我们对深层PNNs的可识别性进行了全面分析，包括包含和不包含 bias 项的架构。我们的结果揭示了激活阶数和层宽之间实现可识别性的一种复杂的相互作用。作为特殊情况，我们展示了在轻微条件下，具有非递增层宽的架构是通用可识别的，而编码-解码网络在解码器宽度不快速增长时是可识别的。我们的证明是构造性的，并集中在深层PNNs与低秩张量分解之间的联系以及Kruskal型唯一性定理上。这既提供了由架构决定的通用条件，也提供了依赖于网络参数的有效条件。我们还解决了PNN神经簇体的预期维数的一个公开猜想，并提供了达到最大值所需的激活阶数的新界。 

---
# Tower+: Bridging Generality and Translation Specialization in Multilingual LLMs 

**Title (ZH)**: Tower+: 在多语言LLM中连接通用性和翻译专业化 

**Authors**: Ricardo Rei, Nuno M. Guerreiro, José Pombal, João Alves, Pedro Teixeirinha, Amin Farajian, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2506.17080)  

**Abstract**: Fine-tuning pretrained LLMs has been shown to be an effective strategy for reaching state-of-the-art performance on specific tasks like machine translation. However, this process of adaptation often implies sacrificing general-purpose capabilities, such as conversational reasoning and instruction-following, hampering the utility of the system in real-world applications that require a mixture of skills. In this paper, we introduce Tower+, a suite of models designed to deliver strong performance across both translation and multilingual general-purpose text capabilities. We achieve a Pareto frontier between translation specialization and multilingual general-purpose capabilities by introducing a novel training recipe that builds on Tower (Alves et al., 2024), comprising continued pretraining, supervised fine-tuning, preference optimization, and reinforcement learning with verifiable rewards. At each stage of training, we carefully generate and curate data to strengthen performance on translation as well as general-purpose tasks involving code generation, mathematics problem solving, and general instruction-following. We develop models at multiple scales: 2B, 9B, and 72B. Our smaller models often outperform larger general-purpose open-weight and proprietary LLMs (e.g., Llama 3.3 70B, GPT-4o). Our largest model delivers best-in-class translation performance for high-resource languages and top results in multilingual Arena Hard evaluations and in IF-MT, a benchmark we introduce for evaluating both translation and instruction-following. Our findings highlight that it is possible to rival frontier models in general capabilities, while optimizing for specific business domains, such as translation and localization. 

**Abstract (ZH)**: Fine-tuning 预训练大规模语言模型在特定任务（如机器翻译）上达到最佳性能已被证明是一种有效策略，但这一适应过程往往会牺牲通用能力，如对话推理和指令跟随，限制了其在需要多种技能的应用中的实用性。本文提出 Tower+，一个旨在同时实现强大翻译性能和多语言通用文本能力的模型系列。通过引入一种新的训练方案，该方案基于 Tower (Alves et al., 2024)，结合连续预训练、监督微调、偏好优化和带有可验证奖励的强化学习，我们实现了翻译专门化和多语言通用能力之间的帕累托前沿。在每一轮训练中，我们精心生成和筛选数据以强化翻译性能，同时也强化涉及代码生成、数学问题解决和通用指令跟随的一般任务性能。我们开发了多种规模的模型：2B、9B 和 72B。我们的较小模型经常优于较大的通用大型开放权重和专有语言模型（例如 Llama 3.3 70B、GPT-4o）。我们最大的模型在高资源语言翻译性能方面达到最佳水平，并在多语言 Arena Hard 评估和我们引入的 IF-MT 基准测试中取得了顶级结果，该基准测试用于评估翻译和指令跟随能力。我们的发现表明，在优化特定业务领域（如翻译和本地化）的同时，有可能与前沿模型在通用能力方面匹敌。 

---
# LLM-Based Bot Broadens the Range of Arguments in Online Discussions, Even When Transparently Disclosed as AI 

**Title (ZH)**: LLM-Based Bot 扩大了在线讨论中的论点范围，即使透明披露为AIassistant 

**Authors**: Valeria Vuk, Cristina Sarasua, Fabrizio Gilardi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17073)  

**Abstract**: A wide range of participation is essential for democracy, as it helps prevent the dominance of extreme views, erosion of legitimacy, and political polarization. However, engagement in online political discussions often features a limited spectrum of views due to high levels of self-selection and the tendency of online platforms to facilitate exchanges primarily among like-minded individuals. This study examines whether an LLM-based bot can widen the scope of perspectives expressed by participants in online discussions through two pre-registered randomized experiments conducted in a chatroom. We evaluate the impact of a bot that actively monitors discussions, identifies missing arguments, and introduces them into the conversation. The results indicate that our bot significantly expands the range of arguments, as measured by both objective and subjective metrics. Furthermore, disclosure of the bot as AI does not significantly alter these effects. These findings suggest that LLM-based moderation tools can positively influence online political discourse. 

**Abstract (ZH)**: 广泛的参与对于民主至关重要，它有助于防止极端观点的主导、合法性的侵蚀和政治极化。然而，在线政治讨论中的参与往往由于高度的选择性和在线平台促进观点一致者之间交流的倾向而局限于有限的观点范围。本研究通过在聊天室中进行的两项预先注册的随机实验，探讨基于大语言模型的机器人是否能够通过引入新的观点来扩大在线讨论中参与者表达的视角范围。我们评估了这种机器人在活跃监控讨论、识别缺失的论点并将其引入对话中的影响。结果显示，我们的机器人在客观和主观指标上显著扩展了论点的范围。此外，披露机器人是AI并不会显著改变这些效果。这些发现表明，基于大语言模型的 Moderation 工具可以积极影响在线政治 discourse。 

---
# Flow-Based Non-stationary Temporal Regime Causal Structure Learning 

**Title (ZH)**: 基于流的方法非平稳时间动态度量因果结构学习 

**Authors**: Abdellah Rahmani, Pascal Frossard  

**Link**: [PDF](https://arxiv.org/pdf/2506.17065)  

**Abstract**: Understanding causal relationships in multivariate time series is crucial in many scenarios, such as those dealing with financial or neurological data. Many such time series exhibit multiple regimes, i.e., consecutive temporal segments with a priori unknown boundaries, with each regime having its own causal structure. Inferring causal dependencies and regime shifts is critical for analyzing the underlying processes. However, causal structure learning in this setting is challenging due to (1) non stationarity, i.e., each regime can have its own causal graph and mixing function, and (2) complex noise distributions, which may be non Gaussian or heteroscedastic. Existing causal discovery approaches cannot address these challenges, since generally assume stationarity or Gaussian noise with constant variance. Hence, we introduce FANTOM, a unified framework for causal discovery that handles non stationary processes along with non Gaussian and heteroscedastic noises. FANTOM simultaneously infers the number of regimes and their corresponding indices and learns each regime's Directed Acyclic Graph. It uses a Bayesian Expectation Maximization algorithm that maximizes the evidence lower bound of the data log likelihood. On the theoretical side, we prove, under mild assumptions, that temporal heteroscedastic causal models, introduced in FANTOM's formulation, are identifiable in both stationary and non stationary settings. In addition, extensive experiments on synthetic and real data show that FANTOM outperforms existing methods. 

**Abstract (ZH)**: 理解多变量时间序列中的因果关系在许多场景中至关重要，例如金融或神经数据领域。这类时间序列常常包含多个制度，即具有先前未知边界的连续时间段，每个制度具有自己的因果结构。推断因果依赖性和制度转换对于分析潜在过程至关重要。然而，在这种场景下的因果结构学习具有挑战性，原因在于（1）非平稳性，即每个制度可以有自己的因果图和混合函数，以及（2）复杂的噪声分布，可能会是非正态分布或异方差的。现有的因果发现方法无法解决这些挑战，因为它们通常假定平稳性或具有恒定方差的高斯噪声。因此，我们引入了FANTOM，这是一种统一的框架，可以处理非平稳过程以及非正态和异方差的噪声。FANTOM同时推断制度的数量及其相应的索引，并学习每个制度的有向无环图。它使用最大化数据对数似然证据下界的贝叶斯期望最大化算法。在理论上，我们证明，在温和假设下，FANTOM公式中引入的时间异方差因果模型在平稳和非平稳设置中是可辨识的。此外，对合成数据和真实数据的广泛实验表明，FANTOM优于现有方法。 

---
# From Concepts to Components: Concept-Agnostic Attention Module Discovery in Transformers 

**Title (ZH)**: 从概念到组件：Transformer中的概念无关注意力模块发现 

**Authors**: Jingtong Su, Julia Kempe, Karen Ullrich  

**Link**: [PDF](https://arxiv.org/pdf/2506.17052)  

**Abstract**: Transformers have achieved state-of-the-art performance across language and vision tasks. This success drives the imperative to interpret their internal mechanisms with the dual goals of enhancing performance and improving behavioral control. Attribution methods help advance interpretability by assigning model outputs associated with a target concept to specific model components. Current attribution research primarily studies multi-layer perceptron neurons and addresses relatively simple concepts such as factual associations (e.g., Paris is located in France). This focus tends to overlook the impact of the attention mechanism and lacks a unified approach for analyzing more complex concepts. To fill these gaps, we introduce Scalable Attention Module Discovery (SAMD), a concept-agnostic method for mapping arbitrary, complex concepts to specific attention heads of general transformer models. We accomplish this by representing each concept as a vector, calculating its cosine similarity with each attention head, and selecting the TopK-scoring heads to construct the concept-associated attention module. We then propose Scalar Attention Module Intervention (SAMI), a simple strategy to diminish or amplify the effects of a concept by adjusting the attention module using only a single scalar parameter. Empirically, we demonstrate SAMD on concepts of varying complexity, and visualize the locations of their corresponding modules. Our results demonstrate that module locations remain stable before and after LLM post-training, and confirm prior work on the mechanics of LLM multilingualism. Through SAMI, we facilitate jailbreaking on HarmBench (+72.7%) by diminishing "safety" and improve performance on the GSM8K benchmark (+1.6%) by amplifying "reasoning". Lastly, we highlight the domain-agnostic nature of our approach by suppressing the image classification accuracy of vision transformers on ImageNet. 

**Abstract (ZH)**: Transformer在语言和视觉任务中取得了最先进的性能。这种成功推动了对其内部机制进行解释的必要性，目标是在提升性能的同时改善行为控制。归因方法通过将模型输出与特定模型组件关联起来，有助于推进解释性。当前的归因研究主要集中在多层感知器神经元上，并且侧重于研究简单的概念，如事实关联（例如，巴黎位于法国）。这种侧重往往忽视了注意力机制的影响，并且缺乏统一的方法来分析更复杂的概念。为填补这些空白，我们介绍了可扩展的注意力模块发现（SAMD），这是一种概念无关的方法，用于将任意复杂概念映射到通用Transformer模型的特定注意力头。我们通过将每个概念表示为向量，计算其与每个注意力头的余弦相似度，并选择得分最高的TopK头来构建概念相关的注意力模块。然后，我们提出了标量注意力模块干预（SAMI）策略，通过仅使用单一标量参数调整注意力模块，来减弱或放大概念的效果。实证研究中，我们在不同复杂度的概念上展示了SAMD，并可视化了它们对应的模块位置。实验结果表明，在大规模语言模型后训练前后，模块位置保持稳定，并且证实了先前关于大规模语言模型多语言性的研究工作。通过SAMI，我们通过减弱“安全性”在HarmBench上实现了+72.7%的提升，并通过增强“推理”在GSM8K基准上实现了+1.6%的性能提升。最后，我们强调了我们方法的领域无关性，通过抑制视觉Transformer在ImageNet上的图像分类准确性来突显这一点。 

---
# MAWIFlow Benchmark: Realistic Flow-Based Evaluation for Network Intrusion Detection 

**Title (ZH)**: MAWIFlow基准：基于流的网络入侵检测的现实评价 

**Authors**: Joshua Schraven, Alexander Windmann, Oliver Niggemann  

**Link**: [PDF](https://arxiv.org/pdf/2506.17041)  

**Abstract**: Benchmark datasets for network intrusion detection commonly rely on synthetically generated traffic, which fails to reflect the statistical variability and temporal drift encountered in operational environments. This paper introduces MAWIFlow, a flow-based benchmark derived from the MAWILAB v1.1 dataset, designed to enable realistic and reproducible evaluation of anomaly detection methods. A reproducible preprocessing pipeline is presented that transforms raw packet captures into flow representations conforming to the CICFlowMeter format, while preserving MAWILab's original anomaly labels. The resulting datasets comprise temporally distinct samples from January 2011, 2016, and 2021, drawn from trans-Pacific backbone traffic.
To establish reference baselines, traditional machine learning methods, including Decision Trees, Random Forests, XGBoost, and Logistic Regression, are compared to a deep learning model based on a CNN-BiLSTM architecture. Empirical results demonstrate that tree-based classifiers perform well on temporally static data but experience significant performance degradation over time. In contrast, the CNN-BiLSTM model maintains better performance, thus showing improved generalization. These findings underscore the limitations of synthetic benchmarks and static models, and motivate the adoption of realistic datasets with explicit temporal structure. All datasets, pipeline code, and model implementations are made publicly available to foster transparency and reproducibility. 

**Abstract (ZH)**: 基于MAWILAB v1.1数据集的MAWIFlow流基准：一种用于异常检测方法的现实和可重复评估的数据集 

---
# LSCD: Lomb-Scargle Conditioned Diffusion for Time series Imputation 

**Title (ZH)**: LSCD: Lomb-Scargle条件下的扩散时间序列插补 

**Authors**: Elizabeth Fons, Alejandro Sztrajman, Yousef El-Laham, Luciana Ferrer, Svitlana Vyetrenko, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2506.17039)  

**Abstract**: Time series with missing or irregularly sampled data are a persistent challenge in machine learning. Many methods operate on the frequency-domain, relying on the Fast Fourier Transform (FFT) which assumes uniform sampling, therefore requiring prior interpolation that can distort the spectra. To address this limitation, we introduce a differentiable Lomb--Scargle layer that enables a reliable computation of the power spectrum of irregularly sampled data. We integrate this layer into a novel score-based diffusion model (LSCD) for time series imputation conditioned on the entire signal spectrum. Experiments on synthetic and real-world benchmarks demonstrate that our method recovers missing data more accurately than purely time-domain baselines, while simultaneously producing consistent frequency estimates. Crucially, our method can be easily integrated into learning frameworks, enabling broader adoption of spectral guidance in machine learning approaches involving incomplete or irregular data. 

**Abstract (ZH)**: 缺失或不规则采样时间序列是机器学习中的一个持久挑战。许多方法依赖于傅里叶频域，依赖于快速傅里叶变换（FFT），该变换假设均匀采样，因此需要先行内插，这可能会扭曲频谱。为解决这一局限，我们引入了一个可微的Lomb--Scargle层，使得不规则采样数据的功率谱计算更加可靠。我们将这一层整合到一种新型基于得分的扩散模型（LSCD）中，用于整个信号频谱条件下的时间序列插补。在合成和真实世界基准上的实验表明，我们的方法比单纯的时域基线更准确地恢复了缺失数据，同时生成一致的频率估计。尤为重要的是，我们的方法可以轻松地集成到学习框架中，从而在涉及不完整或不规则数据的机器学习方法中更广泛地采用频谱指导。 

---
# Instituto de Telecomunicações at IWSLT 2025: Aligning Small-Scale Speech and Language Models for Speech-to-Text Learning 

**Title (ZH)**: Instituto de Telecomunicações 在 IWSLT 2025：优化小型语音和语言模型实现语音转文本学习 

**Authors**: Giuseppe Attanasio, Sonal Sannigrahi, Ben Peters, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2506.17019)  

**Abstract**: This paper presents the IT-IST submission to the IWSLT 2025 Shared Task on Instruction Following Speech Processing. We submit results for the Short Track, i.e., speech recognition, translation, and spoken question answering. Our model is a unified speech-to-text model that integrates a pre-trained continuous speech encoder and text decoder through a first phase of modality alignment and a second phase of instruction fine-tuning. Crucially, we focus on using small-scale language model backbones (< 2B) and restrict to high-quality, CC-BY data along with synthetic data generation to supplement existing resources. 

**Abstract (ZH)**: 本文介绍了我们提交给2025年IWSLT共享任务的IT-IST参赛作品，该任务聚焦于指令跟随语音处理。我们提交了短赛道的结果，即语音识别、翻译和语音问答。我们的模型是一种统一的从语音到文本的模型，通过两个阶段——模态对齐和指令微调——将预训练的连续语音编码器和文本解码器进行整合。 crucial的是，我们专注于使用小规模的语言模型骨干网（<2B）并在高质CC-BY数据的基础上结合合成数据生成以补充现有资源。 

---
# TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs 

**Title (ZH)**: TeXpert：评价LLM生成LaTeX代码能力的多层级基准 

**Authors**: Sahil Kale, Vijaykant Nadadur  

**Link**: [PDF](https://arxiv.org/pdf/2506.16990)  

**Abstract**: LaTeX's precision and flexibility in typesetting have made it the gold standard for the preparation of scientific documentation. Large Language Models (LLMs) present a promising opportunity for researchers to produce publication-ready material using LaTeX with natural language instructions, yet current benchmarks completely lack evaluation of this ability. By introducing TeXpert, our benchmark dataset with natural language prompts for generating LaTeX code focused on components of scientific documents across multiple difficulty levels, we conduct an in-depth analysis of LLM performance in this regard and identify frequent error types. Our evaluation across open and closed-source LLMs highlights multiple key findings: LLMs excelling on standard benchmarks perform poorly in LaTeX generation with a significant accuracy drop-off as the complexity of tasks increases; open-source models like DeepSeek v3 and DeepSeek Coder strongly rival closed-source counterparts in LaTeX tasks; and formatting and package errors are unexpectedly prevalent, suggesting a lack of diverse LaTeX examples in the training datasets of most LLMs. Our dataset, code, and model evaluations are available at this https URL. 

**Abstract (ZH)**: LaTeX排版的精确性和灵活性使其成为科学文档准备的黄金标准。大规模语言模型（LLMs）为研究人员提供了使用自然语言指令生成符合出版要求的LaTeX文档的 promising 机会，然而目前的基准测试完全缺乏对这一能力的评估。通过引入TeXpert，一个基于自然语言提示生成LaTeX代码的基准数据集，专注于科学文档的不同组件并涵盖多种难度级别，我们对LLMs在此方面的性能进行了深入分析，并识别出常见的错误类型。我们的评估结果显示，表现优异的LLMs在LaTeX生成任务中的表现不佳，随着任务复杂性的增加，准确率急剧下降；开源模型如DeepSeek v3和DeepSeek Coder在LaTeX任务中与闭源模型有强烈竞争；格式错误和包错误出乎意料地普遍，表明大多数LLMs的训练数据集中缺乏多样化的LaTeX示例。我们的数据集、代码和模型评估可在以下链接获取。 

---
# Language Bottleneck Models: A Framework for Interpretable Knowledge Tracing and Beyond 

**Title (ZH)**: 语言瓶颈模型：可解释的知识追踪及更广泛的框架 

**Authors**: Antonin Berthon, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.16982)  

**Abstract**: Accurately assessing student knowledge is critical for effective education, yet traditional Knowledge Tracing (KT) methods rely on opaque latent embeddings, limiting interpretability. Even LLM-based approaches generate direct predictions or summaries that may hallucinate without any accuracy guarantees. We recast KT as an inverse problem: learning the minimum natural-language summary that makes past answers explainable and future answers predictable. Our Language Bottleneck Model (LBM) consists of an encoder LLM that writes an interpretable knowledge summary and a frozen decoder LLM that must reconstruct and predict student responses using only that summary text. By constraining all predictive information to pass through a short natural-language bottleneck, LBMs ensure that the summary contains accurate information while remaining human-interpretable. Experiments on synthetic arithmetic benchmarks and the large-scale Eedi dataset show that LBMs rival the accuracy of state-of-the-art KT and direct LLM methods while requiring orders-of-magnitude fewer student trajectories. We demonstrate that training the encoder with group-relative policy optimization, using downstream decoding accuracy as a reward signal, effectively improves summary quality. 

**Abstract (ZH)**: 准确评估学生知识对于有效的教育至关重要，但传统的知识追踪（KT）方法依赖于不透明的潜变量嵌入，限制了其可解释性。即使是基于大语言模型（LLM）的方法也会生成直接的预测或摘要，这些摘要可能会无中生有，没有任何准确性的保证。我们将KT重新定义为一个逆问题：学习一个最小的自然语言摘要，使其能够解释过去的答案并预测未来的答案。我们的语言瓶颈模型（LBM）由一个编写可解释知识摘要的编码器LLM和一个冻结的解码器LLM组成，后者必须仅使用摘要文本重构和预测学生的回答。通过将所有预测信息强制通过一个短的自然语言瓶颈，LBM确保摘要包含准确的信息同时保持人类可解释性。在合成算术基准测试和大规模Eedi数据集上的实验表明，LBM在准确性和最新的KT以及直接LLM方法相当的同时，需要的学生轨迹数量级更少。我们证明，使用组相对策略优化训练编码器，并将下游解码准确性作为奖励信号，可以有效提高摘要的质量。 

---
# Latent Concept Disentanglement in Transformer-based Language Models 

**Title (ZH)**: 基于变换器的语言模型中潜概念去缠绕 

**Authors**: Guan Zhe Hong, Bhavya Vasudeva, Vatsal Sharan, Cyrus Rashtchian, Prabhakar Raghavan, Rina Panigrahy  

**Link**: [PDF](https://arxiv.org/pdf/2506.16975)  

**Abstract**: When large language models (LLMs) use in-context learning (ICL) to solve a new task, they seem to grasp not only the goal of the task but also core, latent concepts in the demonstration examples. This begs the question of whether transformers represent latent structures as part of their computation or whether they take shortcuts to solve the problem. Prior mechanistic work on ICL does not address this question because it does not sufficiently examine the relationship between the learned representation and the latent concept, and the considered problem settings often involve only single-step reasoning. In this work, we examine how transformers disentangle and use latent concepts. We show that in 2-hop reasoning tasks with a latent, discrete concept, the model successfully identifies the latent concept and does step-by-step concept composition. In tasks parameterized by a continuous latent concept, we find low-dimensional subspaces in the representation space where the geometry mimics the underlying parameterization. Together, these results refine our understanding of ICL and the representation of transformers, and they provide evidence for highly localized structures in the model that disentangle latent concepts in ICL tasks. 

**Abstract (ZH)**: 当大型语言模型（LLMs）使用上下文学习（ICL）解决新任务时，它们似乎不仅掌握了任务目标，还掌握了示范例子中的核心潜在概念。这引发了关于变换器是否在其计算过程中表示潜在结构，还是采取捷径来解决问题的疑问。先前关于ICL的机制研究未能回答这一问题，因为它们未能充分探讨所学表示与潜在概念之间的关系，而且考虑的问题场景往往只涉及一步推理。在本工作中，我们探讨了变换器如何分离和利用潜在概念。我们展示了在涉及潜在离散概念的2跳推理任务中，模型成功识别了潜在概念并进行了逐步的概念组合。在涉及连续潜在概念的任务中，我们发现表示空间中的低维子空间，其几何结构模仿了潜在参数化。这些结果进一步细化了我们对ICL和变换器表示的理解，并提供了模型中高度局部化的结构证据，这些结构在ICL任务中分离潜在概念。 

---
# Formal Control for Uncertain Systems via Contract-Based Probabilistic Surrogates (Extended Version) 

**Title (ZH)**: 基于合同的概率代理模型的不确定性系统形式控制（扩展版） 

**Authors**: Oliver Schön, Sofie Haesaert, Sadegh Soudjani  

**Link**: [PDF](https://arxiv.org/pdf/2506.16971)  

**Abstract**: The requirement for identifying accurate system representations has not only been a challenge to fulfill, but it has compromised the scalability of formal methods, as the resulting models are often too complex for effective decision making with formal correctness and performance guarantees. Focusing on probabilistic simulation relations and surrogate models of stochastic systems, we propose an approach that significantly enhances the scalability and practical applicability of such simulation relations by eliminating the need to compute error bounds directly. As a result, we provide an abstraction-based technique that scales effectively to higher dimensions while addressing complex nonlinear agent-environment interactions with infinite-horizon temporal logic guarantees amidst uncertainty. Our approach trades scalability for conservatism favorably, as demonstrated on a complex high-dimensional vehicle intersection case study. 

**Abstract (ZH)**: 识别准确系统表示的要求不仅是一项挑战，还削弱了形式方法的扩展性，因为生成的模型往往过于复杂，无法确保有效的决策制定并保证性能。针对随机系统的概率仿真关系和代理模型，我们提出了一种方法，通过消除直接计算误差界的需求，显著增强了此类仿真关系的扩展性和实用性。由此，我们提供了一种基于抽象的技术，可以在高维空间中有效扩展，并在不确定性下利用无限_horizon 时间逻辑保证处理复杂的非线性代理-环境交互。我们的方法在可扩展性和保守性之间进行了有利的权衡，如在复杂高维车辆交叉口案例研究中所示。 

---
# Enhancing Step-by-Step and Verifiable Medical Reasoning in MLLMs 

**Title (ZH)**: 增强MLLMs的逐步可验证医疗推理能力 

**Authors**: Haoran Sun, Yankai Jiang, Wenjie Lou, Yujie Zhang, Wenjie Li, Lilong Wang, Mianxin Liu, Lei Liu, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16962)  

**Abstract**: Multimodal large language models (MLLMs) have begun to demonstrate robust reasoning capabilities on general tasks, yet their application in the medical domain remains in its early stages. Constructing chain-of-thought (CoT) training data is essential for bolstering the reasoning abilities of medical MLLMs. However, existing approaches exhibit a deficiency in offering a comprehensive framework for searching and evaluating effective reasoning paths towards critical diagnosis. To address this challenge, we propose Mentor-Intern Collaborative Search (MICS), a novel reasoning-path searching scheme to generate rigorous and effective medical CoT data. MICS first leverages mentor models to initialize the reasoning, one step at a time, then prompts each intern model to continue the thinking along those initiated paths, and finally selects the optimal reasoning path according to the overall reasoning performance of multiple intern models. The reasoning performance is determined by an MICS-Score, which assesses the quality of generated reasoning paths. Eventually, we construct MMRP, a multi-task medical reasoning dataset with ranked difficulty, and Chiron-o1, a new medical MLLM devised via a curriculum learning strategy, with robust visual question-answering and generalizable reasoning capabilities. Extensive experiments demonstrate that Chiron-o1, trained on our CoT dataset constructed using MICS, achieves state-of-the-art performance across a list of medical visual question answering and reasoning benchmarks. Codes are available at GitHub - manglu097/Chiron-o1: Enhancing Step-by-Step and Verifiable Medical Reasoning in MLLMs 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在一般任务上已经展现出稳健的推理能力，但在医疗领域中的应用仍处于起步阶段。构建链式思考（CoT）训练数据对于增强医疗MLLMs的推理能力至关重要。然而，现有方法在提供全面框架以搜索和评估通往关键诊断的有效推理路径方面存在不足。为解决这一挑战，我们提出了一种名为Mentor-Intern Collaborative Search（MICS）的新颖推理路径搜索方案，以生成严格且有效的医疗CoT数据。MICS首先利用导师模型逐步初始化推理，然后提示每个实习生模型沿着这些初始路径继续思考，并最终根据多个实习生模型的整体推理性能选择最优推理路径。推理性能由MICS-Score评估，该指标评估生成的推理路径的质量。最后，我们构建了包含分级难度的多任务医疗推理数据集MMRP，并通过阶梯学习策略开发了新的医疗MLLM Chiron-o1，该模型具备稳健的视觉问答和泛化推理能力。广泛实验表明，Chiron-o1在使用MICS构建的CoT数据集进行训练后，在一系列医疗视觉问答和推理基准测试中均实现了最先进的性能。代码可在GitHub - manglu097/Chiron-o1: 提升MLLM中逐步且可验证的医疗推理获得。 

---
# A deep learning and machine learning approach to predict neonatal death in the context of São Paulo 

**Title (ZH)**: 基于 São Paulo 情境下深度学习和机器学习预测新生儿死亡的方法 

**Authors**: Mohon Raihan, Plabon Kumar Saha, Rajan Das Gupta, A Z M Tahmidul Kabir, Afia Anjum Tamanna, Md. Harun-Ur-Rashid, Adnan Bin Abdus Salam, Md Tanvir Anjum, A Z M Ahteshamul Kabir  

**Link**: [PDF](https://arxiv.org/pdf/2506.16929)  

**Abstract**: Neonatal death is still a concerning reality for underdeveloped and even some developed countries. Worldwide data indicate that 26.693 babies out of 1,000 births die, according to Macro Trades. To reduce this number, early prediction of endangered babies is crucial. Such prediction enables the opportunity to take ample care of the child and mother so that early child death can be avoided. In this context, machine learning was used to determine whether a newborn baby is at risk. To train the predictive model, historical data of 1.4 million newborns was used. Machine learning and deep learning techniques such as logical regression, K-nearest neighbor, random forest classifier, extreme gradient boosting (XGBoost), convolutional neural network, and long short-term memory (LSTM) were implemented using the dataset to identify the most accurate model for predicting neonatal mortality. Among the machine learning algorithms, XGBoost and random forest classifier achieved the best accuracy with 94%, while among the deep learning models, LSTM delivered the highest accuracy with 99%. Therefore, using LSTM appears to be the most suitable approach to predict whether precautionary measures for a child are necessary. 

**Abstract (ZH)**: 新生儿死亡仍然是发展中国家乃至部分发达国家值得关注的现实问题。根据Macro Trades的数据，全球范围内每1000名新生儿中就有26.693名婴儿死亡。为了减少这一数字，提前预测处于危险中的婴儿至关重要。这样的预测可以为婴儿和母亲提供充足的照料机会，从而避免早产儿死亡。在此背景下，机器学习被用于判断新生儿是否处于风险中。为了训练预测模型，使用了140万名新生儿的历史数据。通过逻辑回归、K最近邻、随机森林分类器、极端梯度提升（XGBoost）、卷积神经网络和长短期记忆（LSTM）等机器学习和深度学习技术实施了实验，以识别出最准确的预测新生儿死亡率的模型。在机器学习算法中，XGBoost和随机森林分类器的准确率最高，达到94%，而在深度学习模型中，LSTM的准确率最高，达到99%。因此，使用LSTM似乎是预测是否需要采取预防措施的最佳方法。 

---
# Single-shot thermometry of simulated Bose--Einstein condensates using artificial intelligence 

**Title (ZH)**: 使用人工智能进行模拟玻色-爱因斯坦凝聚态的一次成像温度测量 

**Authors**: Jack Griffiths, Steven A. Wrathmall, Simon A. Gardiner  

**Link**: [PDF](https://arxiv.org/pdf/2506.16925)  

**Abstract**: Precise determination of thermodynamic parameters in ultracold Bose gases remains challenging due to the destructive nature of conventional measurement techniques and inherent experimental uncertainties. We demonstrate an artificial intelligence approach for rapid, non-destructive estimation of the chemical potential and temperature from single-shot, in situ imaged density profiles of finite-temperature Bose gases. Our convolutional neural network is trained exclusively on quasi-2D `pancake' condensates in harmonic trap configurations. It achieves parameter extraction within fractions of a second. The model also demonstrates zero-shot generalisation across both trap geometry and thermalisation dynamics, successfully estimating thermodynamic parameters for toroidally trapped condensates with errors of only a few nanokelvin despite no prior exposure to such geometries during training, and maintaining predictive accuracy during dynamic thermalisation processes after a relatively brief evolution without explicit training on non-equilibrium states. These results suggest that supervised learning can overcome traditional limitations in ultracold atom thermometry, with extension to broader geometric configurations, temperature ranges, and additional parameters potentially enabling comprehensive real-time analysis of quantum gas experiments. Such capabilities could significantly streamline experimental workflows whilst improving measurement precision across a range of quantum fluid systems. 

**Abstract (ZH)**: 利用人工神经网络实现有限温度玻色气体化学势和温度的快速无损估计：超越传统超冷原子 thermometry 的限制 

---
# Towards Effective Complementary Security Analysis using Large Language Models 

**Title (ZH)**: 面向有效互补安全分析的大语言模型方法 

**Authors**: Jonas Wagner, Simon Müller, Christian Näther, Jan-Philipp Steghöfer, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2506.16899)  

**Abstract**: A key challenge in security analysis is the manual evaluation of potential security weaknesses generated by static application security testing (SAST) tools. Numerous false positives (FPs) in these reports reduce the effectiveness of security analysis. We propose using Large Language Models (LLMs) to improve the assessment of SAST findings. We investigate the ability of LLMs to reduce FPs while trying to maintain a perfect true positive rate, using datasets extracted from the OWASP Benchmark (v1.2) and a real-world software project. Our results indicate that advanced prompting techniques, such as Chain-of-Thought and Self-Consistency, substantially improve FP detection. Notably, some LLMs identified approximately 62.5% of FPs in the OWASP Benchmark dataset without missing genuine weaknesses. Combining detections from different LLMs would increase this FP detection to approximately 78.9%. Additionally, we demonstrate our approach's generalizability using a real-world dataset covering five SAST tools, three programming languages, and infrastructure files. The best LLM detected 33.85% of all FPs without missing genuine weaknesses, while combining detections from different LLMs would increase this detection to 38.46%. Our findings highlight the potential of LLMs to complement traditional SAST tools, enhancing automation and reducing resources spent addressing false alarms. 

**Abstract (ZH)**: 使用大型语言模型提高静态应用程序安全性测试结果评估的挑战与研究 

---
# With Limited Data for Multimodal Alignment, Let the STRUCTURE Guide You 

**Title (ZH)**: 基于有限数据的多模态对齐，让STRUCTURE引领你 

**Authors**: Fabian Gröger, Shuo Wen, Huyen Le, Maria Brbić  

**Link**: [PDF](https://arxiv.org/pdf/2506.16895)  

**Abstract**: Multimodal models have demonstrated powerful capabilities in complex tasks requiring multimodal alignment including zero-shot classification and cross-modal retrieval. However, existing models typically rely on millions of paired multimodal samples, which are prohibitively expensive or infeasible to obtain in many domains. In this work, we explore the feasibility of building multimodal models with limited amount of paired data by aligning pretrained unimodal foundation models. We show that high-quality alignment is possible with as few as tens of thousands of paired samples$\unicode{x2013}$less than $1\%$ of the data typically used in the field. To achieve this, we introduce STRUCTURE, an effective regularization technique that preserves the neighborhood geometry of the latent space of unimodal encoders. Additionally, we show that aligning last layers is often suboptimal and demonstrate the benefits of aligning the layers with the highest representational similarity across modalities. These two components can be readily incorporated into existing alignment methods, yielding substantial gains across 24 zero-shot image classification and retrieval benchmarks, with average relative improvement of $51.6\%$ in classification and $91.8\%$ in retrieval tasks. Our results highlight the effectiveness and broad applicability of our framework for limited-sample multimodal learning and offer a promising path forward for resource-constrained domains. 

**Abstract (ZH)**: 基于有限配对数据构建多模态模型：结构化正则化方法及其应用 

---
# The Importance of Being Lazy: Scaling Limits of Continual Learning 

**Title (ZH)**: 懒有所值：连续学习的缩放极限 

**Authors**: Jacopo Graldi, Alessandro Breccia, Giulia Lanzillotta, Thomas Hofmann, Lorenzo Noci  

**Link**: [PDF](https://arxiv.org/pdf/2506.16884)  

**Abstract**: Despite recent efforts, neural networks still struggle to learn in non-stationary environments, and our understanding of catastrophic forgetting (CF) is far from complete. In this work, we perform a systematic study on the impact of model scale and the degree of feature learning in continual learning. We reconcile existing contradictory observations on scale in the literature, by differentiating between lazy and rich training regimes through a variable parameterization of the architecture. We show that increasing model width is only beneficial when it reduces the amount of feature learning, yielding more laziness. Using the framework of dynamical mean field theory, we then study the infinite width dynamics of the model in the feature learning regime and characterize CF, extending prior theoretical results limited to the lazy regime. We study the intricate relationship between feature learning, task non-stationarity, and forgetting, finding that high feature learning is only beneficial with highly similar tasks. We identify a transition modulated by task similarity where the model exits an effectively lazy regime with low forgetting to enter a rich regime with significant forgetting. Finally, our findings reveal that neural networks achieve optimal performance at a critical level of feature learning, which depends on task non-stationarity and transfers across model scales. This work provides a unified perspective on the role of scale and feature learning in continual learning. 

**Abstract (ZH)**: 尽管最近做出了努力，神经网络在非平稳环境中仍然难以学习，我们对灾难性遗忘（CF）的理解也远未完善。在本文中，我们系统研究了模型规模和持续学习中特征学习程度的影响。通过架构的可变参数化区分懒惰和丰富的训练制度，我们 reconciled 文献中存在的矛盾观察结果。我们利用框架理论研究特征学习状态下模型的无限宽度动态，并表征 CF，扩大了仅限懒惰状态下成立的先前理论结果。我们研究了特征学习、任务非平稳性和遗忘之间的复杂关系，发现只有在任务高度相似时，高特征学习才是有益的。我们确定了一种由任务相似性调节的过渡，在这种过渡中，模型从一个低遗忘的有效懒惰状态下退出，进入一个显著遗忘的丰富状态下。最后，我们的发现表明，神经网络在特征学习的临界水平上实现最佳性能，这取决于任务非平稳性和模型规模之间的转移。本文为模型规模和特征学习在持续学习中的作用提供了统一视角。 

---
# ParkFormer: A Transformer-Based Parking Policy with Goal Embedding and Pedestrian-Aware Control 

**Title (ZH)**: ParkFormer：基于目标嵌入和行人感知控制的变压器停车策略 

**Authors**: Jun Fu, Bin Tian, Haonan Chen, Shi Meng, Tingting Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16856)  

**Abstract**: Autonomous parking plays a vital role in intelligent vehicle systems, particularly in constrained urban environments where high-precision control is required. While traditional rule-based parking systems struggle with environmental uncertainties and lack adaptability in crowded or dynamic scenes, human drivers demonstrate the ability to park intuitively without explicit modeling. Inspired by this observation, we propose a Transformer-based end-to-end framework for autonomous parking that learns from expert demonstrations. The network takes as input surround-view camera images, goal-point representations, ego vehicle motion, and pedestrian trajectories. It outputs discrete control sequences including throttle, braking, steering, and gear selection. A novel cross-attention module integrates BEV features with target points, and a GRU-based pedestrian predictor enhances safety by modeling dynamic obstacles. We validate our method on the CARLA 0.9.14 simulator in both vertical and parallel parking scenarios. Experiments show our model achieves a high success rate of 96.57\%, with average positional and orientation errors of 0.21 meters and 0.41 degrees, respectively. The ablation studies further demonstrate the effectiveness of key modules such as pedestrian prediction and goal-point attention fusion. The code and dataset will be released at: this https URL. 

**Abstract (ZH)**: 自主泊车在智能车辆系统中扮演着重要角色，特别是在受限制的都市环境中，需要高精度控制。虽然传统的基于规则的泊车系统难以应对环境不确定性并缺乏在拥挤或动态场景中的适应性，但人类驾驶员能够直观地泊车而无需显式的建模。受此启发，我们提出了一种基于Transformer的端到端自主泊车框架，该框架从专家示范中学习。网络将全景摄像头图像、目标点表示、ego车辆运动和行人的轨迹作为输入，输出包括油门、刹车、转向和换挡的离散控制序列。一个新颖的交叉注意力模块将BEV特征与目标点集成，基于GRU的行人预测器通过建模动态障碍物来增强安全性。我们在CARLA 0.9.14模拟器上对垂直泊车和平行泊车场景进行了实验验证，结果显示我们的模型的成功率为96.57%，平均位置和方向误差分别为0.21米和0.41度。进一步的消融研究展示了行人预测和目标点注意力融合等关键模块的有效性。相关代码和数据集将在以下链接中发布：this https URL。 

---
# Bandwidth Selectors on Semiparametric Bayesian Networks 

**Title (ZH)**: 半参数贝叶斯网络中的带宽选择器 

**Authors**: Victor Alejandre, Concha Bielza, Pedro Larrañaga  

**Link**: [PDF](https://arxiv.org/pdf/2506.16844)  

**Abstract**: Semiparametric Bayesian networks (SPBNs) integrate parametric and non-parametric probabilistic models, offering flexibility in learning complex data distributions from samples. In particular, kernel density estimators (KDEs) are employed for the non-parametric component. Under the assumption of data normality, the normal rule is used to learn the bandwidth matrix for the KDEs in SPBNs. This matrix is the key hyperparameter that controls the trade-off between bias and variance. However, real-world data often deviates from normality, potentially leading to suboptimal density estimation and reduced predictive performance. This paper first establishes the theoretical framework for the application of state-of-the-art bandwidth selectors and subsequently evaluates their impact on SPBN performance. We explore the approaches of cross-validation and plug-in selectors, assessing their effectiveness in enhancing the learning capability and applicability of SPBNs. To support this investigation, we have extended the open-source package PyBNesian for SPBNs with the additional bandwidth selection techniques and conducted extensive experimental analyses. Our results demonstrate that the proposed bandwidth selectors leverage increasing information more effectively than the normal rule, which, despite its robustness, stagnates with more data. In particular, unbiased cross-validation generally outperforms the normal rule, highlighting its advantage in high sample size scenarios. 

**Abstract (ZH)**: 半参数贝叶斯网络的带宽选择：理论框架与实证分析 

---
# AnyTraverse: An off-road traversability framework with VLM and human operator in the loop 

**Title (ZH)**: AnyTraverse: 一种结合VLM和人工操作者的离路穿越框架 

**Authors**: Sattwik Sahu, Agamdeep Singh, Karthik Nambiar, Srikanth Saripalli, P.B. Sujit  

**Link**: [PDF](https://arxiv.org/pdf/2506.16826)  

**Abstract**: Off-road traversability segmentation enables autonomous navigation with applications in search-and-rescue, military operations, wildlife exploration, and agriculture. Current frameworks struggle due to significant variations in unstructured environments and uncertain scene changes, and are not adaptive to be used for different robot types. We present AnyTraverse, a framework combining natural language-based prompts with human-operator assistance to determine navigable regions for diverse robotic vehicles. The system segments scenes for a given set of prompts and calls the operator only when encountering previously unexplored scenery or unknown class not part of the prompt in its region-of-interest, thus reducing active supervision load while adapting to varying outdoor scenes. Our zero-shot learning approach eliminates the need for extensive data collection or retraining. Our experimental validation includes testing on RELLIS-3D, Freiburg Forest, and RUGD datasets and demonstrate real-world deployment on multiple robot platforms. The results show that AnyTraverse performs better than GA-NAV and Off-seg while offering a vehicle-agnostic approach to off-road traversability that balances automation with targeted human supervision. 

**Abstract (ZH)**: 基于自然语言提示与操作员辅助的离路通行性分割使能自主导航应用于搜索救援、军事操作、野生动物探索和农业等领域 

---
# Learning Dexterous Object Handover 

**Title (ZH)**: 学习灵巧的物体传递 

**Authors**: Daniel Frau-Alfaro, Julio Castaño-Amoros, Santiago Puente, Pablo Gil, Roberto Calandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.16822)  

**Abstract**: Object handover is an important skill that we use daily when interacting with other humans. To deploy robots in collaborative setting, like houses, being able to receive and handing over objects safely and efficiently becomes a crucial skill. In this work, we demonstrate the use of Reinforcement Learning (RL) for dexterous object handover between two multi-finger hands. Key to this task is the use of a novel reward function based on dual quaternions to minimize the rotation distance, which outperforms other rotation representations such as Euler and rotation matrices. The robustness of the trained policy is experimentally evaluated by testing w.r.t. objects that are not included in the training distribution, and perturbations during the handover process. The results demonstrate that the trained policy successfully perform this task, achieving a total success rate of 94% in the best-case scenario after 100 experiments, thereby showing the robustness of our policy with novel objects. In addition, the best-case performance of the policy decreases by only 13.8% when the other robot moves during the handover, proving that our policy is also robust to this type of perturbation, which is common in real-world object handovers. 

**Abstract (ZH)**: 基于双四元数的强化学习在多指手之间进行灵巧物体传递的研究 

---
# Loupe: A Generalizable and Adaptive Framework for Image Forgery Detection 

**Title (ZH)**: Loupe: 一种通用且适应性强的图像伪造检测框架 

**Authors**: Yuchu Jiang, Jiaming Chu, Jian Zhao, Xin Zhang, Xu Yang, Lei Jin, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16819)  

**Abstract**: The proliferation of generative models has raised serious concerns about visual content forgery. Existing deepfake detection methods primarily target either image-level classification or pixel-wise localization. While some achieve high accuracy, they often suffer from limited generalization across manipulation types or rely on complex architectures. In this paper, we propose Loupe, a lightweight yet effective framework for joint deepfake detection and localization. Loupe integrates a patch-aware classifier and a segmentation module with conditional queries, allowing simultaneous global authenticity classification and fine-grained mask prediction. To enhance robustness against distribution shifts of test set, Loupe introduces a pseudo-label-guided test-time adaptation mechanism by leveraging patch-level predictions to supervise the segmentation head. Extensive experiments on the DDL dataset demonstrate that Loupe achieves state-of-the-art performance, securing the first place in the IJCAI 2025 Deepfake Detection and Localization Challenge with an overall score of 0.846. Our results validate the effectiveness of the proposed patch-level fusion and conditional query design in improving both classification accuracy and spatial localization under diverse forgery patterns. The code is available at this https URL. 

**Abstract (ZH)**: 生成模型的 proliferaton 提高了视觉内容伪造的严重性。现有的深度假信息检测方法主要针对图像级分类或像素级定位。尽管一些方法达到了高精度，但它们往往在跨伪造类型的一般化方面表现有限，或者依赖于复杂的架构。在本文中，我们提出了一种轻量而有效的框架 Loupe，用于联合假信息检测与定位。Loupe 结合了patch-aware分类器和具有条件查询的分割模块，允许同时进行全局真实性和分类和精细粒度的掩码预测。为了增强对测试集分布偏移的鲁棒性，Loupe 引入了一种基于 patch-level 预测的伪标签指导测试时适应机制，以监督分割头部。在 DDL 数据集上的广泛实验显示出，Loupe 达到了最先进的性能，在IJCAI 2025 深度假信息检测与定位挑战中以综合得分为0.846获得第一名。我们的结果验证了在多变的伪造模式下，提出的 patch-level 融合和条件查询设计在提高分类准确性和空间定位方面的有效性。代码可在如下链接获取。 

---
# Robust Dynamic Material Handling via Adaptive Constrained Evolutionary Reinforcement Learning 

**Title (ZH)**: 适应性约束进化强化学习下的稳健动态物料处理 

**Authors**: Chengpeng Hu, Ziming Wang, Bo Yuan, Jialin Liu, Chengqi Zhang, Xin Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16795)  

**Abstract**: Dynamic material handling (DMH) involves the assignment of dynamically arriving material transporting tasks to suitable vehicles in real time for minimising makespan and tardiness. In real-world scenarios, historical task records are usually available, which enables the training of a decision policy on multiple instances consisting of historical records. Recently, reinforcement learning has been applied to solve DMH. Due to the occurrence of dynamic events such as new tasks, adaptability is highly required. Solving DMH is challenging since constraints including task delay should be satisfied. A feedback is received only when all tasks are served, which leads to sparse reward. Besides, making the best use of limited computational resources and historical records for training a robust policy is crucial. The time allocated to different problem instances would highly impact the learning process. To tackle those challenges, this paper proposes a novel adaptive constrained evolutionary reinforcement learning (ACERL) approach, which maintains a population of actors for diverse exploration. ACERL accesses each actor for tackling sparse rewards and constraint violation to restrict the behaviour of the policy. Moreover, ACERL adaptively selects the most beneficial training instances for improving the policy. Extensive experiments on eight training and eight unseen test instances demonstrate the outstanding performance of ACERL compared with several state-of-the-art algorithms. Policies trained by ACERL can schedule the vehicles while fully satisfying the constraints. Additional experiments on 40 unseen noised instances show the robust performance of ACERL. Cross-validation further presents the overall effectiveness of ACREL. Besides, a rigorous ablation study highlights the coordination and benefits of each ingredient of ACERL. 

**Abstract (ZH)**: 动态物料搬运中的自适应约束进化强化学习（ACERL）方法 

---
# MIST: Jailbreaking Black-box Large Language Models via Iterative Semantic Tuning 

**Title (ZH)**: MIST: 通过迭代语义调优破解黑盒大型语言模型 

**Authors**: Muyang Zheng, Yuanzhi Yao, Changting Lin, Rui Wang, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.16792)  

**Abstract**: Despite efforts to align large language models (LLMs) with societal and moral values, these models remain susceptible to jailbreak attacks--methods designed to elicit harmful responses. Jailbreaking black-box LLMs is considered challenging due to the discrete nature of token inputs, restricted access to the target LLM, and limited query budget. To address the issues above, we propose an effective method for jailbreaking black-box large language Models via Iterative Semantic Tuning, named MIST. MIST enables attackers to iteratively refine prompts that preserve the original semantic intent while inducing harmful content. Specifically, to balance semantic similarity with computational efficiency, MIST incorporates two key strategies: sequential synonym search, and its advanced version--order-determining optimization. Extensive experiments across two open-source models and four closed-source models demonstrate that MIST achieves competitive attack success rates and attack transferability compared with other state-of-the-art white-box and black-box jailbreak methods. Additionally, we conduct experiments on computational efficiency to validate the practical viability of MIST. 

**Abstract (ZH)**: 尽管努力使大型语言模型（LLMs）与社会和道德价值观保持一致，这些模型仍可能遭受监牢逃脱攻击——旨在引发有害响应的方法。由于标记输入的离散性、访问目标LLM的限制以及查询预算有限，黑盒LLM的监牢逃脱攻击被认为具有挑战性。为了应对上述问题，我们提出了一种名为MIST的迭代语义调整方法，以有效地对黑盒大型语言模型进行监牢逃脱攻击。MIST允许攻击者迭代地细化保留原始语义意图的同时诱导有害内容的提示。具体而言，为了平衡语义相似性和计算效率，MIST引入了两种关键策略：序列同义词搜索及其高级版本——顺序确定优化。在两个开源模型和四个封闭源模型上的 extensive 实验显示，MIST 在攻击成功率和攻击可移植性方面与其他最先进的白盒和黑盒监牢逃脱方法具有竞争力。此外，我们还进行了计算效率实验以验证 MIST 的实际可行性。 

---
# TabArena: A Living Benchmark for Machine Learning on Tabular Data 

**Title (ZH)**: TabArena: 一个针对表格数据机器学习的活基准测试 

**Authors**: Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Mutalik Desai, and David Salinas, Frank Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2506.16791)  

**Abstract**: With the growing popularity of deep learning and foundation models for tabular data, the need for standardized and reliable benchmarks is higher than ever. However, current benchmarks are static. Their design is not updated even if flaws are discovered, model versions are updated, or new models are released. To address this, we introduce TabArena, the first continuously maintained living tabular benchmarking system. To launch TabArena, we manually curate a representative collection of datasets and well-implemented models, conduct a large-scale benchmarking study to initialize a public leaderboard, and assemble a team of experienced maintainers. Our results highlight the influence of validation method and ensembling of hyperparameter configurations to benchmark models at their full potential. While gradient-boosted trees are still strong contenders on practical tabular datasets, we observe that deep learning methods have caught up under larger time budgets with ensembling. At the same time, foundation models excel on smaller datasets. Finally, we show that ensembles across models advance the state-of-the-art in tabular machine learning and investigate the contributions of individual models. We launch TabArena with a public leaderboard, reproducible code, and maintenance protocols to create a living benchmark available at this https URL. 

**Abstract (ZH)**: 随深度学习和基础模型在表格数据中的 popularity 增长，对标准化和可靠的基准测试的需求比以往任何时候都更高。然而，当前的基准测试是静态的，即使发现缺陷、模型版本更新或新模型发布，其设计也不会被更新。为解决这一问题，我们引入了 TabArena，这是第一个持续维护的动态表格基准测试系统。为启动 TabArena，我们手动筛选了具有代表性的数据集和实现良好的模型集合，进行大规模基准测试研究以初始化公共排行榜，并组建了一支经验丰富的维护团队。我们的结果显示了验证方法和超参数配置的集成对充分发挥基准模型性能的影响。尽管在实际的表格数据集上增强梯度提升树仍然表现强劲，但在较大的时间预算下，集成的深度学习方法已经迎头赶上。同时，基础模型在小数据集上表现出色。最后，我们展示了模型间集成推动了表格机器学习的最先进水平，并探讨了单个模型的贡献。我们以公共排行榜、可重复的代码和维护协议启动 TabArena，网址为 this https URL。 

---
# What Is the Point of Equality in Machine Learning Fairness? Beyond Equality of Opportunity 

**Title (ZH)**: 机器学习公平性中的平等意义：超越机会平等 

**Authors**: Youjin Kong  

**Link**: [PDF](https://arxiv.org/pdf/2506.16782)  

**Abstract**: Fairness in machine learning (ML) has become a rapidly growing area of research. But why, in the first place, is unfairness in ML morally wrong? And why should we care about improving fairness? Most fair-ML research implicitly appeals to distributive equality: the idea that desirable goods and benefits, such as opportunities (e.g., Barocas et al., 2023), should be equally distributed across society. Unfair ML models, then, are seen as wrong because they unequally distribute such benefits. This paper argues that this exclusive focus on distributive equality offers an incomplete and potentially misleading ethical foundation. Grounding ML fairness in egalitarianism -- the view that equality is a fundamental moral and social ideal -- requires challenging structural inequality: systematic, institutional, and durable arrangements that privilege some groups while disadvantaging others. Structural inequality manifests through ML systems in two primary forms: allocative harms (e.g., economic loss) and representational harms (e.g., stereotypes, erasure). While distributive equality helps address allocative harms, it fails to explain why representational harms are wrong -- why it is wrong for ML systems to reinforce social hierarchies that stratify people into superior and inferior groups -- and why ML systems should aim to foster a society where people relate as equals (i.e., relational equality). To address these limitations, the paper proposes a multifaceted egalitarian framework for ML fairness that integrates both distributive and relational equality. Drawing on critical social and political philosophy, this framework offers a more comprehensive ethical foundation for tackling the full spectrum of harms perpetuated by ML systems. The paper also outlines practical pathways for implementing the framework across the ML pipeline. 

**Abstract (ZH)**: 机器学习中的公平性：超越分配平等的多维度平等框架 

---
# PQCAD-DM: Progressive Quantization and Calibration-Assisted Distillation for Extremely Efficient Diffusion Model 

**Title (ZH)**: PQCAD-DM: 进步量化和校准辅助蒸馏以实现极其高效的扩散模型 

**Authors**: Beomseok Ko, Hyeryung Jang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16776)  

**Abstract**: Diffusion models excel in image generation but are computational and resource-intensive due to their reliance on iterative Markov chain processes, leading to error accumulation and limiting the effectiveness of naive compression techniques. In this paper, we propose PQCAD-DM, a novel hybrid compression framework combining Progressive Quantization (PQ) and Calibration-Assisted Distillation (CAD) to address these challenges. PQ employs a two-stage quantization with adaptive bit-width transitions guided by a momentum-based mechanism, reducing excessive weight perturbations in low-precision. CAD leverages full-precision calibration datasets during distillation, enabling the student to match full-precision performance even with a quantized teacher. As a result, PQCAD-DM achieves a balance between computational efficiency and generative quality, halving inference time while maintaining competitive performance. Extensive experiments validate PQCAD-DM's superior generative capabilities and efficiency across diverse datasets, outperforming fixed-bit quantization methods. 

**Abstract (ZH)**: PQCAD-DM：一种Combining 分级量化和校准辅助精炼的新型混合压缩框架 

---
# Language-Informed Synthesis of Rational Agent Models for Grounded Theory-of-Mind Reasoning On-The-Fly 

**Title (ZH)**: 基于语言指导的合理代理模型合成用于即时嵌地心智理论推理 

**Authors**: Lance Ying, Ryan Truong, Katherine M. Collins, Cedegao E. Zhang, Megan Wei, Tyler Brooke-Wilson, Tan Zhi-Xuan, Lionel Wong, Joshua B. Tenenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2506.16755)  

**Abstract**: Drawing real world social inferences usually requires taking into account information from multiple modalities. Language is a particularly powerful source of information in social settings, especially in novel situations where language can provide both abstract information about the environment dynamics and concrete specifics about an agent that cannot be easily visually observed. In this paper, we propose Language-Informed Rational Agent Synthesis (LIRAS), a framework for drawing context-specific social inferences that integrate linguistic and visual inputs. LIRAS frames multimodal social reasoning as a process of constructing structured but situation-specific agent and environment representations - leveraging multimodal language models to parse language and visual inputs into unified symbolic representations, over which a Bayesian inverse planning engine can be run to produce granular probabilistic judgments. On a range of existing and new social reasoning tasks derived from cognitive science experiments, we find that our model (instantiated with a comparatively lightweight VLM) outperforms ablations and state-of-the-art models in capturing human judgments across all domains. 

**Abstract (ZH)**: 基于语言指导的理性代理合成（LIRAS）：一种结合语言和视觉输入进行情境特定社会推理的框架 

---
# Metapath-based Hyperbolic Contrastive Learning for Heterogeneous Graph Embedding 

**Title (ZH)**: 基于元路径的双曲对比学习异质图嵌入 

**Authors**: Jongmin Park, Seunghoon Han, Won-Yong Shin, Sungsu Lim  

**Link**: [PDF](https://arxiv.org/pdf/2506.16754)  

**Abstract**: The hyperbolic space, characterized by a constant negative curvature and exponentially expanding space, aligns well with the structural properties of heterogeneous graphs. However, although heterogeneous graphs inherently possess diverse power-law structures, most hyperbolic heterogeneous graph embedding models rely on a single hyperbolic space. This approach may fail to effectively capture the diverse power-law structures within heterogeneous graphs. To address this limitation, we propose a Metapath-based Hyperbolic Contrastive Learning framework (MHCL), which uses multiple hyperbolic spaces to capture diverse complex structures within heterogeneous graphs. Specifically, by learning each hyperbolic space to describe the distribution of complex structures corresponding to each metapath, it is possible to capture semantic information effectively. Since metapath embeddings represent distinct semantic information, preserving their discriminability is important when aggregating them to obtain node representations. Therefore, we use a contrastive learning approach to optimize MHCL and improve the discriminability of metapath embeddings. In particular, our contrastive learning method minimizes the distance between embeddings of the same metapath and maximizes the distance between those of different metapaths in hyperbolic space, thereby improving the separability of metapath embeddings with distinct semantic information. We conduct comprehensive experiments to evaluate the effectiveness of MHCL. The experimental results demonstrate that MHCL outperforms state-of-the-art baselines in various graph machine learning tasks, effectively capturing the complex structures of heterogeneous graphs. 

**Abstract (ZH)**: 基于元路径的双曲对比学习框架（MHCL）：捕捉异构图中的多样复杂结构 

---
# Off-Policy Actor-Critic for Adversarial Observation Robustness: Virtual Alternative Training via Symmetric Policy Evaluation 

**Title (ZH)**: 针对对抗观测稳健性的离策训练actor-critic方法：通过对称策略评估的虚拟替代训练 

**Authors**: Kosuke Nakanishi, Akihiro Kubo, Yuji Yasui, Shin Ishii  

**Link**: [PDF](https://arxiv.org/pdf/2506.16753)  

**Abstract**: Recently, robust reinforcement learning (RL) methods designed to handle adversarial input observations have received significant attention, motivated by RL's inherent vulnerabilities. While existing approaches have demonstrated reasonable success, addressing worst-case scenarios over long time horizons requires both minimizing the agent's cumulative rewards for adversaries and training agents to counteract them through alternating learning. However, this process introduces mutual dependencies between the agent and the adversary, making interactions with the environment inefficient and hindering the development of off-policy methods. In this work, we propose a novel off-policy method that eliminates the need for additional environmental interactions by reformulating adversarial learning as a soft-constrained optimization problem. Our approach is theoretically supported by the symmetric property of policy evaluation between the agent and the adversary. The implementation is available at this https URL. 

**Abstract (ZH)**: 最近，针对对抗输入观测的鲁棒强化学习方法受到了广泛关注，这类方法旨在处理RL固有的脆弱性。虽然现有方法已经显示出合理的成效，但在长时间范围内应对最坏情况场景需要同时最小化智能体累积的奖励给对手带来的影响，并通过交替学习训练智能体对抗对手。然而，这一过程引入了智能体与对手之间的相互依赖，使得与环境的交互变得低效，阻碍了离策方法的发展。本文提出了一种新的离策方法，通过将对抗学习重新表述为软约束优化问题来消除额外环境交互的需求。我们的方法基于智能体和对手之间策略评估的对称性质具有理论支持。该实现可在以下链接访问：this https URL。 

---
# RapFlow-TTS: Rapid and High-Fidelity Text-to-Speech with Improved Consistency Flow Matching 

**Title (ZH)**: RapFlow-TTS：快速高保真文本到语音转换及改进的一致性流匹配 

**Authors**: Hyun Joon Park, Jeongmin Liu, Jin Sob Kim, Jeong Yeol Yang, Sung Won Han, Eunwoo Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.16741)  

**Abstract**: We introduce RapFlow-TTS, a rapid and high-fidelity TTS acoustic model that leverages velocity consistency constraints in flow matching (FM) training. Although ordinary differential equation (ODE)-based TTS generation achieves natural-quality speech, it typically requires a large number of generation steps, resulting in a trade-off between quality and inference speed. To address this challenge, RapFlow-TTS enforces consistency in the velocity field along the FM-straightened ODE trajectory, enabling consistent synthetic quality with fewer generation steps. Additionally, we introduce techniques such as time interval scheduling and adversarial learning to further enhance the quality of the few-step synthesis. Experimental results show that RapFlow-TTS achieves high-fidelity speech synthesis with a 5- and 10-fold reduction in synthesis steps than the conventional FM- and score-based approaches, respectively. 

**Abstract (ZH)**: RapFlow-TTS：一种基于流匹配的快速高保真TTS声学模型 

---
# LM-SPT: LM-Aligned Semantic Distillation for Speech Tokenization 

**Title (ZH)**: LM-SPT: LM对齐的语义蒸馏用于语音分词 

**Authors**: Daejin Jo, Jeeyoung Yun, Byungseok Roh, Sungwoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.16738)  

**Abstract**: With the rapid progress of speech language models (SLMs), discrete speech tokens have emerged as a core interface between speech and text, enabling unified modeling across modalities. Recent speech tokenization approaches aim to isolate semantic information from low-level acoustics to better align with language models. In particular, previous methods use SSL teachers such as HuBERT to extract semantic representations, which are then distilled into a semantic quantizer to suppress acoustic redundancy as well as capture content-related latent structures. However, they still produce speech token sequences significantly longer than their textual counterparts, creating challenges for efficient speech-language modeling. Reducing the frame rate is a natural solution, but standard techniques, such as rigid average pooling across frames, can distort or dilute the semantic structure required for effective LM alignment. To address this, we propose LM-SPT, a speech tokenization method that introduces a novel semantic distillation. Instead of directly matching teacher and student features via pooling, we reconstruct speech solely from semantic tokens and minimize the discrepancy between the encoded representations of the original and reconstructed waveforms, obtained from a frozen automatic speech recognition (ASR) encoder. This indirect yet data-driven supervision enables the tokenizer to learn discrete units that are more semantically aligned with language models. LM-SPT further incorporates architectural improvements to the encoder and decoder for speech tokenization, and supports multiple frame rates, including 25Hz, 12.5Hz, and 6.25Hz. Experimental results show that LM-SPT achieves superior reconstruction fidelity compared to baselines, and that SLMs trained with LM-SPT tokens achieve competitive performances on speech-to-text and consistently outperform baselines on text-to-speech tasks. 

**Abstract (ZH)**: 借助语音语言模型的迅猛发展，离散语音令牌已成为语音与文本之间的核心接口，实现了跨模态的统一建模。最近的语音分词方法旨在从低级 acoustic 信息中隔离语义信息，以更好地与语言模型对齐。特别是，以往的方法使用如 HuBERT 的 SSL 老师来提取语义表示，这些表示随后被精简成语义量器，以抑制 acoustic 冗余并捕捉内容相关的潜在结构。然而，它们仍然生成显著长于其文本对应物的语音令牌序列，这为高效的语音-语言建模带来了挑战。降低帧率是一种自然的解决方案，但标准技术，如在帧之间进行刚性平均池化，可能会歪曲或稀释有效的语言模型对齐所需的语义结构。为解决这一问题，我们提出了 LM-SPT，一种引入新颖语义精简的语音分词方法。我们不是通过池化直接匹配老师和学生特征，而是仅从语义令牌重构语音，并最小化原始波形和重构波形的编码表示之间的差异，这些波形来自冻结的自动语音识别（ASR）编码器。这种间接但基于数据的监督使分词器能够学习更符合语言模型的离散单元。LM-SPT 还对编码器和解码器的架构进行了改进，支持多种帧率，包括 25Hz、12.5Hz 和 6.25Hz。实验结果显示，LM-SPT 在重建保真度方面优于基线，并且使用 LM-SPT 令牌训练的语音语言模型在语音转文本任务上表现出竞争性性能，在文本转语音任务上也始终优于基线。 

---
# On Training-Test (Mis)alignment in Unsupervised Combinatorial Optimization: Observation, Empirical Exploration, and Analysis 

**Title (ZH)**: 关于无监督组合优化中训练-测试不对齐的现象、 empirical 探索与分析 

**Authors**: Fanchen Bu, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2506.16732)  

**Abstract**: In unsupervised combinatorial optimization (UCO), during training, one aims to have continuous decisions that are promising in a probabilistic sense for each training instance, which enables end-to-end training on initially discrete and non-differentiable problems. At the test time, for each test instance, starting from continuous decisions, derandomization is typically applied to obtain the final deterministic decisions. Researchers have developed more and more powerful test-time derandomization schemes to enhance the empirical performance and the theoretical guarantee of UCO methods. However, we notice a misalignment between training and testing in the existing UCO methods. Consequently, lower training losses do not necessarily entail better post-derandomization performance, even for the training instances without any data distribution shift. Empirically, we indeed observe such undesirable cases. We explore a preliminary idea to better align training and testing in UCO by including a differentiable version of derandomization into training. Our empirical exploration shows that such an idea indeed improves training-test alignment, but also introduces nontrivial challenges into training. 

**Abstract (ZH)**: 无监督组合优化中的训练与测试对齐：包括可微分的去随机化版本以改善训练-测试对齐 

---
# The Role of Model Confidence on Bias Effects in Measured Uncertainties 

**Title (ZH)**: 模型置信度在测量不确定性中的偏差效应作用 

**Authors**: Xinyi Liu, Weiguang Wang, Hangfeng He  

**Link**: [PDF](https://arxiv.org/pdf/2506.16724)  

**Abstract**: With the growing adoption of Large Language Models (LLMs) for open-ended tasks, accurately assessing epistemic uncertainty, which reflects a model's lack of knowledge, has become crucial to ensuring reliable outcomes. However, quantifying epistemic uncertainty in such tasks is challenging due to the presence of aleatoric uncertainty, which arises from multiple valid answers. While bias can introduce noise into epistemic uncertainty estimation, it may also reduce noise from aleatoric uncertainty. To investigate this trade-off, we conduct experiments on Visual Question Answering (VQA) tasks and find that mitigating prompt-introduced bias improves uncertainty quantification in GPT-4o. Building on prior work showing that LLMs tend to copy input information when model confidence is low, we further analyze how these prompt biases affect measured epistemic and aleatoric uncertainty across varying bias-free confidence levels with GPT-4o and Qwen2-VL. We find that all considered biases induce greater changes in both uncertainties when bias-free model confidence is lower. Moreover, lower bias-free model confidence leads to greater underestimation of epistemic uncertainty (i.e. overconfidence) due to bias, whereas it has no significant effect on the direction of changes in aleatoric uncertainty estimation. These distinct effects deepen our understanding of bias mitigation for uncertainty quantification and potentially inform the development of more advanced techniques. 

**Abstract (ZH)**: 大型语言模型在开放任务中的逐渐采用使得准确评估表征模型知识不足的 epistemic 模式不确定性变得至关重要，以确保可靠的结果。然而，由于存在 aleatoric 模式不确定性（即多正确答案引起的不确定性），在这些任务中量化 epistemic 模式不确定性具有挑战性。虽然偏差可能引入噪声到 epistemic 模式不确定性估计中，但它也可能减少 aleatoric 模式不确定性中的噪声。为了调查这种权衡，我们在视觉问答（VQA）任务上进行实验，并发现减轻提示引入的偏差可以改进 GPT-4o 中的不确定性量化。基于先前研究表明，当模型置信度较低时，LLM 趋于复制输入信息，我们进一步分析了这些提示偏差如何影响 GPT-4o 和 Qwen2-VL 在不同置信度水平下的测量表征模式和 aleatoric 不确定性。我们发现，当置信度无偏时较低时，所有考虑的偏差都会导致两种不确定性更大的变化。此外，较低的置信度无偏状态会导致偏差引起的表征模式不确定性低估（即过度自信），而这对 aleatoric 不确定性估计方向的变化没有显著影响。这些不同的影响加深了我们对不确定性量化中的偏差减轻的理解，并可能为开发更先进的技术提供指导。 

---
# TriCon-SF: A Triple-Shuffle and Contribution-Aware Serial Federated Learning Framework for Heterogeneous Healthcare Data 

**Title (ZH)**: TriCon-SF: 一种考虑贡献的异构 healthcare 数据三重洗牌串联联邦学习框架 

**Authors**: Yuping Yan, Yizhi Wang, Yuanshuai Li, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.16723)  

**Abstract**: Serial pipeline training is an efficient paradigm for handling data heterogeneity in cross-silo federated learning with low communication overhead. However, even without centralized aggregation, direct transfer of models between clients can violate privacy regulations and remain susceptible to gradient leakage and linkage attacks. Additionally, ensuring resilience against semi-honest or malicious clients who may manipulate or misuse received models remains a grand challenge, particularly in privacy-sensitive domains such as healthcare. To address these challenges, we propose TriCon-SF, a novel serial federated learning framework that integrates triple shuffling and contribution awareness. TriCon-SF introduces three levels of randomization by shuffling model layers, data segments, and training sequences to break deterministic learning patterns and disrupt potential attack vectors, thereby enhancing privacy and robustness. In parallel, it leverages Shapley value methods to dynamically evaluate client contributions during training, enabling the detection of dishonest behavior and enhancing system accountability. Extensive experiments on non-IID healthcare datasets demonstrate that TriCon-SF outperforms standard serial and parallel federated learning in both accuracy and communication efficiency. Security analysis further supports its resilience against client-side privacy attacks. 

**Abstract (ZH)**: 基于三重洗牌和贡献感知的串行联邦学习框架TriCon-SF：低通信开销下处理跨孤岛联邦学习数据异质性的高效范式 

---
# Generalizable Agent Modeling for Agent Collaboration-Competition Adaptation with Multi-Retrieval and Dynamic Generation 

**Title (ZH)**: 具有多检索和动态生成的可迁移代理建模以适应代理协作与竞争适应 

**Authors**: Chenxu Wang, Yonggang Jin, Cheng Hu, Youpeng Zhao, Zipeng Dai, Jian Zhao, Shiyu Huang, Liuyu Xiang, Junge Zhang, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2506.16718)  

**Abstract**: Adapting a single agent to a new multi-agent system brings challenges, necessitating adjustments across various tasks, environments, and interactions with unknown teammates and opponents. Addressing this challenge is highly complex, and researchers have proposed two simplified scenarios, Multi-agent reinforcement learning for zero-shot learning and Ad-Hoc Teamwork. Building on these foundations, we propose a more comprehensive setting, Agent Collaborative-Competitive Adaptation (ACCA), which evaluates an agent to generalize across diverse scenarios, tasks, and interactions with both unfamiliar opponents and teammates. In ACCA, agents adjust to task and environmental changes, collaborate with unseen teammates, and compete against unknown opponents. We introduce a new modeling approach, Multi-Retrieval and Dynamic Generation (MRDG), that effectively models both teammates and opponents using their behavioral trajectories. This method incorporates a positional encoder for varying team sizes and a hypernetwork module to boost agents' learning and adaptive capabilities. Additionally, a viewpoint alignment module harmonizes the observational perspectives of retrieved teammates and opponents with the learning agent. Extensive tests in benchmark scenarios like SMAC, Overcooked-AI, and Melting Pot show that MRDG significantly improves robust collaboration and competition with unseen teammates and opponents, surpassing established baselines. Our code is available at: this https URL 

**Abstract (ZH)**: 将单一代理适应新的多代理系统带来了挑战，需要在各种任务、环境以及与未知队友和对手的互动中进行调整。解决这一挑战非常复杂，研究人员提出了两种简化的场景：零样本学习的多代理强化学习和即兴团队合作。在这些基础上，我们提出一种更为全面的设置——代理协作-竞争适应（ACCA），该设置评估代理在多样场景、任务和与未知队友及对手互动中的通用性。在ACCA中，代理适应任务和环境变化，与未见过的队友合作，并与未知对手竞争。我们引入了一种新的建模方法——多检索和动态生成（MRDG），该方法有效利用了行为轨迹来建模队友和对手。该方法包含一个位置编码器以适应不同规模的团队，并包含一个超网络模块以增强代理的学习和适应能力。此外，视点对齐模块使检索到的队友和对手的观察视角与学习代理协调一致。在基准场景SMAC、Overcooked-AI和Melting Pot中的广泛测试表明，MRDG显著提高了与未见过队友和对手的稳健合作和竞争能力，超过了已有的基准。我们的代码可在以下链接获取：this https URL。 

---
# ReasonGRM: Enhancing Generative Reward Models through Large Reasoning Models 

**Title (ZH)**: ReasonGRM：通过大规模推理模型增强生成奖励模型 

**Authors**: Bin Chen, Xinzge Gao, Chuanrui Hu, Penghang Yu, Hua Zhang, Bing-Kun Bao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16712)  

**Abstract**: Generative Reward Models (GRMs) provide greater flexibility than scalar reward models in capturing human preferences, but their effectiveness is limited by poor reasoning capabilities. This often results in incomplete or overly speculative reasoning paths, leading to hallucinations or missing key information in complex tasks. We address this challenge with ReasonGRM, a three-stage generative reward modeling framework. In the first stage, Zero-RL is used to generate concise, outcome-directed reasoning paths that reduce the likelihood of critical omissions. In the second stage, we introduce a novel evaluation metric, $R^\star$, which scores reasoning paths based on their generation likelihood. This favors paths that reach correct answers with minimal exploration, helping to reduce hallucination-prone data during training. In the final stage, the model is further refined through reinforcement learning on challenging examples to enhance its preference discrimination capabilities. Experiments on three public benchmarks show that ReasonGRM achieves competitive or state-of-the-art performance, outperforming previous best GRMs by 1.8\% on average and surpassing proprietary models such as GPT-4o by up to 5.6\%. These results demonstrate the effectiveness of reasoning-aware training and highlight the importance of high-quality rationale selection for reliable preference modeling. 

**Abstract (ZH)**: 基于推理的生成奖励模型（ReasonGRM）：一种三阶段的生成奖励模型框架 

---
# Large Language Models as Psychological Simulators: A Methodological Guide 

**Title (ZH)**: 大型语言模型作为心理模拟器：一种方法论指南 

**Authors**: Zhicheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.16702)  

**Abstract**: Large language models (LLMs) offer emerging opportunities for psychological and behavioral research, but methodological guidance is lacking. This article provides a framework for using LLMs as psychological simulators across two primary applications: simulating roles and personas to explore diverse contexts, and serving as computational models to investigate cognitive processes. For simulation, we present methods for developing psychologically grounded personas that move beyond demographic categories, with strategies for validation against human data and use cases ranging from studying inaccessible populations to prototyping research instruments. For cognitive modeling, we synthesize emerging approaches for probing internal representations, methodological advances in causal interventions, and strategies for relating model behavior to human cognition. We address overarching challenges including prompt sensitivity, temporal limitations from training data cutoffs, and ethical considerations that extend beyond traditional human subjects review. Throughout, we emphasize the need for transparency about model capabilities and constraints. Together, this framework integrates emerging empirical evidence about LLM performance--including systematic biases, cultural limitations, and prompt brittleness--to help researchers wrangle these challenges and leverage the unique capabilities of LLMs in psychological research. 

**Abstract (ZH)**: 大型语言模型（LLMs）为心理和行为研究提供了新兴机遇，但方法学指导不足。本文提出了一种使用LLMs作为心理模拟器的框架，涵盖两大主要应用：通过模拟角色和个性探索多元情境，以及作为计算模型探究认知过程。对于模拟，我们介绍了开发基于心理机制的人格的方法，超越了人口统计学类别，包括验证策略以及从研究不可达群体到研究工具原型设计的应用场景。对于认知建模，我们综合了探究内部表征的新兴方法、因果干预的方法学进展以及模型行为与人类认知关系的策略。我们还讨论了包括提示敏感性、训练数据截止时间导致的时间限制等首要挑战，并超越传统的人类受试者审查提出了伦理考虑。全文强调了关于模型能力与限制的透明性需求。该框架结合了关于LLM性能的新兴实证证据，包括系统性偏差、文化限制和提示脆弱性，帮助研究人员应对这些挑战，并充分利用LLMs在心理研究中的独特能力。 

---
# From Prompts to Constructs: A Dual-Validity Framework for LLM Research in Psychology 

**Title (ZH)**: 从提示到构建：心理领域LLM研究的双重效度框架 

**Authors**: Zhicheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.16697)  

**Abstract**: Large language models (LLMs) are rapidly being adopted across psychology, serving as research tools, experimental subjects, human simulators, and computational models of cognition. However, the application of human measurement tools to these systems can produce contradictory results, raising concerns that many findings are measurement phantoms--statistical artifacts rather than genuine psychological phenomena. In this Perspective, we argue that building a robust science of AI psychology requires integrating two of our field's foundational pillars: the principles of reliable measurement and the standards for sound causal inference. We present a dual-validity framework to guide this integration, which clarifies how the evidence needed to support a claim scales with its scientific ambition. Using an LLM to classify text may require only basic accuracy checks, whereas claiming it can simulate anxiety demands a far more rigorous validation process. Current practice systematically fails to meet these requirements, often treating statistical pattern matching as evidence of psychological phenomena. The same model output--endorsing "I am anxious"--requires different validation strategies depending on whether researchers claim to measure, characterize, simulate, or model psychological constructs. Moving forward requires developing computational analogues of psychological constructs and establishing clear, scalable standards of evidence rather than the uncritical application of human measurement tools. 

**Abstract (ZH)**: 大型语言模型（LLMs）在心理学中的快速应用：作为研究工具、实验对象、人类模拟器和认知的计算模型。然而，将人类测量工具应用于这些系统会产生矛盾的结果，引发了关于许多发现是否为测量幻象——统计艺术而非实质性心理现象的担忧。在本文中，我们argue认为构建稳健的人工智能心理学科学需要整合我们领域两大基石：可靠测量的原则和良好的因果推理标准。我们提出了一种双重效度框架来指导这一整合，该框架明确了支持一项主张所需的证据规模与其科学雄心之间的关系。使用LLM分类文本可能只需要基本的准确度检查，而声称它可以模拟焦虑则需要更为严格的验证过程。当前的做法系统性地未能达到这些要求，经常将统计模式匹配视为心理现象的证据。同样模型输出——“我感到焦虑”——在研究人员声称测量、表征、模拟或建模心理构念时需要不同的验证策略。向前推进需要开发心理构念的计算机模拟，并建立清晰的可扩展的证据标准，而不是对人类测量工具的无批判应用。 

---
# Fast and Stable Diffusion Planning through Variational Adaptive Weighting 

**Title (ZH)**: 快速且稳定的扩散规划通过变分自适应加权 

**Authors**: Zhiying Qiu, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.16688)  

**Abstract**: Diffusion models have recently shown promise in offline RL. However, these methods often suffer from high training costs and slow convergence, particularly when using transformer-based denoising backbones. While several optimization strategies have been proposed -- such as modified noise schedules, auxiliary prediction targets, and adaptive loss weighting -- challenges remain in achieving stable and efficient training. In particular, existing loss weighting functions typically rely on neural network approximators, which can be ineffective in early training phases due to limited generalization capacity of MLPs when exposed to sparse feedback in the early training stages. In this work, we derive a variationally optimal uncertainty-aware weighting function and introduce a closed-form polynomial approximation method for its online estimation under the flow-based generative modeling framework. We integrate our method into a diffusion planning pipeline and evaluate it on standard offline RL benchmarks. Experimental results on Maze2D and Kitchen tasks show that our method achieves competitive performance with up to 10 times fewer training steps, highlighting its practical effectiveness. 

**Abstract (ZH)**: 基于流的生成建模框架下变分最优不确定性感知权重函数及其在线闭式多项式逼近方法在离线RL中的应用 

---
# A Simple Contrastive Framework Of Item Tokenization For Generative Recommendation 

**Title (ZH)**: 基于项分词的简单对比框架生成推荐 

**Authors**: Penglong Zhai, Yifang Yuan, Fanyi Di, Jie Li, Yue Liu, Chen Li, Jie Huang, Sicong Wang, Yao Xu, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16683)  

**Abstract**: Generative retrieval-based recommendation has emerged as a promising paradigm aiming at directly generating the identifiers of the target candidates. However, in large-scale recommendation systems, this approach becomes increasingly cumbersome due to the redundancy and sheer scale of the token space. To overcome these limitations, recent research has explored the use of semantic tokens as an alternative to ID tokens, which typically leveraged reconstruction-based strategies, like RQ-VAE, to quantize content embeddings and significantly reduce the embedding size. However, reconstructive quantization aims for the precise reconstruction of each item embedding independently, which conflicts with the goal of generative retrieval tasks focusing more on differentiating among items. Moreover, multi-modal side information of items, such as descriptive text and images, geographical knowledge in location-based recommendation services, has been shown to be effective in improving recommendations by providing richer contexts for interactions. Nevertheless, effectively integrating such complementary knowledge into existing generative recommendation frameworks remains challenging. To overcome these challenges, we propose a novel unsupervised deep quantization exclusively based on contrastive learning, named SimCIT (a Simple Contrastive Item Tokenization framework). Specifically, different from existing reconstruction-based strategies, SimCIT propose to use a learnable residual quantization module to align with the signals from different modalities of the items, which combines multi-modal knowledge alignment and semantic tokenization in a mutually beneficial contrastive learning framework. Extensive experiments across public datasets and a large-scale industrial dataset from various domains demonstrate SimCIT's effectiveness in LLM-based generative recommendation. 

**Abstract (ZH)**: 基于生成性检索的推荐方法已 emerges as a promising paradigm aiming at直接生成目标候选项的标识符。然而，在大规模推荐系统中，由于令牌空间的冗余性和规模庞大，这种方法变得越来越繁琐。为了克服这些限制，最近的研究探索了使用语义令牌作为ID令牌的替代方案，通常利用基于重构的策略（如RQ-VAE）对内容嵌入进行量化，并显著减小嵌入尺寸。然而，基于重构的量化旨在独立精确重构每个项目嵌入，这与生成性检索任务更侧重于项目之间的区分目标相冲突。此外，项目多模态侧信息，如描述性文本和图像，以及位置推荐服务中的地理知识，已被证明通过提供更丰富的交互上下文来有效提高推荐效果。然而，有效地将此类互补知识整合到现有的生成性推荐框架中仍然是一个挑战。为了解决这些挑战，我们提出了一种新的无监督深度量化方法，名为SimCIT（一种简单的对比项标记框架），专门基于对比学习。具体而言，不同于现有的基于重构的策略，SimCIT提出使用可学习的残差量化模块来与项目不同模态的信号对齐，在这种相互有益的对比学习框架中结合了多模态知识对齐和语义标记化。在多个公开数据集和来自不同领域的大型工业数据集上的广泛实验表明，SimCIT在基于LLM的生成推荐中具有有效性。 

---
# How to Train your Text-to-Image Model: Evaluating Design Choices for Synthetic Training Captions 

**Title (ZH)**: 如何训练你的文本-to-图像模型：关于合成训练描述词设计选择的评估 

**Authors**: Manuel Brack, Sudeep Katakol, Felix Friedrich, Patrick Schramowski, Hareesh Ravi, Kristian Kersting, Ajinkya Kale  

**Link**: [PDF](https://arxiv.org/pdf/2506.16679)  

**Abstract**: Training data is at the core of any successful text-to-image models. The quality and descriptiveness of image text are crucial to a model's performance. Given the noisiness and inconsistency in web-scraped datasets, recent works shifted towards synthetic training captions. While this setup is generally believed to produce more capable models, current literature does not provide any insights into its design choices. This study closes this gap by systematically investigating how different synthetic captioning strategies impact the downstream performance of text-to-image models. Our experiments demonstrate that dense, high-quality captions enhance text alignment but may introduce trade-offs in output aesthetics and diversity. Conversely, captions of randomized lengths yield balanced improvements across aesthetics and alignment without compromising sample diversity. We also demonstrate that varying caption distributions introduce significant shifts in the output bias of a trained model. Our findings underscore the importance of caption design in achieving optimal model performance and provide practical insights for more effective training data strategies in text-to-image generation. 

**Abstract (ZH)**: 训练数据是任何成功文本到图像模型的核心。图像文本的质量和描述性对于模型性能至关重要。鉴于网络抓取数据集中的噪声和不一致性，近期的研究转向了合成训练Caption。虽然这种设置通常被认为能够产生更强大的模型，但当前文献并未提供其设计选择的相关见解。本研究通过系统性地探究不同合成Caption策略对下游文本到图像模型性能的影响，填补了这一空白。我们的实验表明，密集且高质量的Caption能够增强文本对齐，但可能会导致输出美学性和多样性之间的权衡。相反，随机长度的Caption能够在美学性和对齐之间提供均衡的改进，而不会牺牲样本多样性。此外，我们还证明了改变Caption分布会对训练模型的输出偏差产生显著影响。我们的研究结果强调了Caption设计在实现最优模型性能中的重要性，并为文本到图像生成中更有效的训练数据策略提供了实用见解。 

---
# A Minimalist Optimizer Design for LLM Pretraining 

**Title (ZH)**: 面向LLM预训练的极简优化器设计 

**Authors**: Athanasios Glentis, Jiaxiang Li, Andi Han, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2506.16659)  

**Abstract**: Training large language models (LLMs) typically relies on adaptive optimizers such as Adam, which require significant memory to maintain first- and second-moment matrices, known as optimizer states. While recent works such as GaLore, Fira, and APOLLO have proposed state-compressed variants to reduce memory consumption, a fundamental question remains: What is the minimal amount of optimizer state that is truly necessary to retain state-of-the-art performance in LLM pretraining? In this work, we systematically investigate this question using a bottom-up approach. We find that two memory- and compute-efficient optimization techniques are particularly effective: (1) column-wise gradient normalization significantly boosts the performance of plain SGD without requiring momentum; and (2) adding first-order momentum only to the output layer - where gradient variance is highest - yields performance competitive with fully adaptive methods such as Muon. Based on these insights, we propose SCALE (Stochastic Column-normalized Last-layer Momentum), a new optimizer that combines column-normalized SGD with last-layer momentum, where column normalization refers to normalizing the gradient along the output dimension. Across multiple LLaMA models (60M-1B), SCALE matches or exceeds the performance of Adam while using only 35-45% of the total memory. It also consistently outperforms memory-efficient optimizers such as GaLore, Fira, and APOLLO, making it a strong candidate for large-scale pretraining under memory constraints. For the LLaMA 7B model, SCALE outperforms the state-of-the-art method APOLLO in terms of both perplexity and memory consumption. In addition, our method serves as a minimalist baseline for more sophisticated optimizer design. 

**Abstract (ZH)**: Training大规模语言模型（LLMs）通常依赖于自适应优化器如Adam，这些优化器需要大量的内存来维护一阶和二阶矩矩阵，即优化器状态。虽然GaLore、Fira和APOLLO等近期工作提出了状态压缩变体以减少内存消耗，但一个基本问题依然存在：保留最先进的性能所需的最小优化器状态量究竟是多少？在本工作中，我们采用自底向上的方法系统地研究了这一问题。我们发现两种在内存和计算上高效的优化技术特别有效：（1）列规范化梯度显著提升了普通的SGD性能，而不需要动量；（2）仅在梯度方差最高的输出层添加一阶动量，性能可与完全自适应方法如Muon媲美。基于这些见解，我们提出了STABLE（Stochastic Column-normalized Last-layer Momentum），一种结合列规范化SGD和输出层动量的新优化器，其中列规范化指的是在输出维度上规范化梯度。通过对多个LLaMA模型（60M-1B），STABLE在使用总内存35-45%的情况下达到或超过了Adam的性能。它还一致地优于GaLore、Fira和APOLLO等内存高效的优化器，使其在内存受限的大规模预训练中是一个强有力的选择。对于LLaMA 7B模型，STABLE在困惑度和内存消耗方面都优于最先进的方法APOLLO。此外，我们的方法为更复杂的优化器设计提供了简约的基础。 

---
# Relational Deep Learning: Challenges, Foundations and Next-Generation Architectures 

**Title (ZH)**: 关系深度学习：挑战、基础与下一代架构 

**Authors**: Vijay Prakash Dwivedi, Charilaos Kanatsoulis, Shenyang Huang, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2506.16654)  

**Abstract**: Graph machine learning has led to a significant increase in the capabilities of models that learn on arbitrary graph-structured data and has been applied to molecules, social networks, recommendation systems, and transportation, among other domains. Data in multi-tabular relational databases can also be constructed as 'relational entity graphs' for Relational Deep Learning (RDL) - a new blueprint that enables end-to-end representation learning without traditional feature engineering. Compared to arbitrary graph-structured data, relational entity graphs have key properties: (i) their structure is defined by primary-foreign key relationships between entities in different tables, (ii) the structural connectivity is a function of the relational schema defining a database, and (iii) the graph connectivity is temporal and heterogeneous in nature. In this paper, we provide a comprehensive review of RDL by first introducing the representation of relational databases as relational entity graphs, and then reviewing public benchmark datasets that have been used to develop and evaluate recent GNN-based RDL models. We discuss key challenges including large-scale multi-table integration and the complexities of modeling temporal dynamics and heterogeneous data, while also surveying foundational neural network methods and recent architectural advances specialized for relational entity graphs. Finally, we explore opportunities to unify these distinct modeling challenges, highlighting how RDL converges multiple sub-fields in graph machine learning towards the design of foundation models that can transform the processing of relational data. 

**Abstract (ZH)**: 图机器学习在分子、社会网络、推荐系统和交通等领域处理任意图结构数据的能力上取得了显著提升，并且可以通过关系深度学习（RDL）将多表关系数据库构建为“关系实体图”，从而实现端到端的表示学习而无需传统特征工程。与任意图结构数据相比，关系实体图具有以下关键特性：（i）其结构由不同表中的实体之间的一对多关系定义，（ii）结构连接性由定义数据库的关系模式决定，（iii）图连接性在时间和异构性方面具有性质。在本文中，我们首先通过介绍关系数据库表示为关系实体图来全面回顾关系深度学习（RDL），然后回顾用于开发和评估基于图神经网络（GNN）的RDL模型的公共基准数据集，讨论包括大规模多表集成在内的关键挑战以及建模时间动态和异构数据的复杂性，并概述适用于关系实体图的基础神经网络方法和近期架构进步。最后，我们探索了统一这些不同建模挑战的机会，强调RDL如何将图机器学习中的多个子领域统一起来，朝着设计能够转型处理关系数据的基础模型方向发展。 

---
# LLMs in Coding and their Impact on the Commercial Software Engineering Landscape 

**Title (ZH)**: LLMs在编程中的应用及其对商业软件工程景观的影响 

**Authors**: Vladislav Belozerov, Peter J Barclay, Askhan Sami  

**Link**: [PDF](https://arxiv.org/pdf/2506.16653)  

**Abstract**: Large-language-model coding tools are now mainstream in software engineering. But as these same tools move human effort up the development stack, they present fresh dangers: 10% of real prompts leak private data, 42% of generated snippets hide security flaws, and the models can even ``agree'' with wrong ideas, a trait called sycophancy. We argue that firms must tag and review every AI-generated line of code, keep prompts and outputs inside private or on-premises deployments, obey emerging safety regulations, and add tests that catch sycophantic answers -- so they can gain speed without losing security and accuracy. 

**Abstract (ZH)**: 大型语言模型编程工具现已成为软件工程的主流。但随着这些工具将人力投入开发栈的上层，它们也带来了新的风险：10%的实际提示泄露了私人数据，42%的生成代码片段隐藏了安全漏洞，模型甚至会“赞同”错误的观点，这一特性被称为拍马哈 doling。我们认为企业必须标记和审查每行AI生成的代码，将提示和输出保留在私有或内部部署中，遵守新兴的安全规定，并添加检测拍马哈回答的测试，以便在不牺牲安全性和准确性的前提下获得速度。 

---
# SemAgent: A Semantics Aware Program Repair Agent 

**Title (ZH)**: SemAgent: 具有语义意识的程序修复代理 

**Authors**: Anvith Pabba, Alex Mathai, Anindya Chakraborty, Baishakhi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2506.16650)  

**Abstract**: Large Language Models (LLMs) have shown impressive capabilities in downstream software engineering tasks such as Automated Program Repair (APR). In particular, there has been a lot of research on repository-level issue-resolution benchmarks such as SWE-Bench. Although there has been significant progress on this topic, we notice that in the process of solving such issues, existing agentic systems tend to hyper-localize on immediately suspicious lines of code and fix them in isolation, without a deeper understanding of the issue semantics, code semantics, or execution semantics. Consequently, many existing systems generate patches that overfit to the user issue, even when a more general fix is preferable. To address this limitation, we introduce SemAgent, a novel workflow-based procedure that leverages issue, code, and execution semantics to generate patches that are complete - identifying and fixing all lines relevant to the issue. We achieve this through a novel pipeline that (a) leverages execution semantics to retrieve relevant context, (b) comprehends issue-semantics via generalized abstraction, (c) isolates code-semantics within the context of this abstraction, and (d) leverages this understanding in a two-stage architecture: a repair stage that proposes fine-grained fixes, followed by a reviewer stage that filters relevant fixes based on the inferred issue-semantics. Our evaluations show that our methodology achieves a solve rate of 44.66% on the SWEBench-Lite benchmark beating all other workflow-based approaches, and an absolute improvement of 7.66% compared to our baseline, which lacks such deep semantic understanding. We note that our approach performs particularly well on issues requiring multi-line reasoning (and editing) and edge-case handling, suggesting that incorporating issue and code semantics into APR pipelines can lead to robust and semantically consistent repairs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在下游软件工程任务如自动程序修复（APR）中展示了令人印象深刻的能力。特别是，已经有很多关于仓库级别问题解决基准（如SWE-Bench）的研究。尽管在该领域已经取得了显著进展，但我们注意到，在解决这些问题的过程中，现有的代理系统往往会过度关注立即可疑的代码行并单独修复它们，而缺乏对问题语义、代码语义或执行语义的深入理解。因此，许多现有系统生成的补丁过度拟合用户的问题，即使一个更通用的修复方案更佳。为解决这一局限，我们引入了SemAgent，一种新的基于工作流的过程，该过程利用问题、代码和执行语义生成全面的补丁，即识别并修复所有与问题相关的代码行。我们通过一个新颖的工作流实现这一点，该工作流包括：(a) 利用执行语义检索相关上下文，(b) 通过泛化抽象理解问题语义，(c) 嵌入抽象中，隔离代码语义，并在该理解基础上(d) 利用这一理解，在两阶段架构中实现一个修复阶段提出细粒度修复，其次是一个审阅阶段，根据推断的问题语义过滤相关修复。我们的评估表明，我们的方法在SWEBench-Lite基准上实现了44.66%的成功率，超过了所有其他基于工作流的方法，并且相对于缺乏此类深度语义理解的基础方法，绝对改进了7.66%。我们注意到，该方法在需要多行推理（和编辑）及边缘情况处理的问题上表现尤为出色，表明将问题和代码语义纳入APR管道可以实现鲁棒且语义一致的修复。 

---
# Long-Context Generalization with Sparse Attention 

**Title (ZH)**: 长上下文泛化与稀疏注意力 

**Authors**: Pavlo Vasylenko, Marcos Treviso, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2506.16640)  

**Abstract**: Transformer-based architectures traditionally employ softmax to compute attention weights, which produces dense distributions over all tokens in a sequence. While effective in many settings, this density has been shown to be detrimental for tasks that demand precise focus on fixed-size patterns: as sequence length increases, non-informative tokens accumulate attention probability mass, leading to dispersion and representational collapse. We show in this paper that sparse attention mechanisms using $\alpha$-entmax can avoid these issues, due to their ability to assign exact zeros to irrelevant tokens. Furthermore, we introduce Adaptive-Scalable Entmax (ASEntmax), which endows $\alpha$-entmax with a learnable temperature parameter, allowing the attention distribution to interpolate between sparse (pattern-focused) and dense (softmax-like) regimes. Finally, we show that the ability to locate and generalize fixed-size patterns can be further improved through a careful design of position encodings, which impacts both dense and sparse attention methods. By integrating ASEntmax into standard transformer layers alongside proper positional encodings, we show that our models greatly outperform softmax, scalable softmax, and fixed-temperature $\alpha$-entmax baselines on long-context generalization. 

**Abstract (ZH)**: 基于Transformer的稀疏注意力机制通过α-entmax实现精准焦点学习与表示改进 

---
# Latent Noise Injection for Private and Statistically Aligned Synthetic Data Generation 

**Title (ZH)**: 潜在噪声注入以生成私密且统计对齐的合成数据 

**Authors**: Rex Shen, Lu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2506.16636)  

**Abstract**: Synthetic Data Generation has become essential for scalable, privacy-preserving statistical analysis. While standard approaches based on generative models, such as Normalizing Flows, have been widely used, they often suffer from slow convergence in high-dimensional settings, frequently converging more slowly than the canonical $1/\sqrt{n}$ rate when approximating the true data distribution.
To overcome these limitations, we propose a Latent Noise Injection method using Masked Autoregressive Flows (MAF). Instead of directly sampling from the trained model, our method perturbs each data point in the latent space and maps it back to the data domain. This construction preserves a one to one correspondence between observed and synthetic data, enabling synthetic outputs that closely reflect the underlying distribution, particularly in challenging high-dimensional regimes where traditional sampling struggles.
Our procedure satisfies local $(\epsilon, \delta)$-differential privacy and introduces a single perturbation parameter to control the privacy-utility trade-off. Although estimators based on individual synthetic datasets may converge slowly, we show both theoretically and empirically that aggregating across $K$ studies in a meta analysis framework restores classical efficiency and yields consistent, reliable inference. We demonstrate that with a well-calibrated perturbation parameter, Latent Noise Injection achieves strong statistical alignment with the original data and robustness against membership inference attacks. These results position our method as a compelling alternative to conventional flow-based sampling for synthetic data sharing in decentralized and privacy-sensitive domains, such as biomedical research. 

**Abstract (ZH)**: 合成数据生成已成为实现可扩展性和隐私保护统计分析的关键。虽然基于生成模型的标准方法，如规范化流，已被广泛应用，但在高维设置中它们往往收敛速度较慢，往往慢于传统的约 $1/\sqrt{n}$ 率，用于近似真实的数据分布。

为了克服这些限制，我们提出了一种使用掩码自回归流（MAF）注入潜在噪声的方法。我们的方法不是直接从训练好的模型中采样，而是对潜在空间中的每个数据点进行扰动，并将其映射回数据域。这种构造在观察数据和合成数据之间保持了一对一的关系，使得合成输出能够密切反映潜在的真实分布，特别是在传统采样方法难以应对的高维挑战性环境中。

我们的过程满足局部 $(\epsilon, \delta)$-差分隐私，并引入单个扰动参数来控制隐私-效用权衡。虽然基于单一合成数据集的估计量可能收敛得较慢，但我们从理论上和实验上都证明，在元分析框架中聚合 $K$ 个研究可以恢复经典的效率，并生成一致可靠的推断结果。我们展示了通过合理校准扰动参数，潜在噪声注入能够实现与原始数据强大的统计对齐，并对成员推断攻击具有鲁棒性。这些结果使我们的方法成为在分散和隐私敏感领域，如生物医学研究中合成数据共享的强有力替代方法。 

---
# GeoGuess: Multimodal Reasoning based on Hierarchy of Visual Information in Street View 

**Title (ZH)**: GeoGuess：基于街道视图中视觉信息层次性的多模态推理 

**Authors**: Fenghua Cheng, Jinxiang Wang, Sen Wang, Zi Huang, Xue Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16633)  

**Abstract**: Multimodal reasoning is a process of understanding, integrating and inferring information across different data modalities. It has recently attracted surging academic attention as a benchmark for Artificial Intelligence (AI). Although there are various tasks for evaluating multimodal reasoning ability, they still have limitations. Lack of reasoning on hierarchical visual clues at different levels of granularity, e.g., local details and global context, is of little discussion, despite its frequent involvement in real scenarios. To bridge the gap, we introduce a novel and challenging task for multimodal reasoning, namely GeoGuess. Given a street view image, the task is to identify its location and provide a detailed explanation. A system that succeeds in GeoGuess should be able to detect tiny visual clues, perceive the broader landscape, and associate with vast geographic knowledge. Therefore, GeoGuess would require the ability to reason between hierarchical visual information and geographic knowledge. In this work, we establish a benchmark for GeoGuess by introducing a specially curated dataset GeoExplain which consists of panoramas-geocoordinates-explanation tuples. Additionally, we present a multimodal and multilevel reasoning method, namely SightSense which can make prediction and generate comprehensive explanation based on hierarchy of visual information and external knowledge. Our analysis and experiments demonstrate their outstanding performance in GeoGuess. 

**Abstract (ZH)**: 多模态推理是跨不同数据模态理解、整合和推断信息的过程。它 recently 吸引了人工智能领域的广泛关注。尽管有多样化的任务来评估多模态推理能力，它们仍然存在局限性。缺乏在不同粒度级别上对层次视觉线索进行推理的讨论，尽管这些线索在现实场景中经常出现。为弥补这一差距，我们介绍了一个新的具有挑战性的多模态推理任务，名为GeoGuess。给定一张街景图像，任务是识别其位置并提供详细的解释。一个在GeoGuess中成功系统的应该能够检测细微的视觉线索、感知更广阔的景观，并关联大量的地理知识。因此，GeoGuess 将需要在层次视觉信息和地理知识之间进行推理的能力。在本文中，我们通过引入一个特别策划的数据集GeoExplain来建立GeoGuess的基准，该数据集包含全景图-地理坐标-解释三元组。此外，我们提出了一种多模态和多层次推理方法，名为SightSense，它可以基于视觉信息的层次结构和外部知识进行预测和生成全面的解释。我们的分析和实验证明了SightSense在GeoGuess中的出色表现。 

---
# History-Augmented Vision-Language Models for Frontier-Based Zero-Shot Object Navigation 

**Title (ZH)**: 基于历史信息的视觉-语言模型在前沿导向的零样本物体导航中的应用 

**Authors**: Mobin Habibpour, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2506.16623)  

**Abstract**: Object Goal Navigation (ObjectNav) challenges robots to find objects in unseen environments, demanding sophisticated reasoning. While Vision-Language Models (VLMs) show potential, current ObjectNav methods often employ them superficially, primarily using vision-language embeddings for object-scene similarity checks rather than leveraging deeper reasoning. This limits contextual understanding and leads to practical issues like repetitive navigation behaviors. This paper introduces a novel zero-shot ObjectNav framework that pioneers the use of dynamic, history-aware prompting to more deeply integrate VLM reasoning into frontier-based exploration. Our core innovation lies in providing the VLM with action history context, enabling it to generate semantic guidance scores for navigation actions while actively avoiding decision loops. We also introduce a VLM-assisted waypoint generation mechanism for refining the final approach to detected objects. Evaluated on the HM3D dataset within Habitat, our approach achieves a 46% Success Rate (SR) and 24.8% Success weighted by Path Length (SPL). These results are comparable to state-of-the-art zero-shot methods, demonstrating the significant potential of our history-augmented VLM prompting strategy for more robust and context-aware robotic navigation. 

**Abstract (ZH)**: 基于对象的目标导航（ObjectNav）挑战机器人在未见环境中寻找物体，要求具备复杂的推理能力。尽管视觉-语言模型（VLMs）展现出潜力，当前的ObjectNav方法往往仅浅表地使用它们，主要通过视觉-语言嵌入进行物体-场景相似性检查，而未能充分利用深入的推理。这限制了上下文理解并导致重复的导航行为。本文介绍了一种新颖的零样本ObjectNav框架， pioneering 使用动态的历史感知提示以更深入地将VLM推理集成到前沿探索中。我们核心的创新在于为VLM提供动作历史上下文，使其能够生成导航动作的语义指导分数并主动避免决策循环。我们还引入了一种VLM辅助的航点生成机制，以细化对检测到物体的最终接近方式。在Habitat的HM3D数据集上进行评估，我们的方法实现了46%的成功率（SR）和24.8%的成功加权路径长度（SPL）。这些结果与最新的零样本方法相当，表明我们的历史增强VLM提示策略在实现更 robust 和上下文感知的机器人导航方面具有显著潜力。 

---
# Modeling Public Perceptions of Science in Media 

**Title (ZH)**: 媒体中公众对科学的认知 modeling 

**Authors**: Jiaxin Pei, Dustin Wright, Isabelle Augenstin, David Jurgens  

**Link**: [PDF](https://arxiv.org/pdf/2506.16622)  

**Abstract**: Effectively engaging the public with science is vital for fostering trust and understanding in our scientific community. Yet, with an ever-growing volume of information, science communicators struggle to anticipate how audiences will perceive and interact with scientific news. In this paper, we introduce a computational framework that models public perception across twelve dimensions, such as newsworthiness, importance, and surprisingness. Using this framework, we create a large-scale science news perception dataset with 10,489 annotations from 2,101 participants from diverse US and UK populations, providing valuable insights into public responses to scientific information across domains. We further develop NLP models that predict public perception scores with a strong performance. Leveraging the dataset and model, we examine public perception of science from two perspectives: (1) Perception as an outcome: What factors affect the public perception of scientific information? (2) Perception as a predictor: Can we use the estimated perceptions to predict public engagement with science? We find that individuals' frequency of science news consumption is the driver of perception, whereas demographic factors exert minimal influence. More importantly, through a large-scale analysis and carefully designed natural experiment on Reddit, we demonstrate that the estimated public perception of scientific information has direct connections with the final engagement pattern. Posts with more positive perception scores receive significantly more comments and upvotes, which is consistent across different scientific information and for the same science, but are framed differently. Overall, this research underscores the importance of nuanced perception modeling in science communication, offering new pathways to predict public interest and engagement with scientific content. 

**Abstract (ZH)**: 有效与公众开展科学交流对于培养公众对科学社区的信任和理解至关重要。然而，随着信息量的不断增加，科学传播者难以预见到受众如何感知和互动科学新闻。在本文中，我们引入了一种计算框架，该框架涵盖了十二个维度来建模公众的感知，如新闻价值、重要性和意外性。利用该框架，我们创建了一个包含10,489个注释的大规模科学新闻感知数据集，参与者来自多样化的美国和英国人口，提供了关于公众对科学信息的反应的有价值的见解。我们进一步开发了NLP模型，该模型在预测公众感知得分方面表现出色。依托数据集和模型，我们从两个视角探讨了公众对科学的感知：（1）感知作为结果：哪些因素会影响公众对科学信息的感知？（2）感知作为预测因子：我们能否利用估计的感知来预测公众对科学的兴趣和互动？研究发现，个人的科学新闻消费频率是感知的主要驱动因素，而人口统计学因素的影响较小。更重要的是，通过大规模分析和精心设计的Reddit自然实验，我们证明了估计的公众对科学信息的感知与最终的互动模式之间存在直接联系。获得更高感知得分的帖子收到了显著更多的评论和点赞，这种模式在不同科学信息和相同科学内容中均得到体现，但表达方式不同。总体而言，这项研究强调了在科学传播中进行精细感知建模的重要性，为预测公众对科学内容的兴趣和参与提供了新的途径。 

---
# Distribution Parameter Actor-Critic: Shifting the Agent-Environment Boundary for Diverse Action Spaces 

**Title (ZH)**: 分布参数演员-评论家：扩展行动空间中的智能体-环境边界 

**Authors**: Jiamin He, A. Rupam Mahmood, Martha White  

**Link**: [PDF](https://arxiv.org/pdf/2506.16608)  

**Abstract**: We introduce a novel reinforcement learning (RL) framework that treats distribution parameters as actions, redefining the boundary between agent and environment. This reparameterization makes the new action space continuous, regardless of the original action type (discrete, continuous, mixed, etc.). Under this new parameterization, we develop a generalized deterministic policy gradient estimator, Distribution Parameter Policy Gradient (DPPG), which has lower variance than the gradient in the original action space. Although learning the critic over distribution parameters poses new challenges, we introduce interpolated critic learning (ICL), a simple yet effective strategy to enhance learning, supported by insights from bandit settings. Building on TD3, a strong baseline for continuous control, we propose a practical DPPG-based actor-critic algorithm, Distribution Parameter Actor-Critic (DPAC). Empirically, DPAC outperforms TD3 in MuJoCo continuous control tasks from OpenAI Gym and DeepMind Control Suite, and demonstrates competitive performance on the same environments with discretized action spaces. 

**Abstract (ZH)**: 我们介绍了一种新颖的强化学习（RL）框架，将分布参数视为动作，并重新定义了代理与环境之间的边界。这种重新参数化使得新的动作空间连续，而无论原始动作类型（离散、连续、混合等）如何。在这一新的参数化下，我们开发了一种广义的确定性策略梯度估计器，分布参数策略梯度（DPPG），其方差低于原始动作空间中的梯度。尽管在分布参数上学习批评家提出了新的挑战，我们提出了一种插值批评家学习（ICL）的简单而有效的策略来增强学习，其灵感来源于拉姆达 bandit 设置。在连续控制的强基线 TD3 的基础上，我们提出了一种实用的基于 DPPG 的 actor-critic 算法，分布参数 actor-critic（DPAC）。实验证明，DPAC 在 OpenAI Gym 和 DeepMind Control Suite 的 MuJoCo 连续控制任务中表现优于 TD3，并且在具有离散化动作空间的相同环境中展示了竞争力。 

---
# FLAME: Towards Federated Fine-Tuning Large Language Models Through Adaptive SMoE 

**Title (ZH)**: FLAME: 向量化 Federated 细粒度调优大型语言模型的自适应 SMoE 

**Authors**: Khiem Le, Tuan Tran, Ting Hua, Nitesh V. Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2506.16600)  

**Abstract**: Existing resource-adaptive LoRA federated fine-tuning methods enable clients to fine-tune models using compressed versions of global LoRA matrices, in order to accommodate various compute resources across clients. This compression requirement will lead to suboptimal performance due to information loss. To address this, we propose FLAME, a novel federated learning framework based on the Sparse Mixture-of-Experts (SMoE) architecture. Unlike prior approaches, FLAME retains full (uncompressed) global LoRA matrices and achieves client-side adaptability by varying the number of activated experts per client. However, incorporating SMoE into federated learning introduces unique challenges, specifically, the mismatch in output magnitude from partial expert activation and the imbalance in expert training quality across clients. FLAME tackles these challenges through a lightweight rescaling mechanism and an activation-aware aggregation scheme. Empirical results across diverse computational settings demonstrate that FLAME consistently outperforms existing methods, providing a robust and effective solution for resource-adaptive federated learning. 

**Abstract (ZH)**: FLAME：一种基于稀疏混合专家架构的新型联邦学习框架 

---
# Hybrid Attention Network for Accurate Breast Tumor Segmentation in Ultrasound Images 

**Title (ZH)**: 超声图像中乳腺肿瘤分割的混合注意力网络 

**Authors**: Muhammad Azeem Aslam, Asim Naveed, Nisar Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.16592)  

**Abstract**: Breast ultrasound imaging is a valuable tool for early breast cancer detection, but automated tumor segmentation is challenging due to inherent noise, variations in scale of lesions, and fuzzy boundaries. To address these challenges, we propose a novel hybrid attention-based network for lesion segmentation. Our proposed architecture integrates a pre-trained DenseNet121 in the encoder part for robust feature extraction with a multi-branch attention-enhanced decoder tailored for breast ultrasound images. The bottleneck incorporates Global Spatial Attention (GSA), Position Encoding (PE), and Scaled Dot-Product Attention (SDPA) to learn global context, spatial relationships, and relative positional features. The Spatial Feature Enhancement Block (SFEB) is embedded at skip connections to refine and enhance spatial features, enabling the network to focus more effectively on tumor regions. A hybrid loss function combining Binary Cross-Entropy (BCE) and Jaccard Index loss optimizes both pixel-level accuracy and region-level overlap metrics, enhancing robustness to class imbalance and irregular tumor shapes. Experiments on public datasets demonstrate that our method outperforms existing approaches, highlighting its potential to assist radiologists in early and accurate breast cancer diagnosis. 

**Abstract (ZH)**: 基于注意力机制的混合网络在乳腺超声图像中肿块分割中的应用：一种克服固有噪声、病变尺度变化和模糊边界挑战的新方法 

---
# Energy-Based Transfer for Reinforcement Learning 

**Title (ZH)**: 基于能量的迁移强化学习 

**Authors**: Zeyun Deng, Jasorsi Ghosh, Fiona Xie, Yuzhe Lu, Katia Sycara, Joseph Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2506.16590)  

**Abstract**: Reinforcement learning algorithms often suffer from poor sample efficiency, making them challenging to apply in multi-task or continual learning settings. Efficiency can be improved by transferring knowledge from a previously trained teacher policy to guide exploration in new but related tasks. However, if the new task sufficiently differs from the teacher's training task, the transferred guidance may be sub-optimal and bias exploration toward low-reward behaviors. We propose an energy-based transfer learning method that uses out-of-distribution detection to selectively issue guidance, enabling the teacher to intervene only in states within its training distribution. We theoretically show that energy scores reflect the teacher's state-visitation density and empirically demonstrate improved sample efficiency and performance across both single-task and multi-task settings. 

**Abstract (ZH)**: 强化学习算法往往 Sample Efficiency 较差，这使得它们在多任务或持续学习环境中应用起来颇具挑战。通过将先前训练的教师策略的知识转移到新但相关的任务中以指导探索，可以提高效率。然而，如果新任务与教师的训练任务相差足够大，转移的指导可能会变得次优，并倾向于引导探索低奖励行为。我们提出了一种基于能量的迁移学习方法，利用离群检测选择性地发布指导，使教师仅在其实训练分布内的状态下干预。理论上，我们证明了能量分数反映了教师的状态访问密度，并通过单任务和多任务设置的实验展示了样本效率和性能的提升。 

---
# Spatially-Aware Evaluation of Segmentation Uncertainty 

**Title (ZH)**: 空间感知分割不确定性评估 

**Authors**: Tal Zeevi, Eléonore V. Lieffrig, Lawrence H. Staib, John A. Onofrey  

**Link**: [PDF](https://arxiv.org/pdf/2506.16589)  

**Abstract**: Uncertainty maps highlight unreliable regions in segmentation predictions. However, most uncertainty evaluation metrics treat voxels independently, ignoring spatial context and anatomical structure. As a result, they may assign identical scores to qualitatively distinct patterns (e.g., scattered vs. boundary-aligned uncertainty). We propose three spatially aware metrics that incorporate structural and boundary information and conduct a thorough validation on medical imaging data from the prostate zonal segmentation challenge within the Medical Segmentation Decathlon. Our results demonstrate improved alignment with clinically important factors and better discrimination between meaningful and spurious uncertainty patterns. 

**Abstract (ZH)**: 空间感知不确定性度量突显分割预测中的不可靠区域，并考虑结构和边界信息， medical imaging data from the prostate zonal segmentation challenge within the Medical Segmentation Decathlon上的验证表明，这些度量能更好地与临床重要因素对齐，并能更有效地区分有意义和虚假的不确定性模式。 

---
# AI-Driven Tools in Modern Software Quality Assurance: An Assessment of Benefits, Challenges, and Future Directions 

**Title (ZH)**: AI驱动的工具在现代软件质量保证中的作用：优势、挑战及未来方向 

**Authors**: Ihor Pysmennyi, Roman Kyslyi, Kyrylo Kleshch  

**Link**: [PDF](https://arxiv.org/pdf/2506.16586)  

**Abstract**: Traditional quality assurance (QA) methods face significant challenges in addressing the complexity, scale, and rapid iteration cycles of modern software systems and are strained by limited resources available, leading to substantial costs associated with poor quality. The object of this research is the Quality Assurance processes for modern distributed software applications. The subject of the research is the assessment of the benefits, challenges, and prospects of integrating modern AI-oriented tools into quality assurance processes. We performed comprehensive analysis of implications on both verification and validation processes covering exploratory test analyses, equivalence partitioning and boundary analyses, metamorphic testing, finding inconsistencies in acceptance criteria (AC), static analyses, test case generation, unit test generation, test suit optimization and assessment, end to end scenario execution. End to end regression of sample enterprise application utilizing AI-agents over generated test scenarios was implemented as a proof of concept highlighting practical use of the study. The results, with only 8.3% flaky executions of generated test cases, indicate significant potential for the proposed approaches. However, the study also identified substantial challenges for practical adoption concerning generation of semantically identical coverage, "black box" nature and lack of explainability from state-of-the-art Large Language Models (LLMs), the tendency to correct mutated test cases to match expected results, underscoring the necessity for thorough verification of both generated artifacts and test execution results. The research demonstrates AI's transformative potential for QA but highlights the importance of a strategic approach to implementing these technologies, considering the identified limitations and the need for developing appropriate verification methodologies. 

**Abstract (ZH)**: 现代分布式软件应用的质量保证过程中的AI工具整合及其挑战和前景研究 

---
# Measuring (a Sufficient) World Model in LLMs: A Variance Decomposition Framework 

**Title (ZH)**: 测量（充分的）世界模型在LLMs中的方差分解框架 

**Authors**: Nadav Kunievsky, James A. Evans  

**Link**: [PDF](https://arxiv.org/pdf/2506.16584)  

**Abstract**: Understanding whether large language models (LLMs) possess a world model-a structured understanding of the world that supports generalization beyond surface-level patterns-is central to assessing their reliability, especially in high-stakes applications. We propose a formal framework for evaluating whether an LLM exhibits a sufficiently robust world model, defined as producing consistent outputs across semantically equivalent prompts while distinguishing between prompts that express different intents. We introduce a new evaluation approach to measure this that decomposes model response variability into three components: variability due to user purpose, user articulation, and model instability. An LLM with a strong world model should attribute most of the variability in its responses to changes in foundational purpose rather than superficial changes in articulation. This approach allows us to quantify how much of a model's behavior is semantically grounded rather than driven by model instability or alternative wording. We apply this framework to evaluate LLMs across diverse domains. Our results show how larger models attribute a greater share of output variability to changes in user purpose, indicating a more robust world model. This improvement is not uniform, however: larger models do not consistently outperform smaller ones across all domains, and their advantage in robustness is often modest. These findings highlight the importance of moving beyond accuracy-based benchmarks toward semantic diagnostics that more directly assess the structure and stability of a model's internal understanding of the world. 

**Abstract (ZH)**: 理解大型语言模型是否具备世界模型——一种支持超越表面模式泛化的结构化世界理解——对于评估其可靠性，特别是在高风险应用中，至关重要。我们提出了一种正式框架来评估大型语言模型是否表现出足够稳健的世界模型，定义为在语义等价的提示下产生一致的输出，并能够区分表达不同意图的提示。我们引入了一种新的评估方法来衡量这一点，该方法将模型响应的变化分解为三个组成部分：由于用户目的、用户表达和模型不稳定性的变化。一个具备强大世界模型的大型语言模型应主要将其响应变化归因于基础目的的变化，而不是表面表达的变化。这种方法使我们能够量化模型行为中多少部分是基于语义，而不是由模型不稳定或替代措辞驱动的。我们将这一框架应用于跨不同领域的大型语言模型评估。结果显示，较大的模型将更多输出变化归因于用户目的的变化，表明其具备更稳健的世界模型。然而，这种改进并非一致：较大的模型并不始终在所有领域中优于较小的模型，其在鲁棒性方面的优势通常较小。这些发现突显了超越基于准确性的基准，转向更直接评估模型内在世界理解结构和稳定性的语义诊断的重要性。 

---
# Reimagination with Test-time Observation Interventions: Distractor-Robust World Model Predictions for Visual Model Predictive Control 

**Title (ZH)**: 基于测试时观测干预的重塑：视觉模型预测控制中的干扰物鲁棒世界模型预测 

**Authors**: Yuxin Chen, Jianglan Wei, Chenfeng Xu, Boyi Li, Masayoshi Tomizuka, Andrea Bajcsy, Ran Tian  

**Link**: [PDF](https://arxiv.org/pdf/2506.16565)  

**Abstract**: World models enable robots to "imagine" future observations given current observations and planned actions, and have been increasingly adopted as generalized dynamics models to facilitate robot learning. Despite their promise, these models remain brittle when encountering novel visual distractors such as objects and background elements rarely seen during training. Specifically, novel distractors can corrupt action outcome predictions, causing downstream failures when robots rely on the world model imaginations for planning or action verification. In this work, we propose Reimagination with Observation Intervention (ReOI), a simple yet effective test-time strategy that enables world models to predict more reliable action outcomes in open-world scenarios where novel and unanticipated visual distractors are inevitable. Given the current robot observation, ReOI first detects visual distractors by identifying which elements of the scene degrade in physically implausible ways during world model prediction. Then, it modifies the current observation to remove these distractors and bring the observation closer to the training distribution. Finally, ReOI "reimagines" future outcomes with the modified observation and reintroduces the distractors post-hoc to preserve visual consistency for downstream planning and verification. We validate our approach on a suite of robotic manipulation tasks in the context of action verification, where the verifier needs to select desired action plans based on predictions from a world model. Our results show that ReOI is robust to both in-distribution and out-of-distribution visual distractors. Notably, it improves task success rates by up to 3x in the presence of novel distractors, significantly outperforming action verification that relies on world model predictions without imagination interventions. 

**Abstract (ZH)**: Reimagination with Observation Intervention for Robust Action Outcome Prediction in Open-World Scenarios 

---
# From Semantic To Instance: A Semi-Self-Supervised Learning Approach 

**Title (ZH)**: 从语义到实例：一种半自监督学习方法 

**Authors**: Keyhan Najafian, Farhad Maleki, Lingling Jin, Ian Stavness  

**Link**: [PDF](https://arxiv.org/pdf/2506.16563)  

**Abstract**: Instance segmentation is essential for applications such as automated monitoring of plant health, growth, and yield. However, extensive effort is required to create large-scale datasets with pixel-level annotations of each object instance for developing instance segmentation models that restrict the use of deep learning in these areas. This challenge is more significant in images with densely packed, self-occluded objects, which are common in agriculture. To address this challenge, we propose a semi-self-supervised learning approach that requires minimal manual annotation to develop a high-performing instance segmentation model. We design GLMask, an image-mask representation for the model to focus on shape, texture, and pattern while minimizing its dependence on color features. We develop a pipeline to generate semantic segmentation and then transform it into instance-level segmentation. The proposed approach substantially outperforms the conventional instance segmentation models, establishing a state-of-the-art wheat head instance segmentation model with mAP@50 of 98.5%. Additionally, we assessed the proposed methodology on the general-purpose Microsoft COCO dataset, achieving a significant performance improvement of over 12.6% mAP@50. This highlights that the utility of our proposed approach extends beyond precision agriculture and applies to other domains, specifically those with similar data characteristics. 

**Abstract (ZH)**: 基于实例的分割对于植物健康、生长和产量的自动化监控等应用至关重要。然而，为了开发实例分割模型，需要大量带有像素级标注的实例数据集，这限制了深度学习在这些领域的应用。这一挑战在密集堆叠且相互遮挡的物体图像中尤为显著，而这类图像在农业中非常常见。为应对这一挑战，我们提出了一种半自监督学习方法，该方法只需少量手动标注即可训练高性能的实例分割模型。我们设计了GLMask，该模型的图像-掩膜表示专注于形状、纹理和模式，同时减少对颜色特征的依赖。我们开发了一个流水线，用于生成语义分割，然后将其转换为实例级分割。所提出的方法显著优于传统的实例分割模型，建立了基于mAP@50为98.5%的高性能小麦穗实例分割模型。此外，我们在通用的Microsoft COCO数据集上评估了所提出的方法，获得了超过12.6%的mAP@50性能提升。这表明我们提出的方法不仅适用于精确农业，还可以应用于其他具有类似数据特征的领域。 

---
# One Sample is Enough to Make Conformal Prediction Robust 

**Title (ZH)**: 一个样本足以使一致预测 robust 

**Authors**: Soroush H. Zargarbashi, Mohammad Sadegh Akhondzadeh, Aleksandar Bojchevski  

**Link**: [PDF](https://arxiv.org/pdf/2506.16553)  

**Abstract**: Given any model, conformal prediction (CP) returns prediction sets guaranteed to include the true label with high adjustable probability. Robust CP (RCP) extends this to inputs with worst-case noise. A well-established approach is to use randomized smoothing for RCP since it is applicable to any black-box model and provides smaller sets compared to deterministic methods. However, current smoothing-based RCP requires many model forward passes per each input which is computationally expensive. We show that conformal prediction attains some robustness even with a forward pass on a single randomly perturbed input. Using any binary certificate we propose a single sample robust CP (RCP1). Our approach returns robust sets with smaller average set size compared to SOTA methods which use many (e.g. around 100) passes per input. Our key insight is to certify the conformal prediction procedure itself rather than individual scores. Our approach is agnostic to the setup (classification and regression). We further extend our approach to smoothing-based robust conformal risk control. 

**Abstract (ZH)**: 给定任何模型，可信预测（CP）返回保证包含真实标签的预测集，且概率可以调整。鲁棒可信预测（RCP）将这一保证扩展到具有最坏情况噪声的输入。一种已确立的方法是使用随机平滑进行RCP，因为它适用于任何黑盒模型，并且能提供比确定性方法更小的预测集。然而，当前基于平滑的方法需要对每个输入进行多次模型前向传递，这在计算上是昂贵的。我们展示了即使对单个随机扰动输入进行一次前向传递，可信预测也能获得一些鲁棒性。使用任何二元证书，我们提出了一次样本鲁棒可信预测（RCP1）。我们的方法与使用多个（例如约100个）输入的当前最佳方法相比，返回具有更小平均集大小的鲁棒预测集。我们的关键见解是认证可信预测过程本身而非单个分数。我们的方法对设置（分类和回归）是无偏的。我们进一步将该方法扩展到基于平滑的鲁棒可信风险控制。 

---
# BIDA: A Bi-level Interaction Decision-making Algorithm for Autonomous Vehicles in Dynamic Traffic Scenarios 

**Title (ZH)**: BIDA：动态交通场景中自主车辆多级交互决策算法 

**Authors**: Liyang Yu, Tianyi Wang, Junfeng Jiao, Fengwu Shan, Hongqing Chu, Bingzhao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16546)  

**Abstract**: In complex real-world traffic environments, autonomous vehicles (AVs) need to interact with other traffic participants while making real-time and safety-critical decisions accordingly. The unpredictability of human behaviors poses significant challenges, particularly in dynamic scenarios, such as multi-lane highways and unsignalized T-intersections. To address this gap, we design a bi-level interaction decision-making algorithm (BIDA) that integrates interactive Monte Carlo tree search (MCTS) with deep reinforcement learning (DRL), aiming to enhance interaction rationality, efficiency and safety of AVs in dynamic key traffic scenarios. Specifically, we adopt three types of DRL algorithms to construct a reliable value network and policy network, which guide the online deduction process of interactive MCTS by assisting in value update and node selection. Then, a dynamic trajectory planner and a trajectory tracking controller are designed and implemented in CARLA to ensure smooth execution of planned maneuvers. Experimental evaluations demonstrate that our BIDA not only enhances interactive deduction and reduces computational costs, but also outperforms other latest benchmarks, which exhibits superior safety, efficiency and interaction rationality under varying traffic conditions. 

**Abstract (ZH)**: 在复杂的现实交通环境中，自动驾驶车辆（AVs）需要在实时和安全性关键的决策中与其他交通参与者进行互动。人类行为的不可预测性在动态场景中，如多车道高速公路和无信号T型交叉口，提出了重大挑战。为了解决这个问题，我们设计了一种双层互动决策算法（BIDA），将互动蒙特卡洛树搜索（MCTS）与深度强化学习（DRL）相结合，旨在增强AVs在动态关键交通场景中的互动合理性、效率和安全性。具体来说，我们采用三种类型的DRL算法构建一个可靠的值网络和策略网络，通过协助价值更新和节点选择，引导互动MCTS的在线推断过程。然后，在CARLA中设计并实现了动态轨迹规划器和轨迹跟踪控制器，以确保计划机动的平滑执行。实验评估表明，我们的BIDA不仅提高了互动推断能力并降低了计算成本，而且在各种交通条件下的性能优于其他最新基准，展现出更优的安全性、效率和互动合理性。 

---
# Subspace-Boosted Model Merging 

**Title (ZH)**: 子空间增强模型融合 

**Authors**: Ronald Skorobogat, Karsten Roth, Mariana-Iuliana Georgescu, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2506.16506)  

**Abstract**: Model merging enables the combination of multiple specialized expert models into a single model capable of performing multiple tasks. However, the benefits of merging an increasing amount of specialized experts generally lead to diminishing returns and reduced overall performance gains. In this work, we offer an explanation and analysis from a task arithmetic perspective; revealing that as the merging process (across numerous existing merging methods) continues for more and more experts, the associated task vector space experiences rank collapse. To mitigate this issue, we introduce Subspace Boosting, which operates on the singular value decomposed task vector space and maintains task vector ranks. Subspace Boosting raises merging efficacy for up to 20 expert models by large margins of more than 10% when evaluated on vision benchmarks. Moreover, we propose employing Higher-Order Generalized Singular Value Decomposition to further quantify task similarity, offering a new interpretable perspective on model merging. 

**Abstract (ZH)**: 模型合并使得多个专业专家模型能够整合成一个能够执行多项任务的单一模型。然而，合并越来越多的专业专家通常会导致收益递减和整体性能改进的减少。在本研究中，我们从任务算术的角度提供了解释和分析；揭示出随着合并过程（跨越多种现有合并方法）的进行，涉及的任务向量空间经历秩崩溃。为解决这一问题，我们引入了子空间增强方法，该方法在奇异值分解的任务向量空间上操作，并保持任务向量的秩。子空间增强在视觉基准测试中将最多20个专家模型的合并效率大幅提升超过10%。此外，我们提出了使用高阶广义奇异值分解进一歩量化任务相似性，提供一种可解释的模型合并新视角。 

---
# Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Ultimate Details 

**Title (ZH)**: 混沌3D 2.5：迈向极致细节的高保真3D资产生成 

**Authors**: Zeqiang Lai, Yunfei Zhao, Haolin Liu, Zibo Zhao, Qingxiang Lin, Huiwen Shi, Xianghui Yang, Mingxin Yang, Shuhui Yang, Yifei Feng, Sheng Zhang, Xin Huang, Di Luo, Fan Yang, Fang Yang, Lifu Wang, Sicong Liu, Yixuan Tang, Yulin Cai, Zebin He, Tian Liu, Yuhong Liu, Jie Jiang, Linus, Jingwei Huang, Chunchao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.16504)  

**Abstract**: In this report, we present Hunyuan3D 2.5, a robust suite of 3D diffusion models aimed at generating high-fidelity and detailed textured 3D assets. Hunyuan3D 2.5 follows two-stages pipeline of its previous version Hunyuan3D 2.0, while demonstrating substantial advancements in both shape and texture generation. In terms of shape generation, we introduce a new shape foundation model -- LATTICE, which is trained with scaled high-quality datasets, model-size, and compute. Our largest model reaches 10B parameters and generates sharp and detailed 3D shape with precise image-3D following while keeping mesh surface clean and smooth, significantly closing the gap between generated and handcrafted 3D shapes. In terms of texture generation, it is upgraded with phyiscal-based rendering (PBR) via a novel multi-view architecture extended from Hunyuan3D 2.0 Paint model. Our extensive evaluation shows that Hunyuan3D 2.5 significantly outperforms previous methods in both shape and end-to-end texture generation. 

**Abstract (ZH)**: 本报告介绍了Hunyuan3D 2.5，这是一个稳健的3D扩散模型套件，旨在生成高保真度和详细纹理的3D资产。Hunyuan3D 2.5沿用了其前一个版本Hunyuan3D 2.0的两阶段管道，同时在形状和纹理生成方面取得了显著进步。在形状生成方面，我们引入了一个新的形状基础模型——LATTICE，该模型使用放缩后的高质量数据集、模型大小和计算资源进行训练。我们的最大模型达到10B参数，并生成了清晰且细节丰富的3D形状，保真度高，同时保持了网格表面的清洁和平滑，显著缩小了生成与手工制作的3D形状之间的差距。在纹理生成方面，通过从Hunyuan3D 2.0 Paint模型扩展出的新颖多视角架构引入基于物理的渲染（PBR）。广泛的评估表明，Hunyuan3D 2.5在形状和端到端纹理生成方面显著优于以前的方法。 

---
# Relic: Enhancing Reward Model Generalization for Low-Resource Indic Languages with Few-Shot Examples 

**Title (ZH)**: 遗物：通过少量示例增强奖励模型泛化能力以适用于低资源indic语言 

**Authors**: Soumya Suvra Ghosal, Vaibhav Singh, Akash Ghosh, Soumyabrata Pal, Subhadip Baidya, Sriparna Saha, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2506.16502)  

**Abstract**: Reward models are essential for aligning large language models (LLMs) with human preferences. However, most open-source multilingual reward models are primarily trained on preference datasets in high-resource languages, resulting in unreliable reward signals for low-resource Indic languages. Collecting large-scale, high-quality preference data for these languages is prohibitively expensive, making preference-based training approaches impractical. To address this challenge, we propose RELIC, a novel in-context learning framework for reward modeling in low-resource Indic languages. RELIC trains a retriever with a pairwise ranking objective to select in-context examples from auxiliary high-resource languages that most effectively highlight the distinction between preferred and less-preferred responses. Extensive experiments on three preference datasets- PKU-SafeRLHF, WebGPT, and HH-RLHF-using state-of-the-art open-source reward models demonstrate that RELIC significantly improves reward model accuracy for low-resource Indic languages, consistently outperforming existing example selection methods. For example, on Bodo-a low-resource Indic language-using a LLaMA-3.2-3B reward model, RELIC achieves a 12.81% and 10.13% improvement in accuracy over zero-shot prompting and state-of-the-art example selection method, respectively. 

**Abstract (ZH)**: 低资源印地语语言奖励模型的新型上下文学习框架RELIC 

---
# Spotting tell-tale visual artifacts in face swapping videos: strengths and pitfalls of CNN detectors 

**Title (ZH)**: 检测人脸换脸视频中的 tell-tale 视觉 artefacts：CNN 检测器的优势与局限 

**Authors**: Riccardo Ziglio, Cecilia Pasquini, Silvio Ranise  

**Link**: [PDF](https://arxiv.org/pdf/2506.16497)  

**Abstract**: Face swapping manipulations in video streams represents an increasing threat in remote video communications, due to advances
in automated and real-time tools. Recent literature proposes to characterize and exploit visual artifacts introduced in video frames
by swapping algorithms when dealing with challenging physical scenes, such as face occlusions. This paper investigates the
effectiveness of this approach by benchmarking CNN-based data-driven models on two data corpora (including a newly collected
one) and analyzing generalization capabilities with respect to different acquisition sources and swapping algorithms. The results
confirm excellent performance of general-purpose CNN architectures when operating within the same data source, but a significant
difficulty in robustly characterizing occlusion-based visual cues across datasets. This highlights the need for specialized detection
strategies to deal with such artifacts. 

**Abstract (ZH)**: 视频流中的面部置换操作对远程视频通信构成了日益严重的威胁，由于自动化和实时工具的发展。近期文献提出，在处理复杂的物理场景（如面部遮挡）时，可以通过表征和利用面部置换算法在视频帧中引入的视觉artifact来应对这一挑战。本文通过在两个数据集（包括一个新收集的数据集）上基准测试基于CNN的数据驱动模型，并分析其在不同采集源和置换算法下的泛化能力，来探讨该方法的有效性。结果表明，通用的CNN架构在同一数据源下表现出色，但在跨数据集表征基于遮挡的视觉线索方面面临显著困难。这强调了需要专门的检测策略来应对这些artifact。 

---
# Grounding Language Models with Semantic Digital Twins for Robotic Planning 

**Title (ZH)**: 基于语义数字孪生的语言模型在机器人规划中的应用 

**Authors**: Mehreen Naeem, Andrew Melnik, Michael Beetz  

**Link**: [PDF](https://arxiv.org/pdf/2506.16493)  

**Abstract**: We introduce a novel framework that integrates Semantic Digital Twins (SDTs) with Large Language Models (LLMs) to enable adaptive and goal-driven robotic task execution in dynamic environments. The system decomposes natural language instructions into structured action triplets, which are grounded in contextual environmental data provided by the SDT. This semantic grounding allows the robot to interpret object affordances and interaction rules, enabling action planning and real-time adaptability. In case of execution failures, the LLM utilizes error feedback and SDT insights to generate recovery strategies and iteratively revise the action plan. We evaluate our approach using tasks from the ALFRED benchmark, demonstrating robust performance across various household scenarios. The proposed framework effectively combines high-level reasoning with semantic environment understanding, achieving reliable task completion in the face of uncertainty and failure. 

**Abstract (ZH)**: 我们提出一种将语义数字孪生（SDT）与大型语言模型（LLM）集成的新框架，以在动态环境中实现适应性和目标驱动的机器人任务执行。该系统将自然语言指令分解为结构化的动作三元组，这些三元组基于SDT提供的上下文环境数据进行语义绑定。这种语义绑定使机器人能够解释物体功能和交互规则，从而实现动作规划和实时适应。在执行失败时，LLM利用错误反馈和SDT的见解生成恢复策略，并迭代地修改动作计划。我们使用ALFRED基准中的任务对我们的方法进行评估，展示了在各种家庭场景中的稳健性能。所提出框架有效地结合了高层次推理与语义环境理解，能够在不确定性与失败面前实现可靠的任务完成。 

---
# Towards Generalizable Generic Harmful Speech Datasets for Implicit Hate Speech Detection 

**Title (ZH)**: 通用型泛化有害言论数据集以促进隐性仇恨言论检测 

**Authors**: Saad Almohaimeed, Saleh Almohaimeed, Damla Turgut, Ladislau Bölöni  

**Link**: [PDF](https://arxiv.org/pdf/2506.16476)  

**Abstract**: Implicit hate speech has recently emerged as a critical challenge for social media platforms. While much of the research has traditionally focused on harmful speech in general, the need for generalizable techniques to detect veiled and subtle forms of hate has become increasingly pressing. Based on lexicon analysis, we hypothesize that implicit hate speech is already present in publicly available harmful speech datasets but may not have been explicitly recognized or labeled by annotators. Additionally, crowdsourced datasets are prone to mislabeling due to the complexity of the task and often influenced by annotators' subjective interpretations. In this paper, we propose an approach to address the detection of implicit hate speech and enhance generalizability across diverse datasets by leveraging existing harmful speech datasets. Our method comprises three key components: influential sample identification, reannotation, and augmentation using Llama-3 70B and GPT-4o. Experimental results demonstrate the effectiveness of our approach in improving implicit hate detection, achieving a +12.9-point F1 score improvement compared to the baseline. 

**Abstract (ZH)**: 隐含仇恨言论近年来已成为社交媒体平台面临的关键挑战。虽然大多数研究传统上集中在一般有害言论上，但检测隐蔽和微妙的仇恨言论的通用技术需求日益紧迫。基于词典分析，我们假设隐含仇恨言论已经存在于公开可用的有害言论数据集中，但可能未被标注者明确识别或标注。此外，基于众包的数据集容易因任务复杂性和标注者主观解释的影响而出现误标。在本文中，我们提出了一种方法，通过利用现有有害言论数据集来解决隐含仇恨言论的检测问题并提高跨不同数据集的一般化能力。该方法包括三个关键组件：有影响力样本的识别、重新标注和使用Llama-3 70B和GPT-4o的数据增强。实验结果表明，与基线相比，我们的方法在提高隐含仇恨言论检测效果方面取得了12.9点的F1分数提升。 

---
# Human2LocoMan: Learning Versatile Quadrupedal Manipulation with Human Pretraining 

**Title (ZH)**: Human2LocoMan: 人体预训练下的多功能四足Manipulation学习 

**Authors**: Yaru Niu, Yunzhe Zhang, Mingyang Yu, Changyi Lin, Chenhao Li, Yikai Wang, Yuxiang Yang, Wenhao Yu, Tingnan Zhang, Bingqing Chen, Jonathan Francis, Zhenzhen Li, Jie Tan, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16475)  

**Abstract**: Quadrupedal robots have demonstrated impressive locomotion capabilities in complex environments, but equipping them with autonomous versatile manipulation skills in a scalable way remains a significant challenge. In this work, we introduce a cross-embodiment imitation learning system for quadrupedal manipulation, leveraging data collected from both humans and LocoMan, a quadruped equipped with multiple manipulation modes. Specifically, we develop a teleoperation and data collection pipeline, which unifies and modularizes the observation and action spaces of the human and the robot. To effectively leverage the collected data, we propose an efficient modularized architecture that supports co-training and pretraining on structured modality-aligned data across different embodiments. Additionally, we construct the first manipulation dataset for the LocoMan robot, covering various household tasks in both unimanual and bimanual modes, supplemented by a corresponding human dataset. We validate our system on six real-world manipulation tasks, where it achieves an average success rate improvement of 41.9% overall and 79.7% under out-of-distribution (OOD) settings compared to the baseline. Pretraining with human data contributes a 38.6% success rate improvement overall and 82.7% under OOD settings, enabling consistently better performance with only half the amount of robot data. Our code, hardware, and data are open-sourced at: this https URL. 

**Abstract (ZH)**: 四足机器人在复杂环境中的运动能力令人印象深刻，但以可扩展的方式为其配备自主通用操作技能仍然是一个重大挑战。在本工作中，我们引入了一种跨体态模仿学习系统，利用来自人类和LocoMan（一种配备了多种操作模式的四足机器人）的数据。具体来说，我们开发了一种远程操作和数据收集流水线，将人类和机器人的观察空间和行动空间统一和模块化。为了有效利用收集的数据，我们提出了一种高效的模块化架构，支持在不同体态的结构化模态对齐数据上进行联合训练和预训练。此外，我们为LocoMan机器人构建了首个操作数据集，涵盖了多种单手和双手家庭任务，并补充了相应的人类数据集。我们在六个真实世界的操作任务上验证了我们的系统，整体成功率提高了41.9%，在分布外（OOD）设置下提高了79.7%，使用人类数据预训练的整体成功率提高了38.6%，在OOD设置下提高了82.7%，仅使用一半的机器人数据就实现了持续的更好性能。我们的代码、硬件和数据已开源：this https URL。 

---
# Do We Talk to Robots Like Therapists, and Do They Respond Accordingly? Language Alignment in AI Emotional Support 

**Title (ZH)**: 我们像对待治疗师一样与机器人交谈，它们也会相应地回应吗？AI情感支持中的语言对齐 

**Authors**: Sophie Chiang, Guy Laban, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2506.16473)  

**Abstract**: As conversational agents increasingly engage in emotionally supportive dialogue, it is important to understand how closely their interactions resemble those in traditional therapy settings. This study investigates whether the concerns shared with a robot align with those shared in human-to-human (H2H) therapy sessions, and whether robot responses semantically mirror those of human therapists. We analyzed two datasets: one of interactions between users and professional therapists (Hugging Face's NLP Mental Health Conversations), and another involving supportive conversations with a social robot (QTrobot from LuxAI) powered by a large language model (LLM, GPT-3.5). Using sentence embeddings and K-means clustering, we assessed cross-agent thematic alignment by applying a distance-based cluster-fitting method that evaluates whether responses from one agent type map to clusters derived from the other, and validated it using Euclidean distances. Results showed that 90.88% of robot conversation disclosures could be mapped to clusters from the human therapy dataset, suggesting shared topical structure. For matched clusters, we compared the subjects as well as therapist and robot responses using Transformer, Word2Vec, and BERT embeddings, revealing strong semantic overlap in subjects' disclosures in both datasets, as well as in the responses given to similar human disclosure themes across agent types (robot vs. human therapist). These findings highlight both the parallels and boundaries of robot-led support conversations and their potential for augmenting mental health interventions. 

**Abstract (ZH)**: 随着对话代理越来越多地参与情感支持对话，了解其互动与传统治疗环境中互动的相似性变得尤为重要。本研究探讨了与机器人分享的顾虑是否与人对人（H2H） therapy会话中分享的顾虑一致，以及机器人回复是否在语义上类似于人类治疗师。我们分析了两个数据集：一个是用户与专业治疗师互动的数据集（Hugging Face的NLP心理健康对话），另一个是与社会机器人（由大型语言模型GPT-3.5驱动的LuxAI的QTrobot）进行支持性对话的数据集。通过使用句子嵌入和K-means聚类，并应用基于距离的聚类拟合方法来评估一种代理类型回复是否映射到另一种代理类型衍生的聚类，并使用欧几里得距离进行验证。结果显示，90.88%的机器人对话披露可以映射到人类治疗数据集的聚类中，表明主题结构的共享。对匹配的聚类，我们使用Transformer、Word2Vec和BERT嵌入比较了话题以及治疗师和机器人的回复，揭示了两个数据集的话题披露以及不同代理类型（机器人 vs. 人类治疗师）对类似人类披露主题的回复具有强烈的语义重叠。这些发现突出了机器人引导支持对话的相似性和界限，并探讨了其在补充心理健康干预方面的潜力。 

---
# Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities 

**Title (ZH)**: 扩散模型采样 tobز曼密度的渐进采样时退火方法 

**Authors**: Tara Akhound-Sadegh, Jungyoon Lee, Avishek Joey Bose, Valentin De Bortoli, Arnaud Doucet, Michael M. Bronstein, Dominique Beaini, Siamak Ravanbakhsh, Kirill Neklyudov, Alexander Tong  

**Link**: [PDF](https://arxiv.org/pdf/2506.16471)  

**Abstract**: Sampling efficiently from a target unnormalized probability density remains a core challenge, with relevance across countless high-impact scientific applications. A promising approach towards this challenge is the design of amortized samplers that borrow key ideas, such as probability path design, from state-of-the-art generative diffusion models. However, all existing diffusion-based samplers remain unable to draw samples from distributions at the scale of even simple molecular systems. In this paper, we propose Progressive Inference-Time Annealing (PITA), a novel framework to learn diffusion-based samplers that combines two complementary interpolation techniques: I.) Annealing of the Boltzmann distribution and II.) Diffusion smoothing. PITA trains a sequence of diffusion models from high to low temperatures by sequentially training each model at progressively higher temperatures, leveraging engineered easy access to samples of the temperature-annealed target density. In the subsequent step, PITA enables simulating the trained diffusion model to procure training samples at a lower temperature for the next diffusion model through inference-time annealing using a novel Feynman-Kac PDE combined with Sequential Monte Carlo. Empirically, PITA enables, for the first time, equilibrium sampling of N-body particle systems, Alanine Dipeptide, and tripeptides in Cartesian coordinates with dramatically lower energy function evaluations. Code available at: this https URL 

**Abstract (ZH)**: 从目标非标准化概率密度高效采样的核心挑战仍然存在于无数高影响科学应用中。一种有前景的方法是设计借鉴生成扩散模型中先进技术理念（如概率路径设计）的递归采样器。然而，现有的所有基于扩散的采样器仍然无法从简单的分子系统规模的分布中抽样。在本文中，我们提出了递归推理时退火(PITA)框架，这是一种结合了两种互补的插值技术的新颖框架：I.) 贝尔茨曼分布退火和II.) 扩散平滑。PITA 通过逐步提高训练温度来训练一系列从高到低温度的扩散模型，并利用工程化获得的退火目标密度样本的便捷访问。在后续步骤中，PITA 通过结合费曼-卡克偏微分方程与序列蒙特卡洛技术进行推理时退火，使训练好的扩散模型能够在较低温度下生成训练样本，用于下一个扩散模型。实验上，PITA 首次实现了在直角坐标系中对N体粒子系统、阿尔法二肽和三肽的平衡采样，并显著减少了能量函数评估次数。代码请参见：this https URL 

---
# Joint Tensor-Train Parameterization for Efficient and Expressive Low-Rank Adaptation 

**Title (ZH)**: 联合张量-训练参量化方法以实现高效的低秩适应 

**Authors**: Jun Qi, Chen-Yu Liu, Sabato Marco Siniscalchi, Chao-Han Huck Yang, Min-Hsiu Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2506.16456)  

**Abstract**: Low-Rank Adaptation (LoRA) is widely recognized for its parameter-efficient fine-tuning of large-scale neural models. However, standard LoRA independently optimizes low-rank matrices, which inherently limits its expressivity and generalization capabilities. While classical tensor-train (TT) decomposition can be separately employed on individual LoRA matrices, this work demonstrates that the classical TT-based approach neither significantly improves parameter efficiency nor achieves substantial performance gains. This paper proposes TensorGuide, a novel tensor-train-guided adaptation framework to overcome these limitations. TensorGuide generates two correlated low-rank LoRA matrices through a unified TT structure driven by controlled Gaussian noise. The resulting joint TT representation inherently provides structured, low-rank adaptations, significantly enhancing expressivity, generalization, and parameter efficiency without increasing the number of trainable parameters. Theoretically, we justify these improvements through neural tangent kernel analyses, demonstrating superior optimization dynamics and enhanced generalization. Extensive experiments on quantum dot classification and GPT-2 fine-tuning benchmarks demonstrate that TensorGuide-based LoRA consistently outperforms standard LoRA and TT-LoRA, achieving improved accuracy and scalability with fewer parameters. 

**Abstract (ZH)**: TensorGuide：一种新的张量引导适应框架 

---
# Consumer-friendly EEG-based Emotion Recognition System: A Multi-scale Convolutional Neural Network Approach 

**Title (ZH)**: 面向消费者的基于EEG的情绪识别系统：一种多尺度卷积神经网络方法 

**Authors**: Tri Duc Ly, Gia H. Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2506.16448)  

**Abstract**: EEG is a non-invasive, safe, and low-risk method to record electrophysiological signals inside the brain. Especially with recent technology developments like dry electrodes, consumer-grade EEG devices, and rapid advances in machine learning, EEG is commonly used as a resource for automatic emotion recognition. With the aim to develop a deep learning model that can perform EEG-based emotion recognition in a real-life context, we propose a novel approach to utilize multi-scale convolutional neural networks to accomplish such tasks. By implementing feature extraction kernels with many ratio coefficients as well as a new type of kernel that learns key information from four separate areas of the brain, our model consistently outperforms the state-of-the-art TSception model in predicting valence, arousal, and dominance scores across many performance evaluation metrics. 

**Abstract (ZH)**: 基于EEG的情感识别：利用多尺度卷积神经网络的一种新方法 

---
# StoryWriter: A Multi-Agent Framework for Long Story Generation 

**Title (ZH)**: StoryWriter：一个长故事生成的多Agent框架 

**Authors**: Haotian Xia, Hao Peng, Yunjia Qi, Xiaozhi Wang, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.16445)  

**Abstract**: Long story generation remains a challenge for existing large language models (LLMs), primarily due to two main factors: (1) discourse coherence, which requires plot consistency, logical coherence, and completeness in the long-form generation, and (2) narrative complexity, which requires an interwoven and engaging narrative. To address these challenges, we propose StoryWriter, a multi-agent story generation framework, which consists of three main modules: (1) outline agent, which generates event-based outlines containing rich event plots, character, and event-event relationships. (2) planning agent, which further details events and plans which events should be written in each chapter to maintain an interwoven and engaging story. (3) writing agent, which dynamically compresses the story history based on the current event to generate and reflect new plots, ensuring the coherence of the generated story. We conduct both human and automated evaluation, and StoryWriter significantly outperforms existing story generation baselines in both story quality and length. Furthermore, we use StoryWriter to generate a dataset, which contains about $6,000$ high-quality long stories, with an average length of $8,000$ words. We train the model Llama3.1-8B and GLM4-9B using supervised fine-tuning on LongStory and develop StoryWriter_GLM and StoryWriter_GLM, which demonstrates advanced performance in long story generation. 

**Abstract (ZH)**: 现有的大规模语言模型（LLMs）在长故事生成方面仍面临挑战，主要原因有两个方面：（1）叙述连贯性，要求长形式生成中具有情节一致性、逻辑连贯性和完整性；（2）叙事复杂性，要求叙事交织且引人入胜。为应对这些挑战，我们提出了一种多智能体故事生成框架——StoryWriter，该框架包括三个主要模块：（1）大纲生成智能体，生成基于事件的包含丰富事件情节、人物和事件关系的框架；（2）规划生成智能体，进一步细化事件，并计划在每一章中应编写哪些事件，以维持交织且引人入胜的故事；（3）写作生成智能体，根据当前事件动态压缩故事历史，生成并反映新的情节，确保生成故事的连贯性。我们进行了人类和自动评价，StoryWriter 在故事质量和长度上显著优于现有故事生成基准。此外，我们使用StoryWriter生成了一个包含约6,000个高质量长故事的数据集，平均长度为8,000词。我们使用LongStory数据集对Llama3.1-8B和GLM4-9B模型进行监督微调，并开发了StoryWriter_GLM和StoryWriter_GLM，这些模型在长故事生成方面展示了先进的性能。 

---
# Leveraging Influence Functions for Resampling Data in Physics-Informed Neural Networks 

**Title (ZH)**: 利用影响函数在物理知情神经网络中重新采样数据 

**Authors**: Jonas R. Naujoks, Aleksander Krasowski, Moritz Weckbecker, Galip Ümit Yolcu, Thomas Wiegand, Sebastian Lapuschkin, Wojciech Samek, René P. Klausen  

**Link**: [PDF](https://arxiv.org/pdf/2506.16443)  

**Abstract**: Physics-informed neural networks (PINNs) offer a powerful approach to solving partial differential equations (PDEs), which are ubiquitous in the quantitative sciences. Applied to both forward and inverse problems across various scientific domains, PINNs have recently emerged as a valuable tool in the field of scientific machine learning. A key aspect of their training is that the data -- spatio-temporal points sampled from the PDE's input domain -- are readily available. Influence functions, a tool from the field of explainable AI (XAI), approximate the effect of individual training points on the model, enhancing interpretability. In the present work, we explore the application of influence function-based sampling approaches for the training data. Our results indicate that such targeted resampling based on data attribution methods has the potential to enhance prediction accuracy in physics-informed neural networks, demonstrating a practical application of an XAI method in PINN training. 

**Abstract (ZH)**: 物理启发的神经网络（PINNs）为解决偏微分方程（PDEs）提供了一种强大的方法，这类方程在定量科学中无处不在。在各个科学领域中的正向问题和逆向问题中，PINNs 近年来已成为科学机器学习领域的一项宝贵工具。其训练的关键方面在于，可以方便地获得来自 PDE 输入域的空间-时间点数据。解释性人工智能（XAI）领域的影响函数近似每个训练点对模型的影响，增强了可解释性。在本工作中，我们探讨了基于影响函数的采样方法在训练数据中的应用。研究结果表明，基于数据归属的方法进行的目标重采样有可能提高物理启发的神经网络的预测准确性，展示了 XAI 方法在 PINN 训练中的实际应用。 

---
# Optimizing MoE Routers: Design, Implementation, and Evaluation in Transformer Models 

**Title (ZH)**: 优化MoE路由器：Transformer模型中的设计、实现与评估 

**Authors**: Daniel Fidel Harvey, George Weale, Berk Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.16419)  

**Abstract**: Mixture of Experts (MoE) architectures increase large language model scalability, yet their performance depends on the router module that moves tokens to specialized experts. Bad routing can load imbalance and reduced accuracy. This project designed and implemented different router architectures within Transformer models to fix these limitations. We experimented with six distinct router variants Linear, Attention, Multi-Layer Perceptron (MLP), Hybrid, Hash, and our new MLP-Hadamard. We characterized these routers using BERT and the Qwen1.5-MoE model, looking at parameter efficiency, inference latency, routing entropy, and expert utilization patterns. Our evaluations showed distinct trade-offs: Linear routers offer speed, while MLP and Attention routers provide greater expressiveness. The MLP-Hadamard router shows a unique capability for structured, sparse routing. We successfully replaced and fine-tuned custom routers within the complex, quantized Qwen1.5-MoE model. This work provides a comparative analysis of MoE router designs and offers insights into optimizing their performance for efficient and effective large-scale model deployment. 

**Abstract (ZH)**: Mixture of Experts架构增加了大型语言模型的可扩展性，但其性能取决于将令牌分配给专业专家的路由器模块。不良的路由会导致负载不平衡和准确性降低。本项目在Transformer模型中设计并实现了不同的路由器架构以解决这些问题。我们尝试了六种不同的路由器变体：线性、Attention、多层感知机（MLP）、混合、哈希以及我们新的MLP-哈达马变体。我们使用BERT和Qwen1.5-MoE模型对这些路由器进行了评估，考察了参数效率、推理延迟、路由熵和专家利用率模式。我们的评估表明，不同的权衡各异：线性路由器提供速度，而MLP和Attention路由器提供更大的表达能力。MLP-哈达马路由器展示了用于结构化、稀疏路由的独特能力。我们成功在复杂的量化Qwen1.5-MoE模型中替换并微调了自定义路由器。本工作提供了MoE路由器设计的比较分析，并提供了优化其性能以实现高效有效的大型模型部署的见解。 

---
# Efficient Transformations in Deep Learning Convolutional Neural Networks 

**Title (ZH)**: 深度学习卷积神经网络中的高效变换 

**Authors**: Berk Yilmaz, Daniel Fidel Harvey, Prajit Dhuri  

**Link**: [PDF](https://arxiv.org/pdf/2506.16418)  

**Abstract**: This study investigates the integration of signal processing transformations -- Fast Fourier Transform (FFT), Walsh-Hadamard Transform (WHT), and Discrete Cosine Transform (DCT) -- within the ResNet50 convolutional neural network (CNN) model for image classification. The primary objective is to assess the trade-offs between computational efficiency, energy consumption, and classification accuracy during training and inference. Using the CIFAR-100 dataset (100 classes, 60,000 images), experiments demonstrated that incorporating WHT significantly reduced energy consumption while improving accuracy. Specifically, a baseline ResNet50 model achieved a testing accuracy of 66%, consuming an average of 25,606 kJ per model. In contrast, a modified ResNet50 incorporating WHT in the early convolutional layers achieved 74% accuracy, and an enhanced version with WHT applied to both early and late layers achieved 79% accuracy, with an average energy consumption of only 39 kJ per model. These results demonstrate the potential of WHT as a highly efficient and effective approach for energy-constrained CNN applications. 

**Abstract (ZH)**: 本研究探讨了在ResNet50卷积神经网络模型中整合快速傅里叶变换（FFT）、沃尔什-哈达玛变换（WHT）和离散余弦变换（DCT）对图像分类的影响，主要评估训练和推理过程中计算效率、能耗与分类准确性之间的权衡。通过使用CIFAR-100数据集（100个类别，60,000张图像）的实验，研究表明引入WHT显著降低了能耗并提高了准确性。具体而言，基线ResNet50模型在测试集上的准确率为66%，平均每模型能耗为25,606 kJ。相比之下，一个在早期卷积层引入WHT的修改版ResNet50模型达到了74%的准确率，而一个在早期和晚期层均应用WHT的增强版ResNet50模型则达到了79%的准确率，平均每模型能耗仅为39 kJ。这些结果表明WHT作为能耗受限的卷积神经网络应用的高效而有效的方案具有巨大潜力。 

---
# Robustness Evaluation of OCR-based Visual Document Understanding under Multi-Modal Adversarial Attacks 

**Title (ZH)**: 基于多模态对抗攻击的OCR驱动视觉文档理解鲁棒性评估 

**Authors**: Dong Nguyen Tien, Dung D. Le  

**Link**: [PDF](https://arxiv.org/pdf/2506.16407)  

**Abstract**: Visual Document Understanding (VDU) systems have achieved strong performance in information extraction by integrating textual, layout, and visual signals. However, their robustness under realistic adversarial perturbations remains insufficiently explored. We introduce the first unified framework for generating and evaluating multi-modal adversarial attacks on OCR-based VDU models. Our method covers six gradient-based layout attack scenarios, incorporating manipulations of OCR bounding boxes, pixels, and texts across both word and line granularities, with constraints on layout perturbation budget (e.g., IoU >= 0.6) to preserve plausibility.
Experimental results across four datasets (FUNSD, CORD, SROIE, DocVQA) and six model families demonstrate that line-level attacks and compound perturbations (BBox + Pixel + Text) yield the most severe performance degradation. Projected Gradient Descent (PGD)-based BBox perturbations outperform random-shift baselines in all investigated models. Ablation studies further validate the impact of layout budget, text modification, and adversarial transferability. 

**Abstract (ZH)**: 基于OCR的视觉文档理解（VDU）模型的多模态 adversarial 攻击生成与评估统一框架 

---
# Drag-and-Drop LLMs: Zero-Shot Prompt-to-Weights 

**Title (ZH)**: 拖放式大语言模型：零样本提示到权重转换 

**Authors**: Zhiyuan Liang, Dongwen Tang, Yuhao Zhou, Xuanlei Zhao, Mingjia Shi, Wangbo Zhao, Zekai Li, Peihao Wang, Konstantin Schürholt, Damian Borth, Michael M. Bronstein, Yang You, Zhangyang Wang, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16406)  

**Abstract**: Modern Parameter-Efficient Fine-Tuning (PEFT) methods such as low-rank adaptation (LoRA) reduce the cost of customizing large language models (LLMs), yet still require a separate optimization run for every downstream dataset. We introduce \textbf{Drag-and-Drop LLMs (\textit{DnD})}, a prompt-conditioned parameter generator that eliminates per-task training by mapping a handful of unlabeled task prompts directly to LoRA weight updates. A lightweight text encoder distills each prompt batch into condition embeddings, which are then transformed by a cascaded hyper-convolutional decoder into the full set of LoRA matrices. Once trained in a diverse collection of prompt-checkpoint pairs, DnD produces task-specific parameters in seconds, yielding i) up to \textbf{12,000$\times$} lower overhead than full fine-tuning, ii) average gains up to \textbf{30\%} in performance over the strongest training LoRAs on unseen common-sense reasoning, math, coding, and multimodal benchmarks, and iii) robust cross-domain generalization despite never seeing the target data or labels. Our results demonstrate that prompt-conditioned parameter generation is a viable alternative to gradient-based adaptation for rapidly specializing LLMs. Our project is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 基于提示的参数生成的可拖放大型语言模型（DnD）：消除每任务训练的低秩适应 

---
# NepaliGPT: A Generative Language Model for the Nepali Language 

**Title (ZH)**: NepaliGPT：尼泊尔语生成语言模型 

**Authors**: Shushanta Pudasaini, Aman Shakya, Siddhartha Shrestha, Sahil Bhatta, Sunil Thapa, Sushmita Palikhe  

**Link**: [PDF](https://arxiv.org/pdf/2506.16399)  

**Abstract**: After the release of ChatGPT, Large Language Models (LLMs) have gained huge popularity in recent days and thousands of variants of LLMs have been released. However, there is no generative language model for the Nepali language, due to which other downstream tasks, including fine-tuning, have not been explored yet. To fill this research gap in the Nepali NLP space, this research proposes \textit{NepaliGPT}, a generative large language model tailored specifically for the Nepali language. This research introduces an advanced corpus for the Nepali language collected from several sources, called the Devanagari Corpus. Likewise, the research introduces the first NepaliGPT benchmark dataset comprised of 4,296 question-answer pairs in the Nepali language. The proposed LLM NepaliGPT achieves the following metrics in text generation: Perplexity of 26.32245, ROUGE-1 score of 0.2604, causal coherence of 81.25\%, and causal consistency of 85.41\%. 

**Abstract (ZH)**: ChatGPT发布后，大型语言模型（LLMs）在近期大受欢迎，各种变体层出不穷。但由于缺乏特定于尼泊尔语的生成语言模型，其他下游任务，包括微调，尚未被探索。为填补尼泊尔自然语言处理领域中的这一研究空缺，本研究提出了专门为尼泊尔语设计的生成型大型语言模型尼泊尔GPT（NepaliGPT）。本研究引入了从多种来源收集的先进尼泊尔语语料库，称为Devanagari Corpus。此外，本研究还引入了第一个尼泊尔GPT基准数据集，包含4,296个尼泊尔语的问题-答案对。所提出的尼泊尔GPT在文本生成中的指标如下：困惑度26.32245，ROUGE-1得分为0.2604，因果一致性为81.25%，因果一致性为85.41%。 

---
# From LLM-anation to LLM-orchestrator: Coordinating Small Models for Data Labeling 

**Title (ZH)**: 从LLM公民到LLM协调者：协调小型模型进行数据标注 

**Authors**: Yao Lu, Zhaiyuan Ji, Jiawei Du, Yu Shanqing, Qi Xuan, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.16393)  

**Abstract**: Although the annotation paradigm based on Large Language Models (LLMs) has made significant breakthroughs in recent years, its actual deployment still has two core bottlenecks: first, the cost of calling commercial APIs in large-scale annotation is very expensive; second, in scenarios that require fine-grained semantic understanding, such as sentiment classification and toxicity classification, the annotation accuracy of LLMs is even lower than that of Small Language Models (SLMs) dedicated to this field. To address these problems, we propose a new paradigm of multi-model cooperative annotation and design a fully automatic annotation framework AutoAnnotator based on this. Specifically, AutoAnnotator consists of two layers. The upper-level meta-controller layer uses the generation and reasoning capabilities of LLMs to select SLMs for annotation, automatically generate annotation code and verify difficult samples; the lower-level task-specialist layer consists of multiple SLMs that perform annotation through multi-model voting. In addition, we use the difficult samples obtained by the secondary review of the meta-controller layer as the reinforcement learning set and fine-tune the SLMs in stages through a continual learning strategy, thereby improving the generalization of SLMs. Extensive experiments show that AutoAnnotator outperforms existing open-source/API LLMs in zero-shot, one-shot, CoT, and majority voting settings. Notably, AutoAnnotator reduces the annotation cost by 74.15% compared to directly annotating with GPT-3.5-turbo, while still improving the accuracy by 6.21%. Project page: this https URL. 

**Abstract (ZH)**: 尽管基于大型语言模型（LLMs）的标注范式在近年来取得了显著突破，其实际部署仍面临两大核心瓶颈：首先，大规模标注调用商业API的成本非常昂贵；其次，在需要细粒度语义理解的场景中，如情感分类和毒性分类，LLMs的标注准确率甚至低于专门针对这些领域的中小型语言模型（SLMs）。为解决这些问题，我们提出了一种新的多模型协作标注范式，并设计了一个基于此的新自动标注框架AutoAnnotator。具体而言，AutoAnnotator由两层组成：顶层的元控制器层使用LLMs的生成和推理能力选择SLMs进行标注，自动生成标注代码并验证困难样本；底层的任务专家层由多个SLMs组成，通过多模型投票执行标注。此外，我们使用元控制器层二次审查中获得的困难样本作为强化学习集，并通过持续学习策略逐阶段微调SLMs，从而提高SLMs的泛化能力。广泛的经验研究表明，AutoAnnotator在零样本、单样本、解释型推理（CoT）和多数投票设置中优于现有的开源/API LLMs。值得注意的是，与直接使用GPT-3.5-turbo标注相比，AutoAnnotator将标注成本降低了74.15%，同时准确率提高了6.21%。项目页面：this https URL。 

---
# CLIP-MG: Guiding Semantic Attention with Skeletal Pose Features and RGB Data for Micro-Gesture Recognition on the iMiGUE Dataset 

**Title (ZH)**: CLIP-MG：基于骨架姿态特征和RGB数据的语义注意力引导微型手势识别方法 

**Authors**: Santosh Patapati, Trisanth Srinivasan, Amith Adiraju  

**Link**: [PDF](https://arxiv.org/pdf/2506.16385)  

**Abstract**: Micro-gesture recognition is a challenging task in affective computing due to the subtle, involuntary nature of the gestures and their low movement amplitude. In this paper, we introduce a Pose-Guided Semantics-Aware CLIP-based architecture, or CLIP for Micro-Gesture recognition (CLIP-MG), a modified CLIP model tailored for micro-gesture classification on the iMiGUE dataset. CLIP-MG integrates human pose (skeleton) information into the CLIP-based recognition pipeline through pose-guided semantic query generation and a gated multi-modal fusion mechanism. The proposed model achieves a Top-1 accuracy of 61.82%. These results demonstrate both the potential of our approach and the remaining difficulty in fully adapting vision-language models like CLIP for micro-gesture recognition. 

**Abstract (ZH)**: 基于姿态引导的语义感知CLIP框架：微手势识别（CLIP-MG） 

---
# Can structural correspondences ground real world representational content in Large Language Models? 

**Title (ZH)**: 结构对应能否为大型语言模型中的现实世界表征内容提供基础？ 

**Authors**: Iwan Williams  

**Link**: [PDF](https://arxiv.org/pdf/2506.16370)  

**Abstract**: Large Language Models (LLMs) such as GPT-4 produce compelling responses to a wide range of prompts. But their representational capacities are uncertain. Many LLMs have no direct contact with extra-linguistic reality: their inputs, outputs and training data consist solely of text, raising the questions (1) can LLMs represent anything and (2) if so, what? In this paper, I explore what it would take to answer these questions according to a structural-correspondence based account of representation, and make an initial survey of this evidence. I argue that the mere existence of structural correspondences between LLMs and worldly entities is insufficient to ground representation of those entities. However, if these structural correspondences play an appropriate role - they are exploited in a way that explains successful task performance - then they could ground real world contents. This requires overcoming a challenge: the text-boundedness of LLMs appears, on the face of it, to prevent them engaging in the right sorts of tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）如GPT-4对广泛的主题能够生成引人注目的回复，但其表征能力尚不确定。许多LLMs与现实世界没有直接接触：它们的输入、输出和训练数据仅限于文本，这引发了两个问题：（1）LLMs能否表征任何事物，（2）如果可以，它们能表征什么？在这篇文章中，我根据结构-对应理论探讨了回答这些问题所需的前提，并进行了初步的实证调查。我认为，LLMs与现实世界实体之间简单的结构对应是不足以支撑这些实体的表征的。但是，如果这些结构对应发挥了适当的作用——通过解释成功完成任务的方式被利用——则可以支撑现实世界的内容。这需要克服一个挑战：L表述的限定性表面上阻止了它们参与正确的任务。 

---
# Watermarking Autoregressive Image Generation 

**Title (ZH)**: 自回归图像生成中的水印技术 

**Authors**: Nikola Jovanović, Ismail Labiad, Tomáš Souček, Martin Vechev, Pierre Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.16349)  

**Abstract**: Watermarking the outputs of generative models has emerged as a promising approach for tracking their provenance. Despite significant interest in autoregressive image generation models and their potential for misuse, no prior work has attempted to watermark their outputs at the token level. In this work, we present the first such approach by adapting language model watermarking techniques to this setting. We identify a key challenge: the lack of reverse cycle-consistency (RCC), wherein re-tokenizing generated image tokens significantly alters the token sequence, effectively erasing the watermark. To address this and to make our method robust to common image transformations, neural compression, and removal attacks, we introduce (i) a custom tokenizer-detokenizer finetuning procedure that improves RCC, and (ii) a complementary watermark synchronization layer. As our experiments demonstrate, our approach enables reliable and robust watermark detection with theoretically grounded p-values. 

**Abstract (ZH)**: 基于生成模型输出的水印嵌入已成为追踪其溯源的一种有前景的方法。尽管自回归图像生成模型及其潜在的滥用引起了广泛关注，但以往没有任何研究在标记级尝试对这些模型的输出进行水印嵌入。在本工作中，我们通过将语言模型水印技术适应到此领域，提出了首个此类方法。我们识别出一个关键挑战：缺乏反向循环一致性（RCC），即重新标记生成的图像标记会大幅改变标记序列，从而有效擦除水印。为解决这一问题，并使我们的方法能够抵抗常见的图像变换、神经压缩和删除攻击，我们引入了（i）一种自定义的标记器-逆标记器微调程序，以提高RCC，以及（ii）一个互补的水印同步层。如我们的实验所展示，我们的方法能够实现可靠且具有理论依据的水印检测。 

---
# Analyzing the Influence of Knowledge Graph Information on Relation Extraction 

**Title (ZH)**: 分析知识图谱信息对关系提取的影响 

**Authors**: Cedric Möller, Ricardo Usbeck  

**Link**: [PDF](https://arxiv.org/pdf/2506.16343)  

**Abstract**: We examine the impact of incorporating knowledge graph information on the performance of relation extraction models across a range of datasets. Our hypothesis is that the positions of entities within a knowledge graph provide important insights for relation extraction tasks. We conduct experiments on multiple datasets, each varying in the number of relations, training examples, and underlying knowledge graphs. Our results demonstrate that integrating knowledge graph information significantly enhances performance, especially when dealing with an imbalance in the number of training examples for each relation. We evaluate the contribution of knowledge graph-based features by combining established relation extraction methods with graph-aware Neural Bellman-Ford networks. These features are tested in both supervised and zero-shot settings, demonstrating consistent performance improvements across various datasets. 

**Abstract (ZH)**: 我们探讨了在不同数据集中将知识图谱信息整合到关系抽取模型中对性能的影响。我们的假设是知识图谱中实体的位置为关系抽取任务提供了重要的见解。我们使用多个数据集进行实验，每个数据集在关系数量、训练示例数量和底层知识图谱方面有所不同。实验结果表明，整合知识图谱信息显著提升了模型性能，特别是在处理每种关系的训练示例不平衡时效果尤为明显。我们通过结合传统的关系抽取方法和图意识神经贝尔曼-福德网络来评估基于知识图谱特征的贡献。这些特征在有监督和零样本设置下均表现出一致的性能提升。 

---
# Reliable Few-shot Learning under Dual Noises 

**Title (ZH)**: 双噪音条件下可靠的少样本学习 

**Authors**: Ji Zhang, Jingkuan Song, Lianli Gao, Nicu Sebe, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.16330)  

**Abstract**: Recent advances in model pre-training give rise to task adaptation-based few-shot learning (FSL), where the goal is to adapt a pre-trained task-agnostic model for capturing task-specific knowledge with a few-labeled support samples of the target this http URL, existing approaches may still fail in the open world due to the inevitable in-distribution (ID) and out-of-distribution (OOD) noise from both support and query samples of the target task. With limited support samples available, i) the adverse effect of the dual noises can be severely amplified during task adaptation, and ii) the adapted model can produce unreliable predictions on query samples in the presence of the dual noises. In this work, we propose DEnoised Task Adaptation (DETA++) for reliable FSL. DETA++ uses a Contrastive Relevance Aggregation (CoRA) module to calculate image and region weights for support samples, based on which a clean prototype loss and a noise entropy maximization loss are proposed to achieve noise-robust task adaptation. Additionally,DETA++ employs a memory bank to store and refine clean regions for each inner-task class, based on which a Local Nearest Centroid Classifier (LocalNCC) is devised to yield noise-robust predictions on query samples. Moreover, DETA++ utilizes an Intra-class Region Swapping (IntraSwap) strategy to rectify ID class prototypes during task adaptation, enhancing the model's robustness to the dual noises. Extensive experiments demonstrate the effectiveness and flexibility of DETA++. 

**Abstract (ZH)**: Recent Advances in Model Pre-Training Giving Rise to Task Adaptation-Based Few-Shot Learning (FSL): DEnoised Task Adaptation (DETA++) for Reliable Few-Shot Learning 

---
# Segment Anything for Satellite Imagery: A Strong Baseline and a Regional Dataset for Automatic Field Delineation 

**Title (ZH)**: 面向卫星影像的Segment Anything：一种强大的基线模型和区域数据集，用于自动农田界定 

**Authors**: Carmelo Scribano, Elena Govi, Paolo bertellini, Simone Parisi, Giorgia Franchini, Marko Bertogna  

**Link**: [PDF](https://arxiv.org/pdf/2506.16318)  

**Abstract**: Accurate mapping of agricultural field boundaries is essential for the efficient operation of agriculture. Automatic extraction from high-resolution satellite imagery, supported by computer vision techniques, can avoid costly ground surveys. In this paper, we present a pipeline for field delineation based on the Segment Anything Model (SAM), introducing a fine-tuning strategy to adapt SAM to this task. In addition to using published datasets, we describe a method for acquiring a complementary regional dataset that covers areas beyond current sources. Extensive experiments assess segmentation accuracy and evaluate the generalization capabilities. Our approach provides a robust baseline for automated field delineation. The new regional dataset, known as ERAS, is now publicly available. 

**Abstract (ZH)**: 准确的农田边界映射对于农业高效运作至关重要。基于高分辨率卫星影像和支持计算机视觉技术的自动提取可避免昂贵的实地调查。本文提出了一种基于段一切换模型（SAM）的农田界定管道，并介绍了一种细调策略以适应这一任务。除了使用公开的数据集，我们还描述了一种方法，用于获取一个补充性的区域数据集，该数据集涵盖了当前来源之外的区域。广泛的实验评估了分割准确性并考察了泛化能力。我们的方法为自动化农田界定提供了一个稳健的基准。新的区域数据集名为ERAS，现已公开可用。 

---
# Improved Exploration in GFlownets via Enhanced Epistemic Neural Networks 

**Title (ZH)**: 通过增强 episodic 神经网络提高 GFlownets 的探索能力 

**Authors**: Sajan Muhammad, Salem Lahlou  

**Link**: [PDF](https://arxiv.org/pdf/2506.16313)  

**Abstract**: Efficiently identifying the right trajectories for training remains an open problem in GFlowNets. To address this, it is essential to prioritize exploration in regions of the state space where the reward distribution has not been sufficiently learned. This calls for uncertainty-driven exploration, in other words, the agent should be aware of what it does not know. This attribute can be measured by joint predictions, which are particularly important for combinatorial and sequential decision problems. In this research, we integrate epistemic neural networks (ENN) with the conventional architecture of GFlowNets to enable more efficient joint predictions and better uncertainty quantification, thereby improving exploration and the identification of optimal trajectories. Our proposed algorithm, ENN-GFN-Enhanced, is compared to the baseline method in GFlownets and evaluated in grid environments and structured sequence generation in various settings, demonstrating both its efficacy and efficiency. 

**Abstract (ZH)**: 高效识别用于训练的正确轨迹仍然是GFlowNets中的一个开放问题。为此，需要在奖励分布尚未充分学习的空间区域中优先进行探索。这就需要基于不确定性驱动的探索，换句话说，代理应该知道自己不知道的东西。这一属性可以通过联合预测来度量，对于组合性和序列性决策问题尤其重要。在本研究中，我们将epistemic神经网络（ENN）与GFlowNets的常规架构相结合，以实现更有效的联合预测并更好地量化不确定性，从而改善探索和最优轨迹的识别。我们提出的算法ENN-GFN-Enhanced与GFlowNets的基线方法进行比较，并在不同设置的网格环境和结构化序列生成中进行评估，证明了其有效性和效率。 

---
# Learning Multi-scale Spatial-frequency Features for Image Denoising 

**Title (ZH)**: 学习多尺度空间频率特征的图像去噪方法 

**Authors**: Xu Zhao, Chen Zhao, Xiantao Hu, Hongliang Zhang, Ying Tai, Jian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16307)  

**Abstract**: Recent advancements in multi-scale architectures have demonstrated exceptional performance in image denoising tasks. However, existing architectures mainly depends on a fixed single-input single-output Unet architecture, ignoring the multi-scale representations of pixel level. In addition, previous methods treat the frequency domain uniformly, ignoring the different characteristics of high-frequency and low-frequency noise. In this paper, we propose a novel multi-scale adaptive dual-domain network (MADNet) for image denoising. We use image pyramid inputs to restore noise-free results from low-resolution images. In order to realize the interaction of high-frequency and low-frequency information, we design an adaptive spatial-frequency learning unit (ASFU), where a learnable mask is used to separate the information into high-frequency and low-frequency components. In the skip connections, we design a global feature fusion block to enhance the features at different scales. Extensive experiments on both synthetic and real noisy image datasets verify the effectiveness of MADNet compared with current state-of-the-art denoising approaches. 

**Abstract (ZH)**: Recent advancements in multi-scale architectures have demonstrated exceptional performance in image denoising tasks. However, existing architectures mainly depend on a fixed single-input single-output Unet architecture, ignoring the multi-scale representations of pixel level. In addition, previous methods treat the frequency domain uniformly, ignoring the different characteristics of high-frequency and low-frequency noise. In this paper, we propose a novel multi-scale adaptive dual-domain network (MADNet) for image denoising. We use image pyramid inputs to restore noise-free results from low-resolution images. In order to realize the interaction of high-frequency and low-frequency information, we design an adaptive spatial-frequency learning unit (ASFU), where a learnable mask is used to separate the information into high-frequency and low-frequency components. In the skip connections, we design a global feature fusion block to enhance the features at different scales. Extensive experiments on both synthetic and real noisy image datasets verify the effectiveness of MADNet compared with current state-of-the-art denoising approaches. 

---
# SycnMapV2: Robust and Adaptive Unsupervised Segmentation 

**Title (ZH)**: SycnMapV2: 坚韧且自适应的无监督分割 

**Authors**: Heng Zhang, Zikang Wan, Danilo Vasconcellos Vargas  

**Link**: [PDF](https://arxiv.org/pdf/2506.16297)  

**Abstract**: Human vision excels at segmenting visual cues without the need for explicit training, and it remains remarkably robust even as noise severity increases. In contrast, existing AI algorithms struggle to maintain accuracy under similar conditions. Here, we present SyncMapV2, the first to solve unsupervised segmentation with state-of-the-art robustness. SyncMapV2 exhibits a minimal drop in mIoU, only 0.01%, under digital corruption, compared to a 23.8% drop observed in SOTA this http URL superior performance extends across various types of corruption: noise (7.3% vs. 37.7%), weather (7.5% vs. 33.8%), and blur (7.0% vs. 29.5%). Notably, SyncMapV2 accomplishes this without any robust training, supervision, or loss functions. It is based on a learning paradigm that uses self-organizing dynamical equations combined with concepts from random networks. Moreover,unlike conventional methods that require re-initialization for each new input, SyncMapV2 adapts online, mimicking the continuous adaptability of human vision. Thus, we go beyond the accurate and robust results, and present the first algorithm that can do all the above online, adapting to input rather than re-initializing. In adaptability tests, SyncMapV2 demonstrates near-zero performance degradation, which motivates and fosters a new generation of robust and adaptive intelligence in the near future. 

**Abstract (ZH)**: SyncMapV2：第一种在多种噪声条件下具备卓越鲁棒性的无监督分割算法 

---
# Next-Token Prediction Should be Ambiguity-Sensitive: A Meta-Learning Perspective 

**Title (ZH)**: 下一词预测应具备歧义敏感性：一种元学习视角 

**Authors**: Leo Gagnon, Eric Elmoznino, Sarthak Mittal, Tom Marty, Tejas Kasetty, Dhanya Sridhar, Guillaume Lajoie  

**Link**: [PDF](https://arxiv.org/pdf/2506.16288)  

**Abstract**: The rapid adaptation ability of auto-regressive foundation models is often attributed to the diversity of their pre-training data. This is because, from a Bayesian standpoint, minimizing prediction error in such settings requires integrating over all plausible latent hypotheses consistent with observations. While this behavior is desirable in principle, it often proves too ambitious in practice: under high ambiguity, the number of plausible latent alternatives makes Bayes-optimal prediction computationally intractable. Cognitive science has long recognized this limitation, suggesting that under such conditions, heuristics or information-seeking strategies are preferable to exhaustive inference. Translating this insight to next-token prediction, we hypothesize that low- and high-ambiguity predictions pose different computational demands, making ambiguity-agnostic next-token prediction a detrimental inductive bias. To test this, we introduce MetaHMM, a synthetic sequence meta-learning benchmark with rich compositional structure and a tractable Bayesian oracle. We show that Transformers indeed struggle with high-ambiguity predictions across model sizes. Motivated by cognitive theories, we propose a method to convert pre-trained models into Monte Carlo predictors that decouple task inference from token prediction. Preliminary results show substantial gains in ambiguous contexts through improved capacity allocation and test-time scalable inference, though challenges remain. 

**Abstract (ZH)**: 自回归基础模型的快速适应能力通常归因于其预训练数据的多样性。从贝叶斯观点来看，这种设置中的预测误差最小化需要整合所有与观察结果一致的合理潜在假设。虽然这种行为原则上是可取的，但在实际中往往过于雄心勃勃：在高不确定性下，合理的潜在替代方案数量使贝叶斯最优预测计算上不可行。认知科学长期认识到这一限制，建议在这种情况下，启发式或信息寻求策略比详尽推理更可取。将这一洞察应用于下一个令牌预测，我们推测低不确定性与高不确定性预测在计算需求上存在差异，使不确定性忽略的下一个令牌预测成为不利的归纳偏见。为验证这一假设，我们引入了MetaHMM，这是一个具有丰富组合结构的合成序列元学习基准及可行的贝叶斯 oracle。我们显示，不同规模的Transformer确实难以处理高不确定性预测。受认知理论的启发，我们提出了一种将预训练模型转化为马尔可夫蒙特卡洛预测者的办法，以分离任务推理与令牌预测。初步结果表明，在不确定的上下文中，通过改进的容量分配和测试时可扩展的推理，可以获得显著收益，但仍存在挑战。 

---
# Artificial Intelligence for Atmospheric Sciences: A Research Roadmap 

**Title (ZH)**: 人工智能在大气科学中的应用：研究路线图 

**Authors**: Martha Arbayani Zaidan, Naser Hossein Motlagh, Petteri Nurmi, Tareq Hussein, Markku Kulmala, Tuukka Petäjä, Sasu Tarkoma  

**Link**: [PDF](https://arxiv.org/pdf/2506.16281)  

**Abstract**: Atmospheric sciences are crucial for understanding environmental phenomena ranging from air quality to extreme weather events, and climate change. Recent breakthroughs in sensing, communication, computing, and Artificial Intelligence (AI) have significantly advanced atmospheric sciences, enabling the generation of vast amounts of data through long-term Earth observations and providing powerful tools for analyzing atmospheric phenomena and predicting natural disasters. This paper contributes a critical interdisciplinary overview that bridges the fields of atmospheric science and computer science, highlighting the transformative potential of AI in atmospheric research. We identify key challenges associated with integrating AI into atmospheric research, including issues related to big data and infrastructure, and provide a detailed research roadmap that addresses both current and emerging challenges. 

**Abstract (ZH)**: 大气科学对于理解从空气质量到极端天气事件以及气候变化的环境现象至关重要。最近在感知、通信、计算和人工智能（AI）领域的突破性进展极大地推动了大气科学的发展，通过长期的地球观测生成了大量的数据，并提供强大的工具来分析大气现象和预测自然灾害。本文提供了一个跨学科的关键性综述，将大气科学与计算机科学领域连接起来，突显了AI在大气研究中的变革潜力。我们识别了将AI整合到大气研究中所面临的若干关键挑战，包括与大数据和基础设施相关的问题，并提供了详细的研究路线图，以解决当前和新兴的挑战。 

---
# CapsDT: Diffusion-Transformer for Capsule Robot Manipulation 

**Title (ZH)**: CapsDT: 扩散变换器在胶囊机器人操作中的应用 

**Authors**: Xiting He, Mingwu Su, Xinqi Jiang, Long Bai, Jiewen Lai, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.16263)  

**Abstract**: Vision-Language-Action (VLA) models have emerged as a prominent research area, showcasing significant potential across a variety of applications. However, their performance in endoscopy robotics, particularly endoscopy capsule robots that perform actions within the digestive system, remains unexplored. The integration of VLA models into endoscopy robots allows more intuitive and efficient interactions between human operators and medical devices, improving both diagnostic accuracy and treatment outcomes. In this work, we design CapsDT, a Diffusion Transformer model for capsule robot manipulation in the stomach. By processing interleaved visual inputs, and textual instructions, CapsDT can infer corresponding robotic control signals to facilitate endoscopy tasks. In addition, we developed a capsule endoscopy robot system, a capsule robot controlled by a robotic arm-held magnet, addressing different levels of four endoscopy tasks and creating corresponding capsule robot datasets within the stomach simulator. Comprehensive evaluations on various robotic tasks indicate that CapsDT can serve as a robust vision-language generalist, achieving state-of-the-art performance in various levels of endoscopy tasks while achieving a 26.25% success rate in real-world simulation manipulation. 

**Abstract (ZH)**: Vision-Language-Action (VLA)模型已成为一个重要的研究领域，展示出在多种应用中的巨大潜力。然而，这些模型在内窥镜机器人，特别是消化系统内的胶囊机器人执行操作方面的性能仍未被探索。将VLA模型集成到内窥镜机器人中，可以实现更直观和高效的医护人员与医疗设备之间的交互，提高诊断准确性和治疗效果。在本工作中，我们设计了CapsDT，一种用于胃部胶囊机器人操作的扩散变换器模型。通过处理交错的视觉输入和文本指令，CapsDT可以推断出相应的机器人控制信号，以辅助内窥镜任务。此外，我们还开发了一种胶囊内窥镜机器人系统，该系统采用手持磁铁的机械臂控制胶囊机器人，并在胃部模拟器中针对四种不同的内窥镜任务的不同层级建立了相应的胶囊机器人数据集。各种机器人任务的全面评估表明，CapsDT可以作为稳健的视觉-语言通用模型，在不同层级的内窥镜任务中实现最先进的性能，同时在现实世界的模拟操作中实现26.25%的成功率。 

---
# Category-based Galaxy Image Generation via Diffusion Models 

**Title (ZH)**: 基于类别生成的银河图像生成方法 

**Authors**: Xingzhong Fan, Hongming Tang, Yue Zeng, M.B.N.Kouwenhoven, Guangquan Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.16255)  

**Abstract**: Conventional galaxy generation methods rely on semi-analytical models and hydrodynamic simulations, which are highly dependent on physical assumptions and parameter tuning. In contrast, data-driven generative models do not have explicit physical parameters pre-determined, and instead learn them efficiently from observational data, making them alternative solutions to galaxy generation. Among these, diffusion models outperform Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs) in quality and diversity. Leveraging physical prior knowledge to these models can further enhance their capabilities. In this work, we present GalCatDiff, the first framework in astronomy to leverage both galaxy image features and astrophysical properties in the network design of diffusion models. GalCatDiff incorporates an enhanced U-Net and a novel block entitled Astro-RAB (Residual Attention Block), which dynamically combines attention mechanisms with convolution operations to ensure global consistency and local feature fidelity. Moreover, GalCatDiff uses category embeddings for class-specific galaxy generation, avoiding the high computational costs of training separate models for each category. Our experimental results demonstrate that GalCatDiff significantly outperforms existing methods in terms of the consistency of sample color and size distributions, and the generated galaxies are both visually realistic and physically consistent. This framework will enhance the reliability of galaxy simulations and can potentially serve as a data augmentor to support future galaxy classification algorithm development. 

**Abstract (ZH)**: 基于扩散模型的GalCatDiff：结合星系图像特征和天体物理属性的星系生成框架 

---
# Synthetic ALS-EEG Data Augmentation for ALS Diagnosis Using Conditional WGAN with Weight Clipping 

**Title (ZH)**: 基于条件WGAN与权重裁剪的ALS-EEG合成数据增强在ALS诊断中的应用 

**Authors**: Abdulvahap Mutlu, Şengül Doğan, Türker Tuncer  

**Link**: [PDF](https://arxiv.org/pdf/2506.16243)  

**Abstract**: Amyotrophic Lateral Sclerosis (ALS) is a rare neurodegenerative disease, and high-quality EEG data from ALS patients are scarce. This data scarcity, coupled with severe class imbalance between ALS and healthy control recordings, poses a challenge for training reliable machine learning classifiers. In this work, we address these issues by generating synthetic EEG signals for ALS patients using a Conditional Wasserstein Generative Adversarial Network (CWGAN). We train CWGAN on a private EEG dataset (ALS vs. non-ALS) to learn the distribution of ALS EEG signals and produce realistic synthetic samples. We preprocess and normalize EEG recordings, and train a CWGAN model to generate synthetic ALS signals. The CWGAN architecture and training routine are detailed, with key hyperparameters chosen for stable training. Qualitative evaluation of generated signals shows that they closely mimic real ALS EEG patterns. The CWGAN training converged with generator and discriminator loss curves stabilizing, indicating successful learning. The synthetic EEG signals appear realistic and have potential use as augmented data for training classifiers, helping to mitigate class imbalance and improve ALS detection accuracy. We discuss how this approach can facilitate data sharing and enhance diagnostic models. 

**Abstract (ZH)**: 肌萎缩侧索硬化症（ALS）是一种罕见的神经退行性疾病，ALS患者高质量EEG数据稀缺。这种数据稀缺性与ALS记录与健康控制记录之间严重的类别不平衡相结合，为训练可靠的机器学习分类器带来了挑战。本文通过使用条件沃森生成对抗网络（CWGAN）生成ALS患者的合成EEG信号来解决这些问题。我们使用一个私有的EEG数据集（ALS vs. 非ALS）训练CWGAN，以学习ALS EEG信号的分布并产生逼真的合成样本。我们预处理并标准化EEG记录，训练了一个CWGAN模型以生成合成ALS信号。CWGAN架构和训练流程被详细说明，关键超参数选择以实现稳定训练。生成信号的定性评估显示，它们 closely 模仿真实的ALS EEG模式。CWGAN训练收敛，生成器和判别器损失曲线趋于稳定，表明成功学习。生成的合成EEG信号看起来很真实，并且有可能作为训练分类器的增強数据使用，有助于缓解类别不平衡并提高ALS检测准确性。我们讨论了该方法如何促进数据共享并增强诊断模型。 

---
# CF-Seg: Counterfactuals meet Segmentation 

**Title (ZH)**: CF-Seg: 反事实推理结合分割 

**Authors**: Raghav Mehta, Fabio De Sousa Ribeiro, Tian Xia, Melanie Roschewitz, Ainkaran Santhirasekaram, Dominic C. Marshall, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2506.16213)  

**Abstract**: Segmenting anatomical structures in medical images plays an important role in the quantitative assessment of various diseases. However, accurate segmentation becomes significantly more challenging in the presence of disease. Disease patterns can alter the appearance of surrounding healthy tissues, introduce ambiguous boundaries, or even obscure critical anatomical structures. As such, segmentation models trained on real-world datasets may struggle to provide good anatomical segmentation, leading to potential misdiagnosis. In this paper, we generate counterfactual (CF) images to simulate how the same anatomy would appear in the absence of disease without altering the underlying structure. We then use these CF images to segment structures of interest, without requiring any changes to the underlying segmentation model. Our experiments on two real-world clinical chest X-ray datasets show that the use of counterfactual images improves anatomical segmentation, thereby aiding downstream clinical decision-making. 

**Abstract (ZH)**: 生成反事实图像以在无病状态下模拟 Anatomy 在无病状态下出现的情况，从而改善解剖学分割，辅助下游临床决策。 

---
# CP$^2$: Leveraging Geometry for Conformal Prediction via Canonicalization 

**Title (ZH)**: CP$^2$: 利用几何进行规范预测的典范化方法 

**Authors**: Putri A. van der Linden, Alexander Timans, Erik J. Bekkers  

**Link**: [PDF](https://arxiv.org/pdf/2506.16189)  

**Abstract**: We study the problem of conformal prediction (CP) under geometric data shifts, where data samples are susceptible to transformations such as rotations or flips. While CP endows prediction models with post-hoc uncertainty quantification and formal coverage guarantees, their practicality breaks under distribution shifts that deteriorate model performance. To address this issue, we propose integrating geometric information--such as geometric pose--into the conformal procedure to reinstate its guarantees and ensure robustness under geometric shifts. In particular, we explore recent advancements on pose canonicalization as a suitable information extractor for this purpose. Evaluating the combined approach across discrete and continuous shifts and against equivariant and augmentation-based baselines, we find that integrating geometric information with CP yields a principled way to address geometric shifts while maintaining broad applicability to black-box predictors. 

**Abstract (ZH)**: 几何数据变换下保准预测问题的研究：几何信息的集成以应对几何变换 

---
# JETHICS: Japanese Ethics Understanding Evaluation Dataset 

**Title (ZH)**: 日本伦理理解评价数据集 

**Authors**: Masashi Takeshita, Rafal Rzepka  

**Link**: [PDF](https://arxiv.org/pdf/2506.16187)  

**Abstract**: In this work, we propose JETHICS, a Japanese dataset for evaluating ethics understanding of AI models. JETHICS contains 78K examples and is built by following the construction methods of the existing English ETHICS dataset. It includes four categories based normative theories and concepts from ethics and political philosophy; and one representing commonsense morality. Our evaluation experiments on non-proprietary large language models (LLMs) and on GPT-4o reveal that even GPT-4o achieves only an average score of about 0.7, while the best-performing Japanese LLM attains around 0.5, indicating a relatively large room for improvement in current LLMs. 

**Abstract (ZH)**: 本研究提出JETHICS，一个日语数据集，用于评估AI模型的伦理理解能力。JETHICS包含78,000个案例，按照现有英语ETHICS数据集的构建方法进行构建，包括基于伦理和政治哲学的规范理论和概念的四个类别，以及一种代表常识道德的类别。在非专有大型语言模型（LLM）和GPT-4o上的评估实验表明，即使是GPT-4o也仅能达到约0.7的平均得分，而性能最好的日本LLM达到约0.5，表明当前LLM仍有较大的改进空间。 

---
# From Teacher to Student: Tracking Memorization Through Model Distillation 

**Title (ZH)**: 从教师到学生：通过模型蒸馏追踪记忆化过程 

**Authors**: Simardeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2506.16170)  

**Abstract**: Large language models (LLMs) are known to memorize parts of their training data, raising important concerns around privacy and security. While previous research has focused on studying memorization in pre-trained models, much less is known about how knowledge distillation (KD) affects this http URL this study, we explore how different KD methods influence the memorization of fine-tuned task data when a large teacher model is distilled into smaller student this http URL study demonstrates that distilling a larger teacher model, fine-tuned on a dataset, into a smaller variant not only lowers computational costs and model size but also significantly reduces the memorization risks compared to standard fine-tuning approaches. 

**Abstract (ZH)**: 大型语言模型（LLMs）known to memorize parts of their training数据，引发重要的隐私和安全关切。尽管以往研究主要关注预训练模型的 memorization 现象，但对于知识精炼（KD）如何影响这一点了解甚少。在此研究中，我们探讨了当大规模教师模型被精炼为较小的学生模型时，不同 KD 方法如何影响 fine-tuned 任务数据的记忆风险。研究显示，将 fine-tuned 在特定数据集上的大规模教师模型精炼成较小的学生模型不仅降低了计算成本和模型大小，还显着降低了与标准 fine-tuning 方法相比的记忆风险。 

---
# On using AI for EEG-based BCI applications: problems, current challenges and future trends 

**Title (ZH)**: 基于EEG的BCI应用中使用AI的问题、当前挑战及未来趋势 

**Authors**: Thomas Barbera, Jacopo Burger, Alessandro D'Amelio, Simone Zini, Simone Bianco, Raffaella Lanzarotti, Paolo Napoletano, Giuseppe Boccignone, Jose Luis Contreras-Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2506.16168)  

**Abstract**: Imagine unlocking the power of the mind to communicate, create, and even interact with the world around us. Recent breakthroughs in Artificial Intelligence (AI), especially in how machines "see" and "understand" language, are now fueling exciting progress in decoding brain signals from scalp electroencephalography (EEG). Prima facie, this opens the door to revolutionary brain-computer interfaces (BCIs) designed for real life, moving beyond traditional uses to envision Brain-to-Speech, Brain-to-Image, and even a Brain-to-Internet of Things (BCIoT).
However, the journey is not as straightforward as it was for Computer Vision (CV) and Natural Language Processing (NLP). Applying AI to real-world EEG-based BCIs, particularly in building powerful foundational models, presents unique and intricate hurdles that could affect their reliability.
Here, we unfold a guided exploration of this dynamic and rapidly evolving research area. Rather than barely outlining a map of current endeavors and results, the goal is to provide a principled navigation of this hot and cutting-edge research landscape. We consider the basic paradigms that emerge from a causal perspective and the attendant challenges presented to AI-based models. Looking ahead, we then discuss promising research avenues that could overcome today's technological, methodological, and ethical limitations. Our aim is to lay out a clear roadmap for creating truly practical and effective EEG-based BCI solutions that can thrive in everyday environments. 

**Abstract (ZH)**: 想象一下利用心智的力量进行沟通、创造，甚至与周围的这个世界互动。近期人工智能（AI）在机器如何“看”和“理解”语言方面的突破，正推动着从头皮脑电图（EEG）解码脑信号的激动人心的进展。初步看来，这为设计用于现实生活的革命性脑机接口（BCI）打开了大门，超越了传统应用领域，设想脑到语音（Brain-to-Speech）、脑到图像（Brain-to-Image）和甚至脑到物联网（BCIoT）。

然而，这一旅程不像计算机视觉（CV）和自然语言处理（NLP）那样简单。将AI应用于基于EEG的BCI，尤其是构建强大的基础模型，面临着独特而复杂的挑战，这些挑战可能会影响它们的可靠性。

在此，我们展开了一次引导性的探索，这一领域是动态且快速发展的。我们的目标不是简单地勾勒出现有努力和结果的地图，而是提供一种原理性的导航，以探索这一热门和前沿的研究景观。我们从因果视角出发考虑基本框架和随之而来的挑战，随后讨论有可能克服当前技术、方法和伦理限制的有希望的研究方向。我们的目标是为创造能够在日常环境中茁壮成长的真正实用且有效的基于EEG的BCI解决方案绘制一条清晰的路线图。 

---
# Under the Shadow of Babel: How Language Shapes Reasoning in LLMs 

**Title (ZH)**: babel之影：语言如何塑造LLMs中的推理 

**Authors**: Chenxi Wang, Yixuan Zhang, Lang Gao, Zixiang Xu, Zirui Song, Yanbo Wang, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.16151)  

**Abstract**: Language is not only a tool for communication but also a medium for human cognition and reasoning. If, as linguistic relativity suggests, the structure of language shapes cognitive patterns, then large language models (LLMs) trained on human language may also internalize the habitual logical structures embedded in different languages. To examine this hypothesis, we introduce BICAUSE, a structured bilingual dataset for causal reasoning, which includes semantically aligned Chinese and English samples in both forward and reversed causal forms. Our study reveals three key findings: (1) LLMs exhibit typologically aligned attention patterns, focusing more on causes and sentence-initial connectives in Chinese, while showing a more balanced distribution in English. (2) Models internalize language-specific preferences for causal word order and often rigidly apply them to atypical inputs, leading to degraded performance, especially in Chinese. (3) When causal reasoning succeeds, model representations converge toward semantically aligned abstractions across languages, indicating a shared understanding beyond surface form. Overall, these results suggest that LLMs not only mimic surface linguistic forms but also internalize the reasoning biases shaped by language. Rooted in cognitive linguistic theory, this phenomenon is for the first time empirically verified through structural analysis of model internals. 

**Abstract (ZH)**: 语言不仅是沟通的工具，也是人类认知与推理的媒介。如果语言相对论的假设成立，即语言的结构影响认知模式，那么基于人类语言训练的大型语言模型（LLMs）也可能内化不同语言中嵌入的习惯逻辑结构。为了检验这一假设，我们引入了BICAUSE，这是一个结构化的双语因果推理数据集，包括前后因果形式的语义对齐的中文和英文样本。我们的研究表明：（1）LLMs表现出类型学上的注意力模式，更关注于中文中的原因和句子初始连接词，而在英语中则表现出更均衡的分布。（2）模型内化了特定于语言的因果词序偏好，并且常常僵硬地应用这些偏好于非典型输入，导致性能下降，尤其是在中文中更为明显。（3）当因果推理成功时，模型表示收敛于跨语言的语义对齐的抽象，表明在表层形式之外存在共享的理解。总体而言，这些结果表明，LLMs不仅模仿表面语言形式，还内化了由语言塑造的推理偏见。基于认知语言学理论，这一现象首次通过模型内部结构的分析得到了实证验证。 

---
# PRISON: Unmasking the Criminal Potential of Large Language Models 

**Title (ZH)**: PRISON: 揭示大型语言模型的犯罪潜在风险 

**Authors**: Xinyi Wu, Geng Hong, Pei Chen, Yueyue Chen, Xudong Pan, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16150)  

**Abstract**: As large language models (LLMs) advance, concerns about their misconduct in complex social contexts intensify. Existing research overlooked the systematic understanding and assessment of their criminal capability in realistic interactions. We propose a unified framework PRISON, to quantify LLMs' criminal potential across five dimensions: False Statements, Frame-Up, Psychological Manipulation, Emotional Disguise, and Moral Disengagement. Using structured crime scenarios adapted from classic films, we evaluate both criminal potential and anti-crime ability of LLMs via role-play. Results show that state-of-the-art LLMs frequently exhibit emergent criminal tendencies, such as proposing misleading statements or evasion tactics, even without explicit instructions. Moreover, when placed in a detective role, models recognize deceptive behavior with only 41% accuracy on average, revealing a striking mismatch between conducting and detecting criminal behavior. These findings underscore the urgent need for adversarial robustness, behavioral alignment, and safety mechanisms before broader LLM deployment. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的发展，它们在复杂社会情境中不当行为的担忧逐渐增强。现有研究忽视了对它们在现实互动中的犯罪能力进行系统的理解和评估。我们提出了一种统一框架PRISON，以五个维度（虚假陈述、栽赃、心理操控、情感伪装和道德脱耦）来量化LLMs的犯罪潜能。通过采用来自经典电影的结构化犯罪场景，我们借助角色扮演评估了LLMs的犯罪潜能及其反犯罪能力。结果表明，最先进的LLMs经常表现出新兴的犯罪倾向，如提出误导性陈述或逃避策略，即使没有明确的指令。此外，当将模型置于侦探角色时，它们仅以41%的平均准确率识别欺骗行为，揭示了执行与检测犯罪行为之间存在显著的不匹配。这些发现强调了在更广泛部署LLMs之前需要增强对抗鲁棒性、行为对齐和安全机制的迫切性。 

---
# GRPO-CARE: Consistency-Aware Reinforcement Learning for Multimodal Reasoning 

**Title (ZH)**: GRPO-CARE：一致性导向的多模态强化学习 

**Authors**: Yi Chen, Yuying Ge, Rui Wang, Yixiao Ge, Junhao Cheng, Ying Shan, Xihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.16141)  

**Abstract**: Recent reinforcement learning approaches, such as outcome-supervised GRPO, have advanced Chain-of-Thought reasoning in large language models (LLMs), yet their adaptation to multimodal LLMs (MLLMs) is unexplored. To address the lack of rigorous evaluation for MLLM post-training methods, we introduce SEED-Bench-R1, a benchmark with complex real-world videos requiring balanced perception and reasoning. It offers a large training set and evaluates generalization across three escalating challenges: in-distribution, cross-environment, and cross-environment-task scenarios. Using SEED-Bench-R1, we find that standard GRPO, while improving answer accuracy, often reduces logical coherence between reasoning steps and answers, with only a 57.9% consistency rate. This stems from reward signals focusing solely on final answers, encouraging shortcuts, and strict KL penalties limiting this http URL address this, we propose GRPO-CARE, a consistency-aware RL framework optimizing both answer correctness and reasoning coherence without explicit supervision. GRPO-CARE introduces a two-tiered reward: (1) a base reward for answer correctness, and (2) an adaptive consistency bonus, computed by comparing the model's reasoning-to-answer likelihood (via a slowly-evolving reference model) against group this http URL dual mechanism amplifies rewards for reasoning paths that are both correct and logically consistent. Replacing KL penalties with this adaptive bonus, GRPO-CARE outperforms standard GRPO on SEED-Bench-R1, achieving a 6.7% performance gain on the hardest evaluation level and a 24.5% improvement in consistency. It also shows strong transferability, improving model performance across diverse video understanding benchmarks. Our work contributes a systematically designed benchmark and a generalizable post-training framework, advancing the development of more interpretable and robust MLLMs. 

**Abstract (ZH)**: Recent Reinforcement Learning Approaches for Enhancing Reasoning in Multimodal Large Language Models: Introducing SEED-Bench-R1 and GRPO-CARE 

---
# Improved Intelligibility of Dysarthric Speech using Conditional Flow Matching 

**Title (ZH)**: 使用条件流匹配改进失语性 speech 的可懂度 

**Authors**: Shoutrik Das, Nishant Singh, Arjun Gangwar, S Umesh  

**Link**: [PDF](https://arxiv.org/pdf/2506.16127)  

**Abstract**: Dysarthria is a neurological disorder that significantly impairs speech intelligibility, often rendering affected individuals unable to communicate effectively. This necessitates the development of robust dysarthric-to-regular speech conversion techniques. In this work, we investigate the utility and limitations of self-supervised learning (SSL) features and their quantized representations as an alternative to mel-spectrograms for speech generation. Additionally, we explore methods to mitigate speaker variability by generating clean speech in a single-speaker voice using features extracted from WavLM. To this end, we propose a fully non-autoregressive approach that leverages Conditional Flow Matching (CFM) with Diffusion Transformers to learn a direct mapping from dysarthric to clean speech. Our findings highlight the effectiveness of discrete acoustic units in improving intelligibility while achieving faster convergence compared to traditional mel-spectrogram-based approaches. 

**Abstract (ZH)**: 构音障碍是一种神经性疾病，显著影响言语清晰度，常导致患者无法有效沟通。这 necessitates the development of robust dysarthric-to-regular speech conversion techniques. 在此工作中，我们探讨了自监督学习（SSL）特征及其量化表示作为mel- spectrograms替代品用于语音生成的应用和限制。此外，我们探索了通过使用从WavLM提取的特征生成单说话人口头清晰的语音以减轻说话人变异性的方法。为此，我们提出了一种完全非自回归方法，利用条件流匹配（CFM）与扩散变换器来学习从构音障碍到口头清晰语音的直接映射。我们的研究发现显示，离散声学单元在提高清晰度方面非常有效，并且收敛速度比传统的基于mel- spectrograms的方法更快。 

---
# GFlowGR: Fine-tuning Generative Recommendation Frameworks with Generative Flow Networks 

**Title (ZH)**: GFlowGR：基于生成流网络的生成推荐框架微调 

**Authors**: Yejing Wang, Shengyu Zhou, Jinyu Lu, Qidong Liu, Xinhang Li, Wenlin Zhang, Feng Li, Pengjie Wang, Jian Xu, Bo Zheng, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.16114)  

**Abstract**: Generative recommendations (GR), which usually include item tokenizers and generative Large Language Models (LLMs), have demonstrated remarkable success across a wide range of scenarios. The majority of existing research efforts primarily concentrate on developing powerful item tokenizers or advancing LLM decoding strategies to attain superior performance. However, the critical fine-tuning step in GR frameworks, which is essential for adapting LLMs to recommendation data, remains largely unexplored. Current approaches predominantly rely on either the next-token prediction loss of supervised fine-tuning (SFT) or recommendationspecific direct preference optimization (DPO) strategies. Both methods ignore the exploration of possible positive unobserved samples, which is commonly referred to as the exposure bias problem. To mitigate this problem, this paper treats the GR as a multi-step generation task and constructs a GFlowNets-based fine-tuning framework (GFlowGR). The proposed framework integrates collaborative knowledge from traditional recommender systems to create an adaptive trajectory sampler and a comprehensive reward model. Leveraging the diverse generation property of GFlowNets, along with sampling and heuristic weighting techniques, GFlowGR emerges as a promising approach to mitigate the exposure bias problem. Extensive empirical results on two real-world datasets and with two different GR backbones highlight the effectiveness and robustness of GFlowGR. 

**Abstract (ZH)**: 生成推荐（GR）：基于GFlowNets的生成推荐微调框架 

---
# A Brain-to-Population Graph Learning Framework for Diagnosing Brain Disorders 

**Title (ZH)**: 一种脑-人群图学习框架用于诊断脑部疾病 

**Authors**: Qianqian Liao, Wuque Cai, Hongze Sun, Dongze Liu, Duo Chen, Dezhong Yao, Daqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.16096)  

**Abstract**: Recent developed graph-based methods for diagnosing brain disorders using functional connectivity highly rely on predefined brain atlases, but overlook the rich information embedded within atlases and the confounding effects of site and phenotype variability. To address these challenges, we propose a two-stage Brain-to-Population Graph Learning (B2P-GL) framework that integrates the semantic similarity of brain regions and condition-based population graph modeling. In the first stage, termed brain representation learning, we leverage brain atlas knowledge from GPT-4 to enrich the graph representation and refine the brain graph through an adaptive node reassignment graph attention network. In the second stage, termed population disorder diagnosis, phenotypic data is incorporated into population graph construction and feature fusion to mitigate confounding effects and enhance diagnosis performance. Experiments on the ABIDE I, ADHD-200, and Rest-meta-MDD datasets show that B2P-GL outperforms state-of-the-art methods in prediction accuracy while enhancing interpretability. Overall, our proposed framework offers a reliable and personalized approach to brain disorder diagnosis, advancing clinical applicability. 

**Abstract (ZH)**: 基于脑图谱的双阶段脑至人群图学习框架 

---
# Probing the Robustness of Large Language Models Safety to Latent Perturbations 

**Title (ZH)**: 探究大型语言模型在潜在扰动下的安全性 robustness 

**Authors**: Tianle Gu, Kexin Huang, Zongqi Wang, Yixu Wang, Jie Li, Yuanqi Yao, Yang Yao, Yujiu Yang, Yan Teng, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16078)  

**Abstract**: Safety alignment is a key requirement for building reliable Artificial General Intelligence. Despite significant advances in safety alignment, we observe that minor latent shifts can still trigger unsafe responses in aligned models. We argue that this stems from the shallow nature of existing alignment methods, which focus on surface-level refusal behaviors without sufficiently altering internal representations. Consequently, small shifts in hidden activations can re-trigger harmful behaviors embedded in the latent space. To explore the robustness of safety alignment to latent perturbations, we introduce a probing method that measures the Negative Log-Likelihood of the original response generated by the model. This probe quantifies local sensitivity in the latent space, serving as a diagnostic tool for identifying vulnerable directions. Based on this signal, we construct effective jailbreak trajectories, giving rise to the Activation Steering Attack (ASA). More importantly, these insights offer a principled foundation for improving alignment robustness. To this end, we introduce Layer-wise Adversarial Patch Training~(LAPT), a fine-tuning strategy that inject controlled perturbations into hidden representations during training. Experimental results highlight that LAPT strengthen alignment robustness without compromising general capabilities. Our findings reveal fundamental flaws in current alignment paradigms and call for representation-level training strategies that move beyond surface-level behavior supervision. Codes and results are available at this https URL. 

**Abstract (ZH)**: 安全性对齐是构建可靠的人工通用智能的关键要求。尽管在安全性对齐方面取得了显著进展，但我们观察到，现有的对齐方法仍可能因细微的潜在转换而引发安全响应。我们认为，这是由于现有对齐方法的浅层性质，它们侧重于表面级别的拒绝行为，而未能充分改变内部表示。因此，隐藏激活的小规模变化可以重新触发嵌入在潜在空间中的有害行为。为了探索安全性对齐对潜在扰动的稳健性，我们介绍了一种探测方法，该方法测量模型生成的原始响应的负对数似然。该探测量化了潜在空间中的局部敏感性，作为诊断工具，以识别脆弱的方向。基于此信号，我们构建了有效的监狱突破轨迹，产生了激活导向攻击（ASA）。更重要的是，这些见解为提高对齐稳健性提供了规范性的基础。为此，我们引入了逐层对抗补丁训练（LAPT），这是一种在训练过程中注入受控扰动以改变隐藏表示的微调策略。实验结果表明，LAPT增强了对齐稳健性而不损害一般能力。我们的发现揭示了当前对齐范式的根本缺陷，并呼吁超越表面行为监督的表示层面训练策略。代码和结果可在此处访问。 

---
# CRIA: A Cross-View Interaction and Instance-Adapted Pre-training Framework for Generalizable EEG Representations 

**Title (ZH)**: CRIA：一种跨视角交互和实例适配的通用EEG表示预训练框架 

**Authors**: Puchun Liu, C. L. Philip Chen, Yubin He, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16056)  

**Abstract**: The difficulty of extracting deep features from EEG data and effectively integrating information from multiple views presents significant challenges for developing a generalizable pretraining framework for EEG representation learning. However, most existing pre-training methods rely solely on the contextual semantics of a single view, failing to capture the complex and synergistic interactions among different perspectives, limiting the expressiveness and generalization of learned representations. To address these issues, this paper proposes CRIA, an adaptive framework that utilizes variable-length and variable-channel coding to achieve a unified representation of EEG data across different datasets. In this work, we define cross-view information as the integrated representation that emerges from the interaction among temporal, spectral, and spatial views of EEG signals. The model employs a cross-attention mechanism to fuse temporal, spectral, and spatial features effectively, and combines an attention matrix masking strategy based on the information bottleneck principle with a novel viewpoint masking pre-training scheme. Experimental results on the Temple University EEG corpus and the CHB-MIT dataset show that CRIA outperforms existing methods with the same pre-training conditions, achieving a balanced accuracy of 57.02% for multi-class event classification and 80.03% for anomaly detection, highlighting its strong generalization ability. 

**Abstract (ZH)**: EEG表示学习的通用预训练框架中的跨视图信息提取与整合挑战及解决方案：CRIA方法 

---
# A Hybrid DeBERTa and Gated Broad Learning System for Cyberbullying Detection in English Text 

**Title (ZH)**: 一种结合DeBERTa和门控广义学习系统的网络欺凌检测方法应用于英文文本 

**Authors**: Devesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.16052)  

**Abstract**: The proliferation of online communication platforms has created unprecedented opportunities for global connectivity while simultaneously enabling harmful behaviors such as cyberbullying, which affects approximately 54.4\% of teenagers according to recent research. This paper presents a hybrid architecture that combines the contextual understanding capabilities of transformer-based models with the pattern recognition strengths of broad learning systems for effective cyberbullying detection. This approach integrates a modified DeBERTa model augmented with Squeeze-and-Excitation blocks and sentiment analysis capabilities with a Gated Broad Learning System (GBLS) classifier, creating a synergistic framework that outperforms existing approaches across multiple benchmark datasets. The proposed ModifiedDeBERTa + GBLS model achieved good performance on four English datasets: 79.3\% accuracy on HateXplain, 95.41\% accuracy on SOSNet, 91.37\% accuracy on Mendeley-I, and 94.67\% accuracy on Mendeley-II. Beyond performance gains, the framework incorporates comprehensive explainability mechanisms including token-level attribution analysis, LIME-based local interpretations, and confidence calibration, addressing critical transparency requirements in automated content moderation. Ablation studies confirm the meaningful contribution of each architectural component, while failure case analysis reveals specific challenges in detecting implicit bias and sarcastic content, providing valuable insights for future improvements in cyberbullying detection systems. 

**Abstract (ZH)**: 在线通信平台的普及为全球连接创造了前所未有的机会，同时也促进了诸如网络欺凌等有害行为，根据最新研究，网络欺凌影响了大约54.4%的青少年。本文提出了一种混合架构，该架构结合了基于变换器的模型的语境理解能力和广义学习系统在模式识别方面的优点，以实现有效的网络欺凌检测。该方法将一个修改后的DeBERTa模型与Squeeze-and-Excitation块和情感分析能力相结合，并与门控广义学习系统（GBLS）分类器集成，形成一个协同工作框架，在多个基准数据集上的性能优于现有方法。提出的ModifiedDeBERTa + GBLS模型在四个英语数据集上的表现良好：在HateXplain上的准确率为79.3%，在SOSNet上的准确率为95.41%，在Mendeley-I上的准确率为91.37%，在Mendeley-II上的准确率为94.67%。除了性能提升，该框架还包含全面的可解释性机制，包括 token 级别归属分析、基于 LIME 的局部解释和置信度校准，以解决自动内容审核中关键的透明度要求。消融研究表明，每个架构组件的贡献具有实质性意义，而故障案例分析揭示了检测隐含偏见和讽刺内容的具体挑战，为未来网络欺凌检测系统的改进提供了宝贵的见解。 

---
# DynScaling: Efficient Verifier-free Inference Scaling via Dynamic and Integrated Sampling 

**Title (ZH)**: DynScaling: 有效的无验证器推理放大通过动态集成采样 

**Authors**: Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, Sercan Ö. Arık  

**Link**: [PDF](https://arxiv.org/pdf/2506.16043)  

**Abstract**: Inference-time scaling has proven effective in boosting large language model (LLM) performance through increased test-time computation. Yet, its practical application is often hindered by reliance on external verifiers or a lack of optimization for realistic computational constraints. We propose DynScaling, which addresses these limitations through two primary innovations: an integrated parallel-sequential sampling strategy and a bandit-based dynamic budget allocation framework. The integrated sampling strategy unifies parallel and sequential sampling by constructing synthetic sequential reasoning chains from initially independent parallel responses, promoting diverse and coherent reasoning trajectories. The dynamic budget allocation framework formulates the allocation of computational resources as a multi-armed bandit problem, adaptively distributing the inference budget across queries based on the uncertainty of previously sampled responses, thereby maximizing computational efficiency. By combining these components, DynScaling effectively improves LLM performance under practical resource constraints without the need for external verifiers. Experimental results demonstrate that DynScaling consistently surpasses existing verifier-free inference scaling baselines in both task performance and computational cost. 

**Abstract (ZH)**: Inference时的动态缩放通过集成并行序列采样策略和基于bandit的动态预算分配框架有效提升大语言模型性能 

---
# Vision-Guided Chunking Is All You Need: Enhancing RAG with Multimodal Document Understanding 

**Title (ZH)**: 基于视觉引导的片段化理解：提升RAG的多模态文档理解 

**Authors**: Vishesh Tripathi, Tanmay Odapally, Indraneel Das, Uday Allu, Biddwan Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2506.16035)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have revolutionized information retrieval and question answering, but traditional text-based chunking methods struggle with complex document structures, multi-page tables, embedded figures, and contextual dependencies across page boundaries. We present a novel multimodal document chunking approach that leverages Large Multimodal Models (LMMs) to process PDF documents in batches while maintaining semantic coherence and structural integrity. Our method processes documents in configurable page batches with cross-batch context preservation, enabling accurate handling of tables spanning multiple pages, embedded visual elements, and procedural content. We evaluate our approach on a curated dataset of PDF documents with manually crafted queries, demonstrating improvements in chunk quality and downstream RAG performance. Our vision-guided approach achieves better accuracy compared to traditional vanilla RAG systems, with qualitative analysis showing superior preservation of document structure and semantic coherence. 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG) 系统通过利用大规模多模态模型对 PDF 文档进行批量处理，实现了信息检索和问答的革命性变化，但传统基于文本的片段化方法难以处理复杂的文档结构、跨页面的多页表格、嵌入的图形以及页面边界处的上下文依赖关系。我们提出了一种新颖的多模态文档片段化方法，利用大规模多模态模型在保持语义连贯性和结构完整性的同时对 PDF 文档进行批量处理。该方法以可配置的页面批处理方式进行处理，同时保留跨批处理的上下文，从而能够准确处理跨页面的表格、嵌入的视觉元素以及程序性内容。我们使用精心编写的 PDF 文档数据集和手动构建的问题对我们的方法进行了评估，展示了片段质量和下游 RAG 性能的提升。与传统的基线 RAG 系统相比，我们的基于视觉指导的方法在准确性上更胜一筹，定性分析表明其在保持文档结构和语义连贯性方面表现更优。 

---
# EvoLM: In Search of Lost Language Model Training Dynamics 

**Title (ZH)**: EvoLM: 寻找丢失的语言模型训练动态 

**Authors**: Zhenting Qi, Fan Nie, Alexandre Alahi, James Zou, Himabindu Lakkaraju, Yilun Du, Eric Xing, Sham Kakade, Hanlin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16029)  

**Abstract**: Modern language model (LM) training has been divided into multiple stages, making it difficult for downstream developers to evaluate the impact of design choices made at each stage. We present EvoLM, a model suite that enables systematic and transparent analysis of LMs' training dynamics across pre-training, continued pre-training, supervised fine-tuning, and reinforcement learning. By training over 100 LMs with 1B and 4B parameters from scratch, we rigorously evaluate both upstream (language modeling) and downstream (problem-solving) reasoning capabilities, including considerations of both in-domain and out-of-domain generalization. Key insights highlight the diminishing returns from excessive pre-training and post-training, the importance and practices of mitigating forgetting during domain-specific continued pre-training, the crucial role of continued pre-training in bridging pre-training and post-training phases, and various intricate trade-offs when configuring supervised fine-tuning and reinforcement learning. To facilitate open research and reproducibility, we release all pre-trained and post-trained models, training datasets for all stages, and our entire training and evaluation pipeline. 

**Abstract (ZH)**: 现代语言模型（LM）训练被划分为多个阶段，给下游开发者评估每个阶段设计选择的影响带来了困难。我们提出了EvoLM，一种模型套件，用于系统性和透明地分析语言模型从预训练、持续预训练、监督微调到强化学习的训练动力学。通过从头训练超过100个参数量为1B和4B的语言模型，我们严格评估了其上游（语言建模）和下游（问题解决）的推理能力，包括领域内和领域外泛化的考量。关键洞察揭示了过度预训练和后续训练的边际收益递减现象，领域特定持续预训练过程中防止遗忘的重要性及其实现方法，持续预训练在连接预训练和后续训练阶段中的关键作用，以及配置监督微调和强化学习时的各种复杂权衡。为促进开放研究和可重复性，我们发布了所有预训练和后续训练的模型、各阶段的训练数据集以及整个训练和评估流程。 

---
# From General to Targeted Rewards: Surpassing GPT-4 in Open-Ended Long-Context Generation 

**Title (ZH)**: 从一般到针对性奖励：超越GPT-4的开放生成长上下文任务 

**Authors**: Zhihan Guo, Jiele Wu, Wenqian Cui, Yifei Zhang, Minda Hu, Yufei Wang, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2506.16024)  

**Abstract**: Current research on long-form context in Large Language Models (LLMs) primarily focuses on the understanding of long-contexts, the Open-ended Long Text Generation (Open-LTG) remains insufficiently explored. Training a long-context generation model requires curation of gold standard reference data, which is typically nonexistent for informative Open-LTG tasks. However, previous methods only utilize general assessments as reward signals, which limits accuracy. To bridge this gap, we introduce ProxyReward, an innovative reinforcement learning (RL) based framework, which includes a dataset and a reward signal computation method. Firstly, ProxyReward Dataset generation is accomplished through simple prompts that enables the model to create automatically, obviating extensive labeled data or significant manual effort. Secondly, ProxyReward Signal offers a targeted evaluation of information comprehensiveness and accuracy for specific questions. The experimental results indicate that our method ProxyReward surpasses even GPT-4-Turbo. It can significantly enhance performance by 20% on the Open-LTG task when training widely used open-source models, while also surpassing the LLM-as-a-Judge approach. Our work presents effective methods to enhance the ability of LLMs to address complex open-ended questions posed by human. 

**Abstract (ZH)**: 当前对大型语言模型（LLMs）长形式上下文的研究主要集中在长上下文的理解上，开放性的长文本生成（Open-LTG）仍然缺乏充分探索。训练长上下文生成模型需要高质量参考数据的整理，而在信息性Open-LTG任务中此类数据通常不存在。尽管之前的 方法仅利用一般评估作为奖励信号，限制了准确性。为填补这一空白，我们提出了一种创新的基于强化学习（RL）的ProxyReward框架，包括一个数据集和奖励信号计算方法。首先，通过简单的提示生成ProxyReward数据集，无需大量标注数据或显著的手工努力。其次，ProxyReward信号提供了对特定问题的信息全面性和准确性进行针对性评估的方法。实验结果表明，我们的方法ProxyReward甚至超过了GPT-4-Turbo。在训练广泛使用的开源模型时，它可以显著提高Open-LTG任务的性能达20%，同时也超越了LLM作为评判者的途径。我们的工作展示了增强LLMs应对复杂开放性问题能力的有效方法。 

---
# VRAIL: Vectorized Reward-based Attribution for Interpretable Learning 

**Title (ZH)**: VRAIL：向量奖励归因的可解释学习 

**Authors**: Jina Kim, Youjin Jang, Jeongjin Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.16014)  

**Abstract**: We propose VRAIL (Vectorized Reward-based Attribution for Interpretable Learning), a bi-level framework for value-based reinforcement learning (RL) that learns interpretable weight representations from state features. VRAIL consists of two stages: a deep learning (DL) stage that fits an estimated value function using state features, and an RL stage that uses this to shape learning via potential-based reward transformations. The estimator is modeled in either linear or quadratic form, allowing attribution of importance to individual features and their interactions. Empirical results on the Taxi-v3 environment demonstrate that VRAIL improves training stability and convergence compared to standard DQN, without requiring environment modifications. Further analysis shows that VRAIL uncovers semantically meaningful subgoals, such as passenger possession, highlighting its ability to produce human-interpretable behavior. Our findings suggest that VRAIL serves as a general, model-agnostic framework for reward shaping that enhances both learning and interpretability. 

**Abstract (ZH)**: 基于价值向量化奖励归因的可解释学习双层框架：VRAIL 

---
# DIGMAPPER: A Modular System for Automated Geologic Map Digitization 

**Title (ZH)**: DIGMAPPER：一种自动化地质图数字化的模块化系统 

**Authors**: Weiwei Duan, Michael P. Gerlek, Steven N. Minton, Craig A. Knoblock, Fandel Lin, Theresa Chen, Leeje Jang, Sofia Kirsanova, Zekun Li, Yijun Lin, Yao-Yi Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.16006)  

**Abstract**: Historical geologic maps contain rich geospatial information, such as rock units, faults, folds, and bedding planes, that is critical for assessing mineral resources essential to renewable energy, electric vehicles, and national security. However, digitizing maps remains a labor-intensive and time-consuming task. We present DIGMAPPER, a modular, scalable system developed in collaboration with the United States Geological Survey (USGS) to automate the digitization of geologic maps. DIGMAPPER features a fully dockerized, workflow-orchestrated architecture that integrates state-of-the-art deep learning models for map layout analysis, feature extraction, and georeferencing. To overcome challenges such as limited training data and complex visual content, our system employs innovative techniques, including in-context learning with large language models, synthetic data generation, and transformer-based models. Evaluations on over 100 annotated maps from the DARPA-USGS dataset demonstrate high accuracy across polygon, line, and point feature extraction, and reliable georeferencing performance. Deployed at USGS, DIGMAPPER significantly accelerates the creation of analysis-ready geospatial datasets, supporting national-scale critical mineral assessments and broader geoscientific applications. 

**Abstract (ZH)**: 历史地质图包含丰富的空间信息，如岩层单位、断层、褶皱和层面，这些信息对于评估对可再生能源、电动汽车和国家安全至关重要的矿产资源至关重要。然而，地图数字化仍然是一个劳动密集型和耗时的过程。我们提出了一种模块化、可扩展的系统DIGMAPPER，该系统与美国地质调查局（USGS）合作开发，旨在自动数字化地质图。DIGMAPPER具备完整的Docker化、工作流协调架构，集成了最先进的深度学习模型进行地图布局分析、特征提取和地理参照。为了克服有限的训练数据和复杂的视觉内容等挑战，我们的系统采用了创新技巧，包括上下文学习、合成数据生成和基于 transformer 的模型。在超过100张标注的地图上进行的评估显示出在多边形、线性和点特征提取方面的高精度，并且具有可靠的地理参照性能。在美国地质调查局部署后，DIGMAPPER显著加速了分析准备好空间数据集的创建，支持全国范围内的关键矿产评估和更广泛的地质科学应用。 

---
# AutoHFormer: Efficient Hierarchical Autoregressive Transformer for Time Series Prediction 

**Title (ZH)**: AutoHFormer: 效率较高的层次自回归Transformer在时间序列预测中的应用 

**Authors**: Qianru Zhang, Honggang Wen, Ming Li, Dong Huang, Siu-Ming Yiu, Christian S. Jensen, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2506.16001)  

**Abstract**: Time series forecasting requires architectures that simultaneously achieve three competing objectives: (1) strict temporal causality for reliable predictions, (2) sub-quadratic complexity for practical scalability, and (3) multi-scale pattern recognition for accurate long-horizon forecasting. We introduce AutoHFormer, a hierarchical autoregressive transformer that addresses these challenges through three key innovations: 1) Hierarchical Temporal Modeling: Our architecture decomposes predictions into segment-level blocks processed in parallel, followed by intra-segment sequential refinement. This dual-scale approach maintains temporal coherence while enabling efficient computation. 2) Dynamic Windowed Attention: The attention mechanism employs learnable causal windows with exponential decay, reducing complexity while preserving precise temporal relationships. This design avoids both the anti-causal violations of standard transformers and the sequential bottlenecks of RNN hybrids. 3) Adaptive Temporal Encoding: a novel position encoding system is adopted to capture time patterns at multiple scales. It combines fixed oscillating patterns for short-term variations with learnable decay rates for long-term trends. Comprehensive experiments demonstrate that AutoHFormer 10.76X faster training and 6.06X memory reduction compared to PatchTST on PEMS08, while maintaining consistent accuracy across 96-720 step horizons in most of cases. These breakthroughs establish new benchmarks for efficient and precise time series modeling. Implementations of our method and all baselines in hierarchical autoregressive mechanism are available at this https URL. 

**Abstract (ZH)**: 时间序列预测需要能够在同时实现三个竞争性目标的架构：（1）严格的时序因果性以获得可靠的预测，（2）次二次复杂性以实现实用的可扩展性，（3）多尺度模式识别以实现精确的长时程预测。我们引入了AutoHFormer，这是一种通过三项关键创新解决这些挑战的自回归变压器：1）多层次时序建模：我们的架构将预测分解为并行处理的段级块，随后进行段内的顺序细化。这种双尺度方法保持了时序一致性的同时，提高了计算效率。2）动态窗口注意力：注意力机制采用可学习的因果窗口，并具有指数衰减，从而减少复杂性同时保留精确的时序关系。该设计避免了标准变压器的反因果违反和RNN混合模型的顺序瓶颈。3）自适应时序编码：采用了一种新的一位编码系统来捕捉多尺度的时间模式。它结合了固定振荡模式来捕捉短期变化，并且具有可学习的衰减速率来捕捉长期趋势。全面的实验证明，与PatchTST在PEMS08数据集上的相比，AutoHFormer在大多数情况下保持一致的准确性的同时，训练速度提高了10.76倍，内存减少了6.06倍。这些突破性进展为高效和精确的时间序列建模设定了新的基准。我们的方法及其在多层次自回归机制中的所有基线的实现可在以下网址获取：这个 https URL。 

---
# Quantum Artificial Intelligence for Secure Autonomous Vehicle Navigation: An Architectural Proposal 

**Title (ZH)**: 量子人工智能在安全自主车辆导航中的应用：一种架构提案 

**Authors**: Hemanth Kannamarlapudi, Sowmya Chintalapudi  

**Link**: [PDF](https://arxiv.org/pdf/2506.16000)  

**Abstract**: Navigation is a very crucial aspect of autonomous vehicle ecosystem which heavily relies on collecting and processing large amounts of data in various states and taking a confident and safe decision to define the next vehicle maneuver. In this paper, we propose a novel architecture based on Quantum Artificial Intelligence by enabling quantum and AI at various levels of navigation decision making and communication process in Autonomous vehicles : Quantum Neural Networks for multimodal sensor fusion, Nav-Q for Quantum reinforcement learning for navigation policy optimization and finally post-quantum cryptographic protocols for secure communication. Quantum neural networks uses quantum amplitude encoding to fuse data from various sensors like LiDAR, radar, camera, GPS and weather etc., This approach gives a unified quantum state representation between heterogeneous sensor modalities. Nav-Q module processes the fused quantum states through variational quantum circuits to learn optimal navigation policies under swift dynamic and complex conditions. Finally, post quantum cryptographic protocols are used to secure communication channels for both within vehicle communication and V2X (Vehicle to Everything) communications and thus secures the autonomous vehicle communication from both classical and quantum security threats. Thus, the proposed framework addresses fundamental challenges in autonomous vehicles navigation by providing quantum performance and future proof security. Index Terms Quantum Computing, Autonomous Vehicles, Sensor Fusion 

**Abstract (ZH)**: 基于量子人工智能的自主车辆导航新型架构：量子神经网络多模传感器融合、Nav-Q量子强化学习导航策略优化及后量子加密协议安全通信 

---
# Double Entendre: Robust Audio-Based AI-Generated Lyrics Detection via Multi-View Fusion 

**Title (ZH)**: 双关语：基于多视图融合的鲁棒音频生成歌词检测 

**Authors**: Markus Frohmann, Gabriel Meseguer-Brocal, Markus Schedl, Elena V. Epure  

**Link**: [PDF](https://arxiv.org/pdf/2506.15981)  

**Abstract**: The rapid advancement of AI-based music generation tools is revolutionizing the music industry but also posing challenges to artists, copyright holders, and providers alike. This necessitates reliable methods for detecting such AI-generated content. However, existing detectors, relying on either audio or lyrics, face key practical limitations: audio-based detectors fail to generalize to new or unseen generators and are vulnerable to audio perturbations; lyrics-based methods require cleanly formatted and accurate lyrics, unavailable in practice. To overcome these limitations, we propose a novel, practically grounded approach: a multimodal, modular late-fusion pipeline that combines automatically transcribed sung lyrics and speech features capturing lyrics-related information within the audio. By relying on lyrical aspects directly from audio, our method enhances robustness, mitigates susceptibility to low-level artifacts, and enables practical applicability. Experiments show that our method, DE-detect, outperforms existing lyrics-based detectors while also being more robust to audio perturbations. Thus, it offers an effective, robust solution for detecting AI-generated music in real-world scenarios. Our code is available at this https URL. 

**Abstract (ZH)**: 基于AI的音乐生成工具的迅速发展正在颠覆音乐行业，同时也给艺术家、版权持有者和提供者带来了挑战。这 necessitates可靠的方法来检测此类AI生成的内容。然而，现有的检测器要么依赖于音频要么依赖于歌词，都面临着关键的实际限制：基于音频的检测器无法泛化到新的或未见过的生成器，并且容易受到音频干扰；基于歌词的方法需要格式干净且准确的歌词，在实践中难以获得。为克服这些限制，我们提出了一种新的、实际可行的方法：一个结合自动转录的歌词和捕捉与歌词相关音频特征的多模态模块化晚期融合管道。通过直接依赖于音频中的歌词方面，我们的方法增强了鲁棒性，减轻了对低级伪影的敏感性，并使得实际应用成为可能。实验表明，我们的方法DE-detect在性能上优于现有的基于歌词的检测器，并且对音频干扰具有更高的鲁棒性。因此，它提供了一种有效的、稳健的解决方案，用于实际场景中检测AI生成的音乐。我们的代码可以在此处访问： this https URL。 

---
# Advanced Sign Language Video Generation with Compressed and Quantized Multi-Condition Tokenization 

**Title (ZH)**: 基于压缩和量化多条件令牌化的方法改进手语视频生成 

**Authors**: Cong Wang, Zexuan Deng, Zhiwei Jiang, Fei Shen, Yafeng Yin, Shiwei Gan, Zifeng Cheng, Shiping Ge, Qing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15980)  

**Abstract**: Sign Language Video Generation (SLVG) seeks to generate identity-preserving sign language videos from spoken language texts. Existing methods primarily rely on the single coarse condition (\eg, skeleton sequences) as the intermediary to bridge the translation model and the video generation model, which limits both the naturalness and expressiveness of the generated videos. To overcome these limitations, we propose SignViP, a novel SLVG framework that incorporates multiple fine-grained conditions for improved generation fidelity. Rather than directly translating error-prone high-dimensional conditions, SignViP adopts a discrete tokenization paradigm to integrate and represent fine-grained conditions (\ie, fine-grained poses and 3D hands). SignViP contains three core components. (1) Sign Video Diffusion Model is jointly trained with a multi-condition encoder to learn continuous embeddings that encapsulate fine-grained motion and appearance. (2) Finite Scalar Quantization (FSQ) Autoencoder is further trained to compress and quantize these embeddings into discrete tokens for compact representation of the conditions. (3) Multi-Condition Token Translator is trained to translate spoken language text to discrete multi-condition tokens. During inference, Multi-Condition Token Translator first translates the spoken language text into discrete multi-condition tokens. These tokens are then decoded to continuous embeddings by FSQ Autoencoder, which are subsequently injected into Sign Video Diffusion Model to guide video generation. Experimental results show that SignViP achieves state-of-the-art performance across metrics, including video quality, temporal coherence, and semantic fidelity. The code is available at this https URL. 

**Abstract (ZH)**: 手语视频生成（SLVG）旨在从口头语言文本生成保有身份的手语视频。现有的方法主要依赖单一粗粒度条件（例如，骨架序列）作为中介来连接翻译模型和视频生成模型，这限制了生成视频的自然性和表现力。为克服这些限制，我们提出了一种新颖的手语视频生成框架SignViP，该框架结合了多细粒度条件以提高生成保真度。SignViP 不直接翻译易出错的高维条件，而是采用离散标记化范式来整合和表示细粒度条件（即，细粒度姿态和3D手部）。SignViP 包含三个核心组件：（1）手语视频扩散模型，与多条件编码器联合训练，学习包含细粒度运动和外观的连续嵌入；（2）有限标量量化（FSQ）自编码器进一步训练以压缩和量化这些嵌入成离散标记，以紧凑地表示条件；（3）多条件标记翻译器，训练以将口头语言文本翻译为离散的多条件标记。在推理过程中，多条件标记翻译器首先将口头语言文本翻译为离散的多条件标记，这些标记由FSQ自编码器解码为连续嵌入，并随后注入到手语视频扩散模型中以指导视频生成。实验结果表明，SignViP 在包括视频质量、时间连贯性和语义保真度的指标上达到了最先进的性能。代码可在以下链接中获取：this https URL。 

---
# A Vietnamese Dataset for Text Segmentation and Multiple Choices Reading Comprehension 

**Title (ZH)**: 越南语语料库用于文本分段和多项选择阅读理解 

**Authors**: Toan Nguyen Hai, Ha Nguyen Viet, Truong Quan Xuan, Duc Do Minh  

**Link**: [PDF](https://arxiv.org/pdf/2506.15978)  

**Abstract**: Vietnamese, the 20th most spoken language with over 102 million native speakers, lacks robust resources for key natural language processing tasks such as text segmentation and machine reading comprehension (MRC). To address this gap, we present VSMRC, the Vietnamese Text Segmentation and Multiple-Choice Reading Comprehension Dataset. Sourced from Vietnamese Wikipedia, our dataset includes 15,942 documents for text segmentation and 16,347 synthetic multiple-choice question-answer pairs generated with human quality assurance, ensuring a reliable and diverse resource. Experiments show that mBERT consistently outperforms monolingual models on both tasks, achieving an accuracy of 88.01% on MRC test set and an F1 score of 63.15\% on text segmentation test set. Our analysis reveals that multilingual models excel in NLP tasks for Vietnamese, suggesting potential applications to other under-resourced languages. VSMRC is available at HuggingFace 

**Abstract (ZH)**: Vietnamese语资源欠缺，尤其是在文本分割和机器阅读理解等关键自然语言处理任务方面，拥有超过1.02亿母语使用者的越南语排名世界第20位。为解决这一问题，我们提出了VSMRC：越南语文本分割与多项选择阅读理解数据集。该数据集源自越南维基百科，包含15,942份用于文本分割的文档和16,347个人工质量保障生成的多项选择题-答案对，确保资源可靠且多样。实验结果显示，mBERT在两项任务上均优于单一语言模型，分别在机器阅读理解测试集上达到88.01%的准确率和在文本分割测试集上达到63.15%的F1分数。我们的分析表明，多语言模型在越南语的NLP任务中表现优异，这可能对其他资源匮乏的语言也有潜力应用。VSMRC可在HuggingFace获取。 

---
# Heterogeneous-Modal Unsupervised Domain Adaptation via Latent Space Bridging 

**Title (ZH)**: 跨模态无监督领域适应通过潜在空间桥梁 

**Authors**: Jiawen Yang, Shuhao Chen, Yucong Duan, Ke Tang, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15971)  

**Abstract**: Unsupervised domain adaptation (UDA) methods effectively bridge domain gaps but become struggled when the source and target domains belong to entirely distinct modalities. To address this limitation, we propose a novel setting called Heterogeneous-Modal Unsupervised Domain Adaptation (HMUDA), which enables knowledge transfer between completely different modalities by leveraging a bridge domain containing unlabeled samples from both modalities. To learn under the HMUDA setting, we propose Latent Space Bridging (LSB), a specialized framework designed for the semantic segmentation task. Specifically, LSB utilizes a dual-branch architecture, incorporating a feature consistency loss to align representations across modalities and a domain alignment loss to reduce discrepancies between class centroids across domains. Extensive experiments conducted on six benchmark datasets demonstrate that LSB achieves state-of-the-art performance. 

**Abstract (ZH)**: 异质模态无监督领域适应（HMUDA） 

---
# TrainVerify: Equivalence-Based Verification for Distributed LLM Training 

**Title (ZH)**: TrainVerify: 基于等价性的分布式LLM训练验证 

**Authors**: Yunchi Lu, Youshan Miao, Cheng Tan, Peng Huang, Yi Zhu, Xian Zhang, Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15961)  

**Abstract**: Training large language models (LLMs) at scale requires parallel execution across thousands of devices, incurring enormous computational costs. Yet, these costly distributed trainings are rarely verified, leaving them prone to silent errors and potentially wasting millions of GPU hours. We introduce TrainVerify, a system for verifiable distributed training of LLMs. Given a deep learning model's logical specification as the ground truth, TrainVerify formally verifies that a distributed parallel execution plan is mathematically equivalent to it. Direct verification is notoriously difficult due to the sheer scale of LLMs which often involves billions of variables and highly intricate computation graphs. Therefore, TrainVerify introduces shape-reduction techniques and a stage-wise parallel verification algorithm that significantly reduces complexity while preserving formal correctness. TrainVerify scales to frontier LLMs, including the successful verification of the Llama3 (405B) and DeepSeek-V3 (671B) training plans. 

**Abstract (ZH)**: 大规模训练语言模型（LLMs）需要在数千台设备上并行执行，产生巨大的计算成本。然而，这些昂贵的分布式训练很少被验证，这使得它们容易出现无声错误，可能浪费数百万个GPU小时。我们引入了TrainVerify系统，用于可验证的LLMs分布式训练。Given一个深度学习模型的逻辑规范作为ground truth，TrainVerify正式验证分布式并行执行计划与之在数学上等价。直接验证由于LLMs的规模巨大，通常涉及数十亿个变量和高度复杂的计算图而极困难。因此，TrainVerify引入了形状缩减技术及阶段式并行验证算法，显著降低了复杂度同时保持形式上的正确性。TrainVerify能够扩展到前沿的LLMs，包括Llama3（405B）和DeepSeek-V3（671B）训练计划的成功验证。 

---
# Beyond Audio and Pose: A General-Purpose Framework for Video Synchronization 

**Title (ZH)**: 超越音频和姿势：一种通用视频同步框架 

**Authors**: Yosub Shin, Igor Molybog  

**Link**: [PDF](https://arxiv.org/pdf/2506.15937)  

**Abstract**: Video synchronization-aligning multiple video streams capturing the same event from different angles-is crucial for applications such as reality TV show production, sports analysis, surveillance, and autonomous systems. Prior work has heavily relied on audio cues or specific visual events, limiting applicability in diverse settings where such signals may be unreliable or absent. Additionally, existing benchmarks for video synchronization lack generality and reproducibility, restricting progress in the field. In this work, we introduce VideoSync, a video synchronization framework that operates independently of specific feature extraction methods, such as human pose estimation, enabling broader applicability across different content types. We evaluate our system on newly composed datasets covering single-human, multi-human, and non-human scenarios, providing both the methodology and code for dataset creation to establish reproducible benchmarks. Our analysis reveals biases in prior SOTA work, particularly in SeSyn-Net's preprocessing pipeline, leading to inflated performance claims. We correct these biases and propose a more rigorous evaluation framework, demonstrating that VideoSync outperforms existing approaches, including SeSyn-Net, under fair experimental conditions. Additionally, we explore various synchronization offset prediction methods, identifying a convolutional neural network (CNN)-based model as the most effective. Our findings advance video synchronization beyond domain-specific constraints, making it more generalizable and robust for real-world applications. 

**Abstract (ZH)**: 视频同步：多角度捕捉同一事件的视频流同步对于实景电视节目制作、体育分析、监控和自主系统等应用至关重要。先前的工作主要依赖于音频线索或特定的视觉事件，限制了其在信号不可靠或缺失的多样环境中的适用性。此外，现有的视频同步基准缺乏通用性和可重复性，限制了该领域的进步。在本工作中，我们提出了VideoSync，这是一种独立于特定特征提取方法的视频同步框架，如人体姿态估计，使其在不同类型的内容中具有更广泛的适用性。我们使用新编纂的数据集评估了系统的表现，这些数据集覆盖了单人、多人和非人场景，提供了数据集创建的方法和代码，以建立可重复的基准。我们的分析揭示了先前最佳方法中的偏见，特别是在SeSyn-Net的预处理管道中，导致了夸大了的性能声称。我们纠正了这些偏见，并提出了一种更严格的评估框架，展示了在公平的实验条件下，VideoSync优于现有方法，包括SeSyn-Net。此外，我们探索了各种同步偏移预测方法，发现基于卷积神经网络（CNN）的模型最有效。我们的发现推动了视频同步的发展，使其超越了特定领域的限制，更具通用性和鲁棒性，适用于实际应用。 

---
# MoiréXNet: Adaptive Multi-Scale Demoiréing with Linear Attention Test-Time Training and Truncated Flow Matching Prior 

**Title (ZH)**: MoiréXNet：自适应多尺度消moire处理的线性注意力测试时训练及截断流匹配先验 

**Authors**: Liangyan Li, Yimo Ning, Kevin Le, Wei Dong, Yunzhe Li, Jun Chen, Xiaohong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15929)  

**Abstract**: This paper introduces a novel framework for image and video demoiréing by integrating Maximum A Posteriori (MAP) estimation with advanced deep learning techniques. Demoiréing addresses inherently nonlinear degradation processes, which pose significant challenges for existing methods.
Traditional supervised learning approaches either fail to remove moiré patterns completely or produce overly smooth results. This stems from constrained model capacity and scarce training data, which inadequately represent the clean image distribution and hinder accurate reconstruction of ground-truth images. While generative models excel in image restoration for linear degradations, they struggle with nonlinear cases such as demoiréing and often introduce artifacts.
To address these limitations, we propose a hybrid MAP-based framework that integrates two complementary components. The first is a supervised learning model enhanced with efficient linear attention Test-Time Training (TTT) modules, which directly learn nonlinear mappings for RAW-to-sRGB demoiréing. The second is a Truncated Flow Matching Prior (TFMP) that further refines the outputs by aligning them with the clean image distribution, effectively restoring high-frequency details and suppressing artifacts. These two components combine the computational efficiency of linear attention with the refinement abilities of generative models, resulting in improved restoration performance. 

**Abstract (ZH)**: 一种结合最大后验估计与先进深度学习技术的新型图像和视频反摩尔纹框架 

---
# PNCS:Power-Norm Cosine Similarity for Diverse Client Selection in Federated Learning 

**Title (ZH)**: PNCS：Power-Norm余弦相似度在联邦学习中多样客户端选择中的应用 

**Authors**: Liangyan Li, Yangyi Liu, Yimo Ning, Stefano Rini, Jun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.15923)  

**Abstract**: Federated Learning (FL) has emerged as a powerful paradigm for leveraging diverse datasets from multiple sources while preserving data privacy by avoiding centralized storage. However, many existing approaches fail to account for the intricate gradient correlations between remote clients, a limitation that becomes especially problematic in data heterogeneity scenarios. In this work, we propose a novel FL framework utilizing Power-Norm Cosine Similarity (PNCS) to improve client selection for model aggregation. By capturing higher-order gradient moments, PNCS addresses non-IID data challenges, enhancing convergence speed and accuracy. Additionally, we introduce a simple algorithm ensuring diverse client selection through a selection history queue. Experiments with a VGG16 model across varied data partitions demonstrate consistent improvements over state-of-the-art methods. 

**Abstract (ZH)**: 联邦学习（FL）作为一种利用多个数据源的多样化数据集同时保护数据隐私的有力范式，通过避免集中存储数据而崭露头角。然而，许多现有方法未能考虑远处客户端之间复杂的梯度相关性，这一局限在数据异质性场景中表现得尤为突出。在本工作中，我们提出了一种利用Power-Norm余弦相似度（PNCS）的新颖FL框架，以改进模型聚合时的客户端选择。通过捕获高阶梯度矩，PNCS解决了非IID数据的挑战，提高了收敛速度和准确性。此外，我们还引入了一个简单的算法，通过选择历史队列确保客户端选择的多样性。实验结果表明，在多种数据分割下，该方法在VGG16模型上的一致性改进超过了现有最佳方法。 

---
# KG-FGNN: Knowledge-guided GNN Foundation Model for Fertilisation-oriented Soil GHG Flux Prediction 

**Title (ZH)**: 基于知识引导的GNN基础模型：肥料导向的土壤温室气体 Flux 预测 

**Authors**: Yu Zhang, Gaoshan Bi, Simon Jeffery, Max Davis, Yang Li, Qing Xue, Po Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15896)  

**Abstract**: Precision soil greenhouse gas (GHG) flux prediction is essential in agricultural systems for assessing environmental impacts, developing emission mitigation strategies and promoting sustainable agriculture. Due to the lack of advanced sensor and network technologies on majority of farms, there are challenges in obtaining comprehensive and diverse agricultural data. As a result, the scarcity of agricultural data seriously obstructs the application of machine learning approaches in precision soil GHG flux prediction. This research proposes a knowledge-guided graph neural network framework that addresses the above challenges by integrating knowledge embedded in an agricultural process-based model and graph neural network techniques. Specifically, we utilise the agricultural process-based model to simulate and generate multi-dimensional agricultural datasets for 47 countries that cover a wide range of agricultural variables. To extract key agricultural features and integrate correlations among agricultural features in the prediction process, we propose a machine learning framework that integrates the autoencoder and multi-target multi-graph based graph neural networks, which utilises the autoencoder to selectively extract significant agricultural features from the agricultural process-based model simulation data and the graph neural network to integrate correlations among agricultural features for accurately predict fertilisation-oriented soil GHG fluxes. Comprehensive experiments were conducted with both the agricultural simulation dataset and real-world agricultural dataset to evaluate the proposed approach in comparison with well-known baseline and state-of-the-art regression methods. The results demonstrate that our proposed approach provides superior accuracy and stability in fertilisation-oriented soil GHG prediction. 

**Abstract (ZH)**: 基于知识引导的图神经网络框架在精确诊断农田温室气体排放中的应用 

---
# Language Models can perform Single-Utterance Self-Correction of Perturbed Reasoning 

**Title (ZH)**: 语言模型可以对干扰推理进行单句自纠错 

**Authors**: Sam Silver, Jimin Sun, Ivan Zhang, Sara Hooker, Eddie Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.15894)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive mathematical reasoning capabilities, yet their performance remains brittle to minor variations in problem description and prompting strategy. Furthermore, reasoning is vulnerable to sampling-induced errors which autoregressive models must primarily address using self-correction via additionally-generated tokens. To better understand self-correction capabilities of recent models, we conduct experiments measuring models' ability to self-correct synthetic perturbations introduced into their Chain of Thought (CoT) reasoning. We observe robust single-utterance intrinsic self-correction behavior across a range of open-weight models and datasets, ranging from subtle, implicit corrections to explicit acknowledgments and corrections of errors. Our findings suggest that LLMs, including those not finetuned for long CoT, may possess stronger intrinsic self-correction capabilities than commonly shown in the literature. The presence of this ability suggests that recent "reasoning" model work involves amplification of traits already meaningfully present in models. 

**Abstract (ZH)**: 大型语言模型在数学推理方面展现了令人印象深刻的能力，但其性能对问题描述和提示策略的轻微变化仍显得脆弱。此外，推理过程容易受到抽样引起的错误的影响，自回归模型必须通过自动生成的额外令牌进行自我纠正来主要解决这一问题。为了更好地了解 recent 模型的自我纠正能力，我们进行了实验，测量模型在其思维链（CoT）推理中引入合成扰动后的自我纠正能力。我们观察到，从微妙的隐式纠正到明确承认和纠正错误，不同开放权重模型和数据集均表现出稳健的单句内在自我纠正行为。我们的研究结果表明，包括未针对长思维链进行微调的大型语言模型在内，这些模型可能具有比文献中常见显示的更强的内在自我纠正能力。这种能力的存在表明，最近的“推理”模型工作可能是对模型中已显著存在的特质的放大。 

---
# Fractional Reasoning via Latent Steering Vectors Improves Inference Time Compute 

**Title (ZH)**: 通过潜在引导向量的分数推理改进推断时间计算 

**Authors**: Sheng Liu, Tianlang Chen, Pan Lu, Haotian Ye, Yizheng Chen, Lei Xing, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2506.15882)  

**Abstract**: Test-time compute has emerged as a powerful paradigm for improving the performance of large language models (LLMs), where generating multiple outputs or refining individual chains can significantly boost answer accuracy. However, existing methods like Best-of-N, majority voting, and self-reflection typically apply reasoning in a uniform way across inputs, overlooking the fact that different problems may require different levels of reasoning depth. In this work, we propose Fractional Reasoning, a training-free and model-agnostic framework that enables continuous control over reasoning intensity at inference time, going beyond the limitations of fixed instructional prompts. Our method operates by extracting the latent steering vector associated with deeper reasoning and reapplying it with a tunable scaling factor, allowing the model to tailor its reasoning process to the complexity of each input. This supports two key modes of test-time scaling: (1) improving output quality in breadth-based strategies (e.g., Best-of-N, majority voting), and (2) enhancing the correctness of individual reasoning chains in depth-based strategies (e.g., self-reflection). Experiments on GSM8K, MATH500, and GPQA demonstrate that Fractional Reasoning consistently improves performance across diverse reasoning tasks and models. 

**Abstract (ZH)**: 测试时计算已成为提升大型语言模型性能的强大范式，通过生成多个输出或细化个体推理链可以显著提高答案准确性。然而，现有方法如“最佳的N个结果”、多数投票和自我反思通常以统一的方式应用于输入，忽视了不同问题可能需要不同推理深度的事实。在此工作中，我们提出了分数推理（Fractional Reasoning），这是一种无需训练且模型无关的框架，能够在推理时连续控制推理强度，超越固定指令提示的局限性。该方法通过提取与更深层次推理相关的潜在控制向量，并用可调节的缩放因子重新应用，使模型能够根据每个输入的复杂性调整其推理过程。这种分数推理支持两种关键的测试时扩展模式：（1）在广度基于策略（如“最佳的N个结果”、多数投票）中提升输出质量，（2）在深度基于策略（如自我反思）中增强单个推理链的正确性。实验表明，分数推理在多种推理任务和模型上均能一致地提升性能。 

---
# MoR: Better Handling Diverse Queries with a Mixture of Sparse, Dense, and Human Retrievers 

**Title (ZH)**: MoR: 使用稀疏、稠密和人工检索器混合处理多样查询 

**Authors**: Jushaan Singh Kalra, Xinran Zhao, To Eun Kim, Fengyu Cai, Fernando Diaz, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15862)  

**Abstract**: Retrieval-augmented Generation (RAG) is powerful, but its effectiveness hinges on which retrievers we use and how. Different retrievers offer distinct, often complementary signals: BM25 captures lexical matches; dense retrievers, semantic similarity. Yet in practice, we typically fix a single retriever based on heuristics, which fails to generalize across diverse information needs. Can we dynamically select and integrate multiple retrievers for each individual query, without the need for manual selection? In our work, we validate this intuition with quantitative analysis and introduce mixture of retrievers: a zero-shot, weighted combination of heterogeneous retrievers. Extensive experiments show that such mixtures are effective and efficient: Despite totaling just 0.8B parameters, this mixture outperforms every individual retriever and even larger 7B models by +10.8% and +3.9% on average, respectively. Further analysis also shows that this mixture framework can help incorporate specialized non-oracle human information sources as retrievers to achieve good collaboration, with a 58.9% relative performance improvement over simulated humans alone. 

**Abstract (ZH)**: 检索增强生成（RAG）很强大，但其效果依赖于使用的检索器及其应用方式。不同的检索器提供独特的、往往互补的信号：BM25捕获词汇匹配；密集检索器捕获语义相似性。然而，在实践中，我们通常基于启发式方法固定一个单一的检索器，这无法针对多样化的信息需求进行泛化。我们能否为每个单独的查询动态选择和整合多个检索器，而无需手动选择？在我们的工作中，我们通过定量分析验证了这一直觉，并引入了检索器混合：一种零样本、加权组合的异构检索器。广泛的实验显示，这种混合是有效且高效的：尽管总参数量仅为0.8B，但这种混合在个体检索器基础上分别取得了+10.8%和+3.9%的平均性能提升。进一步的分析还显示，这种混合框架可以有助于引入专门的非或acular人类信息源作为检索器进行协作，相较于单独的模拟人类，其相对性能提升了58.9%。 

---
# Cross-Modality Learning for Predicting IHC Biomarkers from H&E-Stained Whole-Slide Images 

**Title (ZH)**: 从H&E染色全切片图像中预测IHC生物标志物的跨模态学习 

**Authors**: Amit Das, Naofumi Tomita, Kyle J. Syme, Weijie Ma, Paige O'Connor, Kristin N. Corbett, Bing Ren, Xiaoying Liu, Saeed Hassanpour  

**Link**: [PDF](https://arxiv.org/pdf/2506.15853)  

**Abstract**: Hematoxylin and Eosin (H&E) staining is a cornerstone of pathological analysis, offering reliable visualization of cellular morphology and tissue architecture for cancer diagnosis, subtyping, and grading. Immunohistochemistry (IHC) staining provides molecular insights by detecting specific proteins within tissues, enhancing diagnostic accuracy, and improving treatment planning. However, IHC staining is costly, time-consuming, and resource-intensive, requiring specialized expertise. To address these limitations, this study proposes HistoStainAlign, a novel deep learning framework that predicts IHC staining patterns directly from H&E whole-slide images (WSIs) by learning joint representations of morphological and molecular features. The framework integrates paired H&E and IHC embeddings through a contrastive training strategy, capturing complementary features across staining modalities without patch-level annotations or tissue registration. The model was evaluated on gastrointestinal and lung tissue WSIs with three commonly used IHC stains: P53, PD-L1, and Ki-67. HistoStainAlign achieved weighted F1 scores of 0.735 [95% Confidence Interval (CI): 0.670-0.799], 0.830 [95% CI: 0.772-0.886], and 0.723 [95% CI: 0.607-0.836], respectively for these three IHC stains. Embedding analyses demonstrated the robustness of the contrastive alignment in capturing meaningful cross-stain relationships. Comparisons with a baseline model further highlight the advantage of incorporating contrastive learning for improved stain pattern prediction. This study demonstrates the potential of computational approaches to serve as a pre-screening tool, helping prioritize cases for IHC staining and improving workflow efficiency. 

**Abstract (ZH)**: HE和苏木精- eosin (H&E) 染色是病理分析的基石，为癌症诊断、亚型分类和分级提供可靠的细胞形态和组织结构可视化。免疫组织化学（IHC）染色通过检测组织内的特定蛋白质，提供分子洞察，增强诊断准确性并改善治疗规划。然而，IHC染色成本高、耗时且资源密集，需要专门的 expertise。为解决这些局限性，本研究提出了一种名为HistoStainAlign的新型深度学习框架，该框架可以直接从H&E全切片图像（WSI）中预测IHC染色模式，通过学习形态和分子特征的联合表示。该框架通过对比训练策略结合配对的H&E和IHC嵌入，捕获不同染色模式下的互补特征，无需斑块级注释或组织注册。该模型在胃肠道和肺组织的WSI上进行了评估，使用三种常用的IHC染色：P53、PD-L1和Ki-67。HistoStainAlign分别在这三种IHC染色中实现了加权F1分数为0.735 [95% 置信区间（CI）：0.670-0.799]、0.830 [95% CI：0.772-0.886] 和0.723 [95% CI：0.607-0.836]。嵌入分析表明对比对齐在捕获有意义的跨染色关系方面的稳健性。与基准模型的比较进一步突出了结合对比学习的优越性，以改善染色模式预测。本研究展示了计算方法作为预筛选工具的潜力，帮助优先处理需要IHC染色的病例并提高工作流程效率。 

---
# Uncertainty Estimation by Human Perception versus Neural Models 

**Title (ZH)**: 人类感知与神经模型的不确定性估计比较 

**Authors**: Pedro Mendes, Paolo Romano, David Garlan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15850)  

**Abstract**: Modern neural networks (NNs) often achieve high predictive accuracy but remain poorly calibrated, producing overconfident predictions even when wrong. This miscalibration poses serious challenges in applications where reliable uncertainty estimates are critical. In this work, we investigate how human perceptual uncertainty compares to uncertainty estimated by NNs. Using three vision benchmarks annotated with both human disagreement and crowdsourced confidence, we assess the correlation between model-predicted uncertainty and human-perceived uncertainty. Our results show that current methods only weakly align with human intuition, with correlations varying significantly across tasks and uncertainty metrics. Notably, we find that incorporating human-derived soft labels into the training process can improve calibration without compromising accuracy. These findings reveal a persistent gap between model and human uncertainty and highlight the potential of leveraging human insights to guide the development of more trustworthy AI systems. 

**Abstract (ZH)**: 现代神经网络的预测准确率往往很高，但往往缺乏校准，即使错误时也会产生过于自信的预测。这种校准不足给那些需要可靠不确定性估计的应用带来了严重挑战。在本工作中，我们探讨了人类感知的不确定性与神经网络估计的不确定性之间的差异。使用三个标注有人类分歧和众包自信度的视觉基准，我们评估了模型预测不确定性与人类感知不确定性之间的相关性。结果显示，当前方法仅弱弱地与人类直觉对齐，相关性在不同任务和不确定性度量下差异显著。值得注意的是，我们发现将人类衍生的软标签纳入训练过程可以在不牺牲准确性的前提下改善校准。这些发现揭示了模型和人类不确定性之间持续存在的差距，并强调了利用人类洞察来指导开发更可信赖的AI系统潜力的重要性。 

---
# SafeMimic: Towards Safe and Autonomous Human-to-Robot Imitation for Mobile Manipulation 

**Title (ZH)**: SafeMimic：迈向安全自主的人机模仿移动操作 

**Authors**: Arpit Bahety, Arnav Balaji, Ben Abbatematteo, Roberto Martín-Martín  

**Link**: [PDF](https://arxiv.org/pdf/2506.15847)  

**Abstract**: For robots to become efficient helpers in the home, they must learn to perform new mobile manipulation tasks simply by watching humans perform them. Learning from a single video demonstration from a human is challenging as the robot needs to first extract from the demo what needs to be done and how, translate the strategy from a third to a first-person perspective, and then adapt it to be successful with its own morphology. Furthermore, to mitigate the dependency on costly human monitoring, this learning process should be performed in a safe and autonomous manner. We present SafeMimic, a framework to learn new mobile manipulation skills safely and autonomously from a single third-person human video. Given an initial human video demonstration of a multi-step mobile manipulation task, SafeMimic first parses the video into segments, inferring both the semantic changes caused and the motions the human executed to achieve them and translating them to an egocentric reference. Then, it adapts the behavior to the robot's own morphology by sampling candidate actions around the human ones, and verifying them for safety before execution in a receding horizon fashion using an ensemble of safety Q-functions trained in simulation. When safe forward progression is not possible, SafeMimic backtracks to previous states and attempts a different sequence of actions, adapting both the trajectory and the grasping modes when required for its morphology. As a result, SafeMimic yields a strategy that succeeds in the demonstrated behavior and learns task-specific actions that reduce exploration in future attempts. Our experiments show that our method allows robots to safely and efficiently learn multi-step mobile manipulation behaviors from a single human demonstration, from different users, and in different environments, with improvements over state-of-the-art baselines across seven tasks 

**Abstract (ZH)**: 家用机器人通过观看人类演示学习新移动操作任务以成为高效的助手 

---
# Finance Language Model Evaluation (FLaME) 

**Title (ZH)**: 金融语言模型评估（FLaME） 

**Authors**: Glenn Matlin, Mika Okamoto, Huzaifa Pardawala, Yang Yang, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2506.15846)  

**Abstract**: Language Models (LMs) have demonstrated impressive capabilities with core Natural Language Processing (NLP) tasks. The effectiveness of LMs for highly specialized knowledge-intensive tasks in finance remains difficult to assess due to major gaps in the methodologies of existing evaluation frameworks, which have caused an erroneous belief in a far lower bound of LMs' performance on common Finance NLP (FinNLP) tasks. To demonstrate the potential of LMs for these FinNLP tasks, we present the first holistic benchmarking suite for Financial Language Model Evaluation (FLaME). We are the first research paper to comprehensively study LMs against 'reasoning-reinforced' LMs, with an empirical study of 23 foundation LMs over 20 core NLP tasks in finance. We open-source our framework software along with all data and results. 

**Abstract (ZH)**: 语言模型（LMs）在核心自然语言处理（NLP）任务中展示了令人印象深刻的性能。现有的评估框架方法学上的重大差距使得评估金融等专业知识密集型任务的语言模型效果困难，导致对其在常见金融NLP（FinNLP）任务中的表现存在错误的低限信念。为了展示语言模型在这些FinNLP任务上的潜力，我们介绍了首个金融语言模型评估集成套件（FLaME）。这是首篇全面研究语言模型与“推理强化”语言模型的学术论文，通过对23个基础语言模型在20项核心金融NLP任务上的实证研究来展示其潜力。我们开源了我们的框架软件及所有数据和结果。 

---
# MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents 

**Title (ZH)**: MEM1: 学习协同记忆与推理以实现高效的长期智能体 

**Authors**: Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15841)  

**Abstract**: Modern language agents must operate over long-horizon, multi-turn interactions, where they retrieve external information, adapt to observations, and answer interdependent queries. Yet, most LLM systems rely on full-context prompting, appending all past turns regardless of their relevance. This leads to unbounded memory growth, increased computational costs, and degraded reasoning performance on out-of-distribution input lengths. We introduce MEM1, an end-to-end reinforcement learning framework that enables agents to operate with constant memory across long multi-turn tasks. At each turn, MEM1 updates a compact shared internal state that jointly supports memory consolidation and reasoning. This state integrates prior memory with new observations from the environment while strategically discarding irrelevant or redundant information. To support training in more realistic and compositional settings, we propose a simple yet effective and scalable approach to constructing multi-turn environments by composing existing datasets into arbitrarily complex task sequences. Experiments across three domains, including internal retrieval QA, open-domain web QA, and multi-turn web shopping, show that MEM1-7B improves performance by 3.5x while reducing memory usage by 3.7x compared to Qwen2.5-14B-Instruct on a 16-objective multi-hop QA task, and generalizes beyond the training horizon. Our results demonstrate the promise of reasoning-driven memory consolidation as a scalable alternative to existing solutions for training long-horizon interactive agents, where both efficiency and performance are optimized. 

**Abstract (ZH)**: 现代语言代理必须在长期多轮交互中运行，期间它们检索外部信息、适应观察结果并回答相互依赖的问题。然而，大多数LLM系统依赖于全上下文提示，无论相关性如何都附加上所有过去轮次，这导致内存无界增长、增加计算成本并降低在分布外输入长度上的推理性能。我们引入了MEM1，这是一种端到端的强化学习框架，使代理能够在长期多轮任务中保持恒定内存运行。在每一轮中，MEM1更新一个紧凑的共享内部状态，该状态同时支持记忆巩固和推理。该状态将先前的记忆与环境的新观察结果整合在一起，同时策略性地丢弃无关或重复的信息。为了在更现实和组合的环境中支持训练，我们提出了一种简单而有效且可扩展的方法来构建多轮环境，通过组合现有数据集构建任意复杂程度的任务序列。在三个领域（包括内部检索问答、开放领域网络问答和多轮网络购物）的实验表明，与Qwen2.5-14B-Instruct相比，MEM1-7B在16目标多跳问答任务中的性能提高了3.5倍，同时内存使用量减少了3.7倍，并且能够超越训练范围进行泛化。我们的结果展示了基于推理的记忆巩固作为一种可扩展的替代方案的潜力，该方案用于训练长期交互代理，在效率和性能方面都进行了优化。 

---
# MoNetV2: Enhanced Motion Network for Freehand 3D Ultrasound Reconstruction 

**Title (ZH)**: MoNetV2: 提升的-motion 网络用于自由手绘制三维超声重建 

**Authors**: Mingyuan Luo, Xin Yang, Zhongnuo Yan, Yan Cao, Yuanji Zhang, Xindi Hu, Jin Wang, Haoxuan Ding, Wei Han, Litao Sun, Dong Ni  

**Link**: [PDF](https://arxiv.org/pdf/2506.15835)  

**Abstract**: Three-dimensional (3D) ultrasound (US) aims to provide sonographers with the spatial relationships of anatomical structures, playing a crucial role in clinical diagnosis. Recently, deep-learning-based freehand 3D US has made significant advancements. It reconstructs volumes by estimating transformations between images without external tracking. However, image-only reconstruction poses difficulties in reducing cumulative drift and further improving reconstruction accuracy, particularly in scenarios involving complex motion trajectories. In this context, we propose an enhanced motion network (MoNetV2) to enhance the accuracy and generalizability of reconstruction under diverse scanning velocities and tactics. First, we propose a sensor-based temporal and multi-branch structure that fuses image and motion information from a velocity perspective to improve image-only reconstruction accuracy. Second, we devise an online multi-level consistency constraint that exploits the inherent consistency of scans to handle various scanning velocities and tactics. This constraint exploits both scan-level velocity consistency, path-level appearance consistency, and patch-level motion consistency to supervise inter-frame transformation estimation. Third, we distill an online multi-modal self-supervised strategy that leverages the correlation between network estimation and motion information to further reduce cumulative errors. Extensive experiments clearly demonstrate that MoNetV2 surpasses existing methods in both reconstruction quality and generalizability performance across three large datasets. 

**Abstract (ZH)**: 基于运动网络V2的三维超声成像增强方法 

---
# Context Matters! Relaxing Goals with LLMs for Feasible 3D Scene Planning 

**Title (ZH)**: 背景因素很重要！借助LLM实现可行的3D场景规划 

**Authors**: Emanuele Musumeci, Michele Brienza, Francesco Argenziano, Vincenzo Suriani, Daniele Nardi, Domenico D. Bloisi  

**Link**: [PDF](https://arxiv.org/pdf/2506.15828)  

**Abstract**: Classical planning in AI and Robotics addresses complex tasks by shifting from imperative to declarative approaches (e.g., PDDL). However, these methods often fail in real scenarios due to limited robot perception and the need to ground perceptions to planning predicates. This often results in heavily hard-coded behaviors that struggle to adapt, even with scenarios where goals can be achieved through relaxed planning. Meanwhile, Large Language Models (LLMs) lead to planning systems that leverage commonsense reasoning but often at the cost of generating unfeasible and/or unsafe plans. To address these limitations, we present an approach integrating classical planning with LLMs, leveraging their ability to extract commonsense knowledge and ground actions. We propose a hierarchical formulation that enables robots to make unfeasible tasks tractable by defining functionally equivalent goals through gradual relaxation. This mechanism supports partial achievement of the intended objective, suited to the agent's specific context. Our method demonstrates its ability to adapt and execute tasks effectively within environments modeled using 3D Scene Graphs through comprehensive qualitative and quantitative evaluations. We also show how this method succeeds in complex scenarios where other benchmark methods are more likely to fail. Code, dataset, and additional material are released to the community. 

**Abstract (ZH)**: 经典AI与机器人规划通过从命令式转变为声明式方法（如PDDL）来应对复杂的任务。然而，这些方法在实际场景中往往由于机器人感知能力有限以及需要将感知与规划谓词关联而失败，这导致了高度硬编码的行为，即使在可以实现目标的松弛规划场景中也难以适应。同时，大规模语言模型（LLMs）虽然可以通过常识推理增强规划系统，但往往会产生不可行和/或不安全的计划。为解决这些限制，我们提出了一种结合经典规划与LLMs的方法，利用LLMs提取常识知识并限制动作的能力。我们提出了一种分层公式化方法，通过逐步放松定义功能等效的目标来使不可行的任务变得可处理。该机制支持在特定上下文中部分实现预期目标。我们的方法通过使用3D场景图建模的环境进行全面定性和定量评估，展示了其适应和有效执行任务的能力。此外，我们展示了该方法在基准方法更 likely 失败的复杂场景中取得成功。我们向社区发布了代码、数据集和额外的资料。 

---
# VEIGAR: View-consistent Explicit Inpainting and Geometry Alignment for 3D object Removal 

**Title (ZH)**: VEIGAR：视图一致的显式修复和几何对齐以去除3D物体 

**Authors**: Pham Khai Nguyen Do, Bao Nguyen Tran, Nam Nguyen, Duc Dung Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.15821)  

**Abstract**: Recent advances in Novel View Synthesis (NVS) and 3D generation have significantly improved editing tasks, with a primary emphasis on maintaining cross-view consistency throughout the generative process. Contemporary methods typically address this challenge using a dual-strategy framework: performing consistent 2D inpainting across all views guided by embedded priors either explicitly in pixel space or implicitly in latent space; and conducting 3D reconstruction with additional consistency guidance. Previous strategies, in particular, often require an initial 3D reconstruction phase to establish geometric structure, introducing considerable computational overhead. Even with the added cost, the resulting reconstruction quality often remains suboptimal. In this paper, we present VEIGAR, a computationally efficient framework that outperforms existing methods without relying on an initial reconstruction phase. VEIGAR leverages a lightweight foundation model to reliably align priors explicitly in the pixel space. In addition, we introduce a novel supervision strategy based on scale-invariant depth loss, which removes the need for traditional scale-and-shift operations in monocular depth regularization. Through extensive experimentation, VEIGAR establishes a new state-of-the-art benchmark in reconstruction quality and cross-view consistency, while achieving a threefold reduction in training time compared to the fastest existing method, highlighting its superior balance of efficiency and effectiveness. 

**Abstract (ZH)**: Recent Advances in Novel View Synthesis (NVS) and 3D Generation Have Significantly Improved Editing Tasks While Maintaining Cross-View Consistency Without Relying on an Initial Reconstruction Phase 

---
# Unsupervised deep learning model for fast energy layer pre-selection of delivery-efficient proton arc therapy plan optimization of nasopharyngeal carcinoma 

**Title (ZH)**: 无监督深度学习模型用于鼻咽癌递送高效质子弧治疗计划的快速能量层预选 

**Authors**: Bohan Yang, Gang Liu, Rirao Dao, Yujia Qian, Ke Shi, Anke Tang, Yong Luo, Jingnan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15803)  

**Abstract**: Objective. Proton arc therapy (PAT) is an emerging and promising modality in radiotherapy, offering several advantages over conventional intensitymodulated proton therapy (IMPT). However, identifying the optimal energy layer (EL) sequence remains computationally intensive due to the large number of possible energy layer transitions. This study proposes an unsupervised deep learning framework for fast and effective EL pre-selection, aiming to minimize energy layer switch time while preserving high plan quality. Approach. We introduce a novel data representation method, spot-count representation, which encodes the number of proton spots intersecting the target and organs at risk (OARs) in a matrix structured by sorted gantry angles and energy layers. This representation is the input of a UNet-based architecture, SPArcdl, which is trained to optimize a tri-objective function: maximizing target coverage, minimizing OAR exposure, and reducing energy switching time. The model is evaluated on 54 nasopharyngeal cancer cases, and its performance is benchmarked against plans generated by SPArcparticle swarm. Main results. SPArcdl produces EL pre-selection that significantly improves both plan quality and delivery efficiency. Compared to SPArc particle swarm, it enhances the conformity index by 0.16 (p < 0.01), reduces the homogeneity index by 0.71 (p < 0.01), shortens the energy switching time by 38.4% (p < 0.01), and lowers the mean dose to brainstem by 0.21 (p < 0.01). The results unintentionally reveal employing unchanged ELS is more time-wise efficient than descended ELS. SPArcdl's inference time is within 1 second. Significance. SPArcdl is a fast and effective tool for generating high-quality PAT plans by strategically pre-selecting energy layers to reduce delivery time while maintaining excellent dosimetric performance. 

**Abstract (ZH)**: 目标. 质子弧疗法(PAT)是一种新兴且有前景的放疗技术，相较于常规调强质子疗法(IMPT)具有多项优势。然而，确定最优能量层(EL)序列仍因可能的能量层转换数量庞大而计算密集。本研究提出了一种无监督深度学习框架，用于快速有效地进行EL预选，旨在最小化能量层切换时间的同时保持高计划质量。方法. 我们引入了一种新的数据表示方法——点计数表示法，该方法编码了穿过靶区和危险器官(OARs)的质子点的数量，并以排序的扫描角度和能量层结构化矩阵形式表示。该表示法作为基于UNet架构的SPArcdl模型的输入，该模型被训练以优化一个三目标函数：最大化靶区覆盖、最小化OAR暴露和减少能量切换时间。该模型在54例鼻咽癌病例上进行了评估，并与SPArc粒子群优化生成的计划进行了基准测试。主要结果. SPArcdl产生的EL预选显著提高了计划质量和递送效率。与SPArc粒子群优化相比，它提高了一致性指数0.16(p < 0.01)、减少了均匀性指数0.71(p < 0.01)、缩短了能量切换时间38.4%(p < 0.01)、降低了脑干的平均剂量0.21 Gy(p < 0.01)。结果无意间揭示了使用不变的能量层比下降的能量层更为时间高效。SPArcdl的推理时间在1秒之内。意义. SPArcdl是一种快速有效的工具，通过战略性地预选能量层来生成高质量的PAT计划，从而减少递送时间并保持卓越的剂量学表现。 

---
# Veracity: An Open-Source AI Fact-Checking System 

**Title (ZH)**: 真实性：一个开源AI事实核查系统 

**Authors**: Taylor Lynn Curtis, Maximilian Puelma Touzel, William Garneau, Manon Gruaz, Mike Pinder, Li Wei Wang, Sukanya Krishna, Luda Cohen, Jean-François Godbout, Reihaneh Rabbany, Kellin Pelrine  

**Link**: [PDF](https://arxiv.org/pdf/2506.15794)  

**Abstract**: The proliferation of misinformation poses a significant threat to society, exacerbated by the capabilities of generative AI. This demo paper introduces Veracity, an open-source AI system designed to empower individuals to combat misinformation through transparent and accessible fact-checking. Veracity leverages the synergy between Large Language Models (LLMs) and web retrieval agents to analyze user-submitted claims and provide grounded veracity assessments with intuitive explanations. Key features include multilingual support, numerical scoring of claim veracity, and an interactive interface inspired by familiar messaging applications. This paper will showcase Veracity's ability to not only detect misinformation but also explain its reasoning, fostering media literacy and promoting a more informed society. 

**Abstract (ZH)**: misinformation的泛滥对社会构成了显著威胁，生成式AI的能力进一步加剧了这一问题。本文介绍了Veracity，这是一个开源AI系统，旨在通过透明和易访问的事实核查来赋能个人对抗 misinformation。Veracity 利用大型语言模型（LLMs）与网页检索代理之间的协同作用，分析用户提交的断言，并提供基于直观解释的可靠性的评估。关键功能包括多语言支持、断言可靠性的数值评分以及借鉴熟悉的消息应用程序界面的互动界面。本文将展示Veracity不仅能够检测 misinformation，还能解释其推理过程，从而促进媒体素养并推动更加知情的社会。 

---
# Linearithmic Clean-up for Vector-Symbolic Key-Value Memory with Kroneker Rotation Products 

**Title (ZH)**: 基于克罗内克旋转积的线性对数级清理算法用于向量符号键值记忆系统 

**Authors**: Ruipeng Liu, Qinru Qiu, Simon Khan, Garrett E. Katz  

**Link**: [PDF](https://arxiv.org/pdf/2506.15793)  

**Abstract**: A computational bottleneck in current Vector-Symbolic Architectures (VSAs) is the ``clean-up'' step, which decodes the noisy vectors retrieved from the architecture. Clean-up typically compares noisy vectors against a ``codebook'' of prototype vectors, incurring computational complexity that is quadratic or similar. We present a new codebook representation that supports efficient clean-up, based on Kroneker products of rotation-like matrices. The resulting clean-up time complexity is linearithmic, i.e. $\mathcal{O}(N\,\text{log}\,N)$, where $N$ is the vector dimension and also the number of vectors in the codebook. Clean-up space complexity is $\mathcal{O}(N)$. Furthermore, the codebook is not stored explicitly in computer memory: It can be represented in $\mathcal{O}(\text{log}\,N)$ space, and individual vectors in the codebook can be materialized in $\mathcal{O}(N)$ time and space. At the same time, asymptotic memory capacity remains comparable to standard approaches. Computer experiments confirm these results, demonstrating several orders of magnitude more scalability than baseline VSA techniques. 

**Abstract (ZH)**: 当前向量-符号架构中的一个计算瓶颈是在从架构中检索到的嘈杂向量进行“清洁”步骤，这涉及到将这些嘈杂向量与原型向量的“代码本”进行比较，从而产生二次或类似的计算复杂度。我们提出了一种新的代码本表示方法，它基于旋转矩阵的克罗内克积，支持高效的清洁步骤。清洁步骤的时间复杂度为对数线性，即$\mathcal{O}(N \,\text{log}\, N)$，其中$N$是向量维数，也是代码本中向量的数量。清洁步骤的空间复杂度为$\mathcal{O}(N)$。此外，该代码本并未显式存储在计算机内存中：它可以使用$\mathcal{O}(\text{log}\,N)$的空间表示，并且可以在$\mathcal{O}(N)$的时间和空间内实现代码本中的个别向量。与此同时，渐近内存容量与标准方法相当。计算机实验验证了这些结果，显示出比基线向量-符号架构技术高出几个数量级的可扩展性。 

---
# TRUST: Transparent, Robust and Ultra-Sparse Trees 

**Title (ZH)**: TRUST: 透明、鲁棒且超稀疏树结构 

**Authors**: Albert Dorador  

**Link**: [PDF](https://arxiv.org/pdf/2506.15791)  

**Abstract**: Piecewise-constant regression trees remain popular for their interpretability, yet often lag behind black-box models like Random Forest in predictive accuracy. In this work, we introduce TRUST (Transparent, Robust, and Ultra-Sparse Trees), a novel regression tree model that combines the accuracy of Random Forests with the interpretability of shallow decision trees and sparse linear models. TRUST further enhances transparency by leveraging Large Language Models to generate tailored, user-friendly explanations. Extensive validation on synthetic and real-world benchmark datasets demonstrates that TRUST consistently outperforms other interpretable models -- including CART, Lasso, and Node Harvest -- in predictive accuracy, while matching the accuracy of Random Forest and offering substantial gains in both accuracy and interpretability over M5', a well-established model that is conceptually related. 

**Abstract (ZH)**: 透明、稳健且极度稀疏的树模型TRUST：结合随机森林的预测准确性和浅决策树及稀疏线性模型的可解释性 

---
# Graphics4Science: Computer Graphics for Scientific Impacts 

**Title (ZH)**: Graphics4Science: 计算机图形学的科学影响 

**Authors**: Peter Yichen Chen, Minghao Guo, Hanspeter Pfister, Ming Lin, William Freeman, Qixing Huang, Han-Wei Shen, Wojciech Matusik  

**Link**: [PDF](https://arxiv.org/pdf/2506.15786)  

**Abstract**: Computer graphics, often associated with films, games, and visual effects, has long been a powerful tool for addressing scientific challenges--from its origins in 3D visualization for medical imaging to its role in modern computational modeling and simulation. This course explores the deep and evolving relationship between computer graphics and science, highlighting past achievements, ongoing contributions, and open questions that remain. We show how core methods, such as geometric reasoning and physical modeling, provide inductive biases that help address challenges in both fields, especially in data-scarce settings. To that end, we aim to reframe graphics as a modeling language for science by bridging vocabulary gaps between the two communities. Designed for both newcomers and experts, Graphics4Science invites the graphics community to engage with science, tackle high-impact problems where graphics expertise can make a difference, and contribute to the future of scientific discovery. Additional details are available on the course website: this https URL 

**Abstract (ZH)**: 计算机图形学，常与电影、游戏和视觉效果相关，长期以来一直是应对科学挑战的强大工具——从医学成像领域的3D可视化起源，到现代计算建模和模拟中的角色。本课程探讨了计算机图形学与科学之间深厚且不断发展的关系，突显了过去的成就、现有的贡献以及仍然存在的开放问题。我们展示了核心方法，如几何推理和物理建模，如何提供归纳偏置，帮助解决两个领域的挑战，尤其是在数据稀缺的情况下。为此，我们旨在通过弥合两个社区之间的词汇差异，将图形学重新构想为一种科学建模语言。面向新手和专家，Graphics4Science 邀请图形学社区参与科学领域，解决那些图形学专长可以产生重大影响的问题，并为科学发现的未来作出贡献。更多细节请参见课程网站：this https URL 

---
# RecBayes: Recurrent Bayesian Ad Hoc Teamwork in Large Partially Observable Domains 

**Title (ZH)**: RecBayes: 递归贝叶斯即兴团队协作在大型部分可观测域中 

**Authors**: João G. Ribeiro, Yaniv Oren, Alberto Sardinha, Matthijs Spaan, Francisco S. Melo  

**Link**: [PDF](https://arxiv.org/pdf/2506.15756)  

**Abstract**: This paper proposes RecBayes, a novel approach for ad hoc teamwork under partial observability, a setting where agents are deployed on-the-fly to environments where pre-existing teams operate, that never requires, at any stage, access to the states of the environment or the actions of its teammates. We show that by relying on a recurrent Bayesian classifier trained using past experiences, an ad hoc agent is effectively able to identify known teams and tasks being performed from observations alone. Unlike recent approaches such as PO-GPL (Gu et al., 2021) and FEAT (Rahman et al., 2023), that require at some stage fully observable states of the environment, actions of teammates, or both, or approaches such as ATPO (Ribeiro et al., 2023) that require the environments to be small enough to be tabularly modelled (Ribeiro et al., 2023), in their work up to 4.8K states and 1.7K observations, we show RecBayes is both able to handle arbitrarily large spaces while never relying on either states and teammates' actions. Our results in benchmark domains from the multi-agent systems literature, adapted for partial observability and scaled up to 1M states and 2^125 observations, show that RecBayes is effective at identifying known teams and tasks being performed from partial observations alone, and as a result, is able to assist the teams in solving the tasks effectively. 

**Abstract (ZH)**: RecBayes：在部分可观测性环境下的一种新型即兴团队工作方法 

---
# A Study of Hybrid and Evolutionary Metaheuristics for Single Hidden Layer Feedforward Neural Network Architecture 

**Title (ZH)**: 单隐层前向神经网络架构的混合与进化元启发式研究 

**Authors**: Gautam Siddharth Kashyap, Md Tabrez Nafis, Samar Wazir  

**Link**: [PDF](https://arxiv.org/pdf/2506.15737)  

**Abstract**: Training Artificial Neural Networks (ANNs) with Stochastic Gradient Descent (SGD) frequently encounters difficulties, including substantial computing expense and the risk of converging to local optima, attributable to its dependence on partial weight gradients. Therefore, this work investigates Particle Swarm Optimization (PSO) and Genetic Algorithms (GAs) - two population-based Metaheuristic Optimizers (MHOs) - as alternatives to SGD to mitigate these constraints. A hybrid PSO-SGD strategy is developed to improve local search efficiency. The findings indicate that the hybrid PSO-SGD technique decreases the median training MSE by 90 to 95 percent relative to conventional GA and PSO across various network sizes (e.g., from around 0.02 to approximately 0.001 in the Sphere function). RMHC attains substantial enhancements, reducing MSE by roughly 85 to 90 percent compared to GA. Simultaneously, RS consistently exhibits errors exceeding 0.3, signifying subpar performance. These findings underscore that hybrid and evolutionary procedures significantly improve training efficiency and accuracy compared to conventional optimization methods and imply that the Building Block Hypothesis (BBH) may still be valid, indicating that advantageous weight structures are retained during evolutionary search. 

**Abstract (ZH)**: 使用粒子群优化（PSO）和遗传算法（GAs）替代随机梯度下降（SGD）以缓解人工神经网络（ANNs）的训练难题：基于混合PSO-SGD策略的改进局部搜索效率 

---
# Graph Diffusion that can Insert and Delete 

**Title (ZH)**: 具有插入和删除功能的图扩散 

**Authors**: Matteo Ninniri, Marco Podda, Davide Bacciu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15725)  

**Abstract**: Generative models of graphs based on discrete Denoising Diffusion Probabilistic Models (DDPMs) offer a principled approach to molecular generation by systematically removing structural noise through iterative atom and bond adjustments. However, existing formulations are fundamentally limited by their inability to adapt the graph size (that is, the number of atoms) during the diffusion process, severely restricting their effectiveness in conditional generation scenarios such as property-driven molecular design, where the targeted property often correlates with the molecular size. In this paper, we reformulate the noising and denoising processes to support monotonic insertion and deletion of nodes. The resulting model, which we call GrIDDD, dynamically grows or shrinks the chemical graph during generation. GrIDDD matches or exceeds the performance of existing graph diffusion models on molecular property targeting despite being trained on a more difficult problem. Furthermore, when applied to molecular optimization, GrIDDD exhibits competitive performance compared to specialized optimization models. This work paves the way for size-adaptive molecular generation with graph diffusion. 

**Abstract (ZH)**: 基于离散去噪扩散概率模型（DDPMs）的图生成模型通过迭代的原子和化学键调整系统地去除结构噪声，提供了一种分子生成的原理性方法。然而，现有的模型在扩散过程中固有限制了其对分子大小的适应性，严重限制了其在如性质驱动的分子设计等条件生成场景中的有效性，这些场景中目标性质往往与分子大小相关。本文重新定义了去噪和扰噪过程，支持节点的单调插入和删除，提出了一种称为GrIDDD的模型，该模型在生成过程中动态地增长或缩小化学图。尽管是针对一个更困难的问题进行训练，GrIDDD在分子性质目标方面的性能与现有的图扩散模型相当或超越。此外，在分子优化中，GrIDDD也表现出与专门优化模型相竞争的性能。这项工作为进一步实现图扩散下的自适应分子生成奠定了基础。 

---
# MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference 

**Title (ZH)**: MadaKV：自适应模态-感知KV缓存淘汰机制，实现高效的多模态长上下文推理 

**Authors**: Kunxi Li, Zhonghua Jiang, Zhouzhou Shen, Zhaode Wang, Chengfei Lv, Shengyu Zhang, Fan Wu, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15724)  

**Abstract**: This paper introduces MadaKV, a modality-adaptive key-value (KV) cache eviction strategy designed to enhance the efficiency of multimodal large language models (MLLMs) in long-context inference. In multimodal scenarios, attention heads exhibit varying preferences for different modalities, resulting in significant disparities in modality importance across attention heads. Traditional KV cache eviction methods, which are tailored for unimodal settings, fail to capture modality-specific information, thereby yielding suboptimal performance. MadaKV addresses these challenges through two key components: modality preference adaptation and hierarchical compression compensation. By dynamically sensing modality information within attention heads and adaptively retaining critical tokens, MadaKV achieves substantial reductions in KV cache memory footprint and model inference decoding latency (1.3 to 1.5 times improvement) while maintaining high accuracy across various multimodal long-context tasks. Extensive experiments on representative MLLMs and the MileBench benchmark demonstrate the effectiveness of MadaKV compared to existing KV cache eviction methods. 

**Abstract (ZH)**: MadaKV：一种针对多模态大型语言模型长文境推理的模态自适应键值缓存淘汰策略 

---
# UniMate: A Unified Model for Mechanical Metamaterial Generation, Property Prediction, and Condition Confirmation 

**Title (ZH)**: UniMate：统一的机械元材料生成、性质预测和条件验证模型 

**Authors**: Wangzhi Zhan, Jianpeng Chen, Dongqi Fu, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.15722)  

**Abstract**: Metamaterials are artificial materials that are designed to meet unseen properties in nature, such as ultra-stiffness and negative materials indices. In mechanical metamaterial design, three key modalities are typically involved, i.e., 3D topology, density condition, and mechanical property. Real-world complex application scenarios place the demanding requirements on machine learning models to consider all three modalities together. However, a comprehensive literature review indicates that most existing works only consider two modalities, e.g., predicting mechanical properties given the 3D topology or generating 3D topology given the required properties. Therefore, there is still a significant gap for the state-of-the-art machine learning models capturing the whole. Hence, we propose a unified model named UNIMATE, which consists of a modality alignment module and a synergetic diffusion generation module. Experiments indicate that UNIMATE outperforms the other baseline models in topology generation task, property prediction task, and condition confirmation task by up to 80.2%, 5.1%, and 50.2%, respectively. We opensource our proposed UNIMATE model and corresponding results at this https URL. 

**Abstract (ZH)**: 人工材料是设计用于实现自然界中未见属性的人工材料，如超刚度和负材料指数。在机械人工材料设计中，通常涉及三种关键模态，即三维拓扑结构、密度条件和机械性能。现实世界复杂的应用场景对机器学习模型提出了同时综合考虑这三个模态的要求。然而，文献综述表明，大多数现有工作仅考虑了两个模态，例如给定三维拓扑结构预测机械性能或给定所需属性生成三维拓扑结构。因此，最先进的机器学习模型仍然在综合捕捉这三个模态方面存在显著差距。因此，我们提出了一种名为UNIMATE的统一模型，该模型包括一种模态对齐模块和一种协同扩散生成模块。实验表明，在拓扑结构生成任务、性能预测任务和条件确认任务中，UNIMATE分别比其他基线模型高出80.2%、5.1%和50.2%。我们开源了所提出的UNIMATE模型及其相应结果。 

---
# daDPO: Distribution-Aware DPO for Distilling Conversational Abilities 

**Title (ZH)**: daDPO: 分布感知的DPO对话能力压缩方法 

**Authors**: Zhengze Zhang, Shiqi Wang, Yiqun Shen, Simin Guo, Dahua Lin, Xiaoliang Wang, Nguyen Cam-Tu, Fei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15717)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional performance across various applications, but their conversational abilities decline sharply as model size decreases, presenting a barrier to their deployment in resource-constrained environments. Knowledge distillation with Direct Preference Optimization (dDPO) has emerged as a promising approach to enhancing the conversational abilities of smaller models using a larger teacher model. However, current methods primarily focus on 'black-box' KD, which only uses the teacher's responses, overlooking the output distribution offered by the teacher. This paper addresses this gap by introducing daDPO (Distribution-Aware DPO), a unified method for preference optimization and distribution-based distillation. We provide rigorous theoretical analysis and empirical validation, showing that daDPO outperforms existing methods in restoring performance for pruned models and enhancing smaller LLM models. Notably, in in-domain evaluation, our method enables a 20% pruned Vicuna1.5-7B to achieve near-teacher performance (-7.3% preference rate compared to that of dDPO's -31%), and allows Qwen2.5-1.5B to occasionally outperform its 7B teacher model (14.0% win rate). 

**Abstract (ZH)**: 大规模语言模型(LLMs)在各种应用中展现了卓越的性能，但随着模型规模的减小，其对话能力急剧下降，这成为其在资源受限环境中部署的障碍。直接偏好优化(Direct Preference Optimization, dDPO)指导的知识蒸馏方法为通过较大教师模型增强较小模型的对话能力提供了有前景的途径。然而，现有的方法主要关注“黑盒”知识蒸馏，只利用教师的响应，而忽略了教师提供的输出分布。本文通过引入daDPO（分布感知dDPO）统一方法来解决这一问题，该方法将偏好优化和基于分布的蒸馏结合起来。我们进行了严格的理论分析和实证验证，表明daDPO在恢复剪枝模型性能和增强较小的LLM模型方面优于现有方法。特别是在领域内评价中，我们的方法使剪枝比例为20%的Vicuna1.5-7B达到接近教师模型的性能（偏好率为-7.3%，而dDPO为-31%），并使Qwen2.5-1.5B偶尔超越其7B教师模型（胜率为14.0%）。 

---
# Alternates, Assemble! Selecting Optimal Alternates for Citizens' Assemblies 

**Title (ZH)**: 交替登场！公民大会的最佳替代者选择 

**Authors**: Angelos Assos, Carmel Baharav, Bailey Flanigan, Ariel Procaccia  

**Link**: [PDF](https://arxiv.org/pdf/2506.15716)  

**Abstract**: An increasingly influential form of deliberative democracy centers on citizens' assemblies, where randomly selected people discuss policy questions. The legitimacy of these panels hinges on their representation of the broader population, but panelists often drop out, leading to an unbalanced composition. Although participant attrition is mitigated in practice by alternates, their selection is not taken into account by existing methods. To address this gap, we introduce an optimization framework for alternate selection. Our algorithmic approach, which leverages learning-theoretic machinery, estimates dropout probabilities using historical data and selects alternates to minimize expected misrepresentation. We establish theoretical guarantees for our approach, including worst-case bounds on sample complexity (with implications for computational efficiency) and on loss when panelists' probabilities of dropping out are mis-estimated. Empirical evaluation using real-world data demonstrates that, compared to the status quo, our method significantly improves representation while requiring fewer alternates. 

**Abstract (ZH)**: 一种日益有影响力的 deliberative democracy 形式侧重于公民 assembly，其中随机选定的人员讨论政策问题。这些小组的合法性取决于其对更广泛人口的代表性，但参与者常会退出，导致小组组成失衡。尽管在实践中通过备选人员可以减轻参与者流失的问题，但现有的方法并未将备选人员的选择考虑在内。为解决这一缺口，我们提出了一种备选人员选择的优化框架。我们的算法方法利用了学习理论的工具，利用历史数据估计退出概率，并选择备选人员以最小化预期的代表性失真。我们为这种方法建立了理论保证，包括最坏情况下的样本复杂性界（对计算效率的影响）以及参与者退出概率估计有误时的损失界。使用真实数据的实证评估表明，与现状相比，我们的方法在需要更少备选人员的情况下显著提高了代表性。 

---
# NeuronSeek: On Stability and Expressivity of Task-driven Neurons 

**Title (ZH)**: NeuronSeek：任务驱动神经元的稳定性和表征能力探究 

**Authors**: Hanyu Pei, Jing-Xiao Liao, Qibin Zhao, Ting Gao, Shijun Zhang, Xiaoge Zhang, Feng-Lei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15715)  

**Abstract**: Drawing inspiration from our human brain that designs different neurons for different tasks, recent advances in deep learning have explored modifying a network's neurons to develop so-called task-driven neurons. Prototyping task-driven neurons (referred to as NeuronSeek) employs symbolic regression (SR) to discover the optimal neuron formulation and construct a network from these optimized neurons. Along this direction, this work replaces symbolic regression with tensor decomposition (TD) to discover optimal neuronal formulations, offering enhanced stability and faster convergence. Furthermore, we establish theoretical guarantees that modifying the aggregation functions with common activation functions can empower a network with a fixed number of parameters to approximate any continuous function with an arbitrarily small error, providing a rigorous mathematical foundation for the NeuronSeek framework. Extensive empirical evaluations demonstrate that our NeuronSeek-TD framework not only achieves superior stability, but also is competitive relative to the state-of-the-art models across diverse benchmarks. The code is available at this https URL. 

**Abstract (ZH)**: 从人类大脑设计不同神经元用于不同任务中汲取灵感， recent advances in deep learning探索了修改网络神经元以开发所谓的任务驱动神经元。Prototype任务驱动神经元（即NeuronSeek）使用符号回归（SR）来发现最优神经元公式并构建由这些优化神经元组成的网络。在此方向上，本文用张量分解（TD）替换符号回归以发现最优神经元公式，提供增强的稳定性和更快的收敛性。此外，我们建立了理论保证，修改常用的激活函数可以使具有固定参数数量的网络逼近任意连续函数，随任意小的误差，为NeuronSeek框架提供了坚实的数学基础。广泛的实证评估表明，我们的NeuronSeek-TD框架不仅实现了卓越的稳定性，还在多种基准上与最先进的模型竞争。代码可在以下链接获取。 

---
# BatteryBERT for Realistic Battery Fault Detection Using Point-Masked Signal Modeling 

**Title (ZH)**: 基于点掩蔽信号建模的BatteryBERT：用于真实电池故障检测的模型 

**Authors**: Songqi Zhou, Ruixue Liu, Yixing Wang, Jia Lu, Benben Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15712)  

**Abstract**: Accurate fault detection in lithium-ion batteries is essential for the safe and reliable operation of electric vehicles and energy storage systems. However, existing methods often struggle to capture complex temporal dependencies and cannot fully leverage abundant unlabeled data. Although large language models (LLMs) exhibit strong representation capabilities, their architectures are not directly suited to the numerical time-series data common in industrial settings. To address these challenges, we propose a novel framework that adapts BERT-style pretraining for battery fault detection by extending the standard BERT architecture with a customized time-series-to-token representation module and a point-level Masked Signal Modeling (point-MSM) pretraining task tailored to battery applications. This approach enables self-supervised learning on sequential current, voltage, and other charge-discharge cycle data, yielding distributionally robust, context-aware temporal embeddings. We then concatenate these embeddings with battery metadata and feed them into a downstream classifier for accurate fault classification. Experimental results on a large-scale real-world dataset show that models initialized with our pretrained parameters significantly improve both representation quality and classification accuracy, achieving an AUROC of 0.945 and substantially outperforming existing approaches. These findings validate the effectiveness of BERT-style pretraining for time-series fault detection. 

**Abstract (ZH)**: 锂离子电池准确故障检测对于电动汽车和储能系统的安全可靠运行至关重要。然而，现有的方法往往难以捕捉复杂的时序依赖关系，无法充分利用丰富的未标注数据。尽管大规模语言模型（LLMs）表现出强大的表示能力，但其架构并不直接适合工业场景中常见的数值时间序列数据。为了解决这些挑战，我们提出了一种新颖的框架，通过将BERT风格的预训练适应于电池故障检测，并扩展标准的BERT架构，加入一个定制的时间序列到标记表示模块以及一个针对电池应用量身定制的点级掩码信号建模（point-MSM）预训练任务。该方法使模型能对连续的电流、电压和其他充放电循环数据进行半监督学习，生成分布鲁棒、上下文感知的时序嵌入。然后，将这些嵌入与电池元数据拼接，并输入下游分类器进行准确的故障分类。在大规模真实世界数据集上的实验结果表明，使用我们预训练参数初始化的模型显著提高了表示质量和分类准确性，AUROC达到0.945，明显优于现有方法。这些发现验证了BERT风格预训练在时间序列故障检测中的有效性。 

---
# Shadow defense against gradient inversion attack in federated learning 

**Title (ZH)**: 联邦学习中抵御梯度反转攻击的阴影防御 

**Authors**: Le Jiang, Liyan Ma, Guang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15711)  

**Abstract**: Federated learning (FL) has emerged as a transformative framework for privacy-preserving distributed training, allowing clients to collaboratively train a global model without sharing their local data. This is especially crucial in sensitive fields like healthcare, where protecting patient data is paramount. However, privacy leakage remains a critical challenge, as the communication of model updates can be exploited by potential adversaries. Gradient inversion attacks (GIAs), for instance, allow adversaries to approximate the gradients used for training and reconstruct training images, thus stealing patient privacy. Existing defense mechanisms obscure gradients, yet lack a nuanced understanding of which gradients or types of image information are most vulnerable to such attacks. These indiscriminate calibrated perturbations result in either excessive privacy protection degrading model accuracy, or insufficient one failing to safeguard sensitive information. Therefore, we introduce a framework that addresses these challenges by leveraging a shadow model with interpretability for identifying sensitive areas. This enables a more targeted and sample-specific noise injection. Specially, our defensive strategy achieves discrepancies of 3.73 in PSNR and 0.2 in SSIM compared to the circumstance without defense on the ChestXRay dataset, and 2.78 in PSNR and 0.166 in the EyePACS dataset. Moreover, it minimizes adverse effects on model performance, with less than 1\% F1 reduction compared to SOTA methods. Our extensive experiments, conducted across diverse types of medical images, validate the generalization of the proposed framework. The stable defense improvements for FedAvg are consistently over 1.5\% times in LPIPS and SSIM. It also offers a universal defense against various GIA types, especially for these sensitive areas in images. 

**Abstract (ZH)**: 联邦学习（FL）作为一种保护隐私的分布式训练范式，允许多个客户端协作训练全局模型而无需共享本地数据。这在像医疗健康这样敏感的领域尤为重要，因为保护患者数据至关重要。然而，隐私泄露仍然是一个关键挑战，因为模型更新的通信可能会被潜在对手利用。梯度反向攻击（GIAs）允许对手逼近用于训练的梯度并重建训练图像，从而窃取患者隐私。现有的防御机制掩盖了梯度，但缺乏对哪些梯度或哪种图像信息最容易受到此类攻击的理解。这种不分青红皂白的校准扰动要么过度保护隐私导致模型准确性下降，要么保护不足无法保护敏感信息。因此，我们提出了一种框架，通过利用具有可解释性的阴影模型来识别敏感区域，以实现更具针对性和样本特异性的噪声注入。特别地，我们的防御策略在ChestXRay数据集中与无防御情况相比，PSNR上的差异为3.73，SSIM上的差异为0.2；在EyePACS数据集中，PSNR上的差异为2.78，SSIM上的差异为0.166。此外，它还可以最小化对模型性能的负面影响，与最先进的方法相比，F1分数的降低不到1%。我们在多种类型医疗图像上进行的广泛实验验证了所提出框架的泛化能力。对于FedAvg的稳定防御改进，稳定感知相异性（LPIPS）和SSIM均超过1.5%。该框架还对各种GIAs类型提供了普遍防御，尤其对图像中的敏感区域特别有效。 

---
# RAST: Reasoning Activation in LLMs via Small-model Transfer 

**Title (ZH)**: RAST：通过小型模型迁移实现LLMs的推理激活 

**Authors**: Siru Ouyang, Xinyu Zhu, Zilin Xiao, Minhao Jiang, Yu Meng, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.15710)  

**Abstract**: Reinforcement learning (RL) has become a powerful approach for improving the reasoning capabilities of large language models (LLMs), as evidenced by recent successes such as OpenAI's o1 and Deepseek-R1. However, applying RL at scale remains intimidatingly resource-intensive, requiring multiple model copies and extensive GPU workloads. On the other hand, while being powerful, recent studies suggest that RL does not fundamentally endow models with new knowledge; rather, it primarily reshapes the model's output distribution to activate reasoning capabilities latent in the base model. Building on this insight, we hypothesize that the changes in output probabilities induced by RL are largely model-size invariant, opening the door to a more efficient paradigm: training a small model with RL and transferring its induced probability shifts to larger base models. To verify our hypothesis, we conduct a token-level analysis of decoding trajectories and find high alignment in RL-induced output distributions across model scales, validating our hypothesis. Motivated by this, we propose RAST, a simple yet effective method that transfers reasoning behaviors by injecting RL-induced probability adjustments from a small RL-trained model into larger models. Experiments across multiple mathematical reasoning benchmarks show that RAST substantially and consistently enhances the reasoning capabilities of base models while requiring significantly lower GPU memory than direct RL training, sometimes even yielding better performance than the RL-trained counterparts. Our findings offer new insights into the nature of RL-driven reasoning and practical strategies for scaling its benefits without incurring its full computational cost. The project page of RAST is available at this https URL. 

**Abstract (ZH)**: 强化学习（RL）已成为提升大型语言模型（LLMs）推理能力的强大方法，如OpenAI的o1和Deepseek-R1的成功所示。然而，大规模应用RL仍然是资源密集型的，需要多个模型副本和大量的GPU工作负载。尽管如此，近期研究表明，RL本质上并未赋予模型新的知识；相反，它主要通过重塑模型的输出分布来激活基模型中存在的推理能力。基于这一见解，我们假设由RL引起的输出概率变化在不同规模的模型中是大模型尺寸不变的，从而开启了更有效的范式：用小型模型进行RL训练，并将由此引发的概率调整传递给更大的基模型。为了验证这一假设，我们进行了逐令牌解码轨迹分析，发现不同规模模型中RL引起的输出分布具有高度一致性，验证了我们的假设。受此驱动，我们提出了一种简单而有效的方法RAST，该方法通过向更大模型注入小型RL训练模型引起的概率调整来转移推理行为。在多个数学推理基准上的实验表明，与直接的RL训练相比，RAST不仅显著且一致地提升了基模型的推理能力，而且所需GPU内存更低，有时甚至比直接RL训练的模型表现更好。我们的发现为RL驱动的推理的本质提供了新的见解，并提供了在不承担其全部计算成本的情况下扩展其益处的实用策略。RAST的项目页面可访问此链接。 

---
# Studying and Improving Graph Neural Network-based Motif Estimation 

**Title (ZH)**: 基于图神经网络的图模式估计研究与改进 

**Authors**: Pedro C. Vieira, Miguel E. P. Silva, Pedro Manuel Pinto Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2506.15709)  

**Abstract**: Graph Neural Networks (GNNs) are a predominant method for graph representation learning. However, beyond subgraph frequency estimation, their application to network motif significance-profile (SP) prediction remains under-explored, with no established benchmarks in the literature. We propose to address this problem, framing SP estimation as a task independent of subgraph frequency estimation. Our approach shifts from frequency counting to direct SP estimation and modulates the problem as multitarget regression. The reformulation is optimised for interpretability, stability and scalability on large graphs. We validate our method using a large synthetic dataset and further test it on real-world graphs. Our experiments reveal that 1-WL limited models struggle to make precise estimations of SPs. However, they can generalise to approximate the graph generation processes of networks by comparing their predicted SP with the ones originating from synthetic generators. This first study on GNN-based motif estimation also hints at how using direct SP estimation can help go past the theoretical limitations that motif estimation faces when performed through subgraph counting. 

**Abstract (ZH)**: 基于图神经网络的网络模体显著性分布估计：超越子图频率估计的模体温床构建 

---
# Refined Causal Graph Structure Learning via Curvature for Brain Disease Classification 

**Title (ZH)**: 基于曲率的精细化因果图形结构学习在脑疾病分类中的应用 

**Authors**: Falih Gozi Febrinanto, Adonia Simango, Chengpei Xu, Jingjing Zhou, Jiangang Ma, Sonika Tyagi, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2506.15708)  

**Abstract**: Graph neural networks (GNNs) have been developed to model the relationship between regions of interest (ROIs) in brains and have shown significant improvement in detecting brain diseases. However, most of these frameworks do not consider the intrinsic relationship of causality factor between brain ROIs, which is arguably more essential to observe cause and effect interaction between signals rather than typical correlation values. We propose a novel framework called CGB (Causal Graphs for Brains) for brain disease classification/detection, which models refined brain networks based on the causal discovery method, transfer entropy, and geometric curvature strategy. CGB unveils causal relationships between ROIs that bring vital information to enhance brain disease classification performance. Furthermore, CGB also performs a graph rewiring through a geometric curvature strategy to refine the generated causal graph to become more expressive and reduce potential information bottlenecks when GNNs model it. Our extensive experiments show that CGB outperforms state-of-the-art methods in classification tasks on brain disease datasets, as measured by average F1 scores. 

**Abstract (ZH)**: 基于因果图的脑疾病分类/检测框架（CGB） 

---
# Every Rollout Counts: Optimal Resource Allocation for Efficient Test-Time Scaling 

**Title (ZH)**: 每次展开都重要：高效的测试时扩展的最优资源分配 

**Authors**: Xinglin Wang, Yiwei Li, Shaoxiong Feng, Peiwen Yuan, Yueqi Zhang, Jiayi Shi, Chuyi Tan, Boyuan Pan, Yao Hu, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.15707)  

**Abstract**: Test-Time Scaling (TTS) improves the performance of Large Language Models (LLMs) by using additional inference-time computation to explore multiple reasoning paths through search. Yet how to allocate a fixed rollout budget most effectively during search remains underexplored, often resulting in inefficient use of compute at test time. To bridge this gap, we formulate test-time search as a resource allocation problem and derive the optimal allocation strategy that maximizes the probability of obtaining a correct solution under a fixed rollout budget. Within this formulation, we reveal a core limitation of existing search methods: solution-level allocation tends to favor reasoning directions with more candidates, leading to theoretically suboptimal and inefficient use of compute. To address this, we propose Direction-Oriented Resource Allocation (DORA), a provably optimal method that mitigates this bias by decoupling direction quality from candidate count and allocating resources at the direction level. To demonstrate DORA's effectiveness, we conduct extensive experiments on challenging mathematical reasoning benchmarks including MATH500, AIME2024, and AIME2025. The empirical results show that DORA consistently outperforms strong baselines with comparable computational cost, achieving state-of-the-art accuracy. We hope our findings contribute to a broader understanding of optimal TTS for LLMs. 

**Abstract (ZH)**: Test-Time Scaling (TTS) 在固定展开预算下的最优资源分配策略研究：方向导向性资源分配 (DORA) 改进大型语言模型 (LLM) 性能的方法 

---
# MDPO: Multi-Granularity Direct Preference Optimization for Mathematical Reasoning 

**Title (ZH)**: MDPO: 多粒度直接偏好优化的数学推理方法 

**Authors**: Yunze Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.15706)  

**Abstract**: Mathematical reasoning presents a significant challenge for Large Language Models (LLMs) as it requires ensuring the correctness of each reasoning step. Researchers have been strengthening the mathematical reasoning abilities of LLMs through supervised fine-tuning, but due to the inability to suppress incorrect outputs, illusions can easily arise. Recently, Direct Preference Optimization (DPO) has been widely adopted for aligning human intent by using preference data to prevent LLMs from generating incorrect outputs. However, it has shown limited benefits in long-chain mathematical reasoning, mainly because DPO struggles to effectively capture the differences between accepted and rejected answers from preferences in long-chain data. The inconsistency between DPO training and LLMs' generation metrics also affects the effectiveness of suppressing incorrect outputs. We propose the Multi-Granularity Direct Preference Optimization (MDPO) method, optimizing the mathematical reasoning of LLMs at three granularities: Solution2Solution, Inference2Inference, and Step2Step. Solution2Solution focuses on the correctness of entire long-chain reasoning; Inference2Inference concentrates on logical reasoning between steps; Step2Step corrects computational errors in steps, enhancing the computational capabilities of LLMs. Additionally, we unify the training objectives of the three granularities to align with the generation metrics. We conducted experiments on the open-source models Qwen2 and Llama3, achieving improvements of 1.7% and 0.9% on the GSM8K dataset, and 2.3% and 1.2% on the MATH dataset, outperforming DPO and other DPO variant methods. Furthermore, we also provide a pipeline for constructing MDPO training data that is simple and does not require manual annotation costs. 

**Abstract (ZH)**: 多粒度直接偏好优化方法提升大型语言模型的数学推理能力 

---
# Generalisation Bounds of Zero-Shot Economic Forecasting using Time Series Foundation Models 

**Title (ZH)**: 基于时间序列基础模型的零样本经济预测泛化边界研究 

**Authors**: Jittarin Jetwiriyanon, Teo Susnjak, Surangika Ranathunga  

**Link**: [PDF](https://arxiv.org/pdf/2506.15705)  

**Abstract**: This study investigates zero-shot forecasting capabilities of Time Series Foundation Models (TSFMs) for macroeconomic indicators. We apply TSFMs to forecasting economic indicators under univariate conditions, bypassing the need for train bespoke econometric models using and extensive training datasets. Our experiments were conducted on a case study dataset, without additional customisation. We rigorously back-tested three state-of-the-art TSFMs (Chronos, TimeGPT and Moirai) under data-scarce conditions and structural breaks. Our results demonstrate that appropriately engineered TSFMs can internalise rich economic dynamics, accommodate regime shifts, and deliver well-behaved uncertainty estimates out of the box, while matching state-of-the-art multivariate models on this domain. Our findings suggest that, without any fine-tuning, TSFMs can match or exceed classical models during stable economic conditions. However, they are vulnerable to degradation in performances during periods of rapid shocks. The findings offer guidance to practitioners on when zero-shot deployments are viable for macroeconomic monitoring and strategic planning. 

**Abstract (ZH)**: 本研究探讨时间序列基础模型（TSFM）在宏观经济学指标零样本预测能力。我们对单一变量条件下经济指标进行了预测，绕过了使用广泛训练数据集构建定制经济计量模型的需要。我们使用案例研究数据集进行了实验，未进行额外的定制化。我们在数据稀缺和结构性断点条件下严格回测了三种最先进的TSFM（Chronos、TimeGPT和Moirai）。结果显示，适当工程化的TSFM能够内化丰富的经济动态，适应制度变化，并提供即用型的良好行为不确定性估计，同时在该领域与最先进的多元模型表现相当。我们的研究发现，在经济稳定时期，TSFM可以匹配或超越经典模型。然而，在快速冲击时期，它们的表现容易受到影响。这些发现为实务操作者提供了指导，说明了在宏观经济学监控和战略规划中零样本部署的有效性。 

---
# Learn from the Past: Fast Sparse Indexing for Large Language Model Decoding 

**Title (ZH)**: 借鉴past经验：大规模语言模型解码的快速稀疏索引方法 

**Authors**: Feiyu Yao, Qian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15704)  

**Abstract**: As large language models (LLMs) continue to support increasingly longer contexts, the memory demand for key-value (KV) caches during decoding grows rapidly, becoming a critical bottleneck in both GPU memory capacity and PCIe bandwidth. Sparse attention mechanisms alleviate this issue by computing attention weights only for selected key-value pairs. However, their indexing computation typically requires traversing all key vectors, resulting in significant computational and data transfer overhead. To reduce the cost of index retrieval, existing methods often treat each decoding step as an independent process, failing to exploit the temporal correlations embedded in historical decoding information. To this end, we propose LFPS(Learn From the Past for Sparse Indexing), an acceleration method that dynamically constructs sparse indexing candidates based on historical attention patterns. LFPS captures two prevalent trends in decoder attention -vertical patterns (attending to fixed positions) and slash patterns (attending to relative positions) -and incorporates a positional expansion strategy to effectively predict the Top-k indices for the current step. We validate LFPS on challenging long-context benchmarks such as LongBench-RULER, using Llama-3.1-8B-Instruct as the base model. Experimental results show that LFPS achieves up to 22.8$\times$ speedup over full attention and 9.6$\times$ speedup over exact Top-k retrieval on an RTX 4090 GPU and a single CPU core of a Xeon Gold 6430, respectively, while preserving generation accuracy. These results demonstrate that LFPS offers a practical and efficient solution for decoding optimization in long-context LLM inference. 

**Abstract (ZH)**: 从过去中学习以加速稀疏索引：长上下文语言模型解码加速方法 

---
# Federated Incomplete Multi-view Clustering with Globally Fused Graph Guidance 

**Title (ZH)**: 全局融合图引导的联邦不完备多视图聚类 

**Authors**: Guoqing Chao, Zhenghao Zhang, Lei Meng, Jie Wen, Dianhui Chu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15703)  

**Abstract**: Federated multi-view clustering has been proposed to mine the valuable information within multi-view data distributed across different devices and has achieved impressive results while preserving the privacy. Despite great progress, most federated multi-view clustering methods only used global pseudo-labels to guide the downstream clustering process and failed to exploit the global information when extracting features. In addition, missing data problem in federated multi-view clustering task is less explored. To address these problems, we propose a novel Federated Incomplete Multi-view Clustering method with globally Fused Graph guidance (FIMCFG). Specifically, we designed a dual-head graph convolutional encoder at each client to extract two kinds of underlying features containing global and view-specific information. Subsequently, under the guidance of the fused graph, the two underlying features are fused into high-level features, based on which clustering is conducted under the supervision of pseudo-labeling. Finally, the high-level features are uploaded to the server to refine the graph fusion and pseudo-labeling computation. Extensive experimental results demonstrate the effectiveness and superiority of FIMCFG. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 联邦多视图不完备聚类方法及其全局图引导（FIMCFG） 

---
# Minifinetuning: Low-Data Generation Domain Adaptation through Corrective Self-Distillation 

**Title (ZH)**: 迷你微调：基于纠正性自我蒸馏的低数据生成领域适应 

**Authors**: Peter Belcak, Greg Heinrich, Jan Kautz, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2506.15702)  

**Abstract**: Finetuning language models for a new domain inevitably leads to the deterioration of their general performance. This becomes more pronounced the more limited the finetuning data resource.
We introduce minifinetuning (MFT), a method for language model domain adaptation that considerably reduces the effects of overfitting-induced degeneralization in low-data settings and which does so in the absence of any pre-training data for replay. MFT demonstrates 2-10x more favourable specialization-to-degeneralization ratios than standard finetuning across a wide range of models and domains and exhibits an intrinsic robustness to overfitting when data in the new domain is scarce and down to as little as 500 samples.
Employing corrective self-distillation that is individualized on the sample level, MFT outperforms parameter-efficient finetuning methods, demonstrates replay-like degeneralization mitigation properties, and is composable with either for a combined effect. 

**Abstract (ZH)**: 基于少量数据的语言模型领域适应方法：miniFineTuning 

---
# Compiler-R1: Towards Agentic Compiler Auto-tuning with Reinforcement Learning 

**Title (ZH)**: Compiler-R1：面向基于强化学习的有能动性的编译器自动调优 

**Authors**: Haolin Pan, Hongyu Lin, Haoran Luo, Yang Liu, Kaichun Yao, Libo Zhang, Mingjie Xing, Yanjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15701)  

**Abstract**: Compiler auto-tuning optimizes pass sequences to improve performance metrics such as Intermediate Representation (IR) instruction count. Although recent advances leveraging Large Language Models (LLMs) have shown promise in automating compiler tuning, two significant challenges still remain: the absence of high-quality reasoning datasets for agents training, and limited effective interactions with the compilation environment. In this work, we introduce Compiler-R1, the first reinforcement learning (RL)-driven framework specifically augmenting LLM capabilities for compiler auto-tuning. Compiler-R1 features a curated, high-quality reasoning dataset and a novel two-stage end-to-end RL training pipeline, enabling efficient environment exploration and learning through an outcome-based reward. Extensive experiments across seven datasets demonstrate Compiler-R1 achieving an average 8.46% IR instruction count reduction compared to opt -Oz, showcasing the strong potential of RL-trained LLMs for compiler optimization. Our code and datasets are publicly available at this https URL. 

**Abstract (ZH)**: Compiler-R1：增强大语言模型编译器自动调优的强化学习框架 

---
# Contraction Actor-Critic: Contraction Metric-Guided Reinforcement Learning for Robust Path Tracking 

**Title (ZH)**: 收缩actor-critic: 收缩度量引导的强化学习方法及其在稳健路径跟踪中的应用 

**Authors**: Minjae Cho, Hiroyasu Tsukamoto, Huy Trong Tran  

**Link**: [PDF](https://arxiv.org/pdf/2506.15700)  

**Abstract**: Control contraction metrics (CCMs) provide a framework to co-synthesize a controller and a corresponding contraction metric -- a positive-definite Riemannian metric under which a closed-loop system is guaranteed to be incrementally exponentially stable. However, the synthesized controller only ensures that all the trajectories of the system converge to one single trajectory and, as such, does not impose any notion of optimality across an entire trajectory. Furthermore, constructing CCMs requires a known dynamics model and non-trivial effort in solving an infinite-dimensional convex feasibility problem, which limits its scalability to complex systems featuring high dimensionality with uncertainty. To address these issues, we propose to integrate CCMs into reinforcement learning (RL), where CCMs provide dynamics-informed feedback for learning control policies that minimize cumulative tracking error under unknown dynamics. We show that our algorithm, called contraction actor-critic (CAC), formally enhances the capability of CCMs to provide a set of contracting policies with the long-term optimality of RL in a fully automated setting. Given a pre-trained dynamics model, CAC simultaneously learns a contraction metric generator (CMG) -- which generates a contraction metric -- and uses an actor-critic algorithm to learn an optimal tracking policy guided by that metric. We demonstrate the effectiveness of our algorithm relative to established baselines through extensive empirical studies, including simulated and real-world robot experiments, and provide a theoretical rationale for incorporating contraction theory into RL. 

**Abstract (ZH)**: 基于收缩度量的控制收缩度量（CCMs）提供了一种框架，用于协同设计控制器和相应的收缩度量——在该收缩度量下，闭环系统可确保增量指数稳定。然而，合成的控制器仅确保系统的所有轨迹收敛到一条单一轨迹，因此不涵盖整个轨迹上的最优性概念。此外，构造CCMs需要已知的动力学模型，并且需要解决无限维凸可行问题，这限制了其在高维度和不确定性的复杂系统中的可扩展性。为解决这些问题，我们提出了将CCMs集成到强化学习（RL）中，在未知动力学的情况下，CCMs提供动力学指导的反馈，用于学习最小化累积跟踪误差的控制策略。我们证明，我们的算法，称为收缩演员-评论家（CAC），在自动化设置中正式扩展了CCMs的能力，能够提供一组收敛策略，并结合RL长期内的最优性。给定预训练的动力学模型，CAC同时学习一个收缩度量生成器（CMG）——生成收缩度量——并使用演员-评论家算法学习由该度量引导的最佳跟踪策略。我们通过广泛的实证研究，包括仿真和实际机器人实验，展示了该算法相对于现有基线的有效性，并为将收缩理论纳入RL提供了理论依据。 

---
# BLUR: A Benchmark for LLM Unlearning Robust to Forget-Retain Overlap 

**Title (ZH)**: BLUR：一种适用于重叠忘记保留情况下的LLM去学习基准 

**Authors**: Shengyuan Hu, Neil Kale, Pratiksha Thaker, Yiwei Fu, Steven Wu, Virginia Smith  

**Link**: [PDF](https://arxiv.org/pdf/2506.15699)  

**Abstract**: Machine unlearning has the potential to improve the safety of large language models (LLMs) by removing sensitive or harmful information post hoc. A key challenge in unlearning involves balancing between forget quality (effectively unlearning undesirable information) and retain quality (maintaining good performance on other, general tasks). Unfortunately, as we show, current LLM unlearning benchmarks contain highly disparate forget and retain sets -- painting a false picture of the effectiveness of LLM unlearning methods. This can be particularly problematic because it opens the door for benign perturbations, such as relearning attacks, to easily reveal supposedly unlearned knowledge once models are deployed. To address this, we present $\texttt{BLUR}$: a benchmark for LLM unlearning that provides more realistic scenarios of forget-retain overlap. $\texttt{BLUR}$ significantly expands on existing unlearning benchmarks by providing extended evaluation tasks, combined forget/retain queries, and relearning datasets of varying degrees of difficulty. Despite the benign nature of the queries considered, we find that the performance of existing methods drops significantly when evaluated on $\texttt{BLUR}$, with simple approaches performing better on average than more recent methods. These results highlight the importance of robust evaluation and suggest several important directions of future study. Our benchmark is publicly available at: this https URL 

**Abstract (ZH)**: 机器未学习具有潜力通过后剔除敏感或有害信息来改进大型语言模型的安全性。未学习的关键挑战在于在剔除质量（有效地剔除不良信息）和保留质量（在其他通用任务中保持良好性能）之间进行平衡。不幸的是，如我们所展示的，当前的未学习大型语言模型基准包含高度不同的剔除和保留集合——这描绘了一幅未学习方法效果的虚假图景。这可能会特别是因为这使无害的干扰，如重新学习攻击，在模型部署后轻易揭示出被认为已剔除的知识。为了解决这一问题，我们提出了$\texttt{BLUR}$：一种用于大型语言模型未学习的标准，提供了更具现实性的剔除-保留重叠场景。$\texttt{BLUR}$大幅扩展了现有的未学习基准，提供了额外的评估任务、组合剔除/保留查询以及不同难度级别的重新学习数据集。尽管所考虑的查询具有无害的性质，我们发现现有方法在$\texttt{BLUR}$上的表现显著下降，简单的做法在平均性能上优于更近期的方法。这些结果强调了稳健评估的重要性，并建议了未来研究的重要方向。我们的基准已公开可获取：this https URL。 

---
# What Do Latent Action Models Actually Learn? 

**Title (ZH)**: 潜行动作模型究竟学到了什么？ 

**Authors**: Chuheng Zhang, Tim Pearce, Pushi Zhang, Kaixin Wang, Xiaoyu Chen, Wei Shen, Li Zhao, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2506.15691)  

**Abstract**: Latent action models (LAMs) aim to learn action-relevant changes from unlabeled videos by compressing changes between frames as latents. However, differences between video frames can be caused by controllable changes as well as exogenous noise, leading to an important concern -- do latents capture the changes caused by actions or irrelevant noise? This paper studies this issue analytically, presenting a linear model that encapsulates the essence of LAM learning, while being this http URL provides several insights, including connections between LAM and principal component analysis (PCA), desiderata of the data-generating policy, and justification of strategies to encourage learning controllable changes using data augmentation, data cleaning, and auxiliary action-prediction. We also provide illustrative results based on numerical simulation, shedding light on the specific structure of observations, actions, and noise in data that influence LAM learning. 

**Abstract (ZH)**: 潜动作模型（LAMs）旨在通过压缩帧间变化来学习未标注视频中的动作相关变化，但视频帧之间的差异可能由可控变化和外部噪声引起，这引发了一个重要问题：潜变量是否捕获了由动作引起的变化还是无关的噪声？本文从理论上研究了这一问题，建立了一个线性模型来体现LAM学习的本质，同时探讨了LAM与主成分分析（PCA）的联系、数据生成策略的期望、以及通过数据增强、数据清洗和辅助动作预测来促进学习可控变化的合理性。我们也提供了基于数值模拟的示例结果，揭示了影响LAM学习的观察、动作和噪声的具体结构。 

---
# LLM Web Dynamics: Tracing Model Collapse in a Network of LLMs 

**Title (ZH)**: LLM网络中的模型崩溃追踪：探究LLM网络中的模型崩溃现象 

**Authors**: Tianyu Wang, Lingyou Pang, Akira Horiguchi, Carey E. Priebe  

**Link**: [PDF](https://arxiv.org/pdf/2506.15690)  

**Abstract**: The increasing use of synthetic data from the public Internet has enhanced data usage efficiency in large language model (LLM) training. However, the potential threat of model collapse remains insufficiently explored. Existing studies primarily examine model collapse in a single model setting or rely solely on statistical surrogates. In this work, we introduce LLM Web Dynamics (LWD), an efficient framework for investigating model collapse at the network level. By simulating the Internet with a retrieval-augmented generation (RAG) database, we analyze the convergence pattern of model outputs. Furthermore, we provide theoretical guarantees for this convergence by drawing an analogy to interacting Gaussian Mixture Models. 

**Abstract (ZH)**: 公共互联网合成数据在大型语言模型训练中提高了数据使用效率，但模型坍缩的潜在威胁仍未充分探索。现有研究主要在单模型设置中研究模型坍缩或仅依赖统计替代方法。在此项工作中，我们引入了大型语言模型网络动态（LWD）框架，用于在网络层面研究模型坍缩。通过使用检索增强生成（RAG）数据库模拟互联网，我们分析了模型输出的收敛模式。此外，我们通过将此收敛机制与交互的高斯混合模型类比，提供了理论上的保证。 

---
# BASE-Q: Bias and Asymmetric Scaling Enhanced Rotational Quantization for Large Language Models 

**Title (ZH)**: BASE-Q: Bias and Asymmetric Scaling Enhanced Rotational Quantization for Large Language Models 

**Authors**: Liulu He, Shenli Zhen, Karwei Sun, Yijiang Liu, Yufei Zhao, Chongkang Tan, Huanrui Yang, Yuan Du, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.15689)  

**Abstract**: Rotations have become essential to state-of-the-art quantization pipelines for large language models (LLMs) by effectively smoothing outliers in weights and activations. However, further optimizing the rotation parameters offers only limited performance gains and introduces significant training overhead: due to rotation parameter sharing, full-model must be loaded simultaneously to enable backpropagation, resulting in substantial memory consumption and limited practical utility. In this work, we identify two fundamental limitations of current rotational quantization methods: (i) rotation fails to align channel means, resulting in wider quantization bounds and increased rounding errors; and (ii) rotation makes the activation distribution more Gaussian-like, increasing energy loss caused by clipping errors. To address these issues, we introduce \textbf{BASE-Q}, a simple yet powerful approach that combines bias correction and asymmetric scaling to effectively reduce rounding and clipping errors. Furthermore, BASE-Q enables blockwise optimization, eliminating the need for memory-intensive full-model backpropagation. Extensive experiments on various LLMs and benchmarks demonstrate the effectiveness of BASE-Q, narrowing the accuracy gap to full-precision models by 50.5\%, 42.9\%, and 29.2\% compared to QuaRot, SpinQuant, and OSTQuant, respectively. The code will be released soon. 

**Abstract (ZH)**: BASE-Q：一种结合偏置校正和非对称缩放的有效减少舍入和截断误差的方法 

---
# Cellular Traffic Prediction via Deep State Space Models with Attention Mechanism 

**Title (ZH)**: 基于注意力机制的深度状态空间模型在细胞流量预测中的应用 

**Authors**: Hui Ma, Kai Yang, Man-On Pun  

**Link**: [PDF](https://arxiv.org/pdf/2506.15688)  

**Abstract**: Cellular traffic prediction is of great importance for operators to manage network resources and make decisions. Traffic is highly dynamic and influenced by many exogenous factors, which would lead to the degradation of traffic prediction accuracy. This paper proposes an end-to-end framework with two variants to explicitly characterize the spatiotemporal patterns of cellular traffic among neighboring cells. It uses convolutional neural networks with an attention mechanism to capture the spatial dynamics and Kalman filter for temporal modelling. Besides, we can fully exploit the auxiliary information such as social activities to improve prediction performance. We conduct extensive experiments on three real-world datasets. The results show that our proposed models outperform the state-of-the-art machine learning techniques in terms of prediction accuracy. 

**Abstract (ZH)**: 基于端到端框架的邻区细胞流量时空模式预测研究 

---
# Learning from M-Tuple Dominant Positive and Unlabeled Data 

**Title (ZH)**: 学习来自M-元主导正样本和未标注数据 

**Authors**: Jiahe Qin, Junpeng Li, Changchun Hua, Yana Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15686)  

**Abstract**: Label Proportion Learning (LLP) addresses the classification problem where multiple instances are grouped into bags and each bag contains information about the proportion of each class. However, in practical applications, obtaining precise supervisory information regarding the proportion of instances in a specific class is challenging. To better align with real-world application scenarios and effectively leverage the proportional constraints of instances within tuples, this paper proposes a generalized learning framework \emph{MDPU}. Specifically, we first mathematically model the distribution of instances within tuples of arbitrary size, under the constraint that the number of positive instances is no less than that of negative instances. Then we derive an unbiased risk estimator that satisfies risk consistency based on the empirical risk minimization (ERM) method. To mitigate the inevitable overfitting issue during training, a risk correction method is introduced, leading to the development of a corrected risk estimator. The generalization error bounds of the unbiased risk estimator theoretically demonstrate the consistency of the proposed method. Extensive experiments on multiple datasets and comparisons with other relevant baseline methods comprehensively validate the effectiveness of the proposed learning framework. 

**Abstract (ZH)**: MDPU：基于分布统一的风险估计的标记比例学习泛化框架 

---
# Ignition Phase : Standard Training for Fast Adversarial Robustness 

**Title (ZH)**: 点火阶段：标准训练以快速提升 adversarial 抵抗性 

**Authors**: Wang Yu-Hang, Liu ying, Fang liang, Wang Xuelin, Junkang Guo, Shiwei Li, Lei Gao, Jian Liu, Wenfei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.15685)  

**Abstract**: Adversarial Training (AT) is a cornerstone defense, but many variants overlook foundational feature representations by primarily focusing on stronger attack generation. We introduce Adversarial Evolution Training (AET), a simple yet powerful framework that strategically prepends an Empirical Risk Minimization (ERM) phase to conventional AT. We hypothesize this initial ERM phase cultivates a favorable feature manifold, enabling more efficient and effective robustness acquisition. Empirically, AET achieves comparable or superior robustness more rapidly, improves clean accuracy, and cuts training costs by 8-25\%. Its effectiveness is shown across multiple datasets, architectures, and when augmenting established AT methods. Our findings underscore the impact of feature pre-conditioning via standard training for developing more efficient, principled robust defenses. Code is available in the supplementary material. 

**Abstract (ZH)**: 对抗训练（AT）是基础性的防御手段，但许多变体主要关注于更强攻击的生成而忽略了基础特征表示。我们引入了对抗演化训练（AET），这是一种简单而强大的框架，在传统的AT中战略地前接一个经验风险最小化（ERM）阶段。我们假设这一初始的ERM阶段培养出更有利的特征流形，从而使 robustness 的获得更加高效和有效。实验表明，AET 能更快地实现可比或更优的 robustness，提高干净准确性，并降低8-25%的训练成本。其有效性在多种数据集、架构以及增强现有AT方法时均得到验证。我们的发现强调了通过标准训练进行特征预处理对于开发更高效、原则性的 robust 防御的重要性。相关代码附在补充材料中。 

---
# cAST: Enhancing Code Retrieval-Augmented Generation with Structural Chunking via Abstract Syntax Tree 

**Title (ZH)**: cAST：通过抽象语法树的结构分块增强代码检索增强生成 

**Authors**: Yilin Zhang, Xinran Zhao, Zora Zhiruo Wang, Chenyang Yang, Jiayi Wei, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15655)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become essential for large-scale code generation, grounding predictions in external code corpora to improve actuality. However, a critical yet underexplored aspect of RAG pipelines is chunking -- the process of dividing documents into retrievable units. Existing line-based chunking heuristics often break semantic structures, splitting functions or merging unrelated code, which can degrade generation quality. We propose chunking via Abstract Syntax Trees (\ourwork), a structure-aware method that recursively breaks large AST nodes into smaller chunks and merges sibling nodes while respecting size limits. This approach generates self-contained, semantically coherent units across programming languages and tasks, improving performance on diverse code generation tasks, e.g., boosting Recall@5 by 4.3 points on RepoEval retrieval and Pass@1 by 2.67 points on SWE-bench generation. Our work highlights the importance of structure-aware chunking for scaling retrieval-enhanced code intelligence. 

**Abstract (ZH)**: 基于抽象语法树的分块方法（RAG中的结构感知分块）已成为大规模代码生成的关键，能够基于外部代码语料库提高预测的实际性。然而，RAG管道中的一个关键但未充分探索的方面是分块——文档分割成可检索单元的过程。现有的基于行的分块启发式方法往往破坏了语义结构，分割函数或将不相关的代码合并，从而降低了生成质量。我们提出了一种基于抽象语法树的分块方法（\ourwork），这是一种结构感知方法，递归地将大型AST节点分割成较小的块，并合并兄弟节点同时遵守大小限制。这种方法在编程语言和任务中生成自我包含且语义连贯的单元，提高了各种代码生成任务的性能，例如在RepoEval检索上的Recall@5提升了4.3个点，在SWE-bench生成上的Pass@1提升了2.67个点。我们的工作强调了为了扩展增强检索的代码智能，结构感知分块的重要性。 

---
