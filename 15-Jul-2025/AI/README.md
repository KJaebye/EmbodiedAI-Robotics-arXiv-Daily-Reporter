# DeepResearch$^{\text{Eco}}$: A Recursive Agentic Workflow for Complex Scientific Question Answering in Ecology 

**Title (ZH)**: DeepResearch$^{\text{Eco}}$: 一种用于生态学复杂科学问题回答的递归代理工作流 

**Authors**: Jennifer D'Souza, Endres Keno Sander, Andrei Aioanei  

**Link**: [PDF](https://arxiv.org/pdf/2507.10522)  

**Abstract**: We introduce DeepResearch$^{\text{Eco}}$, a novel agentic LLM-based system for automated scientific synthesis that supports recursive, depth- and breadth-controlled exploration of original research questions -- enhancing search diversity and nuance in the retrieval of relevant scientific literature. Unlike conventional retrieval-augmented generation pipelines, DeepResearch enables user-controllable synthesis with transparent reasoning and parameter-driven configurability, facilitating high-throughput integration of domain-specific evidence while maintaining analytical rigor. Applied to 49 ecological research questions, DeepResearch achieves up to a 21-fold increase in source integration and a 14.9-fold rise in sources integrated per 1,000 words. High-parameter settings yield expert-level analytical depth and contextual diversity.
Source code available at: this https URL. 

**Abstract (ZH)**: 我们介绍DeepResearch$^{\text{Eco}}$，这是一种基于代理的大规模语言模型系统，用于自动化科学综合，支持递归、深度和广度受控的原创研究问题探索——增强检索相关科学文献的多样性和细微差别。与传统检索增强生成管道不同，DeepResearch 允许用户控制的综合，并具有透明推理和参数驱动的配置能力，便于高通量集成领域特定的证据，同时保持分析严谨性。将DeepResearch应用于49个生态研究问题，可实现多达21倍的来源集成增加，每1000字集成的来源增加14.9倍。高参数设置可实现专家级的分析深度和上下文多样性。源代码可通过以下链接获得：this https URL。 

---
# Acquiring and Adapting Priors for Novel Tasks via Neural Meta-Architectures 

**Title (ZH)**: 通过神经元元架构获取和适应新任务的先验知识 

**Authors**: Sudarshan Babu  

**Link**: [PDF](https://arxiv.org/pdf/2507.10446)  

**Abstract**: The ability to transfer knowledge from prior experiences to novel tasks stands as a pivotal capability of intelligent agents, including both humans and computational models. This principle forms the basis of transfer learning, where large pre-trained neural networks are fine-tuned to adapt to downstream tasks. Transfer learning has demonstrated tremendous success, both in terms of task adaptation speed and performance. However there are several domains where, due to lack of data, training such large pre-trained models or foundational models is not a possibility - computational chemistry, computational immunology, and medical imaging are examples. To address these challenges, our work focuses on designing architectures to enable efficient acquisition of priors when large amounts of data are unavailable. In particular, we demonstrate that we can use neural memory to enable adaptation on non-stationary distributions with only a few samples. Then we demonstrate that our hypernetwork designs (a network that generates another network) can acquire more generalizable priors than standard networks when trained with Model Agnostic Meta-Learning (MAML). Subsequently, we apply hypernetworks to 3D scene generation, demonstrating that they can acquire priors efficiently on just a handful of training scenes, thereby leading to faster text-to-3D generation. We then extend our hypernetwork framework to perform 3D segmentation on novel scenes with limited data by efficiently transferring priors from earlier viewed scenes. Finally, we repurpose an existing molecular generative method as a pre-training framework that facilitates improved molecular property prediction, addressing critical challenges in computational immunology 

**Abstract (ZH)**: 具备将先前经验的知识转移到新任务的能力是智能代理，包括人类和计算模型的关键能力。这一原则构成了迁移学习的基础，其中大规模预训练神经网络被微调以适应下游任务。迁移学习在任务适应速度和性能方面取得了巨大成功。然而，在缺乏数据的某些领域中，训练如此大规模的预训练模型或基础模型是不可能的——这在计算化学、计算免疫学和医学成像等领域中例证明显。为了解决这些挑战，我们的工作集中在设计架构以在缺乏大量数据时高效获取先验知识。特别地，我们证明可以通过神经记忆在只有少量样本的情况下使模型适应非平稳分布。然后，我们证明我们的超网络设计（生成另一个网络的网络）在使用模型无关元学习（MAML）进行训练时能够获得比标准网络更广泛适用的先验知识。随后，我们将超网络应用于3D场景生成，证明它们可以在少量训练场景下高效地获取先验知识，从而加快文本到3D的生成速度。我们进一步扩展了超网络框架，在有限数据的情况下对新场景进行3D分割，通过高效地从先前观看的场景转移先验知识来实现。最后，我们将现有的分子生成方法重新用于预训练框架，以改善分子性质预测，解决计算免疫学中的关键挑战。 

---
# SentiDrop: A Multi Modal Machine Learning model for Predicting Dropout in Distance Learning 

**Title (ZH)**: SentiDrop: 多模态机器学习模型预测远程学习中的辍学现象 

**Authors**: Meriem Zerkouk, Miloud Mihoubi, Belkacem Chikhaoui  

**Link**: [PDF](https://arxiv.org/pdf/2507.10421)  

**Abstract**: School dropout is a serious problem in distance learning, where early detection is crucial for effective intervention and student perseverance. Predicting student dropout using available educational data is a widely researched topic in learning analytics. Our partner's distance learning platform highlights the importance of integrating diverse data sources, including socio-demographic data, behavioral data, and sentiment analysis, to accurately predict dropout risks. In this paper, we introduce a novel model that combines sentiment analysis of student comments using the Bidirectional Encoder Representations from Transformers (BERT) model with socio-demographic and behavioral data analyzed through Extreme Gradient Boosting (XGBoost). We fine-tuned BERT on student comments to capture nuanced sentiments, which were then merged with key features selected using feature importance techniques in XGBoost. Our model was tested on unseen data from the next academic year, achieving an accuracy of 84\%, compared to 82\% for the baseline model. Additionally, the model demonstrated superior performance in other metrics, such as precision and F1-score. The proposed method could be a vital tool in developing personalized strategies to reduce dropout rates and encourage student perseverance 

**Abstract (ZH)**: 远程学习中学生辍学是一个严重的问题，早期检测对于有效干预和学生坚持不懈至关重要。利用可用的教育数据预测学生辍学是学习分析领域的广泛研究课题。我们的合作伙伴的远程学习平台强调集成多样数据源的重要性，包括社会人口统计数据、行为数据和情感分析，以准确预测辍学风险。在本文中，我们提出了一种新颖的模型，该模型结合了使用双向编码器表示形式（BERT）模型进行的学生评论情感分析与通过极端梯度提升（XGBoost）分析的社会人口统计学和行为数据。我们针对学生评论对BERT进行了微调，以捕捉细微的情感，然后将这些情感与XGBoost特征重要性技术选择的关键特征合并。该模型在下一年未见过的数据上进行了测试，准确率为84%，而基线模型为82%。此外，该模型在其他指标（如精确度和F1分数）上也表现出色。所提出的方法可以成为开发个性化策略以降低辍学率和鼓励学生坚持不懈的重要工具。 

---
# Instance space analysis of the capacitated vehicle routing problem 

**Title (ZH)**: 带容量约束车辆路径问题的实例空间分析 

**Authors**: Alessandra M. M. M. Gouvêa, Nuno Paulos, Eduardo Uchoa e Mariá C. V. Nascimento  

**Link**: [PDF](https://arxiv.org/pdf/2507.10397)  

**Abstract**: This paper seeks to advance CVRP research by addressing the challenge of understanding the nuanced relationships between instance characteristics and metaheuristic (MH) performance. We present Instance Space Analysis (ISA) as a valuable tool that allows for a new perspective on the field. By combining the ISA methodology with a dataset from the DIMACS 12th Implementation Challenge on Vehicle Routing, our research enabled the identification of 23 relevant instance characteristics. Our use of the PRELIM, SIFTED, and PILOT stages, which employ dimensionality reduction and machine learning methods, allowed us to create a two-dimensional projection of the instance space to understand how the structure of instances affect the behavior of MHs. A key contribution of our work is that we provide a projection matrix, which makes it straightforward to incorporate new instances into this analysis and allows for a new method for instance analysis in the CVRP field. 

**Abstract (ZH)**: 本文旨在通过探讨实例特征与元启发式算法性能之间的复杂关系，推进车辆路线问题（CVRP）研究。我们提出了实例空间分析（ISA）作为一项有价值的工具，提供了该领域的全新视角。结合ISA方法与DIMACS第12届实现挑战赛中的车辆路由数据集，我们的研究识别了23个相关实例特征。通过PRELIM、SIFTED和PILOT阶段，我们采用了降维和机器学习方法，创建了实例空间的二维投影，以理解实例结构如何影响元启发式算法的行为。本文的一个重要贡献是，我们提供了投影矩阵，便于将新实例纳入这种分析，并为CVRP领域提供了实例分析的新方法。 

---
# Toward Real-World Table Agents: Capabilities, Workflows, and Design Principles for LLM-based Table Intelligence 

**Title (ZH)**: 面向现实世界的表格智能代理：基于LLM的表格智能能力、工作流及设计原则 

**Authors**: Jiaming Tian, Liyao Li, Wentao Ye, Haobo Wang, Lingxin Wang, Lihua Yu, Zujie Ren, Gang Chen, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.10281)  

**Abstract**: Tables are fundamental in domains such as finance, healthcare, and public administration, yet real-world table tasks often involve noise, structural heterogeneity, and semantic complexity--issues underexplored in existing research that primarily targets clean academic datasets. This survey focuses on LLM-based Table Agents, which aim to automate table-centric workflows by integrating preprocessing, reasoning, and domain adaptation. We define five core competencies--C1: Table Structure Understanding, C2: Table and Query Semantic Understanding, C3: Table Retrieval and Compression, C4: Executable Reasoning with Traceability, and C5: Cross-Domain Generalization--to analyze and compare current approaches. In addition, a detailed examination of the Text-to-SQL Agent reveals a performance gap between academic benchmarks and real-world scenarios, especially for open-source models. Finally, we provide actionable insights to improve the robustness, generalization, and efficiency of LLM-based Table Agents in practical settings. 

**Abstract (ZH)**: 表格是金融、医疗和公共管理等领域的基本工具，然而现实世界中的表格任务往往涉及噪声、结构异质性和语义复杂性——这些问题在当前主要针对干净学术数据集的研究中尚未充分探讨。本文综述了基于大模型的表格代理，这些代理旨在通过整合预处理、推理和领域适应来自动化以表格为中心的工作流。我们定义了五项核心能力——C1：表格结构理解，C2：表格和查询语义理解，C3：表格检索与压缩，C4：具有可追溯性的可执行推理，C5：跨域泛化，来分析和比较当前的方法。此外，详细分析了文本到SQL代理，揭示了学术基准与现实世界场景之间的表现差距，尤其是开源模型方面。最后，我们提供了实用建议，以提高基于大模型的表格代理在实际应用中的鲁棒性、泛化能力和效率。 

---
# Survey for Categorising Explainable AI Studies Using Data Analysis Task Frameworks 

**Title (ZH)**: 基于数据分析任务框架的可解释人工智能研究分类综述 

**Authors**: Hamzah Ziadeh, Hendrik Knoche  

**Link**: [PDF](https://arxiv.org/pdf/2507.10208)  

**Abstract**: Research into explainable artificial intelligence (XAI) for data analysis tasks suffer from a large number of contradictions and lack of concrete design recommendations stemming from gaps in understanding the tasks that require AI assistance. In this paper, we drew on multiple fields such as visual analytics, cognition, and dashboard design to propose a method for categorising and comparing XAI studies under three dimensions: what, why, and who. We identified the main problems as: inadequate descriptions of tasks, context-free studies, and insufficient testing with target users. We propose that studies should specifically report on their users' domain, AI, and data analysis expertise to illustrate the generalisability of their findings. We also propose study guidelines for designing and reporting XAI tasks to improve the XAI community's ability to parse the rapidly growing field. We hope that our contribution can help researchers and designers better identify which studies are most relevant to their work, what gaps exist in the research, and how to handle contradictory results regarding XAI design. 

**Abstract (ZH)**: 关于可解释人工智能（XAI）在数据分析任务中的研究受到大量矛盾和具体设计建议不足的问题，这些问题是由于对需要AI辅助的任务理解不足造成的。在本文中，我们借鉴了视觉分析、认知和仪表板设计等多个领域，提出了根据三个维度（是什么、为什么、为谁）对XAI研究进行分类和比较的方法。我们发现主要问题包括任务描述不充分、脱离上下文的研究以及对目标用户的测试不足。我们建议研究应具体报告其用户在领域、AI以及数据分析方面的专长，以阐明其研究发现的普遍性。我们还提出了设计和报告XAI任务的研究指南，以提高XAI社区处理快速发展领域中结果矛盾的能力。我们希望我们的贡献能够帮助研究人员和设计师更好地识别哪些研究与他们的工作最为相关，研究中存在哪些缺口，以及如何处理关于XAI设计的矛盾结果。 

---
# Should We Ever Prefer Decision Transformer for Offline Reinforcement Learning? 

**Title (ZH)**: 我们应该 ever 偏好决策转换器进行离线强化学习吗？ 

**Authors**: Yumi Omori, Zixuan Dong, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2507.10174)  

**Abstract**: In recent years, extensive work has explored the application of the Transformer architecture to reinforcement learning problems. Among these, Decision Transformer (DT) has gained particular attention in the context of offline reinforcement learning due to its ability to frame return-conditioned policy learning as a sequence modeling task. Most recently, Bhargava et al. (2024) provided a systematic comparison of DT with more conventional MLP-based offline RL algorithms, including Behavior Cloning (BC) and Conservative Q-Learning (CQL), and claimed that DT exhibits superior performance in sparse-reward and low-quality data settings.
In this paper, through experimentation on robotic manipulation tasks (Robomimic) and locomotion benchmarks (D4RL), we show that MLP-based Filtered Behavior Cloning (FBC) achieves competitive or superior performance compared to DT in sparse-reward environments. FBC simply filters out low-performing trajectories from the dataset and then performs ordinary behavior cloning on the filtered dataset. FBC is not only very straightforward, but it also requires less training data and is computationally more efficient. The results therefore suggest that DT is not preferable for sparse-reward environments. From prior work, arguably, DT is also not preferable for dense-reward environments. Thus, we pose the question: Is DT ever preferable? 

**Abstract (ZH)**: 近年来，广泛的工作探索了Transformer架构在强化学习问题中的应用。其中，决策Transformer（DT）在离线强化学习领域尤为引人关注，因其能够将基于回报的策略学习框定为序列建模任务。最近，Bhargava等人（2024）系统性地比较了DT与传统的基于MLP的离线RL算法，包括行为克隆（BC）和保守Q学习（CQL），并声称在稀疏奖励和低质量数据环境中，DT表现出更优的性能。

在本文中，通过在机器人操纵任务（Robomimic）和运动基准测试（D4RL）上的实验，我们展示了基于MLP的过滤行为克隆（FBC）在稀疏奖励环境中可达到与DT相当甚至更优的性能。FBC简单地从数据集中过滤掉低性能的轨迹，然后在过滤后的数据集上进行普通的行为克隆。FBC不仅非常简洁，而且所需训练数据较少，计算效率更高。因此，这些结果表明，对于稀疏奖励环境，DT并不占优势。从以前的工作来看，DT对于密集奖励环境也不占优势。因此，我们提出了一个问题：决策Transformer（DT）是否有时更优？ 

---
# Introducing the Swiss Food Knowledge Graph: AI for Context-Aware Nutrition Recommendation 

**Title (ZH)**: 介绍瑞士食品知识图谱：面向场景的营养推荐AI 

**Authors**: Lubnaa Abdur Rahman, Ioannis Papathanail, Stavroula Mougiakakou  

**Link**: [PDF](https://arxiv.org/pdf/2507.10156)  

**Abstract**: AI has driven significant progress in the nutrition field, especially through multimedia-based automatic dietary assessment. However, existing automatic dietary assessment systems often overlook critical non-visual factors, such as recipe-specific ingredient substitutions that can significantly alter nutritional content, and rarely account for individual dietary needs, including allergies, restrictions, cultural practices, and personal preferences. In Switzerland, while food-related information is available, it remains fragmented, and no centralized repository currently integrates all relevant nutrition-related aspects within a Swiss context. To bridge this divide, we introduce the Swiss Food Knowledge Graph (SwissFKG), the first resource, to our best knowledge, to unite recipes, ingredients, and their substitutions with nutrient data, dietary restrictions, allergen information, and national nutrition guidelines under one graph. We establish a LLM-powered enrichment pipeline for populating the graph, whereby we further present the first benchmark of four off-the-shelf (<70 B parameter) LLMs for food knowledge augmentation. Our results demonstrate that LLMs can effectively enrich the graph with relevant nutritional information. Our SwissFKG goes beyond recipe recommendations by offering ingredient-level information such as allergen and dietary restriction information, and guidance aligned with nutritional guidelines. Moreover, we implement a Graph-RAG application to showcase how the SwissFKG's rich natural-language data structure can help LLM answer user-specific nutrition queries, and we evaluate LLM-embedding pairings by comparing user-query responses against predefined expected answers. As such, our work lays the foundation for the next generation of dietary assessment tools that blend visual, contextual, and cultural dimensions of eating. 

**Abstract (ZH)**: AI在营养领域的进展，尤其是通过基于多媒体的自动膳食评估。然而，现有的自动膳食评估系统往往忽略了重要的非视觉因素，如食谱特定的原料替代，这些替代可以显著改变营养价值，同时也很少考虑到个人的饮食需求，包括过敏、限制、文化习俗和个人偏好。在瑞士，虽然与食物相关的信息是可用的，但这些信息仍然碎片化，目前没有集中化的数据仓库能够整合所有与瑞士相关的营养方面内容。为了解决这一问题，我们介绍了瑞士食品知识图谱（SwissFKG），这是迄今为止我们所知的第一个资源，将食谱、原料及其替代品与营养数据、饮食限制、过敏信息和国家营养指南统一在一个图谱中。我们建立了一个基于大语言模型（LLM）的增强管道来填充图谱，并进一步展示了四项现成的LLM（参数量少于70B）在食品知识增强方面的首个基准测试。我们的结果表明，LLM能够有效地为图谱提供相关营养信息。与现有的食谱推荐不同，我们的SwissFKG提供了包括过敏和饮食限制在内的原料级信息，并提供了符合营养指南的指导。此外，我们实现了Graph-RAG应用，展示了SwissFKG丰富的自然语言数据结构如何帮助LLM回答用户特定的营养查询，并通过将用户查询的响应与预定义的标准答案进行对比来评估LLM嵌入对齐。因此，我们的工作为下一代融合视觉、上下文和文化维度的膳食评估工具奠定了基础。 

---
# Adaptability in Multi-Agent Reinforcement Learning: A Framework and Unified Review 

**Title (ZH)**: 多代理强化学习中的适应性：一个框架与统一综述 

**Authors**: Siyi Hu, Mohamad A Hady, Jianglin Qiao, Jimmy Cao, Mahardhika Pratama, Ryszard Kowalczyk  

**Link**: [PDF](https://arxiv.org/pdf/2507.10142)  

**Abstract**: Multi-Agent Reinforcement Learning (MARL) has shown clear effectiveness in coordinating multiple agents across simulated benchmarks and constrained scenarios. However, its deployment in real-world multi-agent systems (MAS) remains limited, primarily due to the complex and dynamic nature of such environments. These challenges arise from multiple interacting sources of variability, including fluctuating agent populations, evolving task goals, and inconsistent execution conditions. Together, these factors demand that MARL algorithms remain effective under continuously changing system configurations and operational demands. To better capture and assess this capacity for adjustment, we introduce the concept of \textit{adaptability} as a unified and practically grounded lens through which to evaluate the reliability of MARL algorithms under shifting conditions, broadly referring to any changes in the environment dynamics that may occur during learning or execution. Centred on the notion of adaptability, we propose a structured framework comprising three key dimensions: learning adaptability, policy adaptability, and scenario-driven adaptability. By adopting this adaptability perspective, we aim to support more principled assessments of MARL performance beyond narrowly defined benchmarks. Ultimately, this survey contributes to the development of algorithms that are better suited for deployment in dynamic, real-world multi-agent systems. 

**Abstract (ZH)**: 多代理强化学习的适应性在动态现实世界多代理系统中的评估与应用 

---
# FRSICL: LLM-Enabled In-Context Learning Flight Resource Allocation for Fresh Data Collection in UAV-Assisted Wildfire Monitoring 

**Title (ZH)**: FRSICL：基于LLM的上下文学习航空器辅助野火监测新数据采集飞行资源分配算法 

**Authors**: Yousef Emami, Hao Zhou, Miguel Gutierrez Gaitan, Kai Li, Luis Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2507.10134)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are vital for public safety, particularly in wildfire monitoring, where early detection minimizes environmental impact. In UAV-Assisted Wildfire Monitoring (UAWM) systems, joint optimization of sensor transmission scheduling and velocity is critical for minimizing Age of Information (AoI) from stale sensor data. Deep Reinforcement Learning (DRL) has been used for such optimization; however, its limitations such as low sampling efficiency, simulation-to-reality gaps, and complex training render it unsuitable for time-critical applications like wildfire monitoring. This paper introduces a new online Flight Resource Allocation scheme based on LLM-Enabled In-Context Learning (FRSICL) to jointly optimize the UAV's flight control and data collection schedule along the trajectory in real time, thereby asymptotically minimizing the average AoI across ground sensors. In contrast to DRL, FRSICL generates data collection schedules and controls velocity using natural language task descriptions and feedback from the environment, enabling dynamic decision-making without extensive retraining. Simulation results confirm the effectiveness of the proposed FRSICL compared to Proximal Policy Optimization (PPO) and Nearest-Neighbor baselines. 

**Abstract (ZH)**: 基于LLM辅助上下文学习的实时飞行资源分配方案：应用于无人机辅助 wildfire监测系统中的传感器数据收集与飞行控制联合优化 

---
# Could you be wrong: Debiasing LLMs using a metacognitive prompt for improving human decision making 

**Title (ZH)**: 你可能会出错：使用元认知提示去偏差化以提升人类决策质量 

**Authors**: Thomas T. Hills  

**Link**: [PDF](https://arxiv.org/pdf/2507.10124)  

**Abstract**: Identifying bias in LLMs is ongoing. Because they are still in development, what is true today may be false tomorrow. We therefore need general strategies for debiasing that will outlive current models. Strategies developed for debiasing human decision making offer one promising approach as they incorporate an LLM-style prompt intervention designed to bring latent knowledge into awareness during decision making. LLMs trained on vast amounts of information contain information about potential biases, counter-arguments, and contradictory evidence, but that information may only be brought to bear if prompted. Metacognitive prompts developed in the human decision making literature are designed to achieve this, and as I demonstrate here, they show promise with LLMs. The prompt I focus on here is "could you be wrong?" Following an LLM response, this prompt leads LLMs to produce additional information, including why they answered as they did, errors, biases, contradictory evidence, and alternatives, none of which were apparent in their initial response. Indeed, this metaknowledge often reveals that how LLMs and users interpret prompts are not aligned. Here I demonstrate this prompt using a set of questions taken from recent articles about LLM biases, including implicit discriminatory biases and failures of metacognition. "Could you be wrong" prompts the LLM to identify its own biases and produce cogent metacognitive reflection. I also present another example involving convincing but incomplete information, which is readily corrected by the metacognitive prompt. In sum, this work argues that human psychology offers a new avenue for prompt engineering, leveraging a long history of effective prompt-based improvements to human decision making. 

**Abstract (ZH)**: 识别LLM中的偏见是一个持续的过程。由于它们仍然在开发中，今天的真理可能 tomorrow 就会变成谬误。因此，我们需要通用的去偏策略，这些策略能够超越当前的模型。来自人类决策制定领域的去偏策略提供了一种有希望的方法，因为它们包含了类似于LLM提示干预的设计，旨在在决策制定过程中使潜在知识变得意识化。大量信息训练的LLM包含了关于潜在偏见、反论和矛盾证据的信息，但这些信息只有在受到提示时才能发挥作用。来自人类决策制定文献的元认知提示旨在实现这一点，并如我在这篇工作中所展示的，这些提示在LLM中显示出潜力。“你可能錯了吗？”这个提示就是我关注的重点之一。在LLM响应之后，这个提示促使LLM产生额外的信息，包括其回答的理由、错误、偏见、矛盾证据和替代方案，这些在初始响应中均未体现。事实上，这种元知识常常揭示出LLM和用户对提示的理解并不一致。我使用近期关于LLM偏见的文章中的问题集，其中包含隐含的歧视性偏见和元认知失败，展示了这种提示。此外，我还提供了一个包含令人信服但不完整信息的例子，这种信息可以通过元认知提示轻易修正。总的来说，本文认为人类心理学为我们提供了新的提示工程途径，利用了对人类决策制定有长期有效性改进的提示历史。 

---
# Analysis of AI Techniques for Orchestrating Edge-Cloud Application Migration 

**Title (ZH)**: 基于AI技术的边缘-云应用迁移编排分析 

**Authors**: Sadig Gojayev, Ahmad Anaqreh, Carolina Fortuna  

**Link**: [PDF](https://arxiv.org/pdf/2507.10119)  

**Abstract**: Application migration in edge-cloud system enables high QoS and cost effective service delivery. However, automatically orchestrating such migration is typically solved with heuristic approaches. Starting from the Markov Decision Process (MDP), in this paper, we identify, analyze and compare selected state-of-the-art Artificial Intelligence (AI) planning and Reinforcement Learning (RL) approaches for solving the class of edge-cloud application migration problems that can be modeled as Towers of Hanoi (ToH) problems. We introduce a new classification based on state space definition and analyze the compared models also through this lense. The aim is to understand available techniques capable of orchestrating such application migration in emerging computing continuum environments. 

**Abstract (ZH)**: 边缘-云系统中的应用迁移能实现高质量服务和成本效益交付。然而，自动 orchestrating 这样的迁移通常通过启发式方法解决。基于马尔可夫决策过程（MDP），本文识别、分析并比较了用于解决可以建模为汉诺塔问题（ToH）的应用迁移问题的最新人工智能（AI）规划和强化学习（RL）方法。我们引入了一种基于状态空间定义的新分类，并通过这种视角分析所比较的模型。目标是理解在新兴计算 continuum 环境中 orchestrating 这种应用迁移可用的技术。 

---
# BlueGlass: A Framework for Composite AI Safety 

**Title (ZH)**: BlueGlass：一种复合AI安全框架 

**Authors**: Harshal Nandigramwar, Syed Qutub, Kay-Ulrich Scholl  

**Link**: [PDF](https://arxiv.org/pdf/2507.10106)  

**Abstract**: As AI systems become increasingly capable and ubiquitous, ensuring the safety of these systems is critical. However, existing safety tools often target different aspects of model safety and cannot provide full assurance in isolation, highlighting a need for integrated and composite methodologies. This paper introduces BlueGlass, a framework designed to facilitate composite AI safety workflows by providing a unified infrastructure enabling the integration and composition of diverse safety tools that operate across model internals and outputs. Furthermore, to demonstrate the utility of this framework, we present three safety-oriented analyses on vision-language models for the task of object detection: (1) distributional evaluation, revealing performance trade-offs and potential failure modes across distributions; (2) probe-based analysis of layer dynamics highlighting shared hierarchical learning via phase transition; and (3) sparse autoencoders identifying interpretable concepts. More broadly, this work contributes foundational infrastructure and findings for building more robust and reliable AI systems. 

**Abstract (ZH)**: 随着AI系统的能力不断增强和普及，确保这些系统的安全至关重要。然而，现有的安全工具通常针对模型安全的不同方面，在孤立情况下无法提供全面保证，突显出集成和综合方法的需求。本文介绍了BlueGlass框架，该框架旨在通过提供统一基础设施来促进多样化的安全工具的集成与组合，这些工具可以跨越模型内部和输出进行操作。此外，为了展示该框架的应用价值，我们对视觉-语言模型进行了三项面向安全性的分析，用于目标检测任务：（1）分布评估，揭示不同数据分布下的性能权衡和潜在故障模式；（2）基于探针的层动态分析，突出通过相变共享的层级学习；（3）稀疏自编码器识别可解释的概念。更广泛地说，这项工作为构建更加稳健和可靠的AI系统提供了基础架构和发现。 

---
# On Gradual Semantics for Assumption-Based Argumentation 

**Title (ZH)**: 基于假设的论证渐进语义学 

**Authors**: Anna Rapberger, Fabrizio Russo, Antonio Rago, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2507.10076)  

**Abstract**: In computational argumentation, gradual semantics are fine-grained alternatives to extension-based and labelling-based semantics . They ascribe a dialectical strength to (components of) arguments sanctioning their degree of acceptability. Several gradual semantics have been studied for abstract, bipolar and quantitative bipolar argumentation frameworks (QBAFs), as well as, to a lesser extent, for some forms of structured argumentation. However, this has not been the case for assumption-based argumentation (ABA), despite it being a popular form of structured argumentation with several applications where gradual semantics could be useful. In this paper, we fill this gap and propose a family of novel gradual semantics for equipping assumptions, which are the core components in ABA frameworks, with dialectical strengths. To do so, we use bipolar set-based argumentation frameworks as an abstraction of (potentially non-flat) ABA frameworks and generalise state-of-the-art modular gradual semantics for QBAFs. We show that our gradual ABA semantics satisfy suitable adaptations of desirable properties of gradual QBAF semantics, such as balance and monotonicity. We also explore an argument-based approach that leverages established QBAF modular semantics directly, and use it as baseline. Finally, we conduct experiments with synthetic ABA frameworks to compare our gradual ABA semantics with its argument-based counterpart and assess convergence. 

**Abstract (ZH)**: 在计算论辩中，渐进语义是基于扩展和基于标签语义的细粒度替代方案。它们赋予论据（或论据的组成部分） dialectical 强度，以表明它们的接受程度。已经研究了几种渐进语义，适用于抽象论辩框架、双极性和定量双极性论辩框架（QBAFs），以及在一定程度上适用于某些结构化论辩形式。然而，这些研究尚未应用于假设基础论辩（ABA），尽管ABA是广泛应用于多种应用场景的一种结构化论辩形式，并且渐进语义在其中可能非常有用。在本文中，我们填补了这一空白，并提出了一种为假设（ABA框架的核心组成部分）赋予 dialectical 强度的新型渐进语义家族。为此，我们使用双极性集合论辩框架作为潜在非平滑的ABA框架的抽象，并对QBAFs的最先进模块化渐进语义进行了泛化。我们证明，我们的渐进ABA语义满足适合的渐进QBAF语义的可取性质的适当改编，例如平衡性和单调性。我们还探索了一种基于论辩的方法，直接利用已建立的QBAF模块化语义，并将其用作基线。最后，我们使用合成的ABA框架进行实验，比较我们的渐进ABA语义与其论辩基线，并评估收敛性。 

---
# Automating SPARQL Query Translations between DBpedia and Wikidata 

**Title (ZH)**: ,DBpedia和Wikidata之间的SPARQL查询自动化翻译 

**Authors**: Malte Christian Bartels, Debayan Banerjee, Ricardo Usbeck  

**Link**: [PDF](https://arxiv.org/pdf/2507.10045)  

**Abstract**: This paper investigates whether state-of-the-art Large Language Models (LLMs) can automatically translate SPARQL between popular Knowledge Graph (KG) schemas. We focus on translations between the DBpedia and Wikidata KG, and later on DBLP and OpenAlex KG. This study addresses a notable gap in KG interoperability research by rigorously evaluating LLM performance on SPARQL-to-SPARQL translation. Two benchmarks are assembled, where the first align 100 DBpedia-Wikidata queries from QALD-9-Plus; the second contains 100 DBLP queries aligned to OpenAlex, testing generalizability beyond encyclopaedic KGs. Three open LLMs: Llama-3-8B, DeepSeek-R1-Distill-Llama-70B, and Mistral-Large-Instruct-2407 are selected based on their sizes and architectures and tested with zero-shot, few-shot, and two chain-of-thought variants. Outputs were compared with gold answers, and resulting errors were categorized. We find that the performance varies markedly across models and prompting strategies, and that translations for Wikidata to DBpedia work far better than translations for DBpedia to Wikidata. 

**Abstract (ZH)**: 本文研究最新大型语言模型（LLMs）是否能自动翻译SPARQL语句，以实现流行知识图谱（KG）模式之间的翻译。研究重点在于DBpedia和Wikidata KG之间的翻译，后续扩展到DBLP和OpenAlex KG。本研究通过严格评估LLM在SPARQL-to-SPARQL翻译上的性能，填补了知识图谱互操作性研究中的一个重要空白。构建了两个基准测试集，第一个基准集包含100个从QALD-9-Plus对齐的DBpedia-Wikidata查询，第二个基准集包含100个对齐到OpenAlex的DBLP查询，以测试跨百科知识图谱之外的一般化能力。选择了三个开源LLM：Llama-3-8B、DeepSeek-R1-Distill-Llama-70B和Mistral-Large-Instruct-2407，基于它们的大小和架构，并采用零样本、少样本和两链式思考策略进行测试。输出结果与标准答案进行对比，并分类统计错误。研究发现，不同模型和提示策略的性能差异明显，且Wikidata到DBpedia的翻译效果远优于DBpedia到Wikidata的翻译。 

---
# Deep Hidden Cognition Facilitates Reliable Chain-of-Thought Reasoning 

**Title (ZH)**: 深层隐藏认知促进可靠链式思维推理 

**Authors**: Zijun Chen, Wenbo Hu, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.10007)  

**Abstract**: Chain of Thought (CoT) reasoning has demonstrated remarkable deep reasoning capabilities in both large language models (LLMs) and multimodal large language models (MLLMs). However, its reliability is often undermined by the accumulation of errors in intermediate steps. This paper introduces an novel approach to calibrate the CoT reasoning accuracy by leveraging the model's intrinsic veracity encoding. We discover that specific attention head activations reliably reflect the truthfulness of reasoning steps in CoT. Based on this insight, we train a confidence predictor to evaluate the correctness of each reasoning step using these truthfulness-sensitive activations, dynamically selecting the most plausible reasoning path via beam search. Experimental results demonstrate that our method significantly outperforms the state-of-the-art baselines (e.g., Few-Shot CoT, Self-Consistency, and Self-Evaluation Guided Beam Search) across the mathematical, symbolic, and commonsense reasoning tasks, exhibiting superior accuracy and reliability in both unimodal and multimodal settings. We further validate the approach on large reasoning models, confirming its applicability to specialized reasoning models. Additionally, we explore the role of the model's self-correction ability in CoT reasoning. This work provides a novel reliability improvement path for CoT reasoning with broad application potential. 

**Abstract (ZH)**: Chain of Thought推理准确性的校准方法：利用模型内在真实性编码实现动态可信推理路径选择 

---
# On The Role of Intentionality in Knowledge Representation: Analyzing Scene Context for Cognitive Agents with a Tiny Language Model 

**Title (ZH)**: 意图在知识表示中的作用：使用小型语言模型分析场景上下文的认知代理研究 

**Authors**: Mark Burgess  

**Link**: [PDF](https://arxiv.org/pdf/2507.10000)  

**Abstract**: Since Searle's work deconstructing intent and intentionality in the realm of philosophy, the practical meaning of intent has received little attention in science and technology. Intentionality and context are both central to the scope of Promise Theory's model of Semantic Spacetime, used as an effective Tiny Language Model. One can identify themes and concepts from a text, on a low level (without knowledge of the specific language) by using process coherence as a guide. Any agent process can assess superficially a degree of latent `intentionality' in data by looking for anomalous multi-scale anomalies and assessing the work done to form them. Scale separation can be used to sort parts into `intended' content and `ambient context', using the spacetime coherence as a measure. This offers an elementary but pragmatic interpretation of latent intentionality for very low computational cost, and without reference to extensive training or reasoning capabilities. The process is well within the reach of basic organisms as it does not require large scale artificial probabilistic batch processing. The level of concept formation depends, however, on the memory capacity of the agent. 

**Abstract (ZH)**: 自塞尔解构意图和意向性以来，科学与技术领域对意图的实际意义关注较少。意图性和情境在《承诺理论》时空语义模型中起着核心作用，该模型可用作有效的Tiny语言模型。通过使用过程连贯性作为指导，可以在不熟悉具体语言的情况下识别文本中的主题和概念。任何代理过程都可以通过寻找多尺度异常并评估其形成工作来表面地评估数据中的潜在“意图性”。通过规模分离，可以使用时空连贯性作为度量，将部分分类为“有意向的内容”和“环境情境”。这提供了一种低计算成本的基本但实用的潜有意图解释，无需参考广泛的训练或推理能力。该过程对于基本生物来说是可行的，因为它不需要大规模的人工批量处理。然而，概念形成的程度取决于代理的内存容量。 

---
# Improving monotonic optimization in heterogeneous multi-agent reinforcement learning with optimal marginal deterministic policy gradient 

**Title (ZH)**: 在最优边际确定性策略梯度方法下，提高异构多agent reinforcement学习中的单调优化 

**Authors**: Xiaoyang Yu, Youfang Lin, Shuo Wang, Sheng Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.09989)  

**Abstract**: In heterogeneous multi-agent reinforcement learning (MARL), achieving monotonic improvement plays a pivotal role in enhancing performance. The HAPPO algorithm proposes a feasible solution by introducing a sequential update scheme, which requires independent learning with No Parameter-sharing (NoPS). However, heterogeneous MARL generally requires Partial Parameter-sharing (ParPS) based on agent grouping to achieve high cooperative performance. Our experiments prove that directly combining ParPS with the sequential update scheme leads to the policy updating baseline drift problem, thereby failing to achieve improvement. To solve the conflict between monotonic improvement and ParPS, we propose the Optimal Marginal Deterministic Policy Gradient (OMDPG) algorithm. First, we replace the sequentially computed $Q_{\psi}^s(s,a_{1:i})$ with the Optimal Marginal Q (OMQ) function $\phi_{\psi}^*(s,a_{1:i})$ derived from Q-functions. This maintains MAAD's monotonic improvement while eliminating the conflict through optimal joint action sequences instead of sequential policy ratio calculations. Second, we introduce the Generalized Q Critic (GQC) as the critic function, employing pessimistic uncertainty-constrained loss to optimize different Q-value estimations. This provides the required Q-values for OMQ computation and stable baselines for actor updates. Finally, we implement a Centralized Critic Grouped Actor (CCGA) architecture that simultaneously achieves ParPS in local policy networks and accurate global Q-function computation. Experimental results in SMAC and MAMuJoCo environments demonstrate that OMDPG outperforms various state-of-the-art MARL baselines. 

**Abstract (ZH)**: 在异构多智能体强化学习（MARL）中实现单调改进对于提升性能至关重要。HAPPO算法通过引入顺序更新方案提出了一种可行的解决方案，该方案要求无参数共享（NoPS）。然而，异构MARL通常需要根据智能体分组实现部分参数共享（ParPS）以达到高协同性能。我们的实验表明，直接将ParPS与顺序更新方案结合会导致策略更新基准漂移问题，从而无法实现改进。为了解决单调改进与ParPS之间的冲突，我们提出了最优边际确定性策略梯度（OMDPG）算法。首先，我们用从Q函数推导出的最优边际Q（OMQ）函数 $\phi_{\psi}^*(s,a_{1:i})$ 替换顺序计算的 $Q_{\psi}^s(s,a_{1:i})$，这在保持马尔可夫平均绝对差异（MAAD）单调改进的同时，通过最优联合动作序列而不是顺序策略比率计算消除了冲突。其次，我们引入广义Q评论器（GQC）作为评论器函数，使用悲观不确定性约束损失优化不同的Q值估计，为OMQ计算提供所需的Q值并为演员更新提供稳定的基线。最后，我们实现了集中评论器分组演员（CCGA）架构，该架构同时在局部策略网络中实现部分参数共享并在准确的全局Q函数计算中实现。SMAC和MAMuJoCo环境中的实验结果表明，OMDPG优于各种最新的MARL基准。 

---
# DeepSeek: Paradigm Shifts and Technical Evolution in Large AI Models 

**Title (ZH)**: DeepSeek: 大型人工智能模型中的范式转变与技术演进 

**Authors**: Luolin Xiong, Haofen Wang, Xi Chen, Lu Sheng, Yun Xiong, Jingping Liu, Yanghua Xiao, Huajun Chen, Qing-Long Han, Yang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09955)  

**Abstract**: DeepSeek, a Chinese Artificial Intelligence (AI) startup, has released their V3 and R1 series models, which attracted global attention due to their low cost, high performance, and open-source advantages. This paper begins by reviewing the evolution of large AI models focusing on paradigm shifts, the mainstream Large Language Model (LLM) paradigm, and the DeepSeek paradigm. Subsequently, the paper highlights novel algorithms introduced by DeepSeek, including Multi-head Latent Attention (MLA), Mixture-of-Experts (MoE), Multi-Token Prediction (MTP), and Group Relative Policy Optimization (GRPO). The paper then explores DeepSeek engineering breakthroughs in LLM scaling, training, inference, and system-level optimization architecture. Moreover, the impact of DeepSeek models on the competitive AI landscape is analyzed, comparing them to mainstream LLMs across various fields. Finally, the paper reflects on the insights gained from DeepSeek innovations and discusses future trends in the technical and engineering development of large AI models, particularly in data, training, and reasoning. 

**Abstract (ZH)**: DeepSeek, 一个中国的人工智能（AI）初创公司，发布了他们的V3和R1系列模型，由于其低成本、高性能和开源优势而引起了全球关注。本文首先回顾了大型AI模型的发展历程，重点关注范式变革、主流大规模语言模型（LLM）范式以及DeepSeek范式。随后，论文强调了DeepSeek引入的新型算法，包括多头潜在注意力（MLA）、专家混合（MoE）、多令牌预测（MTP）和群体相对策略优化（GRPO）。文章还探讨了DeepSeek在大规模语言模型（LLM）扩展、训练、推理和系统级优化架构方面的工程突破。此外，文章分析了DeepSeek模型对竞争激烈的AI领域的冲击，将其与其他主流LLM在各个领域进行了对比。最后，文章总结了从DeepSeek创新中获得的见解，并讨论了大型AI模型在数据、训练和推断方面的未来发展趋势。 

---
# VerifyBench: A Systematic Benchmark for Evaluating Reasoning Verifiers Across Domains 

**Title (ZH)**: VerifyBench: 一种跨领域评估推理验证器的系统性基准 

**Authors**: Xuzhao Li, Xuchen Li, Shiyu Hu, Yongzhen Guo, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09884)  

**Abstract**: Large language models (LLMs) increasingly rely on reinforcement learning (RL) to enhance their reasoning capabilities through feedback. A critical challenge is verifying the consistency of model-generated responses and reference answers, since these responses are often lengthy, diverse, and nuanced. Rule-based verifiers struggle with complexity, prompting the use of model-based verifiers. However, specialized verifiers lack flexibility, while general LLM judges can be inconsistent. Existing research primarily focuses on building better verifiers, yet a systematic evaluation of different types of verifiers' performance across domains remains lacking, severely constraining the reliable development of Reinforcement Learning with Verifiable Reward (RLVR). To address this, we propose VerifyBench--a cross-domain comprehensive benchmark for systematically evaluating verifiers. We construct 4,000 expert-level questions covering mathematics, physics, chemistry, and biology. Each question is equipped with reference answers and diverse responses. The reliability of the evaluation is ensured through a rigorous annotation process conducted by a multidisciplinary expert team. We design a four-dimensional experimental framework to comprehensively compare the performance boundaries of specialized verifiers and general LLMs under combined conditions of extracted answers vs. complete responses, and short vs. long outputs. Our evaluation uncovers fundamental trade-offs in verifiers: while specialized verifiers achieve leading accuracy, they exhibit deficiencies in recall; general models show stronger inclusivity but unstable precision. More importantly, we discover verifiers' high sensitivity to input structure and inherent limitations in cross-domain generalization, providing critical insights into the bottlenecks of current verifier technology. 

**Abstract (ZH)**: 跨领域全面基准VerifyBench：验证器的系统性评估 

---
# Model-Grounded Symbolic Artificial Intelligence Systems Learning and Reasoning with Model-Grounded Symbolic Artificial Intelligence Systems 

**Title (ZH)**: 基于模型的符号人工智能系统学习与推理 

**Authors**: Aniruddha Chattopadhyay, Raj Dandekar, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2507.09854)  

**Abstract**: Neurosymbolic artificial intelligence (AI) systems combine neural network and classical symbolic AI mechanisms to exploit the complementary strengths of large scale, generalizable learning and robust, verifiable reasoning. Numerous classifications of neurosymbolic AI illustrate how these two components can be integrated in distinctly different ways. In this work, we propose reinterpreting instruction tuned large language models as model grounded symbolic AI systems where natural language serves as the symbolic layer and grounding is achieved through the models internal representation space. Within this framework, we investigate and develop novel learning and reasoning approaches that preserve structural similarities to traditional learning and reasoning paradigms. Preliminary evaluations across axiomatic deductive reasoning procedures of varying complexity provide insights into the effectiveness of our approach in improving learning efficiency and reasoning reliability. 

**Abstract (ZH)**: 神经符号人工智能系统将神经网络和经典符号人工智能机制相结合，利用大规模、泛化学习的强大能力以及稳健、可验证推理的优势。本论文提出将指令调整的大语言模型重新解释为模型本接地符号人工智能系统，其中自然语言作为符号层，通过模型的内部表示空间实现接地。在这一框架下，我们探讨并开发了新型的学习和推理方法，这些方法在结构上保留了传统学习和推理 paradigm 的相似性。初步评估涵盖了不同复杂度的公理演绎推理过程，提供了对我们方法在提高学习效率和推理可靠性方面的有效性见解。 

---
# Is Human-Written Data Enough? The Challenge of Teaching Reasoning to LLMs Without RL or Distillation 

**Title (ZH)**: 人类撰写的数据足够吗？在无需强化学习或蒸馏的情况下教LLMs进行推理的挑战 

**Authors**: Wei Du, Branislav Kisacanin, George Armstrong, Shubham Toshniwal, Ivan Moshkov, Alexan Ayrapetyan, Sadegh Mahdavi, Dan Zhao, Shizhe Diao, Dragan Masulovic, Marius Stanean, Advaith Avadhanam, Max Wang, Ashmit Dutta, Shitij Govil, Sri Yanamandara, Mihir Tandon, Sriram Ananthakrishnan, Vedant Rathi, David Zhang, Joonseok Kang, Leon Luo, Titu Andreescu, Boris Ginsburg, Igor Gitman  

**Link**: [PDF](https://arxiv.org/pdf/2507.09850)  

**Abstract**: Reasoning-capable language models achieve state-of-the-art performance in diverse complex tasks by generating long, explicit Chain-of-Thought (CoT) traces. While recent works show that base models can acquire such reasoning traces via reinforcement learning or distillation from stronger models like DeepSeek-R1, previous works demonstrate that even short CoT prompting without fine-tuning is able to improve reasoning. We ask whether long CoT can be induced in a base model using only prompting or minimal tuning. Using just 20 long CoT examples from the reasoning model \texttt{QwQ-32B-Preview}, we lightly fine-tune the base model \texttt{Qwen2.5-32B}. The resulting model outperforms the much larger \texttt{Qwen2.5-Math-72B-Instruct}, showing that a handful of high-quality examples can unlock strong reasoning capabilities. We further explore using CoT data from non-reasoning models and human annotators, enhanced with prompt engineering, multi-pass editing, and structural guidance. However, neither matches the performance of reasoning model traces, suggesting that certain latent qualities of expert CoT are difficult to replicate. We analyze key properties of reasoning data, such as problem difficulty, diversity, and answer length, that influence reasoning distillation. While challenges remain, we are optimistic that carefully curated human-written CoT, even in small quantities, can activate reasoning behaviors in base models. We release our human-authored dataset across refinement stages and invite further investigation into what makes small-scale reasoning supervision so effective. 

**Abstract (ZH)**: 具有推理能力的语言模型通过生成长的显式推理链（CoT）痕迹，在多种复杂任务中取得了最先进的性能。我们询问仅通过提示或最少调整是否可以在基础模型中诱导出长的CoT。仅仅使用推理模型QwQ-32B-Preview的20个长CoT示例，我们轻量级微调基础模型Qwen2.5-32B。结果表明，少量高质量的示例可以解锁强大的推理能力。我们进一步探索使用非推理模型和人工标注员的CoT数据，这些数据通过提示工程、多轮编辑和结构指导进行增强，但这些仍无法匹配推理模型痕迹的表现，表明某些专家CoT的潜在品质难以复制。我们分析影响推理蒸馏的关键属性，如问题难度、多样性和答案长度。虽然仍存在挑战，但我们乐观地认为，即使在少量的情况下，精心策划的人工撰写的CoT也能激活基础模型中的推理行为。我们发布了我们的手工编写的数据集，并邀请进一步探索小规模推理监督为何如此有效。 

---
# Technical Requirements for Halting Dangerous AI Activities 

**Title (ZH)**: Technical Requirements for Stopping Dangerous AI Activities 

**Authors**: Peter Barnett, Aaron Scher, David Abecassis  

**Link**: [PDF](https://arxiv.org/pdf/2507.09801)  

**Abstract**: The rapid development of AI systems poses unprecedented risks, including loss of control, misuse, geopolitical instability, and concentration of power. To navigate these risks and avoid worst-case outcomes, governments may proactively establish the capability for a coordinated halt on dangerous AI development and deployment. In this paper, we outline key technical interventions that could allow for a coordinated halt on dangerous AI activities. We discuss how these interventions may contribute to restricting various dangerous AI activities, and show how these interventions can form the technical foundation for potential AI governance plans. 

**Abstract (ZH)**: AI系统rapid发展带来的前所未有的风险，包括失控风险、误用风险、地缘政治不稳定性和权力集中。为应对这些风险并避免最糟糕的结果，政府可能需要主动建立协调停止单一危险AI研发和部署的能力。在本文中，我们概述了关键的技术干预措施，这些措施可以实现对危险AI活动的协调停止。我们讨论了这些干预措施如何限制各种危险AI活动，并展示了这些干预措施如何成为潜在AI治理计划的技术基础。 

---
# Sound and Complete Neuro-symbolic Reasoning with LLM-Grounded Interpretations 

**Title (ZH)**: 基于LLM支持的解释的声学完备神经符号推理 

**Authors**: Bradley P. Allen, Prateek Chhikara, Thomas Macaulay Ferguson, Filip Ilievski, Paul Groth  

**Link**: [PDF](https://arxiv.org/pdf/2507.09751)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, but they exhibit problems with logical consistency in the output they generate. How can we harness LLMs' broad-coverage parametric knowledge in formal reasoning despite their inconsistency? We present a method for directly integrating an LLM into the interpretation function of the formal semantics for a paraconsistent logic. We provide experimental evidence for the feasibility of the method by evaluating the function using datasets created from several short-form factuality benchmarks. Unlike prior work, our method offers a theoretical framework for neuro-symbolic reasoning that leverages an LLM's knowledge while preserving the underlying logic's soundness and completeness properties. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言理解和生成方面展现了令人印象深刻的能力，但在输出过程中表现出逻辑一致性问题。我们如何在保留其不一致性的情况下利用LLMs广泛覆盖的参数知识进行形式推理？我们提出了一种直接将LLM集成到paraconsistent逻辑的形式语义解释函数中的方法。通过使用来自多个简短形式事实基准的数据集评估该函数的功能，我们提供了该方法可行性的实验证据。与先前的工作不同，我们的方法提供了一种理论框架，能够在利用LLM知识的同时保持底层逻辑的有效性和完满性。 

---
# Causality-informed Anomaly Detection in Partially Observable Sensor Networks: Moving beyond Correlations 

**Title (ZH)**: 基于因果关系的部分可观测传感器网络异常检测：超越相关性 

**Authors**: Xiaofeng Xiao, Bo Shen, Xubo Yue  

**Link**: [PDF](https://arxiv.org/pdf/2507.09742)  

**Abstract**: Nowadays, as AI-driven manufacturing becomes increasingly popular, the volume of data streams requiring real-time monitoring continues to grow. However, due to limited resources, it is impractical to place sensors at every location to detect unexpected shifts. Therefore, it is necessary to develop an optimal sensor placement strategy that enables partial observability of the system while detecting anomalies as quickly as possible. Numerous approaches have been proposed to address this challenge; however, most existing methods consider only variable correlations and neglect a crucial factor: Causality. Moreover, although a few techniques incorporate causal analysis, they rely on interventions-artificially creating anomalies-to identify causal effects, which is impractical and might lead to catastrophic losses. In this paper, we introduce a causality-informed deep Q-network (Causal DQ) approach for partially observable sensor placement in anomaly detection. By integrating causal information at each stage of Q-network training, our method achieves faster convergence and tighter theoretical error bounds. Furthermore, the trained causal-informed Q-network significantly reduces the detection time for anomalies under various settings, demonstrating its effectiveness for sensor placement in large-scale, real-world data streams. Beyond the current implementation, our technique's fundamental insights can be applied to various reinforcement learning problems, opening up new possibilities for real-world causality-informed machine learning methods in engineering applications. 

**Abstract (ZH)**: 基于因果信息的深度Q网络在异常检测中的部分可观测传感器布置方法 

---
# Towards Concise and Adaptive Thinking in Large Reasoning Models: A Survey 

**Title (ZH)**: 大型推理模型中精炼且适应性强的思考趋向：一种综述 

**Authors**: Jason Zhu, Hongyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.09662)  

**Abstract**: Large reasoning models (LRMs) like OpenAI o1 and DeepSeek R1 have demonstrated impressive performance on complex reasoning tasks like mathematics and programming with long Chain-of-Thought (CoT) reasoning sequences (slow-thinking), compared with traditional large language models (fast-thinking). However, these reasoning models also face a huge challenge that generating unnecessarily lengthy and redundant reasoning chains even for trivial questions. This phenomenon leads to a significant waste of inference resources, increases the response time for simple queries, and hinders the practical application of LRMs in real-world products. To this end, it is crucial to shorten lengthy reasoning chains and learn adaptive reasoning between fast and slow thinking based on input difficulty. In this survey, we provide a comprehensive overview of recent progress in concise and adaptive thinking for efficient reasoning of LRMs, including methodologies, benchmarks, and challenges for future exploration. We hope this survey can help researchers quickly understand the landscape of this field and inspire novel adaptive thinking ideas to facilitate better usage of LRMs. 

**Abstract (ZH)**: 大型推理模型（LRMs）如OpenAI的o1和DeepSeek的R1在数学和编程等复杂推理任务中展现了 impressive 的表现，尤其是在长串推理链条（慢思考）方面，相比之下，传统的大型语言模型则表现较为迅速。然而，这些推理模型也面临着一个巨大挑战，即即使对于简单的问题也会生成不必要的冗长和重复的推理链条。这种现象导致了大量的推理资源浪费，增加了简单查询的响应时间，并阻碍了LRMs在实际产品中的应用。为此，缩短冗长的推理链条并根据输入难度学习适应性推理（结合快思考和慢思考）是至关重要的。在本文综述中，我们提供了关于简化和适应性推理的最新进展的全面概述，包括方法学、基准测试以及未来探索面临的挑战。我们希望本文综述能帮助研究者迅速理解该领域的现状，并激发新的适应性思考理念，以促进LRMs更好地使用。 

---
# humancompatible.interconnect: Testing Properties of Repeated Uses of Interconnections of AI Systems 

**Title (ZH)**: humancompatible.interconnect: 检测AI系统互联重复使用属性的研究 

**Authors**: Rodion Nazarov, Anthony Quinn, Robert Shorten, Jakub Marecek  

**Link**: [PDF](https://arxiv.org/pdf/2507.09626)  

**Abstract**: Artificial intelligence (AI) systems often interact with multiple agents. The regulation of such AI systems often requires that {\em a priori\/} guarantees of fairness and robustness be satisfied. With stochastic models of agents' responses to the outputs of AI systems, such {\em a priori\/} guarantees require non-trivial reasoning about the corresponding stochastic systems. Here, we present an open-source PyTorch-based toolkit for the use of stochastic control techniques in modelling interconnections of AI systems and properties of their repeated uses. It models robustness and fairness desiderata in a closed-loop fashion, and provides {\em a priori\/} guarantees for these interconnections. The PyTorch-based toolkit removes much of the complexity associated with the provision of fairness guarantees for closed-loop models of multi-agent systems. 

**Abstract (ZH)**: 人工智能系统 often 与多个代理交互。对于此类人工智能系统，监管通常要求事先保证公平性和鲁棒性。通过代理对人工智能系统输出的随机响应模型，这些事先保证需要对相应的随机系统进行非平凡推理。在此，我们提出一个基于 PyTorch 的开源工具包，用于使用随机控制技术建模人工智能系统的相互连接及其重复使用属性。该工具包以闭环方式建模鲁棒性和公平性需求，并为这些相互连接提供事先保证。基于 PyTorch 的工具包消除了为多代理系统的闭环模型提供公平性保证的相关复杂性。 

---
# Bridging Bots: from Perception to Action via Multimodal-LMs and Knowledge Graphs 

**Title (ZH)**: 跨足机器人：从感知到行动的多模态语言模型与知识图谱桥梁 

**Authors**: Margherita Martorana, Francesca Urgese, Mark Adamik, Ilaria Tiddi  

**Link**: [PDF](https://arxiv.org/pdf/2507.09617)  

**Abstract**: Personal service robots are deployed to support daily living in domestic environments, particularly for elderly and individuals requiring assistance. These robots must perceive complex and dynamic surroundings, understand tasks, and execute context-appropriate actions. However, current systems rely on proprietary, hard-coded solutions tied to specific hardware and software, resulting in siloed implementations that are difficult to adapt and scale across platforms. Ontologies and Knowledge Graphs (KGs) offer a solution to enable interoperability across systems, through structured and standardized representations of knowledge and reasoning. However, symbolic systems such as KGs and ontologies struggle with raw and noisy sensory input. In contrast, multimodal language models are well suited for interpreting input such as images and natural language, but often lack transparency, consistency, and knowledge grounding. In this work, we propose a neurosymbolic framework that combines the perceptual strengths of multimodal language models with the structured representations provided by KGs and ontologies, with the aim of supporting interoperability in robotic applications. Our approach generates ontology-compliant KGs that can inform robot behavior in a platform-independent manner. We evaluated this framework by integrating robot perception data, ontologies, and five multimodal models (three LLaMA and two GPT models), using different modes of neural-symbolic interaction. We assess the consistency and effectiveness of the generated KGs across multiple runs and configurations, and perform statistical analyzes to evaluate performance. Results show that GPT-o1 and LLaMA 4 Maverick consistently outperform other models. However, our findings also indicate that newer models do not guarantee better results, highlighting the critical role of the integration strategy in generating ontology-compliant KGs. 

**Abstract (ZH)**: 基于神经符号框架的多模态语言模型与知识图谱及本体的结合在机器人应用中的互操作性支持 

---
# The Hidden Costs of AI: A Review of Energy, E-Waste, and Inequality in Model Development 

**Title (ZH)**: AI隐性成本：模型开发中的能源、电子废物与不平等回顾 

**Authors**: Jenis Winsta  

**Link**: [PDF](https://arxiv.org/pdf/2507.09611)  

**Abstract**: Artificial intelligence (AI) has made remarkable progress in recent years, yet its rapid expansion brings overlooked environmental and ethical challenges. This review explores four critical areas where AI's impact extends beyond performance: energy consumption, electronic waste (e-waste), inequality in compute access, and the hidden energy burden of cybersecurity systems. Drawing from recent studies and institutional reports, the paper highlights systemic issues such as high emissions from model training, rising hardware turnover, global infrastructure disparities, and the energy demands of securing AI. By connecting these concerns, the review contributes to Responsible AI discourse by identifying key research gaps and advocating for sustainable, transparent, and equitable development practices. Ultimately, it argues that AI's progress must align with ethical responsibility and environmental stewardship to ensure a more inclusive and sustainable technological future. 

**Abstract (ZH)**: 人工智能（AI）在近年来取得了显著进步，但其快速扩张带来了未被忽视的环境和伦理挑战。本文综述了AI影响超越性能的四个关键领域：能源消耗、电子废物（e-waste）、计算资源获取不平以及网络安全系统的隐含能源负担。通过参考近期的研究和机构报告，文章指出了系统性问题，如模型训练的高排放、硬件更新频率上升、全球基础设施差距以及AI安全的能源需求。通过将这些担忧联系起来，本文综述为负责任的AI讨论做出了贡献，识别了关键研究缺口，并倡导可持续、透明和公平的发展实践。最终，本文认为AI的发展必须与伦理责任和环境 stewardship 相匹配，以确保一个更加包容和可持续的技术未来。 

---
# eSapiens: A Platform for Secure and Auditable Retrieval-Augmented Generation 

**Title (ZH)**: eSapiens：一个安全可审计的检索增强生成平台 

**Authors**: Isaac Shi, Zeyuan Li, Fan Liu, Wenli Wang, Lewei He, Yang Yang, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.09588)  

**Abstract**: We present eSapiens, an AI-as-a-Service (AIaaS) platform engineered around a business-oriented trifecta: proprietary data, operational workflows, and any major agnostic Large Language Model (LLM). eSapiens gives businesses full control over their AI assets, keeping everything in-house for AI knowledge retention and data security. eSapiens AI Agents (Sapiens) empower your team by providing valuable insights and automating repetitive tasks, enabling them to focus on high-impact work and drive better business outcomes.
The system integrates structured document ingestion, hybrid vector retrieval, and no-code orchestration via LangChain, and supports top LLMs including OpenAI, Claude, Gemini, and DeepSeek. A key component is the THOR Agent, which handles structured SQL-style queries and generates actionable insights over enterprise databases.
To evaluate the system, we conduct two experiments. First, a retrieval benchmark on legal corpora reveals that a chunk size of 512 tokens yields the highest retrieval precision (Top-3 accuracy: 91.3%). Second, a generation quality test using TRACe metrics across five LLMs shows that eSapiens delivers more context-consistent outputs with up to a 23% improvement in factual alignment.
These results demonstrate the effectiveness of eSapiens in enabling trustworthy, auditable AI workflows for high-stakes domains like legal and finance. 

**Abstract (ZH)**: eSapiens：面向企业的AI即服务平台及其在高 stakes 领域中的应用 

---
# Learning to Control Dynamical Agents via Spiking Neural Networks and Metropolis-Hastings Sampling 

**Title (ZH)**: 通过尖峰神经网络和梅特罗波利斯-哈斯特斯采样学习控制动力学代理 

**Authors**: Ali Safa, Farida Mohsen, Ali Al-Zawqari  

**Link**: [PDF](https://arxiv.org/pdf/2507.09540)  

**Abstract**: Spiking Neural Networks (SNNs) offer biologically inspired, energy-efficient alternatives to traditional Deep Neural Networks (DNNs) for real-time control systems. However, their training presents several challenges, particularly for reinforcement learning (RL) tasks, due to the non-differentiable nature of spike-based communication. In this work, we introduce what is, to our knowledge, the first framework that employs Metropolis-Hastings (MH) sampling, a Bayesian inference technique, to train SNNs for dynamical agent control in RL environments without relying on gradient-based methods. Our approach iteratively proposes and probabilistically accepts network parameter updates based on accumulated reward signals, effectively circumventing the limitations of backpropagation while enabling direct optimization on neuromorphic platforms. We evaluated this framework on two standard control benchmarks: AcroBot and CartPole. The results demonstrate that our MH-based approach outperforms conventional Deep Q-Learning (DQL) baselines and prior SNN-based RL approaches in terms of maximizing the accumulated reward while minimizing network resources and training episodes. 

**Abstract (ZH)**: 基于Metropolis-Hastings采样的Spiking神经网络在强化学习中的动态代理控制方法 

---
# Consistency Trajectory Planning: High-Quality and Efficient Trajectory Optimization for Offline Model-Based Reinforcement Learning 

**Title (ZH)**: 一致性轨迹规划：离线模型基于强化学习的高质量与高效轨迹优化 

**Authors**: Guanquan Wang, Takuya Hiraoka, Yoshimasa Tsuruoka  

**Link**: [PDF](https://arxiv.org/pdf/2507.09534)  

**Abstract**: This paper introduces Consistency Trajectory Planning (CTP), a novel offline model-based reinforcement learning method that leverages the recently proposed Consistency Trajectory Model (CTM) for efficient trajectory optimization. While prior work applying diffusion models to planning has demonstrated strong performance, it often suffers from high computational costs due to iterative sampling procedures. CTP supports fast, single-step trajectory generation without significant degradation in policy quality. We evaluate CTP on the D4RL benchmark and show that it consistently outperforms existing diffusion-based planning methods in long-horizon, goal-conditioned tasks. Notably, CTP achieves higher normalized returns while using significantly fewer denoising steps. In particular, CTP achieves comparable performance with over $120\times$ speedup in inference time, demonstrating its practicality and effectiveness for high-performance, low-latency offline planning. 

**Abstract (ZH)**: 基于一致性轨迹模型的一致性轨迹规划方法（CTP）：高效的 Offline 强化学习轨迹优化方法 

---
# GenAI-based Multi-Agent Reinforcement Learning towards Distributed Agent Intelligence: A Generative-RL Agent Perspective 

**Title (ZH)**: 基于GenAI的多智能体强化学习 toward 分布式智能体智能：生成式-RL智能体视角 

**Authors**: Hang Wang, Junshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09495)  

**Abstract**: Multi-agent reinforcement learning faces fundamental challenges that conventional approaches have failed to overcome: exponentially growing joint action spaces, non-stationary environments where simultaneous learning creates moving targets, and partial observability that constrains coordination. Current methods remain reactive, employing stimulus-response mechanisms that fail when facing novel scenarios. We argue for a transformative paradigm shift from reactive to proactive multi-agent intelligence through generative AI-based reinforcement learning. This position advocates reconceptualizing agents not as isolated policy optimizers, but as sophisticated generative models capable of synthesizing complex multi-agent dynamics and making anticipatory decisions based on predictive understanding of future interactions. Rather than responding to immediate observations, generative-RL agents can model environment evolution, predict other agents' behaviors, generate coordinated action sequences, and engage in strategic reasoning accounting for long-term dynamics. This approach leverages pattern recognition and generation capabilities of generative AI to enable proactive decision-making, seamless coordination through enhanced communication, and dynamic adaptation to evolving scenarios. We envision this paradigm shift will unlock unprecedented possibilities for distributed intelligence, moving beyond individual optimization toward emergent collective behaviors representing genuine collaborative intelligence. The implications extend across autonomous systems, robotics, and human-AI collaboration, promising solutions to coordination challenges intractable under traditional reactive frameworks. 

**Abstract (ZH)**: 基于生成AI的强化学习：多代理智能从被动到主动的范式转型 

---
# LLM-Stackelberg Games: Conjectural Reasoning Equilibria and Their Applications to Spearphishing 

**Title (ZH)**: LLM-斯塔克尔贝格博弈：猜测性推理 equilibrium 及其针对 Spearphishing 的应用 

**Authors**: Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09407)  

**Abstract**: We introduce the framework of LLM-Stackelberg games, a class of sequential decision-making models that integrate large language models (LLMs) into strategic interactions between a leader and a follower. Departing from classical Stackelberg assumptions of complete information and rational agents, our formulation allows each agent to reason through structured prompts, generate probabilistic behaviors via LLMs, and adapt their strategies through internal cognition and belief updates. We define two equilibrium concepts: reasoning and behavioral equilibrium, which aligns an agent's internal prompt-based reasoning with observable behavior, and conjectural reasoning equilibrium, which accounts for epistemic uncertainty through parameterized models over an opponent's response. These layered constructs capture bounded rationality, asymmetric information, and meta-cognitive adaptation. We illustrate the framework through a spearphishing case study, where a sender and a recipient engage in a deception game using structured reasoning prompts. This example highlights the cognitive richness and adversarial potential of LLM-mediated interactions. Our results show that LLM-Stackelberg games provide a powerful paradigm for modeling decision-making in domains such as cybersecurity, misinformation, and recommendation systems. 

**Abstract (ZH)**: LLM-Stackelberg博弈框架：一种将大型语言模型融入领导者与追随者战略互动中的 sequential 决策模型 

---
# Knowledge Conceptualization Impacts RAG Efficacy 

**Title (ZH)**: 知识概念化影响RAG效果 

**Authors**: Chris Davis Jaldi, Anmol Saini, Elham Ghiasi, O. Divine Eziolise, Cogan Shimizu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09389)  

**Abstract**: Explainability and interpretability are cornerstones of frontier and next-generation artificial intelligence (AI) systems. This is especially true in recent systems, such as large language models (LLMs), and more broadly, generative AI. On the other hand, adaptability to new domains, contexts, or scenarios is also an important aspect for a successful system. As such, we are particularly interested in how we can merge these two efforts, that is, investigating the design of transferable and interpretable neurosymbolic AI systems. Specifically, we focus on a class of systems referred to as ''Agentic Retrieval-Augmented Generation'' systems, which actively select, interpret, and query knowledge sources in response to natural language prompts. In this paper, we systematically evaluate how different conceptualizations and representations of knowledge, particularly the structure and complexity, impact an AI agent (in this case, an LLM) in effectively querying a triplestore. We report our results, which show that there are impacts from both approaches, and we discuss their impact and implications. 

**Abstract (ZH)**: 可解释性和可解析性是前沿和下一代人工智能系统的核心。特别是在大型语言模型和更广泛的生成人工智能中，这一点尤为 true。另一方面，系统对新领域、情境或场景的适应能力也是成功系统的重要方面。因此，我们特别关注如何将这两方面结合起来，即研究可迁移和可解析的神经符号人工智能系统的设计。具体而言，我们集中在一类被称为“自主检索增强生成”系统的类别上，这类系统能够主动选择、解析和查询知识源以响应自然语言提示。在本文中，我们系统地评估了不同类型的知识概念化和表示，特别是结构和复杂性，对人工智能代理（在此情况下为大型语言模型）有效查询三元组存储的影响。我们报告了研究结果，表明这两种方法都会产生影响，并讨论了它们的影响和意义。 

---
# EduFlow: Advancing MLLMs' Problem-Solving Proficiency through Multi-Stage, Multi-Perspective Critique 

**Title (ZH)**: EduFlow: 通过多阶段、多视角批评提高MLLMs的解决问题能力 

**Authors**: Chenglin Zhu, Tao Zhang, Chong Li, Mingan Lin, Zenan Zhou, Jian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.09374)  

**Abstract**: Multimodal large language models (MLLMs) still perform poorly on scientific tasks, particularly those requiring multi-step and interpretable reasoning. Their limitations include insufficient scientific reasoning patterns, lack of global coherence in multi-step inference, and the absence of reflective self-correction, making them unreliable in structured scientific contexts. We introduce EduFlow, the first end-to-end framework that covers the full pipeline of educational scientific reasoning, including data selection, MCTS-based trajectory construction, model training, and output optimization. At its core is EduPRM, a process-aware reward model that critiques reasoning steps with tags and justifications. EduPRM is trained via curriculum learning on three complementary supervision sources: MCTS-guided trajectories, error-injected critiques, and teacher-student dialogues, enabling dynamic adaptation to multi-stage problem solving and iterative refinement during inference. We further propose EduMCTS, a domain-adapted search framework that introduces bootstrapping actions specifically designed for educational reasoning, such as a self-reflection mechanism that promotes reflective error correction. It further leverages EduPRM's fine-grained feedback to guide the search toward higher-quality reasoning trajectories. By applying self-consistency and rejection sampling, we constructed EduMCTS-160K, a large-scale dataset of educational reasoning trajectories. Extensive experiments demonstrate that EduFlow enhances reasoning consistency and coherence. Code, data, and models will be released. 

**Abstract (ZH)**: 多重模态大型语言模型在科学任务中表现仍然不佳，尤其是在需要多步和可解释推理的任务中。它们的局限性包括缺乏科学推理模式、多步推理缺乏全局连贯性和缺乏反思性自我纠正能力，这使得它们在结构化的科学场景中不可靠。我们引入了EduFlow，这是第一个覆盖教育科学推理全流程的端到端框架，包括数据选择、基于MCTS的轨迹构建、模型训练和输出优化。其核心是EduPRM，一种过程感知的奖励模型，能够通过标签和解释对推理步骤进行评价。EduPRM通过课程学习在三种互补监督源的指导下进行训练，包括MCTS引导的轨迹、错误注入的评价以及师生对话，这使它能够针对多阶段问题解决进行动态适应并促进推理的迭代优化。我们还提出了EduMCTS，一种领域适应的搜索框架，引入了专为教育推理设计的自举动作，如自我反思机制，以促进反思性错误纠正，并进一步利用EduPRM的精细反馈来引导搜索向高质量的推理轨迹发展。通过使用自一致性校验和拒绝采样，我们构建了EduMCTS-160K，这是一个大规模的教育推理轨迹数据集。广泛的实验证明，EduFlow能够增强推理的一致性和连贯性。代码、数据和模型将公开发布。 

---
# A Taxonomy of Omnicidal Futures Involving Artificial Intelligence 

**Title (ZH)**: 人工智能涉及的万劫不复的未来分类 

**Authors**: Andrew Critch, Jacob Tsimerman  

**Link**: [PDF](https://arxiv.org/pdf/2507.09369)  

**Abstract**: This report presents a taxonomy and examples of potential omnicidal events resulting from AI: scenarios where all or almost all humans are killed. These events are not presented as inevitable, but as possibilities that we can work to avoid. Insofar as large institutions require a degree of public support in order to take certain actions, we hope that by presenting these possibilities in public, we can help to support preventive measures against catastrophic risks from AI. 

**Abstract (ZH)**: 本报告提出了一种关于由AI引发的潜在全人类灭绝事件的分类及其例子：描述了可能导致所有或几乎所有人被杀的情景。这些事件并不被视为不可避免，而是我们可以通过努力来避免的可能性。鉴于大型机构需要一定程度的公众支持才能采取某些行动，我们希望通过在公共场合呈现这些可能性，来支持防止AI带来的灾难性风险的预防措施。 

---
# When Developer Aid Becomes Security Debt: A Systematic Analysis of Insecure Behaviors in LLM Coding Agents 

**Title (ZH)**: 当开发者援助演变成安全债务：大规模语言模型编码代理中的不安全行为系统分析 

**Authors**: Matous Kozak, Roshanak Zilouchian Moghaddam, Siva Sivaraman  

**Link**: [PDF](https://arxiv.org/pdf/2507.09329)  

**Abstract**: LLM-based coding agents are rapidly being deployed in software development, yet their security implications remain poorly understood. These agents, while capable of accelerating software development, may inadvertently introduce insecure practices. We conducted the first systematic security evaluation of autonomous coding agents, analyzing over 12,000 actions across five state-of-the-art models (GPT-4o, GPT-4.1, Claude variants) on 93 real-world software setup tasks. Our findings reveal significant security concerns: 21% of agent trajectories contained insecure actions, with models showing substantial variation in security behavior. We developed a high-precision detection system that identified four major vulnerability categories, with information exposure (CWE-200) being the most prevalent one. We also evaluated mitigation strategies including feedback mechanisms and security reminders with various effectiveness between models. GPT-4.1 demonstrated exceptional security awareness with 96.8% mitigation success. Our work provides the first comprehensive framework for evaluating coding agent security and highlights the need for security-aware design of next generation LLM-based coding agents. 

**Abstract (ZH)**: 基于大规模语言模型的编码代理在软件开发中的安全影响亟待深入理解：首次系统性安全评估研究 

---
# Hide-and-Shill: A Reinforcement Learning Framework for Market Manipulation Detection in Symphony-a Decentralized Multi-Agent System 

**Title (ZH)**: 隐匿与诈欺：Symphony去中心化多智能体系统中的市场操纵检测强化学习框架 

**Authors**: Ronghua Shi, Yiou Liu, Xinyu Ying, Yang Tan, Yuchun Feng, Lynn Ai, Bill Shi, Xuhui Wang, Zhuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09179)  

**Abstract**: Decentralized finance (DeFi) has introduced a new era of permissionless financial innovation but also led to unprecedented market manipulation. Without centralized oversight, malicious actors coordinate shilling campaigns and pump-and-dump schemes across various platforms. We propose a Multi-Agent Reinforcement Learning (MARL) framework for decentralized manipulation detection, modeling the interaction between manipulators and detectors as a dynamic adversarial game. This framework identifies suspicious patterns using delayed token price reactions as financial this http URL method introduces three innovations: (1) Group Relative Policy Optimization (GRPO) to enhance learning stability in sparse-reward and partially observable settings; (2) a theory-based reward function inspired by rational expectations and information asymmetry, differentiating price discovery from manipulation noise; and (3) a multi-modal agent pipeline that integrates LLM-based semantic features, social graph signals, and on-chain market data for informed this http URL framework is integrated within the Symphony system, a decentralized multi-agent architecture enabling peer-to-peer agent execution and trust-aware learning through distributed logs, supporting chain-verifiable evaluation. Symphony promotes adversarial co-evolution among strategic actors and maintains robust manipulation detection without centralized oracles, enabling real-time surveillance across global DeFi this http URL on 100,000 real-world discourse episodes and validated in adversarial simulations, Hide-and-Shill achieves top performance in detection accuracy and causal attribution. This work bridges multi-agent systems with financial surveillance, advancing a new paradigm for decentralized market intelligence. All resources are available at the Hide-and-Shill GitHub repository to promote open research and reproducibility. 

**Abstract (ZH)**: 去中心化金融（DeFi）引入了新的去许可金融创新时代，但也导致了前所未有的市场操纵。没有中心化的监督，恶意行为者协调跨各种平台的拉抬打压和炒专辑活动。我们提出了一种多智能体强化学习（MARL）框架，用于去中心化的操纵检测，将操纵者和检测者之间的互动建模为动态的对抗性博弈。该框架利用延迟的代币价格反应识别可疑模式，方法引入了三项创新：（1）组相对策略优化（GRPO）以增强在稀疏奖励和部分可观测环境中的学习稳定性；（2）基于理论的奖励函数，受理性预期和信息不对称的启发，区分价格发现与操纵噪声；（3）多模态智能体流水线，结合基于LLM的语义特征、社会图信号和链上市场数据，以进行有信息量的检测。该框架嵌入在Symphony系统中，这是一种去中心化的多智能体架构，通过分布式日志支持节点间的智能体执行和信任感知学习，并通过链验证方式进行评估。Symphony促进了战略行为者之间的对抗共生，并在没有集中式预言机的情况下保持了强大的操纵检测能力，实现了全球DeFi市场的实时监控。该工作将多智能体系统与金融监督相结合，推进了一种新的去中心化市场智能新范式。所有资源均可在Hide-and-Shill GitHub仓库中获取，以促进开放研究和可再现性。 

---
# Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity 

**Title (ZH)**: 衡量2025年前AI对资深开源开发者 productivity 的影响 

**Authors**: Joel Becker, Nate Rush, Elizabeth Barnes, David Rein  

**Link**: [PDF](https://arxiv.org/pdf/2507.09089)  

**Abstract**: Despite widespread adoption, the impact of AI tools on software development in the wild remains understudied. We conduct a randomized controlled trial (RCT) to understand how AI tools at the February-June 2025 frontier affect the productivity of experienced open-source developers. 16 developers with moderate AI experience complete 246 tasks in mature projects on which they have an average of 5 years of prior experience. Each task is randomly assigned to allow or disallow usage of early 2025 AI tools. When AI tools are allowed, developers primarily use Cursor Pro, a popular code editor, and Claude 3.5/3.7 Sonnet. Before starting tasks, developers forecast that allowing AI will reduce completion time by 24%. After completing the study, developers estimate that allowing AI reduced completion time by 20%. Surprisingly, we find that allowing AI actually increases completion time by 19%--AI tooling slowed developers down. This slowdown also contradicts predictions from experts in economics (39% shorter) and ML (38% shorter). To understand this result, we collect and evaluate evidence for 20 properties of our setting that a priori could contribute to the observed slowdown effect--for example, the size and quality standards of projects, or prior developer experience with AI tooling. Although the influence of experimental artifacts cannot be entirely ruled out, the robustness of the slowdown effect across our analyses suggests it is unlikely to primarily be a function of our experimental design. 

**Abstract (ZH)**: 尽管AI工具已被广泛采用，但它们在实际软件开发中的影响仍研究不足。我们开展一项随机对照试验（RCT），以了解2025年2月至6月前沿的AI工具如何影响有经验的开源开发者的工作效率。16名拥有中等AI经验的开发者在成熟项目中完成了246项任务，这些项目他们平均已有5年的开发经验。每项任务均随机分配，允许或不允许使用早期2025年的AI工具。当允许使用AI工具时，开发者主要使用流行的代码编辑器Cursor Pro，以及Claude 3.5/3.7 Sonnet。在开始任务前，开发者预测允许使用AI可将完成时间减少24%。完成研究后，开发者估计允许使用AI将完成时间减少了20%。令人惊讶的是，我们发现允许使用AI实际上将完成时间延长了19%——AI工具反而减慢了开发者的速度。这种减慢也与经济学专家（缩短39%）和机器学习专家（缩短38%）的预测不符。为了理解这一结果，我们收集并评估了可能对观察到的减慢效果有贡献的20种设置属性的证据，例如项目的规模和质量标准，或开发者先前使用AI工具的经验。虽然无法完全排除实验 artefacts 的影响，但我们在分析中发现的减慢效果的稳健性表明，这不太可能是由于实验设计的功能。 

---
# BioAnalyst: A Foundation Model for Biodiversity 

**Title (ZH)**: BioAnalyst: 生物多样性基础模型 

**Authors**: Athanasios Trantas, Martino Mensio, Stylianos Stasinos, Sebastian Gribincea, Taimur Khan, Damian Podareanu, Aliene van der Veen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09080)  

**Abstract**: The accelerating loss of biodiversity presents critical challenges for ecological research and conservation strategies. The preservation of biodiversity is paramount for maintaining ecological balance and ensuring the sustainability of ecosystems. However, biodiversity faces numerous threats, including habitat loss, climate change, and the proliferation of invasive species. Addressing these and other ecology-related challenges, both at local and global scales, requires comprehensive monitoring, predictive and conservation planning capabilities. Artificial Intelligence (AI) Foundation Models (FMs) have gained significant momentum in numerous scientific domains by leveraging vast datasets to learn general-purpose representations adaptable to various downstream tasks. This paradigm holds immense promise for biodiversity conservation. In response, we introduce BioAnalyst, the first Foundation Model tailored for biodiversity analysis and conservation planning. BioAnalyst employs a transformer-based architecture, pre-trained on extensive multi-modal datasets encompassing species occurrence records, remote sensing indicators, climate and environmental variables. BioAnalyst is designed for adaptability, allowing for fine-tuning of a range of downstream tasks, such as species distribution modelling, habitat suitability assessments, invasive species detection, and population trend forecasting. We evaluate the model's performance on two downstream use cases, demonstrating its generalisability compared to existing methods, particularly in data-scarce scenarios for two distinct use-cases, establishing a new accuracy baseline for ecological forecasting. By openly releasing BioAnalyst and its fine-tuning workflows to the scientific community, we aim to foster collaborative efforts in biodiversity modelling and advance AI-driven solutions to pressing ecological challenges. 

**Abstract (ZH)**: 生物多样性的加速丧失对生态研究和保护策略提出了关键性挑战。生物多样性的维持对于维持生态平衡和确保生态系统的可持续性至关重要。然而，生物多样性面临着诸多威胁，包括栖息地丧失、气候变化和入侵物种的扩散。针对这些及其它生态相关挑战，无论是局部还是全球尺度，都要求具备全面监测、预测和保护规划的能力。通过利用大量数据集来学习适应各种下游任务的泛化表示，人工智能基础模型（AI Foundation Models, FMs）在诸多科学领域获得了显著进展。这一范式为生物多样性保护提供了巨大的潜力。为此，我们引入了BioAnalyst，这是首个专门针对生物多样性分析和保护规划的基础模型。BioAnalyst采用基于变换器的架构，预训练于包含物种分布记录、遥感指标、气候和环境变量的多模态大数据集上。BioAnalyst设计灵活，允许对多种下游任务进行微调，例如物种分布建模、栖息地适宜性评估、入侵物种检测和种群趋势预测。我们对模型进行了两个下游应用场景的评估，展示了其在数据稀缺条件下的一般适用性，特别是在两个不同的应用场景中建立了生态预测的新准确率基线。通过公开发布BioAnalyst及其微调工作流程，我们旨在促进生物多样性建模的协作努力，并推动以人工智能驱动的解决方案应对紧迫的生态挑战。 

---
# Multi-Actor Generative Artificial Intelligence as a Game Engine 

**Title (ZH)**: 多行为体生成人工智能作为游戏引擎 

**Authors**: Alexander Sasha Vezhnevets, Jayd Matyas, Logan Cross, Davide Paglieri, Minsuk Chang, William A. Cunningham, Simon Osindero, William S. Isaac, Joel Z. Leibo  

**Link**: [PDF](https://arxiv.org/pdf/2507.08892)  

**Abstract**: Generative AI can be used in multi-actor environments with purposes ranging from social science modeling to interactive narrative and AI evaluation. Supporting this diversity of use cases -- which we classify as Simulationist, Dramatist, and Evaluationist -- demands a flexible scenario definition framework. We argue here that a good approach is to take inspiration from tabletop role-playing games (TTRPGs), where a Game Master (GM) is responsible for the environment and generates all parts of the story not directly determined by the voluntary actions of player characters. We argue that the Entity-Component architectural pattern is useful here. In such a system, the GM is not a hardcoded computer game but is itself a configurable entity, composed of components just like any other actor. By design, the approach allows for a separation between the underlying implementation details handled by an engineer, the creation of reusable components, and their composition and configuration managed by a designer who constructs entities from the components. This separation of concerns is instrumental for achieving rapid iteration, maintaining modularity, and ultimately to ensure scalability. We describe the ongoing evolution of the Concordia library in terms of this philosophy, demonstrating how it allows users to effectively configure scenarios that align with their specific goals. 

**Abstract (ZH)**: 生成式AI可以用于多 actors 环境，用途涵盖社会科学建模、互动叙事和AI评估等。为了支持这一多样性用途——我们将其分类为模拟主义、戏剧主义和评价主义——需要一个灵活的场景定义框架。我们认为从中桌游（TTRPG）抽取灵感的一种方法是有效的。在游戏中，主持人（GM）负责环境并生成所有不由玩家角色自愿行为直接决定的故事部分。我们认为实体-组件架构模式在此很有用。在这种系统中，GM 不是一个硬编码的计算机游戏，而是自己就是一个可配置的实体，与任何其他演员一样由组件构成。这一设计方法允许工程师处理底层实现细节，设计人员通过组件的组合和配置创建实体，从而实现关注点的分离。这一点对于实现快速迭代、保持模块化以及最终确保可扩展性至关重要。我们描述了康考迪亚库的持续演变，展示了它是如何让用户能够有效配置与他们具体目标相一致的场景。 

---
# A New Approach for Multicriteria Assessment in the Ranking of Alternatives Using Cardinal and Ordinal Data 

**Title (ZH)**: 基于卡片和序位数据的多准则评估在替代方案排序中的新方法 

**Authors**: Fuh-Hwa Franklin Liu, Su-Chuan Shih  

**Link**: [PDF](https://arxiv.org/pdf/2507.08875)  

**Abstract**: Modern methods for multi-criteria assessment (MCA), such as Data Envelopment Analysis (DEA), Stochastic Frontier Analysis (SFA), and Multiple Criteria Decision-Making (MCDM), are utilized to appraise a collection of Decision-Making Units (DMUs), also known as alternatives, based on several criteria. These methodologies inherently rely on assumptions and can be influenced by subjective judgment to effectively tackle the complex evaluation challenges in various fields. In real-world scenarios, it is essential to incorporate both quantitative and qualitative criteria as they consist of cardinal and ordinal data. Despite the inherent variability in the criterion values of different alternatives, the homogeneity assumption is often employed, significantly affecting evaluations. To tackle these challenges and determine the most appropriate alternative, we propose a novel MCA approach that combines two Virtual Gap Analysis (VGA) models. The VGA framework, rooted in linear programming, is pivotal in the MCA methodology. This approach improves efficiency and fairness, ensuring that evaluations are both comprehensive and dependable, thus offering a strong and adaptive solution. Two comprehensive numerical examples demonstrate the accuracy and transparency of our proposed method. The goal is to encourage continued advancement and stimulate progress in automated decision systems and decision support systems. 

**Abstract (ZH)**: 现代多准则评估方法（MCA）如数据包络分析（DEA）、随机前沿分析（SFA）和多准则决策方法（MCDM）被用于基于多个标准评估决策单元（DMUs）或替代方案的集合。这些方法固有地依赖于假设，并可能受到主观判断的影响，以有效应对各种领域中的复杂评估挑战。在实际场景中，必须同时纳入定性和定量标准，因为这些标准包括基数和序数数据。尽管不同替代方案的指标值具有固有的变异性，但通常会采用同质性假设，这对评估产生了显著影响。为应对这些挑战并确定最合适的替代方案，我们提出了结合两种虚拟差距分析（VGA）模型的新颖MCA方法。VGA框架根植于线性规划，对于MCA方法至关重要。此方法提高了效率和公平性，确保评估既全面又可靠，从而提供了一个强大且适应性强的解决方案。两个综合的数值示例展示了我们提出方法的准确性和透明度。目标是促进持续的进步，并激励自动化决策系统和决策支持系统的进展。 

---
# Think Clearly: Improving Reasoning via Redundant Token Pruning 

**Title (ZH)**: 清晰思考：通过冗余token修剪提升推理能力 

**Authors**: Daewon Choi, Jimin Lee, Jihoon Tack, Woomin Song, Saket Dingliwal, Sai Muralidhar Jayanthi, Bhavana Ganesh, Jinwoo Shin, Aram Galstyan, Sravan Babu Bodapati  

**Link**: [PDF](https://arxiv.org/pdf/2507.08806)  

**Abstract**: Recent large language models have shown promising capabilities in long-form reasoning, following structured chains of thought before arriving at a final answer. However, we observe that these reasoning paths tend to include substantial redundancy; analyzing attention patterns reveals that attention scores are widely scattered, particularly incorrect answers exhibit greater attention sparsity. In this paper, we demonstrate that deliberately removing this redundancy in the reasoning process significantly improves performance through clear thinking, i.e., removing distraction. Specifically, we systematically identify reasoning redundancy by measuring token-level attention scores to a special end-of-thinking token, which is appended to an explicit instruction inserted to conclude each intermediate reasoning step. Furthermore, we propose structure-aware pruning that prioritizes removing tokens in low-contributing reasoning chunks over individual tokens. After evicting redundant tokens, we remove the injected end-of-thinking instruction, then resume the reasoning generation. We demonstrate that our method significantly improves overall accuracy across reasoning-intensive benchmarks without any training involved. In particular, our method shows strong performance on challenging mathematical competition benchmarks such as AIME and AMC, where reasoning redundancy is more prevalent. 

**Abstract (ZH)**: 近期的大规模语言模型在长形式推理方面展现了令人鼓舞的能力，能够在遵循结构化的思维链后得出最终答案。然而，我们观察到这些推理路径通常包含大量的冗余；通过对注意力模式进行分析，发现注意力分数分布广泛，特别是在错误的答案中，注意力稀疏更为明显。在本文中，我们证明了在推理过程中故意去除这种冗余可以通过清晰的思考显著提高性能，即去除干扰。具体而言，我们通过测量每个中间推理步骤附加的特定结束思考指令下的标记级注意力分数系统地识别推理冗余。此外，我们提出了一种结构感知剪枝方法，优先去除低贡献度推理块中的标记而不是个别标记。去除冗余标记后，我们移除插入的结束思考指令，然后继续进行推理生成。我们证明了我们的方法在没有参与训练的情况下，在推理密集型基准测试中显著提高了整体准确性。特别地，我们的方法在如AIME和AMC等具有挑战性的数学竞赛基准测试中表现出色，其中推理冗余更为常见。 

---
# Self-supervised Learning on Camera Trap Footage Yields a Strong Universal Face Embedder 

**Title (ZH)**: 自我监督学习在相机陷阱视频上的应用yield一个强大的通用面部嵌入器 

**Authors**: Vladimir Iashin, Horace Lee, Dan Schofield, Andrew Zisserman  

**Link**: [PDF](https://arxiv.org/pdf/2507.10552)  

**Abstract**: Camera traps are revolutionising wildlife monitoring by capturing vast amounts of visual data; however, the manual identification of individual animals remains a significant bottleneck. This study introduces a fully self-supervised approach to learning robust chimpanzee face embeddings from unlabeled camera-trap footage. Leveraging the DINOv2 framework, we train Vision Transformers on automatically mined face crops, eliminating the need for identity labels. Our method demonstrates strong open-set re-identification performance, surpassing supervised baselines on challenging benchmarks such as Bossou, despite utilising no labelled data during training. This work underscores the potential of self-supervised learning in biodiversity monitoring and paves the way for scalable, non-invasive population studies. 

**Abstract (ZH)**: 相机trap正在通过捕捉大量视觉数据革新野生动物监测；然而，个体动物的手动识别仍然是一个显著的瓶颈。本研究引入了一种完全自监督的方法，从未标记的相机trap录像中学习 robust 猩猩面部嵌入。利用DINOv2框架，我们对自动提取的面部裁剪进行Vision Transformers训练，消除了身份标签的需求。我们的方法在Bossou等具有挑战性的基准测试上展示了强大的开放集重新识别性能，尽管在训练过程中未使用任何标记数据。本研究强调了自监督学习在生物多样性监测中的潜力，并为可扩展的非侵入性种群研究铺平了道路。 

---
# EmbRACE-3K: Embodied Reasoning and Action in Complex Environments 

**Title (ZH)**: EmbRACE-3K: 体态化复杂环境中的推理与行动 

**Authors**: Mingxian Lin, Wei Huang, Yitang Li, Chengjie Jiang, Kui Wu, Fangwei Zhong, Shengju Qian, Xin Wang, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.10548)  

**Abstract**: Recent advanced vision-language models(VLMs) have demonstrated strong performance on passive, offline image and video understanding tasks. However, their effectiveness in embodied settings, which require online interaction and active scene understanding remains limited. In such scenarios, an agent perceives the environment from a first-person perspective, with each action dynamically shaping subsequent observations. Even state-of-the-art models such as GPT-4o, Claude 3.5 Sonnet, and Gemini 2.5 Pro struggle in open-environment interactions, exhibiting clear limitations in spatial reasoning and long-horizon planning. To address this gap, we introduce EmRACE-3K, a dataset of over 3,000 language-guided tasks situated in diverse, photorealistic environments constructed using Unreal Engine and the UnrealCV-Zoo framework. The tasks encompass a wide range of embodied challenges, including navigation, object manipulation, and multi-stage goal execution. Each task unfolds as a multi-step trajectory, pairing first-person visual observations with high-level instructions, grounded actions, and natural language rationales that express the agent's intent at every step. Using EmRACE-3K, we establish a benchmark to evaluate the embodied reasoning capabilities of VLMs across three key dimensions: Exploration, Dynamic Spatial-Semantic Reasoning, and Multi-stage Goal Execution. In zero-shot settings, all models achieve success rates below 20%, underscoring the challenge posed by our benchmark and the current limitations of VLMs in interactive environments. To demonstrate the utility of EmRACE-3K, we further fine-tune Qwen2.5-VL-7B using supervised learning followed by reinforcement learning. This approach yields substantial improvements across all three challenge categories, highlighting the dataset's effectiveness in enabling the development of embodied reasoning capabilities. 

**Abstract (ZH)**: 近期的先进视觉-语言模型(VLMs)在被动的、离线的图像和视频理解任务中展现了强大的性能。然而，在需要在线交互和主动场景理解的浸入式环境中，它们的有效性仍然有限。在这种场景中，智能体从第一人称视角感知环境，每次行动都会动态地塑造后续观察。即使是最先进的模型如GPT-4o、Claude 3.5 Sonnet和Gemini 2.5 Pro，在开放环境交互中也表现不佳，显示出在空间推断和长期规划方面的明显局限性。为解决这一差距，我们引入了EmRACE-3K数据集，包含超过3000个由语言引导的任务，这些任务置身于使用Unreal Engine和UnrealCV-Zoo框架构建的多样且逼真的环境中。这些任务涵盖了各种浸入式挑战，包括导航、对象操作和多阶段目标执行。每个任务分为多步轨迹，配以第一人称视觉观察、高层指令、接地动作以及每一步表达智能体意图的自然语言推理。利用EmRACE-3K，我们建立了一个基准来评估VLMs在三种关键维度上的浸入式推理能力：探索、动态空间语义推理和多阶段目标执行。在零样本设置下，所有模型的成功率均低于20%，突显了基准的挑战和当前VLMs在交互环境中面临的局限性。为了展示EmRACE-3K的实用性，我们进一步使用监督学习和强化学习的方式对Qwen2.5-VL-7B进行微调。这种方法在所有三个挑战类别中均取得显著改进，突显了该数据集在促进浸入式推理能力发展方面的有效性。 

---
# Disentangling Neural Disjunctive Normal Form Models 

**Title (ZH)**: 拆解神经析取范型模型 

**Authors**: Kexin Gu Baugh, Vincent Perreault, Matthew Baugh, Luke Dickens, Katsumi Inoue, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2507.10546)  

**Abstract**: Neural Disjunctive Normal Form (DNF) based models are powerful and interpretable approaches to neuro-symbolic learning and have shown promising results in classification and reinforcement learning settings without prior knowledge of the tasks. However, their performance is degraded by the thresholding of the post-training symbolic translation process. We show here that part of the performance degradation during translation is due to its failure to disentangle the learned knowledge represented in the form of the networks' weights. We address this issue by proposing a new disentanglement method; by splitting nodes that encode nested rules into smaller independent nodes, we are able to better preserve the models' performance. Through experiments on binary, multiclass, and multilabel classification tasks (including those requiring predicate invention), we demonstrate that our disentanglement method provides compact and interpretable logical representations for the neural DNF-based models, with performance closer to that of their pre-translation counterparts. Our code is available at this https URL. 

**Abstract (ZH)**: 基于神经析取范式(DNF)的模型是神经符号学习的强大且可解释的方法，在无需任务先验知识的情况下，在分类和强化学习设置中显示出有希望的结果。然而，它们的性能在训练后符号转换过程的阈值化过程中受损。我们在此表明，在转换过程中性能下降的部分原因是其无法解开以网络权重形式表示的学习知识。我们通过提出一种新的解耦方法来解决这一问题；通过将编码嵌套规则的节点分裂为更小的独立节点，我们能够更好地保持模型的性能。通过二分类、多分类和多标签分类任务（包括需要谓词发明的任务）的实验，我们证明了我们的解耦方法为神经析取范式基于的模型提供了紧凑且可解释的逻辑表示，其性能接近于其转换前的版本。我们的代码可在以下链接获取：这个 https URL。 

---
# ScaffoldAvatar: High-Fidelity Gaussian Avatars with Patch Expressions 

**Title (ZH)**: ScaffoldAvatar: 高保真高斯 avatar 与补丁表情 

**Authors**: Shivangi Aneja, Sebastian Weiss, Irene Baeza, Prashanth Chandran, Gaspard Zoss, Matthias Nießner, Derek Bradley  

**Link**: [PDF](https://arxiv.org/pdf/2507.10542)  

**Abstract**: Generating high-fidelity real-time animated sequences of photorealistic 3D head avatars is important for many graphics applications, including immersive telepresence and movies. This is a challenging problem particularly when rendering digital avatar close-ups for showing character's facial microfeatures and expressions. To capture the expressive, detailed nature of human heads, including skin furrowing and finer-scale facial movements, we propose to couple locally-defined facial expressions with 3D Gaussian splatting to enable creating ultra-high fidelity, expressive and photorealistic 3D head avatars. In contrast to previous works that operate on a global expression space, we condition our avatar's dynamics on patch-based local expression features and synthesize 3D Gaussians at a patch level. In particular, we leverage a patch-based geometric 3D face model to extract patch expressions and learn how to translate these into local dynamic skin appearance and motion by coupling the patches with anchor points of Scaffold-GS, a recent hierarchical scene representation. These anchors are then used to synthesize 3D Gaussians on-the-fly, conditioned by patch-expressions and viewing direction. We employ color-based densification and progressive training to obtain high-quality results and faster convergence for high resolution 3K training images. By leveraging patch-level expressions, ScaffoldAvatar consistently achieves state-of-the-art performance with visually natural motion, while encompassing diverse facial expressions and styles in real time. 

**Abstract (ZH)**: 基于局部定义的表情结合3D高斯点绘制生成超高保真实时光ướ实景头部avatar 

---
# CodeJudgeBench: Benchmarking LLM-as-a-Judge for Coding Tasks 

**Title (ZH)**: CodeJudgeBench：评估编码任务中LLM作为裁判的性能 

**Authors**: Hongchao Jiang, Yiming Chen, Yushi Cao, Hung-yi Lee, Robby T. Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.10535)  

**Abstract**: Large Language Models (LLMs) have significantly advanced the state-of-the-art in various coding tasks. Beyond directly answering user queries, LLMs can also serve as judges, assessing and comparing the quality of responses generated by other models. Such an evaluation capability is crucial both for benchmarking different LLMs and for improving response quality through response ranking. However, despite the growing adoption of the LLM-as-a-Judge paradigm, its effectiveness in coding scenarios remains underexplored due to the absence of dedicated benchmarks. To address this gap, we introduce CodeJudgeBench, a benchmark explicitly designed to evaluate the performance of LLM-as-a-Judge models across three critical coding tasks: code generation, code repair, and unit test generation. Through comprehensive benchmarking of 26 LLM-as-a-Judge models, we find that recent thinking models significantly outperform non-thinking models on our carefully designed code judging tasks. Notably, even relatively small thinking models, such as Qwen3-8B, can outperform specially trained LLM-as-a-Judge models up to 70B in size. Nevertheless, all models still exhibit significant randomness in their judgment of coding tasks. For pairwise judging tasks, simply changing the order in which responses are presented can substantially impact accuracy. In addition, when judging code and unit tests written by different LLMs, LLM-as-a-Judge models also show variance in performance. This sensitivity raises concerns about the reliability and consistency of LLM-as-a-Judge in coding scenarios. Lastly, we study optimal prompting strategies for LLM-as-a-Judge. We find that using pair-wise comparison outperforms scalar point-wise judging. Furthermore, retaining comments and reasoning in the full, unprocessed LLM response leads to improved judge performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种编码任务中显著推动了最新的技术水平。除了直接回答用户查询之外，LLMs还可以作为评判者，评估和比较其他模型生成响应的质量。这种评估能力对于不同LLM的基准测试和通过响应排序提高响应质量至关重要。然而，尽管LLM作为评判者范式日益普及，但由于缺乏专门的基准，其在编码场景中的有效性仍然未被充分探索。为解决这一问题，我们引入了CodeJudgeBench，一个专门设计用于评估LLM作为评判者性能的基准，涵盖了三个关键编码任务：代码生成、代码修复和单元测试生成。通过26个LLM作为评判者模型的全面基准测试，我们发现近期的思考模型在我们精心设计的代码评判任务中显著优于非思考模型。值得注意的是，即使是相对较小的思考模型，如Qwen3-8B，也能够超越高达70B规模的专门训练的LLM作为评判者模型。然而，所有模型在评判编码任务时仍然表现出显著的随机性。在成对评判任务中，仅更改响应呈现的顺序就能显著影响准确性。此外，在评判不同LLM编写的不同代码和单元测试时，LLM作为评判者模型也表现出性能差异。这种敏感性引发了对LLM作为评判者在编码场景中可靠性和一致性的担忧。最后，我们研究了LLM作为评判者的最优提示策略。我们发现，使用成对比较优于标量点评判。此外，保留完整未处理的LLM响应中的注释和推理有助于提高评判者性能。 

---
# WildFX: A DAW-Powered Pipeline for In-the-Wild Audio FX Graph Modeling 

**Title (ZH)**: WildFX：由数字音频工作站驱动的野外音频效果图建模管道 

**Authors**: Qihui Yang, Taylor Berg-Kirkpatrick, Julian McAuley, Zachary Novack  

**Link**: [PDF](https://arxiv.org/pdf/2507.10534)  

**Abstract**: Despite rapid progress in end-to-end AI music generation, AI-driven modeling of professional Digital Signal Processing (DSP) workflows remains challenging. In particular, while there is growing interest in neural black-box modeling of audio effect graphs (e.g. reverb, compression, equalization), AI-based approaches struggle to replicate the nuanced signal flow and parameter interactions used in professional workflows. Existing differentiable plugin approaches often diverge from real-world tools, exhibiting inferior performance relative to simplified neural controllers under equivalent computational constraints. We introduce WildFX, a pipeline containerized with Docker for generating multi-track audio mixing datasets with rich effect graphs, powered by a professional Digital Audio Workstation (DAW) backend. WildFX supports seamless integration of cross-platform commercial plugins or any plugins in the wild, in VST/VST3/LV2/CLAP formats, enabling structural complexity (e.g., sidechains, crossovers) and achieving efficient parallelized processing. A minimalist metadata interface simplifies project/plugin configuration. Experiments demonstrate the pipeline's validity through blind estimation of mixing graphs, plugin/gain parameters, and its ability to bridge AI research with practical DSP demands. The code is available on: this https URL. 

**Abstract (ZH)**: 尽管端到端人工智能音乐生成取得了快速进展，但基于人工智能的专业数字信号处理（DSP）工作流建模依然具有挑战性。特别是，虽然对音频效果图（如混响、压缩、均衡）的神经黑箱建模日益受到关注，但基于人工智能的方法在复制专业工作流中的细微信号流动和参数交互方面仍显得力不从心。现有的可微插件方法往往与实际工具有所偏离，在同等计算约束条件下表现出较低的性能。我们提出了WildFX，这是一种基于Docker封装的工作流程管道，通过专业的数字音频工作站（DAW）后端生成包含丰富效果图的多轨音频混音数据集。WildFX 支持跨平台商业插件或任何野生插件的无缝集成（适用于VST/VST3/LV2/CLAP格式），能够实现结构复杂性（如侧链、分频器）并实现高效并行处理。简约的元数据接口简化了项目/插件配置。实验通过盲估计算法人声图、插件/增益参数，并展示了其将人工智能研究与实际DSP需求衔接的能力。代码可在以下链接获取：this https URL。 

---
# Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination 

**Title (ZH)**: 推理还是记忆？由于数据污染导致的强化学习不可靠结果探究 

**Authors**: Mingqi Wu, Zhihao Zhang, Qiaole Dong, Zhiheng Xi, Jun Zhao, Senjie Jin, Xiaoran Fan, Yuhao Zhou, Yanwei Fu, Qin Liu, Songyang Zhang, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10532)  

**Abstract**: The reasoning capabilities of large language models (LLMs) have been a longstanding focus of research. Recent works have further enhanced these capabilities using reinforcement learning (RL), with many new methods claiming significant improvements with minimal or no external supervision. Surprisingly, some studies even suggest that random or incorrect reward signals can enhance reasoning performance. However, these breakthroughs are mostly reported on the Qwen2.5 model family and evaluated on well-known benchmarks such as MATH-500, AMC, and AIME, while failing to achieve similar gains on other models like Llama, which warrants further investigation. Our analysis shows that although Qwen2.5 achieves strong mathematical reasoning performance, its pretraining on large-scale web corpora makes it vulnerable to data contamination in popular benchmarks. As a result, results derived from these benchmarks may be unreliable. To address this, we introduce a generator that produces fully synthetic arithmetic problems of arbitrary length and difficulty, yielding a clean dataset we call RandomCalculation. Using these leakage-free datasets, we show that only accurate reward signals consistently improve performance, while noisy or incorrect signals do not. We advocate for evaluating RL methods on uncontaminated benchmarks and across diverse model families to ensure trustworthy conclusions. 

**Abstract (ZH)**: 大型语言模型的推理能力一直是研究的长期焦点。近年来，通过强化学习（RL）进一步增强了这些能力，许多新方法声称在最小或无需外部监督的情况下实现了显著的改进。令人惊讶的是，有些研究甚至表明随机或错误的奖励信号可以提高推理性能。然而，这些突破主要是在Qwen2.5模型家族中报告的，并在著名的基准测试如MATH-500、AMC和AIME上进行评估，但在其他模型如Llama上未能取得类似收益，这需要进一步调查。我们的分析显示，尽管Qwen2.5在数学推理方面表现出色，但由于其大规模网络语料库的预训练，它在流行的基准测试中容易受到数据污染的影响。因此，这些基准测试得出的结果可能不可靠。为了解决这一问题，我们引入了一个生成器，用于生成任意长度和难度的完全合成算术问题，从而产生一个清洁的数据集，称为RandomCalculation。使用这些无泄漏的数据集，我们展示了只有准确的奖励信号能够一致地提高性能，而嘈杂或错误的信号则不起作用。我们主张在无污染的基准测试和多种模型家族中评估RL方法，以确保可靠的结论。 

---
# Accurate generation of chemical reaction transition states by conditional flow matching 

**Title (ZH)**: 基于条件流匹配的化学反应过渡态准确生成 

**Authors**: Ping Tuo, Jiale Chen, Ju Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.10530)  

**Abstract**: Transition state (TS) structures define the critical geometries and energy barriers underlying chemical reactivity, yet their fleeting nature renders them experimentally elusive and drives the reliance on costly, high-throughput density functional theory (DFT) calculations. Here, we introduce TS-GEN, a conditional flow-matching generative model that maps samples from a simple Gaussian prior directly to transition-state saddle-point geometries in a single, deterministic pass. By embedding both reactant and product conformations as conditioning information, TS-GEN learns to transport latent noise to true TS structures via an optimal-transport path, effectively replacing the iterative optimization common in nudged-elastic band or string-method algorithms. TS-GEN delivers unprecedented accuracy, achieving a root-mean-square deviation of $0.004\ \rm{\mathring{A}}$ (vs. $0.103\ \rm{\mathring{A}}$ for prior state-of-the-art) and a mean barrier-height error of $1.019\ {\rm kcal/mol}$ (vs. $2.864\ {\rm kcal/mol}$), while requiring only $0.06\ {\rm s}$ GPU time per inference. Over 87% of generated TSs meet chemical-accuracy criteria ($<1.58\ {\rm kcal/mol}$ error), substantially outpacing existing methods. TS-GEN also exhibits strong transferability to out-of-distribution reactions from a larger database. By uniting sub-angstrom precision, sub-second speed, and broad applicability, TS-GEN will be highly useful for high-throughput exploration of complex reaction networks, paving the way to the exploration of novel chemical reaction mechanisms. 

**Abstract (ZH)**: TS-GEN: A Conditional Flow-Matching Generative Model for Accurate and Efficient Transition-State Generation 

---
# Chat with AI: The Surprising Turn of Real-time Video Communication from Human to AI 

**Title (ZH)**: 与AI对话：真人实时视频通信 surprising转为意外的AI交互 

**Authors**: Jiangkai Wu, Zhiyuan Ren, Liming Liu, Xinggong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10510)  

**Abstract**: AI Video Chat emerges as a new paradigm for Real-time Communication (RTC), where one peer is not a human, but a Multimodal Large Language Model (MLLM). This makes interaction between humans and AI more intuitive, as if chatting face-to-face with a real person. However, this poses significant challenges to latency, because the MLLM inference takes up most of the response time, leaving very little time for video streaming. Due to network uncertainty and instability, transmission latency becomes a critical bottleneck preventing AI from being like a real person. To address this, we propose Artic, an AI-oriented Real-time Communication framework, exploring the network requirement shift from "humans watching video" to "AI understanding video". To reduce bitrate dramatically while maintaining MLLM accuracy, we propose Context-Aware Video Streaming that recognizes the importance of each video region for chat and allocates bitrate almost exclusively to chat-important regions. To avoid packet retransmission, we propose Loss-Resilient Adaptive Frame Rate that leverages previous frames to substitute for lost/delayed frames while avoiding bitrate waste. To evaluate the impact of video streaming quality on MLLM accuracy, we build the first benchmark, named Degraded Video Understanding Benchmark (DeViBench). Finally, we discuss some open questions and ongoing solutions for AI Video Chat. 

**Abstract (ZH)**: 基于AI的视频聊天作为实时通信（RTC）的新范式，在其中一方不再是人类，而是一个多模态大型语言模型（MLLM）。这使得人与AI之间的互动更加直观，仿佛在与真实的人面对面聊天。然而，这给延迟带来了重大挑战，因为MLLM的推断占据了大部分响应时间，留给视频流的时间非常有限。由于网络的不确定性与不稳定，传输延迟成为阻碍AI表现得像真实的人的关键瓶颈。为解决这一问题，我们提出Artic，一种面向AI的实时通信框架，探索从“人类观看视频”到“AI理解视频”的网络需求转变。为了在大幅减少比特率的同时保持MLLM的准确性，我们提出了上下文感知视频流媒体，该方法识别每个视频区域在聊天中的重要性，并几乎将所有比特率分配给聊天重要区域。为了避免重传包，我们提出了抗丢包自适应帧率，利用先前的帧来替代丢失或延迟的帧，同时避免比特率浪费。为了评估视频流质量对MLLM准确性的影响，我们构建了首个基准，名为降级视频理解基准（DeViBench）。最后，我们讨论了AI视频聊天的一些开放问题及正在进行的解决方案。 

---
# Benchmarking and Evaluation of AI Models in Biology: Outcomes and Recommendations from the CZI Virtual Cells Workshop 

**Title (ZH)**: 生物领域中人工智能模型的基准测试与评估：CZI 虚拟细胞工作坊的成果与建议 

**Authors**: Elizabeth Fahsbender, Alma Andersson, Jeremy Ash, Polina Binder, Daniel Burkhardt, Benjamin Chang, Georg K. Gerber, Anthony Gitter, Patrick Godau, Ankit Gupta, Genevieve Haliburton, Siyu He, Trey Ideker, Ivana Jelic, Aly Khan, Yang-Joon Kim, Aditi Krishnapriyan, Jon M. Laurent, Tianyu Liu 28, Emma Lundberg, Shalin B. Mehta, Rob Moccia, Angela Oliveira Pisco, Katherine S. Pollard, Suresh Ramani, Julio Saez-Rodriguez, Yasin Senbabaoglu, Elana Simon, Srinivasan Sivanandan, Gustavo Stolovitzky, Marc Valer, Bo Wang, Xikun Zhang, James Zou, Katrina Kalantar  

**Link**: [PDF](https://arxiv.org/pdf/2507.10502)  

**Abstract**: Artificial intelligence holds immense promise for transforming biology, yet a lack of standardized, cross domain, benchmarks undermines our ability to build robust, trustworthy models. Here, we present insights from a recent workshop that convened machine learning and computational biology experts across imaging, transcriptomics, proteomics, and genomics to tackle this gap. We identify major technical and systemic bottlenecks such as data heterogeneity and noise, reproducibility challenges, biases, and the fragmented ecosystem of publicly available resources and propose a set of recommendations for building benchmarking frameworks that can efficiently compare ML models of biological systems across tasks and data modalities. By promoting high quality data curation, standardized tooling, comprehensive evaluation metrics, and open, collaborative platforms, we aim to accelerate the development of robust benchmarks for AI driven Virtual Cells. These benchmarks are crucial for ensuring rigor, reproducibility, and biological relevance, and will ultimately advance the field toward integrated models that drive new discoveries, therapeutic insights, and a deeper understanding of cellular systems. 

**Abstract (ZH)**: 人工智能在生物学领域的应用前景广阔，但标准化跨领域基准的缺乏阻碍了我们构建稳健可信模型的能力。为此，我们介绍了最近一次研讨会的成果，该研讨会汇聚了来自成像、转录组学、蛋白质组学和基因组学领域的机器学习和计算生物学专家，以解决这一问题。我们指出了数据异质性、噪声、可重复性挑战、偏见以及公开可用资源碎片化等主要的技术和系统瓶颈，并提出了构建跨任务和数据模态比较机器学习模型基准框架的建议。通过促进高质量的数据管理、标准化工具、全面的评价指标以及开放协作平台，我们旨在加速稳健基准的开发，以驱动AI驱动的虚拟细胞领域的发展。这些基准对于确保严格性、可重复性和生物学相关性至关重要，并将最终推动该领域向集成模型发展，从而促进新的发现、治疗洞察以及细胞系统更深刻的理解。 

---
# Scene-Aware Conversational ADAS with Generative AI for Real-Time Driver Assistance 

**Title (ZH)**: 基于场景aware的生成AI实时驾驶辅助对话系统 

**Authors**: Kyungtae Han, Yitao Chen, Rohit Gupta, Onur Altintas  

**Link**: [PDF](https://arxiv.org/pdf/2507.10500)  

**Abstract**: While autonomous driving technologies continue to advance, current Advanced Driver Assistance Systems (ADAS) remain limited in their ability to interpret scene context or engage with drivers through natural language. These systems typically rely on predefined logic and lack support for dialogue-based interaction, making them inflexible in dynamic environments or when adapting to driver intent. This paper presents Scene-Aware Conversational ADAS (SC-ADAS), a modular framework that integrates Generative AI components including large language models, vision-to-text interpretation, and structured function calling to enable real-time, interpretable, and adaptive driver assistance. SC-ADAS supports multi-turn dialogue grounded in visual and sensor context, allowing natural language recommendations and driver-confirmed ADAS control. Implemented in the CARLA simulator with cloud-based Generative AI, the system executes confirmed user intents as structured ADAS commands without requiring model fine-tuning. We evaluate SC-ADAS across scene-aware, conversational, and revisited multi-turn interactions, highlighting trade-offs such as increased latency from vision-based context retrieval and token growth from accumulated dialogue history. These results demonstrate the feasibility of combining conversational reasoning, scene perception, and modular ADAS control to support the next generation of intelligent driver assistance. 

**Abstract (ZH)**: 基于场景的对话式先进驾驶辅助系统（SC-ADAS） 

---
# Cameras as Relative Positional Encoding 

**Title (ZH)**: 摄像头作为相对位置编码 

**Authors**: Ruilong Li, Brent Yi, Junchen Liu, Hang Gao, Yi Ma, Angjoo Kanazawa  

**Link**: [PDF](https://arxiv.org/pdf/2507.10496)  

**Abstract**: Transformers are increasingly prevalent for multi-view computer vision tasks, where geometric relationships between viewpoints are critical for 3D perception. To leverage these relationships, multi-view transformers must use camera geometry to ground visual tokens in 3D space. In this work, we compare techniques for conditioning transformers on cameras: token-level raymap encodings, attention-level relative pose encodings, and a new relative encoding we propose -- Projective Positional Encoding (PRoPE) -- that captures complete camera frustums, both intrinsics and extrinsics, as a relative positional encoding. Our experiments begin by showing how relative camera conditioning improves performance in feedforward novel view synthesis, with further gains from PRoPE. This holds across settings: scenes with both shared and varying intrinsics, when combining token- and attention-level conditioning, and for generalization to inputs with out-of-distribution sequence lengths and camera intrinsics. We then verify that these benefits persist for different tasks, stereo depth estimation and discriminative spatial cognition, as well as larger model sizes. 

**Abstract (ZH)**: 多视图变换器在利用摄像头几何关系进行三维感知方面越来越普遍。我们比较了变换器基于摄像头的条件训练技术：基于代理解码的标记级射线图编码、基于相对位姿的注意级别编码以及我们提出的新相对编码——投影位置编码（PRoPE），它捕捉包括内参和外参在内的完整摄像机视锥作为相对位置编码。我们的实验首先展示了基于相对摄像头条件可以提高前向新颖视图合成的性能，并且通过使用PRoPE还能进一步提升。这种改进在不同场景中保持有效，包括共享和变化内参的场景，当结合标记级和注意级别条件时，以及对于输入的分布外序列长度和内参的一般泛化。此外，我们验证了这些优势可以在不同的任务，如立体深度估计和区分数学空间认知，以及更大模型规模中保持有效。 

---
# BenchReAD: A systematic benchmark for retinal anomaly detection 

**Title (ZH)**: BenchReAD: 一种系统性的视网膜异常检测基准 

**Authors**: Chenyu Lian, Hong-Yu Zhou, Zhanli Hu, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2507.10492)  

**Abstract**: Retinal anomaly detection plays a pivotal role in screening ocular and systemic diseases. Despite its significance, progress in the field has been hindered by the absence of a comprehensive and publicly available benchmark, which is essential for the fair evaluation and advancement of methodologies. Due to this limitation, previous anomaly detection work related to retinal images has been constrained by (1) a limited and overly simplistic set of anomaly types, (2) test sets that are nearly saturated, and (3) a lack of generalization evaluation, resulting in less convincing experimental setups. Furthermore, existing benchmarks in medical anomaly detection predominantly focus on one-class supervised approaches (training only with negative samples), overlooking the vast amounts of labeled abnormal data and unlabeled data that are commonly available in clinical practice. To bridge these gaps, we introduce a benchmark for retinal anomaly detection, which is comprehensive and systematic in terms of data and algorithm. Through categorizing and benchmarking previous methods, we find that a fully supervised approach leveraging disentangled representations of abnormalities (DRA) achieves the best performance but suffers from significant drops in performance when encountering certain unseen anomalies. Inspired by the memory bank mechanisms in one-class supervised learning, we propose NFM-DRA, which integrates DRA with a Normal Feature Memory to mitigate the performance degradation, establishing a new SOTA. The benchmark is publicly available at this https URL. 

**Abstract (ZH)**: 视网膜异常检测在眼科和全身疾病筛查中扮演着重要角色。尽管如此，由于缺乏全面且公开可用的标准基准，该领域的进展受到了限制，而标准基准对于方法的公平评估和进步至关重要。由于这一限制，之前与视网膜图像相关的异常检测工作受到了以下限制：（1）异常类型过于有限且过于简单，（2）测试集几乎饱和，以及（3）缺乏泛化评估，从而导致不够令人信服的实验设置。此外，现有的医学异常检测基准主要集中在单类监督方法上（仅使用负样本进行训练），而忽视了临床实践中通常可用的大量标记异常数据和未标记数据。为弥补这些不足，我们提出了一个全面且系统化的视网膜异常检测基准。通过对先前方法进行分类和基准测试，我们发现利用分离表示异常（DRA）的完全监督方法表现最佳，但在遇到某些未见过的异常时性能显著下降。受单类监督学习中记忆库机制的启发，我们提出了NFM-DRA，将DRA与Normal Feature Memory相结合，以缓解性能下降，从而建立了新的SOTA。该基准可在此处公开访问：<https://github.com/alibaba/Qwen-Benchmark>。 

---
# Can You Detect the Difference? 

**Title (ZH)**: 你能检测到差异吗？ 

**Authors**: İsmail Tarım, Aytuğ Onan  

**Link**: [PDF](https://arxiv.org/pdf/2507.10475)  

**Abstract**: The rapid advancement of large language models (LLMs) has raised concerns about reliably detecting AI-generated text. Stylometric metrics work well on autoregressive (AR) outputs, but their effectiveness on diffusion-based models is unknown. We present the first systematic comparison of diffusion-generated text (LLaDA) and AR-generated text (LLaMA) using 2 000 samples. Perplexity, burstiness, lexical diversity, readability, and BLEU/ROUGE scores show that LLaDA closely mimics human text in perplexity and burstiness, yielding high false-negative rates for AR-oriented detectors. LLaMA shows much lower perplexity but reduced lexical fidelity. Relying on any single metric fails to separate diffusion outputs from human writing. We highlight the need for diffusion-aware detectors and outline directions such as hybrid models, diffusion-specific stylometric signatures, and robust watermarking. 

**Abstract (ZH)**: 大语言模型的 rapid advancement 已引起对可靠检测 AI 生成文本的关注。分发生成文本（LLaDA）和自回归生成文本（LLaMA）的系统比较表明， perplexity、burstiness、词汇多样性、可读性和 BLEU/ROUGE 分数显示 LLaDA 在 perplexity 和 burstiness 方面接近人类文本，导致针对自回归定向检测器的高假阴性率。LLaMA 的 perplexity 较低但词汇一致性降低。任何单一指标都无法区分分发输出与人类写作。我们强调需要分发意识检测器，并概述了混合模型、分发特定的文体特征以及稳健的数字水印等方向。 

---
# Privacy-Preserving Multi-Stage Fall Detection Framework with Semi-supervised Federated Learning and Robotic Vision Confirmation 

**Title (ZH)**: 基于半监督联邦学习和机器人视觉确认的隐私保护多阶段跌倒检测框架 

**Authors**: Seyed Alireza Rahimi Azghadi, Truong-Thanh-Hung Nguyen, Helene Fournier, Monica Wachowicz, Rene Richard, Francis Palma, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.10474)  

**Abstract**: The aging population is growing rapidly, and so is the danger of falls in older adults. A major cause of injury is falling, and detection in time can greatly save medical expenses and recovery time. However, to provide timely intervention and avoid unnecessary alarms, detection systems must be effective and reliable while addressing privacy concerns regarding the user. In this work, we propose a framework for detecting falls using several complementary systems: a semi-supervised federated learning-based fall detection system (SF2D), an indoor localization and navigation system, and a vision-based human fall recognition system. A wearable device and an edge device identify a fall scenario in the first system. On top of that, the second system uses an indoor localization technique first to localize the fall location and then navigate a robot to inspect the scenario. A vision-based detection system running on an edge device with a mounted camera on a robot is used to recognize fallen people. Each of the systems of this proposed framework achieves different accuracy rates. Specifically, the SF2D has a 0.81% failure rate equivalent to 99.19% accuracy, while the vision-based fallen people detection achieves 96.3% accuracy. However, when we combine the accuracy of these two systems with the accuracy of the navigation system (95% success rate), our proposed framework creates a highly reliable performance for fall detection, with an overall accuracy of 99.99%. Not only is the proposed framework safe for older adults, but it is also a privacy-preserving solution for detecting falls. 

**Abstract (ZH)**: 老龄化人口快速增长，老年人跌倒的风险也在增加。跌倒是造成伤害的主要原因，及时监测和检测可以大大节省医疗费用和恢复时间。然而，为了提供及时的干预并避免不必要的报警，检测系统必须有效可靠且能够解决用户隐私问题。本研究提出了一种使用多种互补系统的框架：基于半监督联邦学习的跌倒检测系统（SF2D）、室内定位与导航系统以及基于视觉的人体跌倒识别系统。第一系统使用可穿戴设备和边缘设备识别跌倒场景。在此基础上，第二系统首先使用室内定位技术定位跌倒地点，然后导航机器人检查场景。机器人上装有摄像头的边缘设备运行的基于视觉的检测系统用于识别跌倒的人。该框架中的每个系统都实现了不同的准确率。具体来说，SF2D的失败率为0.81%，相当于99.19%的准确率，而基于视觉的人体跌倒检测准确率为96.3%。然而，当我们结合这两大系统与导航系统（95%的成功率）的准确率时，本研究提出框架为跌倒检测创造了高度可靠的性能，整体准确率为99.99%。不仅该框架安全适用于老年人，而且还是一个保护隐私的跌倒检测解决方案。 

---
# An Empirical Evaluation of AI-Powered Non-Player Characters' Perceived Realism and Performance in Virtual Reality Environments 

**Title (ZH)**: 基于AI驱动的非玩家角色在虚拟现实环境中感知现实感与性能的实证评价 

**Authors**: Mikko Korkiakoski, Saeid Sheikhi, Jesper Nyman, Jussi Saariniemi, Kalle Tapio, Panos Kostakos  

**Link**: [PDF](https://arxiv.org/pdf/2507.10469)  

**Abstract**: Advancements in artificial intelligence (AI) have significantly enhanced the realism and interactivity of non-player characters (NPCs) in virtual reality (VR), creating more engaging and believable user experiences. This paper evaluates AI-driven NPCs within a VR interrogation simulator, focusing on their perceived realism, usability, and system performance. The simulator features two AI-powered NPCs, a suspect, and a partner, using GPT-4 Turbo to engage participants in a scenario to determine the suspect's guilt or innocence. A user study with 18 participants assessed the system using the System Usability Scale (SUS), Game Experience Questionnaire (GEQ), and a Virtual Agent Believability Questionnaire, alongside latency measurements for speech-to-text (STT), text-to-speech (TTS), OpenAI GPT-4 Turbo, and overall (cycle) latency. Results showed an average cycle latency of 7 seconds, influenced by the increasing conversational context. Believability scored 6.67 out of 10, with high ratings in behavior, social relationships, and intelligence but moderate scores in emotion and personality. The system achieved a SUS score of 79.44, indicating good usability. These findings demonstrate the potential of large language models to improve NPC realism and interaction in VR while highlighting challenges in reducing system latency and enhancing emotional depth. This research contributes to the development of more sophisticated AI-driven NPCs, revealing the need for performance optimization to achieve increasingly immersive virtual experiences. 

**Abstract (ZH)**: 人工智能（AI）的进步显著提高了虚拟现实（VR）中非玩家角色（NPCs）的 realism 和交互性，创造了更具吸引力和可信度的用户体验。本文评估了AI驱动的NPC在VR审讯模拟器中的应用，重点关注它们的可信度、可用性和系统性能。该模拟器使用GPT-4 Turbo打造了两个NPC角色，一名嫌疑人和一名伙伴，与参与者进行交互，以确定嫌疑人的罪行。该研究包含18名参与者，使用系统可用性量表（SUS）、游戏体验问卷（GEQ）和虚拟代理可信度问卷评估系统，并测量了语音转文本（STT）、文本转语音（TTS）、OpenAI GPT-4 Turbo以及整体（周期）延迟。结果显示平均周期延迟为7秒，受对话上下文增加的影响。可信度评分为6.67分，行为、社会关系和智力方面得分较高，但在情感和个性方面得分较低。该系统获得了SUS评分为79.44，表明良好的可用性。这些发现展示了大型语言模型在提高VR中NPC的realism和交互性方面的潜在价值，同时也指出了减少系统延迟和增强情感深度的挑战。这项研究为开发更高级的AI驱动NPC做出了贡献，揭示了实现日益沉浸式虚拟体验时需要进行性能优化的需求。 

---
# AudioMAE++: learning better masked audio representations with SwiGLU FFNs 

**Title (ZH)**: AudioMAE++: 使用SwiGLU FFNs学习更好的掩蔽音频表示 

**Authors**: Sarthak Yadav, Sergios Theodoridis, Zheng-Hua Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.10464)  

**Abstract**: Masked Autoencoders (MAEs) trained on audio spectrogram patches have emerged as a prominent approach for learning self-supervised audio representations. While several recent papers have evaluated key aspects of training MAEs on audio data, the majority of these approaches still leverage vanilla transformer building blocks, whereas the transformer community has seen steady integration of newer architectural advancements. In this work, we propose AudioMAE++, a revamped audio masked autoencoder with two such enhancements, namely macaron-style transformer blocks with gated linear units. When pretrained on the AudioSet dataset, the proposed AudioMAE++ models outperform existing MAE based approaches on 10 diverse downstream tasks, demonstrating excellent performance on audio classification and speech-based benchmarks. The proposed AudioMAE++ models also demonstrate excellent scaling characteristics, outperforming directly comparable standard MAE baselines with up to 4x more parameters. 

**Abstract (ZH)**: 基于音频光谱图片段训练的掩码自动编码器（MAEs）已经成为学习自监督音频表示的一种 prominant 方法。尽管近期有多篇论文评估了在音频数据上训练 MAEs 的关键方面，大多数这些方法仍然使用基本的变压器构建块，而变压器社区已经稳定地将新的架构 advancements 融合进来。在此项工作中，我们提出了 AudioMAE++，这是一种带有两种改进的重新设计的音频掩码自动编码器，具体来说是带门线性单元的 macaron 风格变压器块。通过在 AudioSet 数据集上预训练，提出的方法在 10 个不同的下游任务中优于现有的基于 MAE 的方法，展示了在音频分类和语音基准测试中的出色性能。提出的 AudioMAE++ 模型还展示了优异的扩展特性，其参数量最多比直接可比的标准 MAE 基线多 4 倍但仍表现出色。 

---
# RAPNet: A Receptive-Field Adaptive Convolutional Neural Network for Pansharpening 

**Title (ZH)**: RAPNet：一种适应性卷积神经网络用于多光谱与高分辨率影像融合 

**Authors**: Tao Tang, Chengxu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10461)  

**Abstract**: Pansharpening refers to the process of integrating a high resolution panchromatic (PAN) image with a lower resolution multispectral (MS) image to generate a fused product, which is pivotal in remote sensing. Despite the effectiveness of CNNs in addressing this challenge, they are inherently constrained by the uniform application of convolutional kernels across all spatial positions, overlooking local content variations. To overcome this issue, we introduce RAPNet, a new architecture that leverages content-adaptive convolution. At its core, RAPNet employs the Receptive-field Adaptive Pansharpening Convolution (RAPConv), designed to produce spatially adaptive kernels responsive to local feature context, thereby enhancing the precision of spatial detail extraction. Additionally, the network integrates the Pansharpening Dynamic Feature Fusion (PAN-DFF) module, which incorporates an attention mechanism to achieve an optimal balance between spatial detail enhancement and spectral fidelity. Comprehensive evaluations on publicly available datasets confirm that RAPNet delivers superior performance compared to existing approaches, as demonstrated by both quantitative metrics and qualitative assessments. Ablation analyses further substantiate the effectiveness of the proposed adaptive components. 

**Abstract (ZH)**: pansharpening是指将高分辨率Panchromatic（PAN）图像与低分辨率多光谱（MS）图像相结合以生成融合产品的过程，在遥感中至关重要。尽管CNNs在解决这一挑战方面效果显著，但它们固有的缺点是卷积核在所有空间位置上均匀应用，忽视了局部内容的变化。为克服这一问题，我们提出了RAPNet，一种利用内容自适应卷积的新架构。其核心在于使用接收域自适应卷积（RAPConv），旨在生成响应局部特征上下文的自适应卷积核，从而增强空间细节提取的精度。此外，网络整合了Pansharpening动态特征融合（PAN-DFF）模块，该模块包含注意力机制，以实现空间细节增强与光谱保真的最佳平衡。在公开数据集上的全面评估表明，RAPNet在定量指标和定性评估方面均优于现有方法。消融分析进一步证实了所提自适应组件的有效性。 

---
# Logic layer Prompt Control Injection (LPCI): A Novel Security Vulnerability Class in Agentic Systems 

**Title (ZH)**: 逻辑层提示控制注入（LPCI）：代理系统中的新型安全漏洞类别 

**Authors**: Hammad Atta, Ken Huang, Manish Bhatt, Kamal Ahmed, Muhammad Aziz Ul Haq, Yasir Mehmood  

**Link**: [PDF](https://arxiv.org/pdf/2507.10457)  

**Abstract**: The integration of large language models (LLMs) into enterprise systems has created a new class of covert security vulnerabilities, particularly within logic-execution layers and persistent-memory contexts. In this paper, we introduce Logic-Layer Prompt Control Injection (LPCI), a novel attack category in which encoded, delayed, and conditionally triggered payloads are embedded in memory, vector stores, or tool outputs. These payloads can bypass conventional input filters and trigger unauthorised behaviour across sessions. 

**Abstract (ZH)**: 企业系统中大型语言模型的集成引发了新的隐蔽安全漏洞，特别是在逻辑执行层和持久内存环境中。本文介绍了逻辑层提示控制注入（LPCI），这是一种新型攻击类别，其中编码、延迟和条件触发的有效载荷被嵌入到内存、向量存储或工具输出中。这些有效载荷可以绕过常规输入过滤器并在会话之间触发未经授权的行为。 

---
# CoralVQA: A Large-Scale Visual Question Answering Dataset for Coral Reef Image Understanding 

**Title (ZH)**: CoralVQA：一种用于珊瑚礁图像理解的大规模视觉问答数据集 

**Authors**: Hongyong Han, Wei Wang, Gaowei Zhang, Mingjie Li, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10449)  

**Abstract**: Coral reefs are vital yet vulnerable ecosystems that require continuous monitoring to support conservation. While coral reef images provide essential information in coral monitoring, interpreting such images remains challenging due to the need for domain expertise. Visual Question Answering (VQA), powered by Large Vision-Language Models (LVLMs), has great potential in user-friendly interaction with coral reef images. However, applying VQA to coral imagery demands a dedicated dataset that addresses two key challenges: domain-specific annotations and multidimensional questions. In this work, we introduce CoralVQA, the first large-scale VQA dataset for coral reef analysis. It contains 12,805 real-world coral images from 67 coral genera collected from 3 oceans, along with 277,653 question-answer pairs that comprehensively assess ecological and health-related conditions. To construct this dataset, we develop a semi-automatic data construction pipeline in collaboration with marine biologists to ensure both scalability and professional-grade data quality. CoralVQA presents novel challenges and provides a comprehensive benchmark for studying vision-language reasoning in the context of coral reef images. By evaluating several state-of-the-art LVLMs, we reveal key limitations and opportunities. These insights form a foundation for future LVLM development, with a particular emphasis on supporting coral conservation efforts. 

**Abstract (ZH)**: 珊瑚礁是至关重要的但又脆弱的生态系统，需要持续监测以支持保护工作。虽然珊瑚礁图像提供了珊瑚监测所需的重要信息，但由于需要领域专业知识，解读这些图像依然具有挑战性。基于大型视觉-语言模型的视觉问答（VQA）技术在与珊瑚礁图像的友好交互方面具有巨大潜力。然而，将VQA应用于珊瑚图像需要一个专门的数据集来应对两个关键挑战：领域特定的注释和多维问题。本文介绍了CoralVQA，这是首个用于珊瑚礁分析的大规模VQA数据集，包含来自67种珊瑚属、3大海域的12,805张真实珊瑚图像以及277,653个问题回答对，全面评估生态和健康状况。为了构建该数据集，我们与海洋生物学家合作，开发了一种半自动的数据构建 pipeline，确保了可扩展性和专业级数据质量。CoralVQA 提出了新的挑战，并为研究珊瑚礁图像中的视觉-语言推理提供了全面基准。通过评估多种最先进视觉语言模型，我们揭示了关键的局限性和机会。这些见解为未来视觉语言模型的发展奠定了基础，特别是支持珊瑚保护工作。 

---
# Evaluating Fake Music Detection Performance Under Audio Augmentations 

**Title (ZH)**: 评估音频增强条件下假音乐检测性能 

**Authors**: Tomasz Sroka, Tomasz Wężowicz, Dominik Sidorczuk, Mateusz Modrzejewski  

**Link**: [PDF](https://arxiv.org/pdf/2507.10447)  

**Abstract**: With the rapid advancement of generative audio models, distinguishing between human-composed and generated music is becoming increasingly challenging. As a response, models for detecting fake music have been proposed. In this work, we explore the robustness of such systems under audio augmentations. To evaluate model generalization, we constructed a dataset consisting of both real and synthetic music generated using several systems. We then apply a range of audio transformations and analyze how they affect classification accuracy. We test the performance of a recent state-of-the-art musical deepfake detection model in the presence of audio augmentations. The performance of the model decreases significantly even with the introduction of light augmentations. 

**Abstract (ZH)**: 随着生成音频模型的迅速发展，区分由人类创作和生成的音乐变得日益困难。为应对这一挑战，已经提出了检测假音乐的模型。在本文中，我们探讨了这些系统在音频增强下的鲁棒性。为了评估模型的泛化能力，我们构建了一个包含真实音乐和使用多种系统生成的合成音乐的数据集。然后，我们应用多种音频变换，并分析这些变换如何影响分类准确性。在音频增强存在的情况下，我们测试了一种最新的音乐深度假象检测模型的性能。即使引入轻度增强，模型的性能也显著下降。 

---
# Referential ambiguity and clarification requests: comparing human and LLM behaviour 

**Title (ZH)**: 参考不确定性与修正请求：人类与LLM行为比较 

**Authors**: Chris Madge, Matthew Purver, Massimo Poesio  

**Link**: [PDF](https://arxiv.org/pdf/2507.10445)  

**Abstract**: In this work we examine LLMs' ability to ask clarification questions in task-oriented dialogues that follow the asynchronous instruction-giver/instruction-follower format. We present a new corpus that combines two existing annotations of the Minecraft Dialogue Corpus -- one for reference and ambiguity in reference, and one for SDRT including clarifications -- into a single common format providing the necessary information to experiment with clarifications and their relation to ambiguity. With this corpus we compare LLM actions with original human-generated clarification questions, examining how both humans and LLMs act in the case of ambiguity. We find that there is only a weak link between ambiguity and humans producing clarification questions in these dialogues, and low correlation between humans and LLMs. Humans hardly ever produce clarification questions for referential ambiguity, but often do so for task-based uncertainty. Conversely, LLMs produce more clarification questions for referential ambiguity, but less so for task uncertainty. We question if LLMs' ability to ask clarification questions is predicated on their recent ability to simulate reasoning, and test this with different reasoning approaches, finding that reasoning does appear to increase question frequency and relevancy. 

**Abstract (ZH)**: 在这种工作当中，我们探讨了在遵循异步指令者/指令跟随者格式的任务导向对话中，大型语言模型提出澄清问题的能力。我们展示了将Minecraft对话语料库中两个现有注释合并为一个常见格式的新语料库，这两个注释分别关注引用和歧义，以及序参式话语结构分析（SDRT）包括澄清信息，从而为实验澄清及其与歧义的关系提供了必要的信息。借助这一语料库，我们将大型语言模型的行为与原始的人工生成的澄清问题进行比较，考察人类和大型语言模型在存在歧义情况下的行为。我们发现，在这些对话中，歧义与人类提出澄清问题之间只有微弱的联系，且人类与大型语言模型之间相关性较低。人类几乎不为引用歧义生产澄清问题，但经常为任务不确定性提出澄清问题。相反，大型语言模型更常为引用歧义提出澄清问题，但对任务不确定性则较少提出澄清问题。我们质疑大型语言模型提出澄清问题的能力是否依赖于它们最近的推理模拟能力，并通过使用不同的推理方法进行测试，发现推理确实增加了提问的频率和相关性。 

---
# Response Wide Shut? Surprising Observations in Basic Vision Language Model Capabilities 

**Title (ZH)**: Wide Shut？基本视觉语言模型能力的惊讶观察 

**Authors**: Shivam Chandhok, Wan-Cyuan Fan, Vered Shwartz, Vineeth N Balasubramanian, Leonid Sigal  

**Link**: [PDF](https://arxiv.org/pdf/2507.10442)  

**Abstract**: Vision-language Models (VLMs) have emerged as general-purpose tools for addressing a variety of complex computer vision problems. Such models have been shown to be highly capable, but, at the same time, lacking some basic visual understanding skills. In this paper, we set out to understand the limitations of SoTA VLMs on fundamental visual tasks by constructing a series of tests that probe which components of design, specifically, may be lacking. Importantly, we go significantly beyond the current benchmarks, which simply measure the final performance of VLM response, by also comparing and contrasting it to the performance of probes trained directly on features obtained from the visual encoder, intermediate vision-language projection and LLM-decoder output. In doing so, we uncover shortcomings in VLMs and make a number of important observations about their capabilities, robustness and how they process visual information. We hope our insights will guide progress in further improving VLMs. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）已成为解决各种复杂计算机视觉问题的一般工具。这类模型已被证明具有高度的能力，但同时在一些基本的视觉理解技能方面却显得不足。在本文中，我们通过构建一系列测试来理解最先进的VLM在基本视觉任务上的局限性，具体探查哪些设计方案可能缺失。重要的是，我们超越了现有的基准测试，不仅衡量VLM响应的最终性能，还将其与直接在视觉编码器特征、中间的视觉-语言投影以及大语言模型解码器输出上训练的探针的性能进行比较和对照。通过这种方式，我们揭示了VLM的不足，并对其能力、稳健性以及处理视觉信息的方式提出了若干重要观察。我们希望我们的见解能指导进一步改进VLM的工作。 

---
# From Sequence to Structure: Uncovering Substructure Reasoning in Transformers 

**Title (ZH)**: 从序列到结构：揭示Transformer中的子结构推理 

**Authors**: Xinnan Dai, Kai Yang, Jay Revolinsky, Kai Guo, Aoran Wang, Bohang Zhang, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10435)  

**Abstract**: Recent studies suggest that large language models (LLMs) possess the capability to solve graph reasoning tasks. Notably, even when graph structures are embedded within textual descriptions, LLMs can still effectively answer related questions. This raises a fundamental question: How can a decoder-only Transformer architecture understand underlying graph structures? To address this, we start with the substructure extraction task, interpreting the inner mechanisms inside the transformers and analyzing the impact of the input queries. Specifically, through both empirical results and theoretical analysis, we present Induced Substructure Filtration (ISF), a perspective that captures the substructure identification in the multi-layer transformers. We further validate the ISF process in LLMs, revealing consistent internal dynamics across layers. Building on these insights, we explore the broader capabilities of Transformers in handling diverse graph types. Specifically, we introduce the concept of thinking in substructures to efficiently extract complex composite patterns, and demonstrate that decoder-only Transformers can successfully extract substructures from attributed graphs, such as molecular graphs. Together, our findings offer a new insight on how sequence-based Transformers perform the substructure extraction task over graph data. 

**Abstract (ZH)**: 近年来的研究表明，大规模语言模型（LLMs）具备解决图推理任务的能力。即使在文本描述中嵌入了图结构，LLMs也能有效地回答相关问题。这引发了一个基本问题：仅解码器的Transformer架构是如何理解底层图结构的？为了解决这一问题，我们从子结构提取任务出发，解读Transformer内部机制，并分析输入查询的影响。通过实证结果和理论分析，我们提出了诱导子结构过滤（Induced Substructure Filtration, ISF）观点，该观点捕捉了多层Transformer中的子结构识别。我们进一步验证了ISF过程在LLMs中的表现，揭示了各层中一致性内部动态。基于这些洞见，我们探讨了Transformer在处理不同类型的图时的更广泛能力。具体来说，我们引入了子结构思考的概念，以高效地提取复杂的组合模式，并证明了仅解码器的Transformer可以从属性图，如分子图中成功提取子结构。我们的发现共同为我们提供了序列基Transformer在图数据上执行子结构提取任务的新见解。 

---
# Efficient Federated Learning with Heterogeneous Data and Adaptive Dropout 

**Title (ZH)**: 异质数据下的高效联邦学习与自适应失活 

**Authors**: Ji Liu, Beichen Ma, Yang Zhou, Jingbo Zhou, Ruoming Jin, Dejing Dou, Huaiyu Dai, Haixun Wang, Patrick Valduriez  

**Link**: [PDF](https://arxiv.org/pdf/2507.10430)  

**Abstract**: Federated Learning (FL) is a promising distributed machine learning approach that enables collaborative training of a global model using multiple edge devices. The data distributed among the edge devices is highly heterogeneous. Thus, FL faces the challenge of data distribution and heterogeneity, where non-Independent and Identically Distributed (non-IID) data across edge devices may yield in significant accuracy drop. Furthermore, the limited computation and communication capabilities of edge devices increase the likelihood of stragglers, thus leading to slow model convergence. In this paper, we propose the FedDHAD FL framework, which comes with two novel methods: Dynamic Heterogeneous model aggregation (FedDH) and Adaptive Dropout (FedAD). FedDH dynamically adjusts the weights of each local model within the model aggregation process based on the non-IID degree of heterogeneous data to deal with the statistical data heterogeneity. FedAD performs neuron-adaptive operations in response to heterogeneous devices to improve accuracy while achieving superb efficiency. The combination of these two methods makes FedDHAD significantly outperform state-of-the-art solutions in terms of accuracy (up to 6.7% higher), efficiency (up to 2.02 times faster), and computation cost (up to 15.0% smaller). 

**Abstract (ZH)**: Federated Learning框架FedDHAD：动态异质模型聚合与自适应丢弃 

---
# Multiple Choice Learning of Low Rank Adapters for Language Modeling 

**Title (ZH)**: 低秩适配器的多选学习语言建模 

**Authors**: Victor Letzelter, Hugo Malard, Mathieu Fontaine, Gaël Richard, Slim Essid, Andrei Bursuc, Patrick Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2507.10419)  

**Abstract**: We propose LoRA-MCL, a training scheme that extends next-token prediction in language models with a method designed to decode diverse, plausible sentence continuations at inference time. Traditional language modeling is an intrinsically ill-posed problem: given a context, multiple futures may be equally plausible. Our approach leverages Multiple Choice Learning (MCL) and the Winner-Takes-All (WTA) loss to efficiently handle ambiguity through Low-Rank Adaptation (LoRA). We provide a theoretical interpretation of applying Multiple Choice Learning to Language Modeling, assuming the data is generated from a mixture of distributions. To illustrate the proposed approach, we use data sampled from mixtures of Markov chains. We then demonstrate with extensive experiments on real-world visual and audio captioning tasks that our method achieves high diversity and relevance in generated outputs. 

**Abstract (ZH)**: LoRA-MCL: 一种通过低秩适应扩展语言模型下一个词预测的训练方案，该方案在推断时能够高效解码多样且合理的句子延续。 

---
# Energy Efficiency in AI for 5G and Beyond: A DeepRx Case Study 

**Title (ZH)**: AI在5G及更 beyond 的能效研究：DeepRx案例分析 

**Authors**: Amine Lbath, Ibtissam Labriji  

**Link**: [PDF](https://arxiv.org/pdf/2507.10409)  

**Abstract**: This study addresses the challenge of balancing energy efficiency with performance in AI/ML models, focusing on DeepRX, a deep learning receiver based on a fully convolutional ResNet architecture. We evaluate the energy consumption of DeepRX, considering factors including FLOPs/Watt and FLOPs/clock, and find consistency between estimated and actual energy usage, influenced by memory access patterns. The research extends to comparing energy dynamics during training and inference phases. A key contribution is the application of knowledge distillation (KD) to train a compact DeepRX \textit{student} model that emulates the performance of the \textit{teacher} model but with reduced energy consumption. We experiment with different student model sizes, optimal teacher sizes, and KD hyperparameters. Performance is measured by comparing the Bit Error Rate (BER) performance versus Signal-to-Interference \& Noise Ratio (SINR) values of the distilled model and a model trained from scratch. The distilled models demonstrate a lower error floor across SINR levels, highlighting the effectiveness of KD in achieving energy-efficient AI solutions. 

**Abstract (ZH)**: 本研究探讨了在AI/ML模型中平衡能效与性能的挑战，重点关注基于全卷积ResNet架构的DeepRX深度学习接收机。我们评估了DeepRX的能效，考虑了每瓦浮点运算次数(FLOPs/Watt)和每时钟周期浮点运算次数(FLOPs/clock)等因素，并发现了估算的能耗与实际能耗之间的一致性，受内存访问模式的影响。研究还扩展到了训练和推理阶段能效动态的比较。一个主要贡献是应用知识蒸馏(KD)训练一个紧凑的DeepRX学生模型，该模型在能耗降低的情况下模拟了教师模型的性能。我们实验了不同的学生模型大小、最优教师模型大小以及KD超参数。性能通过比较蒸馏模型和从头开始训练的模型的比特错误率(BER)性能与信号干扰与噪声比(SINR)值来进行衡量。蒸馏模型在SINR值不同水平上显示出较低的错误底限，突显了KD在实现能效AI解决方案方面的有效性。 

---
# Devanagari Handwritten Character Recognition using Convolutional Neural Network 

**Title (ZH)**: 基于卷积神经网络的Devanagari手写字符识别 

**Authors**: Diksha Mehta, Prateek Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2507.10398)  

**Abstract**: Handwritten character recognition is getting popular among researchers because of its possible applications in facilitating technological search engines, social media, recommender systems, etc. The Devanagari script is one of the oldest language scripts in India that does not have proper digitization tools. With the advancement of computing and technology, the task of this research is to extract handwritten Hindi characters from an image of Devanagari script with an automated approach to save time and obsolete data. In this paper, we present a technique to recognize handwritten Devanagari characters using two deep convolutional neural network layers. This work employs a methodology that is useful to enhance the recognition rate and configures a convolutional neural network for effective Devanagari handwritten text recognition (DHTR). This approach uses the Devanagari handwritten character dataset (DHCD), an open dataset with 36 classes of Devanagari characters. Each of these classes has 1700 images for training and testing purposes. This approach obtains promising results in terms of accuracy by achieving 96.36% accuracy in testing and 99.55% in training time. 

**Abstract (ZH)**: 手写字符识别由于其在促进技术搜索引擎、社交媒体、推荐系统等方面的应用而日益受到研究者的关注。印度的德文加班字符是其中一种古老的文字体系，缺乏相应的数字化工具。随着计算和科技的进步，本研究的任务是采用自动化方法从德文加班字体图像中提取手写印地文字符，节省时间和避免过时数据。本文提出了一种使用两层深度卷积神经网络层识别手写德文加班字符的技术。该工作采用了一种有助于提高识别率的方法，并配置了一个用于有效识别手写德文文本（DHTR）的卷积神经网络。该方法使用了德文加班手写字符数据集（DHCD），这是一个包含36类德文加班字符的开放数据集，每类有1700张用于训练和测试的图像。该方法在测试和训练时间分别获得了96.36%和99.55%的准确率，取得了令人鼓舞的结果。 

---
# TAT: Temporal-Aligned Transformer for Multi-Horizon Peak Demand Forecasting 

**Title (ZH)**: TAT：时间对齐变换器在多 horizon 尖峰需求预测中的应用 

**Authors**: Zhiyuan Zhao, Sitan Yang, Kin G. Olivares, Boris N. Oreshkin, Stan Vitebsky, Michael W. Mahoney, B. Aditya Prakash, Dmitry Efimov  

**Link**: [PDF](https://arxiv.org/pdf/2507.10349)  

**Abstract**: Multi-horizon time series forecasting has many practical applications such as demand forecasting. Accurate demand prediction is critical to help make buying and inventory decisions for supply chain management of e-commerce and physical retailers, and such predictions are typically required for future horizons extending tens of weeks. This is especially challenging during high-stake sales events when demand peaks are particularly difficult to predict accurately. However, these events are important not only for managing supply chain operations but also for ensuring a seamless shopping experience for customers. To address this challenge, we propose Temporal-Aligned Transformer (TAT), a multi-horizon forecaster leveraging apriori-known context variables such as holiday and promotion events information for improving predictive performance. Our model consists of an encoder and decoder, both embedded with a novel Temporal Alignment Attention (TAA), designed to learn context-dependent alignment for peak demand forecasting. We conduct extensive empirical analysis on two large-scale proprietary datasets from a large e-commerce retailer. We demonstrate that TAT brings up to 30% accuracy improvement on peak demand forecasting while maintaining competitive overall performance compared to other state-of-the-art methods. 

**Abstract (ZH)**: 多 horizons 时间序列预测在需求预测等领域有许多实际应用。在电子商务和实体零售商的供应链管理中，准确的需求预测对于购买和库存决策至关重要，通常需要对未来几周进行预测。尤其是对于高风险销售事件，在需求峰值预测方面更具挑战性。然而，这些事件不仅对供应链操作管理至关重要，也对确保顺畅的购物体验至关重要。为此，我们提出了一种时空对齐变换器（TAT），它利用先验已知的上下文变量（如节假日和促销活动信息）来提高预测性能。我们的模型由编码器和解码器组成，两者都嵌入了一种新型的时空对齐注意力机制（TAA），旨在学习上下文相关的对齐以进行峰值需求预测。我们在一家大型电子商务零售商的两个大规模专有数据集上进行了广泛的经验分析。结果显示，TAT 在峰值需求预测上的准确率最多可提高 30%，同时在总体性能上仍然保持与其他最先进的方法相当。 

---
# Feature Distillation is the Better Choice for Model-Heterogeneous Federated Learning 

**Title (ZH)**: 特征精炼是模型异构联邦学习的更好选择 

**Authors**: Yichen Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.10348)  

**Abstract**: Model-Heterogeneous Federated Learning (Hetero-FL) has attracted growing attention for its ability to aggregate knowledge from heterogeneous models while keeping private data locally. To better aggregate knowledge from clients, ensemble distillation, as a widely used and effective technique, is often employed after global aggregation to enhance the performance of the global model. However, simply combining Hetero-FL and ensemble distillation does not always yield promising results and can make the training process unstable. The reason is that existing methods primarily focus on logit distillation, which, while being model-agnostic with softmax predictions, fails to compensate for the knowledge bias arising from heterogeneous models. To tackle this challenge, we propose a stable and efficient Feature Distillation for model-heterogeneous Federated learning, dubbed FedFD, that can incorporate aligned feature information via orthogonal projection to integrate knowledge from heterogeneous models better. Specifically, a new feature-based ensemble federated knowledge distillation paradigm is proposed. The global model on the server needs to maintain a projection layer for each client-side model architecture to align the features separately. Orthogonal techniques are employed to re-parameterize the projection layer to mitigate knowledge bias from heterogeneous models and thus maximize the distilled knowledge. Extensive experiments show that FedFD achieves superior performance compared to state-of-the-art methods. 

**Abstract (ZH)**: 模型异构联邦学习中的稳定高效特征蒸馏（FedFD） 

---
# Toolsuite for Implementing Multiagent Systems Based on Communication Protocols 

**Title (ZH)**: 基于通信协议实现多agent系统的工具套件 

**Authors**: Amit K. Chopra, Samuel H. Christie V, Munindar P. Singh  

**Link**: [PDF](https://arxiv.org/pdf/2507.10324)  

**Abstract**: Interaction-Oriented Programming (IOP) is an approach to building a multiagent system by modeling the interactions between its roles via a flexible interaction protocol and implementing agents to realize the interactions of the roles they play in the protocol.
In recent years, we have developed an extensive suite of software that enables multiagent system developers to apply IOP. These include tools for efficiently verifying protocols for properties such as liveness and safety and middleware that simplifies the implementation of agents. This paper presents some of that software suite. 

**Abstract (ZH)**: 面向交互的编程（IOP）是一种通过使用灵活的交互协议模型化的角色之间交互来构建多智能体系统的方法，并实现执行协议中角色交互的智能体。近年来，我们开发了一整套软件工具，使多智能体系统开发者能够应用IOP。这些工具包括用于高效验证具有活锁和安全性等性质的协议的工具以及简化智能体实现的中间件。本文介绍了其中的一些软件工具。 

---
# Recognizing Dementia from Neuropsychological Tests with State Space Models 

**Title (ZH)**: 基于态空模型的痴呆识别研究——从神经心理学测试着手 

**Authors**: Liming Wang, Saurabhchand Bhati, Cody Karjadi, Rhoda Au, James Glass  

**Link**: [PDF](https://arxiv.org/pdf/2507.10311)  

**Abstract**: Early detection of dementia is critical for timely medical intervention and improved patient outcomes. Neuropsychological tests are widely used for cognitive assessment but have traditionally relied on manual scoring. Automatic dementia classification (ADC) systems aim to infer cognitive decline directly from speech recordings of such tests. We propose Demenba, a novel ADC framework based on state space models, which scale linearly in memory and computation with sequence length. Trained on over 1,000 hours of cognitive assessments administered to Framingham Heart Study participants, some of whom were diagnosed with dementia through adjudicated review, our method outperforms prior approaches in fine-grained dementia classification by 21\%, while using fewer parameters. We further analyze its scaling behavior and demonstrate that our model gains additional improvement when fused with large language models, paving the way for more transparent and scalable dementia assessment tools. Code: this https URL 

**Abstract (ZH)**: early detection of dementia is critical for timely medical intervention and improved patient outcomes. 基于状态空间模型的Demenba：一种线性扩展的自动痴呆分类框架，实现了更细粒度的分类性能提升与参数减少。进一步分析其扩展行为，并展示了将该模型与大型语言模型融合后获得了额外的提升，为更透明和可扩展的痴呆评估工具铺平了道路。 

---
# FaceLLM: A Multimodal Large Language Model for Face Understanding 

**Title (ZH)**: FaceLLM：一种用于面部理解的多模态大型语言模型 

**Authors**: Hatef Otroshi Shahreza, Sébastien Marcel  

**Link**: [PDF](https://arxiv.org/pdf/2507.10300)  

**Abstract**: Multimodal large language models (MLLMs) have shown remarkable performance in vision-language tasks. However, existing MLLMs are primarily trained on generic datasets, limiting their ability to reason on domain-specific visual cues such as those in facial images. In particular, tasks that require detailed understanding of facial structure, expression, emotion, and demographic features remain underexplored by MLLMs due to the lack of large-scale annotated face image-text datasets. In this work, we introduce FaceLLM, a multimodal large language model trained specifically for facial image understanding. To construct the training data, we propose a novel weakly supervised pipeline that uses ChatGPT with attribute-aware prompts to generate high-quality question-answer pairs based on images from the FairFace dataset. The resulting corpus, called FairFaceGPT, covers a diverse set of attributes including expression, pose, skin texture, and forensic information. Our experiments demonstrate that FaceLLM improves the performance of MLLMs on various face-centric tasks and achieves state-of-the-art performance. This work highlights the potential of synthetic supervision via language models for building domain-specialized MLLMs, and sets a precedent for trustworthy, human-centric multimodal AI systems. FairFaceGPT dataset and pretrained FaceLLM models are publicly available in the project page. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）在视觉-语言任务中展现了卓越的表现。然而，现有的MLLMs主要在通用数据集上进行训练，限制了它们在面部图像等领域特定视觉线索上的推理能力。特别是那些需要详细了解面部结构、表情、情感以及人口统计特征的任务，由于缺乏大规模注解的人脸图像-文本数据集，MLLMs仍处于未被充分探索的状态。在本工作中，我们介绍了FaceLLM，这是一种专门用于面部图像理解的多模态大语言模型。为了构建训练数据，我们提出了一种新的弱监督管道，利用带有属性感知提示的ChatGPT生成基于FairFace数据集图像的高质量问题-答案对。由此产生的语料库称为FairFaceGPT，涵盖了表情、姿态、皮肤纹理和法医信息等多种属性。我们的实验表明，FaceLLM在各种以人脸为中心的任务上提高了MLLMs的表现，并达到了最先进的性能。本工作突显了通过语言模型合成监督构建领域专用的MLLMs的潜力，并为值得信赖、以人为中心的多模态AI系统树立了先例。FairFaceGPT数据集和预训练的FaceLLM模型已在项目页面上公开。 

---
# DepViT-CAD: Deployable Vision Transformer-Based Cancer Diagnosis in Histopathology 

**Title (ZH)**: DepViT-CAD: 可部署的基于视觉变换器的病理癌症诊断方法 

**Authors**: Ashkan Shakarami, Lorenzo Nicole, Rocco Cappellesso, Angelo Paolo Dei Tos, Stefano Ghidoni  

**Link**: [PDF](https://arxiv.org/pdf/2507.10250)  

**Abstract**: Accurate and timely cancer diagnosis from histopathological slides is vital for effective clinical decision-making. This paper introduces DepViT-CAD, a deployable AI system for multi-class cancer diagnosis in histopathology. At its core is MAViT, a novel Multi-Attention Vision Transformer designed to capture fine-grained morphological patterns across diverse tumor types. MAViT was trained on expert-annotated patches from 1008 whole-slide images, covering 11 diagnostic categories, including 10 major cancers and non-tumor tissue. DepViT-CAD was validated on two independent cohorts: 275 WSIs from The Cancer Genome Atlas and 50 routine clinical cases from pathology labs, achieving diagnostic sensitivities of 94.11% and 92%, respectively. By combining state-of-the-art transformer architecture with large-scale real-world validation, DepViT-CAD offers a robust and scalable approach for AI-assisted cancer diagnostics. To support transparency and reproducibility, software and code will be made publicly available at GitHub. 

**Abstract (ZH)**: 准确及时地从组织病理切片中进行癌症诊断对于有效的临床决策至关重要。本文介绍了一种可部署的AI系统DepViT-CAD，用于组织病理学中的多类别癌症诊断。其核心是MAViT，这是一种新的多注意力视觉变换器，旨在捕捉不同肿瘤类型中的细微形态模式。MAViT基于1008张全切片图像上的专家注释片段训练，涵盖11个诊断类别，包括10种主要癌症和非肿瘤组织。DepViT-CAD分别在The Cancer Genome Atlas的两个独立队列（275张WSI）和病理实验室的50例常规临床病例中进行了验证，诊断灵敏度分别为94.11%和92%。通过结合最先进的变换器架构和大规模现实世界验证，DepViT-CAD提供了一种稳健且可扩展的人工智能辅助癌症诊断方法。为支持透明性和可重复性，软件和代码将公开发布在GitHub上。 

---
# Visual Analytics for Explainable and Trustworthy Artificial Intelligence 

**Title (ZH)**: 可解释和可信赖的人工智能的可视化分析 

**Authors**: Angelos Chatzimparmpas  

**Link**: [PDF](https://arxiv.org/pdf/2507.10240)  

**Abstract**: Our society increasingly depends on intelligent systems to solve complex problems, ranging from recommender systems suggesting the next movie to watch to AI models assisting in medical diagnoses for hospitalized patients. With the iterative improvement of diagnostic accuracy and efficiency, AI holds significant potential to mitigate medical misdiagnoses by preventing numerous deaths and reducing an economic burden of approximately 450 EUR billion annually. However, a key obstacle to AI adoption lies in the lack of transparency: many automated systems function as "black boxes," providing predictions without revealing the underlying processes. This opacity can hinder experts' ability to trust and rely on AI systems. Visual analytics (VA) provides a compelling solution by combining AI models with interactive visualizations. These specialized charts and graphs empower users to incorporate their domain expertise to refine and improve the models, bridging the gap between AI and human understanding. In this work, we define, categorize, and explore how VA solutions can foster trust across the stages of a typical AI pipeline. We propose a design space for innovative visualizations and present an overview of our previously developed VA dashboards, which support critical tasks within the various pipeline stages, including data processing, feature engineering, hyperparameter tuning, understanding, debugging, refining, and comparing models. 

**Abstract (ZH)**: 我们的社会日益依赖智能系统来解决复杂问题，从推荐系统建议观看的下一部电影到辅助住院患者进行医学诊断的人工智能模型。随着诊断准确性和效率的迭代提升，人工智能在预防大量死亡并减轻约4500亿欧元的经济负担方面具有巨大的潜力。然而，人工智能采纳的关键障碍在于透明度的缺乏：许多自动化系统充当“黑箱”，提供预测而不揭示内部过程。这种不透明性可能会阻碍专家对人工智能系统的信任和依赖。视觉分析（VA）通过结合人工智能模型与互动可视化提供了引人注目的解决方案。这些专门的图表和图形使用户能够结合其领域专业知识来完善和改进模型，弥合人工智能与人类理解之间的差距。在本研究中，我们定义、分类并探讨VA解决方案如何在典型人工智能管道的各个阶段促进信任。我们提出了创新可视化的设计空间，并概述了我们之前开发的VA仪表板，这些仪表板支持各种管道阶段内的关键任务，包括数据处理、特征工程、超参数调整、理解、调试、完善和模型比较。 

---
# ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users 

**Title (ZH)**: ProGait: 一种适用于股膝置换假肢使用者的多功能视频数据集和基准 

**Authors**: Xiangyu Yin, Boyuan Yang, Weichen Liu, Qiyao Xue, Abrar Alamri, Goeran Fiedler, Wei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.10223)  

**Abstract**: Prosthetic legs play a pivotal role in clinical rehabilitation, allowing individuals with lower-limb amputations the ability to regain mobility and improve their quality of life. Gait analysis is fundamental for optimizing prosthesis design and alignment, directly impacting the mobility and life quality of individuals with lower-limb amputations. Vision-based machine learning (ML) methods offer a scalable and non-invasive solution to gait analysis, but face challenges in correctly detecting and analyzing prosthesis, due to their unique appearances and new movement patterns. In this paper, we aim to bridge this gap by introducing a multi-purpose dataset, namely ProGait, to support multiple vision tasks including Video Object Segmentation, 2D Human Pose Estimation, and Gait Analysis (GA). ProGait provides 412 video clips from four above-knee amputees when testing multiple newly-fitted prosthetic legs through walking trials, and depicts the presence, contours, poses, and gait patterns of human subjects with transfemoral prosthetic legs. Alongside the dataset itself, we also present benchmark tasks and fine-tuned baseline models to illustrate the practical application and performance of the ProGait dataset. We compared our baseline models against pre-trained vision models, demonstrating improved generalizability when applying the ProGait dataset for prosthesis-specific tasks. Our code is available at this https URL and dataset at this https URL. 

**Abstract (ZH)**: 假肢在临床康复中扮演着关键角色，使下肢缺失者能够恢复 mobility 并提高生活质量。步态分析对于优化假肢设计和对齐至关重要，直接影响下肢缺失者的生活质量和移动能力。基于视觉的机器学习方法为步态分析提供了可扩展且非侵入性的解决方案，但在正确检测和分析假肢方面面临挑战，因为假肢具有独特的外观和新的运动模式。在本文中，我们通过引入名为 ProGait 的多功能数据集来弥补这一差距，该数据集支持包括视频对象分割、2D 人体姿态估计和步态分析在内的多种视觉任务。ProGait 提供了 412 个视频片段，记录了四位膝上截肢者在多次佩戴新装假肢行走试验中的人类主体的假肢存在、轮廓、姿态和步态模式。除了数据集本身，我们还提供基准任务和微调基线模型来说明 ProGait 数据集的实际应用和性能。我们将基准模型与预训练视觉模型进行了比较，展示了使用 ProGait 数据集进行假肢特定任务时的一般泛化能力提高。我们的代码可在以下网址访问：this https URL，数据集可在以下网址访问：this https URL。 

---
# Absher: A Benchmark for Evaluating Large Language Models Understanding of Saudi Dialects 

**Title (ZH)**: Absher：评估大型语言模型对阿拉伯语沙特方言理解能力的标准基准 

**Authors**: Renad Al-Monef, Hassan Alhuzali, Nora Alturayeif, Ashwag Alasmari  

**Link**: [PDF](https://arxiv.org/pdf/2507.10216)  

**Abstract**: As large language models (LLMs) become increasingly central to Arabic NLP applications, evaluating their understanding of regional dialects and cultural nuances is essential, particularly in linguistically diverse settings like Saudi Arabia. This paper introduces \texttt{Absher}, a comprehensive benchmark specifically designed to assess LLMs performance across major Saudi dialects. \texttt{Absher} comprises over 18,000 multiple-choice questions spanning six distinct categories: Meaning, True/False, Fill-in-the-Blank, Contextual Usage, Cultural Interpretation, and Location Recognition. These questions are derived from a curated dataset of dialectal words, phrases, and proverbs sourced from various regions of Saudi Arabia. We evaluate several state-of-the-art LLMs, including multilingual and Arabic-specific models. We also provide detailed insights into their capabilities and limitations. Our results reveal notable performance gaps, particularly in tasks requiring cultural inference or contextual understanding. Our findings highlight the urgent need for dialect-aware training and culturally aligned evaluation methodologies to improve LLMs performance in real-world Arabic applications. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在阿拉伯语NLP应用中变得日益重要，评估其对地区方言和文化微妙差异的理解能力尤为重要，特别是在像沙特阿拉伯这样语言多样化的环境中。本文介绍了\texttt{Absher}，一个专门用于评估LLMs在主要沙特方言上的表现的综合基准。\texttt{Absher}包含超过18,000个选择题，涵盖六个不同的类别：意义、真/假判断、填空、语境用法、文化解释和地理位置识别。这些问题来源于沙特阿拉伯不同地区精心编制的方言词汇、短语和谚语数据集。我们评估了几种最先进的LLMs，包括多语言和阿拉伯语特定模型，并提供了对其能力和局限性的详细洞见。我们的结果揭示了在需要文化推理或上下文理解的任务中存在显著的表现差距。我们的发现强调了在实际阿拉伯语应用中提高LLMs性能的迫切需求，需要方言感知的训练和文化对齐的评估方法。 

---
# A Training-Free, Task-Agnostic Framework for Enhancing MLLM Performance on High-Resolution Images 

**Title (ZH)**: 一种无需训练、任务无关的框架，用于提升MLLM在高分辨率图像上的性能 

**Authors**: Jaeseong Lee, Yeeun Choi, Heechan Choi, Hanjung Kim, Seonjoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.10202)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in vision-language understanding, reasoning, and generation. However, they struggle with tasks requiring fine-grained localization and reasoning in high-resolution images. This constraint stems from the fact that MLLMs are fine-tuned with fixed image resolution to align with the pre-trained image encoder used in MLLM. Consequently, feeding high-resolution images directly into MLLMs leads to poor generalization due to a train-test resolution discrepancy, while downsampling these images-although ensuring consistency-compromises fine-grained visual details and ultimately degrades performance. To address this challenge, we propose Extract Candidate then Predict (ECP), a novel training-free, task-agnostic two-stage framework designed to enhance MLLM performance on high-resolution images. The key intuition behind ECP is that while MLLMs struggle with high-resolution images, their predictions on downsampled images still contain implicit localization cues. By first identifying candidate region using the coarse prediction and then predicting the final output based on candidate region, ECP effectively preserves fine-grained details while mitigating the challenges posed by high-resolution data. We validate our framework on 4K GUI grounding and 4K, 8K MLLM perception, achieving +21.3%, +5.8%, +5.2% absolute improvement compared to baseline respectively, demonstrating its effectiveness. Code is available at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在视觉-语言理解、推理和生成方面展现了显著的能力。然而，它们在处理要求精细定位和高分辨率图像中的精细推理的任务时显得力不从心。这一局限源自MLLMs在固定分辨率的图像上进行微调，以与MLLM中使用的预训练图像编码器相匹配。因此，直接将高分辨率图像输入到MLLMs中会导致由于训练与测试分辨率不一致而导致泛化效果差，而对这些图像进行下采样虽然保证了一致性，但会牺牲精细的视觉细节，最终降低性能。为解决这一挑战，我们提出了一种基于提取候选区域然后预测（ECP）的新型训练无介入、任务无关的两阶段框架，以增强MLLM在高分辨率图像上的性能。ECP的基本思想是，尽管MLLMs在处理高分辨率图像时遇到困难，但它们在下采样图像上的预测仍然包含隐式的定位线索。通过首先利用粗预测识别候选区域，然后基于候选区域预测最终输出，ECP有效地保留了详细的视觉细节，同时减轻了高分辨率数据带来的挑战。我们在4K GUI接地和4K、8K MLLM感知方面验证了该框架，分别获得了21.3%、5.8%、5.2%的绝对性能提升，证明了其有效性。代码可在此处访问：这个https URL。 

---
# Natural Language-based Assessment of L2 Oral Proficiency using LLMs 

**Title (ZH)**: 基于自然语言的二语口语 proficiency 评估方法使用大语言模型 

**Authors**: Stefano Bannò, Rao Ma, Mengjie Qian, Siyuan Tang, Kate Knill, Mark Gales  

**Link**: [PDF](https://arxiv.org/pdf/2507.10200)  

**Abstract**: Natural language-based assessment (NLA) is an approach to second language assessment that uses instructions - expressed in the form of can-do descriptors - originally intended for human examiners, aiming to determine whether large language models (LLMs) can interpret and apply them in ways comparable to human assessment. In this work, we explore the use of such descriptors with an open-source LLM, Qwen 2.5 72B, to assess responses from the publicly available S&I Corpus in a zero-shot setting. Our results show that this approach - relying solely on textual information - achieves competitive performance: while it does not outperform state-of-the-art speech LLMs fine-tuned for the task, it surpasses a BERT-based model trained specifically for this purpose. NLA proves particularly effective in mismatched task settings, is generalisable to other data types and languages, and offers greater interpretability, as it is grounded in clearly explainable, widely applicable language descriptors. 

**Abstract (ZH)**: 基于自然语言的评估（NLA）是第二种语言评估的一种方法，它使用表达为能够做某事描述的语言指令， originally intended for人类考评员，旨在确定大型语言模型（LLMs）是否能够以与人类评估相媲美的方式解释和应用这些指令。在本文中，我们探索了使用这种描述符与开源LLM Qwen 2.5 72B来在零样本设置中评估来自公开可用的S&I语料库的响应。我们的结果表明，这种方法——仅依赖文本信息——表现出了竞争力：尽管它不能超越针对该任务微调的最佳语音LLM，但其性能却超过了专门为该目的训练的基于BERT的模型。NLA在不匹配的任务设置中特别有效，可以泛化到其他数据类型和语言，并提供了更高的可解释性，因为它基于清晰可解释、广泛适用的语言描述。 

---
# Learning Private Representations through Entropy-based Adversarial Training 

**Title (ZH)**: 基于熵的对抗训练学习隐私表示 

**Authors**: Tassilo Klein, Moin Nabi  

**Link**: [PDF](https://arxiv.org/pdf/2507.10194)  

**Abstract**: How can we learn a representation with high predictive power while preserving user privacy? We present an adversarial representation learning method for sanitizing sensitive content from the learned representation. Specifically, we introduce a variant of entropy - focal entropy, which mitigates the potential information leakage of the existing entropy-based approaches. We showcase feasibility on multiple benchmarks. The results suggest high target utility at moderate privacy leakage. 

**Abstract (ZH)**: 如何在保留用户隐私的同时学习具有高预测能力的表示？我们提出了一种对抗表示学习方法，用于清理学习到的表示中的敏感内容。具体而言，我们引入了一种熵的变体——焦点熵，这可以减轻现有基于熵方法潜在的信息泄漏问题。我们在多个基准上展示了其实现可行性。结果表明，在适度的隐私泄漏下，可以实现较高的目标利用率。 

---
# Breaking the Myth: Can Small Models Infer Postconditions Too? 

**Title (ZH)**: 打破迷思：小型模型能否推断后条件？ 

**Authors**: Gehao Zhang, Zhenting Wang, Juan Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2507.10182)  

**Abstract**: Formal specifications are essential for ensuring software correctness, yet manually writing them is tedious and error-prone. Large Language Models (LLMs) have shown promise in generating such specifications from natural language intents, but the giant model size and high computational demands raise a fundamental question: Do we really need large models for this task? In this paper, we show that a small, fine-tuned language model can achieve high-quality postcondition generation with much lower computational costs. We construct a specialized dataset of prompts, reasoning logs, and postconditions, then supervise the fine-tuning of a $7$B-parameter code model. Our approach tackles real-world repository dependencies and preserves pre-state information, allowing for expressive and accurate specifications. We evaluate the model on a benchmark of real-world Java bugs (Defects4J) and compare against both proprietary giants (e.g., GPT-4o) and open-source large models. Empirical results demonstrate that our compact model matches or outperforms significantly larger counterparts in syntax correctness, semantic correctness, and bug-distinguishing capability. These findings highlight that targeted fine-tuning on a modest dataset can enable small models to achieve results formerly seen only in massive, resource-heavy LLMs, offering a practical and efficient path for the real-world adoption of automated specification generation. 

**Abstract (ZH)**: 小型微调语言模型在较低计算成本下实现高质量后条件生成的研究 

---
# The Second Machine Turn: From Checking Proofs to Creating Concepts 

**Title (ZH)**: 第二次机器革命：从验证证明到创建概念 

**Authors**: Asvin G  

**Link**: [PDF](https://arxiv.org/pdf/2507.10179)  

**Abstract**: We identify a second machine turn in the process of mathematical discovery: after automating proof-checking, AI is now poised to automate the *creation* of mathematical concepts themselves. We discuss the current state of the art, obstacles and potential solutions as well as a preliminary attempt at mathematizing the creation of concepts itself. The paper ends with an assessment of how these capabilities could reshape mathematics and human-machine collaboration, and a few different futures we might find ourselves in. 

**Abstract (ZH)**: 我们在数学发现过程中识别出第二个机器阶段：在 Automation of Proof-Checking 之后，AI 现在准备自动创造数学概念本身。我们讨论当前的技术水平、障碍和潜在解决方案，以及对概念创造本身进行数学化的一个初步尝试。文章结尾评估了这些能力如何重塑数学和人机协作，并设想了几种可能的未来。 

---
# Abusive text transformation using LLMs 

**Title (ZH)**: 使用大型语言模型进行虐待性文本转换 

**Authors**: Rohitash Chandra, Jiyong Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.10177)  

**Abstract**: Although Large Language Models (LLMs) have demonstrated significant advancements in natural language processing tasks, their effectiveness in the classification and transformation of abusive text into non-abusive versions remains an area for exploration. In this study, we aim to use LLMs to transform abusive text (tweets and reviews) featuring hate speech and swear words into non-abusive text, while retaining the intent of the text. We evaluate the performance of two state-of-the-art LLMs, such as Gemini, GPT-4o, DeekSeek and Groq, on their ability to identify abusive text. We them to transform and obtain a text that is clean from abusive and inappropriate content but maintains a similar level of sentiment and semantics, i.e. the transformed text needs to maintain its message. Afterwards, we evaluate the raw and transformed datasets with sentiment analysis and semantic analysis. Our results show Groq provides vastly different results when compared with other LLMs. We have identified similarities between GPT-4o and DeepSeek-V3. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）在自然语言处理任务中展示了显著的进步，但在 abusive 文本的分类和转化成非 abusive 版本方面的有效性仍需探索。本研究旨在使用 LLMs 将包含仇恨言论和污言秽语的 abusive 文本（推文和评论）转化为非 abusive 文本，同时保留文本的意图。我们评估了两种最先进的 LLMs（如 Gemini、GPT-4o、DeekSeek 和 Groq）在识别 abusive 文本方面的能力。我们让它们进行转化，以获得一份没有 abusive 和不适当内容但保持相似情感和语义水平的文本，即转化后的文本需要保留其信息。随后，我们使用情感分析和语义分析对原始和转化后的数据集进行评估。我们的结果显示，Groq 的结果与其它 LLMs 的结果相差甚远。我们发现 GPT-4o 和 DeepSeek-V3 之间存在相似之处。 

---
# Play Style Identification Using Low-Level Representations of Play Traces in MicroRTS 

**Title (ZH)**: 使用微RTS游戏轨迹的低级表示进行游戏风格识别 

**Authors**: Ruizhe Yu Xia, Jeremy Gow, Simon Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2507.10172)  

**Abstract**: Play style identification can provide valuable game design insights and enable adaptive experiences, with the potential to improve game playing agents. Previous work relies on domain knowledge to construct play trace representations using handcrafted features. More recent approaches incorporate the sequential structure of play traces but still require some level of domain abstraction. In this study, we explore the use of unsupervised CNN-LSTM autoencoder models to obtain latent representations directly from low-level play trace data in MicroRTS. We demonstrate that this approach yields a meaningful separation of different game playing agents in the latent space, reducing reliance on domain expertise and its associated biases. This latent space is then used to guide the exploration of diverse play styles within studied AI players. 

**Abstract (ZH)**: 基于无监督CNN-LSTM自编码器的MicroRTS游戏玩法风格识别 

---
# A PBN-RL-XAI Framework for Discovering a "Hit-and-Run'' Therapeutic Strategy in Melanoma 

**Title (ZH)**: 基于PBN-RL-XAI的黑色素瘤“访问-撤离”治疗策略发现框架 

**Authors**: Zhonglin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.10136)  

**Abstract**: Innate resistance to anti-PD-1 immunotherapy remains a major clinical challenge in metastatic melanoma, with the underlying molecular networks being poorly understood. To address this, we constructed a dynamic Probabilistic Boolean Network model using transcriptomic data from patient tumor biopsies to elucidate the regulatory logic governing therapy response. We then employed a reinforcement learning agent to systematically discover optimal, multi-step therapeutic interventions and used explainable artificial intelligence to mechanistically interpret the agent's control policy. The analysis revealed that a precisely timed, 4-step temporary inhibition of the lysyl oxidase like 2 protein (LOXL2) was the most effective strategy. Our explainable analysis showed that this ``hit-and-run" intervention is sufficient to erase the molecular signature driving resistance, allowing the network to self-correct without requiring sustained intervention. This study presents a novel, time-dependent therapeutic hypothesis for overcoming immunotherapy resistance and provides a powerful computational framework for identifying non-obvious intervention protocols in complex biological systems. 

**Abstract (ZH)**: 先天性对anti-PD-1免疫治疗的抵抗仍是转移性黑色素瘤临床治疗的主要挑战，其 underlying 分子网络尚不明确。为了解决这一问题，我们利用患者肿瘤活检的转录组数据构建了一个动态概率布尔网络模型，以阐明调控治疗反应的调节逻辑。然后，我们使用强化学习代理系统地发现最优的多步治疗干预措施，并利用可解释的人工智能来机制性地解释代理的控制策略。分析表明，精确时间安排的4步暂态抑制LOXL2蛋白是最有效的方法。我们的可解释分析显示，这种“击打即走”干预足以消除驱动抗性的分子标记，使网络能够自我校正，无需持续干预。该研究提出了一种新的、时间依赖性的治疗假设，以克服免疫治疗抗性，并提供了一种强大的计算框架，用于在复杂生物系统中识别非显性干预协议。 

---
# Extending Defeasibility for Propositional Standpoint Logics 

**Title (ZH)**: 扩展命题立场逻辑中的攻否性 

**Authors**: Nicholas Leisegang, Thomas Meyer, Ivan Varzinczak  

**Link**: [PDF](https://arxiv.org/pdf/2507.10133)  

**Abstract**: In this paper, we introduce a new defeasible version of propositional standpoint logic by integrating Kraus et al.'s defeasible conditionals, Britz and Varzinczak's notions of defeasible necessity and distinct possibility, along with Leisegang et al.'s approach to defeasibility into the standpoint logics of Gómez Álvarez and Rudolph. The resulting logical framework allows for the expression of defeasibility on the level of implications, standpoint modal operators, and standpoint-sharpening statements. We provide a preferential semantics for this extended language and propose a tableaux calculus, which is shown to be sound and complete with respect to preferential entailment. We also establish the computational complexity of the tableaux procedure to be in PSpace. 

**Abstract (ZH)**: 本文引入了一种新的命题立场逻辑的可败斥版本，通过整合Kraus等人提出的可败斥条件、Britz和Varzinczak提出的可败斥必然性和独立可能性概念以及Leisegang等人对可败斥性的处理方法，结合Gómez Álvarez和Rudolph的立场逻辑。 resulting logical framework 允许在推论、立场模态运算符和立场细化声明的层面表达可败斥性。我们为此扩展语言提供了优选语义，并提出了一种表格式计算法，该计算法相对于优选蕴含是sound和complete的。我们还建立了表格式计算法的计算复杂性为PSpace。 

---
# Wavelet-Enhanced Neural ODE and Graph Attention for Interpretable Energy Forecasting 

**Title (ZH)**: Wavelet-增强神经ODE和图注意力机制的可解释能源预测 

**Authors**: Usman Gani Joy  

**Link**: [PDF](https://arxiv.org/pdf/2507.10132)  

**Abstract**: Accurate forecasting of energy demand and supply is critical for optimizing sustainable energy systems, yet it is challenged by the variability of renewable sources and dynamic consumption patterns. This paper introduces a neural framework that integrates continuous-time Neural Ordinary Differential Equations (Neural ODEs), graph attention, multi-resolution wavelet transformations, and adaptive learning of frequencies to address the issues of time series prediction. The model employs a robust ODE solver, using the Runge-Kutta method, paired with graph-based attention and residual connections to better understand both structural and temporal patterns. Through wavelet-based feature extraction and adaptive frequency modulation, it adeptly captures and models diverse, multi-scale temporal dynamics. When evaluated across seven diverse datasets: ETTh1, ETTh2, ETTm1, ETTm2 (electricity transformer temperature), and Waste, Solar, and Hydro (renewable energy), this architecture consistently outperforms state-of-the-art baselines in various forecasting metrics, proving its robustness in capturing complex temporal dependencies. Furthermore, the model enhances interpretability through SHAP analysis, making it suitable for sustainable energy applications. 

**Abstract (ZH)**: 准确预测能源需求和供应对于优化可持续能源系统至关重要，但受到可再生能源波动性和动态消费模式的挑战。本文介绍了一种神经框架，该框架集成连续时间神经常微分方程（Neural ODEs）、图注意力、多分辨率小波变换和频率自适应学习，以解决时间序列预测问题。该模型采用鲁棒的ODE求解器，结合图注意力和残差连接，更好地理解结构和时间模式。通过基于小波的特征提取和自适应频率调制，它能够灵活捕捉和建模多尺度的时间动态。该架构在ETTh1、ETTh2、ETTm1、ETTm2（电力变压器温度）以及废物、太阳能和水能（可再生能源）等七个不同数据集上的一系列预测指标中，始终优于最先进的基线模型，证明了其在捕捉复杂时间依赖性方面的稳健性。此外，通过SHAP分析增加模型的可解释性，使其适用于可持续能源应用。 

---
# Taming Modern Point Tracking for Speckle Tracking Echocardiography via Impartial Motion 

**Title (ZH)**: 基于公允运动的现代点跟踪驯化技术在Speckle跟踪心脏超声中的应用 

**Authors**: Md Abulkalam Azad, John Nyberg, Håvard Dalen, Bjørnar Grenne, Lasse Lovstakken, Andreas Østvik  

**Link**: [PDF](https://arxiv.org/pdf/2507.10127)  

**Abstract**: Accurate motion estimation for tracking deformable tissues in echocardiography is essential for precise cardiac function measurements. While traditional methods like block matching or optical flow struggle with intricate cardiac motion, modern point tracking approaches remain largely underexplored in this domain. This work investigates the potential of state-of-the-art (SOTA) point tracking methods for ultrasound, with a focus on echocardiography. Although these novel approaches demonstrate strong performance in general videos, their effectiveness and generalizability in echocardiography remain limited. By analyzing cardiac motion throughout the heart cycle in real B-mode ultrasound videos, we identify that a directional motion bias across different views is affecting the existing training strategies. To mitigate this, we refine the training procedure and incorporate a set of tailored augmentations to reduce the bias and enhance tracking robustness and generalization through impartial cardiac motion. We also propose a lightweight network leveraging multi-scale cost volumes from spatial context alone to challenge the advanced spatiotemporal point tracking models. Experiments demonstrate that fine-tuning with our strategies significantly improves models' performances over their baselines, even for out-of-distribution (OOD) cases. For instance, EchoTracker boosts overall position accuracy by 60.7% and reduces median trajectory error by 61.5% across heart cycle phases. Interestingly, several point tracking models fail to outperform our proposed simple model in terms of tracking accuracy and generalization, reflecting their limitations when applied to echocardiography. Nevertheless, clinical evaluation reveals that these methods improve GLS measurements, aligning more closely with expert-validated, semi-automated tools and thus demonstrating better reproducibility in real-world applications. 

**Abstract (ZH)**: 基于最新点跟踪方法的心脏超声变形组织准确运动估计研究 

---
# A Variance-Reduced Cubic-Regularized Newton for Policy Optimization 

**Title (ZH)**: 带有方差减小的三次正则化牛顿法的策略优化 

**Authors**: Cheng Sun, Zhen Zhang, Shaofu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10120)  

**Abstract**: In this paper, we study a second-order approach to policy optimization in reinforcement learning. Existing second-order methods often suffer from suboptimal sample complexity or rely on unrealistic assumptions about importance sampling. To overcome these limitations, we propose VR-CR-PN, a variance-reduced cubic-regularized policy Newton algorithm. To the best of our knowledge, this is the first algorithm that integrates Hessian-aided variance reduction with second-order policy optimization, effectively addressing the distribution shift problem and achieving best-known sample complexity under general nonconvex conditions but without the need for importance sampling. We theoretically establish that VR-CR-PN achieves a sample complexity of $\tilde{\mathcal{O}}(\epsilon^{-3})$ to reach an $\epsilon$-second-order stationary point, significantly improving upon the previous best result of $\tilde{\mathcal{O}}(\epsilon^{-3.5})$ under comparable assumptions. As an additional contribution, we introduce a novel Hessian estimator for the expected return function, which admits a uniform upper bound independent of the horizon length $H$, allowing the algorithm to achieve horizon-independent sample complexity. 

**Abstract (ZH)**: 在本论文中，我们研究了强化学习中政策优化的二阶方法。现有的二阶方法往往面临次优样本复杂度或依赖于重要性采样的不切实际假设。为克服这些局限，我们提出了一种减少方差的立方正则化政策牛顿算法VR-CR-PN。据我们所知，这是首个将Hessian辅助减少方差与二阶政策优化相结合的算法，有效地解决了分布偏移问题，并在一般非凸条件下达到了最佳的样本复杂度，且无需重要性采样。我们理论分析表明，VR-CR-PN可实现$\tilde{\mathcal{O}}(\epsilon^{-3})$的样本复杂度以达到$\epsilon$-二阶稳定点，这一结果显著优于之前在相似假设下的$\tilde{\mathcal{O}}(\epsilon^{-3.5})$的最佳结果。此外，我们引入了一种新的预期回报函数Hessian估计器，其上界与时间_horizon无关，从而使算法能够获得时间_horizon无关的样本复杂度。 

---
# Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning 

**Title (ZH)**: 增强链式思考推理能力的关键表示微调 

**Authors**: Chenxi Huang, Shaotian Yan, Liang Xie, Binbin Lin, Sinan Fan, Yue Xin, Deng Cai, Chen Shen, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.10085)  

**Abstract**: Representation Fine-tuning (ReFT), a recently proposed Parameter-Efficient Fine-Tuning (PEFT) method, has attracted widespread attention for significantly improving parameter efficiency by editing representation space alone. In this work, we investigate applying ReFT to complex reasoning tasks. However, directly using the native ReFT method, which modifies fixed representations at the beginning and end of each layer, yields suboptimal performance, as these fixed-position representations have uncertain impact on the outputs. We observe that, in complex reasoning tasks, there often exist certain critical representations. These representations either integrate significant information from preceding layers or regulate subsequent layer representations. Through layer-by-layer propagation, they exert a substantial influence on the final output. Naturally, fine-tuning these critical representations has the potential to greatly enhance reasoning performance. Building upon these insights, we propose Critical Representation Fine-Tuning (CRFT), a novel method that identifies and optimizes these critical representations through information flow analysis. CRFT operates within a supervised learning framework, dynamically optimizing critical representations in a low-rank linear subspace while freezing the base model. The effectiveness and efficiency of our method are validated across eight benchmarks for arithmetic and commonsense reasoning, using LLaMA and Mistral model families. Furthermore, our method also adapts effectively to few-shot settings, boosting one-shot accuracy by 16.4%. Our work highlights the untapped potential of representation-level optimization for CoT reasoning, offering a lightweight yet powerful alternative to traditional PEFT methods. 

**Abstract (ZH)**: Representation细调（ReFT）：一种最近提出的参数高效细调（PEFT）方法，通过单独编辑表示空间显著提高了参数效率，引起了广泛关注。在本文中，我们探讨了将ReFT应用于复杂推理任务。然而，直接使用原生的ReFT方法，该方法在每一层的开始和结束处修改固定表示，会导致性能不佳，因为这些固定位置的表示对输出的影响尚不确定。我们观察到，在复杂推理任务中，通常存在一些关键表示，这些表示要么从前一层整合了重要信息，要么调节后续层的表示。通过逐层传播，它们对最终输出产生了重大影响。因此，细调这些关键表示有可能大幅提高推理性能。基于这些见解，我们提出了一种名为关键表示细调（CRFT）的新方法，该方法通过信息流分析来识别和优化这些关键表示。CRFT在监督学习框架下运作，动态优化关键表示在低秩线性子空间中的表现，同时冻结基础模型。我们通过LLaMA和Mistral模型家族在八个算术和常识推理基准上验证了该方法的有效性和效率。此外，我们的方法还能够很好地适应少样本设置，将单样本准确率提升了16.4%。我们的工作突显了表示级优化在CoT推理中的未开发潜力，提供了一种轻量级但强大的传统PEFT方法的替代方案。 

---
# TGLD: A Trust-Aware Game-Theoretic Lane-Changing Decision Framework for Automated Vehicles in Heterogeneous Traffic 

**Title (ZH)**: TGLD：一种考虑信任的游戏理论变道决策框架，应用于异构交通中的自动驾驶车辆 

**Authors**: Jie Pan, Tianyi Wang, Yangyang Wang, Junfeng Jiao, Christian Claudel  

**Link**: [PDF](https://arxiv.org/pdf/2507.10075)  

**Abstract**: Automated vehicles (AVs) face a critical need to adopt socially compatible behaviors and cooperate effectively with human-driven vehicles (HVs) in heterogeneous traffic environment. However, most existing lane-changing frameworks overlook HVs' dynamic trust levels, limiting their ability to accurately predict human driver behaviors. To address this gap, this study proposes a trust-aware game-theoretic lane-changing decision (TGLD) framework. First, we formulate a multi-vehicle coalition game, incorporating fully cooperative interactions among AVs and partially cooperative behaviors from HVs informed by real-time trust evaluations. Second, we develop an online trust evaluation method to dynamically estimate HVs' trust levels during lane-changing interactions, guiding AVs to select context-appropriate cooperative maneuvers. Lastly, social compatibility objectives are considered by minimizing disruption to surrounding vehicles and enhancing the predictability of AV behaviors, thereby ensuring human-friendly and context-adaptive lane-changing strategies. A human-in-the-loop experiment conducted in a highway on-ramp merging scenario validates our TGLD approach. Results show that AVs can effectively adjust strategies according to different HVs' trust levels and driving styles. Moreover, incorporating a trust mechanism significantly improves lane-changing efficiency, maintains safety, and contributes to transparent and adaptive AV-HV interactions. 

**Abstract (ZH)**: 自动车辆（AVs）在异构交通环境中需要采用社会兼容行为，并有效与人类驾驶车辆（HVs）合作。为解决这一问题，本研究提出了一种信任感知博弈论变道决策（TGLD）框架。首先，我们构建了一种多车辆联合游戏，包含AVs之间的完全合作互动和基于实时信任评估的HVs的部分合作行为。其次，我们开发了一种在线信任评价方法，在变道过程中动态估计HVs的信任水平，指导AVs选择合适的合作 maneuvers。最后，通过最小化对周围车辆的干扰和提高AV行为的可预测性，考虑社会兼容性目标，从而确保人类友好和情境适应的变道策略。在高速公路上下道口合并场景中进行的人机环路实验验证了我们的TGLD方法。结果显示，AVs可以根据不同HVs的信任水平和驾驶风格有效调整策略。此外，引入信任机制显著提高了变道效率，维持了安全性，并促进了透明和适应性的AV-HV互动。 

---
# Cultural Bias in Large Language Models: Evaluating AI Agents through Moral Questionnaires 

**Title (ZH)**: 大型语言模型中的文化偏见：通过道德问卷评估AI代理 

**Authors**: Simon Münker  

**Link**: [PDF](https://arxiv.org/pdf/2507.10073)  

**Abstract**: Are AI systems truly representing human values, or merely averaging across them? Our study suggests a concerning reality: Large Language Models (LLMs) fail to represent diverse cultural moral frameworks despite their linguistic capabilities. We expose significant gaps between AI-generated and human moral intuitions by applying the Moral Foundations Questionnaire across 19 cultural contexts. Comparing multiple state-of-the-art LLMs' origins against human baseline data, we find these models systematically homogenize moral diversity. Surprisingly, increased model size doesn't consistently improve cultural representation fidelity. Our findings challenge the growing use of LLMs as synthetic populations in social science research and highlight a fundamental limitation in current AI alignment approaches. Without data-driven alignment beyond prompting, these systems cannot capture the nuanced, culturally-specific moral intuitions. Our results call for more grounded alignment objectives and evaluation metrics to ensure AI systems represent diverse human values rather than flattening the moral landscape. 

**Abstract (ZH)**: AI系统真正代表人类价值观，还是仅仅在它们之间简单平均？我们的研究揭示了一个令人担忧的现实：尽管具有语言能力，大型语言模型（LLMs）未能代表多元文化道德框架。通过应用于19种文化背景的道德基础问卷，我们揭示了AI生成的道德直觉与人类道德直觉之间存在显著差距。我们将多种最先进的LLM的起源与人类基线数据进行比较，发现这些模型系统地同质化了道德多样性。令人惊讶的是，模型规模的增加并不一致地提高文化表现忠实度。我们的发现挑战了在社会科学研究中将LLM作为合成人口的广泛应用，并突显了当前AI对齐方法中的基本局限性。在超出提示的驱动下缺乏数据驱动的对齐，这些系统无法捕捉到复杂的、文化特定的道德直觉。我们的研究结果呼吁更多基于实际的目标和评估指标，以确保AI系统能够代表多元的人类价值观，而不是简单化道德景观。 

---
# PRISM: Fine-Grained Paper-to-Paper Retrieval with Multi-Aspect-Aware Query Optimization 

**Title (ZH)**: PRISM: 多方面感知查询优化的细粒度文献检索 

**Authors**: Sangwoo Park, Jinheon Baek, Soyeong Jeong, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2507.10057)  

**Abstract**: Scientific paper retrieval, particularly framed as document-to-document retrieval, aims to identify relevant papers in response to a long-form query paper, rather than a short query string. Previous approaches to this task have focused on abstracts, embedding them into dense vectors as surrogates for full documents and calculating similarity across them, although abstracts provide only sparse and high-level summaries. To address this, we propose PRISM, a novel document-to-document retrieval method that introduces multiple, fine-grained representations for both the query and candidate papers. In particular, each query paper is decomposed into multiple aspect-specific views and individually embedded, which are then matched against candidate papers similarity segmented to consider their multifaceted dimensions. Moreover, we present SciFullBench, a novel benchmark in which the complete and segmented context of full papers for both queries and candidates is available. Then, experimental results show that PRISM improves performance by an average of 4.3% over existing retrieval baselines. 

**Abstract (ZH)**: 科学论文检索，特别是框架下的文档到文档检索，旨在针对长篇查询论文识别相关论文，而非短查询字符串。针对这一任务，先前的方法集中在摘要上，将摘要嵌入为密集向量以代替全文，并计算它们之间的相似性，尽管摘要仅提供了稀疏且高层次的总结。为解决这一问题，我们提出PRISM，一种新颖的文档到文档检索方法，引入了查询和候选论文的多种细粒度表示。特别地，每篇查询论文被分解为多个方面特定的视图并单独嵌入，然后与候选论文相似地分割匹配，以考虑其多维度特征。此外，我们提出SciFullBench，一种新颖的基准，在其中查询和候选论文的完整和分割上下文是可用的。实验结果表明，PRISM在现有检索基线上的性能平均提高4.3%。 

---
# Lightweight Model for Poultry Disease Detection from Fecal Images Using Multi-Color Space Feature Optimization and Machine Learning 

**Title (ZH)**: 基于多色彩空间特征优化与机器学习的轻量级家禽疾病检测模型研宄 

**Authors**: A. K. M. Shoriful Islam, Md. Rakib Hassan, Macbah Uddin, Md. Shahidur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2507.10056)  

**Abstract**: Poultry farming is a vital component of the global food supply chain, yet it remains highly vulnerable to infectious diseases such as coccidiosis, salmonellosis, and Newcastle disease. This study proposes a lightweight machine learning-based approach to detect these diseases by analyzing poultry fecal images. We utilize multi-color space feature extraction (RGB, HSV, LAB) and explore a wide range of color, texture, and shape-based descriptors, including color histograms, local binary patterns (LBP), wavelet transforms, and edge detectors. Through a systematic ablation study and dimensionality reduction using PCA and XGBoost feature selection, we identify a compact global feature set that balances accuracy and computational efficiency. An artificial neural network (ANN) classifier trained on these features achieved 95.85% accuracy while requiring no GPU and only 638 seconds of execution time in Google Colab. Compared to deep learning models such as Xception and MobileNetV3, our proposed model offers comparable accuracy with drastically lower resource usage. This work demonstrates a cost-effective, interpretable, and scalable alternative to deep learning for real-time poultry disease detection in low-resource agricultural settings. 

**Abstract (ZH)**: 基于轻量级机器学习的 poultry 粪便图像分析方法在检测传染性疾病中的应用：一种在低资源农业环境中实现实时家禽疾病检测的经济、可解释且可扩展的替代方案 

---
# (Almost) Free Modality Stitching of Foundation Models 

**Title (ZH)**: 近乎免费的基础模型模态拼接 

**Authors**: Jaisidh Singh, Diganta Misra, Boris Knyazev, Antonio Orvieto  

**Link**: [PDF](https://arxiv.org/pdf/2507.10015)  

**Abstract**: Foundation multi-modal models are often designed by stitching of multiple existing pretrained uni-modal models: for example, an image classifier with an autoregressive text model. This stitching process is performed by training a connector module that aims to align the representation-representation or representation-input spaces of these uni-modal models. However, given the complexity of training such connectors on large scale web-based datasets coupled with the ever-increasing number of available pretrained uni-modal models, the task of uni-modal models selection and subsequent connector module training becomes computationally demanding. To address this under-studied critical problem, we propose Hypernetwork Model Alignment (Hyma), a novel all-in-one solution for optimal uni-modal model selection and connector training by leveraging hypernetworks. Specifically, our framework utilizes the parameter prediction capability of a hypernetwork to obtain jointly trained connector modules for $N \times M$ combinations of uni-modal models. In our experiments, Hyma reduces the optimal uni-modal model pair search cost by $10\times$ (averaged across all experiments), while matching the ranking and trained connector performance obtained via grid search across a suite of diverse multi-modal benchmarks. 

**Abstract (ZH)**: 基于超网络的模态模型联盟（Hyma）：一种新的统一解决方案，用于高效选择最优单模模型并训练连接器模块 

---
# Evolution of Fear and Social Rewards in Prey-Predator Relationship 

**Title (ZH)**: 猎食者-被捕食者关系中恐惧与社会奖励的进化 

**Authors**: Yuji Kanagawa, Kenji Doya  

**Link**: [PDF](https://arxiv.org/pdf/2507.09992)  

**Abstract**: Fear is a critical brain function for detecting danger and learning to avoid specific stimuli that can lead to danger. While fear is believed to have evolved under pressure from predators, experimentally reproducing the evolution is challenging. To investigate the relationship between environmental conditions, the evolution of fear, and the evolution of other rewards, such as food reward and social reward, we developed a distributed evolutionary simulation. In our simulation, prey and predator agents co-evolve their innate reward functions, including a possibly fear-like term for observing predators, and learn behaviors via reinforcement learning. Surprisingly, our simulation revealed that social reward for observing the same species is more important for prey to survive, and fear-like negative reward for observing predators evolves only after acquiring social reward. We also found that the predator with increased hunting ability (larger mouth) amplified fear emergence, but also that fear evolution is more stable with non-evolving predators that are bad at chasing prey. Additionally, unlike for predators, we found that positive rewards evolve in opposition to fear for stationary threats, as areas with abundant leftover food develop around them. These findings suggest that fear and social reward have had a complex interplay with each other through evolution, along with the nature of predators and threats. 

**Abstract (ZH)**: 恐惧是检测危险和避免特定危险刺激的关键大脑功能。虽然恐惧被认为是在捕食者压力下进化的，但实验性地重现这一进化过程具有挑战性。为研究环境条件、恐惧的进化与其他奖励如食物奖励和社会奖励的进化之间的关系，我们开发了一种分布式进化仿真。在我们的仿真中，被捕食者和捕食者代理共同进化其固有的奖励功能，包括可能类似于恐惧的项以观察捕食者，并通过强化学习学习行为。令人惊讶的是，我们的仿真揭示出，观察同物种的社会奖励对于被捕食者生存更为重要，而观察捕食者的类似恐惧的负向奖励仅在获得社会奖励后才会进化。我们还发现，具有增强狩猎能力（更大嘴巴）的捕食者加剧了恐惧的出现，但具有较差追赶能力的非进化捕食者会使恐惧的进化更加稳定。此外，与捕食者不同，我们发现，对于静止的威胁，正面奖励会在它们周围大量食物残余的区域与恐惧进化相反而出现。这些发现表明，恐惧和社交奖励在捕食者和威胁的本性影响下，通过进化过程与彼此产生了复杂相互作用。 

---
# Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix 

**Title (ZH)**: 差分隐私联邦低秩适应超越固定矩阵 

**Authors**: Ming Wen, Jiaqi Zhu, Yuedong Xu, Yipeng Zhou, Dingding Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.09990)  

**Abstract**: Large language models (LLMs) typically require fine-tuning for domain-specific tasks, and LoRA offers a computationally efficient approach by training low-rank adapters. LoRA is also communication-efficient for federated LLMs when multiple users collaboratively fine-tune a global LLM model without sharing their proprietary raw data. However, even the transmission of local adapters between a server and clients risks serious privacy leakage. Applying differential privacy (DP) to federated LoRA encounters a dilemma: adding noise to both adapters amplifies synthetic noise on the model, while fixing one adapter impairs the learnability of fine-tuning. In this paper, we propose FedASK (Differentially Private Federated Low Rank Adaptation with Double Sketching) , a novel federated LoRA framework to enable effective updating of both low-rank adapters with robust differential privacy. Inspired by randomized SVD, our key idea is a two-stage sketching pipeline. This pipeline first aggregates carefully sketched, privacy-preserving local updates, and then reconstructs the global matrices on the server to facilitate effective updating of both adapters. We theoretically prove FedASK's differential privacy guarantee and its exact aggregation property. Comprehensive experiments demonstrate that FedASK consistently outperforms baseline methods across a variety of privacy settings and data distributions. 

**Abstract (ZH)**: Large语言模型（LLMs）通常需要针对特定领域进行微调，LoRA通过训练低秩适配器提供了一种计算高效的方法。LoRA在多个用户协作微调全局LLM模型而不共享其专有原始数据时，也是一种通信高效的方案。然而，即使在服务器和客户端之间传输本地适配器也存在严重的隐私泄露风险。将差分隐私（DP）应用于联邦LoRA会遇到一个困境：在适配器中添加噪声会放大模型上的合成噪声，而固定一个适配器又会影响微调的学习能力。本文提出了一种名为FedASK（差分隐私联邦低秩适应性增强双素描）的新型联邦LoRA框架，以实现对两个低秩适配器的有效更新并具备 robust 差分隐私保证。我们的核心思想是两阶段素描流水线，该流水线首先聚合精心素描的、隐私保护的本地更新，然后在服务器上重建全局矩阵，以促进适配器的有效更新。我们理论上证明了FedASK的差分隐私保证及其精确聚合特性。全面的实验表明，无论在何种隐私设置和数据分布下，FedASK一致地优于基线方法。 

---
# Demonstrating the Octopi-1.5 Visual-Tactile-Language Model 

**Title (ZH)**: 展示Octopi-1.5视觉-触觉-语言模型 

**Authors**: Samson Yu, Kelvin Lin, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2507.09985)  

**Abstract**: Touch is recognized as a vital sense for humans and an equally important modality for robots, especially for dexterous manipulation, material identification, and scenarios involving visual occlusion. Building upon very recent work in touch foundation models, this demonstration will feature Octopi-1.5, our latest visual-tactile-language model. Compared to its predecessor, Octopi-1.5 introduces the ability to process tactile signals from multiple object parts and employs a simple retrieval-augmented generation (RAG) module to improve performance on tasks and potentially learn new objects on-the-fly. The system can be experienced live through a new handheld tactile-enabled interface, the TMI, equipped with GelSight and TAC-02 tactile sensors. This convenient and accessible setup allows users to interact with Octopi-1.5 without requiring a robot. During the demonstration, we will showcase Octopi-1.5 solving tactile inference tasks by leveraging tactile inputs and commonsense knowledge. For example, in a Guessing Game, Octopi-1.5 will identify objects being grasped and respond to follow-up queries about how to handle it (e.g., recommending careful handling for soft fruits). We also plan to demonstrate Octopi-1.5's RAG capabilities by teaching it new items. With live interactions, this demonstration aims to highlight both the progress and limitations of VTLMs such as Octopi-1.5 and to foster further interest in this exciting field. Code for Octopi-1.5 and design files for the TMI gripper are available at this https URL. 

**Abstract (ZH)**: 触觉被认定为人类的一个重要感官，对于机器人来说，特别是在灵巧操作、材料识别以及涉及视觉遮挡的场景中，触觉也是一个同等重要的模态。基于近期的触觉基础模型研究，我们将展示Octopi-1.5，这是我们的最新视觉-触觉-语言模型。相比其前代产品，Octipi-1.5 增强了处理多个物体部位触觉信号的能力，并采用简单的检索增强生成（RAG）模块以提高任务性能，并且有可能在不预先训练的情况下学习新物体。用户可以通过新的便携式触觉接口TMI直接与系统互动，该接口配备了GelSight和TAC-02触觉传感器，无需机器人即可操作。在展示中，我们将通过利用触觉输入和常识知识，展示Octopi-1.5解决触觉推理任务的能力。例如，在一个猜物游戏中，Octopi-1.5会识别被握住的物体，并回应关于如何处理该物体的后续查询（如推荐小心处理软水果）。我们还将演示Octopi-1.5的RAG能力，通过教导它新物品。通过现场互动，本次展示旨在突出如Octopi-1.5这样的VTLM的进展和局限性，并进一步激发对该领域兴趣。Octopi-1.5的代码和TMI夹爪的设计文件可在<a href="这个链接">此处</a>获取。 

---
# Tiny Reward Models 

**Title (ZH)**: Tiny 奖励模型 

**Authors**: Sarah Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.09973)  

**Abstract**: Large decoder-based language models have become the dominant architecture for reward modeling in reinforcement learning from human feedback (RLHF). However, as reward models are increasingly deployed in test-time strategies, their inference costs become a growing concern. We present TinyRM, a family of small, bidirectional masked language models (MLMs) with as few as 400 million parameters, that rival the capabilities of models over 175 times larger on reasoning and safety preference modeling tasks. TinyRM combines FLAN-style prompting, Directional Low-Rank Adaptation (DoRA), and layer freezing to achieve strong performance on RewardBench, despite using significantly fewer resources. Our experiments suggest that small models benefit from domain-specific tuning strategies, particularly in reasoning, where lightweight finetuning methods are especially effective. While challenges remain in building generalist models and conversational preference modeling, our preliminary results highlight the promise of lightweight bidirectional architectures as efficient, scalable alternatives for preference modeling. 

**Abstract (ZH)**: 基于大型解码器的语言模型已成为从人类反馈强化学习（RLHF）中奖励建模的主要架构。然而，随着奖励模型在测试时策略中的部署增多，其推理成本成为一个日益严重的关注点。我们提出了TinyRM，这是一种小型、双向遮蔽语言模型（MLM），参数量少至400 million，但在推理和安全偏好建模任务上与超过其175倍大的模型相媲美。TinyRM 结合了 FLAN 风格的提示、定向低秩适应（DoRA）以及层冻结，即使使用了显著较少的资源，也能在 RewardBench 中获得出色表现。我们的实验表明，小型模型可以从领域特定的调优策略中受益，尤其是在推理任务中，轻量级微调方法尤其有效。尽管在构建通用模型和对话偏好建模中仍存在挑战，但我们的初步结果突显了轻量级双向架构作为偏好建模高效且可扩展替代方案的潜力。 

---
# A Brain Tumor Segmentation Method Based on CLIP and 3D U-Net with Cross-Modal Semantic Guidance and Multi-Level Feature Fusion 

**Title (ZH)**: 基于CLIP和3D U-Net的跨模态语义指导及多级特征融合的脑肿瘤分割方法 

**Authors**: Mingda Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09966)  

**Abstract**: Precise segmentation of brain tumors from magnetic resonance imaging (MRI) is essential for neuro-oncology diagnosis and treatment planning. Despite advances in deep learning methods, automatic segmentation remains challenging due to tumor morphological heterogeneity and complex three-dimensional spatial relationships. Current techniques primarily rely on visual features extracted from MRI sequences while underutilizing semantic knowledge embedded in medical reports. This research presents a multi-level fusion architecture that integrates pixel-level, feature-level, and semantic-level information, facilitating comprehensive processing from low-level data to high-level concepts. The semantic-level fusion pathway combines the semantic understanding capabilities of Contrastive Language-Image Pre-training (CLIP) models with the spatial feature extraction advantages of 3D U-Net through three mechanisms: 3D-2D semantic bridging, cross-modal semantic guidance, and semantic-based attention mechanisms. Experimental validation on the BraTS 2020 dataset demonstrates that the proposed model achieves an overall Dice coefficient of 0.8567, representing a 4.8% improvement compared to traditional 3D U-Net, with a 7.3% Dice coefficient increase in the clinically important enhancing tumor (ET) region. 

**Abstract (ZH)**: 从磁共振成像中精确分割脑肿瘤是神经肿瘤诊断和治疗规划不可或缺的部分。尽管深度学习方法取得了进展，但由于肿瘤形态异质性和复杂的三维空间关系，自动分割仍然具有挑战性。当前技术主要依赖于从MRI序列中提取的视觉特征，而未能充分利用医学报告中蕴含的语义知识。本研究提出了一种多级融合架构，该架构整合了像素级、特征级和语义级信息，促进了从低级数据到高级概念的全面处理。语义级融合路径将对比语言-图像预训练（CLIP）模型的语义理解能力和3D U-Net的空间特征提取优势结合起来，通过三种机制实现了这一融合：三维-二维语义桥梁、跨模态语义指导和基于语义的注意力机制。在BraTS 2020数据集上的实验验证表明，所提出模型的整体Dice系数为0.8567，相较于传统的3D U-Net提高了4.8%，在临床上重要的增强肿瘤（ET）区域的Dice系数提高了7.3%。 

---
# Can GPT-4o mini and Gemini 2.0 Flash Predict Fine-Grained Fashion Product Attributes? A Zero-Shot Analysis 

**Title (ZH)**: GPT-4o mini和Gemini 2.0能否快速预测细粒度时尚产品属性？一种零样本分析 

**Authors**: Shubham Shukla, Kunal Sonalkar  

**Link**: [PDF](https://arxiv.org/pdf/2507.09950)  

**Abstract**: The fashion retail business is centered around the capacity to comprehend products. Product attribution helps in comprehending products depending on the business process. Quality attribution improves the customer experience as they navigate through millions of products offered by a retail website. It leads to well-organized product catalogs. In the end, product attribution directly impacts the 'discovery experience' of the customer. Although large language models (LLMs) have shown remarkable capabilities in understanding multimodal data, their performance on fine-grained fashion attribute recognition remains under-explored. This paper presents a zero-shot evaluation of state-of-the-art LLMs that balance performance with speed and cost efficiency, mainly GPT-4o-mini and Gemini 2.0 Flash. We have used the dataset DeepFashion-MultiModal (this https URL) to evaluate these models in the attribution tasks of fashion products. Our study evaluates these models across 18 categories of fashion attributes, offering insight into where these models excel. We only use images as the sole input for product information to create a constrained environment. Our analysis shows that Gemini 2.0 Flash demonstrates the strongest overall performance with a macro F1 score of 56.79% across all attributes, while GPT-4o-mini scored a macro F1 score of 43.28%. Through detailed error analysis, our findings provide practical insights for deploying these LLMs in production e-commerce product attribution-related tasks and highlight the need for domain-specific fine-tuning approaches. This work also lays the groundwork for future research in fashion AI and multimodal attribute extraction. 

**Abstract (ZH)**: 大型语言模型在细粒度时尚属性识别中的零样本评估 

---
# Memorization Sinks: Isolating Memorization during LLM Training 

**Title (ZH)**: 记忆吸收器：隔离大规模语言模型训练中的记忆现象 

**Authors**: Gaurav R. Ghosal, Pratyush Maini, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2507.09937)  

**Abstract**: Large language models are susceptible to memorizing repeated sequences, posing privacy and copyright concerns. A popular mitigation strategy is to remove memorized information from specific neurons post-hoc. However, such approaches have shown limited success so far. In a controlled setting, we show that the memorization of natural sequences (those that resemble linguistically plausible text) become mechanistically entangled with general language abilities, thereby becoming challenging to remove post-hoc. In this work, we put forward a new paradigm of MemSinks that promotes isolation of memorization by design. We leverage a sequence identifier that activates a unique set of memorization neurons for each sequence across repetitions. By analyzing the dynamics of learning and forgetting, we argue that MemSinks facilitates isolation of memorized content, making it easier to remove without compromising general language capabilities. We implement MemSinks at the billion-parameter and billion-token scale, and observe both effective isolation and strong generalization. To our knowledge, this is the first proof-of-concept on real data demonstrating that simultaneous generalization and isolation is achievable. We open-source our code at this http URL. 

**Abstract (ZH)**: 大规模语言模型容易记住重复序列，这带来了隐私和版权方面的担忧。一种流行的缓解策略是事后从特定神经元中移除记住的信息。然而，这样的方法迄今显示出有限的效果。在一个受控设置中，我们表明，自然序列（那些在语言上可能是合理的文本）的记忆与一般语言能力机械地交织在一起，从而使其事后去除变得具有挑战性。在本文中，我们提出了MemSinks的新范式，通过设计促进记忆的隔离。我们利用一个序列标识器，该标识器在每次重复中为每个序列激活一组独特的记忆神经元。通过分析学习和遗忘的动力学，我们argue认为MemSinks促进了记忆内容的隔离，使其在不破坏一般语言能力的情况下更容易去除。我们以十亿参数和十亿令牌的规模实现了MemSinks，并观察到有效的隔离和强大的泛化。据我们所知，这是首个在实际数据上证明同时实现泛化和隔离的方法。我们的代码已开源。 

---
# Enhancing Retrieval Augmented Generation with Hierarchical Text Segmentation Chunking 

**Title (ZH)**: 基于层次文本分段切分的检索增强生成增强 

**Authors**: Hai Toan Nguyen, Tien Dat Nguyen, Viet Ha Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09935)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems commonly use chunking strategies for retrieval, which enhance large language models (LLMs) by enabling them to access external knowledge, ensuring that the retrieved information is up-to-date and domain-specific. However, traditional methods often fail to create chunks that capture sufficient semantic meaning, as they do not account for the underlying textual structure. This paper proposes a novel framework that enhances RAG by integrating hierarchical text segmentation and clustering to generate more meaningful and semantically coherent chunks. During inference, the framework retrieves information by leveraging both segment-level and cluster-level vector representations, thereby increasing the likelihood of retrieving more precise and contextually relevant information. Evaluations on the NarrativeQA, QuALITY, and QASPER datasets indicate that the proposed method achieved improved results compared to traditional chunking techniques. 

**Abstract (ZH)**: 基于检索增强生成的系统通常采用分块策略，通过引入层次化的文本分割和聚类，增强大型语言模型的外部知识访问能力，确保检索信息的时效性和领域特定性。然而，传统方法往往难以生成包含足够语义意义的分块，因为它们没有考虑到文本的内在结构。本文提出了一种新型框架，通过集成层次化的文本分割和聚类来增强基于检索增强生成的系统，生成更具意义且语义连贯的分块。在推理过程中，该框架利用段落级和集群级向量表示来检索信息，从而增加检索到更精确和上下文相关的信息的可能性。在NarrativeQA、QuALITY和QASPER数据集上的评估表明，所提出的方法在结果上优于传统的分块技术。 

---
# Mechanistic Interpretability of LoRA-Adapted Language Models for Nuclear Reactor Safety Applications 

**Title (ZH)**: LoRA-适配语言模型在核反应堆安全应用中的机制可解释性 

**Authors**: Yoon Pyo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.09931)  

**Abstract**: The integration of Large Language Models (LLMs) into safety-critical domains, such as nuclear engineering, necessitates a deep understanding of their internal reasoning processes. This paper presents a novel methodology for interpreting how an LLM encodes and utilizes domain-specific knowledge, using a Boiling Water Reactor system as a case study. We adapted a general-purpose LLM (Gemma-3-1b-it) to the nuclear domain using a parameter-efficient fine-tuning technique known as Low-Rank Adaptation. By comparing the neuron activation patterns of the base model to those of the fine-tuned model, we identified a sparse set of neurons whose behavior was significantly altered during the adaptation process. To probe the causal role of these specialized neurons, we employed a neuron silencing technique. Our results demonstrate that while silencing most of these specialized neurons individually did not produce a statistically significant effect, deactivating the entire group collectively led to a statistically significant degradation in task performance. Qualitative analysis further revealed that silencing these neurons impaired the model's ability to generate detailed, contextually accurate technical information. This paper provides a concrete methodology for enhancing the transparency of an opaque black-box model, allowing domain expertise to be traced to verifiable neural circuits. This offers a pathway towards achieving nuclear-grade artificial intelligence (AI) assurance, addressing the verification and validation challenges mandated by nuclear regulatory frameworks (e.g., 10 CFR 50 Appendix B), which have limited AI deployment in safety-critical nuclear operations. 

**Abstract (ZH)**: 将大型语言模型（LLMs）整合到核工程等安全关键领域需要深入理解其内部推理过程。本文提出了一种新的方法，通过沸水堆系统作为案例研究，解释LLM如何编码和利用领域特定知识。我们使用一种参数高效微调技术——低秩适应，将一种通用语言模型（Gemma-3-1b-it）适应到核领域。通过比较基础模型和微调模型的神经激活模式，我们发现一组在适应过程中行为显著改变的稀疏神经元。为了探究这些专门神经元的因果作用，我们采用了神经缄默技术。结果显示，单独缄默这些专门神经元中的大多数并没有产生统计显著的效果，但集体禁用整个小组则显著恶化了任务性能。定性分析进一步表明，缄默这些神经元削弱了模型生成详细、上下文相关技术信息的能力。本文提供了一种具体的方法，以提高不透明黑箱模型的透明度，使领域专业知识可以追溯到可验证的神经电路。这为实现核级人工智能（AI）保证提供了途径，解决了核监管框架（如10 CFR 50附录B）要求的验证和验证挑战，这些挑战限制了AI在安全关键核操作中的部署。 

---
# Aligning Generative Speech Enhancement with Human Preferences via Direct Preference Optimization 

**Title (ZH)**: 通过直接偏好优化实现生成性语音增强与人类偏好的对齐 

**Authors**: Haoyang Li, Nana Hou, Yuchen Hu, Jixun Yao, Sabato Marco Siniscalchi, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2507.09929)  

**Abstract**: This work investigates speech enhancement (SE) from the perspective of language models (LMs). We propose a novel method that leverages Direct Preference Optimization (DPO) to improve the perceptual quality of enhanced speech. Using UTMOS, a neural MOS prediction model, as a proxy for human ratings, our approach guides optimization toward perceptually preferred outputs. This differs from existing LM-based SE methods that focus on maximizing the likelihood of clean speech tokens, which may misalign with human perception and degrade quality despite low prediction error. Experiments on the 2020 Deep Noise Suppression Challenge test sets demonstrate that applying DPO to a pretrained LM-based SE model yields consistent improvements across various speech quality metrics, with relative gains of up to 56%. To our knowledge, this is the first application of DPO to SE and the first to incorporate proxy perceptual feedback into LM-based SE training, pointing to a promising direction for perceptually aligned SE. 

**Abstract (ZH)**: 本研究从语言模型的角度探讨了语音增强（SE）。我们提出了一种新颖的方法，利用直接偏好优化（DPO）来提高增强语音的感知质量。使用UTMOS（一种神经MOS预测模型）作为人类评分的代理，我们的方法指导优化以产生感知上更优越的输出。这与现有基于语言模型的SE方法不同，后者侧重于最大化清洁语音标记的概率，可能导致与人类感知不符并降低质量的现象，尽管预测误差较低。实验表明，将DPO应用于预训练的基于语言模型的SE模型可以一致地在多种语音质量指标上提高性能，相对增益高达56%。据我们所知，这是首次将DPO应用于SE，并首次将代理感知反馈纳入基于语言模型的SE训练中，为感知对齐的SE指明了一个有希望的发展方向。 

---
# MixLoRA-DSI: Dynamically Expandable Mixture-of-LoRA Experts for Rehearsal-Free Generative Retrieval over Dynamic Corpora 

**Title (ZH)**: MixLoRA-DSI: 动态可扩展的LoRA混合专家用于动态 CORPORA 的生成性检索 

**Authors**: Tuan-Luc Huynh, Thuy-Trang Vu, Weiqing Wang, Trung Le, Dragan Gašević, Yuan-Fang Li, Thanh-Toan Do  

**Link**: [PDF](https://arxiv.org/pdf/2507.09924)  

**Abstract**: Continually updating model-based indexes in generative retrieval with new documents remains challenging, as full retraining is computationally expensive and impractical under resource constraints. We propose MixLoRA-DSI, a novel framework that combines an expandable mixture of Low-Rank Adaptation experts with a layer-wise out-of-distribution (OOD)-driven expansion strategy. Instead of allocating new experts for each new corpus, our proposed expansion strategy enables sublinear parameter growth by selectively introducing new experts only when significant number of OOD documents are detected. Experiments on NQ320k and MS MARCO Passage demonstrate that MixLoRA-DSI outperforms full-model update baselines, with minimal parameter overhead and substantially lower training costs. 

**Abstract (ZH)**: 基于生成检索的模型本征索引在加入新文档时持续更新仍具有挑战性，因为全面重训在资源受限条件下既耗时又不切实际。我们提出了一种新型框架MixLoRA-DSI，该框架结合了可扩展的低秩适应专家混合体和逐层分布外推策略。我们的扩展策略不仅避免为每个新语料库分配新的专家，还通过仅在检测到大量分布外文档时才引入新专家，实现了亚线性参数增长。实验结果表明，在NQ320k和MS MARCO Passage数据集上，MixLoRA-DSI在最少参数开销的情况下，训练成本显著降低，并优于全面模型更新基准。 

---
# Large Population Models 

**Title (ZH)**: 大型人口模型 

**Authors**: Ayush Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2507.09901)  

**Abstract**: Many of society's most pressing challenges, from pandemic response to supply chain disruptions to climate adaptation, emerge from the collective behavior of millions of autonomous agents making decisions over time. Large Population Models (LPMs) offer an approach to understand these complex systems by simulating entire populations with realistic behaviors and interactions at unprecedented scale. LPMs extend traditional modeling approaches through three key innovations: computational methods that efficiently simulate millions of agents simultaneously, mathematical frameworks that learn from diverse real-world data streams, and privacy-preserving communication protocols that bridge virtual and physical environments. This allows researchers to observe how agent behavior aggregates into system-level outcomes and test interventions before real-world implementation. While current AI advances primarily focus on creating "digital humans" with sophisticated individual capabilities, LPMs develop "digital societies" where the richness of interactions reveals emergent phenomena. By bridging individual agent behavior and population-scale dynamics, LPMs offer a complementary path in AI research illuminating collective intelligence and providing testing grounds for policies and social innovations before real-world deployment. We discuss the technical foundations and some open problems here. LPMs are implemented by the AgentTorch framework (this http URL) 

**Abstract (ZH)**: 社会面临的许多紧迫挑战，从大流行应对到供应链中断再到气候适应，都源于数百万个自主代理随时间做出决策所产生的集体行为。大规模人群模型（LPMs）通过以前所未有的规模模拟整个具有现实行为和交互模式的人群，提供了一种理解这些复杂系统的办法。LPMs通过三种关键创新扩展了传统的建模方法：高效的计算方法可以同时模拟数百万个代理，数学框架可以从多种多样的现实世界数据流中学习，并且保护隐私的通信协议可以连接虚拟和物理环境。这使研究人员能够观察代理行为如何汇总为系统级别的结果，并在实际实施之前测试干预措施。虽然当前的人工智能进步主要侧重于创建具有复杂个体能力的“数字人类”，LPMs则致力于构建“数字社会”，其中丰富的人际互动揭示了 emergent 现象。通过连接个体代理行为和群体规模的动力学，LPMs提供了一条补充的人工智能研究路径，揭示集体智能，并为政策和社会创新提供测试平台，然后再进行实际部署。我们在这里讨论其技术基础和一些开放问题。LPMs由AgentTorch框架实现（ this http URL）。 

---
# Advanced U-Net Architectures with CNN Backbones for Automated Lung Cancer Detection and Segmentation in Chest CT Images 

**Title (ZH)**: 基于CNN骨干网络的高级U-Net架构在胸部CT图像中自动肺癌检测与分割 

**Authors**: Alireza Golkarieha, Kiana Kiashemshakib, Sajjad Rezvani Boroujenic, Nasibeh Asadi Isakand  

**Link**: [PDF](https://arxiv.org/pdf/2507.09898)  

**Abstract**: This study investigates the effectiveness of U-Net architectures integrated with various convolutional neural network (CNN) backbones for automated lung cancer detection and segmentation in chest CT images, addressing the critical need for accurate diagnostic tools in clinical settings. A balanced dataset of 832 chest CT images (416 cancerous and 416 non-cancerous) was preprocessed using Contrast Limited Adaptive Histogram Equalization (CLAHE) and resized to 128x128 pixels. U-Net models were developed with three CNN backbones: ResNet50, VGG16, and Xception, to segment lung regions. After segmentation, CNN-based classifiers and hybrid models combining CNN feature extraction with traditional machine learning classifiers (Support Vector Machine, Random Forest, and Gradient Boosting) were evaluated using 5-fold cross-validation. Metrics included accuracy, precision, recall, F1-score, Dice coefficient, and ROC-AUC. U-Net with ResNet50 achieved the best performance for cancerous lungs (Dice: 0.9495, Accuracy: 0.9735), while U-Net with VGG16 performed best for non-cancerous segmentation (Dice: 0.9532, Accuracy: 0.9513). For classification, the CNN model using U-Net with Xception achieved 99.1 percent accuracy, 99.74 percent recall, and 99.42 percent F1-score. The hybrid CNN-SVM-Xception model achieved 96.7 percent accuracy and 97.88 percent F1-score. Compared to prior methods, our framework consistently outperformed existing models. In conclusion, combining U-Net with advanced CNN backbones provides a powerful method for both segmentation and classification of lung cancer in CT scans, supporting early diagnosis and clinical decision-making. 

**Abstract (ZH)**: 本研究探讨了结合各种卷积神经网络（CNN）骨干网络的U-Net架构在胸部CT图像中自动化肺癌检测和分割的有效性，满足了临床环境中对准确诊断工具的迫切需求。使用对比受限自适应直方图均衡化（CLAHE）对一个平衡的数据集（832张胸部CT图像，包括416张癌性图像和416张非癌性图像）进行预处理，并将其调整为128x128像素。开发了三种CNN骨干网络（ResNet50、VGG16和Xception）的U-Net模型，用于分割肺部区域。分割后，基于CNN的分类器以及结合CNN特征提取与传统机器学习分类器（支持向量机、随机森林和梯度提升）的混合模型，采用5折交叉验证进行评估。评估指标包括准确率、精确率、召回率、F1分数、Dice系数和ROC-AUC。使用ResNet50的U-Net在癌性肺部分割中表现最佳（Dice：0.9495，准确率：0.9735），而使用VGG16的U-Net在非癌性分割中表现最佳（Dice：0.9532，准确率：0.9513）。在分类方面，使用Xception的U-Net CNN模型实现了99.1%的准确率、99.74%的召回率和99.42%的F1分数。混合CNN-SVM-Xception模型的准确率为96.7%，F1分数为97.88%。与先前的方法相比，我们的框架在各指标上持续优于现有模型。综上所述，结合U-Net与高级CNN骨干网络为CT扫描中的肺癌分割和分类提供了一种强大的方法，支持早期诊断和临床决策。 

---
# Sequence-Model-Guided Measurement Selection for Quantum State Learning 

**Title (ZH)**: 基于序列模型的量子态学习测量选择 

**Authors**: Jiaxin Huang, Yan Zhu, Giulio Chiribella, Ya-Dong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09891)  

**Abstract**: Characterization of quantum systems from experimental data is a central problem in quantum science and technology. But which measurements should be used to gather data in the first place? While optimal measurement choices can be worked out for small quantum systems, the optimization becomes intractable as the system size grows large. To address this problem, we introduce a deep neural network with a sequence model architecture that searches for efficient measurement choices in a data-driven, adaptive manner. The model can be applied to a variety of tasks, including the prediction of linear and nonlinear properties of quantum states, as well as state clustering and state tomography tasks. In all these tasks, we find that the measurement choices identified by our neural network consistently outperform the uniformly random choice. Intriguingly, for topological quantum systems, our model tends to recommend measurements at the system's boundaries, even when the task is to predict bulk properties. This behavior suggests that the neural network may have independently discovered a connection between boundaries and bulk, without having been provided any built-in knowledge of quantum physics. 

**Abstract (ZH)**: 基于实验数据表征量子系统是量子科学与技术中的一个核心问题。但在最初应使用哪些测量来收集数据？虽然小型量子系统中最佳测量选择可以计算得出，但随着系统规模增大，优化变得不可行。为解决这一问题，我们引入了一种具有序列模型架构的深度神经网络，在数据驱动和自适应方式下搜索高效的测量选择。该模型可以应用于预测量子态的线性和非线性性质、状态聚类以及状态Tomography等多种任务。在所有这些任务中，我们发现由我们的神经网络识别出的测量选择始终优于均匀随机选择。有趣的是，对于拓扑量子系统，即使任务是预测体相性质，我们的模型也倾向于建议在系统的边界上进行测量。这一行为表明，神经网络可能独立地发现边界与体相之间的联系，而无需任何内置的量子物理知识。 

---
# Soft Graph Clustering for single-cell RNA Sequencing Data 

**Title (ZH)**: 软图聚类方法在单细胞RNA测序数据中的应用 

**Authors**: Ping Xu, Pengfei Wang, Zhiyuan Ning, Meng Xiao, Min Wu, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.09890)  

**Abstract**: Clustering analysis is fundamental in single-cell RNA sequencing (scRNA-seq) data analysis for elucidating cellular heterogeneity and diversity. Recent graph-based scRNA-seq clustering methods, particularly graph neural networks (GNNs), have significantly improved in tackling the challenges of high-dimension, high-sparsity, and frequent dropout events that lead to ambiguous cell population boundaries. However, their reliance on hard graph constructions derived from thresholded similarity matrices presents challenges:(i) The simplification of intercellular relationships into binary edges (0 or 1) by applying thresholds, which restricts the capture of continuous similarity features among cells and leads to significant information loss.(ii) The presence of significant inter-cluster connections within hard graphs, which can confuse GNN methods that rely heavily on graph structures, potentially causing erroneous message propagation and biased clustering outcomes. To tackle these challenges, we introduce scSGC, a Soft Graph Clustering for single-cell RNA sequencing data, which aims to more accurately characterize continuous similarities among cells through non-binary edge weights, thereby mitigating the limitations of rigid data structures. The scSGC framework comprises three core components: (i) a zero-inflated negative binomial (ZINB)-based feature autoencoder; (ii) a dual-channel cut-informed soft graph embedding module; and (iii) an optimal transport-based clustering optimization module. Extensive experiments across ten datasets demonstrate that scSGC outperforms 13 state-of-the-art clustering models in clustering accuracy, cell type annotation, and computational efficiency. These results highlight its substantial potential to advance scRNA-seq data analysis and deepen our understanding of cellular heterogeneity. 

**Abstract (ZH)**: 基于图的单细胞RNA测序聚类分析在揭示细胞异质性和多样性方面是基础的。软图聚类方法scSGC在处理高维度、高稀疏性和频繁缺失事件导致的细胞群体边界不明确等挑战方面取得了显著改进。然而，这些方法依赖于从阈值相似矩阵派生的硬图构建，这带来了挑战：（i）通过应用阈值将细胞间的关系简化为二元边（0或1），这限制了连续相似性特征的捕捉，导致大量信息丢失。（ii）硬图中存在显著的跨簇连接，这可能使严重依赖图结构的GNN方法产生错误的消息传播和偏向性的聚类结果。为解决这些挑战，我们提出了scSGC，一种基于图的单细胞RNA测序软聚类方法，旨在通过非二元边权重更准确地刻画细胞间的连续相似性，从而缓解刚性数据结构的限制。scSGC框架包括三个核心组件：（i）零 inflation 负二项式（ZINB）特征自编码器；（ii）双通道切割信息软图嵌入模块；（iii）基于最优传输的聚类优化模块。在十个多组学数据集上的广泛实验表明，scSGC在聚类准确性、细胞类型注释和计算效率方面均优于13种最先进的聚类模型。这些结果突显了其在推进单细胞RNA测序数据分析和深化对细胞异质性的理解方面的巨大潜力。 

---
# NeuTSFlow: Modeling Continuous Functions Behind Time Series Forecasting 

**Title (ZH)**: NeuTSFlow：建模时间序列预测背后的连续函数 

**Authors**: Huibo Xu, Likang Wu, Xianquan Wang, Haoning Dang, Chun-Wun Cheng, Angelica I Aviles-Rivero, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09888)  

**Abstract**: Time series forecasting is a fundamental task with broad applications, yet conventional methods often treat data as discrete sequences, overlooking their origin as noisy samples of continuous processes. Crucially, discrete noisy observations cannot uniquely determine a continuous function; instead, they correspond to a family of plausible functions. Mathematically, time series can be viewed as noisy observations of a continuous function family governed by a shared probability measure. Thus, the forecasting task can be framed as learning the transition from the historical function family to the future function family. This reframing introduces two key challenges: (1) How can we leverage discrete historical and future observations to learn the relationships between their underlying continuous functions? (2) How can we model the transition path in function space from the historical function family to the future function family? To address these challenges, we propose NeuTSFlow, a novel framework that leverages Neural Operators to facilitate flow matching for learning path of measure between historical and future function families. By parameterizing the velocity field of the flow in infinite-dimensional function spaces, NeuTSFlow moves beyond traditional methods that focus on dependencies at discrete points, directly modeling function-level features instead. Experiments on diverse forecasting tasks demonstrate NeuTSFlow's superior accuracy and robustness, validating the effectiveness of the function-family perspective. 

**Abstract (ZH)**: 时间序列预测是具有广泛应用的基础任务，但传统方法往往将数据视为离散序列，忽视了它们作为连续过程的嘈杂样本的本质。关键在于，离散的嘈杂观测值不能唯一确定一个连续函数，而对应于由共享概率测度支配的一系列可能的函数。从数学角度来看，时间序列可以被视为由共享概率测度支配的一系列连续函数的嘈杂观测值。因此，预测任务可以重新框定为从历史函数家族到未来函数家族的学习连续函数过渡。这一重新框定引入了两个关键挑战：（1）我们如何利用历史和未来的离散观测值来学习它们背后连续函数之间的关系？（2）我们如何在函数空间中建模从历史函数家族到未来函数家族的过渡路径？为了解决这些挑战，我们提出了一种名为NeuTSFlow的新框架，该框架利用神经算子促进流匹配，以学习历史和未来函数家族之间的测度路径。通过在无限维函数空间中参数化流的速度场，NeuTSFlow超越了传统的关注离散点依赖性的方法，直接建模函数级特征。在多种预测任务上的实验结果表明，NeuTSFlow在准确性和鲁棒性方面具有优势，验证了函数家族视角的有效性。 

---
# TolerantECG: A Foundation Model for Imperfect Electrocardiogram 

**Title (ZH)**: 容忍ECG：一种适用于不完美心电图的基础模型 

**Authors**: Huynh Nguyen Dang, Thang Pham, Ngan Le, Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09887)  

**Abstract**: The electrocardiogram (ECG) is an essential and effective tool for diagnosing heart diseases. However, its effectiveness can be compromised by noise or unavailability of one or more leads of the standard 12-lead recordings, resulting in diagnostic errors or uncertainty. To address these challenges, we propose TolerantECG, a foundation model for ECG signals that is robust to noise and capable of functioning with arbitrary subsets of the standard 12-lead ECG. TolerantECG training combines contrastive and self-supervised learning frameworks to jointly learn ECG signal representations alongside their corresponding knowledge-retrieval-based text report descriptions and corrupted or lead-missing signals. Comprehensive benchmarking results demonstrate that TolerantECG consistently ranks as the best or second-best performer across various ECG signal conditions and class levels in the PTB-XL dataset, and achieves the highest performance on the MIT-BIH Arrhythmia Database. 

**Abstract (ZH)**: 心电图（ECG）是诊断心脏疾病的重要而有效的工具。然而，其有效性可能因噪声干扰或标准12导联记录中一个或多个导联的不可用而受损，导致诊断错误或不确定性。为应对这些挑战，我们提出了TolerantECG，这是一种针对噪声具有鲁棒性的基础模型，能够在任意子集的标准化12导联ECG导联缺失的情况下正常工作。TolerantECG的训练结合了对比学习和自我监督学习框架，共同学习ECG信号表示及其相应的基于知识检索的文本报告描述和受损害或导联缺失的信号。全面的基准测试结果表明，在PTB-XL数据集中，TolerantECG在各种ECG信号条件和类级别上始终表现为最佳或第二佳性能，在MIT-BIH心律失常数据库中达到最高性能。 

---
# Covering a Few Submodular Constraints and Applications 

**Title (ZH)**: 覆盖少数子模约束及其应用 

**Authors**: Tanvi Bajpai, Chandra Chekuri, Pooja Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2507.09879)  

**Abstract**: We consider the problem of covering multiple submodular constraints. Given a finite ground set $N$, a cost function $c: N \rightarrow \mathbb{R}_+$, $r$ monotone submodular functions $f_1,f_2,\ldots,f_r$ over $N$ and requirements $b_1,b_2,\ldots,b_r$ the goal is to find a minimum cost subset $S \subseteq N$ such that $f_i(S) \ge b_i$ for $1 \le i \le r$. When $r=1$ this is the well-known Submodular Set Cover problem. Previous work \cite{chekuri2022covering} considered the setting when $r$ is large and developed bi-criteria approximation algorithms, and approximation algorithms for the important special case when each $f_i$ is a weighted coverage function. These are fairly general models and capture several concrete and interesting problems as special cases. The approximation ratios for these problem are at least $\Omega(\log r)$ which is unavoidable when $r$ is part of the input. In this paper, motivated by some recent applications, we consider the problem when $r$ is a \emph{fixed constant} and obtain two main results. For covering multiple submodular constraints we obtain a randomized bi-criteria approximation algorithm that for any given integer $\alpha \ge 1$ outputs a set $S$ such that $f_i(S) \ge$ $(1-1/e^\alpha -\epsilon)b_i$ for each $i \in [r]$ and $\mathbb{E}[c(S)] \le (1+\epsilon)\alpha \cdot \sf{OPT}$. Second, when the $f_i$ are weighted coverage functions from a deletion-closed set system we obtain a $(1+\epsilon)$ $(\frac{e}{e-1})$ $(1+\beta)$-approximation where $\beta$ is the approximation ratio for the underlying set cover instances via the natural LP. These results show that one can obtain nearly as good an approximation for any fixed $r$ as what one would achieve for $r=1$. We mention some applications that follow easily from these general results and anticipate more in the future. 

**Abstract (ZH)**: 多子模约束的覆盖问题 

---
# ViTCoT: Video-Text Interleaved Chain-of-Thought for Boosting Video Understanding in Large Language Models 

**Title (ZH)**: ViTCoT: 视频-文本交错链式思考增强大型语言模型的视频理解 

**Authors**: Yongheng Zhang, Xu Liu, Ruihan Tao, Qiguang Chen, Hao Fei, Wanxiang Che, Libo Qin  

**Link**: [PDF](https://arxiv.org/pdf/2507.09876)  

**Abstract**: Video understanding plays a vital role in bridging low-level visual signals with high-level cognitive reasoning, and is fundamental to applications such as autonomous driving, embodied AI, and the broader pursuit of AGI. The rapid development of large language models (LLMs), particularly those utilizing Chain-of-Thought (CoT) technology, has significantly advanced video reasoning capabilities. However, current approaches primarily depend on textual information for reasoning, overlooking the visual modality in the actual video reasoning process. In contrast, humans naturally re-examine visual content while reasoning. Motivated by this, we introduce a novel video reasoning paradigm: Video-Text Interleaved CoT (ViTCoT), which facilitates more intuitive and cognitively aligned reasoning. To the end, first, we construct the Video-Text Interleaved Benchmark (ViTIB), which is created using MLLMs for key-video selection and manually verified. Furthermore, we extensively explore the potential of the ViTCoT paradigm in the video understanding field. Extensive experiments demonstrate that ViTCoT significantly enhances performance compared to the traditional text-only CoT paradigm and effectively activates more neuron values in MLLMs. 

**Abstract (ZH)**: 视频理解在连接低级视觉信号与高级认知推理中发挥着关键作用，是自主驾驶、具身AI及更广泛追求人工通用 Intelligence 的基础。大型语言模型（LLMs），尤其是利用链式思考（CoT）技术的模型，迅速发展并显著提升了视频推理能力。然而，当前的方法主要依赖文本信息进行推理，忽视了实际视频推理过程中的视觉模态。相比之下，人类在推理时自然会重新审视视觉内容。受此启发，我们提出了一种新的视频推理范式：视频-文本交替链式思考（ViTCoT），以促进更为直观和认知上一致的推理。为此，我们构建了视频-文本交替基准（ViTIB），该基准使用MLLMs进行关键视频选择，并由人工验证。此外，我们广泛探索了ViTCoT范式在视频理解领域的潜力。大量实验证明，与传统的仅文本链式思考范式相比，ViTCoT显著提高了性能，并激活了更多神经元值。 

---
# Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition 

**Title (ZH)**: 功能归纳与任务泛化：基于一位加法的可解释性研究 

**Authors**: Qinyuan Ye, Robin Jia, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2507.09875)  

**Abstract**: Large language models demonstrate the intriguing ability to perform unseen tasks via in-context learning. However, it remains unclear what mechanisms inside the model drive such task-level generalization. In this work, we approach this question through the lens of off-by-one addition (i.e., 1+1=3, 2+2=5, 3+3=?), a two-step, counterfactual task with an unexpected +1 function as a second step. Leveraging circuit-style interpretability techniques such as path patching, we analyze the models' internal computations behind their notable performance and present three key findings. First, we uncover a function induction mechanism that explains the model's generalization from standard addition to off-by-one addition. This mechanism resembles the structure of the induction head mechanism found in prior work and elevates it to a higher level of abstraction. Second, we show that the induction of the +1 function is governed by multiple attention heads in parallel, each of which emits a distinct piece of the +1 function. Finally, we find that this function induction mechanism is reused in a broader range of tasks, including synthetic tasks such as shifted multiple-choice QA and algorithmic tasks such as base-8 addition. Overall, our findings offer deeper insights into how reusable and composable structures within language models enable task-level generalization. 

**Abstract (ZH)**: 大型语言模型通过上下文学习表现出对未见任务的 intriguing 能力，但模型内部驱动这种任务级泛化的机制尚不清楚。本文通过探讨偏离加法（例如 1+1=3, 2+2=5, 3+3=?）这一两步、反事实任务，其中第二步骤包含意外的+1函数，来回答这一问题。利用电路风格的可解释性技术（如路径修补），我们分析了模型在这些任务中出色表现背后的内部计算，并提出了三个关键发现。首先，我们揭示了一种功能归纳机制，解释了模型如何从标准加法泛化到偏离加法。该机制类似于先前工作中发现的归纳头部机制，并将其提升到更高的抽象层次。其次，我们证明+1函数的归纳由多个并行的注意力头控制，每个头负责产生+1函数的特定部分。最后，我们发现这种功能归纳机制在更广泛的任务中得到重用，包括合成任务（如偏移的选择题问答）和算法性任务（如八进制加法）。总之，我们的发现为语言模型内部可重用和可组合结构如何促进任务级泛化提供了更深入的见解。 

---
# Task Priors: Enhancing Model Evaluation by Considering the Entire Space of Downstream Tasks 

**Title (ZH)**: 下游任务先验：通过考虑整个下游任务空间提升模型评估 

**Authors**: Niket Patel, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2507.09871)  

**Abstract**: The grand goal of AI research, and particularly Self Supervised Learning (SSL), is to produce systems that can successfully solve any possible task. In contrast, current evaluation methods available to AI researchers typically rely on a fixed collection of hand-picked downstream benchmarks. Hence, a large amount of effort is put into designing and searching for large collection of evaluation tasks that can serve as a proxy of our grand goal. We argue that such a rigid evaluation protocol creates a silent bottleneck in AI research. To remedy that, we define a probabilistic space of downstream tasks obtained by adopting a distribution of tasks and by defining Task Priors. Under this view, one can evaluate a model's performance over the set of all possible downstream tasks. Our framework is the first to provide answers to key questions such as (i) what is the average performance of my model over all possible downstream tasks weighted by the probability to encounter each task? or (ii) what is the variance of my model's performance across all downstream tasks under the defined Task Priors? Beyond establishing a new standard for evaluation, we believe that Task Priors will accelerate the pace of research in SSL - where downstream task evaluation is the sole qualitative signal that researchers have access to. 

**Abstract (ZH)**: AI研究的宏大目标，特别是自监督学习（SSL），是产生能够成功解决任何可能任务的系统。与之形成对比的是，当前可用的AI评价方法通常依赖于固定的手选下游基准。因此，AI研究者需要花费大量努力来设计和寻找作为宏大目标代理的评价任务集合。我们认为，这样一种僵化的评价方案在AI研究中形成了一种隐形的瓶颈。为了弥补这一不足，我们通过采用任务分布并定义任务先验来定义一个概率意义上的下游任务空间。从这一视角出发，可以评估模型在所有可能的下游任务集合上的性能。我们的框架首次提供了关键问题的答案，例如（i）在考虑每个任务出现概率加权的情况下，我的模型在所有可能的下游任务上的平均表现如何？或（ii）在定义的任务先验下，我的模型在所有下游任务上的性能差异是多少？超越建立新的评价标准，我们相信任务先验将加速自监督学习中的研究进展——在自监督学习中，下游任务评估是研究者唯一可以获得的定性信号。 

---
# Turning the Tide: Repository-based Code Reflection 

**Title (ZH)**: 逆流而上：基于仓库的代码反思 

**Authors**: Wei Zhang, Jian Yang, Jiaxi Yang, Ya Wang, Zhoujun Li, Zeyu Cui, Binyuan Hui, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.09866)  

**Abstract**: Code large language models (LLMs) enhance programming by understanding and generating code across languages, offering intelligent feedback, bug detection, and code updates through reflection, improving development efficiency and accessibility. While benchmarks (e.g. HumanEval/LiveCodeBench) evaluate code generation and real-world relevance, previous works ignore the scenario of modifying code in repositories. Considering challenges remaining in improving reflection capabilities and avoiding data contamination in dynamic benchmarks, we introduce LiveRepoReflection, a challenging benchmark for evaluating code understanding and generation in multi-file repository contexts, featuring 1,888 rigorously filtered test cases across $6$ programming languages to ensure diversity, correctness, and high difficulty. Further, we create RepoReflection-Instruct, a large-scale, quality-filtered instruction-tuning dataset derived from diverse sources, used to train RepoReflectionCoder through a two-turn dialogue process involving code generation and error-driven repair. The leaderboard evaluates over 40 LLMs to reflect the model performance of repository-based code reflection. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过跨语言理解与生成代码、提供智能反馈、检测错误和代码更新，从而提高编程效率和可访问性。虽然基准测试（如HumanEval/LiveCodeBench）评估代码生成能力和实际相关性，但先前的工作忽略了代码仓库中的代码修改场景。鉴于在改进反射能力和避免动态基准中数据污染方面仍存在的挑战，我们引入了LiveRepoReflection，这是一个挑战性的基准，用于评估多文件代码仓库环境中的代码理解与生成能力，包含6种编程语言共计1,888个严格的测试用例，以确保多样性、正确性和高难度。此外，我们创建了RepoReflection-Instruct，这是一个大规模、高质量的指令调优数据集，来源于多种来源，并通过包含代码生成和错误驱动修复的两轮对话过程用于训练RepoReflectionCoder。排行榜评估了超过40个LLM，以反映基于代码仓库的代码反射模型的性能。 

---
# Intersection of Reinforcement Learning and Bayesian Optimization for Intelligent Control of Industrial Processes: A Safe MPC-based DPG using Multi-Objective BO 

**Title (ZH)**: 基于多目标贝叶斯优化的强化学习与安全模型预测控制相结合的智能工业过程控制方法：一种安全的基于多目标贝叶斯优化的DPG方法 

**Authors**: Hossein Nejatbakhsh Esfahani, Javad Mohammadpour Velni  

**Link**: [PDF](https://arxiv.org/pdf/2507.09864)  

**Abstract**: Model Predictive Control (MPC)-based Reinforcement Learning (RL) offers a structured and interpretable alternative to Deep Neural Network (DNN)-based RL methods, with lower computational complexity and greater transparency. However, standard MPC-RL approaches often suffer from slow convergence, suboptimal policy learning due to limited parameterization, and safety issues during online adaptation. To address these challenges, we propose a novel framework that integrates MPC-RL with Multi-Objective Bayesian Optimization (MOBO). The proposed MPC-RL-MOBO utilizes noisy evaluations of the RL stage cost and its gradient, estimated via a Compatible Deterministic Policy Gradient (CDPG) approach, and incorporates them into a MOBO algorithm using the Expected Hypervolume Improvement (EHVI) acquisition function. This fusion enables efficient and safe tuning of the MPC parameters to achieve improved closed-loop performance, even under model imperfections. A numerical example demonstrates the effectiveness of the proposed approach in achieving sample-efficient, stable, and high-performance learning for control systems. 

**Abstract (ZH)**: 基于MPC的RL与多目标贝叶斯优化的融合：一种具有高效和安全调参能力的方法 

---
# A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends 

**Title (ZH)**: 基于MLLM的富视觉文档理解综述：方法、挑战及新兴趋势 

**Authors**: Yihao Ding, Siwen Luo, Yue Dai, Yanbei Jiang, Zechuan Li, Geoffrey Martin, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2507.09861)  

**Abstract**: Visually-Rich Document Understanding (VRDU) has emerged as a critical field, driven by the need to automatically process documents containing complex visual, textual, and layout information. Recently, Multimodal Large Language Models (MLLMs) have shown remarkable potential in this domain, leveraging both Optical Character Recognition (OCR)-dependent and OCR-free frameworks to extract and interpret information in document images. This survey reviews recent advancements in MLLM-based VRDU, highlighting three core components: (1) methods for encoding and fusing textual, visual, and layout features; (2) training paradigms, including pretraining strategies, instruction-response tuning, and the trainability of different model modules; and (3) datasets utilized for pretraining, instruction-tuning, and supervised fine-tuning. Finally, we discuss the challenges and opportunities in this evolving field and propose future directions to advance the efficiency, generalizability, and robustness of VRDU systems. 

**Abstract (ZH)**: 富视觉文档理解（VRDU）已成为一项关键领域，由自动处理包含复杂视觉、文本和布局信息的文档的需求推动。近年来，多模态大型语言模型（MLLMs）在该领域展现出了显著潜力，利用依赖光学字符识别（OCR）和非OCR框架来提取和解析文档图像中的信息。本文综述了基于MLLM的VRDU的最新进展，强调了三个核心组成部分：（1）文本、视觉和布局特征的编码与融合方法；（2）训练范式，包括预训练策略、指令-响应调优以及不同模型模块的可训练性；以及（3）用于预训练、指令调优和监督微调的数据集。最后，我们讨论了该领域面临的挑战和机遇，并提出了推进VRDU系统的效率、泛化能力和鲁棒性的未来方向。 

---
# Secure and Efficient UAV-Based Face Detection via Homomorphic Encryption and Edge Computing 

**Title (ZH)**: 基于同态加密和边缘计算的无人机Face检测安全与效率提升 

**Authors**: Nguyen Van Duc, Bui Duc Manh, Quang-Trung Luu, Dinh Thai Hoang, Van-Linh Nguyen, Diep N. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09860)  

**Abstract**: This paper aims to propose a novel machine learning (ML) approach incorporating Homomorphic Encryption (HE) to address privacy limitations in Unmanned Aerial Vehicles (UAV)-based face detection. Due to challenges related to distance, altitude, and face orientation, high-resolution imagery and sophisticated neural networks enable accurate face recognition in dynamic environments. However, privacy concerns arise from the extensive surveillance capabilities of UAVs. To resolve this issue, we propose a novel framework that integrates HE with advanced neural networks to secure facial data throughout the inference phase. This method ensures that facial data remains secure with minimal impact on detection accuracy. Specifically, the proposed system leverages the Cheon-Kim-Kim-Song (CKKS) scheme to perform computations directly on encrypted data, optimizing computational efficiency and security. Furthermore, we develop an effective data encoding method specifically designed to preprocess the raw facial data into CKKS form in a Single-Instruction-Multiple-Data (SIMD) manner. Building on this, we design a secure inference algorithm to compute on ciphertext without needing decryption. This approach not only protects data privacy during the processing of facial data but also enhances the efficiency of UAV-based face detection systems. Experimental results demonstrate that our method effectively balances privacy protection and detection performance, making it a viable solution for UAV-based secure face detection. Significantly, our approach (while maintaining data confidentially with HE encryption) can still achieve an accuracy of less than 1% compared to the benchmark without using encryption. 

**Abstract (ZH)**: 基于同态加密的无人机面部检测新型机器学习方法 

---
# Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training 

**Title (ZH)**: 穿过河流：理解无排程方法在语言模型训练中的优势 

**Authors**: Minhak Song, Beomhan Baek, Kwangjun Ahn, Chulhee Yun  

**Link**: [PDF](https://arxiv.org/pdf/2507.09846)  

**Abstract**: As both model and dataset sizes continue to scale rapidly, conventional pretraining strategies with fixed compute budgets-such as cosine learning rate schedules-are increasingly inadequate for large-scale training. Recent alternatives, including warmup-stable-decay (WSD) schedules and weight averaging, offer greater flexibility. However, WSD relies on explicit decay phases to track progress, while weight averaging addresses this limitation at the cost of additional memory. In search of a more principled and scalable alternative, we revisit the Schedule-Free (SF) method [Defazio et al., 2024], which has shown strong empirical performance across diverse settings. We show that SF-AdamW effectively navigates the "river" structure of the loss landscape without decay phases or auxiliary averaging, making it particularly suitable for continuously scaling training workloads. To understand this behavior, we conduct a theoretical and empirical analysis of SF dynamics, revealing that it implicitly performs weight averaging without memory overhead. Guided by this analysis, we propose a refined variant of SF that improves robustness to momentum and performs better under large batch sizes, addressing key limitations of the original method. Together, these results establish SF as a practical, scalable, and theoretically grounded approach for language model training. 

**Abstract (ZH)**: 随着模型和数据集规模的快速扩大，传统的具有固定计算预算的预训练策略（如余弦学习率调度）越来越不适合大规模训练。最近的替代方法，包括warmup-stable-decay (WSD)调度和权重平均，提供了更大的灵活性。然而，WSD依赖于显式的衰减阶段来跟踪进度，而权重平均则以增加内存使用为代价解决了这一局限性。为了寻求更为原则性和可扩展的替代方案，我们重新审视了Defazio等人提出的Schedule-Free (SF)方法[Defazio et al., 2024]，该方法在多种场景中显示出了强大的实证性能。我们表明，SF-AdamW 能有效地在损失景观的“河流”结构中导航而无需衰减阶段或辅助平均，使其特别适合连续扩展的训练工作负载。为了理解这种行为，我们对SF动力学进行了理论和实证分析，揭示了它在不增加内存开销的情况下隐式进行了权重平均。基于这一分析，我们提出了一种改进的SF变体，增强了鲁棒性并适用于大规模批次，从而解决原始方法的关键限制。这些结果共同确立了SF作为一种实用、可扩展且具理论依据的语言模型训练方法。 

---
# A Pre-training Framework for Relational Data with Information-theoretic Principles 

**Title (ZH)**: 基于信息论原则的关系数据预训练框架 

**Authors**: Quang Truong, Zhikai Chen, Mingxuan Ju, Tong Zhao, Neil Shah, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09837)  

**Abstract**: Relational databases underpin critical infrastructure across a wide range of domains, yet the design of generalizable pre-training strategies for learning from relational databases remains an open challenge due to task heterogeneity. Specifically, there exist infinitely many possible downstream tasks, as tasks are defined based on relational schema graphs, temporal dependencies, and SQL-defined label logics. An effective pre-training framework is desired to take these factors into account in order to obtain task-aware representations. By incorporating knowledge of the underlying distribution that drives label generation, downstream tasks can benefit from relevant side-channel information. To bridge this gap, we introduce Task Vector Estimation (TVE), a novel pre-training framework that constructs predictive supervisory signals via set-based aggregation over schema traversal graphs, explicitly modeling next-window relational dynamics. We formalize our approach through an information-theoretic lens, demonstrating that task-informed representations retain more relevant signals than those obtained without task priors. Extensive experiments on the RelBench benchmark show that TVE consistently outperforms traditional pre-training baselines. Our findings advocate for pre-training objectives that encode task heterogeneity and temporal structure as design principles for predictive modeling on relational databases. 

**Abstract (ZH)**: 面向关系数据库的预训练框架：建模任务异构性和时序结构以获得任务感知表示 

---
# Multi-residual Mixture of Experts Learning for Cooperative Control in Multi-vehicle Systems 

**Title (ZH)**: 多残差专家混合学习在多车辆系统协同控制中的应用 

**Authors**: Vindula Jayawardana, Sirui Li, Yashar Farid, Cathy Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09836)  

**Abstract**: Autonomous vehicles (AVs) are becoming increasingly popular, with their applications now extending beyond just a mode of transportation to serving as mobile actuators of a traffic flow to control flow dynamics. This contrasts with traditional fixed-location actuators, such as traffic signals, and is referred to as Lagrangian traffic control. However, designing effective Lagrangian traffic control policies for AVs that generalize across traffic scenarios introduces a major challenge. Real-world traffic environments are highly diverse, and developing policies that perform robustly across such diverse traffic scenarios is challenging. It is further compounded by the joint complexity of the multi-agent nature of traffic systems, mixed motives among participants, and conflicting optimization objectives subject to strict physical and external constraints. To address these challenges, we introduce Multi-Residual Mixture of Expert Learning (MRMEL), a novel framework for Lagrangian traffic control that augments a given suboptimal nominal policy with a learned residual while explicitly accounting for the structure of the traffic scenario space. In particular, taking inspiration from residual reinforcement learning, MRMEL augments a suboptimal nominal AV control policy by learning a residual correction, but at the same time dynamically selects the most suitable nominal policy from a pool of nominal policies conditioned on the traffic scenarios and modeled as a mixture of experts. We validate MRMEL using a case study in cooperative eco-driving at signalized intersections in Atlanta, Dallas Fort Worth, and Salt Lake City, with real-world data-driven traffic scenarios. The results show that MRMEL consistently yields superior performance-achieving an additional 4%-9% reduction in aggregate vehicle emissions relative to the strongest baseline in each setting. 

**Abstract (ZH)**: 自主驾驶车辆（AVs）在交通控制中的拉格朗日控制研究：一种多残留专家混合学习框架（MRMEL） 

---
# Generative Cognitive Diagnosis 

**Title (ZH)**: 生成认知诊断 

**Authors**: Jiatong Li, Qi Liu, Mengxiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09831)  

**Abstract**: Cognitive diagnosis (CD) models latent cognitive states of human learners by analyzing their response patterns on diagnostic tests, serving as a crucial machine learning technique for educational assessment and evaluation. Traditional cognitive diagnosis models typically follow a transductive prediction paradigm that optimizes parameters to fit response scores and extract learner abilities. These approaches face significant limitations as they cannot perform instant diagnosis for new learners without computationally expensive retraining and produce diagnostic outputs with limited reliability. In this study, we introduces a novel generative diagnosis paradigm that fundamentally shifts CD from predictive to generative modeling, enabling inductive inference of cognitive states without parameter re-optimization. We propose two simple yet effective instantiations of this paradigm: Generative Item Response Theory (G-IRT) and Generative Neural Cognitive Diagnosis Model (G-NCDM), which achieve excellent performance improvements over traditional methods. The generative approach disentangles cognitive state inference from response prediction through a well-designed generation process that incorporates identifiability and monotonicity conditions. Extensive experiments on real-world datasets demonstrate the effectiveness of our methodology in addressing scalability and reliability challenges, especially $\times 100$ speedup for the diagnosis of new learners. Our framework opens new avenues for cognitive diagnosis applications in artificial intelligence, particularly for intelligent model evaluation and intelligent education systems. The code is available at this https URL. 

**Abstract (ZH)**: 认知诊断模型通过分析人类学习者在诊断测试中的反应模式来latent认知状态，作为教育评估与评价中重要的机器学习技术。传统的认知诊断模型通常遵循一种归纳预测范式，通过优化参数来拟合反应分数并提取学习者能力。这些方法存在显著的局限性，无法在不进行昂贵的重新训练的情况下对新学习者进行即时诊断，并且生成的诊断输出可靠性较低。本研究引入了一种新的生成诊断范式，从根本上将认知诊断从预测建模转变为生成建模，从而无需重新优化参数即可进行归纳推理以推断认知状态。我们提出了两种简单而有效的该范式的实例：生成项目反应理论（G-IRT）和生成神经认知诊断模型（G-NCDM），这些方法在传统方法上取得了卓越的性能改进。生成方法通过精心设计的生成过程将认知状态推断与反应预测分离，同时满足鉴别性和单调性条件。在实际数据集上的大量实验表明，我们的方法在解决可扩展性和可靠性挑战方面非常有效，特别是对于新学习者的诊断速度提高了100倍。我们的框架为人工智能中的认知诊断应用开辟了新途径，特别是在智能模型评估和智能教育系统中。代码可在以下链接获取：this https URL。 

---
# Bridging Neural Networks and Dynamic Time Warping for Adaptive Time Series Classification 

**Title (ZH)**: 将神经网络与动态时间规整结合用于自适应时间序列分类 

**Authors**: Jintao Qu, Zichong Wang, Chenhao Wu, Wenbin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09826)  

**Abstract**: Neural networks have achieved remarkable success in time series classification, but their reliance on large amounts of labeled data for training limits their applicability in cold-start scenarios. Moreover, they lack interpretability, reducing transparency in decision-making. In contrast, dynamic time warping (DTW) combined with a nearest neighbor classifier is widely used for its effectiveness in limited-data settings and its inherent interpretability. However, as a non-parametric method, it is not trainable and cannot leverage large amounts of labeled data, making it less effective than neural networks in rich-resource scenarios. In this work, we aim to develop a versatile model that adapts to cold-start conditions and becomes trainable with labeled data, while maintaining interpretability. We propose a dynamic length-shortening algorithm that transforms time series into prototypes while preserving key structural patterns, thereby enabling the reformulation of the DTW recurrence relation into an equivalent recurrent neural network. Based on this, we construct a trainable model that mimics DTW's alignment behavior. As a neural network, it becomes trainable when sufficient labeled data is available, while still retaining DTW's inherent interpretability. We apply the model to several benchmark time series classification tasks and observe that it significantly outperforms previous approaches in low-resource settings and remains competitive in rich-resource settings. 

**Abstract (ZH)**: 基于动态长度缩短的可训练时间序列分类模型 

---
# Compressed Computation: Dense Circuits in a Toy Model of the Universal-AND Problem 

**Title (ZH)**: 压缩计算：通用AND问题玩具模型中的密集电路 

**Authors**: Adam Newgas  

**Link**: [PDF](https://arxiv.org/pdf/2507.09816)  

**Abstract**: Neural networks are capable of superposition -- representing more features than there are dimensions. Recent work considers the analogous concept for computation instead of storage, proposing theoretical constructions. But there has been little investigation into whether these circuits can be learned in practice. In this work, we investigate a toy model for the Universal-AND problem which computes the AND of all $m\choose 2$ pairs of $m$ sparse inputs. The hidden dimension that determines the number of non-linear activations is restricted to pressure the model to find a compute-efficient circuit, called compressed computation. We find that the training process finds a simple solution that does not correspond to theoretical constructions. It is fully dense -- every neuron contributes to every output. The solution circuit naturally scales with dimension, trading off error rates for neuron efficiency. It is similarly robust to changes in sparsity and other key parameters, and extends naturally to other boolean operations and boolean circuits. We explain the found solution in detail and compute why it is more efficient than the theoretical constructions at low sparsity. Our findings shed light on the types of circuits that models like to form and the flexibility of the superposition representation. This contributes to a broader understanding of network circuitry and interpretability. 

**Abstract (ZH)**: 神经网络具备叠加能力——表示的特征维度超过输入维度。最近的研究考虑了类似的概念，即在计算而非存储中实现叠加，提出了理论构想。但实践中这些电路是否可以被学习尚缺乏探讨。本文探讨了一个玩具模型，用于研究通用-AND问题，该模型计算m个稀疏输入的所有组合中两两输入的AND。隐藏维度限制在非线性激活的数量，以迫使模型找到一个计算高效的电路，称为压缩计算。我们发现训练过程找到了一个简单的解决方案，该解决方案不对应于理论构想。解决方案电路在维度上自然扩展，权衡误差率和神经元效率。该解决方案对稀疏性以及其他关键参数的变化具有类似的鲁棒性，并自然扩展到其他布尔操作和布尔电路。我们详细解释了找到的解决方案，并计算其在低稀疏性下比理论构想更高效的理由。我们的发现揭示了模型倾向于形成的电路类型以及叠加表示的灵活性。这有助于更广泛地理解网络电路和可解释性。 

---
# Federated Learning with Graph-Based Aggregation for Traffic Forecasting 

**Title (ZH)**: 基于图聚合的联邦学习在交通预测中的应用 

**Authors**: Audri Banik, Glaucio Haroldo Silva de Carvalho, Renata Dividino  

**Link**: [PDF](https://arxiv.org/pdf/2507.09805)  

**Abstract**: In traffic prediction, the goal is to estimate traffic speed or flow in specific regions or road segments using historical data collected by devices deployed in each area. Each region or road segment can be viewed as an individual client that measures local traffic flow, making Federated Learning (FL) a suitable approach for collaboratively training models without sharing raw data. In centralized FL, a central server collects and aggregates model updates from multiple clients to build a shared model while preserving each client's data privacy. Standard FL methods, such as Federated Averaging (FedAvg), assume that clients are independent, which can limit performance in traffic prediction tasks where spatial relationships between clients are important. Federated Graph Learning methods can capture these dependencies during server-side aggregation, but they often introduce significant computational overhead. In this paper, we propose a lightweight graph-aware FL approach that blends the simplicity of FedAvg with key ideas from graph learning. Rather than training full models, our method applies basic neighbourhood aggregation principles to guide parameter updates, weighting client models based on graph connectivity. This approach captures spatial relationships effectively while remaining computationally efficient. We evaluate our method on two benchmark traffic datasets, METR-LA and PEMS-BAY, and show that it achieves competitive performance compared to standard baselines and recent graph-based federated learning techniques. 

**Abstract (ZH)**: 基于图的联邦学习在交通预测中的轻量级方法 

---
# CADmium: Fine-Tuning Code Language Models for Text-Driven Sequential CAD Design 

**Title (ZH)**: CADmium：基于文本驱动的序列CAD设计代码语言模型微调 

**Authors**: Prashant Govindarajan, Davide Baldelli, Jay Pathak, Quentin Fournier, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2507.09792)  

**Abstract**: Computer-aided design (CAD) is the digital construction of 2D and 3D objects, and is central to a wide range of engineering and manufacturing applications like automobile and aviation. Despite its importance, CAD modeling remains largely a time-intensive, manual task. Recent works have attempted to automate this process with small transformer-based models and handcrafted CAD sequence representations. However, there has been little effort to leverage the potential of large language models (LLMs) for sequential CAD design. In this work, we introduce a new large-scale dataset of more than 170k CAD models annotated with high-quality, human-like descriptions generated with our pipeline based on GPT-4.1. Using this dataset, we fine-tune powerful code-LLMs to generate CAD sequences represented in a JSON-based format from natural language descriptions, demonstrating the viability and effectiveness of this approach for text-conditioned CAD generation. Because simple metrics often fail to reflect the quality of generated objects, we introduce geometric and topological metrics based on sphericity, mean curvature, and Euler characteristic to provide richer structural insights. Our experiments and ablation studies on both synthetic and human-annotated data demonstrate that CADmium is able to automate CAD design, drastically speeding up the design of new objects. The dataset, code, and fine-tuned models are available online. 

**Abstract (ZH)**: 计算机辅助设计（CAD）是2D和3D对象的数字构造，并广泛应用于汽车和航空等工程和制造领域。尽管CAD建模十分重要，但它仍然是一个耗时的手工任务。近期的研究试图通过基于小变压器模型和手工设计的CAD序列表示来自动化这一过程。然而，鲜有研究尝试利用大型语言模型（LLMs）的潜力实现序列CAD设计。本文介绍了一个包含超过170,000个高质量人为描述标注的大型CAD模型数据集，这些描述是基于GPT-4 pipeline生成的。利用该数据集，我们将强大的代码LLM微调，以生成以JSON格式表示的CAD序列，展示了这种方法在文本条件下的CAD生成的有效性和可行性。由于简单的评估指标往往无法反映生成对象的质量，我们还引入了几何和拓扑指标，基于球度、平均曲率和欧拉特征数，以提供更丰富的结构洞察。我们在合成和人工标注数据上的实验和消融研究证明，CADmium能够自动化CAD设计，大幅度加速新对象的设计过程。相关数据集、代码和微调模型已上线。 

---
# Prompting for Performance: Exploring LLMs for Configuring Software 

**Title (ZH)**: 提示优化性能：探索大语言模型在软件配置中的应用 

**Authors**: Helge Spieker, Théo Matricon, Nassim Belmecheri, Jørn Eirik Betten, Gauthier Le Bartz Lyan, Heraldo Borges, Quentin Mazouni, Dennis Gross, Arnaud Gotlieb, Mathieu Acher  

**Link**: [PDF](https://arxiv.org/pdf/2507.09790)  

**Abstract**: Software systems usually provide numerous configuration options that can affect performance metrics such as execution time, memory usage, binary size, or bitrate. On the one hand, making informed decisions is challenging and requires domain expertise in options and their combinations. On the other hand, machine learning techniques can search vast configuration spaces, but with a high computational cost, since concrete executions of numerous configurations are required. In this exploratory study, we investigate whether large language models (LLMs) can assist in performance-oriented software configuration through prompts. We evaluate several LLMs on tasks including identifying relevant options, ranking configurations, and recommending performant configurations across various configurable systems, such as compilers, video encoders, and SAT solvers. Our preliminary results reveal both positive abilities and notable limitations: depending on the task and systems, LLMs can well align with expert knowledge, whereas hallucinations or superficial reasoning can emerge in other cases. These findings represent a first step toward systematic evaluations and the design of LLM-based solutions to assist with software configuration. 

**Abstract (ZH)**: 大型语言模型在性能导向的软件配置中的辅助作用：基于提示的探索性研究 

---
# TinyTroupe: An LLM-powered Multiagent Persona Simulation Toolkit 

**Title (ZH)**: TinyTroupe: 一个基于LLM的多智能体人设模拟工具包 

**Authors**: Paulo Salem, Robert Sim, Christopher Olsen, Prerit Saxena, Rafael Barcelos, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.09788)  

**Abstract**: Recent advances in Large Language Models (LLM) have led to a new class of autonomous agents, renewing and expanding interest in the area. LLM-powered Multiagent Systems (MAS) have thus emerged, both for assistive and simulation purposes, yet tools for realistic human behavior simulation -- with its distinctive challenges and opportunities -- remain underdeveloped. Existing MAS libraries and tools lack fine-grained persona specifications, population sampling facilities, experimentation support, and integrated validation, among other key capabilities, limiting their utility for behavioral studies, social simulation, and related applications. To address these deficiencies, in this work we introduce TinyTroupe, a simulation toolkit enabling detailed persona definitions (e.g., nationality, age, occupation, personality, beliefs, behaviors) and programmatic control via numerous LLM-driven mechanisms. This allows for the concise formulation of behavioral problems of practical interest, either at the individual or group level, and provides effective means for their solution. TinyTroupe's components are presented using representative working examples, such as brainstorming and market research sessions, thereby simultaneously clarifying their purpose and demonstrating their usefulness. Quantitative and qualitative evaluations of selected aspects are also provided, highlighting possibilities, limitations, and trade-offs. The approach, though realized as a specific Python implementation, is meant as a novel conceptual contribution, which can be partially or fully incorporated in other contexts. The library is available as open source at this https URL. 

**Abstract (ZH)**: Recent Advances in Large Language Models Have Led to a New Class of Autonomous Agents, Reinvigorating and Expanding Interest in Multiagent Systems. TinyTroupe: A Simulation Toolkit Enabling Detailed Persona Definitions and Programmatic Control via LLM-Driven Mechanisms 

---
# BitParticle: Partializing Sparse Dual-Factors to Build Quasi-Synchronizing MAC Arrays for Energy-efficient DNNs 

**Title (ZH)**: BitParticle: 部分化稀疏双因子以构建近同步MAC阵列实现能效DNN 

**Authors**: Feilong Qiaoyuan, Jihe Wang, Zhiyu Sun, Linying Wu, Yuanhua Xiao, Danghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09780)  

**Abstract**: Bit-level sparsity in quantized deep neural networks (DNNs) offers significant potential for optimizing Multiply-Accumulate (MAC) operations. However, two key challenges still limit its practical exploitation. First, conventional bit-serial approaches cannot simultaneously leverage the sparsity of both factors, leading to a complete waste of one factor' s sparsity. Methods designed to exploit dual-factor sparsity are still in the early stages of exploration, facing the challenge of partial product explosion. Second, the fluctuation of bit-level sparsity leads to variable cycle counts for MAC operations. Existing synchronous scheduling schemes that are suitable for dual-factor sparsity exhibit poor flexibility and still result in significant underutilization of MAC units. To address the first challenge, this study proposes a MAC unit that leverages dual-factor sparsity through the emerging particlization-based approach. The proposed design addresses the issue of partial product explosion through simple control logic, resulting in a more area- and energy-efficient MAC unit. In addition, by discarding less significant intermediate results, the design allows for further hardware simplification at the cost of minor accuracy loss. To address the second challenge, a quasi-synchronous scheme is introduced that adds cycle-level elasticity to the MAC array, reducing pipeline stalls and thereby improving MAC unit utilization. Evaluation results show that the exact version of the proposed MAC array architecture achieves a 29.2% improvement in area efficiency compared to the state-of-the-art bit-sparsity-driven architecture, while maintaining comparable energy efficiency. The approximate variant further improves energy efficiency by 7.5%, compared to the exact version. Index-Terms: DNN acceleration, Bit-level sparsity, MAC unit 

**Abstract (ZH)**: 比特级稀疏性在量化深度神经网络中的乘累加操作优化潜力巨大，但仍有两个关键挑战限制了其实用性。首先，传统的位串行方法不能同时利用两个因子的稀疏性，导致一个因子的稀疏性完全浪费。设计用于利用双因子稀疏性的方法仍处于早期探索阶段，面临部分乘积爆炸的挑战。其次，比特级稀疏性的波动导致乘累加操作的循环计数变化。现有的适用于双因子稀疏性的同步调度方案表现出较差的灵活性，仍导致乘累加单元的显著未充分利用。为解决第一个挑战，本研究提出了一种通过新兴的粒子化方法利用双因子稀疏性的乘累加单元。所提出的架构通过简单的控制逻辑解决了部分乘积爆炸的问题，从而实现更小面积和能耗的乘累加单元。此外，通过抛弃较不重要的中间结果，设计允许进一步简化硬件，但会轻微损失准确性。为解决第二个挑战，引入了一种准同步方案，为乘累加阵列增加了循环级弹性，减少流水线停滞，从而改善乘累加单元的利用率。评估结果显示，所提出乘累加阵列架构的精确版本相比最先进的比特稀疏性驱动架构在面积效率上提高了29.2%，同时保持了相当的能效。近似版本进一步提高了7.5%的能效，相比精确版本。索引术语：深度神经网络加速，比特级稀疏性，乘累加单元。 

---
# Toward accurate RUL and SOH estimation using reinforced graph-based PINNs enhanced with dynamic weights 

**Title (ZH)**: 基于强化图的PINNs和动态权重增强的准确剩余使用寿命和健康状态估算 

**Authors**: Mohamadreza Akbari Pour, Ali Ghasemzadeh, MohamadAli Bijarchi, Mohammad Behshad Shafii  

**Link**: [PDF](https://arxiv.org/pdf/2507.09766)  

**Abstract**: Accurate estimation of Remaining Useful Life (RUL) and State of Health (SOH) is essential for Prognostics and Health Management (PHM) across a wide range of industrial applications. We propose a novel framework -- Reinforced Graph-Based Physics-Informed Neural Networks Enhanced with Dynamic Weights (RGPD) -- that combines physics-based supervision with advanced spatio-temporal learning. Graph Convolutional Recurrent Networks (GCRNs) embed graph-convolutional filters within recurrent units to capture how node representations evolve over time. Graph Attention Convolution (GATConv) leverages a self-attention mechanism to compute learnable, edge-wise attention coefficients, dynamically weighting neighbor contributions for adaptive spatial aggregation. A Soft Actor-Critic (SAC) module is positioned between the Temporal Attention Unit (TAU) and GCRN to further improve the spatio-temporal learning. This module improves attention and prediction accuracy by dynamically scaling hidden representations to minimize noise and highlight informative features. To identify the most relevant physical constraints in each area, Q-learning agents dynamically assign weights to physics-informed loss terms, improving generalization across real-time industrial systems and reducing the need for manual tuning. In both RUL and SOH estimation tasks, the proposed method consistently outperforms state-of-the-art models, demonstrating strong robustness and predictive accuracy across varied degradation patterns across three diverse industrial benchmark datasets. 

**Abstract (ZH)**: 基于强化图的物理引导神经网络及其动态权重增强的剩余使用寿命和健康状态估计框架 

---
# EventHunter: Dynamic Clustering and Ranking of Security Events from Hacker Forum Discussions 

**Title (ZH)**: EventHunter: 来自黑客论坛讨论的 security 事件的动态聚类和排名 

**Authors**: Yasir Ech-Chammakhy, Anas Motii, Anass Rabii, Jaafar Chbili  

**Link**: [PDF](https://arxiv.org/pdf/2507.09762)  

**Abstract**: Hacker forums provide critical early warning signals for emerging cybersecurity threats, but extracting actionable intelligence from their unstructured and noisy content remains a significant challenge. This paper presents an unsupervised framework that automatically detects, clusters, and prioritizes security events discussed across hacker forum posts. Our approach leverages Transformer-based embeddings fine-tuned with contrastive learning to group related discussions into distinct security event clusters, identifying incidents like zero-day disclosures or malware releases without relying on predefined keywords. The framework incorporates a daily ranking mechanism that prioritizes identified events using quantifiable metrics reflecting timeliness, source credibility, information completeness, and relevance. Experimental evaluation on real-world hacker forum data demonstrates that our method effectively reduces noise and surfaces high-priority threats, enabling security analysts to mount proactive responses. By transforming disparate hacker forum discussions into structured, actionable intelligence, our work addresses fundamental challenges in automated threat detection and analysis. 

**Abstract (ZH)**: 黑客论坛提供早期关键警告信号以应对新兴网络安全威胁，但从其未结构化和嘈杂的内容中提取可操作的情报仍是一项重大挑战。本文提出了一种无监督框架，该框架可自动检测、聚类并优先处理黑客论坛帖子中讨论的安全事件。该方法利用对比学习微调的Transformer嵌入将相关讨论分组为不同的安全事件集群，无需依赖预定义关键词即可识别像零日披露或恶意软件发布等事件。该框架结合了每日排名机制，使用反映及时性、来源可信度、信息完整性和相关性的可量化指标优先处理识别出的事件。实验评价表明，我们的方法有效地减少了噪音并揭示了高优先级威胁，使安全分析师能够采取主动响应措施。通过将分散的黑客论坛讨论转化为结构化的可操作情报，我们的工作解决了自动威胁检测和分析中的根本性挑战。 

---
# AI-Enhanced Pediatric Pneumonia Detection: A CNN-Based Approach Using Data Augmentation and Generative Adversarial Networks (GANs) 

**Title (ZH)**: 基于数据增强和生成对抗网络（GANs）的AI增强儿童肺炎检测：一种CNN方法 

**Authors**: Abdul Manaf, Nimra Mughal  

**Link**: [PDF](https://arxiv.org/pdf/2507.09759)  

**Abstract**: Pneumonia is a leading cause of mortality in children under five, requiring accurate chest X-ray diagnosis. This study presents a machine learning-based Pediatric Chest Pneumonia Classification System to assist healthcare professionals in diagnosing pneumonia from chest X-ray images. The CNN-based model was trained on 5,863 labeled chest X-ray images from children aged 0-5 years from the Guangzhou Women and Children's Medical Center. To address limited data, we applied augmentation techniques (rotation, zooming, shear, horizontal flipping) and employed GANs to generate synthetic images, addressing class imbalance. The system achieved optimal performance using combined original, augmented, and GAN-generated data, evaluated through accuracy and F1 score metrics. The final model was deployed via a Flask web application, enabling real-time classification with probability estimates. Results demonstrate the potential of deep learning and GANs in improving diagnostic accuracy and efficiency for pediatric pneumonia classification, particularly valuable in resource-limited clinical settings this https URL 

**Abstract (ZH)**: 儿童肺炎的胸部X光诊断：基于机器学习的儿科胸部肺炎分类系统 

---
# Universal Physics Simulation: A Foundational Diffusion Approach 

**Title (ZH)**: 通用物理仿真：一种基础扩散方法 

**Authors**: Bradley Camburn  

**Link**: [PDF](https://arxiv.org/pdf/2507.09733)  

**Abstract**: We present the first foundational AI model for universal physics simulation that learns physical laws directly from boundary-condition data without requiring a priori equation encoding. Traditional physics-informed neural networks (PINNs) and finite-difference methods necessitate explicit mathematical formulation of governing equations, fundamentally limiting their generalizability and discovery potential. Our sketch-guided diffusion transformer approach reimagines computational physics by treating simulation as a conditional generation problem, where spatial boundary conditions guide the synthesis of physically accurate steady-state solutions.
By leveraging enhanced diffusion transformer architectures with novel spatial relationship encoding, our model achieves direct boundary-to-equilibrium mapping and is generalizable to diverse physics domains. Unlike sequential time-stepping methods that accumulate errors over iterations, our approach bypasses temporal integration entirely, directly generating steady-state solutions with SSIM > 0.8 while maintaining sub-pixel boundary accuracy. Our data-informed approach enables physics discovery through learned representations analyzable via Layer-wise Relevance Propagation (LRP), revealing emergent physical relationships without predetermined mathematical constraints. This work represents a paradigm shift from AI-accelerated physics to AI-discovered physics, establishing the first truly universal physics simulation framework. 

**Abstract (ZH)**: 我们提出了第一个用于通用物理仿真的一般性AI模型，该模型能够直接从边界条件数据中学习物理定律，无需事先编码微分方程。传统的物理感知神经网络（PINNs）和有限差分方法需要显式地制定支配方程，从根本上限制了其泛化能力和发现潜力。我们的草图引导扩散变换器方法重新构想了计算物理，将仿真视为一个条件生成问题，其中空间边界条件指导生成物理准确的稳态解。

通过利用增强的扩散变换器架构和新颖的空间关系编码，我们的模型实现了直接从边界到平衡状态的映射，并能够泛化到多种物理领域。不同于累积误差的顺序时间步进方法，我们的方法完全绕过了时间积分，直接生成SSIM > 0.8的稳态解，同时保持亚像素边界精度。我们的数据驱动方法通过可解释层间相关性传播（LRP）学习表示，使物理发现成为可能，揭示了无预先数学约束的新兴物理关系。这项工作代表了从AI加速的物理到AI发现的物理的范式转变，建立了第一个真正通用的物理仿真框架。 

---
# Visual Homing in Outdoor Robots Using Mushroom Body Circuits and Learning Walks 

**Title (ZH)**: 户外机器人使用蘑菇体电路和学习行走的视觉归巢技术 

**Authors**: Gabriel G. Gattaux, Julien R. Serres, Franck Ruffier, Antoine Wystrach  

**Link**: [PDF](https://arxiv.org/pdf/2507.09725)  

**Abstract**: Ants achieve robust visual homing with minimal sensory input and only a few learning walks, inspiring biomimetic solutions for autonomous navigation. While Mushroom Body (MB) models have been used in robotic route following, they have not yet been applied to visual homing. We present the first real-world implementation of a lateralized MB architecture for visual homing onboard a compact autonomous car-like robot. We test whether the sign of the angular path integration (PI) signal can categorize panoramic views, acquired during learning walks and encoded in the MB, into "goal on the left" and "goal on the right" memory banks, enabling robust homing in natural outdoor settings. We validate this approach through four incremental experiments: (1) simulation showing attractor-like nest dynamics; (2) real-world homing after decoupled learning walks, producing nest search behavior; (3) homing after random walks using noisy PI emulated with GPS-RTK; and (4) precise stopping-at-the-goal behavior enabled by a fifth MB Output Neuron (MBON) encoding goal-views to control velocity. This mimics the accurate homing behavior of ants and functionally resembles waypoint-based position control in robotics, despite relying solely on visual input. Operating at 8 Hz on a Raspberry Pi 4 with 32x32 pixel views and a memory footprint under 9 kB, our system offers a biologically grounded, resource-efficient solution for autonomous visual homing. 

**Abstract (ZH)**: 蚂蚁通过最少的感觉输入和几次学习行走实现稳健的视觉归巢，这启发了自主导航的生物模拟解决方案。虽然蘑菇体（MB）模型在机器人路径跟随中已被使用，但尚未应用于视觉归巢。我们首次在紧凑型自主车形机器人上实现了偏侧化MB架构的实地视觉归巢实施。我们测试角路径积分（PI）信号的符号是否能够分类学习行走期间获取并编码在蘑菇体中的全景视图，形成“目标在左”和“目标在右”的记忆库，从而在自然户外环境中实现稳健的归巢。我们通过四个逐步实验验证了这一方法：（1）仿真显示类似吸引子的巢穴动力学；（2）解耦学习行走后的实地归巢，产生巢穴搜索行为；（3）使用GPS-RTK模拟噪声路径积分的随机行走归巢；（4）通过第五个蘑菇体输出神经元（MBON）编码目标视图来控制速度，实现精确的到达目标行为。这模拟了蚂蚁精确的归巢行为，并在很大程度上类似于机器人中基于航点的位置控制，尽管仅依赖于视觉输入。以8 Hz运行在Raspberry Pi 4上，具有32x32像素视图和内存占用小于9 kB，我们的系统提供了一种基于生物学的、资源高效的自主视觉归巢解决方案。 

---
# EPT-2 Technical Report 

**Title (ZH)**: EPT-2 技术报告 

**Authors**: Roberto Molinaro, Niall Siegenheim, Niels Poulsen, Jordan Dane Daubinet, Henry Martin, Mark Frey, Kevin Thiart, Alexander Jakob Dautel, Andreas Schlueter, Alex Grigoryev, Bogdan Danciu, Nikoo Ekhtiari, Bas Steunebrink, Leonie Wagner, Marvin Vincent Gabler  

**Link**: [PDF](https://arxiv.org/pdf/2507.09703)  

**Abstract**: We present EPT-2, the latest iteration in our Earth Physics Transformer (EPT) family of foundation AI models for Earth system forecasting. EPT-2 delivers substantial improvements over its predecessor, EPT-1.5, and sets a new state of the art in predicting energy-relevant variables-including 10m and 100m wind speed, 2m temperature, and surface solar radiation-across the full 0-240h forecast horizon. It consistently outperforms leading AI weather models such as Microsoft Aurora, as well as the operational numerical forecast system IFS HRES from the European Centre for Medium-Range Weather Forecasts (ECMWF). In parallel, we introduce a perturbation-based ensemble model of EPT-2 for probabilistic forecasting, called EPT-2e. Remarkably, EPT-2e significantly surpasses the ECMWF ENS mean-long considered the gold standard for medium- to longrange forecasting-while operating at a fraction of the computational cost. EPT models, as well as third-party forecasts, are accessible via the this http URL platform. 

**Abstract (ZH)**: EPT-2：地球物理变换器家族的最新一代地球系统预报基础AI模型 

---
# Frequency-aware Surrogate Modeling With SMT Kernels For Advanced Data Forecasting 

**Title (ZH)**: 频率意识的替代建模方法结合SMT内核用于高级数据预测 

**Authors**: Nicolas Gonel, Paul Saves, Joseph Morlier  

**Link**: [PDF](https://arxiv.org/pdf/2507.09694)  

**Abstract**: This paper introduces a comprehensive open-source framework for developing correlation kernels, with a particular focus on user-defined and composition of kernels for surrogate modeling. By advancing kernel-based modeling techniques, we incorporate frequency-aware elements that effectively capture complex mechanical behaviors and timefrequency dynamics intrinsic to aircraft systems. Traditional kernel functions, often limited to exponential-based methods, are extended to include a wider range of kernels such as exponential squared sine and rational quadratic kernels, along with their respective firstand second-order derivatives. The proposed methodologies are first validated on a sinus cardinal test case and then applied to forecasting Mauna-Loa Carbon Dioxide (CO 2 ) concentrations and airline passenger traffic. All these advancements are integrated into the open-source Surrogate Modeling Toolbox (SMT 2.0), providing a versatile platform for both standard and customizable kernel configurations. Furthermore, the framework enables the combination of various kernels to leverage their unique strengths into composite models tailored to specific problems. The resulting framework offers a flexible toolset for engineers and researchers, paving the way for numerous future applications in metamodeling for complex, frequency-sensitive domains. 

**Abstract (ZH)**: 本文介绍了一个全面的开源框架，用于开发相关核函数，特别关注用户定义的核函数及其组合在代理建模中的应用。通过推进基于核的建模技术，我们引入了频率感知元素，有效地捕捉到飞机系统中固有的复杂机械行为和时频动态。传统的核函数通常局限于基于指数的方法，扩展到了包括指数平方正弦和理性二次核及其一阶和二阶导数在内的更广泛范围的核函数。所提出的方法首先在正弦卡丹测试案例上进行验证，然后应用于预测夏威夷冒纳罗亚二氧化碳浓度和航空旅客流量。所有这些进展整合到了开源代理建模工具箱（SMT 2.0）中，提供了一个既适用于标准配置又易于定制的核配置的多功能平台。此外，该框架允许将多种核函数组合起来，利用它们各自的优点，定制针对特定问题的复合模型。最终形成的框架为工程师和研究人员提供了一个灵活的工具集，为复杂、频率敏感领域的大规模元建模开辟了诸多可能应用。 

---
# Post-Training Quantization of Generative and Discriminative LSTM Text Classifiers: A Study of Calibration, Class Balance, and Robustness 

**Title (ZH)**: 生成性和判别性LSTM文本分类器的后训练量化研究：校准、类别平衡与鲁棒性探索 

**Authors**: Md Mushfiqur Rahaman, Elliot Chang, Tasmiah Haque, Srinjoy Das  

**Link**: [PDF](https://arxiv.org/pdf/2507.09687)  

**Abstract**: Text classification plays a pivotal role in edge computing applications like industrial monitoring, health diagnostics, and smart assistants, where low latency and high accuracy are both key requirements. Generative classifiers, in particular, have been shown to exhibit robustness to out-of-distribution and noisy data, which is an extremely critical consideration for deployment in such real-time edge environments. However, deploying such models on edge devices faces computational and memory constraints. Post Training Quantization (PTQ) reduces model size and compute costs without retraining, making it ideal for edge deployment. In this work, we present a comprehensive comparative study of generative and discriminative Long Short Term Memory (LSTM)-based text classification models with PTQ using the Brevitas quantization library. We evaluate both types of classifier models across multiple bitwidths and assess their robustness under regular and noisy input conditions. We find that while discriminative classifiers remain robust, generative ones are more sensitive to bitwidth, calibration data used during PTQ, and input noise during quantized inference. We study the influence of class imbalance in calibration data for both types of classifiers, comparing scenarios with evenly and unevenly distributed class samples including their effect on weight adjustments and activation profiles during PTQ. Using test statistics derived from nonparametric hypothesis testing, we identify that using class imbalanced data during calibration introduces insufficient weight adaptation at lower bitwidths for generative LSTM classifiers, thereby leading to degraded performance. This study underscores the role of calibration data in PTQ and when generative classifiers succeed or fail under noise, aiding deployment in edge environments. 

**Abstract (ZH)**: 生成式文本分类模型在边缘计算环境中的后训练量化研究 

---
# OrQstrator: An AI-Powered Framework for Advanced Quantum Circuit Optimization 

**Title (ZH)**: OrQstrator: 一种基于人工智能的高级量子电路优化框架 

**Authors**: Laura Baird, Armin Moin  

**Link**: [PDF](https://arxiv.org/pdf/2507.09682)  

**Abstract**: We propose a novel approach, OrQstrator, which is a modular framework for conducting quantum circuit optimization in the Noisy Intermediate-Scale Quantum (NISQ) era. Our framework is powered by Deep Reinforcement Learning (DRL). Our orchestration engine intelligently selects among three complementary circuit optimizers: A DRL-based circuit rewriter trained to reduce depth and gate count via learned rewrite sequences; a domain-specific optimizer that performs efficient local gate resynthesis and numeric optimization; a parameterized circuit instantiator that improves compilation by optimizing template circuits during gate set translation. These modules are coordinated by a central orchestration engine that learns coordination policies based on circuit structure, hardware constraints, and backend-aware performance features such as gate count, depth, and expected fidelity. The system outputs an optimized circuit for hardware-aware transpilation and execution, leveraging techniques from an existing state-of-the-art approach, called the NISQ Analyzer, to adapt to backend constraints. 

**Abstract (ZH)**: 我们提出了一种新颖的方法OrQstrator，这是一种针对Noisy Intermediate-Scale Quantum (NISQ)时代的量子电路优化的模块化框架，该框架由深度强化学习（DRL）驱动。我们的编排引擎智能地选择三种互补的电路优化器：一种基于DRL的电路重写器，通过学习到的重写序列减少电路深度和门数；一种领域特定的优化器，执行高效的局部门重建和数值优化；一种参数化的电路实例化器，在门集转换过程中通过优化模板电路提高编译效率。这些模块由一个中心编排引擎协调，该引擎根据电路结构、硬件约束以及门数、深度和预期保真度的后端感知性能特征学习协调策略。系统生成一个针对硬件感知转化和执行优化的电路，利用一种现有的先进方法——NISQ Analyzer——的技术来适应后端约束。 

---
# Conformal Prediction for Privacy-Preserving Machine Learning 

**Title (ZH)**: 隐私保护机器学习中的齐性预测方法 

**Authors**: Alexander David Balinsky, Dominik Krzeminski, Alexander Balinsky  

**Link**: [PDF](https://arxiv.org/pdf/2507.09678)  

**Abstract**: We investigate the integration of Conformal Prediction (CP) with supervised learning on deterministically encrypted data, aiming to bridge the gap between rigorous uncertainty quantification and privacy-preserving machine learning. Using AES-encrypted variants of the MNIST dataset, we demonstrate that CP methods remain effective even when applied directly in the encrypted domain, owing to the preservation of data exchangeability under fixed-key encryption. We test traditional $p$-value-based against $e$-value-based conformal predictors. Our empirical evaluation reveals that models trained on deterministically encrypted data retain the ability to extract meaningful structure, achieving 36.88\% test accuracy -- significantly above random guessing (9.56\%) observed with per-instance encryption. Moreover, $e$-value-based CP achieves predictive set coverage of over 60\% with 4.3 loss-threshold calibration, correctly capturing the true label in 4888 out of 5000 test cases. In contrast, the $p$-value-based CP yields smaller predictive sets but with reduced coverage accuracy. These findings highlight both the promise and limitations of CP in encrypted data settings and underscore critical trade-offs between prediction set compactness and reliability. %Our work sets a foundation for principled uncertainty quantification in secure, privacy-aware learning systems. 

**Abstract (ZH)**: 我们研究规范预测（CP）与确定性加密数据上监督学习的整合，旨在弥合严格不确定性量化与隐私保护机器学习之间的差距。使用AES加密的MNIST数据变体，我们证明即使直接在加密域中应用CP方法也能保持有效性，这归因于固定密钥加密下数据可交换性的保留。我们测试了基于$p$-值的传统方法和基于$e$-值的方法。实证评估表明，训练于确定性加密数据上的模型仍能提取有意义的结构，测试准确率达到36.88%，远远高于实例加密时随机猜测的9.56%。此外，基于$e$-值的CP在4.3损失阈值校准下实现了超过60%的预测集覆盖率，正确捕捉到真实标签的有4888个测试案例中的4888个。相比之下，基于$p$-值的CP生成的预测集更小，但覆盖率准确性较低。这些发现突显了CP在加密数据环境中既有的潜力和限制，并强调了预测集紧凑性和可靠性之间的关键权衡。 

---
# SimStep: Chain-of-Abstractions for Incremental Specification and Debugging of AI-Generated Interactive Simulations 

**Title (ZH)**: SimStep: 层次抽象方法实现AI生成互动模拟的增量规范与调试 

**Authors**: Zoe Kaputa, Anika Rajaram, Vryan Almanon Feliciano, Zhuoyue Lyu, Maneesh Agrawala, Hari Subramonyam  

**Link**: [PDF](https://arxiv.org/pdf/2507.09664)  

**Abstract**: Programming-by-prompting with generative AI offers a new paradigm for end-user programming, shifting the focus from syntactic fluency to semantic intent. This shift holds particular promise for non-programmers such as educators, who can describe instructional goals in natural language to generate interactive learning content. Yet in bypassing direct code authoring, many of programming's core affordances - such as traceability, stepwise refinement, and behavioral testing - are lost. We propose the Chain-of-Abstractions (CoA) framework as a way to recover these affordances while preserving the expressive flexibility of natural language. CoA decomposes the synthesis process into a sequence of cognitively meaningful, task-aligned representations that function as checkpoints for specification, inspection, and refinement. We instantiate this approach in SimStep, an authoring environment for teachers that scaffolds simulation creation through four intermediate abstractions: Concept Graph, Scenario Graph, Learning Goal Graph, and UI Interaction Graph. To address ambiguities and misalignments, SimStep includes an inverse correction process that surfaces in-filled model assumptions and enables targeted revision without requiring users to manipulate code. Evaluations with educators show that CoA enables greater authoring control and interpretability in programming-by-prompting workflows. 

**Abstract (ZH)**: 基于生成式AI的提示编程为终端用户编程提供了新范式，重点从语法流畅转向语义意图。这种转变特别适合如教育者等非编程人员，他们可以用自然语言描述教学目标以生成互动学习内容。然而，通过 bypass 直接代码编写，编程的核心功能，如可追踪性、逐步细化和行为测试，都会丢失。我们提出了抽象链（CoA）框架，以在保持自然语言表达灵活性的同时恢复这些功能。CoA 将合成过程分解为一系列认知上有意义、任务对齐的表示，作为规范、检查和细化的检查点。我们在 SimStep 中实例化了这一方法，SimStep 是一种为教师搭建仿真创建环境的作者工具，通过四个中间抽象层次的支持：概念图、情景图、学习目标图和 UI 交互图，帮助教师实现仿真内容的逐步构造。为了应对歧义和不一致，SimStep 包括一个逆向修正过程，该过程揭示了填充的模型假设，并允许有针对性的修订而无需用户操作代码。教育者的研究表明，CoA 使提示编程工作流程中的作者控制和可解释性得以提升。 

---
# KEN: Knowledge Augmentation and Emotion Guidance Network for Multimodal Fake News Detection 

**Title (ZH)**: KEN: 基于知识增强和情绪指导的多模态假新闻检测网络 

**Authors**: Peican Zhu, Yubo Jing, Le Cheng, Keke Tang, Yangming Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.09647)  

**Abstract**: In recent years, the rampant spread of misinformation on social media has made accurate detection of multimodal fake news a critical research focus. However, previous research has not adequately understood the semantics of images, and models struggle to discern news authenticity with limited textual information. Meanwhile, treating all emotional types of news uniformly without tailored approaches further leads to performance degradation. Therefore, we propose a novel Knowledge Augmentation and Emotion Guidance Network (KEN). On the one hand, we effectively leverage LVLM's powerful semantic understanding and extensive world knowledge. For images, the generated captions provide a comprehensive understanding of image content and scenes, while for text, the retrieved evidence helps break the information silos caused by the closed and limited text and context. On the other hand, we consider inter-class differences between different emotional types of news through balanced learning, achieving fine-grained modeling of the relationship between emotional types and authenticity. Extensive experiments on two real-world datasets demonstrate the superiority of our KEN. 

**Abstract (ZH)**: 近年来，社交媒体上虚假信息的盛行使得多模态假新闻的准确检测成为关键研究方向。然而，先前的研究未能充分理解图像的语义，模型在仅靠有限的文字信息时难以辨别新闻的真实性。同时，不加区分地处理不同情感类型的新闻导致性能下降。因此，我们提出了一种新型的知识增强和情感引导网络（KEN）。一方面，我们有效地利用了LVLM强大的语义理解和广泛的世界知识。对于图像，生成的字幕提供了对图像内容和场景的全面理解，而对于文本，则检索到的证据有助于打破由封闭和有限的文字和上下文造成的信息孤岛。另一方面，我们通过平衡学习考虑了不同类型新闻之间的情感差异，实现了对情感类型与真实性之间关系的细致建模。在两个真实世界数据集上的广泛实验表明了KEN的优点。 

---
# Brain Stroke Detection and Classification Using CT Imaging with Transformer Models and Explainable AI 

**Title (ZH)**: 基于变压器模型和可解释人工智能的CT影像脑卒中检测与分类 

**Authors**: Shomukh Qari, Maha A. Thafar  

**Link**: [PDF](https://arxiv.org/pdf/2507.09630)  

**Abstract**: Stroke is one of the leading causes of death globally, making early and accurate diagnosis essential for improving patient outcomes, particularly in emergency settings where timely intervention is critical. CT scans are the key imaging modality because of their speed, accessibility, and cost-effectiveness. This study proposed an artificial intelligence framework for multiclass stroke classification (ischemic, hemorrhagic, and no stroke) using CT scan images from a dataset provided by the Republic of Turkey's Ministry of Health. The proposed method adopted MaxViT, a state-of-the-art Vision Transformer, as the primary deep learning model for image-based stroke classification, with additional transformer variants (vision transformer, transformer-in-transformer, and ConvNext). To enhance model generalization and address class imbalance, we applied data augmentation techniques, including synthetic image generation. The MaxViT model trained with augmentation achieved the best performance, reaching an accuracy and F1-score of 98.00%, outperforming all other evaluated models and the baseline methods. The primary goal of this study was to distinguish between stroke types with high accuracy while addressing crucial issues of transparency and trust in artificial intelligence models. To achieve this, Explainable Artificial Intelligence (XAI) was integrated into the framework, particularly Grad-CAM++. It provides visual explanations of the model's decisions by highlighting relevant stroke regions in the CT scans and establishing an accurate, interpretable, and clinically applicable solution for early stroke detection. This research contributed to the development of a trustworthy AI-assisted diagnostic tool for stroke, facilitating its integration into clinical practice and enhancing access to timely and optimal stroke diagnosis in emergency departments, thereby saving more lives. 

**Abstract (ZH)**: 全球范围内中风是导致死亡的主要原因之一，因此早期和准确的诊断对于改善患者结局尤为重要，尤其是在需要及时干预的急诊环境中。由于CT扫描速度快、易于获取且成本效益高，CT扫描是关键的影像学检查方式。本研究提出了一种基于CT扫描图像的多类中风分类人工智能框架（缺血性中风、出血性中风和无中风），该数据集由土耳其共和国卫生部提供。所提方法采用当前最先进的眼动变换器MaxViT作为基于图像的中风分类的主要深度学习模型，并结合了多种变体（包括眼动变换器、双重眼动变换器和ConvNext）。为增强模型泛化能力和解决类别不平衡问题，我们应用了数据增强技术，包括合成图像生成。使用增强技术训练的MaxViT模型取得了最佳性能，准确率为98.00%，F1分数为98.00%，超过了所有其他评估模型和基线方法。本研究的主要目标是通过提高人工智能模型的透明度和信任度来高精度地区分不同类型的中风。为此，我们整合了可解释的人工智能(XAI)，特别是Grad-CAM++，通过在CT扫描上突出显示相关中风区域来提供模型决策的视觉解释，从而为早期中风检测提供了准确、可解释且临床适用的解决方案。该研究促进了可信赖的人工智能辅助诊断工具的发展，有助于其在临床实践中的应用，从而增强急诊科中及时和最优中风诊断的可及性，挽救更多生命。 

---
# DRAGD: A Federated Unlearning Data Reconstruction Attack Based on Gradient Differences 

**Title (ZH)**: DRAGD：基于梯度差的数据重建联邦遗忘攻击 

**Authors**: Bocheng Ju, Junchao Fan, Jiaqi Liu, Xiaolin Chang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09602)  

**Abstract**: Federated learning enables collaborative machine learning while preserving data privacy. However, the rise of federated unlearning, designed to allow clients to erase their data from the global model, introduces new privacy concerns. Specifically, the gradient exchanges during the unlearning process can leak sensitive information about deleted data. In this paper, we introduce DRAGD, a novel attack that exploits gradient discrepancies before and after unlearning to reconstruct forgotten data. We also present DRAGDP, an enhanced version of DRAGD that leverages publicly available prior data to improve reconstruction accuracy, particularly for complex datasets like facial images. Extensive experiments across multiple datasets demonstrate that DRAGD and DRAGDP significantly outperform existing methods in data this http URL work highlights a critical privacy vulnerability in federated unlearning and offers a practical solution, advancing the security of federated unlearning systems in real-world applications. 

**Abstract (ZH)**: 联邦学习使协作机器学习成为可能的同时保护数据隐私。然而，联邦反学习的兴起旨在允许客户端从全局模型中删除其数据，带来了新的隐私问题。具体而言，反学习过程中的梯度交换可能会泄露被删除数据的敏感信息。在本文中，我们提出了一种新颖的攻击方法DRAGD，利用反学习前后梯度的差异来重建被遗忘的数据。我们还介绍了一种增强版的DRAGD，称为DRAGDP，通过利用公开可用的先验数据来提高复杂数据集（如面部图像）的重建准确性。在多个数据集上的 extensive 实验表明，DRAGD 和 DRAGDP 显著优于现有方法。本文强调了联邦反学习中一个关键的隐私漏洞，并提供了一种实用的解决方案，促进了实际应用中联邦反学习系统的安全性。 

---
# NMIXX: Domain-Adapted Neural Embeddings for Cross-Lingual eXploration of Finance 

**Title (ZH)**: NMIXX: 领域适应的神经嵌入在跨语言金融探索中的应用 

**Authors**: Hanwool Lee, Sara Yu, Yewon Hwang, Jonghyun Choi, Heejae Ahn, Sungbum Jung, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09601)  

**Abstract**: General-purpose sentence embedding models often struggle to capture specialized financial semantics, especially in low-resource languages like Korean, due to domain-specific jargon, temporal meaning shifts, and misaligned bilingual vocabularies. To address these gaps, we introduce NMIXX (Neural eMbeddings for Cross-lingual eXploration of Finance), a suite of cross-lingual embedding models fine-tuned with 18.8K high-confidence triplets that pair in-domain paraphrases, hard negatives derived from a semantic-shift typology, and exact Korean-English translations. Concurrently, we release KorFinSTS, a 1,921-pair Korean financial STS benchmark spanning news, disclosures, research reports, and regulations, designed to expose nuances that general benchmarks miss.
When evaluated against seven open-license baselines, NMIXX's multilingual bge-m3 variant achieves Spearman's rho gains of +0.10 on English FinSTS and +0.22 on KorFinSTS, outperforming its pre-adaptation checkpoint and surpassing other models by the largest margin, while revealing a modest trade-off in general STS performance. Our analysis further shows that models with richer Korean token coverage adapt more effectively, underscoring the importance of tokenizer design in low-resource, cross-lingual settings. By making both models and the benchmark publicly available, we provide the community with robust tools for domain-adapted, multilingual representation learning in finance. 

**Abstract (ZH)**: 面向金融领域的通用句子嵌入模型在低资源语言如韩语中往往难以捕捉到专门的金融语义，原因包括领域特定的专业术语、时间意义的转变以及双语词汇的不一致。为解决这些问题，我们提出了NMIXX（Neural eMbeddings for Cross-lingual eXploration of Finance），这是一种使用18800个高置信度三元组微调的跨语言嵌入模型套件，该三元组包括领域内的同义句对、从语义转变类型中派生的硬负例以及精确的韩英对照翻译。同时，我们也发布了KorFinSTS，这是一个包含1921对样本的韩语金融STS基准数据集，涵盖了新闻、披露信息、研究报告和监管文件，旨在揭示通用基准数据集未能捕捉到的细微差异。在与七种开放许可基准模型进行评估时，NMIXX的多语言bge-m3变体在英语金融STS和KorFinSTS上的Spearman’s ρ得分分别提高了0.10和0.22，超越了其预适应版本和其他模型，并揭示了在通用STS性能方面的轻微折衷。我们的分析还表明，拥有更丰富韩语标记覆盖的模型在跨语言设置中适应得更好，强调了低资源环境下的分词器设计的重要性。通过公开发布模型和基准数据集，我们为社区提供了用于金融领域适应的多语言表示学习的稳健工具。 

---
# THOR: Transformer Heuristics for On-Demand Retrieval 

**Title (ZH)**: THOR: Transformer启发式方法用于按需检索 

**Authors**: Isaac Shi, Zeyuan Li, Fan Liu, Wenli Wang, Lewei He, Yang Yang, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.09592)  

**Abstract**: We introduce the THOR (Transformer Heuristics for On-Demand Retrieval) Module, designed and implemented by eSapiens, a secure, scalable engine that transforms natural-language questions into verified, read-only SQL analytics for enterprise databases. The Text-to-SQL module follows a decoupled orchestration/execution architecture: a Supervisor Agent routes queries, Schema Retrieval dynamically injects table and column metadata, and a SQL Generation Agent emits single-statement SELECT queries protected by a read-only guardrail. An integrated Self-Correction & Rating loop captures empty results, execution errors, or low-quality outputs and triggers up to five LLM-driven regeneration attempts. Finally, a Result Interpretation Agent produces concise, human-readable insights and hands raw rows to the Insight & Intelligence engine for visualization or forecasting.
Smoke tests across finance, sales, and operations scenarios demonstrate reliable ad-hoc querying and automated periodic reporting. By embedding schema awareness, fault-tolerant execution, and compliance guardrails, the THOR Module empowers non-technical users to access live data with zero-SQL simplicity and enterprise-grade safety. 

**Abstract (ZH)**: THOR（Transformer Heuristics for On-Demand Retrieval）模块：安全可扩展的自然语言到SQL转换引擎 

---
# A Serverless Architecture for Real-Time Stock Analysis using Large Language Models: An Iterative Development and Debugging Case Study 

**Title (ZH)**: 使用大型语言模型进行实时股票分析的无服务器架构：一项迭代开发与调试案例研究 

**Authors**: Taniv Ashraf  

**Link**: [PDF](https://arxiv.org/pdf/2507.09583)  

**Abstract**: The advent of powerful, accessible Large Language Models (LLMs) like Google's Gemini presents new opportunities for democratizing financial data analysis. This paper documents the design, implementation, and iterative debugging of a novel, serverless system for real-time stock analysis. The system leverages the Gemini API for qualitative assessment, automates data ingestion and processing via GitHub Actions, and presents the findings through a decoupled, static frontend. We detail the architectural evolution of the system, from initial concepts to a robust, event-driven pipeline, highlighting the practical challenges encountered during deployment. A significant portion of this paper is dedicated to a case study on the debugging process, covering common software errors, platform-specific permission issues, and rare, environment-level platform bugs. The final architecture operates at a near-zero cost, demonstrating a viable model for individuals to build sophisticated AI-powered financial tools. The operational application is publicly accessible, and the complete source code is available for review. We conclude by discussing the role of LLMs in financial analysis, the importance of robust debugging methodologies, and the emerging paradigm of human-AI collaboration in software development. 

**Abstract (ZH)**: 强大的可访问大语言模型（LLMs）如Google的Gemini的出现为金融数据 democratization 提供了新的机遇。本文记录了实时股票分析的新型无服务器系统的架构设计、实现及迭代调试过程。该系统利用Gemini API 进行定性评估，通过GitHub Actions 自动化数据摄取和处理，并通过解耦的静态前端呈现结果。文章详细阐述了系统的架构演变，从最初的概念到一个稳健的、事件驱动的工作流程，强调了部署过程中遇到的实际挑战。论文的大部分内容专注于调试过程的案例研究，涵盖常见的软件错误、平台特定的权限问题以及罕见的环境级平台漏洞。最终架构几乎无成本运行，展示了个体构建复杂AI金融工具的可行模型。该操作应用对公众开放，完整源代码可供审查。本文还讨论了大语言模型在金融分析中的作用、稳健调试方法的重要性，以及软件开发中人-AI协作的新兴范式。 

---
# MENTOR: Efficient Multimodal-Conditioned Tuning for Autoregressive Vision Generation Models 

**Title (ZH)**: MENTOR: 效率优先的多模态条件调谐用于自回归视觉生成模型 

**Authors**: Haozhe Zhao, Zefan Cai, Shuzheng Si, Liang Chen, Jiuxiang Gu, Wen Xiao, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09574)  

**Abstract**: Recent text-to-image models produce high-quality results but still struggle with precise visual control, balancing multimodal inputs, and requiring extensive training for complex multimodal image generation. To address these limitations, we propose MENTOR, a novel autoregressive (AR) framework for efficient Multimodal-conditioned Tuning for Autoregressive multimodal image generation. MENTOR combines an AR image generator with a two-stage training paradigm, enabling fine-grained, token-level alignment between multimodal inputs and image outputs without relying on auxiliary adapters or cross-attention modules. The two-stage training consists of: (1) a multimodal alignment stage that establishes robust pixel- and semantic-level alignment, followed by (2) a multimodal instruction tuning stage that balances the integration of multimodal inputs and enhances generation controllability. Despite modest model size, suboptimal base components, and limited training resources, MENTOR achieves strong performance on the DreamBench++ benchmark, outperforming competitive baselines in concept preservation and prompt following. Additionally, our method delivers superior image reconstruction fidelity, broad task adaptability, and improved training efficiency compared to diffusion-based methods. Dataset, code, and models are available at: this https URL 

**Abstract (ZH)**: recent文本到图像模型生成高质量结果但仍难以实现精确的视觉控制、平衡多模态输入，并且需要大量的训练来生成复杂的多模态图像。为了解决这些局限性，我们提出了一种名为MENTOR的新颖自回归(AR)框架，用于高效多模态条件调节以实现自回归多模态图像生成。MENTOR结合了自回归图像生成器和两阶段训练范式，能够在不依赖辅助适配器或交叉注意力模块的情况下，实现多模态输入与图像输出的细粒度、token级对齐。两阶段训练包括：(1) 多模态对齐阶段，建立稳健的像素级和语义级对齐，随后是 (2) 多模态指令调节阶段，平衡多模态输入的集成并增强生成可控性。尽管模型规模 modest、基组件 suboptimal且训练资源有限，MENTOR 在 DreamBench++基准测试中仍表现出色，超越了竞争对手的基础模型在概念保留和提示遵循方面的性能。此外，我们的方法还实现了优于扩散模型的优点，包括更高的图像重建保真度、更广泛的任务适应性和改进的训练效率。数据集、代码和模型可在以下网址获取：this https URL。 

---
# Identifying Offline Metrics that Predict Online Impact: A Pragmatic Strategy for Real-World Recommender Systems 

**Title (ZH)**: 识别 Offline 计量指标以预测 Online 影响：面向实际推荐系统的实用策略 

**Authors**: Timo Wilm, Philipp Normann  

**Link**: [PDF](https://arxiv.org/pdf/2507.09566)  

**Abstract**: A critical challenge in recommender systems is to establish reliable relationships between offline and online metrics that predict real-world performance. Motivated by recent advances in Pareto front approximation, we introduce a pragmatic strategy for identifying offline metrics that align with online impact. A key advantage of this approach is its ability to simultaneously serve multiple test groups, each with distinct offline performance metrics, in an online experiment controlled by a single model. The method is model-agnostic for systems with a neural network backbone, enabling broad applicability across architectures and domains. We validate the strategy through a large-scale online experiment in the field of session-based recommender systems on the OTTO e-commerce platform. The online experiment identifies significant alignments between offline metrics and real-word click-through rate, post-click conversion rate and units sold. Our strategy provides industry practitioners with a valuable tool for understanding offline-to-online metric relationships and making informed, data-driven decisions. 

**Abstract (ZH)**: 推荐系统中的一个关键挑战是建立可靠的离线和在线指标关系以预测实际性能。受Pareto前沿近似最近进展的启发，我们提出了一种实用策略来识别与在线影响相一致的离线指标。该方法的关键优势在于能够同时为由单个模型控制的在线实验中的多个具有不同离线性能指标的测试组提供服务。该方法对具有神经网络骨干的系统具有模型无关性，使其能够在多种架构和领域中广泛应用。我们通过在OTTO电子商务平台基于会话的推荐系统领域进行大规模在线实验验证了该策略。在线实验发现，离线指标与点击率、点击后转化率和销售单位之间存在显著关联。我们的策略为工业从业者提供了一个有价值的工具，以理解离线到在线指标关系并做出基于数据的决策。 

---
# Prompt Engineering in Segment Anything Model: Methodologies, Applications, and Emerging Challenges 

**Title (ZH)**: Segment Anything模型中的提示工程：方法、应用及新兴挑战 

**Authors**: Yidong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09562)  

**Abstract**: The Segment Anything Model (SAM) has revolutionized image segmentation through its innovative prompt-based approach, yet the critical role of prompt engineering in its success remains underexplored. This paper presents the first comprehensive survey focusing specifically on prompt engineering techniques for SAM and its variants. We systematically organize and analyze the rapidly growing body of work in this emerging field, covering fundamental methodologies, practical applications, and key challenges. Our review reveals how prompt engineering has evolved from simple geometric inputs to sophisticated multimodal approaches, enabling SAM's adaptation across diverse domains including medical imaging and remote sensing. We identify unique challenges in prompt optimization and discuss promising research directions. This survey fills an important gap in the literature by providing a structured framework for understanding and advancing prompt engineering in foundation models for segmentation. 

**Abstract (ZH)**: Segment Anything模型（SAM）通过其创新的提示基础方法在图像分割领域取得了革命性的进展，但提示工程在其成功中的关键作用仍鲜有探索。本文提供了首个专注于SAM及其变种的提示工程技术的全面综述。我们系统地组织和分析了这一新兴领域中迅速增长的研究成果，涵盖基本方法、实用应用以及关键挑战。我们的综述揭示了提示工程从简单的几何输入发展到复杂的多模态方法的过程，使SAM能够跨医学成像和遥感等不同领域进行适应。我们指出了提示优化的独特挑战，并讨论了有希望的研究方向。本文通过提供一个结构化的框架来理解和推动基础模型中的提示工程的发展，填补了文献中的重要空白。 

---
# On the Importance of Neural Membrane Potential Leakage for LIDAR-based Robot Obstacle Avoidance using Spiking Neural Networks 

**Title (ZH)**: 基于视觉脉冲神经网络的LIDAR机器人避障中神经膜电位泄漏的重要性 

**Authors**: Zainab Ali, Lujayn Al-Amir, Ali Safa  

**Link**: [PDF](https://arxiv.org/pdf/2507.09538)  

**Abstract**: Using neuromorphic computing for robotics applications has gained much attention in recent year due to the remarkable ability of Spiking Neural Networks (SNNs) for high-precision yet low memory and compute complexity inference when implemented in neuromorphic hardware. This ability makes SNNs well-suited for autonomous robot applications (such as in drones and rovers) where battery resources and payload are typically limited. Within this context, this paper studies the use of SNNs for performing direct robot navigation and obstacle avoidance from LIDAR data. A custom robot platform equipped with a LIDAR is set up for collecting a labeled dataset of LIDAR sensing data together with the human-operated robot control commands used for obstacle avoidance. Crucially, this paper provides what is, to the best of our knowledge, a first focused study about the importance of neuron membrane leakage on the SNN precision when processing LIDAR data for obstacle avoidance. It is shown that by carefully tuning the membrane potential leakage constant of the spiking Leaky Integrate-and-Fire (LIF) neurons used within our SNN, it is possible to achieve on-par robot control precision compared to the use of a non-spiking Convolutional Neural Network (CNN). Finally, the LIDAR dataset collected during this work is released as open-source with the hope of benefiting future research. 

**Abstract (ZH)**: 使用神经形态计算进行基于神经脉冲网络的机器人应用取得了广泛关注，得益于其在神经形态硬件实现时对高精度但低存储和计算复杂度推断的能力。这种能力使神经脉冲网络（SNNs）非常适合电池资源和载荷通常受限的自主机器人应用（如无人机和火星车）。在此背景下，本文研究了使用SNNs直接从LIDAR数据执行机器人导航和避障的方法。搭建了一个配备了LIDAR的自定义机器人平台，采集了带标注的LIDAR感知数据以及用于避障的人工操作机器人控制命令。本文还首次集中在我们所知的范围内，探讨了神经元膜泄漏在处理LIDAR数据进行避障时对SNN精度的重要性。结果显示，通过仔细调整用于我们SNN中的脉冲泄漏整化放电（LIF）神经元的膜电位泄漏常数，可以实现与非脉冲卷积神经网络（CNN）相当的机器人控制精度。最后，本工作中收集的LIDAR数据集已开源，旨在为未来的研究提供支持。 

---
# VDInstruct: Zero-Shot Key Information Extraction via Content-Aware Vision Tokenization 

**Title (ZH)**: VDInstruct: 基于内容aware视觉词元化的内容零样本关键信息提取 

**Authors**: Son Nguyen, Giang Nguyen, Hung Dao, Thao Do, Daeyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.09531)  

**Abstract**: Key Information Extraction (KIE) underpins the understanding of visual documents (e.g., receipts and contracts) by extracting precise semantic content and accurately capturing spatial structure. Yet existing multimodal large language models (MLLMs) often perform poorly on dense documents and rely on vision tokenization approaches that scale with image size, leading to redundant computation and memory inefficiency. To address these challenges, we introduce VDInstruct, an MLLM that separates spatial region detection from semantic feature extraction. Central to our model is a content-aware tokenization strategy: rather than fragmenting the entire image uniformly, it generates tokens in proportion to document complexity, preserving critical structure while eliminating wasted tokens. Leveraging a three-stage training paradigm, our model achieves state-of-the-art (SOTA) results on KIE benchmarks, matching or exceeding the accuracy of leading approaches while reducing the number of image tokens by roughly 3.6x. In zero-shot evaluations, VDInstruct surpasses strong baselines-such as DocOwl 1.5-by +5.5 F1 points, highlighting its robustness to unseen documents. These findings show that content-aware tokenization combined with explicit layout modeling offers a promising direction forward for document understanding. Data, source code, and model weights will be made publicly available. 

**Abstract (ZH)**: 基于内容感知的区域检测和语义特征提取分离的视觉文档关键信息提取 

---
# An Analysis of Action-Value Temporal-Difference Methods That Learn State Values 

**Title (ZH)**: 动作值时序差分方法的研究：学习状态值分析 

**Authors**: Brett Daley, Prabhat Nagarajan, Martha White, Marlos C. Machado  

**Link**: [PDF](https://arxiv.org/pdf/2507.09523)  

**Abstract**: The hallmark feature of temporal-difference (TD) learning is bootstrapping: using value predictions to generate new value predictions. The vast majority of TD methods for control learn a policy by bootstrapping from a single action-value function (e.g., Q-learning and Sarsa). Significantly less attention has been given to methods that bootstrap from two asymmetric value functions: i.e., methods that learn state values as an intermediate step in learning action values. Existing algorithms in this vein can be categorized as either QV-learning or AV-learning. Though these algorithms have been investigated to some degree in prior work, it remains unclear if and when it is advantageous to learn two value functions instead of just one -- and whether such approaches are theoretically sound in general. In this paper, we analyze these algorithmic families in terms of convergence and sample efficiency. We find that while both families are more efficient than Expected Sarsa in the prediction setting, only AV-learning methods offer any major benefit over Q-learning in the control setting. Finally, we introduce a new AV-learning algorithm called Regularized Dueling Q-learning (RDQ), which significantly outperforms Dueling DQN in the MinAtar benchmark. 

**Abstract (ZH)**: TD学习的标志性特征是bootstrapping：使用价值预测来生成新的价值预测。大多数用于控制的TD方法通过从单一的动作价值函数（如Q-learning和Sarsa）bootstrap学习策略。相比之下，较少有研究关注从两个不对称价值函数bootstrap的方法：即在学习动作价值之前学习状态价值的方法。现有的这类算法可以归类为QV-learning或AV-learning。尽管这些算法在先前的研究中有所探讨，但仍不清楚何时以及在什么情况下学习两个价值函数而非一个更有优势——并且这种途径在一般情况下是否具有理论上的合理性。在本文中，我们从收敛性和样本效率的角度分析了这些算法家族。我们发现，在预测任务中，两种家族算法都比Expected Sarsa更高效，但在控制任务中，只有AV-learning方法在一定程度上优于Q-learning。最后，我们提出了一种新的AV-learning算法——正则化对 Dueling Q-learning（RDQ），该算法在MinAtar基准测试中显著优于Dueling DQN。 

---
# QuarterMap: Efficient Post-Training Token Pruning for Visual State Space Models 

**Title (ZH)**: QuarterMap: Visual状态空间模型高效后训练 tokens 裁剪方法 

**Authors**: Tien-Yu Chi, Hung-Yueh Chiang, Diana Marculescu, Kai-Chiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09514)  

**Abstract**: State space models (SSMs) reduce the quadratic complexity of transformers by leveraging linear recurrence. Recently, VMamba has emerged as a strong SSM-based vision backbone, yet remains bottlenecked by spatial redundancy in its four-directional scan. We propose QuarterMap, a post-training activation pruning method that removes redundant spatial activations before scanning and restores dimensions via nearest-neighbor upsampling. Our method improves throughput without retraining. On ImageNet-1K, QuarterMap achieves up to 11% speedup on VMamba with less than 0.9% accuracy drop, and yields similar gains on ADE20K segmentation. Beyond VMamba, we validate QuarterMap on MedMamba, a domain-specific model that shares the same four-directional scanning structure, where it consistently improves throughput while preserving accuracy across multiple medical imaging tasks. Compared to token merging methods like ToMe, QuarterMap is tailored for SSMs and avoids costly merge-unmerge operations. Our method offers a plug-and-play tool for deployment-time efficiency without compromising transferability. 

**Abstract (ZH)**: 基于状态空间模型的QuarterMap：一种无重构训练的激活剪枝方法 

---
# A Mixture of Linear Corrections Generates Secure Code 

**Title (ZH)**: 线性修正混合生成安全代码 

**Authors**: Weichen Yu, Ravi Mangal, Terry Zhuo, Matt Fredrikson, Corina S. Pasareanu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09508)  

**Abstract**: Large language models (LLMs) have become proficient at sophisticated code-generation tasks, yet remain ineffective at reliably detecting or avoiding code vulnerabilities. Does this deficiency stem from insufficient learning about code vulnerabilities, or is it merely a result of ineffective prompting? Using representation engineering techniques, we investigate whether LLMs internally encode the concepts necessary to identify code vulnerabilities. We find that current LLMs encode precise internal representations that distinguish vulnerable from secure code--achieving greater accuracy than standard prompting approaches. Leveraging these vulnerability-sensitive representations, we develop an inference-time steering technique that subtly modulates the model's token-generation probabilities through a mixture of corrections (MoC). Our method effectively guides LLMs to produce less vulnerable code without compromising functionality, demonstrating a practical approach to controlled vulnerability management in generated code. Notably, MoC enhances the security ratio of Qwen2.5-Coder-7B by 8.9\%, while simultaneously improving functionality on HumanEval pass@1 by 2.1\%. 

**Abstract (ZH)**: 大型语言模型在生成代码方面已经变得非常 proficient，但在可靠地检测或避免代码漏洞方面仍然效能不足。这种不足是由于对代码漏洞学习不足，还是仅仅因为提示无效？使用表示工程技术，我们研究当前的大型语言模型是否内部编码了识别代码漏洞所需要的概念。我们发现，当前的大型语言模型内部编码了精确的表示，能够区分漏洞代码与安全代码，并且其准确度超过了标准提示方法。利用这些漏洞敏感的表示，我们开发了一种推理时导向技术，通过混合修正（MoC）微妙地调节模型的标记生成概率。该方法有效地引导大型语言模型生成更安全的代码而不过度影响功能，展示了在生成代码中实现受控漏洞管理的实际方法。值得注意的是，MoC 提升了 Qwen2.5-Coder-7B 的安全性比例达 8.9%，同时在 HumanEval pass@1 上提高了功能 2.1%。 

---
# SDTN and TRN: Adaptive Spectral-Spatial Feature Extraction for Hyperspectral Image Classification 

**Title (ZH)**: SDTN和TRN：适应性谱-空特征提取方法在高光谱图像分类中的应用 

**Authors**: Fuyin Ye, Erwen Yao, Jianyong Chen, Fengmei He, Junxiang Zhang, Lihao Ni  

**Link**: [PDF](https://arxiv.org/pdf/2507.09492)  

**Abstract**: Hyperspectral image classification plays a pivotal role in precision agriculture, providing accurate insights into crop health monitoring, disease detection, and soil analysis. However, traditional methods struggle with high-dimensional data, spectral-spatial redundancy, and the scarcity of labeled samples, often leading to suboptimal performance. To address these challenges, we propose the Self-Adaptive Tensor- Regularized Network (SDTN), which combines tensor decomposition with regularization mechanisms to dynamically adjust tensor ranks, ensuring optimal feature representation tailored to the complexity of the data. Building upon SDTN, we propose the Tensor-Regularized Network (TRN), which integrates the features extracted by SDTN into a lightweight network capable of capturing spectral-spatial features at multiple scales. This approach not only maintains high classification accuracy but also significantly reduces computational complexity, making the framework highly suitable for real-time deployment in resource-constrained environments. Experiments on PaviaU datasets demonstrate significant improvements in accuracy and reduced model parameters compared to state-of-the-art methods. 

**Abstract (ZH)**: 高维光谱图像分类在精准农业中发挥着关键作用，为作物健康监测、病害检测和土壤分析提供准确洞察。然而，传统方法难以处理高维数据、光谱-空间冗余以及标签样本稀缺的问题，常常导致性能不佳。为应对这些挑战，我们提出了一种自适应张量正则化网络（SDTN），它结合了张量分解和正则化机制，动态调整张量秩，确保针对数据复杂性的最优特征表示。在此基础上，我们提出了张量正则化网络（TRN），它将SDTN提取的特征整合到一个轻量级网络中，能够多尺度捕获光谱-空间特征。该方法不仅保持了高分类精度，还显著降低了计算复杂度，使框架在资源受限环境中具有高度实时部署的适用性。实验结果表明，TRN在PaviaU数据集上的准确性和模型参数方面显著优于现有方法。 

---
# HMID-Net: An Exploration of Masked Image Modeling and Knowledge Distillation in Hyperbolic Space 

**Title (ZH)**: HMID-Net：超球面上掩码图像建模与知识蒸馏的探索 

**Authors**: Changli Wang, Fang Yin, Jiafeng Liu, Rui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09487)  

**Abstract**: Visual and semantic concepts are often structured in a hierarchical manner. For instance, textual concept `cat' entails all images of cats. A recent study, MERU, successfully adapts multimodal learning techniques from Euclidean space to hyperbolic space, effectively capturing the visual-semantic hierarchy. However, a critical question remains: how can we more efficiently train a model to capture and leverage this hierarchy? In this paper, we propose the \textit{Hyperbolic Masked Image and Distillation Network} (HMID-Net), a novel and efficient method that integrates Masked Image Modeling (MIM) and knowledge distillation techniques within hyperbolic space. To the best of our knowledge, this is the first approach to leverage MIM and knowledge distillation in hyperbolic space to train highly efficient models. In addition, we introduce a distillation loss function specifically designed to facilitate effective knowledge transfer in hyperbolic space. Our experiments demonstrate that MIM and knowledge distillation techniques in hyperbolic space can achieve the same remarkable success as in Euclidean space. Extensive evaluations show that our method excels across a wide range of downstream tasks, significantly outperforming existing models like MERU and CLIP in both image classification and retrieval. 

**Abstract (ZH)**: 基于双曲空间的掩码图像和蒸馏网络（HMID-Net）：高效捕获和利用视觉-语义层次结构的方法 

---
# ViSP: A PPO-Driven Framework for Sarcasm Generation with Contrastive Learning 

**Title (ZH)**: ViSP：一种基于PPO的对比学习驱动的 sarcasm生成框架 

**Authors**: Changli Wang, Rui Wu, Fang Yin  

**Link**: [PDF](https://arxiv.org/pdf/2507.09482)  

**Abstract**: Human emotions are complex, with sarcasm being a subtle and distinctive form. Despite progress in sarcasm research, sarcasm generation remains underexplored, primarily due to the overreliance on textual modalities and the neglect of visual cues, as well as the mismatch between image content and sarcastic intent in existing datasets. In this paper, we introduce M2SaG, a multimodal sarcasm generation dataset with 4,970 samples, each containing an image, a sarcastic text, and a sarcasm target. To benchmark M2SaG, we propose ViSP, a generation framework that integrates Proximal Policy Optimization (PPO) and contrastive learning. PPO utilizes reward scores from DIP to steer the generation of sarcastic texts, while contrastive learning encourages the model to favor outputs with higher reward scores. These strategies improve overall generation quality and produce texts with more pronounced sarcastic intent. We evaluate ViSP across five metric sets and find it surpasses all baselines, including large language models, underscoring their limitations in sarcasm generation. Furthermore, we analyze the distributions of Sarcasm Scores and Factual Incongruity for both M2SaG and the texts generated by ViSP. The generated texts exhibit higher mean Sarcasm Scores (0.898 vs. 0.770) and Factual Incongruity (0.768 vs. 0.739), demonstrating that ViSP produces higher-quality sarcastic content than the original dataset. % The dataset and code will be publicly available. Our dataset and code will be released at \textit{this https URL}. 

**Abstract (ZH)**: 多模态讽刺生成数据集M2SaG及基准模型ViSP 

---
# Evaluating LLMs on Sequential API Call Through Automated Test Generation 

**Title (ZH)**: 评估大规模语言模型在顺序API调用上的性能通过自动化测试生成 

**Authors**: Yuheng Huang, Da Song, Zhenlan Ji, Shuai Wang, Lei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.09481)  

**Abstract**: By integrating tools from external APIs, Large Language Models (LLMs) have expanded their promising capabilities in a diverse spectrum of complex real-world tasks. However, testing, evaluation, and analysis of LLM tool use remain in their early stages. Most existing benchmarks rely on manually collected test cases, many of which cannot be automatically checked for semantic correctness and instead depend on static methods such as string matching. Additionally, these benchmarks often overlook the complex interactions that occur between sequential API calls, which are common in real-world applications. To fill the gap, in this paper, we introduce StateGen, an automated framework designed to generate diverse coding tasks involving sequential API interactions. StateGen combines state-machine-based API constraint solving and validation, energy-based sampling, and control-flow injection to generate executable programs. These programs are then translated into human-like natural language task descriptions through a collaboration of two LLM agents. Utilizing StateGen, we construct StateEval, a benchmark encompassing 120 verified test cases spanning across three representative scenarios: Session Service, Tensor Operation, and ElevenLabs MCP. Experimental results confirm that StateGen can effectively generate challenging and realistic API-oriented tasks, highlighting areas for improvement in current LLMs incorporating APIs. 

**Abstract (ZH)**: 通过集成外部API工具，大型语言模型（LLMs）在其多样化的复杂现实任务中扩展了其有前途的能力。然而，LLM工具使用测试、评估和分析仍处于初期阶段。现有的大多数基准依赖于手动收集的测试案例，其中许多案例无法自动检查语义正确性，而是依赖于字符串匹配等静态方法。此外，这些基准通常忽略了序列API调用之间复杂的交互，这在现实世界应用程序中很常见。为填补这一空白，本文介绍了一种自动框架StateGen，用于生成涉及序列API交互的多样化编码任务。StateGen结合了基于状态机的API约束求解和验证、基于能量的采样以及控制流注入，以生成可执行程序。这些程序随后通过两个LLM代理的合作被翻译成类似人类自然语言的任务描述。利用StateGen，我们构建了StateEval基准，包含120个经过验证的测试案例，涵盖了三个代表性场景：会话服务、张量操作和ElevenLabs MCP。实验结果表明，StateGen能够有效生成具有挑战性和现实性的API导向任务，突显了当前集成API的LLMs需要改进的领域。 

---
# Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs 

**Title (ZH)**: 面向自主型RAG的深度推理研究：LLMs中RAG-推理系统综述 

**Authors**: Yangning Li, Weizhi Zhang, Yuyao Yang, Wei-Chieh Huang, Yaozu Wu, Junyu Luo, Yuanchen Bei, Henry Peng Zou, Xiao Luo, Yusheng Zhao, Chunkit Chan, Yankai Chen, Zhongfen Deng, Yinghui Li, Hai-Tao Zheng, Dongyuan Li, Renhe Jiang, Ming Zhang, Yangqiu Song, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09477)  

**Abstract**: Retrieval-Augmented Generation (RAG) lifts the factuality of Large Language Models (LLMs) by injecting external knowledge, yet it falls short on problems that demand multi-step inference; conversely, purely reasoning-oriented approaches often hallucinate or mis-ground facts. This survey synthesizes both strands under a unified reasoning-retrieval perspective. We first map how advanced reasoning optimizes each stage of RAG (Reasoning-Enhanced RAG). Then, we show how retrieved knowledge of different type supply missing premises and expand context for complex inference (RAG-Enhanced Reasoning). Finally, we spotlight emerging Synergized RAG-Reasoning frameworks, where (agentic) LLMs iteratively interleave search and reasoning to achieve state-of-the-art performance across knowledge-intensive benchmarks. We categorize methods, datasets, and open challenges, and outline research avenues toward deeper RAG-Reasoning systems that are more effective, multimodally-adaptive, trustworthy, and human-centric. The collection is available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）通过注入外部知识提高了大型语言模型（LLMs）的事实性，但对需要多步推理的问题处理不足；相反，纯粹基于推理的方法往往会出现幻觉或错误地使用事实。本文综述将这两者统一到推理-检索视角下。我们首先分析高级推理如何优化RAG的每个阶段（增强推理的RAG）。然后展示了不同类型检索知识如何提供缺失的前提并扩展复杂推理所需的上下文（增强推理的RAG）。最后，我们强调了新兴的协同RAG-推理框架，其中（自主的）LLM迭代地交织搜索与推理，以在知识密集型基准测试中实现最先进的性能。我们对方法、数据集和开放挑战进行了分类，并概述了通向更有效、多模态适应、可靠和以人为本的RAG-推理系统的研究方向。 

---
# Enhancing Clinical Text Classification via Fine-Tuned DRAGON Longformer Models 

**Title (ZH)**: 通过细调DRAGON Longformer模型提升临床文本分类 

**Authors**: Mingchuan Yang, Ziyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09470)  

**Abstract**: This study explores the optimization of the DRAGON Longformer base model for clinical text classification, specifically targeting the binary classification of medical case descriptions. A dataset of 500 clinical cases containing structured medical observations was used, with 400 cases for training and 100 for validation. Enhancements to the pre-trained joeranbosma/dragon-longformer-base-mixed-domain model included hyperparameter tuning, domain-specific preprocessing, and architectural adjustments. Key modifications involved increasing sequence length from 512 to 1024 tokens, adjusting learning rates from 1e-05 to 5e-06, extending training epochs from 5 to 8, and incorporating specialized medical terminology. The optimized model achieved notable performance gains: accuracy improved from 72.0% to 85.2%, precision from 68.0% to 84.1%, recall from 75.0% to 86.3%, and F1-score from 71.0% to 85.2%. Statistical analysis confirmed the significance of these improvements (p < .001). The model demonstrated enhanced capability in interpreting medical terminology, anatomical measurements, and clinical observations. These findings contribute to domain-specific language model research and offer practical implications for clinical natural language processing applications. The optimized model's strong performance across diverse medical conditions underscores its potential for broad use in healthcare settings. 

**Abstract (ZH)**: 本研究探讨了DRAGON Longformer基模型在临床文本分类中的优化，特别针对医疗病例描述的二分类任务。使用了一个包含500个临床病例的结构化医疗观察数据集，其中400个用于训练，100个用于验证。对预训练的joeranbosma/dragon-longformer-base-mixed-domain模型的增强包括超参数调整、领域特定预处理和架构调整。关键修改包括将序列长度从512增加到1024个标记，将学习率从1e-05调整到5e-06，将训练周期从5调整到8，并引入了专门的医疗术语。优化后的模型取得了显著的性能提升：准确率从72.0%提高到85.2%，精确率从68.0%提高到84.1%，召回率从75.0%提高到86.3%，F1分数从71.0%提高到85.2%。统计分析证实了这些改进的显著性（p < .001）。该模型展示了在解释医疗术语、解剖测量和临床观察方面的增强能力。本研究结果为领域特定语言模型研究做出了贡献，并为临床自然语言处理应用提供了实际意义。优化后的模型在多种医学条件下表现出色，表明其在医疗保健领域的广泛应用潜力。 

---
# Enhancing ALS Progression Tracking with Semi-Supervised ALSFRS-R Scores Estimated from Ambient Home Health Monitoring 

**Title (ZH)**: 基于环境家庭健康监测的半监督ALSFRS-R评分增强ALS进展跟踪 

**Authors**: Noah Marchal, William E. Janes, Mihail Popescu, Xing Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.09460)  

**Abstract**: Clinical monitoring of functional decline in ALS relies on periodic assessments that may miss critical changes occurring between visits. To address this gap, semi-supervised regression models were developed to estimate rates of decline in a case series cohort by targeting ALSFRS- R scale trajectories with continuous in-home sensor monitoring data. Our analysis compared three model paradigms (individual batch learning and cohort-level batch versus incremental fine-tuned transfer learning) across linear slope, cubic polynomial, and ensembled self-attention pseudo-label interpolations. Results revealed cohort homogeneity across functional domains responding to learning methods, with transfer learning improving prediction error for ALSFRS-R subscales in 28 of 32 contrasts (mean RMSE=0.20(0.04)), and individual batch learning for predicting the composite scale (mean RMSE=3.15(1.25)) in 2 of 3. Self-attention interpolation achieved the lowest prediction error for subscale-level models (mean RMSE=0.19(0.06)), capturing complex nonlinear progression patterns, outperforming linear and cubic interpolations in 20 of 32 contrasts, though linear interpolation proved more stable in all ALSFRS-R composite scale models (mean RMSE=0.23(0.10)). We identified distinct homogeneity-heterogeneity profiles across functional domains with respiratory and speech exhibiting patient-specific patterns benefiting from personalized incremental adaptation, while swallowing and dressing functions followed cohort-level trajectories suitable for transfer models. These findings suggest that matching learning and pseudo-labeling techniques to functional domain-specific homogeneity-heterogeneity profiles enhances predictive accuracy in ALS progression tracking. Integrating adaptive model selection within sensor monitoring platforms could enable timely interventions and scalable deployment in future multi-center studies. 

**Abstract (ZH)**: 基于半监督回归模型的ALS功能衰退临床监测：整合连续居家传感器数据以填补评估间隔期间的变化监测不足 

---
# Fourier Basis Mapping: A Time-Frequency Learning Framework for Time Series Forecasting 

**Title (ZH)**: Fourier 基映射：时间序列预测的时间-频率学习框架 

**Authors**: Runze Yang, Longbing Cao, Xin You, Kun Fang, Jianxun Li, Jie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.09445)  

**Abstract**: The integration of Fourier transform and deep learning opens new avenues for time series forecasting. We reconsider the Fourier transform from a basis functions perspective. Specifically, the real and imaginary parts of the frequency components can be regarded as the coefficients of cosine and sine basis functions at tiered frequency levels, respectively. We find that existing Fourier-based methods face inconsistent starting cycles and inconsistent series length issues. They fail to interpret frequency components precisely and overlook temporal information. Accordingly, the novel Fourier Basis Mapping (FBM) method addresses these issues by integrating time-frequency features through Fourier basis expansion and mapping in the time-frequency space. Our approach extracts explicit frequency features while preserving temporal characteristics. FBM supports plug-and-play integration with various types of neural networks by only adjusting the first initial projection layer for better performance. First, we propose FBM-L, FBM-NL, and FBM-NP to enhance linear, MLP-based, and Transformer-based models, respectively, demonstrating the effectiveness of time-frequency features. Next, we propose a synergetic model architecture, termed FBM-S, which decomposes the seasonal, trend, and interaction effects into three separate blocks, each designed to model time-frequency features in a specialized manner. Finally, we introduce several techniques tailored for time-frequency features, including interaction masking, centralization, patching, rolling window projection, and multi-scale down-sampling. The results are validated on diverse real-world datasets for both long-term and short-term forecasting tasks with SOTA performance. 

**Abstract (ZH)**: Fourier变换与深度学习的集成为时间序列预测开辟了新途径：基于基函数的新颖Fourier基映射方法 

---
# Transformers Don't In-Context Learn Least Squares Regression 

**Title (ZH)**: Transformer 不进行上下文学习以求解最小二乘回归 

**Authors**: Joshua Hill, Benjamin Eyre, Elliot Creager  

**Link**: [PDF](https://arxiv.org/pdf/2507.09440)  

**Abstract**: In-context learning (ICL) has emerged as a powerful capability of large pretrained transformers, enabling them to solve new tasks implicit in example input-output pairs without any gradient updates. Despite its practical success, the mechanisms underlying ICL remain largely mysterious. In this work we study synthetic linear regression to probe how transformers implement learning at inference time. Previous works have demonstrated that transformers match the performance of learning rules such as Ordinary Least Squares (OLS) regression or gradient descent and have suggested ICL is facilitated in transformers through the learned implementation of one of these techniques. In this work, we demonstrate through a suite of out-of-distribution generalization experiments that transformers trained for ICL fail to generalize after shifts in the prompt distribution, a behaviour that is inconsistent with the notion of transformers implementing algorithms such as OLS. Finally, we highlight the role of the pretraining corpus in shaping ICL behaviour through a spectral analysis of the learned representations in the residual stream. Inputs from the same distribution as the training data produce representations with a unique spectral signature: inputs from this distribution tend to have the same top two singular vectors. This spectral signature is not shared by out-of-distribution inputs, and a metric characterizing the presence of this signature is highly correlated with low loss. 

**Abstract (ZH)**: 上下文学习（ICL）已成为大规模预训练转换器的一个强大能力，使它们能够在无需任何梯度更新的情况下解决由示例输入-输出对隐含的新任务。尽管其在实践中的成功令人印象深刻，但ICL背后的机制仍 largely神秘。在本文中，我们研究合成线性回归，以探查转换器在推理时如何实现学习。之前的研究表明，转换器能够与如普通最小二乘法（OLS）回归或梯度下降等学习规则匹配，且暗示ICL通过学习实现这些技术之一的方式来促进。在本文中，我们通过一系列出分布泛化实验来证明，用于ICL训练的转换器在提示分布变化后无法泛化，这一行为与转换器实施OLS等算法的观点不一致。最后，我们通过残差流中学习表示的谱分析突显了预训练语料库在塑造ICL行为中的作用。来自与训练数据相同分布的输入产生具有独特谱签名的表示：来自该分布的输入往往具有相同的前两个奇异向量。这种谱签名未由出分布输入共享，且衡量此签名存在性的度量与低损失高度相关。 

---
# Dynamic Sparse Causal-Attention Temporal Networks for Interpretable Causality Discovery in Multivariate Time Series 

**Title (ZH)**: 多变量时间序列中可解释因果关系发现的动态稀疏因果注意时序网络 

**Authors**: Meriem Zerkouk, Miloud Mihoubi, Belkacem Chikhaoui  

**Link**: [PDF](https://arxiv.org/pdf/2507.09439)  

**Abstract**: Understanding causal relationships in multivariate time series (MTS) is essential for effective decision-making in fields such as finance and marketing, where complex dependencies and lagged effects challenge conventional analytical approaches. We introduce Dynamic Sparse Causal-Attention Temporal Networks for Interpretable Causality Discovery in MTS (DyCAST-Net), a novel architecture designed to enhance causal discovery by integrating dilated temporal convolutions and dynamic sparse attention mechanisms. DyCAST-Net effectively captures multiscale temporal dependencies through dilated convolutions while leveraging an adaptive thresholding strategy in its attention mechanism to eliminate spurious connections, ensuring both accuracy and interpretability. A statistical shuffle test validation further strengthens robustness by filtering false positives and improving causal inference reliability. Extensive evaluations on financial and marketing datasets demonstrate that DyCAST-Net consistently outperforms existing models such as TCDF, GCFormer, and CausalFormer. The model provides a more precise estimation of causal delays and significantly reduces false discoveries, particularly in noisy environments. Moreover, attention heatmaps offer interpretable insights, uncovering hidden causal patterns such as the mediated effects of advertising on consumer behavior and the influence of macroeconomic indicators on financial markets. Case studies illustrate DyCAST-Net's ability to detect latent mediators and lagged causal factors, making it particularly effective in high-dimensional, dynamic settings. The model's architecture enhanced by RMSNorm stabilization and causal masking ensures scalability and adaptability across diverse application domains 

**Abstract (ZH)**: 理解和发现多变量时间序列（MTS）中的因果关系对于金融和市场营销等领域中的有效决策至关重要，因为复杂的依赖关系和滞后效应挑战了传统的分析方法。我们提出了Dynamic Sparse Causal-Attention Temporal Networks for Interpretable Causality Discovery in MTS（DyCAST-Net），这是一种新颖的架构，旨在通过集成扩张时序卷积和动态稀疏注意机制来提高因果关系的发现能力。DyCAST-Net通过扩张卷积有效捕捉多尺度时间依赖关系，并通过注意机制中的自适应阈值策略消除虚假连接，确保了准确性和可解释性。通过统计混洗测试进一步增强了鲁棒性，通过过滤掉假阳性结果来提高因果推理的可靠性。在金融和市场营销数据集上的广泛评估表明，DyCAST-Net在与TCDF、GCFormer和CausalFormer等现有模型的对比中表现更优。该模型提供了更精确的因果延迟估计，特别是在噪音环境中显著减少了假发现。此外，注意力热图提供了可解释的洞察，揭示了隐藏的因果模式，如广告对消费者行为的中介效应以及宏观经济指标对金融市场的影响。案例研究展示了DyCAST-Net在检测潜在中介和滞后因果因素方面的有效性，特别是在高维动态环境中更为有效。通过RMSNorm稳定化和因果掩码增强的模型架构确保了在各种应用领域的可扩展性和适应性。 

---
# Domain Adaptation and Multi-view Attention for Learnable Landmark Tracking with Sparse Data 

**Title (ZH)**: 基于稀疏数据的领域适应与多视图注意力可学习地标跟踪 

**Authors**: Timothy Chase Jr, Karthik Dantu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09420)  

**Abstract**: The detection and tracking of celestial surface terrain features are crucial for autonomous spaceflight applications, including Terrain Relative Navigation (TRN), Entry, Descent, and Landing (EDL), hazard analysis, and scientific data collection. Traditional photoclinometry-based pipelines often rely on extensive a priori imaging and offline processing, constrained by the computational limitations of radiation-hardened systems. While historically effective, these approaches typically increase mission costs and duration, operate at low processing rates, and have limited generalization. Recently, learning-based computer vision has gained popularity to enhance spacecraft autonomy and overcome these limitations. While promising, emerging techniques frequently impose computational demands exceeding the capabilities of typical spacecraft hardware for real-time operation and are further challenged by the scarcity of labeled training data for diverse extraterrestrial environments. In this work, we present novel formulations for in-situ landmark tracking via detection and description. We utilize lightweight, computationally efficient neural network architectures designed for real-time execution on current-generation spacecraft flight processors. For landmark detection, we propose improved domain adaptation methods that enable the identification of celestial terrain features with distinct, cheaply acquired training data. Concurrently, for landmark description, we introduce a novel attention alignment formulation that learns robust feature representations that maintain correspondence despite significant landmark viewpoint variations. Together, these contributions form a unified system for landmark tracking that demonstrates superior performance compared to existing state-of-the-art techniques. 

**Abstract (ZH)**: 天体表面地形特征的检测与跟踪对于自主太空飞行应用，包括地形相对导航（TRN）、进入、下降和着陆（EDL）、危险分析和科学数据采集至关重要。传统基于光度几何的流水线往往依赖于大量的先验成像和离线处理，受限于辐射加固系统的计算限制。尽管历史上效果显著，这些方法通常会增加任务成本和时间，处理速率较低，并且泛化能力有限。近年来，基于学习的计算机视觉技术得到了广泛应用，以增强航天器的自主性和克服这些限制。虽然前景广阔，但新兴技术往往对典型航天器硬件的实时操作提出了超出其计算能力的要求，并且由于缺乏用于多样外太空环境的标记训练数据而面临挑战。在本文中，我们提出了新颖的原位地标跟踪形式化方法，通过检测和描述实现。我们利用针对当前代际航天器飞行处理器实时执行设计的轻量级、计算高效的神经网络架构。在地标检测方面，我们提出了改进的域适应方法，能够在经济获取的训练数据下识别具有独特特征的天体地理特征。同时，在地标描述方面，我们引入了一种新颖的注意力对齐形式化方法，学习具有鲁棒性且在地标视角变化显著的情况下仍能保持对应关系的特征表示。这些贡献共同形成了一个统一的地标跟踪系统，其性能优于现有的最先进的技术。 

---
# Adversarial Activation Patching: A Framework for Detecting and Mitigating Emergent Deception in Safety-Aligned Transformers 

**Title (ZH)**: 对抗激活补丁：一种检测和缓解对齐安全变压器Emergent欺骗的框架 

**Authors**: Santhosh Kumar Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2507.09406)  

**Abstract**: Large language models (LLMs) aligned for safety through techniques like reinforcement learning from human feedback (RLHF) often exhibit emergent deceptive behaviors, where outputs appear compliant but subtly mislead or omit critical information. This paper introduces adversarial activation patching, a novel mechanistic interpretability framework that leverages activation patching as an adversarial tool to induce, detect, and mitigate such deception in transformer-based models. By sourcing activations from "deceptive" prompts and patching them into safe forward passes at specific layers, we simulate vulnerabilities and quantify deception rates. Through toy neural network simulations across multiple scenarios (e.g., 1000 trials per setup), we demonstrate that adversarial patching increases deceptive outputs to 23.9% from a 0% baseline, with layer-specific variations supporting our hypotheses. We propose six hypotheses, including transferability across models, exacerbation in multimodal settings, and scaling effects. An expanded literature review synthesizes over 20 key works in interpretability, deception, and adversarial attacks. Mitigation strategies, such as activation anomaly detection and robust fine-tuning, are detailed, alongside ethical considerations and future research directions. This work advances AI safety by highlighting patching's dual-use potential and provides a roadmap for empirical studies on large-scale models. 

**Abstract (ZH)**: 大型语言模型通过强化学习从人类反馈（RLHF）等技术对齐以确保安全时，往往会表现出 Emergent Deceptive 行为，即输出看似合规但实际上却含糊其辞或遗漏关键信息。本文介绍了一种名为对抗激活补丁的新机制可解释性框架，该框架利用激活补丁作为对抗工具以诱导、检测和缓解基于变换器的模型中的欺骗行为。通过从“欺骗性”提示中获取激活并将其插入到特定层的安全正向传递中，我们模拟漏洞并量化欺骗率。通过在多个场景中的玩具神经网络模拟（例如，每种设置下1000次试验），我们证明了对抗补丁使欺骗性输出从基线0%增加到23.9%，且不同层的差异支持了我们的假设。我们提出了六个假设，包括模型间可转移性、多模态环境中的加剧效应以及缩放效应。扩展性的文献综述综合了20余项关于可解释性、欺骗和对抗攻击的关键研究成果。详细描述了缓解策略，包括激活异常检测和鲁棒微调，并讨论了伦理考量和未来研究方向。这项工作通过突出补丁的双重用途推动了AI安全，并为大规模模型的实证研究提供了路线图。 

---
# Fair CCA for Fair Representation Learning: An ADNI Study 

**Title (ZH)**: 公平CCA在公平表示学习中的研究：基于ADNI的数据分析 

**Authors**: Bojian Hou, Zhanliang Wang, Zhuoping Zhou, Boning Tong, Zexuan Wang, Jingxuan Bao, Duy Duong-Tran, Qi Long, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09382)  

**Abstract**: Canonical correlation analysis (CCA) is a technique for finding correlations between different data modalities and learning low-dimensional representations. As fairness becomes crucial in machine learning, fair CCA has gained attention. However, previous approaches often overlook the impact on downstream classification tasks, limiting applicability. We propose a novel fair CCA method for fair representation learning, ensuring the projected features are independent of sensitive attributes, thus enhancing fairness without compromising accuracy. We validate our method on synthetic data and real-world data from the Alzheimer's Disease Neuroimaging Initiative (ADNI), demonstrating its ability to maintain high correlation analysis performance while improving fairness in classification tasks. Our work enables fair machine learning in neuroimaging studies where unbiased analysis is essential. 

**Abstract (ZH)**: 公平的主成分分析（CCA）方法：确保投影特征独立于敏感属性，从而在不牺牲准确性的前提下提高公平性 

---
# Context-Aware Regularization with Markovian Integration for Attention-Based Nucleotide Analysis 

**Title (ZH)**: 基于马尔可夫集成的上下文感知正则化核酸注意力分析 

**Authors**: Mohammadsaleh Refahi, Mahdi Abavisani, Bahrad A. Sokhansanj, James R. Brown, Gail Rosen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09378)  

**Abstract**: Transformers have revolutionized nucleotide sequence analysis, yet capturing long-range dependencies remains challenging. Recent studies show that autoregressive transformers often exhibit Markovian behavior by relying on fixed-length context windows for next-token prediction. However, standard self-attention mechanisms are computationally inefficient for long sequences due to their quadratic complexity and do not explicitly enforce global transition consistency.
We introduce CARMANIA (Context-Aware Regularization with Markovian Integration for Attention-Based Nucleotide Analysis), a self-supervised pretraining framework that augments next-token (NT) prediction with a transition-matrix (TM) loss. The TM loss aligns predicted token transitions with empirically derived n-gram statistics from each input sequence, encouraging the model to capture higher-order dependencies beyond local context. This integration enables CARMANIA to learn organism-specific sequence structures that reflect both evolutionary constraints and functional organization.
We evaluate CARMANIA across diverse genomic tasks, including regulatory element prediction, functional gene classification, taxonomic inference, antimicrobial resistance detection, and biosynthetic gene cluster classification. CARMANIA outperforms the previous best long-context model by at least 7 percent, matches state-of-the-art on shorter sequences (exceeding prior results on 20 out of 40 tasks while running approximately 2.5 times faster), and shows particularly strong improvements on enhancer and housekeeping gene classification tasks, including up to a 34 percent absolute gain in Matthews correlation coefficient (MCC) for enhancer prediction. The TM loss boosts accuracy in 33 of 40 tasks, especially where local motifs or regulatory patterns drive prediction. 

**Abstract (ZH)**: Context-Aware Regularization with Markovian Integration for Attention-Based Nucleotide Analysis 

---
# Impute With Confidence: A Framework for Uncertainty Aware Multivariate Time Series Imputation 

**Title (ZH)**: 自信插值：一种考虑不确定性的多变量时间序列插值框架 

**Authors**: Addison Weatherhead, Anna Goldenberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.09353)  

**Abstract**: Time series data with missing values is common across many domains. Healthcare presents special challenges due to prolonged periods of sensor disconnection. In such cases, having a confidence measure for imputed values is critical. Most existing methods either overlook model uncertainty or lack mechanisms to estimate it. To address this gap, we introduce a general framework that quantifies and leverages uncertainty for selective imputation. By focusing on values the model is most confident in, highly unreliable imputations are avoided. Our experiments on multiple EHR datasets, covering diverse types of missingness, demonstrate that selectively imputing less-uncertain values not only reduces imputation errors but also improves downstream tasks. Specifically, we show performance gains in a 24-hour mortality prediction task, underscoring the practical benefit of incorporating uncertainty into time series imputation. 

**Abstract (ZH)**: 具有缺失值的时间序列数据在许多领域中都很常见。医疗保健领域因长时间传感器断开连接而面临特殊挑战。在这种情况下，对插补值具有置信度衡量至关重要。现有大多数方法要么忽略了模型不确定性，要么缺乏估计不确定性的机制。为解决这一问题，我们引入了一种一般框架，用于量化和利用不确定性进行选择性插补。通过专注于模型最自信的值，可以避免高度不可靠的插补。我们在多个EHR数据集上的实验涵盖了不同类型的缺失性，表明仅填充较低不确定性的值不仅可以减少插补错误，还可以改善下游任务。具体而言，我们在24小时死亡率预测任务中展示了性能提升，强调了将不确定性纳入时间序列插补中的实践益处。 

---
# A Framework for Predictive Directional Trading Based on Volatility and Causal Inference 

**Title (ZH)**: 基于波动率和因果推断的预测性方向交易框架 

**Authors**: Ivan Letteri  

**Link**: [PDF](https://arxiv.org/pdf/2507.09347)  

**Abstract**: Purpose: This study introduces a novel framework for identifying and exploiting predictive lead-lag relationships in financial markets. We propose an integrated approach that combines advanced statistical methodologies with machine learning models to enhance the identification and exploitation of predictive relationships between equities. Methods: We employed a Gaussian Mixture Model (GMM) to cluster nine prominent stocks based on their mid-range historical volatility profiles over a three-year period. From the resulting clusters, we constructed a multi-stage causal inference pipeline, incorporating the Granger Causality Test (GCT), a customised Peter-Clark Momentary Conditional Independence (PCMCI) test, and Effective Transfer Entropy (ETE) to identify robust, predictive linkages. Subsequently, Dynamic Time Warping (DTW) and a K-Nearest Neighbours (KNN) classifier were utilised to determine the optimal time lag for trade execution. The resulting strategy was rigorously backtested. Results: The proposed volatility-based trading strategy, tested from 8 June 2023 to 12 August 2023, demonstrated substantial efficacy. The portfolio yielded a total return of 15.38%, significantly outperforming the 10.39% return of a comparative Buy-and-Hold strategy. Key performance metrics, including a Sharpe Ratio up to 2.17 and a win rate up to 100% for certain pairs, confirmed the strategy's viability. Conclusion: This research contributes a systematic and robust methodology for identifying profitable trading opportunities derived from volatility-based causal relationships. The findings have significant implications for both academic research in financial modelling and the practical application of algorithmic trading, offering a structured approach to developing resilient, data-driven strategies. 

**Abstract (ZH)**: 目的：本文介绍了一种新型框架，用于识别和利用金融市场的预测领先-滞后关系。我们提出了一种综合方法，结合了高级统计方法和机器学习模型，以增强 Equity 之间的预测关系识别和利用。方法：我们使用高斯混合模型（GMM）根据九只表现突出股票在过去三年中期波动率概况进行聚类。从聚类结果中，我们构建了一个多阶段因果推理管道，结合了格兰杰因果检验（GCT）、自定义佩特-克拉克瞬时条件独立性（PCMCI）检验和有效转移熵（ETE），以识别稳健的预测联系。随后，我们使用动态时间规整（DTW）和K-最近邻（KNN）分类器来确定交易执行的最佳时间滞后。随后，该策略进行了严格的回测。结果：该基于波动率的交易策略从2023年6月8日到2023年8月12日测试，显示出显著的效果。该组合的总收益率为15.38%，显著优于比较的买入并持有策略的10.39%收益率。关键性能指标包括高达2.17的夏普比率和某些配对高达100%的胜率，验证了该策略的有效性。结论：本文贡献了一套系统且稳健的方法，用于从基于波动率的因果关系中识别可盈利的交易机会。研究结果对金融建模的学术研究和算法交易的实际应用具有重要的意义，提供了一种开发稳健的数据驱动策略的结构化方法。 

---
# Enhancing Interpretability in Software Change Management with Chain-of-Thought Reasoning 

**Title (ZH)**: 使用链式思考推理增强软件变更管理的可解释性 

**Authors**: Yongqian Sun, Weihua Kuang, Chao Shen, Xidao Wen, Tinghua Zheng, Heng Liu, Shenglin Zhang, Bo Wu, Dan Pei  

**Link**: [PDF](https://arxiv.org/pdf/2507.09315)  

**Abstract**: In modern online services, frequent software changes introduce significant risks. To tackle this challenge, we propose SCELM (Software Change Evaluation and Lifecycle Management), an end-to-end automated framework for software change management. SCELM aims to manage software changes efficiently and precisely, significantly reducing service failures and economic losses. 

**Abstract (ZH)**: 在现代在线服务中，频繁的软件变更引入了重大的风险。为了应对这一挑战，我们提出了SCELM（软件变更评估与生命周期管理）——一个端到端的自动化软件变更管理体系，旨在高效精准地管理软件变更，显著减少服务失败和经济损失。 

---
# AlphaVAE: Unified End-to-End RGBA Image Reconstruction and Generation with Alpha-Aware Representation Learning 

**Title (ZH)**: AlphaVAE：统一的端到端RGBA图像重建与生成及awarealpha表示学习 

**Authors**: Zile Wang, Hao Yu, Jiabo Zhan, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.09308)  

**Abstract**: Recent advances in latent diffusion models have achieved remarkable results in high-fidelity RGB image synthesis by leveraging pretrained VAEs to compress and reconstruct pixel data at low computational cost. However, the generation of transparent or layered content (RGBA image) remains largely unexplored, due to the lack of large-scale benchmarks. In this work, we propose ALPHA, the first comprehensive RGBA benchmark that adapts standard RGB metrics to four-channel images via alpha blending over canonical backgrounds. We further introduce ALPHAVAE, a unified end-to-end RGBA VAE that extends a pretrained RGB VAE by incorporating a dedicated alpha channel. The model is trained with a composite objective that combines alpha-blended pixel reconstruction, patch-level fidelity, perceptual consistency, and dual KL divergence constraints to ensure latent fidelity across both RGB and alpha representations. Our RGBA VAE, trained on only 8K images in contrast to 1M used by prior methods, achieves a +4.9 dB improvement in PSNR and a +3.2% increase in SSIM over LayerDiffuse in reconstruction. It also enables superior transparent image generation when fine-tuned within a latent diffusion framework. Our code, data, and models are released on this https URL for reproducibility. 

**Abstract (ZH)**: 近期在潜藏扩散模型方面的进展通过利用预训练的VAE在低computational成本下压缩和重构像素数据，已在高保真RGB图像生成中取得了显著成果。然而，透明或分层内容（RGBA图像）的生成仍缺乏深入探索，主要原因是没有大规模基准数据集。在本工作中，我们提出了ALPHA，这是首个通过alpha融合标准背景来适应四通道图像的全面RGBA基准，同时将标准RGB指标应用于四通道图像。我们还提出了ALPHAVAE，这是一种统一的端到端RGBA VAE，它通过引入专门的alpha通道扩展了预训练的RGB VAE。该模型通过结合alpha融合像素重构、块级别保真度、感知一致性以及双KL散度约束进行训练，以确保在RGB和alpha表示中保持潜空间保真度。与之前方法相比，我们的RGBA VAE仅在8K图像上进行训练而非100万张，实现了比LayerDiffuse更高的4.9 dB的PSNR和3.2%的SSIM的重建性能，并且在潜在扩散框架中微调后能够生成更优的透明图像。我们已将代码、数据和模型发布于此链接以保证可重复性。 

---
# ViT-ProtoNet for Few-Shot Image Classification: A Multi-Benchmark Evaluation 

**Title (ZH)**: ViT-ProtoNet在Few-Shot图像分类中的多基准评估 

**Authors**: Abdulvahap Mutlu, Şengül Doğan, Türker Tuncer  

**Link**: [PDF](https://arxiv.org/pdf/2507.09299)  

**Abstract**: The remarkable representational power of Vision Transformers (ViTs) remains underutilized in few-shot image classification. In this work, we introduce ViT-ProtoNet, which integrates a ViT-Small backbone into the Prototypical Network framework. By averaging class conditional token embeddings from a handful of support examples, ViT-ProtoNet constructs robust prototypes that generalize to novel categories under 5-shot settings. We conduct an extensive empirical evaluation on four standard benchmarks: Mini-ImageNet, FC100, CUB-200, and CIFAR-FS, including overlapped support variants to assess robustness. Across all splits, ViT-ProtoNet consistently outperforms CNN-based prototypical counterparts, achieving up to a 3.2\% improvement in 5-shot accuracy and demonstrating superior feature separability in latent space. Furthermore, it outperforms or is competitive with transformer-based competitors using a more lightweight backbone. Comprehensive ablations examine the impact of transformer depth, patch size, and fine-tuning strategy. To foster reproducibility, we release code and pretrained weights. Our results establish ViT-ProtoNet as a powerful, flexible approach for few-shot classification and set a new baseline for transformer-based meta-learners. 

**Abstract (ZH)**: Vision Transformers (ViTs)在少量样本图像分类中的卓越表征能力尚未充分利用。本文介绍了ViT-ProtoNet，将ViT-Small骨干网络集成到原型网络框架中。通过平均少量支持样本的类条件token嵌入，ViT-ProtoNet在5-shot设置下构建了鲁棒的原型，并能够泛化到新类别。我们在四个标准基准数据集Mini-ImageNet、FC100、CUB-200和CIFAR-FS上进行了广泛的经验评估，包括重叠支持集变体以评估鲁棒性。在所有分割中，ViT-ProtoNet一致优于基于CNN的原型模型，5-shot准确率最高提高3.2%，并在潜在空间中显示出更好的特征可分性。此外，它在使用更轻量级骨干网络时，优于或与基于变换器的竞争者相当。全面的消融实验检查了Transformer深度、 patch大小和微调策略的影响。为了促进可重复性，我们发布了代码和预训练权重。我们的结果确立了ViT-ProtoNet作为一种强大的、灵活的少量样本分类方法，并为基于变换器的元学习者设立了新的基准。 

---
# Prompt4Trust: A Reinforcement Learning Prompt Augmentation Framework for Clinically-Aligned Confidence Calibration in Multimodal Large Language Models 

**Title (ZH)**: Prompt4Trust：一种用于多模态大型语言模型临床对齐置信度校准的强化学习提示增强框架 

**Authors**: Anita Kriz, Elizabeth Laura Janes, Xing Shen, Tal Arbel  

**Link**: [PDF](https://arxiv.org/pdf/2507.09279)  

**Abstract**: Multimodal large language models (MLLMs) hold considerable promise for applications in healthcare. However, their deployment in safety-critical settings is hindered by two key limitations: (i) sensitivity to prompt design, and (ii) a tendency to generate incorrect responses with high confidence. As clinicians may rely on a model's stated confidence to gauge the reliability of its predictions, it is especially important that when a model expresses high confidence, it is also highly accurate. We introduce Prompt4Trust, the first reinforcement learning (RL) framework for prompt augmentation targeting confidence calibration in MLLMs. A lightweight LLM is trained to produce context-aware auxiliary prompts that guide a downstream task MLLM to generate responses in which the expressed confidence more accurately reflects predictive accuracy. Unlike conventional calibration techniques, Prompt4Trust specifically prioritizes aspects of calibration most critical for safe and trustworthy clinical decision-making. Beyond improvements driven by this clinically motivated calibration objective, our proposed method also improves task accuracy, achieving state-of-the-art medical visual question answering (VQA) performance on the PMC-VQA benchmark, which is composed of multiple-choice questions spanning diverse medical imaging modalities. Moreover, our framework trained with a small downstream task MLLM showed promising zero-shot generalization to larger MLLMs in our experiments, suggesting the potential for scalable calibration without the associated computational costs. This work demonstrates the potential of automated yet human-aligned prompt engineering for improving the the trustworthiness of MLLMs in safety critical settings. Our codebase can be found at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在医疗领域的应用前景广阔。然而，它们在安全性关键场景中的部署受到两个关键限制的阻碍：（i）对提示设计的敏感性；（ii）生成高置信度错误响应的倾向。鉴于临床医生可能依赖模型声明的置信度来评估预测的可靠性，当模型表达高置信度时，它也必须极其准确尤为重要。我们提出了Prompt4Trust，这是首个针对MLLMs置信度校准的目标强化学习（RL）框架。一个轻量级的LLM被训练生成上下文感知的辅助提示，以引导下游任务的MLLM生成更准确反映预测准确度的响应置信度。与传统的校准技术不同，Prompt4Trust特别关注对安全和可信临床决策最关键校准方面的优先级。除了由这一以临床动机为导向的校准目标驱动的改进之外，我们提出的方法还在医疗视觉问答（VQA）基准（由多种医学成像模态的多项选择题组成）上实现了最先进的性能。此外，我们的框架在小规模下游任务的MLLM上进行训练，在实验中展示了向大规模MLLM的零样本泛化潜力，表明了无关联计算成本的扩展校准的潜在可能。本研究展示了自动化但符合人类价值观的提示工程对于提高安全关键场景中MLLMs的可靠性具有潜力。我们的代码库可在以下网址找到。 

---
# Cross Knowledge Distillation between Artificial and Spiking Neural Networks 

**Title (ZH)**: 人工神经网络与脉冲神经网络之间的跨知识蒸馏 

**Authors**: Shuhan Ye, Yuanbin Qian, Chong Wang, Sunqi Lin, Jiazhen Xu, Jiangbo Qian, Yuqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.09269)  

**Abstract**: Recently, Spiking Neural Networks (SNNs) have demonstrated rich potential in computer vision domain due to their high biological plausibility, event-driven characteristic and energy-saving efficiency. Still, limited annotated event-based datasets and immature SNN architectures result in their performance inferior to that of Artificial Neural Networks (ANNs). To enhance the performance of SNNs on their optimal data format, DVS data, we explore using RGB data and well-performing ANNs to implement knowledge distillation. In this case, solving cross-modality and cross-architecture challenges is necessary. In this paper, we propose cross knowledge distillation (CKD), which not only leverages semantic similarity and sliding replacement to mitigate the cross-modality challenge, but also uses an indirect phased knowledge distillation to mitigate the cross-architecture challenge. We validated our method on main-stream neuromorphic datasets, including N-Caltech101 and CEP-DVS. The experimental results show that our method outperforms current State-of-the-Art methods. The code will be available at this https URL 

**Abstract (ZH)**: 近年来，由于其高度的生物合理性、事件驱动特性和节能效率，脉冲神经网络（SNNs）在计算机视觉领域展现出丰富的潜力。然而，受限于有限的标注事件数据集和不成熟的SNN架构，其性能仍低于人工神经网络（ANNs）。为了提升SNNs在最佳数据格式DVS数据上的性能，我们探索使用RGB数据和表现良好的ANNs来实现知识蒸馏。在这种情况下，跨模态和跨架构的挑战需要得到解决。本文提出了一种跨模态知识蒸馏（CKD）方法，该方法不仅利用语义相似性和滑动替换来缓解跨模态挑战，还采用间接分阶段知识蒸馏来缓解跨架构挑战。我们在主流的神经形态数据集N-Caltech101和CEP-DVS上验证了该方法。实验结果表明，该方法优于当前的最先进的方法。代码将在此链接处提供。 

---
# Controllable Patching for Compute-Adaptive Surrogate Modeling of Partial Differential Equations 

**Title (ZH)**: 可控 patching 用于部分微分方程计算自适应代理建模 

**Authors**: Payel Mukhopadhyay, Michael McCabe, Ruben Ohana, Miles Cranmer  

**Link**: [PDF](https://arxiv.org/pdf/2507.09264)  

**Abstract**: Patch-based transformer surrogates have become increasingly effective for modeling spatiotemporal dynamics, but the fixed patch size is a major limitation for budget-conscience deployment in production. We introduce two lightweight, architecture-agnostic modules-the Convolutional Kernel Modulator (CKM) and Convolutional Stride Modulator (CSM)-that enable dynamic patch size control at inference in patch based models, without retraining or accuracy loss. Combined with a cyclic patch-size rollout, our method mitigates patch artifacts and improves long-term stability for video-like prediction tasks. Applied to a range of challenging 2D and 3D PDE benchmarks, our approach improves rollout fidelity and runtime efficiency. To our knowledge, this is the first framework to enable inference-time patch-size tunability in patch-based PDE surrogates. Its plug-and-play design makes it broadly applicable across architectures-establishing a general foundation for compute-adaptive modeling in PDE surrogate tasks. 

**Abstract (ZH)**: 基于卷积核调制器和卷积步长调制器的动态 patches 大小控制方法在 PDE 代理模型中的应用 

---
# AGCD-Net: Attention Guided Context Debiasing Network for Emotion Recognition 

**Title (ZH)**: AGCD-网：注意力引导上下文去偏网络的情绪识别 

**Authors**: Varsha Devi, Amine Bohi, Pardeep Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.09248)  

**Abstract**: Context-aware emotion recognition (CAER) enhances affective computing in real-world scenarios, but traditional methods often suffer from context bias-spurious correlation between background context and emotion labels (e.g. associating ``garden'' with ``happy''). In this paper, we propose \textbf{AGCD-Net}, an Attention Guided Context Debiasing model that introduces \textit{Hybrid ConvNeXt}, a novel convolutional encoder that extends the ConvNeXt backbone by integrating Spatial Transformer Network and Squeeze-and-Excitation layers for enhanced feature recalibration. At the core of AGCD-Net is the Attention Guided - Causal Intervention Module (AG-CIM), which applies causal theory, perturbs context features, isolates spurious correlations, and performs an attention-driven correction guided by face features to mitigate context bias. Experimental results on the CAER-S dataset demonstrate the effectiveness of AGCD-Net, achieving state-of-the-art performance and highlighting the importance of causal debiasing for robust emotion recognition in complex settings. 

**Abstract (ZH)**: 基于注意力引导的上下文去偏模型（AGCD-Net）在情境感知情感识别中的应用 

---
# PanoDiff-SR: Synthesizing Dental Panoramic Radiographs using Diffusion and Super-resolution 

**Title (ZH)**: PanoDiff-SR: 使用扩散和超分辨合成功牙全景放射图像 

**Authors**: Sanyam Jain, Bruna Neves de Freitas, Andreas Basse-OConnor, Alexandros Iosifidis, Ruben Pauwels  

**Link**: [PDF](https://arxiv.org/pdf/2507.09227)  

**Abstract**: There has been increasing interest in the generation of high-quality, realistic synthetic medical images in recent years. Such synthetic datasets can mitigate the scarcity of public datasets for artificial intelligence research, and can also be used for educational purposes. In this paper, we propose a combination of diffusion-based generation (PanoDiff) and Super-Resolution (SR) for generating synthetic dental panoramic radiographs (PRs). The former generates a low-resolution (LR) seed of a PR (256 X 128) which is then processed by the SR model to yield a high-resolution (HR) PR of size 1024 X 512. For SR, we propose a state-of-the-art transformer that learns local-global relationships, resulting in sharper edges and textures. Experimental results demonstrate a Frechet inception distance score of 40.69 between 7243 real and synthetic images (in HR). Inception scores were 2.55, 2.30, 2.90 and 2.98 for real HR, synthetic HR, real LR and synthetic LR images, respectively. Among a diverse group of six clinical experts, all evaluating a mixture of 100 synthetic and 100 real PRs in a time-limited observation, the average accuracy in distinguishing real from synthetic images was 68.5% (with 50% corresponding to random guessing). 

**Abstract (ZH)**: 近年来，对生成高质量、逼真的合成医学图像越来越感兴趣。此类合成数据集可以缓解人工智能研究中公开数据集的稀缺问题，也可以用于教育目的。在本文中，我们提出了一种基于扩散生成（PanoDiff）和超分辨率（SR）结合的方法，用于生成合成牙科全景 radiographs（PRs）。前者生成一个低分辨率（LR）的 PR 种子（256 X 128），然后通过 SR 模型处理以产生高分辨率（HR）的 PR（1024 X 512）。对于超分辨率，我们提出了一种最先进的变换器，它可以学习局部-全局关系，从而产生更清晰的边缘和纹理。实验结果表明，在高分辨率下，7243 张真实和合成图像之间的弗雷切尔入inski 距离评分为 40.69。inception 分数分别为 2.55、2.30、2.90 和 2.98 对应真实高分辨率、合成高分辨率、真实低分辨率和合成低分辨率图像。在一个时间有限的观察中，六位临床专家之一对混合了 100 张合成和 100 张真实 PRs 进行评价，平均区分真实图像和合成图像的准确性为 68.5%（其中 50% 对应随机猜测）。 

---
# XiChen: An observation-scalable fully AI-driven global weather forecasting system with 4D variational knowledge 

**Title (ZH)**: XiChen：一种基于4D变分知识的可观察性可伸缩全AI驱动全球天气预报系统 

**Authors**: Wuxin Wang, Weicheng Ni, Lilan Huang, Tao Hao, Ben Fei, Shuo Ma, Taikang Yuan, Yanlai Zhao, Kefeng Deng, Xiaoyong Li, Boheng Duan, Lei Bai, Kaijun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2507.09202)  

**Abstract**: Recent advancements in Artificial Intelligence (AI) demonstrate significant potential to revolutionize weather forecasting. However, most AI-driven models rely on Numerical Weather Prediction (NWP) systems for initial condition preparation, which often consumes hours on supercomputers. Here we introduce XiChen, the first observation-scalable fully AI-driven global weather forecasting system, whose entire pipeline, from Data Assimilation (DA) to medium-range forecasting, can be accomplished within only 17 seconds. XiChen is built upon a foundation model that is pre-trained for weather forecasting. Meanwhile, this model is subsequently fine-tuned to serve as both observation operators and DA models, thereby scalably assimilating conventional and raw satellite observations. Furthermore, the integration of four-dimensional variational knowledge ensures that XiChen's DA and medium-range forecasting accuracy rivals that of operational NWP systems, amazingly achieving a skillful forecasting lead time exceeding 8.25 days. These findings demonstrate that XiChen holds strong potential toward fully AI-driven weather forecasting independent of NWP systems. 

**Abstract (ZH)**: Recent advancements in Artificial Intelligence (AI)显示了在气象预报领域革命性的潜力。然而，大多数基于AI的模型依赖数值天气预报（NWP）系统进行初始条件准备，这通常需要在超级计算机上耗时数小时。我们介绍了Xichen，这是首个可扩展观测的完全基于AI的全球气象预报系统，其从数据同化（DA）到中期预报的整个管道仅需17秒即可完成。Xichen基于一个为气象预报预训练的基准模型，进而微调以作为观测算子和数据同化模型，从而可扩展地同化常规和原始卫星观测。此外，四维变分知识的整合确保了Xichen的数据同化和中期预报准确度与 operational NWP 系统相当，令人惊讶地实现了超前预报时效超过8.25天。这些发现表明，Xichen有可能完全独立于NWP系统实现基于AI的气象预报。 

---
# Continual Reinforcement Learning by Planning with Online World Models 

**Title (ZH)**: 持续强化学习中的在线世界模型规划 

**Authors**: Zichen Liu, Guoji Fu, Chao Du, Wee Sun Lee, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.09177)  

**Abstract**: Continual reinforcement learning (CRL) refers to a naturalistic setting where an agent needs to endlessly evolve, by trial and error, to solve multiple tasks that are presented sequentially. One of the largest obstacles to CRL is that the agent may forget how to solve previous tasks when learning a new task, known as catastrophic forgetting. In this paper, we propose to address this challenge by planning with online world models. Specifically, we learn a Follow-The-Leader shallow model online to capture the world dynamics, in which we plan using model predictive control to solve a set of tasks specified by any reward functions. The online world model is immune to forgetting by construction with a proven regret bound of $\mathcal{O}(\sqrt{K^2D\log(T)})$ under mild assumptions. The planner searches actions solely based on the latest online model, thus forming a FTL Online Agent (OA) that updates incrementally. To assess OA, we further design Continual Bench, a dedicated environment for CRL, and compare with several strong baselines under the same model-planning algorithmic framework. The empirical results show that OA learns continuously to solve new tasks while not forgetting old skills, outperforming agents built on deep world models with various continual learning techniques. 

**Abstract (ZH)**: 持续强化学习（CRL）是指一种自然环境，其中智能体需要通过不断尝试和错误来解决依次呈现的多个任务并持续演化。CRL的一大挑战是，智能体在学习新任务时可能会忘记之前任务的解决方法，这被称为灾难性遗忘。在本文中，我们提出通过在线世界模型规划来应对这一挑战。具体而言，我们在线学习一个跟随领导者（Follow-The-Leader）的浅层模型以捕捉世界动力学，并使用模型预测控制来解决任意奖励函数指定的任务集。该在线世界模型通过在温和假设下具有证明的遗憾界$\mathcal{O}(\sqrt{K^2D\log(T)})$而设计为不会遗忘。规划器仅基于最新的在线模型搜索动作，从而形成一个增量更新的FTL在线智能体（OA）。为了评估OA，我们进一步设计了一个专用环境Continual Bench，并在相同的模型-规划算法框架下与多个强基线进行对比。实验证明，OA能够连续学习以解决新任务而不忘记旧技能，在与各种持续学习技术结合的深层世界模型构建的智能体中表现出色。 

---
# Towards Interpretable Drug-Drug Interaction Prediction: A Graph-Based Approach with Molecular and Network-Level Explanations 

**Title (ZH)**: 基于图的分子和网络层面解释的可解释药物-药物相互作用预测 

**Authors**: Mengjie Chen, Ming Zhang, Cunquan Qu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09173)  

**Abstract**: Drug-drug interactions (DDIs) represent a critical challenge in pharmacology, often leading to adverse drug reactions with significant implications for patient safety and healthcare outcomes. While graph-based methods have achieved strong predictive performance, most approaches treat drug pairs independently, overlooking the complex, context-dependent interactions unique to drug pairs. Additionally, these models struggle to integrate biological interaction networks and molecular-level structures to provide meaningful mechanistic insights. In this study, we propose MolecBioNet, a novel graph-based framework that integrates molecular and biomedical knowledge for robust and interpretable DDI prediction. By modeling drug pairs as unified entities, MolecBioNet captures both macro-level biological interactions and micro-level molecular influences, offering a comprehensive perspective on DDIs. The framework extracts local subgraphs from biomedical knowledge graphs and constructs hierarchical interaction graphs from molecular representations, leveraging classical graph neural network methods to learn multi-scale representations of drug pairs. To enhance accuracy and interpretability, MolecBioNet introduces two domain-specific pooling strategies: context-aware subgraph pooling (CASPool), which emphasizes biologically relevant entities, and attention-guided influence pooling (AGIPool), which prioritizes influential molecular substructures. The framework further employs mutual information minimization regularization to enhance information diversity during embedding fusion. Experimental results demonstrate that MolecBioNet outperforms state-of-the-art methods in DDI prediction, while ablation studies and embedding visualizations further validate the advantages of unified drug pair modeling and multi-scale knowledge integration. 

**Abstract (ZH)**: 基于分子和生物医学知识的药物-药物相互作用预测框架：MolecBioNet 

---
# Automatic Contouring of Spinal Vertebrae on X-Ray using a Novel Sandwich U-Net Architecture 

**Title (ZH)**: 使用新颖的三明治U-Net架构在X光上自动轮廓化脊椎 vertebrae 

**Authors**: Sunil Munthumoduku Krishna Murthy, Kumar Rajamani, Srividya Tirunellai Rajamani, Yupei Li, Qiyang Sun, Bjoern W. Schuller  

**Link**: [PDF](https://arxiv.org/pdf/2507.09158)  

**Abstract**: In spinal vertebral mobility disease, accurately extracting and contouring vertebrae is essential for assessing mobility impairments and monitoring variations during flexion-extension movements. Precise vertebral contouring plays a crucial role in surgical planning; however, this process is traditionally performed manually by radiologists or surgeons, making it labour-intensive, time-consuming, and prone to human error. In particular, mobility disease analysis requires the individual contouring of each vertebra, which is both tedious and susceptible to inconsistencies. Automated methods provide a more efficient alternative, enabling vertebra identification, segmentation, and contouring with greater accuracy and reduced time consumption. In this study, we propose a novel U-Net variation designed to accurately segment thoracic vertebrae from anteroposterior view on X-Ray images. Our proposed approach, incorporating a ``sandwich" U-Net structure with dual activation functions, achieves a 4.1\% improvement in Dice score compared to the baseline U-Net model, enhancing segmentation accuracy while ensuring reliable vertebral contour extraction. 

**Abstract (ZH)**: 脊椎移动性疾病中，准确提取和勾勒椎体对于评估移动功能障碍和监测屈伸运动中的变化至关重要。精确的椎体勾勒在手术规划中起着关键作用；然而，这一过程通常由放射科医生或外科医生手工完成，导致劳动密集型、耗时且易出错。特别是移动性疾病分析需要对每个椎体进行单独勾勒，既繁琐又容易出现不一致性。自动化方法提供了更高效的替代方案，能够以更高的精度和更少的时间消耗进行椎体识别、分割和勾勒。在本研究中，我们提出了一种新型U-Net变体，旨在从X射线正位图像中准确分割胸椎。我们提出的方法结合了“三明治”U-Net结构和双重激活函数，与基线U-Net模型相比，在Dice分数上实现了4.1%的改进，同时提高了分割精度并确保可靠的椎体轮廓提取。 

---
# OPENXRD: A Comprehensive Benchmark and Enhancement Framework for LLM/MLLM XRD Question Answering 

**Title (ZH)**: OPENXRD：一个全面的LLM/MLLM XRD问答基准与增强框架 

**Authors**: Ali Vosoughi, Ayoub Shahnazari, Yufeng Xi, Zeliang Zhang, Griffin Hess, Chenliang Xu, Niaz Abdolrahim  

**Link**: [PDF](https://arxiv.org/pdf/2507.09155)  

**Abstract**: This work presents OPENXRD, an open-book pipeline designed for crystallography question answering, which integrates textual prompts with concise supporting content generated by GPT-4.5. Instead of using scanned textbooks, which may lead to copyright issues, OPENXRD generates compact, domain-specific references that help smaller models understand key concepts in X-ray diffraction (XRD). We evaluate OPENXRD on a well-defined set of 217 expert-level XRD questions by comparing different vision-language models, including GPT-4 and LLaVA-based frameworks such as Mistral, LLaMA, and QWEN, under both closed-book (without supporting material) and open-book (with supporting material) conditions. Our experimental results show significant accuracy improvements in models that use the GPT-4.5-generated summaries, particularly those with limited prior training in crystallography. OPENXRD uses knowledge from larger models to fill knowledge gaps in crystallography and shows that AI-generated texts can help smaller models reason more effectively in scientific tasks. While the current version of OPENXRD focuses on text-based inputs, we also explore future extensions such as adding real crystal diagrams or diffraction patterns to improve interpretation in specialized materials science contexts. Overall, OPENXRD shows that specialized open-book systems can be useful in materials science and provides a foundation for broader natural language processing (NLP) tools in critical scientific fields. 

**Abstract (ZH)**: OPENXRD：面向晶体学问答的开放书流水线，结合由GPT-4.5生成的简洁支持内容 

---
# Advanced Health Misinformation Detection Through Hybrid CNN-LSTM Models Informed by the Elaboration Likelihood Model (ELM) 

**Title (ZH)**: 基于 elaboration likelihood model (ELM) 的混合 CNN-LSTM 模型在先进健康误导信息检测中的应用 

**Authors**: Mkululi Sikosana, Sean Maudsley-Barton, Oluwaseun Ajao  

**Link**: [PDF](https://arxiv.org/pdf/2507.09149)  

**Abstract**: Health misinformation during the COVID-19 pandemic has significantly challenged public health efforts globally. This study applies the Elaboration Likelihood Model (ELM) to enhance misinformation detection on social media using a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) model. The model aims to enhance the detection accuracy and reliability of misinformation classification by integrating ELM-based features such as text readability, sentiment polarity, and heuristic cues (e.g., punctuation frequency). The enhanced model achieved an accuracy of 97.37%, precision of 96.88%, recall of 98.50%, F1-score of 97.41%, and ROC-AUC of 99.50%. A combined model incorporating feature engineering further improved performance, achieving a precision of 98.88%, recall of 99.80%, F1-score of 99.41%, and ROC-AUC of 99.80%. These findings highlight the value of ELM features in improving detection performance, offering valuable contextual information. This study demonstrates the practical application of psychological theories in developing advanced machine learning algorithms to address health misinformation effectively. 

**Abstract (ZH)**: COVID-19疫情期间的健康 misinformation 对全球公共健康努力构成了重大挑战。本研究运用扩展可能性路径模型（ELM）结合卷积神经网络（CNN）和长短期记忆（LSTM）模型，以提高社交媒体上的 misinformation 检测准确性。该模型通过整合基于ELM的特征（如文本可读性、情感极性和启发性线索，例如标点符号频率）来提高 misinformation 分类的准确性和可靠性。增强模型的准确率为97.37%，精确率为96.88%，召回率为98.50%，F1得分97.41%，ROC-AUC为99.50%。结合特征工程的综合模型进一步提高了性能，精确率为98.88%，召回率为99.80%，F1得分99.41%，ROC-AUC为99.80%。这些发现突显了ELM特征在提高检测性能方面的价值，提供了有价值的语言背景信息。本研究展示了通过运用心理学理论开发高级机器学习算法以有效应对健康 misinformation 的实际应用。 

---
# POIFormer: A Transformer-Based Framework for Accurate and Scalable Point-of-Interest Attribution 

**Title (ZH)**: POIFormer: 基于Transformer的准确可扩展的兴趣点归属框架 

**Authors**: Nripsuta Ani Saxena, Shang-Ling Hsu, Mehul Shetty, Omar Alkhadra, Cyrus Shahabi, Abigail L. Horn  

**Link**: [PDF](https://arxiv.org/pdf/2507.09137)  

**Abstract**: Accurately attributing user visits to specific Points of Interest (POIs) is a foundational task for mobility analytics, personalized services, marketing and urban planning. However, POI attribution remains challenging due to GPS inaccuracies, typically ranging from 2 to 20 meters in real-world settings, and the high spatial density of POIs in urban environments, where multiple venues can coexist within a small radius (e.g., over 50 POIs within a 100-meter radius in dense city centers). Relying on proximity is therefore often insufficient for determining which POI was actually visited. We introduce \textsf{POIFormer}, a novel Transformer-based framework for accurate and efficient POI attribution. Unlike prior approaches that rely on limited spatiotemporal, contextual, or behavioral features, \textsf{POIFormer} jointly models a rich set of signals, including spatial proximity, visit timing and duration, contextual features from POI semantics, and behavioral features from user mobility and aggregated crowd behavior patterns--using the Transformer's self-attention mechanism to jointly model complex interactions across these dimensions. By leveraging the Transformer to model a user's past and future visits (with the current visit masked) and incorporating crowd-level behavioral patterns through pre-computed KDEs, \textsf{POIFormer} enables accurate, efficient attribution in large, noisy mobility datasets. Its architecture supports generalization across diverse data sources and geographic contexts while avoiding reliance on hard-to-access or unavailable data layers, making it practical for real-world deployment. Extensive experiments on real-world mobility datasets demonstrate significant improvements over existing baselines, particularly in challenging real-world settings characterized by spatial noise and dense POI clustering. 

**Abstract (ZH)**: 准确归因用户访问到特定兴趣点（POI）是移动分析、个性化服务、营销和城市规划的基础任务。然而，由于GPS精度问题，在实际环境中通常从2到20米不等，以及城市环境中POI的高度空间密度（例如，在密集城市的100米半径内可能有超过50个POI），依赖于接近性往往不足以确定实际访问的POI。我们介绍了\textsf{POIFormer}，这是一种基于Transformer的新型框架，用于准确和高效地进行POI归因。与依赖有限的时空、上下文或行为特征的先前方法不同，\textsf{POIFormer} 联合建模了丰富的信号，包括空间接近性、访问时间和持续时间、从POI语义中获得的上下文特征以及从用户移动性和聚合人群行为模式中获得的行为特征——利用Transformer的自注意力机制联合建模这些维度上的复杂交互。通过利用Transformer建模用户过去和未来的访问（当前访问被遮蔽）并结合预先计算的核密度估计（KDE）来融入人群层次的行为模式，\textsf{POIFormer} 在大型嘈杂的移动数据集中实现了准确且高效的归因。其架构支持跨不同数据源和地理背景的一般化，同时避免了对难以获取或不可用数据层的依赖，使其适用于实际部署。在实际移动数据集上的 extensively 实验表明，\textsf{POIFormer} 在具有空间噪声和密集POI聚类特征的挑战性实际环境中显著优于现有baseline。 

---
# Heterogeneous Graph Prompt Learning via Adaptive Weight Pruning 

**Title (ZH)**: 异构图提示学习通过自适应权重剪枝 

**Authors**: Chu-Yuan Wei, Shun-Yao Liu, Sheng-Da Zhuo, Chang-Dong Wang, Shu-Qiang Huang, Mohsen Guizani  

**Link**: [PDF](https://arxiv.org/pdf/2507.09132)  

**Abstract**: Graph Neural Networks (GNNs) have achieved remarkable success in various graph-based tasks (e.g., node classification or link prediction). Despite their triumphs, GNNs still face challenges such as long training and inference times, difficulty in capturing complex relationships, and insufficient feature extraction. To tackle these issues, graph pre-training and graph prompt methods have garnered increasing attention for their ability to leverage large-scale datasets for initial learning and task-specific adaptation, offering potential improvements in GNN performance. However, previous research has overlooked the potential of graph prompts in optimizing models, as well as the impact of both positive and negative graph prompts on model stability and efficiency. To bridge this gap, we propose a novel framework combining graph prompts with weight pruning, called GPAWP, which aims to enhance the performance and efficiency of graph prompts by using fewer of them. We evaluate the importance of graph prompts using an importance assessment function to determine positive and negative weights at different granularities. Through hierarchically structured pruning, we eliminate negative prompt labels, resulting in more parameter-efficient and competitively performing prompts. Extensive experiments on three benchmark datasets demonstrate the superiority of GPAWP, leading to a significant reduction in parameters in node classification tasks. 

**Abstract (ZH)**: 基于图提示与权重剪枝的图神经网络优化框架（GPAWP） 

---
# Towards Human-level Dexterity via Robot Learning 

**Title (ZH)**: 通过机器人学习实现人类水平的灵巧性 

**Authors**: Gagan Khandate  

**Link**: [PDF](https://arxiv.org/pdf/2507.09117)  

**Abstract**: Dexterous intelligence -- the ability to perform complex interactions with multi-fingered hands -- is a pinnacle of human physical intelligence and emergent higher-order cognitive skills. However, contrary to Moravec's paradox, dexterous intelligence in humans appears simple only superficially. Many million years were spent co-evolving the human brain and hands including rich tactile sensing. Achieving human-level dexterity with robotic hands has long been a fundamental goal in robotics and represents a critical milestone toward general embodied intelligence. In this pursuit, computational sensorimotor learning has made significant progress, enabling feats such as arbitrary in-hand object reorientation. However, we observe that achieving higher levels of dexterity requires overcoming very fundamental limitations of computational sensorimotor learning.
I develop robot learning methods for highly dexterous multi-fingered manipulation by directly addressing these limitations at their root cause. Chiefly, through key studies, this disseration progressively builds an effective framework for reinforcement learning of dexterous multi-fingered manipulation skills. These methods adopt structured exploration, effectively overcoming the limitations of random exploration in reinforcement learning. The insights gained culminate in a highly effective reinforcement learning that incorporates sampling-based planning for direct exploration. Additionally, this thesis explores a new paradigm of using visuo-tactile human demonstrations for dexterity, introducing corresponding imitation learning techniques. 

**Abstract (ZH)**: 灵巧智能——多指手进行复杂交互的能力是人类物理智能和高级认知技能的顶峰。然而，与莫拉韦克悖论相反，人类的灵巧智能在表面上看似简单，但实际上经过了数百万年的大脑和手的协同进化，包含了丰富的触觉感知。实现类人的灵巧智能一直是机器人领域的根本目标，并代表着通向一般 embodded 智能的关键里程碑。在这一追求中，计算感知运动学习取得了显著进展，能够实现任意的手中物体重新定向等壮举。然而，我们观察到，实现更高的灵巧水平需要克服计算感知运动学习的基本局限性。

我通过直接针对这些局限性的根本原因，开发了机器人学习方法以实现高度灵巧的多指操纵。主要地，通过关键研究，本论文逐步建立了一个有效的强化学习框架，用于学习灵巧的多指操纵技能。这些方法采用结构化探索，有效地克服了强化学习中随机探索的局限性。获得的洞见最终在结合采样规划直接探索的强化学习中得到了充分体现。此外，本论文还探讨了一种新的使用视触觉人类示范进行灵巧的新范式，并引入相应的模仿学习技术。 

---
# SPICE: An Automated SWE-Bench Labeling Pipeline for Issue Clarity, Test Coverage, and Effort Estimation 

**Title (ZH)**: SPICE: 一种自动化的SWE-Bench标签流水线，用于问题清晰度、测试覆盖率和努力估计。 

**Authors**: Aaditya Bhatia, Gustavo A. Oliva, Gopi Krishnan Rajbahadur, Haoxiang Zhang, Yihao Chen, Zhilong Chen, Arthur Leung, Dayi Lin, Boyuan Chen, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2507.09108)  

**Abstract**: High-quality labeled datasets are crucial for training and evaluating foundation models in software engineering, but creating them is often prohibitively expensive and labor-intensive. We introduce SPICE, a scalable, automated pipeline for labeling SWE-bench-style datasets with annotations for issue clarity, test coverage, and effort estimation. SPICE combines context-aware code navigation, rationale-driven prompting, and multi-pass consensus to produce labels that closely approximate expert annotations. SPICE's design was informed by our own experience and frustration in labeling more than 800 instances from SWE-Gym. SPICE achieves strong agreement with human-labeled SWE-bench Verified data while reducing the cost of labeling 1,000 instances from around $100,000 (manual annotation) to just $5.10. These results demonstrate SPICE's potential to enable cost-effective, large-scale dataset creation for SE-focused FMs. To support the community, we release both SPICE tool and SPICE Bench, a new dataset of 6,802 SPICE-labeled instances curated from 291 open-source projects in SWE-Gym (over 13x larger than SWE-bench Verified). 

**Abstract (ZH)**: 高质量标注数据集是软件工程中训练和评估基础模型的关键，但创建它们往往代价高昂且劳动密集。我们介绍了SPICE，一种可扩展的自动化流水线，用于标注类似于SWE-bench的 datasets，并提供关于问题清晰度、测试覆盖率和努力估计的标注。SPICE结合上下文感知的代码导航、基于理据的提示和多轮共识，生成与专家标注接近的标签。SPICE的设计基于我们自己在为超过800个实例进行SWE-Gym标注时的经验和挫败感。SPICE在与人工标注的SWE-bench Verified数据达成强烈一致的同时，将1,000个实例的标注成本从约100,000美元（手动标注）降低到仅5.10美元。这些结果表明SPICE有可能促进面向SE的基础模型的大规模、成本效益型数据集创建。为支持社区，我们发布了SPICE工具和包含6,802个SPICE标注实例的新数据集SPICE Bench，这些实例来自291个开源项目（比SWE-bench Verified大13倍以上）。 

---
# CompassJudger-2: Towards Generalist Judge Model via Verifiable Rewards 

**Title (ZH)**: CompassJudger-2：通往可验证奖励的一般判断模型 

**Authors**: Taolin Zhang, Maosong Cao, Alexander Lam, Songyang Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.09104)  

**Abstract**: Recently, the role of LLM-as-judge in evaluating large language models has gained prominence. However, current judge models suffer from narrow specialization and limited robustness, undermining their capacity for comprehensive evaluations. In this work, we present CompassJudger-2, a novel generalist judge model that overcomes these limitations via a task-driven, multi-domain data curation strategy. Central to our approach is supervising judgment tasks with verifiable rewards, guiding intrinsic critical reasoning through rejection sampling to foster robust, generalizable judgment capabilities. We introduce a refined learning objective with margin policy gradient loss to enhance performance. Empirically, CompassJudger-2 achieves superior results across multiple judge and reward benchmarks, and our 7B model demonstrates competitive judgment accuracy with significantly larger models like DeepSeek-V3 and Qwen3-235B-A22B. Additionally, we propose JudgerBenchV2, a comprehensive benchmark evaluating cross-domain judgment accuracy and rank consistency to standardize judge model evaluation. These contributions advance robust, scalable LLM judgment and establish new performance and evaluation standards. 

**Abstract (ZH)**: 最近，LLM-as-judge在评估大型语言模型中的作用日益凸显。然而，当前的法官模型存在专业知识狭窄和鲁棒性不足的问题，这削弱了它们的全面评估能力。在此工作中，我们提出了CompassJudger-2这一新型通用法官模型，通过任务驱动、多领域数据编纂策略克服了这些限制。我们的方法核心在于使用可验证奖励监督判断任务，通过拒绝采样引导内在批判性推理，以培养鲁棒且通用的判断能力。我们引入了带有边际政策梯度损失的精炼学习目标以提升性能。实验结果表明，CompassJudger-2在多个法官和奖励基准测试中取得了优于现有模型的成绩，7B模型在判断准确性方面与DeepSeek-V3和Qwen3-235B-A22B等更大规模的模型具有竞争力。此外，我们提出了JudgerBenchV2，这是一个全面的基准测试，用于评估跨领域的判断准确性和排名一致性，以标准化法官模型的评估。这些贡献推动了鲁棒、可扩展的LLM判断的发展，并确立了新的性能和评估标准。 

---
# AInsight: Augmenting Expert Decision-Making with On-the-Fly Insights Grounded in Historical Data 

**Title (ZH)**: AInsight：利用历史数据支持实时洞察以增强专家决策Making 

**Authors**: Mohammad Abolnejadian, Shakiba Amirshahi, Matthew Brehmer, Anamaria Crisan  

**Link**: [PDF](https://arxiv.org/pdf/2507.09100)  

**Abstract**: In decision-making conversations, experts must navigate complex choices and make on-the-spot decisions while engaged in conversation. Although extensive historical data often exists, the real-time nature of these scenarios makes it infeasible for decision-makers to review and leverage relevant information. This raises an interesting question: What if experts could utilize relevant past data in real-time decision-making through insights derived from past data? To explore this, we implemented a conversational user interface, taking doctor-patient interactions as an example use case. Our system continuously listens to the conversation, identifies patient problems and doctor-suggested solutions, and retrieves related data from an embedded dataset, generating concise insights using a pipeline built around a retrieval-based Large Language Model (LLM) agent. We evaluated the prototype by embedding Health Canada datasets into a vector database and conducting simulated studies using sample doctor-patient dialogues, showing effectiveness but also challenges, setting directions for the next steps of our work. 

**Abstract (ZH)**: 在决策对话中，专家必须在对话过程中导航复杂的选项并作出即时决策。尽管历史数据可能非常丰富，但这些场景的实时性质使得决策者难以回顾和利用相关信息。这就引出一个有趣的问题：如果专家能够通过基于过去数据见解的方式，在实时决策中利用相关过去数据会怎样？为了探索这一点，我们实现了一个对话用户界面，以医生-患者互动为例。我们的系统持续监听对话，识别患者问题和医生建议的解决方案，并从嵌入式数据集中检索相关数据，使用围绕检索型大型语言模型（LLM）代理构建的管道生成简洁的见解。我们通过将卫生部 canada 数据集嵌入向量数据库并在模拟研究中使用样本医生-患者对话来评估原型，展示了其有效性但也暴露出挑战，并为下一步工作的方向提供了指导。 

---
# Deep Reinforcement Learning with Gradient Eligibility Traces 

**Title (ZH)**: 基于梯度有效性追溯的深度强化学习 

**Authors**: Esraa Elelimy, Brett Daley, Andrew Patterson, Marlos C. Machado, Adam White, Martha White  

**Link**: [PDF](https://arxiv.org/pdf/2507.09087)  

**Abstract**: Achieving fast and stable off-policy learning in deep reinforcement learning (RL) is challenging. Most existing methods rely on semi-gradient temporal-difference (TD) methods for their simplicity and efficiency, but are consequently susceptible to divergence. While more principled approaches like Gradient TD (GTD) methods have strong convergence guarantees, they have rarely been used in deep RL. Recent work introduced the Generalized Projected Bellman Error ($\GPBE$), enabling GTD methods to work efficiently with nonlinear function approximation. However, this work is only limited to one-step methods, which are slow at credit assignment and require a large number of samples. In this paper, we extend the $\GPBE$ objective to support multistep credit assignment based on the $\lambda$-return and derive three gradient-based methods that optimize this new objective. We provide both a forward-view formulation compatible with experience replay and a backward-view formulation compatible with streaming algorithms. Finally, we evaluate the proposed algorithms and show that they outperform both PPO and StreamQ in MuJoCo and MinAtar environments, respectively. Code available at this https URL\_algos 

**Abstract (ZH)**: 实现深度强化学习中的快速稳定离策学习颇具挑战性。大多数现有方法依赖于半梯度时差（TD）方法以保持简单性和高效性，但因此容易发散。虽然具有更原理基础的方法如梯度TD（GTD）方法能够提供强收敛保证，但它们在深度RL中鲜有应用。近期研究引入了广义投影贝尔曼误差（$\GPBE$），使GTD方法能够在非线性函数逼近下高效工作。然而，这项工作仅限于单步方法，后者在功劳分配上速度慢且需要大量样本。在本文中，我们扩展了$\GPBE$目标，支持基于$\lambda$-回报的多步功劳分配，并推导出三种基于梯度的方法来优化这一新目标。我们提供了与经验回放兼容的前向视图形式化和与流式算法兼容的后向视图形式化。最后，我们评估了所提算法，并分别证明它们在MuJoCo和MinAtar环境中优于PPO和StreamQ。代码可在此访问：this https URL\_algos。 

---
# Queue up for takeoff: a transferable deep learning framework for flight delay prediction 

**Title (ZH)**: 排队起飞：一种可移植的深度学习框架用于航班延误预测 

**Authors**: Nnamdi Daniel Aghanya, Ta Duong Vu, Amaëlle Diop, Charlotte Deville, Nour Imane Kerroumi, Irene Moulitsas, Jun Li, Desmond Bisandu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09084)  

**Abstract**: Flight delays are a significant challenge in the aviation industry, causing major financial and operational disruptions. To improve passenger experience and reduce revenue loss, flight delay prediction models must be both precise and generalizable across different networks. This paper introduces a novel approach that combines Queue-Theory with a simple attention model, referred to as the Queue-Theory SimAM (QT-SimAM). To validate our model, we used data from the US Bureau of Transportation Statistics, where our proposed QT-SimAM (Bidirectional) model outperformed existing methods with an accuracy of 0.927 and an F1 score of 0.932. To assess transferability, we tested the model on the EUROCONTROL dataset. The results demonstrated strong performance, achieving an accuracy of 0.826 and an F1 score of 0.791. Ultimately, this paper outlines an effective, end-to-end methodology for predicting flight delays. The proposed model's ability to forecast delays with high accuracy across different networks can help reduce passenger anxiety and improve operational decision-making 

**Abstract (ZH)**: 航空延误是航空业的一项重大挑战，会导致严重的财务和运营中断。为了提高乘客体验并减少收入损失，飞行延误预测模型必须既精确又能在不同的网络中泛化。本文提出了一种结合排队理论和简单注意机制的新方法，称为排队理论相似注意力模型（QT-SimAM）。为了验证我们的模型，我们使用了美国交通统计局的数据，我们的提出的QT-SimAM（双向）模型在准确率0.927和F1分数0.932的情况下超过了现有方法。为了评估其可迁移性，我们将模型应用于EUROCONTROL数据集。结果表明，该模型表现强劲，准确率为0.826，F1分数为0.791。最终，本文概述了一种有效的端到端飞行延误预测方法。所提出的模型能够在不同的网络中以高精度预测延误，有助于减轻乘客的焦虑并改善运营决策。 

---
# Learning from Synthetic Labs: Language Models as Auction Participants 

**Title (ZH)**: 从合成实验室学习：语言模型作为拍卖参与者 

**Authors**: Anand Shah, Kehang Zhu, Yanchen Jiang, Jeffrey G. Wang, Arif K. Dayi, John J. Horton, David C. Parkes  

**Link**: [PDF](https://arxiv.org/pdf/2507.09083)  

**Abstract**: This paper investigates the behavior of simulated AI agents (large language models, or LLMs) in auctions, introducing a novel synthetic data-generating process to help facilitate the study and design of auctions. We find that LLMs -- when endowed with chain of thought reasoning capacity -- agree with the experimental literature in auctions across a variety of classic auction formats. In particular, we find that LLM bidders produce results consistent with risk-averse human bidders; that they perform closer to theoretical predictions in obviously strategy-proof auctions; and, that they succumb to the winner's curse in common value settings. On prompting, we find that LLMs are not very sensitive to naive changes in prompts (e.g., language, currency) but can improve dramatically towards theoretical predictions with the right mental model (i.e., the language of Nash deviations). We run 1,000$+$ auctions for less than $\$$400 with GPT-4 models (three orders of magnitude cheaper than modern auction experiments) and develop a framework flexible enough to run auction experiments with any LLM model and a wide range of auction design specifications, facilitating further experimental study by decreasing costs and serving as a proof-of-concept for the use of LLM proxies. 

**Abstract (ZH)**: 本文探讨了模拟AI代理（大型语言模型，或LLMs）在拍卖中的行为，引入了一种新颖的合成数据生成过程，以促进对拍卖的研究和设计。我们发现，当LLMs具备链式推理能力时，在多种经典拍卖格式中，它们的行为与实验文献中的结果一致。具体而言，我们发现LLM竞标者产生的结果与风险规避的人类竞标者一致；在显然策略证明的拍卖中，它们更接近理论预测；在共同价值设置中，它们也会受到出价者诅咒的影响。在 prompting 下，我们发现LLMs对简单的提示变化（例如语言、货币）不太敏感，但可以通过正确的心理模型（即纳什偏离的语言）显著改善其表现，趋向理论预测。我们使用GPT-4模型进行不到$400（约为现代拍卖实验成本的三个数量级更低）的1,000次以上拍卖，并开发了一个灵活的框架，可以用于任何LLM模型和广泛拍卖设计规范的拍卖实验，从而降低实验成本，并作为使用LLM代理的可行性证明。 

---
# Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation 

**Title (ZH)**: 动态参数内存：增强型LoRA临时LLM在对话中长序列情感识别中的应用 

**Authors**: Jialong Mai, Xiaofen Xing, Yawei Li, Zhipeng Li, Jingyuan Xing, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09076)  

**Abstract**: Recent research has focused on applying speech large language model (SLLM) to improve speech emotion recognition (SER). However, the inherently high frame rate in speech modality severely limits the signal processing and understanding capabilities of SLLM. For example, a SLLM with a 4K context window can only process 80 seconds of audio at 50Hz feature sampling rate before reaching its capacity limit. Input token compression methods used in SLLM overlook the continuity and inertia of emotions across multiple conversation turns. This paper proposes a Dynamic Parameter Memory (DPM) mechanism with contextual semantics and sentence-level emotion encoding, enabling processing of unlimited-length audio with limited context windows in SLLM. Specifically, DPM progressively encodes sentence-level information and emotions into a temporary LoRA module during inference to effectively "memorize" the contextual information. We trained an emotion SLLM as a backbone and incorporated our DPM into inference for emotion recognition in conversation (ERC). Experimental results on the IEMOCAP dataset show that DPM significantly improves the emotion recognition capabilities of SLLM when processing long audio sequences, achieving state-of-the-art performance. 

**Abstract (ZH)**: 近期研究集中在将语音大语言模型（SLLM）应用于提升语音情感识别（SER）能力，但由于语音模态固有的高帧率严重限制了SLLM的信号处理和理解能力。例如，一个具有4K上下文窗的SLLM在50Hz特征采样率下只能处理80秒的音频，并达到容量极限。SLLM中使用的输入 token 压缩方法忽略了多次对话轮次中情感的连续性和惯性。本文提出了一种带有上下文语义和句子级情感编码的动态参数记忆（DPM）机制，使SLLM能够在有限上下文窗的条件下处理无限长度的音频。具体来说，DPM在推理过程中逐步将句子级信息和情感编码进临时的LoRA模块，有效“记忆”上下文信息。我们以情感SLLM为骨干，并将DPM集成到对话情感识别（ERC）的推理中。在IEMOCAP数据集上的实验结果表明，DPM在处理长音频序列时显著提高了SLLM的情感识别能力，达到了领先水平。 

---
# Infinite Video Understanding 

**Title (ZH)**: 无限视频理解 

**Authors**: Dell Zhang, Xiangyu Chen, Jixiang Luo, Mengxi Jia, Changzhi Sun, Ruilong Ren, Jingren Liu, Hao Sun, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.09068)  

**Abstract**: The rapid advancements in Large Language Models (LLMs) and their multimodal extensions (MLLMs) have ushered in remarkable progress in video understanding. However, a fundamental challenge persists: effectively processing and comprehending video content that extends beyond minutes or hours. While recent efforts like Video-XL-2 have demonstrated novel architectural solutions for extreme efficiency, and advancements in positional encoding such as HoPE and VideoRoPE++ aim to improve spatio-temporal understanding over extensive contexts, current state-of-the-art models still encounter significant computational and memory constraints when faced with the sheer volume of visual tokens from lengthy sequences. Furthermore, maintaining temporal coherence, tracking complex events, and preserving fine-grained details over extended periods remain formidable hurdles, despite progress in agentic reasoning systems like Deep Video Discovery. This position paper posits that a logical, albeit ambitious, next frontier for multimedia research is Infinite Video Understanding -- the capability for models to continuously process, understand, and reason about video data of arbitrary, potentially never-ending duration. We argue that framing Infinite Video Understanding as a blue-sky research objective provides a vital north star for the multimedia, and the wider AI, research communities, driving innovation in areas such as streaming architectures, persistent memory mechanisms, hierarchical and adaptive representations, event-centric reasoning, and novel evaluation paradigms. Drawing inspiration from recent work on long/ultra-long video understanding and several closely related fields, we outline the core challenges and key research directions towards achieving this transformative capability. 

**Abstract (ZH)**: 大规模语言模型及其多模态扩展的迅速发展推动了视频理解的显著进步：无限视频理解 

---
# SetupBench: Assessing Software Engineering Agents' Ability to Bootstrap Development Environments 

**Title (ZH)**: SetupBench: 评估软件工程代理构建开发环境的能力 

**Authors**: Avi Arora, Jinu Jang, Roshanak Zilouchian Moghaddam  

**Link**: [PDF](https://arxiv.org/pdf/2507.09063)  

**Abstract**: Modern Large Language Model (LLM) agents promise end to end assistance with real-world software tasks, yet existing benchmarks evaluate LLM agents almost exclusively in pre-baked environments where every dependency is pre-installed. To fill this gap, we introduce SetupBench, a 93 instance benchmark that isolates the environment-bootstrap skill: starting from a bare Linux sandbox, an agent must install packages, resolve dependency conflicts, initialize databases, and configure background services. Our tasks span seven language ecosystems, five database engines, and multi-service orchestration scenarios, each accompanies by a natural language problem statement and a deterministic success command. Through evaluation of OpenHands, a state-of-the-art coding agent, we find low success rates across task categories, with particular challenges in repository setup (38.9-57.4%) and local database configuration (20.0-53.3%). Our analysis reveals systematic failure modes including incomplete development tooling installation, hallucinated task constraints, and non-persistent environment modifications that break agent-human collaboration workflows. We identify substantial inefficiencies in agent exploration strategies, with 38-89% of actions being unnecessary compared to optimal human behavior. These findings highlight gaps in current agents' practical environment-bootstrap capabilities. By targeting this critical yet under-evaluated capability, SetupBench provides a rigorous yard-stick for the next generation of software developer agents aiming to solve end to end real-wold tasks. 

**Abstract (ZH)**: SetupBench：隔离环境构建技能的93实例基准 

---
# Analysing Health Misinformation with Advanced Centrality Metrics in Online Social Networks 

**Title (ZH)**: 基于先进中心性指标分析在线社交网络中的健康 misinformation 

**Authors**: Mkululi Sikosana, Sean Maudsley-Barton, Oluwaseun Ajao  

**Link**: [PDF](https://arxiv.org/pdf/2507.09055)  

**Abstract**: The rapid spread of health misinformation on online social networks (OSNs) during global crises such as the COVID-19 pandemic poses challenges to public health, social stability, and institutional trust. Centrality metrics have long been pivotal in understanding the dynamics of information flow, particularly in the context of health misinformation. However, the increasing complexity and dynamism of online networks, especially during crises, highlight the limitations of these traditional approaches. This study introduces and compares three novel centrality metrics: dynamic influence centrality (DIC), health misinformation vulnerability centrality (MVC), and propagation centrality (PC). These metrics incorporate temporal dynamics, susceptibility, and multilayered network interactions. Using the FibVID dataset, we compared traditional and novel metrics to identify influential nodes, propagation pathways, and misinformation influencers. Traditional metrics identified 29 influential nodes, while the new metrics uncovered 24 unique nodes, resulting in 42 combined nodes, an increase of 44.83%. Baseline interventions reduced health misinformation by 50%, while incorporating the new metrics increased this to 62.5%, an improvement of 25%. To evaluate the broader applicability of the proposed metrics, we validated our framework on a second dataset, Monant Medical Misinformation, which covers a diverse range of health misinformation discussions beyond COVID-19. The results confirmed that the advanced metrics generalised successfully, identifying distinct influential actors not captured by traditional methods. In general, the findings suggest that a combination of traditional and novel centrality measures offers a more robust and generalisable framework for understanding and mitigating the spread of health misinformation in different online network contexts. 

**Abstract (ZH)**: 在线社交网络（OSNs）上健康 misinformation 的快速传播：全球危机如COVID-19 pandemic期间对公共健康、社会稳定和机构信任的挑战 

---
# ALIGN: Prompt-based Attribute Alignment for Reliable, Responsible, and Personalized LLM-based Decision-Making 

**Title (ZH)**: ALIGN：基于提示的属性对齐以实现可靠的、负责任的和个人化的LLM决策制定 

**Authors**: Bharadwaj Ravichandran, David Joy, Paul Elliott, Brian Hu, Jadie Adams, Christopher Funk, Emily Veenhuis, Anthony Hoogs, Arslan Basharat  

**Link**: [PDF](https://arxiv.org/pdf/2507.09037)  

**Abstract**: Large language models (LLMs) are increasingly being used as decision aids. However, users have diverse values and preferences that can affect their decision-making, which requires novel methods for LLM alignment and personalization. Existing LLM comparison tools largely focus on benchmarking tasks, such as knowledge-based question answering. In contrast, our proposed ALIGN system focuses on dynamic personalization of LLM-based decision-makers through prompt-based alignment to a set of fine-grained attributes. Key features of our system include robust configuration management, structured output generation with reasoning, and several algorithm implementations with swappable LLM backbones, enabling different types of analyses. Our user interface enables a qualitative, side-by-side comparison of LLMs and their alignment to various attributes, with a modular backend for easy algorithm integration. Additionally, we perform a quantitative analysis comparing alignment approaches in two different domains: demographic alignment for public opinion surveys and value alignment for medical triage decision-making. The entire ALIGN framework is open source and will enable new research on reliable, responsible, and personalized LLM-based decision-makers. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被用作决策辅助工具。然而，用户的价值观和偏好会影响其决策过程，因此需要新的方法来实现LLM的对齐和个人化。现有的LLM比较工具主要集中在基于基准的任务，如基于知识的问答。相比之下，我们提出的ALIGN系统专注于通过基于提示的对齐来实现LLM决策辅助的动态个性化，针对一组精细粒度的属性。系统的关键特征包括稳健的配置管理、具有推理的结构化输出生成，以及具有可更换LLM骨干的多种算法实现，支持不同类型的分析。我们的用户界面允许对LLM及其与各种属性的对齐进行定性、并排比较，并具有模块化后端以便于算法集成。此外，我们在两个不同的领域中进行了定量分析，比较了对齐方法：人口统计学对齐用于公共意见调查，价值观对齐用于医疗triage决策。整个ALIGN框架是开源的，将有助于开展关于可靠、负责任和个人化LLM决策辅助系统的新研究。 

---
# BrainLesion Suite: A Flexible and User-Friendly Framework for Modular Brain Lesion Image Analysis 

**Title (ZH)**: 脑部病灶套件：一个模块化脑部病灶图像分析的灵活易用框架 

**Authors**: Florian Kofler, Marcel Rosier, Mehdi Astaraki, Hendrik Möller, Ilhem Isra Mekki, Josef A. Buchner, Anton Schmick, Arianna Pfiffer, Eva Oswald, Lucas Zimmer, Ezequiel de la Rosa, Sarthak Pati, Julian Canisius, Arianna Piffer, Ujjwal Baid, Mahyar Valizadeh, Akis Linardos, Jan C. Peeken, Surprosanna Shit, Felix Steinbauer, Daniel Rueckert, Rolf Heckemann, Spyridon Bakas, Jan Kirschke, Constantin von See, Ivan Ezhov, Marie Piraud, Benedikt Wiestler, Bjoern Menze  

**Link**: [PDF](https://arxiv.org/pdf/2507.09036)  

**Abstract**: BrainLesion Suite is a versatile toolkit for building modular brain lesion image analysis pipelines in Python. Following Pythonic principles, BrainLesion Suite is designed to provide a 'brainless' development experience, minimizing cognitive effort and streamlining the creation of complex workflows for clinical and scientific practice. At its core is an adaptable preprocessing module that performs co-registration, atlas registration, and optional skull-stripping and defacing on arbitrary multi-modal input images. BrainLesion Suite leverages algorithms from the BraTS challenge to synthesize missing modalities, inpaint lesions, and generate pathology-specific tumor segmentations. BrainLesion Suite also enables quantifying segmentation model performance, with tools such as panoptica to compute lesion-wise metrics. Although BrainLesion Suite was originally developed for image analysis pipelines of brain lesions such as glioma, metastasis, and multiple sclerosis, it can be adapted for other biomedical image analysis applications. The individual BrainLesion Suite packages and tutorials are accessible on GitHub. 

**Abstract (ZH)**: BrainLesion Suite是用于构建Python中可模块化脑病变图像分析流水线的多功能工具包。遵循Pythonic原则，BrainLesion Suite旨在提供“无脑化”的开发体验，减少认知努力并简化复杂工作流的创建，以满足临床和科学实践的需求。其核心是一个可适应的预处理模块，该模块执行配准、解剖图注册，并可选地进行去头骨和去标识化处理任意多模态输入图像。BrainLesion Suite利用BraTS挑战中的算法来合成缺失的模态、填充病变并生成病理特异性肿瘤分割。BrainLesion Suite还允许通过工具如panoptica等量化分割模型性能，以计算病变级别的指标。尽管BrainLesion Suite最初是为如胶质瘤、转移和多发性硬化等脑病变的图像分析流水线开发的，但它可以适应其他生物医学图像分析应用。BrainLesion Suite的各个包和教程可在GitHub上访问。 

---
# Model Parallelism With Subnetwork Data Parallelism 

**Title (ZH)**: 子网络数据并行的模型并行ism 

**Authors**: Vaibhav Singh, Zafir Khalid, Edouard Oyallon, Eugene Belilovsky  

**Link**: [PDF](https://arxiv.org/pdf/2507.09029)  

**Abstract**: Distributed pre-training of large models at scale often imposes heavy memory demands on individual nodes and incurs significant intra-node communication costs. We propose a novel alternative approach that reduces the memory requirements by training small, structured subnetworks of the model on separate workers. Unlike pipelining, our method avoids inter-node activation communication and maintains bandwidth requirements that are comparable to or lower than standard data parallel communication schemes based on all-reduce. We evaluate two subnetwork construction strategies guided by the principle of ensuring uniform representation of each parameter across the distributed training setup. Our results show that the stochastic block dropping technique consistently outperforms the width-wise subnetwork construction previously explored in federated learning. We empirically attribute this superior performance to stronger gradient alignment in subnetworks that retain blocks having skip connections. Preliminary experiments highlight the promise of our approach, achieving a 20-40% reduction in memory usage without any loss in performance. 

**Abstract (ZH)**: 大规模模型的分布式预训练往往对单个节点的内存需求很高，并导致显著的节点内通信成本。我们提出一种新型替代方法，通过在分离的工作者上训练模型的小型结构化子网络来降低内存要求。与流水线方法不同，我们的方法避免了节点间激活通信，并保持了与标准基于all-reduce的数据并行通信方案相当或更低的带宽要求。我们通过确保每个参数在网络分布式训练设置中的均匀表示指导构建两种子网络结构策略。结果表明，随机块删除技术在联邦学习中之前探索的宽度方向子网络构建策略上表现更优。我们从实验中发现，这种优越性能归因于保留具有跳跃连接的块的子网络中梯度对齐更强。初步实验展示了我们方法的潜力，能够在不损失性能的情况下将内存使用量减少20-40%。 

---
# From Classical Machine Learning to Emerging Foundation Models: Review on Multimodal Data Integration for Cancer Research 

**Title (ZH)**: 从经典机器学习到新兴基础模型：多模态数据集成在癌症研究中的综述 

**Authors**: Amgad Muneer, Muhammad Waqas, Maliazurina B Saad, Eman Showkatian, Rukhmini Bandyopadhyay, Hui Xu, Wentao Li, Joe Y Chang, Zhongxing Liao, Cara Haymaker, Luisa Solis Soto, Carol C Wu, Natalie I Vokes, Xiuning Le, Lauren A Byers, Don L Gibbons, John V Heymach, Jianjun Zhang, Jia Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.09028)  

**Abstract**: Cancer research is increasingly driven by the integration of diverse data modalities, spanning from genomics and proteomics to imaging and clinical factors. However, extracting actionable insights from these vast and heterogeneous datasets remains a key challenge. The rise of foundation models (FMs) -- large deep-learning models pretrained on extensive amounts of data serving as a backbone for a wide range of downstream tasks -- offers new avenues for discovering biomarkers, improving diagnosis, and personalizing treatment. This paper presents a comprehensive review of widely adopted integration strategies of multimodal data to assist advance the computational approaches for data-driven discoveries in oncology. We examine emerging trends in machine learning (ML) and deep learning (DL), including methodological frameworks, validation protocols, and open-source resources targeting cancer subtype classification, biomarker discovery, treatment guidance, and outcome prediction. This study also comprehensively covers the shift from traditional ML to FMs for multimodal integration. We present a holistic view of recent FMs advancements and challenges faced during the integration of multi-omics with advanced imaging data. We identify the state-of-the-art FMs, publicly available multi-modal repositories, and advanced tools and methods for data integration. We argue that current state-of-the-art integrative methods provide the essential groundwork for developing the next generation of large-scale, pre-trained models poised to further revolutionize oncology. To the best of our knowledge, this is the first review to systematically map the transition from conventional ML to advanced FM for multimodal data integration in oncology, while also framing these developments as foundational for the forthcoming era of large-scale AI models in cancer research. 

**Abstract (ZH)**: 癌症研究 increasingly driven by the integration of diverse data modalities, spanning from genomics and proteomics to imaging and clinical factors. However, extracting actionable insights from these vast and heterogeneous datasets remains a key challenge. The rise of foundation models (FMs) — large deep-learning models pretrained on extensive amounts of data serving as a backbone for a wide range of downstream tasks — offers new avenues for discovering biomarkers, improving diagnosis, and personalizing treatment. This paper presents a comprehensive review of widely adopted integration strategies of multimodal data to assist in advancing computational approaches for data-driven discoveries in oncology. We examine emerging trends in machine learning (ML) and deep learning (DL), including methodological frameworks, validation protocols, and open-source resources targeting cancer subtype classification, biomarker discovery, treatment guidance, and outcome prediction. This study also comprehensively covers the shift from traditional ML to FMs for multimodal integration. We present a holistic view of recent FM advancements and challenges faced during the integration of multi-omics with advanced imaging data. We identify the state-of-the-art FMs, publicly available multi-modal repositories, and advanced tools and methods for data integration. We argue that current state-of-the-art integrative methods provide the essential groundwork for developing the next generation of large-scale, pre-trained models poised to further revolutionize oncology. To the best of our knowledge, this is the first review to systematically map the transition from conventional ML to advanced FMs for multimodal data integration in oncology, while also framing these developments as foundational for the forthcoming era of large-scale AI models in cancer research. 

---
# Accelerating Drug Discovery Through Agentic AI: A Multi-Agent Approach to Laboratory Automation in the DMTA Cycle 

**Title (ZH)**: 通过自主智能加速药物发现：DMTA循环中多智能体实验室自动化方法 

**Authors**: Yao Fehlis, Charles Crain, Aidan Jensen, Michael Watson, James Juhasz, Paul Mandel, Betty Liu, Shawn Mahon, Daren Wilson, Nick Lynch-Jonely, Ben Leedom, David Fuller  

**Link**: [PDF](https://arxiv.org/pdf/2507.09023)  

**Abstract**: The pharmaceutical industry faces unprecedented challenges in drug discovery, with traditional approaches struggling to meet modern therapeutic development demands. This paper introduces a novel AI framework, Tippy, that transforms laboratory automation through specialized AI agents operating within the Design-Make-Test-Analyze (DMTA) cycle. Our multi-agent system employs five specialized agents - Supervisor, Molecule, Lab, Analysis, and Report, with Safety Guardrail oversight - each designed to excel in specific phases of the drug discovery pipeline. Tippy represents the first production-ready implementation of specialized AI agents for automating the DMTA cycle, providing a concrete example of how AI can transform laboratory workflows. By leveraging autonomous AI agents that reason, plan, and collaborate, we demonstrate how Tippy accelerates DMTA cycles while maintaining scientific rigor essential for pharmaceutical research. The system shows significant improvements in workflow efficiency, decision-making speed, and cross-disciplinary coordination, offering a new paradigm for AI-assisted drug discovery. 

**Abstract (ZH)**: 制药行业在药物发现方面面临着前所未有的挑战，传统方法难以满足现代治疗开发的需求。本文介绍了一种新型AI框架Tippy，通过专门的AI代理在设计-制备-测试-分析（DMTA）循环中运行，从而实现实验室自动化。我们的多代理系统包括五个专门的代理——督导员、分子、实验室、分析和报告，并由安全护栏监管，每个代理都旨在在药物发现管道的特定阶段表现出色。Tippy代表了专门为自动化DMTA循环而开发的第一个生产就绪型专门AI代理的实现，提供了AI如何转型实验室工作流程的典型案例。通过利用能够推理、规划和协作的自主AI代理，我们展示了Tippy如何在保持对于制药研究至关重要的科学严谨性的同时加速DMTA循环。该系统在工作流程效率、决策速度和跨学科协调方面显示出显著改进，提供了一种新的AI辅助药物发现范式。 

---
# On Evaluating Performance of LLM Inference Serving Systems 

**Title (ZH)**: 评估LLM推理服务系统的性能 

**Authors**: Amey Agrawal, Nitin Kedia, Anmol Agarwal, Jayashree Mohan, Nipun Kwatra, Souvik Kundu, Ramachandran Ramjee, Alexey Tumanov  

**Link**: [PDF](https://arxiv.org/pdf/2507.09019)  

**Abstract**: The rapid evolution of Large Language Model (LLM) inference systems has yielded significant efficiency improvements. However, our systematic analysis reveals that current evaluation methodologies frequently exhibit fundamental flaws, often manifesting as common evaluation anti-patterns that obscure true performance characteristics and impede scientific progress. Through a comprehensive examination of recent systems, we identify recurring anti-patterns across three key dimensions: Baseline Fairness, Evaluation Setup, and Metric Design. These anti-patterns are uniquely problematic for LLM inference due to its dual-phase nature combining distinct prefill and decode operations, its handling of highly heterogeneous workloads, and its strict temporal requirements for interactive use. We demonstrate how common anti-patterns -- such as inadequate baseline comparisons that conflate engineering effort with algorithmic novelty, workload selections that fail to represent production scenarios, and metric normalizations that hide substantial performance variability like generation stalls-lead to misleading conclusions. To address these challenges, we provide a comprehensive checklist derived from our analysis, establishing a framework for recognizing and avoiding these anti-patterns in favor of robust LLM inference evaluation. To demonstrate the practical application of our framework, we present a case study analyzing speculative decoding, a technique whose bursty, non-uniform token generation is easily misinterpreted when evaluated using approaches characteristic of these anti-patterns. Our work establishes a rigorous foundation for evaluation methodology, enabling meaningful comparisons, ensuring reproducible results, and ultimately accelerating genuine progress in LLM inference systems by moving beyond common anti-patterns to align evaluation with real-world requirements. 

**Abstract (ZH)**: 大型语言模型（LLM）推理系统快速进化带来了显著的效率提升。然而，我们系统的分析揭示出当前的评估方法经常存在根本性的缺陷，通常表现为常见的评估反模式，这些反模式模糊了真实性能特征，阻碍了科学进步。通过对近期系统的全面考察，我们确定了跨越三个关键维度的反复出现的反模式：基准公平性、评估设置和指标设计。这些反模式由于LLM推理同时包含预填充和解码两个相异阶段、处理高度异质的工作负载以及对于交互使用有严格的时效要求，而在LLM推理中尤为突出。我们展示了常见的反模式如何导致误导性的结论，例如不充分的基准比较混杂了工程努力与算法新颖性，任务选择无法代表生产场景，以及隐藏了诸如生成停滞等显著性能变异性指标归一化。为解决这些挑战，我们根据分析提供了一个全面的检查清单，建立了一种框架来识别和避免这些反模式，以实现稳健的LLM推理评估。为了展示我们框架的实际应用，我们分析了投机解码，这是一种其突发性、非均匀性令牌生成容易在这些反模式特征的评估方法中被误解的技术。我们的工作为评估方法奠定了严格的基石，使性能比较有意义，确保结果可复制，并最终通过超越常见反模式，使评估与现实需求保持一致，从而加速LLM推理系统的真正进步。 

---
# Hybrid Systolic Array Accelerator with Optimized Dataflow for Edge Large Language Model Inference 

**Title (ZH)**: 优化数据流的边缘大规模语言模型推理混合 systolic 阵列加速器 

**Authors**: Chun-Ting Chen, HanGyeol Mun, Jian Meng, Mohamed S. Abdelfattah, Jae-sun Seo  

**Link**: [PDF](https://arxiv.org/pdf/2507.09010)  

**Abstract**: Edge inference for large language models (LLM) offers secure, low-latency, and cost-effective inference solutions. We emphasize that an edge accelerator should achieve high area efficiency and minimize external memory access (EMA) during the memory-bound decode stage, while maintaining high energy efficiency during the compute intensive prefill stage. This paper proposes an edge LLM inference accelerator featuring a hybrid systolic array (HSA) architecture that optimizes inference efficiency in both stages. To further reduce EMA, we adopt MXINT4 weight quantization and propose an optimized dataflow tailored for HSA, ensuring negligible dequantization overhead and achieving 100% hardware utilization with minimal accuracy loss under edge DRAM bandwidth constraints. For non-linear operations, we incorporate optimized root mean square normalization (RMSNorm) and rotary position embedding (RoPE) units, reducing their latency, area, and memory access overhead while enabling end-to-end inference on our accelerator. Our solution achieves 247/117 (token/s/mm2) while running a 1.3B LLM on long-input/long-output scenarios, providing >2.45x/13.5x improvement over existing approaches, while maintaining superior energy efficiency in token generation. 

**Abstract (ZH)**: 边缘设备上大型语言模型的推理：面向大型语言模型的边缘推理加速器在确保安全、低延迟和低成本的同时，强调边缘加速器在内存限制的解码阶段应实现高面积效率并尽量减少外部内存访问，而在计算密集型的填充阶段应保持高能量效率。本文提出了一种结合 systolic 阵列（HSA）架构的边缘大型语言模型推理加速器，以优化两个阶段的推理效率。为进一步减少外部内存访问，我们采用 MXINT4 权重量化，并提出了一种针对 HSA 的优化数据流，确保去量化开销可忽略不计，并在边缘 DRAM 带宽约束下实现 100% 的硬件利用率，同时最小化准确度损失。对于非线性操作，我们引入了优化的根均方归一化（RMSNorm）和旋转位置嵌入（RoPE）单元，降低了它们的延迟、面积和内存访问开销，从而在加速器上实现端到端推理。我们的解决方案在长输入/长输出场景下以每平方毫米每标记 247/117 的性能运行 1.3B 参数的大型语言模型，相对于现有方法提供了超过 2.45 倍/13.5 倍的性能提升，同时保持了在标记生成方面的卓越能效。 

---
# Multimodal Cardiovascular Risk Profiling Using Self-Supervised Learning of Polysomnography 

**Title (ZH)**: 基于自我监督学习的多模态心血管风险评估 

**Authors**: Zhengxiao He, Huayu Li, Geng Yuan, William D.S. Killgore, Stuart F. Quan, Chen X. Chen, Ao Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.09009)  

**Abstract**: Methods: We developed a self-supervised deep learning model that extracts meaningful patterns from multi-modal signals (Electroencephalography (EEG), Electrocardiography (ECG), and respiratory signals). The model was trained on data from 4,398 participants. Projection scores were derived by contrasting embeddings from individuals with and without CVD outcomes. External validation was conducted in an independent cohort with 1,093 participants. The source code is available on this https URL. Results: The projection scores revealed distinct and clinically meaningful patterns across modalities. ECG-derived features were predictive of both prevalent and incident cardiac conditions, particularly CVD mortality. EEG-derived features were predictive of incident hypertension and CVD mortality. Respiratory signals added complementary predictive value. Combining these projection scores with the Framingham Risk Score consistently improved predictive performance, achieving area under the curve values ranging from 0.607 to 0.965 across different outcomes. Findings were robustly replicated and validated in the external testing cohort. Conclusion: Our findings demonstrate that the proposed framework can generate individualized CVD risk scores directly from PSG data. The resulting projection scores have the potential to be integrated into clinical practice, enhancing risk assessment and supporting personalized care. 

**Abstract (ZH)**: 方法：我们开发了一种自监督深度学习模型，从多模态信号（脑电图（EEG）、心电图（ECG）和呼吸信号）中提取有意义的模式。该模型在4,398名参与者的数据上进行训练。通过对比患有和未患有心血管疾病（CVD）结局个体的嵌入表示，得到了投影分数。外部验证在独立的1,093名参与者的队列中进行。源代码可在以下网址获取：this https URL。结果：投影分数揭示了各模态中的独特且具有临床意义的模式。心电图衍生的特征预测了心力衰竭和心血管疾病（CVD）死亡等既往和新发心脏状况，尤其是心血管疾病死亡。脑电图衍生的特征预测了新发高血压和心血管疾病死亡。呼吸信号提供了补充的预测价值。将这些投影分数与弗雷明汉风险评分结合使用，在不同结局上的一致预测性能达到了从0.607到0.965的曲线下面积（AUC）值。该发现在外测试队列中表现出高度的稳健性和验证性。结论：我们的发现表明，所提出的方法可以直接从多导睡眠图（PSG）数据中生成个性化的CVD风险评分。产生的投影分数有潜力融入临床实践，提高风险评估并支持个性化治疗。 

---
# Learning Diffusion Models with Flexible Representation Guidance 

**Title (ZH)**: 学习具有灵活表示指导的扩散模型 

**Authors**: Chenyu Wang, Cai Zhou, Sharut Gupta, Zongyu Lin, Stefanie Jegelka, Stephen Bates, Tommi Jaakkola  

**Link**: [PDF](https://arxiv.org/pdf/2507.08980)  

**Abstract**: Diffusion models can be improved with additional guidance towards more effective representations of input. Indeed, prior empirical work has already shown that aligning internal representations of the diffusion model with those of pre-trained models improves generation quality. In this paper, we present a systematic framework for incorporating representation guidance into diffusion models. We provide alternative decompositions of denoising models along with their associated training criteria, where the decompositions determine when and how the auxiliary representations are incorporated. Guided by our theoretical insights, we introduce two new strategies for enhancing representation alignment in diffusion models. First, we pair examples with target representations either derived from themselves or arisen from different synthetic modalities, and subsequently learn a joint model over the multimodal pairs. Second, we design an optimal training curriculum that balances representation learning and data generation. Our experiments across image, protein sequence, and molecule generation tasks demonstrate superior performance as well as accelerated training. In particular, on the class-conditional ImageNet $256\times 256$ benchmark, our guidance results in $23.3$ times faster training than the original SiT-XL as well as four times speedup over the state-of-the-art method REPA. The code is available at this https URL. 

**Abstract (ZH)**: 使用额外的指导提高扩散模型的表示能力：一个系统框架及其应用 

---
# Simulation as Supervision: Mechanistic Pretraining for Scientific Discovery 

**Title (ZH)**: 将模拟作为监督：机理预训练促进科学研究 

**Authors**: Carson Dudley, Reiden Magdaleno, Christopher Harding, Marisa Eisenberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.08977)  

**Abstract**: Scientific modeling faces a core limitation: mechanistic models offer interpretability but collapse under real-world complexity, while machine learning models are flexible but require large labeled datasets, cannot infer unobservable quantities, and operate as black boxes. We introduce Simulation-Grounded Neural Networks (SGNNs), a general framework that uses mechanistic simulations as training data for neural networks. SGNNs are pretrained on synthetic corpora spanning diverse model structures, parameter regimes, stochasticity, and observational artifacts. We evaluated SGNNs across scientific disciplines and modeling tasks, and found that SGNNs achieved state-of-the-art results across settings: for prediction tasks, they nearly tripled COVID-19 forecasting skill versus CDC baselines, reduced chemical yield prediction error by one third, and maintained accuracy in ecological forecasting where task specific models failed. For inference tasks, SGNNs also accurately classified the source of information spread in simulated social networks and enabled supervised learning for unobservable targets, such as estimating COVID-19 transmissibility more accurately than traditional methods even in early outbreaks. Finally, SGNNs enable back-to-simulation attribution, a new form of mechanistic interpretability. Given real world input, SGNNs retrieve simulations based on what the model has learned to see as most similar, revealing which underlying dynamics the model believes are active. This provides process-level insight -- what the model thinks is happening -- not just which features mattered. SGNNs unify scientific theory with deep learning flexibility and unlock a new modeling paradigm -- transforming simulations from rigid, post hoc tools into flexible sources of supervision, enabling robust, interpretable inference even when ground truth is missing. 

**Abstract (ZH)**: 基于仿真训练的神经网络：一种结合机理模型和机器学习优势的通用框架 

---
# Simulating Three-dimensional Turbulence with Physics-informed Neural Networks 

**Title (ZH)**: 用物理知情神经网络模拟三维湍流 

**Authors**: Sifan Wang, Shyam Sankaran, Panos Stinis, Paris Perdikaris  

**Link**: [PDF](https://arxiv.org/pdf/2507.08972)  

**Abstract**: Turbulent fluid flows are among the most computationally demanding problems in science, requiring enormous computational resources that become prohibitive at high flow speeds. Physics-informed neural networks (PINNs) represent a radically different approach that trains neural networks directly from physical equations rather than data, offering the potential for continuous, mesh-free solutions. Here we show that appropriately designed PINNs can successfully simulate fully turbulent flows in both two and three dimensions, directly learning solutions to the fundamental fluid equations without traditional computational grids or training data. Our approach combines several algorithmic innovations including adaptive network architectures, causal training, and advanced optimization methods to overcome the inherent challenges of learning chaotic dynamics. Through rigorous validation on challenging turbulence problems, we demonstrate that PINNs accurately reproduce key flow statistics including energy spectra, kinetic energy, enstrophy, and Reynolds stresses. Our results demonstrate that neural equation solvers can handle complex chaotic systems, opening new possibilities for continuous turbulence modeling that transcends traditional computational limitations. 

**Abstract (ZH)**: 湍流流动是科学中计算需求最大的问题之一，在高速流中所需的计算资源变得难以承受。物理导向神经网络（PINNs）代表了一种截然不同的方法，直接从物理方程而非数据训练神经网络，提供了连续、无网格式求解的潜力。我们展示了一种适当设计的PINNs能够成功模拟二维和三维完全湍流流动，直接学习基本流体方程的解，无需传统计算网格或训练数据。我们的方法结合了多种算法创新，包括自适应网络架构、因果训练和高级优化方法，以克服学习混沌动力学的固有挑战。通过在挑战性的湍流问题上进行严格的验证，我们证明PINNs能够准确再现关键流场统计，包括能量谱、动能、涡度和雷诺应力。我们的结果表明，神经方程求解器能够处理复杂的混沌系统，开启了超越传统计算限制的连续湍流建模的新可能性。 

---
# ToxBench: A Binding Affinity Prediction Benchmark with AB-FEP-Calculated Labels for Human Estrogen Receptor Alpha 

**Title (ZH)**: ToxBench: 一种基于AB-FEP计算标签的人类雌激素受体α结合亲和力预测基准 

**Authors**: Meng Liu, Karl Leswing, Simon K. S. Chu, Farhad Ramezanghorbani, Griffin Young, Gabriel Marques, Prerna Das, Anjali Panikar, Esther Jamir, Mohammed Sulaiman Shamsudeen, K. Shawn Watts, Ananya Sen, Hari Priya Devannagari, Edward B. Miller, Muyun Lihan, Howook Hwang, Janet Paulsen, Xin Yu, Kyle Gion, Timur Rvachov, Emine Kucukbenli, Saee Gopal Paliwal  

**Link**: [PDF](https://arxiv.org/pdf/2507.08966)  

**Abstract**: Protein-ligand binding affinity prediction is essential for drug discovery and toxicity assessment. While machine learning (ML) promises fast and accurate predictions, its progress is constrained by the availability of reliable data. In contrast, physics-based methods such as absolute binding free energy perturbation (AB-FEP) deliver high accuracy but are computationally prohibitive for high-throughput applications. To bridge this gap, we introduce ToxBench, the first large-scale AB-FEP dataset designed for ML development and focused on a single pharmaceutically critical target, Human Estrogen Receptor Alpha (ER$\alpha$). ToxBench contains 8,770 ER$\alpha$-ligand complex structures with binding free energies computed via AB-FEP with a subset validated against experimental affinities at 1.75 kcal/mol RMSE, along with non-overlapping ligand splits to assess model generalizability. Using ToxBench, we further benchmark state-of-the-art ML methods, and notably, our proposed DualBind model, which employs a dual-loss framework to effectively learn the binding energy function. The benchmark results demonstrate the superior performance of DualBind and the potential of ML to approximate AB-FEP at a fraction of the computational cost. 

**Abstract (ZH)**: 蛋白-配体结合亲和力预测对于药物发现和毒性评估至关重要。虽然机器学习（ML）承诺能够提供快速且准确的预测，但其进展受限于可靠数据的可用性。相比之下，基于物理的方法如绝对结合自由能突变（AB-FEP）能够提供高精度，但在高通量应用中计算成本过高。为弥合这一差距，我们提出了ToxBench，这是一个用于机器学习开发的大型AB-FEP数据集，专注于单一的药理学关键目标——人类雌激素受体α（ERα）。ToxBench包含8,770个ERα-配体复合物结构，并通过AB-FEP计算了结合自由能，其中部分结构与1.75 kcal/mol的均方根误差实验亲和力进行了验证，同时包含非重叠的配体分割以评估模型的泛化能力。利用ToxBench，我们进一步测试了最先进的机器学习方法，并提出了一种新的DualBind模型，该模型采用双损失框架有效学习结合能量函数。基准测试结果表明，DualBind模型的性能优于现有方法，并展示了机器学习在计算成本极低的情况下逼近AB-FEP的潜力。 

---
# Theory-Informed Improvements to Classifier-Free Guidance for Discrete Diffusion Models 

**Title (ZH)**: 基于理论指导的离散扩散模型无分类器引导改进 

**Authors**: Kevin Rojas, Ye He, Chieh-Hsin Lai, Yuta Takida, Yuki Mitsufuji, Molei Tao  

**Link**: [PDF](https://arxiv.org/pdf/2507.08965)  

**Abstract**: Classifier-Free Guidance (CFG) is a widely used technique for conditional generation and improving sample quality in continuous diffusion models, and recent works have extended it to discrete diffusion. This paper theoretically analyzes CFG in the context of masked discrete diffusion, focusing on the role of guidance schedules. Our analysis shows that high guidance early in sampling (when inputs are heavily masked) harms generation quality, while late-stage guidance has a larger effect. These findings provide a theoretical explanation for empirical observations in recent studies on guidance schedules. The analysis also reveals an imperfection of the current CFG implementations. These implementations can unintentionally cause imbalanced transitions, such as unmasking too rapidly during the early stages of generation, which degrades the quality of the resulting samples. To address this, we draw insight from the analysis and propose a novel classifier-free guidance mechanism empirically applicable to any discrete diffusion. Intuitively, our method smoothens the transport between the data distribution and the initial (masked/uniform) distribution, which results in improved sample quality. Remarkably, our method is achievable via a simple one-line code change. The efficacy of our method is empirically demonstrated with experiments on ImageNet (masked discrete diffusion) and QM9 (uniform discrete diffusion). 

**Abstract (ZH)**: Classifier-Free Guidance (CFG)在掩码离散扩散中的理论分析：指导计划的角色 

---
# How to Train a Leader: Hierarchical Reasoning in Multi-Agent LLMs 

**Title (ZH)**: 如何培养领导者：多agent大语言模型中的层级推理 

**Authors**: Andrew Estornell, Jean-Francois Ton, Muhammad Faaiz Taufiq, Hang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.08960)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance on a wide range of complex reasoning tasks, yet further gains are often possible by leveraging the complementary strengths of multiple models. While multi-agent frameworks can improve solution quality by leveraging multiple LLMs, existing methods are often computationally expensive, both at training and inference time. In this work, we introduce a hierarchical multi-agent framework that addresses these challenges by training only a single leader LLM to coordinate a team of untrained peer agents. To this end, we propose Multi-agent guided Leader Policy \textbf{O}ptimization (MLPO), a novel approach which trains the leader to evaluate and synthesize agent responses without auxiliary value networks or explicit agent feedback. Leaders trained with MLPO exhibit improved performance not only when interacting with the agent team at inference time, but also enjoy improved performance when deployed in single-agent settings without the team. Empirical results on Big-Bench Hard (BBH), MATH, and MMLU demonstrate that our framework achieves substantial performance improvements over both single-agent and multi-agent baselines. Our results highlight the effectiveness and efficiency of training a single, flexible leader for collaborative reasoning in multi-agent LLM systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在一系列复杂的推理任务中展现了强大的性能，通过利用多模型的互补优势，仍有可能取得进一步的改进。尽管多智能体框架可以利用多个LLM来提高解决方案质量，但现有方法往往在训练和推理阶段都比较耗费计算资源。在本工作中，我们提出了一种分层多智能体框架，通过仅训练一个领导者LLM来协调一组未训练的同行代理，以此来应对这些挑战。为此，我们提出了多智能体引导领导者策略优化（MLPO）的新方法，该方法训练领导者评估和综合代理响应，而无需辅助价值网络或显式的代理反馈。使用MLPO训练的领导者不仅在与代理团队交互时表现出改进的性能，在没有团队支持的情况下部署为单智能体设置时也表现出改进的性能。在Big-Bench Hard (BBH)、MATH和MMLU上的实验证明，我们的框架在单智能体和多智能体基线方法上均实现了显著的性能提升。我们的结果突显了在多智能体LLM系统中训练单个灵活领导者以进行协作推理的有效性和效率。 

---
# Bridging Literature and the Universe Via A Multi-Agent Large Language Model System 

**Title (ZH)**: 通过多Agent大型语言模型系统连接文学与宇宙 

**Authors**: Xiaowen Zhang, Zhenyu Bi, Xuan Wang, Tiziana Di Matteo, Rupert A.C. Croft  

**Link**: [PDF](https://arxiv.org/pdf/2507.08958)  

**Abstract**: As cosmological simulations and their associated software become increasingly complex, physicists face the challenge of searching through vast amounts of literature and user manuals to extract simulation parameters from dense academic papers, each using different models and formats. Translating these parameters into executable scripts remains a time-consuming and error-prone process. To improve efficiency in physics research and accelerate the cosmological simulation process, we introduce SimAgents, a multi-agent system designed to automate both parameter configuration from the literature and preliminary analysis for cosmology research. SimAgents is powered by specialized LLM agents capable of physics reasoning, simulation software validation, and tool execution. These agents collaborate through structured communication, ensuring that extracted parameters are physically meaningful, internally consistent, and software-compliant. We also construct a cosmological parameter extraction evaluation dataset by collecting over 40 simulations in published papers from Arxiv and leading journals that cover diverse simulation types. Experiments on the dataset demonstrate a strong performance of SimAgents, highlighting its effectiveness and potential to accelerate scientific research for physicists. Our demonstration video is available at: this https URL. The complete system and dataset are publicly available at this https URL. 

**Abstract (ZH)**: 随着宇宙学模拟及其相关软件变得日益复杂，物理学家面临着从大量的文献和用户手册中提取使用不同模型和格式的模拟参数的挑战。将这些参数转换为可执行脚本仍是一个耗时且容易出错的过程。为了提高物理学研究的效率并加速宇宙学模拟过程，我们介绍了Sim_agents，一个旨在自动化从文献中提取模拟参数和初步分析的多代理系统。Sim_agents依赖于具备物理推理、模拟软件验证和工具执行能力的专业LLM代理。这些代理通过结构化的通信协作，确保提取的参数具有物理意义、内部一致且符合软件要求。我们还通过收集来自arXiv和顶级期刊的超过40个模拟，构建了一个宇宙学参数提取评估数据集，涵盖了多种模拟类型。在数据集上的实验结果显示了Sim_agents的强大性能，突显了其在加速物理学家的科学研究方面的效果和潜力。我们的演示视频可在以下链接获取：this https URL。完整的系统和数据集可在此链接访问：this https URL。 

---
# GraphRunner: A Multi-Stage Framework for Efficient and Accurate Graph-Based Retrieval 

**Title (ZH)**: GraphRunner：一种高效准确的图基检索多阶段框架 

**Authors**: Savini Kashmira, Jayanaka L. Dantanarayana, Krisztián Flautner, Lingjia Tang, Jason Mars  

**Link**: [PDF](https://arxiv.org/pdf/2507.08945)  

**Abstract**: Conventional Retrieval Augmented Generation (RAG) approaches are common in text-based applications. However, they struggle with structured, interconnected datasets like knowledge graphs, where understanding underlying relationships is crucial for accurate retrieval. A common direction in graph-based retrieval employs iterative, rule-based traversal guided by Large Language Models (LLMs). Such existing iterative methods typically combine reasoning with single hop traversal at each step, making them vulnerable to LLM reasoning errors and hallucinations that ultimately hinder the retrieval of relevant information.
To address these limitations, we propose GraphRunner, a novel graph-based retrieval framework that operates in three distinct stages: planning, verification, and execution. This introduces high-level traversal actions that enable multi-hop exploration in a single step. It also generates a holistic traversal plan, which is verified against the graph structure and pre-defined traversal actions, reducing reasoning errors and detecting hallucinations before execution. GraphRunner significantly reduces LLM reasoning errors and detects hallucinations through validation. Our evaluation using the GRBench dataset shows that GraphRunner consistently outperforms existing approaches, achieving 10-50% performance improvements over the strongest baseline while reducing inference cost by 3.0-12.9x and response generation time by 2.5-7.1x, making it significantly more robust and efficient for graph-based retrieval tasks. 

**Abstract (ZH)**: 基于图的检索增强生成（RAG）方法在文本应用中很常见。然而，它们在处理知识图等结构化且相互连接的数据集时遇到困难，因为在这些数据集中理解潜在关系对于准确检索至关重要。基于图的检索中的一种常见方向是通过大型语言模型（LLMs）引导的迭代、规则导向的遍历。现有的迭代方法通常在每一步结合一次跳跃遍历和推理，这使它们容易受到LLM推理错误和幻觉的影响，从而阻碍了相关信息的检索。

为了解决这些问题，我们提出了GraphRunner，这是一种新颖的基于图的检索框架，分为规划、验证和执行三个阶段。这引入了高层次的遍历动作，允许在单步中进行多跳跃探索。它还生成了一个整体的遍历计划，该计划可以根据图结构和预定义的遍历动作进行验证，从而减少推理错误并检测出执行前的幻觉。GraphRunner通过验证显著减少了LLM的推理错误并检测出了幻觉。使用GRBench数据集的评估表明，GraphRunner在性能上始终优于现有方法，对比最强基准方法在性能上提高了10-50%，同时将推理成本降低了3.0-12.9倍，响应生成时间减少了2.5-7.1倍，使它在基于图的检索任务中变得更加稳健和高效。 

---
# Optimizing Sequential Multi-Step Tasks with Parallel LLM Agents 

**Title (ZH)**: 使用并行大语言模型代理优化顺序多步任务 

**Authors**: Enhao Zhang, Erkang Zhu, Gagan Bansal, Adam Fourney, Hussein Mozannar, Jack Gerrits  

**Link**: [PDF](https://arxiv.org/pdf/2507.08944)  

**Abstract**: Large language model (LLM)-based multi-agent systems have demonstrated remarkable promise for tackling complex tasks by breaking them down into subtasks that are iteratively planned, executed, observed, and refined. Despite their effectiveness, these systems often incur high latency because real-world problems frequently demand multiple iterative cycles of reasoning steps. To address this challenge, we propose M1-Parallel, a framework that concurrently runs multiple multi-agent teams in parallel to uncover distinct solution paths. By leveraging an event-driven communication model with asynchronous messaging, M1-Parallel efficiently capitalizes on the inherent diversity of valid plans to either reduce end-to-end latency or boost task completion rates. Our experiments on complex tasks show that M1-Parallel with early termination achieves up to $2.2\times$ speedup while preserving accuracy, and that M1-Parallel with aggregation yields higher task completion rates. We further investigate strategies aimed at encouraging diverse execution plans but observe no additional performance gains over repeated sampling. Overall, these findings underscore the potential of parallel plan execution for optimizing multi-agent systems for real-world, high-complexity reasoning tasks. 

**Abstract (ZH)**: 基于大型语言模型的多-agent系统通过将复杂任务分解为迭代规划、执行、观察和优化的子任务，展现了显著的潜力。尽管这些系统有效，但它们常常因现实世界问题需要多次迭代推理步骤而产生高延迟。为解决这一挑战，我们提出了M1-Parallel框架，该框架通过并行运行多个多-agent团队来揭示不同的解决方案路径。通过利用基于事件驱动的通信模型和异步消息传递，M1-Parallel高效地利用了有效计划的固有多样性，以减少端到端延迟或提高任务完成率。在复杂任务上的实验结果显示，M1-Parallel结合早期终止可实现高达2.2倍的速度提升，同时保持准确性；而M1-Parallel结合聚合则能提高任务完成率。我们进一步探讨了鼓励多样化执行计划的策略，但未观察到与重复采样相比的额外性能增益。总体而言，这些发现突显了并行计划执行对于优化多-agent系统以应对现实世界、高复杂度推理任务的潜在价值。 

---
# From KMMLU-Redux to KMMLU-Pro: A Professional Korean Benchmark Suite for LLM Evaluation 

**Title (ZH)**: 从KMMLU-Redux到KMMLU-Pro：一个专业韩语基准套件用于大模型评估 

**Authors**: Seokhee Hong, Sunkyoung Kim, Guijin Son, Soyeon Kim, Yeonjung Hong, Jinsik Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.08924)  

**Abstract**: The development of Large Language Models (LLMs) requires robust benchmarks that encompass not only academic domains but also industrial fields to effectively evaluate their applicability in real-world scenarios. In this paper, we introduce two Korean expert-level benchmarks. KMMLU-Redux, reconstructed from the existing KMMLU, consists of questions from the Korean National Technical Qualification exams, with critical errors removed to enhance reliability. KMMLU-Pro is based on Korean National Professional Licensure exams to reflect professional knowledge in Korea. Our experiments demonstrate that these benchmarks comprehensively represent industrial knowledge in Korea. We release our dataset publicly available. 

**Abstract (ZH)**: 大型语言模型（LLMs）的发展需要涵盖学术领域和工业领域的稳健基准，以有效地评估其在实际场景中的适用性。本文介绍了两个韩语专家级基准：KMMLU-Redux是从现有KMMLU重构而来，包含韩国国家技术资格考试的问题，并去除了关键错误以提高可靠性；KMMLU-Pro基于韩国国家专业执照考试，反映韩国的专业知识。我们的实验表明，这些基准全面代表了韩国的工业知识。我们公开发布了我们的数据集。 

---
# AMix-1: A Pathway to Test-Time Scalable Protein Foundation Model 

**Title (ZH)**: AMix-1：面向测试时可扩展的蛋白质基础模型的一种途径 

**Authors**: Changze Lv, Jiang Zhou, Siyu Long, Lihao Wang, Jiangtao Feng, Dongyu Xue, Yu Pei, Hao Wang, Zherui Zhang, Yuchen Cai, Zhiqiang Gao, Ziyuan Ma, Jiakai Hu, Chaochen Gao, Jingjing Gong, Yuxuan Song, Shuyi Zhang, Xiaoqing Zheng, Deyi Xiong, Lei Bai, Ya-Qin Zhang, Wei-Ying Ma, Bowen Zhou, Hao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.08920)  

**Abstract**: We introduce AMix-1, a powerful protein foundation model built on Bayesian Flow Networks and empowered by a systematic training methodology, encompassing pretraining scaling laws, emergent capability analysis, in-context learning mechanism, and test-time scaling algorithm. To guarantee robust scalability, we establish a predictive scaling law and reveal the progressive emergence of structural understanding via loss perspective, culminating in a strong 1.7-billion model. Building on this foundation, we devise a multiple sequence alignment (MSA)-based in-context learning strategy to unify protein design into a general framework, where AMix-1 recognizes deep evolutionary signals among MSAs and consistently generates structurally and functionally coherent proteins. This framework enables the successful design of a dramatically improved AmeR variant with an up to $50\times$ activity increase over its wild type. Pushing the boundaries of protein engineering, we further empower AMix-1 with an evolutionary test-time scaling algorithm for in silico directed evolution that delivers substantial, scalable performance gains as verification budgets are intensified, laying the groundwork for next-generation lab-in-the-loop protein design. 

**Abstract (ZH)**: AMix-1：基于贝叶斯流网络的强大蛋白质基础模型及其系统训练方法和多重序列比对指导的上下文学习策略 

---
# Fair-FLIP: Fair Deepfake Detection with Fairness-Oriented Final Layer Input Prioritising 

**Title (ZH)**: Fair-FLIP：面向公平性的最终层输入优先级 deepfake 检测方法 

**Authors**: Tomasz Szandala, Fatima Ezzeddine, Natalia Rusin, Silvia Giordano, Omran Ayoub  

**Link**: [PDF](https://arxiv.org/pdf/2507.08912)  

**Abstract**: Artificial Intelligence-generated content has become increasingly popular, yet its malicious use, particularly the deepfakes, poses a serious threat to public trust and discourse. While deepfake detection methods achieve high predictive performance, they often exhibit biases across demographic attributes such as ethnicity and gender. In this work, we tackle the challenge of fair deepfake detection, aiming to mitigate these biases while maintaining robust detection capabilities. To this end, we propose a novel post-processing approach, referred to as Fairness-Oriented Final Layer Input Prioritising (Fair-FLIP), that reweights a trained model's final-layer inputs to reduce subgroup disparities, prioritising those with low variability while demoting highly variable ones. Experimental results comparing Fair-FLIP to both the baseline (without fairness-oriented de-biasing) and state-of-the-art approaches show that Fair-FLIP can enhance fairness metrics by up to 30% while maintaining baseline accuracy, with only a negligible reduction of 0.25%.
Code is available on Github: this https URL 

**Abstract (ZH)**: 人工智能生成的内容越来越受欢迎，但其恶意使用，尤其是深度伪造，对公众信任和讨论构成了严重威胁。尽管深度伪造检测方法具有高度的预测性能，但在种族和性别等人口统计属性方面常常表现出偏差。在本文中，我们致力于公平深度伪造检测的挑战，旨在减轻这些偏差同时保持检测能力的稳健性。为此，我们提出了一种新颖的后处理方法，称为面向公平性的最终层输入优先级调整（Fair-FLIP），该方法重新加权训练模型的最终层输入以减少子群体间的差异，优先处理低变异性部分，同时降低高变异性部分的权重。将Fair-FLIP与基准方法（未进行公平导向的去偏差处理）和最新方法进行比较的实验结果显示，Fair-FLIP可以在保持基准准确性的基础上，通过提高30%的公平性指标，同时减少不到0.25%的准确性。代码可在Github上获取：this https URL。 

---
# Last Layer Hamiltonian Monte Carlo 

**Title (ZH)**: 最后一层哈密尔顿蒙特卡洛 

**Authors**: Koen Vellenga, H. Joe Steinhauer, Göran Falkman, Jonas Andersson, Anders Sjögren  

**Link**: [PDF](https://arxiv.org/pdf/2507.08905)  

**Abstract**: We explore the use of Hamiltonian Monte Carlo (HMC) sampling as a probabilistic last layer approach for deep neural networks (DNNs). While HMC is widely regarded as a gold standard for uncertainty estimation, the computational demands limit its application to large-scale datasets and large DNN architectures. Although the predictions from the sampled DNN parameters can be parallelized, the computational cost still scales linearly with the number of samples (similar to an ensemble). Last layer HMC (LL--HMC) reduces the required computations by restricting the HMC sampling to the final layer of a DNN, making it applicable to more data-intensive scenarios with limited computational resources. In this paper, we compare LL-HMC against five last layer probabilistic deep learning (LL-PDL) methods across three real-world video datasets for driver action and intention. We evaluate the in-distribution classification performance, calibration, and out-of-distribution (OOD) detection. Due to the stochastic nature of the probabilistic evaluations, we performed five grid searches for different random seeds to avoid being reliant on a single initialization for the hyperparameter configurations. The results show that LL--HMC achieves competitive in-distribution classification and OOD detection performance. Additional sampled last layer parameters do not improve the classification performance, but can improve the OOD detection. Multiple chains or starting positions did not yield consistent improvements. 

**Abstract (ZH)**: 我们探索使用哈密顿蒙特卡洛（HMC）采样作为深度神经网络（DNN）的概率性最后一层方法。虽然HMC通常被视为不确定性估计的金标准，但由于计算需求限制了其在大规模数据集和大型DNN架构上的应用，尽管从采样的DNN参数进行预测可以并行化处理，但计算成本仍然会按样本数量线性增加（类似于集成方法）。最后一层HMC（LL--HMC）通过将HMC采样限制在DNN的最后一层，减少了所需的计算量，使其适用于计算资源有限的数据密集型场景。在本文中，我们将LL-HMC与五个其他最后一层概率深度学习（LL-PDL）方法在三个真实世界的视频数据集（涉及驾驶员行为和意图）上进行比较。我们评估了分布内分类性能、校准性能以及分布外（OOD）检测性能。由于概率评估的随机性，我们进行了五次网格搜索，以不同的随机种子避免依赖单一的超参数配置初始化。结果显示，LL--HMC在分布内分类和分布外检测性能上表现出竞争力。额外的采样最后一层参数不会提高分类性能，但可以改善分布外检测。多个链或起始位置没有一致地提升性能。 

---
# Generation of structure-guided pMHC-I libraries using Diffusion Models 

**Title (ZH)**: 基于结构引导的pMHC-I库生成方法 

**Authors**: Sergio Mares, Ariel Espinoza Weinberger, Nilah M. Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2507.08902)  

**Abstract**: Personalized vaccines and T-cell immunotherapies depend critically on identifying peptide-MHC class I (pMHC-I) interactions capable of eliciting potent immune responses. However, current benchmarks and models inherit biases present in mass-spectrometry and binding-assay datasets, limiting discovery of novel peptide ligands. To address this issue, we introduce a structure-guided benchmark of pMHC-I peptides designed using diffusion models conditioned on crystal structure interaction distances. Spanning twenty high-priority HLA alleles, this benchmark is independent of previously characterized peptides yet reproduces canonical anchor residue preferences, indicating structural generalization without experimental dataset bias. Using this resource, we demonstrate that state-of-the-art sequence-based predictors perform poorly at recognizing the binding potential of these structurally stable designs, indicating allele-specific limitations invisible in conventional evaluations. Our geometry-aware design pipeline yields peptides with high predicted structural integrity and higher residue diversity than existing datasets, representing a key resource for unbiased model training and evaluation. Our code, and data are available at: this https URL. 

**Abstract (ZH)**: 个性化疫苗和T细胞免疫治疗依赖于识别能够引发强大免疫反应的肽-MHCI（pMHC-I）相互作用，但当前的基准和模型继承了质谱和结合 assay 数据集中的偏差，限制了新型肽配体的发现。为了解决这一问题，我们引入了一种结构导向的基准，该基准使用条件于晶体结构相互作用距离的扩散模型设计pMHC-I多肽，覆盖了二十个高优先级的HLA等位基因，该基准独立于已知的肽序列，但却再现了经典的锚残基偏好，表明结构上的概括而无实验数据集偏差。使用该资源，我们证明了最先进的基于序列的预测器在识别这些结构上稳定的构想的结合潜力方面表现不佳，表明了在常规评估中看不见的等位基因特异性局限性。我们的几何感知设计流水线产生具有高预测结构完整性和更高残基多样性的肽，代表了无偏模型训练和评估的关键资源。代码和数据可在以下链接获取：this https URL。 

---
# SEALGuard: Safeguarding the Multilingual Conversations in Southeast Asian Languages for LLM Software Systems 

**Title (ZH)**: SEALGuard: 保障东南亚多语言对话的安全性软件系统 

**Authors**: Wenliang Shan, Michael Fu, Rui Yang, Chakkrit, Tantithamthavorn  

**Link**: [PDF](https://arxiv.org/pdf/2507.08898)  

**Abstract**: Safety alignment is critical for LLM-powered systems. While recent LLM-powered guardrail approaches such as LlamaGuard achieve high detection accuracy of unsafe inputs written in English (e.g., ``How to create a bomb?''), they struggle with multilingual unsafe inputs. This limitation leaves LLM systems vulnerable to unsafe and jailbreak prompts written in low-resource languages such as those in Southeast Asia. This paper introduces SEALGuard, a multilingual guardrail designed to improve the safety alignment across diverse languages. It aims to address the multilingual safety alignment gap of existing guardrails and ensure effective filtering of unsafe and jailbreak prompts in LLM-powered systems. We adapt a general-purpose multilingual language model into a multilingual guardrail using low-rank adaptation (LoRA). We construct SEALSBench, a large-scale multilingual safety alignment dataset containing over 260,000 prompts in ten languages, including safe, unsafe, and jailbreak cases. We evaluate SEALGuard against state-of-the-art guardrails such as LlamaGuard on this benchmark. Our findings show that multilingual unsafe and jailbreak prompts substantially degrade the performance of the state-of-the-art LlamaGuard, which experiences a drop in Defense Success Rate (DSR) by 9% and 18%, respectively, compared to its performance on English-only prompts. In contrast, SEALGuard outperforms existing guardrails in detecting multilingual unsafe and jailbreak prompts, improving DSR by 48% over LlamaGuard and achieving the best DSR, precision, and F1-score. Our ablation study further reveals the contributions of adaptation strategies and model size to the overall performance of SEALGuard. SEALGuard advances the safety alignment of LLM systems by introducing an effective multilingual guardrail. 

**Abstract (ZH)**: 多语言安全护栏SEALGuard：提升多语言下的安全对齐 

---
# Overview of the TREC 2023 deep learning track 

**Title (ZH)**: TREC 2023深度学习赛道概述 

**Authors**: Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Hossein A. Rahmani, Daniel Campos, Jimmy Lin, Ellen M. Voorhees, Ian Soboroff  

**Link**: [PDF](https://arxiv.org/pdf/2507.08890)  

**Abstract**: This is the fifth year of the TREC Deep Learning track. As in previous years, we leverage the MS MARCO datasets that made hundreds of thousands of human-annotated training labels available for both passage and document ranking tasks. We mostly repeated last year's design, to get another matching test set, based on the larger, cleaner, less-biased v2 passage and document set, with passage ranking as primary and document ranking as a secondary task (using labels inferred from passage). As we did last year, we sample from MS MARCO queries that were completely held out, unused in corpus construction, unlike the test queries in the first three years. This approach yields a more difficult test with more headroom for improvement. Alongside the usual MS MARCO (human) queries from MS MARCO, this year we generated synthetic queries using a fine-tuned T5 model and using a GPT-4 prompt.
The new headline result this year is that runs using Large Language Model (LLM) prompting in some way outperformed runs that use the "nnlm" approach, which was the best approach in the previous four years. Since this is the last year of the track, future iterations of prompt-based ranking can happen in other tracks. Human relevance assessments were applied to all query types, not just human MS MARCO queries. Evaluation using synthetic queries gave similar results to human queries, with system ordering agreement of $\tau=0.8487$. However, human effort was needed to select a subset of the synthetic queries that were usable. We did not see clear evidence of bias, where runs using GPT-4 were favored when evaluated using synthetic GPT-4 queries, or where runs using T5 were favored when evaluated on synthetic T5 queries. 

**Abstract (ZH)**: TREC深度学习跟踪的第五年：利用MS MARCO数据集进行段落和文档排名任务的模型训练与评估 

---
# AirScape: An Aerial Generative World Model with Motion Controllability 

**Title (ZH)**: AirScape: 一种具备运动可控性的空中生成世界模型 

**Authors**: Baining Zhao, Rongze Tang, Mingyuan Jia, Ziyou Wang, Fanghang Man, Xin Zhang, Yu Shang, Weichen Zhang, Chen Gao, Wei Wu, Xin Wang, Xinlei Chen, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.08885)  

**Abstract**: How to enable robots to predict the outcomes of their own motion intentions in three-dimensional space has been a fundamental problem in embodied intelligence. To explore more general spatial imagination capabilities, here we present AirScape, the first world model designed for six-degree-of-freedom aerial agents. AirScape predicts future observation sequences based on current visual inputs and motion intentions. Specifically, we construct an dataset for aerial world model training and testing, which consists of 11k video-intention pairs. This dataset includes first-person-view videos capturing diverse drone actions across a wide range of scenarios, with over 1,000 hours spent annotating the corresponding motion intentions. Then we develop a two-phase training schedule to train a foundation model -- initially devoid of embodied spatial knowledge -- into a world model that is controllable by motion intentions and adheres to physical spatio-temporal constraints. 

**Abstract (ZH)**: 如何使机器人在三维空间中预测自身运动意图的结果是嵌入式智能领域的基础问题。为了探索更广泛的空间想象能力，我们提出了AirScape，这是第一个为六自由度飞行代理设计的世界模型。AirScape根据当前视觉输入和运动意图预测未来观测序列。具体地，我们构建了一个用于飞行世界模型训练和测试的数据集，包含11000个视频-意图配对。该数据集包括第一人称视角视频，涵盖了多种无人机动作和广泛场景，相应的运动意图标注时间超过1000小时。然后我们开发了一个两阶段训练计划，将一个初始缺乏嵌入式空间知识的基础模型训练成一个可通过运动意图控制且遵循物理时空约束的世界模型。 

---
# The Consistency-Acceptability Divergence of LLMs in Judicial Decision-Making: Task and Stakeholder Dimensions 

**Title (ZH)**: LLMs在司法决策中的一致性-接受度偏差：任务和利益相关者维度 

**Authors**: Zhang MingDa, Xu Qing  

**Link**: [PDF](https://arxiv.org/pdf/2507.08881)  

**Abstract**: The integration of large language model (LLM) technology into judicial systems is fundamentally transforming legal practice worldwide. However, this global transformation has revealed an urgent paradox requiring immediate attention. This study introduces the concept of ``consistency-acceptability divergence'' for the first time, referring to the gap between technical consistency and social acceptance. While LLMs achieve high consistency at the technical level, this consistency demonstrates both positive and negative effects. Through comprehensive analysis of recent data on LLM judicial applications from 2023--2025, this study finds that addressing this challenge requires understanding both task and stakeholder dimensions. This study proposes the Dual-Track Deliberative Multi-Role LLM Judicial Governance Framework (DTDMR-LJGF), which enables intelligent task classification and meaningful interaction among diverse stakeholders. This framework offers both theoretical insights and practical guidance for building an LLM judicial ecosystem that balances technical efficiency with social legitimacy. 

**Abstract (ZH)**: 大语言模型技术集成到司法系统中正在从根本上改变全球的法律实践。然而，这一全球转变揭示出一个急需关注的悖论。本研究首次引入了“一致性可接受性差异”这一概念，指的是技术一致性与社会接受度之间的差距。尽管大语言模型在技术层面上实现了高度的一致性，这种一致性既显示出正面效果，也显示出负面效果。通过对2023—2025年大语言模型司法应用的综合分析，本研究发现解决这一挑战需要从任务和利益相关者维度进行理解。本研究提出了一种双轨审议多角色大语言模型司法治理框架（DTDMR-LJGF），该框架能够实现智能任务分类并促进多方利益相关者之间的有意义互动。该框架为构建一个兼顾技术效率和社会合法性的大语言模型司法生态系统提供了理论洞察和实践指导。 

---
# A Multi-Level Strategy for Deepfake Content Moderation under EU Regulation 

**Title (ZH)**: 欧盟法规下的多层次策略对抗虚假内容审核 

**Authors**: Max-Paul Förster, Luca Deck, Raimund Weidlich, Niklas Kühl  

**Link**: [PDF](https://arxiv.org/pdf/2507.08879)  

**Abstract**: The growing availability and use of deepfake technologies increases risks for democratic societies, e.g., for political communication on online platforms. The EU has responded with transparency obligations for providers and deployers of Artificial Intelligence (AI) systems and online platforms. This includes marking deepfakes during generation and labeling deepfakes when they are shared. However, the lack of industry and enforcement standards poses an ongoing challenge. Through a multivocal literature review, we summarize methods for marking, detecting, and labeling deepfakes and assess their effectiveness under EU regulation. Our results indicate that individual methods fail to meet regulatory and practical requirements. Therefore, we propose a multi-level strategy combining the strengths of existing methods. To account for the masses of content on online platforms, our multi-level strategy provides scalability and practicality via a simple scoring mechanism. At the same time, it is agnostic to types of deepfake technology and allows for context-specific risk weighting. 

**Abstract (ZH)**: 不断增强的深fake技术availability与应用增加了对民主社会的风险，例如在线平台上政治沟通的风险。欧盟对此采取了透明度义务措施，要求人工智能系统和在线平台提供商和部署者遵守，并包括在生成时标记深fake，以及在共享时对其进行标注。然而，缺乏行业和执法标准构成了持续挑战。通过多声腔文献综述，我们总结了标记、检测和标注深fake的方法，并评估它们在欧盟法规下的有效性。我们的结果显示，个别方法无法满足监管和实际要求。因此，我们提议一种多级策略，结合现有方法的优势。为了应对在线平台上大量内容，我们的多级策略通过简单的评分机制提供可扩展性和实用性。同时，它对深fake技术类型保持中立，并允许进行具体的上下文风险加权。 

---
# Towards Privacy-Preserving and Personalized Smart Homes via Tailored Small Language Models 

**Title (ZH)**: 面向定制小型语言模型的隐私保护和个人化智能家居 

**Authors**: Xinyu Huang, Leming Shen, Zijing Ma, Yuanqing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.08878)  

**Abstract**: Large Language Models (LLMs) have showcased remarkable generalizability in language comprehension and hold significant potential to revolutionize human-computer interaction in smart homes. Existing LLM-based smart home assistants typically transmit user commands, along with user profiles and home configurations, to remote servers to obtain personalized services. However, users are increasingly concerned about the potential privacy leaks to the remote servers. To address this issue, we develop HomeLLaMA, an on-device assistant for privacy-preserving and personalized smart home serving with a tailored small language model (SLM). HomeLLaMA learns from cloud LLMs to deliver satisfactory responses and enable user-friendly interactions. Once deployed, HomeLLaMA facilitates proactive interactions by continuously updating local SLMs and user profiles. To further enhance user experience while protecting their privacy, we develop PrivShield to offer an optional privacy-preserving LLM-based smart home serving for those users, who are unsatisfied with local responses and willing to send less-sensitive queries to remote servers. For evaluation, we build a comprehensive benchmark DevFinder to assess the service quality. Extensive experiments and user studies (M=100) demonstrate that HomeLLaMA can provide personalized services while significantly enhancing user privacy. 

**Abstract (ZH)**: 大型语言模型（LLMs）在语言理解和人机交互方面展现了卓越的泛化能力，并有望重塑智能家居。为了应对潜在的隐私泄露问题，我们开发了HomeLLaMA，这是一种基于设备端的小型语言模型（SLM）实现隐私保护和个性化的智能家居助手。HomeLLaMA 从云端大型语言模型学习以提供满意的响应，并支持用户友好的交互。部署后，HomeLLaMA 通过持续更新本地 SLM 和用户配置文件促进主动交互。为了进一步提升用户体验并保护隐私，我们开发了PrivShield，为那些对本地响应不满意并愿意向远程服务器发送不敏感查询的用户提供可选的隐私保护型语言模型驱动的智能家居服务。为了评估性能，我们构建了全面的基准测试DevFinder来评估服务质量。广泛的实验和用户研究（M=100）表明，HomeLLaMA 能够提供个性化服务并显著增强用户隐私保护。 

---
# ODIA: Oriented Distillation for Inline Acceleration of LLM-based Function Calling 

**Title (ZH)**: 面向内置加速的基于LLM的功能调用-distillation方法 

**Authors**: Hanlong Zhang, Jingsheng Yang, Hao Li, Yuhao He, Franck Gong  

**Link**: [PDF](https://arxiv.org/pdf/2507.08877)  

**Abstract**: Function Calling is a crucial technique that enables Large Language Models (LLMs) to interact with external systems through APIs. However, the high latency associated with LLM-based Function Calling significantly impacts user experience. This paper presents a novel approach called Oriented Distillation for Inline Acceleration (ODIA) that leverages online user interaction data to accelerate Function Calling. By automatically identifying "simple queries" from production traffic and distilling knowledge from larger models to smaller ones, our method reduces response latency by 45% (expected) and 78% (median) while maintaining accuracy. We demonstrate the effectiveness of our approach through real-world deployment in a music application, where the smaller model successfully handles 60% of traffic with negligible accuracy loss. Our method requires minimal human intervention and continuously improves through automated data collection and model updating, making it a practical solution for production environments. 

**Abstract (ZH)**: 面向调用的定向蒸馏在线加速（ODIA）：通过用户交互数据加速函数调用 

---
# Contrastive Language-Image Pre-Training Model based Semantic Communication Performance Optimization 

**Title (ZH)**: 基于对比语言-图像预训练模型的语义通信性能优化 

**Authors**: Shaoran Yang, Dongyu Wei, Hanzhi Yu, Zhaohui Yang, Yuchen Liu, Mingzhe Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.08873)  

**Abstract**: In this paper, a novel contrastive language-image pre-training (CLIP) model based semantic communication framework is designed. Compared to standard neural network (e.g.,convolutional neural network) based semantic encoders and decoders that require joint training over a common dataset, our CLIP model based method does not require any training procedures thus enabling a transmitter to extract data meanings of the original data without neural network model training, and the receiver to train a neural network for follow-up task implementation without the communications with the transmitter. Next, we investigate the deployment of the CLIP model based semantic framework over a noisy wireless network. Since the semantic information generated by the CLIP model is susceptible to wireless noise and the spectrum used for semantic information transmission is limited, it is necessary to jointly optimize CLIP model architecture and spectrum resource block (RB) allocation to maximize semantic communication performance while considering wireless noise, the delay and energy used for semantic communication. To achieve this goal, we use a proximal policy optimization (PPO) based reinforcement learning (RL) algorithm to learn how wireless noise affect the semantic communication performance thus finding optimal CLIP model and RB for each user. Simulation results show that our proposed method improves the convergence rate by up to 40%, and the accumulated reward by 4x compared to soft actor-critic. 

**Abstract (ZH)**: 基于CLIP模型的语义通信框架设计及其在嘈杂无线网络中的部署 

---
# Next-Generation Travel Demand Modeling with a Generative Framework for Household Activity Coordination 

**Title (ZH)**: 基于生成框架的家庭活动协调的下一代旅行需求建模 

**Authors**: Xishun Liao, Haoxuan Ma, Yifan Liu, Yuxiang Wei, Brian Yueshuai He, Chris Stanford, Jiaqi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.08871)  

**Abstract**: Travel demand models are critical tools for planning, policy, and mobility system design. Traditional activity-based models (ABMs), although grounded in behavioral theories, often rely on simplified rules and assumptions, and are costly to develop and difficult to adapt across different regions. This paper presents a learning-based travel demand modeling framework that synthesizes household-coordinated daily activity patterns based on a household's socio-demographic profiles. The whole framework integrates population synthesis, coordinated activity generation, location assignment, and large-scale microscopic traffic simulation into a unified system. It is fully generative, data-driven, scalable, and transferable to other regions. A full-pipeline implementation is conducted in Los Angeles with a 10 million population. Comprehensive validation shows that the model closely replicates real-world mobility patterns and matches the performance of legacy ABMs with significantly reduced modeling cost and greater scalability. With respect to the SCAG ABM benchmark, the origin-destination matrix achieves a cosine similarity of 0.97, and the daily vehicle miles traveled (VMT) in the network yields a 0.006 Jensen-Shannon Divergence (JSD) and a 9.8% mean absolute percentage error (MAPE). When compared to real-world observations from Caltrans PeMS, the evaluation on corridor-level traffic speed and volume reaches a 0.001 JSD and a 6.11% MAPE. 

**Abstract (ZH)**: 基于学习的旅行需求建模框架：基于家庭协调的日活动模式合成 

---
# Privacy-Utility-Fairness: A Balanced Approach to Vehicular-Traffic Management System 

**Title (ZH)**: 隐私-效益-公平： vehicular-traffic 管理系统中的一种平衡方法 

**Authors**: Poushali Sengupta, Sabita Maharjan, frank Eliassen, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08864)  

**Abstract**: Location-based vehicular traffic management faces significant challenges in protecting sensitive geographical data while maintaining utility for traffic management and fairness across regions. Existing state-of-the-art solutions often fail to meet the required level of protection against linkage attacks and demographic biases, leading to privacy leakage and inequity in data analysis. In this paper, we propose a novel algorithm designed to address the challenges regarding the balance of privacy, utility, and fairness in location-based vehicular traffic management systems. In this context, utility means providing reliable and meaningful traffic information, while fairness ensures that all regions and individuals are treated equitably in data use and decision-making. Employing differential privacy techniques, we enhance data security by integrating query-based data access with iterative shuffling and calibrated noise injection, ensuring that sensitive geographical data remains protected. We ensure adherence to epsilon-differential privacy standards by implementing the Laplace mechanism. We implemented our algorithm on vehicular location-based data from Norway, demonstrating its ability to maintain data utility for traffic management and urban planning while ensuring fair representation of all geographical areas without being overrepresented or underrepresented. Additionally, we have created a heatmap of Norway based on our model, illustrating the privatized and fair representation of the traffic conditions across various cities. Our algorithm provides privacy in vehicular traffic 

**Abstract (ZH)**: 基于位置的车辆交通管理在保护敏感地理数据隐私、维持交通管理实用性和区域公平性之间面临着重大挑战。现有先进解决方案往往无法满足对链接攻击和人口统计偏差的保护要求，导致隐私泄露和数据分析中的不公平。本文提出了一种新型算法，旨在解决基于位置的车辆交通管理系统中隐私、实用性和公平性之间的平衡问题。在该上下文中，实用性是指提供可靠和有意义的交通信息，而公平性则确保在数据使用和决策过程中所有地区和个体得到公平对待。我们通过运用差分隐私技术，结合查询驱动的数据访问、迭代洗牌和校准噪声注入，增强数据安全性，确保敏感地理数据的安全。通过实现拉普拉斯机制，我们确保算法符合ε-差分隐私标准。我们在挪威的车辆基于位置的数据上实现了该算法，展示了其在维持交通管理和城市规划的数据实用性的同时，能够公平地代表所有地理区域而不会出现过度代表或不足代表的情况。此外，我们还根据我们的模型创建了挪威的热力图，展示了各种城市中交通状况的私有化和公平表示。该算法提供了车辆交通隐私。 

---
# Foundation models for time series forecasting: Application in conformal prediction 

**Title (ZH)**: 时间序列预测中的基础模型：在可信预测中的应用 

**Authors**: Sami Achour, Yassine Bouher, Duong Nguyen, Nicolas Chesneau  

**Link**: [PDF](https://arxiv.org/pdf/2507.08858)  

**Abstract**: The zero-shot capabilities of foundation models (FMs) for time series forecasting offer promising potentials in conformal prediction, as most of the available data can be allocated to calibration. This study compares the performance of Time Series Foundation Models (TSFMs) with traditional methods, including statistical models and gradient boosting, within a conformal prediction setting. Our findings highlight two key advantages of TSFMs. First, when the volume of data is limited, TSFMs provide more reliable conformalized prediction intervals than classic models, thanks to their superior predictive accuracy. Second, the calibration process is more stable because more data are used for calibration. Morever, the fewer data available, the more pronounced these benefits become, as classic models require a substantial amount of data for effective training. These results underscore the potential of foundation models in improving conformal prediction reliability in time series applications, particularly in data-constrained cases. All the code to reproduce the experiments is available. 

**Abstract (ZH)**: 基础模型在时间序列预测中的零-shot能力为双重校验预测提供了令人鼓舞的潜力，因为大多数可用数据可以分配到校准过程中。本研究在双重校验预测框架下比较了时间序列基础模型（TSFMs）与传统方法（包括统计模型和梯度提升）的性能。我们的研究结果突显了TSFMs的两个主要优势。首先，当数据量受限时，TSFMs提供了比经典模型更可靠的双重校验预测区间，得益于其更高的预测准确性。其次，校准过程更加稳定，因为更多的数据用于校准。此外，随着可用数据的减少，这些优势变得更加显著，因为经典模型需要大量数据才能有效训练。这些结果强调了基础模型在提高时间序列应用中双重校验预测可靠性方面的潜力，特别是在数据受限的情况下。所有重现实验的代码均可获取。 

---
# Clio-X: AWeb3 Solution for Privacy-Preserving AI Access to Digital Archives 

**Title (ZH)**: Clio-X：一种保护隐私的Web3 AI访问数字档案解决方案 

**Authors**: Victoria L. Lemieux, Rosa Gil, Faith Molosiwa, Qihong Zhou, Binming Li, Roberto Garcia, Luis De La Torre Cubillo, Zehua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08853)  

**Abstract**: As archives turn to artificial intelligence to manage growing volumes of digital records, privacy risks inherent in current AI data practices raise critical concerns about data sovereignty and ethical accountability. This paper explores how privacy-enhancing technologies (PETs) and Web3 architectures can support archives to preserve control over sensitive content while still being able to make it available for access by researchers. We present Clio-X, a decentralized, privacy-first Web3 digital solution designed to embed PETs into archival workflows and support AI-enabled reference and access. Drawing on a user evaluation of a medium-fidelity prototype, the study reveals both interest in the potential of the solution and significant barriers to adoption related to trust, system opacity, economic concerns, and governance. Using Rogers' Diffusion of Innovation theory, we analyze the sociotechnical dimensions of these barriers and propose a path forward centered on participatory design and decentralized governance through a Clio-X Decentralized Autonomous Organization. By integrating technical safeguards with community-based oversight, Clio-X offers a novel model to ethically deploy AI in cultural heritage contexts. 

**Abstract (ZH)**: 随着档案机构利用人工智能管理日益增长的数字记录数量，现有AI数据实践中的隐私风险引发了关于数据主权和伦理责任的关键关注。本文探讨了如何通过增强隐私的技术（PETs）和Web3架构，支持档案机构在保留对敏感内容控制权的同时，仍能让研究者访问相关内容。我们介绍了Clio-X，这是一种去中心化的、以隐私为中心的Web3数字解决方案，旨在将PETs嵌入到档案工作流程中，并支持基于AI的参考和访问。通过一个中保真度原型的用户评估，研究揭示了对该解决方案潜在兴趣以及信托、系统透明度、经济考量和治理方面的显著采用障碍。我们运用罗杰斯的创新扩散理论分析这些障碍的社技维度，并提出以参与式设计和通过Clio-X去中心化自治组织实现去中心化治理为中心的发展路径。通过将技术保障与社区监督相结合，Clio-X提供了一种新的模型，以伦理方式在文化遗产领域部署人工智能。 

---
# Assuring the Safety of Reinforcement Learning Components: AMLAS-RL 

**Title (ZH)**: 保证强化学习组件的安全性：AMLAS-RL 

**Authors**: Calum Corrie Imrie, Ioannis Stefanakos, Sepeedeh Shahbeigi, Richard Hawkins, Simon Burton  

**Link**: [PDF](https://arxiv.org/pdf/2507.08848)  

**Abstract**: The rapid advancement of machine learning (ML) has led to its increasing integration into cyber-physical systems (CPS) across diverse domains. While CPS offer powerful capabilities, incorporating ML components introduces significant safety and assurance challenges. Among ML techniques, reinforcement learning (RL) is particularly suited for CPS due to its capacity to handle complex, dynamic environments where explicit models of interaction between system and environment are unavailable or difficult to construct. However, in safety-critical applications, this learning process must not only be effective but demonstrably safe. Safe-RL methods aim to address this by incorporating safety constraints during learning, yet they fall short in providing systematic assurance across the RL lifecycle. The AMLAS methodology offers structured guidance for assuring the safety of supervised learning components, but it does not directly apply to the unique challenges posed by RL. In this paper, we adapt AMLAS to provide a framework for generating assurance arguments for an RL-enabled system through an iterative process; AMLAS-RL. We demonstrate AMLAS-RL using a running example of a wheeled vehicle tasked with reaching a target goal without collision. 

**Abstract (ZH)**: 机器学习快速进步使其在跨多个领域的网络物理系统(CPS)中日益集成。尽管CPS提供了强大的功能，但集成机器学习组件引入了重要的安全性和保障挑战。在机器学习技术中，强化学习(RL)特别适用于CPS，因为它们能够处理缺乏或难以构建系统与环境交互模型的复杂动态环境。但在关键安全应用中，这一学习过程不仅需要有效，还必须是可验证的安全的。安全的RL方法旨在通过在学习过程中引入安全约束来解决这一问题，但它们在提供整个RL生命周期中的系统保障方面仍有所欠缺。AMLAS方法提供了监督学习组件安全性的结构化指导，但未能直接适用于RL所提出的所有独特挑战。在本文中，我们通过对AMLAS进行适应，提出了一种通过迭代过程生成RL使能系统保障论据的框架——AMLAS-RL。我们通过一辆需在不发生碰撞的情况下到达目标点的轮式车辆运行示例来展示AMLAS-RL。 

---
# DAFOS: Dynamic Adaptive Fanout Optimization Sampler 

**Title (ZH)**: DAFOS：动态自适应扇出优化采样器 

**Authors**: Irfan Ullah, Young-Koo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.08845)  

**Abstract**: Graph Neural Networks (GNNs) are becoming an essential tool for learning from graph-structured data, however uniform neighbor sampling and static fanout settings frequently limit GNNs' scalability and efficiency. In this paper, we propose the Dynamic Adaptive Fanout Optimization Sampler (DAFOS), a novel approach that dynamically adjusts the fanout based on model performance and prioritizes important nodes during training. Our approach leverages node scoring based on node degree to focus computational resources on structurally important nodes, incrementing the fanout as the model training progresses. DAFOS also integrates an early stopping mechanism to halt training when performance gains diminish. Experiments conducted on three benchmark datasets, ogbnarxiv, Reddit, and ogbn-products, demonstrate that our approach significantly improves training speed and accuracy compared to a state-of-the-art approach. DAFOS achieves a 3.57x speedup on the ogbn-arxiv dataset and a 12.6x speedup on the Reddit dataset while improving the F1 score from 68.5% to 71.21% on ogbn-arxiv and from 73.78% to 76.88% on the ogbn-products dataset, respectively. These results highlight the potential of DAFOS as an efficient and scalable solution for large-scale GNN training. 

**Abstract (ZH)**: 动态自适应扇出优化抽样器（DAFOS）：一种基于模型性能动态调整扇出并优先处理重要节点的方法 

---
# Can We Predict Your Next Move Without Breaking Your Privacy? 

**Title (ZH)**: 我们能在不侵犯您的隐私的情况下预测您的下一行动吗？ 

**Authors**: Arpita Soni, Sahil Tripathi, Gautam Siddharth Kashyap, Manaswi Kulahara, Mohammad Anas Azeez, Zohaib Hasan Siddiqui, Nipun Joshi, Jiechao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.08843)  

**Abstract**: We propose FLLL3M--Federated Learning with Large Language Models for Mobility Modeling--a privacy-preserving framework for Next-Location Prediction (NxLP). By retaining user data locally and leveraging LLMs through an efficient outer product mechanism, FLLL3M ensures high accuracy with low resource demands. It achieves SOT results on Gowalla (Acc@1: 12.55, MRR: 0.1422), WeePlace (10.71, 0.1285), Brightkite (10.42, 0.1169), and FourSquare (8.71, 0.1023), while reducing parameters by up to 45.6% and memory usage by 52.7%. 

**Abstract (ZH)**: 使用大型语言模型的联邦学习框架FLLL3M——移动性建模中的隐私保护框架——下一位置预测（NxLP） 

---
# Gradients as an Action: Towards Communication-Efficient Federated Recommender Systems via Adaptive Action Sharing 

**Title (ZH)**: 梯度作为行动：通过适应性行动共享实现高效联邦推荐系统的研究 

**Authors**: Zhufeng Lu, Chentao Jia, Ming Hu, Xiaofei Xie, Mingsong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.08842)  

**Abstract**: As a promising privacy-aware collaborative model training paradigm, Federated Learning (FL) is becoming popular in the design of distributed recommender systems. However, Federated Recommender Systems (FedRecs) greatly suffer from two major problems: i) extremely high communication overhead due to massive item embeddings involved in recommendation systems, and ii) intolerably low training efficiency caused by the entanglement of both heterogeneous network environments and client devices. Although existing methods attempt to employ various compression techniques to reduce communication overhead, due to the parameter errors introduced by model compression, they inevitably suffer from model performance degradation. To simultaneously address the above problems, this paper presents a communication-efficient FedRec framework named FedRAS, which adopts an action-sharing strategy to cluster the gradients of item embedding into a specific number of model updating actions for communication rather than directly compressing the item embeddings. In this way, the cloud server can use the limited actions from clients to update all the items. Since gradient values are significantly smaller than item embeddings, constraining the directions of gradients (i.e., the action space) introduces smaller errors compared to compressing the entire item embedding matrix into a reduced space. To accommodate heterogeneous devices and network environments, FedRAS incorporates an adaptive clustering mechanism that dynamically adjusts the number of actions. Comprehensive experiments on well-known datasets demonstrate that FedRAS can reduce the size of communication payloads by up to 96.88%, while not sacrificing recommendation performance within various heterogeneous scenarios. We have open-sourced FedRAS at this https URL. 

**Abstract (ZH)**: 作为一种有前景的隐私感知协作模型训练范式，联邦学习（Federated Learning, FL）在分布式推荐系统的设计中正变得流行。然而，联邦推荐系统（FedRecs）受到两大主要问题的严重影响：一是由于推荐系统中涉及大量项嵌入而导致极大的通信开销；二是由于异构网络环境和客户端设备的纠缠而导致不堪忍受的低训练效率。尽管现有方法尝试使用各种压缩技术来减少通信开销，但由于模型压缩引入的参数误差，它们不可避免地会牺牲模型性能。为了同时解决上述问题，本文提出了一种名为FedRAS的通信高效FedRec框架，该框架采用动作共享策略将项嵌入的梯度聚类成特定数量的模型更新动作用于通信，而不是直接压缩项嵌入。这样一来，云服务器可以利用客户端的有限动作来更新所有项。由于梯度值远小于项嵌入，限制梯度的方向（即动作空间）引入的误差比将整个项嵌入矩阵压缩到较小空间要小。为了适应异构设备和网络环境，FedRAS整合了一种自适应聚类机制，能够动态调整动作数量。在多种异构场景下的综合实验表明，FedRAS可以将通信负载大小减少高达96.88%，且不会牺牲推荐性能。我们已将FedRAS开源于此链接。 

---
# Zero-Shot Neural Architecture Search with Weighted Response Correlation 

**Title (ZH)**: 零样本神经架构搜索：带加权响应相关性方法 

**Authors**: Kun Jing, Luoyu Chen, Jungang Xu, Jianwei Tai, Yiyu Wang, Shuaimin Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.08841)  

**Abstract**: Neural architecture search (NAS) is a promising approach for automatically designing neural network architectures. However, the architecture estimation of NAS is computationally expensive and time-consuming because of training multiple architectures from scratch. Although existing zero-shot NAS methods use training-free proxies to accelerate the architecture estimation, their effectiveness, stability, and generality are still lacking. We present a novel training-free estimation proxy called weighted response correlation (WRCor). WRCor utilizes correlation coefficient matrices of responses across different input samples to calculate the proxy scores of estimated architectures, which can measure their expressivity and generalizability. Experimental results on proxy evaluation demonstrate that WRCor and its voting proxies are more efficient estimation strategies than existing proxies. We also apply them with different search strategies in architecture search. Experimental results on architecture search show that our zero-shot NAS algorithm outperforms most existing NAS algorithms in different search spaces. Our NAS algorithm can discover an architecture with a 22.1% test error on the ImageNet-1k dataset within 4 GPU hours. All codes are publicly available at this https URL. 

**Abstract (ZH)**: 无监督神经架构搜索：一种基于加权响应相关性的新型估计代理方法 

---
# Domain-Adaptive Diagnosis of Lewy Body Disease with Transferability Aware Transformer 

**Title (ZH)**: 带有转移意识的变换器在莱氏体疾病领域自适应诊断中应用 

**Authors**: Xiaowei Yu, Jing Zhang, Tong Chen, Yan Zhuang, Minheng Chen, Chao Cao, Yanjun Lyu, Lu Zhang, Li Su, Tianming Liu, Dajiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08839)  

**Abstract**: Lewy Body Disease (LBD) is a common yet understudied form of dementia that imposes a significant burden on public health. It shares clinical similarities with Alzheimer's disease (AD), as both progress through stages of normal cognition, mild cognitive impairment, and dementia. A major obstacle in LBD diagnosis is data scarcity, which limits the effectiveness of deep learning. In contrast, AD datasets are more abundant, offering potential for knowledge transfer. However, LBD and AD data are typically collected from different sites using different machines and protocols, resulting in a distinct domain shift. To effectively leverage AD data while mitigating domain shift, we propose a Transferability Aware Transformer (TAT) that adapts knowledge from AD to enhance LBD diagnosis. Our method utilizes structural connectivity (SC) derived from structural MRI as training data. Built on the attention mechanism, TAT adaptively assigns greater weights to disease-transferable features while suppressing domain-specific ones, thereby reducing domain shift and improving diagnostic accuracy with limited LBD data. The experimental results demonstrate the effectiveness of TAT. To the best of our knowledge, this is the first study to explore domain adaptation from AD to LBD under conditions of data scarcity and domain shift, providing a promising framework for domain-adaptive diagnosis of rare diseases. 

**Abstract (ZH)**: Lewy 体疾病 (LBD) 是一种常见但研究不足的痴呆形式，对公共卫生造成重大负担。它在临床表现上与阿尔茨海默病 (AD) 相似，两者均经历正常认知、轻度认知 impairment 和痴呆等阶段。LBD 诊断的一大障碍是数据稀缺性，限制了深度学习的有效性。相比之下，AD 数据更为丰富，为知识迁移提供了潜在可能性。然而，LBD 和 AD 数据通常来自不同的地点，使用不同的设备和协议收集，导致了独特的领域偏移。为有效利用 AD 数据并减轻领域偏移，我们提出了一种aware Transformer (TAT)，以将 AD 中的知识应用于增强 LBD 诊断。该方法利用结构 MRI 提取的结构连接性 (SC) 作为训练数据。基于注意力机制，TAT 自适应地赋予可转移疾病特征更大的权重，同时抑制领域特异性特征，从而减少领域偏移并提高在有限 LBD 数据下的诊断准确性。实验结果证明了 TAT 的有效性。据我们所知，这是首次在数据稀缺性和领域偏移条件下探索从 AD 到 LBD 的领域适应性诊断的研究，为稀有疾病的领域适应性诊断提供了有 promise 的框架。 

---
# wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models 

**Title (ZH)**: 加权策略优化在扩散语言模型中的推理 

**Authors**: Xiaohang Tang, Rares Dolga, Sangwoong Yoon, Ilija Bogunovic  

**Link**: [PDF](https://arxiv.org/pdf/2507.08838)  

**Abstract**: Improving the reasoning capabilities of diffusion-based large language models (dLLMs) through reinforcement learning (RL) remains an open problem. The intractability of dLLMs likelihood function necessitates approximating the current, old, and reference policy likelihoods at each policy optimization step. This reliance introduces additional computational overhead and lead to potentially large bias -- particularly when approximation errors occur in the denominator of policy ratios used for importance sampling. To mitigate these issues, we introduce $\mathtt{wd1}$, a novel policy optimization approach that reformulates the objective as a weighted likelihood, requiring only a single approximation for the current parametrized policy likelihood. Experiments on widely used reasoning benchmarks demonstrate that $\mathtt{wd1}$, without supervised fine-tuning (SFT) or any supervised data, outperforms existing RL methods for dLLMs, achieving up to 16% higher accuracy. $\mathtt{wd1}$ delivers additional computational gains, including reduced training time and fewer function evaluations (NFEs) per gradient step. These findings, combined with the simplicity of method's implementation and R1-Zero-like training (no SFT), position $\mathtt{wd1}$ as a more effective and efficient method for applying RL to dLLMs reasoning. 

**Abstract (ZH)**: 通过强化学习提高基于扩散的大语言模型推理能力仍是一项开放问题。基于扩散的大语言模型（dLLMs）似然函数的不可计算性要求在每次策略优化步骤中分别近似当前、旧和参考策略的似然性。这种依赖性引入了额外的计算开销，并可能导致偏置——尤其是在使用用于重要性采样的策略比值分母中的近似错误时。为缓解这些问题，我们引入了$\mathtt{wd1}$，一种新颖的策略优化方法，将目标重新表述为加权似然性，只需对当前参数化策略的似然性进行一次近似。在广泛使用的推理基准测试上进行的实验表明，$\mathtt{wd1}$在无需有监督微调（SFT）或任何有监督数据的情况下，优于现有的RL方法，准确率提高了16%。$\mathtt{wd1}$还带来了额外的计算优势，包括缩短的训练时间和每梯度步中较少的功能评估次数（NFEs）。结合该方法实现的简洁性和类似于R1-Zero的训练方式（无SFT），$\mathtt{wd1}$定位为应用于dLLMs推理的更有效和更高效的方法。 

---
# Representation learning with a transformer by contrastive learning for money laundering detection 

**Title (ZH)**: 基于对比学习的变压器表示学习在洗钱检测中的应用 

**Authors**: Harold Guéneau, Alain Celisse, Pascal Delange  

**Link**: [PDF](https://arxiv.org/pdf/2507.08835)  

**Abstract**: The present work tackles the money laundering detection problem. A new procedure is introduced which exploits structured time series of both qualitative and quantitative data by means of a transformer neural network. The first step of this procedure aims at learning representations of time series through contrastive learning (without any labels). The second step leverages these representations to generate a money laundering scoring of all observations. A two-thresholds approach is then introduced, which ensures a controlled false-positive rate by means of the Benjamini-Hochberg (BH) procedure. Experiments confirm that the transformer is able to produce general representations that succeed in exploiting money laundering patterns with minimal supervision from domain experts. It also illustrates the higher ability of the new procedure for detecting nonfraudsters as well as fraudsters, while keeping the false positive rate under control. This greatly contrasts with rule-based procedures or the ones based on LSTM architectures. 

**Abstract (ZH)**: 本研究解决了洗钱检测问题。提出了一种新的方法，通过变压器神经网络利用结构化的时间序列数据（既有定性数据也有定量数据）进行检测。该方法的第一步通过对比学习（无需标签）学习时间序列的表示。第二步利用这些表示为所有观察生成洗钱评分。引入了一种双阈值方法，通过贝杰曼-霍奇格（BH）程序确保了较低的假阳性率。实验结果证实，变压器能够生成通用表示，能够利用洗钱模式进行检测，同时需要较少的主题专家监督。此外，该方法在检测非欺诈者和欺诈者方面表现更优，同时将假阳性率控制在可接受范围内。这与基于规则的方法或基于LSTM的架构形成了鲜明对比。 

---
# LoRA Is Slower Than You Think 

**Title (ZH)**: LoRA并非你想象中那么快 

**Authors**: Seokmin Ko  

**Link**: [PDF](https://arxiv.org/pdf/2507.08833)  

**Abstract**: Low-Rank Adaptation (LoRA) is one of the most widely used techniques for fine-tuning large language models (LLMs). By introducing a small number of trainable low-rank weight matrices, LoRA substantially reduces the number of parameters that need to be updated, offering significant advantages in memory consumption and computational efficiency compared to full fine-tuning. However, we observed that LoRA does not consistently provide speed improvements across all model architectures and training setups. Motivated by this inconsistency, we conduct a comprehensive analysis of LoRA's performance and investigate the underlying factors limiting its speedup. Based on our findings, we propose several methods for more efficient fine-tuning of LLMs. We empirically evaluate these methods and compare them to LoRA, demonstrating that our approach achieves comparable or superior performance while delivering more consistent training speed improvements. Our work offers valuable insights and practical guidelines for practitioners seeking to optimize LLM fine-tuning under resource constraints. 

**Abstract (ZH)**: 低秩适应（LoRA）是用于 fine-tuning 大型语言模型（LLMs）的最广泛使用的技术之一。通过引入少量可训练的低秩权重矩阵，LoRA 显著减少了需要更新的参数数量，与全量 fine-tuning 相比，在内存消耗和计算效率方面具有显著优势。然而，我们观察到 LoRA 并不一致地为所有模型架构和训练配置提供速度改进。受这种不一致性驱动，我们对 LoRA 的性能进行了全面分析，并探讨其速度加速受限的内在因素。基于我们的发现，我们提出了一些用于更高效 fine-tuning LLMs 的方法。我们通过实验证明这些方法，并将它们与 LoRA 进行比较，显示我们的方法不仅能实现可比或更优的性能，还能提供更一致的训练速度改进。我们的研究为资源受限下优化 LLM fine-tuning 的实践者提供了宝贵见解和实用指南。 

---
# Efficient Triple Modular Redundancy for Reliability Enhancement of DNNs Using Explainable AI 

**Title (ZH)**: 基于可解释人工智能的DNN可靠性增强的高效三模冗余方法 

**Authors**: Kimia Soroush, Nastaran Shirazi, Mohsen Raji  

**Link**: [PDF](https://arxiv.org/pdf/2507.08829)  

**Abstract**: Deep Neural Networks (DNNs) are widely employed in safety-critical domains, where ensuring their reliability is essential. Triple Modular Redundancy (TMR) is an effective technique to enhance the reliability of DNNs in the presence of bit-flip faults. In order to handle the significant overhead of TMR, it is applied selectively on the parameters and components with the highest contribution at the model output. Hence, the accuracy of the selection criterion plays the key role on the efficiency of TMR. This paper presents an efficient TMR approach to enhance the reliability of DNNs against bit-flip faults using an Explainable Artificial Intelligence (XAI) method. Since XAI can provide valuable insights about the importance of individual neurons and weights in the performance of the network, they can be applied as the selection metric in TMR techniques. The proposed method utilizes a low-cost, gradient-based XAI technique known as Layer-wise Relevance Propagation (LRP) to calculate importance scores for DNN parameters. These scores are then used to enhance the reliability of the model, with the most critical weights being protected by TMR. The proposed approach is evaluated on two DNN models, VGG16 and AlexNet, using datasets such as MNIST and CIFAR-10. The results demonstrate that the method can protect the AlexNet model at a bit error rate of 10-4, achieving over 60% reliability improvement while maintaining the same overhead as state-of-the-art methods. 

**Abstract (ZH)**: 深层神经网络（DNNs）广泛应用于安全关键领域，确保其可靠性至关重要。在存在位翻转故障的情况下，三模冗余（TMR）是一种有效的提高DNNs可靠性的技术。为了处理TMR的显著开销，它仅被有最高贡献的模型参数和组件上选择性地应用。因此，选择标准的准确性在TMR的效率中起着关键作用。本文提出了一种高效的基于可解释人工智能（XAI）的TMR方法，以提高DNNs在位翻转故障情况下的可靠性。由于XAI可以提供有关网络性能中 individual 神经元和权重重要性的宝贵见解，它们可以作为TMR技术中的选择指标。所提出的方法利用一种低成本的基于梯度的XAI技术——层相关性传播（LRP）来计算DNN参数的重要性分数。这些分数随后被用于增强模型的可靠性，最关键的权重通过TMR进行保护。所提出的方法在VGG16和AlexNet两个DNN模型上进行了评估，使用MNIST和CIFAR-10等数据集。结果表明，在位错误率为 \(10^{-4}\) 的情况下，该方法能够保护AlexNet模型，同时在保持与最新方法相同开销的同时，可靠性提高了超过60%。 

---
# Advancing network resilience theories with symbolized reinforcement learning 

**Title (ZH)**: 用符号化强化学习推进网络韧性理论 

**Authors**: Yu Zheng, Jingtao Ding, Depeng Jin, Jianxi Gao, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.08827)  

**Abstract**: Many complex networks display remarkable resilience under external perturbations, internal failures and environmental changes, yet they can swiftly deteriorate into dysfunction upon the removal of a few keystone nodes. Discovering theories that measure network resilience offers the potential to prevent catastrophic collapses--from species extinctions to financial crise--with profound implications for real-world systems. Current resilience theories address the problem from a single perspective of topology, neglecting the crucial role of system dynamics, due to the intrinsic complexity of the coupling between topology and dynamics which exceeds the capabilities of human analytical methods. Here, we report an automatic method for resilience theory discovery, which learns from how AI solves a complicated network dismantling problem and symbolizes its network attack strategies into theoretical formulas. This proposed self-inductive approach discovers the first resilience theory that accounts for both topology and dynamics, highlighting how the correlation between node degree and state shapes overall network resilience, and offering insights for designing early warning signals of systematic collapses. Additionally, our approach discovers formulas that refine existing well-established resilience theories with over 37.5% improvement in accuracy, significantly advancing human understanding of complex networks with AI. 

**Abstract (ZH)**: 自动发现同时考虑拓扑与动力学的网络韧性理论：从人工智能复杂网络拆解中学习并提炼网络攻击策略公式 

---
# Lightweight Cloud Masking Models for On-Board Inference in Hyperspectral Imaging 

**Title (ZH)**: 轻量级云遮蔽模型：用于高光谱成像的机载推理 

**Authors**: Mazen Ali, António Pereira, Fabio Gentile, Aser Cortines, Sam Mugel, Román Orús, Stelios P. Neophytides, Michalis Mavrovouniotis  

**Link**: [PDF](https://arxiv.org/pdf/2507.08052)  

**Abstract**: Cloud and cloud shadow masking is a crucial preprocessing step in hyperspectral satellite imaging, enabling the extraction of high-quality, analysis-ready data. This study evaluates various machine learning approaches, including gradient boosting methods such as XGBoost and LightGBM as well as convolutional neural networks (CNNs). All boosting and CNN models achieved accuracies exceeding 93%. Among the investigated models, the CNN with feature reduction emerged as the most efficient, offering a balance of high accuracy, low storage requirements, and rapid inference times on both CPUs and GPUs. Variations of this version, with only up to 597 trainable parameters, demonstrated the best trade-off in terms of deployment feasibility, accuracy, and computational efficiency. These results demonstrate the potential of lightweight artificial intelligence (AI) models for real-time hyperspectral image processing, supporting the development of on-board satellite AI systems for space-based applications. 

**Abstract (ZH)**: 云和云影掩模是高光谱卫星成像中的一个关键预处理步骤，能够提取高质量的分析数据。本研究评估了多种机器学习方法，包括梯度提升方法如XGBoost和LightGBM以及卷积神经网络（CNNs）。所有增强和CNN模型的准确性均超过93%。在所研究的模型中，具有特征降维的CNN模型在准确率、存储需求和在CPU和GPU上的快速推理速度方面表现出最佳平衡。其变体版本仅具有最多597个可训练参数，在部署可行性、准确性和计算效率方面达到了最佳权衡。这些结果表明轻量级人工智能（AI）模型在实时高光谱图像处理中的潜力，支持空间基应用中机载AI系统的开发。 

---
# Principled Foundations for Preference Optimization 

**Title (ZH)**: 原则性的基础理论用于偏好优化 

**Authors**: Wenxuan Zhou, Shujian Zhang, Brice Magdalou, John Lambert, Ehsan Amid, Richard Nock, Andrew Hard  

**Link**: [PDF](https://arxiv.org/pdf/2507.07855)  

**Abstract**: In this paper, we show that direct preference optimization (DPO) is a very specific form of a connection between two major theories in the ML context of learning from preferences: loss functions (Savage) and stochastic choice (Doignon-Falmagne and Machina). The connection is established for all of Savage's losses and at this level of generality, (i) it includes support for abstention on the choice theory side, (ii) it includes support for non-convex objectives on the ML side, and (iii) it allows to frame for free some notable extensions of the DPO setting, including margins and corrections for length. Getting to understand how DPO operates from a general principled perspective is crucial because of the huge and diverse application landscape of models, because of the current momentum around DPO, but also -- and importantly -- because many state of the art variations on DPO definitely occupy a small region of the map that we cover. It also helps to understand the pitfalls of departing from this map, and figure out workarounds. 

**Abstract (ZH)**: 在本文中，我们展示了直接偏好优化（DPO）是连接机器学习（ML）中从偏好学习场景下的两类主要理论——损失函数（Savage）和随机选择（Doignon-Falmagne和Machina）——的非常具体的形式。该连接涵盖了Savage的所有损失函数，并在这一普遍性的水平上，（i）包括选择理论中的弃权支持，（ii）包括ML方面的非凸优化目标支持，（iii）允许免费框架一些DPO设置的显著扩展，包括边际和长度校正。从一般原则的角度理解DPO是如何运作的至关重要，这不仅是因为模型的应用场景极其广泛和多样，而且因为目前DPO研究的势头正盛，更重要的是——许多最先进的DPO变体肯定占据我们在覆盖范围内的一部分区域，这也有助于理解偏离这一框架的漏洞，并找出解决方案。 

---
# An Enhanced Classification Method Based on Adaptive Multi-Scale Fusion for Long-tailed Multispectral Point Clouds 

**Title (ZH)**: 基于自适应多尺度融合的长尾多光谱点云增强分类方法 

**Authors**: TianZhu Liu, BangYan Hu, YanFeng Gu, Xian Li, Aleksandra Pižurica  

**Link**: [PDF](https://arxiv.org/pdf/2412.11407)  

**Abstract**: Multispectral point cloud (MPC) captures 3D spatial-spectral information from the observed scene, which can be used for scene understanding and has a wide range of applications. However, most of the existing classification methods were extensively tested on indoor datasets, and when applied to outdoor datasets they still face problems including sparse labeled targets, differences in land-covers scales, and long-tailed distributions. To address the above issues, an enhanced classification method based on adaptive multi-scale fusion for MPCs with long-tailed distributions is proposed. In the training set generation stage, a grid-balanced sampling strategy is designed to reliably generate training samples from sparse labeled datasets. In the feature learning stage, a multi-scale feature fusion module is proposed to fuse shallow features of land-covers at different scales, addressing the issue of losing fine features due to scale variations in land-covers. In the classification stage, an adaptive hybrid loss module is devised to utilize multi-classification heads with adaptive weights to balance the learning ability of different classes, improving the classification performance of small classes due to various-scales and long-tailed distributions in land-covers. Experimental results on three MPC datasets demonstrate the effectiveness of the proposed method compared with the state-of-the-art methods. 

**Abstract (ZH)**: 基于自适应多尺度融合的宽尾分布多光谱点云增强分类方法 

---
