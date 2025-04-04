# A Chefs KISS -- Utilizing semantic information in both ICP and SLAM framework 

**Title (ZH)**: A Chefs KISS — 利用ICP和SLAM框架中的语义信息 

**Authors**: Sven Ochs, Marc Heinrich, Philip Schörner, Marc René Zofka, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2504.02086)  

**Abstract**: For utilizing autonomous vehicle in urban areas a reliable localization is needed. Especially when HD maps are used, a precise and repeatable method has to be chosen. Therefore accurate map generation but also re-localization against these maps is necessary. Due to best 3D reconstruction of the surrounding, LiDAR has become a reliable modality for localization. The latest LiDAR odometry estimation are based on iterative closest point (ICP) approaches, namely KISS-ICP and SAGE-ICP. We extend the capabilities of KISS-ICP by incorporating semantic information into the point alignment process using a generalizable approach with minimal parameter tuning. This enhancement allows us to surpass KISS-ICP in terms of absolute trajectory error (ATE), the primary metric for map accuracy. Additionally, we improve the Cartographer mapping framework to handle semantic information. Cartographer facilitates loop closure detection over larger areas, mitigating odometry drift and further enhancing ATE accuracy. By integrating semantic information into the mapping process, we enable the filtering of specific classes, such as parked vehicles, from the resulting map. This filtering improves relocalization quality by addressing temporal changes, such as vehicles being moved. 

**Abstract (ZH)**: 利用自动驾驶车辆在城市区域可靠定位的需求，特别是在使用高清地图时，需要选择一种精确且可重复的方法。因此，准确的地图生成和针对这些地图的重定位都是必要的。为了实现最佳的周围环境三维重建，LiDAR已成为定位的一种可靠方式。最新的LiDAR里程计估计基于迭代最近点（ICP）方法，如KISS-ICP和SAGE-ICP。我们通过采用一种可泛化的最小参数调优方法，将语义信息整合到点对齐过程中，扩展了KISS-ICP的功能。这一增强使我们在绝对轨迹误差（ATE）方面超越了KISS-ICP，ATE是衡量地图精度的主要指标。此外，我们还改进了Cartographer建图框架以处理语义信息。Cartographer能够通过更大范围的环路闭合检测，减轻里程计漂移并进一步提高ATE精度。通过将语义信息整合到建图过程中，我们能够从最终地图中过滤出特定类别的对象，如停驶的车辆。这种过滤提高了重定位质量，能够应对例如车辆移动等时间变化。 

---
# A Concise Survey on Lane Topology Reasoning for HD Mapping 

**Title (ZH)**: 一种关于高清地图中车道拓扑推理的简要调研 

**Authors**: Yi Yao, Miao Fan, Shengtong Xu, Haoyi Xiong, Xiangzeng Liu, Wenbo Hu, Wenbing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01989)  

**Abstract**: Lane topology reasoning techniques play a crucial role in high-definition (HD) mapping and autonomous driving applications. While recent years have witnessed significant advances in this field, there has been limited effort to consolidate these works into a comprehensive overview. This survey systematically reviews the evolution and current state of lane topology reasoning methods, categorizing them into three major paradigms: procedural modeling-based methods, aerial imagery-based methods, and onboard sensors-based methods. We analyze the progression from early rule-based approaches to modern learning-based solutions utilizing transformers, graph neural networks (GNNs), and other deep learning architectures. The paper examines standardized evaluation metrics, including road-level measures (APLS and TLTS score), and lane-level metrics (DET and TOP score), along with performance comparisons on benchmark datasets such as OpenLane-V2. We identify key technical challenges, including dataset availability and model efficiency, and outline promising directions for future research. This comprehensive review provides researchers and practitioners with insights into the theoretical frameworks, practical implementations, and emerging trends in lane topology reasoning for HD mapping applications. 

**Abstract (ZH)**: 车道拓扑推理技术在高-definition (HD) 地图和自动驾驶应用中扮演着至关重要的角色。尽管近年来该领域取得了显著进展，但缺乏对该领域工作的综合概述。本文系统回顾了车道拓扑推理方法的发展及其当前状态，将其分为三大范式：基于过程建模的方法、基于航空影像的方法以及基于车载传感器的方法。我们分析了从早期基于规则的方法到现代基于学习的解决方案（利用变压器、图神经网络和其他深度学习架构）的发展过程。文章研究了标准化评估指标，包括道路级别的指标（如APLS和TLTS得分）和车道级别的指标（如DET和TOP得分），并在基准数据集（如OpenLane-V2）上进行了性能比较。我们识别了关键技术挑战，包括数据集可用性和模型效率，并概述了未来研究的有希望的方向。本文为研究人员和 practitioners 提供了关于 HD 地图应用中车道拓扑推理的理论框架、实际实现和新兴趋势的见解。 

---
# Multi-Dimensional AGV Path Planning in 3D Warehouses Using Ant Colony Optimization and Advanced Neural Networks 

**Title (ZH)**: 基于蚁群优化和高级神经网络的三维仓库多维度AGV路径规划 

**Authors**: Bo Zhang, Xiubo Liang, Wei Song, Yulu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.01985)  

**Abstract**: Within modern warehouse scenarios, the rapid expansion of e-commerce and increasingly complex, multi-level storage environments have exposed the limitations of traditional AGV (Automated Guided Vehicle) path planning methods--often reliant on static 2D models and expert-tuned heuristics that struggle to handle dynamic traffic and congestion. Addressing these limitations, this paper introduces a novel AGV path planning approach for 3D warehouse environments that leverages a hybrid framework combining ACO (Ant Colony Optimization) with deep learning models, called NAHACO (Neural Adaptive Heuristic Ant Colony Optimization). NAHACO integrates three key innovations: first, an innovative heuristic algorithm for 3D warehouse cargo modeling using multidimensional tensors, which addresses the challenge of achieving superior heuristic accuracy; second, integration of a congestion-aware loss function within the ACO framework to adjust path costs based on traffic and capacity constraints, called CARL (Congestion-Aware Reinforce Loss), enabling dynamic heuristic calibration for optimizing ACO-based path planning; and third, an adaptive attention mechanism that captures multi-scale spatial features, thereby addressing dynamic heuristic calibration for further optimization of ACO-based path planning and AGV navigation. NAHACO significantly boosts path planning efficiency, yielding faster computation times and superior performance over both vanilla and state-of-the-art methods, while automatically adapting to warehouse constraints for real-time optimization. NAHACO outperforms state-of-the-art methods, lowering the total cost by up to 24.7% on TSP benchmarks. In warehouse tests, NAHACO cuts cost by up to 41.5% and congestion by up to 56.1% compared to previous methods. 

**Abstract (ZH)**: 基于ACO与深度学习的神经自适应蚁群优化算法（NAHACO）在3D仓储环境中的路径规划方法 

---
# X-Capture: An Open-Source Portable Device for Multi-Sensory Learning 

**Title (ZH)**: X-Capture：一个开源便携式多感官学习设备 

**Authors**: Samuel Clarke, Suzannah Wistreich, Yanjie Ze, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02318)  

**Abstract**: Understanding objects through multiple sensory modalities is fundamental to human perception, enabling cross-sensory integration and richer comprehension. For AI and robotic systems to replicate this ability, access to diverse, high-quality multi-sensory data is critical. Existing datasets are often limited by their focus on controlled environments, simulated objects, or restricted modality pairings. We introduce X-Capture, an open-source, portable, and cost-effective device for real-world multi-sensory data collection, capable of capturing correlated RGBD images, tactile readings, and impact audio. With a build cost under $1,000, X-Capture democratizes the creation of multi-sensory datasets, requiring only consumer-grade tools for assembly. Using X-Capture, we curate a sample dataset of 3,000 total points on 500 everyday objects from diverse, real-world environments, offering both richness and variety. Our experiments demonstrate the value of both the quantity and the sensory breadth of our data for both pretraining and fine-tuning multi-modal representations for object-centric tasks such as cross-sensory retrieval and reconstruction. X-Capture lays the groundwork for advancing human-like sensory representations in AI, emphasizing scalability, accessibility, and real-world applicability. 

**Abstract (ZH)**: 通过多种感知模态理解物体是人类感知的基本要素，能够实现跨感知整合和更丰富的理解。为了使AI和机器人系统复制这一能力，获取多样化的高质量多感知数据至关重要。现有数据集往往受限于其对受控环境、模拟对象或模态配对限制的聚焦。我们介绍了X-Capture，这是一种开源、便携且经济高效的设备，能够实时捕捉相关的RGBD图像、触觉读数和冲击声音。X-Capture的构建成本低于1000美元，使多感知数据集的创建民主化，仅需使用消费级工具即可组装。使用X-Capture，我们从多样化的实际环境中收集了500个日常物体的3000个总点数据集，提供丰富性和多样性。我们的实验展示了我们的数据在数量和感官广度方面的价值，对于物体中心任务如跨感知检索和重建的多模态表示的预训练和微调具有重要意义。X-Capture为推进具有人类感知代表性的AI奠定了基础，强调可扩展性、可访问性和实际应用。 

---
# Do Two AI Scientists Agree? 

**Title (ZH)**: 两AI科学家意见一致吗？ 

**Authors**: Xinghong Fu, Ziming Liu, Max Tegmark  

**Link**: [PDF](https://arxiv.org/pdf/2504.02822)  

**Abstract**: When two AI models are trained on the same scientific task, do they learn the same theory or two different theories? Throughout history of science, we have witnessed the rise and fall of theories driven by experimental validation or falsification: many theories may co-exist when experimental data is lacking, but the space of survived theories become more constrained with more experimental data becoming available. We show the same story is true for AI scientists. With increasingly more systems provided in training data, AI scientists tend to converge in the theories they learned, although sometimes they form distinct groups corresponding to different theories. To mechanistically interpret what theories AI scientists learn and quantify their agreement, we propose MASS, Hamiltonian-Lagrangian neural networks as AI Scientists, trained on standard problems in physics, aggregating training results across many seeds simulating the different configurations of AI scientists. Our findings suggests for AI scientists switch from learning a Hamiltonian theory in simple setups to a Lagrangian formulation when more complex systems are introduced. We also observe strong seed dependence of the training dynamics and final learned weights, controlling the rise and fall of relevant theories. We finally demonstrate that not only can our neural networks aid interpretability, it can also be applied to higher dimensional problems. 

**Abstract (ZH)**: 当两个AI模型在同一科学任务上训练时，它们学习相同的理论还是不同的理论？随着科学史的发展，我们见证了由实验验证或证伪驱动的理论兴衰：当实验数据缺乏时，可能存在多种理论共存，但随着可用实验数据的增加，幸存理论的空间变得更加受限。我们展示在AI科学家的情况也是如此。随着训练数据中系统数量的不断增加，AI科学家倾向于在其学习的理论中趋同，尽管有时它们会形成不同的群体，对应于不同的理论。为了机械地解释AI科学家学习的理论及其一致程度，我们提出了MASS，即基于哈密顿-拉格朗日神经网络的AI科学家，并在物理学的标准问题上进行训练，汇总来自许多随机种子的训练结果，模拟不同配置的AI科学家。我们的研究结果表明，当从简单配置到更复杂系统时，AI科学家的学习从哈密顿理论转变为拉格朗日形式。我们还观察到训练动力学和最终学习权重的强烈随机种子依赖性，控制相关理论的兴衰。最后，我们证明我们的神经网络不仅有助于提高可解释性，还可以应用于更高维度的问题。 

---
# Responsible Development of Offensive AI 

**Title (ZH)**: 负责任地开发进攻性AI 

**Authors**: Ryan Marinelli  

**Link**: [PDF](https://arxiv.org/pdf/2504.02701)  

**Abstract**: As AI advances, broader consensus is needed to determine research priorities. This endeavor discusses offensive AI and provides guidance by leveraging Sustainable Development Goals (SDGs) and interpretability techniques. The objective is to more effectively establish priorities that balance societal benefits against risks. The two forms of offensive AI evaluated in this study are vulnerability detection agents, which solve Capture- The-Flag challenges, and AI-powered malware. 

**Abstract (ZH)**: 随着人工智能的发展，需要形成更广泛的共识来确定研究优先级。本研究探讨了攻击型人工智能，并通过可持续发展目标（SDGs）和可解释性技术提供指导，旨在更有效地确立平衡社会利益与风险的优先级。本研究评估了两种形式的攻击型人工智能：漏洞检测代理，它们解决Capture-The-Flag挑战，以及人工智能驱动的恶意软件。 

---
# Reasoning Inconsistencies and How to Mitigate Them in Deep Learning 

**Title (ZH)**: 在深度学习中推理不一致性和如何减轻它们 

**Authors**: Erik Arakelyan  

**Link**: [PDF](https://arxiv.org/pdf/2504.02577)  

**Abstract**: The recent advancements in Deep Learning models and techniques have led to significant strides in performance across diverse tasks and modalities. However, while the overall capabilities of models show promising growth, our understanding of their internal reasoning processes remains limited, particularly concerning systematic inconsistencies or errors patterns of logical or inferential flaws. These inconsistencies may manifest as contradictory outputs, failure to generalize across similar tasks, or erroneous conclusions in specific contexts. Even detecting and measuring such reasoning discrepancies is challenging, as they may arise from opaque internal procedures, biases and imbalances in training data, or the inherent complexity of the task. Without effective methods to detect, measure, and mitigate these errors, there is a risk of deploying models that are biased, exploitable, or logically unreliable. This thesis aims to address these issues by producing novel methods for deep learning models that reason over knowledge graphs, natural language, and images. The thesis contributes two techniques for detecting and quantifying predictive inconsistencies originating from opaque internal procedures in natural language and image processing models. To mitigate inconsistencies from biases in training data, this thesis presents a data efficient sampling method to improve fairness and performance and a synthetic dataset generation approach in low resource scenarios. Finally, the thesis offers two techniques to optimize the models for complex reasoning tasks. These methods enhance model performance while allowing for more faithful and interpretable exploration and exploitation during inference. Critically, this thesis provides a comprehensive framework to improve the robustness, fairness, and interpretability of deep learning models across diverse tasks and modalities. 

**Abstract (ZH)**: 近期深度学习模型和方法的进步在跨多种任务和模态中取得了显著的性能提升。然而，尽管模型的整体能力显示出有希望的增长，我们对其内部推理过程的理解仍然有限，尤其是在系统性不一致或推理错误模式方面的理解更为有限。这些不一致可能表现为矛盾的输出、无法泛化到类似任务或在特定上下文中得出错误结论。即使检测和量化这些推理差异也很具有挑战性，因为它们可能来自不透明的内部程序、训练数据中的偏差和不平衡或任务本身的固有复杂性。缺乏有效的检测、量化和缓解这些错误的方法，存在部署存在偏差、可利用或逻辑上不可靠模型的风险。本论文旨在通过开发新的方法来解决这些问题，这些方法可以对知识图谱、自然语言和图像进行推理。论文提出了两种技术来检测和量化自然语言和图像处理模型中源自不透明内部程序的预测不一致。为减轻由训练数据偏差引起的不一致性，论文提出了一种高效的数据采样方法以提高公平性和性能，并在低资源场景中提出了一种合成数据集生成方法。最后，论文提供了两种技术来优化模型以适应复杂的推理任务。这些方法在提高模型性能的同时，还允许在推理过程中进行更忠实和可解释的探索与利用。本论文提供了一个全面的框架，以提高不同任务和模态下的深度学习模型的稳健性、公平性和可解释性。 

---
# We Need Improved Data Curation and Attribution in AI for Scientific Discovery 

**Title (ZH)**: 我们需要在科学发现中的AI领域提升数据整理和归属规范。 

**Authors**: Mara Graziani, Antonio Foncubierta, Dimitrios Christofidellis, Irina Espejo-Morales, Malina Molnar, Marvin Alberts, Matteo Manica, Jannis Born  

**Link**: [PDF](https://arxiv.org/pdf/2504.02486)  

**Abstract**: As the interplay between human-generated and synthetic data evolves, new challenges arise in scientific discovery concerning the integrity of the data and the stability of the models. In this work, we examine the role of synthetic data as opposed to that of real experimental data for scientific research. Our analyses indicate that nearly three-quarters of experimental datasets available on open-access platforms have relatively low adoption rates, opening new opportunities to enhance their discoverability and usability by automated methods. Additionally, we observe an increasing difficulty in distinguishing synthetic from real experimental data. We propose supplementing ongoing efforts in automating synthetic data detection by increasing the focus on watermarking real experimental data, thereby strengthening data traceability and integrity. Our estimates suggest that watermarking even less than half of the real world data generated annually could help sustain model robustness, while promoting a balanced integration of synthetic and human-generated content. 

**Abstract (ZH)**: 随着人类生成数据与合成数据的互动演变，在科学发现中的数据完整性和模型稳定性面临新的挑战。本研究探讨了合成数据在科学研究中与真实实验数据的角色差异。我们的分析表明，近四分之三在开放访问平台上可用的实验数据集的采用率相对较低，这为通过自动化方法增强其可发现性和可用性提供了新的机会。此外，我们观察到区分合成数据与真实实验数据的难度日益增加。我们建议在自动化合成数据检测的同时，增加真实实验数据水印化的重点，从而加强数据的可追溯性和完整性。我们的估算表明，即使每年对不到一半的真实数据进行水印化，也有助于维持模型的稳健性，并促进合成数据与人类生成内容的平衡整合。 

---
# BOOST: Bootstrapping Strategy-Driven Reasoning Programs for Program-Guided Fact-Checking 

**Title (ZH)**: BOOST: 基于策略驱动推理程序的程序指导事实核查bootstrapping方法 

**Authors**: Qisheng Hu, Quanyu Long, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02467)  

**Abstract**: Program-guided reasoning has shown promise in complex claim fact-checking by decomposing claims into function calls and executing reasoning programs. However, prior work primarily relies on few-shot in-context learning (ICL) with ad-hoc demonstrations, which limit program diversity and require manual design with substantial domain knowledge. Fundamentally, the underlying principles of effective reasoning program generation still remain underexplored, making it challenging to construct effective demonstrations. To address this, we propose BOOST, a bootstrapping-based framework for few-shot reasoning program generation. BOOST explicitly integrates claim decomposition and information-gathering strategies as structural guidance for program generation, iteratively refining bootstrapped demonstrations in a strategy-driven and data-centric manner without human intervention. This enables a seamless transition from zero-shot to few-shot strategic program-guided learning, enhancing interpretability and effectiveness. Experimental results show that BOOST outperforms prior few-shot baselines in both zero-shot and few-shot settings for complex claim verification. 

**Abstract (ZH)**: 基于程序引导的推理在复杂声明事实核查中的潜在应用通过将声明分解为函数调用并执行推理程序展现了一定前景。然而，现有工作主要依赖少量上下文学习（ICL）和 ad-hoc 展示，这限制了程序多样性并需要大量领域知识的ручное проектирование.从根本上说，有效的推理程序生成的基本原理仍然未得到充分探索，使得构建有效展示变得具有挑战性。为解决这一问题，我们提出 BOOST——一种基于自助法的少量样本推理程序生成框架。BOOST 显式地将声明分解和信息收集策略整合为程序生成的结构引导，在策略驱动和数据中心化的迭代方式中逐步优化自助展示，无需人工干预，从而实现从零样本到少量样本战略程序引导学习的无缝过渡，增强可解释性和有效性。实验结果表明，BOOST 在复杂声明验证的零样本和少量样本设置中均优于先前的少量样本基线方法。 

---
# How Artificial Intelligence Leads to Knowledge Why: An Inquiry Inspired by Aristotle's Posterior Analytics 

**Title (ZH)**: 人工智能如何产生知识：受亚里士多德后分析篇启发的探究 

**Authors**: Guus Eelink, Kilian Rückschloß, Felix Weitkämper  

**Link**: [PDF](https://arxiv.org/pdf/2504.02430)  

**Abstract**: Bayesian networks and causal models provide frameworks for handling queries about external interventions and counterfactuals, enabling tasks that go beyond what probability distributions alone can address. While these formalisms are often informally described as capturing causal knowledge, there is a lack of a formal theory characterizing the type of knowledge required to predict the effects of external interventions. This work introduces the theoretical framework of causal systems to clarify Aristotle's distinction between knowledge that and knowledge why within artificial intelligence. By interpreting existing artificial intelligence technologies as causal systems, it investigates the corresponding types of knowledge. Furthermore, it argues that predicting the effects of external interventions is feasible only with knowledge why, providing a more precise understanding of the knowledge necessary for such tasks. 

**Abstract (ZH)**: 贝叶斯网络和因果模型提供了处理关于外部干预和反事实查询的框架，使任务超越了仅凭概率分布所能实现的范围。虽然这些形式主义通常非正式地被认为捕捉了因果知识，但缺乏对预测外部干预效果所需类型的知识的正式理论。本文引入因果系统理论框架来澄清人工智能中的知识that与知识why之间的区别。通过将现有的人工智能技术解释为因果系统，它探讨了相应的知识类型，并进一步认为，仅凭知识that无法预测外部干预的效果，预测外部干预效果仅在具备知识why的前提下才可行，从而为这类任务所需的知识提供了更精确的理解。 

---
# Engineering Artificial Intelligence: Framework, Challenges, and Future Direction 

**Title (ZH)**: 工程化人工智能：框架、挑战及未来方向 

**Authors**: Jay Lee, Hanqi Su, Dai-Yan Ji, Takanobu Minami  

**Link**: [PDF](https://arxiv.org/pdf/2504.02269)  

**Abstract**: Over the past ten years, the application of artificial intelligence (AI) and machine learning (ML) in engineering domains has gained significant popularity, showcasing their potential in data-driven contexts. However, the complexity and diversity of engineering problems often require the development of domain-specific AI approaches, which are frequently hindered by a lack of systematic methodologies, scalability, and robustness during the development process. To address this gap, this paper introduces the "ABCDE" as the key elements of Engineering AI and proposes a unified, systematic engineering AI ecosystem framework, including eight essential layers, along with attributes, goals, and applications, to guide the development and deployment of AI solutions for specific engineering needs. Additionally, key challenges are examined, and nine future research directions are highlighted. By providing a comprehensive perspective, this paper aims to advance the strategic implementation of AI, fostering the development of next-generation engineering AI solutions. 

**Abstract (ZH)**: 过去十年，人工智能（AI）和机器学习（ML）在工程领域中的应用获得了显著 popularity，并在数据驱动的背景下展示了其潜力。然而，工程问题的复杂性和多样性往往需要开发特定领域的 AI 方法，这在开发过程中经常受到缺乏系统方法论、可扩展性和鲁棒性等因素的阻碍。为解决这一差距，本文提出了“ABCDE”作为工程 AI 的关键要素，并提出了一种统一的、系统的工程 AI 生态系统框架，包括八个基本层次及其属性、目标和应用，以指导特定工程需求的 AI 解决方案的开发和部署。此外，本文还探讨了主要挑战，并指出了九个未来研究方向。通过提供一个全面的视角，本文旨在促进 AI 的战略实施，推动下一代工程 AI 解决方案的发展。 

---
# Epistemic Closure and the Irreversibility of Misalignment: Modeling Systemic Barriers to Alignment Innovation 

**Title (ZH)**: 知识闭合与错配的不可逆性：建模系统性对齐创新障碍 

**Authors**: Andy Williams  

**Link**: [PDF](https://arxiv.org/pdf/2504.02058)  

**Abstract**: Efforts to ensure the safe development of artificial general intelligence (AGI) often rely on consensus-based alignment approaches grounded in axiomatic formalism, interpretability, and empirical validation. However, these methods may be structurally unable to recognize or incorporate novel solutions that fall outside their accepted epistemic frameworks. This paper introduces a functional model of epistemic closure, in which cognitive, institutional, social, and infrastructural filters combine to make many alignment proposals illegible to existing evaluation systems. We present a weighted closure model supported by both theoretical and empirical sources, including a meta-analysis performed by an AI system on patterns of rejection and non-engagement with a framework for decentralized collective intelligence (DCI). We argue that the recursive failure to assess models like DCI is not just a sociological oversight but a structural attractor, mirroring the very risks of misalignment we aim to avoid in AGI. Without the adoption of DCI or a similarly recursive model of epistemic correction, we may be on a predictable path toward irreversible misalignment. The development and acceptance of this paper, first through simulated review and then through formal channels, provide a case study supporting its central claim: that epistemic closure can only be overcome by recursive modeling of the constraints that sustain it. 

**Abstract (ZH)**: 确保通用人工智能(AGI)安全发展的努力往往依赖于基于公理形式主义、可解释性和经验验证的共识导向对齐方法。然而，这些方法可能在结构上无法识别或纳入超出其接受的认识框架的新型解决方案。本文引入了一种功能模型，以认知、机构、社会和基础设施过滤器组合的方式，使得许多对齐提案对现有评估系统来说变得不可读。我们提出了一个基于理论和实证来源的加权闭合模型，包括一个AI系统对去中心化集体智能(DCI)框架的拒斥和非参与模式进行的元分析。我们认为，对类似于DCI的模型的递归评估失败不仅是社会学上的疏忽，也是结构上的吸引子，这与我们试图避免的AGI对齐风险相呼应。如果不采用DCI或类似具有递归性的认识纠正模型，我们可能走上一条不可预测的不可逆对齐偏差之路。本文的发展和接受过程，首先通过模拟评审，然后通过正式渠道进行，提供了一个案例研究，支持其核心观点：只有通过递归建模才能克服认识闭合。 

---
# On Vanishing Variance in Transformer Length Generalization 

**Title (ZH)**: Transformer 长度泛化的消失方差现象 

**Authors**: Ruining Li, Gabrijel Boduljak, Jensen, Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.02827)  

**Abstract**: It is a widely known issue that Transformers, when trained on shorter sequences, fail to generalize robustly to longer ones at test time. This raises the question of whether Transformer models are real reasoning engines, despite their impressive abilities in mathematical problem solving and code synthesis. In this paper, we offer a vanishing variance perspective on this issue. To the best of our knowledge, we are the first to demonstrate that even for today's frontier models, a longer sequence length results in a decrease in variance in the output of the multi-head attention modules. On the argmax retrieval and dictionary lookup tasks, our experiments show that applying layer normalization after the attention outputs leads to significantly better length generalization. Our analyses attribute this improvement to a reduction-though not a complete elimination-of the distribution shift caused by vanishing variance. 

**Abstract (ZH)**: 预训练序列较短时Transformer模型在长序列上的泛化能力不足：消失方差视角下的分析 

---
# Towards Green AI-Native Networks: Evaluation of Neural Circuit Policy for Estimating Energy Consumption of Base Stations 

**Title (ZH)**: 面向绿色AI原生网络：基站能耗估算的神经电路策略评估 

**Authors**: Selim Ickin, Shruti Bothe, Aman Raparia, Nitin Khanna, Erik Sanders  

**Link**: [PDF](https://arxiv.org/pdf/2504.02781)  

**Abstract**: Optimization of radio hardware and AI-based network management software yield significant energy savings in radio access networks. The execution of underlying Machine Learning (ML) models, which enable energy savings through recommended actions, may require additional compute and energy, highlighting the opportunity to explore and adopt accurate and energy-efficient ML technologies. This work evaluates the novel use of sparsely structured Neural Circuit Policies (NCPs) in a use case to estimate the energy consumption of base stations. Sparsity in ML models yields reduced memory, computation and energy demand, hence facilitating a low-cost and scalable solution. We also evaluate the generalization capability of NCPs in comparison to traditional and widely used ML models such as Long Short Term Memory (LSTM), via quantifying their sensitivity to varying model hyper-parameters (HPs). NCPs demonstrated a clear reduction in computational overhead and energy consumption. Moreover, results indicated that the NCPs are robust to varying HPs such as number of epochs and neurons in each layer, making them a suitable option to ease model management and to reduce energy consumption in Machine Learning Operations (MLOps) in telecommunications. 

**Abstract (ZH)**: 基于稀疏结构神经电路策略的无线接入网络能耗优化研究 

---
# RBR4DNN: Requirements-based Testing of Neural Networks 

**Title (ZH)**: 基于需求的神经网络测试：RBR4DNN 

**Authors**: Nusrat Jahan Mozumder, Felipe Toledo, Swaroopa Dola, Matthew B. Dwyer  

**Link**: [PDF](https://arxiv.org/pdf/2504.02737)  

**Abstract**: Deep neural network (DNN) testing is crucial for the reliability and safety of critical systems, where failures can have severe consequences. Although various techniques have been developed to create robustness test suites, requirements-based testing for DNNs remains largely unexplored -- yet such tests are recognized as an essential component of software validation of critical systems. In this work, we propose a requirements-based test suite generation method that uses structured natural language requirements formulated in a semantic feature space to create test suites by prompting text-conditional latent diffusion models with the requirement precondition and then using the associated postcondition to define a test oracle to judge outputs of the DNN under test. We investigate the approach using fine-tuned variants of pre-trained generative models. Our experiments on the MNIST, CelebA-HQ, ImageNet, and autonomous car driving datasets demonstrate that the generated test suites are realistic, diverse, consistent with preconditions, and capable of revealing faults. 

**Abstract (ZH)**: 基于需求的深度神经网络测试套件生成方法 

---
# SCMPPI: Supervised Contrastive Multimodal Framework for Predicting Protein-Protein Interactions 

**Title (ZH)**: SCMPPI：监督对比多模态框架用于预测蛋白质-蛋白质相互作用 

**Authors**: Shengrui XU, Tianchi Lu, Zikun Wang, Jixiu Zhai, Jingwan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02698)  

**Abstract**: Protein-Protein Interaction (PPI) prediction is a key task in uncovering cellular functional networks and disease mechanisms. However, traditional experimental methods are time-consuming and costly, and existing computational models face challenges in cross-modal feature fusion, robustness, and false-negative suppression. In this paper, we propose a novel supervised contrastive multimodal framework, SCMPPI, for PPI prediction. By integrating protein sequence features (AAC, DPC, CKSAAP-ESMC) with PPI network topology information (Node2Vec graph embedding), and combining an improved supervised contrastive learning strategy, SCMPPI significantly enhances PPI prediction performance. For the PPI task, SCMPPI introduces a negative sample filtering mechanism and modifies the contrastive loss function, effectively optimizing multimodal features. Experiments on eight benchmark datasets, including yeast, human, and this http URL, show that SCMPPI outperforms existing state-of-the-art methods (such as DF-PPI and TAGPPI) in key metrics such as accuracy ( 98.01%) and AUC (99.62%), and demonstrates strong generalization in cross-species prediction (AUC > 99% on multi-species datasets). Furthermore, SCMPPI has been successfully applied to CD9 networks, the Wnt pathway, and cancer-specific networks, providing a reliable tool for disease target discovery. This framework also offers a new paradigm for multimodal biological information fusion and contrastive learning in collaborative optimization for various combined predictions. 

**Abstract (ZH)**: 蛋白质-蛋白质相互作用（PPI）预测是揭示细胞功能网络和疾病机制的关键任务。然而，传统的实验方法耗时且成本高，现有的计算模型在跨模态特征融合、鲁棒性和减少假阴性方面面临挑战。本文提出了一种新的监督对比多模态框架SCMPPI，用于PPI预测。通过将蛋白质序列特征（AAC, DPC, CKSAAP-ESMC）与PPI网络拓扑信息（Node2Vec图嵌入）相结合，并采用改进的监督对比学习策略，SCMPPI显著提高了PPI预测性能。在酵母、人类以及其他基准数据集中，SCMPPI在关键指标如准确率（98.01%）和AUC（99.62%）上优于现有最先进的方法（如DF-PPI和TAGPPI），并在跨物种预测中展现了强大的泛化能力（多物种数据集中AUC > 99%）。此外，SCMPPI已成功应用于CD9网络、Wnt信号通路和癌症特异性网络，为疾病靶标发现提供了可靠的工具。该框架还为多模态生物信息融合和在各种联合预测中的协作优化提供了新的范式。 

---
# STOOD-X methodology: using statistical nonparametric test for OOD Detection Large-Scale datasets enhanced with explainability 

**Title (ZH)**: STOOD-X 方法学：使用统计非参数测试进行OOD检测，增强可解释性的大规模数据集 

**Authors**: Iván Sevillano-García, Julián Luengo, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2504.02685)  

**Abstract**: Out-of-Distribution (OOD) detection is a critical task in machine learning, particularly in safety-sensitive applications where model failures can have serious consequences. However, current OOD detection methods often suffer from restrictive distributional assumptions, limited scalability, and a lack of interpretability. To address these challenges, we propose STOOD-X, a two-stage methodology that combines a Statistical nonparametric Test for OOD Detection with eXplainability enhancements. In the first stage, STOOD-X uses feature-space distances and a Wilcoxon-Mann-Whitney test to identify OOD samples without assuming a specific feature distribution. In the second stage, it generates user-friendly, concept-based visual explanations that reveal the features driving each decision, aligning with the BLUE XAI paradigm. Through extensive experiments on benchmark datasets and multiple architectures, STOOD-X achieves competitive performance against state-of-the-art post hoc OOD detectors, particularly in high-dimensional and complex settings. In addition, its explainability framework enables human oversight, bias detection, and model debugging, fostering trust and collaboration between humans and AI systems. The STOOD-X methodology therefore offers a robust, explainable, and scalable solution for real-world OOD detection tasks. 

**Abstract (ZH)**: Out-of-Distribution (OOD)检测是机器学习中的一个重要任务，特别是在安全性要求高的应用中，模型失败可能会导致严重后果。然而，当前的OOD检测方法往往存在严格的分布假设、可扩展性有限以及缺乏可解释性的问题。为了解决这些挑战，我们提出了一种名为STOOD-X的两阶段方法，该方法结合了统计非参数检验与可解释性增强。在第一阶段，STOOD-X利用特征空间距离和Wilcoxon-Mann-Whitney检验来识别OOD样本，而不假设特定的特征分布。在第二阶段，它生成用户友好的、基于概念的可视化解释，揭示每个决策背后的关键特征，符合BLUE XAI范式。通过在基准数据集和多种架构上的广泛实验，STOOD-X在高维度和复杂设置中实现了与先进事后OOD检测器相当的性能。此外，其可解释性框架可以实现人类监督、偏差检测和模型调试，促进人类与AI系统的信任与协作。因此，STOOD-X方法为现实世界的OOD检测任务提供了一个稳健、可解释且可扩展的解决方案。 

---
# Efficient Model Editing with Task-Localized Sparse Fine-tuning 

**Title (ZH)**: 任务局部化稀疏微调的高效模型编辑 

**Authors**: Leonardo Iurada, Marco Ciccone, Tatiana Tommasi  

**Link**: [PDF](https://arxiv.org/pdf/2504.02620)  

**Abstract**: Task arithmetic has emerged as a promising approach for editing models by representing task-specific knowledge as composable task vectors. However, existing methods rely on network linearization to derive task vectors, leading to computational bottlenecks during training and inference. Moreover, linearization alone does not ensure weight disentanglement, the key property that enables conflict-free composition of task vectors. To address this, we propose TaLoS which allows to build sparse task vectors with minimal interference without requiring explicit linearization and sharing information across tasks. We find that pre-trained models contain a subset of parameters with consistently low gradient sensitivity across tasks, and that sparsely updating only these parameters allows for promoting weight disentanglement during fine-tuning. Our experiments prove that TaLoS improves training and inference efficiency while outperforming current methods in task addition and negation. By enabling modular parameter editing, our approach fosters practical deployment of adaptable foundation models in real-world applications. 

**Abstract (ZH)**: 任务算术已 emergent as a promising approach for editing models by representing task-specific knowledge as composable task vectors.然而，现有方法依赖于网络线性化来推导任务向量，导致训练和推理中的计算瓶颈。此外，仅线性化并不能确保权重解纠缠，这是使任务向量冲突-free composition 的关键属性。为了解决这一问题，我们提出了 TaLoS，它允许构建稀疏的任务向量，同时最小化相互干扰，而无需明确的线性化和跨任务共享信息。我们发现预训练模型中包含一组参数，其在任务间的梯度敏感性始终较低，仅更新这些参数可以促进解纠缠过程在微调中的进行。我们的实验表明，TaLoS 在提高训练和推理效率的同时，在任务添加和否定方面也优于当前方法。通过使参数编辑模块化，我们的方法促进了一体化基础模型的实际部署和应用。 

---
# Learning Geometrically-Informed Lyapunov Functions with Deep Diffeomorphic RBF Networks 

**Title (ZH)**: 学习几何导向的李雅普unov函数的深层 diffeomorphic RBF 网络方法 

**Authors**: Samuel Tesfazgi, Leonhard Sprandl, Sandra Hirche  

**Link**: [PDF](https://arxiv.org/pdf/2504.02607)  

**Abstract**: The practical deployment of learning-based autonomous systems would greatly benefit from tools that flexibly obtain safety guarantees in the form of certificate functions from data. While the geometrical properties of such certificate functions are well understood, synthesizing them using machine learning techniques still remains a challenge. To mitigate this issue, we propose a diffeomorphic function learning framework where prior structural knowledge of the desired output is encoded in the geometry of a simple surrogate function, which is subsequently augmented through an expressive, topology-preserving state-space transformation. Thereby, we achieve an indirect function approximation framework that is guaranteed to remain in the desired hypothesis space. To this end, we introduce a novel approach to construct diffeomorphic maps based on RBF networks, which facilitate precise, local transformations around data. Finally, we demonstrate our approach by learning diffeomorphic Lyapunov functions from real-world data and apply our method to different attractor systems. 

**Abstract (ZH)**: 基于学习的自主系统实用部署将极大地受益于能够灵活从数据中获取安全保证（以证书函数的形式）的工具。虽然此类证书函数的几何性质已被充分了解，但使用机器学习技术合成它们仍然是一项挑战。为缓解这一问题，我们提出了一种 diffeomorphic 函数学习框架，其中将所需输出的先验结构知识编码在简单替代函数的几何结构中，随后通过一个表征性强且保持拓扑结构的态空间变换进行增强。由此，我们实现了一个间接函数逼近框架，可以保证其始终保持在所需的假设空间。为此，我们提出了一种基于 RBF 网络构建 diffeomorphic 映射的新方法，这些方法能够实现数据周围的精确局部变换。最后，我们通过从真实-world 数据中学习 diffeomorphic 李雅普诺夫函数，并将我们的方法应用到不同的吸引子系统，展示了我们的方法。 

---
# Improving Counterfactual Truthfulness for Molecular Property Prediction through Uncertainty Quantification 

**Title (ZH)**: 通过不确定性量化提高分子性质预测的反事实真实性 

**Authors**: Jonas Teufel, Annika Leinweber, Pascal Friederich  

**Link**: [PDF](https://arxiv.org/pdf/2504.02606)  

**Abstract**: Explainable AI (xAI) interventions aim to improve interpretability for complex black-box models, not only to improve user trust but also as a means to extract scientific insights from high-performing predictive systems. In molecular property prediction, counterfactual explanations offer a way to understand predictive behavior by highlighting which minimal perturbations in the input molecular structure cause the greatest deviation in the predicted property. However, such explanations only allow for meaningful scientific insights if they reflect the distribution of the true underlying property -- a feature we define as counterfactual truthfulness. To increase this truthfulness, we propose the integration of uncertainty estimation techniques to filter counterfactual candidates with high predicted uncertainty. Through computational experiments with synthetic and real-world datasets, we demonstrate that traditional uncertainty estimation methods, such as ensembles and mean-variance estimation, can already substantially reduce the average prediction error and increase counterfactual truthfulness, especially for out-of-distribution settings. Our results highlight the importance and potential impact of incorporating uncertainty estimation into explainability methods, especially considering the relatively high effectiveness of low-effort interventions like model ensembles. 

**Abstract (ZH)**: 可解释人工智能（xAI）干预旨在提高复杂黑盒模型的可解释性，不仅提高用户信任，也是从高性能预测系统中提取科学洞察的有效途径。在分子性质预测中，反事实解释通过突出显示哪些最小的输入分子结构变化导致预测性质的最大偏差，提供了一种理解预测行为的方式。然而，这样的解释只有反映真实基础性质的分布时，才允许获得有意义的科学洞察——我们将其定义为反事实真实性。为了增加这种真实性，我们提出将不确定性估计技术集成到反事实候选者筛选中，以过滤出高预测不确定性的情况。通过使用合成和真实世界数据集的计算实验，我们表明传统的不确定性估计方法（如集成和均方差估计）可以显著降低平均预测误差，并提高反事实真实性，尤其是在分布外设置中。我们的结果强调了将不确定性估计纳入解释方法的重要性及其潜在影响，特别是考虑到模型集成等低投入干预措施的相对高效率。 

---
# Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving 

**Title (ZH)**: 多语言问题解决基准：Multi-SWE-bench 

**Authors**: Daoguang Zan, Zhirong Huang, Wei Liu, Hanwu Chen, Linhao Zhang, Shulin Xin, Lu Chen, Qi Liu, Xiaojian Zhong, Aoyan Li, Siyao Liu, Yongsheng Xiao, Liangqiang Chen, Yuyu Zhang, Jing Su, Tianyu Liu, Rui Long, Kai Shen, Liang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02605)  

**Abstract**: The task of issue resolving is to modify a codebase to generate a patch that addresses a given issue. However, existing benchmarks, such as SWE-bench, focus almost exclusively on Python, making them insufficient for evaluating Large Language Models (LLMs) across diverse software ecosystems. To address this, we introduce a multilingual issue-resolving benchmark, called Multi-SWE-bench, covering Java, TypeScript, JavaScript, Go, Rust, C, and C++. It includes a total of 1,632 high-quality instances, which were carefully annotated from 2,456 candidates by 68 expert annotators, ensuring that the benchmark can provide an accurate and reliable evaluation. Based on Multi-SWE-bench, we evaluate a series of state-of-the-art models using three representative methods (Agentless, SWE-agent, and OpenHands) and present a comprehensive analysis with key empirical insights. In addition, we launch a Multi-SWE-RL open-source community, aimed at building large-scale reinforcement learning (RL) training datasets for issue-resolving tasks. As an initial contribution, we release a set of 4,723 well-structured instances spanning seven programming languages, laying a solid foundation for RL research in this domain. More importantly, we open-source our entire data production pipeline, along with detailed tutorials, encouraging the open-source community to continuously contribute and expand the dataset. We envision our Multi-SWE-bench and the ever-growing Multi-SWE-RL community as catalysts for advancing RL toward its full potential, bringing us one step closer to the dawn of AGI. 

**Abstract (ZH)**: 一种多语言问题解决基准Multi-SWE-bench及其在大语言模型评价中的应用 

---
# Knowledge Graph Completion with Mixed Geometry Tensor Factorization 

**Title (ZH)**: 混合几何张量分解的知识图谱补全 

**Authors**: Viacheslav Yusupov, Maxim Rakhuba, Evgeny Frolov  

**Link**: [PDF](https://arxiv.org/pdf/2504.02589)  

**Abstract**: In this paper, we propose a new geometric approach for knowledge graph completion via low rank tensor approximation. We augment a pretrained and well-established Euclidean model based on a Tucker tensor decomposition with a novel hyperbolic interaction term. This correction enables more nuanced capturing of distributional properties in data better aligned with real-world knowledge graphs. By combining two geometries together, our approach improves expressivity of the resulting model achieving new state-of-the-art link prediction accuracy with a significantly lower number of parameters compared to the previous Euclidean and hyperbolic models. 

**Abstract (ZH)**: 本文提出了一种新的几何方法，通过低秩张量近似来完成知识图谱。该方法基于Tucker张量分解，对一个预训练且成熟的欧几里得模型进行扩充，加入了新型双曲交互项。这使得模型能够更细腻地捕捉与现实世界知识图谱更一致的数据分布特性。通过结合两种几何方法，该方法提升了模型的表现力，实现了新的链接预测准确率状态最in，并且所需参数显著少于之前的欧几里得和双曲模型。 

---
# Deep learning for music generation. Four approaches and their comparative evaluation 

**Title (ZH)**: 基于深度学习的音乐生成：四种方法及其比较评价 

**Authors**: Razvan Paroiu, Stefan Trausan-Matu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02586)  

**Abstract**: This paper introduces four different artificial intelligence algorithms for music generation and aims to compare these methods not only based on the aesthetic quality of the generated music but also on their suitability for specific applications. The first set of melodies is produced by a slightly modified visual transformer neural network that is used as a language model. The second set of melodies is generated by combining chat sonification with a classic transformer neural network (the same method of music generation is presented in a previous research), the third set of melodies is generated by combining the Schillinger rhythm theory together with a classic transformer neural network, and the fourth set of melodies is generated using GPT3 transformer provided by OpenAI. A comparative analysis is performed on the melodies generated by these approaches and the results indicate that significant differences can be observed between them and regarding the aesthetic value of them, GPT3 produced the most pleasing melodies, and the newly introduced Schillinger method proved to generate better sounding music than previous sonification methods. 

**Abstract (ZH)**: 本文介绍了四种不同的人工智能算法在音乐生成中的应用，并旨在不仅从所生成音乐的审美质量，而且从其特定应用的适用性方面比较这些方法。第一组旋律由略微修改的视觉变换神经网络生成，该网络作为语言模型使用。第二组旋律通过结合对话声化和经典变换神经网络生成（音乐生成方法在先前研究中有所呈现），第三组旋律通过结合希尔灵格节奏理论与经典变换神经网络生成，第四组旋律使用由OpenAI提供的GPT3变换器生成。对这些方法生成的旋律进行了比较分析，结果表明，这些方法之间存在显著差异，在审美价值方面，GPT3生成的旋律最为悦耳，新引入的希尔灵格方法证明生成的音乐比之前的声化方法听起来更好。 

---
# GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning 

**Title (ZH)**: GPG：一种简单而强大的模型推理强化学习基线 

**Authors**: Xiangxiang Chu, Hailang Huang, Xiao Zhang, Fei Wei, Yong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02546)  

**Abstract**: Reinforcement Learning (RL) can directly enhance the reasoning capabilities of large language models without extensive reliance on Supervised Fine-Tuning (SFT). In this work, we revisit the traditional Policy Gradient (PG) mechanism and propose a minimalist RL approach termed Group Policy Gradient (GPG). Unlike conventional methods, GPG directly optimize the original RL objective, thus obviating the need for surrogate loss functions. As illustrated in our paper, by eliminating both the critic and reference models, and avoiding KL divergence constraints, our approach significantly simplifies the training process when compared to Group Relative Policy Optimization (GRPO). Our approach achieves superior performance without relying on auxiliary techniques or adjustments. Extensive experiments demonstrate that our method not only reduces computational costs but also consistently outperforms GRPO across various unimodal and multimodal tasks. Our code is available at this https URL. 

**Abstract (ZH)**: reinforcement learning可以直接增强大型语言模型的推理能力，而无需大量依赖监督微调。在这项工作中，我们回顾了传统的策略梯度机制，并提出了一种名为群体策略梯度（GPG）的 minimalist RL方法。与传统方法不同，GPG 直接优化原始的 RL 目标，从而避免使用代理损失函数。我们的研究表明，通过消除批评者和参考模型，并避免 KL 散度约束，与群体相对策略优化（GRPO）相比，我们的方法在训练过程中显著简化。我们的方法不依赖于辅助技术或调整，在各种单模态和多模态任务中均表现出更优性能。我们的代码可从该网址获取。 

---
# Fourier Sliced-Wasserstein Embedding for Multisets and Measures 

**Title (ZH)**: 多重集和度量的傅里叶切片Wasserstein嵌入 

**Authors**: Tal Amir, Nadav Dym  

**Link**: [PDF](https://arxiv.org/pdf/2504.02544)  

**Abstract**: We present the Fourier Sliced-Wasserstein (FSW) embedding - a novel method to embed multisets and measures over $\mathbb{R}^d$ into Euclidean space.
Our proposed embedding approximately preserves the sliced Wasserstein distance on distributions, thereby yielding geometrically meaningful representations that better capture the structure of the input. Moreover, it is injective on measures and bi-Lipschitz on multisets - a significant advantage over prevalent methods based on sum- or max-pooling, which are provably not bi-Lipschitz, and, in many cases, not even injective. The required output dimension for these guarantees is near-optimal: roughly $2 N d$, where $N$ is the maximal input multiset size.
Furthermore, we prove that it is impossible to embed distributions over $\mathbb{R}^d$ into Euclidean space in a bi-Lipschitz manner. Thus, the metric properties of our embedding are, in a sense, the best possible.
Through numerical experiments, we demonstrate that our method yields superior multiset representations that improve performance in practical learning tasks. Specifically, we show that (a) a simple combination of the FSW embedding with an MLP achieves state-of-the-art performance in learning the (non-sliced) Wasserstein distance; and (b) replacing max-pooling with the FSW embedding makes PointNet significantly more robust to parameter reduction, with only minor performance degradation even after a 40-fold reduction. 

**Abstract (ZH)**: Fourier裁断 Wasserstein嵌入：一种将实数域上多重集和测度嵌入欧几里得空间的新方法 

---
# Improving User Experience with FAICO: Towards a Framework for AI Communication in Human-AI Co-Creativity 

**Title (ZH)**: 基于FAICO的用户体验提升：迈向人类-人工智能共同创造中的AI通信框架 

**Authors**: Jeba Rezwana, Corey Ford  

**Link**: [PDF](https://arxiv.org/pdf/2504.02526)  

**Abstract**: How AI communicates with humans is crucial for effective human-AI co-creation. However, many existing co-creative AI tools cannot communicate effectively, limiting their potential as collaborators. This paper introduces our initial design of a Framework for designing AI Communication (FAICO) for co-creative AI based on a systematic review of 107 full-length papers. FAICO presents key aspects of AI communication and their impacts on user experience to guide the design of effective AI communication. We then show actionable ways to translate our framework into two practical tools: design cards for designers and a configuration tool for users. The design cards enable designers to consider AI communication strategies that cater to a diverse range of users in co-creative contexts, while the configuration tool empowers users to customize AI communication based on their needs and creative workflows. This paper contributes new insights within the literature on human-AI co-creativity and Human-Computer Interaction, focusing on designing AI communication to enhance user experience. 

**Abstract (ZH)**: AI沟通框架在有效人机共创中的设计：FAICO方法论 

---
# Graph Attention-Driven Bayesian Deep Unrolling for Dual-Peak Single-Photon Lidar Imaging 

**Title (ZH)**: 基于图注意力的贝叶斯深度 unfolding 技术用于双峰单光子激光雷达成像 

**Authors**: Kyungmin Choi, JaKeoung Koo, Stephen McLaughlin, Abderrahim Halimi  

**Link**: [PDF](https://arxiv.org/pdf/2504.02480)  

**Abstract**: Single-photon Lidar imaging offers a significant advantage in 3D imaging due to its high resolution and long-range capabilities, however it is challenging to apply in noisy environments with multiple targets per pixel. To tackle these challenges, several methods have been proposed. Statistical methods demonstrate interpretability on the inferred parameters, but they are often limited in their ability to handle complex scenes. Deep learning-based methods have shown superior performance in terms of accuracy and robustness, but they lack interpretability or they are limited to a single-peak per pixel. In this paper, we propose a deep unrolling algorithm for dual-peak single-photon Lidar imaging. We introduce a hierarchical Bayesian model for multiple targets and propose a neural network that unrolls the underlying statistical method. To support multiple targets, we adopt a dual depth maps representation and exploit geometric deep learning to extract features from the point cloud. The proposed method takes advantages of statistical methods and learning-based methods in terms of accuracy and quantifying uncertainty. The experimental results on synthetic and real data demonstrate the competitive performance when compared to existing methods, while also providing uncertainty information. 

**Abstract (ZH)**: 单光子LiDAR成像在嘈杂环境下的双峰成像中具有显著优势，但由于目标数量多且像素噪声大，应用挑战较大。为此，已有多种方法被提出。统计方法在推断参数上具有可解释性，但往往难以处理复杂场景。基于深度学习的方法在准确性和鲁棒性上表现出优越性能，但缺乏可解释性或仅限于单峰场景。本文提出了一种用于双峰单光子LiDAR成像的深度拆解算法。我们引入了多层次贝叶斯模型处理多目标问题，并提出了一种神经网络来拆解基础统计方法。通过采用双深度图表示和利用几何深度学习从点云中提取特征，该方法在准确性和不确定性量化方面结合了统计方法和基于学习的方法的优势。实验结果在合成和实际数据上展示了与现有方法相比的竞争性能，并提供了不确定性信息。 

---
# Evaluating AI Recruitment Sourcing Tools by Human Preference 

**Title (ZH)**: 基于人类偏好的评估AI招聘 sourcing 工具 

**Authors**: Vladimir Slaykovskiy, Maksim Zvegintsev, Yury Sakhonchyk, Hrachik Ajamian  

**Link**: [PDF](https://arxiv.org/pdf/2504.02463)  

**Abstract**: This study introduces a benchmarking methodology designed to evaluate the performance of AI-driven recruitment sourcing tools. We created and utilized a dataset to perform a comparative analysis of search results generated by leading AI-based solutions, LinkedIn Recruiter, and our proprietary system, this http URL. Human experts assessed the relevance of the returned candidates, and an Elo rating system was applied to quantitatively measure each tool's comparative performance. Our findings indicate that AI-driven recruitment sourcing tools consistently outperform LinkedIn Recruiter in candidate relevance, with this http URL achieving the highest performance scores. Furthermore, we found a strong alignment between AI-based evaluations and human judgments, highlighting the potential for advanced AI technologies to substantially enhance talent acquisition effectiveness. Code and supporting data are publicly available at this https URL 

**Abstract (ZH)**: 本研究介绍了一种基准测试方法，用于评估人工智能驱动的招聘 sourcing 工具的性能。我们创建并使用了一个数据集，对领先的人工智能解决方案 LinkedIn Recruiter 和我们自主研发系统 this http URL 生成的搜索结果进行了比较分析。人类专家评估了返回候选人的相关性，并应用 Elo 排名体系定量衡量每种工具的相对性能。研究结果表明，人工智能驱动的招聘 sourcing 工具在候选人相关性方面始终优于 LinkedIn Recruiter，且 this http URL 达到了最高的性能评分。此外，我们发现基于人工智能的评估与人类判断之间存在很强的一致性，突显了先进人工智能技术在显著增强人才获取效果方面的潜力。代码和支持数据可在该链接 https:// 这里 公开获取。 

---
# Am I Being Treated Fairly? A Conceptual Framework for Individuals to Ascertain Fairness 

**Title (ZH)**: 我是否得到了公正的待遇？个人认定公正性的一个概念框架 

**Authors**: Juliett Suárez Ferreira, Marija Slavkovik, Jorge Casillas  

**Link**: [PDF](https://arxiv.org/pdf/2504.02461)  

**Abstract**: Current fairness metrics and mitigation techniques provide tools for practitioners to asses how non-discriminatory Automatic Decision Making (ADM) systems are. What if I, as an individual facing a decision taken by an ADM system, would like to know: Am I being treated fairly? We explore how to create the affordance for users to be able to ask this question of ADM. In this paper, we argue for the reification of fairness not only as a property of ADM, but also as an epistemic right of an individual to acquire information about the decisions that affect them and use that information to contest and seek effective redress against those decisions, in case they are proven to be discriminatory. We examine key concepts from existing research not only in algorithmic fairness but also in explainable artificial intelligence, accountability, and contestability. Integrating notions from these domains, we propose a conceptual framework to ascertain fairness by combining different tools that empower the end-users of ADM systems. Our framework shifts the focus from technical solutions aimed at practitioners to mechanisms that enable individuals to understand, challenge, and verify the fairness of decisions, and also serves as a blueprint for organizations and policymakers, bridging the gap between technical requirements and practical, user-centered accountability. 

**Abstract (ZH)**: 当前公平性指标与缓解技术为实践者提供了评估自动决策系统（ADM）非歧视性的工具。但如果我是面对ADM系统决策的个体，我希望能够知道：我是否得到了公平对待？我们探讨了如何为用户提供能力，使其能够对ADM提出这一问题。在本文中，我们不仅将公平性视为ADM的属性，也将其视为个体获取关于影响自己的决策信息的权利，并利用这些信息质疑和寻求有效救济的权利。我们研究了现有研究中的关键概念，不仅限于算法公平性，还包括可解释的人工智能、问责制和可争议性。结合这些领域中的概念，我们提出了一种概念框架，通过结合不同的工具来增强ADM系统终用户的能力，以确证公平性。该框架将关注点从针对实践者的技术解决方案转移到使个体能够理解、质疑和验证决策公平性的机制上，并为组织和政策制定者提供了蓝本，填补了技术要求与以用户为中心的实际问责制之间的差距。 

---
# Steiner Traveling Salesman Problem with Quantum Annealing 

**Title (ZH)**: 量子退火求解Steiner旅行商问题 

**Authors**: Alessia Ciacco, Francesca Guerriero, Eneko Osaba  

**Link**: [PDF](https://arxiv.org/pdf/2504.02388)  

**Abstract**: The Steiner Traveling Salesman Problem (STSP) is a variant of the classical Traveling Salesman Problem. The STSP involves incorporating steiner nodes, which are extra nodes not originally part of the required visit set but that can be added to the route to enhance the overall solution and minimize the total travel cost. Given the NP-hard nature of the STSP, we propose a quantum approach to address it. Specifically, we employ quantum annealing using D-Wave's hardware to explore its potential for solving this problem. To enhance computational feasibility, we develop a preprocessing method that effectively reduces the network size. Our experimental results demonstrate that this reduction technique significantly decreases the problem complexity, making the Quadratic Unconstrained Binary Optimization formulation, the standard input for quantum annealers, better suited for existing quantum hardware. Furthermore, the results highlight the potential of quantum annealing as a promising and innovative approach for solving the STSP. 

**Abstract (ZH)**: Steiner旅行售货商问题的量子方法研究 

---
# Temporal Gaussian Copula For Clinical Multivariate Time Series Data Imputation 

**Title (ZH)**: 基于时间的高斯copula临床多变量时间序列数据 imputation 

**Authors**: Ye Su, Hezhe Qiao, Di Wu, Yuwen Chen, Lin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.02317)  

**Abstract**: The imputation of the Multivariate time series (MTS) is particularly challenging since the MTS typically contains irregular patterns of missing values due to various factors such as instrument failures, interference from irrelevant data, and privacy regulations. Existing statistical methods and deep learning methods have shown promising results in time series imputation. In this paper, we propose a Temporal Gaussian Copula Model (TGC) for three-order MTS imputation. The key idea is to leverage the Gaussian Copula to explore the cross-variable and temporal relationships based on the latent Gaussian representation. Subsequently, we employ an Expectation-Maximization (EM) algorithm to improve robustness in managing data with varying missing rates. Comprehensive experiments were conducted on three real-world MTS datasets. The results demonstrate that our TGC substantially outperforms the state-of-the-art imputation methods. Additionally, the TGC model exhibits stronger robustness to the varying missing ratios in the test dataset. Our code is available at this https URL. 

**Abstract (ZH)**: 多变量时间序列的插值（MTS）由于受到各种因素如仪器故障、无关数据干扰和隐私规定的影响，通常包含不规则的缺失值模式，这使其特别具有挑战性。现有的统计方法和深度学习方法在时间序列插值方面已经显示出有希望的结果。本文提出了一种用于三阶多变量时间序列插值的临时高斯 copula 模型（TGC）。关键思想是利用高斯 copula 基于潜在高斯表示来探索跨变量和 temporal 关系。随后，我们采用期望最大（EM）算法以增强在处理不同缺失率数据时的稳健性。我们在三个真实世界的多变量时间序列数据集上进行了全面实验。结果表明，我们的 TGC 显著优于当前最先进的插值方法。此外，TGC 模型对测试数据集中变异缺失比例的鲁棒性更强。我们的代码可在此处获取。 

---
# Tree-based Models for Vertical Federated Learning: A Survey 

**Title (ZH)**: 基于树的模型在垂直联邦学习中的研究综述 

**Authors**: Bingchen Qian, Yuexiang Xie, Yaliang Li, Bolin Ding, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.02285)  

**Abstract**: Tree-based models have achieved great success in a wide range of real-world applications due to their effectiveness, robustness, and interpretability, which inspired people to apply them in vertical federated learning (VFL) scenarios in recent years. In this paper, we conduct a comprehensive study to give an overall picture of applying tree-based models in VFL, from the perspective of their communication and computation protocols. We categorize tree-based models in VFL into two types, i.e., feature-gathering models and label-scattering models, and provide a detailed discussion regarding their characteristics, advantages, privacy protection mechanisms, and applications. This study also focuses on the implementation of tree-based models in VFL, summarizing several design principles for better satisfying various requirements from both academic research and industrial deployment. We conduct a series of experiments to provide empirical observations on the differences and advances of different types of tree-based models. 

**Abstract (ZH)**: 基于树的模型在垂直联邦学习中的应用：从通信和计算协议视角的综合研究 

---
# Beyond Conventional Transformers: The Medical X-ray Attention (MXA) Block for Improved Multi-Label Diagnosis Using Knowledge Distillation 

**Title (ZH)**: 超越传统变压器：基于医学X射线注意力（MXA）块的知识蒸馏多标签诊断改进方法 

**Authors**: Amit Rand, Hadi Ibrahim  

**Link**: [PDF](https://arxiv.org/pdf/2504.02277)  

**Abstract**: Medical imaging, particularly X-ray analysis, often involves detecting multiple conditions simultaneously within a single scan, making multi-label classification crucial for real-world clinical applications. We present the Medical X-ray Attention (MXA) block, a novel attention mechanism tailored specifically to address the unique challenges of X-ray abnormality detection. The MXA block enhances traditional Multi-Head Self Attention (MHSA) by integrating a specialized module that efficiently captures both detailed local information and broader global context. To the best of our knowledge, this is the first work to propose a task-specific attention mechanism for diagnosing chest X-rays, as well as to attempt multi-label classification using an Efficient Vision Transformer (EfficientViT). By embedding the MXA block within the EfficientViT architecture and employing knowledge distillation, our proposed model significantly improves performance on the CheXpert dataset, a widely used benchmark for multi-label chest X-ray abnormality detection. Our approach achieves an area under the curve (AUC) of 0.85, an absolute improvement of 0.19 compared to our baseline model's AUC of 0.66, corresponding to a substantial approximate 233% relative improvement over random guessing (AUC = 0.5). 

**Abstract (ZH)**: 医学成像，尤其是X射线分析，往往需要在单次扫描中同时检测多种情况，因此多标签分类对于实际临床应用至关重要。我们提出了医学X射线注意力（MXA）模块，这是一种专门针对X射线异常检测的独特挑战而设计的新型注意力机制。MXA模块通过整合一个专门模块来增强传统的多头自注意力（MHSA），该模块能够高效地捕捉详细的局部信息和更广泛的全局上下文。据我们所知，这是首次提出针对胸部X射线诊断的任务特定注意力机制，并尝试使用高效视觉变压器（EfficientViT）进行多标签分类。通过在EfficientViT架构中嵌入MXA模块并使用知识蒸馏，我们提出的方法在CheXpert数据集上的表现显著提升，CheXpert数据集是广泛使用的多标签胸部X射线异常检测基准。我们的方法在曲线下面积（AUC）上达到0.85，相比基线模型的AUC（0.66）绝对提升了0.19，相当于随机猜测（AUC=0.5）约233%的相对改善。 

---
# Implicit Neural Differential Model for Spatiotemporal Dynamics 

**Title (ZH)**: 时空动态的隐式神经微分模型 

**Authors**: Deepak Akhare, Pan Du, Tengfei Luo, Jian-Xun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02260)  

**Abstract**: Hybrid neural-physics modeling frameworks through differentiable programming have emerged as powerful tools in scientific machine learning, enabling the integration of known physics with data-driven learning to improve prediction accuracy and generalizability. However, most existing hybrid frameworks rely on explicit recurrent formulations, which suffer from numerical instability and error accumulation during long-horizon forecasting. In this work, we introduce Im-PiNDiff, a novel implicit physics-integrated neural differentiable solver for stable and accurate modeling of spatiotemporal dynamics. Inspired by deep equilibrium models, Im-PiNDiff advances the state using implicit fixed-point layers, enabling robust long-term simulation while remaining fully end-to-end differentiable. To enable scalable training, we introduce a hybrid gradient propagation strategy that integrates adjoint-state methods with reverse-mode automatic differentiation. This approach eliminates the need to store intermediate solver states and decouples memory complexity from the number of solver iterations, significantly reducing training overhead. We further incorporate checkpointing techniques to manage memory in long-horizon rollouts. Numerical experiments on various spatiotemporal PDE systems, including advection-diffusion processes, Burgers' dynamics, and multi-physics chemical vapor infiltration processes, demonstrate that Im-PiNDiff achieves superior predictive performance, enhanced numerical stability, and substantial reductions in memory and runtime cost relative to explicit and naive implicit baselines. This work provides a principled, efficient, and scalable framework for hybrid neural-physics modeling. 

**Abstract (ZH)**: 基于可微编程的隐式物理集成神经计算框架：稳定且准确的空间时间动态建模 

---
# AC-LoRA: Auto Component LoRA for Personalized Artistic Style Image Generation 

**Title (ZH)**: AC-LoRA: 自动组件LoRA个性化艺术风格图像生成 

**Authors**: Zhipu Cui, Andong Tian, Zhi Ying, Jialiang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02231)  

**Abstract**: Personalized image generation allows users to preserve styles or subjects of a provided small set of images for further image generation. With the advancement in large text-to-image models, many techniques have been developed to efficiently fine-tune those models for personalization, such as Low Rank Adaptation (LoRA). However, LoRA-based methods often face the challenge of adjusting the rank parameter to achieve satisfactory results. To address this challenge, AutoComponent-LoRA (AC-LoRA) is proposed, which is able to automatically separate the signal component and noise component of the LoRA matrices for fast and efficient personalized artistic style image generation. This method is based on Singular Value Decomposition (SVD) and dynamic heuristics to update the hyperparameters during training. Superior performance over existing methods in overcoming model underfitting or overfitting problems is demonstrated. The results were validated using FID, CLIP, DINO, and ImageReward, achieving an average of 9% improvement. 

**Abstract (ZH)**: 个性化图像生成允许用户保留所提供小图像集的风格或主题以进行进一步的图像生成。随着大规模文本到图像模型的进步，已经开发了许多高效微调这些模型以实现个性化的技术，如低秩适应（LoRA）。然而，基于LoRA的方法常常面临调整秩参数以取得满意结果的挑战。为了解决这一挑战，提出了AutoComponent-LoRA（AC-LoRA），该方法能够自动分离LoRA矩阵的信号分量和噪声分量，以实现快速高效的个性化艺术风格图像生成。该方法基于奇异值分解（SVD）和动态启发式，在训练过程中更新超参数。实验结果表明，该方法在克服模型欠拟合或过拟合问题方面优于现有方法，使用FID、CLIP、DINO和ImageReward验证的结果平均提高了9%。 

---
# Learning and Improving Backgammon Strategy 

**Title (ZH)**: 学习并提升背gammon策略 

**Authors**: Gregory R. Galperin  

**Link**: [PDF](https://arxiv.org/pdf/2504.02221)  

**Abstract**: A novel approach to learning is presented, combining features of on-line and off-line methods to achieve considerable performance in the task of learning a backgammon value function in a process that exploits the processing power of parallel supercomputers. The off-line methods comprise a set of techniques for parallelizing neural network training and $TD(\lambda)$ reinforcement learning; here Monte-Carlo ``Rollouts'' are introduced as a massively parallel on-line policy improvement technique which applies resources to the decision points encountered during the search of the game tree to further augment the learned value function estimate. A level of play roughly as good as, or possibly better than, the current champion human and computer backgammon players has been achieved in a short period of learning. 

**Abstract (ZH)**: 一种结合在线和离线方法的新颖学习方法，在利用并行超级计算机的计算能力完成背Gammon价值函数学习任务的过程中实现了显著的性能提升。离线方法包括用于并行化神经网络训练和$TD(\lambda)$强化学习的技术；在此基础上引入了蒙特卡罗“Rollouts”作为大规模并行在线策略改进技术，将资源应用于游戏中遇到的决策点，进一步增强学习到的价值函数估计。在较短的学习时间内，达到了与当前顶级人类和计算机背Gammon玩家水平相当，甚至更高的水平。 

---
# FT-Transformer: Resilient and Reliable Transformer with End-to-End Fault Tolerant Attention 

**Title (ZH)**: FT-Transformer: 具有端到端故障容忍注意机制的稳健可靠变换器 

**Authors**: Huangliang Dai, Shixun Wu, Hairui Zhao, Jiajun Huang, Zizhe Jian, Yue Zhu, Haiyang Hu, Zizhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.02211)  

**Abstract**: Transformer models leverage self-attention mechanisms to capture complex dependencies, demonstrating exceptional performance in various applications. However, the long-duration high-load computations required for model inference impose stringent reliability demands on the computing platform, as soft errors that occur during execution can significantly degrade model performance. Existing fault tolerance methods protect each operation separately using decoupled kernels, incurring substantial computational and memory overhead. In this paper, we propose a novel error-resilient framework for Transformer models, integrating end-to-end fault tolerant attention (EFTA) to improve inference reliability against soft errors. Our approach enables error detection and correction within a fully fused attention kernel, reducing redundant data access and thereby mitigating memory faults. To further enhance error coverage and reduce overhead, we design a hybrid fault tolerance scheme tailored for the EFTA, introducing for the first time: 1) architecture-aware algorithm-based fault tolerance (ABFT) using tensor checksum, which minimizes inter-thread communication overhead on tensor cores during error detection; 2) selective neuron value restriction, which selectively applies adaptive fault tolerance constraints to neuron values, balancing error coverage and overhead; 3) unified verification, reusing checksums to streamline multiple computation steps into a single verification process. Experimental results show that EFTA achieves up to 7.56x speedup over traditional methods with an average fault tolerance overhead of 13.9%. 

**Abstract (ZH)**: Transformer模型利用自注意力机制捕获复杂依赖关系，在各种应用中表现出色。然而，模型推理所需的长时间高负载计算对计算平台提出了严格的可靠性要求，因为执行过程中发生的软错误会显著降低模型性能。现有的容错方法单独保护每个操作，使用解耦内核，导致巨大的计算和内存开销。本文提出了一种新的Transformer模型容错框架，集成了端到端容错注意力(EFTA)，以提高在软错误下的推理可靠性。我们的方法能够在完全融合的注意力内核中进行错误检测和纠正，从而减少冗余数据访问，进而减轻内存故障的影响。为进一步提高错误覆盖并减少开销，我们设计了一种针对EFTA的混合容错方案，并首次引入了：1) 意识架构的算法-Based容错(ABFT)使用张量校验和，以最小化张量核心在错误检测过程中线程间通信开销；2) 选择性神经元值限制，选择性地对神经元值应用自适应容错约束，平衡错误覆盖和开销；3) 统一验证，利用校验和重新使用，将多个计算步骤合并为单一验证过程。实验结果表明，EFTA相比传统方法实现了高达7.56倍的速度提升，平均容错开销为13.9%。 

---
# ESC: Erasing Space Concept for Knowledge Deletion 

**Title (ZH)**: ESC: 删除知识时的.erase空间概念 

**Authors**: Tae-Young Lee, Sundong Park, Minwoo Jeon, Hyoseok Hwang, Gyeong-Moon Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.02199)  

**Abstract**: As concerns regarding privacy in deep learning continue to grow, individuals are increasingly apprehensive about the potential exploitation of their personal knowledge in trained models. Despite several research efforts to address this, they often fail to consider the real-world demand from users for complete knowledge erasure. Furthermore, our investigation reveals that existing methods have a risk of leaking personal knowledge through embedding features. To address these issues, we introduce a novel concept of Knowledge Deletion (KD), an advanced task that considers both concerns, and provides an appropriate metric, named Knowledge Retention score (KR), for assessing knowledge retention in feature space. To achieve this, we propose a novel training-free erasing approach named Erasing Space Concept (ESC), which restricts the important subspace for the forgetting knowledge by eliminating the relevant activations in the feature. In addition, we suggest ESC with Training (ESC-T), which uses a learnable mask to better balance the trade-off between forgetting and preserving knowledge in KD. Our extensive experiments on various datasets and models demonstrate that our proposed methods achieve the fastest and state-of-the-art performance. Notably, our methods are applicable to diverse forgetting scenarios, such as facial domain setting, demonstrating the generalizability of our methods. The code is available at this http URL . 

**Abstract (ZH)**: 随着对深度学习中隐私问题的关注不断增加，个人对训练模型中可能对其个人知识的利用越来越感到担忧。尽管已经进行了多项研究努力来解决这些问题，但它们常常未能考虑用户对完全删除知识的现实需求。此外，我们的研究发现，现有方法存在通过特征嵌入泄露个人知识的风险。为解决这些问题，我们提出了一种新的知识删除（KD）概念，这是一个同时考虑上述两个方面的高级任务，并提出了一种名为知识保留得分（KR）的评估特征空间中知识保留的合适度量方法。为此，我们提出了一种新型无训练删除方法——擦除空间概念（ESC），通过消除特征中的相关激活来限制遗忘知识的重要子空间。此外，我们建议了带有训练的ESC（ESC-T），它使用可学习的掩码来更好地平衡KD中遗忘与保留知识之间的权衡。我们在多种数据集和模型上的广泛实验表明，我们提出的方法达到了最快的和最先进的性能。值得注意的是，我们的方法适用于各种遗忘场景，例如面部域设置，展示了我们方法的普适性。代码可在以下网址获取。 

---
# On the Geometry of Receiver Operating Characteristic and Precision-Recall Curves 

**Title (ZH)**: 关于接收器操作特征和精确度-召回率曲线的几何学 

**Authors**: Reza Sameni  

**Link**: [PDF](https://arxiv.org/pdf/2504.02169)  

**Abstract**: We study the geometry of Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves in binary classification problems. The key finding is that many of the most commonly used binary classification metrics are merely functions of the composition function $G := F_p \circ F_n^{-1}$, where $F_p(\cdot)$ and $F_n(\cdot)$ are the class-conditional cumulative distribution functions of the classifier scores in the positive and negative classes, respectively. This geometric perspective facilitates the selection of operating points, understanding the effect of decision thresholds, and comparison between classifiers. It also helps explain how the shapes and geometry of ROC/PR curves reflect classifier behavior, providing objective tools for building classifiers optimized for specific applications with context-specific constraints. We further explore the conditions for classifier dominance, present analytical and numerical examples demonstrating the effects of class separability and variance on ROC and PR geometries, and derive a link between the positive-to-negative class leakage function $G(\cdot)$ and the Kullback--Leibler divergence. The framework highlights practical considerations, such as model calibration, cost-sensitive optimization, and operating point selection under real-world capacity constraints, enabling more informed approaches to classifier deployment and decision-making. 

**Abstract (ZH)**: 我们研究了二分类问题中接收器操作 characteristic (ROC) 和精确率-召回率 (PR) 曲线的几何结构。主要发现是，许多常用的二分类指标仅仅是类条件累积分布函数的复合函数 $G := F_p \circ F_n^{-1}$ 的函数，其中 $F_p(\cdot)$ 和 $F_n(\cdot)$ 分别是正类和负类中分类器得分的类条件累积分布函数。这种几何视角简化了操作点的选择，有助于理解决策阈值的影响，并在比较不同分类器时提供便利。它还解释了 ROC/PR 曲线的形状和几何结构如何反映分类器的行为，提供了在特定应用场景和情境约束下优化分类器的客观工具。我们进一步探讨了分类器占优的条件，给出了类可分性和方差对 ROC 和 PR 几何结构的影响的分析和数值示例，并推导了正类泄漏函数 $G(\cdot)$ 与克劳德-莱布尼兹散度之间的关系。该框架突出了实际考虑因素，如模型校准、代价敏感优化以及实际容量约束下的操作点选择，使分类器的部署和决策更具指导意义。 

---
# MDP: Multidimensional Vision Model Pruning with Latency Constraint 

**Title (ZH)**: MDP：多维度视觉模型剪枝算法及其延迟约束 

**Authors**: Xinglong Sun, Barath Lakshmanan, Maying Shen, Shiyi Lan, Jingde Chen, Jose M. Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2504.02168)  

**Abstract**: Current structural pruning methods face two significant limitations: (i) they often limit pruning to finer-grained levels like channels, making aggressive parameter reduction challenging, and (ii) they focus heavily on parameter and FLOP reduction, with existing latency-aware methods frequently relying on simplistic, suboptimal linear models that fail to generalize well to transformers, where multiple interacting dimensions impact latency. In this paper, we address both limitations by introducing Multi-Dimensional Pruning (MDP), a novel paradigm that jointly optimizes across a variety of pruning granularities-including channels, query, key, heads, embeddings, and blocks. MDP employs an advanced latency modeling technique to accurately capture latency variations across all prunable dimensions, achieving an optimal balance between latency and accuracy. By reformulating pruning as a Mixed-Integer Nonlinear Program (MINLP), MDP efficiently identifies the optimal pruned structure across all prunable dimensions while respecting latency constraints. This versatile framework supports both CNNs and transformers. Extensive experiments demonstrate that MDP significantly outperforms previous methods, especially at high pruning ratios. On ImageNet, MDP achieves a 28% speed increase with a +1.4 Top-1 accuracy improvement over prior work like HALP for ResNet50 pruning. Against the latest transformer pruning method, Isomorphic, MDP delivers an additional 37% acceleration with a +0.7 Top-1 accuracy improvement. 

**Abstract (ZH)**: 多维度剪枝：一种新型跨粒度优化范式 

---
# Multivariate Temporal Regression at Scale: A Three-Pillar Framework Combining ML, XAI, and NLP 

**Title (ZH)**: 大规模多变量时间序列回归：结合机器学习、解释性人工智能和自然语言处理的三支柱框架 

**Authors**: Jiztom Kavalakkatt Francis, Matthew J Darr  

**Link**: [PDF](https://arxiv.org/pdf/2504.02151)  

**Abstract**: The rapid use of artificial intelligence (AI) in processes such as coding, image processing, and data prediction means it is crucial to understand and validate the data we are working with fully. This paper dives into the hurdles of analyzing high-dimensional data, especially when it gets too complex. Traditional methods in data analysis often look at direct connections between input variables, which can miss out on the more complicated relationships within the data.
To address these issues, we explore several tested techniques, such as removing specific variables to see their impact and using statistical analysis to find connections between multiple variables. We also consider the role of synthetic data and how information can sometimes be redundant across different sensors. These analyses are typically very computationally demanding and often require much human effort to make sense of the results.
A common approach is to treat the entire dataset as one unit and apply advanced models to handle it. However, this can become problematic with larger, noisier datasets and more complex models. So, we suggest methods to identify overall patterns that can help with tasks like classification or regression based on the idea that more straightforward approaches might be more understandable.
Our research looks at two datasets: a real-world dataset and a synthetic one. The goal is to create a methodology that highlights key features on a global scale that lead to predictions, making it easier to validate or quantify the data set. By reducing the dimensionality with this method, we can simplify the models used and thus clarify the insights we gain. Furthermore, our method can reveal unexplored relationships between specific inputs and outcomes, providing a way to validate these new connections further. 

**Abstract (ZH)**: 人工智能在编码、图像处理和数据预测等过程中的快速应用意味着我们需要全面理解并验证我们所处理的数据。本文深入探讨了分析高维数据的挑战，特别是在数据变得过于复杂时。传统数据分析方法通常关注输入变量之间的直接联系，这可能会忽略数据内部更为复杂的关系。为应对这些问题，我们探索了几种经过验证的技术，如移除特定变量以观察其影响，以及利用统计分析来发现多个变量间的联系。我们还考虑了合成数据的作用，以及不同传感器之间信息可能存在的冗余。这些分析通常计算需求非常高，并且经常需要大量的人工努力来理解结果。 

---
# Enhancing Embedding Representation Stability in Recommendation Systems with Semantic ID 

**Title (ZH)**: 在推荐系统中通过语义ID增强嵌入表示稳定性 

**Authors**: Carolina Zheng, Minhui Huang, Dmitrii Pedchenko, Kaushik Rangadurai, Siyu Wang, Gaby Nahum, Jie Lei, Yang Yang, Tao Liu, Zutian Luo, Xiaohan Wei, Dinesh Ramasamy, Jiyan Yang, Yiping Han, Lin Yang, Hangjun Xu, Rong Jin, Shuang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02137)  

**Abstract**: The exponential growth of online content has posed significant challenges to ID-based models in industrial recommendation systems, ranging from extremely high cardinality and dynamically growing ID space, to highly skewed engagement distributions, to prediction instability as a result of natural id life cycles (e.g, the birth of new IDs and retirement of old IDs). To address these issues, many systems rely on random hashing to handle the id space and control the corresponding model parameters (i.e embedding table). However, this approach introduces data pollution from multiple ids sharing the same embedding, leading to degraded model performance and embedding representation instability.
This paper examines these challenges and introduces Semantic ID prefix ngram, a novel token parameterization technique that significantly improves the performance of the original Semantic ID. Semantic ID prefix ngram creates semantically meaningful collisions by hierarchically clustering items based on their content embeddings, as opposed to random assignments. Through extensive experimentation, we demonstrate that Semantic ID prefix ngram not only addresses embedding instability but also significantly improves tail id modeling, reduces overfitting, and mitigates representation shifts. We further highlight the advantages of Semantic ID prefix ngram in attention-based models that contextualize user histories, showing substantial performance improvements. We also report our experience of integrating Semantic ID into Meta production Ads Ranking system, leading to notable performance gains and enhanced prediction stability in live deployments. 

**Abstract (ZH)**: 基于语义的ID前缀n元组：显著改善工业推荐系统中的ID表示 

---
# On Model Protection in Federated Learning against Eavesdropping Attacks 

**Title (ZH)**: 联邦学习中针对窃听攻击的模型保护 

**Authors**: Dipankar Maity, Kushal Chakrabarti  

**Link**: [PDF](https://arxiv.org/pdf/2504.02114)  

**Abstract**: In this study, we investigate the protection offered by federated learning algorithms against eavesdropping adversaries. In our model, the adversary is capable of intercepting model updates transmitted from clients to the server, enabling it to create its own estimate of the model. Unlike previous research, which predominantly focuses on safeguarding client data, our work shifts attention protecting the client model itself. Through a theoretical analysis, we examine how various factors, such as the probability of client selection, the structure of local objective functions, global aggregation at the server, and the eavesdropper's capabilities, impact the overall level of protection. We further validate our findings through numerical experiments, assessing the protection by evaluating the model accuracy achieved by the adversary. Finally, we compare our results with methods based on differential privacy, underscoring their limitations in this specific context. 

**Abstract (ZH)**: 本研究探讨了联邦学习算法在对抗窃听攻击时所提供的保护。在我们的模型中，攻击者能够拦截从客户端传输到服务器的模型更新，从而能够生成自己的模型估计。与以往主要集中在保护客户端数据的研究不同，我们的工作将注意力转向保护客户端模型本身。通过理论分析，我们探讨了参与客户端的选择概率、本地目标函数的结构、服务器端的全局聚合以及攻击者的各种能力等因素如何影响整体保护水平。我们进一步通过数值实验验证了这些发现，通过评估攻击者实现的模型准确性来衡量保护程度。最后，我们将我们的结果与基于差分隐私的方法进行比较，突显了它们在这一特定情境下的局限性。 

---
# An Introductory Survey to Autoencoder-based Deep Clustering -- Sandboxes for Combining Clustering with Deep Learning 

**Title (ZH)**: 基于自动编码器的深度聚类入门综述——结合聚类与深度学习的实验平台 

**Authors**: Collin Leiber, Lukas Miklautz, Claudia Plant, Christian Böhm  

**Link**: [PDF](https://arxiv.org/pdf/2504.02087)  

**Abstract**: Autoencoders offer a general way of learning low-dimensional, non-linear representations from data without labels. This is achieved without making any particular assumptions about the data type or other domain knowledge. The generality and domain agnosticism in combination with their simplicity make autoencoders a perfect sandbox for researching and developing novel (deep) clustering algorithms. Clustering methods group data based on similarity, a task that benefits from the lower-dimensional representation learned by an autoencoder, mitigating the curse of dimensionality. Specifically, the combination of deep learning with clustering, called Deep Clustering, enables to learn a representation tailored to specific clustering tasks, leading to high-quality results. This survey provides an introduction to fundamental autoencoder-based deep clustering algorithms that serve as building blocks for many modern approaches. 

**Abstract (ZH)**: 自动编码器提供了一种从数据中学习低维非线性表示的通用方法，无需标签，无需对数据类型或其他领域知识做出特定假设。其通用性和领域无关性与简洁性使得自动编码器成为研究和开发新型（深度）聚类算法的理想平台。基于聚类的深层表示降低了维数 curse of dimensionality，使得聚类方法能够根据相似性对数据进行分组，特别地，深度学习与聚类的结合，即深层聚类，能够学习适合特定聚类任务的表示，从而获得高质量的结果。本文提供了自动编码器基础的深度聚类算法的综述，这些算法是许多现代方法的基础。 

---
# From Text to Graph: Leveraging Graph Neural Networks for Enhanced Explainability in NLP 

**Title (ZH)**: 从文本到图形：利用图形神经网络提高自然语言处理的可解释性 

**Authors**: Fabio Yáñez-Romero, Andrés Montoyo, Armando Suárez, Yoan Gutiérrez, Ruslan Mitkov  

**Link**: [PDF](https://arxiv.org/pdf/2504.02064)  

**Abstract**: Researchers have relegated natural language processing tasks to Transformer-type models, particularly generative models, because these models exhibit high versatility when performing generation and classification tasks. As the size of these models increases, they achieve outstanding results. Given their widespread use, many explainability techniques are developed based on these models. However, this process becomes computationally expensive due to the large size of the models. Additionally, transformers interpret input information through tokens that fragment input words into sequences lacking inherent semantic meaning, complicating the explanation of the model from the very beginning. This study proposes a novel methodology to achieve explainability in natural language processing tasks by automatically converting sentences into graphs and maintaining semantics through nodes and relations that express fundamental linguistic concepts. It also allows the subsequent exploitation of this knowledge in subsequent tasks, making it possible to obtain trends and understand how the model associates the different elements inside the text with the explained task. The experiments delivered promising results in determining the most critical components within the text structure for a given classification. 

**Abstract (ZH)**: 研究人员将自然语言处理任务交予Transformer类型模型，特别是生成模型，因为这些模型在生成和分类任务中表现出高度的灵活性。随着模型规模的增加，它们取得了出色的结果。鉴于这些模型的广泛应用，许多解释性技术基于这些模型开发出来。然而，由于模型规模庞大，这一过程变得计算成本高昂。此外，Transformer通过令牌将输入词分解成缺乏内在语义意义的序列，从一开始就使模型解释变得复杂。本研究提出了一种新的方法，通过自动将句子转换为图，并通过节点和关系保留语义，来实现自然语言处理任务的解释性，这些节点和关系表达基本的语言概念。这种方法还允许在后续任务中利用这一知识，从而能够获得趋势并理解模型是如何将文本中的不同元素与解释性任务关联起来的。实验结果显示，这种方法在确定给定分类任务中最重要的文本结构组件方面取得了令人鼓舞的结果。 

---
# Antithetic Sampling for Top-k Shapley Identification 

**Title (ZH)**: 反向取样法用于Top-k Shapley值识别 

**Authors**: Patrick Kolpaczki, Tim Nielen, Eyke Hüllermeier  

**Link**: [PDF](https://arxiv.org/pdf/2504.02019)  

**Abstract**: Additive feature explanations rely primarily on game-theoretic notions such as the Shapley value by viewing features as cooperating players. The Shapley value's popularity in and outside of explainable AI stems from its axiomatic uniqueness. However, its computational complexity severely limits practicability. Most works investigate the uniform approximation of all features' Shapley values, needlessly consuming samples for insignificant features. In contrast, identifying the $k$ most important features can already be sufficiently insightful and yields the potential to leverage algorithmic opportunities connected to the field of multi-armed bandits. We propose Comparable Marginal Contributions Sampling (CMCS), a method for the top-$k$ identification problem utilizing a new sampling scheme taking advantage of correlated observations. We conduct experiments to showcase the efficacy of our method in compared to competitive baselines. Our empirical findings reveal that estimation quality for the approximate-all problem does not necessarily transfer to top-$k$ identification and vice versa. 

**Abstract (ZH)**: 加性特征解释主要依赖于合作博弈理论中的Shapley值。Shapley值在可解释AI内外的广泛接受源于其公理唯一性。然而，其计算复杂性严重限制了其实用性。大多数研究关注所有特征Shapley值的均匀近似，无谓地为不重要的特征消耗样本。相反，识别最重要的$k$个特征已经可以提供足够的洞察，并有可能利用与多臂bandit问题相关的算法机会。我们提出了一种称为可比边际贡献采样（CMCS）的方法，用于利用相关观测的新采样方案解决top-$k$识别问题。我们进行了实验以展示我们的方法相对于竞争基线的有效性。我们的实证研究表明，对于近似所有特征的问题，估计质量不一定转移到top-$k$识别问题上，反之亦然。 

---
# HCAF-DTA: drug-target binding affinity prediction with cross-attention fused hypergraph neural networks 

**Title (ZH)**: HCAF-DTA：基于交叉注意力融合超图神经网络的药物-靶点结合亲和力预测 

**Authors**: Jiannuo Li, Lan Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.02014)  

**Abstract**: Accurate prediction of the binding affinity between drugs and target proteins is a core task in computer-aided drug design. Existing deep learning methods tend to ignore the information of internal sub-structural features of drug molecules and drug-target interactions, resulting in limited prediction performance. In this paper, we propose a drug-target association prediction model HCAF-DTA based on cross-attention fusion hypergraph neural network. The model innovatively introduces hypergraph representation in the feature extraction stage: drug molecule hypergraphs are constructed based on the tree decomposition algorithm, and the sub-structural and global features extracted by fusing the hypergraph neural network with the graphical neural network through hopping connections, in which the hyper edges can efficiently characterise the functional functional groups and other key chemical features; for the protein feature extraction, a weighted graph is constructed based on the residues predicted by the ESM model contact maps to construct weighted graphs, and multilayer graph neural networks were used to capture spatial dependencies. In the prediction stage, a bidirectional multi-head cross-attention mechanism is designed to model intermolecular interactions from the dual viewpoints of atoms and amino acids, and cross-modal features with correlated information are fused by attention. Experiments on benchmark datasets such as Davis and KIBA show that HCAF-DTA outperforms state of the arts in all three performance evaluation metrics, with the MSE metrics reaching 0.198 and 0.122, respectively, with an improvement of up to 4% from the optimal baseline. 

**Abstract (ZH)**: 基于跨注意力融合超图神经网络的药物-靶标结合 affinity 预测模型 HCAF-DTA 

---
# Random Conditioning with Distillation for Data-Efficient Diffusion Model Compression 

**Title (ZH)**: 数据高效的扩散模型压缩中的随机条件化与蒸馏 

**Authors**: Dohyun Kim, Sehwan Park, Geonhee Han, Seung Wook Kim, Paul Hongsuck Seo  

**Link**: [PDF](https://arxiv.org/pdf/2504.02011)  

**Abstract**: Diffusion models generate high-quality images through progressive denoising but are computationally intensive due to large model sizes and repeated sampling. Knowledge distillation, which transfers knowledge from a complex teacher to a simpler student model, has been widely studied in recognition tasks, particularly for transferring concepts unseen during student training. However, its application to diffusion models remains underexplored, especially in enabling student models to generate concepts not covered by the training images. In this work, we propose Random Conditioning, a novel approach that pairs noised images with randomly selected text conditions to enable efficient, image-free knowledge distillation. By leveraging this technique, we show that the student can generate concepts unseen in the training images. When applied to conditional diffusion model distillation, our method allows the student to explore the condition space without generating condition-specific images, resulting in notable improvements in both generation quality and efficiency. This promotes resource-efficient deployment of generative diffusion models, broadening their accessibility for both research and real-world applications. Code, models, and datasets are available at this https URL . 

**Abstract (ZH)**: 基于随机条件的扩散模型高效知识蒸馏 

---
# AI Regulation and Capitalist Growth: Balancing Innovation, Ethics, and Global Governance 

**Title (ZH)**: AI监管与资本主义增长：平衡创新、 Ethics 和全球治理 

**Authors**: Vikram Kulothungan, Priya Ranjani Mohan, Deepti Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2504.02000)  

**Abstract**: Artificial Intelligence (AI) is increasingly central to economic growth, promising new efficiencies and markets. This economic significance has sparked debate over AI regulation: do rules and oversight bolster long term growth by building trust and safeguarding the public, or do they constrain innovation and free enterprise? This paper examines the balance between AI regulation and capitalist ideals, focusing on how different approaches to AI data privacy can impact innovation in AI-driven applications. The central question is whether AI regulation enhances or inhibits growth in a capitalist economy. Our analysis synthesizes historical precedents, the current U.S. regulatory landscape, economic projections, legal challenges, and case studies of recent AI policies. We discuss that carefully calibrated AI data privacy regulations-balancing innovation incentives with the public interest can foster sustainable growth by building trust and ensuring responsible data use, while excessive regulation may risk stifling innovation and entrenching incumbents. 

**Abstract (ZH)**: 人工智能（AI）日益成为经济增长的核心，承诺带来新的效率和市场。这种经济意义引发了关于AI监管的争论：规则和监督是否通过建立信任和保护公众来促进长期增长，还是限制创新和自由企业？本文探讨了AI监管与资本主义理想之间的平衡，重点关注不同的人工智能数据隐私方法如何影响人工智能驱动应用的创新。中心问题是AI监管是促进还是阻碍资本主义经济的增长。我们的分析综合了历史先例、当前的美国监管环境、经济增长预测、法律挑战以及近年来AI政策的案例研究。我们讨论了精心设计的人工智能数据隐私监管——平衡创新激励与公众利益可以通过建立信任和确保负责任的数据使用来促进可持续增长，而过度监管可能会限制创新并巩固既有企业。 

---
# Exploring the Societal and Economic Impacts of Artificial Intelligence: A Scenario Generation Methodology 

**Title (ZH)**: 探索人工智能的社会与经济影响：一种情景生成方法论 

**Authors**: Carlos J. Costa, Joao Tiago Aparicio  

**Link**: [PDF](https://arxiv.org/pdf/2504.01992)  

**Abstract**: This paper explores artificial intelligence's potential societal and economic impacts (AI) through generating scenarios that assess how AI may influence various sectors. We categorize and analyze key factors affecting AI's integration and adoption by applying an Impact-Uncertainty Matrix. A proposed methodology involves querying academic databases, identifying emerging trends and topics, and categorizing these into an impact uncertainty framework. The paper identifies critical areas where AI may bring significant change and outlines potential future scenarios based on these insights. This research aims to inform policymakers, industry leaders, and researchers on the strategic planning required to address the challenges and opportunities AI presents 

**Abstract (ZH)**: 本文通过生成情景来探讨人工智能在社会和经济领域的潜在影响（AI），分析AI融入和 Adoption 的关键因素，并将其分类和分析应用到影响-不确定性矩阵中。提出的 methodology 包括查询学术数据库、识别新兴趋势和主题，并将这些内容分类到影响不确定性框架中。本文确定了 AI 可能带来重大变革的关键领域，并基于这些洞察预测潜在的未来情景。此项研究旨在为政策制定者、行业领袖和研究人员提供关于应对 AI 带来的挑战和机遇所需的战略规划信息。 

---
# NLS: Natural-Level Synthesis for Hardware Implementation Through GenAI 

**Title (ZH)**: NLS：通过生成式人工智能实现硬件级别的自然水平合成 

**Authors**: Kaiyuan Yang, Huang Ouyang, Xinyi Wang, Bingjie Lu, Yanbo Wang, Charith Abhayaratne, Sizhao Li, Long Jin, Tiantai Deng  

**Link**: [PDF](https://arxiv.org/pdf/2504.01981)  

**Abstract**: This paper introduces Natural-Level Synthesis, an innovative approach for generating hardware using generative artificial intelligence on both the system level and component-level. NLS bridges a gap in current hardware development processes, where algorithm and application engineers' involvement typically ends at the requirements stage. With NLS, engineers can participate more deeply in the development, synthesis, and test stages by using Gen-AI models to convert natural language descriptions directly into Hardware Description Language code. This approach not only streamlines hardware development but also improves accessibility, fostering a collaborative workflow between hardware and algorithm engineers. We developed the NLS tool to facilitate natural language-driven HDL synthesis, enabling rapid generation of system-level HDL designs while significantly reducing development complexity. Evaluated through case studies and benchmarks using Performance, Power, and Area metrics, NLS shows its potential to enhance resource efficiency in hardware development. This work provides a extensible, efficient solution for hardware synthesis and establishes a Visual Studio Code Extension to assess Gen-AI-driven HDL generation and system integration, laying a foundation for future AI-enhanced and AI-in-the-loop Electronic Design Automation tools. 

**Abstract (ZH)**: 自然水平合成：一种使用生成人工智能进行系统级和组件级硬件生成的新方法 

---
# Information Gain Is Not All You Need 

**Title (ZH)**: 信息增益并非万能 

**Authors**: Ludvig Ericson, José Pedro, Patric Jensfelt  

**Link**: [PDF](https://arxiv.org/pdf/2504.01980)  

**Abstract**: Autonomous exploration in mobile robotics is driven by two competing objectives: coverage, to exhaustively observe the environment; and path length, to do so with the shortest path possible. Though it is difficult to evaluate the best course of action without knowing the unknown, the unknown can often be understood through models, maps, or common sense. However, previous work has shown that improving estimates of information gain through such prior knowledge leads to greedy behavior and ultimately causes backtracking, which degrades coverage performance. In fact, any information gain maximization will exhibit this behavior, even without prior knowledge. Information gained at task completion is constant, and cannot be maximized for. It is therefore an unsuitable choice as an optimization objective. Instead, information gain is a decision criterion for determining which candidate states should still be considered for exploration. The task therefore becomes to reach completion with the shortest total path. Since determining the shortest path is typically intractable, it is necessary to rely on a heuristic or estimate to identify candidate states that minimize the total path length. To address this, we propose a heuristic that reduces backtracking by preferring candidate states that are close to the robot, but far away from other candidate states. We evaluate the performance of the proposed heuristic in simulation against an information gain-based approach and frontier exploration, and show that our method significantly decreases total path length, both with and without prior knowledge of the environment. 

**Abstract (ZH)**: 自主移动机器人中的自动探索受兩個競爭目標驅動：覆蓋，以便 Exhaustively 觀察環境；和路徑長度，以便儘可能使用 shortest path。尽管在不知不了解情況下的最佳行動難以評估，但通過模型、地圖或常識可以理解不了解的部分。然而，先前的研究表明，通過先驗知識來改進信息獲取估計會導致貪婪行為，最終導致回溯，這會]*(損害覆蓋性能。事實上，任何信息獲取最大化都會表現出這種行為，即使在沒有先驗知識的情況下。任務完成時獲取的信*[息是固定的，無法最大化。因此，它不能用作優化目標的合適選擇。相反，信息獲取是確定哪些候選狀態仍然值得探索的決策標準。因此，任務成為使用最短總路徑完成任务。由於確定最短路徑通常是不可解的，因此需要依靠启发式方法或估計來識別最小化總路徑長度的候選狀態。為此，我們提出了一種启发式方法，通过优选距离机器人较近但与其他候選状态较远的候選状态，来減少回溯行为。我们将提出的启发式方法在仿真中与基于信息获取的方法和前沿探索方法进行了性能评估，并展示了在有无环境先验知识的情况下，我们的方法显著降低了总路径长度。 

---
# Correlation-Attention Masked Temporal Transformer for User Identity Linkage Using Heterogeneous Mobility Data 

**Title (ZH)**: 基于异质移动数据的用户身份关联的相关-注意力掩蔽时序变换器 

**Authors**: Ziang Yan, Xingyu Zhao, Hanqing Ma, Wei Chen, Jianpeng Qi, Yanwei Yu, Junyu Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.01979)  

**Abstract**: With the rise of social media and Location-Based Social Networks (LBSN), check-in data across platforms has become crucial for User Identity Linkage (UIL). These data not only reveal users' spatio-temporal information but also provide insights into their behavior patterns and interests. However, cross-platform identity linkage faces challenges like poor data quality, high sparsity, and noise interference, which hinder existing methods from extracting cross-platform user information. To address these issues, we propose a Correlation-Attention Masked Transformer for User Identity Linkage Network (MT-Link), a transformer-based framework to enhance model performance by learning spatio-temporal co-occurrence patterns of cross-platform users. Our model effectively captures spatio-temporal co-occurrence in cross-platform user check-in sequences. It employs a correlation attention mechanism to detect the spatio-temporal co-occurrence between user check-in sequences. Guided by attention weight maps, the model focuses on co-occurrence points while filtering out noise, ultimately improving classification performance. Experimental results show that our model significantly outperforms state-of-the-art baselines by 12.92%~17.76% and 5.80%~8.38% improvements in terms of Macro-F1 and Area Under Curve (AUC). 

**Abstract (ZH)**: 基于相关性注意机制掩码变压器的用户身份链接网络（MT-Link） 

---
# Universally applicable and tunable graph-based coarse-graining for Machine learning force fields 

**Title (ZH)**: 基于图的通用可调粗粒化方法在机器学习力场中的应用 

**Authors**: Christoph Brunken, Sebastien Boyer, Mustafa Omar, Martin Maarand, Olivier Peltre, Solal Attias, Bakary N'tji Diallo, Anastasia Markina, Olaf Othersen, Oliver Bent  

**Link**: [PDF](https://arxiv.org/pdf/2504.01973)  

**Abstract**: Coarse-grained (CG) force field methods for molecular systems are a crucial tool to simulate large biological macromolecules and are therefore essential for characterisations of biomolecular systems. While state-of-the-art deep learning (DL)-based models for all-atom force fields have improved immensely over recent years, we observe and analyse significant limitations of the currently available approaches for DL-based CG simulations. In this work, we present the first transferable DL-based CG force field approach (i.e., not specific to only one narrowly defined system type) applicable to a wide range of biosystems. To achieve this, our CG algorithm does not rely on hard-coded rules and is tuned to output coarse-grained systems optimised for minimal statistical noise in the ground truth CG forces, which results in significant improvement of model training. Our force field model is also the first CG variant that is based on the MACE architecture and is trained on a custom dataset created by a new approach based on the fragmentation of large biosystems covering protein, RNA and lipid chemistry. We demonstrate that our model can be applied in molecular dynamics simulations to obtain stable and qualitatively accurate trajectories for a variety of systems, while also discussing cases for which we observe limited reliability. 

**Abstract (ZH)**: 粗粒化（CG）力场方法在分子系统中的应用对于模拟大型生物大分子至关重要，因此对于生物分子系统的表征至关重要。尽管基于深度学习（DL）的原子力场模型在近年来取得了显著进步，但我们观察并分析了当前可用的DL基粗粒化方法的重要局限性。在此工作中，我们提出了首个适用于广泛生物系统的可转移DL基粗粒化力场方法（即，不限于单一狭义系统类型）。为了实现这一点，我们的粗粒化算法不依赖于硬编码规则，并且是为在真实CG力最小的统计噪声下优化输出粗粒化系统而调整的，从而显著改进了模型训练。此外，我们的力场模型是首个基于MACE架构的CG变体，并在一种新的基于大型生物系统碎片化方法创建的自定义数据集上进行了训练，涵盖了蛋白质、RNA和脂质化学。我们证明，该模型可以应用于分子动力学模拟，以获得多种系统的稳定且定性准确的轨迹，并讨论了一些我们观察到可靠性有限的情况。 

---
# Differentiable Optimization for Deep Learning-Enhanced DC Approximation of AC Optimal Power Flow 

**Title (ZH)**: 基于可微优化的深度学习增强的交流最优功率流的DC逼近 

**Authors**: Andrew Rosemberg, Michael Klamkin  

**Link**: [PDF](https://arxiv.org/pdf/2504.01970)  

**Abstract**: The growing scale of power systems and the increasing uncertainty introduced by renewable energy sources necessitates novel optimization techniques that are significantly faster and more accurate than existing methods. The AC Optimal Power Flow (AC-OPF) problem, a core component of power grid optimization, is often approximated using linearized DC Optimal Power Flow (DC-OPF) models for computational tractability, albeit at the cost of suboptimal and inefficient decisions. To address these limitations, we propose a novel deep learning-based framework for network equivalency that enhances DC-OPF to more closely mimic the behavior of AC-OPF. The approach utilizes recent advances in differentiable optimization, incorporating a neural network trained to predict adjusted nodal shunt conductances and branch susceptances in order to account for nonlinear power flow behavior. The model can be trained end-to-end using modern deep learning frameworks by leveraging the implicit function theorem. Results demonstrate the framework's ability to significantly improve prediction accuracy, paving the way for more reliable and efficient power systems. 

**Abstract (ZH)**: 随着电力系统规模的扩大和可再生能源引入的不确定性增加，亟需比现有方法更快更准确的新优化技术。AC最优功率流（AC-OPF）问题是电力网络优化的核心组成部分，通常为了计算上的可操作性，通过线性化的直流最优功率流（DC-OPF）模型进行近似，但会导致次优和低效的决策。为此，我们提出了一种基于深度学习的网络等效新框架，以使DC-OPF更接近模拟AC-OPF的行为。该方法利用可微优化的最新进展，结合一个经过训练以预测调整节点分流和支路电纳的神经网络，以考虑非线性功率流行为。该模型可以通过利用隐函数定理使用现代深度学习框架进行端到端训练。结果表明，该框架能够显著提高预测准确性，为更可靠和高效的电力系统铺平了道路。 

---
