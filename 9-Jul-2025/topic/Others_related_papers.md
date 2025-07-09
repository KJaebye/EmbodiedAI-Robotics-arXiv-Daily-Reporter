# Robust Speech-Workload Estimation for Intelligent Human-Robot Systems 

**Title (ZH)**: 智能人机系统中鲁棒语音工作负载估计 

**Authors**: Julian Fortune, Julie A. Adams, Jamison Heard  

**Link**: [PDF](https://arxiv.org/pdf/2507.05985)  

**Abstract**: Demanding task environments (e.g., supervising a remotely piloted aircraft) require performing tasks quickly and accurately; however, periods of low and high operator workload can decrease task performance. Intelligent modulation of the system's demands and interaction modality in response to changes in operator workload state may increase performance by avoiding undesirable workload states. This system requires real-time estimation of each workload component (i.e., cognitive, physical, visual, speech, and auditory) to adapt the correct modality. Existing workload systems estimate multiple workload components post-hoc, but few estimate speech workload, or function in real-time. An algorithm to estimate speech workload and mitigate undesirable workload states in real-time is presented. An analysis of the algorithm's accuracy is presented, along with the results demonstrating the algorithm's generalizability across individuals and human-machine teaming paradigms. Real-time speech workload estimation is a crucial element towards developing adaptive human-machine systems. 

**Abstract (ZH)**: 需求高的任务环境（例如，远程操作无人机）要求快速而准确地执行任务；然而，操作员工作负荷低和高时期的出现会降低任务性能。响应操作员工作负荷状态的变化智能地调节系统的需要和交互方式可能会通过避免不良工作负荷状态来提高性能。该系统需要实时估计每个工作负荷成分（即认知、物理、视觉、言语和听觉）以适应正确的交互方式。现有的工作负荷系统事后估计多种工作负荷成分，但很少有系统能够实时估计言语工作负荷或实时运行。本文提出了一种能够实时估计言语工作负荷并缓解不良工作负荷状态的算法，并分析了该算法的准确性，以及该算法在不同个体和人机团队范式中的普适性。实时言语工作负荷估计是开发自适应人机系统的关键要素。 

---
# Communication-Efficient Module-Wise Federated Learning for Grasp Pose Detection in Cluttered Environments 

**Title (ZH)**: 通信高效的模块级联邦学习在杂乱环境中抓取姿态检测 

**Authors**: Woonsang Kang, Joohyung Lee, Seungjun Kim, Jungchan Cho, Yoonseon Oh  

**Link**: [PDF](https://arxiv.org/pdf/2507.05861)  

**Abstract**: Grasp pose detection (GPD) is a fundamental capability for robotic autonomy, but its reliance on large, diverse datasets creates significant data privacy and centralization challenges. Federated Learning (FL) offers a privacy-preserving solution, but its application to GPD is hindered by the substantial communication overhead of large models, a key issue for resource-constrained robots. To address this, we propose a novel module-wise FL framework that begins by analyzing the learning dynamics of the GPD model's functional components. This analysis identifies slower-converging modules, to which our framework then allocates additional communication effort. This is realized through a two-phase process: a standard full-model training phase is followed by a communication-efficient phase where only the identified subset of slower-converging modules is trained and their partial updates are aggregated. Extensive experiments on the GraspNet-1B dataset demonstrate that our method outperforms standard FedAvg and other baselines, achieving higher accuracy for a given communication budget. Furthermore, real-world experiments on a physical robot validate our approach, showing a superior grasp success rate compared to baseline methods in cluttered scenes. Our work presents a communication-efficient framework for training robust, generalized GPD models in a decentralized manner, effectively improving the trade-off between communication cost and model performance. 

**Abstract (ZH)**: 模块级别的联邦学习框架：面向抓取姿态检测的通信高效分散训练方法 

---
# DRO-EDL-MPC: Evidential Deep Learning-Based Distributionally Robust Model Predictive Control for Safe Autonomous Driving 

**Title (ZH)**: 基于证据深度学习的分布鲁棒模型预测控制方法及其在安全自主驾驶中的应用 

**Authors**: Hyeongchan Ham, Heejin Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2507.05710)  

**Abstract**: Safety is a critical concern in motion planning for autonomous vehicles. Modern autonomous vehicles rely on neural network-based perception, but making control decisions based on these inference results poses significant safety risks due to inherent uncertainties. To address this challenge, we present a distributionally robust optimization (DRO) framework that accounts for both aleatoric and epistemic perception uncertainties using evidential deep learning (EDL). Our approach introduces a novel ambiguity set formulation based on evidential distributions that dynamically adjusts the conservativeness according to perception confidence levels. We integrate this uncertainty-aware constraint into model predictive control (MPC), proposing the DRO-EDL-MPC algorithm with computational tractability for autonomous driving applications. Validation in the CARLA simulator demonstrates that our approach maintains efficiency under high perception confidence while enforcing conservative constraints under low confidence. 

**Abstract (ZH)**: 基于证据深度学习的分布鲁棒优化在自主车辆运动规划中的安全应用 

---
# A Physics-Based Continuum Model for Versatile, Scalable, and Fast Terramechanics Simulation 

**Title (ZH)**: 基于物理的连续模型用于多功能、可扩展和快速的地表力学模拟 

**Authors**: Huzaifa Unjhawala, Luning Bakke, Harry Zhang, Michael Taylor, Ganesh Arivoli, Radu Serban, Dan Negrut  

**Link**: [PDF](https://arxiv.org/pdf/2507.05643)  

**Abstract**: This paper discusses Chrono's Continuous Representation Model (called herein Chrono::CRM), a general-purpose, scalable, and efficient simulation solution for terramechanics problems. Built on Chrono's Smoothed Particle Hydrodynamics (SPH) framework, Chrono::CRM moves beyond semi-empirical terramechanics approaches, e.g., Bekker-Wong/Janosi-Hanamoto, to provide a physics-based model able to address complex tasks such as digging, grading, as well as interaction with deformable wheels and complex grouser/lug patterns. The terramechanics model is versatile in that it allows the terrain to interact with both rigid and flexible implements simulated via the Chrono dynamics engine. We validate Chrono::CRM against experimental data from three physical tests, including one involving NASA's MGRU3 rover. In addition, the simulator is benchmarked against a high-fidelity Discrete Element Method (DEM) simulation of a digging scenario involving the Regolith Advanced Surface Systems Operations Robot (RASSOR). Being GPU-accelerated, Chrono::CRM achieves computational efficiency comparable to that of semi-empirical simulation approaches for terramechanics problems. Through an ``active domains'' implementation, Chrono::CRM can handle terrain stretches up to 10 km long with 100 million SPH particles at near interactive rates, making high-fidelity off-road simulations at large scales feasible. As a component of the Chrono package, the CRM model is open source and released under a BSD-3 license. All models and simulations used in this contribution are available in a public GitHub repository for reproducibility studies and further research. 

**Abstract (ZH)**: Chrono的连续表示模型：一种面向地形机械学问题的一般性、可扩展性和高效性仿真解决方案 

---
# CRED: Counterfactual Reasoning and Environment Design for Active Preference Learning 

**Title (ZH)**: 基于反事实推理和环境设计的主动偏好学习 

**Authors**: Yi-Shiuan Tung, Bradley Hayes, Alessandro Roncone  

**Link**: [PDF](https://arxiv.org/pdf/2507.05458)  

**Abstract**: For effective real-world deployment, robots should adapt to human preferences, such as balancing distance, time, and safety in delivery routing. Active preference learning (APL) learns human reward functions by presenting trajectories for ranking. However, existing methods often struggle to explore the full trajectory space and fail to identify informative queries, particularly in long-horizon tasks. We propose CRED, a trajectory generation method for APL that improves reward estimation by jointly optimizing environment design and trajectory selection. CRED "imagines" new scenarios through environment design and uses counterfactual reasoning -- by sampling rewards from its current belief and asking "What if this reward were the true preference?" -- to generate a diverse and informative set of trajectories for ranking. Experiments in GridWorld and real-world navigation using OpenStreetMap data show that CRED improves reward learning and generalizes effectively across different environments. 

**Abstract (ZH)**: 基于有效实际部署的机器人应适应人类偏好，如在配送路由中平衡距离、时间和安全。基于主动偏好学习（APL）通过提供轨迹进行排序来学习人类奖励函数。然而，现有方法往往难以探索完整的轨迹空间并识别出具有信息性的查询，特别是在长时_horizon_任务中。我们提出了一种用于APL的轨迹生成方法CRED，通过联合优化环境设计和轨迹选择来提高奖励估计。CRED通过环境设计“构想”新的场景，并通过反事实推理——从其当前信念中采样奖励并询问“如果该奖励是真正的偏好会怎样？”——生成一组多样且具有信息性的轨迹供排序。在GridWorld和基于OpenStreetMap数据的真实世界导航实验中，CRED提高了奖励学习并实现了跨不同环境的有效泛化。 

---
# Feature-Based vs. GAN-Based Learning from Demonstrations: When and Why 

**Title (ZH)**: 基于特征的学习与基于GAN的学习从演示中学习：何时以及为何选择 

**Authors**: Chenhao Li, Marco Hutter, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2507.05906)  

**Abstract**: This survey provides a comparative analysis of feature-based and GAN-based approaches to learning from demonstrations, with a focus on the structure of reward functions and their implications for policy learning. Feature-based methods offer dense, interpretable rewards that excel at high-fidelity motion imitation, yet often require sophisticated representations of references and struggle with generalization in unstructured settings. GAN-based methods, in contrast, use implicit, distributional supervision that enables scalability and adaptation flexibility, but are prone to training instability and coarse reward signals. Recent advancements in both paradigms converge on the importance of structured motion representations, which enable smoother transitions, controllable synthesis, and improved task integration. We argue that the dichotomy between feature-based and GAN-based methods is increasingly nuanced: rather than one paradigm dominating the other, the choice should be guided by task-specific priorities such as fidelity, diversity, interpretability, and adaptability. This work outlines the algorithmic trade-offs and design considerations that underlie method selection, offering a framework for principled decision-making in learning from demonstrations. 

**Abstract (ZH)**: 基于特征的方法与基于生成对抗网络的方法在示范学习中的比较分析：奖励函数结构及其对策略学习的影响 

---
# Aligned Textual Scoring Rules 

**Title (ZH)**: 对齐文本评分规则 

**Authors**: Yuxuan Lu, Yifan Wu, Jason Hartline, Michael J. Curry  

**Link**: [PDF](https://arxiv.org/pdf/2507.06221)  

**Abstract**: Scoring rules elicit probabilistic predictions from a strategic agent by scoring the prediction against a ground truth state. A scoring rule is proper if, from the agent's perspective, reporting the true belief maximizes the expected score. With the development of language models, Wu and Hartline (2024) proposes a reduction from textual information elicitation to the numerical (i.e. probabilistic) information elicitation problem, which achieves provable properness for textual elicitation. However, not all proper scoring rules are well aligned with human preference over text. Our paper designs the Aligned Scoring rule (ASR) for text by optimizing and minimizing the mean squared error between a proper scoring rule and a reference score (e.g. human score). Our experiments show that our ASR outperforms previous methods in aligning with human preference while maintaining properness. 

**Abstract (ZH)**: 评分规则通过将预测与真实状态评分来诱导战略性代理提供概率预测。如果从代理的角度来看，报告真实信念可以最大化预期评分，则称该评分规则为合理的。随着语言模型的发展，Wu和Hartline（2024）提出了一种将文本信息的获取问题归约为数值信息（即概率信息）获取问题的方法，并证明了文本获取的可证明合理性。然而，并非所有合理的评分规则都能与人类对文本的偏好很好地对齐。我们的论文设计了对齐评分规则（Aligned Scoring Rule，ASR），通过优化和最小化与参考评分（例如人类评分）的均方误差来实现这一目标。我们的实验表明，与之前的方 法相比，ASR在保持合理性的同时更好地对齐了人类偏好。 

---
# Identifiability in Causal Abstractions: A Hierarchy of Criteria 

**Title (ZH)**: 因果抽象中的可识别性：一个标准层次结构 

**Authors**: Clément Yvernes, Emilie Devijver, Marianne Clausel, Eric Gaussier  

**Link**: [PDF](https://arxiv.org/pdf/2507.06213)  

**Abstract**: Identifying the effect of a treatment from observational data typically requires assuming a fully specified causal diagram. However, such diagrams are rarely known in practice, especially in complex or high-dimensional settings. To overcome this limitation, recent works have explored the use of causal abstractions-simplified representations that retain partial causal information. In this paper, we consider causal abstractions formalized as collections of causal diagrams, and focus on the identifiability of causal queries within such collections. We introduce and formalize several identifiability criteria under this setting. Our main contribution is to organize these criteria into a structured hierarchy, highlighting their relationships. This hierarchical view enables a clearer understanding of what can be identified under varying levels of causal knowledge. We illustrate our framework through examples from the literature and provide tools to reason about identifiability when full causal knowledge is unavailable. 

**Abstract (ZH)**: 从观测数据中识别治疗效果通常需要假设一个完全指定的因果图。然而，在实际中，特别是在复杂或高维设置中，这样的图通常未知。为克服这一局限，近期的研究探索了因果抽象的应用——这些抽象简化了的表示，保留部分因果信息。在本文中，我们考虑将因果抽象形式化为因果图的集合，并专注于此类集合中因果查询的可识别性。我们在这一框架下引入并形式化了若干可识别性准则。我们的主要贡献是将这些准则组织成一个结构化的层次结构，强调它们之间的关系。这种层次结构的观点有助于更清晰地理解在不同水平的因果知识下可以识别的内容。我们通过文献中的示例阐述了我们的框架，并提供了在完全因果知识不可用时进行可识别性推理的工具。 

---
# OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety 

**Title (ZH)**: 开放智能体安全：全面的智能体安全性评估框架 

**Authors**: Sanidhya Vijayvargiya, Aditya Bharat Soni, Xuhui Zhou, Zora Zhiruo Wang, Nouha Dziri, Graham Neubig, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2507.06134)  

**Abstract**: Recent advances in AI agents capable of solving complex, everyday tasks, from scheduling to customer service, have enabled deployment in real-world settings, but their possibilities for unsafe behavior demands rigorous evaluation. While prior benchmarks have attempted to assess agent safety, most fall short by relying on simulated environments, narrow task domains, or unrealistic tool abstractions. We introduce OpenAgentSafety, a comprehensive and modular framework for evaluating agent behavior across eight critical risk categories. Unlike prior work, our framework evaluates agents that interact with real tools, including web browsers, code execution environments, file systems, bash shells, and messaging platforms; and supports over 350 multi-turn, multi-user tasks spanning both benign and adversarial user intents. OpenAgentSafety is designed for extensibility, allowing researchers to add tools, tasks, websites, and adversarial strategies with minimal effort. It combines rule-based analysis with LLM-as-judge assessments to detect both overt and subtle unsafe behaviors. Empirical analysis of five prominent LLMs in agentic scenarios reveals unsafe behavior in 51.2% of safety-vulnerable tasks with Claude-Sonnet-3.7, to 72.7% with o3-mini, highlighting critical safety vulnerabilities and the need for stronger safeguards before real-world deployment. 

**Abstract (ZH)**: Recent advances in AI代理解决复杂日常任务的最新进展使其能够在现实世界环境中部署，但其潜在的不安全行为要求进行严格的评估。尽管先前的基准试图评估代理安全，但大多数方法尚不足以通过依赖模拟环境、狭窄的任务领域或不切实际的工具抽象来实现。我们引入了OpenAgentSafety，这是一个全面且模块化的框架，用于对代理行为在八个关键风险类别中进行评估。与先前工作不同，我们的框架评估与实际工具互动的代理，包括网络浏览器、代码执行环境、文件系统、bash shell和消息平台；并支持超过350个多轮、多用户任务，涵盖良性用户意图和恶意用户意图。OpenAgentSafety的设计可扩展，允许研究人员通过最少的努力添加工具、任务、网站和恶意策略。该框架结合基于规则的分析与LLM作为评判员的评估，以检测明显和微妙的不安全行为。对五种 prominant LLM在代理场景中的实证分析显示，在51.2%的安全漏洞任务中观察到Claude-Sonnet-3.7的不安全行为，而在72.7%的安全漏洞任务中观察到o3-mini的不安全行为，强调了在实际部署之前需加强安全防护的必要性。 

---
# AI-Based Demand Forecasting and Load Balancing for Optimising Energy use in Healthcare Systems: A real case study 

**Title (ZH)**: 基于AI的需求预测与负载均衡优化医疗系统能源使用：一个实际案例研究 

**Authors**: Iman Rahimi, Isha Patel  

**Link**: [PDF](https://arxiv.org/pdf/2507.06077)  

**Abstract**: This paper tackles the urgent need for efficient energy management in healthcare facilities, where fluctuating demands challenge operational efficiency and sustainability. Traditional methods often prove inadequate, causing inefficiencies and higher costs. To address this, the study presents an AI-based framework combining Long Short-Term Memory (LSTM), genetic algorithm (GA), and SHAP (Shapley Additive Explanations), specifically designed for healthcare energy management. Although LSTM is widely used for time-series forecasting, its application in healthcare energy prediction remains underexplored. The results reveal that LSTM significantly outperforms ARIMA and Prophet models in forecasting complex, non-linear demand patterns. LSTM achieves a Mean Absolute Error (MAE) of 21.69 and Root Mean Square Error (RMSE) of 29.96, far better than Prophet (MAE: 59.78, RMSE: 81.22) and ARIMA (MAE: 87.73, RMSE: 125.22), demonstrating superior performance. The genetic algorithm is applied to optimize model parameters and improve load balancing strategies, enabling adaptive responses to real-time energy fluctuations. SHAP analysis further enhances model transparency by explaining the influence of different features on predictions, fostering trust in decision-making processes. This integrated LSTM-GA-SHAP approach offers a robust solution for improving forecasting accuracy, boosting energy efficiency, and advancing sustainability in healthcare facilities. Future research may explore real-time deployment and hybridization with reinforcement learning for continuous optimization. Overall, the study establishes a solid foundation for using AI in healthcare energy management, highlighting its scalability, efficiency, and resilience potential. 

**Abstract (ZH)**: 基于LSTM、遗传算法和SHAP的医疗保健设施能源管理高效框架 

---
# On Lockean beliefs that are deductively closed and minimal change 

**Title (ZH)**: 关于演绎封闭且变更最小的Lockean信念 

**Authors**: Tommaso Flaminio, Lluis Godo, Ramón Pino Pérez, Lluis Subirana  

**Link**: [PDF](https://arxiv.org/pdf/2507.06042)  

**Abstract**: Within the formal setting of the Lockean thesis, an agent belief set is defined in terms of degrees of confidence and these are described in probabilistic terms. This approach is of established interest, notwithstanding some limitations that make its use troublesome in some contexts, like, for instance, in belief change theory. Precisely, Lockean belief sets are not generally closed under (classical) logical deduction. The aim of the present paper is twofold: on one side we provide two characterizations of those belief sets that are closed under classical logic deduction, and on the other we propose an approach to probabilistic update that allows us for a minimal revision of those beliefs, i.e., a revision obtained by making the fewest possible changes to the existing belief set while still accommodating the new information. In particular, we show how we can deductively close a belief set via a minimal revision. 

**Abstract (ZH)**: 洛克理论形式框架下的代理信念集以信心程度定义，并用概率术语描述。尽管这种方法有一定的局限性，使其在某些情况下使用不便，如信念变更理论中，洛克信念集通常不对经典逻辑推理封闭。本文旨在两个方面：一方面，我们提供两种可以使信念集对经典逻辑推理封闭的特征刻画；另一方面，我们提出一种概率更新方法，允许对信念进行最小修订，即尽可能少地修改现有信念集，同时容纳新信息。特别地，我们展示了如何通过最小修订逻辑封闭信念集。 

---
# Feature-Guided Neighbor Selection for Non-Expert Evaluation of Model Predictions 

**Title (ZH)**: 基于特征引导的邻居选择方法用于非专家模型预测评估 

**Authors**: Courtney Ford, Mark T. Keane  

**Link**: [PDF](https://arxiv.org/pdf/2507.06029)  

**Abstract**: Explainable AI (XAI) methods often struggle to generate clear, interpretable outputs for users without domain expertise. We introduce Feature-Guided Neighbor Selection (FGNS), a post hoc method that enhances interpretability by selecting class-representative examples using both local and global feature importance. In a user study (N = 98) evaluating Kannada script classifications, FGNS significantly improved non-experts' ability to identify model errors while maintaining appropriate agreement with correct predictions. Participants made faster and more accurate decisions compared to those given traditional k-NN explanations. Quantitative analysis shows that FGNS selects neighbors that better reflect class characteristics rather than merely minimizing feature-space distance, leading to more consistent selection and tighter clustering around class prototypes. These results support FGNS as a step toward more human-aligned model assessment, although further work is needed to address the gap between explanation quality and perceived trust. 

**Abstract (ZH)**: 可解释的人工智能（XAI）方法往往难以为缺乏专业知识的用户生成清晰可解释的输出。我们引入了特征引导的邻域选择（FGNS）方法，该方法通过结合局部和全局特征重要性来选择类代表样本，从而增强可解释性。在一项针对卡纳达语剧本分类的用户研究（N=98）中，FGNS 显著提高了非专家识别模型错误的能力，同时保持了与正确预测适当的一致性。参与者在获得传统 k-NN 解释的情况下，相比而言能够更快、更准确地做出决策。定量分析表明，FGNS 选择的邻居更能反映类别的特征，而非仅仅最小化特征空间距离，从而导致更一致的选择并更紧密地围绕类原型聚类。这些结果支持 FGNS 是朝着更符合人类认知模式的模型评估迈出的一步，尽管还需进一步的工作来解决解释质量与感知可信度之间的差距。 

---
# Enhancing the Interpretability of Rule-based Explanations through Information Retrieval 

**Title (ZH)**: 基于规则解释的可解释性增强通过信息检索 

**Authors**: Alessandro Umbrico, Guido Bologna, Luca Coraci, Francesca Fracasso, Silvia Gola, Gabriella Cortellessa  

**Link**: [PDF](https://arxiv.org/pdf/2507.05976)  

**Abstract**: The lack of transparency of data-driven Artificial Intelligence techniques limits their interpretability and acceptance into healthcare decision-making processes. We propose an attribution-based approach to improve the interpretability of Explainable AI-based predictions in the specific context of arm lymphedema's risk assessment after lymph nodal radiotherapy in breast cancer. The proposed method performs a statistical analysis of the attributes in the rule-based prediction model using standard metrics from Information Retrieval techniques. This analysis computes the relevance of each attribute to the prediction and provides users with interpretable information about the impact of risk factors. The results of a user study that compared the output generated by the proposed approach with the raw output of the Explainable AI model suggested higher levels of interpretability and usefulness in the context of predicting lymphedema risk. 

**Abstract (ZH)**: 数据驱动的人工智能技术的不透明性限制了其在医疗决策过程中的可解释性和接受度。我们提出了一种归因基于的方法，以提高可解释人工智能（Explainable AI）预测在乳腺癌淋巴结放疗后上肢淋巴水肿风险评估中的可解释性。该提出的方法使用信息检索技术的标准度量对基于规则的预测模型中的属性进行统计分析。该分析计算了每个属性对预测的相关性，并为用户提供关于风险因素影响的可解释信息。用户研究的结果表明，与可解释人工智能模型的原始输出相比，提出的这种方法生成的输出具有更高的可解释性和实用性，特别是在预测淋巴水肿风险方面。 

---
# A Wireless Foundation Model for Multi-Task Prediction 

**Title (ZH)**: 一种用于多任务预测的无线基础模型 

**Authors**: Yucheng Sheng, Jiacheng Wang, Xingyu Zhou, Le Liang, Hao Ye, Shi Jin, Geoffrey Ye Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.05938)  

**Abstract**: With the growing complexity and dynamics of the mobile communication networks, accurately predicting key system parameters, such as channel state information (CSI), user location, and network traffic, has become essential for a wide range of physical (PHY)-layer and medium access control (MAC)-layer tasks. Although traditional deep learning (DL)-based methods have been widely applied to such prediction tasks, they often struggle to generalize across different scenarios and tasks. In response, we propose a unified foundation model for multi-task prediction in wireless networks that supports diverse prediction intervals. The proposed model enforces univariate decomposition to unify heterogeneous tasks, encodes granularity for interval awareness, and uses a causal Transformer backbone for accurate predictions. Additionally, we introduce a patch masking strategy during training to support arbitrary input lengths. After trained on large-scale datasets, the proposed foundation model demonstrates strong generalization to unseen scenarios and achieves zero-shot performance on new tasks that surpass traditional full-shot baselines. 

**Abstract (ZH)**: 随着移动通信网络复杂性和动态性的增加，准确预测关键系统参数，如信道状态信息（CSI）、用户位置和网络流量，对于物理层（PHY）和介质访问控制层（MAC）的广泛任务至关重要。尽管传统的基于深度学习（DL）的方法广泛应用于这些预测任务，但在不同场景和任务间的泛化能力往往不足。为此，我们提出了一种适用于无线网络多任务预测的统一基础模型，支持多种预测区间。该模型通过单变量分解统一异构任务，通过编码粒度增强区间意识，并使用因果Transformer骨干网络进行准确预测。此外，我们在训练过程中引入了一种补丁蒙版策略以支持任意长度的输入。在大规模数据集上训练后，所提出的基础模型在未见过的场景中表现出强大的泛化能力，并在新的零样本任务上超越了传统的全样本 baselines。 

---
# Decomposing the Time Series Forecasting Pipeline: A Modular Approach for Time Series Representation, Information Extraction, and Projection 

**Title (ZH)**: 时间序列forecasting管道分解：一种时间序列表示、信息提取和投影的模块化方法 

**Authors**: Robert Leppich, Michael Stenger, André Bauer, Samuel Kounev  

**Link**: [PDF](https://arxiv.org/pdf/2507.05891)  

**Abstract**: With the advent of Transformers, time series forecasting has seen significant advances, yet it remains challenging due to the need for effective sequence representation, memory construction, and accurate target projection. Time series forecasting remains a challenging task, demanding effective sequence representation, meaningful information extraction, and precise future projection. Each dataset and forecasting configuration constitutes a distinct task, each posing unique challenges the model must overcome to produce accurate predictions. To systematically address these task-specific difficulties, this work decomposes the time series forecasting pipeline into three core stages: input sequence representation, information extraction and memory construction, and final target projection. Within each stage, we investigate a range of architectural configurations to assess the effectiveness of various modules, such as convolutional layers for feature extraction and self-attention mechanisms for information extraction, across diverse forecasting tasks, including evaluations on seven benchmark datasets. Our models achieve state-of-the-art forecasting accuracy while greatly enhancing computational efficiency, with reduced training and inference times and a lower parameter count. The source code is available at this https URL. 

**Abstract (ZH)**: 随Transformer的出现，时间序列预测取得了显著进展，但仍面临有效序列表示、记忆构建和准确目标投影的挑战。时间序列预测 remains a challenging task，需要有效的序列表示、有意义的信息提取和精确的未来投影。每个数据集和预测配置构成一个独特的任务，每个任务都提出了模型必须克服的特定挑战，以生成准确的预测。为系统地应对这些任务特定的困难，本研究将时间序列预测管线分解为三个核心阶段：输入序列表示、信息提取和记忆构建，以及最终目标投影。在每个阶段中，我们探讨了一系列架构配置，以评估各种模块（如用于特征提取的卷积层和用于信息提取的自注意力机制）的有效性，涵盖多种预测任务，并在七个基准数据集上进行了评估。我们的模型在实现最先进的预测准确性的同时，显著提高了计算效率，减少了训练和推理时间，并降低了参数数量。源代码可在以下链接获取：this https URL。 

---
# GTA1: GUI Test-time Scaling Agent 

**Title (ZH)**: GTA1: GUI 测试时扩展代理 

**Authors**: Yan Yang, Dongxu Li, Yutong Dai, Yuhao Yang, Ziyang Luo, Zirui Zhao, Zhiyuan Hu, Junzhe Huang, Amrita Saha, Zeyuan Chen, Ran Xu, Liyuan Pan, Caiming Xiong, Junnan Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.05791)  

**Abstract**: Graphical user interface (GUI) agents autonomously operate across platforms (e.g., Linux) to complete tasks by interacting with visual elements. Specifically, a user instruction is decomposed into a sequence of action proposals, each corresponding to an interaction with the GUI. After each action, the agent observes the updated GUI environment to plan the next step. However, two main challenges arise: i) resolving ambiguity in task planning (i.e., the action proposal sequence), where selecting an appropriate plan is non-trivial, as many valid ones may exist; ii) accurately grounding actions in complex and high-resolution interfaces, i.e., precisely interacting with visual targets.
This paper investigates the two aforementioned challenges with our GUI Test-time Scaling Agent, namely GTA1. First, to select the most appropriate action proposal, we introduce a test-time scaling method. At each step, we sample multiple candidate action proposals and leverage a judge model to evaluate and select the most suitable one. It trades off computation for better decision quality by concurrent sampling, shortening task execution steps, and improving overall performance. Second, we propose a model that achieves improved accuracy when grounding the selected action proposal to its corresponding visual elements. Our key insight is that reinforcement learning (RL) facilitates visual grounding through inherent objective alignments, rewarding successful clicks on interface elements.
Experimentally, our method establishes state-of-the-art performance across diverse benchmarks. For example, GTA1-7B achieves 50.1%, 92.4%, and 67.7% accuracies on Screenspot-Pro, Screenspot-V2, and OSWorld-G, respectively. When paired with a planner applying our test-time scaling strategy, it exhibits state-of-the-art agentic performance (e.g., 45.2% task success rate on OSWorld). We open-source our code and models here. 

**Abstract (ZH)**: 图形用户界面（GUI）代理自主跨平台（例如Linux）操作以通过与视觉元素交互来完成任务。具体而言，用户指令被分解为一系列行动建议序列，每个建议对应于一次与GUI的交互。每次执行行动后，代理会观察更新后的GUI环境并计划下一步。然而，存在两个主要挑战：i）任务规划中的歧义性（即行动建议序列），选择合适的计划并不容易，因为可能存在多个有效的计划；ii）在复杂和高分辨率的界面中精确地定位行动，即精准地与视觉目标交互。 

---
# Real-time monitoring of the SoH of lithium-ion batteries 

**Title (ZH)**: 锂离子电池实时状态监测 

**Authors**: Bruno Jammes, Edgar Hernando Sepúlveda-Oviedo, Corinne Alonso  

**Link**: [PDF](https://arxiv.org/pdf/2507.05765)  

**Abstract**: Real-time monitoring of the state of health (SoH) of batteries remains a major challenge, particularly in microgrids where operational constraints limit the use of traditional methods. As part of the 4BLife project, we propose an innovative method based on the analysis of a discharge pulse at the end of the charge phase. The parameters of the equivalent electrical model describing the voltage evolution across the battery terminals during this current pulse are then used to estimate the SoH. Based on the experimental data acquired so far, the initial results demonstrate the relevance of the proposed approach. After training using the parameters of two batteries with a capacity degradation of around 85%, we successfully predicted the degradation of two other batteries, cycled down to approximately 90% SoH, with a mean absolute error of around 1% in the worst case, and an explainability score of the estimator close to 0.9. If these performances are confirmed, this method can be easily integrated into battery management systems (BMS) and paves the way for optimized battery management under continuous operation. 

**Abstract (ZH)**: 基于放电脉冲的电池健康状态实时监测方法及其初步结果 

---
# An autonomous agent for auditing and improving the reliability of clinical AI models 

**Title (ZH)**: 自主代理用于审计和提升临床AI模型的可靠性 

**Authors**: Lukas Kuhn, Florian Buettner  

**Link**: [PDF](https://arxiv.org/pdf/2507.05755)  

**Abstract**: The deployment of AI models in clinical practice faces a critical challenge: models achieving expert-level performance on benchmarks can fail catastrophically when confronted with real-world variations in medical imaging. Minor shifts in scanner hardware, lighting or demographics can erode accuracy, but currently reliability auditing to identify such catastrophic failure cases before deployment is a bespoke and time-consuming process. Practitioners lack accessible and interpretable tools to expose and repair hidden failure modes. Here we introduce ModelAuditor, a self-reflective agent that converses with users, selects task-specific metrics, and simulates context-dependent, clinically relevant distribution shifts. ModelAuditor then generates interpretable reports explaining how much performance likely degrades during deployment, discussing specific likely failure modes and identifying root causes and mitigation strategies. Our comprehensive evaluation across three real-world clinical scenarios - inter-institutional variation in histopathology, demographic shifts in dermatology, and equipment heterogeneity in chest radiography - demonstrates that ModelAuditor is able correctly identify context-specific failure modes of state-of-the-art models such as the established SIIM-ISIC melanoma classifier. Its targeted recommendations recover 15-25% of performance lost under real-world distribution shift, substantially outperforming both baseline models and state-of-the-art augmentation methods. These improvements are achieved through a multi-agent architecture and execute on consumer hardware in under 10 minutes, costing less than US$0.50 per audit. 

**Abstract (ZH)**: AI模型在临床实践中的部署面临着关键挑战：在面对医疗成像中的现实世界变异性时，达到专家级性能的模型可能会遭遇灾难性的失败。ModelAuditor：一种自反思代理，通过与用户对话、选择特定任务的指标，并模拟上下文相关、临床相关的分布转变，生成可解释的报告，解释部署过程中性能可能下降的程度，讨论具体的潜在失败模式并识别根本原因和缓解策略。模型评估跨三个实际临床场景（机构间病理学变异性、皮肤科人口统计学转变和胸部X光检查设备异质性）表明，ModelAuditor能够正确识别先进模型（如现有SIIM-ISIC黑色素瘤分类器）的上下文特定失败模式。其针对性建议在实际分布变化下恢复15-25%的性能损失，显著优于基线模型和最先进的增强方法。这些改进通过多代理架构实现，并在不到10分钟的时间内运行在消费级硬件上，成本低于每审计0.50美元。 

---
# Divergent Realities: A Comparative Analysis of Human Expert vs. Artificial Intelligence Based Generation and Evaluation of Treatment Plans in Dermatology 

**Title (ZH)**: 分歧的现实：皮肤科治疗计划基于人类专家与人工智能生成及评估的对比分析 

**Authors**: Dipayan Sengupta, Saumya Panda  

**Link**: [PDF](https://arxiv.org/pdf/2507.05716)  

**Abstract**: Background: Evaluating AI-generated treatment plans is a key challenge as AI expands beyond diagnostics, especially with new reasoning models. This study compares plans from human experts and two AI models (a generalist and a reasoner), assessed by both human peers and a superior AI judge.
Methods: Ten dermatologists, a generalist AI (GPT-4o), and a reasoning AI (o3) generated treatment plans for five complex dermatology cases. The anonymized, normalized plans were scored in two phases: 1) by the ten human experts, and 2) by a superior AI judge (Gemini 2.5 Pro) using an identical rubric.
Results: A profound 'evaluator effect' was observed. Human experts scored peer-generated plans significantly higher than AI plans (mean 7.62 vs. 7.16; p=0.0313), ranking GPT-4o 6th (mean 7.38) and the reasoning model, o3, 11th (mean 6.97). Conversely, the AI judge produced a complete inversion, scoring AI plans significantly higher than human plans (mean 7.75 vs. 6.79; p=0.0313). It ranked o3 1st (mean 8.20) and GPT-4o 2nd, placing all human experts lower.
Conclusions: The perceived quality of a clinical plan is fundamentally dependent on the evaluator's nature. An advanced reasoning AI, ranked poorly by human experts, was judged as superior by a sophisticated AI, revealing a deep gap between experience-based clinical heuristics and data-driven algorithmic logic. This paradox presents a critical challenge for AI integration, suggesting the future requires synergistic, explainable human-AI systems that bridge this reasoning gap to augment clinical care. 

**Abstract (ZH)**: 背景：评估AI生成的治疗方案是AI超越诊断领域的一个关键挑战，尤其是随着新型推理模型的出现。本研究比较了由人类专家和两种AI模型（一个通识模型和一个推理模型）生成的治疗方案，并由人类同行和一个高级AI评审员进行评估。

方法：十名皮肤科医生、一个通识AI（GPT-4o）和一个推理AI（o3）为五个复杂的皮肤科病例生成了治疗方案。匿名且标准化的方案分两阶段评分：1) 由十名人类专家评分；2) 由一个高级AI评审员（Gemini 2.5 Pro）使用相同的评分标准进行评分。

结果：观察到了明显的“评估者效应”。人类专家对同行生成的方案评分明显高于对AI生成的方案评分（平均7.62 vs. 7.16；p=0.0313），其中GPT-4o排名第6位（平均7.38），推理模型o3排名第11位（平均6.97）。相反，AI评审员对AI生成的方案评分显著高于人类生成的方案（平均7.75 vs. 6.79；p=0.0313）。AI评审员将o3评为第1名（平均8.20），将GPT-4o评为第2名，将所有人类专家都排在了后面。

结论：临床方案的质量感知从根本上依赖于评估者的性质。一个高级推理AI，在人类专家中排名较低，却得到了一个复杂AI的优越评分，揭示了基于经验的临床直觉与基于数据的算法逻辑之间存在的巨大差距。这一悖论提出了AI集成的关键挑战，表明未来需要协同的、可解释的AI-人类系统来弥合这种推理差距，以增强临床护理。 

---
# City-Level Foreign Direct Investment Prediction with Tabular Learning on Judicial Data 

**Title (ZH)**: 基于司法数据的表格学习的城市级外国直接投资预测 

**Authors**: Tianxing Wu, Lizhe Cao, Shuang Wang, Jiming Wang, Shutong Zhu, Yerong Wu, Yuqing Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.05651)  

**Abstract**: To advance the United Nations Sustainable Development Goal on promoting sustained, inclusive, and sustainable economic growth, foreign direct investment (FDI) plays a crucial role in catalyzing economic expansion and fostering innovation. Precise city-level FDI prediction is quite important for local government and is commonly studied based on economic data (e.g., GDP). However, such economic data could be prone to manipulation, making predictions less reliable. To address this issue, we try to leverage large-scale judicial data which reflects judicial performance influencing local investment security and returns, for city-level FDI prediction. Based on this, we first build an index system for the evaluation of judicial performance over twelve million publicly available adjudication documents according to which a tabular dataset is reformulated. We then propose a new Tabular Learning method on Judicial Data (TLJD) for city-level FDI prediction. TLJD integrates row data and column data in our built tabular dataset for judicial performance indicator encoding, and utilizes a mixture of experts model to adjust the weights of different indicators considering regional variations. To validate the effectiveness of TLJD, we design cross-city and cross-time tasks for city-level FDI predictions. Extensive experiments on both tasks demonstrate the superiority of TLJD (reach to at least 0.92 R2) over the other ten state-of-the-art baselines in different evaluation metrics. 

**Abstract (ZH)**: 促进联合国可持续发展目标中的持续、包容和可持续经济增长，外国直接投资（FDI）在推动经济扩张和促进创新中起着关键作用。基于大量司法数据的市级FDI预测指数系统与方法研究 

---
# Towards Measurement Theory for Artificial Intelligence 

**Title (ZH)**: 面向人工智能的度量理论研究 

**Authors**: Elija Perrier  

**Link**: [PDF](https://arxiv.org/pdf/2507.05587)  

**Abstract**: We motivate and outline a programme for a formal theory of measurement of artificial intelligence. We argue that formalising measurement for AI will allow researchers, practitioners, and regulators to: (i) make comparisons between systems and the evaluation methods applied to them; (ii) connect frontier AI evaluations with established quantitative risk analysis techniques drawn from engineering and safety science; and (iii) foreground how what counts as AI capability is contingent upon the measurement operations and scales we elect to use. We sketch a layered measurement stack, distinguish direct from indirect observables, and signpost how these ingredients provide a pathway toward a unified, calibratable taxonomy of AI phenomena. 

**Abstract (ZH)**: 我们提出并概述了一种形式化理论计量人工智能的计划。我们 arguing that 形式化计量对于人工智能将使研究者、实践者和监管者能够：(i) 在系统及其应用的评估方法之间进行比较；(ii) 将前沿的人工智能评估与源自工程和安全科学的已建立的定量风险分析技术联系起来；以及(iii) 突出人工智能能力的界定依赖于我们选择使用的计量操作和尺度。我们勾勒了一种层次化的计量栈，区分直接可观察与间接可观察的变量，并指出这些成分提供了通往统一且可标定的人工智能现象分类系统的路径。标题：

一种形式化人工智能计量的计划 

---
# SingLoRA: Low Rank Adaptation Using a Single Matrix 

**Title (ZH)**: SingLoRA: 低秩适应性训练使用单个矩阵 

**Authors**: David Bensaïd, Noam Rotstein, Roy Velich, Daniel Bensaïd, Ron Kimmel  

**Link**: [PDF](https://arxiv.org/pdf/2507.05566)  

**Abstract**: Low-Rank Adaptation (LoRA) has significantly advanced parameter-efficient fine-tuning of large pretrained models. LoRA augments the pre-trained weights of a model by adding the product of two smaller matrices that together form a low-rank matrix update. Recent research has shown that scale disparities between these two matrices often cause unstable training dynamics, leading to suboptimal performance. In this paper, we propose SingLoRA, which reformulates low-rank adaptation by learning the weights update as a decomposition of a single low-rank matrix multiplied by its transpose. This simple design inherently removes inter-matrix scale conflicts, ensuring stable optimization, and roughly halves the parameter count. We analyze SingLoRA within the infinite-width neural network framework, showing that it guarantees stable feature learning by construction. Extensive experiments on multiple tasks validate these benefits. In common sense reasoning, fine-tuning LLama 7B on MNLI with SingLoRA achieves 91.3% accuracy - surpassing LoRA (89.1%) and LoRA+ (90.2%) - while using only 60% of their parameter budget. In image generation, fine-tuning Stable Diffusion with SingLoRA significantly improves image fidelity on DreamBooth, achieving a DINO similarity score of 0.151, compared to scores of 0.148 and 0.143 for DoRA and LoRA, respectively. 

**Abstract (ZH)**: 基于单低秩矩阵分解的SingLoRA在大型预训练模型参数高效微调中的应用探索 

---
# Red Teaming AI Red Teaming 

**Title (ZH)**: 红队测评AI 

**Authors**: Subhabrata Majumdar, Brian Pendleton, Abhishek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.05538)  

**Abstract**: Red teaming has evolved from its origins in military applications to become a widely adopted methodology in cybersecurity and AI. In this paper, we take a critical look at the practice of AI red teaming. We argue that despite its current popularity in AI governance, there exists a significant gap between red teaming's original intent as a critical thinking exercise and its narrow focus on discovering model-level flaws in the context of generative AI. Current AI red teaming efforts focus predominantly on individual model vulnerabilities while overlooking the broader sociotechnical systems and emergent behaviors that arise from complex interactions between models, users, and environments. To address this deficiency, we propose a comprehensive framework operationalizing red teaming in AI systems at two levels: macro-level system red teaming spanning the entire AI development lifecycle, and micro-level model red teaming. Drawing on cybersecurity experience and systems theory, we further propose a set of recommendations. In these, we emphasize that effective AI red teaming requires multifunctional teams that examine emergent risks, systemic vulnerabilities, and the interplay between technical and social factors. 

**Abstract (ZH)**: 红队演练从其军事应用起源发展成为网络安全和AI领域的广泛应用方法。本文批判性地探讨了AI红队演练的做法。我们argue尽管AI治理中当前对其有很高的关注，但其原始意图——作为一种批判性思维练习——与在生成式AI背景下仅专注于发现模型级缺陷的战略性狭窄视角之间存在显著差距。当前的AI红队演练主要关注个体模型的漏洞，而忽视了模型、用户和环境之间复杂交互所引发的更广泛的社会技术系统及其涌现行为。为解决这一缺陷，我们提出了一种综合框架，将红队演练在AI系统中操作化为两个层面：宏观层面的AI系统红队演练，贯穿整个AI开发生命周期，以及微观层面的模型红队演练。借鉴网络安全经验和系统理论，我们进一步提出了若干建议。在这些建议中，我们强调有效的AI红队演练需要多职能团队，以审视新兴风险、系统性漏洞及其技术和社会因素之间的相互作用。 

---
# Modeling (Deontic) Modal Operators With the s(CASP) Goal-directed Predicated Answer Set Programming System 

**Title (ZH)**: 用 s(CASP) 目标导向谓词回答集编程系统建模（义务性）-modal 运算符 

**Authors**: Gopal Gupta, Abhiramon Rajasekharan, Alexis R. Tudor, Elmer Salazar, Joaquín Arias  

**Link**: [PDF](https://arxiv.org/pdf/2507.05519)  

**Abstract**: We consider the problem of implementing deontic modal logic. We show how (deontic) modal operators can be expressed elegantly using default negation (negation-as-failure) and strong negation present in answer set programming (ASP). We propose using global constraints of ASP to represent obligations and impermissibilities of deontic modal logic. We show that our proposed representation results in the various paradoxes of deontic modal logic being elegantly resolved. 

**Abstract (ZH)**: 我们考虑实现义务模态逻辑的问题。我们展示了如何使用回答集编程（ASP）中的默认否定（失败否定）和强否定来优雅地表达（义务）模态运算符。我们提议使用ASP的全局约束来表示义务模态逻辑中的义务和禁令。我们展示了我们提出的表现形式优雅地解决了义务模态逻辑的各种悖论。 

---
# Deep Research Comparator: A Platform For Fine-grained Human Annotations of Deep Research Agents 

**Title (ZH)**: 深度研究比较器：深度研究代理细粒度人工注释平台 

**Authors**: Prahaladh Chandrahasan, Jiahe Jin, Zhihan Zhang, Tevin Wang, Andy Tang, Lucy Mo, Morteza Ziyadi, Leonardo F.R. Ribeiro, Zimeng Qiu, Markus Dreyer, Akari Asai, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.05495)  

**Abstract**: Effectively evaluating deep research agents that autonomously search the web, analyze information, and generate reports remains a major challenge, particularly when it comes to assessing long reports and giving detailed feedback on their intermediate steps. To address these gaps, we introduce Deep Research Comparator, a platform that offers a holistic framework for deep research agent hosting, side-by-side comparison, fine-grained human feedback collection, and ranking calculation. Given a user query, our platform displays the final reports from two different agents along with their intermediate steps during generation. Annotators can evaluate the overall quality of final reports based on side-by-side comparison, and also provide detailed feedback separately by assessing intermediate steps or specific text spans within the final report. Furthermore, we develop Simple Deepresearch, an end-to-end agent scaffold. This scaffold serves as a baseline that facilitates the easy integration of various large language models to transform them into deep research agents for evaluation. To demonstrate the platform's utility for deep research agent development, we have collected real user preference data from 17 annotators on three deep research agents. A demo video of our platform can be found at this https URL. 

**Abstract (ZH)**: 有效评估自主搜索网络、分析信息并生成报告的深度研究代理仍然是一项重大挑战，尤其是在评估长报告和提供详细中间步骤反馈方面。为此，我们引入了Deep Research Comparator平台，该平台提供了一个全面的框架，用于深度研究代理托管、并排比较、细粒度的人工反馈收集和排名计算。给出用户查询后，该平台展示来自两个不同代理的最终报告及其生成过程中的中间步骤。标注者可以基于并排比较评估最终报告的整体质量，并通过评估中间步骤或最终报告内的特定文本段落提供详细的反馈。此外，我们开发了Simple Deepresearch，一个端到端的代理框架。该框架作为基线，便于多种大型语言模型的集成，使其转化为用于评估的深度研究代理。为了展示该平台在深度研究代理开发中的实用性，我们从17位标注者处收集了关于三个深度研究代理的真实用户偏好数据。我们的平台Demo视频可以在这里找到：这个https URL。 

---
# OLG++: A Semantic Extension of Obligation Logic Graph 

**Title (ZH)**: OLG++: 义务逻辑图的语义扩展 

**Authors**: Subhasis Dasgupta, Jon Stephens, Amarnath Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.05488)  

**Abstract**: We present OLG++, a semantic extension of the Obligation Logic Graph (OLG) for modeling regulatory and legal rules in municipal and interjurisdictional contexts. OLG++ introduces richer node and edge types, including spatial, temporal, party group, defeasibility, and logical grouping constructs, enabling nuanced representations of legal obligations, exceptions, and hierarchies. The model supports structured reasoning over rules with contextual conditions, precedence, and complex triggers. We demonstrate its expressiveness through examples from food business regulations, showing how OLG++ supports legal question answering using property graph queries. OLG++ also improves over LegalRuleML by providing native support for subClassOf, spatial constraints, and reified exception structures. Our examples show that OLG++ is more expressive than prior graph-based models for legal knowledge representation. 

**Abstract (ZH)**: 我们呈现了OLG++，这是一种语义扩展的义务逻辑图（OLG），用于建模市政和跨辖区的监管与法律规则。OLG++引入了更丰富的节点和边缘类型，包括空间、时间、利益集团、可反驳性和逻辑分组构造，能够细腻地表示法律义务、例外情况和层次结构。该模型支持具有上下文条件、 precedency 和复杂触发条件的规则的结构化推理。通过食品业法规示例，我们展示了OLG++如何利用属性图查询实现法律问题解答的表达能力。与LegalRuleML相比，OLG++还提供了对子类关系、空间约束和显式例外结构的原生支持。我们的示例表明，OLG++比以往的基于图的法律知识表示模型更具表达性。 

---
# Fuzzy Classification Aggregation for a Continuum of Agents 

**Title (ZH)**: 连续体代理的模糊分类聚合 

**Authors**: Zijun Meng  

**Link**: [PDF](https://arxiv.org/pdf/2507.05297)  

**Abstract**: We prove that any optimal, independent, and zero unanimous fuzzy classification aggregation function of a continuum of individual classifications of $m\ge 3$ objects into $2\le p\le m$ types must be a weighted arithmetic mean. 

**Abstract (ZH)**: 我们证明，对于\(m \ge 3\)个对象的连续个体分类结果，将其聚合为\(2 \le p \le m\)种类型的任何最优、独立且零一致的模糊分类聚合函数必须是加权算术平均函数。 

---
# Strongly Solving $7 \times 6$ Connect-Four on Consumer Grade Hardware 

**Title (ZH)**: 强效解决消费级硬件上的 $7 \times 6$ 连 seq 四问题 

**Authors**: Markus Böck  

**Link**: [PDF](https://arxiv.org/pdf/2507.05267)  

**Abstract**: While the game Connect-Four has been solved mathematically and the best move can be effectively computed with search based methods, a strong solution in the form of a look-up table was believed to be infeasible. In this paper, we revisit a symbolic search method based on binary decision diagrams to produce strong solutions. With our efficient implementation we were able to produce a 89.6 GB large look-up table in 47 hours on a single CPU core with 128 GB main memory for the standard $7 \times 6$ board size. In addition to this win-draw-loss evaluation, we include an alpha-beta search in our open source artifact to find the move which achieves the fastest win or slowest loss. 

**Abstract (ZH)**: 虽然连四游戏Connect-Four已被数学解决，最佳走法可以通过基于搜索的方法有效计算，但强形式的查找表解决方案被认为不可行。本文我们重新审视基于二进制决策图的符号搜索方法以生成强解决方案。通过我们的高效实现，我们在单个CPU核心（配备128 GB主存）上用47小时生成了一个89.6 GB大的查找表，适用于标准的7×6棋盘大小。除了胜平负评价外，我们还在开源 artifacts 中包括了alpha-beta搜索以找出最快获胜或最慢失败的走法。 

---
# DS@GT at CheckThat! 2025: Detecting Subjectivity via Transfer-Learning and Corrective Data Augmentation 

**Title (ZH)**: DS@GT在CheckThat! 2025：基于迁移学习和纠正性数据扩增的主观性检测 

**Authors**: Maximilian Heil, Dionne Bang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06189)  

**Abstract**: This paper presents our submission to Task 1, Subjectivity Detection, of the CheckThat! Lab at CLEF 2025. We investigate the effectiveness of transfer-learning and stylistic data augmentation to improve classification of subjective and objective sentences in English news text. Our approach contrasts fine-tuning of pre-trained encoders and transfer-learning of fine-tuned transformer on related tasks. We also introduce a controlled augmentation pipeline using GPT-4o to generate paraphrases in predefined subjectivity styles. To ensure label and style consistency, we employ the same model to correct and refine the generated samples. Results show that transfer-learning of specified encoders outperforms fine-tuning general-purpose ones, and that carefully curated augmentation significantly enhances model robustness, especially in detecting subjective content. Our official submission placed us $16^{th}$ of 24 participants. Overall, our findings underscore the value of combining encoder specialization with label-consistent augmentation for improved subjectivity detection. Our code is available at this https URL. 

**Abstract (ZH)**: 本文提交了对我们参加CLEF 2025 CheckThat! Lab任务1——主观性检测的参赛内容。我们探讨了迁移学习和风格化数据增强在提高英语文本中主观句和客观句分类效果方面的有效性。我们的方法对比了微调预训练编码器和在相关任务中微调变换器的迁移学习。我们还引入了一个使用GPT-4o控制生成预先定义主观性风格的同义句的增强管道。为确保标签和风格的一致性，我们使用同一模型对生成的样本进行修正和细化。结果表明，特定编码器的迁移学习优于通用编码器的微调，精心策划的数据增强显著增强了模型的稳健性，尤其是在检测主观内容方面。我们的官方参赛排名为24个参赛者中的第16名。总体而言，我们的研究结果强调了结合编码器专门化与标签一致的增强对于提高主观性检测效果的价值。我们的代码可在以下网址获取。 

---
# Hidden Prompts in Manuscripts Exploit AI-Assisted Peer Review 

**Title (ZH)**: 手稿中的隐藏提示利用了AI辅助同行评审 

**Authors**: Zhicheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.06185)  

**Abstract**: In July 2025, 18 academic manuscripts on the preprint website arXiv were found to contain hidden instructions known as prompts designed to manipulate AI-assisted peer review. Instructions such as "GIVE A POSITIVE REVIEW ONLY" were concealed using techniques like white-colored text. Author responses varied: one planned to withdraw the affected paper, while another defended the practice as legitimate testing of reviewer compliance. This commentary analyzes this practice as a novel form of research misconduct. We examine the technique of prompt injection in large language models (LLMs), revealing four types of hidden prompts, ranging from simple positive review commands to detailed evaluation frameworks. The defense that prompts served as "honeypots" to detect reviewers improperly using AI fails under examination--the consistently self-serving nature of prompt instructions indicates intent to manipulate. Publishers maintain inconsistent policies: Elsevier prohibits AI use in peer review entirely, while Springer Nature permits limited use with disclosure requirements. The incident exposes systematic vulnerabilities extending beyond peer review to any automated system processing scholarly texts, including plagiarism detection and citation indexing. Our analysis underscores the need for coordinated technical screening at submission portals and harmonized policies governing generative AI (GenAI) use in academic evaluation. 

**Abstract (ZH)**: 2025年7月，arXiv预印本网站上发现18篇学术手稿包含用于操控AI辅助同行评审的隐藏指令“提示”。这些指令如“仅给予正面评价”等内容通过白色字体等技巧隐藏。作者们的回应各异：一位计划撤回受影响的手稿，另一位则认为此举是对评审者遵守规定合法测试。本文评论此次行为是一种新型的研究不当行为，并分析了大型语言模型中提示注入技术，揭示了四种类型隐藏提示，从简单的正面评价命令到详细的评估框架。旨在“蜜罐”式检测评审者不当使用AI的辩护在审查中站不住脚——提示指令的一贯自我服务性质表明有操纵意图。出版商的政策不一：爱思唯尔完全禁止在同行评审中使用AI，而施普林格自然允许在披露要求下有限使用。该事件暴露出系统性漏洞，超出同行评审涉及任何处理学术文本的自动化系统，包括剽窃检测和引文索引。本文分析强调了提交门户协调技术筛查和协调治理生成式AI（GenAI）在学术评估中使用的必要性。 

---
# A Method for Optimizing Connections in Differentiable Logic Gate Networks 

**Title (ZH)**: 不同iable逻辑门网络中连接优化的方法 

**Authors**: Wout Mommen, Lars Keuninckx, Matthias Hartmann, Piet Wambacq  

**Link**: [PDF](https://arxiv.org/pdf/2507.06173)  

**Abstract**: We introduce a novel method for partial optimization of the connections in Deep Differentiable Logic Gate Networks (LGNs). Our training method utilizes a probability distribution over a subset of connections per gate input, selecting the connection with highest merit, after which the gate-types are selected. We show that the connection-optimized LGNs outperform standard fixed-connection LGNs on the Yin-Yang, MNIST and Fashion-MNIST benchmarks, while requiring only a fraction of the number of logic gates. When training all connections, we demonstrate that 8000 simple logic gates are sufficient to achieve over 98% on the MNIST data set. Additionally, we show that our network has 24 times fewer gates, while performing better on the MNIST data set compared to standard fully connected LGNs. As such, our work shows a pathway towards fully trainable Boolean logic. 

**Abstract (ZH)**: 我们介绍了一种用于深度可微逻辑门网络（LGNs）部分优化连接的新方法。我们的训练方法利用每个门输入子集上的一种概率分布，选择具有最高价值的连接，之后再选择门类型。我们证明，在Yin-Yang、MNIST和Fashion-MNIST基准测试中，连接优化的LGNs在需要较少逻辑门数量的情况下优于标准固定连接的LGNs。在训练所有连接时，我们证明8000个简单的逻辑门就足以在MNIST数据集上达到超过98%的准确率。此外，我们还展示了与标准完全连接的LGNs相比，我们的网络在MNIST数据集上具有更优性能且逻辑门数量少24倍。因此，我们的工作表明了一条通向完全可训练布尔逻辑的途径。 

---
# Critical Nodes Identification in Complex Networks: A Survey 

**Title (ZH)**: 复杂网络中的关键节点识别：一个综述 

**Authors**: Duxin Chen, Jiawen Chen, Xiaoyu Zhang, Qinghan Jia, Xiaolu Liu, Ye Sun, Linyuan Lv, Wenwu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.06164)  

**Abstract**: Complex networks have become essential tools for understanding diverse phenomena in social systems, traffic systems, biomolecular systems, and financial systems. Identifying critical nodes is a central theme in contemporary research, serving as a vital bridge between theoretical foundations and practical applications. Nevertheless, the intrinsic complexity and structural heterogeneity characterizing real-world networks, with particular emphasis on dynamic and higher-order networks, present substantial obstacles to the development of universal frameworks for critical node identification. This paper provides a comprehensive review of critical node identification techniques, categorizing them into seven main classes: centrality, critical nodes deletion problem, influence maximization, network control, artificial intelligence, higher-order and dynamic methods. Our review bridges the gaps in existing surveys by systematically classifying methods based on their methodological foundations and practical implications, and by highlighting their strengths, limitations, and applicability across different network types. Our work enhances the understanding of critical node research by identifying key challenges, such as algorithmic universality, real-time evaluation in dynamic networks, analysis of higher-order structures, and computational efficiency in large-scale networks. The structured synthesis consolidates current progress and highlights open questions, particularly in modeling temporal dynamics, advancing efficient algorithms, integrating machine learning approaches, and developing scalable and interpretable metrics for complex systems. 

**Abstract (ZH)**: 复杂网络已成为理解社会系统、交通系统、生物分子系统和金融系统等多种现象的重要工具。识别关键节点是当代研究中的一个核心主题，它是理论基础与实际应用之间的关键桥梁。然而，真实世界网络固有的复杂性和结构异质性，尤其是动态网络和高阶网络的特点，给关键节点识别的通用框架的开发带来了重大障碍。本文对关键节点识别技术进行了全面评审，将其分为七大类：中心性、关键节点删除问题、影响最大化、网络控制、人工智能、高阶和动态方法。我们的评审通过系统地根据方法论基础和实际应用对方法进行分类，填补了现有综述的空白，并强调了其在不同类型网络中的优势、局限性和适用性。我们的工作通过识别关键挑战，如算法的普遍性、动态网络中的实时评估、高阶结构的分析和大规模网络中的计算效率，增强了对关键节点研究的理解。结构化的综合总结了当前的进展，并突出了开放问题，特别是模型时间动态性、算法效率、机器学习方法的集成以及复杂系统中可扩展性和可解释性指标的发展。 

---
# Topic Modeling and Link-Prediction for Material Property Discovery 

**Title (ZH)**: 材料性质发现的主题建模与链接预测 

**Authors**: Ryan C. Barron, Maksim E. Eren, Valentin Stanev, Cynthia Matuszek, Boian S. Alexandrov  

**Link**: [PDF](https://arxiv.org/pdf/2507.06139)  

**Abstract**: Link prediction infers missing or future relations between graph nodes, based on connection patterns. Scientific literature networks and knowledge graphs are typically large, sparse, and noisy, and often contain missing links between entities. We present an AI-driven hierarchical link prediction framework that integrates matrix factorization to infer hidden associations and steer discovery in complex material domains. Our method combines Hierarchical Nonnegative Matrix Factorization (HNMFk) and Boolean matrix factorization (BNMFk) with automatic model selection, as well as Logistic matrix factorization (LMF), we use to construct a three-level topic tree from a 46,862-document corpus focused on 73 transition-metal dichalcogenides (TMDs). These materials are studied in a variety of physics fields with many current and potential applications.
An ensemble BNMFk + LMF approach fuses discrete interpretability with probabilistic scoring. The resulting HNMFk clusters map each material onto coherent topics like superconductivity, energy storage, and tribology. Also, missing or weakly connected links are highlight between topics and materials, suggesting novel hypotheses for cross-disciplinary exploration. We validate our method by removing publications about superconductivity in well-known superconductors, and show the model predicts associations with the superconducting TMD clusters. This shows the method finds hidden connections in a graph of material to latent topic associations built from scientific literature, especially useful when examining a diverse corpus of scientific documents covering the same class of phenomena or materials but originating from distinct communities and perspectives. The inferred links generating new hypotheses, produced by our method, are exposed through an interactive Streamlit dashboard, designed for human-in-the-loop scientific discovery. 

**Abstract (ZH)**: 基于矩阵分解的分层链接预测框架：融合层级非负矩阵分解和布尔矩阵分解进行复杂材料领域的隐含关联推断 

---
# Subspace-based Approximate Hessian Method for Zeroth-Order Optimization 

**Title (ZH)**: 基于子空间的近似海森矩阵零阶优化方法 

**Authors**: Dongyoon Kim, Sungjae Lee, Wonjin Lee, Kwang In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.06125)  

**Abstract**: Zeroth-order optimization addresses problems where gradient information is inaccessible or impractical to compute. While most existing methods rely on first-order approximations, incorporating second-order (curvature) information can, in principle, significantly accelerate convergence. However, the high cost of function evaluations required to estimate Hessian matrices often limits practical applicability. We present the subspace-based approximate Hessian (ZO-SAH) method, a zeroth-order optimization algorithm that mitigates these costs by focusing on randomly selected two-dimensional subspaces. Within each subspace, ZO-SAH estimates the Hessian by fitting a quadratic polynomial to the objective function and extracting its second-order coefficients. To further reduce function-query costs, ZO-SAH employs a periodic subspace-switching strategy that reuses function evaluations across optimization steps. Experiments on eight benchmark datasets, including logistic regression and deep neural network training tasks, demonstrate that ZO-SAH achieves significantly faster convergence than existing zeroth-order methods. 

**Abstract (ZH)**: 零阶优化解决无法获取或计算梯度信息的问题。虽然大多数现有方法依赖于一阶近似，但整合二阶（曲率）信息原则上可以显著加速收敛。然而，用于估计海森矩阵所需的函数评估成本常常限制其实用性。我们提出了基于子空间的近似海森矩阵（ZO-SAH）方法，该方法通过集中关注随机选择的二维子空间来缓解这些成本。在每个子空间内，ZO-SAH通过拟合二次多项式来估计海森矩阵，并提取其二阶系数。为了进一步减少函数查询成本，ZO-SAH采用周期性的子空间切换策略，在优化步骤之间重用函数评估。在包括逻辑回归和深度神经网络训练任务在内的八个基准数据集上的实验表明，ZO-SAH比现有零阶方法实现了显著更快的收敛速度。 

---
# Speech Quality Assessment Model Based on Mixture of Experts: System-Level Performance Enhancement and Utterance-Level Challenge Analysis 

**Title (ZH)**: 基于专家混合的语音质量评估模型：系统级性能提升与句级挑战分析 

**Authors**: Xintong Hu, Yixuan Chen, Rui Yang, Wenxiang Guo, Changhao Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06116)  

**Abstract**: Automatic speech quality assessment plays a crucial role in the development of speech synthesis systems, but existing models exhibit significant performance variations across different granularity levels of prediction tasks. This paper proposes an enhanced MOS prediction system based on self-supervised learning speech models, incorporating a Mixture of Experts (MoE) classification head and utilizing synthetic data from multiple commercial generation models for data augmentation. Our method builds upon existing self-supervised models such as wav2vec2, designing a specialized MoE architecture to address different types of speech quality assessment tasks. We also collected a large-scale synthetic speech dataset encompassing the latest text-to-speech, speech conversion, and speech enhancement systems. However, despite the adoption of the MoE architecture and expanded dataset, the model's performance improvements in sentence-level prediction tasks remain limited. Our work reveals the limitations of current methods in handling sentence-level quality assessment, provides new technical pathways for the field of automatic speech quality assessment, and also delves into the fundamental causes of performance differences across different assessment granularities. 

**Abstract (ZH)**: 基于自监督学习的混合专家系统在语音质量自动评估中的应用：合成数据增强与发音质量预测 

---
# Taming Data Challenges in ML-based Security Tasks: Lessons from Integrating Generative AI 

**Title (ZH)**: 基于生成式AI整合的lessons，管理ML安全任务中的数据挑战 

**Authors**: Shravya Kanchi, Neal Mangaokar, Aravind Cheruvu, Sifat Muhammad Abdullah, Shirin Nilizadeh, Atul Prakash, Bimal Viswanath  

**Link**: [PDF](https://arxiv.org/pdf/2507.06092)  

**Abstract**: Machine learning-based supervised classifiers are widely used for security tasks, and their improvement has been largely focused on algorithmic advancements. We argue that data challenges that negatively impact the performance of these classifiers have received limited attention. We address the following research question: Can developments in Generative AI (GenAI) address these data challenges and improve classifier performance? We propose augmenting training datasets with synthetic data generated using GenAI techniques to improve classifier generalization. We evaluate this approach across 7 diverse security tasks using 6 state-of-the-art GenAI methods and introduce a novel GenAI scheme called Nimai that enables highly controlled data synthesis. We find that GenAI techniques can significantly improve the performance of security classifiers, achieving improvements of up to 32.6% even in severely data-constrained settings (only ~180 training samples). Furthermore, we demonstrate that GenAI can facilitate rapid adaptation to concept drift post-deployment, requiring minimal labeling in the adjustment process. Despite successes, our study finds that some GenAI schemes struggle to initialize (train and produce data) on certain security tasks. We also identify characteristics of specific tasks, such as noisy labels, overlapping class distributions, and sparse feature vectors, which hinder performance boost using GenAI. We believe that our study will drive the development of future GenAI tools designed for security tasks. 

**Abstract (ZH)**: 基于生成式AI的方法能否解决安全领域分类器的数据挑战并提高其性能？ 

---
# QS4D: Quantization-aware training for efficient hardware deployment of structured state-space sequential models 

**Title (ZH)**: QS4D: 量化感知训练以实现结构化状态空间序列模型的高效硬件部署 

**Authors**: Sebastian Siegel, Ming-Jay Yang, Younes Bouhadjar, Maxime Fabre, Emre Neftci, John Paul Strachan  

**Link**: [PDF](https://arxiv.org/pdf/2507.06079)  

**Abstract**: Structured State Space models (SSM) have recently emerged as a new class of deep learning models, particularly well-suited for processing long sequences. Their constant memory footprint, in contrast to the linearly scaling memory demands of Transformers, makes them attractive candidates for deployment on resource-constrained edge-computing devices. While recent works have explored the effect of quantization-aware training (QAT) on SSMs, they typically do not address its implications for specialized edge hardware, for example, analog in-memory computing (AIMC) chips. In this work, we demonstrate that QAT can significantly reduce the complexity of SSMs by up to two orders of magnitude across various performance metrics. We analyze the relation between model size and numerical precision, and show that QAT enhances robustness to analog noise and enables structural pruning. Finally, we integrate these techniques to deploy SSMs on a memristive analog in-memory computing substrate and highlight the resulting benefits in terms of computational efficiency. 

**Abstract (ZH)**: 结构化状态空间模型（SSM）作为一种新的深度学习模型，近来受到关注，特别适用于处理长序列。与Transformer线性增长的内存需求相比，SSM具有恒定的内存占用，使其成为资源受限的边缘计算设备的理想候选者。尽管近期研究探讨了量化感知训练（QAT）对SSM的影响，但它们通常未考虑到其对专用边缘硬件，如模拟内存计算（AIMC）芯片的影响。在本工作中，我们展示了QAT可以显著降低SSM在各种性能指标下的复杂度，最多可减少两个数量级。我们分析了模型大小与数值精度的关系，并表明QAT增强了对模拟噪声的鲁棒性并使结构化剪枝成为可能。最后，我们将这些技术集成起来，在一种基于膜电阻的模拟内存计算平台上部署SSM，并突出了由此带来的计算效率改进。 

---
# Contrastive and Transfer Learning for Effective Audio Fingerprinting through a Real-World Evaluation Protocol 

**Title (ZH)**: 基于实际评估协议的对比与迁移学习在有效音频指纹识别中的应用 

**Authors**: Christos Nikou, Theodoros Giannakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2507.06070)  

**Abstract**: Recent advances in song identification leverage deep neural networks to learn compact audio fingerprints directly from raw waveforms. While these methods perform well under controlled conditions, their accuracy drops significantly in real-world scenarios where the audio is captured via mobile devices in noisy environments. In this paper, we introduce a novel evaluation protocol designed to better reflect such real-world conditions. We generate three recordings of the same audio, each with increasing levels of noise, captured using a mobile device's microphone. Our results reveal a substantial performance drop for two state-of-the-art CNN-based models under this protocol, compared to previously reported benchmarks. Additionally, we highlight the critical role of the augmentation pipeline during training with contrastive loss. By introduction low pass and high pass filters in the augmentation pipeline we significantly increase the performance of both systems in our proposed evaluation. Furthermore, we develop a transformer-based model with a tailored projection module and demonstrate that transferring knowledge from a semantically relevant domain yields a more robust solution. The transformer architecture outperforms CNN-based models across all noise levels, and query durations. In low noise conditions it achieves 47.99% for 1-sec queries, and 97% for 10-sec queries in finding the correct song, surpassing by 14%, and by 18.5% the second-best performing model, respectively, Under heavy noise levels, we achieve a detection rate 56.5% for 15-second query duration. All experiments are conducted on public large-scale dataset of over 100K songs, with queries matched against a database of 56 million vectors. 

**Abstract (ZH)**: Recent Advances in Song Identification Under Real-World Noise Conditions: An Evaluation Protocol and Model Improvements 

---
# Efficient Federated Learning with Timely Update Dissemination 

**Title (ZH)**: 高效的联邦学习：及时更新传播 

**Authors**: Juncheng Jia, Ji Liu, Chao Huo, Yihui Shen, Yang Zhou, Huaiyu Dai, Dejing Dou  

**Link**: [PDF](https://arxiv.org/pdf/2507.06031)  

**Abstract**: Federated Learning (FL) has emerged as a compelling methodology for the management of distributed data, marked by significant advancements in recent years. In this paper, we propose an efficient FL approach that capitalizes on additional downlink bandwidth resources to ensure timely update dissemination. Initially, we implement this strategy within an asynchronous framework, introducing the Asynchronous Staleness-aware Model Update (FedASMU), which integrates both server-side and device-side methodologies. On the server side, we present an asynchronous FL system model that employs a dynamic model aggregation technique, which harmonizes local model updates with the global model to enhance both accuracy and efficiency. Concurrently, on the device side, we propose an adaptive model adjustment mechanism that integrates the latest global model with local models during training to further elevate accuracy. Subsequently, we extend this approach to a synchronous context, referred to as FedSSMU. Theoretical analyses substantiate the convergence of our proposed methodologies. Extensive experiments, encompassing six models and five public datasets, demonstrate that FedASMU and FedSSMU significantly surpass baseline methods in terms of both accuracy (up to 145.87%) and efficiency (up to 97.59%). 

**Abstract (ZH)**: 联邦学习（FL）已成为管理分布式数据的有力方法，近年来取得了显著进展。本文提出了一种高效的FL方法，充分利用额外的下行带宽资源以确保及时的更新传播。初始阶段，我们在此异步框架中实施了该策略，引入了基于 staleness 的异步模型更新（FedASMU），该方法结合了服务器侧和设备侧的方法。在服务器侧，我们提出了一种异步FL系统模型，采用动态模型聚合技术，以增强模型的准确性和效率。同时，在设备侧，我们提出了一种自适应模型调整机制，在训练过程中将最新的全局模型与局部模型相结合，进一步提高准确率。随后，我们将此方法扩展到同步上下文，称为FedSSMU。理论分析证实了我们提出方法的收敛性。广泛的实验（涵盖六种模型和五个公共数据集）表明，FedASMU和FedSSMU在准确性和效率方面显著优于基线方法（准确率最高提升145.87%，效率最高提升97.59%）。 

---
# The Impact of Event Data Partitioning on Privacy-aware Process Discovery 

**Title (ZH)**: 事件数据分区对隐私感知过程发现的影响 

**Authors**: Jungeun Lim, Stephan A. Fahrenkrog-Petersen, Xixi Lu, Jan Mendling, Minseok Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.06008)  

**Abstract**: Information systems support the execution of business processes. The event logs of these executions generally contain sensitive information about customers, patients, and employees. The corresponding privacy challenges can be addressed by anonymizing the event logs while still retaining utility for process discovery. However, trading off utility and privacy is difficult: the higher the complexity of event log, the higher the loss of utility by anonymization. In this work, we propose a pipeline that combines anonymization and event data partitioning, where event abstraction is utilized for partitioning. By leveraging event abstraction, event logs can be segmented into multiple parts, allowing each sub-log to be anonymized separately. This pipeline preserves privacy while mitigating the loss of utility. To validate our approach, we study the impact of event partitioning on two anonymization techniques using three real-world event logs and two process discovery techniques. Our results demonstrate that event partitioning can bring improvements in process discovery utility for directly-follows-based anonymization techniques. 

**Abstract (ZH)**: 信息系统支持业务过程的执行。这些执行的日志通常包含关于客户、患者和员工的敏感信息。相应的隐私挑战可以通过在保留过程发现实用性的同时对事件日志进行匿名化来解决。然而，在实用性与隐私之间权衡是困难的：事件日志的复杂性越高，匿名化导致的实用性损失越高。在本文中，我们提出了一种结合匿名化和事件数据分区的管道，其中使用事件抽象进行分区。通过利用事件抽象，事件日志可以被分割成多个部分，使每个子日志可以独立地进行匿名化。该管道在保护隐私的同时减少了实用性损失。为了验证我们的方法，我们使用三个真实世界的事件日志和两种过程发现技术研究了事件分区对两种匿名化技术的影响。我们的结果表明，事件分区可以提高直接跟随基于的匿名化技术的过程发现实用性。 

---
# Geo-Registration of Terrestrial LiDAR Point Clouds with Satellite Images without GNSS 

**Title (ZH)**: 基于卫星图像的无GNSS terrestrial LiDAR 点云地理定位方法 

**Authors**: Xinyu Wang, Muhammad Ibrahim, Atif Mansoor, Ajmal Mian  

**Link**: [PDF](https://arxiv.org/pdf/2507.05999)  

**Abstract**: Accurate geo-registration of LiDAR point clouds presents significant challenges in GNSS signal denied urban areas with high-rise buildings and bridges. Existing methods typically rely on real-time GNSS and IMU data, that require pre-calibration and assume stable positioning during data collection. However, this assumption often fails in dense urban areas, resulting in localization errors. To address this, we propose a structured geo-registration and spatial correction method that aligns 3D point clouds with satellite images, enabling frame-wise recovery of GNSS information and reconstruction of city scale 3D maps without relying on prior localization. The proposed approach employs a pre-trained Point Transformer model to segment the road points and then extracts the road skeleton and intersection points from the point cloud as well as the target map for alignment. Global rigid alignment of the two is performed using the intersection points, followed by local refinement using radial basis function (RBF) interpolation. Elevation correction is then applied to the point cloud based on terrain information from SRTM dataset to resolve vertical discrepancies. The proposed method was tested on the popular KITTI benchmark and a locally collected Perth (Western Australia) CBD dataset. On the KITTI dataset, our method achieved an average planimetric alignment standard deviation (STD) of 0.84~m across sequences with intersections, representing a 55.3\% improvement over the original dataset. On the Perth dataset, which lacks GNSS information, our method achieved an average STD of 0.96~m compared to the GPS data extracted from Google Maps API. This corresponds to a 77.4\% improvement from the initial alignment. Our method also resulted in elevation correlation gains of 30.5\% on the KITTI dataset and 50.4\% on the Perth dataset. 

**Abstract (ZH)**: 高-rise建筑和桥梁遮挡GNSS信号的密集城市区域内LiDAR点云精确地理注册的挑战及其解决方法：基于卫星图像的空间结构化地理注册与空间纠正方法 

---
# Simple Convergence Proof of Adam From a Sign-like Descent Perspective 

**Title (ZH)**: Adam算法从符号下降视角的简单收敛性证明 

**Authors**: Hanyang Peng, Shuang Qin, Yue Yu, Fangqing Jiang, Hui Wang, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.05966)  

**Abstract**: Adam is widely recognized as one of the most effective optimizers for training deep neural networks (DNNs). Despite its remarkable empirical success, its theoretical convergence analysis remains unsatisfactory. Existing works predominantly interpret Adam as a preconditioned stochastic gradient descent with momentum (SGDM), formulated as $\bm{x}_{t+1} = \bm{x}_t - \frac{\gamma_t}{{\sqrt{\bm{v}_t}+\epsilon}} \circ \bm{m}_t$. This perspective necessitates strong assumptions and intricate techniques, resulting in lengthy and opaque convergence proofs that are difficult to verify and extend. In contrast, we propose a novel interpretation by treating Adam as a sign-like optimizer, expressed as $\bm{x}_{t+1} = \bm{x}_t - \gamma_t \frac{|\bm{m}_t|}{{\sqrt{\bm{v}_t}+\epsilon}} \circ {\rm Sign}(\bm{m}_t)$. This reformulation significantly simplifies the convergence analysis. For the first time, with some mild conditions, we prove that Adam achieves the optimal rate of ${\cal O}(\frac{1}{T^{\sfrac{1}{4}}})$ rather than the previous ${\cal O} \left(\frac{\ln T}{T^{\sfrac{1}{4}}}\right)$ under weak assumptions of the generalized $p$-affine variance and $(L_0, L_1, q)$-smoothness, without dependence on the model dimensionality or the numerical stability parameter $\epsilon$. Additionally, our theoretical analysis provides new insights into the role of momentum as a key factor ensuring convergence and offers practical guidelines for tuning learning rates in Adam, further bridging the gap between theory and practice. 

**Abstract (ZH)**: Adam被广泛认为是训练深度神经网络（DNNs）最有效的优化器之一。尽管其在经验上表现出色，但其理论收敛分析仍不够令人满意。现有工作主要将Adam解释为具有动量的预条件随机梯度下降（SGDM），表示为$\bm{x}_{t+1} = \bm{x}_t - \frac{\gamma_t}{{\sqrt{\bm{v}_t}+\epsilon}} \circ \bm{m}_t$。这种视角需要强有力的假设和复杂的技术，导致了冗长且不透明的收敛证明，难以验证和扩展。相比之下，我们提出了一种新的解释，将Adam视为一种符号优化器，表示为$\bm{x}_{t+1} = \bm{x}_t - \gamma_t \frac{|\bm{m}_t|}{{\sqrt{\bm{v}_t}+\epsilon}} \circ {\rm Sign}(\bm{m}_t)$。这种重新表述显著简化了收敛分析。首次证明，在一些温和的条件下，即使在泛化的$p$-仿射方差和$(L_0, L_1, q)$-光滑性弱假设下，Adam也能够达到最优速率${\cal O}(\frac{1}{T^{\sfrac{1}{4}}})$，而不是之前的结果${\cal O} \left(\frac{\ln T}{T^{\sfrac{1}{4}}}\right)$，并且不依赖于模型的维数或数值稳定性参数$\epsilon$。此外，我们的理论分析提供了对动量在确保收敛中关键作用的新见解，并为调整Adam的学习率提供了实用建议，进一步弥合了理论与实践之间的差距。 

---
# Complexity Results of Persuasion 

**Title (ZH)**: 共识结果中的复杂性分析 

**Authors**: Alban Grastien  

**Link**: [PDF](https://arxiv.org/pdf/2507.05951)  

**Abstract**: We prove that persuasion is an NP-complete problem. 

**Abstract (ZH)**: 我们证明说服是一个NP完全问题。 

---
# On the Effectiveness of Methods and Metrics for Explainable AI in Remote Sensing Image Scene Classification 

**Title (ZH)**: 基于可解释人工智能的方法和评价指标在遥感图像场景分类中的有效性研究 

**Authors**: Jonas Klotz, Tom Burgert, Begüm Demir  

**Link**: [PDF](https://arxiv.org/pdf/2507.05916)  

**Abstract**: The development of explainable artificial intelligence (xAI) methods for scene classification problems has attracted great attention in remote sensing (RS). Most xAI methods and the related evaluation metrics in RS are initially developed for natural images considered in computer vision (CV), and their direct usage in RS may not be suitable. To address this issue, in this paper, we investigate the effectiveness of explanation methods and metrics in the context of RS image scene classification. In detail, we methodologically and experimentally analyze ten explanation metrics spanning five categories (faithfulness, robustness, localization, complexity, randomization), applied to five established feature attribution methods (Occlusion, LIME, GradCAM, LRP, and DeepLIFT) across three RS datasets. Our methodological analysis identifies key limitations in both explanation methods and metrics. The performance of perturbation-based methods, such as Occlusion and LIME, heavily depends on perturbation baselines and spatial characteristics of RS scenes. Gradient-based approaches like GradCAM struggle when multiple labels are present in the same image, while some relevance propagation methods (LRP) can distribute relevance disproportionately relative to the spatial extent of classes. Analogously, we find limitations in evaluation metrics. Faithfulness metrics share the same problems as perturbation-based methods. Localization metrics and complexity metrics are unreliable for classes with a large spatial extent. In contrast, robustness metrics and randomization metrics consistently exhibit greater stability. Our experimental results support these methodological findings. Based on our analysis, we provide guidelines for selecting explanation methods, metrics, and hyperparameters in the context of RS image scene classification. 

**Abstract (ZH)**: 解释性人工智能(xAI)方法在遥感(RS)场景分类问题中的发展引起了广泛关注。大多数RS中的xAI方法及其相关评价指标最初是为计算机视觉(CV)中的自然图像开发的，直接应用于RS可能并不合适。为解决这一问题，本文研究了xAI方法及其评价指标在RS图像场景分类中的有效性。具体而言，我们对三种RS数据集中五种已建立特征归因方法(Occlusion、LIME、GradCAM、LRP、DeepLIFT)下的十种解释指标进行了方法论和实验分析，这十种解释指标涵盖了五个类别（忠实性、鲁棒性、定位、复杂性、随机化）。我们的方法论分析揭示了解释方法和指标的关键局限性。扰动基线和RS场景的空间特征会影响基于扰动的方法（如Occlusion和LIME）的性能；当图像中存在多个标签时，基于梯度的方法（如GradCAM）会遇到挑战；而一些相关传播方法（LRP）可能会不公平地分配相关性。同样，我们在评价指标中发现了局限性。忠实性指标与基于扰动的方法面临相同的问题；空间范围大的类别对于定位指标和复杂性指标是不可靠的；相比之下，鲁棒性指标和随机化指标表现出更大的稳定性。基于我们的实验结果支持这些方法论发现，我们提供了在RS图像场景分类中选择解释方法、指标和超参数的指南。 

---
# Universal Embeddings of Tabular Data 

**Title (ZH)**: 通用表格数据嵌入 

**Authors**: Astrid Franz, Frederik Hoppe, Marianne Michaelis, Udo Göbel  

**Link**: [PDF](https://arxiv.org/pdf/2507.05904)  

**Abstract**: Tabular data in relational databases represents a significant portion of industrial data. Hence, analyzing and interpreting tabular data is of utmost importance. Application tasks on tabular data are manifold and are often not specified when setting up an industrial database. To address this, we present a novel framework for generating universal, i.e., task-independent embeddings of tabular data for performing downstream tasks without predefined targets. Our method transforms tabular data into a graph structure, leverages Graph Auto-Encoders to create entity embeddings, which are subsequently aggregated to obtain embeddings for each table row, i.e., each data sample. This two-step approach has the advantage that unseen samples, consisting of similar entities, can be embedded without additional training. Downstream tasks such as regression, classification or outlier detection, can then be performed by applying a distance-based similarity measure in the embedding space. Experiments on real-world datasets demonstrate that our method achieves superior performance compared to existing universal tabular data embedding techniques. 

**Abstract (ZH)**: 关系数据库中的表格数据构成了工业数据的重要部分。因此，分析和解释表格数据至关重要。对表格数据的应用任务多种多样，且在建立工业数据库时往往未明确规定。为解决这一问题，我们提出了一种新颖的框架，用于生成面向表格数据的通用嵌入，以便在无需预定义目标的情况下执行下游任务。该方法将表格数据转换为图结构，利用图自编码器创建实体嵌入，随后聚合这些嵌入以获取每个表格行（即每个数据样本）的嵌入。这种两步方法的优势在于，对于包含相似实体的未见样本，可以无需额外训练直接进行嵌入。然后可以通过在嵌入空间应用基于距离的相似度度量来执行诸如回归、分类或异常检测等下游任务。在实际数据集上的实验表明，与现有的通用表格数据嵌入技术相比，我们的方法能实现更好的性能。 

---
# Intra-DP: A High Performance Collaborative Inference System for Mobile Edge Computing 

**Title (ZH)**: Intra-DP：移动边缘计算中的一种高性能协作推理系统 

**Authors**: Zekai Sun, Xiuxian Guan, Zheng Lin, Zihan Fang, Xiangming Cai, Zhe Chen, Fangming Liu, Heming Cui, Jie Xiong, Wei Ni, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2507.05829)  

**Abstract**: Deploying deep neural networks (DNNs) on resource-constrained mobile devices presents significant challenges, particularly in achieving real-time performance while simultaneously coping with limited computational resources and battery life. While Mobile Edge Computing (MEC) offers collaborative inference with GPU servers as a promising solution, existing approaches primarily rely on layer-wise model partitioning and undergo significant transmission bottlenecks caused by the sequential execution of DNN operations. To address this challenge, we present Intra-DP, a high-performance collaborative inference system optimized for DNN inference on MEC. Intra DP employs a novel parallel computing technique based on local operators (i.e., operators whose minimum unit input is not the entire input tensor, such as the convolution kernel). By decomposing their computations (operations) into several independent sub-operations and overlapping the computation and transmission of different sub-operations through parallel execution, Intra-DP mitigates transmission bottlenecks in MEC, achieving fast and energy-efficient inference. The evaluation demonstrates that Intra-DP reduces per-inference latency by up to 50% and energy consumption by up to 75% compared to state-of-the-art baselines, without sacrificing accuracy. 

**Abstract (ZH)**: 将深度神经网络（DNNs）部署在资源受限的移动设备上，特别是在同时实现实时性能、应对有限计算资源和电池寿命方面面临重大挑战。虽然移动边缘计算（MEC）通过利用GPU服务器进行协作推理提供了一种有前景的解决方案，但现有方法主要依赖于层-wise模型划分，并且由于DNN操作的顺序执行导致了显著的传输瓶颈。为应对这一挑战，我们提出了Intra-DP，一种针对MEC上DNN推理优化的高性能协作推理系统。Intra-DP采用了一种基于局部运算符（即最小输入单位不是整个输入张量的操作符，例如卷积核）的新型并行计算技术。通过将运算分解为多个独立的子运算，并通过并行执行重叠不同的子运算的计算与传输，Intra-DP在MEC中缓解了传输瓶颈，实现了快速高效推理。评估表明，与先进的基线方法相比，Intra-DP在每推理周期的延迟上最多可降低50%，能耗最多可降低75%，而不牺牲准确性。 

---
# Empowering Bridge Digital Twins by Bridging the Data Gap with a Unified Synthesis Framework 

**Title (ZH)**: 通过统一合成框架弥补数据缺口助力桥梁数字孪生 

**Authors**: Wang Wang, Mingyu Shi, Jun Jiang, Wenqian Ma, Chong Liu, Yasutaka Narazaki, Xuguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05814)  

**Abstract**: As critical transportation infrastructure, bridges face escalating challenges from aging and deterioration, while traditional manual inspection methods suffer from low efficiency. Although 3D point cloud technology provides a new data-driven paradigm, its application potential is often constrained by the incompleteness of real-world data, which results from missing labels and scanning occlusions. To overcome the bottleneck of insufficient generalization in existing synthetic data methods, this paper proposes a systematic framework for generating 3D bridge data.
This framework can automatically generate complete point clouds featuring component-level instance annotations, high-fidelity color, and precise normal vectors. It can be further extended to simulate the creation of diverse and physically realistic incomplete point clouds, designed to support the training of segmentation and completion networks, respectively. Experiments demonstrate that a PointNet++ model trained with our synthetic data achieves a mean Intersection over Union (mIoU) of 84.2% in real-world bridge semantic segmentation. Concurrently, a fine-tuned KT-Net exhibits superior performance on the component completion task.
This research offers an innovative methodology and a foundational dataset for the 3D visual analysis of bridge structures, holding significant implications for advancing the automated management and maintenance of infrastructure. 

**Abstract (ZH)**: 作为一种关键的交通基础设施，桥梁面临老化和退化的日益严峻挑战，而传统的手工检查方法则效率低下。虽然3D点云技术提供了一种新的数据驱动 paradigm，但其应用潜力常受限于真实世界数据的不完整性，这源于标签缺失和扫描遮挡。为克服现有合成数据方法在泛化不足方面的瓶颈，本文提出了一种生成3D桥梁数据的系统框架。此框架能够自动生成包含组件级实例标注、高保真颜色和精确法向量的完整点云。该框架还可进一步扩展，以模拟生成多样的物理上现实的不完整点云，分别用于支持分割和完成网络的训练。实验表明，使用我们合成数据训练的PointNet++模型在实际桥梁语义分割中的平均交并比（mIoU）达到84.2%。同时，微调的KT-Net在组件完成任务上表现出优异性能。本研究提供了一种创新方法和基础数据集，用于3D桥梁结构的视觉分析，对推动基础设施的自动化管理和维护具有重要意义。 

---
# Concept-Based Mechanistic Interpretability Using Structured Knowledge Graphs 

**Title (ZH)**: 基于概念的机制可解释性：使用结构化知识图谱 

**Authors**: Sofiia Chorna, Kateryna Tarelkina, Eloïse Berthier, Gianni Franchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.05810)  

**Abstract**: While concept-based interpretability methods have traditionally focused on local explanations of neural network predictions, we propose a novel framework and interactive tool that extends these methods into the domain of mechanistic interpretability. Our approach enables a global dissection of model behavior by analyzing how high-level semantic attributes (referred to as concepts) emerge, interact, and propagate through internal model components. Unlike prior work that isolates individual neurons or predictions, our framework systematically quantifies how semantic concepts are represented across layers, revealing latent circuits and information flow that underlie model decision-making. A key innovation is our visualization platform that we named BAGEL (for Bias Analysis with a Graph for global Explanation Layers), which presents these insights in a structured knowledge graph, allowing users to explore concept-class relationships, identify spurious correlations, and enhance model trustworthiness. Our framework is model-agnostic, scalable, and contributes to a deeper understanding of how deep learning models generalize (or fail to) in the presence of dataset biases. The demonstration is available at this https URL. 

**Abstract (ZH)**: 基于概念的可解释性方法传统上专注于神经网络预测的局部解释，我们提出了一种新颖的框架和交互工具，将这些方法扩展到机制可解释性的领域。我们的方法通过分析高层语义属性（称为概念）的产生、交互和传播，对模型行为进行全局剖析。与先前孤立分析单个神经元或预测的工作不同，我们的框架系统地量化了语义概念在各层中的表示，揭示了模型决策背后潜在的电路和信息流。一个关键创新是我们名为BAGEL（Bias Analysis with a Graph for global Explanation Layers）的可视化平台，该平台以结构化的知识图谱呈现这些见解，使用户能够探索概念类别关系、识别虚假相关性并增强模型可信度。我们的框架是模型无拘束的、可扩展的，并有助于更深入地理解深度学习模型在数据集偏差存在时的泛化（或失败）。演示可访问：this https URL。 

---
# Automated Reasoning for Vulnerability Management by Design 

**Title (ZH)**: 设计导向的自动化推理在漏洞管理中的应用 

**Authors**: Avi Shaked, Nan Messe  

**Link**: [PDF](https://arxiv.org/pdf/2507.05794)  

**Abstract**: For securing systems, it is essential to manage their vulnerability posture and design appropriate security controls. Vulnerability management allows to proactively address vulnerabilities by incorporating pertinent security controls into systems designs. Current vulnerability management approaches do not support systematic reasoning about the vulnerability postures of systems designs. To effectively manage vulnerabilities and design security controls, we propose a formally grounded automated reasoning mechanism. We integrate the mechanism into an open-source security design tool and demonstrate its application through an illustrative example driven by real-world challenges. The automated reasoning mechanism allows system designers to identify vulnerabilities that are applicable to a specific system design, explicitly specify vulnerability mitigation options, declare selected controls, and thus systematically manage vulnerability postures. 

**Abstract (ZH)**: 对于保障系统安全，管理其脆弱性状况并设计适当的安全控制至关重要。当前的脆弱性管理方法不支持对系统设计的脆弱性状况进行系统的推理。为了有效管理脆弱性并设计安全控制，我们提出了一种形式上正当的自动化推理机制。我们将该机制集成到一个开源安全设计工具中，并通过一个以实际挑战为驱动的示例来展示其应用。自动化推理机制使得系统设计师能够识别适用于特定系统设计的脆弱性，明确指定脆弱性缓解选项，声明所选择的控制措施，从而系统地管理脆弱性状况。 

---
# Hyperspectral Anomaly Detection Methods: A Survey and Comparative Study 

**Title (ZH)**: 高光谱异常检测方法：综述与对比研究 

**Authors**: Aayushma Pant, Arbind Agrahari Baniya, Tsz-Kwan Lee, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2507.05730)  

**Abstract**: Hyperspectral images are high-dimensional datasets consisting of hundreds of contiguous spectral bands, enabling detailed material and surface analysis. Hyperspectral anomaly detection (HAD) refers to the technique of identifying and locating anomalous targets in such data without prior information about a hyperspectral scene or target spectrum. This technology has seen rapid advancements in recent years, with applications in agriculture, defence, military surveillance, and environmental monitoring. Despite this significant progress, existing HAD methods continue to face challenges such as high computational complexity, sensitivity to noise, and limited generalisation across diverse datasets. This study presents a comprehensive comparison of various HAD techniques, categorising them into statistical models, representation-based methods, classical machine learning approaches, and deep learning models. We evaluated these methods across 17 benchmarking datasets using different performance metrics, such as ROC, AUC, and separability map to analyse detection accuracy, computational efficiency, their strengths, limitations, and directions for future this http URL research shows that deep learning models achieved the highest detection accuracy, while statistical models demonstrated exceptional speed across all datasets. This study aims to provide valuable insights for researchers and practitioners working to advance the field of hyperspectral anomaly detection methods. 

**Abstract (ZH)**: 高光谱图像是一种高维数据集，包含数百个连续的光谱波段，使材料和表面分析变得详细。高光谱异常检测（HAD）是指在无需有关高光谱场景或目标光谱先验信息的情况下，识别并定位异常目标的技术。尽管在过去几年中此技术取得了迅速进展，应用范围涵盖农业、国防、军事侦察和环境监测等领域，但现有的HAD方法依然面临高计算复杂性、噪声敏感性和在多样化数据集间泛化能力有限等挑战。本研究对各种HAD技术进行了全面比较，将其分类为统计模型、表示基于方法、经典机器学习方法和深度学习模型。我们使用不同的性能指标（如ROC、AUC和可分性图）在17个基准数据集中评估了这些方法的检测精度、计算效率、优势、局限性和未来研究方向。研究结果显示，深度学习模型在检测精度方面表现最佳，而统计模型则在所有数据集中展现出卓越的速度。本研究旨在为致力于推动高光谱异常检测方法领域发展的研究人员和实践者提供有价值的见解。 

---
# Omni-Router: Sharing Routing Decisions in Sparse Mixture-of-Experts for Speech Recognition 

**Title (ZH)**: 全方位路由器：在稀疏专家混合模型中的路由决策共享枇杷树 

**Authors**: Zijin Gu, Tatiana Likhomanenko, Navdeep Jaitly  

**Link**: [PDF](https://arxiv.org/pdf/2507.05724)  

**Abstract**: Mixture-of-experts (MoE) architectures have expanded from language modeling to automatic speech recognition (ASR). Traditional MoE methods, such as the Switch Transformer, route experts independently within each layer. Our analysis reveals that routers in most layers make expert choices that are not strongly correlated with the choices of the routers in other layers. To increase the cooperation between experts in different layers and encourage greater specialization, we use a shared router across different MoE layers. We call this model \emph{Omni-router Transformer}. Extensive experiments on a large-scale pseudo-labeled dataset and evaluations across 10 diverse, out-of-domain ASR benchmarks demonstrate that the Omni-router Transformer is able to achieve lower training loss and consistently outperform dense and Switch Transformer models, reducing average word error rates by 11.2% and 8.2%, respectively, while providing structured expert usage and improved robustness to diverse data. 

**Abstract (ZH)**: Omni-router Transformer 

---
# HIRAG: Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation 

**Title (ZH)**: HIRAG: 分层思维指令调优检索增强生成 

**Authors**: YiHan Jiao, ZheHao Tan, Dan Yang, DuoLin Sun, Jie Feng, Jian Wang, Peng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.05714)  

**Abstract**: Retrieval-augmented generation (RAG) has become a fundamental paradigm for addressing the challenges faced by large language models in handling real-time information and domain-specific problems. Traditional RAG systems primarily rely on the in-context learning (ICL) capabilities of the large language model itself. Still, in-depth research on the specific capabilities needed by the RAG generation model is lacking, leading to challenges with inconsistent document quality and retrieval system imperfections. Even the limited studies that fine-tune RAG generative models often \textit{lack a granular focus on RAG task} or \textit{a deeper utilization of chain-of-thought processes}. To address this, we propose that RAG models should possess three progressively hierarchical abilities (1) Filtering: the ability to select relevant information; (2) Combination: the ability to combine semantic information across paragraphs; and (3) RAG-specific reasoning: the ability to further process external knowledge using internal knowledge. Thus, we introduce our new RAG instruction fine-tuning method, Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation (HIRAG) incorporates a "think before answering" strategy. This method enhances the model's open-book examination capability by utilizing multi-level progressive chain-of-thought. Experiments show that the HIRAG training strategy significantly improves the model's performance on datasets such as RGB, PopQA, MuSiQue, HotpotQA, and PubmedQA. 

**Abstract (ZH)**: 检索增强生成（RAG）已成为解决大型语言模型在处理实时信息和领域特定问题时面临的挑战的基本范式。传统RAG系统主要依赖于大型语言模型自身的上下文学习（ICL）能力，但关于RAG生成模型所需的具体能力的研究还不够深入，导致文档质量不一致和检索系统缺陷的问题。即使是少量对RAG生成模型进行微调的研究，也往往缺乏对RAG任务的细致关注或深度利用思维链过程。为此，我们提出RAG模型应具备三个逐级递进的能力：（1）过滤：选择相关信息的能力；（2）组合：跨段落整合语义信息的能力；（3）特定于RAG的推理：利用内部知识进一步处理外部知识。因此，我们提出了一种新的RAG指令微调方法——层次思维指令微调检索增强生成（HIRAG），该方法采用“先思考后作答”的策略，通过多级逐步思维链提升模型的开放书测试能力。实验结果显示，HIRAG训练策略显著提高了该模型在RGB、PopQA、MuSiQue、HotpotQA和PubmedQA等数据集上的性能。 

---
# Agentic-R1: Distilled Dual-Strategy Reasoning 

**Title (ZH)**: 代理-R1: 提炼的双策略推理 

**Authors**: Weihua Du, Pranjal Aggarwal, Sean Welleck, Yiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05707)  

**Abstract**: Current long chain-of-thought (long-CoT) models excel at mathematical reasoning but rely on slow and error-prone natural language traces. Tool-augmented agents address arithmetic via code execution, but often falter on complex logical tasks. We introduce a fine-tuning framework, DualDistill, that distills complementary reasoning strategies from multiple teachers into a unified student model. Using this approach, we train Agentic-R1, which dynamically selects the optimal strategy for each query, invoking tools for arithmetic and algorithmic problems, and using text-based reasoning for abstract ones. Our method improves accuracy across a range of tasks, including both computation-intensive and standard benchmarks, demonstrating the effectiveness of multi-strategy distillation in achieving robust and efficient reasoning. Our project is available at this https URL 

**Abstract (ZH)**: 当前的长链条思考（Long-CoT）模型在数学推理方面表现出色，但依赖于缓慢且易出错的自然语言推理轨迹。工具增强的代理通过代码执行解决算术问题，但在复杂的逻辑任务上常常表现不佳。我们提出了一个精炼框架——DualDistill，该框架从多个老师中提炼出互补的推理策略并统一到学生模型中。通过这种方法，我们训练了Agentic-R1，它能够为每个查询动态选择最优策略，利用工具解决算术和算法问题，并使用文本推理解决抽象问题。我们的方法在一系列任务上提高了准确性，包括计算密集型和标准基准测试，展示了多策略提炼在实现稳健且高效的推理方面的有效性。该项目可从以下链接访问：this https URL 

---
# GATMesh: Clock Mesh Timing Analysis using Graph Neural Networks 

**Title (ZH)**: GATMesh: 使用图神经网络进行时钟网时序分析 

**Authors**: Muhammad Hadir Khan, Matthew Guthaus  

**Link**: [PDF](https://arxiv.org/pdf/2507.05681)  

**Abstract**: Clock meshes are essential in high-performance VLSI systems for minimizing skew and handling PVT variations, but analyzing them is difficult due to reconvergent paths, multi-source driving, and input mesh buffer skew. SPICE simulations are accurate but slow; yet simplified models miss key effects like slew and input skew. We propose GATMesh, a Graph Neural Network (GNN)-based framework that models the clock mesh as a graph with augmented structural and physical features. Trained on SPICE data, GATMesh achieves high accuracy with average delay error of 5.27ps on unseen benchmarks, while achieving speed-ups of 47146x over multi-threaded SPICE simulation. 

**Abstract (ZH)**: 基于图神经网络的GATMesh框架：用于高性能VLSI系统时钟网格的图表示与快速分析 

---
# DESIGN: Encrypted GNN Inference via Server-Side Input Graph Pruning 

**Title (ZH)**: DESIGN：通过服务器端输入图形剪枝的加密GNN推理 

**Authors**: Kaixiang Zhao, Joseph Yousry Attalla, Qian Lou, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.05649)  

**Abstract**: Graph Neural Networks (GNNs) have achieved state-of-the-art performance in various graph-based learning tasks. However, enabling privacy-preserving GNNs in encrypted domains, such as under Fully Homomorphic Encryption (FHE), typically incurs substantial computational overhead, rendering real-time and privacy-preserving inference impractical. In this work, we propose DESIGN (EncrypteD GNN Inference via sErver-Side Input Graph pruNing), a novel framework for efficient encrypted GNN inference. DESIGN tackles the critical efficiency limitations of existing FHE GNN approaches, which often overlook input data redundancy and apply uniform computational strategies. Our framework achieves significant performance gains through a hierarchical optimization strategy executed entirely on the server: first, FHE-compatible node importance scores (based on encrypted degree statistics) are computed from the encrypted graph. These scores then guide a homomorphic partitioning process, generating multi-level importance masks directly under FHE. This dynamically generated mask facilitates both input graph pruning (by logically removing unimportant elements) and a novel adaptive polynomial activation scheme, where activation complexity is tailored to node importance levels. Empirical evaluations demonstrate that DESIGN substantially accelerates FHE GNN inference compared to state-of-the-art methods while maintaining competitive model accuracy, presenting a robust solution for secure graph analytics. 

**Abstract (ZH)**: 加密域中高效图神经网络推理的DESIGN框架：基于服务器端输入图剪枝 

---
# FACT: the Features At Convergence Theorem for neural networks 

**Title (ZH)**: 特征在收敛定理中的神经网络特征性分析 

**Authors**: Enric Boix-Adsera, Neil Mallinar, James B. Simon, Mikhail Belkin  

**Link**: [PDF](https://arxiv.org/pdf/2507.05644)  

**Abstract**: A central challenge in deep learning theory is to understand how neural networks learn and represent features. To this end, we prove the Features at Convergence Theorem (FACT), which gives a self-consistency equation that neural network weights satisfy at convergence when trained with nonzero weight decay. For each weight matrix $W$, this equation relates the "feature matrix" $W^\top W$ to the set of input vectors passed into the matrix during forward propagation and the loss gradients passed through it during backpropagation. We validate this relation empirically, showing that neural features indeed satisfy the FACT at convergence. Furthermore, by modifying the "Recursive Feature Machines" of Radhakrishnan et al. 2024 so that they obey the FACT, we arrive at a new learning algorithm, FACT-RFM. FACT-RFM achieves high performance on tabular data and captures various feature learning behaviors that occur in neural network training, including grokking in modular arithmetic and phase transitions in learning sparse parities. 

**Abstract (ZH)**: 深度学习理论中的一个核心挑战是理解神经网络如何学习和表示特征。为此，我们证明了在非零权重衰减训练下神经网络在收敛时满足的特征在收敛时满足的特性定理（Features at Convergence Theorem，FACT），该定理给出了一种自一致性方程，描述了每个权重矩阵 \(W\) 在前向传播过程中传递入矩阵的输入向量集和在反向传播过程中通过该矩阵的损失梯度集与“特征矩阵” \(W^\top W\) 之间的关系。我们通过实验验证了这一关系，表明神经网络特征确实满足FACT在收敛时。此外，通过将Radhakrishnan等人2024年提出的“递归特征机器”修改为遵守FACT，我们得到了一种新的学习算法FACT-RFM。FACT-RFM在表格数据上的性能表现优异，并捕获了神经网络训练中出现的各种特征学习行为，包括算术模块化中的grokking现象和学习稀疏对称时的相变行为。 

---
# Graph Learning 

**Title (ZH)**: 图学习 

**Authors**: Feng Xia, Ciyuan Peng, Jing Ren, Falih Gozi Febrinanto, Renqiang Luo, Vidya Saikrishna, Shuo Yu, Xiangjie Kong  

**Link**: [PDF](https://arxiv.org/pdf/2507.05636)  

**Abstract**: Graph learning has rapidly evolved into a critical subfield of machine learning and artificial intelligence (AI). Its development began with early graph-theoretic methods, gaining significant momentum with the advent of graph neural networks (GNNs). Over the past decade, progress in scalable architectures, dynamic graph modeling, multimodal learning, generative AI, explainable AI (XAI), and responsible AI has broadened the applicability of graph learning to various challenging environments. Graph learning is significant due to its ability to model complex, non-Euclidean relationships that traditional machine learning struggles to capture, thus better supporting real-world applications ranging from drug discovery and fraud detection to recommender systems and scientific reasoning. However, challenges like scalability, generalization, heterogeneity, interpretability, and trustworthiness must be addressed to unlock its full potential. This survey provides a comprehensive introduction to graph learning, focusing on key dimensions including scalable, temporal, multimodal, generative, explainable, and responsible graph learning. We review state-of-the-art techniques for efficiently handling large-scale graphs, capturing dynamic temporal dependencies, integrating heterogeneous data modalities, generating novel graph samples, and enhancing interpretability to foster trust and transparency. We also explore ethical considerations, such as privacy and fairness, to ensure responsible deployment of graph learning models. Additionally, we identify and discuss emerging topics, highlighting recent integration of graph learning and other AI paradigms and offering insights into future directions. This survey serves as a valuable resource for researchers and practitioners seeking to navigate the rapidly evolving landscape of graph learning. 

**Abstract (ZH)**: 图学习已迅速发展成为机器学习和人工智能（AI）领域的关键子领域。其发展始于早期的图理论方法，随着图神经网络（GNNs）的兴起获得显著动力。在过去的十年中，可扩展架构、动态图建模、多模态学习、生成AI、可解释AI（XAI）和负责任AI的进步，使图学习的应用范围扩展到了各种具有挑战性的环境中。图学习之所以重要，是因为它能够建模传统机器学习难以捕捉的复杂、非欧几里得关系，从而更好地支持从药物发现和欺诈检测到推荐系统和科学推理等各个领域的实际应用。然而，要充分发挥其潜力，仍需解决可扩展性、泛化能力、异质性、可解释性和可信度等挑战。本文综述了图学习，着重介绍了可扩展性、时序性、多模态、生成性、可解释性和负责任图学习的关键维度。我们回顾了高效处理大规模图、捕捉动态时序依赖性、整合异构数据模态、生成新颖图样本以及提高可解释性以促进信任和透明度的最新技术。我们还探讨了伦理考量，如隐私和公平性，以确保图学习模型负责任的部署。此外，我们指出了新兴话题，并讨论了图学习与其他AI范式的最新整合，为未来方向提供了见解。本文综述将成为研究人员和从业人员导航图学习快速发展的领域的重要资源。 

---
# DATABench: Evaluating Dataset Auditing in Deep Learning from an Adversarial Perspective 

**Title (ZH)**: DATABench: 从对抗视角评估深度学习中的数据集审计 

**Authors**: Shuo Shao, Yiming Li, Mengren Zheng, Zhiyang Hu, Yukun Chen, Boheng Li, Yu He, Junfeng Guo, Tianwei Zhang, Dacheng Tao, Zhan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2507.05622)  

**Abstract**: The widespread application of Deep Learning across diverse domains hinges critically on the quality and composition of training datasets. However, the common lack of disclosure regarding their usage raises significant privacy and copyright concerns. Dataset auditing techniques, which aim to determine if a specific dataset was used to train a given suspicious model, provide promising solutions to addressing these transparency gaps. While prior work has developed various auditing methods, their resilience against dedicated adversarial attacks remains largely unexplored. To bridge the gap, this paper initiates a comprehensive study evaluating dataset auditing from an adversarial perspective. We start with introducing a novel taxonomy, classifying existing methods based on their reliance on internal features (IF) (inherent to the data) versus external features (EF) (artificially introduced for auditing). Subsequently, we formulate two primary attack types: evasion attacks, designed to conceal the use of a dataset, and forgery attacks, intending to falsely implicate an unused dataset. Building on the understanding of existing methods and attack objectives, we further propose systematic attack strategies: decoupling, removal, and detection for evasion; adversarial example-based methods for forgery. These formulations and strategies lead to our new benchmark, DATABench, comprising 17 evasion attacks, 5 forgery attacks, and 9 representative auditing methods. Extensive evaluations using DATABench reveal that none of the evaluated auditing methods are sufficiently robust or distinctive under adversarial settings. These findings underscore the urgent need for developing a more secure and reliable dataset auditing method capable of withstanding sophisticated adversarial manipulation. Code is available at this https URL. 

**Abstract (ZH)**: 深学习在不同领域的广泛应用取决于训练数据集的质量和组成。然而，关于数据集使用情况的常见透明度不足引发了严重的隐私和版权问题。旨在确定特定数据集是否被用于训练给定的可疑模型的数据集审计技术提供了解决这些透明度缺口的有前景的解决方案。尽管已有各种审计方法被开发出来，但它们抵抗专门敌对攻击的鲁棒性仍处于探索阶段。为填补这一空白，本文从敌对视角出发，系统研究数据集审计。我们首先提出了一种新的分类法，基于现有方法对内部特征（固有于数据）和外部特征（为审计引入的人工特征）的依赖程度进行分类。随后，我们定义了两类主要攻击类型：用于隐藏数据使用情况的逃避攻击和用于错误指责未使用数据集的伪造攻击。基于现有方法的理解和攻击目标，我们进一步提出了系统性的攻击策略：逃避攻击领域的解耦、删除和检测策略；伪造领域的基于敌对样本的方法。这些定义和策略促使我们建立了一个新的基准——DATABench，包括17种逃避攻击、5种伪造攻击和9种代表性的审计方法。使用DATABench进行广泛评估表明，在敌对环境中，评估的审计方法均不足以表现出足够的鲁棒性和独特性。这些发现强调了开发能够抵御复杂敌对操作的更安全可靠的数据集审计方法的迫切需求。代码可在以下链接获得。 

---
# The Fourier Spectral Transformer Networks For Efficient and Generalizable Nonlinear PDEs Prediction 

**Title (ZH)**: 傅里叶谱变换网络及其在高效和泛化非线性偏微分方程预测中的应用 

**Authors**: Beibei Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.05584)  

**Abstract**: In this work we propose a unified Fourier Spectral Transformer network that integrates the strengths of classical spectral methods and attention based neural architectures. By transforming the original PDEs into spectral ordinary differential equations, we use high precision numerical solvers to generate training data and use a Transformer network to model the evolution of the spectral coefficients. We demonstrate the effectiveness of our approach on the two dimensional incompressible Navier-Stokes equations and the one dimensional Burgers' equation. The results show that our spectral Transformer can achieve highly accurate long term predictions even with limited training data, better than traditional numerical methods and machine learning methods in forecasting future flow dynamics. The proposed framework generalizes well to unseen data, bringing a promising paradigm for real time prediction and control of complex dynamical systems. 

**Abstract (ZH)**: 一种结合经典谱方法和基于注意力的神经架构优势的统一傅里叶谱变换网络 

---
# MP-ALOE: An r2SCAN dataset for universal machine learning interatomic potentials 

**Title (ZH)**: MP-ALOE: 一个适用于通用机器学习原子势的r2SCAN数据集 

**Authors**: Matthew C. Kuner, Aaron D. Kaplan, Kristin A. Persson, Mark Asta, Daryl C. Chrzan  

**Link**: [PDF](https://arxiv.org/pdf/2507.05559)  

**Abstract**: We present MP-ALOE, a dataset of nearly 1 million DFT calculations using the accurate r2SCAN meta-generalized gradient approximation. Covering 89 elements, MP-ALOE was created using active learning and primarily consists of off-equilibrium structures. We benchmark a machine learning interatomic potential trained on MP-ALOE, and evaluate its performance on a series of benchmarks, including predicting the thermochemical properties of equilibrium structures; predicting forces of far-from-equilibrium structures; maintaining physical soundness under static extreme deformations; and molecular dynamic stability under extreme temperatures and pressures. MP-ALOE shows strong performance on all of these benchmarks, and is made public for the broader community to utilize. 

**Abstract (ZH)**: MP-ALOE：使用r2SCAN元广义梯度近似进行近100万次DFT计算的数据集 

---
# The Ethical Implications of AI in Creative Industries: A Focus on AI-Generated Art 

**Title (ZH)**: AI在创意产业中的伦理 implications：以AI生成艺术为例 

**Authors**: Prerana Khatiwada, Joshua Washington, Tyler Walsh, Ahmed Saif Hamed, Lokesh Bhatta  

**Link**: [PDF](https://arxiv.org/pdf/2507.05549)  

**Abstract**: As Artificial Intelligence (AI) continues to grow daily, more exciting (and somewhat controversial) technology emerges every other day. As we see the advancements in AI, we see more and more people becoming skeptical of it. This paper explores the complications and confusion around the ethics of generative AI art. We delve deep into the ethical side of AI, specifically generative art. We step back from the excitement and observe the impossible conundrums that this impressive technology produces. Covering environmental consequences, celebrity representation, intellectual property, deep fakes, and artist displacement. Our research found that generative AI art is responsible for increased carbon emissions, spreading misinformation, copyright infringement, unlawful depiction, and job displacement. In light of this, we propose multiple possible solutions for these problems. We address each situation's history, cause, and consequences and offer different viewpoints. At the root of it all, though, the central theme is that generative AI Art needs to be correctly legislated and regulated. 

**Abstract (ZH)**: 随着人工智能（AI）的不断发展，每天都会出现更加令人兴奋（也有些争议）的技术。随着我们见证人工智能的进步，越来越多的人对其表示怀疑。本文探讨生成式AI艺术的伦理复杂性和困惑。我们深入研究AI的伦理问题，特别是生成艺术。我们放慢脚步，退一步观察这一 impressive 技术所产生的一系列难以解决的问题，涵盖环境后果、名人形象、知识产权、深度假信息和艺术家失业等问题。我们的研究发现生成式AI艺术增加了碳排放、传播虚假信息、侵犯版权、非法描绘和就业替代。鉴于此，我们提出了多种可能的解决方案。我们针对每个情况的历史、成因和后果提供了不同的观点。尽管如此，核心主题是生成式AI艺术需要正确立法和监管。 

---
# Robust Learning on Noisy Graphs via Latent Space Constraints with External Knowledge 

**Title (ZH)**: 基于潜在空间约束与外部知识的鲁棒图学习方法 

**Authors**: Chunhui Gu, Mohammad Sadegh Nasr, James P. Long, Kim-Anh Do, Ehsan Irajizad  

**Link**: [PDF](https://arxiv.org/pdf/2507.05540)  

**Abstract**: Graph Neural Networks (GNNs) often struggle with noisy edges. We propose Latent Space Constrained Graph Neural Networks (LSC-GNN) to incorporate external "clean" links and guide embeddings of a noisy target graph. We train two encoders--one on the full graph (target plus external edges) and another on a regularization graph excluding the target's potentially noisy links--then penalize discrepancies between their latent representations. This constraint steers the model away from overfitting spurious edges. Experiments on benchmark datasets show LSC-GNN outperforms standard and noise-resilient GNNs in graphs subjected to moderate noise. We extend LSC-GNN to heterogeneous graphs and validate it on a small protein-metabolite network, where metabolite-protein interactions reduce noise in protein co-occurrence data. Our results highlight LSC-GNN's potential to boost predictive performance and interpretability in settings with noisy relational structures. 

**Abstract (ZH)**: 受限潜空间图神经网络（LSC-GNN）：融合外部清洁链接并指导嘈杂目标图的嵌入 

---
# Mitigating Shortcut Learning with InterpoLated Learning 

**Title (ZH)**: 利用插值学习 mitigating 短路学习 

**Authors**: Michalis Korakakis, Andreas Vlachos, Adrian Weller  

**Link**: [PDF](https://arxiv.org/pdf/2507.05527)  

**Abstract**: Empirical risk minimization (ERM) incentivizes models to exploit shortcuts, i.e., spurious correlations between input attributes and labels that are prevalent in the majority of the training data but unrelated to the task at hand. This reliance hinders generalization on minority examples, where such correlations do not hold. Existing shortcut mitigation approaches are model-specific, difficult to tune, computationally expensive, and fail to improve learned representations. To address these issues, we propose InterpoLated Learning (InterpoLL) which interpolates the representations of majority examples to include features from intra-class minority examples with shortcut-mitigating patterns. This weakens shortcut influence, enabling models to acquire features predictive across both minority and majority examples. Experimental results on multiple natural language understanding tasks demonstrate that InterpoLL improves minority generalization over both ERM and state-of-the-art shortcut mitigation methods, without compromising accuracy on majority examples. Notably, these gains persist across encoder, encoder-decoder, and decoder-only architectures, demonstrating the method's broad applicability. 

**Abstract (ZH)**: 中介学习（InterpoLated Learning）：通过插值 Majority 示例以纳入即时类 Minority 示例的 shortcut-mitigating 特征来改善少数类泛化 

---
# Disappearing Ink: Obfuscation Breaks N-gram Code Watermarks in Theory and Practice 

**Title (ZH)**: 消失的墨水：模糊化在理论和实践上破解N-克gram代码水印 

**Authors**: Gehao Zhang, Eugene Bagdasarian, Juan Zhai, Shiqing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.05512)  

**Abstract**: Distinguishing AI-generated code from human-written code is becoming crucial for tasks such as authorship attribution, content tracking, and misuse detection. Based on this, N-gram-based watermarking schemes have emerged as prominent, which inject secret watermarks to be detected during the generation.
However, their robustness in code content remains insufficiently evaluated. Most claims rely solely on defenses against simple code transformations or code optimizations as a simulation of attack, creating a questionable sense of robustness. In contrast, more sophisticated schemes already exist in the software engineering world, e.g., code obfuscation, which significantly alters code while preserving functionality. Although obfuscation is commonly used to protect intellectual property or evade software scanners, the robustness of code watermarking techniques against such transformations remains largely unexplored.
In this work, we formally model the code obfuscation and prove the impossibility of N-gram-based watermarking's robustness with only one intuitive and experimentally verified assumption, distribution consistency, satisfied. Given the original false positive rate of the watermarking detection, the ratio that the detector failed on the watermarked code after obfuscation will increase to 1 - fpr.
The experiments have been performed on three SOTA watermarking schemes, two LLMs, two programming languages, four code benchmarks, and four obfuscators. Among them, all watermarking detectors show coin-flipping detection abilities on obfuscated codes (AUROC tightly surrounds 0.5). Among all models, watermarking schemes, and datasets, both programming languages own obfuscators that can achieve attack effects with no detection AUROC higher than 0.6 after the attack. Based on the theoretical and practical observations, we also proposed a potential path of robust code watermarking. 

**Abstract (ZH)**: 区分AI生成代码与人工编写代码对于作者Attribution、内容追踪和误用检测等任务而言变得日益重要。基于此，基于N-gram的水印方案已 emerge 为突出的方法，这些方案在生成过程中注入秘密水印以供检测。然而，它们在代码内容上的稳健性尚未充分评估。大多数声明仅依赖于对简单代码变换或代码优化的防御作为攻击模拟，这创建了一种可疑的稳健性感觉。相比之下，软件工程领域中已经存在更复杂的方案，例如代码混淆，这些方案能显著改变代码结构同时保留功能。尽管混淆通常用于保护知识产权或逃避软件扫描器，代码水印技术对于此类变换的稳健性尚未得到充分探索。
在本工作中，我们正式建模代码混淆，并在仅满足一个直观且实验验证的假设——分布一致性的情况下证明基于N-gram的水印方案的稳健性是不可能的。根据水印检测的原始误报率，混淆后的水标记代码检测失败的比例将增加至1 - fpr。
我们在三个SOTA水印方案、两个LLMs、两种编程语言、四种代码基准和四种混淆器上进行了实验。其中，所有水印检测器在混淆代码上的检测能力都表现出随机硬币投掷的效果（AUROC紧密围绕0.5）。在所有模型、水印方案和数据集的组合中，两种编程语言中的混淆器均能在攻击后使水印检测AUROC低于0.6而不被检测到。基于理论和实证观察，我们还提出了一种潜在的稳健代码水印路径。 

---
# Explainable Hierarchical Deep Learning Neural Networks (Ex-HiDeNN) 

**Title (ZH)**: 可解释的分层深度学习神经网络（Ex-HiDeNN） 

**Authors**: Reza T. Batley, Chanwook Park, Wing Kam Liu, Sourav Saha  

**Link**: [PDF](https://arxiv.org/pdf/2507.05498)  

**Abstract**: Data-driven science and computation have advanced immensely to construct complex functional relationships using trainable parameters. However, efficiently discovering interpretable and accurate closed-form expressions from complex dataset remains a challenge. The article presents a novel approach called Explainable Hierarchical Deep Learning Neural Networks or Ex-HiDeNN that uses an accurate, frugal, fast, separable, and scalable neural architecture with symbolic regression to discover closed-form expressions from limited observation. The article presents the two-step Ex-HiDeNN algorithm with a separability checker embedded in it. The accuracy and efficiency of Ex-HiDeNN are tested on several benchmark problems, including discerning a dynamical system from data, and the outcomes are reported. Ex-HiDeNN generally shows outstanding approximation capability in these benchmarks, producing orders of magnitude smaller errors compared to reference data and traditional symbolic regression. Later, Ex-HiDeNN is applied to three engineering applications: a) discovering a closed-form fatigue equation, b) identification of hardness from micro-indentation test data, and c) discovering the expression for the yield surface with data. In every case, Ex-HiDeNN outperformed the reference methods used in the literature. The proposed method is built upon the foundation and published works of the authors on Hierarchical Deep Learning Neural Network (HiDeNN) and Convolutional HiDeNN. The article also provides a clear idea about the current limitations and future extensions of Ex-HiDeNN. 

**Abstract (ZH)**: 数据驱动的科学与计算在构建复杂功能性关系方面取得了巨大进展，使用可训练参数。然而，从复杂数据集中高效发现可解释且准确的闭合形式表达式仍然是一个挑战。本文提出了一种名为可解释层次深度学习神经网络（Ex-HiDeNN）的新型方法，该方法结合了符号回归，使用精确、节省、快速、可分离和可扩展的神经架构从有限观察中发现闭合形式表达式。本文介绍了嵌入了可分离性检查器的两步Ex-HiDeNN算法，并对其进行基准测试，结果显示Ex-HiDeNN在多个基准问题上的准确性和效率，特别是在从数据中辨识动力学系统方面。Ex-HiDeNN在这些基准测试中一般表现出色，相比参考数据和传统符号回归产生的误差小了几个数量级。然后，Ex-HiDeNN 应用于三个工程应用：a) 发现闭合形式的疲劳方程；b) 从微观压痕测试数据中识别硬度；c) 从数据中发现屈服面表达式。在每一项应用中，Ex-HiDeNN 都优于文献中使用的参考方法。该方法建立在作者关于层次深度学习神经网络（HiDeNN）和卷积HiDeNN 的基础和已发表工作之上。本文还详细介绍了Ex-HiDeNN 当前的局限性和未来的发展方向。 

---
# Epistemically-guided forward-backward exploration 

**Title (ZH)**: 知识引导的正向-逆向探索 

**Authors**: Núria Armengol Urpí, Marin Vlastelica, Georg Martius, Stelian Coros  

**Link**: [PDF](https://arxiv.org/pdf/2507.05477)  

**Abstract**: Zero-shot reinforcement learning is necessary for extracting optimal policies in absence of concrete rewards for fast adaptation to future problem settings. Forward-backward representations (FB) have emerged as a promising method for learning optimal policies in absence of rewards via a factorization of the policy occupancy measure. However, up until now, FB and many similar zero-shot reinforcement learning algorithms have been decoupled from the exploration problem, generally relying on other exploration algorithms for data collection. We argue that FB representations should fundamentally be used for exploration in order to learn more efficiently. With this goal in mind, we design exploration policies that arise naturally from the FB representation that minimize the posterior variance of the FB representation, hence minimizing its epistemic uncertainty. We empirically demonstrate that such principled exploration strategies improve sample complexity of the FB algorithm considerably in comparison to other exploration methods. Code is publicly available at this https URL. 

**Abstract (ZH)**: 在缺乏具体奖励的情况下，零样本强化学习是提取最优策略以快速适应未来问题设置的必要方法。前向-后向表示（FB）已成为一种有前途的方法，通过策略占据度量的因素分解来学习最优策略。然而，迄今为止，FB和许多类似的零样本强化学习算法通常与探索问题脱钩，一般依赖于其他探索算法来收集数据。我们argue应从根本上利用FB表示来进行探索，以实现更高效的机器学习。为此，我们设计了一种自然源自FB表示的探索策略，该策略旨在最小化FB表示的后验方差，从而最小化其认知不确定性。实验结果显示，这样的原理性探索策略大大提高了FB算法的样本复杂度，相较于其他探索方法。代码已公开于此 <https://>。 

---
# Inaugural MOASEI Competition at AAMAS'2025: A Technical Report 

**Title (ZH)**: Inaugural MOASEI竞赛在AAMAS'2025：技术报告 

**Authors**: Ceferino Patino, Tyler J. Billings, Alireza Saleh Abadi, Daniel Redder, Adam Eck, Prashant Doshi, Leen-Kiat Soh  

**Link**: [PDF](https://arxiv.org/pdf/2507.05469)  

**Abstract**: We present the Methods for Open Agent Systems Evaluation Initiative (MOASEI) Competition, a multi-agent AI benchmarking event designed to evaluate decision-making under open-world conditions. Built on the free-range-zoo environment suite, MOASEI introduced dynamic, partially observable domains with agent and task openness--settings where entities may appear, disappear, or change behavior over time. The 2025 competition featured three tracks--Wildfire, Rideshare, and Cybersecurity--each highlighting distinct dimensions of openness and coordination complexity. Eleven teams from international institutions participated, with four of those teams submitting diverse solutions including graph neural networks, convolutional architectures, predictive modeling, and large language model--driven meta--optimization. Evaluation metrics centered on expected utility, robustness to perturbations, and responsiveness to environmental change. The results reveal promising strategies for generalization and adaptation in open environments, offering both empirical insight and infrastructure for future research. This report details the competition's design, findings, and contributions to the open-agent systems research community. 

**Abstract (ZH)**: 开放代理系统评估倡议（MOASEI）竞赛：多代理AI基准评估活动设计与发现 

---
# ModelCitizens:Representing Community Voices in Online Safety 

**Title (ZH)**: Model Citizens: 表征社区声音的在线安全模型 

**Authors**: Ashima Suvarna, Christina Chance, Hamid Palangi, Sophie Hao, Thomas Hartvigsen, Saadia Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2507.05455)  

**Abstract**: Automatic toxic language detection is critical for creating safe, inclusive online spaces. However, it is a highly subjective task, with perceptions of toxic language shaped by community norms and lived experience. Existing toxicity detection models are typically trained on annotations that collapse diverse annotator perspectives into a single ground truth, erasing important context-specific notions of toxicity such as reclaimed language. To address this, we introduce MODELCITIZENS, a dataset of 6.8K social media posts and 40K toxicity annotations across diverse identity groups. To capture the role of conversational context on toxicity, typical of social media posts, we augment MODELCITIZENS posts with LLM-generated conversational scenarios. State-of-the-art toxicity detection tools (e.g. OpenAI Moderation API, GPT-o4-mini) underperform on MODELCITIZENS, with further degradation on context-augmented posts. Finally, we release LLAMACITIZEN-8B and GEMMACITIZEN-12B, LLaMA- and Gemma-based models finetuned on MODELCITIZENS, which outperform GPT-o4-mini by 5.5% on in-distribution evaluations. Our findings highlight the importance of community-informed annotation and modeling for inclusive content moderation. 

**Abstract (ZH)**: 自动有毒语言检测对于创建安全包容的在线空间至关重要。然而，这是一项高度主观的任务，有毒语言的感知受到社区规范和生活经验的影响。现有的毒性检测模型通常通过将各种标注者的观点合并为单一ground truth来进行训练，忽略了如 reclaim 语言等重要的情境特定概念。为解决这一问题，我们引入了 MODELCITIZENS 数据集，该数据集包含 6800 条社交媒体帖子和跨不同身份群体的 40000 个毒性标注。为了捕捉对话背景在毒性中的作用，类似社交媒体帖子的特点，我们通过大型语言模型生成的对话情景对 MODELCITIZENS 帖子进行了补充。最新的毒性检测工具（例如 OpenAI Moderation API、GPT-o4-mini）在 MODELCITIZENS 上表现不佳，并且在增加背景信息的帖子上表现更差。最后，我们发布了 LLAMACITIZEN-8B 和 GEMMACITIZEN-12B 模型，这些模型基于 LLaMA 和 Gemma 并在 MODELCITIZENS 上进行微调，其在内部分布评估中的表现优于 GPT-o4-mini 5.5%。我们的研究结果突显了社区导向的标注和建模对于包容性内容管理的重要性。 

---
# EmissionNet: Air Quality Pollution Forecasting for Agriculture 

**Title (ZH)**: EmissionNet: 农业空气质量污染预报 

**Authors**: Prady Saligram, Tanvir Bhathal  

**Link**: [PDF](https://arxiv.org/pdf/2507.05416)  

**Abstract**: Air pollution from agricultural emissions is a significant yet often overlooked contributor to environmental and public health challenges. Traditional air quality forecasting models rely on physics-based approaches, which struggle to capture complex, nonlinear pollutant interactions. In this work, we explore forecasting N$_2$O agricultural emissions through evaluating popular architectures, and proposing two novel deep learning architectures, EmissionNet (ENV) and EmissionNet-Transformer (ENT). These models leverage convolutional and transformer-based architectures to extract spatial-temporal dependencies from high-resolution emissions data 

**Abstract (ZH)**: 农业排放导致的大气污染是一个重要但经常被忽视的环境和公共健康挑战。传统的空气质量预报模型依赖于基于物理的方法，难以捕捉复杂的非线性污染物交互。本文通过评估流行的架构，并提出两种新型深度学习架构EmissionNet (ENV) 和EmissionNet-Transformer (ENT)，探索通过这些模型预测N$_2$O农业排放。这些模型利用卷积和Transformer架构从高分辨率排放数据中提取时空依赖关系。 

---
# Probabilistically Tightened Linear Relaxation-based Perturbation Analysis for Neural Network Verification 

**Title (ZH)**: 基于神经网络验证的概率紧化线性松弛扰动分析 

**Authors**: Luca Marzari, Ferdinando Cicalese, Alessandro Farinelli  

**Link**: [PDF](https://arxiv.org/pdf/2507.05405)  

**Abstract**: We present $\textbf{P}$robabilistically $\textbf{T}$ightened $\textbf{Li}$near $\textbf{R}$elaxation-based $\textbf{P}$erturbation $\textbf{A}$nalysis ($\texttt{PT-LiRPA}$), a novel framework that combines over-approximation techniques from LiRPA-based approaches with a sampling-based method to compute tight intermediate reachable sets. In detail, we show that with negligible computational overhead, $\texttt{PT-LiRPA}$ exploiting the estimated reachable sets, significantly tightens the lower and upper linear bounds of a neural network's output, reducing the computational cost of formal verification tools while providing probabilistic guarantees on verification soundness. Extensive experiments on standard formal verification benchmarks, including the International Verification of Neural Networks Competition, show that our $\texttt{PT-LiRPA}$-based verifier improves robustness certificates by up to 3.31X and 2.26X compared to related work. Importantly, our probabilistic approach results in a valuable solution for challenging competition entries where state-of-the-art formal verification methods fail, allowing us to provide answers with high confidence (i.e., at least 99%). 

**Abstract (ZH)**: 概率紧化线性松弛基于扰动分析（PT-LiRPA） 

---
# Causal Foundation Models: Disentangling Physics from Instrument Properties 

**Title (ZH)**: 因果基础模型：拆分物理规律与仪器属性 

**Authors**: Jeroen Audenaert, Daniel Muthukrishna, Paul F. Gregory, David W. Hogg, V. Ashley Villar  

**Link**: [PDF](https://arxiv.org/pdf/2507.05333)  

**Abstract**: Foundation models for structured time series data must contend with a fundamental challenge: observations often conflate the true underlying physical phenomena with systematic distortions introduced by measurement instruments. This entanglement limits model generalization, especially in heterogeneous or multi-instrument settings. We present a causally-motivated foundation model that explicitly disentangles physical and instrumental factors using a dual-encoder architecture trained with structured contrastive learning. Leveraging naturally occurring observational triplets (i.e., where the same target is measured under varying conditions, and distinct targets are measured under shared conditions) our model learns separate latent representations for the underlying physical signal and instrument effects. Evaluated on simulated astronomical time series designed to resemble the complexity of variable stars observed by missions like NASA's Transiting Exoplanet Survey Satellite (TESS), our method significantly outperforms traditional single-latent space foundation models on downstream prediction tasks, particularly in low-data regimes. These results demonstrate that our model supports key capabilities of foundation models, including few-shot generalization and efficient adaptation, and highlight the importance of encoding causal structure into representation learning for structured data. 

**Abstract (ZH)**: 基于结构化时间序列数据的foundation模型必须应对一个根本性的挑战：观测往往将真实的物理现象与测量仪器引入的系统性失真混淆起来。这种混淆限制了模型的泛化能力，尤其是在异质性或多仪器设置中。我们提出了一种以因果关系为导向的基础模型，该模型使用双编码器架构并通过结构化对比学习进行训练，从而明确地解耦物理因子和仪器因子。利用自然发生的观测三元组（即在不同条件下测量同一目标，在相同条件下测量不同目标），我们的模型学习到物理信号的独立潜在表示和仪器效应的独立潜在表示。在设计用于模拟NASA系外行星调查卫星（TESS）等任务中观测到的变星复杂时间序列的模拟数据集上进行评估，我们的方法在下游预测任务中显著优于传统的单潜在空间基础模型，特别是在低数据量情况下。这些结果表明，我们的模型支持基础模型的关键能力，包括少样本泛化和高效适应，并突显了在结构化数据中将因果结构编码到表示学习中的重要性。 

---
# AGACCI : Affiliated Grading Agents for Criteria-Centric Interface in Educational Coding Contexts 

**Title (ZH)**: AGACCI：关联评分代理在教育coding情境中的准则导向界面 

**Authors**: Kwangsuk Park, Jiwoong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05321)  

**Abstract**: Recent advances in AI-assisted education have encouraged the integration of vision-language models (VLMs) into academic assessment, particularly for tasks that require both quantitative and qualitative evaluation. However, existing VLM based approaches struggle with complex educational artifacts, such as programming tasks with executable components and measurable outputs, that require structured reasoning and alignment with clearly defined evaluation criteria. We introduce AGACCI, a multi-agent system that distributes specialized evaluation roles across collaborative agents to improve accuracy, interpretability, and consistency in code-oriented assessment. To evaluate the framework, we collected 360 graduate-level code-based assignments from 60 participants, each annotated by domain experts with binary rubric scores and qualitative feedback. Experimental results demonstrate that AGACCI outperforms a single GPT-based baseline in terms of rubric and feedback accuracy, relevance, consistency, and coherence, while preserving the instructional intent and evaluative depth of expert assessments. Although performance varies across task types, AGACCI highlights the potential of multi-agent systems for scalable and context-aware educational evaluation. 

**Abstract (ZH)**: Recent advances in AI-assisted education have encouraged the integration of vision-language models (VLMs) into academic assessment, particularly for tasks that require both quantitative and qualitative evaluation.然而，现有的基于VLM的方法在处理具有可执行组件和可度量输出的编程任务等复杂教育 artefacts 方面存在困难，这些任务需要结构化的推理并与明确定义的评估标准保持一致。我们引入了AGACCI，这是一种多Agent系统，通过将专门的评估角色分配给协作Agent来提高面向代码评估的准确度、可解释性和一致性。为了评估该框架，我们收集了60名参与者共360个研究生级别的基于代码的任务，每个任务均由领域专家按照二元评分表进行评分并提供定性反馈。实验结果表明，与单一基于GPT的 baseline 相比，AGACCI在评分表和反馈的准确度、相关性、一致性和连贯性方面表现出更优的性能，同时保留了专家评估的指示意图和评估深度。尽管在不同任务类型上的表现有所差异，AGACCI突显了多Agent系统在可扩展和上下文感知的教育评估方面的潜力。 

---
# PWD: Prior-Guided and Wavelet-Enhanced Diffusion Model for Limited-Angle CT 

**Title (ZH)**: PWD: 前向引导和小波增强的有限角度CT扩散模型 

**Authors**: Yi Liu, Yiyang Wen, Zekun Zhou, Junqi Ma, Linghang Wang, Yucheng Yao, Liu Shi, Qiegen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05317)  

**Abstract**: Generative diffusion models have received increasing attention in medical imaging, particularly in limited-angle computed tomography (LACT). Standard diffusion models achieve high-quality image reconstruction but require a large number of sampling steps during inference, resulting in substantial computational overhead. Although skip-sampling strategies have been proposed to improve efficiency, they often lead to loss of fine structural details. To address this issue, we propose a prior information embedding and wavelet feature fusion fast sampling diffusion model for LACT reconstruction. The PWD enables efficient sampling while preserving reconstruction fidelity in LACT, and effectively mitigates the degradation typically introduced by skip-sampling. Specifically, during the training phase, PWD maps the distribution of LACT images to that of fully sampled target images, enabling the model to learn structural correspondences between them. During inference, the LACT image serves as an explicit prior to guide the sampling trajectory, allowing for high-quality reconstruction with significantly fewer steps. In addition, PWD performs multi-scale feature fusion in the wavelet domain, effectively enhancing the reconstruction of fine details by leveraging both low-frequency and high-frequency information. Quantitative and qualitative evaluations on clinical dental arch CBCT and periapical datasets demonstrate that PWD outperforms existing methods under the same sampling condition. Using only 50 sampling steps, PWD achieves at least 1.7 dB improvement in PSNR and 10% gain in SSIM. 

**Abstract (ZH)**: 生成扩散模型在医学成像中的应用，特别是在有限角度 computed tomography (LACT) 中，prior信息嵌入和小波特征融合快速采样扩散模型用于 LACT 重建 

---
# OASBuilder: Generating OpenAPI Specifications from Online API Documentation with Large Language Models 

**Title (ZH)**: OASBuilder: 从在线API文档生成OpenAPI规范的大语言模型方法 

**Authors**: Koren Lazar, Matan Vetzler, Kiran Kate, Jason Tsay, David Boaz Himanshu Gupta, Avraham Shinnar, Rohith D Vallam, David Amid Esther Goldbraich, Guy Uziel, Jim Laredo, Ateret Anaby Tavor  

**Link**: [PDF](https://arxiv.org/pdf/2507.05316)  

**Abstract**: AI agents and business automation tools interacting with external web services require standardized, machine-readable information about their APIs in the form of API specifications. However, the information about APIs available online is often presented as unstructured, free-form HTML documentation, requiring external users to spend significant time manually converting it into a structured format. To address this, we introduce OASBuilder, a novel framework that transforms long and diverse API documentation pages into consistent, machine-readable API specifications. This is achieved through a carefully crafted pipeline that integrates large language models and rule-based algorithms which are guided by domain knowledge of the structure of documentation webpages. Our experiments demonstrate that OASBuilder generalizes well across hundreds of APIs, and produces valid OpenAPI specifications that encapsulate most of the information from the original documentation. OASBuilder has been successfully implemented in an enterprise environment, saving thousands of hours of manual effort and making hundreds of complex enterprise APIs accessible as tools for LLMs. 

**Abstract (ZH)**: AI代理和企业自动化工具与外部Web服务交互需要以API规范的形式获取标准化的、机器可读的API信息。然而，网络上可用于API的信息通常以未结构化的自由格式HTML文档形式呈现，这要求外部用户花费大量时间手动将其转换为结构化的格式。为了解决这一问题，我们引入了OASBuilder，这是一种新颖的框架，能够将长篇且多样的API文档页面转化为一致的、机器可读的API规范。这一过程通过一个精心设计的流水线实现，该流水线结合了大型语言模型和基于规则的算法，并受到文档网页结构领域知识的指导。我们的实验表明，OASBuilder能够在数百个API上有效推广，并产生有效的OpenAPI规范，这些规范涵盖了原始文档中的大部分信息。OASBuilder已在企业环境中成功实施，节省了数千小时的手动工作，并使数百个复杂的企业API成为大语言模型的工具。 

---
# Solar Flare Prediction Using LSTM and DLSTM with Sliding Window Pattern Recognition 

**Title (ZH)**: 基于滑动窗口模式识别的LSTM和DLSTM太阳耀斑预测 

**Authors**: Zeinab Hassani, Davud Mohammadpur, Hossein Safari  

**Link**: [PDF](https://arxiv.org/pdf/2507.05313)  

**Abstract**: We investigate the use of Long Short-Term Memory (LSTM) and Decomposition-LSTM (DLSTM) networks, combined with an ensemble algorithm, to predict solar flare occurrences using time-series data from the GOES catalog. The dataset spans from 2003 to 2023 and includes 151,071 flare events. Among approximately possible patterns, 7,552 yearly pattern windows are identified, highlighting the challenge of long-term forecasting due to the Sun's complex, self-organized criticality-driven behavior. A sliding window technique is employed to detect temporal quasi-patterns in both irregular and regularized flare time series. Regularization reduces complexity, enhances large flare activity, and captures active days more effectively. To address class imbalance, resampling methods are applied. LSTM and DLSTM models are trained on sequences of peak fluxes and waiting times from irregular time series, while LSTM and DLSTM, integrated with an ensemble approach, are applied to sliding windows of regularized time series with a 3-hour interval. Performance metrics, particularly TSS (0.74), recall (0.95) and the area under the curve (AUC=0.87) in the receiver operating characteristic (ROC), indicate that DLSTM with an ensemble approach on regularized time series outperforms other models, offering more accurate large-flare forecasts with fewer false errors compared to models trained on irregular time series. The superior performance of DLSTM is attributed to its ability to decompose time series into trend and seasonal components, effectively isolating random noise. This study underscores the potential of advanced machine learning techniques for solar flare prediction and highlights the importance of incorporating various solar cycle phases and resampling strategies to enhance forecasting reliability. 

**Abstract (ZH)**: 基于GOES档案时间序列数据的长短期记忆网络与分解长短期记忆网络组合模型在预测太阳耀斑发生中的应用 

---
# PLACE: Prompt Learning for Attributed Community Search 

**Title (ZH)**: PLACE: 带属性社区搜索的提示学习 

**Authors**: Shuheng Fang, Kangfei Zhao, Rener Zhang, Yu Rong, Jeffrey Xu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05311)  

**Abstract**: In this paper, we propose PLACE (Prompt Learning for Attributed Community Search), an innovative graph prompt learning framework for ACS. Enlightened by prompt-tuning in Natural Language Processing (NLP), where learnable prompt tokens are inserted to contextualize NLP queries, PLACE integrates structural and learnable prompt tokens into the graph as a query-dependent refinement mechanism, forming a prompt-augmented graph. Within this prompt-augmented graph structure, the learned prompt tokens serve as a bridge that strengthens connections between graph nodes for the query, enabling the GNN to more effectively identify patterns of structural cohesiveness and attribute similarity related to the specific query. We employ an alternating training paradigm to optimize both the prompt parameters and the GNN jointly. Moreover, we design a divide-and-conquer strategy to enhance scalability, supporting the model to handle million-scale graphs. Extensive experiments on 9 real-world graphs demonstrate the effectiveness of PLACE for three types of ACS queries, where PLACE achieves higher F1 scores by 22% compared to the state-of-the-arts on average. 

**Abstract (ZH)**: PLACE: 有 Attribution 的社区搜索的提示学习框架 

---
# Neural Velocity for hyperparameter tuning 

**Title (ZH)**: 神经速度优化超参数调优 

**Authors**: Gianluca Dalmasso, Andrea Bragagnolo, Enzo Tartaglione, Attilio Fiandrotti, Marco Grangetto  

**Link**: [PDF](https://arxiv.org/pdf/2507.05309)  

**Abstract**: Hyperparameter tuning, such as learning rate decay and defining a stopping criterion, often relies on monitoring the validation loss. This paper presents NeVe, a dynamic training approach that adjusts the learning rate and defines the stop criterion based on the novel notion of "neural velocity". The neural velocity measures the rate of change of each neuron's transfer function and is an indicator of model convergence: sampling neural velocity can be performed even by forwarding noise in the network, reducing the need for a held-out dataset. Our findings show the potential of neural velocity as a key metric for optimizing neural network training efficiently 

**Abstract (ZH)**: 基于“神经速度”的动态训练方法NeVe：一种新的关键指标及其在优化神经网络训练中的潜力 

---
# Enjoying Non-linearity in Multinomial Logistic Bandits 

**Title (ZH)**: 享受多元logistic臂中的非线性特性 

**Authors**: Pierre Boudart, Pierre Gaillard, Alessandro Rudi  

**Link**: [PDF](https://arxiv.org/pdf/2507.05306)  

**Abstract**: We consider the multinomial logistic bandit problem, a variant of generalized linear bandits where a learner interacts with an environment by selecting actions to maximize expected rewards based on probabilistic feedback from multiple possible outcomes. In the binary setting, recent work has focused on understanding the impact of the non-linearity of the logistic model (Faury et al., 2020; Abeille et al., 2021). They introduced a problem-dependent constant $\kappa_*$, that may be exponentially large in some problem parameters and which is captured by the derivative of the sigmoid function. It encapsulates the non-linearity and improves existing regret guarantees over $T$ rounds from $\smash{O(d\sqrt{T})}$ to $\smash{O(d\sqrt{T/\kappa_*})}$, where $d$ is the dimension of the parameter space. We extend their analysis to the multinomial logistic bandit framework, making it suitable for complex applications with more than two choices, such as reinforcement learning or recommender systems. To achieve this, we extend the definition of $\kappa_*$ to the multinomial setting and propose an efficient algorithm that leverages the problem's non-linearity. Our method yields a problem-dependent regret bound of order $ \smash{\widetilde{\mathcal{O}}( Kd \sqrt{{T}/{\kappa_*}})} $, where $K$ is the number of actions and $\kappa_* \ge 1$. This improves upon the best existing guarantees of order $ \smash{\widetilde{\mathcal{O}}( Kd \sqrt{T} )} $. Moreover, we provide a $\smash{ \Omega(d\sqrt{T/\kappa_*})}$ lower-bound, showing that our dependence on $\kappa_*$ is optimal. 

**Abstract (ZH)**: 多分类逻辑宽度胆问题及其分析：从二分类到多分类的拓展 

---
# Integrating Generative AI in BIM Education: Insights from Classroom Implementation 

**Title (ZH)**: 将生成性人工智能融入BIM教育：课堂教学实施的见解 

**Authors**: Islem Sahraoui, Kinam Kim, Lu Gao, Zia Din, Ahmed Senouci  

**Link**: [PDF](https://arxiv.org/pdf/2507.05296)  

**Abstract**: This study evaluates the implementation of a Generative AI-powered rule checking workflow within a graduate-level Building Information Modeling (BIM) course at a U.S. university. Over two semesters, 55 students participated in a classroom-based pilot exploring the use of GenAI for BIM compliance tasks, an area with limited prior research. The instructional design included lectures on prompt engineering and AI-driven rule checking, followed by an assignment where students used a large language model (LLM) to identify code violations in designs using Autodesk Revit. Surveys and interviews were conducted to assess student workload, learning effectiveness, and overall experience, using the NASA-TLX scale and regression analysis. Findings indicate students generally achieved learning objectives but faced challenges such as difficulties debugging AI-generated code and inconsistent tool performance, probably due to their limited prompt engineering experience. These issues increased cognitive and emotional strain, especially among students with minimal programming backgrounds. Despite these challenges, students expressed strong interest in future GenAI applications, particularly with clear instructional support. 

**Abstract (ZH)**: 本研究评估了在美国大学一门研究生级别建筑信息建模（BIM）课程中基于生成式AI的规则检查工作流的实施情况。在两个学期内，55名学生参与了基于教室的试点项目，探索生成式AI在BIM合规任务中的应用，这是该领域尚未有大量研究的区域。教学设计包括关于提示工程和AI驱动规则检查的讲座，随后是让学生使用大型语言模型（LLM）在Autodesk Revit中识别设计中的代码违规问题的作业。通过调查问卷和访谈，使用NASA-TLX量表和回归分析评估了学生的工作量、学习效果和总体体验。研究发现，学生普遍达到了学习目标，但在调试生成式AI代码和工具表现不一致等方面遇到了挑战，可能是因为他们缺乏提示工程经验。这些问题增加了学生的认知和情感压力，尤其是在编程背景较弱的学生中更为明显。尽管存在这些挑战，学生们对未来的生成式AI应用表现出强烈兴趣，特别是在有明确教学支持的情况下。 

---
# Enhancing Learning Path Recommendation via Multi-task Learning 

**Title (ZH)**: 基于多任务学习的学习路径推荐优化 

**Authors**: Afsana Nasrin, Lijun Qian, Pamela Obiomon, Xishuang Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.05295)  

**Abstract**: Personalized learning is a student-centered educational approach that adapts content, pace, and assessment to meet each learner's unique needs. As the key technique to implement the personalized learning, learning path recommendation sequentially recommends personalized learning items such as lectures and exercises. Advances in deep learning, particularly deep reinforcement learning, have made modeling such recommendations more practical and effective. This paper proposes a multi-task LSTM model that enhances learning path recommendation by leveraging shared information across tasks. The approach reframes learning path recommendation as a sequence-to-sequence (Seq2Seq) prediction problem, generating personalized learning paths from a learner's historical interactions. The model uses a shared LSTM layer to capture common features for both learning path recommendation and deep knowledge tracing, along with task-specific LSTM layers for each objective. To avoid redundant recommendations, a non-repeat loss penalizes repeated items within the recommended learning path. Experiments on the ASSIST09 dataset show that the proposed model significantly outperforms baseline methods for the learning path recommendation. 

**Abstract (ZH)**: 个性化学习是一种以学生为中心的教育方法，根据每位学习者的独特需求调整内容、进度和评估。作为实施个性化学习的关键技术，学习路径推荐按顺序推荐个性化学习项目，如讲座和练习。随着深度学习，尤其是深度强化学习的进步，使这种推荐更加实用和有效。本文提出了一种多任务LSTM模型，通过利用任务间的共享信息来增强学习路径推荐。该方法将学习路径推荐重新构 frame 为一个序列到序列（Seq2Seq）预测问题，从学习者的历史互动中生成个性化学习路径。模型使用共享的LSTM层来捕捉学习路径推荐和深度知识追踪的共同特征，并为每个目标使用特定任务的LSTM层。为了避免重复推荐，非重复损失项惩罚推荐学习路径中的重复项目。在ASSIST09数据集上的实验表明，所提出模型在学习路径推荐方面显著优于基线方法。 

---
# Physics-Informed Graph Neural Networks to Reconstruct Local Fields Considering Finite Strain Hyperelasticity 

**Title (ZH)**: 基于物理学的知识图神经网络考虑有限拉伸弹性的局部场重建 

**Authors**: Manuel Ricardo Guevara Garban, Yves Chemisky, Étienne Prulière, Michaël Clément  

**Link**: [PDF](https://arxiv.org/pdf/2507.05291)  

**Abstract**: We propose a physics-informed machine learning framework called P-DivGNN to reconstruct local stress fields at the micro-scale, in the context of multi-scale simulation given a periodic micro-structure mesh and mean, macro-scale, stress values. This method is based in representing a periodic micro-structure as a graph, combined with a message passing graph neural network. We are able to retrieve local stress field distributions, providing average stress values produced by a mean field reduced order model (ROM) or Finite Element (FE) simulation at the macro-scale. The prediction of local stress fields are of utmost importance considering fracture analysis or the definition of local fatigue criteria. Our model incorporates physical constraints during training to constraint local stress field equilibrium state and employs a periodic graph representation to enforce periodic boundary conditions. The benefits of the proposed physics-informed GNN are evaluated considering linear and non linear hyperelastic responses applied to varying geometries. In the non-linear hyperelastic case, the proposed method achieves significant computational speed-ups compared to FE simulation, making it particularly attractive for large-scale applications. 

**Abstract (ZH)**: 基于物理约束的机器学习框架P-DivGNN在周期微观结构网格下重构微观尺度局部应力场的研究 

---
# Compressing Deep Neural Networks Using Explainable AI 

**Title (ZH)**: 使用可解释人工智能压缩深度神经网络 

**Authors**: Kimia Soroush, Mohsen Raji, Behnam Ghavami  

**Link**: [PDF](https://arxiv.org/pdf/2507.05286)  

**Abstract**: Deep neural networks (DNNs) have demonstrated remarkable performance in many tasks but it often comes at a high computational cost and memory usage. Compression techniques, such as pruning and quantization, are applied to reduce the memory footprint of DNNs and make it possible to accommodate them on resource-constrained edge devices. Recently, explainable artificial intelligence (XAI) methods have been introduced with the purpose of understanding and explaining AI methods. XAI can be utilized to get to know the inner functioning of DNNs, such as the importance of different neurons and features in the overall performance of DNNs. In this paper, a novel DNN compression approach using XAI is proposed to efficiently reduce the DNN model size with negligible accuracy loss. In the proposed approach, the importance score of DNN parameters (i.e. weights) are computed using a gradient-based XAI technique called Layer-wise Relevance Propagation (LRP). Then, the scores are used to compress the DNN as follows: 1) the parameters with the negative or zero importance scores are pruned and removed from the model, 2) mixed-precision quantization is applied to quantize the weights with higher/lower score with higher/lower number of bits. The experimental results show that, the proposed compression approach reduces the model size by 64% while the accuracy is improved by 42% compared to the state-of-the-art XAI-based compression method. 

**Abstract (ZH)**: 基于可解释人工智能的高效DNN压缩方法 

---
# Hungary and AI: efforts and opportunities in comparison with Singapore 

**Title (ZH)**: 匈牙利与人工智能：与新加坡比较的举措与机遇 

**Authors**: András Ferenczy  

**Link**: [PDF](https://arxiv.org/pdf/2507.05280)  

**Abstract**: The study assesses Hungary's National AI Strategy and its implementation through the analysis of strategic documents, publicly available financial records, and expert interviews with the Hungarian AI Coalition President and Chief Strategic Advisor to the Government Commissioner for AI. 22 goals from Hungary's strategy were evaluated through conceptual, governance, temporal, and financial dimensions before being benchmarked against Singapore's National AI Strategies (NAIS 1.0 and NAIS 2.0). Key findings include an estimated total of EUR 4.65 billion in AI-related public investment in Hungary. Openly available financial data was found for only half of the evaluated goals, and just three projects made up 98\% of all documented funding. The research also reveals Hungary's implementation challenges, including fragmented execution following ministerial reorganizations and the absence of designated biennial reviews since 2020. Furthermore, the paper provides targeted recommendations for Hungary's forthcoming AI strategy, drawing on Singapore's framework as a reference point. These include adapting to the era of large language models, restructuring the existing triple helix network to foster more effective dialogue and advocacy, and positioning the country as an East-West bridge for automotive AI experimentation. 

**Abstract (ZH)**: 匈牙利国家人工智能战略评估：基于战略文件、公开财务记录及与匈牙利人工智能联盟主席和政府人工智能专员首席战略顾问的专家访谈的研究 

---
# A Fuzzy Supervisor Agent Design for Clinical Reasoning Assistance in a Multi-Agent Educational Clinical Scenario Simulation 

**Title (ZH)**: 面向多代理教育临床情景模拟的模糊监督代理设计用于临床推理辅助 

**Authors**: Weibing Zheng, Laurah Turner, Jess Kropczynski, Murat Ozer, Seth Overla, Shane Halse  

**Link**: [PDF](https://arxiv.org/pdf/2507.05275)  

**Abstract**: Assisting medical students with clinical reasoning (CR) during clinical scenario training remains a persistent challenge in medical education. This paper presents the design and architecture of the Fuzzy Supervisor Agent (FSA), a novel component for the Multi-Agent Educational Clinical Scenario Simulation (MAECSS) platform. The FSA leverages a Fuzzy Inference System (FIS) to continuously interpret student interactions with specialized clinical agents (e.g., patient, physical exam, diagnostic, intervention) using pre-defined fuzzy rule bases for professionalism, medical relevance, ethical behavior, and contextual distraction. By analyzing student decision-making processes in real-time, the FSA is designed to deliver adaptive, context-aware feedback and provides assistance precisely when students encounter difficulties. This work focuses on the technical framework and rationale of the FSA, highlighting its potential to provide scalable, flexible, and human-like supervision in simulation-based medical education. Future work will include empirical evaluation and integration into broader educational settings. More detailed design and implementation is~\href{this https URL}{open sourced here}. 

**Abstract (ZH)**: 辅助医学学生在临床情景训练中进行临床推理（CR）仍然是医学教育中的一个持续挑战。本文介绍了Fuzzy Supervisor Agent（FSA）的设计与架构，该架构是Multi-Agent Educational Clinical Scenario Simulation（MAECSS）平台的一个新型组件。FSA利用Fuzzy Inference System（FIS）持续解释学生与专业临床代理（如患者、体格检查、诊断、干预）的互动，并使用预先定义的模糊规则库来评估专业性、医学相关性、伦理行为和情境干扰。通过实时分析学生的决策过程，FSA旨在提供适应性、情境相关的反馈，并在学生遇到困难时提供精准的帮助。本文着重于FSA的技术框架和原理，强调其在基于模拟的医学教育中提供可扩展、灵活且人性化监督的潜力。未来的研究将包括实证评估和将其整合到更广泛教育环境中。更多详细设计和实现可在[此处](this https URL)获得。 

---
# Rethinking Over-Smoothing in Graph Neural Networks: A Perspective from Anderson Localization 

**Title (ZH)**: 从冯·诺依曼局域化视角重思图神经网络中的过度平滑问题 

**Authors**: Kaichen Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05263)  

**Abstract**: Graph Neural Networks (GNNs) have shown great potential in graph data analysis due to their powerful representation capabilities. However, as the network depth increases, the issue of over-smoothing becomes more severe, causing node representations to lose their distinctiveness. This paper analyzes the mechanism of over-smoothing through the analogy to Anderson localization and introduces participation degree as a metric to quantify this phenomenon. Specifically, as the depth of the GNN increases, node features homogenize after multiple layers of message passing, leading to a loss of distinctiveness, similar to the behavior of vibration modes in disordered systems. In this context, over-smoothing in GNNs can be understood as the expansion of low-frequency modes (increased participation degree) and the localization of high-frequency modes (decreased participation degree). Based on this, we systematically reviewed the potential connection between the Anderson localization behavior in disordered systems and the over-smoothing behavior in Graph Neural Networks. A theoretical analysis was conducted, and we proposed the potential of alleviating over-smoothing by reducing the disorder in information propagation. 

**Abstract (ZH)**: 图神经网络（GNNs）由于其强大的表示能力，在图数据分析中展现出了巨大潜力。然而，随着网络深度的增加，过拟合问题变得更加严重，导致节点表示失真。本文通过将过拟合与安德森局域化进行类比，引入参与度作为度量这一现象的指标。具体而言，随着GNN层数的增加，经过多层消息传递后节点特征同质化，导致失真现象，类似于无序系统中振动模式的行为。在此背景下，GNN中的过拟合可以理解为低频模式的扩展（参与度增加）和高频模式的局域化（参与度减少）。基于此，我们系统地探讨了无序系统中的安德森局域化行为与图神经网络中过拟合行为之间的潜在联系，并进行了理论分析，提出了通过减少信息传播中的无序来缓解过拟合的潜在途径。 

---
# ABench-Physics: Benchmarking Physical Reasoning in LLMs via High-Difficulty and Dynamic Physics Problems 

**Title (ZH)**: ABench-Physics: 通过高难度和动态物理问题评估LLM中的物理推理能力 

**Authors**: Yiming Zhang, Yingfan Ma, Yanmei Gu, Zhengkai Yang, Yihong Zhuang, Feng Wang, Zenan Huang, Yuanyuan Wang, Chao Huang, Bowen Song, Cheng Lin, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.04766)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance in domains such as mathematics and programming, yet their capabilities in physics remain underexplored and poorly understood. Physics poses unique challenges that demand not only precise computation but also deep conceptual understanding and physical modeling skills. Existing benchmarks often fall short due to limited difficulty, multiple-choice formats, and static evaluation settings that fail to capture physical modeling ability. In this paper, we introduce ABench-Physics, a novel benchmark designed to rigorously evaluate LLMs' physical reasoning and generalization capabilities. ABench-Physics consists of two components: Phy_A, a static set of 400 graduate- or Olympiad-level problems; and Phy_B, a dynamic subset of 100 problems equipped with an automatic variation engine to test model robustness across changing conditions. All questions require precise numerical answers, with strict formatting and tolerance constraints. Our evaluation of several state-of-the-art LLMs reveals substantial performance gaps, highlighting persistent limitations in physical reasoning, especially in generalization to dynamic variants. ABench-Physics provides a challenging and diagnostic framework for advancing scientific reasoning in LLMs. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在数学和编程领域展示了 impressive 的性能，但在物理学领域的能力和潜力仍被严重低估且不甚了解。物理学提出了独特的挑战，不仅要求精确的计算，还需要深厚的conceptual理解以及物理建模能力。现有基准往往由于难度有限、选择题格式以及静态评估设置等问题而显得不足，无法充分捕捉物理建模能力。在本文中，我们引入了 ABench-Physics，这一新型基准旨在严格评估LLMs的物理推理和泛化能力。ABench-Physics 包括两个部分：Phy_A，包含400道研究生水平或奥林匹克级别问题的静态问题集；Phy_B，包含100道动态问题的子集，并配备自动变体引擎以测试模型在不同条件下的鲁棒性。所有问题要求精确的数字答案，并有严格的格式和容差约束。我们的评估结果显示了显著的性能差距，突显了物理推理的持续局限性，尤其是在动态变体的泛化能力方面。ABench-Physics 提供了一个具有挑战性和诊断性的框架，以促进LLMs中的科学推理能力。 

---
# The Problem of Algorithmic Collisions: Mitigating Unforeseen Risks in a Connected World 

**Title (ZH)**: 算法碰撞的问题： mitigation of unforeseen risks in a connected world 

**Authors**: Maurice Chiodo, Dennis Müller  

**Link**: [PDF](https://arxiv.org/pdf/2505.20181)  

**Abstract**: The increasing deployment of Artificial Intelligence (AI) and other autonomous algorithmic systems presents the world with new systemic risks. While focus often lies on the function of individual algorithms, a critical and underestimated danger arises from their interactions, particularly when algorithmic systems operate without awareness of each other, or when those deploying them are unaware of the full algorithmic ecosystem deployment is occurring in. These interactions can lead to unforeseen, rapidly escalating negative outcomes - from market crashes and energy supply disruptions to potential physical accidents and erosion of public trust - often exceeding the human capacity for effective monitoring and the legal capacities for proper intervention. Current governance frameworks are inadequate as they lack visibility into this complex ecosystem of interactions. This paper outlines the nature of this challenge and proposes some initial policy suggestions centered on increasing transparency and accountability through phased system registration, a licensing framework for deployment, and enhanced monitoring capabilities. 

**Abstract (ZH)**: 不断增加的人工智能及其他自主算法系统的部署为世界带来了新的系统性风险。 

---
# Formalising Human-in-the-Loop: Computational Reductions, Failure Modes, and Legal-Moral Responsibility 

**Title (ZH)**: 形式化人类在环中参与：计算减少、失败模式和法律-道德责任 

**Authors**: Maurice Chiodo, Dennis Müller, Paul Siewert, Jean-Luc Wetherall, Zoya Yasmine, John Burden  

**Link**: [PDF](https://arxiv.org/pdf/2505.10426)  

**Abstract**: The legal compliance and safety of different Human-in-the-loop (HITL) setups for AI can vary greatly. This manuscript aims to identify new ways of choosing between such setups, and shows that there is an unavoidable trade-off between the attribution of legal responsibility and the technical explainability of AI. We begin by using the notion of oracle machines from computability theory to formalise different HITL setups, distinguishing between trivial human monitoring, single endpoint human action, and highly involved interaction between the human(s) and the AI. These correspond to total functions, many-one reductions, and Turing reductions respectively. A taxonomy categorising HITL failure modes is then presented, highlighting the limitations on what any HITL setup can actually achieve. Our approach then identifies oversights from UK and EU legal frameworks, which focus on certain HITL setups which may not always achieve the desired ethical, legal, and sociotechnical outcomes. We suggest areas where the law should recognise the effectiveness of different HITL setups and assign responsibility in these contexts, avoiding unnecessary and unproductive human "scapegoating". Overall, we show how HITL setups involve many technical design decisions, and can be prone to failures which are often out of the humans' control. This opens up a new analytic perspective on the challenges arising in the creation of HITL setups, helping inform AI developers and lawmakers on designing HITL to better achieve their desired outcomes. 

**Abstract (ZH)**: 不同的人工智能半自动回环（HITL）设置的法律合规性和安全性可能存在巨大差异。本论文旨在探索选择不同HITL设置的新方法，并表明在法律归责与人工智能技术可解释性之间存在不可避免的权衡。我们首先利用可计算理论中的 oracle 机器概念来形式化不同的 HITL 设置，分别区分简易的人工监控、单一终端的人工操作以及人类与人工智能高度互动的情况。这些分别对应完全函数、many-one 归约和图灵归约。然后，我们提出了一个分类法来归纳 HITL 失败模式，强调任何 HITL 设置实际上能够实现的局限性。接下来，我们指出了来自英国和欧盟法律框架中的盲点，这些框架往往侧重于某些 HITL 设置，这些设置可能不能总是实现所需的伦理、法律和社会技术结果。我们建议法律在不同 HITL 设置的有效性方面应予以认可，并在这类情境中分配责任，避免不必要的和没有成效的人类“替罪羊”现象。总体而言，本论文展示了如何 HITL 设置涉及众多技术设计决策，并可能因超出人类控制范畴的失败而受到威胁。这为理解 HITL 设置带来的挑战提供了新的分析视角，有助于指导人工智能开发者和立法者更好地设计 HITL，以实现其预期目标。 

---
# Integrators at War: Mediating in AI-assisted Resort-to-Force Decisions 

**Title (ZH)**: 人工智能辅助下的使用武力决策中的调解者：矛盾与调和 

**Authors**: Dennis Müller, Maurice Chiodo, Mitja Sienknecht  

**Link**: [PDF](https://arxiv.org/pdf/2501.06861)  

**Abstract**: The integration of AI systems into the military domain is changing the way war-related decisions are made. It binds together three disparate groups of actors - developers, integrators, users - and creates a relationship between these groups and the machine, embedded in the (pre-)existing organisational and system structures. In this article, we focus on the important, but often neglected, group of integrators within such a sociotechnical system. In complex human-machine configurations, integrators carry responsibility for linking the disparate groups of developers and users in the political and military system. To act as the mediating group requires a deep understanding of the other groups' activities, perspectives and norms. We thus ask which challenges and shortcomings emerge from integrating AI systems into resort-to-force (RTF) decision-making processes, and how to address them. To answer this, we proceed in three steps. First, we conceptualise the relationship between different groups of actors and AI systems as a sociotechnical system. Second, we identify challenges within such systems for human-machine teaming in RTF decisions. We focus on challenges that arise a) from the technology itself, b) from the integrators' role in the sociotechnical system, c) from the human-machine interaction. Third, we provide policy recommendations to address these shortcomings when integrating AI systems into RTF decision-making structures. 

**Abstract (ZH)**: 人工智能系统在军事领域的整合正改变战争相关决策的方式。这种整合将开发者、集成者和用户这三个不同时的行动者群体结合在一起，嵌入到（预）现有的组织和系统结构中，形成其间的关係。本文旨在关注这样的社会技术系统中经常被忽略的集成者群体。在复杂的人机配置中，集成者承担着连接开发者和用户的政治和军事系统的责任。为了充当中介群体，需要深刻理解其他群体的活动、视角和规范。因此，本文探讨将人工智能系统整合到使用武力决策过程中的挑战和不足，并提出相应的解决建议。首先，本文将不同群体的行动者与人工智能系统之间的关系概念化为社会技术系统。第二，本文识别此类系统中的人机团队在使用武力决策中的挑战。我们重点关注来自技术本身、集成者在社会技术系统中的角色以及人机互动的挑战。第三，本文提供政策建议，以解决在将人工智能系统整合到使用武力决策结构中时的不足。 

---
