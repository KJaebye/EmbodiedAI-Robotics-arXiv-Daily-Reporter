# A Supervised Autonomous Resection and Retraction Framework for Transurethral Enucleation of the Prostatic Median Lobe 

**Title (ZH)**: 经尿道前列腺中叶enucleation的监督自主切除和复位框架 

**Authors**: Mariana Smith, Tanner Watts, Susheela Sharma Stern, Brendan Burkhart, Hao Li, Alejandro O. Chara, Nithesh Kumar, James Ferguson, Ayberk Acar, Jesse F. d'Almeida, Lauren Branscombe, Lauren Shepard, Ahmed Ghazi, Ipek Oguz, Jie Ying Wu, Robert J. Webster III, Axel Krieger, Alan Kuntz  

**Link**: [PDF](https://arxiv.org/pdf/2511.08490)  

**Abstract**: Concentric tube robots (CTRs) offer dexterous motion at millimeter scales, enabling minimally invasive procedures through natural orifices. This work presents a coordinated model-based resection planner and learning-based retraction network that work together to enable semi-autonomous tissue resection using a dual-arm transurethral concentric tube robot (the Virtuoso). The resection planner operates directly on segmented CT volumes of prostate phantoms, automatically generating tool trajectories for a three-phase median lobe resection workflow: left/median trough resection, right/median trough resection, and median blunt dissection. The retraction network, PushCVAE, trained on surgeon demonstrations, generates retractions according to the procedural phase. The procedure is executed under Level-3 (supervised) autonomy on a prostate phantom composed of hydrogel materials that replicate the mechanical and cutting properties of tissue. As a feasibility study, we demonstrate that our combined autonomous system achieves a 97.1% resection of the targeted volume of the median lobe. Our study establishes a foundation for image-guided autonomy in transurethral robotic surgery and represents a first step toward fully automated minimally-invasive prostate enucleation. 

**Abstract (ZH)**: 同心管机器人（CTRs）在毫米级尺度上提供灵巧运动，通过自然开口实现微创手术。本研究提出了一种协调的模型驱动切除计划器和基于学习的牵拉网络，共同实现双臂经尿道同心管机器人（Virtuoso）的半自主组织切除。切除计划器直接在前列腺假体的分割CT体积上运行，自动生成三阶段中叶切除工作流程中的工具轨迹：左侧/中叶沟切除、右侧/中叶沟切除和中叶钝性分离。牵拉网络PushCVAE根据手术阶段进行训练，生成相应的牵拉动作。该手术在由模拟组织机械和切割特性的水凝胶材料组成的前列腺假体上以监督级（Level-3）自主性执行。作为可行性研究，我们展示了我们综合的自主系统实现了97.1%的目标中叶体积切除。本研究为经尿道机器人手术中的图像引导自主性奠定了基础，并代表了实现完全自动化微创前列腺剜除的第一步。 

---
# Model Predictive Control via Probabilistic Inference: A Tutorial 

**Title (ZH)**: 概率推断视角下的模型预测控制：一个教程 

**Authors**: Kohei Honda  

**Link**: [PDF](https://arxiv.org/pdf/2511.08019)  

**Abstract**: Model Predictive Control (MPC) is a fundamental framework for optimizing robot behavior over a finite future horizon. While conventional numerical optimization methods can efficiently handle simple dynamics and cost structures, they often become intractable for the nonlinear or non-differentiable systems commonly encountered in robotics. This article provides a tutorial on probabilistic inference-based MPC, presenting a unified theoretical foundation and a comprehensive overview of representative methods. Probabilistic inference-based MPC approaches, such as Model Predictive Path Integral (MPPI) control, have gained significant attention by reinterpreting optimal control as a problem of probabilistic inference. Rather than relying on gradient-based numerical optimization, these methods estimate optimal control distributions through sampling-based techniques, accommodating arbitrary cost functions and dynamics. We first derive the optimal control distribution from the standard optimal control problem, elucidating its probabilistic interpretation and key characteristics. The widely used MPPI algorithm is then derived as a practical example, followed by discussions on prior and variational distribution design, tuning principles, and theoretical aspects. This article aims to serve as a systematic guide for researchers and practitioners seeking to understand, implement, and extend these methods in robotics and beyond. 

**Abstract (ZH)**: 基于概率推断的模型预测控制：一种统一的理论基础和综述 

---
# High-Altitude Balloon Station-Keeping with First Order Model Predictive Control 

**Title (ZH)**: 高 altitude 气球站 Keeping  with 一阶模型预测控制 

**Authors**: Myles Pasetsky, Jiawei Lin, Bradley Guo, Sarah Dean  

**Link**: [PDF](https://arxiv.org/pdf/2511.07761)  

**Abstract**: High-altitude balloons (HABs) are common in scientific research due to their wide range of applications and low cost. Because of their nonlinear, underactuated dynamics and the partial observability of wind fields, prior work has largely relied on model-free reinforcement learning (RL) methods to design near-optimal control schemes for station-keeping. These methods often compare only against hand-crafted heuristics, dismissing model-based approaches as impractical given the system complexity and uncertain wind forecasts. We revisit this assumption about the efficacy of model-based control for station-keeping by developing First-Order Model Predictive Control (FOMPC). By implementing the wind and balloon dynamics as differentiable functions in JAX, we enable gradient-based trajectory optimization for online planning. FOMPC outperforms a state-of-the-art RL policy, achieving a 24% improvement in time-within-radius (TWR) without requiring offline training, though at the cost of greater online computation per control step. Through systematic ablations of modeling assumptions and control factors, we show that online planning is effective across many configurations, including under simplified wind and dynamics models. 

**Abstract (ZH)**: 高海拔气球通过其广泛的应用范围和低成本，在科学研究中十分常见。由于其非线性、欠驱动的动力学特性和风场的部分可观测性，先前的工作主要依赖无模型强化学习（RL）方法设计近最优的驻留控制方案。这些方法通常仅与手工设计的启发式方法进行比较，认为在给定系统复杂性和不确定的风速预报的情况下，基于模型的控制方法不切实际。我们通过开发一阶模型预测控制（FOMPC）重新审视了基于模型的控制方法在驻留控制中的有效性。通过在JAX中将风和气球动力学实现为可微函数，我们实现基于梯度的轨迹优化以进行在线规划。FOMPC在不需要离线训练的情况下，相较于最先进的RL策略，在时间在指定范围内的性能上取得了24%的提升，但每个控制步骤所需的在线计算量有所增加。通过对建模假设和控制因素的系统性消融研究，我们展示了在线规划在多种配置下均有效，包括在简化后的风速和动力学模型下。 

---
# Probabilistic Safety Guarantee for Stochastic Control Systems Using Average Reward MDPs 

**Title (ZH)**: 使用平均奖励MDP的随机控制系统概率安全保证 

**Authors**: Saber Omidi, Marek Petrik, Se Young Yoon, Momotaz Begum  

**Link**: [PDF](https://arxiv.org/pdf/2511.08419)  

**Abstract**: Safety in stochastic control systems, which are subject to random noise with a known probability distribution, aims to compute policies that satisfy predefined operational constraints with high confidence throughout the uncertain evolution of the state variables. The unpredictable evolution of state variables poses a significant challenge for meeting predefined constraints using various control methods. To address this, we present a new algorithm that computes safe policies to determine the safety level across a finite state set. This algorithm reduces the safety objective to the standard average reward Markov Decision Process (MDP) objective. This reduction enables us to use standard techniques, such as linear programs, to compute and analyze safe policies. We validate the proposed method numerically on the Double Integrator and the Inverted Pendulum systems. Results indicate that the average-reward MDPs solution is more comprehensive, converges faster, and offers higher quality compared to the minimum discounted-reward solution. 

**Abstract (ZH)**: 具有已知概率分布的随机噪声影响下的随机控制系统的安全性旨在计算能够在不确定性状态下满足预定义操作约束的策略。状态变量的不可预测演化对使用各种控制方法满足预定义约束构成重大挑战。为此，我们提出了一种新算法，计算安全策略以确定有限状态集的安全级别。该算法将安全性目标转化为标准的平均奖励马尔可夫决策过程（MDP）目标。这种转换使我们能够利用线性规划等标准技术来计算和分析安全策略。我们在双积分器和倒立摆系统上通过数值验证了所提出的方法。结果表明，平均奖励MDP的解决方案更为全面、收敛更快且质量更高，优于最小折现奖励解决方案。 

---
# Statistically Assuring Safety of Control Systems using Ensembles of Safety Filters and Conformal Prediction 

**Title (ZH)**: 使用安全滤波器ensemble和一致性预测保证控制系统的安全性 

**Authors**: Ihab Tabbara, Yuxuan Yang, Hussein Sibai  

**Link**: [PDF](https://arxiv.org/pdf/2511.07899)  

**Abstract**: Safety assurance is a fundamental requirement for deploying learning-enabled autonomous systems. Hamilton-Jacobi (HJ) reachability analysis is a fundamental method for formally verifying safety and generating safe controllers. However, computing the HJ value function that characterizes the backward reachable set (BRS) of a set of user-defined failure states is computationally expensive, especially for high-dimensional systems, motivating the use of reinforcement learning approaches to approximate the value function. Unfortunately, a learned value function and its corresponding safe policy are not guaranteed to be correct. The learned value function evaluated at a given state may not be equal to the actual safety return achieved by following the learned safe policy. To address this challenge, we introduce a conformal prediction-based (CP) framework that bounds such uncertainty. We leverage CP to provide probabilistic safety guarantees when using learned HJ value functions and policies to prevent control systems from reaching failure states. Specifically, we use CP to calibrate the switching between the unsafe nominal controller and the learned HJ-based safe policy and to derive safety guarantees under this switched policy. We also investigate using an ensemble of independently trained HJ value functions as a safety filter and compare this ensemble approach to using individual value functions alone. 

**Abstract (ZH)**: 基于一致预测的不确定边界化方法用于学习引导的哈密尔顿-雅可比可达性分析中的安全保证 

---
# DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs 

**Title (ZH)**: DeepProofLog: 在深 stochastic 逻辑程序中高效证明 

**Authors**: Ying Jiao, Rodrigo Castellano Ontiveros, Luc De Raedt, Marco Gori, Francesco Giannini, Michelangelo Diligenti, Giuseppe Marra  

**Link**: [PDF](https://arxiv.org/pdf/2511.08581)  

**Abstract**: Neurosymbolic (NeSy) AI aims to combine the strengths of neural architectures and symbolic reasoning to improve the accuracy, interpretability, and generalization capability of AI models. While logic inference on top of subsymbolic modules has been shown to effectively guarantee these properties, this often comes at the cost of reduced scalability, which can severely limit the usability of NeSy models. This paper introduces DeepProofLog (DPrL), a novel NeSy system based on stochastic logic programs, which addresses the scalability limitations of previous methods. DPrL parameterizes all derivation steps with neural networks, allowing efficient neural guidance over the proving system. Additionally, we establish a formal mapping between the resolution process of our deep stochastic logic programs and Markov Decision Processes, enabling the application of dynamic programming and reinforcement learning techniques for efficient inference and learning. This theoretical connection improves scalability for complex proof spaces and large knowledge bases. Our experiments on standard NeSy benchmarks and knowledge graph reasoning tasks demonstrate that DPrL outperforms existing state-of-the-art NeSy systems, advancing scalability to larger and more complex settings than previously possible. 

**Abstract (ZH)**: 神经符号（NeSy）AI旨在结合神经架构和符号推理的优点，以提高AI模型的准确性、可解释性和泛化能力。虽然在非符号模块之上进行逻辑推理已被证明能够有效地保证这些特性，但这通常会牺牲可扩展性，从而严重限制了NeSy模型的实用性。本文引入了基于随机逻辑程序的DeepProofLog (DPrL)，这是一种新颖的NeSy系统，解决了一种前一种方法的可扩展性限制。DPrL 使用神经网络参数化所有演绎步骤，允许高效地对证明系统提供神经指导。此外，我们将我们的深度随机逻辑程序的消解过程与马尔可夫决策过程建立了形式映射，使动态规划和强化学习技术能够用于高效的推理和学习。这一理论联系提高了复杂证明空间和大型知识库的可扩展性。我们在标准的NeSy基准测试和知识图推理任务上的实验表明，DPrL 在可扩展性方面超越了现有的最先进的NeSy系统，并使其能够应用于比以往更大的、更复杂的设置。 

---
# Hyperdimensional Decoding of Spiking Neural Networks 

**Title (ZH)**: 高维解码突触神经网络 

**Authors**: Cedrick Kinavuidi, Luca Peres, Oliver Rhodes  

**Link**: [PDF](https://arxiv.org/pdf/2511.08558)  

**Abstract**: This work presents a novel spiking neural network (SNN) decoding method, combining SNNs with Hyperdimensional computing (HDC). The goal is to create a decoding method with high accuracy, high noise robustness, low latency and low energy usage. Compared to analogous architectures decoded with existing approaches, the presented SNN-HDC model attains generally better classification accuracy, lower classification latency and lower estimated energy consumption on multiple test cases from literature. The SNN-HDC achieved estimated energy consumption reductions ranging from 1.24x to 3.67x on the DvsGesture dataset and from 1.38x to 2.27x on the SL-Animals-DVS dataset. The presented decoding method can also efficiently identify unknown classes it has not been trained on. In the DvsGesture dataset the SNN-HDC model can identify 100% of samples from an unseen/untrained class. Given the numerous benefits shown and discussed in this paper, this decoding method represents a very compelling alternative to both rate and latency decoding. 

**Abstract (ZH)**: 一种结合脉冲神经网络与超维计算的新型解码方法：高精度、高抗噪性、低延迟和低能耗的SNN-HDC模型 

---
# Dataset Safety in Autonomous Driving: Requirements, Risks, and Assurance 

**Title (ZH)**: 自动驾驶数据集安全：要求、风险与保障 

**Authors**: Alireza Abbaspour, Tejaskumar Balgonda Patil, B Ravi Kiran, Russel Mohr, Senthil Yogamani  

**Link**: [PDF](https://arxiv.org/pdf/2511.08439)  

**Abstract**: Dataset integrity is fundamental to the safety and reliability of AI systems, especially in autonomous driving. This paper presents a structured framework for developing safe datasets aligned with ISO/PAS 8800 guidelines. Using AI-based perception systems as the primary use case, it introduces the AI Data Flywheel and the dataset lifecycle, covering data collection, annotation, curation, and maintenance. The framework incorporates rigorous safety analyses to identify hazards and mitigate risks caused by dataset insufficiencies. It also defines processes for establishing dataset safety requirements and proposes verification and validation strategies to ensure compliance with safety standards. In addition to outlining best practices, the paper reviews recent research and emerging trends in dataset safety and autonomous vehicle development, providing insights into current challenges and future directions. By integrating these perspectives, the paper aims to advance robust, safety-assured AI systems for autonomous driving applications. 

**Abstract (ZH)**: 基于ISO/PAS 8800指南的AI安全数据集开发结构化框架 

---
# AI-Powered Data Visualization Platform: An Intelligent Web Application for Automated Dataset Analysis 

**Title (ZH)**: 基于AI的数据可视化平台：一个自动化数据集分析的智能网络应用 

**Authors**: Srihari R, Pallavi M, Tejaswini S, Vaishnavi R C  

**Link**: [PDF](https://arxiv.org/pdf/2511.08363)  

**Abstract**: An AI-powered data visualization platform that automates the entire data analysis process, from uploading a dataset to generating an interactive visualization. Advanced machine learning algorithms are employed to clean and preprocess the data, analyse its features, and automatically select appropriate visualizations. The system establishes the process of automating AI-based analysis and visualization from the context of data-driven environments, and eliminates the challenge of time-consuming manual data analysis. The combination of a Python Flask backend to access the dataset, paired with a React frontend, provides a robust platform that automatically interacts with Firebase Cloud Storage for numerous data processing and data analysis solutions and real-time sources. Key contributions include automatic and intelligent data cleaning, with imputation for missing values, and detection of outliers, via analysis of the data set. AI solutions to intelligently select features, using four different algorithms, and intelligent title generation and visualization are determined by the attributes of the dataset. These contributions were evaluated using two separate datasets to assess the platform's performance. In the process evaluation, the initial analysis was performed in real-time on datasets as large as 100000 rows, while the cloud-based demand platform scales to meet requests from multiple users and processes them simultaneously. In conclusion, the cloud-based data visualization application allowed for a significant reduction of manual inputs to the data analysis process while maintaining a high quality, impactful visual outputs, and user experiences 

**Abstract (ZH)**: 一种基于AI的数据可视化平台，自动完成从数据集上传到生成交互式可视化图形的整个数据分析过程。采用先进的机器学习算法对数据进行清洗和预处理，分析其特征，并自动选择合适的可视化方式。该系统从数据驱动环境的角度建立起基于AI的分析与可视化自动化流程，消除了耗时的手动数据分析的挑战。结合Python Flask后端访问数据集与React前端，提供了一个强大的平台，能够自动与Firebase云存储进行交互，实现多种数据处理和数据分析解决方案以及实时数据源的接入。关键贡献包括自动和智能的数据清洗，通过数据集分析进行缺失值填充和离群值检测，智能特征选择使用四种不同算法，以及根据数据集属性进行智能标题生成和可视化。这些贡献通过两个独立的数据集进行了评估，以检验平台性能。在过程评估中，对最多100000行的数据集进行了实时初步分析，而基于云的请求平台能够满足多个用户的请求并同时处理它们。总之，基于云的数据可视化应用在保持高质量、有影响力的视觉输出和用户体验的同时，显著减少了手动输入的数据分析过程。 

---
# JobSphere: An AI-Powered Multilingual Career Copilot for Government Employment Platforms 

**Title (ZH)**: JobSphere：一个依托人工智能技术的多语言职业生涯伴侣平台（政府就业平台应用） 

**Authors**: Srihari R, Adarsha B V, Mohammed Usman Hussain, Shweta Singh  

**Link**: [PDF](https://arxiv.org/pdf/2511.08343)  

**Abstract**: Users of government employment websites commonly face engagement and accessibility challenges linked to navigational complexity, a dearth of language options, and a lack of personalized support. This paper introduces JobSphere, an AI-powered career assistant that is redefining the employment platform in Punjab called PGRKAM. JobSphere employs Retrieval-Augmented Generation (RAG) architecture, and it is multilingual, available in English, Hindi and Punjabi. JobSphere technique uses 4-bit quantization, allowing the platform to deploy on consumer-grade GPUs (i.e., NVIDIA RTX 3050 4GB), making the implementation 89% cheaper than that of cloud-based systems. Key innovations include voice-enabled interaction with the assistant, automated mock tests, resume parsing with skills recognition, and embed-based job recommendation that achieves a precision@10 score of 68%. An evaluation of JobSphere's implementation reveals 94% factual accuracy, a median response time of 1.8 seconds, and a System Usability Scale score of 78.5/100, a 50% improvement compared to the baseline PGRKAM platform context. In conclusion, JobSphere effectively fills significant accessibility gaps for Punjab/Hindi-speaking users in rural locations, while also affirming the users access to trusted job content provided by government agencies. 

**Abstract (ZH)**: 政府就业网站用户常面临与导航复杂性、语言选项匮乏和个人化支持缺乏相关的参与和使用障碍。本文介绍了一种名为JobSphere的AI职业助手，它正在重新定义巴基斯坦旁遮普省PGRKAM就业平台。JobSphere采用检索增强生成（RAG）架构，并提供英语、印地语和旁遮普语多语言支持。JobSphere技术采用4位量化，使平台能够在消费级GPU（如NVIDIA RTX 3050 4GB）上部署，实施成本比基于云的系统低89%。关键创新包括语音交互、自动模拟测试、技能识别的简历解析以及基于嵌入的职位推荐，实现精度@10得分为68%。JobSphere实施评估结果显示，事实准确性为94%，中位响应时间为1.8秒，系统可用性量表得分为78.5/100，比基线PGRKAM平台提高了50%。总之，JobSphere有效地填补了旁遮普地区/印地语使用者在农村地区的显著使用障碍，同时也确保了用户能够访问政府机构提供的可信工作内容。 

---
# Smarter Together: Creating Agentic Communities of Practice through Shared Experiential Learning 

**Title (ZH)**: smarter 一起：通过共享体验学习创建有影响力的实践社区 

**Authors**: Valentin Tablan, Scott Taylor, Gabriel Hurtado, Kristoffer Bernhem, Anders Uhrenholt, Gabriele Farei, Karo Moilanen  

**Link**: [PDF](https://arxiv.org/pdf/2511.08301)  

**Abstract**: The transition from human-centric to agent-centric software development practices is disrupting existing knowledge sharing environments for software developers. Traditional peer-to-peer repositories and developer communities for shared technical knowledge and best practice have witnessed dramatic drops in participation in a short period of time. At the same time, agentic functional equivalents are yet to emerge leaving AI agents, which already generate a significant proportion of all new software code produced, without access to repositories of valuable shared learning.
In this paper, we introduce Spark, a novel shared agentic memory architecture which is designed to emulate the collective intelligence and know-how of human developer communities. Spark enables AI coding agents to both contribute to and draw from a persistent and continuously evolving experiential memory. Agents operating in the same general problem space use the Spark shared memory as a repository of new knowledge to achieve collective continual learning. We evaluate Spark as a coach for AI coding agents performing software development tasks. We demonstrate that recommendations made by Spark improve the quality of code generated by generic code generation models at varying sizes and capability tiers. Boosted by Spark, a small open-weights model with 30 billion parameters was able to match the code quality afforded by a much larger state-of-the-art model. Separately, we measure the intrinsic quality of recommendations generated by Spark against a wide range of criteria inspired by software development best practice, and achieve helpfulness levels of up to 98.2% in the top two (out of five) qualitative helpfulness bands. 

**Abstract (ZH)**: 从人类中心到代理中心的软件开发实践过渡正在颠覆现有的知识共享环境。传统的点对点repositories和开发者社区在短短时间内见证了参与度的显著下降。与此同时，相应的功能替代品尚未出现，导致已经生成了大量新软件代码的AI代理无法访问有价值的共享学习仓库。

在本文中，我们引入了Spark，这是一种新颖的共享代理记忆架构，旨在模拟人类开发者社区的集体智慧和经验技巧。Spark使AI编码代理能够贡献并从中汲取持久且不断演化的经验记忆。在同一问题空间内运作的代理使用Spark共享记忆作为新知识的仓库，以实现集体持续学习。我们评估了Spark作为AI编码代理的教练进行软件开发任务的有效性。我们证明，Spark提供的建议能够提高不同规模和能力级别的通用代码生成模型生成的代码质量。得益于Spark，一个拥有300亿参数的小型开源权重模型能够达到与更大规模的先进模型相当的代码质量。另外，我们测量了Spark生成的建议的内在质量，并基于软件开发最佳实践的一系列标准，实现了高达98.2%的帮助水平，特别是在五个定性帮助水平中的前两个。 

---
# DiagramIR: An Automatic Pipeline for Educational Math Diagram Evaluation 

**Title (ZH)**: DiagramIR：一种自动教育数学图表评估流水线 

**Authors**: Vishal Kumar, Shubhra Mishra, Rebecca Hao, Rizwaan Malik, David Broman, Dorottya Demszky  

**Link**: [PDF](https://arxiv.org/pdf/2511.08283)  

**Abstract**: Large Language Models (LLMs) are increasingly being adopted as tools for learning; however, most tools remain text-only, limiting their usefulness for domains where visualizations are essential, such as mathematics. Recent work shows that LLMs are capable of generating code that compiles to educational figures, but a major bottleneck remains: scalable evaluation of these diagrams. We address this by proposing DiagramIR: an automatic and scalable evaluation pipeline for geometric figures. Our method relies on intermediate representations (IRs) of LaTeX TikZ code. We compare our pipeline to other evaluation baselines such as LLM-as-a-Judge, showing that our approach has higher agreement with human raters. This evaluation approach also enables smaller models like GPT-4.1-Mini to perform comparably to larger models such as GPT-5 at a 10x lower inference cost, which is important for deploying accessible and scalable education technologies. 

**Abstract (ZH)**: 大型语言模型（LLMs）在学习中被越来越多地采用；然而，大多数工具仍为纯文本形式，限制了其在需要可视化的内容（如数学）领域的 usefulness。最近的研究表明，LLMs能够生成能编译为教育图表的代码，但一个主要障碍是这些图表的可扩展评估。我们通过提出DiagramIR：一种用于几何图形的自动和可扩展评估流程来解决这一问题。我们的方法依赖于LaTeX TikZ代码的中间表示（IR）。我们将我们的流程与其他评估基准（如LLM-as-a-Judge）进行比较，表明我们的方法与人类评分者的同意率更高。这种评估方法还使得较小的模型如GPT-4.1-Mini能够在比GPT-5低10倍的推理成本下表现得与较大模型相当，这对于部署无障碍和可扩展的教育技术至关重要。 

---
# Multi-Agent GraphRAG: A Text-to-Cypher Framework for Labeled Property Graphs 

**Title (ZH)**: 多代理GraphRAG框架：一种面向标记属性图的文本到密文框架 

**Authors**: Anton Gusarov, Anastasia Volkova, Valentin Khrulkov, Andrey Kuznetsov, Evgenii Maslov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2511.08274)  

**Abstract**: While Retrieval-Augmented Generation (RAG) methods commonly draw information from unstructured documents, the emerging paradigm of GraphRAG aims to leverage structured data such as knowledge graphs. Most existing GraphRAG efforts focus on Resource Description Framework (RDF) knowledge graphs, relying on triple representations and SPARQL queries. However, the potential of Cypher and Labeled Property Graph (LPG) databases to serve as scalable and effective reasoning engines within GraphRAG pipelines remains underexplored in current research literature. To fill this gap, we propose Multi-Agent GraphRAG, a modular LLM agentic system for text-to-Cypher query generation serving as a natural language interface to LPG-based graph data. Our proof-of-concept system features an LLM-based workflow for automated Cypher queries generation and execution, using Memgraph as the graph database backend. Iterative content-aware correction and normalization, reinforced by an aggregated feedback loop, ensures both semantic and syntactic refinement of generated queries. We evaluate our system on the CypherBench graph dataset covering several general domains with diverse types of queries. In addition, we demonstrate performance of the proposed workflow on a property graph derived from the IFC (Industry Foundation Classes) data, representing a digital twin of a building. This highlights how such an approach can bridge AI with real-world applications at scale, enabling industrial digital automation use cases. 

**Abstract (ZH)**: 多代理GraphRAG：面向LPG的可扩展且有效的Cypher查询生成系统 

---
# Towards Outcome-Oriented, Task-Agnostic Evaluation of AI Agents 

**Title (ZH)**: 面向结果导向、任务无关的AI代理评估 

**Authors**: Waseem AlShikh, Muayad Sayed Ali, Brian Kennedy, Dmytro Mozolevskyi  

**Link**: [PDF](https://arxiv.org/pdf/2511.08242)  

**Abstract**: As AI agents proliferate across industries and applications, evaluating their performance based solely on infrastructural metrics such as latency, time-to-first-token, or token throughput is proving insufficient. These metrics fail to capture the quality of an agent's decisions, its operational autonomy, or its ultimate business value. This white paper proposes a novel, comprehensive framework of eleven outcome-based, task-agnostic performance metrics for AI agents that transcend domain boundaries. These metrics are designed to enable organizations to evaluate agents based on the quality of their decisions, their degree of autonomy, their adaptability to new challenges, and the tangible business value they deliver, regardless of the underlying model architecture or specific use case. We introduce metrics such as Goal Completion Rate (GCR), Autonomy Index (AIx), Multi-Step Task Resilience (MTR), and Business Impact Efficiency (BIE). Through a large-scale simulated experiment involving four distinct agent architectures (ReAct, Chain-of-Thought, Tool-Augmented, Hybrid) across five diverse domains (Healthcare, Finance, Marketing, Legal, and Customer Service), we demonstrate the framework's efficacy. Our results reveal significant performance trade-offs between different agent designs, highlighting the Hybrid Agent as the most consistently high-performing model across the majority of our proposed metrics, achieving an average Goal Completion Rate of 88.8\% and the highest Return on Investment (ROI). This work provides a robust, standardized methodology for the holistic evaluation of AI agents, paving the way for more effective development, deployment, and governance. 

**Abstract (ZH)**: 随着人工智能代理在各个行业和应用中的普及，仅基于基础设施指标如延迟、首个词 token 时间或 token 通量来评估其性能 proven 无效。这些指标未能捕捉到代理决策的质量、操作自主性或最终的业务价值。本白皮书提出了一种新颖的、跨域的综合框架，包含 eleven 项基于结果、任务无关的性能指标，用于评估人工智能代理。这些指标旨在使组织能够根据代理决策的质量、自主性的程度、应对新挑战的适应性以及实际交付的业务价值来评估代理，而不考虑底层模型架构或特定应用场景。我们介绍了如目标完成率（GCR）、自主性指数（AIx）、多步任务 resilience（MTR）和业务影响效率（BIE）等指标。通过涉及四个不同代理架构（ReAct、链式思考、工具增强、混合）和五个不同领域（医疗保健、金融、营销、法律和客户服务）的大型模拟实验，我们展示了该框架的有效性。我们的研究结果显示了不同代理设计之间的显著性能权衡，强调混合代理在大多数提出指标中表现最高，平均目标完成率为 88.8% 并具有最高的投资回报率（ROI）。本工作提供了一种 robust、标准化的评估方法，用于全面评估人工智能代理，为更有效的开发、部署和治理铺平了道路。 

---
# Towards Provably Unlearnable Examples via Bayes Error Optimisation 

**Title (ZH)**: 通过贝叶斯错误优化实现可证明的不可学会例子 

**Authors**: Ruihan Zhang, Jun Sun, Ee-Peng Lim, Peixin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08191)  

**Abstract**: The recent success of machine learning models, especially large-scale classifiers and language models, relies heavily on training with massive data. These data are often collected from online sources. This raises serious concerns about the protection of user data, as individuals may not have given consent for their data to be used in training. To address this concern, recent studies introduce the concept of unlearnable examples, i.e., data instances that appear natural but are intentionally altered to prevent models from effectively learning from them. While existing methods demonstrate empirical effectiveness, they typically rely on heuristic trials and lack formal guarantees. Besides, when unlearnable examples are mixed with clean data, as is often the case in practice, their unlearnability disappears. In this work, we propose a novel approach to constructing unlearnable examples by systematically maximising the Bayes error, a measurement of irreducible classification error. We develop an optimisation-based approach and provide an efficient solution using projected gradient ascent. Our method provably increases the Bayes error and remains effective when the unlearning examples are mixed with clean samples. Experimental results across multiple datasets and model architectures are consistent with our theoretical analysis and show that our approach can restrict data learnability, effectively in practice. 

**Abstract (ZH)**: 最近机器学习模型的成功，尤其是大规模分类器和语言模型，很大程度上依赖于大规模数据的训练。这些数据通常来自网络来源，这引起了关于用户数据保护的严重担忧，因为个人可能并未同意其数据被用于训练。为解决这一问题，近期研究引入了不可学习示例的概念，即看起来自然但实际上故意修改以防止模型有效学习的数据实例。虽然现有方法从实证上表现出有效性，但它们通常依赖于启发式试验，并缺乏形式上的保证。此外，当不可学习示例与干净数据混合时，如实践中常见的情况，它们的不可学习性会消失。在本文中，我们提出了一种新方法，通过系统最大化贝叶斯错误率（衡量不可约分类错误的指标）来构建不可学习示例。我们开发了一种基于优化的方法，并提供了一种使用投影梯度上升的有效解决方案。我们的方法能够证明地增加贝叶斯错误率，并在不可学习示例与干净样本混合时保持有效性。在多个数据集和模型架构上的实验结果与我们的理论分析一致，表明我们的方法可以在实践中有效限制数据的可学习性。 

---
# SciAgent: A Unified Multi-Agent System for Generalistic Scientific Reasoning 

**Title (ZH)**: SciAgent: 通用科学推理的统一多代理系统 

**Authors**: Xuchen Li, Ruitao Wu, Xuanbo Liu, Xukai Wang, Jinbo Hu, Zhixin Bai, Bohan Zeng, Hao Liang, Leheng Chen, Mingrui Chen, Haitian Zhong, Xuanlin Yang, Xu-Yao Zhang, Liu Liu, Jia Li, Kaiqi Huang, Jiahao Xu, Haitao Mi, Wentao Zhang, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2511.08151)  

**Abstract**: Recent advances in large language models have enabled AI systems to achieve expert-level performance on domain-specific scientific tasks, yet these systems remain narrow and handcrafted. We introduce SciAgent, a unified multi-agent system designed for generalistic scientific reasoning-the ability to adapt reasoning strategies across disciplines and difficulty levels. SciAgent organizes problem solving as a hierarchical process: a Coordinator Agent interprets each problem's domain and complexity, dynamically orchestrating specialized Worker Systems, each composed of interacting reasoning Sub-agents for symbolic deduction, conceptual modeling, numerical computation, and verification. These agents collaboratively assemble and refine reasoning pipelines tailored to each task. Across mathematics and physics Olympiads (IMO, IMC, IPhO, CPhO), SciAgent consistently attains or surpasses human gold-medalist performance, demonstrating both domain generality and reasoning adaptability. Additionally, SciAgent has been tested on the International Chemistry Olympiad (IChO) and selected problems from the Humanity's Last Exam (HLE) benchmark, further confirming the system's ability to generalize across diverse scientific domains. This work establishes SciAgent as a concrete step toward generalistic scientific intelligence-AI systems capable of coherent, cross-disciplinary reasoning at expert levels. 

**Abstract (ZH)**: Recent Advances in Large Language Models Have Enabled AI Systems to Achieve Expert-Level Performance on Domain-Specific Scientific Tasks, Yet These Systems Remain Narrow and Handcrafted: Introducing SciAgent, a Unified Multi-Agent System for Generalistic Scientific Reasoning 

---
# National Institute on Aging PREPARE Challenge: Early Detection of Cognitive Impairment Using Speech - The SpeechCARE Solution 

**Title (ZH)**: 国家老龄化研究所 PREPARE 挑战：通过语音早期检测认知障碍 - SpeechCARE 解决方案 

**Authors**: Maryam Zolnoori, Hossein Azadmaleki, Yasaman Haghbin, Ali Zolnour, Mohammad Javad Momeni Nezhad, Sina Rashidi, Mehdi Naserian, Elyas Esmaeili, Sepehr Karimi Arpanahi  

**Link**: [PDF](https://arxiv.org/pdf/2511.08132)  

**Abstract**: Alzheimer's disease and related dementias (ADRD) affect one in five adults over 60, yet more than half of individuals with cognitive decline remain undiagnosed. Speech-based assessments show promise for early detection, as phonetic motor planning deficits alter acoustic features (e.g., pitch, tone), while memory and language impairments lead to syntactic and semantic errors. However, conventional speech-processing pipelines with hand-crafted features or general-purpose audio classifiers often exhibit limited performance and generalizability. To address these limitations, we introduce SpeechCARE, a multimodal speech processing pipeline that leverages pretrained, multilingual acoustic and linguistic transformer models to capture subtle speech-related cues associated with cognitive impairment. Inspired by the Mixture of Experts (MoE) paradigm, SpeechCARE employs a dynamic fusion architecture that weights transformer-based acoustic, linguistic, and demographic inputs, allowing integration of additional modalities (e.g., social factors, imaging) and enhancing robustness across diverse tasks. Its robust preprocessing includes automatic transcription, large language model (LLM)-based anomaly detection, and task identification. A SHAP-based explainability module and LLM reasoning highlight each modality's contribution to decision-making. SpeechCARE achieved AUC = 0.88 and F1 = 0.72 for classifying cognitively healthy, MCI, and AD individuals, with AUC = 0.90 and F1 = 0.62 for MCI detection. Bias analysis showed minimal disparities, except for adults over 80. Mitigation techniques included oversampling and weighted loss. Future work includes deployment in real-world care settings (e.g., VNS Health, Columbia ADRC) and EHR-integrated explainability for underrepresented populations in New York City. 

**Abstract (ZH)**: 阿尔茨海默病及相关痴呆症（ADRD）影响超过60岁成年人中的五分之一，但超过一半存在认知衰退的个体未被诊断。语音评估显示出早期检测的潜力，因为语音发音运动计划缺陷会改变声学特征（如音高、音调），而记忆和语言障碍会导致句法和语义错误。然而，传统的手工特征提取或通用音频分类器的语音处理管道通常表现出受限的性能和泛化能力。为解决这些限制，我们引入了SpeechCARE，这是一种利用预训练多语言声学和语言变换器模型的多模态语音处理管道，以捕捉与认知损害相关的细微语音线索。受到专家混合（MoE）范式的启发，SpeechCARE采用了一种动态融合架构，对基于变换器的声学、语言和人口统计学输入进行加权，允许整合其他模态（如社会因素、成像）并增强跨多样化任务的鲁棒性。稳健的预处理包括自动转录、基于大规模语言模型的异常检测和任务识别。基于SHAP的可解释性模块和大规模语言模型推理突显了每个模态对决策的贡献。SpeechCARE在分类认知健康、轻度认知 impairment（MCI）和阿尔茨海默病（AD）个体方面的AUC为0.88，F1为0.72，而在MCI检测方面的AUC为0.90，F1为0.62。偏差分析显示除80岁以上成人外，差异较小。缓解技术包括过采样和加权损失。未来的工作包括在真实世界护理环境中部署（如VNS Health、哥伦比亚ADRC）以及将解释性融入电子健康记录（EHR）以增强纽约市代表性不足群体的解释性。 

---
# Advancements in synthetic data extraction for industrial injection molding 

**Title (ZH)**: 合成数据提取在工业注塑成型领域的进展 

**Authors**: Georg Rottenwalter, Marcel Tilly, Christian Bielenberg, Katharina Obermeier  

**Link**: [PDF](https://arxiv.org/pdf/2511.08117)  

**Abstract**: Machine learning has significant potential for optimizing various industrial processes. However, data acquisition remains a major challenge as it is both time-consuming and costly. Synthetic data offers a promising solution to augment insufficient data sets and improve the robustness of machine learning models. In this paper, we investigate the feasibility of incorporating synthetic data into the training process of the injection molding process using an existing Long Short-Term Memory architecture. Our approach is to generate synthetic data by simulating production cycles and incorporating them into the training data set. Through iterative experimentation with different proportions of synthetic data, we attempt to find an optimal balance that maximizes the benefits of synthetic data while preserving the authenticity and relevance of real data. Our results suggest that the inclusion of synthetic data improves the model's ability to handle different scenarios, with potential practical industrial applications to reduce manual labor, machine use, and material waste. This approach provides a valuable alternative for situations where extensive data collection and maintenance has been impractical or costly and thus could contribute to more efficient manufacturing processes in the future. 

**Abstract (ZH)**: 机器学习在优化各种工业过程方面具有巨大潜力。然而，数据获取仍然是一个主要挑战，因为它既耗时又昂贵。合成数据为增强不足的数据集和提升机器学习模型的鲁棒性提供了颇有前景的解决方案。在本文中，我们考察了将合成数据融入注射成型过程训练过程的可能性，采用了现有的长短期记忆架构。我们通过模拟生产周期生成合成数据，并将其纳入训练数据集。通过迭代实验不同比例的合成数据，我们尝试找到一个最佳平衡点，以最大化合成数据的益处同时保留真实数据的真实性和相关性。我们的结果表明，合成数据的纳入提高了模型应对不同场景的能力，具有潜在的实际工业应用价值，以减少人工劳动、设备使用和材料浪费。该方法为数据收集和维护难以实施或成本高昂的情况提供了有价值的替代方案，从而有助于未来更高效的制造过程。 

---
# Improving Industrial Injection Molding Processes with Explainable AI for Quality Classification 

**Title (ZH)**: 使用可解释的人工智能改进工业注塑成型工艺的质量分类 

**Authors**: Georg Rottenwalter, Marcel Tilly, Victor Owolabi  

**Link**: [PDF](https://arxiv.org/pdf/2511.08108)  

**Abstract**: Machine learning is an essential tool for optimizing industrial quality control processes. However, the complexity of machine learning models often limits their practical applicability due to a lack of interpretability. Additionally, many industrial machines lack comprehensive sensor technology, making data acquisition incomplete and challenging. Explainable Artificial Intelligence offers a solution by providing insights into model decision-making and identifying the most relevant features for classification. In this paper, we investigate the impact of feature reduction using XAI techniques on the quality classification of injection-molded parts. We apply SHAP, Grad-CAM, and LIME to analyze feature importance in a Long Short-Term Memory model trained on real production data. By reducing the original 19 input features to 9 and 6, we evaluate the trade-off between model accuracy, inference speed, and interpretability. Our results show that reducing features can improve generalization while maintaining high classification performance, with an small increase in inference speed. This approach enhances the feasibility of AI-driven quality control, particularly for industrial settings with limited sensor capabilities, and paves the way for more efficient and interpretable machine learning applications in manufacturing. 

**Abstract (ZH)**: 机器学习是优化工业质量控制过程的重要工具。然而，机器学习模型的复杂性往往因其可解释性不足而限制了其实际应用。此外，许多工业机器缺乏全面的传感器技术，导致数据获取不完整且具有挑战性。可解释的人工智能通过提供模型决策见解并识别分类中最相关的特征来提供解决方案。在本文中，我们探讨了使用XAI技术进行特征减少对注塑件质量分类的影响。我们应用SHAP、Grad-CAM和LIME来分析长短期记忆模型在实际生产数据上的特征重要性。通过将原始的19个输入特征减少到9个和6个，我们评估了模型准确性、推理速度和可解释性之间的权衡。我们的结果表明，减少特征可以提高模型的泛化能力，同时保持高分类性能，并且在推理速度上仅略有增加。这种方法增强了基于AI的 quality 控制的可行性，特别是在传感器能力有限的工业环境中，并为制造业中的更高效且可解释的机器学习应用程序铺平了道路。 

---
# Gateways to Tractability for Satisfiability in Pearl's Causal Hierarchy 

**Title (ZH)**: 珍珠因果层次中可满足性的可处理性途径 

**Authors**: Robert Ganian, Marlene Gründel, Simon Wietheger  

**Link**: [PDF](https://arxiv.org/pdf/2511.08091)  

**Abstract**: Pearl's Causal Hierarchy (PCH) is a central framework for reasoning about probabilistic, interventional, and counterfactual statements, yet the satisfiability problem for PCH formulas is computationally intractable in almost all classical settings. We revisit this challenge through the lens of parameterized complexity and identify the first gateways to tractability. Our results include fixed-parameter and XP-algorithms for satisfiability in key probabilistic and counterfactual fragments, using parameters such as primal treewidth and the number of variables, together with matching hardness results that map the limits of tractability. Technically, we depart from the dynamic programming paradigm typically employed for treewidth-based algorithms and instead exploit structural characterizations of well-formed causal models, providing a new algorithmic toolkit for causal reasoning. 

**Abstract (ZH)**: Pearl的因果层次结构（PCH）是处理概率性、干预性和反事实性语句的核心框架，但在几乎所有经典设置中，PCH公式的可满足性问题都是计算上不可行的。我们通过参数化复杂性的视角重新审视这一挑战，并识别出通向可行性的第一个路径。我们的结果包括针对关键概率性和反事实性片段的固定参数算法和XP算法，使用诸如原始 treewidth 和变量数量等参数，并伴有相应的复杂性结果，精确捕捉可行性的边界。技术上，我们偏离了通常用于 treewidth 基础算法的动态规划范式，而是利用良好形式的因果模型的结构特征，提供了一种新的因果推理算法工具箱。 

---
# Clustering-based Anomaly Detection in Multivariate Time Series Data 

**Title (ZH)**: 基于聚类的多变量时间序列异常检测 

**Authors**: Jinbo Li, Hesam Izakian, Witold Pedrycz, Iqbal Jamal  

**Link**: [PDF](https://arxiv.org/pdf/2511.08072)  

**Abstract**: Multivariate time series data come as a collection of time series describing different aspects of a certain temporal phenomenon. Anomaly detection in this type of data constitutes a challenging problem yet with numerous applications in science and engineering because anomaly scores come from the simultaneous consideration of the temporal and variable relationships. In this paper, we propose a clustering-based approach to detect anomalies concerning the amplitude and the shape of multivariate time series. First, we use a sliding window to generate a set of multivariate subsequences and thereafter apply an extended fuzzy clustering to reveal a structure present within the generated multivariate subsequences. Finally, a reconstruction criterion is employed to reconstruct the multivariate subsequences with the optimal cluster centers and the partition matrix. We construct a confidence index to quantify a level of anomaly detected in the series and apply Particle Swarm Optimization as an optimization vehicle for the problem of anomaly detection. Experimental studies completed on several synthetic and six real-world datasets suggest that the proposed methods can detect the anomalies in multivariate time series. With the help of available clusters revealed by the extended fuzzy clustering, the proposed framework can detect anomalies in the multivariate time series and is suitable for identifying anomalous amplitude and shape patterns in various application domains such as health care, weather data analysis, finance, and disease outbreak detection. 

**Abstract (ZH)**: 基于聚类的多变量时间序列异常检测方法 

---
# Combining LLM Semantic Reasoning with GNN Structural Modeling for Multi-view Multi-Label Feature Selection 

**Title (ZH)**: 结合LLM语义推理与GNN结构建模的多视图多标签特征选择 

**Authors**: Zhiqi Chen, Yuzhou Liu, Jiarui Liu, Wanfu Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.08008)  

**Abstract**: Multi-view multi-label feature selection aims to identify informative features from heterogeneous views, where each sample is associated with multiple interdependent labels. This problem is particularly important in machine learning involving high-dimensional, multimodal data such as social media, bioinformatics or recommendation systems. Existing Multi-View Multi-Label Feature Selection (MVMLFS) methods mainly focus on analyzing statistical information of data, but seldom consider semantic information. In this paper, we aim to use these two types of information jointly and propose a method that combines Large Language Models (LLMs) semantic reasoning with Graph Neural Networks (GNNs) structural modeling for MVMLFS. Specifically, the method consists of three main components. (1) LLM is first used as an evaluation agent to assess the latent semantic relevance among feature, view, and label descriptions. (2) A semantic-aware heterogeneous graph with two levels is designed to represent relations among features, views and labels: one is a semantic graph representing semantic relations, and the other is a statistical graph. (3) A lightweight Graph Attention Network (GAT) is applied to learn node embedding in the heterogeneous graph as feature saliency scores for ranking and selection. Experimental results on multiple benchmark datasets demonstrate the superiority of our method over state-of-the-art baselines, and it is still effective when applied to small-scale datasets, showcasing its robustness, flexibility, and generalization ability. 

**Abstract (ZH)**: 多视图多标签特征选择旨在从异构视图中识别出具有信息性的特征，其中每个样本与多个相互依赖的标签相关联。该问题在涉及高维多模态数据的机器学习中尤为重要，例如社交媒体、生物信息学或推荐系统。现有的多视图多标签特征选择（MVMLS）方法主要侧重于分析数据的统计信息，而很少考虑语义信息。在本文中，我们旨在结合这两种信息，提出一种结合大型语言模型（LLMs）语义推理与图神经网络（GNNs）结构建模的MVMLS方法。具体而言，该方法包含三个主要组件：（1）首先使用LLMs作为评估代理来评估特征、视图和标签描述之间的潜在语义相关性。（2）设计一个具有两层的语义感知异构图来表示特征、视图和标签之间的关系：一个是表示语义关系的语义图，另一个是统计图。（3）应用一个轻量级的图注意力网络（GAT）来从异构图中学习节点嵌入作为特征显著性分数用于排名和选择。在多个基准数据集上的实验结果表明，我们的方法优于现有最先进的基线方法，并且在小型数据集上仍然有效，展示了其鲁棒性、灵活性和泛化能力。 

---
# Multivariate Time series Anomaly Detection:A Framework of Hidden Markov Models 

**Title (ZH)**: 多变量时间序列异常检测：隐马尔可夫模型框架 

**Authors**: Jinbo Li, Witold Pedrycz, Iqbal Jamal  

**Link**: [PDF](https://arxiv.org/pdf/2511.07995)  

**Abstract**: In this study, we develop an approach to multivariate time series anomaly detection focused on the transformation of multivariate time series to univariate time series. Several transformation techniques involving Fuzzy C-Means (FCM) clustering and fuzzy integral are studied. In the sequel, a Hidden Markov Model (HMM), one of the commonly encountered statistical methods, is engaged here to detect anomalies in multivariate time series. We construct HMM-based anomaly detectors and in this context compare several transformation methods. A suite of experimental studies along with some comparative analysis is reported. 

**Abstract (ZH)**: 本研究开发了一种多变量时间序列异常检测方法，专注于将多变量时间序列转换为单变量时间序列。研究了几种涉及Fuzzy C-Means (FCM) 聚类和模糊积分的转换技术。随后，使用常见的统计方法之一隐马尔可夫模型（HMM）来检测多变量时间序列中的异常。构建了基于HMM的异常检测器，并在这一背景下比较了几种转换方法。报告了一系列实验研究及一些对比分析。 

---
# Enhancing Logical Expressiveness in Graph Neural Networks via Path-Neighbor Aggregation 

**Title (ZH)**: 通过路径-邻居聚合增强图神经网络的逻辑表达能力 

**Authors**: Han Yu, Xiaojuan Zhao, Aiping Li, Kai Chen, Ziniu Liu, Zhichao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2511.07994)  

**Abstract**: Graph neural networks (GNNs) can effectively model structural information of graphs, making them widely used in knowledge graph (KG) reasoning. However, existing studies on the expressive power of GNNs mainly focuses on simple single-relation graphs, and there is still insufficient discussion on the power of GNN to express logical rules in KGs. How to enhance the logical expressive power of GNNs is still a key issue. Motivated by this, we propose Path-Neighbor enhanced GNN (PN-GNN), a method to enhance the logical expressive power of GNN by aggregating node-neighbor embeddings on the reasoning path. First, we analyze the logical expressive power of existing GNN-based methods and point out the shortcomings of the expressive power of these methods. Then, we theoretically investigate the logical expressive power of PN-GNN, showing that it not only has strictly stronger expressive power than C-GNN but also that its $(k+1)$-hop logical expressiveness is strictly superior to that of $k$-hop. Finally, we evaluate the logical expressive power of PN-GNN on six synthetic datasets and two real-world datasets. Both theoretical analysis and extensive experiments confirm that PN-GNN enhances the expressive power of logical rules without compromising generalization, as evidenced by its competitive performance in KG reasoning tasks. 

**Abstract (ZH)**: 路径-邻居增强图神经网络：提升知识图谱推理的逻辑表达能力 

---
# Capturing Complex Spatial-Temporal Dependencies in Traffic Forecasting: A Self-Attention Approach 

**Title (ZH)**: 在交通预测中捕捉复杂的空间-时间依赖关系：一种自注意力方法 

**Authors**: Zheng Chenghong, Zongyin Deng, Liu Cheng, Xiong Simin, Di Deshi, Li Guanyao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07980)  

**Abstract**: We study the problem of traffic forecasting, aiming to predict the inflow and outflow of a region in the subsequent time slot. The problem is complex due to the intricate spatial and temporal interdependence among regions. Prior works study the spatial and temporal dependency in a decouple manner, failing to capture their joint effect. In this work, we propose ST-SAM, a novel and efficient Spatial-Temporal Self-Attention Model for traffic forecasting. ST-SAM uses a region embedding layer to learn time-specific embedding from traffic data for regions. Then, it employs a spatial-temporal dependency learning module based on self-attention mechanism to capture the joint spatial-temporal dependency for both nearby and faraway regions. ST-SAM entirely relies on self-attention to capture both local and global spatial-temporal correlations, which make it effective and efficient. Extensive experiments on two real world datasets show that ST-SAM is substantially more accurate and efficient than the state-of-the-art approaches (with an average improvement of up to 15% on RMSE, 17% on MAPE, and 32 times on training time in our experiments). 

**Abstract (ZH)**: 基于空间-时间自注意力机制的交通流量预测模型：ST-SAM 

---
# Towards Fine-Grained Interpretability: Counterfactual Explanations for Misclassification with Saliency Partition 

**Title (ZH)**: 面向细粒度可解释性的基于显著性分区的反事实解释研究：错分类案例的理解 

**Authors**: Lintong Zhang, Kang Yin, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.07974)  

**Abstract**: Attribution-based explanation techniques capture key patterns to enhance visual interpretability; however, these patterns often lack the granularity needed for insight in fine-grained tasks, particularly in cases of model misclassification, where explanations may be insufficiently detailed. To address this limitation, we propose a fine-grained counterfactual explanation framework that generates both object-level and part-level interpretability, addressing two fundamental questions: (1) which fine-grained features contribute to model misclassification, and (2) where dominant local features influence counterfactual adjustments. Our approach yields explainable counterfactuals in a non-generative manner by quantifying similarity and weighting component contributions within regions of interest between correctly classified and misclassified samples. Furthermore, we introduce a saliency partition module grounded in Shapley value contributions, isolating features with region-specific relevance. Extensive experiments demonstrate the superiority of our approach in capturing more granular, intuitively meaningful regions, surpassing fine-grained methods. 

**Abstract (ZH)**: 基于归因的解释技术捕获关键模式以增强视觉可解释性，但在细粒度任务中这些模式往往缺乏必要的粒度，特别是在模型误分类的情况下，解释可能不够详细。为解决这一限制，我们提出了一种细粒度的反事实解释框架，生成对象级和部分级可解释性，以回答两个基本问题：（1）哪些细粒度特征导致模型误分类？（2）局部主要特征在哪里影响反事实调整？我们的方法通过量化相似度并在感兴趣区域的组件贡献之间加权求和，以非生成的方式生成可解释的反事实。此外，我们引入了一个基于Shapley值贡献的显着性分区模块，隔离具有区域特异性相关性的特征。大量实验表明，我们的方法在捕获更精细、直观有意义的区域方面优于细粒度方法。 

---
# Versatile and Risk-Sensitive Cardiac Diagnosis via Graph-Based ECG Signal Representation 

**Title (ZH)**: 基于图的ECG信号表示的多功能和风险敏感性心脏诊断 

**Authors**: Yue Wang, Yuyang Xu, Renjun Hu, Fanqi Shen, Hanyun Jiang, Jun Wang, Jintai Chen, Danny Z. Chen, Jian Wu, Haochao Ying  

**Link**: [PDF](https://arxiv.org/pdf/2511.07973)  

**Abstract**: Despite the rapid advancements of electrocardiogram (ECG) signal diagnosis and analysis methods through deep learning, two major hurdles still limit their clinical adoption: the lack of versatility in processing ECG signals with diverse configurations, and the inadequate detection of risk signals due to sample imbalances. Addressing these challenges, we introduce VersAtile and Risk-Sensitive cardiac diagnosis (VARS), an innovative approach that employs a graph-based representation to uniformly model heterogeneous ECG signals. VARS stands out by transforming ECG signals into versatile graph structures that capture critical diagnostic features, irrespective of signal diversity in the lead count, sampling frequency, and duration. This graph-centric formulation also enhances diagnostic sensitivity, enabling precise localization and identification of abnormal ECG patterns that often elude standard analysis methods. To facilitate representation transformation, our approach integrates denoising reconstruction with contrastive learning to preserve raw ECG information while highlighting pathognomonic patterns. We rigorously evaluate the efficacy of VARS on three distinct ECG datasets, encompassing a range of structural variations. The results demonstrate that VARS not only consistently surpasses existing state-of-the-art models across all these datasets but also exhibits substantial improvement in identifying risk signals. Additionally, VARS offers interpretability by pinpointing the exact waveforms that lead to specific model outputs, thereby assisting clinicians in making informed decisions. These findings suggest that our VARS will likely emerge as an invaluable tool for comprehensive cardiac health assessment. 

**Abstract (ZH)**: 尽管通过深度学习实现了心电图（ECG）信号诊断和分析方法的快速进步，但临床应用仍然受到两大瓶颈的限制：处理具有多种配置的ECG信号缺乏灵活性，以及由于样本不平衡导致的风险信号检测不足。为解决这些挑战，我们引入了一种创新方法——多功能和风险敏感心电图诊断（VARS），该方法采用基于图的表示形式统一建模异质ECG信号。VARS通过将ECG信号转化为多功能的图结构，捕捉关键诊断特征，不受导联数量、采样频率和持续时间的多样性影响。这种图中心的形式表述也提高了诊断灵敏度，能够精确定位和识别标准分析方法常忽略的异常ECG模式。为了促进表示转换，我们的方法结合了降噪重建和对比学习，以保留原始ECG信息并突出特定病理模式。我们严格评估了VARS在三个不同的ECG数据集上的有效性，涵盖多种结构变异性。结果表明，VARS不仅在所有数据集上均超过现有最先进的模型，还在识别风险信号方面表现出显著改进。此外，VARS提供了可解释性，通过指明导致特定模型输出的确切波形，辅助临床决策。这些发现表明，我们的VARS有望成为全面心脏健康评估的重要工具。 

---
# TimeFlow: Towards Stochastic-Aware and Efficient Time Series Generation via Flow Matching Modeling 

**Title (ZH)**: TimeFlow：面向流动匹配建模的 stochastic 意识高效时间序列生成 

**Authors**: He Panjing, Cheng Mingyue, Li Li, Zhang XiaoHan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07968)  

**Abstract**: Generating high-quality time series data has emerged as a critical research topic due to its broad utility in supporting downstream time series mining tasks. A major challenge lies in modeling the intrinsic stochasticity of temporal dynamics, as real-world sequences often exhibit random fluctuations and localized variations. While diffusion models have achieved remarkable success, their generation process is computationally inefficient, often requiring hundreds to thousands of expensive function evaluations per sample. Flow matching has emerged as a more efficient paradigm, yet its conventional ordinary differential equation (ODE)-based formulation fails to explicitly capture stochasticity, thereby limiting the fidelity of generated sequences. By contrast, stochastic differential equation (SDE) are naturally suited for modeling randomness and uncertainty. Motivated by these insights, we propose TimeFlow, a novel SDE-based flow matching framework that integrates a encoder-only architecture. Specifically, we design a component-wise decomposed velocity field to capture the multi-faceted structure of time series and augment the vanilla flow-matching optimization with an additional stochastic term to enhance representational expressiveness. TimeFlow is flexible and general, supporting both unconditional and conditional generation tasks within a unified framework. Extensive experiments across diverse datasets demonstrate that our model consistently outperforms strong baselines in generation quality, diversity, and efficiency. 

**Abstract (ZH)**: 基于SDE的时间流模型：一种新颖的时间序列生成框架 

---
# Toward Practical BCI: A Real-time Wireless Imagined Speech EEG Decoding System 

**Title (ZH)**: 面向实际应用的实时无线想象 speech EEG解码系统 

**Authors**: Ji-Ha Park, Heon-Gyu Kwak, Gi-Hwan Shin, Yoo-In Jeon, Sun-Min Park, Ji-Yeon Hwang, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.07936)  

**Abstract**: Brain-computer interface (BCI) research, while promising, has largely been confined to static and fixed environments, limiting real-world applicability. To move towards practical BCI, we introduce a real-time wireless imagined speech electroencephalogram (EEG) decoding system designed for flexibility and everyday use. Our framework focuses on practicality, demonstrating extensibility beyond wired EEG devices to portable, wireless hardware. A user identification module recognizes the operator and provides a personalized, user-specific service. To achieve seamless, real-time operation, we utilize the lab streaming layer to manage the continuous streaming of live EEG signals to the personalized decoder. This end-to-end pipeline enables a functional real-time application capable of classifying user commands from imagined speech EEG signals, achieving an overall 4-class accuracy of 62.00 % on a wired device and 46.67 % on a portable wireless headset. This paper demonstrates a significant step towards truly practical and accessible BCI technology, establishing a clear direction for future research in robust, practical, and personalized neural interfaces. 

**Abstract (ZH)**: 脑机接口（BCI）研究虽具潜力，但主要局限于静态环境，限制了其在现实世界的应用。为了迈向实用的BCI技术，我们引入了一种实时无线意念语音脑电图（EEG）解码系统，旨在提高灵活性和日常生活中的使用便利性。该框架强调其实用性，展示了超越有线脑电图设备的灵活性，适用于便携式无线硬件。用户识别模块识别操作者并提供个性化服务。为了实现无缝的实时操作，我们利用实验室流传输层（Lab Streaming Layer）管理和传输实时脑电信号至个性化解码器。该端到端管道实现了一个功能性的实时应用，能够从意念语音EEG信号中分类用户命令，有线设备的准确率为62.00%，便携式无线耳机的准确率为46.67%。本文展示了一种向真正实用且易于访问的BCI技术的重要步骤，明确了未来在稳健、实用和个性化的神经界面方面的研究方向。 

---
# Lightweight Diffusion-based Framework for Online Imagined Speech Decoding in Aphasia 

**Title (ZH)**: 基于轻量级扩散的在线失语症想象语音解码框架 

**Authors**: Eunyeong Ko, Soowon Kim, Ha-Na Jo  

**Link**: [PDF](https://arxiv.org/pdf/2511.07920)  

**Abstract**: A diffusion-based neural decoding framework optimized for real-time imagined speech classification in individuals with aphasia. The system integrates a lightweight conditional diffusion encoder and convolutional classifier trained using subject-specific EEG data acquired from a Korean-language paradigm. A dual-criterion early stopping strategy enabled rapid convergence under limited calibration data, while dropout regularization and grouped temporal convolutions ensured stable generalization. During online operation, continuous EEG streams were processed in two-second sliding windows to generate class probabilities that dynamically modulated visual and auditory feedback according to decoding confidence. Across twenty real-time trials, the framework achieved 65% top-1 and 70% top-2 accuracy, outperforming offline evaluation (50% top-1). These results demonstrate the feasibility of deploying diffusion-based EEG decoding under practical clinical constraints, maintaining reliable performance despite environmental variability and minimal preprocessing. The proposed framework advances the translation of imagined speech brain-computer interfaces toward clinical communication support for individuals with severe expressive language impairment. 

**Abstract (ZH)**: 基于扩散的神经解码框架优化实时假想语音分类，适用于失语症患者。该系统整合了轻量级条件扩散编码器和使用韩语范式下特定被试EEG数据训练的卷积分类器。双标准早期停止策略在有限校准数据下实现了快速收敛，而dropout正则化和分组时序卷积确保了稳定的泛化能力。在在线运行期间，持续的EEG流在两秒滑动窗口中处理，以生成解码自信心动态调节视觉和听觉反馈的类概率。在二十次实时试验中，该框架实现了65%的顶级准确率和70%的前两名准确率，优于离线评估（50%的顶级准确率）。这些结果表明，在实际临床条件下部署基于扩散的EEG解码的可行性，能够在环境变化和最少预处理的情况下保持可靠性能。所提出框架推进了想象语音脑-机接口向严重表达语言障碍患者临床通信支持的应用转化。 

---
# Neurophysiological Characteristics of Adaptive Reasoning for Creative Problem-Solving Strategy 

**Title (ZH)**: 适应性推理的神经生理学特征：创造性问题解决策略 

**Authors**: Jun-Young Kim, Young-Seok Kweon, Gi-Hwan Shin, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.07912)  

**Abstract**: Adaptive reasoning enables humans to flexibly adjust inference strategies when environmental rules or contexts change, yet its underlying neural dynamics remain unclear. This study investigated the neurophysiological mechanisms of adaptive reasoning using a card-sorting paradigm combined with electroencephalography and compared human performance with that of a multimodal large language model. Stimulus- and feedback-locked analyses revealed coordinated delta-theta-alpha dynamics: early delta-theta activity reflected exploratory monitoring and rule inference, whereas occipital alpha engagement indicated confirmatory stabilization of attention after successful rule identification. In contrast, the multimodal large language model exhibited only short-term feedback-driven adjustments without hierarchical rule abstraction or genuine adaptive reasoning. These findings identify the neural signatures of human adaptive reasoning and highlight the need for brain-inspired artificial intelligence that incorporates oscillatory feedback coordination for true context-sensitive adaptation. 

**Abstract (ZH)**: 适应性推理使人能够在环境规则或上下文变化时灵活调整推理策略，但其神经动态机制尚不明确。本研究通过结合卡片分类范式和脑电图探讨了适应性推理的神经生理机制，并将人类表现与多模态大型语言模型的表现进行了比较。刺激和反馈锁定分析揭示了协调的δ-θ-α动态：早期的δ-θ活动反映了探索性监控和规则推理，而背侧α振荡的参与则表明在成功识别规则后对注意力的确认性稳定。相比之下，多模态大型语言模型仅表现出短期的反馈驱动调整，而缺乏层次规则抽象或真正的适应性推理。这些发现识别了人类适应性推理的神经特征，并强调了需要借鉴脑启发式方法的类脑人工智能，以实现真正的上下文感知适应。 

---
# DANS-KGC: Diffusion Based Adaptive Negative Sampling for Knowledge Graph Completion 

**Title (ZH)**: DANS-KGC: 基于扩散的自适应负采样知识图谱完成算法 

**Authors**: Haoning Li, Qinghua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07901)  

**Abstract**: Negative sampling (NS) strategies play a crucial role in knowledge graph representation. In order to overcome the limitations of existing negative sampling strategies, such as vulnerability to false negatives, limited generalization, and lack of control over sample hardness, we propose DANS-KGC (Diffusion-based Adaptive Negative Sampling for Knowledge Graph Completion). DANS-KGC comprises three key components: the Difficulty Assessment Module (DAM), the Adaptive Negative Sampling Module (ANS), and the Dynamic Training Mechanism (DTM). DAM evaluates the learning difficulty of entities by integrating semantic and structural features. Based on this assessment, ANS employs a conditional diffusion model with difficulty-aware noise scheduling, leveraging semantic and neighborhood information during the denoising phase to generate negative samples of diverse hardness. DTM further enhances learning by dynamically adjusting the hardness distribution of negative samples throughout training, enabling a curriculum-style progression from easy to hard examples. Extensive experiments on six benchmark datasets demonstrate the effectiveness and generalization ability of DANS-KGC, with the method achieving state-of-the-art results on all three evaluation metrics for the UMLS and YAGO3-10 datasets. 

**Abstract (ZH)**: 基于扩散的自适应负采样知识图填充值网络（Diffusion-based Adaptive Negative Sampling for Knowledge Graph Completion） 

---
# Toward Robust EEG-based Intention Decoding during Misarticulated Speech in Aphasia 

**Title (ZH)**: 面向失语症患者错音 speech 期间EEG基意图解码的鲁棒性研究 

**Authors**: Ha-Na Jo, Jung-Sun Lee, Eunyeong Ko  

**Link**: [PDF](https://arxiv.org/pdf/2511.07895)  

**Abstract**: Aphasia severely limits verbal communication due to impaired language production, often leading to frequent misarticulations during speech attempts. Despite growing interest in brain-computer interface technologies, relatively little attention has been paid to developing EEG-based communication support systems tailored for aphasic patients. To address this gap, we recruited a single participant with expressive aphasia and conducted an Korean-based automatic speech task. EEG signals were recorded during task performance, and each trial was labeled as either correct or incorrect depending on whether the intended word was successfully spoken. Spectral analysis revealed distinct neural activation patterns between the two trial types: misarticulated trials exhibited excessive delta power across widespread channels and increased theta-alpha activity in frontal regions. Building upon these findings, we developed a soft multitask learning framework with maximum mean discrepancy regularization that focus on delta features to jointly optimize class discrimination while aligning the EEG feature distributions of correct and misarticulated trials. The proposed model achieved 58.6 % accuracy for correct and 45.5 % for misarticulated trials-outperforming the baseline by over 45 % on the latter-demonstrating robust intention decoding even under articulation errors. These results highlight the feasibility of EEG-based assistive systems capable of supporting real-world, imperfect speech conditions in aphasia patients. 

**Abstract (ZH)**: 脑卒中导致的语言障碍严重限制了言语交流，常常导致言语尝试中频繁的发音错误。尽管脑-机接口技术引起越来越多的兴趣，但仍较少关注为语言障碍患者量身开发的基于脑电图的交流支持系统。为填补这一空白，我们招募了一名表达性失语症患者，并开展了基于韩语的自动语音任务。在任务执行过程中记录了脑电图信号，并根据是否成功发出目标词语将每次试验标记为正确或错误。频谱分析揭示了两类试验之间不同的神经激活模式：发音错误的试验广泛通道表现出过多的delta功率，并且前脑区域theta-alpha活动增加。基于这些发现，我们开发了一种以最大均值差异正则化的软多任务学习框架，重点关注delta特征，以联合优化类别区分并使正确和发音错误试验的脑电图特征分布对齐。所提出模型在正确试验中达到了58.6%的准确率，在发音错误试验中达到了45.5%的准确率，后者比基线模型高出超过45%，即使在发音错误下也展示了稳健的意图解码能力。这些结果突显了基于脑电图的辅助系统的可行性，能够在失语症患者的实际、不完美的言语条件下支持其交流。 

---
# Confidence-Aware Neural Decoding of Overt Speech from EEG: Toward Robust Brain-Computer Interfaces 

**Title (ZH)**: 基于EEG的语音信度意识神经解码：面向稳健的脑-机接口 

**Authors**: Soowon Kim, Byung-Kwan Ko, Seo-Hyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.07890)  

**Abstract**: Non-invasive brain-computer interfaces that decode spoken commands from electroencephalogram must be both accurate and trustworthy. We present a confidence-aware decoding framework that couples deep ensembles of compact, speech-oriented convolutional networks with post-hoc calibration and selective classification. Uncertainty is quantified using ensemble-based predictive entropy, top-two margin, and mutual information, and decisions are made with an abstain option governed by an accuracy-coverage operating point. The approach is evaluated on a multi-class overt speech dataset using a leakage-safe, block-stratified split that respects temporal contiguity. Compared with widely used baselines, the proposed method yields more reliable probability estimates, improved selective performance across operating points, and balanced per-class acceptance. These results suggest that confidence-aware neural decoding can provide robust, deployment-oriented behavior for real-world brain-computer interface communication systems. 

**Abstract (ZH)**: 非侵入式脑-机接口从脑电图解码 spoken commands 必须both 准确和可信赖。我们提出了一种基于信心的解码框架，该框架结合了紧凑且面向语音的卷积网络的深度集成、后续校准和选择性分类。不确定性通过集成预测熵、top-two差距和互信息来量化，并通过准确率-覆盖操作点来控制弃权选项进行决策。该方法在遵循时间连续性的泄漏安全、块分层划分的多类别显性语音数据集上进行评估。与广泛使用的基线方法相比，所提出的方法提供了更可靠的概率估计、在整个操作点上的改进的选择性能以及平衡的每类别接受度。这些结果表明，信心意识的神经解码可以为实际世界的脑-机接口通信系统提供稳健且面向部署的行为。 

---
# GAMA: A Neural Neighborhood Search Method with Graph-aware Multi-modal Attention for Vehicle Routing Problem 

**Title (ZH)**: GAMA：一种具有图意识多模态注意力机制的神经邻域搜索方法用于车辆路由问题 

**Authors**: Xiangling Chen, Yi Mei, Mengjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07850)  

**Abstract**: Recent advances in neural neighborhood search methods have shown potential in tackling Vehicle Routing Problems (VRPs). However, most existing approaches rely on simplistic state representations and fuse heterogeneous information via naive concatenation, limiting their ability to capture rich structural and semantic context. To address these limitations, we propose GAMA, a neural neighborhood search method with Graph-aware Multi-modal Attention model in VRP. GAMA encodes the problem instance and its evolving solution as distinct modalities using graph neural networks, and models their intra- and inter-modal interactions through stacked self- and cross-attention layers. A gated fusion mechanism further integrates the multi-modal representations into a structured state, enabling the policy to make informed and generalizable operator selection decisions. Extensive experiments conducted across various synthetic and benchmark instances demonstrate that the proposed algorithm GAMA significantly outperforms the recent neural baselines. Further ablation studies confirm that both the multi-modal attention mechanism and the gated fusion design play a key role in achieving the observed performance gains. 

**Abstract (ZH)**: 近期神经邻域搜索方法在车辆路线问题（VRP）中的进展展示了潜在的应用前景。然而，大多数现有方法依赖于简单的状态表示，并通过朴素的连接融合异构信息，限制了其捕捉丰富结构和语义上下文的能力。为了解决这些限制，我们提出了一种基于图的多模态注意力模型的神经邻域搜索方法GAMA，这种方法在VRP中用于车辆路线问题。GAMA 使用图神经网络将问题实例及其 evolving 解决策略编码为不同的模态，并通过堆叠的自注意力和跨注意力层建模其跨模态交互。分门控融合机制进一步将多模态表示整合为结构化状态，使策略能够做出知情且泛化的操作选择决策。在各种合成和基准实例上的广泛实验表明，所提出的算法GAMA 显著优于最近的神经基线方法。进一步的消融研究证实，多模态注意力机制和分门控融合设计对于实现观察到的性能提升起着关键作用。 

---
# Agentic Educational Content Generation for African Languages on Edge Devices 

**Title (ZH)**: 边缘设备上的代理型非洲语言教育内容生成 

**Authors**: Ravi Gupta, Guneet Bhatia  

**Link**: [PDF](https://arxiv.org/pdf/2511.07437)  

**Abstract**: Addressing educational inequity in Sub-Saharan Africa, this research presents an autonomous agent-orchestrated framework for decentralized, culturally adaptive educational content generation on edge devices. The system leverages four specialized agents that work together to generate contextually appropriate educational content. Experimental validation on platforms including Raspberry Pi 4B and NVIDIA Jetson Nano demonstrates significant performance achievements. InkubaLM on Jetson Nano achieved a Time-To-First-Token (TTFT) of 129 ms, an average inter-token latency of 33 ms, and a throughput of 45.2 tokens per second while consuming 8.4 W. On Raspberry Pi 4B, InkubaLM also led with 326 ms TTFT and 15.9 tokens per second at 5.8 W power consumption. The framework consistently delivered high multilingual quality, averaging a BLEU score of 0.688, cultural relevance of 4.4/5, and fluency of 4.2/5 across tested African languages. Through potential partnerships with active community organizations including African Youth & Community Organization (AYCO) and Florida Africa Foundation, this research aims to establish a practical foundation for accessible, localized, and sustainable AI-driven education in resource-constrained environments. Keeping focus on long-term viability and cultural appropriateness, it contributes to United Nations SDGs 4, 9, and 10. Index Terms - Multi-Agent Systems, Edge AI Computing, Educational Technology, African Languages, Rural Education, Sustainable Development, UN SDG. 

**Abstract (ZH)**: addressing 教育不平等在撒哈拉以南非洲地区的研究：一种自主代理 orchestrated 框架在边缘设备上进行分散且文化适应性教育内容生成的研究 

---
# SENCA-st: Integrating Spatial Transcriptomics and Histopathology with Cross Attention Shared Encoder for Region Identification in Cancer Pathology 

**Title (ZH)**: SENCA-st：结合空间转录组学和组织病理学的跨注意力共享编码器方法用于癌症病理学中的区域识别 

**Authors**: Shanaka Liyanaarachchi, Chathurya Wijethunga, Shihab Aaquil Ahamed, Akthas Absar, Ranga Rodrigo  

**Link**: [PDF](https://arxiv.org/pdf/2511.08573)  

**Abstract**: Spatial transcriptomics is an emerging field that enables the identification of functional regions based on the spatial distribution of gene expression. Integrating this functional information present in transcriptomic data with structural data from histopathology images is an active research area with applications in identifying tumor substructures associated with cancer drug resistance. Current histopathology-spatial-transcriptomic region segmentation methods suffer due to either making spatial transcriptomics prominent by using histopathology features just to assist processing spatial transcriptomics data or using vanilla contrastive learning that make histopathology images prominent due to only promoting common features losing functional information. In both extremes, the model gets either lost in the noise of spatial transcriptomics or overly smoothed, losing essential information. Thus, we propose our novel architecture SENCA-st (Shared Encoder with Neighborhood Cross Attention) that preserves the features of both modalities. More importantly, it emphasizes regions that are structurally similar in histopathology but functionally different on spatial transcriptomics using cross-attention. We demonstrate the superior performance of our model that surpasses state-of-the-art methods in detecting tumor heterogeneity and tumor micro-environment regions, a clinically crucial aspect. 

**Abstract (ZH)**: 空间转录组学是一种新兴领域，能够基于基因表达的空间分布识别功能区域。将转录组学数据中的功能性信息与病理图像的结构数据相结合是该领域的一个活跃研究方向，用于识别与癌症药物耐药性相关的肿瘤亚结构。当前的病理-空间转录组学区域分割方法要么过分强调空间转录组学而仅仅利用病理特征辅助处理空间转录组学数据，要么使用普通的对比学习方法过分强调病理图像而丢失功能性信息。在这两种极端情况下，模型要么被空间转录组学的噪声困扰，要么被过度平滑，丢失关键信息。因此，我们提出了一种新颖的架构SENCA-st（共享编码器与邻域交叉注意力），能够在保留两种模态特征的同时，利用交叉注意力强调在病理图像中结构相似但在空间转录组学中功能差异显著的区域。我们展示了该模型在检测肿瘤异质性和肿瘤微环境区域方面的出色性能，超过了现有的先进方法，这是临床研究中一个重要方面。 

---
# Automatic Grid Updates for Kolmogorov-Arnold Networks using Layer Histograms 

**Title (ZH)**: 使用层直方图的柯莫哥洛夫-阿诺尔德网络自动网格更新 

**Authors**: Jamison Moody, James Usevitch  

**Link**: [PDF](https://arxiv.org/pdf/2511.08570)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) are a class of neural networks that have received increased attention in recent literature. In contrast to MLPs, KANs leverage parameterized, trainable activation functions and offer several benefits including improved interpretability and higher accuracy on learning symbolic equations. However, the original KAN architecture requires adjustments to the domain discretization of the network (called the "domain grid") during training, creating extra overhead for the user in the training process. Typical KAN layers are not designed with the ability to autonomously update their domains in a data-driven manner informed by the changing output ranges of previous layers. As an added benefit, this histogram algorithm may also be applied towards detecting out-of-distribution (OOD) inputs in a variety of settings. We demonstrate that AdaptKAN exceeds or matches the performance of prior KAN architectures and MLPs on four different tasks: learning scientific equations from the Feynman dataset, image classification from frozen features, learning a control Lyapunov function, and detecting OOD inputs on the OpenOOD v1.5 benchmark. 

**Abstract (ZH)**: 柯尔莫戈罗夫-阿诺德网络（KANs）是一类在近期文献中受到越来越多关注的神经网络。与MLPs不同，KANs利用参数化和可训练的激活函数，并提供了包括更好的可解释性和在学习符号方程时更高的准确性的多项优势。然而，原始的KAN架构在训练过程中需要对网络的领域离散化（称为“领域网格”）进行调整，这为用户在训练过程中增加了额外的负担。典型的KAN层不具备根据前一层输出范围变化自主更新其领域的能力。此外，这一直方图算法还可以应用于多种场景中检测异常输入（OOD）。我们展示了AdaptKAN在四个不同任务上超过了或匹配了先前的KAN架构和MLPs的表现：从费曼数据集学习科学方程、基于冻结特征进行图像分类、学习控制李雅普诺夫函数以及在OpenOOD v1.5基准上检测异常输入。 

---
# The Path Not Taken: RLVR Provably Learns Off the Principals 

**Title (ZH)**: 未被探索的道路：RLVR 证明能学习远离原则 

**Authors**: Hanqing Zhu, Zhenyu Zhang, Hanxian Huang, DiJia Su, Zechun Liu, Jiawei Zhao, Igor Fedorov, Hamed Pirsiavash, Zhizhou Sha, Jinwon Lee, David Z. Pan, Zhangyang Wang, Yuandong Tian, Kai Sheng Tai  

**Link**: [PDF](https://arxiv.org/pdf/2511.08567)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) reliably improves the reasoning performance of large language models, yet it appears to modify only a small fraction of parameters. We revisit this paradox and show that sparsity is a surface artifact of a model-conditioned optimization bias: for a fixed pretrained model, updates consistently localize to preferred parameter regions, highly consistent across runs and largely invariant to datasets and RL recipes. We mechanistically explain these dynamics with a Three-Gate Theory: Gate I (KL Anchor) imposes a KL-constrained update; Gate II (Model Geometry) steers the step off principal directions into low-curvature, spectrum-preserving subspaces; and Gate III (Precision) hides micro-updates in non-preferred regions, making the off-principal bias appear as sparsity. We then validate this theory and, for the first time, provide a parameter-level characterization of RLVR's learning dynamics: RLVR learns off principal directions in weight space, achieving gains via minimal spectral drift, reduced principal-subspace rotation, and off-principal update alignment. In contrast, SFT targets principal weights, distorts the spectrum, and even lags RLVR.
Together, these results provide the first parameter-space account of RLVR's training dynamics, revealing clear regularities in how parameters evolve. Crucially, we show that RL operates in a distinct optimization regime from SFT, so directly adapting SFT-era parameter-efficient fine-tuning (PEFT) methods can be flawed, as evidenced by our case studies on advanced sparse fine-tuning and LoRA variants. We hope this work charts a path toward a white-box understanding of RLVR and the design of geometry-aware, RLVR-native learning algorithms, rather than repurposed SFT-era heuristics. 

**Abstract (ZH)**: 验证奖励的强化学习（RLVR）可靠地提高了大型语言模型的推理性能，但却似乎只修改了一小部分参数。我们重新审视这一悖论，并展示稀疏性是模型条件优化偏差的结果表象：对于固定的预训练模型，更新总是一致地集中在优选的参数区域，这一过程在不同运行中高度一致，并且对数据集和RL食谱变化不大。我们通过三门理论（Three-Gate Theory）机制性地解释这些动态过程：第一门（KL锚点）施加KL约束更新；第二门（模型几何）引导步骤偏离主方向，进入低曲率、保持谱特征的子空间；第三门（精度）隐藏微更新在非优选区域，使偏离主方向的偏差表现为稀疏性。然后验证这一理论，并首次在参数层面描述RLVR的学习动态：RLVR在权重空间中超主方向学习，通过最小的谱漂移、减少主子空间旋转和微更新对齐来实现性能提升。相比之下，SFT针对主权重，扭曲谱特征，甚至滞后于RLVR。这些结果提供了RLVR训练动态的第一个参数空间解释，揭示了参数演化中的明确规律。关键的是，我们证明RL位于与SFT不同的优化区间，因此直接适应SFT时代的参数高效微调（PEFT）方法可能是有缺陷的，这一点通过我们对高级稀疏微调和LoRA变体的案例研究得到了体现。我们希望这项工作能够为RLVR提供一个透明的理解路径，并设计几何感知的、RLVR原生的学习算法，而不是重新利用SFT时代的启发式方法。 

---
# LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics 

**Title (ZH)**: LeJEPA: 可证明和可扩展的无需启发式的自我监督学习 

**Authors**: Randall Balestriero, Yann LeCun  

**Link**: [PDF](https://arxiv.org/pdf/2511.08544)  

**Abstract**: Learning manipulable representations of the world and its dynamics is central to AI. Joint-Embedding Predictive Architectures (JEPAs) offer a promising blueprint, but lack of practical guidance and theory has led to ad-hoc R&D. We present a comprehensive theory of JEPAs and instantiate it in {\bf LeJEPA}, a lean, scalable, and theoretically grounded training objective. First, we identify the isotropic Gaussian as the optimal distribution that JEPAs' embeddings should follow to minimize downstream prediction risk. Second, we introduce a novel objective--{\bf Sketched Isotropic Gaussian Regularization} (SIGReg)--to constrain embeddings to reach that ideal distribution. Combining the JEPA predictive loss with SIGReg yields LeJEPA with numerous theoretical and practical benefits: (i) single trade-off hyperparameter, (ii) linear time and memory complexity, (iii) stability across hyper-parameters, architectures (ResNets, ViTs, ConvNets) and domains, (iv) heuristics-free, e.g., no stop-gradient, no teacher-student, no hyper-parameter schedulers, and (v) distributed training-friendly implementation requiring only $\approx$50 lines of code. Our empirical validation covers 10+ datasets, 60+ architectures, all with varying scales and domains. As an example, using imagenet-1k for pretraining and linear evaluation with frozen backbone, LeJEPA reaches 79\% with a ViT-H/14. We hope that the simplicity and theory-friendly ecosystem offered by LeJEPA will reestablish self-supervised pre-training as a core pillar of AI research (\href{git@github.com:rbalestr-lab/lejepa.git}{GitHub repo}). 

**Abstract (ZH)**: 学习可操控的世界及其动态的表示是AI的核心。联合嵌入预测架构（JEPAs）提供了一个有前景的设计蓝图，但由于缺乏实用指导和理论支持，导致了粗放的研发。我们提出了一套全面的JEPAs理论，并在此基础上构建了LeJEPA，一个精简、可扩展且理论基础坚实的学习目标。首先，我们确定各向同性高斯分布是JEPAs嵌入应当遵循的最优分布，以最小化下游预测风险。其次，我们引入了一种新的目标——草图各向同性高斯正则化（SIGReg），以约束嵌入达到理想分布。将JEPAs预测损失与SIGReg结合，获得了具有众多理论和实践优势的LeJEPA：（i）单一交易超参数，（ii）线性时间和内存复杂度，（iii）超参数、架构（残差网络、视觉变换器、卷积网络）和领域下的稳定性，（iv）无需启发式方法，例如无需梯度停止、无需教师-学生训练、无需超参数调度器，（v）仅需要约50行代码的分布式训练友好实现。我们的实证验证涵盖了10多个数据集、60多种架构，这些数据集和架构在规模和领域上各不相同。例如，使用imagenet-1k进行预训练和冻结主干的线性评估，LeJEPA达到了79%的性能，使用的架构是ViT-H/14。我们希望LeJEPA提供的简洁性和理论友好型生态系统能重新确立自我监督预训练作为AI研究核心支柱的地位（GitHub仓库：<git@github.com:rbalestr-lab/lejepa.git>）。 

---
# Introducing A Bangla Sentence - Gloss Pair Dataset for Bangla Sign Language Translation and Research 

**Title (ZH)**: 引入一种孟加拉语句子- gloss 对数据集，用于孟加拉语手语翻译与研究 

**Authors**: Neelavro Saha, Rafi Shahriyar, Nafis Ashraf Roudra, Saadman Sakib, Annajiat Alim Rasel  

**Link**: [PDF](https://arxiv.org/pdf/2511.08507)  

**Abstract**: Bangla Sign Language (BdSL) translation represents a low-resource NLP task due to the lack of large-scale datasets that address sentence-level translation. Correspondingly, existing research in this field has been limited to word and alphabet level detection. In this work, we introduce Bangla-SGP, a novel parallel dataset consisting of 1,000 human-annotated sentence-gloss pairs which was augmented with around 3,000 synthetically generated pairs using syntactic and morphological rules through a rule-based Retrieval-Augmented Generation (RAG) pipeline. The gloss sequences of the spoken Bangla sentences are made up of individual glosses which are Bangla sign supported words and serve as an intermediate representation for a continuous sign. Our dataset consists of 1000 high quality Bangla sentences that are manually annotated into a gloss sequence by a professional signer. The augmentation process incorporates rule-based linguistic strategies and prompt engineering techniques that we have adopted by critically analyzing our human annotated sentence-gloss pairs and by working closely with our professional signer. Furthermore, we fine-tune several transformer-based models such as mBart50, Google mT5, GPT4.1-nano and evaluate their sentence-to-gloss translation performance using BLEU scores, based on these evaluation metrics we compare the model's gloss-translation consistency across our dataset and the RWTH-PHOENIX-2014T benchmark. 

**Abstract (ZH)**: Bangla手语（BdSL）翻译代表了一个低资源NLP任务，由于缺乏针对句级翻译的大规模数据集。相应地，该领域的现有研究仅限于单词和字母级别的检测。在这项工作中，我们介绍了Bangla-SGP，这是一个新颖的平行数据集，包含1000个人工标注的句子-手语对照对，并通过基于规则的检索增强生成（RAG）管道生成了约3000个合成对照对，使用了句法和形态规则。所提到的手语序列由单个手语支持词汇组成，作为连续手语的中间表示。我们的数据集包括1000个高质量的孟加拉语句子，由专业手语使用者手工标注成手语序列。扩增过程结合了我们通过仔细分析人工标注的句子-手语对照对以及与专业手语使用者紧密合作所采用的基于规则的语言策略和提示工程技术。此外，我们微调了多个基于变换器的模型，如mBart50、Google mT5、GPT4.1-nano，并使用BLEU分数评估它们的句到手语翻译性能。基于这些评估指标，我们比较了模型的手语翻译一致性，包括在我们的数据集和RWTH-PHOENIX-2014T基准上的表现。 

---
# HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios 

**Title (ZH)**: HQ-SVC：面向低资源场景的高质量零样本唱歌语音转换 

**Authors**: Bingsong Bai, Yizhong Geng, Fengping Wang, Cong Wang, Puyuan Guo, Yingming Gao, Ya Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.08496)  

**Abstract**: Zero-shot singing voice conversion (SVC) transforms a source singer's timbre to an unseen target speaker's voice while preserving melodic content without fine-tuning. Existing methods model speaker timbre and vocal content separately, losing essential acoustic information that degrades output quality while requiring significant computational resources. To overcome these limitations, we propose HQ-SVC, an efficient framework for high-quality zero-shot SVC. HQ-SVC first extracts jointly content and speaker features using a decoupled codec. It then enhances fidelity through pitch and volume modeling, preserving critical acoustic information typically lost in separate modeling approaches, and progressively refines outputs via differentiable signal processing and diffusion techniques. Evaluations confirm HQ-SVC significantly outperforms state-of-the-art zero-shot SVC methods in conversion quality and efficiency. Beyond voice conversion, HQ-SVC achieves superior voice naturalness compared to specialized audio super-resolution methods while natively supporting voice super-resolution tasks. 

**Abstract (ZH)**: 高保真零样本唱歌声音转换(HQ-SVC) 

---
# Binary Split Categorical feature with Mean Absolute Error Criteria in CART 

**Title (ZH)**: 基于均绝对误差准则的二元分裂分类特征在CART中的应用 

**Authors**: Peng Yu, Yike Chen, Chao Xu, Albert Bifet, Jesse Read  

**Link**: [PDF](https://arxiv.org/pdf/2511.08470)  

**Abstract**: In the context of the Classification and Regression Trees (CART) algorithm, the efficient splitting of categorical features using standard criteria like GINI and Entropy is well-established. However, using the Mean Absolute Error (MAE) criterion for categorical features has traditionally relied on various numerical encoding methods. This paper demonstrates that unsupervised numerical encoding methods are not viable for the MAE criteria. Furthermore, we present a novel and efficient splitting algorithm that addresses the challenges of handling categorical features with the MAE criterion. Our findings underscore the limitations of existing approaches and offer a promising solution to enhance the handling of categorical data in CART algorithms. 

**Abstract (ZH)**: 在分类和回归树（CART）算法的背景下，使用GINI、熵等标准准则高效划分分类特征已经得到确立。然而，使用均绝对误差（MAE）准则对分类特征进行划分一直依赖于多种数值编码方法。本文证明了无监督的数值编码方法不适合MAE准则。此外，我们提出了一种新颖且高效的划分算法，以解决使用MAE准则处理分类特征的挑战。我们的研究突出了现有方法的局限性，并提供了一种增强CART算法中分类数据处理的有前景的解决方案。 

---
# Contrastive Integrated Gradients: A Feature Attribution-Based Method for Explaining Whole Slide Image Classification 

**Title (ZH)**: 对比集成梯度：一种基于特征 attribution 的Whole Slide图像分类解释方法 

**Authors**: Anh Mai Vu, Tuan L. Vo, Ngoc Lam Quang Bui, Nam Nguyen Le Binh, Akash Awasthi, Huy Quoc Vo, Thanh-Huy Nguyen, Zhu Han, Chandra Mohan, Hien Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.08464)  

**Abstract**: Interpretability is essential in Whole Slide Image (WSI) analysis for computational pathology, where understanding model predictions helps build trust in AI-assisted diagnostics. While Integrated Gradients (IG) and related attribution methods have shown promise, applying them directly to WSIs introduces challenges due to their high-resolution nature. These methods capture model decision patterns but may overlook class-discriminative signals that are crucial for distinguishing between tumor subtypes. In this work, we introduce Contrastive Integrated Gradients (CIG), a novel attribution method that enhances interpretability by computing contrastive gradients in logit space. First, CIG highlights class-discriminative regions by comparing feature importance relative to a reference class, offering sharper differentiation between tumor and non-tumor areas. Second, CIG satisfies the axioms of integrated attribution, ensuring consistency and theoretical soundness. Third, we propose two attribution quality metrics, MIL-AIC and MIL-SIC, which measure how predictive information and model confidence evolve with access to salient regions, particularly under weak supervision. We validate CIG across three datasets spanning distinct cancer types: CAMELYON16 (breast cancer metastasis in lymph nodes), TCGA-RCC (renal cell carcinoma), and TCGA-Lung (lung cancer). Experimental results demonstrate that CIG yields more informative attributions both quantitatively, using MIL-AIC and MIL-SIC, and qualitatively, through visualizations that align closely with ground truth tumor regions, underscoring its potential for interpretable and trustworthy WSI-based diagnostics 

**Abstract (ZH)**: Whole Slide Image分析中的可解释性对于计算病理学至关重要，通过理解模型预测来建立AI辅助诊断的信任。虽然集成梯度（IG）及相关归因方法显示出前景，但直接应用于高分辨率的WSIs时引入了挑战。这些方法捕捉模型决策模式，但可能忽略区分肿瘤亚型的关键分类信号。在此工作中，我们引入对比集成梯度（CIG），这是一种新型归因方法，通过计算logit空间中的对比梯度来增强可解释性。首先，CIG通过相对参考类比较特征重要性来突出显示分类区分区域，提供肿瘤和非肿瘤区域的更清晰区分。其次，CIG满足集成归因的公理，确保一致性和理论严密性。第三，我们提出两种归因质量指标MIL-AIC和MIL-SIC，衡量可用明显区域的预测信息和模型信心的变化情况，特别是在弱监督条件下。我们跨越三种不同癌症类型的三个数据集验证了CIG：CAMELYON16（淋巴结中的乳腺癌转移）、TCGA-RCC（肾细胞癌）和TCGA-Lung（肺癌）。实验结果表明，CIG在量化和定性方面均能提供更具有信息量的归因，视觉化结果与真实肿瘤区域高度一致，突显了其在可解释和可信WSI基诊断中的潜力。 

---
# Unifying Model and Layer Fusion for Speech Foundation Models 

**Title (ZH)**: 统一模型和层融合用于语音基础模型 

**Authors**: Yi-Jen Shih, David Harwath  

**Link**: [PDF](https://arxiv.org/pdf/2511.08389)  

**Abstract**: Speech Foundation Models have gained significant attention recently. Prior works have shown that the fusion of representations from multiple layers of the same model or the fusion of multiple models can improve performance on downstream tasks. We unify these two fusion strategies by proposing an interface module that enables fusion across multiple upstream speech models while integrating information across their layers. We conduct extensive experiments on different self-supervised and supervised models across various speech tasks, including ASR and paralinguistic analysis, and demonstrate that our method outperforms prior fusion approaches. We further analyze its scalability concerning model size and count, highlighting the importance of selecting appropriate upstream models. Our results show that the proposed interface provides an additional performance boost when given a suitable upstream model selection, making it a promising approach for utilizing Speech Foundation Models. 

**Abstract (ZH)**: Speech 基础模型近年来受到了广泛关注。先前的工作表明，同一模型的多层表示融合或多个模型的融合可以改善下游任务的性能。我们通过提出一个接口模块统一了这两种融合策略，该模块能够在多个上游语音模型之间进行融合，并整合它们的跨层信息。我们在不同自监督和监督模型的多种语音任务上进行了广泛的实验，包括自动语音识别(ASR)和副语言分析，并证明了我们的方法优于先前的融合方法。我们进一步分析了其在模型规模和数量方面的可扩展性，突出了选择适当上游模型的重要性。研究结果表明，在给定合适的上游模型选择时，提出的接口可以提供额外的性能提升，使其成为利用语音基础模型的一种有前景的方法。 

---
# Bid Farewell to Seesaw: Towards Accurate Long-tail Session-based Recommendation via Dual Constraints of Hybrid Intents 

**Title (ZH)**: 告别摇摆：通过混合意图的双重约束迈向精准长尾会话推荐 

**Authors**: Xiao Wang, Ke Qin, Dongyang Zhang, Xiurui Xie, Shuang Liang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08378)  

**Abstract**: Session-based recommendation (SBR) aims to predict anonymous users' next interaction based on their interaction sessions. In the practical recommendation scenario, low-exposure items constitute the majority of interactions, creating a long-tail distribution that severely compromises recommendation diversity. Existing approaches attempt to address this issue by promoting tail items but incur accuracy degradation, exhibiting a "see-saw" effect between long-tail and accuracy performance. We attribute such conflict to session-irrelevant noise within the tail items, which existing long-tail approaches fail to identify and constrain effectively. To resolve this fundamental conflict, we propose \textbf{HID} (\textbf{H}ybrid \textbf{I}ntent-based \textbf{D}ual Constraint Framework), a plug-and-play framework that transforms the conventional "see-saw" into "win-win" through introducing the hybrid intent-based dual constraints for both long-tail and accuracy. Two key innovations are incorporated in this framework: (i) \textit{Hybrid Intent Learning}, where we reformulate the intent extraction strategies by employing attribute-aware spectral clustering to reconstruct the item-to-intent mapping. Furthermore, discrimination of session-irrelevant noise is achieved through the assignment of the target and noise intents to each session. (ii) \textit{Intent Constraint Loss}, which incorporates two novel constraint paradigms regarding the \textit{diversity} and \textit{accuracy} to regulate the representation learning process of both items and sessions. These two objectives are unified into a single training loss through rigorous theoretical derivation. Extensive experiments across multiple SBR models and datasets demonstrate that HID can enhance both long-tail performance and recommendation accuracy, establishing new state-of-the-art performance in long-tail recommender systems. 

**Abstract (ZH)**: 基于会话的推荐（SBR）旨在根据用户的交互会话预测其下一次交互。在实际推荐场景中，低曝光项构成了主要的交互，形成了长尾分布，严重损害了推荐的多样性。现有方法试图通过促进尾部项来解决这一问题，但会导致准确率下降，表现出长尾和准确率之间的“跷跷板”效应。我们将这种冲突归因于尾部项中的会话无关噪声，现有长尾方法无法有效识别和约束这些噪声。为了解决这一根本冲突，我们提出了HID（混合意图双约束框架），这是一个即插即用框架，通过引入针对长尾和准确性的混合意图双约束，将传统的“跷跷板”效应转化为了“双赢”效应。该框架包含两个关键创新：（i）混合意图学习，通过使用属性感知谱聚类重新制定意图提取策略，重建项目到意图的映射，并通过为每个会话分配目标意图和噪声意图来实现会话无关噪声的区分；（ii）意图约束损失，该损失结合了关于多样性和准确性的两种新型约束范式，以规范项目和会话的表示学习过程。这两种目标通过严格的理论推导统一为一个训练损失。在多个SBR模型和数据集上的大量实验表明，HID可以提升长尾性能和推荐准确率，建立了长尾推荐系统的最新最佳性能。 

---
# Extreme Model Compression with Structured Sparsity at Low Precision 

**Title (ZH)**: 低精度下的结构化稀疏极端模型压缩 

**Authors**: Dan Liu, Nikita Dvornik, Xue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.08360)  

**Abstract**: Deep neural networks (DNNs) are used in many applications, but their large size and high computational cost make them hard to run on devices with limited resources. Two widely used techniques to address this challenge are weight quantization, which lowers the precision of all weights, and structured sparsity, which removes unimportant weights while retaining the important ones at full precision. Although both are effective individually, they are typically studied in isolation due to their compounded negative impact on model accuracy when combined. In this work, we introduce SLOPE Structured Sparsity at Low Precision), a unified framework, to effectively combine structured sparsity and low-bit quantization in a principled way. We show that naively combining sparsity and quantization severely harms performance due to the compounded impact of both techniques. To address this, we propose a training-time regularization strategy that minimizes the discrepancy between full-precision weights and their sparse, quantized counterparts by promoting angular alignment rather than direct matching. On ResNet-18, SLOPE achieves $\sim20\times$ model size reduction while retaining $\sim$99% of the original accuracy. It consistently outperforms state-of-the-art quantization and structured sparsity methods across classification, detection, and segmentation tasks on models such as ResNet-18, ViT-Small, and Mask R-CNN. 

**Abstract (ZH)**: 低精度SLOPE结构稀疏性方法 

---
# Hybrid Quantum-Classical Selective State Space Artificial Intelligence 

**Title (ZH)**: 混合量子经典选择性状态空间人工智能 

**Authors**: Amin Ebrahimi, Farzan Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2511.08349)  

**Abstract**: Hybrid Quantum Classical (HQC) algorithms constitute one of the most effective paradigms for exploiting the computational advantages of quantum systems in large-scale numerical tasks. By operating in high-dimensional Hilbert spaces, quantum circuits enable exponential speed-ups and provide access to richer representations of cost landscapes compared to purely classical methods. These capabilities are particularly relevant for machine learning, where state-of-the-art models especially in Natural Language Processing (NLP) suffer from prohibitive time complexity due to massive matrix multiplications and high-dimensional optimization.
In this manuscript, we propose a Hybrid Quantum Classical selection mechanism for the Mamba architecture, designed specifically for temporal sequence classification problems. Our approach leverages Variational Quantum Circuits (VQCs) as quantum gating modules that both enhance feature extraction and improve suppression of irrelevant information. This integration directly addresses the computational bottlenecks of deep learning architectures by exploiting quantum resources for more efficient representation learning.
We analyze how introducing quantum subroutines into large language models (LLMs) impacts their generalization capability, expressivity, and parameter efficiency. The results highlight the potential of quantum-enhanced gating mechanisms as a path toward scalable, resource-efficient NLP models, in a limited simulation step. Within the first four epochs on a reshaped MNIST dataset with input format (batch, 784, d_model), our hybrid model achieved 24.6% accuracy while using one quantum layer and achieve higher expressivity, compared to 21.6% obtained by a purely classical selection mechanism. we state No founding 

**Abstract (ZH)**: 混合量子经典(Hybrid Quantum Classical)算法构成了在大规模数值任务中充分利用量子系统计算优势的一种最有效范式。通过在高维希尔伯特空间中操作，量子电路能够实现指数级加速，并提供与纯经典方法相比更丰富的成本景观表示。这些能力特别适用于机器学习，特别是在自然语言处理(NLP)领域，最先进的模型由于大规模矩阵乘法和高维优化而面临难以承受的时间复杂度问题。

在这篇手稿中，我们为Mamba架构提出了一个混合量子经典选择机制，专门用于时间序列分类问题。我们的方法利用变量子电路(Variational Quantum Circuits, VQCs)作为量子门模块，既能增强特征提取，又能抑制无关信息。这种集成直接通过利用量子资源来应对深度学习架构中的计算瓶颈，实现更有效的表示学习。

我们分析了将量子子程序引入大型语言模型(Large Language Models, LLMs)如何影响其泛化能力、表达能力和参数效率。结果突显了量子增强门机制作为实现可扩展、资源高效NLP模型途径的潜力，在少量模拟步骤内得以验证。在重塑后的MNIST数据集上，输入格式为(batch, 784, d_model)，在前四个时期内，我们的混合模型使用一个量子层实现了24.6%的准确率，相比纯经典选择机制获得的21.6%有了更高的表达性。 

---
# Towards Open-Set Myoelectric Gesture Recognition via Dual-Perspective Inconsistency Learning 

**Title (ZH)**: 面向开放集肌电手势识别的双视角不一致性学习 

**Authors**: Chen Liu, Can Han, Weishi Xu, Yaqi Wang, Dahong Qian  

**Link**: [PDF](https://arxiv.org/pdf/2511.08344)  

**Abstract**: Surface electromyography (sEMG)-based gesture recognition plays a critical role in human-machine interaction (HMI), particularly for rehabilitation and prosthetic control. However, sEMG-based systems often suffer from the scarcity of informative training data, leading to overfitting and poor generalization in deep learning models. Data augmentation offers a promising approach to increasing the size and diversity of training data, where faithfulness and diversity are two critical factors to effectiveness. However, promoting untargeted diversity can result in redundant samples with limited utility. To address these challenges, we propose a novel diffusion-based data augmentation approach, Sparse-Aware Semantic-Guided Diffusion Augmentation (SASG-DA). To enhance generation faithfulness, we introduce the Semantic Representation Guidance (SRG) mechanism by leveraging fine-grained, task-aware semantic representations as generation conditions. To enable flexible and diverse sample generation, we propose a Gaussian Modeling Semantic Modeling (GMSS) strategy, which models the semantic representation distribution and allows stochastic sampling to produce both faithful and diverse samples. To enhance targeted diversity, we further introduce a Sparse-Aware Semantic Sampling strategy to explicitly explore underrepresented regions, improving distribution coverage and sample utility. Extensive experiments on benchmark sEMG datasets, Ninapro DB2, DB4, and DB7, demonstrate that SASG-DA significantly outperforms existing augmentation methods. Overall, our proposed data augmentation approach effectively mitigates overfitting and improves recognition performance and generalization by offering both faithful and diverse samples. 

**Abstract (ZH)**: 基于表面肌电图（sEMG）的手势识别在人机交互（HMI）中起着关键作用，尤其是在康复和假肢控制领域。然而，基于sEMG的系统常常受到信息性训练数据稀缺的困扰，导致深度学习模型过拟合和泛化能力差。数据扩增提供了一种增加训练数据量和多样性有前景的方法，其中忠实性和多样性是 effectiveness 的关键因素。然而，促进未瞄准的多样性可能导致冗余样本，这些样本的实用性有限。为了解决这些挑战，我们提出了一种新颖的基于扩散的数据扩增方法，即稀疏意识语义引导扩增（SASG-DA）。为了增强生成忠实性，我们通过利用细粒度的任务意识语义表示作为生成条件引入了语义表示指导（SRG）机制。为了实现灵活和多样的样本生成，我们提出了高斯建模语义建模（GMSS）策略，该策略建模了语义表示分布并允许随机采样以生成忠实和多样化的样本。为了增强针对性多样性，我们进一步引入了稀疏意识语义采样策略，明确探索欠代表区域，从而提高分布覆盖率和样本实用性。在基准sEMG数据集Ninapro DB2、DB4和DB7上的广泛实验表明，SASG-DA 显著优于现有扩增方法。总的来说，我们提出的数据扩增方法通过提供忠实和多样的样本有效地缓解了过拟合并提高了识别性能和泛化能力。 

---
# HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting 

**Title (ZH)**: 基于超网络的多变量时间序列预测 

**Authors**: Andrey Savchenko, Oleg Kachan  

**Link**: [PDF](https://arxiv.org/pdf/2511.08340)  

**Abstract**: Accurate forecasting of multivariate time series data remains a formidable challenge, particularly due to the growing complexity of temporal dependencies in real-world scenarios. While neural network-based models have achieved notable success in this domain, complex channel-dependent models often suffer from performance degradation compared to channel-independent models that do not consider the relationship between components but provide high robustness due to small capacity. In this work, we propose HN-MVTS, a novel architecture that integrates a hypernetwork-based generative prior with an arbitrary neural network forecasting model. The input of this hypernetwork is a learnable embedding matrix of time series components. To restrict the number of new parameters, the hypernetwork learns to generate the weights of the last layer of the target forecasting networks, serving as a data-adaptive regularizer that improves generalization and long-range predictive accuracy. The hypernetwork is used only during the training, so it does not increase the inference time compared to the base forecasting model. Extensive experiments on eight benchmark datasets demonstrate that application of HN-MVTS to the state-of-the-art models (DLinear, PatchTST, TSMixer, etc.) typically improves their performance. Our findings suggest that hypernetwork-driven parameterization offers a promising direction for enhancing existing forecasting techniques in complex scenarios. 

**Abstract (ZH)**: 基于超网络的多变量时间序列数据准确预测仍是一项艰巨的挑战，特别是在现实场景中时间依赖关系日益复杂的情况下。虽然基于神经网络的模型在这个领域取得了显著成功，但依赖通道的复杂模型在性能上通常不如不考虑组件间关系但具有较小容量因此更具鲁棒性的独立通道模型。在本文中，我们提出了一种名为HN-MVTS的新架构，该架构将基于超网络的生成先验与任意神经网络预测模型相结合。超网络的输入是一个可学习的时间序列组件嵌入矩阵。为了限制新参数的数量，超网络学习生成目标预测网络最后一层的权重，作为数据自适应正则化器，提高泛化能力和长期预测准确性。超网络仅在训练期间使用，因此不会增加与基础预测模型相比的推理时间。在八个基准数据集上的 extensive 实验表明，将 HN-MVTS 应用于最先进的模型（DLinear、PatchTST、TSMixer 等）通常会提高其性能。我们的研究结果表明，超网络驱动的参数化为在复杂场景下增强现有预测技术提供了有前景的方向。 

---
# Improving the accuracy and generalizability of molecular property regression models with a substructure-substitution-rule-informed framework 

**Title (ZH)**: 基于子结构替代规则指引的框架以提高分子性质回归模型的准确性和普适性 

**Authors**: Xiaoyu Fan, Lin Guo, Ruizhen Jia, Yang Tian, Zhihao Yang, Boxue Tian  

**Link**: [PDF](https://arxiv.org/pdf/2511.08314)  

**Abstract**: Artificial Intelligence (AI)-aided drug discovery is an active research field, yet AI models often exhibit poor accuracy in regression tasks for molecular property prediction, and perform catastrophically poorly for out-of-distribution (OOD) molecules. Here, we present MolRuleLoss, a substructure-substitution-rule-informed framework that improves the accuracy and generalizability of multiple molecular property regression models (MPRMs) such as GEM and UniMol for diverse molecular property prediction tasks. MolRuleLoss incorporates partial derivative constraints for substructure substitution rules (SSRs) into an MPRM's loss function. When using GEM models for predicting lipophilicity, water solubility, and solvation-free energy (using lipophilicity, ESOL, and freeSolv datasets from MoleculeNet), the root mean squared error (RMSE) values with and without MolRuleLoss were 0.587 vs. 0.660, 0.777 vs. 0.798, and 1.252 vs. 1.877, respectively, representing 2.6-33.3% performance improvements. We show that both the number and the quality of SSRs contribute to the magnitude of prediction accuracy gains obtained upon adding MolRuleLoss to an MPRM. MolRuleLoss improved the generalizability of MPRMs for "activity cliff" molecules in a lipophilicity prediction task and improved the generalizability of MPRMs for OOD molecules in a melting point prediction task. In a molecular weight prediction task for OOD molecules, MolRuleLoss reduced the RMSE value of a GEM model from 29.507 to 0.007. We also provide a formal demonstration that the upper bound of the variation for property change of SSRs is positively correlated with an MPRM's error. Together, we show that using the MolRuleLoss framework as a bolt-on boosts the prediction accuracy and generalizability of multiple MPRMs, supporting diverse applications in areas like cheminformatics and AI-aided drug discovery. 

**Abstract (ZH)**: AI辅助的药物发现是活跃的研究领域，但AI模型在分子性质预测的回归任务中往往表现出较差的准确性，并且对于分布外（OOD）分子表现极为糟糕。我们提出了一种名为MolRuleLoss的框架，这是一种基于亚结构替代规则的信息框架，可以提高多种分子性质回归模型（MPRM）如GEM和UniMol在多种分子性质预测任务中的准确性和泛化能力。MolRuleLoss将亚结构替代规则（SSR）的偏导数约束纳入到MPRM的损失函数中。在使用GEM模型预测脂溶性、水溶性和解溶热自由能（使用MoleculeNet的lipophilicity、ESOL和freeSolv数据集）时，加入MolRuleLoss后的均方根误差（RMSE）值分别为0.587 vs. 0.660、0.777 vs. 0.798和1.252 vs. 1.877，分别代表2.6%-33.3%的性能提升。我们展示了SSR的数量和质量都对MPRM加入MolRuleLoss后的预测准确性的提升有贡献。MolRuleLoss提高了MPRM在脂溶性预测任务中对“活性陡坡”分子的泛化能力，并在熔点预测任务中提高了对分布外（OOD）分子的泛化能力。在分布外分子的分子量预测任务中，MolRuleLoss将GEM模型的RMSE值从29.507降低到了0.007。我们还提供了一个形式化证明，表明SSR属性变化的变异上界与MPRM的误差正相关。总之，我们展示了使用MolRuleLoss框架作为增强手段可以提高多种MPRM的预测准确性和泛化能力，支持包括计算化学和AI辅助药物发现在内的多种应用。 

---
# Test-time Diverse Reasoning by Riemannian Activation Steering 

**Title (ZH)**: 基于黎曼激活引导的测试时多样化推理 

**Authors**: Ly Tran Ho Khanh, Dongxuan Zhu, Man-Chung Yue, Viet Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.08305)  

**Abstract**: Best-of-$N$ reasoning improves the accuracy of language models in solving complex tasks by sampling multiple candidate solutions and then selecting the best one based on some criteria. A critical bottleneck for this strategy is the output diversity limit, which occurs when the model generates similar outputs despite stochastic sampling, and hence recites the same error. To address this lack of variance in reasoning paths, we propose a novel unsupervised activation steering strategy that simultaneously optimizes the steering vectors for multiple reasoning trajectories at test time. At any synchronization anchor along the batch generation process, we find the steering vectors that maximize the total volume spanned by all possible intervened activation subsets. We demonstrate that these steering vectors can be determined by solving a Riemannian optimization problem over the product of spheres with a log-determinant objective function. We then use a Riemannian block-coordinate descent algorithm with a well-tuned learning rate to obtain a stationary point of the problem, and we apply these steering vectors until the generation process reaches the subsequent synchronization anchor. Empirical evaluations on popular mathematical benchmarks demonstrate that our test-time Riemannian activation steering strategy outperforms vanilla sampling techniques in terms of generative diversity and solution accuracy. 

**Abstract (ZH)**: Best-of-$N$ 策略通过采样多个候选解决方案并根据某些标准选择最佳一个，提高了语言模型在解决复杂任务时的准确性。这一策略的关键瓶颈是输出多样性限制，当模型在随机采样过程中生成相似输出时，就会出现这一现象，从而重复相同的错误。为解决这一推理路径中的缺乏变异性问题，我们提出了一种新的无监督激活导向策略，在测试时同时优化多个推理轨迹的导向向量。在整个批次生成过程中，在任何同步锚点处，我们找到最大化所有可能干预激活子集所覆盖的总体积的导向向量。我们证明这些导向向量可以通过在球积上求解具有对数行列式目标函数的黎曼优化问题来确定。然后，我们使用精心调优的学习率的黎曼块坐标下降算法来获得该问题的稳定点，并应用这些导向向量直到生成过程达到下一个同步锚点。在流行的数学基准上的实证评估表明，我们的测试时黎曼激活导向策略在生成多样性和解的准确性方面优于vanilla采样技术。 

---
# Dual-Kernel Graph Community Contrastive Learning 

**Title (ZH)**: 双内核图社区对比学习 

**Authors**: Xiang Chen, Kun Yue, Wenjie Liu, Zhenyu Zhang, Liang Duan  

**Link**: [PDF](https://arxiv.org/pdf/2511.08287)  

**Abstract**: Graph Contrastive Learning (GCL) has emerged as a powerful paradigm for training Graph Neural Networks (GNNs) in the absence of task-specific labels. However, its scalability on large-scale graphs is hindered by the intensive message passing mechanism of GNN and the quadratic computational complexity of contrastive loss over positive and negative node pairs. To address these issues, we propose an efficient GCL framework that transforms the input graph into a compact network of interconnected node sets while preserving structural information across communities. We firstly introduce a kernelized graph community contrastive loss with linear complexity, enabling effective information transfer among node sets to capture hierarchical structural information of the graph. We then incorporate a knowledge distillation technique into the decoupled GNN architecture to accelerate inference while maintaining strong generalization performance. Extensive experiments on sixteen real-world datasets of varying scales demonstrate that our method outperforms state-of-the-art GCL baselines in both effectiveness and scalability. 

**Abstract (ZH)**: 基于图的对比学习（GCL）已成为在缺乏任务特定标签的情况下训练图神经网络（GNNs）的一种强大范式。然而，GNN密集的信息传递机制和对比损失在正负节点对上的二次计算复杂性限制了其在大规模图上的扩展性。为了解决这些问题，我们提出了一种高效的GCL框架，将输入图转换为紧凑的节点集合互联系统，同时保持社区间的结构信息。我们首先引入了一种核化图社区对比损失，其线性复杂度使得节点集之间能够有效传递信息，捕获图的分层结构信息。然后，我们将知识蒸馏技术融入解耦的GNN架构中，以加速推理过程并保持强大的泛化性能。在十六个不同规模的真实世界数据集上的广泛实验表明，我们的方法在有效性和扩展性方面均优于最先进的GCL基线方法。 

---
# Bi-Objective Evolutionary Optimization for Large-Scale Open Pit Mine Scheduling Problem under Uncertainty with Chance Constraints 

**Title (ZH)**: 带机会约束的不确定性大型露天矿调度双目标进化优化 

**Authors**: Ishara Hewa Pathiranage, Aneta Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2511.08275)  

**Abstract**: The open-pit mine scheduling problem (OPMSP) is a complex, computationally expensive process in long-term mine planning, constrained by operational and geological dependencies. Traditional deterministic approaches often ignore geological uncertainty, leading to suboptimal and potentially infeasible production schedules. Chance constraints allow modeling of stochastic components by ensuring probabilistic constraints are satisfied with high probability. This paper presents a bi-objective formulation of the OPMSP that simultaneously maximizes expected net present value and minimizes scheduling risk, independent of the confidence level required for the constraint. Solutions are represented using integer encoding, inherently satisfying reserve constraints. We introduce a domain-specific greedy randomized initialization and a precedence-aware period-swap mutation operator. We integrate these operators into three multi-objective evolutionary algorithms: the global simple evolutionary multi-objective optimizer (GSEMO), a mutation-only variant of multi-objective evolutionary algorithm based on decomposition (MOEA/D), and non-dominated sorting genetic algorithm II (NSGA-II). We compare our bi-objective formulation against the single-objective approach, which depends on a specific confidence level, by analyzing mine deposits consisting of up to 112 687 blocks. Results demonstrate that the proposed bi-objective formulation yields more robust and balanced trade-offs between economic value and risk compared to single-objective, confidence-dependent approach. 

**Abstract (ZH)**: 露天矿场调度问题（OPMSP）在长期矿场规划中是一个复杂且计算成本高的过程，受到操作和地质依赖性的限制。传统确定性方法通常忽略地质不确定性，导致不最优甚至不可行的生产计划。机会约束通过确保约束以高概率被满足来建模随机成分。本文提出了一种双目标露天矿场调度问题（OPMSP）的建模，同时最大化预期净现值并最小化调度风险，与所需的置信水平无关。解决方案使用整数编码表示，内在满足储量约束。我们引入了领域特定的贪婪随机初始化和考虑先行关系的周期交换变异操作符。我们将这些操作符整合到三种多目标进化算法中：全局简单多目标进化优化器（GSEMO）、基于分解的多目标进化算法（MOEA/D）的仅变异版本以及非支配排序遗传算法II（NSGA-II）。通过分析由多达112,687个块组成的矿藏，我们将提出的双目标建模与依赖特定置信水平的单目标方法进行了比较。结果表明，提出的双目标建模相比依赖置信水平的单目标方法，提供了更稳健且平衡的经济价值与风险之间的权衡。 

---
# NERVE: Neighbourhood & Entropy-guided Random-walk for training free open-Vocabulary sEgmentation 

**Title (ZH)**: NERVE: 基于邻里和熵引导的随机漫步无训练集开放词汇分割 

**Authors**: Kunal Mahatha, Jose Dolz, Christian Desrosiers  

**Link**: [PDF](https://arxiv.org/pdf/2511.08248)  

**Abstract**: Despite recent advances in Open-Vocabulary Semantic Segmentation (OVSS), existing training-free methods face several limitations: use of computationally expensive affinity refinement strategies, ineffective fusion of transformer attention maps due to equal weighting or reliance on fixed-size Gaussian kernels to reinforce local spatial smoothness, enforcing isotropic neighborhoods. We propose a strong baseline for training-free OVSS termed as NERVE (Neighbourhood \& Entropy-guided Random-walk for open-Vocabulary sEgmentation), which uniquely integrates global and fine-grained local information, exploiting the neighbourhood structure from the self-attention layer of a stable diffusion model. We also introduce a stochastic random walk for refining the affinity rather than relying on fixed-size Gaussian kernels for local context. This spatial diffusion process encourages propagation across connected and semantically related areas, enabling it to effectively delineate objects with arbitrary shapes. Whereas most existing approaches treat self-attention maps from different transformer heads or layers equally, our method uses entropy-based uncertainty to select the most relevant maps. Notably, our method does not require any conventional post-processing techniques like Conditional Random Fields (CRF) or Pixel-Adaptive Mask Refinement (PAMR). Experiments are performed on 7 popular semantic segmentation benchmarks, yielding an overall state-of-the-art zero-shot segmentation performance, providing an effective approach to open-vocabulary semantic segmentation. 

**Abstract (ZH)**: 邻域与熵guidance随机游walk驱动的开放词汇语义分割（NERVE）：一种无训练的强基准 

---
# FedPoP: Federated Learning Meets Proof of Participation 

**Title (ZH)**: FedPoP：联邦学习与参与证明的结合 

**Authors**: Devriş İşler, Elina van Kempen, Seoyeon Hwang, Nikolaos Laoutaris  

**Link**: [PDF](https://arxiv.org/pdf/2511.08207)  

**Abstract**: Federated learning (FL) offers privacy preserving, distributed machine learning, allowing clients to contribute to a global model without revealing their local data. As models increasingly serve as monetizable digital assets, the ability to prove participation in their training becomes essential for establishing ownership. In this paper, we address this emerging need by introducing FedPoP, a novel FL framework that allows nonlinkable proof of participation while preserving client anonymity and privacy without requiring either extensive computations or a public ledger. FedPoP is designed to seamlessly integrate with existing secure aggregation protocols to ensure compatibility with real-world FL deployments. We provide a proof of concept implementation and an empirical evaluation under realistic client dropouts. In our prototype, FedPoP introduces 0.97 seconds of per-round overhead atop securely aggregated FL and enables a client to prove its participation/contribution to a model held by a third party in 0.0612 seconds. These results indicate FedPoP is practical for real-world deployments that require auditable participation without sacrificing privacy. 

**Abstract (ZH)**: 联邦学习（FL）提供了隐私保护的分布式机器学习，允许客户端在不泄露本地数据的情况下 contribute 到全球模型中。随着模型逐渐成为可变现的数字资产，证明其训练参与度的能力对于确立所有权至关重要。本文通过引入 FedPoP 这一新型FL框架来应对这一新兴需求，该框架允许在保护客户端匿名性和隐私的同时提供非可链接的参与证明，无需进行繁重的计算或使用公共账本。FedPoP 经设计以无缝集成现有的安全聚合协议，确保与实际部署的FL环境兼容。我们提供了概念验证实现并在现实的客户端退出场景下进行了实证评估。在我们的原型中，FedPoP 在安全聚合的FL之上每轮增加0.97秒的开销，并使客户端能在0.0612秒内证明其对第三方持有的模型的参与/贡献。这些结果表明，FedPoP 在不牺牲隐私的前提下，适用于需要可审计参与度的实际部署环境。 

---
# Deep (Predictive) Discounted Counterfactual Regret Minimization 

**Title (ZH)**: 深度（预测性）折扣反事实遗憾最小化 

**Authors**: Hang Xu, Kai Li, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.08174)  

**Abstract**: Counterfactual regret minimization (CFR) is a family of algorithms for effectively solving imperfect-information games. To enhance CFR's applicability in large games, researchers use neural networks to approximate its behavior. However, existing methods are mainly based on vanilla CFR and struggle to effectively integrate more advanced CFR variants. In this work, we propose an efficient model-free neural CFR algorithm, overcoming the limitations of existing methods in approximating advanced CFR variants. At each iteration, it collects variance-reduced sampled advantages based on a value network, fits cumulative advantages by bootstrapping, and applies discounting and clipping operations to simulate the update mechanisms of advanced CFR variants. Experimental results show that, compared with model-free neural algorithms, it exhibits faster convergence in typical imperfect-information games and demonstrates stronger adversarial performance in a large poker game. 

**Abstract (ZH)**: 基于值网络的高效模型自由神经CFR算法：克服高级CFR变体逼近的限制并展示更快的收敛性和更强的对抗性能 

---
# ProbSelect: Stochastic Client Selection for GPU-Accelerated Compute Devices in the 3D Continuum 

**Title (ZH)**: ProbSelect: 基于概率的客户端选择方法在3D连续体中加速GPU加速计算设备 

**Authors**: Andrija Stanisic, Stefan Nastic  

**Link**: [PDF](https://arxiv.org/pdf/2511.08147)  

**Abstract**: Integration of edge, cloud and space devices into a unified 3D continuum imposes significant challenges for client selection in federated learning systems. Traditional approaches rely on continuous monitoring and historical data collection, which becomes impractical in dynamic environments where satellites and mobile devices frequently change operational conditions. Furthermore, existing solutions primarily consider CPU-based computation, failing to capture complex characteristics of GPU-accelerated training that is prevalent across the 3D continuum. This paper introduces ProbSelect, a novel approach utilizing analytical modeling and probabilistic forecasting for client selection on GPU-accelerated devices, without requiring historical data or continuous monitoring. We model client selection within user-defined SLOs. Extensive evaluation across diverse GPU architectures and workloads demonstrates that ProbSelect improves SLO compliance by 13.77% on average while achieving 72.5% computational waste reduction compared to baseline approaches. 

**Abstract (ZH)**: 将边缘、云和太空设备整合到统一的3D连贯体中，对联邦学习系统的客户端选择提出了重大挑战。传统方法依赖连续监控和历史数据收集，在卫星和移动设备频繁改变运行状态的动态环境中变得不切实际。此外，现有的解决方案主要考虑基于CPU的计算，未能捕捉到在整个3D连贯体中普遍存在的GPU加速训练的复杂特性。本文介绍了一种名为ProbSelect的新型方法，该方法利用分析建模和概率预测来选择GPU加速设备上的客户端，无需历史数据或连续监控。我们在多种GPU架构和工作负载上进行了广泛的评估，结果显示，ProbSelect在平均提高了13.77%的服务水平目标(SLO)合规率的同时，实现了72.5%的计算浪费减少，相比基准方法性能显著提高。 

---
# SafeMIL: Learning Offline Safe Imitation Policy from Non-Preferred Trajectories 

**Title (ZH)**: SafeMIL：从非首选轨迹中学习 Offline 安全模仿策略 

**Authors**: Returaj Burnwal, Nirav Pravinbhai Bhatt, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2511.08136)  

**Abstract**: In this work, we study the problem of offline safe imitation learning (IL). In many real-world settings, online interactions can be risky, and accurately specifying the reward and the safety cost information at each timestep can be difficult. However, it is often feasible to collect trajectories reflecting undesirable or risky behavior, implicitly conveying the behavior the agent should avoid. We refer to these trajectories as non-preferred trajectories. Unlike standard IL, which aims to mimic demonstrations, our agent must also learn to avoid risky behavior using non-preferred trajectories. In this paper, we propose a novel approach, SafeMIL, to learn a parameterized cost that predicts if the state-action pair is risky via \textit{Multiple Instance Learning}. The learned cost is then used to avoid non-preferred behaviors, resulting in a policy that prioritizes safety. We empirically demonstrate that our approach can learn a safer policy that satisfies cost constraints without degrading the reward performance, thereby outperforming several baselines. 

**Abstract (ZH)**: 在线下安全imitation learning中的问题研究：SafeMIL方法及其应用 

---
# A robust methodology for long-term sustainability evaluation of Machine Learning models 

**Title (ZH)**: 一种用于机器学习模型长期可持续性评估的稳健方法论 

**Authors**: Jorge Paz-Ruza, João Gama, Amparo Alonso-Betanzos, Bertha Guijarro-Berdiñas  

**Link**: [PDF](https://arxiv.org/pdf/2511.08120)  

**Abstract**: Sustainability and efficiency have become essential considerations in the development and deployment of Artificial Intelligence systems, yet existing regulatory and reporting practices lack standardized, model-agnostic evaluation protocols. Current assessments often measure only short-term experimental resource usage and disproportionately emphasize batch learning settings, failing to reflect real-world, long-term AI lifecycles. In this work, we propose a comprehensive evaluation protocol for assessing the long-term sustainability of ML models, applicable to both batch and streaming learning scenarios. Through experiments on diverse classification tasks using a range of model types, we demonstrate that traditional static train-test evaluations do not reliably capture sustainability under evolving data and repeated model updates. Our results show that long-term sustainability varies significantly across models, and in many cases, higher environmental cost yields little performance benefit. 

**Abstract (ZH)**: 人工智能系统的长期可持续性和效率已成为发展和部署中的关键考量，现有监管和报告实践缺乏标准化且模型无关的评估协议。当前的评估通常仅衡量短期实验资源使用情况，并过度强调批量学习设置，未能反映真实世界的长期AI生命周期。在本工作中，我们提出了一种全面的评估协议，用于评估ML模型的长期可持续性，该协议适用于批量学习和流式学习场景。通过使用多种模型类型对多样化的分类任务进行实验，我们展示了传统静态训练-测试评估在数据演变和反复模型更新的情况下并不可靠地捕捉可持续性。我们的结果显示，模型之间的长期可持续性差异显著，并且在许多情况下，更高的环境代价几乎没有性能改进。 

---
# BARD10: A New Benchmark Reveals Significance of Bangla Stop-Words in Authorship Attribution 

**Title (ZH)**: BARD10: 一个新的基准揭示孟加拉停用词在作者ship归属中的重要性 

**Authors**: Abdullah Muhammad Moosa, Nusrat Sultana, Mahdi Muhammad Moosa, Md. Miraiz Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2511.08085)  

**Abstract**: This research presents a comprehensive investigation into Bangla authorship attribution, introducing a new balanced benchmark corpus BARD10 (Bangla Authorship Recognition Dataset of 10 authors) and systematically analyzing the impact of stop-word removal across classical and deep learning models to uncover the stylistic significance of Bangla stop-words. BARD10 is a curated corpus of Bangla blog and opinion prose from ten contemporary authors, alongside the methodical assessment of four representative classifiers: SVM (Support Vector Machine), Bangla BERT (Bidirectional Encoder Representations from Transformers), XGBoost, and a MLP (Multilayer Perception), utilizing uniform preprocessing on both BARD10 and the benchmark corpora BAAD16 (Bangla Authorship Attribution Dataset of 16 authors). In all datasets, the classical TF-IDF + SVM baseline outperformed, attaining a macro-F1 score of 0.997 on BAAD16 and 0.921 on BARD10, while Bangla BERT lagged by as much as five points. This study reveals that BARD10 authors are highly sensitive to stop-word pruning, while BAAD16 authors remain comparatively robust highlighting genre-dependent reliance on stop-word signatures. Error analysis revealed that high frequency components transmit authorial signatures that are diminished or reduced by transformer models. Three insights are identified: Bangla stop-words serve as essential stylistic indicators; finely calibrated ML models prove effective within short-text limitations; and BARD10 connects formal literature with contemporary web dialogue, offering a reproducible benchmark for future long-context or domain-adapted transformers. 

**Abstract (ZH)**: Bangla作者识别研究：BARD10数据集的引入与停用词去除影响的系统分析 

---
# Hierarchical Structure-Property Alignment for Data-Efficient Molecular Generation and Editing 

**Title (ZH)**: 数据高效分子生成与编辑的层级结构-属性对齐 

**Authors**: Ziyu Fan, Zhijian Huang, Yahan Li, Xiaowen Hu, Siyuan Shen, Yunliang Wang, Zeyu Zhong, Shuhong Liu, Shuning Yang, Shangqian Wu, Min Wu, Lei Deng  

**Link**: [PDF](https://arxiv.org/pdf/2511.08080)  

**Abstract**: Property-constrained molecular generation and editing are crucial in AI-driven drug discovery but remain hindered by two factors: (i) capturing the complex relationships between molecular structures and multiple properties remains challenging, and (ii) the narrow coverage and incomplete annotations of molecular properties weaken the effectiveness of property-based models. To tackle these limitations, we propose HSPAG, a data-efficient framework featuring hierarchical structure-property alignment. By treating SMILES and molecular properties as complementary modalities, the model learns their relationships at atom, substructure, and whole-molecule levels. Moreover, we select representative samples through scaffold clustering and hard samples via an auxiliary variational auto-encoder (VAE), substantially reducing the required pre-training data. In addition, we incorporate a property relevance-aware masking mechanism and diversified perturbation strategies to enhance generation quality under sparse annotations. Experiments demonstrate that HSPAG captures fine-grained structure-property relationships and supports controllable generation under multiple property constraints. Two real-world case studies further validate the editing capabilities of HSPAG. 

**Abstract (ZH)**: 基于层级结构-性质对齐的高效数据驱动分子生成与编辑框架：克服AI驱动药物发现中的挑战 

---
# Constrained and Robust Policy Synthesis with Satisfiability-Modulo-Probabilistic-Model-Checking 

**Title (ZH)**: 基于满足度模概率模型检验的约束与鲁棒策略合成 

**Authors**: Linus Heck, Filip Macák, Milan Češka, Sebastian Junges  

**Link**: [PDF](https://arxiv.org/pdf/2511.08078)  

**Abstract**: The ability to compute reward-optimal policies for given and known finite Markov decision processes (MDPs) underpins a variety of applications across planning, controller synthesis, and verification. However, we often want policies (1) to be robust, i.e., they perform well on perturbations of the MDP and (2) to satisfy additional structural constraints regarding, e.g., their representation or implementation cost. Computing such robust and constrained policies is indeed computationally more challenging. This paper contributes the first approach to effectively compute robust policies subject to arbitrary structural constraints using a flexible and efficient framework. We achieve flexibility by allowing to express our constraints in a first-order theory over a set of MDPs, while the root for our efficiency lies in the tight integration of satisfiability solvers to handle the combinatorial nature of the problem and probabilistic model checking algorithms to handle the analysis of MDPs. Experiments on a few hundred benchmarks demonstrate the feasibility for constrained and robust policy synthesis and the competitiveness with state-of-the-art methods for various fragments of the problem. 

**Abstract (ZH)**: 基于给定和已知有限马尔可夫决策过程（MDPs）计算奖励最优策略的能力支撑着规划、控制器合成和验证等领域的多种应用。然而，我们往往希望策略（1）具有鲁棒性，即在MDPs的扰动下表现良好，（2）满足额外的结构约束，例如它们的表示或实施成本。计算此类鲁棒性和约束条件下的策略确实更具计算挑战性。本文提出了一种有效计算满足任意结构约束的鲁棒策略的方法，利用了一个灵活且高效的框架。通过允许在MDP集合上的一阶理论中表达约束条件来实现灵活性，而我们的高效性则源于将满足性求解器紧密集成以处理问题的组合性质，并结合概率模型检验算法来处理MDPs的分析。在数百个基准测试上的实验展示了约束和鲁棒策略合成的可行性，并且在问题的不同片段上与最先进的方法具有竞争力。 

---
# An Integrated Fusion Framework for Ensemble Learning Leveraging Gradient Boosting and Fuzzy Rule-Based Models 

**Title (ZH)**: 基于梯度增强和模糊规则模型的集成融合框架 

**Authors**: Jinbo Li, Peng Liu, Long Chen, Witold Pedrycz, Weiping Ding  

**Link**: [PDF](https://arxiv.org/pdf/2511.08077)  

**Abstract**: The integration of different learning paradigms has long been a focus of machine learning research, aimed at overcoming the inherent limitations of individual methods. Fuzzy rule-based models excel in interpretability and have seen widespread application across diverse fields. However, they face challenges such as complex design specifications and scalability issues with large datasets. The fusion of different techniques and strategies, particularly Gradient Boosting, with Fuzzy Rule-Based Models offers a robust solution to these challenges. This paper proposes an Integrated Fusion Framework that merges the strengths of both paradigms to enhance model performance and interpretability. At each iteration, a Fuzzy Rule-Based Model is constructed and controlled by a dynamic factor to optimize its contribution to the overall ensemble. This control factor serves multiple purposes: it prevents model dominance, encourages diversity, acts as a regularization parameter, and provides a mechanism for dynamic tuning based on model performance, thus mitigating the risk of overfitting. Additionally, the framework incorporates a sample-based correction mechanism that allows for adaptive adjustments based on feedback from a validation set. Experimental results substantiate the efficacy of the presented gradient boosting framework for fuzzy rule-based models, demonstrating performance enhancement, especially in terms of mitigating overfitting and complexity typically associated with many rules. By leveraging an optimal factor to govern the contribution of each model, the framework improves performance, maintains interpretability, and simplifies the maintenance and update of the models. 

**Abstract (ZH)**: 不同学习范式的集成：一种模糊规则基于模型与梯度增强融合框架 

---
# Radar-APLANC: Unsupervised Radar-based Heartbeat Sensing via Augmented Pseudo-Label and Noise Contrast 

**Title (ZH)**: 雷达-APLANC：基于增强伪标签和噪声对比的无监督雷达心率感知 

**Authors**: Ying Wang, Zhaodong Sun, Xu Cheng, Zuxian He, Xiaobai Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.08071)  

**Abstract**: Frequency Modulated Continuous Wave (FMCW) radars can measure subtle chest wall oscillations to enable non-contact heartbeat sensing. However, traditional radar-based heartbeat sensing methods face performance degradation due to noise. Learning-based radar methods achieve better noise robustness but require costly labeled signals for supervised training. To overcome these limitations, we propose the first unsupervised framework for radar-based heartbeat sensing via Augmented Pseudo-Label and Noise Contrast (Radar-APLANC). We propose to use both the heartbeat range and noise range within the radar range matrix to construct the positive and negative samples, respectively, for improved noise robustness. Our Noise-Contrastive Triplet (NCT) loss only utilizes positive samples, negative samples, and pseudo-label signals generated by the traditional radar method, thereby avoiding dependence on expensive ground-truth physiological signals. We further design a pseudo-label augmentation approach featuring adaptive noise-aware label selection to improve pseudo-label signal quality. Extensive experiments on the Equipleth dataset and our collected radar dataset demonstrate that our unsupervised method achieves performance comparable to state-of-the-art supervised methods. Our code, dataset, and supplementary materials can be accessed from this https URL. 

**Abstract (ZH)**: 基于FMCW雷达的无监督心率检测方法：Augmented Pseudo-Label and Noise Contrast (Radar-APLANC) 

---
# Taming Identity Consistency and Prompt Diversity in Diffusion Models via Latent Concatenation and Masked Conditional Flow Matching 

**Title (ZH)**: 通过潜在连接和掩码条件流匹配来驯服扩散模型中的身份一致性和提示多样性 

**Authors**: Aditi Singhania, Arushi Jain, Krutik Malani, Riddhi Dhawan, Souymodip Chakraborty, Vineet Batra, Ankit Phogat  

**Link**: [PDF](https://arxiv.org/pdf/2511.08061)  

**Abstract**: Subject-driven image generation aims to synthesize novel depictions of a specific subject across diverse contexts while preserving its core identity features. Achieving both strong identity consistency and high prompt diversity presents a fundamental trade-off. We propose a LoRA fine-tuned diffusion model employing a latent concatenation strategy, which jointly processes reference and target images, combined with a masked Conditional Flow Matching (CFM) objective. This approach enables robust identity preservation without architectural modifications. To facilitate large-scale training, we introduce a two-stage Distilled Data Curation Framework: the first stage leverages data restoration and VLM-based filtering to create a compact, high-quality seed dataset from diverse sources; the second stage utilizes these curated examples for parameter-efficient fine-tuning, thus scaling the generation capability across various subjects and contexts. Finally, for filtering and quality assessment, we present CHARIS, a fine-grained evaluation framework that performs attribute-level comparisons along five key axes: identity consistency, prompt adherence, region-wise color fidelity, visual quality, and transformation diversity. 

**Abstract (ZH)**: 主题驱动的图像生成旨在在多样化的背景下合成特定主题的新颖表现，同时保留其核心身份特征。在实现强烈的身份一致性与高提示多样性之间取得平衡存在一个基本的权衡。我们提出了一种采用潜在连接策略的LoRA微调扩散模型，并结合掩蔽条件流匹配（CFM）目标，这种方法能够在无需架构修改的情况下实现稳健的身份保存。为了促进大规模训练，我们引入了一种两级蒸馏数据编目框架：第一阶段利用数据恢复和VLM基于的过滤来从多种来源创建一个紧凑且高质量的种子数据集；第二阶段利用这些编目示例进行参数高效的微调，从而在各种主题和背景下扩展生成能力。最后，为了过滤和质量评估，我们提出了CHARIS，这是一种精细粒度的评估框架，沿五个关键维度进行属性级比较：身份一致性、提示一致性、区域色彩保真度、视觉质量以及转换多样性。 

---
# ProSona: Prompt-Guided Personalization for Multi-Expert Medical Image Segmentation 

**Title (ZH)**: ProSona: 提示引导的多专家医疗影像分割个性化方法 

**Authors**: Aya Elgebaly, Nikolaos Delopoulos, Juliane Hörner-Rieber, Carolin Rippke, Sebastian Klüter, Luca Boldrini, Lorenzo Placidi, Riccardo Dal Bello, Nicolaus Andratschke, Michael Baumgartl, Claus Belka, Christopher Kurz, Guillaume Landry, Shadi Albarqouni  

**Link**: [PDF](https://arxiv.org/pdf/2511.08046)  

**Abstract**: Automated medical image segmentation suffers from high inter-observer variability, particularly in tasks such as lung nodule delineation, where experts often disagree. Existing approaches either collapse this variability into a consensus mask or rely on separate model branches for each annotator. We introduce ProSona, a two-stage framework that learns a continuous latent space of annotation styles, enabling controllable personalization via natural language prompts. A probabilistic U-Net backbone captures diverse expert hypotheses, while a prompt-guided projection mechanism navigates this latent space to generate personalized segmentations. A multi-level contrastive objective aligns textual and visual representations, promoting disentangled and interpretable expert styles. Across the LIDC-IDRI lung nodule and multi-institutional prostate MRI datasets, ProSona reduces the Generalized Energy Distance by 17% and improves mean Dice by more than one point compared with DPersona. These results demonstrate that natural-language prompts can provide flexible, accurate, and interpretable control over personalized medical image segmentation. Our implementation is available online 1 . 

**Abstract (ZH)**: 自动化医学图像分割遭受高观测者间变异性的问题，特别是在肺结节勾勒等任务中，专家常常存在分歧。现有的方法要么将这种变异性合并到共识掩模中，要么依赖于为每个标注员单独的模型分支。我们引入了ProSona，这是一种两阶段框架，学习注释风格的连续潜在空间，通过自然语言提示实现可控的个性化。概率U-Net骨干捕捉多样化的专家假设，而提示导向的投影机制导航这一潜在空间以生成个性化分割。多级对比目标对齐文本和视觉表示，促进独立且可解释的专家风格。在LIDC-IDRI肺结节和多机构前列腺MRI数据集中，ProSona将广义能量距离降低17%，并且平均Dice系数提高超过1分，相比DPersona。这些结果表明，自然语言提示可以提供灵活、准确且可解释的个性化医学图像分割控制。我们的实现已在线可用。 

---
# Morphing Through Time: Diffusion-Based Bridging of Temporal Gaps for Robust Alignment in Change Detection 

**Title (ZH)**: 时间变换中的形态转换：基于扩散的时间间隔桥接方法以实现稳健的变迁检测对齐 

**Authors**: Seyedehanita Madani, Vishal M. Patel  

**Link**: [PDF](https://arxiv.org/pdf/2511.07976)  

**Abstract**: Remote sensing change detection is often challenged by spatial misalignment between bi-temporal images, especially when acquisitions are separated by long seasonal or multi-year gaps. While modern convolutional and transformer-based models perform well on aligned data, their reliance on precise co-registration limits their robustness in real-world conditions. Existing joint registration-detection frameworks typically require retraining and transfer poorly across domains. We introduce a modular pipeline that improves spatial and temporal robustness without altering existing change detection networks. The framework integrates diffusion-based semantic morphing, dense registration, and residual flow refinement. A diffusion module synthesizes intermediate morphing frames that bridge large appearance gaps, enabling RoMa to estimate stepwise correspondences between consecutive frames. The composed flow is then refined through a lightweight U-Net to produce a high-fidelity warp that co-registers the original image pair. Extensive experiments on LEVIR-CD, WHU-CD, and DSIFN-CD show consistent gains in both registration accuracy and downstream change detection across multiple backbones, demonstrating the generality and effectiveness of the proposed approach. 

**Abstract (ZH)**: 远程 sensing 变化检测常常受到生物同期影像间空间错位的挑战，尤其是在收购时间间隔较长的季节性或多年时间跨度之后。虽然现代卷积和基于变换模型在对齐数据上表现良好，但它们对精确对齐的依赖限制了其在实际条件下的鲁棒性。现有的联合注册-检测框架通常需要重新训练，并且跨领域迁移效果不佳。我们提出了一种模块化管道，以无需修改现有变化检测网络的方式提高空间和时间鲁棒性。该框架整合了基于扩散的语义形变、密集注册和残差流优化。扩散模块合成中间形变帧，以桥接大量外观差异，使 RoMa 能够估计连续帧之间的逐步对应关系。然后，通过一个轻量级 U-Net 对合成的流进行优化，生成高质量的插值，实现原始影像对的精确对齐。在 LEVIR-CD、WHU-CD 和 DSIFN-CD 上的广泛实验显示，该方法在多个骨干网络下在注册精度和下游变化检测方面都取得了稳定的提升，证明了该方法的普遍性和有效性。 

---
# Reliable and Private Utility Signaling for Data Markets 

**Title (ZH)**: 可靠且私密的实用信号传递机制在数据市场中的应用 

**Authors**: Li Peng, Jiayao Zhang, Yihang Wu, Weiran Liu, Jinfei Liu, Zheng Yan, Kui Ren, Lei Zhang, Lin Qu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07975)  

**Abstract**: The explosive growth of data has highlighted its critical role in driving economic growth through data marketplaces, which enable extensive data sharing and access to high-quality datasets. To support effective trading, signaling mechanisms provide participants with information about data products before transactions, enabling informed decisions and facilitating trading. However, due to the inherent free-duplication nature of data, commonly practiced signaling methods face a dilemma between privacy and reliability, undermining the effectiveness of signals in guiding decision-making.
To address this, this paper explores the benefits and develops a non-TCP-based construction for a desirable signaling mechanism that simultaneously ensures privacy and reliability. We begin by formally defining the desirable utility signaling mechanism and proving its ability to prevent suboptimal decisions for both participants and facilitate informed data trading. To design a protocol to realize its functionality, we propose leveraging maliciously secure multi-party computation (MPC) to ensure the privacy and robustness of signal computation and introduce an MPC-based hash verification scheme to ensure input reliability. In multi-seller scenarios requiring fair data valuation, we further explore the design and optimization of the MPC-based KNN-Shapley method with improved efficiency. Rigorous experiments demonstrate the efficiency and practicality of our approach. 

**Abstract (ZH)**: 数据爆炸性增长凸显了通过数据市场促进经济增长的关键作用，数据市场使得广泛的数据共享和高质量数据集的访问成为可能。为了支持有效的交易，信号机制在交易前为参与者提供数据产品的信息，使他们能够做出明智的决策并促进交易。然而，由于数据的固有可复制性，常见的信号方法面临着隐私与可靠性的困境，这削弱了信号在指导决策方面的有效性。

为此，本文探讨了信号机制的优势并开发了一种非TCP基础的构建，确保同时实现隐私和可靠性。我们首先正式定义了理想的有用信号机制，并证明其能够防止双方做出次优决策并促进知情的数据交易。为了实现其功能，我们提出利用恶意安全多方计算（MPC）来确保信号计算的隐私性和鲁棒性，并引入基于MPC的哈希验证方案以确保输入的可靠性。在需要公平数据估值的多卖家场景中，我们进一步探讨了MPC基础的KNN-Shapley方法的设计和优化，以提高效率。严格的实验验证了我们方法的高效性和实用性。 

---
# Balance Equation-based Distributionally Robust Offline Imitation Learning 

**Title (ZH)**: 基于平衡方程的分布鲁棒离线 imitation 学习 

**Authors**: Rishabh Agrawal, Yusuf Alvi, Rahul Jain, Ashutosh Nayyar  

**Link**: [PDF](https://arxiv.org/pdf/2511.07942)  

**Abstract**: Imitation Learning (IL) has proven highly effective for robotic and control tasks where manually designing reward functions or explicit controllers is infeasible. However, standard IL methods implicitly assume that the environment dynamics remain fixed between training and deployment. In practice, this assumption rarely holds where modeling inaccuracies, real-world parameter variations, and adversarial perturbations can all induce shifts in transition dynamics, leading to severe performance degradation. We address this challenge through Balance Equation-based Distributionally Robust Offline Imitation Learning, a framework that learns robust policies solely from expert demonstrations collected under nominal dynamics, without requiring further environment interaction. We formulate the problem as a distributionally robust optimization over an uncertainty set of transition models, seeking a policy that minimizes the imitation loss under the worst-case transition distribution. Importantly, we show that this robust objective can be reformulated entirely in terms of the nominal data distribution, enabling tractable offline learning. Empirical evaluations on continuous-control benchmarks demonstrate that our approach achieves superior robustness and generalization compared to state-of-the-art offline IL baselines, particularly under perturbed or shifted environments. 

**Abstract (ZH)**: 基于平衡方程的分布鲁棒离线 imitation 学习 

---
# SpeechJudge: Towards Human-Level Judgment for Speech Naturalness 

**Title (ZH)**: SpeechJudge:向人类水平的语音自然度判断迈进 

**Authors**: Xueyao Zhang, Chaoren Wang, Huan Liao, Ziniu Li, Yuancheng Wang, Li Wang, Dongya Jia, Yuanzhe Chen, Xiulin Li, Zhuo Chen, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07931)  

**Abstract**: Aligning large generative models with human feedback is a critical challenge. In speech synthesis, this is particularly pronounced due to the lack of a large-scale human preference dataset, which hinders the development of models that truly align with human perception. To address this, we introduce SpeechJudge, a comprehensive suite comprising a dataset, a benchmark, and a reward model centered on naturalness--one of the most fundamental subjective metrics for speech synthesis. First, we present SpeechJudge-Data, a large-scale human feedback corpus of 99K speech pairs. The dataset is constructed using a diverse set of advanced zero-shot text-to-speech (TTS) models across diverse speech styles and multiple languages, with human annotations for both intelligibility and naturalness preference. From this, we establish SpeechJudge-Eval, a challenging benchmark for speech naturalness judgment. Our evaluation reveals that existing metrics and AudioLLMs struggle with this task; the leading model, Gemini-2.5-Flash, achieves less than 70% agreement with human judgment, highlighting a significant gap for improvement. To bridge this gap, we develop SpeechJudge-GRM, a generative reward model (GRM) based on Qwen2.5-Omni-7B. It is trained on SpeechJudge-Data via a two-stage post-training process: Supervised Fine-Tuning (SFT) with Chain-of-Thought rationales followed by Reinforcement Learning (RL) with GRPO on challenging cases. On the SpeechJudge-Eval benchmark, the proposed SpeechJudge-GRM demonstrates superior performance, achieving 77.2% accuracy (and 79.4% after inference-time scaling @10) compared to a classic Bradley-Terry reward model (72.7%). Furthermore, SpeechJudge-GRM can be also employed as a reward function during the post-training of speech generation models to facilitate their alignment with human preferences. 

**Abstract (ZH)**: 用人类反馈对齐大型生成模型是关键挑战。在语音合成中，由于缺乏大规模人类偏好数据集，这阻碍了能够真正与人类感知对齐的模型的发展。为此，我们引入了SpeechJudge，这是一个全面的套件，包括一个数据集、一个基准和一个以自然度为中心的奖励模型——这是语音合成中最基本的主观度量之一。首先，我们介绍了SpeechJudge-Data，这是一个包含99K语音对的大规模人类反馈语料库。该数据集使用了跨多种语音风格和多种语言的多样化先进零样本文本到语音（TTS）模型，并有人类对明晰度和自然度偏好的注释。基于此，我们建立了SpeechJudge-Eval，一个具有挑战性的语音自然度判断基准。我们的评估表明，现有度量标准和AudioLLMs在这一任务中表现不佳；领先模型Gemini-2.5-Flash的匹配人类判断的准确率低于70%，突显了改进空间的巨大需求。为了弥合这一差距，我们开发了SpeechJudge-GRM，一个基于Qwen2.5-Omni-7B的生成奖励模型（GRM）。该模型通过两级后训练过程进行训练：带有链式思考理由的有监督微调（SFT）后，使用GRPO在挑战性案例上进行强化学习（RL）。在SpeechJudge-Eval基准测试上，提出的SpeechJudge-GRM表现出色，准确率达到77.2%（并在推理时放大10倍后达到79.4%），高于经典的Bradley-Terry奖励模型（72.7%）。此外，SpeechJudge-GRM还可以用作语音生成模型后训练的奖励函数，以促进其与人类偏好的对齐。 

---
# CNN-Based Automated Parameter Extraction Framework for Modeling Memristive Devices 

**Title (ZH)**: 基于CNN的 memristive 器件建模参数自动化提取框架 

**Authors**: Akif Hamid, Orchi Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07926)  

**Abstract**: Resistive random access memory (RRAM) is a promising candidate for next-generation nonvolatile memory (NVM) and in-memory computing applications. Compact models are essential for analyzing the circuit and system-level performance of experimental RRAM devices. However, most existing RRAM compact models rely on multiple fitting parameters to reproduce the device I-V characteristics, and in most cases, as the parameters are not directly related to measurable quantities, their extraction requires extensive manual tuning, making the process time-consuming and limiting adaptability across different devices. This work presents an automated framework for extracting the fitting parameters of the widely used Stanford RRAM model directly from the device I-V characteristics. The framework employs a convolutional neural network (CNN) trained on a synthetic dataset to generate initial parameter estimates, which are then refined through three heuristic optimization blocks that minimize errors via adaptive binary search in the parameter space. We evaluated the framework using four key NVM metrics: set voltage, reset voltage, hysteresis loop area, and low resistance state (LRS) slope. Benchmarking against RRAM device characteristics derived from previously reported Stanford model fits, other analytical models, and experimental data shows that the framework achieves low error across diverse device characteristics, offering a fast, reliable, and robust solution for RRAM modeling. 

**Abstract (ZH)**: 基于电阻随机访问记忆体(RRAM)的自动生成紧凑模型框架 

---
# Toward Adaptive BCIs: Enhancing Decoding Stability via User State-Aware EEG Filtering 

**Title (ZH)**: 面向自适应BCI：基于用户状态的EEG滤波增强解码稳定性 

**Authors**: Yeon-Woo Choi, Hye-Bin Shin, Dan Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.07891)  

**Abstract**: Brain-computer interfaces (BCIs) often suffer from limited robustness and poor long-term adaptability. Model performance rapidly degrades when user attention fluctuates, brain states shift over time, or irregular artifacts appear during interaction. To mitigate these issues, we introduce a user state-aware electroencephalogram (EEG) filtering framework that refines neural representations before decoding user intentions. The proposed method continuously estimates the user's cognitive state (e.g., focus or distraction) from EEG features and filters unreliable segments by applying adaptive weighting based on the estimated attention level. This filtering stage suppresses noisy or out-of-focus epochs, thereby reducing distributional drift and improving the consistency of subsequent decoding. Experiments on multiple EEG datasets that emulate real BCI scenarios demonstrate that the proposed state-aware filtering enhances classification accuracy and stability across different user states and sessions compared with conventional preprocessing pipelines. These findings highlight that leveraging brain-derived state information--even without additional user labels--can substantially improve the reliability of practical EEG-based BCIs. 

**Abstract (ZH)**: 基于用户状态意识的脑电图（EEG）滤波框架：提升脑机接口的鲁棒性和长期适应性 

---
# Generating Sketches in a Hierarchical Auto-Regressive Process for Flexible Sketch Drawing Manipulation at Stroke-Level 

**Title (ZH)**: 在层次自回归过程中生成草图以实现.stroke级灵活草图绘制 manipulatingstroke级灵活草图绘制 

**Authors**: Sicong Zang, Shuhui Gao, Zhijun Fang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07889)  

**Abstract**: Generating sketches with specific patterns as expected, i.e., manipulating sketches in a controllable way, is a popular task. Recent studies control sketch features at stroke-level by editing values of stroke embeddings as conditions. However, in order to provide generator a global view about what a sketch is going to be drawn, all these edited conditions should be collected and fed into generator simultaneously before generation starts, i.e., no further manipulation is allowed during sketch generating process. In order to realize sketch drawing manipulation more flexibly, we propose a hierarchical auto-regressive sketch generating process. Instead of generating an entire sketch at once, each stroke in a sketch is generated in a three-staged hierarchy: 1) predicting a stroke embedding to represent which stroke is going to be drawn, and 2) anchoring the predicted stroke on the canvas, and 3) translating the embedding to a sequence of drawing actions to form the full sketch. Moreover, the stroke prediction, anchoring and translation are proceeded auto-regressively, i.e., both the recently generated strokes and their positions are considered to predict the current one, guiding model to produce an appropriate stroke at a suitable position to benefit the full sketch generation. It is flexible to manipulate stroke-level sketch drawing at any time during generation by adjusting the exposed editable stroke embeddings. 

**Abstract (ZH)**: 基于特定模式的素描生成：一种可控的分层自回归生成过程 

---
# Meta-cognitive Multi-scale Hierarchical Reasoning for Motor Imagery Decoding 

**Title (ZH)**: 元认知多尺度层次推理在运动想象解码中的应用 

**Authors**: Si-Hyun Kim, Heon-Gyu Kwak, Byoung-Hee Kwon, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.07884)  

**Abstract**: Brain-computer interface (BCI) aims to decode motor intent from noninvasive neural signals to enable control of external devices, but practical deployment remains limited by noise and variability in motor imagery (MI)-based electroencephalogram (EEG) signals. This work investigates a hierarchical and meta-cognitive decoding framework for four-class MI classification. We introduce a multi-scale hierarchical signal processing module that reorganizes backbone features into temporal multi-scale representations, together with an introspective uncertainty estimation module that assigns per-cycle reliability scores and guides iterative refinement. We instantiate this framework on three standard EEG backbones (EEGNet, ShallowConvNet, and DeepConvNet) and evaluate four-class MI decoding using the BCI Competition IV-2a dataset under a subject-independent setting. Across all backbones, the proposed components improve average classification accuracy and reduce inter-subject variance compared to the corresponding baselines, indicating increased robustness to subject heterogeneity and noisy trials. These results suggest that combining hierarchical multi-scale processing with introspective confidence estimation can enhance the reliability of MI-based BCI systems. 

**Abstract (ZH)**: 基于层级和元认知解码框架的四类运动想象分类研究 

---
# A General Method for Proving Networks Universal Approximation Property 

**Title (ZH)**: 一种证明网络通用逼近性质的通用方法 

**Authors**: Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07857)  

**Abstract**: Deep learning architectures are highly diverse. To prove their universal approximation properties, existing works typically rely on model-specific proofs. Generally, they construct a dedicated mathematical formulation for each architecture (e.g., fully connected networks, CNNs, or Transformers) and then prove their universal approximability. However, this approach suffers from two major limitations: first, every newly proposed architecture often requires a completely new proof from scratch; second, these proofs are largely isolated from one another, lacking a common analytical foundation. This not only incurs significant redundancy but also hinders unified theoretical understanding across different network families. To address these issues, this paper proposes a general and modular framework for proving universal approximation. We define a basic building block (comprising one or multiple layers) that possesses the universal approximation property as a Universal Approximation Module (UAM). Under this condition, we show that any deep network composed of such modules inherently retains the universal approximation property. Moreover, the overall approximation process can be interpreted as a progressive refinement across modules. This perspective not only unifies the analysis of diverse architectures but also enables a step-by-step understanding of how expressive power evolves through the network. 

**Abstract (ZH)**: 深度学习架构高度多样化。为了证明其通用逼近性质，现有工作通常依赖于特定模型的证明。一般而言，它们为每个架构（例如全连接网络、CNN或Transformer）构建专门的数学表述，然后证明其通用逼近性。然而，这种方法存在两个主要局限性：首先，每一项新提出的架构往往需要从头开始构建全新的证明；其次，这些证明彼此相互隔离，缺乏一个共同的分析基础。这不仅导致了显著的冗余，还阻碍了对不同网络家族进行统一的理论理解。为了解决这些问题，本文提出了一种通用且模块化的框架来证明通用逼近性质。我们定义了一个基本构建块（由一个或多个层组成），并将其称为通用逼近模块（UAM），在满足这一条件的情况下，我们证明由这些模块组成的任何深度网络本身都保留了通用逼近性质。此外，整体逼近过程可以解释为模块间的逐步优化。这一视角不仅统一了对各种架构的分析，还允许逐步理解表达能力如何随网络发展而演变。 

---
# PRISM: Privacy-preserving Inference System with Homomorphic Encryption and Modular Activation 

**Title (ZH)**: PRISM: 保留隐私的同态加密与模块化激活推理系统 

**Authors**: Zeinab Elkhatib, Ali Sekmen, Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07807)  

**Abstract**: With the rapid advancements in machine learning, models have become increasingly capable of learning and making predictions in various industries. However, deploying these models in critical infrastructures presents a major challenge, as concerns about data privacy prevent unrestricted data sharing. Homomor- phic encryption (HE) offers a solution by enabling computations on encrypted data, but it remains incompatible with machine learning models like convolutional neural networks (CNNs), due to their reliance on non-linear activation functions. To bridge this gap, this work proposes an optimized framework that replaces standard non-linear functions with homomorphically compatible approximations, ensuring secure computations while minimizing computational overhead. The proposed approach restructures the CNN architecture and introduces an efficient activation function approximation method to mitigate the performance trade-offs in- troduced by encryption. Experiments on CIFAR-10 achieve 94.4% accuracy with 2.42 s per single encrypted sample and 24,000 s per 10,000 encrypted samples, using a degree-4 polynomial and Softplus activation under CKKS, balancing accuracy and privacy. 

**Abstract (ZH)**: 基于同态加密的优化卷积神经网络框架：平衡准确率与隐私保护 

---
# HybridGuard: Enhancing Minority-Class Intrusion Detection in Dew-Enabled Edge-of-Things Networks 

**Title (ZH)**: HybridGuard: 提高Dew使能的边缘物联网网络中少数类入侵检测的性能 

**Authors**: Binayak Kara, Ujjwal Sahua, Ciza Thomas, Jyoti Prakash Sahoo  

**Link**: [PDF](https://arxiv.org/pdf/2511.07793)  

**Abstract**: Securing Dew-Enabled Edge-of-Things (EoT) networks against sophisticated intrusions is a critical challenge. This paper presents HybridGuard, a framework that integrates machine learning and deep learning to improve intrusion detection. HybridGuard addresses data imbalance through mutual information based feature selection, ensuring that the most relevant features are used to improve detection performance, especially for minority attack classes. The framework leverages Wasserstein Conditional Generative Adversarial Networks with Gradient Penalty (WCGAN-GP) to further reduce class imbalance and enhance detection precision. It adopts a two-phase architecture called DualNetShield to support advanced traffic analysis and anomaly detection, improving the granular identification of threats in complex EoT environments. HybridGuard is evaluated on the UNSW-NB15, CIC-IDS-2017, and IOTID20 datasets, where it demonstrates strong performance across diverse attack scenarios and outperforms existing solutions in adapting to evolving cybersecurity threats. This approach establishes HybridGuard as an effective tool for protecting EoT networks against modern intrusions. 

**Abstract (ZH)**: 利用机器学习和深度学习保护雾Enabled边缘事物(EoT)网络免受复杂入侵的HybridGuard框架 

---
# Physical Consistency of Aurora's Encoder: A Quantitative Study 

**Title (ZH)**: Aurora编码器的物理一致性：一项定量研究 

**Authors**: Benjamin Richards, Pushpa Kumar Balan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07787)  

**Abstract**: The high accuracy of large-scale weather forecasting models like Aurora is often accompanied by a lack of transparency, as their internal representations remain largely opaque. This "black box" nature hinders their adoption in high-stakes operational settings. In this work, we probe the physical consistency of Aurora's encoder by investigating whether its latent representations align with known physical and meteorological concepts. Using a large-scale dataset of embeddings, we train linear classifiers to identify three distinct concepts: the fundamental land-sea boundary, high-impact extreme temperature events, and atmospheric instability. Our findings provide quantitative evidence that Aurora learns physically consistent features, while also highlighting its limitations in capturing the rarest events. This work underscores the critical need for interpretability methods to validate and build trust in the next generation of Al-driven weather models. 

**Abstract (ZH)**: 大规模天气预报模型如Aurora的高精度往往伴随着透明度的缺乏，因为其内部表示仍然 largely 不透明。这种“黑盒”性质阻碍了它们在高风险 operational 设置中的采用。在本文中，我们通过调查Aurora 编码器的物理一致性，探讨其潜在表示是否与已知的物理和气象概念相一致。利用大规模嵌入数据集，我们训练线性分类器来识别三个不同的概念：基本的陆海边界、具有高影响的极端温度事件以及大气不稳定。我们的研究提供了定量证据表明Aurora 学习了物理上一致的特征，同时也指出了它在捕捉最罕见事件方面的局限性。本文强调了对可解释性方法的需求，以验证和建立对下一代基于AI的天气模型的信任。 

---
# TurboSAT: Gradient-Guided Boolean Satisfiability Accelerated on GPU-CPU Hybrid System 

**Title (ZH)**: TurboSAT：由GPU-CPU混合系统加速的梯度导向布尔可满足性求解 

**Authors**: Steve Dai, Cunxi Yu, Kalyan Krishnamani, Brucek Khailany  

**Link**: [PDF](https://arxiv.org/pdf/2511.07737)  

**Abstract**: While accelerated computing has transformed many domains of computing, its impact on logical reasoning, specifically Boolean satisfiability (SAT), remains limited. State-of-the-art SAT solvers rely heavily on inherently sequential conflict-driven search algorithms that offer powerful heuristics but limit the amount of parallelism that could otherwise enable significantly more scalable SAT solving. Inspired by neural network training, we formulate the SAT problem as a binarized matrix-matrix multiplication layer that could be optimized using a differentiable objective function. Enabled by this encoding, we combine the strengths of parallel differentiable optimization and sequential search to accelerate SAT on a hybrid GPU-CPU system. In this system, the GPUs leverage parallel differentiable solving to rapidly evaluate SAT clauses and use gradients to stochastically explore the solution space and optimize variable assignments. Promising partial assignments generated by the GPUs are post-processed on many CPU threads which exploit conflict-driven sequential search to further traverse the solution subspaces and identify complete assignments. Prototyping the hybrid solver on an NVIDIA DGX GB200 node, our solver achieves runtime speedups up to over 200x when compared to a state-of-the-art CPU-based solver on public satisfiable benchmark problems from the SAT Competition. 

**Abstract (ZH)**: 尽管加速计算已经改变了众多计算领域，但它对逻辑推理，特别是布尔可满足性（SAT）的影响仍然有限。基于神经网络训练的启发，我们将SAT问题形式化为二值化矩阵-矩阵乘法层，并使用可微目标函数进行优化。借助这一编码，我们将并行可微优化和顺序搜索的优势相结合，以加速SAT问题在混合GPU-CPU系统上的求解。在这个系统中，GPU利用并行可微求解来快速评估SAT子句，并使用梯度进行随机搜索和优化变量赋值。由GPU生成的有希望的部分赋值在多个CPU线程上进行后处理，利用冲突驱动的顺序搜索进一步探索解空间并识别完整赋值。在NVIDIA DGX GB200节点上对混合求解器进行原型设计，我们的求解器在公共可满足性基准问题（来自SAT竞赛）与最先进的基于CPU的求解器相比，可实现超过200倍的运行时加速。 

---
# Global Optimization on Graph-Structured Data via Gaussian Processes with Spectral Representations 

**Title (ZH)**: 基于谱表示的图形结构数据的高斯过程全局优化 

**Authors**: Shu Hong, Yongsheng Mei, Mahdi Imani, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2511.07734)  

**Abstract**: Bayesian optimization (BO) is a powerful framework for optimizing expensive black-box objectives, yet extending it to graph-structured domains remains challenging due to the discrete and combinatorial nature of graphs. Existing approaches often rely on either full graph topology-impractical for large or partially observed graphs-or incremental exploration, which can lead to slow convergence. We introduce a scalable framework for global optimization over graphs that employs low-rank spectral representations to build Gaussian process (GP) surrogates from sparse structural observations. The method jointly infers graph structure and node representations through learnable embeddings, enabling efficient global search and principled uncertainty estimation even with limited data. We also provide theoretical analysis establishing conditions for accurate recovery of underlying graph structure under different sampling regimes. Experiments on synthetic and real-world datasets demonstrate that our approach achieves faster convergence and improved optimization performance compared to prior methods. 

**Abstract (ZH)**: 基于图的全局优化的可扩展框架：通过低秩谱表示构建高斯过程代理模型 

---
# Diffusion Guided Adversarial State Perturbations in Reinforcement Learning 

**Title (ZH)**: 基于扩散引导的对抗状态扰动在强化学习中的应用 

**Authors**: Xiaolin Sun, Feidi Liu, Zhengming Ding, ZiZhan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.07701)  

**Abstract**: Reinforcement learning (RL) systems, while achieving remarkable success across various domains, are vulnerable to adversarial attacks. This is especially a concern in vision-based environments where minor manipulations of high-dimensional image inputs can easily mislead the agent's behavior. To this end, various defenses have been proposed recently, with state-of-the-art approaches achieving robust performance even under large state perturbations. However, after closer investigation, we found that the effectiveness of the current defenses is due to a fundamental weakness of the existing $l_p$ norm-constrained attacks, which can barely alter the semantics of image input even under a relatively large perturbation budget. In this work, we propose SHIFT, a novel policy-agnostic diffusion-based state perturbation attack to go beyond this limitation. Our attack is able to generate perturbed states that are semantically different from the true states while remaining realistic and history-aligned to avoid detection. Evaluations show that our attack effectively breaks existing defenses, including the most sophisticated ones, significantly outperforming existing attacks while being more perceptually stealthy. The results highlight the vulnerability of RL agents to semantics-aware adversarial perturbations, indicating the importance of developing more robust policies. 

**Abstract (ZH)**: 基于强化学习的对抗攻击：超越$L_p$范数约束的新型语义差异扩散状态扰动攻击 

---
# Stress Testing Factual Consistency Metrics for Long-Document Summarization 

**Title (ZH)**: 长文档摘要中事实一致性度量的压测研究 

**Authors**: Zain Muhammad Mujahid, Dustin Wright, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2511.07689)  

**Abstract**: Evaluating the factual consistency of abstractive text summarization remains a significant challenge, particularly for long documents, where conventional metrics struggle with input length limitations and long-range dependencies. In this work, we systematically evaluate the reliability of six widely used reference-free factuality metrics, originally proposed for short-form summarization, in the long-document setting. We probe metric robustness through seven factuality-preserving perturbations applied to summaries, namely paraphrasing, simplification, synonym replacement, logically equivalent negations, vocabulary reduction, compression, and source text insertion, and further analyze their sensitivity to retrieval context and claim information density. Across three long-form benchmark datasets spanning science fiction, legal, and scientific domains, our results reveal that existing short-form metrics produce inconsistent scores for semantically equivalent summaries and exhibit declining reliability for information-dense claims whose content is semantically similar to many parts of the source document. While expanding the retrieval context improves stability in some domains, no metric consistently maintains factual alignment under long-context conditions. Finally, our results highlight concrete directions for improving factuality evaluation, including multi-span reasoning, context-aware calibration, and training on meaning-preserving variations to enhance robustness in long-form summarization. We release all code, perturbed data, and scripts required to reproduce our results at this https URL. 

**Abstract (ZH)**: 评估提取式文本概要的事实一致性依然是一项重大挑战，特别是在长文档中，传统度量标准受到输入长度限制和长范围依赖性的困扰。在这项工作中，我们系统地评估了六种广泛使用的无参考事实一致性度量标准在长文档设置中的可靠性，这些度量标准最初是为短形式概要设计的。我们通过应用七种保持事实一致性的扰动来测试度量标准的稳健性，包括改写、简化、同义词替换、逻辑等价否定、词汇减少、压缩以及源文本插入，并进一步分析这些扰动对检索上下文和声明信息密度的敏感性。通过跨三个涵盖科幻、法律和科学领域的长形式基准数据集，我们的结果揭示了现有短形式度量标准在语义等价概要上产生不一致分数，并且对于内容与源文档多个部分在语义上相似的信息密集声明，其表现可靠性下降。尽管扩展检索上下文在某些领域提高了稳定性，但在长上下文条件下，没有一种度量标准能一直保持事实一致性。最后，我们的结果指出了改进事实一致性评估的具体方向，包括多跨度推理、上下文感知校准以及通过语义保持的变化进行训练以增强长形式概要中的鲁棒性。我们已在以下网址发布了所有用于复现结果的代码、扰动数据和脚本：this https URL。 

---
# Designing and Evaluating Malinowski's Lens: An AI-Native Educational Game for Ethnographic Learning 

**Title (ZH)**: Designing and Evaluating Malinowski's Lens: 一种面向AI的民族志学习教育游戏的设计与评估 

**Authors**: Michael Hoffmann, Jophin John, Jan Fillies, Adrian Paschke  

**Link**: [PDF](https://arxiv.org/pdf/2511.07682)  

**Abstract**: This study introduces 'Malinowski's Lens', the first AI-native educational game for anthropology that transforms Bronislaw Malinowski's 'Argonauts of the Western Pacific' (1922) into an interactive learning experience. The system combines Retrieval-Augmented Generation with DALL-E 3 text-to-image generation, creating consistent VGA-style visuals as players embody Malinowski during his Trobriand Islands fieldwork (1915-1918). To address ethical concerns, indigenous peoples appear as silhouettes while Malinowski is detailed, prompting reflection on anthropological representation. Two validation studies confirmed effectiveness: Study 1 with 10 non-specialists showed strong learning outcomes (average quiz score 7.5/10) and excellent usability (SUS: 83/100). Study 2 with 4 expert anthropologists confirmed pedagogical value, with one senior researcher discovering "new aspects" of Malinowski's work through gameplay. The findings demonstrate that AI-driven educational games can effectively convey complex anthropological concepts while sparking disciplinary curiosity. This study advances AI-native educational game design and provides a replicable model for transforming academic texts into engaging interactive experiences. 

**Abstract (ZH)**: 这一研究引入了“马林诺夫斯基之镜”，这是首款基于AI的教育游戏，将Bronislaw Malinowski的《西方太平洋的航海者》（1922年）转变为互动学习体验。该系统结合了检索增强生成与DALL-E 3文本转图像生成技术，创造连贯的VGA风格视觉效果，让玩家在特罗布里ands群岛田野工作期间化身马林诺夫斯基（1915-1918年）。为解决伦理关切，原住民以剪影形式出现，而马林诺夫斯基则被详细描绘，促使对人类学表现的反思。两项验证研究证实了其有效性：第一项研究有10名非专家参与者，显示了强大的学习成果（平均测验得分7.5/10）和优良的用户体验（SUS：83/100）。第二项研究有4名专家人类学家确认其教学价值，其中一位资深研究员通过游戏发现马林诺夫斯基工作的“新方面”。研究结果表明，AI驱动的教育游戏能够有效地传达复杂的人类学概念，并激发学科兴趣。该研究推进了AI原生教育游戏设计，并提供了一个可复制的模型，将学术文本转化为引人入胜的互动体验。 

---
# Speech Separation for Hearing-Impaired Children in the Classroom 

**Title (ZH)**: 课堂环境中听觉障碍儿童的语音分离 

**Authors**: Feyisayo Olalere, Kiki van der Heijden, H. Christiaan Stronks, Jeroen Briaire, Johan H. M. Frijns, Yagmur Güçlütürk  

**Link**: [PDF](https://arxiv.org/pdf/2511.07677)  

**Abstract**: Classroom environments are particularly challenging for children with hearing impairments, where background noise, multiple talkers, and reverberation degrade speech perception. These difficulties are greater for children than adults, yet most deep learning speech separation models for assistive devices are developed using adult voices in simplified, low-reverberation conditions. This overlooks both the higher spectral similarity of children's voices, which weakens separation cues, and the acoustic complexity of real classrooms. We address this gap using MIMO-TasNet, a compact, low-latency, multi-channel architecture suited for real-time deployment in bilateral hearing aids or cochlear implants. We simulated naturalistic classroom scenes with moving child-child and child-adult talker pairs under varying noise and distance conditions. Training strategies tested how well the model adapts to children's speech through spatial cues. Models trained on adult speech, classroom data, and finetuned variants were compared to assess data-efficient adaptation. Results show that adult-trained models perform well in clean scenes, but classroom-specific training greatly improves separation quality. Finetuning with only half the classroom data achieved comparable gains, confirming efficient transfer learning. Training with diffuse babble noise further enhanced robustness, and the model preserved spatial awareness while generalizing to unseen distances. These findings demonstrate that spatially aware architectures combined with targeted adaptation can improve speech accessibility for children in noisy classrooms, supporting future on-device assistive technologies. 

**Abstract (ZH)**: 教室环境对听力受损儿童构成了特别挑战，背景噪音、多名发言者和混响削弱了语音感知。儿童在这些困难上的体验比成人更为严峻，然而，现有的辅助设备中的深度学习语音分离模型大多是在简化且无明显混响的条件下使用成人语音进行开发的。这忽视了儿童语音在频谱上的高度相似性，削弱了分离线索，以及实际教室中的声学复杂性。我们利用MIMO-TasNet这一紧凑、低延迟、多通道架构来填补这一空白，该架构适合于双耳助听器或人工耳蜗植入设备的实时部署。我们模拟了自然场景中的教室环境，其中包含移动的儿童对和儿童-成人发言者对，并在不同的噪声和距离条件下进行了场景设置。训练策略测试了模型如何通过空间线索适应儿童的语音。我们对比了训练于成人语音、教室数据以及微调变体的模型，评估了其数据效能适应能力。结果显示，使用成人语音训练的模型在干净场景中表现良好，而针对教室环境的特定训练显著提高了分离质量。仅使用一半教室数据进行微调取得了相当的提升，这证实了高效的知识迁移。使用扩散背景噪音进行训练进一步增强了模型的鲁棒性，同时保持了空间意识，并能够泛化到未见过的距离。这些发现表明，结合空间意识架构和目标适应可以提高在噪声教室环境中儿童的语音可访问性，支持未来设备上的辅助技术的发展。 

---
# FractalCloud: A Fractal-Inspired Architecture for Efficient Large-Scale Point Cloud Processing 

**Title (ZH)**: FractalCloud：一种面向大规模点云处理的分形启发式架构 

**Authors**: Yuzhe Fu, Changchun Zhou, Hancheng Ye, Bowen Duan, Qiyu Huang, Chiyue Wei, Cong Guo, Hai "Helen'' Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.07665)  

**Abstract**: Three-dimensional (3D) point clouds are increasingly used in applications such as autonomous driving, robotics, and virtual reality (VR). Point-based neural networks (PNNs) have demonstrated strong performance in point cloud analysis, originally targeting small-scale inputs. However, as PNNs evolve to process large-scale point clouds with hundreds of thousands of points, all-to-all computation and global memory access in point cloud processing introduce substantial overhead, causing $O(n^2)$ computational complexity and memory traffic where n is the number of points}. Existing accelerators, primarily optimized for small-scale workloads, overlook this challenge and scale poorly due to inefficient partitioning and non-parallel architectures. To address these issues, we propose FractalCloud, a fractal-inspired hardware architecture for efficient large-scale 3D point cloud processing. FractalCloud introduces two key optimizations: (1) a co-designed Fractal method for shape-aware and hardware-friendly partitioning, and (2) block-parallel point operations that decompose and parallelize all point operations. A dedicated hardware design with on-chip fractal and flexible parallelism further enables fully parallel processing within limited memory resources. Implemented in 28 nm technology as a chip layout with a core area of 1.5 $mm^2$, FractalCloud achieves 21.7x speedup and 27x energy reduction over state-of-the-art accelerators while maintaining network accuracy, demonstrating its scalability and efficiency for PNN inference. 

**Abstract (ZH)**: 基于分形的FractalCloud：一种高效的大型3D点云处理硬件架构 

---
# Adaptive Graph Learning with Transformer for Multi-Reservoir Inflow Prediction 

**Title (ZH)**: 基于变换器的自适应图学习在多水库来水预测中的应用 

**Authors**: Pengfei Hu, Ming Fan, Xiaoxue Han, Chang Lu, Wei Zhang, Hyun Kang, Yue Ning, Dan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07649)  

**Abstract**: Reservoir inflow prediction is crucial for water resource management, yet existing approaches mainly focus on single-reservoir models that ignore spatial dependencies among interconnected reservoirs. We introduce AdaTrip as an adaptive, time-varying graph learning framework for multi-reservoir inflow forecasting. AdaTrip constructs dynamic graphs where reservoirs are nodes with directed edges reflecting hydrological connections, employing attention mechanisms to automatically identify crucial spatial and temporal dependencies. Evaluation on thirty reservoirs in the Upper Colorado River Basin demonstrates superiority over existing baselines, with improved performance for reservoirs with limited records through parameter sharing. Additionally, AdaTrip provides interpretable attention maps at edge and time-step levels, offering insights into hydrological controls to support operational decision-making. Our code is available at this https URL. 

**Abstract (ZH)**: 基于时空依赖性的多水电站来水预测：AdaTrip自适应时变图学习框架 

---
# Partial Action Replacement: Tackling Distribution Shift in Offline MARL 

**Title (ZH)**: 部分行动替换：应对离线MARL中的分布偏移 

**Authors**: Yue Jin, Giovanni Montana  

**Link**: [PDF](https://arxiv.org/pdf/2511.07629)  

**Abstract**: Offline multi-agent reinforcement learning (MARL) is severely hampered by the challenge of evaluating out-of-distribution (OOD) joint actions. Our core finding is that when the behavior policy is factorized - a common scenario where agents act fully or partially independently during data collection - a strategy of partial action replacement (PAR) can significantly mitigate this challenge. PAR updates a single or part of agents' actions while the others remain fixed to the behavioral data, reducing distribution shift compared to full joint-action updates. Based on this insight, we develop Soft-Partial Conservative Q-Learning (SPaCQL), using PAR to mitigate OOD issue and dynamically weighting different PAR strategies based on the uncertainty of value estimation. We provide a rigorous theoretical foundation for this approach, proving that under factorized behavior policies, the induced distribution shift scales linearly with the number of deviating agents rather than exponentially with the joint-action space. This yields a provably tighter value error bound for this important class of offline MARL problems. Our theoretical results also indicate that SPaCQL adaptively addresses distribution shift using uncertainty-informed weights. Our empirical results demonstrate SPaCQL enables more effective policy learning, and manifest its remarkable superiority over baseline algorithms when the offline dataset exhibits the independence structure. 

**Abstract (ZH)**: 离线多智能体强化学习（MARL）受到评估流出分布（OOD）联合动作的挑战严重影响。我们的核心发现是在数据收集过程中智能体表现出完全或部分独立行为时，部分动作替换（PAR）策略可以显著缓解这一挑战。PAR更新单个或部分智能体的动作，而其余动作保持在行为数据中，相比联合动作的完全更新，可减少分布偏移。基于这一洞察，我们开发了软部分保守Q学习（SPaCQL），使用PAR来缓解OOD问题，并根据价值估计的不确定性动态加权不同PAR策略。我们为这一方法提供了严格的理论基础，证明在因素分解行为策略下，诱导的分布偏移随偏离智能体数量线性变化，而非联合动作空间指数变化。这为这一重要的离线MARL问题提供了一个可证明更紧的价值误差界。我们的理论结果还表明，SPaCQL通过不确定性驱动的加权方式适应性地应对分布偏移。我们的实验结果证明，SPaCQL能够更有效地进行策略学习，并在离线数据集表现出独立结构时，显著优于基线算法。 

---
# One Router to Route Them All: Homogeneous Expert Routing for Heterogeneous Graph Transformers 

**Title (ZH)**: 一个路由器兼 route 所有模型： homogenous 专家路由机制用于异构图 transformer 

**Authors**: Georgiy Shakirov, Albert Arakelov  

**Link**: [PDF](https://arxiv.org/pdf/2511.07603)  

**Abstract**: A common practice in heterogeneous graph neural networks (HGNNs) is to condition parameters on node/edge types, assuming types reflect semantic roles. However, this can cause overreliance on surface-level labels and impede cross-type knowledge transfer. We explore integrating Mixture-of-Experts (MoE) into HGNNs--a direction underexplored despite MoE's success in homogeneous settings. Crucially, we question the need for type-specific experts. We propose Homogeneous Expert Routing (HER), an MoE layer for Heterogeneous Graph Transformers (HGT) that stochastically masks type embeddings during routing to encourage type-agnostic specialization. Evaluated on IMDB, ACM, and DBLP for link prediction, HER consistently outperforms standard HGT and a type-separated MoE baseline. Analysis on IMDB shows HER experts specialize by semantic patterns (e.g., movie genres) rather than node types, confirming routing is driven by latent semantics. Our work demonstrates that regularizing type dependence in expert routing yields more generalizable, efficient, and interpretable representations--a new design principle for heterogeneous graph learning. 

**Abstract (ZH)**: 一种常见的异质图神经网络（HGNN）做法是在参数上条件化节点/边类型，假设类型反映语义角色。然而，这可能导致对表面标签的过度依赖，并阻碍不同类型之间的知识转移。我们探讨将Mixture-of-Experts（MoE）集成到HGNN中——尽管MoE在同质设置中取得了成功，但在异质设置中的应用仍未充分探索。关键的是，我们质疑是否需要类型特定的专家。我们提出了Homogeneous Expert Routing（HER），一种适用于异质图变换器（HGT）的MoE层，在路由过程中随机遮蔽类型嵌入以促进类型无关的专门化。在IMDB、ACM和DBLP上的链路预测评估中，HER始终优于标准HGT和类型分离的MoE基线。IMDB上的分析显示，HER专家通过语义模式（例如，电影类型）进行专门化，而不是节点类型，证实了路由受潜在语义驱动。我们的工作证明，正则化专家路由中的类型依赖能够产生更通用、更高效和更可解释的表示——这是异质图学习的一种新设计原则。 

---
# Leveraging the Power of AI and Social Interactions to Restore Trust in Public Polls 

**Title (ZH)**: 利用AI和社交互动的力量恢复公众民意调查中的信任 

**Authors**: Amr Akmal Abouelmagd, Amr Hilal  

**Link**: [PDF](https://arxiv.org/pdf/2511.07593)  

**Abstract**: The emergence of crowdsourced data has significantly reshaped social science, enabling extensive exploration of collective human actions, viewpoints, and societal dynamics. However, ensuring safe, fair, and reliable participation remains a persistent challenge. Traditional polling methods have seen a notable decline in engagement over recent decades, raising concerns about the credibility of collected data. Meanwhile, social and peer-to-peer networks have become increasingly widespread, but data from these platforms can suffer from credibility issues due to fraudulent or ineligible participation. In this paper, we explore how social interactions can help restore credibility in crowdsourced data collected over social networks. We present an empirical study to detect ineligible participation in a polling task through AI-based graph analysis of social interactions among imperfect participants composed of honest and dishonest actors. Our approach focuses solely on the structure of social interaction graphs, without relying on the content being shared. We simulate different levels and types of dishonest behavior among participants who attempt to propagate the task within their social networks. We conduct experiments on real-world social network datasets, using different eligibility criteria and modeling diverse participation patterns. Although structural differences in social interaction graphs introduce some performance variability, our study achieves promising results in detecting ineligibility across diverse social and behavioral profiles, with accuracy exceeding 90% in some configurations. 

**Abstract (ZH)**: crowdsourced数据的兴起显著重塑了社会科学， Enable了对集体人类行为、观点和社会动态的广泛探索。然而，确保安全、公平和可靠的参与依旧是一个持久的挑战。传统的民意调查方法在近年来的参与度显著下降，引发了对收集数据可信度的担忧。同时，社交和去中心化网络的应用越来越广泛，但这些平台的数据可能会因欺诈或不合格参与而出现可信度问题。在本文中，我们探讨了如何通过基于人工智能的社交互动图分析来恢复社交网络中收集的 crowdsourced 数据的可信度。我们通过实证研究来检测一项民意调查任务中的不合格参与，这些互动包含了诚实和不诚实的行为者。我们的方法依赖于社交互动图的结构，而不涉及所共享的内容。我们模拟了不同水平和类型的不诚实行为，这些行为在参与者试图在其社交网络中传播任务时发生。我们使用不同的合格标准在真实世界的社交网络数据集上进行了实验，并对多种参与模式进行了建模。尽管社交互动图结构的差异引入了一定的性能变异性，但我们的研究在多种社会和行为配置中取得了令人鼓舞的结果，某些配置中的准确率超过90%。 

---
# SemanticForge: Repository-Level Code Generation through Semantic Knowledge Graphs and Constraint Satisfaction 

**Title (ZH)**: SemanticsForge：通过语义知识图和约束满足实现仓库级别代码生成 

**Authors**: Wuyang Zhang, Chenkai Zhang, Zhen Luo, Jianming Ma, Wangming Yuan, Chuqiao Gu, Chenwei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2511.07584)  

**Abstract**: Large language models (LLMs) have transformed software development by enabling automated code generation, yet they frequently suffer from systematic errors that limit practical deployment. We identify two critical failure modes: \textit{logical hallucination} (incorrect control/data-flow reasoning) and \textit{schematic hallucination} (type mismatches, signature violations, and architectural inconsistencies). These errors stem from the absence of explicit, queryable representations of repository-wide semantics.
This paper presents \textbf{SemanticForge}, which introduces four fundamental algorithmic advances for semantically-aware code generation: (1) a novel automatic reconciliation algorithm for dual static-dynamic knowledge graphs, unifying compile-time and runtime program semantics; (2) a neural approach that learns to generate structured graph queries from natural language, achieving 73\% precision versus 51\% for traditional retrieval; (3) a novel beam search algorithm with integrated SMT solving, enabling real-time constraint verification during generation rather than post-hoc validation; and (4) an incremental maintenance algorithm that updates knowledge graphs in $O(|\Delta R| \cdot \log n)$ time while maintaining semantic equivalence. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过启用自动化代码生成已经改变了软件开发，但它们经常遭受系统性错误的困扰，限制了其实用部署。我们识别出两种关键失败模式：逻辑幻觉（错误的控制/数据流推理）和模式幻觉（类型不匹配、签名违反和架构不一致性）。这些错误源于仓库范围 semanticforge：大型语言模型（LLMs）通过启用自动化代码生成已经改变了软件开发，但它们经常遭受系统性错误的困扰，限制了其实用部署。我们识别出两种关键失败模式：逻辑幻觉（错误的控制/数据流推理）和模式幻觉（类型不匹配、签名违反和架构不一致性）。这些错误源于仓库范围的语义缺乏显式的可查询表示。本文提出了 SemanticForge，它引入了四种基本的算法进步，以实现语义意识的代码生成：（1）一种新的自动统一编译时和运行时程序语义的双重静态-动态知识图谱的自动协调算法；（2）一种神经方法，通过自然语言学习生成结构化图查询，其精准度为73%，而传统检索仅为51%；（3）一种新的束搜索算法，结合了SMT求解器，使生成过程中可以实时验证约束，而不是在生成后进行验证；（4）一种增量维护算法，在O(|ΔR|·log n)时间复杂度内更新知识图谱，同时保持语义等价性。 

---
# N-ReLU: Zero-Mean Stochastic Extension of ReLU 

**Title (ZH)**: N-ReLU: 零均值随机扩展的ReLU 

**Authors**: Md Motaleb Hossen Manik, Md Zabirul Islam, Ge Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07559)  

**Abstract**: Activation functions are fundamental for enabling nonlinear representations in deep neural networks. However, the standard rectified linear unit (ReLU) often suffers from inactive or "dead" neurons caused by its hard zero cutoff. To address this issue, we introduce N-ReLU (Noise-ReLU), a zero-mean stochastic extension of ReLU that replaces negative activations with Gaussian noise while preserving the same expected output. This expectation-aligned formulation maintains gradient flow in inactive regions and acts as an annealing-style regularizer during training. Experiments on the MNIST dataset using both multilayer perceptron (MLP) and convolutional neural network (CNN) architectures show that N-ReLU achieves accuracy comparable to or slightly exceeding that of ReLU, LeakyReLU, PReLU, GELU, and RReLU at moderate noise levels (sigma = 0.05-0.10), with stable convergence and no dead neurons observed. These results demonstrate that lightweight Gaussian noise injection offers a simple yet effective mechanism to enhance optimization robustness without modifying network structures or introducing additional parameters. 

**Abstract (ZH)**: 噪声ReLU：零均值随机扩展以激活神经网络中的非线性表示 

---
# FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models 

**Title (ZH)**: FedRW：高效的数据重加权方法以增强语言模型的联邦学习中的隐私保护 

**Authors**: Pukang Ye, Junwei Luo, Xiaolei Dong, Yunbo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07505)  

**Abstract**: Data duplication within large-scale corpora often impedes large language models' (LLMs) performance and privacy. In privacy-concerned federated learning scenarios, conventional deduplication methods typically rely on trusted third parties to perform uniform deletion, risking loss of informative samples while introducing privacy vulnerabilities. To address these gaps, we propose Federated ReWeighting (FedRW), the first privacy-preserving framework, to the best of our knowledge, that performs soft deduplication via sample reweighting instead of deletion in federated LLM training, without assuming a trusted third party. At its core, FedRW proposes a secure, frequency-aware reweighting protocol through secure multi-party computation, coupled with a parallel orchestration strategy to ensure efficiency and scalability. During training, FedRW utilizes an adaptive reweighting mechanism with global sample frequencies to adjust individual loss contributions, effectively improving generalization and robustness. Empirical results demonstrate that FedRW outperforms the state-of-the-art method by achieving up to 28.78x speedup in preprocessing and approximately 11.42% improvement in perplexity, while offering enhanced security guarantees. FedRW thus establishes a new paradigm for managing duplication in federated LLM training. 

**Abstract (ZH)**: 联邦重权分配：一种无需信任第三方的隐私保护软去重框架 

---
# Biologically-Informed Hybrid Membership Inference Attacks on Generative Genomic Models 

**Title (ZH)**: 生物启发的混合成员推断攻击针对生成基因组模型 

**Authors**: Asia Belfiore, Jonathan Passerat-Palmbach, Dmitrii Usynin  

**Link**: [PDF](https://arxiv.org/pdf/2511.07503)  

**Abstract**: The increased availability of genetic data has transformed genomics research, but raised many privacy concerns regarding its handling due to its sensitive nature. This work explores the use of language models (LMs) for the generation of synthetic genetic mutation profiles, leveraging differential privacy (DP) for the protection of sensitive genetic data. We empirically evaluate the privacy guarantees of our DP modes by introducing a novel Biologically-Informed Hybrid Membership Inference Attack (biHMIA), which combines traditional black box MIA with contextual genomics metrics for enhanced attack power. Our experiments show that both small and large transformer GPT-like models are viable synthetic variant generators for small-scale genomics, and that our hybrid attack leads, on average, to higher adversarial success compared to traditional metric-based MIAs. 

**Abstract (ZH)**: 增加的遗传数据可用性推动了基因组研究的发展，但由于其敏感性，也引发了对其处理的许多隐私问题。本文探讨了利用语言模型（LMs）生成合成遗传突变谱的方法，并利用差分隐私（DP）保护敏感遗传数据。我们通过引入一种新的生物信息学启发式混合成员推理攻击（biHMIA），结合传统的黑盒MIA和上下文基因组指标，来实证评估我们的DP模式下的隐私保证。实验结果表明，无论是小型还是大型变压器GPT-like模型都适用于小型基因组中的合成变异生成，并且我们的混合攻击在平均情况下比传统的基于度量的MIA具有更高的对抗成功率。 

---
# Laplacian Score Sharpening for Mitigating Hallucination in Diffusion Models 

**Title (ZH)**: Laplacian评分增强以减轻扩散模型中的幻觉 

**Authors**: Barath Chandran.C, Srinivas Anumasa, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07496)  

**Abstract**: Diffusion models, though successful, are known to suffer from hallucinations that create incoherent or unrealistic samples. Recent works have attributed this to the phenomenon of mode interpolation and score smoothening, but they lack a method to prevent their generation during sampling. In this paper, we propose a post-hoc adjustment to the score function during inference that leverages the Laplacian (or sharpness) of the score to reduce mode interpolation hallucination in unconditional diffusion models across 1D, 2D, and high-dimensional image data. We derive an efficient Laplacian approximation for higher dimensions using a finite-difference variant of the Hutchinson trace estimator. We show that this correction significantly reduces the rate of hallucinated samples across toy 1D/2D distributions and a high- dimensional image dataset. Furthermore, our analysis explores the relationship between the Laplacian and uncertainty in the score. 

**Abstract (ZH)**: 扩散模型虽然取得了成功，但由于会出现幻觉现象而产生不连贯或不现实的样本。现有研究将这一现象归因于模式内插和分数光滑化，但缺乏在采样过程中防止生成这些幻觉的方法。本文提出了一种后验调整方法，在推理过程中针对分数函数利用分数的拉普拉斯（或锐度）来减少无条件扩散模型在1D、2D和高维图像数据中的模式内插幻觉。我们通过有限差分变体的Hutchinson迹估计推导出高维条件下的高效拉普拉斯近似。我们表明，这种修正可以显著减少玩具1D/2D分布以及高维图像数据集中幻觉样本的生成频率。此外，我们的分析探讨了拉普拉斯与分数不确定性之间的关系。 

---
# Enabling Automatic Self-Talk Detection via Earables 

**Title (ZH)**: 通过可穿戴设备实现自动自我对话检测 

**Authors**: Euihyeok Lee, Seonghyeon Kim, SangHun Im, Heung-Seon Oh, Seungwoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07493)  

**Abstract**: Self-talk-an internal dialogue that can occur silently or be spoken aloud-plays a crucial role in emotional regulation, cognitive processing, and motivation, yet has remained largely invisible and unmeasurable in everyday life. In this paper, we present MutterMeter, a mobile system that automatically detects vocalized self-talk from audio captured by earable microphones in real-world settings. Detecting self-talk is technically challenging due to its diverse acoustic forms, semantic and grammatical incompleteness, and irregular occurrence patterns, which differ fundamentally from assumptions underlying conventional speech understanding models. To address these challenges, MutterMeter employs a hierarchical classification architecture that progressively integrates acoustic, linguistic, and contextual information through a sequential processing pipeline, adaptively balancing accuracy and computational efficiency. We build and evaluate MutterMeter using a first-of-its-kind dataset comprising 31.1 hours of audio collected from 25 participants. Experimental results demonstrate that MutterMeter achieves robust performance with a macro-averaged F1 score of 0.84, outperforming conventional approaches, including LLM-based and speech emotion recognition models. 

**Abstract (ZH)**: 自言自语——一种可以默念或出声发生的内部对话——在情绪调节、认知处理和动机中扮演着关键角色，但在日常生活中仍 largely invisible 和 unmeasurable。在本文中，我们介绍了 MutterMeter，这是一种移动系统，能够自动检测耳佩戴麦克风在外场记录的音频中所发出的自言自语。检测自言自语在技术上具有挑战性，因为它具有多样的声学形式、语义和句法上的不完整性以及不规则的出现模式，这些都与传统语音理解模型的基本假设存在根本差异。为应对这些挑战，MutterMeter 采用了一种分层分类架构，通过顺序处理管道逐步整合声学、语言和上下文信息，动态平衡准确性和计算效率。我们使用一个前所未有的数据集构建和评估了 MutterMeter，该数据集包含 25 名参与者提供的 31.1 小时音频。实验结果表明，MutterMeter 获得了稳健的表现，宏均 F1 得分为 0.84，优于传统方法，包括基于大语言模型和语音情感识别模型的方法。 

---
# When Are Learning Biases Equivalent? A Unifying Framework for Fairness, Robustness, and Distribution Shift 

**Title (ZH)**: 学习偏差何时等价？一个统一的公平性、鲁棒性与分布偏移框架 

**Authors**: Sushant Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2511.07485)  

**Abstract**: Machine learning systems exhibit diverse failure modes: unfairness toward protected groups, brittleness to spurious correlations, poor performance on minority sub-populations, which are typically studied in isolation by distinct research communities. We propose a unifying theoretical framework that characterizes when different bias mechanisms produce quantitatively equivalent effects on model performance. By formalizing biases as violations of conditional independence through information-theoretic measures, we prove formal equivalence conditions relating spurious correlations, subpopulation shift, class imbalance, and fairness violations. Our theory predicts that a spurious correlation of strength $\alpha$ produces equivalent worst-group accuracy degradation as a sub-population imbalance ratio $r \approx (1+\alpha)/(1-\alpha)$ under feature overlap assumptions. Empirical validation in six datasets and three architectures confirms that predicted equivalences hold within the accuracy of the worst group 3\%, enabling the principled transfer of debiasing methods across problem domains. This work bridges the literature on fairness, robustness, and distribution shifts under a common perspective. 

**Abstract (ZH)**: 机器学习系统表现出多样化的失败模式：对受保护群体的不公平、对虚假相关性的脆弱性、在少数子群体中的性能不佳，这些通常由不同的研究社区孤立研究。我们提出了一种统一的理论框架，以描述不同的偏见机制在模型性能上产生定量等效效应的条件。通过使用信息论度量将偏见形式化为条件独立性的违反，我们证明了虚假相关性、子群体转移、类别不平衡和公平性违反之间的形式等价条件。我们的理论预测，假设特征重叠的情况下，强度为α的虚假相关性导致最差群体准确率降级效果等同于子群体不平衡比率为r≈(1+α)/(1-α)。在六个数据集和三种架构上的实证验证确认了预测的等价性在最差群体准确率3%的范围内成立，从而允许在不同的问题域中根据原则转移去偏方法。这项工作在共同的视角下弥合了公平性、鲁棒性和分布转移的研究文献。 

---
# KG-DF: A Black-box Defense Framework against Jailbreak Attacks Based on Knowledge Graphs 

**Title (ZH)**: KG-DF：基于知识图谱的黑盒防御框架对抗 Jailbreak 攻击 

**Authors**: Shuyuan Liu, Jiawei Chen, Xiao Yang, Hang Su, Zhaoxia Yin  

**Link**: [PDF](https://arxiv.org/pdf/2511.07480)  

**Abstract**: With the widespread application of large language models (LLMs) in various fields, the security challenges they face have become increasingly prominent, especially the issue of jailbreak. These attacks induce the model to generate erroneous or uncontrolled outputs through crafted inputs, threatening the generality and security of the model. Although existing defense methods have shown some effectiveness, they often struggle to strike a balance between model generality and security. Excessive defense may limit the normal use of the model, while insufficient defense may lead to security vulnerabilities. In response to this problem, we propose a Knowledge Graph Defense Framework (KG-DF). Specifically, because of its structured knowledge representation and semantic association capabilities, Knowledge Graph(KG) can be searched by associating input content with safe knowledge in the knowledge base, thus identifying potentially harmful intentions and providing safe reasoning paths. However, traditional KG methods encounter significant challenges in keyword extraction, particularly when confronted with diverse and evolving attack strategies. To address this issue, we introduce an extensible semantic parsing module, whose core task is to transform the input query into a set of structured and secure concept representations, thereby enhancing the relevance of the matching process. Experimental results show that our framework enhances defense performance against various jailbreak attack methods, while also improving the response quality of the LLM in general QA scenarios by incorporating domain-general knowledge. 

**Abstract (ZH)**: 基于知识图谱的防御框架（KG-DF）：应对大语言模型的脱逃攻击 

---
# Optimizing Classification of Infrequent Labels by Reducing Variability in Label Distribution 

**Title (ZH)**: 通过减少标签分布的变异性来优化罕见标签的分类 

**Authors**: Ashutosh Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2511.07459)  

**Abstract**: This paper presents a novel solution, LEVER, designed to address the challenges posed by underperforming infrequent categories in Extreme Classification (XC) tasks. Infrequent categories, often characterized by sparse samples, suffer from high label inconsistency, which undermines classification performance. LEVER mitigates this problem by adopting a robust Siamese-style architecture, leveraging knowledge transfer to reduce label inconsistency and enhance the performance of One-vs-All classifiers. Comprehensive testing across multiple XC datasets reveals substantial improvements in the handling of infrequent categories, setting a new benchmark for the field. Additionally, the paper introduces two newly created multi-intent datasets, offering essential resources for future XC research. 

**Abstract (ZH)**: 本文提出了一种名为LEVER的新型解决方案，旨在解决极分类（XC）任务中表现不佳的少见类别的挑战。少见类别通常以稀疏样本为特征，导致标签不一致问题，从而影响分类性能。LEVER通过采用一种鲁棒的Siamese风格架构，利用知识迁移减少标签不一致并增强One-vs-All分类器的表现。在多个XC数据集上的全面测试显示，在处理少见类别方面取得了显著改进，并为该领域设立了新的性能基准。此外，本文还介绍了两个新创建的多意图数据集，为未来的XC研究提供了宝贵的资源。 

---
# A Preliminary Study of RAG for Taiwanese Historical Archives 

**Title (ZH)**: 台裔历史档案中RAG的初步研究 

**Authors**: Claire Lin, Bo-Han Feng, Xuanjun Chen, Te-Lun Yang, Hung-yi Lee, Jyh-Shing Roger Jang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07445)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising approach for knowledge-intensive tasks. However, few studies have examined RAG for Taiwanese Historical Archives. In this paper, we present an initial study of a RAG pipeline applied to two historical Traditional Chinese datasets, Fort Zeelandia and the Taiwan Provincial Council Gazette, along with their corresponding open-ended query sets. We systematically investigate the effects of query characteristics and metadata integration strategies on retrieval quality, answer generation, and the performance of the overall system. The results show that early-stage metadata integration enhances both retrieval and answer accuracy while also revealing persistent challenges for RAG systems, including hallucinations during generation and difficulties in handling temporal or multi-hop historical queries. 

**Abstract (ZH)**: 检索增强生成（RAG）已成为知识密集型任务的一种有前途的方法。然而，few studies have examined RAG for Taiwanese Historical Archives。本文探讨了RAG管道在两个历史 Traditional Chinese 数据集，Fort Zeelandia和台湾省参议会公报，及其相应的开放查询集上的应用。系统研究了查询特征和元数据集成策略对检索质量、答案生成以及整体系统性能的影响。结果显示，早期阶段的元数据集成可以同时提高检索和答案的准确性，同时也揭示了RAG系统中持续存在的挑战，包括生成过程中的幻觉以及处理时间性和多跳历史查询的困难。 

---
# Pinching Antennas Meet AI in Next-Generation Wireless Networks 

**Title (ZH)**: -pinching天线遇见AI在下一代无线网络中的应用- 

**Authors**: Fang Fang, Zhiguo Ding, Victor C. M. Leung, Lajos Hanzo  

**Link**: [PDF](https://arxiv.org/pdf/2511.07442)  

**Abstract**: Next-generation (NG) wireless networks must embrace innate intelligence in support of demanding emerging applications, such as extended reality and autonomous systems, under ultra-reliable and low-latency requirements. Pinching antennas (PAs), a new flexible low-cost technology, can create line-of-sight links by dynamically activating small dielectric pinches along a waveguide on demand. As a compelling complement, artificial intelligence (AI) offers the intelligence needed to manage the complex control of PA activation positions and resource allocation in these dynamic environments. This article explores the "win-win" cooperation between AI and PAs: AI facilitates the adaptive optimization of PA activation positions along the waveguide, while PAs support edge AI tasks such as federated learning and over-the-air aggregation. We also discuss promising research directions including large language model-driven PA control frameworks, and how PA-AI integration can advance semantic communications, and integrated sensing and communication. This synergy paves the way for adaptive, resilient, and self-optimizing NG networks. 

**Abstract (ZH)**: 下一代无线网络必须集成内在智能以支持严苛的新兴应用，如扩展现实和自主系统，并满足超可靠低延迟的要求。可调节天线（PAs），一种新型灵活低成本技术，可以通过按需动态激活波导沿线的小介电椒盐结构来创建视距链路。作为有力补充，人工智能（AI）提供了管理PA激活位置复杂控制和资源分配所需的能力，在这些动态环境中。本文探讨了AI与PAs之间的“双赢”合作：AI促进波导沿线可适应的PA激活位置优化，而PAs支持边缘AI任务，如联邦学习和空中聚合。我们还讨论了包括以大型语言模型驱动的PA控制框架在内的有前途的研究方向，以及PA-AI集成如何推动语义通信和集成传感与通信的发展。这种协同作用为自适应、抗毁性和自优化的下一代网络铺平了道路。 

---
# Benchmarking Simulacra AI's Quantum Accurate Synthetic Data Generation for Chemical Sciences 

**Title (ZH)**: Simulacra AI的量子精确合成数据生成在化学科学中的基准测试 

**Authors**: Fabio Falcioni, Elena Orlova, Timothy Heightman, Philip Mantrov, Aleksei Ustimenko  

**Link**: [PDF](https://arxiv.org/pdf/2511.07433)  

**Abstract**: In this work, we benchmark \simulacra's synthetic data generation pipeline against a state-of-the-art Microsoft pipeline on a dataset of small to large systems. By analyzing the energy quality, autocorrelation times, and effective sample size, our findings show that Simulacra's Large Wavefunction Models (LWM) pipeline, paired with state-of-the-art Variational Monte Carlo (VMC) sampling algorithms, reduces data generation costs by 15-50x, while maintaining parity in energy accuracy, and 2-3x compared to traditional CCSD methods on the scale of amino acids. This enables the creation of affordable, large-scale \textit{ab-initio} datasets, accelerating AI-driven optimization and discovery in the pharmaceutical industry and beyond. Our improvements are based on a novel and proprietary sampling scheme called Replica Exchange with Langevin Adaptive eXploration (RELAX). 

**Abstract (ZH)**: 本研究将Simulacra的合成数据生成管道与最先进的Microsoft管道在小到大规模系统数据集上进行了基准测试。通过分析能量质量、自相关时间和有效样本大小，我们的研究发现，结合了最先进的变分蒙特卡洛（VMC）采样算法的Simulacra大型波函数模型（LWM）管道，将数据生成成本减少了15-50倍，同时在能量精度方面保持一致，并且在氨基酸规模上比传统CCSD方法快2-3倍。这使得创建可负担得起的大规模从头计算数据集成为可能，从而加速了制药行业及其他领域的AI驱动的优化和发现。我们的改进基于一种新颖的专有采样方案，称为Replica Exchange with Langevin Adaptive eXploration（RELAX）。 

---
# Hybrid Bit and Semantic Communications 

**Title (ZH)**: 混合位级和语义通信 

**Authors**: Kaiwen Yu, Renhe Fan, Gang Wu, Zhijin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2404.19477)  

**Abstract**: Semantic communication technology is regarded as a method surpassing the Shannon limit of bit transmission, capable of effectively enhancing transmission efficiency. However, current approaches that directly map content to transmission symbols are challenging to deploy in practice, imposing significant limitations on the development of semantic communication. To address this challenge, we propose a hybrid bit and semantic communication system, named HybridBSC, in which encoded semantic information is inserted into bit information for transmission via conventional digital communication systems utilizing same spectrum resources. The system can be easily deployed using existing communication architecture to achieve bit and semantic information transmission. Particularly, we design a semantic insertion and extraction scheme to implement this strategy. Furthermore, we conduct experimental validation based on the pluto-based software defined radio (SDR) platform in a real wireless channel, demonstrating that the proposed strategy can simultaneously transmit semantic and bit information. 

**Abstract (ZH)**: 基于比特和语义的混合通信系统：Hybrid Bit and Semantic Communication System 

---
