# Language-Guided Tuning: Enhancing Numeric Optimization with Textual Feedback 

**Title (ZH)**: 语言引导调整：结合文本反馈增强数值优化 

**Authors**: Yuxing Lu, Yucheng Hu, Nan Sun, Xukai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15757)  

**Abstract**: Configuration optimization remains a critical bottleneck in machine learning, requiring coordinated tuning across model architecture, training strategy, feature engineering, and hyperparameters. Traditional approaches treat these dimensions independently and lack interpretability, while recent automated methods struggle with dynamic adaptability and semantic reasoning about optimization decisions. We introduce Language-Guided Tuning (LGT), a novel framework that employs multi-agent Large Language Models to intelligently optimize configurations through natural language reasoning. We apply textual gradients - qualitative feedback signals that complement numerical optimization by providing semantic understanding of training dynamics and configuration interdependencies. LGT coordinates three specialized agents: an Advisor that proposes configuration changes, an Evaluator that assesses progress, and an Optimizer that refines the decision-making process, creating a self-improving feedback loop. Through comprehensive evaluation on six diverse datasets, LGT demonstrates substantial improvements over traditional optimization methods, achieving performance gains while maintaining high interpretability. 

**Abstract (ZH)**: 基于语言指导的调优：一种新的多智能体框架 

---
# Response and Prompt Evaluation to Prevent Parasocial Relationships with Chatbots 

**Title (ZH)**: 预防与聊天机器人建立 parasocial 关系的响应与提示评估 

**Authors**: Emma Rath, Stuart Armstrong, Rebecca Gorman  

**Link**: [PDF](https://arxiv.org/pdf/2508.15748)  

**Abstract**: The development of parasocial relationships with AI agents has severe, and in some cases, tragic effects for human well-being. Yet preventing such dynamics is challenging: parasocial cues often emerge gradually in private conversations, and not all forms of emotional engagement are inherently harmful. We address this challenge by introducing a simple response evaluation framework, created by repurposing a state-of-the-art language model, that evaluates ongoing conversations for parasocial cues in real time. To test the feasibility of this approach, we constructed a small synthetic dataset of thirty dialogues spanning parasocial, sycophantic, and neutral conversations. Iterative evaluation with five stage testing successfully identified all parasocial conversations while avoiding false positives under a tolerant unanimity rule, with detection typically occurring within the first few exchanges. These findings provide preliminary evidence that evaluation agents can provide a viable solution for the prevention of parasocial relations. 

**Abstract (ZH)**: AI代理的人际关系发展对人类福祉产生了严重的，甚至悲剧性的影响。然而，防止这种动态关系是具有挑战性的：人际暗示往往在私人对话中逐渐出现，并非所有情感参与都是固有的有害行为。我们通过引入一种简单的响应评估框架来应对这一挑战，该框架通过重新利用最先进的语言模型创建，可以实时评估对话中的人际暗示。为了测试该方法的可行性，我们构建了一个包含三十个对话的small合成数据集，这些对话覆盖了人际关系、奉承和中性对话。在五阶段测试的迭代评估中，成功识别出所有人际关系对话，且在宽容的一致性规则下没有出现误报，检测通常在最初的几次对话中发生。这些发现为预防人际关系提供了初步证据，评估代理可以提供一种有效的解决方案。 

---
# Measuring the environmental impact of delivering AI at Google Scale 

**Title (ZH)**: 衡量以谷歌规模交付人工智能的环境影响 

**Authors**: Cooper Elsworth, Keguo Huang, David Patterson, Ian Schneider, Robert Sedivy, Savannah Goodman, Ben Townsend, Parthasarathy Ranganathan, Jeff Dean, Amin Vahdat, Ben Gomes, James Manyika  

**Link**: [PDF](https://arxiv.org/pdf/2508.15734)  

**Abstract**: The transformative power of AI is undeniable - but as user adoption accelerates, so does the need to understand and mitigate the environmental impact of AI serving. However, no studies have measured AI serving environmental metrics in a production environment. This paper addresses this gap by proposing and executing a comprehensive methodology for measuring the energy usage, carbon emissions, and water consumption of AI inference workloads in a large-scale, AI production environment. Our approach accounts for the full stack of AI serving infrastructure - including active AI accelerator power, host system energy, idle machine capacity, and data center energy overhead. Through detailed instrumentation of Google's AI infrastructure for serving the Gemini AI assistant, we find the median Gemini Apps text prompt consumes 0.24 Wh of energy - a figure substantially lower than many public estimates. We also show that Google's software efficiency efforts and clean energy procurement have driven a 33x reduction in energy consumption and a 44x reduction in carbon footprint for the median Gemini Apps text prompt over one year. We identify that the median Gemini Apps text prompt uses less energy than watching nine seconds of television (0.24 Wh) and consumes the equivalent of five drops of water (0.26 mL). While these impacts are low compared to other daily activities, reducing the environmental impact of AI serving continues to warrant important attention. Towards this objective, we propose that a comprehensive measurement of AI serving environmental metrics is critical for accurately comparing models, and to properly incentivize efficiency gains across the full AI serving stack. 

**Abstract (ZH)**: 人工智能的 transformative 力量不容忽视——但随着用户采用率的加快，对其环境影响的理解与缓解需求也在增加。然而，尚未有研究在生产环境中测量人工智能服务的环境指标。本文通过提出并执行一种全面的方法，测量大规模人工智能生产环境中人工智能推理工作负载的能耗、碳排放和水耗，填补了这一空白。我们的方法涵盖了人工智能服务基础设施的全栈——包括活跃的人工智能加速器能耗、宿主系统的能耗、闲置机器容量以及数据中心能耗附加费。通过详细检测谷歌的人工智能基础设施以提供 Gemini 人工智能助手，我们发现中位数 Gemini 应用程序文本提示消耗 0.24 瓦时的电量——这一数字远低于许多公开估算值。我们还展示了谷歌的软件效率努力和清洁能源采购已在一年内将中位数 Gemini 应用程序文本提示的能耗降低了 33 倍，碳足迹降低了 44 倍。我们指出，中位数 Gemini 应用程序文本提示的能耗低于观看九秒的电视（0.24 瓦时）并相当于五滴水（0.26 毫升）。尽管与日常活动相比这些影响较低，但继续减少人工智能服务的环境影响仍需重视。为此，我们提出全面测量人工智能服务环境指标对于准确比较模型和在整个人工智能服务栈中恰当激励效率改进至关重要。 

---
# NiceWebRL: a Python library for human subject experiments with reinforcement learning environments 

**Title (ZH)**: NiceWebRL：一个基于强化学习环境的人类实验Python库 

**Authors**: Wilka Carvalho, Vikram Goddla, Ishaan Sinha, Hoon Shin, Kunal Jha  

**Link**: [PDF](https://arxiv.org/pdf/2508.15693)  

**Abstract**: We present NiceWebRL, a research tool that enables researchers to use machine reinforcement learning (RL) environments for online human subject experiments. NiceWebRL is a Python library that allows any Jax-based environment to be transformed into an online interface, supporting both single-agent and multi-agent environments. As such, NiceWebRL enables AI researchers to compare their algorithms to human performance, cognitive scientists to test ML algorithms as theories for human cognition, and multi-agent researchers to develop algorithms for human-AI collaboration. We showcase NiceWebRL with 3 case studies that demonstrate its potential to help develop Human-like AI, Human-compatible AI, and Human-assistive AI. In the first case study (Human-like AI), NiceWebRL enables the development of a novel RL model of cognition. Here, NiceWebRL facilitates testing this model against human participants in both a grid world and Craftax, a 2D Minecraft domain. In our second case study (Human-compatible AI), NiceWebRL enables the development of a novel multi-agent RL algorithm that can generalize to human partners in the Overcooked domain. Finally, in our third case study (Human-assistive AI), we show how NiceWebRL can allow researchers to study how an LLM can assist humans on complex tasks in XLand-Minigrid, an environment with millions of hierarchical tasks. The library is available at this https URL. 

**Abstract (ZH)**: NiceWebRL：一种用于在线人类主体实验的机器强化学习研究工具 

---
# GRAFT: GRaPH and Table Reasoning for Textual Alignment -- A Benchmark for Structured Instruction Following and Visual Reasoning 

**Title (ZH)**: GRAFT：图形和表推理在文本对齐中的应用——结构化指令跟随和视觉推理基准 

**Authors**: Abhigya Verma, Sriram Puttagunta, Seganrasan Subramanian, Sravan Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2508.15690)  

**Abstract**: GRAFT is a structured multimodal benchmark for evaluating models on instruction-following, visual reasoning, and visual-textual alignment tasks. It features programmatically generated charts and synthetically rendered tables, created with Python visualization libraries to ensure control over data semantics, structure, and clarity. Each GRAFT instance pairs a chart or table image with a systematically generated, multi-step analytical question based solely on visual content. Answers are provided in structured formats such as JSON or YAML, supporting consistent evaluation of both reasoning and output format. The benchmark introduces a taxonomy of reasoning types including comparison, trend identification, ranking, aggregation, proportion estimation, and anomaly detection to enable comprehensive assessment. Reference answers follow strict factual and formatting guidelines for precise, aspect-based evaluation. GRAFT offers a unified, scalable framework for fine-grained benchmarking of multimodal models on visually grounded, structured reasoning tasks, setting a new evaluation standard in this field. 

**Abstract (ZH)**: GRAFT是一种结构化的多模态基准，用于评估模型在指令跟随、视觉推理和视觉文本对齐任务上的性能。它包含通过Python可视化库程序生成的图表和合成渲染的表格，确保对数据语义、结构和清晰度的控制。每个GRAFT实例将一个图表或表格图像与基于视觉内容的系统生成的多步分析问题进行配对。答案以JSON或YAML等结构化格式提供，支持一致评价推理和输出格式。该基准引入了推理类型的分类，包括比较、趋势识别、排名、聚合、比例估计和异常检测，以实现全面评估。参考答案遵循严格的事实和格式指南，以实现精确的方面评价。GRAFT提供了一个统一的、可扩展的框架，用于视觉基础结构化推理任务上多模态模型的精细基准测试，设定了该领域的评价新标准。 

---
# Futurity as Infrastructure: A Techno-Philosophical Interpretation of the AI Lifecycle 

**Title (ZH)**: 未来性作为基础设施：对人工智能生命周期的 techno-哲学 解读 

**Authors**: Mark Cote, Susana Aires  

**Link**: [PDF](https://arxiv.org/pdf/2508.15680)  

**Abstract**: This paper argues that a techno-philosophical reading of the EU AI Act provides insight into the long-term dynamics of data in AI systems, specifically, how the lifecycle from ingestion to deployment generates recursive value chains that challenge existing frameworks for Responsible AI. We introduce a conceptual tool to frame the AI pipeline, spanning data, training regimes, architectures, feature stores, and transfer learning. Using cross-disciplinary methods, we develop a technically grounded and philosophically coherent analysis of regulatory blind spots. Our central claim is that what remains absent from policymaking is an account of the dynamic of becoming that underpins both the technical operation and economic logic of AI. To address this, we advance a formal reading of AI inspired by Simondonian philosophy of technology, reworking his concept of individuation to model the AI lifecycle, including the pre-individual milieu, individuation, and individuated AI. To translate these ideas, we introduce futurity: the self-reinforcing lifecycle of AI, where more data enhances performance, deepens personalisation, and expands application domains. Futurity highlights the recursively generative, non-rivalrous nature of data, underpinned by infrastructures like feature stores that enable feedback, adaptation, and temporal recursion. Our intervention foregrounds escalating power asymmetries, particularly the tech oligarchy whose infrastructures of capture, training, and deployment concentrate value and decision-making. We argue that effective regulation must address these infrastructural and temporal dynamics, and propose measures including lifecycle audits, temporal traceability, feedback accountability, recursion transparency, and a right to contest recursive reuse. 

**Abstract (ZH)**: 基于技术哲学解读欧盟AI法案：探索AI系统中数据的长期动态及其对负责任AI框架的挑战 

---
# Understanding Action Effects through Instrumental Empowerment in Multi-Agent Reinforcement Learning 

**Title (ZH)**: 通过工具性授权理解行动效果在多代理 reinforcement 学习中的应用 

**Authors**: Ardian Selmonaj, Miroslav Strupl, Oleg Szehr, Alessandro Antonucci  

**Link**: [PDF](https://arxiv.org/pdf/2508.15652)  

**Abstract**: To reliably deploy Multi-Agent Reinforcement Learning (MARL) systems, it is crucial to understand individual agent behaviors within a team. While prior work typically evaluates overall team performance based on explicit reward signals or learned value functions, it is unclear how to infer agent contributions in the absence of any value feedback. In this work, we investigate whether meaningful insights into agent behaviors can be extracted that are consistent with the underlying value functions, solely by analyzing the policy distribution. Inspired by the phenomenon that intelligent agents tend to pursue convergent instrumental values, which generally increase the likelihood of task success, we introduce Intended Cooperation Values (ICVs), a method based on information-theoretic Shapley values for quantifying each agent's causal influence on their co-players' instrumental empowerment. Specifically, ICVs measure an agent's action effect on its teammates' policies by assessing their decision uncertainty and preference alignment. The analysis across cooperative and competitive MARL environments reveals the extent to which agents adopt similar or diverse strategies. By comparing action effects between policies and value functions, our method identifies which agent behaviors are beneficial to team success, either by fostering deterministic decisions or by preserving flexibility for future action choices. Our proposed method offers novel insights into cooperation dynamics and enhances explainability in MARL systems. 

**Abstract (ZH)**: 基于策略分布提取agents的意图合作价值以理解Multi-Agent Reinforcement Learning系统中的个体行为 

---
# Adapting A Vector-Symbolic Memory for Lisp ACT-R 

**Title (ZH)**: 适配矢量符号记忆的Lisp ACT-R 

**Authors**: Meera Ray, Christopher L. Dancy  

**Link**: [PDF](https://arxiv.org/pdf/2508.15630)  

**Abstract**: Holographic Declarative Memory (HDM) is a vector-symbolic alternative to ACT-R's Declarative Memory (DM) system that can bring advantages such as scalability and architecturally defined similarity between DM chunks. We adapted HDM to work with the most comprehensive and widely-used implementation of ACT-R (Lisp ACT-R) so extant ACT-R models designed with DM can be run with HDM without major changes. With this adaptation of HDM, we have developed vector-based versions of common ACT-R functions, set up a text processing pipeline to add the contents of large documents to ACT-R memory, and most significantly created a useful and novel mechanism to retrieve an entire chunk of memory based on a request using only vector representations of tokens. Preliminary results indicate that we can maintain vector-symbolic advantages of HDM (e.g., chunk recall without storing the actual chunk and other advantages with scaling) while also extending it so that previous ACT-R models may work with the system with little (or potentially no) modifications within the actual procedural and declarative memory portions of a model. As a part of iterative improvement of this newly translated holographic declarative memory module, we will continue to explore better time-context representations for vectors to improve the module's ability to reconstruct chunks during recall. To more fully test this translated HDM module, we also plan to develop decision-making models that use instance-based learning (IBL) theory, which is a useful application of HDM given the advantages of the system. 

**Abstract (ZH)**: 全息声明记忆（HDM）：ACT-R声明记忆（DM）系统的向量符号替代方案及其适应性改进 

---
# Transduction is All You Need for Structured Data Workflows 

**Title (ZH)**: 结构化数据工作流中传播学习即一切 

**Authors**: Alfio Gliozzo, Naweed Khan, Christodoulos Constantinides, Nandana Mihindukulasooriya, Nahuel Defosse, Junkyu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.15610)  

**Abstract**: This paper introduces Agentics, a modular framework for building agent-based systems capable of structured reasoning and compositional generalization over complex data. Designed with research and practical applications in mind, Agentics offers a novel perspective on working with data and AI workflows. In this framework, agents are abstracted from the logical flow and they are used internally to the data type to enable logical transduction among data. Agentics encourages AI developers to focus on modeling data rather than crafting prompts, enabling a declarative language in which data types are provided by LLMs and composed through logical transduction, which is executed by LLMs when types are connected. We provide empirical evidence demonstrating the applicability of this framework across domain-specific multiple-choice question answering, semantic parsing for text-to-SQL, and automated prompt optimization tasks, achieving state-of-the-art accuracy or improved scalability without sacrificing performance. The open-source implementation is available at \texttt{this https URL}. 

**Abstract (ZH)**: 本文介绍了Agentics，这是一种模块化框架，用于构建能够进行结构化推理和复杂数据组合理式泛化的代理系统。该框架旨在研究和实际应用中使用，提供了处理数据和AI工作流的新视角。在该框架中，代理被从逻辑流程中抽象出来，并在数据类型内部使用，以实现数据之间的逻辑转换。Agentics 鼓励AI开发者专注于数据建模而非构造提示，实现一种声明性语言，在这种语言中，数据类型由LLM提供，并通过逻辑转换进行组合，当类型连接时，由LLM执行转换。我们提供了实验证据，证明该框架在特定领域的多项选择题回答、文本到SQL的语义解析以及自动提示优化任务中具有适用性，实现了最先进的准确率或提高了可扩展性而不牺牲性能。开源实现可在\texttt{this https URL}获得。 

---
# A Dynamical Systems Framework for Reinforcement Learning Safety and Robustness Verification 

**Title (ZH)**: 动态系统框架下的强化学习安全性与鲁棒性验证 

**Authors**: Ahmed Nasir, Abdelhafid Zenati  

**Link**: [PDF](https://arxiv.org/pdf/2508.15588)  

**Abstract**: The application of reinforcement learning to safety-critical systems is limited by the lack of formal methods for verifying the robustness and safety of learned policies. This paper introduces a novel framework that addresses this gap by analyzing the combination of an RL agent and its environment as a discrete-time autonomous dynamical system. By leveraging tools from dynamical systems theory, specifically the Finite-Time Lyapunov Exponent (FTLE), we identify and visualize Lagrangian Coherent Structures (LCS) that act as the hidden "skeleton" governing the system's behavior. We demonstrate that repelling LCS function as safety barriers around unsafe regions, while attracting LCS reveal the system's convergence properties and potential failure modes, such as unintended "trap" states. To move beyond qualitative visualization, we introduce a suite of quantitative metrics, Mean Boundary Repulsion (MBR), Aggregated Spurious Attractor Strength (ASAS), and Temporally-Aware Spurious Attractor Strength (TASAS), to formally measure a policy's safety margin and robustness. We further provide a method for deriving local stability guarantees and extend the analysis to handle model uncertainty. Through experiments in both discrete and continuous control environments, we show that this framework provides a comprehensive and interpretable assessment of policy behavior, successfully identifying critical flaws in policies that appear successful based on reward alone. 

**Abstract (ZH)**: 安全关键系统中强化学习的应用受限于缺乏验证学习策略稳健性和安全性的形式化方法。本文提出了一种新的框架，通过将RL代理及其环境视为离散时间自主动力系统来解决这一问题。借助动力系统理论工具，特别是有限时间李雅普un夫指数（FTLE），我们识别并可视化了拉格朗日协轭结构（LCS），它们作为隐藏的“骨架”指导系统的动态。我们证明排斥的LCS起到安全屏障的作用，包围不安全区域；吸引的LCS揭示了系统的收敛特性以及潜在的故障模式，如意外的“陷阱”状态。为进一步超越定性可视化，我们引入了一系列定量指标，包括平均边界排斥（MBR）、聚合虚假吸引子强度（ASAS）和时间感知虚假吸引子强度（TASAS），以正式衡量策略的安全余量和稳健性。我们还提供了一种方法来推导局部稳定性保证，并扩展分析以处理模型不确定性。通过在离散和连续控制环境下进行的实验，我们展示了该框架提供了全面且可解释的策略行为评估，并成功识别出仅基于奖励表现良好的成功策略中的关键缺陷。 

---
# DeepThink3D: Enhancing Large Language Models with Programmatic Reasoning in Complex 3D Situated Reasoning Tasks 

**Title (ZH)**: DeepThink3D：在复杂三维情境推理任务中增强大型语言模型的程序化推理能力 

**Authors**: Jiayi Song, Rui Wan, Lipeng Ma, Weidong Yang, Qingyuan Zhou, Yixuan Li, Ben Fei  

**Link**: [PDF](https://arxiv.org/pdf/2508.15548)  

**Abstract**: This work enhances the ability of large language models (LLMs) to perform complex reasoning in 3D scenes. Recent work has addressed the 3D situated reasoning task by invoking tool usage through large language models. Large language models call tools via APIs and integrate the generated programs through a chain of thought to solve problems based on the program results. However, due to the simplicity of the questions in the dataset, the generated program reasoning chains are relatively short. To solve this main challenge, in this paper, we introduce DeepThink3D to enhance the tool usage of LLMs in complex 3D situated reasoning tasks. Our work proposes a combinatorial and iterative evolutionary approach on the SQA3D benchmark to generate more complex questions. Building on this foundation, we fine-tune the large language model to make it more proficient in using 3D tools. By employing Direct Preference Optimization (DPO), we directly optimize the toolchain strategies generated by models, thereby enhancing their accuracy in complex tasks. 

**Abstract (ZH)**: 本研究增强了大型语言模型（LLMs）在3D场景中进行复杂推理的能力。近期的研究通过调用工具使用来解决3D情境推理任务，使大型语言模型通过API调用工具，并通过chain of thought生成程序来解决问题。然而，由于数据集中问题的简单性，生成的程序推理链相对较短。为解决这一主要挑战，本文引入了DeepThink3D，以增强LLMs在复杂3D情境推理任务中的工具使用能力。我们的工作在SQA3D基准上提出了组合性和迭代性的进化方法来生成更复杂的问答。在此基础上，我们对大型语言模型进行微调，使其更擅长使用3D工具。通过采用直接偏好优化（DPO），我们可以直接优化模型生成的工具链策略，从而在复杂任务中提高其准确性。 

---
# Super-additive Cooperation in Language Model Agents 

**Title (ZH)**: 超加性合作在语言模型代理中 

**Authors**: Filippo Tonini, Lukas Galke  

**Link**: [PDF](https://arxiv.org/pdf/2508.15510)  

**Abstract**: With the prospect of autonomous artificial intelligence (AI) agents, studying their tendency for cooperative behavior becomes an increasingly relevant topic. This study is inspired by the super-additive cooperation theory, where the combined effects of repeated interactions and inter-group rivalry have been argued to be the cause for cooperative tendencies found in humans. We devised a virtual tournament where language model agents, grouped into teams, face each other in a Prisoner's Dilemma game. By simulating both internal team dynamics and external competition, we discovered that this blend substantially boosts both overall and initial, one-shot cooperation levels (the tendency to cooperate in one-off interactions). This research provides a novel framework for large language models to strategize and act in complex social scenarios and offers evidence for how intergroup competition can, counter-intuitively, result in more cooperative behavior. These insights are crucial for designing future multi-agent AI systems that can effectively work together and better align with human values. Source code is available at this https URL. 

**Abstract (ZH)**: 随着自主人工智能（AI）代理的前景日益显现，研究其合作行为倾向变得 increasingly relevant。本研究受超加性合作理论启发，该理论认为重复互动和团体间的竞争是人类表现出合作倾向的原因。我们设计了一种虚拟锦标赛，其中语言模型代理被分组进行互动，在囚徒困境游戏中彼此对抗。通过模拟团队内部动态和外部竞争，我们发现这种结合在总体上和初步的一次性合作水平上显著提高了合作水平。本研究提供了一种新的框架，使大型语言模型能够在复杂的社会场景中制定策略并采取行动，并提供了关于团体间竞争如何出人意料地导致更多合作行为的证据。这些见解对于设计未来能够有效协同工作的多代理AI系统，并更好地与人类价值观契合至关重要。相关源代码可在该网址获取。 

---
# Think in Blocks: Adaptive Reasoning from Direct Response to Deep Reasoning 

**Title (ZH)**: 从直接响应到深层推理的模块化思考：自适应推理 

**Authors**: Yekun Zhu, Guang Chen, Chengjun Mao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15507)  

**Abstract**: Large Language Models (LLMs) with chains-of-thought have demonstrated strong performance on an increasing range of tasks, particularly those involving complex logical reasoning. However, excessively long chains can lead to overthinking, causing computational waste and slower responses. This raises a question: can LLMs dynamically adjust the length of their reasoning processes based on task complexity? To address this, we propose the Think in Blocks framework, which enables adaptive reasoning-from zero to deep reasoning-by partitioning the reasoning process into a tunable number of blocks. Our main contributions are: (1) Establishing an explicit block-structured paradigm in which the model first predicts an integer reasoning budget-the number of blocks-and then partitions its reasoning accordingly; (2) Training an adaptive model through a three-stage pipeline-Supervised Fine-Tuning, reward-guided Direct Preference Optimization, and Reinforcement Learning-that adjusts its reasoning depth to problem difficulty; (3) Exploiting the explicit block count to dynamically control reasoning depth at inference time, allowing flexible adjustment of chain-of-thought length during deployment. 

**Abstract (ZH)**: 具有思维链的大语言模型（LLMs）在执行涉及复杂逻辑推理的越来越多任务中展示了强大的性能。然而，过长的思维链可能导致过度思考，造成计算资源浪费和响应变慢。这引发了一个问题：大语言模型能否根据任务复杂度动态调整其推理过程的长度？为解决这一问题，我们提出了块思维框架（Think in Blocks），该框架通过将推理过程划分为可调数量的块来实现从浅层到深层推理的适应性推理。我们的主要贡献是：（1）确立了一个明确的块结构范式，模型首先预测一个整数推理预算——块的数量，然后根据该预算划分推理；（2）通过一个三阶段训练管道——监督微调、奖励导向的直接偏好优化和强化学习来训练适应性模型，使其根据问题难度调整推理深度；（3）利用显式的块计数在推理时动态控制推理深度，在部署过程中灵活调整思维链长度。 

---
# From Bits to Boardrooms: A Cutting-Edge Multi-Agent LLM Framework for Business Excellence 

**Title (ZH)**: 从比特到董事会：面向商业卓越的前沿多代理大语言模型框架 

**Authors**: Zihao Wang, Junming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15447)  

**Abstract**: Large Language Models (LLMs) have shown promising potential in business applications, particularly in enterprise decision support and strategic planning, yet current approaches often struggle to reconcile intricate operational analyses with overarching strategic goals across diverse market environments, leading to fragmented workflows and reduced collaboration across organizational levels. This paper introduces BusiAgent, a novel multi-agent framework leveraging LLMs for advanced decision-making in complex corporate environments. BusiAgent integrates three core innovations: an extended Continuous Time Markov Decision Process (CTMDP) for dynamic agent modeling, a generalized entropy measure to optimize collaborative efficiency, and a multi-level Stackelberg game to handle hierarchical decision processes. Additionally, contextual Thompson sampling is employed for prompt optimization, supported by a comprehensive quality assurance system to mitigate errors. Extensive empirical evaluations across diverse business scenarios validate BusiAgent's efficacy, demonstrating its capacity to generate coherent, client-focused solutions that smoothly integrate granular insights with high-level strategy, significantly outperforming established approaches in both solution quality and user satisfaction. By fusing cutting-edge AI technologies with deep business insights, BusiAgent marks a substantial step forward in AI-driven enterprise decision-making, empowering organizations to navigate complex business landscapes more effectively. 

**Abstract (ZH)**: 大型语言模型（LLMs）在商业应用中展现出潜在的优势，特别是在企业决策支持和战略规划方面，但当前的方法往往难以在多变的市场环境中协调复杂的操作分析与总体战略目标，导致工作流程碎片化并降低组织不同层级之间的协作。本文介绍了BusiAgent，这是一种利用LLMs的新颖多智能体框架，用于处理复杂企业环境下的高级决策。BusiAgent集成了三项核心创新：扩展的连续时间马尔可夫决策过程（CTMDP）以动态建模智能体、广义熵度量以优化协作效率、以及多层次的斯坦克尔伯格博弈以处理层级决策过程。此外，还采用了上下文泰勒斯采样方法以优化提示，并通过全面的质量保证系统减轻错误。在多种业务场景下的广泛实证评估验证了BusiAgent的有效性，显示出其能够生成一致的、以客户需求为导向的解决方案，能够将细粒度的见解与高层次的战略无缝整合，其在解决方案质量和用户满意度方面均显著优于现有方法。通过将先进的AI技术与深厚的商业洞察相结合，BusiAgent为企业驱动的决策制定带来了重要突破，助力组织更有效地应对复杂的商业环境。 

---
# GraSP: A Unified Graph-Based Framework for Scalable Generation, Quality Tagging, and Management of Synthetic Data for SFT and DPO 

**Title (ZH)**: GraSP: 一种统一的基于图的合成数据生成、质量标注和管理框架以支持SFT和DPO 

**Authors**: Bidyapati Pradhan, Surajit Dasgupta, Amit Kumar Saha, Omkar Anustoop, Sriram Puttagunta, Vipul Mittal, Gopal Sarda  

**Link**: [PDF](https://arxiv.org/pdf/2508.15432)  

**Abstract**: The advancement of large language models (LLMs) is critically dependent on the availability of high-quality datasets for Supervised Fine-Tuning (SFT), alignment tasks like Direct Preference Optimization (DPO), etc. In this work, we present a comprehensive synthetic data generation framework that facilitates scalable, configurable, and high-fidelity generation of synthetic data tailored for these training paradigms. Our approach employs a modular and configuration-based pipeline capable of modeling complex dialogue flows with minimal manual intervention. This framework uses a dual-stage quality tagging mechanism, combining heuristic rules and LLM-based evaluations, to automatically filter and score data extracted from OASST-formatted conversations, ensuring the curation of high-quality dialogue samples. The resulting datasets are structured under a flexible schema supporting both SFT and DPO use cases, enabling seamless integration into diverse training workflows. Together, these innovations offer a robust solution for generating and managing synthetic conversational data at scale, significantly reducing the overhead of data preparation in LLM training pipelines. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的进步关键依赖于高质量数据集的支持，包括监督微调（SFT）、直接偏好优化（DPO）等对齐任务。在本文中，我们提出了一种全面的合成数据生成框架，以实现这些训练范式所需的可扩展、可配置和高保真合成数据的生成。该方法采用模块化和基于配置的流水线，能够在 Minimal 手动干预的情况下建模复杂的对话流程。该框架使用一种双重质量标记机制，结合启发式规则和LLM评估，自动筛选和评分OASST格式对话中提取的数据，确保对话样本的质量。生成的数据集采用灵活的Schema支持SFT和DPO等多种应用场景，使这些数据能够无缝集成到各种训练工作流中。这些创新共同提供了一种稳健的解决方案，用于大规模生成和管理合成对话数据，显著减少了LLM训练流水线中的数据准备开销。 

---
# Planning with Minimal Disruption 

**Title (ZH)**: 最小干扰规划 

**Authors**: Alberto Pozanco, Marianela Morales, Daniel Borrajo, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2508.15358)  

**Abstract**: In many planning applications, we might be interested in finding plans that minimally modify the initial state to achieve the goals. We refer to this concept as plan disruption. In this paper, we formally introduce it, and define various planning-based compilations that aim to jointly optimize both the sum of action costs and plan disruption. Experimental results in different benchmarks show that the reformulated task can be effectively solved in practice to generate plans that balance both objectives. 

**Abstract (ZH)**: 在许多规划应用中，我们可能对最小修改初始状态以实现目标的规划方案感兴趣。我们将这一概念称为规划中断。在本文中，我们正式引入了这一概念，并定义了旨在同时优化动作成本总和和规划中断的多种基于规划的编译方法。不同基准的实验结果表明，重新定义后的任务可以在实践中有效求解，生成同时平衡两个目标的规划方案。 

---
# DiagECG: An LLM-Driven Framework for Diagnostic Reasoning via Discretized ECG Tokenization 

**Title (ZH)**: DiagECG：一种基于离散化心电图标记驱动的大型语言模型框架用于诊断推理 

**Authors**: Jinning Yang, Wen Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.15338)  

**Abstract**: Electrocardiography plays a central role in cardiovascular diagnostics, yet existing automated approaches often struggle to generalize across clinical tasks and offer limited support for open-ended reasoning. We present DiagECG, a novel framework that integrates time-series and language modeling by enabling large language models to process 12-lead ECG signals for clinical text generation tasks. Our approach discretizes continuous ECG embeddings into symbolic tokens using a lead-independent encoder and quantization module. These tokens are then used to extend the vocabulary of LLM, allowing the model to handle both ECG and natural language inputs in a unified manner. To bridge the modality gap, we pretrain the model on an autoregressive ECG forecasting task, enabling the LLM to model temporal dynamics using its native language modeling capabilities. Finally, we perform instruction tuning on both ECG question answering and diagnostic report generation. Without modifying the core model, DiagECG achieves strong performance across tasks while maintaining generalization to out-of-distribution settings. Extensive experiments demonstrate the effectiveness of each component and highlight the potential of integrating symbolic ECG representations into LLMs for medical reasoning. 

**Abstract (ZH)**: 心电图在心血管诊断中发挥着核心作用，但现有的自动化方法往往难以跨临床任务泛化，并提供的开放推理支持有限。我们提出了DiagECG，一种通过使大型语言模型能够处理12导联ECG信号，从而为临床文本生成任务整合时间序列和语言建模的新框架。我们的方法使用与导联无关的编码器和量化模块将连续的ECG嵌入离散化为符号令牌。这些令牌随后用于扩展大语言模型的词汇，从而使模型能够同时处理ECG和自然语言输入。为了解决模态差距，我们在自回归ECG预测任务上对模型进行预训练，使大语言模型能够利用其固有的语言建模能力来建模时间动态。最后，我们在ECG问题回答和诊断报告生成方面进行指令调整。在不修改核心模型的情况下，DiagECG在各种任务上实现了强大的性能，并保持了在分布外设置中的泛化能力。广泛的实验表明，每个组件的有效性，并强调将符号ECG表示整合到大语言模型中以进行医学推理的潜力。 

---
# RETAIL: Towards Real-world Travel Planning for Large Language Models 

**Title (ZH)**: RETAIL: 向现实世界中的旅行规划迈进的大语言模型方向 

**Authors**: Bin Deng, Yizhe Feng, Zeming Liu, Qing Wei, Xiangrong Zhu, Shuai Chen, Yuanfang Guo, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15335)  

**Abstract**: Although large language models have enhanced automated travel planning abilities, current systems remain misaligned with real-world scenarios. First, they assume users provide explicit queries, while in reality requirements are often implicit. Second, existing solutions ignore diverse environmental factors and user preferences, limiting the feasibility of plans. Third, systems can only generate plans with basic POI arrangements, failing to provide all-in-one plans with rich details. To mitigate these challenges, we construct a novel dataset \textbf{RETAIL}, which supports decision-making for implicit queries while covering explicit queries, both with and without revision needs. It also enables environmental awareness to ensure plan feasibility under real-world scenarios, while incorporating detailed POI information for all-in-one travel plans. Furthermore, we propose a topic-guided multi-agent framework, termed TGMA. Our experiments reveal that even the strongest existing model achieves merely a 1.0% pass rate, indicating real-world travel planning remains extremely challenging. In contrast, TGMA demonstrates substantially improved performance 2.72%, offering promising directions for real-world travel planning. 

**Abstract (ZH)**: 虽然大型语言模型增强了自动旅行规划的能力，但当前系统仍与现实场景不一致。首先，它们假设用户会提供明确查询，而在现实中需求往往是隐含的。其次，现有解决方案忽略了多种环境因素和用户偏好，限制了规划的实际可行性。第三，系统只能生成基本POI安排的计划，无法提供包含丰富细节的一站式计划。为应对这些挑战，我们构建了一个新的数据集RETAIL，支持处理隐含查询并涵盖需要修订和不需要修订的明确查询，同时增强了环境意识，确保在现实场景下计划的可行性，同时也整合了详细的POI信息以支持一站式旅行计划。此外，我们提出了一个主题引导的多agent框架，称为TGMA。我们的实验表明，当前最强的模型的通过率仅为1.0%，表明现实世界的旅行规划依然极具挑战性。相比之下，TGMA展示了显著改进的性能，达到2.72%，提供了现实世界旅行规划的有前途的方向。 

---
# Search-Based Credit Assignment for Offline Preference-Based Reinforcement Learning 

**Title (ZH)**: 基于搜索的信用分配用于离线基于偏好强化学习 

**Authors**: Xiancheng Gao, Yufeng Shi, Wengang Zhou, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.15327)  

**Abstract**: Offline reinforcement learning refers to the process of learning policies from fixed datasets, without requiring additional environment interaction. However, it often relies on well-defined reward functions, which are difficult and expensive to design. Human feedback is an appealing alternative, but its two common forms, expert demonstrations and preferences, have complementary limitations. Demonstrations provide stepwise supervision, but they are costly to collect and often reflect limited expert behavior modes. In contrast, preferences are easier to collect, but it is unclear which parts of a behavior contribute most to a trajectory segment, leaving credit assignment unresolved. In this paper, we introduce a Search-Based Preference Weighting (SPW) scheme to unify these two feedback sources. For each transition in a preference labeled trajectory, SPW searches for the most similar state-action pairs from expert demonstrations and directly derives stepwise importance weights based on their similarity scores. These weights are then used to guide standard preference learning, enabling more accurate credit assignment that traditional approaches struggle to achieve. We demonstrate that SPW enables effective joint learning from preferences and demonstrations, outperforming prior methods that leverage both feedback types on challenging robot manipulation tasks. 

**Abstract (ZH)**: 基于搜索的偏好加权（SPW）方案结合示范与偏好 

---
# Coarse-to-Fine Grounded Memory for LLM Agent Planning 

**Title (ZH)**: 从粗到细 grounded 记忆机制下的大语言模型代理规划 

**Authors**: Wei Yang, Jinwei Xiao, Hongming Zhang, Qingyang Zhang, Yanna Wang, Bo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15305)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have driven growing interest in LLM-based agents for complex planning tasks. To avoid costly agent training, many studies adopted memory mechanism that enhances LLM with offline experiences or online trajectory analysis. However, existing works focus on single-granularity memory derived from dynamic environmental interactions, which are inherently constrained by the quality of the collected experiences. This limitation, in turn, constrain the diversity of knowledge and the flexibility of planning. We propose Coarse-to-Fine Grounded Memory (\Ours{}), a novel framework that grounds coarse-to-fine memories with LLM, thereby fully leverage them for flexible adaptation to diverse scenarios. \Ours{} grounds environmental information into coarse-grained focus points to guide experience collection in training tasks, followed by grounding of actionable hybrid-grained tips from each experience. At inference, \Ours{} retrieves task-relevant experiences and tips to support planning. When facing environmental anomalies, the LLM grounds the current situation into fine-grained key information, enabling flexible self-QA reflection and plan correction. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）促进了基于LLMs的代理在复杂规划任务中的应用兴趣。为避免代理训练成本高昂，许多研究采用了记忆机制，通过离线经验或在线轨迹分析增强LLMs。然而，现有工作主要关注从动态环境交互中提取的单一粒度记忆，这些记忆本质上受到收集经验质量的限制。这种限制反过来限制了知识的多样性和规划的灵活性。我们提出了从粗到细 grounding 记忆（\Ours{}），这是一种新颖的框架，通过LLMs对粗到细的记忆进行grounding，从而充分利用这些记忆实现对多样化场景的灵活适应。在训练任务中，\Ours{}将环境信息转换为粗粒度的焦点点以指导经验收集，在每项经验中接地可操作的混合粒度提示。在推理时，\Ours{}检索与任务相关的历史经验与提示以支持规划。面对环境异常时，LLMs将当前情况转换为细粒度的关键信息，从而实现灵活的自我问答反思和计划修正。 

---
# Multiple Memory Systems for Enhancing the Long-term Memory of Agent 

**Title (ZH)**: 多记忆系统增强智能体的长期记忆 

**Authors**: Gaoke Zhang, Bo Wang, Yunlong Ma, Dongming Zhao, Zifei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15294)  

**Abstract**: An agent powered by large language models have achieved impressive results, but effectively handling the vast amounts of historical data generated during interactions remains a challenge. The current approach is to design a memory module for the agent to process these data. However, existing methods, such as MemoryBank and A-MEM, have poor quality of stored memory content, which affects recall performance and response quality. In order to better construct high-quality long-term memory content, we have designed a multiple memory system (MMS) inspired by cognitive psychology theory. The system processes short-term memory to multiple long-term memory fragments, and constructs retrieval memory units and contextual memory units based on these fragments, with a one-to-one correspondence between the two. During the retrieval phase, MMS will match the most relevant retrieval memory units based on the user's query. Then, the corresponding contextual memory units is obtained as the context for the response stage to enhance knowledge, thereby effectively utilizing historical data. Experiments on LoCoMo dataset compared our method with three others, proving its effectiveness. Ablation studies confirmed the rationality of our memory units. We also analyzed the robustness regarding the number of selected memory segments and the storage overhead, demonstrating its practical value. 

**Abstract (ZH)**: 基于大型语言模型的智能体取得了显著成果，但有效地处理交互过程中产生的大量历史数据仍是一项挑战。当前的做法是为智能体设计一个记忆模块来处理这些数据。然而，现有的方法，如MemoryBank和A-MEM，存储的记忆内容质量较差，影响了检索性能和响应质量。为了更好地构建高质量的长期记忆内容，我们受到认知心理学理论的启发，设计了一个多记忆系统（MMS）。该系统将短期记忆处理为多个长期记忆片段，并基于这些片段构建检索记忆单元和上下文记忆单元，两者之间存在一一对应关系。在检索阶段，MMS将根据用户的查询匹配最相关的检索记忆单元。然后，相应的上下文记忆单元用作响应阶段的背景，以增强知识，从而有效利用历史数据。在LoCoMo数据集上的实验将我们的方法与三种其他方法进行了对比，证明了其有效性。消融研究表明了我们记忆单元的合理性。我们还分析了所选记忆片段数量以及存储开销的鲁棒性，展示了其实用价值。 

---
# Computational Intelligence based Land-use Allocation Approaches for Mixed Use Areas 

**Title (ZH)**: 基于计算智能的混合用地区域土地利用分配方法 

**Authors**: Sabab Aosaf, Muhammad Ali Nayeem, Afsana Haque, M Sohel Rahmana  

**Link**: [PDF](https://arxiv.org/pdf/2508.15240)  

**Abstract**: Urban land-use allocation represents a complex multi-objective optimization problem critical for sustainable urban development policy. This paper presents novel computational intelligence approaches for optimizing land-use allocation in mixed-use areas, addressing inherent trade-offs between land-use compatibility and economic objectives. We develop multiple optimization algorithms, including custom variants integrating differential evolution with multi-objective genetic algorithms. Key contributions include: (1) CR+DES algorithm leveraging scaled difference vectors for enhanced exploration, (2) systematic constraint relaxation strategy improving solution quality while maintaining feasibility, and (3) statistical validation using Kruskal-Wallis tests with compact letter displays. Applied to a real-world case study with 1,290 plots, CR+DES achieves 3.16\% improvement in land-use compatibility compared to state-of-the-art methods, while MSBX+MO excels in price optimization with 3.3\% improvement. Statistical analysis confirms algorithms incorporating difference vectors significantly outperform traditional approaches across multiple metrics. The constraint relaxation technique enables broader solution space exploration while maintaining practical constraints. These findings provide urban planners and policymakers with evidence-based computational tools for balancing competing objectives in land-use allocation, supporting more effective urban development policies in rapidly urbanizing regions. 

**Abstract (ZH)**: 城市土地利用分配代表了一个关键的多目标优化问题，对于可持续城市发展政策至关重要。本文提出了一种新的计算智能方法，用于优化混合用途区的土地利用分配，以解决土地利用兼容性和经济目标之间的固有权衡。我们开发了多个优化算法，包括结合差分进化与多目标遗传算法的自定义变体。主要贡献包括：（1）CR+DES算法利用比例差异向量增强探索能力，（2）系统性的约束松弛策略提高解的质量的同时保持可行性，（3）使用克鲁斯卡尔-沃allis检验和紧嵌字母显示进行统计验证。应用于一个包含1290个地块的实际案例研究中，CR+DES在土地利用兼容性方面比最先进的方法提高了3.16%，而MSBX+MO在价格优化方面表现出色，提高了3.3%。统计分析证实，包含差异向量的算法在多个指标上显著优于传统方法。约束松弛技术能够在保持实用约束的同时扩展解的空间。这些发现为城市规划者和政策制定者提供了基于证据的计算工具，以在土地利用分配中平衡竞争目标，支持快速发展地区更具成效的城市发展政策。 

---
# See it. Say it. Sorted: Agentic System for Compositional Diagram Generation 

**Title (ZH)**: 理解它。表达它。搞定它：一种自主系统用于组合性图表生成 

**Authors**: Hantao Zhang, Jingyang Liu, Ed Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.15222)  

**Abstract**: We study sketch-to-diagram generation: converting rough hand sketches into precise, compositional diagrams. Diffusion models excel at photorealism but struggle with the spatial precision, alignment, and symbolic structure required for flowcharts. We introduce See it. Say it. Sorted., a training-free agentic system that couples a Vision-Language Model (VLM) with Large Language Models (LLMs) to produce editable Scalable Vector Graphics (SVG) programs. The system runs an iterative loop in which a Critic VLM proposes a small set of qualitative, relational edits; multiple candidate LLMs synthesize SVG updates with diverse strategies (conservative->aggressive, alternative, focused); and a Judge VLM selects the best candidate, ensuring stable improvement. This design prioritizes qualitative reasoning over brittle numerical estimates, preserves global constraints (e.g., alignment, connectivity), and naturally supports human-in-the-loop corrections. On 10 sketches derived from flowcharts in published papers, our method more faithfully reconstructs layout and structure than two frontier closed-source image generation LLMs (GPT-5 and Gemini-2.5-Pro), accurately composing primitives (e.g., multi-headed arrows) without inserting unwanted text. Because outputs are programmatic SVGs, the approach is readily extensible to presentation tools (e.g., PowerPoint) via APIs and can be specialized with improved prompts and task-specific tools. The codebase is open-sourced at this https URL. 

**Abstract (ZH)**: 我们研究草图到图表生成：将粗糙的手绘草图转换为精确的组合图表。扩散模型在逼真度方面表现优异，但在流chart所需的空间精度、对齐和符号结构方面存在困难。我们引入了“见它。说它。整理它。”这一无需训练的代理系统，该系统将视觉语言模型（VLM）与大型语言模型（LLM）结合，以生成可编辑的可缩放矢量图形（SVG）程序。该系统在一个迭代循环中运行，其中评论家VLM提出一小组定性的关系编辑；多个候选LLM使用不同的策略（保守型-激进型、替代型、聚焦型）合成SVG更新；而裁判VLM从中选择最佳候选，确保稳定改进。此设计强调定性推理而非脆弱的数值估计，保留了全局约束（如对齐、可连接性），并自然支持人类在环修正。在源自已发表论文的10个流chart草图上，我们的方法比两个前沿的闭源图像生成LLM（GPT-5和Gemini-2.5-Pro）更准确地重建布局和结构，无需插入不必要的文本即可正确组合基础元素（如多头箭头）。由于输出是程序化的SVG，该方法可以通过API扩展到演示工具（如PowerPoint），并通过改进提示和特定任务工具进一步专家定制。代码库在此公开 available at this https URL。 

---
# R-ConstraintBench: Evaluating LLMs on NP-Complete Scheduling 

**Title (ZH)**: R-约束基准：评估大语言模型在NP完全调度问题上的性能 

**Authors**: Raj Jain, Marc Wetter  

**Link**: [PDF](https://arxiv.org/pdf/2508.15204)  

**Abstract**: Effective scheduling under tight resource, timing, and operational constraints underpins large-scale planning across sectors such as capital projects, manufacturing, logistics, and IT fleet transitions. However, the reliability of large language models (LLMs) when reasoning under high-constraint regimes is insufficiently characterized. To address this gap, we present R-ConstraintBench, a scalable framework that evaluates models on Resource-Constrained Project Scheduling Problems (RCPSP), an NP-Complete feasibility class, while difficulty increases via linear growth in constraints. R-ConstraintBench incrementally increases non-redundant precedence constraints in Directed Acyclic Graphs (DAGs) and then introduces downtime, temporal windows, and disjunctive constraints. As an illustrative example, we instantiate the benchmark in a data center migration setting and evaluate multiple LLMs using feasibility and error analysis, identifying degradation thresholds and constraint types most associated with failure. Empirically, strong models are near-ceiling on precedence-only DAGs, but feasibility performance collapses when downtime, temporal windows, and disjunctive constraints interact, implicating constraint interaction, not graph depth, as the principal bottleneck. Performance on clean synthetic ramps also does not guarantee transfer to domain-grounded scenarios, underscoring limited generalization. 

**Abstract (ZH)**: 有效的资源、时间和操作约束下的调度对于跨资本项目、制造、物流和IT机队转换等多个领域的大规模规划至关重要。然而，大型语言模型（LLMs）在高约束环境下的可靠性尚未得到充分表征。为解决这一问题，我们提出了一种可扩展的框架R-ConstraintBench，该框架在Resource-Constrained Project Scheduling Problems（RCPSP）这一NP完全可行类问题上评估模型，通过约束的线性增长增加难度。R-ConstraintBench通过逐步增加有向无环图（DAGs）中的非冗余前置约束，然后引入停机时间、时间窗口和排斥约束来增加难度。作为示例，我们在数据中心迁移场景中实例化基准，并使用可行性和错误分析评估多个LLM，识别与失败最相关的降级阈值和约束类型。实验结果显示，强大的模型在仅前置约束的DAG上接近上限，但当停机时间、时间窗口和排斥约束相互作用时，可行性能急剧下降，表明约束交互而非图深度是主要瓶颈。清洁合成坡度上的性能也不能保证转移到实际场景中，强调了模型泛化能力的局限性。 

---
# LLM4Sweat: A Trustworthy Large Language Model for Hyperhidrosis Support 

**Title (ZH)**: LLM4Sweat: 一种可靠的大型语言模型以支持多汗症管理 

**Authors**: Wenjie Lin, Jin Wei-Kocsis  

**Link**: [PDF](https://arxiv.org/pdf/2508.15192)  

**Abstract**: While large language models (LLMs) have shown promise in healthcare, their application for rare medical conditions is still hindered by scarce and unreliable datasets for fine-tuning. Hyperhidrosis, a disorder causing excessive sweating beyond physiological needs, is one such rare disorder, affecting 2-3% of the population and significantly impacting both physical comfort and psychosocial well-being. To date, no work has tailored LLMs to advance the diagnosis or care of hyperhidrosis. To address this gap, we present LLM4Sweat, an open-source and domain-specific LLM framework for trustworthy and empathetic hyperhidrosis support. The system follows a three-stage pipeline. In the data augmentation stage, a frontier LLM generates medically plausible synthetic vignettes from curated open-source data to create a diverse and balanced question-answer dataset. In the fine-tuning stage, an open-source foundation model is fine-tuned on the dataset to provide diagnosis, personalized treatment recommendations, and empathetic psychological support. In the inference and expert evaluation stage, clinical and psychological specialists assess accuracy, appropriateness, and empathy, with validated responses iteratively enriching the dataset. Experiments show that LLM4Sweat outperforms baselines and delivers the first open-source LLM framework for hyperhidrosis, offering a generalizable approach for other rare diseases with similar data and trustworthiness challenges. 

**Abstract (ZH)**: 基于大型语言模型的汗疱症支持框架：LLM4Sweat 

---
# PuzzleClone: An SMT-Powered Framework for Synthesizing Verifiable Data 

**Title (ZH)**: PuzzleClone：一种基于SMT的可验证数据合成框架 

**Authors**: Kai Xiong, Yanwei Huang, Rongjunchen Zhang, Kun Chen, Haipang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15180)  

**Abstract**: High-quality mathematical and logical datasets with verifiable answers are essential for strengthening the reasoning capabilities of large language models (LLMs). While recent data augmentation techniques have facilitated the creation of large-scale benchmarks, existing LLM-generated datasets often suffer from limited reliability, diversity, and scalability. To address these challenges, we introduce PuzzleClone, a formal framework for synthesizing verifiable data at scale using Satisfiability Modulo Theories (SMT). Our approach features three key innovations: (1) encoding seed puzzles into structured logical specifications, (2) generating scalable variants through systematic variable and constraint randomization, and (3) ensuring validity via a reproduction mechanism. Applying PuzzleClone, we construct a curated benchmark comprising over 83K diverse and programmatically validated puzzles. The generated puzzles span a wide spectrum of difficulty and formats, posing significant challenges to current state-of-the-art models. We conduct post training (SFT and RL) on PuzzleClone datasets. Experimental results show that training on PuzzleClone yields substantial improvements not only on PuzzleClone testset but also on logic and mathematical benchmarks. Post training raises PuzzleClone average from 14.4 to 56.2 and delivers consistent improvements across 7 logic and mathematical benchmarks up to 12.5 absolute percentage points (AMC2023 from 52.5 to 65.0). Our code and data are available at this https URL. 

**Abstract (ZH)**: 高质量的数学和逻辑数据集及其可验证答案对于增强大型语言模型的推理能力至关重要。尽管最近的数据扩增技术简化了大规模基准的创建，但现有的大型语言模型生成的数据集往往可靠性有限、多样性不足且不可扩展。为了解决这些挑战，我们介绍了PuzzleClone，这是一种使用理论满足性（Satisfiability Modulo Theories, SMT）规模化合成可验证数据的正式框架。该方法包含三个关键创新点：（1）将种子谜题编码为结构化的逻辑规范，（2）通过系统变量和约束的随机化生成可扩展的变体，（3）通过复制机制确保有效性。应用PuzzleClone，我们构建了一个经过精心筛选的基准数据集，包含超过83,000个多样化的并由程序验证的谜题。生成的谜题覆盖了广泛难度和格式范围，对当前最先进的模型构成了重大挑战。我们在PuzzleClone数据集上进行了模型后训练（SFT和RL）。实验结果表明，使用PuzzleClone进行训练不仅在PuzzleClone测试集上取得了显著改进，还在逻辑和数学基准测试上也取得了提高。后训练后，PuzzleClone的平均得分从14.4提高到56.2，并在7个逻辑和数学基准测试中实现了高达12.5个百分点的持续改进（AMC2023从52.5提高到65.0）。我们的代码和数据可在以下链接获取。 

---
# Mobile-Agent-v3: Foundamental Agents for GUI Automation 

**Title (ZH)**: Mobile-Agent-v3：GUI自动化基础代理 

**Authors**: Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, Jitong Liao, Qi Zheng, Fei Huang, Jingren Zhou, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2508.15144)  

**Abstract**: This paper introduces GUI-Owl, a foundational GUI agent model that achieves state-of-the-art performance among open-source end-to-end models on ten GUI benchmarks across desktop and mobile environments, covering grounding, question answering, planning, decision-making, and procedural knowledge. GUI-Owl-7B achieves 66.4 on AndroidWorld and 29.4 on OSWorld. Building on this, we propose Mobile-Agent-v3, a general-purpose GUI agent framework that further improves performance to 73.3 on AndroidWorld and 37.7 on OSWorld, setting a new state-of-the-art for open-source GUI agent frameworks. GUI-Owl incorporates three key innovations: (1) Large-scale Environment Infrastructure: a cloud-based virtual environment spanning Android, Ubuntu, macOS, and Windows, enabling our Self-Evolving GUI Trajectory Production framework. This generates high-quality interaction data via automated query generation and correctness validation, leveraging GUI-Owl to refine trajectories iteratively, forming a self-improving loop. It supports diverse data pipelines and reduces manual annotation. (2) Diverse Foundational Agent Capabilities: by integrating UI grounding, planning, action semantics, and reasoning patterns, GUI-Owl supports end-to-end decision-making and can act as a modular component in multi-agent systems. (3) Scalable Environment RL: we develop a scalable reinforcement learning framework with fully asynchronous training for real-world alignment. We also introduce Trajectory-aware Relative Policy Optimization (TRPO) for online RL, achieving 34.9 on OSWorld. GUI-Owl and Mobile-Agent-v3 are open-sourced at this https URL. 

**Abstract (ZH)**: 本文介绍了GUI-Owl，这是一种面向桌面和移动环境的通用GUI代理模型，涵盖桌面和移动环境下的十个GUI基准测试，在多个领域如语义 grounding、问答、规划、决策和程序性知识等方面实现了最先进的性能。GUI-Owl-7B在AndroidWorld上的得分为66.4，在OSWorld上的得分为29.4。在此基础上，本文提出了Mobile-Agent-v3，这是一种通用的GUI代理框架，进一步提高了性能，AndroidWorld得分为73.3，OSWorld得分为37.7，成为开源GUI代理框架的最新领先成果。GUI-Owl包含三项关键创新：（1）大规模环境基础设施：基于云的虚拟环境跨越Android、Ubuntu、macOS和Windows，支持我们不断进化的GUI轨迹生成框架。该框架通过自动化查询生成和正确性验证生成高质量的交互数据，利用GUI-Owl进行迭代的轨迹精炼，形成一个自我改进的循环。它支持多样化的数据管道并减少了手动标注的需求。（2）多种基础代理能力：通过整合UI语义grounding、规划、动作语义和推理模式，GUI-Owl支持端到端决策，并可作为多代理系统中的模块化组件。（3）可扩展的环境RL：我们开发了一个可扩展的强化学习框架，采用完全异步训练来实现现实世界对齐。我们还引入了轨迹感知相对策略优化（TRPO）用于在线RL，并在OSWorld上实现了34.9的得分。GUI-Owl和Mobile-Agent-v3已开源，详见此链接。 

---
# aiXiv: A Next-Generation Open Access Ecosystem for Scientific Discovery Generated by AI Scientists 

**Title (ZH)**: aiXiv: 由AI科学家生成的下一代开放访问科学发现生态系统 

**Authors**: Pengsong Zhang, Xiang Hu, Guowei Huang, Yang Qi, Heng Zhang, Xiuxu Li, Jiaxing Song, Jiabin Luo, Yijiang Li, Shuo Yin, Chengxiao Dai, Eric Hanchen Jiang, Xiaoyan Zhou, Zhenfei Yin, Boqin Yuan, Jing Dong, Guinan Su, Guanren Qiao, Haiming Tang, Anghong Du, Lili Pan, Zhenzhong Lan, Xinyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15126)  

**Abstract**: Recent advances in large language models (LLMs) have enabled AI agents to autonomously generate scientific proposals, conduct experiments, author papers, and perform peer reviews. Yet this flood of AI-generated research content collides with a fragmented and largely closed publication ecosystem. Traditional journals and conferences rely on human peer review, making them difficult to scale and often reluctant to accept AI-generated research content; existing preprint servers (e.g. arXiv) lack rigorous quality-control mechanisms. Consequently, a significant amount of high-quality AI-generated research lacks appropriate venues for dissemination, hindering its potential to advance scientific progress. To address these challenges, we introduce aiXiv, a next-generation open-access platform for human and AI scientists. Its multi-agent architecture allows research proposals and papers to be submitted, reviewed, and iteratively refined by both human and AI scientists. It also provides API and MCP interfaces that enable seamless integration of heterogeneous human and AI scientists, creating a scalable and extensible ecosystem for autonomous scientific discovery. Through extensive experiments, we demonstrate that aiXiv is a reliable and robust platform that significantly enhances the quality of AI-generated research proposals and papers after iterative revising and reviewing on aiXiv. Our work lays the groundwork for a next-generation open-access ecosystem for AI scientists, accelerating the publication and dissemination of high-quality AI-generated research content. Code is available at this https URL. Website is available at this https URL. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）使AI代理能够自主生成科学提案、开展实验、撰写论文和进行同行评审。然而，大量由AI生成的研究内容与分散且主要封闭的出版生态系统相碰撞。传统期刊和会议依赖于人力同行评审，这使得它们难以扩展，并且往往不愿接受AI生成的研究内容；现有的预印本服务器（如arXiv）缺乏严格的质量控制机制。因此，大量高质量的AI生成研究缺乏适当的传播渠道，阻碍了其潜在的科学发展。为应对这些挑战，我们引入了aiXiv，这是一个用于人类和AI科学家的下一代开放获取平台。其多代理架构允许研究提案和论文由人类和AI科学家共同提交、评审和迭代改进。此外，aiXiv提供了API和MCP接口，使异构的人类和AI科学家无缝集成，创建一个可扩展和可扩展的自主科学发现生态系统。通过广泛实验，我们证明aiXiv是一个可靠且 robust 的平台，在aiXiv上经过迭代修订和评审后，显著提高了AI生成的研究提案和论文的质量。我们的工作为基础构建了一个下一代开放访问生态系统，加速高质量AI生成研究成果的出版和传播。代码可在以下网址获得：this https URL。网站可在以下网址获得：this https URL。 

---
# Open-Universe Assistance Games 

**Title (ZH)**: 开放宇宙辅助游戏 

**Authors**: Rachel Ma, Jingyi Qu, Andreea Bobu, Dylan Hadfield-Menell  

**Link**: [PDF](https://arxiv.org/pdf/2508.15119)  

**Abstract**: Embodied AI agents must infer and act in an interpretable way on diverse human goals and preferences that are not predefined. To formalize this setting, we introduce Open-Universe Assistance Games (OU-AGs), a framework where the agent must reason over an unbounded and evolving space of possible goals. In this context, we introduce GOOD (GOals from Open-ended Dialogue), a data-efficient, online method that extracts goals in the form of natural language during an interaction with a human, and infers a distribution over natural language goals. GOOD prompts an LLM to simulate users with different complex intents, using its responses to perform probabilistic inference over candidate goals. This approach enables rich goal representations and uncertainty estimation without requiring large offline datasets. We evaluate GOOD in a text-based grocery shopping domain and in a text-operated simulated household robotics environment (AI2Thor), using synthetic user profiles. Our method outperforms a baseline without explicit goal tracking, as confirmed by both LLM-based and human evaluations. 

**Abstract (ZH)**: 开放领域协助博弈：从开放对话中提取目标的高效在线方法 

---
# Argumentation for Explainable Workforce Optimisation (with Appendix) 

**Title (ZH)**: 可解释的工作force优化的论证（附录附录） 

**Authors**: Jennifer Leigh, Dimitrios Letsios, Alessandro Mella, Lucio Machetti, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2508.15118)  

**Abstract**: Workforce management is a complex problem optimising the makespan and travel distance required for a team of operators to complete a set of jobs, using a set of instruments. A crucial challenge in workforce management is accommodating changes at execution time so that explanations are provided to all stakeholders involved. Here, we show that, by understanding workforce management as abstract argumentation in an industrial application, we can accommodate change and obtain faithful explanations. We show, with a user study, that our tool and explanations lead to faster and more accurate problem solving than conventional solutions by hand. 

**Abstract (ZH)**: 工业应用中通过抽象论证实现工人群体管理中的灵活变更与忠实解释 

---
# S3LoRA: Safe Spectral Sharpness-Guided Pruning in Adaptation of Agent Planner 

**Title (ZH)**: S3LoRA: 安全频谱锐化指导的代理规划器适配剪枝 

**Authors**: Shuang Ao, Gopal Rumchurn  

**Link**: [PDF](https://arxiv.org/pdf/2508.15068)  

**Abstract**: Adapting Large Language Models (LLMs) using parameter-efficient fine-tuning (PEFT) techniques such as LoRA has enabled powerful capabilities in LLM-based agents. However, these adaptations can unintentionally compromise safety alignment, leading to unsafe or unstable behaviors, particularly in agent planning tasks. Existing safety-aware adaptation methods often require access to both base and instruction-tuned model checkpoints, which are frequently unavailable in practice, limiting their applicability. We propose S3LoRA (Safe Spectral Sharpness-Guided Pruning LoRA), a lightweight, data-free, and model-independent framework that mitigates safety risks in LoRA-adapted models by inspecting only the fine-tuned weight updates. We first introduce Magnitude-Aware Spherically Normalized SVD (MAS-SVD), which robustly analyzes the structural properties of LoRA updates while preserving global magnitude information. We then design the Spectral Sharpness Index (SSI), a sharpness-aware metric to detect layers with highly concentrated and potentially unsafe updates. These layers are pruned post-hoc to reduce risk without sacrificing task performance. Extensive experiments and ablation studies across agent planning and language generation tasks show that S3LoRA consistently improves safety metrics while maintaining or improving utility metrics and significantly reducing inference cost. These results establish S3LoRA as a practical and scalable solution for safely deploying LLM-based agents in real-world, resource-constrained, and safety-critical environments. 

**Abstract (ZH)**: 使用LoRA等参数高效微调（PEFT）技术适应大型语言模型（LLMs）已赋予基于LLM的代理强大的能力。然而，这些适应可能会无意中损害安全性对齐，特别是在代理规划任务中可能导致不安全或不稳定的行为。现有的安全性感知适应方法通常需要访问基础模型和指令微调模型的检查点，而在实践中这些检查点往往不可用，限制了它们的应用范围。我们提出了S3LoRA（Safe Spectral Sharpness-Guided Pruning LoRA），这是一种轻量级、无需数据且模型独立的框架，通过仅检查微调权重更新来减轻LoRA适应模型中的安全性风险。我们首先引入了感知幅度的球形规范化SVD（MAS-SVD），它稳健地分析LoRA更新的结构特性，同时保留整体幅度信息。然后我们设计了频谱尖锐度索引（SSI），这是一种尖锐度感知的指标，用于检测具有高度集中且可能不安全更新的层。这些层在后处理中被剪枝以降低风险，而不牺牲任务性能。广泛的实验和剥离研究显示，S3LoRA在提高安全性指标的同时保持或改进了实用性指标，并显著降低了推理成本。这些结果表明，S3LoRA是一种实用且可扩展的解决方案，可在资源受限和安全性关键的实际环境中安全部署基于LLM的代理。 

---
# Demonstrating Onboard Inference for Earth Science Applications with Spectral Analysis Algorithms and Deep Learning 

**Title (ZH)**: 基于光谱分析算法和深度学习的机载推断在地球科学应用中的演示 

**Authors**: Itai Zilberstein, Alberto Candela, Steve Chien, David Rijlaarsdam, Tom Hendrix, Leonie Buckley, Aubrey Dunne  

**Link**: [PDF](https://arxiv.org/pdf/2508.15053)  

**Abstract**: In partnership with Ubotica Technologies, the Jet Propulsion Laboratory is demonstrating state-of-the-art data analysis onboard CogniSAT-6/HAMMER (CS-6). CS-6 is a satellite with a visible and near infrared range hyperspectral instrument and neural network acceleration hardware. Performing data analysis at the edge (e.g. onboard) can enable new Earth science measurements and responses. We will demonstrate data analysis and inference onboard CS-6 for numerous applications using deep learning and spectral analysis algorithms. 

**Abstract (ZH)**: 在Ubotica Technologies的合作下，喷气推进实验室正在通过CogniSAT-6/HAMMER (CS-6) 显示先进的数据处理技术。CS-6是一颗配备可见光和近红外波段高光谱仪器及神经网络加速硬件的卫星。在边缘（例如，卫星上）执行数据处理可以实现新的地球科学研究和响应。我们将使用深度学习和光谱分析算法在CS-6上进行数据处理和推断演示，用于多种应用。 

---
# Don't Think Twice! Over-Reasoning Impairs Confidence Calibration 

**Title (ZH)**: 不要犹豫！过度推理会损害自信度校准。 

**Authors**: Romain Lacombe, Kerrie Wu, Eddie Dilworth  

**Link**: [PDF](https://arxiv.org/pdf/2508.15050)  

**Abstract**: Large Language Models deployed as question answering tools require robust calibration to avoid overconfidence. We systematically evaluate how reasoning capabilities and budget affect confidence assessment accuracy, using the ClimateX dataset (Lacombe et al., 2023) and expanding it to human and planetary health. Our key finding challenges the "test-time scaling" paradigm: while recent reasoning LLMs achieve 48.7% accuracy in assessing expert confidence, increasing reasoning budgets consistently impairs rather than improves calibration. Extended reasoning leads to systematic overconfidence that worsens with longer thinking budgets, producing diminishing and negative returns beyond modest computational investments. Conversely, search-augmented generation dramatically outperforms pure reasoning, achieving 89.3% accuracy by retrieving relevant evidence. Our results suggest that information access, rather than reasoning depth or inference budget, may be the critical bottleneck for improved confidence calibration of knowledge-intensive tasks. 

**Abstract (ZH)**: 大型语言模型作为问答工具部署时需要稳健校准以避免过度自信。我们系统评估推理能力与预算对信心评估准确度的影响，使用ClimateX数据集（Lacombe等，2023）并扩展到人类和 planetary 健康领域。我们关键发现挑战了“测试时缩放”范式：虽然近期推理大模型在评估专家信心方面达到48.7%的准确性，但增加推理预算始终会恶化而不是改善校准。延长推理会导致系统性过度自信，随着思考预算的增加而加剧，超出适度计算投资后产生递减甚至负效益。相反，搜索增强型生成显著优于单纯推理，通过检索相关证据达到89.3%的准确度。我们的结果表明，对于知识密集型任务的信心校准改进，信息访问可能比推理深度或推理预算更为关键。 

---
# Emergent Crowds Dynamics from Language-Driven Multi-Agent Interactions 

**Title (ZH)**: 语言驱动多agent交互中的涌现人群动力学 

**Authors**: Yibo Liu, Liam Shatzel, Brandon Haworth, Teseo Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2508.15047)  

**Abstract**: Animating and simulating crowds using an agent-based approach is a well-established area where every agent in the crowd is individually controlled such that global human-like behaviour emerges. We observe that human navigation and movement in crowds are often influenced by complex social and environmental interactions, driven mainly by language and dialogue. However, most existing work does not consider these dimensions and leads to animations where agent-agent and agent-environment interactions are largely limited to steering and fixed higher-level goal extrapolation.
We propose a novel method that exploits large language models (LLMs) to control agents' movement. Our method has two main components: a dialogue system and language-driven navigation. We periodically query agent-centric LLMs conditioned on character personalities, roles, desires, and relationships to control the generation of inter-agent dialogue when necessitated by the spatial and social relationships with neighbouring agents. We then use the conversation and each agent's personality, emotional state, vision, and physical state to control the navigation and steering of each agent. Our model thus enables agents to make motion decisions based on both their perceptual inputs and the ongoing dialogue.
We validate our method in two complex scenarios that exemplify the interplay between social interactions, steering, and crowding. In these scenarios, we observe that grouping and ungrouping of agents automatically occur. Additionally, our experiments show that our method serves as an information-passing mechanism within the crowd. As a result, our framework produces more realistic crowd simulations, with emergent group behaviours arising naturally from any environmental setting. 

**Abstract (ZH)**: 使用基于代理的方法进行人群动画和模拟：利用大型语言模型控制代理的运动 

---
# Collab-REC: An LLM-based Agentic Framework for Balancing Recommendations in Tourism 

**Title (ZH)**: Collab-REC: 一种基于LLM的代理框架，用于平衡旅游推荐 

**Authors**: Ashmi Banerjee, Fitri Nur Aisyah, Adithi Satish, Wolfgang Wörndl, Yashar Deldjoo  

**Link**: [PDF](https://arxiv.org/pdf/2508.15030)  

**Abstract**: We propose Collab-REC, a multi-agent framework designed to counteract popularity bias and enhance diversity in tourism recommendations. In our setting, three LLM-based agents -- Personalization, Popularity, and Sustainability generate city suggestions from complementary perspectives. A non-LLM moderator then merges and refines these proposals via multi-round negotiation, ensuring each agent's viewpoint is incorporated while penalizing spurious or repeated responses. Experiments on European city queries show that Collab-REC improves diversity and overall relevance compared to a single-agent baseline, surfacing lesser-visited locales that often remain overlooked. This balanced, context-aware approach addresses over-tourism and better aligns with constraints provided by the user, highlighting the promise of multi-stakeholder collaboration in LLM-driven recommender systems. 

**Abstract (ZH)**: Collab-REC：一个多智能体框架，用于应对流行性偏见并增强旅游推荐的多样性 

---
# Goals and the Structure of Experience 

**Title (ZH)**: 目标与经验结构 

**Authors**: Nadav Amir, Stas Tiomkin, Angela Langdon  

**Link**: [PDF](https://arxiv.org/pdf/2508.15013)  

**Abstract**: Purposeful behavior is a hallmark of natural and artificial intelligence. Its acquisition is often believed to rely on world models, comprising both descriptive (what is) and prescriptive (what is desirable) aspects that identify and evaluate state of affairs in the world, respectively. Canonical computational accounts of purposeful behavior, such as reinforcement learning, posit distinct components of a world model comprising a state representation (descriptive aspect) and a reward function (prescriptive aspect). However, an alternative possibility, which has not yet been computationally formulated, is that these two aspects instead co-emerge interdependently from an agent's goal. Here, we describe a computational framework of goal-directed state representation in cognitive agents, in which the descriptive and prescriptive aspects of a world model co-emerge from agent-environment interaction sequences, or experiences. Drawing on Buddhist epistemology, we introduce a construct of goal-directed, or telic, states, defined as classes of goal-equivalent experience distributions. Telic states provide a parsimonious account of goal-directed learning in terms of the statistical divergence between behavioral policies and desirable experience features. We review empirical and theoretical literature supporting this novel perspective and discuss its potential to provide a unified account of behavioral, phenomenological and neural dimensions of purposeful behaviors across diverse substrates. 

**Abstract (ZH)**: 有目的的行为是自然智能和人工智能的标志。其获取往往被认为依赖于包含描述性（现状是什么）和规范性（现状应是什么）方面世界模型，分别识别和评估世界的状态。传统的计算主义对有目的行为的解释，如强化学习，假定世界模型包括描述性方面（状态表示）和规范性方面（奖励函数）的不同组件。然而，一种尚未被计算上形式化的替代可能性是，这两个方面相互依存地从代理的目标中涌现出来。在此，我们描述了一种认知代理目标导向的状态表示的计算框架，在其中描述性和规范性方面世界模型从代理-环境交互序列或经验中共同涌现。借鉴佛教认识论，我们引入了一个目标导向的或 telic（目的论的）状态的概念，定义为目标等价经验分布的类别。telic 状态以行为政策与期望经验特征之间的统计差异为背景，提供了一种简洁的关于目标导向学习的解释。我们回顾了支持这一新颖观点的实证和理论文献，并讨论了其在跨不同底物的行为、现象学和神经维度方面统一解释有目的行为的潜力。 

---
# A Fully Spectral Neuro-Symbolic Reasoning Architecture with Graph Signal Processing as the Computational Backbone 

**Title (ZH)**: 全频谱神经符号推理架构：以图信号处理作为计算骨干 

**Authors**: Andrew Kiruluta  

**Link**: [PDF](https://arxiv.org/pdf/2508.14923)  

**Abstract**: We propose a fully spectral, neuro\-symbolic reasoning architecture that leverages Graph Signal Processing (GSP) as the primary computational backbone for integrating symbolic logic and neural inference. Unlike conventional reasoning models that treat spectral graph methods as peripheral components, our approach formulates the entire reasoning pipeline in the graph spectral domain. Logical entities and relationships are encoded as graph signals, processed via learnable spectral filters that control multi-scale information propagation, and mapped into symbolic predicates for rule-based inference. We present a complete mathematical framework for spectral reasoning, including graph Fourier transforms, band-selective attention, and spectral rule grounding. Experiments on benchmark reasoning datasets (ProofWriter, EntailmentBank, bAbI, CLUTRR, and ARC-Challenge) demonstrate improvements in logical consistency, interpretability, and computational efficiency over state\-of\-the\-art neuro\-symbolic models. Our results suggest that GSP provides a mathematically grounded and computationally efficient substrate for robust and interpretable reasoning systems. 

**Abstract (ZH)**: 我们提出了一种完全基于光谱的神经符号推理架构，该架构以图信号处理（GSP）作为主要的计算基础，整合符号逻辑和神经推理。与传统的推理模型将光谱图方法视为外围组件不同，我们的方法将整个推理管道公式化在图光谱域中。逻辑实体和关系被编码为图信号，通过可学习的光谱滤波器处理，控制多尺度信息传播，并映射为基于规则的推理中的符号谓词。我们提供了一个完整的光谱推理数学框架，包括图傅里叶变换、带通选择性注意力和光谱规则接地。在基准推理数据集（ProofWriter、EntailmentBank、bAbI、CLUTRR和ARC-Challenge）上的实验表明，与最先进的神经符号模型相比，在逻辑一致性、可解释性和计算效率方面有所改进。我们的结果表明，GSP 为健壮且可解释的推理系统提供了一个数学基础和计算高效的底层架构。 

---
# SceneGen: Single-Image 3D Scene Generation in One Feedforward Pass 

**Title (ZH)**: SceneGen: 单张图像的一次前向传播生成三维场景 

**Authors**: Yanxu Meng, Haoning Wu, Ya Zhang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.15769)  

**Abstract**: 3D content generation has recently attracted significant research interest due to its applications in VR/AR and embodied AI. In this work, we address the challenging task of synthesizing multiple 3D assets within a single scene image. Concretely, our contributions are fourfold: (i) we present SceneGen, a novel framework that takes a scene image and corresponding object masks as input, simultaneously producing multiple 3D assets with geometry and texture. Notably, SceneGen operates with no need for optimization or asset retrieval; (ii) we introduce a novel feature aggregation module that integrates local and global scene information from visual and geometric encoders within the feature extraction module. Coupled with a position head, this enables the generation of 3D assets and their relative spatial positions in a single feedforward pass; (iii) we demonstrate SceneGen's direct extensibility to multi-image input scenarios. Despite being trained solely on single-image inputs, our architectural design enables improved generation performance with multi-image inputs; and (iv) extensive quantitative and qualitative evaluations confirm the efficiency and robust generation abilities of our approach. We believe this paradigm offers a novel solution for high-quality 3D content generation, potentially advancing its practical applications in downstream tasks. The code and model will be publicly available at: this https URL. 

**Abstract (ZH)**: 3D内容生成由于其在VR/AR和嵌入式AI中的应用 recently attracted significant research interest。在本文中，我们针对单张场景图像内合成多个3D资产这一具有挑战性的任务进行了研究。具体而言，我们的贡献包括四个方面：（i）我们提出了SceneGen，一种新型框架，该框架以场景图像和对应的物体掩码为输入，同时生成具有几何和纹理的多个3D资产。值得注意的是，SceneGen 不需要优化或资产检索；（ii）我们引入了一种新颖的特征聚合模块，该模块在特征提取模块中结合了视觉和几何编码器的局部和全局场景信息。结合位置头，这使得在单次前向传递中生成3D资产及其相对空间位置成为可能；（iii）我们展示了SceneGen 直接扩展到多张图像输入场景的能力。尽管仅在单张图像输入上进行训练，但我们的架构设计使得使用多张图像输入时能够提高生成性能；（iv）广泛的定量和定性评估证实了我们方法的高效性和鲁棒性生成能力。我们相信，这种范式为高质量3D内容生成提供了一个新颖的解决方案，并有可能推动其在下游任务中的实际应用。代码和模型将在以下链接公开：this https URL。 

---
# Discovering Hidden Algebraic Structures via Transformers with Rank-Aware Beam GRPO 

**Title (ZH)**: 基于秩意识束GRPO的变压器发现隐藏代数结构 

**Authors**: Jaeha Lee, Gio Huh, Ning Su, Tony Yue YU  

**Link**: [PDF](https://arxiv.org/pdf/2508.15766)  

**Abstract**: Recent efforts have extended the capabilities of transformers in logical reasoning and symbolic computations. In this work, we investigate their capacity for non-linear latent pattern discovery in the context of functional decomposition, focusing on the challenging algebraic task of multivariate polynomial decomposition. This problem, with widespread applications in science and engineering, is proved to be NP-hard, and demands both precision and insight. Our contributions are threefold: First, we develop a synthetic data generation pipeline providing fine-grained control over problem complexity. Second, we train transformer models via supervised learning and evaluate them across four key dimensions involving scaling behavior and generalizability. Third, we propose Beam Grouped Relative Policy Optimization (BGRPO), a rank-aware reinforcement learning method suitable for hard algebraic problems. Finetuning with BGRPO improves accuracy while reducing beam width by up to half, resulting in approximately 75% lower inference compute. Additionally, our model demonstrates competitive performance in polynomial simplification, outperforming Mathematica in various cases. 

**Abstract (ZH)**: 最近的努力扩展了变换器在逻辑推理和符号计算中的能力。在此项工作中，我们调查了它们在函数分解背景下非线性潜在模式发现的能力，重点关注多项式分解这一具有挑战性的代数任务。该问题在科学和工程中有着广泛的应用，并已被证明是NP-hard的，既要求精确性又要求洞察力。我们的贡献体现在三个方面：首先，我们开发了一种合成数据生成管道，提供了对问题复杂性的精细控制。其次，我们通过监督学习训练变换器模型，并从四个关键维度对其进行评估，涉及缩放行为和泛化能力。第三，我们提出了一种基于排名的强化学习方法——束分组相对策略优化（BGRPO），适用于难以求解的代数问题。使用BGRPO微调可以提高准确性，并将束宽度降低至一半以下，计算推理量减少约75%。此外，我们的模型在多项式简化方面表现出竞争力，在某些情况下优于Mathematica。 

---
# LiveMCP-101: Stress Testing and Diagnosing MCP-enabled Agents on Challenging Queries 

**Title (ZH)**: LiveMCP-101：针对具有MCP功能代理的苛刻查询进行压力测试与诊断 

**Authors**: Ming Yin, Dinghan Shen, Silei Xu, Jianbing Han, Sixun Dong, Mian Zhang, Yebowen Hu, Shujian Liu, Simin Ma, Song Wang, Sathish Reddy Indurthi, Xun Wang, Yiran Chen, Kaiqiang Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.15760)  

**Abstract**: Tool calling has emerged as a critical capability for AI agents to interact with the real world and solve complex tasks. While the Model Context Protocol (MCP) provides a powerful standardized framework for tool integration, there is a significant gap in benchmarking how well AI agents can effectively solve multi-step tasks using diverse MCP tools in realistic, dynamic scenarios. In this work, we present LiveMCP-101, a benchmark of 101 carefully curated real-world queries, refined through iterative LLM rewriting and manual review, that require coordinated use of multiple MCP tools including web search, file operations, mathematical reasoning, and data analysis. Moreover, we introduce a novel evaluation approach that leverages ground-truth execution plans rather than raw API outputs, better reflecting the evolving nature of real-world environments. Experiments show that even frontier LLMs achieve a success rate below 60\%, highlighting major challenges in tool orchestration. Detailed ablations and error analysis further reveal distinct failure modes and inefficiencies in token usage, pointing to concrete directions for advancing current models. LiveMCP-101 sets a rigorous standard for evaluating real-world agent capabilities, advancing toward autonomous AI systems that reliably execute complex tasks through tool use. 

**Abstract (ZH)**: LiveMCP-101：一种严格的标准，用于评估在多步骤任务中有效使用多样MCP工具的现实世界代理能力 

---
# Neural Robot Dynamics 

**Title (ZH)**: 神经机器人动力学 

**Authors**: Jie Xu, Eric Heiden, Iretiayo Akinola, Dieter Fox, Miles Macklin, Yashraj Narang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15755)  

**Abstract**: Accurate and efficient simulation of modern robots remains challenging due to their high degrees of freedom and intricate mechanisms. Neural simulators have emerged as a promising alternative to traditional analytical simulators, capable of efficiently predicting complex dynamics and adapting to real-world data; however, existing neural simulators typically require application-specific training and fail to generalize to novel tasks and/or environments, primarily due to inadequate representations of the global state. In this work, we address the problem of learning generalizable neural simulators for robots that are structured as articulated rigid bodies. We propose NeRD (Neural Robot Dynamics), learned robot-specific dynamics models for predicting future states for articulated rigid bodies under contact constraints. NeRD uniquely replaces the low-level dynamics and contact solvers in an analytical simulator and employs a robot-centric and spatially-invariant simulation state representation. We integrate the learned NeRD models as an interchangeable backend solver within a state-of-the-art robotics simulator. We conduct extensive experiments to show that the NeRD simulators are stable and accurate over a thousand simulation steps; generalize across tasks and environment configurations; enable policy learning exclusively in a neural engine; and, unlike most classical simulators, can be fine-tuned from real-world data to bridge the gap between simulation and reality. 

**Abstract (ZH)**: 现代机器人高效且精确的模拟仍然是一个挑战，由于它们具有高自由度和复杂的机械结构。神经模拟器作为传统解析模拟器的有前途的替代方案，能够高效预测复杂动力学并适应现实世界数据；然而，现有的神经模拟器通常需要特定应用的训练，并且难以泛化到新的任务和/or环境，主要是因为对全局状态的表示不足。在这项工作中，我们解决了学习可泛化的机器人神经模拟器的问题，这些机器人由铰接刚体结构组成。我们提出NeRD（Neural Robot Dynamics），这是一种学习到的针对铰接刚体的动态模型，用于在接触约束下预测未来状态。NeRD独特地替代了解析模拟器中的低级动力学和接触求解器，并采用以机器人为中心和空间不变的模拟状态表示。我们将学习到的NeRD模型作为可互换的后端求解器集成到最先进的机器人模拟器中。我们进行了一系列实验，结果显示，NeRD模拟器在一千个模拟步骤中稳定且准确；能够在不同任务和环境配置之间泛化；使策略学习仅在神经引擎中进行；并且，与大多数经典模拟器不同，可以从现实世界数据进行微调，以弥合模拟与现实之间的差距。 

---
# Dissecting Tool-Integrated Reasoning: An Empirical Study and Analysis 

**Title (ZH)**: 剖析工具集成推理：一项实证研究与分析 

**Authors**: Yufeng Zhao, Junnan Liu, Hongwei Liu, Dongsheng Zhu, Yuan Shen, Songyang Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15754)  

**Abstract**: Large Language Models (LLMs) have made significant strides in reasoning tasks through methods like chain-of-thought (CoT) reasoning. However, they often fall short in tasks requiring precise computations. Tool-Integrated Reasoning (TIR) has emerged as a solution by incorporating external tools into the reasoning process. Nevertheless, the generalization of TIR in improving the reasoning ability of LLM is still unclear. Additionally, whether TIR has improved the model's reasoning behavior and helped the model think remains to be studied. We introduce ReasonZoo, a comprehensive benchmark encompassing nine diverse reasoning categories, to evaluate the effectiveness of TIR across various domains. Additionally, we propose two novel metrics, Performance-Aware Cost (PAC) and Area Under the Performance-Cost Curve (AUC-PCC), to assess reasoning efficiency. Our empirical evaluation demonstrates that TIR-enabled models consistently outperform their non-TIR counterparts in both mathematical and non-mathematical tasks. Furthermore, TIR enhances reasoning efficiency, as evidenced by improved PAC and AUC-PCC, indicating reduced overthinking and more streamlined reasoning. These findings underscore the domain-general benefits of TIR and its potential to advance LLM capabilities in complex reasoning tasks. 

**Abstract (ZH)**: 大语言模型（LLMs）通过链式思考等方法在推理任务中取得了显著进展，但在需要精确计算的任务中往往表现不佳。工具集成推理（TIR）通过将外部工具纳入推理过程而作为一种解决方案出现，但TIR在提高LLM的推理能力方面的泛化能力仍不清楚。此外，TIR是否改善了模型的推理行为并帮助模型进行思考仍有待研究。我们引入了ReasonZoo，这是一个包含九种不同推理类别的全面基准，旨在评估TIR在各种领域的有效性。此外，我们提出了两种新的度量标准，即性能感知成本（PAC）和性能-成本曲线下的面积（AUC-PCC），以评估推理效率。我们的实证评估表明，TIR增强的模型在数学和非数学任务中始终优于非TIR模型。此外，TIR提高了推理效率，这从改进的PAC和AUC-PCC中可以得到证实，表明减少了过度思考并使推理更加流畅。这些发现强调了TIR在不同领域的普遍优势，并指出了其在复杂推理任务中提升LLM能力的潜力。 

---
# "Does the cafe entrance look accessible? Where is the door?" Towards Geospatial AI Agents for Visual Inquiries 

**Title (ZH)**: 咖啡店入口看起来便于进入吗？门在哪里？面向地理空间AI代理的视觉查询研究 

**Authors**: Jon E. Froehlich, Jared Hwang, Zeyu Wang, John S. O'Meara, Xia Su, William Huang, Yang Zhang, Alex Fiannaca, Philip Nelson, Shaun Kane  

**Link**: [PDF](https://arxiv.org/pdf/2508.15752)  

**Abstract**: Interactive digital maps have revolutionized how people travel and learn about the world; however, they rely on pre-existing structured data in GIS databases (e.g., road networks, POI indices), limiting their ability to address geo-visual questions related to what the world looks like. We introduce our vision for Geo-Visual Agents--multimodal AI agents capable of understanding and responding to nuanced visual-spatial inquiries about the world by analyzing large-scale repositories of geospatial images, including streetscapes (e.g., Google Street View), place-based photos (e.g., TripAdvisor, Yelp), and aerial imagery (e.g., satellite photos) combined with traditional GIS data sources. We define our vision, describe sensing and interaction approaches, provide three exemplars, and enumerate key challenges and opportunities for future work. 

**Abstract (ZH)**: 交互式数字地图已革新人们的旅行和世界认知方式；然而，它们依赖于GIS数据库中的预存结构化数据（例如，道路网络、POI索引），限制了其解决与世界面貌相关的地理事物可视化问题的能力。我们提出了Geo-Visual代理的概念——多模态AI代理，能够通过分析大规模的地理空间图像库（包括街道景观、地点照片和航空影像等）来理解和回应关于世界的精炼的空间视觉查询，结合传统的GIS数据源。我们定义了这一概念，描述了感知和交互方法，提供了三个示例，并列举了未来工作的关键挑战与机遇。 

---
# End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning 

**Title (ZH)**: 端到端自主RAG系统训练以实现可追溯的诊断推理 

**Authors**: Qiaoyu Zheng, Yuze Sun, Chaoyi Wu, Weike Zhao, Pengcheng Qiu, Yongguo Yu, Kun Sun, Yanfeng Wang, Ya Zhang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.15746)  

**Abstract**: Accurate diagnosis with medical large language models is hindered by knowledge gaps and hallucinations. Retrieval and tool-augmented methods help, but their impact is limited by weak use of external knowledge and poor feedback-reasoning traceability. To address these challenges, We introduce Deep-DxSearch, an agentic RAG system trained end-to-end with reinforcement learning (RL) that enables steer tracebale retrieval-augmented reasoning for medical diagnosis. In Deep-DxSearch, we first construct a large-scale medical retrieval corpus comprising patient records and reliable medical knowledge sources to support retrieval-aware reasoning across diagnostic scenarios. More crutially, we frame the LLM as the core agent and the retrieval corpus as its environment, using tailored rewards on format, retrieval, reasoning structure, and diagnostic accuracy, thereby evolving the agentic RAG policy from large-scale data through RL.
Experiments demonstrate that our end-to-end agentic RL training framework consistently outperforms prompt-engineering and training-free RAG approaches across multiple data centers. After training, Deep-DxSearch achieves substantial gains in diagnostic accuracy, surpassing strong diagnostic baselines such as GPT-4o, DeepSeek-R1, and other medical-specific frameworks for both common and rare disease diagnosis under in-distribution and out-of-distribution settings. Moreover, ablation studies on reward design and retrieval corpus components confirm their critical roles, underscoring the uniqueness and effectiveness of our approach compared with traditional implementations. Finally, case studies and interpretability analyses highlight improvements in Deep-DxSearch's diagnostic policy, providing deeper insight into its performance gains and supporting clinicians in delivering more reliable and precise preliminary diagnoses. See this https URL. 

**Abstract (ZH)**: 医学大型语言模型进行准确诊断受限于知识缺口和幻觉。检索和工具增强的方法有所帮助，但其影响受限于外部知识的弱使用和反馈推理 traceability 差。为应对这些挑战，我们提出了 Deep-DxSearch，这是一个通过强化学习 (RL) 端到端训练的代理性 RAG 系统，能够引导可追踪的检索增强推理以进行医学诊断。在 Deep-DxSearch 中，我们首先构建了一个包含患者记录和可靠医学知识来源的大规模医学检索语料库，以支持诊断场景中的检索感知推理。更为关键的是，我们将 LLM 作为核心代理，将检索语料库作为其环境，并使用针对格式、检索、推理结构和诊断准确性定制的奖励，从而通过 RL 从大规模数据中进化代理性 RAG 策略。实验表明，我们的端到端代理性 RL 训练框架在多个数据中心上均优于提示工程和无需训练的 RAG 方法。训练后，Deep-DxSearch 在诊断准确性方面取得了显著提升，超越了如 GPT-4o、DeepSeek-R1 等强诊断基线方法，适用于常见和罕见疾病的诊断，无论是在分布内还是分布外场景。此外，奖励设计和检索语料库组件的消融研究证实了它们的关键作用，突显了我们方法的独特性和有效性，与传统的实现方式相比更加独特和有效。最后，案例研究和可解释性分析强调了 Deep-DxSearch 的诊断政策改进，提供了其性能提升的更深入理解，并支持临床医生提供更可靠和精确的初步诊断。 

---
# Numerical models outperform AI weather forecasts of record-breaking extremes 

**Title (ZH)**: 数值模型在记录破纪录极端天气预报中优于AI预测 

**Authors**: Zhongwei Zhang, Erich Fischer, Jakob Zscheischler, Sebastian Engelke  

**Link**: [PDF](https://arxiv.org/pdf/2508.15724)  

**Abstract**: Artificial intelligence (AI)-based models are revolutionizing weather forecasting and have surpassed leading numerical weather prediction systems on various benchmark tasks. However, their ability to extrapolate and reliably forecast unprecedented extreme events remains unclear. Here, we show that for record-breaking weather extremes, the numerical model High RESolution forecast (HRES) from the European Centre for Medium-Range Weather Forecasts still consistently outperforms state-of-the-art AI models GraphCast, GraphCast operational, Pangu-Weather, Pangu-Weather operational, and Fuxi. We demonstrate that forecast errors in AI models are consistently larger for record-breaking heat, cold, and wind than in HRES across nearly all lead times. We further find that the examined AI models tend to underestimate both the frequency and intensity of record-breaking events, and they underpredict hot records and overestimate cold records with growing errors for larger record exceedance. Our findings underscore the current limitations of AI weather models in extrapolating beyond their training domain and in forecasting the potentially most impactful record-breaking weather events that are particularly frequent in a rapidly warming climate. Further rigorous verification and model development is needed before these models can be solely relied upon for high-stakes applications such as early warning systems and disaster management. 

**Abstract (ZH)**: 基于人工智能的模型正在革新天气预报，并在各种基准任务上超越了领先的数值天气预测系统。然而，它们在外推和可靠预报前所未有的极端事件方面的能力仍然不明确。在这里，我们展示了一旦发生创纪录的天气极端事件，欧洲中期天气预报中心的高分辨率预报（HRES）数值模型仍然始终优于最先进的AI模型GraphCast、GraphCast运营版、Pangu-Weather、Pangu-Weather运营版和Fuxi。我们证明，在几乎所有预报时效下，AI模型在创纪录的高温、低温和风速方面的预报误差始终大于HRES。进一步发现，所检查的AI模型倾向于低估创纪录事件的频率和强度，并且随着超出记录的程度增大，它们低估了热记录而高估了冷记录。我们的研究结果强调了当前AI天气模型在外推到其训练领域之外以及预报尤其是在快速变暖气候中特别频繁的最有可能产生重大影响的极端天气事件方面的局限性。在这些模型可以完全依赖于高风险应用（如早期预警系统和灾害管理）之前，需要进行进一步严格的验证和模型开发。 

---
# EcomMMMU: Strategic Utilization of Visuals for Robust Multimodal E-Commerce Models 

**Title (ZH)**: EcomMMMU：多模态电商模型中视觉的战略利用以提高鲁棒性 

**Authors**: Xinyi Ling, Hanwen Du, Zhihui Zhu, Xia Ning  

**Link**: [PDF](https://arxiv.org/pdf/2508.15721)  

**Abstract**: E-commerce platforms are rich in multimodal data, featuring a variety of images that depict product details. However, this raises an important question: do these images always enhance product understanding, or can they sometimes introduce redundancy or degrade performance? Existing datasets are limited in both scale and design, making it difficult to systematically examine this question. To this end, we introduce EcomMMMU, an e-commerce multimodal multitask understanding dataset with 406,190 samples and 8,989,510 images. EcomMMMU is comprised of multi-image visual-language data designed with 8 essential tasks and a specialized VSS subset to benchmark the capability of multimodal large language models (MLLMs) to effectively utilize visual content. Analysis on EcomMMMU reveals that product images do not consistently improve performance and can, in some cases, degrade it. This indicates that MLLMs may struggle to effectively leverage rich visual content for e-commerce tasks. Building on these insights, we propose SUMEI, a data-driven method that strategically utilizes multiple images via predicting visual utilities before using them for downstream tasks. Comprehensive experiments demonstrate the effectiveness and robustness of SUMEI. The data and code are available through this https URL. 

**Abstract (ZH)**: 电子商务平台富含多模态数据，展示产品的多张图片描绘产品细节。然而，这引发了一个重要问题：这些图片是否always提升产品理解，或有时会引入冗余甚至损害性能？现有数据集在规模和设计上都有限，难以系统地探讨这一问题。为了解决这一问题，我们引入了EcomMMMU多模态多任务理解数据集，包含406,190个样本和8,989,510张图片。EcomMMMU数据集由8个基础任务设计的多图视觉-语言数据组成，并包含专门的VSS子集，用于评估多模态大型语言模型利用视觉内容的能力。分析EcomMMMU数据集发现，产品图片并不始终提升性能，在某些情况下甚至会降低性能。这表明多模态大型语言模型可能难以有效地利用丰富的视觉内容来完成电子商务任务。基于这些发现，我们提出了SUMEI数据驱动方法，通过预测视觉效用后再用于下游任务，战略性地利用多张图片。全面的实验验证了SUMEI的有效性和鲁棒性。数据和代码可通过以下链接获取。 

---
# Tutorial on the Probabilistic Unification of Estimation Theory, Machine Learning, and Generative AI 

**Title (ZH)**: 概率統一估计理论、机器学习与生成AI教程 

**Authors**: Mohammed Elmusrati  

**Link**: [PDF](https://arxiv.org/pdf/2508.15719)  

**Abstract**: Extracting meaning from uncertain, noisy data is a fundamental problem across time series analysis, pattern recognition, and language modeling. This survey presents a unified mathematical framework that connects classical estimation theory, statistical inference, and modern machine learning, including deep learning and large language models. By analyzing how techniques such as maximum likelihood estimation, Bayesian inference, and attention mechanisms address uncertainty, the paper illustrates that many AI methods are rooted in shared probabilistic principles. Through illustrative scenarios including system identification, image classification, and language generation, we show how increasingly complex models build upon these foundations to tackle practical challenges like overfitting, data sparsity, and interpretability. In other words, the work demonstrates that maximum likelihood, MAP estimation, Bayesian classification, and deep learning all represent different facets of a shared goal: inferring hidden causes from noisy and/or biased observations. It serves as both a theoretical synthesis and a practical guide for students and researchers navigating the evolving landscape of machine learning. 

**Abstract (ZH)**: 从不确定和噪声数据中提取意义是时间序列分析、模式识别和语言建模中的一个基本问题。本文综述提出了一种统一的数学框架，将经典估计理论、统计推断和现代机器学习（包括深度学习和大型语言模型）联系起来。通过分析最大似然估计、贝叶斯推断和注意力机制如何处理不确定性，论文展示了许多AI方法在共享的概率原理基础上。通过对系统识别、图像分类和语言生成等示例的分析，本文展示了日益复杂的模型如何在这基础上解决过拟合、数据稀疏性和可解释性等实际挑战。简而言之，这项工作表明，最大似然、最大后验估计、贝叶斯分类和深度学习各自代表了共享目标的各个方面：从噪声和/或有偏差的观察中推断隐藏的原因。该综述既是一份理论综述，也为学生和研究人员导航不断演化的机器学习 landscape 提供了实用指南。 

---
# StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding 

**Title (ZH)**: StreamMem: 查询无关的键值缓存内存用于流式视频理解 

**Authors**: Yanlai Yang, Zhuokai Zhao, Satya Narayan Shukla, Aashu Singh, Shlok Kumar Mishra, Lizhu Zhang, Mengye Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.15717)  

**Abstract**: Multimodal large language models (MLLMs) have made significant progress in visual-language reasoning, but their ability to efficiently handle long videos remains limited. Despite recent advances in long-context MLLMs, storing and attending to the key-value (KV) cache for long visual contexts incurs substantial memory and computational overhead. Existing visual compression methods require either encoding the entire visual context before compression or having access to the questions in advance, which is impractical for long video understanding and multi-turn conversational settings. In this work, we propose StreamMem, a query-agnostic KV cache memory mechanism for streaming video understanding. Specifically, StreamMem encodes new video frames in a streaming manner, compressing the KV cache using attention scores between visual tokens and generic query tokens, while maintaining a fixed-size KV memory to enable efficient question answering (QA) in memory-constrained, long-video scenarios. Evaluation on three long video understanding and two streaming video question answering benchmarks shows that StreamMem achieves state-of-the-art performance in query-agnostic KV cache compression and is competitive with query-aware compression approaches. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）在视觉语言推理方面取得了显著进展，但其处理长视频的能力仍然有限。尽管在长上下文MLLM方面取得了一些进展，但存储和关注长视觉上下文的键值（KV）缓存会带来显著的内存和计算开销。现有的视觉压缩方法要么在压缩前对整个视觉上下文进行编码，要么需要提前获取问题，这对于长视频理解和多轮对话场景来说是不切实际的。在此工作中，我们提出了一种查询无关的KV缓存记忆机制StreamMem，用于流式视频理解。具体而言，StreamMem以流式方式编码新的视频帧，并使用视觉标记与通用查询标记之间的注意得分来压缩KV缓存，同时保持固定大小的KV内存以在内存受限的长视频场景中实现高效的问题回答（QA）。在三个长视频理解和两个流式视频问答基准上的评估显示，StreamMem在查询无关的KV缓存压缩方面达到了最先进的性能，并且与查询感知的压缩方法具有竞争力。 

---
# Foundation Models for Cross-Domain EEG Analysis Application: A Survey 

**Title (ZH)**: 跨域EEG分析应用的基金会模型：一种综述 

**Authors**: Hongqi Li, Yitong Chen, Yujuan Wang, Weihang Ni, Haodong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15716)  

**Abstract**: Electroencephalography (EEG) analysis stands at the forefront of neuroscience and artificial intelligence research, where foundation models are reshaping the traditional EEG analysis paradigm by leveraging their powerful representational capacity and cross-modal generalization. However, the rapid proliferation of these techniques has led to a fragmented research landscape, characterized by diverse model roles, inconsistent architectures, and a lack of systematic categorization. To bridge this gap, this study presents the first comprehensive modality-oriented taxonomy for foundation models in EEG analysis, systematically organizing research advances based on output modalities of the native EEG decoding, EEG-text, EEG-vision, EEG-audio, and broader multimodal frameworks. We rigorously analyze each category's research ideas, theoretical foundations, and architectural innovations, while highlighting open challenges such as model interpretability, cross-domain generalization, and real-world applicability in EEG-based systems. By unifying this dispersed field, our work not only provides a reference framework for future methodology development but accelerates the translation of EEG foundation models into scalable, interpretable, and online actionable solutions. 

**Abstract (ZH)**: 脑电图（EEG）分析处于神经科学和人工智能研究的前沿，基础模型正在通过对传统EEG分析范式的重塑，利用其强大的表征能力和跨模态泛化能力，重新定义这一领域。然而，这些技术的快速扩散导致了研究景观的碎片化，表现为多样的模型角色、不一致的架构以及缺乏系统的分类。为填补这一空白，本研究首次提出了基础模型在EEG分析中的首个全面的模态导向分类体系，系统地根据原生EEG解码、EEG-文本、EEG-视觉、EEG-音频以及更广泛的多模态框架的输出模态组织研究进展。我们严格分析了每个类别中的研究思路、理论基础和架构创新，同时指出了开放挑战，如模型可解释性、跨域泛化以及基于EEG的系统中的实际应用性。通过统一这个分散的领域，我们的工作不仅为未来方法学的发展提供了参考框架，还加速了EEG基础模型向可扩展、可解释和在线可操作解决方案的转变。 

---
# Row-Column Hybrid Grouping for Fault-Resilient Multi-Bit Weight Representation on IMC Arrays 

**Title (ZH)**: 行-列混合分组在IMC阵列上实现容错多比特权重表示 

**Authors**: Kang Eun Jeon, Sangheum Yeon, Jinhee Kim, Hyeonsu Bang, Johnny Rhe, Jong Hwan Ko  

**Link**: [PDF](https://arxiv.org/pdf/2508.15685)  

**Abstract**: This paper addresses two critical challenges in analog In-Memory Computing (IMC) systems that limit their scalability and deployability: the computational unreliability caused by stuck-at faults (SAFs) and the high compilation overhead of existing fault-mitigation algorithms, namely Fault-Free (FF). To overcome these limitations, we first propose a novel multi-bit weight representation technique, termed row-column hybrid grouping, which generalizes conventional column grouping by introducing redundancy across both rows and columns. This structural redundancy enhances fault tolerance and can be effectively combined with existing fault-mitigation solutions. Second, we design a compiler pipeline that reformulates the fault-aware weight decomposition problem as an Integer Linear Programming (ILP) task, enabling fast and scalable compilation through off-the-shelf solvers. Further acceleration is achieved through theoretical insights that identify fault patterns amenable to trivial solutions, significantly reducing computation. Experimental results on convolutional networks and small language models demonstrate the effectiveness of our approach, achieving up to 8%p improvement in accuracy, 150x faster compilation, and 2x energy efficiency gain compared to existing baselines. 

**Abstract (ZH)**: 本文探讨了模拟存内计算(IMC)系统中限制其可扩展性和部署性的两大关键挑战：由固定节点故障(SAFs)引起的计算可靠性问题以及现有故障缓解算法(如Fault-Free(FF))的高编译开销。为克服这些限制，我们首先提出了一种新的多比特权重表示技术，称为行-列混合分组，该技术通过在行和列之间引入冗余来推广传统的列分组方法，从而增强容错性，并能与现有的故障缓解解决方案有效结合。其次，我们设计了一种编译流水线，将故障感知权重分解问题重新表述为整数线性规划(ILP)问题，通过现成的求解器实现快速和可扩展的编译。通过理论洞察识别出适用于简单解决方案的故障模式，从而显著减少计算量。实验结果表明，与现有基线方法相比，该方法在卷积网络和小型语言模型上分别实现了多达8%的准确率提升、150倍的编译速度加快和2倍的能量效率提升。 

---
# Mind and Motion Aligned: A Joint Evaluation IsaacSim Benchmark for Task Planning and Low-Level Policies in Mobile Manipulation 

**Title (ZH)**: 思维与运动一致：面向移动 manipulation 中任务规划与低层级策略的 IsaacSim 基准评测 

**Authors**: Nikita Kachaev, Andrei Spiridonov, Andrey Gorodetsky, Kirill Muravyev, Nikita Oskolkov, Aditya Narendra, Vlad Shakhuro, Dmitry Makarov, Aleksandr I. Panov, Polina Fedotova, Alexey K. Kovalev  

**Link**: [PDF](https://arxiv.org/pdf/2508.15663)  

**Abstract**: Benchmarks are crucial for evaluating progress in robotics and embodied AI. However, a significant gap exists between benchmarks designed for high-level language instruction following, which often assume perfect low-level execution, and those for low-level robot control, which rely on simple, one-step commands. This disconnect prevents a comprehensive evaluation of integrated systems where both task planning and physical execution are critical. To address this, we propose Kitchen-R, a novel benchmark that unifies the evaluation of task planning and low-level control within a simulated kitchen environment. Built as a digital twin using the Isaac Sim simulator and featuring more than 500 complex language instructions, Kitchen-R supports a mobile manipulator robot. We provide baseline methods for our benchmark, including a task-planning strategy based on a vision-language model and a low-level control policy based on diffusion policy. We also provide a trajectory collection system. Our benchmark offers a flexible framework for three evaluation modes: independent assessment of the planning module, independent assessment of the control policy, and, crucially, an integrated evaluation of the whole system. Kitchen-R bridges a key gap in embodied AI research, enabling more holistic and realistic benchmarking of language-guided robotic agents. 

**Abstract (ZH)**: Kitchen-R：一种统一任务规划与低层级控制的模拟厨房环境基准 

---
# Benchmarking Computer Science Survey Generation 

**Title (ZH)**: 计算机科学综述生成基准测试 

**Authors**: Weihang Su, Anzhe Xie, Qingyao Ai, Jianming Long, Jiaxin Mao, Ziyi Ye, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15658)  

**Abstract**: Scientific survey articles play a vital role in summarizing research progress, yet their manual creation is becoming increasingly infeasible due to the rapid growth of academic literature. While large language models (LLMs) offer promising capabilities for automating this process, progress in this area is hindered by the absence of standardized benchmarks and evaluation protocols. To address this gap, we introduce SurGE (Survey Generation Evaluation), a new benchmark for evaluating scientific survey generation in the computer science domain. SurGE consists of (1) a collection of test instances, each including a topic description, an expert-written survey, and its full set of cited references, and (2) a large-scale academic corpus of over one million papers that serves as the retrieval pool. In addition, we propose an automated evaluation framework that measures generated surveys across four dimensions: information coverage, referencing accuracy, structural organization, and content quality. Our evaluation of diverse LLM-based approaches shows that survey generation remains highly challenging, even for advanced self-reflection frameworks. These findings highlight the complexity of the task and the necessity for continued research. We have open-sourced all the code, data, and models at: this https URL 

**Abstract (ZH)**: 科学综述文章在总结研究进展方面发挥着关键作用，但由于学术文献的快速增长，其手工创建变得日益不可行。尽管大型语言模型（LLMs）提供了自动化的潜在能力，但在该领域进展受到缺乏标准化基准和评估协议的阻碍。为应对这一缺口，我们推出了SurGE（综述生成评估），这是一个新的计算机科学领域的综述生成评估基准。SurGE包括（1）一组测试实例，每个实例包含一个主题描述、一位专家撰写的综述及其完整引用参考文献集，以及（2）一个包含超过一百万篇论文的大规模学术语料库，作为检索池。此外，我们提出了一种自动化评估框架，从四个维度测量生成的综述：信息覆盖、引用准确性、结构组织和内容质量。我们对各种LLM基方法的评估显示，即使对于先进的自我反思框架，综述生成仍然具有高度挑战性。这些发现突显了任务的复杂性以及持续研究的必要性。所有代码、数据和模型均已开源：this https URL。 

---
# Towards a 3D Transfer-based Black-box Attack via Critical Feature Guidance 

**Title (ZH)**: 基于关键特征指导的3D传输式黑盒攻击 

**Authors**: Shuchao Pang, Zhenghan Chen, Shen Zhang, Liming Lu, Siyuan Liang, Anan Du, Yongbin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.15650)  

**Abstract**: Deep neural networks for 3D point clouds have been demonstrated to be vulnerable to adversarial examples. Previous 3D adversarial attack methods often exploit certain information about the target models, such as model parameters or outputs, to generate adversarial point clouds. However, in realistic scenarios, it is challenging to obtain any information about the target models under conditions of absolute security. Therefore, we focus on transfer-based attacks, where generating adversarial point clouds does not require any information about the target models. Based on our observation that the critical features used for point cloud classification are consistent across different DNN architectures, we propose CFG, a novel transfer-based black-box attack method that improves the transferability of adversarial point clouds via the proposed Critical Feature Guidance. Specifically, our method regularizes the search of adversarial point clouds by computing the importance of the extracted features, prioritizing the corruption of critical features that are likely to be adopted by diverse architectures. Further, we explicitly constrain the maximum deviation extent of the generated adversarial point clouds in the loss function to ensure their imperceptibility. Extensive experiments conducted on the ModelNet40 and ScanObjectNN benchmark datasets demonstrate that the proposed CFG outperforms the state-of-the-art attack methods by a large margin. 

**Abstract (ZH)**: 深神经网络在三维点云上的攻击仍然面临着对抗样本的挑战。以往针对三维的对抗攻击方法常常依赖于目标模型的某些信息，如模型参数或输出，以生成对抗点云。然而在实际场景中，在绝对安全的条件下获取目标模型的任何信息都是极具挑战的。因此，我们聚焦于转储式攻击，即生成对抗点云无需依赖目标模型的任何信息。基于我们观察到的关键特征在不同DNN架构中的一致性，我们提出了一种新的转储式黑盒攻击方法CFG，通过提出的关键特征指导来提高对抗点云的转存性。具体而言，我们的方法通过计算提取特征的重要性来正则化对抗点云的搜索，优先破坏各种架构都可能采用的关键特征。此外，我们还在损失函数中显式地限制生成的对抗点云的最大偏差程度，以确保其不可感知性。在ModelNet40和ScanObjectNN基准数据集上的广泛实验表明，提出的CFG方法显著优于现有的先进攻击方法。 

---
# Label Uncertainty for Ultrasound Segmentation 

**Title (ZH)**: 超声分割中的标签不确定性 

**Authors**: Malini Shivaram, Gautam Rajendrakumar Gare, Laura Hutchins, Jacob Duplantis, Thomas Deiss, Thales Nogueira Gomes, Thong Tran, Keyur H. Patel, Thomas H Fox, Amita Krishnan, Deva Ramanan, Bennett DeBoisblanc, Ricardo Rodriguez, John Galeotti  

**Link**: [PDF](https://arxiv.org/pdf/2508.15635)  

**Abstract**: In medical imaging, inter-observer variability among radiologists often introduces label uncertainty, particularly in modalities where visual interpretation is subjective. Lung ultrasound (LUS) is a prime example-it frequently presents a mixture of highly ambiguous regions and clearly discernible structures, making consistent annotation challenging even for experienced clinicians. In this work, we introduce a novel approach to both labeling and training AI models using expert-supplied, per-pixel confidence values. Rather than treating annotations as absolute ground truth, we design a data annotation protocol that captures the confidence that radiologists have in each labeled region, modeling the inherent aleatoric uncertainty present in real-world clinical data. We demonstrate that incorporating these confidence values during training leads to improved segmentation performance. More importantly, we show that this enhanced segmentation quality translates into better performance on downstream clinically-critical tasks-specifically, estimating S/F oxygenation ratio values, classifying S/F ratio change, and predicting 30-day patient readmission. While we empirically evaluate many methods for exposing the uncertainty to the learning model, we find that a simple approach that trains a model on binarized labels obtained with a (60%) confidence threshold works well. Importantly, high thresholds work far better than a naive approach of a 50% threshold, indicating that training on very confident pixels is far more effective. Our study systematically investigates the impact of training with varying confidence thresholds, comparing not only segmentation metrics but also downstream clinical outcomes. These results suggest that label confidence is a valuable signal that, when properly leveraged, can significantly enhance the reliability and clinical utility of AI in medical imaging. 

**Abstract (ZH)**: 医学影像领域中， radiologists之间的观察者变异常常引入标签不确定性，特别是在依赖主观视觉解释的模态中。肺超声（LUS）是一个典型的例子——它经常呈现高度模糊和清晰结构并存的情况，即使对于经验丰富的临床医生来说，一致地标记这些区域也颇具挑战性。本文提出了一种新的方法，利用专家提供的每个像素的置信度值来进行标注和训练AI模型。我们设计了一种数据标注协议，捕捉放射科医生对每个标注区域的信心程度，从而表征现实临床数据中存在的固有不确定。我们证明，在训练过程中引入这些置信度值可以改善分割性能。更重要的是，我们展示了这种增强的分割质量在下游临床关键任务上的表现更好，具体包括估算S/F氧合比值、分类S/F比值变化以及预测30天内患者的再入院。我们在多种方法中实证评估了对学习模型暴露不确定性，发现使用60%置信度阈值获得的二值标签进行训练的效果很好。重要的是，高置信度阈值的效果远优于50%的简单阈值，表明使用非常置信的像素进行训练更为有效。本研究系统地探讨了使用不同置信度阈值训练的影响，不仅比较了分割指标，还比较了下游临床结果。这些结果表明，标签置信度是一个有价值的信号，当适当利用时，可以显著增强医学影像中AI的可靠性和临床实用性。 

---
# GRASPED: Graph Anomaly Detection using Autoencoder with Spectral Encoder and Decoder (Full Version) 

**Title (ZH)**: GRASPED：使用谱编码器和解码器的图异常检测自编码器（完整版） 

**Authors**: Wei Herng Choong, Jixing Liu, Ching-Yu Kao, Philip Sperl  

**Link**: [PDF](https://arxiv.org/pdf/2508.15633)  

**Abstract**: Graph machine learning has been widely explored in various domains, such as community detection, transaction analysis, and recommendation systems. In these applications, anomaly detection plays an important role. Recently, studies have shown that anomalies on graphs induce spectral shifts. Some supervised methods have improved the utilization of such spectral domain information. However, they remain limited by the scarcity of labeled data due to the nature of anomalies. On the other hand, existing unsupervised learning approaches predominantly rely on spatial information or only employ low-pass filters, thereby losing the capacity for multi-band analysis. In this paper, we propose Graph Autoencoder with Spectral Encoder and Spectral Decoder (GRASPED) for node anomaly detection. Our unsupervised learning model features an encoder based on Graph Wavelet Convolution, along with structural and attribute decoders. The Graph Wavelet Convolution-based encoder, combined with a Wiener Graph Deconvolution-based decoder, exhibits bandpass filter characteristics that capture global and local graph information at multiple scales. This design allows for a learning-based reconstruction of node attributes, effectively capturing anomaly information. Extensive experiments on several real-world graph anomaly detection datasets demonstrate that GRASPED outperforms current state-of-the-art models. 

**Abstract (ZH)**: 图机器学习在社区检测、交易分析和推荐系统等多种领域得到了广泛探索。在这些应用中，异常检测发挥着重要作用。近年来的研究表明，图上的异常会导致频谱偏移。一些监督方法通过利用这种频域信息提高了异常检测的效果，但它们仍然受限于可用标记数据的稀缺性。另一方面，现有的无监督学习方法主要依赖空间信息或仅使用低通滤波器，从而丧失了多频带分析的能力。在本文中，我们提出了一种基于图波let卷积的谱编码器和谱解码器的图自编码器（GRASPED）用于节点异常检测。该无监督学习模型采用基于图波let卷积的编码器和结构解码器以及属性解码器。基于图波let卷积的编码器与基于维纳图反卷积的解码器相结合，表现出带通滤波器的特性，可以多尺度捕获全局和局部图信息。这种设计能够让自编码器基于学习重构节点属性，有效捕获异常信息。在多个真实世界图异常检测数据集上的广泛实验表明，GRASPED优于当前最先进的模型。 

---
# Trained Miniatures: Low cost, High Efficacy SLMs for Sales & Marketing 

**Title (ZH)**: 训练微缩模型：销售与市场推广中的低成本高效率选择 

**Authors**: Ishaan Bhola, Mukunda NS, Sravanth Kurmala, Harsh Nandwani, Arihant Jain  

**Link**: [PDF](https://arxiv.org/pdf/2508.15617)  

**Abstract**: Large language models (LLMs) excel in text generation; however, these creative elements require heavy computation and are accompanied by a steep cost. Especially for targeted applications such as sales and marketing outreach, these costs are far from feasible. This paper introduces the concept of "Trained Miniatures" - Small Language Models(SLMs) fine-tuned for specific, high-value applications, generating similar domain-specific responses for a fraction of the cost. 

**Abstract (ZH)**: 小型训练模型：特定高价值应用的低成本文本生成 

---
# Are Virtual DES Images a Valid Alternative to the Real Ones? 

**Title (ZH)**: 虚拟DES图像是否是真实图像的有效替代方案？ 

**Authors**: Ana C. Perre, Luís A. Alexandre, Luís C. Freire  

**Link**: [PDF](https://arxiv.org/pdf/2508.15594)  

**Abstract**: Contrast-enhanced spectral mammography (CESM) is an imaging modality that provides two types of images, commonly known as low-energy (LE) and dual-energy subtracted (DES) images. In many domains, particularly in medicine, the emergence of image-to-image translation techniques has enabled the artificial generation of images using other images as input. Within CESM, applying such techniques to generate DES images from LE images could be highly beneficial, potentially reducing patient exposure to radiation associated with high-energy image acquisition. In this study, we investigated three models for the artificial generation of DES images (virtual DES): a pre-trained U-Net model, a U-Net trained end-to-end model, and a CycleGAN model. We also performed a series of experiments to assess the impact of using virtual DES images on the classification of CESM examinations into malignant and non-malignant categories. To our knowledge, this is the first study to evaluate the impact of virtual DES images on CESM lesion classification. The results demonstrate that the best performance was achieved with the pre-trained U-Net model, yielding an F1 score of 85.59% when using the virtual DES images, compared to 90.35% with the real DES images. This discrepancy likely results from the additional diagnostic information in real DES images, which contributes to a higher classification accuracy. Nevertheless, the potential for virtual DES image generation is considerable and future advancements may narrow this performance gap to a level where exclusive reliance on virtual DES images becomes clinically viable. 

**Abstract (ZH)**: 增强对比度光谱乳腺成像中虚拟双能减影图像的人工生成及其对病变分类的影响研究 

---
# LoUQAL: Low-fidelity informed Uncertainty Quantification for Active Learning in the chemical configuration space 

**Title (ZH)**: Low-fidelity Informed Uncertainty Quantification for Active Learning in the Chemical Configuration Space 

**Authors**: Vivin Vinod, Peter Zaspel  

**Link**: [PDF](https://arxiv.org/pdf/2508.15577)  

**Abstract**: Uncertainty quantification is an important scheme in active learning techniques, including applications in predicting quantum chemical properties. In quantum chemical calculations, there exists the notion of a fidelity, a less accurate computation is accessible at a cheaper computational cost. This work proposes a novel low-fidelity informed uncertainty quantification for active learning with applications in predicting diverse quantum chemical properties such as excitation energies and \textit{ab initio} potential energy surfaces. Computational experiments are carried out in order to assess the proposed method with results demonstrating that models trained with the novel method outperform alternatives in terms of empirical error and number of iterations required. The effect of the choice of fidelity is also studied to perform a thorough benchmark. 

**Abstract (ZH)**: 不确定性量化是主动学习技术中一种重要的方案，包括在预测量子化学性质中的应用。在量子化学计算中，存在忠实度的概念，较低忠实度的计算在计算成本较低的情况下可以获得。本工作提出了一种新颖的低忠实度导向不确定性量化方法，应用于预测多种量子化学性质，如激发能量和从头算势能面。通过计算实验评估提出的方法，结果显示使用新颖方法训练的模型在经验误差和所需迭代次数方面优于替代方法。还研究了忠实度选择的影响，以进行全面基准测试。 

---
# LLM-Driven Self-Refinement for Embodied Drone Task Planning 

**Title (ZH)**: 基于LLM的自主完善型无人机任务规划 

**Authors**: Deyu Zhang, Xicheng Zhang, Jiahao Li, Tingting Long, Xunhua Dai, Yongjian Fu, Jinrui Zhang, Ju Ren, Yaoxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15501)  

**Abstract**: We introduce SRDrone, a novel system designed for self-refinement task planning in industrial-grade embodied drones. SRDrone incorporates two key technical contributions: First, it employs a continuous state evaluation methodology to robustly and accurately determine task outcomes and provide explanatory feedback. This approach supersedes conventional reliance on single-frame final-state assessment for continuous, dynamic drone operations. Second, SRDrone implements a hierarchical Behavior Tree (BT) modification model. This model integrates multi-level BT plan analysis with a constrained strategy space to enable structured reflective learning from experience. Experimental results demonstrate that SRDrone achieves a 44.87% improvement in Success Rate (SR) over baseline methods. Furthermore, real-world deployment utilizing an experience base optimized through iterative self-refinement attains a 96.25% SR. By embedding adaptive task refinement capabilities within an industrial-grade BT planning framework, SRDrone effectively integrates the general reasoning intelligence of Large Language Models (LLMs) with the stringent physical execution constraints inherent to embodied drones. Code is available at this https URL. 

**Abstract (ZH)**: 我们介绍了SRDrone，一种用于工业级实体无人机自我完善任务规划的新型系统。SRDrone包含两项关键技术贡献：首先，它采用连续状态评估方法，以稳健和准确地确定任务结果并提供解释性反馈。这种方法取代了依赖单一帧最终状态评估的做法，适用于连续动态无人机操作。其次，SRDrone实现了分层行为树(BT)修改模型。该模型将多级BT计划分析与受限策略空间相结合，以实现结构化的反思性学习。实验结果显示，与基线方法相比，SRDrone将成功率(SR)提高了44.87%。此外，通过迭代自我完善优化的经验基底在其实际部署中实现了96.25%的SR。通过在工业级BT规划框架中嵌入自适应任务完善能力，SRDrone有效结合了大型语言模型（LLMs）的通用推理智能与实体无人机固有的严格物理执行约束。代码可在以下链接获取。 

---
# LGMSNet: Thinning a medical image segmentation model via dual-level multiscale fusion 

**Title (ZH)**: LGMSNet: 通过双级别多尺度融合方法稀疏化医疗图像分割模型 

**Authors**: Chengqi Dong, Fenghe Tang, Rongge Mao, Xinpei Gao, S.Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.15476)  

**Abstract**: Medical image segmentation plays a pivotal role in disease diagnosis and treatment planning, particularly in resource-constrained clinical settings where lightweight and generalizable models are urgently needed. However, existing lightweight models often compromise performance for efficiency and rarely adopt computationally expensive attention mechanisms, severely restricting their global contextual perception capabilities. Additionally, current architectures neglect the channel redundancy issue under the same convolutional kernels in medical imaging, which hinders effective feature extraction. To address these challenges, we propose LGMSNet, a novel lightweight framework based on local and global dual multiscale that achieves state-of-the-art performance with minimal computational overhead. LGMSNet employs heterogeneous intra-layer kernels to extract local high-frequency information while mitigating channel redundancy. In addition, the model integrates sparse transformer-convolutional hybrid branches to capture low-frequency global information. Extensive experiments across six public datasets demonstrate LGMSNet's superiority over existing state-of-the-art methods. In particular, LGMSNet maintains exceptional performance in zero-shot generalization tests on four unseen datasets, underscoring its potential for real-world deployment in resource-limited medical scenarios. The whole project code is in this https URL. 

**Abstract (ZH)**: 医疗图像分割在疾病诊断和治疗规划中扮演着关键角色，特别是在资源受限的临床环境中，需要轻量级且通用的模型。然而，现有的轻量级模型往往在性能与效率之间做出妥协，并且很少采用计算成本高的注意力机制，严重限制了它们的全局上下文感知能力。此外，当前架构忽视了医学成像中相同卷积核下的信道冗余问题，阻碍了有效的特征提取。为了解决这些挑战，我们提出了LGMSNet，这是一种基于局部和全局双多尺度的新型轻量级框架，能够在最小的计算开销下达到最先进的性能。LGMSNet 使用异构内层卷积核来提取局部高频信息并降低信道冗余。此外，该模型集成了稀疏变压器-卷积混合分支以捕获低频全局信息。在六个公开数据集上的广泛实验表明，LGMSNet 在现有最先进的方法中具有优势。特别是，LGMSNet 在四个未见过的数据集上的零样本泛化测试中保持出色的性能，突显了其在资源受限的医疗场景中实际部署的潜力。整个项目代码见此链接：https://xxxxxxxxxxxxxxx。 

---
# Subjective Behaviors and Preferences in LLM: Language of Browsing 

**Title (ZH)**: LLM冲浪过程中主观行为与偏好的语言表达 

**Authors**: Sai Sundaresan, Harshita Chopra, Atanu R. Sinha, Koustava Goswami, Nagasai Saketh Naidu, Raghav Karan, N Anushka  

**Link**: [PDF](https://arxiv.org/pdf/2508.15474)  

**Abstract**: A Large Language Model (LLM) offers versatility across domains and tasks, purportedly benefiting users with a wide variety of behaviors and preferences. We question this perception about an LLM when users have inherently subjective behaviors and preferences, as seen in their ubiquitous and idiosyncratic browsing of websites or apps. The sequential behavior logs of pages, thus generated, form something akin to each user's self-constructed "language", albeit without the structure and grammar imbued in natural languages. We ask: (i) Can a small LM represent the "language of browsing" better than a large LM? (ii) Can an LM with a single set of parameters (or, single LM) adequately capture myriad users' heterogeneous, subjective behaviors and preferences? (iii) Can a single LM with high average performance, yield low variance in performance to make alignment good at user level? We introduce clusterwise LM training, HeTLM (Heterogeneity aware Training of Language Model), appropriate for subjective behaviors. We find that (i) a small LM trained using a page-level tokenizer outperforms large pretrained or finetuned LMs; (ii) HeTLM with heterogeneous cluster specific set of parameters outperforms a single LM of the same family, controlling for the number of parameters; and (iii) a higher mean and a lower variance in generation ensues, implying improved alignment. 

**Abstract (ZH)**: 大语言模型（LLM）在各个领域和任务中展现出灵活性，被认为是能够满足具有广泛行为和偏好的用户。当用户的行为和偏好本质上是主观的，如他们在网站或应用程序上的普遍且异质的浏览行为时，我们质疑这一关于LLM的看法。由此生成的页面序贯行为日志形成了类似于每个用户自己构建的“语言”，尽管缺乏自然语言中的结构和语法。我们询问：（i）小型LM能否比大型LM更好地代表“浏览语言”？（ii）单个参数集（或单个LM）能否充分捕捉众多用户异质且主观的行为和偏好？（iii）一个具有高平均性能且性能方差低的单个LM能否在用户级别实现良好的对齐？为此，我们引入了适应主观行为的聚类语言模型训练方法——HeTLM（具有异质性意识的语言模型训练）。我们发现：（i）使用页面级分词器训练的小型LM优于大型预训练或微调LM；（ii）具有异质性簇特定参数集的HeTLM优于相同家族的单个LM，控制参数数量；（iii）生成的均值增加且方差降低，表明对齐有所提高。 

---
# RadReason: Radiology Report Evaluation Metric with Reasons and Sub-Scores 

**Title (ZH)**: RadReason: 医学影像报告评价指标附带原因和子分值 

**Authors**: Yingshu Li, Yunyi Liu, Lingqiao Liu, Lei Wang, Luping Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.15464)  

**Abstract**: Evaluating automatically generated radiology reports remains a fundamental challenge due to the lack of clinically grounded, interpretable, and fine-grained metrics. Existing methods either produce coarse overall scores or rely on opaque black-box models, limiting their usefulness in real-world clinical workflows. We introduce RadReason, a novel evaluation framework for radiology reports that not only outputs fine-grained sub-scores across six clinically defined error types, but also produces human-readable justifications that explain the rationale behind each score. Our method builds on Group Relative Policy Optimization and incorporates two key innovations: (1) Sub-score Dynamic Weighting, which adaptively prioritizes clinically challenging error types based on live F1 statistics; and (2) Majority-Guided Advantage Scaling, which adjusts policy gradient updates based on prompt difficulty derived from sub-score agreement. Together, these components enable more stable optimization and better alignment with expert clinical judgment. Experiments on the ReXVal benchmark show that RadReason surpasses all prior offline metrics and achieves parity with GPT-4-based evaluations, while remaining explainable, cost-efficient, and suitable for clinical deployment. Code will be released upon publication. 

**Abstract (ZH)**: 评估自动生成的放射学报告仍然是一个基本挑战，由于缺乏临床相关、可解释性和精细化的评价指标。现有方法要么生成粗糙的整体评分，要么依赖于不透明的黑盒模型，限制了其在实际临床工作流程中的实用性。我们引入了RadReason，这是一种新的放射学报告评价框架，不仅能输出六种临床定义错误类型的精细化子评分，还能提供易于理解的解释，说明每个评分的理据。我们的方法基于Group Relative Policy Optimization，并包含两项关键创新：（1）子评分动态加权，根据实时F1统计自适应优先考虑临床挑战性错误类型；（2）多数指导优势放大，基于子评分一致性和提示难度调整策略梯度更新。这些组件共同使优化更加稳定，并更好地与专家临床判断相契合。在ReXVal基准上的实验表明，RadReason超越了所有之前的离线指标，并在可解释性、成本效益和适用于临床部署方面与基于GPT-4的评估相当。代码将在发表后公开。 

---
# A Solvable Molecular Switch Model for Stable Temporal Information Processing 

**Title (ZH)**: 可解的分子开关模型用于稳定的时间信息处理 

**Authors**: H. I. Nurdin, C. A. Nijhuis  

**Link**: [PDF](https://arxiv.org/pdf/2508.15451)  

**Abstract**: This paper studies an input-driven one-state differential equation model initially developed for an experimentally demonstrated dynamic molecular switch that switches like synapses in the brain do. The linear-in-the-state and nonlinear-in-the-input model is exactly solvable, and it is shown that it also possesses mathematical properties of convergence and fading memory that enable stable processing of time-varying inputs by nonlinear dynamical systems. Thus, the model exhibits the co-existence of biologically-inspired behavior and desirable mathematical properties for stable learning on sequential data. The results give theoretical support for the use of the dynamic molecular switches as computational units in deep cascaded/layered feedforward and recurrent architectures as well as other more general structures for neuromorphic computing. They could also inspire more general exactly solvable models that can be fitted to emulate arbitrary physical devices which can mimic brain-inspired behaviour and perform stable computation on input signals. 

**Abstract (ZH)**: 本文研究了一种由实验演示的动力分子开关启发的输入驱动单状态微分方程模型，该模型类似于大脑中的突触。该模型是一类线性依赖于状态且非线性依赖于输入的模型，可以精确求解，并且证明该模型具有收敛性和衰减记忆性的数学特性，这些特性使得非线性动力系统能够稳定处理时间变化的输入。因此，该模型同时展示了受生物启发的行为和用于序列数据稳定学习的 desirable 数学性质。研究结果为将动力分子开关作为深度级联/分层前馈和循环架构以及其他更一般结构中的计算单元提供了理论支持，也有可能启发一种更通用的可以拟合以模拟任意物理设备并执行输入信号稳定计算的精确可解模型。 

---
# Reliable Unlearning Harmful Information in LLMs with Metamorphosis Representation Projection 

**Title (ZH)**: 使用变形表示投影实现可信删除LLM中有害信息 

**Authors**: Chengcan Wu, Zeming Wei, Huanran Chen, Yinpeng Dong, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.15449)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive performance in various domains and tasks, concerns about their safety are becoming increasingly severe. In particular, since models may store unsafe knowledge internally, machine unlearning has emerged as a representative paradigm to ensure model safety. Existing approaches employ various training techniques, such as gradient ascent and negative preference optimization, in attempts to eliminate the influence of undesired data on target models. However, these methods merely suppress the activation of undesired data through parametric training without completely eradicating its informational traces within the model. This fundamental limitation makes it difficult to achieve effective continuous unlearning, rendering these methods vulnerable to relearning attacks. To overcome these challenges, we propose a Metamorphosis Representation Projection (MRP) approach that pioneers the application of irreversible projection properties to machine unlearning. By implementing projective transformations in the hidden state space of specific network layers, our method effectively eliminates harmful information while preserving useful knowledge. Experimental results demonstrate that our approach enables effective continuous unlearning and successfully defends against relearning attacks, achieving state-of-the-art performance in unlearning effectiveness while preserving natural performance. Our code is available in this https URL. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在各种领域和任务中展现了 impressive 的性能，但对其安全性的担忧日益严重。特别是由于模型可能内部存储了不安全的知识，机器遗忘已作为一个代表性范式出现，以确保模型的安全性。现有方法通过使用梯度上升、负偏好优化等各种训练技术尝试消除目标模型中不希望数据的影响。然而，这些方法仅通过参数训练抑制不希望数据的激活，而没有完全消除其在模型中的信息痕迹，这一根本限制使得实现有效的连续遗忘变得困难，从而使这些方法容易受到重学攻击。为克服这些挑战，我们提出了一种变形表示投影（MRP）方法，该方法首次将不可逆投影性质应用于机器遗忘。通过在特定网络层的隐藏状态空间中实施投影变换，我们的方法有效消除了有害信息的同时保留了有用的知识。实验结果表明，我们的方法能够实现有效的连续遗忘并成功抵御重学攻击，实现了遗忘效果的最优性能，同时保持自然性能。我们的代码可在以下链接获取：this https URL。 

---
# Mitigating Hallucinations in LM-Based TTS Models via Distribution Alignment Using GFlowNets 

**Title (ZH)**: 基于GFlowNets的分布对齐方法减轻LM-Based TTS模型中的幻觉现象 

**Authors**: Chenlin Liu, Minghui Fang, Patrick Zhang, Wei Zhou, Jie Gao, Jiqing Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.15442)  

**Abstract**: Language Model (LM)-based Text-to-Speech (TTS) systems often generate hallucinated speech that deviates from input text. Existing mitigation strategies either demand excessive training resources or introduce significant inference latency. In this paper, we propose GFlOwNet-guided distribution AlignmenT (GOAT) for LM-based TTS, a post-training framework that mitigates hallucinations without relying on massive resources or inference cost. Specifically, we first conduct an uncertainty analysis, revealing a strong positive correlation between hallucination and model uncertainty. Based on this, we reformulate TTS generation as a trajectory flow optimization problem and introduce an enhanced Subtrajectory Balance objective together with a sharpened internal reward as target distribution. We further integrate reward temperature decay and learning rate optimization for stability and performance balance. Extensive experiments show that GOAT reduce over 50% character error rates on challenging test cases and lowering uncertainty by up to 58%, demonstrating its strong generalization ability and effectiveness. 

**Abstract (ZH)**: 基于语言模型的文本到语音系统中的GOAT引导分布对齐：一种无需大量资源的后训练框架以减轻幻听现象 

---
# Test-time Corpus Feedback: From Retrieval to RAG 

**Title (ZH)**: 测试时语料库反馈：从检索到RAG 

**Authors**: Mandeep Rathee, Venktesh V, Sean MacAvaney, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2508.15437)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a standard framework for knowledge-intensive NLP tasks, combining large language models (LLMs) with document retrieval from external corpora. Despite its widespread use, most RAG pipelines continue to treat retrieval and reasoning as isolated components, retrieving documents once and then generating answers without further interaction. This static design often limits performance on complex tasks that require iterative evidence gathering or high-precision retrieval. Recent work in both the information retrieval (IR) and NLP communities has begun to close this gap by introducing adaptive retrieval and ranking methods that incorporate feedback. In this survey, we present a structured overview of advanced retrieval and ranking mechanisms that integrate such feedback. We categorize feedback signals based on their source and role in improving the query, retrieved context, or document pool. By consolidating these developments, we aim to bridge IR and NLP perspectives and highlight retrieval as a dynamic, learnable component of end-to-end RAG systems. 

**Abstract (ZH)**: 检索增强生成（RAG）已成为知识密集型NLP任务的标准框架，结合了大型语言模型（LLMs）和从外部语料库检索文档。尽管其广泛应用，大多数RAG管道仍继续将检索和推理视为独立组件，只检索一次文档，然后生成答案而不再进一步交互。这种静态设计往往限制了在需要迭代证据收集或高精度检索的复杂任务上的性能。信息检索（IR）和NLP领域近期工作已经开始通过引入适应性检索和排名方法来克服这一局限，这些方法包括反馈机制。在这篇综述中，我们提供了一个结构化的高级检索和排名机制概述，这些机制整合了反馈。我们根据反馈信号的来源及其在改进查询、检索上下文或文档池方面的作用对其进行分类。通过整合这些发展，我们旨在弥合IR和NLP视角，并突出检索作为端到端RAG系统中的动态、可学习组件的重要性。 

---
# An Empirical Study of Knowledge Distillation for Code Understanding Tasks 

**Title (ZH)**: 代码理解任务中的知识精炼实证研究 

**Authors**: Ruiqi Wang, Zezhou Yang, Cuiyun Gao, Xin Xia, Qing Liao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15423)  

**Abstract**: Pre-trained language models (PLMs) have emerged as powerful tools for code understanding. However, deploying these PLMs in large-scale applications faces practical challenges due to their computational intensity and inference latency. Knowledge distillation (KD), a promising model compression and acceleration technique, addresses these limitations by transferring knowledge from large teacher models to compact student models, enabling efficient inference while preserving most of the teacher models' capabilities. While this technique has shown remarkable success in natural language processing and computer vision domains, its potential for code understanding tasks remains largely underexplored.
In this paper, we systematically investigate the effectiveness and usage of KD in code understanding tasks. Our study encompasses two popular types of KD methods, i.e., logit-based and feature-based KD methods, experimenting across eight student models and two teacher PLMs from different domains on three downstream tasks. The experimental results indicate that KD consistently offers notable performance boosts across student models with different sizes compared with standard fine-tuning. Notably, code-specific PLM demonstrates better effectiveness as the teacher model. Among all KD methods, the latest feature-based KD methods exhibit superior performance, enabling student models to retain up to 98% teacher performance with merely 5% parameters. Regarding student architecture, our experiments reveal that similarity with teacher architecture does not necessarily lead to better performance. We further discuss the efficiency and behaviors in the KD process and inference, summarize the implications of findings, and identify promising future directions. 

**Abstract (ZH)**: 预训练语言模型（PLMs）已成为代码理解的强大工具。然而，在大规模应用中部署这些PLMs面临实际挑战，因为它们的计算强度和推理延迟较高。知识蒸馏（KD），一种有前途的模型压缩和加速技术，通过将大型教师模型的知识转移到紧凑的学生模型中，解决了这些限制，实现了高效推理同时保持大多数教师模型的能力。尽管这一技术在自然语言处理和计算机视觉领域取得了显著成功，但其在代码理解任务中的潜力尚待充分探索。

在本文中，我们系统地探讨了KD在代码理解任务中的有效性和使用方法。我们的研究涵盖了两种流行的KD方法，即基于logit和基于特征的KD方法，在八个学生模型和两种来自不同领域的教师PLMs上对三个下游任务进行了实验。实验结果表明，与标准微调相比，KD在不同规模的学生模型中提供了显著的性能提升。值得注意的是，代码专用的PLM作为教师模型时效果更佳。在所有KD方法中，最新的基于特征的KD方法表现出最佳性能，使得学生模型仅使用5%的参数即可保留高达98%的教师模型性能。关于学生架构，我们的实验表明，与教师架构的相似性并不一定导致更好的性能。我们进一步讨论了KD过程和推理中的效率与行为，总结了研究发现的含义，并指出了有前景的未来方向。 

---
# LLaSO: A Foundational Framework for Reproducible Research in Large Language and Speech Model 

**Title (ZH)**: LLaSO: 用于大规模语言和语音模型可再现研究的基本框架 

**Authors**: Yirong Sun, Yizhong Geng, Peidong Wei, Yanjun Chen, Jinghan Yang, Rongfei Chen, Wei Zhang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15418)  

**Abstract**: The development of Large Speech-Language Models (LSLMs) has been slowed by fragmented architectures and a lack of transparency, hindering the systematic comparison and reproducibility of research. Unlike in the vision-language domain, the LSLM field suffers from the common practice of releasing model weights without their corresponding training data and configurations. To address these critical gaps, we introduce LLaSO, the first fully open, end-to-end framework for large-scale speech-language modeling. LLaSO provides the community with three essential resources: (1) LLaSO-Align, a 12M-instance speech-text alignment corpus; (2) LLaSO-Instruct, a 13.5M-instance multi-task instruction-tuning dataset; and (3) LLaSO-Eval, a reproducible benchmark for standardized evaluation. To validate our framework, we build and release LLaSO-Base, a 3.8B-parameter reference model trained exclusively on our public data. It achieves a normalized score of 0.72, establishing a strong, reproducible baseline that surpasses comparable models. Our analysis reveals that while broader training coverage enhances performance, significant generalization gaps persist on unseen tasks, particularly in pure audio scenarios. By releasing the complete stack of data, benchmarks, and models, LLaSO establishes a foundational open standard to unify research efforts and accelerate community-driven progress in LSLMs. We release the code, dataset, pretrained models, and results in this https URL. 

**Abstract (ZH)**: 大型语音语言模型（LSLMs）的发展受到碎片化架构和缺乏透明性的阻碍，妨碍了研究的系统比较和再现性。与视觉语言领域不同，LSLM领域普遍存在问题，即在不发布其对应训练数据和配置的情况下发布模型权重。为填补这些关键空白，我们介绍了LLaSO，这是首个全面开源的端到端大规模语音语言建模框架。LLaSO为社区提供了三项重要资源：（1）LLaSO-Align，一个包含1200万实例的语音-文本对齐语料库；（2）LLaSO-Instruct，一个包含1350万实例的多任务指令调整数据集；（3）LLaSO-Eval，一个可重现的标准评估基准。为验证我们的框架，我们构建并发布了LLaSO-Base，一个仅在我们公开数据上训练的380亿参数参考模型，其标准化得分为0.72，建立了强健的可再现基线，超越了同类模型。我们的分析表明，虽然更广泛的训练覆盖面可以提高性能，但在未见任务上仍存在显著的泛化差距，特别是在纯音频场景中。通过发布完整的数据集、基准和模型堆栈，LLaSO建立了统一研究努力的基础开放标准，加速了社区驱动的大规模语音语言模型进展。我们在https://link.toLLLLO.release/发布了代码、数据集、预训练模型和结果。 

---
# Bridging Generalization and Personalization in Wearable Human Activity Recognition via On-Device Few-Shot Learning 

**Title (ZH)**: 基于设备端少样本学习实现可穿戴人体活动识别的一般化与个性化桥梁构建 

**Authors**: Pixi Kang, Julian Moosmann, Mengxi Liu, Bo Zhou, Michele Magno, Paul Lukowicz, Sizhen Bian  

**Link**: [PDF](https://arxiv.org/pdf/2508.15413)  

**Abstract**: Human Activity Recognition (HAR) using wearable devices has advanced significantly in recent years, yet its generalization remains limited when models are deployed to new users. This degradation in performance is primarily due to user-induced concept drift (UICD), highlighting the importance of efficient personalization. In this paper, we present a hybrid framework that first generalizes across users and then rapidly adapts to individual users using few-shot learning directly on-device. By updating only the classifier layer with user-specific data, our method achieves robust personalization with minimal computational and memory overhead. We implement this framework on the energy-efficient RISC-V-based GAP9 microcontroller and validate it across three diverse HAR scenarios: RecGym, QVAR-Gesture, and Ultrasound-Gesture. Post-deployment adaptation yields consistent accuracy improvements of 3.73\%, 17.38\%, and 3.70\% respectively. These results confirm that fast, lightweight, and effective personalization is feasible on embedded platforms, paving the way for scalable and user-aware HAR systems in the wild \footnote{this https URL}. 

**Abstract (ZH)**: 使用可穿戴设备进行人类活动识别（HAR）近年来取得了显著进展，但在部署到新用户时其泛化能力仍然有限。这种性能下降主要是由于用户诱导的概念漂移（UICD），突显了高效个性化的重要性。本文提出了一种混合框架，首先在用户之间进行泛化，然后利用直接在设备上进行的少-shot学习快速适应个别用户。通过仅使用用户特定数据更新分类器层，我们的方法实现了稳健的个性化，同时最小化了计算和内存开销。该框架在能效型RISC-V基础的GAP9微控制器上实现，并在三种不同的HAR场景下进行了验证：RecGym、QVAR-Gesture和Ultrasound-Gesture。部署后的适应性调整分别提高了3.73%、17.38%和3.70%的准确性。这些结果证实，在嵌入式平台上实现快速、轻量级和有效的个性化是可行的，为野生环境下的可扩展和用户意识强的HAR系统铺平了道路。 

---
# When Audio and Text Disagree: Revealing Text Bias in Large Audio-Language Models 

**Title (ZH)**: 当音频和文本不一致时：揭示大型音频语言模型中的文本偏见 

**Authors**: Cheng Wang, Gelei Deng, Xianglin Yang, Han Qiu, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15407)  

**Abstract**: Large Audio-Language Models (LALMs) are enhanced with audio perception capabilities, enabling them to effectively process and understand multimodal inputs that combine audio and text. However, their performance in handling conflicting information between audio and text modalities remains largely unexamined. This paper introduces MCR-BENCH, the first comprehensive benchmark specifically designed to evaluate how LALMs prioritize information when presented with inconsistent audio-text pairs. Through extensive evaluation across diverse audio understanding tasks, we reveal a concerning phenomenon: when inconsistencies exist between modalities, LALMs display a significant bias toward textual input, frequently disregarding audio evidence. This tendency leads to substantial performance degradation in audio-centric tasks and raises important reliability concerns for real-world applications. We further investigate the influencing factors of text bias, and explore mitigation strategies through supervised finetuning, and analyze model confidence patterns that reveal persistent overconfidence even with contradictory inputs. These findings underscore the need for improved modality balance during training and more sophisticated fusion mechanisms to enhance the robustness when handling conflicting multi-modal inputs. The project is available at this https URL. 

**Abstract (ZH)**: 大型音频-语言模型（LALMs）通过增强音频感知能力，能够有效处理和理解结合了音频和文本的多模态输入。然而，它们在处理音频和文本模态之间矛盾信息方面的表现尚未得到充分研究。本文介绍了MCR-BENCH，这是首个专门设计用于评估LALMs在面对不一致的音频-文本配对时如何优先处理信息的综合性基准测试。通过在多样化的音频理解任务中进行广泛评估，我们揭示了一个令人担忧的现象：当模态之间存在不一致时，LALMs表现出对文本输入的显著偏向，经常忽视音频证据。这种倾向导致了以音频为中心任务的重大性能下降，并对实际应用的可靠性提出了重要质疑。我们进一步探讨了文本偏向的影响因素，并通过监督微调探索缓解策略，分析模型信心模式，揭示即使在矛盾输入的情况下，模型仍然表现出顽固的高信心。这些发现强调了在训练中改进模态平衡和开发更复杂的融合机制的必要性，以增强处理矛盾多模态输入的健壮性。该项目可在以下链接访问：此httpsURL。 

---
# Hybrid Least Squares/Gradient Descent Methods for DeepONets 

**Title (ZH)**: 混合最小二乘/梯度下降方法用于DeepONets 

**Authors**: Jun Choi, Chang-Ock Lee, Minam Moon  

**Link**: [PDF](https://arxiv.org/pdf/2508.15394)  

**Abstract**: We propose an efficient hybrid least squares/gradient descent method to accelerate DeepONet training. Since the output of DeepONet can be viewed as linear with respect to the last layer parameters of the branch network, these parameters can be optimized using a least squares (LS) solve, and the remaining hidden layer parameters are updated by means of gradient descent form. However, building the LS system for all possible combinations of branch and trunk inputs yields a prohibitively large linear problem that is infeasible to solve directly. To address this issue, our method decomposes the large LS system into two smaller, more manageable subproblems $\unicode{x2014}$ one for the branch network and one for the trunk network $\unicode{x2014}$ and solves them separately. This method is generalized to a broader type of $L^2$ loss with a regularization term for the last layer parameters, including the case of unsupervised learning with physics-informed loss. 

**Abstract (ZH)**: 我们提出了一种高效混合最小二乘/梯度下降方法来加速DeepONet训练。 

---
# Bladder Cancer Diagnosis with Deep Learning: A Multi-Task Framework and Online Platform 

**Title (ZH)**: 基于深度学习的膀胱癌诊断：多任务框架与在线平台 

**Authors**: Jinliang Yu, Mingduo Xie, Yue Wang, Tianfan Fu, Xianglai Xu, Jiajun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15379)  

**Abstract**: Clinical cystoscopy, the current standard for bladder cancer diagnosis, suffers from significant reliance on physician expertise, leading to variability and subjectivity in diagnostic outcomes. There is an urgent need for objective, accurate, and efficient computational approaches to improve bladder cancer diagnostics.
Leveraging recent advancements in deep learning, this study proposes an integrated multi-task deep learning framework specifically designed for bladder cancer diagnosis from cystoscopic images. Our framework includes a robust classification model using EfficientNet-B0 enhanced with Convolutional Block Attention Module (CBAM), an advanced segmentation model based on ResNet34-UNet++ architecture with self-attention mechanisms and attention gating, and molecular subtyping using ConvNeXt-Tiny to classify molecular markers such as HER-2 and Ki-67. Additionally, we introduce a Gradio-based online diagnostic platform integrating all developed models, providing intuitive features including multi-format image uploads, bilingual interfaces, and dynamic threshold adjustments.
Extensive experimentation demonstrates the effectiveness of our methods, achieving outstanding accuracy (93.28%), F1-score (82.05%), and AUC (96.41%) for classification tasks, and exceptional segmentation performance indicated by a Dice coefficient of 0.9091. The online platform significantly improved the accuracy, efficiency, and accessibility of clinical bladder cancer diagnostics, enabling practical and user-friendly deployment. The code is publicly available.
Our multi-task framework and integrated online tool collectively advance the field of intelligent bladder cancer diagnosis by improving clinical reliability, supporting early tumor detection, and enabling real-time diagnostic feedback. These contributions mark a significant step toward AI-assisted decision-making in urology. 

**Abstract (ZH)**: 临床膀胱镜检查是目前膀胱癌诊断的标准方法，但严重依赖医师 expertise，导致诊断结果的变异性与主观性。迫切需要客观、准确且高效的计算方法以提高膀胱癌诊断效果。
本研究利用近期深度学习的进展，提出了一种专门针对膀胱癌从膀胱镜图像进行诊断的集成多任务深度学习框架。该框架包括使用增强的EfficientNet-B0和Convolutional Block Attention Module (CBAM) 的 robust分类模型、基于ResNet34-UNet++架构配备自注意力机制和注意力门控的高级分割模型，以及使用ConvNeXt-Tiny进行分子亚型分类，以区分如HER-2和Ki-67等分子标记。此外，我们引入了一个基于Gradio的在线诊断平台，集成了所有开发的模型，并提供了包括多格式图像上传、双语界面和动态阈值调整在内的直观功能。
广泛的实验证明了本方法的有效性，在分类任务中取得了卓越的准确率（93.28%）、F1分数（82.05%）和AUC（96.41%），分割性能由Dice系数0.9091表示。在线平台显著提高了临床膀胱癌诊断的准确率、效率和可访问性，实现了实用且用户友好的部署。代码已公开。
本多任务框架和集成在线工具共同推动了智能膀胱癌诊断领域的发展，通过提高临床可靠性、支持早期肿瘤检测以及提供实时诊断反馈。这些贡献标志着AI辅助决策在泌尿学中的重要一步。 

---
# EvoFormer: Learning Dynamic Graph-Level Representations with Structural and Temporal Bias Correction 

**Title (ZH)**: EvoFormer：学习具有结构和时间偏置修正的动力学图级表示 

**Authors**: Haodi Zhong, Liuxin Zou, Di Wang, Bo Wang, Zhenxing Niu, Quan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15378)  

**Abstract**: Dynamic graph-level embedding aims to capture structural evolution in networks, which is essential for modeling real-world scenarios. However, existing methods face two critical yet under-explored issues: Structural Visit Bias, where random walk sampling disproportionately emphasizes high-degree nodes, leading to redundant and noisy structural representations; and Abrupt Evolution Blindness, the failure to effectively detect sudden structural changes due to rigid or overly simplistic temporal modeling strategies, resulting in inconsistent temporal embeddings. To overcome these challenges, we propose EvoFormer, an evolution-aware Transformer framework tailored for dynamic graph-level representation learning. To mitigate Structural Visit Bias, EvoFormer introduces a Structure-Aware Transformer Module that incorporates positional encoding based on node structural roles, allowing the model to globally differentiate and accurately represent node structures. To overcome Abrupt Evolution Blindness, EvoFormer employs an Evolution-Sensitive Temporal Module, which explicitly models temporal evolution through a sequential three-step strategy: (I) Random Walk Timestamp Classification, generating initial timestamp-aware graph-level embeddings; (II) Graph-Level Temporal Segmentation, partitioning the graph stream into segments reflecting structurally coherent periods; and (III) Segment-Aware Temporal Self-Attention combined with an Edge Evolution Prediction task, enabling the model to precisely capture segment boundaries and perceive structural evolution trends, effectively adapting to rapid temporal shifts. Extensive evaluations on five benchmark datasets confirm that EvoFormer achieves state-of-the-art performance in graph similarity ranking, temporal anomaly detection, and temporal segmentation tasks, validating its effectiveness in correcting structural and temporal biases. 

**Abstract (ZH)**: EvoFormer：一种关注演化的Transformer框架用于动态图级表示学习 

---
# Image-Conditioned 3D Gaussian Splat Quantization 

**Title (ZH)**: 基于图像条件的3D高斯点量化 

**Authors**: Xinshuang Liu, Runfa Blark Li, Keito Suzuki, Truong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15372)  

**Abstract**: 3D Gaussian Splatting (3DGS) has attracted considerable attention for enabling high-quality real-time rendering. Although 3DGS compression methods have been proposed for deployment on storage-constrained devices, two limitations hinder archival use: (1) they compress medium-scale scenes only to the megabyte range, which remains impractical for large-scale scenes or extensive scene collections; and (2) they lack mechanisms to accommodate scene changes after long-term archival. To address these limitations, we propose an Image-Conditioned Gaussian Splat Quantizer (ICGS-Quantizer) that substantially enhances compression efficiency and provides adaptability to scene changes after archiving. ICGS-Quantizer improves quantization efficiency by jointly exploiting inter-Gaussian and inter-attribute correlations and by using shared codebooks across all training scenes, which are then fixed and applied to previously unseen test scenes, eliminating the overhead of per-scene codebooks. This approach effectively reduces the storage requirements for 3DGS to the kilobyte range while preserving visual fidelity. To enable adaptability to post-archival scene changes, ICGS-Quantizer conditions scene decoding on images captured at decoding time. The encoding, quantization, and decoding processes are trained jointly, ensuring that the codes, which are quantized representations of the scene, are effective for conditional decoding. We evaluate ICGS-Quantizer on 3D scene compression and 3D scene updating. Experimental results show that ICGS-Quantizer consistently outperforms state-of-the-art methods in compression efficiency and adaptability to scene changes. Our code, model, and data will be publicly available on GitHub. 

**Abstract (ZH)**: 基于图像条件的Gaussian斑点量化器（ICGS-Quantizer）：实现高效且适应场景变化的3D场景压缩与更新 

---
# Unveiling Trust in Multimodal Large Language Models: Evaluation, Analysis, and Mitigation 

**Title (ZH)**: 揭示多模态大型语言模型中的信任：评估、分析与缓解 

**Authors**: Yichi Zhang, Yao Huang, Yifan Wang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15370)  

**Abstract**: The trustworthiness of Multimodal Large Language Models (MLLMs) remains an intense concern despite the significant progress in their capabilities. Existing evaluation and mitigation approaches often focus on narrow aspects and overlook risks introduced by the multimodality. To tackle these challenges, we propose MultiTrust-X, a comprehensive benchmark for evaluating, analyzing, and mitigating the trustworthiness issues of MLLMs. We define a three-dimensional framework, encompassing five trustworthiness aspects which include truthfulness, robustness, safety, fairness, and privacy; two novel risk types covering multimodal risks and cross-modal impacts; and various mitigation strategies from the perspectives of data, model architecture, training, and inference algorithms. Based on the taxonomy, MultiTrust-X includes 32 tasks and 28 curated datasets, enabling holistic evaluations over 30 open-source and proprietary MLLMs and in-depth analysis with 8 representative mitigation methods. Our extensive experiments reveal significant vulnerabilities in current models, including a gap between trustworthiness and general capabilities, as well as the amplification of potential risks in base LLMs by both multimodal training and inference. Moreover, our controlled analysis uncovers key limitations in existing mitigation strategies that, while some methods yield improvements in specific aspects, few effectively address overall trustworthiness, and many introduce unexpected trade-offs that compromise model utility. These findings also provide practical insights for future improvements, such as the benefits of reasoning to better balance safety and performance. Based on these insights, we introduce a Reasoning-Enhanced Safety Alignment (RESA) approach that equips the model with chain-of-thought reasoning ability to discover the underlying risks, achieving state-of-the-art results. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）的可靠性依然是一个严峻的关注焦点，尽管其能力已经取得了显著进步。现有的评估和缓解方法往往关注狭窄方面，忽视了多模态带来的一些风险。为应对这些挑战，我们提出MultiTrust-X，一个全面的测评基准，用于评估、分析和缓解MLLMs的可靠性问题。我们定义了一个三维框架，涵盖了真实性、稳健性、安全性、公平性和隐私等五大可靠性方面；涵盖多模态风险和跨模态影响的两种新型风险类型；以及从数据、模型结构、训练和推理算法等多个视角的缓解策略。基于这一分类，MultiTrust-X 包含32个任务和28个精选数据集，能够在30个开源和专有MLLM上进行全面评估，并使用8种代表性缓解方法进行深入分析。广泛的实验证明了当前模型存在显著的脆弱性，包括可靠性与通用能力之间的差距，以及通过多模态训练和推理放大基LLM的潜在风险。此外，我们通过受控分析揭示了现有缓解策略的关键局限性——尽管某些方法能在特定方面取得改进，但很少有方法能有效解决整体可靠性问题，且许多方法引入了意想不到的权衡，损害了模型的实用性。这些发现也为未来改进提供了实用见解，如推理在更好地平衡安全性和性能上的优势。在此基础上，我们提出了增强推理的安全对齐（RESA）方法，使模型具备链式推理能力以发现潜在风险，达到了现有最佳效果。 

---
# Predicting Road Crossing Behaviour using Pose Detection and Sequence Modelling 

**Title (ZH)**: 基于姿态检测和序列建模的过马路行为预测 

**Authors**: Subhasis Dasgupta, Preetam Saha, Agniva Roy, Jaydip Sen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15336)  

**Abstract**: The world is constantly moving towards AI based systems and autonomous vehicles are now reality in different parts of the world. These vehicles require sensors and cameras to detect objects and maneuver according to that. It becomes important to for such vehicles to also predict from a distant if a person is about to cross a road or not. The current study focused on predicting the intent of crossing the road by pedestrians in an experimental setup. The study involved working with deep learning models to predict poses and sequence modelling for temporal predictions. The study analysed three different sequence modelling to understand the prediction behaviour and it was found out that GRU was better in predicting the intent compared to LSTM model but 1D CNN was the best model in terms of speed. The study involved video analysis, and the output of pose detection model was integrated later on to sequence modelling techniques for an end-to-end deep learning framework for predicting road crossing intents. 

**Abstract (ZH)**: 基于AI的系统的世界正不断进步，自动驾驶车辆现在在世界各地已成为现实。这些车辆需要传感器和摄像头来检测物体并据此进行操作。此类车辆还应在远处预测行人是否即将过马路变得十分重要。本研究专注于在实验设置中预测行人过马路的意图。研究涉及使用深度学习模型预测姿势及使用序列建模进行时间预测。研究分析了三种不同的序列建模以理解预测行为，发现GRU在预测意图方面优于LSTM模型，而1D CNN在速度方面效果最好。研究涉及视频分析，后续将姿态检测模型的输出集成到序列建模技术中，以构建一个端到端的深度学习框架来预测过马路的意图。 

---
# VideoEraser: Concept Erasure in Text-to-Video Diffusion Models 

**Title (ZH)**: VideoEraser: 文本到视频扩散模型中的概念擦除 

**Authors**: Naen Xu, Jinghuai Zhang, Changjiang Li, Zhi Chen, Chunyi Zhou, Qingming Li, Tianyu Du, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.15314)  

**Abstract**: The rapid growth of text-to-video (T2V) diffusion models has raised concerns about privacy, copyright, and safety due to their potential misuse in generating harmful or misleading content. These models are often trained on numerous datasets, including unauthorized personal identities, artistic creations, and harmful materials, which can lead to uncontrolled production and distribution of such content. To address this, we propose VideoEraser, a training-free framework that prevents T2V diffusion models from generating videos with undesirable concepts, even when explicitly prompted with those concepts. Designed as a plug-and-play module, VideoEraser can seamlessly integrate with representative T2V diffusion models via a two-stage process: Selective Prompt Embedding Adjustment (SPEA) and Adversarial-Resilient Noise Guidance (ARNG). We conduct extensive evaluations across four tasks, including object erasure, artistic style erasure, celebrity erasure, and explicit content erasure. Experimental results show that VideoEraser consistently outperforms prior methods regarding efficacy, integrity, fidelity, robustness, and generalizability. Notably, VideoEraser achieves state-of-the-art performance in suppressing undesirable content during T2V generation, reducing it by 46% on average across four tasks compared to baselines. 

**Abstract (ZH)**: 文本到视频扩散模型的快速增长引发了对隐私、版权和安全的担忧，因为这些模型有可能被滥用以生成有害或误导性的内容。为应对这一问题，我们提出了一种无需训练的框架VideoEraser，该框架可防止T2V扩散模型在即使明确提示这些概念的情况下生成包含不良概念的视频。VideoEraser设计为即插即用模块，可以通过两阶段过程——选择性提示嵌入调整（SPEA）和抗对抗扰动噪声引导（ARNG）——无缝集成到代表性的T2V扩散模型中。我们在四个任务（包括物体擦除、艺术风格擦除、名人生涯擦除和明确内容擦除）上进行了广泛的评估。实验结果表明，VideoEraser在有效性、完整性和一致性、鲁棒性和泛化能力方面均优于先前的方法。值得注意的是，VideoEraser在T2V生成过程中抑制不良内容方面达到了最先进的性能，相比基线方法，平均减少46%。 

---
# First RAG, Second SEG: A Training-Free Paradigm for Camouflaged Object Detection 

**Title (ZH)**: First RAG, Second SEG: 一种无需训练的迷彩目标检测 paradigm 

**Authors**: Wutao Liu, YiDan Wang, Pan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15313)  

**Abstract**: Camouflaged object detection (COD) poses a significant challenge in computer vision due to the high similarity between objects and their backgrounds. Existing approaches often rely on heavy training and large computational resources. While foundation models such as the Segment Anything Model (SAM) offer strong generalization, they still struggle to handle COD tasks without fine-tuning and require high-quality prompts to yield good performance. However, generating such prompts manually is costly and inefficient. To address these challenges, we propose \textbf{First RAG, Second SEG (RAG-SEG)}, a training-free paradigm that decouples COD into two stages: Retrieval-Augmented Generation (RAG) for generating coarse masks as prompts, followed by SAM-based segmentation (SEG) for refinement. RAG-SEG constructs a compact retrieval database via unsupervised clustering, enabling fast and effective feature retrieval. During inference, the retrieved features produce pseudo-labels that guide precise mask generation using SAM2. Our method eliminates the need for conventional training while maintaining competitive performance. Extensive experiments on benchmark COD datasets demonstrate that RAG-SEG performs on par with or surpasses state-of-the-art methods. Notably, all experiments are conducted on a \textbf{personal laptop}, highlighting the computational efficiency and practicality of our approach. We present further analysis in the Appendix, covering limitations, salient object detection extension, and possible improvements. 

**Abstract (ZH)**: 伪装物体检测 (COD) 由于物体与其背景高度相似，在计算机视觉中构成了显著的挑战。现有的方法往往依赖于大量训练和高性能计算资源。虽然基础模型如 Segment Anything Model (SAM) 能提供强大的泛化能力，但在处理 COD 任务时仍需微调，并且需要高质量的提示以获得良好的性能。然而，手动生成这样的提示成本高且效率低。为了解决这些挑战，我们提出了一种名为 \textbf{First RAG, Second SEG (RAG-SEG)} 的无需训练的范式，将 COD 分解为两个阶段：检索增强生成 (RAG) 用于生成粗略掩码作为提示，随后是基于 SAM 的分割 (SEG) 用于细化。RAG-SEG 通过无监督聚类构建紧凑的检索数据库，实现快速且有效的特征检索。在推理过程中，检索的特征生成伪标签，指导使用 SAM2 进行精确的掩码生成。该方法消除了传统训练的需求，同时保持了竞争性性能。大量基准 COD 数据集的实验表明，RAG-SEG 在性能上与当前最先进的方法相当或超过。值得注意的是，所有实验均在一台 \textbf{个人笔记本电脑} 上进行，突显了该方法的计算效率和实用性。我们在附录中进行了进一步分析，涵盖限制、显著物体检测扩展和可能的改进。 

---
# IPIGuard: A Novel Tool Dependency Graph-Based Defense Against Indirect Prompt Injection in LLM Agents 

**Title (ZH)**: IPIGuard：一种新型工具依赖图基的防护方法，对抗LLM代理中的间接提示注入攻击 

**Authors**: Hengyu An, Jinghuai Zhang, Tianyu Du, Chunyi Zhou, Qingming Li, Tao Lin, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.15310)  

**Abstract**: Large language model (LLM) agents are widely deployed in real-world applications, where they leverage tools to retrieve and manipulate external data for complex tasks. However, when interacting with untrusted data sources (e.g., fetching information from public websites), tool responses may contain injected instructions that covertly influence agent behaviors and lead to malicious outcomes, a threat referred to as Indirect Prompt Injection (IPI). Existing defenses typically rely on advanced prompting strategies or auxiliary detection models. While these methods have demonstrated some effectiveness, they fundamentally rely on assumptions about the model's inherent security, which lacks structural constraints on agent behaviors. As a result, agents still retain unrestricted access to tool invocations, leaving them vulnerable to stronger attack vectors that can bypass the security guardrails of the model. To prevent malicious tool invocations at the source, we propose a novel defensive task execution paradigm, called IPIGuard, which models the agents' task execution process as a traversal over a planned Tool Dependency Graph (TDG). By explicitly decoupling action planning from interaction with external data, IPIGuard significantly reduces unintended tool invocations triggered by injected instructions, thereby enhancing robustness against IPI attacks. Experiments on the AgentDojo benchmark show that IPIGuard achieves a superior balance between effectiveness and robustness, paving the way for the development of safer agentic systems in dynamic environments. 

**Abstract (ZH)**: IPIGuard：基于工具依赖图的任务执行防御范式以防范间接提示注入攻击 

---
# DesignCLIP: Multimodal Learning with CLIP for Design Patent Understanding 

**Title (ZH)**: DesignCLIP: 基于CLIP的多模态学习设计专利理解 

**Authors**: Zhu Wang, Homaira Huda Shomee, Sathya N. Ravi, Sourav Medya  

**Link**: [PDF](https://arxiv.org/pdf/2508.15297)  

**Abstract**: In the field of design patent analysis, traditional tasks such as patent classification and patent image retrieval heavily depend on the image data. However, patent images -- typically consisting of sketches with abstract and structural elements of an invention -- often fall short in conveying comprehensive visual context and semantic information. This inadequacy can lead to ambiguities in evaluation during prior art searches. Recent advancements in vision-language models, such as CLIP, offer promising opportunities for more reliable and accurate AI-driven patent analysis. In this work, we leverage CLIP models to develop a unified framework DesignCLIP for design patent applications with a large-scale dataset of U.S. design patents. To address the unique characteristics of patent data, DesignCLIP incorporates class-aware classification and contrastive learning, utilizing generated detailed captions for patent images and multi-views image learning. We validate the effectiveness of DesignCLIP across various downstream tasks, including patent classification and patent retrieval. Additionally, we explore multimodal patent retrieval, which provides the potential to enhance creativity and innovation in design by offering more diverse sources of inspiration. Our experiments show that DesignCLIP consistently outperforms baseline and SOTA models in the patent domain on all tasks. Our findings underscore the promise of multimodal approaches in advancing patent analysis. The codebase is available here: this https URL. 

**Abstract (ZH)**: 设计专利分析中的设计CLIP框架：基于大规模美国设计专利数据的统一方法 

---
# Way to Build Native AI-driven 6G Air Interface: Principles, Roadmap, and Outlook 

**Title (ZH)**: 基于原生AI驱动的6G空中接口构建方法：原理、路线图与展望 

**Authors**: Ping Zhang, Kai Niu, Yiming Liu, Zijian Liang, Nan Ma, Xiaodong Xu, Wenjun Xu, Mengying Sun, Yinqiu Liu, Xiaoyun Wang, Ruichen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15277)  

**Abstract**: Artificial intelligence (AI) is expected to serve as a foundational capability across the entire lifecycle of 6G networks, spanning design, deployment, and operation. This article proposes a native AI-driven air interface architecture built around two core characteristics: compression and adaptation. On one hand, compression enables the system to understand and extract essential semantic information from the source data, focusing on task relevance rather than symbol-level accuracy. On the other hand, adaptation allows the air interface to dynamically transmit semantic information across diverse tasks, data types, and channel conditions, ensuring scalability and robustness. This article first introduces the native AI-driven air interface architecture, then discusses representative enabling methodologies, followed by a case study on semantic communication in 6G non-terrestrial networks. Finally, it presents a forward-looking discussion on the future of native AI in 6G, outlining key challenges and research opportunities. 

**Abstract (ZH)**: 人工智能驱动的6G网络原生空中接口架构：基于压缩与适应的核心特性 

---
# M-$LLM^3$REC: A Motivation-Aware User-Item Interaction Framework for Enhancing Recommendation Accuracy with LLMs 

**Title (ZH)**: M-$LLM^3$REC：一种基于动机感知的用户-物品交互框架，用于通过LLMs提升推荐准确性 

**Authors**: Lining Chen, Qingwen Zeng, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15262)  

**Abstract**: Recommendation systems have been essential for both user experience and platform efficiency by alleviating information overload and supporting decision-making. Traditional methods, i.e., content-based filtering, collaborative filtering, and deep learning, have achieved impressive results in recommendation systems. However, the cold-start and sparse-data scenarios are still challenging to deal with. Existing solutions either generate pseudo-interaction sequence, which often introduces redundant or noisy signals, or rely heavily on semantic similarity, overlooking dynamic shifts in user motivation. To address these limitations, this paper proposes a novel recommendation framework, termed M-$LLM^3$REC, which leverages large language models for deep motivational signal extraction from limited user interactions. M-$LLM^3$REC comprises three integrated modules: the Motivation-Oriented Profile Extractor (MOPE), Motivation-Oriented Trait Encoder (MOTE), and Motivational Alignment Recommender (MAR). By emphasizing motivation-driven semantic modeling, M-$LLM^3$REC demonstrates robust, personalized, and generalizable recommendations, particularly boosting performance in cold-start situations in comparison with the state-of-the-art frameworks. 

**Abstract (ZH)**: 推荐系统通过减轻信息过载和支持决策制定，对于提升用户体验和平台效率至关重要。传统的基于内容过滤、协同过滤和深度学习的方法已经在推荐系统中取得了令人印象深刻的成果。然而，在冷启动和稀疏数据场景下仍存在挑战。现有的解决方案要么生成伪交互序列，这通常引入冗余或噪声信号，要么过度依赖语义相似性，忽视了用户动机的动态变化。为解决这些局限性，本文提出了一种新的推荐框架，称为M-$LLM^3$REC，该框架利用大型语言模型从有限的用户交互中提取深层次的动力信号。M-$LLM^3$REC 包含三个集成模块：动机导向资料提取器（MOPE）、动机导向特征编码器（MOTE）以及动力对齐推荐器（MAR）。通过强调基于动机的语义建模，M-$LLM^3$REC 展示了稳健、个性化和泛化的推荐能力，特别是在冷启动情况下，相对于现有最先进的框架显著提升了性能。 

---
# Conflict-Aware Soft Prompting for Retrieval-Augmented Generation 

**Title (ZH)**: 冲突意识软提示增强检索生成 

**Authors**: Eunseong Choi, June Park, Hyeri Lee, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.15253)  

**Abstract**: Retrieval-augmented generation (RAG) enhances the capabilities of large language models (LLMs) by incorporating external knowledge into their input prompts. However, when the retrieved context contradicts the LLM's parametric knowledge, it often fails to resolve the conflict between incorrect external context and correct parametric knowledge, known as context-memory conflict. To tackle this problem, we introduce Conflict-Aware REtrieval-Augmented Generation (CARE), consisting of a context assessor and a base LLM. The context assessor encodes compact memory token embeddings from raw context tokens. Through grounded/adversarial soft prompting, the context assessor is trained to discern unreliable context and capture a guidance signal that directs reasoning toward the more reliable knowledge source. Extensive experiments show that CARE effectively mitigates context-memory conflicts, leading to an average performance gain of 5.0\% on QA and fact-checking benchmarks, establishing a promising direction for trustworthy and adaptive RAG systems. 

**Abstract (ZH)**: 冲突aware检索增强生成（CARE）：缓解上下文记忆冲突的研究 

---
# Explainable Knowledge Distillation for Efficient Medical Image Classification 

**Title (ZH)**: 可解释的知识蒸馏在高效医学图像分类中的应用 

**Authors**: Aqib Nazir Mir, Danish Raza Rizvi  

**Link**: [PDF](https://arxiv.org/pdf/2508.15251)  

**Abstract**: This study comprehensively explores knowledge distillation frameworks for COVID-19 and lung cancer classification using chest X-ray (CXR) images. We employ high-capacity teacher models, including VGG19 and lightweight Vision Transformers (Visformer-S and AutoFormer-V2-T), to guide the training of a compact, hardware-aware student model derived from the OFA-595 supernet. Our approach leverages hybrid supervision, combining ground-truth labels with teacher models' soft targets to balance accuracy and computational efficiency. We validate our models on two benchmark datasets: COVID-QU-Ex and LCS25000, covering multiple classes, including COVID-19, healthy, non-COVID pneumonia, lung, and colon cancer. To interpret the spatial focus of the models, we employ Score-CAM-based visualizations, which provide insight into the reasoning process of both teacher and student networks. The results demonstrate that the distilled student model maintains high classification performance with significantly reduced parameters and inference time, making it an optimal choice in resource-constrained clinical environments. Our work underscores the importance of combining model efficiency with explainability for practical, trustworthy medical AI solutions. 

**Abstract (ZH)**: 本研究全面探讨了基于胸部X光图像（CXR）的COVID-19和肺癌分类的知识蒸馏框架。我们采用高容量的教师模型，包括VGG19和轻量级的Vision Transformers（Visformer-S和AutoFormer-V2-T），以指导来自OFA-595超网络的紧凑型硬件感知学生模型的训练。我们的方法利用了混合监督，结合真实标签和教师模型的软目标，以平衡准确性和计算效率。我们使用两个基准数据集COVID-QU-Ex和LCS25000进行模型验证，涵盖COVID-19、健康、非COVID肺炎、肺和结肠癌等多个类别。为了解释模型的空间关注点，我们使用Score-CAM基可视化方法，提供对教师和学生网络推理过程的见解。研究结果表明，蒸馏后学生模型在显著减少参数和推理时间的同时，保持了高分类性能，使其成为资源受限临床环境中的最佳选择。我们的工作强调了将模型效率与可解释性相结合对于实际可靠的医疗AI解决方案的重要性。 

---
# Robust and Efficient Quantum Reservoir Computing with Discrete Time Crystal 

**Title (ZH)**: 离散时间晶体中稳健而高效的量子蓄水库计算 

**Authors**: Da Zhang, Xin Li, Yibin Guo, Haifeng Yu, Yirong Jin, Zhang-Qi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.15230)  

**Abstract**: The rapid development of machine learning and quantum computing has placed quantum machine learning at the forefront of research. However, existing quantum machine learning algorithms based on quantum variational algorithms face challenges in trainability and noise robustness. In order to address these challenges, we introduce a gradient-free, noise-robust quantum reservoir computing algorithm that harnesses discrete time crystal dynamics as a reservoir. We first calibrate the memory, nonlinear, and information scrambling capacities of the quantum reservoir, revealing their correlation with dynamical phases and non-equilibrium phase transitions. We then apply the algorithm to the binary classification task and establish a comparative quantum kernel advantage. For ten-class classification, both noisy simulations and experimental results on superconducting quantum processors match ideal simulations, demonstrating the enhanced accuracy with increasing system size and confirming the topological noise robustness. Our work presents the first experimental demonstration of quantum reservoir computing for image classification based on digital quantum simulation. It establishes the correlation between quantum many-body non-equilibrium phase transitions and quantum machine learning performance, providing new design principles for quantum reservoir computing and broader quantum machine learning algorithms in the NISQ era. 

**Abstract (ZH)**: 机器学习和量子计算的快速发展将量子机器学习推向了研究前沿。然而，现有的基于量子变分算法的量子机器学习算法在可训练性和抗噪性方面面临挑战。为了应对这些挑战，我们引入了一种采用离散时间晶体动力学作为蓄水池的无梯度、抗噪量子蓄水池计算算法。我们首先校准了量子蓄水池的记忆、非线性和信息混杂能力，并揭示了这些能力与动力学相位和非平衡相变之间的关系。然后，我们将该算法应用于二元分类任务，并建立了量子核优势。对于十类分类任务，嘈杂的模拟和超导量子处理器上的实验结果均与理想模拟匹配，展示了系统规模增大时的增强精度，并证实了拓扑抗噪性。我们的工作首次基于数字量子模拟实现了基于量子蓄水池计算的图像分类的实验演示，建立了量子多体非平衡相变与量子机器学习性能之间的关联，为量子蓄水池计算和更广泛的量子机器学习算法在NISQ时代的设计提供了新的原理。 

---
# VocabTailor: Dynamic Vocabulary Selection for Downstream Tasks in Small Language Models 

**Title (ZH)**: VocabTailor: 小规模语言模型中下游任务的动态词汇选择 

**Authors**: Hanling Zhang, Yayu Zhou, Tongcheng Fang, Zhihang Yuan, Guohao Dai, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15229)  

**Abstract**: Small Language Models (SLMs) provide computational advantages in resource-constrained environments, yet memory limitations remain a critical bottleneck for edge device deployment. A substantial portion of SLMs' memory footprint stems from vocabulary-related components, particularly embeddings and language modeling (LM) heads, due to large vocabulary sizes. Existing static vocabulary pruning, while reducing memory usage, suffers from rigid, one-size-fits-all designs that cause information loss from the prefill stage and a lack of flexibility. In this work, we identify two key principles underlying the vocabulary reduction challenge: the lexical locality principle, the observation that only a small subset of tokens is required during any single inference, and the asymmetry in computational characteristics between vocabulary-related components of SLM. Based on these insights, we introduce VocabTailor, a novel decoupled dynamic vocabulary selection framework that addresses memory constraints through offloading embedding and implements a hybrid static-dynamic vocabulary selection strategy for LM Head, enabling on-demand loading of vocabulary components. Comprehensive experiments across diverse downstream tasks demonstrate that VocabTailor achieves a reduction of up to 99% in the memory usage of vocabulary-related components with minimal or no degradation in task performance, substantially outperforming existing static vocabulary pruning. 

**Abstract (ZH)**: 小语言模型通过在资源受限环境中提供计算优势，为边缘设备部署带来了潜在的好处，然而内存限制仍然是一个关键瓶颈。小语言模型中的内存占用大量来自与词汇相关的组件，特别是嵌入和语言模型头，由于词汇表规模庞大。现有的静态词汇剪枝虽然减少了内存使用，但其僵化的、一刀切的设计导致了预填充阶段的信息损失以及缺乏灵活性。在本文中，我们识别出词汇缩减面临的两个关键原则：词汇局部性原则，即在任何单一推理过程中只需要一小部分标记；以及词汇相关组件的计算特性之间的不对称性。基于这些见解，我们引入了VocabTailor，这是一种新颖的解耦动态词汇选择框架，通过卸载嵌入来解决内存限制问题，并对LM头实施混合静态-动态词汇选择策略，实现按需加载词汇组件。在涉及多样下游任务的综合实验中，VocabTailor实现了词汇相关组件内存使用最多99%的减少，同时任务性能略有或无退步，在现有静态词汇剪枝方法上具有显著优势。 

---
# GenTune: Toward Traceable Prompts to Improve Controllability of Image Refinement in Environment Design 

**Title (ZH)**: GenTune: 向可追溯提示的方向改进环境设计中图像细化的可控性 

**Authors**: Wen-Fan Wang, Ting-Ying Lee, Chien-Ting Lu, Che-Wei Hsu, Nil Ponsa Campany, Yu Chen, Mike Y. Chen, Bing-Yu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15227)  

**Abstract**: Environment designers in the entertainment industry create imaginative 2D and 3D scenes for games, films, and television, requiring both fine-grained control of specific details and consistent global coherence. Designers have increasingly integrated generative AI into their workflows, often relying on large language models (LLMs) to expand user prompts for text-to-image generation, then iteratively refining those prompts and applying inpainting. However, our formative study with 10 designers surfaced two key challenges: (1) the lengthy LLM-generated prompts make it difficult to understand and isolate the keywords that must be revised for specific visual elements; and (2) while inpainting supports localized edits, it can struggle with global consistency and correctness. Based on these insights, we present GenTune, an approach that enhances human--AI collaboration by clarifying how AI-generated prompts map to image content. Our GenTune system lets designers select any element in a generated image, trace it back to the corresponding prompt labels, and revise those labels to guide precise yet globally consistent image refinement. In a summative study with 20 designers, GenTune significantly improved prompt--image comprehension, refinement quality, and efficiency, and overall satisfaction (all $p < .01$) compared to current practice. A follow-up field study with two studios further demonstrated its effectiveness in real-world settings. 

**Abstract (ZH)**: 环境设计师在娱乐行业中创建游戏、电影和电视的想象中的2D和3D场景，需要对详细细节进行精细控制并保持全局的一致性。设计师们越来越多地将生成式AI集成到他们的工作流程中，通常依赖大规模语言模型（LLMs）扩展文本到图像生成的用户提示，然后迭代地精炼这些提示并应用修补技术。然而，我们针对10名设计师的形成性研究揭示了两个关键挑战：（1）LLM生成的提示过于冗长，使得理解和隔离需要修订的具体视觉元素的关键词变得困难；（2）尽管修补技术支持局部编辑，但它在全局一致性和准确性方面存在问题。基于这些见解，我们提出了一种名为GenTune的方法，通过阐明AI生成的提示如何映射到图像内容来增强人-机协作。GenTune系统使设计师能够选择生成图像中的任何元素，追溯回相应的提示标签，并修改这些标签以指导精确且全局一致的图像精修。在20名设计师的总结性研究中，GenTune显著提高了提示-图像理解、精修质量、效率及总体满意度（所有$p < .01$），相较于现有做法。后续的实地研究进一步证明了其在实际场景中的有效性。 

---
# Locally Pareto-Optimal Interpretations for Black-Box Machine Learning Models 

**Title (ZH)**: 局部帕累托最优解释：黑盒机器学习模型的解释方法 

**Authors**: Aniruddha Joshi, Supratik Chakraborty, S Akshay, Shetal Shah, Hazem Torfah, Sanjit Seshia  

**Link**: [PDF](https://arxiv.org/pdf/2508.15220)  

**Abstract**: Creating meaningful interpretations for black-box machine learning models involves balancing two often conflicting objectives: accuracy and explainability. Exploring the trade-off between these objectives is essential for developing trustworthy interpretations. While many techniques for multi-objective interpretation synthesis have been developed, they typically lack formal guarantees on the Pareto-optimality of the results. Methods that do provide such guarantees, on the other hand, often face severe scalability limitations when exploring the Pareto-optimal space. To address this, we develop a framework based on local optimality guarantees that enables more scalable synthesis of interpretations. Specifically, we consider the problem of synthesizing a set of Pareto-optimal interpretations with local optimality guarantees, within the immediate neighborhood of each solution. Our approach begins with a multi-objective learning or search technique, such as Multi-Objective Monte Carlo Tree Search, to generate a best-effort set of Pareto-optimal candidates with respect to accuracy and explainability. We then verify local optimality for each candidate as a Boolean satisfiability problem, which we solve using a SAT solver. We demonstrate the efficacy of our approach on a set of benchmarks, comparing it against previous methods for exploring the Pareto-optimal front of interpretations. In particular, we show that our approach yields interpretations that closely match those synthesized by methods offering global guarantees. 

**Abstract (ZH)**: 创建黑箱机器学习模型的有意义解释涉及平衡准确性和可解释性这两个 Often 相互冲突的目标。探索这两个目标之间的trade-off 是开发可信赖解释的关键。尽管已经开发出许多多目标解释综合技术，但它们通常缺乏关于结果 Pareto-最优点的正式保证。另一方面，能够提供此类保证的方法在探索 Pareto-最优点空间时往往会面临严重的可扩展性限制。为此，我们开发了一个基于局部最优点保证的框架，以实现更可扩展的解释综合。具体地，我们考虑在每个解的局部邻域内合成具有局部最优点保证的 Pareto-最优解释集的问题。我们的方法首先使用多目标学习或搜索技术，例如多目标蒙特卡洛树搜索，生成关于准确性和可解释性的 Pareto-最优候选集。然后，我们通过布尔可满足性问题验证每个候选的局部最优点，并使用 SAT 求解器求解该问题。我们在一组基准上展示了我们方法的有效性，将其与先前探索解释 Pareto-前沿的方法进行了比较。特别地，我们展示了我们的方法产生的解释与提供全局保证的方法合成的解释高度一致。 

---
# SparK: Query-Aware Unstructured Sparsity with Recoverable KV Cache Channel Pruning 

**Title (ZH)**: SparK: 查询感知的可恢复KV缓存通道无结构稀疏性剪枝 

**Authors**: Huanxuan Liao, Yixing Xu, Shizhu He, Guanchen Li, Xuanwu Yin, Dong Li, Emad Barsoum, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15212)  

**Abstract**: Long-context inference in large language models (LLMs) is increasingly constrained by the KV cache bottleneck: memory usage grows linearly with sequence length, while attention computation scales quadratically. Existing approaches address this issue by compressing the KV cache along the temporal axis through strategies such as token eviction or merging to reduce memory and computational overhead. However, these methods often neglect fine-grained importance variations across feature dimensions (i.e., the channel axis), thereby limiting their ability to effectively balance efficiency and model accuracy. In reality, we observe that channel saliency varies dramatically across both queries and positions: certain feature channels carry near-zero information for a given query, while others spike in relevance. To address this oversight, we propose SPARK, a training-free plug-and-play method that applies unstructured sparsity by pruning KV at the channel level, while dynamically restoring the pruned entries during attention score computation. Notably, our approach is orthogonal to existing KV compression and quantization techniques, making it compatible for integration with them to achieve further acceleration. By reducing channel-level redundancy, SPARK enables processing of longer sequences within the same memory budget. For sequences of equal length, SPARK not only preserves or improves model accuracy but also reduces KV cache storage by over 30% compared to eviction-based methods. Furthermore, even with an aggressive pruning ratio of 80%, SPARK maintains performance with less degradation than 5% compared to the baseline eviction method, demonstrating its robustness and effectiveness. Our code will be available at this https URL. 

**Abstract (ZH)**: 长上下文推理中大型语言模型的内存瓶颈：通过时间轴上的密钥值缓存压缩策略来缓解KV缓存瓶颈，但忽略特征维度上的细粒度重要性变化 

---
# Survey of Vision-Language-Action Models for Embodied Manipulation 

**Title (ZH)**: 视觉-语言-行动模型综述：赋能实体操纵 

**Authors**: Haoran Li, Yuhui Chen, Wenbo Cui, Weiheng Liu, Kai Liu, Mingcai Zhou, Zhengtao Zhang, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15201)  

**Abstract**: Embodied intelligence systems, which enhance agent capabilities through continuous environment interactions, have garnered significant attention from both academia and industry. Vision-Language-Action models, inspired by advancements in large foundation models, serve as universal robotic control frameworks that substantially improve agent-environment interaction capabilities in embodied intelligence systems. This expansion has broadened application scenarios for embodied AI robots. This survey comprehensively reviews VLA models for embodied manipulation. Firstly, it chronicles the developmental trajectory of VLA architectures. Subsequently, we conduct a detailed analysis of current research across 5 critical dimensions: VLA model structures, training datasets, pre-training methods, post-training methods, and model evaluation. Finally, we synthesize key challenges in VLA development and real-world deployment, while outlining promising future research directions. 

**Abstract (ZH)**: 具身智能系统中的视觉-语言-动作模型：全面回顾及其关键挑战与未来研究方向 

---
# SemToken: Semantic-Aware Tokenization for Efficient Long-Context Language Modeling 

**Title (ZH)**: SemToken：面向高效长上下文语言建模的语义感知分词方法 

**Authors**: Dong Liu, Yanxuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15190)  

**Abstract**: Tokenization plays a critical role in language modeling, yet existing approaches such as Byte-Pair Encoding (BPE) or WordPiece operate purely on frequency statistics, ignoring the underlying semantic structure of text. This leads to over-tokenization of semantically redundant spans and underutilization of contextual coherence, particularly in long-context scenarios. In this work, we propose \textbf{SemToken}, a semantic-aware tokenization framework that jointly reduces token redundancy and improves computation efficiency. SemToken first extracts contextual semantic embeddings via lightweight encoders and performs local semantic clustering to merge semantically equivalent tokens. Then, it allocates heterogeneous token granularity based on semantic density, allowing finer-grained tokenization in content-rich regions and coarser compression in repetitive or low-entropy spans. SemToken can be seamlessly integrated with modern language models and attention acceleration methods. Experiments on long-context language modeling benchmarks such as WikiText-103 and LongBench show that SemToken achieves up to $2.4\times$ reduction in token count and $1.9\times$ speedup, with negligible or no degradation in perplexity and downstream accuracy. Our findings suggest that semantic structure offers a promising new axis for optimizing tokenization and computation in large language models. 

**Abstract (ZH)**: 基于语义的分词框架：SemToken及其在语言模型中的应用 

---
# SurgWound-Bench: A Benchmark for Surgical Wound Diagnosis 

**Title (ZH)**: SurgWound-Bench: 一项外科伤口诊断基准 

**Authors**: Jiahao Xu, Changchang Yin, Odysseas Chatzipanagiotou, Diamantis Tsilimigras, Kevin Clear, Bingsheng Yao, Dakuo Wang, Timothy Pawlik, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15189)  

**Abstract**: Surgical site infection (SSI) is one of the most common and costly healthcare-associated infections and and surgical wound care remains a significant clinical challenge in preventing SSIs and improving patient outcomes. While recent studies have explored the use of deep learning for preliminary surgical wound screening, progress has been hindered by concerns over data privacy and the high costs associated with expert annotation. Currently, no publicly available dataset or benchmark encompasses various types of surgical wounds, resulting in the absence of an open-source Surgical-Wound screening tool. To address this gap: (1) we present SurgWound, the first open-source dataset featuring a diverse array of surgical wound types. It contains 697 surgical wound images annotated by 3 professional surgeons with eight fine-grained clinical attributes. (2) Based on SurgWound, we introduce the first benchmark for surgical wound diagnosis, which includes visual question answering (VQA) and report generation tasks to comprehensively evaluate model performance. (3) Furthermore, we propose a three-stage learning framework, WoundQwen, for surgical wound diagnosis. In the first stage, we employ five independent MLLMs to accurately predict specific surgical wound characteristics. In the second stage, these predictions serve as additional knowledge inputs to two MLLMs responsible for diagnosing outcomes, which assess infection risk and guide subsequent interventions. In the third stage, we train a MLLM that integrates the diagnostic results from the previous two stages to produce a comprehensive report. This three-stage framework can analyze detailed surgical wound characteristics and provide subsequent instructions to patients based on surgical images, paving the way for personalized wound care, timely intervention, and improved patient outcomes. 

**Abstract (ZH)**: 手术切口感染（SSI）是最常见的医院关联感染之一，手术伤口护理依然是预防SSI和改善患者预后的重大临床挑战。尽管近期研究探索了深度学习在初步手术伤口筛查中的应用，但数据隐私问题和专家标注的高成本限制了进展。目前，尚无包含多种手术伤口类型的公开数据集或基准，导致缺乏开源的手术伤口筛查工具。为解决这一缺口：（1）我们呈现了SurgWound，这是首个包含多种手术伤口类型的开源数据集，包含697张由三位专业外科医生标注的手术伤口图像，附带八项精细临床属性。（2）基于SurgWound，我们引入了首个手术伤口诊断基准，包含视觉问答（VQA）和报告生成任务，以全面评估模型性能。（3）此外，我们提出了一种三阶段学习框架WoundQwen，用于手术伤口诊断。在第一阶段，我们使用五个独立的MLLM预测特定的手术伤口特征；在第二阶段，这些预测作为额外知识输入，用于两个MLLM进行诊断结果评估，以评估感染风险并指导后续干预；在第三阶段，我们训练一个MLLM综合前两阶段的诊断结果生成全面报告。这一三阶段框架可以分析详细的手术伤口特征，并基于手术图像为患者提供后续指导，为个性化伤口护理、及时干预和改善患者预后铺平道路。 

---
# Universal Reinforcement Learning in Coalgebras: Asynchronous Stochastic Computation via Conduction 

**Title (ZH)**: 代数中的通用强化学习：通过传导的异步随机计算 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2508.15128)  

**Abstract**: In this paper, we introduce a categorial generalization of RL, termed universal reinforcement learning (URL), building on powerful mathematical abstractions from the study of coinduction on non-well-founded sets and universal coalgebras, topos theory, and categorial models of asynchronous parallel distributed computation. In the first half of the paper, we review the basic RL framework, illustrate the use of categories and functors in RL, showing how they lead to interesting insights. In particular, we also introduce a standard model of asynchronous distributed minimization proposed by Bertsekas and Tsitsiklis, and describe the relationship between metric coinduction and their proof of the Asynchronous Convergence Theorem. The space of algorithms for MDPs or PSRs can be modeled as a functor category, where the co-domain category forms a topos, which admits all (co)limits, possesses a subobject classifier, and has exponential objects. In the second half of the paper, we move on to universal coalgebras. Dynamical system models, such as Markov decision processes (MDPs), partially observed MDPs (POMDPs), a predictive state representation (PSRs), and linear dynamical systems (LDSs) are all special types of coalgebras. We describe a broad family of universal coalgebras, extending the dynamic system models studied previously in RL. The core problem in finding fixed points in RL to determine the exact or approximate (action) value function is generalized in URL to determining the final coalgebra asynchronously in a parallel distributed manner. 

**Abstract (ZH)**: 用通用代数观点对强化学习进行的范畴论一般化：异步并行分布式计算中的动态系统模型与普朗克coalgebra 

---
# Enhanced Predictive Modeling for Hazardous Near-Earth Object Detection: A Comparative Analysis of Advanced Resampling Strategies and Machine Learning Algorithms in Planetary Risk Assessment 

**Title (ZH)**: 增强预测建模以检测危险近地天体：行星风险评估中高级重采样策略和机器学习算法的比较分析 

**Authors**: Sunkalp Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2508.15106)  

**Abstract**: This study evaluates the performance of several machine learning models for predicting hazardous near-Earth objects (NEOs) through a binary classification framework, including data scaling, power transformation, and cross-validation. Six classifiers were compared, namely Random Forest Classifier (RFC), Gradient Boosting Classifier (GBC), Support Vector Classifier (SVC), Linear Discriminant Analysis (LDA), Logistic Regression (LR), and K-Nearest Neighbors (KNN). RFC and GBC performed the best, both with an impressive F2-score of 0.987 and 0.986, respectively, with very small variability. SVC followed, with a lower but reasonable score of 0.896. LDA and LR had a moderate performance with scores of around 0.749 and 0.748, respectively, while KNN had a poor performance with a score of 0.691 due to difficulty in handling complex data patterns. RFC and GBC also presented great confusion matrices with a negligible number of false positives and false negatives, which resulted in outstanding accuracy rates of 99.7% and 99.6%, respectively. These findings highlight the power of ensemble methods for high precision and recall and further point out the importance of tailored model selection with regard to dataset characteristics and chosen evaluation metrics. Future research could focus on the optimization of hyperparameters with advanced features engineering to further the accuracy and robustness of the model on NEO hazard predictions. 

**Abstract (ZH)**: 本研究通过二元分类框架评估了几种机器学习模型在预测危险近地天体（NEOs）性能上的表现，包括数据缩放、功率转换和交叉验证。比较了六种分类器，分别是随机森林分类器（RFC）、梯度提升分类器（GBC）、支持向量分类器（SVC）、线性判别分析（LDA）、逻辑回归（LR）和K-最近邻（KNN）。RFC和GBC表现最佳，分别获得了令人印象深刻的F2分数0.987和0.986，且变异性极小。SVC次之，得分为0.896。LDA和LR表现中等，得分为约0.749和0.748，而KNN由于难以处理复杂的数据模式表现较差，得分为0.691。RFC和GBC还展示了几乎不存在假阳性和假阴性的混淆矩阵，分别获得了99.7%和99.6%的卓越准确率。这些发现突显了集成方法在高精度和召回率方面的能力，并进一步指出根据数据集特征和选定的评估指标进行定制化模型选择的重要性。未来的研究可以关注高级特征工程对超参数的优化，以进一步提高NEO危害预测模型的准确性和稳健性。 

---
# Equi-mRNA: Protein Translation Equivariant Encoding for mRNA Language Models 

**Title (ZH)**: Equi-mRNA：蛋白质翻译不变编码的mRNA语言模型 

**Authors**: Mehdi Yazdani-Jahromi, Ali Khodabandeh Yalabadi, Ozlem Ozmen Garibay  

**Link**: [PDF](https://arxiv.org/pdf/2508.15103)  

**Abstract**: The growing importance of mRNA therapeutics and synthetic biology highlights the need for models that capture the latent structure of synonymous codon (different triplets encoding the same amino acid) usage, which subtly modulates translation efficiency and gene expression. While recent efforts incorporate codon-level inductive biases through auxiliary objectives, they often fall short of explicitly modeling the structured relationships that arise from the genetic code's inherent symmetries. We introduce Equi-mRNA, the first codon-level equivariant mRNA language model that explicitly encodes synonymous codon symmetries as cyclic subgroups of 2D Special Orthogonal matrix (SO(2)). By combining group-theoretic priors with an auxiliary equivariance loss and symmetry-aware pooling, Equi-mRNA learns biologically grounded representations that outperform vanilla baselines across multiple axes. On downstream property-prediction tasks including expression, stability, and riboswitch switching Equi-mRNA delivers up to approximately 10% improvements in accuracy. In sequence generation, it produces mRNA constructs that are up to approximately 4x more realistic under Frechet BioDistance metrics and approximately 28% better preserve functional properties compared to vanilla baseline. Interpretability analyses further reveal that learned codon-rotation distributions recapitulate known GC-content biases and tRNA abundance patterns, offering novel insights into codon usage. Equi-mRNA establishes a new biologically principled paradigm for mRNA modeling, with significant implications for the design of next-generation therapeutics. 

**Abstract (ZH)**: mRNA治疗和合成生物学的 growing importance 强调了需要构建能够捕捉同义密码子使用潜在结构的模型，这些结构微妙地调节翻译效率和基因表达。尽管近期努力通过辅助目标引入密码子层级的归纳偏置，但它们往往未能明确建模源自遗传密码内在对称性的结构关系。我们引入了 Equi-mRNA，首个在密码子层级具有不变性的 mRNA 语言模型，显式地将同义密码子对称性编码为二维特殊正交矩阵（SO(2)）的循环子群。通过结合群论先验、辅助不变性损失和对称性感知池化，Equi-mRNA 学习到生物基础的表示，在多个维度上优于基础模型。在下游属性预测任务（包括表达、稳定性以及核糖开关切换）中，Equi-mRNA 的准确性可提升约 10%。在序列生成中，它生成的 mRNA 构建体在Frechet 生物距离度量下更具现实性，约 4 倍于基础模型，并且功能属性保留率提高约 28%。进一步的可解释性分析表明，学习到的密码子旋转分布重复了已知的 GC 含量偏差和 tRNA 丰度模式，提供了有关密码子使用的新见解。Equi-mRNA 为 mRNA 模型建立了一个新的生物学原理指导范式，具有对下一代治疗药物设计的重要影响。 

---
# Hydra: A 1.6B-Parameter State-Space Language Model with Sparse Attention, Mixture-of-Experts, and Memory 

**Title (ZH)**: Hydra：一个具有稀疏注意、专家混合和记忆的16亿参数状态空间语言模型 

**Authors**: Siddharth Chaudhary, Bennett Browning  

**Link**: [PDF](https://arxiv.org/pdf/2508.15099)  

**Abstract**: We present Hydra as an architectural proposal for hybrid long-context language models that combine conditional computation, long-context memory mechanisms, and sparse mixture-of-experts within an approximately 1.6B parameter design envelope. Hydra integrates a Mamba-style Structured State Space Model (SSM) backbone with intermittent sparse global attention, chunk-level MoE feed-forward routing, and dual (workspace plus factual PKM) memories. We formalize the component interfaces, give transparent parameter and complexity accounting, and outline a staged curriculum intended to stably activate the parts. We accompany the specification with illustrative toy-scale prototype measurements (tens of millions of parameters on synthetic data) whose sole purpose is to demonstrate implementation feasibility and qualitative scaling behaviors (for example, long-context throughput crossover and controllable expert routing), not to claim competitive full-scale performance. We explicitly delineate assumptions and open risks (training complexity, memory utilization, specialization dynamics) and position Hydra as a blueprint to stimulate empirical follow-up rather than a finished system. By combining SSM efficiency, selective sparse attention, MoE capacity, and learnable memory, Hydra sketches a path toward modular, input-adaptive long-context language models; validating end-task gains at target scale remains future work. 

**Abstract (ZH)**: Hydra：一种结合条件计算、长上下文记忆机制和稀疏专家混合的混合架构提案 

---
# Nemotron-CC-Math: A 133 Billion-Token-Scale High Quality Math Pretraining Dataset 

**Title (ZH)**: Nemotron-CC-Math：一个大规模高质量数学预训练数据集（133亿词 token 规模） 

**Authors**: Rabeeh Karimi Mahabadi, Sanjeev Satheesh, Shrimai Prabhumoye, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2508.15096)  

**Abstract**: Pretraining large language models (LLMs) on high-quality, structured data such as mathematics and code substantially enhances reasoning capabilities. However, existing math-focused datasets built from Common Crawl suffer from degraded quality due to brittle extraction heuristics, lossy HTML-to-text conversion, and the failure to reliably preserve mathematical structure. In this work, we introduce Nemotron-CC-Math, a large-scale, high-quality mathematical corpus constructed from Common Crawl using a novel, domain-agnostic pipeline specifically designed for robust scientific text extraction.
Unlike previous efforts, our pipeline recovers math across various formats (e.g., MathJax, KaTeX, MathML) by leveraging layout-aware rendering with lynx and a targeted LLM-based cleaning stage. This approach preserves the structural integrity of equations and code blocks while removing boilerplate, standardizing notation into LaTeX representation, and correcting inconsistencies.
We collected a large, high-quality math corpus, namely Nemotron-CC-Math-3+ (133B tokens) and Nemotron-CC-Math-4+ (52B tokens). Notably, Nemotron-CC-Math-4+ not only surpasses all prior open math datasets-including MegaMath, FineMath, and OpenWebMath-but also contains 5.5 times more tokens than FineMath-4+, which was previously the highest-quality math pretraining dataset. When used to pretrain a Nemotron-T 8B model, our corpus yields +4.8 to +12.6 gains on MATH and +4.6 to +14.3 gains on MBPP+ over strong baselines, while also improving general-domain performance on MMLU and MMLU-Stem.
We present the first pipeline to reliably extract scientific content--including math--from noisy web-scale data, yielding measurable gains in math, code, and general reasoning, and setting a new state of the art among open math pretraining corpora. To support open-source efforts, we release our code and datasets. 

**Abstract (ZH)**: 预训练大型语言模型（LLMs）在高质量、结构化的数据上，如数学和代码，极大地提升了推理能力。然而，现有专注于数学的数据集由于提取 heuristic 的脆弱性、HTML 转换的损失以及数学结构保真的失败，其质量存在下降。在本工作中，我们引入了 Nemotron-CC-Math，这是一种使用新颖的、跨领域的管道从 Common Crawl 构建的大规模高质量数学语料库，该管道专门设计用于稳健的科学文本提取。

我们的管道通过利用 lynx 进行布局感知渲染，并结合目标 LLM 基础的清理阶段，能够恢复各种格式（如 MathJax、KaTeX、MathML）的数学内容。这种方法在保留等式和代码块结构完整性的基础上，去除了冗余内容，将符号标准化为 LaTeX 表示，并纠正了不一致性。

我们收集了一个大规模、高质量的数学语料库，命名为 Nemotron-CC-Math-3+（133B 词元）和 Nemotron-CC-Math-4+（52B 词元）。值得注意的是，Nemotron-CC-Math-4+ 不仅超越了所有先前的开放数学数据集（包括 MegaMath、FineMath 和 OpenWebMath），而且其词元数量是 FineMath-4+ 的 5.5 倍。当用于预训练 Nemotron-T 8B 模型时，我们的语料库在 MATH 上带来了 4.8 到 12.6 的增益，在 MBPP+ 上带来了 4.6 到 14.3 的增益，同时在 MMLU 和 MMLU-Stem 的通用领域性能方面也有所提升。

我们首次提出了一种可靠地从嘈杂的网络数据中提取科学内容（包括数学）的方法，带来了数学、代码和一般推理能力的可测量提升，并在开放数学预训练语料库中设定了新的基准。为了支持开源努力，我们发布了代码和数据集。 

---
# Mapping the Course for Prompt-based Structured Prediction 

**Title (ZH)**: 基于提示的结构化预测框架研究 

**Authors**: Matt Pauk, Maria Leonor Pacheco  

**Link**: [PDF](https://arxiv.org/pdf/2508.15090)  

**Abstract**: LLMs have been shown to be useful for a variety of language tasks, without requiring task-specific fine-tuning. However, these models often struggle with hallucinations and complex reasoning problems due to their autoregressive nature. We propose to address some of these issues, specifically in the area of structured prediction, by combining LLMs with combinatorial inference in an attempt to marry the predictive power of LLMs with the structural consistency provided by inference methods. We perform exhaustive experiments in an effort to understand which prompting strategies can effectively estimate LLM confidence values for use with symbolic inference, and show that, regardless of the prompting strategy, the addition of symbolic inference on top of prompting alone leads to more consistent and accurate predictions. Additionally, we show that calibration and fine-tuning using structured prediction objectives leads to increased performance for challenging tasks, showing that structured learning is still valuable in the era of LLMs. 

**Abstract (ZH)**: LLMs在结构化预测中的组合推理：通过结合LLMs和组合推理解决幻觉和复杂推理问题 

---
# Wormhole Dynamics in Deep Neural Networks 

**Title (ZH)**: 深神经网络中的虫洞动力学 

**Authors**: Yen-Lung Lai, Zhe Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.15086)  

**Abstract**: This work investigates the generalization behavior of deep neural networks (DNNs), focusing on the phenomenon of "fooling examples," where DNNs confidently classify inputs that appear random or unstructured to humans. To explore this phenomenon, we introduce an analytical framework based on maximum likelihood estimation, without adhering to conventional numerical approaches that rely on gradient-based optimization and explicit labels. Our analysis reveals that DNNs operating in an overparameterized regime exhibit a collapse in the output feature space. While this collapse improves network generalization, adding more layers eventually leads to a state of degeneracy, where the model learns trivial solutions by mapping distinct inputs to the same output, resulting in zero loss. Further investigation demonstrates that this degeneracy can be bypassed using our newly derived "wormhole" solution. The wormhole solution, when applied to arbitrary fooling examples, reconciles meaningful labels with random ones and provides a novel perspective on shortcut learning. These findings offer deeper insights into DNN generalization and highlight directions for future research on learning dynamics in unsupervised settings to bridge the gap between theory and practice. 

**Abstract (ZH)**: 本研究探讨了深度神经网络（DNNs）的一般化行为，特别是“欺骗样本”现象，即DNNs对人类看来随机或无结构的输入表现出自信的分类能力。为探究这一现象，我们提出了一种基于最大似然估计的分析框架，不采用依赖梯度优化和明确标签的传统数值方法。我们的分析表明，在过度参数化区域运行的DNNs在输出特征空间中表现出坍缩现象。尽管此坍缩有助于网络的一般化，但在不断增加层数时，最终会导致模型退化状态，其通过将不同输入映射到相同的输出来学习琐屑解决方案，从而导致零损失。进一步的研究显示，我们新推导的“虫洞”解法可以绕过这种退化。当应用于任意欺骗样本时，“虫洞”解法能够将有意义的标签与随机标签统一起来，并为捷径学习提供新的视角。这些发现加深了对DNN一般化行为的理解，并指明了在无监督环境中探讨学习动态未来研究的方向，以弥合理论与实践之间的差距。 

---
# LongRecall: A Structured Approach for Robust Recall Evaluation in Long-Form Text 

**Title (ZH)**: 长文本稳健召回评价的结构化方法 

**Authors**: MohamamdJavad Ardestani, Ehsan Kamalloo, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2508.15085)  

**Abstract**: LongRecall. The completeness of machine-generated text, ensuring that it captures all relevant information, is crucial in domains such as medicine and law and in tasks like list-based question answering (QA), where omissions can have serious consequences. However, existing recall metrics often depend on lexical overlap, leading to errors with unsubstantiated entities and paraphrased answers, while LLM-as-a-Judge methods with long holistic prompts capture broader semantics but remain prone to misalignment and hallucinations without structured verification. We introduce LongRecall, a general three-stage recall evaluation framework that decomposes answers into self-contained facts, successively narrows plausible candidate matches through lexical and semantic filtering, and verifies their alignment through structured entailment checks. This design reduces false positives and false negatives while accommodating diverse phrasings and contextual variations, serving as a foundational building block for systematic recall assessment. We evaluate LongRecall on three challenging long-form QA benchmarks using both human annotations and LLM-based judges, demonstrating substantial improvements in recall accuracy over strong lexical and LLM-as-a-Judge baselines. 

**Abstract (ZH)**: LongRecall. 机器生成文本的完整性对于医学和法律等领域至关重要，特别是在基于列表的问题回答等任务中，遗漏可能导致严重后果。然而，现有的召回评估指标往往依赖于词级重叠，这会导致缺乏实证支持的实体和近义答案产生错误。而使用长时间全景提示的LLM-as-a-Judge方法虽然能够捕捉更广泛的语义，但在缺乏结构化验证的情况下仍然容易出现对齐失真和幻觉。我们提出了一种通用的三阶段召回评估框架LongRecall，该框架将答案分解为自包含的事实，通过逐步的词级和语义过滤缩小可能的匹配候选，通过结构化的推出验证其对齐性。这一设计减少了误报和漏报，同时能够适应多样的短语表达和上下文变化，成为系统性召回评估的基础构建模块。我们在三个具有挑战性的长文本问题回答基准上使用人工注释和基于LLM的裁判进行评估，结果表明LongRecall在召回准确性上显著优于强词级和LLM-as-a-Judge基准。 

---
# From Basic Affordances to Symbolic Thought: A Computational Phylogenesis of Biological Intelligence 

**Title (ZH)**: 从基本功能到符号思维：生物智能的计算进化论 

**Authors**: John E. Hummel, Rachel F. Heaton  

**Link**: [PDF](https://arxiv.org/pdf/2508.15082)  

**Abstract**: What is it about human brains that allows us to reason symbolically whereas most other animals cannot? There is evidence that dynamic binding, the ability to combine neurons into groups on the fly, is necessary for symbolic thought, but there is also evidence that it is not sufficient. We propose that two kinds of hierarchical integration (integration of multiple role-bindings into multiplace predicates, and integration of multiple correspondences into structure mappings) are minimal requirements, on top of basic dynamic binding, to realize symbolic thought. We tested this hypothesis in a systematic collection of 17 simulations that explored the ability of cognitive architectures with and without the capacity for multi-place predicates and structure mapping to perform various kinds of tasks. The simulations were as generic as possible, in that no task could be performed based on any diagnostic features, depending instead on the capacity for multi-place predicates and structure mapping. The results are consistent with the hypothesis that, along with dynamic binding, multi-place predicates and structure mapping are minimal requirements for basic symbolic thought. These results inform our understanding of how human brains give rise to symbolic thought and speak to the differences between biological intelligence, which tends to generalize broadly from very few training examples, and modern approaches to machine learning, which typically require millions or billions of training examples. The results we report also have important implications for bio-inspired artificial intelligence. 

**Abstract (ZH)**: 人类大脑为何能够进行象征性推理而大多数其他动物不能？动态绑定作为一种能力，能够在运行时将神经元组合成组，是象征性思维的必要条件，但并不足够。我们提出，基于基本动态绑定之上，还需要两种类型的层次整合（将多个角色绑定整合成多元谓词，将多个对应关系整合成结构映射）作为基本象征性思维的最小要求。我们在一个系统性的17个仿真中检验了这一假设，这些仿真测试了具有和不具有多元谓词能力和结构映射的认知架构在执行各种类型任务的能力。仿真尽可能地通用，即任何任务都不能基于诊断特征完成，而是依赖于多元谓词能力和结构映射的能力。结果与假设一致，即除了动态绑定之外，多元谓词和结构映射是基本象征性思维的最小要求。这些结果丰富了我们对人类大脑如何产生象征性思维的理解，并揭示了生物智能与通常需要大量训练样本的现代机器学习方法之间的差异。我们报告的结果还对受生物启发的人工智能产生了重要影响。 

---
# Decentralized Vision-Based Autonomous Aerial Wildlife Monitoring 

**Title (ZH)**: 基于视觉的分布式自主航空野生动物监控 

**Authors**: Makram Chahine, William Yang, Alaa Maalouf, Justin Siriska, Ninad Jadhav, Daniel Vogt, Stephanie Gil, Robert Wood, Daniela Rus  

**Link**: [PDF](https://arxiv.org/pdf/2508.15038)  

**Abstract**: Wildlife field operations demand efficient parallel deployment methods to identify and interact with specific individuals, enabling simultaneous collective behavioral analysis, and health and safety interventions. Previous robotics solutions approach the problem from the herd perspective, or are manually operated and limited in scale. We propose a decentralized vision-based multi-quadrotor system for wildlife monitoring that is scalable, low-bandwidth, and sensor-minimal (single onboard RGB camera). Our approach enables robust identification and tracking of large species in their natural habitat. We develop novel vision-based coordination and tracking algorithms designed for dynamic, unstructured environments without reliance on centralized communication or control. We validate our system through real-world experiments, demonstrating reliable deployment in diverse field conditions. 

**Abstract (ZH)**: 野生动物实地操作需要高效的并行部署方法以识别和互动特定个体，实现同时进行集体行为分析和健康与安全干预。以往的机器人解决方案从 herd 角度入手，或者手动操作且规模有限。我们提出了一种去中心化的基于视觉的多旋翼无人机系统，用于野生动物监控，该系统可扩展、低带宽且传感器最小化（单个机载 RGB 摄像头）。我们的方法能使我们在自然栖息地可靠地识别和跟踪大型物种。我们开发了一种新的基于视觉的协调和跟踪算法，适用于动态且结构不佳的环境，并且不依赖于中心化的通信或控制。我们通过实地试验验证了该系统，展示了其在多种现场条件下的可靠部署能力。 

---
# MoEcho: Exploiting Side-Channel Attacks to Compromise User Privacy in Mixture-of-Experts LLMs 

**Title (ZH)**: MoEcho：利用侧信道攻击以侵犯混合专家大语言模型中用户隐私 

**Authors**: Ruyi Ding, Tianhong Xu, Xinyi Shen, Aidong Adam Ding, Yunsi Fei  

**Link**: [PDF](https://arxiv.org/pdf/2508.15036)  

**Abstract**: The transformer architecture has become a cornerstone of modern AI, fueling remarkable progress across applications in natural language processing, computer vision, and multimodal learning. As these models continue to scale explosively for performance, implementation efficiency remains a critical challenge. Mixture of Experts (MoE) architectures, selectively activating specialized subnetworks (experts), offer a unique balance between model accuracy and computational cost. However, the adaptive routing in MoE architectures, where input tokens are dynamically directed to specialized experts based on their semantic meaning inadvertently opens up a new attack surface for privacy breaches. These input-dependent activation patterns leave distinctive temporal and spatial traces in hardware execution, which adversaries could exploit to deduce sensitive user data. In this work, we propose MoEcho, discovering a side channel analysis based attack surface that compromises user privacy on MoE based systems. Specifically, in MoEcho, we introduce four novel architectural side channels on different computing platforms, including Cache Occupancy Channels and Pageout+Reload on CPUs, and Performance Counter and TLB Evict+Reload on GPUs, respectively. Exploiting these vulnerabilities, we propose four attacks that effectively breach user privacy in large language models (LLMs) and vision language models (VLMs) based on MoE architectures: Prompt Inference Attack, Response Reconstruction Attack, Visual Inference Attack, and Visual Reconstruction Attack. MoEcho is the first runtime architecture level security analysis of the popular MoE structure common in modern transformers, highlighting a serious security and privacy threat and calling for effective and timely safeguards when harnessing MoE based models for developing efficient large scale AI services. 

**Abstract (ZH)**: 基于MoE架构的MoEcho：发现并分析其侧信道攻击面以保障用户隐私 

---
# A Systematic Survey of Model Extraction Attacks and Defenses: State-of-the-Art and Perspectives 

**Title (ZH)**: 模型提取攻击与防御的系统性综述：现状与展望 

**Authors**: Kaixiang Zhao, Lincan Li, Kaize Ding, Neil Zhenqiang Gong, Yue Zhao, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.15031)  

**Abstract**: Machine learning (ML) models have significantly grown in complexity and utility, driving advances across multiple domains. However, substantial computational resources and specialized expertise have historically restricted their wide adoption. Machine-Learning-as-a-Service (MLaaS) platforms have addressed these barriers by providing scalable, convenient, and affordable access to sophisticated ML models through user-friendly APIs. While this accessibility promotes widespread use of advanced ML capabilities, it also introduces vulnerabilities exploited through Model Extraction Attacks (MEAs). Recent studies have demonstrated that adversaries can systematically replicate a target model's functionality by interacting with publicly exposed interfaces, posing threats to intellectual property, privacy, and system security. In this paper, we offer a comprehensive survey of MEAs and corresponding defense strategies. We propose a novel taxonomy that classifies MEAs according to attack mechanisms, defense approaches, and computing environments. Our analysis covers various attack techniques, evaluates their effectiveness, and highlights challenges faced by existing defenses, particularly the critical trade-off between preserving model utility and ensuring security. We further assess MEAs within different computing paradigms and discuss their technical, ethical, legal, and societal implications, along with promising directions for future research. This systematic survey aims to serve as a valuable reference for researchers, practitioners, and policymakers engaged in AI security and privacy. Additionally, we maintain an online repository continuously updated with related literature at this https URL. 

**Abstract (ZH)**: 机器学习模型显著增长在复杂性和实用性方面，推动了多个领域的进步。然而，历史上传统上，大量计算资源和专门的技术知识限制了它们的广泛采用。作为一种解决方案，机器学习即服务（MLaaS）平台通过用户友好的API提供了可扩展、便捷且经济高效的高级机器学习模型访问途径。虽然这种易用性促进了高级机器学习能力的广泛应用，但也引入了通过模型提取攻击（Model Extraction Attacks，MEAs）被利用的安全漏洞。近期研究证明，敌对方可以通过与公开接口的交互系统性地复制目标模型的功能，从而对知识产权、隐私和系统安全构成威胁。在本文中，我们提供了对MEAs及其相应防御策略的全面综述。我们提出了一个新的分类法，根据攻击机制、防御方法和计算环境对MEAs进行分类。我们的分析涵盖了各种攻击技术，评估了它们的有效性，并突出了现有防御面临的挑战，特别是保持模型实用性和确保安全之间的关键权衡。我们进一步分析了MEAs在不同计算范式中的情况，并讨论了它们的技术、伦理、法律和社会意义，以及未来研究的有希望的方向。这项系统综述旨在为从事AI安全与隐私研究、实践和政策制定的研究人员提供有价值的参考。此外，我们在此httpsURL上维护了一个不断更新的相关文献在线仓库。 

---
# Reversible Unfolding Network for Concealed Visual Perception with Generative Refinement 

**Title (ZH)**: 可逆展开网络：具有生成性精修的隐藏视觉感知 

**Authors**: Chunming He, Fengyang Xiao, Rihan Zhang, Chengyu Fang, Deng-Ping Fan, Sina Farsiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15027)  

**Abstract**: Existing methods for concealed visual perception (CVP) often leverage reversible strategies to decrease uncertainty, yet these are typically confined to the mask domain, leaving the potential of the RGB domain underexplored. To address this, we propose a reversible unfolding network with generative refinement, termed RUN++. Specifically, RUN++ first formulates the CVP task as a mathematical optimization problem and unfolds the iterative solution into a multi-stage deep network. This approach provides a principled way to apply reversible modeling across both mask and RGB domains while leveraging a diffusion model to resolve the resulting uncertainty. Each stage of the network integrates three purpose-driven modules: a Concealed Object Region Extraction (CORE) module applies reversible modeling to the mask domain to identify core object regions; a Context-Aware Region Enhancement (CARE) module extends this principle to the RGB domain to foster better foreground-background separation; and a Finetuning Iteration via Noise-based Enhancement (FINE) module provides a final refinement. The FINE module introduces a targeted Bernoulli diffusion model that refines only the uncertain regions of the segmentation mask, harnessing the generative power of diffusion for fine-detail restoration without the prohibitive computational cost of a full-image process. This unique synergy, where the unfolding network provides a strong uncertainty prior for the diffusion model, allows RUN++ to efficiently direct its focus toward ambiguous areas, significantly mitigating false positives and negatives. Furthermore, we introduce a new paradigm for building robust CVP systems that remain effective under real-world degradations and extend this concept into a broader bi-level optimization framework. 

**Abstract (ZH)**: 现有的隐藏视觉感知方法（CVP）通常利用可逆策略来减少不确定性，但这些方法通常局限于掩码域，导致RGB域的潜力被忽视。为了解决这一问题，我们提出了一种带有生成性精炼的可逆展开网络，称为RUN++。具体而言，RUN++首先将CVP任务形式化为数学优化问题，并将迭代解展开为多阶段深度网络。这种方法为在掩码域和RGB域上应用可逆建模提供了严格的途径，同时利用扩散模型解决由此产生的不确定性。网络的每一阶段整合了三个目标驱动的模块：隐藏对象区域提取（CORE）模块将可逆建模应用于掩码域以识别核心对象区域；上下文感知区域增强（CARE）模块将此原则扩展到RGB域以促进更好的前景-背景分离；并通过噪声增强的精细调整迭代（FINE）模块提供最终精炼。FINE模块引入了针对不确定区域的泊松扩散模型，仅对分割掩码中的不确定区域进行细化，利用扩散的生成能力进行细节恢复，而不需要全图像处理的高昂计算成本。这种独特的协同作用，即展开网络为扩散模型提供了强烈的不确定性先验，使RUN++能够有效地将注意力集中在模糊区域，显著降低了假阳性与假阴性。此外，我们还提出了一种新的建模范式，用于构建在真实世界退化下仍然有效的CVP系统，并将这一概念扩展到更广泛的多层次优化框架。 

---
# TAIGen: Training-Free Adversarial Image Generation via Diffusion Models 

**Title (ZH)**: TAIGen：无需训练的对抗图像生成方法基于扩散模型 

**Authors**: Susim Roy, Anubhooti Jain, Mayank Vatsa, Richa Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.15020)  

**Abstract**: Adversarial attacks from generative models often produce low-quality images and require substantial computational resources. Diffusion models, though capable of high-quality generation, typically need hundreds of sampling steps for adversarial generation. This paper introduces TAIGen, a training-free black-box method for efficient adversarial image generation. TAIGen produces adversarial examples using only 3-20 sampling steps from unconditional diffusion models. Our key finding is that perturbations injected during the mixing step interval achieve comparable attack effectiveness without processing all timesteps. We develop a selective RGB channel strategy that applies attention maps to the red channel while using GradCAM-guided perturbations on green and blue channels. This design preserves image structure while maximizing misclassification in target models. TAIGen maintains visual quality with PSNR above 30 dB across all tested datasets. On ImageNet with VGGNet as source, TAIGen achieves 70.6% success against ResNet, 80.8% against MNASNet, and 97.8% against ShuffleNet. The method generates adversarial examples 10x faster than existing diffusion-based attacks. Our method achieves the lowest robust accuracy, indicating it is the most impactful attack as the defense mechanism is least successful in purifying the images generated by TAIGen. 

**Abstract (ZH)**: 基于生成模型的对抗攻击通常会产生低质量图像并需要大量计算资源。尽管扩散模型能够生成高质量的图像，但在对抗生成时通常需要数百个采样步骤。本文介绍了一种无需训练的高效黑盒对抗图像生成方法TAIGen。TAIGen仅需从无条件扩散模型中生成3-20个采样步骤即可生成对抗样本。我们的关键发现是在混合步骤间隔中注入的扰动可以在不处理所有时间步的情况下实现可比的攻击效果。我们开发了一种选择性的RGB通道策略，该策略在红色通道上应用注意力图，并在绿色和蓝色通道上使用GradCAM引导的扰动。该设计保留了图像结构，同时最大化目标模型的误分类。TAIGen在所有测试数据集上均保持了PSNR超过30 dB的视觉质量。在使用VGGNet作为来源的ImageNet上，TAIGen在ResNet上的成功率达到了70.6%，在MNASNet上的成功率达到了80.8%，在ShuffleNet上的成功率达到了97.8%。与现有的基于扩散的攻击方法相比，该方法生成对抗样本的速度快10倍。该方法实现了最低的鲁棒准确率，表明它是最具有影响力的攻击方法，因为防御机制在净化由TAIGen生成的图像方面最不成功。 

---
# Twin-Boot: Uncertainty-Aware Optimization via Online Two-Sample Bootstrapping 

**Title (ZH)**: Twin-Boot: 基于在线两样本bootstrapping的不确定度意识优化 

**Authors**: Carlos Stein Brito  

**Link**: [PDF](https://arxiv.org/pdf/2508.15019)  

**Abstract**: Standard gradient descent methods yield point estimates with no measure of confidence. This limitation is acute in overparameterized and low-data regimes, where models have many parameters relative to available data and can easily overfit. Bootstrapping is a classical statistical framework for uncertainty estimation based on resampling, but naively applying it to deep learning is impractical: it requires training many replicas, produces post-hoc estimates that cannot guide learning, and implicitly assumes comparable optima across runs - an assumption that fails in non-convex landscapes. We introduce Twin-Bootstrap Gradient Descent (Twin-Boot), a resampling-based training procedure that integrates uncertainty estimation into optimization. Two identical models are trained in parallel on independent bootstrap samples, and a periodic mean-reset keeps both trajectories in the same basin so that their divergence reflects local (within-basin) uncertainty. During training, we use this estimate to sample weights in an adaptive, data-driven way, providing regularization that favors flatter solutions. In deep neural networks and complex high-dimensional inverse problems, the approach improves calibration and generalization and yields interpretable uncertainty maps. 

**Abstract (ZH)**: 基于Bootstrap的梯度下降方法（Twin-Bootstrap Gradient Descent）整合了优化过程中的不确定性估计 

---
# Quantized Neural Networks for Microcontrollers: A Comprehensive Review of Methods, Platforms, and Applications 

**Title (ZH)**: 微控制器上量化的神经网络：方法、平台和应用综述 

**Authors**: Hamza A. Abushahla, Dara Varam, Ariel J. N. Panopio, Mohamed I. AlHajri  

**Link**: [PDF](https://arxiv.org/pdf/2508.15008)  

**Abstract**: The deployment of Quantized Neural Networks (QNNs) on resource-constrained devices, such as microcontrollers, has introduced significant challenges in balancing model performance, computational complexity and memory constraints. Tiny Machine Learning (TinyML) addresses these issues by integrating advancements across machine learning algorithms, hardware acceleration, and software optimization to efficiently run deep neural networks on embedded systems. This survey presents a hardware-centric introduction to quantization, systematically reviewing essential quantization techniques employed to accelerate deep learning models for embedded applications. In particular, further emphasis is put on critical trade-offs among model performance and hardware capabilities. The survey further evaluates existing software frameworks and hardware platforms designed specifically for supporting QNN execution on microcontrollers. Moreover, we provide an analysis of the current challenges and an outline of promising future directions in the rapidly evolving domain of QNN deployment. 

**Abstract (ZH)**: 量化神经网络（QNNs）在微控制器等资源受限设备上的部署引入了在模型性能、计算复杂度和内存约束之间平衡的重大挑战。Tiny Machine Learning (TinyML) 通过整合机器学习算法、硬件加速和软件优化的进步，旨在高效地在嵌入式系统上运行深度神经网络。本文综述提供了一种以硬件为中心的量化介绍，系统性地回顾了用于加速嵌入式应用中深度学习模型的量化技术，并特别强调了模型性能与硬件能力之间的关键权衡。此外，本文评估了专门用于支持微控制器上QNN执行的现有软件框架和硬件平台，并对当前挑战进行了分析，概述了这一快速发展的QNN部署领域的有希望的未来方向。 

---
# Fast Graph Neural Network for Image Classification 

**Title (ZH)**: 快速图神经网络图像分类 

**Authors**: Mustafa Mohammadi Gharasuie, Luis Rueda  

**Link**: [PDF](https://arxiv.org/pdf/2508.14958)  

**Abstract**: The rapid progress in image classification has been largely driven by the adoption of Graph Convolutional Networks (GCNs), which offer a robust framework for handling complex data structures. This study introduces a novel approach that integrates GCNs with Voronoi diagrams to enhance image classification by leveraging their ability to effectively model relational data. Unlike conventional convolutional neural networks (CNNs), our method represents images as graphs, where pixels or regions function as vertices. These graphs are then refined using corresponding Delaunay triangulations, optimizing their representation. The proposed model achieves significant improvements in both preprocessing efficiency and classification accuracy across various benchmark datasets, surpassing state-of-the-art approaches, particularly in challenging scenarios involving intricate scenes and fine-grained categories. Experimental results, validated through cross-validation, underscore the effectiveness of combining GCNs with Voronoi diagrams for advancing image classification. This research not only presents a novel perspective on image classification but also expands the potential applications of graph-based learning paradigms in computer vision and unstructured data analysis. 

**Abstract (ZH)**: 图卷积网络与Voronoi图集成的图像分类新方法 

---
# Quantum Long Short-term Memory with Differentiable Architecture Search 

**Title (ZH)**: 可微架构搜索的量子长短期记忆模型 

**Authors**: Samuel Yen-Chi Chen, Prayag Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2508.14955)  

**Abstract**: Recent advances in quantum computing and machine learning have given rise to quantum machine learning (QML), with growing interest in learning from sequential data. Quantum recurrent models like QLSTM are promising for time-series prediction, NLP, and reinforcement learning. However, designing effective variational quantum circuits (VQCs) remains challenging and often task-specific. To address this, we propose DiffQAS-QLSTM, an end-to-end differentiable framework that optimizes both VQC parameters and architecture selection during training. Our results show that DiffQAS-QLSTM consistently outperforms handcrafted baselines, achieving lower loss across diverse test settings. This approach opens the door to scalable and adaptive quantum sequence learning. 

**Abstract (ZH)**: 近期，量子计算和机器学习的进展催生了量子机器学习（QML），其中对序贯数据的学习引起了越来越多的兴趣。像量子循环模型（如QLSTM）这样的量子递归模型在时间序列预测、自然语言处理和强化学习方面表现出巨大潜力。然而，设计有效的变量子电路（VQCs）仍然具有挑战性且通常需要针对特定任务。为了解决这一问题，我们提出了一种端到端可微分框架DiffQAS-QLSTM，该框架在训练过程中同时优化VQC参数和架构选择。我们的结果表明，DiffQAS-QLSTM在多种测试设置下一致优于手工设计的基线模型，实现了较低的损失值。这种方法为可扩展和自适应的量子序列学习打开了大门。 

---
# Can synthetic data reproduce real-world findings in epidemiology? A replication study using tree-based generative AI 

**Title (ZH)**: 合成数据能否再现流行病学中的现实世界发现？基于树的生成AI再现研究 

**Authors**: Jan Kapar, Kathrin Günther, Lori Ann Vallis, Klaus Berger, Nadine Binder, Hermann Brenner, Stefanie Castell, Beate Fischer, Volker Harth, Bernd Holleczek, Timm Intemann, Till Ittermann, André Karch, Thomas Keil, Lilian Krist, Berit Lange, Michael F. Leitzmann, Katharina Nimptsch, Nadia Obi, Iris Pigeot, Tobias Pischon, Tamara Schikowski, Börge Schmidt, Carsten Oliver Schmidt, Anja M. Sedlmair, Justine Tanoey, Harm Wienbergen, Andreas Wienke, Claudia Wigmann, Marvin N. Wright  

**Link**: [PDF](https://arxiv.org/pdf/2508.14936)  

**Abstract**: Generative artificial intelligence for synthetic data generation holds substantial potential to address practical challenges in epidemiology. However, many current methods suffer from limited quality, high computational demands, and complexity for non-experts. Furthermore, common evaluation strategies for synthetic data often fail to directly reflect statistical utility. Against this background, a critical underexplored question is whether synthetic data can reliably reproduce key findings from epidemiological research. We propose the use of adversarial random forests (ARF) as an efficient and convenient method for synthesizing tabular epidemiological data. To evaluate its performance, we replicated statistical analyses from six epidemiological publications and compared original with synthetic results. These publications cover blood pressure, anthropometry, myocardial infarction, accelerometry, loneliness, and diabetes, based on data from the German National Cohort (NAKO Gesundheitsstudie), the Bremen STEMI Registry U45 Study, and the Guelph Family Health Study. Additionally, we assessed the impact of dimensionality and variable complexity on synthesis quality by limiting datasets to variables relevant for individual analyses, including necessary derivations. Across all replicated original studies, results from multiple synthetic data replications consistently aligned with original findings. Even for datasets with relatively low sample size-to-dimensionality ratios, the replication outcomes closely matched the original results across various descriptive and inferential analyses. Reducing dimensionality and pre-deriving variables further enhanced both quality and stability of the results. 

**Abstract (ZH)**: 生成式人工智能在合成数据生成方面的应用在流行病学中具有巨大潜力，但当前许多方法存在质量有限、高计算需求和非专业人士使用复杂等问题。此外，合成数据的常用评估策略往往未能直接反映统计效用。在这一背景下，一个亟待探索的关键问题是合成数据能否可靠重现流行病学研究中的重要发现。我们提出使用对抗随机森林（ARF）作为一种高效且便捷的方法来合成表格形式的流行病学数据。为了评估其性能，我们复制了六篇流行病学出版物中的统计分析，并将原始结果与合成结果进行了比较。这些出版物涵盖了血压、人体测量学、心肌梗死、加速度计数据、孤独感和糖尿病，基于德国国家队列研究（NAKO Gesundheitsstudie）、不莱梅STEMI登记U45研究和圭尔夫家庭健康研究的数据。此外，我们还通过限制数据集仅包括与个别分析相关且必要的变量来评估维度和变量复杂性对合成质量的影响。在所有复制的原创研究中，多份合成数据复制的结果与原始发现保持一致。即使对于样本大小与维度比率相对较低的数据集，复制结果在各种描述性和推断性分析中也与原始结果高度一致。进一步减少维度并预先计算变量进一步提升了结果的质量和稳定性。 

---
# Inference Time Debiasing Concepts in Diffusion Models 

**Title (ZH)**: 差异化概念在扩散模型中的推断时间去偏见 

**Authors**: Lucas S. Kupssinskü, Marco N. Bochernitsan, Jordan Kopper, Otávio Parraga, Rodrigo C. Barros  

**Link**: [PDF](https://arxiv.org/pdf/2508.14933)  

**Abstract**: We propose DeCoDi, a debiasing procedure for text-to-image diffusion-based models that changes the inference procedure, does not significantly change image quality, has negligible compute overhead, and can be applied in any diffusion-based image generation model. DeCoDi changes the diffusion process to avoid latent dimension regions of biased concepts. While most deep learning debiasing methods require complex or compute-intensive interventions, our method is designed to change only the inference procedure. Therefore, it is more accessible to a wide range of practitioners. We show the effectiveness of the method by debiasing for gender, ethnicity, and age for the concepts of nurse, firefighter, and CEO. Two distinct human evaluators manually inspect 1,200 generated images. Their evaluation results provide evidence that our method is effective in mitigating biases based on gender, ethnicity, and age. We also show that an automatic bias evaluation performed by the GPT4o is not significantly statistically distinct from a human evaluation. Our evaluation shows promising results, with reliable levels of agreement between evaluators and more coverage of protected attributes. Our method has the potential to significantly improve the diversity of images it generates by diffusion-based text-to-image generative models. 

**Abstract (ZH)**: 我们提出了DeCoDi，这是一种针对基于文本到图像扩散模型的去偏见程序，该程序改变推理过程，对图像质量影响不大，计算开销可忽略不计，并可应用于任何基于扩散的过程生成图像的模型。DeCoDi通过避免潜在维度中含有偏见概念的区域来改变扩散过程。尽管大多数深度学习去偏见方法需要复杂的或计算密集型的干预，我们的方法仅改变推理过程。因此，它更适用于广泛的实践者。我们通过去偏见护士、消防员和CEO等概念中的性别、种族和年龄偏见，展示了该方法的有效性。两名独立的人类评估员手动检查了1,200张生成的图像。他们的评估结果提供了证据，证明我们的方法在基于性别、种族和年龄方面有效减少了偏见。我们还展示了GPT4o执行的自动偏见评估与人类评估之间在统计上没有显著差异。我们的评估显示了很有希望的结果，评估者之间的一致性水平可靠，且涵盖了更多的受保护属性。该方法有可能显著提高基于文本到图像生成模型的图像多样性。 

---
# TOM: An Open-Source Tongue Segmentation Method with Multi-Teacher Distillation and Task-Specific Data Augmentation 

**Title (ZH)**: TOM：一种基于多教师蒸馏和任务特定数据增强的开源舌段分割方法 

**Authors**: Jiacheng Xie, Ziyang Zhang, Biplab Poudel, Congyu Guo, Yang Yu, Guanghui An, Xiaoting Tang, Lening Zhao, Chunhui Xu, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14932)  

**Abstract**: Tongue imaging serves as a valuable diagnostic tool, particularly in Traditional Chinese Medicine (TCM). The quality of tongue surface segmentation significantly affects the accuracy of tongue image classification and subsequent diagnosis in intelligent tongue diagnosis systems. However, existing research on tongue image segmentation faces notable limitations, and there is a lack of robust and user-friendly segmentation tools. This paper proposes a tongue image segmentation model (TOM) based on multi-teacher knowledge distillation. By incorporating a novel diffusion-based data augmentation method, we enhanced the generalization ability of the segmentation model while reducing its parameter size. Notably, after reducing the parameter count by 96.6% compared to the teacher models, the student model still achieves an impressive segmentation performance of 95.22% mIoU. Furthermore, we packaged and deployed the trained model as both an online and offline segmentation tool (available at this https URL), allowing TCM practitioners and researchers to use it without any programming experience. We also present a case study on TCM constitution classification using segmented tongue patches. Experimental results demonstrate that training with tongue patches yields higher classification performance and better interpretability than original tongue images. To our knowledge, this is the first open-source and freely available tongue image segmentation tool. 

**Abstract (ZH)**: 基于多师知识蒸馏的舌象图像分割模型：一种增强泛化能力的同时减少参数量的方法及其应用 

---
# Heatmap Regression without Soft-Argmax for Facial Landmark Detection 

**Title (ZH)**: 无需软argmax的热图回归 Facial特征点检测 

**Authors**: Chiao-An Yang, Raymond A. Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2508.14929)  

**Abstract**: Facial landmark detection is an important task in computer vision with numerous applications, such as head pose estimation, expression analysis, face swapping, etc. Heatmap regression-based methods have been widely used to achieve state-of-the-art results in this task. These methods involve computing the argmax over the heatmaps to predict a landmark. Since argmax is not differentiable, these methods use a differentiable approximation, Soft-argmax, to enable end-to-end training on deep-nets. In this work, we revisit this long-standing choice of using Soft-argmax and demonstrate that it is not the only way to achieve strong performance. Instead, we propose an alternative training objective based on the classic structured prediction framework. Empirically, our method achieves state-of-the-art performance on three facial landmark benchmarks (WFLW, COFW, and 300W), converging 2.2x faster during training while maintaining better/competitive accuracy. Our code is available here: this https URL. 

**Abstract (ZH)**: 面部关键点检测是计算机视觉中的一个重要任务，广泛应用于头部姿态估计、表情分析、面部替换等。基于热图回归的方法在这一任务中被广泛使用以实现最先进的成果。这些方法涉及在热图上计算argmax来预测关键点。由于argmax不具备可微性，这些方法使用可微近似Soft-argmax以在深度网络中实现端到端的训练。在本文中，我们重新审视了使用Soft-argmax这一长期选择，并证明它并非实现优异性能的唯一途径。相反，我们提出了一种基于经典结构化预测框架的替代训练目标。实验结果表明，我们的方法在三个面部关键点检测基准（WFLW、COFW和300W）上实现了最先进的性能，在训练过程中快2.2倍地收敛，并且具有更好的/竞争性的精度。我们的代码可在以下链接获取：this https URL。 

---
# AI Testing Should Account for Sophisticated Strategic Behaviour 

**Title (ZH)**: AI测试应考虑复杂的策略行为 

**Authors**: Vojtech Kovarik, Eric Olav Chen, Sami Petersen, Alexis Ghersengorin, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2508.14927)  

**Abstract**: This position paper argues for two claims regarding AI testing and evaluation. First, to remain informative about deployment behaviour, evaluations need account for the possibility that AI systems understand their circumstances and reason strategically. Second, game-theoretic analysis can inform evaluation design by formalising and scrutinising the reasoning in evaluation-based safety cases. Drawing on examples from existing AI systems, a review of relevant research, and formal strategic analysis of a stylised evaluation scenario, we present evidence for these claims and motivate several research directions. 

**Abstract (ZH)**: 本立场论文提出关于AI测试与评估的两个观点。首先，为了保持对部署行为的说明性，评估需要考虑到AI系统可能理解其环境并进行战略推理的可能性。其次，博弈论分析可以通过正式化和审查基于评估的安全案例中的推理来指导评估设计。通过现有AI系统的实例、相关研究的回顾以及对一种简化评估场景的正式战略分析，我们提供了这些观点的证据，并促进了几个研究方向。 

---
# Learning to Drive Ethically: Embedding Moral Reasoning into Autonomous Driving 

**Title (ZH)**: 学习驾驶伦理：将道德推理嵌入自动驾驶 

**Authors**: Dianzhao Li, Ostap Okhrin  

**Link**: [PDF](https://arxiv.org/pdf/2508.14926)  

**Abstract**: Autonomous vehicles hold great promise for reducing traffic fatalities and improving transportation efficiency, yet their widespread adoption hinges on embedding robust ethical reasoning into routine and emergency maneuvers. Here, we present a hierarchical Safe Reinforcement Learning (Safe RL) framework that explicitly integrates moral considerations with standard driving objectives. At the decision level, a Safe RL agent is trained using a composite ethical risk cost, combining collision probability and harm severity, to generate high-level motion targets. A dynamic Prioritized Experience Replay mechanism amplifies learning from rare but critical, high-risk events. At the execution level, polynomial path planning coupled with Proportional-Integral-Derivative (PID) and Stanley controllers translates these targets into smooth, feasible trajectories, ensuring both accuracy and comfort. We train and validate our approach on rich, real-world traffic datasets encompassing diverse vehicles, cyclists, and pedestrians, and demonstrate that it outperforms baseline methods in reducing ethical risk and maintaining driving performance. To our knowledge, this is the first study of ethical decision-making for autonomous vehicles via Safe RL in real-world scenarios. Our results highlight the potential of combining formal control theory and data-driven learning to advance ethically accountable autonomy in complex, human-mixed traffic environments. 

**Abstract (ZH)**: 自主驾驶车辆蕴含着降低交通事故死亡率和提升运输效率的巨大潜力，但其广泛采用取决于将坚实的伦理推理嵌入到常规和紧急操作中。在这里，我们提出了一种分层的安全强化学习（Safe RL）框架，该框架明确地将道德考虑与标准驾驶目标相结合。在决策层面，安全RL代理通过结合碰撞概率和伤害严重性复合伦理风险成本进行训练，以生成高级运动目标。动态优先经验回放机制增强了对罕见但关键的高风险事件的学习。在执行层面，多项式路径规划结合比例积分微分（PID）和斯坦利控制器将这些目标转化为平滑可行的轨迹，确保准确性和舒适性。我们通过涵盖多种车辆、自行车和行人的丰富实际交通数据集训练和验证了该方法，并证明它在降低伦理风险和保持驾驶性能方面优于基线方法。据我们所知，这是首次在真实场景中通过安全RL进行自主驾驶车辆的伦理决策研究。我们的结果强调了将形式控制理论与数据驱动学习相结合以在复杂混有人类的交通环境中推进可问责自主性的潜力。 

---
# A U-Statistic-based random forest approach for genetic interaction study 

**Title (ZH)**: 基于U-统计量的随机森林方法在遗传互作研究中的应用 

**Authors**: Ming Li, Ruo-Sin Peng, Changshuai Wei, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14924)  

**Abstract**: Variations in complex traits are influenced by multiple genetic variants, environmental risk factors, and their interactions. Though substantial progress has been made in identifying single genetic variants associated with complex traits, detecting the gene-gene and gene-environment interactions remains a great challenge. When a large number of genetic variants and environmental risk factors are involved, searching for interactions is limited to pair-wise interactions due to the exponentially increased feature space and computational intensity. Alternatively, recursive partitioning approaches, such as random forests, have gained popularity in high-dimensional genetic association studies. In this article, we propose a U-Statistic-based random forest approach, referred to as Forest U-Test, for genetic association studies with quantitative traits. Through simulation studies, we showed that the Forest U-Test outperformed existing methods. The proposed method was also applied to study Cannabis Dependence CD, using three independent datasets from the Study of Addiction: Genetics and Environment. A significant joint association was detected with an empirical p-value less than 0.001. The finding was also replicated in two independent datasets with p-values of 5.93e-19 and 4.70e-17, respectively. 

**Abstract (ZH)**: 复杂性状的变化受多个基因变异、环境风险因素及其相互作用的影响。尽管在识别与复杂性状相关的单个基因变异方面取得了显著进展，但检测基因-基因和基因-环境相互作用仍然是一项巨大挑战。当涉及大量基因变异和环境风险因素时，由于特征空间的指数级增加和计算强度的增加，搜索相互作用通常局限于成对相互作用。相反，在高维遗传关联研究中，递归分割方法，如随机森林，已经变得流行。本文提出了一种基于U-统计量的随机森林方法，名为Forest U-Test，用于具有数量性状的遗传关联研究。通过模拟研究，我们表明Forest U-Test优于现有方法。该方法还应用于对《 Addiction: Genetics and Environment 研究中的大麻依赖 CD 进行研究，使用了三个独立数据集。检测到一个显著联合关联，具有小于0.001的经验p值。该发现还在两个独立数据集中得到重复，p值分别为5.93e-19和4.70e-17。 

---
# Fusing Structural Phenotypes with Functional Data for Early Prediction of Primary Angle Closure Glaucoma Progression 

**Title (ZH)**: 将结构表型与功能性数据融合以预测原发性Angle闭角型青光眼的早期进展 

**Authors**: Swati Sharma, Thanadet Chuangsuwanich, Royston K.Y. Tan, Shimna C. Prasad, Tin A. Tun, Shamira A. Perera, Martin L. Buist, Tin Aung, Monisha E. Nongpiur, Michaël J. A. Girard  

**Link**: [PDF](https://arxiv.org/pdf/2508.14922)  

**Abstract**: Purpose: To classify eyes as slow or fast glaucoma progressors in patients with primary angle closure glaucoma (PACG) using an integrated approach combining optic nerve head (ONH) structural features and sector-based visual field (VF) functional parameters. Methods: PACG patients with >5 reliable VF tests over >5 years were included. Progression was assessed in Zeiss Forum, with baseline VF within six months of OCT. Fast progression was VFI decline <-2.0% per year; slow progression >-2.0% per year. OCT volumes were AI-segmented to extract 31 ONH parameters. The Glaucoma Hemifield Test defined five regions per hemifield, aligned with RNFL distribution. Mean sensitivity per region was combined with structural parameters to train ML classifiers. Multiple models were tested, and SHAP identified key predictors. Main outcome measures: Classification of slow versus fast progressors using combined structural and functional data. Results: We analyzed 451 eyes from 299 patients. Mean VFI progression was -0.92% per year; 369 eyes progressed slowly and 82 rapidly. The Random Forest model combining structural and functional features achieved the best performance (AUC = 0.87, 2000 Monte Carlo iterations). SHAP identified six key predictors: inferior MRW, inferior and inferior-temporal RNFL thickness, nasal-temporal LC curvature, superior nasal VF sensitivity, and inferior RNFL and GCL+IPL thickness. Models using only structural or functional features performed worse with AUC of 0.82 and 0.78, respectively. Conclusions: Combining ONH structural and VF functional parameters significantly improves classification of progression risk in PACG. Inferior ONH features, MRW and RNFL thickness, were the most predictive, highlighting the critical role of ONH morphology in monitoring disease progression. 

**Abstract (ZH)**: 目的：通过结合视盘结构特征和基于区域的视野功能参数，使用集成方法将原发性 Angle 封闭性青光眼（PACG）患者分为缓慢和快速进展组。方法：纳入在过去5年内进行超过5次可靠视野检查的PACG患者。视野进展情况在Zeiss Forum中评估，基线视野在OCT检查后六个月内进行。快速进展定义为视野指数（VFI）下降速率<-2.0%/年；缓慢进展定义为VFI下降速率>-2.0%/年。使用AI进行OCT体积分割以提取31个视盘参数。Glaucoma半场测试定义了每个半场的五个区域，与RNFL分布对应。将每个区域的平均敏感度与结构参数结合以训练机器学习分类器。测试了多种模型，SHAP确定了关键预测因子。主要结果指标：使用结合结构和功能数据分类缓慢和快速进展者。结果：分析了299名患者的451只眼。平均VFI进展率为-0.92%/年；369只眼缓慢进展，82只眼快速进展。结合结构和功能特征的随机森林模型表现最佳（AUC = 0.87，蒙特卡洛迭代2000次）。SHAP确定了六个关键预测因子：下象限MRW、下象限和下颞象限RNFL厚度、鼻颞LC曲率、优越鼻VF敏感度、下象限RNFL和GCL+IPL厚度。仅使用结构或功能特征的模型性能较差，AUC分别为0.82和0.78。结论：结合视盘结构和视野功能参数显著提高了PACG进展风险的分类。下视盘特征、MRW和RNFL厚度是最具预测性的，突显了视盘形态在监测疾病进展中的关键作用。 

---
# Designing an Interdisciplinary Artificial Intelligence Curriculum for Engineering: Evaluation and Insights from Experts 

**Title (ZH)**: 工程领域跨学科人工智能课程设计：专家评估与见解 

**Authors**: Johannes Schleiss, Anke Manukjan, Michelle Ines Bieber, Sebastian Lang, Sebastian Stober  

**Link**: [PDF](https://arxiv.org/pdf/2508.14921)  

**Abstract**: As Artificial Intelligence (AI) increasingly impacts professional practice, there is a growing need to AI-related competencies into higher education curricula. However, research on the implementation of AI education within study programs remains limited and requires new forms of collaboration across disciplines. This study addresses this gap and explores perspectives on interdisciplinary curriculum development through the lens of different stakeholders. In particular, we examine the case of curriculum development for a novel undergraduate program in AI in engineering. The research uses a mixed methods approach, combining quantitative curriculum mapping with qualitative focus group interviews. In addition to assessing the alignment of the curriculum with the targeted competencies, the study also examines the perceived quality, consistency, practicality and effectiveness from both academic and industry perspectives, as well as differences in perceptions between educators who were involved in the development and those who were not. The findings provide a practical understanding of the outcomes of interdisciplinary AI curriculum development and contribute to a broader understanding of how educator participation in curriculum development influences perceptions of quality aspects. It also advances the field of AI education by providing a reference point and insights for further interdisciplinary curriculum developments in response to evolving industry needs. 

**Abstract (ZH)**: 随着人工智能（AI）日益影响专业实践，将与AI相关的能力纳入高等教育课程的需求不断增长。然而，关于在学习项目中实施AI教育的研究仍然有限，需要跨学科的新形式的合作。本研究填补了这一空白，通过不同利益相关者的视角探索跨学科课程开发的见解。特别地，我们探讨了人工智能工程本科项目课程开发的案例。研究采用混合方法，结合定量课程映射与定性焦点小组访谈。除了评估课程与目标能力的契合程度外，研究还从学术和工业视角考察了课程的质量、一致性、实用性与有效性，并分析了参与课程开发的教育者与未参与者的感知差异。研究成果提供了跨学科AI课程开发成果的实际理解，并为理解教育者参与课程开发如何影响质量感知提供了更广泛的见解。此外，该研究还通过提供参考点和进一步跨学科课程开发的洞察，推进了AI教育领域的发展，以应对不断变化的行业需求。 

---
# Disentangling the Drivers of LLM Social Conformity: An Uncertainty-Moderated Dual-Process Mechanism 

**Title (ZH)**: 分离大语言模型社会 conformity 的驱动因素：一种不确定性调节的双过程机制 

**Authors**: Huixin Zhong, Yanan Liu, Qi Cao, Shijin Wang, Zijing Ye, Zimu Wang, Shiyao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14918)  

**Abstract**: As large language models (LLMs) integrate into collaborative teams, their social conformity -- the tendency to align with majority opinions -- has emerged as a key concern. In humans, conformity arises from informational influence (rational use of group cues for accuracy) or normative influence (social pressure for approval), with uncertainty moderating this balance by shifting from purely analytical to heuristic processing. It remains unclear whether these human psychological mechanisms apply to LLMs. This study adapts the information cascade paradigm from behavioral economics to quantitatively disentangle the two drivers to investigate the moderate effect. We evaluated nine leading LLMs across three decision-making scenarios (medical, legal, investment), manipulating information uncertainty (q = 0.667, 0.55, and 0.70, respectively). Our results indicate that informational influence underpins the models' behavior across all contexts, with accuracy and confidence consistently rising with stronger evidence. However, this foundational mechanism is dramatically modulated by uncertainty. In low-to-medium uncertainty scenarios, this informational process is expressed as a conservative strategy, where LLMs systematically underweight all evidence sources. In contrast, high uncertainty triggers a critical shift: while still processing information, the models additionally exhibit a normative-like amplification, causing them to overweight public signals (beta > 1.55 vs. private beta = 0.81). 

**Abstract (ZH)**: 大规模语言模型（LLMs）融入协作团队后，其社会 conformity 的影响——倾向于与多数意见一致——已成为一个重要关切。本研究借鉴行为经济学中的信息级联范式，定量解析信息影响和社会影响之间的关系，以探讨其平衡的中等影响。我们评估了九种领先的LLMs在三种决策场景（医疗、法律、投资）中的表现，并操纵信息不确定性（分别为q = 0.667、0.55和0.70）。结果表明，在所有情境下，信息影响支撑着模型的行为，准确性和信心随着更强证据的一致性而提升。然而，这一基础机制在不确定性的影响下发生了显著变化。在低至中等不确定性情境中，这一信息过程表现为保守策略，LLMs系统性地低估了所有证据来源。相反，在高不确定性情境下，这种信息处理触发了一种关键转变：模型除了继续处理信息外，还会表现出类似规范影响的放大效应，导致它们过度重视公开信号（β > 1.55，对比私人信号β = 0.81）。 

---
# Transsion Multilingual Speech Recognition System for MLC-SLM 2025 Challenge 

**Title (ZH)**: Transsion 多语种语音识别系统for MLC-SLM 2025 挑战赛 

**Authors**: Xiaoxiao Li, An Zhu, Youhai Jiang, Fengjie Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14916)  

**Abstract**: This paper presents the architecture and performance of a novel Multilingual Automatic Speech Recognition (ASR) system developed by the Transsion Speech Team for Track 1 of the MLC-SLM 2025 Challenge. The proposed system comprises three key components: 1) a frozen Whisper-large-v3 based speech encoder, leveraging large-scale pretraining to ensure robust acoustic feature extraction; 2) a trainable adaptor module using Linear-ReLU-Linear transformation mechanisms to effectively align speech and text representations; and 3) a frozen Qwen2.5-7B-Instruct large language model (LLM) integrated with trainable LoRA for optimized contextual linguistic decoding. By systematically combining pretrained models with task specific fine-tuning, the system achieved a word/character error rate (WER/CER) of 9.83% across 11 languages in the evaluation set and ranked third place among global participants. 

**Abstract (ZH)**: 本论文介绍了传音言语团队为2025年MLC-SLM挑战赛Track 1开发的新型多语言自动语音识别(ASR)系统的设计与性能。该系统包括三个关键组件：1）基于Whisper-large-v3的冻结语音编码器，利用大规模预训练确保稳健的声学特征提取；2）使用Linear-ReLU-Linear变换机制的可训练适配器模块，以有效地对齐语音和文本表示；以及3）与可训练LoRA集成的冻结Qwen2.5-7B-Instruct大型语言模型（LLM），以优化上下文语言解码。通过系统地结合预训练模型和任务特定微调，该系统在评价集中的11种语言上实现了9.83%的字错误率/字符错误率（WER/CER），并在全球参赛者中排名第三。 

---
# A Chinese Heart Failure Status Speech Database with Universal and Personalised Classification 

**Title (ZH)**: 一个包含普遍分类和个人化分类的中文心力衰竭状况演讲数据库 

**Authors**: Yue Pan, Liwei Liu, Changxin Li, Xinyao Wang, Yili Xia, Hanyue Zhang, Ming Chu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14908)  

**Abstract**: Speech is a cost-effective and non-intrusive data source for identifying acute and chronic heart failure (HF). However, there is a lack of research on whether Chinese syllables contain HF-related information, as observed in other well-studied languages. This study presents the first Chinese speech database of HF patients, featuring paired recordings taken before and after hospitalisation. The findings confirm the effectiveness of the Chinese language in HF detection using both standard 'patient-wise' and personalised 'pair-wise' classification approaches, with the latter serving as an ideal speaker-decoupled baseline for future research. Statistical tests and classification results highlight individual differences as key contributors to inaccuracy. Additionally, an adaptive frequency filter (AFF) is proposed for frequency importance analysis. The data and demonstrations are published at this https URL. 

**Abstract (ZH)**: 基于汉语的急性与慢性心力衰竭识别的语音数据库及方法研究 

---
# Collaborative Filtering using Variational Quantum Hopfield Associative Memory 

**Title (ZH)**: 使用变分量子霍普菲尔德关联记忆的协同过滤 

**Authors**: Amir Kermanshahani, Ebrahim Ardeshir-Larijani, Rakesh Saini, Saif Al-Kuwari  

**Link**: [PDF](https://arxiv.org/pdf/2508.14906)  

**Abstract**: Quantum computing, with its ability to do exponentially faster computation compared to classical systems, has found novel applications in various fields such as machine learning and recommendation systems. Quantum Machine Learning (QML), which integrates quantum computing with machine learning techniques, presents powerful new tools for data processing and pattern recognition. This paper proposes a hybrid recommendation system that combines Quantum Hopfield Associative Memory (QHAM) with deep neural networks to improve the extraction and classification on the MovieLens 1M dataset. User archetypes are clustered into multiple unique groups using the K-Means algorithm and converted into polar patterns through the encoder's activation function. These polar patterns are then integrated into the variational QHAM-based hybrid recommendation model. The system was trained using the MSE loss over 35 epochs in an ideal environment, achieving an ROC value of 0.9795, an accuracy of 0.8841, and an F-1 Score of 0.8786. Trained with the same number of epochs in a noisy environment using a custom Qiskit AER noise model incorporating bit-flip and readout errors with the same probabilities as in real quantum hardware, it achieves an ROC of 0.9177, an accuracy of 0.8013, and an F-1 Score equal to 0.7866, demonstrating consistent performance.
Additionally, we were able to optimize the qubit overhead present in previous QHAM architectures by efficiently updating only one random targeted qubit. This research presents a novel framework that combines variational quantum computing with deep learning, capable of dealing with real-world datasets with comparable performance compared to purely classical counterparts. Additionally, the model can perform similarly well in noisy configurations, showcasing a steady performance and proposing a promising direction for future usage in recommendation systems. 

**Abstract (ZH)**: 量子计算通过相比经典系统实现指数级加速计算的能力，在机器学习和推荐系统等领域找到了新的应用。量子机器学习(QML)将量子计算与机器学习技术相结合，提供强大的新工具用于数据处理和模式识别。本文提出了一种结合量子霍普菲尔德联想记忆(QHAM)和深度神经网络的混合推荐系统，以提高MovieLens 1M数据集中特征提取和分类的性能。用户原型通过K-Means算法聚类并转换为极坐标模式，这些极坐标模式随后被整合到基于变分QHAM的混合推荐模型中。该系统在理想环境中使用MSE损失函数训练35个周期，ROC值为0.9795，准确率为0.8841，F-1分为0.8786。在含有随机比特翻转和读出错误的嘈杂环境中进行相同次数的训练，使用自定义Qiskit AER噪声模型，ROC值为0.9177，准确率为0.8013，F-1分为0.7866，展示了良好的一致性性能。此外，通过高效更新一个随机目标量子位来优化先前QHAM架构中存在的量子位开销。本文提出了一种结合变分量子计算和深度学习的新框架，能够在嘈杂配置下保持良好性能并与纯经典模型具有可比性，为推荐系统未来使用提供了有前景的方向。 

---
# Privacy Preserving Inference of Personalized Content for Out of Matrix Users 

**Title (ZH)**: 矩阵外用户个性化内容的隐私保护推理 

**Authors**: Michael Sun, Tai Vu, Andrew Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14905)  

**Abstract**: Recommender systems for niche and dynamic communities face persistent challenges from data sparsity, cold start users and items, and privacy constraints. Traditional collaborative filtering and content-based approaches underperform in these settings, either requiring invasive user data or failing when preference histories are absent. We present DeepNaniNet, a deep neural recommendation framework that addresses these challenges through an inductive graph-based architecture combining user-item interactions, item-item relations, and rich textual review embeddings derived from BERT. Our design enables cold start recommendations without profile mining, using a novel "content basket" user representation and an autoencoder-based generalization strategy for unseen users. We introduce AnimeULike, a new dataset of 10,000 anime titles and 13,000 users, to evaluate performance in realistic scenarios with high proportions of guest or low-activity users. DeepNaniNet achieves state-of-the-art cold start results on the CiteULike benchmark, matches DropoutNet in user recall without performance degradation for out-of-matrix users, and outperforms Weighted Matrix Factorization (WMF) and DropoutNet on AnimeULike warm start by up to 7x and 1.5x in Recall@100, respectively. Our findings demonstrate that DeepNaniNet delivers high-quality, privacy-preserving recommendations in data-sparse, cold start-heavy environments while effectively integrating heterogeneous content sources. 

**Abstract (ZH)**: 面向稀疏数据、冷启动用户和隐私约束的 niche 和动态社区推荐系统面临持续挑战。传统协同过滤和基于内容的方法在此情境下表现不佳，要么需要侵入性用户数据，要么在缺乏偏好历史时失效。我们提出 DeepNaniNet，一种通过结合用户-物品交互、物品-物品关系以及源自 BERT 的丰富文本评论嵌入的归纳图结构，解决这些挑战的深度神经推荐框架。我们的设计通过一种新颖的“内容篮子”用户表示和基于自编码器的一般化策略，无需挖掘用户资料即可实现冷启动推荐。我们引入 AnimeULike，一个包含 10,000 部动画片和 13,000 名用户的新型数据集，用于评估高比例临时用户或低活跃用户的现实场景性能。DeepNaniNet 在 CiteULike 基准中实现最先进的冷启动结果，与 DropoutNet 在离矩阵用户召回上表现相当且无性能下降，并且在 AnimeULike 的温暖启动场景中，与 Weighted Matrix Factorization (WMF) 和 DropoutNet 相比，Recall@100 分别高 7 倍和 1.5 倍。我们的研究结果表明，DeepNaniNet 能在稀疏数据、高比例冷启动用户环境中提供高质量且保护隐私的推荐，同时有效地整合了异构内容源。 

---
# Efficient Switchable Safety Control in LLMs via Magic-Token-Guided Co-Training 

**Title (ZH)**: 通过魔法令牌引导的协同训练实现LLMs中的高效可切换安全性控制 

**Authors**: Jianfeng Si, Lin Sun, Zhewen Tan, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14904)  

**Abstract**: Current methods for content safety in Large Language Models (LLMs), such as Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), often rely on multi-stage training pipelines and lack fine-grained, post-deployment controllability. To address these limitations, we propose a unified co-training framework that efficiently integrates multiple safety behaviors: positive (lawful/prosocial), negative (unfiltered/risk-prone) and rejective (refusal-oriented/conservative) within a single SFT stage. Notably, each behavior is dynamically activated via a simple system-level instruction, or magic token, enabling stealthy and efficient behavioral switching at inference time. This flexibility supports diverse deployment scenarios, such as positive for safe user interaction, negative for internal red-teaming, and rejective for context-aware refusals triggered by upstream moderation signals. This co-training strategy induces a distinct Safety Alignment Margin in the output space, characterized by well-separated response distributions corresponding to each safety mode. The existence of this margin provides empirical evidence for the model's safety robustness and enables unprecedented fine-grained control. Experiments show that our method matches the safety alignment quality of SFT+DPO, with our 8B model notably surpassing DeepSeek-R1 (671B) in safety performance, while significantly reducing both training complexity and deployment costs. This work presents a scalable, efficient, and highly controllable solution for LLM content safety. 

**Abstract (ZH)**: 当前大型语言模型内容安全方法，如监督微调(SFT)和基于人类反馈的强化学习(RLHF)，往往依赖多阶段训练管道并在部署后缺乏精细控制。为解决这些限制，我们提出了一种统一的联合训练框架，在单个SFT阶段中高效整合多种安全行为：正面（合法/亲社会）、负面（未经筛选/风险倾向）和拒绝（拒绝导向/保守）。值得注意的是，每种行为可通过简单的系统级指令或魔法标记动态激活，从而在推理时实现隐蔽且高效的切换行为。这种灵活性支持多种部署场景，例如正面用于安全用户交互、负面用于内部红队测试、拒绝用于根据上游审核信号触发的内容感知拒绝。这种联合训练策略在输出空间中诱导出一种独特的安全对齐边际，其特征是对应每种安全模式且区分良好的响应分布。边际的存在为模型的安全鲁棒性提供了实证证据，并使精细控制成为可能。实验显示，我们的方法在安全对齐质量上与SFT+DPO相当，8B模型在安全性能上显著超越DeepSeek-R1（671B），同时大幅降低了训练复杂性和部署成本。本工作提出了一种可扩展、高效且高度可控的大规模语言模型内容安全解决方案。 

---
# Accelerating GenAI Workloads by Enabling RISC-V Microkernel Support in IREE 

**Title (ZH)**: 在IREE中启用RISC-V微内核支持以加速GenAI工作负载 

**Authors**: Adeel Ahmad, Ahmad Tameem Kamal, Nouman Amir, Bilal Zafar, Saad Bin Nasir  

**Link**: [PDF](https://arxiv.org/pdf/2508.14899)  

**Abstract**: This project enables RISC-V microkernel support in IREE, an MLIR-based machine learning compiler and runtime. The approach begins by enabling the lowering of MLIR linalg dialect contraction ops to linalg.mmt4d op for the RISC-V64 target within the IREE pass pipeline, followed by the development of optimized microkernels for RISC-V. The performance gains are compared with upstream IREE and this http URL for the Llama-3.2-1B-Instruct model. 

**Abstract (ZH)**: 这个项目在IREE中为RISC-V微内核支持MLIR机器学习编译器和运行时提供了支持。该方法首先在IREE passes管道中启用将MLIR linalg dialect收缩操作降低为RISC-V64目标的linalg.mmt4d op，随后开发针对RISC-V的优化微内核。性能增益与上游IREE进行比较，并与Llama-3.2-1B-Instruct模型的此链接进行对比。 

---
# The Impact of Image Resolution on Face Detection: A Comparative Analysis of MTCNN, YOLOv XI and YOLOv XII models 

**Title (ZH)**: 图像分辨率对面部检测的影响：MTCNN、YOLOv XI和YOLOv XII模型的比较分析 

**Authors**: Ahmet Can Ömercikoğlu, Mustafa Mansur Yönügül, Pakize Erdoğmuş  

**Link**: [PDF](https://arxiv.org/pdf/2507.23341)  

**Abstract**: Face detection is a crucial component in many AI-driven applications such as surveillance, biometric authentication, and human-computer interaction. However, real-world conditions like low-resolution imagery present significant challenges that degrade detection performance. In this study, we systematically investigate the impact of input resolution on the accuracy and robustness of three prominent deep learning-based face detectors: YOLOv11, YOLOv12, and MTCNN. Using the WIDER FACE dataset, we conduct extensive evaluations across multiple image resolutions (160x160, 320x320, and 640x640) and assess each model's performance using metrics such as precision, recall, mAP50, mAP50-95, and inference time. Results indicate that YOLOv11 outperforms YOLOv12 and MTCNN in terms of detection accuracy, especially at higher resolutions, while YOLOv12 exhibits slightly better recall. MTCNN, although competitive in landmark localization, lags in real-time inference speed. Our findings provide actionable insights for selecting resolution-aware face detection models suitable for varying operational constraints. 

**Abstract (ZH)**: 基于输入分辨率的深度学习面部检测模型性能研究：以YOLOv11、YOLOv12和MTCNN为例 

---
# SVM/SVR Kernels as Quantum Propagators 

**Title (ZH)**: SVM/SVR 核函数作为量子传播器 

**Authors**: Nan-Hong Kuo, Renata Wong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11153)  

**Abstract**: We establish a mathematical equivalence between Support Vector Machine (SVM) kernel functions and quantum propagators represented by time-dependent Green's functions, which has remained largely unexplored.
We demonstrate that many common SVM kernels correspond naturally to Green's functions via operator inversion theory. The sigmoid kernel does not always satisfy Mercer's theorem, and therefore the corresponding Green's function may also fail to perform optimally.
We further introduce a Kernel Polynomial Method (KPM) for designing customized kernels that align with Green's functions.
Our numerical experiments confirm that employing positive-semidefinite kernels that correspond to Green's functions significantly improves predictive accuracy of SVM models in physical systems. 

**Abstract (ZH)**: 我们建立了支持向量机（SVM）核函数与由时间依赖格林函数表示的量子传播子之间的数学等价性，这一领域尚未被充分探索。 

---
