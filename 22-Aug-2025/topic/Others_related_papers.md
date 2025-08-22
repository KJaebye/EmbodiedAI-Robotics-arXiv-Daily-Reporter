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
# Adapting A Vector-Symbolic Memory for Lisp ACT-R 

**Title (ZH)**: 适配矢量符号记忆的Lisp ACT-R 

**Authors**: Meera Ray, Christopher L. Dancy  

**Link**: [PDF](https://arxiv.org/pdf/2508.15630)  

**Abstract**: Holographic Declarative Memory (HDM) is a vector-symbolic alternative to ACT-R's Declarative Memory (DM) system that can bring advantages such as scalability and architecturally defined similarity between DM chunks. We adapted HDM to work with the most comprehensive and widely-used implementation of ACT-R (Lisp ACT-R) so extant ACT-R models designed with DM can be run with HDM without major changes. With this adaptation of HDM, we have developed vector-based versions of common ACT-R functions, set up a text processing pipeline to add the contents of large documents to ACT-R memory, and most significantly created a useful and novel mechanism to retrieve an entire chunk of memory based on a request using only vector representations of tokens. Preliminary results indicate that we can maintain vector-symbolic advantages of HDM (e.g., chunk recall without storing the actual chunk and other advantages with scaling) while also extending it so that previous ACT-R models may work with the system with little (or potentially no) modifications within the actual procedural and declarative memory portions of a model. As a part of iterative improvement of this newly translated holographic declarative memory module, we will continue to explore better time-context representations for vectors to improve the module's ability to reconstruct chunks during recall. To more fully test this translated HDM module, we also plan to develop decision-making models that use instance-based learning (IBL) theory, which is a useful application of HDM given the advantages of the system. 

**Abstract (ZH)**: 全息声明记忆（HDM）：ACT-R声明记忆（DM）系统的向量符号替代方案及其适应性改进 

---
# A Dynamical Systems Framework for Reinforcement Learning Safety and Robustness Verification 

**Title (ZH)**: 动态系统框架下的强化学习安全性与鲁棒性验证 

**Authors**: Ahmed Nasir, Abdelhafid Zenati  

**Link**: [PDF](https://arxiv.org/pdf/2508.15588)  

**Abstract**: The application of reinforcement learning to safety-critical systems is limited by the lack of formal methods for verifying the robustness and safety of learned policies. This paper introduces a novel framework that addresses this gap by analyzing the combination of an RL agent and its environment as a discrete-time autonomous dynamical system. By leveraging tools from dynamical systems theory, specifically the Finite-Time Lyapunov Exponent (FTLE), we identify and visualize Lagrangian Coherent Structures (LCS) that act as the hidden "skeleton" governing the system's behavior. We demonstrate that repelling LCS function as safety barriers around unsafe regions, while attracting LCS reveal the system's convergence properties and potential failure modes, such as unintended "trap" states. To move beyond qualitative visualization, we introduce a suite of quantitative metrics, Mean Boundary Repulsion (MBR), Aggregated Spurious Attractor Strength (ASAS), and Temporally-Aware Spurious Attractor Strength (TASAS), to formally measure a policy's safety margin and robustness. We further provide a method for deriving local stability guarantees and extend the analysis to handle model uncertainty. Through experiments in both discrete and continuous control environments, we show that this framework provides a comprehensive and interpretable assessment of policy behavior, successfully identifying critical flaws in policies that appear successful based on reward alone. 

**Abstract (ZH)**: 安全关键系统中强化学习的应用受限于缺乏验证学习策略稳健性和安全性的形式化方法。本文提出了一种新的框架，通过将RL代理及其环境视为离散时间自主动力系统来解决这一问题。借助动力系统理论工具，特别是有限时间李雅普un夫指数（FTLE），我们识别并可视化了拉格朗日协轭结构（LCS），它们作为隐藏的“骨架”指导系统的动态。我们证明排斥的LCS起到安全屏障的作用，包围不安全区域；吸引的LCS揭示了系统的收敛特性以及潜在的故障模式，如意外的“陷阱”状态。为进一步超越定性可视化，我们引入了一系列定量指标，包括平均边界排斥（MBR）、聚合虚假吸引子强度（ASAS）和时间感知虚假吸引子强度（TASAS），以正式衡量策略的安全余量和稳健性。我们还提供了一种方法来推导局部稳定性保证，并扩展分析以处理模型不确定性。通过在离散和连续控制环境下进行的实验，我们展示了该框架提供了全面且可解释的策略行为评估，并成功识别出仅基于奖励表现良好的成功策略中的关键缺陷。 

---
# Planning with Minimal Disruption 

**Title (ZH)**: 最小干扰规划 

**Authors**: Alberto Pozanco, Marianela Morales, Daniel Borrajo, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2508.15358)  

**Abstract**: In many planning applications, we might be interested in finding plans that minimally modify the initial state to achieve the goals. We refer to this concept as plan disruption. In this paper, we formally introduce it, and define various planning-based compilations that aim to jointly optimize both the sum of action costs and plan disruption. Experimental results in different benchmarks show that the reformulated task can be effectively solved in practice to generate plans that balance both objectives. 

**Abstract (ZH)**: 在许多规划应用中，我们可能对最小修改初始状态以实现目标的规划方案感兴趣。我们将这一概念称为规划中断。在本文中，我们正式引入了这一概念，并定义了旨在同时优化动作成本总和和规划中断的多种基于规划的编译方法。不同基准的实验结果表明，重新定义后的任务可以在实践中有效求解，生成同时平衡两个目标的规划方案。 

---
# Computational Intelligence based Land-use Allocation Approaches for Mixed Use Areas 

**Title (ZH)**: 基于计算智能的混合用地区域土地利用分配方法 

**Authors**: Sabab Aosaf, Muhammad Ali Nayeem, Afsana Haque, M Sohel Rahmana  

**Link**: [PDF](https://arxiv.org/pdf/2508.15240)  

**Abstract**: Urban land-use allocation represents a complex multi-objective optimization problem critical for sustainable urban development policy. This paper presents novel computational intelligence approaches for optimizing land-use allocation in mixed-use areas, addressing inherent trade-offs between land-use compatibility and economic objectives. We develop multiple optimization algorithms, including custom variants integrating differential evolution with multi-objective genetic algorithms. Key contributions include: (1) CR+DES algorithm leveraging scaled difference vectors for enhanced exploration, (2) systematic constraint relaxation strategy improving solution quality while maintaining feasibility, and (3) statistical validation using Kruskal-Wallis tests with compact letter displays. Applied to a real-world case study with 1,290 plots, CR+DES achieves 3.16\% improvement in land-use compatibility compared to state-of-the-art methods, while MSBX+MO excels in price optimization with 3.3\% improvement. Statistical analysis confirms algorithms incorporating difference vectors significantly outperform traditional approaches across multiple metrics. The constraint relaxation technique enables broader solution space exploration while maintaining practical constraints. These findings provide urban planners and policymakers with evidence-based computational tools for balancing competing objectives in land-use allocation, supporting more effective urban development policies in rapidly urbanizing regions. 

**Abstract (ZH)**: 城市土地利用分配代表了一个关键的多目标优化问题，对于可持续城市发展政策至关重要。本文提出了一种新的计算智能方法，用于优化混合用途区的土地利用分配，以解决土地利用兼容性和经济目标之间的固有权衡。我们开发了多个优化算法，包括结合差分进化与多目标遗传算法的自定义变体。主要贡献包括：（1）CR+DES算法利用比例差异向量增强探索能力，（2）系统性的约束松弛策略提高解的质量的同时保持可行性，（3）使用克鲁斯卡尔-沃allis检验和紧嵌字母显示进行统计验证。应用于一个包含1290个地块的实际案例研究中，CR+DES在土地利用兼容性方面比最先进的方法提高了3.16%，而MSBX+MO在价格优化方面表现出色，提高了3.3%。统计分析证实，包含差异向量的算法在多个指标上显著优于传统方法。约束松弛技术能够在保持实用约束的同时扩展解的空间。这些发现为城市规划者和政策制定者提供了基于证据的计算工具，以在土地利用分配中平衡竞争目标，支持快速发展地区更具成效的城市发展政策。 

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
# Argumentation for Explainable Workforce Optimisation (with Appendix) 

**Title (ZH)**: 可解释的工作force优化的论证（附录附录） 

**Authors**: Jennifer Leigh, Dimitrios Letsios, Alessandro Mella, Lucio Machetti, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2508.15118)  

**Abstract**: Workforce management is a complex problem optimising the makespan and travel distance required for a team of operators to complete a set of jobs, using a set of instruments. A crucial challenge in workforce management is accommodating changes at execution time so that explanations are provided to all stakeholders involved. Here, we show that, by understanding workforce management as abstract argumentation in an industrial application, we can accommodate change and obtain faithful explanations. We show, with a user study, that our tool and explanations lead to faster and more accurate problem solving than conventional solutions by hand. 

**Abstract (ZH)**: 工业应用中通过抽象论证实现工人群体管理中的灵活变更与忠实解释 

---
# Demonstrating Onboard Inference for Earth Science Applications with Spectral Analysis Algorithms and Deep Learning 

**Title (ZH)**: 基于光谱分析算法和深度学习的机载推断在地球科学应用中的演示 

**Authors**: Itai Zilberstein, Alberto Candela, Steve Chien, David Rijlaarsdam, Tom Hendrix, Leonie Buckley, Aubrey Dunne  

**Link**: [PDF](https://arxiv.org/pdf/2508.15053)  

**Abstract**: In partnership with Ubotica Technologies, the Jet Propulsion Laboratory is demonstrating state-of-the-art data analysis onboard CogniSAT-6/HAMMER (CS-6). CS-6 is a satellite with a visible and near infrared range hyperspectral instrument and neural network acceleration hardware. Performing data analysis at the edge (e.g. onboard) can enable new Earth science measurements and responses. We will demonstrate data analysis and inference onboard CS-6 for numerous applications using deep learning and spectral analysis algorithms. 

**Abstract (ZH)**: 在Ubotica Technologies的合作下，喷气推进实验室正在通过CogniSAT-6/HAMMER (CS-6) 显示先进的数据处理技术。CS-6是一颗配备可见光和近红外波段高光谱仪器及神经网络加速硬件的卫星。在边缘（例如，卫星上）执行数据处理可以实现新的地球科学研究和响应。我们将使用深度学习和光谱分析算法在CS-6上进行数据处理和推断演示，用于多种应用。 

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
# Discovering Hidden Algebraic Structures via Transformers with Rank-Aware Beam GRPO 

**Title (ZH)**: 基于秩意识束GRPO的变压器发现隐藏代数结构 

**Authors**: Jaeha Lee, Gio Huh, Ning Su, Tony Yue YU  

**Link**: [PDF](https://arxiv.org/pdf/2508.15766)  

**Abstract**: Recent efforts have extended the capabilities of transformers in logical reasoning and symbolic computations. In this work, we investigate their capacity for non-linear latent pattern discovery in the context of functional decomposition, focusing on the challenging algebraic task of multivariate polynomial decomposition. This problem, with widespread applications in science and engineering, is proved to be NP-hard, and demands both precision and insight. Our contributions are threefold: First, we develop a synthetic data generation pipeline providing fine-grained control over problem complexity. Second, we train transformer models via supervised learning and evaluate them across four key dimensions involving scaling behavior and generalizability. Third, we propose Beam Grouped Relative Policy Optimization (BGRPO), a rank-aware reinforcement learning method suitable for hard algebraic problems. Finetuning with BGRPO improves accuracy while reducing beam width by up to half, resulting in approximately 75% lower inference compute. Additionally, our model demonstrates competitive performance in polynomial simplification, outperforming Mathematica in various cases. 

**Abstract (ZH)**: 最近的努力扩展了变换器在逻辑推理和符号计算中的能力。在此项工作中，我们调查了它们在函数分解背景下非线性潜在模式发现的能力，重点关注多项式分解这一具有挑战性的代数任务。该问题在科学和工程中有着广泛的应用，并已被证明是NP-hard的，既要求精确性又要求洞察力。我们的贡献体现在三个方面：首先，我们开发了一种合成数据生成管道，提供了对问题复杂性的精细控制。其次，我们通过监督学习训练变换器模型，并从四个关键维度对其进行评估，涉及缩放行为和泛化能力。第三，我们提出了一种基于排名的强化学习方法——束分组相对策略优化（BGRPO），适用于难以求解的代数问题。使用BGRPO微调可以提高准确性，并将束宽度降低至一半以下，计算推理量减少约75%。此外，我们的模型在多项式简化方面表现出竞争力，在某些情况下优于Mathematica。 

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
# LoUQAL: Low-fidelity informed Uncertainty Quantification for Active Learning in the chemical configuration space 

**Title (ZH)**: Low-fidelity Informed Uncertainty Quantification for Active Learning in the Chemical Configuration Space 

**Authors**: Vivin Vinod, Peter Zaspel  

**Link**: [PDF](https://arxiv.org/pdf/2508.15577)  

**Abstract**: Uncertainty quantification is an important scheme in active learning techniques, including applications in predicting quantum chemical properties. In quantum chemical calculations, there exists the notion of a fidelity, a less accurate computation is accessible at a cheaper computational cost. This work proposes a novel low-fidelity informed uncertainty quantification for active learning with applications in predicting diverse quantum chemical properties such as excitation energies and \textit{ab initio} potential energy surfaces. Computational experiments are carried out in order to assess the proposed method with results demonstrating that models trained with the novel method outperform alternatives in terms of empirical error and number of iterations required. The effect of the choice of fidelity is also studied to perform a thorough benchmark. 

**Abstract (ZH)**: 不确定性量化是主动学习技术中一种重要的方案，包括在预测量子化学性质中的应用。在量子化学计算中，存在忠实度的概念，较低忠实度的计算在计算成本较低的情况下可以获得。本工作提出了一种新颖的低忠实度导向不确定性量化方法，应用于预测多种量子化学性质，如激发能量和从头算势能面。通过计算实验评估提出的方法，结果显示使用新颖方法训练的模型在经验误差和所需迭代次数方面优于替代方法。还研究了忠实度选择的影响，以进行全面基准测试。 

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
# An Empirical Study of Knowledge Distillation for Code Understanding Tasks 

**Title (ZH)**: 代码理解任务中的知识精炼实证研究 

**Authors**: Ruiqi Wang, Zezhou Yang, Cuiyun Gao, Xin Xia, Qing Liao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15423)  

**Abstract**: Pre-trained language models (PLMs) have emerged as powerful tools for code understanding. However, deploying these PLMs in large-scale applications faces practical challenges due to their computational intensity and inference latency. Knowledge distillation (KD), a promising model compression and acceleration technique, addresses these limitations by transferring knowledge from large teacher models to compact student models, enabling efficient inference while preserving most of the teacher models' capabilities. While this technique has shown remarkable success in natural language processing and computer vision domains, its potential for code understanding tasks remains largely underexplored.
In this paper, we systematically investigate the effectiveness and usage of KD in code understanding tasks. Our study encompasses two popular types of KD methods, i.e., logit-based and feature-based KD methods, experimenting across eight student models and two teacher PLMs from different domains on three downstream tasks. The experimental results indicate that KD consistently offers notable performance boosts across student models with different sizes compared with standard fine-tuning. Notably, code-specific PLM demonstrates better effectiveness as the teacher model. Among all KD methods, the latest feature-based KD methods exhibit superior performance, enabling student models to retain up to 98% teacher performance with merely 5% parameters. Regarding student architecture, our experiments reveal that similarity with teacher architecture does not necessarily lead to better performance. We further discuss the efficiency and behaviors in the KD process and inference, summarize the implications of findings, and identify promising future directions. 

**Abstract (ZH)**: 预训练语言模型（PLMs）已成为代码理解的强大工具。然而，在大规模应用中部署这些PLMs面临实际挑战，因为它们的计算强度和推理延迟较高。知识蒸馏（KD），一种有前途的模型压缩和加速技术，通过将大型教师模型的知识转移到紧凑的学生模型中，解决了这些限制，实现了高效推理同时保持大多数教师模型的能力。尽管这一技术在自然语言处理和计算机视觉领域取得了显著成功，但其在代码理解任务中的潜力尚待充分探索。

在本文中，我们系统地探讨了KD在代码理解任务中的有效性和使用方法。我们的研究涵盖了两种流行的KD方法，即基于logit和基于特征的KD方法，在八个学生模型和两种来自不同领域的教师PLMs上对三个下游任务进行了实验。实验结果表明，与标准微调相比，KD在不同规模的学生模型中提供了显著的性能提升。值得注意的是，代码专用的PLM作为教师模型时效果更佳。在所有KD方法中，最新的基于特征的KD方法表现出最佳性能，使得学生模型仅使用5%的参数即可保留高达98%的教师模型性能。关于学生架构，我们的实验表明，与教师架构的相似性并不一定导致更好的性能。我们进一步讨论了KD过程和推理中的效率与行为，总结了研究发现的含义，并指出了有前景的未来方向。 

---
# Bridging Generalization and Personalization in Wearable Human Activity Recognition via On-Device Few-Shot Learning 

**Title (ZH)**: 基于设备端少样本学习实现可穿戴人体活动识别的一般化与个性化桥梁构建 

**Authors**: Pixi Kang, Julian Moosmann, Mengxi Liu, Bo Zhou, Michele Magno, Paul Lukowicz, Sizhen Bian  

**Link**: [PDF](https://arxiv.org/pdf/2508.15413)  

**Abstract**: Human Activity Recognition (HAR) using wearable devices has advanced significantly in recent years, yet its generalization remains limited when models are deployed to new users. This degradation in performance is primarily due to user-induced concept drift (UICD), highlighting the importance of efficient personalization. In this paper, we present a hybrid framework that first generalizes across users and then rapidly adapts to individual users using few-shot learning directly on-device. By updating only the classifier layer with user-specific data, our method achieves robust personalization with minimal computational and memory overhead. We implement this framework on the energy-efficient RISC-V-based GAP9 microcontroller and validate it across three diverse HAR scenarios: RecGym, QVAR-Gesture, and Ultrasound-Gesture. Post-deployment adaptation yields consistent accuracy improvements of 3.73\%, 17.38\%, and 3.70\% respectively. These results confirm that fast, lightweight, and effective personalization is feasible on embedded platforms, paving the way for scalable and user-aware HAR systems in the wild \footnote{this https URL}. 

**Abstract (ZH)**: 使用可穿戴设备进行人类活动识别（HAR）近年来取得了显著进展，但在部署到新用户时其泛化能力仍然有限。这种性能下降主要是由于用户诱导的概念漂移（UICD），突显了高效个性化的重要性。本文提出了一种混合框架，首先在用户之间进行泛化，然后利用直接在设备上进行的少-shot学习快速适应个别用户。通过仅使用用户特定数据更新分类器层，我们的方法实现了稳健的个性化，同时最小化了计算和内存开销。该框架在能效型RISC-V基础的GAP9微控制器上实现，并在三种不同的HAR场景下进行了验证：RecGym、QVAR-Gesture和Ultrasound-Gesture。部署后的适应性调整分别提高了3.73%、17.38%和3.70%的准确性。这些结果证实，在嵌入式平台上实现快速、轻量级和有效的个性化是可行的，为野生环境下的可扩展和用户意识强的HAR系统铺平了道路。 

---
# Hybrid Least Squares/Gradient Descent Methods for DeepONets 

**Title (ZH)**: 混合最小二乘/梯度下降方法用于DeepONets 

**Authors**: Jun Choi, Chang-Ock Lee, Minam Moon  

**Link**: [PDF](https://arxiv.org/pdf/2508.15394)  

**Abstract**: We propose an efficient hybrid least squares/gradient descent method to accelerate DeepONet training. Since the output of DeepONet can be viewed as linear with respect to the last layer parameters of the branch network, these parameters can be optimized using a least squares (LS) solve, and the remaining hidden layer parameters are updated by means of gradient descent form. However, building the LS system for all possible combinations of branch and trunk inputs yields a prohibitively large linear problem that is infeasible to solve directly. To address this issue, our method decomposes the large LS system into two smaller, more manageable subproblems $\unicode{x2014}$ one for the branch network and one for the trunk network $\unicode{x2014}$ and solves them separately. This method is generalized to a broader type of $L^2$ loss with a regularization term for the last layer parameters, including the case of unsupervised learning with physics-informed loss. 

**Abstract (ZH)**: 我们提出了一种高效混合最小二乘/梯度下降方法来加速DeepONet训练。 

---
# EvoFormer: Learning Dynamic Graph-Level Representations with Structural and Temporal Bias Correction 

**Title (ZH)**: EvoFormer：学习具有结构和时间偏置修正的动力学图级表示 

**Authors**: Haodi Zhong, Liuxin Zou, Di Wang, Bo Wang, Zhenxing Niu, Quan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15378)  

**Abstract**: Dynamic graph-level embedding aims to capture structural evolution in networks, which is essential for modeling real-world scenarios. However, existing methods face two critical yet under-explored issues: Structural Visit Bias, where random walk sampling disproportionately emphasizes high-degree nodes, leading to redundant and noisy structural representations; and Abrupt Evolution Blindness, the failure to effectively detect sudden structural changes due to rigid or overly simplistic temporal modeling strategies, resulting in inconsistent temporal embeddings. To overcome these challenges, we propose EvoFormer, an evolution-aware Transformer framework tailored for dynamic graph-level representation learning. To mitigate Structural Visit Bias, EvoFormer introduces a Structure-Aware Transformer Module that incorporates positional encoding based on node structural roles, allowing the model to globally differentiate and accurately represent node structures. To overcome Abrupt Evolution Blindness, EvoFormer employs an Evolution-Sensitive Temporal Module, which explicitly models temporal evolution through a sequential three-step strategy: (I) Random Walk Timestamp Classification, generating initial timestamp-aware graph-level embeddings; (II) Graph-Level Temporal Segmentation, partitioning the graph stream into segments reflecting structurally coherent periods; and (III) Segment-Aware Temporal Self-Attention combined with an Edge Evolution Prediction task, enabling the model to precisely capture segment boundaries and perceive structural evolution trends, effectively adapting to rapid temporal shifts. Extensive evaluations on five benchmark datasets confirm that EvoFormer achieves state-of-the-art performance in graph similarity ranking, temporal anomaly detection, and temporal segmentation tasks, validating its effectiveness in correcting structural and temporal biases. 

**Abstract (ZH)**: EvoFormer：一种关注演化的Transformer框架用于动态图级表示学习 

---
# Way to Build Native AI-driven 6G Air Interface: Principles, Roadmap, and Outlook 

**Title (ZH)**: 基于原生AI驱动的6G空中接口构建方法：原理、路线图与展望 

**Authors**: Ping Zhang, Kai Niu, Yiming Liu, Zijian Liang, Nan Ma, Xiaodong Xu, Wenjun Xu, Mengying Sun, Yinqiu Liu, Xiaoyun Wang, Ruichen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15277)  

**Abstract**: Artificial intelligence (AI) is expected to serve as a foundational capability across the entire lifecycle of 6G networks, spanning design, deployment, and operation. This article proposes a native AI-driven air interface architecture built around two core characteristics: compression and adaptation. On one hand, compression enables the system to understand and extract essential semantic information from the source data, focusing on task relevance rather than symbol-level accuracy. On the other hand, adaptation allows the air interface to dynamically transmit semantic information across diverse tasks, data types, and channel conditions, ensuring scalability and robustness. This article first introduces the native AI-driven air interface architecture, then discusses representative enabling methodologies, followed by a case study on semantic communication in 6G non-terrestrial networks. Finally, it presents a forward-looking discussion on the future of native AI in 6G, outlining key challenges and research opportunities. 

**Abstract (ZH)**: 人工智能驱动的6G网络原生空中接口架构：基于压缩与适应的核心特性 

---
# Robust and Efficient Quantum Reservoir Computing with Discrete Time Crystal 

**Title (ZH)**: 离散时间晶体中稳健而高效的量子蓄水库计算 

**Authors**: Da Zhang, Xin Li, Yibin Guo, Haifeng Yu, Yirong Jin, Zhang-Qi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.15230)  

**Abstract**: The rapid development of machine learning and quantum computing has placed quantum machine learning at the forefront of research. However, existing quantum machine learning algorithms based on quantum variational algorithms face challenges in trainability and noise robustness. In order to address these challenges, we introduce a gradient-free, noise-robust quantum reservoir computing algorithm that harnesses discrete time crystal dynamics as a reservoir. We first calibrate the memory, nonlinear, and information scrambling capacities of the quantum reservoir, revealing their correlation with dynamical phases and non-equilibrium phase transitions. We then apply the algorithm to the binary classification task and establish a comparative quantum kernel advantage. For ten-class classification, both noisy simulations and experimental results on superconducting quantum processors match ideal simulations, demonstrating the enhanced accuracy with increasing system size and confirming the topological noise robustness. Our work presents the first experimental demonstration of quantum reservoir computing for image classification based on digital quantum simulation. It establishes the correlation between quantum many-body non-equilibrium phase transitions and quantum machine learning performance, providing new design principles for quantum reservoir computing and broader quantum machine learning algorithms in the NISQ era. 

**Abstract (ZH)**: 机器学习和量子计算的快速发展将量子机器学习推向了研究前沿。然而，现有的基于量子变分算法的量子机器学习算法在可训练性和抗噪性方面面临挑战。为了应对这些挑战，我们引入了一种采用离散时间晶体动力学作为蓄水池的无梯度、抗噪量子蓄水池计算算法。我们首先校准了量子蓄水池的记忆、非线性和信息混杂能力，并揭示了这些能力与动力学相位和非平衡相变之间的关系。然后，我们将该算法应用于二元分类任务，并建立了量子核优势。对于十类分类任务，嘈杂的模拟和超导量子处理器上的实验结果均与理想模拟匹配，展示了系统规模增大时的增强精度，并证实了拓扑抗噪性。我们的工作首次基于数字量子模拟实现了基于量子蓄水池计算的图像分类的实验演示，建立了量子多体非平衡相变与量子机器学习性能之间的关联，为量子蓄水池计算和更广泛的量子机器学习算法在NISQ时代的设计提供了新的原理。 

---
# Locally Pareto-Optimal Interpretations for Black-Box Machine Learning Models 

**Title (ZH)**: 局部帕累托最优解释：黑盒机器学习模型的解释方法 

**Authors**: Aniruddha Joshi, Supratik Chakraborty, S Akshay, Shetal Shah, Hazem Torfah, Sanjit Seshia  

**Link**: [PDF](https://arxiv.org/pdf/2508.15220)  

**Abstract**: Creating meaningful interpretations for black-box machine learning models involves balancing two often conflicting objectives: accuracy and explainability. Exploring the trade-off between these objectives is essential for developing trustworthy interpretations. While many techniques for multi-objective interpretation synthesis have been developed, they typically lack formal guarantees on the Pareto-optimality of the results. Methods that do provide such guarantees, on the other hand, often face severe scalability limitations when exploring the Pareto-optimal space. To address this, we develop a framework based on local optimality guarantees that enables more scalable synthesis of interpretations. Specifically, we consider the problem of synthesizing a set of Pareto-optimal interpretations with local optimality guarantees, within the immediate neighborhood of each solution. Our approach begins with a multi-objective learning or search technique, such as Multi-Objective Monte Carlo Tree Search, to generate a best-effort set of Pareto-optimal candidates with respect to accuracy and explainability. We then verify local optimality for each candidate as a Boolean satisfiability problem, which we solve using a SAT solver. We demonstrate the efficacy of our approach on a set of benchmarks, comparing it against previous methods for exploring the Pareto-optimal front of interpretations. In particular, we show that our approach yields interpretations that closely match those synthesized by methods offering global guarantees. 

**Abstract (ZH)**: 创建黑箱机器学习模型的有意义解释涉及平衡准确性和可解释性这两个 Often 相互冲突的目标。探索这两个目标之间的trade-off 是开发可信赖解释的关键。尽管已经开发出许多多目标解释综合技术，但它们通常缺乏关于结果 Pareto-最优点的正式保证。另一方面，能够提供此类保证的方法在探索 Pareto-最优点空间时往往会面临严重的可扩展性限制。为此，我们开发了一个基于局部最优点保证的框架，以实现更可扩展的解释综合。具体地，我们考虑在每个解的局部邻域内合成具有局部最优点保证的 Pareto-最优解释集的问题。我们的方法首先使用多目标学习或搜索技术，例如多目标蒙特卡洛树搜索，生成关于准确性和可解释性的 Pareto-最优候选集。然后，我们通过布尔可满足性问题验证每个候选的局部最优点，并使用 SAT 求解器求解该问题。我们在一组基准上展示了我们方法的有效性，将其与先前探索解释 Pareto-前沿的方法进行了比较。特别地，我们展示了我们的方法产生的解释与提供全局保证的方法合成的解释高度一致。 

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
# A Systematic Survey of Model Extraction Attacks and Defenses: State-of-the-Art and Perspectives 

**Title (ZH)**: 模型提取攻击与防御的系统性综述：现状与展望 

**Authors**: Kaixiang Zhao, Lincan Li, Kaize Ding, Neil Zhenqiang Gong, Yue Zhao, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.15031)  

**Abstract**: Machine learning (ML) models have significantly grown in complexity and utility, driving advances across multiple domains. However, substantial computational resources and specialized expertise have historically restricted their wide adoption. Machine-Learning-as-a-Service (MLaaS) platforms have addressed these barriers by providing scalable, convenient, and affordable access to sophisticated ML models through user-friendly APIs. While this accessibility promotes widespread use of advanced ML capabilities, it also introduces vulnerabilities exploited through Model Extraction Attacks (MEAs). Recent studies have demonstrated that adversaries can systematically replicate a target model's functionality by interacting with publicly exposed interfaces, posing threats to intellectual property, privacy, and system security. In this paper, we offer a comprehensive survey of MEAs and corresponding defense strategies. We propose a novel taxonomy that classifies MEAs according to attack mechanisms, defense approaches, and computing environments. Our analysis covers various attack techniques, evaluates their effectiveness, and highlights challenges faced by existing defenses, particularly the critical trade-off between preserving model utility and ensuring security. We further assess MEAs within different computing paradigms and discuss their technical, ethical, legal, and societal implications, along with promising directions for future research. This systematic survey aims to serve as a valuable reference for researchers, practitioners, and policymakers engaged in AI security and privacy. Additionally, we maintain an online repository continuously updated with related literature at this https URL. 

**Abstract (ZH)**: 机器学习模型显著增长在复杂性和实用性方面，推动了多个领域的进步。然而，历史上传统上，大量计算资源和专门的技术知识限制了它们的广泛采用。作为一种解决方案，机器学习即服务（MLaaS）平台通过用户友好的API提供了可扩展、便捷且经济高效的高级机器学习模型访问途径。虽然这种易用性促进了高级机器学习能力的广泛应用，但也引入了通过模型提取攻击（Model Extraction Attacks，MEAs）被利用的安全漏洞。近期研究证明，敌对方可以通过与公开接口的交互系统性地复制目标模型的功能，从而对知识产权、隐私和系统安全构成威胁。在本文中，我们提供了对MEAs及其相应防御策略的全面综述。我们提出了一个新的分类法，根据攻击机制、防御方法和计算环境对MEAs进行分类。我们的分析涵盖了各种攻击技术，评估了它们的有效性，并突出了现有防御面临的挑战，特别是保持模型实用性和确保安全之间的关键权衡。我们进一步分析了MEAs在不同计算范式中的情况，并讨论了它们的技术、伦理、法律和社会意义，以及未来研究的有希望的方向。这项系统综述旨在为从事AI安全与隐私研究、实践和政策制定的研究人员提供有价值的参考。此外，我们在此httpsURL上维护了一个不断更新的相关文献在线仓库。 

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
# TOM: An Open-Source Tongue Segmentation Method with Multi-Teacher Distillation and Task-Specific Data Augmentation 

**Title (ZH)**: TOM：一种基于多教师蒸馏和任务特定数据增强的开源舌段分割方法 

**Authors**: Jiacheng Xie, Ziyang Zhang, Biplab Poudel, Congyu Guo, Yang Yu, Guanghui An, Xiaoting Tang, Lening Zhao, Chunhui Xu, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14932)  

**Abstract**: Tongue imaging serves as a valuable diagnostic tool, particularly in Traditional Chinese Medicine (TCM). The quality of tongue surface segmentation significantly affects the accuracy of tongue image classification and subsequent diagnosis in intelligent tongue diagnosis systems. However, existing research on tongue image segmentation faces notable limitations, and there is a lack of robust and user-friendly segmentation tools. This paper proposes a tongue image segmentation model (TOM) based on multi-teacher knowledge distillation. By incorporating a novel diffusion-based data augmentation method, we enhanced the generalization ability of the segmentation model while reducing its parameter size. Notably, after reducing the parameter count by 96.6% compared to the teacher models, the student model still achieves an impressive segmentation performance of 95.22% mIoU. Furthermore, we packaged and deployed the trained model as both an online and offline segmentation tool (available at this https URL), allowing TCM practitioners and researchers to use it without any programming experience. We also present a case study on TCM constitution classification using segmented tongue patches. Experimental results demonstrate that training with tongue patches yields higher classification performance and better interpretability than original tongue images. To our knowledge, this is the first open-source and freely available tongue image segmentation tool. 

**Abstract (ZH)**: 基于多师知识蒸馏的舌象图像分割模型：一种增强泛化能力的同时减少参数量的方法及其应用 

---
# AI Testing Should Account for Sophisticated Strategic Behaviour 

**Title (ZH)**: AI测试应考虑复杂的策略行为 

**Authors**: Vojtech Kovarik, Eric Olav Chen, Sami Petersen, Alexis Ghersengorin, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2508.14927)  

**Abstract**: This position paper argues for two claims regarding AI testing and evaluation. First, to remain informative about deployment behaviour, evaluations need account for the possibility that AI systems understand their circumstances and reason strategically. Second, game-theoretic analysis can inform evaluation design by formalising and scrutinising the reasoning in evaluation-based safety cases. Drawing on examples from existing AI systems, a review of relevant research, and formal strategic analysis of a stylised evaluation scenario, we present evidence for these claims and motivate several research directions. 

**Abstract (ZH)**: 本立场论文提出关于AI测试与评估的两个观点。首先，为了保持对部署行为的说明性，评估需要考虑到AI系统可能理解其环境并进行战略推理的可能性。其次，博弈论分析可以通过正式化和审查基于评估的安全案例中的推理来指导评估设计。通过现有AI系统的实例、相关研究的回顾以及对一种简化评估场景的正式战略分析，我们提供了这些观点的证据，并促进了几个研究方向。 

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
# Accelerating GenAI Workloads by Enabling RISC-V Microkernel Support in IREE 

**Title (ZH)**: 在IREE中启用RISC-V微内核支持以加速GenAI工作负载 

**Authors**: Adeel Ahmad, Ahmad Tameem Kamal, Nouman Amir, Bilal Zafar, Saad Bin Nasir  

**Link**: [PDF](https://arxiv.org/pdf/2508.14899)  

**Abstract**: This project enables RISC-V microkernel support in IREE, an MLIR-based machine learning compiler and runtime. The approach begins by enabling the lowering of MLIR linalg dialect contraction ops to linalg.mmt4d op for the RISC-V64 target within the IREE pass pipeline, followed by the development of optimized microkernels for RISC-V. The performance gains are compared with upstream IREE and this http URL for the Llama-3.2-1B-Instruct model. 

**Abstract (ZH)**: 这个项目在IREE中为RISC-V微内核支持MLIR机器学习编译器和运行时提供了支持。该方法首先在IREE passes管道中启用将MLIR linalg dialect收缩操作降低为RISC-V64目标的linalg.mmt4d op，随后开发针对RISC-V的优化微内核。性能增益与上游IREE进行比较，并与Llama-3.2-1B-Instruct模型的此链接进行对比。 

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
