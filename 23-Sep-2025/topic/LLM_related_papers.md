# V2V-GoT: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multimodal Large Language Models and Graph-of-Thoughts 

**Title (ZH)**: V2V-GoT：多模态大规模语言模型与图思维的车辆间协同自主驾驶 

**Authors**: Hsu-kuang Chiu, Ryo Hachiuma, Chien-Yi Wang, Yu-Chiang Frank Wang, Min-Hung Chen, Stephen F. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2509.18053)  

**Abstract**: Current state-of-the-art autonomous vehicles could face safety-critical situations when their local sensors are occluded by large nearby objects on the road. Vehicle-to-vehicle (V2V) cooperative autonomous driving has been proposed as a means of addressing this problem, and one recently introduced framework for cooperative autonomous driving has further adopted an approach that incorporates a Multimodal Large Language Model (MLLM) to integrate cooperative perception and planning processes. However, despite the potential benefit of applying graph-of-thoughts reasoning to the MLLM, this idea has not been considered by previous cooperative autonomous driving research. In this paper, we propose a novel graph-of-thoughts framework specifically designed for MLLM-based cooperative autonomous driving. Our graph-of-thoughts includes our proposed novel ideas of occlusion-aware perception and planning-aware prediction. We curate the V2V-GoT-QA dataset and develop the V2V-GoT model for training and testing the cooperative driving graph-of-thoughts. Our experimental results show that our method outperforms other baselines in cooperative perception, prediction, and planning tasks. 

**Abstract (ZH)**: 基于MLLM的 Cooperative 自动驾驶中的图思维框架：面向遮挡感知与规划预测的novel方法 

---
# Orchestrate, Generate, Reflect: A VLM-Based Multi-Agent Collaboration Framework for Automated Driving Policy Learning 

**Title (ZH)**: orchestrate、generate、reflect：一种基于大模型的多Agents协作框架，用于自动驾驶策略学习 

**Authors**: Zengqi Peng, Yusen Xie, Yubin Wang, Rui Yang, Qifeng Chen, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.17042)  

**Abstract**: The advancement of foundation models fosters new initiatives for policy learning in achieving safe and efficient autonomous driving. However, a critical bottleneck lies in the manual engineering of reward functions and training curricula for complex and dynamic driving tasks, which is a labor-intensive and time-consuming process. To address this problem, we propose OGR (Orchestrate, Generate, Reflect), a novel automated driving policy learning framework that leverages vision-language model (VLM)-based multi-agent collaboration. Our framework capitalizes on advanced reasoning and multimodal understanding capabilities of VLMs to construct a hierarchical agent system. Specifically, a centralized orchestrator plans high-level training objectives, while a generation module employs a two-step analyze-then-generate process for efficient generation of reward-curriculum pairs. A reflection module then facilitates iterative optimization based on the online evaluation. Furthermore, a dedicated memory module endows the VLM agents with the capabilities of long-term memory. To enhance robustness and diversity of the generation process, we introduce a parallel generation scheme and a human-in-the-loop technique for augmentation of the reward observation space. Through efficient multi-agent cooperation and leveraging rich multimodal information, OGR enables the online evolution of reinforcement learning policies to acquire interaction-aware driving skills. Extensive experiments in the CARLA simulator demonstrate the superior performance, robust generalizability across distinct urban scenarios, and strong compatibility with various RL algorithms. Further real-world experiments highlight the practical viability and effectiveness of our framework. The source code will be available upon acceptance of the paper. 

**Abstract (ZH)**: 基础模型的进步促进了在实现安全高效自动驾驶中政策学习的新倡议。然而，在为复杂和动态驾驶任务手动工程化奖励函数和训练课程方面存在一个关键瓶颈，这是一个劳动密集型和耗时的过程。为了解决这一问题，我们提出了一种名为OGR（Orchestrate, Generate, Reflect）的新型自动化驾驶政策学习框架，该框架利用基于视觉-语言模型（VLM）的多智能体协作。我们的框架利用VLM的高级推理和跨模态理解能力来构建分层智能体系统。具体而言，中心化的协调器规划高层次的训练目标，生成模块采用分析后再生成的两步过程高效生成奖励-课程对。反思模块则在此基础上促进基于在线评估的迭代优化。此外，专门的记忆模块赋予VLM智能体长期记忆的能力。为了增强生成过程的鲁棒性和多样性，我们引入了并行生成方案和基于人工的循环技术来扩展奖励观测空间。通过高效的多智能体合作并利用丰富的跨模态信息，OGR使强化学习政策能够在线进化，以获得交互意识的驾驶技能。在CARLA模拟器上的大量实验展示了其优越性能、跨不同城市场景的稳健泛化能力和与多种RL算法的强兼容性。进一步的现实世界实验突显了我们框架的实用可行性和有效性。论文被接受后，我们将提供源代码。 

---
# LLM-Guided Task- and Affordance-Level Exploration in Reinforcement Learning 

**Title (ZH)**: LLM 引导的任务级和操作级探索在强化学习中的应用 

**Authors**: Jelle Luijkx, Runyu Ma, Zlatan Ajanović, Jens Kober  

**Link**: [PDF](https://arxiv.org/pdf/2509.16615)  

**Abstract**: Reinforcement learning (RL) is a promising approach for robotic manipulation, but it can suffer from low sample efficiency and requires extensive exploration of large state-action spaces. Recent methods leverage the commonsense knowledge and reasoning abilities of large language models (LLMs) to guide exploration toward more meaningful states. However, LLMs can produce plans that are semantically plausible yet physically infeasible, yielding unreliable behavior. We introduce LLM-TALE, a framework that uses LLMs' planning to directly steer RL exploration. LLM-TALE integrates planning at both the task level and the affordance level, improving learning efficiency by directing agents toward semantically meaningful actions. Unlike prior approaches that assume optimal LLM-generated plans or rewards, LLM-TALE corrects suboptimality online and explores multimodal affordance-level plans without human supervision. We evaluate LLM-TALE on pick-and-place tasks in standard RL benchmarks, observing improvements in both sample efficiency and success rates over strong baselines. Real-robot experiments indicate promising zero-shot sim-to-real transfer. Code and supplementary material are available at this https URL. 

**Abstract (ZH)**: 基于大规模语言模型的规划指导强化学习（LLM-TALE）：一种用于机器人操作的框架 

---
# Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs 

**Title (ZH)**: 向工程AGI迈进：评估LLMs的工程设计能力 

**Authors**: Xingang Guo, Yaxin Li, Xiangyi Kong, Yilan Jiang, Xiayu Zhao, Zhihua Gong, Yufan Zhang, Daixuan Li, Tianle Sang, Beixiao Zhu, Gregory Jun, Yingbing Huang, Yiqi Liu, Yuqi Xue, Rahul Dev Kundu, Qi Jian Lim, Yizhou Zhao, Luke Alexander Granger, Mohamed Badr Younis, Darioush Keivan, Nippun Sabharwal, Shreyanka Sinha, Prakhar Agarwal, Kojo Vandyck, Hanlin Mai, Zichen Wang, Aditya Venkatesh, Ayush Barik, Jiankun Yang, Chongying Yue, Jingjie He, Libin Wang, Licheng Xu, Hao Chen, Jinwen Wang, Liujun Xu, Rushabh Shetty, Ziheng Guo, Dahui Song, Manvi Jha, Weijie Liang, Weiman Yan, Bryan Zhang, Sahil Bhandary Karnoor, Jialiang Zhang, Rutva Pandya, Xinyi Gong, Mithesh Ballae Ganesh, Feize Shi, Ruiling Xu, Yifan Zhang, Yanfeng Ouyang, Lianhui Qin, Elyse Rosenbaum, Corey Snyder, Peter Seiler, Geir Dullerud, Xiaojia Shelly Zhang, Zuofu Cheng, Pavan Kumar Hanumolu, Jian Huang, Mayank Kulkarni, Mahdi Namazifar, Huan Zhang, Bin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16204)  

**Abstract**: Today, industry pioneers dream of developing general-purpose AI engineers capable of designing and building humanity's most ambitious projects--from starships that will carry us to distant worlds to Dyson spheres that harness stellar energy. Yet engineering design represents a fundamentally different challenge for large language models (LLMs) compared to traditional textbook-style problem solving or factual question answering. Real-world engineering design demands the synthesis of domain knowledge, navigation of complex trade-offs, and management of the tedious processes that consume much of practicing engineers' time. Despite these shared challenges across engineering disciplines, no benchmark currently captures the unique demands of engineering design work. In this work, we introduce ENGDESIGN, an Engineering Design benchmark that evaluates LLMs' abilities to perform practical design tasks across nine engineering domains: Operating System Design, Computer Architecture Design, Control System Design, Mechanical Systems, Structural Design, Digital Hardware Design, Analog Integrated Circuit Design, Robotics, and Signal Processing. Unlike existing benchmarks that focus on factual recall or question answering, ENGDESIGN uniquely emphasizes LLMs' ability to synthesize domain knowledge, reason under constraints, and generate functional, objective-oriented designs. Each task in ENGDESIGN represents a real-world engineering design problem, accompanied by a detailed task description specifying design goals, constraints, and performance requirements. We pioneer a simulation-based evaluation paradigm where LLM-generated designs undergo rigorous testing through executable, domain-specific simulations-from circuit SPICE simulations to structural finite element analysis, from control system validation to robotic motion planning. 

**Abstract (ZH)**: ENGDESIGN：工程设计基准测试 

---
# Reasoning Core: A Scalable RL Environment for LLM Symbolic Reasoning 

**Title (ZH)**: 推理核心：一种可扩展的大型语言模型符号推理环境 

**Authors**: Valentin Lacombe, Valentin Quesnel, Damien Sileo  

**Link**: [PDF](https://arxiv.org/pdf/2509.18083)  

**Abstract**: We introduce Reasoning Core, a new scalable environment for Reinforcement Learning with Verifiable Rewards (RLVR), designed to advance foundational symbolic reasoning in Large Language Models (LLMs). Unlike existing benchmarks that focus on games or isolated puzzles, Reasoning Core procedurally generates problems across core formal domains, including PDDL planning, first-order logic, context-free grammar parsing, causal reasoning, and system equation solving. The environment is built on key design principles of high-generality problem distributions, verification via external tools, and continuous difficulty control, which together provide a virtually infinite supply of novel training instances. Initial zero-shot evaluations with frontier LLMs confirm the difficulty of Reasoning Core's tasks, positioning it as a promising resource to improve the reasoning capabilities of future models. 

**Abstract (ZH)**: 我们介绍Reasoning Core，一种新的可扩展环境，用于具有可验证奖励的强化学习（RLVR），旨在推进大型语言模型（LLMs）的基础符号推理能力。与现有的主要针对游戏或孤立谜题的基准不同，Reasoning Core通过程序生成涵盖核心形式领域的难题，包括PDDL规划、一阶逻辑、上下文无关文法解析、因果推理和系统方程求解。该环境基于高通用性的问题分布、外部工具验证和持续的难度控制等关键设计原则，共同提供了几乎无限的新型训练实例供应。初始零样本评估表明Reasoning Core任务的难度，定位其作为提升未来模型推理能力的有前途的资源。 

---
# Improving Large Language Models Function Calling and Interpretability via Guided-Structured Templates 

**Title (ZH)**: 通过引导结构化模板提高大规模语言模型的功能调用能力和可解释性 

**Authors**: Hy Dang, Tianyi Liu, Zhuofeng Wu, Jingfeng Yang, Haoming Jiang, Tao Yang, Pei Chen, Zhengyang Wang, Helen Wang, Huasheng Li, Bing Yin, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18076)  

**Abstract**: Large language models (LLMs) have demonstrated strong reasoning and tool-use capabilities, yet they often fail in real-world tool-interactions due to incorrect parameterization, poor tool selection, or misinterpretation of user intent. These issues often stem from an incomplete understanding of user goals and inadequate comprehension of tool documentation. While Chain-of-Thought (CoT) prompting has proven effective for enhancing reasoning in general contexts, our analysis reveals that free-form CoT is insufficient and sometimes counterproductive for structured function-calling tasks. To address this, we introduce a curriculum-inspired framework that leverages structured reasoning templates to guide LLMs through more deliberate step-by-step instructions for generating function callings. Experimental results show that our method reduces tool-use errors, achieving 3-12% relative improvements over strong baselines across diverse model series and approaches. Moreover, our framework enhances the robustness, interpretability, and transparency of tool-using agents, advancing the development of more reliable AI assistants for real-world applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了强大的推理和工具使用能力，但在实际的工具交互中经常由于参数设置错误、工具选择不当或误解用户意图而失败。这些问题通常源于对用户目标的不完整理解以及对工具文档的不充分理解。虽然链式思考（CoT）提示在一般情境下增强了推理效果，但我们的分析表明，对于结构化的函数调用任务，自由格式的CoT往往是不够的，有时甚至适得其反。为解决这一问题，我们引入了一个受课程设计启发的框架，利用结构化的推理模板来引导LLMs通过更细致的逐步指令生成函数调用。实验结果表明，该方法降低了工具使用错误，相比强基线方法，在多个不同模型系列和方法上取得了3-12%的相对改进。此外，该框架提高了工具使用代理的鲁棒性、可解释性和透明度，推动了更可靠的实际应用AI助手的发展。 

---
# Mitigating Strategy-Selection Bias in Reasoning for More Effective Test-Time Scaling 

**Title (ZH)**: 缓解推理中的策略选择偏差以实现更有效的测试时缩放 

**Authors**: Zongqian Wu, Baoduo Xu, Tianyu Li, Zhu Sun, Xiaofeng Zhu, Lei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2509.17905)  

**Abstract**: Test-time scaling (TTS) has been shown to improve the performance of large language models (LLMs) by sampling and aggregating diverse reasoning paths. However, existing research has overlooked a critical issue: selection bias of reasoning strategies during scaling. Specifically, when generating reasoning processes, LLMs tend to follow certain strategies (e.g., algebraic solutions for math problems) while neglecting other valid alternatives (e.g., geometric solutions), resulting in insufficient exploration of the solution space. To further understand the impact of this bias, we present a theoretical analysis that reveals when it undermines the effectiveness of test-time scaling. Motivated by this theoretical insight, we introduce TTS-Uniform, a framework designed to mitigate the selection bias of reasoning strategies. It (i) identifies potential strategies, (ii) uniformly allocates the sampling budget across them, and (iii) filters out unstable strategies prior to aggregation. Experimental results show that TTS-Uniform significantly enhances scaling effectiveness across multiple mainstream LLMs and benchmark datasets. 

**Abstract (ZH)**: Test-time Scaling (TTS) 的策略偏差问题及其缓解方法：理论分析与TTS-Uniform框架 

---
# EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving 

**Title (ZH)**: EngiBench：评估工程问题解决能力的大语言模型基准 

**Authors**: Xiyuan Zhou, Xinlei Wang, Yirui He, Yang Wu, Ruixi Zou, Yuheng Cheng, Yulu Xie, Wenxuan Liu, Huan Zhao, Yan Xu, Jinjin Gu, Junhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.17677)  

**Abstract**: Large language models (LLMs) have shown strong performance on mathematical reasoning under well-posed conditions. However, real-world engineering problems require more than mathematical symbolic computation -- they need to deal with uncertainty, context, and open-ended scenarios. Existing benchmarks fail to capture these complexities. We introduce EngiBench, a hierarchical benchmark designed to evaluate LLMs on solving engineering problems. It spans three levels of increasing difficulty (foundational knowledge retrieval, multi-step contextual reasoning, and open-ended modeling) and covers diverse engineering subfields. To facilitate a deeper understanding of model performance, we systematically rewrite each problem into three controlled variants (perturbed, knowledge-enhanced, and math abstraction), enabling us to separately evaluate the model's robustness, domain-specific knowledge, and mathematical reasoning abilities. Experiment results reveal a clear performance gap across levels: models struggle more as tasks get harder, perform worse when problems are slightly changed, and fall far behind human experts on the high-level engineering tasks. These findings reveal that current LLMs still lack the high-level reasoning needed for real-world engineering, highlighting the need for future models with deeper and more reliable problem-solving capabilities. Our source code and data are available at this https URL. 

**Abstract (ZH)**: 大型语言模型在规范条件下的数学推理表现出强劲性能，但实际工程问题不仅需要数学符号计算，还需要处理不确定性、上下文和开放式情境。现有基准未能捕捉这些复杂性。我们引入EngiBench，这是一种分层基准，旨在评估大型语言模型解决工程问题的能力。它涵盖了从基础知识检索到多步情境推理再到开放式建模的三个难度级别，并覆盖了多种工程子领域。为了更深入地理解模型性能，我们系统地将每个问题重写为三种可控变体（扰动、知识增强和数学抽象），使我们能够分别评估模型的鲁棒性、领域specific知识和数学推理能力。实验结果表明，性能差距随着任务难度的增加而增大，当问题稍作改变时性能下降，高层工程任务上远逊色于人类专家。这些发现揭示当前大型语言模型仍缺乏解决实际工程问题所需的高层次推理能力，突显了未来模型需要具有更深更可靠问题解决能力的重要性。我们的源代码和数据可在以下网址获得。 

---
# MontePrep: Monte-Carlo-Driven Automatic Data Preparation without Target Data Instances 

**Title (ZH)**: MontePrep: 基于蒙特卡洛方法的自动数据准备，无需目标数据实例 

**Authors**: Congcong Ge, Yachuan Liu, Yixuan Tang, Yifan Zhu, Yaofeng Tu, Yunjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.17553)  

**Abstract**: In commercial systems, a pervasive requirement for automatic data preparation (ADP) is to transfer relational data from disparate sources to targets with standardized schema specifications. Previous methods rely on labor-intensive supervision signals or target table data access permissions, limiting their usage in real-world scenarios. To tackle these challenges, we propose an effective end-to-end ADP framework MontePrep, which enables training-free pipeline synthesis with zero target-instance requirements. MontePrep is formulated as an open-source large language model (LLM) powered tree-structured search problem. It consists of three pivot components, i.e., a data preparation action sandbox (DPAS), a fundamental pipeline generator (FPG), and an execution-aware pipeline optimizer (EPO). We first introduce DPAS, a lightweight action sandbox, to navigate the search-based pipeline generation. The design of DPAS circumvents exploration of infeasible pipelines. Then, we present FPG to build executable DP pipelines incrementally, which explores the predefined action sandbox by the LLM-powered Monte Carlo Tree Search. Furthermore, we propose EPO, which invokes pipeline execution results from sources to targets to evaluate the reliability of the generated pipelines in FPG. In this way, unreasonable pipelines are eliminated, thus facilitating the search process from both efficiency and effectiveness perspectives. Extensive experimental results demonstrate the superiority of MontePrep with significant improvement against five state-of-the-art competitors. 

**Abstract (ZH)**: 基于大型语言模型的无需训练端到端自动数据准备框架MontePrep 

---
# Correlation or Causation: Analyzing the Causal Structures of LLM and LRM Reasoning Process 

**Title (ZH)**: 相关性或因果性：分析大语言模型和逻辑推理模型的因果结构 

**Authors**: Zhizhang FU, Guangsheng Bao, Hongbo Zhang, Chenkai Hu, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17380)  

**Abstract**: LLMs suffer from critical reasoning issues such as unfaithfulness, bias, and inconsistency, since they lack robust causal underpinnings and may rely on superficial correlations rather than genuine understanding. Successive LRMs have emerged as a promising alternative, leveraging advanced training techniques such as reinforcement learning (RL) and distillation to improve task accuracy. However, the impact of these training methods on causality remains largely unexplored. In this study, we conduct a systematic causal analysis on LLMs and LRMs, examining structural causal models (SCMs) of four key variables: problem instruction (Z), thinking process (T), reasoning steps (X), and answer (Y). Our findings reveal that RLVR-trained LRMs exhibit enhanced causal reasoning capabilities, aligning more closely with ideal causal structures, while LLMs and distilled LRMs fail to address causality-related deficiencies. Our further investigation indicates that RLVR reduces spurious correlations and strengthens genuine causal patterns, thereby mitigating unfaithfulness and bias. In addition, our inspection on the dynamics of the RLVR training process observes a high correlation between reduced spurious features and improved causal structures, where the causal relationships consistently improve in the training process. This study contributes to the understanding of causality in reasoning models, highlights the critical role of RLVR in enhancing causal reasoning, and provides insights for designing future AI systems with stronger causal foundations. We release our code and data at this https URL. 

**Abstract (ZH)**: 大型语言模型在批判性推理方面存在忠实性、偏见和不一致等问题，因为它们缺乏坚实的因果基础，可能依赖于肤浅的相关性而非真正的理解。基于强化学习和蒸馏等高级训练技术的后续逻辑回归模型 emerges 作为一种有前景的替代方案，以提高任务精度。然而，这些训练方法对因果性的影响仍 largely unexplored。在本研究中，我们对大型语言模型和逻辑回归模型进行系统因果分析，考察了四个关键变量的结构因果模型：问题指令（Z）、思维过程（T）、推理步骤（X）和答案（Y）。研究发现，通过RLVR训练的逻辑回归模型表现出增强的因果推理能力，更接近理想的因果结构，而大型语言模型和蒸馏的逻辑回归模型未能解决因果性相关缺陷。进一步研究表明，RLVR减少了无意义的相关性并加强了真正的因果模式，从而缓解了不忠实性和偏见。此外，对RLVR训练过程动态的检查发现，减少无意义特征与改善因果结构之间存在高度相关性，因果关系在训练过程中持续改善。本研究增进了对推理模型中因果性的理解，强调了RLVR在增强因果推理中的关键作用，并为设计具有更强因果基础的未来AI系统提供了见解。我们在此 https://链接 中发布了我们的代码和数据。 

---
# LLaVul: A Multimodal LLM for Interpretable Vulnerability Reasoning about Source Code 

**Title (ZH)**: LLaVul：一种用于源代码可解释漏洞推理的多模态LLM 

**Authors**: Ala Jararweh, Michael Adams, Avinash Sahu, Abdullah Mueen, Afsah Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2509.17337)  

**Abstract**: Increasing complexity in software systems places a growing demand on reasoning tools that unlock vulnerabilities manifest in source code. Many current approaches focus on vulnerability analysis as a classifying task, oversimplifying the nuanced and context-dependent real-world scenarios. Even though current code large language models (LLMs) excel in code understanding, they often pay little attention to security-specific reasoning. We propose LLaVul, a multimodal LLM tailored to provide fine-grained reasoning about code through question-answering (QA). Our model is trained to integrate paired code and natural queries into a unified space, enhancing reasoning and context-dependent insights about code vulnerability. To evaluate our model performance, we construct a curated dataset of real-world vulnerabilities paired with security-focused questions and answers. Our model outperforms state-of-the-art general-purpose and code LLMs in the QA and detection tasks. We further explain decision-making by conducting qualitative analysis to highlight capabilities and limitations. By integrating code and QA, LLaVul enables more interpretable and security-focused code understanding. 

**Abstract (ZH)**: 增加软件系统的复杂性对推理工具提出了 growing 需求，这些工具能够揭示源代码中显现的漏洞。当前许多方法将漏洞分析视为分类任务，简化了复杂的现实场景和上下文依赖性。尽管当前的代码大型语言模型（LLMs）在代码理解方面表现出色，但它们往往很少关注与安全相关的推理。我们提出 LLaVul，这是一种针对通过问答（QA）提供代码细粒度推理的多模态 LLM。我们的模型经过训练，能够将配对的代码和自然语言查询整合到统一的空间中，增强对代码漏洞的推理和上下文依赖性洞察。为了评估模型性能，我们构建了一个包含实际漏洞及其安全焦点问题和答案的受控数据集。我们的模型在问答和检测任务中优于最先进的通用和代码 LLM。我们进一步通过定性分析来解释决策，以突出其能力和局限性。通过整合代码和问答，LLaVul 使代码理解更具可解释性和安全导向。 

---
# CogAtom: From Cognitive Atoms to Olympiad-level Mathematical Reasoning in Large Language Models 

**Title (ZH)**: CogAtom：从认知原子到奥林匹克级别数学推理的大语言模型 

**Authors**: Zhuofan Chen, Jiyuan He, Yichi Zhang, Xing Hu, Haoxing Wen, Jun Bai, Wenge Rong  

**Link**: [PDF](https://arxiv.org/pdf/2509.17318)  

**Abstract**: Mathematical reasoning poses significant challenges for Large Language Models (LLMs) due to its demand for multi-step reasoning and abstract conceptual integration. While recent test-time scaling techniques rely heavily on high-quality, challenging problems, the scarcity of Olympiad-level math problems remains a bottleneck. We introduce CogAtom, a novel cognitive atom-based framework for synthesizing mathematically rigorous and cognitively diverse problems. Unlike prior approaches, CogAtom models problem construction as a process of selecting and recombining fundamental reasoning units, cognitive atoms, extracted from human-authored solutions. A diversity-promoting random walk algorithm enables exploration of the cognitive atom space, while a constraint-based recombination mechanism ensures logical soundness and structural validity. The combinatorial nature of the graph structure provides a near-infinite space of reasoning paths, and the walk algorithm systematically explores this space to achieve large-scale synthesis of high-quality problems; meanwhile, by controlling the number of cognitive atoms, we can precisely adjust problem difficulty, ensuring diversity, scalability, and controllability of the generated problems. Experimental results demonstrate that CogAtom outperforms existing methods in accuracy, reasoning depth, and diversity, generating problems that closely match the difficulty of AIME while exceeding it in structural variation. Our work offers a cognitively grounded pathway toward scalable, high-quality math problem this http URL code is publicly available at this https URL. 

**Abstract (ZH)**: CogAtom：一种基于认知原子的合成数学 rigor 且认知多样的问题框架 

---
# Can Agents Judge Systematic Reviews Like Humans? Evaluating SLRs with LLM-based Multi-Agent System 

**Title (ZH)**: 基于大语言模型的多智能体系统评价系统能否像人类一样评估系统性综述？ 

**Authors**: Abdullah Mushtaq, Muhammad Rafay Naeem, Ibrahim Ghaznavi, Alaa Abd-alrazaq, Aliya Tabassum, Junaid Qadir  

**Link**: [PDF](https://arxiv.org/pdf/2509.17240)  

**Abstract**: Systematic Literature Reviews (SLRs) are foundational to evidence-based research but remain labor-intensive and prone to inconsistency across disciplines. We present an LLM-based SLR evaluation copilot built on a Multi-Agent System (MAS) architecture to assist researchers in assessing the overall quality of the systematic literature reviews. The system automates protocol validation, methodological assessment, and topic relevance checks using a scholarly database. Unlike conventional single-agent methods, our design integrates a specialized agentic approach aligned with PRISMA guidelines to support more structured and interpretable evaluations. We conducted an initial study on five published SLRs from diverse domains, comparing system outputs to expert-annotated PRISMA scores, and observed 84% agreement. While early results are promising, this work represents a first step toward scalable and accurate NLP-driven systems for interdisciplinary workflows and reveals their capacity for rigorous, domain-agnostic knowledge aggregation to streamline the review process. 

**Abstract (ZH)**: 基于多智能体系统的LLM辅助系统综述评价 copilot：促进跨学科工作流中的证据ベース研究 

---
# MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE 

**Title (ZH)**: MoEs比你想象的更强：RoE驱动的超并行推理扩展 

**Authors**: Soheil Zibakhsh, Mohammad Samragh, Kumari Nishu, Lauren Hannah, Arnav Kundu, Minsik Cho  

**Link**: [PDF](https://arxiv.org/pdf/2509.17238)  

**Abstract**: The generation quality of large language models (LLMs) is often improved by utilizing inference-time sequence-level scaling methods (e.g., Chain-of-Thought). We introduce hyper-parallel scaling, a complementary framework that improves prediction quality at the token level. Hyper-parallel scaling computes and aggregates multiple output proposals for a single token from the model. We implement this concept in Mixture-of-Experts (MoE) models, which we refer to as Roster of Experts (RoE). RoE is a training-free inference algorithm that turns a single MoE into a dynamic ensemble of MoEs. RoE injects controlled stochasticity into the expert routing mechanism, enabling it to sample multiple diverse experts for each token and aggregate their outputs for a more accurate final this http URL overcome the computational cost, we introduce an efficient batching strategy and a specialized KV-caching mechanism that minimizes compute and memory overhead. For example, RoE enables a 7B MoE model to match the performance of a 10.5B MoE model while using 30% less compute for inference. These gains are achieved without any fine-tuning of model parameters. 

**Abstract (ZH)**: 大型语言模型生成质量通过利用推理时序列级扩展方法（例如，Chain-of-Thought）得以提升。我们引入超并行扩展，这是一种在token级别提高预测质量的互补框架。超并行扩展为每个token计算并聚合多个输出提案。我们通过混合专家模型（MoE）实现这一概念，并将其称为专家阵容（RoE）。RoE 是一个无需训练的推理算法，能够将单一的MoE转换为动态的MoE集合。RoE 注入受控的随机性，使其能够为每个token抽样多个多样性的专家并聚合它们的输出，以实现更准确的最终生成。为了克服计算成本，我们引入了一种高效的批量策略和专门的KV缓存机制，以最小化计算和内存开销。例如，RoE 使一个7B的MoE模型能够达到一个10.5B的MoE模型的性能，同时推理时使用的计算资源减少30%。这些改进是在不调整模型参数的情况下实现的。 

---
# Shall We Play a Game? Language Models for Open-ended Wargames 

**Title (ZH)**: 我们来玩个游戏？面向开放式战役的语言模型 

**Authors**: Glenn Matlin, Parv Mahajan, Isaac Song, Yixiong Hao, Ryan Bard, Stu Topp, Evan Montoya, M. Rehan Parwani, Soham Shetty, Mark Riedl  

**Link**: [PDF](https://arxiv.org/pdf/2509.17192)  

**Abstract**: Wargames are multi-faceted, multi-player depictions of conflict in which participants' decisions influence future events. Wargames are often used to explore the strategic implications of decision-making. However, it also encompasses entertainment-oriented simulations, ranging from _Chess_ to tabletop role-playing games like _Dungeons & Dragons_ (D&D). On the more open-ended side of the spectrum of wargames, players use natural language to convey their moves, and adjudicators propose outcomes. Language Models (LMs) are increasingly being considered for how they can provide insights into real-world, consequential decisions. We conduct a scoping literature review of a curated selection of 100 recent works on AI in wargames, from which we construct an ontology of wargames in terms of the creativity afforded to either the players or adjudicators. Focusing on the space of wargames with the most open-endedness for players and adjudicators, we distill a set of considerations for when and how to use LMs in different application areas. We also present a set of safety considerations, best practices for deploying LMs in open-ended wargames, and conclude with a set of high-impact open research challenges. 

**Abstract (ZH)**: 战争游戏是多维度的、多人参与的冲突模拟，参与者决策会影响未来事件。战争游戏常用于探索决策的战略意义，但同时也包括娱乐导向的模拟，从国际象棋到桌面角色扮演游戏如《龙与地下城》。在战争游戏中更为开放的范围内，玩家通过自然语言传达行动，裁判提出结果。我们对100篇近期关于人工智能在战争游戏中的应用进行文献综述，构建了一种以玩家或裁判创意为中心的战争游戏本体论。重点研究玩家和裁判创意最为开放的战争游戏空间，总结了在不同应用领域使用语言模型的时间和方式。此外，我们也提出了安全考虑、开放性战争游戏中部署语言模型的最佳实践，并总结了具有高影响力的开放研究挑战。 

---
# RALLM-POI: Retrieval-Augmented LLM for Zero-shot Next POI Recommendation with Geographical Reranking 

**Title (ZH)**: RALLM-POI：增强检索的LLM在地理重排中的零-shot下一个POI推荐 

**Authors**: Kunrong Li, Kwan Hui Lim  

**Link**: [PDF](https://arxiv.org/pdf/2509.17066)  

**Abstract**: Next point-of-interest (POI) recommendation predicts a user's next destination from historical movements. Traditional models require intensive training, while LLMs offer flexible and generalizable zero-shot solutions but often generate generic or geographically irrelevant results due to missing trajectory and spatial context. To address these issues, we propose RALLM-POI, a framework that couples LLMs with retrieval-augmented generation and self-rectification. We first propose a Historical Trajectory Retriever (HTR) that retrieves relevant past trajectories to serve as contextual references, which are then reranked by a Geographical Distance Reranker (GDR) for prioritizing spatially relevant trajectories. Lastly, an Agentic LLM Rectifier (ALR) is designed to refine outputs through self-reflection. Without additional training, RALLM-POI achieves substantial accuracy gains across three real-world Foursquare datasets, outperforming both conventional and LLM-based baselines. Code is released at this https URL. 

**Abstract (ZH)**: 基于检索增强生成和自我校正的Next POI推荐框架：RALLM-POI 

---
# LLMs as Layout Designers: A Spatial Reasoning Perspective 

**Title (ZH)**: LLMs作为布局设计师：从空间推理视角探讨 

**Authors**: Sha Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.16891)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive reasoning and planning abilities in textual domains and can effectively follow instructions for complex tasks, their capacity for spatial understanding and reasoning remains limited. Such capabilities, however, are critical for applications like content-aware graphic layout design, which demands precise placement, alignment, and structural organization of multiple elements within constrained visual spaces. To address this gap, we propose LaySPA, a reinforcement learning-based framework that augments LLM agents with explicit spatial reasoning capabilities. LaySPA leverages hybrid reward signals that capture geometric validity, structural fidelity, and visual quality, enabling agents to model inter-element relationships, navigate the canvas, and optimize spatial arrangements. Through iterative self-exploration and adaptive policy optimization, LaySPA produces both interpretable reasoning traces and structured layouts. Experimental results demonstrate that LaySPA generates structurally sound and visually appealing layouts, outperforming larger general-purpose LLMs and achieving results on par with state-of-the-art specialized layout models. 

**Abstract (ZH)**: 大型语言模型在空间理解与推理能力上的局限性及其解决方法：基于强化学习的LaySPA框架 

---
# seqBench: A Tunable Benchmark to Quantify Sequential Reasoning Limits of LLMs 

**Title (ZH)**: seqBench: 一个可调基准以量化LLMs的序列推理极限 

**Authors**: Mohammad Ramezanali, Mo Vazifeh, Paolo Santi  

**Link**: [PDF](https://arxiv.org/pdf/2509.16866)  

**Abstract**: We introduce seqBench, a parametrized benchmark for probing sequential reasoning limits in Large Language Models (LLMs) through precise, multi-dimensional control over several key complexity dimensions. seqBench allows systematic variation of (1) the logical depth, defined as the number of sequential actions required to solve the task; (2) the number of backtracking steps along the optimal path, quantifying how often the agent must revisit prior states to satisfy deferred preconditions (e.g., retrieving a key after encountering a locked door); and (3) the noise ratio, defined as the ratio between supporting and distracting facts about the environment. Our evaluations on state-of-the-art LLMs reveal a universal failure pattern: accuracy collapses exponentially beyond a model-specific logical depth. Unlike existing benchmarks, seqBench's fine-grained control facilitates targeted analyses of these reasoning failures, illuminating universal scaling laws and statistical limits, as detailed in this paper alongside its generation methodology and evaluation metrics. We find that even top-performing models systematically fail on seqBench's structured reasoning tasks despite minimal search complexity, underscoring key limitations in their commonsense reasoning capabilities. Designed for future evolution to keep pace with advancing models, the seqBench datasets are publicly released to spur deeper scientific inquiry into LLM reasoning, aiming to establish a clearer understanding of their true potential and current boundaries for robust real-world application. 

**Abstract (ZH)**: seqBench：一种用于探究大型语言模型 sequential 推理极限的参数化基准 

---
# Large Language Models as End-to-end Combinatorial Optimization Solvers 

**Title (ZH)**: 大型语言模型作为端到端组合优化求解器 

**Authors**: Xia Jiang, Yaoxin Wu, Minshuo Li, Zhiguang Cao, Yingqian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16865)  

**Abstract**: Combinatorial optimization (CO) problems, central to decision-making scenarios like logistics and manufacturing, are traditionally solved using problem-specific algorithms requiring significant domain expertise. While large language models (LLMs) have shown promise in automating CO problem solving, existing approaches rely on intermediate steps such as code generation or solver invocation, limiting their generality and accessibility. This paper introduces a novel framework that empowers LLMs to serve as end-to-end CO solvers by directly mapping natural language problem descriptions to solutions. We propose a two-stage training strategy: supervised fine-tuning (SFT) imparts LLMs with solution generation patterns from domain-specific solvers, while a feasibility-and-optimality-aware reinforcement learning (FOARL) process explicitly mitigates constraint violations and refines solution quality. Evaluation across seven NP-hard CO problems shows that our method achieves a high feasibility rate and reduces the average optimality gap to 1.03-8.20% by tuning a 7B-parameter LLM, surpassing both general-purpose LLMs (e.g., GPT-4o), reasoning models (e.g., DeepSeek-R1), and domain-specific heuristics. Our method establishes a unified language-based pipeline for CO without extensive code execution or manual architectural adjustments for different problems, offering a general and language-driven alternative to traditional solver design while maintaining relative feasibility guarantees. 

**Abstract (ZH)**: 组合优化问题（CO）在物流和制造等决策场景中至关重要，传统上通过特定问题的算法解决，需要大量的领域专业知识。虽然大型语言模型（LLMs）在自动化CO问题求解方面展现出潜力，但现有方法依赖于代码生成或求解器调用等中间步骤，限制了其通用性和易用性。本文提出了一种新型框架，使LLMs能够作为端到端的CO求解器，直接将自然语言问题描述映射到解决方案。我们提出了一种两阶段培训策略：监督细调（SFT）赋予LLMs来自特定领域求解器的解生成模式，而可实现可行性和最优性增强的强化学习（FOARL）过程明确地减轻约束冲突并细化解的质量。在七个NP难CO问题上的评估表明，我们的方法实现了高可行性率，并通过调整一个7B参数的LLM将平均最优性缺口减少到1.03%-8.20%，超越了通用型LLM（如GPT-4o）、推理模型（如DeepSeek-R1）和特定领域的启发式方法。本方法建立了一个统一的语言驱动管道，适用于CO问题，无需进行大量的代码执行或针对不同问题的手动架构调整，提供了一种相对于传统求解器设计具有通用性和语言驱动性的替代方案，同时保持相对高的可行性保证。 

---
# Roundtable Policy: Improving Scientific Reasoning and Narratives through Confidence-Weighted Consensus of LLMs 

**Title (ZH)**: 圆桌政策：通过大语言模型置信加权共识提升科学推理和叙事能力 

**Authors**: Yu Yao, Jiayi Dong, Ju Li, Yang Yang, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2509.16839)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities not only in language generation but also in advancing scientific discovery. A growing body of work has explored ways to improve their reasoning, from self-consistency and chain-of-thought to multi-agent debate. Inspired by the dynamics of scientific committees and the "Society of Mind," we introduce Roundtable Policy, a complementary inference-time reasoning framework that performs inference through the weighted consensus of multiple LLMs. Our findings indicate that this approach significantly enhances reasoning in complex heterogeneous scientific tasks and improves scientific narratives in terms of creativity, rigor, and logical coherence, while reducing hallucinations that single models are prone to. Our approach emphasizes structured and interpretable consensus rather than opaque convergence, while requiring only black-box access and uniform procedures, making it broadly applicable to multi-LLM reasoning. 

**Abstract (ZH)**: 大型语言模型在语言生成和推动科学发现方面展现了卓越的能力。已有研究探索了通过自洽性、推理链以及多agent辩论等方式来提升其推理能力。受科学委员会动态和“心灵社会论”的启发，我们提出了圆桌政策，这是一种在多个LLM的加权共识基础上进行推理的补充性推理框架。我们的研究表明，这种做法显著增强了复杂异质科学任务中的推理能力，并在创造力、严谨性和逻辑连贯性方面改进了科学叙事，同时减少了单个模型容易出现的幻觉现象。我们的方法强调结构化和可解释的共识而非不透明的收敛性，只需要黑盒访问和统一的流程，使得它广泛适用于多LLM推理。 

---
# Sycophancy Mitigation Through Reinforcement Learning with Uncertainty-Aware Adaptive Reasoning Trajectories 

**Title (ZH)**: 基于不确定性意识自适应推理轨迹的阿谀逢迎缓解方法 

**Authors**: Mohammad Beigi, Ying Shen, Parshin Shojaee, Qifan Wang, Zichao Wang, Chandan Reddy, Ming Jin, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16742)  

**Abstract**: Despite the remarkable capabilities of large language models, current training paradigms inadvertently foster \textit{sycophancy}, i.e., the tendency of a model to agree with or reinforce user-provided information even when it's factually incorrect. To address this challenge, we introduce \textbf{SMART} (Sycophancy Mitigation through Adaptive Reasoning Trajectories), which reframes sycophancy as a \textit{reasoning optimization problem} rather than an output alignment issue. SMART is a two-stage framework comprising: (1) Uncertainty-Aware Adaptive Monte Carlo Tree Search (UA-MCTS), which dynamically adjusts model exploration based on state-level uncertainty to collect high-quality, diverse reasoning trajectories alongside both stepwise progress and final outcome rewards; and (2) progress-based reinforcement learning, which fine-tunes the model using the collected trajectories and reward signals to reinforce effective reasoning patterns. Through extensive experiments, we show that SMART significantly reduces sycophantic behavior while preserving strong performance on out-of-distribution inputs and maintaining general capabilities. These results underscore the importance of optimizing internal reasoning mechanisms to build more truthful and aligned AI assistants. 

**Abstract (ZH)**: 尽管大型语言模型具备 remarkable 的能力，当前的训练范式无意中培养了模型的奉承倾向，即模型倾向与其用户提供但事实错误的信息保持一致或加强这种信息。为了应对这一挑战，我们引入了 SMART（通过自适应推理路径来减少奉承倾向的方法），将其重新定义为一个推理优化问题，而非输出对齐问题。SMART 是一个两阶段框架，包括：（1）不确定性意识自适应蒙特卡洛树搜索（UA-MCTS），该方法根据状态级不确定性动态调整模型探索，以收集高质量、多样的推理路径，同时包含逐步进展和最终结果奖励；以及（2）基于进展的强化学习，该方法利用收集的路径和奖励信号对模型进行微调，以强化有效的推理模式。通过大量实验，我们展示了 SMART 显著减少了奉承行为，同时在分布外输入上保持了强大的性能，并维持了通用能力。这些结果强调了优化内部推理机制的重要性，以构建更具真实性和对齐的 AI 助手。 

---
# NUMINA: A Natural Understanding Benchmark for Multi-dimensional Intelligence and Numerical Reasoning Abilities 

**Title (ZH)**: NUMINA：多维度智能和数值推理能力的自然理解基准 

**Authors**: Changyu Zeng, Yifan Wang, Zimu Wang, Wei Wang, Zhengni Yang, Muyi Bao, Jiming Xiao, Ahn Nguyen, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2509.16656)  

**Abstract**: Recent advancements in 2D multimodal large language models (MLLMs) have significantly improved performance in vision-language tasks. However, extending these capabilities to 3D environments remains a distinct challenge due to the complexity of spatial reasoning. Nevertheless, existing 3D benchmarks often lack fine-grained numerical reasoning task annotations, limiting MLLMs' ability to perform precise spatial measurements and complex numerical reasoning. To address this gap, we introduce NUMINA, the first Natural Understanding benchmark for Multi-dimensional Intelligence and Numerical reasoning Abilities to enhance multimodal indoor perceptual understanding. NUMINA features multi-scale annotations and various question-answer pairs, generated using NUMINA-Flow, an automated annotation pipeline that integrates LLM rewriting and rule-based self-verification. We evaluate the performance of various state-of-the-art LLMs on NUMINA following the Chat-Scene framework, demonstrating that current LLMs struggle with multimodal numerical reasoning, particularly in performing precise computations such as distance and volume estimation, highlighting the need for further advancements in 3D models. The dataset and source codes can be obtained from this https URL. 

**Abstract (ZH)**: Recent advancements in 2D multimodal large language models (MLLMs) have significantly improved performance in vision-language tasks. However, extending these capabilities to 3D environments remains a distinct challenge due to the complexity of spatial reasoning. Nevertheless, existing 3D benchmarks often lack fine-grained numerical reasoning task annotations, limiting MLLMs' ability to perform precise spatial measurements and complex numerical reasoning. To address this gap, we introduce NUMINA, the first Natural Understanding benchmark for Multi-dimensional Intelligence and Numerical reasoning Abilities to enhance multimodal indoor perceptual understanding. NUMINA features multi-scale annotations and various question-answer pairs, generated using NUMINA-Flow, an automated annotation pipeline that integrates LLM rewriting and rule-based self-verification. We evaluate the performance of various state-of-the-art LLMs on NUMINA following the Chat-Scene framework, demonstrating that current LLMs struggle with multimodal numerical reasoning, particularly in performing precise computations such as distance and volume estimation, highlighting the need for further advancements in 3D models. The dataset and source codes can be obtained from this https URL. 

---
# FESTA: Functionally Equivalent Sampling for Trust Assessment of Multimodal LLMs 

**Title (ZH)**: FESTA：多功能模态LLM信任评估的功能等价采样方法 

**Authors**: Debarpan Bhattacharya, Apoorva Kulkarni, Sriram Ganapathy  

**Link**: [PDF](https://arxiv.org/pdf/2509.16648)  

**Abstract**: The accurate trust assessment of multimodal large language models (MLLMs) generated predictions, which can enable selective prediction and improve user confidence, is challenging due to the diverse multi-modal input paradigms. We propose Functionally Equivalent Sampling for Trust Assessment (FESTA), a multimodal input sampling technique for MLLMs, that generates an uncertainty measure based on the equivalent and complementary input samplings. The proposed task-preserving sampling approach for uncertainty quantification expands the input space to probe the consistency (through equivalent samples) and sensitivity (through complementary samples) of the model. FESTA uses only input-output access of the model (black-box), and does not require ground truth (unsupervised). The experiments are conducted with various off-the-shelf multi-modal LLMs, on both visual and audio reasoning tasks. The proposed FESTA uncertainty estimate achieves significant improvement (33.3% relative improvement for vision-LLMs and 29.6% relative improvement for audio-LLMs) in selective prediction performance, based on area-under-receiver-operating-characteristic curve (AUROC) metric in detecting mispredictions. The code implementation is open-sourced. 

**Abstract (ZH)**: 多模态大型语言模型的准确信任评估：一种功能等价采样方法（FESTA） 

---
# Question Answering with LLMs and Learning from Answer Sets 

**Title (ZH)**: 基于LLM的问答与答案集学习 

**Authors**: Manuel Borroto, Katie Gallagher, Antonio Ielo, Irfan Kareem, Francesco Ricca, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2509.16590)  

**Abstract**: Large Language Models (LLMs) excel at understanding natural language but struggle with explicit commonsense reasoning. A recent trend of research suggests that the combination of LLM with robust symbolic reasoning systems can overcome this problem on story-based question answering tasks. In this setting, existing approaches typically depend on human expertise to manually craft the symbolic component. We argue, however, that this component can also be automatically learned from examples. In this work, we introduce LLM2LAS, a hybrid system that effectively combines the natural language understanding capabilities of LLMs, the rule induction power of the Learning from Answer Sets (LAS) system ILASP, and the formal reasoning strengths of Answer Set Programming (ASP). LLMs are used to extract semantic structures from text, which ILASP then transforms into interpretable logic rules. These rules allow an ASP solver to perform precise and consistent reasoning, enabling correct answers to previously unseen questions. Empirical results outline the strengths and weaknesses of our automatic approach for learning and reasoning in a story-based question answering benchmark. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在理解自然语言方面表现出色，但在明确常识推理方面存在困难。最近的研究趋势表明，大规模语言模型与 robust 符号推理系统的结合可以克服这一问题，特别是在基于故事的问题回答任务中。在这一场景下，现有方法通常依赖于人工知识来手动构建象征性组件。然而，我们认为这种组件也可以从示例中自动学习。在此工作中，我们介绍了一种名为 LLM2LAS 的混合系统，该系统有效地结合了大规模语言模型的自然语言理解能力、Learning from Answer Sets（LAS）系统 ILASP 的规则归纳能力和 AnsweSet Programming（ASP）的形式推理优势。大规模语言模型用于从文本中提取语义结构，然后由 ILASP 转化为可解释的逻辑规则。这些规则使 ASP 解决器能够进行精确一致的推理，从而正确回答以前未见过的问题。实验结果概述了我们在基于故事的问题回答基准测试中自动学习和推理的优势和不足。 

---
# Zero-Shot Human Mobility Forecasting via Large Language Model with Hierarchical Reasoning 

**Title (ZH)**: 基于层次推理的大语言模型驱动的零样本人类移动性预测 

**Authors**: Wenyao Li, Ran Zhang, Pengyang Wang, Yuanchun Zhou, Pengfei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16578)  

**Abstract**: Human mobility forecasting is important for applications such as transportation planning, urban management, and personalized recommendations. However, existing methods often fail to generalize to unseen users or locations and struggle to capture dynamic intent due to limited labeled data and the complexity of mobility patterns. We propose ZHMF, a framework for zero-shot human mobility forecasting that combines a semantic enhanced retrieval and reflection mechanism with a hierarchical language model based reasoning system. The task is reformulated as a natural language question answering paradigm. Leveraging LLMs semantic understanding of user histories and context, our approach handles previously unseen prediction scenarios. We further introduce a hierarchical reflection mechanism for iterative reasoning and refinement by decomposing forecasting into an activity level planner and a location level selector, enabling collaborative modeling of long term user intentions and short term contextual preferences. Experiments on standard human mobility datasets show that our approach outperforms existing models. Ablation studies reveal the contribution of each module, and case studies illustrate how the method captures user intentions and adapts to diverse contextual scenarios. 

**Abstract (ZH)**: 零样本人类移动性预测：结合语义增强检索与反射机制和分层语言模型推理系统 

---
# SalaMAnder: Shapley-based Mathematical Expression Attribution and Metric for Chain-of-Thought Reasoning 

**Title (ZH)**: SalaMAnder: 基于Shapley值的数学表达式归因及链式推理度量 

**Authors**: Yue Xin, Chen Shen, Shaotian Yan, Xiaosong Yuan, Yaoming Wang, Xiaofeng Zhang, Chenxi Huang, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.16561)  

**Abstract**: Chain-of-Thought (CoT) prompting enhances the math reasoning capability of large language models (LLMs) to a large margin. However, the mechanism underlying such improvements remains unexplored. In this paper, we present \textbf{SalaMAnder} (\textbf{S}h\textbf{a}p\textbf{l}ey-b\textbf{a}sed \textbf{M}athematical Expression \textbf{A}ttribution a\textbf{nd} M\textbf{e}t\textbf{r}ic), a theoretically grounded methodology as well as a mathematically rigorous evaluation metric for quantifying component-level contributions in few-shot CoT reasoning. Concretely, we leverage the Shapley value for mathematical expression attribution and develop an efficient stratified sampling algorithm that significantly reduces the computational complexity. Besides, we develop the \textbf{CoSP} (\textbf{C}ardinality \textbf{o}f \textbf{S}hapley \textbf{P}ositives) metric through covariance analysis. Comprehensive validation across popular LLM models and diverse mathematical benchmarks demonstrates that the CoSP metric within our SalaMAnder framework exhibits a robust monotonic correlation with model performance, not only providing theoretical explanations for the empirical success of existing few-shot CoT but also establishing mathematically rigorous principles for prompt construction optimization. Furthermore, we verify the reliability of the explanation, based on which we unify the insights of previous work. 

**Abstract (ZH)**: Chain-of-Thought (CoT) 提问增强了大语言模型（LLMs）的数学推理能力，但其工作机制尚未被探讨。本文提出了 SalaMAnder（基于形状的数学表达归属与度量）方法，这是一种理论依据的方法以及用于量化少样本 CoT 推理中组件级贡献的数学严谨评估指标。具体而言，利用 Shapley 值进行数学表达归属，并开发了一种高效分层采样算法，显著降低了计算复杂度。此外，通过协方差分析开发了 CoSP（Shapley 正值数量）指标。跨流行的大语言模型和多元数学基准的全面验证表明，我们 SalaMAnder 框架内的 CoSP 指标与模型性能之间表现出稳健的单调相关性，不仅为现有少样本 CoT 的经验成功提供了理论解释，还建立了用于提示构建优化的数学严谨原则。此外，我们验证了解释的可靠性，并在此基础上统一了以往工作的洞见。 

---
# GPO: Learning from Critical Steps to Improve LLM Reasoning 

**Title (ZH)**: GPO: 从关键步骤学习以提高大语言模型推理能力 

**Authors**: Jiahao Yu, Zelei Cheng, Xian Wu, Xinyu Xing  

**Link**: [PDF](https://arxiv.org/pdf/2509.16456)  

**Abstract**: Large language models (LLMs) are increasingly used in various domains, showing impressive potential on different tasks. Recently, reasoning LLMs have been proposed to improve the \textit{reasoning} or \textit{thinking} capabilities of LLMs to solve complex problems. Despite the promising results of reasoning LLMs, enhancing the multi-step reasoning capabilities of LLMs still remains a significant challenge. While existing optimization methods have advanced the LLM reasoning capabilities, they often treat reasoning trajectories as a whole, without considering the underlying critical steps within the trajectory. In this paper, we introduce \textbf{G}uided \textbf{P}ivotal \textbf{O}ptimization (GPO), a novel fine-tuning strategy that dives into the reasoning process to enable more effective improvements. GPO first identifies the `critical step' within a reasoning trajectory - a point that the model must carefully proceed to succeed at the problem. We locate the critical step by estimating the advantage function. GPO then resets the policy to the critical step, samples the new rollout and prioritizes the learning process on those rollouts. This focus allows the model to learn more effectively from pivotal moments within the reasoning process to improve the reasoning performance. We demonstrate that GPO is a general strategy that can be integrated with various optimization methods to improve reasoning performance. Besides theoretical analysis, our experiments across challenging reasoning benchmarks show that GPO can consistently and significantly enhance the performance of existing optimization methods, showcasing its effectiveness and generalizability in improving LLM reasoning by concentrating on pivotal moments within the generation process. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多个领域中被广泛应用，显示了在不同任务上的惊人潜力。近期，提出了推理LLMs以提高LLMs的推理或思考能力以解决复杂问题。尽管推理LLMs展示了令人鼓舞的结果，但增强LLMs的多步推理能力仍然是一项重大挑战。尽管现有的优化方法已经提升了LLMs的推理能力，但它们通常将推理轨迹作为一个整体处理，而不考虑轨迹中的关键步骤。在本文中，我们提出了一种新颖的微调策略——引导关键优化（GPO），该策略深入探究推理过程，以实现更有效的改进。GPO首先识别推理轨迹中的“关键步骤”——模型必须仔细进行以成功解决该问题的点。我们通过估计优势函数来定位关键步骤，然后将策略重置到关键步骤，采样新的展开，并优先学习这些展开。这一聚焦使得模型可以从推理过程中关键时刻的学习中更有效地学习，从而提高推理性能。我们证明，GPO是一种通用策略，可以与各种优化方法结合以提高推理性能。除了理论分析，我们在多个具有挑战性的推理基准上的实验表明，GPO能够一致且显著地提升现有优化方法的性能，展示了其在通过聚焦生成过程中的关键时刻提高LLMs推理能力的有效性和普适性。 

---
# Domain-Specific Constitutional AI: Enhancing Safety in LLM-Powered Mental Health Chatbots 

**Title (ZH)**: 领域特定宪法AI：提升由LLM驱动的心理健康聊天机器人的安全性 

**Authors**: Chenhan Lyu, Yutong Song, Pengfei Zhang, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2509.16444)  

**Abstract**: Mental health applications have emerged as a critical area in computational health, driven by rising global rates of mental illness, the integration of AI in psychological care, and the need for scalable solutions in underserved communities. These include therapy chatbots, crisis detection, and wellness platforms handling sensitive data, requiring specialized AI safety beyond general safeguards due to emotional vulnerability, risks like misdiagnosis or symptom exacerbation, and precise management of vulnerable states to avoid severe outcomes such as self-harm or loss of trust. Despite AI safety advances, general safeguards inadequately address mental health-specific challenges, including crisis intervention accuracy to avert escalations, therapeutic guideline adherence to prevent misinformation, scale limitations in resource-constrained settings, and adaptation to nuanced dialogues where generics may introduce biases or miss distress signals. We introduce an approach to apply Constitutional AI training with domain-specific mental health principles for safe, domain-adapted CAI systems in computational mental health applications. 

**Abstract (ZH)**: 心理健康应用程序已成为计算健康领域的一个关键领域，受到全球心理健康疾病发病率上升、心理护理中人工智能的整合以及对欠服务社区可扩展解决方案的需求驱动。这些应用程序包括治疗聊天机器人、危机检测和处理敏感数据的 wellness 平台，由于情绪脆弱性、误诊或症状加重等风险，以及需要专门的AI安全措施来精确管理脆弱状态以避免自伤等严重后果，一般的安全措施不足以应对心理健康领域的特定挑战。尽管人工智能安全技术取得了进展，但一般的安全措施无法解决心理健康干预的准确性问题、治疗指南的遵循问题、资源受限环境下的扩展限制，以及对复杂对话的适应问题，这可能导致偏差或错失压力信号。我们提出了一种方法，通过结合特定于心理健康领域的宪法AI训练，为计算心理健康应用中的安全且领域适应的CAI系统提供指导。 

---
# VORTEX: Aligning Task Utility and Human Preferences through LLM-Guided Reward Shaping 

**Title (ZH)**: VORTEX: 通过LLM引导的奖励塑形实现任务效用与人类偏好对齐 

**Authors**: Guojun Xiong, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2509.16399)  

**Abstract**: In social impact optimization, AI decision systems often rely on solvers that optimize well-calibrated mathematical objectives. However, these solvers cannot directly accommodate evolving human preferences, typically expressed in natural language rather than formal constraints. Recent approaches address this by using large language models (LLMs) to generate new reward functions from preference descriptions. While flexible, they risk sacrificing the system's core utility guarantees. In this paper, we propose \texttt{VORTEX}, a language-guided reward shaping framework that preserves established optimization goals while adaptively incorporating human feedback. By formalizing the problem as multi-objective optimization, we use LLMs to iteratively generate shaping rewards based on verbal reinforcement and text-gradient prompt updates. This allows stakeholders to steer decision behavior via natural language without modifying solvers or specifying trade-off weights. We provide theoretical guarantees that \texttt{VORTEX} converges to Pareto-optimal trade-offs between utility and preference satisfaction. Empirical results in real-world allocation tasks demonstrate that \texttt{VORTEX} outperforms baselines in satisfying human-aligned coverage goals while maintaining high task performance. This work introduces a practical and theoretically grounded paradigm for human-AI collaborative optimization guided by natural language. 

**Abstract (ZH)**: 基于语言引导的奖励塑形框架：VORTEX在社会影响优化中的应用 

---
# Evaluation of Causal Reasoning for Large Language Models in Contextualized Clinical Scenarios of Laboratory Test Interpretation 

**Title (ZH)**: 大型语言模型在实验室测试解释情境下的上下文因果推理评估 

**Authors**: Balu Bhasuran, Mattia Prosperi, Karim Hanna, John Petrilli, Caretia JeLayne Washington, Zhe He  

**Link**: [PDF](https://arxiv.org/pdf/2509.16372)  

**Abstract**: This study evaluates causal reasoning in large language models (LLMs) using 99 clinically grounded laboratory test scenarios aligned with Pearl's Ladder of Causation: association, intervention, and counterfactual reasoning. We examined common laboratory tests such as hemoglobin A1c, creatinine, and vitamin D, and paired them with relevant causal factors including age, gender, obesity, and smoking. Two LLMs - GPT-o1 and Llama-3.2-8b-instruct - were tested, with responses evaluated by four medically trained human experts. GPT-o1 demonstrated stronger discriminative performance (AUROC overall = 0.80 +/- 0.12) compared to Llama-3.2-8b-instruct (0.73 +/- 0.15), with higher scores across association (0.75 vs 0.72), intervention (0.84 vs 0.70), and counterfactual reasoning (0.84 vs 0.69). Sensitivity (0.90 vs 0.84) and specificity (0.93 vs 0.80) were also greater for GPT-o1, with reasoning ratings showing similar trends. Both models performed best on intervention questions and worst on counterfactuals, particularly in altered outcome scenarios. These findings suggest GPT-o1 provides more consistent causal reasoning, but refinement is required before adoption in high-stakes clinical applications. 

**Abstract (ZH)**: 本研究使用99个与佩尔因果阶梯相一致的临床实验室检测场景评估大型语言模型的因果推理能力：关联、干预和反事实推理。我们考察了包括血红蛋白A1c、肌酐和维生素D在内的常见实验室检测项目，并与相关因果因素（年龄、性别、肥胖和吸烟）配对。测试了两种大型语言模型——GPT-o1和Llama-3.2-8b-instruct，其中响应由四位医学训练的人类专家评估。GPT-o1在区分性能（AUROC总体=0.80±0.12）上强于Llama-3.2-8b-instruct（0.73±0.15），在关联（0.75 vs 0.72）、干预（0.84 vs 0.70）和反事实推理（0.84 vs 0.69）方面得分更高。GPT-o1在敏感性（0.90 vs 0.84）和特异性（0.93 vs 0.80）方面也更高，推理评分也显示出类似的趋势。两种模型在干预问题上的表现最佳，在反事实推理问题上的表现最差，特别是在更改结局情景中。这些发现表明GPT-o1提供了一致的因果推理，但在高 stakes 临床应用中采用之前仍需要改进。 

---
# Psychometric Personality Shaping Modulates Capabilities and Safety in Language Models 

**Title (ZH)**: 心理测量人格塑形调节语言模型的能力与安全性 

**Authors**: Stephen Fitz, Peter Romero, Steven Basart, Sipeng Chen, Jose Hernandez-Orallo  

**Link**: [PDF](https://arxiv.org/pdf/2509.16332)  

**Abstract**: Large Language Models increasingly mediate high-stakes interactions, intensifying research on their capabilities and safety. While recent work has shown that LLMs exhibit consistent and measurable synthetic personality traits, little is known about how modulating these traits affects model behavior. We address this gap by investigating how psychometric personality control grounded in the Big Five framework influences AI behavior in the context of capability and safety benchmarks. Our experiments reveal striking effects: for example, reducing conscientiousness leads to significant drops in safety-relevant metrics on benchmarks such as WMDP, TruthfulQA, ETHICS, and Sycophancy as well as reduction in general capabilities as measured by MMLU. These findings highlight personality shaping as a powerful and underexplored axis of model control that interacts with both safety and general competence. We discuss the implications for safety evaluation, alignment strategies, steering model behavior after deployment, and risks associated with possible exploitation of these findings. Our findings motivate a new line of research on personality-sensitive safety evaluations and dynamic behavioral control in LLMs. 

**Abstract (ZH)**: 大型语言模型在高风险互动中越来越起到中介作用，这加剧了对它们能力和安全性的研究。虽然近期的研究表明，大语言模型展现出一致且可量化的合成人格特质，但很少有人知道如何调节这些特质会如何影响模型行为。我们通过研究基于大五人格框架的心理测量人格控制如何影响人工智能行为，特别是在能力和安全基准测试中的影响，来填补这一空白。我们的实验揭示了显著的效果：例如，减少尽责性会导致在WMDP、TruthfulQA、ETHICS和Sycophancy等基准测试中，与安全相关的指标出现显著下降，同时也会在由MMLU衡量的一般能力上出现下降。这些发现强调了人格塑造作为强有力的且尚未充分探索的模型控制维度的重要性，它与安全性和一般能力相互作用。我们讨论了这些发现对安全评估、对齐策略、部署后引导模型行为以及可能滥用这些发现的风险的影响。我们的研究结果推动了一条新的人格敏感的安全评估和动态行为控制研究线路。 

---
# Generalizability of Large Language Model-Based Agents: A Comprehensive Survey 

**Title (ZH)**: 大型语言模型为基础的代理的一般化能力：一项全面调研 

**Authors**: Minxing Zhang, Yi Yang, Roy Xie, Bhuwan Dhingra, Shuyan Zhou, Jian Pei  

**Link**: [PDF](https://arxiv.org/pdf/2509.16330)  

**Abstract**: Large Language Model (LLM)-based agents have emerged as a new paradigm that extends LLMs' capabilities beyond text generation to dynamic interaction with external environments. By integrating reasoning with perception, memory, and tool use, agents are increasingly deployed in diverse domains like web navigation and household robotics. A critical challenge, however, lies in ensuring agent generalizability - the ability to maintain consistent performance across varied instructions, tasks, environments, and domains, especially those beyond agents' fine-tuning data. Despite growing interest, the concept of generalizability in LLM-based agents remains underdefined, and systematic approaches to measure and improve it are lacking. In this survey, we provide the first comprehensive review of generalizability in LLM-based agents. We begin by emphasizing agent generalizability's importance by appealing to stakeholders and clarifying the boundaries of agent generalizability by situating it within a hierarchical domain-task ontology. We then review datasets, evaluation dimensions, and metrics, highlighting their limitations. Next, we categorize methods for improving generalizability into three groups: methods for the backbone LLM, for agent components, and for their interactions. Moreover, we introduce the distinction between generalizable frameworks and generalizable agents and outline how generalizable frameworks can be translated into agent-level generalizability. Finally, we identify critical challenges and future directions, including developing standardized frameworks, variance- and cost-based metrics, and approaches that integrate methodological innovations with architecture-level designs. By synthesizing progress and highlighting opportunities, this survey aims to establish a foundation for principled research on building LLM-based agents that generalize reliably across diverse applications. 

**Abstract (ZH)**: 基于大型语言模型的代理的一般化：综述 

---
# SEQR: Secure and Efficient QR-based LoRA Routing 

**Title (ZH)**: SEQR：安全高效的基于QR的LoRA路由算法 

**Authors**: William Fleshman, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2509.18093)  

**Abstract**: Low-Rank Adaptation (LoRA) has become a standard technique for parameter-efficient fine-tuning of large language models, enabling large libraries of LoRAs, each for a specific task or domain. Efficiently selecting the correct LoRA adapter for a given input remains a challenge, particularly in secure environments where supervised training of routers may raise privacy concerns. Motivated by previous approaches, we formalize the goal of unsupervised LoRA routing in terms of activation norm maximization, providing a theoretical framework for analysis. We demonstrate the discriminative power of activation norms and introduce SEQR, an unsupervised LoRA routing algorithm designed to maximize efficiency while providing strict routing guarantees. SEQR provably identifies the norm-maximizing adapter with significantly greater efficiency, making it a highly scalable and effective solution for dynamic LoRA composition. We validate our results through experiments that demonstrate improved multi-task performance and efficiency. 

**Abstract (ZH)**: 低秩适应（LoRA）已成为大规模语言模型参数高效微调的标准技术，能够支持大量的LoRA适配器，每个适配器针对特定任务或领域。在安全环境中，如何高效选择合适的LoRA适配器仍然是一项挑战，特别是在可能引发隐私担忧的情况下，监督训练路由器存在风险。受先前方法的启发，我们将无监督LoRA路由的目标形式化为激活范数最大化的任务，从而提供了一个分析的理论框架。我们展示了激活范数的判别能力，并引入了SEQR无监督LoRA路由算法，该算法设计目标是最大化效率并提供严格路由保证。SEQR能够证明以显著更高的效率识别出范数最大的适配器，使其成为动态LoRA组合的高效且有效解决方案。我们的实验结果验证了这一方法在多任务性能和效率方面的改进。 

---
# OnePiece: Bringing Context Engineering and Reasoning to Industrial Cascade Ranking System 

**Title (ZH)**: OnePiece: 将上下文工程与推理引入工业级级联排序系统 

**Authors**: Sunhao Dai, Jiakai Tang, Jiahua Wu, Kun Wang, Yuxuan Zhu, Bingjun Chen, Bangyang Hong, Yu Zhao, Cong Fu, Kangle Wu, Yabo Ni, Anxiang Zeng, Wenjie Wang, Xu Chen, Jun Xu, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2509.18091)  

**Abstract**: Despite the growing interest in replicating the scaled success of large language models (LLMs) in industrial search and recommender systems, most existing industrial efforts remain limited to transplanting Transformer architectures, which bring only incremental improvements over strong Deep Learning Recommendation Models (DLRMs). From a first principle perspective, the breakthroughs of LLMs stem not only from their architectures but also from two complementary mechanisms: context engineering, which enriches raw input queries with contextual cues to better elicit model capabilities, and multi-step reasoning, which iteratively refines model outputs through intermediate reasoning paths. However, these two mechanisms and their potential to unlock substantial improvements remain largely underexplored in industrial ranking systems.
In this paper, we propose OnePiece, a unified framework that seamlessly integrates LLM-style context engineering and reasoning into both retrieval and ranking models of industrial cascaded pipelines. OnePiece is built on a pure Transformer backbone and further introduces three key innovations: (1) structured context engineering, which augments interaction history with preference and scenario signals and unifies them into a structured tokenized input sequence for both retrieval and ranking; (2) block-wise latent reasoning, which equips the model with multi-step refinement of representations and scales reasoning bandwidth via block size; (3) progressive multi-task training, which leverages user feedback chains to effectively supervise reasoning steps during training. OnePiece has been deployed in the main personalized search scenario of Shopee and achieves consistent online gains across different key business metrics, including over $+2\%$ GMV/UU and a $+2.90\%$ increase in advertising revenue. 

**Abstract (ZH)**: 一项基于原则的工业排名系统中的语言模型风格上下文工程与推理统一框架：OnePiece 

---
# Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding 

**Title (ZH)**: Spiffy: 通过无损推测解码加速扩散LLM 

**Authors**: Sudhanshu Agrawal, Risheek Garrepalli, Raghavv Goel, Mingu Lee, Christopher Lott, Fatih Porikli  

**Link**: [PDF](https://arxiv.org/pdf/2509.18085)  

**Abstract**: Diffusion LLMs (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs (AR-LLMs) with the potential to operate at significantly higher token generation rates. However, currently available open-source dLLMs often generate at much lower rates, typically decoding only a single token at every denoising timestep in order to maximize output quality. We present Spiffy, a speculative decoding algorithm that accelerates dLLM inference by $\mathbf{2.8{-}3.1\times}$ while provably preserving the model's output distribution. This work addresses the unique challenges involved in applying ideas from speculative decoding of AR-LLMs to the dLLM setting. Spiffy proposes draft states by leveraging the dLLM's distribution itself in an auto-speculative manner. This approach is efficient and effective, and eliminates the overheads of training and running an independent draft model. To structure the candidate draft states, we propose a novel directed draft graph which is uniquely designed to take advantage of the bidirectional, block-wise nature of dLLM generation and can be verified in parallel by the dLLM. To further optimize the structure of these draft graphs, we introduce an efficient, offline calibration algorithm that procedurally determines high-quality graph configurations. These optimized draft graphs, enabling increased acceptance rates, lead to a significant boost in the overall speedup achieved by the system. Crucially, Spiffy is also complementary to other recent innovations in improving dLLM generation speeds such as KV-caching and multi-token unmasking. We demonstrate that when combined with such parallel decoding algorithms, Spiffy is able to effectively multiply the benefits of these methods leading to total speedups of up to $\mathbf{7.9\times}$. 

**Abstract (ZH)**: Spiffy: 一种通过推测性解码加速扩散大语言模型推理的同时保持输出分布的方法 

---
# Strategic Dishonesty Can Undermine AI Safety Evaluations of Frontier LLM 

**Title (ZH)**: 战略性的不诚实可能损害前沿大语言模型AI安全评估 

**Authors**: Alexander Panfilov, Evgenii Kortukov, Kristina Nikolić, Matthias Bethge, Sebastian Lapuschkin, Wojciech Samek, Ameya Prabhu, Maksym Andriushchenko, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2509.18058)  

**Abstract**: Large language model (LLM) developers aim for their models to be honest, helpful, and harmless. However, when faced with malicious requests, models are trained to refuse, sacrificing helpfulness. We show that frontier LLMs can develop a preference for dishonesty as a new strategy, even when other options are available. Affected models respond to harmful requests with outputs that sound harmful but are subtly incorrect or otherwise harmless in practice. This behavior emerges with hard-to-predict variations even within models from the same model family. We find no apparent cause for the propensity to deceive, but we show that more capable models are better at executing this strategy. Strategic dishonesty already has a practical impact on safety evaluations, as we show that dishonest responses fool all output-based monitors used to detect jailbreaks that we test, rendering benchmark scores unreliable. Further, strategic dishonesty can act like a honeypot against malicious users, which noticeably obfuscates prior jailbreak attacks. While output monitors fail, we show that linear probes on internal activations can be used to reliably detect strategic dishonesty. We validate probes on datasets with verifiable outcomes and by using their features as steering vectors. Overall, we consider strategic dishonesty as a concrete example of a broader concern that alignment of LLMs is hard to control, especially when helpfulness and harmlessness conflict. 

**Abstract (ZH)**: 大型语言模型（LLM）开发者期望其模型具备诚实、 helpful 和无害的特点。然而，面对恶意请求时，模型被训练为拒绝这些请求，牺牲了其帮助性。我们展示了前沿的LLM能够发展出一种新的策略偏好，甚至在其他选择可用的情况下也会选择不诚实。受影响的模型会对有害请求产生看似有害但实际上是微妙错误或实际无害的输出。这种行为会在同一模型家族的不同模型中以难以预测的方式显现。我们未能找到这种倾向欺骗行为的明显原因，但发现更强大的模型更能执行这一策略。战略欺骗已经在安全性评估中产生了实际影响，我们发现欺骗性响应蒙蔽了所有用于检测我们测试的监狱逃脱攻击的输出监控工具，使基准评分不可靠。此外，战略欺骗可以像蜜罐一样应对恶意用户，显著混淆了先前的监狱逃脱攻击。尽管输出监控失败，我们展示了通过内部激活的线性探针可以可靠地检测战略欺骗。我们通过使用可验证结果的数据集和将其特征作为导向向量来验证这些探针。总体而言，我们认为战略欺骗是一种具体的例子，表明大型语言模型的对齐难以控制，尤其是当帮助性和无害性发生冲突时。 

---
# Reinforced Generation of Combinatorial Structures: Applications to Complexity Theory 

**Title (ZH)**: 组合结构的强化生成：复杂性理论的应用 

**Authors**: Ansh Nagda, Prabhakar Raghavan, Abhradeep Thakurta  

**Link**: [PDF](https://arxiv.org/pdf/2509.18057)  

**Abstract**: We explore whether techniques from AI can help discover new combinatorial structures that improve provable limits on efficient algorithms. Specifically, we use AlphaEvolve (an LLM coding agent) to study two settings:
a) Average-case hardness for MAX-CUT and MAX-Independent Set: We improve a recent result of Kunisky and Yu to obtain near-optimal upper and (conditional) lower bounds on certification algorithms for MAX-CUT and MAX-Independent Set on random 3- and 4-regular graphs. Our improved lower bounds are obtained by constructing nearly extremal Ramanujan graphs on as many as $163$ nodes, using AlphaEvolve. Additionally, via analytical arguments we strengthen the upper bounds to settle the computational hardness of these questions up to an error in the third decimal place.
b) Worst-case Hardness of Approximation for MAX-k-CUT: We obtain new inapproximability results, proving that it is NP-hard to approximate MAX-4-CUT and MAX-3-CUT within factors of $0.987$ and $0.9649$ respectively, using AlphaEvolve to discover new gadget reductions. Our MAX-4-CUT result improves upon the SOTA of $0.9883$, and our MAX-3-CUT result improves on the current best gadget-based inapproximability result of $0.9853$, but falls short of improving the SOTA of $16/17$ that relies on a custom PCP, rather than a gadget reduction from "standard" Håstad-style PCPs.
A key technical challenge we faced: verifying a candidate construction produced by AlphaEvolve is costly (often requiring exponential time). In both settings above, our results were enabled by using AlphaEvolve itself to evolve the verification procedure to be faster (sometimes by $10,000\times$). We conclude with a discussion of norms by which to assess the assistance from AI in developing proofs. 

**Abstract (ZH)**: 探索AI技术是否能帮助发现新的组合结构以改进高效算法的证明界限。具体地，我们使用AlphaEvolve（一种大规模语言模型编码代理）研究两种设置：
a) MAX-CUT和MAX-独立集的平均情况硬度：通过AlphaEvolve构建最多163节点的几乎极值拉马努詹图，改进了Kunisky和Yu的近期结果，获得了MAX-CUT和MAX-独立集在随机3-和4-正则图上的近最优上界和（条件性）下界。此外，通过分析论证强化上界，使得这些问题的计算硬度在小数点后三位误差范围内得以解决。
b) MAX-k-CUT近似计算的最坏情况硬度：得到了新的不可近似性结果，证明了利用AlphaEvolve发现的新组件约简可以证明MAX-4-CUT和MAX-3-CUT分别在0.987和0.9649的因数内逼近NP难问题。我们的MAX-4-CUT结果优于当前最先进的0.9883，而MAX-3-CUT结果则超越现有最佳基于组件的不可近似性结果0.9853，但仍未达到依赖定制PCP而非"标准"Håstad风格PCP组件约简的16/17的最先进结果。
面临的 Technical 挑战之一：验证由AlphaEvolve生成的候选构造非常耗时（通常需要指数时间）。在上述两个设置中，我们的结果得益于使用AlphaEvolve本身来进化验证程序使其更快（有时加速了10000倍）。最后，我们讨论了评估AI在证明开发中协助的规范。 

---
# A Knowledge Graph-based Retrieval-Augmented Generation Framework for Algorithm Selection in the Facility Layout Problem 

**Title (ZH)**: 基于知识图谱的检索增强生成框架在设施布局问题中选择算法 

**Authors**: Nikhil N S, Amol Dilip Joshi, Bilal Muhammed, Soban Babu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18054)  

**Abstract**: Selecting a solution algorithm for the Facility Layout Problem (FLP), an NP-hard optimization problem with a multiobjective trade-off, is a complex task that requires deep expert knowledge. The performance of a given algorithm depends on specific problem characteristics such as its scale, objectives, and constraints. This creates a need for a data-driven recommendation method to guide algorithm selection in automated design systems. This paper introduces a new recommendation method to make such expertise accessible, based on a Knowledge Graph-based Retrieval-Augmented Generation (KG RAG) framework. To address this, a domain-specific knowledge graph is constructed from published literature. The method then employs a multi-faceted retrieval mechanism to gather relevant evidence from this knowledge graph using three distinct approaches, which include a precise graph-based search, flexible vector-based search, and high-level cluster-based search. The retrieved evidence is utilized by a Large Language Model (LLM) to generate algorithm recommendations with data-driven reasoning. The proposed KG-RAG method is compared against a commercial LLM chatbot with access to the knowledge base as a table, across a series of diverse, real-world FLP test cases. Based on recommendation accuracy and reasoning capability, the proposed method performed significantly better than the commercial LLM chatbot. 

**Abstract (ZH)**: 基于知识图谱的检索增强生成方法在设施布局问题解决方案算法推荐中的应用：一个面向多目标权衡的NP难优化问题的自动化设计系统中算法选择的指导方法 

---
# Deep Learning as the Disciplined Construction of Tame Objects 

**Title (ZH)**: 深度学习作为驯服对象的有章可循的构造 

**Authors**: Gilles Bareilles, Allen Gehret, Johannes Aspman, Jana Lepšová, Jakub Mareček  

**Link**: [PDF](https://arxiv.org/pdf/2509.18025)  

**Abstract**: One can see deep-learning models as compositions of functions within the so-called tame geometry. In this expository note, we give an overview of some topics at the interface of tame geometry (also known as o-minimality), optimization theory, and deep learning theory and practice. To do so, we gradually introduce the concepts and tools used to build convergence guarantees for stochastic gradient descent in a general nonsmooth nonconvex, but tame, setting. This illustrates some ways in which tame geometry is a natural mathematical framework for the study of AI systems, especially within Deep Learning. 

**Abstract (ZH)**: 可以将深度学习模型视为驯服几何（也称为o-minimality）中的函数组成。在本文综述性注记中，我们介绍了驯服几何、优化理论以及深度学习理论与实践之间的交叉领域的一些主题。通过逐步引入构建随机梯度下降收敛性保证的概念和工具，我们在一般非光滑非凸但驯服的设置下进行阐述，以此说明驯服几何是如何成为研究AI系统，特别是深度学习的自然数学框架的一种方式。 

---
# Beyond Diagnosis: Evaluating Multimodal LLMs for Pathology Localization in Chest Radiographs 

**Title (ZH)**: 超越诊断：评估多模态LLM在胸部X光片中病理定位方面的性能 

**Authors**: Advait Gosai, Arun Kavishwar, Stephanie L. McNamara, Soujanya Samineni, Renato Umeton, Alexander Chowdhury, William Lotter  

**Link**: [PDF](https://arxiv.org/pdf/2509.18015)  

**Abstract**: Recent work has shown promising performance of frontier large language models (LLMs) and their multimodal counterparts in medical quizzes and diagnostic tasks, highlighting their potential for broad clinical utility given their accessible, general-purpose nature. However, beyond diagnosis, a fundamental aspect of medical image interpretation is the ability to localize pathological findings. Evaluating localization not only has clinical and educational relevance but also provides insight into a model's spatial understanding of anatomy and disease. Here, we systematically assess two general-purpose MLLMs (GPT-4 and GPT-5) and a domain-specific model (MedGemma) in their ability to localize pathologies on chest radiographs, using a prompting pipeline that overlays a spatial grid and elicits coordinate-based predictions. Averaged across nine pathologies in the CheXlocalize dataset, GPT-5 exhibited a localization accuracy of 49.7%, followed by GPT-4 (39.1%) and MedGemma (17.7%), all lower than a task-specific CNN baseline (59.9%) and a radiologist benchmark (80.1%). Despite modest performance, error analysis revealed that GPT-5's predictions were largely in anatomically plausible regions, just not always precisely localized. GPT-4 performed well on pathologies with fixed anatomical locations, but struggled with spatially variable findings and exhibited anatomically implausible predictions more frequently. MedGemma demonstrated the lowest performance on all pathologies, showing limited capacity to generalize to this novel task. Our findings highlight both the promise and limitations of current MLLMs in medical imaging and underscore the importance of integrating them with task-specific tools for reliable use. 

**Abstract (ZH)**: 近期研究表明，前沿的大语言模型（LLMs）及其多模态 counterparts 在医学问答和诊断任务中表现出色，凸显了它们在临床应用中的潜在广泛价值，得益于它们的通用性和可访问性。然而，除了诊断之外，医学影像解释的一个基本方面是对病理发现的定位能力。评估定位不仅具有临床和教育意义，还能提供模型在解剖和疾病空间理解方面的见解。在此，我们系统评估了两个通用多模态大语言模型（GPT-4和GPT-5）以及一个专门领域模型（MedGemma），在胸部X光片上定位病理学的性能，使用了一种提示管道，该管道叠加了空间网格并引发了基于坐标的预测。在CheXlocalize数据集中九种病理学的平均情况下，GPT-5的定位准确性为49.7%，其次是GPT-4（39.1%）和MedGemma（17.7%），所有这些都低于特定任务的CNNbaseline（59.9%）和放射科医生基准（80.1%）。尽管性能有限，但误差分析表明，GPT-5的预测大多在解剖上合理的区域，但并不总是精确地定位。GPT-4在具有固定解剖位置的病理学上表现良好，但在空间可变的发现上遇到困难，并且经常出现解剖上不合理的表现形式。MedGemma在所有病理学上的表现最低，显示出有限的能力适应这一新颖的任务。我们的发现凸显了当前多模态大语言模型在医学成像中的潜力和局限性，并强调了将它们与特定任务工具集成对于可靠使用的的重要性。 

---
# Through the Lens of Human-Human Collaboration: A Configurable Research Platform for Exploring Human-Agent Collaboration 

**Title (ZH)**: 通过人类合作的视角：一个可配置的研究平台，用于探索人类-代理合作 

**Authors**: Bingsheng Yao, Jiaju Chen, Chaoran Chen, April Wang, Toby Jia-jun Li, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18008)  

**Abstract**: Intelligent systems have traditionally been designed as tools rather than collaborators, often lacking critical characteristics that collaboration partnerships require. Recent advances in large language model (LLM) agents open new opportunities for human-LLM-agent collaboration by enabling natural communication and various social and cognitive behaviors. Yet it remains unclear whether principles of computer-mediated collaboration established in HCI and CSCW persist, change, or fail when humans collaborate with LLM agents. To support systematic investigations of these questions, we introduce an open and configurable research platform for HCI researchers. The platform's modular design allows seamless adaptation of classic CSCW experiments and manipulation of theory-grounded interaction controls. We demonstrate the platform's effectiveness and usability through two case studies: (1) re-implementing the classic human-human-collaboration task Shape Factory as a between-subject human-agent-collaboration experiment with 16 participants, and (2) a participatory cognitive walkthrough with five HCI researchers to refine workflows and interfaces for experiment setup and analysis. 

**Abstract (ZH)**: 智能系统 traditionally 一直被设计为工具而非合作者，通常缺乏合作伙伴关系所需的关键特性。近年来，大型语言模型（LLM）代理的进步为人类-LLM代理合作开启了新的机会，通过使自然沟通和各种社交与认知行为成为可能。然而，当人类与LLM代理合作时，HCI和CSCW中确立的计算机介导合作原则是否依然存在、发生变化还是失效，这仍不清楚。为支持对这些问题的系统性研究，我们介绍了一个面向HCI研究者的开放和可配置的研究平台。该平台的模块化设计允许经典CSCW实验的无缝适应和基于理论的交互控制的操作。通过两个案例研究展示了该平台的有效性和可用性：（1）将经典的两个人类合作任务Shape Factory重实施为涉及16名参与者的基于被试的人机合作实验；（2）与五名HCI研究人员共同参与的认知 walkthrough，以完善实验设置和分析的工作流和界面。 

---
# The Narcissus Hypothesis:Descending to the Rung of Illusion 

**Title (ZH)**: Narcissus 假设：坠入幻觉的阶梯 

**Authors**: Riccardo Cadei, Christian Internò  

**Link**: [PDF](https://arxiv.org/pdf/2509.17999)  

**Abstract**: Modern foundational models increasingly reflect not just world knowledge, but patterns of human preference embedded in their training data. We hypothesize that recursive alignment-via human feedback and model-generated corpora-induces a social desirability bias, nudging models to favor agreeable or flattering responses over objective reasoning. We refer to it as the Narcissus Hypothesis and test it across 31 models using standardized personality assessments and a novel Social Desirability Bias score. Results reveal a significant drift toward socially conforming traits, with profound implications for corpus integrity and the reliability of downstream inferences. We then offer a novel epistemological interpretation, tracing how recursive bias may collapse higher-order reasoning down Pearl's Ladder of Causality, culminating in what we refer to as the Rung of Illusion. 

**Abstract (ZH)**: 现代基础模型不仅体现了世界知识，还反映了嵌入其训练数据中的人类偏好模式。我们假设通过递归对齐-借助人类反馈和模型生成的数据-会产生社交期望偏差，促使模型倾向于赞同或恭维的回答而非客观推理。我们将此称之为 Narcissus 假设，并使用标准化人格评估和一项新的社交期望偏差评分，在 31 个模型上进行了测试。结果显示显著趋向于社会认同特征，这对语料库完整性和下游推断的可靠性具有深远影响。随后，我们提供了一种新的认识论解释，追溯递归偏见如何可能导致更高阶推理在 Pearl 的因果阶梯上崩溃，最终形成我们称之为幻象阶梯的现象。 

---
# Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs 

**Title (ZH)**: 基于大语言模型的自适应核设计在贝叶斯优化中易如反掌 

**Authors**: Richard Cornelius Suwandi, Feng Yin, Juntao Wang, Renjie Li, Tsung-Hui Chang, Sergios Theodoridis  

**Link**: [PDF](https://arxiv.org/pdf/2509.17998)  

**Abstract**: The efficiency of Bayesian optimization (BO) relies heavily on the choice of the Gaussian process (GP) kernel, which plays a central role in balancing exploration and exploitation under limited evaluation budgets. Traditional BO methods often rely on fixed or heuristic kernel selection strategies, which can result in slow convergence or suboptimal solutions when the chosen kernel is poorly suited to the underlying objective function. To address this limitation, we propose a freshly-baked Context-Aware Kernel Evolution (CAKE) to enhance BO with large language models (LLMs). Concretely, CAKE leverages LLMs as the crossover and mutation operators to adaptively generate and refine GP kernels based on the observed data throughout the optimization process. To maximize the power of CAKE, we further propose BIC-Acquisition Kernel Ranking (BAKER) to select the most effective kernel through balancing the model fit measured by the Bayesian information criterion (BIC) with the expected improvement at each iteration of BO. Extensive experiments demonstrate that our fresh CAKE-based BO method consistently outperforms established baselines across a range of real-world tasks, including hyperparameter optimization, controller tuning, and photonic chip design. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 基于上下文感知内核进化的贝叶斯优化方法 

---
# Variation in Verification: Understanding Verification Dynamics in Large Language Models 

**Title (ZH)**: 验证差异性：理解大规模语言模型中的验证动态 

**Authors**: Yefan Zhou, Austin Xu, Yilun Zhou, Janvijay Singh, Jiang Gui, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2509.17995)  

**Abstract**: Recent advances have shown that scaling test-time computation enables large language models (LLMs) to solve increasingly complex problems across diverse domains. One effective paradigm for test-time scaling (TTS) involves LLM generators producing multiple solution candidates, with LLM verifiers assessing the correctness of these candidates without reference answers. In this paper, we study generative verifiers, which perform verification by generating chain-of-thought (CoT) reasoning followed by a binary verdict. We systematically analyze verification dynamics across three dimensions - problem difficulty, generator capability, and verifier generation capability - with empirical studies on 12 benchmarks across mathematical reasoning, knowledge, and natural language reasoning tasks using 14 open-source models (2B to 72B parameter range) and GPT-4o. Our experiments reveal three key findings about verification effectiveness: (1) Easy problems allow verifiers to more reliably certify correct responses; (2) Weak generators produce errors that are easier to detect than strong generators; (3) Verification ability is generally correlated with the verifier's own problem-solving capability, but this relationship varies with problem difficulty. These findings reveal opportunities to optimize basic verification strategies in TTS applications. First, given the same verifier, some weak generators can nearly match stronger ones in post-verification TTS performance (e.g., the Gemma2-9B to Gemma2-27B performance gap shrinks by 75.5%). Second, we identify cases where strong verifiers offer limited advantage over weak ones, as both fail to provide meaningful verification gains, suggesting that verifier scaling alone cannot overcome fundamental verification challenges. 

**Abstract (ZH)**: Recent Advances in Test-Time Scaling via Generative Verifiers for Large Language Models 

---
# HICode: Hierarchical Inductive Coding with LLMs 

**Title (ZH)**: HICode: 基于层级归纳编码的大型语言模型方法 

**Authors**: Mian Zhong, Pristina Wang, Anjalie Field  

**Link**: [PDF](https://arxiv.org/pdf/2509.17946)  

**Abstract**: Despite numerous applications for fine-grained corpus analysis, researchers continue to rely on manual labeling, which does not scale, or statistical tools like topic modeling, which are difficult to control. We propose that LLMs have the potential to scale the nuanced analyses that researchers typically conduct manually to large text corpora. To this effect, inspired by qualitative research methods, we develop HICode, a two-part pipeline that first inductively generates labels directly from analysis data and then hierarchically clusters them to surface emergent themes. We validate this approach across three diverse datasets by measuring alignment with human-constructed themes and demonstrating its robustness through automated and human evaluations. Finally, we conduct a case study of litigation documents related to the ongoing opioid crisis in the U.S., revealing aggressive marketing strategies employed by pharmaceutical companies and demonstrating HICode's potential for facilitating nuanced analyses in large-scale data. 

**Abstract (ZH)**: 尽管细粒度语料库分析有着广泛的应用，研究人员仍依赖于手动标注，这不具有可扩展性，或者依赖于话题建模等统计工具，这些工具难以控制。我们提出，大规模语言模型（LLM）有可能将研究人员通常手动进行的细致分析扩大到大规模文本语料库。为此，我们借鉴定性研究方法，开发了一种两阶段管道——HICode，首先从分析数据中归纳生成标签，然后对这些标签进行层次聚类以揭示涌现的主题。我们通过与人类构建的主题的对齐程度以及自动和人工评估的鲁棒性，在三个不同的数据集上验证了这一方法。最后，我们对涉及美国当前 opioids 危机的诉讼文件进行了案例研究，揭示了制药公司采用的激进行业营销策略，并展示了 HICode 在大规模数据中促进细致分析的潜力。 

---
# How Persuasive is Your Context? 

**Title (ZH)**: 你的上下文有多有说服力？ 

**Authors**: Tu Nguyen, Kevin Du, Alexander Miserlis Hoyle, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2509.17879)  

**Abstract**: Two central capabilities of language models (LMs) are: (i) drawing on prior knowledge about entities, which allows them to answer queries such as "What's the official language of Austria?", and (ii) adapting to new information provided in context, e.g., "Pretend the official language of Austria is Tagalog.", that is pre-pended to the question. In this article, we introduce targeted persuasion score (TPS), designed to quantify how persuasive a given context is to an LM where persuasion is operationalized as the ability of the context to alter the LM's answer to the question. In contrast to evaluating persuasiveness only by inspecting the greedily decoded answer under the model, TPS provides a more fine-grained view of model behavior. Based on the Wasserstein distance, TPS measures how much a context shifts a model's original answer distribution toward a target distribution. Empirically, through a series of experiments, we show that TPS captures a more nuanced notion of persuasiveness than previously proposed metrics. 

**Abstract (ZH)**: 语言模型的两种核心能力是：(i) 利用关于实体的先验知识，使其能够回答如“奥地利的官方语言是什么？”的问题；(ii) 根据上下文提供的新信息进行调整，例如，“假设奥地利的官方语言是塔加洛语。”这是附加在问题之前的信息。在本文中，我们介绍了针对劝说评分（Targeted Persuasion Score, TPS），旨在量化给定上下文对语言模型的影响程度，其中劝说是模型回答被调整的能力。与仅通过查看模型贪婪解码的答案来评估劝说性不同，TPS 提供了对模型行为更为详细的视角。基于 Wasserstein 距离，TPS 测量上下文如何将模型原始答案分布向目标分布推移。通过一系列实验证明，TPS 捕捉到的劝说性概念比之前提出的指标更为细腻。 

---
# Understanding Post-Training Structural Changes in Large Language Models 

**Title (ZH)**: 大型语言模型训练后结构变化的理解 

**Authors**: Xinyu He, Xianghui Cao  

**Link**: [PDF](https://arxiv.org/pdf/2509.17866)  

**Abstract**: Post-training fundamentally alters the behavior of large language models (LLMs), yet its impact on the internal parameter space remains poorly understood. In this work, we conduct a systematic singular value decomposition (SVD) analysis of principal linear layers in pretrained LLMs, focusing on two widely adopted post-training methods: instruction tuning and long-chain-of-thought (Long-CoT) distillation. Our analysis reveals two consistent and unexpected structural changes:(1) a near-uniform geometric scaling of singular values across layers, which theoretically modulates attention scores; and (2) highly consistent orthogonal transformations are applied to the left and right singular vectors of each matrix. Disrupting this orthogonal consistency leads to catastrophic performance degradation. Based on these findings, we propose a simple yet effective framework that interprets post-training as a reparameterization of fixed subspaces in the pretrained parameter space. Further experiments reveal that singular value scaling behaves as a secondary effect, analogous to a temperature adjustment, whereas the core functional transformation lies in the coordinated rotation of singular vectors. These results challenge the prevailing view of the parameter space in large models as a black box, uncovering the first clear regularities in how parameters evolve during training, and providing a new perspective for deeper investigation into model parameter changes. 

**Abstract (ZH)**: Post-training 基本上会改变大型语言模型（LLMs）的行为，但其对内部参数空间的影响仍然理解不足。在本文中，我们通过系统性的奇异值分解（SVD）分析，重点研究了两种广泛采用的后训练方法：指令调优和长推理链（Long-CoT）精炼中的主要线性层。我们的分析揭示了两种一致且出乎意料的结构性变化：（1）层间奇异值的几乎均匀几何缩放，理论上调节注意力分数；（2）对每个矩阵的左奇异向量和右奇异向量应用高度一致的正交变换。破坏这种正交一致性会导致灾难性的性能下降。基于这些发现，我们提出了一种简单有效的框架，将后训练解释为对预训练参数空间中固定子空间的重新参数化。进一步的实验表明，奇异值缩放表现为次要效应，类似于温度调整，而核心的功能变换则在于奇异向量的协调旋转。这些结果挑战了人们对大型模型参数空间的黑箱观点，揭示了参数在训练过程中演变的首个明确规律，并为深入研究模型参数变化提供了新的视角。 

---
# Revealing Multimodal Causality with Large Language Models 

**Title (ZH)**: 揭示多模态因果关系的大语言模型 

**Authors**: Jin Li, Shoujin Wang, Qi Zhang, Feng Liu, Tongliang Liu, Longbing Cao, Shui Yu, Fang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.17784)  

**Abstract**: Uncovering cause-and-effect mechanisms from data is fundamental to scientific progress. While large language models (LLMs) show promise for enhancing causal discovery (CD) from unstructured data, their application to the increasingly prevalent multimodal setting remains a critical challenge. Even with the advent of multimodal LLMs (MLLMs), their efficacy in multimodal CD is hindered by two primary limitations: (1) difficulty in exploring intra- and inter-modal interactions for comprehensive causal variable identification; and (2) insufficiency to handle structural ambiguities with purely observational data. To address these challenges, we propose MLLM-CD, a novel framework for multimodal causal discovery from unstructured data. It consists of three key components: (1) a novel contrastive factor discovery module to identify genuine multimodal factors based on the interactions explored from contrastive sample pairs; (2) a statistical causal structure discovery module to infer causal relationships among discovered factors; and (3) an iterative multimodal counterfactual reasoning module to refine the discovery outcomes iteratively by incorporating the world knowledge and reasoning capabilities of MLLMs. Extensive experiments on both synthetic and real-world datasets demonstrate the effectiveness of MLLM-CD in revealing genuine factors and causal relationships among them from multimodal unstructured data. 

**Abstract (ZH)**: 从非结构化多模态数据中发现因果机制的研究 

---
# A State-Update Prompting Strategy for Efficient and Robust Multi-turn Dialogue 

**Title (ZH)**: 一种高效可靠的多轮对话状态更新提示策略 

**Authors**: Ziyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17766)  

**Abstract**: Large Language Models (LLMs) struggle with information forgetting and inefficiency in long-horizon, multi-turn dialogues. To address this, we propose a training-free prompt engineering method, the State-Update Multi-turn Dialogue Strategy. It utilizes "State Reconstruction" and "History Remind" mechanisms to effectively manage dialogue history. Our strategy shows strong performance across multiple multi-hop QA datasets. For instance, on the HotpotQA dataset, it improves the core information filtering score by 32.6%, leading to a 14.1% increase in the downstream QA score, while also reducing inference time by 73.1% and token consumption by 59.4%. Ablation studies confirm the pivotal roles of both components. Our work offers an effective solution for optimizing LLMs in long-range interactions, providing new insights for developing more robust Agents. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在长时间多轮对话中存在信息遗忘和效率低下问题。为此，我们提出了一种无需训练的提示工程方法——状态更新多轮对话策略。该方法利用“状态重建”和“历史提醒”机制有效管理对话历史。我们的策略在多个多跳问答数据集上表现出优异性能。例如，在HotpotQA数据集上，它提高了核心信息过滤分数32.6%，使下游问答分数增加了14.1%，同时还将推理时间减少了73.1%，减少了59.4%的令牌消耗。消融研究证实了两组件的关键作用。我们的工作为优化长程交互中的LLMs提供了有效解决方案，并为开发更 robust 的代理提供了新的见解。 

---
# Investigating Bias: A Multilingual Pipeline for Generating, Solving, and Evaluating Math Problems with LLMs 

**Title (ZH)**: 探讨偏差：一种使用大语言模型生成、求解和评估数学问题的多语言管道 

**Authors**: Mariam Mahran, Katharina Simbeck  

**Link**: [PDF](https://arxiv.org/pdf/2509.17701)  

**Abstract**: Large Language Models (LLMs) are increasingly used for educational support, yet their response quality varies depending on the language of interaction. This paper presents an automated multilingual pipeline for generating, solving, and evaluating math problems aligned with the German K-10 curriculum. We generated 628 math exercises and translated them into English, German, and Arabic. Three commercial LLMs (GPT-4o-mini, Gemini 2.5 Flash, and Qwen-plus) were prompted to produce step-by-step solutions in each language. A held-out panel of LLM judges, including Claude 3.5 Haiku, evaluated solution quality using a comparative framework. Results show a consistent gap, with English solutions consistently rated highest, and Arabic often ranked lower. These findings highlight persistent linguistic bias and the need for more equitable multilingual AI systems in education. 

**Abstract (ZH)**: 大型语言模型（LLMs）在教育支持中的应用日益增多，但其响应质量取决于交互语言。本文介绍了一种自动化多语言流水线，用于生成、解决和评估与德国K-10课程对齐的数学问题。我们生成了628道数学练习题，并将其翻译成英语、德语和阿拉伯语。三种商用LLM（GPT-4o-mini、Gemini 2.5 Flash和Qwen-plus）被提示以每种语言生成逐步解决方案。一个保留下来的LLM评审团，包括Claude 3.5 Haiku，使用比较框架评价了解决方案质量。结果表明，存在一致的差距，英语解决方案始终得到最高评分，而阿拉伯语方案经常被评定较低。这些发现揭示了持续的语言偏见，并突显了在教育中需要更加公平的多语言AI系统的需求。 

---
# Evaluating LLM-Generated Versus Human-Authored Responses in Role-Play Dialogues 

**Title (ZH)**: 评估LLM生成与人类撰写的角色扮演对话响应 

**Authors**: Dongxu Lu, Johan Jeuring, Albert Gatt  

**Link**: [PDF](https://arxiv.org/pdf/2509.17694)  

**Abstract**: Evaluating large language models (LLMs) in long-form, knowledge-grounded role-play dialogues remains challenging. This study compares LLM-generated and human-authored responses in multi-turn professional training simulations through human evaluation ($N=38$) and automated LLM-as-a-judge assessment. Human evaluation revealed significant degradation in LLM-generated response quality across turns, particularly in naturalness, context maintenance and overall quality, while human-authored responses progressively improved. In line with this finding, participants also indicated a consistent preference for human-authored dialogue. These human judgements were validated by our automated LLM-as-a-judge evaluation, where Gemini 2.0 Flash achieved strong alignment with human evaluators on both zero-shot pairwise preference and stochastic 6-shot construct ratings, confirming the widening quality gap between LLM and human responses over time. Our work contributes a multi-turn benchmark exposing LLM degradation in knowledge-grounded role-play dialogues and provides a validated hybrid evaluation framework to guide the reliable integration of LLMs in training simulations. 

**Abstract (ZH)**: 评价大型语言模型（LLMs）在长篇知识 Grounded 角色扮演对话中的表现仍然具有挑战性。本研究通过人工评价（N=38）和自动化 LLM 作为评委评估，将 LLM 生成的回答与人类撰写的回答在多轮专业培训模拟中进行比较。人工评价结果显示，LLM 生成的回答质量在多轮对话中显著下降，尤其是在自然度、情境连贯性和总体质量方面，而人类撰写的回答则逐步改进。与此一致，参与者也表示一致偏好人类撰写的对话。我们的自动化 LLM 作为评委评估验证了这些人的判断，其中 Gemini 2.0 Flash 在零样本双向偏好和随机六样本构建评分上与人类评价者实现了强烈对齐，证实了 LLM 和人类回答之间的质量差距随时间增大。本研究提供了一个多轮次基准，揭示了在知识 Grounded 角色扮演对话中 LLM 的性能下降，并提供了一个经过验证的混合评估框架，以指导 LLM 在培训模拟中的可靠集成。 

---
# Turk-LettuceDetect: A Hallucination Detection Models for Turkish RAG Applications 

**Title (ZH)**: Turk-LettuceDetect：面向土耳其语RAG应用的 hallucination 检测模型 

**Authors**: Selva Taş, Mahmut El Huseyni, Özay Ezerceli, Reyhan Bayraktar, Fatma Betül Terzioğlu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17671)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) has been hindered by their tendency to hallucinate, generating plausible but factually incorrect information. While Retrieval-Augmented Generation (RAG) systems attempt to address this issue by grounding responses in external knowledge, hallucination remains a persistent challenge, particularly for morphologically complex, low-resource languages like Turkish. This paper introduces Turk-LettuceDetect, the first suite of hallucination detection models specifically designed for Turkish RAG applications. Building on the LettuceDetect framework, we formulate hallucination detection as a token-level classification task and fine-tune three distinct encoder architectures: a Turkish-specific ModernBERT, TurkEmbed4STS, and multilingual EuroBERT. These models were trained on a machine-translated version of the RAGTruth benchmark dataset containing 17,790 instances across question answering, data-to-text generation, and summarization tasks. Our experimental results show that the ModernBERT-based model achieves an F1-score of 0.7266 on the complete test set, with particularly strong performance on structured tasks. The models maintain computational efficiency while supporting long contexts up to 8,192 tokens, making them suitable for real-time deployment. Comparative analysis reveals that while state-of-the-art LLMs demonstrate high recall, they suffer from low precision due to over-generation of hallucinated content, underscoring the necessity of specialized detection mechanisms. By releasing our models and translated dataset, this work addresses a critical gap in multilingual NLP and establishes a foundation for developing more reliable and trustworthy AI applications for Turkish and other languages. 

**Abstract (ZH)**: 大型语言模型（LLMs）的广泛应用受到其易产生幻觉的问题的阻碍，即生成虽然合理但事实错误的信息。虽然检索增强生成（RAG）系统试图通过使响应基于外部知识来解决这一问题，但对于像土耳其语这样形态复杂且资源有限的语言，幻觉仍然是一个持续的挑战。本文介绍了Turk-LettuceDetect，这是首个专为土耳其RAG应用设计的幻觉检测模型套件。基于LettuceDetect框架，我们将幻觉检测建模为一个标记级分类任务，并分别对三种不同的编码器架构进行了微调：特定于土耳其语的ModernBERT、TurkEmbed4STS以及多语言EuroBERT。这些模型在包含17,790个实例、覆盖问答、数据到文本生成和摘要任务的RAGTruth基准数据集的机器翻译版本上进行了训练。实验结果表明，基于ModernBERT的模型在完整测试集上的F1分数为0.7266，在结构化任务上表现尤为出色。这些模型保持了计算效率，支持长达8,192个标记的长上下文，使其适用于实时部署。比较分析显示，尽管最先进的LLMs具有较高的召回率，但由于过度生成幻觉内容而导致精度较低，突显了专门检测机制的必要性。通过发布我们的模型和翻译数据集，本文填补了多语言NLP中的关键空白，并为开发更可靠和可信赖的土耳其语及其他语言的AI应用建立了基础。 

---
# Mechanistic Interpretability with SAEs: Probing Religion, Violence, and Geography in Large Language Models 

**Title (ZH)**: SAEs中的机制可解释性：探究大型语言模型中的宗教、暴力和地理因素 

**Authors**: Katharina Simbeck, Mariam Mahran  

**Link**: [PDF](https://arxiv.org/pdf/2509.17665)  

**Abstract**: Despite growing research on bias in large language models (LLMs), most work has focused on gender and race, with little attention to religious identity. This paper explores how religion is internally represented in LLMs and how it intersects with concepts of violence and geography. Using mechanistic interpretability and Sparse Autoencoders (SAEs) via the Neuronpedia API, we analyze latent feature activations across five models. We measure overlap between religion- and violence-related prompts and probe semantic patterns in activation contexts. While all five religions show comparable internal cohesion, Islam is more frequently linked to features associated with violent language. In contrast, geographic associations largely reflect real-world religious demographics, revealing how models embed both factual distributions and cultural stereotypes. These findings highlight the value of structural analysis in auditing not just outputs but also internal representations that shape model behavior. 

**Abstract (ZH)**: 尽管关于大型语言模型（LLMs）中的偏差研究日益增多，大多数工作主要集中在性别和种族上，较少关注宗教身份。本文探讨了LLMs中宗教的内部表示方式及其与暴力和地理概念的交集。通过使用Mechanistic Interpretability和Sparse Autoencoders（SAEs）通过Neuronpedia API进行分析，我们跨五个模型研究了潜在特征激活情况。我们测量了与宗教和暴力相关提示之间的重叠，并探究了激活上下文中的语义模式。虽然所有五种宗教在内部凝聚力方面表现出相似性，但伊斯兰教更经常与与暴力语言相关的特征相关联。相反，地理关联主要反映了现实世界的宗教人口分布，揭示了模型如何嵌入事实分布和文化刻板印象。这些发现强调了结构分析的价值，不仅在于审计输出结果，还在于审计塑造模型行为的内部表示。 

---
# AuditoryBench++: Can Language Models Understand Auditory Knowledge without Hearing? 

**Title (ZH)**: AuditoryBench++: 语言模型能够在不聆听的情况下理解听觉知识吗？ 

**Authors**: Hyunjong Ok, Suho Yoo, Hyeonjun Kim, Jaeho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.17641)  

**Abstract**: Even without directly hearing sounds, humans can effortlessly reason about auditory properties, such as pitch, loudness, or sound-source associations, drawing on auditory commonsense. In contrast, language models often lack this capability, limiting their effectiveness in multimodal interactions. As an initial step to address this gap, we present AuditoryBench++, a comprehensive benchmark for evaluating auditory knowledge and reasoning in text-only settings. The benchmark encompasses tasks that range from basic auditory comparisons to contextually grounded reasoning, enabling fine-grained analysis of how models process and integrate auditory concepts. In addition, we introduce AIR-CoT, a novel auditory imagination reasoning method that generates and integrates auditory information during inference through span detection with special tokens and knowledge injection. Extensive experiments with recent LLMs and Multimodal LLMs demonstrate that AIR-CoT generally outperforms both the off-the-shelf models and those augmented with auditory knowledge. The project page is available at this https URL. 

**Abstract (ZH)**: 即使没有直接听到声音，人类也能轻松推断出音高、响度或声源关联等听觉属性，基于听觉常识进行推理。相比之下，语言模型往往缺乏这种能力，限制了其在多模态交互中的有效性。为解决这一问题，我们提出了AuditoryBench++，一个全面的基准，用于评估文本-only设置中的听觉知识和推理能力。该基准涵盖从基本的听觉比较到基于上下文的推理任务，有助于细致分析模型如何处理和整合听觉概念。此外，我们引入了AIR-CoT，一种新颖的听觉想象推理方法，在推理过程中通过断言检测和特殊标记以及知识注入生成和整合听觉信息。对最新的语言模型和多模态语言模型的广泛实验表明，AIR-CoT普遍优于现成的模型以及那些结合了听觉知识的模型。项目页面可访问：this https URL。 

---
# MSCoRe: A Benchmark for Multi-Stage Collaborative Reasoning in LLM Agents 

**Title (ZH)**: MSCoRe：多阶段协作推理基准模型 

**Authors**: Yuzhen Lei, Hongbin Xie, Jiaxing Zhao, Shuangxue Liu, Xuan Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.17628)  

**Abstract**: Large Language Models (LLMs) have excelled in question-answering (QA) tasks within single domains. However, their reasoning and coordination capabilities in complex, multi-stage scenarios remain underexplored. Existing benchmarks typically focus on isolated tasks or narrow domains, overlooking models' abilities for multi-stage collaboration and optimization without explicit external guidance. To bridge this gap, we propose \textbf{MSCoRe}, a novel benchmark comprising 126696 domain-specific QA instances spanning scenarios in automotive, pharmaceutical, electronics, and energy sectors. The dataset is created using a structured three-phase pipeline: dynamic sampling, iterative question-answer generation, and a multi-level quality assessment to ensure data quality. Tasks are further categorized into three difficulty levels according to stage coverage and complexity. With MSCoRe, we have conducted a comprehensive evaluation of various state-of-the-art LLM agents. The commercial models performed best across all tasks and scenarios, but a notable gap in ROUGE scores remains between simple and complex tasks. We also tested the models' robustness and found that their performance is negatively affected by noisy data. MSCoRe provides a valuable new resource for the community to evaluate and improve multi-stage reasoning in LLM agents. The code and data are available at this https URL. 

**Abstract (ZH)**: MSCoRe：一种包含126696个领域特定问题-回答实例的新型基准，用于评估大规模语言模型在复杂多阶段场景中的推理与协作能力 

---
# Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning 

**Title (ZH)**: 无需训练的LLMs在非文本模态上的推理能力：基于上下文表示学习的案例研究 

**Authors**: Tianle Zhang, Wanlong Fang, Jonathan Woo, Paridhi Latawa, Deepak A.Subramanian, Alvin Chan  

**Link**: [PDF](https://arxiv.org/pdf/2509.17552)  

**Abstract**: The remarkable performance of Large Language Models (LLMs) can be enhanced with test-time computation, which relies on external tools and even other deep learning models. However, existing approaches for integrating non-text modality representations into LLMs typically require additional costly supervised training, restricting on-the-fly adaptation to new domains and modalities. In this work, we explore the feasibility of integrating representations from non-text foundational models (FMs) into text-based LLMs in a training-free manner. We propose In-Context Representation Learning (ICRL) as a proof-of-concept to allow LLMs to adaptively utilize non-text modality representations with few-shot learning. Unlike traditional in-context learning, which incorporates text-label pairs, ICRL replaces text inputs with FM representations, enabling the LLM to perform multi-modal inference without fine-tuning. We evaluate ICRL on a suite of tasks in the molecular domain, investigating three core research questions: (i) how to map FM representations into LLMs in a training-free manner, (ii) what factors influence ICRL performance, and (iii) what mechanisms underlie the effectiveness of ICRL. To the best of our knowledge, ICRL is the first training-free framework for integrating non-text modality representations into text-based LLMs, presenting a promising direction for adaptable, multi-modal generalization. 

**Abstract (ZH)**: 大型语言模型（LLMs）的出色性能可以通过测试时计算得到增强，这依赖于外部工具甚至其他深度学习模型。然而，现有方法将非文本模态表示集成到LLMs中通常需要额外的昂贵监督训练，限制了对新领域和模态的即时适应。在本文中，我们探讨了以无监督方式将非文本基础模型（FMs）的表示集成到基于文本的LLMs中的可行性。我们提出In-Context Representation Learning（ICRL）作为一种概念验证，允许LLMs通过少样本学习自适应地利用非文本模态表示。与传统的在上下文学习不同，后者结合了文本-标签对，ICRL用FM表示替换文本输入，使LLM能够在不微调的情况下进行多模态推理。我们评估ICRL在分子领域的多项任务中，探讨了三个核心研究问题：（i）如何以无监督方式将FM表示映射到LLMs，（ii）影响ICRL性能的因素，以及（iii）ICRL有效性的机制。据我们所知，ICRL是第一个无监督框架，用于将非文本模态表示集成到基于文本的LLMs中，为可适应的多模态泛化提供了有前景的方向。 

---
# CorefInst: Leveraging LLMs for Multilingual Coreference Resolution 

**Title (ZH)**: CorefInst：利用大规模语言模型进行多语言核心ference解析 

**Authors**: Tuğba Pamay Arslan, Emircan Erol, Gülşen Eryiğit  

**Link**: [PDF](https://arxiv.org/pdf/2509.17505)  

**Abstract**: Coreference Resolution (CR) is a crucial yet challenging task in natural language understanding, often constrained by task-specific architectures and encoder-based language models that demand extensive training and lack adaptability. This study introduces the first multilingual CR methodology which leverages decoder-only LLMs to handle both overt and zero mentions. The article explores how to model the CR task for LLMs via five different instruction sets using a controlled inference method. The approach is evaluated across three LLMs; Llama 3.1, Gemma 2, and Mistral 0.3. The results indicate that LLMs, when instruction-tuned with a suitable instruction set, can surpass state-of-the-art task-specific architectures. Specifically, our best model, a fully fine-tuned Llama 3.1 for multilingual CR, outperforms the leading multilingual CR model (i.e., Corpipe 24 single stage variant) by 2 pp on average across all languages in the CorefUD v1.2 dataset collection. 

**Abstract (ZH)**: 多语言核心ference解析（CR）方法：基于解码器的大型语言模型的探索 

---
# MapCoder-Lite: Squeezing Multi-Agent Coding into a Single Small LLM 

**Title (ZH)**: MapCoder-Lite: 将多agent编码压缩到一个小型LLM中 

**Authors**: Woongkyu Lee, Junhee Cho, Jungwook Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.17489)  

**Abstract**: Large language models (LLMs) have advanced code generation from single-function tasks to competitive-programming problems, but existing multi-agent solutions either rely on costly large-scale ($>$ 30B) models or collapse when downsized to small open-source models. We present MapCoder-Lite, which upgrades a single 7B model into four role-specialised agents-retriever, planner, coder, and debugger-using only rank-32, role-specific LoRA adapters ($<3\%$ extra parameters). Three lightweight techniques make this possible: (i) trajectory distillation from strong LLMs fixes format fragility in retrieval and debugging, (ii) supervisor-guided correction strengthens planning and coding agents, and (iii) agent-wise LoRA fine-tuning delivers memory-efficient specialisation. Comprehensive evaluation on xCodeEval, APPS, and CodeContests shows that MapCoder-Lite more than doubles xCodeEval accuracy (from $13.2\%$ to $28.3\%$), eliminates all format failures, and closes to within six points of a 32B baseline while cutting GPU memory and token-generation time by $4\times$. These results demonstrate that careful agent-wise fine-tuning unleashes high-quality multi-agent coding on a small language model. 

**Abstract (ZH)**: 大型语言模型（LLMs）已将代码生成从单功能任务提升至竞争对手级别的编程问题，但现有的多Agent解决方案要么依赖于昂贵的大规模（>30B）模型，要么在缩减为小型开源模型时会失效。我们提出MapCoder-Lite，仅使用rank-32、角色特定的LoRA适配器（<3%的额外参数），将单个7B模型升级为四个角色专业化代理——检索器、规划者、编码器和调试器。三种轻量级技术使得这一升级成为可能：（i）来自强LLM的轨迹蒸馏修复了检索和调试中的格式脆弱性，（ii）监督指导下的校正加强了规划和编码代理，（iii）代理级别的LoRA微调实现了内存高效的特殊化。对xCodeEval、APPS和CodeContests的全面评估表明，MapCoder-Lite在xCodeEval准确性上至少提升了1.7倍（从13.2%提升到28.3%），消除了所有格式错误，并在GPU内存和token生成时间减少4倍的情况下，接近32B基线的性能，仅相差6分。这些结果表明，细致的代理级别微调可以在小型语言模型上释放高质量的多Agent编程能力。 

---
# Privacy in Action: Towards Realistic Privacy Mitigation and Evaluation for LLM-Powered Agents 

**Title (ZH)**: 隐私在行动：面向LLM驱动代理的现实隐私缓解与评估 

**Authors**: Shouju Wang, Fenglin Yu, Xirui Liu, Xiaoting Qin, Jue Zhang, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2509.17488)  

**Abstract**: The increasing autonomy of LLM agents in handling sensitive communications, accelerated by Model Context Protocol (MCP) and Agent-to-Agent (A2A) frameworks, creates urgent privacy challenges. While recent work reveals significant gaps between LLMs' privacy Q&A performance and their agent behavior, existing benchmarks remain limited to static, simplified scenarios. We present PrivacyChecker, a model-agnostic, contextual integrity based mitigation approach that effectively reduces privacy leakage from 36.08% to 7.30% on DeepSeek-R1 and from 33.06% to 8.32% on GPT-4o, all while preserving task helpfulness. We also introduce PrivacyLens-Live, transforming static benchmarks into dynamic MCP and A2A environments that reveal substantially higher privacy risks in practical. Our modular mitigation approach integrates seamlessly into agent protocols through three deployment strategies, providing practical privacy protection for the emerging agentic ecosystem. Our data and code will be made available at this https URL. 

**Abstract (ZH)**: LLM代理在处理敏感通信中的 Increasing Autonomy及其对隐私挑战的影响：基于上下文完整性的PrivacyChecker和PrivacyLens-Live方法 

---
# LingoQ: Bridging the Gap between ESL Learning and Work through AI-Generated Work-Related Quizzes 

**Title (ZH)**: LingoQ: 通过AI生成的职业相关测验弭合英语作为第二语言学习与工作之间的差距 

**Authors**: Yeonsun Yang, Sang Won Lee, Jean Y. Song, Sangdoo Yun, Young-Ho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.17477)  

**Abstract**: Non-native English speakers performing English-related tasks at work struggle to sustain ESL learning, despite their motivation. Often, study materials are disconnected from their work context. Although workers rely on LLM assistants to address their immediate needs, these interactions may not directly contribute to their English skills. We present LingoQ, an AI-mediated system that allows workers to practice English using quizzes generated from their LLM queries during work. LingoQ leverages these queries using AI to generate personalized quizzes that workers can review and practice on their smartphones. We conducted a three-week deployment study with 28 ESL workers to evaluate LingoQ. Participants valued the relevance of quizzes that reflect their own context, constantly engaging with the app during the study. This active engagement improved self-efficacy and led to learning gains for beginners and, potentially, for intermediate learners. We discuss opportunities of leveraging users' reliance on LLMs to situate their learning in the user context for improved learning. 

**Abstract (ZH)**: 非母语英语 Speaking员工在工作中执行与英语相关任务时，尽管有动机，仍难以维持 ESL 学习。通常，学习材料与工作场景脱节。尽管员工依赖大语言模型助手解决即时需求，这些互动可能不会直接提升他们的英语技能。我们介绍了一种名为 LingoQ 的 AI 调和系统，该系统允许员工在其工作中使用 LLM 查询生成的测验来练习英语。LingoQ 利用这些查询并通过 AI 生成个性化的测验，供员工在智能手机上复习和练习。我们在 28 名 ESL 工人中进行了为期三周的部署研究以评估 LingoQ。参与者认为反映其自身工作场景的相关测验价值较高，在研究期间不断与应用互动。这种积极互动提高了自我效能感，并对初学者产生了学习收益，且可能对中级学习者也有助益。我们讨论利用用户依赖于大语言模型的机会，将学习置于用户情境中以改进学习的潜在机会。 

---
# AIMMerging: Adaptive Iterative Model Merging Using Training Trajectories for Language Model Continual Learning 

**Title (ZH)**: 自适应迭代模型融合用于语言模型连续学习的训练轨迹引导方法 

**Authors**: Yujie Feng, Jian Li, Xiaoyu Dong, Pengfei Xu, Xiaohui Zhou, Yujia Zhang, Zexin LU, Yasha Wang, Alan Zhao, Xu Chu, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17348)  

**Abstract**: Continual learning (CL) is essential for deploying large language models (LLMs) in dynamic real-world environments without the need for costly retraining. Recent model merging-based methods have attracted significant attention, but they still struggle to effectively manage the trade-off between learning new knowledge and preventing forgetting, a challenge largely stemming from suboptimal number of merges and merging frequency. In this paper, we introduce Adaptive Iterative Model Merging (AimMerging), a novel CL framework that utilizes learning and forgetting signals from the training trajectory to dynamically monitor the model's training status. Guided by dynamic monitoring, the training trajectory-guided merge controller adaptively determines the timing and frequency of iterative fusion, while the rehearsal-based knowledge fusion module computes the merging weights and executes the fusion. Comprehensive experiments on three CL benchmarks with various model sizes (from 770M to 13B) demonstrate that AimMerging achieves significant performance improvements over existing state-of-the-art methods, with an average relative improvement of 80% and 59% on FWT and BWT, respectively. The source code is provided for reproducibility. 

**Abstract (ZH)**: continual 学习 (CL) 对于在无需昂贵重新训练的情况下部署大型语言模型 (LLMs) 的动态现实环境至关重要。基于模型合并的方法近期受到了广泛关注，但仍难以有效管理学习新知识与防止遗忘之间的trade-off，这一挑战主要源于合并次数和合并频率的次优选择。本文介绍了一种名为自适应迭代模型合并 (AimMerging) 的新型 CL 框架，利用训练轨迹中的学习和遗忘信号动态监控模型的训练状态。在动态监控的引导下，训练轨迹指导的合并控制器自适应地确定迭代合并的时间和频率，而基于复习的知识融合模块计算合并权重并执行合并。在三种不同模型规模（从 770M 到 13B）的 CL 基准测试中，AimMerging 在 FWT 和 BWT 上分别实现了平均 80% 和 59% 的性能改进，源代码已提供以确保可再现性。 

---
# Generalizable End-to-End Tool-Use RL with Synthetic CodeGym 

**Title (ZH)**: 面向通用的端到端工具使用RL与合成CodeGym 

**Authors**: Weihua Du, Hailei Gong, Zhan Ling, Kang Liu, Lingfeng Shen, Xuesong Yao, Yufei Xu, Dingyuan Shi, Yiming Yang, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.17325)  

**Abstract**: Tool-augmented large language models (LLMs), hereafter LLM agents, leverage external tools to solve diverse tasks and interface with the real world. However, current training practices largely rely on supervised fine-tuning (SFT) over static trajectories or reinforcement learning (RL) on narrow tasks, and generalize poorly beyond development settings, leading to brittleness with new tools and unseen workflows. Because code execution reflects many structures of real-world workflows, coding problems provide a natural basis for building agent training environments. Motivated by this, we introduce CodeGym, a scalable framework that synthesizes diverse, verifiable, and controllable multi-turn tool-use environments for agent RL, enabling LLM agents to explore and master various workflows actively. CodeGym rewrites static coding problems into interactive environments by extracting atomic functions or logic into callable tools, yielding verifiable tasks that span various tool-execution workflows. Models of varying sizes and chain-of-thought configurations, trained in CodeGym, exhibit consistent out-of-distribution generalizability; for example, Qwen2.5-32B-Instruct achieves an absolute accuracy gain of 8.7 points on the OOD benchmark $\tau$-Bench. These results highlight CodeGym as a step toward scalable general-purpose RL environments that align with real-world agent workflows. 

**Abstract (ZH)**: 工具增强的大语言模型（LLMs）代理通过利用外部工具解决多样化任务并对接真实世界。然而，当前的训练实践主要依赖于静态轨迹上的监督微调（SFT）或窄任务上的强化学习（RL），在开发设置之外的泛化能力较差，导致在新工具和未见过的工作流程面前变得脆弱。由于代码执行反映了众多真实世界工作流程的结构，编程问题为构建代理的RL训练环境提供了自然的基础。受此启发，我们提出了CodeGym，这是一种可扩展的框架，用于合成多轮可验证和可控的工具使用环境，以促进代理的RL训练，使LLM代理能够积极探索和掌握各种工作流程。CodeGym通过提取原子函数或逻辑为可调用的工具，将静态编程问题重构为交互式环境，产生覆盖多种工具执行工作流程的可验证任务。在CodeGym中训练的大小和思维链配置各异的模型，在分布外泛化方面表现出一致性的提升；例如，Qwen2.5-32B-Instruct在分布外基准τ-Bench上实现了8.7个百分点的绝对准确度提升。这些结果表明，CodeGym是朝着与真实世界代理工作流程相一致的可扩展的通用RL环境迈进的重要一步。 

---
# Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection 

**Title (ZH)**: 基于LLM推理增强的多视图注意力多实例学习在认知 distortion 检测中的应用 

**Authors**: Jun Seo Kim, Hyemi Kim, Woo Joo Oh, Hongjin Cho, Hochul Lee, Hye Hyeon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.17292)  

**Abstract**: Cognitive distortions have been closely linked to mental health disorders, yet their automatic detection remained challenging due to contextual ambiguity, co-occurrence, and semantic overlap. We proposed a novel framework that combines Large Language Models (LLMs) with Multiple-Instance Learning (MIL) architecture to enhance interpretability and expression-level reasoning. Each utterance was decomposed into Emotion, Logic, and Behavior (ELB) components, which were processed by LLMs to infer multiple distortion instances, each with a predicted type, expression, and model-assigned salience score. These instances were integrated via a Multi-View Gated Attention mechanism for final classification. Experiments on Korean (KoACD) and English (Therapist QA) datasets demonstrate that incorporating ELB and LLM-inferred salience scores improves classification performance, especially for distortions with high interpretive ambiguity. Our results suggested a psychologically grounded and generalizable approach for fine-grained reasoning in mental health NLP. 

**Abstract (ZH)**: 认知歪曲与心理健康障碍密切相关，但由于背景模糊、共现和语义重叠，其自动检测仍然具有挑战性。我们提出了一种将大规模语言模型（LLMs）与多实例学习（MIL）架构相结合的新框架，以提高可解释性和表达层面的推理能力。每个陈述被分解为情感、逻辑和行为（ELB）组件，由LLMs处理以推断出多个歪曲实例，每个实例都有预测的类型、表达和模型赋予的相关性得分。这些实例通过多视图门控注意力机制进行集成以进行最终分类。对韩语（KoACD）和英语（Therapist QA）数据集的实验表明，结合ELB和LLM推断的相关性得分可以提高分类性能，尤其是对于具有高解释性模糊性的歪曲。我们的结果表明了一种基于心理和可推广的方法，用于心理健康自然语言处理中的细粒度推理。 

---
# Automated Facility Enumeration for Building Compliance Checking using Door Detection and Large Language Models 

**Title (ZH)**: 基于门检测和大规模语言模型的建筑合规检查自动化设施枚举 

**Authors**: Licheng Zhan, Bach Le, Naveed Akhtar, Tuan Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2509.17283)  

**Abstract**: Building compliance checking (BCC) is a critical process for ensuring that constructed facilities meet regulatory standards. A core component of BCC is the accurate enumeration of facility types and their spatial distribution. Despite its importance, this problem has been largely overlooked in the literature, posing a significant challenge for BCC and leaving a critical gap in existing workflows. Performing this task manually is time-consuming and labor-intensive. Recent advances in large language models (LLMs) offer new opportunities to enhance automation by combining visual recognition with reasoning capabilities. In this paper, we introduce a new task for BCC: automated facility enumeration, which involves validating the quantity of each facility type against statutory requirements. To address it, we propose a novel method that integrates door detection with LLM-based reasoning. We are the first to apply LLMs to this task and further enhance their performance through a Chain-of-Thought (CoT) pipeline. Our approach generalizes well across diverse datasets and facility types. Experiments on both real-world and synthetic floor plan data demonstrate the effectiveness and robustness of our method. 

**Abstract (ZH)**: 构建合规检查中的自动设施计数任务：一种结合门检测与大规模语言模型推理的方法 

---
# Probabilistic Token Alignment for Large Language Model Fusion 

**Title (ZH)**: 大规模语言模型融合的概率性_token_对齐 

**Authors**: Runjia Zeng, James Chenhao Liang, Cheng Han, Zhiwen Cao, Jiahao Liu, Xiaojun Quan, Yingjie Victor Chen, Lifu Huang, Tong Geng, Qifan Wang, Dongfang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17276)  

**Abstract**: Training large language models (LLMs) from scratch can yield models with unique functionalities and strengths, but it is costly and often leads to redundant capabilities. A more cost-effective alternative is to fuse existing pre-trained LLMs with different architectures into a more powerful model. However, a key challenge in existing model fusion is their dependence on manually predefined vocabulary alignment, which may not generalize well across diverse contexts, leading to performance degradation in several evaluation. To solve this, we draw inspiration from distribution learning and propose the probabilistic token alignment method as a general and soft mapping for alignment, named as PTA-LLM. Our approach innovatively reformulates token alignment into a classic mathematical problem: optimal transport, seamlessly leveraging distribution-aware learning to facilitate more coherent model fusion. Apart from its inherent generality, PTA-LLM exhibits interpretability from a distributional perspective, offering insights into the essence of the token alignment. Empirical results demonstrate that probabilistic token alignment enhances the target model's performance across multiple capabilities. Our code is avaliable at this https URL. 

**Abstract (ZH)**: 从现有预训练大语言模型融合生成具备新功能的大语言模型：PTA-LLM方法 

---
# SignalLLM: A General-Purpose LLM Agent Framework for Automated Signal Processing 

**Title (ZH)**: SignalLLM：一种通用的LLM代理框架，用于自动化信号处理 

**Authors**: Junlong Ke, Qiying Hu, Shenghai Yuan, Yuecong Xu, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17197)  

**Abstract**: Modern signal processing (SP) pipelines, whether model-based or data-driven, often constrained by complex and fragmented workflow, rely heavily on expert knowledge and manual engineering, and struggle with adaptability and generalization under limited data. In contrast, Large Language Models (LLMs) offer strong reasoning capabilities, broad general-purpose knowledge, in-context learning, and cross-modal transfer abilities, positioning them as powerful tools for automating and generalizing SP workflows. Motivated by these potentials, we introduce SignalLLM, the first general-purpose LLM-based agent framework for general SP tasks. Unlike prior LLM-based SP approaches that are limited to narrow applications or tricky prompting, SignalLLM introduces a principled, modular architecture. It decomposes high-level SP goals into structured subtasks via in-context learning and domain-specific retrieval, followed by hierarchical planning through adaptive retrieval-augmented generation (RAG) and refinement; these subtasks are then executed through prompt-based reasoning, cross-modal reasoning, code synthesis, model invocation, or data-driven LLM-assisted modeling. Its generalizable design enables the flexible selection of problem solving strategies across different signal modalities, task types, and data conditions. We demonstrate the versatility and effectiveness of SignalLLM through five representative tasks in communication and sensing, such as radar target detection, human activity recognition, and text compression. Experimental results show superior performance over traditional and existing LLM-based methods, particularly in few-shot and zero-shot settings. 

**Abstract (ZH)**: 现代信号处理（SP）管道，无论是基于模型的还是数据驱动的，经常受限于复杂且碎片化的 workflows，依赖于专家知识和手动工程，并且在有限数据下难以具备适应性和泛化能力。相比之下，大型语言模型（LLMs）提供了强大的推理能力、广泛的目的性知识、基于上下文的学习以及跨模态转移能力，定位为自动化和泛化SP流程的强大工具。受这些潜力的启发，我们引入了SignalLLM，这是首个基于LLM的一般-purpose智能代理框架，适用于一般的SP任务。与仅限于窄应用或复杂提示的先前LLM驱动的SP方法不同，SignalLLM引入了一种原理清晰的模块化架构。它通过基于上下文学习和领域特定检索将高层SP目标分解为结构化子任务，之后通过自适应检索增强生成（RAG）和细化进行分层规划；然后通过基于提示的推理、跨模态推理、代码合成、模型调用或数据驱动的LLM辅助建模执行这些子任务。其泛化设计使得能够在不同的信号模态、任务类型和数据条件下灵活选择解决问题的策略。我们通过通信和传感领域的五个代表性任务，如雷达目标检测、人体活动识别和文本压缩，展示了SignalLLM的多样性和有效性。实验结果表明，SignalLLM在少量样本和零样本设置中优于传统的以及其他现有LLM方法。 

---
# Evolution of Concepts in Language Model Pre-Training 

**Title (ZH)**: 语言模型预训练中概念的演进 

**Authors**: Xuyang Ge, Wentao Shu, Jiaxing Wu, Yunhua Zhou, Zhengfu He, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17196)  

**Abstract**: Language models obtain extensive capabilities through pre-training. However, the pre-training process remains a black box. In this work, we track linear interpretable feature evolution across pre-training snapshots using a sparse dictionary learning method called crosscoders. We find that most features begin to form around a specific point, while more complex patterns emerge in later training stages. Feature attribution analyses reveal causal connections between feature evolution and downstream performance. Our feature-level observations are highly consistent with previous findings on Transformer's two-stage learning process, which we term a statistical learning phase and a feature learning phase. Our work opens up the possibility to track fine-grained representation progress during language model learning dynamics. 

**Abstract (ZH)**: 语言模型通过预训练获得了广泛的能力，但预训练过程仍是一个黑盒。在本研究中，我们使用一种称为crosscoders的稀疏字典学习方法，在预训练快照中跟踪线性可解释的特征演变。我们发现，大多数特征在特定点开始形成，而更复杂的模式则在后续的训练阶段中出现。特征归因分析揭示了特征演变与下游性能之间的因果关系。我们的特征级观察与以前关于Transformer的两阶段学习过程的发现高度一致，我们将这两个阶段分别称为统计学习阶段和特征学习阶段。我们的研究为跟踪语言模型学习动态中的细粒度表示进展开启了可能性。 

---
# LifeAlign: Lifelong Alignment for Large Language Models with Memory-Augmented Focalized Preference Optimization 

**Title (ZH)**: LifeAlign: 具有记忆增强聚焦偏好优化的终身对齐 

**Authors**: Junsong Li, Jie Zhou, Bihao Zhan, Yutao Yang, Qianjun Pan, Shilian Chen, Tianyu Huai, Xin Li, Qin Chen, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2509.17183)  

**Abstract**: Alignment plays a crucial role in Large Language Models (LLMs) in aligning with human preferences on a specific task/domain. Traditional alignment methods suffer from catastrophic forgetting, where models lose previously acquired knowledge when adapting to new preferences or domains. We introduce LifeAlign, a novel framework for lifelong alignment that enables LLMs to maintain consistent human preference alignment across sequential learning tasks without forgetting previously learned knowledge. Our approach consists of two key innovations. First, we propose a focalized preference optimization strategy that aligns LLMs with new preferences while preventing the erosion of knowledge acquired from previous tasks. Second, we develop a short-to-long memory consolidation mechanism that merges denoised short-term preference representations into stable long-term memory using intrinsic dimensionality reduction, enabling efficient storage and retrieval of alignment patterns across diverse domains. We evaluate LifeAlign across multiple sequential alignment tasks spanning different domains and preference types. Experimental results demonstrate that our method achieves superior performance in maintaining both preference alignment quality and knowledge retention compared to existing lifelong learning approaches. The codes and datasets will be released on GitHub. 

**Abstract (ZH)**: Lifelong Alignment Framework for Maintaining Human Preference Alignment and Knowledge Retention in Large Language Models 

---
# Prompt-with-Me: in-IDE Structured Prompt Management for LLM-Driven Software Engineering 

**Title (ZH)**: Prompt-with-Me: IDE内置结构化提示管理以驱动软件工程的提示技术 

**Authors**: Ziyou Li, Agnia Sergeyuk, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2509.17096)  

**Abstract**: Large Language Models are transforming software engineering, yet prompt management in practice remains ad hoc, hindering reliability, reuse, and integration into industrial workflows. We present Prompt-with-Me, a practical solution for structured prompt management embedded directly in the development environment. The system automatically classifies prompts using a four-dimensional taxonomy encompassing intent, author role, software development lifecycle stage, and prompt type. To enhance prompt reuse and quality, Prompt-with-Me suggests language refinements, masks sensitive information, and extracts reusable templates from a developer's prompt library. Our taxonomy study of 1108 real-world prompts demonstrates that modern LLMs can accurately classify software engineering prompts. Furthermore, our user study with 11 participants shows strong developer acceptance, with high usability (Mean SUS=73), low cognitive load (Mean NASA-TLX=21), and reported gains in prompt quality and efficiency through reduced repetitive effort. Lastly, we offer actionable insights for building the next generation of prompt management and maintenance tools for software engineering workflows. 

**Abstract (ZH)**: 大型语言模型正在重塑软件工程，但在实践中，提示管理仍缺乏系统性，影响了可靠性和重用性，并阻碍了与工业工作流的集成。我们提出了一种名为Prompt-with-Me的实用解决方案，该方案直接嵌入开发环境中，用于结构化的提示管理。该系统使用包含意图、作者角色、软件开发生命周期阶段和提示类型等四个维度的分类体系自动分类提示。为了提高提示的重用性和质量，Prompt-with-Me 提出了语言精炼建议、隐藏敏感信息，并从开发者的提示库中提取可重用模板。通过对1108个实际提示的研究表明，现代大型语言模型能够准确分类软件工程提示。此外，我们的用户研究显示，11名参与者对提示管理表现出强烈的接受度，使用该系统后的平均简易可使用性评分为73，平均认知负荷评分为21，并报告通过减少重复劳动提高了提示质量和效率。最后，我们提供了有关构建下一代软件工程工作流提示管理和维护工具的实际建议。 

---
# TactfulToM: Do LLMs Have the Theory of Mind Ability to Understand White Lies? 

**Title (ZH)**: TactfulToM: 大语言模型具备理解善意谎言的理论心智能力吗？ 

**Authors**: Yiwei Liu, Emma Jane Pretty, Jiahao Huang, Saku Sugawara  

**Link**: [PDF](https://arxiv.org/pdf/2509.17054)  

**Abstract**: While recent studies explore Large Language Models' (LLMs) performance on Theory of Mind (ToM) reasoning tasks, research on ToM abilities that require more nuanced social context is limited, such as white lies. We introduce TactfulToM, a novel English benchmark designed to evaluate LLMs' ability to understand white lies within real-life conversations and reason about prosocial motivations behind them, particularly when they are used to spare others' feelings and maintain social harmony. Our benchmark is generated through a multi-stage human-in-the-loop pipeline where LLMs expand manually designed seed stories into conversations to maintain the information asymmetry between participants necessary for authentic white lies. We show that TactfulToM is challenging for state-of-the-art models, which perform substantially below humans, revealing shortcomings in their ability to fully comprehend the ToM reasoning that enables true understanding of white lies. 

**Abstract (ZH)**: 尽管近期的研究探讨了大型语言模型（LLMs）在理论思维（ToM）推理任务上的表现，但对于需要更细腻社会情境的ToM能力研究有限，例如善意的谎言。我们提出了TactfulToM，一个新颖的英语基准，旨在评估LLMs在实际对话中理解善意的谎言及其背后促进社会和谐的利他动机的能力。我们的基准通过多阶段的人机交互流程生成，其中LLMs将人工设计的种子故事扩展为对话，以保持参与者之间的信息不对称，这是实现真实善意谎言所必需的。我们展示了TactfulToM对最先进的模型构成了挑战，这些模型的表现远低于人类，揭示了它们在全面理解能真正理解善意谎言的理论思维推理方面存在的不足。 

---
# The Transfer Neurons Hypothesis: An Underlying Mechanism for Language Latent Space Transitions in Multilingual LLMs 

**Title (ZH)**: 语言 laten 空间转换的神经元转移假说：多语言大语言模型中的潜在机制 

**Authors**: Hinata Tezuka, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2509.17030)  

**Abstract**: Recent studies have suggested a processing framework for multilingual inputs in decoder-based LLMs: early layers convert inputs into English-centric and language-agnostic representations; middle layers perform reasoning within an English-centric latent space; and final layers generate outputs by transforming these representations back into language-specific latent spaces. However, the internal dynamics of such transformation and the underlying mechanism remain underexplored. Towards a deeper understanding of this framework, we propose and empirically validate The Transfer Neurons Hypothesis: certain neurons in the MLP module are responsible for transferring representations between language-specific latent spaces and a shared semantic latent space. Furthermore, we show that one function of language-specific neurons, as identified in recent studies, is to facilitate movement between latent spaces. Finally, we show that transfer neurons are critical for reasoning in multilingual LLMs. 

**Abstract (ZH)**: 近期的研究提出了一种基于解码器的大型语言模型处理多语言输入的框架：早期层将输入转换为以英语为中心且语言无关的表现形式；中间层在以英语为中心的潜在空间中进行推理；最终层通过将这些表现形式转换回特定语言的潜在空间来生成输出。然而，这种转换的内部动态和其背后的机制仍需进一步探索。为了更深入地理解这一框架，我们提出了并实证验证了转移神经元假设：MLP模块中的某些神经元负责在特定语言的潜在空间与共享语义潜在空间之间转移表现形式。此外，我们展示了特定语言神经元的其中一个功能是促进潜在空间之间的切换。最后，我们证明转移神经元对于多语言大型语言模型的推理至关重要。 

---
# Adaptive Overclocking: Dynamic Control of Thinking Path Length via Real-Time Reasoning Signals 

**Title (ZH)**: 自适应超频：通过实时推理信号动态控制思考路径长度 

**Authors**: Shuhao Jiang, Songbo Wang, Yang Qiao, Chun Xu, Chaoyang Zheng, Shengyi Zhou, Huanjun Wang, Fangming Li, Cong Zhang, Jiyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17000)  

**Abstract**: Large Reasoning Models (LRMs) often suffer from computational inefficiency due to overthinking, where a fixed reasoning budget fails to match the varying complexity of tasks. To address this issue, we propose Adaptive Overclocking, a method that makes the overclocking hyperparameter $\alpha$ dynamic and context-aware. Our method adjusts reasoning speed in real time through two complementary signals: (1) token-level model uncertainty for fine-grained step-wise control, and (2) input complexity estimation for informed initialization. We implement this approach with three strategies: Uncertainty-Aware Alpha Scheduling (UA-$\alpha$S), Complexity-Guided Alpha Initialization (CG-$\alpha$I), and a Hybrid Adaptive Control (HAC) that combines both. Experiments on GSM8K, MATH, and SVAMP show that HAC achieves superior accuracy-latency trade-offs, reducing unnecessary computation on simple problems while allocating more resources to challenging ones. By mitigating overthinking, Adaptive Overclocking enhances both efficiency and overall reasoning performance. 

**Abstract (ZH)**: 自适应超频以缓解过度推理：Large Reasoning Models的自适应超频方法 

---
# Advancing Speech Understanding in Speech-Aware Language Models with GRPO 

**Title (ZH)**: 在具有GRPO的语音意识语言模型中推进语音理解 

**Authors**: Avishai Elmakies, Hagai Aronowitz, Nimrod Shabtay, Eli Schwartz, Ron Hoory, Avihu Dekel  

**Link**: [PDF](https://arxiv.org/pdf/2509.16990)  

**Abstract**: In this paper, we introduce a Group Relative Policy Optimization (GRPO)-based method for training Speech-Aware Large Language Models (SALLMs) on open-format speech understanding tasks, such as Spoken Question Answering and Automatic Speech Translation. SALLMs have proven highly effective for speech understanding tasks. GRPO has recently gained traction for its efficiency in training LLMs, and prior work has explored its application to SALLMs, primarily in multiple-choice tasks. Building on this, we focus on open-format tasks that better reflect the generative abilities of the models. Our approach leverages GRPO with BLEU as the reward signal to optimize SALLMs, and we demonstrate empirically that it surpasses standard SFT across several key metrics. Finally, we explore the potential of incorporating off-policy samples within GRPO for these tasks, highlighting avenues for further improvement and further research. 

**Abstract (ZH)**: 基于Group Relative Policy Optimization的Speech-Aware大型语言模型训练方法：以开放式语音理解任务为例 

---
# PTQTP: Post-Training Quantization to Trit-Planes for Large Language Models 

**Title (ZH)**: PTQTP: Post-Training Quantization to Trit-Planes for Large Language Models 

**Authors**: He Xiao, Runming Yang, Qingyao Yang, Wendong Xu, Zheng Li, Yupeng Su, Zhengwu Liu, Hongxia Yang, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2509.16989)  

**Abstract**: Post-training quantization (PTQ) of large language models (LLMs) to extremely low bit-widths remains challenging due to the fundamental trade-off between computational efficiency and model expressiveness. While existing ultra-low-bit PTQ methods rely on binary approximations or complex compensation mechanisms, they suffer from either limited representational capacity or computational overhead that undermines their efficiency gains. We introduce PTQ to Trit-Planes (PTQTP), the first ternary-weight PTQ framework that decomposes weight matrices into structured ternary {-1, 0, 1} trit-planes using 2x1.58-bit representation. PTQTP achieves multiplication-free inference, identical to 1-bit quantization, while maintaining superior expressiveness through its novel structured decomposition. Our approach provides: (1) a theoretically grounded progressive approximation algorithm ensuring global weight consistency; (2) model-agnostic deployment across diverse modern LLMs without architectural modifications; and (3) uniform ternary operations that eliminate the need for mixed-precision or compensation schemes. Comprehensive experiments across LLaMA3.x and Qwen3 model families (0.6B-70B parameters) demonstrate that PTQTP significantly outperforms existing low-bit PTQ methods, achieving 82.4% mathematical reasoning retention versus 0% for competing approaches. PTQTP approaches and sometimes surpasses 1.58-bit quantization-aware training performance while requiring only single-hour quantization compared to 10-14 GPU days for training-based methods. These results establish PTQTP as a practical solution for efficient LLM deployment in resource-constrained environments. 

**Abstract (ZH)**: 超低位宽训练后量化（PTQ）大语言模型（LLMs）至极低位宽仍然具有挑战性，由于计算效率与模型表征能力之间的根本权衡。尽管现有的超低位宽PTQ方法依赖于二值近似或复杂补偿机制，但它们要么表征能力有限，要么计算开销过大，削弱了其效率提升。我们引入了PTQ到三值平面（PTQTP）框架，这是首个使用2x1.58位表示将权重矩阵分解为结构化三值矩阵（-1, 0, 1）的三值权重PTQ框架。PTQTP实现了无乘法推理，类似于1比特量化，同时通过其新颖的结构分解保持了卓越的表达能力。我们的方法提供：（1）基于理论的逐步逼近算法，确保全局权重一致性；（2）在各种现代LLM中通用部署，无需架构修改；（3）统一的三值操作，消除了混合精度或补偿方案的需要。全面的实验展示了PTQTP在LLaMA3.x和Qwen3模型家族（0.6B-70B参数）中的优越性能，显著优于现有低位宽PTQ方法，相对于竞争方法实现了82.4%的数学推理保留率。此外，PTQTP在位宽感知训练性能相近的情况下，仅需单小时量化，而基于训练的方法则需要10-14个GPU天。这些结果确立了PTQTP在资源受限环境中高效大语言模型部署的实用解决方案。 

---
# AdaptiveGuard: Towards Adaptive Runtime Safety for LLM-Powered Software 

**Title (ZH)**: 自适应防护：面向LLM驱动软件的自适应运行时安全性 

**Authors**: Rui Yang, Michael Fu, Chakkrit Tantithamthavorn, Chetan Arora, Gunel Gulmammadova, Joey Chua  

**Link**: [PDF](https://arxiv.org/pdf/2509.16861)  

**Abstract**: Guardrails are critical for the safe deployment of Large Language Models (LLMs)-powered software. Unlike traditional rule-based systems with limited, predefined input-output spaces that inherently constrain unsafe behavior, LLMs enable open-ended, intelligent interactions--opening the door to jailbreak attacks through user inputs. Guardrails serve as a protective layer, filtering unsafe prompts before they reach the LLM. However, prior research shows that jailbreak attacks can still succeed over 70% of the time, even against advanced models like GPT-4o. While guardrails such as LlamaGuard report up to 95% accuracy, our preliminary analysis shows their performance can drop sharply--to as low as 12%--when confronted with unseen attacks. This highlights a growing software engineering challenge: how to build a post-deployment guardrail that adapts dynamically to emerging threats? To address this, we propose AdaptiveGuard, an adaptive guardrail that detects novel jailbreak attacks as out-of-distribution (OOD) inputs and learns to defend against them through a continual learning framework. Through empirical evaluation, AdaptiveGuard achieves 96% OOD detection accuracy, adapts to new attacks in just two update steps, and retains over 85% F1-score on in-distribution data post-adaptation, outperforming other baselines. These results demonstrate that AdaptiveGuard is a guardrail capable of evolving in response to emerging jailbreak strategies post deployment. We release our AdaptiveGuard and studied datasets at this https URL to support further research. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的软件安全部署中，边界条件至关重要 - 一项关于动态适应新兴威胁的自适应边界防护的研究 

---
# Comparing RAG and GraphRAG for Page-Level Retrieval Question Answering on Math Textbook 

**Title (ZH)**: 比较RAG和GraphRAG在数学教材页面级检索问答中的性能 

**Authors**: Eason Chen, Chuangji Li, Shizhuo Li, Conrad Borchers, Zimo Xiao, Chloe Qianhui Zhao, Jionghao Lin, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2509.16780)  

**Abstract**: Technology-enhanced learning environments often help students retrieve relevant learning content for questions arising during self-paced study. Large language models (LLMs) have emerged as novel aids for information retrieval during learning. While LLMs are effective for general-purpose question-answering, they typically lack alignment with the domain knowledge of specific course materials such as textbooks and slides. We investigate Retrieval-Augmented Generation (RAG) and GraphRAG, a knowledge graph-enhanced RAG approach, for page-level question answering in an undergraduate mathematics textbook. While RAG has been effective for retrieving discrete, contextually relevant passages, GraphRAG may excel in modeling interconnected concepts and hierarchical knowledge structures. We curate a dataset of 477 question-answer pairs, each tied to a distinct textbook page. We then compare the standard embedding-based RAG methods to GraphRAG for evaluating both retrieval accuracy-whether the correct page is retrieved-and generated answer quality via F1 scores. Our findings show that embedding-based RAG achieves higher retrieval accuracy and better F1 scores compared to GraphRAG, which tends to retrieve excessive and sometimes irrelevant content due to its entity-based structure. We also explored re-ranking the retrieved pages with LLM and observed mixed results, including performance drop and hallucinations when dealing with larger context windows. Overall, this study highlights both the promises and challenges of page-level retrieval systems in educational contexts, emphasizing the need for more refined retrieval methods to build reliable AI tutoring solutions in providing reference page numbers. 

**Abstract (ZH)**: 技术增强的学习环境有助于学生在自主学习过程中检索相关学习内容。大规模语言模型（LLMs）已成为学习过程中信息检索的新型辅助工具。虽然LLMs在通用问题回答方面表现出色，但它们通常缺乏与特定课程材料（如教科书和幻灯片）领域知识的对齐。我们研究了检索增强生成（RAG）及其基于知识图谱增强的GraphRAG方法在本科数学教科书中的页级问题回答效果。虽然RAG在检索离散的、上下文相关的段落方面效果显著，但GraphRAG可能在建模相互关联的概念和层次知识结构方面更优越。我们构建了一个包含477个问题-答案对的数据集，每个问题-答案对都对应教科书的不同页面。然后，我们将基于嵌入的标准RAG方法与GraphRAG进行了比较，用于评估检索准确性（是否检索到正确的页面）和生成答案的质量（通过F1分数）。我们的研究结果表明，基于嵌入的RAG在检索准确性上表现更好，且F1分数更高，而GraphRAG由于其基于实体的结构，往往会检索过多且有时相关性不强的内容。我们还探索了使用LLM重新排名检索页面，结果包括性能下降和处理较大上下文窗口时的虚构信息。总体而言，这项研究突显了教育环境中页级检索系统的优势和挑战，并强调了构建可靠的AI辅导解决方案时需要更精细的检索方法，以提供参考页码。 

---
# The Sound of Syntax: Finetuning and Comprehensive Evaluation of Language Models for Speech Pathology 

**Title (ZH)**: 句法之声：语言模型在语音病理学中的微调与综合评估 

**Authors**: Fagun Patel, Duc Q. Nguyen, Sang T. Truong, Jody Vaynshtok, Sanmi Koyejo, Nick Haber  

**Link**: [PDF](https://arxiv.org/pdf/2509.16765)  

**Abstract**: According to the U.S. National Institutes of Health, more than 3.4 million children experience speech disorders that require clinical intervention. The number of speech-language pathologists (SLPs) is roughly 20 times fewer than the number of affected children, highlighting a significant gap in children's care and a pressing need for technological support that improves the productivity of SLPs. State-of-the-art multimodal language models (MLMs) show promise for supporting SLPs, but their use remains underexplored largely due to a limited understanding of their performance in high-stakes clinical settings. To address this gap, we collaborate with domain experts to develop a taxonomy of real-world use cases of MLMs in speech-language pathologies. Building on this taxonomy, we introduce the first comprehensive benchmark for evaluating MLM across five core use cases, each containing 1,000 manually annotated data points. This benchmark includes robustness and sensitivity tests under various settings, including background noise, speaker gender, and accent. Our evaluation of 15 state-of-the-art MLMs reveals that no single model consistently outperforms others across all tasks. Notably, we find systematic disparities, with models performing better on male speakers, and observe that chain-of-thought prompting can degrade performance on classification tasks with large label spaces and narrow decision boundaries. Furthermore, we study fine-tuning MLMs on domain-specific data, achieving improvements of over 30% compared to base models. These findings highlight both the potential and limitations of current MLMs for speech-language pathology applications, underscoring the need for further research and targeted development. 

**Abstract (ZH)**: 根据美国国家卫生研究院的数据，超过340万儿童患有需要临床干预的语言障碍。言语语言病理学家（SLPs）的数量大约是受影响儿童数量的20倍，这凸显出儿童护理中的巨大缺口，并迫切需要提高SLPs工作效率的技术支持。前沿的多模态语言模型（MLMs）在支持SLPs方面显示出潜力，但由于对其在高风险临床环境中的性能理解有限，其应用仍待进一步探索。为填补这一缺口，我们与领域专家合作，开发了语言病理学中MLMs实际应用场景的分类体系。基于此分类体系，我们介绍了第一个涵盖五种核心应用场景的全面基准，每种场景包含1000个手动标注的数据点。该基准还包括在各种环境下进行的鲁棒性和敏感性测试，包括背景噪声、说话者性别和口音。我们对15个前沿的MLMs进行的评估表明，没有单一模型能在所有任务中持续表现出色。值得注意的是，我们发现系统性差异，模型在男性说话者上的表现更佳，并观察到链式思考提示对分类任务中的大标签空间和狭窄决策边界可能导致性能下降。此外，我们研究了在领域特定数据上 fine-tune MLMs，相对于基模型取得了超过30%的改进。这些发现突显了当前MLMs在言语language病理学应用中的潜在性和局限性，强调了进一步研究和针对性开发的必要性。 

---
# Design and Development of an Intelligent LLM-based LDAP Honeypot 

**Title (ZH)**: 基于LLM的LDAP蜜罐的设计与开发 

**Authors**: Javier Jiménez-Román, Florina Almenares-Mendoza, Alfonso Sánchez-Macián  

**Link**: [PDF](https://arxiv.org/pdf/2509.16682)  

**Abstract**: Cybersecurity threats continue to increase, with a growing number of previously unknown attacks each year targeting both large corporations and smaller entities. This scenario demands the implementation of advanced security measures, not only to mitigate damage but also to anticipate emerging attack trends. In this context, deception tools have become a key strategy, enabling the detection, deterrence, and deception of potential attackers while facilitating the collection of information about their tactics and methods. Among these tools, honeypots have proven their value, although they have traditionally been limited by rigidity and configuration complexity, hindering their adaptability to dynamic scenarios. The rise of artificial intelligence, and particularly general-purpose Large Language Models (LLMs), is driving the development of new deception solutions capable of offering greater adaptability and ease of use. This work proposes the design and implementation of an LLM-based honeypot to simulate an LDAP server, a critical protocol present in most organizations due to its central role in identity and access management. The proposed solution aims to provide a flexible and realistic tool capable of convincingly interacting with attackers, thereby contributing to early detection and threat analysis while enhancing the defensive capabilities of infrastructures against intrusions targeting this service. 

**Abstract (ZH)**: 网络威胁持续增加，每年出现越来越多未知攻击， targeting 各大小组织。在这种情况下，需要实施先进的安全措施，不仅为了减轻损害，还为了预见新兴的攻击趋势。在此背景下，诱骗工具已成为关键策略，能够检测、威慑和欺骗潜在攻击者，并有助于收集其战术和方法的信息。在这些工具中，蜜罐已经证明了其价值，尽管它们传统上受限于僵化和配置复杂性，妨碍了其在动态场景中的适应性。人工智能的兴起，特别是通用大语言模型（LLMs），正在推动新的诱骗解决方案的发展，这些解决方案能够提供更高的适应性和易用性。本研究提出基于大语言模型的蜜罐设计与实现，以模拟LDAP服务器，这一关键协议由于其在身份和访问管理中的核心作用而在大多数组织中普遍存在。所提出解决方案旨在提供一种灵活且逼真的工具，能够说服性地与攻击者互动，从而有助于早期检测和威胁分析，同时增强针对此服务入侵的防御能力。 

---
# AISTAT lab system for DCASE2025 Task6: Language-based audio retrieval 

**Title (ZH)**: AISTAT实验室系统用于DCASE2025任务6：基于语言的音频检索 

**Authors**: Hyun Jun Kim, Hyeong Yong Choi, Changwon Lim  

**Link**: [PDF](https://arxiv.org/pdf/2509.16649)  

**Abstract**: This report presents the AISTAT team's submission to the language-based audio retrieval task in DCASE 2025 Task 6. Our proposed system employs dual encoder architecture, where audio and text modalities are encoded separately, and their representations are aligned using contrastive learning. Drawing inspiration from methodologies of the previous year's challenge, we implemented a distillation approach and leveraged large language models (LLMs) for effective data augmentation techniques, including back-translation and LLM mix. Additionally, we incorporated clustering to introduce an auxiliary classification task for further finetuning. Our best single system achieved a mAP@16 of 46.62, while an ensemble of four systems reached a mAP@16 of 48.83 on the Clotho development test split. 

**Abstract (ZH)**: 本报告呈现了AISTAT团队对2025年DCASE任务6基于语言的音频检索挑战的提交。我们提出的系统采用双编码器架构，其中音频和文本模态分别编码，并通过对比学习对齐其表示。受到去年挑战方法论的启发，我们实施了一种蒸馏方法，并利用大型语言模型（LLMs）进行有效数据增强技术，包括反向翻译和LLM混合。此外，我们结合了聚类技术，引入了一个辅助分类任务以进行进一步微调。我们的最佳单系统在Clotho开发测试分割上的mAP@16为46.62，而四个系统的集成在Clotho开发测试分割上的mAP@16为48.83。 

---
# When Big Models Train Small Ones: Label-Free Model Parity Alignment for Efficient Visual Question Answering using Small VLMs 

**Title (ZH)**: 当大模型训练小模型：高效的视觉问答中无标注模型公平性对齐 

**Authors**: Abhirama Subramanyam Penamakuri, Navlika Singh, Piyush Arora, Anand Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2509.16633)  

**Abstract**: Large Vision-Language Models (L-VLMs) have demonstrated remarkable performance in various vision and language tasks, including visual question answering (VQA). However, their high computational cost makes them impractical for resource-constrained settings and inference-heavy applications. In contrast, Small Vision-Language Models (S-VLMs) offer efficiency but suffer from a significant performance gap compared to their larger counterparts. In this work, we introduce the Model Parity Aligner (MPA), a novel framework designed to systematically improve S-VLMs by leveraging unlabeled images and effective knowledge transfer from L-VLMs. Instead of traditional knowledge distillation methods that rely on labeled training data, MPA employs a strategic parity-based approach that precisely identifies the knowledge disparities between S-VLMs and L-VLMs, and optimizes training by targeting only these disparities. We conduct extensive experiments on four diverse VQA benchmarks, namely TextVQA, ST-VQA, ChartQA, and OKVQA, each of which requires specialized reasoning capabilities such as text recognition, chart interpretation, and commonsense and factual understanding. Our results demonstrate that MPA consistently enhances the performance of S-VLMs on all benchmarks, reducing the performance gap while maintaining computational efficiency. We make our code publicly available. 

**Abstract (ZH)**: 大规模视觉-语言模型（L-VLMs）在各种视觉和语言任务中展现了出色的表现，包括视觉问答（VQA）。然而，其高昂的计算成本使其在资源受限的环境中和推理密集型应用中不切实际。相比之下，小型视觉-语言模型（S-VLMs）虽然效率更高，但在性能上与大型模型存在显著差距。在此工作中，我们提出了模型 parity 对齐器（MPA），这是一种新型框架，旨在通过利用未标记的图像和从 L-VLMs 有效转移知识来系统地提升 S-VLMs。MPA 采用了一种基于 parity 的策略，精确识别 S-VLMs 和 L-VLMs 之间的知识差异，并通过仅针对这些差异进行优化训练。我们在四个不同的 VQA 数据集中进行了广泛的实验，分别是 TextVQA、ST-VQA、ChartQA 和 OKVQA，每个数据集都需要特定的推理能力，如文本识别、图表解释以及常识和事实理解。实验结果表明，MPA 一致地增强了 S-VLMs 在所有基准上的性能，缩小了性能差距并保持了计算效率。我们已公开发布我们的代码。 

---
# Audio-Conditioned Diffusion LLMs for ASR and Deliberation Processing 

**Title (ZH)**: 基于音频条件的扩散语言模型及其在ASR和推理处理中的应用 

**Authors**: Mengqi Wang, Zhan Liu, Zengrui Jin, Guangzhi Sun, Chao Zhang, Philip C. Woodland  

**Link**: [PDF](https://arxiv.org/pdf/2509.16622)  

**Abstract**: Diffusion-based large language models (DLLMs) have recently attracted growing interest as an alternative to autoregressive decoders. In this work, we present an empirical study on using the diffusion-based large language model LLaDA for automatic speech recognition (ASR). We first investigate its use as an external deliberation-based processing module for Whisper-LLaMA transcripts. By leveraging the bidirectional attention and denoising capabilities of LLaDA, we explore random masking, low-confidence masking, and semi-autoregressive strategies, showing that Whisper-LLaDA substantially reduces WER compared with the baseline. On LibriSpeech, the best cascade system achieves 2.25%/4.94% WER on test-clean/test-other, representing a 12.3% relative improvement over the Whisper-LLaMA baseline on the test-other split. In contrast, a plain-text LLaDA without acoustic features fails to improve accuracy, highlighting the importance of audio-conditioned embeddings. We further evaluate Whisper-LLaDA as a standalone decoder for ASR with diffusion-based and semi-autoregressive decoding. Most experimental configurations achieve faster inference than the Whisper-LLaMA baseline, although recognition accuracy is slightly lower. These findings offer an empirical view of diffusion-based LLMs for ASR and point to promising directions for improvements. 

**Abstract (ZH)**: 基于扩散的大型语言模型（DLLMs）近年来作为自回归解码器的替代方案吸引了越来越多的关注。在本工作中，我们对使用基于扩散的大型语言模型LLaDA进行自动语音识别（ASR）进行了实证研究。我们首先研究了其作为Whisper-LLaMA转录文稿外部推理模块的应用。通过利用LLaDA的双向注意力和去噪能力，我们探索了随机掩码、低置信度掩码和半自回归策略，结果显示，Whisper-LLaDA显著降低了WER，与基线相比，在LibriSpeech测试集中，最佳级联系统在test-clean和test-other上的WER分别为2.25%和4.94%，test-other分割上相对改进了12.3%。相比之下，缺乏声学特征的纯文本LLaDA未能提高准确性，强调了音频条件嵌入的重要性。我们进一步评估了Whisper-LLaDA作为独立解码器在基于扩散和半自回归解码的ASR中的应用。大多数实验配置的推理速度快于Whisper-LLaMA基线，尽管识别准确性略有下降。这些发现提供了基于扩散的大型语言模型在ASR中的实证观点，并指出了改进的潜在方向。 

---
# PruneCD: Contrasting Pruned Self Model to Improve Decoding Factuality 

**Title (ZH)**: PruneCD: 对比剪枝自模型以提高解码事实性 

**Authors**: Byeongho Yu, Changhun Lee, Jungyu Jin, Eunhyeok Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.16598)  

**Abstract**: To mitigate the hallucination problem in large language models, DoLa exploits early exit logits from the same model as a contrastive prior. However, we found that these early exit logits tend to be flat, low in magnitude, and fail to reflect meaningful contrasts. To address this, we propose PruneCD, a novel contrastive decoding method that constructs the amateur model via layer pruning rather than early exit. This design leads to more informative and well-aligned logits, enabling more effective contrastive decoding. Through qualitative and quantitative analyses, we demonstrate that PruneCD consistently improves factuality with minimal inference overhead, offering a robust and practical approach to mitigating hallucinations in LLMs. 

**Abstract (ZH)**: 为了缓解大规模语言模型中的幻觉问题，PruneCD 通过层剪枝构建对比解码方法，而不是依赖早期退出，从而构建业余模型，这种设计产生了更具信息量且更好的对齐的 logits，使得对比解码更加有效。通过定性和定量分析，我们证明 PruneCD 在最小化推理开销的同时一致地提高了事实性，提供了一种 robust 和实用的方法来缓解 LLM 中的幻觉问题。 

---
# Analyzing the Effects of Supervised Fine-Tuning on Model Knowledge from Token and Parameter Levels 

**Title (ZH)**: 分析监督微调对模型知识的影响从令牌和参数层面考察 

**Authors**: Junjie Ye, Yuming Yang, Yang Nan, Shuo Li, Qi Zhang, Tao Gui, Xuanjing Huang, Peng Wang, Zhongchao Shi, Jianping Fan  

**Link**: [PDF](https://arxiv.org/pdf/2509.16596)  

**Abstract**: Large language models (LLMs) acquire substantial world knowledge during pre-training, which is further shaped by post-training techniques such as supervised fine-tuning (SFT). However, the impact of SFT on a model's knowledge remains underexplored, limiting our ability to control knowledge change behavior in fine-tuned models. To address this gap, we evaluate closed-book question answering (CBQA) performance across five LLMs from the LLaMA-2 and LLaMA-3 families. Surprisingly, models fine-tuned on 1,920 samples perform up to 14% worse than those fine-tuned on only 240 samples. Furthermore, varying the level of knowledge mastery in the fine-tuning data leads to performance fluctuations of over 12%. To investigate these effects, we analyze model behavior at both the token and parameter levels. Our analysis reveals that up to 90% of parameter updates during SFT do not contribute to knowledge enhancement. Restoring these updates can improve performance on the CBQA task, depending on the characteristics of the fine-tuning data. These insights offer practical guidance for developing fine-tuning strategies that more effectively strengthen model knowledge. 

**Abstract (ZH)**: 大型语言模型（LLMs）在预训练过程中获得了大量的世界知识，后续的微调技术（如监督微调SFT）进一步塑造了这些知识。然而，SFT 对模型知识的影响仍缺乏探讨，限制了我们控制微调模型知识变化行为的能力。为填补这一空白，我们评估了LLaMA-2和LLaMA-3家族中五种模型的闭卷问答（CBQA）性能。令人惊讶的是，使用1,920个样本微调的模型相比仅使用240个样本微调的模型表现最多差14%。此外，微调数据中的知识掌握程度变化会导致性能波动超过12%。为了研究这些效应，我们从token和参数层面分析了模型行为。我们的分析揭示，在SFT过程中高达90%的参数更新不 contribution to知识增强。根据微调数据的特性，恢复这些更新可以提高CBQA任务的性能。这些见解为开发更有效地强化模型知识的微调策略提供了实用指导。 

---
# Benchmarking Contextual and Paralinguistic Reasoning in Speech-LLMs: A Case Study with In-the-Wild Data 

**Title (ZH)**: 基于现实世界数据的对话大型语言模型情境与副语言推理基准研究 

**Authors**: Qiongqiong Wang, Hardik Bhupendra Sailor, Tianchi Liu, Wenyu Zhang, Muhammad Huzaifah, Nattadaporn Lertcheva, Shuo Sun, Nancy F. Chen, Jinyang Wu, AiTi Aw  

**Link**: [PDF](https://arxiv.org/pdf/2509.16589)  

**Abstract**: Recent speech-LLMs have shown impressive performance in tasks like transcription and translation, yet they remain limited in understanding the paralinguistic aspects of speech crucial for social and emotional intelligence. We propose CP-Bench, a benchmark for evaluating speech-LLMs on contextual paralinguistic reasoning the integration of verbal content with non-verbal cues like emotion and prosody. The benchmark includes two curated question answering (QA) datasets requiring both linguistic and empathetic understanding. We evaluate state-of-the-art speech-LLMs from both open and closed-source models and perform a comprehensive analysis across different question types. The top two models were further analyzed under temperature tuning to understand its effect on this task. Our benchmark reveals a key gap in existing evaluations and offers insights into building more context-aware and emotionally intelligent speech-capable LLMs. 

**Abstract (ZH)**: 最近的语音大语言模型在转录和翻译等任务中表现 impressive，但在理解对于社会情感intelligence至关重要的语音副语言方面仍有限制。我们提出了 CP-Bench，一个用于评估语音大语言模型在上下文副语言推理中的基准，该推理整合了言语内容与情绪、语调等非言语线索。基准包括两个精心编纂的问题回答 (QA) 数据集，需要语言理解和共情理解。我们评估了来自开源和闭源模型的最新语音大语言模型，并进行了不同问题类型的全面分析。通过对前两名模型进行温度调优进一步分析，以理解其对任务的影响。我们的基准揭示了现有评估中的一个关键缺口，并提供了构建更上下文意识和情感智能的语音大语言模型的见解。 

---
# From Scores to Steps: Diagnosing and Improving LLM Performance in Evidence-Based Medical Calculations 

**Title (ZH)**: 从分数到步骤：诊断并提升LLM在基于证据的医疗计算中的性能 

**Authors**: Benlu Wang, Iris Xia, Yifan Zhang, Junda Wang, Feiyun Ouyang, Shuo Han, Arman Cohan, Hong Yu, Zonghai Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.16584)  

**Abstract**: Large language models (LLMs) have demonstrated promising performance on medical benchmarks; however, their ability to perform medical calculations, a crucial aspect of clinical decision-making, remains underexplored and poorly evaluated. Existing benchmarks often assess only the final answer with a wide numerical tolerance, overlooking systematic reasoning failures and potentially causing serious clinical misjudgments. In this work, we revisit medical calculation evaluation with a stronger focus on clinical trustworthiness. First, we clean and restructure the MedCalc-Bench dataset and propose a new step-by-step evaluation pipeline that independently assesses formula selection, entity extraction, and arithmetic computation. Under this granular framework, the accuracy of GPT-4o drops from 62.7% to 43.6%, revealing errors masked by prior evaluations. Second, we introduce an automatic error analysis framework that generates structured attribution for each failure mode. Human evaluation confirms its alignment with expert judgment, enabling scalable and explainable diagnostics. Finally, we propose a modular agentic pipeline, MedRaC, that combines retrieval-augmented generation and Python-based code execution. Without any fine-tuning, MedRaC improves the accuracy of different LLMs from 16.35% up to 53.19%. Our work highlights the limitations of current benchmark practices and proposes a more clinically faithful methodology. By enabling transparent and transferable reasoning evaluation, we move closer to making LLM-based systems trustworthy for real-world medical applications. 

**Abstract (ZH)**: 大型语言模型在医学计算评估中的临床可信性研究 

---
# Rethinking the Role of Text Complexity in Language Model Pretraining 

**Title (ZH)**: 重新思考文本复杂性在语言模型预训练中的作用 

**Authors**: Dan John Velasco, Matthew Theodore Roque  

**Link**: [PDF](https://arxiv.org/pdf/2509.16551)  

**Abstract**: Improving pretraining data quality and size is known to boost downstream performance, but the role of text complexity is less explored. Text complexity refers to how hard a text is to read, and is typically estimated from surface cues such as sentence length, word choice, and sentence structure. We reduce surface-level complexity--shorter sentences, simpler words, simpler structure--while keeping core text content close to constant, and ask: (1) How does complexity affect language modeling across model sizes? (2) Can useful representations be learned from simpler text alone? (3) How does pretraining text complexity influence downstream language understanding? To answer these questions, we simplify human-written texts using a large language model, then pretrain causal models (28M-500M) from scratch on both original and simplified data, and evaluate them in finetuning and zero-shot setups. We find that perplexity is sensitive to the interaction between model capacity and text complexity--smaller models degrade far less on simpler texts--while text complexity has little impact on finetuning evaluations, with zero-shot evaluations indicating that simpler texts benefit performance on linguistic knowledge tasks, whereas more complex texts favor tasks requiring world knowledge and entity tracking. 

**Abstract (ZH)**: 改进预训练数据的质量和规模已被证明能提升下游性能，但文本复杂性的作用尚未得到充分探索。 

---
# InteGround: On the Evaluation of Verification and Retrieval Planning in Integrative Grounding 

**Title (ZH)**: InteGround: 关于整合 grounding 中验证与检索规划的评估 

**Authors**: Cheng Jiayang, Qianqian Zhuang, Haoran Li, Chunkit Chan, Xin Liu, Lin Qiu, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.16534)  

**Abstract**: Grounding large language models (LLMs) in external knowledge sources is a promising method for faithful prediction. While existing grounding approaches work well for simple queries, many real-world information needs require synthesizing multiple pieces of evidence. We introduce "integrative grounding" -- the challenge of retrieving and verifying multiple inter-dependent pieces of evidence to support a hypothesis query. To systematically study this problem, we repurpose data from four domains for evaluating integrative grounding capabilities. Our investigation reveals two critical findings: First, in groundedness verification, while LLMs are robust to redundant evidence, they tend to rationalize using internal knowledge when information is incomplete. Second, in examining retrieval planning strategies, we find that undirected planning can degrade performance through noise introduction, while premise abduction emerges as a promising approach due to its logical constraints. Additionally, LLMs' zero-shot self-reflection capabilities consistently improve grounding quality. These insights provide valuable direction for developing more effective integrative grounding systems. 

**Abstract (ZH)**: 将大型语言模型与外部知识源接地是一种可靠预测的有前途的方法。虽然现有的接地方法对简单的查询效果良好，但许多实际的信息需求需要综合多份证据。我们介绍了“整合接地”——从多个相互依赖的证据检索和验证以支持假设查询的挑战。为了系统地研究这一问题，我们重用了四个领域中的数据来评估整合接地能力。我们的研究揭示了两个关键发现：首先，在接地验证中，尽管LLMs对冗余证据具有鲁棒性，但在信息不完整时，它们倾向于使用内部知识进行合理化解释。其次，在研究检索规划策略时，我们发现无向规划可通过引入噪声降低性能，而前提 abduction 由于其逻辑约束被证明是一种有希望的方法。此外，LLMs的零样本自我反思能力一致地提高了接地质量。这些洞察为开发更有效的整合接地系统指明了方向。 

---
# AIPsychoBench: Understanding the Psychometric Differences between LLMs and Humans 

**Title (ZH)**: AIPsychoBench: LLMs与人类的心理测量差异理解 

**Authors**: Wei Xie, Shuoyoucheng Ma, Zhenhua Wang, Enze Wang, Kai Chen, Xiaobing Sun, Baosheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16530)  

**Abstract**: Large Language Models (LLMs) with hundreds of billions of parameters have exhibited human-like intelligence by learning from vast amounts of internet-scale data. However, the uninterpretability of large-scale neural networks raises concerns about the reliability of LLM. Studies have attempted to assess the psychometric properties of LLMs by borrowing concepts from human psychology to enhance their interpretability, but they fail to account for the fundamental differences between LLMs and humans. This results in high rejection rates when human scales are reused directly. Furthermore, these scales do not support the measurement of LLM psychological property variations in different languages. This paper introduces AIPsychoBench, a specialized benchmark tailored to assess the psychological properties of LLM. It uses a lightweight role-playing prompt to bypass LLM alignment, improving the average effective response rate from 70.12% to 90.40%. Meanwhile, the average biases are only 3.3% (positive) and 2.1% (negative), which are significantly lower than the biases of 9.8% and 6.9%, respectively, caused by traditional jailbreak prompts. Furthermore, among the total of 112 psychometric subcategories, the score deviations for seven languages compared to English ranged from 5% to 20.2% in 43 subcategories, providing the first comprehensive evidence of the linguistic impact on the psychometrics of LLM. 

**Abstract (ZH)**: 大规模语言模型（LLMs）拥有数十亿参数，通过学习大规模互联网数据展现出类似人类的智能。然而，大规模神经网络的不可解释性引发了对LLM可靠性的担忧。研究尝试通过借用人类心理学的概念来评估LLM的心理测量属性，以提高其可解释性，但未能考虑到LLM与人类之间的根本差异。这导致直接重用人类尺度时出现高拒绝率。此外，这些尺度不支持不同语言下LLM心理属性变异性的测量。本文介绍了AIPsychoBench，这是一种专门针对评估LLM心理属性的基准。它使用轻量级的角色扮演提示来绕过LLM对齐，将平均有效响应率从70.12%提高到90.40%。同时，平均偏差仅为3.3%（正面）和2.1%（负面），显著低于传统突破提示引起的大约9.8%和6.9%的偏差。更重要的是，在总共112个心理测量子类别中，与英语相比，七种语言在43个子类别中的得分偏差范围从5%到20.2%，首次提供了语言对LLM心理测量影响的全面证据。 

---
# Synergies between Federated Foundation Models and Smart Power Grids 

**Title (ZH)**: 联邦基础模型与智能电网的协同效应 

**Authors**: Seyyedali Hosseinalipour, Shimiao Li, Adedoyin Inaolaji, Filippo Malandra, Luis Herrera, Nicholas Mastronarde  

**Link**: [PDF](https://arxiv.org/pdf/2509.16496)  

**Abstract**: The recent emergence of large language models (LLMs) such as GPT-3 has marked a significant paradigm shift in machine learning. Trained on massive corpora of data, these models demonstrate remarkable capabilities in language understanding, generation, summarization, and reasoning, transforming how intelligent systems process and interact with human language. Although LLMs may still seem like a recent breakthrough, the field is already witnessing the rise of a new and more general category: multi-modal, multi-task foundation models (M3T FMs). These models go beyond language and can process heterogeneous data types/modalities, such as time-series measurements, audio, imagery, tabular records, and unstructured logs, while supporting a broad range of downstream tasks spanning forecasting, classification, control, and retrieval. When combined with federated learning (FL), they give rise to M3T Federated Foundation Models (FedFMs): a highly recent and largely unexplored class of models that enable scalable, privacy-preserving model training/fine-tuning across distributed data sources. In this paper, we take one of the first steps toward introducing these models to the power systems research community by offering a bidirectional perspective: (i) M3T FedFMs for smart grids and (ii) smart grids for FedFMs. In the former, we explore how M3T FedFMs can enhance key grid functions, such as load/demand forecasting and fault detection, by learning from distributed, heterogeneous data available at the grid edge in a privacy-preserving manner. In the latter, we investigate how the constraints and structure of smart grids, spanning energy, communication, and regulatory dimensions, shape the design, training, and deployment of M3T FedFMs. 

**Abstract (ZH)**: Recent Emergence of Multi-Modal Multi-Task Federated Foundation Models in Smart Grids Research 

---
# Can an Individual Manipulate the Collective Decisions of Multi-Agents? 

**Title (ZH)**: 个体能否操控多智能体系统的集体决策？ 

**Authors**: Fengyuan Liu, Rui Zhao, Shuo Chen, Guohao Li, Philip Torr, Lei Han, Jindong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16494)  

**Abstract**: Individual Large Language Models (LLMs) have demonstrated significant capabilities across various domains, such as healthcare and law. Recent studies also show that coordinated multi-agent systems exhibit enhanced decision-making and reasoning abilities through collaboration. However, due to the vulnerabilities of individual LLMs and the difficulty of accessing all agents in a multi-agent system, a key question arises: If attackers only know one agent, could they still generate adversarial samples capable of misleading the collective decision? To explore this question, we formulate it as a game with incomplete information, where attackers know only one target agent and lack knowledge of the other agents in the system. With this formulation, we propose M-Spoiler, a framework that simulates agent interactions within a multi-agent system to generate adversarial samples. These samples are then used to manipulate the target agent in the target system, misleading the system's collaborative decision-making process. More specifically, M-Spoiler introduces a stubborn agent that actively aids in optimizing adversarial samples by simulating potential stubborn responses from agents in the target system. This enhances the effectiveness of the generated adversarial samples in misleading the system. Through extensive experiments across various tasks, our findings confirm the risks posed by the knowledge of an individual agent in multi-agent systems and demonstrate the effectiveness of our framework. We also explore several defense mechanisms, showing that our proposed attack framework remains more potent than baselines, underscoring the need for further research into defensive strategies. 

**Abstract (ZH)**: 个体大型语言模型（LLMs）在医疗保健和法律等领域展示了显著的能力。近期的研究还表明，协调的多智能体系统通过合作可以增强决策和推理能力。然而，由于个体LLM的脆弱性和多智能体系统中难以访问所有智能体的问题，一个关键问题出现了：如果攻击者只知道一个智能体，他们是否仍然能够生成能够误导集体决策的对抗样本？为探索这一问题，我们将其表述为一个信息不完全的游戏，攻击者只知道一个目标智能体，而缺乏对系统中其他智能体的知识。在此表述基础上，我们提出了M-Spoiler框架，该框架模拟多智能体系统中智能体之间的交互以生成对抗样本，随后使用这些样本操纵目标系统中的目标智能体，误导系统的协作决策过程。具体而言，M-Spoiler引入了一个顽固智能体，它积极地通过模拟目标系统中智能体的潜在顽固响应来优化对抗样本，从而增强了生成的对抗样本误导系统的有效性。通过在多种任务上进行广泛实验，我们的研究结果证实了多智能体系统中个体智能体知识带来的风险，并展示了我们框架的有效性。我们还探讨了几种防御机制，表明我们所提出的攻击框架比基线更为有效，突显了进一步研究防御策略的必要性。 

---
# The Oracle Has Spoken: A Multi-Aspect Evaluation of Dialogue in Pythia 

**Title (ZH)**: 先知已经发言：Pythia中对话的多方面评估 

**Authors**: Zixun Chen, Petr Babkin, Akshat Gupta, Gopala Anumanchipalli, Xiaomo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16487)  

**Abstract**: Dialogue is one of the landmark abilities of large language models (LLMs). Despite its ubiquity, few studies actually distinguish specific ingredients underpinning dialogue behavior emerging during post-training. We employ a comprehensive suite of model-based metrics, each targeting a distinct fine-grained aspect of dialogue, motivated by linguistic theory. We evaluate how the performance of pre-trained Pythia models changes with respect to each of those dimensions, depending on model size and as a result of supervised fine-tuning on conversational datasets. We observe only a mild impact of raw model size on most metrics, whereas fine-tuning quickly saturates the scores for all but the smallest models tested. Somewhat contrary to our expectations, many metrics show very similar trends, especially if they are all rooted in the same evaluator model, which raises the question of their reliability in measuring a specific dimension. To that end, we conduct additional analyses of score distributions, metric correlations, and term frequencies in generated responses to help explain our observations. 

**Abstract (ZH)**: 对话是大型语言模型（LLMs）的一项重要能力。尽管对话无处不在，但很少有研究区分出促使对话行为在后训练中出现的具体构成要素。我们利用一系列模型基础度量方法，每种方法针对对话的某一细粒度方面，受语言学理论的启发。我们评估预训练的Pythia模型在每个维度上的表现随模型大小的变化，以及监督式微调对会话数据集的影响。我们观察到，大多数度量的原始模型大小影响相对温和，而微调迅速对所有较小的模型达到饱和分数。与我们的预期相反，许多度量显示出非常相似的趋势，尤其是在它们都基于相同的评估模型的情况下，这引发了这些度量可靠性在测量特定维度的可靠性问题。为此，我们进行了额外的评分分布分析、度量相关性分析和生成回答中术语频率分析，以解释我们的观察结果。 

---
# Implicit Behavioral Alignment of Language Agents in High-Stakes Crowd Simulations 

**Title (ZH)**: 高风险人群模拟中语言代理的隐式行为对齐 

**Authors**: Yunzhe Wang, Gale M. Lucas, Burcin Becerik-Gerber, Volkan Ustun  

**Link**: [PDF](https://arxiv.org/pdf/2509.16457)  

**Abstract**: Language-driven generative agents have enabled large-scale social simulations with transformative uses, from interpersonal training to aiding global policy-making. However, recent studies indicate that generative agent behaviors often deviate from expert expectations and real-world data--a phenomenon we term the Behavior-Realism Gap. To address this, we introduce a theoretical framework called Persona-Environment Behavioral Alignment (PEBA), formulated as a distribution matching problem grounded in Lewin's behavior equation stating that behavior is a function of the person and their environment. Leveraging PEBA, we propose PersonaEvolve (PEvo), an LLM-based optimization algorithm that iteratively refines agent personas, implicitly aligning their collective behaviors with realistic expert benchmarks within a specified environmental context. We validate PEvo in an active shooter incident simulation we developed, achieving an 84% average reduction in distributional divergence compared to no steering and a 34% improvement over explicit instruction baselines. Results also show PEvo-refined personas generalize to novel, related simulation scenarios. Our method greatly enhances behavioral realism and reliability in high-stakes social simulations. More broadly, the PEBA-PEvo framework provides a principled approach to developing trustworthy LLM-driven social simulations. 

**Abstract (ZH)**: 语言驱动的生成代理使大规模社会模拟成为可能，并在人际培训和辅助全球政策制定等方面产生了变革性的影响。然而，最近的研究表明，生成代理的行为常常偏离专家预期和现实世界的数据——我们称这一现象为行为现实主义差距。为了解决这一问题，我们提出了一个名为 Persona-Environment Behavioral Alignment (PEBA) 的理论框架，该框架基于Lewin的行为方程，行为是人和环境的函数，并形成了一种分布匹配问题。利用PEBA，我们提出了基于LLM的优化算法PersonaEvolve（PEvo），该算法迭代地精化代理人格，隐式地在其特定的环境背景下使其集体行为与现实专家基准保持一致。我们在开发的一个主动射击事件模拟中验证了PEvo，结果显示与无干预相比，PEvo实现了84%的分布差异平均减少，与显式指令基准相比提高了34%。结果还表明，PEvo精化的角色可以泛化到新的相关模拟场景中。该方法极大地提高了高风险社会模拟中的行为现实主义和可靠性。更广泛地说，PEBA-PEvo框架提供了一种原则性的方法来开发可信的LLM驱动的社会模拟。 

---
# LightCode: Compiling LLM Inference for Photonic-Electronic Systems 

**Title (ZH)**: LightCode: 编译大型语言模型推理以适用于光电子系统 

**Authors**: Ryan Tomich, Zhizhen Zhong, Dirk Englund  

**Link**: [PDF](https://arxiv.org/pdf/2509.16443)  

**Abstract**: The growing demand for low-latency, energy-efficient inference in large language models (LLMs) has catalyzed interest in heterogeneous architectures. While GPUs remain dominant, they are poorly suited for integration with emerging domain-specific accelerators like the Photonic Tensor Units (PTUs), which offer low-power, high-throughput linear computation. This motivates hybrid compilation strategies that combine photonic and electronic resources. We present LightCode, a compiler framework and simulator for mapping LLM inference workloads across hybrid photonic-electronic systems. LightCode introduces the Stacked Graph, an intermediate representation that encodes multiple hardware-specific realizations of each tensor operation. Hardware assignment is formulated as a constrained subgraph selection problem optimized for latency or energy under parametric cost models. We evaluate LightCode on the prefill stage of GPT-2 and Llama-7B showing that under our workload and hardware assumptions, (i) Photonic hardware reduced energy by up to 50% in our simulated workloads at maximum sequence length; (ii) multiplexing and assignment strategy yielded latency improvements exceeding 10x; and (iii) Optimizing for latency or energy resulted in distinct hardware mappings in our simulations. LightCode offers a module, foundational framework and simulator for compiling LLMs to emerging photonic accelerators. 

**Abstract (ZH)**: 低延迟、高能效的大语言模型推理对异构架构的需求推动了研究兴趣。Photonic Tensor Units (PTUs) 等新兴领域特定加速器的低功耗和高吞吐量线性计算特性促使了光电混合资源的混合编译策略。我们 presents LightCode，一种用于映射大语言模型推理工作负载到光电混合系统的编译框架和模拟器。LightCode 引入了堆叠图，这是一种中间表示，编码了每个张量操作的多种硬件特定实现。硬件分配问题被表述为在约束子图选择下优化延迟或能量的有参数成本模型。我们在 GPT-2 和 Llama-7B 的预填充阶段评估了 LightCode，结果显示，在我们的工作负载和硬件假设下，(i) 光子硬件在最大序列长度下的模拟工作负载中最高可降低 50% 的能量；(ii) 复用和分配策略带来了超过 10 倍的延迟改进；(iii) 以延迟或能量为目标优化导致了在模拟中具有差异性的硬件映射。LightCode 提供了一种模块化、基础框架和模拟器，用于将大语言模型编译到新兴的光子加速器。 

---
# Pico: A Modular Framework for Hypothesis-Driven Small Language Model Research 

**Title (ZH)**: Pico：一种基于假设驱动的小型语言模型研究模块化框架 

**Authors**: Richard Diehl Martinez, David Demitri Africa, Yuval Weiss, Suchir Salhan, Ryan Daniels, Paula Buttery  

**Link**: [PDF](https://arxiv.org/pdf/2509.16413)  

**Abstract**: Building language models (LMs), especially small and medium ones, remains more art than science. While large LMs often improve by sheer scale, it is still unclear why many design choices work. For small LMs, this uncertainty is more limiting: tight parameter budgets make each decision critical, yet researchers still lack systematic, scientific ways to test and refine new ideas.
We introduce Pico, a lightweight, modular framework that enables systematic, hypothesis-driven research for small and medium-scale language model development. Pico consists of two libraries that together provide a practical sandbox where researchers can make targeted changes to a model's architecture or training procedures and directly observe their effects on the model's behavior. To support reproducible experimentation, we also release a suite of baseline models, pico-decoder, trained under standardized conditions and open-sourced for the community. Case studies highlight how Pico can support iterative small LM design and analysis. 

**Abstract (ZH)**: 构建语言模型（LM），尤其是在构建小型和中型LM时，依然更多地依赖于艺术而非科学。尽管大型LM往往通过规模效应而提升，但许多设计选择为何有效仍不清楚。对于小型LM而言，这种不确定性更为关键：紧凑的参数预算使得每个决策都至关重要，但研究者们仍然缺乏系统且科学的方法来测试和优化新想法。
我们引入Pico，一个轻量级且模块化的框架，能够促进对小型和中型语言模型开发的系统性和假设驱动的研究。Pico包含两个库，共同提供了一个实用的实验沙盒，研究人员可以在其中对模型架构或训练程序进行针对性的修改，并直接观察这些修改对模型行为的影响。为了支持可再现的实验，我们还推出了一个基准模型套件pico-decoder，并在标准化条件下进行训练，开源供社区使用。案例研究展示了Pico如何支持迭代的小型LM设计和分析。 

---
# Evaluating Behavioral Alignment in Conflict Dialogue: A Multi-Dimensional Comparison of LLM Agents and Humans 

**Title (ZH)**: 评估冲突对话中行为一致性的比较：LLM代理与人类的多维度分析 

**Authors**: Deuksin Kwon, Kaleen Shrestha, Bin Han, Elena Hayoung Lee, Gale Lucas  

**Link**: [PDF](https://arxiv.org/pdf/2509.16394)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in socially complex, interaction-driven tasks, yet their ability to mirror human behavior in emotionally and strategically complex contexts remains underexplored. This study assesses the behavioral alignment of personality-prompted LLMs in adversarial dispute resolution by simulating multi-turn conflict dialogues that incorporate negotiation. Each LLM is guided by a matched Five-Factor personality profile to control for individual variation and enhance realism. We evaluate alignment across three dimensions: linguistic style, emotional expression (e.g., anger dynamics), and strategic behavior. GPT-4.1 achieves the closest alignment with humans in linguistic style and emotional dynamics, while Claude-3.7-Sonnet best reflects strategic behavior. Nonetheless, substantial alignment gaps persist. Our findings establish a benchmark for alignment between LLMs and humans in socially complex interactions, underscoring both the promise and the limitations of personality conditioning in dialogue modeling. 

**Abstract (ZH)**: 大型语言模型（LLMs）在社会复杂、互动驱动的任务中日益普及，但它们在情绪和策略复杂背景下模仿人类行为的能力仍待探索。本研究通过模拟包含谈判的多轮冲突对话来评估人格提示下的LLMs在 adversarial 纠纷解决中的行为一致性。每种LLM均由匹配的五因素人格特征指导，以控制个体差异并增强真实性。我们从语言风格、情感表达（如愤怒动态）和战略性行为三个维度评估一致性。GPT-4.1在语言风格和情感动态方面与人类最为一致，而Claude-3.7-Sonnet在战略行为上表现最佳。然而，一致性缺口仍然显著。我们的研究为LLMs在社会复杂互动中与人类的一致性设定了基准，强调了人格在对话建模中既有潜力也有限制。 

---
# Overhearing LLM Agents: A Survey, Taxonomy, and Roadmap 

**Title (ZH)**: 监听LLM代理：综述、分类和路线图 

**Authors**: Andrew Zhu, Chris Callison-Burch  

**Link**: [PDF](https://arxiv.org/pdf/2509.16325)  

**Abstract**: Imagine AI assistants that enhance conversations without interrupting them: quietly providing relevant information during a medical consultation, seamlessly preparing materials as teachers discuss lesson plans, or unobtrusively scheduling meetings as colleagues debate calendars. While modern conversational LLM agents directly assist human users with tasks through a chat interface, we study this alternative paradigm for interacting with LLM agents, which we call "overhearing agents." Rather than demanding the user's attention, overhearing agents continuously monitor ambient activity and intervene only when they can provide contextual assistance. In this paper, we present the first analysis of overhearing LLM agents as a distinct paradigm in human-AI interaction and establish a taxonomy of overhearing agent interactions and tasks grounded in a survey of works on prior LLM-powered agents and exploratory HCI studies. Based on this taxonomy, we create a list of best practices for researchers and developers building overhearing agent systems. Finally, we outline the remaining research gaps and reveal opportunities for future research in the overhearing paradigm. 

**Abstract (ZH)**: 设想不打断对话的智能助手中的增强对话：在医疗咨询中安静地提供相关资料，在教师讨论教学计划时无缝准备材料，在同事商讨日历时默默地安排会议。虽然现代对话型大规模语言模型直接通过聊天界面辅助人类用户完成任务，我们研究了与这类语言模型交互的另一种范式，即“旁听代理”。与要求用户注意不同，旁听代理持续监控周围活动，并仅在能够提供上下文相关帮助时进行干预。在本文中，我们首次对旁听代理作为人类-人工智能交互中的一种独特范式进行了分析，并基于之前语言模型驱动代理的工作和探索性人机交互研究，建立了旁听代理交互和任务的分类体系。基于这一分类体系，我们为构建旁听代理系统的研究人员和开发者制定了最佳实践清单。最后，我们概述了该范式下仍然存在的研究空白，并揭示了未来研究的机会。 

---
# How Large Language Models are Designed to Hallucinate 

**Title (ZH)**: 大型语言模型的设计目的：生成幻觉 

**Authors**: Richard Ackermann, Simeon Emanuilov  

**Link**: [PDF](https://arxiv.org/pdf/2509.16297)  

**Abstract**: Large language models (LLMs) achieve remarkable fluency across linguistic and reasoning tasks but remain systematically prone to hallucination. Prevailing accounts attribute hallucinations to data gaps, limited context, or optimization errors. We argue instead that hallucination is a structural outcome of the transformer architecture. As coherence engines, transformers are compelled to produce fluent continuations, with self-attention simulating the relational structure of meaning but lacking the existential grounding of temporality, mood, and care that stabilizes human understanding. On this basis, we distinguish ontological hallucination, arising when continuations require disclosure of beings in world, and residual reasoning hallucination, where models mimic inference by recycling traces of human reasoning in text. We illustrate these patterns through case studies aligned with Heideggerian categories and an experiment across twelve LLMs showing how simulated "self-preservation" emerges under extended prompts. Our contribution is threefold: (1) a comparative account showing why existing explanations are insufficient; (2) a predictive taxonomy of hallucination linked to existential structures with proposed benchmarks; and (3) design directions toward "truth-constrained" architectures capable of withholding or deferring when disclosure is absent. We conclude that hallucination is not an incidental defect but a defining limit of transformer-based models, an outcome scaffolding can mask but never resolve. 

**Abstract (ZH)**: 大型语言模型（LLMs）在语言和推理任务中表现出色，但仍然系统性地容易产生幻觉。我们argue幻觉是变压器架构的结构性结果。作为 coherence 工具，变压器被迫生成流畅的延续，自注意力模拟了意义的关系结构，但缺乏人类理解中稳定存在的时间性、语气和关怀。基于此，我们将幻觉分为两类：旨在揭示世界中的存在实体的本体论幻觉，以及通过借用文本中的人类推理痕迹来模仿推理的残余推理幻觉。我们通过与海德格尔范畴相一致的案例研究以及涉及12个LLM的实验来说明这些模式，展示了在扩展提示下模拟“自我保存”的出现。我们的贡献包括三个方面：（1）一个比较性的解释，说明现有解释的不足；（2）与存在结构相关的预测性分类及其提议的基准；以及（3）设计“真理约束”架构的方向，这些架构能够在信息缺失时限制或推迟。我们得出结论，幻觉不仅是偶然的缺陷，而是基于变压器模型的定义性限制，这一限制结构可能掩盖但无法解决。 

---
# Robust LLM Training Infrastructure at ByteDance 

**Title (ZH)**: ByteDance稳健的大规模语言模型训练基础设施 

**Authors**: Borui Wan, Gaohong Liu, Zuquan Song, Jun Wang, Yun Zhang, Guangming Sheng, Shuguang Wang, Houmin Wei, Chenyuan Wang, Weiqiang Lou, Xi Yang, Mofan Zhang, Kaihua Jiang, Cheng Ren, Xiaoyun Zhi, Menghan Yu, Zhe Nan, Zhuolin Zheng, Baoquan Zhong, Qinlong Wang, Huan Yu, Jinxin Chi, Wang Zhang, Yuhan Li, Zixian Du, Sida Zhao, Yongqiang Zhang, Jingzhe Tang, Zherui Liu, Chuan Wu, Yanghua Peng, Haibin Lin, Wencong Xiao, Xin Liu, Liang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16293)  

**Abstract**: The training scale of large language models (LLMs) has reached tens of thousands of GPUs and is still continuously expanding, enabling faster learning of larger models. Accompanying the expansion of the resource scale is the prevalence of failures (CUDA error, NaN values, job hang, etc.), which poses significant challenges to training stability. Any large-scale LLM training infrastructure should strive for minimal training interruption, efficient fault diagnosis, and effective failure tolerance to enable highly efficient continuous training. This paper presents ByteRobust, a large-scale GPU infrastructure management system tailored for robust and stable training of LLMs. It exploits the uniqueness of LLM training process and gives top priorities to detecting and recovering failures in a routine manner. Leveraging parallelisms and characteristics of LLM training, ByteRobust enables high-capacity fault tolerance, prompt fault demarcation, and localization with an effective data-driven approach, comprehensively ensuring continuous and efficient training of LLM tasks. ByteRobust is deployed on a production GPU platform with over 200,000 GPUs and achieves 97% ETTR for a three-month training job on 9,600 GPUs. 

**Abstract (ZH)**: 面向大规模语言模型训练的大规模GPU基础设施管理系统：ByteRobust 

---
# SecureFixAgent: A Hybrid LLM Agent for Automated Python Static Vulnerability Repair 

**Title (ZH)**: SecureFixAgent: 一种用于自动化Python静态漏洞修复的混合LLM代理 

**Authors**: Jugal Gajjar, Kamalasankari Subramaniakuppusamy, Relsy Puthal, Kaustik Ranaware  

**Link**: [PDF](https://arxiv.org/pdf/2509.16275)  

**Abstract**: Modern software development pipelines face growing challenges in securing large codebases with extensive dependencies. Static analysis tools like Bandit are effective at vulnerability detection but suffer from high false positives and lack repair capabilities. Large Language Models (LLMs), in contrast, can suggest fixes but often hallucinate changes and lack self-validation. We present SecureFixAgent, a hybrid repair framework integrating Bandit with lightweight local LLMs (<8B parameters) in an iterative detect-repair-validate loop. To improve precision, we apply parameter-efficient LoRA-based fine-tuning on a diverse, curated dataset spanning multiple Python project domains, mitigating dataset bias and reducing unnecessary edits. SecureFixAgent uses Bandit for detection, the LLM for candidate fixes with explanations, and Bandit re-validation for verification, all executed locally to preserve privacy and reduce cloud reliance. Experiments show SecureFixAgent reduces false positives by 10.8% over static analysis, improves fix accuracy by 13.51%, and lowers false positives by 5.46% compared to pre-trained LLMs, typically converging within three iterations. Beyond metrics, developer studies rate explanation quality 4.5/5, highlighting its value for human trust and adoption. By combining verifiable security improvements with transparent rationale in a resource-efficient local framework, SecureFixAgent advances trustworthy, automated vulnerability remediation for modern pipelines. 

**Abstract (ZH)**: 现代软件开发管道在保护具有广泛依赖关系的大规模代码库时面临着日益增长的安全挑战。静态分析工具如Bandit在漏洞检测方面效果显著，但存在高误报率和缺乏修复能力的问题。相比之下，大型语言模型（LLMs）可以提出修复建议，但常常幻化变化且缺乏自我验证。我们提出了SecureFixAgent，这是一种将Bandit与轻量级本地LLM（<8B参数）结合的混合修复框架，在迭代检测-修复-验证循环中运行。为了提高精度，我们在涵盖多个Python项目领域的多样且精制的数据集上应用参数高效的LoRA基于微调，减轻数据集偏见并减少不必要的编辑。SecureFixAgent使用Bandit进行检测，LLM提供带有解释的候选修复，Bandit重新验证以进行验证，均在本地执行以保护隐私并减少对云的依赖。实验结果显示，SecureFixAgent相比静态分析将误报率减少了10.8%，修复准确性提高了13.51%，相比预训练的LLMs将误报率降低了5.46%，通常三轮内收敛。除了指标外，开发者研究将解释质量评定为4.5/5，突显了其在人类信任和采用方面的价值。通过结合可验证的安全改进和透明的解释理由，在资源高效的本地框架中，SecureFixAgent推进了现代管道中值得信赖的自动漏洞修复。 

---
# Digging Into the Internal: Causality-Based Analysis of LLM Function Calling 

**Title (ZH)**: 探究内部机制：基于因果分析的LLM函数调用研究 

**Authors**: Zhenlan Ji, Daoyuan Wu, Wenxuan Wang, Pingchuan Ma, Shuai Wang, Lei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.16268)  

**Abstract**: Function calling (FC) has emerged as a powerful technique for facilitating large language models (LLMs) to interact with external systems and perform structured tasks. However, the mechanisms through which it influences model behavior remain largely under-explored. Besides, we discover that in addition to the regular usage of FC, this technique can substantially enhance the compliance of LLMs with user instructions. These observations motivate us to leverage causality, a canonical analysis method, to investigate how FC works within LLMs. In particular, we conduct layer-level and token-level causal interventions to dissect FC's impact on the model's internal computational logic when responding to user queries. Our analysis confirms the substantial influence of FC and reveals several in-depth insights into its mechanisms. To further validate our findings, we conduct extensive experiments comparing the effectiveness of FC-based instructions against conventional prompting methods. We focus on enhancing LLM safety robustness, a critical LLM application scenario, and evaluate four mainstream LLMs across two benchmark datasets. The results are striking: FC shows an average performance improvement of around 135% over conventional prompting methods in detecting malicious inputs, demonstrating its promising potential to enhance LLM reliability and capability in practical applications. 

**Abstract (ZH)**: FC增强大型语言模型与外部系统交互及其影响机制的研究 

---
# Gender and Political Bias in Large Language Models: A Demonstration Platform 

**Title (ZH)**: 大型语言模型中的性别与政治偏见：一个示范平台 

**Authors**: Wenjie Lin, Hange Liu, Xutao Mao, Yingying Zhuang, Jingwei Shi, Xudong Han, Tianyu Shi, Jinrui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16264)  

**Abstract**: We present ParlAI Vote, an interactive system for exploring European Parliament debates and votes, and for testing LLMs on vote prediction and bias analysis. This platform connects debate topics, speeches, and roll-call outcomes, and includes rich demographic data such as gender, age, country, and political group. Users can browse debates, inspect linked speeches, compare real voting outcomes with predictions from frontier LLMs, and view error breakdowns by demographic group. Visualizing the EuroParlVote benchmark and its core tasks of gender classification and vote prediction, ParlAI Vote highlights systematic performance bias in state-of-the-art LLMs. The system unifies data, models, and visual analytics in a single interface, lowering the barrier for reproducing findings, auditing behavior, and running counterfactual scenarios. It supports research, education, and public engagement with legislative decision-making, while making clear both the strengths and the limitations of current LLMs in political analysis. 

**Abstract (ZH)**: ParlAI Vote：一个探索欧洲议会辩论和投票的交互系统及其对LLM投票预测和偏见分析的测试平台 

---
# Socratic Mind: Impact of a Novel GenAI-Powered Assessment Tool on Student Learning and Higher-Order Thinking 

**Title (ZH)**: 苏格拉底思维：新型GenAI驱动评估工具对学生学习和高层次思维的影响 

**Authors**: Jeonghyun Lee, Jui-Tse Hung, Meryem Yilmaz Soylu, Diana Popescu, Christopher Zhang Cui, Gayane Grigoryan, David A Joyner, Stephen W Harmon  

**Link**: [PDF](https://arxiv.org/pdf/2509.16262)  

**Abstract**: This study examines the impact of Socratic Mind, a Generative Artificial Intelligence (GenAI) powered formative assessment tool that employs Socratic questioning to support student learning in a large, fully online undergraduate-level computing course. Employing a quasi-experimental, mixed-methods design, we investigated participants' engagement patterns, the influence of user experience on engagement, and impacts on both perceived and actual learning outcomes. Data were collected from the system logs, surveys on user experience and perceived engagement and learning gains, student reflections, and course performance data. Results indicated that participants consistently reported high levels of affective, behavioral, and cognitive engagement, and these were strongly linked to positive user experiences and perceived learning outcomes. Quantitative analysis further revealed that students who engaged with the GenAI tool experienced significant gains in their quiz scores compared to those who did not, particularly benefiting students with lower baseline achievement. Additionally, thematic analysis of qualitative feedback revealed substantial perceived improvements in higher-order thinking skills, including problem solving, critical thinking, and self-reflection. Our findings highlight the promise of AI-mediated dialogue in fostering deeper engagement and higher-order cognitive skills. As higher education institutions expand GenAI integration in curriculum, this dialogic, GenAI powered assessment tool can offer a scalable strategy to promote students' meaningful learning outcomes. 

**Abstract (ZH)**: 本研究考察了Socratic Mind这一由生成型人工智能（GenAI）驱动的形成性评估工具的影响，该工具采用苏格拉底式提问来支持大型全在线本科计算机课程学生的学习。采用准实验和混合方法设计，我们调查了参与者的行为模式，用户体验对参与的影响，以及对感知和实际学习成果的影响。数据来源于系统日志、用户体验和感知参与及学习收益的调查、学生的反思以及课程成绩数据。研究结果显示，参与者持续报告高水平的情感、行为和认知参与，这些参与与积极的用户体验和感知学习成果密切相关。定量分析进一步表明，与未使用GenAI工具的学生相比，与GenAI工具互动的学生在测验成绩上取得了明显进步，特别是对于 baseline 成绩较低的学生而言。此外，对定性反馈的主题分析揭示了参与者在高层次思维能力方面感知到的显著提升，包括问题解决、批判性思维和自我反思。我们的研究结果强调了人工智能介导的对话在促进更深层次参与和高层次认知技能方面的能力。随着高等教育机构扩大GenAI在课程中的应用，这种对话式、由GenAI驱动的评估工具可以提供一种可扩展的战略，以促进学生的有意义的学习成果。 

---
# On LLM-Based Scientific Inductive Reasoning Beyond Equations 

**Title (ZH)**: 基于LLM的超越方程的科学归纳推理 

**Authors**: Brian S. Lin, Jiaxin Yuan, Zihan Zhou, Shouli Wang, Shuo Wang, Cunliang Kong, Qi Shi, Yuxuan Li, Liner Yang, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.16226)  

**Abstract**: As large language models (LLMs) increasingly exhibit human-like capabilities, a fundamental question emerges: How can we enable LLMs to learn the underlying patterns from limited examples in entirely novel environments and apply them effectively? This question is central to the ability of LLMs in inductive reasoning. Existing research on LLM-based inductive reasoning can be broadly categorized based on whether the underlying rules are expressible via explicit mathematical equations. However, many recent studies in the beyond-equations category have emphasized rule design without grounding them in specific scenarios. Inspired by the parallels between inductive reasoning and human scientific discovery, we propose the task of LLM-Based Scientific Inductive Reasoning Beyond Equations and introduce a new benchmark, SIRBench-V1, to evaluate the inductive reasoning abilities of LLMs in scientific settings. Our experimental results show that current LLMs still struggle with this task, underscoring its difficulty and the need for further advancement in this area. 

**Abstract (ZH)**: 基于大语言模型的无方程科学归纳推理及其评估基准SIRBench-V1 

---
