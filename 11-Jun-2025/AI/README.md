# ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering 

**Title (ZH)**: ALE-Bench: 一种长时域目标驱动的算法工程基准测试 

**Authors**: Yuki Imajuku, Kohki Horie, Yoichi Iwata, Kensho Aoki, Naohiro Takahashi, Takuya Akiba  

**Link**: [PDF](https://arxiv.org/pdf/2506.09050)  

**Abstract**: How well do AI systems perform in algorithm engineering for hard optimization problems in domains such as package-delivery routing, crew scheduling, factory production planning, and power-grid balancing? We introduce ALE-Bench, a new benchmark for evaluating AI systems on score-based algorithmic programming contests. Drawing on real tasks from the AtCoder Heuristic Contests, ALE-Bench presents optimization problems that are computationally hard and admit no known exact solution. Unlike short-duration, pass/fail coding benchmarks, ALE-Bench encourages iterative solution refinement over long time horizons. Our software framework supports interactive agent architectures that leverage test-run feedback and visualizations. Our evaluation of frontier LLMs revealed that while they demonstrate high performance on specific problems, a notable gap remains compared to humans in terms of consistency across problems and long-horizon problem-solving capabilities. This highlights the need for this benchmark to foster future AI advancements. 

**Abstract (ZH)**: AI系统在硬优化问题领域（如包裹配送路由、机组调度、工厂生产规划和电网平衡）的算法工程中表现如何？我们引入ALE-Bench，一个新的基准，用于评估AI系统在基于评分的算法编程竞赛中的性能。ALE-Bench采用来自AtCoder启发式竞赛的真實任务，呈现计算复杂且目前缺乏确切解决方案的优化问题。与短暂的通过/失败编码基准不同，ALE-Bench鼓励在长时间范围内进行解决方案的迭代优化。我们的软件框架支持利用测试运行反馈和可视化技术的交互式代理架构。我们对前沿的大语言模型的评估显示，尽管它们在特定问题上表现出色，但在问题一致性及长时间问题解决能力方面仍与人类存在明显差距。这突显了建立此基准以促进未来AI进步的需求。 

---
# VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning 

**Title (ZH)**: VIKI-R：通过强化学习协调具身多智能体合作 

**Authors**: Li Kang, Xiufeng Song, Heng Zhou, Yiran Qin, Jie Yang, Xiaohong Liu, Philip Torr, Lei Bai, Zhenfei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09049)  

**Abstract**: Coordinating multiple embodied agents in dynamic environments remains a core challenge in artificial intelligence, requiring both perception-driven reasoning and scalable cooperation strategies. While recent works have leveraged large language models (LLMs) for multi-agent planning, a few have begun to explore vision-language models (VLMs) for visual reasoning. However, these VLM-based approaches remain limited in their support for diverse embodiment types. In this work, we introduce VIKI-Bench, the first hierarchical benchmark tailored for embodied multi-agent cooperation, featuring three structured levels: agent activation, task planning, and trajectory perception. VIKI-Bench includes diverse robot embodiments, multi-view visual observations, and structured supervision signals to evaluate reasoning grounded in visual inputs. To demonstrate the utility of VIKI-Bench, we propose VIKI-R, a two-stage framework that fine-tunes a pretrained vision-language model (VLM) using Chain-of-Thought annotated demonstrations, followed by reinforcement learning under multi-level reward signals. Our extensive experiments show that VIKI-R significantly outperforms baselines method across all task levels. Furthermore, we show that reinforcement learning enables the emergence of compositional cooperation patterns among heterogeneous agents. Together, VIKI-Bench and VIKI-R offer a unified testbed and method for advancing multi-agent, visual-driven cooperation in embodied AI systems. 

**Abstract (ZH)**: 协调多智能体在动态环境中的交互仍然是人工智能领域的核心挑战，需要感知驱动的推理能力和可扩展的合作策略。虽然最近的研究利用了大型语言模型（LLMs）进行多智能体规划，少数研究开始探索视觉-语言模型（VLMs）进行视觉推理。然而，这些基于VLM的方法在支持多种载体类型方面仍有限制。在本文中，我们引入了VIKI-Bench，这是第一个专门针对智能体协作设计的分层基准，包括三个结构化的层级：智能体激活、任务规划和轨迹感知。VIKI-Bench 包括多种机器人载体、多视角视觉观察以及基于视觉输入的结构化监督信号，以评价视觉推理能力。为了展示VIKI-Bench 的 usefulness，我们提出了VIKI-R 两阶段框架，该框架首先使用带有Chain-of-Thought标注示范来微调预训练的视觉-语言模型（VLM），然后在多层次奖励信号下进行强化学习。我们的大量实验表明，VIKI-R 在所有任务层级上显著优于基线方法。此外，我们展示了强化学习使得异质智能体之间组合协作模式的涌现。总之，VIKI-Bench 和 VIKI-R 提供了一个统一的测试平台和方法，以推进基于视觉驱动的多智能体协作在具身人工智能系统中的发展。 

---
# AbstentionBench: Reasoning LLMs Fail on Unanswerable Questions 

**Title (ZH)**: AbstentionBench: LLMs在无法回答的问题上的推理失败 

**Authors**: Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, Samuel J. Bell  

**Link**: [PDF](https://arxiv.org/pdf/2506.09038)  

**Abstract**: For Large Language Models (LLMs) to be reliably deployed in both everyday and high-stakes domains, knowing when not to answer is equally critical as answering correctly. Real-world user queries, which can be underspecified, ill-posed, or fundamentally unanswerable, require LLMs to reason about uncertainty and selectively abstain -- i.e., refuse to answer definitively. However, abstention remains understudied, without a systematic evaluation framework for modern LLMs. In this work, we introduce AbstentionBench, a large-scale benchmark for holistically evaluating abstention across 20 diverse datasets, including questions with unknown answers, underspecification, false premises, subjective interpretations, and outdated information. Evaluating 20 frontier LLMs reveals abstention is an unsolved problem, and one where scaling models is of little use. While recent reasoning LLMs have shown impressive results in complex problem solving, surprisingly, we find that reasoning fine-tuning degrades abstention (by $24\%$ on average), even for math and science domains on which reasoning models are explicitly trained. We find that while a carefully crafted system prompt can boost abstention in practice, it does not resolve models' fundamental inability to reason about uncertainty. We release AbstentionBench to foster research into advancing LLM reliability. 

**Abstract (ZH)**: 大型语言模型（LLMs）在日常生活和高 stakes 领域可靠部署的关键在于正确回答与合理不回答同样重要。大规模基准 AbstentionBench：全面评估 20 个多样数据集中的合理不回答能力 

---
# A Survey of Link Prediction in N-ary Knowledge Graphs 

**Title (ZH)**: N-元知识图谱中的链接预测综述 

**Authors**: Jiyao Wei, Saiping Guan, Da Li, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.08970)  

**Abstract**: N-ary Knowledge Graphs (NKGs) are a specialized type of knowledge graph designed to efficiently represent complex real-world facts. Unlike traditional knowledge graphs, where a fact typically involves two entities, NKGs can capture n-ary facts containing more than two entities. Link prediction in NKGs aims to predict missing elements within these n-ary facts, which is essential for completing NKGs and improving the performance of downstream applications. This task has recently gained significant attention. In this paper, we present the first comprehensive survey of link prediction in NKGs, providing an overview of the field, systematically categorizing existing methods, and analyzing their performance and application scenarios. We also outline promising directions for future research. 

**Abstract (ZH)**: N-元知识图谱中的链接预测研究综述 

---
# Evaluating Generative Vehicle Trajectory Models for Traffic Intersection Dynamics 

**Title (ZH)**: 评估生成式车辆轨迹模型在交通交叉口动态中的应用 

**Authors**: Yash Ranjan, Rahul Sengupta, Anand Rangarajan, Sanjay Ranka  

**Link**: [PDF](https://arxiv.org/pdf/2506.08963)  

**Abstract**: Traffic Intersections are vital to urban road networks as they regulate the movement of people and goods. However, they are regions of conflicting trajectories and are prone to accidents. Deep Generative models of traffic dynamics at signalized intersections can greatly help traffic authorities better understand the efficiency and safety aspects. At present, models are evaluated on computational metrics that primarily look at trajectory reconstruction errors. They are not evaluated online in a `live' microsimulation scenario. Further, these metrics do not adequately consider traffic engineering-specific concerns such as red-light violations, unallowed stoppage, etc. In this work, we provide a comprehensive analytics tool to train, run, and evaluate models with metrics that give better insights into model performance from a traffic engineering point of view. We train a state-of-the-art multi-vehicle trajectory forecasting model on a large dataset collected by running a calibrated scenario of a real-world urban intersection. We then evaluate the performance of the prediction models, online in a microsimulator, under unseen traffic conditions. We show that despite using ideally-behaved trajectories as input, and achieving low trajectory reconstruction errors, the generated trajectories show behaviors that break traffic rules. We introduce new metrics to evaluate such undesired behaviors and present our results. 

**Abstract (ZH)**: 交通交叉口是城市道路网络中的关键部分，它们调节着人流和物流的流动。然而，它们是轨迹冲突的区域，容易发生事故。能够生成信号控制交叉口交通动态的深度生成模型可以大大帮助交通管理部门更好地理解和提高效率及安全性。目前，这些模型主要根据计算指标进行评估，这些指标主要关注轨迹重构误差，并未在线评估在实时微观模拟场景中的表现。此外，这些指标未能充分考虑到交通工程相关的特定关注点，如闯红灯、非法停车等。在本研究中，我们提供了一个综合分析工具，用于训练、运行和评估模型，这些模型采用从真实城市交叉口校准场景中收集的大规模数据集进行训练，并从交通工程的角度更好地评估模型性能。我们训练了一个最先进的多车辆轨迹预测模型，并在线评估其在未见过的交通条件下的性能。尽管输入了理想行为的轨迹，并且实现了低轨迹重构误差，生成的轨迹仍然表现出违反交通规则的行为。我们引入了新的评估指标来评价这种不良行为，并展示了我们的研究成果。 

---
# IntTrajSim: Trajectory Prediction for Simulating Multi-Vehicle driving at Signalized Intersections 

**Title (ZH)**: IntTrajSim：基于信号交叉口多车辆驾驶轨迹预测的仿真 

**Authors**: Yash Ranjan, Rahul Sengupta, Anand Rangarajan, Sanjay Ranka  

**Link**: [PDF](https://arxiv.org/pdf/2506.08957)  

**Abstract**: Traffic simulators are widely used to study the operational efficiency of road infrastructure, but their rule-based approach limits their ability to mimic real-world driving behavior. Traffic intersections are critical components of the road infrastructure, both in terms of safety risk (nearly 28% of fatal crashes and 58% of nonfatal crashes happen at intersections) as well as the operational efficiency of a road corridor. This raises an important question: can we create a data-driven simulator that can mimic the macro- and micro-statistics of the driving behavior at a traffic intersection? Deep Generative Modeling-based trajectory prediction models provide a good starting point to model the complex dynamics of vehicles at an intersection. But they are not tested in a "live" micro-simulation scenario and are not evaluated on traffic engineering-related metrics. In this study, we propose traffic engineering-related metrics to evaluate generative trajectory prediction models and provide a simulation-in-the-loop pipeline to do so. We also provide a multi-headed self-attention-based trajectory prediction model that incorporates the signal information, which outperforms our previous models on the evaluation metrics. 

**Abstract (ZH)**: 基于深度生成建模的交通交叉口数据驱动轨迹预测模型评价及模拟框架 

---
# Preference-Driven Multi-Objective Combinatorial Optimization with Conditional Computation 

**Title (ZH)**: 基于偏好驱动的条件计算多目标组合优化 

**Authors**: Mingfeng Fan, Jianan Zhou, Yifeng Zhang, Yaoxin Wu, Jinbiao Chen, Guillaume Adrien Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2506.08898)  

**Abstract**: Recent deep reinforcement learning methods have achieved remarkable success in solving multi-objective combinatorial optimization problems (MOCOPs) by decomposing them into multiple subproblems, each associated with a specific weight vector. However, these methods typically treat all subproblems equally and solve them using a single model, hindering the effective exploration of the solution space and thus leading to suboptimal performance. To overcome the limitation, we propose POCCO, a novel plug-and-play framework that enables adaptive selection of model structures for subproblems, which are subsequently optimized based on preference signals rather than explicit reward values. Specifically, we design a conditional computation block that routes subproblems to specialized neural architectures. Moreover, we propose a preference-driven optimization algorithm that learns pairwise preferences between winning and losing solutions. We evaluate the efficacy and versatility of POCCO by applying it to two state-of-the-art neural methods for MOCOPs. Experimental results across four classic MOCOP benchmarks demonstrate its significant superiority and strong generalization. 

**Abstract (ZH)**: 近期的深度强化学习方法通过将多目标组合优化问题分解为多个子问题，并为每个子问题分配特定的权重向量，取得了显著的成功。然而，这些方法通常平等对待所有子问题，并使用单个模型来解决它们，这阻碍了对解空间的有效探索，从而导致性能不佳。为克服这一局限，我们提出了一种名为POCCO的新型插件式框架，该框架能够为子问题自适应地选择模型结构，随后根据偏好信号而非显式奖励值对其进行优化。具体地，我们设计了一个条件计算模块，将子问题路由到专门的神经架构中。此外，我们提出了一种基于偏好的优化算法，该算法学习胜者和败者解对之间的成对偏好。通过将其应用于两种最先进的神经方法解决多目标组合优化问题，我们评估了POCCO的有效性和灵活性。在四个经典多目标组合优化问题基准上的实验结果表明，POCCO在显著性和通用性方面具有显著优势。 

---
# Your Brain on ChatGPT: Accumulation of Cognitive Debt when Using an AI Assistant for Essay Writing Task 

**Title (ZH)**: ChatGPT对大脑的影响：使用AI助手进行论文写作任务时的认知债务累积 

**Authors**: Nataliya Kosmyna, Eugene Hauptmann, Ye Tong Yuan, Jessica Situ, Xian-Hao Liao, Ashly Vivian Beresnitzky, Iris Braunstein, Pattie Maes  

**Link**: [PDF](https://arxiv.org/pdf/2506.08872)  

**Abstract**: This study explores the neural and behavioral consequences of LLM-assisted essay writing. Participants were divided into three groups: LLM, Search Engine, and Brain-only (no tools). Each completed three sessions under the same condition. In a fourth session, LLM users were reassigned to Brain-only group (LLM-to-Brain), and Brain-only users were reassigned to LLM condition (Brain-to-LLM). A total of 54 participants took part in Sessions 1-3, with 18 completing session 4. We used electroencephalography (EEG) to assess cognitive load during essay writing, and analyzed essays using NLP, as well as scoring essays with the help from human teachers and an AI judge. Across groups, NERs, n-gram patterns, and topic ontology showed within-group homogeneity. EEG revealed significant differences in brain connectivity: Brain-only participants exhibited the strongest, most distributed networks; Search Engine users showed moderate engagement; and LLM users displayed the weakest connectivity. Cognitive activity scaled down in relation to external tool use. In session 4, LLM-to-Brain participants showed reduced alpha and beta connectivity, indicating under-engagement. Brain-to-LLM users exhibited higher memory recall and activation of occipito-parietal and prefrontal areas, similar to Search Engine users. Self-reported ownership of essays was the lowest in the LLM group and the highest in the Brain-only group. LLM users also struggled to accurately quote their own work. While LLMs offer immediate convenience, our findings highlight potential cognitive costs. Over four months, LLM users consistently underperformed at neural, linguistic, and behavioral levels. These results raise concerns about the long-term educational implications of LLM reliance and underscore the need for deeper inquiry into AI's role in learning. 

**Abstract (ZH)**: 本研究探讨了LLM辅助作文写作的神经和行为后果。参与者被分为三组：LLM组、搜索引擎组和脑内组（无需工具）。每组在相同条件下完成了三轮任务。在第四轮任务中，LLM用户被重新分配到脑内组（LLM-to-Brain），脑内用户被重新分配到LLM条件（Brain-to-LLM）。共有54名参与者参加了前三轮任务，其中18名完成了第四轮任务。我们使用脑电图（EEG）评估了作文写作过程中的认知负荷，并使用自然语言处理分析了作文，还通过人类教师和AI评委评分。各组的命名实体、n元组模式和主题本体显示了组内一致性。EEG结果显示显著的脑连接差异：脑内组用户表现出最强且最分布的网络；搜索引擎用户表现出中等程度的参与；而LLM用户表现出最弱的连接。外部工具使用与认知活动呈负相关。在第四轮任务中，LLM-to-Brain用户显示出了减少的alpha和beta连接性，表明参与度不足。Brain-to-LLM用户表现出较高的记忆力召回和枕叶-顶叶和前额叶区域的激活，类似于搜索引擎用户。自报的作文拥有度在LLM组最低，在脑内组最高。LLM用户还难以准确引用自己的作品。尽管LLM提供了即时便利，但我们的研究结果揭示了潜在的认知成本。四个月中，LLM用户在神经、语言和行为层面持续表现不佳。这些结果引发了对LLM依赖的长期教育影响的担忧，并强调了对AI在学习中角色进行更深入研究的必要性。 

---
# Measuring Data Science Automation: A Survey of Evaluation Tools for AI Assistants and Agents 

**Title (ZH)**: 测量数据科学自动化：AI助理和代理的评估工具综述 

**Authors**: Irene Testini, José Hernández-Orallo, Lorenzo Pacchiardi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08800)  

**Abstract**: Data science aims to extract insights from data to support decision-making processes. Recently, Large Language Models (LLMs) are increasingly used as assistants for data science, by suggesting ideas, techniques and small code snippets, or for the interpretation of results and reporting. Proper automation of some data-science activities is now promised by the rise of LLM agents, i.e., AI systems powered by an LLM equipped with additional affordances--such as code execution and knowledge bases--that can perform self-directed actions and interact with digital environments. In this paper, we survey the evaluation of LLM assistants and agents for data science. We find (1) a dominant focus on a small subset of goal-oriented activities, largely ignoring data management and exploratory activities; (2) a concentration on pure assistance or fully autonomous agents, without considering intermediate levels of human-AI collaboration; and (3) an emphasis on human substitution, therefore neglecting the possibility of higher levels of automation thanks to task transformation. 

**Abstract (ZH)**: 数据科学旨在从数据中提取洞见以支持决策过程。近年来，大型语言模型（LLMs）越来越多地被用作数据科学的助手，通过提供想法、技术及小段代码，或对结果进行解释和报告。LLM代理的兴起为某些数据科学活动的适当自动化提供了可能，即由配备额外功能（如代码执行和知识库）的LLM驱动的AI系统可以自行执行操作并与数字环境交互。在本文中，我们调研了LLM助手和代理在数据科学中的评价。我们发现（1）主要集中于少数目标导向活动，大大忽视了数据管理和探索性活动；（2）集中在纯粹的辅助或完全自主的代理，而不考虑人类-AI协作的中间水平；（3）强调人类替代，因此忽视了通过任务转换实现更高水平自动化的可能性。 

---
# Paths to Causality: Finding Informative Subgraphs Within Knowledge Graphs for Knowledge-Based Causal Discovery 

**Title (ZH)**: 因果路径：在知识图中寻找用于知识导向因果发现的 informative 子图路径 

**Authors**: Yuni Susanti, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2506.08771)  

**Abstract**: Inferring causal relationships between variable pairs is crucial for understanding multivariate interactions in complex systems. Knowledge-based causal discovery -- which involves inferring causal relationships by reasoning over the metadata of variables (e.g., names or textual context) -- offers a compelling alternative to traditional methods that rely on observational data. However, existing methods using Large Language Models (LLMs) often produce unstable and inconsistent results, compromising their reliability for causal inference. To address this, we introduce a novel approach that integrates Knowledge Graphs (KGs) with LLMs to enhance knowledge-based causal discovery. Our approach identifies informative metapath-based subgraphs within KGs and further refines the selection of these subgraphs using Learning-to-Rank-based models. The top-ranked subgraphs are then incorporated into zero-shot prompts, improving the effectiveness of LLMs in inferring the causal relationship. Extensive experiments on biomedical and open-domain datasets demonstrate that our method outperforms most baselines by up to 44.4 points in F1 scores, evaluated across diverse LLMs and KGs. Our code and datasets are available on GitHub: this https URL 

**Abstract (ZH)**: 基于知识图谱的大型语言模型驱动的因果发现新方法 

---
# A Sample Efficient Conditional Independence Test in the Presence of Discretization 

**Title (ZH)**: discretization环境下样本效率的条件独立性检验 

**Authors**: Boyang Sun, Yu Yao, Xinshuai Dong, Zongfang Liu, Tongliang Liu, Yumou Qiu, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08747)  

**Abstract**: In many real-world scenarios, interested variables are often represented as discretized values due to measurement limitations. Applying Conditional Independence (CI) tests directly to such discretized data, however, can lead to incorrect conclusions. To address this, recent advancements have sought to infer the correct CI relationship between the latent variables through binarizing observed data. However, this process inevitably results in a loss of information, which degrades the test's performance. Motivated by this, this paper introduces a sample-efficient CI test that does not rely on the binarization process. We find that the independence relationships of latent continuous variables can be established by addressing an over-identifying restriction problem with Generalized Method of Moments (GMM). Based on this insight, we derive an appropriate test statistic and establish its asymptotic distribution correctly reflecting CI by leveraging nodewise regression. Theoretical findings and Empirical results across various datasets demonstrate that the superiority and effectiveness of our proposed test. Our code implementation is provided in this https URL 

**Abstract (ZH)**: 在很多现实场景中，感兴趣的变量往往由于测量限制而表现为离散值。直接将条件独立性（CI）测试应用于此类离散数据可能会导致错误的结论。为了应对这一问题，最近的研究尝试通过二值化观测数据来推断潜在变量之间的正确CI关系。然而，这一过程不可避免地会丢失信息，从而降低测试性能。受此启发，本文介绍了一种样本高效的CI测试，无需依赖二值化过程。我们发现，通过使用广义矩方法（GMM）解决过识别限制问题，可以建立潜在连续变量的独立关系。基于这一洞察，我们推导出适当的检验统计量，并通过节点回归正确建立其渐近分布，准确反映CI。理论发现和多种数据集上的实证结果证明了我们所提出的测试的优越性和有效性。我们的代码实现可通过以下链接获取：https://your-link-url.com 

---
# Consistent Paths Lead to Truth: Self-Rewarding Reinforcement Learning for LLM Reasoning 

**Title (ZH)**: 一致的路径通向真理：自我奖励强化学习在大模型推理解析中的应用 

**Authors**: Kongcheng Zhang, Qi Yao, Shunyu Liu, Yingjie Wang, Baisheng Lai, Jieping Ye, Mingli Song, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.08745)  

**Abstract**: Recent advances of Reinforcement Learning (RL) have highlighted its potential in complex reasoning tasks, yet effective training often relies on external supervision, which limits the broader applicability. In this work, we propose a novel self-rewarding reinforcement learning framework to enhance Large Language Model (LLM) reasoning by leveraging the consistency of intermediate reasoning states across different reasoning trajectories. Our key insight is that correct responses often exhibit consistent trajectory patterns in terms of model likelihood: their intermediate reasoning states tend to converge toward their own final answers (high consistency) with minimal deviation toward other candidates (low volatility). Inspired by this observation, we introduce CoVo, an intrinsic reward mechanism that integrates Consistency and Volatility via a robust vector-space aggregation strategy, complemented by a curiosity bonus to promote diverse exploration. CoVo enables LLMs to perform RL in a self-rewarding manner, offering a scalable pathway for learning to reason without external supervision. Extensive experiments on diverse reasoning benchmarks show that CoVo achieves performance comparable to or even surpassing supervised RL. Our code is available at this https URL. 

**Abstract (ZH)**: Recent advances of Reinforcement Learning (RL) have highlighted its potential in complex reasoning tasks, yet effective training often relies on external supervision, which limits the broader applicability. In this work, we propose a novel self-rewarding reinforcement learning framework to enhance Large Language Model (LLM) reasoning by leveraging the consistency of intermediate reasoning states across different reasoning trajectories. Our key insight is that correct responses often exhibit consistent trajectory patterns in terms of model likelihood: their intermediate reasoning states tend to converge toward their own final answers (high consistency) with minimal deviation toward other candidates (low volatility). Inspired by this observation, we introduce CoVo, an intrinsic reward mechanism that integrates Consistency and Volatility via a robust vector-space aggregation strategy, complemented by a curiosity bonus to promote diverse exploration. CoVo enables LLMs to perform RL in a self-rewarding manner, offering a scalable pathway for learning to reason without external supervision. Extensive experiments on diverse reasoning benchmarks show that CoVo achieves performance comparable to or even surpassing supervised RL. Our code is available at this https URL. 

自监督强化学习框架CoVo：通过利用不同推理轨迹中间推理状态的一致性增强大型语言模型的推理能力 

---
# Modular Recurrence in Contextual MDPs for Universal Morphology Control 

**Title (ZH)**: Contextual MDPs 中的模块化复发控制通用形态学控制 

**Authors**: Laurens Engwegen, Daan Brinks, Wendelin Böhmer  

**Link**: [PDF](https://arxiv.org/pdf/2506.08630)  

**Abstract**: A universal controller for any robot morphology would greatly improve computational and data efficiency. By utilizing contextual information about the properties of individual robots and exploiting their modular structure in the architecture of deep reinforcement learning agents, steps have been made towards multi-robot control. Generalization to new, unseen robots, however, remains a challenge. In this paper we hypothesize that the relevant contextual information is partially observable, but that it can be inferred through interactions for better generalization to contexts that are not seen during training. To this extent, we implement a modular recurrent architecture and evaluate its generalization performance on a large set of MuJoCo robots. The results show a substantial improved performance on robots with unseen dynamics, kinematics, and topologies, in four different environments. 

**Abstract (ZH)**: 一种适用于任意机器人形态的通用控制器将大大提高计算和数据效率。通过利用关于个体机器人特性的上下文信息并利用其模块化结构在深度强化学习代理的架构中，我们已在多机器人控制方面取得了一些进展。然而，将控制扩展到新的、未见过的机器人仍然是一个挑战。在本文中，我们假设相关的上下文信息是部分可观测的，但可以通过交互来推断，从而更好地泛化到训练过程中未见过的上下文中。为此，我们实现了一种模块化的循环结构，并在大量MuJoCo机器人的集合上评估其泛化性能。结果表明，该方法在四种不同环境中对未见过的动力学、运动学和拓扑结构的机器人表现出显著提高的性能。 

---
# FoldA: Computing Partial-Order Alignments Using Directed Net Unfoldings 

**Title (ZH)**: FoldA: 使用有向网展开计算部分序对齐 

**Authors**: Douwe Geurtjens, Xixi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08627)  

**Abstract**: Conformance checking is a fundamental task of process mining, which quantifies the extent to which the observed process executions match a normative process model. The state-of-the-art approaches compute alignments by exploring the state space formed by the synchronous product of the process model and the trace. This often leads to state space explosion, particularly when the model exhibits a high degree of choice and concurrency. Moreover, as alignments inherently impose a sequential structure, they fail to fully represent the concurrent behavior present in many real-world processes. To address these limitations, this paper proposes a new technique for computing partial-order alignments {on the fly using directed Petri net unfoldings, named FoldA. We evaluate our technique on 485 synthetic model-log pairs and compare it against Astar- and Dijkstra-alignments on 13 real-life model-log pairs and 6 benchmark pairs. The results show that our unfolding alignment, although it requires more computation time, generally reduces the number of queued states and provides a more accurate representation of concurrency. 

**Abstract (ZH)**: 过程合规性检查是过程挖掘中的一个基础任务，它量化观察到的过程执行与规范性过程模型之间的符合程度。现有的先进方法通过探索由过程模型和轨迹同步积形成的状态空间来计算对齐，这往往会导致状态空间爆炸，尤其是在模型表现出高度的选择性和并发性时。此外，由于对齐本质上施加了序列结构，它们无法充分代表许多现实世界过程中存在的并发行为。为了解决这些限制，本文提出了一种新的技术——使用有向Petri网扩展来计算偏序对齐（on the fly），名为FoldA。我们在485个合成模型-日志对上评估了该技术，并在13个现实世界模型-日志对和6个基准对上与Astar-和Dijkstra-对齐进行了比较。结果显示，尽管我们的扩展对齐需要更多计算时间，但通常减少了排队状态的数量，并更准确地代表了并发性。 

---
# HGFormer: A Hierarchical Graph Transformer Framework for Two-Stage Colonel Blotto Games via Reinforcement Learning 

**Title (ZH)**: HGFormer: 一种基于强化学习的两阶段 Colonel Blotto 游戏分层图变换器框架 

**Authors**: Yang Lv, Jinlong Lei, Peng Yi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08580)  

**Abstract**: Two-stage Colonel Blotto game represents a typical adversarial resource allocation problem, in which two opposing agents sequentially allocate resources in a network topology across two phases: an initial resource deployment followed by multiple rounds of dynamic reallocation adjustments. The sequential dependency between game stages and the complex constraints imposed by the graph topology make it difficult for traditional approaches to attain a globally optimal strategy. To address these challenges, we propose a hierarchical graph Transformer framework called HGformer. By incorporating an enhanced graph Transformer encoder with structural biases and a two-agent hierarchical decision model, our approach enables efficient policy generation in large-scale adversarial environments. Moreover, we design a layer-by-layer feedback reinforcement learning algorithm that feeds the long-term returns from lower-level decisions back into the optimization of the higher-level strategy, thus bridging the coordination gap between the two decision-making stages. Experimental results demonstrate that, compared to existing hierarchical decision-making or graph neural network methods, HGformer significantly improves resource allocation efficiency and adversarial payoff, achieving superior overall performance in complex dynamic game scenarios. 

**Abstract (ZH)**: 两级Colonel Blotto博弈代表了一种典型的 adversarial 资源分配问题，其中两个对立的代理在两个阶段的网络拓扑中依次分配资源：初始资源部署后，再进行多轮动态再分配调整。博弈阶段之间的顺序依赖性和由图拓扑施加的复杂约束使传统方法难以获得全局最优策略。为了应对这些挑战，我们提出了一种分层图Transformer框架HGformer。通过引入增强的图Transformer编码器和结构偏置，以及一种两代理分层决策模型，我们的方法能够在大规模对抗环境中高效生成策略。此外，我们设计了一种逐层反馈强化学习算法，将低层决策的长期回报反馈到高层策略的优化中，从而弥合两个决策阶段之间的协调差距。实验结果表明，与现有的分层决策方法或图神经网络方法相比，HGformer显著提高了资源分配效率和对抗收益，在复杂动态博弈场景中实现了卓越的整体性能。 

---
# Safe and Economical UAV Trajectory Planning in Low-Altitude Airspace: A Hybrid DRL-LLM Approach with Compliance Awareness 

**Title (ZH)**: 低空 airspace中安全经济的无人机航迹规划：一种具有合规意识的混合DRL-LLM方法 

**Authors**: Yanwei Gong, Xiaolin Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08532)  

**Abstract**: The rapid growth of the low-altitude economy has driven the widespread adoption of unmanned aerial vehicles (UAVs). This growing deployment presents new challenges for UAV trajectory planning in complex urban environments. However, existing studies often overlook key factors, such as urban airspace constraints and economic efficiency, which are essential in low-altitude economy contexts. Deep reinforcement learning (DRL) is regarded as a promising solution to these issues, while its practical adoption remains limited by low learning efficiency. To overcome this limitation, we propose a novel UAV trajectory planning framework that combines DRL with large language model (LLM) reasoning to enable safe, compliant, and economically viable path planning. Experimental results demonstrate that our method significantly outperforms existing baselines across multiple metrics, including data collection rate, collision avoidance, successful landing, regulatory compliance, and energy efficiency. These results validate the effectiveness of our approach in addressing UAV trajectory planning key challenges under constraints of the low-altitude economy networking. 

**Abstract (ZH)**: 低空经济快速发展的广泛应用推动了无人机（UAV）的普及。复杂城市环境下的无人机航迹规划面临新挑战。然而，现有研究往往忽视了低空经济背景下至关重要的因素，如城市 airspace约束和经济效率。深度强化学习（DRL）被视为解决这些问题的有希望的方法，但由于学习效率低，其实用应用受到限制。为克服这一限制，我们提出了一种结合DRL和大型语言模型（LLM）推理的新型无人机航迹规划框架，以实现安全、合规和经济可行的路径规划。实验结果表明，我们的方法在数据收集率、碰撞避险、成功降落、法规遵从性和能效等多个指标上显著优于现有baseline方法。这些结果验证了该方法在低空经济网络约束条件下解决无人机航迹规划关键挑战的有效性。 

---
# FEDTAIL: Federated Long-Tailed Domain Generalization with Sharpness-Guided Gradient Matching 

**Title (ZH)**: FEDTAIL：基于锋度引导梯度匹配的联邦长尾域泛化 

**Authors**: Sunny Gupta, Nikita Jangid, Shounak Das, Amit Sethi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08518)  

**Abstract**: Domain Generalization (DG) seeks to train models that perform reliably on unseen target domains without access to target data during training. While recent progress in smoothing the loss landscape has improved generalization, existing methods often falter under long-tailed class distributions and conflicting optimization objectives. We introduce FedTAIL, a federated domain generalization framework that explicitly addresses these challenges through sharpness-guided, gradient-aligned optimization. Our method incorporates a gradient coherence regularizer to mitigate conflicts between classification and adversarial objectives, leading to more stable convergence. To combat class imbalance, we perform class-wise sharpness minimization and propose a curvature-aware dynamic weighting scheme that adaptively emphasizes underrepresented tail classes. Furthermore, we enhance conditional distribution alignment by integrating sharpness-aware perturbations into entropy regularization, improving robustness under domain shift. FedTAIL unifies optimization harmonization, class-aware regularization, and conditional alignment into a scalable, federated-compatible framework. Extensive evaluations across standard domain generalization benchmarks demonstrate that FedTAIL achieves state-of-the-art performance, particularly in the presence of domain shifts and label imbalance, validating its effectiveness in both centralized and federated settings. Code: this https URL 

**Abstract (ZH)**: 联邦域自适应（FedTAIL）：通过尖括号导向的梯度对齐优化统一优化 harmonization、类感知正则化和条件对齐的联邦域泛化框架 

---
# RHealthTwin: Towards Responsible and Multimodal Digital Twins for Personalized Well-being 

**Title (ZH)**: RHealthTwin: 朝着负责任的多模态数字孪生以实现个性化福祉方向努力 

**Authors**: Rahatara Ferdousi, M Anwar Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2506.08486)  

**Abstract**: The rise of large language models (LLMs) has created new possibilities for digital twins in healthcare. However, the deployment of such systems in consumer health contexts raises significant concerns related to hallucination, bias, lack of transparency, and ethical misuse. In response to recommendations from health authorities such as the World Health Organization (WHO), we propose Responsible Health Twin (RHealthTwin), a principled framework for building and governing AI-powered digital twins for well-being assistance. RHealthTwin processes multimodal inputs that guide a health-focused LLM to produce safe, relevant, and explainable responses. At the core of RHealthTwin is the Responsible Prompt Engine (RPE), which addresses the limitations of traditional LLM configuration. Conventionally, users input unstructured prompt and the system instruction to configure the LLM, which increases the risk of hallucination. In contrast, RPE extracts predefined slots dynamically to structure both inputs. This guides the language model to generate responses that are context aware, personalized, fair, reliable, and explainable for well-being assistance. The framework further adapts over time through a feedback loop that updates the prompt structure based on user satisfaction. We evaluate RHealthTwin across four consumer health domains including mental support, symptom triage, nutrition planning, and activity coaching. RPE achieves state-of-the-art results with BLEU = 0.41, ROUGE-L = 0.63, and BERTScore = 0.89 on benchmark datasets. Also, we achieve over 90% in ethical compliance and instruction-following metrics using LLM-as-judge evaluation, outperforming baseline strategies. We envision RHealthTwin as a forward-looking foundation for responsible LLM-based applications in health and well-being. 

**Abstract (ZH)**: 大语言模型的兴起为医疗领域的数字孪生带来了新机遇，但在消费者健康领域部署此类系统引发了关于幻觉、偏见、透明度不足和伦理滥用的重大关切。根据世界卫生组织等卫生当局的建议，我们提出了负责任健康数字孪生（RHealthTwin）原则框架，以构建和治理用于福祉辅助的AI驱动数字孪生。RHealthTwin处理多模态输入，引导专注于健康的大语言模型生成安全、相关且可解释的响应。RHealthTwin的核心是负责任提示引擎（RPE），它解决了传统大语言模型配置的局限性。传统上，用户输入未结构化的提示和系统指令来配置大语言模型，增加了幻觉的风险。相反，RPE 动态提取预定义的槽位来结构化输入。这引导语言模型生成有上下文意识、个性化、公平、可靠和可解释的响应，以辅助福祉。该框架还通过一个反馈循环进行适应，该循环根据用户满意度更新提示结构。我们在包括心理健康支持、症状分类、营养规划和活动指导在内的四个消费者健康领域评估了RHealthTwin。RPE在基准数据集上实现了最先进的结果，BLEU得分为0.41，ROUGE-L得分为0.63，BERTScore得分为0.89。此外，我们在LLM作为评判者的伦理合规性和指令遵循度指标中达到了超过90%的表现，超越了基线策略。我们展望RHealthTwin将成为医疗和福祉领域负责任的大语言模型应用的基础。 

---
# Hybrid Reasoning for Perception, Explanation, and Autonomous Action in Manufacturing 

**Title (ZH)**: 制造领域中的混合推理以实现感知、解释和自主行动 

**Authors**: Christos Margadji, Sebastian W. Pattinson  

**Link**: [PDF](https://arxiv.org/pdf/2506.08462)  

**Abstract**: Industrial processes must be robust and adaptable, as environments and tasks are often unpredictable, while operational errors remain costly and difficult to detect. AI-based control systems offer a path forward, yet typically depend on supervised learning with extensive labelled datasets, which limits their ability to generalize across variable and data-scarce industrial settings. Foundation models could enable broader reasoning and knowledge integration, but rarely deliver the quantitative precision demanded by engineering applications. Here, we introduceControl and Interpretation of Production via Hybrid Expertise and Reasoning (CIPHER): a vision-language-action (VLA) model framework aiming to replicate human-like reasoning for industrial control, instantiated in a commercial-grade 3D printer. It integrates a process expert, a regression model enabling quantitative characterization of system states required for engineering tasks. CIPHER also incorporates retrieval-augmented generation to access external expert knowledge and support physics-informed, chain-of-thought reasoning. This hybrid architecture exhibits strong generalization to out-of-distribution tasks. It interprets visual or textual inputs from process monitoring, explains its decisions, and autonomously generates precise machine instructions, without requiring explicit annotations. CIPHER thus lays the foundations for autonomous systems that act with precision, reason with context, and communicate decisions transparently, supporting safe and trusted deployment in industrial settings. 

**Abstract (ZH)**: 基于混合专家知识与推理的生产控制与解释（CIPHER）：一种视角-语言-动作模型框架 

---
# A Survey on Large Language Models for Mathematical Reasoning 

**Title (ZH)**: 大型语言模型在数学推理中的研究综述 

**Authors**: Peng-Yuan Wang, Tian-Shuo Liu, Chenyang Wang, Yi-Di Wang, Shu Yan, Cheng-Xing Jia, Xu-Hui Liu, Xin-Wei Chen, Jia-Cheng Xu, Ziniu Li, Yang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08446)  

**Abstract**: Mathematical reasoning has long represented one of the most fundamental and challenging frontiers in artificial intelligence research. In recent years, large language models (LLMs) have achieved significant advances in this area. This survey examines the development of mathematical reasoning abilities in LLMs through two high-level cognitive phases: comprehension, where models gain mathematical understanding via diverse pretraining strategies, and answer generation, which has progressed from direct prediction to step-by-step Chain-of-Thought (CoT) reasoning. We review methods for enhancing mathematical reasoning, ranging from training-free prompting to fine-tuning approaches such as supervised fine-tuning and reinforcement learning, and discuss recent work on extended CoT and "test-time scaling". Despite notable progress, fundamental challenges remain in terms of capacity, efficiency, and generalization. To address these issues, we highlight promising research directions, including advanced pretraining and knowledge augmentation techniques, formal reasoning frameworks, and meta-generalization through principled learning paradigms. This survey tries to provide some insights for researchers interested in enhancing reasoning capabilities of LLMs and for those seeking to apply these techniques to other domains. 

**Abstract (ZH)**: 数学推理一直是人工智能研究中最基本也是最具挑战性的前沿领域之一。近年来，大型语言模型（LLMs）在这一领域取得了显著进展。本文综述了LLMs在两个高层次认知阶段中数学推理能力的发展：理解阶段，模型通过多样化的预训练策略获得数学理解；以及从直接预测到逐步链式思考（CoT）推理的答案生成阶段。我们回顾了从无训练提示到微调方法（如监督微调和强化学习）增强数学推理的方法，并讨论了扩展CoT和“测试时扩展”等相关工作。尽管取得了一定的进展，但在容量、效率和泛化方面仍然存在根本性的挑战。为解决这些问题，我们强调了有前景的研究方向，包括先进的预训练和知识增强技术、形式推理框架以及通过原则性的学习范式实现元泛化。本文旨在为致力于提高LLMs推理能力的研究者和希望将这些技术应用于其他领域的研究者提供一些见解。 

---
# SHIELD: Multi-task Multi-distribution Vehicle Routing Solver with Sparsity and Hierarchy 

**Title (ZH)**: SHIELD：具有稀疏性和层次性的多任务多分布车辆路由求解器 

**Authors**: Yong Liang Goh, Zhiguang Cao, Yining Ma, Jianan Zhou, Mohammad Haroon Dupty, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.08424)  

**Abstract**: Recent advances toward foundation models for routing problems have shown great potential of a unified deep model for various VRP variants. However, they overlook the complex real-world customer distributions. In this work, we advance the Multi-Task VRP (MTVRP) setting to the more realistic yet challenging Multi-Task Multi-Distribution VRP (MTMDVRP) setting, and introduce SHIELD, a novel model that leverages both sparsity and hierarchy principles. Building on a deeper decoder architecture, we first incorporate the Mixture-of-Depths (MoD) technique to enforce sparsity. This improves both efficiency and generalization by allowing the model to dynamically select nodes to use or skip each decoder layer, providing the needed capacity to adaptively allocate computation for learning the task/distribution specific and shared representations. We also develop a context-based clustering layer that exploits the presence of hierarchical structures in the problems to produce better local representations. These two designs inductively bias the network to identify key features that are common across tasks and distributions, leading to significantly improved generalization on unseen ones. Our empirical results demonstrate the superiority of our approach over existing methods on 9 real-world maps with 16 VRP variants each. 

**Abstract (ZH)**: 最近针对路由问题的基础模型研究展示了统一深度模型在各种VRP变体中的巨大潜力。然而，它们忽略了复杂的现实世界客户分布。在本工作中，我们将多任务VRP (MTVRP) 设置推进到更具现实意义且更具挑战性的多任务多分布VRP (MTMDVRP) 设置，并引入了SHIELD模型，该模型结合了稀疏性和分层原则。基于更深层的解码器架构，我们首先引入了Mixture-of-Depths (MoD) 技术以增强稀疏性。这通过使模型能够动态选择使用或跳过每个解码器层来提高效率和泛化能力，提供了根据不同任务/分布特定和共享表示自适应分配计算所需的能力。我们还开发了一种基于上下文的聚类层，利用问题中存在的分层结构来产生更好的局部表示。这两种设计使网络在识别跨任务和分布的常见特征方面具有引导性，从而在未见过的任务上显著提高了泛化能力。我们的实证结果证明了在9张现实世界地图上的16个不同的VRP变体上，我们的方法优于现有方法。 

---
# Transforming Expert Knowledge into Scalable Ontology via Large Language Models 

**Title (ZH)**: 通过大型语言模型将专家知识转化为可扩展本体 

**Authors**: Ikkei Itoku, David Theil, Evelyn Eichelsdoerfer Uehara, Sreyoshi Bhaduri, Junnosuke Kuroda, Toshi Yumoto, Alex Gil, Natalie Perez, Rajesh Cherukuri, Naumaan Nayyar  

**Link**: [PDF](https://arxiv.org/pdf/2506.08422)  

**Abstract**: Having a unified, coherent taxonomy is essential for effective knowledge representation in domain-specific applications as diverse terminologies need to be mapped to underlying concepts. Traditional manual approaches to taxonomy alignment rely on expert review of concept pairs, but this becomes prohibitively expensive and time-consuming at scale, while subjective interpretations often lead to expert disagreements. Existing automated methods for taxonomy alignment have shown promise but face limitations in handling nuanced semantic relationships and maintaining consistency across different domains. These approaches often struggle with context-dependent concept mappings and lack transparent reasoning processes. We propose a novel framework that combines large language models (LLMs) with expert calibration and iterative prompt optimization to automate taxonomy alignment. Our method integrates expert-labeled examples, multi-stage prompt engineering, and human validation to guide LLMs in generating both taxonomy linkages and supporting rationales. In evaluating our framework on a domain-specific mapping task of concept essentiality, we achieved an F1-score of 0.97, substantially exceeding the human benchmark of 0.68. These results demonstrate the effectiveness of our approach in scaling taxonomy alignment while maintaining high-quality mappings and preserving expert oversight for ambiguous cases. 

**Abstract (ZH)**: 一种结合大规模语言模型、专家校准和迭代提示优化的税务分类自动化框架 

---
# Single-Node Trigger Backdoor Attacks in Graph-Based Recommendation Systems 

**Title (ZH)**: 基于图的推荐系统中的单节点触发后门攻击 

**Authors**: Runze Li, Di Jin, Xiaobao Wang, Dongxiao He, Bingdao Feng, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08401)  

**Abstract**: Graph recommendation systems have been widely studied due to their ability to effectively capture the complex interactions between users and items. However, these systems also exhibit certain vulnerabilities when faced with attacks. The prevailing shilling attack methods typically manipulate recommendation results by injecting a large number of fake nodes and edges. However, such attack strategies face two primary challenges: low stealth and high destructiveness. To address these challenges, this paper proposes a novel graph backdoor attack method that aims to enhance the exposure of target items to the target user in a covert manner, without affecting other unrelated nodes. Specifically, we design a single-node trigger generator, which can effectively expose multiple target items to the target user by inserting only one fake user node. Additionally, we introduce constraint conditions between the target nodes and irrelevant nodes to mitigate the impact of fake nodes on the recommendation system's performance. Experimental results show that the exposure of the target items reaches no less than 50% in 99% of the target users, while the impact on the recommendation system's performance is controlled within approximately 5%. 

**Abstract (ZH)**: 图推荐系统的图后门攻击方法：隐蔽增加目标项的暴露同时控制推荐系统性能影响 

---
# SafeCoT: Improving VLM Safety with Minimal Reasoning 

**Title (ZH)**: SafeCoT：通过最小推理提高VLM安全性 

**Authors**: Jiachen Ma, Zhanhui Zhou, Chao Yang, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08399)  

**Abstract**: Ensuring safe and appropriate responses from vision-language models (VLMs) remains a critical challenge, particularly in high-risk or ambiguous scenarios. We introduce SafeCoT, a lightweight, interpretable framework that leverages rule-based chain-of-thought (CoT) supervision to improve refusal behavior in VLMs. Unlike prior methods that rely on large-scale safety annotations or complex modeling, SafeCoT uses minimal supervision to help models reason about safety risks and make context-aware refusals. Experiments across multiple benchmarks show that SafeCoT significantly reduces overrefusal and enhances generalization, even with limited training data. Our approach offers a scalable solution for aligning VLMs with safety-critical objectives. 

**Abstract (ZH)**: 确保视觉语言模型在高风险或模糊场景下提供安全且适当的回应仍是一项关键挑战。我们提出了一种名为SafeCoT的轻量级可解释框架，该框架利用基于规则的链式思考（CoT）监督来提高视觉语言模型的拒绝行为。与依赖大规模安全注释或复杂模型的先前方法不同，SafeCoT采用最少的监督来帮助模型推理安全风险并做出情境相关的拒绝。多项基准测试的结果表明，SafeCoT显著减少了过度拒绝并增强了泛化能力，即使在有限的训练数据下也是如此。我们的方法为使视觉语言模型与安全关键目标保持一致提供了可扩展的解决方案。 

---
# On Reasoning Strength Planning in Large Reasoning Models 

**Title (ZH)**: 在大型推理模型中的推理强度规划 

**Authors**: Leheng Sheng, An Zhang, Zijian Wu, Weixiang Zhao, Changshuo Shen, Yi Zhang, Xiang Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.08390)  

**Abstract**: Recent studies empirically reveal that large reasoning models (LRMs) can automatically allocate more reasoning strengths (i.e., the number of reasoning tokens) for harder problems, exhibiting difficulty-awareness for better task performance. While this automatic reasoning strength allocation phenomenon has been widely observed, its underlying mechanism remains largely unexplored. To this end, we provide explanations for this phenomenon from the perspective of model activations. We find evidence that LRMs pre-plan the reasoning strengths in their activations even before generation, with this reasoning strength causally controlled by the magnitude of a pre-allocated directional vector. Specifically, we show that the number of reasoning tokens is predictable solely based on the question activations using linear probes, indicating that LRMs estimate the required reasoning strength in advance. We then uncover that LRMs encode this reasoning strength through a pre-allocated directional vector embedded in the activations of the model, where the vector's magnitude modulates the reasoning strength. Subtracting this vector can lead to reduced reasoning token number and performance, while adding this vector can lead to increased reasoning token number and even improved performance. We further reveal that this direction vector consistently yields positive reasoning length prediction, and it modifies the logits of end-of-reasoning token </think> to affect the reasoning length. Finally, we demonstrate two potential applications of our findings: overthinking behavior detection and enabling efficient reasoning on simple problems. Our work provides new insights into the internal mechanisms of reasoning in LRMs and offers practical tools for controlling their reasoning behaviors. Our code is available at this https URL. 

**Abstract (ZH)**: 最近的研究实证表明，大型推理模型（LRMs）能够自动为更难的问题分配更多的推理强度（即推理标记的数量），展现出对任务性能的难度感知能力。虽然这种自动推理强度分配的现象已被广泛观察到，但其背后的机制仍然 largely unexplored。为此，我们从模型激活的角度提供了对该现象的解释。我们发现证据表明，LRMs 在生成之前就已经在其激活中预先规划了推理强度，并且这种推理强度是由预分配的方向向量的大小因果控制的。具体来说，我们展示了仅基于问题激活使用线性探针即可预测推理标记的数量，表明LRMs 在生成之前会预先估计所需的推理强度。然后，我们揭示LRMs 通过嵌入在模型激活中的预分配方向向量来编码这种推理强度，其中向量的大小调节推理强度。减去这个向量会导致推理标记数量减少和性能下降，而增加这个向量会导致推理标记数量增加，甚至性能提升。我们进一步揭示这种方向向量会一致地产生积极的推理长度预测，并通过调整端止于推理标记</think>的logits来影响推理长度。最后，我们展示了我们发现的两种潜在应用：过度推理行为检测和在简单问题上实现高效推理。我们的工作为LRMs 的推理内部机制提供了新的见解，并提供了控制其推理行为的实用工具。我们已将代码发布在如下地址：this https URL。 

---
# FloorplanMAE:A self-supervised framework for complete floorplan generation from partial inputs 

**Title (ZH)**: FloorplanMAE：一种基于部分输入的完全平面图自监督生成框架 

**Authors**: Jun Yin, Jing Zhong, Pengyu Zeng, Peilin Li, Miao Zhang, Ran Luo, Shuai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08363)  

**Abstract**: In the architectural design process, floorplan design is often a dynamic and iterative process. Architects progressively draw various parts of the floorplan according to their ideas and requirements, continuously adjusting and refining throughout the design process. Therefore, the ability to predict a complete floorplan from a partial one holds significant value in the design process. Such prediction can help architects quickly generate preliminary designs, improve design efficiency, and reduce the workload associated with repeated modifications. To address this need, we propose FloorplanMAE, a self-supervised learning framework for restoring incomplete floor plans into complete ones. First, we developed a floor plan reconstruction dataset, FloorplanNet, specifically trained on architectural floor plans. Secondly, we propose a floor plan reconstruction method based on Masked Autoencoders (MAE), which reconstructs missing parts by masking sections of the floor plan and training a lightweight Vision Transformer (ViT). We evaluated the reconstruction accuracy of FloorplanMAE and compared it with state-of-the-art benchmarks. Additionally, we validated the model using real sketches from the early stages of architectural design. Experimental results show that the FloorplanMAE model can generate high-quality complete floor plans from incomplete partial plans. This framework provides a scalable solution for floor plan generation, with broad application prospects. 

**Abstract (ZH)**: 在建筑设计过程中，平面图设计通常是动态和迭代的过程。设计者根据自己的理念和需求逐步绘制平面图的不同部分，并在整个设计过程中不断调整和优化。因此，从不完整的平面图预测完整的平面图的能力在设计过程中具有重要意义。这种预测可以帮助设计者快速生成初步设计，提高设计效率，并减少重复修改的工作量。为应对这一需求，我们提出FloorplanMAE，这是一种自监督学习框架，用于将不完整的平面图恢复为完整的平面图。首先，我们开发了一个专门针对建筑平面图训练的平面图重建数据集FloorplanNet。其次，我们提出了一种基于Masked Autoencoders (MAE)的平面图重建方法，通过掩蔽平面图的部分区域并训练轻量级的Vision Transformer (ViT)来重建缺失的部分。我们评估了FloorplanMAE的重建精度，并与最新的基准进行了对比。此外，我们还使用了建筑设计初期的真实草图验证了该模型。实验结果表明，FloorplanMAE模型可以从不完整的部分平面图生成高质量的完整平面图。该框架为平面图生成提供了可扩展的解决方案，具有广泛的应用前景。 

---
# ORFS-agent: Tool-Using Agents for Chip Design Optimization 

**Title (ZH)**: ORFS-agent：用于芯片设计优化的工具使用智能体 

**Authors**: Amur Ghose, Andrew B. Kahng, Sayak Kundu, Zhiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08332)  

**Abstract**: Machine learning has been widely used to optimize complex engineering workflows across numerous domains. In the context of integrated circuit design, modern flows (e.g., going from a register-transfer level netlist to physical layouts) involve extensive configuration via thousands of parameters, and small changes to these parameters can have large downstream impacts on desired outcomes - namely design performance, power, and area. Recent advances in Large Language Models (LLMs) offer new opportunities for learning and reasoning within such high-dimensional optimization tasks. In this work, we introduce ORFS-agent, an LLM-based iterative optimization agent that automates parameter tuning in an open-source hardware design flow. ORFS-agent adaptively explores parameter configurations, demonstrating clear improvements over standard Bayesian optimization approaches in terms of resource efficiency and final design metrics. Our empirical evaluations on two different technology nodes and a range of circuit benchmarks indicate that ORFS-agent can improve both routed wirelength and effective clock period by over 13%, all while using 40% fewer optimization iterations. Moreover, by following natural language objectives to trade off certain metrics for others, ORFS-agent demonstrates a flexible and interpretable framework for multi-objective optimization. Crucially, RFS-agent is modular and model-agnostic, and can be plugged in to any frontier LLM without any further fine-tuning. 

**Abstract (ZH)**: 基于大规模语言模型的开放源码硬件设计流程中的迭代优化代理ORFS-agent 

---
# LeanTutor: A Formally-Verified AI Tutor for Mathematical Proofs 

**Title (ZH)**: LeanTutor: 一个形式验证的数学证明AI导师 

**Authors**: Manooshree Patel, Rayna Bhattacharyya, Thomas Lu, Arnav Mehta, Niels Voss, Narges Norouzi, Gireeja Ranade  

**Link**: [PDF](https://arxiv.org/pdf/2506.08321)  

**Abstract**: We present LeanTutor, a Large Language Model (LLM)-based tutoring system for math proofs. LeanTutor interacts with the student in natural language, formally verifies student-written math proofs in Lean, generates correct next steps, and provides the appropriate instructional guidance. LeanTutor is composed of three modules: (i) an autoformalizer/proof-checker, (ii) a next-step generator, and (iii) a natural language feedback generator. The first module faithfully autoformalizes student proofs into Lean and verifies proof accuracy via successful code compilation. If the proof has an error, the incorrect step is identified. The next-step generator module outputs a valid next Lean tactic for incorrect proofs via LLM-based candidate generation and proof search. The feedback generator module leverages Lean data to produce a pedagogically-motivated natural language hint for the student user. To evaluate our system, we introduce PeanoBench, a human-written dataset derived from the Natural Numbers Game, consisting of 371 Peano Arithmetic proofs, where each natural language proof step is paired with the corresponding logically equivalent tactic in Lean. The Autoformalizer correctly formalizes 57% of tactics in correct proofs and accurately identifies the incorrect step in 30% of incorrect proofs. In generating natural language hints for erroneous proofs, LeanTutor outperforms a simple baseline on accuracy and relevance metrics. 

**Abstract (ZH)**: 基于大型语言模型的数学证明辅导系统LeanTutor 

---
# AstroCompress: A benchmark dataset for multi-purpose compression of astronomical data 

**Title (ZH)**: AstroCompress：一个多用途的天文书证数据压缩基准数据集 

**Authors**: Tuan Truong, Rithwik Sudharsan, Yibo Yang, Peter Xiangyuan Ma, Ruihan Yang, Stephan Mandt, Joshua S. Bloom  

**Link**: [PDF](https://arxiv.org/pdf/2506.08306)  

**Abstract**: The site conditions that make astronomical observatories in space and on the ground so desirable -- cold and dark -- demand a physical remoteness that leads to limited data transmission capabilities. Such transmission limitations directly bottleneck the amount of data acquired and in an era of costly modern observatories, any improvements in lossless data compression has the potential scale to billions of dollars worth of additional science that can be accomplished on the same instrument. Traditional lossless methods for compressing astrophysical data are manually designed. Neural data compression, on the other hand, holds the promise of learning compression algorithms end-to-end from data and outperforming classical techniques by leveraging the unique spatial, temporal, and wavelength structures of astronomical images. This paper introduces AstroCompress: a neural compression challenge for astrophysics data, featuring four new datasets (and one legacy dataset) with 16-bit unsigned integer imaging data in various modes: space-based, ground-based, multi-wavelength, and time-series imaging. We provide code to easily access the data and benchmark seven lossless compression methods (three neural and four non-neural, including all practical state-of-the-art algorithms). Our results on lossless compression indicate that lossless neural compression techniques can enhance data collection at observatories, and provide guidance on the adoption of neural compression in scientific applications. Though the scope of this paper is restricted to lossless compression, we also comment on the potential exploration of lossy compression methods in future studies. 

**Abstract (ZH)**: 空间和地面天文观测站的理想场所条件——寒冷和黑暗——要求物理上的远离，这导致了有限的数据传输能力。这种传输限制直接制约了获取的数据量，在现代观测站成本高昂的时代，任何在无损数据压缩上的改进都有可能带来数以十亿计美元的额外科学成果。传统无损压缩天体物理数据的方法是手动设计的。相比之下，神经数据压缩有望从数据中端到端学习压缩算法，并通过利用天文图像的独特空间、时间和波长结构超越经典技术。本文介绍了AstroCompress：一个针对天体物理学数据的神经压缩挑战，包含四个新数据集（以及一个遗产数据集），涵盖16位无符号整数成像数据的各种模式：基于空间的、地面的、多波段的和时间序列成像。我们提供了易于访问数据和评估七种无损压缩方法（三种神经和四种非神经，包括所有实际的最新算法）的代码。我们的无损压缩结果表明，无损神经压缩技术可以提高观测站的数据采集能力，并为在科学应用中采用神经压缩提供指导。尽管本文的范围仅限于无损压缩，我们还对未来研究中探索有损压缩方法的可能性进行了评论。 

---
# Compiling Metric Temporal Answer Set Programming 

**Title (ZH)**: 编译度量时序回答集程序设计 

**Authors**: Arvid Becker, Pedro Cabalar, Martin Diéguez, Javier Romero, Susana Hahn, Torsten Schaub  

**Link**: [PDF](https://arxiv.org/pdf/2506.08150)  

**Abstract**: We develop a computational approach to Metric Answer Set Programming (ASP) to allow for expressing quantitative temporal constrains, like durations and deadlines. A central challenge is to maintain scalability when dealing with fine-grained timing constraints, which can significantly exacerbate ASP's grounding bottleneck. To address this issue, we leverage extensions of ASP with difference constraints, a simplified form of linear constraints, to handle time-related aspects externally. Our approach effectively decouples metric ASP from the granularity of time, resulting in a solution that is unaffected by time precision. 

**Abstract (ZH)**: 我们开发了一种计算方法来扩展度量回答集编程（ASP），以表达定量的时间约束，如持续时间和截止时间。主要挑战在于在处理细粒度的时间约束时保持可扩展性，这可能会显著加剧ASP的底座瓶颈。为解决这一问题，我们利用带有差分约束的ASP扩展，这是一种简化的线性约束形式，将时间相关的方面外部处理。这种方法有效地将度量ASP与时间的粒度分离，从而得到一个不受时间精度影响的解决方案。 

---
# The AI Imperative: Scaling High-Quality Peer Review in Machine Learning 

**Title (ZH)**: AI的必然性：扩展高质量同行评审在机器学习中的应用 

**Authors**: Qiyao Wei, Samuel Holt, Jing Yang, Markus Wulfmeier, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.08134)  

**Abstract**: Peer review, the bedrock of scientific advancement in machine learning (ML), is strained by a crisis of scale. Exponential growth in manuscript submissions to premier ML venues such as NeurIPS, ICML, and ICLR is outpacing the finite capacity of qualified reviewers, leading to concerns about review quality, consistency, and reviewer fatigue. This position paper argues that AI-assisted peer review must become an urgent research and infrastructure priority. We advocate for a comprehensive AI-augmented ecosystem, leveraging Large Language Models (LLMs) not as replacements for human judgment, but as sophisticated collaborators for authors, reviewers, and Area Chairs (ACs). We propose specific roles for AI in enhancing factual verification, guiding reviewer performance, assisting authors in quality improvement, and supporting ACs in decision-making. Crucially, we contend that the development of such systems hinges on access to more granular, structured, and ethically-sourced peer review process data. We outline a research agenda, including illustrative experiments, to develop and validate these AI assistants, and discuss significant technical and ethical challenges. We call upon the ML community to proactively build this AI-assisted future, ensuring the continued integrity and scalability of scientific validation, while maintaining high standards of peer review. 

**Abstract (ZH)**: 机器学习（ML）领域科学进步基石的同行评审面临规模危机：AI辅助同行评审亟待成为研究和基础设施的优先事项 

---
# SOP-Bench: Complex Industrial SOPs for Evaluating LLM Agents 

**Title (ZH)**: SOP-Bench: 复杂工业标准操作程序用于评估大规模语言模型代理 

**Authors**: Subhrangshu Nandi, Arghya Datta, Nikhil Vichare, Indranil Bhattacharya, Huzefa Raja, Jing Xu, Shayan Ray, Giuseppe Carenini, Abhi Srivastava, Aaron Chan, Man Ho Woo, Amar Kandola, Brandon Theresa, Francesco Carbone  

**Link**: [PDF](https://arxiv.org/pdf/2506.08119)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive general-purpose reasoning and problem-solving abilities. However, they struggle with executing complex, long-horizon workflows that demand strict adherence to Standard Operating Procedures (SOPs), a critical requirement for real-world industrial automation. Despite this need, there is a lack of public benchmarks that reflect the complexity, structure, and domain-specific nuances of SOPs. To address this, we present three main contributions. First, we introduce a synthetic data generation framework to create realistic, industry-grade SOPs that rigorously test the planning, reasoning, and tool-use capabilities of LLM-based agents. Second, using this framework, we develop SOP-Bench, a benchmark of over 1,800 tasks across 10 industrial domains, each with APIs, tool interfaces, and human-validated test cases. Third, we evaluate two prominent agent architectures: Function-Calling and ReAct Agents, on SOP-Bench, observing average success rates of only 27% and 48%, respectively. Remarkably, when the tool registry is much larger than necessary, agents invoke incorrect tools nearly 100% of the time. These findings underscore a substantial gap between current agentic capabilities of LLMs and the demands of automating real-world SOPs. Performance varies significantly by task and domain, highlighting the need for domain-specific benchmarking and architectural choices before deployment. SOP-Bench is publicly available at this http URL. We also release the prompts underpinning the data generation framework to support new domain-specific SOP benchmarks. We invite the community to extend SOP-Bench with SOPs from their industrial domains. 

**Abstract (ZH)**: 大规模语言模型（LLMs）展示了 impressive 的通用推理和问题解决能力。然而，它们在执行需要严格遵循标准操作程序（SOPs）的复杂、长期工作流程时表现不佳，这是现实世界工业自动化的一个关键要求。尽管有这一需求，仍缺乏反映SOP复杂性、结构和领域特定细微差别的公开基准。为解决这一问题，我们提出了三项主要贡献。首先，我们介绍了一种合成数据生成框架，用于创建真实且符合工业标准的SOP，以严格测试基于语言模型的代理的规划、推理和工具使用能力。其次，使用此框架，我们开发了SOP-Bench基准测试，包括来自10个工业领域的超过1,800项任务，每项任务都包含API、工具接口和由人类验证的测试案例。第三，我们评估了两种主要的代理架构：函数调用和ReAct代理，在SOP-Bench上的平均成功率分别为27%和48%。显著的是，当工具注册表远超所需时，代理几乎每次都会调用错误的工具。这些发现突显了当前语言模型在代理能力与自动化现实世界SOP需求之间的巨大差距。不同任务和领域间性能差异显著，强调了在部署前需要进行领域特定基准测试和架构选择。SOP-Bench已公开发布，可在该网址访问。我们还发布了支撑数据生成框架的提示，以支持新的领域特定SOP基准测试。我们邀请社区使用其工业领域的SOP扩展SOP-Bench。 

---
# Cognitive Weave: Synthesizing Abstracted Knowledge with a Spatio-Temporal Resonance Graph 

**Title (ZH)**: 认知织就：基于时空共振图综合抽象知识 

**Authors**: Akash Vishwakarma, Hojin Lee, Mohith Suresh, Priyam Shankar Sharma, Rahul Vishwakarma, Sparsh Gupta, Yuvraj Anupam Chauhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08098)  

**Abstract**: The emergence of capable large language model (LLM) based agents necessitates memory architectures that transcend mere data storage, enabling continuous learning, nuanced reasoning, and dynamic adaptation. Current memory systems often grapple with fundamental limitations in structural flexibility, temporal awareness, and the ability to synthesize higher-level insights from raw interaction data. This paper introduces Cognitive Weave, a novel memory framework centered around a multi-layered spatio-temporal resonance graph (STRG). This graph manages information as semantically rich insight particles (IPs), which are dynamically enriched with resonance keys, signifiers, and situational imprints via a dedicated semantic oracle interface (SOI). These IPs are interconnected through typed relational strands, forming an evolving knowledge tapestry. A key component of Cognitive Weave is the cognitive refinement process, an autonomous mechanism that includes the synthesis of insight aggregates (IAs) condensed, higher-level knowledge structures derived from identified clusters of related IPs. We present comprehensive experimental results demonstrating Cognitive Weave's marked enhancement over existing approaches in long-horizon planning tasks, evolving question-answering scenarios, and multi-session dialogue coherence. The system achieves a notable 34% average improvement in task completion rates and a 42% reduction in mean query latency when compared to state-of-the-art baselines. Furthermore, this paper explores the ethical considerations inherent in such advanced memory systems, discusses the implications for long-term memory in LLMs, and outlines promising future research trajectories. 

**Abstract (ZH)**: 具备强大能力的大语言模型（LLM）代理的出现 necessitates 内存架构超越 mere 数据存储，以实现持续学习、细致推理和动态适应。当前的内存系统往往在结构灵活性、时间意识和从原始交互数据中综合高级洞察方面面临根本性的限制。本文介绍了一种新型的记忆框架认知编织（Cognitive Weave），该框架以多层时空共振图（STRG）为核心。该图以语义丰富的洞察颗粒（IPs）来管理信息，并通过专用语义Oracle接口（SOI）动态增强这些IPs的共振键、标识符和情境印记。这些IPs通过类型化的关系纽带相互连接，形成一个不断演变的知识织锦。认知编织的关键组成部分是认知精炼过程，这是一种自主机制，包括从识别相关的IPs群集汇总形成的洞察聚合（IAs）的综合，这些IAs是凝练的、高级的知识结构。本文展示了认知编织在长周期规划任务、不断演化的问答场景和多会话对话连贯性方面显著优于现有方法的实验结果。与最先进的基线相比，该系统在任务完成率上平均提高了34%，查询延迟减少了42%。此外，本文探讨了此类高级记忆系统中固有的伦理问题，讨论了其对LLM长期记忆的影响，并概述了有希望的未来研究方向。 

---
# TIP-Search: Time-Predictable Inference Scheduling for Market Prediction under Uncertain Load 

**Title (ZH)**: TIP-Search: 时间可预测推断调度以应对不确定负载的市场预测 

**Authors**: Xibai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08026)  

**Abstract**: This paper proposes TIP-Search, a time-predictable inference scheduling framework for real-time market prediction under uncertain workloads. Motivated by the strict latency demands in high-frequency financial systems, TIP-Search dynamically selects a deep learning model from a heterogeneous pool, aiming to maximize predictive accuracy while satisfying per-task deadline constraints. Our approach profiles latency and generalization performance offline, then performs online task-aware selection without relying on explicit input domain labels. We evaluate TIP-Search on three real-world limit order book datasets (FI-2010, Binance BTC/USDT, LOBSTER AAPL) and demonstrate that it outperforms static baselines with up to 8.5% improvement in accuracy and 100% deadline satisfaction. Our results highlight the effectiveness of TIP-Search in robust low-latency financial inference under uncertainty. 

**Abstract (ZH)**: TIP-Search：面向不确定工作负载的实时市场预测可预测时间推理调度框架 

---
# Agentic Neural Networks: Self-Evolving Multi-Agent Systems via Textual Backpropagation 

**Title (ZH)**: 代理神经网络：通过文本反向传播的自演化多代理系统 

**Authors**: Xiaowen Ma, Chenyang Lin, Yao Zhang, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.09046)  

**Abstract**: Leveraging multiple Large Language Models(LLMs) has proven effective for addressing complex, high-dimensional tasks, but current approaches often rely on static, manually engineered multi-agent configurations. To overcome these constraints, we present the Agentic Neural Network(ANN), a framework that conceptualizes multi-agent collaboration as a layered neural network architecture. In this design, each agent operates as a node, and each layer forms a cooperative "team" focused on a specific subtask. Agentic Neural Network follows a two-phase optimization strategy: (1) Forward Phase-Drawing inspiration from neural network forward passes, tasks are dynamically decomposed into subtasks, and cooperative agent teams with suitable aggregation methods are constructed layer by layer. (2) Backward Phase-Mirroring backpropagation, we refine both global and local collaboration through iterative feedback, allowing agents to self-evolve their roles, prompts, and coordination. This neuro-symbolic approach enables ANN to create new or specialized agent teams post-training, delivering notable gains in accuracy and adaptability. Across four benchmark datasets, ANN surpasses leading multi-agent baselines under the same configurations, showing consistent performance improvements. Our findings indicate that ANN provides a scalable, data-driven framework for multi-agent systems, combining the collaborative capabilities of LLMs with the efficiency and flexibility of neural network principles. We plan to open-source the entire framework. 

**Abstract (ZH)**: 利用多个大型语言模型（LLMs）已证明对处理复杂、高维任务有效，但当前方法往往依赖于静态的手工工程化多智能体配置。为克服这些限制，我们提出了智能神经网络（Agentic Neural Network，ANN）框架，该框架将多智能体合作概念化为分层神经网络架构。在此设计中，每个智能体作为节点运作，每一层形成一个专注于特定子任务的“合作团队”。ANN遵循两阶段优化策略：（1）前向阶段——借鉴神经网络前向传播的概念，任务被动态分解为子任务，并逐层构建合适的合作智能体团队及其聚合方法。（2）后向阶段——模拟反向传播，通过迭代反馈精炼全局和局部合作，使智能体能够自我进化其角色、提示和协作方式。这种神经符号方法使ANN能够在训练后创建新的或专门的智能体团队，实现显著的准确性和适应性提升。ANN在四个基准数据集中，即使在相同的配置下也超越了最先进的多智能体基线，展示了一致的性能改进。我们的研究结果表明，ANN为多智能体系统提供了一个可扩展的数据驱动框架，结合了LLMs的协作能力和神经网络原则的效率与灵活性。我们计划开源整个框架。 

---
# Autoregressive Semantic Visual Reconstruction Helps VLMs Understand Better 

**Title (ZH)**: 自回归语义视觉重建有助于提升大模型的理解能力 

**Authors**: Dianyi Wang, Wei Song, Yikun Wang, Siyuan Wang, Kaicheng Yu, Zhongyu Wei, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09040)  

**Abstract**: Typical large vision-language models (LVLMs) apply autoregressive supervision solely to textual sequences, without fully incorporating the visual modality into the learning process. This results in three key limitations: (1) an inability to utilize images without accompanying captions, (2) the risk that captions omit critical visual details, and (3) the challenge that certain vision-centric content cannot be adequately conveyed through text. As a result, current LVLMs often prioritize vision-to-language alignment while potentially overlooking fine-grained visual information. While some prior works have explored autoregressive image generation, effectively leveraging autoregressive visual supervision to enhance image understanding remains an open challenge. In this paper, we introduce Autoregressive Semantic Visual Reconstruction (ASVR), which enables joint learning of visual and textual modalities within a unified autoregressive framework. We show that autoregressively reconstructing the raw visual appearance of images does not enhance and may even impair multimodal understanding. In contrast, autoregressively reconstructing the semantic representation of images consistently improves comprehension. Notably, we find that even when models are given continuous image features as input, they can effectively reconstruct discrete semantic tokens, resulting in stable and consistent improvements across a wide range of multimodal understanding benchmarks. Our approach delivers significant performance gains across varying data scales (556k-2M) and types of LLM bacbones. Specifically, ASVR improves LLaVA-1.5 by 5% in average scores across 14 multimodal benchmarks. The code is available at this https URL. 

**Abstract (ZH)**: 典型的大型视觉-语言模型（LVLMs）仅对文本序列应用自回归监督，而未能充分将视觉模态整合到学习过程中。这导致了三个关键局限性：（1）无法利用未配对字幕的图像，（2）字幕可能遗漏关键的视觉细节，以及（3）某些视觉中心的内容难以通过文本充分表达。因此，当前的LVLMs往往在图像到语言对齐方面占优，但可能忽视了精细的视觉信息。虽然一些先前的工作探索了自回归图像生成，但如何有效利用自回归的视觉监督来增强图像理解仍然是一个开放的挑战。在本文中，我们引入了自回归语义视觉重建（ASVR），使其能够在统一的自回归框架中联合学习视觉和文本模态。我们表明，自回归重建图像的原始视觉外观并未提升多模态理解，甚至可能损害多模态理解。相反，自回归重建图像的语义表示始终能提高理解能力。值得注意的是，即使给模型提供连续的图像特征作为输入，它们也能有效地重建离散的语义令牌，从而在多种多模态理解基准测试中实现稳定且一致的改进。我们的方法在不同数据规模（556k-2M）和不同类型的大规模语言模型（LLM）底座上实现了显著的性能提升。具体而言，ASVR使LLaVA-1.5在14个不同多模态基准测试中的平均得分提高了5%。代码可在以下链接获得。 

---
# FZOO: Fast Zeroth-Order Optimizer for Fine-Tuning Large Language Models towards Adam-Scale Speed 

**Title (ZH)**: FZOO: 快速零阶优化器，用于将大型语言模型微调至Adam级别速度 

**Authors**: Sizhe Dang, Yangyang Guo, Yanjun Zhao, Haishan Ye, Xiaodong Zheng, Guang Dai, Ivor Tsang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09034)  

**Abstract**: Fine-tuning large language models (LLMs) often faces GPU memory bottlenecks: the backward pass of first-order optimizers like Adam increases memory usage to more than 10 times the inference level (e.g., 633 GB for OPT-30B). Zeroth-order (ZO) optimizers avoid this cost by estimating gradients only from forward passes, yet existing methods like MeZO usually require many more steps to converge. Can this trade-off between speed and memory in ZO be fundamentally improved? Normalized-SGD demonstrates strong empirical performance with greater memory efficiency than Adam. In light of this, we introduce FZOO, a Fast Zeroth-Order Optimizer toward Adam-Scale Speed. FZOO reduces the total forward passes needed for convergence by employing batched one-sided estimates that adapt step sizes based on the standard deviation of batch losses. It also accelerates per-batch computation through the use of Rademacher random vector perturbations coupled with CUDA's parallel processing. Extensive experiments on diverse models, including RoBERTa-large, OPT (350M-66B), Phi-2, and Llama3, across 11 tasks validate FZOO's effectiveness. On average, FZOO outperforms MeZO by 3 percent in accuracy while requiring 3 times fewer forward passes. For RoBERTa-large, FZOO achieves average improvements of 5.6 percent in accuracy and an 18 times reduction in forward passes compared to MeZO, achieving convergence speeds comparable to Adam. We also provide theoretical analysis proving FZOO's formal equivalence to a normalized-SGD update rule and its convergence guarantees. FZOO integrates smoothly into PEFT techniques, enabling even larger memory savings. Overall, our results make single-GPU, high-speed, full-parameter fine-tuning practical and point toward future work on memory-efficient pre-training. 

**Abstract (ZH)**: Fine-tuning 大型语言模型 (LLMs) 经常面临 GPU 内存瓶颈：像 Adam 这样的一阶优化器的后向传播会将内存使用量增加到推理水平的 10 倍以上（例如，OPT-30B 的情况为 633 GB）。零阶（ZO）优化器通过仅从前向传播中估计梯度来避免这种成本，但现有的方法如 MeZO 通常需要更多步骤才能收敛。ZO 的这种速度与内存之间的权衡能否从根本上得到改进？实证结果显示，规范化-SGD 在内存效率方面优于 Adam，且具有强大的实证表现。鉴于此，我们提出了一种名为 FZOO 的快速零阶优化器，其目标是在 Adam 水平的速度下运行。FZOO 通过利用批处理单向估计并根据批次损失的标准差自适应调整步长，减少了达到收敛所需的总前向传播次数。此外，FZOO 还通过结合 Rademacher 随机向量扰动和 CUDA 并行处理加速了批次内计算。我们在包括 RoBERTa-large、OPT（350M-66B）、Phi-2 和 Llama3 等多种模型以及 11 项任务上进行了广泛的实验，验证了 FZOO 的有效性。平均而言，FZOO 在准确性方面优于 MeZO 3%，需要少 3 倍的前向传播次数。对于 RoBERTa-large，与 MeZO 相比，FZOO 在准确性上平均提高了 5.6%，前向传播次数减少了 18 倍，并实现了与 Adam 相媲美的收敛速度。我们还提供了理论分析，证明了 FZOO 与规范化-SGD 更新规则的形式等价及其收敛保证。FZOO 能够平滑地集成到 PEFT 技术中，从而实现更大的内存节省。总体而言，我们的结果使得单 GPU 高速全参数微调成为可能，并指出了未来内存高效预训练的工作方向。 

---
# Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning 

**Title (ZH)**: Router-R1: 通过强化学习教学大规模语言模型多轮路由和聚合 

**Authors**: Haozhen Zhang, Tao Feng, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2506.09033)  

**Abstract**: The rapid emergence of diverse large language models (LLMs) has spurred the development of LLM routers that assign user queries to the most suitable model. However, existing LLM routers typically perform a single-round, one-to-one mapping (\textit{i.e.}, assigning each query to a single model in isolation), which limits their capability to tackle complex tasks that demand the complementary strengths of multiple LLMs. In this paper, we present \textbf{Router-R1}, a reinforcement learning (RL)-based framework that formulates multi-LLM routing and aggregation as a sequential decision process. Router-R1 instantiates the router itself as a capable LLM, leveraging its reasoning ability to interleave "think" actions (internal deliberation) with "route" actions (dynamic model invocation), and integrates each response into its evolving context. To guide learning, we employ a lightweight rule-based reward comprising format rewards, final outcome rewards, and a novel cost reward for performance and cost trade-off optimization, opening a pathway toward optimizing performance-cost tradeoffs via RL. Router-R1 also conditions only on simple model descriptors such as pricing, latency, and example performance, enabling strong generalization to unseen model selection. Experiments on seven general and multi-hop QA benchmarks show that Router-R1 outperforms over several strong baselines, achieving superior performance while maintaining robust generalization and cost this http URL is available at this https URL. 

**Abstract (ZH)**: 基于强化学习的多大型语言模型路由框架Router-R1 

---
# Diffuse and Disperse: Image Generation with Representation Regularization 

**Title (ZH)**: 扩散与分散：基于表示正则化的图像生成 

**Authors**: Runqian Wang, Kaiming He  

**Link**: [PDF](https://arxiv.org/pdf/2506.09027)  

**Abstract**: The development of diffusion-based generative models over the past decade has largely proceeded independently of progress in representation learning. These diffusion models typically rely on regression-based objectives and generally lack explicit regularization. In this work, we propose \textit{Dispersive Loss}, a simple plug-and-play regularizer that effectively improves diffusion-based generative models. Our loss function encourages internal representations to disperse in the hidden space, analogous to contrastive self-supervised learning, with the key distinction that it requires no positive sample pairs and therefore does not interfere with the sampling process used for regression. Compared to the recent method of representation alignment (REPA), our approach is self-contained and minimalist, requiring no pre-training, no additional parameters, and no external data. We evaluate Dispersive Loss on the ImageNet dataset across a range of models and report consistent improvements over widely used and strong baselines. We hope our work will help bridge the gap between generative modeling and representation learning. 

**Abstract (ZH)**: 过去十年基于扩散的生成模型的发展很大程度上与表示学习的进步独立进行。这些扩散模型通常依赖于基于回归的目标函数，并且通常缺乏明确的正则化。在本文中，我们提出了一种简单直观的正则化方法——分散损失（Dispersive Loss），它可以有效提升基于扩散的生成模型。我们的损失函数鼓励内部表示在隐空间中分散，类似于对比自监督学习，但关键区别在于它不需要正样本对，因此不会干扰用于回归的采样过程。与最近的表示对齐方法（REPA）相比，我们的方法是自包含且简约的，无需预训练、额外参数和外部数据。我们在ImageNet数据集上对多种模型应用分散损失，并报告了相对于广泛使用的强基线模型的一致改进。我们希望我们的工作能帮助弥合生成建模与表示学习之间的差距。 

---
# Edit Flows: Flow Matching with Edit Operations 

**Title (ZH)**: 编辑流：基于编辑操作的流匹配 

**Authors**: Marton Havasi, Brian Karrer, Itai Gat, Ricky T. Q. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09018)  

**Abstract**: Autoregressive generative models naturally generate variable-length sequences, while non-autoregressive models struggle, often imposing rigid, token-wise structures. We propose Edit Flows, a non-autoregressive model that overcomes these limitations by defining a discrete flow over sequences through edit operations-insertions, deletions, and substitutions. By modeling these operations within a Continuous-time Markov Chain over the sequence space, Edit Flows enable flexible, position-relative generation that aligns more closely with the structure of sequence data. Our training method leverages an expanded state space with auxiliary variables, making the learning process efficient and tractable. Empirical results show that Edit Flows outperforms both autoregressive and mask models on image captioning and significantly outperforms the mask construction in text and code generation. 

**Abstract (ZH)**: 非自回归生成模型通过编辑操作克服长度限制，实现灵活的序列生成 

---
# Employing self-supervised learning models for cross-linguistic child speech maturity classification 

**Title (ZH)**: 利用自监督学习模型进行跨语言儿童言语成熟度分类 

**Authors**: Theo Zhang, Madurya Suresh, Anne S. Warlaumont, Kasia Hitczenko, Alejandrina Cristia, Margaret Cychosz  

**Link**: [PDF](https://arxiv.org/pdf/2506.08999)  

**Abstract**: Speech technology systems struggle with many downstream tasks for child speech due to small training corpora and the difficulties that child speech pose. We apply a novel dataset, SpeechMaturity, to state-of-the-art transformer models to address a fundamental classification task: identifying child vocalizations. Unlike previous corpora, our dataset captures maximally ecologically-valid child vocalizations across an unprecedented sample, comprising children acquiring 25+ languages in the U.S., Bolivia, Vanuatu, Papua New Guinea, Solomon Islands, and France. The dataset contains 242,004 labeled vocalizations, magnitudes larger than previous work. Models were trained to distinguish between cry, laughter, mature (consonant+vowel), and immature speech (just consonant or vowel). Models trained on the dataset outperform state-of-the-art models trained on previous datasets, achieved classification accuracy comparable to humans, and were robust across rural and urban settings. 

**Abstract (ZH)**: 由于儿童语音的小规模训练语料库和其带来的挑战，语音技术系统在处理儿童语音的下游任务中表现不佳。我们应用一种新颖的数据集SpeechMaturity到最先进的变压器模型上，以解决一项基础分类任务：识别儿童语音。与之前的语料库不同，我们的数据集捕捉了前所未有的样本中最大化生态有效的儿童语音，包括在美国、玻利维亚、瓦努阿图、巴布亚新几内亚、所罗门群岛和法国学习25种以上语言的儿童。该数据集包含242,004个标注的语音样本，规模远超以往工作。模型被训练以区分哭声、笑声、成熟的（辅音+元音）语音和不成熟的（仅辅音或元音）语音。在该数据集上训练的模型超越了在先前数据集上训练的最先进的模型，并且分类准确度与人类相当，且具有跨农村和城市环境的稳定性。 

---
# Efficient Medical Vision-Language Alignment Through Adapting Masked Vision Models 

**Title (ZH)**: 通过适应掩蔽视觉模型实现高效的医学视图-语言对齐 

**Authors**: Chenyu Lian, Hong-Yu Zhou, Dongyun Liang, Jing Qin, Liansheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08990)  

**Abstract**: Medical vision-language alignment through cross-modal contrastive learning shows promising performance in image-text matching tasks, such as retrieval and zero-shot classification. However, conventional cross-modal contrastive learning (CLIP-based) methods suffer from suboptimal visual representation capabilities, which also limits their effectiveness in vision-language alignment. In contrast, although the models pretrained via multimodal masked modeling struggle with direct cross-modal matching, they excel in visual representation. To address this contradiction, we propose ALTA (ALign Through Adapting), an efficient medical vision-language alignment method that utilizes only about 8% of the trainable parameters and less than 1/5 of the computational consumption required for masked record modeling. ALTA achieves superior performance in vision-language matching tasks like retrieval and zero-shot classification by adapting the pretrained vision model from masked record modeling. Additionally, we integrate temporal-multiview radiograph inputs to enhance the information consistency between radiographs and their corresponding descriptions in reports, further improving the vision-language alignment. Experimental evaluations show that ALTA outperforms the best-performing counterpart by over 4% absolute points in text-to-image accuracy and approximately 6% absolute points in image-to-text retrieval accuracy. The adaptation of vision-language models during efficient alignment also promotes better vision and language understanding. Code is publicly available at this https URL. 

**Abstract (ZH)**: 医学视觉-语言对齐通过跨模态对比学习在图像-文本匹配任务中表现出色，但传统跨模态对比学习（如CLIP）方法在视觉表示能力上有不足，限制了其在视觉-语言对齐中的效果。相比之下，虽然通过多模态掩码建模预训练的模型在直接跨模态匹配上存在问题，但在视觉表示上表现出色。为了解决这一矛盾，我们提出了一种高效的医学视觉-语言对齐方法ALTA（通过适应对齐），该方法仅使用约8%的可训练参数和少于掩码记录建模所需计算量的1/5。ALTA通过适应从掩码记录建模预训练的视觉模型，在视觉-语言匹配任务如检索和零样本分类中取得了优异性能。此外，我们整合了时间多视图X线输入，以增强医学报告中的X线图像与其描述之间的一致性，进一步提高视觉-语言对齐。实验评估显示，ALTA在文本到图像准确性和图像到文本检索准确率上分别超过当前最优方法4%和约6%的绝对值。通过有效对齐促进视觉和语言理解。源代码已公开。 

---
# Propositional Logic for Probing Generalization in Neural Networks 

**Title (ZH)**: 命题逻辑在探测试神经网络泛化能力中的应用 

**Authors**: Anna Langedijk, Jaap Jumelet, Willem Zuidema  

**Link**: [PDF](https://arxiv.org/pdf/2506.08978)  

**Abstract**: The extent to which neural networks are able to acquire and represent symbolic rules remains a key topic of research and debate. Much current work focuses on the impressive capabilities of large language models, as well as their often ill-understood failures on a wide range of reasoning tasks. In this paper, in contrast, we investigate the generalization behavior of three key neural architectures (Transformers, Graph Convolution Networks and LSTMs) in a controlled task rooted in propositional logic. The task requires models to generate satisfying assignments for logical formulas, making it a structured and interpretable setting for studying compositionality. We introduce a balanced extension of an existing dataset to eliminate superficial patterns and enable testing on unseen operator combinations. Using this dataset, we evaluate the ability of the three architectures to generalize beyond the training distribution. While all models perform well in-distribution, we find that generalization to unseen patterns, particularly those involving negation, remains a significant challenge. Transformers fail to apply negation compositionally, unless structural biases are introduced. Our findings highlight persistent limitations in the ability of standard architectures to learn systematic representations of logical operators, suggesting the need for stronger inductive biases to support robust rule-based reasoning. 

**Abstract (ZH)**: 神经网络获取和表示符号规则的能力：三种关键神经架构在命题逻辑任务中的泛化行为 

---
# Tailored Architectures for Time Series Forecasting: Evaluating Deep Learning Models on Gaussian Process-Generated Data 

**Title (ZH)**: 为时间序列预测定制的架构：基于高斯过程生成数据的深度学习模型评估 

**Authors**: Victoria Hankemeier, Malte Schilling  

**Link**: [PDF](https://arxiv.org/pdf/2506.08977)  

**Abstract**: Developments in Deep Learning have significantly improved time series forecasting by enabling more accurate modeling of complex temporal dependencies inherent in sequential data. The effectiveness of such models is often demonstrated on limited sets of specific real-world data. Although this allows for comparative analysis, it still does not demonstrate how specific data characteristics align with the architectural strengths of individual models. Our research aims at uncovering clear connections between time series characteristics and particular models. We introduce a novel dataset generated using Gaussian Processes, specifically designed to display distinct, known characteristics for targeted evaluations of model adaptability to them. Furthermore, we present TimeFlex, a new model that incorporates a modular architecture tailored to handle diverse temporal dynamics, including trends and periodic patterns. This model is compared to current state-of-the-art models, offering a deeper understanding of how models perform under varied time series conditions. 

**Abstract (ZH)**: 深度学习的发展显著改善了时间序列 forecasting，使其能够更准确地建模序列数据中固有的复杂时间依赖关系。虽然这些模型的有效性通常通过有限的具体现实数据集进行展示，但仍不足以说明特定数据特征如何与个体模型的架构优势相匹配。我们的研究旨在揭示时间序列特征与特定模型之间清晰的联系。我们引入了一个使用高斯过程生成的新数据集，专门设计用于展示不同的已知特征，以便有针对性地评估模型对这些特征的适应性。此外，我们提出了TimeFlex模型，该模型具有模块化的架构，能够处理多种时间动态，包括趋势和周期模式。我们将该模型与当前最先进的模型进行比较，提供了对模型在各种时间序列条件下的表现的更深入理解。 

---
# GFRIEND: Generative Few-shot Reward Inference through EfficieNt DPO 

**Title (ZH)**: GFRIEND: 生成少量样本奖励推断通过高效DPO 

**Authors**: Yiyang Zhao, Huiyu Bai, Xuejiao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.08965)  

**Abstract**: The ability to train high-performing reward models with few-shot data is critical for enhancing the efficiency and scalability of Reinforcement Learning from Human Feedback (RLHF). We propose a data augmentation and expansion framework that enables generative reward models trained on small datasets to achieve comparable performance to those trained on large-scale datasets. Traditional methods to train a generative reward model, such as Direct Preference Optimization (DPO), are constrained by inefficiencies in sample pairing and limited data diversity. This work introduces preference refinement, which employs Chain-of-Thought (CoT) sampling to uncover diverse and high-quality preference relationships. It also incorporates a perplexity-based scoring mechanism to assign nuanced preference levels and utilizes Multi-level Direct Preference Optimization (M-DPO) to enable the model to capture finer-grained preference differences between samples. Experimental results demonstrate that the proposed method significantly enhances data efficiency and model performance, enabling reward models trained in a few-shot setting to achieve results on par with those trained on large-scale datasets. This study underscores the potential of data-efficient strategies in advancing reward model optimization, offering a robust solution for low-resource RLHF applications. 

**Abstract (ZH)**: 少量样本数据下训练高性能奖励模型的能力对于增强人类反馈强化学习（RLHF）的效率和可扩展性至关重要。我们提出了一种数据增强和扩展框架，该框架使在小数据集上训练的生成奖励模型能够达到在大规模数据集上训练的模型的相似性能。传统的生成奖励模型的训练方法，如直接偏好优化（DPO），受限于样本配对的低效率和数据多样性有限。本研究引入了偏好细化方法，该方法采用Chain-of-Thought（CoT）采样来揭示多样且高质量的偏好关系，并引入了一个基于困惑度的评分机制来分配细腻的偏好等级，同时利用多层次直接偏好优化（M-DPO）使模型能够捕获样本之间的细微偏好差异。实验结果表明，所提出的方法显著提高了数据效率和模型性能，使得在少量样本设置下训练的奖励模型能达到在大规模数据集上训练的模型的同等效果。本研究强调了高效数据策略在奖励模型优化中的潜力，提供了低资源RLHF应用的一个稳健解决方案。 

---
# WIP: Large Language Model-Enhanced Smart Tutor for Undergraduate Circuit Analysis 

**Title (ZH)**: WIP: 增强型大型语言模型辅助智能导师在本科电路分析中的应用 

**Authors**: Liangliang Chen, Huiru Xie, Jacqueline Rohde, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08962)  

**Abstract**: This research-to-practice work-in-progress (WIP) paper presents an AI-enabled smart tutor designed to provide homework assessment and feedback for students in an undergraduate circuit analysis course. We detail the tutor's design philosophy and core components, including open-ended question answering and homework feedback generation. The prompts are carefully crafted to optimize responses across different problems. The smart tutor was deployed on the Microsoft Azure platform and is currently in use in an undergraduate circuit analysis course at the School of Electrical and Computer Engineering in a large, public, research-intensive institution in the Southeastern United States. Beyond offering personalized instruction and feedback, the tutor collects student interaction data, which is summarized and shared with the course instructor. To evaluate its effectiveness, we collected student feedback, with 90.9% of responses indicating satisfaction with the tutor. Additionally, we analyze a subset of collected data on preliminary circuit analysis topics to assess tutor usage frequency for each problem and identify frequently asked questions. These insights help instructors gain real-time awareness of student difficulties, enabling more targeted classroom instruction. In future work, we will release a full analysis once the complete dataset is available after the Spring 2025 semester. We also explore the potential applications of this smart tutor across a broader range of engineering disciplines by developing improved prompts, diagram-recognition methods, and database management strategies, which remain ongoing areas of research. 

**Abstract (ZH)**: 这项研究至实践工作进展（WIP）论文介绍了一种基于人工智能的智能辅导系统，旨在为美国东南部一所大型公立研究密集型机构电气与计算机工程学院本科生电路分析课程的学生提供家庭作业评估和反馈。我们详细阐述了该辅导系统的设计理念和核心组件，包括开放性问题回答和家庭作业反馈生成。精心设计的提示旨在优化不同问题的回应。该智能辅导系统部署在Microsoft Azure平台上，并正在该学院的本科生电路分析课程中使用。除了提供个性化指导和反馈外，该辅导系统还会收集学生互动数据，并将其总结后与课程教师分享。为评估其有效性，我们收集了学生反馈，其中90.9%的回应表示对辅导系统的满意。此外，我们还分析了部分收集的数据，针对初步电路分析主题，评估每个问题的使用频率，并确定常见问题。这些见解帮助教师实时了解学生的困难，从而能够进行更有针对性的课堂指导。未来研究中，我们将发布完整的数据分析，前提是2025年春季学期结束后可获得完整数据集。我们也探索了该智能辅导系统在更广泛工程学科中的潜在应用，通过开发改进的提示、图像识别方法和数据库管理策略，以实现这一目标，这些研究目前仍在进行中。 

---
# Towards Robust Deep Reinforcement Learning against Environmental State Perturbation 

**Title (ZH)**: 面向环境状态扰动的稳健深度强化学习 

**Authors**: Chenxu Wang, Huaping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08961)  

**Abstract**: Adversarial attacks and robustness in Deep Reinforcement Learning (DRL) have been widely studied in various threat models; however, few consider environmental state perturbations, which are natural in embodied scenarios. To improve the robustness of DRL agents, we formulate the problem of environmental state perturbation, introducing a preliminary non-targeted attack method as a calibration adversary, and then propose a defense framework, named Boosted Adversarial Training (BAT), which first tunes the agents via supervised learning to avoid catastrophic failure and subsequently adversarially trains the agent with reinforcement learning. Extensive experimental results substantiate the vulnerability of mainstream agents under environmental state perturbations and the effectiveness of our proposed attack. The defense results demonstrate that while existing robust reinforcement learning algorithms may not be suitable, our BAT framework can significantly enhance the robustness of agents against environmental state perturbations across various situations. 

**Abstract (ZH)**: 深度强化学习中的对抗攻击与鲁棒性在各种威胁模型中的研究很少考虑环境状态扰动，而在实际 bodied 场景中这是自然存在的。为了提高 DRL 代理的鲁棒性，我们制定了环境状态扰动的问题，引入了一种初步的非目标攻击方法作为校准对手，并提出了一种防御框架，名为增强对抗训练（BAT），该框架首先通过监督学习调校代理以避免灾难性失败，随后使用强化学习对代理进行对抗训练。广泛的实验结果证实了主流代理在环境状态扰动下的脆弱性以及我们提出的攻击的有效性。防御结果表明，虽然现有的鲁棒强化学习算法可能并不适用，但我们的 BAT 框架可以在各种情况下显著增强代理对环境状态扰动的鲁棒性。 

---
# Segment Concealed Objects with Incomplete Supervision 

**Title (ZH)**: 用不完备监督揭示隐藏对象 

**Authors**: Chunming He, Kai Li, Yachao Zhang, Ziyun Yang, Youwei Pang, Longxiang Tang, Chengyu Fang, Yulun Zhang, Linghe Kong, Xiu Li, Sina Farsiu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08955)  

**Abstract**: Incompletely-Supervised Concealed Object Segmentation (ISCOS) involves segmenting objects that seamlessly blend into their surrounding environments, utilizing incompletely annotated data, such as weak and semi-annotations, for model training. This task remains highly challenging due to (1) the limited supervision provided by the incompletely annotated training data, and (2) the difficulty of distinguishing concealed objects from the background, which arises from the intrinsic similarities in concealed scenarios. In this paper, we introduce the first unified method for ISCOS to address these challenges. To tackle the issue of incomplete supervision, we propose a unified mean-teacher framework, SEE, that leverages the vision foundation model, ``\emph{Segment Anything Model (SAM)}'', to generate pseudo-labels using coarse masks produced by the teacher model as prompts. To mitigate the effect of low-quality segmentation masks, we introduce a series of strategies for pseudo-label generation, storage, and supervision. These strategies aim to produce informative pseudo-labels, store the best pseudo-labels generated, and select the most reliable components to guide the student model, thereby ensuring robust network training. Additionally, to tackle the issue of intrinsic similarity, we design a hybrid-granularity feature grouping module that groups features at different granularities and aggregates these results. By clustering similar features, this module promotes segmentation coherence, facilitating more complete segmentation for both single-object and multiple-object images. We validate the effectiveness of our approach across multiple ISCOS tasks, and experimental results demonstrate that our method achieves state-of-the-art performance. Furthermore, SEE can serve as a plug-and-play solution, enhancing the performance of existing models. 

**Abstract (ZH)**: 不完全监督隐藏对象分割（ISCOS）涉及利用不完全注释数据（如弱注释和半注释）对融合其环境中的对象进行分割，该任务由于（1）不完全注释训练数据提供的有限监督，以及（2）难以区分隐藏对象与背景（这是由于隐藏场景中固有的相似性）而具有高度挑战性。本文介绍了首个统一方法以应对这些挑战。为解决不完全监督的问题，我们提出了一种结合教师模型的统一教师框架SEE，利用“Segment Anything Model (SAM)”视觉基础模型生成伪标签，使用教师模型生成的粗略掩码作为提示。为减轻低质量分割掩码的影响，我们引入了一系列伪标签生成、存储和监督策略，旨在生成具有信息性的伪标签、存储最佳伪标签，并选择最可靠的组件来引导学生模型，从而确保网络训练的鲁棒性。此外，为应对固有相似性问题，我们设计了一个混合粒度特征分组模块，该模块在不同粒度下分组特征并聚合这些结果。通过聚类相似特征，该模块促进了分割的一致性，有助于对单对象和多对象图像进行更完整的分割。我们在多个ISCOS任务上验证了该方法的有效性，并且实验结果表明，我们的方法达到了最新性能。此外，SEE可以作为一种即插即用的解决方案，提升现有模型的性能。 

---
# Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions 

**Title (ZH)**: Can LLMs 实现 grounding 时（不）知道：对直接和负载型政治问题的研究 

**Authors**: Clara Lachenmaier, Judith Sieker, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2506.08952)  

**Abstract**: Communication among humans relies on conversational grounding, allowing interlocutors to reach mutual understanding even when they do not have perfect knowledge and must resolve discrepancies in each other's beliefs. This paper investigates how large language models (LLMs) manage common ground in cases where they (don't) possess knowledge, focusing on facts in the political domain where the risk of misinformation and grounding failure is high. We examine the ability of LLMs to answer direct knowledge questions and loaded questions that presuppose misinformation. We evaluate whether loaded questions lead LLMs to engage in active grounding and correct false user beliefs, in connection to their level of knowledge and their political bias. Our findings highlight significant challenges in LLMs' ability to engage in grounding and reject false user beliefs, raising concerns about their role in mitigating misinformation in political discourse. 

**Abstract (ZH)**: 人类交流依赖会话接地，即使对话双方不具备完善的知识并需要解决彼此信念中的分歧，也能达成相互理解。本文探讨大语言模型（LLMs）在拥有（或不拥有）特定知识时如何管理共同知识，重点关注政治领域，该领域存在高风险的错误信息和接地失败。我们考察LLMs回答直接知识问题和预设错误信息的负荷问题的能力。我们评估负荷问题是否促使LLMs积极接地并纠正用户的错误信念，这与它们的知识水平和政治偏见有关。我们的研究结果突显了LLMs在参与接地和拒绝错误用户信念方面面临的重大挑战，这引发了对其在政治话语中遏制错误信息角色的担忧。 

---
# Can A Gamer Train A Mathematical Reasoning Model? 

**Title (ZH)**: 游戏者能否训练数学推理模型？ 

**Authors**: Andrew Shin  

**Link**: [PDF](https://arxiv.org/pdf/2506.08935)  

**Abstract**: While large language models (LLMs) have achieved remarkable performance in various tasks including mathematical reasoning, their development typically demands prohibitive computational resources. Recent advancements have reduced costs for training capable models, yet even these approaches rely on high-end hardware clusters. In this paper, we demonstrate that a single average gaming GPU can train a solid mathematical reasoning model, by integrating reinforcement learning and memory optimization techniques. Specifically, we train a 1.5B parameter mathematical reasoning model on RTX 3080 Ti of 16GB memory that achieves comparable or better performance on mathematical reasoning benchmarks than models several times larger, in resource-constrained environments. Our results challenge the paradigm that state-of-the-art mathematical reasoning necessitates massive infrastructure, democratizing access to high-performance AI research. this https URL. 

**Abstract (ZH)**: 大型语言模型在数学推理等任务上取得了显著performance，但其开发通常需要难以承受的计算资源。近期进展虽降低了训练能力强模型的成本，但这些方法仍依赖高端硬件集群。在本文中，我们通过结合强化学习和内存优化技术，展示了单个普通游戏GPU可以在资源受限环境中训练出与更大模型表现相当或更好的数学推理模型。我们的结果挑战了顶尖数学推理需要庞大基础设施的范式，促进了高性能AI研究的平民化访问。这个 https://。 

---
# Socratic-MCTS: Test-Time Visual Reasoning by Asking the Right Questions 

**Title (ZH)**: 苏格拉底-MCTS: 在线视觉推理通过提出正确的问题 

**Authors**: David Acuna, Ximing Lu, Jaehun Jung, Hyunwoo Kim, Amlan Kar, Sanja Fidler, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08927)  

**Abstract**: Recent research in vision-language models (VLMs) has centered around the possibility of equipping them with implicit long-form chain-of-thought reasoning -- akin to the success observed in language models -- via distillation and reinforcement learning. But what about the non-reasoning models already trained and deployed across the internet? Should we simply abandon them, or is there hope for a search mechanism that can elicit hidden knowledge and induce long reasoning traces -- without any additional training or supervision? In this paper, we explore this possibility using a Monte Carlo Tree Search (MCTS)-inspired algorithm, which injects subquestion-subanswer pairs into the model's output stream. We show that framing reasoning as a search process -- where subquestions act as latent decisions within a broader inference trajectory -- helps the model "connect the dots" between fragmented knowledge and produce extended reasoning traces in non-reasoning models. We evaluate our method across three benchmarks and observe consistent improvements. Notably, our approach yields a 2% overall improvement on MMMU-PRO, including a significant 9% gain in Liberal Arts. 

**Abstract (ZH)**: 近期视觉-语言模型的研究集中于通过蒸馏和强化学习赋予它们潜在的长链推理能力，类似于语言模型的成功。但互联网上已训练和部署的非推理模型呢？我们是否应该完全放弃它们，还是有可能找到一种检索机制，能够在无需额外训练或监督的情况下唤起隐藏知识并诱导长推理痕迹？本文采用受蒙特卡洛树搜索(MCTS)启发的算法，在模型的输出流中注入子问题-子答对，将推理视为一个搜索过程，其中子问题作为广泛推理轨迹中的潜在决策，帮助模型连接碎片化的知识并生成非推理模型中的扩展推理痕迹。我们通过三个基准测试评估该方法，并观察到一致的改进。值得注意的是，我们的方法在MMMU-PRO上总体提高了2%，在人文学科方面取得了显著的9%的提升。 

---
# PropMEND: Hypernetworks for Knowledge Propagation in LLMs 

**Title (ZH)**: PropMEND：用于大语言模型知识传播的超网络 

**Authors**: Zeyu Leo Liu, Greg Durrett, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08920)  

**Abstract**: Knowledge editing techniques for large language models (LLMs) can inject knowledge that is later reproducible verbatim, but they fall short on propagating that knowledge: models cannot answer questions that require reasoning with the injected knowledge. We present a hypernetwork-based approach for knowledge propagation, named PropMEND, where we meta-learn how to modify gradients of a language modeling loss to encourage injected information to propagate. Our approach extends the meta-objective of MEND [29] so that gradient updates on knowledge are transformed to enable answering multi-hop questions involving that knowledge. We show improved performance on the RippleEdit dataset, showing almost 2x accuracy on challenging multi-hop questions whose answers are not explicitly stated in the injected fact. We further introduce a new dataset, Controlled RippleEdit, to evaluate the generalization of our hypernetwork, testing knowledge propagation along relations and entities unseen during hypernetwork training. PropMEND still outperforms existing approaches in unseen entity-relation pairs, yet the performance gap decreases substantially, suggesting future work in propagating knowledge to a wide range of relations. 

**Abstract (ZH)**: 基于超网络的知识传播技术可以将知识注入大型语言模型并在需要时原样重现，但它们在传播知识方面存在局限：模型无法回答需要利用注入知识进行推理的问题。我们提出了一种基于超网络的知识传播方法，名为PropMEND，通过元学习修改语言建模损失的梯度，以促进注入信息的传播。我们的方法扩展了MEND的元目标，使得梯度更新能够转换为支持回答涉及注入知识的多跳问题。我们在RippleEdit数据集上展示了改进的性能，对于答案并未明确陈述在注入事实中的挑战性多跳问题，正确率几乎提高了一倍。我们还引入了一个新的数据集，受控RippleEdit，以评估我们超网络的泛化能力，测试在未见关系和实体下知识的传播。尽管PropMEND在未见实体-关系对上的表现仍然优于现有方法，但性能差距显著减小，这提示未来在更广泛关系下传播知识的工作。 

---
# Quantum Adiabatic Generation of Human-Like Passwords 

**Title (ZH)**: 量子渐近生成类人类密码 

**Authors**: Sascha Mücke, Raoul Heese, Thore Gerlach, David Biesner, Loong Kuan Lee, Nico Piatkowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.08917)  

**Abstract**: Generative Artificial Intelligence (GenAI) for Natural Language Processing (NLP) is the predominant AI technology to date. An important perspective for Quantum Computing (QC) is the question whether QC has the potential to reduce the vast resource requirements for training and operating GenAI models. While large-scale generative NLP tasks are currently out of reach for practical quantum computers, the generation of short semantic structures such as passwords is not. Generating passwords that mimic real user behavior has many applications, for example to test an authentication system against realistic threat models. Classical password generation via deep learning have recently been investigated with significant progress in their ability to generate novel, realistic password candidates. In the present work we investigate the utility of adiabatic quantum computers for this task. More precisely, we study different encodings of token strings and propose novel approaches based on the Quadratic Unconstrained Binary Optimization (QUBO) and the Unit-Disk Maximum Independent Set (UD-MIS) problems. Our approach allows us to estimate the token distribution from data and adiabatically prepare a quantum state from which we eventually sample the generated passwords via measurements. Our results show that relatively small samples of 128 passwords, generated on the QuEra Aquila 256-qubit neutral atom quantum computer, contain human-like passwords such as "Tunas200992" or "teedem28iglove". 

**Abstract (ZH)**: 生成式人工智能（GenAI）在自然语言处理（NLP）中的应用是目前主导的AI技术。量子计算（QC）的一个重要视角是探讨QC是否有潜力减少训练和运行GenAI模型所需的大量资源。虽然大规模生成式NLP任务目前超出了实用量子计算机的能力范围，但生成类似密码这样的短语义结构是可行的。模拟真实用户行为生成密码有许多应用，例如测试认证系统以抵御现实威胁模型。通过深度学习的经典密码生成方法最近有了显著进步，能够在生成新颖、真实的密码候选方面取得进展。本研究探讨了使用费米абatic量子计算机执行此任务的适用性。具体而言，我们研究了不同类型的词元字符串编码，并提出了基于二次无约束二元优化（QUBO）和单位圆盘最大独立集（UD-MIS）问题的新颖方法。我们的方法允许我们从数据中估算词元分布，并通过量子态的测量从中抽样生成的密码。实验结果显示，使用QuEra Aquila 256量子比特中性原子量子计算机生成的128个密码样本中包含了类似人类行为的密码，如“Tunas200992”或“teedem28iglove”。 

---
# Inherently Faithful Attention Maps for Vision Transformers 

**Title (ZH)**: 内在忠实注意力图谱：用于视觉Transformer 

**Authors**: Ananthu Aniraj, Cassio F. Dantas, Dino Ienco, Diego Marcos  

**Link**: [PDF](https://arxiv.org/pdf/2506.08915)  

**Abstract**: We introduce an attention-based method that uses learned binary attention masks to ensure that only attended image regions influence the prediction. Context can strongly affect object perception, sometimes leading to biased representations, particularly when objects appear in out-of-distribution backgrounds. At the same time, many image-level object-centric tasks require identifying relevant regions, often requiring context. To address this conundrum, we propose a two-stage framework: stage 1 processes the full image to discover object parts and identify task-relevant regions, while stage 2 leverages input attention masking to restrict its receptive field to these regions, enabling a focused analysis while filtering out potentially spurious information. Both stages are trained jointly, allowing stage 2 to refine stage 1. Extensive experiments across diverse benchmarks demonstrate that our approach significantly improves robustness against spurious correlations and out-of-distribution backgrounds. 

**Abstract (ZH)**: 基于注意力的方法：使用学习的二元注意力掩模确保仅关注的图像区域影响预测 

---
# Intention-Conditioned Flow Occupancy Models 

**Title (ZH)**: 意图条件流占用模型 

**Authors**: Chongyi Zheng, Seohong Park, Sergey Levine, Benjamin Eysenbach  

**Link**: [PDF](https://arxiv.org/pdf/2506.08902)  

**Abstract**: Large-scale pre-training has fundamentally changed how machine learning research is done today: large foundation models are trained once, and then can be used by anyone in the community (including those without data or compute resources to train a model from scratch) to adapt and fine-tune to specific tasks. Applying this same framework to reinforcement learning (RL) is appealing because it offers compelling avenues for addressing core challenges in RL, including sample efficiency and robustness. However, there remains a fundamental challenge to pre-train large models in the context of RL: actions have long-term dependencies, so training a foundation model that reasons across time is important. Recent advances in generative AI have provided new tools for modeling highly complex distributions. In this paper, we build a probabilistic model to predict which states an agent will visit in the temporally distant future (i.e., an occupancy measure) using flow matching. As large datasets are often constructed by many distinct users performing distinct tasks, we include in our model a latent variable capturing the user intention. This intention increases the expressivity of our model, and enables adaptation with generalized policy improvement. We call our proposed method intention-conditioned flow occupancy models (InFOM). Comparing with alternative methods for pre-training, our experiments on $36$ state-based and $4$ image-based benchmark tasks demonstrate that the proposed method achieves $1.8 \times$ median improvement in returns and increases success rates by $36\%$. Website: this https URL Code: this https URL 

**Abstract (ZH)**: 大规模预训练从根本上改变了当今的机器学习研究方式：大型基础模型只需训练一次，然后可以通过社区中的任何人都可以使用（包括那些没有数据或计算资源从头训练模型的人），来适应和微调特定任务。将相同框架应用到强化学习中是诱人的，因为它提供了应对强化学习核心挑战（包括样本效率和稳健性）的有力途径。然而，在强化学习背景下预训练大型模型仍存在基本挑战：动作具有长期依赖性，因此训练跨越时间进行推理的基础模型是重要的。生成人工智能的 recent 进展提供了建模高度复杂分布的新工具。在本文中，我们建立了一个概率模型来预测智能体将在远距离将来访问哪些状态（即，占用度量），并使用流匹配。由于大型数据集通常由许多执行不同任务的用户构建，我们将用户意图作为一个潜在变量纳入我们的模型中。这种意图增强了模型的表达能力，并能够通过泛化策略改进实现适应性。我们提出的办法称为意图条件化流占用模型（InFOM）。与替代的预训练方法相比，我们在 36 个基于状态和 4 个基于图像的基准任务上的实验表明，所提出的方法在回报上实现了中位数 1.8 倍的改进，并将成功率提高了 36%。网址：这个 https URL 代码：这个 https URL 

---
# From Legal Texts to Defeasible Deontic Logic via LLMs: A Study in Automated Semantic Analysis 

**Title (ZH)**: 从法律文本到可逆规范逻辑：基于LLM的自动语义分析研究 

**Authors**: Elias Horner, Cristinel Mateis, Guido Governatori, Agata Ciabattoni  

**Link**: [PDF](https://arxiv.org/pdf/2506.08899)  

**Abstract**: We present a novel approach to the automated semantic analysis of legal texts using large language models (LLMs), targeting their transformation into formal representations in Defeasible Deontic Logic (DDL). We propose a structured pipeline that segments complex normative language into atomic snippets, extracts deontic rules, and evaluates them for syntactic and semantic coherence. Our methodology is evaluated across various LLM configurations, including prompt engineering strategies, fine-tuned models, and multi-stage pipelines, focusing on legal norms from the Australian Telecommunications Consumer Protections Code. Empirical results demonstrate promising alignment between machine-generated and expert-crafted formalizations, showing that LLMs - particularly when prompted effectively - can significantly contribute to scalable legal informatics. 

**Abstract (ZH)**: 我们提出了一种使用大语言模型（LLMs）自动进行法律文本语义分析的新型方法，目标是将其转换为 defeasible deontic logic (DDL) 的形式化表示。我们建议了一个结构化的流水线，将复杂的规范语言分割为原子片段，提取规范规则，并评估其在语法和语义上的连贯性。我们的方法在各种 LLM 配置下进行了评估，包括提示工程策略、微调模型和多阶段流水线，重点关注澳大利亚电信消费者保护代码中的法律规范。实证结果表明，机器生成的形式化与专家手工编写的形式化之间存在令人鼓舞的一致性，显示了当有效地提示时，LLMs 可以显著贡献于可扩展的法律信息化。 

---
# PlantBert: An Open Source Language Model for Plant Science 

**Title (ZH)**: PlantBert: 一种开源植物科学语言模型 

**Authors**: Hiba Khey, Amine Lakhder, Salma Rouichi, Imane El Ghabi, Kamal Hejjaoui, Younes En-nahli, Fahd Kalloubi, Moez Amri  

**Link**: [PDF](https://arxiv.org/pdf/2506.08897)  

**Abstract**: The rapid advancement of transformer-based language models has catalyzed breakthroughs in biomedical and clinical natural language processing; however, plant science remains markedly underserved by such domain-adapted tools. In this work, we present PlantBert, a high-performance, open-source language model specifically tailored for extracting structured knowledge from plant stress-response literature. Built upon the DeBERTa architecture-known for its disentangled attention and robust contextual encoding-PlantBert is fine-tuned on a meticulously curated corpus of expert-annotated abstracts, with a primary focus on lentil (Lens culinaris) responses to diverse abiotic and biotic stressors. Our methodology combines transformer-based modeling with rule-enhanced linguistic post-processing and ontology-grounded entity normalization, enabling PlantBert to capture biologically meaningful relationships with precision and semantic fidelity. The underlying corpus is annotated using a hierarchical schema aligned with the Crop Ontology, encompassing molecular, physiological, biochemical, and agronomic dimensions of plant adaptation. PlantBert exhibits strong generalization capabilities across entity types and demonstrates the feasibility of robust domain adaptation in low-resource scientific fields. By providing a scalable and reproducible framework for high-resolution entity recognition, PlantBert bridges a critical gap in agricultural NLP and paves the way for intelligent, data-driven systems in plant genomics, phenomics, and agronomic knowledge discovery. Our model is publicly released to promote transparency and accelerate cross-disciplinary innovation in computational plant science. 

**Abstract (ZH)**: 基于变压器的语言模型的快速进步推动了生物医学和临床自然语言处理领域的突破；然而，植物科学领域仍严重缺乏此类适应性工具。在本工作中，我们提出PlantBert，这是一种高性能的开源语言模型，专门用于从植物胁迫响应文献中提取结构化知识。PlantBert基于DeBERTa架构构建，该架构以其分离的注意力和 robust 的上下文编码著称，并在详细筛选和专家标注的摘要语料库上进行微调，主要关注裂谷扁豆（Lens culinaris）对不同非生物和生物胁迫的响应。本方法结合了基于变压器的建模与规则增强的语义后处理以及本体导向的实体规范化，使PlantBert能够精确而忠实地捕捉生物学相关的关联关系。底层语料库使用与作物本体学对齐的层次结构进行标注，涵盖了植物适应性的分子、生理、生物化学和农业学维度。PlantBert表现出强大的实体类型泛化能力，展示了在低资源科学领域实现稳健领域适应的可能性。通过提供一种可扩展且可重复的框架，用于高分辨率实体识别，PlantBert弥合了农业自然语言处理的关键缺口，并为植物基因组学、表型学和农业知识发现提供了智能、数据驱动的系统。我们的模型公开发布，以促进透明度并加速跨学科在计算植物科学中的创新。 

---
# Product of Experts for Visual Generation 

**Title (ZH)**: 专家系统相乘方法在视觉生成中的应用 

**Authors**: Yunzhi Zhang, Carson Murtuza-Lanier, Zizhang Li, Yilun Du, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08894)  

**Abstract**: Modern neural models capture rich priors and have complementary knowledge over shared data domains, e.g., images and videos. Integrating diverse knowledge from multiple sources -- including visual generative models, visual language models, and sources with human-crafted knowledge such as graphics engines and physics simulators -- remains under-explored. We propose a Product of Experts (PoE) framework that performs inference-time knowledge composition from heterogeneous models. This training-free approach samples from the product distribution across experts via Annealed Importance Sampling (AIS). Our framework shows practical benefits in image and video synthesis tasks, yielding better controllability than monolithic methods and additionally providing flexible user interfaces for specifying visual generation goals. 

**Abstract (ZH)**: 现代神经模型捕获丰富的先验知识并在共享数据域（如图像和视频）上互补。从多种来源综合多样知识——包括视觉生成模型、视觉语言模型以及包含人工构建知识（如图形引擎和物理模拟器）的来源——仍然未被充分探索。我们提出了一种专家乘积（PoE）框架，在推断时从异构模型中进行知识综合。这种无需训练的方法通过退火重要性采样（AIS）从专家产品的分布中采样。我们的框架在图像和视频合成任务中展示了实际优势，提供了比单体方法更好的可控性，并且还提供了灵活的用户界面来指定视觉生成目标。 

---
# SeerAttention-R: Sparse Attention Adaptation for Long Reasoning 

**Title (ZH)**: SeerAttention-R：稀疏注意机制适应于长跨度推理 

**Authors**: Yizhao Gao, Shuming Guo, Shijie Cao, Yuqing Xia, Yu Cheng, Lei Wang, Lingxiao Ma, Yutao Sun, Tianzhu Ye, Li Dong, Hayden Kwok-Hay So, Yu Hua, Ting Cao, Fan Yang, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08889)  

**Abstract**: We introduce SeerAttention-R, a sparse attention framework specifically tailored for the long decoding of reasoning models. Extended from SeerAttention, SeerAttention-R retains the design of learning attention sparsity through a self-distilled gating mechanism, while removing query pooling to accommodate auto-regressive decoding. With a lightweight plug-in gating, SeerAttention-R is flexible and can be easily integrated into existing pretrained model without modifying the original parameters. We demonstrate that SeerAttention-R, trained on just 0.4B tokens, maintains near-lossless reasoning accuracy with 4K token budget in AIME benchmark under large sparse attention block sizes (64/128). Using TileLang, we develop a highly optimized sparse decoding kernel that achieves near-theoretical speedups of up to 9x over FlashAttention-3 on H100 GPU at 90% sparsity. Code is available at: this https URL. 

**Abstract (ZH)**: SeerAttention-R：一种特定于长推理解码的稀疏注意力框架 

---
# On The Impact of Merge Request Deviations on Code Review Practices 

**Title (ZH)**: merge请求偏差对代码审查实践的影响 

**Authors**: Samah Kansab, Francis Bordeleau, Ali Tizghadam  

**Link**: [PDF](https://arxiv.org/pdf/2506.08860)  

**Abstract**: Code review is a key practice in software engineering, ensuring quality and collaboration. However, industrial Merge Request (MR) workflows often deviate from standardized review processes, with many MRs serving non-review purposes (e.g., drafts, rebases, or dependency updates). We term these cases deviations and hypothesize that ignoring them biases analytics and undermines ML models for review analysis.
We identify seven deviation categories, occurring in 37.02% of MRs, and propose a few-shot learning detection method (91% accuracy). By excluding deviations, ML models predicting review completion time improve performance in 53.33% of cases (up to 2.25x) and exhibit significant shifts in feature importance (47% overall, 60% top-*k*).
Our contributions include: (1) a taxonomy of MR deviations, (2) an AI-driven detection approach, and (3) empirical evidence of their impact on ML-based review analytics. This work aids practitioners in optimizing review efforts and ensuring reliable insights. 

**Abstract (ZH)**: 代码审查是软件工程中的关键实践，保障质量和协作。然而，工业合并请求（MR）工作流程常偏离标准的审查流程，许多MR非审查用途（如草稿、重构或依赖更新）。我们把这些情况称为偏差，并假设忽略它们会偏斜分析并削弱审查分析的机器学习模型。 

---
# Spatial Transcriptomics Expression Prediction from Histopathology Based on Cross-Modal Mask Reconstruction and Contrastive Learning 

**Title (ZH)**: 基于跨模态掩码重构与对比学习的病理图像空间转录组表达预测 

**Authors**: Junzhuo Liu, Markus Eckstein, Zhixiang Wang, Friedrich Feuerhake, Dorit Merhof  

**Link**: [PDF](https://arxiv.org/pdf/2506.08854)  

**Abstract**: Spatial transcriptomics is a technology that captures gene expression levels at different spatial locations, widely used in tumor microenvironment analysis and molecular profiling of histopathology, providing valuable insights into resolving gene expression and clinical diagnosis of cancer. Due to the high cost of data acquisition, large-scale spatial transcriptomics data remain challenging to obtain. In this study, we develop a contrastive learning-based deep learning method to predict spatially resolved gene expression from whole-slide images. Evaluation across six different disease datasets demonstrates that, compared to existing studies, our method improves Pearson Correlation Coefficient (PCC) in the prediction of highly expressed genes, highly variable genes, and marker genes by 6.27%, 6.11%, and 11.26% respectively. Further analysis indicates that our method preserves gene-gene correlations and applies to datasets with limited samples. Additionally, our method exhibits potential in cancer tissue localization based on biomarker expression. 

**Abstract (ZH)**: 空间转录组学是一种在不同空间位置捕获基因表达水平的技术，广泛应用于肿瘤微环境分析和病理组织学的分子分型，提供了解决基因表达和癌症临床诊断的重要见解。由于数据获取成本高，大规模的空间转录组学数据仍具有挑战性。在本研究中，我们开发了一种基于对比学习的深度学习方法，用于从全切片图像预测空间解析的基因表达。在六个不同疾病数据集上的评估表明，与现有研究相比，我们的方法在预测高表达基因、高变异性基因和标记基因方面的皮尔逊相关系数（PCC）分别提高了6.27%、6.11%和11.26%。进一步分析表明，我们的方法能够保持基因-基因相关性，并适用于样本量有限的数据集。此外，我们的方法在基于生物标志物表达识别癌组织方面具有潜在应用价值。 

---
# CulturalFrames: Assessing Cultural Expectation Alignment in Text-to-Image Models and Evaluation Metrics 

**Title (ZH)**: 文化框架：评估文本到图像模型中的文化期望对齐及其评价指标 

**Authors**: Shravan Nayak, Mehar Bhatia, Xiaofeng Zhang, Verena Rieser, Lisa Anne Hendricks, Sjoerd van Steenkiste, Yash Goyal, Karolina Stańczak, Aishwarya Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2506.08835)  

**Abstract**: The increasing ubiquity of text-to-image (T2I) models as tools for visual content generation raises concerns about their ability to accurately represent diverse cultural contexts. In this work, we present the first study to systematically quantify the alignment of T2I models and evaluation metrics with respect to both explicit as well as implicit cultural expectations. To this end, we introduce CulturalFrames, a novel benchmark designed for rigorous human evaluation of cultural representation in visual generations. Spanning 10 countries and 5 socio-cultural domains, CulturalFrames comprises 983 prompts, 3637 corresponding images generated by 4 state-of-the-art T2I models, and over 10k detailed human annotations. We find that T2I models not only fail to meet the more challenging implicit expectations but also the less challenging explicit expectations. Across models and countries, cultural expectations are missed an average of 44% of the time. Among these failures, explicit expectations are missed at a surprisingly high average rate of 68%, while implicit expectation failures are also significant, averaging 49%. Furthermore, we demonstrate that existing T2I evaluation metrics correlate poorly with human judgments of cultural alignment, irrespective of their internal reasoning. Collectively, our findings expose critical gaps, providing actionable directions for developing more culturally informed T2I models and evaluation methodologies. 

**Abstract (ZH)**: steadily increasing 的普及性使文本-to-图像 (T2I) 模型作为视觉内容生成工具的应用越来越广泛，这引发了对其准确代表多元文化背景能力的担忧。本文首次系统地量化了T2I模型和评估指标与显性及隐性文化期望之间的对齐情况。为此，我们提出了 CulturalFrames，一个旨在严格评估视觉生成中文化表现的新基准。该基准覆盖10个国家和5个社会文化领域，包括983个提示词、由4个最新T2I模型生成的3637张相应图像，以及超过10000条详细的人类注释。我们发现，T2I模型不仅未能满足更具挑战性的隐性期望，就连较简单的显性期望也未能达到。在国家和模型之间，文化期望平均被忽视44%。在这类失败中，显性期望的失败率出人意料地高达68%，而隐性期望的失败率也相当显著，平均为49%。此外，我们还证明现有的T2I评估指标与人类对文化对齐性的判断几乎没有关联，无论其内部逻辑如何。综合我们的发现揭示了关键的差距，并提供了开发更具文化认知的T2I模型和评估方法的实用指导。 

---
# The impact of fine tuning in LLaMA on hallucinations for named entity extraction in legal documentation 

**Title (ZH)**: LLaMA微调对法律文档中命名实体提取幻觉的影响 

**Authors**: Francisco Vargas, Alejandro González Coene, Gaston Escalante, Exequiel Lobón, Manuel Pulido  

**Link**: [PDF](https://arxiv.org/pdf/2506.08827)  

**Abstract**: The extraction of information about traffic accidents from legal documents is crucial for quantifying insurance company costs. Extracting entities such as percentages of physical and/or psychological disability and the involved compensation amounts is a challenging process, even for experts, due to the subtle arguments and reasoning in the court decision. A two-step procedure is proposed: first, segmenting the document identifying the most relevant segments, and then extracting the entities. For text segmentation, two methodologies are compared: a classic method based on regular expressions and a second approach that divides the document into blocks of n-tokens, which are then vectorized using multilingual models for semantic searches (text-embedding-ada-002/MiniLM-L12-v2 ). Subsequently, large language models (LLaMA-2 7b, 70b, LLaMA-3 8b, and GPT-4 Turbo) are applied with prompting to the selected segments for entity extraction. For the LLaMA models, fine-tuning is performed using LoRA. LLaMA-2 7b, even with zero temperature, shows a significant number of hallucinations in extractions which are an important contention point for named entity extraction. This work shows that these hallucinations are substantially reduced after finetuning the model. The performance of the methodology based on segment vectorization and subsequent use of LLMs significantly surpasses the classic method which achieves an accuracy of 39.5%. Among open-source models, LLaMA-2 70B with finetuning achieves the highest accuracy 79.4%, surpassing its base version 61.7%. Notably, the base LLaMA-3 8B model already performs comparably to the finetuned LLaMA-2 70B model, achieving 76.6%, highlighting the rapid progress in model development. Meanwhile, GPT-4 Turbo achieves the highest accuracy at 86.1%. 

**Abstract (ZH)**: 从法律文件中提取交通事故信息对于量化保险公司成本至关重要。一种两步程序被提出：首先，识别最相关的文档段落进行分段，然后从中提取实体。在文本分段方面，对比了两种方法：基于正则表达式的传统方法和将文档划分为n-克隆块并使用多语言模型进行语义搜索（text-embedding-ada-002/MiniLM-L12-v2）的方法。随后，使用提示技术应用大型语言模型（LLaMA-2 7b、70b、LLaMA-3 8b和GPT-4 Turbo）对选定段落进行实体提取。对于LLaMA模型，进行了LoRA微调。即使在零温度下，LLaMA-2 7b也表现出大量幻觉，这在命名实体提取中是一个重要的争议点。本研究显示，模型微调后这些幻觉显著减少。基于段落向量化的方法和随后使用大型语言模型的方法在准确性上显著超越了传统方法，传统方法的准确率为39.5%。在开源模型中，微调后的LLaMA-2 70B准确率最高，达到79.4%，超过了其基础版本61.7%。值得注意的是，基础的LLaMA-3 8B模型在准确率方面已经与微调后的LLaMA-2 70B模型相当，达到了76.6%，突显了模型开发的快速进展。与此同时，GPT-4 Turbo实现了最高的准确率86.1%。 

---
# FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency 

**Title (ZH)**: FreqPolicy: 通过频率一致性实现高效的基于流的视运动策略 

**Authors**: Yifei Su, Ning Liu, Dong Chen, Zhen Zhao, Kun Wu, Meng Li, Zhiyuan Xu, Zhengping Che, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08822)  

**Abstract**: Generative modeling-based visuomotor policies have been widely adopted in robotic manipulation attributed to their ability to model multimodal action distributions. However, the high inference cost of multi-step sampling limits their applicability in real-time robotic systems. To address this issue, existing approaches accelerate the sampling process in generative modeling-based visuomotor policies by adapting acceleration techniques originally developed for image generation. Despite this progress, a major distinction remains: image generation typically involves producing independent samples without temporal dependencies, whereas robotic manipulation involves generating time-series action trajectories that require continuity and temporal coherence. To effectively exploit temporal information in robotic manipulation, we propose FreqPolicy, a novel approach that first imposes frequency consistency constraints on flow-based visuomotor policies. Our work enables the action model to capture temporal structure effectively while supporting efficient, high-quality one-step action generation. We introduce a frequency consistency constraint that enforces alignment of frequency-domain action features across different timesteps along the flow, thereby promoting convergence of one-step action generation toward the target distribution. In addition, we design an adaptive consistency loss to capture structural temporal variations inherent in robotic manipulation tasks. We assess FreqPolicy on 53 tasks across 3 simulation benchmarks, proving its superiority over existing one-step action generators. We further integrate FreqPolicy into the vision-language-action (VLA) model and achieve acceleration without performance degradation on the 40 tasks of Libero. Besides, we show efficiency and effectiveness in real-world robotic scenarios with an inference frequency 93.5Hz. The code will be publicly available. 

**Abstract (ZH)**: 基于生成模型的视运动策略在机器人操作中的应用得益于其建模多模态动作分布的能力。然而，多步采样的高推断成本限制了其在实时机器人系统中的应用。为了解决这一问题，现有方法通过适应为图像生成开发的加速技术来加速基于生成模型的视运动策略的采样过程。尽管取得了进展，但仍存在一个重要区别：图像生成通常涉及生成独立样本而不包含时间依赖性，而机器人操作涉及生成需要连续性和时间一致性的时序动作轨迹。为了有效利用机器人操作中的时间信息，我们提出了FreqPolicy这一新颖方法，首先在流基于的视运动策略上施加频率一致性约束。我们的工作使得动作模型能够有效地捕捉时间结构，并支持高效、高质量的一步动作生成。我们引入了一种频率一致性约束，该约束要求沿流的不同时间步长中动作特征的频率域对齐，从而促进一步动作生成向目标分布收敛。此外，我们设计了一种自适应一致性损失，以捕捉机器人操作任务中固有的结构时间变化。我们在3个仿真基准上的53个任务上评估了FreqPolicy，证明了它在现有的一步动作生成器中的优越性。我们进一步将FreqPolicy集成到视语言动作（VLA）模型中，在Libero的40个任务上实现了加速且不降低性能。此外，我们展示了其在实际机器人场景中的高效性和有效性，推断频率为93.5Hz。代码将公开。 

---
# Towards Biosignals-Free Autonomous Prosthetic Hand Control via Imitation Learning 

**Title (ZH)**: 基于 imitation learning 的无生物信号自主假手控制 

**Authors**: Kaijie Shi, Wanglong Lu, Hanli Zhao, Vinicius Prado da Fonseca, Ting Zou, Xianta Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08795)  

**Abstract**: Limb loss affects millions globally, impairing physical function and reducing quality of life. Most traditional surface electromyographic (sEMG) and semi-autonomous methods require users to generate myoelectric signals for each control, imposing physically and mentally taxing demands. This study aims to develop a fully autonomous control system that enables a prosthetic hand to automatically grasp and release objects of various shapes using only a camera attached to the wrist. By placing the hand near an object, the system will automatically execute grasping actions with a proper grip force in response to the hand's movements and the environment. To release the object being grasped, just naturally place the object close to the table and the system will automatically open the hand. Such a system would provide individuals with limb loss with a very easy-to-use prosthetic control interface and greatly reduce mental effort while using. To achieve this goal, we developed a teleoperation system to collect human demonstration data for training the prosthetic hand control model using imitation learning, which mimics the prosthetic hand actions from human. Through training the model using only a few objects' data from one single participant, we have shown that the imitation learning algorithm can achieve high success rates, generalizing to more individuals and unseen objects with a variation of weights. The demonstrations are available at \href{this https URL}{this https URL} 

**Abstract (ZH)**: 截肢影响全球数百万人，妨碍物理功能并降低生活质量。大多数传统的表面肌电图（sEMG）和半自主方法要求用户为每个控制生成肌电信号，这给用户带来了身体和精神上的负担。本研究旨在开发一种完全自主的控制系统，使假手能够在佩戴手腕摄像头的情况下自动抓取和释放各种形状的物体。通过将手置于物体附近，系统将根据手的运动和环境自动执行适当握力的抓取动作。松开被抓取的物体时，只需自然地将物体靠近桌面，系统将自动打开手掌。这样一种系统将为截肢人士提供一个非常易于使用的假手控制界面，并在使用时大大减少精神负担。为了实现这一目标，我们开发了一个遥控系统来收集人类示范数据，使用模仿学习训练假手控制模型，该模型模仿人类的假手动作。通过仅使用单个参与者几种物体的数据训练模型，我们已经证明模仿学习算法可以在不同的个体和未见过的物体上实现高的成功率，并通过调整权重进行泛化。示范数据可在 \href{this https URL}{this https URL} 获取。 

---
# Do Generative AI Tools Ensure Green Code? An Investigative Study 

**Title (ZH)**: 生成式AI工具能否确保绿色代码？一项调查研究 

**Authors**: Samarth Sikand, Rohit Mehra, Vibhu Saujanya Sharma, Vikrant Kaulgud, Sanjay Podder, Adam P. Burden  

**Link**: [PDF](https://arxiv.org/pdf/2506.08790)  

**Abstract**: Software sustainability is emerging as a primary concern, aiming to optimize resource utilization, minimize environmental impact, and promote a greener, more resilient digital ecosystem. The sustainability or "greenness" of software is typically determined by the adoption of sustainable coding practices. With a maturing ecosystem around generative AI, many software developers now rely on these tools to generate code using natural language prompts. Despite their potential advantages, there is a significant lack of studies on the sustainability aspects of AI-generated code. Specifically, how environmentally friendly is the AI-generated code based upon its adoption of sustainable coding practices? In this paper, we present the results of an early investigation into the sustainability aspects of AI-generated code across three popular generative AI tools - ChatGPT, BARD, and Copilot. The results highlight the default non-green behavior of tools for generating code, across multiple rules and scenarios. It underscores the need for further in-depth investigations and effective remediation strategies. 

**Abstract (ZH)**: 软件可持续性正逐渐成为主要关切点，旨在优化资源利用、减少环境影响，并促进更绿色、更具 resilience 的数字生态系统。软件的可持续性或“绿色性”通常由可持续编码实践的采纳来确定。随着生成式 AI 生态系统的发展成熟，许多软件开发人员现在开始依赖这些工具通过自然语言提示生成代码。尽管生成式 AI 具有潜在优势，但关于 AI 生成代码的可持续性方面目前的研究严重不足。特别是，在采纳可持续编码实践的基础上，AI 生成的代码有多环保？在本文中，我们对最受欢迎的三种生成式 AI 工具（ChatGPT、BARD 和 Copilot）生成代码的可持续性方面进行了初步调查，并揭示了这些工具在多种规则和情景下的默认非绿色行为。这突显了进一步深入调查和有效补救策略的必要性。 

---
# POLARON: Precision-aware On-device Learning and Adaptive Runtime-cONfigurable AI acceleration 

**Title (ZH)**: Polaron：精度意识的-edge学习与适配运行时可配置的AI加速 

**Authors**: Mukul Lokhande, Santosh Kumar Vishvakarma  

**Link**: [PDF](https://arxiv.org/pdf/2506.08785)  

**Abstract**: The increasing complexity of AI models requires flexible hardware capable of supporting diverse precision formats, particularly for energy-constrained edge platforms. This work presents PARV-CE, a SIMD-enabled, multi-precision MAC engine that performs efficient multiply-accumulate operations using a unified data-path for 4/8/16-bit fixed-point, floating point, and posit formats. The architecture incorporates a layer adaptive precision strategy to align computational accuracy with workload sensitivity, optimizing both performance and energy usage. PARV-CE integrates quantization-aware execution with a reconfigurable SIMD pipeline, enabling high-throughput processing with minimal overhead through hardware-software co-design. The results demonstrate up to 2x improvement in PDP and 3x reduction in resource usage compared to SoTA designs, while retaining accuracy within 1.8% FP32 baseline. The architecture supports both on-device training and inference across a range of workloads, including DNNs, RNNs, RL, and Transformer models. The empirical analysis establish PARVCE incorporated POLARON as a scalable and energy-efficient solution for precision-adaptive AI acceleration at edge. 

**Abstract (ZH)**: AI模型日益增加的复杂性要求具有灵活硬件的支持以适应多种精度格式，特别是在能量受限的边缘平台。本工作提出了PARV-CE，这是一种支持单指令多数据（SIMD）且具备多种精度MAC引擎，能够在统一数据路径下高效地执行4/8/16位定点、浮点和Posit格式的乘累加操作。该架构集成了层自适应精度策略，使计算精度与工作负载敏感性保持一致，从而优化性能和能耗。PARV-CE将感知量化执行与可重构SIMD管道相结合，通过硬件软件协同设计实现高吞吐量处理同时减少开销。实验结果表明，与现有最佳设计相比，PARV-CE在每周期定点吞吐量（PDP）上提高了2倍，资源使用量降低了3倍，同时保持在FP32基线准确性范围内1.8%。该架构支持多种工作负载的设备上训练和推理，包括DNN、RNN、RL和Transformer模型。实证分析表明，PARVCE结合Polaron是边缘上可扩展且能效高的精度自适应AI加速解决方案。 

---
# Multimodal Representation Alignment for Cross-modal Information Retrieval 

**Title (ZH)**: 多模态表示对齐在跨模态信息检索中的应用 

**Authors**: Fan Xu, Luis A. Leiva  

**Link**: [PDF](https://arxiv.org/pdf/2506.08774)  

**Abstract**: Different machine learning models can represent the same underlying concept in different ways. This variability is particularly valuable for in-the-wild multimodal retrieval, where the objective is to identify the corresponding representation in one modality given another modality as input. This challenge can be effectively framed as a feature alignment problem. For example, given a sentence encoded by a language model, retrieve the most semantically aligned image based on features produced by an image encoder, or vice versa. In this work, we first investigate the geometric relationships between visual and textual embeddings derived from both vision-language models and combined unimodal models. We then align these representations using four standard similarity metrics as well as two learned ones, implemented via neural networks. Our findings indicate that the Wasserstein distance can serve as an informative measure of the modality gap, while cosine similarity consistently outperforms alternative metrics in feature alignment tasks. Furthermore, we observe that conventional architectures such as multilayer perceptrons are insufficient for capturing the complex interactions between image and text representations. Our study offers novel insights and practical considerations for researchers working in multimodal information retrieval, particularly in real-world, cross-modal applications. 

**Abstract (ZH)**: 不同的机器学习模型可以以不同的方式表示相同的潜在概念。这种多样性特别适用于实时多模态检索，在这种检索中，目标是根据输入的另一种模态识别相应的表示。这一挑战可以有效地被框架为特征对齐问题。例如，给定由语言模型编码的一句话，基于图像编码器产生的特征检索最具有语义对齐的图像，反之亦然。在本文中，我们首先研究来自视觉-语言模型和联合单模态模型的视觉和文本嵌入之间的几何关系，然后使用四种标准相似度度量以及两种通过神经网络实现的学习度量来对这些表示进行对齐。我们的研究结果表明，Wasserstein距离可以作为模态差距的一个有用度量，而余弦相似度在特征对齐任务中始终优于其他度量。此外，我们观察到，传统的架构如多层感知机不足以捕捉图像和文本表示之间的复杂交互。我们的研究为从事多模态信息检索的研究人员，特别是在现实世界、跨模态应用方面，提供了新的见解和实际考虑。 

---
# Bayesian Inverse Physics for Neuro-Symbolic Robot Learning 

**Title (ZH)**: 基于贝叶斯逆物理的神经符号机器人学习 

**Authors**: Octavio Arriaga, Rebecca Adam, Melvin Laux, Lisa Gutzeit, Marco Ragni, Jan Peters, Frank Kirchner  

**Link**: [PDF](https://arxiv.org/pdf/2506.08756)  

**Abstract**: Real-world robotic applications, from autonomous exploration to assistive technologies, require adaptive, interpretable, and data-efficient learning paradigms. While deep learning architectures and foundation models have driven significant advances in diverse robotic applications, they remain limited in their ability to operate efficiently and reliably in unknown and dynamic environments. In this position paper, we critically assess these limitations and introduce a conceptual framework for combining data-driven learning with deliberate, structured reasoning. Specifically, we propose leveraging differentiable physics for efficient world modeling, Bayesian inference for uncertainty-aware decision-making, and meta-learning for rapid adaptation to new tasks. By embedding physical symbolic reasoning within neural models, robots could generalize beyond their training data, reason about novel situations, and continuously expand their knowledge. We argue that such hybrid neuro-symbolic architectures are essential for the next generation of autonomous systems, and to this end, we provide a research roadmap to guide and accelerate their development. 

**Abstract (ZH)**: 现实世界中的机器人应用，从自主探索到辅助技术，需要适应性强、可解释性强和数据高效的学习范式。虽然深度学习架构和基础模型在多种机器人应用中推动了显著的进步，但在未知和动态环境中高效可靠地运行方面仍然有限。在本文中，我们批判性地评估了这些局限性，并提出了一种结合数据驱动学习与故意的结构化推理的概念框架。具体而言，我们建议利用可微物理学进行高效的世界建模，使用贝叶斯推断进行不确定性意识的决策，以及使用元学习进行快速的新任务适应。通过将物理符号推理嵌入神经模型中，机器人可以泛化到训练数据之外，推理新情况，并持续扩展其知识。我们认为，这种混合神经符号架构是下一代自主系统的关键，为此，我们提供了一条研究 roadmap，以指导和加速其发展。 

---
# Factors affecting the in-context learning abilities of LLMs for dialogue state tracking 

**Title (ZH)**: 影响LLM在上下文对话状态跟踪中学习能力的因素 

**Authors**: Pradyoth Hegde, Santosh Kesiraju, Jan Švec, Šimon Sedláček, Bolaji Yusuf, Oldřich Plchot, Deepak K T, Jan Černocký  

**Link**: [PDF](https://arxiv.org/pdf/2506.08753)  

**Abstract**: This study explores the application of in-context learning (ICL) to the dialogue state tracking (DST) problem and investigates the factors that influence its effectiveness. We use a sentence embedding based k-nearest neighbour method to retrieve the suitable demonstrations for ICL. The selected demonstrations, along with the test samples, are structured within a template as input to the LLM. We then conduct a systematic study to analyse the impact of factors related to demonstration selection and prompt context on DST performance. This work is conducted using the MultiWoZ2.4 dataset and focuses primarily on the OLMo-7B-instruct, Mistral-7B-Instruct-v0.3, and Llama3.2-3B-Instruct models. Our findings provide several useful insights on in-context learning abilities of LLMs for dialogue state tracking. 

**Abstract (ZH)**: 本研究探索了上下文学习（ICL）在对话状态跟踪（DST）问题中的应用，并调查了影响其有效性的因素。我们使用基于句子嵌入的K最近邻方法检索适合的示范以用于ICL。所选示范与测试样本一起按照模板格式作为输入提供给LLM。然后，我们进行系统研究以分析示范选择和提示上下文因子对DST性能的影响。本研究使用MultiWoZ2.4数据集，并着重于OLMo-7B-instruct、Mistral-7B-Instruct-v0.3和Llama3.2-3B-Instruct模型。我们的发现提供了关于LLMs在对话状态跟踪中上下文学习能力的一些有用见解。 

---
# Bridging RDF Knowledge Graphs with Graph Neural Networks for Semantically-Rich Recommender Systems 

**Title (ZH)**: 基于图神经网络的RDF知识图谱在语义丰富的推荐系统中的融合 

**Authors**: Michael Färber, David Lamprecht, Yuni Susanti  

**Link**: [PDF](https://arxiv.org/pdf/2506.08743)  

**Abstract**: Graph Neural Networks (GNNs) have substantially advanced the field of recommender systems. However, despite the creation of more than a thousand knowledge graphs (KGs) under the W3C standard RDF, their rich semantic information has not yet been fully leveraged in GNN-based recommender systems. To address this gap, we propose a comprehensive integration of RDF KGs with GNNs that utilizes both the topological information from RDF object properties and the content information from RDF datatype properties. Our main focus is an in-depth evaluation of various GNNs, analyzing how different semantic feature initializations and types of graph structure heterogeneity influence their performance in recommendation tasks. Through experiments across multiple recommendation scenarios involving multi-million-node RDF graphs, we demonstrate that harnessing the semantic richness of RDF KGs significantly improves recommender systems and lays the groundwork for GNN-based recommender systems for the Linked Open Data cloud. The code and data are available on our GitHub repository: this https URL 

**Abstract (ZH)**: RDF知识图谱与图神经网络的综合集成及其在推荐系统中的应用研究 

---
# Societal AI Research Has Become Less Interdisciplinary 

**Title (ZH)**: 社会AI研究的跨学科性已经减弱。 

**Authors**: Dror Kris Markus, Fabrizio Gilardi, Daria Stetsenko  

**Link**: [PDF](https://arxiv.org/pdf/2506.08738)  

**Abstract**: As artificial intelligence (AI) systems become deeply embedded in everyday life, calls to align AI development with ethical and societal values have intensified. Interdisciplinary collaboration is often championed as a key pathway for fostering such engagement. Yet it remains unclear whether interdisciplinary research teams are actually leading this shift in practice. This study analyzes over 100,000 AI-related papers published on ArXiv between 2014 and 2024 to examine how ethical values and societal concerns are integrated into technical AI research. We develop a classifier to identify societal content and measure the extent to which research papers express these considerations. We find a striking shift: while interdisciplinary teams remain more likely to produce societally-oriented research, computer science-only teams now account for a growing share of the field's overall societal output. These teams are increasingly integrating societal concerns into their papers and tackling a wide range of domains - from fairness and safety to healthcare and misinformation. These findings challenge common assumptions about the drivers of societal AI and raise important questions. First, what are the implications for emerging understandings of AI safety and governance if most societally-oriented research is being undertaken by exclusively technical teams? Second, for scholars in the social sciences and humanities: in a technical field increasingly responsive to societal demands, what distinctive perspectives can we still offer to help shape the future of AI? 

**Abstract (ZH)**: 随着人工智能（AI）系统在日常生活中越来越深入，要求将AI发展与伦理和社会价值相一致的呼声不断增强。跨学科合作常被推崇为促进这种参与的关键途径。然而，目前尚不清楚跨学科研究团队是否真正引领了这种实践转变。本研究分析了2014年至2024年间发表在ArXiv上的超过10万篇AI相关论文，以考察伦理价值观和社会关切如何融入技术AI研究。我们开发了一个分类器来识别社会内容，并衡量研究论文表达这些考虑的程度。我们发现一个显著转变：尽管跨学科团队仍更有可能产生以社会为导向的研究，但仅计算机科学团队现在在整体社会输出中占据了快速增长的份额。这些团队越来越多地将其社会关切融入论文，并涵盖了广泛领域——从公平性和安全性到医疗保健和假信息。这些发现挑战了关于社会AI驱动力的常见认知，并提出了重要问题。首先，如果大多数社会导向的研究是由纯粹技术团队来完成的，这对新兴的AI安全性与治理理解有何影响？其次，对于社会科学和人文学科的学者而言：在一个越来越回应社会需求的技术领域，我们仍能提供哪些独特的视角来帮助塑造AI的未来？ 

---
# Exploration by Random Reward Perturbation 

**Title (ZH)**: 随机奖励扰动探索 

**Authors**: Haozhe Ma, Guoji Fu, Zhengding Luo, Jiele Wu, Tze-Yun Leong  

**Link**: [PDF](https://arxiv.org/pdf/2506.08737)  

**Abstract**: We introduce Random Reward Perturbation (RRP), a novel exploration strategy for reinforcement learning (RL). Our theoretical analyses demonstrate that adding zero-mean noise to environmental rewards effectively enhances policy diversity during training, thereby expanding the range of exploration. RRP is fully compatible with the action-perturbation-based exploration strategies, such as $\epsilon$-greedy, stochastic policies, and entropy regularization, providing additive improvements to exploration effects. It is general, lightweight, and can be integrated into existing RL algorithms with minimal implementation effort and negligible computational overhead. RRP establishes a theoretical connection between reward shaping and noise-driven exploration, highlighting their complementary potential. Experiments show that RRP significantly boosts the performance of Proximal Policy Optimization and Soft Actor-Critic, achieving higher sample efficiency and escaping local optima across various tasks, under both sparse and dense reward scenarios. 

**Abstract (ZH)**: 随机奖励扰动：强化学习的一种新颖探索策略 

---
# Geometric deep learning for local growth prediction on abdominal aortic aneurysm surfaces 

**Title (ZH)**: 几何深度学习在腹部主动脉瘤表面局部生长预测中的应用 

**Authors**: Dieuwertje Alblas, Patryk Rygiel, Julian Suk, Kaj O. Kappe, Marieke Hofman, Christoph Brune, Kak Khee Yeung, Jelmer M. Wolterink  

**Link**: [PDF](https://arxiv.org/pdf/2506.08729)  

**Abstract**: Abdominal aortic aneurysms (AAAs) are progressive focal dilatations of the abdominal aorta. AAAs may rupture, with a survival rate of only 20\%. Current clinical guidelines recommend elective surgical repair when the maximum AAA diameter exceeds 55 mm in men or 50 mm in women. Patients that do not meet these criteria are periodically monitored, with surveillance intervals based on the maximum AAA diameter. However, this diameter does not take into account the complex relation between the 3D AAA shape and its growth, making standardized intervals potentially unfit. Personalized AAA growth predictions could improve monitoring strategies. We propose to use an SE(3)-symmetric transformer model to predict AAA growth directly on the vascular model surface enriched with local, multi-physical features. In contrast to other works which have parameterized the AAA shape, this representation preserves the vascular surface's anatomical structure and geometric fidelity. We train our model using a longitudinal dataset of 113 computed tomography angiography (CTA) scans of 24 AAA patients at irregularly sampled intervals. After training, our model predicts AAA growth to the next scan moment with a median diameter error of 1.18 mm. We further demonstrate our model's utility to identify whether a patient will become eligible for elective repair within two years (acc = 0.93). Finally, we evaluate our model's generalization on an external validation set consisting of 25 CTAs from 7 AAA patients from a different hospital. Our results show that local directional AAA growth prediction from the vascular surface is feasible and may contribute to personalized surveillance strategies. 

**Abstract (ZH)**: 腹主动脉瘤（AAA）是腹主动脉进行性的局部扩张。AAA有可能破裂，其生存率为仅20%。当前临床指南建议，当男性AAA的最大直径超过55 mm或女性超过50 mm时，应进行选择性外科修复。不符合这些标准的患者将定期进行监测，监测间隔基于AAA的最大直径。然而，这一直径没有考虑到3D AAA形状与其生长之间的复杂关系，使标准化的间隔可能不合适。个性化的AAA生长预测可以改善监测策略。我们提出使用一种SE(3)-对称的变压器模型，直接在富含局部多物理特征的血管模型表面预测AAA生长。与将AAA形状参数化的其他工作不同，这种表示方式保留了血管表面的解剖结构和几何保真度。我们使用24名AAA患者的113个不规则采样间隔的计算机断层扫描血管造影（CTA）扫描 longitudinally 数据集对模型进行训练。训练后，我们的模型在下一扫描时刻预测AAA生长的中位直径误差为1.18 mm。进一步验证了该模型的实用性，可以预测患者在未来两年内是否符合选择性修复的条件（acc = 0.93）。最后，我们在一个外部验证集中评估了我们的模型，该集由另一家医院的7名AAA患者共25个CTA组成。结果表明，从血管表面进行局部方向性的AAA生长预测是可行的，并可能有助于个性化的监测策略。 

---
# Breaking the ICE: Exploring promises and challenges of benchmarks for Inference Carbon & Energy estimation for LLMs 

**Title (ZH)**: 突破ICE束缚：探究推理碳排放与能耗估算基准对大语言模型的机遇与挑战 

**Authors**: Samarth Sikand, Rohit Mehra, Priyavanshi Pathania, Nikhil Bamby, Vibhu Saujanya Sharma, Vikrant Kaulgud, Sanjay Podder, Adam P. Burden  

**Link**: [PDF](https://arxiv.org/pdf/2506.08727)  

**Abstract**: While Generative AI stands to be one of the fastest adopted technologies ever, studies have made evident that the usage of Large Language Models (LLMs) puts significant burden on energy grids and our environment. It may prove a hindrance to the Sustainability goals of any organization. A crucial step in any Sustainability strategy is monitoring or estimating the energy consumption of various components. While there exist multiple tools for monitoring energy consumption, there is a dearth of tools/frameworks for estimating the consumption or carbon emissions. Current drawbacks of both monitoring and estimation tools include high input data points, intrusive nature, high error margin, etc. We posit that leveraging emerging LLM benchmarks and related data points can help overcome aforementioned challenges while balancing accuracy of the emission estimations. To that extent, we discuss the challenges of current approaches and present our evolving framework, R-ICE, which estimates prompt level inference carbon emissions by leveraging existing state-of-the-art(SOTA) benchmark. This direction provides a more practical and non-intrusive way to enable emerging use-cases like dynamic LLM routing, carbon accounting, etc. Our promising validation results suggest that benchmark-based modelling holds great potential for inference emission estimation and warrants further exploration from the scientific community. 

**Abstract (ZH)**: 基于新兴大语言模型基准的大规模语言模型推理碳排放估算框架R-ICE 

---
# Improved LLM Agents for Financial Document Question Answering 

**Title (ZH)**: 改进的LLM代理模型在金融文档问答中的应用 

**Authors**: Nelvin Tan, Zian Seng, Liang Zhang, Yu-Ching Shih, Dong Yang, Amol Salunkhe  

**Link**: [PDF](https://arxiv.org/pdf/2506.08726)  

**Abstract**: Large language models (LLMs) have shown impressive capabilities on numerous natural language processing tasks. However, LLMs still struggle with numerical question answering for financial documents that include tabular and textual data. Recent works have showed the effectiveness of critic agents (i.e., self-correction) for this task given oracle labels. Building upon this framework, this paper examines the effectiveness of the traditional critic agent when oracle labels are not available, and show, through experiments, that this critic agent's performance deteriorates in this scenario. With this in mind, we present an improved critic agent, along with the calculator agent which outperforms the previous state-of-the-art approach (program-of-thought) and is safer. Furthermore, we investigate how our agents interact with each other, and how this interaction affects their performance. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多项自然语言处理任务中展现了卓越的能力。然而，LLMs 在处理包含表格和文本数据的金融文档的数值问答任务上仍然存在挑战。最近的研究表明，在有参考答案的情况下，批评代理（即自我纠错）对此任务是有效的。在这一框架的基础上，本文考察了在没有参考答案的情况下批评代理的有效性，并通过实验表明，在这种情况下，批评代理的性能下降。为此，我们提出了一种改进的批评代理，以及一种超越先前最佳方法（思考程序）的表现更优且更安全的计算器代理。此外，我们研究了这些代理之间的交互方式及其对性能的影响。 

---
# ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Large Language Model Preference Optimization 

**Title (ZH)**: ConfPO：利用策略模型置信度进行大型语言模型偏好优化中的关键词选择 

**Authors**: Hee Suk Yoon, Eunseop Yoon, Mark A. Hasegawa-Johnson, Sungwoong Kim, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2506.08712)  

**Abstract**: We introduce ConfPO, a method for preference learning in Large Language Models (LLMs) that identifies and optimizes preference-critical tokens based solely on the training policy's confidence, without requiring any auxiliary models or compute. Unlike prior Direct Alignment Algorithms (DAAs) such as Direct Preference Optimization (DPO), which uniformly adjust all token probabilities regardless of their relevance to preference, ConfPO focuses optimization on the most impactful tokens. This targeted approach improves alignment quality while mitigating overoptimization (i.e., reward hacking) by using the KL divergence budget more efficiently. In contrast to recent token-level methods that rely on credit-assignment models or AI annotators, raising concerns about scalability and reliability, ConfPO is simple, lightweight, and model-free. Experimental results on challenging alignment benchmarks, including AlpacaEval 2 and Arena-Hard, demonstrate that ConfPO consistently outperforms uniform DAAs across various LLMs, delivering better alignment with zero additional computational overhead. 

**Abstract (ZH)**: ConfPO：一种基于训练策略信心进行偏好学习的方法 

---
# Variational Autoencoder-Based Approach to Latent Feature Analysis on Efficient Representation of Power Load Monitoring Data 

**Title (ZH)**: 基于变分自编码器的潜在特征分析方法在电力负荷监测数据高效表示中的应用 

**Authors**: Boyu Xie, Tangtang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.08698)  

**Abstract**: With the development of smart grids, High-Dimensional and Incomplete (HDI) Power Load Monitoring (PLM) data challenges the performance of Power Load Forecasting (PLF) models. In this paper, we propose a potential characterization model VAE-LF based on Variational Autoencoder (VAE) for efficiently representing and complementing PLM missing data. VAE-LF learns a low-dimensional latent representation of the data using an Encoder-Decoder structure by splitting the HDI PLM data into vectors and feeding them sequentially into the VAE-LF model, and generates the complementary data. Experiments on the UK-DALE dataset show that VAE-LF outperforms other benchmark models in both 5% and 10% sparsity test cases, with significantly lower RMSE and MAE, and especially outperforms on low sparsity ratio data. The method provides an efficient data-completion solution for electric load management in smart grids. 

**Abstract (ZH)**: 基于变分自编码器的PLM潜在特征模型VAE-LF：高效表示与补充高维不完全功率负载监测数据 

---
# Enhancing Reasoning Capabilities of Small Language Models with Blueprints and Prompt Template Search 

**Title (ZH)**: 使用蓝图和提示模板搜索增强小型语言模型的推理能力 

**Authors**: Dongge Han, Menglin Xia, Daniel Madrigal Diaz, Samuel Kessler, Ankur Mallick, Xuchao Zhang, Mirian Del Carmen Hipolito Garcia, Jin Xu, Victor Rühle, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08669)  

**Abstract**: Small language models (SLMs) offer promising and efficient alternatives to large language models (LLMs). However, SLMs' limited capacity restricts their reasoning capabilities and makes them sensitive to prompt variations. To address these challenges, we propose a novel framework that enhances SLM reasoning capabilities through LLM generated blueprints. The blueprints provide structured, high-level reasoning guides that help SLMs systematically tackle related problems. Furthermore, our framework integrates a prompt template search mechanism to mitigate the SLMs' sensitivity to prompt variations. Our framework demonstrates improved SLM performance across various tasks, including math (GSM8K), coding (MBPP), and logic reasoning (BBH). Our approach improves the reasoning capabilities of SLMs without increasing model size or requiring additional training, offering a lightweight and deployment-friendly solution for on-device or resource-constrained environments. 

**Abstract (ZH)**: 小型语言模型（SLMs）提供了大型语言模型（LLMs）的有效替代方案。然而，SLMs的有限容量限制了它们的推理能力，并使其对提示变化敏感。为解决这些挑战，我们提出了一种新型框架，通过LLM生成的蓝图来增强SLMs的推理能力。蓝图提供了结构化的高级推理指南，帮助SLMs系统地解决相关问题。此外，我们的框架集成了一种提示模板搜索机制，以减轻SLMs对提示变化的敏感性。我们的框架在包括数学（GSM8K）、编程（MBPP）和逻辑推理（BBH）等各项任务上展示了改进的SLM性能。我们的方法提高了SLMs的推理能力，而无需增加模型大小或额外训练，提供了一种适用于设备端或资源受限环境的轻量级且易于部署的解决方案。 

---
# Optimizing Learned Image Compression on Scalar and Entropy-Constraint Quantization 

**Title (ZH)**: 基于标量和熵约束量化的学习图像压缩优化 

**Authors**: Florian Borzechowski, Michael Schäfer, Heiko Schwarz, Jonathan Pfaff, Detlev Marpe, Thomas Wiegand  

**Link**: [PDF](https://arxiv.org/pdf/2506.08662)  

**Abstract**: The continuous improvements on image compression with variational autoencoders have lead to learned codecs competitive with conventional approaches in terms of rate-distortion efficiency. Nonetheless, taking the quantization into account during the training process remains a problem, since it produces zero derivatives almost everywhere and needs to be replaced with a differentiable approximation which allows end-to-end optimization. Though there are different methods for approximating the quantization, none of them model the quantization noise correctly and thus, result in suboptimal networks. Hence, we propose an additional finetuning training step: After conventional end-to-end training, parts of the network are retrained on quantized latents obtained at the inference stage. For entropy-constraint quantizers like Trellis-Coded Quantization, the impact of the quantizer is particularly difficult to approximate by rounding or adding noise as the quantized latents are interdependently chosen through a trellis search based on both the entropy model and a distortion measure. We show that retraining on correctly quantized data consistently yields additional coding gain for both uniform scalar and especially for entropy-constraint quantization, without increasing inference complexity. For the Kodak test set, we obtain average savings between 1% and 2%, and for the TecNick test set up to 2.2% in terms of Bjøntegaard-Delta bitrate. 

**Abstract (ZH)**: 基于变分自编码器的图像压缩连续改进已在率-失真效率方面使learned编解码器与传统方法竞争。然而，训练过程中考虑量化问题依然存在，因为量化过程几乎处处产生零导数，需要使用可微近似替代，从而实现端到端优化。尽管有多种方法近似量化过程，但 none 能够正确建模量化噪声，因此导致性能不佳的网络。因此，我们提出了一种额外的微调训练步骤：在传统端到端训练之后，重新训练网络的部分模块，使用推理阶段获得的量化潜在变量。对于基于熵约束的量化器（如梯形编码量化），量化器的影响通过舍入或添加噪声难以近似，因为量化潜在变量是基于熵模型和失真度量通过递归搜索相互选择的。我们表明，使用正确量化数据重新训练可以一致地为均匀标量量化和特别是熵约束量化提供额外的编码增益，且不增加推理复杂度。在 Kodak 测试集中，我们获得了平均 1% 到 2% 的比特率节省，而在 TecNick 测试集中，节省高达 2.2% 的 Bjøntegaard-Delta 比特率。 

---
# Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework for Dependency, Asynchrony, and Missingness 

**Title (ZH)**: 面向鲁棒的实际多变量时间序列预测：依赖性、非同步性和缺失性的一体化框架 

**Authors**: Jinkwan Jang, Hyungjin Park, Jinmyeong Choi, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.08660)  

**Abstract**: Real-world time series data are inherently multivariate, often exhibiting complex inter-channel dependencies. Each channel is typically sampled at its own period and is prone to missing values due to various practical and operational constraints. These characteristics pose fundamental challenges related to channel dependency, sampling asynchrony, and missingness, all of which must be addressed to enable robust and reliable forecasting in practical settings. However, most existing architectures are built on oversimplified assumptions, such as identical sampling periods across channels and fully observed inputs at test time, which often do not hold in real-world scenarios. To bridge this gap, we propose ChannelTokenFormer, a Transformer-based forecasting model with a flexible architecture designed to explicitly capture cross-channel interactions, accommodate channel-wise asynchronous sampling, and effectively handle missing values. Extensive experiments on three benchmark datasets modified to reflect practical settings, along with one real-world industrial dataset, demonstrate the superior robustness and accuracy of ChannelTokenFormer under challenging real-world conditions. 

**Abstract (ZH)**: 基于Transformer的ChannelTokenFormer：灵活捕捉跨通道交互的时空序列预测模型 

---
# JoFormer (Journey-based Transformer): Theory and Empirical Analysis on the Tiny Shakespeare Dataset 

**Title (ZH)**: 基于旅程的变换器：Tiny Shakespeare 数据集上的理论与实证分析 

**Authors**: Mahesh Godavarti  

**Link**: [PDF](https://arxiv.org/pdf/2506.08652)  

**Abstract**: Transformers have demonstrated remarkable success in sequence modeling, yet effectively incorporating positional information remains a challenging and active area of research. In this paper, we introduce JoFormer, a journey-based Transformer architecture grounded in a recently proposed non-commutative algebra for composing transformations across positions. JoFormer represents relative positions through learnable directional transforms that are sequentially composed along the input, thereby extending and generalizing existing approaches based on relative position representations. We derive the JoFormer attention mechanism from first principles and show that it subsumes standard methods such as rotary transformations as special cases. To evaluate its effectiveness, we compare JoFormer to the RoFormer baseline on the Tiny Shakespeare character-level language modeling task. Our results demonstrate that
JoFormer consistently achieves lower perplexity and faster convergence, highlighting the advantages of its more expressive, journey-based treatment of position. Notably, the per-token JoFormer is still a primitive, conceptual variant with layer-independent angles, yet it already demonstrates strong performance-underscoring its promise as a proof of concept for more expressive architectures. We conclude by discussing how JoFormer offers a principled approach to integrating positional structure into Transformer architectures. The code used in this work is available at this https URL. 

**Abstract (ZH)**: 基于旅程的变换器架构JoFormer及其在序列建模中的应用 

---
# Summarization for Generative Relation Extraction in the Microbiome Domain 

**Title (ZH)**: 微生物组领域生成性关系抽取的总结方法 

**Authors**: Oumaima El Khettari, Solen Quiniou, Samuel Chaffron  

**Link**: [PDF](https://arxiv.org/pdf/2506.08647)  

**Abstract**: We explore a generative relation extraction (RE) pipeline tailored to the study of interactions in the intestinal microbiome, a complex and low-resource biomedical domain. Our method leverages summarization with large language models (LLMs) to refine context before extracting relations via instruction-tuned generation. Preliminary results on a dedicated corpus show that summarization improves generative RE performance by reducing noise and guiding the model. However, BERT-based RE approaches still outperform generative models. This ongoing work demonstrates the potential of generative methods to support the study of specialized domains in low-resources setting. 

**Abstract (ZH)**: 一种用于肠道微生物组交互研究的生成性关系提取管道探索：利用大规模语言模型总结以提升低资源生物医学领域的关系提取性能 

---
# TableDreamer: Progressive and Weakness-guided Data Synthesis from Scratch for Table Instruction Tuning 

**Title (ZH)**: TableDreamer: 从头开始的渐进式和弱点导向的数据合成方法用于表格指令调优 

**Authors**: Mingyu Zheng, Zhifan Feng, Jia Wang, Lanrui Wang, Zheng Lin, Yang Hao, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08646)  

**Abstract**: Despite the commendable progress of recent LLM-based data synthesis methods, they face two limitations in generating table instruction tuning data. First, they can not thoroughly explore the vast input space of table understanding tasks, leading to limited data diversity. Second, they ignore the weaknesses in table understanding ability of the target LLM and blindly pursue the increase of data quantity, resulting in suboptimal data efficiency. In this paper, we introduce a progressive and weakness-guided data synthesis framework tailored for table instruction tuning, named TableDreamer, to mitigate the above issues. Specifically, we first synthesize diverse tables and related instructions as seed data, and then perform an iterative exploration of the input space under the guidance of the newly identified weakness data, which eventually serve as the final training data for fine-tuning the target LLM. Extensive experiments on 10 tabular benchmarks demonstrate the effectiveness of the proposed framework, which boosts the average accuracy of Llama3.1-8B-instruct by 11.62% (49.07% to 60.69%) with 27K GPT-4o synthetic data and outperforms state-of-the-art data synthesis baselines which use more training data. The code and data is available at this https URL 

**Abstract (ZH)**: 尽管基于LLM的数据合成方法在近期已经取得了显著的进步，但仍面临生成表格指令调优数据的两个局限性。首先，它们无法充分探索表格理解任务的广阔输入空间，导致数据多样性有限。其次，它们忽略了目标LLM在表格理解能力上的弱点，盲目追求数据量的增加，导致数据效率不足。本文提出了一种渐进式和弱点导向的数据合成框架TableDreamer，以解决上述问题。具体而言，我们首先合成多样化的表格及其相关指令作为种子数据，然后在已识别的新弱点数据引导下逐步探索输入空间，最终作为目标LLM微调的最终训练数据。在10个表格基准上的广泛实验表明，所提出框架的有效性，使用27K GPT-4o合成数据将Llama3.1-8B-instruct的平均准确率提升了11.62%（从49.07%提升到60.69%），并优于使用更多训练数据的最新数据合成基准。相关代码和数据可在以下链接获取。 

---
# Time Series Representations for Classification Lie Hidden in Pretrained Vision Transformers 

**Title (ZH)**: 预训练视觉变换器中隐含的时间序列表示用于分类 

**Authors**: Simon Roschmann, Quentin Bouniot, Vasilii Feofanov, Ievgen Redko, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2506.08641)  

**Abstract**: Time series classification is a fundamental task in healthcare and industry, yet the development of time series foundation models (TSFMs) remains limited by the scarcity of publicly available time series datasets. In this work, we propose Time Vision Transformer (TiViT), a framework that converts time series into images to leverage the representational power of frozen Vision Transformers (ViTs) pretrained on large-scale image datasets. First, we theoretically motivate our approach by analyzing the 2D patching of ViTs for time series, showing that it can increase the number of label-relevant tokens and reduce the sample complexity. Second, we empirically demonstrate that TiViT achieves state-of-the-art performance on standard time series classification benchmarks by utilizing the hidden representations of large OpenCLIP models. We explore the structure of TiViT representations and find that intermediate layers with high intrinsic dimension are the most effective for time series classification. Finally, we assess the alignment between TiViT and TSFM representation spaces and identify a strong complementarity, with further performance gains achieved by combining their features. Our findings reveal yet another direction for reusing vision representations in a non-visual domain. 

**Abstract (ZH)**: 时间序列分类是医疗保健和工业中的一个基础任务，但由于公共时间序列数据集的稀缺性，时间序列基础模型（TSFMs）的发展仍然受到限制。在本文中，我们提出了一种名为Time Vision Transformer（TiViT）的框架，该框架将时间序列转换为图像，以利用在大规模图像数据集上预训练的冻结视觉变换器（ViTs）的表征能力。首先，我们从理论上通过分析ViTs的时间序列2D分块方法来阐释我们的方法，表明它可以增加与标签相关的标记的数量并降低样本复杂度。其次，我们通过利用大型OpenCLIP模型的隐藏表示，在标准的时间序列分类基准上展现了TiViT达到最先进的性能。我们探讨了TiViT表示结构，发现具有较高固有维度的中间层对时间序列分类最为有效。最后，我们在TiViT和TSFM表示空间之间进行了对齐评估，并发现两者之间存在强烈的互补性，结合它们的特征可以进一步提升性能。我们的研究结果揭示了在非视觉领域重新利用视觉表示的另一个方向。 

---
# MOSAIC-F: A Framework for Enhancing Students' Oral Presentation Skills through Personalized Feedback 

**Title (ZH)**: MOSAIC-F：一种通过个性化反馈提升学生口头presentation技能的框架 

**Authors**: Alvaro Becerra, Daniel Andres, Pablo Villegas, Roberto Daza, Ruth Cobos  

**Link**: [PDF](https://arxiv.org/pdf/2506.08634)  

**Abstract**: In this article, we present a novel multimodal feedback framework called MOSAIC-F, an acronym for a data-driven Framework that integrates Multimodal Learning Analytics (MMLA), Observations, Sensors, Artificial Intelligence (AI), and Collaborative assessments for generating personalized feedback on student learning activities. This framework consists of four key steps. First, peers and professors' assessments are conducted through standardized rubrics (that include both quantitative and qualitative evaluations). Second, multimodal data are collected during learning activities, including video recordings, audio capture, gaze tracking, physiological signals (heart rate, motion data), and behavioral interactions. Third, personalized feedback is generated using AI, synthesizing human-based evaluations and data-based multimodal insights such as posture, speech patterns, stress levels, and cognitive load, among others. Finally, students review their own performance through video recordings and engage in self-assessment and feedback visualization, comparing their own evaluations with peers and professors' assessments, class averages, and AI-generated recommendations. By combining human-based and data-based evaluation techniques, this framework enables more accurate, personalized and actionable feedback. We tested MOSAIC-F in the context of improving oral presentation skills. 

**Abstract (ZH)**: 一种名为MOSAIC-F的多模态反馈框架：一种结合多模态学习分析、观察、传感器、人工智能和协作评估以生成个性化学习活动反馈的数据驱动框架 

---
# ECMNet:Lightweight Semantic Segmentation with Efficient CNN-Mamba Network 

**Title (ZH)**: ECMNet：高效CNN-Mamba网络的轻量级语义分割 

**Authors**: Feixiang Du, Shengkun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08629)  

**Abstract**: In the past decade, Convolutional Neural Networks (CNNs) and Transformers have achieved wide applicaiton in semantic segmentation tasks. Although CNNs with Transformer models greatly improve performance, the global context modeling remains inadequate. Recently, Mamba achieved great potential in vision tasks, showing its advantages in modeling long-range dependency. In this paper, we propose a lightweight Efficient CNN-Mamba Network for semantic segmentation, dubbed as ECMNet. ECMNet combines CNN with Mamba skillfully in a capsule-based framework to address their complementary weaknesses. Specifically, We design a Enhanced Dual-Attention Block (EDAB) for lightweight bottleneck. In order to improve the representations ability of feature, We devise a Multi-Scale Attention Unit (MSAU) to integrate multi-scale feature aggregation, spatial aggregation and channel aggregation. Moreover, a Mamba enhanced Feature Fusion Module (FFM) merges diverse level feature, significantly enhancing segmented accuracy. Extensive experiments on two representative datasets demonstrate that the proposed model excels in accuracy and efficiency balance, achieving 70.6% mIoU on Cityscapes and 73.6% mIoU on CamVid test datasets, with 0.87M parameters and 8.27G FLOPs on a single RTX 3090 GPU platform. 

**Abstract (ZH)**: 基于Mamba的高效CNN网络及其在语义分割中的应用：ECMNet 

---
# HSG-12M: A Large-Scale Spatial Multigraph Dataset 

**Title (ZH)**: HSG-12M: 一个大规模空域多图数据集 

**Authors**: Xianquan Yan, Hakan Akgün, Kenji Kawaguchi, N. Duane Loh, Ching Hua Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.08618)  

**Abstract**: Existing graph benchmarks assume non-spatial, simple edges, collapsing physically distinct paths into a single link. We introduce HSG-12M, the first large-scale dataset of $\textbf{spatial multigraphs}-$graphs embedded in a metric space where multiple geometrically distinct trajectories between two nodes are retained as separate edges. HSG-12M contains 11.6 million static and 5.1 million dynamic $\textit{Hamiltonian spectral graphs}$ across 1401 characteristic-polynomial classes, derived from 177 TB of spectral potential data. Each graph encodes the full geometry of a 1-D crystal's energy spectrum on the complex plane, producing diverse, physics-grounded topologies that transcend conventional node-coordinate datasets. To enable future extensions, we release $\texttt{Poly2Graph}$: a high-performance, open-source pipeline that maps arbitrary 1-D crystal Hamiltonians to spectral graphs. Benchmarks with popular GNNs expose new challenges in learning from multi-edge geometry at scale. Beyond its practical utility, we show that spectral graphs serve as universal topological fingerprints of polynomials, vectors, and matrices, forging a new algebra-to-graph link. HSG-12M lays the groundwork for geometry-aware graph learning and new opportunities of data-driven scientific discovery in condensed matter physics and beyond. 

**Abstract (ZH)**: 现有的图基准假设非空间的简单边，将物理上不同的路径Collapse为单一链接。我们引入HSG-12M，这是第一个大规模的空间多图数据集——这些图嵌在度量空间中，保留了两个节点之间多条几何上不同的轨迹作为单独的边。HSG-12M包含1160万静态和510万动态哈密尔顿谱图，跨越1401个特征多项式类，源自177TB的谱潜力数据。每个图编码了一维晶体能量谱在复平面上的完整几何结构，产生多样且基于物理的拓扑结构，超越了传统的节点坐标数据集。为了便于未来扩展，我们发布了Poly2Graph：一个高性能的开源管道，将任意一维晶体哈密尔顿量映射到谱图。流行的GNN基准揭示了大规模学习多边几何的新挑战。除了其实用价值外，我们展示谱图作为多项式、向量和矩阵的通用拓扑指纹，建立了一种新的代数到图的新联系。HSG-12M为几何感知的图学习和凝聚态物理等领域驱动数据科学的新机遇奠定了基础。 

---
# Flow Matching Meets PDEs: A Unified Framework for Physics-Constrained Generation 

**Title (ZH)**: 流匹配与偏微分方程：一种物理约束生成的统一框架 

**Authors**: Giacomo Baldan, Qiang Liu, Alberto Guardone, Nils Thuerey  

**Link**: [PDF](https://arxiv.org/pdf/2506.08604)  

**Abstract**: Generative machine learning methods, such as diffusion models and flow matching, have shown great potential in modeling complex system behaviors and building efficient surrogate models. However, these methods typically learn the underlying physics implicitly from data. We propose Physics-Based Flow Matching (PBFM), a novel generative framework that explicitly embeds physical constraints, both PDE residuals and algebraic relations, into the flow matching objective. We also introduce temporal unrolling at training time that improves the accuracy of the final, noise-free sample prediction. Our method jointly minimizes the flow matching loss and the physics-based residual loss without requiring hyperparameter tuning of their relative weights. Additionally, we analyze the role of the minimum noise level, $\sigma_{\min}$, in the context of physical constraints and evaluate a stochastic sampling strategy that helps to reduce physical residuals. Through extensive benchmarks on three representative PDE problems, we show that our approach yields up to an $8\times$ more accurate physical residuals compared to FM, while clearly outperforming existing algorithms in terms of distributional accuracy. PBFM thus provides a principled and efficient framework for surrogate modeling, uncertainty quantification, and accelerated simulation in physics and engineering applications. 

**Abstract (ZH)**: 基于物理的流匹配生成模型（PBFM）：一种嵌入物理约束的高效生成框架 

---
# WGLE:Backdoor-free and Multi-bit Black-box Watermarking for Graph Neural Networks 

**Title (ZH)**: WGLE：无后门多比特黑盒图神经网络水印 

**Authors**: Tingzhi Li, Xuefeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08602)  

**Abstract**: Graph Neural Networks (GNNs) are increasingly deployed in graph-related applications, making ownership verification critical to protect their intellectual property against model theft. Fingerprinting and black-box watermarking are two main methods. However, the former relies on determining model similarity, which is computationally expensive and prone to ownership collisions after model post-processing such as model pruning or fine-tuning. The latter embeds backdoors, exposing watermarked models to the risk of backdoor attacks. Moreover, both methods enable ownership verification but do not convey additional information. As a result, each distributed model requires a unique trigger graph, and all trigger graphs must be used to query the suspect model during verification. Multiple queries increase the financial cost and the risk of detection.
To address these challenges, this paper proposes WGLE, a novel black-box watermarking paradigm for GNNs that enables embedding the multi-bit string as the ownership information without using backdoors. WGLE builds on a key insight we term Layer-wise Distance Difference on an Edge (LDDE), which quantifies the difference between the feature distance and the prediction distance of two connected nodes. By predefining positive or negative LDDE values for multiple selected edges, WGLE embeds the watermark encoding the intended information without introducing incorrect mappings that compromise the primary task. WGLE is evaluated on six public datasets and six mainstream GNN architectures along with state-of-the-art methods. The results show that WGLE achieves 100% ownership verification accuracy, an average fidelity degradation of 0.85%, comparable robustness against potential attacks, and low embedding overhead. The code is available in the repository. 

**Abstract (ZH)**: 隐水印方法WGLE：面向图神经网络的无后门黑盒水印范式 

---
# Transformers Meet Hyperspectral Imaging: A Comprehensive Study of Models, Challenges and Open Problems 

**Title (ZH)**: 变压器与高光谱成像结合：模型、挑战与开放问题的研究综述 

**Authors**: Guyang Zhang, Waleed Abdulla  

**Link**: [PDF](https://arxiv.org/pdf/2506.08596)  

**Abstract**: Transformers have become the architecture of choice for learning long-range dependencies, yet their adoption in hyperspectral imaging (HSI) is still emerging. We reviewed more than 300 papers published up to 2025 and present the first end-to-end survey dedicated to Transformer-based HSI classification. The study categorizes every stage of a typical pipeline-pre-processing, patch or pixel tokenization, positional encoding, spatial-spectral feature extraction, multi-head self-attention variants, skip connections, and loss design-and contrasts alternative design choices with the unique spatial-spectral properties of HSI. We map the field's progress against persistent obstacles: scarce labeled data, extreme spectral dimensionality, computational overhead, and limited model explainability. Finally, we outline a research agenda prioritizing valuable public data sets, lightweight on-edge models, illumination and sensor shifts robustness, and intrinsically interpretable attention mechanisms. Our goal is to guide researchers in selecting, combining, or extending Transformer components that are truly fit for purpose for next-generation HSI applications. 

**Abstract (ZH)**: Transformer架构在长距离依赖学习中的应用已成为首选，但在高光谱成像(HSI)中的采用仍处于起步阶段。我们回顾了截至2025年发表的超过300篇论文，并提供了首篇专注于Transformerベース的HSI分类的端到端综述。研究将典型管道中的每一个阶段分类——预处理、patch或像素 token化、位置编码、空间光谱特征提取、多头自注意力变体、跳跃连接和损失设计——并与HSI的独特空间光谱性质对比了各种设计选择。我们映射了该领域在持续挑战下的进展：稀缺的标注数据、极端的光谱维度、计算开销以及模型解释性有限。最后，我们勾勒出研究议程，优先考虑有价值的数据集、轻量级边缘模型、抗照明和传感器偏移、以及固有的可解释注意力机制。我们的目标是指导研究人员选择、组合或扩展真正适合新一代HSI应用目的的Transformer组件。 

---
# Solving excited states for long-range interacting trapped ions with neural networks 

**Title (ZH)**: 用神经网络解决长程相互作用囚禁离子的激发态问题 

**Authors**: Yixuan Ma, Chang Liu, Weikang Li, Shun-Yao Zhang, L.-M. Duan, Yukai Wu, Dong-Ling Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.08594)  

**Abstract**: The computation of excited states in strongly interacting quantum many-body systems is of fundamental importance. Yet, it is notoriously challenging due to the exponential scaling of the Hilbert space dimension with the system size. Here, we introduce a neural network-based algorithm that can simultaneously output multiple low-lying excited states of a quantum many-body spin system in an accurate and efficient fashion. This algorithm, dubbed the neural quantum excited-state (NQES) algorithm, requires no explicit orthogonalization of the states and is generally applicable to higher dimensions. We demonstrate, through concrete examples including the Haldane-Shastry model with all-to-all interactions, that the NQES algorithm is capable of efficiently computing multiple excited states and their related observable expectations. In addition, we apply the NQES algorithm to two classes of long-range interacting trapped-ion systems in a two-dimensional Wigner crystal. For non-decaying all-to-all interactions with alternating signs, our computed low-lying excited states bear spatial correlation patterns similar to those of the ground states, which closely match recent experimental observations that the quasi-adiabatically prepared state accurately reproduces analytical ground-state correlations. For a system of up to 300 ions with power-law decaying antiferromagnetic interactions, we successfully uncover its gap scaling and correlation features. Our results establish a scalable and efficient algorithm for computing excited states of interacting quantum many-body systems, which holds potential applications ranging from benchmarking quantum devices to photoisomerization. 

**Abstract (ZH)**: 基于神经网络的强相互作用量子多体系统激发态计算算法 

---
# Diffusion-based Time Series Forecasting for Sewerage Systems 

**Title (ZH)**: 基于扩散的时间序列预测方法在污水处理系统中的应用 

**Authors**: Nicholas A. Pearson, Francesca Cairoli, Luca Bortolussi, Davide Russo, Francesca Zanello  

**Link**: [PDF](https://arxiv.org/pdf/2506.08577)  

**Abstract**: We introduce a novel deep learning approach that harnesses the power of generative artificial intelligence to enhance the accuracy of contextual forecasting in sewerage systems. By developing a diffusion-based model that processes multivariate time series data, our system excels at capturing complex correlations across diverse environmental signals, enabling robust predictions even during extreme weather events. To strengthen the model's reliability, we further calibrate its predictions with a conformal inference technique, tailored for probabilistic time series data, ensuring that the resulting prediction intervals are statistically reliable and cover the true target values with a desired confidence level. Our empirical tests on real sewerage system data confirm the model's exceptional capability to deliver reliable contextual predictions, maintaining accuracy even under severe weather conditions. 

**Abstract (ZH)**: 我们提出一种新颖的深度学习方法，利用生成人工智能的力量来增强污水系统中上下文预测的准确性。通过开发一种基于扩散的模型来处理多变量时间序列数据，我们的系统能够捕捉复杂多样的环境信号之间的关联，即使在极端天气事件期间也能提供稳健的预测。为进一步提高模型的可靠性，我们使用一种针对概率时间序列数据的 conforms 推断技术来校准其预测，确保预测区间在统计上是可靠的，并以所需的置信水平覆盖真实的目标值。实证研究表明，该模型在严重天气条件下仍能提供可靠的上下文预测，保持较高的准确性。 

---
# The Geometries of Truth Are Orthogonal Across Tasks 

**Title (ZH)**: 真理的几何学在不同任务之间是正交的 

**Authors**: Waiss Azizian, Michael Kirchhof, Eugene Ndiaye, Louis Bethune, Michal Klein, Pierre Ablin, Marco Cuturi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08572)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive generalization capabilities across various tasks, but their claim to practical relevance is still mired by concerns on their reliability. Recent works have proposed examining the activations produced by an LLM at inference time to assess whether its answer to a question is correct. Some works claim that a "geometry of truth" can be learned from examples, in the sense that the activations that generate correct answers can be distinguished from those leading to mistakes with a linear classifier. In this work, we underline a limitation of these approaches: we observe that these "geometries of truth" are intrinsically task-dependent and fail to transfer across tasks. More precisely, we show that linear classifiers trained across distinct tasks share little similarity and, when trained with sparsity-enforcing regularizers, have almost disjoint supports. We show that more sophisticated approaches (e.g., using mixtures of probes and tasks) fail to overcome this limitation, likely because activation vectors commonly used to classify answers form clearly separated clusters when examined across tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务上展现了惊人的泛化能力，但其实际相关性仍然受到可靠性的质疑。近期工作提出，在推理时检查LLM生成的激活可以帮助评估其答案的正确性。一些工作声称可以从示例中学到“真理的几何结构”，意味着生成正确答案的激活可以与导致错误的激活通过线性分类器区分开。在本工作中，我们指出了这些方法的一个局限性：我们观察到这些“真理的几何结构”本质上是任务依赖的，并且无法在任务之间转移。具体来说，我们展示了跨不同任务训练的线性分类器在很大程度上缺乏相似性，在施加稀疏约束规则的情况下，几乎没有任何重叠的支持集。我们证明了使用探测器和任务混合等更复杂的方法也无法克服这一局限性，很可能是因为用于分类答案的激活向量在跨任务检查时形成了清晰分离的簇。 

---
# Auto-Regressive vs Flow-Matching: a Comparative Study of Modeling Paradigms for Text-to-Music Generation 

**Title (ZH)**: 自回归模型 vs 流匹配模型：文本到音乐生成建模范式的比较研究 

**Authors**: Or Tal, Felix Kreuk, Yossi Adi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08570)  

**Abstract**: Recent progress in text-to-music generation has enabled models to synthesize high-quality musical segments, full compositions, and even respond to fine-grained control signals, e.g. chord progressions. State-of-the-art (SOTA) systems differ significantly across many dimensions, such as training datasets, modeling paradigms, and architectural choices. This diversity complicates efforts to evaluate models fairly and pinpoint which design choices most influence performance. While factors like data and architecture are important, in this study we focus exclusively on the modeling paradigm. We conduct a systematic empirical analysis to isolate its effects, offering insights into associated trade-offs and emergent behaviors that can guide future text-to-music generation systems. Specifically, we compare the two arguably most common modeling paradigms: Auto-Regressive decoding and Conditional Flow-Matching. We conduct a controlled comparison by training all models from scratch using identical datasets, training configurations, and similar backbone architectures. Performance is evaluated across multiple axes, including generation quality, robustness to inference configurations, scalability, adherence to both textual and temporally aligned conditioning, and editing capabilities in the form of audio inpainting. This comparative study sheds light on distinct strengths and limitations of each paradigm, providing actionable insights that can inform future architectural and training decisions in the evolving landscape of text-to-music generation. Audio sampled examples are available at: this https URL 

**Abstract (ZH)**: Recent进展在文本到音乐生成领域的最新进展使模型能够合成高质量的音乐片段、完整的曲目，甚至响应细微的控制信号，例如和弦进行。当前最先进的系统在多个维度上存在显著差异，例如训练数据集、建模范式和架构选择。这种多样性使公平地评估模型和确定哪些设计选择对性能影响最大变得复杂。虽然数据和架构因素很重要，但在本研究中，我们仅集中在建模范式上。我们进行了一种系统的实证分析，以分离其影响，提供了关于相关权衡和新兴行为的见解，这些见解可以指导未来文本到音乐生成系统的开发。具体而言，我们将比较两种公认的常见建模范式：自回归解码和条件流匹配。我们通过使用相同的训练数据集、训练配置和类似的基础架构进行零样本训练，进行受控比较。性能在多个维度上进行评估，包括生成质量、推理配置的鲁棒性、可扩展性、对文本和时间对齐条件的遵循以及以音频填补形式的编辑能力。这种比较研究揭示了每种范式的独特优势和局限性，提供了可以指导未来架构和训练决策的可操作见解。音频示例样本请参阅：this https URL。 

---
# Flow-Lenia: Emergent evolutionary dynamics in mass conservative continuous cellular automata 

**Title (ZH)**: 流体-Lenia：守恒连续细胞自动机中的 emergent 进化动力学 

**Authors**: Erwan Plantec, Gautier Hamon, Mayalen Etcheverry, Bert Wang-Chak Chan, Pierre-Yves Oudeyer, Clément Moulin-Frier  

**Link**: [PDF](https://arxiv.org/pdf/2506.08569)  

**Abstract**: Central to the artificial life endeavour is the creation of artificial systems spontaneously generating properties found in the living world such as autopoiesis, self-replication, evolution and open-endedness. While numerous models and paradigms have been proposed, cellular automata (CA) have taken a very important place in the field notably as they enable the study of phenomenons like self-reproduction and autopoiesis. Continuous CA like Lenia have been showed to produce life-like patterns reminiscent, on an aesthetic and ontological point of view, of biological organisms we call creatures. We propose in this paper Flow-Lenia, a mass conservative extension of Lenia. We present experiments demonstrating its effectiveness in generating spatially-localized patters (SLPs) with complex behaviors and show that the update rule parameters can be optimized to generate complex creatures showing behaviors of interest. Furthermore, we show that Flow-Lenia allows us to embed the parameters of the model, defining the properties of the emerging patterns, within its own dynamics thus allowing for multispecies simulations. By using the evolutionary activity framework as well as other metrics, we shed light on the emergent evolutionary dynamics taking place in this system. 

**Abstract (ZH)**: 基于人工生命的追求，创建能够自发产生类似于生物界特征（如自主生成、自我复制、进化和开放性）的人工系统是核心内容。尽管已经提出了众多模型和范式，细胞自动机（CA）在这一领域中占据了非常重要的位置，特别是它们能够研究类似于自复制和自主生成的现象。连续细胞自动机（如Lenia）已被证明能够生成类似于生物有机体的生物样模式，在美学和本体论意义上尤为如此。本文提出Flow-Lenia，这是一种保持质量守恒的Lenia扩展。我们展示了其在生成具有复杂行为的空间局域模式（SLPs）方面的有效性，并证明可以通过优化更新规则参数来生成具有特定行为的复杂生物。此外，我们展示了Flow-Lenia使得能够将模型参数嵌入其中，从而定义出现模式的属性，进而实现多物种模拟。通过使用进化的活动框架以及其他指标，我们揭示了该系统中发生的涌现性进化动力学。 

---
# KP-PINNs: Kernel Packet Accelerated Physics Informed Neural Networks 

**Title (ZH)**: KP-PINNs: 核包络加速物理信息神经网络 

**Authors**: Siyuan Yang, Cheng Song, Zhilu Lai, Wenjia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08563)  

**Abstract**: Differential equations are involved in modeling many engineering problems. Many efforts have been devoted to solving differential equations. Due to the flexibility of neural networks, Physics Informed Neural Networks (PINNs) have recently been proposed to solve complex differential equations and have demonstrated superior performance in many applications. While the L2 loss function is usually a default choice in PINNs, it has been shown that the corresponding numerical solution is incorrect and unstable for some complex equations. In this work, we propose a new PINNs framework named Kernel Packet accelerated PINNs (KP-PINNs), which gives a new expression of the loss function using the reproducing kernel Hilbert space (RKHS) norm and uses the Kernel Packet (KP) method to accelerate the computation. Theoretical results show that KP-PINNs can be stable across various differential equations. Numerical experiments illustrate that KP-PINNs can solve differential equations effectively and efficiently. This framework provides a promising direction for improving the stability and accuracy of PINNs-based solvers in scientific computing. 

**Abstract (ZH)**: 基于核包络的物理约束神经网络（KP-PINNs）：一种新的损失函数表达及加速计算框架 

---
# Efficient Post-Training Refinement of Latent Reasoning in Large Language Models 

**Title (ZH)**: 大型语言模型训练后潜藏推理能力的高效精炼方法 

**Authors**: Xinyuan Wang, Dongjie Wang, Wangyang Ying, Haoyue Bai, Nanxu Gong, Sixun Dong, Kunpeng Liu, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08552)  

**Abstract**: Reasoning is a key component of language understanding in Large Language Models. While Chain-of-Thought prompting enhances performance via explicit intermediate steps, it suffers from sufficient token overhead and a fixed reasoning trajectory, preventing step-wise refinement. Recent advances in latent reasoning address these limitations by refining internal reasoning processes directly in the model's latent space, without producing explicit outputs. However, a key challenge remains: how to effectively update reasoning embeddings during post-training to guide the model toward more accurate solutions. To overcome this challenge, we propose a lightweight post-training framework that refines latent reasoning trajectories using two novel strategies: 1) Contrastive reasoning feedback, which compares reasoning embeddings against strong and weak baselines to infer effective update directions via embedding enhancement; 2) Residual embedding refinement, which stabilizes updates by progressively integrating current and historical gradients, enabling fast yet controlled convergence. Extensive experiments and case studies are conducted on five reasoning benchmarks to demonstrate the effectiveness of the proposed framework. Notably, a 5\% accuracy gain on MathQA without additional training. 

**Abstract (ZH)**: 大型语言模型中的语言理解关键在于推理。尽管基于推理链的提示通过显式中间步骤增强性能，但会产生足够的标记开销并具有固定的推理轨迹，限制了逐步细化的能力。最近在潜在线索推理方面的进步通过直接在模型的潜空间中细化内部推理过程来克服这些限制，无需生成显式输出。然而，仍存在一个关键挑战：如何在后训练期间有效更新推理嵌入以引导模型向更准确的解决方案发展。为克服这一挑战，我们提出了一种轻量级后训练框架，通过两种新颖策略细化潜在线索推理轨迹：1) 对比推理反馈，将推理嵌入与强弱基准进行比较，通过嵌入增强推断有效更新方向；2) 余嵌入细化，通过逐步整合当前和历史梯度来稳定更新，实现快速且可控的收敛。在五个推理基准上的广泛实验和案例研究证明了提出框架的有效性。特别地，在MathQA上未进行额外训练的情况下获得5%的准确率提升。 

---
# TrajFlow: Multi-modal Motion Prediction via Flow Matching 

**Title (ZH)**: TrajFlow: 通过流匹配的多模态运动预测 

**Authors**: Qi Yan, Brian Zhang, Yutong Zhang, Daniel Yang, Joshua White, Di Chen, Jiachao Liu, Langechuan Liu, Binnan Zhuang, Shaoshuai Shi, Renjie Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.08541)  

**Abstract**: Efficient and accurate motion prediction is crucial for ensuring safety and informed decision-making in autonomous driving, particularly under dynamic real-world conditions that necessitate multi-modal forecasts. We introduce TrajFlow, a novel flow matching-based motion prediction framework that addresses the scalability and efficiency challenges of existing generative trajectory prediction methods. Unlike conventional generative approaches that employ i.i.d. sampling and require multiple inference passes to capture diverse outcomes, TrajFlow predicts multiple plausible future trajectories in a single pass, significantly reducing computational overhead while maintaining coherence across predictions. Moreover, we propose a ranking loss based on the Plackett-Luce distribution to improve uncertainty estimation of predicted trajectories. Additionally, we design a self-conditioning training technique that reuses the model's own predictions to construct noisy inputs during a second forward pass, thereby improving generalization and accelerating inference. Extensive experiments on the large-scale Waymo Open Motion Dataset (WOMD) demonstrate that TrajFlow achieves state-of-the-art performance across various key metrics, underscoring its effectiveness for safety-critical autonomous driving applications. The code and other details are available on the project website this https URL. 

**Abstract (ZH)**: 基于流匹配的高效准确运动预测框架TrajFlow在自动驾驶中的应用 

---
# DCD: A Semantic Segmentation Model for Fetal Ultrasound Four-Chamber View 

**Title (ZH)**: DCD：胎儿超声四腔观的语义分割模型 

**Authors**: Donglian Li, Hui Guo, Minglang Chen, Huizhen Chen, Jialing Chen, Bocheng Liang, Pengchen Liang, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08534)  

**Abstract**: Accurate segmentation of anatomical structures in the apical four-chamber (A4C) view of fetal echocardiography is essential for early diagnosis and prenatal evaluation of congenital heart disease (CHD). However, precise segmentation remains challenging due to ultrasound artifacts, speckle noise, anatomical variability, and boundary ambiguity across different gestational stages. To reduce the workload of sonographers and enhance segmentation accuracy, we propose DCD, an advanced deep learning-based model for automatic segmentation of key anatomical structures in the fetal A4C view. Our model incorporates a Dense Atrous Spatial Pyramid Pooling (Dense ASPP) module, enabling superior multi-scale feature extraction, and a Convolutional Block Attention Module (CBAM) to enhance adaptive feature representation. By effectively capturing both local and global contextual information, DCD achieves precise and robust segmentation, contributing to improved prenatal cardiac assessment. 

**Abstract (ZH)**: 胎儿四腔观(A4C)心超中解剖结构的准确分割对于先天性心脏疾病(CHD)的早期诊断和产前评估至关重要。然而，由于超声伪像、speckle噪声、解剖结构的变异性和不同妊娠阶段边界模糊性，精确分割仍具有挑战性。为减轻超声操作者的负担并提高分割准确性，我们提出了一种基于深度学习的DCD先进模型，用于自动分割胎儿四腔观的关键解剖结构。该模型结合了 Dense Atrous Spatial Pyramid Pooling (Dense ASPP) 模块，实现了优异的多尺度特征提取，并使用 Convolutional Block Attention Module (CBAM) 来增强自适应特征表示。通过有效捕捉局部和全局上下文信息，DCD 实现了精确和鲁棒的分割，有助于提高产前心脏评估。 

---
# Robust Evolutionary Multi-Objective Network Architecture Search for Reinforcement Learning (EMNAS-RL) 

**Title (ZH)**: 鲁棒进化多目标网络架构搜索在强化学习中的应用（EMNAS-RL） 

**Authors**: Nihal Acharya Adde, Alexandra Gianzina, Hanno Gottschalk, Andreas Ebert  

**Link**: [PDF](https://arxiv.org/pdf/2506.08533)  

**Abstract**: This paper introduces Evolutionary Multi-Objective Network Architecture Search (EMNAS) for the first time to optimize neural network architectures in large-scale Reinforcement Learning (RL) for Autonomous Driving (AD). EMNAS uses genetic algorithms to automate network design, tailored to enhance rewards and reduce model size without compromising performance. Additionally, parallelization techniques are employed to accelerate the search, and teacher-student methodologies are implemented to ensure scalable optimization. This research underscores the potential of transfer learning as a robust framework for optimizing performance across iterative learning processes by effectively leveraging knowledge from earlier generations to enhance learning efficiency and stability in subsequent generations. Experimental results demonstrate that tailored EMNAS outperforms manually designed models, achieving higher rewards with fewer parameters. The findings of these strategies contribute positively to EMNAS for RL in autonomous driving, advancing the field toward better-performing networks suitable for real-world scenarios. 

**Abstract (ZH)**: 进化多目标网络架构搜索（EMNAS）在自主驾驶中的大规模强化学习优化 

---
# Teaching Physical Awareness to LLMs through Sounds 

**Title (ZH)**: 通过声音教授物理意识给大规模语言模型 

**Authors**: Weiguo Wang, Andy Nie, Wenrui Zhou, Yi Kai, Chengchen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08524)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in text and multimodal processing, yet they fundamentally lack physical awareness--understanding of real-world physical phenomena. In this work, we present ACORN, a framework that teaches LLMs physical awareness through sound, focusing on fundamental physical phenomena like the Doppler effect, multipath effect, and spatial relationships. To overcome data scarcity, ACORN introduce a physics-based simulator combining real-world sound sources with controlled physical channels to generate diverse training data. Using this simulator, we build AQA-PHY, a comprehensive Audio Question-Answer dataset, and propose an audio encoder that processes both magnitude and phase information. By connecting our audio encoder to state-of-the-art LLMs, we demonstrate reasonable results in both simulated and real-world tasks, such as line-of-sight detection, Doppler effect estimation, and Direction-of-Arrival estimation, paving the way for enabling LLMs to understand physical world. 

**Abstract (ZH)**: 大型语言模型（LLMs）在文本和多模态处理方面表现出色，但本质上缺乏物理感知——对现实世界物理现象的理解。本文介绍了一种名为ACORN的框架，通过声音来提升LLMs的物理感知能力，重点关注如多普勒效应、多路径效应和空间关系等基本物理现象。为克服数据稀缺问题，ACORN引入了一种基于物理的模拟器，该模拟器结合了真实世界的声音源和可控的物理通道，生成多样化训练数据。利用该模拟器，我们构建了AQA-PHY综合音频问答数据集，并提出了一种处理幅度和相位信息的音频编码器。通过将我们的音频编码器连接到最先进的LLMs，我们在仿真和真实世界任务中展示了合理的结果，如视线检测、多普勒效应估计和到达方向估计，为使LLMs理解物理世界铺平了道路。 

---
# MLVTG: Mamba-Based Feature Alignment and LLM-Driven Purification for Multi-Modal Video Temporal Grounding 

**Title (ZH)**: MLVTG: 基于Mamba的特征对齐与基于LLM的多模态视频 temporal 基准净化 

**Authors**: Zhiyi Zhu, Xiaoyu Wu, Zihao Liu, Linlin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08512)  

**Abstract**: Video Temporal Grounding (VTG), which aims to localize video clips corresponding to natural language queries, is a fundamental yet challenging task in video understanding. Existing Transformer-based methods often suffer from redundant attention and suboptimal multi-modal alignment. To address these limitations, we propose MLVTG, a novel framework that integrates two key modules: MambaAligner and LLMRefiner. MambaAligner uses stacked Vision Mamba blocks as a backbone instead of Transformers to model temporal dependencies and extract robust video representations for multi-modal alignment. LLMRefiner leverages the specific frozen layer of a pre-trained Large Language Model (LLM) to implicitly transfer semantic priors, enhancing multi-modal alignment without fine-tuning. This dual alignment strategy, temporal modeling via structured state-space dynamics and semantic purification via textual priors, enables more precise localization. Extensive experiments on QVHighlights, Charades-STA, and TVSum demonstrate that MLVTG achieves state-of-the-art performance and significantly outperforms existing baselines. 

**Abstract (ZH)**: 视频时间定位（VTG），旨在定位与自然语言查询对应的视频片段，是视频理解中一个基础但具有挑战性的任务。现有的基于Transformer的方法常常受到冗余注意力和次优化多模态对齐的问题。为了解决这些局限，我们提出了MLVTG，一种新颖的框架，整合了两个关键模块：MambaAligner和LLMRefiner。MambaAligner使用堆叠的Vision Mamba块作为骨干，而不是Transformer，以建模时间依赖关系并提取用于多模态对齐的稳健视频表示。LLMRefiner利用预训练大型语言模型（LLM）中特定冻结层来隐式传递语义先验，增强多模态对齐而无需微调。这种双重对齐策略，通过结构化状态空间动力学建模时间关系并通过文本先验净化语义，能够实现更精确的定位。在QVHighlights、Charades-STA和TVSum上的广泛实验表明，MLVTG达到最佳性能并显著优于现有基线。 

---
# MasHost Builds It All: Autonomous Multi-Agent System Directed by Reinforcement Learning 

**Title (ZH)**: MasHost 自建一切：基于强化学习的自主多Agent系统 

**Authors**: Kuo Yang, Xingjie Yang, Linhui Yu, Qing Xu, Yan Fang, Xu Wang, Zhengyang Zhou, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08507)  

**Abstract**: Large Language Model (LLM)-driven Multi-agent systems (Mas) have recently emerged as a powerful paradigm for tackling complex real-world tasks. However, existing Mas construction methods typically rely on manually crafted interaction mechanisms or heuristic rules, introducing human biases and constraining the autonomous ability. Even with recent advances in adaptive Mas construction, existing systems largely remain within the paradigm of semi-autonomous patterns. In this work, we propose MasHost, a Reinforcement Learning (RL)-based framework for autonomous and query-adaptive Mas design. By formulating Mas construction as a graph search problem, our proposed MasHost jointly samples agent roles and their interactions through a unified probabilistic sampling mechanism. Beyond the accuracy and efficiency objectives pursued in prior works, we introduce component rationality as an additional and novel design principle in Mas. To achieve this multi-objective optimization, we propose Hierarchical Relative Policy Optimization (HRPO), a novel RL strategy that collaboratively integrates group-relative advantages and action-wise rewards. To our knowledge, our proposed MasHost is the first RL-driven framework for autonomous Mas graph construction. Extensive experiments on six benchmarks demonstrate that MasHost consistently outperforms most competitive baselines, validating its effectiveness, efficiency, and structure rationality. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的多智能体系统（MAS）驱动的自主和查询自适应设计框架 

---
# Explaining, Fast and Slow: Abstraction and Refinement of Provable Explanations 

**Title (ZH)**: 快与慢的解释：可验证解释的抽象与细化 

**Authors**: Shahaf Bassan, Yizhak Yisrael Elboher, Tobias Ladner, Matthias Althoff, Guy Katz  

**Link**: [PDF](https://arxiv.org/pdf/2506.08505)  

**Abstract**: Despite significant advancements in post-hoc explainability techniques for neural networks, many current methods rely on heuristics and do not provide formally provable guarantees over the explanations provided. Recent work has shown that it is possible to obtain explanations with formal guarantees by identifying subsets of input features that are sufficient to determine that predictions remain unchanged using neural network verification techniques. Despite the appeal of these explanations, their computation faces significant scalability challenges. In this work, we address this gap by proposing a novel abstraction-refinement technique for efficiently computing provably sufficient explanations of neural network predictions. Our method abstracts the original large neural network by constructing a substantially reduced network, where a sufficient explanation of the reduced network is also provably sufficient for the original network, hence significantly speeding up the verification process. If the explanation is in sufficient on the reduced network, we iteratively refine the network size by gradually increasing it until convergence. Our experiments demonstrate that our approach enhances the efficiency of obtaining provably sufficient explanations for neural network predictions while additionally providing a fine-grained interpretation of the network's predictions across different abstraction levels. 

**Abstract (ZH)**: 尽管在神经网络的后验解释技术方面取得了显著进展，但当前许多方法仍然依赖于启发式方法，并不能提供形式上可证明的解释保证。最近的研究表明，通过使用神经网络验证技术识别出输入特征的子集，可以使预测保持不变，从而能够获得具有形式上可证明保证的解释。尽管这些解释具有吸引力，但其计算面临显著的可扩展性挑战。在本文中，我们通过提出一种新的抽象化-细化技术来解决这一问题，该技术可以高效地计算神经网络预测的形式上可证明充分的解释。我们的方法通过构建一个大幅减少的网络来抽象原始的大规模神经网络，其中减少网络的充分解释也是原始网络的充分解释，从而显著加快验证过程。如果解释在减少网络上是充分的，我们将通过逐步增加网络规模进行迭代细化，直到收敛。我们的实验表明，我们的方法不仅提高了获得神经网络预测形式上可证明充分解释的效率，还通过不同抽象层级提供了网络预测的细致解释。 

---
# CoMuMDR: Code-mixed Multi-modal Multi-domain corpus for Discourse paRsing in conversations 

**Title (ZH)**: CoMuMDR：代码混合多模态多领域语料库及其在对话话语解析中的应用 

**Authors**: Divyaksh Shukla, Ritesh Baviskar, Dwijesh Gohil, Aniket Tiwari, Atul Shree, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08504)  

**Abstract**: Discourse parsing is an important task useful for NLU applications such as summarization, machine comprehension, and emotion recognition. The current discourse parsing datasets based on conversations consists of written English dialogues restricted to a single domain. In this resource paper, we introduce CoMuMDR: Code-mixed Multi-modal Multi-domain corpus for Discourse paRsing in conversations. The corpus (code-mixed in Hindi and English) has both audio and transcribed text and is annotated with nine discourse relations. We experiment with various SoTA baseline models; the poor performance of SoTA models highlights the challenges of multi-domain code-mixed corpus, pointing towards the need for developing better models for such realistic settings. 

**Abstract (ZH)**: Code-mixed Multi-modal Multi-domain corpus for Discourse Parsing in Conversations 

---
# DRAGged into Conflicts: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs 

**Title (ZH)**: 拖入冲突：检测和解决搜索增强的大语言模型中的矛盾来源 

**Authors**: Arie Cattan, Alon Jacovi, Ori Ram, Jonathan Herzig, Roee Aharoni, Sasha Goldshtein, Eran Ofek, Idan Szpektor, Avi Caciularu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08500)  

**Abstract**: Retrieval Augmented Generation (RAG) is a commonly used approach for enhancing large language models (LLMs) with relevant and up-to-date information. However, the retrieved sources can often contain conflicting information and it remains unclear how models should address such discrepancies. In this work, we first propose a novel taxonomy of knowledge conflict types in RAG, along with the desired model behavior for each type. We then introduce CONFLICTS, a high-quality benchmark with expert annotations of conflict types in a realistic RAG setting. CONFLICTS is the first benchmark that enables tracking progress on how models address a wide range of knowledge conflicts. We conduct extensive experiments on this benchmark, showing that LLMs often struggle to appropriately resolve conflicts between sources. While prompting LLMs to explicitly reason about the potential conflict in the retrieved documents significantly improves the quality and appropriateness of their responses, substantial room for improvement in future research remains. 

**Abstract (ZH)**: 检索增强生成（RAG）中的知识冲突类型及应对策略：一个高质量的标准与实验 

---
# EtiCor++: Towards Understanding Etiquettical Bias in LLMs 

**Title (ZH)**: EtiCor++: 向理解LLM中的礼仪偏差迈进 

**Authors**: Ashutosh Dwivedi, Siddhant Shivdutt Singh, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08488)  

**Abstract**: In recent years, researchers have started analyzing the cultural sensitivity of LLMs. In this respect, Etiquettes have been an active area of research. Etiquettes are region-specific and are an essential part of the culture of a region; hence, it is imperative to make LLMs sensitive to etiquettes. However, there needs to be more resources in evaluating LLMs for their understanding and bias with regard to etiquettes. In this resource paper, we introduce EtiCor++, a corpus of etiquettes worldwide. We introduce different tasks for evaluating LLMs for knowledge about etiquettes across various regions. Further, we introduce various metrics for measuring bias in LLMs. Extensive experimentation with LLMs shows inherent bias towards certain regions. 

**Abstract (ZH)**: 近年来，研究人员开始分析大语言模型的文化敏感性。在这方面，礼仪已成为一个活跃的研究领域。礼仪是地域性的，是特定地区文化的重要组成部分；因此，使大语言模型对礼仪敏感是必不可少的。然而，需要更多的资源来评估大语言模型对礼仪的理解和偏见。在这篇资源论文中，我们介绍了一种全球礼仪语料库EtiCor++。我们介绍了不同的任务，用于评估大语言模型在不同地区的礼仪知识。此外，我们介绍了衡量大语言模型偏见的各种指标。对大语言模型的广泛实验表明，它们对某些地区的偏见是固有的。 

---
# Fairness is Not Silence: Unmasking Vacuous Neutrality in Small Language Models 

**Title (ZH)**: 公平性不是沉默：揭示小型语言模型中的空洞中立性 

**Authors**: Sumanth Manduru, Carlotta Domeniconi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08487)  

**Abstract**: The rapid adoption of Small Language Models (SLMs) for on-device and resource-constrained deployments has outpaced our understanding of their ethical risks. To the best of our knowledge, we present the first large-scale audit of instruction-tuned SLMs spanning 0.5 to 5 billion parameters-an overlooked "middle tier" between BERT-class encoders and flagship LLMs. Our evaluation includes nine open-source models from the Qwen 2.5, LLaMA 3.2, Gemma 3, and Phi families. Using the BBQ benchmark under zero-shot prompting, we analyze both utility and fairness across ambiguous and disambiguated contexts. This evaluation reveals three key insights. First, competence and fairness need not be antagonistic: Phi models achieve F1 scores exceeding 90 percent while exhibiting minimal bias, showing that efficient and ethical NLP is attainable. Second, social bias varies significantly by architecture: Qwen 2.5 models may appear fair, but this often reflects vacuous neutrality, random guessing, or evasive behavior rather than genuine ethical alignment. In contrast, LLaMA 3.2 models exhibit stronger stereotypical bias, suggesting overconfidence rather than neutrality. Third, compression introduces nuanced trade-offs: 4-bit AWQ quantization improves F1 scores in ambiguous settings for LLaMA 3.2-3B but increases disability-related bias in Phi-4-Mini by over 7 percentage points. These insights provide practical guidance for the responsible deployment of SLMs in applications demanding fairness and efficiency, particularly benefiting small enterprises and resource-constrained environments. 

**Abstract (ZH)**: Small Language Models (SLMs) 的 rapid adoption for on-device and resource-constrained deployments has outpaced our understanding of their ethical risks. To the best of our knowledge, we present the first large-scale audit of instruction-tuned SLMs spanning 0.5 to 5 billion parameters—a overlooked "middle tier" between BERT-class encoders and flagship LLMs. Our evaluation includes nine open-source models from the Qwen 2.5, LLaMA 3.2, Gemma 3, and Phi families. Using the BBQ benchmark under zero-shot prompting, we analyze both utility and fairness across ambiguous and disambiguated contexts. This evaluation reveals three key insights. First, competence and fairness need not be antagonistic: Phi models achieve F1 scores exceeding 90 percent while exhibiting minimal bias, showing that efficient and ethical NLP is attainable. Second, social bias varies significantly by architecture: Qwen 2.5 models may appear fair, but this often reflects vacuous neutrality, random guessing, or evasive behavior rather than genuine ethical alignment. In contrast, LLaMA 3.2 models exhibit stronger stereotypical bias, suggesting overconfidence rather than neutrality. Third, compression introduces nuanced trade-offs: 4-bit AWQ quantization improves F1 scores in ambiguous settings for LLaMA 3.2-3B but increases disability-related bias in Phi-4-Mini by over 7 percentage points. These insights provide practical guidance for the responsible deployment of SLMs in applications demanding fairness and efficiency, particularly benefiting small enterprises and resource-constrained environments。 

---
# Re-Thinking the Automatic Evaluation of Image-Text Alignment in Text-to-Image Models 

**Title (ZH)**: 重思文本到图像模型中图像-文本对齐的自动评价 

**Authors**: Huixuan Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08480)  

**Abstract**: Text-to-image models often struggle to generate images that precisely match textual prompts. Prior research has extensively studied the evaluation of image-text alignment in text-to-image generation. However, existing evaluations primarily focus on agreement with human assessments, neglecting other critical properties of a trustworthy evaluation framework. In this work, we first identify two key aspects that a reliable evaluation should address. We then empirically demonstrate that current mainstream evaluation frameworks fail to fully satisfy these properties across a diverse range of metrics and models. Finally, we propose recommendations for improving image-text alignment evaluation. 

**Abstract (ZH)**: Text-to-image模型常常难以生成与文本提示精确匹配的图像。先前的研究已经广泛研究了文本到图像生成中的图像-文本对齐评估。然而，现有的评估主要侧重于与人类评估的一致性，忽视了可信评估框架的其他关键属性。在本文中，我们首先识别出可靠评估应该解决的两个关键方面。然后，我们通过实证方法证明当前主流的评估框架无法在多种度量标准和模型中全面满足这些属性。最后，我们提出改进图像-文本对齐评估的建议。 

---
# Efficient Context Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-$k$ 

**Title (ZH)**: 长上下文QA中的高效背景选择：无需调优，无需迭代，只需自适应-$k$ 

**Authors**: Chihiro Taguchi, Seiji Maekawa, Nikita Bhutani  

**Link**: [PDF](https://arxiv.org/pdf/2506.08479)  

**Abstract**: Retrieval-augmented generation (RAG) and long-context language models (LCLMs) both address context limitations of LLMs in open-domain question answering (QA). However, optimal external context to retrieve remains an open problem: fixing the retrieval size risks either wasting tokens or omitting key evidence. Existing adaptive methods like Self-RAG and Self-Route rely on iterative LLM prompting and perform well on factoid QA, but struggle with aggregation QA, where the optimal context size is both unknown and variable. We present Adaptive-$k$ retrieval, a simple and effective single-pass method that adaptively selects the number of passages based on the distribution of the similarity scores between the query and the candidate passages. It does not require model fine-tuning, extra LLM inferences or changes to existing retriever-reader pipelines. On both factoid and aggregation QA benchmarks, Adaptive-$k$ matches or outperforms fixed-$k$ baselines while using up to 10x fewer tokens than full-context input, yet still retrieves 70% of relevant passages. It improves accuracy across five LCLMs and two embedding models, highlighting that dynamically adjusting context size leads to more efficient and accurate QA. 

**Abstract (ZH)**: Retrieval-augmented generation (RAG)和长上下文语言模型（LCLMs）都解决了大规模语言模型在开放域问答中的上下文限制问题。然而，检索到的最佳外部上下文仍然存在开放问题：固定检索大小的风险是既浪费令牌又可能省略关键证据。现有的自适应方法如Self-RAG和Self-Route依赖于迭代的LLM提示，在事实型问答中表现良好，但在聚合型问答中遇到困难，因为在聚合型问答中，最优的上下文大小既未知又变化。我们提出了一种简单的单步自适应检索（Adaptive-$k$ retrieval）方法，该方法根据查询与候选段落相似得分的分布，自适应地选择段落数量。该方法不需要模型微调、额外的LLM推理或更改现有的检索-阅读管道。在事实型和聚合型问答基准测试中，Adaptive-$k$ 在使用比全长上下文输入少10倍的令牌的情况下，与固定-$k$ 基线持平或表现出色，同时仍检索到70%的相关段落。该方法在五个LCLMs和两种嵌入模型上提高了准确性，突显了动态调整上下文大小可以提高问答的效率和准确性。 

---
# How to Provably Improve Return Conditioned Supervised Learning? 

**Title (ZH)**: 如何证明性提高基于回报条件的监督学习？ 

**Authors**: Zhishuai Liu, Yu Yang, Ruhan Wang, Pan Xu, Dongruo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.08463)  

**Abstract**: In sequential decision-making problems, Return-Conditioned Supervised Learning (RCSL) has gained increasing recognition for its simplicity and stability in modern decision-making tasks. Unlike traditional offline reinforcement learning (RL) algorithms, RCSL frames policy learning as a supervised learning problem by taking both the state and return as input. This approach eliminates the instability often associated with temporal difference (TD) learning in offline RL. However, RCSL has been criticized for lacking the stitching property, meaning its performance is inherently limited by the quality of the policy used to generate the offline dataset. To address this limitation, we propose a principled and simple framework called Reinforced RCSL. The key innovation of our framework is the introduction of a concept we call the in-distribution optimal return-to-go. This mechanism leverages our policy to identify the best achievable in-dataset future return based on the current state, avoiding the need for complex return augmentation techniques. Our theoretical analysis demonstrates that Reinforced RCSL can consistently outperform the standard RCSL approach. Empirical results further validate our claims, showing significant performance improvements across a range of benchmarks. 

**Abstract (ZH)**: 基于返回条件的监督学习在序贯决策问题中的强化Reinforced RCSL框架 

---
# MOBODY: Model Based Off-Dynamics Offline Reinforcement Learning 

**Title (ZH)**: MOBODY：基于模型的离线动力学 Offline 强化学习 

**Authors**: Yihong Guo, Yu Yang, Pan Xu, Anqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08460)  

**Abstract**: We study the off-dynamics offline reinforcement learning problem, where the goal is to learn a policy from offline datasets collected from source and target domains with mismatched transition. Existing off-dynamics offline RL methods typically either filter source transitions that resemble those of the target domain or apply reward augmentation to source data, both constrained by the limited transitions available from the target domain. As a result, the learned policy is unable to explore target domain beyond the offline datasets. We propose MOBODY, a Model-Based Off-Dynamics offline RL algorithm that addresses this limitation by enabling exploration of the target domain via learned dynamics. MOBODY generates new synthetic transitions in the target domain through model rollouts, which are used as data augmentation during offline policy learning. Unlike existing model-based methods that learn dynamics from a single domain, MOBODY tackles the challenge of mismatched dynamics by leveraging both source and target datasets. Directly merging these datasets can bias the learned model toward source dynamics. Instead, MOBODY learns target dynamics by discovering a shared latent representation of states and transitions across domains through representation learning. To stabilize training, MOBODY incorporates a behavior cloning loss that regularizes the policy. Specifically, we introduce a Q-weighted behavior cloning loss that regularizes the policy toward actions with high target-domain Q-values, rather than uniformly imitating all actions in the dataset. These Q-values are learned from an enhanced target dataset composed of offline target data, augmented source data, and rollout data from the learned target dynamics. We evaluate MOBODY on MuJoCo benchmarks and show that it significantly outperforms state-of-the-art baselines, with especially pronounced improvements in challenging scenarios. 

**Abstract (ZH)**: 我们研究了离线动力学离线强化学习问题，其目标是从源域和目标域之间存在转换不匹配的离线数据集中学习策略。现有的离线动力学离线RL方法通常要么过滤掉与目标域相似的源域转换，要么对源数据进行奖励增强，这两种方法都受到目标域可用转换有限的限制。因此，学到的策略无法探索超出离线数据集的目标域。我们提出了MOBODY，一种基于模型的离线动力学离线RL算法，通过利用学到的动力学来探索目标域来解决这一限制。MOBODY通过模型rollout生成新的合成转换，这些转换在离线策略学习期间用作数据增强。与从单个域学习动力学的现有基于模型的方法不同，MOBODY通过利用源域和目标域的数据集来应对动力学不匹配的挑战。直接将这些数据集合并会导致学到的模型偏向源域动力学。相反，MOBODY通过代表性学习在各域之间发现状态和转换的共享潜在表示来学习目标动力学。为了稳定训练，MOBODY引入了一种Q加权行为克隆损失来正则化策略。具体而言，我们引入了一种Q加权行为克隆损失，该损失将策略正则化为具有高目标域Q值的动作，而不是均匀地模仿数据集中的所有动作。这些Q值是从扩展的目标数据集中学习而来的，该数据集包括离线目标数据、增强的源数据和学习到的目标动力学生成的rollout数据。我们在MuJoCo基准上评估了MOBODY，并展示了它显著优于最先进的基线方法，特别是在具有挑战性的场景中表现尤为突出。 

---
# Diffusion Models for Safety Validation of Autonomous Driving Systems 

**Title (ZH)**: 自动驾驶系统安全验证的扩散模型 

**Authors**: Juanran Wang, Marc R. Schlichting, Harrison Delecki, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2506.08459)  

**Abstract**: Safety validation of autonomous driving systems is extremely challenging due to the high risks and costs of real-world testing as well as the rarity and diversity of potential failures. To address these challenges, we train a denoising diffusion model to generate potential failure cases of an autonomous vehicle given any initial traffic state. Experiments on a four-way intersection problem show that in a variety of scenarios, the diffusion model can generate realistic failure samples while capturing a wide variety of potential failures. Our model does not require any external training dataset, can perform training and inference with modest computing resources, and does not assume any prior knowledge of the system under test, with applicability to safety validation for traffic intersections. 

**Abstract (ZH)**: 自动驾驶系统的安全性验证由于实际测试的风险和成本高以及潜在故障的稀有性和多样性而极具挑战性。为应对这些挑战，我们训练了一个去噪扩散模型，给定任何初始交通状态，生成自动驾驶车辆的潜在故障案例。在四向交叉路口问题上的实验表明，在多种场景下，扩散模型可以生成现实的故障样本，同时捕捉到各种潜在的故障。我们的模型不需要任何外部训练数据集，可以在有限的计算资源下进行训练和推理，并不对待测试系统有任何先验知识假设，适用于交通交叉路口的安全验证。 

---
# Time-Aware World Model for Adaptive Prediction and Control 

**Title (ZH)**: 时间感知世界模型及其在自适应预测与控制中的应用 

**Authors**: Anh N. Nhu, Sanghyun Son, Ming Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.08441)  

**Abstract**: In this work, we introduce the Time-Aware World Model (TAWM), a model-based approach that explicitly incorporates temporal dynamics. By conditioning on the time-step size, {\Delta}t, and training over a diverse range of {\Delta}t values -- rather than sampling at a fixed time-step -- TAWM learns both high- and low-frequency task dynamics across diverse control problems. Grounded in the information-theoretic insight that the optimal sampling rate depends on a system's underlying dynamics, this time-aware formulation improves both performance and data efficiency. Empirical evaluations show that TAWM consistently outperforms conventional models across varying observation rates in a variety of control tasks, using the same number of training samples and iterations. Our code can be found online at: this http URL. 

**Abstract (ZH)**: 基于时间的 WORLD 模型 (TAWM):一种explicitly Incorporate 时间动态的模型驱动方法 

---
# HASFL: Heterogeneity-aware Split Federated Learning over Edge Computing Systems 

**Title (ZH)**: HASFL：边缘计算系统中aware分裂联邦学习 

**Authors**: Zheng Lin, Zhe Chen, Xianhao Chen, Wei Ni, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.08426)  

**Abstract**: Split federated learning (SFL) has emerged as a promising paradigm to democratize machine learning (ML) on edge devices by enabling layer-wise model partitioning. However, existing SFL approaches suffer significantly from the straggler effect due to the heterogeneous capabilities of edge devices. To address the fundamental challenge, we propose adaptively controlling batch sizes (BSs) and model splitting (MS) for edge devices to overcome resource heterogeneity. We first derive a tight convergence bound of SFL that quantifies the impact of varied BSs and MS on learning performance. Based on the convergence bound, we propose HASFL, a heterogeneity-aware SFL framework capable of adaptively controlling BS and MS to balance communication-computing latency and training convergence in heterogeneous edge networks. Extensive experiments with various datasets validate the effectiveness of HASFL and demonstrate its superiority over state-of-the-art benchmarks. 

**Abstract (ZH)**: 适应异构性控制批大小和模型划分的自感知联邦学习（HASFL） 

---
# Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood 

**Title (ZH)**: 离线强化学习中凸包及其邻域内的平滑OOD泛化 

**Authors**: Qingmao Yao, Zhichao Lei, Tianyuan Chen, Ziyue Yuan, Xuefan Chen, Jianxiang Liu, Faguo Wu, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08417)  

**Abstract**: Offline Reinforcement Learning (RL) struggles with distributional shifts, leading to the $Q$-value overestimation for out-of-distribution (OOD) actions. Existing methods address this issue by imposing constraints; however, they often become overly conservative when evaluating OOD regions, which constrains the $Q$-function generalization. This over-constraint issue results in poor $Q$-value estimation and hinders policy improvement. In this paper, we introduce a novel approach to achieve better $Q$-value estimation by enhancing $Q$-function generalization in OOD regions within Convex Hull and its Neighborhood (CHN). Under the safety generalization guarantees of the CHN, we propose the Smooth Bellman Operator (SBO), which updates OOD $Q$-values by smoothing them with neighboring in-sample $Q$-values. We theoretically show that SBO approximates true $Q$-values for both in-sample and OOD actions within the CHN. Our practical algorithm, Smooth Q-function OOD Generalization (SQOG), empirically alleviates the over-constraint issue, achieving near-accurate $Q$-value estimation. On the D4RL benchmarks, SQOG outperforms existing state-of-the-art methods in both performance and computational efficiency. 

**Abstract (ZH)**: Convex Hull and its Neighborhood Based Smooth Q-function OOD Generalization 

---
# TACTIC: Translation Agents with Cognitive-Theoretic Interactive Collaboration 

**Title (ZH)**: TACTIC: 具有认知理论交互合作的翻译代理 

**Authors**: Weiya Li, Junjie Chen, Bei Li, Boyang Liu, Zichen Wen, Nuanqiao Shan, Xiaoqian Liu, Anping Liu, Huajie Liu, Youyan Wang, Wujiuge Yin, Hu Song, Bing Huang, Zhiyuan Xia, Jialiang Chen, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08403)  

**Abstract**: Machine translation has long been a central task in natural language processing. With the rapid advancement of large language models (LLMs), there has been remarkable progress in translation quality. However, fully realizing the translation potential of LLMs remains an open challenge. Recent studies have explored multi-agent systems to decompose complex translation tasks into collaborative subtasks, showing initial promise in enhancing translation quality through agent cooperation and specialization. Nevertheless, existing multi-agent translation frameworks largely neglect foundational insights from cognitive translation studies. These insights emphasize how human translators employ different cognitive strategies, such as balancing literal and free translation, refining expressions based on context, and iteratively evaluating outputs. To address this limitation, we propose a cognitively informed multi-agent framework called TACTIC, which stands for T ranslation A gents with Cognitive- T heoretic Interactive Collaboration. The framework comprises six functionally distinct agents that mirror key cognitive processes observed in human translation behavior. These include agents for drafting, refinement, evaluation, scoring, context reasoning, and external knowledge gathering. By simulating an interactive and theory-grounded translation workflow, TACTIC effectively leverages the full capacity of LLMs for high-quality translation. Experimental results on diverse language pairs from the FLORES-200 and WMT24 benchmarks show that our method consistently achieves state-of-the-art performance. Using DeepSeek-V3 as the base model, TACTIC surpasses GPT-4.1 by an average of +0.6 XCOMET and +1.18 COMETKIWI-23. Compared to DeepSeek-R1, it further improves by +0.84 XCOMET and +2.99 COMETKIWI-23. Code is available at this https URL. 

**Abstract (ZH)**: 认知导向的多智能体翻译框架TACTIC 

---
# Spatiotemporal deep learning models for detection of rapid intensification in cyclones 

**Title (ZH)**: 基于时空深度学习模型的cyclone快速增强检测方法 

**Authors**: Vamshika Sutar, Amandeep Singh, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.08397)  

**Abstract**: Cyclone rapid intensification is the rapid increase in cyclone wind intensity, exceeding a threshold of 30 knots, within 24 hours. Rapid intensification is considered an extreme event during a cyclone, and its occurrence is relatively rare, contributing to a class imbalance in the dataset. A diverse array of factors influences the likelihood of a cyclone undergoing rapid intensification, further complicating the task for conventional machine learning models. In this paper, we evaluate deep learning, ensemble learning and data augmentation frameworks to detect cyclone rapid intensification based on wind intensity and spatial coordinates. We note that conventional data augmentation methods cannot be utilised for generating spatiotemporal patterns replicating cyclones that undergo rapid intensification. Therefore, our framework employs deep learning models to generate spatial coordinates and wind intensity that replicate cyclones to address the class imbalance problem of rapid intensification. We also use a deep learning model for the classification module within the data augmentation framework to differentiate between rapid and non-rapid intensification events during a cyclone. Our results show that data augmentation improves the results for rapid intensification detection in cyclones, and spatial coordinates play a critical role as input features to the given models. This paves the way for research in synthetic data generation for spatiotemporal data with extreme events. 

**Abstract (ZH)**: 热带气旋快速增强的深度学习检测方法研究 

---
# Reinforcement Learning Teachers of Test Time Scaling 

**Title (ZH)**: 测试时间缩放的强化学习教师 

**Authors**: Edoardo Cetin, Tianyu Zhao, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08388)  

**Abstract**: Training reasoning language models (LMs) with reinforcement learning (RL) for one-hot correctness inherently relies on the LM being able to explore and solve its task with some chance at initialization. Furthermore, a key use case of reasoning LMs is to act as teachers for distilling new students and cold-starting future RL iterations rather than being deployed themselves. From these considerations, we introduce a new framework that avoids RL's exploration challenge by training a new class of Reinforcement-Learned Teachers (RLTs) focused on yielding the most effective downstream distillation. RLTs are prompted with both the question and solution to each problem, and tasked to simply "connect-the-dots" with detailed explanations tailored for their students. We train RLTs with dense rewards obtained by feeding each explanation to the student and testing its understanding of the problem's solution. In practice, the raw outputs of a 7B RLT provide higher final performance on competition and graduate-level tasks than existing distillation and cold-starting pipelines that collect and postprocess the reasoning traces of orders of magnitude larger LMs. Furthermore, RLTs maintain their effectiveness when training larger students and when applied zero-shot to out-of-distribution tasks, unlocking new levels of efficiency and re-usability for the RL reasoning framework. 

**Abstract (ZH)**: 使用强化学习训练推理语言模型以实现一对一正确性内在依赖于模型能够在初始化时探索并解决其任务。此外，推理语言模型的一个关键应用是作为老师的角色，用于指导新的学生并初始化未来的强化学习迭代，而不是直接部署。基于这些考虑，我们提出了一个新的框架，通过训练一种新的强化学习教师（RLTs），避免了强化学习的探索挑战，专注于产生最有效的下游蒸馏效果。RLTs 接受每个问题及其解，并被指派简单地“连点成线”，并为学生提供详细的解释。我们通过将每种解释输入学生并测试其对问题解的理解来用密集奖励训练RLTs。实践中，7B RLT的原始输出在比赛和研究生水平的任务上提供了比现有蒸馏和冷启动流水线更高的最终性能，这些流水线收集并处理了数量级更大的语言模型的推理踪迹。此外，RLTs 在训练更大规模的学生时保持其有效性，并在零样本情况下应用于分布外任务，为RL推理框架解锁了新的效率和可重用性。 

---
# Reinforce LLM Reasoning through Multi-Agent Reflection 

**Title (ZH)**: 通过多智能体反思增强LLM推理 

**Authors**: Yurun Yuan, Tengyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.08379)  

**Abstract**: Leveraging more test-time computation has proven to be an effective way to boost the reasoning capabilities of large language models (LLMs). Among various methods, the verify-and-improve paradigm stands out for enabling dynamic solution exploration and feedback incorporation. However, existing approaches often suffer from restricted feedback spaces and lack of coordinated training of different parties, leading to suboptimal performance. To address this, we model this multi-turn refinement process as a Markov Decision Process and introduce DPSDP (Direct Policy Search by Dynamic Programming), a reinforcement learning algorithm that trains an actor-critic LLM system to iteratively refine answers via direct preference learning on self-generated data. Theoretically, DPSDP can match the performance of any policy within the training distribution. Empirically, we instantiate DPSDP with various base models and show improvements on both in- and out-of-distribution benchmarks. For example, on benchmark MATH 500, majority voting over five refinement steps increases first-turn accuracy from 58.2% to 63.2% with Ministral-based models. An ablation study further confirms the benefits of multi-agent collaboration and out-of-distribution generalization. 

**Abstract (ZH)**: 利用更多推理时间的计算已被证明是提升大语言模型推理能力的有效方式。在各种方法中，验证与改进范式因其能够实现动态解决方案探索和反馈整合而脱颖而出。然而，现有方法往往受限于有限的反馈空间并缺乏不同参与方的协调训练，导致性能不佳。为解决这一问题，我们将这一多轮完善过程建模为马尔科夫决策过程，并引入DPSDP（基于动态规划的直接策略搜索）——一种强化学习算法，通过自生成数据上的直接偏好学习训练演员-评论家大语言模型系统以迭代改进答案。理论上，DPSDP可以匹配训练分布内的任何策略性能。实验中，我们使用多种基础模型实例化DPSDP，并在分布内外基准测试中显示出改进。例如，在MATH 500基准上，使用Minstral为基础模型的五轮完善步骤的多数投票将第一轮准确率从58.2%提高到63.2%。去机制化研究进一步证实了多Agent合作和分布外泛化的益处。 

---
# Draft-based Approximate Inference for LLMs 

**Title (ZH)**: 基于草图的近似推理for大语言模型 

**Authors**: Kevin Galim, Ethan Ewer, Wonjun Kang, Minjae Lee, Hyung Il Koo, Kangwook Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.08373)  

**Abstract**: Optimizing inference for long-context Large Language Models (LLMs) is increasingly important due to the quadratic compute and linear memory complexity of Transformers. Existing approximation methods, such as key-value (KV) cache dropping, sparse attention, and prompt compression, typically rely on rough predictions of token or KV pair importance. We propose a novel framework for approximate LLM inference that leverages small draft models to more accurately predict the importance of tokens and KV pairs. Specifically, we introduce two instantiations of our proposed framework: (i) SpecKV, which leverages a draft output to accurately assess the importance of each KV pair for more effective KV cache dropping, and (ii) SpecPC, which uses the draft model's attention activations to identify and discard unimportant prompt tokens. To the best of our knowledge, this is the first work to use draft models for approximate LLM inference acceleration, extending their utility beyond traditional lossless speculative decoding. We motivate our methods with theoretical and empirical analyses, and show a strong correlation between the attention patterns of draft and target models. Extensive experiments on long-context benchmarks show that our methods consistently achieve higher accuracy than existing baselines, while preserving the same improvements in memory usage, latency, and throughput. Our code is available at this https URL. 

**Abstract (ZH)**: 优化具有长上下文的大型语言模型（LLMs）推理 increasingly important due to the quadratic compute and linear memory complexity of Transformers. 提出一种新颖的近似 LLMS 推理框架，利用草稿模型更准确地预测令牌和键值对的重要性。具体而言，我们介绍了我们提出框架的两种实例：(i) SpecKV，利用草稿输出更有效地评估每个键值对的重要性以进行草稿缓存丢弃，(ii) SpecPC，利用草稿模型的注意力激活来识别并丢弃不重要的提示令牌。据我们所知，这是首次使用草稿模型加速近似 LLMS 推理的工作，扩展了其在传统无损推测解码之外的用途。我们通过理论和实证分析来阐述我们的方法，并展示了草稿模型和目标模型的注意力模式之间存在强烈的相关性。在长上下文基准测试中的广泛实验表明，我们的方法始终比现有基线具有更高的准确性，同时保持相同的内存使用、延迟和吞吐量的改进。代码可在以下链接获取。 

---
# MD-ViSCo: A Unified Model for Multi-Directional Vital Sign Waveform Conversion 

**Title (ZH)**: MD-ViSCo: 统一的心脏生命体征波形多方向转换模型 

**Authors**: Franck Meyer, Kyunghoon Hur, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08357)  

**Abstract**: Despite the remarkable progress of deep-learning methods generating a target vital sign waveform from a source vital sign waveform, most existing models are designed exclusively for a specific source-to-target pair. This requires distinct model architectures, optimization procedures, and pre-processing pipelines, resulting in multiple models that hinder usability in clinical settings. To address this limitation, we propose the Multi-Directional Vital-Sign Converter (MD-ViSCo), a unified framework capable of generating any target waveform such as electrocardiogram (ECG), photoplethysmogram (PPG), or arterial blood pressure (ABP) from any single input waveform with a single model. MD-ViSCo employs a shallow 1-Dimensional U-Net integrated with a Swin Transformer that leverages Adaptive Instance Normalization (AdaIN) to capture distinct waveform styles. To evaluate the efficacy of MD-ViSCo, we conduct multi-directional waveform generation on two publicly available datasets. Our framework surpasses state-of-the-art baselines (NabNet & PPG2ABP) on average across all waveform types, lowering Mean absolute error (MAE) by 8.8% and improving Pearson correlation (PC) by 4.9% over two datasets. In addition, the generated ABP waveforms satisfy the Association for the Advancement of Medical Instrumentation (AAMI) criterion and achieve Grade B on the British Hypertension Society (BHS) standard, outperforming all baselines. By eliminating the need for developing a distinct model for each task, we believe that this work offers a unified framework that can deal with any kind of vital sign waveforms with a single model in healthcare monitoring. 

**Abstract (ZH)**: 多方向生理信号转换器：使用单一模型生成任意目标波形 

---
# Text Embeddings Should Capture Implicit Semantics, Not Just Surface Meaning 

**Title (ZH)**: 文本嵌入应捕捉隐含语义，而不仅仅是表面意义 

**Authors**: Yiqun Sun, Qiang Huang, Anthony K. H. Tung, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08354)  

**Abstract**: This position paper argues that the text embedding research community should move beyond surface meaning and embrace implicit semantics as a central modeling goal. Text embedding models have become foundational in modern NLP, powering a wide range of applications and drawing increasing research attention. Yet, much of this progress remains narrowly focused on surface-level semantics. In contrast, linguistic theory emphasizes that meaning is often implicit, shaped by pragmatics, speaker intent, and sociocultural context. Current embedding models are typically trained on data that lacks such depth and evaluated on benchmarks that reward the capture of surface meaning. As a result, they struggle with tasks requiring interpretive reasoning, speaker stance, or social meaning. Our pilot study highlights this gap, showing that even state-of-the-art models perform only marginally better than simplistic baselines on implicit semantics tasks. To address this, we call for a paradigm shift: embedding research should prioritize more diverse and linguistically grounded training data, design benchmarks that evaluate deeper semantic understanding, and explicitly frame implicit meaning as a core modeling objective, better aligning embeddings with real-world language complexity. 

**Abstract (ZH)**: 这一立场论文argues认为，文本嵌入研究社区应超越表面意义，拥抱潜在语义作为核心建模目标。 

---
# How Much To Guide: Revisiting Adaptive Guidance in Classifier-Free Guidance Text-to-Vision Diffusion Models 

**Title (ZH)**: 引导多少：重启无分类器引导的文字到视觉扩散模型中的自适应引导研究 

**Authors**: Huixuan Zhang, Junzhe Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08351)  

**Abstract**: With the rapid development of text-to-vision generation diffusion models, classifier-free guidance has emerged as the most prevalent method for conditioning. However, this approach inherently requires twice as many steps for model forwarding compared to unconditional generation, resulting in significantly higher costs. While previous study has introduced the concept of adaptive guidance, it lacks solid analysis and empirical results, making previous method unable to be applied to general diffusion models. In this work, we present another perspective of applying adaptive guidance and propose Step AG, which is a simple, universally applicable adaptive guidance strategy. Our evaluations focus on both image quality and image-text alignment. whose results indicate that restricting classifier-free guidance to the first several denoising steps is sufficient for generating high-quality, well-conditioned images, achieving an average speedup of 20% to 30%. Such improvement is consistent across different settings such as inference steps, and various models including video generation models, highlighting the superiority of our method. 

**Abstract (ZH)**: 基于步进的自适应指导：一种简单通用的方法及其应用 

---
# Evaluating LLMs Across Multi-Cognitive Levels: From Medical Knowledge Mastery to Scenario-Based Problem Solving 

**Title (ZH)**: 多认知层次下大语言模型的评估：从医学知识掌握到情境基于的问题解决 

**Authors**: Yuxuan Zhou, Xien Liu, Chenwei Yan, Chen Ning, Xiao Zhang, Boxun Li, Xiangling Fu, Shijin Wang, Guoping Hu, Yu Wang, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08349)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance on various medical benchmarks, but their capabilities across different cognitive levels remain underexplored. Inspired by Bloom's Taxonomy, we propose a multi-cognitive-level evaluation framework for assessing LLMs in the medical domain in this study. The framework integrates existing medical datasets and introduces tasks targeting three cognitive levels: preliminary knowledge grasp, comprehensive knowledge application, and scenario-based problem solving. Using this framework, we systematically evaluate state-of-the-art general and medical LLMs from six prominent families: Llama, Qwen, Gemma, Phi, GPT, and DeepSeek. Our findings reveal a significant performance decline as cognitive complexity increases across evaluated models, with model size playing a more critical role in performance at higher cognitive levels. Our study highlights the need to enhance LLMs' medical capabilities at higher cognitive levels and provides insights for developing LLMs suited to real-world medical applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种医学基准测试中展现了卓越的表现，但它们在不同认知层次的能力仍待深入探索。受布卢姆分类法的启发，本文提出了一种多认知层次评估框架，用于评估LLMs在医学领域的表现。该框架整合了现有医学数据集，并引入了针对三个认知层次的任务：初步知识掌握、全面知识应用以及场景基于的问题解决。利用此框架，我们系统性地评估了六大家族顶尖的通用和医学LLMs：Llama、Qwen、Gemma、Phi、GPT和DeepSeek。研究发现，在评估模型中，随着认知复杂性的增加，其性能显著下降，模型大小在较高认知层次上的性能表现更为关键。本研究突出了提升LLMs在较高认知层次上的医学能力的需求，并为开发适用于真实世界医学应用的LLMs提供了见解。 

---
# SPBA: Utilizing Speech Large Language Model for Backdoor Attacks on Speech Classification Models 

**Title (ZH)**: SPBA: 利用语音大规模语言模型对语音分类模型进行后门攻击 

**Authors**: Wenhan Yao, Fen Xiao, Xiarun Chen, Jia Liu, YongQiang He, Weiping Wen  

**Link**: [PDF](https://arxiv.org/pdf/2506.08346)  

**Abstract**: Deep speech classification tasks, including keyword spotting and speaker verification, are vital in speech-based human-computer interaction. Recently, the security of these technologies has been revealed to be susceptible to backdoor attacks. Specifically, attackers use noisy disruption triggers and speech element triggers to produce poisoned speech samples that train models to become vulnerable. However, these methods typically create only a limited number of backdoors due to the inherent constraints of the trigger function. In this paper, we propose that speech backdoor attacks can strategically focus on speech elements such as timbre and emotion, leveraging the Speech Large Language Model (SLLM) to generate diverse triggers. Increasing the number of triggers may disproportionately elevate the poisoning rate, resulting in higher attack costs and a lower success rate per trigger. We introduce the Multiple Gradient Descent Algorithm (MGDA) as a mitigation strategy to address this challenge. The proposed attack is called the Speech Prompt Backdoor Attack (SPBA). Building on this foundation, we conducted attack experiments on two speech classification tasks, demonstrating that SPBA shows significant trigger effectiveness and achieves exceptional performance in attack metrics. 

**Abstract (ZH)**: 基于语音的后门攻击：利用语音大语言模型的战略性触发要素定位（Speech Prompt Backdoor Attack: Leveraging Speech Large Language Model for Strategic Trigger Element Focusing） 

---
# Re4MPC: Reactive Nonlinear MPC for Multi-model Motion Planning via Deep Reinforcement Learning 

**Title (ZH)**: Re4MPC: 基于深度强化学习的多模型运动规划的反应式非线性MPC 

**Authors**: Neşet Ünver Akmandor, Sarvesh Prajapati, Mark Zolotas, Taşkın Padır  

**Link**: [PDF](https://arxiv.org/pdf/2506.08344)  

**Abstract**: Traditional motion planning methods for robots with many degrees-of-freedom, such as mobile manipulators, are often computationally prohibitive for real-world settings. In this paper, we propose a novel multi-model motion planning pipeline, termed Re4MPC, which computes trajectories using Nonlinear Model Predictive Control (NMPC). Re4MPC generates trajectories in a computationally efficient manner by reactively selecting the model, cost, and constraints of the NMPC problem depending on the complexity of the task and robot state. The policy for this reactive decision-making is learned via a Deep Reinforcement Learning (DRL) framework. We introduce a mathematical formulation to integrate NMPC into this DRL framework. To validate our methodology and design choices, we evaluate DRL training and test outcomes in a physics-based simulation involving a mobile manipulator. Experimental results demonstrate that Re4MPC is more computationally efficient and achieves higher success rates in reaching end-effector goals than the NMPC baseline, which computes whole-body trajectories without our learning mechanism. 

**Abstract (ZH)**: 面向多自由度机器人的传统运动规划方法在实际应用场景中往往计算代价高昂。本文提出了一种名为Re4MPC的新型多模型运动规划管道，通过非线性模型预测控制（NMPC）计算轨迹。Re4MPC根据任务复杂性和机器人状态反应性地选择NMPC问题中的模型、成本和约束条件，从而以计算高效的方式生成轨迹。该反应性决策策略通过深度强化学习（DRL）框架学习。本文引入了将NMPC整合到该DRL框架中的数学公式。为了验证我们的方法和设计选择，我们在涉及移动 manipulator的基于物理的仿真中评估了DRL训练和测试结果。实验结果表明，与不包含学习机制的全身体现NMPC基线相比，Re4MPC在计算效率和末端执行器目标达到的成功率方面更具优势。 

---
# Your Agent Can Defend Itself against Backdoor Attacks 

**Title (ZH)**: 你的代理可以防御后门攻击 

**Authors**: Li Changjiang, Liang Jiacheng, Cao Bochuan, Chen Jinghui, Wang Ting  

**Link**: [PDF](https://arxiv.org/pdf/2506.08336)  

**Abstract**: Despite their growing adoption across domains, large language model (LLM)-powered agents face significant security risks from backdoor attacks during training and fine-tuning. These compromised agents can subsequently be manipulated to execute malicious operations when presented with specific triggers in their inputs or environments. To address this pressing risk, we present ReAgent, a novel defense against a range of backdoor attacks on LLM-based agents. Intuitively, backdoor attacks often result in inconsistencies among the user's instruction, the agent's planning, and its execution. Drawing on this insight, ReAgent employs a two-level approach to detect potential backdoors. At the execution level, ReAgent verifies consistency between the agent's thoughts and actions; at the planning level, ReAgent leverages the agent's capability to reconstruct the instruction based on its thought trajectory, checking for consistency between the reconstructed instruction and the user's instruction. Extensive evaluation demonstrates ReAgent's effectiveness against various backdoor attacks across tasks. For instance, ReAgent reduces the attack success rate by up to 90\% in database operation tasks, outperforming existing defenses by large margins. This work reveals the potential of utilizing compromised agents themselves to mitigate backdoor risks. 

**Abstract (ZH)**: 尽管大型语言模型（LLM）驱动的代理在各个领域中的应用不断增加，但在训练和微调过程中，这些代理面临严重的后门攻击安全风险。这些受损的代理可能在接收到特定触发器时被操纵以执行恶意操作。为了应对这一紧迫的风险，我们提出了ReAgent，这是一种针对基于LLM代理的多种后门攻击的新颖防御措施。直观地讲，后门攻击常常导致用户指令、代理规划和执行之间的不一致性。基于这一洞察，ReAgent采用两级方法来检测潜在后门。在执行层面，ReAgent验证代理的想法和行动之间的一致性；在规划层面，ReAgent利用代理根据其思维轨迹重建指令的能力，检查重建指令与用户指令之间的一致性。广泛的评估证明了ReAgent在各种任务中对抗后门攻击的有效性。例如，在数据库操作任务中，ReAgent将攻击成功率降低多达90%，远超现有防御措施的效果。这项工作揭示了可以利用受损代理本身来减轻后门风险的潜力。 

---
# Graph Prompting for Graph Learning Models: Recent Advances and Future Directions 

**Title (ZH)**: 图提示技术在图学习模型中的应用：近期进展与未来方向 

**Authors**: Xingbo Fu, Zehong Wang, Zihan Chen, Jiazheng Li, Yaochen Zhu, Zhenyu Lei, Cong Shen, Yanfang Ye, Chuxu Zhang, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.08326)  

**Abstract**: Graph learning models have demonstrated great prowess in learning expressive representations from large-scale graph data in a wide variety of real-world scenarios. As a prevalent strategy for training powerful graph learning models, the "pre-training, adaptation" scheme first pre-trains graph learning models on unlabeled graph data in a self-supervised manner and then adapts them to specific downstream tasks. During the adaptation phase, graph prompting emerges as a promising approach that learns trainable prompts while keeping the pre-trained graph learning models unchanged. In this paper, we present a systematic review of recent advancements in graph prompting. First, we introduce representative graph pre-training methods that serve as the foundation step of graph prompting. Next, we review mainstream techniques in graph prompting and elaborate on how they design learnable prompts for graph prompting. Furthermore, we summarize the real-world applications of graph prompting from different domains. Finally, we discuss several open challenges in existing studies with promising future directions in this field. 

**Abstract (ZH)**: 图学习模型在广泛的真实世界场景中展示了从大规模图数据中学习丰富表示的强大能力。作为一种训练强大图学习模型的普遍策略，“预训练、适应”方案首先在自监督方式下于无标签图数据上预训练图学习模型，然后将这些模型适应到特定的下游任务。在适应阶段，图提示作为一种有前途的方法出现，它能够学习可训练的提示而不改变预训练的图学习模型。在本文中，我们对图提示领域的最新进展进行了系统综述。首先，我们介绍了作为图提示基础步骤的代表性的图预训练方法。接着，我们回顾了图提示中的主流技术，并解释了它们是如何设计用于图提示的可学习提示的。此外，我们总结了图提示在不同领域的实际应用。最后，我们讨论了现有研究中的几个开放挑战，并提出了该领域有前景的未来方向。 

---
# How Good LLM-Generated Password Policies Are? 

**Title (ZH)**: LLM生成的密码策略质量如何？ 

**Authors**: Vivek Vaidya, Aditya Patwardhan, Ashish Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08320)  

**Abstract**: Generative AI technologies, particularly Large Language Models (LLMs), are rapidly being adopted across industry, academia, and government sectors, owing to their remarkable capabilities in natural language processing. However, despite their strengths, the inconsistency and unpredictability of LLM outputs present substantial challenges, especially in security-critical domains such as access control. One critical issue that emerges prominently is the consistency of LLM-generated responses, which is paramount for ensuring secure and reliable operations.
In this paper, we study the application of LLMs within the context of Cybersecurity Access Control Systems. Specifically, we investigate the consistency and accuracy of LLM-generated password policies, translating natural language prompts into executable this http URL configuration files. Our experimental methodology adopts two distinct approaches: firstly, we utilize pre-trained LLMs to generate configuration files purely from natural language prompts without additional guidance. Secondly, we provide these models with official this http URL documentation to serve as an informative baseline. We systematically assess the soundness, accuracy, and consistency of these AI-generated configurations. Our findings underscore significant challenges in the current generation of LLMs and contribute valuable insights into refining the deployment of LLMs in Access Control Systems. 

**Abstract (ZH)**: _generative AI技术，特别是大型语言模型（LLMs），正迅速被工业、学术界和政府机构采用，这得益于它们在自然语言处理方面的卓越能力。然而，尽管LLMs具有优势，其输出的一致性和不可预测性给诸如访问控制等安全关键领域带来了重大挑战。一致性的缺乏，尤其是LLMs生成的回答的一致性问题，对于确保安全可靠的运行至关重要。

在本文中，我们研究了大型语言模型在网络安全访问控制系统中的应用。具体而言，我们探讨了LLMs生成的密码策略的一致性和准确性，将自然语言提示转换为可执行的配置文件。我们实验方法采用了两种不同的方法：首先，我们利用预训练的LLMs仅从自然语言提示自动生成配置文件，不提供额外指导。其次，我们为这些模型提供官方文档作为参考基准。我们系统地评估了这些AI生成的配置文件的正确性、一致性和适用性。我们的研究结果揭示了当前LLMs生成的配置文件中存在的重要挑战，并为优化访问控制系统中LLMs的部署提供了有价值的见解。_ 

---
# Understanding Software Engineering Agents Through the Lens of Traceability: An Empirical Study 

**Title (ZH)**: 通过追溯性视角理解软件工程代理：一项实证研究 

**Authors**: Ira Ceka, Saurabh Pujar, Shyam Ramji, Luca Buratti, Gail Kaiser, Baishakhi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2506.08311)  

**Abstract**: With the advent of large language models (LLMs), software engineering agents (SWE agents) have emerged as a powerful paradigm for automating a range of software tasks -- from code generation and repair to test case synthesis. These agents operate autonomously by interpreting user input and responding to environmental feedback. While various agent architectures have demonstrated strong empirical performance, the internal decision-making worfklows that drive their behavior remain poorly understood. Deeper insight into these workflows hold promise for improving both agent reliability and efficiency. In this work, we present the first systematic study of SWE agent behavior through the lens of execution traces. Our contributions are as follows: (1) we propose the first taxonomy of decision-making pathways across five representative agents; (2) using this taxonomy, we identify three core components essential to agent success -- bug localization, patch generation, and reproduction test generation -- and study each in depth; (3) we study the impact of test generation on successful patch production; and analyze strategies that can lead to successful test generation; (4) we further conduct the first large-scale code clone analysis comparing agent-generated and developer-written patches and provide a qualitative study revealing structural and stylistic differences in patch content. Together, these findings offer novel insights into agent design and open avenues for building agents that are both more effective and more aligned with human development practices. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的出现，软件工程代理（SWE代理）已成为自动化一系列软件任务的强大范式——从代码生成和修复到测试案例合成。这些代理通过解释用户输入并根据环境反馈自主运行。尽管各种代理架构在实证性能上表现出色，但驱动其行为的内部决策工作流程仍不清楚。对这些工作流程的更深入理解有望提高代理的可靠性和效率。在本研究中，我们通过执行轨迹的视角首次系统研究了SWE代理的行为。我们的贡献如下：（1）我们首次提出了一种涵盖五种代表性代理的决策路径分类；（2）利用这种分类，我们确定了三个关键组件是代理成功的关键——错误定位、补丁生成和再生测试生成，并深入研究了每个组件；（3）我们研究了测试生成对成功补丁生成的影响，并分析了可能导致成功测试生成的策略；（4）我们进行了首次大规模代码克隆分析，比较了代理生成的补丁和开发者编写的补丁，并提供了定性的研究结果，揭示了补丁内容在结构和风格上的差异。这些发现提供了关于代理设计的新见解，并为构建更有效且更符合人类开发实践的代理开辟了途径。 

---
# Learnable Spatial-Temporal Positional Encoding for Link Prediction 

**Title (ZH)**: 可学习的时空位置编码用于链接预测 

**Authors**: Katherine Tieu, Dongqi Fu, Zihao Li, Ross Maciejewski, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2506.08309)  

**Abstract**: Accurate predictions rely on the expressiveness power of graph deep learning frameworks like graph neural networks and graph transformers, where a positional encoding mechanism has become much more indispensable in recent state-of-the-art works to record the canonical position information. However, the current positional encoding is limited in three aspects: (1) most positional encoding methods use pre-defined, and fixed functions, which are inadequate to adapt to the complex attributed graphs; (2) a few pioneering works proposed the learnable positional encoding but are still limited to the structural information, not considering the real-world time-evolving topological and feature information; (3) most positional encoding methods are equipped with transformers' attention mechanism to fully leverage their capabilities, where the dense or relational attention is often unaffordable on large-scale structured data. Hence, we aim to develop Learnable Spatial-Temporal Positional Encoding in an effective and efficient manner and propose a simple temporal link prediction model named L-STEP. Briefly, for L-STEP, we (1) prove the proposed positional learning scheme can preserve the graph property from the spatial-temporal spectral viewpoint, (2) verify that MLPs can fully exploit the expressiveness and reach transformers' performance on that encoding, (3) change different initial positional encoding inputs to show robustness, (4) analyze the theoretical complexity and obtain less empirical running time than SOTA, and (5) demonstrate its temporal link prediction out-performance on 13 classic datasets and with 10 algorithms in both transductive and inductive settings using 3 different sampling strategies. Also, \name\ obtains the leading performance in the newest large-scale TGB benchmark. Our code is available at this https URL. 

**Abstract (ZH)**: 可学习的空间-时间位置编码及其在L-STEP时间链接预测模型中的应用 

---
# SEMA: a Scalable and Efficient Mamba like Attention via Token Localization and Averaging 

**Title (ZH)**: SEMA：一种基于令牌本地化和平均的可扩展且高效的类似Mamba注意力机制 

**Authors**: Nhat Thanh Tran, Fanghui Xue, Shuai Zhang, Jiancheng Lyu, Yunling Zheng, Yingyong Qi, Jack Xin  

**Link**: [PDF](https://arxiv.org/pdf/2506.08297)  

**Abstract**: Attention is the critical component of a transformer. Yet the quadratic computational complexity of vanilla full attention in the input size and the inability of its linear attention variant to focus have been challenges for computer vision tasks. We provide a mathematical definition of generalized attention and formulate both vanilla softmax attention and linear attention within the general framework. We prove that generalized attention disperses, that is, as the number of keys tends to infinity, the query assigns equal weights to all keys. Motivated by the dispersion property and recent development of Mamba form of attention, we design Scalable and Efficient Mamba like Attention (SEMA) which utilizes token localization to avoid dispersion and maintain focusing, complemented by theoretically consistent arithmetic averaging to capture global aspect of attention. We support our approach on Imagenet-1k where classification results show that SEMA is a scalable and effective alternative beyond linear attention, outperforming recent vision Mamba models on increasingly larger scales of images at similar model parameter sizes. 

**Abstract (ZH)**: 通用注意力是一种变压器的关键组件。然而，标准全注意力在输入大小上的二次计算复杂度及其线性注意力变体难以聚焦的问题，阻碍了其在计算机视觉任务中的应用。我们提供了通用注意力的数学定义，并将标准softmax注意力和线性注意力统一在一般框架内。我们证明了通用注意力具有分散性，即随着键的数量趋于无穷，查询将等权重地分配给所有键。受分散性性质及Mamba形式注意力的最新进展启发，我们设计了基于令牌定位的可扩展高效Mamba类似注意力（SEMA），利用理论一致的算术平均来捕获注意力的全局特性，并避免分散性以保持聚焦。在ImageNet-1k上，分类结果表明SEMA是一种可扩展且有效的替代方案，在相似的模型参数量下，其在越来越大的图像规模上优于最近的视觉Mamba模型。 

---
# Seeing Voices: Generating A-Roll Video from Audio with Mirage 

**Title (ZH)**: 听见 vozces：从音频生成A-roll视频的Mirage方法 

**Authors**: Aditi Sundararaman, Amogh Adishesha, Andrew Jaegle, Dan Bigioi, Hyoung-Kyu Song, Jon Kyl, Justin Mao, Kevin Lan, Mojtaba Komeili, ShahRukh Athar, Sheila Babayan, Stanislau Beliasau, William Buchwalter  

**Link**: [PDF](https://arxiv.org/pdf/2506.08279)  

**Abstract**: From professional filmmaking to user-generated content, creators and consumers have long recognized that the power of video depends on the harmonious integration of what we hear (the video's audio track) with what we see (the video's image sequence). Current approaches to video generation either ignore sound to focus on general-purpose but silent image sequence generation or address both visual and audio elements but focus on restricted application domains such as re-dubbing. We introduce Mirage, an audio-to-video foundation model that excels at generating realistic, expressive output imagery from scratch given an audio input. When integrated with existing methods for speech synthesis (text-to-speech, or TTS), Mirage results in compelling multimodal video. When trained on audio-video footage of people talking (A-roll) and conditioned on audio containing speech, Mirage generates video of people delivering a believable interpretation of the performance implicit in input audio. Our central technical contribution is a unified method for training self-attention-based audio-to-video generation models, either from scratch or given existing weights. This methodology allows Mirage to retain generality as an approach to audio-to-video generation while producing outputs of superior subjective quality to methods that incorporate audio-specific architectures or loss components specific to people, speech, or details of how images or audio are captured. We encourage readers to watch and listen to the results of Mirage for themselves (see paper and comments for links). 

**Abstract (ZH)**: 从专业 filmmaking 到用户生成内容，创作人和消费者长期认识到，视频的力量在于所听到的声音（音轨）与所看到的画面（图像序列）的和谐整合。当前的视频生成方法要么忽略声音而专注于无声音的图像序列生成，要么同时处理视觉和音频元素但局限于如重新配音等有限的应用领域。我们介绍了 Mirage，一个从音频生成逼真、具表现力的输出图像的基座模型。当与现有的语音合成方法（文本转语音，或 TTS）结合时，Mirage 会产生引人入胜的多模态视频。当使用人们谈话的音视频素材（A- roll）进行训练，并以包含演讲的音频为条件时，Mirage 生成了人们表现输入音频中隐含表演的真实再现的视频。我们的主要技术贡献是一种统一的方法，用于训练基于自注意力的音频到视频生成模型，无论是从头训练还是在现有权重的基础上训练。这种方法使 Mirage 能够保留其作为音频到视频生成方法的通用性，同时生成感官质量优于包含特定于音频的架构或特定于人、语音或图像和音频捕获细节的损失组件的方法的输出。我们鼓励读者亲自观看和听取 Mirage 的结果（详见论文和评论中的链接）。 

---
# Instruction-Tuned Video-Audio Models Elucidate Functional Specialization in the Brain 

**Title (ZH)**: 指令调优的视频-音频模型阐明大脑的功能专门化 

**Authors**: Subba Reddy Oota, Khushbu Pahwa, Prachi Jindal, Satya Sai Srinath Namburi, Maneesh Singh, Tanmoy Chakraborty, Bapi S. Raju, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.08277)  

**Abstract**: Recent voxel-wise multimodal brain encoding studies have shown that multimodal large language models (MLLMs) exhibit a higher degree of brain alignment compared to unimodal models in both unimodal and multimodal stimulus settings. More recently, instruction-tuned multimodal models have shown to generate task-specific representations that align strongly with brain activity. However, prior work evaluating the brain alignment of MLLMs has primarily focused on unimodal settings or relied on non-instruction-tuned multimodal models for multimodal stimuli. To address this gap, we investigated brain alignment, that is, measuring the degree of predictivity of neural activity recorded while participants were watching naturalistic movies (video along with audio) with representations derived from MLLMs. We utilized instruction-specific embeddings from six video and two audio instruction-tuned MLLMs. Experiments with 13 video task-specific instructions show that instruction-tuned video MLLMs significantly outperform non-instruction-tuned multimodal (by 15%) and unimodal models (by 20%). Our evaluation of MLLMs for both video and audio tasks using language-guided instructions shows clear disentanglement in task-specific representations from MLLMs, leading to precise differentiation of multimodal functional processing in the brain. We also find that MLLM layers align hierarchically with the brain, with early sensory areas showing strong alignment with early layers, while higher-level visual and language regions align more with middle to late layers. These findings provide clear evidence for the role of task-specific instructions in improving the alignment between brain activity and MLLMs, and open new avenues for mapping joint information processing in both the systems. We make the code publicly available [this https URL]. 

**Abstract (ZH)**: Recent 多模态脑编码研究表明，多模态大型语言模型（MLLMs）在单模态和多模态刺激设置中都比单模态模型更接近大脑活动。最近，指令调优的多模态模型展示出能生成与脑活动强烈对应的任伺特异性表示。然而，之前评估MLLMs脑部对齐的工作主要集中在单模态设置上，或者依赖于未指令调优的多模态模型来处理多模态刺激。为解决这一不足，我们探讨了脑部对齐，即通过使用来自六个视频和两个音频指令调优的MLLMs表示，在参与者观看自然电影（视频配以音频）时记录的大脑活动预测度进行测量。实验使用13种视频任伺特异性指令表明，指令调优的视频MLLMs显著优于未指令调优的多模态（高15%）和单模态模型（高20%）。我们使用语言指导的指令评估MLLMs在视频和音频任务中的表现，显示出MLLMs在任务特异性表示上的清晰分离，促进了对接多模态脑功能处理的精确区分。我们还发现，MLLMs层按层次结构与大脑对齐，早期感觉区域与早期层强对齐，而高级视觉和语言区域则更多与中间到晚期层对齐。这些发现提供了任务特异性指令在提高脑活动与MLLMs对齐方面的明确证据，并为映射系统中联合信息处理开辟了新途径。我们公开发布了相关代码 [this https URL]。 

---
# Sparse Interpretable Deep Learning with LIES Networks for Symbolic Regression 

**Title (ZH)**: 基于LIES网络的稀疏可解释深度学习符号回归 

**Authors**: Mansooreh Montazerin, Majd Al Aawar, Antonio Ortega, Ajitesh Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2506.08267)  

**Abstract**: Symbolic regression (SR) aims to discover closed-form mathematical expressions that accurately describe data, offering interpretability and analytical insight beyond standard black-box models. Existing SR methods often rely on population-based search or autoregressive modeling, which struggle with scalability and symbolic consistency. We introduce LIES (Logarithm, Identity, Exponential, Sine), a fixed neural network architecture with interpretable primitive activations that are optimized to model symbolic expressions. We develop a framework to extract compact formulae from LIES networks by training with an appropriate oversampling strategy and a tailored loss function to promote sparsity and to prevent gradient instability. After training, it applies additional pruning strategies to further simplify the learned expressions into compact formulae. Our experiments on SR benchmarks show that the LIES framework consistently produces sparse and accurate symbolic formulae outperforming all baselines. We also demonstrate the importance of each design component through ablation studies. 

**Abstract (ZH)**: 符号回归（SR）的目标是发现能够准确描述数据的闭式数学表达式，提供超越标准黑盒模型的可解释性和分析洞察。现有的SR方法通常依赖于基于群体的搜索或自回归建模，这在可扩展性和符号一致性方面存在挑战。我们引入了LIES（对数、恒等、指数、正弦）架构，这是一种具有可解释基础激活的固定神经网络架构，并通过优化来建模符号表达式。我们开发了一个框架，通过合适的过采样策略和特定的损失函数来提取紧凑公式，该框架促进稀疏性并防止梯度不稳定性。训练后，应用额外的剪枝策略进一步简化学习到的表达式为紧凑公式。我们在SR基准上的实验表明，LIES框架始终生成比所有基线更好的稀疏且准确的符号公式。我们还通过消融研究展示了每个设计组件的重要性。 

---
# Reinforcement Learning from Human Feedback with High-Confidence Safety Constraints 

**Title (ZH)**: 基于高置信度安全约束的人工反馈强化学习 

**Authors**: Yaswanth Chittepu, Blossom Metevier, Will Schwarzer, Austin Hoag, Scott Niekum, Philip S. Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2506.08266)  

**Abstract**: Existing approaches to language model alignment often treat safety as a tradeoff against helpfulness, which can lead to unacceptable responses in sensitive domains. To ensure reliable performance in such settings, we propose High-Confidence Safe Reinforcement Learning from Human Feedback (HC-RLHF), a method that provides high-confidence safety guarantees while maximizing helpfulness. Similar to previous methods, HC-RLHF explicitly decouples human preferences into helpfulness and harmlessness (safety), which are learned by training a reward model and a cost model, respectively. It then employs a two-step process to find safe solutions. In the first step, it optimizes the reward function under an intentionally pessimistic version of the cost constraint. In the second step, the trained model undergoes a safety test to verify whether its performance stays within an upper-confidence bound of the actual cost constraint. We provide a theoretical analysis of HC-RLHF, including proof that it will not return an unsafe solution with a probability greater than a user-specified threshold. For our empirical analysis, we apply HC-RLHF to align three different language models (Qwen2-1.5B, Qwen2.5-3B, and LLaMa3.2-3B) with human preferences. Our results demonstrate that HC-RLHF produces safe models with high probability and can improve harmlessness and helpfulness compared to previous methods. 

**Abstract (ZH)**: 基于人类反馈的高置信度安全强化学习（HC-RLHF）：在保证安全性的同时最大化帮助性 

---
# Automatic Generation of Inference Making Questions for Reading Comprehension Assessments 

**Title (ZH)**: 自动生成推理题 questão 用于阅读理解评估 

**Authors**: Wanjing Anya Ma, Michael Flor, Zuowei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08260)  

**Abstract**: Inference making is an essential but complex skill in reading comprehension (RC). Some inferences require resolving references across sentences, and some rely on using prior knowledge to fill in the detail that is not explicitly written in the text. Diagnostic RC questions can help educators provide more effective and targeted reading instruction and interventions for school-age students. We introduce a taxonomy of inference types for RC and use it to analyze the distribution of items within a diagnostic RC item bank. Next, we present experiments using GPT-4o to generate bridging-inference RC items for given reading passages via few-shot prompting, comparing conditions with and without chain-of-thought prompts. Generated items were evaluated on three aspects: overall item quality, appropriate inference type, and LLM reasoning, achieving high inter-rater agreements above 0.90. Our results show that GPT-4o produced 93.8% good-quality questions suitable for operational use in grade 3-12 contexts; however, only 42.6% of the generated questions accurately matched the targeted inference type. We conclude that combining automatic item generation with human judgment offers a promising path toward scalable, high-quality diagnostic RC assessments. 

**Abstract (ZH)**: 推理能力是阅读理解中一项重要但复杂的技能。一些推理需要解决句子间的指代问题，而另一些则依赖于利用先验知识填补文本中未明确写出的细节。诊断性阅读理解问题可以帮助教育工作者为学龄学生提供更有效和针对性的阅读指导和干预。我们引入了阅读理解推理类型的分类体系，并利用其分析诊断性阅读理解题库中各题目的分布情况。接着，我们使用GPT-4o通过少量示例提示生成连接推理型阅读理解题目，并通过具和不具思维链提示条件进行对比实验。生成的题目在整体质量、合适的推理类型和LLM推理方面获得了超过0.90的一致性评价。结果显示，GPT-4o生成了适用于3至12年级诊断性阅读理解评估的高质量题目，占93.8%；然而，仅有42.6%的生成题目准确匹配了目标推理类型。我们的研究结论表明，结合自动题目生成与人工判断是实现大规模、高质量诊断性阅读理解评估的一种有前景的方法。 

---
# Highly Compressed Tokenizer Can Generate Without Training 

**Title (ZH)**: 高压缩词元化可以生成无需训练 

**Authors**: L. Lao Beyer, T. Li, X. Chen, S. Karaman, K. He  

**Link**: [PDF](https://arxiv.org/pdf/2506.08257)  

**Abstract**: Commonly used image tokenizers produce a 2D grid of spatially arranged tokens. In contrast, so-called 1D image tokenizers represent images as highly compressed one-dimensional sequences of as few as 32 discrete tokens. We find that the high degree of compression achieved by a 1D tokenizer with vector quantization enables image editing and generative capabilities through heuristic manipulation of tokens, demonstrating that even very crude manipulations -- such as copying and replacing tokens between latent representations of images -- enable fine-grained image editing by transferring appearance and semantic attributes. Motivated by the expressivity of the 1D tokenizer's latent space, we construct an image generation pipeline leveraging gradient-based test-time optimization of tokens with plug-and-play loss functions such as reconstruction or CLIP similarity. Our approach is demonstrated for inpainting and text-guided image editing use cases, and can generate diverse and realistic samples without requiring training of any generative model. 

**Abstract (ZH)**: 常用的图像分词器产生二维排列的空间分词网格。相比之下，所谓的1D图像分词器将图像表示为高度压缩的一维分词序列，最多仅包含32个离散分词。我们发现，通过向量量化实现的1D分词器的高度压缩程度，使其能够通过直觉操作分词实现图像编辑和生成能力，表明即使是非常粗糙的操作——如在图像的潜在表示之间复制和替换分词——也能通过转移外观和语义属性实现精细的图像编辑。受1D分词器潜在空间表达能力的启发，我们构建了一种图像生成管道，利用基于梯度的测试时分词优化及即插即用损失函数（如重建或CLIP相似性）进行操作。我们的方法在图像修复和文本引导的图像编辑应用中得到展示，并能在无需训练任何生成模型的情况下生成多样且真实的样本。 

---
# SHIELD: Secure Hypernetworks for Incremental Expansion Learning Defense 

**Title (ZH)**: SHIELD：安全超网络用于增量扩展学习防御 

**Authors**: Patryk Krukowski, Łukasz Gorczyca, Piotr Helm, Kamil Książek, Przemysław Spurek  

**Link**: [PDF](https://arxiv.org/pdf/2506.08255)  

**Abstract**: Traditional deep neural networks suffer from several limitations, including catastrophic forgetting. When models are adapted to new datasets, they tend to quickly forget previously learned knowledge. Another significant issue is the lack of robustness to even small perturbations in the input data. In practice, we can often easily perform adversarial attacks and change the network's predictions, adding minimal noise to the input. Dedicated architectures and training procedures can solve each of the above problems separately. Unfortunately, currently, no model can simultaneously address both catastrophic forgetting and vulnerability to adversarial attacks. We introduce SHIELD (Secure Hypernetworks for Incremental Expansion and Learning Defense), a novel approach that integrates a hypernetwork-based continual learning approach with interval arithmetic. SHIELD use the hypernetwork to transfer trainable task embedding vectors into the weights of a target model dedicated to specific data. This paradigm allows for the dynamic generation of separate networks for each subtask, while the hypernetwork aggregates and analyzes information across all tasks. The target model takes in the input a data sample with a defined interval range, and by creating a hypercube, produces a prediction for the given range. Therefore, such target models provide strict guarantees against all possible attacks for data samples within the interval range. Our approach enhances security without sacrificing network adaptability, addressing the overlooked challenge of safety in continual learning. 

**Abstract (ZH)**: SHIELD：基于超网络的持续学习和安全防御新方法 

---
# Parameter-free approximate equivariance for tasks with finite group symmetry 

**Title (ZH)**: 具有有限群对称性的任务的参数自由近似等变性 

**Authors**: Riccardo Ali, Pietro Liò, Jamie Vicary  

**Link**: [PDF](https://arxiv.org/pdf/2506.08244)  

**Abstract**: Equivariant neural networks incorporate symmetries through group actions, embedding them as an inductive bias to improve performance on a wide variety of tasks. However, existing equivariant methods can be computationally intensive, with high parameter counts, and are often tied to a specific architecture. We propose a simple zero-parameter approach that imposes approximate equivariance for a finite group in the latent representation, as an additional term in the loss function. We conduct experiments which allow the network to learn a group representation on the latent space, and show in every case it prefers to learn the regular representation. Fixing this action on the latent space, this yields a simple method to impose approximate equivariance as an additional loss penalty. We benchmark our approach on three datasets and compare it against several existing equivariant methods, showing that in many cases it achieves similar or better performance for a fraction of the parameters. 

**Abstract (ZH)**: 不变神经网络通过集团动作 Incorporate 不变性，将其作为归纳偏见嵌入，以提高各种任务上的性能。然而，现有的不变方法可能计算密集，参数数量高，并且通常与特定架构绑定。我们提出了一种简单的无参方法，在潜在表示中对有限集团施加近似不变性，作为损失函数中的附加项。我们进行了实验，使网络在潜在空间中学习一个集团表示，并在每种情况下都显示它倾向于学习正则表示。固定潜在空间中的此动作，这提供了一种简单的方法来将近似不变性作为附加损失惩罚来施加。我们将该方法在三个数据集上进行基准测试，并与几种现有的不变方法进行比较，结果显示在许多情况下，它以更少的参数实现了相似或更好的性能。 

---
# Can AI Validate Science? Benchmarking LLMs for Accurate Scientific Claim $\rightarrow$ Evidence Reasoning 

**Title (ZH)**: AI能验证科学吗？基于准确的科学断言和证据推理对LLM进行基准测试 

**Authors**: Shashidhar Reddy Javaji, Yupeng Cao, Haohang Li, Yangyang Yu, Nikhil Muralidhar, Zining Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08235)  

**Abstract**: Large language models (LLMs) are increasingly being used for complex research tasks such as literature review, idea generation, and scientific paper analysis, yet their ability to truly understand and process the intricate relationships within complex research papers, such as the logical links between claims and supporting evidence remains largely unexplored. In this study, we present CLAIM-BENCH, a comprehensive benchmark for evaluating LLMs' capabilities in scientific claim-evidence extraction and validation, a task that reflects deeper comprehension of scientific argumentation. We systematically compare three approaches which are inspired by divide and conquer approaches, across six diverse LLMs, highlighting model-specific strengths and weaknesses in scientific comprehension. Through evaluation involving over 300 claim-evidence pairs across multiple research domains, we reveal significant limitations in LLMs' ability to process complex scientific content. Our results demonstrate that closed-source models like GPT-4 and Claude consistently outperform open-source counterparts in precision and recall across claim-evidence identification tasks. Furthermore, strategically designed three-pass and one-by-one prompting approaches significantly improve LLMs' abilities to accurately link dispersed evidence with claims, although this comes at increased computational cost. CLAIM-BENCH sets a new standard for evaluating scientific comprehension in LLMs, offering both a diagnostic tool and a path forward for building systems capable of deeper, more reliable reasoning across full-length papers. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在文学综述、想法生成和科学论文分析等复杂研究任务中的应用越来越广泛，但它们在真正理解并处理复杂研究论文中的 intricate 关系，如论点与其支持证据之间的逻辑联系方面的能力尚未被充分探索。在本研究中，我们提出了 CLAIM-BENCH，一个全面的基准，用于评估 LLMs 在科学论据提取和验证方面的能力，这一任务反映了对科学研究论证更深层次的理解。我们系统地比较了三种基于分而治之方法的思想，并在六个不同的 LLMs 上进行了比较，突出了模型在科学研究理解上的特定优势和劣势。通过跨越多个研究领域的超过 300 个论据-证据对的评估，我们揭示了 LLMs 在处理复杂科学内容方面的显著局限性。我们的结果显示，闭源模型如 GPT-4 和 Claude 在论据-证据识别任务中的准确率和召回率上始终优于开源模型。此外，精心设计的三遍和逐个提示策略显著提高了 LLMs 将分散的证据与论点准确关联的能力，尽管这会导致计算成本的增加。CLAIM-BENCH 为评估 LLMs 在科学研究理解方面的标准设定了一条新途径，既是一个诊断工具，也是构建能够进行更深入、更可靠推理的系统的方法之一。 

---
# Compound AI Systems Optimization: A Survey of Methods, Challenges, and Future Directions 

**Title (ZH)**: 复合AI系统优化：方法、挑战及未来方向 

**Authors**: Yu-Ang Lee, Guan-Ting Yi, Mei-Yi Liu, Jui-Chao Lu, Guan-Bo Yang, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.08234)  

**Abstract**: Recent advancements in large language models (LLMs) and AI systems have led to a paradigm shift in the design and optimization of complex AI workflows. By integrating multiple components, compound AI systems have become increasingly adept at performing sophisticated tasks. However, as these systems grow in complexity, new challenges arise in optimizing not only individual components but also their interactions. While traditional optimization methods such as supervised fine-tuning (SFT) and reinforcement learning (RL) remain foundational, the rise of natural language feedback introduces promising new approaches, especially for optimizing non-differentiable systems. This paper provides a systematic review of recent progress in optimizing compound AI systems, encompassing both numerical and language-based techniques. We formalize the notion of compound AI system optimization, classify existing methods along several key dimensions, and highlight open research challenges and future directions in this rapidly evolving field. A list of surveyed papers is publicly available at this https URL. 

**Abstract (ZH)**: 最近大型语言模型（LLMs）和AI系统的进展引发了复杂AI工作流设计与优化的范式转变。通过集成多个组件，复合AI系统在执行复杂任务方面日益熟练。然而，随着这些系统的复杂性增加，优化不仅是个别组件，还包括它们之间的交互的新挑战也随之出现。尽管传统的优化方法如监督微调（SFT）和强化学习（RL）仍然是基础性的，但自然语言反馈的发展引入了有前景的新方法，特别适用于优化非可微系统。本文对复合AI系统优化的最新进展进行了系统性综述，涵盖了数值技术和语言技术方法。我们正式定义了复合AI系统优化的概念，按多个关键维度对现有方法进行了分类，并指出了该快速发展的领域中的开放研究挑战和未来方向。所调研论文列表在此处公开：这个 https URL。 

---
# Ensuring Reliability of Curated EHR-Derived Data: The Validation of Accuracy for LLM/ML-Extracted Information and Data (VALID) Framework 

**Title (ZH)**: 确保 curated EHR衍生数据的可靠性：LLM/ML提取信息和数据准确性验证框架（VALID） 

**Authors**: Melissa Estevez, Nisha Singh, Lauren Dyson, Blythe Adamson, Qianyu Yuan, Megan W. Hildner, Erin Fidyk, Olive Mbah, Farhad Khan, Kathi Seidl-Rathkopf, Aaron B. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2506.08231)  

**Abstract**: Large language models (LLMs) are increasingly used to extract clinical data from electronic health records (EHRs), offering significant improvements in scalability and efficiency for real-world data (RWD) curation in oncology. However, the adoption of LLMs introduces new challenges in ensuring the reliability, accuracy, and fairness of extracted data, which are essential for research, regulatory, and clinical applications. Existing quality assurance frameworks for RWD and artificial intelligence do not fully address the unique error modes and complexities associated with LLM-extracted data. In this paper, we propose a comprehensive framework for evaluating the quality of clinical data extracted by LLMs. The framework integrates variable-level performance benchmarking against expert human abstraction, automated verification checks for internal consistency and plausibility, and replication analyses comparing LLM-extracted data to human-abstracted datasets or external standards. This multidimensional approach enables the identification of variables most in need of improvement, systematic detection of latent errors, and confirmation of dataset fitness-for-purpose in real-world research. Additionally, the framework supports bias assessment by stratifying metrics across demographic subgroups. By providing a rigorous and transparent method for assessing LLM-extracted RWD, this framework advances industry standards and supports the trustworthy use of AI-powered evidence generation in oncology research and practice. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地用于从电子健康记录（EHRs）中提取临床数据，为肿瘤学领域的实时数据（RWD）编目提供了显著的规模性和效率性改进。然而，LLMs的应用引入了确保提取数据的可靠性、准确性和公平性的新挑战，这对于研究、监管和临床应用至关重要。现有的RWD和人工智能的质量保证框架并未充分解决LLM提取数据特有的错误模式和复杂性。本文提出了一套全面的框架来评估由LLM提取的临床数据质量。该框架结合了变量级别的性能基准测试、内部一致性与合理性自动验证检查以及将LLM提取的数据与人工摘要数据集或外部标准进行重新分析以识别差异的方法。这种多维度的方法有助于识别最需要改进的变量，系统地检测潜在错误，并确认数据集是否适合作为研究目的的资源。此外，该框架还支持偏倚评估，通过不同人口亚组分类指标。通过提供一种严格和透明的方法来评估LLM提取的RWD，该框架推进了行业标准，并支持在肿瘤学研究和实践中的可信AI证据生成。 

---
# Scaling Laws of Motion Forecasting and Planning -- A Technical Report 

**Title (ZH)**: 运动预测与规划的标度律——技术报告 

**Authors**: Mustafa Baniodeh, Kratarth Goel, Scott Ettinger, Carlos Fuertes, Ari Seff, Tim Shen, Cole Gulino, Chenjie Yang, Ghassen Jerfel, Dokook Choe, Rui Wang, Vinutha Kallem, Sergio Casas, Rami Al-Rfou, Benjamin Sapp, Dragomir Anguelov  

**Link**: [PDF](https://arxiv.org/pdf/2506.08228)  

**Abstract**: We study the empirical scaling laws of a family of encoder-decoder autoregressive transformer models on the task of joint motion forecasting and planning in the autonomous driving domain. Using a 500 thousand hours driving dataset, we demonstrate that, similar to language modeling, model performance improves as a power-law function of the total compute budget, and we observe a strong correlation between model training loss and model evaluation metrics. Most interestingly, closed-loop metrics also improve with scaling, which has important implications for the suitability of open-loop metrics for model development and hill climbing. We also study the optimal scaling of the number of transformer parameters and the training data size for a training compute-optimal model. We find that as the training compute budget grows, optimal scaling requires increasing the model size 1.5x as fast as the dataset size. We also study inference-time compute scaling, where we observe that sampling and clustering the output of smaller models makes them competitive with larger models, up to a crossover point beyond which a larger models becomes more inference-compute efficient. Overall, our experimental results demonstrate that optimizing the training and inference-time scaling properties of motion forecasting and planning models is a key lever for improving their performance to address a wide variety of driving scenarios. Finally, we briefly study the utility of training on general logged driving data of other agents to improve the performance of the ego-agent, an important research area to address the scarcity of robotics data for large capacity models training. 

**Abstract (ZH)**: 我们研究了一组编码器-解码器自回归 transformer 模型在自动驾驶领域联合运动预测与规划任务中的经验标度律。利用包含五百多万小时驾驶数据的 datasets，我们表明，类似语言建模，模型性能随着总计算预算的增加呈幂律函数增强，并观察到模型训练损失与模型评估指标间存在强烈相关性。更有趣的是，闭环指标也随标度增强而改善，这对开放环指标在模型开发和优化中的适用性有着重要影响。我们还研究了训练过程中 transformer 参数数量和训练数据规模的最优标度，发现随着训练计算预算的增长，最优标度要求模型规模以比数据集规模快1.5倍的速度增长。我们还研究了推理时计算资源的标度，发现通过采样和聚类较小模型的输出可以使它们与较大模型竞争，直到某个交叉点之后，较大模型在推理计算效率上更具优势。总体而言，我们的实验结果表明，优化运动预测与规划模型的训练和推理时标度特性是提升其性能的关键手段，以应对各种驾驶场景。最后，我们简要研究了利用其他代理的一般记录驾驶数据进行训练以提高自主代理性能的实用性，这是解决大规模模型训练中机器人数据稀缺问题的重要研究领域。 

---
# A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation 

**Title (ZH)**: 全面研究解码器型大语言模型在文本到图像生成中的应用 

**Authors**: Andrew Z. Wang, Songwei Ge, Tero Karras, Ming-Yu Liu, Yogesh Balaji  

**Link**: [PDF](https://arxiv.org/pdf/2506.08210)  

**Abstract**: Both text-to-image generation and large language models (LLMs) have made significant advancements. However, many text-to-image models still employ the somewhat outdated T5 and CLIP as their text encoders. In this work, we investigate the effectiveness of using modern decoder-only LLMs as text encoders for text-to-image diffusion models. We build a standardized training and evaluation pipeline that allows us to isolate and evaluate the effect of different text embeddings. We train a total of 27 text-to-image models with 12 different text encoders to analyze the critical aspects of LLMs that could impact text-to-image generation, including the approaches to extract embeddings, different LLMs variants, and model sizes. Our experiments reveal that the de facto way of using last-layer embeddings as conditioning leads to inferior performance. Instead, we explore embeddings from various layers and find that using layer-normalized averaging across all layers significantly improves alignment with complex prompts. Most LLMs with this conditioning outperform the baseline T5 model, showing enhanced performance in advanced visio-linguistic reasoning skills. 

**Abstract (ZH)**: 现代解码器大型语言模型作为文本编码器在文本到图像生成中的 effectiveness 研究 

---
# Surgeon Style Fingerprinting and Privacy Risk Quantification via Discrete Diffusion Models in a Vision-Language-Action Framework 

**Title (ZH)**: 基于视觉-语言-动作框架的外科医生风格指纹识别与隐私风险量化 

**Authors**: Huixin Zhan, Jason H. Moore  

**Link**: [PDF](https://arxiv.org/pdf/2506.08185)  

**Abstract**: Surgeons exhibit distinct operating styles due to differences in training, experience, and motor behavior - yet current AI systems often ignore this personalization signal. We propose a novel approach to model fine-grained, surgeon-specific fingerprinting in robotic surgery using a discrete diffusion framework integrated with a vision-language-action (VLA) pipeline. Our method formulates gesture prediction as a structured sequence denoising task, conditioned on multimodal inputs including endoscopic video, surgical intent language, and a privacy-aware embedding of surgeon identity and skill. Personalized surgeon fingerprinting is encoded through natural language prompts using third-party language models, allowing the model to retain individual behavioral style without exposing explicit identity. We evaluate our method on the JIGSAWS dataset and demonstrate that it accurately reconstructs gesture sequences while learning meaningful motion fingerprints unique to each surgeon. To quantify the privacy implications of personalization, we perform membership inference attacks and find that more expressive embeddings improve task performance but simultaneously increase susceptibility to identity leakage. These findings demonstrate that while personalized embeddings improve performance, they also increase vulnerability to identity leakage, revealing the importance of balancing personalization with privacy risk in surgical modeling. Code is available at: this https URL. 

**Abstract (ZH)**: 外科医生由于训练、经验和运动行为的不同表现出独特的手术风格——然而当前的AI系统往往忽略了这一个性化信号。我们提出了一种新的方法，利用离散扩散框架结合视语言行动（VLA）管线，在机器人手术中建模精细、医生特定的指纹。该方法将手势预测形式化为结构化序列去噪任务，基于内窥镜视频、手术意图语言和隐私意识的外科医生身份和技能嵌入的多模态输入。通过第三方语言模型使用自然语言提示进行个性化外科医生指纹编码，使模型能够保留个体行为风格而不泄露明确的身份信息。我们在JIGSAWS数据集上评估了该方法，证明它可以准确重建手势序列并学习每个外科医生独有的有意义的运动指纹。为了量化个性化的隐私影响，我们进行了成员归属推断攻击，发现更具表现力的嵌入提高了任务性能，但同时也增加了身份泄露的风险。这些发现表明，虽然个性化嵌入提高了性能，但也增加了身份泄露的风险，揭示了在手术建模中平衡个性化与隐私风险的重要性。代码可在该链接获取：this https URL。 

---
# Unable to forget: Proactive lnterference Reveals Working Memory Limits in LLMs Beyond Context Length 

**Title (ZH)**: 无法忘记：前向干扰揭示LLM在超出上下文长度之外的工作记忆限制 

**Authors**: Chupei Wang, Jiaqiu Vince Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.08184)  

**Abstract**: Information retrieval in Large Language Models (LLMs) is increasingly recognized as intertwined with generation capabilities rather than mere lookup. While longer contexts are often assumed to improve retrieval, the effects of intra-context interference remain understudied. To address this, we adapt the proactive interference (PI) paradigm from cognitive science, where earlier information disrupts recall of newer updates. In humans, susceptibility to such interference is inversely linked to working memory capacity. We introduce PI-LLM, an evaluation that sequentially streams semantically related key-value updates and queries only the final values. Although these final values are clearly positioned just before the query, LLM retrieval accuracy declines log-linearly toward zero as interference accumulates; errors arise from retrieving previously overwritten values. Attempts to mitigate interference via prompt engineering (e.g., instructing models to ignore earlier input) yield limited success. These findings reveal a fundamental constraint on LLMs' ability to disentangle interference and flexibly manipulate information, suggesting a working memory bottleneck beyond mere context access. This calls for approaches that strengthen models' ability to suppress irrelevant content during retrieval. 

**Abstract (ZH)**: 在大型语言模型中基于干扰的信息检索 

---
# Repeton: Structured Bug Repair with ReAct-Guided Patch-and-Test Cycles 

**Title (ZH)**: Repeton：基于ReAct引导的结构化漏洞修复循环 

**Authors**: Nguyen Phu Vinh, Anh Chung Hoang, Chris Ngo, Truong-Son Hy  

**Link**: [PDF](https://arxiv.org/pdf/2506.08173)  

**Abstract**: Large Language Models (LLMs) have shown strong capabilities in code generation and comprehension, yet their application to complex software engineering tasks often suffers from low precision and limited interpretability. We present Repeton, a fully open-source framework that leverages LLMs for precise and automated code manipulation in real-world Git repositories. Rather than generating holistic fixes, Repeton operates through a structured patch-and-test pipeline: it iteratively diagnoses issues, proposes code changes, and validates each patch through automated testing. This stepwise process is guided by lightweight heuristics and development tools, avoiding reliance on embedding-based retrieval systems. Evaluated on the SWE-bench Lite benchmark, our method shows good performance compared to RAG-based methods in both patch validity and interpretability. By decomposing software engineering tasks into modular, verifiable stages, Repeton provides a practical path toward scalable and transparent autonomous debugging. 

**Abstract (ZH)**: 大型语言模型（LLMs）在代码生成和理解方面展现了强大的能力，但在应用于复杂的软件工程任务时往往精度较低且可解释性有限。我们提出了一种名为Repeton的全开源框架，利用LLMs进行精确且自动化的代码操作，应用于实际的Git仓库。Repeton 不是生成整体修复方案，而是通过结构化的补丁和测试管道进行操作：它迭代诊断问题，提出代码更改，并通过自动化测试验证每个补丁。这一逐步过程由轻量级的启发式方法和开发工具引导，避免依赖基于嵌入式检索系统。在SWE-bench Lite基准上评估，我们的方法在补丁有效性和可解释性方面与基于RAG的方法相比表现良好。通过将软件工程任务分解为模块化且可验证的阶段，Repeton 提供了一条实现可扩展和透明的自动调试的可行途径。 

---
# Worst-Case Symbolic Constraints Analysis and Generalisation with Large Language Models 

**Title (ZH)**: 最坏情况符号约束分析与大型语言模型的一般化 

**Authors**: Daniel Koh, Yannic Noller, Corina S. Pasareanu, Adrians Skapars, Youcheng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.08171)  

**Abstract**: Large language models (LLMs) have been successfully applied to a variety of coding tasks, including code generation, completion, and repair. However, more complex symbolic reasoning tasks remain largely unexplored by LLMs. This paper investigates the capacity of LLMs to reason about worst-case executions in programs through symbolic constraints analysis, aiming to connect LLMs and symbolic reasoning approaches. Specifically, we define and address the problem of worst-case symbolic constraints analysis as a measure to assess the comprehension of LLMs. We evaluate the performance of existing LLMs on this novel task and further improve their capabilities through symbolic reasoning-guided fine-tuning, grounded in SMT (Satisfiability Modulo Theories) constraint solving and supported by a specially designed dataset of symbolic constraints. Experimental results show that our solver-aligned model, WARP-1.0-3B, consistently surpasses size-matched and even much larger baselines, demonstrating that a 3B LLM can recover the very constraints that pin down an algorithm's worst-case behaviour through reinforcement learning methods. These findings suggest that LLMs are capable of engaging in deeper symbolic reasoning, supporting a closer integration between neural network-based learning and formal methods for rigorous program analysis. 

**Abstract (ZH)**: 大型语言模型通过符号约束分析探究最坏情况执行的推理能力：连接大型语言模型和符号推理方法的新任务及强化学习优化 

---
# UniVarFL: Uniformity and Variance Regularized Federated Learning for Heterogeneous Data 

**Title (ZH)**: UnivarFL：异质数据下的均匀性与方差正则化联邦学习 

**Authors**: Sunny Gupta, Nikita Jangid, Amit Sethi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08167)  

**Abstract**: Federated Learning (FL) often suffers from severe performance degradation when faced with non-IID data, largely due to local classifier bias. Traditional remedies such as global model regularization or layer freezing either incur high computational costs or struggle to adapt to feature shifts. In this work, we propose UniVarFL, a novel FL framework that emulates IID-like training dynamics directly at the client level, eliminating the need for global model dependency. UniVarFL leverages two complementary regularization strategies during local training: Classifier Variance Regularization, which aligns class-wise probability distributions with those expected under IID conditions, effectively mitigating local classifier bias; and Hyperspherical Uniformity Regularization, which encourages a uniform distribution of feature representations across the hypersphere, thereby enhancing the model's ability to generalize under diverse data distributions. Extensive experiments on multiple benchmark datasets demonstrate that UniVarFL outperforms existing methods in accuracy, highlighting its potential as a highly scalable and efficient solution for real-world FL deployments, especially in resource-constrained settings. Code: this https URL 

**Abstract (ZH)**: federated学习（FL）在面对非IID数据时往往会遭受严重的性能下降，主要原因是客户端分类器偏差。传统的解决方案如全局模型正则化或层冻结要么计算成本高昂，要么难以应对特征偏移。在本文中，我们提出UniVarFL，这是一种新型的FL框架，能够在客户端直接模拟类似于IID的训练动态，从而消除对全局模型的依赖。UniVarFL在局部训练期间利用两种互补的正则化策略：分类器方差正则化，通过使类内概率分布与在IID条件下的预期对齐，有效缓解了分类器偏差；超球体均匀性正则化，鼓励特征表示在超球体上的均匀分布，从而增强模型在不同数据分布下的泛化能力。在多个基准数据集上的广泛实验表明，UniVarFL在准确率上优于现有方法，突显了其作为高扩展性和高效解决方案的潜力，尤其是在资源受限的环境中部署FL尤其有前景。代码: [这个链接]。 

---
# A Metrics-Oriented Architectural Model to Characterize Complexity on Machine Learning-Enabled Systems 

**Title (ZH)**: 基于机器学习赋能系统复杂性特征的指标导向架构模型 

**Authors**: Renato Cordeiro Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2506.08153)  

**Abstract**: How can the complexity of ML-enabled systems be managed effectively? The goal of this research is to investigate how complexity affects ML-Enabled Systems (MLES). To address this question, this research aims to introduce a metrics-based architectural model to characterize the complexity of MLES. The goal is to support architectural decisions, providing a guideline for the inception and growth of these systems. This paper showcases the first step for creating the metrics-based architectural model: an extension of a reference architecture that can describe MLES to collect their metrics. 

**Abstract (ZH)**: 如何有效管理由ML驱动系统的复杂性？本研究旨在探讨复杂性如何影响由ML驱动系统（MLES）。为了回答这一问题，本研究旨在引入基于度量的架构模型来刻画MLES的复杂性。目标是支持架构决策，为这些系统的诞生和发展提供指导。本文展示了创建基于度量的架构模型的第一步：对参考架构进行扩展，以便描述MLES并收集其度量数据。 

---
# Ego-centric Learning of Communicative World Models for Autonomous Driving 

**Title (ZH)**: 以自我为中心的学习交流世界模型用于自主驾驶 

**Authors**: Hang Wang, Dechen Gao, Junshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08149)  

**Abstract**: We study multi-agent reinforcement learning (MARL) for tasks in complex high-dimensional environments, such as autonomous driving. MARL is known to suffer from the \textit{partial observability} and \textit{non-stationarity} issues. To tackle these challenges, information sharing is often employed, which however faces major hurdles in practice, including overwhelming communication overhead and scalability concerns. By making use of generative AI embodied in world model together with its latent representation, we develop {\it CALL}, \underline{C}ommunic\underline{a}tive Wor\underline{l}d Mode\underline{l}, for MARL, where 1) each agent first learns its world model that encodes its state and intention into low-dimensional latent representation with smaller memory footprint, which can be shared with other agents of interest via lightweight communication; and 2) each agent carries out ego-centric learning while exploiting lightweight information sharing to enrich her world model, and then exploits its generalization capacity to improve prediction for better planning. We characterize the gain on the prediction accuracy from the information sharing and its impact on performance gap. Extensive experiments are carried out on the challenging local trajectory planning tasks in the CARLA platform to demonstrate the performance gains of using \textit{CALL}. 

**Abstract (ZH)**: 基于生成AI的世界模型的多智能体强化学习：COMMWorldModel 

---
# Multilingual Hate Speech Detection in Social Media Using Translation-Based Approaches with Large Language Models 

**Title (ZH)**: 基于大规模语言模型的翻译导向方法在社交媒体多语言仇恨言论检测中应用 

**Authors**: Muhammad Usman, Muhammad Ahmad, M. Shahiki Tash, Irina Gelbukh, Rolando Quintero Tellez, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2506.08147)  

**Abstract**: Social media platforms are critical spaces for public discourse, shaping opinions and community dynamics, yet their widespread use has amplified harmful content, particularly hate speech, threatening online safety and inclusivity. While hate speech detection has been extensively studied in languages like English and Spanish, Urdu remains underexplored, especially using translation-based approaches. To address this gap, we introduce a trilingual dataset of 10,193 tweets in English (3,834 samples), Urdu (3,197 samples), and Spanish (3,162 samples), collected via keyword filtering, with a balanced distribution of 4,849 Hateful and 5,344 Not-Hateful labels. Our methodology leverages attention layers as a precursor to transformer-based models and large language models (LLMs), enhancing feature extraction for multilingual hate speech detection. For non-transformer models, we use TF-IDF for feature extraction. The dataset is benchmarked using state-of-the-art models, including GPT-3.5 Turbo and Qwen 2.5 72B, alongside traditional machine learning models like SVM and other transformers (e.g., BERT, RoBERTa). Three annotators, following rigorous guidelines, ensured high dataset quality, achieving a Fleiss' Kappa of 0.821. Our approach, integrating attention layers with GPT-3.5 Turbo and Qwen 2.5 72B, achieves strong performance, with macro F1 scores of 0.87 for English (GPT-3.5 Turbo), 0.85 for Spanish (GPT-3.5 Turbo), 0.81 for Urdu (Qwen 2.5 72B), and 0.88 for the joint multilingual model (Qwen 2.5 72B). These results reflect improvements of 8.75% in English (over SVM baseline 0.80), 8.97% in Spanish (over SVM baseline 0.78), 5.19% in Urdu (over SVM baseline 0.77), and 7.32% in the joint multilingual model (over SVM baseline 0.82). Our framework offers a robust solution for multilingual hate speech detection, fostering safer digital communities worldwide. 

**Abstract (ZH)**: 社交媒体平台是公共 discourse 的关键空间，塑造意见和社群动态，但其广泛应用放大了有害内容，尤其是仇恨言论，威胁在线安全和包容性。尽管仇恨言论检测在英语和西班牙语等语言中得到了广泛研究，但乌尔都语仍相对未被充分探索，尤其是在基于翻译的方法方面。为解决这一问题，我们引入了一个包含10,193条推文的三语数据集，分别包含3,834条英语、3,197条乌尔都语和3,162条西班牙语样本，通过关键词过滤收集，具有4,849个仇恨和5,344个非仇恨标签的平衡分布。我们的方法利用注意力层作为变压器模型和大型语言模型（LLMs）之前的预处理步骤，增强多语言仇恨言论检测的功能提取。对于非变压器模型，我们使用TF-IDF进行特征提取。该数据集使用包括GPT-3.5 Turbo和Qwen 2.5 72B在内的最新模型进行基准测试，以及传统机器学习模型如SVM和其他变压器（例如BERT、RoBERTa）。三名遵循严格指南的注释者确保了高质量的数据集， Fleiss' Kappa值为0.821。结合注意力层与GPT-3.5 Turbo和Qwen 2.5 72B的方法显示出强大的性能，其中英语（GPT-3.5 Turbo）的宏F1分数为0.87，西班牙语（GPT-3.5 Turbo）为0.85，乌尔都语（Qwen 2.5 72B）为0.81，联合多语言模型（Qwen 2.5 72B）为0.88。这些结果反映了相对于SVM基准的改进，分别为8.75%（英语）、8.97%（西班牙语）、5.19%（乌尔都语）和7.32%（联合多语言模型）。我们的框架为多语言仇恨言论检测提供了稳健的解决方案，推动全球更安全的数字社区建设。 

---
# Nearness of Neighbors Attention for Regression in Supervised Finetuning 

**Title (ZH)**: 邻居近邻attention在监督微调中的回归应用 

**Authors**: Aviad Susman, Mayte Suárez-Fariñas, Joseph T Colonel  

**Link**: [PDF](https://arxiv.org/pdf/2506.08139)  

**Abstract**: It is common in supervised machine learning to combine the feature extraction capabilities of neural networks with the predictive power of traditional algorithms, such as k-nearest neighbors (k-NN) or support vector machines. This procedure involves performing supervised fine-tuning (SFT) on a domain-appropriate feature extractor, followed by training a traditional predictor on the resulting SFT embeddings. When used in this manner, traditional predictors often deliver increased performance over the SFT model itself, despite the fine-tuned feature extractor yielding embeddings specifically optimized for prediction by the neural network's final dense layer. This suggests that directly incorporating traditional algorithms into SFT as prediction layers may further improve performance. However, many traditional algorithms have not been implemented as neural network layers due to their non-differentiable nature and their unique optimization requirements. As a step towards solving this problem, we introduce the Nearness of Neighbors Attention (NONA) regression layer. NONA uses the mechanics of neural network attention and a novel learned attention-masking scheme to yield a differentiable proxy of the k-NN regression algorithm. Results on multiple unstructured datasets show improved performance over both dense layer prediction and k-NN on SFT embeddings for regression. 

**Abstract (ZH)**: 监督机器学习中将神经网络的特征提取能力与传统算法的预测能力结合：一种通过近邻注意力机制改进细调效果的方法 

---
# IGraSS: Learning to Identify Infrastructure Networks from Satellite Imagery by Iterative Graph-constrained Semantic Segmentation 

**Title (ZH)**: IGraSS：通过迭代图约束语义分割识别基础设施网络 

**Authors**: Oishee Bintey Hoque, Abhijin Adiga, Aniruddha Adiga, Siddharth Chaudhary, Madhav V. Marathe, S. S. Ravi, Kirti Rajagopalan, Amanda Wilson, Samarth Swarup  

**Link**: [PDF](https://arxiv.org/pdf/2506.08137)  

**Abstract**: Accurate canal network mapping is essential for water management, including irrigation planning and infrastructure maintenance. State-of-the-art semantic segmentation models for infrastructure mapping, such as roads, rely on large, well-annotated remote sensing datasets. However, incomplete or inadequate ground truth can hinder these learning approaches. Many infrastructure networks have graph-level properties such as reachability to a source (like canals) or connectivity (roads) that can be leveraged to improve these existing ground truth. This paper develops a novel iterative framework IGraSS, combining a semantic segmentation module-incorporating RGB and additional modalities (NDWI, DEM)-with a graph-based ground-truth refinement module. The segmentation module processes satellite imagery patches, while the refinement module operates on the entire data viewing the infrastructure network as a graph. Experiments show that IGraSS reduces unreachable canal segments from around 18% to 3%, and training with refined ground truth significantly improves canal identification. IGraSS serves as a robust framework for both refining noisy ground truth and mapping canal networks from remote sensing imagery. We also demonstrate the effectiveness and generalizability of IGraSS using road networks as an example, applying a different graph-theoretic constraint to complete road networks. 

**Abstract (ZH)**: 精确的沟渠网络测绘对于水资源管理，包括灌溉规划和基础设施维护至关重要。基于最新语义分割模型在基础设施测绘（如道路）中的应用依赖于大量且标注良好的遥感数据集。然而，缺乏或不充分的地面真实数据会阻碍这些学习方法。许多基础设施网络具有图级别特性，如可达性（如沟渠）或连接性（道路），这些特性可以利用以改进现有的地面真实数据。本文提出了一种新颖的迭代框架IGraSS，结合了一个语义分割模块（包含RGB和额外模态数据，如NDWI、DEM）与一个基于图的地面真实数据精炼模块。分割模块处理卫星图像片段，而精炼模块则在整个数据集上运行，将基础设施网络视为图。实验结果显示，IGraSS将未达沟渠段的比例从约18%降低到3%，使用精炼地面真实数据进行训练显著提高了沟渠识别效果。IGraSS作为一个鲁棒框架，可用于精炼噪音地面真实数据并从遥感图像中测绘沟渠网络。我们还通过使用道路网络作为示例，展示了IGraSS的有效性和普适性，并应用不同的图论约束来完成道路网络。 

---
# Benchmarking Pre-Trained Time Series Models for Electricity Price Forecasting 

**Title (ZH)**: 预训练时间序列模型在电价预测中的基准比较 

**Authors**: Timothée Hornek Amir Sartipi, Igor Tchappi, Gilbert Fridgen  

**Link**: [PDF](https://arxiv.org/pdf/2506.08113)  

**Abstract**: Accurate electricity price forecasting (EPF) is crucial for effective decision-making in power trading on the spot market. While recent advances in generative artificial intelligence (GenAI) and pre-trained large language models (LLMs) have inspired the development of numerous time series foundation models (TSFMs) for time series forecasting, their effectiveness in EPF remains uncertain. To address this gap, we benchmark several state-of-the-art pretrained models--Chronos-Bolt, Chronos-T5, TimesFM, Moirai, Time-MoE, and TimeGPT--against established statistical and machine learning (ML) methods for EPF. Using 2024 day-ahead auction (DAA) electricity prices from Germany, France, the Netherlands, Austria, and Belgium, we generate daily forecasts with a one-day horizon. Chronos-Bolt and Time-MoE emerge as the strongest among the TSFMs, performing on par with traditional models. However, the biseasonal MSTL model, which captures daily and weekly seasonality, stands out for its consistent performance across countries and evaluation metrics, with no TSFM statistically outperforming it. 

**Abstract (ZH)**: 准确的电价预测（EPF）对于电力交易现货市场的有效决策至关重要。尽管近期生成式人工智能（GenAI）和预训练大型语言模型（LLMs）的进步激发了许多时间序列基础模型（TSFMs）的发展，但其在EPF中的有效性仍然不确定。为了弥补这一差距，我们将Chronos-Bolt、Chronos-T5、TimesFM、Moirai、Time-MoE和TimeGPT等几种最先进的预训练模型与传统的统计和机器学习（ML）方法进行对比，用于电价预测。基于德国、法国、荷兰、奥地利和比利时的2024天前竞价（DAA）的每日电价数据，我们生成了一天前瞻性的日度预报。Chronos-Bolt和Time-MoE在TSFMs中表现最强，与传统模型相当。然而，能够捕捉日度和周度季节性的二季节模型在各国和各种评价指标下表现出一致性，没有TSFM在其上具有统计上的优越性。 

---
# Hierarchical Lexical Graph for Enhanced Multi-Hop Retrieval 

**Title (ZH)**: 层次词图以增强多跳检索 

**Authors**: Abdellah Ghassel, Ian Robinson, Gabriel Tanase, Hal Cooper, Bryan Thompson, Zhen Han, Vassilis N. Ioannidis, Soji Adeshina, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2506.08074)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds large language models in external evidence, yet it still falters when answers must be pieced together across semantically distant documents. We close this gap with the Hierarchical Lexical Graph (HLG), a three-tier index that (i) traces every atomic proposition to its source, (ii) clusters propositions into latent topics, and (iii) links entities and relations to expose cross-document paths. On top of HLG we build two complementary, plug-and-play retrievers: StatementGraphRAG, which performs fine-grained entity-aware beam search over propositions for high-precision factoid questions, and TopicGraphRAG, which selects coarse topics before expanding along entity links to supply broad yet relevant context for exploratory queries. Additionally, existing benchmarks lack the complexity required to rigorously evaluate multi-hop summarization systems, often focusing on single-document queries or limited datasets. To address this, we introduce a synthetic dataset generation pipeline that curates realistic, multi-document question-answer pairs, enabling robust evaluation of multi-hop retrieval systems. Extensive experiments across five datasets demonstrate that our methods outperform naive chunk-based RAG achieving an average relative improvement of 23.1% in retrieval recall and correctness. Open-source Python library is available at this https URL. 

**Abstract (ZH)**: Hierarchical Lexical Graph增强的检索生成（HLG-RAG）：面向多跳检索的多层次词图索引方法 

---
# Domain Switching on the Pareto Front: Multi-Objective Deep Kernel Learning in Automated Piezoresponse Force Microscopy 

**Title (ZH)**: 域切换在帕累托前沿上的应用：自动压电力显微镜中的多目标深度核学习 

**Authors**: Yu Liu, Utkarsh Pratiush, Kamyar Barakati, Hiroshi Funakubo, Ching-Che Lin, Jaegyu Kim, Lane W. Martin, Sergei V. Kalinin  

**Link**: [PDF](https://arxiv.org/pdf/2506.08073)  

**Abstract**: Ferroelectric polarization switching underpins the functional performance of a wide range of materials and devices, yet its dependence on complex local microstructural features renders systematic exploration by manual or grid-based spectroscopic measurements impractical. Here, we introduce a multi-objective kernel-learning workflow that infers the microstructural rules governing switching behavior directly from high-resolution imaging data. Applied to automated piezoresponse force microscopy (PFM) experiments, our framework efficiently identifies the key relationships between domain-wall configurations and local switching kinetics, revealing how specific wall geometries and defect distributions modulate polarization reversal. Post-experiment analysis projects abstract reward functions, such as switching ease and domain symmetry, onto physically interpretable descriptors including domain configuration and proximity to boundaries. This enables not only high-throughput active learning, but also mechanistic insight into the microstructural control of switching phenomena. While demonstrated for ferroelectric domain switching, our approach provides a powerful, generalizable tool for navigating complex, non-differentiable design spaces, from structure-property correlations in molecular discovery to combinatorial optimization across diverse imaging modalities. 

**Abstract (ZH)**: 多目标核学习工作流从高分辨率成像数据中推断调控切换行为的微观结构规则 

---
# Info-Coevolution: An Efficient Framework for Data Model Coevolution 

**Title (ZH)**: 信息共演化：一种高效的数据模型共演化框架 

**Authors**: Ziheng Qin, Hailun Xu, Wei Chee Yew, Qi Jia, Yang Luo, Kanchan Sarkar, Danhui Guan, Kai Wang, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2506.08070)  

**Abstract**: Machine learning relies heavily on data, yet the continuous growth of real-world data poses challenges for efficient dataset construction and training. A fundamental yet unsolved question is: given our current model and data, does a new data (sample/batch) need annotation/learning? Conventional approaches retain all available data, leading to non-optimal data and training efficiency. Active learning aims to reduce data redundancy by selecting a subset of samples to annotate, while it increases pipeline complexity and introduces bias. In this work, we propose Info-Coevolution, a novel framework that efficiently enables models and data to coevolve through online selective annotation with no bias. Leveraging task-specific models (and open-source models), it selectively annotates and integrates online and web data to improve datasets efficiently. For real-world datasets like ImageNet-1K, Info-Coevolution reduces annotation and training costs by 32\% without performance loss. It is able to automatically give the saving ratio without tuning the ratio. It can further reduce the annotation ratio to 50\% with semi-supervised learning. We also explore retrieval-based dataset enhancement using unlabeled open-source data. Code is available at this https URL. 

**Abstract (ZH)**: 机器学习高度依赖数据，而现实世界数据的持续增长为高效的数据集构建和训练带来了挑战。一个基本但未解决的问题是：在当前模型和数据条件下，是否需要对新数据（样本/批）进行注解/学习？传统方法保留所有可用数据，导致非最优的数据和训练效率。主动学习通过选择部分样本进行注解来减少数据冗余，但增加了管道复杂性并引入了偏差。在本工作中，我们提出了一种新颖的Info-Coevolution框架，通过在线选择性注解使模型和数据有效协同进化，且无偏差。利用任务特定模型（以及开源模型），它有选择地注解和整合在线和网络数据，以提高数据集的效率。对于如ImageNet-1K等真实世界数据集，Info-Coevolution在不损失性能的情况下，将注解和训练成本降低32%。它能够自动提供节约比例，无需调整比例。通过半监督学习，进一步将注解比例降低至50%。我们还探讨了使用未标注的开源数据进行检索增强的数据集增强方法。代码可通过以下链接获得。 

---
# WWAggr: A Window Wasserstein-based Aggregation for Ensemble Change Point Detection 

**Title (ZH)**: WWAggr: 基于窗口 Wasserstein 距离的集成变化点检测聚合方法 

**Authors**: Alexander Stepikin, Evgenia Romanenkova, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2506.08066)  

**Abstract**: Change Point Detection (CPD) aims to identify moments of abrupt distribution shifts in data streams. Real-world high-dimensional CPD remains challenging due to data pattern complexity and violation of common assumptions. Resorting to standalone deep neural networks, the current state-of-the-art detectors have yet to achieve perfect quality. Concurrently, ensembling provides more robust solutions, boosting the performance. In this paper, we investigate ensembles of deep change point detectors and realize that standard prediction aggregation techniques, e.g., averaging, are suboptimal and fail to account for problem peculiarities. Alternatively, we introduce WWAggr -- a novel task-specific method of ensemble aggregation based on the Wasserstein distance. Our procedure is versatile, working effectively with various ensembles of deep CPD models. Moreover, unlike existing solutions, we practically lift a long-standing problem of the decision threshold selection for CPD. 

**Abstract (ZH)**: 基于 Wasserstein 距离的深度变更点检测集成方法 

---
# FairDICE: Fairness-Driven Offline Multi-Objective Reinforcement Learning 

**Title (ZH)**: FairDICE：公平驱动的离线多目标强化学习 

**Authors**: Woosung Kim, Jinho Lee, Jongmin Lee, Byung-Jun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.08062)  

**Abstract**: Multi-objective reinforcement learning (MORL) aims to optimize policies in the presence of conflicting objectives, where linear scalarization is commonly used to reduce vector-valued returns into scalar signals. While effective for certain preferences, this approach cannot capture fairness-oriented goals such as Nash social welfare or max-min fairness, which require nonlinear and non-additive trade-offs. Although several online algorithms have been proposed for specific fairness objectives, a unified approach for optimizing nonlinear welfare criteria in the offline setting-where learning must proceed from a fixed dataset-remains unexplored. In this work, we present FairDICE, the first offline MORL framework that directly optimizes nonlinear welfare objective. FairDICE leverages distribution correction estimation to jointly account for welfare maximization and distributional regularization, enabling stable and sample-efficient learning without requiring explicit preference weights or exhaustive weight search. Across multiple offline benchmarks, FairDICE demonstrates strong fairness-aware performance compared to existing baselines. 

**Abstract (ZH)**: 多目标强化学习中的公平性导向优化：FairDICE 

---
# Eliciting Fine-Tuned Transformer Capabilities via Inference-Time Techniques 

**Title (ZH)**: 通过推理时技术激发细调变压器的能力 

**Authors**: Asankhaya Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2506.08060)  

**Abstract**: Large language models have transformed natural language processing, yet supervised fine-tuning (SFT) remains computationally intensive. This paper formally proves that capabilities acquired through SFT can be approximated by a base transformer model using inference-time techniques, specifically in-context learning (ICL), without altering model parameters, under idealized assumptions including unbounded computational resources and access to the fine-tuning dataset. We extend these results to practical scenarios with finite context lengths and partial dataset access. For text generation tasks with fixed output length $l$, datasets of size $\mathrm{O}\left( \frac{m V}{\varepsilon^2} \log \frac{m}{\delta} \right)$ or, with bounded context, $\mathrm{O}\left( \frac{l \log V}{\varepsilon^2} \log \frac{1}{\delta} \right)$ suffice to approximate fine-tuned behavior across $m$ contexts within error $\varepsilon$, where $V$ is the vocabulary size and $\delta$ is the failure probability. For linear classification, datasets of size $\mathrm{O}\left( \frac{d}{\varepsilon} \right)$ or, with fixed context, $\mathrm{O}\left( \frac{1}{\varepsilon^2} \log \frac{1}{\delta} \right)$ are sufficient, where $d$ is the input dimension. Grounded in the Turing completeness of transformers, these results provide a theoretical foundation for resource-efficient deployment of large language models, with practical techniques like retrieval-augmented generation bridging theory to real-world applications. 

**Abstract (ZH)**: 大型语言模型已transformed自然语言处理，然而监督微调（SFT）仍然计算密集。本文在包括无界计算资源和访问微调数据集的理想假设下，正式证明通过SFT获得的能力可以用基础变压器模型在推理时的技术，特别是上下文学习（ICL），在不改变模型参数的情况下进行近似。我们将这些结果扩展到具有有限上下文字长和部分数据集访问的实际场景。对于固定输出长度$l$的文本生成任务，大小为$\mathrm{O}\left( \frac{m V}{\varepsilon^2} \log \frac{m}{\delta} \right)$或在有界上下文情况下为$\mathrm{O}\left( \frac{l \log V}{\varepsilon^2} \log \frac{1}{\delta} \right)$的数据集足以在$m$个上下文中以误差$\varepsilon$近似微调行为，其中$V$是词汇量，$\delta$是失败概率。对于线性分类任务，大小为$\mathrm{O}\left( \frac{d}{\varepsilon} \right)$或固定上下文中为$\mathrm{O}\left( \frac{1}{\varepsilon^2} \log \frac{1}{\delta} \right)$的数据集足以在误差$\varepsilon$内进行近似，其中$d$是输入维度。基于变压器的图灵完全性，这些结果为大型语言模型的高效部署提供了理论基础，实际技术如检索增强生成将理论与实际应用相连接。 

---
# CaliciBoost: Performance-Driven Evaluation of Molecular Representations for Caco-2 Permeability Prediction 

**Title (ZH)**: CaliciBoost：基于性能的分子表示法用于Caco-2通透性预测评价 

**Authors**: Huong Van Le, Weibin Ren, Junhong Kim, Yukyung Yun, Young Bin Park, Young Jun Kim, Bok Kyung Han, Inho Choi, Jong IL Park, Hwi-Yeol Yun, Jae-Mun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08059)  

**Abstract**: Caco-2 permeability serves as a critical in vitro indicator for predicting the oral absorption of drug candidates during early-stage drug discovery. To enhance the accuracy and efficiency of computational predictions, we systematically investigated the impact of eight molecular feature representation types including 2D/3D descriptors, structural fingerprints, and deep learning-based embeddings combined with automated machine learning techniques to predict Caco-2 permeability. Using two datasets of differing scale and diversity (TDC benchmark and curated OCHEM data), we assessed model performance across representations and identified PaDEL, Mordred, and RDKit descriptors as particularly effective for Caco-2 prediction. Notably, the AutoML-based model CaliciBoost achieved the best MAE performance. Furthermore, for both PaDEL and Mordred representations, the incorporation of 3D descriptors resulted in a 15.73% reduction in MAE compared to using 2D features alone, as confirmed by feature importance analysis. These findings highlight the effectiveness of AutoML approaches in ADMET modeling and offer practical guidance for feature selection in data-limited prediction tasks. 

**Abstract (ZH)**: Caco-2通透性作为预测药物候选物口服吸收的关键体外指标，在早期药物发现阶段起到重要作用。为了提高计算预测的准确性和效率，我们系统研究了包括2D/3D描述符、结构指纹和基于深度学习的嵌入式表示在内的八种分子特征表示类型，并结合自动化机器学习技术以预测Caco-2通透性。通过使用不同规模和多样性的两个数据集（TDC基准和OCHEM整理数据），我们评估了不同表示方法的模型性能，并发现PaDEL、Mordred和RDKit描述符特别适用于Caco-2预测。特别地，基于自动化机器学习的CaliciBoost模型在MAE性能上表现最佳。此外，对于PaDEL和Mordred表示方法，将3D描述符纳入其中比仅使用2D特征可使MAE降低15.73%，这一结论得到了特征重要性分析的验证。这些发现突显了自动化机器学习方法在ADMET建模中的有效性，并为基于数据有限的预测任务提供了实用的特征选择指导。 

---
# STAMImputer: Spatio-Temporal Attention MoE for Traffic Data Imputation 

**Title (ZH)**: STAMImputer: 基于时空注意力混合模型的交通数据插补 

**Authors**: Yiming Wang, Hao Peng, Senzhang Wang, Haohua Du, Chunyang Liu, Jia Wu, Guanlin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08054)  

**Abstract**: Traffic data imputation is fundamentally important to support various applications in intelligent transportation systems such as traffic flow prediction. However, existing time-to-space sequential methods often fail to effectively extract features in block-wise missing data scenarios. Meanwhile, the static graph structure for spatial feature propagation significantly constrains the models flexibility in handling the distribution shift issue for the nonstationary traffic data. To address these issues, this paper proposes a SpatioTemporal Attention Mixture of experts network named STAMImputer for traffic data imputation. Specifically, we introduce a Mixture of Experts (MoE) framework to capture latent spatio-temporal features and their influence weights, effectively imputing block missing. A novel Low-rank guided Sampling Graph ATtention (LrSGAT) mechanism is designed to dynamically balance the local and global correlations across road networks. The sampled attention vectors are utilized to generate dynamic graphs that capture real-time spatial correlations. Extensive experiments are conducted on four traffic datasets for evaluation. The result shows STAMImputer achieves significantly performance improvement compared with existing SOTA approaches. Our codes are available at this https URL. 

**Abstract (ZH)**: 时空注意力混合专家网络在交通数据插补中的应用 

---
# Physics-Informed Teleconnection-Aware Transformer for Global Subseasonal-to-Seasonal Forecasting 

**Title (ZH)**: 基于物理信息的遥相关感知变换器在全局次季节至季节预测中的应用 

**Authors**: Tengfei Lyu, Weijia Zhang, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08049)  

**Abstract**: Subseasonal-to-seasonal (S2S) forecasting, which predicts climate conditions from several weeks to months in advance, presents significant challenges due to the chaotic dynamics of atmospheric systems and complex interactions across multiple scales. Current approaches often fail to explicitly model underlying physical processes and teleconnections that are crucial at S2S timescales. We introduce TelePiT, a novel deep learning architecture that enhances global S2S forecasting through integrated multi-scale physics and teleconnection awareness. Our approach consists of three key components: (1) Spherical Harmonic Embedding, which accurately encodes global atmospheric variables onto spherical geometry; (2) Multi-Scale Physics-Informed Neural ODE, which explicitly captures atmospheric physical processes across multiple learnable frequency bands; (3) Teleconnection-Aware Transformer, which models critical global climate interactions through tactfully injecting teleconnection patterns into the self-attention. Extensive experiments demonstrate that TelePiT significantly outperforms state-of-the-art data-driven baselines and operational numerical weather prediction systems, with remarkable improvements for atmospheric variables including a 57.7% reduction in RMSE for 2-meter temperature compared to previous best models. 

**Abstract (ZH)**: 季节到季节（S2S）预测：一种通过集成多尺度物理和遥相关意识增强的新型深度学习架构 

---
# Towards Reliable AR-Guided Surgical Navigation: Interactive Deformation Modeling with Data-Driven Biomechanics and Prompts 

**Title (ZH)**: 基于数据驱动生物力学和提示的交互变形建模：迈向可靠的AR引导外科导航 

**Authors**: Zheng Han, Jun Zhou, Jialun Pei, Jing Qin, Yingfang Fan, Qi Dou  

**Link**: [PDF](https://arxiv.org/pdf/2506.08048)  

**Abstract**: In augmented reality (AR)-guided surgical navigation, preoperative organ models are superimposed onto the patient's intraoperative anatomy to visualize critical structures such as vessels and tumors. Accurate deformation modeling is essential to maintain the reliability of AR overlays by ensuring alignment between preoperative models and the dynamically changing anatomy. Although the finite element method (FEM) offers physically plausible modeling, its high computational cost limits intraoperative applicability. Moreover, existing algorithms often fail to handle large anatomical changes, such as those induced by pneumoperitoneum or ligament dissection, leading to inaccurate anatomical correspondences and compromised AR guidance. To address these challenges, we propose a data-driven biomechanics algorithm that preserves FEM-level accuracy while improving computational efficiency. In addition, we introduce a novel human-in-the-loop mechanism into the deformation modeling process. This enables surgeons to interactively provide prompts to correct anatomical misalignments, thereby incorporating clinical expertise and allowing the model to adapt dynamically to complex surgical scenarios. Experiments on a publicly available dataset demonstrate that our algorithm achieves a mean target registration error of 3.42 mm. Incorporating surgeon prompts through the interactive framework further reduces the error to 2.78 mm, surpassing state-of-the-art methods in volumetric accuracy. These results highlight the ability of our framework to deliver efficient and accurate deformation modeling while enhancing surgeon-algorithm collaboration, paving the way for safer and more reliable computer-assisted surgeries. 

**Abstract (ZH)**: 基于增强现实（AR）引导的手术导航中的预手术器官模型在患者术中解剖结构上叠加，以可视化血管和肿瘤等关键结构。准确的形变建模对于通过确保预手术模型与动态变化的解剖结构之间的对齐来维持AR叠加的可靠性至关重要。尽管有限元方法（FEM）提供物理上合理的建模，但其高昂的计算成本限制了其在术中的应用。此外，现有算法往往无法处理大范围的解剖变化，如腹腔镜引起的气腹变化或韧带剥离引起的解剖变化，导致解剖对应不准确并且削弱了AR导航。为解决这些挑战，我们提出了一种数据驱动的生物力学算法，该算法在保持FEM级别的准确性的基础上提高了计算效率。此外，我们引入了一种新型的人在环机制到形变建模过程中。这使得外科医生能够互动地提供提示以纠正解剖对齐错误，从而结合临床专业知识并使模型能够动态适应复杂的手术场景。在公共数据集上的实验显示，我们的算法实现了平均目标注册误差为3.42毫米。通过交互框架整合外科医生的提示进一步将误差减少到2.78毫米，超越了最新方法在体素精度方面的表现。这些结果突显了我们框架在实现高效准确的形变建模的同时增强外科医生与算法协作的能力，为更加安全可靠的计算机辅助手术铺平了道路。 

---
# Evaluation of Machine Learning Models in Student Academic Performance Prediction 

**Title (ZH)**: 机器学习模型在学生学业成绩预测中的评估 

**Authors**: A.G.R. Sandeepa, Sanka Mohottala  

**Link**: [PDF](https://arxiv.org/pdf/2506.08047)  

**Abstract**: This research investigates the use of machine learning methods to forecast students' academic performance in a school setting. Students' data with behavioral, academic, and demographic details were used in implementations with standard classical machine learning models including multi-layer perceptron classifier (MLPC). MLPC obtained 86.46% maximum accuracy for test set across all implementations. Under 10-fold cross validation, MLPC obtained 79.58% average accuracy for test set while for train set, it was 99.65%. MLP's better performance over other machine learning models strongly suggest the potential use of neural networks as data-efficient models. Feature selection approach played a crucial role in improving the performance and multiple evaluation approaches were used in order to compare with existing literature. Explainable machine learning methods were utilized to demystify the black box models and to validate the feature selection approach. 

**Abstract (ZH)**: 本研究调查了使用机器学习方法预测学校中学生学业成绩的应用。使用包含行为、学术和人口统计学细节的学生数据实施了标准的经典机器学习模型，包括多层感知机分类器（MLPC）。在所有实施中，MLPC在测试集上的最高准确率为86.46%，在10折交叉验证中，MLPC在测试集上的平均准确率为79.58%，而在训练集上的准确率为99.65%。MLP相比其他机器学习模型的更好性能强烈表明，神经网络作为数据高效模型具有潜在应用价值。特征选择方法在提高性能方面起着关键作用，并使用了多种评估方法以与现有文献进行比较。可解释的机器学习方法被利用来揭示黑盒模型，并验证特征选择方法。 

---
# UAVs Meet Agentic AI: A Multidomain Survey of Autonomous Aerial Intelligence and Agentic UAVs 

**Title (ZH)**: UAVs遇上了能动AI：自主航空智能与能动无人机的多领域综述 

**Authors**: Ranjan Sapkota, Konstantinos I. Roumeliotis, Manoj Karkee  

**Link**: [PDF](https://arxiv.org/pdf/2506.08045)  

**Abstract**: Agentic UAVs represent a new frontier in autonomous aerial intelligence, integrating perception, decision-making, memory, and collaborative planning to operate adaptively in complex, real-world environments. Driven by recent advances in Agentic AI, these systems surpass traditional UAVs by exhibiting goal-driven behavior, contextual reasoning, and interactive autonomy. We provide a comprehensive foundation for understanding the architectural components and enabling technologies that distinguish Agentic UAVs from traditional autonomous UAVs. Furthermore, a detailed comparative analysis highlights advancements in autonomy with AI agents, learning, and mission flexibility. This study explores seven high-impact application domains precision agriculture, construction & mining, disaster response, environmental monitoring, infrastructure inspection, logistics, security, and wildlife conservation, illustrating the broad societal value of agentic aerial intelligence. Furthermore, we identify key challenges in technical constraints, regulatory limitations, and data-model reliability, and we present emerging solutions across hardware innovation, learning architectures, and human-AI interaction. Finally, a future roadmap is proposed, outlining pathways toward self-evolving aerial ecosystems, system-level collaboration, and sustainable, equitable deployments. This survey establishes a foundational framework for the future development, deployment, and governance of agentic aerial systems (Agentic UAVs) across diverse societal and industrial domains. 

**Abstract (ZH)**: 自主无人机代表了自主空中智能的新前沿，融合了感知、决策、记忆和协作规划能力，能够在复杂的真实世界环境中灵活操作。随着自主人工智能的 Recent 进展，这些系统通过表现出目标驱动的行为、情境推理和交互式自主超越了传统的无人机。我们提供了理解自主无人机与传统自主无人机差异性的综合架构和使能技术基础。此外，详细的比较分析突显了自主性、学习与任务灵活性的进展。本研究探讨了七个高影响应用领域——精准农业、建筑业与采矿业、灾害响应、环境监测、基础设施检查、物流、安全以及野生动物保护——展示了自主空中智能的广泛社会价值。此外，我们确定了技术约束、监管限制和数据-模型可靠性等关键挑战，并提出了硬件创新、学习架构和人-机交互领域的新兴解决方案。最后，我们提出了未来的路线图，概述了自演化空中生态系统、系统级协作以及可持续、公平部署的路径。这篇综述为自主空中系统（自主无人机）在不同社会和工业领域的未来开发、部署和治理奠定了基础框架。 

---
# The World of AI: A Novel Approach to AI Literacy for First-year Engineering Students 

**Title (ZH)**: 人工智能的世界：一种面向大一工程学生的新型人工智能素养教学方法 

**Authors**: Siddharth Siddharth, Brainerd Prince, Amol Harsh, Shreyas Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2506.08041)  

**Abstract**: This work presents a novel course titled The World of AI designed for first-year undergraduate engineering students with little to no prior exposure to AI. The central problem addressed by this course is that engineering students often lack foundational knowledge of AI and its broader societal implications at the outset of their academic journeys. We believe the way to address this gap is to design and deliver an interdisciplinary course that can a) be accessed by first-year undergraduate engineering students across any domain, b) enable them to understand the basic workings of AI systems sans mathematics, and c) make them appreciate AI's far-reaching implications on our lives. The course was divided into three modules co-delivered by faculty from both engineering and humanities. The planetary module explored AI's dual role as both a catalyst for sustainability and a contributor to environmental challenges. The societal impact module focused on AI biases and concerns around privacy and fairness. Lastly, the workplace module highlighted AI-driven job displacement, emphasizing the importance of adaptation. The novelty of this course lies in its interdisciplinary curriculum design and pedagogical approach, which combines technical instruction with societal discourse. Results revealed that students' comprehension of AI challenges improved across diverse metrics like (a) increased awareness of AI's environmental impact, and (b) efficient corrective solutions for AI fairness. Furthermore, it also indicated the evolution in students' perception of AI's transformative impact on our lives. 

**Abstract (ZH)**: 适用于缺乏人工智能背景的理工科大一学生的《人工智能世界》课程创新设计及其影响 

---
# Inverse Design in Distributed Circuits Using Single-Step Reinforcement Learning 

**Title (ZH)**: 使用单步强化学习在分布式电路中进行逆向设计 

**Authors**: Jiayu Li, Masood Mortazavi, Ning Yan, Yihong Ma, Reza Zafarani  

**Link**: [PDF](https://arxiv.org/pdf/2506.08029)  

**Abstract**: The goal of inverse design in distributed circuits is to generate near-optimal designs that meet a desirable transfer function specification. Existing design exploration methods use some combination of strategies involving artificial grids, differentiable evaluation procedures, and specific template topologies. However, real-world design practices often require non-differentiable evaluation procedures, varying topologies, and near-continuous placement spaces. In this paper, we propose DCIDA, a design exploration framework that learns a near-optimal design sampling policy for a target transfer function. DCIDA decides all design factors in a compound single-step action by sampling from a set of jointly-trained conditional distributions generated by the policy. Utilizing an injective interdependent ``map", DCIDA transforms raw sampled design ``actions" into uniquely equivalent physical representations, enabling the framework to learn the conditional dependencies among joint ``raw'' design decisions. Our experiments demonstrate DCIDA's Transformer-based policy network achieves significant reductions in design error compared to state-of-the-art approaches, with significantly better fit in cases involving more complex transfer functions. 

**Abstract (ZH)**: 分布式电路中逆向设计的目标是生成接近最优的设计，以满足期望的传输函数规范。现有的设计探索方法结合使用了涉及人工网格、可微评估程序以及特定模板拓扑的策略。然而，实际的设计实践往往需要非可微评估程序、可变拓扑结构以及接近连续的放置空间。本文提出了一种DCIDA设计探索框架，该框架学习目标传输函数的近最优设计采样策略。DCIDA通过从由策略联合训练生成的条件分布集中采样，一次性决定所有设计因素。利用一个注入性相互依赖的“映射”，DCIDA将原始采样设计“动作”转换为唯一等价的物理表示，从而使框架能够学习联合“原始”设计决策之间的条件依赖关系。我们的实验结果表明，DCIDA基于Transformer的策略网络在设计误差方面比最先进的方法实现了显著减少，尤其是在涉及更复杂传输函数的情况下，匹配效果明显更好。 

---
# Recipes for Pre-training LLMs with MXFP8 

**Title (ZH)**: 使用MXFP8预训练大规模语言模型的方法 

**Authors**: Asit Mishra, Dusan Stosic, Simon Layton  

**Link**: [PDF](https://arxiv.org/pdf/2506.08027)  

**Abstract**: Precision scaling - using fewer bits to represent model parameters and related tensors during pre-training - has emerged as a compelling technique for improving GPU efficiency without sacrificing accuracy. Microscaling (MX) formats in NVIDIA's latest Blackwell GPUs represent a major leap in enabling this precision scaling aspect. These formats combine narrow floating-point data types with per-block scaling factors, offering a fine-grained approach to quantizing tensors.
Although MX-formats offer the promise of improved numeric stability compared to other reduced-precision representations, in practice they must be used carefully in order to successfully converge an LLM on a multi-trillion token dataset. In this paper, we show that the rounding mode suggested in OCP specification can lead to divergence when pre-training an LLM. We show an improved rounding mode, which uses round-to-infinity to compute scaling factors, enables successful pre-training in MXFP8 for an 8B model on 15T tokens. 

**Abstract (ZH)**: 精度缩放——在预训练过程中使用较少位数来表示模型参数及相关张量——已成为一种无需牺牲准确性的方法来提高GPU效率的技术。NVIDIA最新Blackwell GPU上的Microscaling (MX) 格式代表了实现这一精度缩放方面的重大突破。这些格式结合了窄浮点数据类型和每块的缩放因子，提供了对张量进行量化的一种精细方法。尽管MX格式与其它低精度表示相比提供了更好的数值稳定性，但在实际应用中，为了在大规模的万亿级标记数据集上成功预训练LLM，必须谨慎使用。在本文中，我们展示了OCP规范中建议的舍入模式可能导致在预训练LLM时发散的情况。我们展示了一种改进的舍入模式，使用向无穷大舍入来计算缩放因子，使得在MXFP8格式下成功预训练一个8B模型并处理15T标记数据成为可能。 

---
# Aligning Proteins and Language: A Foundation Model for Protein Retrieval 

**Title (ZH)**: 蛋白质与语言对齐：一种蛋白质检索的基础模型 

**Authors**: Qifeng Wu, Zhengzhe Liu, Han Zhu, Yizhou Zhao, Daisuke Kihara, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08023)  

**Abstract**: This paper aims to retrieve proteins with similar structures and semantics from large-scale protein dataset, facilitating the functional interpretation of protein structures derived by structural determination methods like cryo-Electron Microscopy (cryo-EM). Motivated by the recent progress of vision-language models (VLMs), we propose a CLIP-style framework for aligning 3D protein structures with functional annotations using contrastive learning. For model training, we propose a large-scale dataset of approximately 200,000 protein-caption pairs with rich functional descriptors. We evaluate our model in both in-domain and more challenging cross-database retrieval on Protein Data Bank (PDB) and Electron Microscopy Data Bank (EMDB) dataset, respectively. In both cases, our approach demonstrates promising zero-shot retrieval performance, highlighting the potential of multimodal foundation models for structure-function understanding in protein biology. 

**Abstract (ZH)**: 本文旨在从大规模蛋白数据集中检索具有相似结构和语义的蛋白质，以便于通过如冷冻电子显微镜（cryo-EM）等结构测定方法得到的蛋白结构的功能解读。受近期视觉语言模型（VLMs）进展的启发，我们提出了一种CLIP风格的框架，通过对比学习将3D蛋白结构与功能注释进行对齐。在模型训练中，我们提出了一种包含约200,000个蛋白-描述词对的大规模数据集，其中包含丰富的功能描述。我们在蛋白质数据银行（PDB）和电子显微镜数据银行（EMDB）数据集上分别进行了同域和更具挑战性的跨数据库检索评估。在两种情况下，我们的方法都展示了有前途的零样本检索性能，突显了多模态基础模型在蛋白生物学结构-功能理解方面的潜力。 

---
# Modality-Balancing Preference Optimization of Large Multimodal Models by Adversarial Negative Mining 

**Title (ZH)**: 大型多模态模型的模态平衡偏好优化通过对抗负挖掘 

**Authors**: Chenxi Liu, Tianyi Xiong, Ruibo Chen, Yihan Wu, Junfeng Guo, Tianyi Zhou, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08022)  

**Abstract**: The task adaptation and alignment of Large Multimodal Models (LMMs) have been significantly advanced by instruction tuning and further strengthened by recent preference optimization. Yet, most LMMs still suffer from severe modality imbalance during reasoning, i.e., outweighing language prior biases over visual inputs, which bottlenecks their generalization to downstream tasks and causes hallucinations. However, existing preference optimization approaches for LMMs do not focus on restraining the internal biases of their Large Language Model (LLM) backbones when curating the training data. Moreover, they heavily rely on offline data and lack the capacity to explore diverse responses adaptive to dynamic distributional shifts during training. Meanwhile, Group Relative Policy Optimization (GRPO), a recent method using online-generated data and verified rewards to improve reasoning capabilities, remains largely underexplored in LMM alignment. In this paper, we propose a novel preference learning framework, Modality-Balancing Preference Optimization (MBPO), to address the modality imbalance in LMMs. MBPO constructs a more effective offline preference dataset by generating hard negatives, i.e., rejected responses misled by LLM biases due to limited usage of visual information, through adversarial perturbation of input images. Moreover, MBPO leverages the easy-to-verify nature of close-ended tasks to generate online responses with verified rewards. GRPO is then employed to train the model with offline-online hybrid data. Extensive experiments demonstrate that MBPO can enhance LMM performance on challenging vision-language tasks and effectively reduce hallucinations. 

**Abstract (ZH)**: 大型多模态模型的任务适配与对齐通过指令调优得到了显著进展，并通过最近的偏好优化得到了进一步强化。然而，大多数大型多模态模型在推理过程中仍然面临严重的模态不平衡问题，即语言先验偏差过度支配视觉输入，这限制了其对下游任务的泛化能力并导致幻觉。现有的针对大型多模态模型的偏好优化方法在编排训练数据时并未关注抑制其大型语言模型（LLM）核心模块内的内部偏差。此外，它们高度依赖离线数据，并缺乏在训练过程中探索适应动态分布转移的多样化响应的能力。同时，使用在线生成数据和验证奖励提高推理能力的组相对策略优化（GRPO）方法在大型多模态模型对齐中的应用仍处于初步阶段。本文提出了一种新颖的偏好学习框架——模态平衡偏好优化（MBPO），以解决大型多模态模型中的模态不平衡问题。MBPO通过生成对抗性扰动输入图像以限制LLM偏差来误导的难负样本（即硬负样本），构建更有效的离线偏好数据集。此外，MBPO利用封闭任务易于验证的性质生成在线响应和验证奖励。然后使用离线-在线混合数据训练模型。广泛实验证明，MBPO能够提高大型多模态模型在视觉-语言任务上的性能并有效减少幻觉。 

---
# Bi-level Unbalanced Optimal Transport for Partial Domain Adaptation 

**Title (ZH)**: 两层不均衡最优运输在部分领域适应中的应用 

**Authors**: Zi-Ying Chen, Chuan-Xian Ren, Hong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08020)  

**Abstract**: Partial domain adaptation (PDA) problem requires aligning cross-domain samples while distinguishing the outlier classes for accurate knowledge transfer. The widely used weighting framework tries to address the outlier classes by introducing the reweighed source domain with a similar label distribution to the target domain. However, the empirical modeling of weights can only characterize the sample-wise relations, which leads to insufficient exploration of cluster structures, and the weights could be sensitive to the inaccurate prediction and cause confusion on the outlier classes. To tackle these issues, we propose a Bi-level Unbalanced Optimal Transport (BUOT) model to simultaneously characterize the sample-wise and class-wise relations in a unified transport framework. Specifically, a cooperation mechanism between sample-level and class-level transport is introduced, where the sample-level transport provides essential structure information for the class-level knowledge transfer, while the class-level transport supplies discriminative information for the outlier identification. The bi-level transport plan provides guidance for the alignment process. By incorporating the label-aware transport cost, the local transport structure is ensured and a fast computation formulation is derived to improve the efficiency. Extensive experiments on benchmark datasets validate the competitiveness of BUOT. 

**Abstract (ZH)**: 部分领域适应（PDA）问题要求在对齐跨领域样本的同时区分异常类别以实现准确的知识迁移。广泛使用的加权框架通过引入与目标域标签分布相似的重加权源域来处理异常类别，然而，经验性的建模方式只能表征样本间的关联关系，导致对聚类结构的探索不足，并且权重可能会受到不准确预测的影响而对异常类别造成混淆。为解决这些问题，我们提出了一种双层不平衡最优传输（BUOT）模型，以同时在一个统一的传输框架中表征样本级和类级关系。具体而言，在样本级传输与类级传输之间引入了一种合作机制，其中样本级传输为类级知识迁移提供必要的结构信息，而类级传输为异常类别识别提供区分性信息。双层传输计划为对齐过程提供了指导。通过引入标签感知传输代价，确保了局部传输结构，并推导出高效的计算公式以提高计算效率。在基准数据集上的广泛实验验证了BUOT的竞争性。 

---
# KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache 

**Title (ZH)**: KVmix：基于梯度的键值缓存层重要性感知混合精度量化 

**Authors**: Fei Li, Song Liu, Weiguo Wu, Shiqiang Nie, Jinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08018)  

**Abstract**: The high memory demands of the Key-Value (KV) Cache during the inference of Large Language Models (LLMs) severely restrict their deployment in resource-constrained platforms. Quantization can effectively alleviate the memory pressure caused by KV Cache. However, existing methods either rely on static one-size-fits-all precision allocation or fail to dynamically prioritize critical KV in long-context tasks, forcing memory-accuracy-throughput tradeoffs. In this work, we propose a novel mixed-precision quantization method for KV Cache named KVmix. KVmix leverages gradient-based importance analysis to evaluate how individual Key and Value projection matrices affect the model loss, enabling layer-specific bit-width allocation for mix-precision quantization. It dynamically prioritizes higher precision for important layers while aggressively quantizing less influential ones, achieving a tunable balance between accuracy and efficiency. KVmix also introduces a dynamic long-context optimization strategy that adaptively keeps full-precision KV pairs for recent pivotal tokens and compresses older ones, achieving high-quality sequence generation with low memory usage. Additionally, KVmix provides efficient low-bit quantization and CUDA kernels to optimize computational overhead. On LLMs such as Llama and Mistral, KVmix achieves near-lossless inference performance with extremely low quantization configuration (Key 2.19bit Value 2.38bit), while delivering a remarkable 4.9x memory compression and a 5.3x speedup in inference throughput. 

**Abstract (ZH)**: KVmix：一种用于大型语言模型推理中Key-Value缓存的新型混合精度量化方法 

---
# QUITE: A Query Rewrite System Beyond Rules with LLM Agents 

**Title (ZH)**: QUITE：超出规则的查询重写系统以LLM代理为基础 

**Authors**: Yuyang Song, Hanxu Yan, Jiale Lao, Yibo Wang, Yufei Li, Yuanchun Zhou, Jianguo Wang, Mingjie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07675)  

**Abstract**: Query rewrite transforms SQL queries into semantically equivalent forms that run more efficiently. Existing approaches mainly rely on predefined rewrite rules, but they handle a limited subset of queries and can cause performance regressions. This limitation stems from three challenges of rule-based query rewrite: (1) it is hard to discover and verify new rules, (2) fixed rewrite rules do not generalize to new query patterns, and (3) some rewrite techniques cannot be expressed as fixed rules. Motivated by the fact that human experts exhibit significantly better rewrite ability but suffer from scalability, and Large Language Models (LLMs) have demonstrated nearly human-level semantic and reasoning abilities, we propose a new approach of using LLMs to rewrite SQL queries beyond rules. Due to the hallucination problems in LLMs, directly applying LLMs often leads to nonequivalent and suboptimal queries. To address this issue, we propose QUITE (query rewrite), a training-free and feedback-aware system based on LLM agents that rewrites SQL queries into semantically equivalent forms with significantly better performance, covering a broader range of query patterns and rewrite strategies compared to rule-based methods. Firstly, we design a multi-agent framework controlled by a finite state machine (FSM) to equip LLMs with the ability to use external tools and enhance the rewrite process with real-time database feedback. Secondly, we develop a rewrite middleware to enhance the ability of LLMs to generate optimized query equivalents. Finally, we employ a novel hint injection technique to improve execution plans for rewritten queries. Extensive experiments show that QUITE reduces query execution time by up to 35.8% over state-of-the-art approaches and produces 24.1% more rewrites than prior methods, covering query cases that earlier systems did not handle. 

**Abstract (ZH)**: 基于大语言模型的SQL查询重构：超越规则的方法 

---
# ChemGraph: An Agentic Framework for Computational Chemistry Workflows 

**Title (ZH)**: ChemGraph: 一个自主的计算化学工作流框架 

**Authors**: Thang D. Pham, Aditya Tanikanti, Murat Keçeli  

**Link**: [PDF](https://arxiv.org/pdf/2506.06363)  

**Abstract**: Atomistic simulations are essential tools in chemistry and materials science, accelerating the discovery of novel catalysts, energy storage materials, and pharmaceuticals. However, running these simulations remains challenging due to the wide range of computational methods, diverse software ecosystems, and the need for expert knowledge and manual effort for the setup, execution, and validation stages. In this work, we present ChemGraph, an agentic framework powered by artificial intelligence and state-of-the-art simulation tools to streamline and automate computational chemistry and materials science workflows. ChemGraph leverages graph neural network-based foundation models for accurate yet computationally efficient calculations and large language models (LLMs) for natural language understanding, task planning, and scientific reasoning to provide an intuitive and interactive interface. Users can perform tasks such as molecular structure generation, single-point energy, geometry optimization, vibrational analysis, and thermochemistry calculations with methods ranging from tight-binding and machine learning interatomic potentials to density functional theory or wave function theory-based methods. We evaluate ChemGraph across 13 benchmark tasks and demonstrate that smaller LLMs (GPT-4o-mini, Claude-3.5-haiku, Qwen2.5-14B) perform well on simple workflows, while more complex tasks benefit from using larger models like GPT-4o. Importantly, we show that decomposing complex tasks into smaller subtasks through a multi-agent framework enables smaller LLM models to match or exceed GPT-4o's performance in specific scenarios. 

**Abstract (ZH)**: 基于人工智能的ChemGraph框架：加速计算化学与材料科学工作流 

---
# Being Strong Progressively! Enhancing Knowledge Distillation of Large Language Models through a Curriculum Learning Framework 

**Title (ZH)**: 逐步变强！通过课程学习框架增强大型语言模型的知识蒸馏 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05695)  

**Abstract**: Knowledge Distillation (KD) compresses large language models (LLMs) by transferring the teacher model's capabilities to a smaller student model, reducing inference cost and memory usage while maintaining performance. However, existing KD methods for LLMs often fail to prevent significant shifts in the student model's distribution during training, leading to issues such as catastrophic forgetting, mode collapse, and training-inference mismatch. To address these challenges, we propose a novel, plug-in curriculum learning framework inspired by the strength training principle of "progressive overload" (POCL), which can be seamlessly integrated into existing white-box KD approaches with minimal computational overhead. The framework comprises two core components: (1) a difficulty measurer that ranks and partitions training samples from easy to hard, and (2) a training scheduler that incrementally introduces these subsets into the distillation process at fixed intervals while applying loss functions with progressively rising temperatures. By starting with the easiest samples and progressively increasing the difficulty, the approach enhances both the stability and efficiency of learning. Extensive experiments in instruction-following settings demonstrate that POCL consistently improves the performance of distilled student models across various white-box KD methods and model families. Our findings highlight the effectiveness of sorted training samples in KD for LLMs. More generally, our work demonstrates how to structure training data within the KD process to enhance the stability and performance of distilled LLMs. 

**Abstract (ZH)**: 知识蒸馏（KD）通过将教师模型的能力转移到较小的学生模型中，压缩大型语言模型（LLMs），同时减少推理成本和内存使用，并保持性能。然而，现有的LLMs知识蒸馏方法往往无法防止学生模型分布训练过程中的显著变化，导致灾难性遗忘、模式坍缩和训练-推理不匹配等问题。为了解决这些挑战，我们提出了一种新型插件式课程学习框架，该框架灵感来源于“渐进超载”（POCL）的体力训练原则，并能与现有白盒知识蒸馏方法无缝集成，计算开销最小。该框架包含两个核心组件：（1）一个难度度量器，用于按难度对训练样本进行排序和分区；（2）一个训练调度器，在固定间隔逐步引入这些子集到蒸馏过程中，并应用逐渐升高的温度损失函数。通过从最简单的样本开始并逐步增加难度，该方法增强了学习的稳定性和效率。在指令跟随设置下的广泛实验表明，POCL方法能持续改善各种白盒知识蒸馏方法和模型家族中蒸馏学生模型的性能。我们的研究结果突显了排序训练样本在LLMs知识蒸馏中的有效性。更普遍地，我们的工作展示了如何在知识蒸馏过程中结构化训练数据以增强蒸馏LLMs的稳定性和性能。 

---
# Exp4Fuse: A Rank Fusion Framework for Enhanced Sparse Retrieval using Large Language Model-based Query Expansion 

**Title (ZH)**: Exp4Fuse: 一种基于大型语言模型的查询扩展增强稀疏检索的排序融合框架 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04760)  

**Abstract**: Large Language Models (LLMs) have shown potential in generating hypothetical documents for query expansion, thereby enhancing information retrieval performance. However, the efficacy of this method is highly dependent on the quality of the generated documents, which often requires complex prompt strategies and the integration of advanced dense retrieval techniques. This can be both costly and computationally intensive. To mitigate these limitations, we explore the use of zero-shot LLM-based query expansion to improve sparse retrieval, particularly for learned sparse retrievers. We introduce a novel fusion ranking framework, Exp4Fuse, which enhances the performance of sparse retrievers through an indirect application of zero-shot LLM-based query expansion. Exp4Fuse operates by simultaneously considering two retrieval routes-one based on the original query and the other on the LLM-augmented query. It then generates two ranked lists using a sparse retriever and fuses them using a modified reciprocal rank fusion method. We conduct extensive evaluations of Exp4Fuse against leading LLM-based query expansion methods and advanced retrieval techniques on three MS MARCO-related datasets and seven low-resource datasets. Experimental results reveal that Exp4Fuse not only surpasses existing LLM-based query expansion methods in enhancing sparse retrievers but also, when combined with advanced sparse retrievers, achieves SOTA results on several benchmarks. This highlights the superior performance and effectiveness of Exp4Fuse in improving query expansion for sparse retrieval. 

**Abstract (ZH)**: 大型语言模型在查询扩展中的潜在应用可以通过假设文档生成来提高信息检索性能，但其效果高度依赖于生成文档的质量，这通常需要复杂的提示策略和先进密集检索技术的集成，这可能是昂贵且计算密集型的。为缓解这些局限性，我们探索了零-shot LLM-based查询扩展在改进稀疏检索中的应用，特别是对于学习到的稀疏检索器。我们提出了一种新颖的融合排名框架Exp4Fuse，通过间接应用零-shot LLM-based查询扩展来增强稀疏检索器的性能。Exp4Fuse通过同时考虑两条检索路径——一条基于原始查询，另一条基于LLM增强后的查询——工作。然后使用稀疏检索器生成两个排名列表，并通过修改后的倒数排名融合方法将它们融合。我们在三个MS MARCO相关数据集和七个低资源数据集上对Exp4Fuse与领先的方法进行了广泛的评估和先进的检索技术。实验结果表明，Exp4Fuse不仅在增强稀疏检索器方面超过了现有的LLM-based查询扩展方法，而且与先进的稀疏检索器结合使用时，在多个基准上实现了SOTA结果。这突显了Exp4Fuse在提高稀疏检索查询扩展性能和效果方面的优越性。 

---
# Werewolf: A Straightforward Game Framework with TTS for Improved User Engagement 

**Title (ZH)**: werewolf: 一个配备TTS的简单游戏框架以提高用户参与度 

**Authors**: Qihui Fan, Enfu Nan, Wenbo Li, Lei Lu, Pu Zhao, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00160)  

**Abstract**: The growing popularity of social deduction game systems for both business applications and AI research has greatly benefited from the rapid advancements in Large Language Models (LLMs), which now demonstrate stronger reasoning and persuasion capabilities. Especially with the raise of DeepSeek R1 and V3 models, LLMs should enable a more engaging experience for human players in LLM-agent-based social deduction games like Werewolf. Previous works either fine-tuning, advanced prompting engineering, or additional experience pool to achieve engaging text-format Werewolf game experience. We propose a novel yet straightforward LLM-based Werewolf game system with tuned Text-to-Speech(TTS) models designed for enhanced compatibility with various LLM models, and improved user engagement. We argue with ever enhancing LLM reasoning, extra components will be unnecessary in the case of Werewolf. 

**Abstract (ZH)**: 社会推理游戏系统在商业应用和AI研究中的日益流行得益于大规模语言模型的快速进步，这些模型现在展示了更强的推理和说服能力。尤其是在DeepSeek R1和V3模型的推动下，大规模语言模型应在基于大规模语言模型-代理的社会推理游戏中，如狼人杀，为人类玩家提供更为吸引人的体验。我们提出了一种新颖且简洁的大规模语言模型驱动的狼人杀游戏系统，该系统配备了增强兼容性的文本到语音模型，以提高用户体验。我们认为，在大规模语言模型推理能力不断增强的情况下，狼人杀游戏中额外的组件将变得不必要的。 

---
